"""Resolve GeoPackage input rows across parcel, HUC, and domain scopes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import geopandas as gpd
import pandas as pd

from .input_distributions import statistic_columns
from .input_schema import INPUT_VARIABLE_SPECS, KNOWN_INPUT_TABLES
from .input_validation import (
    FRACTION_DOMAIN,
    NONNEGATIVE_DOMAIN,
    require_columns,
    validate_numeric_columns_in_domain,
    validate_stats_rows,
    validate_unique_rows,
)
from .io_utils import list_geopackage_tables, read_geodataframe
from .utils import ci_get, normalize_columns, normalize_pollutant_label


def normalize_huc_id(value: Any) -> str:
    """Normalize a hydrologic unit code without discarding leading zeroes.

    Parameters
    ----------
    value : Any
        HUC identifier stored as text. Leading and trailing whitespace is
        removed; numerical identifiers are not accepted because they can lose
        significant leading zeroes.

    Returns
    -------
    str
        ASCII numeric HUC identifier containing 2, 4, ..., or 16 digits.
        Its length indicates the HUC level.

    Raises
    ------
    ValueError
        If ``value`` is not a text HUC code of a supported length.
    """
    if not isinstance(value, str):
        raise ValueError(f"HUC identifiers must be stored as text, got {value!r}")
    code = value.strip()
    if len(code) not in range(2, 17, 2) or not code.isascii() or not code.isdigit():
        raise ValueError(f"Invalid HUC identifier {value!r}; expected 2, 4, ..., 16 digits")
    return code


def validate_huc_polygons(hucs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate and standardize a single-level set of HUC polygons.

    Parameters
    ----------
    hucs : geopandas.GeoDataFrame
        Nonempty polygon or multipolygon features with a declared CRS and a
        text ``huc_id`` column. An optional ``huc_level`` must equal the number
        of digits in each identifier.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy with normalized ``huc_id`` and integer ``huc_level`` columns.

    Raises
    ------
    ValueError
        If identifiers are missing, malformed, duplicated, or mixed in HUC
        level; if the CRS or geometry is invalid; or if polygons overlap by
        a positive area. Boundary sharing without area overlap is allowed.
    """
    hucs = hucs.copy()
    if hucs.empty or hucs.crs is None:
        raise ValueError("HUC polygons must be nonempty and declare a CRS")
    if "huc_id" not in hucs:
        raise ValueError("HUC polygons require a huc_id column")
    hucs["huc_id"] = hucs["huc_id"].map(normalize_huc_id)
    levels = hucs["huc_id"].str.len()
    if levels.nunique() != 1:
        raise ValueError("One HUC level is supported per package; mixed/nested levels are ambiguous")
    if "huc_level" in hucs and not pd.to_numeric(hucs["huc_level"], errors="raise").eq(levels).all():
        raise ValueError("huc_level must match the number of digits in huc_id")
    hucs["huc_level"] = levels
    validate_unique_rows(hucs, ["huc_id"], "HUC polygons")
    if (
        hucs.geometry.isna().any()
        or hucs.geometry.is_empty.any()
        or (~hucs.geometry.is_valid).any()
        or (~hucs.geometry.geom_type.isin(["Polygon", "MultiPolygon"])).any()
    ):
        raise ValueError("HUC polygons contain null, empty, invalid, or non-polygon geometry")
    left, right = hucs.sindex.query(hucs.geometry, predicate="intersects")
    for a, b in zip(left, right):
        if a < b and hucs.geometry.iloc[a].intersection(hucs.geometry.iloc[b]).area > 0:
            raise ValueError(
                f"HUC polygons overlap: {hucs.huc_id.iloc[a]} and {hucs.huc_id.iloc[b]}"
            )
    return hucs


def assign_parcels_to_hucs(
    parcels: gpd.GeoDataFrame, hucs: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Assign each parcel to its greatest-area-overlap HUC polygon.

    Parcel/HUC intersections are measured after projecting both datasets to
    the estimated UTM CRS of the parcels. Multiple intersection pieces for a
    parcel-HUC pair are summed, and an equal-area tie uses the lowest HUC ID.
    This is a categorical assignment: HUC parameter values are not blended.

    Parameters
    ----------
    parcels : geopandas.GeoDataFrame
        Modeled parcel polygons with unique, non-null ``pid`` values and a
        declared CRS.
    hucs : geopandas.GeoDataFrame
        HUC polygons accepted by :func:`validate_huc_polygons`.

    Returns
    -------
    pandas.DataFrame
        One row per parcel, with columns ``pid``, ``huc_id``, and
        ``area_fraction`` (intersection area divided by parcel area, clipped
        to [0, 1] against numerical roundoff).

    Raises
    ------
    ValueError
        If the geometries, CRS, or parcel identifiers are invalid, a parcel
        has zero area, or a parcel has no positive-area HUC intersection.
    """
    hucs = validate_huc_polygons(hucs)
    if parcels.empty or parcels.crs is None:
        raise ValueError("Parcels must be nonempty and declare a CRS for HUC assignment")
    if "pid" not in parcels or parcels.pid.isna().any():
        raise ValueError("Parcels require non-null pid identifiers for HUC assignment")
    validate_unique_rows(parcels, ["pid"], "Parcels for HUC assignment")
    if (
        parcels.geometry.isna().any()
        or parcels.geometry.is_empty.any()
        or (~parcels.geometry.is_valid).any()
        or (~parcels.geometry.geom_type.isin(["Polygon", "MultiPolygon"])).any()
    ):
        raise ValueError("Parcels require valid, nonempty polygon geometries for HUC assignment")
    crs = parcels.estimate_utm_crs()
    if crs is None:
        raise ValueError("Could not determine a projected CRS for HUC assignment")
    parcels_m = parcels[["pid", "geometry"]].to_crs(crs)
    areas = parcels_m.set_index("pid").geometry.area
    if (areas <= 0).any():
        raise ValueError("Parcels must have positive area for HUC assignment")
    intersections = gpd.overlay(
        parcels_m, hucs[["huc_id", "geometry"]].to_crs(crs),
        how="intersection", keep_geom_type=False,
    )
    intersections["overlap"] = intersections.geometry.area
    intersections = intersections[intersections.overlap > 0]
    # A multipart intersection can produce several pieces for the same pair.
    totals = intersections.groupby(["pid", "huc_id"], as_index=False)["overlap"].sum()
    dominant = totals.sort_values(
        ["pid", "overlap", "huc_id"], ascending=[True, False, True], kind="stable"
    ).drop_duplicates("pid").copy()
    missing = sorted(set(parcels.pid) - set(dominant.pid))
    if missing:
        raise ValueError(f"Parcels do not intersect a HUC polygon; missing pid values: {missing[:10]}")
    dominant["area_fraction"] = (dominant.overlap / dominant.pid.map(areas)).clip(0, 1)
    return dominant[["pid", "huc_id", "area_fraction"]].reset_index(drop=True)


def _input_tables(package: Any) -> set[str]:
    """Discover parameter tables in one input GeoPackage.

    Parameters
    ----------
    package : Any
        Path-like package identifier accepted by
        :func:`src.io_utils.list_geopackage_tables`.

    Returns
    -------
    set[str]
        SQLite table names beginning with ``input_``. These are discovered
        by name, whether or not they are registered in ``gpkg_contents``.

    Raises
    ------
    FileNotFoundError
        If the input GeoPackage does not exist.
    """
    return {name for name in list_geopackage_tables(package) if name.startswith("input_")}


def load_scoped_input_tables(
    cfg: Mapping[str, Any],
    parcels: gpd.GeoDataFrame,
    logger: Any,
    reader: Callable[..., pd.DataFrame],
) -> dict[str, pd.DataFrame] | None:
    """Resolve model input rows across parcel, domain, and HUC scopes.

    Domain tables omit ``pid``; HUC tables replace it with ``huc_id`` (or
    legacy ``huc12``). Other schema keys and fixed/distribution columns are
    unchanged. For each parcel and natural key, precedence is exact parcel
    row, HUC row, domain row, then parcel-default (NULL ``pid``) row.
    A HUC may be assigned via ``parcel_huc``/``parcel_huc12`` or greatest-area
    overlap. Numeric HUC/domain distributions receive a scope-stable
    ``sample_group`` unless one is explicitly provided.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Model configuration containing ``parcels`` and ``domain`` paths, and
        optionally ``huc_parameters`` (legacy ``plet_forcing``).
    parcels : geopandas.GeoDataFrame
        Modeled parcels with ``pid`` and geometry, optionally bearing
        ``attrs['source_pid_universe']`` for IDs outside the clipped domain.
    logger : Any
        Logger with ``info`` and ``warning`` methods.
    reader : Callable[..., pandas.DataFrame]
        Input-table reader accepting ``(package, table, required_columns,
        label, logger)``.

    Returns
    -------
    dict[str, pandas.DataFrame] or None
        Tables keyed by their ``input_*`` name, with scoped rows expanded to
        modeled parcel IDs. Returns ``None`` if no HUC package or domain
        parameter tables are configured. Input files are not modified.

    Raises
    ------
    ValueError
        If packages contain unknown/invalid input tables, HUC geometry or
        assignments are invalid, scoped rows conflict, or values violate
        model input constraints.
    FileNotFoundError
        If an explicitly configured input GeoPackage is missing.

    Notes
    -----
    HUC inheritance uses exactly one HUC per parcel; ``area_fraction`` is
    diagnostic, not a blending weight. Classifications such as land cover and
    HSG are resolved first so conditional parameter rows can be filtered.
    """
    huc_path = ci_get(cfg, "huc_parameters")
    legacy_path = ci_get(cfg, "plet_forcing")
    if huc_path is not None and legacy_path is not None:
        raise ValueError("Specify huc_parameters or plet_forcing, not both")
    if legacy_path is not None:
        logger.warning("plet_forcing is deprecated; use huc_parameters")
        huc_path = legacy_path
    domain_path = ci_get(cfg, "domain")
    domain_names = (
        _input_tables(domain_path)
        if domain_path is not None and Path(domain_path).suffix.lower() == ".gpkg"
        and Path(domain_path).exists() else set()
    )
    if huc_path is None and not domain_names:
        return None
    huc_names = _input_tables(huc_path) if huc_path is not None else set()
    for label, names in (("domain", domain_names), ("HUC", huc_names)):
        unknown = sorted(names - KNOWN_INPUT_TABLES)
        if unknown:
            raise ValueError(f"Unknown input_* table(s) in {label} GeoPackage: {unknown}")

    parcel_path = ci_get(cfg, "parcels")
    parcel_names = set(list_geopackage_tables(parcel_path))
    huc_key = "huc_id"
    mapping = None
    hucs = None
    if huc_path is not None:
        layers = set(list_geopackage_tables(huc_path))
        candidates = [name for name in ("hucs", "huc12") if name in layers]
        if len(candidates) != 1:
            raise ValueError("HUC GeoPackage requires exactly one hucs (or legacy huc12) layer")
        layer = candidates[0]
        huc_key = "huc12" if layer == "huc12" else "huc_id"
        hucs = read_geodataframe(huc_path, layer=layer)
        hucs = normalize_columns(hucs).rename(columns={huc_key: "huc_id"})
        hucs = validate_huc_polygons(hucs)
        if layer == "huc12" and not hucs.huc_level.eq(12).all():
            raise ValueError("Legacy huc12 layer requires 12-digit identifiers")
        relationship = "parcel_huc" if "parcel_huc" in parcel_names else (
            "parcel_huc12" if huc_key == "huc12" and "parcel_huc12" in parcel_names else None
        )
        if relationship is None:
            logger.info("Assigning parcels to HUCs by greatest area overlap")
            mapping = assign_parcels_to_hucs(parcels, hucs)
        else:
            key = "huc12" if relationship == "parcel_huc12" else "huc_id"
            mapping = reader(parcel_path, relationship, ["pid", key], relationship, logger)
            mapping = mapping.rename(columns={key: "huc_id"})
            mapping["huc_id"] = mapping.huc_id.map(normalize_huc_id)
            validate_unique_rows(mapping, ["pid"], relationship)
            source_ids = set(parcels.attrs.get("source_pid_universe", parcels.pid))
            unknown = set(mapping.pid) - source_ids
            if unknown:
                raise ValueError(f"{relationship} references unknown parcel IDs: {sorted(unknown)[:10]}")
            unknown_hucs = set(mapping.huc_id) - set(hucs.huc_id)
            if unknown_hucs:
                raise ValueError(f"{relationship} references unknown HUCs: {sorted(unknown_hucs)[:10]}")
            mapping = mapping[mapping.pid.isin(parcels.pid)].copy()
            missing = set(parcels.pid) - set(mapping.pid)
            if missing:
                raise ValueError(f"{relationship} must assign every modeled parcel; missing pid values: {sorted(missing)[:10]}")
        if "area_fraction" in mapping:
            validate_numeric_columns_in_domain(mapping, ["area_fraction"], FRACTION_DOMAIN, "parcel_huc")
            low = mapping[pd.to_numeric(mapping.area_fraction) < 0.75]
            if not low.empty:
                logger.warning(
                    "Some parcels have less than 75% of their area in the assigned dominant HUC; "
                    f"review boundary parcels: {low.head(10).to_dict(orient='records')}",
                    extra={"file_only": True},
                )

    tables: dict[str, pd.DataFrame] = {}
    classifications: dict[str, pd.Series] = {}
    order = ["input_land_cover", "input_hsg"] + sorted(
        KNOWN_INPUT_TABLES - {"input_land_cover", "input_hsg"}
    )
    for name in order:
        spec = INPUT_VARIABLE_SPECS[name]
        scoped = name in domain_names or name in huc_names
        frames: list[pd.DataFrame] = []
        for scope, package, names in (
            ("parcel", parcel_path, parcel_names),
            ("domain", domain_path, domain_names),
            ("huc", huc_path, huc_names),
        ):
            if name not in names:
                continue
            keys = list(spec.key_columns)
            if scope != "parcel":
                keys = [key for key in keys if key != "pid"]
                if scope == "huc":
                    keys.insert(0, huc_key)
            table = reader(package, name, keys, name, logger).copy()
            if scope != "parcel" and "pid" in table:
                raise ValueError(f"{scope} {name} must omit pid; its scope supplies parcel assignment")
            if scope == "parcel" and "pid" in table:
                source_ids = set(parcels.attrs.get("source_pid_universe", parcels.pid))
                unknown_pids = set(table.pid.dropna()) - source_ids
                if unknown_pids:
                    raise ValueError(f"{name} references unknown parcel IDs: {sorted(unknown_pids)[:10]}")
                # Rows for parcels clipped out of the modeled domain must not
                # survive scope inheritance, even when valid in the source.
                table = table[table.pid.isna() | table.pid.isin(parcels.pid)].copy()
                if "oid" in spec.key_columns and (table.pid.isna() ^ table.oid.isna()).any():
                    raise ValueError(f"{name} supports exact pid/oid rows or one row with both pid and oid NULL")
            if scope == "huc":
                table = table.rename(columns={huc_key: "huc_id"})
                table["huc_id"] = table.huc_id.map(normalize_huc_id)
                unknown = set(table.huc_id) - set(hucs.huc_id)
                if unknown:
                    raise ValueError(f"{name} references HUCs absent from the spatial layer: {sorted(unknown)[:10]}")
            if "pollutant" in table:
                if table.pollutant.isna().any() or table.pollutant.astype(str).str.strip().eq("").any():
                    raise ValueError(f"{scope} {name} requires nonblank pollutant keys")
                table["pollutant"] = table.pollutant.map(normalize_pollutant_label)
            if "pathway" in table:
                from .input_config import _normalize_pathway_column
                table = _normalize_pathway_column(table, name, logger)
            from .plet_rusle import normalize_plet_hsg, normalize_plet_land_cover
            for column, normalize in (("land_cover", normalize_plet_land_cover), ("hsg", normalize_plet_hsg)):
                if column in table:
                    table[column] = table[column].map(normalize)
            if spec.kind == "classification":
                require_columns(table, ["value"], name, logger)
                other = [c for c in statistic_columns(table.columns) if c != "value"]
                if other and table[other].notna().any(axis=1).any():
                    raise ValueError(f"{name} classifications require fixed values, not distributions")
                normalize = normalize_plet_hsg if spec.name == "hsg" else normalize_plet_land_cover
                table["value"] = table.value.map(normalize)
            elif spec.kind == "fixed_numeric":
                other = [c for c in statistic_columns(table.columns) if c != "value"]
                if "value" not in table or (other and table[other].notna().any(axis=1).any()):
                    raise ValueError(f"{name} is deterministic and only allows fixed value")
                physical = NONNEGATIVE_DOMAIN if spec.name == "selection_weight" else FRACTION_DOMAIN
                validate_numeric_columns_in_domain(table, ["value"], physical, name)
            else:
                checked = table.assign(parameter=spec.name)
                label = (
                    "pollutant_concentrations"
                    if spec.name in {"surface_concentration", "subsurface_concentration"}
                    else spec.name
                )
                validate_stats_rows(checked, label)
            natural = [c for c in ("pid", "huc_id", "land_cover", "hsg", "pollutant", "pathway", "oid") if c in table]
            if natural:
                validate_unique_rows(table, natural, f"{scope} {name}")
            elif len(table) > 1:
                raise ValueError(f"domain {name} permits only one row per natural key")
            if not scoped:
                tables[name] = table
                continue
            if scope == "parcel":
                if "pid" not in table:
                    table["pid"] = pd.NA
                table["_priority"] = table.pid.notna().astype(int) * 3
            else:
                table["_priority"] = 2 if scope == "huc" else 1
                if "sample_group" not in table:
                    table["sample_group"] = None
                for index, row in table.iterrows():
                    group = row["sample_group"]
                    if pd.isna(group) or not str(group).strip():
                        identity = ":".join(str(row[c]) for c in natural if c != "pid")
                        prefix = f"huc{len(str(row['huc_id']))}" if scope == "huc" else "domain"
                        table.at[index, "sample_group"] = f"{prefix}:{name}:{identity}"
                if scope == "huc":
                    table = mapping[["pid", "huc_id"]].merge(table, on="huc_id").drop(columns="huc_id")
                else:
                    table = pd.DataFrame({"pid": parcels.pid}).merge(table, how="cross")
            if table.empty:
                frames.append(table)
                continue
            defaults = table[table.pid.isna()].drop(columns="pid")
            table = pd.concat(
                [table[table.pid.notna()], pd.DataFrame({"pid": parcels.pid}).merge(defaults, how="cross")],
                ignore_index=True,
            )
            for column in ("land_cover", "hsg"):
                if column in table and column not in spec.key_columns:
                    effective = classifications.get(column)
                    if effective is None:
                        raise ValueError(f"{name} uses {column} keys but no effective {column} inputs were supplied")
                    table = table[table[column].eq(table.pid.map(effective))].drop(columns=column)
            if "oid" in table and table.oid.isna().any():
                edges = reader(parcel_path, "parcel_outlets", ["pid", "oid"], "parcel_outlets", logger)
                defaults = table[table.oid.isna()].drop(columns="oid").merge(edges, on="pid")
                table = pd.concat([table[table.oid.notna()], defaults], ignore_index=True)
            frames.append(table)
        if scoped and frames:
            merged = pd.concat(frames, ignore_index=True)
            keys = ["pid"] + [c for c in spec.key_columns if c != "pid"]
            if "pathway" in merged:
                keys.append("pathway")
                if merged.pathway.isna().any() and merged.pathway.notna().any():
                    raise ValueError(f"{name} cannot mix aggregate and pathway-specific rows across scopes")
            validate_unique_rows(merged, keys + ["_priority"], name)
            merged = merged.sort_values("_priority", kind="stable").drop_duplicates(keys, keep="last")
            tables[name] = merged.drop(columns="_priority").reset_index(drop=True)
        if name in {"input_land_cover", "input_hsg"} and name in tables:
            table = tables[name]
            defaults = table[table.pid.isna()]
            values = pd.Series(index=parcels.pid, dtype=object)
            if not defaults.empty:
                values[:] = defaults.iloc[0]["value"]
            values.update(table[table.pid.notna()].set_index("pid")["value"])
            classifications[spec.name] = values
    return tables
