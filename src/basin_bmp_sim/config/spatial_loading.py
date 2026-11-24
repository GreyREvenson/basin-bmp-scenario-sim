"""Spatial domain, parcel connectivity, outlets, selection, and delivery inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import geopandas as gpd
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.parcel_graph import validate_acyclic_downstream_map
from basin_bmp_sim.domain.constants import (
    CFG_DELIVERY_RATIOS,
    CFG_DOMAIN,
    CFG_OUTLETS,
    CFG_PARCELS,
    COL_MEAN,
    COL_OID,
    COL_PID,
    COL_PID_DOWN,
    COL_POLLUTANT,
    COL_PROBABILITY,
    COL_TARGET,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    GPKG_PARCELS_LAYER,
    GPKG_PARCEL_DOWN_TABLE,
    GPKG_PARCEL_OUTLETS_TABLE,
    GPKG_OUTLETS_LAYER,
    GPKG_OUTLET_STATS_TABLE,
    GPKG_INPUT_SELECTION_WEIGHT,
    GPKG_DELIVERY_RATIO_TABLES,
)
from basin_bmp_sim.common.io import (
    MissingInputTableError,
    list_geopackage_tables,
    read_geodataframe,
    read_geopackage_integer_ids,
)
from basin_bmp_sim.common.utils import ci_get
from basin_bmp_sim.domain.constants import FRACTION_DOMAIN, NONNEGATIVE_DOMAIN, POSITIVE_DOMAIN
from basin_bmp_sim.config.validation.statistics import require_columns, validate_unique_rows
from basin_bmp_sim.config.validation.domains import validate_numeric_columns_in_domain
from basin_bmp_sim.config.table_sources import (
    _normalize_identifier_columns,
    _read_gpkg_input_table,
    _load_fixed_numeric_variable_table,
    _normalize_pollutant_column,
    _ensure_projected,
)


def _load_domain(cfg: Dict[str, Any], logger: Any) -> gpd.GeoDataFrame:
    """Load and normalize the model domain boundary.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    geopandas.GeoDataFrame
        Domain boundary in a projected CRS with lowercase columns.
    """
    raw_domain = ci_get(cfg, CFG_DOMAIN)
    if raw_domain is None:
        raise ValueError("domain is required")
    domain_path = Path(raw_domain)
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    layer = None
    if domain_path.suffix.lower() == ".gpkg" and "domain" in list_geopackage_tables(domain_path):
        layer = "domain"
    domain = read_geodataframe(domain_path, layer=layer) if layer else read_geodataframe(domain_path)
    domain = _ensure_projected(domain, logger)
    return domain.rename(columns={c: c.lower() for c in domain.columns})


def _load_parcels(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load the spatial ``parcels`` layer from the consolidated parcel GeoPackage."""
    raw_parcels = ci_get(cfg, CFG_PARCELS)
    if raw_parcels is None:
        raise ValueError("parcels is required and must point to the consolidated parcels GeoPackage")
    parcels_path = Path(raw_parcels)
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels GeoPackage not found: {parcels_path}")
    try:
        parcels = read_geodataframe(
            parcels_path, layer=GPKG_PARCELS_LAYER, fid_as_index=True
        )
    except TypeError:
        # Older GeoPandas/Fiona combinations may not accept ``fid_as_index``.
        parcels = read_geodataframe(parcels_path, layer=GPKG_PARCELS_LAYER)
    except Exception as exc:
        raise ValueError(
            f"Parcels GeoPackage must contain spatial layer '{GPKG_PARCELS_LAYER}': {parcels_path}"
        ) from exc

    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if COL_PID in parcels.columns:
        parcels = parcels.reset_index(drop=True)
    else:
        # In GeoPackage schema v3, ``pid`` is the layer's INTEGER PRIMARY KEY.
        # OGR/GDAL usually exposes that field as the feature ID rather than as a
        # normal attribute.  Prefer the FID index when the reader provides it.
        index_name = str(parcels.index.name or "").lower()
        if index_name in {"fid", COL_PID}:
            feature_ids = parcels.index.to_numpy(copy=True)
            parcels = parcels.reset_index(drop=True)
            parcels.insert(0, COL_PID, feature_ids)
        else:
            # Some GeoPandas/Fiona versions ignore ``fid_as_index`` and return a
            # zero-based RangeIndex. Recover the real GeoPackage feature IDs
            # directly from SQLite instead of mistaking row numbers for parcel IDs.
            feature_ids = read_geopackage_integer_ids(
                parcels_path, GPKG_PARCELS_LAYER, COL_PID
            )
            if len(feature_ids) != len(parcels):
                raise ValueError(
                    "Could not recover parcel IDs from the GeoPackage: the number "
                    "of pid values does not match the number of parcel features"
                )
            parcels = parcels.reset_index(drop=True)
            parcels.insert(0, COL_PID, feature_ids)
    parcels = _normalize_identifier_columns(parcels, [COL_PID], CFG_PARCELS)
    if parcels[COL_PID].duplicated().any():
        dup_pids = sorted(parcels.loc[parcels[COL_PID].duplicated(), COL_PID].unique().tolist())
        raise ValueError(f"Parcel IDs must be unique in the source parcels layer; duplicates found: {dup_pids}")
    source_pid_universe = set(parcels[COL_PID].tolist())
    if parcels.crs is None:
        raise ValueError("Parcels layer must declare a CRS")
    # The domain establishes the one analysis CRS used for all geometry math.
    parcels = parcels.to_crs(domain.crs)
    parcels = gpd.overlay(parcels, domain, how="intersection")
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if parcels.empty:
        raise ValueError("No parcels remain after clipping to the domain")
    parcels = _normalize_identifier_columns(parcels, [COL_PID], CFG_PARCELS)
    if parcels[COL_PID].duplicated().any():
        dup_pids = sorted(parcels.loc[parcels[COL_PID].duplicated(), COL_PID].unique().tolist())
        raise ValueError(f"Parcel IDs must be unique after clipping; duplicates found: {dup_pids}")
    if parcels.geometry.isna().any() or parcels.geometry.is_empty.any():
        raise ValueError("Parcels contain null or empty geometry after clipping")
    if (~parcels.geometry.is_valid).any():
        raise ValueError("Parcels contain invalid geometry after clipping")
    parcels["area_m2"] = parcels.geometry.area
    parcels["perim_m"] = parcels.geometry.length
    parcels["area_ha"] = parcels["area_m2"] / 10000.0
    validate_numeric_columns_in_domain(
        parcels, ["area_m2", "area_ha", "perim_m"], POSITIVE_DOMAIN, CFG_PARCELS
    )
    parcels.attrs["source_pid_universe"] = sorted(source_pid_universe)
    return parcels


def _load_parcel_graph(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load required ``parcel_down(pid,pid_down)``; no legacy fallback."""
    package = ci_get(cfg, CFG_PARCELS)
    try:
        return _read_gpkg_input_table(
            package,
            GPKG_PARCEL_DOWN_TABLE,
            [COL_PID, COL_PID_DOWN],
            GPKG_PARCEL_DOWN_TABLE,
            logger,
        )
    except MissingInputTableError as exc:
        raise ValueError(
            f"Required 'parcel_down(pid,pid_down)' table missing from {package}. "
            "Legacy parcel_up is not supported. Regenerate the drainage table "
            "using utils/create_parcel_down.py, or migrate the GeoPackage. "
            "An empty parcel_down table is valid when all parcels are outlets."
        ) from exc


def _build_parcel_down_map(
    downstream_rows: pd.DataFrame,
    parcel_ids: Sequence[str],
    source_parcel_ids: Optional[Sequence[str]] = None,
    logger: Any = None,
) -> Dict[str, Optional[str]]:
    """Validate one immediate downstream PID per source, filtering clipped edges.

    The GeoPackage stores ``source pid -> pid_down``. The model derives the
    reverse immediate-upstream adjacency in memory for wetland catchments.
    Unmodeled-domain links are filtered only *after* all source IDs have been
    checked, so missing/typo IDs do not masquerade as domain clipping.
    """
    ordered_pids = [str(pid).strip() for pid in parcel_ids]
    modeled = set(ordered_pids)
    source = ({str(pid).strip() for pid in source_parcel_ids}
              if source_parcel_ids is not None else modeled)
    downstream: Dict[str, Optional[str]] = {pid: None for pid in ordered_pids}
    source_to_target: dict[str,str] = {}
    unknown: set[str] = set()
    clipped_edges = 0

    def normalize_pid(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
            text = str(int(value))
        return text

    for row_idx, row in downstream_rows.iterrows():
        pid = normalize_pid(row[COL_PID])
        pid_down = normalize_pid(row[COL_PID_DOWN])
        if not pid or not pid_down:
            raise ValueError(f"parcel_down row {row_idx} requires nonblank pid and pid_down")
        if (pid == '*' or pid_down == '*' or ',' in pid or ',' in pid_down):
            raise ValueError("parcel_down requires a single integer pid_down per row; "
                             "wildcards and comma-separated values are invalid")
        if pid == pid_down:
            raise ValueError(f"parcel_down contains a self-link / directed cycle: {pid} -> {pid}")
        if pid not in source:
            unknown.add(pid)
        if pid_down not in source:
            unknown.add(pid_down)
        if pid in source_to_target:
            raise ValueError(f"parcel_down contains more than one downstream "
                             f"relationship for pid {pid}")
        source_to_target[pid] = pid_down
        if pid in modeled and pid_down in modeled:
            downstream[pid] = pid_down
        else:
            clipped_edges += 1
    if unknown:
        raise ValueError("parcel_down references parcel IDs absent from source "
                         f"parcels: {sorted(unknown)[:10]}")
    if clipped_edges and logger is not None:
        logger.verbose(f"Filtered {clipped_edges} parcel_down link(s) outside modeled domain")

    # Check acyclicity after domain clipping. The reverse graph is used only
    # for validation and wetland traversal; the stored table is downstream.
    upstream: Dict[str, List[str]] = {pid: [] for pid in ordered_pids}
    for pid, to in downstream.items():
        if to is not None:
            upstream[to].append(pid)
    validate_acyclic_downstream_map(downstream)
    return downstream


def _validate_parcel_outlet_topology(
    parcel_down_map: Mapping[str, Optional[str]],
    parcel_out_map: Mapping[str, Sequence[str]],
) -> None:
    """Check complete downstream outlet membership for the representative graph.

    If p drains immediately to q, every modeled outlet downstream of q must
    also receive p's load. Extra outlets for p are valid (nested gauges).
    Parcels with no downstream neighbor may have any outlet membership.
    The check uses the post-domain-clipping mappings, not source-wide edges.
    """
    violations: List[str] = []
    for pid, downstream_pid in parcel_down_map.items():
        if downstream_pid is None:
            continue
        source_outlets = set(map(str, parcel_out_map.get(str(pid), ())))
        downstream_outlets = set(map(str, parcel_out_map.get(str(downstream_pid), ())))
        missing = sorted(downstream_outlets - source_outlets)
        if missing:
            violations.append(
                f'pid={pid} -> pid_down={downstream_pid}: missing downstream outlet(s) {missing}'
            )
    if violations:
        raise ValueError(
            'parcel_outlets is inconsistent with parcel_down: each upstream '
            'parcel must list every modeled outlet associated with its '
            'immediate downstream parcel. Example violation(s): '
            + '; '.join(violations[:10])
        )


def _load_parcel_outlets(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load normalized parcel-to-outlet edges from ``parcels.gpkg``."""
    return _read_gpkg_input_table(
        ci_get(cfg, CFG_PARCELS),
        GPKG_PARCEL_OUTLETS_TABLE,
        [COL_PID, COL_OID],
        GPKG_PARCEL_OUTLETS_TABLE,
        logger,
    )


def _load_parcel_selection(
    cfg: Dict[str, Any], parcels: pd.DataFrame, logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Load optional parcel selection weights from ``input_selection_weight``.

    A row with ``pid IS NULL`` supplies the default weight; exact integer PID
    rows override it.
    """
    if parcels.empty:
        raise ValueError("No parcels available for selection")
    table = _load_fixed_numeric_variable_table(
        cfg, GPKG_INPUT_SELECTION_WEIGHT, [COL_PID], logger,
        required=False, domain=NONNEGATIVE_DOMAIN,
        input_tables=input_tables,
    )
    parcel_ids = [int(pid) for pid in parcels[COL_PID].tolist()]
    if table is None or table.empty:
        return pd.DataFrame({
            COL_PID: parcel_ids,
            COL_PROBABILITY: np.full(len(parcels), 1.0 / len(parcels)),
        })
    table = table.copy()
    default_rows = table[table[COL_PID].isna()]
    if len(default_rows) > 1:
        raise ValueError(
            f"{GPKG_INPUT_SELECTION_WEIGHT} may contain at most one NULL-pid default row"
        )
    default_value = None if default_rows.empty else float(default_rows.iloc[0]["value"])
    exact = {
        int(row[COL_PID]): float(row["value"])
        for _, row in table[table[COL_PID].notna()].iterrows()
    }
    unknown = sorted(set(exact) - set(parcel_ids))
    if unknown:
        raise ValueError(
            f"{GPKG_INPUT_SELECTION_WEIGHT} references parcel IDs not found after clipping: {unknown[:10]}"
        )
    weights = []
    for pid in parcel_ids:
        if pid in exact:
            weights.append(exact[pid])
        elif default_value is not None:
            weights.append(default_value)
        else:
            raise ValueError(
                f"{GPKG_INPUT_SELECTION_WEIGHT} has no value for pid={pid}; "
                "supply the parcel explicitly or add one row with pid=NULL"
            )
    weights_arr = np.asarray(weights, dtype=float)
    if ((~np.isfinite(weights_arr)) | (weights_arr < 0.0)).any():
        raise ValueError(
            f"{GPKG_INPUT_SELECTION_WEIGHT} values must be finite and >= 0"
        )
    total = float(weights_arr.sum())
    if total <= 0.0:
        raise ValueError(f"{GPKG_INPUT_SELECTION_WEIGHT} values sum to zero or negative")
    return pd.DataFrame({
        COL_PID: parcel_ids,
        COL_PROBABILITY: weights_arr / total,
    })


def _load_outlet_loc(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load the spatial ``outlets`` layer from the consolidated outlet GeoPackage."""
    raw_outlets = ci_get(cfg, CFG_OUTLETS)
    if raw_outlets is None:
        raise ValueError("outlets is required and must point to the consolidated outlets GeoPackage")
    outlet_path = Path(raw_outlets)
    if not outlet_path.exists():
        raise FileNotFoundError(f"Outlets GeoPackage not found: {outlet_path}")
    try:
        outlet_loc = read_geodataframe(outlet_path, layer=GPKG_OUTLETS_LAYER).to_crs(domain.crs)
    except Exception as exc:
        raise ValueError(
            f"Outlets GeoPackage must contain spatial layer '{GPKG_OUTLETS_LAYER}': {outlet_path}"
        ) from exc
    outlet_loc = outlet_loc.rename(columns={c: c.lower() for c in outlet_loc.columns})
    require_columns(outlet_loc, [COL_OID], CFG_OUTLETS, logger)
    outlet_loc = _normalize_identifier_columns(outlet_loc, [COL_OID], CFG_OUTLETS)
    validate_unique_rows(outlet_loc, [COL_OID], CFG_OUTLETS)
    if outlet_loc.geometry.isna().any() or outlet_loc.geometry.is_empty.any():
        raise ValueError("Outlets contain null or empty geometry")
    if (~outlet_loc.geometry.is_valid).any():
        raise ValueError("Outlets contain invalid geometry")
    non_points = ~outlet_loc.geometry.geom_type.isin(["Point", "MultiPoint"])
    if non_points.any():
        raise ValueError("Outlets layer must contain point geometry")
    return outlet_loc.reset_index(drop=True)


def _load_optional_outlet_stats(
    cfg: Dict[str, Any],
    value_col: str,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Load one optional statistic column from the consolidated ``outlet_stats`` table."""
    if value_col not in {COL_TARGET, COL_MEAN}:
        raise ValueError(f"Unsupported outlet statistic column: {value_col}")
    label = f"{GPKG_OUTLET_STATS_TABLE}.{value_col}"
    required_cols = [COL_OID, COL_POLLUTANT, value_col]
    package = ci_get(cfg, CFG_OUTLETS)
    base_required = [COL_OID, COL_POLLUTANT]
    try:
        df = _read_gpkg_input_table(
            package, GPKG_OUTLET_STATS_TABLE, base_required, "outlet_stats", logger
        )
    except MissingInputTableError:
        logger.verbose("outlet_stats table not present; skipping optional outlet statistics")
        return None
    if value_col not in df.columns:
        logger.verbose(f"outlet_stats does not contain {value_col}; skipping {label}")
        return None
    df = df[df[value_col].notna()].copy()
    if df.empty:
        return None
    require_columns(df, required_cols, label, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, label, logger)
    validate_numeric_columns_in_domain(df, [value_col], NONNEGATIVE_DOMAIN, label)
    validate_unique_rows(df, [COL_OID, COL_POLLUTANT], label)
    return df[list(required_cols)].reset_index(drop=True)


def _load_delivery_ratios(
    cfg: Dict[str, Any], logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """Load the four parcel-to-outlet delivery variables from dedicated tables.

    A global default row is represented by ``pid IS NULL AND oid IS NULL``.
    Partial-NULL keys are invalid.
    """
    base = _load_parcel_outlets(cfg, logger)[[COL_PID, COL_OID]].copy()
    base[COL_PID] = base[COL_PID].astype("int64")
    base[COL_OID] = base[COL_OID].astype(str).str.strip()
    result = base.copy()
    any_supplied = False
    valid_pairs = set(zip(base[COL_PID], base[COL_OID]))

    for column, table_name in GPKG_DELIVERY_RATIO_TABLES.items():
        table = _load_fixed_numeric_variable_table(
            cfg, table_name, [COL_PID, COL_OID], logger,
            required=False, domain=FRACTION_DOMAIN,
            input_tables=input_tables,
        )
        if table is None or table.empty:
            result[column] = 1.0
            continue
        any_supplied = True
        table = table.copy()
        default_mask = table[COL_PID].isna() & table[COL_OID].isna()
        defaults = table[default_mask]
        if len(defaults) > 1:
            raise ValueError(f"{table_name} may contain at most one NULL/NULL default row")
        partial_null = table[COL_PID].isna() ^ table[COL_OID].isna()
        if partial_null.any():
            raise ValueError(
                f"{table_name} supports exact pid/oid rows or one row with both pid and oid NULL"
            )
        default = 1.0 if defaults.empty else float(defaults.iloc[0]["value"])
        exact_rows = table[~default_mask]
        exact = {
            (int(row[COL_PID]), str(row[COL_OID])): float(row["value"])
            for _, row in exact_rows.iterrows()
        }
        unknown = sorted(set(exact) - valid_pairs, key=lambda pair: (pair[0], pair[1]))
        if unknown:
            raise ValueError(
                f"{table_name} references parcel/outlet relationships not present in parcel_outlets: {unknown[:10]}"
            )
        result[column] = [exact.get(pair, default) for pair in zip(result[COL_PID], result[COL_OID])]

    if not any_supplied:
        logger.verbose("No delivery-ratio input tables present; using neutral defaults")
    return result.reset_index(drop=True)


def _complete_delivery_ratio_defaults(
    delivery_ratios: Optional[pd.DataFrame],
    parcel_out_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Return a complete parcel/outlet delivery table with neutral defaults.

    Parameters
    ----------
    delivery_ratios : Optional[pd.DataFrame]
        Parcel-to-outlet delivery-ratio table, if configured.
    parcel_out_map : Mapping[str, Sequence[str]]
        Mapping from parcel IDs to connected outlet IDs.

    Returns
    -------
    pd.DataFrame
        Complete parcel-to-outlet delivery-ratio table.

    Raises
    ------
    ValueError
        If any configured delivery-ratio value lies outside ``[0, 1]``.
    """
    columns = [COL_PID, COL_OID, COL_SDR_F_TO_S, COL_SDR_S_TO_O, COL_NDR_F_TO_S, COL_NDR_S_TO_O]
    if delivery_ratios is None:
        out = pd.DataFrame(columns=columns)
    else:
        out = delivery_ratios.copy()
    existing = {(str(row[COL_PID]), str(row[COL_OID])) for _, row in out.iterrows()}
    rows: List[Dict[str, Any]] = []
    for pid, outlet_ids in parcel_out_map.items():
        for oid in outlet_ids:
            key = (str(pid), str(oid))
            if key in existing:
                continue
            rows.append({
                COL_PID: pid,
                COL_OID: key[1],
                COL_SDR_F_TO_S: 1.0,
                COL_SDR_S_TO_O: 1.0,
                COL_NDR_F_TO_S: 1.0,
                COL_NDR_S_TO_O: 1.0,
            })
    if rows:
        defaults = pd.DataFrame(rows, columns=columns)
        out = defaults if out.empty else pd.concat([out, defaults], ignore_index=True)
    for column in (COL_SDR_F_TO_S, COL_SDR_S_TO_O, COL_NDR_F_TO_S, COL_NDR_S_TO_O):
        if column not in out.columns:
            out[column] = pd.Series(dtype=float)
        out[column] = pd.to_numeric(out[column], errors="raise")
        validate_numeric_columns_in_domain(
            out, [column], FRACTION_DOMAIN, CFG_DELIVERY_RATIOS
        )
    return out.reset_index(drop=True)
