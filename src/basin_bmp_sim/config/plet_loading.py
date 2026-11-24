"""PLET/RUSLE hydrology, concentration, and HUC12 input preparation."""

from __future__ import annotations
from basin_bmp_sim.science.plet_classification import normalize_plet_hsg, normalize_plet_land_cover
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, Tuple
import pandas as pd
from basin_bmp_sim.domain.constants import (
    CFG_PARCELS,
    CFG_PLET_FORCING,
    LOAD_PLET_INPUTS,
    LOAD_CONCENTRATIONS,
    COL_PID,
    COL_POLLUTANT,
    COL_PATHWAY,
    PLET_PATHWAY_VALUES,
    PLET_HYDROLOGY_LABEL as _PLET_HYDROLOGY_LABEL,
    GPKG_PARCEL_HUC12_TABLE,
    GPKG_PLET_HUC12_LAYER,
    HUC12_PLET_PARAMETER_INPUT_TABLES,
    GPKG_INPUT_SURFACE_CONCENTRATION,
    GPKG_INPUT_SUBSURFACE_CONCENTRATION,
    COL_HUC12,
    COL_AREA_FRACTION,
)
from basin_bmp_sim.common.io import (
    MissingInputTableError,
    list_geopackage_tables,
    read_csv_table,
    read_geodataframe,
)
from basin_bmp_sim.common.utils import ci_get, normalize_columns
from basin_bmp_sim.config.distributions import statistic_columns, stats_from_row
from basin_bmp_sim.domain.constants import FRACTION_DOMAIN
from basin_bmp_sim.config.validation.statistics import (
    require_columns,
    validate_distribution_bounds,
    validate_stats_rows,
    validate_unique_rows,
)
from basin_bmp_sim.config.validation.domains import validate_numeric_columns_in_domain
from basin_bmp_sim.config.validation.plet import validate_plet_input_table
from basin_bmp_sim.config.table_sources import (
    _load_table_source,
    _normalize_huc12,
    _normalize_identifier_columns,
    _normalize_pathway_column,
    _normalize_pollutant_column,
    _read_gpkg_input_table,
    _rows_for_pid,
    _validate_explicit_references,
)


def _effective_plet_classification_pairs(
    table: pd.DataFrame, parcel_ids: Sequence[str]
) -> List[Tuple[str, str]]:
    """Return normalized land-cover/HSG pairs actually used by modeled parcels."""
    from basin_bmp_sim.science.plet_classification import canonical_parameter_name, normalize_plet_hsg, normalize_plet_land_cover

    pairs: set[Tuple[str, str]] = set()
    for pid in map(str, parcel_ids):
        effective = {canonical_parameter_name(row["parameter"]): row for row in _rows_for_pid(table, pid)}
        land_row = effective.get("land_cover")
        hsg_row = effective.get("hsg")
        if land_row is None or hsg_row is None:
            continue
        pairs.add((normalize_plet_land_cover(land_row["value"]), normalize_plet_hsg(hsg_row["value"])))
    return sorted(pairs)


def _load_plet_hydrology_records(
    lookup_path: Union[str, Path],
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Read and validate fixed PLET CN/infiltration records for deterministic helpers.

    Parameters
    ----------
    lookup_path : Union[str, Path]
        Path to the user-supplied PLET hydrology lookup table.

    Returns
    -------
    Dict[Tuple[str, str], Tuple[float, float]]
        Hydrology values keyed by normalized land-cover and HSG classes.

    Raises
    ------
    FileNotFoundError
        If the configured PLET hydrology lookup file does not exist.
    ValueError
        If the deterministic lookup is missing required columns or values,
        contains duplicate rows, or contains stochastic definitions.
    """
    from basin_bmp_sim.domain.constants import PLET_DERIVED_PARAMETERS as _PLET_DERIVED_PARAMETERS
    from basin_bmp_sim.science.plet_classification import canonical_parameter_name, normalize_plet_hsg, normalize_plet_land_cover
    path = Path(lookup_path)
    if not path.exists():
        raise FileNotFoundError(f"PLET hydrology lookup table not found: {path}")
    table = read_csv_table(path)
    table = normalize_columns(table)
    records: Dict[Tuple[str, str], Dict[str, float]] = {}
    if {"land_cover", "hsg", "parameter"} <= set(table.columns):
        for row_index, row in table.iterrows():
            land_cover = normalize_plet_land_cover(row["land_cover"])
            hsg = normalize_plet_hsg(row["hsg"])
            parameter = canonical_parameter_name(row["parameter"])
            if parameter not in _PLET_DERIVED_PARAMETERS:
                continue
            stats = stats_from_row(row)
            if set(stats) != {"value"}:
                raise ValueError(
                    "The deterministic PLET hydrology helper requires fixed lookup values; "
                    f"row {row_index} for {land_cover}/{hsg}/{parameter} is stochastic"
                )
            records.setdefault((land_cover, hsg), {})[parameter] = float(stats["value"])
        output: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for key, values in records.items():
            if set(values) != set(_PLET_DERIVED_PARAMETERS):
                raise ValueError(f"PLET hydrology lookup is incomplete for {key}")
            output[key] = (float(values["cn"]), float(values["infiltration_fraction"]))
        return output
    required = {"land_cover", "hsg", "cn", "infiltration_fraction"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"PLET hydrology lookup is missing required columns: {missing}")
    output = {}

    for _, row in table.iterrows():
        key = (normalize_plet_land_cover(row["land_cover"]), normalize_plet_hsg(row["hsg"]))
        if key in output:
            raise ValueError(f"Duplicate PLET hydrology lookup row for {key}")
        output[key] = (float(row["cn"]), float(row["infiltration_fraction"]))
    return output


def _load_parameter_stats_table(
    source: Any,
    label: str,
    logger: Any,
    parcel_ids: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Load a parcel parameter statistics table from a frame or CSV path(s)."""
    if source is None:
        return None
    from basin_bmp_sim.science.plet_classification import canonical_parameter_name
    df = _load_table_source(source, [COL_PID, "parameter"], label, logger)
    df = _normalize_identifier_columns(df, [COL_PID], label, allow_null_pid=True)
    if parcel_ids is not None:
        _validate_explicit_references(
            df[COL_PID].tolist(), parcel_ids, label=label, allow_default=True
        )
    df["parameter"] = df["parameter"].map(canonical_parameter_name)
    validate_unique_rows(df, [COL_PID, "parameter"], label)
    validate_stats_rows(df, label)
    return df


def _load_plet_parameter_table(
    source: Any,
    parcel_ids: Sequence[str],
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Load PLET numeric parameters and fixed land-cover/HSG classifications."""
    if source is None:
        return None
    from basin_bmp_sim.science.plet_classification import canonical_parameter_name
    from basin_bmp_sim.domain.constants import PLET_CLASSIFICATION_PARAMETERS
    df = _load_table_source(source, [COL_PID, "parameter"], LOAD_PLET_INPUTS, logger)
    df = _normalize_identifier_columns(df, [COL_PID], LOAD_PLET_INPUTS, allow_null_pid=True)
    _validate_explicit_references(
        df[COL_PID].tolist(), parcel_ids, label=LOAD_PLET_INPUTS, allow_default=True
    )
    df["parameter"] = df["parameter"].map(canonical_parameter_name)
    validate_unique_rows(df, [COL_PID, "parameter"], LOAD_PLET_INPUTS)

    categorical_mask = df["parameter"].isin(PLET_CLASSIFICATION_PARAMETERS)
    categorical_rows = df.loc[categorical_mask].copy()
    categorical_stat_cols = [
        col for col in statistic_columns(categorical_rows.columns)
        if str(col).strip().lower() != "value"
    ]
    if categorical_stat_cols and not categorical_rows.empty:
        bad_stats = categorical_rows[categorical_stat_cols].notna().any(axis=1)
        if bad_stats.any():
            rows = categorical_rows.index[bad_stats].tolist()
            raise ValueError(
                "PLET land_cover and hsg are classifications and must use only a fixed value; "
                f"distribution statistics were supplied at rows {rows}"
            )
    numeric_rows = df.loc[~categorical_mask].copy()
    if not numeric_rows.empty:
        validate_stats_rows(numeric_rows, LOAD_PLET_INPUTS)
    df = pd.concat([categorical_rows, numeric_rows], axis=0).sort_index()
    return validate_plet_input_table(df, parcel_ids)


def _load_plet_hydrology_lookup(
    path: Any,
    logger: Any,
    required_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    required_parcel_pairs: Optional[Mapping[int, Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """Load required land-cover/HSG hydrology distributions for PLET mode.

    The table is long-form with one row per ``land_cover`` x ``hsg`` x
    ``parameter``. Exactly two parameters are required for every supported
    pairing: ``cn`` and ``infiltration_fraction``. Each row follows the same
    fixed-value/distribution schema as other numeric model inputs.

    Parameters
    ----------
    path : Any
        Path to the required PLET hydrology lookup CSV file.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        Validated PLET hydrology lookup table.

    Raises
    ------
    ValueError
        If the required hydrology lookup is absent, contains unsupported parameters, or lacks required land-cover/HSG coverage.
    """
    if path is None:
        raise ValueError(
            "PLET hydrology requires input_curve_number and "
            "input_infiltration_fraction in parcels.gpkg"
        )
    from basin_bmp_sim.domain.constants import PLET_HSG_VALUES, PLET_LAND_COVERS
    from basin_bmp_sim.science.plet_classification import canonical_parameter_name, normalize_plet_hsg, normalize_plet_land_cover
    if isinstance(path, pd.DataFrame):
        table = normalize_columns(path.copy())
        require_columns(
            table, ["land_cover", "hsg", "parameter"],
            _PLET_HYDROLOGY_LABEL, logger
        )
    else:
        paths = [path] if isinstance(path, (str, Path)) else list(path)
        frames: List[pd.DataFrame] = []
        for item in paths:
            logger.verbose(f"Reading {_PLET_HYDROLOGY_LABEL} from {item}")
            frame = read_csv_table(item)
            frame = normalize_columns(frame)
            require_columns(
                frame,
                ["land_cover", "hsg", "parameter"],
                f"{_PLET_HYDROLOGY_LABEL} ({item})",
                logger,
            )
            frames.append(frame)
        table = pd.concat(frames, ignore_index=True)
    table["land_cover"] = table["land_cover"].map(normalize_plet_land_cover)
    table["hsg"] = table["hsg"].map(normalize_plet_hsg)
    table["parameter"] = table["parameter"].map(canonical_parameter_name)

    allowed_parameters = {"cn", "infiltration_fraction"}
    unexpected_parameters = sorted(set(table["parameter"]) - allowed_parameters)
    if unexpected_parameters:
        raise ValueError(
            f"{_PLET_HYDROLOGY_LABEL} contains unsupported parameters: "
            f"{unexpected_parameters}; expected only cn and infiltration_fraction"
        )

    validate_stats_rows(table, _PLET_HYDROLOGY_LABEL)
    keys = ["land_cover", "hsg", "parameter"]
    per_parcel = COL_PID in table.columns
    if per_parcel:
        keys.insert(0, COL_PID)
        if required_parcel_pairs is None:
            raise ValueError("Parcel-keyed hydrology inputs require modeled parcel classifications")
        table = _normalize_identifier_columns(
            table, [COL_PID], _PLET_HYDROLOGY_LABEL, allow_null_pid=True
        )
        validate_unique_rows(table, keys, _PLET_HYDROLOGY_LABEL)
        _validate_explicit_references(
            table[COL_PID], list(required_parcel_pairs), label=_PLET_HYDROLOGY_LABEL,
            allow_default=True,
        )
        defaults = table[table[COL_PID].isna()].drop(columns=COL_PID)
        table = pd.concat([
            pd.DataFrame({COL_PID: list(required_parcel_pairs)}).merge(defaults, how="cross"),
            table[table[COL_PID].notna()],
        ], ignore_index=True).drop_duplicates(keys, keep="last")
    validate_unique_rows(table, keys, _PLET_HYDROLOGY_LABEL)

    if required_pairs is None:
        pairs = {(land_cover, hsg) for land_cover in PLET_LAND_COVERS for hsg in PLET_HSG_VALUES}
    else:
        pairs = {
            (normalize_plet_land_cover(land_cover), normalize_plet_hsg(hsg))
            for land_cover, hsg in required_pairs
        }
    expected = {
        (land_cover, hsg, parameter)
        for land_cover, hsg in pairs
        for parameter in ("cn", "infiltration_fraction")
    }
    supplied = set(
        zip(table["land_cover"], table["hsg"], table["parameter"])
    )
    if per_parcel:
        expected = {
            (pid, land_cover, hsg, parameter)
            for pid, (land_cover, hsg) in required_parcel_pairs.items()
            for parameter in ("cn", "infiltration_fraction")
        }
        supplied = set(table[keys].itertuples(index=False, name=None))
    missing = sorted(expected - supplied)
    # Extra valid rows are harmless and make a reusable hydrology table possible;
    # only the land-cover/HSG combinations used by this run must be complete.
    if missing:
        scope = "every supported land_cover x hsg pairing" if required_pairs is None else "every land_cover x hsg pairing used by modeled parcels"
        raise ValueError(
            f"{_PLET_HYDROLOGY_LABEL} must define cn and infiltration_fraction for {scope}; "
            f"missing={missing}"
        )

    validate_distribution_bounds(
        table,
        _PLET_HYDROLOGY_LABEL,
        parameter_col="parameter",
        bounds={
            "cn": (1.0e-9, 100.0),
            "infiltration_fraction": (0.0, 1.0),
        },
    )
    return table.reset_index(drop=True)


def _load_pollutant_concentrations(
    source: Any,
    pollutants: List[str],
    logger: Any,
    parcel_ids: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Load the unified surface/subsurface concentration table."""
    if source is None:
        return None
    df = _load_table_source(
        source,
        [COL_PID, COL_POLLUTANT, COL_PATHWAY],
        LOAD_CONCENTRATIONS,
        logger,
    )
    df = _normalize_pollutant_column(df, COL_POLLUTANT, LOAD_CONCENTRATIONS, logger)
    df = _normalize_pathway_column(df, LOAD_CONCENTRATIONS, logger)
    df = _normalize_identifier_columns(df, [COL_PID], LOAD_CONCENTRATIONS, allow_null_pid=True)
    if parcel_ids is not None:
        _validate_explicit_references(
            df[COL_PID].tolist(), parcel_ids, label=LOAD_CONCENTRATIONS, allow_default=True
        )
    df = df[df[COL_POLLUTANT].isin(pollutants)].copy()
    invalid_pathways = sorted(set(df[COL_PATHWAY]) - set(PLET_PATHWAY_VALUES))
    if invalid_pathways:
        raise ValueError(
            f"{LOAD_CONCENTRATIONS} contains unsupported pathways {invalid_pathways}; "
            f"expected only {list(PLET_PATHWAY_VALUES)}"
        )
    validate_unique_rows(
        df, [COL_PID, COL_POLLUTANT, COL_PATHWAY], LOAD_CONCENTRATIONS
    )
    pathways = df[COL_PATHWAY].copy().reset_index(drop=True)
    numeric_df = df.drop(columns=[COL_PATHWAY]).reset_index(drop=True)
    validate_stats_rows(numeric_df, LOAD_CONCENTRATIONS)
    numeric_df.insert(2, COL_PATHWAY, pathways)
    return numeric_df.reset_index(drop=True)


def _split_plet_concentrations(
    table: Optional[pd.DataFrame],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Split the unified concentration table into runtime pathway tables.

    The simulation historically carries surface and groundwater/subsurface
    concentrations separately in its in-memory state.  Keeping that internal
    representation avoids changing scenario-state and output behavior while
    allowing users to maintain only one concentration input file.
    """
    if table is None:
        return None, None
    surface = table[table[COL_PATHWAY] == "surface"].copy().reset_index(drop=True)
    subsurface = table[table[COL_PATHWAY] == "subsurface"].copy().reset_index(drop=True)
    return surface, subsurface


def _load_parcel_huc12(
    cfg: Dict[str, Any],
    parcel_ids: Sequence[int],
    source_parcel_ids: Sequence[int],
    logger: Any,
) -> pd.DataFrame:
    """Load the dominant-HUC12 assignment for each modeled parcel."""
    table = _read_gpkg_input_table(
        ci_get(cfg, CFG_PARCELS),
        GPKG_PARCEL_HUC12_TABLE,
        [COL_PID, COL_HUC12],
        GPKG_PARCEL_HUC12_TABLE,
        logger,
    ).copy()
    table[COL_HUC12] = table[COL_HUC12].map(
        lambda value: _normalize_huc12(value, label=f"{GPKG_PARCEL_HUC12_TABLE}.{COL_HUC12}")
    )
    validate_unique_rows(table, [COL_PID], GPKG_PARCEL_HUC12_TABLE)

    if COL_AREA_FRACTION in table.columns:
        table[COL_AREA_FRACTION] = pd.to_numeric(table[COL_AREA_FRACTION], errors="raise")
        validate_numeric_columns_in_domain(
            table, [COL_AREA_FRACTION], FRACTION_DOMAIN, GPKG_PARCEL_HUC12_TABLE
        )

    source_ids = {int(pid) for pid in source_parcel_ids}
    modeled_ids = {int(pid) for pid in parcel_ids}
    unknown = sorted(set(table[COL_PID].astype(int)) - source_ids)
    if unknown:
        raise ValueError(
            f"{GPKG_PARCEL_HUC12_TABLE} references parcel IDs not found in the source parcels layer: {unknown[:10]}"
        )
    table = table.loc[table[COL_PID].astype(int).isin(modeled_ids)].copy()
    missing = sorted(modeled_ids - set(table[COL_PID].astype(int)))
    if missing:
        raise ValueError(
            f"{GPKG_PARCEL_HUC12_TABLE} must assign every modeled parcel to one HUC12; missing pid values: {missing[:10]}"
        )
    if COL_AREA_FRACTION in table.columns:
        low = table[table[COL_AREA_FRACTION] < 0.75]
        if not low.empty:
            preview = low[[COL_PID, COL_HUC12, COL_AREA_FRACTION]].head(10).to_dict(orient="records")
            logger.warning(
                "Some parcels have less than 75% of their area in the assigned dominant HUC12; "
                f"review boundary parcels if needed: {preview}",
                extra={"file_only": True},
            )
    return table.reset_index(drop=True)


def _load_plet_huc12_layer(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Validate the spatial HUC12 layer and return its identifiers/metadata.

    HUC12 geometry is structural/provenance information only. PLET model
    variables are stored in separate ``input_*`` attribute tables in the same
    GeoPackage so they can use the standard fixed-value/distribution schema.
    """
    package = ci_get(cfg, CFG_PLET_FORCING)
    if package is None:
        return pd.DataFrame(columns=[COL_HUC12])
    package = Path(package)
    if not package.exists():
        raise FileNotFoundError(f"PLET forcing GeoPackage not found: {package}")

    tables = set(list_geopackage_tables(package))
    if GPKG_PLET_HUC12_LAYER not in tables:
        raise MissingInputTableError(
            f"PLET forcing GeoPackage {package} must contain spatial layer "
            f"{GPKG_PLET_HUC12_LAYER!r}"
        )

    spatial = read_geodataframe(package, layer=GPKG_PLET_HUC12_LAYER)
    if spatial.empty:
        raise ValueError(f"{package}:{GPKG_PLET_HUC12_LAYER} must not be empty")
    if spatial.crs is None:
        raise ValueError(f"{package}:{GPKG_PLET_HUC12_LAYER} must declare a CRS")
    if spatial.geometry.isna().any() or spatial.geometry.is_empty.any():
        raise ValueError(
            f"{package}:{GPKG_PLET_HUC12_LAYER} contains null or empty geometry"
        )
    if (~spatial.geometry.is_valid).any():
        raise ValueError(f"{package}:{GPKG_PLET_HUC12_LAYER} contains invalid geometry")

    table = pd.DataFrame(spatial.drop(columns=[spatial.geometry.name], errors="ignore"))
    table = normalize_columns(table)
    technical_ids = [c for c in ("fid", "id", "row_id") if c in table.columns]
    if technical_ids:
        table = table.drop(columns=technical_ids)
    require_columns(
        table, [COL_HUC12],
        f"{GPKG_PLET_HUC12_LAYER} ({package})", logger,
    )
    table[COL_HUC12] = table[COL_HUC12].map(
        lambda value: _normalize_huc12(
            value, label=f"{GPKG_PLET_HUC12_LAYER}.{COL_HUC12}"
        )
    )
    validate_unique_rows(table, [COL_HUC12], GPKG_PLET_HUC12_LAYER)
    return table.reset_index(drop=True)


def _assemble_huc12_plet_parameter_source(
    cfg: Dict[str, Any],
    huc12_layer: pd.DataFrame,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Assemble HUC12 ``input_*`` tables into the runtime long form.

    Every numeric HUC12-scale PLET variable uses the same distribution schema
    as parcel inputs. One row describes one HUC12/parameter combination.
    """
    package = ci_get(cfg, CFG_PLET_FORCING)
    if package is None:
        return None

    valid_hucs = set(huc12_layer[COL_HUC12].astype(str))
    frames: List[pd.DataFrame] = []
    existing_tables = set(list_geopackage_tables(package))
    for parameter, table_name in HUC12_PLET_PARAMETER_INPUT_TABLES.items():
        if table_name not in existing_tables:
            continue
        table = _read_gpkg_input_table(
            package, table_name, [COL_HUC12], table_name, logger
        ).copy()
        if table.empty:
            continue
        table[COL_HUC12] = table[COL_HUC12].map(
            lambda value: _normalize_huc12(value, label=f"{table_name}.{COL_HUC12}")
        )
        if parameter in {"r", "k", "ls", "c", "p"}:
            require_columns(table, ["land_cover"], table_name, logger)
            from basin_bmp_sim.science.plet_classification import normalize_plet_land_cover
            table["land_cover"] = table["land_cover"].map(normalize_plet_land_cover)
        keys = [COL_HUC12] + (["land_cover"] if parameter in {"r", "k", "ls", "c", "p"} else [])
        validate_unique_rows(table, keys, table_name)
        unknown = sorted(set(table[COL_HUC12].astype(str)) - valid_hucs)
        if unknown:
            raise ValueError(
                f"{table_name} references HUC12 value(s) not found in the spatial "
                f"{GPKG_PLET_HUC12_LAYER} layer: {unknown[:10]}"
            )

        # Add the known parameter context before validating distribution rows.
        # Unit labels such as ``fraction`` are intentionally ambiguous without
        # parameter/table context because they may represent several
        # dimensionless model quantities.  Parcel long-form inputs already
        # carry a ``parameter`` column; HUC12 one-variable-per-table inputs must
        # materialize that context here before unit-aware validation.
        table.insert(1, "parameter", parameter)
        if parameter == "hsg":
            from basin_bmp_sim.science.plet_classification import normalize_plet_hsg
            require_columns(table, ["value"], table_name, logger)
            other_stats = [col for col in statistic_columns(table.columns) if col != "value"]
            if other_stats and table[other_stats].notna().any(axis=1).any():
                raise ValueError(f"{table_name} must use fixed HSG values, not distributions")
            table["value"] = table["value"].map(normalize_plet_hsg)
        else:
            validate_stats_rows(table, table_name)
        frames.append(table)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def _merge_huc12_plet_parameters(
    parcel_source: pd.DataFrame,
    parcel_huc12: pd.DataFrame,
    huc_source: Optional[pd.DataFrame],
    parcel_ids: Sequence[int],
) -> pd.DataFrame:
    """Merge HUC12 parameter rows into parcel PLET rows with explicit precedence.

    Precedence is parcel-specific ``input_*`` row > HUC12 ``input_*`` row >
    NULL-pid parcel default. Distributional HUC12 inputs are materialized for
    parcels assigned to that HUC12. Unless the user supplies ``sample_group``,
    an automatic HUC12/parameter sample group is assigned so one value is drawn
    per HUC12 per scenario and shared by all parcels in that HUC12.
    """
    from basin_bmp_sim.science.plet_classification import canonical_parameter_name

    out = parcel_source.copy()
    if huc_source is None or huc_source.empty:
        return out

    mapping = parcel_huc12.set_index(COL_PID)
    huc_rows = huc_source.copy()
    huc_rows["parameter"] = huc_rows["parameter"].map(canonical_parameter_name)
    huc_rows["land_cover"] = huc_rows.get("land_cover", pd.Series(index=huc_rows.index, dtype=object))
    validate_unique_rows(huc_rows, [COL_HUC12, "parameter", "land_cover"], "HUC12 PLET inputs")

    available_hucs = set(huc_rows[COL_HUC12].astype(str))
    used_hucs = set(mapping[COL_HUC12].astype(str))
    # A HUC may legitimately omit a particular parameter because parcel-level
    # defaults can provide it, but every mapped HUC must exist in at least one
    # HUC input row when a HUC forcing source is configured.
    missing_hucs = sorted(used_hucs - available_hucs)
    if missing_hucs:
        raise ValueError(
            "HUC12 PLET input tables have no parameter rows for HUC12 value(s): "
            f"{missing_hucs[:10]}"
        )

    canonical = out["parameter"].map(canonical_parameter_name)
    exact_keys = {
        (int(pid), str(param))
        for pid, param in zip(out[COL_PID], canonical)
        if not pd.isna(pid)
    }

    rows: List[Dict[str, Any]] = []
    for pid in map(int, parcel_ids):
        huc12 = str(mapping.at[pid, COL_HUC12])
        subset = huc_rows[huc_rows[COL_HUC12].astype(str) == huc12]
        land_rows = [row for row in _rows_for_pid(out, pid) if row["parameter"] == "land_cover"]
        land_cover = str(land_rows[0]["value"]).strip().lower() if land_rows else None
        for _, source_row in subset.iterrows():
            parameter = str(source_row["parameter"])
            source_land_cover = source_row["land_cover"]
            if pd.notna(source_land_cover) and source_land_cover != land_cover:
                continue
            if (pid, parameter) in exact_keys:
                continue
            row = source_row.to_dict()
            row.pop(COL_HUC12, None)
            row.pop("land_cover", None)
            row[COL_PID] = pid
            raw_group = row.get("sample_group")
            if raw_group is None or pd.isna(raw_group) or str(raw_group).strip() == "":
                group_key = f"{huc12}:{parameter}"
                if pd.notna(source_land_cover):
                    group_key += f":{source_land_cover}"
                row["sample_group"] = f"huc12:{group_key}"
            note = row.get("notes")
            prefix = f"HUC12 forcing from {huc12}"
            row["notes"] = prefix if note is None or pd.isna(note) or not str(note).strip() else f"{prefix}; {note}"
            rows.append(row)

    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return out


def _merge_huc12_plet_concentrations(
    cfg: Dict[str, Any],
    parcel_source: pd.DataFrame,
    parcel_huc12: pd.DataFrame,
    plet_inputs: pd.DataFrame,
    parcel_ids: Sequence[int],
    logger: Any,
) -> pd.DataFrame:
    """Apply HUC12/land-cover concentrations beneath parcel-specific rows."""
    from basin_bmp_sim.science.plet_classification import normalize_plet_land_cover

    package = ci_get(cfg, CFG_PLET_FORCING)
    if package is None:
        return parcel_source
    existing = set(list_geopackage_tables(package))
    frames = []
    for pathway, table_name in (
        ("surface", GPKG_INPUT_SURFACE_CONCENTRATION),
        ("subsurface", GPKG_INPUT_SUBSURFACE_CONCENTRATION),
    ):
        if table_name not in existing:
            continue
        table = _read_gpkg_input_table(
            package, table_name, [COL_HUC12, "land_cover", COL_POLLUTANT], table_name, logger
        ).copy()
        if table.empty:
            continue
        table[COL_HUC12] = table[COL_HUC12].map(
            lambda value: _normalize_huc12(value, label=f"{table_name}.{COL_HUC12}")
        )
        table["land_cover"] = table["land_cover"].map(normalize_plet_land_cover)
        validate_unique_rows(table, [COL_HUC12, "land_cover", COL_POLLUTANT], table_name)
        validate_stats_rows(table, table_name)
        table.insert(3, COL_PATHWAY, pathway)
        frames.append(table)
    if not frames:
        return parcel_source

    huc_rows = pd.concat(frames, ignore_index=True)
    mapping = parcel_huc12.set_index(COL_PID)[COL_HUC12].astype(str)
    # The spatial layer is the authority for valid HUC identifiers, including
    # those not currently assigned to a modeled parcel.
    layer = _load_plet_huc12_layer(cfg, logger)
    unknown = sorted(set(huc_rows[COL_HUC12]) - set(layer[COL_HUC12]))
    if unknown:
        raise ValueError(f"HUC12 concentration rows reference unknown HUC12s: {unknown[:10]}")
    exact = {
        (int(row[COL_PID]), str(row[COL_POLLUTANT]), str(row[COL_PATHWAY]))
        for _, row in parcel_source[parcel_source[COL_PID].notna()].iterrows()
    }
    rows: List[Dict[str, Any]] = []
    for pid in map(int, parcel_ids):
        huc12 = mapping.at[pid]
        land_rows = [row for row in _rows_for_pid(plet_inputs, pid) if row["parameter"] == "land_cover"]
        if not land_rows:
            continue
        land_cover = normalize_plet_land_cover(land_rows[0]["value"])
        subset = huc_rows[
            (huc_rows[COL_HUC12] == huc12) & (huc_rows["land_cover"] == land_cover)
        ]
        for _, source in subset.iterrows():
            key = (pid, str(source[COL_POLLUTANT]), str(source[COL_PATHWAY]))
            if key in exact:
                continue
            row = source.to_dict()
            row.pop(COL_HUC12)
            row.pop("land_cover")
            row[COL_PID] = pid
            group = row.get("sample_group")
            if group is None or pd.isna(group) or not str(group).strip():
                row["sample_group"] = f"huc12:{huc12}:{land_cover}:{key[1]}:{key[2]}"
            rows.append(row)
    if not rows:
        return parcel_source
    return pd.concat([parcel_source, pd.DataFrame(rows)], ignore_index=True, sort=False)


def plet_hydrology_from_classifications(
    land_cover: Any,
    hsg: Any,
    *,
    lookup_path: Path,
) -> Dict[str, Any]:
    """Resolve fixed hydrology values from a lookup table.

        This deterministic helper requires an explicit user-supplied hydrology
        lookup path. The actual PLET/RUSLE scenario path samples the validated
        hydrology input table loaded from configuration, allowing each CN and
        infiltration fraction to be a fixed value or a distribution.

        Parameters
        ----------
        land_cover : Any
            PLET land-cover classification.
        hsg : Any
            Hydrologic soil group classification.
        lookup_path : Path
            Path to the user-supplied PLET hydrology lookup table.

        Returns
        -------
        Dict[str, Any]
            Resolved PLET hydrology values for the land-cover/HSG combination.

    """
    canonical_land_cover = normalize_plet_land_cover(land_cover)
    canonical_hsg = normalize_plet_hsg(hsg)
    records = _load_plet_hydrology_records(lookup_path)
    curve_number, infiltration_fraction = records[(canonical_land_cover, canonical_hsg)]
    return {
        "land_cover": canonical_land_cover,
        "hsg": canonical_hsg,
        "cn": curve_number,
        "infiltration_fraction": infiltration_fraction,
    }
