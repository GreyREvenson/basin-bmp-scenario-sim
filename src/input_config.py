"""Normalize, default, load, and assemble model inputs.

Filesystem and serialization operations live in :mod:`src.io_utils`, while
validation policy lives in :mod:`src.input_validation`. This module owns
configuration defaults, aliases, type normalization, input preparation, and
construction of the validated data bundle consumed by the simulation.
"""

from __future__ import annotations

from pathlib import Path
import difflib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, Tuple
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd

from .constants import (
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_BMP_EFFICIENCY,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BMP_SEL_PROB_VIA_COSTS,
    CFG_BUFFER_DEPTH_FT,
    CFG_OUTPUTS,
    CFG_VERBOSE,
    CFG_CPS,
    CFG_DELIVERY_RATIOS,
    CFG_DOMAIN,
    CFG_N_SCENARIOS,
    CFG_OUTLETS,
    CFG_PARALLEL,
    CFG_PARCELS,
    CFG_POLLUTANT_LOAD_RATE,
    CFG_POLLUTANTS,
    CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE,
    CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW,
    CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    CFG_RANDOM_SEED,
    CFG_LOAD_GENERATION,
    CFG_INPUT_DISTRIBUTIONS,
    LOAD_MODE_STATISTICAL,
    LOAD_MODE_PLET_RUSLE,
    LOAD_PLET_INPUTS,
    LOAD_RUSLE_INPUTS,
    LOAD_CONCENTRATIONS,
    LOAD_GROUNDWATER_CONCENTRATIONS,
    LOAD_GROUNDWATER_LOADS,
    LOAD_TREAT_GROUNDWATER_WITH_BMPS,
    COL_CPS,
    COL_MEAN,
    COL_OID,
    COL_PID,
    COL_PID_UP,
    COL_POLLUTANT,
    COL_PROBABILITY,
    COL_TARGET,
    COL_UNIT,
    COL_PATHWAY,
    PATHWAY_VALUES,
    PLET_PATHWAY_VALUES,
    DEFAULT_BMP_FAIL_REDUCTION,
    DEFAULT_BUFFER_DEPTH_FT,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    GPKG_PARCELS_LAYER,
    GPKG_PARCEL_UP_TABLE,
    GPKG_PARCEL_OUTLETS_TABLE,
    GPKG_OUTLETS_LAYER,
    GPKG_OUTLET_STATS_TABLE,
    PARCEL_PARAMETER_INPUT_TABLES,
    GPKG_INPUT_SELECTION_WEIGHT,
    GPKG_INPUT_CURVE_NUMBER,
    GPKG_INPUT_INFILTRATION_FRACTION,
    GPKG_INPUT_POLLUTANT_LOAD_RATE,
    GPKG_INPUT_SURFACE_CONCENTRATION,
    GPKG_INPUT_SUBSURFACE_CONCENTRATION,
    GPKG_DELIVERY_RATIO_TABLES,
    RUSLE_PARAMETER_NAMES,
)
from .io_utils import (
    MissingInputTableError,
    list_geopackage_tables,
    read_csv_table,
    read_geodataframe,
    read_geopackage_table,
    read_parquet_table,
)
from .utils import ci_get, normalize_columns, normalize_pollutant_label
from .input_units import row_unit, unit_labels_same_scale
from .logging_utils import log_scope
from .input_distributions import (
    DISTRIBUTION_ID,
    statistic_columns,
    stats_from_row,
)
from .input_validation import (
    FRACTION_DOMAIN,
    NONNEGATIVE_DOMAIN,
    POSITIVE_DOMAIN,
    require_columns,
    validate_config,
    validate_distribution_bounds,
    validate_distribution_catalog,
    validate_numeric_columns_in_domain,
    validate_numeric_distribution_rows,
    validate_plet_input_table,
    validate_plet_runtime_inputs,
    validate_statistical_efficiency_coverage,
    validate_statistical_load_rates,
    validate_stats_rows,
    validate_stats_table,
    validate_unique_rows,
    validate_bmp_selection_table,
    validate_trajectory_table,
)

from .input_schema import (
    INPUT_SCHEMA_TABLE,
    INPUT_SCHEMA_VERSION,
    INPUT_VARIABLE_SPECS,
    KNOWN_INPUT_TABLES,
    required_input_tables,
)


_PLET_HYDROLOGY_LABEL = "plet_hydrology_inputs"


def _normalize_identifier_value(value: Any, label: str) -> str:
    """Normalize one relational identifier and reject missing/blank values."""
    if value is None or pd.isna(value):
        raise ValueError(f"{label} must not be null")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an identifier, not a boolean")
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be blank")
    return text


def _normalize_identifier_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> pd.DataFrame:
    """Return a copy with normalized PID/OID-style identifier columns."""
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        values: List[str] = []
        for index, value in out[column].items():
            values.append(_normalize_identifier_value(value, f"{label} row {index} {column}"))
        out[column] = values
    return out


def _validate_explicit_references(
    values: Sequence[Any], valid_values: Sequence[Any], *, label: str, allow_wildcard: bool = True
) -> None:
    """Reject explicit foreign-key values that do not exist in the target universe."""
    valid = {str(value) for value in valid_values}
    supplied = {str(value) for value in values}
    if allow_wildcard:
        supplied.discard("*")
    unknown = sorted(supplied - valid)
    if unknown:
        raise ValueError(f"{label} references unknown identifiers: {unknown[:10]}")



def _normalize_cps_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalize a CPS column and reject booleans, fractions, and non-finite codes."""
    if COL_CPS not in df.columns:
        return df
    out = df.copy()
    normalized: List[int] = []
    for index, value in out[COL_CPS].items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} row {index} cps must be a finite integer, not a boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} row {index} cps={value!r} must be a finite integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{label} row {index} cps={value!r} must be a finite integer")
        normalized.append(int(numeric))
    out[COL_CPS] = normalized
    return out

def _sort_input_table(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Return deterministic row ordering without changing stored key values."""
    if df.empty or not keys:
        return df.reset_index(drop=True)
    out = df.copy()
    temp_cols: List[str] = []
    for index, key in enumerate(keys):
        if key not in out.columns:
            continue
        temp = f"__sort_key_{index}"
        out[temp] = out[key].map(lambda value: "" if pd.isna(value) else str(value).strip())
        temp_cols.append(temp)
    if temp_cols:
        out = out.sort_values(temp_cols, kind="stable").drop(columns=temp_cols)
    return out.reset_index(drop=True)


def _validate_input_package_schema(cfg: Dict[str, Any], load_mode: str, logger: Any) -> None:
    """Reject unknown input tables and require mode-specific tables."""
    raw_package = ci_get(cfg, CFG_PARCELS)
    # The spatial parcel loader owns path existence/requiredness. Keeping this
    # helper focused on schema discovery also makes it easy to unit-test loader
    # stages independently with mocked spatial inputs.
    if raw_package is None:
        return
    package = Path(raw_package)
    if not package.exists():
        return
    table_names = set(list_geopackage_tables(package))
    supplied_input_tables = {name for name in table_names if name.startswith("input_")}
    unknown = sorted(supplied_input_tables - set(KNOWN_INPUT_TABLES))
    if unknown:
        details = []
        known = sorted(KNOWN_INPUT_TABLES)
        for name in unknown:
            match = difflib.get_close_matches(name, known, n=1, cutoff=0.65)
            details.append(f"{name!r}" + (f" (did you mean {match[0]!r}?)" if match else ""))
        raise ValueError("Unknown input_* table(s) in parcels GeoPackage: " + ", ".join(details))

    missing = sorted(set(required_input_tables(load_mode)) - table_names)
    if missing:
        raise ValueError(
            f"load_generation.mode={load_mode!r} requires input table(s) missing from parcels GeoPackage: {missing}"
        )

    if INPUT_SCHEMA_TABLE in table_names:
        schema = read_geopackage_table(package, INPUT_SCHEMA_TABLE)
        schema = normalize_columns(schema)
        require_columns(schema, ["schema_version"], INPUT_SCHEMA_TABLE, logger)
        versions = pd.to_numeric(schema["schema_version"], errors="coerce")
        if len(schema) != 1 or versions.isna().any() or int(versions.iloc[0]) != INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"{INPUT_SCHEMA_TABLE} must contain exactly one row with schema_version={INPUT_SCHEMA_VERSION}"
            )
    else:
        logger.warning(
            f"{INPUT_SCHEMA_TABLE} is not present in {package.name}; accepting the current table layout, "
            f"but future input packages should declare schema_version={INPUT_SCHEMA_VERSION}"
        )


def _merge_csvs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one or more CSV files and combine them into one table.

    Parameters
    ----------
    paths : str, pathlib.Path, or sequence of str or pathlib.Path
        One or more CSV file paths.
    required_cols : sequence of str
        Columns that must be present in every file.
    label : str
        Human-readable dataset name used in logs and errors.
    logger : Any
        Logger used for progress and duplicate warnings.

    Returns
    -------
    pandas.DataFrame
        Concatenated dataframe with duplicates removed on the required key
        columns.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames: List[pd.DataFrame] = []
    for p in paths:
        logger.verbose(f"Reading {label} from {p}")
        df = read_csv_table(p)
        df = normalize_columns(df)
        require_columns(df, required_cols, f"{label} ({p})", logger)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    dedup_subset = list(required_cols)
    if COL_PATHWAY in out.columns and COL_PATHWAY not in dedup_subset:
        dedup_subset.append(COL_PATHWAY)

    # Exact duplicate rows are harmless and may arise when users concatenate
    # identical source files. Conflicting rows for the same logical key are not:
    # accepting them would make file order determine model behavior.
    out = out.drop_duplicates(keep="first")
    dup = out.duplicated(subset=dedup_subset, keep=False)
    if dup.any():
        preview = out.loc[dup, dedup_subset].head(10).to_dict(orient="records")
        raise ValueError(
            f"{label} contains conflicting duplicate rows for logical key {dedup_subset}: {preview}"
        )
    return out.reset_index(drop=True)


def _read_gpkg_input_table(
    package_path: Union[str, Path],
    table_name: str,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one named attribute table from a consolidated GeoPackage input.

    The GeoPackage is a physical container only: each logical dataset retains
    its own normalized relational table and downstream model contracts remain
    unchanged.
    """
    package = Path(package_path)
    if not package.exists():
        raise FileNotFoundError(f"{label} GeoPackage not found: {package}")
    logger.verbose(f"Reading {label} from {package} layer={table_name}")
    try:
        frame = read_geopackage_table(package, table_name)
    except MissingInputTableError as exc:
        raise MissingInputTableError(
            f"{label} requires table/layer '{table_name}' in {package}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Failed to read {label} table/layer '{table_name}' from {package}: {exc}"
        ) from exc
    frame = normalize_columns(pd.DataFrame(frame))
    if "fid" in frame.columns:
        frame = frame.drop(columns=["fid"])
    require_columns(frame, required_cols, f"{label} ({package}:{table_name})", logger)
    identifier_cols = [column for column in (COL_PID, COL_PID_UP, COL_OID) if column in frame.columns]
    if identifier_cols:
        frame = _normalize_identifier_columns(frame, identifier_cols, label)
    sort_cols = list(required_cols)
    if COL_PATHWAY in frame.columns and COL_PATHWAY not in sort_cols:
        sort_cols.append(COL_PATHWAY)
    return _sort_input_table(frame, sort_cols)


def _try_read_gpkg_input_table(
    package_path: Union[str, Path],
    table_name: str,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Read an optional GeoPackage attribute table, returning ``None`` if absent."""
    try:
        return _read_gpkg_input_table(
            package_path, table_name, required_cols, label, logger
        )
    except MissingInputTableError:
        return None


def _load_fixed_numeric_variable_table(
    cfg: Dict[str, Any],
    table_name: str,
    key_cols: Sequence[str],
    logger: Any,
    *,
    required: bool = False,
    domain: Any = None,
) -> Optional[pd.DataFrame]:
    """Load a deterministic per-variable table using the standard input schema."""
    package = ci_get(cfg, CFG_PARCELS)
    reader = _read_gpkg_input_table if required else _try_read_gpkg_input_table
    table = reader(package, table_name, list(key_cols), table_name, logger)
    if table is None:
        return None
    if "value" not in table.columns:
        raise ValueError(f"{table_name} requires a value column")
    if DISTRIBUTION_ID in table.columns:
        bad = table[DISTRIBUTION_ID].notna() & table[DISTRIBUTION_ID].astype(str).str.strip().ne("")
        if bad.any():
            raise ValueError(f"{table_name} is deterministic and does not allow distribution_id")
    other_stats = [c for c in statistic_columns(table.columns) if str(c).lower() != "value"]
    if other_stats and table[other_stats].notna().any(axis=1).any():
        raise ValueError(f"{table_name} is deterministic and only allows fixed value")
    table = table.copy()
    table["value"] = pd.to_numeric(table["value"], errors="raise")
    validate_unique_rows(table, list(key_cols), table_name)
    if domain is not None:
        validate_numeric_columns_in_domain(table, ["value"], domain, table_name)
    return table.reset_index(drop=True)


def _assemble_parcel_parameter_source(
    cfg: Dict[str, Any],
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Assemble dedicated ``input_*`` parameter tables into the runtime long form."""
    package = ci_get(cfg, CFG_PARCELS)
    frames: List[pd.DataFrame] = []
    for parameter, table_name in PARCEL_PARAMETER_INPUT_TABLES.items():
        table = _try_read_gpkg_input_table(
            package, table_name, [COL_PID], table_name, logger
        )
        if table is None:
            continue
        table = table.copy()
        table.insert(1, "parameter", parameter)
        frames.append(table)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def _assemble_plet_hydrology_source(
    cfg: Dict[str, Any],
    logger: Any,
) -> pd.DataFrame:
    """Assemble CN and infiltration tables into the existing hydrology long form."""
    package = ci_get(cfg, CFG_PARCELS)
    specs = (
        ("cn", GPKG_INPUT_CURVE_NUMBER),
        ("infiltration_fraction", GPKG_INPUT_INFILTRATION_FRACTION),
    )
    frames: List[pd.DataFrame] = []
    for parameter, table_name in specs:
        table = _read_gpkg_input_table(
            package, table_name, ["land_cover", "hsg"], table_name, logger
        ).copy()
        table.insert(2, "parameter", parameter)
        frames.append(table)
    return pd.concat(frames, ignore_index=True, sort=False)


def _assemble_plet_concentration_source(
    cfg: Dict[str, Any],
    logger: Any,
) -> pd.DataFrame:
    """Assemble dedicated surface/subsurface concentration variable tables."""
    package = ci_get(cfg, CFG_PARCELS)
    frames: List[pd.DataFrame] = []
    for pathway, table_name in (
        ("surface", GPKG_INPUT_SURFACE_CONCENTRATION),
        ("subsurface", GPKG_INPUT_SUBSURFACE_CONCENTRATION),
    ):
        table = _try_read_gpkg_input_table(
            package, table_name, [COL_PID, COL_POLLUTANT], table_name, logger
        )
        if table is None:
            continue
        table = table.copy()
        table.insert(2, COL_PATHWAY, pathway)
        frames.append(table)
    if not frames:
        return pd.DataFrame(columns=[COL_PID, COL_POLLUTANT, COL_PATHWAY, "value"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_table_source(
    source: Any,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Load a logical input table from either an in-memory frame or CSV path(s)."""
    if isinstance(source, pd.DataFrame):
        frame = normalize_columns(source.copy())
        require_columns(frame, required_cols, label, logger)
        return _sort_input_table(frame, required_cols)
    return _merge_csvs(source, required_cols, label, logger)


def load_bmp_selection_probabilities(
    path: Union[str, Path],
    cps: Sequence[int],
    logger: Any,
) -> pd.DataFrame:
    """Read and normalize explicit BMP-selection probabilities.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the BMP-selection probability CSV file.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        Normalized BMP-selection probability table.
    """
    logger.verbose(f"Reading BMP selection probabilities from {path}")
    df = normalize_columns(read_csv_table(path))

    if COL_PROBABILITY not in df.columns:
        alias = next((name for name in ("pr", "p") if name in df.columns), None)
        if alias is not None:
            df[COL_PROBABILITY] = df[alias]

    if COL_CPS in df.columns:
        cps_numeric = pd.to_numeric(df[COL_CPS], errors="coerce")
        finite_integer = np.isfinite(cps_numeric) & (cps_numeric % 1 == 0)
        df.loc[finite_integer, COL_CPS] = cps_numeric.loc[finite_integer].astype(int)

    if COL_PROBABILITY in df.columns:
        df[COL_PROBABILITY] = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")

    configured_cps = {int(value) for value in cps}
    if COL_CPS in df.columns:
        df = df[df[COL_CPS].isin(configured_cps)].copy()

    validate_bmp_selection_table(df, cps)
    df[COL_CPS] = df[COL_CPS].astype(int)
    df[COL_PROBABILITY] = df[COL_PROBABILITY].astype(float)
    df[COL_PROBABILITY] = df[COL_PROBABILITY] / float(df[COL_PROBABILITY].sum())

    result = df[[COL_CPS, COL_PROBABILITY]].reset_index(drop=True)
    logger.verbose(
        f"Loaded explicit BMP selection probabilities from {path}: "
        f"{result.to_dict(orient='records')}"
    )
    return result


def load_trajectory_records(
    path: Union[str, Path],
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Read and normalize canonical outlet-trajectory records for plotting.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the canonical trajectory Parquet file.

    Returns
    -------
    Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
        Trajectory records keyed by pollutant, outlet, x-axis, and y-axis definitions.
    """
    df = read_parquet_table(path)
    validate_trajectory_table(df)

    for column in ("scenario", "step", "x_value", "y_value"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    out: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
    df = df.sort_values(["scenario", "pollutant", "oid", "x_axis", "y_axis", "step"])
    for _, row in df.iterrows():
        key = (str(row["pollutant"]), str(row["oid"]), str(row["x_axis"]), str(row["y_axis"]))
        out[key].append((int(row["scenario"]), float(row["x_value"]), float(row["y_value"])))
    return out


def _ensure_projected(gdf: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Return geometry in a projected CRS whose linear units are meters.

    Area/perimeter values are model inputs, so merely being projected is not
    sufficient: State Plane feet, for example, must not be interpreted as
    meters.  The model therefore establishes a metric analysis CRS once from
    the domain and transforms all other spatial inputs into that CRS.
    """
    if gdf.crs is None:
        raise ValueError("Spatial inputs must declare a CRS; a missing CRS cannot be safely inferred")

    crs = gdf.crs
    axis_info = list(getattr(crs, "axis_info", ()) or ())
    uses_meters = bool(
        crs.is_projected
        and axis_info
        and all(
            getattr(axis, "unit_conversion_factor", None) is not None
            and abs(float(axis.unit_conversion_factor) - 1.0) < 1e-12
            for axis in axis_info[:2]
        )
    )
    if uses_meters:
        return gdf

    est = gdf.estimate_utm_crs()
    if est is None:
        raise ValueError(f"Could not determine a metric analysis CRS from input CRS {crs}")
    logger.info(f"Reprojecting analysis geometry from {crs} to metric CRS: {est}")
    return gdf.to_crs(est)


def _normalize_pollutant_column(df: pd.DataFrame, col: str, label: str, logger: Any) -> pd.DataFrame:
    """Normalize pollutant labels in a dataframe column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    col : str
        Name of the pollutant column.
    label : str
        Dataset label used in error messages.
    logger : Any
        Logger retained for interface consistency.

    Returns
    -------
    pandas.DataFrame
        Dataframe with standardized pollutant labels.

    Raises
    ------
    ValueError
        If the pollutant column is missing or cannot be normalized.
    """
    if col not in df.columns:
        raise ValueError(f"{label} missing required column '{col}'")
    try:
        df[col] = [normalize_pollutant_label(x) for x in df[col]]
    except Exception as ex:
        raise ValueError(f"Failed to normalize pollutant labels in {label}: {ex}") from ex
    return df


def _normalize_pathway_label(value: Any) -> str:
    """Return a stable, user-extensible pathway label.

    Parameters
    ----------
    value : Any
        Input value to normalize or evaluate.

    Returns
    -------
    str
        Normalized pathway label.
    """
    label = str(value).strip().lower().replace("_", " ")
    return " ".join(label.split())


def _normalize_pathway_column(df: pd.DataFrame, label: str, logger: Any) -> pd.DataFrame:
    """Normalize pathway labels without restricting user-defined pathways.

    Statistical mode may use any non-empty pathway labels shared by the parcel
    load-rate and BMP efficiency inputs. PLET/RUSLE-specific pathway restrictions
    are applied later, after the load-generation mode is known.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    label : str
        Context label used in diagnostics and validation errors.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        Copy of the table with normalized pathway labels.

    Raises
    ------
    ValueError
        If any pathway label is blank after normalization.
    """
    del logger
    if COL_PATHWAY not in df.columns:
        return df
    df[COL_PATHWAY] = df[COL_PATHWAY].map(_normalize_pathway_label)
    bad = df[COL_PATHWAY].eq("")
    if bad.any():
        raise ValueError(f"{label} contains blank pathway labels")
    return df


def _nonblank(value: Any) -> bool:
    """Return whether an input cell contains a nonblank value.

    Parameters
    ----------
    value : Any
        Input value to normalize or evaluate.

    Returns
    -------
    bool
        ``True`` when the value is nonblank; otherwise ``False``.
    """
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _row_stats_raw(row: Mapping[str, Any]) -> Dict[str, float]:
    """Return normalized numeric statistics from an input row.

    Parameters
    ----------
    row : Mapping[str, Any]
        Input table row.

    Returns
    -------
    Dict[str, float]
        Normalized numeric statistics extracted from the row.
    """
    return stats_from_row(row)


def load_distribution_catalog(path: Any, logger: Any = None) -> Optional[pd.DataFrame]:
    """Read and validate the optional reusable input-distribution catalog.

    Parameters
    ----------
    path : Any
        Path or paths to reusable distribution-catalog CSV files.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    Optional[pd.DataFrame]
        Validated distribution catalog, or ``None`` when no catalog is configured.

    Raises
    ------
    ValueError
        If a catalog file is missing the required ``distribution_id`` column.
    """
    if path is None:
        return None
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    frames: List[pd.DataFrame] = []
    for item in paths:
        if logger is not None:
            logger.verbose(f"Reading reusable input distributions from {item}")
        frame = read_csv_table(item)
        frame = normalize_columns(frame)
        if DISTRIBUTION_ID not in frame.columns:
            raise ValueError(f"input_distributions ({item}) is missing required column '{DISTRIBUTION_ID}'")
        frames.append(frame)
    catalog = pd.concat(frames, ignore_index=True)
    validate_distribution_catalog(catalog)
    catalog[DISTRIBUTION_ID] = catalog[DISTRIBUTION_ID].astype(str).str.strip()
    return catalog.reset_index(drop=True)


def resolve_distribution_references(
    df: pd.DataFrame,
    catalog: Optional[pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    """Expand distribution references into inline statistics during input loading.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    catalog : Optional[pd.DataFrame]
        Reusable distribution catalog, if configured.
    label : str
        Context label used in diagnostics and validation errors.

    Returns
    -------
    pd.DataFrame
        Table with distribution references expanded to inline statistics.

    Raises
    ------
    ValueError
        If a distribution reference conflicts with inline statistics, is unknown,
        or cannot be resolved because no catalog is configured.
    """
    out = df.copy()
    if DISTRIBUTION_ID not in out.columns:
        return out
    catalog_map: Dict[str, pd.Series] = {}
    if catalog is not None:
        catalog_map = {str(row[DISTRIBUTION_ID]).strip(): row for _, row in catalog.iterrows()}
    all_stat_cols = set(statistic_columns(out.columns))
    if catalog is not None:
        all_stat_cols.update(statistic_columns(catalog.columns))
    for column in all_stat_cols:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for index, row in out.iterrows():
        ref = row.get(DISTRIBUTION_ID)
        if not _nonblank(ref):
            continue
        ref_id = str(ref).strip()
        inline = _row_stats_raw(row)
        if inline:
            raise ValueError(
                f"{label} row {index} specifies distribution_id={ref_id!r} and inline statistics; use one or the other"
            )
        if not catalog_map:
            raise ValueError(
                f"{label} row {index} references distribution_id={ref_id!r}, but no input_distributions catalog is configured"
            )
        if ref_id not in catalog_map:
            raise ValueError(f"{label} row {index} references unknown distribution_id={ref_id!r}")
        source = catalog_map[ref_id]
        source_units = row_unit(source)
        use_units = row_unit(row)
        if source_units is not None and use_units is not None and not unit_labels_same_scale(source_units, use_units):
            raise ValueError(
                f"{label} row {index} references distribution_id={ref_id!r} defined using "
                f"units {source_units!r}, but the use-site supplies {use_units!r}; "
                "a distribution reference may not reinterpret catalog statistics at a different scale"
            )
        for source_col in statistic_columns(source.index):
            value = source.get(source_col)
            if not pd.isna(value):
                out.at[index, source_col] = float(value)
        if source_units is not None and use_units is None:
            if "units" not in out.columns:
                out["units"] = np.nan
            out.at[index, "units"] = source_units
    return out


def _rows_for_pid(table: Optional[pd.DataFrame], pid: str) -> List[pd.Series]:
    """Resolve wildcard input rows plus parcel-specific overrides for one parcel.

    Parameters
    ----------
    table : Optional[pd.DataFrame]
        Input table containing model data.
    pid : str
        Parcel identifier.

    Returns
    -------
    List[pd.Series]
        Rows applicable to the specified parcel.
    """
    if table is None or table.empty:
        return []
    from .plet_rusle import canonical_parameter_name

    pids = table[COL_PID].astype(str)
    wildcard_rows = table[pids == "*"]
    exact_rows = table[pids == str(pid)]
    combined = pd.concat([wildcard_rows, exact_rows], ignore_index=True)
    if combined.empty:
        return []
    combined = combined.assign(
        _canonical_parameter=combined["parameter"].map(canonical_parameter_name)
    )
    combined = combined.drop_duplicates(subset=["_canonical_parameter"], keep="last")
    return [row for _, row in combined.iterrows()]



def _effective_plet_classification_pairs(
    table: pd.DataFrame, parcel_ids: Sequence[str]
) -> List[Tuple[str, str]]:
    """Return normalized land-cover/HSG pairs actually used by modeled parcels."""
    from .plet_rusle import canonical_parameter_name, normalize_plet_hsg, normalize_plet_land_cover

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
    from .plet_rusle import (
        _PLET_DERIVED_PARAMETERS,
        canonical_parameter_name,
        normalize_plet_hsg,
        normalize_plet_land_cover,
    )
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


def _plet_parameter_defaults(pollutants: Sequence[str]) -> Dict[str, float]:
    """Return the centralized PLET/RUSLE parameter defaults.

    Parameters
    ----------
    pollutants : Sequence[str]
        Pollutant names in model order.

    Returns
    -------
    Dict[str, float]
        Centralized default PLET/RUSLE parameter values.
    """
    defaults: Dict[str, float] = {
        # Required PLET inputs are intentionally excluded. Missing required
        # variables must fail validation rather than being synthesized.
        "ia_ratio": 0.0,
        "runoff_multiplier": 1.0,
        "groundwater_multiplier": 1.0,
        "sediment_multiplier": 1.0,
        "sediment_delivery_multiplier": 1.0,
        "enrichment_ratio": 2.0,
        "sediment_n_pct": 0.0,
        "sediment_p_pct": 0.0,
    }
    for pollutant in pollutants:
        defaults[f"load_multiplier_{str(pollutant).lower()}"] = 1.0
    return defaults


def apply_plet_parameter_defaults(
    parameters: Mapping[str, Any],
    pollutants: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return a parameter mapping with centralized PLET/RUSLE defaults applied.

    Parameters
    ----------
    parameters : Mapping[str, Any]
        Model parameter values keyed by canonical parameter name.
    pollutants : Sequence[str]
        Pollutant names in model order.

    Returns
    -------
    Dict[str, Any]
        Parameter mapping with centralized defaults applied.
    """
    out = dict(parameters)
    for parameter, value in _plet_parameter_defaults(pollutants).items():
        if parameter not in out or out[parameter] is None:
            out[parameter] = value
    return out


def _append_parameter_defaults(
    table: pd.DataFrame,
    pollutants: Sequence[str],
) -> pd.DataFrame:
    """Append wildcard PLET parameter defaults before scenario sampling.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing model data.
    pollutants : Sequence[str]
        Pollutant names in model order.

    Returns
    -------
    pd.DataFrame
        Table containing explicit wildcard default rows.
    """
    defaults = _plet_parameter_defaults(pollutants)
    out = table.copy()
    if "_default_applied" not in out.columns:
        out["_default_applied"] = False
    if "value" not in out.columns:
        out["value"] = np.nan
    existing_wildcards = set(
        out.loc[out[COL_PID].astype(str) == "*", "parameter"].astype(str).tolist()
    )
    rows: List[Dict[str, Any]] = []
    for parameter, value in defaults.items():
        if parameter in existing_wildcards:
            continue
        row = {column: np.nan for column in out.columns}
        row[COL_PID] = "*"
        row["parameter"] = parameter
        row["value"] = value
        row["_default_applied"] = True
        rows.append(row)
    if rows:
        out = pd.concat([out, pd.DataFrame(rows, columns=out.columns)], ignore_index=True)
    return out


def _set_case_insensitive_default(mapping: Dict[str, Any], key: str, value: Any) -> None:
    """Set one input default while respecting existing case-insensitive keys.

    Parameters
    ----------
    mapping : Dict[str, Any]
        Input mapping.
    key : str
        Configuration or mapping key.
    value : Any
        Input value to normalize or evaluate.
    """
    matching_key = next((existing for existing in mapping if str(existing).lower() == key.lower()), None)
    if matching_key is None:
        mapping[key] = value
    elif mapping[matching_key] is None:
        mapping[matching_key] = value


def apply_config_defaults(cfg: Dict[str, Any]) -> None:
    """Apply all supported top-level configuration defaults in one place.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Normalized model configuration mapping.
    """
    _set_case_insensitive_default(cfg, CFG_N_SCENARIOS, 1)
    _set_case_insensitive_default(cfg, CFG_OUTPUTS, "./outputs")
    _set_case_insensitive_default(cfg, CFG_VERBOSE, False)
    _set_case_insensitive_default(cfg, CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT)
    _set_case_insensitive_default(cfg, CFG_BMP_SEL_PROB_VIA_COSTS, False)
    _set_case_insensitive_default(cfg, CFG_BMP_FAIL_RATE, 0.0)
    _set_case_insensitive_default(cfg, CFG_BMP_FAIL_REDUCTION, DEFAULT_BMP_FAIL_REDUCTION)
    _set_case_insensitive_default(cfg, CFG_PARALLEL, {"n_jobs": 1})
    _set_case_insensitive_default(cfg, CFG_LOAD_GENERATION, {})


_CONFIG_PATH_KEYS = (
    CFG_DOMAIN,
    CFG_PARCELS,
    CFG_OUTLETS,
    CFG_BMP_EFFICIENCY,
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_INPUT_DISTRIBUTIONS,
    CFG_OUTPUTS,
)

_LOAD_GENERATION_PATH_KEYS: tuple[str, ...] = ()


def _resolve_config_path_value(value: Any, base_dir: Path) -> Any:
    """Resolve one configured filesystem path relative to a config directory.

    Scalar paths and sequences of paths are supported. Non-path values are
    returned unchanged so this helper can be applied only to known path keys.
    """
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())
    if isinstance(value, list):
        return [_resolve_config_path_value(item, base_dir) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_config_path_value(item, base_dir) for item in value)
    return value


def resolve_config_paths(cfg: Dict[str, Any], config_path: Union[str, Path]) -> Dict[str, Any]:
    """Resolve configured file locations against the YAML file's directory.

    This removes any dependency on the process working directory. Resolution
    happens once in the parent process, before inputs are loaded and before
    parallel scenario workers are launched. Absolute paths are preserved.

    Parameters
    ----------
    cfg : dict[str, Any]
        Normalized model configuration.
    config_path : str or pathlib.Path
        Path to the YAML file that supplied ``cfg``.

    Returns
    -------
    dict[str, Any]
        The same mapping object with known path-valued entries resolved to
        absolute paths.
    """
    config_file = Path(config_path).expanduser().resolve()
    base_dir = config_file.parent

    for key in _CONFIG_PATH_KEYS:
        value = ci_get(cfg, key)
        if value is not None:
            cfg[key] = _resolve_config_path_value(value, base_dir)

    load_generation = ci_get(cfg, CFG_LOAD_GENERATION)
    if isinstance(load_generation, dict):
        normalized_load_generation = {str(key).lower(): value for key, value in load_generation.items()}
        load_generation.clear()
        load_generation.update(normalized_load_generation)
        for key in _LOAD_GENERATION_PATH_KEYS:
            if key in load_generation and load_generation[key] is not None:
                load_generation[key] = _resolve_config_path_value(
                    load_generation[key], base_dir
                )

    return cfg


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configuration keys and apply centralized defaults in place.

    Parameters
    ----------
    cfg : dict[str, Any]
        Raw configuration mapping, typically returned by
        :func:`src.io_utils.read_config`.

    Returns
    -------
    dict[str, Any]
        The same mapping object after top-level key normalization and default
        application.
    """
    normalized = {str(key).lower(): value for key, value in cfg.items()}
    cfg.clear()
    cfg.update(normalized)
    apply_config_defaults(cfg)
    parallel = ci_get(cfg, CFG_PARALLEL)
    if isinstance(parallel, dict):
        normalized_parallel = {str(key).lower(): value for key, value in parallel.items()}
        parallel.clear()
        parallel.update(normalized_parallel)
        if "n_jobs" not in parallel or parallel["n_jobs"] is None:
            parallel["n_jobs"] = 1
    return cfg


def _load_parameter_stats_table(
    source: Any,
    label: str,
    logger: Any,
    distribution_catalog: Optional[pd.DataFrame] = None,
    parcel_ids: Optional[Sequence[str]] = None,
) -> Optional[pd.DataFrame]:
    """Load a parcel parameter statistics table from a frame or CSV path(s)."""
    if source is None:
        return None
    from .plet_rusle import canonical_parameter_name
    df = _load_table_source(source, [COL_PID, "parameter"], label, logger)
    df = _normalize_identifier_columns(df, [COL_PID], label)
    if parcel_ids is not None:
        _validate_explicit_references(
            df[COL_PID].tolist(), parcel_ids, label=label, allow_wildcard=True
        )
    df["parameter"] = df["parameter"].map(canonical_parameter_name)
    validate_unique_rows(df, [COL_PID, "parameter"], label)
    df = resolve_distribution_references(df, distribution_catalog, label)
    validate_stats_rows(df, label)
    return df


def _load_plet_parameter_table(
    source: Any,
    parcel_ids: Sequence[str],
    logger: Any,
    distribution_catalog: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Load PLET numeric parameters and fixed land-cover/HSG classifications."""
    if source is None:
        return None
    from .plet_rusle import canonical_parameter_name, PLET_CLASSIFICATION_PARAMETERS
    df = _load_table_source(source, [COL_PID, "parameter"], LOAD_PLET_INPUTS, logger)
    df = _normalize_identifier_columns(df, [COL_PID], LOAD_PLET_INPUTS)
    _validate_explicit_references(
        df[COL_PID].tolist(), parcel_ids, label=LOAD_PLET_INPUTS, allow_wildcard=True
    )
    df["parameter"] = df["parameter"].map(canonical_parameter_name)
    validate_unique_rows(df, [COL_PID, "parameter"], LOAD_PLET_INPUTS)

    categorical_mask = df["parameter"].isin(PLET_CLASSIFICATION_PARAMETERS)
    categorical_rows = df.loc[categorical_mask].copy()
    if DISTRIBUTION_ID in categorical_rows.columns:
        bad = categorical_rows[DISTRIBUTION_ID].notna() & categorical_rows[DISTRIBUTION_ID].astype(str).str.strip().ne("")
        if bad.any():
            raise ValueError("PLET land_cover and hsg are classifications and must use fixed value, not distribution_id")
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
        numeric_rows = resolve_distribution_references(
            numeric_rows, distribution_catalog, LOAD_PLET_INPUTS
        )
        validate_stats_rows(numeric_rows, LOAD_PLET_INPUTS)
    df = pd.concat([categorical_rows, numeric_rows], axis=0).sort_index()
    return validate_plet_input_table(df, parcel_ids)


def _load_plet_hydrology_lookup(
    path: Any,
    logger: Any,
    distribution_catalog: Optional[pd.DataFrame] = None,
    required_pairs: Optional[Sequence[Tuple[str, str]]] = None,
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
    distribution_catalog : Optional[pd.DataFrame]
        Reusable distribution catalog used to resolve referenced statistics.

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
    from .plet_rusle import (
        PLET_HSG_VALUES,
        PLET_LAND_COVERS,
        canonical_parameter_name,
        normalize_plet_hsg,
        normalize_plet_land_cover,
    )
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

    table = resolve_distribution_references(
        table, distribution_catalog, _PLET_HYDROLOGY_LABEL
    )
    validate_stats_rows(table, _PLET_HYDROLOGY_LABEL)
    validate_unique_rows(
        table, ["land_cover", "hsg", "parameter"], _PLET_HYDROLOGY_LABEL
    )

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
    distribution_catalog: Optional[pd.DataFrame] = None,
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
    df = _normalize_identifier_columns(df, [COL_PID], LOAD_CONCENTRATIONS)
    if parcel_ids is not None:
        _validate_explicit_references(
            df[COL_PID].tolist(), parcel_ids, label=LOAD_CONCENTRATIONS, allow_wildcard=True
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
    numeric_df = resolve_distribution_references(
        numeric_df, distribution_catalog, LOAD_CONCENTRATIONS
    )
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


def _load_pollutants(cfg: Dict[str, Any]) -> List[str]:
    """Load pollutant names from configuration.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.

    Returns
    -------
    list[str]
        Normalized pollutant names.
    """
    pols = ci_get(cfg, CFG_POLLUTANTS)
    if isinstance(pols, str):
        pols = [pols]
    if not pols:
        raise ValueError(f"At least one {CFG_POLLUTANTS} value must be specified")
    return [normalize_pollutant_label(p) for p in pols]


def _load_cps(cfg: Dict[str, Any]) -> List[int]:
    """Load CPS codes while rejecting booleans, fractions, and non-finite values."""
    cps = ci_get(cfg, CFG_CPS)
    if isinstance(cps, (int, float, np.integer, np.floating)) and not isinstance(cps, (bool, np.bool_)):
        cps = [cps]
    if not isinstance(cps, (list, tuple)) or not cps:
        raise ValueError("At least one cps code must be specified")
    result: List[int] = []
    for index, value in enumerate(cps):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"cps[{index}] must be a finite integer, not a boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cps[{index}]={value!r} must be a finite integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"cps[{index}]={value!r} must be a finite integer")
        result.append(int(numeric))
    if len(set(result)) != len(result):
        raise ValueError("cps contains duplicate codes")
    return result


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
    domain = read_geodataframe(domain_path)
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
        parcels = read_geodataframe(parcels_path, layer=GPKG_PARCELS_LAYER)
    except Exception as exc:
        raise ValueError(
            f"Parcels GeoPackage must contain spatial layer '{GPKG_PARCELS_LAYER}': {parcels_path}"
        ) from exc
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if COL_PID not in parcels.columns:
        raise ValueError("Parcels layer must include a 'pid' column")
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
    """Load optional normalized parcel-to-parcel upstream edges from ``parcels.gpkg``."""
    package = ci_get(cfg, CFG_PARCELS)
    try:
        return _read_gpkg_input_table(
            package,
            GPKG_PARCEL_UP_TABLE,
            [COL_PID, COL_PID_UP],
            GPKG_PARCEL_UP_TABLE,
            logger,
        )
    except MissingInputTableError:
        logger.verbose(
            "parcel_up table not present; treating all parcels as having no upstream parcels"
        )
        return pd.DataFrame(columns=[COL_PID, COL_PID_UP])

def _build_parcel_up_map(
    upstream_rows: pd.DataFrame,
    parcel_ids: Sequence[str],
    source_parcel_ids: Optional[Sequence[str]] = None,
    logger: Any = None,
) -> Dict[str, List[str]]:
    """Build the modeled parcel graph, filtering relationships clipped by the domain.

    References to IDs that exist in the source parcel layer but fall outside the
    modeled domain are omitted. References to IDs that never existed in the
    source layer remain hard errors, which distinguishes clipping from typos.
    """
    ordered_pids = [str(pid).strip() for pid in parcel_ids]
    valid_pids = set(ordered_pids)
    source_pids = (
        {str(pid).strip() for pid in source_parcel_ids}
        if source_parcel_ids is not None
        else set(valid_pids)
    )
    parcel_up_map: Dict[str, List[str]] = {pid: [] for pid in ordered_pids}
    seen_by_pid = {pid: set() for pid in ordered_pids}
    unknown: set[str] = set()
    clipped_edges = 0

    def normalize_pid(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
            text = str(int(value))
        return text

    for row_idx, row in upstream_rows.iterrows():
        pid = normalize_pid(row[COL_PID])
        pid_up = normalize_pid(row[COL_PID_UP])
        if not pid or not pid_up:
            raise ValueError(
                f"{GPKG_PARCEL_UP_TABLE} row {row_idx} must contain nonblank pid and pid_up"
            )
        if pid == "*" or pid_up == "*" or "," in pid or "," in pid_up:
            raise ValueError(
                f"{GPKG_PARCEL_UP_TABLE} uses normalized one-edge-per-row relationships; "
                "wildcards and comma-separated IDs are not supported"
            )
        if pid not in source_pids:
            unknown.add(pid)
            continue
        if pid_up not in source_pids:
            unknown.add(pid_up)
            continue
        if pid not in valid_pids or pid_up not in valid_pids:
            clipped_edges += 1
            continue
        if pid_up not in seen_by_pid[pid]:
            parcel_up_map[pid].append(pid_up)
            seen_by_pid[pid].add(pid_up)

    if unknown:
        values = sorted(unknown)
        raise ValueError(
            "parcel_up references parcel IDs not found in the source parcels layer: "
            f"{values[:10]}"
        )
    if clipped_edges and logger is not None:
        logger.verbose(
            f"Filtered {clipped_edges} parcel_up relationship(s) involving parcels outside the modeled domain"
        )
    return parcel_up_map

def _load_parcel_outlets(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load normalized parcel-to-outlet edges from ``parcels.gpkg``."""
    return _read_gpkg_input_table(
        ci_get(cfg, CFG_PARCELS),
        GPKG_PARCEL_OUTLETS_TABLE,
        [COL_PID, COL_OID],
        GPKG_PARCEL_OUTLETS_TABLE,
        logger,
    )

def _load_parcel_selection(cfg: Dict[str, Any], parcels: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Load optional parcel selection weights from ``input_selection_weight``."""
    if parcels.empty:
        raise ValueError("No parcels available for selection")
    table = _load_fixed_numeric_variable_table(
        cfg, GPKG_INPUT_SELECTION_WEIGHT, [COL_PID], logger,
        required=False, domain=NONNEGATIVE_DOMAIN,
    )
    parcel_ids = parcels[COL_PID].astype(str).tolist()
    if table is None or table.empty:
        return pd.DataFrame({
            COL_PID: parcel_ids,
            COL_PROBABILITY: np.full(len(parcels), 1.0 / len(parcels)),
        })
    table = table.copy()
    table[COL_PID] = table[COL_PID].astype(str).str.strip()
    default_rows = table[table[COL_PID] == "*"]
    if len(default_rows) > 1:
        raise ValueError(f"{GPKG_INPUT_SELECTION_WEIGHT} may contain at most one pid='*' row")
    default_value = None if default_rows.empty else float(default_rows.iloc[0]["value"])
    exact = {
        str(row[COL_PID]): float(row["value"])
        for _, row in table[table[COL_PID] != "*"].iterrows()
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
                "supply the parcel explicitly or add pid='*'"
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

def _load_delivery_ratios(cfg: Dict[str, Any], logger: Any) -> Optional[pd.DataFrame]:
    """Load the four parcel-to-outlet delivery variables from dedicated tables."""
    base = _load_parcel_outlets(cfg, logger)[[COL_PID, COL_OID]].copy()
    base[COL_PID] = base[COL_PID].astype(str).str.strip()
    base[COL_OID] = base[COL_OID].astype(str).str.strip()
    result = base.copy()
    any_supplied = False
    valid_pairs = set(zip(base[COL_PID], base[COL_OID]))

    for column, table_name in GPKG_DELIVERY_RATIO_TABLES.items():
        table = _load_fixed_numeric_variable_table(
            cfg, table_name, [COL_PID, COL_OID], logger,
            required=False, domain=FRACTION_DOMAIN,
        )
        if table is None or table.empty:
            result[column] = 1.0
            continue
        any_supplied = True
        table = table.copy()
        table[COL_PID] = table[COL_PID].astype(str).str.strip()
        table[COL_OID] = table[COL_OID].astype(str).str.strip()
        defaults = table[(table[COL_PID] == "*") & (table[COL_OID] == "*")]
        if len(defaults) > 1:
            raise ValueError(f"{table_name} may contain at most one (*, *) default row")
        default = 1.0 if defaults.empty else float(defaults.iloc[0]["value"])
        exact_rows = table[~((table[COL_PID] == "*") & (table[COL_OID] == "*"))]
        bad_wildcards = exact_rows[(exact_rows[COL_PID] == "*") | (exact_rows[COL_OID] == "*")]
        if not bad_wildcards.empty:
            raise ValueError(
                f"{table_name} supports either exact pid/oid rows or one (*, *) default row"
            )
        exact = {
            (str(row[COL_PID]), str(row[COL_OID])): float(row["value"])
            for _, row in exact_rows.iterrows()
        }
        unknown = sorted(set(exact) - valid_pairs)
        if unknown:
            raise ValueError(
                f"{table_name} references parcel/outlet relationships not present in parcel_outlets: {unknown[:10]}"
            )
        result[column] = [exact.get(pair, default) for pair in zip(result[COL_PID], result[COL_OID])]

    if not any_supplied:
        logger.verbose("No delivery-ratio input tables present; using neutral defaults")
    return result.reset_index(drop=True)

def _efficiency_stat_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that can define an efficiency distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        BMP efficiency input table.

    Returns
    -------
    list[str]
        Statistic columns present in the table.
    """
    named_stats = {
        "value",
        "mean",
        "average",
        "avg",
        "sd",
        "std",
        "min",
        "minimum",
        "max",
        "maximum",
        "p0",
        "p100",
    }
    return [
        col
        for col in df.columns
        if str(col).lower() in named_stats
        or (
            str(col).lower().startswith("p")
            and str(col).lower()[1:].isdigit()
        )
    ]


def _plet_numeric_spec_columns(df: pd.DataFrame) -> List[str]:
    """Return standardized numeric-specification columns present in a BMP-efficiency table.

    Parameters
    ----------
    df : pandas.DataFrame
        BMP-efficiency input table.

    Returns
    -------
    list[str]
        Numeric-specification columns present in the table.
    """
    candidates = [
        "value",
        "distribution_id",
        "mean",
        "sd",
        "min",
        "p05",
        "p50",
        "p95",
        "max",
    ]
    return [col for col in candidates if col in df.columns]


def _complete_bmp_efficiency_coverage(
    df: pd.DataFrame,
    cps: Sequence[int],
    pollutants: Sequence[str],
    logger: Any,
) -> pd.DataFrame:
    """Legacy three-path completion used by the public loader API.

    Surface is required. Missing shallow/deep subsurface values are completed
    as fixed zero distributions with verbose logging. Production mode-specific
    validation bypasses this compatibility layer.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : Sequence[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        BMP efficiency table with complete legacy pathway coverage.

    Raises
    ------
    ValueError
        If required surface-efficiency coverage is missing for a configured CPS/pollutant combination.
    """
    completed = df.copy()
    completed[COL_CPS] = completed[COL_CPS].astype(int)
    if COL_PATHWAY not in completed.columns:
        completed[COL_PATHWAY] = PATHWAY_VALUES[0]
        logger.verbose(
            "bmp_efficiency has no pathway column; treating supplied "
            "CPS x pollutant efficiencies as surface efficiencies"
        )

    configured_cps = [int(cps_code) for cps_code in cps]
    configured_pollutants = [str(pollutant) for pollutant in pollutants]
    surface_pathway = PATHWAY_VALUES[0]
    subsurface_pathways = PATHWAY_VALUES[1:]
    missing_surface: List[Tuple[int, str]] = []
    for cps_code in configured_cps:
        for pollutant in configured_pollutants:
            mask = (
                (completed[COL_CPS] == cps_code)
                & (completed[COL_POLLUTANT] == pollutant)
                & (completed[COL_PATHWAY] == surface_pathway)
            )
            if not mask.any():
                missing_surface.append((cps_code, pollutant))
    if missing_surface:
        details = ", ".join(
            f"cps={cps_code}, pollutant={pollutant}"
            for cps_code, pollutant in missing_surface
        )
        raise ValueError(
            "bmp_efficiency is missing required surface efficiency coverage "
            f"for configured CPS x pollutant combinations: {details}"
        )

    stat_columns = _efficiency_stat_columns(completed)
    added_rows: List[pd.Series] = []
    for cps_code in configured_cps:
        for pollutant in configured_pollutants:
            pair_mask = (
                (completed[COL_CPS] == cps_code)
                & (completed[COL_POLLUTANT] == pollutant)
            )
            surface_row = completed[
                pair_mask & (completed[COL_PATHWAY] == surface_pathway)
            ].iloc[0]
            for pathway in subsurface_pathways:
                pathway_mask = pair_mask & (completed[COL_PATHWAY] == pathway)
                if pathway_mask.any():
                    row_index = completed[pathway_mask].index[0]
                    has_values = any(
                        not pd.isna(completed.at[row_index, column])
                        for column in stat_columns
                    )
                    if has_values:
                        continue
                    for column in stat_columns:
                        completed.at[row_index, column] = 0.0
                else:
                    default_row = surface_row.copy()
                    default_row[COL_PATHWAY] = pathway
                    for column in stat_columns:
                        default_row[column] = 0.0
                    added_rows.append(default_row)
                logger.verbose(
                    "No bmp_efficiency value specified for "
                    f"cps={cps_code}, pollutant={pollutant}, pathway='{pathway}'; "
                    "assuming efficiency=0"
                )

    if added_rows:
        completed = pd.concat(
            [completed, pd.DataFrame(added_rows, columns=completed.columns)],
            ignore_index=True,
        )

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)
    pathway_order = {pathway: idx for idx, pathway in enumerate(PATHWAY_VALUES)}
    completed["_pathway_order"] = completed[COL_PATHWAY].map(pathway_order)
    completed = completed.sort_values(
        [COL_CPS, COL_POLLUTANT, "_pathway_order"], kind="stable"
    ).drop(columns="_pathway_order")
    return completed.reset_index(drop=True)


def _complete_plet_bmp_efficiency_coverage(
    df: pd.DataFrame, cps: Sequence[int], pollutants: Sequence[str], logger: Any
) -> pd.DataFrame:
    """Require PLET surface efficiencies and default missing subsurface to zero.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : Sequence[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        BMP efficiency table with required PLET pathway coverage.

    Raises
    ------
    ValueError
        If PLET/RUSLE surface-efficiency coverage is missing for a configured CPS/pollutant combination.
    """
    completed = df.copy()
    completed[COL_CPS] = completed[COL_CPS].astype(int)

    if COL_PATHWAY not in completed.columns:
        completed[COL_PATHWAY] = "surface"
        logger.verbose(
            "plet_rusle bmp_efficiency has no pathway column; treating supplied "
            "CPS x pollutant rows as surface efficiencies"
        )
    else:
        completed[COL_PATHWAY] = completed[COL_PATHWAY].astype(str).str.strip().str.lower()
        unexpected = sorted(
            set(completed[COL_PATHWAY].astype(str)) - set(PLET_PATHWAY_VALUES)
        )
        if unexpected:
            logger.warning(
                "plet_rusle recognizes only pathway labels 'surface' and "
                f"'subsurface'. Ignoring unexpected bmp_efficiency pathway labels: {unexpected}. "
                "If no correctly labeled subsurface efficiency remains for a CPS/pollutant, "
                "subsurface efficiency will be assumed to be 0."
            )
        completed = completed[completed[COL_PATHWAY].isin(PLET_PATHWAY_VALUES)].copy()

    validate_unique_rows(
        completed, [COL_CPS, COL_POLLUTANT, COL_PATHWAY], CFG_BMP_EFFICIENCY
    )

    numeric_cols = _plet_numeric_spec_columns(completed)
    added_rows: List[pd.Series] = []
    missing_surface: List[Tuple[int, str]] = []

    for cps_code in [int(x) for x in cps]:
        for pollutant in [str(x) for x in pollutants]:
            pair = (completed[COL_CPS] == cps_code) & (completed[COL_POLLUTANT] == pollutant)
            surf = completed[pair & (completed[COL_PATHWAY] == "surface")]
            if surf.empty:
                missing_surface.append((cps_code, pollutant))
                continue

            sub = completed[pair & (completed[COL_PATHWAY] == "subsurface")]
            sub_has_stats = False
            if not sub.empty:
                row = sub.iloc[0]
                sub_has_stats = any(not pd.isna(row.get(c)) for c in numeric_cols)

            if not sub_has_stats:
                template = surf.iloc[0].copy()
                template[COL_PATHWAY] = "subsurface"
                for c in numeric_cols:
                    template[c] = pd.NA
                if "value" in numeric_cols:
                    template["value"] = 0.0
                if COL_UNIT in template.index:
                    template[COL_UNIT] = "fraction"
                if "notes" in template.index:
                    prior = template["notes"]
                    auto_note = (
                        "Auto-added in plet_rusle: missing subsurface efficiency defaults to 0"
                    )
                    template["notes"] = (
                        auto_note
                        if pd.isna(prior) or str(prior).strip() == ""
                        else f"{prior}; {auto_note}"
                    )
                if not sub.empty:
                    completed = completed.drop(index=sub.index)
                added_rows.append(template)
                logger.warning(
                    "plet_rusle: no correctly labeled subsurface BMP efficiency was "
                    f"defined for cps={cps_code}, pollutant={pollutant}; assuming efficiency=0"
                )

    if missing_surface:
        details = ", ".join(f"cps={c}, pollutant={p}" for c, p in missing_surface)
        raise ValueError(
            "plet_rusle requires a surface bmp_efficiency for every configured "
            f"CPS x pollutant combination; missing: {details}"
        )

    if added_rows:
        completed = pd.concat([completed, pd.DataFrame(added_rows, columns=completed.columns)], ignore_index=True)

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)

    order = {"surface": 0, "subsurface": 1}
    completed["_pathway_order"] = completed[COL_PATHWAY].map(order)
    return completed.sort_values(
        [COL_CPS, COL_POLLUTANT, "_pathway_order"], kind="stable"
    ).drop(columns="_pathway_order").reset_index(drop=True)


def _load_bmp_efficiency(
    cfg: Dict[str, Any],
    cps: List[int],
    pollutants: List[str],
    logger: Any,
    *,
    complete_legacy: bool = True,
    distribution_catalog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load and normalize BMP effectiveness inputs.

    ``complete_legacy=True`` preserves the public three-path loader behavior
    used by existing callers/tests. The main model loader passes ``False`` and
    then performs mode-specific validation for PLET/RUSLE or statistical mode.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Normalized model configuration mapping.
    cps : List[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : List[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.
    complete_legacy : bool
        Whether to complete legacy three-pathway BMP efficiency coverage.
    distribution_catalog : Optional[pd.DataFrame]
        Reusable distribution catalog used to resolve referenced statistics.

    Returns
    -------
    pd.DataFrame
        Normalized BMP efficiency table.

    Raises
    ------
    ValueError
        If no BMP-efficiency records remain for the configured CPS codes and pollutants.
    """
    df = _merge_csvs(ci_get(cfg, CFG_BMP_EFFICIENCY), [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    df = _normalize_cps_column(df, CFG_BMP_EFFICIENCY)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pathway_column(df, CFG_BMP_EFFICIENCY, logger)
    df = resolve_distribution_references(df, distribution_catalog, CFG_BMP_EFFICIENCY)
    df = df[df[COL_CPS].isin(cps) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")

    if complete_legacy:
        return _complete_bmp_efficiency_coverage(df, cps, pollutants, logger)
    return df


def _load_bmp_cost(cfg: Dict[str, Any], cps: List[int], logger: Any, distribution_catalog: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Optionally load BMP cost inputs.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    cps : list[int]
        BMP CPS codes to retain.
    logger : Any
        Logger used for progress and warning messages.
    distribution_catalog : Optional[pd.DataFrame]
        Reusable distribution catalog used to resolve referenced statistics.

    Returns
    -------
    pandas.DataFrame or None
        BMP cost table filtered to the requested BMPs, or ``None`` when no
        usable cost table is configured.
    """
    path = ci_get(cfg, CFG_BMP_COST)
    if path is None:
        return None
    df = _merge_csvs(path, [COL_CPS, COL_UNIT], CFG_BMP_COST, logger)
    df = _normalize_cps_column(df, CFG_BMP_COST)
    df = resolve_distribution_references(df, distribution_catalog, CFG_BMP_COST)
    validate_stats_table(df, CFG_BMP_COST)
    df = df[df[COL_CPS].isin(cps)].copy()
    if df.empty:
        logger.warning("bmp_cost has no records for specified cps; proceeding without costing")
        return None
    return df


def _expand_pollutant_load_rate_defaults(
    df: pd.DataFrame,
    parcel_ids: Sequence[str],
    pollutants: Sequence[str],
) -> pd.DataFrame:
    """Expand ``pid='*'`` load-rate defaults while preserving exact overrides.

    This lets large statistical-mode applications define one distribution for
    many or all parcels and add only the parcel-specific exceptions. Exact
    parcel rows override wildcard rows for the same pollutant/pathway.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    parcel_ids : Sequence[str]
        Parcel identifiers in model order.
    pollutants : Sequence[str]
        Pollutant names in model order.

    Returns
    -------
    pd.DataFrame
        Load-rate table with wildcard parcel defaults expanded.
    """
    out = df.copy()
    out[COL_PID] = out[COL_PID].astype(str)
    valid_pids = {str(pid) for pid in parcel_ids}
    out = out[
        out[COL_PID].isin(valid_pids | {"*"})
        & out[COL_POLLUTANT].isin(list(pollutants))
    ].copy()
    if out.empty or not (out[COL_PID] == "*").any():
        return out[out[COL_PID].isin(valid_pids)].reset_index(drop=True)

    explicit = COL_PATHWAY in out.columns
    keys = [COL_PID, COL_POLLUTANT] + ([COL_PATHWAY] if explicit else [])
    validate_unique_rows(out, keys, CFG_POLLUTANT_LOAD_RATE)
    pathways: List[Optional[str]] = (
        list(dict.fromkeys(out[COL_PATHWAY].astype(str).tolist()))
        if explicit else [None]
    )

    defaults: Dict[Tuple[str, Optional[str]], pd.Series] = {}
    exact: Dict[Tuple[str, str, Optional[str]], pd.Series] = {}
    for _, row in out.iterrows():
        path = str(row[COL_PATHWAY]) if explicit else None
        pollutant = str(row[COL_POLLUTANT])
        pid = str(row[COL_PID])
        if pid == "*":
            defaults[(pollutant, path)] = row
        else:
            exact[(pid, pollutant, path)] = row

    expanded: List[pd.Series] = []
    for pid in map(str, parcel_ids):
        for pollutant in map(str, pollutants):
            for path in pathways:
                row = exact.get((pid, pollutant, path))
                if row is None:
                    row = defaults.get((pollutant, path))
                if row is None:
                    continue
                copied = row.copy()
                copied[COL_PID] = pid
                expanded.append(copied)
    if not expanded:
        return out.iloc[0:0].copy()
    return pd.DataFrame(expanded).reset_index(drop=True)


def _load_pollutant_load_rate(
    cfg: Dict[str, Any],
    parcels: pd.DataFrame,
    pollutants: List[str],
    logger: Any,
    distribution_catalog: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load statistical parcel pollutant load rates from ``parcels.gpkg``."""
    df = _read_gpkg_input_table(
        ci_get(cfg, CFG_PARCELS),
        GPKG_INPUT_POLLUTANT_LOAD_RATE,
        [COL_PID, COL_POLLUTANT],
        CFG_POLLUTANT_LOAD_RATE,
        logger,
    )
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_POLLUTANT_LOAD_RATE, logger)
    df = _normalize_pathway_column(df, CFG_POLLUTANT_LOAD_RATE, logger)
    df = resolve_distribution_references(df, distribution_catalog, CFG_POLLUTANT_LOAD_RATE)
    validate_stats_table(df, CFG_POLLUTANT_LOAD_RATE)
    df = _normalize_identifier_columns(df, [COL_PID], CFG_POLLUTANT_LOAD_RATE)
    parcel_ids = parcels[COL_PID].astype(str).tolist()
    _validate_explicit_references(
        df[COL_PID].tolist(), parcel_ids, label=CFG_POLLUTANT_LOAD_RATE, allow_wildcard=True
    )
    df = _expand_pollutant_load_rate_defaults(df, parcel_ids, pollutants)
    if df.empty:
        raise ValueError(f"{GPKG_INPUT_POLLUTANT_LOAD_RATE} has no records for specified parcels+pollutants")
    validate_stats_rows(df, CFG_POLLUTANT_LOAD_RATE)
    return df

def _resolve_aggregate_pathway_fractions(
    cfg: Dict[str, Any], load_generation: Dict[str, Any], pathways: Sequence[str]
) -> Dict[str, float]:
    """Resolve fractions used to split one sampled aggregate parcel load rate.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Normalized model configuration mapping.
    load_generation : Dict[str, Any]
        Load-generation configuration mapping.
    pathways : Sequence[str]
        Pollutant transport pathway names.

    Returns
    -------
    Dict[str, float]
        Normalized pathway fractions keyed by pathway name.

    Raises
    ------
    ValueError
        If pathway fractions are malformed, reference unknown pathways, fall
        outside ``[0, 1]``, or do not sum to one.
    """
    pathways = list(pathways)
    if len(pathways) == 1:
        return {pathways[0]: 1.0}
    raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS)
    if raw is None:
        raw = load_generation.get(CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS)
    fractions: Dict[str, float] = {}
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError(f"{CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS} must be a mapping")
        fractions = {_normalize_pathway_label(k): float(v) for k, v in raw.items()}
    else:
        surf_raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE)
        shallow_raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW)
        if surf_raw is not None:
            fractions["surface"] = float(surf_raw)
        if shallow_raw is not None:
            fractions["shallow subsurface"] = float(shallow_raw)
        if shallow_raw is not None and surf_raw is None and "surface" in pathways:
            fractions["surface"] = 1.0 - float(shallow_raw)
        elif surf_raw is not None and shallow_raw is not None and "deep subsurface" in pathways:
            fractions["deep subsurface"] = 1.0 - float(surf_raw) - float(shallow_raw)
    unknown = set(fractions) - set(pathways)
    if unknown:
        raise ValueError(
            f"Pathway fractions refer to pathways not defined by bmp_efficiency: {sorted(unknown)}"
        )
    if not fractions:
        raise ValueError(
            "Statistical mode uses one aggregate pollutant_load_rate per parcel but multiple BMP pathways. "
            f"Define {CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS}, e.g. {{'shallow subsurface': 0.2, 'surface': 0.8}}."
        )
    for path in pathways:
        fractions.setdefault(path, 0.0)
    vals = np.asarray(list(fractions.values()), dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("pollutant load rate pathway fractions must be finite")
    if (vals < 0.0).any() or (vals > 1.0).any():
        raise ValueError("pollutant load rate pathway fractions must each be in [0,1]")
    total = float(vals.sum())
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError(f"pollutant load rate pathway fractions must sum to 1.0; got {total:.12g}")
    return {path: float(fractions[path]) for path in pathways}


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
                COL_PID: key[0],
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


def load_and_validate_all(cfg: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    """Load, validate, and assemble all scenario inputs.

    Parameters
    ----------
    cfg : dict[str, Any]
        Scenario configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    dict[str, Any]
        Data bundle containing the validated inputs and derived lookup
        structures required by the model.

    Raises
    ------
    ValueError
        If configuration values are invalid or required inputs are missing.
    FileNotFoundError
        If a configured input file does not exist.
    """
    normalize_config(cfg)
    validate_config(cfg)
    logger.info("Loading and validating input datasets")
    with log_scope(logger=logger):
        domain = _load_domain(cfg, logger)
        parcels = _load_parcels(cfg, domain, logger)

        up = _load_parcel_graph(cfg, logger)
        out = _load_parcel_outlets(cfg, logger)
        sel = _load_parcel_selection(cfg, parcels, logger)

        parcel_ids = parcels[COL_PID].astype(str).tolist()
        source_parcel_ids = list(parcels.attrs.get("source_pid_universe", parcel_ids))
        parcel_up_map = _build_parcel_up_map(
            up, parcel_ids, source_parcel_ids=source_parcel_ids, logger=logger
        )

        out = out.copy()
        out[COL_PID] = out[COL_PID].astype(str).str.strip()
        out[COL_OID] = out[COL_OID].astype(str).str.strip()
        validate_unique_rows(out, [COL_PID, COL_OID], GPKG_PARCEL_OUTLETS_TABLE)
        unknown_out_pids = sorted(set(out[COL_PID]) - set(source_parcel_ids))
        if unknown_out_pids:
            raise ValueError(
                "parcel_outlets references parcel IDs not found in the source parcels layer: "
                f"{unknown_out_pids[:10]}"
            )
        clipped_out_rows = ~out[COL_PID].isin(parcel_ids)
        if clipped_out_rows.any():
            logger.verbose(
                f"Filtered {int(clipped_out_rows.sum())} parcel_outlets relationship(s) for parcels outside the modeled domain"
            )
            out = out.loc[~clipped_out_rows].copy()
        parcel_out_map: Dict[str, List[str]] = {pid: [] for pid in parcel_ids}
        for row in out.itertuples(index=False):
            pid = str(getattr(row, COL_PID))
            oid = str(getattr(row, COL_OID))
            parcel_out_map[pid].append(oid)

        pollutants = _load_pollutants(cfg)
        cps = _load_cps(cfg)

        load_generation = ci_get(cfg, CFG_LOAD_GENERATION) or {}
        if not isinstance(load_generation, dict):
            raise ValueError("load_generation must be a mapping")
        load_generation = {str(k).lower(): v for k, v in load_generation.items()}
        if "mode" not in load_generation or load_generation["mode"] is None:
            load_generation["mode"] = LOAD_MODE_STATISTICAL
        load_mode = str(load_generation["mode"]).strip().lower()
        if load_mode not in {LOAD_MODE_STATISTICAL, LOAD_MODE_PLET_RUSLE}:
            raise ValueError(f"Unsupported load_generation mode: {load_mode}")
        load_generation["mode"] = load_mode
        if "pathway_mode" in load_generation:
            raise ValueError(
                "load_generation.pathway_mode has been removed; "
                "plet_rusle mode always derives pathway loads from PLET/RUSLE inputs"
            )
        _validate_input_package_schema(cfg, load_mode, logger)

        distribution_catalog = load_distribution_catalog(
            ci_get(cfg, CFG_INPUT_DISTRIBUTIONS), logger
        )

        outlet_loc = _load_outlet_loc(cfg, domain, logger)
        valid_oids = set(outlet_loc[COL_OID].astype(str))
        referenced_oids = {oid for values in parcel_out_map.values() for oid in values}
        unknown_oids = sorted(referenced_oids - valid_oids)
        if unknown_oids:
            raise ValueError(
                "parcel_outlets references outlet IDs not found in outlets layer: "
                f"{unknown_oids[:10]}"
            )
        outlet_target = _load_optional_outlet_stats(cfg, COL_TARGET, logger)
        outlet_mean = _load_optional_outlet_stats(cfg, COL_MEAN, logger)
        for label, table in (("outlet_stats.target", outlet_target), ("outlet_stats.mean", outlet_mean)):
            if table is not None:
                _validate_explicit_references(
                    table[COL_OID].tolist(), valid_oids, label=label, allow_wildcard=False
                )

        supplied_legacy_groundwater_keys = (
            LOAD_GROUNDWATER_LOADS in load_generation
            or LOAD_TREAT_GROUNDWATER_WITH_BMPS in load_generation
        )
        if LOAD_GROUNDWATER_LOADS not in load_generation or load_generation[LOAD_GROUNDWATER_LOADS] is None:
            load_generation[LOAD_GROUNDWATER_LOADS] = False
        groundwater_loads = bool(load_generation[LOAD_GROUNDWATER_LOADS])
        if LOAD_TREAT_GROUNDWATER_WITH_BMPS not in load_generation or load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] is None:
            load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] = False
        treat_groundwater_with_bmps = bool(load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS])
        load_generation[LOAD_GROUNDWATER_LOADS] = groundwater_loads
        load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] = treat_groundwater_with_bmps
        if load_mode == LOAD_MODE_PLET_RUSLE and supplied_legacy_groundwater_keys:
            logger.verbose(
                "plet_rusle now always estimates lookup-derived subsurface loads; "
                "groundwater_loads/treat_groundwater_with_bmps do not alter pathway generation. "
                "Subsurface BMP treatment is controlled by the subsurface efficiency."
            )

        if ci_get(cfg, CFG_BMP_EFFICIENCY) is None:
            raise ValueError("bmp_efficiency is required")
        bmp_eff = _load_bmp_efficiency(
            cfg, cps, pollutants, logger, complete_legacy=False,
            distribution_catalog=distribution_catalog,
        )
        bmp_cost = _load_bmp_cost(cfg, cps, logger, distribution_catalog)

        plet_inputs = None
        rusle_inputs = None
        pollutant_concentrations = None
        groundwater_concentrations = None
        if load_mode == LOAD_MODE_PLET_RUSLE:
            package_parameters = _assemble_parcel_parameter_source(cfg, logger)
            if package_parameters is None:
                raise ValueError(
                    "plet_rusle mode requires dedicated input_* parameter tables in parcels.gpkg"
                )
            from .plet_rusle import canonical_parameter_name
            canonical_names = package_parameters["parameter"].map(canonical_parameter_name)
            rusle_mask = canonical_names.isin(set(RUSLE_PARAMETER_NAMES))
            plet_source = package_parameters.loc[~rusle_mask].copy()
            rusle_rows = package_parameters.loc[rusle_mask].copy()
            rusle_source = None if rusle_rows.empty else rusle_rows

            plet_inputs = _load_plet_parameter_table(
                plet_source,
                sel[COL_PID].astype(str).tolist(),
                logger,
                distribution_catalog,
            )
            if plet_inputs is None:
                raise ValueError(
                    "plet_rusle mode requires dedicated PLET input_* tables in parcels.gpkg"
                )
            required_hydrology_pairs = _effective_plet_classification_pairs(
                plet_inputs, parcel_ids
            )
            plet_hydrology_lookup = _load_plet_hydrology_lookup(
                _assemble_plet_hydrology_source(cfg, logger),
                logger,
                distribution_catalog,
                required_pairs=required_hydrology_pairs,
            )
            load_generation["_hydrology_lookup_table"] = plet_hydrology_lookup
            rusle_inputs = _load_parameter_stats_table(
                rusle_source, LOAD_RUSLE_INPUTS, logger, distribution_catalog,
                parcel_ids=parcel_ids,
            )
            unified_concentrations = _load_pollutant_concentrations(
                _assemble_plet_concentration_source(cfg, logger),
                pollutants,
                logger,
                distribution_catalog,
                parcel_ids=parcel_ids,
            )
            pollutant_concentrations, groundwater_concentrations = _split_plet_concentrations(
                unified_concentrations
            )
            # Validate required coverage before optional defaults are appended.
            # This prevents missing required inputs from silently becoming
            # plausible-looking zero/default model values.
            validate_plet_runtime_inputs(
                plet_inputs,
                rusle_inputs,
                pollutant_concentrations,
                groundwater_concentrations,
                parcel_ids,
                pollutants,
            )
            plet_inputs = _append_parameter_defaults(plet_inputs, pollutants)
            pollutant_load_rate = None
            pathways = list(PLET_PATHWAY_VALUES)
            pollutant_load_rate_is_aggregate = False
            pollutant_load_rate_pathway_fractions: Dict[str, float] = {}
            bmp_eff = _complete_plet_bmp_efficiency_coverage(bmp_eff, cps, pollutants, logger)
        else:
            pollutant_load_rate = _load_pollutant_load_rate(
                cfg, parcels, pollutants, logger, distribution_catalog
            )
            load_rate_pathways, pollutant_load_rate_is_aggregate = validate_statistical_load_rates(
                pollutant_load_rate, parcels, pollutants
            )
            if pollutant_load_rate_is_aggregate:
                if COL_PATHWAY in bmp_eff.columns:
                    pathways = list(dict.fromkeys(bmp_eff[COL_PATHWAY].astype(str).tolist()))
                else:
                    pathways = ["surface"]
                pollutant_load_rate_pathway_fractions = _resolve_aggregate_pathway_fractions(
                    cfg, load_generation, pathways
                )
            else:
                pathways = load_rate_pathways
                pollutant_load_rate_pathway_fractions = {}
            bmp_eff = validate_statistical_efficiency_coverage(
                bmp_eff, cps, pollutants, pathways
            )

        delivery_ratios = _load_delivery_ratios(cfg, logger)
        delivery_ratios = _complete_delivery_ratio_defaults(delivery_ratios, parcel_out_map)

        avg_area_ha = float(parcels["area_ha"].mean())
        avg_perim_m = float(parcels["perim_m"].mean())

        logger.info("Input validation complete; assembling data payload")

    return dict(
        parcels=parcels,
        parcel_p=sel,
        parcel_up_map=parcel_up_map,
        parcel_out_map=parcel_out_map,
        pollutants=pollutants,
        cps=cps,
        outlet_loc=outlet_loc,
        outlet_target=outlet_target,
        outlet_mean=outlet_mean,
        bmp_eff=bmp_eff,
        bmp_cost=bmp_cost,
        pollutant_load_rate=pollutant_load_rate,
        delivery_ratios=delivery_ratios,
        load_generation=load_generation,
        plet_inputs=plet_inputs,
        rusle_inputs=rusle_inputs,
        pollutant_concentrations=pollutant_concentrations,
        groundwater_concentrations=groundwater_concentrations,
        pathways=pathways,
        pollutant_load_rate_pathway_fractions=pollutant_load_rate_pathway_fractions,
        pollutant_load_rate_is_aggregate=pollutant_load_rate_is_aggregate,
        bmp_limit_n=ci_get(cfg, CFG_BMP_LIMIT_N),
        bmp_limit_usd=ci_get(cfg, CFG_BMP_LIMIT_USD),
        n_scenarios=int(ci_get(cfg, CFG_N_SCENARIOS)),
        random_seed=ci_get(cfg, CFG_RANDOM_SEED),
        avg_area_ha=avg_area_ha,
        avg_perim_m=avg_perim_m,
        parallel=ci_get(cfg, CFG_PARALLEL),
    )


def format_input_validation_report(data: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    """Return a concise human-readable report for ``--validate-only`` runs."""
    parcels = data.get("parcels")
    outlets = data.get("outlet_loc")
    plet_inputs = data.get("plet_inputs")
    package = ci_get(cfg, CFG_PARCELS)
    input_tables: List[str] = []
    if package is not None and Path(package).exists():
        input_tables = sorted(
            name for name in list_geopackage_tables(package) if name.startswith("input_")
        )

    wildcard_rows = 0
    override_rows = 0
    defaults_applied = 0
    if isinstance(plet_inputs, pd.DataFrame) and not plet_inputs.empty:
        pid_values = plet_inputs[COL_PID].astype(str)
        wildcard_rows = int((pid_values == "*").sum())
        override_rows = int((pid_values != "*").sum())
        if "_default_applied" in plet_inputs.columns:
            defaults_applied = int(
                plet_inputs["_default_applied"].fillna(False).astype(bool).sum()
            )

    load_generation = data.get("load_generation") or {}
    mode = str(load_generation.get("mode", "statistical"))
    crs_text = "unknown"
    if isinstance(parcels, gpd.GeoDataFrame) and parcels.crs is not None:
        crs_text = parcels.crs.to_string()
    parcel_count = len(parcels) if parcels is not None else 0
    outlet_count = len(outlets) if outlets is not None else 0

    lines = [
        "INPUT VALIDATION",
        "----------------",
        f"Mode:                     {mode}",
        f"Parcels:                  {parcel_count}",
        f"Analysis CRS:             {crs_text} (metric)",
        f"Outlets:                  {outlet_count}",
        f"Pollutants:               {', '.join(map(str, data.get('pollutants', [])))}",
        f"CPS codes:                {', '.join(map(str, data.get('cps', [])))}",
        f"Recognized input tables:  {len(input_tables)}",
        f"Wildcard parameter rows:  {wildcard_rows}",
        f"Parcel overrides:         {override_rows}",
        f"Optional defaults added:  {defaults_applied}",
        "Unknown input tables:     0",
        "Unknown parcel/outlet IDs: 0",
        "Duplicate logical keys:   0",
        "",
        "VALID",
    ]
    return "\n".join(lines)
