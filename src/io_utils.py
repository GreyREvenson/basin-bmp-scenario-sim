"""Read, validate, and normalize model input data.

This module loads the scenario's CSV and geospatial inputs, validates required
columns and values, normalizes naming conventions, and assembles the data
bundle consumed by the simulation model.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
from collections import defaultdict
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from .constants import (
    CFG_BMP_COST,
    CFG_BMP_EFFICIENCY,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_CPS,
    CFG_DELIVERY_RATIOS,
    CFG_DOMAIN,
    CFG_N_SCENARIOS,
    CFG_OUTLET_LOC,
    CFG_OUTLET_MEAN,
    CFG_OUTLET_TARGET,
    CFG_PARALLEL,
    CFG_PARCEL_OUT,
    CFG_PARCEL_P,
    CFG_PARCEL_UP,
    CFG_PARCELS,
    CFG_POLLUTANT_YIELD,
    CFG_POLLUTANTS,
    CFG_RANDOM_SEED,
    CFG_LOAD_GENERATION,
    LOAD_MODE_STATISTICAL,
    LOAD_MODE_PLET_RUSLE,
    LOAD_PLET_INPUTS,
    LOAD_RUSLE_INPUTS,
    LOAD_CONCENTRATIONS,
    LOAD_GROUNDWATER_CONCENTRATIONS,
    LOAD_PATHWAY_MODE,
    LOAD_PATHWAY_MODE_FIXED,
    LOAD_PATHWAY_MODE_DERIVED,
    LOAD_GROUNDWATER_LOADS,
    LOAD_TREAT_GROUNDWATER_WITH_BMPS,
    COL_AREA_HA,
    COL_AREA_M2,
    COL_CPS,
    COL_MEAN,
    COL_MAX,
    COL_MIN,
    COL_OID,
    COL_OIDS,
    COL_PERIM_M,
    COL_PID,
    COL_PID_UP,
    COL_POLLUTANT,
    COL_PROBABILITY,
    COL_SD,
    COL_TARGET,
    COL_UNIT,
    COL_PATHWAY,
)
from .utils import ci_get, normalize_columns, normalize_pollutant_label
from .logging_utils import log_scope



def _write_parquet_atomic(df: pd.DataFrame, path: Path, *, logger: logging.Logger) -> None:
    """Write a parquet file atomically.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to write.
    path : pathlib.Path
        Destination parquet path.
    logger : logging.Logger
        Logger retained for interface consistency.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If no parquet engine is available.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    except (ImportError, ModuleNotFoundError) as ex:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            "Parquet output is required but no parquet engine is installed. "
            "Install pyarrow or fastparquet in the active environment."
        ) from ex


def _flatten_plot_records(
    merged: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
) -> pd.DataFrame:
    """Convert plot records into a normalized trajectory table.

    Parameters
    ----------
    merged : dict[tuple[str, str, str, str], list[tuple[int, float, float]]]
        Plot records keyed by pollutant, outlet, x-axis, and y-axis.

    Returns
    -------
    pandas.DataFrame
        Normalized trajectory table sorted by scenario and axis fields.
    """
    rows: List[Dict[str, Any]] = []
    step_counters: Dict[Tuple[int, str, str, str, str], int] = defaultdict(int)
    for (pol, oid, xax, yax), points in merged.items():
        for sid, xval, yval in points:
            counter_key = (int(sid), str(pol), str(oid), str(xax), str(yax))
            step_counters[counter_key] += 1
            rows.append(
                {
                    "scenario": int(sid),
                    "pollutant": str(pol),
                    "oid": str(oid),
                    "x_axis": str(xax),
                    "y_axis": str(yax),
                    "step": int(step_counters[counter_key]),
                    "x_value": float(xval),
                    "y_value": float(yval),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["scenario", "pollutant", "oid", "x_axis", "y_axis", "step", "x_value", "y_value"]
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["scenario", "pollutant", "oid", "x_axis", "y_axis", "step"]).reset_index(drop=True)


def _require_cols(df: pd.DataFrame, required: Sequence[str], label: str, logger: Any) -> None:
    """Validate that a dataframe contains required columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to validate.
    required : sequence of str
        Columns that must be present.
    label : str
        Human-readable table name used in error messages.
    logger : Any
        Logger object retained for interface consistency.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


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
        df = pd.read_csv(p)
        df = normalize_columns(df)
        _require_cols(df, required_cols, f"{label} ({p})", logger)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    dedup_subset = list(required_cols)
    if COL_PATHWAY in out.columns and COL_PATHWAY not in dedup_subset:
        dedup_subset.append(COL_PATHWAY)

    dup = out.duplicated(subset=dedup_subset, keep=False)
    if dup.any():
        logger.warning(f"Duplicate rows detected in {label}; keeping first occurrence")
        out = out.drop_duplicates(subset=dedup_subset, keep="first")
    return out


def _ensure_projected(gdf: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Ensure a geospatial dataframe uses a projected CRS.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input geometry table.
    logger : Any
        Logger used to report reprojection activity.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame in a projected coordinate reference system.
    """
    if gdf.crs is None or not gdf.crs.is_projected:
        est = gdf.estimate_utm_crs()
        logger.info(f"Reprojecting to projected CRS: {est}")
        return gdf.to_crs(est)
    return gdf


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
    except Exception as ex:  # pylint: disable=broad-except
        raise ValueError(f"Failed to normalize pollutant labels in {label}: {ex}") from ex
    return df


def _normalize_pathway_column(df: pd.DataFrame, label: str, logger: Any) -> pd.DataFrame:
    """Normalize and validate pathway labels when present.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    label : str
        Dataset label used in error messages.
    logger : Any
        Logger retained for interface consistency.

    Returns
    -------
    pandas.DataFrame
        Dataframe with normalized pathway labels.

    Raises
    ------
    ValueError
        If any pathway label is not recognized.
    """
    if COL_PATHWAY not in df.columns:
        return df
    df[COL_PATHWAY] = df[COL_PATHWAY].astype(str).str.strip().str.lower()
    alias = {
        "shallow_subsurface": "shallow subsurface",
        "deep_subsurface": "deep subsurface",
        "surface_flow": "surface",
    }
    df[COL_PATHWAY] = df[COL_PATHWAY].map(lambda x: alias.get(x, x))
    allowed = {"surface", "shallow subsurface", "deep subsurface"}
    bad = sorted(set(df[COL_PATHWAY]) - allowed)
    if bad:
        raise ValueError(f"{label} pathway contains invalid values: {bad}; expected one of {sorted(allowed)}")
    return df


def _validate_stats_table(df: pd.DataFrame, label: str) -> None:
    """Validate that a table exposes sampling statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to validate.
    label : str
        Human-readable dataset name used in errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the table does not contain fixed values, summary statistics, or
        percentile columns.
    """
    cols = set(df.columns)
    ok = (
        ({"mean", "sd"} <= cols)
        or ({"min", "max"} <= cols)
        or ("value" in cols)
        or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    )
    if not ok:
        raise ValueError(f"{label} must provide value, mean/sd, min/max, or percentiles")


def _validate_stats_rows(df: pd.DataFrame, label: str) -> None:
    """Validate that each row contains usable sampling statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to validate.
    label : str
        Human-readable dataset name used in errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any row lacks a fixed value or a complete set of statistics.
    """
    stat_cols = {"value", "mean", "sd", "std", "min", "max", "minimum", "maximum", "p0", "p100"}
    stat_cols.update(c for c in df.columns if str(c).startswith("p") and str(c)[1:].isdigit())
    if not stat_cols.intersection(df.columns):
        raise ValueError(f"{label} must provide value, mean/sd, min/max, or percentiles")
    for idx, row in df.iterrows():
        supplied = [c for c in stat_cols if c in df.columns and not pd.isna(row.get(c))]
        if not supplied:
            raise ValueError(f"{label} row {idx} has no value or distribution statistics")
        if "value" not in supplied:
            row_cols = set(supplied)
            valid = ("mean" in row_cols and ({"sd", "std"} & row_cols)) or ({"min", "max"} <= row_cols) or ({"minimum", "maximum"} <= row_cols) or ({"p0", "p100"} <= row_cols)
            if not valid:
                raise ValueError(
                    f"{label} row {idx} needs a fixed value, mean/sd, or complete lower/upper bounds"
                )


def _load_parameter_stats_table(path: Any, label: str, logger: Any) -> Optional[pd.DataFrame]:
    """Load a parcel parameter statistics table.

    Parameters
    ----------
    path : Any
        CSV file path or sequence of paths.
    label : str
        Dataset label used in logs and errors.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame or None
        Loaded parameter statistics table, or ``None`` when no path is
        provided.
    """
    if path is None:
        return None
    df = _merge_csvs(path, [COL_PID, "parameter"], label, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df["parameter"] = df["parameter"].astype(str).str.strip().str.lower()
    _validate_stats_rows(df, label)
    return df


def _load_pollutant_concentrations(path: Any, pollutants: List[str], logger: Any) -> Optional[pd.DataFrame]:
    """Load parcel pollutant concentration inputs.

    Parameters
    ----------
    path : Any
        CSV file path or sequence of paths.
    pollutants : list[str]
        Pollutant names to retain.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame or None
        Filtered pollutant concentration table, or ``None`` when no path is
        provided.
    """
    if path is None:
        return None
    df = _merge_csvs(path, [COL_PID, COL_POLLUTANT], LOAD_CONCENTRATIONS, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, LOAD_CONCENTRATIONS, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df = df[df[COL_POLLUTANT].isin(pollutants)].copy()
    _validate_stats_rows(df, LOAD_CONCENTRATIONS)
    return df


def _load_groundwater_concentrations(path: Any, pollutants: List[str], logger: Any) -> Optional[pd.DataFrame]:
    """Load optional parcel groundwater concentration inputs.

    Parameters
    ----------
    path : Any
        CSV file path or sequence of paths.
    pollutants : list[str]
        Pollutant names to retain.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame or None
        Filtered groundwater concentration table, or ``None`` when no path is
        provided.
    """
    if path is None:
        return None
    df = _merge_csvs(path, [COL_PID, COL_POLLUTANT], LOAD_GROUNDWATER_CONCENTRATIONS, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, LOAD_GROUNDWATER_CONCENTRATIONS, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df = df[df[COL_POLLUTANT].isin(pollutants)].copy()
    _validate_stats_rows(df, LOAD_GROUNDWATER_CONCENTRATIONS)
    return df



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
    """Load BMP CPS codes from configuration.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.

    Returns
    -------
    list[int]
        BMP CPS codes as integers.
    """
    cps = ci_get(cfg, CFG_CPS)
    if isinstance(cps, int):
        cps = [cps]
    if not cps:
        raise ValueError("At least one cps code must be specified")
    return [int(c) for c in cps]


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
    domain_path = Path(ci_get(cfg, CFG_DOMAIN))
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    domain = gpd.read_file(domain_path)
    domain = _ensure_projected(domain, logger)
    return domain.rename(columns={c: c.lower() for c in domain.columns})


def _load_parcels(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load parcels, clip them to the domain, and compute geometry metrics.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    domain : geopandas.GeoDataFrame
        Domain boundary used for clipping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    geopandas.GeoDataFrame
        Parcel dataframe with area and perimeter columns.

    Raises
    ------
    ValueError
        If the parcel file is missing required columns, becomes empty after
        clipping, or contains duplicate parcel IDs.
    """
    parcels_path = Path(ci_get(cfg, CFG_PARCELS))
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path}")
    parcels = gpd.read_file(parcels_path)
    parcels = _ensure_projected(parcels, logger)
    parcels = gpd.overlay(parcels, domain, how="intersection")
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if "pid" not in parcels.columns:
        raise ValueError("Parcels must include a 'pid' column")
    if parcels.empty:
        raise ValueError("No parcels remain after clipping to the domain")
    if parcels["pid"].astype(str).duplicated().any():
        dup_pids = sorted(parcels.loc[parcels["pid"].astype(str).duplicated(), "pid"].astype(str).unique().tolist())
        raise ValueError(f"Parcel IDs must be unique after clipping; duplicates found: {dup_pids}")
    parcels["area_m2"] = parcels.geometry.area
    parcels["perim_m"] = parcels.geometry.length
    parcels["area_ha"] = parcels["area_m2"] / 10000.0
    return parcels


def _load_parcel_graph(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load parcel-to-parcel upstream relationships.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame
        Table describing which parcels flow into which others.
    """
    up_path = Path(ci_get(cfg, CFG_PARCEL_UP))
    if not up_path.exists():
        raise FileNotFoundError(f"{CFG_PARCEL_UP} not found: {up_path}")
    df = _merge_csvs(up_path, [COL_PID, COL_PID_UP], CFG_PARCEL_UP, logger)
    return df


def _build_parcel_up_map(
    upstream_rows: pd.DataFrame,
    parcel_ids: Sequence[str],
) -> Dict[str, List[str]]:
    """Build a validated mapping of parcels to upstream parcel IDs.

    Each ``pid_up`` cell may contain one ID, a comma-separated list of IDs, or
    no value. IDs are stripped of surrounding whitespace, deduplicated while
    preserving their input order, and checked against the loaded parcel set.

    Parameters
    ----------
    upstream_rows : pandas.DataFrame
        Parcel graph table containing ``pid`` and ``pid_up`` columns.
    parcel_ids : sequence of str
        Valid parcel IDs after the parcel layer has been clipped to the model
        domain.

    Returns
    -------
    dict[str, list[str]]
        Upstream parcel IDs keyed by receiving parcel ID.

    Raises
    ------
    ValueError
        If a receiving or upstream parcel ID is missing from the loaded parcel
        set, or if a graph row has a blank receiving parcel ID.
    """
    ordered_pids = [str(pid).strip() for pid in parcel_ids]
    valid_pids = set(ordered_pids)
    parcel_up_map: Dict[str, List[str]] = {pid: [] for pid in ordered_pids}
    seen_by_pid = {pid: set() for pid in ordered_pids}
    unknown_pids = set()

    def resolve_pid(value: Any) -> str:
        """Match numeric CSV values such as ``4.0`` to parcel ID ``4``."""
        pid = str(value).strip()
        if pid in valid_pids:
            return pid
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
            integer_pid = str(int(value))
            if integer_pid in valid_pids:
                return integer_pid
        return pid

    for row_idx, row in upstream_rows.iterrows():
        raw_pid = row[COL_PID]
        if pd.isna(raw_pid) or not str(raw_pid).strip():
            raise ValueError(f"{CFG_PARCEL_UP} row {row_idx} has a blank {COL_PID}")

        pid = resolve_pid(raw_pid)
        if pid not in valid_pids:
            unknown_pids.add(pid)
            continue

        raw_upstream = row[COL_PID_UP]
        if pd.isna(raw_upstream):
            continue

        if isinstance(raw_upstream, (int, float, np.integer, np.floating)):
            raw_upstream_pids = [raw_upstream]
        else:
            raw_upstream_pids = str(raw_upstream).split(",")

        for raw_upstream_pid in raw_upstream_pids:
            upstream_pid = resolve_pid(raw_upstream_pid)
            if not upstream_pid:
                continue
            if upstream_pid not in valid_pids:
                unknown_pids.add(upstream_pid)
                continue
            if upstream_pid not in seen_by_pid[pid]:
                parcel_up_map[pid].append(upstream_pid)
                seen_by_pid[pid].add(upstream_pid)

    if unknown_pids:
        unknown = sorted(unknown_pids)
        preview = unknown[:10]
        suffix = f" (and {len(unknown) - len(preview)} more)" if len(unknown) > len(preview) else ""
        raise ValueError(
            f"{CFG_PARCEL_UP} references parcel IDs not found in parcels after clipping: "
            f"{preview}{suffix}"
        )

    return parcel_up_map


def _load_parcel_outlets(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load parcel-to-outlet relationships.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame
        Table describing which outlets each parcel drains to.
    """
    out_path = Path(ci_get(cfg, CFG_PARCEL_OUT))
    if not out_path.exists():
        raise FileNotFoundError(f"{CFG_PARCEL_OUT} not found: {out_path}")
    df = _merge_csvs(out_path, [COL_PID, COL_OIDS], CFG_PARCEL_OUT, logger)
    return df


def _load_parcel_selection(cfg: Dict[str, Any], parcels: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Load or synthesize parcel selection probabilities.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    parcels : pandas.DataFrame
        Parcel table used to determine available parcel IDs.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame
        Parcel IDs and normalized selection probabilities.
    """
    if parcels.empty:
        raise ValueError("No parcels available for selection")
    p_cfg = ci_get(cfg, CFG_PARCEL_P)
    if p_cfg is not None:
        df = _merge_csvs(p_cfg, [COL_PID, COL_PROBABILITY], CFG_PARCEL_P, logger)
        probs = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")
        invalid = (~np.isfinite(probs)) | (probs < 0.0)
        if invalid.any():
            bad_rows = df.loc[invalid, [COL_PID, COL_PROBABILITY]]
            preview = bad_rows.head(5).to_dict(orient="records")
            raise ValueError(
                f"{CFG_PARCEL_P} contains invalid probability values (must be finite and >= 0). "
                f"Example rows: {preview}"
            )
        df[COL_PROBABILITY] = probs.astype(float)
        parcel_pids = set(parcels[COL_PID].astype(str))
        pid_mask = df[COL_PID].astype(str).isin(parcel_pids)
        removed = df[~pid_mask]
        df = df[pid_mask].copy()
        if not removed.empty:
            logger.warning(f"{CFG_PARCEL_P}: some PIDs not found in parcels after clipping; they were removed")
        if df.empty:
            raise ValueError(f"{CFG_PARCEL_P} has no {COL_PID}s that exist in parcels after clipping")
        if df[COL_PID].astype(str).duplicated().any():
            dup_pids = sorted(df.loc[df[COL_PID].astype(str).duplicated(), COL_PID].astype(str).unique().tolist())
            raise ValueError(f"{CFG_PARCEL_P} must contain one row per parcel; duplicates found: {dup_pids}")
        total_prob = df[COL_PROBABILITY].sum()
        if total_prob <= 0:
            raise ValueError(f"{CFG_PARCEL_P} probabilities sum to zero or negative")
        df[COL_PROBABILITY] /= total_prob
        return df[[COL_PID, COL_PROBABILITY]].copy()
    # synthesize uniform
    return pd.DataFrame({COL_PID: parcels[COL_PID].values, COL_PROBABILITY: np.full(len(parcels), 1 / len(parcels))})


def _load_outlet_loc(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load outlet locations and align them to the domain CRS.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    domain : geopandas.GeoDataFrame
        Domain boundary whose CRS is used for alignment.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    geopandas.GeoDataFrame
        Outlet location dataframe in the domain CRS.
    """
    outlet_path = Path(ci_get(cfg, CFG_OUTLET_LOC))
    if not outlet_path.exists():
        raise FileNotFoundError(f"Outlet location not found: {outlet_path}")
    outlet_loc = gpd.read_file(outlet_path).to_crs(domain.crs)
    outlet_loc = outlet_loc.rename(columns={c: c.lower() for c in outlet_loc.columns})
    _require_cols(outlet_loc, [COL_OID], CFG_OUTLET_LOC, logger)
    return outlet_loc


def _load_optional_outlet_stats(
    cfg: Dict[str, Any],
    key: str,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Optionally load an outlet summary table.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    key : str
        Configuration key for the table path.
    required_cols : sequence of str
        Columns that must be present in the table.
    label : str
        Dataset label used in logs and errors.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame or None
        Loaded table with normalized pollutant names, or ``None`` when the
        configuration key is absent.
    """
    if ci_get(cfg, key) is None:
        logger.verbose(f"Optional key {key} not provided; skipping {label}")
        return None
    df = _merge_csvs(ci_get(cfg, key), required_cols, label, logger)
    return _normalize_pollutant_column(df, COL_POLLUTANT, label, logger)


def _load_delivery_ratios(cfg: Dict[str, Any], logger: Any) -> Optional[pd.DataFrame]:
    """Optionally load parcel-to-outlet delivery ratios.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress and warning messages.

    Returns
    -------
    pandas.DataFrame or None
        Delivery ratio table, or ``None`` when not configured or missing.
    """
    dr_cfg = ci_get(cfg, CFG_DELIVERY_RATIOS)
    if dr_cfg is None:
        logger.verbose("No delivery ratios configured; using default delivery coefficients")
        return None
    dr_path = Path(dr_cfg)
    if not dr_path.exists():
        logger.warning(f"{CFG_DELIVERY_RATIOS} specified but file not found: {dr_cfg}; skipping delivery ratios")
        return None
    return _merge_csvs(
        dr_cfg,
        [COL_PID, COL_OID, "sdr_f_to_s", "sdr_s_to_o", "ndr_f_to_s", "ndr_s_to_o"],
        CFG_DELIVERY_RATIOS,
        logger,
    )


def _load_bmp_efficiency(cfg: Dict[str, Any], cps: List[int], pollutants: List[str], logger: Any) -> pd.DataFrame:
    """Load BMP effectiveness inputs.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    cps : list[int]
        BMP CPS codes to retain.
    pollutants : list[str]
        Pollutants to retain.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame
        BMP effectiveness table filtered to the requested BMPs and pollutants.
    """
    df = _merge_csvs(ci_get(cfg, CFG_BMP_EFFICIENCY), [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pathway_column(df, CFG_BMP_EFFICIENCY, logger)
    _validate_stats_table(df, CFG_BMP_EFFICIENCY)
    df = df[df[COL_CPS].astype(int).isin(cps) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")
    return df


def _load_bmp_cost(cfg: Dict[str, Any], cps: List[int], logger: Any) -> Optional[pd.DataFrame]:
    """Optionally load BMP cost inputs.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    cps : list[int]
        BMP CPS codes to retain.
    logger : Any
        Logger used for progress and warning messages.

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
    _validate_stats_table(df, CFG_BMP_COST)
    df = df[df[COL_CPS].astype(int).isin(cps)].copy()
    if df.empty:
        logger.warning("bmp_cost has no records for specified cps; proceeding without costing")
        return None
    return df


def _load_pollutant_yield(cfg: Dict[str, Any], parcels: pd.DataFrame, pollutants: List[str], logger: Any) -> pd.DataFrame:
    """Load parcel pollutant yields for non-PLET mode.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    parcels : pandas.DataFrame
        Parcel table used to filter valid IDs.
    pollutants : list[str]
        Pollutants to retain.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    pandas.DataFrame
        Parcel pollutant yield table filtered to valid parcels and pollutants.
    """
    df = _merge_csvs(ci_get(cfg, CFG_POLLUTANT_YIELD), [COL_PID, COL_POLLUTANT], CFG_POLLUTANT_YIELD, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_POLLUTANT_YIELD, logger)
    _validate_stats_table(df, CFG_POLLUTANT_YIELD)
    df = df[df[COL_PID].astype(str).isin(parcels[COL_PID].astype(str)) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("pollutant_yield has no records for specified parcels+pollutants")
    return df


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
    logger.info("Loading and validating input datasets")
    with log_scope(logger=logger):
        domain = _load_domain(cfg, logger)
        parcels = _load_parcels(cfg, domain, logger)

        up = _load_parcel_graph(cfg, logger)
        out = _load_parcel_outlets(cfg, logger)
        sel = _load_parcel_selection(cfg, parcels, logger)

        # Upstream list mapping. A pid_up cell may contain multiple
        # comma-separated IDs, matching the format written by
        # examples/utils/create_parcel_up.py.
        parcel_up_map = _build_parcel_up_map(
            up,
            parcels[COL_PID].astype(str).tolist(),
        )

        # Parcel->outlet mapping
        parcel_out_map: Dict[str, List[str]] = {}
        for pid in parcels[COL_PID].astype(str):
            oids: List[str] = []
            rows = out[out[COL_PID].astype(str) == str(pid)]
            if not rows.empty:
                for value in rows[COL_OIDS].tolist():
                    oids.extend([str(x).strip() for x in str(value).split(",") if str(x).strip()])
            parcel_out_map[str(pid)] = list(dict.fromkeys(oids))

        pollutants = _load_pollutants(cfg)
        cps = _load_cps(cfg)

        load_generation = ci_get(cfg, CFG_LOAD_GENERATION) or {}
        if not isinstance(load_generation, dict):
            raise ValueError("load_generation must be a mapping")
        load_generation = {str(k).lower(): v for k, v in load_generation.items()}
        load_mode = str(load_generation.get("mode", LOAD_MODE_STATISTICAL)).strip().lower()
        if load_mode not in {LOAD_MODE_STATISTICAL, LOAD_MODE_PLET_RUSLE}:
            raise ValueError(f"Unsupported load_generation mode: {load_mode}")
        load_generation["mode"] = load_mode

        outlet_loc = _load_outlet_loc(cfg, domain, logger)
        outlet_target = _load_optional_outlet_stats(cfg, CFG_OUTLET_TARGET, [COL_OID, COL_POLLUTANT, COL_TARGET], CFG_OUTLET_TARGET, logger)
        outlet_mean = _load_optional_outlet_stats(cfg, CFG_OUTLET_MEAN, [COL_OID, COL_POLLUTANT, COL_MEAN], CFG_OUTLET_MEAN, logger)

        pathway_mode = str(load_generation.get(LOAD_PATHWAY_MODE, LOAD_PATHWAY_MODE_FIXED)).strip().lower()
        groundwater_loads = bool(load_generation.get(LOAD_GROUNDWATER_LOADS, False))
        treat_groundwater_with_bmps = bool(load_generation.get(LOAD_TREAT_GROUNDWATER_WITH_BMPS, False))
        if pathway_mode not in {LOAD_PATHWAY_MODE_FIXED, LOAD_PATHWAY_MODE_DERIVED}:
            raise ValueError(
                "load_generation.pathway_mode must be 'fixed_fractions' or 'derive_from_plet'"
            )
        load_generation[LOAD_PATHWAY_MODE] = pathway_mode
        load_generation[LOAD_GROUNDWATER_LOADS] = groundwater_loads
        load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] = treat_groundwater_with_bmps

        if ci_get(cfg, CFG_BMP_EFFICIENCY) is None:
            raise ValueError("bmp_efficiency is required")
        bmp_eff = _load_bmp_efficiency(cfg, cps, pollutants, logger)
        bmp_cost = _load_bmp_cost(cfg, cps, logger)

        plet_inputs = None
        rusle_inputs = None
        pollutant_concentrations = None
        groundwater_concentrations = None
        if load_mode == LOAD_MODE_PLET_RUSLE:
            plet_inputs = _load_parameter_stats_table(
                load_generation.get(LOAD_PLET_INPUTS), LOAD_PLET_INPUTS, logger
            )
            if plet_inputs is None:
                raise ValueError("load_generation.plet_inputs is required for mode='plet_rusle'")
            rusle_inputs = _load_parameter_stats_table(
                load_generation.get(LOAD_RUSLE_INPUTS), LOAD_RUSLE_INPUTS, logger
            )
            pollutant_concentrations = _load_pollutant_concentrations(
                load_generation.get(LOAD_CONCENTRATIONS), pollutants, logger
            )
            groundwater_concentrations = _load_groundwater_concentrations(
                load_generation.get(LOAD_GROUNDWATER_CONCENTRATIONS), pollutants, logger
            )
            if any(p in {"TN", "TP"} for p in pollutants) and pollutant_concentrations is None:
                raise ValueError(
                    "load_generation.pollutant_concentrations is required for TN or TP in plet_rusle mode"
                )
            if groundwater_loads and any(p in {"TN", "TP"} for p in pollutants) and groundwater_concentrations is None:
                raise ValueError(
                    "load_generation.groundwater_concentrations is required when groundwater_loads is true"
                )
            pollutant_yield = None
        else:
            pollutant_yield = _load_pollutant_yield(cfg, parcels, pollutants, logger)

        delivery_ratios = _load_delivery_ratios(cfg, logger)

        # Precompute averages for selection heuristics and reporting
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
        pollutant_yield=pollutant_yield,
        delivery_ratios=delivery_ratios,
        load_generation=load_generation,
        plet_inputs=plet_inputs,
        rusle_inputs=rusle_inputs,
        pollutant_concentrations=pollutant_concentrations,
        groundwater_concentrations=groundwater_concentrations,
        bmp_limit_n=ci_get(cfg, CFG_BMP_LIMIT_N),
        bmp_limit_usd=ci_get(cfg, CFG_BMP_LIMIT_USD),
        n_scenarios=int(ci_get(cfg, CFG_N_SCENARIOS) or 1),
        random_seed=ci_get(cfg, CFG_RANDOM_SEED),
        avg_area_ha=avg_area_ha,
        avg_perim_m=avg_perim_m,
        parallel=ci_get(cfg, CFG_PARALLEL),
    )
