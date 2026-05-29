"""
I/O helpers and input validation.

Centralized filesystem and ingestion utilities:
- Reading geospatial/tabular inputs and normalizing labels
- Validations (required columns, supported stats forms)
- Ensuring projected CRS for area/perimeter calculations
- Writing cross-scenario consolidated outputs

Notes
-----
- Areas and perimeters are only meaningful under a projected CRS. Inputs are
  reprojected to a suitable UTM when necessary.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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
)
from .utils import ci_get, normalize_columns, normalize_pollutant_label


def _require_cols(df: pd.DataFrame, required: Sequence[str], label: str, logger: Any) -> None:
    """Raise if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _merge_csvs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one or multiple CSVs, normalize columns, validate and concat.

    Parameters
    ----------
    paths : str | Path | Sequence[str | Path]
        One or more CSV file paths.
    required_cols : Sequence[str]
        Columns required to exist in each input.
    label : str
        Human label for messages and errors (e.g., 'bmp_efficiency').
    logger : logging.Logger
        Logger for messages.

    Returns
    -------
    pandas.DataFrame
        Concatenated frame with normalized column labels and duplicates dropped.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames: List[pd.DataFrame] = []
    for p in paths:
        logger.debug(f"Reading {label} from {p}")
        df = pd.read_csv(p)
        df = normalize_columns(df)
        _require_cols(df, required_cols, f"{label} ({p})", logger)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(subset=required_cols, keep=False)
    if dup.any():
        logger.warning(f"Duplicate rows detected in {label}; keeping first occurrence")
        out = out.drop_duplicates(subset=required_cols, keep="first")
    return out


def _ensure_projected(gdf: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame uses a projected CRS.

    Returns either the original or a reprojected copy. Areas, perimeters, and
    distances computed later are only meaningful in a projected CRS. If the
    input CRS is missing or geographic, a suitable UTM is estimated and used.
    """
    if gdf.crs is None or not gdf.crs.is_projected:
        est = gdf.estimate_utm_crs()
        logger.info(f"Reprojecting to projected CRS: {est}")
        return gdf.to_crs(est)
    return gdf


def _normalize_pollutant_column(df: pd.DataFrame, col: str, label: str, logger: Any) -> pd.DataFrame:
    """Normalize a pollutant column to canonical labels (adds detailed error context)."""
    if col not in df.columns:
        raise ValueError(f"{label} missing required column '{col}'")
    try:
        df[col] = [normalize_pollutant_label(x) for x in df[col]]
    except Exception as ex:  # pylint: disable=broad-except
        raise ValueError(f"Failed to normalize pollutant labels in {label}: {ex}") from ex
    return df


def _validate_stats_table(df: pd.DataFrame, label: str) -> None:
    """Validate that a stats table provides mean/sd, min/max or percentile columns."""
    cols = set(df.columns)
    ok = (
        ({"mean", "sd"} <= cols)
        or ({"min", "max"} <= cols)
        or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    )
    if not ok:
        raise ValueError(f"{label} must provide mean/sd or min/max or percentiles")


def _load_pollutants(cfg: Dict[str, Any]) -> List[str]:
    """Normalize and return the list of pollutants from config."""
    pols = ci_get(cfg, CFG_POLLUTANTS)
    if isinstance(pols, str):
        pols = [pols]
    if not pols:
        raise ValueError(f"At least one {CFG_POLLUTANTS} value must be specified")
    return [normalize_pollutant_label(p) for p in pols]


def _load_cps(cfg: Dict[str, Any]) -> List[int]:
    """Return CPS list from config as ints."""
    cps = ci_get(cfg, CFG_CPS)
    if isinstance(cps, int):
        cps = [cps]
    if not cps:
        raise ValueError("At least one cps code must be specified")
    return [int(c) for c in cps]


def _load_domain(cfg: Dict[str, Any], logger: Any) -> gpd.GeoDataFrame:
    """Load domain polygon(s) and ensure a projected CRS."""
    domain_path = Path(ci_get(cfg, CFG_DOMAIN))
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    domain = gpd.read_file(domain_path)
    domain = _ensure_projected(domain, logger)
    return domain.rename(columns={c: c.lower() for c in domain.columns})


def _load_parcels(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load parcels, ensure projected CRS, clip to domain, and compute area/perimeter."""
    parcels_path = Path(ci_get(cfg, CFG_PARCELS))
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels not found: {parcels_path}")
    parcels = gpd.read_file(parcels_path)
    parcels = _ensure_projected(parcels, logger)
    parcels = gpd.overlay(parcels, domain, how="intersection")
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if "pid" not in parcels.columns:
        raise ValueError("Parcels must include a 'pid' column")
    parcels["area_m2"] = parcels.geometry.area
    parcels["perim_m"] = parcels.geometry.length
    parcels["area_ha"] = parcels["area_m2"] / 10000.0
    return parcels


def _load_parcel_graph(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load parcel adjacency (up-gradient graph)."""
    up_path = Path(ci_get(cfg, CFG_PARCEL_UP))
    if not up_path.exists():
        raise FileNotFoundError(f"{CFG_PARCEL_UP} not found: {up_path}")
    df = _merge_csvs(up_path, [COL_PID, COL_PID_UP], CFG_PARCEL_UP, logger)
    return df


def _load_parcel_outlets(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load parcel-to-outlet associations."""
    out_path = Path(ci_get(cfg, CFG_PARCEL_OUT))
    if not out_path.exists():
        raise FileNotFoundError(f"{CFG_PARCEL_OUT} not found: {out_path}")
    df = _merge_csvs(out_path, [COL_PID, COL_OIDS], CFG_PARCEL_OUT, logger)
    return df


def _load_parcel_selection(cfg: Dict[str, Any], parcels: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Load or synthesize parcel selection probabilities (normalized to sum=1)."""
    p_cfg = ci_get(cfg, CFG_PARCEL_P)
    if p_cfg is not None:
        df = _merge_csvs(p_cfg, [COL_PID, COL_PROBABILITY], CFG_PARCEL_P, logger)
        df = df[df[COL_PID].astype(str).isin(parcels[COL_PID].astype(str))].copy()
        removed = df[~df[COL_PID].astype(str).isin(parcels[COL_PID].astype(str))]
        if not removed.empty:
            logger.warning(f"{CFG_PARCEL_P}: some PIDs not found in parcels after clipping; they were removed")
        if df.empty:
            raise ValueError(f"{CFG_PARCEL_P} has no {COL_PID}s that exist in parcels after clipping")
        total_prob = df[COL_PROBABILITY].sum()
        if total_prob <= 0:
            raise ValueError(f"{CFG_PARCEL_P} probabilities sum to zero or negative")
        df[COL_PROBABILITY] /= total_prob
        return df[[COL_PID, COL_PROBABILITY]].copy()
    # synthesize uniform
    return pd.DataFrame({COL_PID: parcels[COL_PID].values, COL_PROBABILITY: np.full(len(parcels), 1 / len(parcels))})


def _load_outlet_loc(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load outlet points and project to domain CRS."""
    outlet_path = Path(ci_get(cfg, CFG_OUTLET_LOC))
    if not outlet_path.exists():
        raise FileNotFoundError(f"Outlet location not found: {outlet_path}")
    outlet_loc = gpd.read_file(outlet_path).to_crs(domain.crs)
    return outlet_loc.rename(columns={c: c.lower() for c in outlet_loc.columns})


def _load_optional_outlet_stats(
    cfg: Dict[str, Any],
    key: str,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Optionally load per-outlet stats (target or mean), normalizing pollutant labels."""
    if ci_get(cfg, key) is None:
        logger.debug(f"Optional key {key} not provided; skipping {label}")
        return None
    df = _merge_csvs(ci_get(cfg, key), required_cols, label, logger)
    return _normalize_pollutant_column(df, COL_POLLUTANT, label, logger)


def _load_delivery_ratios(cfg: Dict[str, Any], logger: Any) -> Optional[pd.DataFrame]:
    """Load optional parcel->outlet delivery ratio table."""
    dr_cfg = ci_get(cfg, CFG_DELIVERY_RATIOS)
    if dr_cfg is None:
        logger.debug("No delivery ratios configured; using default delivery coefficients")
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
    """Load BMP efficiency stats filtered to requested CPS and pollutants."""
    df = _merge_csvs(ci_get(cfg, CFG_BMP_EFFICIENCY), [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    _validate_stats_table(df, CFG_BMP_EFFICIENCY)
    df = df[df[COL_CPS].astype(int).isin(cps) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")
    return df


def _load_bmp_cost(cfg: Dict[str, Any], cps: List[int], logger: Any) -> Optional[pd.DataFrame]:
    """Load optional BMP cost stats filtered to requested CPS."""
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
    """Load baseline pollutant yield stats per parcel and pollutant."""
    df = _merge_csvs(ci_get(cfg, CFG_POLLUTANT_YIELD), [COL_PID, COL_POLLUTANT], CFG_POLLUTANT_YIELD, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_POLLUTANT_YIELD, logger)
    _validate_stats_table(df, CFG_POLLUTANT_YIELD)
    df = df[df[COL_PID].astype(str).isin(parcels[COL_PID].astype(str)) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("pollutant_yield has no records for specified parcels+pollutants")
    return df


def _assemble_delivery_coeffs(
    parcel_out: pd.DataFrame,
    delivery_ratios: Optional[pd.DataFrame],
    logger: Any,
) -> Dict[tuple[str, str], Dict[str, float]]:
    """Create a mapping {(pid, oid) -> delivery coeff dict}, defaulting to 1.0 for missing pairs."""
    coeffs: Dict[tuple[str, str], Dict[str, float]] = {}
    for _, row in parcel_out.iterrows():
        pid = str(row[COL_PID])
        for oid in str(row[COL_OIDS]).split(","):
            coeffs[(pid, str(oid))] = dict(sdr_f_to_s=1.0, sdr_s_to_o=1.0, ndr_f_to_s=1.0, ndr_s_to_o=1.0)

    if delivery_ratios is None:
        return coeffs

    for _, row in delivery_ratios.iterrows():
        pid = str(row[COL_PID])
        oid = str(row[COL_OID])
        coeffs[(pid, oid)] = dict(
            sdr_f_to_s=float(row["sdr_f_to_s"]),
            sdr_s_to_o=float(row["sdr_s_to_o"]),
            ndr_f_to_s=float(row["ndr_f_to_s"]),
            ndr_s_to_o=float(row["ndr_s_to_o"]),
        )
    return coeffs


def load_and_validate_all(cfg: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    """Load, normalize, and validate all inputs; return a data payload for Model.

    Returns
    -------
    Dict[str, Any]
        Keys
        - parcels : geopandas.GeoDataFrame
        - parcel_p : pandas.DataFrame
        - parcel_up_map : Dict[str, List[str]]
        - parcel_out_map : Dict[str, List[str]]
        - pollutants : List[str]
        - cps : List[int]
        - outlet_loc : geopandas.GeoDataFrame
        - outlet_target : Optional[pandas.DataFrame]
        - outlet_mean : Optional[pandas.DataFrame]
        - bmp_eff : pandas.DataFrame
        - bmp_cost : Optional[pandas.DataFrame]
        - pollutant_yield : pandas.DataFrame
        - delivery_ratios : Optional[pandas.DataFrame]
        - bmp_limit_n : Optional[int]
        - bmp_limit_usd : Optional[float]
        - n_scenarios : int
        - random_seed : Optional[int]
        - avg_area_ha : float
        - avg_perim_m : float
        - parallel : Optional[Dict[str, Any]]
    """
    domain = _load_domain(cfg, logger)
    parcels = _load_parcels(cfg, domain, logger)

    up = _load_parcel_graph(cfg, logger)
    out = _load_parcel_outlets(cfg, logger)
    sel = _load_parcel_selection(cfg, parcels, logger)

    # Upstream list mapping
    parcel_up_map: Dict[str, List[str]] = {}
    for pid in parcels[COL_PID].astype(str):
        ups = up[up[COL_PID].astype(str) == str(pid)][COL_PID_UP].astype(str).tolist()
        parcel_up_map[str(pid)] = ups

    # Parcel->outlet mapping
    parcel_out_map: Dict[str, List[str]] = {}
    for pid in parcels[COL_PID].astype(str):
        oids = []
        row = out[out[COL_PID].astype(str) == str(pid)]
        if not row.empty:
            oids = str(row.iloc[0][COL_OIDS]).split(",")
        parcel_out_map[str(pid)] = [str(x) for x in oids if str(x)]

    pollutants = _load_pollutants(cfg)
    cps = _load_cps(cfg)

    outlet_loc = _load_outlet_loc(cfg, domain, logger)
    outlet_target = _load_optional_outlet_stats(cfg, CFG_OUTLET_TARGET, [COL_OID, COL_POLLUTANT, COL_TARGET], CFG_OUTLET_TARGET, logger)
    outlet_mean = _load_optional_outlet_stats(cfg, CFG_OUTLET_MEAN, [COL_OID, COL_POLLUTANT, COL_MEAN], CFG_OUTLET_MEAN, logger)

    bmp_eff = _load_bmp_efficiency(cfg, cps, pollutants, logger)
    bmp_cost = _load_bmp_cost(cfg, cps, logger)
    pollutant_yield = _load_pollutant_yield(cfg, parcels, pollutants, logger)
    delivery_ratios = _load_delivery_ratios(cfg, logger)

    # Precompute averages for selection heuristics and reporting
    avg_area_ha = float(parcels["area_ha"].mean())
    avg_perim_m = float(parcels["perim_m"].mean())

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
        bmp_limit_n=ci_get(cfg, CFG_BMP_LIMIT_N),
        bmp_limit_usd=ci_get(cfg, CFG_BMP_LIMIT_USD),
        n_scenarios=int(ci_get(cfg, CFG_N_SCENARIOS) or 1),
        random_seed=ci_get(cfg, CFG_RANDOM_SEED),
        avg_area_ha=avg_area_ha,
        avg_perim_m=avg_perim_m,
        parallel=ci_get(cfg, CFG_PARALLEL),
    )


def consolidate_transposed_summaries(outputs_dir: Path, logger) -> Path:
    """Consolidate all per-scenario transposed summaries into one CSV.

    Parameters
    ----------
    outputs_dir : Path
        Root outputs directory (contains 'summaries' subfolder).
    logger : logging.Logger
        Logger for status messages.

    Returns
    -------
    Path
        Path to outputs/summaries/all_scenarios.csv.

    Notes
    -----
    - Reads: outputs/summaries/s*.csv (each with a 'field' column)
    - Writes: outputs/summaries/all_scenarios.csv
    - Merge: outer-join on 'field'. Columns are per-scenario labels like
      's1-All CPS' or 's1-Grassed Waterway(412)'. Within each scenario,
      'All CPS' is ordered first via the regex-based sort key.
    """
    outputs_dir = Path(outputs_dir)
    summaries_dir = outputs_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    out_path = summaries_dir / "all_scenarios.csv"

    files = sorted(p for p in summaries_dir.glob("s*.csv") if p.name != out_path.name)
    if not files:
        logger.info("No per-scenario summaries found to consolidate.")
        pd.DataFrame({"field": []}).to_csv(out_path, index=False)
        return out_path

    logger.info(f"Consolidating {len(files)} per-scenario summaries into {out_path}")

    combined = None
    for p in files:
        df = pd.read_csv(p)
        if "field" not in df.columns:
            logger.warning(f"Skipping {p} (no 'field' column)")
            continue
        df = df.set_index("field")
        combined = df if combined is None else combined.join(df, how="outer")

    if combined is None or combined.empty:
        logger.warning("No valid per-scenario summary data found; writing empty file.")
        pd.DataFrame({"field": []}).to_csv(out_path, index=False)
        return out_path

    def col_key(cname: str):
        m = re.match(r"s(\d+)-(.*)", str(cname))
        if not m:
            return (10**9, 1, str(cname))
        sid = int(m.group(1))
        tail = m.group(2)
        is_all = 0 if tail.strip() == "All CPS" else 1
        return (sid, is_all, tail)

    ordered_cols = sorted([c for c in combined.columns], key=col_key)
    combined = combined[ordered_cols].reset_index()
    combined.to_csv(out_path, index=False)
    logger.info(f"Wrote consolidated transposed summaries: {out_path}")
    return out_path