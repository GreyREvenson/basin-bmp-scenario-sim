"""Read input files, check them, and prepare clean data for the model."""

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
    CFG_LOAD_GENERATION,
    LOAD_MODE_STATISTICAL,
    LOAD_MODE_PLET_RUSLE,
    LOAD_PLET_INPUTS,
    LOAD_RUSLE_INPUTS,
    LOAD_CONCENTRATIONS,
    LOAD_PROCESS_MODE,
    LOAD_PROCESS_EFFECTS,
    LOAD_PROCESS_FALLBACK,
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


def _require_cols(df: pd.DataFrame, required: Sequence[str], label: str, logger: Any) -> None:
    """Stop with a clear error if a table is missing needed columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _merge_csvs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one or more CSV files and combine them into one clean table."""
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
    """Make sure map data uses a coordinate system that supports distance/area math."""
    if gdf.crs is None or not gdf.crs.is_projected:
        est = gdf.estimate_utm_crs()
        logger.info(f"Reprojecting to projected CRS: {est}")
        return gdf.to_crs(est)
    return gdf


def _normalize_pollutant_column(df: pd.DataFrame, col: str, label: str, logger: Any) -> pd.DataFrame:
    """Convert pollutant names to the standard names used by the model."""
    if col not in df.columns:
        raise ValueError(f"{label} missing required column '{col}'")
    try:
        df[col] = [normalize_pollutant_label(x) for x in df[col]]
    except Exception as ex:  # pylint: disable=broad-except
        raise ValueError(f"Failed to normalize pollutant labels in {label}: {ex}") from ex
    return df


def _normalize_pathway_column(df: pd.DataFrame, label: str, logger: Any) -> pd.DataFrame:
    """Clean and validate pathway names in a table, when present."""
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
    """Check that a table has enough number fields to sample values."""
    cols = set(df.columns)
    ok = (
        ({"mean", "sd"} <= cols)
        or ({"min", "max"} <= cols)
        or ("value" in cols)
        or any(str(c).lower().startswith("p") and str(c)[1:].isdigit() for c in cols)
    )
    if not ok:
        raise ValueError(f"{label} must provide mean/sd or min/max or percentiles")




def _validate_stats_rows(df: pd.DataFrame, label: str) -> None:
    """Check that each row has enough number fields to sample a value."""
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
    """Load a table of per-parcel parameter values or value ranges."""
    if path is None:
        return None
    df = _merge_csvs(path, [COL_PID, "parameter"], label, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df["parameter"] = df["parameter"].astype(str).str.strip().str.lower()
    _validate_stats_rows(df, label)
    return df


def _load_pollutant_concentrations(path: Any, pollutants: List[str], logger: Any) -> Optional[pd.DataFrame]:
    """Load pollutant concentration inputs by parcel."""
    if path is None:
        return None
    df = _merge_csvs(path, [COL_PID, COL_POLLUTANT], LOAD_CONCENTRATIONS, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, LOAD_CONCENTRATIONS, logger)
    df[COL_PID] = df[COL_PID].astype(str)
    df = df[df[COL_POLLUTANT].isin(pollutants)].copy()
    _validate_stats_rows(df, LOAD_CONCENTRATIONS)
    return df


def _load_process_effects(path: Any, cps: List[int], logger: Any) -> Optional[pd.DataFrame]:
    """Load optional BMP rules that change process values."""
    if path is None:
        return None
    df = _merge_csvs(path, [COL_CPS, "parameter"], LOAD_PROCESS_EFFECTS, logger)
    df[COL_CPS] = df[COL_CPS].astype(int)
    df = df[df[COL_CPS].isin(cps)].copy()
    if df.empty:
        raise ValueError("bmp_parameter_effects has no records for specified cps values")
    if "operation" not in df.columns:
        df["operation"] = "multiply"
    df["operation"] = df["operation"].astype(str).str.strip().str.lower()
    allowed = {"multiply", "scale", "add", "delta", "set", "replace", "reduce", "reduction_fraction"}
    bad = sorted(set(df["operation"]) - allowed)
    if bad:
        raise ValueError(f"bmp_parameter_effects contains unsupported operations: {bad}")
    _validate_stats_rows(df, LOAD_PROCESS_EFFECTS)
    return df


def _zero_efficiency_table(cps: List[int], pollutants: List[str]) -> pd.DataFrame:
    """Create a fallback table where every BMP efficiency is zero."""
    return pd.DataFrame(
        [{COL_CPS: int(c), COL_POLLUTANT: p, "value": 0.0} for c in cps for p in pollutants]
    )


def _load_pollutants(cfg: Dict[str, Any]) -> List[str]:
    """Read pollutant names from config and convert to standard names."""
    pols = ci_get(cfg, CFG_POLLUTANTS)
    if isinstance(pols, str):
        pols = [pols]
    if not pols:
        raise ValueError(f"At least one {CFG_POLLUTANTS} value must be specified")
    return [normalize_pollutant_label(p) for p in pols]


def _load_cps(cfg: Dict[str, Any]) -> List[int]:
    """Read BMP type codes from config and return them as integers."""
    cps = ci_get(cfg, CFG_CPS)
    if isinstance(cps, int):
        cps = [cps]
    if not cps:
        raise ValueError("At least one cps code must be specified")
    return [int(c) for c in cps]


def _load_domain(cfg: Dict[str, Any], logger: Any) -> gpd.GeoDataFrame:
    """Load the model boundary shape and make sure its map projection is usable."""
    domain_path = Path(ci_get(cfg, CFG_DOMAIN))
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    domain = gpd.read_file(domain_path)
    domain = _ensure_projected(domain, logger)
    return domain.rename(columns={c: c.lower() for c in domain.columns})


def _load_parcels(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load parcels, clip them to the boundary, and calculate size/edge length."""
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
    """Load which parcels flow into which other parcels."""
    up_path = Path(ci_get(cfg, CFG_PARCEL_UP))
    if not up_path.exists():
        raise FileNotFoundError(f"{CFG_PARCEL_UP} not found: {up_path}")
    df = _merge_csvs(up_path, [COL_PID, COL_PID_UP], CFG_PARCEL_UP, logger)
    return df


def _load_parcel_outlets(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load which outlets each parcel drains to."""
    out_path = Path(ci_get(cfg, CFG_PARCEL_OUT))
    if not out_path.exists():
        raise FileNotFoundError(f"{CFG_PARCEL_OUT} not found: {out_path}")
    df = _merge_csvs(out_path, [COL_PID, COL_OIDS], CFG_PARCEL_OUT, logger)
    return df


def _load_parcel_selection(cfg: Dict[str, Any], parcels: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Load parcel selection chances, or create equal chances if none are provided."""
    p_cfg = ci_get(cfg, CFG_PARCEL_P)
    if p_cfg is not None:
        df = _merge_csvs(p_cfg, [COL_PID, COL_PROBABILITY], CFG_PARCEL_P, logger)
        parcel_pids = set(parcels[COL_PID].astype(str))
        pid_mask = df[COL_PID].astype(str).isin(parcel_pids)
        removed = df[~pid_mask]
        df = df[pid_mask].copy()
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
    """Load outlet locations and align them to the same map projection as the domain."""
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
    """Optionally load outlet target/mean tables and clean pollutant names."""
    if ci_get(cfg, key) is None:
        logger.verbose(f"Optional key {key} not provided; skipping {label}")
        return None
    df = _merge_csvs(ci_get(cfg, key), required_cols, label, logger)
    return _normalize_pollutant_column(df, COL_POLLUTANT, label, logger)


def _load_delivery_ratios(cfg: Dict[str, Any], logger: Any) -> Optional[pd.DataFrame]:
    """Optionally load parcel-to-outlet delivery ratio values."""
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
    """Load BMP effectiveness inputs for the selected BMPs and pollutants."""
    df = _merge_csvs(ci_get(cfg, CFG_BMP_EFFICIENCY), [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pathway_column(df, CFG_BMP_EFFICIENCY, logger)
    _validate_stats_table(df, CFG_BMP_EFFICIENCY)
    df = df[df[COL_CPS].astype(int).isin(cps) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")
    return df


def _load_bmp_cost(cfg: Dict[str, Any], cps: List[int], logger: Any) -> Optional[pd.DataFrame]:
    """Optionally load BMP cost inputs for the selected BMP types."""
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
    """Load starting parcel pollutant loads used in non-PLET mode."""
    df = _merge_csvs(ci_get(cfg, CFG_POLLUTANT_YIELD), [COL_PID, COL_POLLUTANT], CFG_POLLUTANT_YIELD, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_POLLUTANT_YIELD, logger)
    _validate_stats_table(df, CFG_POLLUTANT_YIELD)
    df = df[df[COL_PID].astype(str).isin(parcels[COL_PID].astype(str)) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("pollutant_yield has no records for specified parcels+pollutants")
    return df


def load_and_validate_all(cfg: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    """Load every needed input file, clean it, validate it, and return one data bundle."""
    logger.info("Loading and validating input datasets")
    with log_scope(logger=logger):
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

        process_mode = bool(load_generation.get(LOAD_PROCESS_MODE, False))
        process_fallback = str(load_generation.get(LOAD_PROCESS_FALLBACK, "efficiency")).strip().lower()
        if process_fallback not in {"efficiency", "none"}:
            raise ValueError("process_parameter_fallback must be 'efficiency' or 'none'")
        load_generation[LOAD_PROCESS_FALLBACK] = process_fallback

        if ci_get(cfg, CFG_BMP_EFFICIENCY) is None:
            if process_mode:
                bmp_eff = _zero_efficiency_table(cps, pollutants)
            else:
                raise ValueError("bmp_efficiency is required unless process_parameter_mode is enabled")
        else:
            bmp_eff = _load_bmp_efficiency(cfg, cps, pollutants, logger)
        bmp_cost = _load_bmp_cost(cfg, cps, logger)

        plet_inputs = None
        rusle_inputs = None
        pollutant_concentrations = None
        bmp_parameter_effects = None
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
            if any(p in {"TN", "TP"} for p in pollutants) and pollutant_concentrations is None:
                raise ValueError(
                    "load_generation.pollutant_concentrations is required for TN or TP in plet_rusle mode"
                )
            pollutant_yield = None
        else:
            pollutant_yield = _load_pollutant_yield(cfg, parcels, pollutants, logger)

        if process_mode:
            if load_mode != LOAD_MODE_PLET_RUSLE:
                raise ValueError("process_parameter_mode currently requires mode='plet_rusle'")
            bmp_parameter_effects = _load_process_effects(
                load_generation.get(LOAD_PROCESS_EFFECTS), cps, logger
            )
            if bmp_parameter_effects is None:
                raise ValueError("process_parameter_mode requires load_generation.bmp_parameter_effects")

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
        bmp_parameter_effects=bmp_parameter_effects,
        bmp_limit_n=ci_get(cfg, CFG_BMP_LIMIT_N),
        bmp_limit_usd=ci_get(cfg, CFG_BMP_LIMIT_USD),
        n_scenarios=int(ci_get(cfg, CFG_N_SCENARIOS) or 1),
        random_seed=ci_get(cfg, CFG_RANDOM_SEED),
        avg_area_ha=avg_area_ha,
        avg_perim_m=avg_perim_m,
        parallel=ci_get(cfg, CFG_PARALLEL),
    )


def consolidate_transposed_summaries(outputs_dir: Path, logger) -> Path:
    """Combine all scenario summary files into one easy-to-open CSV."""
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
        """Help sort columns by scenario number and then BMP label."""
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