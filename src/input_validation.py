"""Validation rules for model configuration and normalized input tables.

This module contains model-facing validation policy only. File deserialization
lives in :mod:`src.io_utils`, while defaults, normalization, input loading, and
assembly live in :mod:`src.input_config`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import (
    CFG_BMP_EFFICIENCY,
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BUFFER_DEPTH_FT,
    CFG_N_SCENARIOS,
    CFG_PARALLEL,
    CFG_POLLUTANT_LOAD_RATE,
    COL_CPS,
    COL_PATHWAY,
    COL_PID,
    COL_POLLUTANT,
    COL_PROBABILITY,
)
from .input_distributions import DISTRIBUTION_ID, stats_from_row
from .utils import ci_get


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str, logger: Any = None) -> None:
    """Require a normalized input table to contain the requested columns."""
    del logger
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def _nonblank(value: Any) -> bool:
    """Return whether an input cell contains a nonblank value."""
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _row_stats_raw(row: Mapping[str, Any]) -> Dict[str, float]:
    """Return normalized numeric statistics from an input row."""
    return stats_from_row(row)


def validate_numeric_distribution_rows(df: pd.DataFrame, label: str) -> None:
    """Validate the standardized numeric value/distribution contract row-by-row."""
    if df is None or df.empty:
        return
    for index, row in df.iterrows():
        distribution_id = row.get(DISTRIBUTION_ID)
        stats = _row_stats_raw(row)
        if not stats:
            suffix = f" (distribution_id={distribution_id!r})" if _nonblank(distribution_id) else ""
            raise ValueError(f"{label} row {index} has no fixed value or distribution statistics{suffix}")
        has_value = "value" in stats
        distribution_keys = set(stats) - {"value"}
        if has_value and distribution_keys:
            raise ValueError(
                f"{label} row {index} mixes fixed value with distribution statistics; "
                "use either value or a distribution, not both"
            )
        for key, value in stats.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{label} row {index} statistic {key!r} must be finite")
        if has_value:
            continue
        has_mean = "mean" in stats
        has_sd = "sd" in stats
        has_min = "min" in stats
        has_max = "max" in stats
        percentiles = sorted((q, stats[f"p{q}"]) for q in range(1, 100) if f"p{q}" in stats)
        if has_sd and float(stats["sd"]) < 0.0:
            raise ValueError(f"{label} row {index} sd must be >= 0")
        if has_min != has_max:
            raise ValueError(f"{label} row {index} must provide both min and max when either is supplied")
        if has_min and float(stats["min"]) > float(stats["max"]):
            raise ValueError(f"{label} row {index} has min > max")
        if not ((has_mean and has_sd) or (has_min and has_max)):
            raise ValueError(
                f"{label} row {index} needs value, mean+sd, or min+max (with optional percentiles)"
            )
        if percentiles and not (has_min and has_max):
            raise ValueError(f"{label} row {index} percentile distributions require min and max endpoints")
        if percentiles and (has_mean or has_sd):
            raise ValueError(
                f"{label} row {index} mixes percentile and normal-distribution statistics; "
                "use min/max + percentile columns OR mean/sd (optionally with min/max)"
            )
        if has_min and has_max:
            points = [(0, float(stats["min"]))]
            points.extend((q, float(value)) for q, value in percentiles)
            points.append((100, float(stats["max"])))
            for (p0, q0), (p1, q1) in zip(points[:-1], points[1:]):
                if q1 < q0:
                    raise ValueError(
                        f"{label} row {index} distribution is not monotonic: p{p0}={q0} > p{p1}={q1}"
                    )
            if has_mean and not (float(stats["min"]) <= float(stats["mean"]) <= float(stats["max"])):
                raise ValueError(f"{label} row {index} mean must lie between min and max")


def validate_distribution_bounds(
    df: pd.DataFrame,
    label: str,
    *,
    parameter_col: str,
    bounds: Mapping[str, Tuple[Optional[float], Optional[float]]],
) -> None:
    """Validate fixed/support statistics against parameter-specific bounds."""
    if df is None or df.empty:
        return
    for index, row in df.iterrows():
        parameter = str(row[parameter_col]).strip().lower()
        if parameter not in bounds:
            continue
        low, high = bounds[parameter]
        stats = _row_stats_raw(row)
        for name, value in stats.items():
            if name == "sd":
                continue
            numeric = float(value)
            if low is not None and numeric < low:
                raise ValueError(f"{label} row {index} {parameter}.{name}={numeric} is below minimum {low}")
            if high is not None and numeric > high:
                raise ValueError(f"{label} row {index} {parameter}.{name}={numeric} exceeds maximum {high}")


def validate_stats_table(df: pd.DataFrame, label: str) -> None:
    """Validate a table using the shared numeric input-distribution schema."""
    validate_numeric_distribution_rows(df, label)


def validate_stats_rows(df: pd.DataFrame, label: str) -> None:
    """Validate every row using the shared numeric input-distribution schema."""
    validate_numeric_distribution_rows(df, label)


def validate_distribution_catalog(catalog: pd.DataFrame) -> None:
    """Validate reusable distribution identifiers and numeric definitions."""
    require_columns(catalog, [DISTRIBUTION_ID], "input_distributions")
    raw_ids = catalog[DISTRIBUTION_ID]
    blank_ids = raw_ids.isna() | raw_ids.astype(str).str.strip().eq("")
    if blank_ids.any():
        rows = catalog.index[blank_ids].tolist()
        raise ValueError(f"input_distributions contains blank distribution_id values at rows {rows}")
    normalized_ids = raw_ids.astype(str).str.strip()
    duplicate = normalized_ids.duplicated(keep=False)
    if duplicate.any():
        ids = sorted(normalized_ids.loc[duplicate].unique().tolist())
        raise ValueError(f"input_distributions contains duplicate distribution_id values: {ids}")
    validate_numeric_distribution_rows(catalog, "input_distributions")


def validate_bmp_selection_table(df: pd.DataFrame, cps: Sequence[int]) -> None:
    """Validate a normalized explicit BMP-selection probability table."""
    required = {COL_CPS, COL_PROBABILITY}
    if not required.issubset(df.columns):
        raise ValueError(
            f"bmp selection file must contain {sorted(required)} or an accepted probability alias"
        )
    cps_numeric = pd.to_numeric(df[COL_CPS], errors="coerce")
    invalid_cps = (~np.isfinite(cps_numeric)) | (cps_numeric % 1 != 0)
    if invalid_cps.any():
        bad_rows = df.loc[invalid_cps, [COL_CPS, COL_PROBABILITY]].head(5).to_dict(orient="records")
        raise ValueError(
            "bmp selection cps values must be finite integers; "
            f"example bad rows: {bad_rows}"
        )
    configured_cps = {int(value) for value in cps}
    filtered_cps = cps_numeric.astype(int)
    duplicated = filtered_cps.duplicated(keep=False)
    if duplicated.any():
        duplicate_cps = sorted(filtered_cps.loc[duplicated].unique().tolist())
        raise ValueError(
            "bmp selection file contains duplicate probability rows for cps values: "
            f"{duplicate_cps}"
        )
    probs = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")
    invalid = (~np.isfinite(probs)) | (probs < 0.0)
    if invalid.any():
        bad_rows = df.loc[invalid, [COL_CPS, COL_PROBABILITY]].head(5).to_dict(orient="records")
        raise ValueError(
            "bmp selection probabilities must be finite and nonnegative; "
            f"example bad rows: {bad_rows}"
        )
    found_cps = set(filtered_cps.tolist())
    missing = sorted(configured_cps - found_cps)
    if missing:
        raise ValueError(f"bmp selection file is missing probability rows for cps values: {missing}")
    if float(probs.sum()) <= 0.0:
        raise ValueError("bmp_sel probabilities sum to zero or negative")


def validate_trajectory_table(df: pd.DataFrame) -> None:
    """Validate the canonical outlet-trajectory table schema and required values."""
    required = {
        "scenario", "pollutant", "oid", "x_axis", "y_axis", "step", "x_value", "y_value"
    }
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Canonical trajectory table is missing required columns: {missing}")
    for column in ("scenario", "step", "x_value", "y_value"):
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(numeric)
        if invalid.any():
            bad_rows = df.loc[invalid].head(5).to_dict(orient="records")
            raise ValueError(
                f"Canonical trajectory table column {column!r} contains non-finite values; "
                f"example bad rows: {bad_rows}"
            )
    for column in ("pollutant", "oid", "x_axis", "y_axis"):
        blank = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"Canonical trajectory table contains blank values in required column {column!r}"
            )


def _rows_for_pid(table: Optional[pd.DataFrame], pid: str) -> List[pd.Series]:
    """Resolve wildcard parameter rows plus parcel-specific overrides."""
    if table is None or table.empty:
        return []
    from .load_generation import canonical_parameter_name
    pids = table[COL_PID].astype(str)
    combined = pd.concat([table[pids == "*"], table[pids == str(pid)]], ignore_index=True)
    if combined.empty:
        return []
    combined = combined.assign(
        _canonical_parameter=combined["parameter"].map(canonical_parameter_name)
    )
    combined = combined.drop_duplicates(subset=["_canonical_parameter"], keep="last")
    return [row for _, row in combined.iterrows()]


def validate_plet_input_table(table: pd.DataFrame, parcel_ids: Sequence[str]) -> pd.DataFrame:
    """Validate and normalize required PLET classifications for every parcel."""
    from .load_generation import (
        PLET_CLASSIFICATION_PARAMETERS,
        _PLET_DERIVED_PARAMETERS,
        canonical_parameter_name,
        normalize_plet_hsg,
        normalize_plet_land_cover,
    )
    normalized = table.copy()
    raw_parameter_labels = normalized["parameter"].map(
        lambda value: str(value).strip().lower().replace("-", "_").replace(" ", "_")
    )
    removed_parameters = sorted(set(raw_parameter_labels[raw_parameter_labels.str.startswith("irrigat")]))
    if removed_parameters:
        raise ValueError(f"PLET irrigation parameters are no longer supported: {removed_parameters}")
    normalized["parameter"] = normalized["parameter"].map(canonical_parameter_name)
    supplied_derived = sorted(
        set(normalized.loc[normalized["parameter"].isin(_PLET_DERIVED_PARAMETERS), "parameter"])
    )
    if supplied_derived:
        raise ValueError(
            "PLET parcel inputs may not specify "
            f"{supplied_derived}; define cn and infiltration_fraction in the required "
            "load_generation.hydrology_lookup table instead"
        )
    classification_mask = normalized["parameter"].isin(PLET_CLASSIFICATION_PARAMETERS)
    if classification_mask.any() and "value" not in normalized.columns:
        raise ValueError("PLET land_cover and hsg classifications require a fixed value column")
    for row_index in normalized.index[classification_mask]:
        parameter = str(normalized.at[row_index, "parameter"])
        value = normalized.at[row_index, "value"]
        if pd.isna(value) or str(value).strip() == "":
            raise ValueError(f"PLET classification row {row_index} for {parameter} requires a fixed value")
        if parameter == "land_cover":
            normalized.at[row_index, "value"] = normalize_plet_land_cover(value)
        else:
            normalized.at[row_index, "value"] = normalize_plet_hsg(value)
    for pid in map(str, parcel_ids):
        effective = {canonical_parameter_name(row["parameter"]): row for row in _rows_for_pid(normalized, pid)}
        missing = [parameter for parameter in PLET_CLASSIFICATION_PARAMETERS if parameter not in effective]
        if missing:
            raise ValueError(f"PLET inputs for pid={pid} are missing required classifications: {missing}")
    return normalized


def validate_plet_runtime_inputs(
    plet_inputs: pd.DataFrame,
    rusle_inputs: Optional[pd.DataFrame],
    pollutant_concentrations: Optional[pd.DataFrame],
    groundwater_concentrations: Optional[pd.DataFrame],
    parcel_ids: Sequence[str],
    pollutants: Sequence[str],
) -> None:
    """Validate all PLET/RUSLE coverage before the simulation worker starts."""
    from .load_generation import _REQUIRED_PLET_INPUTS, _REQUIRED_RUSLE, canonical_parameter_name

    def effective_parameters(table: Optional[pd.DataFrame], pid: str) -> Dict[str, pd.Series]:
        if table is None:
            return {}
        return {canonical_parameter_name(row["parameter"]): row for row in _rows_for_pid(table, pid)}

    def has_concentration(table: Optional[pd.DataFrame], pid: str, pollutant: str) -> bool:
        if table is None or table.empty:
            return False
        pids = table[COL_PID].astype(str)
        pols = table[COL_POLLUTANT].astype(str)
        return bool((((pids == "*") | (pids == pid)) & (pols == pollutant)).any())

    for pid in map(str, parcel_ids):
        plet_effective = effective_parameters(plet_inputs, pid)
        missing_plet = [name for name in _REQUIRED_PLET_INPUTS if name not in plet_effective]
        if missing_plet:
            raise ValueError(f"PLET inputs for pid={pid} are missing required parameters: {missing_plet}")
        rusle_effective = effective_parameters(rusle_inputs, pid)
        if rusle_effective:
            missing_rusle = [name for name in _REQUIRED_RUSLE if name not in rusle_effective]
            if missing_rusle:
                raise ValueError(f"RUSLE inputs for pid={pid} are incomplete; missing: {missing_rusle}")
            if "sdr" not in rusle_effective and "watershed_area_mi2" not in rusle_effective:
                raise ValueError(f"RUSLE inputs for pid={pid} require 'sdr' or 'watershed_area_mi2'")
        for pollutant in pollutants:
            pol = str(pollutant).upper()
            if pol in {"TN", "TP"} and not has_concentration(pollutant_concentrations, pid, pol):
                raise ValueError(f"Runoff concentration for pid={pid}, pollutant={pol} is required")
            if pol != "TSS" and not has_concentration(groundwater_concentrations, pid, pol):
                raise ValueError(
                    f"Groundwater concentration for pid={pid}, pollutant={pol} is required in plet_rusle mode"
                )
            if pol == "TSS" and not rusle_effective and not has_concentration(pollutant_concentrations, pid, pol):
                raise ValueError(f"TSS for pid={pid} requires complete RUSLE inputs or a TSS concentration")


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate configuration values after defaults and normalization are applied."""
    if int(ci_get(cfg, CFG_N_SCENARIOS)) < 1:
        raise ValueError(f"{CFG_N_SCENARIOS} must be >= 1")
    if float(ci_get(cfg, CFG_BUFFER_DEPTH_FT)) <= 0.0:
        raise ValueError(f"{CFG_BUFFER_DEPTH_FT} must be > 0")
    fail_rate = float(ci_get(cfg, CFG_BMP_FAIL_RATE))
    fail_reduction = float(ci_get(cfg, CFG_BMP_FAIL_REDUCTION))
    if not 0.0 <= fail_rate <= 1.0:
        raise ValueError(f"{CFG_BMP_FAIL_RATE} must be in [0, 1]")
    if not 0.0 <= fail_reduction <= 1.0:
        raise ValueError(f"{CFG_BMP_FAIL_REDUCTION} must be in [0, 1]")
    parallel = ci_get(cfg, CFG_PARALLEL)
    if not isinstance(parallel, dict):
        raise ValueError(f"{CFG_PARALLEL} must be a mapping")
    if "n_jobs" not in parallel or parallel["n_jobs"] is None:
        raise ValueError("parallel.n_jobs must be specified after configuration normalization")
    if int(parallel["n_jobs"]) < 1:
        raise ValueError("parallel.n_jobs must be >= 1")


def validate_unique_rows(df: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    """Reject duplicate rows for a logical table key."""
    duplicate = df.duplicated(list(keys), keep=False)
    if duplicate.any():
        preview = df.loc[duplicate, list(keys)].head(10).to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate rows for {list(keys)}: {preview}")


def validate_statistical_efficiency_coverage(
    df: pd.DataFrame, cps: Sequence[int], pollutants: Sequence[str], pathways: Sequence[str]
) -> pd.DataFrame:
    """Require complete CPS x pollutant x pathway coverage in statistical mode."""
    out = df.copy()
    if COL_PATHWAY not in out.columns:
        if list(pathways) != ["surface"]:
            raise ValueError(
                "statistical mode with multiple pathways requires a pathway column in bmp_efficiency"
            )
        out[COL_PATHWAY] = "surface"
    supplied = set(out[COL_PATHWAY].astype(str))
    expected = set(pathways)
    if supplied != expected:
        raise ValueError(
            "statistical mode requires pollutant_load_rate and bmp_efficiency to use the same pathways; "
            f"expected {sorted(expected)}, found {sorted(supplied)} in bmp_efficiency"
        )
    validate_unique_rows(out, [COL_CPS, COL_POLLUTANT, COL_PATHWAY], CFG_BMP_EFFICIENCY)
    missing = []
    for cps_code in [int(value) for value in cps]:
        for pollutant in [str(value) for value in pollutants]:
            for pathway in pathways:
                mask = (
                    (out[COL_CPS].astype(int) == cps_code)
                    & (out[COL_POLLUTANT] == pollutant)
                    & (out[COL_PATHWAY] == pathway)
                )
                if not mask.any():
                    missing.append((cps_code, pollutant, pathway))
    if missing:
        preview = ", ".join(
            f"cps={cps_code}, pollutant={pollutant}, pathway={pathway}"
            for cps_code, pollutant, pathway in missing[:20]
        )
        raise ValueError(f"bmp_efficiency is missing required statistical-mode coverage: {preview}")
    validate_stats_rows(out, CFG_BMP_EFFICIENCY)
    return out.reset_index(drop=True)


def validate_statistical_load_rates(
    df: pd.DataFrame,
    parcels: pd.DataFrame,
    pollutants: Sequence[str],
) -> Tuple[List[str], bool]:
    """Validate statistical parcel-load-rate coverage and return pathways/mode."""
    parcel_ids = parcels[COL_PID].astype(str).tolist()
    explicit = COL_PATHWAY in df.columns
    pathways = list(dict.fromkeys(df[COL_PATHWAY].astype(str).tolist())) if explicit else []
    keys = [COL_PID, COL_POLLUTANT] + ([COL_PATHWAY] if explicit else [])
    validate_unique_rows(df, keys, CFG_POLLUTANT_LOAD_RATE)
    missing = []
    if explicit:
        for pid in parcel_ids:
            for pollutant in pollutants:
                for pathway in pathways:
                    mask = (
                        (df[COL_PID] == pid)
                        & (df[COL_POLLUTANT] == pollutant)
                        & (df[COL_PATHWAY] == pathway)
                    )
                    if not mask.any():
                        missing.append((pid, pollutant, pathway))
    else:
        for pid in parcel_ids:
            for pollutant in pollutants:
                mask = (df[COL_PID] == pid) & (df[COL_POLLUTANT] == pollutant)
                if not mask.any():
                    missing.append((pid, pollutant, None))
    if missing:
        preview = ", ".join(
            f"pid={pid}, pollutant={pollutant}" + (f", pathway={pathway}" if pathway else "")
            for pid, pollutant, pathway in missing[:20]
        )
        raise ValueError(f"pollutant_load_rate is missing required statistical-mode coverage: {preview}")
    return pathways, not explicit
