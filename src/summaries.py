"""Build per-scenario summary statistics from BMP records.

This module aggregates BMP placement records into scenario-level summary rows
and rollups. Pollutant-performance summaries are mass based: masses are summed
first, then dimensionless ratios are calculated from those totals. This avoids
averaging efficiencies across BMPs or future timesteps with very different
pollutant loads.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .constants import (
    BMP_CPS_NAME_MAP,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_REMOVED_PREFIX,
    OUTPUT_TREATED_PREFIX,
    OUTPUT_WETLAND_AREA,
    OUTPUT_COST_USD,
    OUTPUT_TOTAL_COST_USD,
    OUTPUT_BMP_FAILED,
)


BASELINE_MASS_PREFIX = "baseline_mass_"
TREATED_BASELINE_MASS_PREFIX = "treated_baseline_mass_"
REMOVED_MASS_PREFIX = "removed_mass_"
MASS_SUFFIX = "_kg"
TREATMENT_EXPOSURE_PREFIX = "treatment_exposure_fraction_"
REALIZED_EFFICIENCY_PREFIX = "realized_efficiency_"
OVERALL_REDUCTION_PREFIX = "overall_reduction_fraction_"


def _compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute common descriptive statistics for finite numeric values."""
    if values.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else np.nan,
        "min": float(np.min(valid)),
        "p25": float(np.percentile(valid, 25)),
        "p50": float(np.percentile(valid, 50)),
        "p75": float(np.percentile(valid, 75)),
        "max": float(np.max(valid)),
    }


def _safe_mass_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return a dimensionless mass ratio, or ``None`` if undefined.

    Ratios are intentionally not clipped. Signed BMP effectiveness is supported,
    so removed mass and the resulting realized/overall reduction ratios may be
    negative when a BMP increases load.
    """
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return numerator / denominator


def _mass_col(prefix: str, pollutant: str) -> str:
    return f"{prefix}{pollutant}{MASS_SUFFIX}"


def _finite_record_values(records: Sequence[Dict[str, Any]], column: str) -> List[float]:
    values: List[float] = []
    for rec in records:
        raw = rec.get(column)
        if raw is None:
            continue
        value = float(raw)
        if np.isfinite(value):
            values.append(value)
    return values


def _add_pollutant_mass_summary(
    summary: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    pollutants: Sequence[str],
) -> None:
    """Add mass statistics and mass-weighted performance ratios to ``summary``."""
    for pol in pollutants:
        baseline_col = _mass_col(BASELINE_MASS_PREFIX, pol)
        treated_col = _mass_col(TREATED_BASELINE_MASS_PREFIX, pol)
        removed_col = _mass_col(REMOVED_MASS_PREFIX, pol)

        baseline_vals = _finite_record_values(records, baseline_col)
        treated_vals = _finite_record_values(records, treated_col)
        removed_vals = _finite_record_values(records, removed_col)

        for column, values in (
            (baseline_col, baseline_vals),
            (treated_col, treated_vals),
            (removed_col, removed_vals),
        ):
            if values:
                stats = _compute_statistics(np.asarray(values, dtype=float))
                for stat_name, stat_val in stats.items():
                    summary[f"{column}_{stat_name}"] = stat_val
                summary[f"{column}_total"] = float(np.sum(values))

        # Preserve the established treated_/removed_ descriptive-stat columns as
        # backward-compatible aliases. Their values are masses for the current
        # one-year timestep; new code should prefer the explicit *_mass_* names.
        legacy_treated = _finite_record_values(records, f"{OUTPUT_TREATED_PREFIX}{pol}")
        legacy_removed = _finite_record_values(records, f"{OUTPUT_REMOVED_PREFIX}{pol}")
        if legacy_treated:
            for stat_name, stat_val in _compute_statistics(np.asarray(legacy_treated, dtype=float)).items():
                summary[f"treated_{pol}_{stat_name}"] = stat_val
        if legacy_removed:
            for stat_name, stat_val in _compute_statistics(np.asarray(legacy_removed, dtype=float)).items():
                summary[f"removed_{pol}_{stat_name}"] = stat_val

        baseline_total = float(np.sum(baseline_vals)) if baseline_vals else 0.0
        treated_total = float(np.sum(treated_vals)) if treated_vals else 0.0
        removed_total = float(np.sum(removed_vals)) if removed_vals else 0.0

        exposure = _safe_mass_ratio(treated_total, baseline_total)
        realized = _safe_mass_ratio(removed_total, treated_total)
        overall = _safe_mass_ratio(removed_total, baseline_total)

        summary[f"{TREATMENT_EXPOSURE_PREFIX}{pol}"] = np.nan if exposure is None else float(exposure)
        summary[f"{REALIZED_EFFICIENCY_PREFIX}{pol}"] = np.nan if realized is None else float(realized)
        summary[f"{OVERALL_REDUCTION_PREFIX}{pol}"] = np.nan if overall is None else float(overall)


class BMPSummaryCollector:
    """Collect BMP records and build per-CPS and all-CPS summary tables."""

    def __init__(self, pollutants: List[str], scenario_id: int) -> None:
        self.pollutants = pollutants
        self.scenario_id = scenario_id
        self.bmp_by_cps: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"records": [], "attributes": defaultdict(list)}
        )

    def add_bmp_record(
        self,
        bmp_record: Dict[str, Any],
        pid_baseline_yields: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add one BMP record.

        ``pid_baseline_yields`` is retained only for call-site compatibility with
        older code. Mass-based performance metrics are read directly from the BMP
        record, so a yield-rate denominator is never used to calculate efficiency.
        """
        del pid_baseline_yields
        cps = int(bmp_record["cps"])
        group = self.bmp_by_cps[cps]
        group["records"].append(bmp_record)

        if cps == 656:  # Constructed Wetland
            wetland_area = bmp_record.get(OUTPUT_WETLAND_AREA)
            catchment_ratio = bmp_record.get(OUTPUT_CATCHMENT_RATIO)
            if wetland_area is not None:
                group["attributes"]["wetland_area_ha"].append(float(wetland_area))
            if catchment_ratio is not None:
                group["attributes"]["catchment_ratio"].append(float(catchment_ratio))
        elif cps == 412:  # Grassed Waterway
            buffer_area = bmp_record.get(OUTPUT_BUFFER_AREA)
            linear_length = bmp_record.get(OUTPUT_LINEAR_LENGTH)
            if buffer_area is not None:
                group["attributes"]["buffer_area_ha"].append(float(buffer_area))
            if linear_length is not None:
                group["attributes"]["linear_length_m"].append(float(linear_length))

        failed_flag = bool(bmp_record.get(OUTPUT_BMP_FAILED, False))
        group["attributes"]["failed"].append(1 if failed_flag else 0)
        group["attributes"]["pid"].append(str(bmp_record.get("pid", "")))

    @staticmethod
    def _add_type_attributes(summary: Dict[str, Any], attrs: Dict[str, List[float]]) -> None:
        for attr_name in ("wetland_area_ha", "catchment_ratio", "buffer_area_ha", "linear_length_m"):
            values = attrs.get(attr_name, [])
            if values:
                stats = _compute_statistics(np.asarray(values, dtype=float))
                for stat_name, stat_val in stats.items():
                    summary[f"{attr_name}_{stat_name}"] = stat_val

    @staticmethod
    def _add_cost_summary(summary: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> None:
        costs = _finite_record_values(records, OUTPUT_COST_USD)
        if not costs:
            return
        stats = _compute_statistics(np.asarray(costs, dtype=float))
        for stat_name, stat_val in stats.items():
            summary[f"{OUTPUT_COST_USD}_{stat_name}"] = stat_val
        summary[OUTPUT_TOTAL_COST_USD] = float(np.sum(costs))

    def generate_summary_dataframe(self) -> pd.DataFrame:
        """Generate one summary row per BMP type used in the scenario."""
        summaries: List[Dict[str, Any]] = []
        for cps in sorted(self.bmp_by_cps.keys()):
            group = self.bmp_by_cps[cps]
            records = group["records"]
            attrs = group["attributes"]
            summary: Dict[str, Any] = {
                "scenario": self.scenario_id,
                "cps": cps,
                "cps_name": BMP_CPS_NAME_MAP.get(cps, f"CPS {cps}"),
                "bmp_count": len(records),
                "failures_count": int(np.sum(np.asarray(attrs.get("failed", []), dtype=float)))
                if "failed" in attrs
                else 0,
            }
            self._add_type_attributes(summary, attrs)
            _add_pollutant_mass_summary(summary, records, self.pollutants)
            self._add_cost_summary(summary, records)
            summaries.append(summary)
        return pd.DataFrame(summaries)

    def generate_rollup_summary(self) -> Dict[str, Any]:
        """Generate one combined summary row across all BMP types."""
        all_records: List[Dict[str, Any]] = []
        all_attrs: Dict[str, List[float]] = defaultdict(list)
        for cps in sorted(self.bmp_by_cps.keys()):
            group = self.bmp_by_cps[cps]
            all_records.extend(group["records"])
            for key, values in group["attributes"].items():
                all_attrs[key].extend(values)

        summary: Dict[str, Any] = {
            "scenario": self.scenario_id,
            "cps": 0,
            "cps_name": "All CPS",
            "bmp_count": len(all_records),
            "failures_count": int(np.sum(np.asarray(all_attrs.get("failed", []), dtype=float)))
            if "failed" in all_attrs
            else 0,
        }
        self._add_type_attributes(summary, all_attrs)
        _add_pollutant_mass_summary(summary, all_records, self.pollutants)
        self._add_cost_summary(summary, all_records)
        return summary
