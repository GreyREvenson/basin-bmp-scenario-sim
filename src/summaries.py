"""Build per-scenario summary statistics from BMP records.

This module aggregates BMP placement records into scenario-level summary rows
and rollups, including counts, costs, efficiencies, and type-specific
attributes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

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


def _compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute common descriptive statistics.

    Parameters
    ----------
    values : numpy.ndarray
        Numeric values to summarize.

    Returns
    -------
    dict[str, float]
        Summary statistics including count, mean, standard deviation, minimum,
        quartiles, and maximum. Empty or all-NaN inputs yield NaN statistics.
    """
    if values.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "max": np.nan}
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return {"count": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "max": np.nan}
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


def _compute_efficiency(treated: float, baseline_yield: float) -> Optional[float]:
    """Compute the treated share of a baseline load.

    Parameters
    ----------
    treated : float
        Treated pollutant load.
    baseline_yield : float
        Baseline pollutant load before treatment.

    Returns
    -------
    float or None
        Fraction of baseline load treated, clipped to ``[0, 1]``. Returns
        ``None`` when ``baseline_yield`` is not positive.
    """
    if baseline_yield <= 0:
        return None
    return min(1.0, treated / baseline_yield)


class BMPSummaryCollector:
    """Collect BMP records and build summary tables.

    The collector groups BMP records by CPS code, retains the fields needed for
    summary statistics, and exposes methods for per-CPS and rollup summaries.
    """

    def __init__(self, pollutants: List[str], scenario_id: int) -> None:
        """Initialize an empty summary collector.

        Parameters
        ----------
        pollutants : list[str]
            Pollutants tracked in the scenario.
        scenario_id : int
            Scenario identifier.

        Returns
        -------
        None
        """
        self.pollutants = pollutants
        self.scenario_id = scenario_id
        self.bmp_by_cps: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"records": [], "attributes": defaultdict(list)}
        )

    def add_bmp_record(
        self,
        bmp_record: Dict[str, Any],
        pid_baseline_yields: Dict[str, float],
    ) -> None:
        """Add one BMP record to the collector.

        Parameters
        ----------
        bmp_record : dict[str, Any]
            BMP output row produced during scenario execution.
        pid_baseline_yields : dict[str, float]
            Baseline parcel yields keyed by pollutant.

        Returns
        -------
        None
        """
        cps = int(bmp_record["cps"])
        group = self.bmp_by_cps[cps]
        group["records"].append(bmp_record)

        # Type-specific attributes
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

        # Per-BMP failure flag for counting
        failed_flag = bool(bmp_record.get(OUTPUT_BMP_FAILED, False))
        group["attributes"]["failed"].append(1 if failed_flag else 0)

        # Parcel ID and baseline yields (for per-BMP efficiency)
        group["attributes"]["pid"].append(str(bmp_record.get("pid", "")))
        for pol in self.pollutants:
            group["attributes"][f"baseline_{pol}"].append(float(pid_baseline_yields.get(pol, 0.0)))

    def generate_summary_dataframe(self) -> pd.DataFrame:
        """Generate one summary row per BMP type.

        Returns
        -------
        pandas.DataFrame
            Summary table with one row per CPS code used in the scenario.
        """
        summaries: List[Dict[str, Any]] = []
        for cps in sorted(self.bmp_by_cps.keys()):
            group = self.bmp_by_cps[cps]
            bmp_records = group["records"]
            attrs = group["attributes"]

            summary: Dict[str, Any] = {
                "scenario": self.scenario_id,
                "cps": cps,
                "cps_name": BMP_CPS_NAME_MAP.get(cps, f"CPS {cps}"),
                "bmp_count": len(bmp_records),
                "failures_count": int(np.sum(np.asarray(attrs.get("failed", []), dtype=float))) if "failed" in attrs else 0,
            }

            # Type-specific attributes
            for attr_name in ("wetland_area_ha", "catchment_ratio", "buffer_area_ha", "linear_length_m"):
                if attr_name in attrs and len(attrs[attr_name]) > 0:
                    stats = _compute_statistics(np.asarray(attrs[attr_name], dtype=float))
                    for stat_name, stat_val in stats.items():
                        summary[f"{attr_name}_{stat_name}"] = stat_val

            # Treated/removed per pollutant
            for pol in self.pollutants:
                t_col = f"{OUTPUT_TREATED_PREFIX}{pol}"
                r_col = f"{OUTPUT_REMOVED_PREFIX}{pol}"

                t_vals = [float(rec.get(t_col)) for rec in bmp_records if rec.get(t_col) is not None]
                r_vals = [float(rec.get(r_col)) for rec in bmp_records if rec.get(r_col) is not None]

                if t_vals:
                    stats = _compute_statistics(np.asarray(t_vals, dtype=float))
                    for stat_name, stat_val in stats.items():
                        summary[f"treated_{pol}_{stat_name}"] = stat_val
                if r_vals:
                    stats = _compute_statistics(np.asarray(r_vals, dtype=float))
                    for stat_name, stat_val in stats.items():
                        summary[f"removed_{pol}_{stat_name}"] = stat_val

            # Efficiency per pollutant (treated/baseline), per BMP, then stats
            for pol in self.pollutants:
                t_col = f"{OUTPUT_TREATED_PREFIX}{pol}"
                blist = attrs.get(f"baseline_{pol}", [])
                efficiencies: List[float] = []
                for i, rec in enumerate(bmp_records):
                    treated = rec.get(t_col)
                    if treated is None:
                        continue
                    eff = _compute_efficiency(float(treated), float(blist[i] if i < len(blist) else 0.0))
                    if eff is not None:
                        efficiencies.append(eff)
                if efficiencies:
                    stats = _compute_statistics(np.asarray(efficiencies, dtype=float))
                    for stat_name, stat_val in stats.items():
                        summary[f"efficiency_{pol}_{stat_name}"] = stat_val

            # Cost per BMP (USD)
            costs = [float(rec.get(OUTPUT_COST_USD)) for rec in bmp_records if rec.get(OUTPUT_COST_USD) is not None]
            if costs:
                stats = _compute_statistics(np.asarray(costs, dtype=float))
                for stat_name, stat_val in stats.items():
                    summary[f"{OUTPUT_COST_USD}_{stat_name}"] = stat_val
                summary[OUTPUT_TOTAL_COST_USD] = float(np.sum(costs))

            summaries.append(summary)

        return pd.DataFrame(summaries)

    def generate_rollup_summary(self) -> Dict[str, Any]:
        """Generate one combined summary row across all BMP types.

        Returns
        -------
        dict[str, Any]
            Rollup summary covering all BMP records in the scenario.
        """
        all_records: List[Dict[str, Any]] = []
        all_attrs: Dict[str, List[float]] = defaultdict(list)

        for cps in sorted(self.bmp_by_cps.keys()):
            group = self.bmp_by_cps[cps]
            all_records.extend(group["records"])
            for k, vals in group["attributes"].items():
                all_attrs[k].extend(vals)

        summary: Dict[str, Any] = {
            "scenario": self.scenario_id,
            "cps": 0,
            "cps_name": "All CPS",
            "bmp_count": len(all_records),
            "failures_count": int(np.sum(np.asarray(all_attrs.get("failed", []), dtype=float))) if "failed" in all_attrs else 0,
        }

        # Type-specific attributes
        for attr_name in ("wetland_area_ha", "catchment_ratio", "buffer_area_ha", "linear_length_m"):
            vals = all_attrs.get(attr_name, [])
            if vals:
                stats = _compute_statistics(np.asarray(vals, dtype=float))
                for stat_name, stat_val in stats.items():
                    summary[f"{attr_name}_{stat_name}"] = stat_val

        # Treated/Removed across all BMPs
        for pol in self.pollutants:
            t_col = f"{OUTPUT_TREATED_PREFIX}{pol}"
            r_col = f"{OUTPUT_REMOVED_PREFIX}{pol}"
            t_vals = [float(rec.get(t_col)) for rec in all_records if rec.get(t_col) is not None]
            r_vals = [float(rec.get(r_col)) for rec in all_records if rec.get(r_col) is not None]
            if t_vals:
                stats = _compute_statistics(np.asarray(t_vals, dtype=float))
                for stat_name, stat_val in stats.items():
                    summary[f"treated_{pol}_{stat_name}"] = stat_val
            if r_vals:
                stats = _compute_statistics(np.asarray(r_vals, dtype=float))
                for stat_name, stat_val in stats.items():
                    summary[f"removed_{pol}_{stat_name}"] = stat_val

        # Efficiency across all BMPs
        for pol in self.pollutants:
            t_col = f"{OUTPUT_TREATED_PREFIX}{pol}"
            efficiencies: List[float] = []
            # iterate per CPS to align with baseline lists
            for cps in sorted(self.bmp_by_cps.keys()):
                group = self.bmp_by_cps[cps]
                recs = group["records"]
                blist = group["attributes"].get(f"baseline_{pol}", [])
                for i, rec in enumerate(recs):
                    treated = rec.get(t_col)
                    if treated is None:
                        continue
                    baseline = blist[i] if i < len(blist) else 0.0
                    eff = _compute_efficiency(float(treated), float(baseline))
                    if eff is not None:
                        efficiencies.append(eff)
            if efficiencies:
                stats = _compute_statistics(np.asarray(efficiencies, dtype=float))
                for stat_name, stat_val in stats.items():
                    summary[f"efficiency_{pol}_{stat_name}"] = stat_val

        # Cost across all BMPs
        all_costs = [float(rec.get(OUTPUT_COST_USD)) for rec in all_records if rec.get(OUTPUT_COST_USD) is not None]
        if all_costs:
            stats = _compute_statistics(np.asarray(all_costs, dtype=float))
            for stat_name, stat_val in stats.items():
                summary[f"{OUTPUT_COST_USD}_{stat_name}"] = stat_val
            summary[OUTPUT_TOTAL_COST_USD] = float(np.sum(all_costs))

        return summary