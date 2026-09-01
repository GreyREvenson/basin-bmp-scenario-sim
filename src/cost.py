"""BMP costing and cost-based selection helpers.

This module estimates BMP placement costs from cost tables and derives
selection probabilities that favor lower-cost BMPs when configured to do so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model
from .constants import (
    COL_CPS,
    DATA_AVG_PERIM_M,
    DATA_AVG_AREA_HA,
    DATA_BMP_COST,
    DATA_CPS,
    COL_PROBABILITY,
    COL_UNIT,
    CFG_BUFFER_DEPTH_FT,
    DEFAULT_BUFFER_DEPTH_FT,
)
from .input_distributions import stats_from_row
from .input_units import canonical_cost_unit, convert_cost_value
from .logging_utils import log_scope
# Code-level constants used ONLY for selection-time average-cost heuristics
PROB_EST_WETLAND_MAX_AREA_HA: float = 0.8
PROB_EST_BUFFER_PERIM_FRACTION: float = 0.2

FT_TO_M = 0.3048  # meters per foot


def _get_bmp_cost(
    self: "Model",
    cps: Union[int, str],
    quantity: float,
) -> float:
    """Estimate the cost of one BMP placement.
    Parameters
    ----------
    self : Model
        Active simulation model instance.
    cps : int or str
        BMP CPS code.
    quantity : float
        Realized BMP quantity used for costing, such as area or length.
    Returns
    -------
    float
        Estimated BMP placement cost in USD.
    """
    with log_scope(label=f"get_bmp_cost cps={cps}", logger=self.logger):
        self.logger.verbose("calling _get_bmp_cost")
        bmp_cost_df = self.data.get(DATA_BMP_COST)
        if bmp_cost_df is None or bmp_cost_df.empty:
            self.logger.verbose("no BMP cost table configured; returning cost=$0.0")
            return 0.0
        bmp_cost_df = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
        if bmp_cost_df.empty:
            self.logger.verbose(f"no cost entry found for cps={cps}; returning cost=$0.0")
            return 0.0
        row = bmp_cost_df.iloc[0]  # Assumes one row per CPS; validated upstream
        unit, _ = canonical_cost_unit(row[COL_UNIT])
        stats = stats_from_row(row, {COL_CPS})
        rate_value = float(self._sample_from_stats(stats, kind=None))
        self.logger.verbose(f"sampled cost rate {rate_value:.4f} for cps={cps}, unit={unit}")
        cost_total: float
        if unit == "usd/ha":
            if quantity and quantity > 0:
                area_ha = float(quantity)
            else:
                if int(cps) in (656, 657):
                    area_ha = float(min(PROB_EST_WETLAND_MAX_AREA_HA, self.data[DATA_AVG_AREA_HA]))
                else:
                    area_ha = float(self.data[DATA_AVG_AREA_HA])
            cost_total = rate_value * area_ha
        elif unit == "usd/m":
            if quantity and quantity > 0:
                # quantity represents area_ha for grassed buffers; convert to length via depth (m)
                depth_ft = float(self.cfg.get(CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT))
                depth_m = depth_ft * FT_TO_M
                area_m2 = float(quantity) * 10000.0
                length_m = area_m2 / max(depth_m, 1e-9)
            else:
                # Fallback to average-perimeter heuristic (selection-time heuristic reused)
                length_m = float(PROB_EST_BUFFER_PERIM_FRACTION * self.data[DATA_AVG_PERIM_M])
            cost_total = rate_value * length_m
        elif unit == "usd/project":
            cost_total = rate_value
        else:  # pragma: no cover - canonical_cost_unit prevents this state.
            raise ValueError(f"Unsupported BMP cost unit {unit!r} for cps={cps}")

        self.logger.verbose(
            f"computed cost for cps={cps} using rate={rate_value:.4f}, unit='{unit}', "
            f"realized_quantity={quantity:.4f} => cost={cost_total:.2f}"
        )
        return float(cost_total)


def _select_cost_rate_median(
    self: "Model",
    row: pd.Series,
    cps: Optional[Union[int, str]] = None,
) -> float:
    """Select a representative cost rate from one cost row.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    row : pandas.Series
        Cost table row.
    cps : int or str, optional
        BMP CPS code used for logging.

    Returns
    -------
    float
        Representative cost rate in the canonical rate unit.
    Raises
    ------
    ValueError
        If no representative rate can be inferred from the row.
    """
    with log_scope(label=f"select_cost_rate_median cps={cps}", logger=self.logger):
        self.logger.verbose("calling _select_cost_rate_median")
        stats = stats_from_row(row, {COL_CPS})
        if "value" in stats:
            rate_value = float(stats["value"])
        elif "p50" in stats:
            rate_value = float(stats["p50"])
        elif "median" in row.index and not pd.isna(row.get("median")):
            rate_value = convert_cost_value(row["median"], row[COL_UNIT])
        elif "mean" in stats:
            rate_value = float(stats["mean"])
        elif "min" in stats and "max" in stats:
            rate_value = (float(stats["min"]) + float(stats["max"])) / 2.0
        else:
            raise ValueError(f"Could not determine cost rate for cps={cps}")
        self.logger.verbose(f"selected representative cost rate {rate_value:.4f} for cps={cps}")
        return float(rate_value)


def _estimate_costs_for_probabilities(self: "Model") -> pd.DataFrame:
    """Estimate BMP selection probabilities from cost heuristics.
    The function computes a representative total cost for each BMP type and
    assigns probabilities inversely proportional to those costs.

    Parameters
    ----------
    self : Model
        Active simulation model instance.

    Returns
    -------
    pandas.DataFrame
        Two-column dataframe containing CPS codes and normalized selection
        probabilities.
    Raises
    ------
    ValueError
        If no usable cost information exists for probability estimation.
    """
    with log_scope(label="estimate_costs_for_probabilities", logger=self.logger):
        self.logger.verbose("calling _estimate_costs_for_probabilities")
        rows: list[Dict[str, float]] = []
        for cps in sorted(set(int(x) for x in self.data[DATA_CPS])):
            bmp_cost_df = self.data[DATA_BMP_COST]
            sub = bmp_cost_df[bmp_cost_df[COL_CPS].astype(int) == int(cps)]
            if sub.empty:
                self.logger.warning(f"no cost entry found for cps={cps}; assigning small placeholder cost for probability estimation")
                rows.append({"cps": int(cps), "est_total_cost": float(0.01)})
                continue
            row = sub.iloc[0]
            unit, _ = canonical_cost_unit(row[COL_UNIT])
            rate_value = self._select_cost_rate_median(row, cps=cps)

            if unit == "usd/ha":
                if cps in (656, 657):
                    area_ha = float(min(PROB_EST_WETLAND_MAX_AREA_HA, self.data[DATA_AVG_AREA_HA]))
                else:
                    area_ha = float(self.data[DATA_AVG_AREA_HA])
                total = rate_value * area_ha
            elif unit == "usd/m":
                length_m = float(PROB_EST_BUFFER_PERIM_FRACTION * self.data[DATA_AVG_PERIM_M])
                total = rate_value * length_m
            elif unit == "usd/project":
                total = rate_value
            else:  # pragma: no cover - canonical_cost_unit prevents this state.
                raise ValueError(f"Unsupported BMP cost unit {unit!r} for cps={cps}")

            rows.append({"cps": int(cps), "est_total_cost": float(max(total, 0.01))})
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Could not estimate costs for probability computation")

        inv = 1.0 / df["est_total_cost"].values
        probs = inv / inv.sum()
        df[COL_PROBABILITY] = probs
        self.logger.verbose(
            "Probability estimation constants: "
            f"PROB_EST_WETLAND_MAX_AREA_HA={PROB_EST_WETLAND_MAX_AREA_HA}, "
            f"PROB_EST_BUFFER_PERIM_FRACTION={PROB_EST_BUFFER_PERIM_FRACTION}"
        )
        self.logger.verbose(f"estimated probabilities: {df[[COL_CPS, COL_PROBABILITY]].to_dict(orient='records')}")
        return df[[COL_CPS, COL_PROBABILITY]]
