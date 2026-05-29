"""
BMP simulation utilities.

Implements stochastic selection of BMP types and parcel-level simulation of
constructed wetlands, grassed waterways/buffers, and in-field practices.

Units and conventions
---------------------
- Areas in hectares (ha), lengths in meters (m), depths in feet (ft) in config and
  converted to m.
- Parcel yields are in load per unit area (e.g., mass/ha).
- Side effects: wetland/grassed/in-field simulators mutate the yields array in place and
  populate per-BMP records with type-specific attributes and per-pollutant results.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

from .constants import (
    CFG_BUFFER_DEPTH_FT,
    CFG_BMP_SEL,
    BMP_CPS_NAME_MAP,
    COL_CPS,
    COL_PROBABILITY,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_PORTION_TREATED,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
    OUTPUT_WETLAND_AREA,
    DATA_BMP_COST,
    DATA_CPS,
    DEFAULT_BUFFER_DEPTH_FT,
)

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

FT_TO_M = 0.3048  # meters per foot


def _select_bmp_type(self: "Model") -> int:
    """Choose a BMP type code from the precomputed probability distribution."""
    idx = self.rng.choice(len(self.bmp_cps), p=self.bmp_selection_probs)
    cps = int(self.bmp_cps[idx])
    self.logger.debug(f"selected bmp {cps} ({self._get_bmp_name(cps)})")
    return cps


def _get_bmp_name(self: "Model", cps: Union[int, str]) -> str:
    """Return the human-readable name for the BMP CPS code."""
    key = int(cps)
    return BMP_CPS_NAME_MAP.get(key, f"CPS {key}")


def _sample_efficiency(self: "Model", cps: Union[int, str], pol_idx: int) -> float:
    """Sample BMP efficiency for a specific CPS code and pollutant in [0, 1]."""
    stats = self.bmp_efficiency_stats[int(cps)][pol_idx]
    eff = self._sample_from_stats(stats, kind="efficiency")
    self.logger.debug(f"selected efficiency value {eff:.2f} for pollutant={self.pollutants[pol_idx]}")
    return eff


def _simulate_wetland(
    self: "Model",
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
    cps: Union[int, str] = 656,
) -> None:
    """Simulate a constructed wetland BMP and update parcel yields.

    Parameters
    ----------
    parcel_idx : int
        Index into parcel arrays.
    eff : Sequence[float]
        Per-pollutant efficiency samples in [0, 1].
    yields : np.ndarray, shape (n_parcels, n_pollutants)
        Mutable array of parcel yields (units: load/ha). Updated in place.
    bmp_rec : Dict[str, Any]
        Per-BMP record populated with:
        - OUTPUT_WETLAND_AREA (ha)
        - OUTPUT_CATCHMENT_RATIO (dimensionless)
        - OUTPUT_IMPACTED_PIDS (comma-separated up-gradient PIDs when >1)
    bmp_outputs : Dict[str, np.ndarray]
        Aggregators for per-pollutant treated and removed loads.

    Notes
    -----
    - Wetland area and catchment ratio are sampled from heuristic percentiles
      and clipped by available parcel area; upstream traversal adjusts the
      ratio when insufficient up-gradient area exists, updating impacted parcel
      lists accordingly.
    """
    self.logger.debug("calling simulate_wetland")

    # wetland area (ha), clipped by field area
    area_field_ha = float(self.parcel_area_ha[parcel_idx])
    wet_area_stats = {"min": 0.1, "p25": 0.4, "p50": 0.81, "p75": 2.0, "max": 4.0}  # heuristic
    wet_area = self._sample_from_stats(stats=wet_area_stats, kind=None)
    wet_area = min(wet_area, area_field_ha)

    # catchment area ratio (dimensionless)
    ratio_stats = {"min": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "max": 100.0}  # heuristic
    cat_ratio = self._sample_from_stats(stats=ratio_stats, kind=None)
    cat_ratio = max(0.0, float(cat_ratio))

    # Impacted area to satisfy ratio
    impacted_idxs: List[int] = [parcel_idx]
    impacted_area_ha: float = wet_area * (1.0 + cat_ratio)
    total_available_ha = float(self.parcel_area_ha[parcel_idx])

    for up_idx in self.parcel_up_idxs[parcel_idx]:
        if up_idx not in impacted_idxs:
            impacted_idxs.append(up_idx)
            total_available_ha += float(self.parcel_area_ha[up_idx])
            if total_available_ha >= impacted_area_ha:
                break

    # Adjust ratio when upstream area is insufficient
    if impacted_area_ha > total_available_ha:
        impacted_area_ha = total_available_ha
        cat_ratio = max(0.0, (impacted_area_ha - wet_area) / max(wet_area, 1e-9))

    bmp_rec[OUTPUT_WETLAND_AREA] = float(wet_area)
    bmp_rec[OUTPUT_CATCHMENT_RATIO] = float(cat_ratio)
    bmp_rec[OUTPUT_IMPACTED_PIDS] = ",".join([self.parcel_ids[idx] for idx in impacted_idxs] if len(impacted_idxs) > 1 else [])

    # Apply reductions across impacted parcels
    remaining = impacted_area_ha
    for p_idx in impacted_idxs:
        A = float(self.parcel_area_ha[p_idx])
        if remaining <= 0:
            frac = 0.0
        elif remaining < A:
            frac = remaining / A
        else:
            frac = 1.0

        for pol_idx, pollutant in enumerate(self.pollutants):
            y = float(yields[p_idx, pol_idx])
            reduction = y * (A * frac) * eff[pol_idx]
            treated = y * (A * frac)
            bmp_outputs[OUTPUT_TREATED][pol_idx] += treated
            bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
            y_new = y - reduction / A
            yields[p_idx, pol_idx] = max(0.0, y_new)

        remaining -= A


def _simulate_grassed(
    self: "Model",
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
) -> None:
    """Simulate a grassed waterway/buffer BMP and update parcel yields.

    Parameters
    ----------
    parcel_idx : int
        Index into parcel arrays.
    eff : Sequence[float]
        Per-pollutant efficiency samples in [0, 1].
    yields : np.ndarray, shape (n_parcels, n_pollutants)
        Mutable array of parcel yields (units: load/ha). Updated in place.
    bmp_rec : Dict[str, Any]
        Per-BMP record populated with:
        - OUTPUT_LINEAR_LENGTH (m)
        - OUTPUT_BUFFER_AREA (ha)
        - OUTPUT_PORTION_TREATED (dimensionless fraction of parcel area)
    bmp_outputs : Dict[str, np.ndarray]
        Aggregators for per-pollutant treated and removed loads.

    Notes
    -----
    - Linear length is sampled as a fraction of parcel perimeter; depth is taken
      from cfg (ft) and converted to meters; area is computed in ha.
    - Side effects: mutates 'yields' in place; updates treated/removed accumulators.
    """
    self.logger.debug("calling simulate_grassed")

    # Determine linear length as a fraction of parcel perimeter
    perim_m = float(self.parcel_perim_m[parcel_idx])
    frac_stats = {"min": 0.1, "max": 0.3, "mean": 0.2}  # heuristic
    perim_frac = self._sample_from_stats(stats=frac_stats, kind=None)
    length_m = perim_m * perim_frac

    # Depth and area (length * depth -> m^2 -> ha)
    depth_ft = float(self.cfg.get(CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT))
    depth_m = depth_ft * FT_TO_M
    area_ha = (length_m * depth_m) / 10000.0

    # Portion treated
    frac_stats = {"min": 0.2, "max": 0.4, "mean": 0.3}  # heuristic
    frac_treated = self._sample_from_stats(stats=frac_stats, kind=None)

    # Update record and outputs
    bmp_rec[OUTPUT_LINEAR_LENGTH] = float(length_m)
    bmp_rec[OUTPUT_BUFFER_AREA] = float(area_ha)
    bmp_rec[OUTPUT_PORTION_TREATED] = float(frac_treated)

    A = float(self.parcel_area_ha[parcel_idx])
    for pol_idx, pollutant in enumerate(self.pollutants):
        y = float(yields[parcel_idx, pol_idx])
        reduction = y * (A * frac_treated) * eff[pol_idx]
        bmp_outputs[OUTPUT_TREATED][pol_idx] += y * (A * frac_treated)
        bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
        y_new = y - reduction / A
        yields[parcel_idx, pol_idx] = max(0.0, y_new)


def _simulate_infield(
    self: "Model",
    parcel_idx: int,
    eff: Sequence[float],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
) -> None:
    """Simulate an in-field BMP and update the parcel yield state.

    Notes
    -----
    - Treated equals baseline yield times parcel area; removed equals treated times efficiency.
    - Side effects: mutates 'yields' in place and updates 'bmp_outputs'.
    """
    self.logger.debug("calling _simulate_infield")

    A = float(self.parcel_area_ha[parcel_idx])
    for pol_idx, pollutant in enumerate(self.pollutants):
        y = float(yields[parcel_idx, pol_idx])
        reduction = y * A * eff[pol_idx]
        bmp_outputs[OUTPUT_TREATED][pol_idx] += y * A
        bmp_outputs[OUTPUT_REMOVED][pol_idx] += reduction
        y_new = y - reduction / A
        yields[parcel_idx, pol_idx] = max(0.0, y_new)


def _get_bmp_selection_probs(self: "Model", bmp_sel_path: Optional[str]) -> pd.DataFrame:
    """Return BMP type selection probabilities.

    Behavior
    --------
    - If an explicit probability file is provided via cfg, use it directly (normalized).
    - Otherwise, derive weights from estimated costs so lower-cost BMPs are more likely.
    """
    if bmp_sel_path:
        df = pd.read_csv(bmp_sel_path)
        df.columns = [c.lower() for c in df.columns]
        df = df[df[COL_CPS].astype(int).isin(self.data[DATA_CPS])].copy()
        if COL_PROBABILITY not in df.columns and "pr" in df.columns:
            df[COL_PROBABILITY] = df["pr"]
        elif COL_PROBABILITY not in df.columns and "p" in df.columns:
            df[COL_PROBABILITY] = df["p"]
        s = df[COL_PROBABILITY].sum()
        if s <= 0:
            raise ValueError(f"{CFG_BMP_SEL} probabilities sum to zero or negative")
        df[COL_PROBABILITY] = df[COL_PROBABILITY] / s
        self.logger.debug(
            f"Loaded explicit BMP selection probabilities from {bmp_sel_path}: "
            f"{df[[COL_CPS, COL_PROBABILITY]].to_dict(orient='records')}"
        )
        return df[[COL_CPS, COL_PROBABILITY]]
    else:
        if self.data[DATA_BMP_COST] is None:
            probs = np.full(len(self.data[DATA_CPS]), 1.0 / len(self.data[DATA_CPS]))
            return pd.DataFrame({COL_CPS: self.data[DATA_CPS], COL_PROBABILITY: probs})
        else:
            df = self._estimate_costs_for_probabilities()
            return df