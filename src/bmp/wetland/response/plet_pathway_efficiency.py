"""BMP selection and load-reduction helpers.

This module contains the logic used to choose best management practices
(BMPs), sample their effectiveness, and apply their impacts to parcel-level
pollutant loads. The functions are implemented as model helpers and operate
on shared model state such as parcel geometry, pollutant load_rates, and
simulation outputs.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from src.driver.monte_carlo_driver import Model

from src.constants import (
    FT_TO_M,
    CFG_BUFFER_DEPTH_FT,
    CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
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
    PATHWAY_VALUES,
)
from src.logging_utils import log_scope
from src.parcel_graph import iter_full_upstream_by_distance
from src.input_config import load_bmp_selection_probabilities

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

from src.bmp.pathways.plet_pathway_efficiency import _active_pathways

def _simulate_wetland(
    self: "Model",
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    load_rates: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_mass_rate_outputs: Dict[str, np.ndarray],
    cps: Union[int, str] = 656,
) -> None:
    """Apply a wetland BMP and update parcel loads.

    The wetland treatment area is sampled from heuristic area statistics and
    clipped to the parcel area. The function then allocates the wetland to a
    contributing set of upstream parcels, computes the impacted drainage area,
    and applies pathway-specific reductions to each affected parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel receiving the wetland BMP.
    eff_maps : sequence of dict[str, float]
        Effectiveness maps, one per pollutant, keyed by pathway name.
    load_rates : numpy.ndarray
        Parcel-by-pollutant areal load-rate array updated in place.
    bmp_rec : dict[str, Any]
        Record for the current BMP placement, updated with wetland metadata.
    bmp_mass_rate_outputs : dict[str, numpy.ndarray]
        Accumulators for treated and removed pollutant loads.
    cps : int or str, optional
        BMP CPS code associated with the wetland. Default is ``656``.

    Returns
    -------
    None
    """
    with log_scope(label="simulate_wetland", logger=self.logger):
        self.logger.verbose("calling simulate_wetland")

        # wetland area (ha), clipped by field area
        area_field_ha = float(self.parcel_area_ha[parcel_idx])
        deterministic_design = self.cfg.get(CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS, False)
        if deterministic_design:
            wet_area = 0.20 * area_field_ha
        else:
            wet_area_stats = {"min": 0.1, "p25": 0.4, "p50": 0.81, "p75": 2.0, "max": 4.0}  # heuristic
            wet_area = self._sample_from_stats(stats=wet_area_stats, kind=None)
            wet_area = min(wet_area, area_field_ha)
        # Restored from legacy: detailed diagnostics
        self.logger.verbose(
            f"selected wetland area of {wet_area:.2f} ha in parcel idx={parcel_idx} of area={area_field_ha:.2f} ha"
        )

        # catchment area ratio (dimensionless)
        if deterministic_design:
            cat_ratio = 4.0
        else:
            ratio_stats = {"min": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "max": 100.0}  # heuristic
            cat_ratio = float(self._sample_from_stats(stats=ratio_stats, kind=None))

        # Impacted area to satisfy ratio
        impacted_idxs: List[int] = [parcel_idx]
        impacted_area_ha: float = (
            area_field_ha if deterministic_design else wet_area * (1.0 + cat_ratio)
        )
        total_available_ha = float(self.parcel_area_ha[parcel_idx])

        # Walk the complete upstream DAG, nearest (fewest graph edges) first.
        # On a tie, process ascending PID. Only the first upstream parcels
        # needed to reach the sampled treatment catchment are included.
        for up_idx in iter_full_upstream_by_distance(
            parcel_idx, self.parcel_up_idxs, self.parcel_ids
        ):
            if total_available_ha >= impacted_area_ha:
                break
            if up_idx not in impacted_idxs:
                impacted_idxs.append(up_idx)
                total_available_ha += float(self.parcel_area_ha[up_idx])
                self.logger.verbose(
                    f"added upgradient parcel (pid={self.parcel_ids[up_idx]}) with area "
                    f"{self.parcel_area_ha[up_idx]:.2f} ha to wetland-impacted parcels"
                )
                if total_available_ha >= impacted_area_ha:
                    break

        # Adjust ratio when upstream area is insufficient
        if impacted_area_ha > total_available_ha:
            self.logger.verbose(
                f"total available upgradient area ({total_available_ha:.2f} ha) < impacted area "
                f"(wetland+catchment) ({impacted_area_ha:.2f} ha)"
            )
            impacted_area_ha = total_available_ha
            # This is a derived-state constraint, not input sanitization. The
            # denominator guard protects the ratio calculation if a degenerate
            # zero-area parcel reaches this internal heuristic.
            cat_ratio = (impacted_area_ha - wet_area) / max(wet_area, 1e-9)
            self.logger.verbose(
                f"reduced impacted area to {impacted_area_ha:.2f} ha and catchment ratio to {cat_ratio:.2f}"
            )

        bmp_rec[OUTPUT_WETLAND_AREA] = float(wet_area)
        bmp_rec[OUTPUT_CATCHMENT_RATIO] = float(cat_ratio)
        bmp_rec[OUTPUT_IMPACTED_PIDS] = ",".join([self.parcel_ids[idx] for idx in impacted_idxs] if len(impacted_idxs) > 1 else [])

        # Apply reductions across impacted parcels
        remaining = impacted_area_ha
        for p_idx in impacted_idxs:
            parcel_area_ha = float(self.parcel_area_ha[p_idx])
            if remaining <= 0:
                treated_area_fraction = 0.0
            elif remaining < parcel_area_ha:
                treated_area_fraction = remaining / parcel_area_ha
            else:
                treated_area_fraction = 1.0
            self.logger.verbose(
                f"processing wetland-impacted parcel pid={self.parcel_ids[p_idx]}, "
                f"area={parcel_area_ha:.2f} ha, fraction draining={treated_area_fraction:.2f}"
            )

            for pol_idx, pollutant in enumerate(self.pollutants):
                load_rate = float(load_rates[p_idx, pol_idx])
                pathway_load_rates_by_name = self._get_pathway_load_rates(p_idx, pol_idx, load_rate)
                efficiency_by_pathway = eff_maps[pol_idx]

                treated_baseline_mass_rate_kg_per_yr = sum(pathway_load_rates_by_name.values()) * (parcel_area_ha * treated_area_fraction)
                removed_mass_rate_kg_per_yr = (parcel_area_ha * treated_area_fraction) * sum(
                    pathway_load_rates_by_name[path] * efficiency_by_pathway[path]
                    for path in _active_pathways(self)
                )
                self._apply_pathway_reduction(p_idx, pol_idx, treated_area_fraction, efficiency_by_pathway)

                bmp_mass_rate_outputs[OUTPUT_TREATED][pol_idx] += treated_baseline_mass_rate_kg_per_yr
                bmp_mass_rate_outputs[OUTPUT_REMOVED][pol_idx] += removed_mass_rate_kg_per_yr
                updated_load_rate = (
                    self._get_current_total_load_rate(p_idx, pol_idx, load_rate)
                    if getattr(self, "current_pathway_load_rates", None) is not None
                    else (load_rate - removed_mass_rate_kg_per_yr / parcel_area_ha)
                )
                load_rates[p_idx, pol_idx] = updated_load_rate

            remaining -= parcel_area_ha

