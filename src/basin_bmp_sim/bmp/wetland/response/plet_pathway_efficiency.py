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
from typing import Any, Callable, Dict, List, Sequence, Union


from basin_bmp_sim.domain.constants import OUTPUT_REMOVED, OUTPUT_TREATED
from basin_bmp_sim.common.logging import log_scope


from basin_bmp_sim.bmp.pathways.plet_pathway_efficiency import _active_pathways

def _simulate_wetland(
    self: Any,
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    load_rates: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_mass_rate_outputs: Dict[str, np.ndarray],
    cps: Union[int, str] = 656,
    design=None,
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

        # The response consumes an explicit physical design.  A design is
        # sampled here only for legacy direct callers of this helper.
        if design is None:
            from basin_bmp_sim.scenarios.bmp_designs import sample_bmp_design
            design = sample_bmp_design(self, parcel_idx, int(cps))
        if design.family != "wetland":
            raise ValueError(f"Expected wetland design, got {design.family!r}")
        bmp_rec.update(design.record_fields())
        for p_idx, treated_area_fraction in design.treated:
            parcel_area_ha = float(self.parcel_area_ha[p_idx])
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
