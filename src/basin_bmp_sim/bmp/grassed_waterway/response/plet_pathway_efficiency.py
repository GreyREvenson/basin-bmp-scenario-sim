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
    from basin_bmp_sim.execution.monte_carlo_driver import Model

from basin_bmp_sim.domain.constants import (
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
from basin_bmp_sim.common.logging import log_scope
from basin_bmp_sim.domain.parcel_graph import iter_full_upstream_by_distance
from basin_bmp_sim.config.loading import load_bmp_selection_probabilities

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

from basin_bmp_sim.bmp.pathways.plet_pathway_efficiency import _active_pathways

def _simulate_grassed(
    self: "Model",
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    load_rates: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_mass_rate_outputs: Dict[str, np.ndarray],
    design=None,
) -> None:
    """Apply a grassed buffer BMP to the selected parcel.

    The buffer length is sampled as a fraction of parcel perimeter, converted
    to area using the configured buffer depth, and then used to compute the
    treated and removed pollutant loads for the parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel receiving the BMP.
    eff_maps : sequence of dict[str, float]
        Effectiveness maps, one per pollutant, keyed by pathway name.
    load_rates : numpy.ndarray
        Parcel-by-pollutant areal load-rate array updated in place.
    bmp_rec : dict[str, Any]
        Record for the current BMP placement, updated with buffer metadata.
    bmp_mass_rate_outputs : dict[str, numpy.ndarray]
        Accumulators for treated and removed pollutant loads.

    Returns
    -------
    None
    """
    with log_scope(label="simulate_grassed", logger=self.logger):
        self.logger.verbose("calling simulate_grassed")

        if design is None:
            from basin_bmp_sim.scenarios.bmp_designs import sample_bmp_design
            design = sample_bmp_design(self, parcel_idx, 412)
        if design.family != "grassed":
            raise ValueError(f"Expected grassed-waterway design, got {design.family!r}")
        bmp_rec.update(design.record_fields())
        treated_area_fraction = design.treated[0][1]
        parcel_area_ha = float(self.parcel_area_ha[parcel_idx])
        for pol_idx, pollutant in enumerate(self.pollutants):
            load_rate = float(load_rates[parcel_idx, pol_idx])
            pathway_load_rates_by_name = self._get_pathway_load_rates(parcel_idx, pol_idx, load_rate)
            efficiency_by_pathway = eff_maps[pol_idx]

            treated_baseline_mass_rate_kg_per_yr = sum(pathway_load_rates_by_name.values()) * (parcel_area_ha * treated_area_fraction)
            removed_mass_rate_kg_per_yr = (parcel_area_ha * treated_area_fraction) * sum(
                pathway_load_rates_by_name[path] * efficiency_by_pathway[path]
                for path in _active_pathways(self)
            )
            self._apply_pathway_reduction(parcel_idx, pol_idx, treated_area_fraction, efficiency_by_pathway)

            bmp_mass_rate_outputs[OUTPUT_TREATED][pol_idx] += treated_baseline_mass_rate_kg_per_yr
            bmp_mass_rate_outputs[OUTPUT_REMOVED][pol_idx] += removed_mass_rate_kg_per_yr
            updated_load_rate = self._get_current_total_load_rate(parcel_idx, pol_idx, load_rate) if getattr(self, "current_pathway_load_rates", None) is not None else (load_rate - removed_mass_rate_kg_per_yr / parcel_area_ha)
            load_rates[parcel_idx, pol_idx] = updated_load_rate

