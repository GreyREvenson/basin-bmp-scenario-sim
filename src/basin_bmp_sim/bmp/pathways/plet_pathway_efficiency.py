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



def _active_pathways(self: "Model") -> List[str]:
    """Return the model's active pathways, using the configured pathways or the standard three-path default.

        Returns
        -------
        List[str]
            Active pathway names in model order.
        
    """
    configured = getattr(self, "pathway_names", None)
    return list(configured) if configured else list(PATHWAY_VALUES)


def _get_pathway_load_rates(self: "Model", parcel_idx: int, pol_idx: int, total_load_rate: float) -> Dict[str, float]:
    """Return current parcel areal load rate contributions by active pathway.

        Parameters
        ----------
        parcel_idx : int
            Zero-based index of the parcel.
        pol_idx : int
            Zero-based index of the pollutant.
        total_load_rate : float
            Current total pollutant load rate for the parcel.

        Returns
        -------
        Dict[str, float]
            Current load rate for each active pathway.

        Raises
        ------
        ValueError
            If multiple pathways are active but neither pathway-resolved state nor pathway fractions are available.
        
    """
    pathways = _active_pathways(self)
    pathway_load_rates = getattr(self, "current_pathway_load_rates", None)
    if pathway_load_rates is not None:
        values = pathway_load_rates[parcel_idx, pol_idx, :]
        feedlot = getattr(self, "current_afo_feedlot_pathway_rates", None)
        if feedlot is not None:
            direct = feedlot[parcel_idx, pol_idx, :]
            values = direct if getattr(self, "current_bmp_targets_feedlot", False) else values - direct
        return {pathways[i]: float(values[i]) for i in range(len(pathways))}

    fractions = dict(
        getattr(self, "pollutant_load_rate_pathway_fractions", None)
        or getattr(self, "pollutant_load_rate_pathway_fractions", {})
        or {}
    )
    if not fractions:
        if len(pathways) != 1:
            raise ValueError("Multiple active pathways require tracked pathway load_rates or pathway fractions")
        fractions = {pathways[0]: 1.0}
    return {path: float(total_load_rate) * float(fractions[path]) for path in pathways}


def _get_current_total_load_rate(
    self: "Model", parcel_idx: int, pol_idx: int, fallback_load_rate: float
) -> float:
    """Return the current total across tracked pathways.

        Protected groundwater load rate is included when present in the active scenario state.

        Parameters
        ----------
        parcel_idx : int
            Zero-based index of the parcel.
        pol_idx : int
            Zero-based index of the pollutant.
        fallback_load_rate : float
            Load rate to use when pathway-resolved state is unavailable.

        Returns
        -------
        float
            Current total pollutant load rate across tracked pathways.
        
    """
    pathway_load_rates = getattr(self, "current_pathway_load_rates", None)
    if pathway_load_rates is None:
        return float(fallback_load_rate)
    protected = getattr(self, "current_untreated_groundwater_load_rates", None)
    protected_value = 0.0 if protected is None else float(protected[parcel_idx, pol_idx])
    return float(np.sum(pathway_load_rates[parcel_idx, pol_idx, :]) + protected_value)


def _apply_pathway_reduction(
    self: "Model",
    parcel_idx: int,
    pol_idx: int,
    treatment_fraction: float,
    eff_map: Dict[str, float],
) -> float:
    """Apply pathway-specific BMP reduction to in-memory pathway load_rates.

    The reduction is applied only when the model is tracking pathway-specific
    loads. Each pathway is reduced by ``treatment_fraction * eff_map[path]``.
    Hydrologic partitioning and BMP treatability are intentionally independent.
    Each active pathway is affected only by its own pathway-specific efficiency.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel being treated.
    pol_idx : int
        Index of the pollutant being treated.
    treatment_fraction : float
        Fraction of the parcel or drainage area treated by the BMP.
    eff_map : dict[str, float]
        Pathway-specific effectiveness values keyed by pathway name.

    Returns
    -------
    float
        Total load removed across all pathways.
    """
    pathway_load_rates = getattr(self, "current_pathway_load_rates", None)
    if pathway_load_rates is None:
        return 0.0

    removed_load_rate = 0.0
    for path_idx, path in enumerate(_active_pathways(self)):
        total_load_rate = float(pathway_load_rates[parcel_idx, pol_idx, path_idx])
        direct_component = getattr(self, "current_afo_feedlot_pathway_rates", None)
        if direct_component is None:
            current_load_rate = total_load_rate
        elif getattr(self, "current_bmp_targets_feedlot", False):
            current_load_rate = float(direct_component[parcel_idx, pol_idx, path_idx])
        else:
            current_load_rate = total_load_rate - float(direct_component[parcel_idx, pol_idx, path_idx])
        eff = float(eff_map[path])
        if current_load_rate < 0.0:
            raise RuntimeError("Current pathway load rate cannot be negative")
        if not 0.0 <= float(treatment_fraction) <= 1.0:
            raise RuntimeError("Treatment fraction must remain in [0, 1]")
        if eff > 1.0:
            raise RuntimeError("BMP efficiency must not exceed 1 after input validation")
        new_value = current_load_rate * (1.0 - treatment_fraction * eff)
        removed_load_rate += current_load_rate - new_value
        if direct_component is not None and getattr(self, "current_bmp_targets_feedlot", False):
            direct_component[parcel_idx, pol_idx, path_idx] = new_value
        if direct_component is not None and not getattr(self, "current_bmp_targets_feedlot", False):
            multiplier = 1.0 - treatment_fraction * eff
            manure = getattr(self, "current_afo_manure_pathway_rates", None)
            if manure is not None:
                manure[parcel_idx, pol_idx, path_idx] *= multiplier
            for source in getattr(self, "afo_bmp_sources_by_parcel", {}).get(parcel_idx, []):
                if source["pol_idx"] == pol_idx and source["path_idx"] == path_idx:
                    source["full_rate"] *= multiplier
                    source["current_rate"] *= multiplier
        if direct_component is None:
             pathway_load_rates[parcel_idx, pol_idx, path_idx] = new_value
        elif getattr(self, "current_bmp_targets_feedlot", False):
             pathway_load_rates[parcel_idx, pol_idx, path_idx] = (total_load_rate - current_load_rate) + new_value
        else:
             pathway_load_rates[parcel_idx, pol_idx, path_idx] = float(direct_component[parcel_idx, pol_idx, path_idx]) + new_value
    return float(removed_load_rate)

