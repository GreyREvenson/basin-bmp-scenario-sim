"""Plet initialization for the annual PLET/RUSLE model."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
from basin_bmp_sim.domain.constants import (
    PLET_PATHWAY_VALUES as PLET_PATHWAY_NAMES,
    RUSLE_REQUIRED_INPUTS as _REQUIRED_RUSLE,
)
from basin_bmp_sim.science.plet_defaults import apply_plet_parameter_defaults
from basin_bmp_sim.science.nutrients.surface_load_plet_emc import calculate_plet_pathway_load_rates
from basin_bmp_sim.science.plet_sampling import (
    _sample_concentrations,
    _sample_parameter_table,
    _sample_plet_hydrology,
)
from basin_bmp_sim.science.plet_state import LoadState


def initialize_plet_rusle_state(ctx: Any, *, process_methods=None) -> Tuple[np.ndarray, LoadState]:
    """Create the initial parcel load state for a scenario.

    This function samples parcel-level parameters and concentrations, verifies
    that required inputs are present, computes baseline pathway load_rates, and
    returns both the total load array and the mutable scenario state.

    Parameters
    ----------
    ctx : Any
        Scenario context containing the input tables, pollutant list, and
        configuration flags.

    Returns
    -------
    tuple[numpy.ndarray, LoadState]
        Baseline parcel loads and the corresponding mutable load state.

    Raises
    ------
    ValueError
        If required inputs are missing or incomplete for any parcel.
    """

    # Scenario workers provide the full hydrologic parcel universe via
    # ``parcel_ids``.  Fall back to ``parcel_selection_ids`` only for legacy
    # direct callers/tests that construct a minimal context without parcel_ids.
    # TODO: remove legacy fallback to parcel_selection_ids. double-check that it
    # will not break existing tests or direct calls.
    parcel_ids = [
        str(pid)
        for pid in getattr(ctx, "parcel_ids", getattr(ctx, "parcel_selection_ids", []))
    ]
    plet = _sample_parameter_table(ctx, ctx.plet_inputs, parcel_ids, cache_prefix="plet")
    rusle = _sample_parameter_table(ctx, ctx.rusle_inputs, parcel_ids, cache_prefix="rusle")
    concentrations = _sample_concentrations(ctx, ctx.pollutant_concentrations, parcel_ids)
    groundwater_concentrations = _sample_concentrations(ctx, getattr(ctx, "groundwater_concentrations", None), parcel_ids)

    parameters: List[Dict[str, Any]] = []
    has_rusle: List[bool] = []
    load_rates = np.zeros((len(parcel_ids), len(ctx.pollutants)), dtype=float)
    pathway_load_rates = np.zeros((len(parcel_ids), len(ctx.pollutants), len(PLET_PATHWAY_NAMES)), dtype=float)
    untreated_groundwater_load_rates = np.zeros((len(parcel_ids), len(ctx.pollutants)), dtype=float)
    hydrology_cache: Dict[Tuple[str, str], float] = {}
    for i, pid in enumerate(parcel_ids):
        # Input completeness and concentration coverage are validated once in
        # input_config before scenario workers start.
        derived_hydrology = _sample_plet_hydrology(
            ctx,
            plet[i]["land_cover"],
            plet[i]["hsg"],
            pid=pid,
            cache=hydrology_cache,
        )
        combined = dict(plet[i])
        combined.update(derived_hydrology)
        combined.update(rusle[i])
        combined = apply_plet_parameter_defaults(combined, ctx.pollutants)
        parameters.append(combined)
        has_rusle.append(all(name in combined for name in _REQUIRED_RUSLE))
        nutrient_method = (
            process_methods.nutrients if process_methods is not None
            else calculate_plet_pathway_load_rates
        )
        parcel_pathway_load_rates = nutrient_method(
            combined,
            concentrations[i],
            groundwater_concentrations[i],
            ctx.pollutants,
            **({"process_methods": process_methods} if process_methods is not None else {}),
        )
        pathway_load_rates[i, :, :] = parcel_pathway_load_rates
        # Compatibility field retained for callers/tests that inspect the old
        # protected-groundwater array. Actual PLET subsurface load is now in
        # pathway_load_rates[:, :, 1], so this stays zero.
        # TODO: remove untreated_groundwater_load_rates but confirm that it
        # will not break existing functionality
        untreated_groundwater_load_rates[i, :] = 0.0
        load_rates[i, :] = np.sum(pathway_load_rates[i, :, :], axis=1)

    state = LoadState(
        parcel_ids=parcel_ids,
        parameters=parameters,
        concentrations=concentrations,
        groundwater_concentrations=groundwater_concentrations,
        has_rusle=has_rusle,
        pollutants=list(ctx.pollutants),
        pathway_load_rates=pathway_load_rates,
        untreated_groundwater_load_rates=untreated_groundwater_load_rates,
        baseline_parameters=[dict(values) for values in parameters],
        baseline_concentrations=[dict(values) for values in concentrations],
        baseline_groundwater_concentrations=[dict(values) for values in groundwater_concentrations],
        baseline_pathway_load_rates=pathway_load_rates.copy(),
        baseline_untreated_groundwater_load_rates=untreated_groundwater_load_rates.copy(),
    )
    return load_rates.copy(), state
