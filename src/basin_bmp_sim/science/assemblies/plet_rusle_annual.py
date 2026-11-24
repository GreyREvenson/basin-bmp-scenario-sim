"""Annual PLET/RUSLE baseline generation with original numerical behavior."""
from __future__ import annotations
from basin_bmp_sim.science.plet_rusle import initialize_plet_rusle_state
from basin_bmp_sim.science.afo import apply_afo_baseline, mass_afo_realization


def initialize_plet_rusle_baseline(ctx, logger, *, process_methods=None):
    """Sample baseline parcel loads, PLET state, and AFO assignments."""
    afo_assignments = []
    manure_mass_ledger = []
    baseline_load_rates, load_state = initialize_plet_rusle_state(
        ctx, process_methods=process_methods
    )
    ctx.current_pathway_load_rates = load_state.pathway_load_rates
    if (ctx.afo_inputs or {}).get("method") == "mass_based":
        afo_assignments, manure_mass_ledger = mass_afo_realization(ctx, state=load_state)
    else:
        afo_assignments = apply_afo_baseline(ctx, load_state)
    if afo_assignments:
        baseline_load_rates = load_state.pathway_load_rates.sum(axis=2)
    load_rates = baseline_load_rates.copy()
    ctx.current_untreated_groundwater_load_rates = load_state.untreated_groundwater_load_rates
    ctx.pathway_names = ["surface", "subsurface"]
    logger.info("Generated baseline_load_rates PLET/RUSLE load_rates for surface and subsurface pathways")

    return baseline_load_rates, load_state, afo_assignments, manure_mass_ledger
