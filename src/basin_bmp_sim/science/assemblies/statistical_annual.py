"""Annual statistical baseline generation with original sampling behavior."""
from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from basin_bmp_sim.science.afo.manure import mass_afo_realization
from basin_bmp_sim.config.distributions import sample_group_key


def initialize_statistical_baseline(ctx, n_pol, logger):
    """Sample baseline parcel and pathway pollutant loads."""
    load_state = None
    afo_assignments = []
    manure_mass_ledger = []
    baseline_load_rates = np.zeros((len(ctx.parcel_ids), n_pol), dtype=float)
    load_rates = np.zeros_like(baseline_load_rates)
    pathway_load_rates = np.zeros(
        (len(ctx.parcel_ids), n_pol, len(ctx.pathway_names)), dtype=float
    )
    load_rate_cache: Dict[Tuple[str, str], float] = {}

    def sample_load(row: Dict[str, Any], pid: str, pollutant: str, pathway: str) -> float:
        key = sample_group_key(
            row, pid=pid, variable=f"load_rate:{pollutant}:{pathway}"
        )
        if key not in load_rate_cache:
            load_rate_cache[key] = float(ctx._sample_from_stats(row, kind="load_rate"))
        return load_rate_cache[key]

    for i, _pid in enumerate(ctx.parcel_ids):
        for pol_idx in range(n_pol):
            entry = ctx.pollutant_load_rate_stats[i][pol_idx]
            if ctx.pollutant_load_rate_is_aggregate:
                aggregate_load_rate = sample_load(entry, str(_pid), ctx.pollutants[pol_idx], "aggregate")
                for path_idx, path in enumerate(ctx.pathway_names):
                    pathway_load_rates[i, pol_idx, path_idx] = (
                        aggregate_load_rate * float(ctx.pollutant_load_rate_pathway_fractions[path])
                    )
            else:
                for path_idx, path in enumerate(ctx.pathway_names):
                    pathway_load_rates[i, pol_idx, path_idx] = sample_load(
                        entry[path], str(_pid), ctx.pollutants[pol_idx], path
                    )
            baseline_load_rates[i, pol_idx] = float(np.sum(pathway_load_rates[i, pol_idx, :]))
            load_rates[i, pol_idx] = baseline_load_rates[i, pol_idx]
    ctx.current_pathway_load_rates = pathway_load_rates
    if (ctx.afo_inputs or {}).get("method") == "mass_based":
        afo_assignments, manure_mass_ledger = mass_afo_realization(ctx)
        baseline_load_rates[:, :] = pathway_load_rates.sum(axis=2)
        load_rates[:, :] = baseline_load_rates
    logger.info(
        "Generated statistical baseline_load_rates parcel load_rates across pathways: "
        + ", ".join(ctx.pathway_names)
    )

    return baseline_load_rates, load_state, afo_assignments, manure_mass_ledger
