"""Model-owned derived load diagnostics exposed as per-parcel records."""
from __future__ import annotations
from typing import Any, Dict
from basin_bmp_sim.science.plet_rusle import calculate_load_diagnostics


def model_parameter_records(ctx, sid: int, load_state) -> list[dict]:
    """Return realized process parameters and derived PLET/RUSLE diagnostics."""
    result: list[dict] = []
    # Realized stochastic PLET/RUSLE inputs and derived diagnostics.
    if load_state is not None:
        for parcel_idx, pid_i in enumerate(ctx.parcel_ids):
            initial_params = load_state.baseline_parameters[parcel_idx]
            final_params = load_state.parameters[parcel_idx]
            initial_conc = load_state.baseline_concentrations[parcel_idx]
            final_conc = load_state.concentrations[parcel_idx]
            initial_gw_conc = load_state.baseline_groundwater_concentrations[parcel_idx]
            final_gw_conc = load_state.groundwater_concentrations[parcel_idx]
            row: Dict[str, Any] = {"scenario": sid, "pid": str(pid_i)}
            for key in sorted(set(initial_params) | set(final_params)):
                row[f"initial_{key}"] = initial_params.get(key)
                row[f"final_{key}"] = final_params.get(key)
            for pol in sorted(set(initial_conc) | set(final_conc)):
                label = str(pol).lower()
                row[f"initial_concentration_{label}_mg_l"] = initial_conc.get(pol)
                row[f"final_concentration_{label}_mg_l"] = final_conc.get(pol)
            for pol in sorted(set(initial_gw_conc) | set(final_gw_conc)):
                label = str(pol).lower()
                row[f"initial_groundwater_concentration_{label}_mg_l"] = initial_gw_conc.get(pol)
                row[f"final_groundwater_concentration_{label}_mg_l"] = final_gw_conc.get(pol)
            for pol_idx, pol in enumerate(ctx.pollutants):
                label = str(pol).lower()
                for path_idx, path in enumerate(ctx.pathway_names):
                    path_label = str(path).lower().replace(" ", "_")
                    initial_pathway_load_rate = float(
                        load_state.baseline_pathway_load_rates[parcel_idx, pol_idx, path_idx]
                    )
                    final_pathway_load_rate = float(
                        load_state.pathway_load_rates[parcel_idx, pol_idx, path_idx]
                    )
                    # Explicit annual areal-load-rate columns.
                    row[f"initial_{path_label}_{label}_load_rate_kg_ha_yr"] = initial_pathway_load_rate
                    row[f"final_{path_label}_{label}_load_rate_kg_ha_yr"] = final_pathway_load_rate
            for key, value in calculate_load_diagnostics(initial_params).items():
                row[f"initial_{key}"] = value
            for key, value in calculate_load_diagnostics(final_params).items():
                row[f"final_{key}"] = value
            result.append(row)
    return result
