"""AFO source accounting and conservation checks owned by the scientific model."""
from __future__ import annotations
import numpy as np

def build_operation_budget_rows(ctx, sid: int, manure_df) -> list[dict]:
    """Compute per-operation nutrient allocation budgets and enforce conservation."""
    # One row per operation prevents accidentally summing replicated
    # operation-wide excretion figures across multiple receiving fields.
    budget_rows = []
    for op_id, grouped in manure_df.groupby("afo_id", sort=True):
        first = grouped.iloc[0]
        budget = {"scenario": sid, "afo_id": op_id,
                  "receiver_count": len(grouped)}
        for source, target in (
            ("animal_n_produced_kg", "tn_produced_kg"),
            ("animal_p_produced_kg", "tp_produced_kg"),
            ("operation_available_tn_kg", "tn_available_kg"),
            ("operation_available_tp_kg", "tp_available_kg"),
        ):
            budget[target] = float(first[source])
        for key in ("tn_applied_kg", "tp_applied_kg",
                    "tn_exported_kg", "tp_exported_kg"):
            budget[key] = float(grouped[key].sum())
        op_state = getattr(ctx, "afo_bmp_operation_state", {}).get(op_id)
        if op_state:
            for nutrient in ("tn", "tp"):
                full = op_state["available_full"][nutrient.upper()]
                budget[f"final_{nutrient}_available_kg"] = full * op_state["current_collection"]
                budget[f"final_{nutrient}_applied_kg"] = full * op_state["current_collection"]
                pol_idx = next(i for i, pol in enumerate(ctx.pollutants)
                               if str(pol).upper() == nutrient.upper())
                budget[f"final_{nutrient}_exported_kg"] = sum(
                    basin_bmp_sim["current_rate"] * float(ctx.parcel_area_ha[basin_bmp_sim["parcel_idx"]])
                    for basin_bmp_sim in op_state["sources"] if basin_bmp_sim["pol_idx"] == pol_idx
                )
        for nutrient in ("tn", "tp"):
            if not np.isclose(budget[f"{nutrient}_available_kg"], budget[f"{nutrient}_applied_kg"], rtol=1e-10, atol=1e-8):
                raise AssertionError(f"AFO {op_id}: non-conserved {nutrient} allocation")
        budget_rows.append(budget)
    return budget_rows


def build_load_component_rows(ctx, sid: int) -> list[dict]:
    """Expose initial and final direct-feedlot/manure-field source contributions."""
    component_rows = []
    for idx, pid in enumerate(ctx.parcel_ids):
        area_ha = float(ctx.parcel_area_ha[idx])
        for pol_idx, pollutant in enumerate(ctx.pollutants):
            row = {"scenario": sid, "pid": str(pid), "pollutant": pollutant}
            for label, initial, final in (
                ("direct_feedlot", ctx.baseline_afo_feedlot_pathway_rates,
                 ctx.current_afo_feedlot_pathway_rates),
                ("manure_field", ctx.baseline_afo_manure_pathway_rates,
                 ctx.current_afo_manure_pathway_rates),
            ):
                for path_idx, path in enumerate(ctx.pathway_names):
                    row[f"baseline_{label}_{path}_kg_yr"] = float(initial[idx, pol_idx, path_idx]) * area_ha
                    row[f"final_{label}_{path}_kg_yr"] = float(final[idx, pol_idx, path_idx]) * area_ha
            component_rows.append(row)
    return component_rows
