"""Scientific execution of a single annual watershed-model realization.

Owns baseline state, sequential BMP response, manure-induced source changes,
mass accounting, and delivery to outlets. It has no IO, configuration parsing,
or stochastic BMP placement policy; the driver coordinates those concerns.
This is a native Python model contract, not yet a CSDMS BMI implementation.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np
from basin_bmp_sim.science.afo import apply_afo_feedlot_manure_effect
from basin_bmp_sim.science.formulations.formulation_builder import ScientificMethodPlan, build_method_plan
from basin_bmp_sim.domain.constants import (
    OUTPUT_TREATED, OUTPUT_REMOVED, OUTPUT_COST_USD, OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH, OUTPUT_BUFFER_AREA, OUTPUT_PORTION_TREATED,
    OUTPUT_WETLAND_AREA, OUTPUT_CATCHMENT_RATIO, OUTPUT_BMP_FAILED,
    CURRENT_TIMESTEP_YEARS as _CURRENT_TIMESTEP_YEARS,
)
from basin_bmp_sim.science.accounting.bmp_mass import (
    _bmp_impacted_parcel_indices, _baseline_mass_before_bmp_kg,
    _add_mass_metrics_to_bmp_record, _removed_mass_by_parcel_kg,
)
from basin_bmp_sim.science.routing.outlet_delivery import _accumulate_delivered_bmp_reductions
from basin_bmp_sim.science.plet_rusle import plet_annual_surface_runoff_in


@dataclass
class PendingPlacementResponse:
    """Physical BMP result before the generator realizes placement cost."""
    bmp_record: Dict[str, Any]
    pre_bmp_load_rates: np.ndarray
    mass_rate_outputs: Dict[str, np.ndarray]
    direct_removed_mass_rate: np.ndarray
    manure_changes: list[dict]
    recovery_gains: tuple[float, ...]
    parcel_index: int


@dataclass
class AnnualModelRealization:
    """Mutable state of one modeled realization, independent of the driver.

    All scientific methods use this isolated state and the context passed to
    the scientific kernels; the driver handles only scenario policy and IO.
    """
    ctx: Any
    baseline_load_rates: np.ndarray
    load_rates: np.ndarray
    load_state: Any
    afo_assignments: list
    manure_mass_ledger: list
    method_plan: ScientificMethodPlan | None = None
    delivered_reductions: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))

    @classmethod
    def initialize(cls, ctx: Any, logger) -> "AnnualModelRealization":
        methods = build_method_plan(ctx.cfg, ctx.load_generation_mode)
        baseline, load_state, afo, ledger = methods.initialize_baseline(ctx, logger)
        return cls(ctx, baseline, baseline.copy(), load_state, afo, ledger, methods)

    def apply_placement(self, placement_plan, sid: int) -> PendingPlacementResponse:
        """Apply physical treatment and AFO recovery, preserving RNG ordering.

        Cost is intentionally sampled *after* this call by the generator.
        """
        ctx = self.ctx
        load_rates = self.load_rates
        load_state = self.load_state
        n_pol = len(ctx.pollutants)
        parcel_idx = placement_plan.choice.parcel_index
        pid = placement_plan.choice.pid
        cps = placement_plan.choice.cps
        eff_maps = placement_plan.efficiency_maps()
        failed_flag = placement_plan.failed
        recovery_effect_scale = placement_plan.recovery_effect_scale
        # Per-BMP record
        bmp_rec: Dict[str, Any] = dict(
            scenario=sid,
            cps=cps,
            cps_name=ctx._get_bmp_name(cps),
            pid=str(pid),
            **{
                OUTPUT_IMPACTED_PIDS: "",
                OUTPUT_LINEAR_LENGTH: None,
                OUTPUT_BUFFER_AREA: None,
                OUTPUT_PORTION_TREATED: None,
                OUTPUT_WETLAND_AREA: None,
                OUTPUT_CATCHMENT_RATIO: None,
            },
        )
        bmp_rec[OUTPUT_BMP_FAILED] = bool(failed_flag)
        bmp_rec["load_generation_mode"] = str(ctx.load_generation_mode)

        bmp_mass_rate_outputs = {OUTPUT_TREATED: np.zeros(n_pol, dtype=float), OUTPUT_REMOVED: np.zeros(n_pol, dtype=float)}

        # Preserve the current pre-BMP load state long enough to derive a
        # true mass denominator after the BMP routine identifies its
        # impacted parcels. This is especially important for wetlands.
        pre_bmp_load_rates = load_rates.copy()

        # Feedlot CPS treat feedlot source only; all other CPS treat
        # agricultural/background loads, not the direct feedlot source.
        ctx.current_bmp_targets_feedlot = bool(cps in getattr(ctx, "feedlot_bmp_cps", set()))
        if ctx.current_bmp_targets_feedlot and not getattr(ctx, "afo_inputs", None):
            raise ValueError("feedlot-specific CPS requires configured AFO inputs")
        if ctx.current_bmp_targets_feedlot and cps in (412, 656, 657):
            raise ValueError("Wetland/buffer CPS cannot be classified as feedlot-only")

        # Scientific BMP-response dispatch is resolved from the reviewed
        # formulation; CPS selection and sampled design remain generator-owned.
        if self.method_plan is None:
            raise RuntimeError("Scientific realization has no resolved method plan")
        response_method = self.method_plan.bmp_response(cps)
        response_method.apply(
            ctx, parcel_idx, eff_maps, load_rates, bmp_rec,
            bmp_mass_rate_outputs, placement_plan.design,
        )

        # A manure-capturing feedlot BMP can increase field application
        # and therefore *increase* loading at different parcels. Count
        # those signed changes in both net BMP mass and outlet impacts.
        direct_bmp_removed = np.asarray(bmp_mass_rate_outputs[OUTPUT_REMOVED], dtype=float).copy()
        # AFO gains are sampled after BMP response in the original
        # model. Record them so replay avoids any recovery resampling.
        previous_afo_effects = len(getattr(ctx, "afo_bmp_effect_records", []))
        manure_changes = (
            apply_afo_feedlot_manure_effect(
                ctx, str(pid), cps, load_rates,
                bmp_effect_scale=recovery_effect_scale,
                realized_recovery_gains=placement_plan.recovery_gains,
            ) if ctx.current_bmp_targets_feedlot else []
        )
        recovery_gains = tuple(
            float(effect["recovery_fraction"])
            for effect in getattr(ctx, "afo_bmp_effect_records", [])[previous_afo_effects:]
        )
        for change in manure_changes:
            bmp_mass_rate_outputs[OUTPUT_REMOVED][change["pol_idx"]] -= change["added_export_kg"]
            if load_state is not None:
                # Keep PLET's equivalent manure concentration diagnostic
                # aligned with the reallocated surface nutrient mass.
                idx = change["parcel_idx"]
                pol = str(ctx.pollutants[change["pol_idx"]]).upper()
                runoff_mm = plet_annual_surface_runoff_in(load_state.parameters[idx])[3] * 25.4
                if runoff_mm > 0 and pol in load_state.concentrations[idx]:
                    rate = change["added_export_kg"] / float(ctx.parcel_area_ha[idx])
                    load_state.concentrations[idx][pol] += 100 * rate / runoff_mm
        for pol_idx, pol in enumerate(ctx.pollutants):
            bmp_rec[f"induced_manure_{str(pol).lower()}_kg_yr"] = sum(
                ch["added_export_kg"] for ch in manure_changes if ch["pol_idx"] == pol_idx
            )

        return PendingPlacementResponse(
            bmp_rec, pre_bmp_load_rates, bmp_mass_rate_outputs, direct_bmp_removed,
            manure_changes, recovery_gains, parcel_idx,
        )

    def complete_placement(self, response: PendingPlacementResponse, cost_this: float) -> dict:
        """Reconcile parcel changes and compute scientifically derived metrics."""
        ctx = self.ctx
        load_rates = self.load_rates
        bmp_rec = response.bmp_record
        bmp_mass_rate_outputs = response.mass_rate_outputs
        pre_bmp_load_rates = response.pre_bmp_load_rates
        direct_bmp_removed = response.direct_removed_mass_rate
        manure_changes = response.manure_changes
        parcel_idx = response.parcel_index
        # Finalize the BMP record using explicit mass fields. The current
        # one-year timestep converts annual mass rates to timestep mass.
        bmp_rec[OUTPUT_COST_USD] = cost_this
        treated_mass_kg = np.asarray(bmp_mass_rate_outputs[OUTPUT_TREATED], dtype=float) * _CURRENT_TIMESTEP_YEARS
        removed_mass_kg = np.asarray(bmp_mass_rate_outputs[OUTPUT_REMOVED], dtype=float) * _CURRENT_TIMESTEP_YEARS
        induced_receiver_indices = tuple(sorted({
            int(ch["parcel_idx"]) for ch in manure_changes
            if float(ch["added_export_kg"]) != 0.
        }))
        accounting_indices = sorted(
            set(_bmp_impacted_parcel_indices(ctx, parcel_idx, bmp_rec))
            | set(induced_receiver_indices)
        )
        # Distinguish the physically treated parcels from the wider
        # accounting boundary used for net mass and its percentage.
        bmp_rec["mass_accounting_pids"] = ",".join(
            str(ctx.parcel_ids[idx]) for idx in accounting_indices
        )
        baseline_mass_kg = _baseline_mass_before_bmp_kg(
            ctx, pre_bmp_load_rates, parcel_idx, bmp_rec,
            extra_parcel_indices=induced_receiver_indices,
        )
        for pol_idx, pol in enumerate(ctx.pollutants):
            bmp_rec[f"direct_bmp_removed_{pol}_kg_yr"] = float(
                direct_bmp_removed[pol_idx]
            )
        _add_mass_metrics_to_bmp_record(
            bmp_rec,
            list(ctx.pollutants),
            baseline_mass_kg,
            treated_mass_kg,
            removed_mass_kg,
        )
        # Attribute actual signed load changes to the affected parcel's
        # outlets and delivery factors, not just the BMP host. Wetlands
        # may change several upstream parcels, while feedlot recovery
        # may increase manure export on separate receiving parcels.
        changes = _removed_mass_by_parcel_kg(
            ctx,
            pre_bmp_load_rates,
            load_rates,
            parcel_idx,
            bmp_rec,
            removed_mass_kg,
            extra_parcel_indices=induced_receiver_indices,
        )
        _accumulate_delivered_bmp_reductions(ctx, changes, self.delivered_reductions)
        return bmp_rec
