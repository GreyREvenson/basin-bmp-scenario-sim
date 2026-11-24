"""Phase 3 scientific-model boundary and pre-refactor execution semantics."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from basin_bmp_sim.execution.monte_carlo_driver import _ScenarioContext
from basin_bmp_sim.science.assemblies.annual_model_realization import AnnualModelRealization
from basin_bmp_sim.scenarios.generator import StochasticScenarioGenerator
from test_phase2_scenario_generation import _case


@pytest.mark.parametrize("case", ["mixed_budget", "statistical_multi", "afo_recovery"])
def test_native_model_api_preserves_response_and_replay(case, tmp_path):
    """The driver and file output system are not required to apply a placement."""
    payload, cfg, seed = _case(case, tmp_path)
    ctx = _ScenarioContext(cfg, payload, logging.getLogger(f"p3.{case}"), seed)
    scientific = AnnualModelRealization.initialize(ctx, ctx.logger)
    initial = scientific.baseline_load_rates.copy()
    np.testing.assert_array_equal(scientific.load_rates, initial)

    generator = StochasticScenarioGenerator(ctx, cfg)
    generator.validate_replay(ctx, 1, seed)
    plan = generator.next_realization(ctx)
    assert plan is not None

    pending = scientific.apply_placement(plan, 1)
    assert pending.bmp_record["cps"] == plan.choice.cps
    assert pending.bmp_record["pid"] == plan.choice.pid
    assert "cost_usd" not in pending.bmp_record  # generator still owns cost

    realized = generator.complete_placement(ctx, plan, recovery_gains=pending.recovery_gains)
    record = scientific.complete_placement(pending, realized.cost_usd)
    assert record["cost_usd"] == realized.cost_usd
    assert generator.total_bmp == 1
    assert all(np.isfinite(values) for reductions in scientific.delivered_reductions.values()
               for values in reductions.values())

    scenario = generator.complete_scenario(ctx, 1, seed)
    assert scenario.placements[0] == realized


def test_context_helpers_are_statically_declared(tmp_path):
    """Methods are descriptors on the class, not individually bound at runtime."""
    payload, cfg, seed = _case("mixed_budget", tmp_path)
    ctx = _ScenarioContext(cfg, payload, logging.getLogger("p3.context"), seed)
    for name in (
        "_sample_from_stats", "_simulate_wetland", "_simulate_grassed",
        "_simulate_infield", "_apply_pathway_reduction", "_delivery_coeffs",
    ):
        assert name not in vars(ctx)
        assert callable(getattr(ctx, name))
    assert ctx.afo_inputs is None


def test_context_requires_shared_domain_fields(tmp_path):
    payload, cfg, seed = _case("mixed_budget", tmp_path)
    del payload["parcel_ids"]
    with pytest.raises(ValueError, match="parcel_ids"):
        _ScenarioContext(cfg, payload, logging.getLogger("p3.missing"), seed)


def test_unsupported_scientific_model_rejected(tmp_path):
    payload, cfg, seed = _case("mixed_budget", tmp_path)
    payload["load_generation_mode"] = "not_a_model"
    ctx = _ScenarioContext(cfg, payload, logging.getLogger("p3.invalid"), seed)
    with pytest.raises(ValueError, match="Unsupported load-generation mode"):
        AnnualModelRealization.initialize(ctx, ctx.logger)
