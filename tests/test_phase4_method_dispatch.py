"""Phase 4: approved formulation methods are genuinely used at runtime."""
from __future__ import annotations

from dataclasses import replace
import logging

import pytest

from basin_bmp_sim.bmp.response_method_registry import (
    BMPResponseMethod, RESPONSE_IMPLEMENTATIONS, response_method_for_cps,
)
from basin_bmp_sim.science.formulations.formulation_builder import (
    build_method_plan, validate_registered_formulation,
)
from basin_bmp_sim.science.formulations.formulation_registry import FORMULATIONS, validate_formulation_selection
from basin_bmp_sim.science.process_method_registry import (
    PROCESS_IMPLEMENTATIONS, ProcessImplementation,
    resolve_process_method, resolve_plet_process_methods,
)
from basin_bmp_sim.science.assemblies.annual_model_realization import AnnualModelRealization
from basin_bmp_sim.execution.monte_carlo_driver import _ScenarioContext
from basin_bmp_sim.scenarios.generator import StochasticScenarioGenerator
from test_phase2_scenario_generation import _case


def test_approved_formulations_bind_existing_kernels():
    plet = build_method_plan({"load_generation": {"mode": "plet_rusle"}}, "plet_rusle")
    assert plet.formulation.identifier == "plet_rusle_annual_v1"
    assert plet.process_methods is not None
    assert plet.process_methods.runoff is resolve_process_method("runoff", "plet_curve_number").compute
    assert plet.process_methods.infiltration is resolve_process_method("infiltration", "plet_balance").compute
    assert plet.process_methods.sediment is resolve_process_method("sediment", "rusle").compute
    assert plet.process_methods.nutrients is resolve_process_method("nutrients", "plet_emc").compute
    assert plet.bmp_response(656).family == "wetland"
    assert plet.bmp_response(412).family == "grassed"
    assert plet.bmp_response(340).family == "infield"
    assert plet.bmp_response(999).family == "infield"  # legacy fallback remains

    statistical = build_method_plan({}, "statistical")
    assert statistical.process_methods is None
    assert callable(statistical.initialize_baseline)
    assert statistical.formulation.identifier == "statistical_annual_v1"


def test_runoff_and_nutrient_registry_dispatch_actually_executed(tmp_path, monkeypatch):
    """Checks scientific equations are called via the injected approved methods."""
    base, cfg, seed = _case("mixed_budget", tmp_path)
    calls = {"runoff": 0, "nutrients": 0}
    for family, identifier in (("runoff", "plet_curve_number"), ("nutrients", "plet_emc")):
        current = PROCESS_IMPLEMENTATIONS[family][identifier]

        def wrapper(*args, _family=family, _compute=current.compute, **kwargs):
            calls[_family] += 1
            return _compute(*args, **kwargs)

        monkeypatch.setitem(
            PROCESS_IMPLEMENTATIONS[family], identifier,
            ProcessImplementation(family, identifier, wrapper),
        )
    ctx = _ScenarioContext(cfg, base, logging.getLogger("phase4.process"), seed)
    scientific = AnnualModelRealization.initialize(ctx, ctx.logger)
    assert scientific.method_plan.formulation.identifier == "plet_rusle_annual_v1"
    assert calls["runoff"] > 0
    assert calls["nutrients"] > 0


def test_bmp_response_registry_dispatch_used_by_realization(tmp_path, monkeypatch):
    base, cfg, seed = _case("mixed_budget", tmp_path)
    events = []
    family = "infield"
    method = RESPONSE_IMPLEMENTATIONS[family]["plet_pathway_efficiency"]

    def observing_response(*args):
        events.append(args[1])
        return method.apply(*args)

    monkeypatch.setitem(
        RESPONSE_IMPLEMENTATIONS[family], method.identifier,
        BMPResponseMethod(family, method.identifier, observing_response),
    )
    # Limit the choices to an in-field BMP for the injected dispatch check.
    base["bmp_cps"] = [340]
    base["bmp_selection_probs"] = [1.0]
    ctx = _ScenarioContext(cfg, base, logging.getLogger("phase4.bmp"), seed)
    scientific = AnnualModelRealization.initialize(ctx, ctx.logger)
    generator = StochasticScenarioGenerator(ctx, cfg)
    placement = generator.next_realization(ctx)
    assert placement.choice.cps == 340
    pending = scientific.apply_placement(placement, 1)
    assert len(events) == 1
    realized = generator.complete_placement(ctx, placement, recovery_gains=pending.recovery_gains)
    scientific.complete_placement(pending, realized.cost_usd)


def test_incomplete_or_unregistered_formulations_rejected_preflight(monkeypatch):
    original = FORMULATIONS["plet_rusle_annual_v1"]
    unregistered = replace(original, process_methods={**original.process_methods, "runoff": "not_registered"})
    with pytest.raises(ValueError, match="Unregistered scientific process method"):
        validate_registered_formulation(unregistered)
    with pytest.raises(ValueError, match="Unregistered BMP response method"):
        validate_registered_formulation(replace(
            original, bmp_response_methods={**original.bmp_response_methods, "wetland": "unreviewed"}
        ))
    with pytest.raises(ValueError, match="process set must contain exactly"):
        validate_registered_formulation(replace(
            original, process_methods={"runoff": "plet_curve_number"}
        ))

    monkeypatch.setitem(FORMULATIONS, original.identifier, unregistered)
    with pytest.raises(ValueError, match="Unregistered scientific process method"):
        validate_formulation_selection({"load_generation": {"mode": "plet_rusle"}})


def test_explicit_incompatible_model_and_unknown_methods_fail():
    with pytest.raises(ValueError, match="does not match worker baseline mode"):
        build_method_plan({"load_generation": {"mode": "statistical"}}, "plet_rusle")
    with pytest.raises(ValueError, match="Unsupported load-generation mode"):
        build_method_plan({}, "invented")
    with pytest.raises(ValueError, match="Unregistered BMP response method"):
        response_method_for_cps(656, {"wetland": "first_order_areal_retention"})
    with pytest.raises(ValueError, match="Unregistered scientific process method"):
        resolve_plet_process_methods({
            "runoff": "invented", "infiltration": "plet_balance",
            "sediment": "rusle", "nutrients": "plet_emc",
        })
