"""Phase 6: independent BMI exchange, replay and numerical fidelity tests."""
from __future__ import annotations

from dataclasses import replace
import logging

import numpy as np
import pytest

from basin_bmp_sim.execution.bmi_coupling import AnnualBmiScenarioCoupler, get_bmi_value, transfer_bmi_value
from basin_bmp_sim.execution.monte_carlo_driver import _ScenarioContext
from basin_bmp_sim.interfaces.bmi.live_scenario_generator_bmi import LiveScenarioGeneratorBmi
from basin_bmp_sim.interfaces.bmi.watershed_model_bmi import WatershedModelBmi
from basin_bmp_sim.interfaces.placement_exchange import PLACEMENT_FIELDS, placement_arrays, placement_from_arrays
from basin_bmp_sim.scenarios.generator import StochasticScenarioGenerator
from test_phase2_scenario_generation import _case, _capture


@pytest.mark.parametrize("case", ["mixed_budget", "statistical_multi", "afo_recovery"])
def test_complete_recorded_placements_are_lossless_in_bmi_contract(tmp_path, monkeypatch, case):
    """All design families, replay gain values, and efficiency order roundtrip."""
    _, replay = _capture(monkeypatch, tmp_path / "capture", case)
    shared, cfg, seed = _case(case, tmp_path / "x")
    ctx = _ScenarioContext(cfg, shared, logging.getLogger("phase6.roundtrip"), seed)
    for original in replay.placements:
        packed = placement_arrays(original, ctx)
        assert set(packed) == set(PLACEMENT_FIELDS)
        loaded = placement_from_arrays({k: np.asarray(v) for k, v in packed.items()}, ctx)
        assert loaded.choice == original.choice
        assert loaded.design == original.design
        assert loaded.efficiency_by_pollutant == original.efficiency_by_pollutant
        assert loaded.failed == original.failed
        assert loaded.recovery_effect_scale == original.recovery_effect_scale
        assert loaded.recovery_gains == (original.recovery_gains or None)


def _fresh_coupler(case, tmp_path):
    shared, cfg, seed = _case(case, tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger("phase6.coupler"), seed)
    coupler = AnnualBmiScenarioCoupler(ctx, cfg, ctx.logger, scenario_id=1, seed=seed)
    coupler.initialize()
    return coupler


@pytest.mark.parametrize("case", ["mixed_budget", "statistical_multi", "afo_recovery"])
def test_driver_coupler_transfers_all_design_and_response_fields(tmp_path, case, monkeypatch):
    """Native placement and recovery properties must not be used by driver."""
    def not_a_bmi_exchange(*args, **kwargs):
        raise AssertionError("Native model placement transfer bypassed BMI")

    monkeypatch.setattr(WatershedModelBmi, "set_placement", not_a_bmi_exchange)
    monkeypatch.setattr(LiveScenarioGeneratorBmi, "pending_placement",
                        property(not_a_bmi_exchange))
    monkeypatch.setattr(WatershedModelBmi, "pending_recovery_gains",
                        property(not_a_bmi_exchange))
    coupler = _fresh_coupler(case, tmp_path)
    try:
        first = coupler.next_bmp()
        assert first is not None
        assert first.completed_count == 1
        assert first.cost_usd == first.bmp_record["cost_usd"]
        assert np.all(np.isfinite(first.delivered_reductions))
        assert int(get_bmi_value(coupler.generator, "placement__completed")[0]) == 1
        assert int(get_bmi_value(coupler.model, "scenario__identifier")[0]) == 1
        for v in PLACEMENT_FIELDS:
            assert v in coupler.generator.get_output_var_names()
            assert v in coupler.model.get_input_var_names()
        scenario = coupler.completed_scenario() if coupler.generator.get_end_time() == 1 else None
        if scenario is not None:
            assert len(scenario.placements) == 1
    finally:
        coupler.finalize()


def test_missing_and_invalid_scientific_inputs_rejected_before_model_update(tmp_path):
    coupler = _fresh_coupler("mixed_budget", tmp_path)
    try:
        coupler.generator.update()
        with pytest.raises(RuntimeError, match="Queue a placement"):
            coupler.model.update()
        for name in PLACEMENT_FIELDS:
            if name != "placement__efficiency":
                transfer_bmi_value(coupler.generator, coupler.model, name)
        with pytest.raises(RuntimeError, match="placement__efficiency"):
            coupler.model.update()
        transfer_bmi_value(coupler.generator, coupler.model, "placement__efficiency")
        # The input contract validates treatment fractions, not just shape.
        fractions = get_bmi_value(coupler.model, "placement__affected_parcel_fraction")
        coupler.model.set_value("placement__affected_parcel_fraction", fractions * 0 + 2.0)
        with pytest.raises(ValueError, match="fraction"):
            coupler.model.update()
        transfer_bmi_value(coupler.generator, coupler.model, "placement__affected_parcel_fraction")
        coupler.model.update()
        with pytest.raises(RuntimeError, match="placement__cost"):
            coupler.model.complete_pending()
        with pytest.raises(RuntimeError, match="placement__recovery_gains"):
            coupler.generator.complete_pending()
        transfer_bmi_value(coupler.model, coupler.generator, "placement__recovery_gains")
        coupler.generator.complete_pending()
        transfer_bmi_value(coupler.generator, coupler.model, "placement__cost")
        response = coupler.model.complete_pending()
        assert response["cost_usd"] >= 0
    finally:
        coupler.finalize()


def test_mediator_rejects_units_and_type_mismatch(tmp_path, monkeypatch):
    coupler = _fresh_coupler("afo_recovery", tmp_path)
    try:
        coupler.generator.update()
        original = coupler.generator.get_var_units
        monkeypatch.setattr(coupler.generator, "get_var_units",
                            lambda name: "invalid" if name == "placement__cost_quantity_ha" else original(name))
        with pytest.raises(ValueError, match="unit mismatch"):
            transfer_bmi_value(coupler.generator, coupler.model, "placement__cost_quantity_ha")
        monkeypatch.setattr(coupler.generator, "get_var_units", original)
        transfer_bmi_value(coupler.generator, coupler.model, "placement__cost_quantity_ha")
    finally:
        # The pending state is intentionally left incomplete here.
        coupler.generator._pending = None
        coupler.finalize()


def test_bmi_only_live_replay_is_exact(tmp_path, monkeypatch):
    """Stored BMP designs and all AFO recovery gains replay without draws."""
    _, replay = _capture(monkeypatch, tmp_path / "generated", "afo_recovery")
    shared, cfg, seed = _case("afo_recovery", tmp_path / "replay")
    ctx = _ScenarioContext(cfg, shared, logging.getLogger("phase6.replay"), seed)
    coupler = AnnualBmiScenarioCoupler(ctx, cfg, ctx.logger, scenario_id=1, seed=seed, replay=replay)
    coupler.initialize()
    try:
        actual = []
        while True:
            value = coupler.next_bmp()
            if value is None:
                break
            actual.append(value.bmp_record)
        assert len(actual) == len(replay.placements)
        assert coupler.completed_scenario() == replay
    finally:
        coupler.finalize()
