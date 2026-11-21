"""Phase 21: frozen BMI public API and black-box runtime conformance.

No test samples an extra random value, changes scientific methods, or writes
project examples. The official bmipy abstract-method comparison remains an
independently optional test in test_phase7_bmipy_optional.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.constants import DATA_N_SCENARIOS
from src.driver.bmi_coupling import AnnualBmiScenarioCoupler, get_bmi_value
from src.driver.monte_carlo_driver import _ScenarioContext
from src.interfaces.bmi import MonteCarloDriverBmi, StochasticScenarioBmi
from src.interfaces.bmi import public_contract as contract_module
from src.interfaces.bmi.public_contract import (
    BMI_2_0_METHODS, PUBLIC_COMPONENTS, PROJECT_EXTENSIONS, public_contract,
    verify_component, verify_public_contract,
)
from test_phase2_scenario_generation import _case


def _coupler(tmp_path):
    shared, cfg, seed = _case("afo_recovery", tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger("phase21.contract"), seed)
    return AnnualBmiScenarioCoupler(ctx, cfg, ctx.logger, scenario_id=1, seed=seed), shared, cfg


def test_public_api_manifest_is_frozen_and_versioned():
    verify_public_contract()
    spec = public_contract()
    assert spec["bmi_version"] == "2.0"
    assert spec["schema_version"] == 1
    assert spec["exchange_schema_version"] == 2
    assert spec["lifecycle_schema_version"] == 1
    assert len(spec["bmi_methods"]) == len(set(BMI_2_0_METHODS))
    assert set(spec["components"]) == set(PUBLIC_COMPONENTS)
    for role, variables in PUBLIC_COMPONENTS.items():
        manifest = spec["components"][role]
        assert manifest["project_extensions"] == PROJECT_EXTENSIONS[role]
        assert len(manifest["variables"]) == len(variables)
        assert len({v["name"] for v in manifest["variables"]}) == len(variables)


def test_undocumented_exchange_or_lifecycle_change_fails_closed(monkeypatch):
    monkeypatch.setattr(contract_module, "EXCHANGE_SCHEMA_VERSION", 987)
    with pytest.raises(ValueError, match="contract drift.*exchange_schema_version"):
        verify_public_contract()


def test_live_coupling_audit_is_observational_and_survives_bmp_updates(tmp_path):
    coupler, _, _ = _coupler(tmp_path)
    coupler.initialize()
    try:
        generator = coupler.generator
        model = coupler.model
        old_rng = get_bmi_value(generator, "coupling__generator_rng_state_utf8")
        old_model_rng = get_bmi_value(model, "coupling__model_rng_state_utf8")
        old_loads = get_bmi_value(model, "model__parcel_current_load_rate")
        assert verify_component(generator, "live_generator")["checked_variables"] > 25
        assert verify_component(model, "annual_model")["checked_variables"] > 30
        np.testing.assert_array_equal(old_rng, get_bmi_value(generator, "coupling__generator_rng_state_utf8"))
        np.testing.assert_array_equal(old_model_rng, get_bmi_value(model, "coupling__model_rng_state_utf8"))
        np.testing.assert_array_equal(old_loads, get_bmi_value(model, "model__parcel_current_load_rate"))
        assert coupler.next_bmp() is not None
        assert verify_component(generator, "live_generator")["checked_variables"] > 25
        assert verify_component(model, "annual_model")["checked_variables"] > 30
    finally:
        coupler.finalize()


def test_recorded_scenario_and_driver_have_their_own_public_surfaces(tmp_path):
    recorded = StochasticScenarioBmi([])
    recorded.initialize()
    try:
        result = verify_component(recorded, "recorded_scenario")
        assert result["checked_variables"] > 5
    finally:
        recorded.finalize()
    shared, cfg, _ = _case("mixed_budget", tmp_path / "driver")
    model = SimpleNamespace(
        cfg=cfg,
        data={DATA_N_SCENARIOS: 0, "random_seed": 9},
        pollutants=shared["pollutants"], outlet_oids=shared["outlet_oids"],
        _shared_payload=lambda: shared,
    )
    driver = MonteCarloDriverBmi(model)
    driver.initialize()
    try:
        result = verify_component(driver, "monte_carlo_driver")
        assert result["checked_variables"] >= 5
    finally:
        driver.finalize()


class _Proxy:
    def __init__(self, target, **overrides):
        self.target = target
        self.overrides = overrides

    def __getattr__(self, name):
        return self.overrides.get(name, getattr(self.target, name))


def test_auditor_rejects_wrong_variable_unit_and_virtual_coordinates(tmp_path):
    bmi = StochasticScenarioBmi([])
    bmi.initialize()
    try:
        units = lambda name: "invalid" if name == "bmp__cps_code" else bmi.get_var_units(name)
        with pytest.raises(ValueError, match="metadata mismatch.*bmp__cps_code"):
            verify_component(_Proxy(bmi, get_var_units=units), "recorded_scenario")

        def wrong_grid_type(grid):
            return "points" if grid != 0 else "scalar"

        with pytest.raises(ValueError, match="virtual grid type"):
            verify_component(_Proxy(bmi, get_grid_type=wrong_grid_type), "recorded_scenario")

        def wrong_indexed(name, dest, indices):
            bmi.get_value_at_indices(name, dest, indices)
            dest[:] = 0

        # Use a nonempty variable so the corrupted indexed read is observable.
        bmi._publish({"bmp__affected_parcel_offset": [2, 3]})
        with pytest.raises(ValueError, match="indexed value mismatch"):
            verify_component(_Proxy(bmi, get_value_at_indices=wrong_indexed), "recorded_scenario")
    finally:
        bmi.finalize()


def test_auditor_rejects_missing_extension_and_bad_time(tmp_path):
    coupler, _, _ = _coupler(tmp_path)
    coupler.initialize()
    try:
        with pytest.raises(ValueError, match="missing required methods.*export_replay"):
            verify_component(_Proxy(coupler.generator, export_replay=None), "live_generator")
        with pytest.raises(ValueError, match="logical time metadata"):
            verify_component(_Proxy(coupler.model, get_current_time=lambda: -1.0), "annual_model")
    finally:
        coupler.finalize()
