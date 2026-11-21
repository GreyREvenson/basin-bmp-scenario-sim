"""Phase 20: publicly negotiated lifecycle and replay exchange.

These tests check the extension protocol without changing scientific draws.
The model-neutral/external backend and legacy couplers are covered by the
existing golden/replay/integration fixtures.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from src.driver.bmi_coupling import AnnualBmiScenarioCoupler, get_bmi_value
from src.driver.monte_carlo_driver import _ScenarioContext
from src.interfaces.coupling_lifecycle import (
    CouplerPhase, LIFECYCLE_SCHEMA_VERSION, LIFECYCLE_VARIABLE,
    REPLAY_VARIABLE, negotiate_lifecycle,
)
from src.interfaces.bmi.live_scenario_generator_bmi import LiveScenarioGeneratorBmi
from test_phase2_scenario_generation import _case
from test_phase9_independent_bmi_initialization import _mocked_config_inputs


def _coupler(tmp_path, name="mixed_budget"):
    shared, cfg, seed = _case(name, tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger("phase20"), seed)
    return AnnualBmiScenarioCoupler(ctx, cfg, ctx.logger, scenario_id=1, seed=seed)


def test_lifecycle_protocol_and_published_replay_do_not_change_rng(tmp_path):
    coupler = _coupler(tmp_path)
    assert coupler.lifecycle_phase == CouplerPhase.CREATED.value
    with pytest.raises(RuntimeError, match="Initialize"):
        coupler.completed_scenario()
    coupler.initialize()
    try:
        assert coupler.lifecycle_phase == "ready"
        negotiate_lifecycle(coupler.generator, coupler.model)
        for c in (coupler.generator, coupler.model):
            assert c.get_var_units(LIFECYCLE_VARIABLE) == "1"
            assert get_bmi_value(c, LIFECYCLE_VARIABLE).tolist() == [LIFECYCLE_SCHEMA_VERSION]
        rows = []
        while (row := coupler.next_bmp()) is not None:
            rows.append(row)
        assert coupler.lifecycle_phase == "exhausted"
        assert coupler.next_bmp() is None
        assert rows
        model_before = get_bmi_value(coupler.model, "coupling__model_rng_state_utf8")
        generator_before = get_bmi_value(coupler.generator, "coupling__generator_rng_state_utf8")
        first = coupler.completed_scenario()
        second = coupler.completed_scenario()
        assert first == second
        assert len(first.placements) == len(rows)
        assert json.loads(get_bmi_value(coupler.generator, REPLAY_VARIABLE).tobytes()) == first.to_dict()
        np.testing.assert_array_equal(model_before, get_bmi_value(coupler.model, "coupling__model_rng_state_utf8"))
        np.testing.assert_array_equal(generator_before, get_bmi_value(coupler.generator, "coupling__generator_rng_state_utf8"))
    finally:
        coupler.finalize()
    assert coupler.lifecycle_phase == "finalized"
    coupler.finalize()  # repeat cleanup must be safe
    with pytest.raises(RuntimeError, match="Initialize"):
        coupler.next_bmp()


def test_incompatible_lifecycle_version_fails_before_sampling_and_cleans_up(monkeypatch, tmp_path):
    coupler = _coupler(tmp_path)
    original = LiveScenarioGeneratorBmi.initialize

    def version_mismatch(self, config_file=""):
        original(self, config_file)
        self._values[LIFECYCLE_VARIABLE][0] = 99

    monkeypatch.setattr(LiveScenarioGeneratorBmi, "initialize", version_mismatch)
    with pytest.raises(ValueError, match="lifecycle schema version"):
        coupler.initialize()
    assert coupler.lifecycle_phase == "finalized"
    assert not coupler.generator._initialized
    assert not coupler.model._initialized
    assert coupler.generator._generator is None
    with pytest.raises(RuntimeError, match="initialized or finalized"):
        coupler.initialize()


def test_missing_replay_exchange_extension_rejected_before_domain_transfer(monkeypatch, tmp_path):
    coupler = _coupler(tmp_path)
    monkeypatch.setattr(coupler.generator, "export_replay", None)
    with pytest.raises(ValueError, match="missing lifecycle extension export_replay"):
        coupler.initialize()
    assert coupler.lifecycle_phase == "finalized"
    assert not coupler.model._initialized


def test_coupler_refuses_invalid_or_misidentified_public_replay(monkeypatch, tmp_path):
    coupler = _coupler(tmp_path, "statistical_multi")
    coupler.initialize()
    try:
        while coupler.next_bmp() is not None:
            pass
        original = coupler.generator.export_replay

        def bad_json(seed):
            original(seed)
            coupler.generator._publish({REPLAY_VARIABLE: np.frombuffer(b"not-json", dtype=np.uint8).copy()})

        monkeypatch.setattr(coupler.generator, "export_replay", bad_json)
        with pytest.raises(ValueError, match="Invalid BMI replay JSON"):
            coupler.completed_scenario()
        monkeypatch.setattr(coupler.generator, "export_replay", original)

        def wrong_id(seed):
            original(seed)
            payload = json.loads(get_bmi_value(coupler.generator, REPLAY_VARIABLE).tobytes())
            payload["scenario_id"] = 999
            coupler.generator._publish({REPLAY_VARIABLE: np.frombuffer(
                json.dumps(payload).encode(), dtype=np.uint8).copy()})

        monkeypatch.setattr(coupler.generator, "export_replay", wrong_id)
        with pytest.raises(ValueError, match="identity or seed mismatch"):
            coupler.completed_scenario()
    finally:
        coupler.finalize()


def test_file_initialized_coupler_keeps_native_context_private(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    coupler = AnnualBmiScenarioCoupler.from_manifest(manifest)
    assert coupler.ctx is None and coupler.generator_context is None
    coupler.initialize()
    try:
        assert coupler.ctx is None and coupler.generator_context is None
        assert coupler.cfg is None and coupler.logger is None
        assert coupler.next_bmp() is not None
        assert coupler.completed_scenario().placements
    finally:
        coupler.finalize()


def test_model_finalized_even_if_generator_cleanup_raises(monkeypatch, tmp_path):
    coupler = _coupler(tmp_path)
    coupler.initialize()
    finalized = []
    original_model_finalize = coupler.model.finalize

    def record_model_finalize():
        finalized.append(True)
        original_model_finalize()

    def fail_generator_finalize():
        raise RuntimeError("generator cleanup failed")

    monkeypatch.setattr(coupler.model, "finalize", record_model_finalize)
    monkeypatch.setattr(coupler.generator, "finalize", fail_generator_finalize)
    with pytest.raises(RuntimeError, match="generator cleanup failed"):
        coupler.finalize()
    assert finalized == [True]
    assert not coupler.model._initialized
    assert coupler.lifecycle_phase == "finalized"
