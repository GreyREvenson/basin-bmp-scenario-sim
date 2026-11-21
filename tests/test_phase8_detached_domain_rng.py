"""Phase 8: independent generator ownership, parcel domain and RNG handoffs."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from src.driver.bmi_coupling import AnnualBmiScenarioCoupler, get_bmi_value
from src.driver.monte_carlo_driver import _ScenarioContext
from src.interfaces.bmi.watershed_model_bmi import WatershedModelBmi
from src.interfaces.domain_exchange import (
    DOMAIN_FIELDS, export_placement_domain, install_placement_domain,
    validate_placement_domain,
)
from src.interfaces.rng_exchange import encode_rng_state, restore_rng_state
from src.interfaces.exchange_variables import EXCHANGE_SCHEMA_VERSION
from test_phase2_scenario_generation import _case, _capture


def _coupler(case, tmp_path, *, replay=None):
    shared, cfg, seed = _case(case, tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger("phase8"), seed)
    return AnnualBmiScenarioCoupler(ctx, cfg, ctx.logger,
                                    scenario_id=1, seed=seed, replay=replay)


@pytest.mark.parametrize("case", ["mixed_budget", "statistical_multi", "afo_recovery"])
def test_generator_has_detached_context_and_bmi_domain(case, tmp_path):
    coupler = _coupler(case, tmp_path)
    assert coupler.ctx is coupler.model._ctx
    assert coupler.generator._ctx is coupler.generator_context
    assert coupler.generator._ctx is not coupler.model._ctx
    assert coupler.generator._ctx.rng is not coupler.model._ctx.rng
    assert coupler.generator._ctx.cfg is not coupler.ctx.cfg
    assert coupler.generator._ctx.data is not coupler.ctx.data
    for name in ("parcel_area_ha", "parcel_perim_m", "selection_source_idxs", "parcel_selection_probs"):
        assert not np.shares_memory(getattr(coupler.generator._ctx, name), getattr(coupler.ctx, name))
    coupler.initialize()
    try:
        assert EXCHANGE_SCHEMA_VERSION == 2
        for name in DOMAIN_FIELDS:
            np.testing.assert_array_equal(get_bmi_value(coupler.model, name),
                                          get_bmi_value(coupler.generator, name))
        np.testing.assert_array_equal(coupler.generator._ctx.parcel_area_ha, coupler.ctx.parcel_area_ha)
        assert coupler.generator._ctx.parcel_up_idxs == coupler.ctx.parcel_up_idxs
        assert coupler.generator._domain_ready
        first = coupler.next_bmp()
        assert first is not None
        # The same logical RNG sequence is shared by explicit BMI state
        # exchange, not by pointing both components at one Generator object.
        np.testing.assert_array_equal(encode_rng_state(coupler.ctx.rng),
                                      encode_rng_state(coupler.generator._ctx.rng))
        if len(coupler.generator._ctx.parcel_area_ha):
            old = float(coupler.ctx.parcel_area_ha[0])
            coupler.generator._ctx.parcel_area_ha[0] += 100.0
            assert float(coupler.ctx.parcel_area_ha[0]) == old
    finally:
        coupler.finalize()


def test_corrupt_domain_rejected_before_placement(tmp_path, monkeypatch):
    coupler = _coupler("mixed_budget", tmp_path)
    original = WatershedModelBmi.initialize

    def broken_domain(self, config_file=""):
        original(self, config_file)
        bad = self._values["domain__upstream_offset"].copy()
        bad[-1] += 1
        self._values["domain__upstream_offset"] = bad

    monkeypatch.setattr(WatershedModelBmi, "initialize", broken_domain)
    try:
        with pytest.raises(ValueError, match="upstream connectivity"):
            coupler.initialize()
        assert coupler.generator.get_current_time() == 0
        assert coupler.model.get_current_time() == 0
    finally:
        coupler.generator.finalize()
        coupler.model.finalize()


def test_domain_validation_rejects_negative_geometry_and_invalid_indices(tmp_path):
    coupler = _coupler("mixed_budget", tmp_path)
    domain = export_placement_domain(coupler.ctx)
    n = len(coupler.ctx.parcel_ids)
    m = len(coupler.ctx.parcel_selection_ids)
    validate_placement_domain(domain, n_parcels=n, n_selection=m)
    for name, value in [
        ("domain__parcel_area_ha", -1.0),
        ("domain__parcel_perimeter_m", np.inf),
    ]:
        corrupted = {k: v.copy() for k, v in domain.items()}
        corrupted[name][0] = value
        with pytest.raises(ValueError, match="geometry"):
            install_placement_domain(coupler.generator_context, corrupted)
        np.testing.assert_array_equal(coupler.generator_context.parcel_area_ha,
                                      coupler.ctx.parcel_area_ha)
    corrupted = {k: v.copy() for k, v in domain.items()}
    corrupted["domain__selection_source_index"][0] = n + 1
    with pytest.raises(ValueError, match="selection indices"):
        validate_placement_domain(corrupted, n_parcels=n, n_selection=m)
    corrupted = {k: v.copy() for k, v in domain.items()}
    corrupted["domain__schema_version"][0] = 100
    with pytest.raises(ValueError, match="schema version"):
        validate_placement_domain(corrupted, n_parcels=n, n_selection=m)


def test_invalid_rng_payload_rejected_without_advancing_either_stream(tmp_path):
    coupler = _coupler("afo_recovery", tmp_path)
    coupler.initialize()
    try:
        before_model = encode_rng_state(coupler.model._ctx.rng)
        before_generator = encode_rng_state(coupler.generator._ctx.rng)
        for bad in (b"invalid json", b'{"schema_version":900,"state":{}}'):
            buf = np.frombuffer(bad, dtype=np.uint8)
            with pytest.raises(ValueError, match="RNG"):
                coupler.model.set_value("coupling__generator_rng_state_utf8", buf)
            with pytest.raises(ValueError, match="RNG"):
                coupler.generator.set_value("coupling__model_rng_state_utf8", buf)
        np.testing.assert_array_equal(encode_rng_state(coupler.model._ctx.rng), before_model)
        np.testing.assert_array_equal(encode_rng_state(coupler.generator._ctx.rng), before_generator)
        # Round-trip does not sample/advance the RNG stream.
        restore_rng_state(coupler.generator._ctx.rng, before_model)
        np.testing.assert_array_equal(encode_rng_state(coupler.generator._ctx.rng), before_model)
    finally:
        coupler.finalize()


def test_recorded_afo_replay_with_detached_rng(tmp_path, monkeypatch):
    _, replay = _capture(monkeypatch, tmp_path / "original", "afo_recovery")
    coupler = _coupler("afo_recovery", tmp_path / "replay", replay=replay)
    coupler.initialize()
    try:
        actual = []
        while True:
            bmp = coupler.next_bmp()
            if bmp is None:
                break
            actual.append(bmp)
        assert len(actual) == len(replay.placements)
        assert coupler.completed_scenario() == replay
    finally:
        coupler.finalize()


def test_domain_cycle_is_rejected():
    domain = {
        "domain__schema_version": np.array([1], dtype=np.int64),
        "domain__parcel_area_ha": np.array([2.0, 3.0], dtype=np.float64),
        "domain__parcel_perimeter_m": np.array([15.0, 17.0], dtype=np.float64),
        "domain__upstream_offset": np.array([0, 1, 2], dtype=np.int64),
        "domain__upstream_index": np.array([1, 0], dtype=np.int64),
        "domain__selection_source_index": np.array([0, 1], dtype=np.int64),
        "domain__selection_probability": np.array([0.5, 0.5], dtype=np.float64),
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_placement_domain(domain, n_parcels=2, n_selection=2)
