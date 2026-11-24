"""Phase 7: CSDMS BMI 2.0 surface coverage and safe coupling contracts.

Tests never change scientific algorithms or mutate on-disk example inputs.
"""
from __future__ import annotations

import json
import hashlib
import logging

import numpy as np
import pytest

from basin_bmp_sim.execution.bmi_coupling import (
    AnnualBmiScenarioCoupler, get_bmi_value, validate_bmi_index_contract,
)
from basin_bmp_sim.execution.monte_carlo_driver import _ScenarioContext
from basin_bmp_sim.interfaces.bmi.array_component import BmiArrayComponent
from basin_bmp_sim.interfaces.bmi.live_scenario_generator_bmi import LiveScenarioGeneratorBmi
from basin_bmp_sim.interfaces.bmi.watershed_model_bmi import WatershedModelBmi
from basin_bmp_sim.interfaces.domain_index_contract import index_contract_values, index_manifest
from basin_bmp_sim.interfaces.exchange_variables import ExchangeVariable, EXCHANGE_SCHEMA_VERSION
from test_phase2_scenario_generation import _case
from basin_bmp_sim.science.model_context import ScenarioModelContext as _ScenarioModelContext


# Required BMI 2.0 methods (not the optional get_bmi_version extension).
BMI_2_0_METHODS = (
    "initialize", "update", "update_until", "finalize",
    "get_component_name", "get_input_item_count", "get_output_item_count",
    "get_input_var_names", "get_output_var_names", "get_var_grid",
    "get_var_type", "get_var_units", "get_var_itemsize", "get_var_nbytes",
    "get_var_location", "get_current_time", "get_start_time",
    "get_end_time", "get_time_units", "get_time_step", "get_value",
    "get_value_ptr", "get_value_at_indices", "set_value",
    "set_value_at_indices", "get_grid_rank", "get_grid_size",
    "get_grid_type", "get_grid_shape", "get_grid_spacing",
    "get_grid_origin", "get_grid_x", "get_grid_y", "get_grid_z",
    "get_grid_node_count", "get_grid_edge_count", "get_grid_face_count",
    "get_grid_edge_nodes", "get_grid_face_edges", "get_grid_face_nodes",
    "get_grid_nodes_per_face",
)


class _ArrayFixture(BmiArrayComponent):
    def __init__(self):
        super().__init__((
            ExchangeVariable("test__input", "1", "int64", "input", rank=1, resizable=True),
            ExchangeVariable("test__output", "1", "float64", rank=1),
            ExchangeVariable("test__scalar", "1", "int64", rank=0),
        ), name="BMI conformance test")

    def initialize(self, config_file=""):
        self._initialized = True
        self._publish({"test__input": [1, 2, 3], "test__output": [1.5, 2.5, 3.5], "test__scalar": 1})
        self._end_time = 1.0

    def update(self):
        self._time = 1.0


def test_required_bmi2_surface_and_item_counts():
    from basin_bmp_sim.interfaces.bmi import (
        MonteCarloDriverBmi, StochasticScenarioBmi,
        LiveScenarioGeneratorBmi, WatershedModelBmi,
    )
    for cls in (MonteCarloDriverBmi, StochasticScenarioBmi,
                LiveScenarioGeneratorBmi, WatershedModelBmi):
        for method in BMI_2_0_METHODS:
            assert callable(getattr(cls, method, None)), f"{cls.__name__}.{method}"

    component = _ArrayFixture()
    component.initialize()
    assert component.get_input_item_count() == len(component.get_input_var_names()) == 1
    assert component.get_output_item_count() == len(component.get_output_var_names()) == 2
    assert component.get_grid_type(0) == "scalar"
    assert component.get_grid_rank(0) == 0
    component.get_grid_shape(0, np.empty(0, dtype=np.int64))
    with pytest.raises(NotImplementedError):
        component.get_grid_face_edges(component.get_var_grid("test__output"), np.empty(0, dtype=np.int64))
    component.finalize()


def test_indexed_getter_setter_and_dynamic_extent():
    component = _ArrayFixture()
    component.initialize()
    basin_bmp_sim = np.array([7, 8], dtype=np.int64)
    component.set_value_at_indices("test__input", np.array([0, 2], dtype=np.int64), basin_bmp_sim)
    values = np.empty(3, dtype=np.int64)
    component.get_value("test__input", values)
    np.testing.assert_array_equal(values, [7, 2, 8])
    selected = np.empty(2, dtype=np.int64)
    component.get_value_at_indices("test__input", selected, np.array([2, 0], dtype=np.int64))
    np.testing.assert_array_equal(selected, [8, 7])
    # A previously retrieved pointer is invalid after a variable-size update.
    component.set_value("test__input", np.array([4, 5], dtype=np.int64))
    assert component.get_grid_size(component.get_var_grid("test__input")) == 2
    assert component.get_var_nbytes("test__input") == 16
    component.finalize()


@pytest.mark.parametrize("bad_indices", [
    [-1], [3], [0, 0], [0.25],
])
def test_invalid_indices_are_rejected(bad_indices):
    component = _ArrayFixture()
    component.initialize()
    with pytest.raises(ValueError, match="indices"):
        component.set_value_at_indices(
            "test__input", np.asarray(bad_indices),
            np.zeros(len(bad_indices), dtype=np.int64),
        )
    component.finalize()


def test_bmi_array_inputs_are_not_silently_truncated_or_wrapped():
    component = _ArrayFixture()
    component.initialize()
    for bad in (np.array([1.7, 2.0], dtype=np.float64),
                np.array([-1], dtype=np.int32),
                np.array([1, 2], dtype=np.uint64)):
        with pytest.raises(ValueError, match="dtype"):
            component.set_value("test__input", bad)
    with pytest.raises(ValueError, match="read-only"):
        component.set_value_at_indices("test__output", np.array([0]), np.array([0.0]))
    values = np.empty(3, dtype=np.int64)
    component.get_value("test__input", values)
    np.testing.assert_array_equal(values, [1, 2, 3])
    component.finalize()


def test_getter_does_not_silently_write_into_copy_of_destination():
    component = _ArrayFixture()
    component.initialize()
    contiguous = np.empty(6, dtype=np.float64)[::2]
    assert not contiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="contiguous"):
        component.get_value("test__output", contiguous)
    with pytest.raises(ValueError, match="dtype"):
        component.get_value("test__output", np.empty(3, dtype=np.int64))
    with pytest.raises(ValueError, match="indices"):
        component.get_value_at_indices("test__output", np.empty(1), np.array([-1]))
    component.finalize()


def _ctx(case, tmp_path):
    shared, cfg, seed = _case(case, tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger("phase7.index"), seed)
    return ctx, cfg, seed


def test_domain_manifest_is_canonical_and_exposes_index_order(tmp_path):
    ctx, _, _ = _ctx("mixed_budget", tmp_path)
    data = index_contract_values(ctx)
    manifest = json.loads(bytes(data["coupling__index_manifest_utf8"]).decode("utf8"))
    assert manifest["schema_version"] == 1
    assert manifest["parcel_ids"] == [str(x) for x in ctx.parcel_ids]
    assert manifest["outlet_oids"] == [str(x) for x in ctx.outlet_oids]
    assert manifest["pollutants"] == [str(x) for x in ctx.pollutants]
    assert manifest["pathway_names"] == [str(x) for x in ctx.pathway_names]
    assert data["coupling__index_fingerprint"].size == 32
    assert data["coupling__exchange_schema_version"].tolist() == [EXCHANGE_SCHEMA_VERSION]
    assert index_manifest(ctx) == bytes(data["coupling__index_manifest_utf8"])
    # A changed index order must produce a different manifest and fingerprint.
    if len(ctx.parcel_ids) > 1:
        copied = object.__new__(_ScenarioModelContext)
        copied.__dict__.update(ctx.__dict__)
        copied.parcel_ids = list(reversed(ctx.parcel_ids))
        assert not np.array_equal(data["coupling__index_fingerprint"],
                                  index_contract_values(copied)["coupling__index_fingerprint"])



def test_coupler_rejects_inconsistent_domain_before_first_placement(tmp_path, monkeypatch):
    ctx, cfg, seed = _ctx("mixed_budget", tmp_path)
    coupler = AnnualBmiScenarioCoupler(ctx, cfg, ctx.logger, scenario_id=1, seed=seed)
    original = LiveScenarioGeneratorBmi.initialize

    def initialize_with_mismatch(self, config_file=""):
        original(self, config_file)
        manifest = json.loads(bytes(self._values["coupling__index_manifest_utf8"]).decode("utf8"))
        manifest["parcel_ids"][0] += "_other"
        data = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf8")
        self._values["coupling__index_manifest_utf8"] = np.frombuffer(data, dtype=np.uint8).copy()
        self._values["coupling__index_fingerprint"] = np.frombuffer(hashlib.sha256(data).digest(),
                                                                      dtype=np.uint8).copy()

    monkeypatch.setattr(LiveScenarioGeneratorBmi, "initialize", initialize_with_mismatch)
    try:
        with pytest.raises(ValueError, match="index-domain mismatch"):
            coupler.initialize()
        assert coupler.model.get_current_time() == 0
        assert coupler.generator.get_current_time() == 0
    finally:
        coupler.generator.finalize()
        coupler.model.finalize()


def test_coupler_rejects_unsupported_schema_version(tmp_path, monkeypatch):
    ctx, cfg, seed = _ctx("mixed_budget", tmp_path)
    coupler = AnnualBmiScenarioCoupler(ctx, cfg, ctx.logger, scenario_id=1, seed=seed)
    original = LiveScenarioGeneratorBmi.initialize

    def initialize_with_bad_version(self, config_file=""):
        original(self, config_file)
        self._values["coupling__exchange_schema_version"][0] = 987

    monkeypatch.setattr(LiveScenarioGeneratorBmi, "initialize", initialize_with_bad_version)
    try:
        with pytest.raises(ValueError, match="schema version"):
            coupler.initialize()
    finally:
        coupler.generator.finalize()
        coupler.model.finalize()


def test_bmi_index_contract_matches_in_real_coupler(tmp_path):
    ctx, cfg, seed = _ctx("afo_recovery", tmp_path)
    coupler = AnnualBmiScenarioCoupler(ctx, cfg, ctx.logger, scenario_id=1, seed=seed)
    coupler.initialize()
    try:
        validate_bmi_index_contract(coupler.generator, coupler.model)
        for name in (
            "coupling__exchange_schema_version", "coupling__index_manifest_utf8",
            "coupling__index_fingerprint",
        ):
            np.testing.assert_array_equal(get_bmi_value(coupler.generator, name),
                                          get_bmi_value(coupler.model, name))
        assert coupler.next_bmp() is not None
    finally:
        coupler.finalize()


def test_generator_recovery_indexed_write_rejects_invalid_value_without_mutating(tmp_path):
    ctx, cfg, seed = _ctx("afo_recovery", tmp_path)
    generator = LiveScenarioGeneratorBmi(ctx, cfg, scenario_id=1, seed=seed)
    generator.initialize()
    generator.update()
    try:
        generator.set_value("placement__recovery_gains", np.array([0.2], dtype=np.float64))
        with pytest.raises(ValueError, match="recovery gains"):
            generator.set_value_at_indices(
                "placement__recovery_gains", np.array([0], dtype=np.int64),
                np.array([2.0], dtype=np.float64),
            )
        np.testing.assert_array_equal(get_bmi_value(generator, "placement__recovery_gains"), [0.2])
    finally:
        generator._pending = None
        generator.finalize()


def test_model_cost_indexed_write_respects_transaction_phase(tmp_path):
    ctx, cfg, seed = _ctx("mixed_budget", tmp_path)
    model = WatershedModelBmi(ctx, ctx.logger)
    model.initialize()
    try:
        with pytest.raises(RuntimeError, match="after scientific model update"):
            model.set_value_at_indices(
                "placement__cost", np.array([0], dtype=np.int64),
                np.array([8.0], dtype=np.float64),
            )
        np.testing.assert_array_equal(get_bmi_value(model, "placement__cost"), [0.0])
    finally:
        model.finalize()


def test_indexed_getter_can_read_repeated_valid_positions():
    c = _ArrayFixture()
    c.initialize()
    dest = np.empty(3, dtype=np.float64)
    c.get_value_at_indices("test__output", dest, np.array([1, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(dest, [2.5, 2.5, 1.5])
    c.finalize()
