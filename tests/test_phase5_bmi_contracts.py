"""Phase 5: BMI 2.0 variable contracts and behavior-preserving adapters."""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.constants import DATA_N_SCENARIOS, XAXIS_COUNT, YAXIS_TOTAL
from src.driver.monte_carlo_driver import _ScenarioContext, _run_one_scenario
from src.interfaces.bmi import (
    MonteCarloDriverBmi, StochasticScenarioBmi, LiveScenarioGeneratorBmi, WatershedModelBmi,
)
from src.interfaces.exchange_variables import scenario_arrays
from src.scenarios.stochastic_scenario_generator import BMPScenario, StochasticScenarioGenerator
from test_phase2_scenario_generation import _case, _capture


BMI_METHODS = (
    "initialize", "update", "update_until", "finalize", "get_component_name", "get_bmi_version",
    "get_input_var_names", "get_output_var_names", "get_var_grid", "get_var_type",
    "get_var_units", "get_var_itemsize", "get_var_nbytes", "get_var_location",
    "get_current_time", "get_start_time", "get_end_time", "get_time_units", "get_time_step",
    "get_value", "get_value_ptr", "get_value_at_indices", "set_value", "set_value_at_indices",
    "get_grid_rank", "get_grid_size", "get_grid_type", "get_grid_shape", "get_grid_spacing",
    "get_grid_origin", "get_grid_x", "get_grid_y", "get_grid_z", "get_grid_node_count",
    "get_grid_edge_count", "get_grid_face_count", "get_grid_edge_nodes",
    "get_grid_face_nodes", "get_grid_nodes_per_face",
)


def value(bmi, name):
    dest = np.empty_like(bmi.get_value_ptr(name))
    bmi.get_value(name, dest)
    return dest


def assert_bmi_metadata(bmi):
    assert bmi.get_bmi_version() == "2.0"
    assert bmi.get_component_name()
    for method in BMI_METHODS:
        assert callable(getattr(bmi, method)), method
    for name in bmi.get_output_var_names():
        arr = value(bmi, name)
        np.testing.assert_array_equal(arr, bmi.get_value_ptr(name))
        assert bmi.get_var_nbytes(name) == arr.nbytes
        assert bmi.get_var_itemsize(name) == arr.itemsize
        assert bmi.get_var_type(name) == str(arr.dtype)
        assert bmi.get_var_units(name)
        assert bmi.get_var_location(name) == "node"
        grid = bmi.get_var_grid(name)
        assert bmi.get_grid_size(grid) == arr.size
        if grid != 0:
            assert bmi.get_grid_rank(grid) == 1
            shape = np.empty(1, dtype=np.int64)
            bmi.get_grid_shape(grid, shape)
            np.testing.assert_array_equal(shape, [arr.size])
            coords = np.empty(arr.size, dtype=float)
            bmi.get_grid_x(grid, coords)
            np.testing.assert_array_equal(coords, np.arange(arr.size))
            spacing = np.empty(1)
            bmi.get_grid_spacing(grid, spacing)
            assert spacing[0] == 1.0
        with pytest.raises(ValueError, match="read-only"):
            bmi.set_value(name, arr)


@pytest.mark.parametrize("case", ["mixed_budget", "statistical_multi", "afo_recovery"])
def test_recorded_scenario_bmi_matches_saved_replay(tmp_path, monkeypatch, case):
    _, scenario = _capture(monkeypatch, tmp_path / "generated", case)
    b = StochasticScenarioBmi([scenario])
    b.initialize()
    assert_bmi_metadata(b)
    assert b.get_current_time() == 0.0
    assert b.get_time_units() == "none"
    b.update_until(1)
    assert b.get_current_time() == 1.0
    np.testing.assert_array_equal(value(b, "bmp__cps_code"),
                                  [p.choice.cps for p in scenario.placements])
    encoded = scenario_arrays(scenario)
    for name, expected in encoded.items():
        np.testing.assert_array_equal(value(b, name), expected)
    assert len(value(b, "bmp__affected_parcel_offset")) == len(scenario.placements) + 1
    with pytest.raises(RuntimeError, match="No remaining"):
        b.update()
    b.finalize()
    with pytest.raises(RuntimeError, match="initialize"):
        value(b, "scenario__bmp_count")


def test_scenario_file_bmi_and_csr_multi_parcel(tmp_path, monkeypatch):
    _, scenario = _capture(monkeypatch, tmp_path / "outputs", "mixed_budget")
    assert any(p.design.family == "wetland" for p in scenario.placements)
    b = StochasticScenarioBmi()
    b.initialize(str(tmp_path / "outputs" / "scenario_definitions"))
    b.update()
    offsets = value(b, "bmp__affected_parcel_offset")
    idx = value(b, "bmp__affected_parcel_index")
    frac = value(b, "bmp__affected_parcel_fraction")
    for i, p in enumerate(scenario.placements):
        np.testing.assert_array_equal(idx[offsets[i]:offsets[i+1]],
                                      [pair[0] for pair in p.design.treated])
        np.testing.assert_array_equal(frac[offsets[i]:offsets[i+1]],
                                      [pair[1] for pair in p.design.treated])
    b.finalize()


@pytest.mark.parametrize("case", ["mixed_budget", "statistical_multi", "afo_recovery"])
def test_watershed_bmi_uses_existing_scientific_kernels(tmp_path, case):
    shared, cfg, seed = _case(case, tmp_path)
    logger = logging.getLogger(f"phase5.model.{case}")
    ctx = _ScenarioContext(cfg, shared, logger, seed)
    model = WatershedModelBmi(ctx, logger)
    model.initialize()
    assert_bmi_metadata(model)
    np.testing.assert_array_equal(value(model, "model__parcel_baseline_load_rate"),
                                  model.native_model.baseline_load_rates.ravel())
    generator = StochasticScenarioGenerator(ctx, cfg)
    placement = generator.next_realization(ctx)
    assert placement is not None
    model.set_placement(placement, scenario_id=1)
    model.update_until(1)
    completed = generator.complete_placement(
        ctx, placement, recovery_gains=model.pending_recovery_gains,
    )
    record = model.complete_pending(completed.cost_usd)
    assert record["cost_usd"] == completed.cost_usd
    np.testing.assert_array_equal(value(model, "model__parcel_current_load_rate"),
                                  model.native_model.load_rates.ravel())
    expected = [model.native_model.delivered_reductions[p][o]
                for p in ctx.pollutants for o in ctx.outlet_oids]
    np.testing.assert_allclose(value(model, "model__outlet_removed_mass"), expected, rtol=0, atol=0)
    assert_bmi_metadata(model)
    with pytest.raises(RuntimeError, match="Queue a placement"):
        model.update()
    model.finalize()


def test_driver_bmi_reproduces_native_worker_trajectory(tmp_path, monkeypatch):
    from src.driver import monte_carlo_driver as module
    payload, cfg, _ = _case("mixed_budget", tmp_path / "driver")
    written = {}

    def fake_parquet(df, path, *, logger):
        written[str(path)] = df.copy()

    monkeypatch.setattr(module, "_write_parquet_atomic", fake_parquet)
    model = SimpleNamespace(
        cfg=cfg,
        data={DATA_N_SCENARIOS: 1, "random_seed": 99},
        pollutants=payload["pollutants"],
        outlet_oids=payload["outlet_oids"],
        logger=logging.getLogger("phase5.driver"),
        _shared_payload=lambda: payload,
    )
    bmi = MonteCarloDriverBmi(model)
    bmi.initialize()
    assert_bmi_metadata(bmi)
    assert bmi.get_end_time() == 1
    with pytest.raises(ValueError, match="integer"):
        bmi.update_until(0.5)
    bmi.update()
    assert int(value(bmi, "ensemble__completed_scenario_count")[0]) == 1
    assert_bmi_metadata(bmi)
    reductions = value(bmi, "scenario__outlet_removed_mass")
    captured_seed = bmi._child_seeds[0]
    native = _run_one_scenario(payload, cfg, 0, captured_seed, tmp_path / "direct")
    expected = np.array([
        native.get((pol, oid, XAXIS_COUNT, YAXIS_TOTAL), [(1, 0, 0)])[-1][2]
        for pol in payload["pollutants"] for oid in payload["outlet_oids"]
    ])
    np.testing.assert_array_equal(reductions, expected)
    replay1 = BMPScenario.load(tmp_path / "driver" / "scenario_definitions" / "s1.json")
    replay2 = BMPScenario.load(tmp_path / "direct" / "scenario_definitions" / "s1.json")
    assert replay1 == replay2
    monkeypatch.setattr("src.driver.results.trajectory_output_manager.collect_and_write_trajectories", lambda *a, **kw: {})
    bmi.finalize()


def test_bmi_rejects_invalid_writes_and_unknown_variables(tmp_path, monkeypatch):
    _, scenario = _capture(monkeypatch, tmp_path / "generated", "afo_recovery")
    bmi = StochasticScenarioBmi([scenario])
    with pytest.raises(RuntimeError, match="initialize"):
        bmi.get_value_ptr("bmp__cps_code")
    bmi.initialize()
    with pytest.raises(KeyError, match="Unknown BMI variable"):
        bmi.get_var_type("invented__field")
    with pytest.raises(ValueError, match="read-only"):
        bmi.set_value_at_indices("scenario__bmp_count", [0], [7])
    bmi.update()
    with pytest.raises(ValueError, match="destination"):
        bmi.get_value("bmp__cps_code", np.empty(50, dtype=np.int64))
    bmi.finalize()


@pytest.mark.parametrize("case", ["mixed_budget", "statistical_multi", "afo_recovery"])
def test_live_generator_bmi_matches_native_rng_and_scientific_response(tmp_path, case):
    """The two adapters retain the exact interleaved generator/model RNG order."""
    from src.models.assemblies.annual_model_realization import AnnualModelRealization
    payload, cfg, seed = _case(case, tmp_path)
    logger = logging.getLogger("p5.live." + case)

    ctx_native = _ScenarioContext(cfg, payload, logger, seed)
    native_model = AnnualModelRealization.initialize(ctx_native, logger)
    native_gen = StochasticScenarioGenerator(ctx_native, cfg)
    native_plan = native_gen.next_realization(ctx_native)
    native_pending = native_model.apply_placement(native_plan, 1)
    native_done = native_gen.complete_placement(
        ctx_native, native_plan, recovery_gains=native_pending.recovery_gains,
    )
    native_record = native_model.complete_placement(native_pending, native_done.cost_usd)

    ctx_bmi = _ScenarioContext(cfg, payload, logger, seed)
    model_bmi = WatershedModelBmi(ctx_bmi, logger)
    model_bmi.initialize()
    gen_bmi = LiveScenarioGeneratorBmi(ctx_bmi, cfg, seed=seed)
    gen_bmi.initialize()
    assert_bmi_metadata(gen_bmi)
    gen_bmi.update()
    np.testing.assert_array_equal(value(gen_bmi, "placement__affected_parcel_index"),
                                  [i for i, f in native_plan.design.treated])
    np.testing.assert_array_equal(value(gen_bmi, "placement__affected_parcel_fraction"),
                                  [f for i, f in native_plan.design.treated])
    assert int(value(gen_bmi, "placement__cps_code")[0]) == native_plan.choice.cps
    assert gen_bmi.pending_placement == native_plan
    with pytest.raises(RuntimeError, match="pending placement"):
        gen_bmi.update()
    model_bmi.set_placement(gen_bmi.pending_placement)
    model_bmi.update()
    completed = gen_bmi.complete_pending(recovery_gains=model_bmi.pending_recovery_gains)
    bmi_record = model_bmi.complete_pending(completed.cost_usd)
    assert completed == native_done
    assert bmi_record == native_record
    np.testing.assert_array_equal(value(model_bmi, "model__parcel_current_load_rate"),
                                  native_model.load_rates.ravel())
    assert int(value(gen_bmi, "placement__completed")[0]) == 1
    np.testing.assert_array_equal(value(gen_bmi, "scenario__bmp_count"), [1])
    assert gen_bmi.completed_scenario(seed).placements == (native_done,)
    gen_bmi.finalize()
    model_bmi.finalize()
