"""Phase 12: deterministic model-neutral ensembles without native driver changes."""
from __future__ import annotations

from dataclasses import replace
import json
import logging
import sys
from types import ModuleType

import numpy as np
import pytest

import run_model
from src.driver.backend_selection import BackendSelection, run_backend_selection, run_selected_scenario
from src.driver.ensemble_runner import (
    ENSEMBLE_SEED_POLICY, ensemble_worker_seeds, summarize_ensemble,
    export_ensemble_replays,
)
from src.driver.monte_carlo_driver import _ScenarioContext
from src.interfaces.backend_capabilities import AnnualBackendCapabilities, ExternalBackendBundle
from src.scenarios.stochastic_scenario_generator import BMPScenario
from test_phase9_independent_bmi_initialization import _mocked_config_inputs
from test_phase10_model_neutral_exchange import IndependentToyBackend, _domain


def _selection(tmp_path, manifest, *, backend="native", n=3, **changes):
    payload = {
        "schema_version": 2, "backend": backend,
        "component_manifest": manifest.name,
        "ensemble": {"n_scenarios": n},
    }
    if backend == "external":
        payload.update({
            "external_factory": "phase12_toy_plugin:make_backend",
            "backend_options": {"response_scale": 1.0},
            "required_features": ["afo_recovery"],
        })
    payload.update(changes)
    path = tmp_path / "selection_ensemble.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class AnyScenarioBackend(IndependentToyBackend):
    def apply_bmp(self, placement, scenario_id, rng):
        assert scenario_id >= 1
        assert self.pending is None
        self.pending = placement
        self.received.append(placement)
        delta = rng.random() * .01
        for parcel_idx, fraction in zip(placement.affected_indices, placement.affected_fractions):
            self.current[parcel_idx, :] -= delta * fraction
            self.outlet[:, 0] += delta * fraction
        return ()


def _plugin(monkeypatch, shared, cfg, seed, *, fail_at_id=None):
    domain = _domain(_ScenarioContext(cfg, shared, logging.getLogger("phase12.backend"), seed))
    module = ModuleType("phase12_toy_plugin")
    calls = []

    def make_backend(*, startup, options):
        if startup.scenario_id == fail_at_id:
            raise RuntimeError("Deliberately failed scenario initialization")
        assert options == {"response_scale": 1.0}
        backend = AnyScenarioBackend(domain)
        calls.append((startup.scenario_id, startup.worker_seed, backend))
        return ExternalBackendBundle(
            domain, backend, AnnualBackendCapabilities(
                supported_cps=domain.bmp_cps,
                pollutants=domain.pollutants,
                pathways=domain.pathway_names,
                features=frozenset({"signed_efficiency", "multi_parcel_treatment", "afo_recovery"}),
            ),
        )

    module.make_backend = make_backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


@pytest.mark.parametrize("invalid,message", [
    ({"n_scenarios": 0}, "positive integer"),
    ({"n_scenarios": -5}, "positive integer"),
    ({"n_scenarios": True}, "positive integer"),
    ({"n_scenarios": 1.5}, "positive integer"),
    ({"n_scenarios": "3"}, "positive integer"),
    ({"n_scenarios": 2, "n_jobs": 4}, "only n_scenarios"),
    ([], "only n_scenarios"),
])
def test_invalid_ensemble_selection_rejected(tmp_path, invalid, message):
    (tmp_path / "model.yaml").write_text("stub: yes\n", encoding="utf-8")
    manifest = tmp_path / "startup.json"
    manifest.write_text(json.dumps({"schema_version": 1, "model_config": "model.yaml",
                                    "scenario_id": 1, "worker_seed": 17}), encoding="utf-8")
    selection = _selection(tmp_path, manifest, ensemble=invalid)
    with pytest.raises(ValueError, match=message):
        BackendSelection.read(selection)


def test_v1_compatibility_and_v2_require_ensemble(tmp_path):
    (tmp_path / "model.yaml").write_text("stub: yes\n", encoding="utf-8")
    manifest = tmp_path / "startup.json"
    manifest.write_text(json.dumps({"schema_version": 1, "model_config": "model.yaml",
                                    "scenario_id": 1, "worker_seed": 17}), encoding="utf-8")
    legacy = _selection(tmp_path, manifest, schema_version=1, ensemble={"n_scenarios": 2})
    with pytest.raises(ValueError, match="known fields"):
        BackendSelection.read(legacy)
    modern = _selection(tmp_path, manifest, ensemble=None)
    with pytest.raises(ValueError, match="requires ensemble"):
        BackendSelection.read(modern)
    valid = _selection(tmp_path, manifest, schema_version=1)
    raw = json.loads(valid.read_text())
    raw.pop("ensemble")
    valid.write_text(json.dumps(raw), encoding="utf-8")
    assert BackendSelection.read(valid).ensemble is None


def test_ensemble_rejects_shared_replay_file(tmp_path):
    (tmp_path / "model.yaml").write_text("stub: yes\n", encoding="utf-8")
    (tmp_path / "replay.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "startup.json"
    manifest.write_text(json.dumps({"schema_version": 1, "model_config": "model.yaml",
                                    "scenario_id": 1, "worker_seed": 17,
                                    "replay_file": "replay.json"}), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot set replay_file"):
        BackendSelection.read(_selection(tmp_path, manifest))


def test_spawned_seeds_are_reproducible_and_not_the_native_stream():
    one = ensemble_worker_seeds(12345, 9)
    assert len(one) == len(set(one)) == 9
    assert one == ensemble_worker_seeds(12345, 9)
    assert one[:3] == ensemble_worker_seeds(12345, 3)
    assert one != ensemble_worker_seeds(12346, 9)
    with pytest.raises(ValueError, match="uint64"):
        ensemble_worker_seeds(True, 3)


@pytest.mark.parametrize("case", ["mixed_budget", "statistical_multi", "afo_recovery"])
def test_native_ensemble_is_reproducible_summarized_and_replayable(monkeypatch, tmp_path, case):
    manifest, _, _, seed = _mocked_config_inputs(monkeypatch, tmp_path, case)
    selection = _selection(tmp_path, manifest, n=3)
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    run_backend_selection(selection, first)
    run_backend_selection(selection, second)
    assert first.read_bytes() == second.read_bytes()
    result = json.loads(first.read_text(encoding="utf-8"))
    assert result["result_schema_version"] == 2
    assert result["n_scenarios"] == 3
    assert result["seed_policy"] == ENSEMBLE_SEED_POLICY
    assert result["base_seed"] == seed
    assert len(result["scenarios"]) == len(result["scenario_summaries"]) == 3
    assert [v["worker_seed"] for v in result["scenarios"]] == list(ensemble_worker_seeds(seed, 3))
    assert [v["scenario_id"] for v in result["scenarios"]] == [1, 2, 3]
    assert len({row["worker_seed"] for row in result["scenarios"]}) == 3
    assert all(row["index_manifest"] == result["index_manifest"] for row in result["scenarios"])
    cost = [row["total_cost_usd"] for row in result["scenario_summaries"]]
    assert result["statistics"]["total_cost_usd"]["mean"] == np.mean(cost)
    outlets = np.array([v["outlet_removed_mass_kg"] for v in result["scenarios"]])
    np.testing.assert_allclose(result["statistics"]["outlet_removed_mass_kg"]["mean"],
                               outlets.mean(axis=0), rtol=0, atol=0)
    for scenario in result["scenarios"]:
        parsed = BMPScenario.from_dict(scenario["replay_scenario"])
        assert parsed.scenario_id == scenario["scenario_id"]
        assert parsed.seed == scenario["worker_seed"]
    extracted = export_ensemble_replays(first, tmp_path / "replay_exports")
    assert len(extracted) == 3
    assert [BMPScenario.load(path).to_dict() for path in extracted] == [
        row["replay_scenario"] for row in result["scenarios"]
    ]
    with pytest.raises(FileExistsError, match="overwrite"):
        export_ensemble_replays(first, tmp_path / "replay_exports")

    # Re-run one realization with exactly the seed and replay artifact the
    # ensemble produced; no native stochastic/scientific state is reused.
    selected = result["scenarios"][1]
    replay_file = tmp_path / "second_replay.json"
    BMPScenario.from_dict(selected["replay_scenario"]).save(replay_file)
    cfg_file = tmp_path / "replay_startup.json"
    cfg_file.write_text(json.dumps({
        "schema_version": 1, "model_config": str(manifest.parent / "watershed.yaml"),
        "scenario_id": selected["scenario_id"], "worker_seed": selected["worker_seed"],
        "replay_file": str(replay_file),
    }), encoding="utf-8")
    replay_result = run_selected_scenario(replace(BackendSelection.read(selection),
                                                  ensemble=None, component_manifest=cfg_file))
    assert replay_result == selected


def test_external_ensemble_uses_fresh_backends_and_cli(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    calls = _plugin(monkeypatch, shared, cfg, seed)
    selection = _selection(tmp_path, manifest, backend="external", n=3)
    output = tmp_path / "external_ensemble.json"
    monkeypatch.setattr(sys, "argv", ["run_model.py", "--backend-selection",
                                       str(selection), "--coupled-result", str(output)])
    run_model.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["backend"] == "external"
    assert [c[0] for c in calls] == [1, 2, 3]
    assert [c[1] for c in calls] == list(ensemble_worker_seeds(seed, 3))
    assert len({id(c[2]) for c in calls}) == 3
    assert all(c[2].finished for c in calls)
    assert len(result["scenarios"]) == 3
    assert all(result["scenarios"][i]["scenario_id"] == i + 1 for i in range(3))


def test_failed_ensemble_does_not_replace_existing_output(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    calls = _plugin(monkeypatch, shared, cfg, seed, fail_at_id=2)
    selection = _selection(tmp_path, manifest, backend="external", n=3)
    dest = tmp_path / "previous.json"
    dest.write_bytes(b"original verified output")
    with pytest.raises(RuntimeError, match="Deliberately failed"):
        run_backend_selection(selection, dest)
    assert dest.read_bytes() == b"original verified output"
    assert calls[0][2].finished
    assert not list(tmp_path.glob("*.tmp"))


def test_cross_scenario_manifest_mismatch_and_signed_statistics():
    manifest = {"parcel_ids": ["p1"], "pollutants": ["TN"], "outlet_oids": ["o1"]}

    expected_seeds = ensemble_worker_seeds(123, 2)
    def make_record(sid, out, cost):
        return {
            "result_schema_version": 1, "backend": "external", "scenario_id": sid,
            "worker_seed": expected_seeds[sid - 1], "index_manifest": manifest.copy(),
            "baseline_load_rate_kg_ha_yr": [2.0],
            "current_load_rate_kg_ha_yr": [2.0 - out],
            "outlet_removed_mass_kg": [out],
            "bmp_responses": [{"cumulative_cost_usd": cost}],
            "replay_scenario": {"scenario_id": sid, "seed": expected_seeds[sid - 1],
                                "placements": [{}]},
        }

    rows = [make_record(1, -2., 10.), make_record(2, 4., 20.)]
    result = summarize_ensemble(rows, backend="external", base_seed=123, first_scenario_id=1)
    assert result["statistics"]["outlet_removed_mass_kg"]["mean"] == [1.0]
    assert result["statistics"]["outlet_removed_mass_kg"]["min"] == [-2.0]
    assert result["statistics"]["outlet_removed_mass_kg"]["std_population"] == [3.0]
    assert result["statistics"]["total_cost_usd"]["median"] == 15.
    rows[1]["index_manifest"] = {**manifest, "parcel_ids": ["p2"]}
    with pytest.raises(ValueError, match="incompatible index manifests"):
        summarize_ensemble(rows, backend="external", base_seed=123, first_scenario_id=1)
