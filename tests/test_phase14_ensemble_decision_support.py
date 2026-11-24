"""Phase 14: signed annual targets, cost-effectiveness and optional CLI reports."""
from __future__ import annotations

from hashlib import sha256
import json
import sys

import numpy as np
import pytest

from basin_bmp_sim.execution import cli as run_model
from basin_bmp_sim.execution.backend_selection import run_backend_selection
from basin_bmp_sim.analysis.ensemble_analysis import (
    DecisionSupportConfig, ReductionTarget, analyze_ensemble, write_ensemble_analysis,
)
from basin_bmp_sim.execution.ensemble_runner import ENSEMBLE_SEED_POLICY, ensemble_worker_seeds
from test_phase12_backend_ensembles import _plugin, _selection
from test_phase13_checkpointed_ensembles import _v3
from test_phase9_independent_bmi_initialization import _mocked_config_inputs


def _synthetic_ensemble():
    # Flattened [pollutant, outlet] order; signed removal includes negative.
    removed = [(12, -2, 0, 5), (8, 0, 0, -1), (0, 0, 0, 6), (-4, 0, 0, 12)]
    costs = [100, 60, 0, 200]
    index_manifest = {
        "schema_version": 1, "pollutants": ["N", "P"],
        "outlet_oids": ["a", "b"], "parcel_ids": ["x"],
        "pathway_names": ["surface"], "bmp_cps": [340],
    }
    rows, scenarios = [], []
    for i, (mass, cost) in enumerate(zip(removed, costs)):
        seed = ensemble_worker_seeds(7, 4)[i]
        responses = [{"cumulative_cost_usd": cost}]
        rows.append({
            "scenario_id": 10 + i, "worker_seed": seed, "bmp_count": 1,
            "total_cost_usd": cost, "outlet_removed_mass_kg": list(mass),
        })
        scenarios.append({
            "result_schema_version": 1, "backend": "external",
            "index_manifest": index_manifest, "scenario_id": 10 + i,
            "worker_seed": seed, "outlet_removed_mass_kg": list(mass),
            "bmp_responses": responses,
        })
    return {
        "result_schema_version": 2, "run_type": "ensemble",
        "backend": "external", "base_seed": 7, "first_scenario_id": 10,
        "n_scenarios": 4, "seed_policy": ENSEMBLE_SEED_POLICY,
        "index_manifest": index_manifest, "scenario_summaries": rows,
        "scenarios": scenarios,
    }


def _analysis_config(tmp_path, targets=None, **kwargs):
    if targets is None:
        targets = [
            {"name": "N reduction", "pollutant": "N", "outlet_oid": "a",
             "minimum_removed_mass_kg": 10},
            {"name": "P reduction", "pollutant": "P", "outlet_oid": "b",
             "minimum_removed_mass_kg": 5},
        ]
    path = tmp_path / "analysis_targets.json"
    path.write_text(json.dumps({"schema_version": 1, "targets": targets, **kwargs}), encoding="utf-8")
    return path


def test_target_probability_budget_joint_and_cost_effectiveness(tmp_path):
    config = DecisionSupportConfig.read(_analysis_config(tmp_path, budget_usd=100))
    report = analyze_ensemble(_synthetic_ensemble(), config, source_sha256="abc")
    assert report["source_ensemble_sha256"] == "abc"
    assert report["n_scenarios"] == 4
    assert report["implementation_cost_usd"]["mean"] == 90
    assert report["within_budget"]["count"] == 3
    assert report["within_budget"]["probability"] == .75
    n, p = report["targets"]
    assert n["attainment"]["count"] == 1
    assert n["attainment"]["probability"] == .25
    assert n["attainment_within_budget"]["probability"] == .25
    assert n["conditional_attainment_given_within_budget"] == pytest.approx(1 / 3)
    assert n["signed_removed_mass_kg"]["min"] == -4
    assert n["cost_effectiveness"]["positive_reduction_scenarios"] == 2
    assert n["cost_effectiveness"]["undefined_zero_or_negative_reduction_scenarios"] == 2
    np.testing.assert_allclose(n["cost_effectiveness"]["usd_per_kg_annual_removed"]["median"],
                               np.median([100 / 12, 60 / 8]))
    assert p["attainment"]["count"] == 3
    assert p["cost_effectiveness"]["positive_reduction_scenarios"] == 3
    assert report["joint_all_targets"]["count"] == 1
    assert report["joint_all_targets"]["all_targets_within_budget"]["count"] == 1
    for row in report["targets"]:
        lo, hi = row["attainment"]["wilson_95_mc_interval"]
        assert 0 <= lo <= row["attainment"]["probability"] <= hi <= 1


def test_zero_or_negative_removals_have_undefined_ratio_but_count_as_failures(tmp_path):
    config = DecisionSupportConfig.read(_analysis_config(tmp_path, targets=[{
        "name": "No reduction", "pollutant": "N", "outlet_oid": "b",
        "minimum_removed_mass_kg": 0,
    }]))
    report = analyze_ensemble(_synthetic_ensemble(), config, source_sha256="0")
    target = report["targets"][0]
    assert target["attainment"]["count"] == 3  # equality passes, negative fails
    assert target["cost_effectiveness"]["positive_reduction_scenarios"] == 0
    assert target["cost_effectiveness"]["usd_per_kg_annual_removed"] is None
    assert report["budget_usd"] is None
    assert "attainment_within_budget" not in target


def test_zero_scenarios_within_budget_has_null_conditional_probability(tmp_path):
    ensemble = _synthetic_ensemble()
    for record, summary in zip(ensemble["scenarios"], ensemble["scenario_summaries"]):
        summary["total_cost_usd"] = 1
        record["bmp_responses"][-1]["cumulative_cost_usd"] = 1
    cfg = DecisionSupportConfig.read(_analysis_config(tmp_path, budget_usd=0))
    target = analyze_ensemble(ensemble, cfg, source_sha256="1")["targets"][0]
    assert target["attainment_within_budget"]["count"] == 0
    assert target["conditional_attainment_given_within_budget"] is None


@pytest.mark.parametrize("change,match", [
    ({"budget_usd": -1}, "budget_usd"),
    ({"budget_usd": True}, "finite JSON number"),
    ({"budget_usd": float("nan")}, "finite JSON number"),
    ({"extra": 10}, "unknown fields"),
    ({"schema_version": 2}, "schema version"),
    ({"targets": []}, "nonempty list"),
    ({"targets": [
        {"name": "duplicate", "pollutant": "N", "outlet_oid": "a", "minimum_removed_mass_kg": 1},
        {"name": "duplicate", "pollutant": "P", "outlet_oid": "b", "minimum_removed_mass_kg": 1},
    ]}, "Duplicate"),
    ({"targets": [
        {"name": "negative", "pollutant": "N", "outlet_oid": "a", "minimum_removed_mass_kg": -5},
    ]}, "must be >="),
])
def test_invalid_analysis_configuration_rejected(tmp_path, change, match):
    filename = _analysis_config(tmp_path)
    raw = json.loads(filename.read_text())
    raw.update(change)
    filename.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=match):
        DecisionSupportConfig.read(filename)


@pytest.mark.parametrize("mutation,match", [
    (lambda e: e.update(result_schema_version=1), "Phase 12/13"),
    (lambda e: e.update(seed_policy="different"), "Phase 12/13"),
    (lambda e: e["scenario_summaries"][1].update(worker_seed=1), "seed mismatch"),
    (lambda e: e["scenario_summaries"][0].update(outlet_removed_mass_kg=[1]), "shape"),
    (lambda e: e["scenario_summaries"][0].update(total_cost_usd=-5), "must be >="),
    (lambda e: e["scenario_summaries"][0].update(outlet_removed_mass_kg=[float("inf")] * 4), "finite"),
    (lambda e: e["scenarios"][0].update(outlet_removed_mass_kg=[1] * 4), "summary mismatch"),
    (lambda e: e["index_manifest"].update(pollutants=["N", "N"]), "duplicated"),
    (lambda e: e.pop("scenarios"), "Incomplete ensemble original"),
])
def test_invalid_ensemble_rejected(tmp_path, mutation, match):
    ensemble = _synthetic_ensemble()
    mutation(ensemble)
    cfg = DecisionSupportConfig.read(_analysis_config(tmp_path))
    with pytest.raises(ValueError, match=match):
        analyze_ensemble(ensemble, cfg, source_sha256="0")


def test_unknown_target_prevents_report_overwrite_and_preserves_ensemble(tmp_path):
    ensemble_file = tmp_path / "ensemble.json"
    ensemble_file.write_text(json.dumps(_synthetic_ensemble()))
    original = ensemble_file.read_bytes()
    output = tmp_path / "analysis.json"
    output.write_bytes(b"previous verified analysis")
    cfg = _analysis_config(tmp_path, targets=[{
        "name": "missing", "pollutant": "N", "outlet_oid": "wrong",
        "minimum_removed_mass_kg": 1,
    }])
    with pytest.raises(ValueError, match="Unknown pollutant or outlet"):
        write_ensemble_analysis(ensemble_file, cfg, output)
    assert output.read_bytes() == b"previous verified analysis"
    assert ensemble_file.read_bytes() == original
    with pytest.raises(ValueError, match="differ"):
        write_ensemble_analysis(ensemble_file, cfg, ensemble_file)


def test_analysis_is_repeatable_and_source_hash_is_exact(tmp_path):
    source = tmp_path / "ensemble.json"
    source.write_text(json.dumps(_synthetic_ensemble(), indent=2))
    cfg = _analysis_config(tmp_path, budget_usd=100)
    one, two = tmp_path / "a.json", tmp_path / "b.json"
    write_ensemble_analysis(source, cfg, one)
    write_ensemble_analysis(source, cfg, two)
    assert one.read_bytes() == two.read_bytes()
    assert json.loads(one.read_text())["source_ensemble_sha256"] == sha256(source.read_bytes()).hexdigest()


@pytest.mark.parametrize("backend", ["native", "external"])
def test_real_selected_ensemble_analysis_leaves_science_results_unchanged(monkeypatch, tmp_path, backend):
    manifest, shared, cfg, _ = _mocked_config_inputs(monkeypatch, tmp_path, "afo_recovery")
    if backend == "external":
        _plugin(monkeypatch, shared, cfg, 42)
    selected = _selection(tmp_path, manifest, backend=backend, n=2)
    expected = tmp_path / "normal.json"
    run_backend_selection(selected, expected)
    baseline = expected.read_bytes()
    source = json.loads(baseline)
    pollutant = source["index_manifest"]["pollutants"][0]
    outlet = source["index_manifest"]["outlet_oids"][0]
    cfgpath = _analysis_config(tmp_path, targets=[{
        "name": "first outlet", "pollutant": pollutant,
        "outlet_oid": outlet, "minimum_removed_mass_kg": 0,
    }], budget_usd=100000)
    actual = tmp_path / "with_analysis.json"
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "basin-bmp-sim", "--backend-selection", str(selected), "--coupled-result", str(actual),
        "--analysis-config", str(cfgpath), "--analysis-result", str(report),
    ])
    run_model.main()
    assert actual.read_bytes() == baseline
    answer = json.loads(report.read_text())
    assert answer["backend"] == backend
    assert answer["n_scenarios"] == 2
    assert answer["source_ensemble_sha256"] == sha256(actual.read_bytes()).hexdigest()


def test_checkpointed_ensemble_analysis_is_the_same_as_uncheckpointed(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    v2 = _selection(tmp_path, manifest, n=2)
    original = tmp_path / "v2.json"
    run_backend_selection(v2, original)
    v3 = _v3(tmp_path, manifest, n=2)
    checkpointed = tmp_path / "v3.json"
    run_backend_selection(v3, checkpointed)
    assert checkpointed.read_bytes() == original.read_bytes()
    result = json.loads(original.read_text())
    config = _analysis_config(tmp_path, targets=[{
        "name": "outlet reduction", "pollutant": result["index_manifest"]["pollutants"][0],
        "outlet_oid": result["index_manifest"]["outlet_oids"][0],
        "minimum_removed_mass_kg": 0,
    }])
    report1, report2 = tmp_path / "report1.json", tmp_path / "report2.json"
    write_ensemble_analysis(original, config, report1)
    write_ensemble_analysis(checkpointed, config, report2)
    assert report1.read_bytes() == report2.read_bytes()
    checkpoints = tmp_path / "checkpoints"
    assert len(list(checkpoints.glob("realization_*.json"))) == 2


def test_cli_rejects_analysis_on_single_scenario_or_missing_pair(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    selected = _selection(tmp_path, manifest, n=2)
    cfg = _analysis_config(tmp_path)
    result = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["basin-bmp-sim", "--backend-selection", str(selected),
                                        "--coupled-result", str(result), "--analysis-config", str(cfg)])
    with pytest.raises(SystemExit) as exc:
        run_model.main()
    assert exc.value.code == 1
    assert not result.exists()
    raw = json.loads(selected.read_text())
    raw["schema_version"] = 1
    del raw["ensemble"]
    selected.write_text(json.dumps(raw))
    monkeypatch.setattr(sys, "argv", ["basin-bmp-sim", "--backend-selection", str(selected),
                                        "--coupled-result", str(result), "--analysis-config", str(cfg),
                                        "--analysis-result", str(tmp_path / "report.json")])
    with pytest.raises(SystemExit) as exc:
        run_model.main()
    assert exc.value.code == 1
    assert not result.exists()
