"""Phase 15: multi-strategy, signed annual decision support (no model reruns)."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from src.driver.ensemble_comparison import (
    StrategyComparisonConfig, _load_and_compare, _pareto_flags,
    write_strategy_comparison,
)
from src.driver.ensemble_analysis import DecisionSupportConfig
from test_phase14_ensemble_decision_support import _analysis_config, _synthetic_ensemble


def _strategy_data(tmp_path, *, mode="descriptive", confirm=False, model_id=None,
                   second_cost=50, second_mass=20, with_third=False):
    one = _synthetic_ensemble()
    two = deepcopy(one)
    for row, original in zip(two["scenario_summaries"], two["scenarios"]):
        row["total_cost_usd"] = second_cost
        original["bmp_responses"][-1]["cumulative_cost_usd"] = second_cost
        row["outlet_removed_mass_kg"][0] = second_mass
        original["outlet_removed_mass_kg"][0] = second_mass
    ensembles = [one, two]
    if with_third:
        third = deepcopy(two)
        for row, original in zip(third["scenario_summaries"], third["scenarios"]):
            row["total_cost_usd"] = 10
            original["bmp_responses"][-1]["cumulative_cost_usd"] = 10
            row["outlet_removed_mass_kg"][0] = -1
            original["outlet_removed_mass_kg"][0] = -1
        ensembles.append(third)
    items = []
    for i, ensemble in enumerate(ensembles):
        name = ["Current", "Improved", "Low-cost"][i]
        filename = tmp_path / f"ensemble_{i}.json"
        filename.write_text(json.dumps(ensemble), encoding="utf-8")
        items.append({"name": name, "ensemble_result": filename.name})
    selection = {
        "schema_version": 1, "strategies": items, "primary_target": "N reduction",
        "comparison_mode": mode,
        "scientific_comparability_confirmed": confirm,
    }
    if model_id is not None:
        selection["scientific_model_id"] = model_id
    config = tmp_path / "strategies.json"
    config.write_text(json.dumps(selection), encoding="utf-8")
    targets = _analysis_config(tmp_path, budget_usd=100)
    return config, targets, [tmp_path / item["ensemble_result"] for item in items]


def test_descriptive_comparison_signed_removal_cost_target_probability_and_provenance(tmp_path):
    selection, targets, paths = _strategy_data(tmp_path)
    dest = tmp_path / "summary.json"
    write_strategy_comparison(selection, targets, dest)
    result = json.loads(dest.read_text())
    assert result["comparison_schema_version"] == 1
    assert result["pareto_frontiers"] is None  # no unverified scientific ranking
    assert result["budget_usd"] == 100
    a, b = result["strategies"]
    assert a["mean_implementation_cost_usd"] == 90
    assert b["mean_implementation_cost_usd"] == 50
    assert a["targets"][0]["signed_removed_mass_kg"]["mean"] == 4
    assert b["targets"][0]["signed_removed_mass_kg"]["mean"] == 20
    assert a["targets"][0]["attainment"]["probability"] == 0.25
    assert b["targets"][0]["attainment"]["probability"] == 1.0
    assert a["targets"][0]["cost_effectiveness"]["undefined_zero_or_negative_reduction_scenarios"] == 2
    assert a["joint_all_targets"]["count"] == 1
    assert a["source_sha256"] == sha256(paths[0].read_bytes()).hexdigest()
    assert b["source_sha256"] == sha256(paths[1].read_bytes()).hexdigest()
    assert paths[0].read_bytes() == json.dumps(_synthetic_ensemble()).encode("utf-8")


def test_comparable_mode_requires_explicit_assertion_and_reports_mean_pareto(tmp_path):
    config, targets, _ = _strategy_data(tmp_path, mode="comparable", confirm=True,
                                         model_id="same-scientific-model:v1", with_third=True)
    output = tmp_path / "comparison.json"
    write_strategy_comparison(config, targets, output)
    report = json.loads(output.read_text())
    frontier = report["pareto_frontiers"]["N reduction"]
    assert frontier["mean_cost_vs_mean_annual_kg"] == {
        "Current": False, "Improved": True, "Low-cost": True,
    }
    assert frontier["mean_cost_vs_attainment_probability"] == {
        "Current": False, "Improved": True, "Low-cost": True,
    }
    assert report["scientific_model_id"] == "same-scientific-model:v1"
    assert "Unpaired" in report["interpretation"]["comparison"]


def test_pareto_tradeoffs_ties_and_signed_values():
    assert _pareto_flags([20, 10, 10, 50], [-5, -5, -5, 5]) == [False, True, True, True]
    assert _pareto_flags([0, 0], [0, 0]) == [True, True]


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(schema_version=3), "schema"),
    (lambda d: d.update(extra=5), "unknown"),
    (lambda d: d.update(strategies=d["strategies"][:1]), "two strategies"),
    (lambda d: d["strategies"][1].update(name="Current"), "Duplicate strategy"),
    (lambda d: d["strategies"][1].update(ensemble_result="ensemble_0.json"), "distinct ensemble"),
    (lambda d: d.update(comparison_mode="invalid"), "comparison_mode"),
    (lambda d: d.update(scientific_comparability_confirmed=1), "boolean"),
    (lambda d: d.update(primary_target="unknown"), "primary_target"),
    (lambda d: d.update(comparison_mode="comparable"), "confirmed scientific comparability"),
    (lambda d: d.update(scientific_comparability_confirmed=True), "only valid in comparable"),
])
def test_invalid_comparison_configuration_does_not_replace_published_result(tmp_path, mutation, match):
    config, target, _ = _strategy_data(tmp_path)
    raw = json.loads(config.read_text())
    mutation(raw)
    config.write_text(json.dumps(raw))
    output = tmp_path / "previous.json"
    output.write_bytes(b"prior report")
    with pytest.raises(ValueError, match=match):
        write_strategy_comparison(config, target, output)
    assert output.read_bytes() == b"prior report"


def test_comparable_refuses_mixed_backends_or_index_manifest(tmp_path):
    selection, targets, paths = _strategy_data(tmp_path, mode="comparable", confirm=True,
                                                model_id="same:v1")
    modified = json.loads(paths[1].read_text())
    modified["backend"] = "native"
    for scenario in modified["scenarios"]:
        scenario["backend"] = "native"
    paths[1].write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="identical backend"):
        write_strategy_comparison(selection, targets, tmp_path / "compare.json")
    modified["backend"] = "external"
    for scenario in modified["scenarios"]:
        scenario["backend"] = "external"
    modified["index_manifest"]["parcel_ids"] = ["different-parcel"]
    # Preserve internal agreement after testing a different spatial domain.
    for scenario in modified["scenarios"]:
        scenario["index_manifest"] = modified["index_manifest"]
    paths[1].write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="full index manifest"):
        write_strategy_comparison(selection, targets, tmp_path / "compare.json")


def test_descriptive_allows_mixed_backends_with_matching_pollutants_outlets(tmp_path):
    selection, targets, paths = _strategy_data(tmp_path)
    second = json.loads(paths[1].read_text())
    second["backend"] = "native"
    for scenario in second["scenarios"]:
        scenario["backend"] = "native"
    paths[1].write_text(json.dumps(second))
    output = tmp_path / "comparison.json"
    write_strategy_comparison(selection, targets, output)
    result = json.loads(output.read_text())
    assert [item["backend"] for item in result["strategies"]] == ["external", "native"]
    assert result["pareto_frontiers"] is None


def test_rejects_mismatched_identity_and_corrupt_source_without_clobbering(tmp_path):
    selection, targets, paths = _strategy_data(tmp_path)
    output = tmp_path / "comparison.json"
    output.write_bytes(b"previous valid")
    second = json.loads(paths[1].read_text())
    second["index_manifest"]["outlet_oids"] = ["wrong", "b"]
    for scenario in second["scenarios"]:
        scenario["index_manifest"] = second["index_manifest"]
    paths[1].write_text(json.dumps(second))
    with pytest.raises(ValueError, match="pollutant or outlet"):
        write_strategy_comparison(selection, targets, output)
    assert output.read_bytes() == b"previous valid"
    paths[1].write_text("{not-json")
    with pytest.raises(ValueError, match="Invalid ensemble JSON"):
        write_strategy_comparison(selection, targets, output)
    assert output.read_bytes() == b"previous valid"


def test_chart_creation_is_deterministic_and_xml_escaped(tmp_path):
    selection, targets, paths = _strategy_data(tmp_path)
    raw = json.loads(selection.read_text())
    raw["strategies"][0]["name"] = '<script>alert("x")</script>'
    selection.write_text(json.dumps(raw))
    original_sources = [file.read_bytes() for file in paths]
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    charts = tmp_path / "charts"
    write_strategy_comparison(selection, targets, first, charts_dir=charts)
    before = [p.read_bytes() for p in sorted(charts.glob("*.svg"))]
    write_strategy_comparison(selection, targets, second, charts_dir=charts)
    after = [p.read_bytes() for p in sorted(charts.glob("*.svg"))]
    assert first.read_bytes() == second.read_bytes()
    assert before == after
    assert len(before) == 2
    assert all(b"<svg" in svg and b"</svg>" in svg for svg in before)
    assert all(ET.fromstring(svg).tag.endswith("svg") for svg in before)
    assert all(b"&lt;script&gt;" in svg for svg in before)
    assert all(b"<script>" not in svg for svg in before)
    assert [p.read_bytes() for p in paths] == original_sources


def test_rejects_output_collision_with_source_or_chart(tmp_path):
    selection, targets, paths = _strategy_data(tmp_path)
    with pytest.raises(ValueError, match="must not overwrite"):
        write_strategy_comparison(selection, targets, paths[0])
    with pytest.raises(ValueError, match="must not overwrite"):
        write_strategy_comparison(selection, targets, tmp_path / "charts" / "cost_vs_reduction.svg",
                                  charts_dir=tmp_path / "charts")
    with pytest.raises(ValueError, match="must not overwrite"):
        write_strategy_comparison(selection, targets, selection)


def test_comparison_is_post_processing_and_does_not_initialize_models(tmp_path, monkeypatch):
    selection, targets, _ = _strategy_data(tmp_path)
    import src.driver.backend_selection as selector
    monkeypatch.setattr(selector, "run_selected_scenario", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model run")))
    result = tmp_path / "compare.json"
    write_strategy_comparison(selection, targets, result)
    assert result.is_file()


def test_cli_writes_comparison_and_two_charts(tmp_path):
    selection, targets, _ = _strategy_data(tmp_path)
    import subprocess
    import sys
    output, charts = tmp_path / "result.json", tmp_path / "charts"
    completed = subprocess.run([sys.executable, "-m", "src.driver.ensemble_comparison",
                                str(selection), str(targets), str(output),
                                "--charts-dir", str(charts)],
                               cwd=Path(__file__).resolve().parents[1],
                               capture_output=True, text=True, timeout=25)
    assert completed.returncode == 0, completed.stderr
    assert output.exists() and (charts / "cost_vs_reduction.svg").exists()
    assert (charts / "attainment_curves.svg").exists()
