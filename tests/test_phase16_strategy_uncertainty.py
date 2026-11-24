"""Phase 16: separate, explicitly justified Monte Carlo comparison intervals."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree

import pytest

from basin_bmp_sim.execution.ensemble_runner import ensemble_worker_seeds
from basin_bmp_sim.analysis.ensemble_comparison import StrategyComparisonConfig
from basin_bmp_sim.analysis.ensemble_analysis import DecisionSupportConfig
from basin_bmp_sim.analysis.strategy_uncertainty import (
    UncertaintyConfig, _single_realization_win_probability,
    analyze_strategy_uncertainty, write_strategy_uncertainty,
)
from test_phase14_ensemble_decision_support import _synthetic_ensemble
from test_phase15_strategy_comparison import _strategy_data


def _settings(tmp_path, design="independent", **changes):
    cfg = {
        "schema_version": 1, "sampling_design": design,
        "sampling_design_confirmed": True,
        "n_bootstrap": 400, "random_seed": 91,
        "confidence_level": 0.95,
    }
    if design == "paired":
        cfg["pairing_id"] = "documented-common-draws-v1"
    cfg.update(changes)
    filename = tmp_path / "uncertainty.json"
    filename.write_text(json.dumps(cfg))
    return filename


def _reseed(path: Path, base_seed: int):
    ensemble = json.loads(path.read_text())
    ensemble["base_seed"] = base_seed
    seeds = ensemble_worker_seeds(base_seed, ensemble["n_scenarios"])
    for row, raw, seed in zip(ensemble["scenario_summaries"], ensemble["scenarios"], seeds):
        row["worker_seed"] = seed
        raw["worker_seed"] = seed
    path.write_text(json.dumps(ensemble))


def _independent(tmp_path):
    selection, targets, paths = _strategy_data(
        tmp_path, mode="comparable", confirm=True, model_id="one-model-v1")
    _reseed(paths[1], 9999)
    return selection, targets, paths


def _paired(tmp_path):
    selection, targets, paths = _strategy_data(
        tmp_path, mode="comparable", confirm=True, model_id="one-model-v1")
    one = json.loads(paths[0].read_text())
    two = deepcopy(one)
    # Same underlying scenario draw; strategy B has 5 kg more and USD 10 higher
    # in every matched realization, yielding a degenerate paired bootstrap CI.
    for row, raw in zip(two["scenario_summaries"], two["scenarios"]):
        row["total_cost_usd"] += 10
        raw["bmp_responses"][-1]["cumulative_cost_usd"] += 10
        row["outlet_removed_mass_kg"][0] += 5
        raw["outlet_removed_mass_kg"][0] += 5
    paths[1].write_text(json.dumps(two))
    return selection, targets, paths


def test_independent_bootstrap_differences_and_reproducible_results(tmp_path):
    selection, targets, paths = _independent(tmp_path)
    settings = _settings(tmp_path)
    chart = tmp_path / "intervals.svg"
    out = tmp_path / "results.json"
    write_strategy_uncertainty(selection, targets, settings, out, chart_path=chart)
    result = json.loads(out.read_text())
    assert result["uncertainty_schema_version"] == 1
    assert result["sampling_design"] == "independent"
    assert result["scientific_model_id"] == "one-model-v1"
    assert result["strategies"][0]["source_sha256"] == sha256(paths[0].read_bytes()).hexdigest()
    pair = result["pairwise_differences"][0]
    assert pair["difference_definition"] == "B minus A"
    assert pair["n_a"] == pair["n_b"] == 4
    assert pair["implementation_cost_usd"]["difference_b_minus_a"] == -40.0
    assert pair["targets"][0]["signed_annual_removed_mass_kg"]["difference_b_minus_a"] == 16.0
    assert pair["targets"][0]["attainment_probability"]["difference_b_minus_a"] == .75
    ci = pair["targets"][0]["signed_annual_removed_mass_kg"]["bootstrap_percentile_interval"]
    assert ci[0] < 16 < ci[1]
    assert pair["targets"][0]["signed_annual_removed_mass_kg"]["single_realization_probability_b_greater_with_half_ties"] == 1.0
    assert pair["budget_outcomes"] is not None
    assert pair["joint_attainment_probability"] is not None
    ElementTree.parse(chart)
    assert 'N reduction' in chart.read_text()
    old, old_svg = out.read_bytes(), chart.read_bytes()
    write_strategy_uncertainty(selection, targets, settings, out, chart_path=chart)
    assert out.read_bytes() == old
    assert chart.read_bytes() == old_svg
    assert paths[0].read_bytes() == json.dumps(_synthetic_ensemble()).encode()


def test_paired_bootstrap_preserves_within_draw_differences(tmp_path):
    selection, targets, _ = _paired(tmp_path)
    cfg = _settings(tmp_path, "paired")
    report = analyze_strategy_uncertainty(
        StrategyComparisonConfig.read(selection), DecisionSupportConfig.read(targets),
        UncertaintyConfig.read(cfg))
    p = report["pairwise_differences"][0]
    mass = p["targets"][0]["signed_annual_removed_mass_kg"]
    assert mass["difference_b_minus_a"] == 5
    assert mass["bootstrap_percentile_interval"] == [5, 5]
    assert mass["single_realization_probability_b_greater_with_half_ties"] == 1
    assert mass["bootstrap_fraction_favoring_b_with_half_ties"] == 1
    cost = p["implementation_cost_usd"]
    assert cost["difference_b_minus_a"] == 10
    assert cost["bootstrap_percentile_interval"] == [10, 10]
    assert cost["bootstrap_fraction_favoring_b_with_half_ties"] == 0
    assert cost["single_realization_probability_b_cheaper_with_half_ties"] == 0


def test_seed_overlap_requires_paired_design_not_an_automatic_inference(tmp_path):
    selection, targets, _ = _paired(tmp_path)
    report = tmp_path / "report.json"
    report.write_bytes(b"previously published")
    with pytest.raises(ValueError, match="non-overlapping worker seeds"):
        write_strategy_uncertainty(selection, targets, _settings(tmp_path), report)
    assert report.read_bytes() == b"previously published"


def test_incorrect_pairing_fails_before_overwriting_report(tmp_path):
    selection, targets, paths = _independent(tmp_path)
    dest = tmp_path / "report.json"
    dest.write_bytes(b"previous report")
    with pytest.raises(ValueError, match="matching ordered scenario IDs and worker seeds"):
        write_strategy_uncertainty(selection, targets, _settings(tmp_path, "paired"), dest)
    assert dest.read_bytes() == b"previous report"


def test_paired_rejects_matching_seeds_if_scenario_ids_differ(tmp_path):
    selection, targets, paths = _paired(tmp_path)
    data = json.loads(paths[1].read_text())
    data["first_scenario_id"] += 1
    for summary, record in zip(data["scenario_summaries"], data["scenarios"]):
        summary["scenario_id"] += 1
        record["scenario_id"] += 1
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="matching ordered scenario IDs"):
        write_strategy_uncertainty(selection, targets, _settings(tmp_path, "paired"),
                                   tmp_path / "report.json")


def test_noncomparable_and_incompatible_backends_rejected(tmp_path):
    selection, targets, paths = _strategy_data(tmp_path)
    with pytest.raises(ValueError, match="explicitly confirmed comparable"):
        write_strategy_uncertainty(selection, targets, _settings(tmp_path), tmp_path / "report.json")
    raw = json.loads(selection.read_text())
    raw.update(comparison_mode="comparable", scientific_comparability_confirmed=True,
               scientific_model_id="only-asserted")
    selection.write_text(json.dumps(raw))
    _reseed(paths[1], 999)
    ensemble = json.loads(paths[1].read_text())
    ensemble["backend"] = "native"
    for scenario in ensemble["scenarios"]:
        scenario["backend"] = "native"
    paths[1].write_text(json.dumps(ensemble))
    with pytest.raises(ValueError, match="identical backend"):
        write_strategy_uncertainty(selection, targets, _settings(tmp_path), tmp_path / "report.json")


@pytest.mark.parametrize("changes,reason", [
    ({"schema_version": 2}, "schema"),
    ({"extra": "unknown"}, "unknown"),
    ({"sampling_design": "unclear"}, "sampling_design"),
    ({"sampling_design_confirmed": False}, "explicitly true"),
    ({"n_bootstrap": 10}, "n_bootstrap"),
    ({"n_bootstrap": True}, "n_bootstrap"),
    ({"confidence_level": 1}, "confidence_level"),
    ({"confidence_level": float("nan")}, "confidence_level"),
    ({"random_seed": -1}, "random_seed"),
    ({"random_seed": True}, "random_seed"),
    ({"pairing_id": "not-for-independent"}, "pairing_id"),
])
def test_invalid_uncertainty_config_rejected(tmp_path, changes, reason):
    path = _settings(tmp_path, **changes)
    with pytest.raises(ValueError, match=reason):
        UncertaintyConfig.read(path)


def test_pairing_id_is_required_and_no_single_scenario_bootstrap(tmp_path):
    with pytest.raises(ValueError, match="pairing_id"):
        UncertaintyConfig.read(_settings(tmp_path, "paired", pairing_id=""))
    selection, targets, paths = _independent(tmp_path)
    for path in paths:
        ensemble = json.loads(path.read_text())
        ensemble["n_scenarios"] = 1
        ensemble["scenario_summaries"] = ensemble["scenario_summaries"][:1]
        ensemble["scenarios"] = ensemble["scenarios"][:1]
        path.write_text(json.dumps(ensemble))
    with pytest.raises(ValueError, match="at least two realizations"):
        write_strategy_uncertainty(selection, targets, _settings(tmp_path), tmp_path / "report.json")


def test_outperformance_probability_respects_ties_and_benefit_direction():
    import numpy as np
    a = np.array([0., 2.])
    b = np.array([1., 2.])
    # B beats A in two of four cross pairs, ties in one.
    assert _single_realization_win_probability(a, b, paired=False, larger_is_better=True) == .625
    assert _single_realization_win_probability(a, b, paired=False, larger_is_better=False) == .375
    assert _single_realization_win_probability(a, b, paired=True, larger_is_better=True) == .75
    assert _single_realization_win_probability(a, b, paired=True, larger_is_better=False) == .25


def test_destinations_are_protected_and_cli_operates(tmp_path):
    selection, targets, paths = _independent(tmp_path)
    settings = _settings(tmp_path)
    with pytest.raises(ValueError, match="must not overwrite"):
        write_strategy_uncertainty(selection, targets, settings, paths[0])
    with pytest.raises(ValueError, match="must not overwrite"):
        write_strategy_uncertainty(selection, targets, settings, tmp_path / "fine.json", chart_path=paths[0])
    output = tmp_path / "uncertainty_result.json"
    cmd = [sys.executable, "-m", "basin_bmp_sim.analysis.strategy_uncertainty", str(selection),
           str(targets), str(settings), str(output), "--chart", str(tmp_path / "intervals.svg")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert json.loads(output.read_text())["sampling_design"] == "independent"
    assert (tmp_path / "intervals.svg").is_file()


def test_no_budget_yields_null_budget_metrics(tmp_path):
    selection, targets, _ = _independent(tmp_path)
    cfg = json.loads(targets.read_text())
    del cfg["budget_usd"]
    targets.write_text(json.dumps(cfg))
    output = tmp_path / "result.json"
    write_strategy_uncertainty(selection, targets, _settings(tmp_path), output)
    data = json.loads(output.read_text())
    assert data["budget_usd"] is None
    assert data["pairwise_differences"][0]["budget_outcomes"] is None


def test_chart_label_escapes_html_and_rejects_invalid_target(tmp_path):
    selection, targets, _ = _independent(tmp_path)
    raw = json.loads(selection.read_text())
    raw["strategies"][0]["name"] = "Wetland <&>"
    selection.write_text(json.dumps(raw))
    chart = tmp_path / "out.svg"
    write_strategy_uncertainty(selection, targets, _settings(tmp_path), tmp_path / "report.json",
                               chart_path=chart)
    ElementTree.parse(chart)
    assert "Wetland &lt;&amp;&gt;" in chart.read_text()


def test_three_strategies_produce_all_ordered_pairs_and_distinct_seed_streams(tmp_path):
    selection, targets, paths = _strategy_data(
        tmp_path, mode="comparable", confirm=True, model_id="one-model-v1",
        with_third=True)
    _reseed(paths[1], 12345)
    _reseed(paths[2], 54321)
    report = tmp_path / "three.json"
    write_strategy_uncertainty(selection, targets, _settings(tmp_path), report)
    pairs = json.loads(report.read_text())["pairwise_differences"]
    assert [(r["strategy_a"], r["strategy_b"]) for r in pairs] == [
        ("Current", "Improved"), ("Current", "Low-cost"),
        ("Improved", "Low-cost"),
    ]
    assert all(len(p["targets"]) == 2 for p in pairs)
    assert all(p["budget_outcomes"] is not None for p in pairs)


def test_corrupted_source_preserves_prior_report_and_chart(tmp_path):
    selection, targets, paths = _independent(tmp_path)
    previous = tmp_path / "existing.json"
    chart = tmp_path / "existing.svg"
    previous.write_bytes(b"existing published report")
    chart.write_bytes(b"existing published SVG")
    data = json.loads(paths[1].read_text())
    data["scenario_summaries"][0]["worker_seed"] = -5
    paths[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed mismatch"):
        write_strategy_uncertainty(selection, targets, _settings(tmp_path), previous,
                                   chart_path=chart)
    assert previous.read_bytes() == b"existing published report"
    assert chart.read_bytes() == b"existing published SVG"
