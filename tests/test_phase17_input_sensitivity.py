"""Phase 17: trace-aligned, descriptive input/output diagnostics."""
from __future__ import annotations

from hashlib import sha256
import json
import subprocess
import sys
from xml.etree import ElementTree

import numpy as np
import pytest

from basin_bmp_sim.analysis.input_sensitivity import (
    SensitivityConfig, _average_ranks, _spearman, analyze_input_sensitivity,
    write_input_sensitivity,
)
from basin_bmp_sim.analysis.ensemble_analysis import DecisionSupportConfig
from test_phase14_ensemble_decision_support import _analysis_config, _synthetic_ensemble


def _trace(ensemble):
    # Input features are independent of stored BMP response fields; the source
    # must be a real sampling trace for scientific interpretation in practice.
    return {
        "schema_version": 1,
        "sampling_design": "independent",
        "sampling_design_confirmed": True,
        "trace_provenance": "Explicit test input draw log captured before scientific execution",
        "features": [
            {"name": "infiltration_multiplier", "units": "1", "role": "sampled_input"},
            {"name": "farm_selection_count", "units": "count", "role": "realized_diagnostic"},
            {"name": "fixed_parameter", "units": "1", "role": "sampled_input"},
        ],
        "realizations": [{
            "scenario_id": row["scenario_id"],
            "worker_seed": row["worker_seed"],
            "values": [float(i), float([0, 0, 2, 2][i]), 9.0],
        } for i, row in enumerate(ensemble["scenario_summaries"])],
    }


def _files(tmp_path):
    ensemble = _synthetic_ensemble()
    result = tmp_path / "ensemble.json"
    trace = tmp_path / "trace.json"
    targets = _analysis_config(tmp_path)
    config = tmp_path / "sensitivity.json"
    result.write_text(json.dumps(ensemble))
    trace.write_text(json.dumps(_trace(ensemble)))
    config.write_text(json.dumps({"schema_version": 1, "n_bootstrap": 200,
                                  "confidence_level": 0.95, "random_seed": 25,
                                  "primary_target": "N reduction"}))
    return result, trace, targets, config


def test_tied_rank_calculation_negative_correlation_and_constant():
    np.testing.assert_allclose(_average_ranks(np.array([4.0, 2, 2, 7, -1])), [4, 2.5, 2.5, 5, 1])
    assert _spearman(np.array([1., 2., 3., 4.]), np.array([-1., -2., -3., -4.])) == pytest.approx(-1)
    assert _spearman(np.array([1., 1., 1., 1.]), np.arange(4.)) is None
    assert _spearman(np.arange(4.), np.array([1., 1., 1., 1.])) is None
    assert _spearman(np.arange(4.), np.array([4., 2., 3., 1.])) < 0


def test_signed_mass_cost_and_attainment_are_distinct_outputs(tmp_path):
    source, trace, targets, cfg = _files(tmp_path)
    dest = tmp_path / "analysis.json"
    write_input_sensitivity(source, trace, targets, cfg, dest)
    report = json.loads(dest.read_text())
    assert report["sensitivity_schema_version"] == 1
    assert report["n_scenarios"] == 4
    assert report["primary_target"] == "N reduction"
    assert report["sampling_design_confirmed"] is True
    assert report["source_ensemble_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert report["input_trace_sha256"] == sha256(trace.read_bytes()).hexdigest()
    mass = next(row for row in report["outcomes"] if row["outcome"] ==
                "signed_annual_removed_mass_kg::N reduction")
    assert mass["associations"][0]["spearman_rho"] == pytest.approx(-1)
    assert mass["associations"][2]["spearman_rho"] is None
    assert mass["associations"][2]["bootstrap_percentile_interval"] is None
    assert mass["associations"][0]["valid_bootstrap_replicates"] > 0
    cost = next(row for row in report["outcomes"] if row["outcome"] == "implementation_cost_usd")
    assert cost["units"] == "USD"
    event = next(row for row in report["outcomes"] if row["outcome"] ==
                 "target_attained::N reduction")
    assert event["units"] == "boolean (0 or 1)"
    assert len(report["input_input_correlations"]) == 3
    assert "not Sobol" in report["interpretation"]["scope"]


def test_repeatability_and_valid_svg_without_source_changes(tmp_path):
    source, trace, targets, config = _files(tmp_path)
    before = {p: p.read_bytes() for p in (source, trace, targets, config)}
    a, b = tmp_path / "first.json", tmp_path / "second.json"
    graphic_a, graphic_b = tmp_path / "first.svg", tmp_path / "second.svg"
    write_input_sensitivity(source, trace, targets, config, a, chart_path=graphic_a)
    write_input_sensitivity(source, trace, targets, config, b, chart_path=graphic_b)
    assert a.read_bytes() == b.read_bytes()
    assert graphic_a.read_bytes() == graphic_b.read_bytes()
    ElementTree.parse(graphic_a)
    assert [p.read_bytes() for p in before] == list(before.values())
    trace_data = json.loads(trace.read_text())
    trace_data["features"][0]["name"] = "<infiltration & runoff>"
    trace.write_text(json.dumps(trace_data))
    write_input_sensitivity(source, trace, targets, config, a, chart_path=graphic_a)
    assert "&lt;infiltration &amp; runoff&gt;" in graphic_a.read_text()


@pytest.mark.parametrize("change,expected", [
    (lambda t: t["realizations"][1].update(worker_seed=3), "seed mismatch"),
    (lambda t: t["realizations"][1].update(scenario_id=777), "identity"),
    (lambda t: t["realizations"].reverse(), "identity"),
    (lambda t: t["realizations"].pop(), "one realization"),
    (lambda t: t["realizations"][0].update(values=[1.]), "feature count"),
    (lambda t: t["realizations"][0]["values"].__setitem__(0, True), "not a boolean"),
    (lambda t: t["realizations"][0]["values"].__setitem__(0, float("inf")), "Invalid input trace JSON"),
    (lambda t: t["features"][0].update(role="model_output"), "metadata"),
    (lambda t: t["features"][1].update(name="infiltration_multiplier"), "duplicated"),
    (lambda t: t.update(trace_provenance=""), "trace_provenance"),
    (lambda t: t.update(sampling_design="paired"), "independent-realization"),
    (lambda t: t.update(sampling_design_confirmed=1), "independent-realization"),
    (lambda t: t.update(schema_version=2), "schema"),
    (lambda t: t.update(unexpected="extra"), "schema"),
])
def test_invalid_trace_preserves_report(tmp_path, change, expected):
    source, trace, targets, config = _files(tmp_path)
    prior = tmp_path / "existing.json"
    prior.write_bytes(b"existing valid artifact")
    data = json.loads(trace.read_text())
    change(data)
    trace.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=expected):
        write_input_sensitivity(source, trace, targets, config, prior)
    assert prior.read_bytes() == b"existing valid artifact"


@pytest.mark.parametrize("field,value", [
    ("n_bootstrap", 3), ("n_bootstrap", True),
    ("confidence_level", 1.0), ("confidence_level", 0),
    ("random_seed", -1), ("random_seed", 0.4),
    ("primary_target", "missing"),
])
def test_invalid_config_or_primary_target_is_rejected(tmp_path, field, value):
    ensemble, trace, targets, config = _files(tmp_path)
    data = json.loads(config.read_text())
    data[field] = value
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        write_input_sensitivity(ensemble, trace, targets, config, tmp_path / "result.json")


def test_rejects_destination_collision_and_unknown_target(tmp_path):
    source, trace, targets, config = _files(tmp_path)
    for bad in (source, trace, targets, config):
        with pytest.raises(ValueError, match="must not overwrite"):
            write_input_sensitivity(source, trace, targets, config, bad)
    with pytest.raises(ValueError, match="must not overwrite"):
        write_input_sensitivity(source, trace, targets, config, tmp_path / "report.json",
                                chart_path=trace)
    with pytest.raises(ValueError, match="must not overwrite"):
        write_input_sensitivity(source, trace, targets, config, tmp_path / "report.json",
                                chart_path=tmp_path / "report.json")
    config_data = json.loads(config.read_text())
    config_data["primary_target"] = "P reduction"
    config.write_text(json.dumps(config_data))
    input_data = json.loads(targets.read_text())
    input_data["targets"][1]["outlet_oid"] = "missing"
    targets.write_text(json.dumps(input_data))
    with pytest.raises(ValueError, match="Unknown pollutant or outlet"):
        write_input_sensitivity(source, trace, targets, config, tmp_path / "report.json")


def test_constant_target_indicates_null_not_false_zero(tmp_path):
    source, trace, targets, config = _files(tmp_path)
    data = json.loads(source.read_text())
    for summary, scenario in zip(data["scenario_summaries"], data["scenarios"]):
        summary["outlet_removed_mass_kg"][0] = 3
        scenario["outlet_removed_mass_kg"][0] = 3
    source.write_text(json.dumps(data))
    write_input_sensitivity(source, trace, targets, config, tmp_path / "out.json")
    outcomes = json.loads((tmp_path / "out.json").read_text())["outcomes"]
    for row in outcomes:
        if "N reduction" in row["outcome"]:
            assert row["constant_outcome"]
            assert all(a["spearman_rho"] is None for a in row["associations"])


def test_cli_matches_direct_api(tmp_path):
    source, trace, targets, config = _files(tmp_path)
    expected = tmp_path / "expected.json"
    actual = tmp_path / "actual.json"
    write_input_sensitivity(source, trace, targets, config, expected)
    subprocess.run([sys.executable, "-m", "basin_bmp_sim.analysis.input_sensitivity",
                    str(source), str(trace), str(targets), str(config), str(actual)],
                   check=True, capture_output=True, text=True)
    assert actual.read_bytes() == expected.read_bytes()


def test_in_memory_api_rejects_tampered_ensemble(tmp_path):
    source, trace, targets, config = _files(tmp_path)
    data = json.loads(source.read_text())
    data["scenario_summaries"][0]["outlet_removed_mass_kg"][0] += 100
    with pytest.raises(ValueError, match="summary mismatch"):
        analyze_input_sensitivity(
            data, json.loads(trace.read_text()), DecisionSupportConfig.read(targets),
            SensitivityConfig.read(config), ensemble_sha256="", trace_sha256="")


def test_svg_feature_limit_only_affects_optional_chart(tmp_path):
    source, trace, targets, config = _files(tmp_path)
    data = json.loads(trace.read_text())
    # Expand a trace without changing its scenario identity or ordering.
    for i in range(41):
        data["features"].append({"name": f"f{i}", "units": "1", "role": "sampled_input"})
    for record in data["realizations"]:
        record["values"].extend([1.] * 41)
    trace.write_text(json.dumps(data))
    output = tmp_path / "out.json"
    write_input_sensitivity(source, trace, targets, config, output)
    prior = tmp_path / "previous.json"
    prior.write_bytes(b"previous")
    with pytest.raises(ValueError, match="at most 40"):
        write_input_sensitivity(source, trace, targets, config, prior,
                                chart_path=tmp_path / "chart.svg")
    assert prior.read_bytes() == b"previous"
