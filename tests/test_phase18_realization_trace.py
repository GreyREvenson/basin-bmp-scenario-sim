"""Phase 18: actual replay-derived input recording without RNG or science changes."""
from __future__ import annotations

from hashlib import sha256
import json
import sys

import pytest

from basin_bmp_sim.execution import cli as run_model
from basin_bmp_sim.execution.backend_selection import run_backend_selection
from basin_bmp_sim.analysis.input_sensitivity import write_input_sensitivity
from basin_bmp_sim.analysis.realization_trace import (
    extract_realization_trace, write_realization_trace,
)
from test_phase12_backend_ensembles import _selection
from test_phase13_checkpointed_ensembles import _v3
from test_phase9_independent_bmi_initialization import _mocked_config_inputs


def _ensemble(monkeypatch, tmp_path, *, n=4, case="mixed_budget"):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, case)
    selection = _selection(tmp_path, manifest, n=n)
    source = tmp_path / "ensemble.json"
    run_backend_selection(selection, source)
    return selection, source


def test_trace_contains_actual_realized_bmp_values_and_is_deterministic(monkeypatch, tmp_path):
    _, source = _ensemble(monkeypatch, tmp_path)
    original = source.read_bytes()
    a, b = tmp_path / "trace_a.json", tmp_path / "trace_b.json"
    ledger = tmp_path / "ledger.json"
    write_realization_trace(source, a, ledger_file=ledger)
    write_realization_trace(source, b)
    assert a.read_bytes() == b.read_bytes()
    assert source.read_bytes() == original
    data = json.loads(source.read_text())
    trace = json.loads(a.read_text())
    raw = json.loads(ledger.read_text())
    assert len(trace["realizations"]) == len(data["scenarios"]) == 4
    assert all(x["role"] == "realized_diagnostic" for x in trace["features"])
    assert raw["source_ensemble_sha256"] == sha256(original).hexdigest()
    names = [x["name"] for x in trace["features"]]
    for trace_row, original_row, ledger_row in zip(trace["realizations"], data["scenarios"], raw["realizations"]):
        assert trace_row["scenario_id"] == original_row["scenario_id"]
        assert trace_row["worker_seed"] == original_row["worker_seed"]
        assert trace_row["values"][names.index("realized_bmp_count")] == len(original_row["replay_scenario"]["placements"])
        assert ledger_row["placements"] == original_row["replay_scenario"]["placements"]
        assert trace_row["values"][names.index("realized_bmp_design_cost_quantity_sum")] == pytest.approx(
            sum(x["design"]["cost_quantity_ha"] for x in ledger_row["placements"]))
        assert trace_row["values"][names.index("realized_failed_bmp_count")] == sum(
            x["failed"] for x in ledger_row["placements"])


def test_phase17_accepts_automatic_trace(monkeypatch, tmp_path):
    _, source = _ensemble(monkeypatch, tmp_path, n=4)
    trace = tmp_path / "trace.json"
    write_realization_trace(source, trace)
    domain = json.loads(source.read_text())["index_manifest"]
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"schema_version": 1, "targets": [{
        "name": "TN mass", "pollutant": domain["pollutants"][0],
        "outlet_oid": domain["outlet_oids"][0],
        "minimum_removed_mass_kg": 0.0,
    }]}))
    config = tmp_path / "sensitivity.json"
    config.write_text(json.dumps({"schema_version": 1, "n_bootstrap": 200, "random_seed": 4}))
    output = tmp_path / "sensitivity_output.json"
    write_input_sensitivity(source, trace, targets, config, output)
    assert json.loads(output.read_text())["n_scenarios"] == 4


def test_checkpointed_and_serial_export_are_identical(monkeypatch, tmp_path):
    selection, source = _ensemble(monkeypatch, tmp_path, n=3)
    manifest = tmp_path / json.loads(selection.read_text())["component_manifest"]
    selection_v3 = _v3(tmp_path, manifest, n=3)
    checkpointed = tmp_path / "checkpointed.json"
    run_backend_selection(selection_v3, checkpointed)
    assert source.read_bytes() == checkpointed.read_bytes()
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_realization_trace(source, a)
    write_realization_trace(checkpointed, b)
    assert a.read_bytes() == b.read_bytes()


def test_cli_opt_in_writes_sidecars_without_changing_ensemble(monkeypatch, tmp_path):
    selection, original = _ensemble(monkeypatch, tmp_path, n=3)
    copied = tmp_path / "cli_ensemble.json"
    trace, ledger = tmp_path / "cli_trace.json", tmp_path / "cli_ledger.json"
    monkeypatch.setattr(sys, "argv", ["basin-bmp-sim", "--backend-selection", str(selection),
                 "--coupled-result", str(copied), "--input-trace-result", str(trace),
                 "--input-ledger-result", str(ledger)])
    run_model.main()
    assert copied.read_bytes() == original.read_bytes()
    assert trace.is_file() and ledger.is_file()


def test_external_requires_explicit_independent_draw_assertion(monkeypatch, tmp_path):
    _, source = _ensemble(monkeypatch, tmp_path, n=2)
    synthetic = json.loads(source.read_text())
    synthetic["backend"] = "external"
    for record in synthetic["scenarios"]:
        record["backend"] = "external"
    with pytest.raises(ValueError, match="confirmation"):
        extract_realization_trace(synthetic, source_sha256="fixture")
    trace, _ = extract_realization_trace(synthetic, source_sha256="fixture",
                                         confirm_external_independence=True)
    assert trace["sampling_design_confirmed"]
    assert "affirmed by analyst" in trace["trace_provenance"]


def test_invalid_and_tampered_inputs_never_replace_existing_trace(monkeypatch, tmp_path):
    _, source = _ensemble(monkeypatch, tmp_path, n=2)
    prior = tmp_path / "existing_trace.json"
    prior.write_bytes(b"prior valid trace")
    original = source.read_bytes()
    for candidate in [source, prior]:
        with pytest.raises(ValueError, match="must not overwrite"):
            write_realization_trace(source, candidate, ledger_file=prior if candidate == source else source)
    assert source.read_bytes() == original
    tampered = json.loads(source.read_text())
    tampered["scenarios"][0]["replay_scenario"]["placements"][0]["cost_usd"] += 0.01
    source.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="cost differs"):
        write_realization_trace(source, prior)
    assert prior.read_bytes() == b"prior valid trace"
    tampered["scenarios"][0]["replay_scenario"]["placements"][0]["efficiency_by_pollutant"][0].popitem()
    source.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        write_realization_trace(source, prior)
    assert prior.read_bytes() == b"prior valid trace"
