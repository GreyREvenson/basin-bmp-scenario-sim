"""Phase 11: explicit backend selection, scientific preflight and CLI routing."""
from __future__ import annotations

import json
import logging
import sys
from types import ModuleType

import numpy as np
import pytest

from basin_bmp_sim.execution import cli as run_model
from basin_bmp_sim.execution.backend_selection import (
    BackendSelection, build_selected_coupler, run_backend_selection,
)
from basin_bmp_sim.execution.monte_carlo_driver import _ScenarioContext
from basin_bmp_sim.interfaces.backend_capabilities import (
    AnnualBackendCapabilities, ExternalBackendBundle,
)
from basin_bmp_sim.interfaces.component_bootstrap import ComponentStartup
from test_phase10_model_neutral_exchange import IndependentToyBackend, _domain
from test_phase9_independent_bmi_initialization import _mocked_config_inputs


def _selection(tmp_path, manifest, **updates):
    payload = {
        "schema_version": 1,
        "backend": "external",
        "component_manifest": manifest.name,
        "external_factory": "phase11_toy_plugin:make_backend",
        "backend_options": {"response_scale": 1.0},
        "required_features": ["afo_recovery"],
    }
    payload.update(updates)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _toy_plugin(monkeypatch, shared, cfg, seed, *, features=None, cps=None, pollutant_override=None):
    domain = _domain(_ScenarioContext(cfg, shared, logging.getLogger("phase11.toy"), seed))
    observed = {}
    plugin = ModuleType("phase11_toy_plugin")

    def make_backend(*, startup, options):
        observed["startup"] = startup
        observed["options"] = options
        backend = IndependentToyBackend(domain)
        observed["backend"] = backend
        return ExternalBackendBundle(
            domain=domain,
            backend=backend,
            capabilities=AnnualBackendCapabilities(
                supported_cps=tuple(cps) if cps is not None else domain.bmp_cps,
                pollutants=pollutant_override or domain.pollutants,
                pathways=domain.pathway_names,
                features=frozenset(features if features is not None else (
                    "signed_efficiency", "multi_parcel_treatment", "afo_recovery"
                )),
            ),
        )

    plugin.make_backend = make_backend
    monkeypatch.setitem(sys.modules, "phase11_toy_plugin", plugin)
    return observed


@pytest.mark.parametrize("patch,error", [
    ({"schema_version": False}, "schema version"),
    ({"schema_version": 999}, "schema version"),
    ({"backend": "my_fake_backend"}, "native.*external"),
    ({"component_manifest": "missing.json"}, "missing.json"),
    ({"required_features": ["invented_feature"]}, "unknown features"),
    ({"required_features": ["afo_recovery", "afo_recovery"]}, "duplicates"),
    ({"required_features": [True]}, "list of strings"),
    ({"external_factory": "bad.factory"}, "module.path:factory"),
    ({"backend_options": []}, "JSON object"),
    ({"unrecognized": True}, "known fields"),
])
def test_invalid_backend_selection_rejected(tmp_path, patch, error):
    path = tmp_path / "manifest.json"
    config = tmp_path / "config.yaml"
    config.write_text("dummy: yes\n", encoding="utf-8")
    path.write_text(json.dumps({"schema_version": 1, "model_config": "config.yaml",
                                "worker_seed": 7, "scenario_id": 1}), encoding="utf-8")
    selection = _selection(tmp_path, path, **patch)
    with pytest.raises((ValueError, FileNotFoundError), match=error):
        BackendSelection.read(selection)


def test_external_run_selects_factory_and_writes_self_describing_result(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    observed = _toy_plugin(monkeypatch, shared, cfg, seed)
    selection = _selection(tmp_path, manifest)
    result_path = run_backend_selection(selection, tmp_path / "outputs" / "selected.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert observed["startup"] == ComponentStartup.read(manifest)
    assert observed["options"] == {"response_scale": 1.0}
    assert observed["backend"].finished
    assert result["backend"] == "external"
    assert result["scenario_id"] == 1
    assert result["worker_seed"] == seed
    assert len(result["bmp_responses"]) > 0
    assert len(result["bmp_responses"]) == len(result["replay_scenario"]["placements"])
    assert result["bmp_responses"][-1]["outlet_removed_kg"]
    assert len(result["current_load_rate_kg_ha_yr"]) == (
        len(result["index_manifest"]["parcel_ids"]) * len(result["index_manifest"]["pollutants"])
    )
    assert result["bmp_responses"][-1]["cumulative_cost_usd"] >= 0
    assert not list((tmp_path / "outputs").glob("*.tmp"))


def test_capability_rejection_precedes_baseline_initialize_and_rng_draw(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    observed = _toy_plugin(monkeypatch, shared, cfg, seed,
                           features={"signed_efficiency", "multi_parcel_treatment"})
    selection = _selection(tmp_path, manifest)
    with pytest.raises(ValueError, match="afo_recovery"):
        build_selected_coupler(BackendSelection.read(selection))
    assert observed["backend"].baseline is None
    _toy_plugin(monkeypatch, shared, cfg, seed,
                cps=tuple(cps for cps in _domain(_ScenarioContext(
                    cfg, shared, logging.getLogger("phase11.cps"), seed)).bmp_cps if cps != 656))
    with pytest.raises(ValueError, match="CPS codes"):
        build_selected_coupler(BackendSelection.read(selection))
    _toy_plugin(monkeypatch, shared, cfg, seed, pollutant_override=("incorrect",))
    with pytest.raises(ValueError, match="pollutant identities"):
        build_selected_coupler(BackendSelection.read(selection))


def test_required_live_capabilities_cannot_be_omitted(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    _toy_plugin(monkeypatch, shared, cfg, seed, features={"afo_recovery", "multi_parcel_treatment"})
    selection = _selection(tmp_path, manifest, required_features=[])
    with pytest.raises(ValueError, match="signed_efficiency"):
        build_selected_coupler(BackendSelection.read(selection))


def test_native_selection_uses_independent_native_startup(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    selection = _selection(tmp_path, manifest, backend="native", external_factory=None,
                           backend_options={}, required_features=[])
    coupled = build_selected_coupler(BackendSelection.read(selection))
    coupled.initialize()
    try:
        assert coupled.model._ctx is not coupled.generator._ctx
        assert coupled.next_bmp() is not None
    finally:
        coupled.finalize()


def test_native_selection_rejects_external_options(tmp_path):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "config.yaml").write_text("dummy: yes\n", encoding="utf-8")
    manifest.write_text(json.dumps({"schema_version": 1, "model_config": "config.yaml",
                                    "worker_seed": 7, "scenario_id": 1}), encoding="utf-8")
    path = _selection(tmp_path, manifest, backend="native")
    with pytest.raises(ValueError, match="cannot contain external"):
        BackendSelection.read(path)


def test_missing_plugin_factory_is_an_error(tmp_path):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "config.yaml").write_text("dummy: yes\n", encoding="utf-8")
    manifest.write_text(json.dumps({"schema_version": 1, "model_config": "config.yaml",
                                    "worker_seed": 7, "scenario_id": 1}), encoding="utf-8")
    path = _selection(tmp_path, manifest, external_factory="no_phase11_backend:factory")
    with pytest.raises(ValueError, match="Unable to load"):
        build_selected_coupler(BackendSelection.read(path))


def test_legacy_cli_kept_and_selection_route_explicit(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(run_model, "run_backend_selection", lambda selection, output: captured.update(
        selection=selection, output=output))
    path = tmp_path / "selection.json"
    dest = tmp_path / "coupled.json"
    monkeypatch.setattr(sys, "argv", ["basin-bmp-sim", "--backend-selection", str(path),
                                        "--coupled-result", str(dest)])
    run_model.main()
    assert captured == {"selection": path, "output": dest}
    monkeypatch.setattr(sys, "argv", ["basin-bmp-sim", "config.yaml",
                                        "--backend-selection", str(path),
                                        "--coupled-result", str(dest)])
    with pytest.raises(SystemExit) as exc:
        run_model.main()
    assert exc.value.code == 1
    monkeypatch.setattr(sys, "argv", ["basin-bmp-sim", "--backend-selection", str(path)])
    with pytest.raises(SystemExit) as exc:
        run_model.main()
    assert exc.value.code == 1


def test_external_selection_is_reproducible_and_cli_executes_it(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "afo_recovery")
    _toy_plugin(monkeypatch, shared, cfg, seed)
    selection = _selection(tmp_path, manifest)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    run_backend_selection(selection, first)
    monkeypatch.setattr(sys, "argv", ["basin-bmp-sim", "--backend-selection", str(selection),
                                        "--coupled-result", str(second)])
    run_model.main()
    assert first.read_bytes() == second.read_bytes()
    record = json.loads(first.read_text(encoding="utf-8"))
    assert len(record["bmp_responses"]) == len(record["replay_scenario"]["placements"])
    assert record["bmp_responses"]


def test_backend_capability_contract_versions_rejected():
    for field, value, message in (("annual_contract_version", 77, "annual"),
                                  ("exchange_schema_version", 77, "exchange")):
        fields = {
            "supported_cps": (340,), "pollutants": ("TN",),
            "pathways": ("surface",),
            "features": frozenset({"signed_efficiency", "multi_parcel_treatment"}),
            field: value,
        }
        with pytest.raises(ValueError, match=message):
            AnnualBackendCapabilities(**fields)


def test_native_selection_writes_model_neutral_json_without_changing_native_driver(monkeypatch, tmp_path):
    manifest, _, _, seed = _mocked_config_inputs(monkeypatch, tmp_path, "statistical_multi")
    selection = _selection(tmp_path, manifest, backend="native", external_factory=None,
                           backend_options={}, required_features=[])
    output = run_backend_selection(selection, tmp_path / "native_bmi.json")
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["backend"] == "native"
    assert result["worker_seed"] == seed
    assert result["replay_scenario"]["placements"]
    assert result["index_manifest"]["parcel_ids"]
