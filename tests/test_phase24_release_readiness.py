"""Phase 24: release evidence must be complete, bound and verifiable.

All inputs are synthetic report documents; no repository sample data is read.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from src.interfaces.bmi import release_gate
from src.interfaces.bmi.release_evidence import public_contract_sha256, sha256_file
from src.interfaces.bmi.acceptance import accept_backend
from test_phase19_public_interoperability import _fixture


def _identity():
    return {
        "selection_sha256": "a" * 64,
        "startup_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "public_contract_sha256": public_contract_sha256(),
    }


def _acceptance():
    return {
        "acceptance_schema_version": 1,
        "scope": "project_annual_bmi_black_box_not_csdms_certification",
        "backend": "external",
        "external_factory": "independent_backend.model:factory",
        "deterministic_repeat": True,
        "bmp_count": 3,
        "result_sha256": "0" * 64,
        "generator": {"variables": [{"name": "bmp__cps_code"}]},
        "model": {"variables": [{"name": "model__outlet_removed_mass"}]},
        **_identity(),
    }


def _portability(system):
    return {
        "portability_schema_version": 1,
        "scope": "project_annual_bmi_serial_parallel_spawn_and_resume_not_csdms_certification",
        "backend": "external",
        "external_factory": "independent_backend.model:factory",
        "operating_system": system,
        "process_start_method": "spawn",
        "n_scenarios": 2,
        "n_jobs": 2,
        "serial_parallel_byte_identical": True,
        "resume_byte_identical": True,
        "result_sha256": "1" * 64,
        "official_bmipy": {
            "bmipy_version": "2.0",
            "official_abstract_methods_checked": 40,
            "adapters_checked": ["ExternalWatershedModelBmi", "LiveScenarioGeneratorBmi",
                                 "WatershedModelBmi", "MonteCarloDriverBmi",
                                 "StochasticScenarioBmi"],
        },
        **_identity(),
    }


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _evidence(tmp_path):
    return {
        "acceptance": _write(tmp_path, "acceptance.json", _acceptance()),
        "windows": _write(tmp_path, "win.json", _portability("Windows")),
        "linux": _write(tmp_path, "linux.json", _portability("Linux")),
    }


def test_missing_real_world_evidence_never_claims_architecture_certification():
    result = release_gate.evaluate_release_gate()
    assert not result["ready"]
    assert result["status"] == "pending_evidence"
    assert result["checks"]["public_contract"] == "passed"
    assert result["checks"]["windows_spawn_and_bmipy"] == "missing"
    assert result["checks"]["external_backend_acceptance"] == "missing"


def test_complete_evidence_yields_review_ready_but_not_scientific_certification(tmp_path):
    paths = _evidence(tmp_path)
    first = release_gate.evaluate_release_gate(**paths)
    second = release_gate.evaluate_release_gate(**paths)
    assert first == second
    assert first["ready"]
    assert first["status"] == "ready_for_architecture_release_review"
    assert all(value == "passed" for value in first["checks"].values())
    for key, value in paths.items():
        slot = {"acceptance": "external_backend_acceptance",
                "windows": "windows_spawn_and_bmipy", "linux": "linux_spawn_and_bmipy"}[key]
        assert first["evidence"][slot]["sha256"] == sha256_file(value)
    assert any("not certify" in s for s in first["limitations"])


def test_one_missing_platform_cannot_be_declared_ready(tmp_path):
    paths = _evidence(tmp_path)
    del paths["windows"]
    result = release_gate.evaluate_release_gate(**paths)
    assert not result["ready"]
    assert result["checks"]["windows_spawn_and_bmipy"] == "missing"


@pytest.mark.parametrize("field,value,error", [
    ("backend", "native", "External-backend acceptance"),
    ("deterministic_repeat", False, "External-backend acceptance"),
    ("external_factory", "", "External-backend acceptance"),
    ("public_contract_sha256", "f" * 64, "different public BMI contract"),
    ("selection_sha256", "not_sha", "valid versioned source"),
])
def test_reject_invalid_acceptance(tmp_path, field, value, error):
    report = _acceptance()
    report[field] = value
    with pytest.raises(ValueError, match=error):
        release_gate.evaluate_release_gate(acceptance=_write(tmp_path, "a.json", report))


@pytest.mark.parametrize("field,value", [
    ("operating_system", "Linux"),
    ("process_start_method", "fork"),
    ("n_scenarios", 1),
    ("n_jobs", 1),
    ("serial_parallel_byte_identical", False),
    ("resume_byte_identical", False),
    ("official_bmipy", "not_checked"),
])
def test_reject_incomplete_windows_checks(tmp_path, field, value):
    report = _portability("Windows")
    report[field] = value
    with pytest.raises(ValueError, match="Windows report lacks completed"):
        release_gate.evaluate_release_gate(windows=_write(tmp_path, "p.json", report))


def test_reject_nonmatching_configuration_and_backend_factory(tmp_path):
    paths = _evidence(tmp_path)
    other = _portability("Linux")
    other["startup_sha256"] = "9" * 64
    paths["linux"] = _write(tmp_path, "other.json", other)
    with pytest.raises(ValueError, match="different selection, startup"):
        release_gate.evaluate_release_gate(**paths)
    other["startup_sha256"] = _identity()["startup_sha256"]
    other["external_factory"] = "different_backend.model:factory"
    paths["linux"] = _write(tmp_path, "other.json", other)
    with pytest.raises(ValueError, match="different external backend"):
        release_gate.evaluate_release_gate(**paths)


def test_reject_malformed_json_without_replacing_prior_report(tmp_path, monkeypatch):
    source = tmp_path / "broken.json"
    source.write_text("not json", encoding="utf-8")
    target = tmp_path / "release.json"
    target.write_bytes(b"previous review")
    monkeypatch.setattr(sys, "argv", ["gate", "--windows", str(source),
                                      "--report", str(target)])
    with pytest.raises(ValueError, match="Malformed release evidence"):
        release_gate.main()
    assert target.read_bytes() == b"previous review"


def test_strict_cli_writes_pending_status_but_exits_nonzero(tmp_path, monkeypatch):
    target = tmp_path / "readiness.json"
    monkeypatch.setattr(sys, "argv", ["gate", "--report", str(target), "--require-ready"])
    with pytest.raises(SystemExit) as exc:
        release_gate.main()
    assert exc.value.code == 1
    assert json.loads(target.read_text())["status"] == "pending_evidence"


def test_import_audit_only_accepts_documented_reference_surface():
    result = release_gate.audit_public_reference_backend()
    assert result["passed"]
    assert set(result["project_imports"]) == {
        "src.interfaces.annual_model_contract", "src.interfaces.backend_capabilities"}
    with pytest.raises(ValueError, match="unpublished model APIs"):
        release_gate.audit_public_reference_backend("from src.models.assemblies import something")
    with pytest.raises(ValueError, match="unpublished model APIs"):
        release_gate.audit_public_reference_backend("import src.driver.monte_carlo_driver")


def test_real_external_acceptance_is_bound_to_source_file_hashes(tmp_path, monkeypatch):
    _, selection, _, _ = _fixture(monkeypatch, tmp_path)
    report = accept_backend(selection)
    assert report["selection_sha256"] == sha256_file(selection)
    assert report["public_contract_sha256"] == public_contract_sha256()
    assert report["external_factory"] == "reference_backend.annual_linear:make_backend"
