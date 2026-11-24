"""Phase 26: collect real-platform evidence locally, without hosted CI files."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from basin_bmp_sim.certification import local_certification as local
from test_phase24_release_readiness import _acceptance, _identity, _portability


def _selection(tmp_path: Path) -> Path:
    (tmp_path / "model.yaml").write_text("synthetic: true\n", encoding="utf-8")
    (tmp_path / "startup.json").write_text(json.dumps({
        "schema_version": 1, "model_config": "model.yaml",
        "scenario_id": 1, "worker_seed": 123,
    }), encoding="utf-8")
    (tmp_path / "asset.json").write_text("{}", encoding="utf-8")
    selection = tmp_path / "backend.json"
    selection.write_text(json.dumps({
        "schema_version": 1,
        "backend": "external",
        "component_manifest": "startup.json",
        "external_factory": "independent_backend.model:factory",
        "backend_options": {"domain_file": "asset.json"},
    }), encoding="utf-8")
    return selection


def _fake_checks(monkeypatch):
    called = []
    monkeypatch.setattr(local.platform, "system", lambda: "Linux")
    def identity(*args, **kwargs):
        value = _identity()
        value["additional_scientific_input_count"] = len(kwargs.get("additional_files", ()))
        return value

    monkeypatch.setattr(local, "source_identity", identity)

    def accepted(*args, **kwargs):
        called.append("acceptance")
        return _acceptance()

    def portable(*args, **kwargs):
        called.append("portability")
        return _portability("Linux")

    monkeypatch.setattr(local, "accept_backend", accepted)
    monkeypatch.setattr(local, "certify_portability", portable)
    return called


def test_local_run_writes_pending_evidence_without_other_platform(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    called = _fake_checks(monkeypatch)
    directory = tmp_path / "certification"
    result = local.run_local_certification(selection, directory,
                                           additional_files=("asset.json",))
    assert called == ["acceptance", "portability"]
    assert result["release_gate_status"] == "pending_evidence"
    assert not result["ready_for_architecture_release_review"]
    assert json.loads((directory / "acceptance.json").read_text()) == _acceptance()
    assert json.loads((directory / "linux.json").read_text()) == _portability("Linux")
    assert not (directory / "windows.json").exists()


def test_completed_cross_platform_bundle_is_review_ready(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    _fake_checks(monkeypatch)
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "windows.json").write_text(json.dumps(_portability("Windows")))
    result = local.run_local_certification(selection, directory,
                                           additional_files=("asset.json",))
    assert result["release_gate_status"] == "ready_for_architecture_release_review"
    assert result["ready_for_architecture_release_review"]


def test_incompatible_cross_platform_evidence_does_not_replace_reports(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    _fake_checks(monkeypatch)
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "acceptance.json").write_bytes(b"old acceptance")
    (directory / "linux.json").write_bytes(b"old linux")
    win = _portability("Windows")
    win["scientific_inputs_sha256"] = "f" * 64
    (directory / "windows.json").write_text(json.dumps(win))
    with pytest.raises(ValueError, match="different selection, startup"):
        local.run_local_certification(selection, directory, additional_files=("asset.json",))
    assert (directory / "acceptance.json").read_bytes() == b"old acceptance"
    assert (directory / "linux.json").read_bytes() == b"old linux"


def test_failed_execution_cannot_overwrite_valid_evidence(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    _fake_checks(monkeypatch)
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "acceptance.json").write_bytes(b"previous")
    monkeypatch.setattr(local, "certify_portability", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("spawn failed")))
    with pytest.raises(RuntimeError, match="spawn failed"):
        local.run_local_certification(selection, directory, additional_files=("asset.json",))
    assert (directory / "acceptance.json").read_bytes() == b"previous"
    assert not (directory / "linux.json").exists()


def test_official_bmi_must_be_checked_for_release(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    _fake_checks(monkeypatch)
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "windows.json").write_text(json.dumps(_portability("Windows")))
    result = local.run_local_certification(selection, directory, require_bmipy=False,
                                           additional_files=("asset.json",))
    assert result["release_gate_status"] == "pending_official_bmipy"
    assert result["ready_for_architecture_release_review"] is False


def test_refuses_undeclared_backend_assets_and_wrong_os(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    _fake_checks(monkeypatch)
    with pytest.raises(ValueError, match="declared scientific asset"):
        local.run_local_certification(selection, tmp_path / "certification")
    with pytest.raises(ValueError, match="other than the current host"):
        local.run_local_certification(selection, tmp_path / "certification",
                                      operating_system="Windows", additional_files=("asset.json",))


def test_rejects_native_selection_and_invalid_parallel_counts(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    _fake_checks(monkeypatch)
    source = json.loads(selection.read_text())
    source.update(backend="native", external_factory=None, backend_options={})
    selection.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="single-scenario external"):
        local.run_local_certification(selection, tmp_path / "certification", additional_files=("asset.json",))
    source.update(backend="external", external_factory="independent_backend.model:factory")
    selection.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="at least two scenarios"):
        local.run_local_certification(selection, tmp_path / "certification", n_scenarios=1,
                                      additional_files=("asset.json",))


def test_windows_run_uses_real_host_label_and_merges_linux_evidence(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    _fake_checks(monkeypatch)
    monkeypatch.setattr(local.platform, "system", lambda: "Windows")
    monkeypatch.setattr(local, "certify_portability", lambda *a, **k: _portability("Windows"))
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "linux.json").write_text(json.dumps(_portability("Linux")))
    result = local.run_local_certification(selection, directory,
                                           additional_files=("asset.json",))
    assert result["operating_system"] == "Windows"
    assert result["ready_for_architecture_release_review"] is True
    assert json.loads((directory / "windows.json").read_text())["operating_system"] == "Windows"


def test_evidence_destination_cannot_overwrite_listed_scientific_asset(tmp_path, monkeypatch):
    selection = _selection(tmp_path)
    _fake_checks(monkeypatch)
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "linux.json").write_text("scientific input", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite an input configuration"):
        local.run_local_certification(selection, directory,
                                      additional_files=("evidence/linux.json",))
    assert (directory / "linux.json").read_text() == "scientific input"
