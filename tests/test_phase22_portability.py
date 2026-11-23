"""Phase 22: official BMI gate, portable process certification, atomic reports.

The optional full scientific spawn run is activated with
BASIN_BMP_BMI_SPAWN_SELECTION, not by the ordinary unit-test suite.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.interfaces.bmi import portability


def _selection(tmp_path: Path) -> Path:
    config = tmp_path / "watershed.yaml"
    config.write_text("not actually loaded by the mocked runner\n", encoding="utf-8")
    manifest = tmp_path / "startup.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "model_config": config.name,
        "scenario_id": 1, "worker_seed": 123,
    }), encoding="utf-8")
    selection = tmp_path / "backend.json"
    selection.write_text(json.dumps({
        "schema_version": 1, "backend": "native",
        "component_manifest": manifest.name,
    }), encoding="utf-8")
    return selection


def _fake_run(calls):
    def execute(selection, target):
        plan = selection.ensemble
        calls.append((plan.n_jobs, plan.resume))
        checkpoint_dir = plan.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for i in range(plan.n_scenarios):
            (checkpoint_dir / f"realization_{i:08d}.json").write_text("{}")
        payload = {"n_scenarios": plan.n_scenarios,
                   "scenarios": [{"scenario_id": i + 1} for i in range(plan.n_scenarios)]}
        Path(target).write_text(json.dumps(payload, sort_keys=True) + "\n")
        return Path(target)
    return execute


def test_serial_parallel_resume_gate_and_report_provenance(monkeypatch, tmp_path):
    selection = _selection(tmp_path)
    calls = []
    monkeypatch.setattr(portability, "run_backend_ensemble", _fake_run(calls))
    monkeypatch.setattr(portability, "check_official_bmipy", lambda: {"verified": True})
    report = portability.certify_portability(selection)
    assert calls == [(1, False), (2, False), (2, True)]
    assert report["serial_parallel_byte_identical"]
    assert report["resume_byte_identical"]
    assert report["official_bmipy"] == {"verified": True}
    assert len(report["model_config_sha256"]) == 64
    assert len(report["public_contract_sha256"]) == 64
    assert report["operating_system"] in ("Linux", "Windows", "Darwin")
    assert report["process_start_method"] == "spawn"
    assert report["scope"].endswith("not_csdms_certification")


def test_no_official_package_never_misrepresented_as_checked(monkeypatch, tmp_path):
    selection = _selection(tmp_path)
    monkeypatch.setattr(portability, "run_backend_ensemble", _fake_run([]))
    monkeypatch.setattr(portability, "check_official_bmipy",
                        lambda: pytest.fail("bmipy should not be checked"))
    report = portability.certify_portability(selection, require_bmipy=False)
    assert report["official_bmipy"] == "not_checked"


def test_parallel_difference_rejected_before_report_published(monkeypatch, tmp_path):
    selection = _selection(tmp_path)
    report = tmp_path / "already_verified.json"
    report.write_bytes(b"old report")
    def unequal_run(selection, target):
        plan = selection.ensemble
        plan.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for i in range(plan.n_scenarios):
            (plan.checkpoint_dir / f"realization_{i:08d}.json").write_text("{}")
        Path(target).write_text(json.dumps({
            "n_scenarios": plan.n_scenarios,
            "scenarios": [{"value": plan.n_jobs} for _ in range(plan.n_scenarios)],
        }))
        return Path(target)
    monkeypatch.setattr(portability, "run_backend_ensemble", unequal_run)
    monkeypatch.setattr(portability, "check_official_bmipy", lambda: {})
    with pytest.raises(ValueError, match="differs from serial"):
        portability.certify_portability(selection)
    assert report.read_bytes() == b"old report"


def test_broken_resume_detected(monkeypatch, tmp_path):
    selection = _selection(tmp_path)
    runner = _fake_run([])
    def corrupt_resume(selection, target):
        runner(selection, target)
        if selection.ensemble.resume:
            Path(target).write_bytes(b"not the original result")
        return Path(target)
    monkeypatch.setattr(portability, "run_backend_ensemble", corrupt_resume)
    with pytest.raises(ValueError, match="Resumed"):
        portability.certify_portability(selection, require_bmipy=False)


@pytest.mark.parametrize("count,jobs", [(1, 2), (2, 1), (True, 2), (2, True), (2, 65)])
def test_invalid_parallel_certification_settings(count, jobs, tmp_path):
    with pytest.raises(ValueError, match="requires"):
        portability.certify_portability(_selection(tmp_path), n_scenarios=count, n_jobs=jobs)


def test_atomic_report_does_not_replace_on_serialization_failure(tmp_path):
    target = tmp_path / "certification.json"
    target.write_bytes(b"original report")
    with pytest.raises(TypeError):
        portability._atomic_report(target, {"invalid": object()})
    assert target.read_bytes() == b"original report"
    assert not list(tmp_path.glob("*.tmp"))


def test_official_bmipy_abc_compares_actual_abstract_methods(monkeypatch):
    class FakeBmi(ABC):
        @abstractmethod
        def update(self): ...
        @abstractmethod
        def initialize(self, config_file=""): ...

    orig = portability.importlib.import_module
    monkeypatch.setattr(portability.importlib, "import_module",
                        lambda name: SimpleNamespace(Bmi=FakeBmi, __version__="test")
                        if name == "bmipy" else orig(name))
    result = portability.check_official_bmipy()
    assert result["official_abstract_methods_checked"] == 2
    assert len(result["adapters_checked"]) == 5


def test_official_bmipy_abc_rejects_missing_method(monkeypatch):
    class FakeBmi(ABC):
        @abstractmethod
        def imaginary_required_bmi_method(self): ...

    orig = portability.importlib.import_module
    monkeypatch.setattr(portability.importlib, "import_module",
                        lambda name: SimpleNamespace(Bmi=FakeBmi)
                        if name == "bmipy" else orig(name))
    with pytest.raises(ValueError, match="imaginary_required_bmi_method"):
        portability.check_official_bmipy()


def test_real_spawn_pool_uses_two_independent_worker_processes():
    # Actual spawned processes, using a tiny importable function. The full
    # science test below is opt-in so default unit tests have no data burden.
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    from src.driver.ensemble_checkpoints import _digest

    payloads = [{"scenario_id": 1}, {"scenario_id": 2}]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        actual = list(pool.map(_digest, payloads))
    assert actual == [_digest(x) for x in payloads]


def test_model_neutral_external_outputs_do_not_write_native_parquet(monkeypatch, tmp_path):
    from src.driver.backend_selection import run_backend_selection
    from test_phase19_public_interoperability import _fixture

    _, selection, _, _ = _fixture(monkeypatch, tmp_path)
    result_path = tmp_path / "neutral.json"
    run_backend_selection(selection, result_path)
    data = json.loads(result_path.read_text())
    assert data["backend"] == "external"
    assert data["result_schema_version"] == 1
    assert "replay_scenario" in data and "index_manifest" in data
    assert not list(tmp_path.rglob("*.parquet"))
    assert not list(tmp_path.rglob("*.xlsx"))
