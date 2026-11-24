"""Phase 13: exact Phase 12 parity, resumable checkpoints and parallel scheduling."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from basin_bmp_sim.execution.backend_selection import BackendSelection, run_backend_selection
from basin_bmp_sim.execution import ensemble_checkpoints as checkpoint
from test_phase12_backend_ensembles import _plugin, _selection
from test_phase9_independent_bmi_initialization import _mocked_config_inputs


def _v3(tmp_path, manifest, *, backend="native", n=3, n_jobs=1,
        checkpoint_dir="checkpoints", resume=False):
    path = _selection(tmp_path, manifest, backend=backend, n=n)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 3
    data["ensemble"] = {
        "n_scenarios": n, "n_jobs": n_jobs,
        "checkpoint_dir": checkpoint_dir, "resume": resume,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("bad,message", [
    ({"n_jobs": 0}, "n_jobs"),
    ({"n_jobs": True}, "n_jobs"),
    ({"n_jobs": 65}, "n_jobs"),
    ({"resume": 4}, "resume"),
    ({"checkpoint_dir": ""}, "checkpoint_dir"),
    ({"extra": 3}, "known fields"),
])
def test_schema_v3_rejects_invalid_options(tmp_path, bad, message):
    (tmp_path / "model.yaml").write_text("stub: yes\n", encoding="utf-8")
    manifest = tmp_path / "startup.json"
    manifest.write_text(json.dumps({"schema_version": 1, "model_config": "model.yaml",
                                    "scenario_id": 1, "worker_seed": 10}), encoding="utf-8")
    selection = _v3(tmp_path, manifest)
    payload = json.loads(selection.read_text())
    payload["ensemble"].update(bad)
    selection.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        BackendSelection.read(selection)


@pytest.mark.parametrize("case", ["mixed_budget", "afo_recovery", "statistical_multi"])
def test_serial_checkpoint_output_is_byte_identical_to_phase12(monkeypatch, tmp_path, case):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, case)
    selection = _selection(tmp_path, manifest, n=3)
    original = tmp_path / "original.json"
    run_backend_selection(selection, original)
    selected = _v3(tmp_path, manifest, n=3)
    resumed = tmp_path / "checkpointed.json"
    run_backend_selection(selected, resumed)
    assert resumed.read_bytes() == original.read_bytes()
    assert len(list((tmp_path / "checkpoints").glob("realization_*.json"))) == 3
    with pytest.raises(FileExistsError, match="not empty"):
        run_backend_selection(selected, resumed)

    _v3(tmp_path, manifest, n=3, resume=True)
    # If resume loaded any scenario again, the worker would raise.
    monkeypatch.setattr(checkpoint, "_run_one_worker", lambda *a: pytest.fail("unnecessary rerun"))
    run_backend_selection(selected, resumed)
    assert resumed.read_bytes() == original.read_bytes()


def test_external_resume_only_restarts_missing_scenario_and_keeps_output_atomic(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    _plugin(monkeypatch, shared, cfg, seed)
    v2 = _selection(tmp_path, manifest, backend="external", n=3)
    expected = tmp_path / "expected.json"
    run_backend_selection(v2, expected)
    calls = _plugin(monkeypatch, shared, cfg, seed, fail_at_id=3)
    v3 = _v3(tmp_path, manifest, backend="external", n=3)
    output = tmp_path / "checkpointed.json"
    output.write_bytes(b"old verified output")
    with pytest.raises(RuntimeError, match="Deliberately failed"):
        run_backend_selection(v3, output)
    assert output.read_bytes() == b"old verified output"
    assert [c[0] for c in calls] == [1, 2]
    assert sorted(p.name for p in (tmp_path / "checkpoints").glob("realization_*.json")) == [
        "realization_00000000.json", "realization_00000001.json"]
    calls = _plugin(monkeypatch, shared, cfg, seed)
    _v3(tmp_path, manifest, backend="external", n=3, resume=True)
    run_backend_selection(v3, output)
    assert [c[0] for c in calls] == [3]
    assert output.read_bytes() == expected.read_bytes()


def test_resume_rejects_changed_config_and_corrupt_record_without_replacing_output(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    selection = _v3(tmp_path, manifest, n=2)
    dest = tmp_path / "finished.json"
    run_backend_selection(selection, dest)
    original = dest.read_bytes()
    _v3(tmp_path, manifest, n=2, resume=True)
    source = tmp_path / "watershed.yaml"
    source.write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        run_backend_selection(selection, dest)
    assert dest.read_bytes() == original
    source.write_text("stub: yes\n", encoding="utf-8")
    corrupt = tmp_path / "checkpoints" / "realization_00000001.json"
    envelope = json.loads(corrupt.read_text())
    envelope["record"]["worker_seed"] += 1
    corrupt.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="checksum"):
        run_backend_selection(selection, dest)
    assert dest.read_bytes() == original


def test_parallel_out_of_order_completions_have_serial_equivalent_results(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "afo_recovery")
    _plugin(monkeypatch, shared, cfg, seed)
    v2 = _selection(tmp_path, manifest, backend="external", n=4)
    serial_output = tmp_path / "serial.json"
    run_backend_selection(v2, serial_output)
    parallel = _v3(tmp_path, manifest, backend="external", n=4, n_jobs=2)
    # A thread executor tests the deterministic completion/order logic while
    # allowing in-memory fixture factories; real process spawn is separately
    # covered by the CLI smoke test with an on-disk native configuration.
    monkeypatch.setattr(checkpoint, "ProcessPoolExecutor", lambda *, max_workers, mp_context:
                        ThreadPoolExecutor(max_workers=max_workers))
    actual = tmp_path / "parallel.json"
    run_backend_selection(parallel, actual)
    assert actual.read_bytes() == serial_output.read_bytes()
    assert len(list((tmp_path / "checkpoints").glob("realization_*.json"))) == 4


def test_parallel_worker_failure_preserves_completed_checkpoints(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    _plugin(monkeypatch, shared, cfg, seed, fail_at_id=2)
    selection = _v3(tmp_path, manifest, backend="external", n=3, n_jobs=2)
    monkeypatch.setattr(checkpoint, "ProcessPoolExecutor", lambda *, max_workers, mp_context:
                        ThreadPoolExecutor(max_workers=max_workers))
    output = tmp_path / "old.json"
    output.write_text("existing data")
    with pytest.raises(RuntimeError, match="Deliberately failed"):
        run_backend_selection(selection, output)
    assert output.read_text() == "existing data"
    assert (tmp_path / "checkpoints" / "run.json").is_file()


def test_spawn_process_pool_can_run_importable_checkpoint_functions():
    """Use real spawn workers, not thread mocks, for multiprocessing portability."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    values = [{"a": 1}, {"b": 2}]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        assert list(pool.map(checkpoint._digest, values)) == [checkpoint._digest(x) for x in values]


def test_exclusive_checkpoint_directory_lock_prevents_second_writer(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    selection = _v3(tmp_path, manifest, n=2)
    parts = tmp_path / "checkpoints"
    parts.mkdir()
    (parts / ".runner.lock").write_text("active\n")
    with pytest.raises(RuntimeError, match="locked"):
        run_backend_selection(selection, tmp_path / "not-written.json")
    assert not (tmp_path / "not-written.json").exists()
