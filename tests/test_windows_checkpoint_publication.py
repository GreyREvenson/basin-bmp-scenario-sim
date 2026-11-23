"""Atomic checkpoint publication and completed-resume behavior."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.driver import ensemble_checkpoints as checkpoints


def _windows_permission_error(code: int = 5) -> PermissionError:
    error = PermissionError(13, "Access is denied")
    error.winerror = code
    return error


def _publish(destination: Path) -> None:
    checkpoints._write_streamed_result(
        destination,
        {"result_schema_version": 1, "backend": "test", "n_scenarios": 1},
        ({"scenario_id": 10, "worker_seed": 42},),
    )


def _assert_no_temporary_files(tmp_path: Path) -> None:
    assert list(tmp_path.glob(".*.tmp")) == []


def test_identical_completed_resume_does_not_replace_existing_result(tmp_path, monkeypatch):
    destination = tmp_path / "ensemble.json"
    _publish(destination)
    expected = destination.read_bytes()
    before_mtime = destination.stat().st_mtime_ns

    def fail_if_replaced(self, target):
        pytest.fail("An unchanged result should not be replaced")

    monkeypatch.setattr(Path, "replace", fail_if_replaced)
    _publish(destination)
    assert destination.read_bytes() == expected
    assert destination.stat().st_mtime_ns == before_mtime
    _assert_no_temporary_files(tmp_path)


@pytest.mark.parametrize("winerror", [5, 32])
def test_retry_transient_windows_replacement_failure(tmp_path, monkeypatch, winerror):
    destination = tmp_path / "ensemble.json"
    destination.write_bytes(b"previous result")
    original_replace = Path.replace
    calls = []
    delays = []

    def flaky_replace(self, target):
        calls.append((self, target))
        if len(calls) <= 2:
            raise _windows_permission_error(winerror)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(checkpoints, "sleep", delays.append)
    _publish(destination)
    assert len(calls) == 3
    assert delays == [0.05, 0.1]
    assert destination.read_bytes() != b"previous result"
    _assert_no_temporary_files(tmp_path)


def test_persistent_windows_denial_preserves_existing_result(tmp_path, monkeypatch):
    destination = tmp_path / "ensemble.json"
    destination.write_bytes(b"previous result")
    calls = []
    delays = []

    def always_denied(self, target):
        calls.append(target)
        raise _windows_permission_error()

    monkeypatch.setattr(Path, "replace", always_denied)
    monkeypatch.setattr(checkpoints, "sleep", delays.append)
    with pytest.raises(PermissionError, match="Access is denied"):
        _publish(destination)
    assert len(calls) == 7
    assert len(delays) == 6
    assert destination.read_bytes() == b"previous result"
    _assert_no_temporary_files(tmp_path)


def test_non_windows_permission_denial_is_not_retried(tmp_path, monkeypatch):
    destination = tmp_path / "ensemble.json"
    destination.write_bytes(b"previous result")
    calls = []

    def denied(self, target):
        calls.append(target)
        raise PermissionError("ordinary permission denial")

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(checkpoints, "sleep", lambda _: pytest.fail("Must not retry"))
    with pytest.raises(PermissionError, match="ordinary permission denial"):
        _publish(destination)
    assert len(calls) == 1
    assert destination.read_bytes() == b"previous result"
    _assert_no_temporary_files(tmp_path)


def test_checkpoint_metadata_publication_retries_windows_denial(tmp_path, monkeypatch):
    destination = tmp_path / "run.json"
    original_replace = Path.replace
    calls = []
    monkeypatch.setattr(checkpoints, "sleep", lambda _: None)

    def flaky_replace(self, target):
        calls.append(target)
        if len(calls) == 1:
            raise _windows_permission_error(32)
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    checkpoints._atomic_json(destination, {"run": 1})
    assert len(calls) == 2
    assert destination.read_text(encoding="utf-8") == '{\n  "run": 1\n}\n'
    _assert_no_temporary_files(tmp_path)


def test_modified_published_result_is_rebuilt_from_checkpoint_records(tmp_path):
    destination = tmp_path / "ensemble.json"
    _publish(destination)
    original = destination.read_bytes()
    destination.write_bytes(b"incomplete or modified result")
    _publish(destination)
    assert destination.read_bytes() == original
    _assert_no_temporary_files(tmp_path)
