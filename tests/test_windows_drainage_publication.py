"""Windows-safe atomic publication for the parcel-routing CLI outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from utils import atomic_files
from utils.whitebox_parcel_routing import write_candidate_diagnostics


def _windows_lock(code: int) -> PermissionError:
    error = PermissionError(13, "Access is denied")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32])
def test_transient_windows_lock_retries_atomic_diagnostic_write(monkeypatch, tmp_path, code):
    """The original failure occurred on the temp-to-staged CSV rename."""
    original_replace = atomic_files.os.replace
    sleeps: list[float] = []
    attempts = 0

    def replacement(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise _windows_lock(code)
        return original_replace(source, destination)

    monkeypatch.setattr(atomic_files.os, "replace", replacement)
    monkeypatch.setattr(atomic_files, "sleep", sleeps.append)
    path = tmp_path / ".diagnostics.csv.run.staged"
    write_candidate_diagnostics(path, [
        {"upstream_pid": 11, "downstream_pid": None, "origin_cells": 2,
         "origin_area_m2": 1.0, "parcel_cells": 3, "share": 2 / 3,
         "decision": "outside_parcel_domain", "reason": "largest_share"},
    ])
    assert attempts == 3
    assert sleeps == [0.05, 0.1]
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["downstream_pid"] == "OUTSIDE"
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_windows_lock_preserves_prior_file(monkeypatch, tmp_path):
    original = tmp_path / "diagnostics.csv"
    original.write_bytes(b"previous completed file")
    source = tmp_path / "candidate.tmp"
    source.write_bytes(b"new content")
    attempts = 0

    def denied(_source, _destination):
        nonlocal attempts
        attempts += 1
        raise _windows_lock(5)

    monkeypatch.setattr(atomic_files.os, "replace", denied)
    monkeypatch.setattr(atomic_files, "sleep", lambda _: None)
    with pytest.raises(PermissionError, match="Access is denied"):
        atomic_files.replace_with_windows_retry(source, original)
    assert attempts == 7
    assert original.read_bytes() == b"previous completed file"
    assert source.read_bytes() == b"new content"


def test_non_windows_permission_error_is_not_retried(monkeypatch, tmp_path):
    attempts = 0

    def denied(*_args):
        nonlocal attempts
        attempts += 1
        raise PermissionError("actual file permission denial")

    monkeypatch.setattr(atomic_files.os, "replace", denied)
    monkeypatch.setattr(atomic_files, "sleep", lambda _: pytest.fail("unexpected retry"))
    with pytest.raises(PermissionError, match="actual file permission denial"):
        atomic_files.replace_with_windows_retry(tmp_path / "a", tmp_path / "b")
    assert attempts == 1


def test_failed_diagnostic_publication_cleans_temp_file(monkeypatch, tmp_path):
    monkeypatch.setattr(atomic_files.os, "replace", lambda *_: (_ for _ in ()).throw(_windows_lock(32)))
    monkeypatch.setattr(atomic_files, "sleep", lambda _: None)
    dest = tmp_path / "diagnostics.csv"
    dest.write_bytes(b"original")
    with pytest.raises(PermissionError):
        write_candidate_diagnostics(dest, [])
    assert dest.read_bytes() == b"original"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["diagnostics.csv"]
