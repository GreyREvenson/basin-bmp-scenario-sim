"""Phase 13: bounded parallel workers, durable checkpoints and resumable BMI ensembles.

Version 3 is opt-in. Workers exchange only JSON-compatible Phase 11 records;
parent process alone writes checkpoints and the final published result. The
Phase 12 seed policy, result schema, statistics and replay schema are unchanged.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from src.interfaces.component_bootstrap import ComponentStartup

CHECKPOINT_SCHEMA_VERSION = 1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _run_one_worker(selection, model_config: str, index: int, seed: int,
                    scenario_id: int) -> dict[str, Any]:
    """Top-level, spawn-picklable worker; no shared model or RNG state."""
    from src.driver.backend_selection import run_selected_scenario

    with TemporaryDirectory(prefix="basin-bmp-worker-") as directory:
        manifest = Path(directory) / "startup.json"
        manifest.write_text(json.dumps({
            "schema_version": 1, "model_config": model_config,
            "worker_seed": seed, "scenario_id": scenario_id,
        }), encoding="utf-8")
        return run_selected_scenario(replace(selection, component_manifest=manifest, ensemble=None))


def _fingerprint_input(selection, root: ComponentStartup) -> dict[str, Any]:
    """Bind checkpoints to the source YAML and selected scientific interface.

    Auxiliary data referenced by the YAML, external plugin code, and their
    versions cannot be fingerprinted generically; users must freeze them for
    scientifically valid resume. A changed YAML is detected here.
    """
    return {
        "algorithm": "phase13-checkpoints-v1",
        "backend": selection.backend,
        "external_factory": selection.external_factory,
        "backend_options": dict(selection.backend_options or {}),
        "required_features": sorted(selection.required_features),
        "model_config": str(root.model_config),
        "model_config_sha256": sha256(root.model_config.read_bytes()).hexdigest(),
        "seed_policy": "numpy-seedsequence-spawn-uint64-v1",
        "base_seed": root.worker_seed,
        "first_scenario_id": root.scenario_id,
        "n_scenarios": selection.ensemble.n_scenarios,
    }


def _atomic_json(path: Path, value: Any) -> None:
    """Write a complete JSON file by atomic same-directory replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_directory_lock(directory: Path):
    """Prevent concurrent checkpoint writers; a hard-kill may leave a lock.

    If a process dies without running ``finally``, remove ``.runner.lock``
    *only after verifying that no earlier run is active*.
    """
    path = directory / ".runner.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Ensemble checkpoint directory is locked: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def _record_file(directory: Path, index: int) -> Path:
    return directory / f"realization_{index:08d}.json"


def _record_envelope(record: Mapping[str, Any], *, fingerprint: str,
                     index: int, seed: int, scenario_id: int) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "index": index,
        "scenario_id": scenario_id,
        "worker_seed": seed,
        "record_sha256": _digest(record),
        "record": record,
    }


def _load_record(directory: Path, index: int, *, fingerprint: str,
                 seed: int, scenario_id: int, backend: str) -> dict[str, Any]:
    path = _record_file(directory, index)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable ensemble checkpoint: {path}") from exc
    if not isinstance(stored, dict) or any(stored.get(k) != v for k, v in {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "index": index, "scenario_id": scenario_id, "worker_seed": seed,
    }.items()):
        raise ValueError(f"Ensemble checkpoint identity/fingerprint mismatch: {path}")
    record = stored.get("record")
    if not isinstance(record, dict) or stored.get("record_sha256") != _digest(record):
        raise ValueError(f"Ensemble checkpoint checksum mismatch: {path}")
    if (record.get("scenario_id") != scenario_id or record.get("worker_seed") != seed or
            record.get("backend") != backend or record.get("result_schema_version") != 1):
        raise ValueError(f"Ensemble checkpoint result identity mismatch: {path}")
    replay = record.get("replay_scenario")
    if (not isinstance(replay, dict) or replay.get("scenario_id") != scenario_id or
            replay.get("seed") != seed):
        raise ValueError(f"Ensemble checkpoint replay identity mismatch: {path}")
    return record


class _DiskRecords(Sequence[Mapping[str, Any]]):
    """Reload validated records on iteration; never retains all BMP histories."""

    def __init__(self, directory: Path, *, fingerprint: str, seeds: tuple[int, ...],
                 first_scenario_id: int, backend: str):
        self.directory = directory
        self.fingerprint = fingerprint
        self.seeds = seeds
        self.first_scenario_id = first_scenario_id
        self.backend = backend

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if not isinstance(index, int):
            raise TypeError("Checkpoint index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return _load_record(self.directory, index, fingerprint=self.fingerprint,
                            seed=self.seeds[index],
                            scenario_id=self.first_scenario_id + index,
                            backend=self.backend)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for index in range(len(self)):
            yield self[index]


def _write_streamed_result(destination: str | Path, summary: dict[str, Any],
                           records: Sequence[Mapping[str, Any]]) -> Path:
    """Publish Phase 12 JSON without collecting all scenario records in RAM.

    The statistics still retain compact numeric arrays across realizations to
    compute exact percentiles, but BMP histories are read one at a time.
    """
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=dest.parent,
                                prefix=f".{dest.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            header = json.dumps(summary, indent=2, allow_nan=False)
            file.write(header[:-2] + ',\n  "scenarios": [\n')
            for index, record in enumerate(records):
                if index:
                    file.write(",\n")
                # Match json.dumps(..., indent=2) nested object indentation.
                file.write("    " + json.dumps(record, indent=2, allow_nan=False).replace("\n", "\n    "))
            file.write("\n  ]\n}\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(dest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dest


def _parallel_missing(selection, root: ComponentStartup, seeds: tuple[int, ...],
                      missing: list[int], directory: Path, fingerprint: str) -> None:
    """Bound in-flight futures to 2 * n_jobs; checkpoint in parent only."""
    n_jobs = selection.ensemble.n_jobs
    model_config = str(root.model_config)

    def store(index: int, record: dict[str, Any]) -> None:
        scenario_id = root.scenario_id + index
        # Reject identity inconsistencies before persistence; deeper checks
        # follow via the Phase 12 aggregate validator before publication.
        if (record.get("scenario_id") != scenario_id or
                record.get("worker_seed") != seeds[index] or
                record.get("backend") != selection.backend or
                record.get("result_schema_version") != 1):
            raise ValueError(f"Worker {index} returned incompatible result metadata")
        _atomic_json(_record_file(directory, index), _record_envelope(
            record, fingerprint=fingerprint, index=index, seed=seeds[index],
            scenario_id=scenario_id))

    if n_jobs == 1:
        for index in missing:
            store(index, _run_one_worker(selection, model_config, index, seeds[index],
                                         root.scenario_id + index))
        return

    context = multiprocessing.get_context("spawn")  # Windows/macOS/Linux portability
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=context) as executor:
        source = iter(missing)
        pending = {}

        def launch() -> None:
            while len(pending) < 2 * n_jobs:
                try:
                    index = next(source)
                except StopIteration:
                    return
                future = executor.submit(_run_one_worker, selection, model_config,
                                         index, seeds[index], root.scenario_id + index)
                pending[future] = index

        launch()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    store(index, future.result())
                launch()
        except BaseException:
            for future in pending:
                future.cancel()
            raise


def run_checkpointed_ensemble(selection, result_file: str | Path) -> Path:
    """Run/resume a version 3 ensemble, preserving completed validated work.

    A fresh run refuses nonempty checkpoint directories; resume checks run
    identity *and* each record before starting missing scenarios. The published
    result is atomically replaced only when all scenarios validate.
    """
    from src.driver.ensemble_runner import ensemble_worker_seeds, summarize_ensemble

    plan = selection.ensemble
    if plan is None or plan.checkpoint_dir is None:
        raise ValueError("Checkpointed ensemble requires version 3 configuration")
    root = ComponentStartup.read(selection.component_manifest)
    if root.replay_file is not None:
        raise ValueError("Ensemble startup cannot set replay_file")
    seeds = ensemble_worker_seeds(root.worker_seed, plan.n_scenarios)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Ensemble worker seed collision; choose a different base seed")
    directory = plan.checkpoint_dir
    run_input = _fingerprint_input(selection, root)
    fingerprint = _digest(run_input)
    metadata = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "run_identity": run_input,
    }
    # Creating the folder is harmless; the lock is acquired before reading
    # or writing any run metadata. Never start two writers in one directory.
    if not plan.resume:
        directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise ValueError("Checkpoint directory does not exist for resume")
    with _exclusive_directory_lock(directory):
        return _run_locked(selection, result_file, root, seeds, directory,
                           fingerprint, metadata)


def _run_locked(selection, result_file, root, seeds, directory, fingerprint, metadata):
    from src.driver.ensemble_runner import summarize_ensemble

    plan = selection.ensemble
    run_file = directory / "run.json"
    if plan.resume:
        if not run_file.is_file():
            raise ValueError("Resume requires an existing checkpoint run.json")
        try:
            existing = json.loads(run_file.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid ensemble checkpoint run.json") from exc
        if existing != metadata:
            raise ValueError("Ensemble checkpoint run identity/fingerprint mismatch")
    else:
        if any(path.name != ".runner.lock" for path in directory.iterdir()):
            raise FileExistsError("Checkpoint directory is not empty; choose a new directory or resume")
        _atomic_json(run_file, metadata)

    # Validate *all* existing entries before doing new science. Incomplete
    # temporary files can be ignored after an interrupted atomic write.
    expected_names = {f"realization_{i:08d}.json" for i in range(len(seeds))}
    unknown = [path for path in directory.glob("realization_*.json") if
               path.name not in expected_names]
    if unknown:
        raise ValueError("Unknown scenario checkpoints in ensemble directory")
    records = _DiskRecords(directory, fingerprint=fingerprint, seeds=seeds,
                           first_scenario_id=root.scenario_id, backend=selection.backend)
    missing = []
    for index in range(len(seeds)):
        if _record_file(directory, index).is_file():
            _ = records[index]
        else:
            missing.append(index)
    _parallel_missing(selection, root, seeds, missing, directory, fingerprint)
    summary = summarize_ensemble(records, backend=selection.backend,
                                 base_seed=root.worker_seed,
                                 first_scenario_id=root.scenario_id,
                                 include_scenarios=False)
    return _write_streamed_result(result_file, summary, records)
