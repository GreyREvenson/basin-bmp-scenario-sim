"""Opt-in, process-level portability gate for the *published* annual BMI path.

This is a reproducibility / public-interface acceptance check, not proof of
scientific validity or third-party CSDMS certification. It uses the normal
backend-selection APIs and a real ``spawn`` process pool (never threads).
The user's configuration and output files are not modified.
"""
from __future__ import annotations

import argparse
from abc import ABCMeta
from dataclasses import replace
from hashlib import sha256
import importlib
import inspect
import json
import platform
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import numpy as np

from basin_bmp_sim.execution.backend_selection import BackendSelection, EnsemblePlan
from basin_bmp_sim.execution.ensemble_runner import run_backend_ensemble
from basin_bmp_sim.interfaces.bmi.public_contract import verify_public_contract
from basin_bmp_sim.certification.release_evidence import source_identity
from basin_bmp_sim.interfaces.component_bootstrap import ComponentStartup


PORTABILITY_REPORT_SCHEMA_VERSION = 2


def check_official_bmipy() -> dict[str, Any]:
    """Check concrete adapters against the installed official BMI ABC.

    This supplements the frozen project contract. Subclassing ``bmipy.Bmi``
    is *not* required by the project, so a temporary mixed class checks the
    real ABC requirements without changing adapter inheritance or behavior.
    """
    try:
        bmipy = importlib.import_module("bmipy")
    except ImportError as exc:
        raise RuntimeError(
            "Official BMI check requires bmipy; install with pip install -e '.[bmi]'"
        ) from exc

    from basin_bmp_sim.interfaces.bmi import (
        ExternalWatershedModelBmi, LiveScenarioGeneratorBmi,
        MonteCarloDriverBmi, StochasticScenarioBmi, WatershedModelBmi,
    )

    required = set(bmipy.Bmi.__abstractmethods__)
    if not required:
        raise ValueError("Official bmipy.Bmi has no abstract methods")
    adapters = (
        ExternalWatershedModelBmi, LiveScenarioGeneratorBmi,
        MonteCarloDriverBmi, StochasticScenarioBmi, WatershedModelBmi,
    )
    names = []
    for cls in adapters:
        concrete = ABCMeta(f"{cls.__name__}OfficialBmiCheck", (cls, bmipy.Bmi), {})
        if inspect.isabstract(concrete):
            raise ValueError(
                f"{cls.__name__} does not implement official BMI methods: "
                f"{sorted(concrete.__abstractmethods__)}"
            )
        names.append(cls.__name__)
    return {
        "bmipy_version": str(getattr(bmipy, "__version__", "unknown")),
        "official_abstract_methods_checked": len(required),
        "adapters_checked": names,
    }


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_report(path: Path, report: dict[str, Any]) -> None:
    """Publish only after every check succeeds; never truncate old results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def certify_portability(selection_file: str | Path, *, n_scenarios: int = 2,
                        n_jobs: int = 2, require_bmipy: bool = True,
                        additional_files: tuple[str | Path, ...] = ()) -> dict[str, Any]:
    """Compare real serial and spawned-process ensembles and resume by bytes.

    The Phase 13 seed schedule and normal results are used for both sides.
    Both runs get independent checkpoint directories inside a temporary
    workspace; no existing user checkpoint or result is consumed/overwritten.
    This may be expensive for large model configurations. Run on a small
    representative configuration first.
    """
    if type(n_scenarios) is not int or n_scenarios < 2:
        raise ValueError("Portability certification requires at least two scenarios")
    if type(n_jobs) is not int or not 2 <= n_jobs <= 64:
        raise ValueError("Portability certification requires 2–64 process workers")
    selection_path = Path(selection_file).expanduser().resolve(strict=True)
    selection = BackendSelection.read(selection_path)
    root = ComponentStartup.read(selection.component_manifest)
    if root.replay_file is not None:
        raise ValueError("Portability certification cannot use a replay startup")
    verify_public_contract()
    official = check_official_bmipy() if require_bmipy else None
    identity = source_identity(selection_path, selection.component_manifest,
                               additional_files=additional_files)

    with TemporaryDirectory(prefix="basin-bmp-bmi-portability-") as temp_dir:
        work = Path(temp_dir)
        declared_assets = (selection.ensemble.additional_input_files
                           if selection.ensemble is not None else ())
        serial_plan = EnsemblePlan(
            n_scenarios=n_scenarios, n_jobs=1, checkpoint_dir=work / "serial",
            additional_input_files=declared_assets,
        )
        parallel_plan = EnsemblePlan(
            n_scenarios=n_scenarios, n_jobs=n_jobs, checkpoint_dir=work / "parallel",
            additional_input_files=declared_assets,
        )
        serial = work / "serial.json"
        parallel = work / "parallel.json"
        run_backend_ensemble(replace(selection, ensemble=serial_plan), serial)
        run_backend_ensemble(replace(selection, ensemble=parallel_plan), parallel)
        expected, actual = serial.read_bytes(), parallel.read_bytes()
        if expected != actual:
            raise ValueError("Spawned parallel ensemble differs from serial ensemble")

        # A resumed run must reuse every validated checkpoint and write the
        # exact same record. Corrupt or incomplete checkpoints fail in the
        # existing Phase 13 checkpoint validator.
        run_backend_ensemble(replace(
            selection, ensemble=replace(parallel_plan, resume=True)), parallel)
        if parallel.read_bytes() != expected:
            raise ValueError("Resumed parallel ensemble differs from serial ensemble")
        count = len(list((work / "parallel").glob("realization_*.json")))
        if count != n_scenarios:
            raise ValueError("Parallel ensemble did not produce the expected checkpoints")
        result = json.loads(expected)
        if (result.get("n_scenarios") != n_scenarios or
                len(result.get("scenarios", [])) != n_scenarios):
            raise ValueError("Parallel ensemble result has an incorrect scenario count")
        result_hash = sha256(expected).hexdigest()

    if source_identity(selection_path, selection.component_manifest,
                       additional_files=additional_files) != identity:
        raise ValueError("Release evidence scientific inputs changed during portability run")
    return {
        "portability_schema_version": PORTABILITY_REPORT_SCHEMA_VERSION,
        "scope": "project_annual_bmi_serial_parallel_spawn_and_resume_not_csdms_certification",
        "backend": selection.backend,
        "external_factory": selection.external_factory,
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "process_start_method": "spawn",
        "n_scenarios": n_scenarios,
        "n_jobs": n_jobs,
        "serial_parallel_byte_identical": True,
        "resume_byte_identical": True,
        "result_sha256": result_hash,
        **identity,
        "official_bmipy": official if official is not None else "not_checked",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify annual BMI serial/spawn/resume repeatability for a backend selection"
    )
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--n-scenarios", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--without-bmipy", action="store_true",
                        help="Run process checks only; report official BMI as unchecked")
    parser.add_argument("--input-file", action="append", default=[],
                        help="Additional scientific asset (repeatable; selection-relative)")
    args = parser.parse_args()
    report = certify_portability(
        args.selection_json, n_scenarios=args.n_scenarios,
        n_jobs=args.n_jobs, require_bmipy=not args.without_bmipy,
        additional_files=tuple(args.input_file),
    )
    target = args.report_json.expanduser().resolve()
    _atomic_report(target, report)
    print(target)


if __name__ == "__main__":
    main()
