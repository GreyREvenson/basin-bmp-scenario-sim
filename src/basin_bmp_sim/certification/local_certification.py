"""Run release-evidence checks locally, without a hosted CI service.

This is an orchestration tool, not a new BMI interface or a scientific model.
A completed report is evidence for architecture review, not certification of
scientific validity or third-party BMI endorsement.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

from basin_bmp_sim.execution.backend_selection import BackendSelection
from basin_bmp_sim.certification.acceptance import accept_backend
from basin_bmp_sim.certification.portability import _atomic_report, certify_portability
from basin_bmp_sim.certification.release_evidence import source_identity
from basin_bmp_sim.certification.release_gate import evaluate_release_gate
from basin_bmp_sim.interfaces.component_bootstrap import ComponentStartup


IDENTITY_KEYS = (
    "selection_sha256", "startup_sha256", "model_config_sha256",
    "public_contract_sha256", "scientific_inputs_sha256",
    "external_backend_source_sha256", "additional_scientific_input_count",
)


def _identity(report: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(report.get(key) for key in IDENTITY_KEYS)


def run_local_certification(
    selection_file: str | Path,
    evidence_dir: str | Path,
    *,
    n_scenarios: int = 2,
    n_jobs: int = 2,
    require_bmipy: bool = True,
    additional_files: Sequence[str | Path] = (),
    operating_system: str | None = None,
) -> dict[str, Any]:
    """Collect acceptance and spawn evidence on the *actual* local OS.

    The other operating system's report can be copied into ``evidence_dir``.
    All execution and report comparisons finish before any existing evidence
    is replaced. Cross-platform report consistency is checked before publish.
    A run without official bmipy is useful for diagnostics but will not satisfy
    the release gate.
    """
    selection_file = Path(selection_file).expanduser().resolve(strict=True)
    selection = BackendSelection.read(selection_file)
    startup = ComponentStartup.read(selection.component_manifest)
    if selection.backend != "external" or selection.ensemble is not None:
        raise ValueError("Local certification requires a single-scenario external backend selection")
    if startup.replay_file is not None:
        raise ValueError("Local certification requires a fresh startup without replay_file")
    system = platform.system()
    if operating_system is not None and operating_system != system:
        raise ValueError("Cannot certify an operating system other than the current host")
    if system not in {"Windows", "Linux"}:
        raise ValueError("Local release portability certification is supported on Windows and Linux")
    if type(n_scenarios) is not int or n_scenarios < 2 or type(n_jobs) is not int or not 2 <= n_jobs <= 64:
        raise ValueError("Local certification requires at least two scenarios and 2–64 workers")

    evidence_dir = Path(evidence_dir).expanduser().resolve()
    output_names = {"acceptance.json", f"{system.lower()}.json"}
    inputs = {selection_file, selection.component_manifest, startup.model_config}
    for item in additional_files:
        candidate = Path(item).expanduser()
        inputs.add((candidate if candidate.is_absolute() else selection_file.parent / candidate)
                   .resolve(strict=True))
    if any((evidence_dir / name).resolve() in inputs for name in output_names):
        raise ValueError("Evidence output would overwrite an input configuration file")

    # Preflight provenance prior to the expensive scientific runs. In
    # particular, third-party assets must be explicitly declared.
    expected = source_identity(selection_file, selection.component_manifest,
                               additional_files=additional_files)
    if expected["additional_scientific_input_count"] < 1:
        raise ValueError("External backend certification requires at least one declared scientific asset")

    acceptance = accept_backend(selection_file, additional_files=tuple(additional_files))
    portability = certify_portability(
        selection_file, n_scenarios=n_scenarios, n_jobs=n_jobs,
        require_bmipy=require_bmipy, additional_files=tuple(additional_files),
    )
    if (portability.get("operating_system") != system or
            acceptance.get("backend") != "external" or
            portability.get("backend") != "external" or
            acceptance.get("external_factory") != selection.external_factory or
            portability.get("external_factory") != selection.external_factory or
            _identity(acceptance) != _identity(portability) or
            _identity(acceptance) != _identity(expected) or
            source_identity(selection_file, selection.component_manifest,
                            additional_files=additional_files) != expected):
        raise ValueError("Acceptance and portability evidence refer to different scientific inputs or backends")

    # Stage files and validate the release gate, including any existing report
    # from the other OS, before replacing anything in the destination.
    with TemporaryDirectory(prefix="bmi-evidence-stage-") as temp:
        staging = Path(temp)
        staged_acceptance = staging / "acceptance.json"
        staged_portability = staging / f"{system.lower()}.json"
        _atomic_report(staged_acceptance, acceptance)
        _atomic_report(staged_portability, portability)
        windows = (staged_portability if system == "Windows"
                   else evidence_dir / "windows.json")
        linux = (staged_portability if system == "Linux"
                 else evidence_dir / "linux.json")
        windows = windows if windows.is_file() else None
        linux = linux if linux.is_file() else None
        # Official BMI is optional for diagnostic runs. The release gate
        # rejects unchecked reports; do not pretend those checks passed.
        if require_bmipy:
            review = evaluate_release_gate(
                acceptance=staged_acceptance, windows=windows, linux=linux)
        else:
            review = {"ready": False, "status": "pending_official_bmipy",
                      "limitations": ["Official bmipy check was not run"]}
            # Still check cross-platform provenance when another report exists.
            other = linux if system == "Windows" else windows
            if other is not None:
                data = json.loads(other.read_text(encoding="utf-8"))
                if (_identity(data) != _identity(expected) or
                        data.get("external_factory") != selection.external_factory):
                    raise ValueError("Existing cross-platform evidence refers to different scientific inputs or backend")

        evidence_dir.mkdir(parents=True, exist_ok=True)
        _atomic_report(evidence_dir / "acceptance.json", acceptance)
        _atomic_report(evidence_dir / f"{system.lower()}.json", portability)
    return {
        "operating_system": system,
        "acceptance_report": str(evidence_dir / "acceptance.json"),
        "portability_report": str(evidence_dir / f"{system.lower()}.json"),
        "release_gate_status": review["status"],
        "ready_for_architecture_release_review": review["ready"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect public BMI acceptance and serial/spawn/resume evidence locally")
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--n-scenarios", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--input-file", action="append", default=[],
                        help="Extra scientific asset, relative to selection JSON (repeatable)")
    parser.add_argument("--without-bmipy", action="store_true",
                        help="Diagnostic run only; cannot satisfy release gate")
    args = parser.parse_args()
    result = run_local_certification(
        args.selection_json, args.evidence_dir,
        n_scenarios=args.n_scenarios, n_jobs=args.n_jobs,
        additional_files=tuple(args.input_file),
        require_bmipy=not args.without_bmipy,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
