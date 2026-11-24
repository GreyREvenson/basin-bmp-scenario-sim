"""Evidence-based annual BMI architecture-release gate.

This gate evaluates locally supplied test reports and an immutable public BMI
contract. It never claims that an absent cross-platform run has succeeded.
The reports themselves are not signed CI attestations or scientific reviews.
"""
from __future__ import annotations

import argparse
import ast
from hashlib import sha256
from importlib import resources
import json
from pathlib import Path
from typing import Any

from basin_bmp_sim.interfaces.bmi.public_contract import verify_public_contract
from basin_bmp_sim.certification.release_evidence import public_contract_sha256
from basin_bmp_sim.certification.portability import _atomic_report

GATE_SCHEMA_VERSION = 2
_IDENTITY_FIELDS = (
    "selection_sha256", "startup_sha256", "model_config_sha256",
    "public_contract_sha256", "scientific_inputs_sha256",
    "external_backend_source_sha256", "additional_scientific_input_count",
)
_PUBLIC_REFERENCE_IMPORTS = frozenset({
    "basin_bmp_sim.interfaces.annual_model_contract",
    "basin_bmp_sim.interfaces.backend_capabilities",
})


def audit_public_reference_backend(source: str | None = None) -> dict[str, Any]:
    """Ensure the bundled demonstration backend imports no private EPA model.

    This *static* audit is deliberately limited to the bundled reference
    backend; it does not claim to audit arbitrary third-party dependencies.
    """
    if source is None:
        source = (resources.files("reference_backend").joinpath("annual_linear.py")
                  .read_text(encoding="utf-8"))
    tree = ast.parse(source)
    checked: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "basin_bmp_sim" or node.module.startswith("basin_bmp_sim.")):
            checked.add(node.module)
        elif isinstance(node, ast.Import):
            checked.update(alias.name for alias in node.names
                           if alias.name == "basin_bmp_sim" or alias.name.startswith("basin_bmp_sim."))
    forbidden = checked - _PUBLIC_REFERENCE_IMPORTS
    if forbidden:
        raise ValueError("Reference backend imports unpublished model APIs: "
                         + ", ".join(sorted(forbidden)))
    return {"passed": True, "project_imports": sorted(checked)}


def _read_report(path: str | Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Release evidence must be a file: {path}")
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed release evidence JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Release evidence must contain a JSON object: {path}")
    return data, {"sha256": sha256(raw).hexdigest(), "file": str(path)}


def _sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def _validate_identity(data: dict[str, Any], label: str) -> tuple[str, ...]:
    identity = tuple(data.get(field) for field in _IDENTITY_FIELDS)
    if not all(_sha(value) for value in identity[:-1]) or (
            type(identity[-1]) is not int or identity[-1] < 1):
        raise ValueError(f"{label} lacks valid versioned source and contract hashes")
    if data["public_contract_sha256"] != public_contract_sha256():
        raise ValueError(f"{label} was generated for a different public BMI contract")
    return identity


def _validate_acceptance(data: dict[str, Any]) -> tuple[str, ...]:
    if (data.get("acceptance_schema_version") != 2 or
            data.get("scope") != "project_annual_bmi_black_box_not_csdms_certification" or
            data.get("backend") != "external" or
            not isinstance(data.get("external_factory"), str) or
            ":" not in data["external_factory"] or
            data.get("deterministic_repeat") is not True or
            type(data.get("bmp_count")) is not int or data["bmp_count"] < 0 or
            not _sha(data.get("result_sha256"))):
        raise ValueError("External-backend acceptance evidence is incomplete or incompatible")
    for key in ("generator", "model"):
        value = data.get(key)
        if not isinstance(value, dict) or not isinstance(value.get("variables"), list):
            raise ValueError("Acceptance evidence lacks a public BMI variable probe")
    return _validate_identity(data, "Acceptance report")


def _validate_portability(data: dict[str, Any], operating_system: str) -> tuple[str, ...]:
    official = data.get("official_bmipy")
    required_adapters = {
        "ExternalWatershedModelBmi", "LiveScenarioGeneratorBmi",
        "WatershedModelBmi", "MonteCarloDriverBmi", "StochasticScenarioBmi",
    }
    if (data.get("portability_schema_version") != 2 or
            data.get("scope") != "project_annual_bmi_serial_parallel_spawn_and_resume_not_csdms_certification" or
            data.get("backend") != "external" or
            not isinstance(data.get("external_factory"), str) or
            ":" not in data["external_factory"] or
            data.get("operating_system") != operating_system or
            data.get("process_start_method") != "spawn" or
            type(data.get("n_scenarios")) is not int or data["n_scenarios"] < 2 or
            type(data.get("n_jobs")) is not int or data["n_jobs"] < 2 or
            data.get("serial_parallel_byte_identical") is not True or
            data.get("resume_byte_identical") is not True or
            not _sha(data.get("result_sha256")) or
            not isinstance(official, dict) or
            type(official.get("official_abstract_methods_checked")) is not int or
            official["official_abstract_methods_checked"] < 1 or
            not isinstance(official.get("adapters_checked"), list) or
            not required_adapters.issubset(set(
                str(name) for name in official["adapters_checked"]))):
        raise ValueError(f"{operating_system} report lacks completed spawned-process and official BMI checks")
    return _validate_identity(data, f"{operating_system} portability report")


def evaluate_release_gate(*, acceptance: str | Path | None = None,
                          windows: str | Path | None = None,
                          linux: str | Path | None = None) -> dict[str, Any]:
    """Evaluate proof of an annual BMI *architecture* release candidate.

    All three reports must come from the same backend selection, startup,
    model YAML and published contract. Matching bytes across operating
    systems is not demanded: each OS must independently pass serial/spawn and
    resume reproducibility for that fixed scientific configuration.
    """
    verify_public_contract()
    audit = audit_public_reference_backend()
    evidence: dict[str, dict[str, str]] = {}
    statuses: dict[str, str] = {"public_contract": "passed",
                                  "reference_backend_public_imports": "passed"}
    identity: tuple[str, ...] | None = None
    factory: str | None = None
    for label, source, validate in (
        ("external_backend_acceptance", acceptance, _validate_acceptance),
        ("windows_spawn_and_bmipy", windows, lambda data: _validate_portability(data, "Windows")),
        ("linux_spawn_and_bmipy", linux, lambda data: _validate_portability(data, "Linux")),
    ):
        if source is None:
            statuses[label] = "missing"
            continue
        data, origin = _read_report(source)
        current = validate(data)
        if identity is not None and current != identity:
            raise ValueError("Release reports refer to different selection, startup or scientific configuration")
        if factory is not None and data["external_factory"] != factory:
            raise ValueError("Release reports refer to different external backend factories")
        identity = current
        factory = data["external_factory"]
        evidence[label] = origin
        statuses[label] = "passed"
    ready = all(status == "passed" for status in statuses.values())
    return {
        "release_gate_schema_version": GATE_SCHEMA_VERSION,
        "status": "ready_for_architecture_release_review" if ready else "pending_evidence",
        "ready": ready,
        "public_contract_sha256": public_contract_sha256(),
        "reference_backend_audit": audit,
        "checks": statuses,
        "evidence": evidence,
        "backend_factory": factory,
        "limitations": [
            "Evidence reports are local unsigned JSON; confirm CI provenance independently.",
            "This does not certify CSDMS compliance or scientific correctness of any backend.",
            "Undeclared scientific inputs, transitive backend source files, dependency versions and native-only diagnostics remain outside this gate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Review independent annual BMI architecture-release evidence")
    parser.add_argument("--acceptance", type=Path, help="External-backend acceptance JSON")
    parser.add_argument("--windows", type=Path, help="Windows BMI portability JSON")
    parser.add_argument("--linux", type=Path, help="Linux BMI portability JSON")
    parser.add_argument("--report", type=Path, help="Write a deterministic JSON readiness report")
    parser.add_argument("--require-ready", action="store_true",
                        help="Exit nonzero if evidence is missing (or invalid)")
    args = parser.parse_args()
    report = evaluate_release_gate(acceptance=args.acceptance,
                                   windows=args.windows, linux=args.linux)
    if args.report is not None:
        _atomic_report(args.report.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_ready and not report["ready"]:
        parser.exit(1, "Architecture release gate: required evidence is missing\n")


if __name__ == "__main__":
    main()
