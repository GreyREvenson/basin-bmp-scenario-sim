"""Phase 18: reproducible, RNG-free export of *recorded* BMP realizations.

Read actual Phase 12/13 replay data already persisted in an ensemble; never
resample or infer missing scientific baseline inputs.  The compact trace is
compatible with Phase 17's schema.  Its aggregate quantities are *diagnostics*,
not individual independent input draws or causal sensitivity estimands.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.driver.ensemble_analysis import _validated_rows
from src.driver.ensemble_checkpoints import _atomic_json

TRACE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1


def _finite(value: Any, label: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum) or (
            maximum is not None and number > maximum):
        raise ValueError(f"{label} must be a finite number in its physical domain")
    return number


def extract_realization_trace(ensemble: Mapping[str, Any], *, source_sha256: str,
                              confirm_external_independence: bool = False
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a Phase 17-ready feature matrix and inspectable placement ledger.

    The source must contain complete embedded replay definitions; no guessed
    samples or RNG calls are permitted. The scientifically independent units
    here are annual *scenarios*, not their potentially dependent BMPs/parcels.
    """
    if not isinstance(ensemble, dict):
        raise ValueError("Input ensemble must be a JSON object")
    if type(confirm_external_independence) is not bool:
        raise ValueError("confirm_external_independence must be boolean")
    manifest, _, _ = _validated_rows(ensemble)
    backend = ensemble["backend"]
    if backend == "external" and not confirm_external_independence:
        raise ValueError("External backend requires explicit independent-draw confirmation")
    pollutants = manifest.get("pollutants")
    pathways = manifest.get("pathway_names")
    cps_codes = manifest.get("bmp_cps")
    if any(not isinstance(v, list) or not v for v in (pollutants, pathways, cps_codes)):
        raise ValueError("Complete pollutant, pathway and CPS manifest required for recorded-input trace")
    if (any(type(cps) is not int for cps in cps_codes) or len(set(cps_codes)) != len(cps_codes)
            or any(type(s) is not str or not s.strip() for v in (pollutants, pathways) for s in v)
            or len(set(pathways)) != len(pathways)):
        raise ValueError("Invalid ordered domain metadata for recorded-input trace")

    features: list[dict[str, str]] = []
    def add(name: str, units: str) -> None:
        features.append({"name": name, "units": units, "role": "realized_diagnostic"})

    add("realized_bmp_count", "count")
    add("realized_failed_bmp_count", "count")
    add("realized_bmp_design_cost_quantity_sum", "ha")
    add("realized_treatment_fraction_sum", "1")
    add("realized_recovery_gain_sum", "1")
    for cps in cps_codes:
        add(f"realized_bmp_count__cps_{cps}", "count")
    for pol in pollutants:
        for pathway in pathways:
            add(f"realized_signed_efficiency_sum__{pol}__{pathway}", "1")
    if len(features) > 100:
        raise ValueError("Phase 17 accepts at most 100 trace features; reduce the model domain")
    if len({f["name"] for f in features}) != len(features):
        raise ValueError("Trace feature name collision in domain metadata")

    rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for i, (summary, scenario) in enumerate(zip(ensemble["scenario_summaries"], ensemble["scenarios"])):
        replay = scenario.get("replay_scenario")
        responses = scenario.get("bmp_responses")
        if not isinstance(replay, dict) or not isinstance(responses, list):
            raise ValueError(f"Missing replay or BMP response list for scenario {i}")
        if (replay.get("scenario_id") != summary["scenario_id"] or
                replay.get("seed") != summary["worker_seed"] or
                type(replay.get("schema_version")) is not int or replay["schema_version"] != 1):
            raise ValueError(f"Replay version or scenario identity mismatch for scenario {i}")
        placements = replay.get("placements")
        if not isinstance(placements, list) or len(placements) != len(responses) or (
                len(placements) != summary["bmp_count"]):
            raise ValueError(f"Replay placement count mismatch for scenario {i}")
        counts = {cps: 0 for cps in cps_codes}
        effectiveness = {(pol, path): 0.0 for pol in pollutants for path in pathways}
        failed = 0
        quantity = 0.0
        fractions = 0.0
        recovery = 0.0
        for j, (placement, response) in enumerate(zip(placements, responses)):
            if not isinstance(placement, dict) or not isinstance(response, dict):
                raise ValueError(f"Invalid recorded BMP placement {i}:{j}")
            cps = placement.get("cps")
            if type(cps) is not int or cps not in counts or cps != response.get("cps"):
                raise ValueError(f"Recorded CPS differs from domain or response at {i}:{j}")
            if placement.get("parcel_index") != response.get("host_parcel_index"):
                raise ValueError(f"Recorded host parcel differs from response at {i}:{j}")
            if _finite(placement.get("cost_usd"), "Recorded placement cost", minimum=0) != _finite(
                    response.get("cost_usd"), "Response placement cost", minimum=0):
                raise ValueError(f"Recorded cost differs from response at {i}:{j}")
            design = placement.get("design")
            if not isinstance(design, dict) or not isinstance(design.get("treated"), list):
                raise ValueError(f"Invalid recorded BMP design at {i}:{j}")
            quantity += _finite(design.get("cost_quantity_ha"), "Design cost quantity", minimum=0)
            for affected in design["treated"]:
                if (not isinstance(affected, list) or len(affected) != 2 or
                        type(affected[0]) is not int or affected[0] < 0 or
                        affected[0] >= len(manifest["parcel_ids"])):
                    raise ValueError(f"Invalid treated parcel in recorded design at {i}:{j}")
                fractions += _finite(affected[1], "Treated fraction", minimum=0, maximum=1)
            if type(placement.get("failed")) is not bool:
                raise ValueError(f"Invalid recorded BMP failure status at {i}:{j}")
            failed += int(placement["failed"])
            gains = placement.get("recovery_gains")
            if not isinstance(gains, list):
                raise ValueError(f"Missing recorded AFO recovery gains at {i}:{j}")
            recovery += sum(_finite(v, "AFO recovery gain", minimum=0, maximum=1) for v in gains)
            maps = placement.get("efficiency_by_pollutant")
            if not isinstance(maps, list) or len(maps) != len(pollutants):
                raise ValueError(f"Recorded pollutant efficiency shape mismatch at {i}:{j}")
            for pol, values in zip(pollutants, maps):
                if not isinstance(values, dict) or set(values) != set(pathways):
                    raise ValueError(f"Recorded pathway efficiency shape mismatch at {i}:{j}")
                for path in pathways:
                    effectiveness[pol, path] += _finite(values[path], "Signed BMP efficiency", maximum=1)
            counts[cps] += 1
        values = [float(len(placements)), float(failed), quantity, fractions, recovery]
        values += [float(counts[cps]) for cps in cps_codes]
        values += [effectiveness[pol, path] for pol in pollutants for path in pathways]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Non-finite realized diagnostic for scenario {i}")
        rows.append({"scenario_id": summary["scenario_id"],
                     "worker_seed": summary["worker_seed"], "values": values})
        # Actual recorded realized values are retained verbatim. The source
        # replay already contains sampled/fixed design and response values;
        # the ledger makes provenance explicit instead of back-sampling.
        ledger_rows.append({"scenario_id": summary["scenario_id"],
                            "worker_seed": summary["worker_seed"],
                            "placements": placements})
    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "sampling_design": "independent",
        "sampling_design_confirmed": True,
        "trace_provenance": (
            "RNG-free extraction of actual, recorded Phase 2 BMP placements from "
            f"Phase 12/13 {backend} ensemble SHA-256 {source_sha256}; all summarized "
            "features are realized diagnostics, not independent baseline input draws."
            + (" External-backend independent-draw design affirmed by analyst." if backend == "external" else "")
        ),
        "features": features,
        "realizations": rows,
    }
    ledger = {
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "source_ensemble_sha256": source_sha256,
        "backend": backend,
        "index_manifest": manifest,
        "scope": "Recorded BMP realization inputs only: designs, signed efficiencies, failures, AFO gains and costs; no baseline scientific input draws",
        "realizations": ledger_rows,
    }
    return trace, ledger


def write_realization_trace(ensemble_file: str | Path, trace_file: str | Path, *,
                            ledger_file: str | Path | None = None,
                            confirm_external_independence: bool = False) -> Path:
    """Export after a completed run, leaving the ensemble file byte-unchanged.

    Validation precedes any writes. The trace and optional ledger each have
    an atomic replacement, but publication of both is not a transaction.
    """
    source = Path(ensemble_file).expanduser().resolve(strict=True)
    trace_dest = Path(trace_file).expanduser().resolve()
    ledger_dest = Path(ledger_file).expanduser().resolve() if ledger_file is not None else None
    destinations = [trace_dest] + ([ledger_dest] if ledger_dest is not None else [])
    if len(set(destinations + [source])) != len(destinations) + 1:
        raise ValueError("Trace and ledger must not overwrite their source or one another")
    blob = source.read_bytes()
    try:
        data = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid ensemble JSON for realization trace") from exc
    trace, ledger = extract_realization_trace(data, source_sha256=sha256(blob).hexdigest(),
                                               confirm_external_independence=confirm_external_independence)
    # Serialize and check both artifacts before publishing either one.
    json.dumps(trace, allow_nan=False)
    json.dumps(ledger, allow_nan=False)
    _atomic_json(trace_dest, trace)
    if ledger_dest is not None:
        _atomic_json(ledger_dest, ledger)
    return trace_dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 17-ready BMP realization traces from completed BMI ensembles")
    parser.add_argument("ensemble", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--confirm-external-independent-draws", action="store_true")
    args = parser.parse_args()
    write_realization_trace(args.ensemble, args.trace, ledger_file=args.ledger,
                            confirm_external_independence=args.confirm_external_independent_draws)


if __name__ == "__main__":
    main()
