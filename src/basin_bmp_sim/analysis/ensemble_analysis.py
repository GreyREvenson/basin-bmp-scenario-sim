"""Phase 14: opt-in, model-neutral decision support for completed BMI ensembles.

The input is an *existing* Phase 12/13 result-schema-v2 ensemble JSON, not
native Parquet output. This module never initializes a model or draws RNG.
Targets refer to signed **annual outlet mass removed** (kg), not outlet
concentration or compliance with an absolute ambient water-quality standard.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from basin_bmp_sim.execution.ensemble_runner import (
    ENSEMBLE_RESULT_SCHEMA_VERSION, ENSEMBLE_SEED_POLICY,
    _summary_stats, ensemble_worker_seeds,
)

ANALYSIS_CONFIG_SCHEMA_VERSION = 1
ANALYSIS_RESULT_SCHEMA_VERSION = 1
_WILSON_Z_95 = 1.959963984540054


def _finite_number(value: Any, *, field: str, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite JSON number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a finite JSON number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite JSON number")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return number


@dataclass(frozen=True)
class ReductionTarget:
    """A target for a single named pollutant and outlet, in annual kg removed."""

    name: str
    pollutant: str
    outlet_oid: str
    minimum_removed_mass_kg: float


@dataclass(frozen=True)
class DecisionSupportConfig:
    targets: tuple[ReductionTarget, ...]
    budget_usd: float | None = None

    @classmethod
    def read(cls, filename: str | Path) -> "DecisionSupportConfig":
        path = Path(filename).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Decision-support configuration must be a JSON file")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid decision-support JSON") from exc
        if (not isinstance(raw, dict) or set(raw) - {"schema_version", "targets", "budget_usd"}
                or raw.get("schema_version") != ANALYSIS_CONFIG_SCHEMA_VERSION
                or type(raw.get("schema_version")) is not int):
            raise ValueError("Decision-support configuration has unknown fields or schema version")
        incoming = raw.get("targets")
        if not isinstance(incoming, list) or not incoming:
            raise ValueError("Decision-support targets must be a nonempty list")
        targets = []
        names = set()
        for index, item in enumerate(incoming):
            if (not isinstance(item, dict) or
                    set(item) != {"name", "pollutant", "outlet_oid", "minimum_removed_mass_kg"}):
                raise ValueError(f"Target {index} must contain exactly the defined target fields")
            for key in ("name", "pollutant", "outlet_oid"):
                if not isinstance(item[key], str) or not item[key].strip():
                    raise ValueError(f"Target {index} {key} must be a nonempty string")
            if item["name"] in names:
                raise ValueError(f"Duplicate decision-support target name: {item['name']}")
            names.add(item["name"])
            targets.append(ReductionTarget(
                name=item["name"], pollutant=item["pollutant"],
                outlet_oid=item["outlet_oid"],
                minimum_removed_mass_kg=_finite_number(
                    item["minimum_removed_mass_kg"],
                    field=f"Target {index} minimum_removed_mass_kg", minimum=0.0),
            ))
        budget = (None if "budget_usd" not in raw else
                  _finite_number(raw["budget_usd"], field="budget_usd", minimum=0.0))
        return cls(tuple(targets), budget)


def _event(event: np.ndarray) -> dict[str, Any]:
    """Empirical probability and a Wilson interval for Monte Carlo sampling error.

    This interval assumes independent Monte Carlo realizations; it does not
    characterize structural or observation uncertainty in a watershed model.
    """
    n = len(event)
    count = int(np.count_nonzero(event))
    p = count / n
    z2 = _WILSON_Z_95**2
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    radius = (_WILSON_Z_95 / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n**2))
    return {
        "count": count,
        "n_scenarios": n,
        "probability": p,
        "wilson_95_mc_interval": [max(0.0, center - radius), min(1.0, center + radius)],
    }


def _validated_rows(ensemble: Mapping[str, Any]):
    """Read ordered scenario summaries with manifest and seed consistency checks."""
    if (not isinstance(ensemble, dict) or
            ensemble.get("result_schema_version") != ENSEMBLE_RESULT_SCHEMA_VERSION or
            type(ensemble.get("result_schema_version")) is not int or
            ensemble.get("run_type") != "ensemble" or
            ensemble.get("seed_policy") != ENSEMBLE_SEED_POLICY or
            ensemble.get("backend") not in ("native", "external")):
        raise ValueError("Decision support requires a Phase 12/13 ensemble result")
    count = ensemble.get("n_scenarios")
    first = ensemble.get("first_scenario_id")
    base = ensemble.get("base_seed")
    if (type(count) is not int or count < 1 or type(first) is not int or first < 0 or
            type(base) is not int):
        raise ValueError("Invalid ensemble scenario count or seed/ID metadata")
    manifest = ensemble.get("index_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Missing ensemble index manifest")
    pollutants = manifest.get("pollutants")
    outlets = manifest.get("outlet_oids")
    if any(not isinstance(v, list) or not v or
           any(not isinstance(item, str) or not item for item in v) or
           len(set(v)) != len(v) for v in (pollutants, outlets)):
        raise ValueError("Invalid or duplicated outlet/pollutant identities")
    rows = ensemble.get("scenario_summaries")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Incomplete ensemble scenario summaries")
    seeds = ensemble_worker_seeds(base, count)
    costs = np.empty(count, dtype=np.float64)
    removals = np.empty((count, len(pollutants) * len(outlets)), dtype=np.float64)
    for i, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("scenario_id")) is not int or
                type(row.get("worker_seed")) is not int or
                row["scenario_id"] != first + i or row["worker_seed"] != seeds[i]):
            raise ValueError(f"Ensemble scenario identity or seed mismatch at index {i}")
        costs[i] = _finite_number(row.get("total_cost_usd"), field="total_cost_usd",
                                  minimum=0.0)
        raw = row.get("outlet_removed_mass_kg")
        if not isinstance(raw, list) or len(raw) != removals.shape[1]:
            raise ValueError(f"Invalid outlet removal shape for scenario {i}")
        removals[i] = [_finite_number(v, field="outlet_removed_mass_kg") for v in raw]
    # The published ensemble contains the original Phase 11 records. Check
    # that the compact summaries have not been detached from those records.
    scenarios = ensemble.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != count:
        raise ValueError("Incomplete ensemble original scenarios")
    for i, scenario in enumerate(scenarios):
        if (not isinstance(scenario, dict) or
                scenario.get("scenario_id") != rows[i]["scenario_id"] or
                scenario.get("worker_seed") != rows[i]["worker_seed"] or
                scenario.get("index_manifest") != manifest or
                scenario.get("backend") != ensemble["backend"] or
                scenario.get("outlet_removed_mass_kg") != rows[i]["outlet_removed_mass_kg"]):
            raise ValueError(f"Ensemble original scenario/summary mismatch at index {i}")
        responses = scenario.get("bmp_responses")
        if (not isinstance(responses, list) or
                any(not isinstance(item, dict) for item in responses)):
            raise ValueError(f"Invalid BMP responses for scenario {i}")
        final_cost = responses[-1].get("cumulative_cost_usd") if responses else 0.0
        if (type(final_cost) not in (int, float) or not math.isfinite(final_cost) or
                float(final_cost) != costs[i]):
            raise ValueError(f"Ensemble original scenario/cost mismatch at index {i}")
    return manifest, costs, removals


def analyze_ensemble(ensemble: Mapping[str, Any], config: DecisionSupportConfig,
                     *, source_sha256: str) -> dict[str, Any]:
    """Summarize annual mass-reduction targets without modifying ensemble data.

    The ratio USD / (kg removed per modeled annual realization) is a *proxy*,
    not lifecycle or present-value cost-effectiveness. Nonpositive signed
    removals have undefined ratios; they are reported, never dropped from
    attainment probabilities.
    """
    if not isinstance(config, DecisionSupportConfig) or not config.targets:
        raise ValueError("Expected nonempty DecisionSupportConfig")
    manifest, costs, removals = _validated_rows(ensemble)
    pollutants, outlets = manifest["pollutants"], manifest["outlet_oids"]
    count = len(costs)
    budget_event = None if config.budget_usd is None else costs <= config.budget_usd
    all_met = np.ones(count, dtype=np.bool_)
    target_rows = []
    for target in config.targets:
        if target.pollutant not in pollutants or target.outlet_oid not in outlets:
            raise ValueError(f"Unknown pollutant or outlet identity for target {target.name!r}")
        column = pollutants.index(target.pollutant) * len(outlets) + outlets.index(target.outlet_oid)
        amounts = removals[:, column]
        met = amounts >= target.minimum_removed_mass_kg
        all_met &= met
        positive = amounts > 0.0
        ratios = np.divide(costs[positive], amounts[positive])
        item = {
            "name": target.name,
            "pollutant": target.pollutant,
            "outlet_oid": target.outlet_oid,
            "minimum_removed_mass_kg": target.minimum_removed_mass_kg,
            "attainment": _event(met),
            "signed_removed_mass_kg": _summary_stats(amounts),
            "cost_effectiveness": {
                "units": "USD / (kg removed in modeled annual realization)",
                "positive_reduction_scenarios": int(np.count_nonzero(positive)),
                "undefined_zero_or_negative_reduction_scenarios": int(np.count_nonzero(~positive)),
                "usd_per_kg_annual_removed": _summary_stats(ratios) if ratios.size else None,
            },
        }
        if budget_event is not None:
            item["attainment_within_budget"] = _event(met & budget_event)
            within = int(np.count_nonzero(budget_event))
            item["conditional_attainment_given_within_budget"] = (
                int(np.count_nonzero(met & budget_event)) / within if within else None)
        target_rows.append(item)
    joint = _event(all_met)
    if budget_event is not None:
        joint = {**joint, "all_targets_within_budget": _event(all_met & budget_event)}
    return {
        "analysis_schema_version": ANALYSIS_RESULT_SCHEMA_VERSION,
        "source_ensemble_sha256": source_sha256,
        "ensemble_result_schema_version": ENSEMBLE_RESULT_SCHEMA_VERSION,
        "backend": ensemble["backend"],
        "base_seed": ensemble["base_seed"],
        "first_scenario_id": ensemble["first_scenario_id"],
        "n_scenarios": count,
        "index_manifest": manifest,
        "implementation_cost_usd": _summary_stats(costs),
        "budget_usd": config.budget_usd,
        "within_budget": _event(budget_event) if budget_event is not None else None,
        "targets": target_rows,
        "joint_all_targets": joint,
        "interpretation": {
            "attainment": "Empirical fraction of independent annual Monte Carlo scenarios whose signed outlet removal is at least the specified kg target, inclusive of equality.",
            "uncertainty": "Wilson 95% intervals represent finite-sample Monte Carlo precision under independent scenarios, not scientific-model uncertainty or regulatory confidence.",
            "cost": "Implementation USD divided by modeled annual kg removed; not a discounted/lifetime cost effectiveness estimate.",
            "limitations": "No outlet baseline load or concentration is supplied; absolute concentration/standard attainment cannot be inferred. Different scientific backends are not assumed equivalent.",
        },
    }


def write_ensemble_analysis(ensemble_file: str | Path, config_file: str | Path,
                            destination: str | Path) -> Path:
    """Analyze a completed ensemble and atomically publish a separate JSON.

    The original ensemble and checkpoint files are read-only. Validation is
    completed before touching the destination; on failure an earlier report
    remains intact. The Phase 12/13 JSON is parsed in memory.
    """
    source = Path(ensemble_file).expanduser().resolve(strict=True)
    dest = Path(destination).expanduser().resolve()
    if source == dest:
        raise ValueError("Analysis destination must differ from ensemble source")
    if Path(config_file).expanduser().resolve() == dest:
        raise ValueError("Analysis destination must differ from analysis configuration")
    config = DecisionSupportConfig.read(config_file)
    content = source.read_bytes()
    try:
        result = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid ensemble result JSON") from exc
    report = analyze_ensemble(result, config, source_sha256=sha256(content).hexdigest())
    from basin_bmp_sim.execution.ensemble_checkpoints import _atomic_json
    _atomic_json(dest, report)
    return dest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Analyze a completed Phase 12/13 BMI ensemble")
    parser.add_argument("ensemble_result", type=Path)
    parser.add_argument("analysis_config", type=Path)
    parser.add_argument("analysis_result", type=Path)
    args = parser.parse_args()
    print(write_ensemble_analysis(args.ensemble_result, args.analysis_config, args.analysis_result))


if __name__ == "__main__":
    main()
