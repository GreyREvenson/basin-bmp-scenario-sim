"""Deterministic ensembles for selected annual BMI backends.

The driver operates exclusively on per-realization Phase 11 JSON records. A
fresh independently initialized generator/model pair is constructed for each
scenario, so AFO/cost RNG ordering is preserved *within* each realization.
No native PLET/RUSLE accounting or Parquet schema is assumed.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

import numpy as np

from basin_bmp_sim.interfaces.component_bootstrap import ComponentStartup

ENSEMBLE_RESULT_SCHEMA_VERSION = 2
ENSEMBLE_SEED_POLICY = "numpy-seedsequence-spawn-uint64-v1"


def ensemble_worker_seeds(base_seed: int, n_scenarios: int) -> tuple[int, ...]:
    """Derive reproducible independent integer seeds, not native CLI seeds.

    The native Monte Carlo driver uses uint32 child state; this opt-in BMI
    ensemble uses uint64 states. Seed equivalence to the native CLI is *not*
    asserted, even if both are given the same seed and science backend.
    """
    if type(base_seed) is not int or not 0 <= base_seed < 2**64:
        raise ValueError("Ensemble base seed must be a uint64 integer")
    if type(n_scenarios) is not int or n_scenarios < 1:
        raise ValueError("Ensemble n_scenarios must be a positive integer")
    return tuple(int(child.generate_state(1, dtype=np.uint64)[0])
                 for child in np.random.SeedSequence(base_seed).spawn(n_scenarios))


def _summary_stats(data: np.ndarray) -> dict[str, Any]:
    """Cross-scenario descriptive statistics, axis 0 is always realization."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("Ensemble statistics require nonempty finite scenario values")
    quantiles = np.percentile(values, [5, 50, 95], axis=0)

    def as_json(value):
        return np.asarray(value).tolist()

    return {
        "mean": as_json(values.mean(axis=0)),
        "std_population": as_json(values.std(axis=0, ddof=0)),
        "min": as_json(values.min(axis=0)),
        "max": as_json(values.max(axis=0)),
        "p05": as_json(quantiles[0]),
        "median": as_json(quantiles[1]),
        "p95": as_json(quantiles[2]),
    }


def summarize_ensemble(records: Sequence[Mapping[str, Any]], *,
                       backend: str, base_seed: int, first_scenario_id: int,
                       include_scenarios: bool = True) -> dict[str, Any]:
    """Validate index identity and compute signed, model-neutral summaries.

    Flattened arrays retain Phase 11 row-major order. Statistics are across
    independent annual stochastic realizations, *not* temporal variation.
    No negative load reduction is clipped. Percentiles use NumPy linear
    interpolation; standard deviations use the population denominator N.
    """
    if not records:
        raise ValueError("Ensemble must contain at least one scenario")
    domain = records[0]["index_manifest"]
    if not isinstance(domain, dict):
        raise ValueError("Ensemble index manifest must be an object")
    try:
        n_parcels = len(domain["parcel_ids"])
        n_pollutants = len(domain["pollutants"])
        n_outlets = len(domain["outlet_oids"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Ensemble index manifest is missing required identities") from exc
    if min(n_parcels, n_pollutants, n_outlets) < 1:
        raise ValueError("Ensemble domain must contain parcels, pollutants and outlets")

    baseline_rows, current_rows, outlet_rows = [], [], []
    counts, costs, rows = [], [], []
    identities: set[int] = set()
    expected_seeds = ensemble_worker_seeds(base_seed, len(records))
    for index, record in enumerate(records):
        if record.get("index_manifest") != domain:
            raise ValueError("Ensemble scenarios have incompatible index manifests")
        if record.get("backend") != backend or record.get("result_schema_version") != 1:
            raise ValueError("Ensemble scenarios have incompatible backend or result schemas")
        scenario_id = record.get("scenario_id")
        seed = record.get("worker_seed")
        if (type(scenario_id) is not int or scenario_id in identities or
                type(seed) is not int or scenario_id != first_scenario_id + index or
                seed != expected_seeds[index]):
            raise ValueError("Ensemble scenario identities or seeds disagree with the schedule")
        identities.add(scenario_id)
        replay = record.get("replay_scenario")
        if (not isinstance(replay, dict) or replay.get("scenario_id") != scenario_id or
                replay.get("seed") != seed):
            raise ValueError("Ensemble replay metadata disagrees with scenario identity")
        responses = record.get("bmp_responses")
        if not isinstance(responses, list):
            raise ValueError("Ensemble scenario BMP responses must be a list")
        n_bmps = len(responses)
        if not isinstance(replay.get("placements"), list) or len(replay["placements"]) != n_bmps:
            raise ValueError("Ensemble BMP response count disagrees with replay placements")
        total_cost = float(responses[-1]["cumulative_cost_usd"]) if responses else 0.0
        if not np.isfinite(total_cost) or total_cost < 0:
            raise ValueError("Ensemble cost must be finite and nonnegative")
        expected = (("baseline_load_rate_kg_ha_yr", baseline_rows, n_parcels * n_pollutants),
                    ("current_load_rate_kg_ha_yr", current_rows, n_parcels * n_pollutants),
                    ("outlet_removed_mass_kg", outlet_rows, n_pollutants * n_outlets))
        for name, target, size in expected:
            values = np.asarray(record.get(name), dtype=np.float64)
            if values.ndim != 1 or values.size != size or not np.isfinite(values).all():
                raise ValueError(f"Ensemble {name} has incompatible shape or nonfinite values")
            target.append(values)
        counts.append(n_bmps)
        costs.append(total_cost)
        rows.append({
            "scenario_id": scenario_id,
            "worker_seed": seed,
            "bmp_count": n_bmps,
            "total_cost_usd": total_cost,
            "outlet_removed_mass_kg": outlet_rows[-1].tolist(),
        })

    summary = {
        "result_schema_version": ENSEMBLE_RESULT_SCHEMA_VERSION,
        "run_type": "ensemble",
        "backend": backend,
        "seed_policy": ENSEMBLE_SEED_POLICY,
        "base_seed": base_seed,
        "first_scenario_id": first_scenario_id,
        "n_scenarios": len(records),
        "index_manifest": domain,
        "statistics": {
            "bmp_count": _summary_stats(np.asarray(counts)),
            "total_cost_usd": _summary_stats(np.asarray(costs)),
            "outlet_removed_mass_kg": {
                "shape": [n_pollutants, n_outlets],
                **_summary_stats(np.stack(outlet_rows)),
            },
            "baseline_load_rate_kg_ha_yr": {
                "shape": [n_parcels, n_pollutants],
                **_summary_stats(np.stack(baseline_rows)),
            },
            "current_load_rate_kg_ha_yr": {
                "shape": [n_parcels, n_pollutants],
                **_summary_stats(np.stack(current_rows)),
            },
        },
        "scenario_summaries": rows,
    }
    if include_scenarios:
        summary["scenarios"] = list(records)
    return summary


def run_backend_ensemble(selection, result_file: str | Path) -> Path:
    """Execute n independently initialized BMI scenarios, committing once.

    The original startup manifest and model YAML are never rewritten. One
    temporary per-worker manifest supplies the new seed/scenario ID to both
    components. No output is published until *all* scenarios and summaries
    pass validation. The resulting JSON embeds each Phase 11 replay scenario.
    """
    from basin_bmp_sim.execution.backend_selection import run_selected_scenario, write_selected_result

    if selection.ensemble is None:
        raise ValueError("Ensemble runner requires an ensemble selection")
    if selection.ensemble.checkpoint_dir is not None:
        from basin_bmp_sim.execution.ensemble_checkpoints import run_checkpointed_ensemble
        return run_checkpointed_ensemble(selection, result_file)
    root = ComponentStartup.read(selection.component_manifest)
    if root.replay_file is not None:
        raise ValueError("Ensemble startup cannot set replay_file")
    n_scenarios = selection.ensemble.n_scenarios
    seeds = ensemble_worker_seeds(root.worker_seed, n_scenarios)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Ensemble worker seed collision; choose a different base seed")
    records = []
    with TemporaryDirectory(prefix="basin-bmp-ensemble-") as temp_dir:
        for index, seed in enumerate(seeds):
            scenario_id = root.scenario_id + index
            manifest = Path(temp_dir) / f"scenario_{scenario_id}.json"
            manifest.write_text(json.dumps({
                "schema_version": 1,
                "model_config": str(root.model_config),
                "worker_seed": seed,
                "scenario_id": scenario_id,
            }), encoding="utf-8")
            # Rebuild the selected factory, backend, RNG and generator per
            # realization. The coupler internally preserves AFO->cost draws.
            record = run_selected_scenario(replace(selection, component_manifest=manifest))
            records.append(record)
    result = summarize_ensemble(records, backend=selection.backend,
                                base_seed=root.worker_seed,
                                first_scenario_id=root.scenario_id)
    return write_selected_result(result_file, result)


def export_ensemble_replays(result_file: str | Path, output_dir: str | Path) -> tuple[Path, ...]:
    """Export independently loadable Phase 2 replay artifacts from an ensemble.

    Validate *every* scenario and destination before writing any output.
    Existing files are never silently overwritten. Replay is meaningful only
    against the same generator/domain, configuration and scientific backend.
    """
    from basin_bmp_sim.scenarios.generator import BMPScenario

    ensemble = json.loads(Path(result_file).read_text(encoding="utf-8"))
    if (not isinstance(ensemble, dict) or
            ensemble.get("result_schema_version") != ENSEMBLE_RESULT_SCHEMA_VERSION or
            ensemble.get("run_type") != "ensemble"):
        raise ValueError("Expected a Phase 12 BMI ensemble result")
    items = ensemble.get("scenarios")
    if not isinstance(items, list) or len(items) != ensemble.get("n_scenarios"):
        raise ValueError("Ensemble replay list has an invalid length")
    scenarios = []
    seen: set[int] = set()
    for row in items:
        if not isinstance(row, dict):
            raise ValueError("Ensemble scenario result must be an object")
        scenario = BMPScenario.from_dict(row["replay_scenario"])
        if (scenario.scenario_id != row.get("scenario_id") or
                scenario.seed != row.get("worker_seed") or
                scenario.scenario_id in seen):
            raise ValueError("Ensemble replay identity mismatch or duplicate")
        seen.add(scenario.scenario_id)
        scenarios.append(scenario)
    root = Path(output_dir).expanduser().resolve()
    paths = tuple(root / f"s{scenario.scenario_id}.json" for scenario in scenarios)
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite an existing BMI replay file")
    root.mkdir(parents=True, exist_ok=True)
    for scenario, path in zip(scenarios, paths):
        scenario.save(path)
    return paths


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Export replay files from a Phase 12 BMI ensemble JSON")
    parser.add_argument("ensemble_result", type=Path)
    parser.add_argument("replay_dir", type=Path)
    args = parser.parse_args()
    for path in export_ensemble_replays(args.ensemble_result, args.replay_dir):
        print(path)


if __name__ == "__main__":
    main()
