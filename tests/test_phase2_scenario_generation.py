"""Phase 2 golden outputs and exact replay of completed BMP realizations.

The extended golden was captured from the untouched dev_arch_refac (1).zip,
not from the phase 2 implementation. It verifies cost-constrained selection,
multi-practice responses, statistical mode and mass-based AFO recovery.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.model as model
from src.scenarios.stochastic_scenario_generator import BMPScenario
from test_architecture_result_equivalence import _compare_values
from test_plet_end_to_end_regression import _shared_payload, _single_mass_afo_payload, _config

GOLDEN = json.loads(
    (Path(__file__).parent / "golden/pre_phase2_seeded_extended_results.json").read_text()
)


def _normalize(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v)


def _case(case: str, output: Path):
    cfg = _config(output)
    if case == "mixed_budget":
        base = _shared_payload()
        base["bmp_cps"] = [340, 412, 656]
        base["bmp_selection_probs"] = np.array([1 / 3] * 3)
        for cps in (412, 656):
            base["bmp_efficiency_stats"][cps] = base["bmp_efficiency_stats"][340]
        base["data"]["bmp_cost"] = pd.DataFrame([
            {"cps": 340, "cost_component": "install", "unit": "usd/project", "mean": 60., "min": 40., "max": 80.},
            {"cps": 340, "cost_component": "maintenance", "unit": "usd/project", "value": 10.},
            {"cps": 412, "cost_component": "install", "unit": "usd/project", "value": 70.},
            {"cps": 656, "cost_component": "install", "unit": "usd/project", "value": 80.},
        ])
        cfg.update(bmp_limit_n=10, bmp_limit_usd=175.0, bmp_fail_rate=0.4)
        seed = 12345
    elif case == "afo_recovery":
        base = _single_mass_afo_payload(statistical=False)
        base["afo_inputs"]["management"][0]["collection_fraction"] = 0.5
        base["afo_inputs"]["feedlot_manure_effects"] = [
            {"afo_id": "demo", "cps": 340, "additional_collection_fraction": 0.5}
        ]
        seed = 1234
    elif case == "statistical_multi":
        base = _shared_payload()
        base.update(
            load_generation_mode="statistical",
            pollutant_load_rate_is_aggregate=True,
            pollutant_load_rate_pathway_fractions={"surface": 0.6, "subsurface": 0.4},
            pollutant_load_rate_stats=[[{"value": 12.5}, {"value": 3.8}, {"value": 60.0}]],
        )
        base["bmp_cps"] = [340, 412, 656]
        base["bmp_selection_probs"] = np.array([1 / 3] * 3)
        for cps in (412, 656):
            base["bmp_efficiency_stats"][cps] = base["bmp_efficiency_stats"][340]
        base["data"]["bmp_cost"] = pd.DataFrame([
            {"cps": 340, "unit": "usd/project", "value": 20.0},
            {"cps": 412, "unit": "usd/project", "value": 21.0},
            {"cps": 656, "unit": "usd/project", "value": 22.0},
        ])
        cfg["bmp_limit_n"] = 3
        seed = 456
    else:
        raise AssertionError(case)
    return base, cfg, seed


def _capture(monkeypatch, output: Path, case: str, scenario: BMPScenario | None = None):
    payload, cfg, seed = _case(case, output)
    written = {}

    def record(df, path, *, logger):
        written[f"{path.parent.name}/{path.name}"] = {
            str(col): [_normalize(v) for v in df[col].tolist()] for col in df.columns
        }

    with monkeypatch.context() as scope:
        scope.setattr(model, "_write_parquet_atomic", record)
        trajectory = model._run_one_scenario(
            payload, cfg, 0, seed, output, scenario_definition=scenario
        )
    results = {
        "tables": written,
        "trajectory": {
            "|".join(str(x) for x in key): [[_normalize(v) for v in rec] for rec in rows]
            for key, rows in trajectory.items()
        },
    }
    return results, BMPScenario.load(output / "scenario_definitions/s1.json")


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_phase2_matches_unmodified_branch_and_replays(case, tmp_path, monkeypatch):
    generated, realization = _capture(monkeypatch, tmp_path / "generated", case)
    _compare_values(GOLDEN[case], generated)
    assert len(realization.placements) == len(generated["tables"]["bmps/s1.parquet"]["cps"])
    assert realization == BMPScenario.from_dict(realization.to_dict())

    # Replay must not run any placement-selection or design sampling.
    from src.scenarios import stochastic_scenario_generator as generator_module
    with monkeypatch.context() as scope:
        scope.setattr(generator_module.StochasticScenarioGenerator, "next_placement",
                      lambda *args: (_ for _ in ()).throw(AssertionError("resampled placement")))
        scope.setattr(generator_module, "sample_bmp_design",
                      lambda *args: (_ for _ in ()).throw(AssertionError("resampled design")))
        replayed, replay_record = _capture(scope, tmp_path / "replayed", case, realization)
    _compare_values(generated, replayed)
    assert realization == replay_record


def test_replay_rejects_wrong_seed_and_domain(tmp_path, monkeypatch):
    _, realization = _capture(monkeypatch, tmp_path / "generated", "mixed_budget")
    from src.driver.monte_carlo_driver import _ScenarioContext
    import logging
    payload, cfg, seed = _case("mixed_budget", tmp_path / "another")
    ctx = _ScenarioContext(cfg, payload, logging.getLogger("replay_test"), seed)
    from src.scenarios.stochastic_scenario_generator import StochasticScenarioGenerator
    wrong_seed = replace(realization, seed=42)
    with pytest.raises(ValueError, match="ID/seed"):
        StochasticScenarioGenerator(ctx, cfg, replay=wrong_seed).validate_replay(ctx, 1, seed)
    wrong_domain = replace(realization, parcel_ids_digest="deadbeef")
    with pytest.raises(ValueError, match="parcel IDs"):
        StochasticScenarioGenerator(ctx, cfg, replay=wrong_domain).validate_replay(ctx, 1, seed)
    with pytest.raises(ValueError, match="schema version"):
        BMPScenario.from_dict({**realization.to_dict(), "schema_version": 200})
