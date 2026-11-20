"""Five pre-refactor seeded golden results: scientific behavior must not change.

The golden JSON was generated from the unmodified dev_arch_refac upload before
splitting BMP, model and driver modules. This test captures tables in memory,
so it does not require the optional Parquet engine to validate numeric output.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.model as model
from test_plet_end_to_end_regression import (
    _config, _shared_payload, _single_mass_afo_payload,
)

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "pre_refactor_seeded_results.json").read_text()
)


def _normalize(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value)


def _capture_case(monkeypatch, tmp_path: Path, case: str) -> dict:
    base = _single_mass_afo_payload(statistical=False) if case == "afo_mass_plet" else _shared_payload()
    cps = {"plet_infield": 340, "plet_wetland": 656,
           "plet_grassed": 412, "statistical_infield": 340,
           "afo_mass_plet": 340}[case]
    if case == "statistical_infield":
        base.update(
            load_generation_mode="statistical", pollutant_load_rate_is_aggregate=True,
            pollutant_load_rate_pathway_fractions={"surface": 0.6, "subsurface": 0.4},
            pollutant_load_rate_stats=[[{"value": 12.5}, {"value": 3.8}, {"value": 60.0}]],
        )
    base["bmp_cps"] = [cps]
    base["bmp_selection_probs"] = np.asarray([1.0])
    base["feedlot_bmp_cps"] = base.get("feedlot_bmp_cps", set())
    base["bmp_efficiency_stats"][cps] = base["bmp_efficiency_stats"][340]
    cfg = _config(tmp_path / case)
    if cps != 340:
        base["data"]["bmp_cost"] = None
        cfg["bmp_cost"] = None
    written = {}

    def capture(df: pd.DataFrame, path: Path, *, logger):
        written[str(path.parent.name) + "/" + path.name] = {
            str(col): [_normalize(v) for v in df[col].tolist()] for col in df.columns
        }

    monkeypatch.setattr(model, "_write_parquet_atomic", capture)
    records = model._run_one_scenario(
        base, cfg, sidx=0, seed=(123 if case == "afo_mass_plet" else 12345),
        outputs_dir=Path(cfg["outputs"]),
    )
    return {
        "tables": written,
        "trajectory": {
            "|".join(str(v) for v in key): [list(map(float, record)) for record in vals]
            for key, vals in records.items()
        },
    }


def _compare_values(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            _compare_values(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _compare_values(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), path
    else:
        assert a == b, path


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_matches_original_branch_seeded_outputs(case, monkeypatch, tmp_path):
    """All numeric tables, BMP details, and outlet trajectories match baseline."""
    actual = _capture_case(monkeypatch, tmp_path, case)
    _compare_values(GOLDEN[case], actual)
