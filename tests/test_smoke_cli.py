"""Synthetic full-model smoke tests.

Despite the historical filename, these tests intentionally do not exercise the
CLI or any shipped example YAML.  CLI/config-path behavior is covered by the
run_model and config-path unit tests; these tests exercise model execution from
normalized in-memory inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from model_test_cases import make_plet_rusle_case, make_statistical_case
from src.model import Model


def _capture_model_writes(monkeypatch):
    written: dict[Path, pd.DataFrame] = {}

    def capture(df: pd.DataFrame, path: Path, *, logger) -> None:
        del logger
        written[Path(path).relative_to(Path(path).parents[1])] = df.copy()

    monkeypatch.setattr("src.model._write_parquet_atomic", capture)
    return written


def test_synthetic_statistical_case_runs_end_to_end(tmp_path, monkeypatch) -> None:
    written = _capture_model_writes(monkeypatch)
    cfg, data = make_statistical_case(
        tmp_path / "outputs",
        n_jobs=1,
        n_scenarios=1,
        random_seed=31001,
        bmp_limit_n=2,
    )

    records = Model(cfg, data, logging.getLogger("smoke.statistical")).run_all_scenarios()

    assert records
    assert Path("bmps", "s1.parquet") in written
    assert Path("parcels", "s1.parquet") in written
    assert Path("scenario_metrics", "s1.parquet") in written


def test_synthetic_plet_case_uses_supplied_hydrology_and_subsurface_efficiency(
    tmp_path, monkeypatch
) -> None:
    written = _capture_model_writes(monkeypatch)
    cfg, data = make_plet_rusle_case(
        tmp_path / "outputs",
        n_jobs=1,
        n_scenarios=1,
        random_seed=31002,
        bmp_limit_n=1,
    )

    # Force a single known CPS so pathway-efficiency expectations are deterministic.
    data["cps"] = [340]
    data["bmp_eff"] = data["bmp_eff"][data["bmp_eff"]["cps"] == 340].reset_index(drop=True)

    Model(cfg, data, logging.getLogger("smoke.plet")).run_all_scenarios()

    parameter_frames = [
        df for key, df in written.items() if key == Path("load_parameters", "s1.parquet")
    ]
    assert len(parameter_frames) == 1
    params = parameter_frames[0]

    # The synthetic case deliberately uses non-reference values.  Their presence
    # demonstrates that model behavior follows supplied hydrology input rather
    # than a source-level fallback table.
    assert params["initial_cn"].astype(float).to_numpy() == pytest.approx([73.0, 73.0, 73.0])
    assert params["initial_infiltration_fraction"].astype(float).to_numpy() == pytest.approx(
        [0.37, 0.37, 0.37]
    )

    # CPS 340 has TN subsurface efficiency fixed at zero, while TP is 0.10.
    # Therefore TN subsurface load is unchanged for every parcel; TP changes only
    # on the parcel receiving the BMP.
    assert (params["initial_subsurface_tn_load_rate_kg_ha_yr"] == params[
        "final_subsurface_tn_load_rate_kg_ha_yr"
    ]).all()
    assert (
        params["final_subsurface_tp_load_rate_kg_ha_yr"]
        <= params["initial_subsurface_tp_load_rate_kg_ha_yr"]
    ).all()
