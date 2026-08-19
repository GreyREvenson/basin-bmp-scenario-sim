from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import run_model


def test_example_config_runs_end_to_end(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_cfg = repo_root / "examples" / "east_fork" / "east_fork.yaml"
    cfg = yaml.safe_load(example_cfg.read_text(encoding="utf-8"))

    cfg["outputs"] = str(tmp_path / "outputs")
    cfg["n_scenarios"] = 1
    cfg["bmp_limit_n"] = 5
    cfg["parallel"] = {"n_jobs": 1}
    cfg["verbose"] = False

    smoke_cfg = tmp_path / "smoke.yaml"
    smoke_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["run_model.py", str(smoke_cfg)])

    run_model.main()

    outputs = tmp_path / "outputs"
    assert (outputs / "log.txt").exists()
    assert (outputs / "logs" / "s1.txt").exists()
    assert (outputs / "bmps" / "s1.parquet").exists()
    assert (outputs / "parcels" / "s1.parquet").exists()
    assert (outputs / "scenario_metrics" / "s1.parquet").exists()
    assert (outputs / "outlet_trajectories" / "all_scenarios.parquet").exists()


def test_plet_groundwater_is_rain_corrected_and_unchanged_by_bmps(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_cfg = repo_root / "examples" / "east_fork" / "east_fork_plet.yaml"
    cfg = yaml.safe_load(example_cfg.read_text(encoding="utf-8"))

    cfg["outputs"] = str(tmp_path / "outputs")
    cfg["n_scenarios"] = 1
    cfg["bmp_limit_n"] = 5
    cfg["parallel"] = {"n_jobs": 1}
    cfg["verbose"] = False

    smoke_cfg = tmp_path / "plet_smoke.yaml"
    smoke_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["run_model.py", str(smoke_cfg)])

    run_model.main()

    outputs = tmp_path / "outputs"
    load_parameters = pd.read_parquet(outputs / "load_parameters" / "s1.parquet")
    parcels = pd.read_parquet(outputs / "parcels" / "s1.parquet")
    merged = load_parameters.merge(parcels, on=["scenario", "pid"], validate="one_to_one")

    assert np.allclose(load_parameters["initial_annual_infiltration_in"], 42.0 * 0.90 * 0.18)
    for pollutant in ("tn", "tp"):
        initial_groundwater = load_parameters[f"initial_untreated_groundwater_{pollutant}_kg_ha"]
        final_groundwater = load_parameters[f"final_untreated_groundwater_{pollutant}_kg_ha"]
        assert np.allclose(initial_groundwater, final_groundwater)

        initial_components = (
            merged[f"initial_surface_{pollutant}_kg_ha"]
            + merged[f"initial_shallow_{pollutant}_kg_ha"]
            + merged[f"initial_deep_{pollutant}_kg_ha"]
            + merged[f"initial_untreated_groundwater_{pollutant}_kg_ha"]
        )
        final_components = (
            merged[f"final_surface_{pollutant}_kg_ha"]
            + merged[f"final_shallow_{pollutant}_kg_ha"]
            + merged[f"final_deep_{pollutant}_kg_ha"]
            + merged[f"final_untreated_groundwater_{pollutant}_kg_ha"]
        )
        assert np.allclose(merged[f"baseline_{pollutant.upper()}"], initial_components)
        assert np.allclose(merged[f"final_{pollutant.upper()}"], final_components)
