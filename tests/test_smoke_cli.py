from __future__ import annotations

import numpy as np
import pandas as pd

import run_model
from test_reproducibility_helpers import example_config_path


def test_example_config_runs_end_to_end_from_unrelated_cwd(tmp_path, monkeypatch) -> None:
    example_cfg = example_config_path("statistical")
    outputs = tmp_path / "outputs"

    # Config-relative paths must not depend on the process working directory.
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    run_model.run_from_config(
        example_cfg,
        outputs=outputs,
        n_scenarios=1,
        bmp_limit_n=5,
        n_jobs=1,
        verbose=False,
        make_plots=False,
    )

    assert (outputs / "log.txt").exists()
    assert (outputs / "logs" / "s1.txt").exists()
    assert (outputs / "bmps" / "s1.parquet").exists()
    assert (outputs / "parcels" / "s1.parquet").exists()
    assert (outputs / "scenario_metrics" / "s1.parquet").exists()
    assert (outputs / "outlet_trajectories" / "all_scenarios.parquet").exists()


def test_plet_groundwater_is_rain_corrected_and_unchanged_by_bmps(tmp_path) -> None:
    example_cfg = example_config_path("plet_rusle")
    outputs = tmp_path / "outputs"

    run_model.run_from_config(
        example_cfg,
        outputs=outputs,
        n_scenarios=1,
        bmp_limit_n=5,
        n_jobs=1,
        verbose=True,
        make_plots=False,
    )

    load_parameters = pd.read_parquet(outputs / "load_parameters" / "s1.parquet")
    parcels = pd.read_parquet(outputs / "parcels" / "s1.parquet")
    merged = load_parameters.merge(parcels, on=["scenario", "pid"], validate="one_to_one")

    # Cropland on HSG B resolves to PLET's 0.300 infiltration fraction.
    assert np.allclose(
        load_parameters["initial_annual_infiltration_in"],
        42.0 * 0.90 * 0.300,
    )
    assert np.allclose(load_parameters["initial_cn"], 78.0)
    assert set(load_parameters["initial_land_cover"]) == {"cropland"}
    assert set(load_parameters["initial_hsg"]) == {"B"}

    # In PLET/RUSLE mode, infiltration-derived nutrient load is the canonical
    # subsurface pathway. The example's subsurface BMP efficiencies are zero,
    # so that pathway must remain unchanged by BMP application.
    for pollutant in ("tn", "tp"):
        initial_subsurface_load_rate = load_parameters[
            f"initial_subsurface_{pollutant}_load_rate_kg_ha_yr"
        ]
        final_subsurface_load_rate = load_parameters[
            f"final_subsurface_{pollutant}_load_rate_kg_ha_yr"
        ]
        assert np.allclose(initial_subsurface_load_rate, final_subsurface_load_rate)

        initial_components = (
            merged[f"initial_surface_{pollutant}_load_rate_kg_ha_yr"]
            + merged[f"initial_subsurface_{pollutant}_load_rate_kg_ha_yr"]
        )
        final_components = (
            merged[f"final_surface_{pollutant}_load_rate_kg_ha_yr"]
            + merged[f"final_subsurface_{pollutant}_load_rate_kg_ha_yr"]
        )
        pollutant_upper = pollutant.upper()
        assert np.allclose(
            merged[f"baseline_load_rate_{pollutant_upper}_kg_ha_yr"],
            initial_components,
        )
        assert np.allclose(
            merged[f"final_load_rate_{pollutant_upper}_kg_ha_yr"],
            final_components,
        )
