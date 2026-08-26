from __future__ import annotations

import pandas as pd
import pytest

import src.load_generation as load_generation
from src.load_generation import (
    calculate_load_diagnostics,
    plet_annual_surface_runoff_in,
    validate_plet_input_table,
)


def _plet_parameters() -> dict[str, float]:
    return {
        "annual_precip_in": 42.0,
        "rain_days": 120.0,
        "rain_correction_fraction": 0.9,
        "runoff_day_fraction": 0.35,
        "cn": 78.0,
        "infiltration_fraction": 0.30,
        "ia_ratio": 0.0,
    }


def test_removed_irrigation_helper_is_not_public() -> None:
    assert not hasattr(load_generation, "plet_annual_irrigation_runoff_in")


def test_surface_runoff_total_is_precipitation_runoff_only() -> None:
    parameters = _plet_parameters()
    _, _, annual_storm_runoff, annual_total_runoff = plet_annual_surface_runoff_in(
        parameters
    )
    assert annual_total_runoff == pytest.approx(annual_storm_runoff)


def test_surface_runoff_ignores_legacy_irrigation_like_mapping_keys() -> None:
    parameters = _plet_parameters()
    baseline = plet_annual_surface_runoff_in(parameters)
    with_legacy_keys = plet_annual_surface_runoff_in(
        parameters
        | {
            "irrigated_fraction": 1.0,
            "irrigation_depth_in": 20.0,
            "irrigation_frequency": 100.0,
        }
    )
    assert with_legacy_keys == pytest.approx(baseline)


def test_load_diagnostics_have_no_irrigation_component() -> None:
    diagnostics = calculate_load_diagnostics(_plet_parameters())
    assert "annual_irrigation_runoff_in" not in diagnostics


@pytest.mark.parametrize(
    "parameter",
    [
        "irrigated_fraction",
        "irrigated_area_fraction",
        "irrigation_area_fraction",
        "irrigation_depth_in",
        "irrigation_depth",
        "irrigation_inches",
        "irrigation_frequency",
        "irrigation_frequency_per_year",
    ],
)
def test_plet_input_table_rejects_removed_irrigation_parameters(parameter: str) -> None:
    table = pd.DataFrame(
        {
            "pid": ["*", "*", "*"],
            "parameter": ["land_cover", "hsg", parameter],
            "value": ["cropland", "B", 1.0],
        }
    )
    with pytest.raises(ValueError, match="no longer supported"):
        validate_plet_input_table(table, ["p1"])
