from __future__ import annotations

import logging
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from src.bmp import (
    _apply_pathway_reduction,
    _get_current_total_yield,
    _get_pathway_yields,
    _simulate_infield,
)
from src.constants import OUTPUT_REMOVED, OUTPUT_TREATED
from src.load_generation import (
    INCH_OVER_HA_TO_LITERS,
    calculate_load_components,
    calculate_parcel_yields,
    plet_annual_infiltration_in,
)


def _plet_parameters() -> dict[str, float]:
    return {
        "annual_precip_in": 10.0,
        "rain_days": 100.0,
        "rain_correction_fraction": 0.5,
        "runoff_day_fraction": 0.5,
        "cn": 70.0,
        "ia_ratio": 0.2,
        "infiltration_fraction": 0.2,
    }


def test_plet_infiltration_uses_rain_corrected_precipitation() -> None:
    parameters = _plet_parameters() | {"groundwater_multiplier": 1.25}

    assert plet_annual_infiltration_in(parameters) == pytest.approx(
        10.0 * 0.5 * 0.2 * 1.25
    )


@pytest.mark.parametrize("pathway_mode", ["derive_from_plet", "fixed_fractions"])
def test_untreated_groundwater_is_separate_and_mass_balanced(pathway_mode: str) -> None:
    parameters = _plet_parameters()
    expected_groundwater = (
        4.0
        * plet_annual_infiltration_in(parameters)
        * INCH_OVER_HA_TO_LITERS
        / 1_000_000.0
    )

    pathways, untreated_groundwater = calculate_load_components(
        parameters,
        {"TN": 0.0},
        {"TN": 4.0},
        ["TN"],
        pathway_mode=pathway_mode,
        surface_fraction=0.3,
        shallow_fraction=0.4,
        groundwater_loads=True,
        treat_groundwater_with_bmps=False,
    )
    totals = calculate_parcel_yields(
        parameters,
        {"TN": 0.0},
        ["TN"],
        groundwater_concentrations={"TN": 4.0},
        pathway_mode=pathway_mode,
        surface_fraction=0.3,
        shallow_fraction=0.4,
        groundwater_loads=True,
        treat_groundwater_with_bmps=False,
    )

    assert pathways[0] == pytest.approx([0.0, 0.0, 0.0])
    assert untreated_groundwater[0] == pytest.approx(expected_groundwater)
    assert totals[0] == pytest.approx(pathways[0].sum() + untreated_groundwater[0])


def test_treatable_groundwater_uses_configured_shallow_deep_split() -> None:
    parameters = _plet_parameters() | {"fraction_subsurface_shallow": 0.25}
    expected_groundwater = (
        4.0
        * plet_annual_infiltration_in(parameters)
        * INCH_OVER_HA_TO_LITERS
        / 1_000_000.0
    )

    pathways, untreated_groundwater = calculate_load_components(
        parameters,
        {"TN": 0.0},
        {"TN": 4.0},
        ["TN"],
        pathway_mode="derive_from_plet",
        groundwater_loads=True,
        treat_groundwater_with_bmps=True,
    )

    assert pathways[0] == pytest.approx(
        [0.0, expected_groundwater * 0.25, expected_groundwater * 0.75]
    )
    assert untreated_groundwater[0] == pytest.approx(0.0)


def test_infield_bmp_does_not_treat_or_reduce_protected_groundwater() -> None:
    model = SimpleNamespace(
        logger=logging.getLogger("test-groundwater"),
        pollutants=["TN"],
        parcel_area_ha=np.array([2.0]),
        current_pathway_yields=np.array([[[10.0, 0.0, 0.0]]]),
        current_untreated_groundwater_yields=np.array([[4.0]]),
    )
    model._get_pathway_yields = MethodType(_get_pathway_yields, model)
    model._get_current_total_yield = MethodType(_get_current_total_yield, model)
    model._apply_pathway_reduction = MethodType(_apply_pathway_reduction, model)

    yields = np.array([[14.0]])
    outputs = {
        OUTPUT_TREATED: np.zeros(1, dtype=float),
        OUTPUT_REMOVED: np.zeros(1, dtype=float),
    }
    efficiency = [{
        "surface": 0.5,
        "shallow subsurface": 0.5,
        "deep subsurface": 0.5,
    }]

    _simulate_infield(model, 0, efficiency, yields, {}, outputs)

    assert model.current_pathway_yields[0, 0] == pytest.approx([5.0, 0.0, 0.0])
    assert model.current_untreated_groundwater_yields[0, 0] == pytest.approx(4.0)
    assert yields[0, 0] == pytest.approx(9.0)
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(20.0)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(10.0)
