"""Opt-in comparison to the official Python BMI abstract method catalog.

Run after installing the ``bmi`` optional extra. This check does not certify
interoperability with third-party frameworks, grids or physical time models.
"""
from __future__ import annotations

import pytest


def test_adapters_cover_bmipy_abstract_method_names():
    bmipy = pytest.importorskip("bmipy", reason="install project[bmi] for BMI comparison")
    from src.interfaces.bmi import (
        LiveScenarioGeneratorBmi, StochasticScenarioBmi,
        WatershedModelBmi, MonteCarloDriverBmi, ExternalWatershedModelBmi,
    )
    required = set(bmipy.Bmi.__abstractmethods__)
    assert required, "Official BMI abstract-method catalog was empty"
    for cls in (LiveScenarioGeneratorBmi, StochasticScenarioBmi,
                WatershedModelBmi, MonteCarloDriverBmi, ExternalWatershedModelBmi):
        missing = sorted(name for name in required if not callable(getattr(cls, name, None)))
        assert not missing, f"{cls.__name__} is missing BMI methods: {missing}"
