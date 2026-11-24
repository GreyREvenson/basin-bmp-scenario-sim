"""Opt-in real scientific two-process acceptance test.

Use a small, scientifically representative *file-backed* model configuration.
Set BASIN_BMP_BMI_SPAWN_SELECTION to a v1/v2/v3 backend-selection JSON path
and install the optional bmipy package. This is deliberately outside the
ordinary tests/ unit suite because large watersheds can be expensive.
"""
from __future__ import annotations

import os

import pytest

from basin_bmp_sim.certification.portability import certify_portability


@pytest.mark.skipif(
    not os.environ.get("BASIN_BMP_BMI_SPAWN_SELECTION"),
    reason="BASIN_BMP_BMI_SPAWN_SELECTION is required for process certification",
)
def test_real_scientific_parallel_ensemble_and_resume():
    report = certify_portability(
        os.environ["BASIN_BMP_BMI_SPAWN_SELECTION"],
        n_scenarios=2, n_jobs=2, require_bmipy=True,
    )
    assert report["serial_parallel_byte_identical"]
    assert report["resume_byte_identical"]
    assert report["official_bmipy"] != "not_checked"
