"""Regression checks for the canonical AFO/config module ownership."""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import pandas as pd

from basin_bmp_sim.science.plet_defaults import apply_plet_parameter_defaults
from basin_bmp_sim.config.settings import normalize_config, resolve_config_paths
from basin_bmp_sim.science.afo.excretion import reference_excretion_factors
from utils.create_plet_model_inputs import _sheet


def test_afo_submodules_have_single_canonical_owners() -> None:
    pkg = importlib.import_module("basin_bmp_sim.science.afo")
    assert pkg.__file__.endswith("__init__.py")
    # These symbols must not be duplicated at a legacy import location.
    assert not hasattr(pkg, "mass_afo_realization")
    assert not hasattr(pkg, "load_afo_inputs")
    for name in ("inputs", "shared", "allocation", "feedlot", "excretion", "manure", "recovery"):
        assert importlib.import_module(f"basin_bmp_sim.science.afo.{name}")


def test_scientific_defaults_do_not_require_file_loader() -> None:
    assert apply_plet_parameter_defaults({}, ["TN"])["load_multiplier_tn"] == 1.0
    assert apply_plet_parameter_defaults({"runoff_multiplier": 2.0})["runoff_multiplier"] == 2.0


def test_config_path_normalization_preserves_relative_file_semantics(tmp_path: Path) -> None:
    config_file = tmp_path / "config" / "run.yaml"
    config = normalize_config({"parcels": "../parcels.gpkg", "parallel": {}})
    resolved = resolve_config_paths(config, config_file)
    assert resolved["parcels"] == str((tmp_path / "parcels.gpkg").resolve())
    assert resolved["parallel"]["n_jobs"] == 1


def test_bundled_afo_reference_still_available_after_package_move() -> None:
    factors = reference_excretion_factors()
    assert factors
    assert "representative_mass_kg" in next(iter(factors.values()))


def test_plet_workbook_sheet_read_avoids_deprecated_parser(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        pd.DataFrame({" col ": [1.0]}).to_excel(writer, sheet_name="Data", index=False)
    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*ExcelFile.parse.*")
            result = _sheet(workbook, "Data", {"COL"})
    assert result["COL"].tolist() == [1.0]
