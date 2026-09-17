from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src import input_config
from src.input_validation import validate_config


class Logger:
    def verbose(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass


def _write_tables(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with sqlite3.connect(path) as con:
        for name, frame in tables.items():
            frame.to_sql(name, con, if_exists="replace", index=False)


def test_parameter_tables_are_assembled_into_runtime_long_form(tmp_path) -> None:
    gpkg = tmp_path / "anything.gpkg"
    _write_tables(
        gpkg,
        {
            "input_annual_precip_in": pd.DataFrame(
                [{"pid": "*", "value": 42.0, "units": "in/year"}]
            ),
            "input_land_cover": pd.DataFrame(
                [{"pid": "*", "value": "cropland", "units": "classification"}]
            ),
            "input_rusle_k": pd.DataFrame(
                [{"pid": "*", "mean": 0.32, "sd": 0.03, "min": 0.2, "max": 0.5}]
            ),
        },
    )
    out = input_config._assemble_parcel_parameter_source({"parcels": str(gpkg)}, Logger())
    assert set(out["parameter"]) == {"annual_precip_in", "land_cover", "k"}
    assert out.loc[out["parameter"] == "land_cover", "value"].iloc[0] == "cropland"


def test_hydrology_tables_are_separate_and_preserve_distributions(tmp_path) -> None:
    gpkg = tmp_path / "arbitrary.gpkg"
    _write_tables(
        gpkg,
        {
            "input_curve_number": pd.DataFrame(
                [{"land_cover": "cropland", "hsg": "B", "mean": 78.0, "sd": 2.0, "min": 70.0, "max": 85.0}]
            ),
            "input_infiltration_fraction": pd.DataFrame(
                [{"land_cover": "cropland", "hsg": "B", "value": 0.30}]
            ),
        },
    )
    out = input_config._assemble_plet_hydrology_source({"parcels": str(gpkg)}, Logger())
    assert set(out["parameter"]) == {"cn", "infiltration_fraction"}
    cn = out[out["parameter"] == "cn"].iloc[0]
    assert cn["mean"] == pytest.approx(78.0)
    assert cn["sd"] == pytest.approx(2.0)


def test_surface_and_subsurface_concentrations_are_distinct_variable_tables(tmp_path) -> None:
    gpkg = tmp_path / "arbitrary.gpkg"
    _write_tables(
        gpkg,
        {
            "input_surface_concentration": pd.DataFrame(
                [{"pid": "*", "pollutant": "TN", "value": 2.0}]
            ),
            "input_subsurface_concentration": pd.DataFrame(
                [{"pid": "*", "pollutant": "TN", "value": 5.5}]
            ),
        },
    )
    out = input_config._assemble_plet_concentration_source({"parcels": str(gpkg)}, Logger())
    got = {(r.pollutant, r.pathway): r.value for r in out.itertuples()}
    assert got[("TN", "surface")] == pytest.approx(2.0)
    assert got[("TN", "subsurface")] == pytest.approx(5.5)


def test_delivery_ratio_tables_support_global_default_and_exact_override(tmp_path) -> None:
    gpkg = tmp_path / "arbitrary.gpkg"
    _write_tables(
        gpkg,
        {
            "parcel_outlets": pd.DataFrame([{"pid": "1", "oid": "A"}, {"pid": "2", "oid": "A"}]),
            "input_sdr_f_to_s": pd.DataFrame(
                [
                    {"pid": "*", "oid": "*", "value": 0.9},
                    {"pid": "2", "oid": "A", "value": 0.5},
                ]
            ),
        },
    )
    out = input_config._load_delivery_ratios({"parcels": str(gpkg)}, Logger())
    rows = {(r.pid, r.oid): r for r in out.itertuples()}
    assert rows[("1", "A")].sdr_f_to_s == pytest.approx(0.9)
    assert rows[("2", "A")].sdr_f_to_s == pytest.approx(0.5)
    assert rows[("1", "A")].sdr_s_to_o == pytest.approx(1.0)
    assert rows[("1", "A")].ndr_f_to_s == pytest.approx(1.0)
    assert rows[("1", "A")].ndr_s_to_o == pytest.approx(1.0)


def test_old_hydrology_path_config_is_rejected() -> None:
    cfg = input_config.normalize_config(
        {"n_scenarios": 1, "load_generation": {"mode": "plet_rusle", "hydrology_lookup": "old.csv"}}
    )
    with pytest.raises(ValueError, match="hydrology_lookup"):
        validate_config(cfg)
