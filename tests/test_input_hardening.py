from __future__ import annotations

import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from src import input_config
from src.input_config import (
    _ensure_projected,
    _load_cps,
    _load_plet_hydrology_lookup,
    _merge_csvs,
    _plet_parameter_defaults,
    _validate_input_package_schema,
    resolve_distribution_references,
)
from src.io_utils import MissingInputTableError, read_geopackage_table


class Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def verbose(self, *args, **kwargs):
        pass


def _write_table(path: Path, name: str, frame: pd.DataFrame) -> None:
    with sqlite3.connect(path) as con:
        frame.to_sql(name, con, if_exists="replace", index=False)


def test_required_plet_parameters_are_not_synthesized_by_defaults() -> None:
    defaults = _plet_parameter_defaults(["TN"])
    for required in (
        "annual_precip_in",
        "rain_days",
        "rain_correction_fraction",
        "runoff_day_fraction",
        "land_cover",
        "hsg",
    ):
        assert required not in defaults


def test_cps_rejects_fractional_and_boolean_values() -> None:
    with pytest.raises(ValueError, match="finite integer"):
        _load_cps({"cps": [340.5]})
    with pytest.raises(ValueError, match="boolean"):
        _load_cps({"cps": [True]})


def test_merge_csvs_rejects_conflicting_logical_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "eff.csv"
    pd.DataFrame(
        [
            {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.1},
            {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.9},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _merge_csvs(path, ["cps", "pollutant"], "bmp_efficiency", Logger())


def test_merge_csvs_collapses_exact_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "eff.csv"
    row = {"cps": 340, "pollutant": "TN", "pathway": "surface", "value": 0.1}
    pd.DataFrame([row, row]).to_csv(path, index=False)
    loaded = _merge_csvs(path, ["cps", "pollutant"], "bmp_efficiency", Logger())
    assert len(loaded) == 1


def test_unknown_input_table_is_rejected_with_suggestion(tmp_path: Path) -> None:
    package = tmp_path / "parcels.gpkg"
    _write_table(package, "input_anual_precip_in", pd.DataFrame([{"pid": "*", "value": 42.0}]))
    with pytest.raises(ValueError, match="input_anual_precip_in.*did you mean.*input_annual_precip_in"):
        _validate_input_package_schema(
            {"parcels": str(package)}, "plet_rusle", Logger()
        )


def test_missing_geopackage_table_has_specific_exception(tmp_path: Path) -> None:
    package = tmp_path / "parcels.gpkg"
    _write_table(package, "present", pd.DataFrame([{"x": 1}]))
    with pytest.raises(MissingInputTableError):
        read_geopackage_table(package, "missing")


def test_projected_feet_are_reprojected_to_metric_analysis_crs() -> None:
    geographic = gpd.GeoDataFrame(
        {"id": [1]}, geometry=[Point(-82.95, 39.98)], crs="EPSG:4326"
    )
    # Ohio South State Plane uses US survey feet.
    feet = geographic.to_crs("EPSG:3735")
    assert feet.crs.is_projected
    assert any("foot" in axis.unit_name.lower() for axis in feet.crs.axis_info)

    metric = _ensure_projected(feet, Logger())
    assert metric.crs.is_projected
    assert all(abs(float(axis.unit_conversion_factor) - 1.0) < 1e-12 for axis in metric.crs.axis_info[:2])


def test_hydrology_lookup_can_be_reusable_but_only_used_pairs_must_be_complete() -> None:
    table = pd.DataFrame(
        [
            {"land_cover": "cropland", "hsg": "B", "parameter": "cn", "value": 78.0},
            {"land_cover": "cropland", "hsg": "B", "parameter": "infiltration_fraction", "value": 0.30},
            # Extra incomplete unused pairing should not invalidate this run.
            {"land_cover": "forest", "hsg": "A", "parameter": "cn", "value": 39.0},
        ]
    )
    loaded = _load_plet_hydrology_lookup(
        table, Logger(), required_pairs=[("cropland", "B")]
    )
    assert len(loaded) == 3


def test_distribution_unit_compatibility_is_run_local_not_global() -> None:
    catalog = pd.DataFrame(
        [
            {
                "distribution_id": "rain",
                "mean": 1000.0,
                "sd": 100.0,
                "units": "mm/year",
            }
        ]
    )
    use = pd.DataFrame(
        [
            {
                "pid": "*",
                "parameter": "annual_precip_in",
                "distribution_id": "rain",
                "units": "in/year",
            }
        ]
    )
    with pytest.raises(ValueError, match="may not reinterpret"):
        resolve_distribution_references(use, catalog, "plet_inputs")


def test_clipped_structural_relationships_are_filtered_but_true_unknown_ids_error() -> None:
    rows = pd.DataFrame(
        [
            {"pid": "P2", "pid_up": "P1"},
            {"pid": "P3", "pid_up": "P2"},  # P3 exists in source but is outside domain.
        ]
    )
    result = input_config._build_parcel_up_map(
        rows,
        ["P1", "P2"],
        source_parcel_ids=["P1", "P2", "P3"],
        logger=Logger(),
    )
    assert result["P2"] == ["P1"]

    bad = pd.DataFrame([{"pid": "P2", "pid_up": "TYPO"}])
    with pytest.raises(ValueError, match="source parcels layer"):
        input_config._build_parcel_up_map(
            bad,
            ["P1", "P2"],
            source_parcel_ids=["P1", "P2", "P3"],
            logger=Logger(),
        )


def test_validate_config_rejects_unknown_keys() -> None:
    from src.input_validation import validate_config

    cfg = input_config.normalize_config({"n_scenarios": 1, "mdoe": "plet_rusle"})
    with pytest.raises(ValueError, match="Unknown configuration key.*mdoe"):
        validate_config(cfg)


def test_validate_config_rejects_unknown_load_generation_keys() -> None:
    from src.input_validation import validate_config

    cfg = input_config.normalize_config(
        {"n_scenarios": 1, "load_generation": {"mode": "plet_rusle", "mdoe": "x"}}
    )
    with pytest.raises(ValueError, match="Unknown load_generation key.*mdoe"):
        validate_config(cfg)
