from __future__ import annotations

import sqlite3
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box, mapping

from src import input_config, sampling
from src.logging_utils import make_logger
from src.plet_rusle import _sample_parameter_table
from utils import download_wbd_huc12 as wbd
from utils.create_parcel_huc12 import assign_parcels_to_huc12, write_parcel_huc12


class Logger:
    def verbose(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass


class Ctx:
    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def _sample_from_stats(self, *args, **kwargs):
        return sampling._sample_from_stats(self, *args, **kwargs)

    def _trunc_normal(self, *args, **kwargs):
        return sampling._trunc_normal(self, *args, **kwargs)

    def _piecewise_quantile_sample(self, *args, **kwargs):
        return sampling._piecewise_quantile_sample(self, *args, **kwargs)


@pytest.mark.parametrize("verbose", [False, True])
def test_low_parcel_huc12_overlap_is_logged_without_console_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], verbose: bool
) -> None:
    parcels = tmp_path / "parcels.gpkg"
    with sqlite3.connect(parcels) as con:
        pd.DataFrame([
            {"pid": 1, "huc12": "050902021001", "area_fraction": 0.6},
            {"pid": 2, "huc12": "050902021001", "area_fraction": 0.9},
        ]).to_sql("parcel_huc12", con, index=False)
    logger, log_path = make_logger(tmp_path / "outputs", verbose=verbose, console=True)

    result = input_config._load_parcel_huc12(
        {"parcels": str(parcels)}, [1, 2], [1, 2], logger
    )
    logger.warning("ordinary warning stays on console")

    assert len(result) == 2
    assert "less than 75%" in log_path.read_text(encoding="utf-8")
    assert "'pid': 1" in log_path.read_text(encoding="utf-8")
    console = capsys.readouterr()
    assert "ordinary warning stays on console" in console.err
    assert "less than 75%" not in console.err


def test_huc12_parameter_precedence_is_exact_then_huc_then_default() -> None:
    parcel_source = pd.DataFrame(
        [
            {"pid": None, "parameter": "avg_rain_in", "value": 0.50},
            {"pid": None, "parameter": "rain_days", "value": 120.0},
            {"pid": None, "parameter": "runoff_day_fraction", "value": 0.40},
            {"pid": 2, "parameter": "avg_rain_in", "value": 0.99},
            {"pid": None, "parameter": "land_cover", "value": "cropland"},
            {"pid": None, "parameter": "hsg", "value": "B"},
        ]
    )
    parcel_huc12 = pd.DataFrame(
        [
            {"pid": 1, "huc12": "050902021001", "area_fraction": 1.0},
            {"pid": 2, "huc12": "050902021002", "area_fraction": 0.8},
        ]
    )
    huc_source = pd.DataFrame(
        [
            {"huc12": "050902021001", "parameter": "avg_rain_in", "value": 0.60},
            {"huc12": "050902021001", "parameter": "rain_days", "value": 150.0},
            {"huc12": "050902021002", "parameter": "avg_rain_in", "value": 0.70},
            {"huc12": "050902021002", "parameter": "rain_days", "value": 160.0},
        ]
    )

    merged = input_config._merge_huc12_plet_parameters(
        parcel_source, parcel_huc12, huc_source, [1, 2]
    )

    def exact(pid: int, parameter: str) -> float:
        rows = merged[(merged["pid"] == pid) & (merged["parameter"] == parameter)]
        assert len(rows) == 1
        return float(rows.iloc[0]["value"])

    assert exact(1, "avg_rain_in") == pytest.approx(0.60)
    assert exact(1, "rain_days") == pytest.approx(150.0)
    assert exact(2, "avg_rain_in") == pytest.approx(0.99)  # parcel override
    assert exact(2, "rain_days") == pytest.approx(160.0)
    assert len(merged[(merged["pid"].isna()) & (merged["parameter"] == "runoff_day_fraction")]) == 1


def test_huc12_distribution_is_sampled_once_per_huc12_per_scenario() -> None:
    parcel_source = pd.DataFrame(
        [
            {"pid": None, "parameter": "rain_days", "value": 120.0},
            {"pid": None, "parameter": "runoff_day_fraction", "value": 0.40},
            {"pid": None, "parameter": "land_cover", "value": "cropland"},
            {"pid": None, "parameter": "hsg", "value": "B"},
        ]
    )
    parcel_huc12 = pd.DataFrame(
        [
            {"pid": 1, "huc12": "050902021001", "area_fraction": 1.0},
            {"pid": 2, "huc12": "050902021001", "area_fraction": 1.0},
            {"pid": 3, "huc12": "050902021002", "area_fraction": 1.0},
        ]
    )
    huc_source = pd.DataFrame(
        [
            {"huc12": "050902021001", "parameter": "avg_rain_in", "mean": 0.60, "sd": 0.05},
            {"huc12": "050902021002", "parameter": "avg_rain_in", "mean": 0.70, "sd": 0.05},
        ]
    )
    merged = input_config._merge_huc12_plet_parameters(
        parcel_source, parcel_huc12, huc_source, [1, 2, 3]
    )
    rows = merged[(merged["parameter"] == "avg_rain_in") & merged["pid"].notna()]
    groups = rows.set_index("pid")["sample_group"].to_dict()
    assert groups[1] == groups[2] == "huc12:050902021001:avg_rain_in"
    assert groups[3] == "huc12:050902021002:avg_rain_in"

    values = _sample_parameter_table(Ctx(seed=7), merged, ["1", "2", "3"], cache_prefix="plet")
    assert values[0]["avg_rain_in"] == values[1]["avg_rain_in"]
    assert values[2]["avg_rain_in"] != values[0]["avg_rain_in"]


def test_plet_huc12_loader_requires_spatial_huc12_layer(tmp_path: Path) -> None:
    path = tmp_path / "plet_inputs.gpkg"
    with sqlite3.connect(path) as con:
        pd.DataFrame([{"huc12": "050902021001", "value": 0.598}]).to_sql(
            "input_avg_rain_in", con, index=False
        )
    with pytest.raises(input_config.MissingInputTableError, match="spatial layer 'huc12'"):
        input_config._load_plet_huc12_layer({"plet_forcing": str(path)}, Logger())


def _write_polygon_gpkg(path: Path, layer: str, schema: dict, features: list[dict]) -> None:
    with fiona.open(path, "w", driver="GPKG", layer=layer, schema=schema, crs="EPSG:4326") as dst:
        for feature in features:
            dst.write(feature)


def test_create_parcel_huc12_uses_greatest_area_overlap(tmp_path: Path) -> None:
    parcels = tmp_path / "parcels.gpkg"
    hucs = tmp_path / "hucs.gpkg"
    _write_polygon_gpkg(
        parcels,
        "parcels",
        {"geometry": "Polygon", "properties": {}},
        [
            {"geometry": mapping(box(0.0, 0.0, 1.0, 1.0)), "properties": {}},
            {"geometry": mapping(box(0.8, 0.0, 1.8, 1.0)), "properties": {}},
        ],
    )
    _write_polygon_gpkg(
        hucs,
        "huc12",
        {"geometry": "Polygon", "properties": {"huc12": "str:12"}},
        [
            {"geometry": mapping(box(-1.0, -1.0, 1.0, 2.0)), "properties": {"huc12": "050902021001"}},
            {"geometry": mapping(box(1.0, -1.0, 3.0, 2.0)), "properties": {"huc12": "050902021002"}},
        ],
    )

    result = assign_parcels_to_huc12(parcels, hucs, huc_layer="huc12")
    by_pid = result.set_index("pid")
    assert by_pid.loc[1, "huc12"] == "050902021001"
    assert by_pid.loc[1, "area_fraction"] == pytest.approx(1.0, abs=1e-4)
    assert by_pid.loc[2, "huc12"] == "050902021002"
    assert by_pid.loc[2, "area_fraction"] == pytest.approx(0.8, abs=0.02)

    write_parcel_huc12(parcels, result)
    with sqlite3.connect(parcels) as con:
        rows = con.execute(
            "SELECT pid, huc12, area_fraction FROM parcel_huc12 ORDER BY pid"
        ).fetchall()
        registered = con.execute(
            "SELECT data_type FROM gpkg_contents WHERE table_name='parcel_huc12'"
        ).fetchone()
    assert rows[0][1] == "050902021001"
    assert rows[1][1] == "050902021002"
    assert registered == ("attributes",)


def test_resolve_config_paths_resolves_plet_forcing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config" / "model.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text("n_scenarios: 1\n", encoding="utf-8")
    cfg = {"plet_forcing": "../plet/plet_inputs.gpkg"}
    input_config.resolve_config_paths(cfg, cfg_path)
    assert Path(cfg["plet_forcing"]) == (tmp_path / "plet" / "plet_inputs.gpkg").resolve()


def test_huc12_input_tables_are_loaded_separately_from_spatial_layer(tmp_path: Path) -> None:
    path = tmp_path / "plet_inputs.gpkg"
    hucs = gpd.GeoDataFrame(
        {
            "huc12": ["050902021001"],
            "name": ["Example"],
        },
        geometry=[box(-84.0, 39.0, -83.9, 39.1)],
        crs="EPSG:4326",
    )
    tables = {
        "avg_rain_in": pd.DataFrame(
            [{"huc12": "050902021001", "mean": 0.598, "sd": 0.03, "units": "in/event"}]
        ),
        "rain_days": pd.DataFrame(
            [{"huc12": "050902021001", "value": 155.9, "units": "days/year"}]
        ),
    }
    wbd.write_huc12_package(path, hucs, tables)

    layer = input_config._load_plet_huc12_layer({"plet_forcing": str(path)}, Logger())
    loaded = input_config._assemble_huc12_plet_parameter_source(
        {"plet_forcing": str(path)}, layer, Logger()
    )
    assert "avg_rain_in" not in layer.columns
    avg = loaded[loaded["parameter"] == "avg_rain_in"].iloc[0]
    assert avg["huc12"] == "050902021001"
    assert avg["mean"] == pytest.approx(0.598)
    assert avg["sd"] == pytest.approx(0.03)
    rain = loaded[loaded["parameter"] == "rain_days"].iloc[0]
    assert rain["value"] == pytest.approx(155.9)


def test_huc12_fraction_units_are_validated_with_parameter_context(tmp_path: Path) -> None:
    path = tmp_path / "plet_inputs.gpkg"
    hucs = gpd.GeoDataFrame(
        {"huc12": ["050902021001"], "name": ["Example"]},
        geometry=[box(-84.0, 39.0, -83.9, 39.1)],
        crs="EPSG:4326",
    )
    tables = {
        "rain_correction_fraction": pd.DataFrame(
            [{"huc12": "050902021001", "value": 0.861, "units": "fraction"}]
        ),
        "runoff_day_fraction": pd.DataFrame(
            [{"huc12": "050902021001", "value": 0.412, "units": "fraction"}]
        ),
    }
    wbd.write_huc12_package(path, hucs, tables)

    layer = input_config._load_plet_huc12_layer({"plet_forcing": str(path)}, Logger())
    loaded = input_config._assemble_huc12_plet_parameter_source(
        {"plet_forcing": str(path)}, layer, Logger()
    )

    rcor = loaded[loaded["parameter"] == "rain_correction_fraction"].iloc[0]
    rdcor = loaded[loaded["parameter"] == "runoff_day_fraction"].iloc[0]
    assert rcor["value"] == pytest.approx(0.861)
    assert rdcor["value"] == pytest.approx(0.412)
