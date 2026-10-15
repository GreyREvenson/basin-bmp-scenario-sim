from __future__ import annotations

import sqlite3
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box, mapping

from src import input_config
from utils import download_wbd_huc12 as wbd


class Logger:
    def verbose(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def info(self, *args, **kwargs):
        pass


def _write_parcels(path: Path) -> None:
    schema = {"geometry": "Polygon", "properties": {}}
    with fiona.open(path, "w", driver="GPKG", layer="parcels", schema=schema, crs="EPSG:4326") as dst:
        dst.write({"geometry": mapping(box(-84.10, 39.00, -84.00, 39.10)), "properties": {}})
        dst.write({"geometry": mapping(box(-83.99, 39.00, -83.89, 39.10)), "properties": {}})


def _fake_hucs(*, include_sliver: bool = False) -> gpd.GeoDataFrame:
    rows = {
        "huc12": ["050902021001", "050902021002"],
        "name": ["A", "B"],
        "states": ["OH", "OH"],
        "areaacres": [1000.0, 1200.0],
        "areasqkm": [4.0, 5.0],
    }
    geoms = [
        box(-84.2, 38.9, -83.995, 39.2),
        box(-83.995, 38.9, -83.8, 39.2),
    ]
    if include_sliver:
        rows["huc12"].append("050902021099")
        rows["name"].append("Sliver")
        rows["states"].append("OH")
        rows["areaacres"].append(100.0)
        rows["areasqkm"].append(0.4)
        geoms.append(box(-84.10, 39.00, -84.095, 39.10))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def test_download_wbd_huc12_uses_arcgis_pagination(monkeypatch) -> None:
    calls = []

    def fake_request(url, params, *, timeout):
        calls.append(dict(params))
        offset = int(params["resultOffset"])
        if offset == 0:
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"huc12": "050902021001", "name": "A"},
                        "geometry": mapping(box(-84.2, 38.9, -84.0, 39.2)),
                    }
                ],
            }
        if offset == 1:
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"huc12": "050902021002", "name": "B"},
                        "geometry": mapping(box(-84.0, 38.9, -83.8, 39.2)),
                    }
                ],
            }
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(wbd, "_request_json", fake_request)
    result = wbd.download_wbd_huc12((-84.2, 38.9, -83.8, 39.2), page_size=1)
    assert list(result["huc12"]) == ["050902021001", "050902021002"]
    assert [int(call["resultOffset"]) for call in calls] == [0, 1, 2]
    assert all(call["f"] == "geojson" for call in calls)


def test_prepare_huc12_forcing_creates_variable_tables_and_populates_export_values(
    tmp_path: Path, monkeypatch
) -> None:
    parcels = tmp_path / "parcels.gpkg"
    forcing = tmp_path / "plet_inputs.gpkg"
    _write_parcels(parcels)

    monkeypatch.setattr(
        wbd, "download_wbd_huc12", lambda *args, **kwargs: _fake_hucs(include_sliver=True)
    )
    monkeypatch.setattr(
        wbd,
        "read_initial_plet_export",
        lambda path: pd.DataFrame(
            [
                {"huc12": "050902021001", "avg_rain_in": 0.60, "rain_days": 150.0, "annual_precip_in": 44.0},
                {"huc12": "050902021002", "avg_rain_in": 0.70, "rain_days": 160.0, "annual_precip_in": 45.0},
            ]
        ),
    )

    hucs, assignments = wbd.prepare_huc12_forcing(
        parcels,
        forcing,
        area_fraction_threshold=0.75,
        initial_plet_export=tmp_path / "plet.xlsx",
    )

    assert list(hucs["huc12"]) == ["050902021001", "050902021002"]
    layers = set(fiona.listlayers(forcing))
    assert layers == {"huc12", *wbd.PLET_PARAMETER_TABLES.values()}
    assert list(assignments.sort_values("pid")["huc12"]) == ["050902021001", "050902021002"]

    spatial = gpd.read_file(forcing, layer="huc12")
    assert not any(field in spatial.columns for field in wbd.PLET_FORCING_FIELDS)

    with sqlite3.connect(forcing) as con:
        avg = con.execute(
            "SELECT huc12, value, units FROM input_avg_rain_in ORDER BY huc12"
        ).fetchall()
        rain = con.execute(
            "SELECT huc12, value FROM input_rain_days ORDER BY huc12"
        ).fetchall()
        annual = con.execute(
            "SELECT huc12, value FROM input_annual_precip_in ORDER BY huc12"
        ).fetchall()
        empty_rcor = con.execute("SELECT COUNT(*) FROM input_rain_correction_fraction").fetchone()[0]
        empty_rdcor = con.execute("SELECT COUNT(*) FROM input_runoff_day_fraction").fetchone()[0]
    assert avg[0] == ("050902021001", pytest.approx(0.60), "in/event")
    assert rain[1][1] == pytest.approx(160.0)
    assert annual[0][1] == pytest.approx(44.0)
    assert empty_rcor == 0
    assert empty_rdcor == 0


def test_refresh_preserves_distribution_rows_from_geopackage(tmp_path: Path, monkeypatch) -> None:
    parcels = tmp_path / "parcels.gpkg"
    forcing = tmp_path / "plet_inputs.gpkg"
    _write_parcels(parcels)

    existing_tables = {
        "avg_rain_in": pd.DataFrame(
            [
                {"huc12": "050902021001", "mean": 0.61, "sd": 0.04, "units": "in/event"},
                {"huc12": "050902021002", "mean": 0.71, "sd": 0.05, "units": "in/event"},
            ]
        ),
        "rain_days": pd.DataFrame(
            [
                {"huc12": "050902021001", "value": 151.0, "units": "days/year"},
                {"huc12": "050902021002", "value": 161.0, "units": "days/year"},
            ]
        ),
    }
    wbd.write_huc12_package(forcing, _fake_hucs(), existing_tables)

    monkeypatch.setattr(
        wbd, "download_wbd_huc12", lambda *args, **kwargs: _fake_hucs(include_sliver=True)
    )

    hucs, _ = wbd.prepare_huc12_forcing(parcels, forcing, area_fraction_threshold=0.75)
    assert "050902021099" not in set(hucs["huc12"])

    tables = wbd.read_existing_parameter_tables(forcing)
    avg = tables["avg_rain_in"].set_index("huc12")
    assert avg.loc["050902021001", "mean"] == pytest.approx(0.61)
    assert avg.loc["050902021001", "sd"] == pytest.approx(0.04)
    rain = tables["rain_days"].set_index("huc12")
    assert rain.loc["050902021002", "value"] == pytest.approx(161.0)

    layer = input_config._load_plet_huc12_layer({"plet_forcing": str(forcing)}, Logger())
    loaded = input_config._assemble_huc12_plet_parameter_source(
        {"plet_forcing": str(forcing)}, layer, Logger()
    )
    row = loaded[
        (loaded["huc12"] == "050902021002") & (loaded["parameter"] == "avg_rain_in")
    ].iloc[0]
    assert row["mean"] == pytest.approx(0.71)
    assert row["sd"] == pytest.approx(0.05)


def test_threshold_filter_uses_same_below_threshold_semantics(tmp_path: Path) -> None:
    parcels = tmp_path / "parcels.gpkg"
    _write_parcels(parcels)
    parcels_wgs84, _ = wbd.parcel_footprint_wgs84(parcels)

    hucs = _fake_hucs(include_sliver=True)
    retained = wbd.retain_hucs_by_parcel_overlap_threshold(
        hucs, parcels_wgs84, threshold=0.75
    )
    assert set(retained["huc12"]) == {"050902021001", "050902021002"}


def test_read_initial_plet_export(tmp_path: Path) -> None:
    path = tmp_path / "plet_export.xlsx"
    source = pd.DataFrame(
        [
            {
                "WATERSHED": "050902021001 - Turtle Creek",
                "AVG_RAIN": 0.598115499443718,
                "RAIN_DAYS": 155.933,
                "ANNUAL_RAINFALL": 44.629,
            },
            {
                "WATERSHED": "050902021002 - Headwaters",
                "AVG_RAIN": 0.5736663692569752,
                "RAIN_DAYS": 153.633,
                "ANNUAL_RAINFALL": 43.348,
            },
        ]
    )
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        source.to_excel(writer, sheet_name="1. Watershed Land Use", index=False)

    table = wbd.read_initial_plet_export(path)
    assert len(table) == 2
    by_huc = table.set_index("huc12")
    assert by_huc.loc["050902021001", "avg_rain_in"] == pytest.approx(0.598115499443718)
    assert by_huc.loc["050902021001", "rain_days"] == pytest.approx(155.933)
    assert by_huc.loc["050902021001", "annual_precip_in"] == pytest.approx(44.629)


def test_initial_plet_export_rejected_after_huc12_inputs_have_values(tmp_path: Path, monkeypatch) -> None:
    parcels = tmp_path / "parcels.gpkg"
    forcing = tmp_path / "plet_inputs.gpkg"
    _write_parcels(parcels)
    wbd.write_huc12_package(
        forcing,
        _fake_hucs(),
        {"avg_rain_in": pd.DataFrame([{"huc12": "050902021001", "value": 0.6}])},
    )
    monkeypatch.setattr(wbd, "download_wbd_huc12", lambda *args, **kwargs: _fake_hucs())

    with pytest.raises(ValueError, match="before HUC12 input tables contain values"):
        wbd.prepare_huc12_forcing(
            parcels,
            forcing,
            initial_plet_export=tmp_path / "plet.xlsx",
        )
