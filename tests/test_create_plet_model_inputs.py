"""Offline checks for the combined PLET preparation utility."""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

from src import input_config
from utils import create_plet_model_inputs as utility


WORKBOOK = (Path(__file__).resolve().parents[1] / "examples" / "east_fork"
            / "inputs" / "misc" / "EastFork_plet_inputs.xlsx")
HUC_GEOMETRY = (Path(__file__).resolve().parents[1] / "examples" / "east_fork"
                / "inputs" / "plet" / "plet_inputs_per_huc12.gpkg")
EAST_FORK_PARCELS = (Path(__file__).resolve().parents[1] / "examples" / "east_fork"
                     / "inputs" / "parcels" / "parcels_plet_constants.gpkg")
HUC_A = "050902021001"
HUC_B = "050902021005"


@pytest.mark.parametrize("module_name", [
    "utils.create_plet_model_inputs", "create_plet_model_inputs",
])
def test_workbook_reads_after_geospatial_io_exit_cleanly(module_name):
    script = textwrap.dedent("""
        import importlib
        import sys
        from pathlib import Path

        import geopandas as gpd
        import pandas as pd

        root = Path.cwd()
        sys.path.insert(0, str(root / "utils"))
        utility = importlib.import_module(sys.argv[1])
        workbook = Path(sys.argv[2])
        geometry = Path(sys.argv[3])
        pd.options.io.excel.xlsx.reader = "openpyxl"
        for _ in range(3):
            hucs = gpd.read_file(geometry, layer="huc12")
            export = utility.read_initial_plet_export(workbook)
            tables, hydrology = utility.workbook_tables(
                workbook, {"050902021001", "050902021005"}, export=export)
            assert len(export) == len(hucs) == 18
            assert len(tables["surface_concentration"]) == 30
            assert len(hydrology["cn"]) == 16
        assert "openpyxl" not in sys.modules
        assert "lxml.etree" not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script,
         module_name, str(WORKBOOK), str(HUC_GEOMETRY)],
        cwd=WORKBOOK.parents[4],
        env={**os.environ, "OPENPYXL_LXML": "True"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"Workbook/geospatial subprocess exited with {result.returncode}\n"
        f"{result.stdout}\n{result.stderr}"
    )


def _raster(bounds):
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=20, height=10, count=1,
                      dtype="uint8", crs="EPSG:5070",
                      transform=from_origin(-300, 300, 30, 30), nodata=0) as dst:
            values = np.full((10, 20), 82, dtype="uint8")
            values[:, 10:] = 41
            dst.write(values, 1)
        return utility.NLCDImage(mem.read(), 2024, utility.NLCD_SERVICE_URL)


def _parcel_fixture(path: Path):
    parcels = gpd.GeoDataFrame(
        {"pid": [1, 2], "geometry": [box(-290, 10, -30, 290),
                                      box(30, 10, 290, 290)]}, crs="EPSG:5070",
    )
    parcels.to_file(path, layer="parcels", driver="GPKG")
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE input_rain_days (id INTEGER PRIMARY KEY, pid INTEGER, value REAL)")
        con.execute("INSERT INTO input_rain_days(pid,value) VALUES (1,111)")
        con.execute("CREATE TABLE input_hsg (id INTEGER PRIMARY KEY, pid INTEGER, value TEXT)")
        con.execute("INSERT INTO input_hsg(pid,value) VALUES (1,'D')")
        con.execute("CREATE TABLE custom_overrides (id INTEGER PRIMARY KEY, value TEXT)")
        con.execute("INSERT INTO custom_overrides(value) VALUES ('untouched')")
    return parcels


def _hucs():
    return gpd.GeoDataFrame(
        {"huc12": [HUC_A, HUC_B],
         "geometry": [box(-300, -50, 0, 350), box(0, -50, 300, 350)]},
        crs="EPSG:5070",
    ).to_crs("EPSG:4326")


def test_workbook_numeric_tables_and_reconstruction(caplog):
    with caplog.at_level(logging.WARNING, logger=utility.LOGGER.name):
        tables, hydrology = utility.workbook_tables(WORKBOOK, {HUC_A, HUC_B})
    assert tables["hsg"].set_index("huc12").loc[HUC_A, "value"] == "C"
    for parameter in ("avg_rain_in", "annual_precip_in", "rain_days"):
        assert set(tables[parameter]["huc12"]) == {HUC_A, HUC_B}
    for parameter, expected, field in (
        ("sediment_n_pct", .08, "SOIL_N_CONC"),
        ("sediment_p_pct", .0308, "SOIL_P_CONC"),
    ):
        rows = tables[parameter].set_index("huc12")
        assert set(rows.index) == {HUC_A, HUC_B}
        assert rows.loc[HUC_A, "value"] == pytest.approx(expected)
        assert rows.loc[HUC_A, "units"] == "%"
        assert f"PLET sheet 4, {field}" in rows.loc[HUC_A, "notes"]
    crop = tables["r"].query("huc12 == @HUC_A and land_cover == 'cropland'")
    assert crop.iloc[0]["value"] == 150
    assert "urban" not in set(tables["r"]["land_cover"])
    assert len(tables["surface_concentration"]) == 30
    assert len(tables["subsurface_concentration"]) == 20
    assert set(tables["subsurface_concentration"]["value"]) == {0}
    tss = tables["surface_concentration"].query("pollutant == 'TSS'")
    assert len(tss) == 2 * len(utility.LAND_COVERS)
    assert set(zip(tss["huc12"], tss["land_cover"])) == {
        (huc, cover) for huc in (HUC_A, HUC_B) for cover in utility.LAND_COVERS
    }
    assert set(tss["value"]) == {0.0}
    assert tss["notes"].str.contains("PLACEHOLDER").all()
    assert tables["subsurface_concentration"].query("pollutant == 'TSS'").empty
    assert any("TSS PLACEHOLDER" in rec.message and "urban RUSLE" in rec.message
               for rec in caplog.records)
    assert hydrology["cn"].query("land_cover == 'urban' and hsg == 'A'").iloc[0]["value"] == 83
    assert len(hydrology["cn"]) == 16  # PLET exports zero, undefined user-defined CNs.
    assert hydrology["infiltration_fraction"].query(
        "land_cover == 'forest' and hsg == 'B'").iloc[0]["value"] == .30

    export = utility.read_initial_plet_export(WORKBOOK)
    rcor, rdcor, audit = utility.reconstruct_rain_factors(
        export[export["huc12"].isin((HUC_A, HUC_B))], _hucs())
    assert audit["relative_error"].max() < 1e-7
    assert {"annual_rainfall", "rain_days", "absolute_error"} <= set(audit.columns)
    assert set(audit["huc12"]) == {HUC_A, HUC_B}
    assert all(.840 <= value <= .900 for value in rcor["value"])
    assert all(.350 <= value <= .500 for value in rdcor["value"])
    assert "ambiguous" in audit.set_index("huc12").loc[HUC_B, "status"]
    assert "not independently verified" in rcor.set_index("huc12").loc[HUC_A, "notes"]


def test_east_fork_existing_inputs_cover_missing_workbook_fields():
    with sqlite3.connect(f"{EAST_FORK_PARCELS.resolve().as_uri()}?mode=ro", uri=True) as con:
        for parameter in ("r", "k", "ls", "c", "p"):
            assert con.execute(
                f"SELECT COUNT(*) FROM input_rusle_{parameter} "
                "WHERE pid IS NULL AND value IS NOT NULL"
            ).fetchone() == (1,)
        for pollutant in ("TN", "TP"):
            for pathway in ("surface", "subsurface"):
                assert con.execute(
                    f"SELECT COUNT(*) FROM input_{pathway}_concentration "
                    "WHERE pid IS NULL AND pollutant=? AND value IS NOT NULL",
                    (pollutant,),
                ).fetchone() == (1,)
        assert con.execute(
            "SELECT COUNT(*) FROM input_curve_number "
            "WHERE land_cover='user_defined' AND value>0"
        ).fetchone() == (4,)


def test_sheet_4_missing_and_invalid_huc_values_fail(monkeypatch):
    real_huc_sheet = utility._huc_sheet

    def sheet_with_missing_huc(workbook, name, fields):
        frame = real_huc_sheet(workbook, name, fields)
        return frame.drop(index=HUC_B) if name == "4. Nutrient and E.Coli Content" else frame

    monkeypatch.setattr(utility, "_huc_sheet", sheet_with_missing_huc)
    with pytest.raises(ValueError, match="PLET sheet 4 missing HUC12s"):
        utility.workbook_tables(WORKBOOK, {HUC_A, HUC_B})

    def sheet_with_invalid_pct(workbook, name, fields):
        frame = real_huc_sheet(workbook, name, fields)
        if name == "4. Nutrient and E.Coli Content":
            frame.loc[HUC_A, "SOIL_N_CONC"] = 101
        return frame

    monkeypatch.setattr(utility, "_huc_sheet", sheet_with_invalid_pct)
    with pytest.raises(ValueError, match="SOIL_N_CONC must be a percent"):
        utility.workbook_tables(WORKBOOK, {HUC_A})


def test_east_fork_ambiguous_pairs_use_neighboring_unique_huc_geometry(caplog):
    export = utility.read_initial_plet_export(WORKBOOK)
    # Read only polygons; previously exported factors must not influence estimates.
    hucs = gpd.read_file(HUC_GEOMETRY, layer="huc12")[["huc12", "geometry"]]
    with caplog.at_level(logging.INFO, logger=utility.LOGGER.name):
        rcor, rdcor, audit = utility.reconstruct_rain_factors(export, hucs)
    rcor = rcor.set_index("huc12")
    rdcor = rdcor.set_index("huc12")
    audit = audit.set_index("huc12")
    assert (rcor.loc[HUC_B, "value"], rdcor.loc[HUC_B, "value"]) == (.870, .435)
    assert (rcor.loc["050902021101", "value"],
            rdcor.loc["050902021101", "value"]) == (.874, .437)
    assert len(audit) == len(export) == 18
    assert audit["relative_error"].max() < 1e-7
    for huc in (HUC_B, "050902021101"):
        assert "nearest unique HUC12s" in rcor.loc[huc, "notes"]
        assert "not independently verified" in rdcor.loc[huc, "notes"]
        row = export.set_index("huc12").loc[huc]
        assert rdcor.loc[huc, "value"] / rcor.loc[huc, "value"] == pytest.approx(
            row["annual_precip_in"] / (row["avg_rain_in"] * row["rain_days"]),
            rel=1e-7,
        )
    rain_audit = [record.message for record in caplog.records if "Rain audit HUC12" in record.message]
    assert len(rain_audit) == 18
    assert all("annual_rainfall=" in line and "rain_days=" in line and
               "exported_avg_rain=" in line and "reconstructed_avg_rain=" in line and
               "absolute_error=" in line and "relative_error=" in line and
               "status=" in line for line in rain_audit)


def test_reconstruction_rejects_no_solution_and_unanchored_ambiguity():
    sample = pd.DataFrame([{"huc12": HUC_A, "avg_rain_in": 999,
                            "rain_days": 100, "annual_precip_in": 40}])
    with pytest.raises(ValueError, match="no Rcor/RDcor grid solution"):
        utility.reconstruct_rain_factors(sample, _hucs())
    sample["avg_rain_in"] = 40 * .85 / (100 * .425)
    with pytest.raises(ValueError, match="no uniquely reconstructed HUC12"):
        utility.reconstruct_rain_factors(sample, _hucs())


def test_nlcd_dominant_cover_and_unmapped_code():
    parcels = gpd.GeoDataFrame(
        {"pid": [1, 2], "geometry": [box(-290, 10, -30, 290),
                                      box(30, 10, 290, 290)]}, crs="EPSG:5070")
    classified = utility.parcel_nlcd_land_cover(parcels, source=_raster, verbose=True)
    assert classified.set_index("pid")["value"].to_dict() == {1: "cropland", 2: "forest"}
    assert all("NLCD 2024" in note and utility.NLCD_SERVICE_URL in note
               for note in classified["notes"])
    def unmapped(bounds):
        with MemoryFile(_raster(bounds).data) as mem:
            with mem.open() as src:
                profile = src.profile
            with MemoryFile() as dest:
                with dest.open(**profile) as raster:
                    raster.write(np.full((10, 20), 11, dtype="uint8"), 1)
                return utility.NLCDImage(dest.read(), 2024, utility.NLCD_SERVICE_URL)
    with pytest.raises(ValueError, match=r"Unmapped dominant NLCD codes: 11: 2 parcel"):
        utility.parcel_nlcd_land_cover(parcels, source=unmapped)


def test_nlcd_assumptions_are_visible(caplog):
    parcels = gpd.GeoDataFrame(
        {"pid": [802, 803], "geometry": [box(0, 0, 30, 30), box(30, 0, 60, 30)]},
        crs="EPSG:5070",
    )

    def barren(bounds):
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=2, height=1, count=1,
                          dtype="uint8", crs="EPSG:5070",
                          transform=from_origin(0, 30, 30, 30), nodata=0) as raster:
                raster.write(np.array([[31, 90]], dtype="uint8"), 1)
            return utility.NLCDImage(mem.read(), 2024, utility.NLCD_SERVICE_URL)

    with caplog.at_level(logging.INFO, logger=utility.LOGGER.name):
        classified = utility.parcel_nlcd_land_cover(parcels, source=barren, verbose=True)
    assert classified.set_index("pid")["value"].to_dict() == {
        802: "pastureland", 803: "forest",
    }
    assert "ASSUMPTION: barren (31) treated as PLET pastureland" in classified.iloc[0]["notes"]
    assert "ASSUMPTION: woody wetlands (90) treated as PLET forest" in classified.iloc[1]["notes"]
    assert any(
        record.levelno == logging.WARNING
        and "barren class 31 as PLET pastureland for 1 parcel(s)" in record.message
        and "802" in record.message
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.WARNING
        and "woody wetlands class 90 as PLET forest for 1 parcel(s)" in record.message
        and "803" in record.message
        for record in caplog.records
    )
    assert any("pid=802" in record.message and "31=100.00%" in record.message
               for record in caplog.records)
    assert any("pid=803" in record.message and "90=100.00%" in record.message
               for record in caplog.records)


def test_nlcd_area_weighted_fractions_include_tiny_parcel_without_pixel_center(caplog):
    def raster(bounds):
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=2, height=1, count=1, dtype="uint8",
                          crs="EPSG:5070", transform=from_origin(0, 30, 30, 30),
                          nodata=0) as dataset:
                dataset.write(np.array([[82, 41]], dtype="uint8"), 1)
            return utility.NLCDImage(mem.read(), 2024, utility.NLCD_SERVICE_URL)

    parcels = gpd.GeoDataFrame(
        {"pid": [1, 2], "geometry": [box(2, 28, 4, 30), box(25, 5, 50, 25)]},
        crs="EPSG:5070",
    )
    with caplog.at_level(logging.INFO, logger=utility.LOGGER.name):
        classified = utility.parcel_nlcd_land_cover(parcels, source=raster, verbose=True)
    assert classified.set_index("pid")["value"].to_dict() == {1: "cropland", 2: "forest"}
    assert all(f"NLCD 2024" in note and utility.NLCD_SERVICE_URL in note
               for note in classified["notes"])
    details = [r.message for r in caplog.records if "NLCD parcel pid=" in r.message]
    assert len(details) == 2
    assert "cropland=100.00%" in details[0]
    assert "82=20.00%" in details[1] and "41=80.00%" in details[1]
    assert "cropland=20.00%" in details[1] and "forest=80.00%" in details[1]

    def nodata(bounds):
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=1, height=1, count=1, dtype="uint8",
                          crs="EPSG:5070", transform=from_origin(0, 30, 30, 30),
                          nodata=0) as dataset:
                dataset.write(np.zeros((1, 1), dtype="uint8"), 1)
            return utility.NLCDImage(mem.read(), 2024, utility.NLCD_SERVICE_URL)

    with pytest.raises(ValueError, match="No valid NLCD pixel area inside parcel 1"):
        utility.parcel_nlcd_land_cover(parcels.iloc[[0]], source=nodata)

    def partially_nodata(bounds):
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=2, height=1, count=1, dtype="uint8",
                          crs="EPSG:5070", transform=from_origin(0, 30, 30, 30),
                          nodata=0) as dataset:
                dataset.write(np.array([[82, 0]], dtype="uint8"), 1)
            return utility.NLCDImage(mem.read(), 2024, utility.NLCD_SERVICE_URL)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=utility.LOGGER.name):
        partial = utility.parcel_nlcd_land_cover(parcels.iloc[[1]],
                                                 source=partially_nodata, verbose=True)
    assert partial.iloc[0]["value"] == "cropland"
    assert any("only 20.00% valid raster coverage" in record.message
               for record in caplog.records)
    assert any("cropland=100.00%" in record.message and "valid_coverage=20.00%" in record.message
               for record in caplog.records)

    def undeclared_nodata(bounds):
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", width=1, height=1, count=1, dtype="uint8",
                          crs="EPSG:5070", transform=from_origin(0, 30, 30, 30)) as dataset:
                dataset.write(np.zeros((1, 1), dtype="uint8"), 1)
            return utility.NLCDImage(mem.read(), 2024, utility.NLCD_SERVICE_URL)

    with pytest.raises(ValueError, match="code 0 intersects parcel 1"):
        utility.parcel_nlcd_land_cover(parcels.iloc[[0]], source=undeclared_nodata)


def test_nlcd_arcgis_latest_year_and_export_are_validated(monkeypatch):
    class Response:
        def __init__(self, payload=None, content=b""):
            self.payload = payload
            self.content = content

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    calls = []
    def get(url, *, params=None, timeout=None):
        calls.append((url, params))
        if url.endswith("/query"):
            return Response({"features": [{"attributes": {"Year": 2021}},
                                          {"attributes": {"Year": 2024}}]})
        if url.endswith("/exportImage"):
            return Response({"href": f"{utility.NLCD_SERVICE_URL}/images/nlcd.tif"})
        return Response(content=_raster((0, 0, 30, 30)).data)

    monkeypatch.setattr(utility.requests, "get", get)
    result = utility.download_nlcd((-300, -300, 300, 300))
    assert result.year == 2024
    assert result.source == utility.NLCD_SERVICE_URL
    assert result.data.startswith(b"II*\x00")
    assert calls[0][1]["returnDistinctValues"] == "true"
    assert calls[0][1]["orderByFields"] == "Year DESC"
    assert calls[1][1]["bboxSR"] == calls[1][1]["imageSR"] == 5070
    assert calls[1][1]["format"] == "tiff"
    assert calls[1][1]["pixelType"] == "U8"
    assert calls[1][1]["interpolation"] == "RSP_NearestNeighbor"
    import json
    assert json.loads(calls[1][1]["mosaicRule"]) == {
        "mosaicMethod": "esriMosaicAttribute", "where": "Year = 2024",
        "sortField": "Year", "sortValue": 2024,
    }

    def invalid_year(url, *, params=None, timeout=None):
        return Response({"features": [{"attributes": {"date": 2024}}]})
    monkeypatch.setattr(utility.requests, "get", invalid_year)
    with pytest.raises(RuntimeError, match="invalid Year"):
        utility.download_nlcd((-300, -300, 300, 300))

    def bad_href(url, *, params=None, timeout=None):
        if url.endswith("/query"):
            return Response({"features": [{"attributes": {"Year": 2024}}]})
        return Response({"href": "http://example.com/nlcd.tif"})
    monkeypatch.setattr(utility.requests, "get", bad_href)
    with pytest.raises(RuntimeError, match="unexpected href"):
        utility.download_nlcd((-300, -300, 300, 300))


def test_nlcd_verbose_logs_parcel_class_percentages(caplog):
    parcels = gpd.GeoDataFrame(
        {"pid": [1, 2], "geometry": [box(-290, 10, -30, 290),
                                      box(30, 10, 290, 290)]}, crs="EPSG:5070")
    with caplog.at_level(logging.INFO, logger=utility.LOGGER.name):
        utility.parcel_nlcd_land_cover(parcels, source=_raster, verbose=True)
    details = [record.message for record in caplog.records if "NLCD parcel pid=" in record.message]
    assert len(details) == 2
    assert all("year=2024" in item and "code_percentages=[" in item
               and "category_percentages=[" in item for item in details)
    assert "cropland=100.00%" in details[0]
    assert "forest=100.00%" in details[1]


def test_cli_verbose_writes_audit_log_and_console_warning(tmp_path, monkeypatch, capsys):
    log_path = tmp_path / "audit.log"
    parcels = tmp_path / "parcels.gpkg"
    workbook = tmp_path / "export.xlsx"
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        utility.LOGGER.info("NLCD parcel pid=1 category_percentages=[cropland=100.00%]")
        utility.LOGGER.warning("TSS PLACEHOLDER: review zero surface concentrations")
        return tmp_path / "parcels_plet.gpkg", tmp_path / "parcels_plet_huc12.gpkg"

    monkeypatch.setattr(utility, "prepare_plet_inputs", prepare)
    monkeypatch.setattr(sys, "argv", ["create_plet_model_inputs.py", str(parcels),
                                      "parcels", str(workbook), "--verbose",
                                      "--log-path", str(log_path)])
    utility.main()
    assert calls[0][1]["verbose"] is True
    assert "NLCD parcel pid=1 category_percentages=[cropland=100.00%]" in log_path.read_text()
    assert "TSS PLACEHOLDER: review zero surface concentrations" in log_path.read_text()
    assert "TSS PLACEHOLDER: review zero surface concentrations" in capsys.readouterr().err


def test_offline_end_to_end_preserves_existing_overrides(tmp_path, monkeypatch):
    parcel_path = tmp_path / "site.gpkg"
    _parcel_fixture(parcel_path)
    with sqlite3.connect(parcel_path) as con:
        con.execute("CREATE TABLE input_surface_concentration (id INTEGER PRIMARY KEY, "
                    "pid INTEGER, pollutant TEXT, value REAL)")
        con.execute("INSERT INTO input_surface_concentration(pid,pollutant,value) "
                    "VALUES (1,'TSS',12.5)")
        con.execute("CREATE TABLE input_curve_number (id INTEGER PRIMARY KEY, "
                    "land_cover TEXT, hsg TEXT, value REAL, notes TEXT)")
        con.execute("INSERT INTO input_curve_number(land_cover,hsg,value,notes) "
                    "VALUES ('cropland','A',91,'existing user override')")
        con.execute("CREATE TABLE input_infiltration_fraction (id INTEGER PRIMARY KEY, "
                    "land_cover TEXT, hsg TEXT, value REAL, mean REAL, sd REAL, notes TEXT)")
        con.execute("INSERT INTO input_infiltration_fraction(land_cover,hsg,mean,sd,notes) "
                    "VALUES ('forest','B',0.82,0.03,'existing user override')")
    hucs = _hucs()
    monkeypatch.setattr(utility, "download_wbd_huc12", lambda bounds: hucs)
    monkeypatch.setattr(utility, "retain_hucs_by_parcel_overlap_threshold",
                        lambda polygons, parcels, threshold: polygons)

    parcel_out, forcing = utility.prepare_plet_inputs(
        parcel_path, "parcels", WORKBOOK, nlcd_source=_raster)
    assert parcel_out.name == "site_plet.gpkg"
    assert forcing.name == "site_plet_huc12.gpkg"
    with sqlite3.connect(parcel_out) as con:
        assert con.execute("SELECT value FROM custom_overrides").fetchone() == ("untouched",)
        assert con.execute("SELECT value FROM input_rain_days").fetchone() == (111,)
        assert con.execute("SELECT count(*) FROM sqlite_master WHERE name='input_hsg'").fetchone() == (0,)
        assert con.execute("SELECT value FROM input_land_cover ORDER BY pid").fetchall() == [
            ("cropland",), ("forest",)]
        assert con.execute(
            "SELECT value FROM input_surface_concentration WHERE pid=1 AND pollutant='TSS'"
        ).fetchone() == (12.5,)
        assert con.execute("SELECT count(*) FROM input_curve_number").fetchone() == (16,)
        assert con.execute(
            "SELECT value,notes FROM input_curve_number "
            "WHERE land_cover='cropland' AND hsg='A'"
        ).fetchone() == (91, "existing user override")
        assert con.execute(
            "SELECT value,mean,sd,notes FROM input_infiltration_fraction "
            "WHERE land_cover='forest' AND hsg='B'"
        ).fetchone() == (None, .82, .03, "existing user override")
        assert con.execute("SELECT count(*) FROM parcel_huc12").fetchone() == (2,)
    with sqlite3.connect(forcing) as con:
        assert con.execute("SELECT count(*) FROM input_surface_concentration").fetchone() == (30,)
        assert con.execute("SELECT COUNT(*) FROM input_hsg").fetchone() == (2,)
        for table_name, expected in (("input_sediment_n_pct", .08),
                                     ("input_sediment_p_pct", .0308)):
            assert con.execute(
                f"SELECT huc12,value,units,notes FROM {table_name} ORDER BY huc12"
            ).fetchall() == [
                (huc, expected, "%", f"PLET sheet 4, {'SOIL_N_CONC' if expected == .08 else 'SOIL_P_CONC'}; "
                 "soil nutrient concentration in percent")
                for huc in (HUC_A, HUC_B)
            ]
        assert con.execute(
            "SELECT COUNT(*) FROM input_surface_concentration "
            "WHERE pollutant='TSS' AND value=0 AND notes LIKE '%PLACEHOLDER%'"
        ).fetchone() == (10,)
        assert con.execute("SELECT count(*) FROM input_runoff_day_fraction").fetchone() == (2,)
    class Logger:
        def verbose(self, *args, **kwargs):
            pass
        def warning(self, *args, **kwargs):
            pass
        def info(self, *args, **kwargs):
            pass

    logger = Logger()
    config = {"parcels": str(parcel_out), "plet_forcing": str(forcing)}
    layer = input_config._load_plet_huc12_layer(config, logger)
    huc_parameters = input_config._assemble_huc12_plet_parameter_source(config, layer, logger)
    parcel_parameters = input_config._assemble_parcel_parameter_source(config, logger)
    assignments = input_config._load_parcel_huc12(config, [1, 2], [1, 2], logger)
    merged = input_config._merge_huc12_plet_parameters(
        parcel_parameters, assignments, huc_parameters, [1, 2]
    )
    resolved = input_config._load_plet_parameter_table(
        merged.loc[~merged.parameter.isin(("r", "k", "ls", "c", "p",
                                           "sediment_n_pct", "sediment_p_pct"))],
        [1, 2], logger,
    )
    assert set(resolved.loc[resolved.parameter == "hsg", "value"]) == {"B", "C"}
    assert len(merged.loc[merged.parameter == "sediment_n_pct"]) == 2
    concentration_source = input_config._merge_huc12_plet_concentrations(
        config, input_config._assemble_plet_concentration_source(config, logger),
        assignments, resolved, [1, 2], logger,
    )
    assert concentration_source.loc[
        (concentration_source.pid == 1) & (concentration_source.pollutant == "TSS"),
        "value",
    ].tolist() == [12.5]
    concentrations = input_config._load_pollutant_concentrations(
        concentration_source, ["TN", "TP", "TSS"], logger, parcel_ids=[1, 2]
    )
    assert concentrations.loc[
        (concentrations.pid == 2) & (concentrations.pollutant == "TN")
        & (concentrations.pathway == "surface"), "units",
    ].tolist() == ["mg/L"]
    assert concentrations.loc[
        (concentrations.pid == 1) & (concentrations.pollutant == "TSS")
        & (concentrations.pathway == "surface"), "value",
    ].tolist() == [12.5]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        utility.prepare_plet_inputs(parcel_path, "parcels", WORKBOOK, nlcd_source=_raster)


def test_ssurgo_mode_writes_dominant_hsg(tmp_path, monkeypatch):
    parcel_path = tmp_path / "site.gpkg"
    _parcel_fixture(parcel_path)
    hucs = _hucs()
    monkeypatch.setattr(utility, "download_wbd_huc12", lambda bounds: hucs)
    monkeypatch.setattr(utility, "retain_hucs_by_parcel_overlap_threshold",
                        lambda polygons, parcels, threshold: polygons)
    monkeypatch.setattr(utility, "download_ssurgo", lambda domain, tile_degrees, timeout:
                        gpd.GeoDataFrame({"hydgrpdcd": ["A", "B"]}, geometry=hucs.geometry,
                                         crs=hucs.crs))
    out, forcing = utility.prepare_plet_inputs(
        parcel_path, "parcels", WORKBOOK, sssurgo_hsg_true=True, nlcd_source=_raster)
    with sqlite3.connect(out) as con:
        assert con.execute("SELECT value FROM input_hsg ORDER BY pid").fetchall() == [
            ("A",), ("B",)]
    with sqlite3.connect(forcing) as con:
        assert con.execute("SELECT COUNT(*) FROM input_hsg").fetchone() == (0,)


def test_custom_parcel_layer_becomes_canonical_with_ids_and_metadata(tmp_path, monkeypatch):
    package = tmp_path / "alternate.gpkg"
    source = gpd.GeoDataFrame(
        {"geometry": [box(-290, 10, -30, 290), box(30, 10, 290, 290)]},
        crs="EPSG:5070",
    )
    source.to_file(package, layer="alternate", driver="GPKG")
    with sqlite3.connect(package) as con:
        con.execute("ALTER TABLE alternate RENAME COLUMN fid TO pid")
        con.execute("CREATE TABLE custom_relation "
                    "(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES alternate(pid), value TEXT)")
        con.execute("INSERT INTO custom_relation(pid,value) VALUES (1,'linked')")
        con.execute("INSERT INTO gpkg_contents(table_name,data_type,identifier,description,srs_id) "
                    "VALUES ('custom_relation','attributes','custom_relation','related rows',NULL)")
    hucs = _hucs()
    monkeypatch.setattr(utility, "download_wbd_huc12", lambda bounds: hucs)
    monkeypatch.setattr(utility, "retain_hucs_by_parcel_overlap_threshold",
                        lambda polygons, parcels, threshold: polygons)
    out, _ = utility.prepare_plet_inputs(
        package, "alternate", WORKBOOK, nlcd_source=_raster)
    assert list(gpd.read_file(out, layer="parcels", fid_as_index=True).index) == [1, 2]
    assert list(gpd.read_file(package, layer="alternate", fid_as_index=True).index) == [1, 2]
    with sqlite3.connect(out) as con:
        assert con.execute(
            "SELECT table_name,identifier,data_type FROM gpkg_contents "
            "WHERE table_name='parcels'"
        ).fetchone() == ("parcels", "parcels", "features")
        assert con.execute("SELECT table_name FROM gpkg_geometry_columns "
                           "WHERE table_name='parcels'").fetchone() == ("parcels",)
        assert con.execute("SELECT name FROM sqlite_master "
                           "WHERE name LIKE 'rtree_parcels_%' AND type='table' "
                           "LIMIT 1").fetchone()
        assert con.execute("SELECT value FROM custom_relation WHERE pid=1").fetchone() == ("linked",)
        assert con.execute("SELECT sql FROM sqlite_master WHERE name='custom_relation' "
                           "AND type='table'").fetchone()[0].find("REFERENCES \"parcels\"") >= 0
        assert con.execute("PRAGMA foreign_key_check").fetchall() == []
        assert con.execute('SELECT pid FROM parcel_huc12 ORDER BY pid').fetchall() == [(1,), (2,)]
    with sqlite3.connect(package) as con:
        assert con.execute("SELECT table_name FROM gpkg_contents "
                           "WHERE data_type='features'").fetchone() == ("alternate",)


def test_noncanonical_layer_rejects_existing_canonical_table_before_download(tmp_path, monkeypatch):
    package = tmp_path / "conflict.gpkg"
    gpd.GeoDataFrame({"pid": [1], "geometry": [box(0, 0, 1, 1)]},
                     crs="EPSG:5070").to_file(package, layer="alternate", driver="GPKG")
    with sqlite3.connect(package) as con:
        con.execute("CREATE TABLE parcels (pid INTEGER PRIMARY KEY)")
    monkeypatch.setattr(utility, "download_wbd_huc12",
                        lambda bounds: pytest.fail("WBD must not be downloaded"))
    with pytest.raises(ValueError, match="already contains a parcels table"):
        utility.prepare_plet_inputs(package, "alternate", WORKBOOK, nlcd_source=_raster)
    assert not package.with_name("conflict_plet.gpkg").exists()


def test_noncanonical_layer_requires_pid_feature_key(tmp_path, monkeypatch):
    package = tmp_path / "wrong_key.gpkg"
    gpd.GeoDataFrame({"pid": [1], "geometry": [box(0, 0, 1, 1)]},
                     crs="EPSG:5070").to_file(package, layer="alternate", driver="GPKG")
    monkeypatch.setattr(utility, "download_wbd_huc12",
                        lambda bounds: pytest.fail("WBD must not be downloaded"))
    with pytest.raises(ValueError, match="pid INTEGER PRIMARY KEY"):
        utility.prepare_plet_inputs(package, "alternate", WORKBOOK, nlcd_source=_raster)
