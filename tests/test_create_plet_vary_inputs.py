"""Tests for distributional PLET GeoPackages created from fixed inputs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from src.input_validation import validate_stats_rows
from utils import create_plet_vary_inputs as utility


def _sql(path: Path, statement: str):
    with sqlite3.connect(path) as con:
        return con.execute(statement).fetchall()


def _write_input_table(
    package: Path, name: str, key: str, rows: list[dict],
) -> None:
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(1, len(frame) + 1))
    with sqlite3.connect(package) as con:
        frame.to_sql(name, con, index=False)


def _input_row(key: str, identifier: object, value: object, **extra) -> dict:
    return {
        key: identifier,
        "value": value,
        "mean": float("nan"),
        "sd": float("nan"),
        "min": float("nan"),
        "p05": float("nan"),
        "p10": float("nan"),
        "p25": float("nan"),
        "p50": float("nan"),
        "p75": float("nan"),
        "p90": float("nan"),
        "p95": float("nan"),
        "max": float("nan"),
        "sample_group": None,
        "units": None,
        "notes": "synthetic input",
        **extra,
    }


def _write_synthetic_package(path: Path, layer: str, key: str, identifier: object) -> None:
    gpd.GeoDataFrame(
        {key: [identifier], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:5070",
    ).to_file(path, layer=layer, driver="GPKG")


def test_synthetic_inputs_are_copied_and_use_valid_distributions(tmp_path: Path) -> None:
    parcel_source = tmp_path / "parcels_source.gpkg"
    huc_source = tmp_path / "hucs_source.gpkg"
    parcels = tmp_path / "parcels_plet_stochastic.gpkg"
    hucs = tmp_path / "huc12s_plet_stochastic.gpkg"
    _write_synthetic_package(parcel_source, "parcels", "pid", 1)
    _write_synthetic_package(huc_source, "huc12", "huc12", "123456789001")
    _write_input_table(
        parcel_source,
        "input_curve_number",
        "land_cover",
        [
            _input_row("land_cover", "cropland", 90.0, hsg="B"),
            _input_row("land_cover", "forest", 0.0, hsg="A"),
        ],
    )
    _write_input_table(
        parcel_source,
        "input_infiltration_fraction",
        "land_cover",
        [_input_row("land_cover", "cropland", 0.30, hsg="B")],
    )
    _write_input_table(
        parcel_source,
        "input_selection_weight",
        "pid",
        [_input_row("pid", 1, 1.0)],
    )
    _write_input_table(
        parcel_source,
        "input_land_cover",
        "pid",
        [_input_row("pid", 1, "cropland")],
    )
    _write_input_table(
        huc_source,
        "input_avg_rain_in",
        "huc12",
        [_input_row("huc12", "123456789001", 2.0)],
    )
    _write_input_table(
        huc_source,
        "input_rusle_r",
        "huc12",
        [
            _input_row("huc12", "123456789001", 150.0, land_cover=cover)
            for cover in ("cropland", "pastureland", "forest", "user_defined")
        ],
    )
    _write_input_table(
        huc_source,
        "input_surface_concentration",
        "huc12",
        [
            _input_row("huc12", "123456789001", 5.0, pollutant="TN"),
            _input_row("huc12", "123456789001", 0.0, pollutant="TP"),
        ],
    )
    _write_input_table(
        huc_source,
        "input_rusle_p",
        "huc12",
        [_input_row("huc12", "123456789001", 1.0, land_cover="cropland")],
    )

    summary = utility.create_vary_inputs(
        parcel_source=parcel_source,
        huc_source=huc_source,
        parcel_output=parcels,
        huc_output=hucs,
    )
    assert summary["parcels"]["input_curve_number"] == 1
    assert summary["parcels"]["input_infiltration_fraction"] == 1
    assert summary["huc12s"]["input_avg_rain_in"] == 1
    assert summary["huc12s"]["input_rusle_r"] == 4

    for src, dst, spatial in (
        (parcel_source, parcels, "parcels"),
        (huc_source, hucs, "huc12"),
    ):
        assert _sql(src, f'SELECT COUNT(*) FROM "{spatial}"') == _sql(dst, f'SELECT COUNT(*) FROM "{spatial}"')
        with sqlite3.connect(src) as con_src, sqlite3.connect(dst) as con_dst:
            tables_src = {x[0] for x in con_src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            tables_dst = {x[0] for x in con_dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert tables_src == tables_dst
            for name in sorted(t for t in tables_src if t.startswith("input_")):
                cols_src = [x[1] for x in con_src.execute(f'PRAGMA table_info("{name}")')]
                cols_dst = [x[1] for x in con_dst.execute(f'PRAGMA table_info("{name}")')]
                assert cols_src == cols_dst
                assert con_src.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone() == con_dst.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()
                data = pd.read_sql_query(f'SELECT * FROM "{name}"', con_dst)
                if data.empty or "value" not in data.columns or name in {"input_land_cover", "input_hsg"}:
                    continue
                # All output rows must respect the model's input schema.
                validate_stats_rows(data.drop(columns=["units"], errors="ignore"), name)
        assert _sql(dst, "PRAGMA quick_check") == [("ok",)]

    cn = _sql(parcels, "SELECT value,min,p10,p50,p90,max FROM input_curve_number WHERE land_cover='cropland' AND hsg='B'")[0]
    assert cn[0] is None
    assert cn[1] < cn[2] < cn[3] < cn[4] < cn[5] <= 100
    assert _sql(parcels, 'SELECT value FROM input_selection_weight')[0][0] == 1
    assert _sql(parcels, 'SELECT value FROM input_land_cover')[0][0] == "cropland"
    assert _sql(parcels, "SELECT value FROM input_curve_number WHERE land_cover='forest'")[0][0] == 0
    assert _sql(hucs, 'SELECT value FROM input_surface_concentration WHERE pollutant="TP"')[0][0] == 0
    assert _sql(hucs, 'SELECT value FROM input_rusle_p')[0][0] == 1

    groups = _sql(hucs, "SELECT huc12,COUNT(DISTINCT sample_group) FROM input_rusle_r GROUP BY huc12")
    assert groups == [("123456789001", 1)]

    # Distinct outcomes are produced by the actual shared sampler.
    from src import sampling
    import numpy as np
    class SampleCtx:
        rng = np.random.default_rng(7)
        _piecewise_quantile_sample = sampling._piecewise_quantile_sample
        _trunc_normal = sampling._trunc_normal
    stat = _sql(hucs, "SELECT min,p10,p50,p90,max FROM input_avg_rain_in LIMIT 1")[0]
    ctx = SampleCtx()
    values = [sampling._sample_from_stats(ctx, dict(zip(("min", "p10", "p50", "p90", "max"), stat)), kind="nonnegative") for _ in range(40)]
    assert len(set(round(value, 9) for value in values)) > 20


def test_existing_output_is_not_overwritten_without_flag(tmp_path: Path) -> None:
    parcels = tmp_path / "parcels_plet_stochastic.gpkg"
    hucs = tmp_path / "huc12s_plet_stochastic.gpkg"
    parcel_source = tmp_path / "parcels_source.gpkg"
    huc_source = tmp_path / "hucs_source.gpkg"
    _write_synthetic_package(parcel_source, "parcels", "pid", 1)
    _write_synthetic_package(huc_source, "huc12", "huc12", "123456789001")
    _write_input_table(
        parcel_source, "input_curve_number", "land_cover",
        [_input_row("land_cover", "cropland", 90.0, hsg="B")],
    )
    _write_input_table(
        huc_source, "input_avg_rain_in", "huc12",
        [_input_row("huc12", "123456789001", 2.0)],
    )
    utility.create_vary_inputs(
        parcel_source, huc_source, parcels, hucs,
    )
    before = parcels.read_bytes()
    with pytest.raises(FileExistsError, match="Output exists"):
        utility.create_vary_inputs(
            parcel_source, huc_source, parcels, hucs,
        )
    assert parcels.read_bytes() == before
    again = utility.create_vary_inputs(
        parcel_source, huc_source, parcels, hucs, overwrite=True,
    )
    assert again["huc12s"]["input_avg_rain_in"] == 1
    assert _sql(parcels, "PRAGMA quick_check") == [("ok",)]


def test_refuses_to_replace_source(tmp_path: Path) -> None:
    parcel_source = tmp_path / "parcels_source.gpkg"
    huc_source = tmp_path / "hucs_source.gpkg"
    _write_synthetic_package(parcel_source, "parcels", "pid", 1)
    _write_synthetic_package(huc_source, "huc12", "huc12", "123456789001")
    with pytest.raises(ValueError, match="four distinct"):
        utility.create_vary_inputs(
            parcel_source, huc_source, parcel_source, tmp_path / "test.gpkg",
        )
