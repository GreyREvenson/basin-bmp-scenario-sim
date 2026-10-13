"""Tests for distributional PLET GeoPackages created from fixed inputs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.input_validation import validate_stats_rows
from utils import create_plet_vary_inputs as utility


def _sql(path: Path, statement: str):
    with sqlite3.connect(path) as con:
        return con.execute(statement).fetchall()


def test_real_east_fork_copies_preserve_features_and_use_valid_distributions(tmp_path: Path) -> None:
    parcels = tmp_path / "parcels_plet_stochastic.gpkg"
    hucs = tmp_path / "huc12s_plet_stochastic.gpkg"
    summary = utility.create_vary_inputs(parcel_output=parcels, huc_output=hucs)
    assert summary["parcels"]["input_curve_number"] == 20
    assert summary["parcels"]["input_infiltration_fraction"] == 20
    assert summary["huc12s"]["input_avg_rain_in"] == 18
    assert summary["huc12s"]["input_rusle_r"] == 72

    for src, dst, spatial in (
        (utility.DEFAULT_PARCELS, parcels, "parcels"),
        (utility.DEFAULT_HUCS, hucs, "huc12"),
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
    assert _sql(parcels, 'SELECT COUNT(*) FROM input_selection_weight WHERE value IS NOT NULL')[0][0] == 2681
    assert _sql(parcels, 'SELECT COUNT(*) FROM input_land_cover WHERE value IS NOT NULL')[0][0] == 2681
    assert _sql(hucs, 'SELECT COUNT(*) FROM input_subsurface_concentration WHERE value=0')[0][0] == 180
    assert _sql(hucs, 'SELECT COUNT(*) FROM input_surface_concentration WHERE value=0')[0][0] == 126
    assert _sql(hucs, 'SELECT COUNT(*) FROM input_rusle_p WHERE value=1')[0][0] == 37
    assert _sql(hucs, 'SELECT COUNT(*) FROM input_annual_precip_in WHERE value IS NOT NULL')[0][0] == 18

    groups = _sql(hucs, "SELECT huc12,COUNT(DISTINCT sample_group) FROM input_rusle_r GROUP BY huc12")
    assert len(groups) == 18
    assert all(n == 1 for _, n in groups)

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
    utility.create_vary_inputs(parcel_output=parcels, huc_output=hucs)
    before = parcels.read_bytes()
    with pytest.raises(FileExistsError, match="Output exists"):
        utility.create_vary_inputs(parcel_output=parcels, huc_output=hucs)
    assert parcels.read_bytes() == before
    again = utility.create_vary_inputs(parcel_output=parcels, huc_output=hucs, overwrite=True)
    assert again["huc12s"]["input_avg_rain_in"] == 18
    assert _sql(parcels, "PRAGMA quick_check") == [("ok",)]


def test_refuses_to_replace_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="four distinct"):
        utility.create_vary_inputs(parcel_output=utility.DEFAULT_PARCELS, huc_output=tmp_path / "test.gpkg")
