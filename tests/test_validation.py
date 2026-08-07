from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.bmp import _get_bmp_selection_probs
from src.constants import CFG_PARCEL_P, DATA_CPS, DATA_PARCELS
from src.io_utils import _load_parcel_selection
from src.model import Model


class DummyLogger:
    def log(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def verbose(self, *args, **kwargs):
        return None


def test_load_parcel_selection_rejects_empty_parcels() -> None:
    with pytest.raises(ValueError, match="No parcels available"):
        _load_parcel_selection({}, pd.DataFrame(columns=["pid"]), DummyLogger())


def test_load_parcel_selection_rejects_duplicate_selection_rows(tmp_path) -> None:
    parcel_p = tmp_path / "parcel_p.csv"
    pd.DataFrame({"pid": ["p1", "p1"], "probability": [0.4, 0.6]}).to_csv(parcel_p, index=False)

    cfg = {CFG_PARCEL_P: str(parcel_p)}
    parcels = pd.DataFrame({"pid": ["p1", "p2"]})

    with pytest.raises(ValueError, match="must contain one row per parcel"):
        _load_parcel_selection(cfg, parcels, DummyLogger())


def test_get_bmp_selection_probs_rejects_invalid_probabilities(tmp_path) -> None:
    bmp_sel = tmp_path / "bmp_sel.csv"
    pd.DataFrame({"cps": [329, 412], "probability": [0.8, -0.2]}).to_csv(bmp_sel, index=False)

    model = SimpleNamespace(
        data={DATA_CPS: [329, 412]},
        cfg={},
        logger=DummyLogger(),
    )

    with pytest.raises(ValueError, match="nonnegative"):
        _get_bmp_selection_probs(model, str(bmp_sel))


def test_get_bmp_selection_probs_rejects_missing_cps_rows(tmp_path) -> None:
    bmp_sel = tmp_path / "bmp_sel.csv"
    pd.DataFrame({"cps": [329], "probability": [1.0]}).to_csv(bmp_sel, index=False)

    model = SimpleNamespace(
        data={DATA_CPS: [329, 412]},
        cfg={},
        logger=DummyLogger(),
    )

    with pytest.raises(ValueError, match="missing probability rows"):
        _get_bmp_selection_probs(model, str(bmp_sel))


def test_prepare_lookup_tables_rejects_duplicate_parcel_ids() -> None:
    model = Model.__new__(Model)
    model.data = {
        DATA_PARCELS: pd.DataFrame(
            {
                "pid": ["p1", "p1"],
                "area_ha": [1.0, 2.0],
                "perim_m": [10.0, 20.0],
            }
        )
    }
    model.logger = DummyLogger()

    with pytest.raises(ValueError, match="Duplicate parcel IDs"):
        Model._prepare_lookup_tables(model)
