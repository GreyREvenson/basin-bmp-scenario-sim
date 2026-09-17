from __future__ import annotations

import pandas as pd
import pytest

from src.constants import (
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    COL_OID,
    COL_PID,
    COL_PID_UP,
    COL_PROBABILITY,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
)
import src.input_config as input_config
from src.input_config import (
    _build_parcel_up_map,
    _complete_delivery_ratio_defaults,
)


class RecordingLogger:
    def warning(self, message: str) -> None:
        pass

    def verbose(self, message: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass


def test_selection_weight_table_is_normalized(monkeypatch) -> None:
    parcels = pd.DataFrame({COL_PID: ["A", "B", "C"]})
    weights = pd.DataFrame({COL_PID: ["A", "B", "C"], "value": [1.0, 2.0, 1.0]})
    monkeypatch.setattr(input_config, "_load_fixed_numeric_variable_table", lambda *args, **kwargs: weights)
    loaded = input_config._load_parcel_selection({}, parcels, RecordingLogger())
    assert loaded[COL_PID].tolist() == ["A", "B", "C"]
    assert loaded[COL_PROBABILITY].tolist() == pytest.approx([0.25, 0.50, 0.25])


def test_missing_selection_weight_means_equal_weights(monkeypatch) -> None:
    parcels = pd.DataFrame({COL_PID: ["A", "B", "C"]})
    monkeypatch.setattr(input_config, "_load_fixed_numeric_variable_table", lambda *args, **kwargs: None)
    loaded = input_config._load_parcel_selection({}, parcels, RecordingLogger())
    assert loaded[COL_PROBABILITY].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_parcel_up_uses_one_edge_per_row() -> None:
    edges = pd.DataFrame(
        [
            {COL_PID: "B", COL_PID_UP: "A"},
            {COL_PID: "C", COL_PID_UP: "A"},
            {COL_PID: "C", COL_PID_UP: "B"},
        ]
    )
    assert _build_parcel_up_map(edges, ["A", "B", "C", "D"]) == {
        "A": [],
        "B": ["A"],
        "C": ["A", "B"],
        "D": [],
    }


def test_parcel_up_rejects_legacy_compound_relationships() -> None:
    edges = pd.DataFrame([{COL_PID: "C", COL_PID_UP: "A,B"}])
    with pytest.raises(ValueError, match="one-edge-per-row"):
        _build_parcel_up_map(edges, ["A", "B", "C"])


def test_missing_delivery_ratio_columns_receive_neutral_defaults() -> None:
    out = _complete_delivery_ratio_defaults(
        None,
        {"A": ["1"], "B": ["1", "2"]},
    )
    got = {
        (str(row[COL_PID]), str(row[COL_OID])): (
            float(row[COL_SDR_F_TO_S]),
            float(row[COL_SDR_S_TO_O]),
            float(row[COL_NDR_F_TO_S]),
            float(row[COL_NDR_S_TO_O]),
        )
        for _, row in out.iterrows()
    }
    assert got == {
        ("A", "1"): (1.0, 1.0, 1.0, 1.0),
        ("B", "1"): (1.0, 1.0, 1.0, 1.0),
        ("B", "2"): (1.0, 1.0, 1.0, 1.0),
    }
