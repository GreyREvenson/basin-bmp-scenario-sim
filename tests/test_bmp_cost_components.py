from pathlib import Path

import pandas as pd
import pytest

from src.constants import CFG_BMP_COST, COL_COST_COMPONENT
from src.input_config import _load_bmp_cost


class DummyLogger:
    def verbose(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def test_load_bmp_cost_allows_multiple_named_components_per_cps(tmp_path: Path):
    path = tmp_path / "bmp_cost.csv"
    pd.DataFrame(
        [
            {"cps": 340, "cost_component": "seed", "unit": "USD/ha", "value": 50.0},
            {"cps": 340, "cost_component": "planting", "unit": "USD/ha", "value": 40.0},
        ]
    ).to_csv(path, index=False)

    loaded = _load_bmp_cost({CFG_BMP_COST: str(path)}, [340], DummyLogger())

    assert loaded is not None
    assert loaded[COL_COST_COMPONENT].tolist() == ["seed", "planting"]


def test_load_bmp_cost_preserves_legacy_single_row_files(tmp_path: Path):
    path = tmp_path / "bmp_cost.csv"
    pd.DataFrame([{"cps": 340, "unit": "USD/ha", "value": 100.0}]).to_csv(path, index=False)

    loaded = _load_bmp_cost({CFG_BMP_COST: str(path)}, [340], DummyLogger())

    assert loaded is not None
    assert loaded[COL_COST_COMPONENT].tolist() == ["total"]


def test_load_bmp_cost_rejects_duplicate_component_key(tmp_path: Path):
    path = tmp_path / "bmp_cost.csv"
    pd.DataFrame(
        [
            {"cps": 340, "cost_component": "seed", "unit": "USD/ha", "value": 50.0},
            {"cps": 340, "cost_component": "seed", "unit": "USD/ha", "value": 60.0},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="conflicting duplicate rows"):
        _load_bmp_cost({CFG_BMP_COST: str(path)}, [340], DummyLogger())


def test_load_bmp_cost_rejects_same_component_with_different_units(tmp_path: Path):
    path = tmp_path / "bmp_cost.csv"
    pd.DataFrame(
        [
            {"cps": 340, "cost_component": "seed", "unit": "USD/ha", "value": 50.0},
            {"cps": 340, "cost_component": "seed", "unit": "USD/project", "value": 60.0},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="conflicting duplicate rows"):
        _load_bmp_cost({CFG_BMP_COST: str(path)}, [340], DummyLogger())
