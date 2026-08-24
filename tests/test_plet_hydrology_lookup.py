from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.io_utils import _load_plet_parameter_table
from src.load_generation import (
    PLET_HYDROLOGY_LOOKUP_PATH,
    plet_hydrology_from_classifications,
    validate_plet_input_table,
)


PLET_REFERENCE_VALUES = {
    ("urban", "A"): (83.0, 0.360),
    ("urban", "B"): (89.0, 0.240),
    ("urban", "C"): (92.0, 0.120),
    ("urban", "D"): (93.0, 0.060),
    ("cropland", "A"): (67.0, 0.450),
    ("cropland", "B"): (78.0, 0.300),
    ("cropland", "C"): (85.0, 0.150),
    ("cropland", "D"): (89.0, 0.075),
    ("pastureland", "A"): (49.0, 0.450),
    ("pastureland", "B"): (69.0, 0.300),
    ("pastureland", "C"): (79.0, 0.150),
    ("pastureland", "D"): (84.0, 0.075),
    ("forest", "A"): (39.0, 0.450),
    ("forest", "B"): (60.0, 0.300),
    ("forest", "C"): (73.0, 0.150),
    ("forest", "D"): (79.0, 0.075),
    ("user_defined", "A"): (50.0, 0.450),
    ("user_defined", "B"): (70.0, 0.300),
    ("user_defined", "C"): (80.0, 0.150),
    ("user_defined", "D"): (85.0, 0.075),
}


@pytest.mark.parametrize(
    ("classification", "expected"),
    PLET_REFERENCE_VALUES.items(),
)
def test_lookup_matches_plet_v2_reference_tables(
    classification: tuple[str, str],
    expected: tuple[float, float],
) -> None:
    land_cover, hsg = classification

    resolved = plet_hydrology_from_classifications(land_cover, hsg)

    assert resolved["cn"] == pytest.approx(expected[0])
    assert resolved["infiltration_fraction"] == pytest.approx(expected[1])


def test_bundled_lookup_contains_exactly_the_plet_reference_rows() -> None:
    table = pd.read_csv(PLET_HYDROLOGY_LOOKUP_PATH)
    observed = {
        (str(row.land_cover), str(row.hsg)): (
            float(row.curve_number),
            float(row.infiltration_fraction),
        )
        for row in table.itertuples(index=False)
    }

    assert observed == PLET_REFERENCE_VALUES


def test_plet_table_requires_land_cover_and_hsg_for_each_parcel() -> None:
    table = pd.DataFrame(
        {
            "pid": ["*", "*"],
            "parameter": ["annual_precip_in", "land_cover"],
            "value": [42.0, "cropland"],
        }
    )

    with pytest.raises(ValueError, match="missing required classifications.*hsg"):
        validate_plet_input_table(table, ["p1"])


@pytest.mark.parametrize("parameter", ["cn", "curve_number", "infiltration_fraction"])
def test_plet_table_rejects_user_supplied_lookup_outputs(parameter: str) -> None:
    table = pd.DataFrame(
        {
            "pid": ["*", "*", "*"],
            "parameter": ["land_cover", "hsg", parameter],
            "value": ["cropland", "B", 78.0],
        }
    )

    with pytest.raises(ValueError, match="may not specify"):
        validate_plet_input_table(table, ["p1"])


def test_plet_parameter_loader_accepts_string_classifications(tmp_path) -> None:
    input_path = tmp_path / "plet_inputs.csv"
    pd.DataFrame(
        {
            "pid": ["*", "*", "*"],
            "parameter": ["annual_precip_in", "land use", "hsg"],
            "value": [42.0, "Pasture", "c"],
            "units": ["in/year", "classification", "classification"],
        }
    ).to_csv(input_path, index=False)

    loaded = _load_plet_parameter_table(
        input_path,
        ["p1", "p2"],
        logging.getLogger("test-plet-loader"),
    )

    assert loaded is not None
    values = dict(zip(loaded["parameter"], loaded["value"]))
    assert values["land_cover"] == "pastureland"
    assert values["hsg"] == "C"


def test_feedlot_is_rejected_because_plet_requires_percent_paved() -> None:
    with pytest.raises(ValueError, match="Unsupported PLET land_cover"):
        plet_hydrology_from_classifications("feedlot", "B")
