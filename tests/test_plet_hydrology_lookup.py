from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.input_config import _load_plet_hydrology_lookup, _load_plet_parameter_table
from src.plet_rusle import plet_hydrology_from_classifications
from src.input_validation import validate_plet_input_table

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


def _write_reference_lookup(path) -> None:
    rows = []
    for (land_cover, hsg), (cn, infiltration_fraction) in PLET_REFERENCE_VALUES.items():
        rows.append({"land_cover": land_cover, "hsg": hsg, "parameter": "cn", "value": cn})
        rows.append(
            {
                "land_cover": land_cover,
                "hsg": hsg,
                "parameter": "infiltration_fraction",
                "value": infiltration_fraction,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize(("classification", "expected"), PLET_REFERENCE_VALUES.items())
def test_lookup_matches_supplied_reference_values(
    classification: tuple[str, str], expected: tuple[float, float], tmp_path
) -> None:
    lookup_path = tmp_path / "hydrology.csv"
    _write_reference_lookup(lookup_path)
    land_cover, hsg = classification
    resolved = plet_hydrology_from_classifications(land_cover, hsg, lookup_path=lookup_path)
    assert resolved["cn"] == pytest.approx(expected[0])
    assert resolved["infiltration_fraction"] == pytest.approx(expected[1])



def test_hydrology_helper_requires_explicit_user_lookup() -> None:
    with pytest.raises(TypeError):
        plet_hydrology_from_classifications("cropland", "B")


def test_hydrology_helper_uses_user_supplied_values(tmp_path) -> None:
    lookup_path = tmp_path / "custom_hydrology.csv"
    pd.DataFrame(
        [
            {"land_cover": "cropland", "hsg": "B", "parameter": "cn", "value": 74.5},
            {
                "land_cover": "cropland",
                "hsg": "B",
                "parameter": "infiltration_fraction",
                "value": 0.41,
            },
        ]
    ).to_csv(lookup_path, index=False)

    resolved = plet_hydrology_from_classifications(
        "cropland", "B", lookup_path=lookup_path
    )

    assert resolved["cn"] == pytest.approx(74.5)
    assert resolved["infiltration_fraction"] == pytest.approx(0.41)

def test_plet_table_requires_land_cover_and_hsg_for_each_parcel() -> None:
    table = pd.DataFrame({"pid": ["*", "*"], "parameter": ["annual_precip_in", "land_cover"], "value": [42.0, "cropland"]})
    with pytest.raises(ValueError, match="missing required classifications.*hsg"):
        validate_plet_input_table(table, ["p1"])


@pytest.mark.parametrize("parameter", ["cn", "curve_number", "infiltration_fraction"])
def test_plet_table_rejects_hydrology_parameters_in_parcel_table(parameter: str) -> None:
    table = pd.DataFrame({"pid": ["*", "*", "*"], "parameter": ["land_cover", "hsg", parameter], "value": ["cropland", "B", 78.0]})
    with pytest.raises(ValueError, match="input_curve_number"):
        validate_plet_input_table(table, ["p1"])


def test_plet_parameter_loader_accepts_string_classifications(tmp_path) -> None:
    input_path = tmp_path / "plet_inputs.csv"
    pd.DataFrame({
        "pid": ["*", "*", "*"],
        "parameter": ["annual_precip_in", "land use", "hsg"],
        "value": [42.0, "Pasture", "c"],
        "units": ["in/year", "classification", "classification"],
    }).to_csv(input_path, index=False)
    loaded = _load_plet_parameter_table(input_path, ["p1", "p2"], logging.getLogger("test-plet-loader"))
    assert loaded is not None
    values = dict(zip(loaded["parameter"], loaded["value"]))
    assert values["land_cover"] == "pastureland"
    assert values["hsg"] == "C"


def test_hydrology_lookup_requires_both_parameters_for_every_pair(tmp_path) -> None:
    path = tmp_path / "hydrology.csv"
    _write_reference_lookup(path)
    table = pd.read_csv(path).iloc[:-1]
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must define cn and infiltration_fraction"):
        _load_plet_hydrology_lookup(path, logging.getLogger("hydrology"))


def test_feedlot_is_rejected_because_plet_requires_percent_paved(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported PLET land_cover"):
        plet_hydrology_from_classifications("feedlot", "B", lookup_path=tmp_path / "unused.csv")
