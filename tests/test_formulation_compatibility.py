"""Approved formulations reject arbitrary, scientifically unreviewed mixes."""
from __future__ import annotations
import pytest
from basin_bmp_sim.science.formulations.formulation_registry import validate_formulation_selection
from basin_bmp_sim.bmp.practice_registry import resolve_practice, response_family


@pytest.mark.parametrize("mode,identifier", [
    ("statistical", "statistical_annual_v1"),
    ("plet_rusle", "plet_rusle_annual_v1"),
])
def test_legacy_modes_use_registered_formulations(mode, identifier):
    assert validate_formulation_selection({"load_generation":{"mode":mode}}).identifier == identifier


def test_explicit_valid_selection():
    form=validate_formulation_selection({
        "load_generation":{"mode":"plet_rusle"},
        "simulation":{"formulation":"plet_rusle_annual_v1",
                      "bmp_methods":{"wetland":"plet_pathway_efficiency"}},
    })
    assert form.baseline_mode == "plet_rusle"


@pytest.mark.parametrize("simulation", [
    {"formulation":"statistical_annual_v1"},
    {"formulation":"unknown"},
    {"formulation":"plet_rusle_annual_v1", "bmp_methods":{"wetland":"first_order_areal_retention"}},
    {"formulation":"plet_rusle_annual_v1", "bmp_methods":{"random":"plet_pathway_efficiency"}},
    {"formulation":"plet_rusle_annual_v1", "process_methods":{"runoff":"arbitrary"}},
])
def test_rejects_incompatible_or_unsupported_methods(simulation):
    with pytest.raises(ValueError):
        validate_formulation_selection({"load_generation":{"mode":"plet_rusle"},
                                        "simulation": simulation})


def test_practice_registry_preserves_existing_response_classes():
    assert response_family(656) == response_family(657) == "wetland"
    assert response_family(412) == "grassed"
    assert all(response_family(c) == "infield" for c in (340,329,590,999))
    assert resolve_practice(656).cps == 656
