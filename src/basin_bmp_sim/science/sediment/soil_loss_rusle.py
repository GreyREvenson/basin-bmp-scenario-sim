"""RUSLE annual soil loss and delivery calculation."""
from __future__ import annotations
from typing import Any, Mapping
from basin_bmp_sim.domain.constants import RUSLE_REQUIRED_INPUTS as _REQUIRED_RUSLE, TON_PER_ACRE_TO_KG_PER_HA
from basin_bmp_sim.config.loading import apply_plet_parameter_defaults
from basin_bmp_sim.config.validation import validate_parameter_value


def rusle_sediment_load_rate_kg_ha_yr(parameters: Mapping[str, Any]) -> float:
    """Estimate the annual areal sediment load rate from RUSLE inputs.

    Parameters
    ----------
    parameters : Mapping[str, Any]
        Mapping containing the required RUSLE factors and optional delivery
        modifiers.

    Returns
    -------
    float
        Annual sediment load rate in kilograms per hectare per year.
    """

    parameters = apply_plet_parameter_defaults(parameters)
    if not all(name in parameters for name in _REQUIRED_RUSLE):
        return 0.0
    gross_ton_ac = 1.0
    for name in _REQUIRED_RUSLE:
        gross_ton_ac *= validate_parameter_value(name, parameters[name])

    sdr = 1.0
    if "sdr" in parameters:
        sdr = validate_parameter_value("sdr", parameters["sdr"])

    sediment_multiplier = validate_parameter_value(
        "sediment_multiplier", parameters["sediment_multiplier"]
    )
    delivery_multiplier = validate_parameter_value(
        "sediment_delivery_multiplier", parameters["sediment_delivery_multiplier"]
    )
    return gross_ton_ac * sdr * TON_PER_ACRE_TO_KG_PER_HA * sediment_multiplier * delivery_multiplier

