"""PLET concentration times water volume nutrient-loading equations."""
from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
from src.constants import (RUSLE_REQUIRED_INPUTS as _REQUIRED_RUSLE,
    PLET_REQUIRED_RESOLVED_INPUTS as _REQUIRED_RESOLVED_PLET,
    PLET_PATHWAY_VALUES as PLET_PATHWAY_NAMES,
    INCH_OVER_HA_TO_LITERS)
from src.input_config import apply_plet_parameter_defaults
from src.input_validation import NONNEGATIVE_DOMAIN, validate_parameter_value, validate_scalar_in_domain
from src.models.hydrology.runoff_plet_curve_number import plet_annual_surface_runoff_in
from src.models.hydrology.infiltration_plet_balance import plet_annual_infiltration_in
from src.models.sediment.soil_loss_rusle import rusle_sediment_load_rate_kg_ha_yr


def calculate_load_diagnostics(parameters: Mapping[str, Any]) -> Dict[str, float]:
    """Calculate intermediate load-generation diagnostics.

    Parameters
    ----------
    parameters : Mapping[str, Any]
        Parameter mapping used for runoff, infiltration, and sediment
        calculations.

    Returns
    -------
    dict[str, float]
        Diagnostic values useful for reporting and debugging.
    """

    event_rainfall, event_runoff, annual_storm_runoff, annual_runoff = plet_annual_surface_runoff_in(parameters)
    has_rusle = all(name in parameters for name in _REQUIRED_RUSLE)
    sediment_load_rate_kg_ha_yr = rusle_sediment_load_rate_kg_ha_yr(parameters) if has_rusle else 0.0
    return {
        "event_rainfall_in": float(event_rainfall),
        "event_runoff_in": float(event_runoff),
        "annual_storm_runoff_in": float(annual_storm_runoff),
        "annual_runoff_in": float(annual_runoff),
        "annual_infiltration_in": float(plet_annual_infiltration_in(parameters)),
        "sediment_load_rate_kg_ha_yr": float(sediment_load_rate_kg_ha_yr),
    }


def calculate_plet_pathway_load_rates(
    parameters: Mapping[str, Any],
    concentrations: Mapping[str, float],
    groundwater_concentrations: Optional[Mapping[str, float]],
    pollutants: Sequence[str],
) -> np.ndarray:
    """Calculate the two PLET/RUSLE pathways: surface and subsurface.

        Surface nutrient load is runoff-derived load plus any RUSLE sediment-bound
        nutrient contribution. Subsurface nutrient load is groundwater concentration
        multiplied by PLET annual infiltration volume. The infiltration fraction in
        ``parameters`` is resolved from the PLET land-cover/HSG lookup table before
        this function is called. TSS has no subsurface component.

        Parameters
        ----------
        parameters : Mapping[str, Any]
            Model parameter values keyed by canonical parameter name.
        concentrations : Mapping[str, float]
            Pollutant concentrations keyed by pollutant.
        groundwater_concentrations : Optional[Mapping[str, float]]
            Groundwater pollutant concentrations or concentration table, if configured.
        pollutants : Sequence[str]
            Pollutant names in model order.

        Returns
        -------
        np.ndarray
            Areal pollutant load rates for the surface and subsurface pathways.

        Raises
        ------
        ValueError
            If required resolved PLET parameters are missing.
        
    """
    parameters = apply_plet_parameter_defaults(parameters, pollutants)
    missing = [name for name in _REQUIRED_RESOLVED_PLET if name not in parameters]
    if missing:
        raise ValueError(f"PLET inputs are missing required parameters: {missing}")

    _, _, _, annual_runoff_in = plet_annual_surface_runoff_in(parameters)
    runoff_l_ha = annual_runoff_in * INCH_OVER_HA_TO_LITERS
    infiltration_l_ha = plet_annual_infiltration_in(parameters) * INCH_OVER_HA_TO_LITERS
    has_rusle = all(name in parameters for name in _REQUIRED_RUSLE)
    sediment_load_rate_kg_ha_yr = rusle_sediment_load_rate_kg_ha_yr(parameters) if has_rusle else 0.0
    enrichment_ratio = validate_parameter_value(
        "enrichment_ratio", parameters["enrichment_ratio"]
    )
    groundwater_concentrations = groundwater_concentrations or {}

    pathway_load_rates = np.zeros((len(pollutants), len(PLET_PATHWAY_NAMES)), dtype=float)
    for idx, pollutant in enumerate(pollutants):
        pol = str(pollutant).upper()
        runoff_areal_load_rate = 0.0
        if pol != "TSS" or not has_rusle:
            runoff_concentration = validate_scalar_in_domain(
                concentrations[pol], NONNEGATIVE_DOMAIN, f"{pol} runoff concentration"
            )
            runoff_areal_load_rate = runoff_concentration * runoff_l_ha / 1_000_000.0
        subsurface_areal_load_rate = 0.0
        if pol != "TSS":
            groundwater_concentration = validate_scalar_in_domain(
                groundwater_concentrations[pol],
                NONNEGATIVE_DOMAIN,
                f"{pol} groundwater concentration",
            )
            subsurface_areal_load_rate = (
                groundwater_concentration * infiltration_l_ha / 1_000_000.0
            )

        if pol == "TSS":
            surface_areal_load_rate = sediment_load_rate_kg_ha_yr if has_rusle else runoff_areal_load_rate
        elif pol == "TN":
            sediment_fraction = validate_parameter_value(
                "sediment_n_pct", parameters["sediment_n_pct"]
            ) / 100.0
            surface_areal_load_rate = runoff_areal_load_rate + sediment_load_rate_kg_ha_yr * sediment_fraction * enrichment_ratio
        elif pol == "TP":
            sediment_fraction = validate_parameter_value(
                "sediment_p_pct", parameters["sediment_p_pct"]
            ) / 100.0
            surface_areal_load_rate = runoff_areal_load_rate + sediment_load_rate_kg_ha_yr * sediment_fraction * enrichment_ratio
        else:
            surface_areal_load_rate = runoff_areal_load_rate

        multiplier_name = f"load_multiplier_{pol.lower()}"
        multiplier = validate_parameter_value(
            multiplier_name, parameters[multiplier_name]
        )
        pathway_load_rates[idx, 0] = surface_areal_load_rate * multiplier
        pathway_load_rates[idx, 1] = subsurface_areal_load_rate * multiplier
    return pathway_load_rates

