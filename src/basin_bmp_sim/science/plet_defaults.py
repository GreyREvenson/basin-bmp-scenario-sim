"""Science-owned default values for the annual PLET/RUSLE equations."""
from __future__ import annotations
from typing import Any, Dict, Mapping, Sequence


def _plet_parameter_defaults(pollutants: Sequence[str]) -> Dict[str, float]:
    """Return the centralized PLET/RUSLE parameter defaults.

    Parameters
    ----------
    pollutants : Sequence[str]
        Pollutant names in model order.

    Returns
    -------
    Dict[str, float]
        Centralized default PLET/RUSLE parameter values.
    """
    defaults: Dict[str, float] = {
        # Required PLET inputs are intentionally excluded. Missing required
        # variables must fail validation rather than being synthesized.
        "ia_ratio": 0.0,
        "runoff_multiplier": 1.0,
        "groundwater_multiplier": 1.0,
        "sediment_multiplier": 1.0,
        "sediment_delivery_multiplier": 1.0,
        "enrichment_ratio": 2.0,
        "sediment_n_pct": 0.0,
        "sediment_p_pct": 0.0,
    }
    for pollutant in pollutants:
        defaults[f"load_multiplier_{str(pollutant).lower()}"] = 1.0
    return defaults


def apply_plet_parameter_defaults(
    parameters: Mapping[str, Any],
    pollutants: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return a parameter mapping with centralized PLET/RUSLE defaults applied.

    Parameters
    ----------
    parameters : Mapping[str, Any]
        Model parameter values keyed by canonical parameter name.
    pollutants : Sequence[str]
        Pollutant names in model order.

    Returns
    -------
    Dict[str, Any]
        Parameter mapping with centralized defaults applied.
    """
    out = dict(parameters)
    for parameter, value in _plet_parameter_defaults(pollutants).items():
        if parameter not in out or out[parameter] is None:
            out[parameter] = value
    return out
