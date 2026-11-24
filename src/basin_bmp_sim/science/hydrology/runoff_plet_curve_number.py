"""PLET curve-number annual runoff and rainfall-correction equations."""
from __future__ import annotations
from typing import Any, Mapping, Tuple
from basin_bmp_sim.science.plet_defaults import apply_plet_parameter_defaults
from basin_bmp_sim.config.validation import NONNEGATIVE_DOMAIN, validate_parameter_value, validate_scalar_in_domain


def _plet_event_runoff_depth_in(
    event_rainfall_in: float,
    runoff_days: float,
    cn: float,
    ia_ratio: float = 0.0,
) -> Tuple[float, float, float]:
    """Calculate event and annual runoff from an event rainfall depth."""
    event_rainfall = validate_parameter_value("avg_rain_in", event_rainfall_in)
    runoff_days = validate_scalar_in_domain(
        runoff_days, NONNEGATIVE_DOMAIN, "corrected runoff days"
    )
    cn = validate_parameter_value("cn", cn)
    ia_ratio = validate_parameter_value("ia_ratio", ia_ratio)

    if runoff_days <= 0.0:
        return event_rainfall, 0.0, 0.0

    retention = (1000.0 / cn) - 10.0
    # CN is validated in (0, 100]; max() only guards floating-point roundoff
    # in this derived state and never repairs a user input.
    retention = max(0.0, retention)
    initial_abstraction = ia_ratio * retention
    if event_rainfall <= initial_abstraction:
        event_runoff = 0.0
    else:
        numerator = (event_rainfall - initial_abstraction) ** 2
        denominator = event_rainfall - initial_abstraction + retention
        event_runoff = numerator / denominator if denominator > 0.0 else 0.0

    return event_rainfall, event_runoff, event_runoff * runoff_days


def plet_runoff_depth_in(
    annual_precip_in: float,
    rain_days: float,
    rain_correction_fraction: float,
    runoff_day_fraction: float,
    cn: float,
    ia_ratio: float = 0.0,
) -> Tuple[float, float, float]:
    """Calculate PLET runoff using annual rainfall and correction factors.

    This is the original PLET forcing form. ``plet_annual_surface_runoff_in``
    also accepts ``avg_rain_in`` directly, matching the ``AVG_RAIN`` field
    supplied by the PLET Input Data Server.
    """
    annual_precip_in = validate_parameter_value("annual_precip_in", annual_precip_in)
    rain_days = validate_parameter_value("rain_days", rain_days)
    rain_correction_fraction = validate_parameter_value(
        "rain_correction_fraction", rain_correction_fraction
    )
    runoff_day_fraction = validate_parameter_value(
        "runoff_day_fraction", runoff_day_fraction
    )
    runoff_days = rain_days * runoff_day_fraction
    if runoff_days <= 0.0:
        return 0.0, 0.0, 0.0
    event_rainfall = annual_precip_in * rain_correction_fraction / runoff_days
    return _plet_event_runoff_depth_in(event_rainfall, runoff_days, cn, ia_ratio)


def plet_avg_rain_runoff_depth_in(
    avg_rain_in: float,
    rain_days: float,
    runoff_day_fraction: float,
    cn: float,
    ia_ratio: float = 0.0,
) -> Tuple[float, float, float]:
    """Calculate PLET runoff using Input Data Server ``AVG_RAIN`` directly.

    ``AVG_RAIN`` is the representative runoff-producing event rainfall depth
    (PLET Equation 3). Annual runoff still uses the corrected number of rain
    days, ``rain_days * runoff_day_fraction`` (PLET Equation 5).
    """
    avg_rain_in = validate_parameter_value("avg_rain_in", avg_rain_in)
    rain_days = validate_parameter_value("rain_days", rain_days)
    runoff_day_fraction = validate_parameter_value(
        "runoff_day_fraction", runoff_day_fraction
    )
    runoff_days = rain_days * runoff_day_fraction
    return _plet_event_runoff_depth_in(avg_rain_in, runoff_days, cn, ia_ratio)


def _plet_corrected_annual_precip_in(parameters: Mapping[str, Any]) -> float:
    """Return precipitation represented by runoff-producing events.

    When ``avg_rain_in`` is supplied it is authoritative and the corrected
    annual precipitation is reconstructed as AVG_RAIN times corrected rain
    days. Otherwise the original annual-rainfall/correction-factor form is
    used.
    """
    if parameters.get("avg_rain_in") is not None:
        rain_days = validate_parameter_value("rain_days", parameters["rain_days"])
        runoff_day_fraction = validate_parameter_value(
            "runoff_day_fraction", parameters["runoff_day_fraction"]
        )
        runoff_days = rain_days * runoff_day_fraction
        avg_rain = validate_parameter_value("avg_rain_in", parameters["avg_rain_in"])
        return float(avg_rain * runoff_days)

    missing = [
        name
        for name in ("annual_precip_in", "rain_correction_fraction")
        if parameters.get(name) is None
    ]
    if missing:
        raise ValueError(
            "PLET precipitation inputs must provide avg_rain_in (PLET AVG_RAIN) "
            "or both annual_precip_in and rain_correction_fraction; "
            f"missing {missing}"
        )
    annual_precip = validate_parameter_value(
        "annual_precip_in", parameters["annual_precip_in"]
    )
    rain_correction_fraction = validate_parameter_value(
        "rain_correction_fraction", parameters["rain_correction_fraction"]
    )
    return float(annual_precip * rain_correction_fraction)


def plet_annual_surface_runoff_in(
    parameters: Mapping[str, Any],
) -> Tuple[float, float, float, float]:
    """Estimate annual surface runoff depth from PLET precipitation inputs.

    Two precipitation forms are supported:

    * ``avg_rain_in`` + ``rain_days`` + ``runoff_day_fraction``; or
    * ``annual_precip_in`` + ``rain_correction_fraction`` + ``rain_days`` +
      ``runoff_day_fraction``.

    If ``avg_rain_in`` is present it is used directly as event rainfall.
    """
    parameters = apply_plet_parameter_defaults(parameters)
    if parameters.get("avg_rain_in") is not None:
        event_rainfall, event_runoff, annual_storm_runoff = (
            plet_avg_rain_runoff_depth_in(
                parameters["avg_rain_in"],
                parameters["rain_days"],
                parameters["runoff_day_fraction"],
                parameters["cn"],
                parameters["ia_ratio"],
            )
        )
    else:
        missing = [
            name
            for name in ("annual_precip_in", "rain_correction_fraction")
            if parameters.get(name) is None
        ]
        if missing:
            raise ValueError(
                "PLET precipitation inputs must provide avg_rain_in (PLET AVG_RAIN) "
                "or both annual_precip_in and rain_correction_fraction; "
                f"missing {missing}"
            )
        event_rainfall, event_runoff, annual_storm_runoff = plet_runoff_depth_in(
            parameters["annual_precip_in"],
            parameters["rain_days"],
            parameters["rain_correction_fraction"],
            parameters["runoff_day_fraction"],
            parameters["cn"],
            parameters["ia_ratio"],
        )
    runoff_multiplier = validate_parameter_value(
        "runoff_multiplier", parameters["runoff_multiplier"]
    )
    annual_total_runoff = annual_storm_runoff * runoff_multiplier
    return event_rainfall, event_runoff, annual_storm_runoff, annual_total_runoff

