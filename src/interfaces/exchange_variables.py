"""Versioned, numeric exchange-variable contracts for BMI adapters.

These are *local names*, NOT CSDMS Standard Names.  Indices refer to the
validated model's parcel/pollutant lists; they are not geospatial coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EXCHANGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExchangeVariable:
    """Metadata for one public numerical exchange variable."""
    name: str
    units: str
    dtype: str
    intent: str = "output"  # input or output
    description: str = ""
    rank: int = 1  # rank 0 scalar; rank 1 indexed vector

    def __post_init__(self):
        if self.intent not in {"input", "output"}:
            raise ValueError(f"Invalid exchange-variable intent: {self.intent}")
        if self.rank not in (0, 1):
            raise ValueError("Only BMI scalars and 1-D index-space vectors are supported")
        if np.dtype(self.dtype).kind not in "iufb":
            raise ValueError("Exchange variables must be numeric")


def numeric_vector(value: Any, dtype: str) -> np.ndarray:
    """Return a contiguous 1-D copy, rejecting invalid numeric values."""
    array = np.asarray(value, dtype=np.dtype(dtype))
    if array.ndim > 1:
        raise ValueError("BMI exchange values must be flat vectors")
    if array.ndim == 0:
        array = array.reshape(1)
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError("BMI exchange values must be finite")
    return np.ascontiguousarray(array)


def scenario_arrays(scenario) -> dict[str, np.ndarray]:
    """Encode one *completed* BMPScenario using CSR affected-parcel arrays.

    Affected offsets are [0, count_BMP_1, count_BMP_1+count_BMP_2, ...].
    The original full scenario JSON remains authoritative for named metadata,
    pollutant-specific efficiency maps, and replay.
    """
    offsets = [0]
    affected_indices = []
    affected_fractions = []
    for placement in scenario.placements:
        if placement.cost_usd is None:
            raise ValueError("Only completed BMP placements can be exported")
        for idx, fraction in placement.design.treated:
            affected_indices.append(int(idx))
            affected_fractions.append(float(fraction))
        offsets.append(len(affected_indices))
    result = {
        "scenario__identifier": numeric_vector(scenario.scenario_id, "int64"),
        "scenario__random_seed": numeric_vector(scenario.seed, "int64"),
        "scenario__bmp_count": numeric_vector(len(scenario.placements), "int64"),
        "bmp__cps_code": numeric_vector([p.choice.cps for p in scenario.placements], "int64"),
        "bmp__host_parcel_index": numeric_vector([p.choice.parcel_index for p in scenario.placements], "int64"),
        "bmp__implementation_cost": numeric_vector([p.cost_usd for p in scenario.placements], "float64"),
        "bmp__affected_parcel_offset": numeric_vector(offsets, "int64"),
        "bmp__affected_parcel_index": numeric_vector(affected_indices, "int64"),
        "bmp__affected_parcel_fraction": numeric_vector(affected_fractions, "float64"),
    }
    if any(x < 0 or x >= len(affected_indices) for x in offsets[:-1]) and affected_indices:
        raise ValueError("Invalid CSR offsets")
    return result


SCENARIO_VARIABLES = (
    ExchangeVariable("scenario__identifier", "1", "int64", description="One-based scenario identifier", rank=0),
    ExchangeVariable("scenario__random_seed", "1", "int64", description="Worker seed", rank=0),
    ExchangeVariable("scenario__bmp_count", "1", "int64", description="Number of realized BMP placements", rank=0),
    ExchangeVariable("bmp__cps_code", "1", "int64", description="BMP CPS codes in implementation order"),
    ExchangeVariable("bmp__host_parcel_index", "1", "int64", description="Zero-based full parcel-universe indices"),
    ExchangeVariable("bmp__implementation_cost", "USD", "float64", description="Realized implementation cost per placement"),
    ExchangeVariable("bmp__affected_parcel_offset", "1", "int64", description="CSR offsets of affected parcels; length is BMP count plus one"),
    ExchangeVariable("bmp__affected_parcel_index", "1", "int64", description="Affected indices into the full hydrologic parcel universe"),
    ExchangeVariable("bmp__affected_parcel_fraction", "1", "float64", description="Corresponding fractional treatment of affected parcel"),
)

MODEL_VARIABLES = (
    ExchangeVariable("model__parcel_count", "1", "int64", rank=0),
    ExchangeVariable("model__pollutant_count", "1", "int64", rank=0),
    ExchangeVariable("model__parcel_baseline_load_rate", "kg ha-1 yr-1", "float64",
                     description="Row-major (parcel,pollutant) baseline load-rate matrix"),
    ExchangeVariable("model__parcel_current_load_rate", "kg ha-1 yr-1", "float64",
                     description="Row-major (parcel,pollutant) treated load-rate matrix"),
    ExchangeVariable("model__outlet_count", "1", "int64", rank=0),
    ExchangeVariable("model__outlet_removed_mass", "kg", "float64",
                     description="Row-major (pollutant,outlet) signed delivered mass change"),
)

DRIVER_VARIABLES = (
    ExchangeVariable("ensemble__completed_scenario_count", "1", "int64", rank=0),
    ExchangeVariable("ensemble__requested_scenario_count", "1", "int64", rank=0),
    ExchangeVariable("scenario__bmp_count", "1", "int64", rank=0),
    ExchangeVariable("scenario__total_cost", "USD", "float64", rank=0),
    ExchangeVariable("scenario__outlet_removed_mass", "kg", "float64",
                     description="Row-major (pollutant,outlet) last scenario signed delivered change"),
)

LIVE_GENERATOR_VARIABLES = (
    ExchangeVariable("placement__cps_code", "1", "int64", rank=0),
    ExchangeVariable("placement__host_parcel_index", "1", "int64", rank=0),
    ExchangeVariable("placement__affected_parcel_index", "1", "int64"),
    ExchangeVariable("placement__affected_parcel_fraction", "1", "float64"),
    ExchangeVariable("placement__efficiency", "1", "float64",
                     description="Row-major (pollutant,pathway) sampled BMP efficiency"),
    ExchangeVariable("placement__failed", "1", "int64", rank=0),
    ExchangeVariable("placement__recovery_effect_scale", "1", "float64", rank=0),
    ExchangeVariable("placement__completed", "1", "int64", rank=0),
    ExchangeVariable("placement__cost", "USD", "float64", rank=0,
                     description="Zero until placement__completed becomes 1"),
    ExchangeVariable("scenario__bmp_count", "1", "int64", rank=0),
    ExchangeVariable("scenario__cumulative_cost", "USD", "float64", rank=0),
)
