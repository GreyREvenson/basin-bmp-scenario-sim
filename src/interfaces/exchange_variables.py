"""Versioned, numeric exchange-variable contracts for BMI adapters.

These are *local names*, NOT CSDMS Standard Names.  Indices refer to the
validated model's parcel/pollutant lists; they are not geospatial coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EXCHANGE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ExchangeVariable:
    """Metadata for one public numerical exchange variable."""
    name: str
    units: str
    dtype: str
    intent: str = "output"  # input or output
    description: str = ""
    rank: int = 1  # rank 0 scalar; rank 1 indexed vector
    resizable: bool = False  # only explicit variable-width BMI inputs

    def __post_init__(self):
        if self.intent not in {"input", "output"}:
            raise ValueError(f"Invalid exchange-variable intent: {self.intent}")
        if self.rank not in (0, 1):
            raise ValueError("Only BMI scalars and 1-D index-space vectors are supported")
        if self.resizable and (self.intent != "input" or self.rank == 0):
            raise ValueError("Only vector BMI inputs may be resizable")
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

# One shared, explicit numerical contract for each live BMP placement.
# Values with family-specific meaning are zero on other practice families.
# The scientific model reconstructs native BMPDesign only inside its adapter.
PLACEMENT_OUTPUT_VARIABLES = (
    ExchangeVariable("placement__cps_code", "1", "int64", rank=0),
    ExchangeVariable("placement__host_parcel_index", "1", "int64", rank=0),
    ExchangeVariable("placement__affected_parcel_index", "1", "int64"),
    ExchangeVariable("placement__affected_parcel_fraction", "1", "float64"),
    ExchangeVariable("placement__efficiency", "1", "float64",
                     description="Row-major (pollutant,pathway) in approved formulation order"),
    ExchangeVariable("placement__failed", "1", "int64", rank=0),
    ExchangeVariable("placement__recovery_effect_scale", "1", "float64", rank=0),
    ExchangeVariable("placement__cost_quantity_ha", "ha", "float64", rank=0),
    ExchangeVariable("placement__wetland_area_ha", "ha", "float64", rank=0),
    ExchangeVariable("placement__catchment_area_ratio", "1", "float64", rank=0),
    ExchangeVariable("placement__buffer_length_m", "m", "float64", rank=0),
    ExchangeVariable("placement__buffer_area_ha", "ha", "float64", rank=0),
    ExchangeVariable("placement__portion_treated", "1", "float64", rank=0),
    ExchangeVariable("placement__recorded_recovery_gains", "1", "float64",
                     description="For replay: pre-recorded AFO gains; empty in live generation"),
)

# Shared, read-only BMI metadata for validating ordered index domains.
# The manifest provides stable parcel/outlet identifiers, pollutant/pathway
# positions and available BMP CPS codes. It does not encode parcel geometry,
# routing topology, or approve arbitrary scientific formulations.
DOMAIN_INDEX_VARIABLES = (
    ExchangeVariable("coupling__exchange_schema_version", "1", "int64", rank=0),
    ExchangeVariable("coupling__index_manifest_utf8", "1", "uint8",
                     description="Versioned UTF-8 JSON of index-domain identities"),
    ExchangeVariable("coupling__index_fingerprint", "1", "uint8",
                     description="SHA-256 digest of the versioned index manifest"),
)

# Physical domain is published by the watershed component and installed by
# the detached generator. CSR upstream arrays retain the original neighbor
# ordering used in wetland geometry draws.
DOMAIN_OUTPUT_VARIABLES = (
    ExchangeVariable("domain__schema_version", "1", "int64", rank=0),
    ExchangeVariable("domain__parcel_area_ha", "ha", "float64"),
    ExchangeVariable("domain__parcel_perimeter_m", "m", "float64"),
    ExchangeVariable("domain__upstream_offset", "1", "int64"),
    ExchangeVariable("domain__upstream_index", "1", "int64"),
    ExchangeVariable("domain__selection_source_index", "1", "int64"),
    ExchangeVariable("domain__selection_probability", "1", "float64"),
)
DOMAIN_INPUT_VARIABLES = tuple(
    ExchangeVariable(v.name, v.units, v.dtype, intent="input", rank=v.rank,
                     description=v.description, resizable=(v.rank == 1))
    for v in DOMAIN_OUTPUT_VARIABLES
)

MODEL_VARIABLES = DOMAIN_INDEX_VARIABLES + DOMAIN_OUTPUT_VARIABLES + (
    ExchangeVariable("coupling__model_rng_state_utf8", "1", "uint8",
                     description="Logical RNG state after watershed-model draws"),
    ExchangeVariable("coupling__generator_rng_state_utf8", "1", "uint8",
                     intent="input", resizable=True),
    ExchangeVariable("model__parcel_count", "1", "int64", rank=0),
    ExchangeVariable("model__pollutant_count", "1", "int64", rank=0),
    ExchangeVariable("model__parcel_baseline_load_rate", "kg ha-1 yr-1", "float64",
                     description="Row-major (parcel,pollutant) baseline load-rate matrix"),
    ExchangeVariable("model__parcel_current_load_rate", "kg ha-1 yr-1", "float64",
                     description="Row-major (parcel,pollutant) treated load-rate matrix"),
    ExchangeVariable("model__outlet_count", "1", "int64", rank=0),
    ExchangeVariable("model__outlet_removed_mass", "kg", "float64",
                     description="Row-major (pollutant,outlet) signed delivered mass change"),
    ExchangeVariable("placement__recovery_gains", "1", "float64",
                     description="Pending AFO recovery gains to send to generator before its cost draw"),
    ExchangeVariable("model__last_bmp_record_utf8", "1", "uint8",
                     description="UTF-8 JSON output record for driver-managed storage only"),
) + tuple(
    ExchangeVariable(v.name, v.units, v.dtype, intent="input", rank=v.rank,
                     description=v.description, resizable=(v.rank == 1))
    for v in PLACEMENT_OUTPUT_VARIABLES
) + (
    ExchangeVariable("placement__cost", "USD", "float64", intent="input", rank=0),
    ExchangeVariable("scenario__identifier", "1", "int64", intent="input", rank=0),
)

DRIVER_VARIABLES = DOMAIN_INDEX_VARIABLES + (
    ExchangeVariable("ensemble__completed_scenario_count", "1", "int64", rank=0),
    ExchangeVariable("ensemble__requested_scenario_count", "1", "int64", rank=0),
    ExchangeVariable("scenario__bmp_count", "1", "int64", rank=0),
    ExchangeVariable("scenario__total_cost", "USD", "float64", rank=0),
    ExchangeVariable("scenario__outlet_removed_mass", "kg", "float64",
                     description="Row-major (pollutant,outlet) last scenario signed delivered change"),
)

LIVE_GENERATOR_VARIABLES = DOMAIN_INDEX_VARIABLES + DOMAIN_INPUT_VARIABLES + PLACEMENT_OUTPUT_VARIABLES + (
    ExchangeVariable("coupling__generator_rng_state_utf8", "1", "uint8",
                     description="Logical RNG state after generator draws"),
    ExchangeVariable("coupling__model_rng_state_utf8", "1", "uint8",
                     intent="input", resizable=True),
    ExchangeVariable("placement__completed", "1", "int64", rank=0),
    ExchangeVariable("placement__cost", "USD", "float64", rank=0,
                     description="Zero until placement__completed becomes 1"),
    ExchangeVariable("scenario__bmp_count", "1", "int64", rank=0),
    ExchangeVariable("scenario__cumulative_cost", "USD", "float64", rank=0),
    ExchangeVariable("placement__recovery_gains", "1", "float64", intent="input",
                     description="AFO recovery effects transferred from the model before cost sampling",
                     resizable=True),
)
