"""Declarative schema for consolidated model input GeoPackages.

The registry in this module is the single source of truth for user-facing
``input_*`` tables stored in ``parcels.gpkg``.  Loaders may still perform
variable-specific semantic validation, but table discovery, key columns,
required modes, and basic input kinds are defined here so the contract is not
spread across several modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .constants import (
    GPKG_DELIVERY_RATIO_TABLES,
    GPKG_INPUT_CURVE_NUMBER,
    GPKG_INPUT_INFILTRATION_FRACTION,
    GPKG_INPUT_POLLUTANT_LOAD_RATE,
    GPKG_INPUT_SELECTION_WEIGHT,
    GPKG_INPUT_SUBSURFACE_CONCENTRATION,
    GPKG_INPUT_SURFACE_CONCENTRATION,
    PARCEL_PARAMETER_INPUT_TABLES,
)

INPUT_SCHEMA_TABLE = "model_input_schema"
INPUT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class InputVariableSpec:
    """Description of one user-facing ``input_*`` table."""

    name: str
    table: str
    key_columns: Tuple[str, ...]
    kind: str
    required_modes: Tuple[str, ...] = ()
    canonical_units: str | None = None
    wildcard_allowed: bool = False


_PLET_REQUIRED_PARAMETERS = {
    "annual_precip_in",
    "rain_days",
    "rain_correction_fraction",
    "runoff_day_fraction",
    "land_cover",
    "hsg",
}


def _build_registry() -> Dict[str, InputVariableSpec]:
    registry: Dict[str, InputVariableSpec] = {}

    for parameter, table in PARCEL_PARAMETER_INPUT_TABLES.items():
        kind = "classification" if parameter in {"land_cover", "hsg"} else "numeric_distribution"
        required_modes = ("plet_rusle",) if parameter in _PLET_REQUIRED_PARAMETERS else ()
        registry[table] = InputVariableSpec(
            name=parameter,
            table=table,
            key_columns=("pid",),
            kind=kind,
            required_modes=required_modes,
            wildcard_allowed=True,
        )

    registry[GPKG_INPUT_SELECTION_WEIGHT] = InputVariableSpec(
        name="selection_weight",
        table=GPKG_INPUT_SELECTION_WEIGHT,
        key_columns=("pid",),
        kind="fixed_numeric",
        wildcard_allowed=True,
    )
    registry[GPKG_INPUT_CURVE_NUMBER] = InputVariableSpec(
        name="cn",
        table=GPKG_INPUT_CURVE_NUMBER,
        key_columns=("land_cover", "hsg"),
        kind="numeric_distribution",
        required_modes=("plet_rusle",),
    )
    registry[GPKG_INPUT_INFILTRATION_FRACTION] = InputVariableSpec(
        name="infiltration_fraction",
        table=GPKG_INPUT_INFILTRATION_FRACTION,
        key_columns=("land_cover", "hsg"),
        kind="numeric_distribution",
        required_modes=("plet_rusle",),
    )
    registry[GPKG_INPUT_POLLUTANT_LOAD_RATE] = InputVariableSpec(
        name="pollutant_load_rate",
        table=GPKG_INPUT_POLLUTANT_LOAD_RATE,
        key_columns=("pid", "pollutant"),
        kind="numeric_distribution",
        required_modes=("statistical",),
        wildcard_allowed=True,
    )
    registry[GPKG_INPUT_SURFACE_CONCENTRATION] = InputVariableSpec(
        name="surface_concentration",
        table=GPKG_INPUT_SURFACE_CONCENTRATION,
        key_columns=("pid", "pollutant"),
        kind="numeric_distribution",
        wildcard_allowed=True,
    )
    registry[GPKG_INPUT_SUBSURFACE_CONCENTRATION] = InputVariableSpec(
        name="subsurface_concentration",
        table=GPKG_INPUT_SUBSURFACE_CONCENTRATION,
        key_columns=("pid", "pollutant"),
        kind="numeric_distribution",
        wildcard_allowed=True,
    )

    for name, table in GPKG_DELIVERY_RATIO_TABLES.items():
        registry[table] = InputVariableSpec(
            name=name,
            table=table,
            key_columns=("pid", "oid"),
            kind="fixed_numeric",
            wildcard_allowed=True,
        )

    return registry


INPUT_VARIABLE_SPECS: Dict[str, InputVariableSpec] = _build_registry()
KNOWN_INPUT_TABLES = frozenset(INPUT_VARIABLE_SPECS)


def required_input_tables(mode: str) -> frozenset[str]:
    """Return tables that must physically exist for a load-generation mode."""
    normalized = str(mode).strip().lower()
    return frozenset(
        spec.table
        for spec in INPUT_VARIABLE_SPECS.values()
        if normalized in spec.required_modes
    )
