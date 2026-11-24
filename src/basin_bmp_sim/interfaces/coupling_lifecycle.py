"""Versioned project-specific *extensions* to the CSDMS BMI 2.0 API.

BMI defines initialize/update/get_value/set_value/finalize; it does not define
live two-stage BMP responses, physical-domain import or replay export. These
three operations remain explicit extensions and are negotiated independently
of the numeric exchange-schema version. Callers use only public methods and
BMI output variables; component private contexts must not cross this boundary.
"""
from __future__ import annotations

from enum import Enum

import numpy as np


LIFECYCLE_SCHEMA_VERSION = 1
LIFECYCLE_VARIABLE = "coupling__lifecycle_schema_version"
REPLAY_VARIABLE = "scenario__replay_utf8"


class CouplerPhase(str, Enum):
    CREATED = "created"
    READY = "ready"
    EXHAUSTED = "exhausted"
    FINALIZED = "finalized"


def negotiate_lifecycle(generator, model) -> None:
    """Validate the same extension version and required operations on both owners.

    Called after BMI initialize and before parcel-domain transfer or placement
    sampling. No internal Python scientific state is inspected.
    """
    for role, component, extensions in (
        ("generator", generator, ("accept_domain", "complete_pending", "export_replay")),
        ("watershed model", model, ("complete_pending",)),
    ):
        if LIFECYCLE_VARIABLE not in component.get_output_var_names():
            raise ValueError(f"BMI {role} missing lifecycle schema variable")
        if (component.get_var_type(LIFECYCLE_VARIABLE) != "int64" or
                component.get_var_units(LIFECYCLE_VARIABLE) != "1" or
                component.get_grid_rank(component.get_var_grid(LIFECYCLE_VARIABLE)) != 0):
            raise ValueError(f"BMI {role} lifecycle metadata mismatch")
        size = component.get_grid_size(component.get_var_grid(LIFECYCLE_VARIABLE))
        if size != 1:
            raise ValueError(f"BMI {role} lifecycle schema must be scalar")
        value = np.empty(1, dtype=np.int64)
        component.get_value(LIFECYCLE_VARIABLE, value)
        if int(value[0]) != LIFECYCLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported BMI {role} lifecycle schema version {int(value[0])}")
        for method in extensions:
            if not callable(getattr(component, method, None)):
                raise ValueError(f"BMI {role} missing lifecycle extension {method}()")
    if REPLAY_VARIABLE not in generator.get_output_var_names():
        raise ValueError("BMI generator missing replay output variable")
    if (generator.get_var_type(REPLAY_VARIABLE) != "uint8" or
            generator.get_var_units(REPLAY_VARIABLE) != "1" or
            generator.get_grid_rank(generator.get_var_grid(REPLAY_VARIABLE)) != 1):
        raise ValueError("BMI generator replay output metadata mismatch")
