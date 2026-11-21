"""Versioned, testable public annual-BMI compatibility surface.

This is the project's *BMI 2.0 plus annual coupling extension* contract, not
CSDMS Standard Names or external BMI certification. No scientific or driver
implementation classes are imported. A frozen snapshot is distributed with
this package; intentional incompatible changes require an explicit version
and snapshot migration instead of silently changing the live exchange surface.
"""
from __future__ import annotations

import argparse
from importlib import resources
import json
from typing import Any

import numpy as np

from src.interfaces.coupling_lifecycle import LIFECYCLE_SCHEMA_VERSION
from src.interfaces.domain_index_contract import DOMAIN_INDEX_SCHEMA_VERSION
from src.interfaces.exchange_variables import (
    DRIVER_VARIABLES, EXCHANGE_SCHEMA_VERSION, LIVE_GENERATOR_VARIABLES,
    MODEL_VARIABLES, SCENARIO_VARIABLES, ExchangeVariable,
)


PUBLIC_CONTRACT_VERSION = 1
SNAPSHOT_NAME = "public_contract_v1.json"

# BMI 2.0 core abstract method names. The optional bmipy package is tested
# separately against its own abstract method set (see test_phase7_bmipy_optional).
BMI_2_0_METHODS = (
    "initialize", "update", "update_until", "finalize",
    "get_component_name", "get_input_item_count", "get_output_item_count",
    "get_input_var_names", "get_output_var_names", "get_var_grid",
    "get_var_type", "get_var_units", "get_var_itemsize", "get_var_nbytes",
    "get_var_location", "get_current_time", "get_start_time",
    "get_end_time", "get_time_units", "get_time_step", "get_value",
    "get_value_ptr", "get_value_at_indices", "set_value",
    "set_value_at_indices", "get_grid_rank", "get_grid_size",
    "get_grid_type", "get_grid_shape", "get_grid_spacing",
    "get_grid_origin", "get_grid_x", "get_grid_y", "get_grid_z",
    "get_grid_node_count", "get_grid_edge_count", "get_grid_face_count",
    "get_grid_edge_nodes", "get_grid_face_edges", "get_grid_face_nodes",
    "get_grid_nodes_per_face",
)

# These represent separate surfaces; a driver is not a live generator and a
# saved-scenario reader is not a live stochastic generator. External scientific
# adapters use the same annual_model surface as the bundled PLET adapter.
PUBLIC_COMPONENTS: dict[str, tuple[ExchangeVariable, ...]] = {
    "annual_model": MODEL_VARIABLES,
    "live_generator": LIVE_GENERATOR_VARIABLES,
    "recorded_scenario": SCENARIO_VARIABLES,
    "monte_carlo_driver": DRIVER_VARIABLES,
}
PROJECT_EXTENSIONS = {
    "annual_model": ["complete_pending"],
    "live_generator": ["accept_domain", "complete_pending", "export_replay"],
    "recorded_scenario": [],
    "monte_carlo_driver": [],
}


def public_contract() -> dict[str, Any]:
    """Return JSON-safe public metadata, in the declared variable order."""
    return {
        "schema_version": PUBLIC_CONTRACT_VERSION,
        "bmi_version": "2.0",
        "exchange_schema_version": EXCHANGE_SCHEMA_VERSION,
        "index_manifest_schema_version": DOMAIN_INDEX_SCHEMA_VERSION,
        "lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
        "time_interpretation": "logical_step_not_physical_year",
        "spatial_interpretation": "one_dimensional_index_space_not_geospatial_grid",
        "bmi_methods": list(BMI_2_0_METHODS),
        "project_convenience_methods": ["get_bmi_version"],
        "components": {
            role: {
                "project_extensions": list(PROJECT_EXTENSIONS[role]),
                "variables": [{
                    "name": v.name,
                    "intent": v.intent,
                    "dtype": np.dtype(v.dtype).name,
                    "units": v.units,
                    "rank": v.rank,
                    "resizable": v.resizable,
                } for v in variables],
            }
            for role, variables in PUBLIC_COMPONENTS.items()
        },
    }


def _canonical(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode("utf-8")


def verify_public_contract() -> None:
    """Fail if the published catalog changed without a versioned migration."""
    reference = json.loads(resources.files("src.interfaces.bmi").joinpath(SNAPSHOT_NAME)
                           .read_text(encoding="utf-8"))
    current = public_contract()
    if _canonical(reference) != _canonical(current):
        changed = []
        for key in sorted(set(reference) | set(current)):
            if reference.get(key) != current.get(key):
                changed.append(key)
        raise ValueError("Public annual BMI contract drift: " + ", ".join(changed)
                         + ". Review compatibility and version the contract before changing it.")


def verify_component(component: object, role: str) -> dict[str, Any]:
    """Audit an initialized adapter using only advertised public BMI methods.

    This checks structural *and runtime array* consistency, never invokes
    update() or setters, and never reads a private scientific context. It is
    suitable for independently implemented annual-model adapters, provided
    they expose the same public index-space contract. Extra variables are
    allowed for additive backwards-compatible extension.
    """
    if role not in PUBLIC_COMPONENTS:
        raise ValueError(f"Unknown public BMI component role: {role}")
    missing = [name for name in BMI_2_0_METHODS
               if not callable(getattr(component, name, None))]
    missing.extend(name for name in PROJECT_EXTENSIONS[role]
                   if not callable(getattr(component, name, None)))
    if missing:
        raise ValueError(f"BMI {role} missing required methods: {sorted(set(missing))}")
    if component.get_bmi_version() != "2.0":
        raise ValueError(f"BMI {role} must advertise BMI 2.0")
    inputs = tuple(component.get_input_var_names())
    outputs = tuple(component.get_output_var_names())
    if (len(inputs) != len(set(inputs)) or len(outputs) != len(set(outputs))
            or set(inputs) & set(outputs)
            or component.get_input_item_count() != len(inputs)
            or component.get_output_item_count() != len(outputs)):
        raise ValueError(f"BMI {role} inconsistent input/output declaration")
    for v in PUBLIC_COMPONENTS[role]:
        names = inputs if v.intent == "input" else outputs
        if v.name not in names:
            raise ValueError(f"BMI {role} missing {v.intent} variable {v.name}")
        expected = (np.dtype(v.dtype).name, v.units, v.rank)
        actual = (
            np.dtype(component.get_var_type(v.name)).name,
            component.get_var_units(v.name),
            component.get_grid_rank(component.get_var_grid(v.name)),
        )
        if actual != expected:
            raise ValueError(f"BMI {role} metadata mismatch for {v.name}: "
                             f"expected {expected}, got {actual}")
    examined = 0
    for name in (*inputs, *outputs):
        dtype = np.dtype(component.get_var_type(name))
        grid = component.get_var_grid(name)
        rank = component.get_grid_rank(grid)
        count = component.get_grid_size(grid)
        if rank not in (0, 1) or count < 0:
            raise ValueError(f"BMI {role} invalid index grid for {name}")
        expected_grid_type = "scalar" if rank == 0 else "uniform_rectilinear"
        if component.get_grid_type(grid) != expected_grid_type:
            raise ValueError(f"BMI {role} invalid virtual grid type for {name}")
        if (component.get_var_itemsize(name) != dtype.itemsize
                or component.get_var_nbytes(name) != count * dtype.itemsize
                or component.get_var_location(name) != "node"):
            raise ValueError(f"BMI {role} inconsistent size or location for {name}")
        shape = np.empty(rank, dtype=np.int64)
        component.get_grid_shape(grid, shape)
        if tuple(shape) != (() if rank == 0 else (count,)) or (rank == 0 and count != 1):
            raise ValueError(f"BMI {role} invalid grid shape for {name}")
        if rank == 1:
            spacing = np.empty(1, dtype=np.float64)
            origin = np.empty(1, dtype=np.float64)
            x = np.empty(count, dtype=np.float64)
            component.get_grid_spacing(grid, spacing)
            component.get_grid_origin(grid, origin)
            component.get_grid_x(grid, x)
            if (not np.array_equal(spacing, [1.0])
                    or not np.array_equal(origin, [0.0])
                    or not np.array_equal(x, np.arange(count, dtype=np.float64))):
                raise ValueError(f"BMI {role} inconsistent index coordinates for {name}")
        values = np.empty(count, dtype=dtype)
        component.get_value(name, values)
        pointer = np.asarray(component.get_value_ptr(name))
        if pointer.shape != (count,) or pointer.dtype != dtype or not np.array_equal(pointer, values):
            raise ValueError(f"BMI {role} pointer/copy mismatch for {name}")
        if count:
            indices = np.array(sorted({0, count - 1}), dtype=np.int64)
            selected = np.empty(len(indices), dtype=dtype)
            component.get_value_at_indices(name, selected, indices)
            if not np.array_equal(selected, values[indices]):
                raise ValueError(f"BMI {role} indexed value mismatch for {name}")
        examined += 1
    start, now, end, step = (float(fn()) for fn in (
        component.get_start_time, component.get_current_time,
        component.get_end_time, component.get_time_step))
    if (not all(np.isfinite(x) for x in (start, now, end, step))
            or start > now or now > end or step <= 0
            or component.get_time_units() != "none"):
        raise ValueError(f"BMI {role} inconsistent logical time metadata")
    return {"role": role, "component": component.get_component_name(),
            "checked_variables": examined, "time_units": "none"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or print the versioned annual BMI public contract")
    parser.add_argument("--show", action="store_true", help="Print the current public contract as JSON")
    args = parser.parse_args()
    verify_public_contract()
    if args.show:
        print(json.dumps(public_contract(), sort_keys=True, indent=2))
    else:
        print(f"Annual BMI public contract v{PUBLIC_CONTRACT_VERSION}: verified")


if __name__ == "__main__":
    main()
