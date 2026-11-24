"""Black-box acceptance probes for the project's annual BMI exchange surface.

Only *public BMI methods* are consulted. This is an executable project
interoperability check, not third-party CSDMS BMI certification, nor a test of
scientific validity. Never invoke a component's private model context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from basin_bmp_sim.execution.backend_selection import BackendSelection, build_selected_coupler
from basin_bmp_sim.execution.bmi_coupling import get_bmi_value, validate_annual_model_bmi
from basin_bmp_sim.interfaces.bmi.public_contract import verify_component, verify_public_contract
from basin_bmp_sim.certification.release_evidence import source_identity


def probe_component(component) -> dict[str, Any]:
    """Check BMI variables, array-copy/index API, and virtual grid metadata."""
    inputs = component.get_input_var_names()
    outputs = component.get_output_var_names()
    if (len(set(inputs) | set(outputs)) != len(inputs) + len(outputs)
            or component.get_input_item_count() != len(inputs)
            or component.get_output_item_count() != len(outputs)):
        raise ValueError("BMI input/output metadata is inconsistent")
    if component.get_bmi_version() != "2.0":
        raise ValueError("Expected a BMI 2.0 component")
    examined: list[dict[str, Any]] = []
    for name in (*inputs, *outputs):
        dtype = np.dtype(component.get_var_type(name))
        grid = component.get_var_grid(name)
        rank = component.get_grid_rank(grid)
        count = component.get_grid_size(grid)
        if rank not in (0, 1) or count < 0:
            raise ValueError(f"BMI {name} has invalid index grid")
        if component.get_var_itemsize(name) != dtype.itemsize or component.get_var_nbytes(name) != count * dtype.itemsize:
            raise ValueError(f"BMI {name} size metadata disagrees with array")
        if component.get_var_location(name) != "node":
            raise ValueError(f"BMI {name} uses an unsupported index-grid location")
        shape = np.empty(rank, dtype=np.int64)
        component.get_grid_shape(grid, shape)
        if (rank == 1 and tuple(shape) != (count,)) or (rank == 0 and count != 1):
            raise ValueError(f"BMI {name} grid shape disagrees with size")
        copied = np.empty(count, dtype=dtype)
        component.get_value(name, copied)
        # Read via BMI rather than private backing arrays; an independent
        # adapter need not implement get_value_ptr as a NumPy-owning pointer.
        via_helper = get_bmi_value(component, name)
        np.testing.assert_array_equal(copied, via_helper)
        if count:
            indices = np.array(sorted({0, count - 1}), dtype=np.int64)
            selected = np.empty(len(indices), dtype=dtype)
            component.get_value_at_indices(name, selected, indices)
            np.testing.assert_array_equal(selected, copied[indices])
        examined.append({
            "name": name, "intent": "input" if name in inputs else "output",
            "dtype": dtype.name, "units": component.get_var_units(name),
            "rank": rank, "size": count,
        })
    return {"component": component.get_component_name(), "variables": examined}


def _run_probe(selection: BackendSelection) -> dict[str, Any]:
    coupler = build_selected_coupler(selection)
    initialized = False
    try:
        coupler.initialize()
        initialized = True
        verify_public_contract()
        verify_component(coupler.generator, "live_generator")
        verify_component(coupler.model, "annual_model")
        generator = probe_component(coupler.generator)
        model = probe_component(coupler.model)
        validate_annual_model_bmi(coupler.model)
        count = 0
        while (row := coupler.next_bmp()) is not None:
            count += 1
            actual = get_bmi_value(coupler.model, "model__outlet_removed_mass")
            np.testing.assert_array_equal(actual, row.delivered_reductions.ravel(order="C"))
        replay = coupler.completed_scenario().to_dict()
        if len(replay["placements"]) != count:
            raise ValueError("BMI response count and replay placement count disagree")
        # Only hashes of these runtime artifacts are required in the report.
        # Avoid publishing all model state in an acceptance report.
        final = {
            "replay": replay,
            "baseline": get_bmi_value(coupler.model, "model__parcel_baseline_load_rate").tolist(),
            "current": get_bmi_value(coupler.model, "model__parcel_current_load_rate").tolist(),
            "outlet": get_bmi_value(coupler.model, "model__outlet_removed_mass").tolist(),
        }
        blob = json.dumps(final, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return {
            "generator": generator, "model": model,
            "bmp_count": count, "result_sha256": hashlib.sha256(blob).hexdigest(),
        }
    finally:
        if initialized:
            coupler.finalize()


def accept_backend(selection_file: str | Path, *,
                   additional_files: tuple[str | Path, ...] = ()) -> dict[str, Any]:
    """Twice execute one startup, verify deterministic public-interface behavior.

    Only schema v1 single-scenario selection is accepted to keep the probe
    small and unambiguous; ensemble scheduling has its own integration tests.
    """
    selection_path = Path(selection_file).expanduser().resolve(strict=True)
    selection = BackendSelection.read(selection_path)
    if selection.ensemble is not None:
        raise ValueError("BMI acceptance requires single-scenario backend selection")
    identity = source_identity(selection_path, selection.component_manifest,
                               additional_files=additional_files)
    first = _run_probe(selection)
    second = _run_probe(selection)
    if first != second:
        raise ValueError("BMI backend repeated-startup determinism or metadata mismatch")
    if source_identity(selection_path, selection.component_manifest,
                       additional_files=additional_files) != identity:
        raise ValueError("Release evidence scientific inputs changed during acceptance")
    return {
        "acceptance_schema_version": 2,
        "scope": "project_annual_bmi_black_box_not_csdms_certification",
        "backend": selection.backend,
        "external_factory": selection.external_factory,
        "deterministic_repeat": True,
        **identity,
        **first,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe public annual BMI adapter methods and repeatability")
    parser.add_argument("selection_json", type=Path)
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--input-file", action="append", default=[],
                        help="Additional scientific asset (repeatable; selection-relative)")
    args = parser.parse_args()
    report = accept_backend(args.selection_json, additional_files=tuple(args.input_file))
    target = args.report_json.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    print(target)


if __name__ == "__main__":
    main()
