"""Feedlot BMP manure recovery response."""

from __future__ import annotations
from typing import Any
import numpy as np

from .shared import _draw, _finite


def apply_afo_feedlot_manure_effect(
    ctx: Any, host_pid: str, cps: int, load_rates: np.ndarray,
    *, bmp_effect_scale: float = 1.0,
    realized_recovery_gains: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    """Apply an explicitly configured manure-collection gain after a feedlot BMP.

    The gain is the recovered fraction of the *remaining uncollected* manure,
    applied sequentially. The returned changes in kg/year are signed increases
    in receiving-field surface exports, after earlier field BMP treatment.
    No recovery effect is inferred from a CPS code or a runoff efficiency.
    """
    inputs = getattr(ctx, "afo_inputs", None) or {}
    if inputs.get("method") != "mass_based":
        return []
    operation_state = getattr(ctx, "afo_bmp_operation_state", {})
    if not operation_state:
        return []
    bmp_effect_scale = _finite(bmp_effect_scale, "bmp_effect_scale", 0, 1)
    rows = getattr(ctx, "afo_bmp_effects_by_host_cps", {}).get(
        (str(host_pid), int(cps)), []
    )
    if realized_recovery_gains is not None and len(realized_recovery_gains) != len(rows):
        raise ValueError("Replayed AFO recovery count does not match configured effects")
    changes: list[dict[str, Any]] = []
    for effect_index, effect in enumerate(rows):
        op_id = str(effect["afo_id"])
        op = operation_state[op_id]
        if realized_recovery_gains is None:
            gain = _draw(ctx, effect, "additional_collection_fraction", fraction=True)
            gain *= bmp_effect_scale
        else:
            # The stored gain is already scaled by BMP failure; do not resample.
            gain = _finite(realized_recovery_gains[effect_index],
                           "replayed additional_collection_fraction", 0, 1)
        old = op["current_collection"]
        new = old + (1. - old) * gain
        for source in op["sources"]:
            updated_rate = source["full_rate"] * new
            delta_rate = updated_rate - source["current_rate"]
            source["current_rate"] = updated_rate
            idx, pol_idx, path_idx = source["parcel_idx"], source["pol_idx"], source["path_idx"]
            ctx.current_pathway_load_rates[idx, pol_idx, path_idx] += delta_rate
            ctx.current_afo_manure_pathway_rates[idx, pol_idx, path_idx] += delta_rate
            load_rates[idx, pol_idx] += delta_rate
            if delta_rate:
                changes.append(dict(afo_id=op_id, cps=int(cps),
                                    receiving_pid=str(ctx.parcel_ids[idx]),
                                    parcel_idx=idx, pol_idx=pol_idx,
                                    added_export_kg=delta_rate * float(ctx.parcel_area_ha[idx])))
        op["current_collection"] = new
        ctx.afo_bmp_effect_records.append(dict(
            afo_id=op_id, cps=int(cps), host_pid=str(host_pid),
            recovery_fraction=gain, collection_before=old,
            collection_after=new,
            added_available_tn_kg=(new - old) * op["available_full"]["TN"],
            added_available_tp_kg=(new - old) * op["available_full"]["TP"],
            added_surface_tn_kg=sum(rec["added_export_kg"] for rec in changes
                                    if rec["afo_id"] == op_id and
                                    str(ctx.pollutants[rec["pol_idx"]]).upper() == "TN"),
            added_surface_tp_kg=sum(rec["added_export_kg"] for rec in changes
                                    if rec["afo_id"] == op_id and
                                    str(ctx.pollutants[rec["pol_idx"]]).upper() == "TP"),
        ))
    return changes
