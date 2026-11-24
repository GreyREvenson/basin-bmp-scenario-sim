"""Choose manure receiving fields."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import pandas as pd

from .shared import _draw, _finite


def _select_manure_fields(ctx: Any, inputs: Mapping[str, Any], eligible_area: Mapping[str, float]) -> list[dict[str, Any]]:
    """Select receivers once per realization with shared explicit/stochastic rules."""
    operations = inputs["operations"]
    fields = inputs["fields"]
    geometry = inputs["geometry"]
    assignments: list[dict[str, Any]] = []
    # Reuse exactly the same spatial selection for the legacy concentration
    # mode and the new mass-conserving nutrient-allocation mode.
    for op in sorted(operations, key=lambda r: str(r["afo_id"])):
        op_id, host = str(op["afo_id"]), str(op["pid"])
        if op["allocation"] == "explicit":
            chosen = [str(r["pid"]) for r in fields if str(r["afo_id"]) == op_id]
            if len(chosen) != len(set(chosen)):
                raise ValueError(f"Duplicate receiving parcel in {op_id}")
        else:
            candidates = sorted(eligible_area, key=lambda p: int(p))
            # In statistical mode the field universe is fixed and prevalidated;
            # truncate unbounded distributions to its physical capacity.
            n_draw = _draw(
                ctx, op, "n_fields", minimum=1.,
                maximum=len(candidates) if inputs.get("mode") == "statistical" else None,
            )
            if not np.isfinite(n_draw) or n_draw < 1:
                raise ValueError(f"{op_id}: sampled n_fields must be at least 1")
            n_fields = int(np.floor(n_draw + 0.5))
            if n_fields > len(candidates):
                raise ValueError(f"{op_id}: sampled {n_fields} fields but only {len(candidates)} eligible")
            chosen = [host] if host in eligible_area else []
            candidates = [p for p in candidates if p != host]
            if op.get("selection", "nearest") == "random":
                candidate_order = list(ctx.rng.permutation(candidates))
            else:
                candidate_order = sorted(candidates, key=lambda p: (geometry[host].distance(geometry[p]), int(p)))
            chosen.extend(candidate_order[:n_fields - len(chosen)])
        for pid in chosen:
            if pid not in eligible_area:
                raise ValueError(f"{op_id}: manure receiving pid {pid} lacks eligible agricultural area")
            rec = {"afo_id": op_id, "host_pid": host, "receiving_pid": pid,
                   "eligible_area_ha": float(eligible_area[pid]),
                   "allocation": op["allocation"]}
            if inputs.get("method") != "mass_based":
                rec["manure_months"] = float(op["manure_months"])
            field_row = next((r for r in fields if str(r["afo_id"]) == op_id and str(r["pid"]) == pid), None)
            if field_row is not None and field_row.get("weight") is not None and not pd.isna(field_row["weight"]):
                rec["weight"] = _finite(field_row["weight"], "afo_manure_fields.weight")
            assignments.append(rec)
    return assignments
