"""Explicit, lossless numeric BMI contract for a *pending* BMP placement.

The exchange uses parcel indices in the current hydrologic universe and
pollutant/pathway indices from the validated, approved formulation.  No PIDs,
models, loggers, or native placement objects cross the BMI boundary.

Scientific methods, cost draws, and sampling remain in their original owners.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from src.bmp.bmp_response_registry import response_family
from src.constants import (
    OUTPUT_BUFFER_AREA, OUTPUT_CATCHMENT_RATIO, OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH, OUTPUT_PORTION_TREATED, OUTPUT_WETLAND_AREA,
)
from src.scenarios.bmp_designs import BMPDesign
from src.scenarios.stochastic_scenario_generator import PlacementChoice, RealizedBMPPlacement

# Same variable names on the producer and consumer; BMI getters/setters mediate
# the values, while the context supplies the prevalidated domain/ID mapping.
PLACEMENT_FIELDS = (
    "placement__cps_code", "placement__host_parcel_index",
    "placement__affected_parcel_index", "placement__affected_parcel_fraction",
    "placement__efficiency", "placement__failed",
    "placement__recovery_effect_scale", "placement__cost_quantity_ha",
    "placement__wetland_area_ha", "placement__catchment_area_ratio",
    "placement__buffer_length_m", "placement__buffer_area_ha",
    "placement__portion_treated", "placement__recorded_recovery_gains",
)

SCALAR_FIELDS = set(PLACEMENT_FIELDS) - {
    "placement__affected_parcel_index", "placement__affected_parcel_fraction",
    "placement__efficiency", "placement__recorded_recovery_gains",
}


def placement_arrays(placement: RealizedBMPPlacement, ctx) -> dict[str, object]:
    """Encode one pending design/response draw with no numerical rounding."""
    d = placement.design
    metadata = d.record_fields()
    efficiencies = [float(value) for pol in placement.efficiency_by_pollutant for _, value in pol]
    expected_paths = tuple(str(p) for p in ctx.pathway_names)
    if len(placement.efficiency_by_pollutant) != len(ctx.pollutants) or any(
        tuple(k for k, _ in pol) != expected_paths for pol in placement.efficiency_by_pollutant
    ):
        raise ValueError("BMI placement efficiency pathway ordering disagrees with model formulation")
    return {
        "placement__cps_code": placement.choice.cps,
        "placement__host_parcel_index": placement.choice.parcel_index,
        "placement__affected_parcel_index": [i for i, _ in d.treated],
        "placement__affected_parcel_fraction": [f for _, f in d.treated],
        "placement__efficiency": efficiencies,
        "placement__failed": int(placement.failed),
        "placement__recovery_effect_scale": placement.recovery_effect_scale,
        "placement__cost_quantity_ha": d.cost_quantity_ha,
        "placement__wetland_area_ha": float(metadata.get(OUTPUT_WETLAND_AREA) or 0.0),
        "placement__catchment_area_ratio": float(metadata.get(OUTPUT_CATCHMENT_RATIO) or 0.0),
        "placement__buffer_length_m": float(metadata.get(OUTPUT_LINEAR_LENGTH) or 0.0),
        "placement__buffer_area_ha": float(metadata.get(OUTPUT_BUFFER_AREA) or 0.0),
        "placement__portion_treated": float(metadata.get(OUTPUT_PORTION_TREATED) or 0.0),
        "placement__recorded_recovery_gains": ([] if placement.recovery_gains is None
                                             else list(placement.recovery_gains)),
    }


def placement_from_arrays(values: Mapping[str, np.ndarray], ctx) -> RealizedBMPPlacement:
    """Rebuild the exact native placement object from BMI numerical values.

    Reject mismatched array dimensions, invalid parcel indices, impossible
    practice geometry, and unsupported CPS *before* scientific execution.
    """
    missing = set(PLACEMENT_FIELDS) - set(values)
    if missing:
        raise ValueError(f"Incomplete BMI placement exchange: {sorted(missing)}")

    def scalar(name: str):
        a = np.asarray(values[name])
        if a.size != 1:
            raise ValueError(f"BMI scalar {name} must contain one value")
        return a.reshape(-1)[0].item()

    def positive_int(name: str):
        v = scalar(name)
        if not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)):
            raise ValueError(f"BMI {name} must be integer")
        return int(v)

    cps = positive_int("placement__cps_code")
    idx = positive_int("placement__host_parcel_index")
    if idx < 0 or idx >= len(ctx.parcel_ids):
        raise ValueError("BMI host parcel index is outside the hydrologic parcel universe")
    if cps not in ctx.bmp_cps:
        raise ValueError(f"BMI placement CPS {cps} is not in the approved formulation")
    family = response_family(cps)
    affected = np.asarray(values["placement__affected_parcel_index"])
    fractions = np.asarray(values["placement__affected_parcel_fraction"], dtype=float)
    if affected.ndim != 1 or fractions.ndim != 1 or not len(affected) or len(affected) != len(fractions):
        raise ValueError("BMI placement affected-parcel vectors must be equally sized and nonempty")
    if affected.dtype.kind not in "iu" or any(i < 0 or i >= len(ctx.parcel_ids) for i in affected):
        raise ValueError("BMI placement affected parcel indices are invalid")
    if int(affected[0]) != idx or len(set(affected.tolist())) != len(affected):
        raise ValueError("BMI placement must treat host first, without duplicate parcels")
    if not np.all(np.isfinite(fractions)) or np.any(fractions < 0) or np.any(fractions > 1):
        raise ValueError("BMI placement treatment fractions must lie in [0, 1]")
    treated = tuple((int(i), float(f)) for i, f in zip(affected, fractions))

    eff = np.asarray(values["placement__efficiency"], dtype=float)
    expected_n = len(ctx.pollutants) * len(ctx.pathway_names)
    if eff.ndim != 1 or eff.size != expected_n or not np.all(np.isfinite(eff)) or np.any(eff > 1):
        raise ValueError("BMI placement efficiencies do not match the approved pollutant/pathway layout")
    eff_rows = tuple(
        tuple((str(path), float(v)) for path, v in zip(ctx.pathway_names, row))
        for row in eff.reshape(len(ctx.pollutants), len(ctx.pathway_names))
    )
    failed = positive_int("placement__failed")
    if failed not in (0, 1):
        raise ValueError("BMI placement failure flag must be 0 or 1")
    scale = float(scalar("placement__recovery_effect_scale"))
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("BMI recovery effect scale must be between zero and one")
    cost_quantity = float(scalar("placement__cost_quantity_ha"))
    if not math.isfinite(cost_quantity) or cost_quantity < 0:
        raise ValueError("BMI design cost quantity must be finite and nonnegative")

    if family == "wetland":
        area = float(scalar("placement__wetland_area_ha"))
        ratio = float(scalar("placement__catchment_area_ratio"))
        if not math.isfinite(area) or area <= 0 or not math.isfinite(ratio) or ratio < 0:
            raise ValueError("BMI wetland area and catchment ratio are invalid")
        metadata = ((OUTPUT_WETLAND_AREA, area), (OUTPUT_CATCHMENT_RATIO, ratio),
                    (OUTPUT_IMPACTED_PIDS, ",".join(str(ctx.parcel_ids[i]) for i in affected)
                     if len(affected) > 1 else ""))
    elif family == "grassed":
        length = float(scalar("placement__buffer_length_m"))
        area = float(scalar("placement__buffer_area_ha"))
        portion = float(scalar("placement__portion_treated"))
        if len(treated) != 1 or any(not math.isfinite(v) or v < 0 for v in (length, area, portion)):
            raise ValueError("BMI grassed waterway design is invalid")
        if portion != treated[0][1]:
            raise ValueError("BMI grassed waterway portion and treatment fraction disagree")
        metadata = ((OUTPUT_LINEAR_LENGTH, length), (OUTPUT_BUFFER_AREA, area),
                    (OUTPUT_PORTION_TREATED, portion))
    elif family == "infield":
        if treated != ((idx, 1.0),):
            raise ValueError("BMI in-field BMP must cover exactly its host parcel")
        metadata = ()
    else:
        raise ValueError(f"Unsupported BMI placement practice family {family!r}")

    recovery = np.asarray(values["placement__recorded_recovery_gains"], dtype=float)
    if recovery.ndim != 1 or not np.all(np.isfinite(recovery)) or np.any(recovery < 0) or np.any(recovery > 1):
        raise ValueError("BMI recorded AFO recovery gains must lie in [0, 1]")
    # A replayed placement already has the complete cost and recovery-gain
    # record; cost is transferred after response through a separate variable.
    return RealizedBMPPlacement(
        choice=PlacementChoice(idx, str(ctx.parcel_ids[idx]), cps),
        design=BMPDesign(family, metadata, treated, cost_quantity),
        efficiency_by_pollutant=eff_rows, failed=bool(failed),
        recovery_effect_scale=scale,
        recovery_gains=tuple(float(v) for v in recovery) if len(recovery) else None,
    )
