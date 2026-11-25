"""Explicit, lossless numeric BMI contract for a *pending* BMP placement.

The exchange uses parcel indices in the current hydrologic universe and
pollutant/pathway indices from the validated, approved formulation.  No PIDs,
models, loggers, or native placement objects cross the BMI boundary.

Scientific methods, cost draws, and sampling remain in their original owners.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from basin_bmp_sim.bmp.practice_registry import response_family
from basin_bmp_sim.domain.constants import (
    OUTPUT_BUFFER_AREA, OUTPUT_CATCHMENT_RATIO, OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH, OUTPUT_PORTION_TREATED, OUTPUT_WETLAND_AREA,
)
from basin_bmp_sim.scenarios.bmp_designs import BMPDesign
from basin_bmp_sim.scenarios.generator import PlacementChoice, RealizedBMPPlacement

from basin_bmp_sim.interfaces.annual_model_contract import AnnualBMPPlacement, PLACEMENT_FIELDS
from basin_bmp_sim.interfaces.exchange_variables import PLACEMENT_OUTPUT_VARIABLES, numeric_vector


def placement_arrays(placement: RealizedBMPPlacement, ctx) -> dict[str, np.ndarray]:
    """Encode one pending design/response draw with no numerical rounding."""
    d = placement.design
    metadata = d.record_fields()
    efficiencies = [float(value) for pol in placement.efficiency_by_pollutant for _, value in pol]
    expected_paths = tuple(str(p) for p in ctx.pathway_names)
    if len(placement.efficiency_by_pollutant) != len(ctx.pollutants) or any(
        tuple(k for k, _ in pol) != expected_paths for pol in placement.efficiency_by_pollutant
    ):
        raise ValueError("BMI placement efficiency pathway ordering disagrees with model formulation")
    fields = {
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

    return {v.name: numeric_vector(fields[v.name], v.dtype) for v in PLACEMENT_OUTPUT_VARIABLES}


def placement_from_arrays(values: Mapping[str, np.ndarray], ctx) -> RealizedBMPPlacement:
    """Rebuild the exact native placement object from BMI numerical values.

    Reject mismatched array dimensions, invalid parcel indices, impossible
    practice geometry, and unsupported CPS *before* scientific execution.
    """
    request = AnnualBMPPlacement.from_bmi(values, ctx)
    cps, idx = request.cps, request.host_parcel_index
    family = response_family(cps)
    treated = tuple(zip(request.affected_indices, request.affected_fractions))
    eff_rows = tuple(
        tuple((str(path), float(v)) for path, v in zip(ctx.pathway_names, row))
        for row in request.efficiency
    )

    if family == "wetland":
        area = request.wetland_area_ha
        ratio = request.catchment_area_ratio
        if area <= 0:
            raise ValueError("BMI wetland area and catchment ratio are invalid")
        metadata = ((OUTPUT_WETLAND_AREA, area), (OUTPUT_CATCHMENT_RATIO, ratio),
                    (OUTPUT_IMPACTED_PIDS, ",".join(str(ctx.parcel_ids[i]) for i in request.affected_indices)
                     if len(request.affected_indices) > 1 else ""))
    elif family == "grassed":
        length = request.buffer_length_m
        area = request.buffer_area_ha
        portion = request.portion_treated
        if len(treated) != 1:
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

    # A replayed placement already has the complete cost and recovery-gain
    # record; cost is transferred after response through a separate variable.
    return RealizedBMPPlacement(
        choice=PlacementChoice(idx, str(ctx.parcel_ids[idx]), cps),
        design=BMPDesign(family, metadata, treated, request.cost_quantity_ha),
        efficiency_by_pollutant=eff_rows, failed=request.failed,
        recovery_effect_scale=request.recovery_effect_scale,
        recovery_gains=request.recorded_recovery_gains or None,
    )
