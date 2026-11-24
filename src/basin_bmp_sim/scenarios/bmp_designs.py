"""Model-independent, replayable BMP placement designs.

The existing wetland and grassed-waterway design heuristics are retained
exactly. No water-quality state is consulted here. Only geometry, upstream
connectivity, configuration, and the supplied stochastic sampler are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from basin_bmp_sim.bmp.practice_registry import response_family
from basin_bmp_sim.domain.constants import (
    CFG_BUFFER_DEPTH_FT, CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
    FT_TO_M, OUTPUT_BUFFER_AREA, OUTPUT_CATCHMENT_RATIO,
    OUTPUT_IMPACTED_PIDS, OUTPUT_LINEAR_LENGTH, OUTPUT_PORTION_TREATED,
    OUTPUT_WETLAND_AREA,
)
from basin_bmp_sim.domain.parcel_graph import iter_full_upstream_by_distance


@dataclass(frozen=True)
class BMPDesign:
    """Physical implementation, independent of pollutant model state.

    ``treated`` uses indices into the full hydrologic parcel universe, not
    the BMP-eligible selection sub-universe. ``metadata`` contains existing
    BMP output fields, so model responses do not have to sample or infer them.
    """
    family: str
    metadata: tuple[tuple[str, float | str | None], ...]
    treated: tuple[tuple[int, float], ...]
    cost_quantity_ha: float

    def record_fields(self) -> dict[str, float | str | None]:
        return dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "metadata": [list(p) for p in self.metadata],
                "treated": [list(p) for p in self.treated],
                "cost_quantity_ha": self.cost_quantity_ha}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BMPDesign":
        return cls(
            family=str(value["family"]),
            metadata=tuple((str(k), v) for k, v in value["metadata"]),
            treated=tuple((int(i), float(f)) for i, f in value["treated"]),
            cost_quantity_ha=float(value["cost_quantity_ha"]),
        )


def sample_bmp_design(ctx: Any, parcel_idx: int, cps: int) -> BMPDesign:
    """Sample one BMP's design, preserving original drawing order and rules.

    This function must be called *after* the BMP efficiency/failure draws.
    The caller must not pre-generate the entire scenario with the legacy RNG.
    """
    family = response_family(cps)
    area_field_ha = float(ctx.parcel_area_ha[parcel_idx])
    deterministic = ctx.cfg.get(CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS, False)

    if family == "wetland":
        if deterministic:
            wet_area = 0.20 * area_field_ha
        else:
            wet_area_stats = {"min": 0.1, "p25": 0.4, "p50": 0.81, "p75": 2.0, "max": 4.0}
            wet_area = ctx._sample_from_stats(stats=wet_area_stats, kind=None)
            wet_area = min(wet_area, area_field_ha)
        ctx.logger.verbose(
            f"selected wetland area of {wet_area:.2f} ha in parcel idx={parcel_idx} of area={area_field_ha:.2f} ha"
        )
        if deterministic:
            cat_ratio = 4.0
        else:
            ratio_stats = {"min": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "max": 100.0}
            cat_ratio = float(ctx._sample_from_stats(stats=ratio_stats, kind=None))

        impacted_idxs = [parcel_idx]
        impacted_area_ha = area_field_ha if deterministic else wet_area * (1.0 + cat_ratio)
        total_available_ha = float(ctx.parcel_area_ha[parcel_idx])
        for up_idx in iter_full_upstream_by_distance(parcel_idx, ctx.parcel_up_idxs, ctx.parcel_ids):
            if total_available_ha >= impacted_area_ha:
                break
            if up_idx not in impacted_idxs:
                impacted_idxs.append(up_idx)
                total_available_ha += float(ctx.parcel_area_ha[up_idx])
                ctx.logger.verbose(
                    f"added upgradient parcel (pid={ctx.parcel_ids[up_idx]}) with area "
                    f"{ctx.parcel_area_ha[up_idx]:.2f} ha to wetland-impacted parcels"
                )
                if total_available_ha >= impacted_area_ha:
                    break

        if impacted_area_ha > total_available_ha:
            ctx.logger.verbose(
                f"total available upgradient area ({total_available_ha:.2f} ha) < impacted area "
                f"(wetland+catchment) ({impacted_area_ha:.2f} ha)"
            )
            impacted_area_ha = total_available_ha
            cat_ratio = (impacted_area_ha - wet_area) / max(wet_area, 1e-9)
            ctx.logger.verbose(
                f"reduced impacted area to {impacted_area_ha:.2f} ha and catchment ratio to {cat_ratio:.2f}"
            )
        remaining = impacted_area_ha
        treated = []
        for p_idx in impacted_idxs:
            parcel_area_ha = float(ctx.parcel_area_ha[p_idx])
            if remaining <= 0:
                fraction = 0.0
            elif remaining < parcel_area_ha:
                fraction = remaining / parcel_area_ha
            else:
                fraction = 1.0
            treated.append((int(p_idx), float(fraction)))
            remaining -= parcel_area_ha
        return BMPDesign(
            family="wetland",
            metadata=((OUTPUT_WETLAND_AREA, float(wet_area)),
                      (OUTPUT_CATCHMENT_RATIO, float(cat_ratio)),
                      (OUTPUT_IMPACTED_PIDS, ",".join(
                          [ctx.parcel_ids[idx] for idx in impacted_idxs]
                          if len(impacted_idxs) > 1 else []))),
            treated=tuple(treated), cost_quantity_ha=float(wet_area),
        )

    if family == "grassed":
        perim_m = float(ctx.parcel_perim_m[parcel_idx])
        if deterministic:
            perim_frac = 0.25
        else:
            frac_stats = {"min": 0.1, "max": 0.3, "mean": 0.2}
            perim_frac = ctx._sample_from_stats(stats=frac_stats, kind=None)
        length_m = perim_m * perim_frac
        ctx.logger.verbose(
            f"grassed buffer length={length_m:.2f} m from fraction={perim_frac:.2f} of perimeter={perim_m:.2f} m"
        )
        depth_ft = float(ctx.cfg[CFG_BUFFER_DEPTH_FT])
        depth_m = depth_ft * FT_TO_M
        area_ha = (length_m * depth_m) / 10000.0
        ctx.logger.verbose(
            f"grassed buffer depth={depth_ft:.2f} ft ({depth_m:.2f} m), area={area_ha:.4f} ha"
        )
        if deterministic:
            fraction = 0.25
        else:
            frac_stats = {"min": 0.2, "max": 0.4, "mean": 0.3}
            fraction = ctx._sample_from_stats(stats=frac_stats, kind=None)
        return BMPDesign(
            family="grassed",
            metadata=((OUTPUT_LINEAR_LENGTH, float(length_m)),
                      (OUTPUT_BUFFER_AREA, float(area_ha)),
                      (OUTPUT_PORTION_TREATED, float(fraction))),
            treated=((int(parcel_idx), float(fraction)),),
            cost_quantity_ha=float(area_ha),
        )

    return BMPDesign(
        family="infield", metadata=(), treated=((int(parcel_idx), 1.0),),
        cost_quantity_ha=area_field_ha,
    )
