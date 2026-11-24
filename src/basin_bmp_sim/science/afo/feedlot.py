"""Feedlot runoff, baseline loads and implementation cost quantities."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from basin_bmp_sim.domain.constants import (
    AFO_FEEDLOT_CURVE_NUMBERS as _FEEDLOT_CN,
    AFO_FEEDLOT_MAX_CONCENTRATIONS as _FEEDLOT_MAX_CONC,
    INCH_OVER_HA_TO_LITERS,
)
from basin_bmp_sim.science.hydrology.runoff_plet_curve_number import plet_annual_surface_runoff_in

# Screening assumptions for statistical-mode feedlot rainfall, not PLET defaults.
STAT_FEEDLOT_RAIN_DAYS = 120.0
STAT_FEEDLOT_RAIN_CORRECTION = 0.90
STAT_FEEDLOT_RUNOFF_DAY_FRACTION = 0.50

from .shared import _draw, _finite, _sample_stats
from .allocation import _select_manure_fields


def feedlot_bmp_cost_quantities(inputs: Mapping[str, Any], host_pid: str, cps: int) -> tuple[float, float | None]:
    """Return treated feedlot area and optional configured length for one BMP.

    Multiple AFOs sharing a host are treated by one parcel/BMP placement, so
    their operation-level dimensions are added. A missing area override uses
    the actual AFO footprint. Length is never inferred from parcel area.
    """
    operations = [op for op in inputs.get("operations", ()) if str(op["pid"]) == str(host_pid)]
    if not operations:
        raise ValueError(f"Feedlot BMP CPS {cps} host pid {host_pid} has no AFO operations")
    dimensions = inputs.get("feedlot_cost_dimensions", {})
    area = 0.
    length = 0.
    all_lengths = True
    for op in operations:
        dims = dimensions.get((str(op["afo_id"]), int(cps)), {})
        area += float(dims.get("cost_area_ha", op["feedlot_area_ha"]))
        if "cost_length_m" not in dims:
            all_lengths = False
        else:
            length += float(dims["cost_length_m"])
    return area, (length if all_lengths else None)


def _feedlot_concentration(op: Mapping[str, Any], animals: Sequence[Mapping[str, Any]], pol: str) -> float:
    """PLET feedlot mg/L from pollutant-specific equivalent animal density."""
    if pol not in _FEEDLOT_MAX_CONC:
        return 0.
    area_ac = float(op["feedlot_area_ha"]) / 0.40468564224
    eau = sum(float(animal["count"]) * float(animal[f"factor_{pol.lower()}"])
              for animal in animals if str(animal["afo_id"]) == str(op["afo_id"]))
    return min(eau / area_ac, 100.) / 100. * _FEEDLOT_MAX_CONC[pol]


def _feedlot_runoff_kg(op: Mapping[str, Any], animals: Sequence[Mapping[str, Any]],
                       parameters: Mapping[str, Any], pollutants: Sequence[str]) -> dict[str, float]:
    """PLET feedlot CN (91–94), EAU density, manure pack, concentration and load."""
    paved = float(op["paved_pct"])
    cn = _FEEDLOT_CN[min(int(paved // 25), 3)]
    forcing = dict(parameters)
    forcing["cn"] = cn
    # PLET's feedlot uses same event rainfall/day forcing as other land uses.
    _, _, _, annual_runoff_in = plet_annual_surface_runoff_in(forcing)
    area_ha = float(op["feedlot_area_ha"])
    area_ac = area_ha / 0.40468564224
    outputs = {}
    for pollutant in pollutants:
        pol = str(pollutant).upper()
        if pol not in _FEEDLOT_MAX_CONC:
            continue
        concentration = _feedlot_concentration(op, animals, pol)
        # mg/L * L/ha * ha -> kg/yr, using same unit handling as PLET/RUSLE.
        outputs[pol] = annual_runoff_in * INCH_OVER_HA_TO_LITERS * area_ha * concentration / 1_000_000.
    return outputs


def apply_afo_baseline(ctx: Any, state: Any) -> list[dict[str, Any]]:
    """Apply operation runoff and manure-exposure to sampled PLET pathway loads.

    Called only for enabled AFO runs, once per worker before BMPs. The sampled
    manure assignment is shared by that worker's baseline and BMP trajectory.
    """
    inputs = getattr(ctx, "afo_inputs", None)
    if not inputs:
        return []
    pid_to_index = {pid: k for k, pid in enumerate(state.parcel_ids)}
    operations = inputs["operations"]
    # Legacy concentration mode uses each realized animal count for both
    # feedlot concentration and runoff mass. Fixed counts remain deterministic.
    animals = []
    for animal in inputs["animals"]:
        realized = dict(animal)
        realized["count"] = _draw(ctx, animal, "count")
        animals.append(realized)
    fields = inputs["fields"]
    areas = inputs["area_ha"]
    geometry = inputs["geometry"]
    # The sampled parcel PLET classification decides eligibility, not a static
    # shapefile field. Purely urban/forest parcels cannot receive field manure.
    agricultural = {pid for pid, idx in pid_to_index.items()
                    if state.parameters[idx]["land_cover"] in {"cropland", "pastureland"}}
    feedlot_area = {pid: 0. for pid in pid_to_index}
    for op in operations:
        feedlot_area[str(op["pid"])] += float(op["feedlot_area_ha"])
    eligible_area = {pid: areas[pid] - feedlot_area[pid] for pid in agricultural
                     if areas[pid] - feedlot_area[pid] > 1e-10}
    assignments = _select_manure_fields(ctx, inputs, eligible_area)
    manure_months = {}
    for rec in assignments:
        pid = rec["receiving_pid"]
        manure_months[pid] = max(manure_months.get(pid, 0.), rec["manure_months"])
    # An increment is a standalone nonnegative concentration added to each
    # parcel's sampled non-manured concentration. It is deliberately NOT an
    # absolute manured concentration, and is drawn independently of the base.
    # Multiple AFOs on the same field do not create repeated full increments.
    manure_increment_rows = {
        (r["land_cover"], r["pollutant"]): r
        for r in inputs["manure_concentration_increments"]
    }
    drawn_increments = {}
    field_component = state.pathway_load_rates.copy()
    manure_component = np.zeros_like(state.pathway_load_rates)
    for pid, months in manure_months.items():
        if months == 0.:
            continue
        idx = pid_to_index[pid]
        cover = state.parameters[idx]["land_cover"]
        for pol_idx, pol in enumerate(ctx.pollutants):
            pol = str(pol).upper()
            if pol not in {"TN", "TP", "BOD"}:
                continue
            key = (cover, pol)
            if key not in manure_increment_rows:
                raise ValueError(f"Missing manure concentration increment for {cover}/{pol}")
            if key not in drawn_increments:
                increment = _sample_stats(ctx, manure_increment_rows[key], kind="nonnegative")
                drawn_increments[key] = _finite(
                    increment, f"sampled manure concentration increment {cover}/{pol}"
                )
            base = float(state.concentrations[idx][pol])
            fraction = float(months) / 12.
            # The parcel-average concentration reflects only the eligible
            # non-feedlot agricultural acreage of this receiving parcel.
            area_fraction = eligible_area[pid] / areas[pid]
            incremental_concentration = fraction * drawn_increments[key] * area_fraction
            state.concentrations[idx][pol] = base + incremental_concentration
            runoff_in = plet_annual_surface_runoff_in(state.parameters[idx])[3]
            increment = runoff_in * INCH_OVER_HA_TO_LITERS * incremental_concentration / 1_000_000.
            state.pathway_load_rates[idx, pol_idx, 0] += increment * float(state.parameters[idx][f"load_multiplier_{pol.lower()}"])
            # Field component is later restricted to non-feedlot acreage.
            field_component[idx, pol_idx, 0] += (increment / area_fraction) * float(state.parameters[idx][f"load_multiplier_{pol.lower()}"])
            manure_component[idx, pol_idx, 0] += (increment / area_fraction) * float(state.parameters[idx][f"load_multiplier_{pol.lower()}"])
    # Separate direct feedlot source so feedlot-only and agricultural BMPs do
    # not automatically treat one another's loads.
    feedlot_component = np.zeros_like(state.pathway_load_rates)
    for op in operations:
        pid = str(op["pid"])
        idx = pid_to_index[pid]
        loads = _feedlot_runoff_kg(op, animals, state.parameters[idx], ctx.pollutants)
        for pol_idx, pol in enumerate(ctx.pollutants):
            pol = str(pol).upper()
            kg_ha = loads.get(pol, 0.) / areas[pid]
            feedlot_component[idx, pol_idx, 0] += kg_ha
    for pid, fraction_area in feedlot_area.items():
        if fraction_area <= 0.:
            continue
        idx = pid_to_index[pid]
        fraction = fraction_area / areas[pid]
        # Displace background PLET loads over the feedlot footprint. The
        # feedlot groundwater pathway is not modeled; avoid double counting.
        field_component[idx, :, :] *= (1. - fraction)
        manure_component[idx, :, :] *= (1. - fraction)
    state.pathway_load_rates[:, :, :] = field_component + feedlot_component
    state.baseline_pathway_load_rates = state.pathway_load_rates.copy()
    state.baseline_concentrations = [dict(x) for x in state.concentrations]
    ctx.current_afo_feedlot_pathway_rates = feedlot_component
    ctx.current_afo_manure_pathway_rates = manure_component
    ctx.baseline_afo_feedlot_pathway_rates = feedlot_component.copy()
    ctx.baseline_afo_manure_pathway_rates = manure_component.copy()
    return assignments
