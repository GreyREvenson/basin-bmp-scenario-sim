"""Mass-based manure loading and annual export."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from basin_bmp_sim.science.hydrology.runoff_plet_curve_number import plet_annual_surface_runoff_in

from .shared import _finite, _draw, _nan
from .allocation import _select_manure_fields
from .feedlot import _feedlot_runoff_kg, STAT_FEEDLOT_RAIN_DAYS, STAT_FEEDLOT_RAIN_CORRECTION, STAT_FEEDLOT_RUNOFF_DAY_FRACTION
from .excretion import animal_production


def manure_surface_export_kg(
    *,
    applied_n_kg: float,
    applied_p_kg: float,
    tn_surface_export_fraction: float,
    tp_surface_export_fraction: float,
    runoff_adjusted_export: bool = False,
    runoff_mm: float | None = None,
    reference_runoff_mm: float | None = None,
) -> dict[str, float]:
    """Same-year manure-attributable surface TN/TP exports (kg/year).

    Default: applied nutrient mass times its annual export fraction.
    Opt-in: multiply each fraction by annual runoff/reference runoff and cap
    at unity. This is an empirical screening approximation, not APLE.
    """
    n = _finite(applied_n_kg, "applied_n_kg")
    p = _finite(applied_p_kg, "applied_p_kg")
    n_fraction = _finite(tn_surface_export_fraction, "tn_surface_export_fraction", 0, 1)
    p_fraction = _finite(tp_surface_export_fraction, "tp_surface_export_fraction", 0, 1)
    if not isinstance(runoff_adjusted_export, bool):
        raise ValueError("runoff_adjusted_export must be boolean")
    if runoff_adjusted_export:
        if runoff_mm is None or reference_runoff_mm is None:
            raise ValueError("Runoff adjustment requires runoff_mm and reference_runoff_mm")
        runoff = _finite(runoff_mm, "runoff_mm")
        reference = _finite(reference_runoff_mm, "reference_runoff_mm", low=1e-12)
        factor = runoff / reference
        n_fraction = min(1., n_fraction * factor)
        p_fraction = min(1., p_fraction * factor)
    return {"TN": n * n_fraction, "TP": p * p_fraction}


def mass_afo_realization(
    ctx: Any, *, state: Any = None
) -> tuple[list[dict], list[dict]]:
    """Add AFO loads to PLET or statistical pathway arrays, once per scenario.

    Loads are kg/ha/year in current_pathway_load_rates. The manure allocation
    ledger records kg, and is unaffected by subsequent BMP applications.
    """
    inputs = ctx.afo_inputs
    if not inputs or inputs.get("method") != "mass_based":
        return [], []
    pids = list(map(str, ctx.parcel_ids))
    idx_map = {pid: i for i, pid in enumerate(pids)}
    areas = inputs["area_ha"]
    ops = inputs["operations"]
    # Draw animal inventory once per realization; feedlot density and manure
    # production must use the same realized animal counts.
    realized_animals = []
    for animal in inputs["animals"]:
        sampled = dict(animal)
        sampled["count"] = _draw(ctx, animal, "count")
        for key in list(sampled):
            if key.startswith("count_"):
                del sampled[key]
        realized_animals.append(sampled)
    footprint = {pid: 0. for pid in pids}
    for op in ops:
        footprint[str(op["pid"])] += float(op["feedlot_area_ha"])
    if state is not None:
        eligible = {
            pid: areas[pid] - footprint[pid]
            for pid in pids
            if state.parameters[idx_map[pid]]["land_cover"] in {"cropland", "pastureland"}
            and areas[pid] - footprint[pid] > 1e-12
        }
        surface_idx = 0
    else:
        eligible = {
            pid: areas[pid] - footprint[pid]
            for pid in pids
            if inputs["eligible"].get(pid, False)
            and areas[pid] - footprint[pid] > 1e-12
        }
        surface = inputs["surface_pathway"]
        if surface not in ctx.pathway_names:
            raise ValueError(f"AFO surface pathway {surface!r} not present in statistical pathways")
        surface_idx = ctx.pathway_names.index(surface)
    assignments = _select_manure_fields(ctx, inputs, eligible)
    by_op: dict[str, list[dict]] = {}
    for rec in assignments:
        by_op.setdefault(rec["afo_id"], []).append(rec)
    n_parcels, n_pol, n_paths = ctx.current_pathway_load_rates.shape
    feedlot = np.zeros((n_parcels, n_pol, n_paths), float)
    manure = np.zeros_like(feedlot)
    ledger: list[dict] = []
    # Sparse per-operation field sources allow later feedlot BMPs to change
    # manure availability without erasing earlier receiving-field BMP effects.
    operation_state: dict[str, dict[str, Any]] = {}
    sources_by_parcel: dict[int, list[dict[str, Any]]] = {}
    management = {str(rec["afo_id"]): rec for rec in inputs["management"]}
    excretion_cache: dict[tuple[str, str], float] = {}
    for op in sorted(ops, key=lambda row: str(row["afo_id"])):
        op_id = str(op["afo_id"])
        mgt = management[op_id]
        produced = animal_production(ctx, realized_animals, op_id, excretion_cache=excretion_cache)
        collection = _draw(ctx, mgt, "collection_fraction", default=1, fraction=True)
        export_fraction = _draw(ctx, mgt, "export_fraction", default=0, fraction=True)
        available_full = {
            "TN": produced["TN"] * (1 - export_fraction)
            * _draw(ctx, mgt, "n_retention_fraction", default=1, fraction=True),
            "TP": produced["TP"] * (1 - export_fraction)
            * _draw(ctx, mgt, "p_retention_fraction", default=1, fraction=True),
        }
        available = {pol: mass * collection for pol, mass in available_full.items()}
        operation_state[op_id] = dict(host_pid=str(op["pid"]),
                                      initial_collection=collection,
                                      current_collection=collection,
                                      available_full=available_full,
                                      sources=[])
        tn_export_fraction = _draw(ctx, mgt, "tn_surface_export_fraction", fraction=True)
        tp_export_fraction = _draw(ctx, mgt, "tp_surface_export_fraction", fraction=True)
        adjust_for_runoff = inputs.get("runoff_adjusted_export", False)
        reference_runoff = inputs.get("reference_runoff_mm")
        chosen = by_op[op_id]
        custom_weights = [rec.get("weight") for rec in chosen]
        if any(weight is not None for weight in custom_weights):
            if not all(weight is not None for weight in custom_weights):
                raise ValueError(f"{op_id}: all receiving parcels must have a weight if one does")
            weights = [float(weight) for weight in custom_weights]
        else:
            weights = [float(eligible[rec["receiving_pid"]]) for rec in chosen]
        if sum(weights) <= 0:
            raise ValueError(f"{op_id}: allocated field weights must sum positive")
        for rec, weight in zip(chosen, weights):
            pid = rec["receiving_pid"]
            idx = idx_map[pid]
            frac = weight / sum(weights)
            applied = {key: total * frac for key, total in available.items()}
            # For the default method, manure export needs no runoff input.
            # No artificial zero-runoff constraint is imposed in that mode.
            runoff_mm = None
            if adjust_for_runoff:
                runoff_mm, _, _ = _hydrology(inputs, pid, state, idx, ctx)
            elif state is not None:
                # Observed PLET runoff is used only for reporting and an
                # equivalent concentration, never as a factor in Option A.
                runoff_mm = plet_annual_surface_runoff_in(state.parameters[idx])[3] * 25.4
            exported_full = manure_surface_export_kg(
                applied_n_kg=available_full["TN"] * frac,
                applied_p_kg=available_full["TP"] * frac,
                tn_surface_export_fraction=tn_export_fraction,
                tp_surface_export_fraction=tp_export_fraction,
                runoff_adjusted_export=adjust_for_runoff,
                runoff_mm=runoff_mm,
                reference_runoff_mm=reference_runoff,
            )
            exported = {pol: val * collection for pol, val in exported_full.items()}
            for pol_idx, pol in enumerate(ctx.pollutants):
                pol_upper = str(pol).upper()
                full_rate = exported_full.get(pol_upper, 0.) / areas[pid]
                add = full_rate * collection
                manure[idx, pol_idx, surface_idx] += add
                if pol_upper in ("TN", "TP"):
                    source = dict(op_id=op_id, parcel_idx=idx, pol_idx=pol_idx,
                                  path_idx=surface_idx, full_rate=full_rate,
                                  current_rate=add)
                    operation_state[op_id]["sources"].append(source)
                    sources_by_parcel.setdefault(idx, []).append(source)
                if state is not None and runoff_mm > 0 and pol_upper in ("TN", "TP"):
                    # 1 mm over 1 hectare = 10,000 L; kg/ha -> mg/L = *100/mm.
                    state.concentrations[idx][pol_upper] += 100 * add / runoff_mm
            ledger.append(dict(
                **rec,
                animal_n_produced_kg=produced["TN"],
                animal_p_produced_kg=produced["TP"],
                operation_available_tn_kg=available["TN"],
                operation_available_tp_kg=available["TP"],
                tn_applied_kg=applied["TN"],
                tp_applied_kg=applied["TP"],
                tn_exported_kg=exported["TN"],
                tp_exported_kg=exported["TP"],
                runoff_adjusted_export=adjust_for_runoff,
                reference_runoff_mm=reference_runoff,
                runoff_mm=runoff_mm,
                tn_application_kg_ha=applied["TN"] / eligible[pid],
                tp_application_kg_ha=applied["TP"] / eligible[pid],
                tn_surface_export_fraction=tn_export_fraction,
                tp_surface_export_fraction=tp_export_fraction,
                tn_effective_export_fraction=(exported["TN"] / applied["TN"] if applied["TN"] else 0.),
                tp_effective_export_fraction=(exported["TP"] / applied["TP"] if applied["TP"] else 0.),
            ))
        # Feedlot source is separate from manure-field loading.
        host = str(op["pid"])
        host_idx = idx_map[host]
        if state is not None:
            direct_kg = _feedlot_runoff_kg(
                op, realized_animals, state.parameters[host_idx], ctx.pollutants
            )
        else:
            _, host_precip_mm, rain_days = _hydrology(
                inputs, host, state, host_idx, ctx
            )
            # Reuse exactly the PLET feedlot CN and concentration calculations
            # already used in PLET/RUSLE mode; no independent runoff input.
            direct_kg = _feedlot_runoff_kg(
                op, realized_animals,
                {"annual_precip_in": host_precip_mm / 25.4,
                 "rain_days": STAT_FEEDLOT_RAIN_DAYS if rain_days is None else rain_days,
                 "rain_correction_fraction": STAT_FEEDLOT_RAIN_CORRECTION,
                 "runoff_day_fraction": STAT_FEEDLOT_RUNOFF_DAY_FRACTION},
                ctx.pollutants,
            )
        for pol_idx, pol in enumerate(ctx.pollutants):
            feedlot[host_idx, pol_idx, surface_idx] += (
                direct_kg.get(str(pol).upper(), 0.) / areas[host]
            )
    # Baseline over the AFO footprint is displaced, not counted twice. Manure
    # loads are already absolute parcel area-normalized and must not be scaled.
    baseline = ctx.current_pathway_load_rates
    for pid, footprint_ha in footprint.items():
        if footprint_ha > 0:
            baseline[idx_map[pid], :, :] *= 1 - footprint_ha / areas[pid]
    baseline += manure + feedlot
    if state is not None:
        state.baseline_pathway_load_rates = baseline.copy()
        state.baseline_concentrations = [dict(row) for row in state.concentrations]
    ctx.afo_bmp_operation_state = operation_state
    ctx.afo_bmp_sources_by_parcel = sources_by_parcel
    effect_index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for effect in sorted(inputs.get("feedlot_manure_effects", []),
                         key=lambda rec: str(rec["afo_id"])):
        host = operation_state[str(effect["afo_id"])]["host_pid"]
        effect_index.setdefault((host, int(effect["cps"])), []).append(effect)
    ctx.afo_bmp_effects_by_host_cps = effect_index
    ctx.afo_bmp_effect_records = []
    ctx.current_afo_feedlot_pathway_rates = feedlot
    ctx.current_afo_manure_pathway_rates = manure
    ctx.baseline_afo_feedlot_pathway_rates = feedlot.copy()
    ctx.baseline_afo_manure_pathway_rates = manure.copy()
    return assignments, ledger


def _hydrology(
    inputs: Mapping, pid: str, state: Any, idx: int, ctx: Any
) -> tuple[float, float, float | None]:
    """Runoff and actual (not runoff-day-corrected) precipitation, in mm/yr."""
    if state is None:
        supplied = inputs["hydrology"].get(pid)
        if not supplied:
            raise ValueError(f"Statistical AFO parcel {pid} is missing runoff/precipitation inputs")
        # Inherited hydrology draws are shared, and each parcel's draw is reused
        # when multiple operations use that field.
        cache = getattr(ctx, "_afo_hydro_cache", None)
        if cache is None:
            cache = {}
            ctx._afo_hydro_cache = cache
        key = id(supplied)
        if key not in cache:
            def optional_parameter_present(field):
                return any(
                    (column == field or column.startswith(field + "_"))
                    and value is not None and not _nan(value)
                    for column, value in supplied.items()
                )

            def optional_draw(field):
                return _draw(ctx, supplied, field) if optional_parameter_present(field) else None
            cache[key] = (
                optional_draw("annual_runoff_mm"),
                optional_draw("annual_precip_mm"),
                _draw(ctx, supplied, "rain_days", minimum=1., maximum=366.)
                if optional_parameter_present("rain_days") else None,
            )
        return cache[key]
    params = state.parameters[idx]
    runoff_mm = plet_annual_surface_runoff_in(params)[3] * 25.4
    rain = params.get("annual_precip_in")
    if rain is None:
        correction = float(params.get("rain_correction_fraction", 1.0))
        rain = (
            float(params["avg_rain_in"])
            * float(params["rain_days"])
            * float(params["runoff_day_fraction"])
            / correction
        )
    return runoff_mm, float(rain) * 25.4, None
