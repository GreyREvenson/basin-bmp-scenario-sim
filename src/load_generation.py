"""Create parcel pollution numbers and update them when BMPs are applied.

This file turns input tables into yearly pollutant loads for each parcel.
It also updates those loads when a BMP changes runoff or other settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


INCH_OVER_HA_TO_LITERS = 254_000.0
TON_PER_ACRE_TO_KG_PER_HA = 907.18474 / 0.40468564224
ACRES_PER_SQUARE_MILE = 640.0
PATHWAY_NAMES = ("surface", "shallow subsurface", "deep subsurface")

# Canonical parameter names used internally.  Aliases make user-authored files
# less brittle without introducing ambiguous units.
_PARAMETER_ALIASES: Dict[str, str] = {
    "annual_rainfall_in": "annual_precip_in",
    "annual_precipitation_in": "annual_precip_in",
    "ar": "annual_precip_in",
    "rdays": "rain_days",
    "rainfall_correction": "rain_correction_fraction",
    "rcor": "rain_correction_fraction",
    "rain_day_correction": "runoff_day_fraction",
    "rdcor": "runoff_day_fraction",
    "curve_number": "cn",
    "initial_abstraction_ratio": "ia_ratio",
    "alpha": "ia_ratio",
    "rusle_r": "r",
    "rusle_k": "k",
    "rusle_ls": "ls",
    "rusle_c": "c",
    "rusle_p": "p",
    "delivery_ratio": "sdr",
    "sediment_delivery_ratio": "sdr",
    "watershed_area_sqmi": "watershed_area_mi2",
    "watershed_area_sq_mi": "watershed_area_mi2",
    "soil_n_percent": "sediment_n_pct",
    "soil_p_percent": "sediment_p_pct",
    "enrichment": "enrichment_ratio",
    "infiltration_frac": "infiltration_fraction",
    "infiltration_factor": "infiltration_fraction",
    "gw_infiltration_fraction": "infiltration_fraction",
    "groundwater_infiltration_fraction": "infiltration_fraction",
    "irrigated_area_fraction": "irrigated_fraction",
    "irrigation_area_fraction": "irrigated_fraction",
    "irrigation_depth": "irrigation_depth_in",
    "irrigation_inches": "irrigation_depth_in",
    "irrigation_frequency_per_year": "irrigation_frequency",
}

_REQUIRED_PLET = (
    "annual_precip_in",
    "rain_days",
    "rain_correction_fraction",
    "runoff_day_fraction",
    "cn",
)
_REQUIRED_RUSLE = ("r", "k", "ls", "c", "p")


@dataclass
class LoadState:
    """Store each parcel's current values while one scenario is running.

    You can think of this as a live worksheet:
    - current values
    - starting values
    - and parcel order/IDs

    As BMPs are applied, this object is updated and used to recalculate loads.
    """

    parcel_ids: List[str]
    parameters: List[Dict[str, float]]
    concentrations: List[Dict[str, float]]
    groundwater_concentrations: List[Dict[str, float]]
    has_rusle: List[bool]
    pollutants: List[str]
    pathway_yields: np.ndarray
    baseline_parameters: List[Dict[str, float]] = field(default_factory=list)
    baseline_concentrations: List[Dict[str, float]] = field(default_factory=list)
    baseline_groundwater_concentrations: List[Dict[str, float]] = field(default_factory=list)
    baseline_pathway_yields: Optional[np.ndarray] = None

    @property
    def index_by_pid(self) -> Dict[str, int]:
        """Return a quick map from parcel ID to its position in the lists."""
        return {pid: i for i, pid in enumerate(self.parcel_ids)}


def canonical_parameter_name(value: Any) -> str:
    """Convert different spellings of the same input name to one standard name.

    Example: ``"curve number"``, ``"curve_number"``, and ``"cn"`` all resolve
    to ``"cn"`` so the rest of the model uses one consistent key.
    """

    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _PARAMETER_ALIASES.get(label, label)


def plet_runoff_depth_in(
    annual_precip_in: float,
    rain_days: float,
    rain_correction_fraction: float,
    runoff_day_fraction: float,
    cn: float,
    ia_ratio: float = 0.0,
) -> Tuple[float, float, float]:
    """Calculate event runoff and yearly runoff depth (in inches)."""

    annual_precip_in = max(0.0, float(annual_precip_in))
    rain_days = max(0.0, float(rain_days))
    rain_correction_fraction = float(np.clip(rain_correction_fraction, 0.0, 1.0))
    runoff_day_fraction = float(np.clip(runoff_day_fraction, 0.0, 1.0))
    cn = float(np.clip(cn, 1.0e-6, 100.0))
    ia_ratio = max(0.0, float(ia_ratio))

    runoff_days = rain_days * runoff_day_fraction
    if runoff_days <= 0.0:
        return 0.0, 0.0, 0.0

    event_rainfall = annual_precip_in * rain_correction_fraction / runoff_days
    retention = (1000.0 / cn) - 10.0
    retention = max(0.0, retention)
    initial_abstraction = ia_ratio * retention
    if event_rainfall <= initial_abstraction:
        event_runoff = 0.0
    else:
        numerator = (event_rainfall - initial_abstraction) ** 2
        denominator = event_rainfall - initial_abstraction + retention
        event_runoff = numerator / denominator if denominator > 0.0 else 0.0

    return event_rainfall, event_runoff, event_runoff * runoff_days


def plet_sediment_delivery_ratio(watershed_area_mi2: float) -> float:
    """Estimate how much sediment reaches the outlet based on watershed size."""

    area_mi2 = max(float(watershed_area_mi2), 1.0e-12)
    area_acres = area_mi2 * ACRES_PER_SQUARE_MILE
    if area_acres < 200.0:
        dr = 0.42 * area_mi2 ** (-0.125)
    else:
        dr = 0.417662 * area_mi2 ** (-0.134958) - 0.127097
    return float(np.clip(dr, 0.0, 1.0))


def rusle_sediment_yield_kg_ha(parameters: Mapping[str, float]) -> float:
    """Estimate yearly sediment load per hectare using RUSLE-style inputs."""

    if not all(name in parameters for name in _REQUIRED_RUSLE):
        return 0.0
    gross_ton_ac = 1.0
    for name in _REQUIRED_RUSLE:
        gross_ton_ac *= max(0.0, float(parameters[name]))

    if "sdr" in parameters:
        sdr = float(np.clip(parameters["sdr"], 0.0, 1.0))
    elif "watershed_area_mi2" in parameters:
        sdr = plet_sediment_delivery_ratio(parameters["watershed_area_mi2"])
    else:
        raise ValueError("RUSLE inputs require either 'sdr' or 'watershed_area_mi2'")

    sediment_multiplier = max(0.0, float(parameters.get("sediment_multiplier", 1.0)))
    delivery_multiplier = max(0.0, float(parameters.get("sediment_delivery_multiplier", 1.0)))
    return gross_ton_ac * sdr * TON_PER_ACRE_TO_KG_PER_HA * sediment_multiplier * delivery_multiplier


def plet_annual_irrigation_runoff_in(parameters: Mapping[str, float]) -> float:
    """Estimate optional irrigation runoff depth using the same CN runoff equation."""
    irrigation_depth_in = max(0.0, float(parameters.get("irrigation_depth_in", 0.0)))
    irrigation_frequency = max(0.0, float(parameters.get("irrigation_frequency", 0.0)))
    irrigated_fraction = float(np.clip(parameters.get("irrigated_fraction", 1.0), 0.0, 1.0))
    if irrigation_depth_in <= 0.0 or irrigation_frequency <= 0.0 or irrigated_fraction <= 0.0:
        return 0.0

    retention = max(0.0, (1000.0 / float(np.clip(parameters.get("cn", 100.0), 1.0e-6, 100.0))) - 10.0)
    ia_ratio = max(0.0, float(parameters.get("ia_ratio", 0.0)))
    initial_abstraction = ia_ratio * retention
    if irrigation_depth_in <= initial_abstraction:
        event_runoff = 0.0
    else:
        numerator = (irrigation_depth_in - initial_abstraction) ** 2
        denominator = irrigation_depth_in - initial_abstraction + retention
        event_runoff = numerator / denominator if denominator > 0.0 else 0.0
    return float(event_runoff * irrigation_frequency * irrigated_fraction)


def plet_annual_surface_runoff_in(parameters: Mapping[str, float]) -> Tuple[float, float, float, float]:
    """Estimate annual surface runoff depth including optional irrigation runoff."""
    event_rainfall, event_runoff, annual_storm_runoff = plet_runoff_depth_in(
        parameters["annual_precip_in"],
        parameters["rain_days"],
        parameters["rain_correction_fraction"],
        parameters["runoff_day_fraction"],
        parameters["cn"],
        parameters.get("ia_ratio", 0.0),
    )
    annual_irrigation_runoff = plet_annual_irrigation_runoff_in(parameters)
    runoff_multiplier = max(0.0, float(parameters.get("runoff_multiplier", 1.0)))
    annual_total_runoff = (annual_storm_runoff + annual_irrigation_runoff) * runoff_multiplier
    return event_rainfall, event_runoff, annual_storm_runoff, annual_total_runoff


def plet_annual_infiltration_in(parameters: Mapping[str, float]) -> float:
    """Estimate annual groundwater infiltration depth from an optional infiltration fraction."""
    infiltration_fraction = float(np.clip(parameters.get("infiltration_fraction", 0.0), 0.0, 1.0))
    annual_precip = max(0.0, float(parameters.get("annual_precip_in", 0.0)))
    infiltration = annual_precip * infiltration_fraction
    infiltration *= max(0.0, float(parameters.get("groundwater_multiplier", 1.0)))
    return float(max(0.0, infiltration))


def _stats_from_row(row: Mapping[str, Any], exclude: Iterable[str]) -> Dict[str, float]:
    """Pull only number fields used for random sampling from one table row."""
    excluded = {str(x).lower() for x in exclude}
    stats: Dict[str, float] = {}
    for key, value in row.items():
        key_l = str(key).strip().lower()
        if key_l in excluded or pd.isna(value):
            continue
        if key_l == "value" or key_l in {"mean", "sd", "std", "min", "max", "minimum", "maximum", "p0", "p100"}:
            stats[key_l] = float(value)
        elif key_l.startswith("p") and key_l[1:].isdigit():
            stats[key_l] = float(value)
    return stats


def _sample_stats(ctx: Any, stats: Mapping[str, float], *, nonnegative: bool = False) -> float:
    """Pick one value from the row's numbers, optionally forcing it to be nonnegative."""
    if "value" in stats:
        value = float(stats["value"])
    else:
        value = float(ctx._sample_from_stats(dict(stats), kind="yield" if nonnegative else None))
    return max(0.0, value) if nonnegative else value


def _rows_for_pid(table: Optional[pd.DataFrame], pid: str) -> List[pd.Series]:
    """Return the table rows that apply to one parcel.

    Rows with ``pid="*"`` act as defaults.
    Rows for the exact parcel ID override those defaults.
    """
    if table is None or table.empty:
        return []
    pids = table["pid"].astype(str)
    defaults = table[pids == "*"]
    exact = table[pids == str(pid)]
    combined = pd.concat([defaults, exact], ignore_index=True)
    if combined.empty:
        return []
    # A parcel-specific row overrides a wildcard row for the same parameter.
    combined = combined.drop_duplicates(subset=["parameter"], keep="last")
    return [row for _, row in combined.iterrows()]


def _sample_parameter_table(
    ctx: Any,
    table: Optional[pd.DataFrame],
    parcel_ids: Sequence[str],
    *,
    cache_prefix: str,
) -> List[Dict[str, float]]:
    """Build one sampled parameter set for each parcel."""
    sampled: List[Dict[str, float]] = []
    cache: Dict[Tuple[str, str, str], float] = {}
    for pid in parcel_ids:
        values: Dict[str, float] = {}
        for row in _rows_for_pid(table, str(pid)):
            parameter = canonical_parameter_name(row["parameter"])
            group_value = row.get("group_id", None)
            if group_value is None or pd.isna(group_value) or str(group_value).strip() == "":
                group_id = "__global__" if str(row.get("pid", "")) == "*" else str(pid)
            else:
                group_id = str(group_value)
            cache_key = (cache_prefix, parameter, group_id)
            if cache_key not in cache:
                stats = _stats_from_row(row, {"pid", "parameter", "group_id", "units"})
                if not stats:
                    raise ValueError(f"No value or statistics supplied for {cache_prefix} parameter '{parameter}'")
                cache[cache_key] = _sample_stats(ctx, stats, nonnegative=parameter not in {"load_delta"})
            values[parameter] = cache[cache_key]
        sampled.append(values)
    return sampled


def _sample_concentrations(ctx: Any, table: Optional[pd.DataFrame], parcel_ids: Sequence[str]) -> List[Dict[str, float]]:
    """Build one sampled concentration set for each parcel."""
    sampled: List[Dict[str, float]] = []
    cache: Dict[Tuple[str, str], float] = {}
    if table is None:
        return [{} for _ in parcel_ids]
    for pid in parcel_ids:
        values: Dict[str, float] = {}
        pids = table["pid"].astype(str)
        defaults = table[pids == "*"]
        exact = table[pids == str(pid)]
        combined = pd.concat([defaults, exact], ignore_index=True)
        if not combined.empty:
            combined = combined.drop_duplicates(subset=["pollutant"], keep="last")
        for _, row in combined.iterrows():
            pollutant = str(row["pollutant"]).strip().upper()
            group_value = row.get("group_id", None)
            if group_value is None or pd.isna(group_value) or str(group_value).strip() == "":
                group_id = "__global__" if str(row.get("pid", "")) == "*" else str(pid)
            else:
                group_id = str(group_value)
            key = (pollutant, group_id)
            if key not in cache:
                stats = _stats_from_row(row, {"pid", "pollutant", "group_id", "units"})
                if not stats:
                    raise ValueError(f"No concentration value or statistics supplied for {pid}/{pollutant}")
                cache[key] = _sample_stats(ctx, stats, nonnegative=True)
            values[pollutant] = cache[key]
        sampled.append(values)
    return sampled


def calculate_load_diagnostics(parameters: Mapping[str, float]) -> Dict[str, float]:
    """Return helpful intermediate values for checking and reporting."""

    event_rainfall, event_runoff, annual_storm_runoff, annual_runoff = plet_annual_surface_runoff_in(parameters)
    has_rusle = all(name in parameters for name in _REQUIRED_RUSLE)
    sediment = rusle_sediment_yield_kg_ha(parameters) if has_rusle else 0.0
    return {
        "event_rainfall_in": float(event_rainfall),
        "event_runoff_in": float(event_runoff),
        "annual_storm_runoff_in": float(annual_storm_runoff),
        "annual_irrigation_runoff_in": float(plet_annual_irrigation_runoff_in(parameters)),
        "annual_runoff_in": float(annual_runoff),
        "annual_infiltration_in": float(plet_annual_infiltration_in(parameters)),
        "sediment_kg_ha": float(sediment),
    }


def calculate_pathway_yields(
    parameters: Mapping[str, float],
    concentrations: Mapping[str, float],
    groundwater_concentrations: Optional[Mapping[str, float]],
    pollutants: Sequence[str],
    *,
    pathway_mode: str = "fixed_fractions",
    surface_fraction: float = 0.0,
    shallow_fraction: float = 0.0,
    groundwater_loads: bool = False,
    treat_groundwater_with_bmps: bool = False,
) -> np.ndarray:
    """Calculate parcel yields split by surface, shallow, and deep pathways."""

    missing = [name for name in _REQUIRED_PLET if name not in parameters]
    if missing:
        raise ValueError(f"PLET inputs are missing required parameters: {missing}")

    _, _, _, annual_runoff_in = plet_annual_surface_runoff_in(parameters)
    runoff_l_ha = annual_runoff_in * INCH_OVER_HA_TO_LITERS
    infiltration_l_ha = plet_annual_infiltration_in(parameters) * INCH_OVER_HA_TO_LITERS

    has_rusle = all(name in parameters for name in _REQUIRED_RUSLE)
    sediment_kg_ha = rusle_sediment_yield_kg_ha(parameters) if has_rusle else 0.0
    enrichment_ratio = max(0.0, float(parameters.get("enrichment_ratio", 2.0)))

    pathway_mode = str(pathway_mode).strip().lower()
    groundwater_concentrations = groundwater_concentrations or {}
    out = np.zeros((len(pollutants), len(PATHWAY_NAMES)), dtype=float)
    for idx, pollutant in enumerate(pollutants):
        pol = str(pollutant).upper()
        runoff_load = max(0.0, float(concentrations.get(pol, 0.0))) * runoff_l_ha / 1_000_000.0
        groundwater_load = 0.0
        if groundwater_loads and pol != "TSS":
            groundwater_load = (
                max(0.0, float(groundwater_concentrations.get(pol, 0.0))) * infiltration_l_ha / 1_000_000.0
            )
        if pol == "TSS":
            load = sediment_kg_ha if has_rusle else runoff_load
            surface_load = load
            shallow_load = 0.0
            deep_load = 0.0
        elif pol == "TN":
            sediment_fraction = max(0.0, float(parameters.get("sediment_n_pct", 0.0))) / 100.0
            load = runoff_load + sediment_kg_ha * sediment_fraction * enrichment_ratio
            surface_load = load
            shallow_load = groundwater_load if treat_groundwater_with_bmps else 0.0
            deep_load = 0.0 if treat_groundwater_with_bmps else groundwater_load
        elif pol == "TP":
            sediment_fraction = max(0.0, float(parameters.get("sediment_p_pct", 0.0))) / 100.0
            load = runoff_load + sediment_kg_ha * sediment_fraction * enrichment_ratio
            surface_load = load
            shallow_load = groundwater_load if treat_groundwater_with_bmps else 0.0
            deep_load = 0.0 if treat_groundwater_with_bmps else groundwater_load
        else:
            load = runoff_load
            surface_load = runoff_load
            shallow_load = groundwater_load if treat_groundwater_with_bmps else 0.0
            deep_load = 0.0 if treat_groundwater_with_bmps else groundwater_load

        total_load = (load + groundwater_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
        if pathway_mode == "derive_from_plet":
            out[idx, 0] = max(0.0, surface_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
            out[idx, 1] = max(0.0, shallow_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
            out[idx, 2] = max(0.0, deep_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
        else:
            fixed_surface = float(np.clip(surface_fraction, 0.0, 1.0))
            fixed_shallow = float(np.clip(shallow_fraction, 0.0, 1.0))
            fixed_deep = max(0.0, 1.0 - fixed_surface - fixed_shallow)
            out[idx, 0] = max(0.0, total_load) * fixed_surface
            out[idx, 1] = max(0.0, total_load) * fixed_shallow
            out[idx, 2] = max(0.0, total_load) * fixed_deep
    return out


def calculate_parcel_yields(
    parameters: Mapping[str, float],
    concentrations: Mapping[str, float],
    pollutants: Sequence[str],
    *,
    groundwater_concentrations: Optional[Mapping[str, float]] = None,
    pathway_mode: str = "fixed_fractions",
    surface_fraction: float = 0.0,
    shallow_fraction: float = 0.0,
    groundwater_loads: bool = False,
    treat_groundwater_with_bmps: bool = False,
) -> np.ndarray:
    """Calculate yearly pollutant load per hectare for the requested pollutants."""

    pathway_yields = calculate_pathway_yields(
        parameters,
        concentrations,
        groundwater_concentrations,
        pollutants,
        pathway_mode=pathway_mode,
        surface_fraction=surface_fraction,
        shallow_fraction=shallow_fraction,
        groundwater_loads=groundwater_loads,
        treat_groundwater_with_bmps=treat_groundwater_with_bmps,
    )
    return np.sum(pathway_yields, axis=1)


def initialize_plet_rusle_state(ctx: Any) -> Tuple[np.ndarray, LoadState]:
    """Create the starting load state and baseline parcel loads for a scenario."""

    parcel_ids = [str(pid) for pid in ctx.parcel_selection_ids]
    plet = _sample_parameter_table(ctx, ctx.plet_inputs, parcel_ids, cache_prefix="plet")
    rusle = _sample_parameter_table(ctx, ctx.rusle_inputs, parcel_ids, cache_prefix="rusle")
    concentrations = _sample_concentrations(ctx, ctx.pollutant_concentrations, parcel_ids)
    groundwater_concentrations = _sample_concentrations(ctx, getattr(ctx, "groundwater_concentrations", None), parcel_ids)

    parameters: List[Dict[str, float]] = []
    has_rusle: List[bool] = []
    yields = np.zeros((len(parcel_ids), len(ctx.pollutants)), dtype=float)
    pathway_yields = np.zeros((len(parcel_ids), len(ctx.pollutants), len(PATHWAY_NAMES)), dtype=float)
    for i, pid in enumerate(parcel_ids):
        missing_plet = [name for name in _REQUIRED_PLET if name not in plet[i]]
        if missing_plet:
            raise ValueError(
                f"PLET inputs for pid={pid} are missing required parameters: {missing_plet}"
            )

        if rusle[i]:
            missing_rusle = [name for name in _REQUIRED_RUSLE if name not in rusle[i]]
            if missing_rusle:
                raise ValueError(
                    f"RUSLE inputs for pid={pid} are incomplete; missing: {missing_rusle}"
                )
            if "sdr" not in rusle[i] and "watershed_area_mi2" not in rusle[i]:
                raise ValueError(
                    f"RUSLE inputs for pid={pid} require 'sdr' or 'watershed_area_mi2'"
                )

        for pollutant in ctx.pollutants:
            pol = str(pollutant).upper()
            if pol in {"TN", "TP"} and pol not in concentrations[i]:
                raise ValueError(
                    f"Runoff concentration for pid={pid}, pollutant={pol} is required"
                )
            if bool(getattr(ctx, "groundwater_loads", False)) and pol in {"TN", "TP"} and pol not in groundwater_concentrations[i]:
                raise ValueError(
                    f"Groundwater concentration for pid={pid}, pollutant={pol} is required when groundwater loads are enabled"
                )
            if pol == "TSS" and not rusle[i] and pol not in concentrations[i]:
                raise ValueError(
                    f"TSS for pid={pid} requires complete RUSLE inputs or a TSS concentration"
                )

        combined = dict(plet[i])
        combined.update(rusle[i])
        combined.setdefault("ia_ratio", 0.0)
        combined.setdefault("runoff_multiplier", 1.0)
        combined.setdefault("groundwater_multiplier", 1.0)
        combined.setdefault("sediment_multiplier", 1.0)
        combined.setdefault("sediment_delivery_multiplier", 1.0)
        for pol in ctx.pollutants:
            combined.setdefault(f"load_multiplier_{str(pol).lower()}", 1.0)
        parameters.append(combined)
        has_rusle.append(all(name in combined for name in _REQUIRED_RUSLE))
        pathway_yields[i, :, :] = calculate_pathway_yields(
            combined,
            concentrations[i],
            groundwater_concentrations[i],
            ctx.pollutants,
            pathway_mode=str(getattr(ctx, "pathway_mode", "fixed_fractions")),
            surface_fraction=float(getattr(ctx, "pollutant_yield_frac_surface", 0.0)),
            shallow_fraction=float(getattr(ctx, "pollutant_yield_frac_shallow", 0.0)),
            groundwater_loads=bool(getattr(ctx, "groundwater_loads", False)),
            treat_groundwater_with_bmps=bool(getattr(ctx, "load_generation", {}).get("treat_groundwater_with_bmps", False)),
        )
        yields[i, :] = np.sum(pathway_yields[i, :, :], axis=1)

    state = LoadState(
        parcel_ids=parcel_ids,
        parameters=parameters,
        concentrations=concentrations,
        groundwater_concentrations=groundwater_concentrations,
        has_rusle=has_rusle,
        pollutants=list(ctx.pollutants),
        pathway_yields=pathway_yields,
        baseline_parameters=[dict(values) for values in parameters],
        baseline_concentrations=[dict(values) for values in concentrations],
        baseline_groundwater_concentrations=[dict(values) for values in groundwater_concentrations],
        baseline_pathway_yields=pathway_yields.copy(),
    )
    return yields.copy(), state


def has_process_effects(ctx: Any, cps: int) -> bool:
    """Return True when this BMP type has any process-effect rules configured."""
    table = getattr(ctx, "bmp_parameter_effects", None)
    if table is None or table.empty:
        return False
    return bool((table["cps"].astype(int) == int(cps)).any())


def _affected_fractions(ctx: Any, cps: int, parcel_idx: int, bmp_rec: Mapping[str, Any]) -> List[Tuple[int, float]]:
    """Return which parcels are affected and what share of each parcel is treated."""
    if int(cps) in (656, 657):
        impacted = [str(ctx.parcel_selection_ids[parcel_idx])]
        extra = str(bmp_rec.get("impacted_pids", "") or "")
        impacted.extend([x for x in extra.split(",") if x])
        impacted = list(dict.fromkeys(impacted))
        target_area = float(bmp_rec.get("wetland_area_ha", 0.0) or 0.0) * (
            1.0 + float(bmp_rec.get("catchment_to_wetland_ratio", 0.0) or 0.0)
        )
        remaining = max(0.0, target_area)
        index_by_pid = ctx.pid_to_index
        affected: List[Tuple[int, float]] = []
        for pid in impacted:
            idx = index_by_pid.get(str(pid))
            if idx is None:
                continue
            area = max(float(ctx.parcel_area_ha[idx]), 1.0e-12)
            fraction = float(np.clip(remaining / area, 0.0, 1.0))
            affected.append((idx, fraction))
            remaining -= area
            if remaining <= 0.0:
                break
        return affected
    if int(cps) == 412:
        fraction = float(np.clip(float(bmp_rec.get("portion_treated", 0.0) or 0.0), 0.0, 1.0))
        return [(parcel_idx, fraction)]
    return [(parcel_idx, 1.0)]


def _get_effect_value(state: LoadState, idx: int, parameter: str) -> float:
    """Read one current value from the scenario state."""
    if parameter.startswith("groundwater_concentration_"):
        pollutant = parameter.removeprefix("groundwater_concentration_").upper()
        return float(state.groundwater_concentrations[idx].get(pollutant, 0.0))
    if parameter.startswith("concentration_"):
        pollutant = parameter.removeprefix("concentration_").upper()
        return float(state.concentrations[idx].get(pollutant, 0.0))
    return float(state.parameters[idx].get(parameter, 1.0 if parameter.startswith("load_multiplier_") else 0.0))


def _set_effect_value(state: LoadState, idx: int, parameter: str, value: float) -> None:
    """Save one updated value back to the scenario state, with safe limits."""
    if parameter.startswith("groundwater_concentration_"):
        pollutant = parameter.removeprefix("groundwater_concentration_").upper()
        state.groundwater_concentrations[idx][pollutant] = max(0.0, float(value))
        return
    if parameter.startswith("concentration_"):
        pollutant = parameter.removeprefix("concentration_").upper()
        state.concentrations[idx][pollutant] = max(0.0, float(value))
        return

    value_f = float(value)
    if parameter == "cn":
        value_f = float(np.clip(value_f, 1.0e-6, 100.0))
    elif parameter in {"rain_correction_fraction", "runoff_day_fraction", "sdr"}:
        value_f = float(np.clip(value_f, 0.0, 1.0))
    elif parameter == "ia_ratio":
        value_f = max(0.0, value_f)
    elif parameter in {"c", "p"}:
        value_f = max(0.0, value_f)
    elif parameter.startswith("load_multiplier_") or parameter.endswith("_multiplier"):
        value_f = max(0.0, value_f)
    state.parameters[idx][parameter] = value_f


def _sample_effect_value(ctx: Any, row: Mapping[str, Any]) -> float:
    """Pick one effect amount from a BMP effect table row."""
    stats = _stats_from_row(row, {"cps", "parameter", "operation", "units", "notes"})
    if not stats:
        raise ValueError(
            f"No value or statistics supplied for process effect cps={row.get('cps')} parameter={row.get('parameter')}"
        )
    return _sample_stats(ctx, stats, nonnegative=False)


def apply_process_parameter_bmp(
    ctx: Any,
    state: LoadState,
    yields: np.ndarray,
    cps: int,
    parcel_idx: int,
    bmp_rec: MutableMapping[str, Any],
    *,
    effect_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Apply a BMP's rule-based changes and recalculate affected parcel loads.

    Returns how much load was treated, how much was removed, and a list of the
    exact value changes that were made.
    """

    table = ctx.bmp_parameter_effects
    rows = table[table["cps"].astype(int) == int(cps)]
    if rows.empty:
        zeros = np.zeros(len(ctx.pollutants), dtype=float)
        return zeros.copy(), zeros.copy(), []

    effect_scale = float(np.clip(effect_scale, 0.0, 1.0))
    treated = np.zeros(len(ctx.pollutants), dtype=float)
    removed = np.zeros(len(ctx.pollutants), dtype=float)
    changes: List[Dict[str, Any]] = []

    for idx, treatment_fraction in _affected_fractions(ctx, cps, parcel_idx, bmp_rec):
        if treatment_fraction <= 0.0:
            continue
        area_ha = float(ctx.parcel_area_ha[idx])
        before = yields[idx, :].copy()
        treated += before * area_ha * treatment_fraction

        for _, row in rows.iterrows():
            parameter = canonical_parameter_name(row["parameter"])
            operation = str(row.get("operation", "multiply")).strip().lower()
            sampled = _sample_effect_value(ctx, row)
            old = _get_effect_value(state, idx, parameter)
            if operation in {"multiply", "scale"}:
                target = old * sampled
            elif operation in {"add", "delta"}:
                target = old + sampled
            elif operation in {"set", "replace"}:
                target = sampled
            elif operation in {"reduce", "reduction_fraction"}:
                target = old * (1.0 - sampled)
            else:
                raise ValueError(f"Unsupported process effect operation: {operation}")

            blend = treatment_fraction * effect_scale
            new = old + blend * (target - old)
            _set_effect_value(state, idx, parameter, new)
            changes.append(
                {
                    "pid": state.parcel_ids[idx],
                    "parameter": parameter,
                    "operation": operation,
                    "sampled_effect": float(sampled),
                    "old": float(old),
                    "new": float(_get_effect_value(state, idx, parameter)),
                    "treated_fraction": float(treatment_fraction),
                    "effect_scale": float(effect_scale),
                }
            )

        state.pathway_yields[idx, :, :] = calculate_pathway_yields(
            state.parameters[idx],
            state.concentrations[idx],
            state.groundwater_concentrations[idx],
            ctx.pollutants,
            pathway_mode=str(getattr(ctx, "pathway_mode", "fixed_fractions")),
            surface_fraction=float(getattr(ctx, "pollutant_yield_frac_surface", 0.0)),
            shallow_fraction=float(getattr(ctx, "pollutant_yield_frac_shallow", 0.0)),
            groundwater_loads=bool(getattr(ctx, "groundwater_loads", False)),
            treat_groundwater_with_bmps=bool(getattr(ctx, "load_generation", {}).get("treat_groundwater_with_bmps", False)),
        )
        after = np.sum(state.pathway_yields[idx, :, :], axis=1)
        yields[idx, :] = after
        removed += (before - after) * area_ha

    bmp_rec["process_parameter_changes"] = json.dumps(changes, separators=(",", ":"))
    return treated, removed, changes
