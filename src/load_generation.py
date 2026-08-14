"""Build parcel pollutant loads and update them as BMPs are applied.

This module converts scenario input tables into parcel-level annual pollutant
loads, splits those loads across flow pathways, and recalculates the affected
values when BMP rules or treatment fractions change. It contains the core load
generation logic used by the scenario simulator.
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
    "shallow_subsurface_fraction": "fraction_subsurface_shallow",
    "fraction_shallow_subsurface": "fraction_subsurface_shallow",
    "subsurface_shallow_fraction": "fraction_subsurface_shallow",
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
    """Container for per-parcel scenario state.

    The state stores current sampled values, baseline values, parcel ordering,
    and pathway-level yields so that BMP application can update the scenario in
    place while still preserving the original starting point.

    Attributes
    ----------
    parcel_ids : list[str]
        Parcel identifiers in the same order used by the numeric arrays.
    parameters : list[dict[str, float]]
        Current sampled parameter values for each parcel.
    concentrations : list[dict[str, float]]
        Current runoff concentrations for each parcel.
    groundwater_concentrations : list[dict[str, float]]
        Current groundwater concentrations for each parcel.
    has_rusle : list[bool]
        Flags indicating whether each parcel has complete RUSLE inputs.
    pollutants : list[str]
        Pollutants tracked by the simulation.
    pathway_yields : numpy.ndarray
        Current pathway-specific parcel yields with shape
        ``(n_parcels, n_pollutants, n_pathways)``.
    baseline_parameters : list[dict[str, float]]
        Snapshot of the original parameter values.
    baseline_concentrations : list[dict[str, float]]
        Snapshot of the original runoff concentrations.
    baseline_groundwater_concentrations : list[dict[str, float]]
        Snapshot of the original groundwater concentrations.
    baseline_pathway_yields : numpy.ndarray or None
        Snapshot of the original pathway-specific yields.
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
        """Map parcel IDs to their positional index.

        Returns
        -------
        dict[str, int]
            Dictionary mapping each parcel ID to its index in the state
            arrays.
        """
        return {pid: i for i, pid in enumerate(self.parcel_ids)}


def canonical_parameter_name(value: Any) -> str:
    """Normalize an input parameter label to the internal canonical name.

    The loader accepts multiple spellings and aliases for user-authored input
    tables. This helper collapses those variants to a single lowercase,
    underscore-delimited key so the rest of the module can work with one
    consistent naming scheme.

    Parameters
    ----------
    value : Any
        Raw parameter name from an input table.

    Returns
    -------
    str
        Canonical parameter name.
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
    """Calculate event rainfall and runoff depths using a CN-style equation.

    The function computes a representative storm event depth, the associated
    runoff depth, and annualized storm runoff depth using annual precipitation,
    rain-day frequency, and curve number assumptions.

    Parameters
    ----------
    annual_precip_in : float
        Annual precipitation depth in inches.
    rain_days : float
        Number of rain days per year.
    rain_correction_fraction : float
        Fraction of annual precipitation attributed to runoff-producing events.
    runoff_day_fraction : float
        Fraction of rain days that generate runoff.
    cn : float
        Curve number used to estimate retention.
    ia_ratio : float, optional
        Initial abstraction ratio applied to retention. Default is ``0.0``.

    Returns
    -------
    tuple[float, float, float]
        Event rainfall depth, event runoff depth, and annual runoff depth in
        inches.
    """

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
    """Estimate sediment delivery ratio from watershed size.

    Parameters
    ----------
    watershed_area_mi2 : float
        Watershed area in square miles.

    Returns
    -------
    float
        Estimated sediment delivery ratio clipped to ``[0, 1]``.
    """

    area_mi2 = max(float(watershed_area_mi2), 1.0e-12)
    area_acres = area_mi2 * ACRES_PER_SQUARE_MILE
    if area_acres < 200.0:
        dr = 0.42 * area_mi2 ** (-0.125)
    else:
        dr = 0.417662 * area_mi2 ** (-0.134958) - 0.127097
    return float(np.clip(dr, 0.0, 1.0))


def rusle_sediment_yield_kg_ha(parameters: Mapping[str, float]) -> float:
    """Estimate annual sediment yield per hectare from RUSLE inputs.

    Parameters
    ----------
    parameters : Mapping[str, float]
        Mapping containing the required RUSLE factors and optional delivery
        modifiers.

    Returns
    -------
    float
        Annual sediment yield in kilograms per hectare.

    Raises
    ------
    ValueError
        If neither ``sdr`` nor ``watershed_area_mi2`` is provided.
    """

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
    """Estimate annual irrigation runoff depth.

    The calculation uses the same curve-number runoff equation as the storm
    runoff helper, then scales the per-event runoff by irrigation frequency and
    the irrigated fraction.

    Parameters
    ----------
    parameters : Mapping[str, float]
        Mapping containing irrigation and curve-number parameters.

    Returns
    -------
    float
        Annual irrigation runoff depth in inches.
    """
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
    """Estimate annual surface runoff depth including irrigation runoff.

    Parameters
    ----------
    parameters : Mapping[str, float]
        Mapping containing runoff and irrigation parameters.

    Returns
    -------
    tuple[float, float, float, float]
        Event rainfall depth, event runoff depth, annual storm runoff depth,
        and annual total runoff depth.
    """
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
    """Estimate annual infiltration depth from precipitation.

    Parameters
    ----------
    parameters : Mapping[str, float]
        Mapping that may contain ``infiltration_fraction``,
        ``annual_precip_in``, and ``groundwater_multiplier``.

    Returns
    -------
    float
        Annual infiltration depth in inches.
    """
    infiltration_fraction = float(np.clip(parameters.get("infiltration_fraction", 0.0), 0.0, 1.0))
    annual_precip = max(0.0, float(parameters.get("annual_precip_in", 0.0)))
    infiltration = annual_precip * infiltration_fraction
    infiltration *= max(0.0, float(parameters.get("groundwater_multiplier", 1.0)))
    return float(max(0.0, infiltration))


def _stats_from_row(row: Mapping[str, Any], exclude: Iterable[str]) -> Dict[str, float]:
    """Extract sampling statistics from a table row.

    Parameters
    ----------
    row : Mapping[str, Any]
        Source row containing values and sampling statistics.
    exclude : Iterable[str]
        Column names that should not be considered as numeric statistics.

    Returns
    -------
    dict[str, float]
        Numeric statistics suitable for sampling.
    """
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
    """Sample a single numeric value from statistics.

    Parameters
    ----------
    ctx : Any
        Object providing ``_sample_from_stats``.
    stats : Mapping[str, float]
        Sampling statistics or a fixed ``value`` entry.
    nonnegative : bool, optional
        If ``True``, clamp the returned value at zero. Default is ``False``.

    Returns
    -------
    float
        Sampled numeric value.
    """
    if "value" in stats:
        value = float(stats["value"])
    else:
        value = float(ctx._sample_from_stats(dict(stats), kind="yield" if nonnegative else None))
    return max(0.0, value) if nonnegative else value


def _rows_for_pid(table: Optional[pd.DataFrame], pid: str) -> List[pd.Series]:
    """Return the rows from a table that apply to one parcel.

    Rows with ``pid="*"`` act as defaults. Rows for the exact parcel ID
    override those defaults for matching parameters.

    Parameters
    ----------
    table : pandas.DataFrame or None
        Input table containing parcel-specific rows.
    pid : str
        Parcel identifier.

    Returns
    -------
    list[pandas.Series]
        Matching rows in evaluation order.
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
    """Build sampled parameter dictionaries for each parcel.

    Parameters
    ----------
    ctx : Any
        Context object providing sampling helpers and input tables.
    table : pandas.DataFrame or None
        Source table containing parameter rows.
    parcel_ids : sequence of str
        Parcel identifiers to sample.
    cache_prefix : str
        Prefix used to separate cached samples for different input tables.

    Returns
    -------
    list[dict[str, float]]
        Sampled parameter values for each parcel.
    """
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
    """Build sampled concentration dictionaries for each parcel.

    Parameters
    ----------
    ctx : Any
        Context object providing sampling helpers.
    table : pandas.DataFrame or None
        Source table containing pollutant concentration rows.
    parcel_ids : sequence of str
        Parcel identifiers to sample.

    Returns
    -------
    list[dict[str, float]]
        Sampled concentrations for each parcel.
    """
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
    """Calculate intermediate load-generation diagnostics.

    Parameters
    ----------
    parameters : Mapping[str, float]
        Parameter mapping used for runoff, infiltration, and sediment
        calculations.

    Returns
    -------
    dict[str, float]
        Diagnostic values useful for reporting and debugging.
    """

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
    """Calculate pathway-specific parcel yields.

    The calculation supports two modes. In ``derive_from_plet`` mode, yields
    are derived directly from runoff, infiltration, and groundwater
    concentrations. In ``fixed_fractions`` mode, a total load is partitioned
    across the three pathways using the supplied fractions.

    Parameters
    ----------
    parameters : Mapping[str, float]
        Parcel parameters used to compute runoff and other drivers.
    concentrations : Mapping[str, float]
        Runoff concentrations keyed by pollutant name.
    groundwater_concentrations : Mapping[str, float] or None
        Groundwater concentrations keyed by pollutant name.
    pollutants : sequence of str
        Pollutants to calculate.
    pathway_mode : str, optional
        Either ``"fixed_fractions"`` or ``"derive_from_plet"``. Default is
        ``"fixed_fractions"``.
    surface_fraction : float, optional
        Fraction of load assigned to surface flow in fixed-fraction mode.
    shallow_fraction : float, optional
        Fraction of load assigned to shallow subsurface flow in fixed-fraction
        mode.
    groundwater_loads : bool, optional
        Whether groundwater concentrations should contribute to pollutant
        loads. Default is ``False``.
    treat_groundwater_with_bmps : bool, optional
        Whether groundwater loads should use the configured pathway split.
        When ``False``, groundwater is assigned to the deep pathway so BMPs
        do not reduce it. Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_pollutants, 3)`` containing surface, shallow
        subsurface, and deep subsurface loads.
    """

    missing = [name for name in _REQUIRED_PLET if name not in parameters]
    if missing:
        raise ValueError(f"PLET inputs are missing required parameters: {missing}")

    _, _, _, annual_runoff_in = plet_annual_surface_runoff_in(parameters)
    runoff_l_ha = annual_runoff_in * INCH_OVER_HA_TO_LITERS
    infiltration_l_ha = plet_annual_infiltration_in(parameters) * INCH_OVER_HA_TO_LITERS

    has_rusle = all(name in parameters for name in _REQUIRED_RUSLE)
    sediment_kg_ha = rusle_sediment_yield_kg_ha(parameters) if has_rusle else 0.0
    enrichment_ratio = max(0.0, float(parameters.get("enrichment_ratio", 2.0)))  #TODO: default is 2 which should increase TN and TP loads. Should this be revised or even removed?

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
        # PLET resolves surface runoff and total subsurface/groundwater load,
        # but not the shallow-versus-deep subsurface split used by the BMP
        # simulator.  Keep those concepts separate by partitioning total
        # subsurface load with an explicit, independently sampled parameter.
        if pathway_mode == "derive_from_plet" and groundwater_loads and pol != "TSS" and treat_groundwater_with_bmps:
            if "fraction_subsurface_shallow" not in parameters:
                raise ValueError(
                    "derive_from_plet with groundwater loads requires the PLET "
                    "parameter 'fraction_subsurface_shallow' (0 to 1)"
                )
            fraction_subsurface_shallow = float(
                np.clip(parameters["fraction_subsurface_shallow"], 0.0, 1.0) #TODO: Can we set a range of values?
            )
        else:
            fraction_subsurface_shallow = 0.0

        if pol == "TSS":
            load = sediment_kg_ha if has_rusle else runoff_load
            surface_load = load
            shallow_load = 0.0
            deep_load = 0.0
        elif pol == "TN":
            sediment_fraction = max(0.0, float(parameters.get("sediment_n_pct", 0.0))) / 100.0
            load = runoff_load + sediment_kg_ha * sediment_fraction * enrichment_ratio
            surface_load = load
            shallow_load = groundwater_load * fraction_subsurface_shallow
            deep_load = groundwater_load * (1.0 - fraction_subsurface_shallow)
        elif pol == "TP":
            sediment_fraction = max(0.0, float(parameters.get("sediment_p_pct", 0.0))) / 100.0
            load = runoff_load + sediment_kg_ha * sediment_fraction * enrichment_ratio
            surface_load = load
            shallow_load = groundwater_load * fraction_subsurface_shallow
            deep_load = groundwater_load * (1.0 - fraction_subsurface_shallow)
        else:
            load = runoff_load
            surface_load = runoff_load
            shallow_load = groundwater_load * fraction_subsurface_shallow
            deep_load = groundwater_load * (1.0 - fraction_subsurface_shallow)

        total_load = (load + groundwater_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
        if pathway_mode == "derive_from_plet":
            out[idx, 0] = max(0.0, surface_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
            if treat_groundwater_with_bmps:
                out[idx, 1] = max(0.0, shallow_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
                out[idx, 2] = max(0.0, deep_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
            else:
                out[idx, 2] = max(0.0, groundwater_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
        else:
            fixed_surface = float(np.clip(surface_fraction, 0.0, 1.0))
            fixed_shallow = float(np.clip(shallow_fraction, 0.0, 1.0))
            fixed_deep = max(0.0, 1.0 - fixed_surface - fixed_shallow)
            partitioned_load = total_load if treat_groundwater_with_bmps else max(0.0, load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
            out[idx, 0] = partitioned_load * fixed_surface
            out[idx, 1] = partitioned_load * fixed_shallow
            out[idx, 2] = partitioned_load * fixed_deep
            if not treat_groundwater_with_bmps:
                out[idx, 2] += max(0.0, groundwater_load) * max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
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
    """Calculate total parcel pollutant loads per hectare.

    Parameters
    ----------
    parameters : Mapping[str, float]
        Parcel parameters used to derive the pathway loads.
    concentrations : Mapping[str, float]
        Runoff concentrations keyed by pollutant name.
    pollutants : sequence of str
        Pollutants to calculate.
    groundwater_concentrations : Mapping[str, float] or None, optional
        Groundwater concentrations keyed by pollutant name.
    pathway_mode : str, optional
        See :func:`calculate_pathway_yields`.
    surface_fraction : float, optional
        See :func:`calculate_pathway_yields`.
    shallow_fraction : float, optional
        See :func:`calculate_pathway_yields`.
    groundwater_loads : bool, optional
        See :func:`calculate_pathway_yields`.
    treat_groundwater_with_bmps : bool, optional
        See :func:`calculate_pathway_yields`.

    Returns
    -------
    numpy.ndarray
        Total annual loads per pollutant.
    """

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
    """Create the initial parcel load state for a scenario.

    This function samples parcel-level parameters and concentrations, verifies
    that required inputs are present, computes baseline pathway yields, and
    returns both the total load array and the mutable scenario state.

    Parameters
    ----------
    ctx : Any
        Scenario context containing the input tables, pollutant list, and
        configuration flags.

    Returns
    -------
    tuple[numpy.ndarray, LoadState]
        Baseline parcel loads and the corresponding mutable load state.

    Raises
    ------
    ValueError
        If required inputs are missing or incomplete for any parcel.
    """

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
    """Check whether a BMP has process-effect rules configured.

    Parameters
    ----------
    ctx : Any
        Context object containing the process-effect table.
    cps : int
        BMP CPS code to query.

    Returns
    -------
    bool
        ``True`` when at least one rule exists for the BMP, otherwise
        ``False``.
    """
    table = getattr(ctx, "bmp_parameter_effects", None)
    if table is None or table.empty:
        return False
    return bool((table["cps"].astype(int) == int(cps)).any())


def _affected_fractions(ctx: Any, cps: int, parcel_idx: int, bmp_rec: Mapping[str, Any]) -> List[Tuple[int, float]]:
    """Determine which parcels are treated and by what fraction.

    Parameters
    ----------
    ctx : Any
        Scenario context containing parcel metadata.
    cps : int
        BMP CPS code.
    parcel_idx : int
        Index of the parcel where the BMP is placed.
    bmp_rec : Mapping[str, Any]
        BMP placement record containing BMP-specific metadata.

    Returns
    -------
    list[tuple[int, float]]
        Pairs of parcel index and treated fraction.
    """
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
    """Read a current value from the scenario state.

    Parameters
    ----------
    state : LoadState
        Mutable scenario state.
    idx : int
        Parcel index.
    parameter : str
        Parameter name, concentration name, or groundwater concentration name.

    Returns
    -------
    float
        Current value for the requested field.
    """
    if parameter.startswith("groundwater_concentration_"):
        pollutant = parameter.removeprefix("groundwater_concentration_").upper()
        return float(state.groundwater_concentrations[idx].get(pollutant, 0.0))
    if parameter.startswith("concentration_"):
        pollutant = parameter.removeprefix("concentration_").upper()
        return float(state.concentrations[idx].get(pollutant, 0.0))
    return float(state.parameters[idx].get(parameter, 1.0 if parameter.startswith("load_multiplier_") else 0.0))


def _set_effect_value(state: LoadState, idx: int, parameter: str, value: float) -> None:
    """Write an updated value back to the scenario state.

    Values are clipped to safe bounds for concentration and parameter fields
    so that downstream recalculations remain numerically stable.

    Parameters
    ----------
    state : LoadState
        Mutable scenario state.
    idx : int
        Parcel index.
    parameter : str
        Parameter name, concentration name, or groundwater concentration name.
    value : float
        New value to store.

    Returns
    -------
    None
    """
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
    elif parameter in {"rain_correction_fraction", "runoff_day_fraction", "sdr", "fraction_subsurface_shallow"}:
        value_f = float(np.clip(value_f, 0.0, 1.0))
    elif parameter == "ia_ratio":
        value_f = max(0.0, value_f)
    elif parameter in {"c", "p"}:
        value_f = max(0.0, value_f)
    elif parameter.startswith("load_multiplier_") or parameter.endswith("_multiplier"):
        value_f = max(0.0, value_f)
    state.parameters[idx][parameter] = value_f


def _sample_effect_value(ctx: Any, row: Mapping[str, Any]) -> float:
    """Sample a BMP process-effect value from one table row.

    Parameters
    ----------
    ctx : Any
        Context object providing ``_sample_from_stats``.
    row : Mapping[str, Any]
        Process-effect table row.

    Returns
    -------
    float
        Sampled effect value.

    Raises
    ------
    ValueError
        If the row does not contain any usable numeric statistics.
    """
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
    """Apply rule-based BMP effects and recalculate parcel loads.

    The function applies all configured process-effect rows for the BMP,
    updates the mutable scenario state, recalculates the affected pathway
    yields, and returns aggregate treated and removed loads plus a detailed
    change log.

    Parameters
    ----------
    ctx : Any
        Scenario context containing pollutant lists and effect tables.
    state : LoadState
        Mutable scenario state updated in place.
    yields : numpy.ndarray
        Current total parcel yields updated in place.
    cps : int
        BMP CPS code.
    parcel_idx : int
        Index of the parcel receiving the BMP.
    bmp_rec : MutableMapping[str, Any]
        BMP record updated with JSON-encoded process change details.
    effect_scale : float, optional
        Fraction of the sampled effect to apply. Default is ``1.0``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, list[dict[str, Any]]]
        Treated loads, removed loads, and a list of change records.

    Raises
    ------
    ValueError
        If a process-effect operation is not supported.
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
