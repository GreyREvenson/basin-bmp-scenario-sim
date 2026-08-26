"""Build parcel pollutant loads for statistical and PLET/RUSLE scenarios.

This module converts scenario input tables into parcel-level annual pollutant
loads, splits those loads across flow pathways, and calculates load diagnostics
used by the scenario simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .input_distributions import (
    sample_group_key,
    sample_stats_bounded,
    stats_from_row,
)


INCH_OVER_HA_TO_LITERS = 254_000.0
TON_PER_ACRE_TO_KG_PER_HA = 907.18474 / 0.40468564224
ACRES_PER_SQUARE_MILE = 640.0
# Legacy public helper pathway order retained for backward compatibility.
# Production plet_rusle scenarios do NOT use this three-path tuple.
LEGACY_PATHWAY_NAMES = ("surface", "shallow subsurface", "deep subsurface")
PATHWAY_NAMES = LEGACY_PATHWAY_NAMES
PLET_PATHWAY_NAMES = ("surface", "subsurface")
PLET_CLASSIFICATION_PARAMETERS = ("land_cover", "hsg")
PLET_LAND_COVERS = ("urban", "cropland", "pastureland", "forest", "user_defined")
PLET_HSG_VALUES = ("A", "B", "C", "D")
# Default example path retained only for the public deterministic helper.
# Production plet_rusle runs require load_generation.hydrology_lookup explicitly.
PLET_HYDROLOGY_LOOKUP_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples" / "east_fork" / "inputs" / "plet" / "plet_hydrology_lookup.csv"
)

_PLET_DERIVED_PARAMETERS = ("cn", "infiltration_fraction")
_PLET_LAND_COVER_ALIASES: Dict[str, str] = {
    "urban": "urban",
    "developed": "urban",
    "cropland": "cropland",
    "crop": "cropland",
    "row_crop": "cropland",
    "row_crops": "cropland",
    "pasture": "pastureland",
    "pastureland": "pastureland",
    "forest": "forest",
    "forested": "forest",
    "woodland": "forest",
    "woods": "forest",
    "user_defined": "user_defined",
    "userdefined": "user_defined",
}

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
    "land_use": "land_cover",
    "landuse": "land_cover",
    "land_cover_class": "land_cover",
    "land_cover_classification": "land_cover",
    "hydrologic_soil_group": "hsg",
    "soil_hydrologic_group": "hsg",
    "soil_group": "hsg",
    "hsg_classification": "hsg",
    "shg": "hsg",
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

_REQUIRED_PLET_INPUTS = (
    "annual_precip_in",
    "rain_days",
    "rain_correction_fraction",
    "runoff_day_fraction",
    "land_cover",
    "hsg",
)
_REQUIRED_RESOLVED_PLET = (
    "annual_precip_in",
    "rain_days",
    "rain_correction_fraction",
    "runoff_day_fraction",
    "cn",
    "infiltration_fraction",
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
    parameters : list[dict[str, Any]]
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
        ``(n_parcels, n_pollutants, 2)``. In PLET/RUSLE mode the two pathways
        are ``surface`` and ``subsurface``.
    untreated_groundwater_yields : numpy.ndarray
        Deprecated compatibility array. PLET/RUSLE subsurface loads are now
        included in the ``subsurface`` pathway, so this array is always zero.
    baseline_parameters : list[dict[str, Any]]
        Snapshot of the original parameter values.
    baseline_concentrations : list[dict[str, float]]
        Snapshot of the original runoff concentrations.
    baseline_groundwater_concentrations : list[dict[str, float]]
        Snapshot of the original groundwater concentrations.
    baseline_pathway_yields : numpy.ndarray or None
        Snapshot of the original pathway-specific yields.
    baseline_untreated_groundwater_yields : numpy.ndarray or None
        Snapshot of the original groundwater yields excluded from BMP
        treatment.
    """

    parcel_ids: List[str]
    parameters: List[Dict[str, Any]]
    concentrations: List[Dict[str, float]]
    groundwater_concentrations: List[Dict[str, float]]
    has_rusle: List[bool]
    pollutants: List[str]
    pathway_yields: np.ndarray
    untreated_groundwater_yields: np.ndarray
    baseline_parameters: List[Dict[str, Any]] = field(default_factory=list)
    baseline_concentrations: List[Dict[str, float]] = field(default_factory=list)
    baseline_groundwater_concentrations: List[Dict[str, float]] = field(default_factory=list)
    baseline_pathway_yields: Optional[np.ndarray] = None
    baseline_untreated_groundwater_yields: Optional[np.ndarray] = None

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


def normalize_plet_land_cover(value: Any) -> str:
    """Normalize a PLET land-cover classification.

    Parameters
    ----------
    value : Any
        Raw land-cover label from a PLET input row.

    Returns
    -------
    str
        One of the canonical values in ``PLET_LAND_COVERS``.

    Raises
    ------
    ValueError
        If the classification is empty or is not represented in PLET's
        reference curve-number table.
    """

    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _PLET_LAND_COVER_ALIASES.get(label)
    if canonical is None:
        allowed = ", ".join(PLET_LAND_COVERS)
        raise ValueError(
            f"Unsupported PLET land_cover classification {value!r}; "
            f"expected one of: {allowed}"
        )
    return canonical


def normalize_plet_hsg(value: Any) -> str:
    """Normalize a PLET hydrologic soil group classification.

    Parameters
    ----------
    value : Any
        Raw hydrologic soil group label.

    Returns
    -------
    str
        One of ``A``, ``B``, ``C``, or ``D``.

    Raises
    ------
    ValueError
        If the supplied value is not a single PLET HSG class.
    """

    label = str(value).strip().upper()
    if label not in PLET_HSG_VALUES:
        raise ValueError(
            f"Unsupported PLET hsg classification {value!r}; expected one of: "
            f"{', '.join(PLET_HSG_VALUES)}"
        )
    return label


@lru_cache(maxsize=None)
def _load_plet_hydrology_records(
    lookup_path: str,
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Load fixed CN/infiltration values for the deterministic public helper.

    New production inputs use a long-form table with one ``parameter`` row for
    ``cn`` and one for ``infiltration_fraction`` per land-cover/HSG pairing.
    The legacy wide format is still accepted here for backward-compatible
    helper/tests, but stochastic rows cannot be resolved without a model RNG.
    """
    path = Path(lookup_path)
    if not path.exists():
        raise FileNotFoundError(f"PLET hydrology lookup table not found: {path}")

    table = pd.read_csv(path)
    table.columns = [str(c).strip().lower() for c in table.columns]
    records: Dict[Tuple[str, str], Dict[str, float]] = {}

    if {"land_cover", "hsg", "parameter"} <= set(table.columns):
        for row_index, row in table.iterrows():
            land_cover = normalize_plet_land_cover(row["land_cover"])
            hsg = normalize_plet_hsg(row["hsg"])
            parameter = canonical_parameter_name(row["parameter"])
            if parameter not in _PLET_DERIVED_PARAMETERS:
                continue
            stats = stats_from_row(row)
            if set(stats) != {"value"}:
                raise ValueError(
                    "plet_hydrology_from_classifications is deterministic and "
                    f"cannot resolve stochastic {parameter} row {row_index}; "
                    "production plet_rusle runs sample this table through the scenario context"
                )
            records.setdefault((land_cover, hsg), {})[parameter] = float(stats["value"])
    elif {"land_cover", "hsg", "curve_number", "infiltration_fraction"} <= set(table.columns):
        # Legacy compatibility while users migrate existing lookup files.
        for row_index, row in table.iterrows():
            land_cover = normalize_plet_land_cover(row["land_cover"])
            hsg = normalize_plet_hsg(row["hsg"])
            records[(land_cover, hsg)] = {
                "cn": float(row["curve_number"]),
                "infiltration_fraction": float(row["infiltration_fraction"]),
            }
    else:
        raise ValueError(
            "PLET hydrology lookup must use long-form columns "
            "land_cover,hsg,parameter plus value/distribution statistics"
        )

    expected_keys = {
        (land_cover, hsg)
        for land_cover in PLET_LAND_COVERS
        for hsg in PLET_HSG_VALUES
    }
    if set(records) != expected_keys:
        missing = sorted(expected_keys - set(records))
        extra = sorted(set(records) - expected_keys)
        raise ValueError(
            "PLET hydrology lookup table must contain every supported "
            f"land-cover/HSG combination; missing={missing}, unexpected={extra}"
        )

    resolved: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for key, values in records.items():
        if set(values) != set(_PLET_DERIVED_PARAMETERS):
            raise ValueError(
                f"PLET hydrology lookup {key} must define both cn and infiltration_fraction"
            )
        curve_number = float(values["cn"])
        infiltration_fraction = float(values["infiltration_fraction"])
        if not np.isfinite(curve_number) or not 0.0 < curve_number <= 100.0:
            raise ValueError(f"PLET hydrology lookup cn must be in (0,100]; {key} has {curve_number}")
        if not np.isfinite(infiltration_fraction) or not 0.0 <= infiltration_fraction <= 1.0:
            raise ValueError(
                "PLET hydrology lookup infiltration_fraction must be in [0,1]; "
                f"{key} has {infiltration_fraction}"
            )
        resolved[key] = (curve_number, infiltration_fraction)
    return resolved


def plet_hydrology_from_classifications(
    land_cover: Any,
    hsg: Any,
    *,
    lookup_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Resolve fixed hydrology values from a lookup table.

    This helper remains for compatibility and deterministic validation. The
    actual PLET/RUSLE scenario path samples the required hydrology input table,
    allowing each CN and infiltration fraction to be a fixed value or a
    distribution.
    """
    canonical_land_cover = normalize_plet_land_cover(land_cover)
    canonical_hsg = normalize_plet_hsg(hsg)
    path = Path(lookup_path or PLET_HYDROLOGY_LOOKUP_PATH).resolve()
    records = _load_plet_hydrology_records(str(path))
    curve_number, infiltration_fraction = records[(canonical_land_cover, canonical_hsg)]
    return {
        "land_cover": canonical_land_cover,
        "hsg": canonical_hsg,
        "cn": curve_number,
        "infiltration_fraction": infiltration_fraction,
    }


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


def rusle_sediment_yield_kg_ha(parameters: Mapping[str, Any]) -> float:
    """Estimate annual sediment yield per hectare from RUSLE inputs.

    Parameters
    ----------
    parameters : Mapping[str, Any]
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

    sdr = 1.0
    if "sdr" in parameters:
        sdr = float(parameters["sdr"])
        if sdr < 0.0 or sdr > 1.0:
            raise ValueError("RUSLE input parameter value for sdr must be between 0.0 and 1.0")

    sediment_multiplier = max(0.0, float(parameters.get("sediment_multiplier", 1.0)))
    delivery_multiplier = max(0.0, float(parameters.get("sediment_delivery_multiplier", 1.0)))
    return gross_ton_ac * sdr * TON_PER_ACRE_TO_KG_PER_HA * sediment_multiplier * delivery_multiplier


def plet_annual_irrigation_runoff_in(parameters: Mapping[str, Any]) -> float:
    """Estimate annual irrigation runoff depth.

    The calculation uses the same curve-number runoff equation as the storm
    runoff helper, then scales the per-event runoff by irrigation frequency and
    the irrigated fraction.

    Parameters
    ----------
    parameters : Mapping[str, Any]
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


def plet_annual_surface_runoff_in(
    parameters: Mapping[str, Any],
) -> Tuple[float, float, float, float]:
    """Estimate annual surface runoff depth including irrigation runoff.

    Parameters
    ----------
    parameters : Mapping[str, Any]
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


def plet_annual_infiltration_in(parameters: Mapping[str, Any]) -> float:
    """Estimate annual infiltration depth from precipitation.

    Parameters
    ----------
    parameters : Mapping[str, Any]
        Mapping containing the lookup-derived ``infiltration_fraction`` and
        precipitation inputs, plus an optional ``groundwater_multiplier``.

    Returns
    -------
    float
        Annual infiltration depth in inches.
    """
    if "infiltration_fraction" not in parameters:
        raise ValueError(
            "Resolved PLET parameters are missing infiltration_fraction"
        )
    infiltration_fraction = float(
        np.clip(parameters["infiltration_fraction"], 0.0, 1.0)
    )
    annual_precip = max(0.0, float(parameters.get("annual_precip_in", 0.0)))
    rain_correction_fraction = float(
        np.clip(parameters.get("rain_correction_fraction", 1.0), 0.0, 1.0)
    )
    infiltration = annual_precip * rain_correction_fraction * infiltration_fraction
    infiltration *= max(0.0, float(parameters.get("groundwater_multiplier", 1.0)))
    return float(max(0.0, infiltration))


def _stats_from_row(row: Mapping[str, Any], exclude: Iterable[str]) -> Dict[str, float]:
    """Backward-compatible wrapper around the shared statistics extractor."""
    return stats_from_row(row, exclude)


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
    combined = combined.assign(
        _canonical_parameter=combined["parameter"].map(canonical_parameter_name)
    )
    combined = combined.drop_duplicates(
        subset=["_canonical_parameter"], keep="last"
    )
    return [row for _, row in combined.iterrows()]


def validate_plet_input_table(
    table: pd.DataFrame,
    parcel_ids: Sequence[str],
) -> pd.DataFrame:
    """Validate required PLET classifications for every modeled parcel.

    ``land_cover`` and ``hsg`` must be supplied as fixed ``value`` rows. Curve
    number and infiltration fraction are intentionally kept out of the parcel
    parameter table because they are supplied in the required
    ``load_generation.hydrology_lookup`` input, where each land-cover/HSG
    pairing can define a fixed value or a distribution.

    Parameters
    ----------
    table : pandas.DataFrame
        Long-form PLET parameter table.
    parcel_ids : sequence of str
        Parcel identifiers that will be modeled.

    Returns
    -------
    pandas.DataFrame
        A normalized copy of the table.

    Raises
    ------
    ValueError
        If classifications are missing or invalid, or if a derived hydrology
        parameter is supplied directly.
    """

    normalized = table.copy()
    normalized["parameter"] = normalized["parameter"].map(
        canonical_parameter_name
    )

    supplied_derived = sorted(
        set(normalized.loc[
            normalized["parameter"].isin(_PLET_DERIVED_PARAMETERS),
            "parameter",
        ])
    )
    if supplied_derived:
        raise ValueError(
            "PLET parcel inputs may not specify "
            f"{supplied_derived}; define cn and infiltration_fraction in the "
            "required load_generation.hydrology_lookup table instead"
        )

    classification_mask = normalized["parameter"].isin(
        PLET_CLASSIFICATION_PARAMETERS
    )
    if classification_mask.any() and "value" not in normalized.columns:
        raise ValueError(
            "PLET land_cover and hsg classifications require a fixed value column"
        )

    for row_index in normalized.index[classification_mask]:
        parameter = str(normalized.at[row_index, "parameter"])
        value = normalized.at[row_index, "value"]
        if pd.isna(value) or str(value).strip() == "":
            raise ValueError(
                f"PLET classification row {row_index} for {parameter} "
                "requires a fixed value"
            )
        if parameter == "land_cover":
            normalized.at[row_index, "value"] = normalize_plet_land_cover(value)
        else:
            normalized.at[row_index, "value"] = normalize_plet_hsg(value)

    for pid in map(str, parcel_ids):
        effective = {
            canonical_parameter_name(row["parameter"]): row
            for row in _rows_for_pid(normalized, pid)
        }
        missing = [
            parameter
            for parameter in PLET_CLASSIFICATION_PARAMETERS
            if parameter not in effective
        ]
        if missing:
            raise ValueError(
                f"PLET inputs for pid={pid} are missing required "
                f"classifications: {missing}"
            )
    return normalized


def _sample_parameter_table(
    ctx: Any,
    table: Optional[pd.DataFrame],
    parcel_ids: Sequence[str],
    *,
    cache_prefix: str,
) -> List[Dict[str, Any]]:
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
    list[dict[str, Any]]
        Sampled parameter values for each parcel.
    """
    sampled: List[Dict[str, Any]] = []
    cache: Dict[Tuple[str, str, str], Any] = {}
    for pid in parcel_ids:
        values: Dict[str, Any] = {}
        for row in _rows_for_pid(table, str(pid)):
            parameter = canonical_parameter_name(row["parameter"])
            variable_key, group_key = sample_group_key(
                row, pid=str(pid), variable=parameter
            )
            cache_key = (cache_prefix, variable_key, group_key)
            if cache_key not in cache:
                if parameter in PLET_CLASSIFICATION_PARAMETERS:
                    raw_value = row.get("value")
                    if pd.isna(raw_value) or str(raw_value).strip() == "":
                        raise ValueError(
                            f"No fixed value supplied for PLET classification "
                            f"'{parameter}'"
                        )
                    if parameter == "land_cover":
                        cache[cache_key] = normalize_plet_land_cover(raw_value)
                    else:
                        cache[cache_key] = normalize_plet_hsg(raw_value)
                else:
                    stats = _stats_from_row(
                        row, {"pid", "parameter", "sample_group", "distribution_id", "units"}
                    )
                    if not stats:
                        raise ValueError(
                            f"No value or statistics supplied for {cache_prefix} "
                            f"parameter '{parameter}'"
                        )
                    cache[cache_key] = _sample_stats(
                        ctx,
                        stats,
                        nonnegative=parameter not in {"load_delta"},
                    )
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
            variable_key, group_key = sample_group_key(
                row, pid=str(pid), variable=pollutant
            )
            key = (variable_key, group_key)
            if key not in cache:
                stats = _stats_from_row(
                    row, {"pid", "pollutant", "sample_group", "distribution_id", "units"}
                )
                if not stats:
                    raise ValueError(f"No concentration value or statistics supplied for {pid}/{pollutant}")
                cache[key] = _sample_stats(ctx, stats, nonnegative=True)
            values[pollutant] = cache[key]
        sampled.append(values)
    return sampled


def _sample_plet_hydrology(
    ctx: Any,
    land_cover: Any,
    hsg: Any,
    *,
    pid: str,
    cache: Dict[Tuple[str, str], float],
) -> Dict[str, Any]:
    """Sample CN and infiltration fraction for one parcel's PLET class pair.

    Production ``plet_rusle`` runs require a user-supplied long-form hydrology
    table stored in ``load_generation.hydrology_lookup``.  Each land-cover/HSG
    pairing has one ``cn`` row and one ``infiltration_fraction`` row, and each
    row may be a fixed value or a distribution.  Rows are sampled independently
    for each parcel unless an explicit ``sample_group`` requests a shared draw.
    """
    load_generation = getattr(ctx, "load_generation", {}) or {}
    table = load_generation.get("_hydrology_lookup_table")
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError(
            "plet_rusle requires a validated load_generation.hydrology_lookup table"
        )

    canonical_land_cover = normalize_plet_land_cover(land_cover)
    canonical_hsg = normalize_plet_hsg(hsg)
    subset = table[
        (table["land_cover"].astype(str) == canonical_land_cover)
        & (table["hsg"].astype(str) == canonical_hsg)
    ]
    values: Dict[str, Any] = {
        "land_cover": canonical_land_cover,
        "hsg": canonical_hsg,
    }
    for parameter in _PLET_DERIVED_PARAMETERS:
        rows = subset[subset["parameter"].astype(str) == parameter]
        if len(rows) != 1:
            raise ValueError(
                "hydrology_lookup must contain exactly one row for "
                f"land_cover={canonical_land_cover}, hsg={canonical_hsg}, "
                f"parameter={parameter}; found {len(rows)}"
            )
        row = rows.iloc[0]
        stats = stats_from_row(
            row,
            {
                "land_cover",
                "hsg",
                "parameter",
                "distribution_id",
                "sample_group",
                "units",
                "notes",
            },
        )
        if not stats:
            raise ValueError(
                f"hydrology_lookup has no value/distribution for "
                f"{canonical_land_cover}/{canonical_hsg}/{parameter}"
            )
        variable = f"hydrology:{canonical_land_cover}:{canonical_hsg}:{parameter}"
        variable_key, group_key = sample_group_key(
            row, pid=str(pid), variable=variable
        )
        cache_key = (variable_key, group_key)
        if cache_key not in cache:
            if parameter == "cn":
                cache[cache_key] = sample_stats_bounded(
                    ctx, stats, low=1.0e-9, high=100.0
                )
            else:
                cache[cache_key] = sample_stats_bounded(
                    ctx, stats, low=0.0, high=1.0
                )
        values[parameter] = float(cache[cache_key])
    return values


def calculate_load_diagnostics(parameters: Mapping[str, Any]) -> Dict[str, float]:
    """Calculate intermediate load-generation diagnostics.

    Parameters
    ----------
    parameters : Mapping[str, Any]
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


def calculate_load_components(
    parameters: Mapping[str, Any],
    concentrations: Mapping[str, float],
    groundwater_concentrations: Optional[Mapping[str, float]],
    pollutants: Sequence[str],
    *,
    groundwater_loads: bool = False,
    treat_groundwater_with_bmps: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate BMP-treatable pathways and protected groundwater yields.

    Pathway yields are derived directly from PLET/RUSLE runoff, infiltration,
    sediment, and pollutant concentrations. Surface runoff and sediment loads
    remain in the surface pathway. BMP-treatable groundwater is split between
    shallow and deep subsurface pathways using
    ``fraction_subsurface_shallow``.

    Parameters
    ----------
    parameters : Mapping[str, Any]
        Parcel parameters used to compute runoff and other drivers.
    concentrations : Mapping[str, float]
        Runoff concentrations keyed by pollutant name.
    groundwater_concentrations : Mapping[str, float] or None
        Groundwater concentrations keyed by pollutant name.
    pollutants : sequence of str
        Pollutants to calculate.
    groundwater_loads : bool, optional
        Whether groundwater concentrations should contribute to pollutant
        loads. Default is ``False``.
    treat_groundwater_with_bmps : bool, optional
        Whether groundwater loads should be exposed to pathway-specific BMP
        efficiencies. When ``False``, groundwater is returned as a separate
        protected load so pathway BMP efficiencies cannot reduce it. Default
        is ``False``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The BMP-treatable surface, shallow-subsurface, and deep-subsurface
        loads with shape ``(n_pollutants, 3)``, followed by the protected
        groundwater loads with shape ``(n_pollutants,)``.
    """

    missing = [
        name for name in _REQUIRED_RESOLVED_PLET if name not in parameters
    ]
    if missing:
        raise ValueError(f"PLET inputs are missing required parameters: {missing}")

    _, _, _, annual_runoff_in = plet_annual_surface_runoff_in(parameters)
    runoff_l_ha = annual_runoff_in * INCH_OVER_HA_TO_LITERS
    infiltration_l_ha = plet_annual_infiltration_in(parameters) * INCH_OVER_HA_TO_LITERS

    has_rusle = all(name in parameters for name in _REQUIRED_RUSLE)
    sediment_kg_ha = rusle_sediment_yield_kg_ha(parameters) if has_rusle else 0.0
    enrichment_ratio = max(0.0, float(parameters.get("enrichment_ratio", 2.0)))

    groundwater_concentrations = groundwater_concentrations or {}
    out = np.zeros((len(pollutants), len(PATHWAY_NAMES)), dtype=float)
    untreated_groundwater = np.zeros(len(pollutants), dtype=float)
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
        if groundwater_loads and pol != "TSS" and treat_groundwater_with_bmps:
            if "fraction_subsurface_shallow" not in parameters:
                raise ValueError(
                    "BMP-treatable groundwater loads require the PLET "
                    "parameter 'fraction_subsurface_shallow' (0 to 1)"
                )
            fraction_subsurface_shallow = float(
                np.clip(parameters["fraction_subsurface_shallow"], 0.0, 1.0)
            )
        else:
            fraction_subsurface_shallow = 0.0

        if pol == "TSS":
            surface_load = sediment_kg_ha if has_rusle else runoff_load
            shallow_load = 0.0
            deep_load = 0.0
        elif pol == "TN":
            sediment_fraction = max(0.0, float(parameters.get("sediment_n_pct", 0.0))) / 100.0
            surface_load = runoff_load + sediment_kg_ha * sediment_fraction * enrichment_ratio
            shallow_load = groundwater_load * fraction_subsurface_shallow
            deep_load = groundwater_load * (1.0 - fraction_subsurface_shallow)
        elif pol == "TP":
            sediment_fraction = max(0.0, float(parameters.get("sediment_p_pct", 0.0))) / 100.0
            surface_load = runoff_load + sediment_kg_ha * sediment_fraction * enrichment_ratio
            shallow_load = groundwater_load * fraction_subsurface_shallow
            deep_load = groundwater_load * (1.0 - fraction_subsurface_shallow)
        else:
            surface_load = runoff_load
            shallow_load = groundwater_load * fraction_subsurface_shallow
            deep_load = groundwater_load * (1.0 - fraction_subsurface_shallow)

        load_multiplier = max(
            0.0,
            float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)),
        )
        out[idx, 0] = max(0.0, surface_load) * load_multiplier
        if treat_groundwater_with_bmps:
            out[idx, 1] = max(0.0, shallow_load) * load_multiplier
            out[idx, 2] = max(0.0, deep_load) * load_multiplier
        else:
            untreated_groundwater[idx] = max(0.0, groundwater_load) * load_multiplier
    return out, untreated_groundwater



def calculate_plet_pathway_yields(
    parameters: Mapping[str, Any],
    concentrations: Mapping[str, float],
    groundwater_concentrations: Optional[Mapping[str, float]],
    pollutants: Sequence[str],
) -> np.ndarray:
    """Calculate the two PLET/RUSLE pathways: surface and subsurface.

    Surface nutrient load is runoff-derived load plus any RUSLE sediment-bound
    nutrient contribution. Subsurface nutrient load is groundwater concentration
    multiplied by PLET annual infiltration volume. The infiltration fraction in
    ``parameters`` is resolved from the PLET land-cover/HSG lookup table before
    this function is called. TSS has no subsurface component.
    """
    missing = [name for name in _REQUIRED_RESOLVED_PLET if name not in parameters]
    if missing:
        raise ValueError(f"PLET inputs are missing required parameters: {missing}")

    _, _, _, annual_runoff_in = plet_annual_surface_runoff_in(parameters)
    runoff_l_ha = annual_runoff_in * INCH_OVER_HA_TO_LITERS
    infiltration_l_ha = plet_annual_infiltration_in(parameters) * INCH_OVER_HA_TO_LITERS
    has_rusle = all(name in parameters for name in _REQUIRED_RUSLE)
    sediment_kg_ha = rusle_sediment_yield_kg_ha(parameters) if has_rusle else 0.0
    enrichment_ratio = max(0.0, float(parameters.get("enrichment_ratio", 2.0)))
    groundwater_concentrations = groundwater_concentrations or {}

    out = np.zeros((len(pollutants), len(PLET_PATHWAY_NAMES)), dtype=float)
    for idx, pollutant in enumerate(pollutants):
        pol = str(pollutant).upper()
        runoff_load = (
            max(0.0, float(concentrations.get(pol, 0.0)))
            * runoff_l_ha / 1_000_000.0
        )
        subsurface_load = 0.0
        if pol != "TSS":
            subsurface_load = (
                max(0.0, float(groundwater_concentrations.get(pol, 0.0)))
                * infiltration_l_ha / 1_000_000.0
            )

        if pol == "TSS":
            surface_load = sediment_kg_ha if has_rusle else runoff_load
        elif pol == "TN":
            sediment_fraction = max(0.0, float(parameters.get("sediment_n_pct", 0.0))) / 100.0
            surface_load = runoff_load + sediment_kg_ha * sediment_fraction * enrichment_ratio
        elif pol == "TP":
            sediment_fraction = max(0.0, float(parameters.get("sediment_p_pct", 0.0))) / 100.0
            surface_load = runoff_load + sediment_kg_ha * sediment_fraction * enrichment_ratio
        else:
            surface_load = runoff_load

        multiplier = max(0.0, float(parameters.get(f"load_multiplier_{pol.lower()}", 1.0)))
        out[idx, 0] = max(0.0, surface_load) * multiplier
        out[idx, 1] = max(0.0, subsurface_load) * multiplier
    return out


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

    parameters: List[Dict[str, Any]] = []
    has_rusle: List[bool] = []
    yields = np.zeros((len(parcel_ids), len(ctx.pollutants)), dtype=float)
    pathway_yields = np.zeros((len(parcel_ids), len(ctx.pollutants), len(PLET_PATHWAY_NAMES)), dtype=float)
    untreated_groundwater_yields = np.zeros((len(parcel_ids), len(ctx.pollutants)), dtype=float)
    hydrology_cache: Dict[Tuple[str, str], float] = {}
    for i, pid in enumerate(parcel_ids):
        missing_plet = [
            name for name in _REQUIRED_PLET_INPUTS if name not in plet[i]
        ]
        if missing_plet:
            raise ValueError(
                f"PLET inputs for pid={pid} are missing required parameters: {missing_plet}"
            )
        supplied_derived = [
            name for name in _PLET_DERIVED_PARAMETERS if name in plet[i]
        ]
        if supplied_derived:
            raise ValueError(
                f"PLET inputs for pid={pid} may not specify {supplied_derived}; "
                "these values must be defined in load_generation.hydrology_lookup "
                "for the parcel's land_cover and hsg"
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
            if pol != "TSS" and pol not in groundwater_concentrations[i]:
                raise ValueError(
                    f"Groundwater concentration for pid={pid}, pollutant={pol} is required in plet_rusle mode"
                )
            if pol == "TSS" and not rusle[i] and pol not in concentrations[i]:
                raise ValueError(
                    f"TSS for pid={pid} requires complete RUSLE inputs or a TSS concentration"
                )

        derived_hydrology = _sample_plet_hydrology(
            ctx,
            plet[i]["land_cover"],
            plet[i]["hsg"],
            pid=pid,
            cache=hydrology_cache,
        )
        combined = dict(plet[i])
        combined.update(derived_hydrology)
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
        parcel_pathway_yields = calculate_plet_pathway_yields(
            combined,
            concentrations[i],
            groundwater_concentrations[i],
            ctx.pollutants,
        )
        pathway_yields[i, :, :] = parcel_pathway_yields
        # Compatibility field retained for callers/tests that inspect the old
        # protected-groundwater array. Actual PLET subsurface load is now in
        # pathway_yields[:, :, 1], so this stays zero.
        untreated_groundwater_yields[i, :] = 0.0
        yields[i, :] = np.sum(pathway_yields[i, :, :], axis=1)

    state = LoadState(
        parcel_ids=parcel_ids,
        parameters=parameters,
        concentrations=concentrations,
        groundwater_concentrations=groundwater_concentrations,
        has_rusle=has_rusle,
        pollutants=list(ctx.pollutants),
        pathway_yields=pathway_yields,
        untreated_groundwater_yields=untreated_groundwater_yields,
        baseline_parameters=[dict(values) for values in parameters],
        baseline_concentrations=[dict(values) for values in concentrations],
        baseline_groundwater_concentrations=[dict(values) for values in groundwater_concentrations],
        baseline_pathway_yields=pathway_yields.copy(),
        baseline_untreated_groundwater_yields=untreated_groundwater_yields.copy(),
    )
    return yields.copy(), state
