"""Build parcel pollutant loads for statistical and PLET/RUSLE scenarios.

This module converts scenario input tables into parcel-level annual pollutant
loads, splits those loads across flow pathways, and calculates load diagnostics
used by the scenario simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .constants import (
    INCH_OVER_HA_TO_LITERS,
    PLET_CLASSIFICATION_PARAMETERS,
    PLET_DERIVED_PARAMETERS as _PLET_DERIVED_PARAMETERS,
    PLET_HSG_VALUES,
    PLET_LAND_COVER_ALIASES as _PLET_LAND_COVER_ALIASES,
    PLET_LAND_COVERS,
    PLET_PARAMETER_ALIASES as _PARAMETER_ALIASES,
    PLET_PATHWAY_VALUES as PLET_PATHWAY_NAMES,
    PLET_PRECIPITATION_INPUT_ALTERNATIVES as _PLET_PRECIPITATION_INPUT_ALTERNATIVES,
    PLET_REQUIRED_INPUTS as _REQUIRED_PLET_INPUTS,
    PLET_REQUIRED_RESOLVED_INPUTS as _REQUIRED_RESOLVED_PLET,
    RUSLE_REQUIRED_INPUTS as _REQUIRED_RUSLE,
    TON_PER_ACRE_TO_KG_PER_HA,
)
from .input_distributions import (
    sample_group_key,
    stats_from_row,
)
from .input_config import (
    _load_plet_hydrology_records,
    _rows_for_pid,
    apply_plet_parameter_defaults,
)
from .input_validation import (
    NONNEGATIVE_DOMAIN,
    sampling_kind_for_parameter,
    validate_parameter_value,
    validate_plet_input_table,
    validate_scalar_in_domain,
)


# Compatibility re-exports: process algorithms now live under src.models.
from src.models.hydrology.runoff_plet_curve_number import (
    _plet_event_runoff_depth_in, plet_runoff_depth_in,
    plet_avg_rain_runoff_depth_in, _plet_corrected_annual_precip_in,
    plet_annual_surface_runoff_in,
)
from src.models.hydrology.infiltration_plet_balance import plet_annual_infiltration_in
from src.models.sediment.soil_loss_rusle import rusle_sediment_load_rate_kg_ha_yr
from src.models.nutrients.surface_load_plet_emc import (
    calculate_load_diagnostics, calculate_plet_pathway_load_rates,
)

@dataclass
class LoadState:
    """Container for per-parcel scenario state.

    The state stores current sampled values, baseline values, parcel ordering,
    and pathway-level load_rates so that BMP application can update the scenario in
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
    pathway_load_rates : numpy.ndarray
        Current pathway-specific parcel load_rates with shape
        ``(n_parcels, n_pollutants, 2)``. In PLET/RUSLE mode the two pathways
        are ``surface`` and ``subsurface``.
    untreated_groundwater_load_rates : numpy.ndarray
        Deprecated compatibility array. PLET/RUSLE subsurface loads are now
        included in the ``subsurface`` pathway, so this array is always zero.
    baseline_parameters : list[dict[str, Any]]
        Snapshot of the original parameter values.
    baseline_concentrations : list[dict[str, float]]
        Snapshot of the original runoff concentrations.
    baseline_groundwater_concentrations : list[dict[str, float]]
        Snapshot of the original groundwater concentrations.
    baseline_pathway_load_rates : numpy.ndarray or None
        Snapshot of the original pathway-specific load_rates.
    baseline_untreated_groundwater_load_rates : numpy.ndarray or None
        Snapshot of the original groundwater load_rates excluded from BMP
        treatment.
    """

    parcel_ids: List[str]
    parameters: List[Dict[str, Any]]
    concentrations: List[Dict[str, float]]
    groundwater_concentrations: List[Dict[str, float]]
    has_rusle: List[bool]
    pollutants: List[str]
    pathway_load_rates: np.ndarray
    untreated_groundwater_load_rates: np.ndarray
    baseline_parameters: List[Dict[str, Any]] = field(default_factory=list)
    baseline_concentrations: List[Dict[str, float]] = field(default_factory=list)
    baseline_groundwater_concentrations: List[Dict[str, float]] = field(default_factory=list)
    baseline_pathway_load_rates: Optional[np.ndarray] = None
    baseline_untreated_groundwater_load_rates: Optional[np.ndarray] = None

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
    return _PARAMETER_ALIASES[label] if label in _PARAMETER_ALIASES else label


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




def plet_hydrology_from_classifications(
    land_cover: Any,
    hsg: Any,
    *,
    lookup_path: Path,
) -> Dict[str, Any]:
    """Resolve fixed hydrology values from a lookup table.

        This deterministic helper requires an explicit user-supplied hydrology
        lookup path. The actual PLET/RUSLE scenario path samples the validated
        hydrology input table loaded from configuration, allowing each CN and
        infiltration fraction to be a fixed value or a distribution.

        Parameters
        ----------
        land_cover : Any
            PLET land-cover classification.
        hsg : Any
            Hydrologic soil group classification.
        lookup_path : Path
            Path to the user-supplied PLET hydrology lookup table.

        Returns
        -------
        Dict[str, Any]
            Resolved PLET hydrology values for the land-cover/HSG combination.
        
    """
    canonical_land_cover = normalize_plet_land_cover(land_cover)
    canonical_hsg = normalize_plet_hsg(hsg)
    records = _load_plet_hydrology_records(lookup_path)
    curve_number, infiltration_fraction = records[(canonical_land_cover, canonical_hsg)]
    return {
        "land_cover": canonical_land_cover,
        "hsg": canonical_hsg,
        "cn": curve_number,
        "infiltration_fraction": infiltration_fraction,
    }



















def _sample_stats(
    ctx: Any,
    stats: Mapping[str, float],
    *,
    kind: Optional[str] = None,
) -> float:
    """Sample one numeric value using an explicit physical sampling domain.

    Parameters
    ----------
    ctx : Any
        Object providing ``_sample_from_stats``.
    stats : Mapping[str, float]
        Sampling statistics or a fixed ``value`` entry.
    kind : str or None, optional
        Physical sampling semantic. Bounded Normal inputs are sampled as
        truncated Normal distributions; fixed/explicit support values outside
        the domain are rejected rather than clipped.

    Returns
    -------
    float
        Sampled numeric value.
    """
    return float(ctx._sample_from_stats(dict(stats), kind=kind))


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
                    stats = stats_from_row(
                        row, {"pid", "parameter", "sample_group", "units"}
                    )
                    if not stats:
                        raise ValueError(
                            f"No value or statistics supplied for {cache_prefix} "
                            f"parameter '{parameter}'"
                        )
                    cache[cache_key] = _sample_stats(
                        ctx,
                        stats,
                        kind=sampling_kind_for_parameter(parameter),
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
        defaults = table[table["pid"].isna()]
        try:
            target_pid = int(pid)
            pids = pd.to_numeric(table["pid"], errors="coerce")
            exact = table[table["pid"].notna() & (pids == target_pid)]
        except (TypeError, ValueError):
            exact = table[
                table["pid"].notna() & (table["pid"].astype(str) == str(pid))
            ]
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
                stats = stats_from_row(
                    row, {"pid", "pollutant", "sample_group", "units"}
                )
                if not stats:
                    raise ValueError(f"No concentration value or statistics supplied for {pid}/{pollutant}")
                cache[key] = _sample_stats(ctx, stats, kind="nonnegative")
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

        Production ``plet_rusle`` runs assemble a validated hydrology table from
        ``input_curve_number`` and ``input_infiltration_fraction`` across input scopes.
        Each land-cover/HSG pairing has one row for each variable, and each row may
        be a fixed value or a distribution. Rows are sampled independently
        for each parcel unless a ``sample_group`` requests a shared draw.
        HUC/domain rows receive automatic source-row groups during inheritance.

        Parameters
        ----------
        ctx : Any
            Active scenario or model context.
        land_cover : Any
            PLET land-cover classification.
        hsg : Any
            Hydrologic soil group classification.
        pid : str
            Parcel identifier.
        cache : Dict[Tuple[str, str], float]
            Cache of previously sampled values.

        Returns
        -------
        Dict[str, Any]
            Sampled PLET hydrology parameters for the parcel.

        Raises
        ------
        ValueError
            If the validated hydrology lookup is unavailable, ambiguous, or lacks statistics for the requested class pair.
        
    """
    load_generation = getattr(ctx, "load_generation", {}) or {}
    table = load_generation.get("_hydrology_lookup_table")
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError(
            "plet_rusle requires validated curve-number and infiltration-fraction input tables"
        )

    canonical_land_cover = normalize_plet_land_cover(land_cover)
    canonical_hsg = normalize_plet_hsg(hsg)
    subset = table[
        (table["land_cover"].astype(str) == canonical_land_cover)
        & (table["hsg"].astype(str) == canonical_hsg)
    ]
    if "pid" in subset:
        subset = subset[subset["pid"].map(str) == str(pid)]
    values: Dict[str, Any] = {
        "land_cover": canonical_land_cover,
        "hsg": canonical_hsg,
    }
    for parameter in _PLET_DERIVED_PARAMETERS:
        rows = subset[subset["parameter"].astype(str) == parameter]
        if len(rows) != 1:
            raise ValueError(
                "PLET hydrology inputs must contain exactly one row for "
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
                "sample_group",
                "units",
                "notes",
            },
        )
        if not stats:
            raise ValueError(
                f"PLET hydrology inputs have no value/distribution for "
                f"{canonical_land_cover}/{canonical_hsg}/{parameter}"
            )
        variable = f"hydrology:{canonical_land_cover}:{canonical_hsg}:{parameter}"
        variable_key, group_key = sample_group_key(
            row, pid=str(pid), variable=variable
        )
        cache_key = (variable_key, group_key)
        if cache_key not in cache:
            cache[cache_key] = _sample_stats(
                ctx, stats, kind=sampling_kind_for_parameter(parameter)
            )
        values[parameter] = float(cache[cache_key])
    return values







def initialize_plet_rusle_state(ctx: Any) -> Tuple[np.ndarray, LoadState]:
    """Create the initial parcel load state for a scenario.

    This function samples parcel-level parameters and concentrations, verifies
    that required inputs are present, computes baseline pathway load_rates, and
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

    # Scenario workers provide the full hydrologic parcel universe via
    # ``parcel_ids``.  Fall back to ``parcel_selection_ids`` only for legacy
    # direct callers/tests that construct a minimal context without parcel_ids.
    # TODO: remove legacy fallback to parcel_selection_ids. double-check that it
    # will not break existing tests or direct calls.
    parcel_ids = [
        str(pid)
        for pid in getattr(ctx, "parcel_ids", getattr(ctx, "parcel_selection_ids", []))
    ]
    plet = _sample_parameter_table(ctx, ctx.plet_inputs, parcel_ids, cache_prefix="plet")
    rusle = _sample_parameter_table(ctx, ctx.rusle_inputs, parcel_ids, cache_prefix="rusle")
    concentrations = _sample_concentrations(ctx, ctx.pollutant_concentrations, parcel_ids)
    groundwater_concentrations = _sample_concentrations(ctx, getattr(ctx, "groundwater_concentrations", None), parcel_ids)

    parameters: List[Dict[str, Any]] = []
    has_rusle: List[bool] = []
    load_rates = np.zeros((len(parcel_ids), len(ctx.pollutants)), dtype=float)
    pathway_load_rates = np.zeros((len(parcel_ids), len(ctx.pollutants), len(PLET_PATHWAY_NAMES)), dtype=float)
    untreated_groundwater_load_rates = np.zeros((len(parcel_ids), len(ctx.pollutants)), dtype=float)
    hydrology_cache: Dict[Tuple[str, str], float] = {}
    for i, pid in enumerate(parcel_ids):
        # Input completeness and concentration coverage are validated once in
        # input_config before scenario workers start.
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
        combined = apply_plet_parameter_defaults(combined, ctx.pollutants)
        parameters.append(combined)
        has_rusle.append(all(name in combined for name in _REQUIRED_RUSLE))
        parcel_pathway_load_rates = calculate_plet_pathway_load_rates(
            combined,
            concentrations[i],
            groundwater_concentrations[i],
            ctx.pollutants,
        )
        pathway_load_rates[i, :, :] = parcel_pathway_load_rates
        # Compatibility field retained for callers/tests that inspect the old
        # protected-groundwater array. Actual PLET subsurface load is now in
        # pathway_load_rates[:, :, 1], so this stays zero.
        # TODO: remove untreated_groundwater_load_rates but confirm that it
        # will not break existing functionality
        untreated_groundwater_load_rates[i, :] = 0.0
        load_rates[i, :] = np.sum(pathway_load_rates[i, :, :], axis=1)

    state = LoadState(
        parcel_ids=parcel_ids,
        parameters=parameters,
        concentrations=concentrations,
        groundwater_concentrations=groundwater_concentrations,
        has_rusle=has_rusle,
        pollutants=list(ctx.pollutants),
        pathway_load_rates=pathway_load_rates,
        untreated_groundwater_load_rates=untreated_groundwater_load_rates,
        baseline_parameters=[dict(values) for values in parameters],
        baseline_concentrations=[dict(values) for values in concentrations],
        baseline_groundwater_concentrations=[dict(values) for values in groundwater_concentrations],
        baseline_pathway_load_rates=pathway_load_rates.copy(),
        baseline_untreated_groundwater_load_rates=untreated_groundwater_load_rates.copy(),
    )
    return load_rates.copy(), state
