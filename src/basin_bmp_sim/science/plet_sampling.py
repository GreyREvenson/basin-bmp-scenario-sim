"""Plet sampling for the annual PLET/RUSLE model."""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import pandas as pd
from basin_bmp_sim.domain.constants import (
    PLET_CLASSIFICATION_PARAMETERS,
    PLET_DERIVED_PARAMETERS as _PLET_DERIVED_PARAMETERS,
)
from basin_bmp_sim.common.distributions import sample_group_key, stats_from_row
from basin_bmp_sim.science.plet_rows import _rows_for_pid
from basin_bmp_sim.domain.parameter_validation import sampling_kind_for_parameter
from basin_bmp_sim.science.plet_classification import (
    canonical_parameter_name,
    normalize_plet_hsg,
    normalize_plet_land_cover,
)


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
