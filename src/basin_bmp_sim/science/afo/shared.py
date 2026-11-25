"""Sampling and validation primitives shared by AFO processes."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.constants import AFO_SPECIES_ALIASES as _SPECIES_ALIAS, AFO_SPECIES
from basin_bmp_sim.common.distributions import stats_from_row, sample_stats_bounded
from basin_bmp_sim.domain.statistics_validation import validate_named_parameter_stats

def _animal_species(value: Any) -> str:
    species = str(value).strip().lower()
    species = _SPECIES_ALIAS.get(species, species)
    if species not in AFO_SPECIES:
        raise ValueError(f"Unknown AFO species {value!r}; expected one of {', '.join(AFO_SPECIES)}")
    return species


def _need_columns(df: pd.DataFrame, name: str, required: Sequence[str]) -> None:
    missing = set(required).difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _finite(value: Any, name: str, low: float = 0., high: float = float("inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out) or not low <= out <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return out


def _sample_stats(ctx: Any, row: Mapping[str, Any], *, prefix: str = "", kind: str = "nonnegative") -> float:
    if prefix:
        stats = {str(key)[len(prefix):]: value for key, value in row.items()
                 if str(key).startswith(prefix) and not pd.isna(value)}
    else:
        stats = stats_from_row(row, {"land_cover", "pollutant", "sample_group", "units"})
    if not stats:
        raise ValueError(f"Missing fixed value or distribution for {prefix or 'concentration'}")
    return float(ctx._sample_from_stats(stats, kind=kind))


def _draw(
    ctx: Any,
    row: Mapping[str, Any],
    key: str,
    *,
    default: float | None = None,
    fraction: bool = False,
    minimum: float = 0.,
    maximum: float | None = None,
) -> float:
    """Sample a named scalar or distribution with shared validation semantics."""
    stats = validate_named_parameter_stats(
        row, key, f"AFO sample {row.get('afo_id', '?')}",
        required=default is None, low=minimum, high=(1. if fraction else maximum),
    )
    if stats:
        if set(stats) == {"value"} and key in row and pd.notna(row[key]):
            # Preserve the existing no-RNG path for deterministic named values.
            value = float(stats["value"])
        elif maximum is not None or minimum > 0.:
            value = sample_stats_bounded(ctx, stats, low=minimum, high=maximum)
        else:
            value = ctx._sample_from_stats(
                stats, kind="fraction" if fraction else "nonnegative"
            )
    else:
        value = default
    return _finite(value, key, minimum, 1 if fraction else
                   (maximum if maximum is not None else float("inf")))


def _nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
