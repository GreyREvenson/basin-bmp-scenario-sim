"""Shared input-distribution normalization and sampling utilities.

All numeric model inputs may use the same long-form statistics schema. A row
may contain either a fixed ``value`` or a distribution described by ``mean`` /
``sd``, ``min`` / ``max``, and optional percentile columns (``p05``, ``p50``,
``p95``, etc.). Reusable distributions may be stored once in a distribution
catalog and referenced by ``distribution_id``.

``distribution_id`` controls *which distribution is used*. It does not imply
that different parcels share the same random draw. Use ``sample_group`` only
when a shared draw is intentionally required.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

DISTRIBUTION_ID = "distribution_id"
SAMPLE_GROUP = "sample_group"

# Canonical names used by new files. Existing aliases remain accepted.
_STAT_ALIASES = {
    "average": "mean",
    "avg": "mean",
    "std": "sd",
    "minimum": "min",
    "maximum": "max",
    "p0": "min",
    "p100": "max",
}
_CANONICAL_NAMED_STATS = ("value", "mean", "sd", "min", "max")


def _percentile_number(name: Any) -> Optional[int]:
    label = str(name).strip().lower()
    if label.startswith("p") and label[1:].isdigit():
        value = int(label[1:])
        if 0 <= value <= 100:
            return value
    return None


def statistic_columns(columns: Iterable[Any]) -> list[str]:
    """Return recognized value/distribution columns in stable input order."""
    result: list[str] = []
    for column in columns:
        label = str(column).strip().lower()
        canonical = _STAT_ALIASES[label] if label in _STAT_ALIASES else label
        if canonical in _CANONICAL_NAMED_STATS or _percentile_number(canonical) is not None:
            if label not in result:
                result.append(label)
    return result


def stats_from_row(row: Mapping[str, Any], exclude: Iterable[str] = ()) -> Dict[str, float]:
    """Extract normalized numeric sampling statistics from a row.

    Metadata such as units, distribution IDs, and sample groups are ignored.
    Aliases are normalized to ``mean``, ``sd``, ``min`` and ``max``.
    """
    excluded = {str(value).strip().lower() for value in exclude}
    excluded.update({DISTRIBUTION_ID, SAMPLE_GROUP, "units", "unit", "notes"})
    out: Dict[str, float] = {}
    for key, value in row.items():
        label = str(key).strip().lower()
        if label in excluded or pd.isna(value):
            continue
        canonical = _STAT_ALIASES[label] if label in _STAT_ALIASES else label
        percentile = _percentile_number(canonical)
        if percentile is not None:
            canonical = f"p{percentile}"
        if canonical in _CANONICAL_NAMED_STATS or percentile is not None:
            out[canonical] = float(value)
    return out


def _nonblank(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""












def sample_group_key(row: Mapping[str, Any], *, pid: str, variable: str) -> Tuple[str, str]:
    """Return a cache key for optional shared draws.

    Without an explicit ``sample_group``, wildcard defaults are sampled
    independently for each parcel. Reusing a ``distribution_id`` never creates
    correlation by itself.
    """
    raw = row.get(SAMPLE_GROUP)
    if _nonblank(raw):
        return variable, f"group:{str(raw).strip()}"
    return variable, f"pid:{pid}"


def sample_stats_bounded(
    ctx: Any,
    stats: Mapping[str, float],
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> float:
    """Sample one numeric value using model sampling semantics and hard bounds.

    Unlike the generic sampler, this helper lets an input parameter impose
    physical bounds (for example CN in ``(0, 100]`` and infiltration fraction
    in ``[0, 1]``) even when the row is defined only by ``mean`` and ``sd``.
    """
    cols = {str(k).lower(): float(v) for k, v in stats.items()}
    has_min = "min" in cols
    has_max = "max" in cols
    has_mean = "mean" in cols
    has_sd = "sd" in cols
    has_percentiles = any(_percentile_number(k) not in (None, 0, 100) for k in cols)

    row_low = cols.get("min") if has_min else None
    row_high = cols.get("max") if has_max else None
    effective_low = low if row_low is None else (row_low if low is None else max(low, row_low))
    effective_high = high if row_high is None else (row_high if high is None else min(high, row_high))
    if effective_low is not None and effective_high is not None and effective_low > effective_high:
        raise ValueError("Distribution bounds do not overlap the parameter's allowed range")

    if "value" in cols:
        value = cols["value"]
    elif has_min and has_max and has_percentiles:
        value = float(ctx._piecewise_quantile_sample(cols, size=1)[0])
    elif has_mean and has_sd:
        value = float(
            ctx._trunc_normal(
                cols["mean"], cols["sd"], low=effective_low, high=effective_high, size=1
            )[0]
        )
    elif has_min and has_max and has_mean:
        sd = max((cols["max"] - cols["min"]) / 4.0, 1.0e-12)
        value = float(
            ctx._trunc_normal(
                cols["mean"], sd, low=effective_low, high=effective_high, size=1
            )[0]
        )
    elif has_min and has_max:
        lo = cols["min"] if effective_low is None else effective_low
        hi = cols["max"] if effective_high is None else effective_high
        value = float(ctx.rng.uniform(lo, hi))
    else:
        raise ValueError("Insufficient distribution statistics to sample")

    if low is not None and value < low:
        raise ValueError(f"Sampled value {value} is below allowed minimum {low}")
    if high is not None and value > high:
        raise ValueError(f"Sampled value {value} exceeds allowed maximum {high}")
    return float(value)
