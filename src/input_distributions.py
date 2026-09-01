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

import re
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
    """Parse a percentile-style label into its numeric percentile.

    Parameters
    ----------
    name : Any
        Input name or label.

    Returns
    -------
    Optional[int]
        Percentile as an integer from 0 through 100, or ``None`` when
        unrecognized.
    """
    label = str(name).strip().lower()
    if label.startswith("p") and label[1:].isdigit():
        value = int(label[1:])
        if 0 <= value <= 100:
            return value
    return None


def _canonical_statistic_name(name: Any) -> Optional[str]:
    """Return the canonical statistic represented by a column label.

    Parameters
    ----------
    name : Any
        Input column label.

    Returns
    -------
    str or None
        Canonical statistic name, or ``None`` when the label is not a
        recognized statistic column.
    """
    label = str(name).strip().lower()
    canonical = _STAT_ALIASES.get(label, label)
    if canonical in _CANONICAL_NAMED_STATS:
        return canonical

    percentile = _percentile_number(canonical)
    if percentile is None:
        return None
    if percentile == 0:
        return "min"
    if percentile == 100:
        return "max"
    return f"p{percentile}"


def _validate_percentile_style_label(name: Any) -> None:
    """Reject malformed or out-of-range percentile-like column labels.

    Labels that clearly represent an attempted percentile, such as ``p-5``,
    ``p5.5``, or ``p105``, must not be silently treated as metadata. Valid
    percentile columns use integer labels from ``p0`` through ``p100``.

    Parameters
    ----------
    name : Any
        Input column label.

    Raises
    ------
    ValueError
        If the label looks like a percentile statistic but is not supported.
    """
    label = str(name).strip().lower()
    if _canonical_statistic_name(label) is not None:
        return
    if re.fullmatch(r"p[+-]?(?:\d+(?:\.\d*)?|\.\d+)", label):
        raise ValueError(
            f"Invalid percentile statistic column {name!r}; use an integer "
            "percentile label from p0 through p100"
        )


def statistic_columns(columns: Iterable[Any]) -> list[str]:
    """Return recognized value/distribution columns in stable input order.

    Parameters
    ----------
    columns : Iterable[Any]
        Input column labels.

    Returns
    -------
    list[str]
        Recognized statistic column names in stable input order.

    Raises
    ------
    ValueError
        If a column looks like a percentile statistic but has an invalid label.
    """
    result: list[str] = []
    for column in columns:
        _validate_percentile_style_label(column)
        label = str(column).strip().lower()
        if _canonical_statistic_name(label) is not None and label not in result:
            result.append(label)
    return result


def stats_from_row(row: Mapping[str, Any], exclude: Iterable[str] = ()) -> Dict[str, float]:
    """Extract normalized numeric sampling statistics from a row.

    Metadata such as units, distribution IDs, and sample groups are ignored.
    Aliases are normalized to ``mean``, ``sd``, ``min`` and ``max``. Multiple
    populated columns that resolve to the same statistic are rejected rather
    than being applied in last-column-wins order.

    Parameters
    ----------
    row : Mapping[str, Any]
        Input table row.
    exclude : Iterable[str]
        Column names to exclude from statistic extraction.

    Returns
    -------
    Dict[str, float]
        Normalized numeric statistics extracted from the input row.

    Raises
    ------
    ValueError
        If a percentile-like column label is invalid or multiple populated
        columns define the same canonical statistic.
    """
    excluded = {str(value).strip().lower() for value in exclude}
    excluded.update({DISTRIBUTION_ID, SAMPLE_GROUP, "units", "unit", "notes"})

    for key in row.keys():
        label = str(key).strip().lower()
        if label not in excluded:
            _validate_percentile_style_label(key)

    out: Dict[str, float] = {}
    source_labels: Dict[str, str] = {}
    for key, value in row.items():
        label = str(key).strip().lower()
        if label in excluded or pd.isna(value):
            continue
        canonical = _canonical_statistic_name(label)
        if canonical is None:
            continue
        if canonical in out:
            first_label = source_labels[canonical]
            raise ValueError(
                f"Multiple populated columns define statistic {canonical!r}: "
                f"{first_label!r} and {label!r}"
            )
        out[canonical] = float(value)
        source_labels[canonical] = label
    return out


def _nonblank(value: Any) -> bool:
    """Return whether a value contains nonblank input.

        Parameters
        ----------
        value : Any
            Input value to normalize or evaluate.

        Returns
        -------
        bool
            ``True`` when the value is nonblank; otherwise ``False``.
        
    """
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def sample_group_key(row: Mapping[str, Any], *, pid: str, variable: str) -> Tuple[str, str]:
    """Return a cache key for optional shared draws.

        Without an explicit ``sample_group``, wildcard defaults are sampled
        independently for each parcel. Reusing a ``distribution_id`` never creates
        correlation by itself.

        Parameters
        ----------
        row : Mapping[str, Any]
            Input table row.
        pid : str
            Parcel identifier.
        variable : str
            Variable name used to distinguish sampled values.

        Returns
        -------
        Tuple[str, str]
            Cache key identifying the random-draw group.
        
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

        Parameters
        ----------
        ctx : Any
            Active scenario or model context.
        stats : Mapping[str, float]
            Numeric sampling statistics for one value or distribution.
        low : Optional[float]
            Optional lower bound for sampled values.
        high : Optional[float]
            Optional upper bound for sampled values.

        Returns
        -------
        float
            Sampled numeric value satisfying the requested bounds.

        Raises
        ------
        ValueError
            If distribution statistics are insufficient, bounds do not overlap, or the sampled value violates a requested bound.
        
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
