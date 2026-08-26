"""Shared input-distribution parsing, validation, and sampling utilities.

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

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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
        canonical = _STAT_ALIASES.get(label, label)
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
        canonical = _STAT_ALIASES.get(label, label)
        percentile = _percentile_number(canonical)
        if percentile is not None:
            canonical = f"p{percentile}"
        if canonical in _CANONICAL_NAMED_STATS or percentile is not None:
            out[canonical] = float(value)
    return out


def _nonblank(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _row_stats_raw(row: Mapping[str, Any]) -> Dict[str, float]:
    return stats_from_row(row)


def validate_numeric_distribution_rows(df: pd.DataFrame, label: str) -> None:
    """Validate the standardized numeric value/distribution contract row-by-row.

    Accepted forms are:

    * ``value`` (fixed value), by itself;
    * ``mean`` + ``sd`` (optionally bounded by ``min``/``max``);
    * ``min`` + ``max`` (uniform when no other statistics are present);
    * ``min`` + ``max`` plus one or more percentile columns.

    A row may not combine a fixed ``value`` with distribution statistics.
    Distribution references must be resolved before calling this function.
    """
    if df is None or df.empty:
        return

    for index, row in df.iterrows():
        distribution_id = row.get(DISTRIBUTION_ID, None)
        stats = _row_stats_raw(row)
        if not stats:
            suffix = f" (distribution_id={distribution_id!r})" if _nonblank(distribution_id) else ""
            raise ValueError(f"{label} row {index} has no fixed value or distribution statistics{suffix}")

        has_value = "value" in stats
        distribution_keys = set(stats) - {"value"}
        if has_value and distribution_keys:
            raise ValueError(
                f"{label} row {index} mixes fixed value with distribution statistics; "
                "use either value or a distribution, not both"
            )

        for key, value in stats.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{label} row {index} statistic {key!r} must be finite")

        if has_value:
            continue

        has_mean = "mean" in stats
        has_sd = "sd" in stats
        has_min = "min" in stats
        has_max = "max" in stats
        percentiles = sorted(
            (p, stats[f"p{p}"])
            for p in range(1, 100)
            if f"p{p}" in stats
        )

        if has_sd and float(stats["sd"]) < 0.0:
            raise ValueError(f"{label} row {index} sd must be >= 0")
        if has_min != has_max:
            raise ValueError(f"{label} row {index} must provide both min and max when either is supplied")
        if has_min and float(stats["min"]) > float(stats["max"]):
            raise ValueError(f"{label} row {index} has min > max")

        valid = (has_mean and has_sd) or (has_min and has_max)
        if not valid:
            raise ValueError(
                f"{label} row {index} needs value, mean+sd, or min+max "
                "(with optional percentiles)"
            )
        if percentiles and not (has_min and has_max):
            raise ValueError(
                f"{label} row {index} percentile distributions require min and max endpoints"
            )
        if percentiles and (has_mean or has_sd):
            raise ValueError(
                f"{label} row {index} mixes percentile and normal-distribution statistics; "
                "use min/max + percentile columns OR mean/sd (optionally with min/max)"
            )

        if has_min and has_max:
            points = [(0, float(stats["min"]))]
            points.extend((p, float(v)) for p, v in percentiles)
            points.append((100, float(stats["max"])))
            for (p0, q0), (p1, q1) in zip(points[:-1], points[1:]):
                if q1 < q0:
                    raise ValueError(
                        f"{label} row {index} distribution is not monotonic: "
                        f"p{p0}={q0} > p{p1}={q1}"
                    )
            if has_mean and not (float(stats["min"]) <= float(stats["mean"]) <= float(stats["max"])):
                raise ValueError(f"{label} row {index} mean must lie between min and max")


def validate_distribution_bounds(
    df: pd.DataFrame,
    label: str,
    *,
    parameter_col: str,
    bounds: Mapping[str, Tuple[Optional[float], Optional[float]]],
) -> None:
    """Validate fixed/support statistics against parameter-specific bounds."""
    if df is None or df.empty:
        return
    for index, row in df.iterrows():
        parameter = str(row[parameter_col]).strip().lower()
        if parameter not in bounds:
            continue
        low, high = bounds[parameter]
        stats = _row_stats_raw(row)
        # mean is checked too; sd itself is not a sampled support value.
        for name, value in stats.items():
            if name == "sd":
                continue
            numeric = float(value)
            if low is not None and numeric < low:
                raise ValueError(
                    f"{label} row {index} {parameter}.{name}={numeric} is below minimum {low}"
                )
            if high is not None and numeric > high:
                raise ValueError(
                    f"{label} row {index} {parameter}.{name}={numeric} exceeds maximum {high}"
                )


def load_distribution_catalog(path: Any, logger: Any = None) -> Optional[pd.DataFrame]:
    """Load an optional reusable distribution catalog.

    The catalog has one row per ``distribution_id`` and the same statistic
    columns accepted by all other numeric input tables.
    """
    if path is None:
        return None
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    frames = []
    for item in paths:
        if logger is not None:
            logger.verbose(f"Reading reusable input distributions from {item}")
        frame = pd.read_csv(item)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if DISTRIBUTION_ID not in frame.columns:
            raise ValueError(f"input_distributions ({item}) is missing required column '{DISTRIBUTION_ID}'")
        frames.append(frame)
    catalog = pd.concat(frames, ignore_index=True)
    raw_ids = catalog[DISTRIBUTION_ID]
    blank_ids = raw_ids.isna() | raw_ids.astype(str).str.strip().eq("")
    if blank_ids.any():
        rows = catalog.index[blank_ids].tolist()
        raise ValueError(
            f"input_distributions contains blank distribution_id values at rows {rows}"
        )
    catalog[DISTRIBUTION_ID] = raw_ids.astype(str).str.strip()
    dup = catalog[DISTRIBUTION_ID].duplicated(keep=False)
    if dup.any():
        ids = sorted(catalog.loc[dup, DISTRIBUTION_ID].unique().tolist())
        raise ValueError(f"input_distributions contains duplicate distribution_id values: {ids}")
    validate_numeric_distribution_rows(catalog, "input_distributions")
    return catalog.reset_index(drop=True)


def resolve_distribution_references(
    df: pd.DataFrame,
    catalog: Optional[pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    """Expand ``distribution_id`` references into inline statistics.

    The returned table keeps ``distribution_id`` for provenance, but downstream
    sampling sees ordinary inline statistics. A reference row may not also
    define inline statistics, which prevents ambiguous precedence rules.
    """
    out = df.copy()
    if DISTRIBUTION_ID not in out.columns:
        return out

    catalog_map: Dict[str, pd.Series] = {}
    if catalog is not None:
        catalog_map = {
            str(row[DISTRIBUTION_ID]).strip(): row
            for _, row in catalog.iterrows()
        }

    all_stat_cols = set(statistic_columns(out.columns))
    if catalog is not None:
        all_stat_cols.update(statistic_columns(catalog.columns))
    for column in all_stat_cols:
        if column not in out.columns:
            out[column] = np.nan
        # CSVs can be read with Arrow-backed string dtypes when a statistic
        # column contains a mixture of blanks and numeric-looking text.
        # Distribution expansion writes numeric values into these columns, so
        # normalize every recognized statistic column to a numeric dtype first.
        out[column] = pd.to_numeric(out[column], errors="coerce")

    for index, row in out.iterrows():
        ref = row.get(DISTRIBUTION_ID, None)
        if not _nonblank(ref):
            continue
        ref_id = str(ref).strip()
        inline = _row_stats_raw(row)
        if inline:
            raise ValueError(
                f"{label} row {index} specifies distribution_id={ref_id!r} and inline statistics; "
                "use one or the other"
            )
        if not catalog_map:
            raise ValueError(
                f"{label} row {index} references distribution_id={ref_id!r}, but no input_distributions catalog is configured"
            )
        if ref_id not in catalog_map:
            raise ValueError(f"{label} row {index} references unknown distribution_id={ref_id!r}")
        source = catalog_map[ref_id]
        for source_col in statistic_columns(source.index):
            value = source.get(source_col)
            if not pd.isna(value):
                out.at[index, source_col] = float(value)
        # Units can be defined once in the catalog, but a use-site unit wins.
        if "units" in source.index and ("units" not in out.columns or not _nonblank(row.get("units", None))):
            if "units" not in out.columns:
                out["units"] = np.nan
            if _nonblank(source.get("units", None)):
                out.at[index, "units"] = source.get("units")

    return out


def sample_group_key(row: Mapping[str, Any], *, pid: str, variable: str) -> Tuple[str, str]:
    """Return a cache key for optional shared draws.

    Without an explicit ``sample_group``, wildcard defaults are sampled
    independently for each parcel. Reusing a ``distribution_id`` never creates
    correlation by itself.
    """
    raw = row.get(SAMPLE_GROUP, None)
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
