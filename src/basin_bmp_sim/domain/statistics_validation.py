"""Shared statistical input validation, independent of file ingestion."""

from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from basin_bmp_sim.common.distributions import stats_from_row
from basin_bmp_sim.common.utils import is_nonblank
from basin_bmp_sim.domain.parameter_validation import (
    validate_physical_distribution_rows,
)


def require_columns(df: pd.DataFrame, required: Sequence[str], label: str, logger: Any = None) -> None:
    """Require a normalized input table to contain the requested columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        required : Sequence[str]
            Required column names.
        label : str
            Context label used in diagnostics and validation errors.
        logger : Any
            Logger used for diagnostic and progress messages.

        Raises
        ------
        ValueError
            If one or more required columns are absent.

    """
    del logger
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def validate_named_parameter_stats(
    row: Mapping[str, Any],
    parameter: str,
    label: str,
    *,
    required: bool = False,
    low: Optional[float] = None,
    high: Optional[float] = None,
    exclusive_low: bool = False,
) -> Dict[str, float]:
    """Validate one wide-format parameter using the standard model validator.

    ``parameter`` may be supplied as a fixed column (e.g. ``count``) OR as
    prefixed inline statistics (``count_value``, ``count_mean``, ...). Both
    formats cannot be populated together. Standard long-format model inputs
    and these named inputs share :func:`validate_numeric_distribution_rows`.

    Only the *supplied* center, value, endpoints and percentiles are checked
    against ``low``/``high``. A nonnegative ``sd`` is allowed for a normal
    distribution because bounded sampling truncates to the physical domain.
    """
    name = str(parameter).strip().lower()
    columns = {str(k).strip().lower(): v for k, v in row.items()}
    base = columns.get(name)
    prefix = name + "_"
    supplied = {
        key[len(prefix):]: value for key, value in columns.items()
        if key.startswith(prefix) and is_nonblank(value)
    }
    if is_nonblank(base):
        if supplied:
            raise ValueError(
                f"{label}: {name} mixes fixed value with distribution statistics "
                f"({name} and {sorted(prefix + key for key in supplied)}); "
                "provide exactly one representation"
            )
        supplied = {"value": base}
    if not supplied:
        if required:
            raise ValueError(f"{label}: {name} requires a fixed value or distribution")
        return {}
    stats = stats_from_row({"parameter": name, **supplied})
    if not stats:
        raise ValueError(f"{label}: {name} has no supported distribution statistics")
    validate_numeric_distribution_rows(
        pd.DataFrame([{"parameter": name, **stats}]), label
    )
    for statistic, value in stats.items():
        if statistic == "sd":
            continue
        if low is not None and (value <= low if exclusive_low else value < low):
            relation = ">" if exclusive_low else ">="
            raise ValueError(
                f"{label}: {name}.{statistic}={value:g} is outside physical domain "
                f"({relation} {low:g})"
            )
        if high is not None and value > high:
            raise ValueError(
                f"{label}: {name}.{statistic}={value:g} is outside physical domain "
                f"(<= {high:g})"
            )
    return stats


def validate_numeric_distribution_rows(df: pd.DataFrame, label: str) -> None:
    """Validate the standardized numeric value/distribution contract row-by-row.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        label : str
            Context label used in diagnostics and validation errors.

        Raises
        ------
        ValueError
            If a row contains incomplete, conflicting, non-finite, or non-monotonic distribution statistics.

    """
    if df is None or df.empty:
        return
    for index, row in df.iterrows():
        stats = stats_from_row(row)
        if not stats:
            raise ValueError(f"{label} row {index} has no fixed value or distribution statistics")
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
        percentiles = sorted((q, stats[f"p{q}"]) for q in range(1, 100) if f"p{q}" in stats)
        if has_sd and float(stats["sd"]) < 0.0:
            raise ValueError(f"{label} row {index} sd must be >= 0")
        if has_min != has_max:
            raise ValueError(f"{label} row {index} must provide both min and max when either is supplied")
        if has_min and float(stats["min"]) > float(stats["max"]):
            raise ValueError(f"{label} row {index} has min > max")
        if not ((has_mean and has_sd) or (has_min and has_max)):
            raise ValueError(
                f"{label} row {index} needs value, mean+sd, or min+max (with optional percentiles)"
            )
        if percentiles and not (has_min and has_max):
            raise ValueError(f"{label} row {index} percentile distributions require min and max endpoints")
        if percentiles and (has_mean or has_sd):
            raise ValueError(
                f"{label} row {index} mixes percentile and normal-distribution statistics; "
                "use min/max + percentile columns OR mean/sd (optionally with min/max)"
            )
        if has_min and has_max:
            points = [(0, float(stats["min"]))]
            points.extend((q, float(value)) for q, value in percentiles)
            points.append((100, float(stats["max"])))
            for (p0, q0), (p1, q1) in zip(points[:-1], points[1:]):
                if q1 < q0:
                    raise ValueError(
                        f"{label} row {index} distribution is not monotonic: p{p0}={q0} > p{p1}={q1}"
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
    """Validate fixed/support statistics against parameter-specific bounds.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        label : str
            Context label used in diagnostics and validation errors.
        parameter_col : str
            Column containing canonical parameter names.
        bounds : Mapping[str, Tuple[Optional[float], Optional[float]]]
            Allowed lower and upper bounds keyed by parameter name.

        Raises
        ------
        ValueError
            If a fixed value, mean, endpoint, or percentile lies outside the allowed parameter bounds.

    """
    if df is None or df.empty:
        return
    for index, row in df.iterrows():
        parameter = str(row[parameter_col]).strip().lower()
        if parameter not in bounds:
            continue
        low, high = bounds[parameter]
        stats = stats_from_row(row)
        for name, value in stats.items():
            if name == "sd":
                continue
            numeric = float(value)
            if low is not None and numeric < low:
                raise ValueError(f"{label} row {index} {parameter}.{name}={numeric} is below minimum {low}")
            if high is not None and numeric > high:
                raise ValueError(f"{label} row {index} {parameter}.{name}={numeric} exceeds maximum {high}")




def validate_stats_rows(df: pd.DataFrame, label: str) -> None:
    """Validate every row using the shared numeric input-distribution schema.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        label : str
            Context label used in diagnostics and validation errors.

    """
    validate_numeric_distribution_rows(df, label)
    validate_physical_distribution_rows(df, label)


def validate_unique_rows(df: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    """Reject duplicate rows for a logical table key.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        keys : Sequence[str]
            Logical key columns used to identify duplicate rows.
        label : str
            Context label used in diagnostics and validation errors.

        Raises
        ------
        ValueError
            If duplicate rows exist for the requested logical key.

    """
    duplicate = df.duplicated(list(keys), keep=False)
    if duplicate.any():
        preview = df.loc[duplicate, list(keys)].head(10).to_dict(orient="records")
        raise ValueError(f"{label} contains duplicate rows for {list(keys)}: {preview}")
