"""Domains for the configuration."""

from __future__ import annotations

from basin_bmp_sim.common.utils import is_nonblank
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.constants import (
    NONNEGATIVE_DOMAIN,
    PhysicalDomain,
    _PARAMETER_PHYSICAL_DOMAINS,
    _SAMPLING_KIND_PHYSICAL_DOMAINS,
    _TABLE_PHYSICAL_DOMAINS,
)
from basin_bmp_sim.common.distributions import stats_from_row



def physical_parameter_domain(parameter: Any) -> Optional[PhysicalDomain]:
    """Return the physical domain for a canonical model parameter, if known."""
    name = str(parameter).strip().lower()
    if name.startswith("load_multiplier_"):
        return NONNEGATIVE_DOMAIN
    return _PARAMETER_PHYSICAL_DOMAINS.get(name)


def physical_domain_for_sampling_kind(kind: Optional[str]) -> Optional[PhysicalDomain]:
    """Return the physical domain represented by a sampling semantic hint."""
    if kind is None:
        return None
    return _SAMPLING_KIND_PHYSICAL_DOMAINS.get(str(kind).strip().lower())


def sampling_kind_for_parameter(parameter: Any) -> Optional[str]:
    """Return the sampler semantic required by a physical parameter domain."""
    name = str(parameter).strip().lower()
    if name == "cn":
        return "cn"
    if name == "ia_ratio":
        return "ia_ratio"
    if name in {"infiltration_fraction", "rain_correction_fraction", "runoff_day_fraction", "sdr", "fraction_subsurface_shallow"}:
        return "fraction"
    if name in {"sediment_n_pct", "sediment_p_pct"}:
        return "percent"
    domain = physical_parameter_domain(name)
    if domain == NONNEGATIVE_DOMAIN:
        return "nonnegative"
    return None


def validate_scalar_in_domain(value: Any, domain: PhysicalDomain, label: str) -> float:
    """Validate one scalar against an explicit physical domain and return it."""
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    if not domain.contains(numeric):
        raise ValueError(
            f"{label}={numeric:g} is outside the allowed physical domain {domain.describe()}"
        )
    return numeric


def validate_parameter_value(parameter: Any, value: Any, *, label: Optional[str] = None) -> float:
    """Validate one model parameter value against its centralized physical domain."""
    name = str(parameter).strip().lower()
    domain = physical_parameter_domain(name)
    if domain is None:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{label or name} must be finite")
        return numeric
    return validate_scalar_in_domain(value, domain, label or name)


def validate_numeric_columns_in_domain(
    df: pd.DataFrame,
    columns: Sequence[str],
    domain: PhysicalDomain,
    label: str,
) -> None:
    """Validate dataframe columns against one physical numeric domain.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the columns to validate.
    columns : sequence of str
        Numeric columns that share the physical domain.
    domain : PhysicalDomain
        Allowed physical domain.
    label : str
        Dataset label used in validation errors.
    """
    if df is None or df.empty:
        return
    for column in columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
        if domain.low is not None:
            invalid |= (numeric < domain.low).to_numpy() if domain.low_inclusive else (numeric <= domain.low).to_numpy()
        if domain.high is not None:
            invalid |= (numeric > domain.high).to_numpy() if domain.high_inclusive else (numeric >= domain.high).to_numpy()
        if np.any(invalid):
            bad_index = df.index[np.asarray(invalid, dtype=bool)][0]
            bad_value = df.loc[bad_index, column]
            raise ValueError(
                f"{label} row {bad_index} {column}={bad_value!r} is outside the "
                f"allowed physical domain {domain.describe()}"
            )


def validate_physical_distribution_rows(df: pd.DataFrame, label: str) -> None:
    """Validate supplied distribution statistics against physical input domains.

    Normal rows are allowed for bounded quantities because the sampler treats
    the physical domain as explicit truncation support. The supplied mean and
    any explicit endpoints/percentiles must nevertheless be physically valid.
    Standard deviation is a spread parameter and is checked separately by
    :func:`validate_numeric_distribution_rows`.
    """
    if df is None or df.empty:
        return
    table_domain = _TABLE_PHYSICAL_DOMAINS.get(str(label).strip().lower())
    for index, row in df.iterrows():
        domain = table_domain
        parameter = None
        if "parameter" in row.index and is_nonblank(row.get("parameter")):
            parameter = str(row.get("parameter")).strip().lower()
            domain = physical_parameter_domain(parameter) or domain
        if domain is None:
            continue
        stats = stats_from_row(row)
        for statistic, value in stats.items():
            if statistic == "sd":
                continue
            item = f"{parameter}.{statistic}" if parameter else statistic
            validate_scalar_in_domain(
                value, domain, f"{label} row {index} {item}"
            )
