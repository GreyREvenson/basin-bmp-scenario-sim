"""Shared utility helpers.

This module contains small reusable helpers for case-insensitive dictionary
lookups, column normalization, pollutant label normalization, and parsing
percentile-style column names.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def ci_get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Look up a mapping key case-insensitively.

    Parameters
    ----------
    d : Mapping[str, Any]
        Mapping to search.
    key : str
        Key to look up.
    default : Any, optional
        Value returned when no matching key is found. Default is ``None``.

    Returns
    -------
    Any
        Matching value if found, otherwise ``default``.
    """
    key_l = str(key).lower()
    for k, v in d.items():
        if str(k).lower() == key_l:
            return v
    return default


def normalize_columns(df: Any) -> Any:
    """Normalize dataframe column names to lowercase text.

    Parameters
    ----------
    df : Any
        Table-like object with a ``columns`` attribute.

    Returns
    -------
    Any
        The same object, returned for convenience.
    """
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def normalize_pollutant_label(label: str) -> str:
    """Normalize a pollutant label to the canonical model code.

    Parameters
    ----------
    label : str
        Raw pollutant label.

    Returns
    -------
    str
        Canonical pollutant code.

    Raises
    ------
    ValueError
        If the label cannot be mapped to a known canonical name.
    """
    from .constants import POLLUTANT_ALIAS_MAP

    canonical = POLLUTANT_ALIAS_MAP.get(str(label).strip().lower())
    if canonical is None:
        raise ValueError(f"Unknown pollutant label: {label}")
    return canonical


def parse_percent_keys(cols: Iterable[Any]) -> Dict[int, Any]:
    """Extract percentile-style column labels.

    Parameters
    ----------
    cols : Iterable[Any]
        Column labels to inspect for percentile names.

    Returns
    -------
    dict[int, Any]
        Mapping from percentile number to the original column label.
    """
    import re

    percents: Dict[int, Any] = {}
    for c in cols:
        c_l = str(c).lower().strip()
        m = re.fullmatch(r"p(\d{1,2}|100)", c_l)
        if m:
            p = int(m.group(1))
            percents[p] = c
    return percents