"""Small shared helper functions used in many places."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def ci_get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Look up a dictionary key without caring about uppercase/lowercase.

    Parameters
    ----------
    d : Mapping[str, Any]
        Input dictionary-like object.
    key : str
        Key to look up, case-insensitively.
    default : Any, optional
        Default value when key is not found.

    Returns
    -------
    Any
        Value if found; otherwise default.
    """
    key_l = str(key).lower()
    for k, v in d.items():
        if str(k).lower() == key_l:
            return v
    return default


def normalize_columns(df: Any) -> Any:
    """Rename table columns to lowercase text (in place).

    Parameters
    ----------
    df : pandas.DataFrame
        Input table; its column names are changed directly.

    Returns
    -------
    pandas.DataFrame
        The same table object, returned for convenience.
    """
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def normalize_pollutant_label(label: str) -> str:
    """Convert a pollutant name to the standard code used by this model.

    Parameters
    ----------
    label : str
        Arbitrary label (e.g., 'tp', 'TP', 'phosphorus').

    Returns
    -------
    str
        Standard pollutant code ('TN', 'TP', 'TSS').

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
    """Find percentile-style column names like p10, p50, or p95.

    Parameters
    ----------
    cols : Iterable[Any]
        Column labels to inspect.

    Returns
    -------
    Dict[int, Any]
        Map from percentile number to the original column label.
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