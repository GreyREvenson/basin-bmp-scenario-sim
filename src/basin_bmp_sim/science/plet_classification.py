"""Plet classification for the annual PLET/RUSLE model."""

from __future__ import annotations
from typing import Any
from basin_bmp_sim.domain.constants import (
    PLET_HSG_VALUES,
    PLET_LAND_COVER_ALIASES as _PLET_LAND_COVER_ALIASES,
    PLET_LAND_COVERS,
    PLET_PARAMETER_ALIASES as _PARAMETER_ALIASES,
)



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
