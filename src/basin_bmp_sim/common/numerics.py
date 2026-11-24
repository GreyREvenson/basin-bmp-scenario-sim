"""Shared numerical helpers used by scientific accounting and reporting."""

from __future__ import annotations

from typing import Optional
import numpy as np


def _safe_mass_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return a dimensionless mass ratio, or ``None`` when undefined.

        Parameters
        ----------
        numerator : float
            Numerator of the dimensionless ratio.
        denominator : float
            Denominator of the dimensionless ratio.

        Returns
        -------
        Optional[float]
            Dimensionless ratio, or ``None`` when the denominator is not positive or finite.

    """
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return numerator / denominator
