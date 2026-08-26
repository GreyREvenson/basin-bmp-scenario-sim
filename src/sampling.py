"""Random sampling helpers for model inputs.

This module provides the sampling primitives used to draw values from fixed
inputs, summary statistics, and percentile-based distributions while
respecting optional bounds for loads and preserving signed BMP effects.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

def _trunc_normal(
    self: "Model",
    mean: float,
    sd: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
    size: Optional[int] = None,
) -> np.ndarray:
    """Draw truncated normal samples.
    Parameters
    ----------
    self : Model
        Active simulation model instance providing the RNG.
    mean : float
        Mean of the normal distribution.
    sd : float
        Standard deviation of the normal distribution.
    low : float or None, optional
        Minimum allowed value. Default is ``None``.
    high : float or None, optional
        Maximum allowed value. Default is ``None``.
    size : int or None, optional
        Number of values to sample. Default is ``None``.
    Returns
    -------
    numpy.ndarray
        Array of sampled values clipped to the requested bounds.
    """
    n = int(size or 1)
    if sd <= 0:
        val = mean
        if low is not None:
            val = max(low, val)
        if high is not None:
            val = min(high, val)
        return np.full(n, float(val))

    out = np.empty(n, dtype=float)
    filled = 0
    batch = max(4, n)
    max_tries = 20
    tries = 0
    while filled < n and tries < max_tries:
        x = self.rng.normal(mean, sd, size=batch)
        if low is not None:
            x = x[x >= low]
        if high is not None:
            x = x[x <= high]
        k = min(len(x), n - filled)
        if k > 0:
            out[filled : filled + k] = x[:k]
            filled += k
        tries += 1
        batch = min(max(batch * 2, n - filled), (n - filled) * 8 + 1024)
    if filled < n:
        fallback = mean
        if low is not None:
            fallback = max(low, fallback)
        if high is not None:
            fallback = min(high, fallback)
        out[filled:] = float(fallback)

    return out


def _piecewise_quantile_sample(
    self: "Model",
    stats: Dict[str, float],
    size: int = 1,
) -> np.ndarray:
    """Sample values from piecewise percentile statistics.
    Parameters
    ----------
    self : Model
        Active simulation model instance providing the RNG.
    stats : dict[str, float]
        Mapping containing percentile statistics such as ``min``, ``p50``,
        and ``max``.
    size : int, optional
        Number of values to sample. Default is ``1``.

    Returns
    -------
    numpy.ndarray
        Sampled values interpolated between the supplied percentile points.
    Raises
    ------
    ValueError
        If either a minimum or maximum statistic is missing.
    """
    cols = {str(k).lower(): v for k, v in stats.items()}

    pts = []
    if any(k in cols for k in ("min", "minimum", "p0")):
        qmin = float(cols.get("min", cols.get("minimum", cols.get("p0"))))
        pts.append((0.0, qmin))
    else:
        raise ValueError("Piecewise sampler requires min")
    percs = {}
    for k, v in list(cols.items()):
        if k.startswith("p") and k[1:].isdigit():
            percs[int(k[1:])] = float(v)
    for p in sorted(percs.keys()):
        if 0 < p < 100:
            pts.append((p / 100.0, percs[p]))

    if any(k in cols for k in ("max", "maximum", "p100")):
        qmax = float(cols.get("max", cols.get("maximum", cols.get("p100"))))
        pts.append((1.0, qmax))
    else:
        raise ValueError("Piecewise sampler requires max")
    pts = sorted(pts, key=lambda t: t[0])

    u = self.rng.uniform(0.0, 1.0, size=size)
    samples = np.empty(size, dtype=float)
    for i, ui in enumerate(u):
        for (p0, q0), (p1, q1) in zip(pts[:-1], pts[1:]):
            if p0 <= ui <= p1:
                if p1 == p0:
                    samples[i] = q0
                else:
                    t = (ui - p0) / (p1 - p0)
                    samples[i] = q0 + t * (q1 - q0)
                break
    return samples

def _sample_from_stats(
    self: "Model",
    stats: Dict[str, float],
    kind: Optional[str] = None,
) -> float:
    """Sample one value from summary statistics.

    The sampler chooses an appropriate strategy based on the available
    statistics. Fixed values are returned directly. Mean/standard-deviation
    rows are sampled with a truncated normal distribution, honoring any row
    ``min``/``max`` bounds; min/max rows are sampled uniformly; and percentile
    rows are sampled by piecewise linear interpolation.
    Parameters
    ----------
    self : Model
        Active simulation model instance providing the RNG and sampling
        helpers.
    stats : dict[str, float]
        Summary statistics for one sampled value.
    kind : str or None, optional
        Optional semantic hint. Use ``"efficiency"`` for a signed BMP effect
        capped at ``1`` or ``"yield"`` to clamp the result at zero. Negative
        efficiencies are preserved because they represent load increases.
        Default is ``None``.
    Returns
    -------
    float
        Sampled numeric value.

    Raises
    ------
    ValueError
        If the supplied statistics are insufficient to determine a sample.
    """
    cols = {str(k).lower(): v for k, v in stats.items()}
    has_min = any(k in cols for k in ("min", "minimum", "p0"))
    has_max = any(k in cols for k in ("max", "maximum", "p100"))
    has_sd = any(k in cols for k in ("sd", "std"))
    has_mean = any(k in cols for k in ("mean", "average", "avg"))
    has_percentiles = any(str(k).startswith("p") and str(k)[1:].isdigit() for k in cols.keys())

    low, high = None, None
    if kind == "efficiency":
        high = 1.0
    elif kind == "yield":
        low = 0.0
    if "value" in cols:
        s = float(cols["value"])
    elif has_min and has_max and has_percentiles:
        s = float(self._piecewise_quantile_sample(cols, size=1)[0])
    elif has_min and has_max and has_mean and not has_sd:
        mn = float(cols.get("mean", cols.get("average", cols.get("avg"))))
        lo = float(cols.get("min", cols.get("minimum", cols.get("p0"))))
        hi = float(cols.get("max", cols.get("maximum", cols.get("p100"))))
        sd = max((hi - lo) / 4.0, 1e-12)
        s = float(self._trunc_normal(mn, sd, low=lo if low is None else max(low, lo), high=hi if high is None else min(high, hi), size=1)[0])
    elif has_min and has_max and not has_mean and not has_sd and not has_percentiles:
        lo = float(cols.get("min", cols.get("minimum", cols.get("p0"))))
        hi = float(cols.get("max", cols.get("maximum", cols.get("p100"))))
        lo = max(lo, low) if low is not None else lo
        hi = min(hi, high) if high is not None else hi
        s = float(self.rng.uniform(lo, hi))
    elif has_mean and has_sd:
        mn = float(cols.get("mean", cols.get("average", cols.get("avg"))))
        sd = float(cols.get("sd", cols.get("std")))
        if has_min:
            row_low = float(cols.get("min", cols.get("minimum", cols.get("p0"))))
            low = row_low if low is None else max(low, row_low)
        if has_max:
            row_high = float(cols.get("max", cols.get("maximum", cols.get("p100"))))
            high = row_high if high is None else min(high, row_high)
        s = float(self._trunc_normal(mn, sd, low=low, high=high, size=1)[0])
    else:
        raise ValueError("Insufficient distribution statistics to sample")
    if low is not None and s < low:
        s = low
    if high is not None and s > high:
        s = high
    return float(s)
