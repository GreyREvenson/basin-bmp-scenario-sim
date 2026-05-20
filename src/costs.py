import logging
import pandas as pd
from numpy.random import Generator

from .sampling import sample_from_stats

from typing import Optional, Union

def compute_bmp_cost_usd(
    rng: Generator,
    bmp_type_cps: Union[int, str],
    unit_row: Optional[pd.Series],
    quantity: float,
    logger: logging.Logger,
) -> float:
    """Compute total BMP cost in USD from sampled unit cost and quantity.

    The function samples a cost rate based on the provided statistics, validates
    that the sampled rate and resulting total are non-negative, and returns the
    USD cost for the given BMP quantity.
    """
    if unit_row is None:
        return 0.0
    stats = {k: unit_row[k] for k in unit_row.index if k in ("mean","sd","min","max") or (str(k).startswith("p") and str(k)[1:].isdigit())}
    rate = sample_from_stats(rng, stats, kind=None, verbose_logger=logger, ctx=f"cps={bmp_type_cps}")
    if rate < 0:
        raise ValueError("Negative cost-rate sampled")
    total = rate * quantity
    if total < 0:
        raise ValueError("Negative total cost computed")
    return float(total)
