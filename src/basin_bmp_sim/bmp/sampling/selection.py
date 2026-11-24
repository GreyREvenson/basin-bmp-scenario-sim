"""BMP selection and load-reduction helpers.

This module contains the logic used to choose best management practices
(BMPs), sample their effectiveness, and apply their impacts to parcel-level
pollutant loads. The functions are implemented as model helpers and operate
on shared model state such as parcel geometry, pollutant load_rates, and
simulation outputs.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from basin_bmp_sim.execution.monte_carlo_driver import Model

from basin_bmp_sim.domain.constants import (
    FT_TO_M,
    CFG_BUFFER_DEPTH_FT,
    CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
    BMP_CPS_NAME_MAP,
    COL_CPS,
    COL_PROBABILITY,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_PORTION_TREATED,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
    OUTPUT_WETLAND_AREA,
    DATA_BMP_COST,
    DATA_CPS,
    PATHWAY_VALUES,
)
from basin_bmp_sim.common.logging import log_scope
from basin_bmp_sim.domain.parcel_graph import iter_full_upstream_by_distance
from basin_bmp_sim.config.bmp_loading import load_bmp_selection_probabilities

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]



def _select_bmp_type(self: "Model") -> int:
    """Randomly choose the next BMP CPS code.

    The selection is made from the probability table produced by
    :func:`_get_bmp_selection_probs` and stored on the model as
    ``self.bmp_selection_probs``.

    Returns
    -------
    int
        Selected BMP CPS code.
    """
    idx = self.rng.choice(len(self.bmp_cps), p=self.bmp_selection_probs)
    cps = int(self.bmp_cps[idx])
    self.logger.verbose(f"selected bmp {cps} ({self._get_bmp_name(cps)})")
    return cps


def _get_bmp_name(self: "Model", cps: Union[int, str]) -> str:
    """Return a human-readable BMP name for a CPS code.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    cps : int or str
        BMP CPS identifier.

    Returns
    -------
    str
        Known BMP name when available, otherwise ``"CPS {code}"``.
    """
    key = int(cps)
    return BMP_CPS_NAME_MAP[key] if key in BMP_CPS_NAME_MAP else f"CPS {key}"


def _get_bmp_selection_probs(self: "Model", bmp_sel_path: Optional[str]) -> pd.DataFrame:
    """Build the BMP selection probability table.

    The function first attempts to read an explicit probability CSV. If no
    file is provided, it may estimate probabilities from cost heuristics when
    enabled in the configuration. Otherwise, all BMP types receive equal
    probability.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    bmp_sel_path : str or None
        Optional path to a CSV file containing BMP selection probabilities.

    Returns
    -------
    pandas.DataFrame
        Two-column dataframe with ``COL_CPS`` and ``COL_PROBABILITY``.

    Raises
    ------
    ValueError
        If the probability file is malformed, incomplete, or contains invalid
        probability values.
    """
    if bmp_sel_path:
        return load_bmp_selection_probabilities(
            bmp_sel_path,
            self.data[DATA_CPS],
            self.logger,
        )
    else:
        est_via_costs = self.cfg["bmp_sel_prob_via_costs"]
        if est_via_costs and self.data[DATA_BMP_COST] is not None:
            self.logger.info("estimating BMP selection probabilities via cost heuristics")
            df = self._estimate_costs_for_probabilities()
            return df[[COL_CPS, COL_PROBABILITY]]
        else:
            probs = np.full(len(self.data[DATA_CPS]), 1.0 / len(self.data[DATA_CPS]))
            return pd.DataFrame({COL_CPS: self.data[DATA_CPS], COL_PROBABILITY: probs})
