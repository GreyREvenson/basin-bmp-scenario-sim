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
    from src.driver.monte_carlo_driver import Model

from src.constants import (
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
from src.logging_utils import log_scope
from src.parcel_graph import iter_full_upstream_by_distance
from src.input_config import load_bmp_selection_probabilities

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

from src.bmp.pathways.plet_pathway_efficiency import _active_pathways

def _sample_efficiency_map(self: "Model", cps: Union[int, str], pol_idx: int) -> Dict[str, float]:
    """Sample one BMP efficiency for every active pathway.

        Input validation guarantees complete coverage. In PLET/RUSLE mode this
        means surface is explicitly required and subsurface is either explicitly
        supplied or has already been inserted as a fixed zero distribution.
        Statistical mode requires explicit coverage for every active user-defined
        pathway.

        Parameters
        ----------
        cps : Union[int, str]
            Conservation Practice Standard (CPS) code or codes.
        pol_idx : int
            Zero-based index of the pollutant.

        Returns
        -------
        Dict[str, float]
            Sampled BMP efficiencies keyed by active pathway.

        Raises
        ------
        ValueError
            If BMP efficiency coverage is incomplete for the requested CPS, pollutant, or active pathway.
        
    """
    entry = self.bmp_efficiency_stats[int(cps)][pol_idx]
    pathways = _active_pathways(self)
    missing_paths = (
        list(pathways)
        if not isinstance(entry, dict)
        else [path for path in pathways if path not in entry or not isinstance(entry[path], dict)]
    )
    if missing_paths:
        raise ValueError(
            "Incomplete bmp_efficiency coverage for "
            f"cps={int(cps)}, pollutant={self.pollutants[pol_idx]}; "
            f"missing pathways: {missing_paths}"
        )
    return {
        path: float(self._sample_from_stats(entry[path], kind="efficiency"))
        for path in pathways
    }

