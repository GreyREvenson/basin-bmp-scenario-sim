"""Detached, minimal read/write context for the stochastic BMP generator.

The watershed model owns its scientific context. This generator context owns
its own RNG, parcel-domain arrays, selection state and cost/AFO configuration;
no numpy or pandas object used in live generation is shared with the model.
The Phase 8 BMI domain packet overwrites its physical-domain arrays before
sampling. Scientific load state is never copied into the generator.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from basin_bmp_sim.domain.constants import DATA_BMP_COST, DATA_AVG_AREA_HA, DATA_AVG_PERIM_M
from basin_bmp_sim.scenarios.sampling import _sample_from_stats, _piecewise_quantile_sample, _trunc_normal
from basin_bmp_sim.domain.parcel import _sample_parcel_index
from basin_bmp_sim.bmp.sampling.efficiency import _sample_efficiency_map
from basin_bmp_sim.scenarios.costs import _get_bmp_cost


class GeneratorModelContext:
    """Independent stochastic-placement state, limited to generator inputs."""

    # Only sampling/placement services; do not depend on a watershed-model
    # context or bind unused scientific-response/routing methods.
    _sample_from_stats = _sample_from_stats
    _piecewise_quantile_sample = _piecewise_quantile_sample
    _trunc_normal = _trunc_normal
    _sample_parcel_index = _sample_parcel_index
    _sample_efficiency_map = _sample_efficiency_map
    _get_bmp_cost = _get_bmp_cost

    _FIELDS = (
        "parcel_ids", "pid_to_index", "parcel_area_ha", "parcel_perim_m",
        "parcel_up_idxs", "parcel_selection_ids", "parcel_selection_probs",
        "selection_source_idxs", "bmp_cps", "bmp_selection_probs",
        "pollutants", "pathway_names", "outlet_oids", "bmp_efficiency_stats",
        "feedlot_bmp_cps", "afo_inputs", "load_generation_mode",
    )

    def __init__(self, source, seed: int):
        self.cfg = deepcopy(source.cfg)
        self.logger = source.logger
        self.rng = np.random.default_rng(seed)
        for key in self._FIELDS:
            setattr(self, key, deepcopy(getattr(source, key)))
        self.data = {key: deepcopy(source.data.get(key)) for key in
                     (DATA_BMP_COST, DATA_AVG_AREA_HA, DATA_AVG_PERIM_M)}
        # Keep any custom instance sampler overrides in test/extension contexts.
        for key in ("_sample_from_stats", "_sample_efficiency_map", "_get_bmp_cost"):
            if key in source.__dict__:
                setattr(self, key, source.__dict__[key])
