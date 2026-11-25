"""Worker scientific state and shared stochastic sampling operations."""
from __future__ import annotations
from typing import Any, Dict, Mapping
from numpy.random import default_rng
from basin_bmp_sim.scenarios.sampling_context import SamplingContext


class ScenarioModelContext(SamplingContext):
    """Isolated per-worker context, built from the validated shared payload.

    The copied mapping is deliberately explicit about required vs optional
    fields. Mutable scientific arrays are created when the model initializes,
    not shared between worker realizations.
    """
    REQUIRED = (
        "parcel_ids", "pid_to_index", "pollutants", "parcel_area_ha", "parcel_perim_m",
        "parcel_out_oids", "parcel_up_idxs", "parcel_selection_ids",
        "parcel_selection_probs", "selection_source_idxs", "outlet_oids",
        "outlet_target_map", "outlet_mean_map", "delivery_coeffs",
        "bmp_efficiency_stats", "pollutant_load_rate_stats", "load_generation_mode",
        "bmp_cps", "bmp_selection_probs",
    )

    def __init__(self, cfg: Dict[str, Any], shared: Mapping[str, Any], logger, seed: int) -> None:
        self.cfg = cfg
        self.logger = logger
        self.rng = default_rng(seed)
        absent = [name for name in self.REQUIRED if name not in shared]
        if absent:
            raise ValueError(f"Missing shared model inputs: {', '.join(absent)}")
        for name, value in shared.items():
            if name in ("rng", "logger"):
                raise ValueError(f"Shared payload cannot override local {name}")
            setattr(self, name, value)
        self.afo_inputs = getattr(self, "afo_inputs", None)
        self.feedlot_bmp_cps = set(getattr(self, "feedlot_bmp_cps", set()) or set())
        self.pid_to_index = {str(pid): int(idx) for pid, idx in self.pid_to_index.items()}
