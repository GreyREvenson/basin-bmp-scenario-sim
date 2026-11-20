"""Explicit helper interface for model/scenario state.

Normal Python descriptors replace per-instance ``types.MethodType`` binding.
Scientific algorithms still accept a context with named data attributes;
no scientific helper is attached dynamically during scenario construction.
"""
from __future__ import annotations
from typing import Any, Dict, Mapping
from numpy.random import default_rng
from src.bmp import (
    _apply_pathway_reduction, _get_bmp_name, _get_current_total_load_rate,
    _get_pathway_load_rates, _get_bmp_selection_probs, _sample_efficiency_map,
    _select_bmp_type, _simulate_grassed, _simulate_infield, _simulate_wetland,
)
from src.cost import _estimate_costs_for_probabilities, _get_bmp_cost, _select_cost_rate_median
from src.parcel import (
    _get_delivery_coeffs, _get_parcel_metadata, _get_parcel_out_oids,
    _get_parcel_up_list, _sample_parcel_index, _sample_load_rate,
)
from src.sampling import _piecewise_quantile_sample, _sample_from_stats, _trunc_normal


class ModelContextServices:
    """Statically declared helper methods used by the legacy scientific kernels."""
    _sample_from_stats = _sample_from_stats
    _piecewise_quantile_sample = _piecewise_quantile_sample
    _trunc_normal = _trunc_normal
    _select_bmp_type = _select_bmp_type
    _get_bmp_name = _get_bmp_name
    _get_pathway_load_rates = _get_pathway_load_rates
    _get_current_total_load_rate = _get_current_total_load_rate
    _apply_pathway_reduction = _apply_pathway_reduction
    _sample_efficiency_map = _sample_efficiency_map
    _simulate_wetland = _simulate_wetland
    _simulate_grassed = _simulate_grassed
    _simulate_infield = _simulate_infield
    _get_bmp_selection_probs = _get_bmp_selection_probs
    _get_bmp_cost = _get_bmp_cost
    _sample_parcel_index = _sample_parcel_index
    _sample_load_rate = _sample_load_rate
    _get_parcel_metadata = _get_parcel_metadata
    _get_parcel_up_list = _get_parcel_up_list
    _get_parcel_out_oids = _get_parcel_out_oids
    _delivery_coeffs = _get_delivery_coeffs
    _estimate_costs_for_probabilities = _estimate_costs_for_probabilities
    _select_cost_rate_median = _select_cost_rate_median


class ScenarioModelContext(ModelContextServices):
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
