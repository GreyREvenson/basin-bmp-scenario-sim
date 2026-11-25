"""Shared stochastic operations for prepared, generator and worker state."""
from basin_bmp_sim.scenarios.sampling import _piecewise_quantile_sample, _sample_from_stats, _trunc_normal
from basin_bmp_sim.bmp.sampling.efficiency import _sample_efficiency_map
from basin_bmp_sim.scenarios.costs import _get_bmp_cost
from basin_bmp_sim.domain.parcel import _sample_parcel_index


class SamplingContext:
    """Shared stochastic operations; fixed scientific algorithms are direct calls."""
    _sample_from_stats = _sample_from_stats
    _piecewise_quantile_sample = _piecewise_quantile_sample
    _trunc_normal = _trunc_normal
    _sample_efficiency_map = _sample_efficiency_map
    _get_bmp_cost = _get_bmp_cost
    _sample_parcel_index = _sample_parcel_index
