from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from src.bmp import (
    _apply_pathway_reduction,
    _get_current_total_load_rate,
    _get_current_total_yield,
    _get_pathway_load_rates,
    _get_pathway_yields,
    _simulate_infield,
)
from src.constants import OUTPUT_REMOVED, OUTPUT_TREATED
from src.load_generation import (
    LoadState,
    calculate_load_components,
    calculate_load_rate_components,
    calculate_plet_pathway_load_rates,
    calculate_plet_pathway_yields,
)
from src.parcel import _sample_load_rate, _sample_yield
from src.sampling import _sample_from_stats


class _Logger:
    def verbose(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


def _bmp_ctx() -> SimpleNamespace:
    ctx = SimpleNamespace(
        pathway_names=["surface", "subsurface"],
        parcel_area_ha=np.asarray([2.0]),
        pollutants=["TN"],
        current_pathway_load_rates=np.asarray([[[6.0, 4.0]]], dtype=float),
        current_untreated_groundwater_load_rates=np.zeros((1, 1), dtype=float),
        logger=_Logger(),
    )
    ctx._get_pathway_load_rates = MethodType(_get_pathway_load_rates, ctx)
    ctx._get_current_total_load_rate = MethodType(_get_current_total_load_rate, ctx)
    ctx._apply_pathway_reduction = MethodType(_apply_pathway_reduction, ctx)
    return ctx


def test_new_pathway_load_rate_helpers_and_legacy_aliases_agree() -> None:
    ctx = _bmp_ctx()
    expected = {"surface": 6.0, "subsurface": 4.0}
    assert _get_pathway_load_rates(ctx, 0, 0, 10.0) == expected
    assert _get_pathway_yields(ctx, 0, 0, 10.0) == expected
    assert _get_current_total_load_rate(ctx, 0, 0, 10.0) == pytest.approx(10.0)
    assert _get_current_total_yield(ctx, 0, 0, 10.0) == pytest.approx(10.0)


def test_infield_uses_explicit_areal_load_rate_and_mass_rate_names_semantically() -> None:
    ctx = _bmp_ctx()
    load_rates = np.asarray([[10.0]], dtype=float)  # kg/ha/yr
    outputs = {
        OUTPUT_TREATED: np.zeros(1, dtype=float),
        OUTPUT_REMOVED: np.zeros(1, dtype=float),
    }
    _simulate_infield(
        ctx,
        0,
        [{"surface": 0.5, "subsurface": 0.5}],
        load_rates,
        {},
        outputs,
    )
    # 10 kg/ha/yr * 2 ha = 20 kg/yr exposed; 50% removed = 10 kg/yr.
    assert outputs[OUTPUT_TREATED][0] == pytest.approx(20.0)
    assert outputs[OUTPUT_REMOVED][0] == pytest.approx(10.0)
    assert load_rates[0, 0] == pytest.approx(5.0)
    assert ctx.current_pathway_load_rates[0, 0].tolist() == pytest.approx([3.0, 2.0])


def test_load_state_new_names_have_deprecated_read_aliases() -> None:
    rates = np.asarray([[[1.0, 2.0]]], dtype=float)
    protected = np.zeros((1, 1), dtype=float)
    state = LoadState(
        parcel_ids=["1"],
        parameters=[{}],
        concentrations=[{}],
        groundwater_concentrations=[{}],
        has_rusle=[False],
        pollutants=["TN"],
        pathway_load_rates=rates,
        untreated_groundwater_load_rates=protected,
        baseline_pathway_load_rates=rates.copy(),
        baseline_untreated_groundwater_load_rates=protected.copy(),
    )
    assert state.pathway_yields is state.pathway_load_rates
    assert state.untreated_groundwater_yields is state.untreated_groundwater_load_rates
    assert np.array_equal(state.baseline_pathway_yields, state.baseline_pathway_load_rates)
    assert np.array_equal(
        state.baseline_untreated_groundwater_yields,
        state.baseline_untreated_groundwater_load_rates,
    )


def test_renamed_load_generation_helpers_keep_legacy_aliases() -> None:
    assert calculate_load_components is calculate_load_rate_components
    assert calculate_plet_pathway_yields is calculate_plet_pathway_load_rates


def test_parcel_load_rate_sampler_keeps_legacy_helper_alias() -> None:
    ctx = SimpleNamespace(
        pollutant_load_rate_stats=[[{"value": 12.5}]],
        parcel_ids=["p1"],
        pollutants=["TN"],
        _sample_from_stats=lambda stats, kind=None: float(stats["value"]),
    )
    assert _sample_load_rate(ctx, 0, 0) == pytest.approx(12.5)
    assert _sample_yield(ctx, 0, 0) == pytest.approx(12.5)


def test_load_rate_sampling_hint_replaces_yield_hint_but_keeps_alias() -> None:
    ctx = SimpleNamespace(
        rng=np.random.default_rng(1),
        _piecewise_quantile_sample=lambda cols, size=1: np.asarray([0.0]),
        _trunc_normal=lambda mean, sd, low=None, high=None, size=1: np.asarray([mean]),
    )
    assert _sample_from_stats(ctx, {"value": -2.0}, kind="load_rate") == pytest.approx(0.0)
    assert _sample_from_stats(ctx, {"value": -2.0}, kind="yield") == pytest.approx(0.0)


def test_negative_efficiency_returns_signed_removed_load_rate() -> None:
    ctx = _bmp_ctx()
    removed_load_rate = _apply_pathway_reduction(
        ctx, 0, 0, 0.5, {"surface": -0.4, "subsurface": 0.0}
    )
    assert removed_load_rate == pytest.approx(-1.2)
    assert ctx.current_pathway_load_rates[0, 0].tolist() == pytest.approx([7.2, 4.0])
