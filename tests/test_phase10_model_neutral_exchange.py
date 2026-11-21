"""Phase 10: alternative annual backend with no PLET/RUSLE native context."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from src.driver.bmi_coupling import (
    AnnualBmiScenarioCoupler, get_bmi_value, transfer_bmi_value,
    validate_annual_model_bmi,
)
from src.driver.monte_carlo_driver import _ScenarioContext
from src.interfaces.annual_model_contract import (
    AnnualBMPPlacement, AnnualLoadSnapshot, AnnualModelDomain,
)
from src.interfaces.bmi import ExternalWatershedModelBmi, LiveScenarioGeneratorBmi
from src.interfaces.placement_exchange import PLACEMENT_FIELDS
from test_phase2_scenario_generation import _case
from test_phase9_independent_bmi_initialization import _mocked_config_inputs


def _domain(ctx):
    upstream = [0]
    items = []
    for group in ctx.parcel_up_idxs:
        items.extend(group)
        upstream.append(len(items))
    return AnnualModelDomain(
        tuple(map(str, ctx.parcel_ids)), tuple(map(str, ctx.outlet_oids)),
        tuple(map(str, ctx.pollutants)), tuple(map(str, ctx.pathway_names)),
        tuple(map(int, ctx.bmp_cps)), ctx.parcel_area_ha, ctx.parcel_perim_m,
        upstream, items, ctx.selection_source_idxs, ctx.parcel_selection_probs,
    )


class IndependentToyBackend:
    """Deliberately non-PLET scientific law, with an independently owned state."""

    def __init__(self, domain):
        self.domain = domain
        self.baseline = None
        self.current = None
        self.outlet = None
        self.pending = None
        self.received = []
        self.finished = False

    def initialize(self, rng):
        n, p, o = len(self.domain.parcel_ids), len(self.domain.pollutants), len(self.domain.outlet_oids)
        # Consume a baseline RNG draw, demonstrating explicit stream handoff.
        self.baseline = np.full((n, p), 5.0 + rng.random())
        self.current = self.baseline.copy()
        self.outlet = np.zeros((p, o))

    def snapshot(self):
        return AnnualLoadSnapshot(self.baseline, self.current, self.outlet)

    def apply_bmp(self, placement, scenario_id, rng):
        assert isinstance(placement, AnnualBMPPlacement)
        assert scenario_id == 1
        assert self.pending is None
        self.pending = placement
        self.received.append(placement)
        # A deliberately simplistic response distinct from PLET/RUSLE.
        delta = rng.random() * 0.01
        for parcel_idx, fraction in zip(placement.affected_indices, placement.affected_fractions):
            self.current[parcel_idx, :] -= delta * fraction
            self.outlet[:, 0] += delta * fraction
        return ()

    def complete_bmp(self, cost_usd):
        assert self.pending is not None
        record = {"cps": self.pending.cps, "cost_usd": cost_usd,
                  "host_index": self.pending.host_parcel_index}
        self.pending = None
        return record

    def finalize(self):
        self.finished = True


def test_domain_and_snapshot_are_typed_validated_and_defensively_copied(tmp_path):
    shared, cfg, seed = _case('mixed_budget', tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger('p10.domain'), seed)
    dom = _domain(ctx)
    original = dom.parcel_area_ha.copy()
    ctx.parcel_area_ha[:] = 300.0
    np.testing.assert_array_equal(dom.parcel_area_ha, original)
    assert not dom.parcel_area_ha.flags.writeable
    snapshot = AnnualLoadSnapshot(np.ones((1, 3)), np.ones((1, 3)), np.array([[-2.], [0.], [1.]]))
    actual = snapshot.validated(dom)
    assert actual.outlet_removed_kg[0, 0] == -2.0  # signed outlet increases are legal
    assert not actual.current_rates.flags.writeable
    with pytest.raises(ValueError, match='shape'):
        AnnualLoadSnapshot(np.ones((3, 1)), np.ones((1, 3)), np.zeros((3, 1))).validated(dom)
    with pytest.raises(ValueError, match='finite'):
        AnnualLoadSnapshot(np.ones((1, 3)), np.ones((1, 3)), np.array([[np.nan], [0], [0]])).validated(dom)
    with pytest.raises(ValueError, match='probabilities'):
        AnnualModelDomain(dom.parcel_ids, dom.outlet_oids, dom.pollutants, dom.pathway_names,
                          dom.bmp_cps, dom.parcel_area_ha, dom.parcel_perim_m,
                          dom.upstream_offset, dom.upstream_index, dom.selection_source_idxs, [0.25])


def test_external_model_couples_to_unmodified_generator_without_native_context(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, 'mixed_budget')
    ctx = _ScenarioContext(cfg, shared, logging.getLogger('p10.toy'), seed)
    domain = _domain(ctx)
    backend = IndependentToyBackend(domain)
    external = ExternalWatershedModelBmi(domain, backend, np.random.default_rng(seed))
    generator = LiveScenarioGeneratorBmi(require_domain_import=True)
    coupler = AnnualBmiScenarioCoupler.from_components(
        generator, external, generator_config_file=str(manifest), scenario_id=1, seed=seed,
    )
    assert coupler.ctx is None
    assert not hasattr(external, '_ctx')  # No prepared PLET/RUSLE context is needed.
    coupler.initialize()
    try:
        validate_annual_model_bmi(external)
        rows = []
        while (row := coupler.next_bmp()) is not None:
            rows.append(row)
        assert rows
        assert len(rows) == len(backend.received)
        assert all(row.cost_usd == row.bmp_record['cost_usd'] for row in rows)
        assert all(row.delivered_reductions.shape == (len(domain.pollutants), len(domain.outlet_oids))
                   for row in rows)
        assert coupler.completed_scenario().placements
        assert len(coupler.completed_scenario().placements) == len(rows)
        assert rows[-1].delivered_reductions[0, 0] > 0
    finally:
        coupler.finalize()
    assert backend.finished


def test_invalid_external_request_rejected_before_backend_call(tmp_path):
    shared, cfg, seed = _case('mixed_budget', tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger('p10.invalid'), seed)
    domain = _domain(ctx)
    backend = IndependentToyBackend(domain)
    external = ExternalWatershedModelBmi(domain, backend, np.random.default_rng(seed))
    external.initialize()
    try:
        for name in PLACEMENT_FIELDS:
            dtype = external.get_var_type(name)
            if name == 'placement__efficiency':
                values = np.zeros(len(domain.pollutants) * len(domain.pathway_names), dtype=dtype)
            elif name == 'placement__affected_parcel_index':
                values = np.array([0], dtype=dtype)
            elif name == 'placement__affected_parcel_fraction':
                values = np.array([1.0], dtype=dtype)
            elif name == 'placement__cps_code':
                values = np.array([domain.bmp_cps[0]], dtype=dtype)
            elif name in ('placement__recovery_effect_scale',):
                values = np.array([1.0], dtype=dtype)
            elif name == 'placement__recorded_recovery_gains':
                values = np.array([], dtype=dtype)
            else:
                values = np.array([0], dtype=dtype)
            external.set_value(name, values)
        external.set_value('placement__affected_parcel_fraction', np.array([1.4]))
        with pytest.raises(ValueError, match='fractions'):
            external.update()
        assert not backend.received
        external.set_value('placement__affected_parcel_fraction', np.array([1.0]))
        external.update()
        assert len(backend.received) == 1
        with pytest.raises(RuntimeError, match='placement__cost'):
            external.complete_pending()
        external.set_value('placement__cost', np.array([1.75]))
        record = external.complete_pending()
        assert record['cost_usd'] == 1.75
    finally:
        external.finalize()


def test_preflight_rejects_altered_external_units(tmp_path, monkeypatch):
    shared, cfg, seed = _case('mixed_budget', tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger('p10.preflight'), seed)
    dom = _domain(ctx)
    model = ExternalWatershedModelBmi(dom, IndependentToyBackend(dom), np.random.default_rng(seed))
    model.initialize()
    try:
        original = model.get_var_units
        monkeypatch.setattr(model, 'get_var_units', lambda n: 'tonnes' if n == 'model__outlet_removed_mass' else original(n))
        with pytest.raises(ValueError, match='unit mismatch'):
            validate_annual_model_bmi(model)
    finally:
        model.finalize()


def test_external_backend_fails_preflight_if_pollutant_identity_differs(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, 'mixed_budget')
    ctx = _ScenarioContext(cfg, shared, logging.getLogger('p10.mismatch'), seed)
    base = _domain(ctx)
    incompatible = AnnualModelDomain(
        base.parcel_ids, base.outlet_oids, tuple(reversed(base.pollutants)),
        base.pathway_names, base.bmp_cps, base.parcel_area_ha, base.parcel_perim_m,
        base.upstream_offset, base.upstream_index, base.selection_source_idxs,
        base.parcel_selection_probs,
    )
    backend = IndependentToyBackend(incompatible)
    model = ExternalWatershedModelBmi(incompatible, backend, np.random.default_rng(seed))
    generator = LiveScenarioGeneratorBmi(require_domain_import=True)
    coupled = AnnualBmiScenarioCoupler.from_components(generator, model,
        generator_config_file=str(manifest), seed=seed)
    with pytest.raises(ValueError, match='index-domain mismatch'):
        coupled.initialize()
    assert not backend.received
    generator.finalize()
    model.finalize()


def test_signed_efficiencies_are_preserved_in_neutral_placement_request(tmp_path):
    shared, cfg, seed = _case('mixed_budget', tmp_path)
    ctx = _ScenarioContext(cfg, shared, logging.getLogger('p10.signed'), seed)
    dom = _domain(ctx)
    backend = IndependentToyBackend(dom)
    model = ExternalWatershedModelBmi(dom, backend, np.random.default_rng(seed))
    model.initialize()
    try:
        for name in PLACEMENT_FIELDS:
            dtype = model.get_var_type(name)
            if name == 'placement__efficiency':
                vals = np.array([-0.3] * (len(dom.pollutants) * len(dom.pathway_names)), dtype=dtype)
            elif name == 'placement__affected_parcel_index':
                vals = np.array([0], dtype=dtype)
            elif name == 'placement__affected_parcel_fraction':
                vals = np.array([1.0], dtype=dtype)
            elif name == 'placement__cps_code':
                vals = np.array([dom.bmp_cps[0]], dtype=dtype)
            elif name == 'placement__recovery_effect_scale':
                vals = np.array([1.0], dtype=dtype)
            elif name == 'placement__recorded_recovery_gains':
                vals = np.array([], dtype=dtype)
            else:
                vals = np.array([0], dtype=dtype)
            model.set_value(name, vals)
        model.update()
        np.testing.assert_array_equal(backend.received[0].efficiency,
                                      -0.3 * np.ones((len(dom.pollutants), len(dom.pathway_names))))
        model.set_value('placement__cost', np.array([0.0]))
        model.complete_pending()
    finally:
        model.finalize()


def test_upstream_cycle_is_invalid_even_with_valid_csr_shape():
    with pytest.raises(ValueError, match='cycle'):
        AnnualModelDomain(
            ('p0', 'p1'), ('out',), ('TN',), ('surface',), (340,),
            [1.0, 1.0], [100.0, 100.0], [0, 1, 2], [1, 0],
            [0, 1], [0.5, 0.5],
        )
