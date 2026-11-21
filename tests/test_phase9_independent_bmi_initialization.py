"""Phase 9: separate source-file initialization without changing BMI coupling."""
from __future__ import annotations

from copy import deepcopy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src.driver.bmi_coupling import AnnualBmiScenarioCoupler, validate_bmi_index_contract
from src.driver.monte_carlo_driver import _ScenarioContext, Model
from src.interfaces.bmi.live_scenario_generator_bmi import LiveScenarioGeneratorBmi
from src.interfaces.bmi.watershed_model_bmi import WatershedModelBmi
from src.interfaces.component_bootstrap import ComponentStartup
from src.models.prepared_inputs import PreparedModelInputs
from test_phase2_scenario_generation import _case


def _manifest(tmp_path: Path, **overrides) -> Path:
    cfg_path = tmp_path / 'watershed.yaml'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text('stub: yes\n', encoding='utf-8')
    payload = {
        'schema_version': 1, 'model_config': 'watershed.yaml',
        'worker_seed': 12345, 'scenario_id': 1,
    }
    payload.update(overrides)
    path = tmp_path / 'bootstrap.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return path


@pytest.mark.parametrize('payload,expected', [
    ({'schema_version': 123}, 'schema version'),
    ({'worker_seed': True}, 'worker_seed'),
    ({'worker_seed': -2}, 'worker_seed'),
    ({'scenario_id': 0}, 'scenario_id'),
    ({'scenario_id': False}, 'scenario_id'),
    ({'model_config': ''}, 'model_config'),
    ({'extra': 1}, 'missing or unknown'),
    ({'replay_file': 'missing.json'}, 'missing.json'),
])
def test_invalid_startup_manifest_rejected_without_component_mutation(tmp_path, payload, expected):
    path = _manifest(tmp_path, **payload)
    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        ComponentStartup.read(path)
    model = WatershedModelBmi()
    with pytest.raises((ValueError, FileNotFoundError)):
        model.initialize(str(path))
    assert not model._initialized
    assert model._ctx is None


def test_startup_relative_paths_are_resolved_from_manifest_not_cwd(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path / 'inputs')
    monkeypatch.chdir(tmp_path)
    parsed = ComponentStartup.read(manifest)
    assert parsed.model_config == tmp_path / 'inputs' / 'watershed.yaml'
    assert parsed.worker_seed == 12345
    assert parsed.scenario_id == 1


def _mocked_config_inputs(monkeypatch, tmp_path, case):
    """Use real BMI kernels with already-validated small input tables.

    Patch *only* the project file-loading layer: no shipped user-config/YAML
    fixture or optional parquet dependency is required for these interface
    tests. Both component owners still construct independent model contexts.
    """
    from src.interfaces import component_bootstrap as bootstrap
    shared, cfg, seed = _case(case, tmp_path)
    manifest = _manifest(tmp_path, worker_seed=seed)
    monkeypatch.setattr(bootstrap, 'read_config', lambda _: {'fixture': case})
    monkeypatch.setattr(bootstrap, 'normalize_config', lambda _: deepcopy(cfg))
    monkeypatch.setattr(bootstrap, 'resolve_config_paths', lambda *a, **kw: None)
    monkeypatch.setattr(bootstrap, 'validate_config', lambda _: None)
    monkeypatch.setattr(bootstrap, 'load_and_validate_all', lambda *a: deepcopy(shared['data']))
    class StubPrepared:
        def __init__(self, config, data, logger):
            self.data = data
        def _shared_payload(self):
            return deepcopy(shared)
    monkeypatch.setattr(bootstrap, 'PreparedModelInputs', StubPrepared)
    return manifest, shared, cfg, seed


@pytest.mark.parametrize('case', ['mixed_budget', 'statistical_multi', 'afo_recovery'])
def test_file_initialized_coupler_matches_legacy_bmi_coupler(monkeypatch, tmp_path, case):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, case)
    expected_context = _ScenarioContext(cfg, deepcopy(shared), logging.getLogger('phase9.native'), seed)
    expected = AnnualBmiScenarioCoupler(expected_context, cfg, expected_context.logger,
                                        scenario_id=1, seed=seed)
    independent = AnnualBmiScenarioCoupler.from_manifest(manifest)
    assert independent.model._ctx is None
    assert independent.generator._ctx is None
    expected.initialize()
    independent.initialize()
    try:
        assert independent.model._ctx is not independent.generator._ctx
        assert independent.model._ctx.rng is not independent.generator._ctx.rng
        assert independent.generator._ctx.data is not independent.model._ctx.data
        validate_bmi_index_contract(independent.generator, independent.model)
        actual_rows, expected_rows = [], []
        while True:
            baseline = expected.next_bmp()
            copied = independent.next_bmp()
            assert (baseline is None) == (copied is None)
            if baseline is None:
                break
            assert copied.cps == baseline.cps
            assert copied.parcel_index == baseline.parcel_index
            assert copied.cost_usd == baseline.cost_usd
            assert copied.bmp_record == baseline.bmp_record
            np.testing.assert_array_equal(copied.delivered_reductions, baseline.delivered_reductions)
            actual_rows.append(copied)
            expected_rows.append(baseline)
        assert len(actual_rows) == len(expected_rows)
        assert independent.completed_scenario() == expected.completed_scenario()
    finally:
        expected.finalize()
        independent.finalize()
    assert independent.model._ctx is None
    assert independent.generator._ctx is None


def test_file_initialized_individual_adapters(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, 'mixed_budget')
    model, generator = WatershedModelBmi(), LiveScenarioGeneratorBmi()
    with pytest.raises(ValueError, match='startup manifest'):
        model.initialize()
    with pytest.raises(ValueError, match='startup manifest'):
        generator.initialize()
    model.initialize(str(manifest))
    generator.initialize(str(manifest))
    try:
        validate_bmi_index_contract(generator, model)
        with pytest.raises(RuntimeError, match='domain'):
            generator.update()
        assert model._ctx is not generator._ctx
        assert model._ctx.rng is not generator._ctx.rng
    finally:
        generator.finalize()
        model.finalize()


def test_native_model_inherits_shared_preparation_service():
    assert issubclass(Model, PreparedModelInputs)
    assert Model._prepare_lookup_tables is PreparedModelInputs._prepare_lookup_tables
    assert Model._shared_payload is PreparedModelInputs._shared_payload


def test_independent_file_initialized_afo_replay(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, 'afo_recovery')
    first = AnnualBmiScenarioCoupler.from_manifest(manifest)
    first.initialize()
    try:
        first_records = []
        while (row := first.next_bmp()) is not None:
            first_records.append(row)
        scenario = first.completed_scenario()
    finally:
        first.finalize()
    replay_path = tmp_path / 'scenario.json'
    scenario.save(replay_path)
    manifest_data = json.loads(manifest.read_text(encoding='utf-8'))
    manifest_data['replay_file'] = 'scenario.json'
    manifest.write_text(json.dumps(manifest_data), encoding='utf-8')

    second = AnnualBmiScenarioCoupler.from_manifest(manifest)
    second.initialize()
    try:
        second_records = []
        while (row := second.next_bmp()) is not None:
            second_records.append(row)
        assert second.completed_scenario() == scenario
        assert len(second_records) == len(first_records)
        for actual, expected in zip(second_records, first_records):
            assert actual.bmp_record == expected.bmp_record
            assert actual.cost_usd == expected.cost_usd
            np.testing.assert_array_equal(actual.delivered_reductions, expected.delivered_reductions)
    finally:
        second.finalize()
