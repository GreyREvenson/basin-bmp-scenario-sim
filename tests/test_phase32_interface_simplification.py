"""The native and external placement paths share one numerical contract."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from basin_bmp_sim.interfaces.annual_model_contract import AnnualBMPPlacement, PLACEMENT_FIELDS
from basin_bmp_sim.interfaces.bmi.array_component import BmiArrayComponent
from basin_bmp_sim.interfaces.bmi.external_watershed_model_bmi import ExternalWatershedModelBmi
from basin_bmp_sim.interfaces.bmi.live_scenario_generator_bmi import LiveScenarioGeneratorBmi
from basin_bmp_sim.interfaces.bmi.watershed_model_bmi import WatershedModelBmi
from basin_bmp_sim.interfaces.exchange_variables import PLACEMENT_OUTPUT_VARIABLES
from basin_bmp_sim.interfaces.placement_exchange import placement_arrays, placement_from_arrays
from basin_bmp_sim.scenarios.bmp_designs import BMPDesign
from basin_bmp_sim.scenarios.generator import PlacementChoice, RealizedBMPPlacement


def _packet():
    ctx = SimpleNamespace(parcel_ids=['P1', 'P2'], bmp_cps=[340],
                          pollutants=['TN'], pathway_names=['surface'])
    placement = RealizedBMPPlacement(
        choice=PlacementChoice(0, 'P1', 340),
        design=BMPDesign('infield', (), ((0, 1.0),), 1.0),
        efficiency_by_pollutant=((('surface', -0.2),),),
        failed=False, recovery_effect_scale=1.0,
    )
    return ctx, placement, placement_arrays(placement, ctx)


def test_encoder_uses_declared_dtype_and_vector_rank():
    ctx, original, packet = _packet()
    assert tuple(packet) == PLACEMENT_FIELDS
    for spec in PLACEMENT_OUTPUT_VARIABLES:
        assert packet[spec.name].dtype == np.dtype(spec.dtype)
        assert packet[spec.name].ndim == 1
    assert placement_from_arrays(packet, ctx) == original
    neutral = AnnualBMPPlacement.from_bmi(packet, ctx)
    assert neutral.efficiency[0, 0] == -0.2
    assert not neutral.efficiency.flags.writeable


@pytest.mark.parametrize(('field', 'invalid'), [
    ('placement__host_parcel_index', np.array([0.0])),
    ('placement__affected_parcel_fraction', np.array([1.1])),
    ('placement__efficiency', np.array([1.01])),
    ('placement__recorded_recovery_gains', np.array([np.nan])),
    ('placement__failed', np.array([2], dtype='int64')),
    ('placement__wetland_area_ha', np.array([-1.0])),
])
def test_both_paths_reject_invalid_packet(field, invalid):
    ctx, _, packet = _packet()
    packet[field] = invalid
    for decode in (AnnualBMPPlacement.from_bmi, placement_from_arrays):
        with pytest.raises(ValueError):
            decode(packet, ctx)


def test_native_family_check_is_not_imposed_on_external_backend():
    ctx, _, packet = _packet()
    packet['placement__affected_parcel_fraction'] = np.array([0.5])
    assert AnnualBMPPlacement.from_bmi(packet, ctx).affected_fractions == (0.5,)
    with pytest.raises(ValueError, match='in-field'):
        placement_from_arrays(packet, ctx)


@pytest.mark.parametrize('component', [BmiArrayComponent, WatershedModelBmi,
                                      ExternalWatershedModelBmi, LiveScenarioGeneratorBmi])
def test_bmi_source_keyword(component):
    assert list(inspect.signature(component.set_value).parameters) == ['self', 'name', 'src']
    assert list(inspect.signature(component.set_value_at_indices).parameters) == ['self', 'name', 'inds', 'src']


def test_generic_statistics_validation_has_one_nonconfiguration_owner():
    from pathlib import Path
    package = Path(__file__).resolve().parents[1] / 'src' / 'basin_bmp_sim'
    assert (package / 'domain/statistics_validation.py').is_file()
    assert not (package / 'config/validation/statistics.py').exists()
    for folder in ('science', 'scenarios', 'bmp', 'domain'):
        for path in (package / folder).rglob('*.py'):
            assert 'basin_bmp_sim.config.validation.statistics' not in path.read_text(encoding='utf-8')
