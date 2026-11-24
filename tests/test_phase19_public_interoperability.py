"""Phase 19: importable independent backend and black-box BMI acceptance.

The fixture may use native helpers to create a generator-matching domain
JSON; the reference scientific factory and BMI acceptance tool do not.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from reference_backend.annual_linear import LinearAnnualBackend, make_backend
from basin_bmp_sim.execution.backend_selection import run_backend_selection
from basin_bmp_sim.interfaces.annual_model_contract import AnnualBMPPlacement
from basin_bmp_sim.certification.acceptance import accept_backend
from basin_bmp_sim.execution.monte_carlo_driver import _ScenarioContext
from test_phase10_model_neutral_exchange import _domain
from test_phase9_independent_bmi_initialization import _mocked_config_inputs


def _fixture(monkeypatch, tmp_path):
    manifest, shared, cfg, seed = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    domain = _domain(_ScenarioContext(cfg, shared, logging.getLogger("p19.fixture"), seed))
    n, p, o, k = map(len, (domain.parcel_ids, domain.pollutants,
                            domain.outlet_oids, domain.pathway_names))
    rows = {
        "schema_version": 1,
        "parcel_ids": list(domain.parcel_ids),
        "outlet_oids": list(domain.outlet_oids),
        "pollutants": list(domain.pollutants),
        "pathway_names": list(domain.pathway_names),
        "bmp_cps": list(domain.bmp_cps),
        "parcel_area_ha": domain.parcel_area_ha.tolist(),
        "parcel_perim_m": domain.parcel_perim_m.tolist(),
        "upstream_offset": domain.upstream_offset.tolist(),
        "upstream_index": domain.upstream_index.tolist(),
        "selection_source_idxs": domain.selection_source_idxs.tolist(),
        "parcel_selection_probs": domain.parcel_selection_probs.tolist(),
        "baseline_rates_kg_ha_yr": [[10.0 + i] * p for i in range(n)],
        "pathway_weights": [[1.0 / k] * k for _ in range(p)],
        "outlet_delivery_fraction": [[0.5 / o] * o for _ in range(n)],
    }
    domain_file = tmp_path / "reference_domain.json"
    domain_file.write_text(json.dumps(rows), encoding="utf-8")
    selection = {
        "schema_version": 1, "backend": "external",
        "component_manifest": manifest.name,
        "external_factory": "reference_backend.annual_linear:make_backend",
        "backend_options": {"domain_file": domain_file.name},
        "required_features": ["multi_parcel_treatment", "signed_efficiency"],
    }
    selected_file = tmp_path / "external_selection.json"
    selected_file.write_text(json.dumps(selection), encoding="utf-8")
    return manifest, selected_file, domain_file, rows


def test_importable_reference_backend_complete_cli_and_mass_balance(monkeypatch, tmp_path):
    _, selection, _, rows = _fixture(monkeypatch, tmp_path)
    first = tmp_path / "reference_1.json"
    second = tmp_path / "reference_2.json"
    run_backend_selection(selection, first)
    run_backend_selection(selection, second)
    assert first.read_bytes() == second.read_bytes()
    result = json.loads(first.read_text())
    assert result["backend"] == "external"
    assert result["bmp_responses"]
    assert len(result["bmp_responses"]) == len(result["replay_scenario"]["placements"])
    p = len(rows["pollutants"])
    n = len(rows["parcel_ids"])
    o = len(rows["outlet_oids"])
    baseline = np.asarray(result["baseline_load_rate_kg_ha_yr"]).reshape(n, p)
    current = np.asarray(result["current_load_rate_kg_ha_yr"]).reshape(n, p)
    outlet = np.asarray(result["outlet_removed_mass_kg"]).reshape(p, o)
    expected = np.einsum("np,n,no->po", baseline - current,
                         np.asarray(rows["parcel_area_ha"]),
                         np.asarray(rows["outlet_delivery_fraction"]))
    np.testing.assert_allclose(outlet, expected, atol=1e-10, rtol=0)
    assert all(response["bmp_record"]["cost_usd"] == response["cost_usd"]
               for response in result["bmp_responses"])


def test_public_bmi_acceptance_uses_only_advertised_methods(monkeypatch, tmp_path):
    _, selection, _, _ = _fixture(monkeypatch, tmp_path)
    report = accept_backend(selection)
    assert report["backend"] == "external"
    assert report["bmp_count"] > 0
    assert report["deterministic_repeat"] is True
    assert len(report["result_sha256"]) == 64
    assert any(v["name"] == "model__outlet_removed_mass"
               for v in report["model"]["variables"])
    assert report == accept_backend(selection)


def test_reference_external_ensemble_is_repeatable_and_checkpointable(monkeypatch, tmp_path):
    _, selection, _, _ = _fixture(monkeypatch, tmp_path)
    payload = json.loads(selection.read_text())
    payload["schema_version"] = 2
    payload["ensemble"] = {"n_scenarios": 2}
    selection.write_text(json.dumps(payload))
    first = tmp_path / "ensemble.json"
    run_backend_selection(selection, first)
    assert len(json.loads(first.read_text())["scenarios"]) == 2
    payload["schema_version"] = 3
    payload["ensemble"] = {
        "n_scenarios": 2, "n_jobs": 1, "checkpoint_dir": "reference_ckpt", "resume": False,
    }
    selection.write_text(json.dumps(payload))
    checkpointed = tmp_path / "checkpointed.json"
    run_backend_selection(selection, checkpointed)
    assert first.read_bytes() == checkpointed.read_bytes()
    payload["ensemble"]["resume"] = True
    selection.write_text(json.dumps(payload))
    run_backend_selection(selection, checkpointed)
    assert first.read_bytes() == checkpointed.read_bytes()


def test_file_schema_or_domain_mismatch_fails_before_replacing_results(monkeypatch, tmp_path):
    _, selection, domain_file, rows = _fixture(monkeypatch, tmp_path)
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"previous valid output")
    rows["outlet_delivery_fraction"][0] = [1.0] * len(rows["outlet_oids"])
    if len(rows["outlet_oids"]) == 1:
        rows["outlet_delivery_fraction"][0] = [-0.1]
    domain_file.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="outlet_delivery_fraction"):
        run_backend_selection(selection, existing)
    assert existing.read_bytes() == b"previous valid output"
    assert not list(tmp_path.glob("*.tmp"))
    rows["outlet_delivery_fraction"][0] = [0.5 / len(rows["outlet_oids"])] * len(rows["outlet_oids"])
    rows["pollutants"].reverse()
    domain_file.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="index-domain|domain|identit"):
        run_backend_selection(selection, existing)
    assert existing.read_bytes() == b"previous valid output"


def test_signed_reference_annual_response_is_not_clipped(monkeypatch, tmp_path):
    manifest, _, domain_file, rows = _fixture(monkeypatch, tmp_path)
    from basin_bmp_sim.interfaces.component_bootstrap import ComponentStartup
    bundle = make_backend(startup=ComponentStartup.read(manifest),
                          options={"domain_file": str(domain_file)})
    backend = bundle.backend
    backend.initialize(np.random.default_rng(12))
    n_pollutants = len(rows["pollutants"])
    n_pathways = len(rows["pathway_names"])
    placement = AnnualBMPPlacement(
        cps=rows["bmp_cps"][0], host_parcel_index=0,
        affected_indices=(0,), affected_fractions=(1.0,),
        efficiency=np.full((n_pollutants, n_pathways), -0.2),
        failed=False, recovery_effect_scale=1.0, cost_quantity_ha=0,
        wetland_area_ha=0, catchment_area_ratio=0,
        buffer_length_m=0, buffer_area_ha=0, portion_treated=0,
        recorded_recovery_gains=(),
    )
    assert backend.apply_bmp(placement, 1, np.random.default_rng(12)) == ()
    backend.complete_bmp(0)
    snap = backend.snapshot()
    assert (snap.current_rates[0] > snap.baseline_rates[0]).all()
    assert (snap.outlet_removed_kg < 0).all()
    backend.finalize()


def test_acceptance_cli_writes_an_atomic_machine_readable_report(monkeypatch, tmp_path):
    import sys
    from basin_bmp_sim.certification.acceptance import main
    _, selection, _, _ = _fixture(monkeypatch, tmp_path)
    dest = tmp_path / "reports" / "acceptance.json"
    monkeypatch.setattr(sys, "argv", ["annual_bmi_acceptance", str(selection), str(dest)])
    main()
    saved = json.loads(dest.read_text())
    assert saved == accept_backend(selection)
    assert not (dest.parent / "acceptance.json.tmp").exists()


def test_reference_backend_does_not_claim_unimplemented_afo_capability(monkeypatch, tmp_path):
    _, selection, _, _ = _fixture(monkeypatch, tmp_path)
    data = json.loads(selection.read_text())
    data["required_features"].append("afo_recovery")
    selection.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing required scientific capabilities"):
        run_backend_selection(selection, tmp_path / "not_written.json")
    assert not (tmp_path / "not_written.json").exists()
