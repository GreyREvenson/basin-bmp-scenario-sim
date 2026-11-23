"""Scientific-input fingerprints protect resumable BMI ensembles."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.driver import ensemble_checkpoints as checkpoints
from src.driver.backend_selection import BackendSelection, run_backend_selection
from src.driver.input_provenance import scientific_input_manifest
from test_phase13_checkpointed_ensembles import _v3
from test_phase9_independent_bmi_initialization import _mocked_config_inputs


def _set_auxiliary_source(manifest: Path, tmp_path: Path) -> Path:
    # The tests own their entire file domain; no repository input datasets.
    auxiliary = tmp_path / "inputs.csv"
    auxiliary.write_bytes(b"mean,value\n1,2\n")
    (tmp_path / "watershed.yaml").write_text(
        "parcels: inputs.csv\noutputs: generated-results\n", encoding="utf-8"
    )
    return auxiliary


def _add_declared_extra(selection_file: Path, filename: str) -> None:
    selected = json.loads(selection_file.read_text(encoding="utf-8"))
    selected["ensemble"]["additional_input_files"] = [filename]
    selection_file.write_text(json.dumps(selected), encoding="utf-8")


def test_input_inventory_tracks_known_paths_without_tracking_outputs(tmp_path):
    (tmp_path / "src.csv").write_text("1\n", encoding="utf-8")
    (tmp_path / "factors.csv").write_text("2\n", encoding="utf-8")
    (tmp_path / "shape.shp").write_bytes(b"geometry")
    (tmp_path / "shape.dbf").write_bytes(b"attributes")
    (tmp_path / "shape.SHX").write_bytes(b"index")
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "data.txt").write_text("fields", encoding="utf-8")
    config = tmp_path / "model.yaml"
    config.write_text(
        "parcels: shape.shp\n"
        "outlets: [src.csv, folder]\n"
        "manure_loading:\n  excretion_factors: factors.csv\n"
        "outputs: never-created-output\n", encoding="utf-8",
    )
    inventory = scientific_input_manifest(config)
    paths = {Path(item["path"]).name: item for item in inventory["files"]}
    assert inventory["schema_version"] == 1
    assert set(paths) == {"src.csv", "factors.csv", "shape.shp", "shape.dbf",
                          "shape.SHX", "data.txt"}
    assert paths["shape.dbf"]["labels"] == ["parcels"]
    assert paths["data.txt"]["labels"] == ["outlets[1]"]
    assert all(len(item["sha256"]) == 64 for item in inventory["files"])
    assert scientific_input_manifest(config) == inventory
    (tmp_path / "shape.dbf").write_bytes(b"different attributes")
    assert scientific_input_manifest(config) != inventory


def test_missing_or_empty_science_inputs_fail_before_checkpoint_creation(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    (tmp_path / "watershed.yaml").write_text("parcels: absent.csv\n", encoding="utf-8")
    selection = _v3(tmp_path, manifest, n=1)
    destination = tmp_path / "existing.json"
    destination.write_bytes(b"original result")
    with pytest.raises(FileNotFoundError, match="Scientific input"):
        run_backend_selection(selection, destination)
    assert not (tmp_path / "checkpoints").exists()
    assert destination.read_bytes() == b"original result"


def test_resume_detects_changed_scientific_input_content_not_just_mtime(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    asset = _set_auxiliary_source(manifest, tmp_path)
    selection = _v3(tmp_path, manifest, n=2)
    result = tmp_path / "ensemble.json"
    run_backend_selection(selection, result)
    original = result.read_bytes()
    identity = json.loads((tmp_path / "checkpoints" / "run.json").read_text())
    inventory = identity["run_identity"]["scientific_input_manifest"]["files"]
    assert [Path(f["path"]).name for f in inventory] == ["inputs.csv"]
    assert identity["checkpoint_schema_version"] == 2
    _v3(tmp_path, manifest, n=2, resume=True)
    stat = asset.stat()
    # Equal-length in-place replacement and restored timestamps must still be caught.
    asset.write_bytes(b"mean,value\n9,2\n")
    os.utime(asset, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="fingerprint"):
        run_backend_selection(selection, result)
    assert result.read_bytes() == original

    asset.write_bytes(b"mean,value\n1,2\n")
    monkeypatch.setattr(checkpoints, "_run_one_worker", lambda *a: pytest.fail("resume reran science"))
    run_backend_selection(selection, result)
    assert result.read_bytes() == original


def test_additional_backend_input_files_are_explicitly_fingerprinted(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    extra = tmp_path / "backend_factors.csv"
    extra.write_bytes(b"coeff\n0.75\n")
    selection = _v3(tmp_path, manifest, n=1)
    _add_declared_extra(selection, extra.name)
    parsed = BackendSelection.read(selection)
    assert parsed.ensemble.additional_input_files == (extra,)
    result = tmp_path / "ensemble.json"
    run_backend_selection(selection, result)
    assert result.is_file()
    _v3(tmp_path, manifest, n=1, resume=True)
    _add_declared_extra(selection, extra.name)
    extra.write_bytes(b"coeff\n0.76\n")
    with pytest.raises(ValueError, match="fingerprint"):
        run_backend_selection(selection, result)


@pytest.mark.parametrize("bad", ["one.csv", [""], ["a.csv", "a.csv"], [3]])
def test_rejects_bad_declared_backend_input_file_lists(tmp_path, bad):
    (tmp_path / "watershed.yaml").write_text("stub: yes\n")
    startup = tmp_path / "startup.json"
    startup.write_text(json.dumps({"schema_version": 1, "model_config": "watershed.yaml",
                                   "worker_seed": 2, "scenario_id": 1}))
    selection = _v3(tmp_path, startup, n=1)
    payload = json.loads(selection.read_text())
    payload["ensemble"]["additional_input_files"] = bad
    selection.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="additional_input_files"):
        BackendSelection.read(selection)


def test_changed_inputs_during_simulation_prevent_publication(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    auxiliary = _set_auxiliary_source(manifest, tmp_path)
    selection = _v3(tmp_path, manifest, n=1)
    result = tmp_path / "existing.json"
    result.write_text("existing result")
    original_worker = checkpoints._run_one_worker

    def mutate_after_execution(*args):
        record = original_worker(*args)
        auxiliary.write_text("mean,value\n3,4\n", encoding="utf-8")
        return record

    monkeypatch.setattr(checkpoints, "_run_one_worker", mutate_after_execution)
    with pytest.raises(ValueError, match="changed during ensemble execution"):
        run_backend_selection(selection, result)
    assert result.read_text() == "existing result"
    assert (tmp_path / "checkpoints" / "realization_00000000.json").exists()


def test_legacy_checkpoint_requires_new_directory(monkeypatch, tmp_path):
    manifest, _, _, _ = _mocked_config_inputs(monkeypatch, tmp_path, "mixed_budget")
    selection = _v3(tmp_path, manifest, n=1)
    result = tmp_path / "ensemble.json"
    run_backend_selection(selection, result)
    original = result.read_bytes()
    run_meta = tmp_path / "checkpoints" / "run.json"
    payload = json.loads(run_meta.read_text())
    payload["checkpoint_schema_version"] = 1
    run_meta.write_text(json.dumps(payload))
    _v3(tmp_path, manifest, n=1, resume=True)
    with pytest.raises(ValueError, match="predates scientific-input provenance"):
        run_backend_selection(selection, result)
    assert result.read_bytes() == original


def test_default_afo_reference_is_fingerprinted_when_mass_based(tmp_path):
    config = tmp_path / "watershed.yaml"
    (tmp_path / "afo.gpkg").write_bytes(b"sample")
    config.write_text("animal_operations: afo.gpkg\nmanure_loading:\n  method: mass_based\n")
    manifest = scientific_input_manifest(config)
    names = {Path(entry["path"]).name: entry["labels"] for entry in manifest["files"]}
    assert names["manure_excretion_reference.csv"] == ["bundled_afo_excretion_factors"]
    assert names["afo.gpkg"] == ["animal_operations"]
