"""Opt-in selection of native/external annual BMI backends and executions.

Version 1 retains a single model-neutral scenario. Version 2 adds explicit
serial ensemble execution; version 3 adds checkpointed/resumable execution
with optional process-level parallelism.
The historical Monte Carlo CLI and Parquet outputs remain unchanged; no
third-party results are presented as scientifically equivalent to PLET/RUSLE.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.driver.bmi_coupling import AnnualBmiScenarioCoupler, get_bmi_value
from src.interfaces.backend_capabilities import (
    ExternalBackendBundle, SUPPORTED_FEATURES,
)
from src.interfaces.bmi import ExternalWatershedModelBmi, LiveScenarioGeneratorBmi
from src.interfaces.component_bootstrap import ComponentStartup

SELECTION_SCHEMA_VERSION = 3
_SELECTION_FIELDS = frozenset({
    "schema_version", "backend", "component_manifest", "external_factory",
    "backend_options", "required_features", "ensemble",
})


@dataclass(frozen=True)
class EnsemblePlan:
    """Opt-in deterministic ensemble using the startup seed as a root seed."""

    n_scenarios: int
    n_jobs: int = 1
    checkpoint_dir: Path | None = None
    resume: bool = False
    additional_input_files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if type(self.n_scenarios) is not int or self.n_scenarios < 1:
            raise ValueError("Ensemble n_scenarios must be a positive integer")
        if type(self.n_jobs) is not int or not 1 <= self.n_jobs <= 64:
            raise ValueError("Ensemble n_jobs must be an integer in [1, 64]")
        if type(self.resume) is not bool:
            raise ValueError("Ensemble resume must be a boolean")
        if self.resume and self.checkpoint_dir is None:
            raise ValueError("Ensemble resume requires checkpoint_dir")
        if not isinstance(self.additional_input_files, tuple) or any(
                not isinstance(path, Path) for path in self.additional_input_files):
            raise ValueError("Ensemble additional_input_files must contain paths")


@dataclass(frozen=True)
class BackendSelection:
    """Versioned, file-resolved backend selection (single or ensemble)."""

    backend: str
    component_manifest: Path
    external_factory: str | None = None
    backend_options: Mapping[str, Any] | None = None
    required_features: frozenset[str] = frozenset()
    ensemble: EnsemblePlan | None = None

    @classmethod
    def read(cls, filename: str | Path) -> "BackendSelection":
        path = Path(filename).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Backend selection must be a JSON file")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid backend selection JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("Backend selection must be an object with known fields only")
        version = data.get("schema_version")
        if type(version) is not int or version not in (1, 2, SELECTION_SCHEMA_VERSION):
            raise ValueError("Unsupported backend selection schema version")
        allowed_fields = _SELECTION_FIELDS if version >= 2 else _SELECTION_FIELDS - {"ensemble"}
        if set(data) - allowed_fields:
            raise ValueError("Backend selection must be an object with known fields only")
        if version in (2, 3):
            ensemble_data = data.get("ensemble")
            if version == 2:
                if not isinstance(ensemble_data, dict) or set(ensemble_data) != {"n_scenarios"}:
                    raise ValueError("Version 2 requires ensemble with only n_scenarios")
                ensemble = EnsemblePlan(ensemble_data["n_scenarios"])
            else:
                allowed = {"n_scenarios", "n_jobs", "checkpoint_dir", "resume",
                           "additional_input_files"}
                if (not isinstance(ensemble_data, dict) or
                        set(ensemble_data) - allowed or
                        not {"n_scenarios", "checkpoint_dir"} <= set(ensemble_data)):
                    raise ValueError("Version 3 ensemble requires n_scenarios and checkpoint_dir with known fields only")
                directory = ensemble_data["checkpoint_dir"]
                if not isinstance(directory, str) or not directory.strip():
                    raise ValueError("Ensemble checkpoint_dir must be a nonempty path")
                extra = ensemble_data.get("additional_input_files", [])
                if (not isinstance(extra, list) or
                        any(not isinstance(item, str) or not item.strip() for item in extra) or
                        len(extra) != len(set(extra))):
                    raise ValueError("Ensemble additional_input_files must be a list of distinct nonempty paths")
                ensemble = EnsemblePlan(
                    ensemble_data["n_scenarios"],
                    n_jobs=ensemble_data.get("n_jobs", 1),
                    checkpoint_dir=(path.parent / directory).resolve(),
                    resume=ensemble_data.get("resume", False),
                    additional_input_files=tuple((path.parent / entry).resolve() for entry in extra),
                )
        else:
            ensemble = None
        backend = data.get("backend")
        if backend not in ("native", "external"):
            raise ValueError("Backend selection requires 'native' or 'external'")
        manifest = data.get("component_manifest")
        if not isinstance(manifest, str) or not manifest.strip():
            raise ValueError("Backend selection requires component_manifest path")
        component_manifest = (path.parent / manifest).resolve(strict=True)
        # Validate identity/seed/replay paths now; not after partial simulation.
        startup = ComponentStartup.read(component_manifest)
        factory = data.get("external_factory")
        options = data.get("backend_options", {})
        features = data.get("required_features", [])
        if not isinstance(features, list) or any(not isinstance(f, str) for f in features):
            raise ValueError("Backend required_features must be a list of strings")
        if len(set(features)) != len(features) or not set(features) <= SUPPORTED_FEATURES:
            raise ValueError("Backend required_features has duplicates or unknown features")
        if not isinstance(options, dict):
            raise ValueError("Backend options must be a JSON object")
        if backend == "native":
            if factory is not None or options or features:
                raise ValueError("Native backend selection cannot contain external settings")
        elif (not isinstance(factory, str) or
              len(factory.split(":")) != 2 or
              not all(part and part.isidentifier() for part in factory.replace(".", ":").split(":"))):
            raise ValueError("External backend requires a module.path:factory function")
        if ensemble is not None and startup.replay_file is not None:
            raise ValueError("Ensemble startup cannot set replay_file; use saved per-scenario replays")
        return cls(backend, component_manifest, factory, options, frozenset(features), ensemble)


def _load_external_factory(qualified: str):
    module_name, function_name = qualified.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Unable to load configured external backend factory {qualified!r}") from exc
    if not callable(factory):
        raise ValueError(f"External backend factory {qualified!r} is not callable")
    return factory


def build_selected_coupler(selection: BackendSelection) -> AnnualBmiScenarioCoupler:
    """Build a selected component pair, negotiating before RNG consumption.

    A trusted factory has keyword signature ``factory(*, startup, options)``
    and returns an ``ExternalBackendBundle``. The supplied backend is *not*
    imported until the user explicitly selects the external execution path.
    """
    spec = ComponentStartup.read(selection.component_manifest)
    if selection.backend == "native":
        return AnnualBmiScenarioCoupler.from_manifest(selection.component_manifest)
    factory = _load_external_factory(selection.external_factory)
    bundle = factory(startup=spec, options=dict(selection.backend_options or {}))
    if not isinstance(bundle, ExternalBackendBundle):
        raise TypeError("External backend factory must return ExternalBackendBundle")
    bundle.validate(selection.required_features)
    model = ExternalWatershedModelBmi(
        bundle.domain, bundle.backend, np.random.default_rng(spec.worker_seed),
    )
    generator = LiveScenarioGeneratorBmi(require_domain_import=True)
    return AnnualBmiScenarioCoupler.from_components(
        generator, model, seed=spec.worker_seed, scenario_id=spec.scenario_id,
        generator_config_file=str(selection.component_manifest),
    )


def run_selected_scenario(selection: BackendSelection) -> dict[str, Any]:
    """Execute one independently initialized realization; return its JSON record.

    A scenario owns a new coupler, model and generator. No scientific state is
    retained across calls; this also supports deterministic ensemble execution.
    """
    coupler = build_selected_coupler(selection)
    initialized = False
    try:
        coupler.initialize()
        initialized = True
        responses = []
        while (response := coupler.next_bmp()) is not None:
            responses.append({
                "cps": response.cps,
                "host_parcel_index": response.parcel_index,
                "cost_usd": response.cost_usd,
                "bmp_count": response.completed_count,
                "cumulative_cost_usd": response.total_cost_usd,
                "outlet_removed_kg": response.delivered_reductions.tolist(),
                "bmp_record": response.bmp_record,
            })
        result = {
            "result_schema_version": 1,
            "backend": selection.backend,
            "scenario_id": coupler.scenario_id,
            "worker_seed": coupler.seed,
            "index_manifest": json.loads(get_bmi_value(
                coupler.model, "coupling__index_manifest_utf8").tobytes().decode("utf-8")),
            "baseline_load_rate_kg_ha_yr": get_bmi_value(
                coupler.model, "model__parcel_baseline_load_rate").tolist(),
            "current_load_rate_kg_ha_yr": get_bmi_value(
                coupler.model, "model__parcel_current_load_rate").tolist(),
            "outlet_removed_mass_kg": get_bmi_value(
                coupler.model, "model__outlet_removed_mass").tolist(),
            "bmp_responses": responses,
            "replay_scenario": coupler.completed_scenario().to_dict(),
        }
        return result
    finally:
        if initialized:
            coupler.finalize()


def write_selected_result(result_file: str | Path, result: Mapping[str, Any]) -> Path:
    """Write one complete result, never a partially completed ensemble."""
    dest = Path(result_file).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + ".tmp")
    try:
        temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(dest)
    finally:
        if temp.exists():
            temp.unlink()
    return dest


def run_backend_selection(selection_file: str | Path, result_file: str | Path) -> Path:
    """Execute the selected single realization (v1) or ensemble (v2)."""
    selection = BackendSelection.read(selection_file)
    if selection.ensemble is not None:
        from src.driver.ensemble_runner import run_backend_ensemble
        return run_backend_ensemble(selection, result_file)
    return write_selected_result(result_file, run_selected_scenario(selection))
