"""Opt-in selection and execution of native or external annual BMI models.

The project's historical Monte Carlo CLI continues to use its native model
and Parquet outputs. This module supplies a separate model-neutral execution
path, with explicitly trusted external Python factories. It does not silently
re-label third-party outputs as calibrated PLET/RUSLE results.
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

SELECTION_SCHEMA_VERSION = 1
_SELECTION_FIELDS = frozenset({
    "schema_version", "backend", "component_manifest", "external_factory",
    "backend_options", "required_features",
})


@dataclass(frozen=True)
class BackendSelection:
    """Versioned, file-resolved backend selection for a *single* scenario."""

    backend: str
    component_manifest: Path
    external_factory: str | None = None
    backend_options: Mapping[str, Any] | None = None
    required_features: frozenset[str] = frozenset()

    @classmethod
    def read(cls, filename: str | Path) -> "BackendSelection":
        path = Path(filename).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Backend selection must be a JSON file")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid backend selection JSON") from exc
        if not isinstance(data, dict) or (set(data) - _SELECTION_FIELDS):
            raise ValueError("Backend selection must be an object with known fields only")
        if type(data.get("schema_version")) is not int or data["schema_version"] != SELECTION_SCHEMA_VERSION:
            raise ValueError("Unsupported backend selection schema version")
        backend = data.get("backend")
        if backend not in ("native", "external"):
            raise ValueError("Backend selection requires 'native' or 'external'")
        manifest = data.get("component_manifest")
        if not isinstance(manifest, str) or not manifest.strip():
            raise ValueError("Backend selection requires component_manifest path")
        component_manifest = (path.parent / manifest).resolve(strict=True)
        # Validate identity/seed/replay paths now; not after partial simulation.
        ComponentStartup.read(component_manifest)
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
        return cls(backend, component_manifest, factory, options, frozenset(features))


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


def run_backend_selection(selection_file: str | Path, result_file: str | Path) -> Path:
    """Run one selected BMI realization and atomically save model-neutral JSON.

    This intentionally does *not* write the native PLET/AFO Parquet tables or
    claim that third-party BMP records have the native output schema.
    """
    selection = BackendSelection.read(selection_file)
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
    finally:
        if initialized:
            coupler.finalize()
