"""Versioned, file-based startup for independently owned BMI components.

The manifest supplies *identity and initialization parameters*, not Python
objects or pickled state. Each component loads and validates its own model
inputs from the named source YAML and constructs its own worker context.
The shared scientific *index* contract and physical domain still cross BMI.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

from src.input_config import load_and_validate_all, normalize_config, resolve_config_paths
from src.input_validation import validate_config
from src.io_utils import read_config
from src.models.model_context import ScenarioModelContext
from src.models.prepared_inputs import PreparedModelInputs
from src.scenarios.generator_context import GeneratorModelContext


COMPONENT_STARTUP_SCHEMA_VERSION = 1
_ALLOWED_FIELDS = frozenset({
    "schema_version", "model_config", "worker_seed", "scenario_id", "replay_file",
})
_REQUIRED_FIELDS = frozenset({"schema_version", "model_config", "worker_seed", "scenario_id"})


@dataclass(frozen=True)
class ComponentStartup:
    """Inputs needed to build a single independent realization component."""

    model_config: Path
    worker_seed: int
    scenario_id: int
    replay_file: Path | None = None

    @classmethod
    def read(cls, path: str | Path) -> "ComponentStartup":
        manifest_path = Path(path).resolve(strict=True)
        if not manifest_path.is_file():
            raise ValueError("BMI startup manifest must be a JSON file")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid BMI startup manifest JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("BMI startup manifest must be a JSON object")
        if _REQUIRED_FIELDS - raw.keys() or raw.keys() - _ALLOWED_FIELDS:
            raise ValueError("BMI startup manifest has missing or unknown fields")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != COMPONENT_STARTUP_SCHEMA_VERSION:
            raise ValueError("Unsupported BMI startup schema version")
        for field in ("worker_seed", "scenario_id"):
            if type(raw[field]) is not int:
                raise ValueError(f"BMI startup {field} must be an integer")
        if raw["worker_seed"] < 0 or raw["worker_seed"] >= 2**64:
            raise ValueError("BMI startup worker_seed must be in [0, 2**64)")
        if raw["scenario_id"] < 1:
            raise ValueError("BMI startup scenario_id must be positive")
        if not isinstance(raw["model_config"], str) or not raw["model_config"].strip():
            raise ValueError("BMI startup model_config must be a nonempty path")
        if "replay_file" in raw and (not isinstance(raw["replay_file"], str) or not raw["replay_file"].strip()):
            raise ValueError("BMI startup replay_file must be a nonempty path")
        def resolved(value: str) -> Path:
            candidate = Path(value)
            return (candidate if candidate.is_absolute() else manifest_path.parent / candidate).resolve(strict=True)
        cfg = resolved(raw["model_config"])
        if not cfg.is_file():
            raise ValueError("BMI startup model_config must reference a file")
        replay = resolved(raw["replay_file"]) if "replay_file" in raw else None
        if replay is not None and not replay.is_file():
            raise ValueError("BMI startup replay_file must reference a file")
        return cls(cfg, raw["worker_seed"], raw["scenario_id"], replay)


def build_scientific_context(spec: ComponentStartup, *, logger: logging.Logger | None = None) -> ScenarioModelContext:
    """Load/validate project inputs locally; no Monte Carlo Model is created."""
    cfg = normalize_config(read_config(spec.model_config))
    resolve_config_paths(cfg, spec.model_config)
    validate_config(cfg)
    own_logger = logger or logging.getLogger("basin_bmp.bmi.component")
    data = load_and_validate_all(cfg, own_logger)
    prepared = PreparedModelInputs(cfg, data, own_logger)
    return ScenarioModelContext(cfg, prepared._shared_payload(), own_logger, spec.worker_seed)


def build_generator_context(spec: ComponentStartup, *, logger: logging.Logger | None = None) -> GeneratorModelContext:
    """Build a generator-owned snapshot without sharing watershed state."""
    ctx = build_scientific_context(spec, logger=logger)
    return GeneratorModelContext(ctx, spec.worker_seed)
