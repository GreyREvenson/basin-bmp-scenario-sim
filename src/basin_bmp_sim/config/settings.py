"""Configuration normalization, path resolution, and default settings."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Union
from basin_bmp_sim.common.utils import ci_get
from basin_bmp_sim.domain.constants import (
    CFG_N_SCENARIOS, CFG_OUTPUTS, CFG_VERBOSE, CFG_BUFFER_DEPTH_FT,
    CFG_BMP_SEL_PROB_VIA_COSTS, CFG_BMP_FAIL_RATE, CFG_BMP_FAIL_REDUCTION,
    CFG_DETERMINISTIC_PARCEL_ORDER, CFG_DETERMINISTIC_BMP_TYPE_ORDER,
    CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS, CFG_PARALLEL, CFG_LOAD_GENERATION,
    DEFAULT_BUFFER_DEPTH_FT, DEFAULT_BMP_FAIL_REDUCTION,
    CONFIG_PATH_KEYS as _CONFIG_PATH_KEYS,
    LOAD_GENERATION_PATH_KEYS as _LOAD_GENERATION_PATH_KEYS,
)

def _set_case_insensitive_default(mapping: Dict[str, Any], key: str, value: Any) -> None:
    """Set one input default while respecting existing case-insensitive keys.

    Parameters
    ----------
    mapping : Dict[str, Any]
        Input mapping.
    key : str
        Configuration or mapping key.
    value : Any
        Input value to normalize or evaluate.
    """
    matching_key = next((existing for existing in mapping if str(existing).lower() == key.lower()), None)
    if matching_key is None:
        mapping[key] = value
    elif mapping[matching_key] is None:
        mapping[matching_key] = value

def apply_config_defaults(cfg: Dict[str, Any]) -> None:
    """Apply all supported top-level configuration defaults in one place.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Normalized model configuration mapping.
    """
    _set_case_insensitive_default(cfg, CFG_N_SCENARIOS, 1)
    _set_case_insensitive_default(cfg, CFG_OUTPUTS, "./outputs")
    _set_case_insensitive_default(cfg, CFG_VERBOSE, False)
    _set_case_insensitive_default(cfg, CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT)
    _set_case_insensitive_default(cfg, CFG_BMP_SEL_PROB_VIA_COSTS, False)
    _set_case_insensitive_default(cfg, CFG_BMP_FAIL_RATE, 0.0)
    _set_case_insensitive_default(cfg, CFG_BMP_FAIL_REDUCTION, DEFAULT_BMP_FAIL_REDUCTION)
    _set_case_insensitive_default(cfg, CFG_DETERMINISTIC_PARCEL_ORDER, False)
    _set_case_insensitive_default(cfg, CFG_DETERMINISTIC_BMP_TYPE_ORDER, False)
    _set_case_insensitive_default(cfg, CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS, False)
    _set_case_insensitive_default(cfg, CFG_PARALLEL, {"n_jobs": 1})
    _set_case_insensitive_default(cfg, CFG_LOAD_GENERATION, {})

def _resolve_config_path_value(value: Any, base_dir: Path) -> Any:
    """Resolve one configured filesystem path relative to a config directory.

    Scalar paths and sequences of paths are supported. Non-path values are
    returned unchanged so this helper can be applied only to known path keys.
    """
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return str(path.resolve())
    if isinstance(value, list):
        return [_resolve_config_path_value(item, base_dir) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_config_path_value(item, base_dir) for item in value)
    return value

def resolve_config_paths(cfg: Dict[str, Any], config_path: Union[str, Path]) -> Dict[str, Any]:
    """Resolve configured file locations against the YAML file's directory.

    This removes any dependency on the process working directory. Resolution
    happens once in the parent process, before inputs are loaded and before
    parallel scenario workers are launched. Absolute paths are preserved.

    Parameters
    ----------
    cfg : dict[str, Any]
        Normalized model configuration.
    config_path : str or pathlib.Path
        Path to the YAML file that supplied ``cfg``.

    Returns
    -------
    dict[str, Any]
        The same mapping object with known path-valued entries resolved to
        absolute paths.
    """
    config_file = Path(config_path).expanduser().resolve()
    base_dir = config_file.parent

    for key in _CONFIG_PATH_KEYS:
        value = ci_get(cfg, key)
        if value is not None:
            cfg[key] = _resolve_config_path_value(value, base_dir)

    load_generation = ci_get(cfg, CFG_LOAD_GENERATION)
    if isinstance(load_generation, dict):
        normalized_load_generation = {str(key).lower(): value for key, value in load_generation.items()}
        load_generation.clear()
        load_generation.update(normalized_load_generation)
        for key in _LOAD_GENERATION_PATH_KEYS:
            if key in load_generation and load_generation[key] is not None:
                load_generation[key] = _resolve_config_path_value(
                    load_generation[key], base_dir
                )

    manure_loading = ci_get(cfg, "manure_loading")
    if isinstance(manure_loading, dict):
        normalized_manure_loading = {str(key).lower(): value for key, value in manure_loading.items()}
        manure_loading.clear()
        manure_loading.update(normalized_manure_loading)
        if manure_loading.get("excretion_factors") is not None:
            manure_loading["excretion_factors"] = _resolve_config_path_value(
                manure_loading["excretion_factors"], base_dir
            )

    return cfg

def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize configuration keys and apply centralized defaults in place.

    Parameters
    ----------
    cfg : dict[str, Any]
        Raw configuration mapping, typically returned by
        :func:`basin_bmp_sim.common.io.read_config`.

    Returns
    -------
    dict[str, Any]
        The same mapping object after top-level key normalization and default
        application.
    """
    normalized = {str(key).lower(): value for key, value in cfg.items()}
    cfg.clear()
    cfg.update(normalized)
    apply_config_defaults(cfg)
    parallel = ci_get(cfg, CFG_PARALLEL)
    if isinstance(parallel, dict):
        normalized_parallel = {str(key).lower(): value for key, value in parallel.items()}
        parallel.clear()
        parallel.update(normalized_parallel)
        if "n_jobs" not in parallel or parallel["n_jobs"] is None:
            parallel["n_jobs"] = 1
    return cfg
