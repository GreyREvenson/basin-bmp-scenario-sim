"""Content-addressed scientific-input inventory for BMI ensemble checkpoints.

The native loader defines explicit filesystem-valued YAML keys; do not try
heuristically to interpret arbitrary configuration strings as filenames. The
inventory covers each existing file at those keys, shapefile sidecars, the
bundled AFO default reference when used, and optional external-backend assets
explicitly declared by the operator. Outputs are intentionally excluded.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from basin_bmp_sim.domain.constants import CONFIG_PATH_KEYS, LOAD_GENERATION_PATH_KEYS
from basin_bmp_sim.common.io import read_config
from basin_bmp_sim.common.utils import ci_get

_MANIFEST_VERSION = 1
_INPUT_PATH_KEYS = tuple(key for key in CONFIG_PATH_KEYS if key != "outputs")
_SHAPEFILE_SIDECARS = (".dbf", ".shx", ".prj", ".cpg", ".qpj")


def _path_values(value: Any, label: str) -> Iterable[tuple[str, str]]:
    """Yield scalar/list native path references; inline mappings are not files."""
    if isinstance(value, (str, Path)):
        if not str(value).strip():
            raise ValueError(f"Scientific input {label} has a blank path")
        yield label, str(value)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _path_values(item, f"{label}[{index}]")
    elif value is not None and not isinstance(value, dict):
        raise ValueError(f"Scientific input {label} has an unsupported path value")


def _file_content_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_for_path(path: Path) -> tuple[Path, ...]:
    """Expand GeoPackage/CSV files, shapefile datasets, and directory inputs."""
    if path.is_file():
        paths = [path]
        if path.suffix.lower() == ".shp":
            # A shapefile's geometry, attributes and projection live in
            # separate files; a .shp hash alone would be insufficient.
            paths.extend(side for side in path.parent.iterdir()
                         if side.is_file() and side.stem.casefold() == path.stem.casefold()
                         and side.suffix.lower() in _SHAPEFILE_SIDECARS)
        return tuple(paths)
    if path.is_dir():
        files = tuple(sorted((item for item in path.rglob("*") if item.is_file()),
                             key=lambda item: item.as_posix()))
        if not files:
            raise ValueError(f"Scientific input directory is empty: {path}")
        return files
    raise FileNotFoundError(f"Scientific input file or directory does not exist: {path}")


def scientific_input_manifest(model_config: Path,
                              additional_files: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Record content hashes of file-backed scientific inputs without sampling.

    Paths are absolute, so the inventory is independent of the caller's CWD.
    Identical paths used by multiple config keys are recorded once, with all
    referencing labels. Scientific input *contents* (not mtimes) determine
    the run fingerprint. The YAML's own SHA is handled by the caller.
    """
    cfg = read_config(model_config)
    base = model_config.parent
    references: list[tuple[str, Path]] = []
    for key in _INPUT_PATH_KEYS:
        for label, raw in _path_values(ci_get(cfg, key), key):
            references.append((label, (base / Path(raw).expanduser()).resolve()))
    generation = ci_get(cfg, "load_generation")
    if isinstance(generation, dict):
        for key in LOAD_GENERATION_PATH_KEYS:
            for label, raw in _path_values(ci_get(generation, key), f"load_generation.{key}"):
                references.append((label, (base / Path(raw).expanduser()).resolve()))
    manure = ci_get(cfg, "manure_loading")
    if isinstance(manure, dict):
        for label, raw in _path_values(ci_get(manure, "excretion_factors"),
                                       "manure_loading.excretion_factors"):
            references.append((label, (base / Path(raw).expanduser()).resolve()))
        if (ci_get(cfg, "animal_operations") is not None and
                str(ci_get(manure, "method") or "concentration_increment").lower() == "mass_based" and
                ci_get(manure, "excretion_factors") is None):
            from basin_bmp_sim.science.afo import excretion
            references.append(("bundled_afo_excretion_factors",
                               Path(excretion.__file__).resolve().parents[2] / "data" / "afo" /
                               "manure_excretion_reference.csv"))
    for index, path in enumerate(additional_files):
        references.append((f"additional_files[{index}]", path.resolve()))

    labels_by_path: dict[Path, set[str]] = {}
    for label, source in references:
        for data_file in _files_for_path(source):
            resolved = data_file.resolve()
            labels_by_path.setdefault(resolved, set()).add(label)
    inventory = [
        {"path": str(path), "labels": sorted(labels),
         "size_bytes": path.stat().st_size, "sha256": _file_content_sha256(path)}
        for path, labels in sorted(labels_by_path.items(), key=lambda pair: str(pair[0]))
    ]
    return {"schema_version": _MANIFEST_VERSION, "files": inventory}
