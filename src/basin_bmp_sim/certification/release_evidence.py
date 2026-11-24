"""Cross-platform content identities for annual BMI release evidence.

Scientific files are hashed by their *contents and semantic labels*, not by
machine-specific absolute paths. Third-party assets must be explicitly listed;
this does not authenticate unsigned reports or freeze installed dependencies.
"""
from __future__ import annotations

from hashlib import sha256
from importlib import resources, util
import json
from pathlib import Path
from typing import Sequence

from basin_bmp_sim.execution.input_provenance import scientific_input_manifest
from basin_bmp_sim.interfaces.component_bootstrap import ComponentStartup
from basin_bmp_sim.interfaces.bmi.public_contract import SNAPSHOT_NAME


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_contract_sha256() -> str:
    return sha256(resources.files("basin_bmp_sim.interfaces.bmi").joinpath(SNAPSHOT_NAME)
                  .read_bytes()).hexdigest()


def _external_factory_source_hash(factory: str | None) -> str | None:
    """Identify the factory entrypoint file, without assuming its dependencies."""
    if factory is None:
        return None
    module_name = factory.split(":", 1)[0]
    try:
        spec = util.find_spec(module_name)
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        raise ValueError(f"Cannot locate external backend source {module_name!r}") from exc
    if spec is None or spec.origin is None or not Path(spec.origin).is_file():
        raise ValueError(f"External backend source is not a regular file: {module_name}")
    return sha256_file(spec.origin)


def _portable_input_digest(model_config: Path, additional: tuple[Path, ...]) -> str:
    """Exclude absolute paths so identical Windows/Linux datasets can agree.

    The same input label and suffix are stable across OSes; contents, byte
    length, and the number of matching files are still part of the digest.
    """
    manifest = scientific_input_manifest(model_config, additional)
    rows = sorted((tuple(item["labels"]), Path(item["path"]).suffix.lower(),
                   item["size_bytes"], item["sha256"]) for item in manifest["files"])
    payload = json.dumps({"schema_version": 1, "files": rows},
                         separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


def source_identity(selection_file: Path, manifest_file: Path, *,
                    additional_files: Sequence[str | Path] = ()) -> dict[str, str | int | None]:
    """Bind evidence to selection, startup, scientific data, backend source and API.

    Additional paths are resolved relative to the selection file. Each evidence
    command must receive the *same ordered* asset list on each platform.
    Existing v3 ensemble additional-input declarations are also honored.
    """
    from basin_bmp_sim.execution.backend_selection import BackendSelection

    selected = BackendSelection.read(selection_file)
    startup = ComponentStartup.read(manifest_file)
    base = Path(selection_file).resolve().parent
    extras: list[Path] = []
    for item in additional_files:
        if not isinstance(item, (str, Path)) or not str(item).strip():
            raise ValueError("Release evidence input files must be nonempty paths")
        candidate = Path(item).expanduser()
        extras.append((candidate if candidate.is_absolute() else base / candidate).resolve(strict=True))
    if selected.ensemble is not None:
        extras.extend(selected.ensemble.additional_input_files)
    if len(extras) != len(set(extras)):
        raise ValueError("Release evidence input files must be distinct")
    return {
        "selection_sha256": sha256_file(selection_file),
        "startup_sha256": sha256_file(manifest_file),
        "model_config_sha256": sha256_file(startup.model_config),
        "public_contract_sha256": public_contract_sha256(),
        "scientific_inputs_sha256": _portable_input_digest(startup.model_config, tuple(extras)),
        "external_backend_source_sha256": _external_factory_source_hash(selected.external_factory),
        "additional_scientific_input_count": len(extras),
    }
