"""Content identities shared by annual BMI acceptance and portability reports.

Only hashes of configuration/source files are recorded; third-party backend
scientific data must also be inventoried through the checkpoint provenance
contract. A hash is not authentication or proof of scientific validity.
"""
from __future__ import annotations

from hashlib import sha256
from importlib import resources
from pathlib import Path

from src.interfaces.component_bootstrap import ComponentStartup
from src.interfaces.bmi.public_contract import SNAPSHOT_NAME


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_contract_sha256() -> str:
    return sha256(resources.files("src.interfaces.bmi").joinpath(SNAPSHOT_NAME)
                  .read_bytes()).hexdigest()


def source_identity(selection_file: Path, manifest_file: Path) -> dict[str, str]:
    """Bind an acceptance result to the exact selected startup and API."""
    startup = ComponentStartup.read(manifest_file)
    return {
        "selection_sha256": sha256_file(selection_file),
        "startup_sha256": sha256_file(manifest_file),
        "model_config_sha256": sha256_file(startup.model_config),
        "public_contract_sha256": public_contract_sha256(),
    }
