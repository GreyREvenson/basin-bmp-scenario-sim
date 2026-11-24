"""Check the clean, canonical project layout and import boundaries."""

from __future__ import annotations

import ast
from importlib import resources
from pathlib import Path

import basin_bmp_sim
from basin_bmp_sim.certification.release_gate import audit_public_reference_backend
from basin_bmp_sim.interfaces.bmi.public_contract import verify_public_contract


def test_source_tree_has_single_package_and_no_legacy_alias_files():
    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    package = source / "basin_bmp_sim"
    assert package.is_dir()
    assert (source / "reference_backend").is_dir()
    assert {p.name for p in source.iterdir() if p.is_file() and p.suffix == ".py"} == set()
    assert not (root / "basin-bmp-sim").exists()
    assert not (package / "model.py").exists()
    assert not (package / "bmp.py").exists()
    assert not (package / "plotting.py").exists()
    assert not (package / "summaries.py").exists()


def test_internal_python_imports_use_canonical_package_name():
    package = Path(basin_bmp_sim.__file__).parent
    for file in package.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name == "src" or alias.name.startswith("src.")
                               for alias in node.names), file
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module != "src" and not node.module.startswith("src."), file


def test_packaged_scientific_assets_and_public_contract_are_accessible():
    root = resources.files("basin_bmp_sim")
    table = root.joinpath("data", "afo", "manure_excretion_reference.csv")
    assert table.is_file() and table.read_text(encoding="utf-8").startswith("species,")
    verify_public_contract()
    assert audit_public_reference_backend()["passed"]
