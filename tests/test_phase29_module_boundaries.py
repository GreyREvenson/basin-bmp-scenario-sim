"""Verify canonical module ownership after splitting configuration and PLET code."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path


_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "basin_bmp_sim"


def test_removed_monoliths_have_no_compatibility_stubs() -> None:
    assert not (_PACKAGE / "config" / "validation.py").exists()
    assert not (_PACKAGE / "science" / "plet_rusle.py").exists()

    # Validate canonical imports rather than relying on old re-export modules.
    names = (
        "config.validation.domains",
        "config.validation.statistics",
        "config.validation.configuration",
        "config.validation.plet",
        "config.validation.selection",
        "config.validation.statistical",
        "config.table_sources",
        "config.plet_loading",
        "science.plet_state",
        "science.plet_classification",
        "science.plet_sampling",
        "science.plet_initialization",
    )
    for name in names:
        import_module("basin_bmp_sim." + name)


def test_scientific_classification_does_not_import_input_loaders() -> None:
    source = (_PACKAGE / "science" / "plet_classification.py").read_text(encoding="utf-8")
    parsed = ast.parse(source)
    imports = [node.module for node in ast.walk(parsed) if isinstance(node, ast.ImportFrom)]
    assert all(not (name or "").startswith("basin_bmp_sim.config") for name in imports)


def test_code_imports_only_canonical_plet_and_validation_modules() -> None:
    removed = {
        "basin_bmp_sim.config.validation",
        "basin_bmp_sim.science.plet_rusle",
    }
    for source_path in _PACKAGE.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in removed, (source_path, node.lineno)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in removed, (source_path, node.lineno)
