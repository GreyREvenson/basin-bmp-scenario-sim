"""Enforce the canonical input-loading ownership after the loader split."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "basin_bmp_sim"


def test_loader_is_orchestrator_not_a_legacy_export_surface() -> None:
    loading = import_module("basin_bmp_sim.config.loading")
    tree = ast.parse((PACKAGE / "config" / "loading.py").read_text(encoding="utf-8"))
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert functions == {"_load_pollutants", "_load_cps", "load_and_validate_all"}
    assert callable(loading.load_and_validate_all)
    for obsolete in (
        "load_bmp_selection_probabilities",
        "load_trajectory_records",
        "_load_bmp_cost",
        "_load_pollutant_load_rate",
        "_load_optional_outlet_stats",
    ):
        assert not hasattr(loading, obsolete)


def test_new_input_modules_own_their_implementations() -> None:
    modules = {
        "config.spatial_loading": ("_load_domain", "_load_parcels", "_load_delivery_ratios"),
        "config.bmp_loading": ("_load_bmp_cost", "_load_bmp_efficiency", "load_bmp_selection_probabilities"),
        "config.statistical_loading": ("_load_pollutant_load_rate", "_resolve_aggregate_pathway_fractions"),
        "results.trajectory_loading": ("load_trajectory_records",),
    }
    for path, functions in modules.items():
        module = import_module("basin_bmp_sim." + path)
        for name in functions:
            assert callable(getattr(module, name)), (path, name)


def test_lower_level_modules_do_not_import_input_orchestrator() -> None:
    paths = (
        PACKAGE / "config" / "spatial_loading.py",
        PACKAGE / "config" / "bmp_loading.py",
        PACKAGE / "config" / "statistical_loading.py",
        PACKAGE / "results" / "trajectory_loading.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "basin_bmp_sim.config.loading", path
            elif isinstance(node, ast.Import):
                assert all(alias.name != "basin_bmp_sim.config.loading" for alias in node.names), path
