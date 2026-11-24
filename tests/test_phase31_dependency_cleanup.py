"""Architectural boundaries and deduplicated implementation contracts."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

from basin_bmp_sim.common.utils import is_nonblank
from basin_bmp_sim.common.numerics import _safe_mass_ratio
from basin_bmp_sim.common.distributions import stats_from_row
from basin_bmp_sim.domain.parameter_validation import validate_parameter_value
from basin_bmp_sim.science.plet_rows import _rows_for_pid

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "basin_bmp_sim"


def test_no_obsolete_import_surfaces() -> None:
    removed = (
        "config/distributions.py", "config/units.py", "config/validation/domains.py",
    )
    for name in removed:
        assert not (PACKAGE / name).exists(), name
    for name in (
        "common.distributions", "common.units", "domain.parameter_validation",
        "common.utils", "common.numerics", "science.plet_rows",
    ):
        import_module("basin_bmp_sim." + name)
    for source in PACKAGE.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            elif isinstance(node, ast.Import):
                imports = [entry.name for entry in node.names]
            else:
                continue
            assert all(not value.startswith((
                "basin_bmp_sim.config.distributions",
                "basin_bmp_sim.config.units",
                "basin_bmp_sim.config.validation.domains",
            )) for value in imports), (source, node.lineno)


def test_low_level_science_has_no_loader_or_driver_dependency() -> None:
    for package in ("science", "bmp", "domain", "scenarios"):
        for source in (PACKAGE / package).rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imports = [node.module or ""]
                elif isinstance(node, ast.Import):
                    imports = [entry.name for entry in node.names]
                else:
                    continue
                for name in imports:
                    assert not name.startswith((
                        "basin_bmp_sim.execution.",
                        "basin_bmp_sim.config.loading",
                        "basin_bmp_sim.config.table_sources",
                    )), (source, node.lineno, name)


def test_one_owner_for_repeated_helpers() -> None:
    functions = {}
    for source in PACKAGE.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.setdefault(node.name, []).append(source.relative_to(PACKAGE).as_posix())
    assert functions.get("_rows_for_pid") == ["science/plet_rows.py"]
    assert functions.get("_safe_mass_ratio") == ["common/numerics.py"]
    assert "_nonblank" not in functions
    assert "_row_stats_raw" not in functions
    assert "validate_stats_table" not in functions


def test_shared_functions_preserve_values_and_override_precedence() -> None:
    assert is_nonblank("  x  ")
    assert not is_nonblank(" ")
    assert not is_nonblank(np.nan)
    assert not is_nonblank(None)
    assert _safe_mass_ratio(-2, 4) == -0.5
    assert _safe_mass_ratio(2, 0) is None
    assert _safe_mass_ratio(float("nan"), 4) is None
    assert stats_from_row({"value": 3.0}) == {"value": 3.0}
    assert validate_parameter_value("cn", 75) == 75

    table = pd.DataFrame([
        {"pid": pd.NA, "parameter": "cn", "value": 60},
        {"pid": pd.NA, "parameter": "ia_ratio", "value": 0.2},
        {"pid": 1, "parameter": "cn", "value": 70},
    ])
    rows = _rows_for_pid(table, "1")
    assert {row["parameter"]: row["value"] for row in rows} == {"cn": 70, "ia_ratio": 0.2}
    rows_other = _rows_for_pid(table, "2")
    assert {row["parameter"]: row["value"] for row in rows_other} == {"cn": 60, "ia_ratio": 0.2}
