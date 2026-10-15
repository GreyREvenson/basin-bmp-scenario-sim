"""Guard ordinary tests against dependencies on repository example data."""

from __future__ import annotations

from pathlib import Path


def test_tests_do_not_reference_repository_examples() -> None:
    tests_dir = Path(__file__).resolve().parent
    offenders: list[str] = []
    roots = (tests_dir, tests_dir.parent / "tests_cfg")
    paths = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*.py")
        if path.resolve() != Path(__file__).resolve()
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if "examples" in text.lower():
            offenders.append(str(path.relative_to(tests_dir.parent)))

    assert not offenders, (
        "Tests must not read repository examples; use synthetic inputs or "
        "tmp_path-generated config files instead. Offenders: "
        + ", ".join(offenders)
    )
