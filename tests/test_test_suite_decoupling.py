"""Guard against re-coupling ordinary tests to the shipped East Fork example."""

from __future__ import annotations

from pathlib import Path


def test_ordinary_tests_do_not_reference_east_fork_example_files() -> None:
    tests_dir = Path(__file__).resolve().parent
    forbidden = (
        "examples" + "/east_fork",
        "examples" + "\\east_fork",
        "east_fork" + ".yaml",
        "east_fork_plet" + ".yaml",
    )

    offenders: list[str] = []
    for path in sorted(tests_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.name}: {token}")

    assert not offenders, (
        "Ordinary tests must not depend on shipped East Fork paths or config names; "
        "use synthetic inputs or tmp_path-generated config files instead. Offenders: "
        + ", ".join(offenders)
    )
