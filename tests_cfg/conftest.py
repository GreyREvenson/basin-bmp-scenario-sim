from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--model-config",
        action="store",
        default=None,
        help="Path to a real model YAML to validate with the optional integration test.",
    )


@pytest.fixture
def model_config_path(request) -> Path:
    raw = request.config.getoption("--model-config")
    if not raw:
        pytest.skip("Optional integration test requires --model-config PATH_TO_YAML")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"Configured YAML does not exist: {path}")
    return path
