"""Ensure subprocess-backed tests can import the installed source-layout package.

The test runner adds the source directory to its own import path, but child
Python processes need the equivalent path in their environment. This setup is
only for source-tree testing; regular users install the package normally.
"""

from __future__ import annotations

import os
from pathlib import Path


def pytest_configure(config) -> None:
    source = str((Path(__file__).resolve().parents[1] / "src").resolve())
    entries = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if source not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([source, *entries])
