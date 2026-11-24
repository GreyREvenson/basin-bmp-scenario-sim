"""Atomic same-directory file publication with bounded Windows lock retries."""

from __future__ import annotations

import os
from pathlib import Path
from time import sleep


def replace_with_windows_retry(source: str | Path, destination: str | Path) -> None:
    """Publish atomically; retry only transient Windows sharing/access errors.

    Do not unlink the destination on failure. A persistent permission error
    remains an error, and the caller retains responsibility for temp cleanup.
    """
    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == 6:
                raise
            sleep(0.05 * (2 ** attempt))
