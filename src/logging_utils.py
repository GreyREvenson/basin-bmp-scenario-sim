"""
Logging helpers for driver and worker processes.

- Driver logger optionally writes to console and/or a file.
- Worker loggers write a dedicated file per scenario under outputs/logs/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple


def make_logger(outputs_dir: Path, verbose: bool = True, scenario_id: Optional[int] = None) -> Tuple[logging.Logger, Optional[Path]]:
    """Create a driver logger.

    Parameters
    ----------
    outputs_dir : Path
        Root outputs directory.
    verbose : bool, default True
        If True, also log to console at INFO level.
    scenario_id : Optional[int]
        If provided, the driver also logs to outputs/logs/s{scenario_id}.txt.

    Returns
    -------
    (logging.Logger, Optional[Path])
        The logger and optional log file path if scenario_id is provided.
    """
    outputs_dir = Path(outputs_dir)
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bmp-sim")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    log_path = None
    if scenario_id is not None:
        log_path = logs_dir / f"s{scenario_id}.txt"
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger, log_path


def make_worker_logger(outputs_dir: Path, scenario_id: int) -> logging.Logger:
    """Create a per-scenario logger writing into outputs/logs/s{scenario_id}.txt.

    Notes
    -----
    Workers do not log to console to avoid interleaving stdout with the driver.
    """
    outputs_dir = Path(outputs_dir)
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"bmp-sim-s{scenario_id}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    log_path = logs_dir / f"s{scenario_id}.txt"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger