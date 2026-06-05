"""
Logging helpers for driver and worker processes.

- Adds a custom VERBOSE level (between INFO and DEBUG) for detailed progress logs.
- Adds a stack-based indented formatter for all log messages.
- Driver logger optionally writes to console and/or a file.
- Worker loggers write a dedicated file per scenario under outputs/logs/.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

# Define a VERBOSE level between INFO (20) and DEBUG (10)
VERBOSE_LEVEL_NUM = 15
logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")


def _verbose(self: logging.Logger, msg: str, *args, **kwargs) -> None:
    """Logger.verbose(msg, ...) -> log at VERBOSE level."""
    if self.isEnabledFor(VERBOSE_LEVEL_NUM):
        self.log(VERBOSE_LEVEL_NUM, msg, *args, **kwargs)


# Add the .verbose() convenience method to all Logger instances
logging.Logger.verbose = _verbose  # type: ignore[attr-defined]


# Thread-local indentation depth
_TL = threading.local()
_TL.depth = 0


def _get_depth() -> int:
    d = getattr(_TL, "depth", 0)
    try:
        return int(d)
    except Exception:
        return 0


def push_indent(n: int = 1) -> None:
    """Increase the current indentation depth by n (default 1)."""
    setattr(_TL, "depth", max(0, _get_depth() + int(n)))


def pop_indent(n: int = 1) -> None:
    """Decrease the current indentation depth by n (default 1)."""
    setattr(_TL, "depth", max(0, _get_depth() - int(n)))


@contextmanager
def log_scope(label: Optional[str] = None, logger: Optional[logging.Logger] = None, level: int = VERBOSE_LEVEL_NUM):
    """Context manager that indents logs within the scope and optionally logs BEGIN/END.

    Parameters
    ----------
    label : Optional[str]
        If provided with logger, logs 'BEGIN {label}' and 'END {label}'.
    logger : Optional[logging.Logger]
        Logger to emit BEGIN/END lines to.
    level : int
        Logging level for BEGIN/END; defaults to VERBOSE.
    """
    if label and logger is not None:
        logger.log(level, f"BEGIN {label}")
    push_indent(1)
    try:
        yield
    finally:
        pop_indent(1)
        if label and logger is not None:
            logger.log(level, f"END {label}")


class StackIndentFilter(logging.Filter):
    """Injects an 'indent' attribute based on the current thread-local depth."""
    def __init__(self, indent_unit: str = "  "):
        super().__init__()
        self.indent_unit = indent_unit

    def filter(self, record: logging.LogRecord) -> bool:
        depth = _get_depth()
        # Precompute indent string for the formatter
        record.indent = self.indent_unit * depth
        return True


class StackIndentFormatter(logging.Formatter):
    """Formatter that respects an 'indent' attribute inserted by StackIndentFilter.

    If 'indent' is missing, it defaults to empty string.
    """
    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "indent"):
            record.indent = ""
        return super().format(record)


def _make_console_handler(verbose: bool) -> logging.Handler:
    ch = logging.StreamHandler()
    ch.addFilter(StackIndentFilter(indent_unit="  "))
    ch.setFormatter(StackIndentFormatter("%(indent)s%(message)s"))
    ch.setLevel(VERBOSE_LEVEL_NUM if verbose else logging.INFO)
    return ch


def _make_file_handler(path: Path, verbose: bool) -> logging.Handler:
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.addFilter(StackIndentFilter(indent_unit="  "))
    fh.setFormatter(StackIndentFormatter("%(asctime)s | %(levelname)s | %(indent)s%(message)s"))
    fh.setLevel(VERBOSE_LEVEL_NUM if verbose else logging.INFO)
    return fh


def make_logger(
    outputs_dir: Path,
    verbose: bool = True,
    scenario_id: Optional[int] = None,
) -> Tuple[logging.Logger, Optional[Path]]:
    """Create a driver logger.

    Parameters
    ----------
    outputs_dir : Path
        Root outputs directory.
    verbose : bool, default True
        If True, also log to console and set logger/handlers to VERBOSE.
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
    logger.handlers = []
    logger.filters = []  # reset filters to avoid duplication on repeated setup
    logger.propagate = False

    # Attach a single indent filter at logger level (applies to all handlers)
    logger.addFilter(StackIndentFilter(indent_unit="  "))

    # Default threshold is INFO; when verbose is True, lower to VERBOSE
    logger.setLevel(VERBOSE_LEVEL_NUM if verbose else logging.INFO)

    log_path = None
    if scenario_id is not None:
        log_path = logs_dir / f"s{scenario_id}.txt"
        logger.addHandler(_make_file_handler(log_path, verbose=verbose))

    # Console handler: attach only when verbose console logging is desired
    if verbose:
        logger.addHandler(_make_console_handler(verbose=True))
    else:
        # Even when not verbose, emit INFO+ to console
        logger.addHandler(_make_console_handler(verbose=False))

    # Initialization message (using VERBOSE)
    logger.verbose("Driver logger initialized")

    return logger, log_path


def make_worker_logger(outputs_dir: Path, scenario_id: int, verbose: bool = False) -> logging.Logger:
    """Create a per-scenario logger writing into outputs/logs/s{scenario_id}.txt.

    Notes
    -----
    Workers do not log to console to avoid interleaving stdout with the driver.
    """
    outputs_dir = Path(outputs_dir)
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"bmp-sim-s{scenario_id}")
    logger.handlers = []
    logger.filters = []
    logger.propagate = False

    # Attach indent filter at logger level
    logger.addFilter(StackIndentFilter(indent_unit="  "))

    # Default threshold is INFO; lower to VERBOSE when verbose True
    logger.setLevel(VERBOSE_LEVEL_NUM if verbose else logging.INFO)

    log_path = logs_dir / f"s{scenario_id}.txt"
    logger.addHandler(_make_file_handler(log_path, verbose=verbose))

    # Initialization message (using VERBOSE)
    logger.verbose(f"Worker logger initialized for scenario {scenario_id}")

    return logger