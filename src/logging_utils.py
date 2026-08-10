"""Logging helpers for readable run output.

This module adds a custom VERBOSE level, indentation-aware formatting, and
logger factory helpers for the main process and worker scenarios.
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
    """Log a message at the custom VERBOSE level.

    Parameters
    ----------
    self : logging.Logger
        Logger instance.
    msg : str
        Log message.
    *args
        Positional formatting arguments.
    **kwargs
        Keyword formatting arguments.

    Returns
    -------
    None
    """
    if self.isEnabledFor(VERBOSE_LEVEL_NUM):
        self.log(VERBOSE_LEVEL_NUM, msg, *args, **kwargs)


# Add the .verbose() convenience method to all Logger instances
logging.Logger.verbose = _verbose  # type: ignore[attr-defined]


# Thread-local indentation depth
_TL = threading.local()
_TL.depth = 0


def _get_depth() -> int:
    """Return the current indentation depth for the active thread.

    Returns
    -------
    int
        Current indentation depth, clamped to a valid integer.
    """
    d = getattr(_TL, "depth", 0)
    try:
        return int(d)
    except Exception:
        return 0


def push_indent(n: int = 1) -> None:
    """Increase thread-local log indentation.

    Parameters
    ----------
    n : int, optional
        Number of indentation levels to add. Default is ``1``.

    Returns
    -------
    None
    """
    setattr(_TL, "depth", max(0, _get_depth() + int(n)))


def pop_indent(n: int = 1) -> None:
    """Decrease thread-local log indentation.

    Parameters
    ----------
    n : int, optional
        Number of indentation levels to remove. Default is ``1``.

    Returns
    -------
    None
    """
    setattr(_TL, "depth", max(0, _get_depth() - int(n)))


@contextmanager
def log_scope(label: Optional[str] = None, logger: Optional[logging.Logger] = None, level: int = VERBOSE_LEVEL_NUM):
    """Create an indentation scope for a block of logs.

    Parameters
    ----------
    label : str or None, optional
        Optional label written as BEGIN/END messages. Default is ``None``.
    logger : logging.Logger or None, optional
        Logger used for BEGIN/END messages. Default is ``None``.
    level : int, optional
        Log level used for scope messages. Default is ``VERBOSE_LEVEL_NUM``.

    Yields
    ------
    None
        Control to the wrapped block.
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
    """Attach indentation text to log records.

    The filter reads the current thread-local indentation depth and stores a
    precomputed ``indent`` field on each record.
    """
    def __init__(self, indent_unit: str = "  "):
        """Initialize the filter.

        Parameters
        ----------
        indent_unit : str, optional
            Text used for one indentation level. Default is two spaces.

        Returns
        -------
        None
        """
        super().__init__()
        self.indent_unit = indent_unit

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach indentation text before formatting.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to update.

        Returns
        -------
        bool
            Always ``True`` so the record continues through the pipeline.
        """
        depth = _get_depth()
        # Precompute indent string for the formatter
        record.indent = self.indent_unit * depth
        return True


class StackIndentFormatter(logging.Formatter):
    """Format log records while preserving indentation text."""
    def format(self, record: logging.LogRecord) -> str:
        """Format one log line safely.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format.

        Returns
        -------
        str
            Formatted log line.
        """
        if not hasattr(record, "indent"):
            record.indent = ""
        return super().format(record)


def _make_console_handler(verbose: bool) -> logging.Handler:
    """Create the console log handler.

    Parameters
    ----------
    verbose : bool
        Retained for interface consistency.

    Returns
    -------
    logging.Handler
        Configured console handler.
    """
    ch = logging.StreamHandler()
    ch.addFilter(StackIndentFilter(indent_unit="  "))
    ch.setFormatter(StackIndentFormatter("%(indent)s%(message)s"))
    ch.setLevel(logging.INFO)  # INFO-only on console
    return ch


def _make_file_handler(path: Path, verbose: bool) -> logging.Handler:
    """Create the file log handler.

    Parameters
    ----------
    path : pathlib.Path
        File path for the log output.
    verbose : bool
        Whether the handler should emit VERBOSE-level messages.

    Returns
    -------
    logging.Handler
        Configured file handler.
    """
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.addFilter(StackIndentFilter(indent_unit="  "))
    fh.setFormatter(StackIndentFormatter("%(asctime)s | %(levelname)s | %(indent)s%(message)s"))
    fh.setLevel(VERBOSE_LEVEL_NUM if verbose else logging.INFO)
    return fh


def _reset_logger(logger: logging.Logger) -> None:
    """Remove handlers and filters from a logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger to reset.

    Returns
    -------
    None
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            # Best effort: keep logger reconfiguration resilient.
            pass
    logger.filters = []


def make_logger(
    outputs_dir: Path,
    verbose: bool = True,
    scenario_id: Optional[int] = None,
    console: bool = True,
) -> Tuple[logging.Logger, Optional[Path]]:
    """Create the main application logger.

    Parameters
    ----------
    outputs_dir : pathlib.Path
        Root output directory.
    verbose : bool, optional
        Whether to enable VERBOSE-level logging. Default is ``True``.
    scenario_id : int or None, optional
        Optional scenario identifier used to name the log file.
    console : bool, optional
        Whether to attach a console handler. Default is ``True``.

    Returns
    -------
    tuple[logging.Logger, pathlib.Path or None]
        Configured logger and the path to the file log.
    """
    outputs_dir = Path(outputs_dir)
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("bmp-sim")
    _reset_logger(logger)
    logger.propagate = False

    # Attach a single indent filter at logger level (applies to all handlers)
    logger.addFilter(StackIndentFilter(indent_unit="  "))

    # Default threshold is INFO; when verbose is True, lower to VERBOSE
    logger.setLevel(VERBOSE_LEVEL_NUM if verbose else logging.INFO)

    # File handler
    if scenario_id is not None:
        log_path = logs_dir / f"s{scenario_id}.txt"
    else:
        log_path = outputs_dir / "log.txt"
    logger.addHandler(_make_file_handler(log_path, verbose=verbose))

    # Console handler: attach only when console True (INFO-only)
    if console:
        logger.addHandler(_make_console_handler(verbose=verbose))

    logger.log(VERBOSE_LEVEL_NUM, "Driver logger initialized")
    return logger, log_path


def make_worker_logger(outputs_dir: Path, scenario_id: int, verbose: bool = False) -> logging.Logger:
    """Create a logger for one scenario worker.

    Parameters
    ----------
    outputs_dir : pathlib.Path
        Root output directory.
    scenario_id : int
        Scenario identifier used to name the log file.
    verbose : bool, optional
        Whether to enable VERBOSE-level logging. Default is ``False``.

    Returns
    -------
    logging.Logger
        Configured worker logger.
    """
    outputs_dir = Path(outputs_dir)
    logs_dir = outputs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"bmp-sim-s{scenario_id}")
    _reset_logger(logger)
    logger.propagate = False

    # Attach indent filter at logger level
    logger.addFilter(StackIndentFilter(indent_unit="  "))

    # Default threshold is INFO; lower to VERBOSE when verbose True
    logger.setLevel(VERBOSE_LEVEL_NUM if verbose else logging.INFO)

    log_path = logs_dir / f"s{scenario_id}.txt"
    logger.addHandler(_make_file_handler(log_path, verbose=verbose))

    # Initialization message (using VERBOSE)
    logger.log(VERBOSE_LEVEL_NUM, f"Worker logger initialized for scenario {scenario_id}")

    return logger