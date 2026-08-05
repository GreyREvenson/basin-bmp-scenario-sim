"""Set up logging so run progress is easy to read in files and console."""

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
    """Log a message at the custom VERBOSE level."""
    if self.isEnabledFor(VERBOSE_LEVEL_NUM):
        self.log(VERBOSE_LEVEL_NUM, msg, *args, **kwargs)


# Add the .verbose() convenience method to all Logger instances
logging.Logger.verbose = _verbose  # type: ignore[attr-defined]


# Thread-local indentation depth
_TL = threading.local()
_TL.depth = 0


def _get_depth() -> int:
    """Return current indent depth for this thread."""
    d = getattr(_TL, "depth", 0)
    try:
        return int(d)
    except Exception:
        return 0


def push_indent(n: int = 1) -> None:
    """Increase log indentation by ``n`` levels."""
    setattr(_TL, "depth", max(0, _get_depth() + int(n)))


def pop_indent(n: int = 1) -> None:
    """Decrease log indentation by ``n`` levels."""
    setattr(_TL, "depth", max(0, _get_depth() - int(n)))


@contextmanager
def log_scope(label: Optional[str] = None, logger: Optional[logging.Logger] = None, level: int = VERBOSE_LEVEL_NUM):
    """Temporarily indent logs for a block, and optionally log BEGIN/END lines."""
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
    """Add indentation text to each log record."""
    def __init__(self, indent_unit: str = "  "):
        """Set the text used for one indent level."""
        super().__init__()
        self.indent_unit = indent_unit

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach indentation text before the message is formatted."""
        depth = _get_depth()
        # Precompute indent string for the formatter
        record.indent = self.indent_unit * depth
        return True


class StackIndentFormatter(logging.Formatter):
    """Format log records while keeping any indentation text."""
    def format(self, record: logging.LogRecord) -> str:
        """Format one log line and handle missing indentation safely."""
        if not hasattr(record, "indent"):
            record.indent = ""
        return super().format(record)


def _make_console_handler(verbose: bool) -> logging.Handler:
    """Create the console logger handler (INFO level only)."""
    ch = logging.StreamHandler()
    ch.addFilter(StackIndentFilter(indent_unit="  "))
    ch.setFormatter(StackIndentFormatter("%(indent)s%(message)s"))
    ch.setLevel(logging.INFO)  # INFO-only on console
    return ch


def _make_file_handler(path: Path, verbose: bool) -> logging.Handler:
    """Create a file logger handler with indentation-aware formatting."""
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.addFilter(StackIndentFilter(indent_unit="  "))
    fh.setFormatter(StackIndentFormatter("%(asctime)s | %(levelname)s | %(indent)s%(message)s"))
    fh.setLevel(VERBOSE_LEVEL_NUM if verbose else logging.INFO)
    return fh


def _reset_logger(logger: logging.Logger) -> None:
    """Clear existing handlers so logger setup can start fresh."""
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
    """Create the main logger used by the run driver."""
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
    """Create a logger for one scenario worker process."""
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