"""Driver-owned collection and persistence of outlet trajectory outputs."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from src.constants import DIR_OUTLET_TRAJECTORIES, FILE_ALL_SCENARIOS_PARQUET
from src.io_utils import _flatten_plot_records


def collect_and_write_trajectories(
    results: list[dict], outputs_dir: Path, logger: Any,
    *, write_parquet: Callable,
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Merge worker results in scenario order and persist the canonical table.

    ``write_parquet`` is injected so existing CLI and test harnesses can
    intercept model writes without modifying the output manager.
    """
    merged = defaultdict(list)
    for recs in results:
        for key, values in recs.items():
            merged[key].extend(values)
    trajectory_path = Path(outputs_dir) / DIR_OUTLET_TRAJECTORIES / FILE_ALL_SCENARIOS_PARQUET
    write_parquet(_flatten_plot_records(merged), trajectory_path, logger=logger)
    logger.info(f"Wrote canonical outlet trajectories parquet: {trajectory_path}")
    return merged
