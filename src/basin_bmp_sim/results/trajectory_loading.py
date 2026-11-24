"""Read and validate canonical outlet trajectory results for plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union, Tuple
from collections import defaultdict
import pandas as pd
from basin_bmp_sim.common.io import read_parquet_table
from basin_bmp_sim.config.validation.selection import validate_trajectory_table


def load_trajectory_records(
    path: Union[str, Path],
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Read and normalize canonical outlet-trajectory records for plotting.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the canonical trajectory Parquet file.

    Returns
    -------
    Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
        Trajectory records keyed by pollutant, outlet, x-axis, and y-axis definitions.
    """
    df = read_parquet_table(path)
    validate_trajectory_table(df)

    for column in ("scenario", "step", "x_value", "y_value"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    out: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
    df = df.sort_values(["scenario", "pollutant", "oid", "x_axis", "y_axis", "step"])
    for _, row in df.iterrows():
        key = (str(row["pollutant"]), str(row["oid"]), str(row["x_axis"]), str(row["y_axis"]))
        out[key].append((int(row["scenario"]), float(row["x_value"]), float(row["y_value"])))
    return out
