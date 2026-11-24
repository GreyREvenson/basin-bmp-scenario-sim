"""Materialize PLET scientific defaults as rows in parcel input tables."""
from __future__ import annotations
from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.constants import COL_PID
from basin_bmp_sim.science.plet_defaults import _plet_parameter_defaults


def _append_parameter_defaults(
    table: pd.DataFrame,
    pollutants: Sequence[str],
) -> pd.DataFrame:
    """Append NULL-key PLET parameter defaults before scenario sampling.

    Parameters
    ----------
    table : pd.DataFrame
        Input table containing model data.
    pollutants : Sequence[str]
        Pollutant names in model order.

    Returns
    -------
    pd.DataFrame
        Table containing explicit NULL-key default rows.
    """
    defaults = _plet_parameter_defaults(pollutants)
    out = table.copy()
    if "_default_applied" not in out.columns:
        out["_default_applied"] = False
    if "value" not in out.columns:
        out["value"] = np.nan
    existing_defaults = set(
        out.loc[out[COL_PID].isna(), "parameter"].astype(str).tolist()
    )
    rows: List[Dict[str, Any]] = []
    for parameter, value in defaults.items():
        if parameter in existing_defaults:
            continue
        row = {column: np.nan for column in out.columns}
        row[COL_PID] = pd.NA
        row["parameter"] = parameter
        row["value"] = value
        row["_default_applied"] = True
        rows.append(row)
    if rows:
        out = pd.concat([out, pd.DataFrame(rows, columns=out.columns)], ignore_index=True)
    return out
