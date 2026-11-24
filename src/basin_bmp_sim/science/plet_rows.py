"""Resolve PLET parcel-parameter overrides over NULL-key defaults.

Shared by input validation, configuration assembly and model sampling.
"""

from __future__ import annotations

from typing import List, Optional, Union
import pandas as pd

from basin_bmp_sim.domain.constants import COL_PID
from basin_bmp_sim.science.plet_classification import canonical_parameter_name


def _rows_for_pid(table: Optional[pd.DataFrame], pid: Union[int, str]) -> List[pd.Series]:
    """Resolve NULL default rows plus parcel-specific overrides for one parcel."""
    if table is None or table.empty:
        return []

    default_rows = table[table[COL_PID].isna()]
    try:
        target_pid = int(pid)
        numeric_pids = pd.to_numeric(table[COL_PID], errors="coerce")
        exact_rows = table[table[COL_PID].notna() & (numeric_pids == target_pid)]
    except (TypeError, ValueError):
        exact_rows = table[
            table[COL_PID].notna() & (table[COL_PID].astype(str) == str(pid))
        ]
    combined = pd.concat([default_rows, exact_rows], ignore_index=True)
    if combined.empty:
        return []
    combined = combined.assign(
        _canonical_parameter=combined["parameter"].map(canonical_parameter_name)
    )
    combined = combined.drop_duplicates(subset=["_canonical_parameter"], keep="last")
    return [row for _, row in combined.iterrows()]
