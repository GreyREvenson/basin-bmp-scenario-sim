"""Statistical pollutant-load inputs and pathway fraction resolution."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.constants import (
    CFG_POLLUTANT_LOAD_RATE,
    CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE,
    CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW,
    CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    COL_PID,
    COL_POLLUTANT,
    COL_PATHWAY,
    GPKG_INPUT_POLLUTANT_LOAD_RATE,
)
from basin_bmp_sim.common.utils import ci_get
from basin_bmp_sim.config.validation.statistics import (
    validate_stats_rows,
    validate_unique_rows,
)
from basin_bmp_sim.config.table_sources import (
    _normalize_identifier_columns,
    _validate_explicit_references,
    _read_model_input_table,
    _normalize_pathway_label,
    _normalize_pathway_column,
    _normalize_pollutant_column,
)


def _expand_pollutant_load_rate_defaults(
    df: pd.DataFrame,
    parcel_ids: Sequence[Any],
    pollutants: Sequence[str],
) -> pd.DataFrame:
    """Expand NULL-pid load-rate defaults while preserving exact overrides.

    External GeoPackage inputs have already been normalized to integer PIDs.
    Keeping this helper tolerant of programmatic string IDs makes isolated
    model/unit-test contexts usable without weakening the file-input contract.
    """
    def pid_key(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return str(value)
        text = str(value).strip()
        try:
            numeric = float(text)
            if np.isfinite(numeric) and numeric.is_integer():
                return int(numeric)
        except ValueError:
            pass
        return text

    out = df.copy()
    valid_pids = {pid_key(pid) for pid in parcel_ids}
    normalized_pid = out[COL_PID].map(pid_key)
    out = out[
        (out[COL_PID].isna() | normalized_pid.isin(valid_pids))
        & out[COL_POLLUTANT].isin(list(pollutants))
    ].copy()
    normalized_pid = out[COL_PID].map(pid_key)
    if out.empty or not out[COL_PID].isna().any():
        out = out[out[COL_PID].notna()].copy()
        out[COL_PID] = out[COL_PID].map(pid_key)
        return out.reset_index(drop=True)

    explicit = COL_PATHWAY in out.columns
    keys = [COL_PID, COL_POLLUTANT] + ([COL_PATHWAY] if explicit else [])
    validate_unique_rows(out, keys, CFG_POLLUTANT_LOAD_RATE)
    pathways: List[Optional[str]] = (
        list(dict.fromkeys(out[COL_PATHWAY].astype(str).tolist()))
        if explicit else [None]
    )

    defaults: Dict[Tuple[str, Optional[str]], pd.Series] = {}
    exact: Dict[Tuple[Any, str, Optional[str]], pd.Series] = {}
    for _, row in out.iterrows():
        path = str(row[COL_PATHWAY]) if explicit else None
        pollutant = str(row[COL_POLLUTANT])
        key = pid_key(row[COL_PID])
        if key is None:
            defaults[(pollutant, path)] = row
        else:
            exact[(key, pollutant, path)] = row

    expanded: List[Dict[str, Any]] = []
    for raw_pid in parcel_ids:
        pid = pid_key(raw_pid)
        for pollutant in map(str, pollutants):
            for path in pathways:
                row = exact.get((pid, pollutant, path))
                if row is None:
                    row = defaults.get((pollutant, path))
                if row is None:
                    continue
                copied = row.to_dict()
                copied[COL_PID] = pid
                expanded.append(copied)
    if not expanded:
        return out.iloc[0:0].copy()
    return pd.DataFrame(expanded).reset_index(drop=True)


def _load_pollutant_load_rate(
    cfg: Dict[str, Any],
    parcels: pd.DataFrame,
    pollutants: List[str],
    logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Load statistical parcel pollutant load rates from ``parcels.gpkg``."""
    df = _read_model_input_table(
        cfg, GPKG_INPUT_POLLUTANT_LOAD_RATE, [COL_PID, COL_POLLUTANT], logger,
        required=True, input_tables=input_tables,
    )
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_POLLUTANT_LOAD_RATE, logger)
    df = _normalize_pathway_column(df, CFG_POLLUTANT_LOAD_RATE, logger)
    validate_stats_rows(df, CFG_POLLUTANT_LOAD_RATE)
    df = _normalize_identifier_columns(df, [COL_PID], CFG_POLLUTANT_LOAD_RATE, allow_null_pid=True)
    parcel_ids = [int(pid) for pid in parcels[COL_PID].tolist()]
    _validate_explicit_references(
        df[COL_PID].tolist(), parcel_ids, label=CFG_POLLUTANT_LOAD_RATE, allow_default=True
    )
    df = _expand_pollutant_load_rate_defaults(df, parcel_ids, pollutants)
    if df.empty:
        raise ValueError(f"{GPKG_INPUT_POLLUTANT_LOAD_RATE} has no records for specified parcels+pollutants")
    validate_stats_rows(df, CFG_POLLUTANT_LOAD_RATE)
    return df


def _resolve_aggregate_pathway_fractions(
    cfg: Dict[str, Any], load_generation: Dict[str, Any], pathways: Sequence[str]
) -> Dict[str, float]:
    """Resolve fractions used to split one sampled aggregate parcel load rate.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Normalized model configuration mapping.
    load_generation : Dict[str, Any]
        Load-generation configuration mapping.
    pathways : Sequence[str]
        Pollutant transport pathway names.

    Returns
    -------
    Dict[str, float]
        Normalized pathway fractions keyed by pathway name.

    Raises
    ------
    ValueError
        If pathway fractions are malformed, reference unknown pathways, fall
        outside ``[0, 1]``, or do not sum to one.
    """
    pathways = list(pathways)
    if len(pathways) == 1:
        return {pathways[0]: 1.0}
    raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS)
    if raw is None:
        raw = load_generation.get(CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS)
    fractions: Dict[str, float] = {}
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError(f"{CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS} must be a mapping")
        fractions = {_normalize_pathway_label(k): float(v) for k, v in raw.items()}
    else:
        surf_raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE)
        shallow_raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW)
        if surf_raw is not None:
            fractions["surface"] = float(surf_raw)
        if shallow_raw is not None:
            fractions["shallow subsurface"] = float(shallow_raw)
        if shallow_raw is not None and surf_raw is None and "surface" in pathways:
            fractions["surface"] = 1.0 - float(shallow_raw)
        elif surf_raw is not None and shallow_raw is not None and "deep subsurface" in pathways:
            fractions["deep subsurface"] = 1.0 - float(surf_raw) - float(shallow_raw)
    unknown = set(fractions) - set(pathways)
    if unknown:
        raise ValueError(
            f"Pathway fractions refer to pathways not defined by bmp_efficiency: {sorted(unknown)}"
        )
    if not fractions:
        raise ValueError(
            "Statistical mode uses one aggregate pollutant_load_rate per parcel but multiple BMP pathways. "
            f"Define {CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS}, e.g. {{'shallow subsurface': 0.2, 'surface': 0.8}}."
        )
    for path in pathways:
        fractions.setdefault(path, 0.0)
    vals = np.asarray(list(fractions.values()), dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("pollutant load rate pathway fractions must be finite")
    if (vals < 0.0).any() or (vals > 1.0).any():
        raise ValueError("pollutant load rate pathway fractions must each be in [0,1]")
    total = float(vals.sum())
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError(f"pollutant load rate pathway fractions must sum to 1.0; got {total:.12g}")
    return {path: float(fractions[path]) for path in pathways}
