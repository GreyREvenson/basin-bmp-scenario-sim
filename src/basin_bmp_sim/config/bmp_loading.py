"""BMP selection, effectiveness, and component-cost input tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.constants import (
    CFG_BMP_COST,
    CFG_BMP_EFFICIENCY,
    COL_CPS,
    COL_COST_COMPONENT,
    COL_POLLUTANT,
    COL_PROBABILITY,
    COL_UNIT,
    COL_PATHWAY,
    PATHWAY_VALUES,
    PLET_PATHWAY_VALUES,
)
from basin_bmp_sim.common.io import read_csv_table
from basin_bmp_sim.common.utils import ci_get, normalize_columns
from basin_bmp_sim.config.validation.statistics import (
    require_columns,
    validate_stats_rows,
    validate_unique_rows,
)
from basin_bmp_sim.config.validation.selection import validate_bmp_selection_table
from basin_bmp_sim.config.table_sources import (
    _normalize_cps_column,
    _merge_csvs,
    _normalize_pathway_column,
    _normalize_pollutant_column,
)


def load_bmp_selection_probabilities(
    path: Union[str, Path],
    cps: Sequence[int],
    logger: Any,
) -> pd.DataFrame:
    """Read and normalize explicit BMP-selection probabilities.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the BMP-selection probability CSV file.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        Normalized BMP-selection probability table.
    """
    logger.verbose(f"Reading BMP selection probabilities from {path}")
    df = normalize_columns(read_csv_table(path))

    if COL_PROBABILITY not in df.columns:
        alias = next((name for name in ("pr", "p") if name in df.columns), None)
        if alias is not None:
            df[COL_PROBABILITY] = df[alias]

    if COL_CPS in df.columns:
        cps_numeric = pd.to_numeric(df[COL_CPS], errors="coerce")
        finite_integer = np.isfinite(cps_numeric) & (cps_numeric % 1 == 0)
        df.loc[finite_integer, COL_CPS] = cps_numeric.loc[finite_integer].astype(int)

    if COL_PROBABILITY in df.columns:
        df[COL_PROBABILITY] = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")

    configured_cps = {int(value) for value in cps}
    if COL_CPS in df.columns:
        df = df[df[COL_CPS].isin(configured_cps)].copy()

    validate_bmp_selection_table(df, cps)
    df[COL_CPS] = df[COL_CPS].astype(int)
    df[COL_PROBABILITY] = df[COL_PROBABILITY].astype(float)
    df[COL_PROBABILITY] = df[COL_PROBABILITY] / float(df[COL_PROBABILITY].sum())

    result = df[[COL_CPS, COL_PROBABILITY]].reset_index(drop=True)
    logger.verbose(
        f"Loaded explicit BMP selection probabilities from {path}: "
        f"{result.to_dict(orient='records')}"
    )
    return result


def _efficiency_stat_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that can define an efficiency distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        BMP efficiency input table.

    Returns
    -------
    list[str]
        Statistic columns present in the table.
    """
    named_stats = {
        "value",
        "mean",
        "average",
        "avg",
        "sd",
        "std",
        "min",
        "minimum",
        "max",
        "maximum",
        "p0",
        "p100",
    }
    return [
        col
        for col in df.columns
        if str(col).lower() in named_stats
        or (
            str(col).lower().startswith("p")
            and str(col).lower()[1:].isdigit()
        )
    ]


def _plet_numeric_spec_columns(df: pd.DataFrame) -> List[str]:
    """Return standardized numeric-specification columns present in a BMP-efficiency table.

    Parameters
    ----------
    df : pandas.DataFrame
        BMP-efficiency input table.

    Returns
    -------
    list[str]
        Numeric-specification columns present in the table.
    """
    candidates = [
        "value",
        "mean",
        "sd",
        "min",
        "p05",
        "p50",
        "p95",
        "max",
    ]
    return [col for col in candidates if col in df.columns]


def _complete_bmp_efficiency_coverage(
    df: pd.DataFrame,
    cps: Sequence[int],
    pollutants: Sequence[str],
    logger: Any,
) -> pd.DataFrame:
    """Legacy three-path completion used by the public loader API.

    Surface is required. Missing shallow/deep subsurface values are completed
    as fixed zero distributions with verbose logging. Production mode-specific
    validation bypasses this compatibility layer.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : Sequence[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        BMP efficiency table with complete legacy pathway coverage.

    Raises
    ------
    ValueError
        If required surface-efficiency coverage is missing for a configured CPS/pollutant combination.
    """
    completed = df.copy()
    completed[COL_CPS] = completed[COL_CPS].astype(int)
    if COL_PATHWAY not in completed.columns:
        completed[COL_PATHWAY] = PATHWAY_VALUES[0]
        logger.verbose(
            "bmp_efficiency has no pathway column; treating supplied "
            "CPS x pollutant efficiencies as surface efficiencies"
        )

    configured_cps = [int(cps_code) for cps_code in cps]
    configured_pollutants = [str(pollutant) for pollutant in pollutants]
    surface_pathway = PATHWAY_VALUES[0]
    subsurface_pathways = PATHWAY_VALUES[1:]
    missing_surface: List[Tuple[int, str]] = []
    for cps_code in configured_cps:
        for pollutant in configured_pollutants:
            mask = (
                (completed[COL_CPS] == cps_code)
                & (completed[COL_POLLUTANT] == pollutant)
                & (completed[COL_PATHWAY] == surface_pathway)
            )
            if not mask.any():
                missing_surface.append((cps_code, pollutant))
    if missing_surface:
        details = ", ".join(
            f"cps={cps_code}, pollutant={pollutant}"
            for cps_code, pollutant in missing_surface
        )
        raise ValueError(
            "bmp_efficiency is missing required surface efficiency coverage "
            f"for configured CPS x pollutant combinations: {details}"
        )

    stat_columns = _efficiency_stat_columns(completed)
    added_rows: List[pd.Series] = []
    for cps_code in configured_cps:
        for pollutant in configured_pollutants:
            pair_mask = (
                (completed[COL_CPS] == cps_code)
                & (completed[COL_POLLUTANT] == pollutant)
            )
            surface_row = completed[
                pair_mask & (completed[COL_PATHWAY] == surface_pathway)
            ].iloc[0]
            for pathway in subsurface_pathways:
                pathway_mask = pair_mask & (completed[COL_PATHWAY] == pathway)
                if pathway_mask.any():
                    row_index = completed[pathway_mask].index[0]
                    has_values = any(
                        not pd.isna(completed.at[row_index, column])
                        for column in stat_columns
                    )
                    if has_values:
                        continue
                    for column in stat_columns:
                        completed.at[row_index, column] = 0.0
                else:
                    default_row = surface_row.copy()
                    default_row[COL_PATHWAY] = pathway
                    for column in stat_columns:
                        default_row[column] = 0.0
                    added_rows.append(default_row)
                logger.verbose(
                    "No bmp_efficiency value specified for "
                    f"cps={cps_code}, pollutant={pollutant}, pathway='{pathway}'; "
                    "assuming efficiency=0"
                )

    if added_rows:
        completed = pd.concat(
            [completed, pd.DataFrame(added_rows, columns=completed.columns)],
            ignore_index=True,
        )

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)
    pathway_order = {pathway: idx for idx, pathway in enumerate(PATHWAY_VALUES)}
    completed["_pathway_order"] = completed[COL_PATHWAY].map(pathway_order)
    completed = completed.sort_values(
        [COL_CPS, COL_POLLUTANT, "_pathway_order"], kind="stable"
    ).drop(columns="_pathway_order")
    return completed.reset_index(drop=True)


def _complete_plet_bmp_efficiency_coverage(
    df: pd.DataFrame, cps: Sequence[int], pollutants: Sequence[str], logger: Any
) -> pd.DataFrame:
    """Require PLET surface efficiencies and default missing subsurface to zero.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : Sequence[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        BMP efficiency table with required PLET pathway coverage.

    Raises
    ------
    ValueError
        If PLET/RUSLE surface-efficiency coverage is missing for a configured CPS/pollutant combination.
    """
    completed = df.copy()
    completed[COL_CPS] = completed[COL_CPS].astype(int)

    if COL_PATHWAY not in completed.columns:
        completed[COL_PATHWAY] = "surface"
        logger.verbose(
            "plet_rusle bmp_efficiency has no pathway column; treating supplied "
            "CPS x pollutant rows as surface efficiencies"
        )
    else:
        completed[COL_PATHWAY] = completed[COL_PATHWAY].astype(str).str.strip().str.lower()
        unexpected = sorted(
            set(completed[COL_PATHWAY].astype(str)) - set(PLET_PATHWAY_VALUES)
        )
        if unexpected:
            logger.warning(
                "plet_rusle recognizes only pathway labels 'surface' and "
                f"'subsurface'. Ignoring unexpected bmp_efficiency pathway labels: {unexpected}. "
                "If no correctly labeled subsurface efficiency remains for a CPS/pollutant, "
                "subsurface efficiency will be assumed to be 0."
            )
        completed = completed[completed[COL_PATHWAY].isin(PLET_PATHWAY_VALUES)].copy()

    validate_unique_rows(
        completed, [COL_CPS, COL_POLLUTANT, COL_PATHWAY], CFG_BMP_EFFICIENCY
    )

    numeric_cols = _plet_numeric_spec_columns(completed)
    added_rows: List[pd.Series] = []
    missing_surface: List[Tuple[int, str]] = []

    for cps_code in [int(x) for x in cps]:
        for pollutant in [str(x) for x in pollutants]:
            pair = (completed[COL_CPS] == cps_code) & (completed[COL_POLLUTANT] == pollutant)
            surf = completed[pair & (completed[COL_PATHWAY] == "surface")]
            if surf.empty:
                missing_surface.append((cps_code, pollutant))
                continue

            sub = completed[pair & (completed[COL_PATHWAY] == "subsurface")]
            sub_has_stats = False
            if not sub.empty:
                row = sub.iloc[0]
                sub_has_stats = any(not pd.isna(row.get(c)) for c in numeric_cols)

            if not sub_has_stats:
                template = surf.iloc[0].copy()
                template[COL_PATHWAY] = "subsurface"
                for c in numeric_cols:
                    template[c] = pd.NA
                if "value" in numeric_cols:
                    template["value"] = 0.0
                if COL_UNIT in template.index:
                    template[COL_UNIT] = "fraction"
                if "notes" in template.index:
                    prior = template["notes"]
                    auto_note = (
                        "Auto-added in plet_rusle: missing subsurface efficiency defaults to 0"
                    )
                    template["notes"] = (
                        auto_note
                        if pd.isna(prior) or str(prior).strip() == ""
                        else f"{prior}; {auto_note}"
                    )
                if not sub.empty:
                    completed = completed.drop(index=sub.index)
                added_rows.append(template)
                logger.warning(
                    "plet_rusle: no correctly labeled subsurface BMP efficiency was "
                    f"defined for cps={cps_code}, pollutant={pollutant}; assuming efficiency=0"
                )

    if missing_surface:
        details = ", ".join(f"cps={c}, pollutant={p}" for c, p in missing_surface)
        raise ValueError(
            "plet_rusle requires a surface bmp_efficiency for every configured "
            f"CPS x pollutant combination; missing: {details}"
        )

    if added_rows:
        completed = pd.concat([completed, pd.DataFrame(added_rows, columns=completed.columns)], ignore_index=True)

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)

    order = {"surface": 0, "subsurface": 1}
    completed["_pathway_order"] = completed[COL_PATHWAY].map(order)
    return completed.sort_values(
        [COL_CPS, COL_POLLUTANT, "_pathway_order"], kind="stable"
    ).drop(columns="_pathway_order").reset_index(drop=True)


def _load_bmp_efficiency(
    cfg: Dict[str, Any],
    cps: List[int],
    pollutants: List[str],
    logger: Any,
    *,
    complete_legacy: bool = True,
) -> pd.DataFrame:
    """Load and normalize BMP effectiveness inputs.

    ``complete_legacy=True`` preserves the public three-path loader behavior
    used by existing callers/tests. The main model loader passes ``False`` and
    then performs mode-specific validation for PLET/RUSLE or statistical mode.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Normalized model configuration mapping.
    cps : List[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : List[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.
    complete_legacy : bool
        Whether to complete legacy three-pathway BMP efficiency coverage.

    Returns
    -------
    pd.DataFrame
        Normalized BMP efficiency table.

    Raises
    ------
    ValueError
        If no BMP-efficiency records remain for the configured CPS codes and pollutants.
    """
    df = _merge_csvs(ci_get(cfg, CFG_BMP_EFFICIENCY), [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    df = _normalize_cps_column(df, CFG_BMP_EFFICIENCY)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pathway_column(df, CFG_BMP_EFFICIENCY, logger)
    df = df[df[COL_CPS].isin(cps) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")

    if complete_legacy:
        return _complete_bmp_efficiency_coverage(df, cps, pollutants, logger)
    return df


def _load_bmp_cost(cfg: Dict[str, Any], cps: List[int], logger: Any) -> Optional[pd.DataFrame]:
    """Optionally load BMP cost inputs.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    cps : list[int]
        BMP CPS codes to retain.
    logger : Any
        Logger used for progress and warning messages.

    Returns
    -------
    pandas.DataFrame or None
        BMP cost table filtered to the requested BMPs, or ``None`` when no
        usable cost table is configured.
    """
    path = ci_get(cfg, CFG_BMP_COST)
    if path is None:
        return None
    # BMP cost tables allow more than one row per CPS. Each row represents an
    # additive cost component (for example seed + establishment + termination).
    # Legacy one-row-per-CPS files remain valid: when ``cost_component`` is
    # absent, the loader assigns the component name ``total``.
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    frames: List[pd.DataFrame] = []
    for p in paths:
        logger.verbose(f"Reading {CFG_BMP_COST} from {p}")
        frame = normalize_columns(read_csv_table(p))
        require_columns(frame, [COL_CPS, COL_UNIT], f"{CFG_BMP_COST} ({p})", logger)
        if COL_COST_COMPONENT not in frame.columns:
            frame[COL_COST_COMPONENT] = "total"
        else:
            component = frame[COL_COST_COMPONENT].astype("string").str.strip()
            frame[COL_COST_COMPONENT] = component.mask(component.isna() | (component == ""), "total")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True).drop_duplicates(keep="first")
    logical_key = [COL_CPS, COL_COST_COMPONENT]
    dup = df.duplicated(subset=logical_key, keep=False)
    if dup.any():
        preview = df.loc[dup, logical_key].head(10).to_dict(orient="records")
        raise ValueError(
            f"{CFG_BMP_COST} contains conflicting duplicate rows for logical key "
            f"{logical_key}: {preview}"
        )
    df = df.reset_index(drop=True)
    df = _normalize_cps_column(df, CFG_BMP_COST)
    validate_stats_rows(df, CFG_BMP_COST)
    df = df[df[COL_CPS].isin(cps)].copy()
    if df.empty:
        logger.warning("bmp_cost has no records for specified cps; proceeding without costing")
        return None
    return df
