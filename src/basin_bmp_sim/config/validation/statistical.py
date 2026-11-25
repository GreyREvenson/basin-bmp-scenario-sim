"""Statistical for the configuration."""

from __future__ import annotations
from typing import List, Sequence, Tuple
import pandas as pd
from basin_bmp_sim.domain.constants import (
    CFG_BMP_EFFICIENCY,
    CFG_POLLUTANT_LOAD_RATE,
    COL_CPS,
    COL_PATHWAY,
    COL_PID,
    COL_POLLUTANT,
)
from basin_bmp_sim.domain.statistics_validation import validate_stats_rows, validate_unique_rows


def validate_statistical_efficiency_coverage(
    df: pd.DataFrame, cps: Sequence[int], pollutants: Sequence[str], pathways: Sequence[str]
) -> pd.DataFrame:
    """Require complete CPS x pollutant x pathway coverage in statistical mode.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        cps : Sequence[int]
            Conservation Practice Standard (CPS) code or codes.
        pollutants : Sequence[str]
            Pollutant names in model order.
        pathways : Sequence[str]
            Pollutant transport pathway names.

        Returns
        -------
        pd.DataFrame
            Validated BMP-efficiency table with complete pathway coverage.

        Raises
        ------
        ValueError
            If statistical-mode BMP efficiencies do not provide complete CPS/pollutant/pathway coverage.

    """
    out = df.copy()
    if COL_PATHWAY not in out.columns:
        if list(pathways) != ["surface"]:
            raise ValueError(
                "statistical mode with multiple pathways requires a pathway column in bmp_efficiency"
            )
        out[COL_PATHWAY] = "surface"
    supplied = set(out[COL_PATHWAY].astype(str))
    expected = set(pathways)
    if supplied != expected:
        raise ValueError(
            "statistical mode requires pollutant_load_rate and bmp_efficiency to use the same pathways; "
            f"expected {sorted(expected)}, found {sorted(supplied)} in bmp_efficiency"
        )
    validate_unique_rows(out, [COL_CPS, COL_POLLUTANT, COL_PATHWAY], CFG_BMP_EFFICIENCY)
    missing = []
    for cps_code in [int(value) for value in cps]:
        for pollutant in [str(value) for value in pollutants]:
            for pathway in pathways:
                mask = (
                    (out[COL_CPS].astype(int) == cps_code)
                    & (out[COL_POLLUTANT] == pollutant)
                    & (out[COL_PATHWAY] == pathway)
                )
                if not mask.any():
                    missing.append((cps_code, pollutant, pathway))
    if missing:
        preview = ", ".join(
            f"cps={cps_code}, pollutant={pollutant}, pathway={pathway}"
            for cps_code, pollutant, pathway in missing[:20]
        )
        raise ValueError(f"bmp_efficiency is missing required statistical-mode coverage: {preview}")
    validate_stats_rows(out, CFG_BMP_EFFICIENCY)
    return out.reset_index(drop=True)


def validate_statistical_load_rates(
    df: pd.DataFrame,
    parcels: pd.DataFrame,
    pollutants: Sequence[str],
) -> Tuple[List[str], bool]:
    """Validate statistical parcel-load-rate coverage and return pathways/mode.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        parcels : pd.DataFrame
            Parcel table or geospatial parcel dataset.
        pollutants : Sequence[str]
            Pollutant names in model order.

        Returns
        -------
        Tuple[List[str], bool]
            Detected pathway names and whether load rates are supplied in aggregate form.

        Raises
        ------
        ValueError
            If statistical pollutant load rates do not provide complete parcel/pollutant/pathway coverage.

    """
    parcel_ids = parcels[COL_PID].tolist()
    explicit = COL_PATHWAY in df.columns
    pathways = list(dict.fromkeys(df[COL_PATHWAY].astype(str).tolist())) if explicit else []
    keys = [COL_PID, COL_POLLUTANT] + ([COL_PATHWAY] if explicit else [])
    validate_unique_rows(df, keys, CFG_POLLUTANT_LOAD_RATE)
    missing = []
    if explicit:
        for pid in parcel_ids:
            for pollutant in pollutants:
                for pathway in pathways:
                    mask = (
                        (df[COL_PID] == pid)
                        & (df[COL_POLLUTANT] == pollutant)
                        & (df[COL_PATHWAY] == pathway)
                    )
                    if not mask.any():
                        missing.append((pid, pollutant, pathway))
    else:
        for pid in parcel_ids:
            for pollutant in pollutants:
                mask = (df[COL_PID] == pid) & (df[COL_POLLUTANT] == pollutant)
                if not mask.any():
                    missing.append((pid, pollutant, None))
    if missing:
        preview = ", ".join(
            f"pid={pid}, pollutant={pollutant}" + (f", pathway={pathway}" if pathway else "")
            for pid, pollutant, pathway in missing[:20]
        )
        raise ValueError(f"pollutant_load_rate is missing required statistical-mode coverage: {preview}")
    return pathways, not explicit
