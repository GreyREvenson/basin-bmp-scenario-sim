"""Selection for the configuration."""

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.constants import COL_CPS, COL_PROBABILITY



def validate_bmp_selection_table(df: pd.DataFrame, cps: Sequence[int]) -> None:
    """Validate a normalized explicit BMP-selection weight table.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.
        cps : Sequence[int]
            Conservation Practice Standard (CPS) code or codes.

        Raises
        ------
        ValueError
            If CPS identifiers or selection weights are invalid, duplicated, missing, or do not define positive total weight.

    """
    required = {COL_CPS, COL_PROBABILITY}
    if not required.issubset(df.columns):
        raise ValueError(
            f"bmp selection file must contain {sorted(required)} or an accepted selection-weight alias"
        )
    cps_numeric = pd.to_numeric(df[COL_CPS], errors="coerce")
    invalid_cps = (~np.isfinite(cps_numeric)) | (cps_numeric % 1 != 0)
    if invalid_cps.any():
        bad_rows = df.loc[invalid_cps, [COL_CPS, COL_PROBABILITY]].head(5).to_dict(orient="records")
        raise ValueError(
            "bmp selection cps values must be finite integers; "
            f"example bad rows: {bad_rows}"
        )
    configured_cps = {int(value) for value in cps}
    filtered_cps = cps_numeric.astype(int)
    duplicated = filtered_cps.duplicated(keep=False)
    if duplicated.any():
        duplicate_cps = sorted(filtered_cps.loc[duplicated].unique().tolist())
        raise ValueError(
            "bmp selection file contains duplicate selection-weight rows for cps values: "
            f"{duplicate_cps}"
        )
    probs = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")
    invalid = (~np.isfinite(probs)) | (probs < 0.0)
    if invalid.any():
        bad_rows = df.loc[invalid, [COL_CPS, COL_PROBABILITY]].head(5).to_dict(orient="records")
        raise ValueError(
            "bmp selection weights must be finite and nonnegative; "
            f"example bad rows: {bad_rows}"
        )
    found_cps = set(filtered_cps.tolist())
    missing = sorted(configured_cps - found_cps)
    if missing:
        raise ValueError(f"bmp selection file is missing probability rows for cps values: {missing}")
    if float(probs.sum()) <= 0.0:
        raise ValueError("bmp_sel selection weights sum to zero or negative")


def validate_trajectory_table(df: pd.DataFrame) -> None:
    """Validate the canonical outlet-trajectory table schema and required values.

        Parameters
        ----------
        df : pd.DataFrame
            Input table to process.

        Raises
        ------
        ValueError
            If required trajectory columns are missing, blank, or contain non-finite numeric values.

    """
    required = {
        "scenario", "pollutant", "oid", "x_axis", "y_axis", "step", "x_value", "y_value"
    }
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Canonical trajectory table is missing required columns: {missing}")
    for column in ("scenario", "step", "x_value", "y_value"):
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(numeric)
        if invalid.any():
            bad_rows = df.loc[invalid].head(5).to_dict(orient="records")
            raise ValueError(
                f"Canonical trajectory table column {column!r} contains non-finite values; "
                f"example bad rows: {bad_rows}"
            )
    for column in ("pollutant", "oid", "x_axis", "y_axis"):
        blank = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if blank.any():
            raise ValueError(
                f"Canonical trajectory table contains blank values in required column {column!r}"
            )
