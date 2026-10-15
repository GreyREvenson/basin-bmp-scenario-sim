from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import CFG_BMP_EFFICIENCY
from src.input_config import _complete_plet_bmp_efficiency_coverage
from src.input_validation import validate_stats_rows


class _ListLogger:
    """
    Collect warning and verbose messages emitted during validation-oriented tests.
    """

    def __init__(self) -> None:
        """
        Initialize empty in-memory message stores.
        """
        self.warning_messages: list[str] = []
        self.verbose_messages: list[str] = []

    def warning(self, msg, *args) -> None:
        """
        Record a formatted warning message.

        Parameters
        ----------
        msg : str
            Format string or message body.
        *args
            Positional arguments interpolated into ``msg`` using ``%`` formatting.
        """
        if args:
            msg = msg % args
        self.warning_messages.append(str(msg))

    def verbose(self, msg, *args) -> None:
        """
        Record a formatted verbose message.

        Parameters
        ----------
        msg : str
            Format string or message body.
        *args
            Positional arguments interpolated into ``msg`` using ``%`` formatting.
        """
        if args:
            msg = msg % args
        self.verbose_messages.append(str(msg))

    def text(self) -> str:
        """
        Return all collected messages as a single lowercase string.

        Returns
        -------
        str
            Concatenated message text.
        """
        return "\n".join(self.warning_messages + self.verbose_messages).lower()


def _base_efficiency_table() -> pd.DataFrame:
    """
    Construct a valid canonical PLET BMP-efficiency fixture table.

    Returns
    -------
    pandas.DataFrame
        Fixture table covering one CPS over three pollutants and the two
        canonical PLET pathways.
    """
    return pd.DataFrame(
        [
            {
                "cps": np.int64(340),
                "name": "Cover Crop",
                "pollutant": "TN",
                "pathway": "surface",
                "value": pd.NA,
                "mean": np.float64(0.35),
                "sd": pd.NA,
                "min": np.float64(0.15),
                "p05": pd.NA,
                "p50": pd.NA,
                "p95": pd.NA,
                "max": np.float64(0.55),
                "unit": "fraction",
                "notes": pd.NA,
            },
            {
                "cps": np.int64(340),
                "name": "Cover Crop",
                "pollutant": "TP",
                "pathway": "surface",
                "value": pd.NA,
                "mean": np.float64(0.40),
                "sd": pd.NA,
                "min": np.float64(0.20),
                "p05": pd.NA,
                "p50": pd.NA,
                "p95": pd.NA,
                "max": np.float64(0.65),
                "unit": "fraction",
                "notes": pd.NA,
            },
            {
                "cps": np.int64(340),
                "name": "Cover Crop",
                "pollutant": "TSS",
                "pathway": "surface",
                "value": pd.NA,
                "mean": np.float64(0.60),
                "sd": pd.NA,
                "min": np.float64(0.40),
                "p05": pd.NA,
                "p50": pd.NA,
                "p95": pd.NA,
                "max": np.float64(0.85),
                "unit": "fraction",
                "notes": pd.NA,
            },
        ]
    )


def test_plet_missing_subsurface_rows_are_completed_as_valid_fixed_zero_rows() -> None:
    """
    Verify that missing PLET subsurface rows are synthesized as valid fixed-zero rows.
    """
    logger = _ListLogger()
    df = _base_efficiency_table()

    completed = _complete_plet_bmp_efficiency_coverage(
        df,
        cps=[np.int64(340)],
        pollutants=["TN", "TP", "TSS"],
        logger=logger,
    )

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)

    subsurface = completed.loc[completed["pathway"].str.lower() == "subsurface"].copy()
    assert len(subsurface) == 3
    assert set(subsurface["pollutant"].tolist()) == {"TN", "TP", "TSS"}
    np.testing.assert_allclose(subsurface["value"].astype(float).to_numpy(), np.zeros(3, dtype=float))

    for col in ["mean", "sd", "min", "p05", "p50", "p95", "max"]:
        assert subsurface[col].isna().all()

    log_text = logger.text()
    assert "assuming efficiency=0" in log_text
    assert "subsurface" in log_text


@pytest.mark.parametrize("bad_label", ["groundwater", "shallow subsurface", "deep subsurface"])
def test_plet_unexpected_pathway_labels_are_ignored_and_zero_subsurface_is_added(bad_label: str) -> None:
    """
    Verify that noncanonical pathway labels are ignored and replaced by zero subsurface rows.
    """
    logger = _ListLogger()
    df = _base_efficiency_table()

    bogus = df.copy()
    bogus["pathway"] = bad_label
    mutated = pd.concat([df, bogus], ignore_index=True)

    completed = _complete_plet_bmp_efficiency_coverage(
        mutated,
        cps=[np.int64(340)],
        pollutants=["TN", "TP", "TSS"],
        logger=logger,
    )

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)

    observed_paths = set(completed["pathway"].astype(str).str.lower().tolist())
    assert observed_paths == {"surface", "subsurface"}

    subsurface_rows = completed.loc[completed["pathway"].str.lower() == "subsurface"].copy()
    assert len(subsurface_rows) == 3
    np.testing.assert_allclose(subsurface_rows["value"].astype(float).to_numpy(), np.zeros(3, dtype=float))

    log_text = logger.text()
    assert "ignoring unexpected bmp_efficiency pathway labels" in log_text or "ignoring unexpected" in log_text
    assert bad_label.lower() in log_text


def test_plet_missing_surface_row_is_error() -> None:
    """
    Verify that every CPS-pollutant pair still requires a canonical surface row.
    """
    logger = _ListLogger()
    df = _base_efficiency_table()
    df = df[df["pollutant"] != "TN"].copy()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "cps": np.int64(340),
                        "name": "Cover Crop",
                        "pollutant": "TN",
                        "pathway": "subsurface",
                        "value": np.float64(0.0),
                                "mean": pd.NA,
                        "sd": pd.NA,
                        "min": pd.NA,
                        "p05": pd.NA,
                        "p50": pd.NA,
                        "p95": pd.NA,
                        "max": pd.NA,
                        "unit": "fraction",
                        "notes": pd.NA,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="plet_rusle requires a surface bmp_efficiency for every configured CPS x pollutant combination",
    ):
        _complete_plet_bmp_efficiency_coverage(
            df,
            cps=[np.int64(340)],
            pollutants=["TN", "TP", "TSS"],
            logger=logger,
        )