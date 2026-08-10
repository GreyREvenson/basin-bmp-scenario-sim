"""Parcel lookup and routing helpers.

This module provides small helper functions used to sample parcel-level loads,
select parcels for BMP placement, and resolve routing and delivery metadata.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

from .constants import (
    COL_AREA_HA,
    COL_CPS,
    COL_PERIM_M,
    COL_PID,
    DATA_PARCELS,
    DATA_PARCEL_UP_MAP,
)


def _sample_yield(self: "Model", parcel_idx: int, pol_idx: int) -> float:
    """Sample a baseline pollutant yield for one parcel and pollutant.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel in the model arrays.
    pol_idx : int
        Index of the pollutant in ``self.pollutants``.

    Returns
    -------
    float
        Sampled pollutant yield for the requested parcel/pollutant pair.

    Raises
    ------
    KeyError
        If no yield statistics exist for the requested parcel and pollutant.
    """
    stats = self.pollutant_yield_stats[parcel_idx][pol_idx]
    if stats is None:
        raise KeyError(
            f"No pollutant yield stats found for pid={self.parcel_ids[parcel_idx]}, pollutant={self.pollutants[pol_idx]}"
        )
    return self._sample_from_stats(stats, kind="yield")


def _sample_parcel_index(self: "Model") -> int:
    """Sample the next parcel to receive a BMP attempt.

    Parameters
    ----------
    self : Model
        Active simulation model instance.

    Returns
    -------
    int
        Index into ``self.parcel_selection_ids`` for the selected parcel.
    """
    idx = self.rng.choice(len(self.parcel_selection_ids), p=self.parcel_selection_probs)
    self.logger.verbose(f"selected parcel idx={idx} with pid={self.parcel_selection_ids[idx]}")
    return idx


def _get_parcel_metadata(self: "Model", pid: Union[int, str]) -> pd.Series:
    """Return metadata for one parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    pid : int or str
        Parcel identifier.

    Returns
    -------
    pandas.Series
        Row of parcel metadata for the requested parcel.

    Raises
    ------
    KeyError
        If the parcel ID is not present after clipping to the domain.
    """
    sub = self.data[DATA_PARCELS]
    match = sub[sub[COL_PID].astype(str) == str(pid)]
    if match.empty:
        raise KeyError(
            f"Selected pid {pid} not found in parcels after clipping. "
            f"Ensure parcel_p PIDs exist in parcels and are within the domain."
        )
    return match.iloc[0]


def _get_parcel_up_list(self: "Model", pid: Union[int, str]) -> List[str]:
    """Return upstream parcel IDs for one parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    pid : int or str
        Parcel identifier.

    Returns
    -------
    list[str]
        Parcel IDs that drain into the requested parcel.
    """
    return list(self.data[DATA_PARCEL_UP_MAP].get(str(pid), []))


def _get_parcel_out_oids(self: "Model", parcel_idx: int) -> List[str]:
    """Return outlet IDs connected to one parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Parcel index.

    Returns
    -------
    list[str]
        Outlet IDs connected to the selected parcel.
    """
    return list(self.parcel_out_oids[parcel_idx])


def _get_delivery_coeffs(self: "Model", pid: Union[int, str], oid: Union[int, str]) -> Dict[str, float]:
    """Return delivery coefficients for a parcel-to-outlet path.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    pid : int or str
        Parcel identifier.
    oid : int or str
        Outlet identifier.

    Returns
    -------
    dict[str, float]
        Delivery factors for sediment and nutrient routing. Missing entries
        default to ``1.0`` for every factor.
    """
    return self.delivery_coeffs.get(
        (str(pid), str(oid)),
        dict(sdr_f_to_s=1.0, sdr_s_to_o=1.0, ndr_f_to_s=1.0, ndr_s_to_o=1.0),
    )