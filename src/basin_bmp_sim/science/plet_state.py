"""Plet state for the annual PLET/RUSLE model."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np



@dataclass
class LoadState:
    """Container for per-parcel scenario state.

    The state stores current sampled values, baseline values, parcel ordering,
    and pathway-level load_rates so that BMP application can update the scenario in
    place while still preserving the original starting point.

    Attributes
    ----------
    parcel_ids : list[str]
        Parcel identifiers in the same order used by the numeric arrays.
    parameters : list[dict[str, Any]]
        Current sampled parameter values for each parcel.
    concentrations : list[dict[str, float]]
        Current runoff concentrations for each parcel.
    groundwater_concentrations : list[dict[str, float]]
        Current groundwater concentrations for each parcel.
    has_rusle : list[bool]
        Flags indicating whether each parcel has complete RUSLE inputs.
    pollutants : list[str]
        Pollutants tracked by the simulation.
    pathway_load_rates : numpy.ndarray
        Current pathway-specific parcel load_rates with shape
        ``(n_parcels, n_pollutants, 2)``. In PLET/RUSLE mode the two pathways
        are ``surface`` and ``subsurface``.
    untreated_groundwater_load_rates : numpy.ndarray
        Deprecated compatibility array. PLET/RUSLE subsurface loads are now
        included in the ``subsurface`` pathway, so this array is always zero.
    baseline_parameters : list[dict[str, Any]]
        Snapshot of the original parameter values.
    baseline_concentrations : list[dict[str, float]]
        Snapshot of the original runoff concentrations.
    baseline_groundwater_concentrations : list[dict[str, float]]
        Snapshot of the original groundwater concentrations.
    baseline_pathway_load_rates : numpy.ndarray or None
        Snapshot of the original pathway-specific load_rates.
    baseline_untreated_groundwater_load_rates : numpy.ndarray or None
        Snapshot of the original groundwater load_rates excluded from BMP
        treatment.
    """

    parcel_ids: List[str]
    parameters: List[Dict[str, Any]]
    concentrations: List[Dict[str, float]]
    groundwater_concentrations: List[Dict[str, float]]
    has_rusle: List[bool]
    pollutants: List[str]
    pathway_load_rates: np.ndarray
    untreated_groundwater_load_rates: np.ndarray
    baseline_parameters: List[Dict[str, Any]] = field(default_factory=list)
    baseline_concentrations: List[Dict[str, float]] = field(default_factory=list)
    baseline_groundwater_concentrations: List[Dict[str, float]] = field(default_factory=list)
    baseline_pathway_load_rates: Optional[np.ndarray] = None
    baseline_untreated_groundwater_load_rates: Optional[np.ndarray] = None

    @property
    def index_by_pid(self) -> Dict[str, int]:
        """Map parcel IDs to their positional index.

        Returns
        -------
        dict[str, int]
            Dictionary mapping each parcel ID to its index in the state
            arrays.
        """
        return {pid: i for i, pid in enumerate(self.parcel_ids)}
