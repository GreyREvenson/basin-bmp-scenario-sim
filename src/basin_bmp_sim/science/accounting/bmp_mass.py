"""Model-owned BMP mass accounting and reproducible parcel-level deltas.

No RNG, output serialization, or driver-level orchestration belongs here.
All computations preserve the historical kg/timestep convention.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from basin_bmp_sim.domain.constants import CURRENT_TIMESTEP_YEARS as _CURRENT_TIMESTEP_YEARS
from basin_bmp_sim.domain.constants import OUTPUT_IMPACTED_PIDS

def _safe_mass_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return a dimensionless mass ratio, or ``None`` when undefined.

        Parameters
        ----------
        numerator : float
            Numerator of the dimensionless ratio.
        denominator : float
            Denominator of the dimensionless ratio.

        Returns
        -------
        Optional[float]
            Dimensionless ratio, or ``None`` when the denominator is not positive or finite.

    """
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return numerator / denominator


def _bmp_impacted_parcel_indices(ctx: Any, parcel_idx: int, bmp_rec: Dict[str, Any]) -> List[int]:
    """Return full-hydrologic-universe parcel indices represented by one BMP.

        For wetlands, ``impacted_pids`` may include upstream parcels in addition to
        the placement parcel. Other BMP types operate on the placement parcel only.
        The placement parcel is always included defensively even when the wetland
        output contains an empty or partial impacted-PID string.

        Parameters
        ----------
        ctx : Any
            Active scenario or model context.
        parcel_idx : int
            Zero-based index of the parcel.
        bmp_rec : Dict[str, Any]
            Mutable record for the current BMP placement.

        Returns
        -------
        List[int]
            Indices of parcels represented by the BMP placement.

    """
    pids: List[str] = [str(ctx.parcel_ids[parcel_idx])]
    raw = str(bmp_rec.get(OUTPUT_IMPACTED_PIDS) or "")
    for token in raw.split(","):
        pid = token.strip()
        if pid and pid not in pids:
            pids.append(pid)
    pid_to_index = getattr(ctx, "pid_to_index", None) or {
        str(pid): idx for idx, pid in enumerate(ctx.parcel_ids)
    }
    return [int(pid_to_index[pid]) for pid in pids if pid in pid_to_index]


def _removed_mass_by_parcel_kg(
    ctx: Any,
    pre_bmp_load_rates: np.ndarray,
    post_bmp_load_rates: np.ndarray,
    parcel_idx: int,
    bmp_rec: Dict[str, Any],
    recorded_removed_mass_kg: np.ndarray,
    *,
    extra_parcel_indices: Tuple[int, ...] = (),
) -> List[Tuple[int, np.ndarray]]:
    """Compute signed BMP mass changes on each affected parcel (kg/timestep).

    Optional extra indices capture manure-receiving parcels affected by AFO
    feedlot BMPs, even when they are not physically treated by the BMP.
    These deltas are the canonical source for outlet trajectories: a wetland
    can affect its host AND multiple upstream parcels, each of which has its
    own parcel-to-outlet membership and delivery coefficients. The recorded
    BMP removal must reconcile with these changes rather than being assigned
    wholesale to the host's outlet and delivery factors.
    """
    changes: List[Tuple[int, np.ndarray]] = []
    total = np.zeros(pre_bmp_load_rates.shape[1], dtype=float)
    accounted_indices = set(_bmp_impacted_parcel_indices(ctx, parcel_idx, bmp_rec))
    accounted_indices.update(int(idx) for idx in extra_parcel_indices)
    for idx in sorted(accounted_indices):
        change = (
            np.asarray(pre_bmp_load_rates[idx, :] - post_bmp_load_rates[idx, :], dtype=float)
            * float(ctx.parcel_area_ha[idx])
            * _CURRENT_TIMESTEP_YEARS
        )
        changes.append((idx, change))
        total += change
    if not np.allclose(total, recorded_removed_mass_kg, rtol=1e-7, atol=1e-8):
        raise RuntimeError(
            'BMP removal mass does not reconcile with the affected parcel load changes: '
            f'pid={ctx.parcel_ids[parcel_idx]}, cps={bmp_rec.get("cps")}, '
            f'parcel_delta_kg={total.tolist()}, '
            f'recorded_removed_kg={np.asarray(recorded_removed_mass_kg).tolist()}'
        )
    return changes



def _baseline_mass_before_bmp_kg(
    ctx: Any,
    pre_bmp_load_rates: np.ndarray,
    parcel_idx: int,
    bmp_rec: Dict[str, Any],
    *,
    extra_parcel_indices: Tuple[int, ...] = (),
) -> np.ndarray:
    """Calculate pre-BMP pollutant mass for the parcels represented by a BMP.

        ``pre_bmp_load_rates`` contains the current load rate immediately before this
        BMP is applied. For the present annual model, rate × area × 1 year gives kg.
        Wetland denominators include all hydrologically impacted parcels identified
        by the wetland routine, including parcels that are not BMP-selectable.
        For manure-capturing AFO BMPs, also include receiving parcels where
        the BMP induces additional nutrient loading. Each parcel is counted once.

        Parameters
        ----------
        ctx : Any
            Active scenario or model context.
        pre_bmp_load_rates : np.ndarray
            Parcel pollutant load rates immediately before BMP application.
        parcel_idx : int
            Zero-based index of the parcel.
        bmp_rec : Dict[str, Any]
            Mutable record for the current BMP placement.

        Returns
        -------
        np.ndarray
            Pre-BMP pollutant masses for the represented parcels, in kilograms.

    """
    n_pol = int(pre_bmp_load_rates.shape[1])
    mass = np.zeros(n_pol, dtype=float)
    accounted_indices = set(_bmp_impacted_parcel_indices(ctx, parcel_idx, bmp_rec))
    accounted_indices.update(int(idx) for idx in extra_parcel_indices)
    for idx in sorted(accounted_indices):
        area_ha = float(ctx.parcel_area_ha[idx])
        mass += np.asarray(pre_bmp_load_rates[idx, :], dtype=float) * area_ha * _CURRENT_TIMESTEP_YEARS
    return mass


def _add_mass_metrics_to_bmp_record(
    bmp_rec: Dict[str, Any],
    pollutants: List[str],
    baseline_mass_kg: np.ndarray,
    treated_mass_kg: np.ndarray,
    removed_mass_kg: np.ndarray,
) -> None:
    """Write explicit mass accounting and dimensionless BMP metrics to a record.

        Parameters
        ----------
        bmp_rec : Dict[str, Any]
            Mutable record for the current BMP placement.
        pollutants : List[str]
            Pollutant names in model order.
        baseline_mass_kg : np.ndarray
            Baseline pollutant masses, in kilograms.
        treated_mass_kg : np.ndarray
            Pollutant masses subjected to BMP treatment, in kilograms.
        removed_mass_kg : np.ndarray
            Pollutant masses removed by the BMP, in kilograms.

    """
    bmp_rec["mass_timestep_years"] = float(_CURRENT_TIMESTEP_YEARS)
    for pol_idx, pol in enumerate(pollutants):
        baseline_mass = float(baseline_mass_kg[pol_idx])
        treated_mass = float(treated_mass_kg[pol_idx])
        removed_mass = float(removed_mass_kg[pol_idx])

        bmp_rec[f"baseline_mass_{pol}_kg"] = baseline_mass
        bmp_rec[f"treated_baseline_mass_{pol}_kg"] = treated_mass
        bmp_rec[f"removed_mass_{pol}_kg"] = removed_mass

        exposure = _safe_mass_ratio(treated_mass, baseline_mass)
        realized = _safe_mass_ratio(removed_mass, treated_mass)
        overall = _safe_mass_ratio(removed_mass, baseline_mass)
        bmp_rec[f"treatment_exposure_fraction_{pol}"] = exposure
        bmp_rec[f"realized_efficiency_{pol}"] = realized
        bmp_rec[f"overall_reduction_fraction_{pol}"] = overall
