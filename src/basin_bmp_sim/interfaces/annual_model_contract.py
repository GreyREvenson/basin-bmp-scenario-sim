"""Model-neutral contract for a single annual watershed realization.

The public objects contain **only** ordered identifiers, units-defined numeric
arrays and a transferable BMP request. They do not import PLET, RUSLE, AFO,
model contexts, or driver implementation details. Alternative scientific
backends own all baseline, response, mass-balance and outlet-routing physics.

Contract version 1 uses the Phase 8 BMI index/placement schema (version 2).
A new BMP-specific parameter requires a negotiated schema extension rather
than inferring it from undocumented Python attributes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from basin_bmp_sim.interfaces.domain_exchange import validate_placement_domain
from basin_bmp_sim.interfaces.exchange_variables import PLACEMENT_OUTPUT_VARIABLES

PLACEMENT_FIELDS = tuple(v.name for v in PLACEMENT_OUTPUT_VARIABLES)

ANNUAL_MODEL_CONTRACT_VERSION = 1


def _immutable_array(value, dtype: str) -> np.ndarray:
    # Numeric indices are identity-bearing: silently truncating floats or
    # coercing bools could redirect loads/BMPs onto different parcels.
    original = np.asarray(value)
    if original.size and np.dtype(dtype).kind in "iu" and original.dtype.kind not in "iu":
        raise ValueError("Annual model integer indices must not be coerced from other types")
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _identifiers(values, name: str) -> tuple[str, ...]:
    result = tuple(str(v) for v in values)
    if not result or any(not v for v in result) or len(set(result)) != len(result):
        raise ValueError(f"Annual model {name} must contain unique nonempty identifiers")
    return result


@dataclass(frozen=True)
class AnnualModelDomain:
    """Index identities and physical domain required for *BMP generation*.

    All positions refer to the full hydrologic parcel universe. Upstream
    neighbors are **immediate**, ordered and acyclic. The selection indices
    reference that universe; probabilities sum to one.
    """

    parcel_ids: tuple[str, ...]
    outlet_oids: tuple[str, ...]
    pollutants: tuple[str, ...]
    pathway_names: tuple[str, ...]
    bmp_cps: tuple[int, ...]
    parcel_area_ha: np.ndarray
    parcel_perim_m: np.ndarray
    upstream_offset: np.ndarray
    upstream_index: np.ndarray
    selection_source_idxs: np.ndarray
    parcel_selection_probs: np.ndarray

    def __post_init__(self) -> None:
        for name in ("parcel_ids", "outlet_oids", "pollutants", "pathway_names"):
            object.__setattr__(self, name, _identifiers(getattr(self, name), name))
        cps = tuple(self.bmp_cps)
        if (not cps or any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) or int(x) <= 0 for x in cps)
                or len(set(cps)) != len(cps)):
            raise ValueError("Annual model BMP CPS codes must be unique positive integers")
        object.__setattr__(self, "bmp_cps", tuple(int(x) for x in cps))
        arrays = {
            "parcel_area_ha": "float64", "parcel_perim_m": "float64",
            "upstream_offset": "int64", "upstream_index": "int64",
            "selection_source_idxs": "int64", "parcel_selection_probs": "float64",
        }
        for name, dtype in arrays.items():
            object.__setattr__(self, name, _immutable_array(getattr(self, name), dtype))
        validate_placement_domain(self.domain_values(), n_parcels=len(self.parcel_ids),
                                  n_selection=len(self.selection_source_idxs))

    @property
    def parcel_up_idxs(self) -> list[list[int]]:
        return [list(map(int, self.upstream_index[self.upstream_offset[i]:self.upstream_offset[i + 1]]))
                for i in range(len(self.parcel_ids))]

    def domain_values(self) -> dict[str, np.ndarray]:
        from basin_bmp_sim.interfaces.domain_exchange import DOMAIN_SCHEMA_VERSION
        return {
            "domain__schema_version": np.array([DOMAIN_SCHEMA_VERSION], dtype="int64"),
            "domain__parcel_area_ha": self.parcel_area_ha,
            "domain__parcel_perimeter_m": self.parcel_perim_m,
            "domain__upstream_offset": self.upstream_offset,
            "domain__upstream_index": self.upstream_index,
            "domain__selection_source_index": self.selection_source_idxs,
            "domain__selection_probability": self.parcel_selection_probs,
        }


@dataclass(frozen=True)
class AnnualLoadSnapshot:
    """Scientific output arrays; the adapter preserves signed mass changes.

    Load rates: kg ha-1 yr-1 in (parcel, pollutant) order. Outlet delivered
    *removed* mass: kg in (pollutant, outlet) order. Negative outlet removals
    represent net pollutant increases and must not be clipped.
    """

    baseline_rates: np.ndarray
    current_rates: np.ndarray
    outlet_removed_kg: np.ndarray

    def validated(self, domain: AnnualModelDomain) -> "AnnualLoadSnapshot":
        baseline = _immutable_array(self.baseline_rates, "float64")
        current = _immutable_array(self.current_rates, "float64")
        outlet = _immutable_array(self.outlet_removed_kg, "float64")
        expected_rates = (len(domain.parcel_ids), len(domain.pollutants))
        expected_outlet = (len(domain.pollutants), len(domain.outlet_oids))
        if baseline.shape != expected_rates or current.shape != expected_rates:
            raise ValueError(f"Annual model load-rate matrices must have shape {expected_rates}")
        if outlet.shape != expected_outlet:
            raise ValueError(f"Annual model outlet removed-mass matrix must have shape {expected_outlet}")
        if not (np.isfinite(baseline).all() and np.isfinite(current).all()
                and np.isfinite(outlet).all()):
            raise ValueError("Annual model load and outlet arrays must be finite")
        return AnnualLoadSnapshot(baseline, current, outlet)


@dataclass(frozen=True)
class AnnualBMPPlacement:
    """Complete model-neutral BMP *request*, decoded from numeric BMI inputs.

    Negative efficiencies are permitted; family-specific design semantics
    belong to the receiving scientific backend. No native BMPDesign object
    or PLET-specific context crosses this boundary.
    """

    cps: int
    host_parcel_index: int
    affected_indices: tuple[int, ...]
    affected_fractions: tuple[float, ...]
    efficiency: np.ndarray  # (pollutant, pathway), dimensionless
    failed: bool
    recovery_effect_scale: float
    cost_quantity_ha: float
    wetland_area_ha: float
    catchment_area_ratio: float
    buffer_length_m: float
    buffer_area_ha: float
    portion_treated: float
    recorded_recovery_gains: tuple[float, ...]

    @classmethod
    def from_bmi(cls, values: Mapping[str, np.ndarray], domain: AnnualModelDomain) -> "AnnualBMPPlacement":
        missing = set(PLACEMENT_FIELDS) - set(values)
        if missing:
            raise ValueError(f"Incomplete annual BMI placement request: {sorted(missing)}")

        def scalar(name: str, dtype: str) -> int | float:
            arr = np.asarray(values[name])
            if arr.shape != (1,) or arr.dtype != np.dtype(dtype):
                raise ValueError(f"Annual BMI {name} must be one {dtype} value")
            return arr[0].item()

        cps = scalar("placement__cps_code", "int64")
        idx = scalar("placement__host_parcel_index", "int64")
        if cps not in domain.bmp_cps or not 0 <= idx < len(domain.parcel_ids):
            raise ValueError("Annual BMI placement has an unsupported CPS or host parcel")
        affected = np.asarray(values["placement__affected_parcel_index"])
        fractions = np.asarray(values["placement__affected_parcel_fraction"])
        if (affected.dtype != np.int64 or fractions.dtype != np.float64 or
                affected.ndim != 1 or fractions.ndim != 1 or not len(affected) or
                affected.shape != fractions.shape or int(affected[0]) != idx or
                (affected < 0).any() or (affected >= len(domain.parcel_ids)).any() or
                len(set(affected.tolist())) != len(affected) or not np.isfinite(fractions).all()
                or (fractions < 0).any() or (fractions > 1).any()):
            raise ValueError("Annual BMI affected-parcel indices or fractions are invalid")
        eff = np.asarray(values["placement__efficiency"])
        expected = (len(domain.pollutants), len(domain.pathway_names))
        if (eff.dtype != np.float64 or eff.shape != (expected[0] * expected[1],)
                or not np.isfinite(eff).all() or (eff > 1).any()):
            raise ValueError("Annual BMI placement efficiency layout or values are invalid")
        failed = scalar("placement__failed", "int64")
        if failed not in (0, 1):
            raise ValueError("Annual BMI failed flag must be zero or one")
        scale = scalar("placement__recovery_effect_scale", "float64")
        if not np.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("Annual BMI recovery scale must lie in [0, 1]")
        geom_names = ("placement__cost_quantity_ha", "placement__wetland_area_ha",
                      "placement__catchment_area_ratio", "placement__buffer_length_m",
                      "placement__buffer_area_ha", "placement__portion_treated")
        geometry = [scalar(n, "float64") for n in geom_names]
        if any(not np.isfinite(x) or x < 0 for x in geometry):
            raise ValueError("Annual BMI physical design fields must be finite and nonnegative")
        recorded = np.asarray(values["placement__recorded_recovery_gains"])
        if (recorded.dtype != np.float64 or recorded.ndim != 1 or
                not np.isfinite(recorded).all() or (recorded < 0).any() or (recorded > 1).any()):
            raise ValueError("Annual BMI recorded AFO gains must be finite and in [0, 1]")
        return cls(int(cps), int(idx), tuple(map(int, affected)), tuple(map(float, fractions)),
                   _immutable_array(eff.reshape(expected), "float64"), bool(failed), float(scale),
                   *(float(x) for x in geometry), tuple(map(float, recorded)))


class AnnualWatershedBackend(Protocol):
    """Scientific model extension point, independent of this project's kernels.

    `apply_bmp` may advance scientific state and draw using the supplied RNG;
    `complete_bmp` runs *after* the generator has drawn BMP cost. The backend
    owns pollution physics, mass balances, and its own pending-response state.
    """

    def initialize(self, rng: np.random.Generator) -> None: ...
    def snapshot(self) -> AnnualLoadSnapshot: ...
    def apply_bmp(self, placement: AnnualBMPPlacement, scenario_id: int,
                  rng: np.random.Generator) -> Sequence[float]: ...
    def complete_bmp(self, cost_usd: float) -> Mapping[str, Any]: ...
    def finalize(self) -> None: ...
