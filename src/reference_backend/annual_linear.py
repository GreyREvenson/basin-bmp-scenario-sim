"""Standalone, file-configured annual scientific backend *example*.

This is an independently written response law, not EPA's PLET/RUSLE model,
not a validated runoff/routing formulation, and not a scientific replacement.
It is designed to verify that a third-party backend can be selected and run
using only the published annual-model and capability contracts.

All rates are kg ha^-1 yr^-1. Outlet removal is signed kg yr^-1, accumulated
within a single realization. A parcel's delivered fraction is allocated across
outlets with row sum <= 1 to avoid duplicating removed mass.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from basin_bmp_sim.interfaces.annual_model_contract import (
    AnnualBMPPlacement, AnnualLoadSnapshot, AnnualModelDomain,
)
from basin_bmp_sim.interfaces.backend_capabilities import (
    AnnualBackendCapabilities, ExternalBackendBundle,
)

SCHEMA_VERSION = 1
_DOMAIN_FIELDS = frozenset({
    "schema_version", "parcel_ids", "outlet_oids", "pollutants",
    "pathway_names", "bmp_cps", "parcel_area_ha", "parcel_perim_m",
    "upstream_offset", "upstream_index", "selection_source_idxs",
    "parcel_selection_probs", "baseline_rates_kg_ha_yr",
    "pathway_weights", "outlet_delivery_fraction",
})


def _matrix(raw: Any, shape: tuple[int, int], name: str,
            *, nonnegative: bool = False) -> np.ndarray:
    try:
        value = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Reference backend {name} must be numeric") from exc
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Reference backend {name} must be finite with shape {shape}")
    if nonnegative and (value < 0).any():
        raise ValueError(f"Reference backend {name} must be nonnegative")
    return np.array(value, copy=True)


class LinearAnnualBackend:
    """Small independent annual source-response and outlet-delivery law.

    For affected parcel i, pollutant p, treatment fraction f and pathway k,
    the signed rate removed is current_rate[i,p] * f *
    sum_k(pathway_weight[p,k] * efficiency[p,k]). The current rate is updated
    after each BMP, so stacking is order-dependent. A failed BMP does nothing.
    Each resulting kg/yr difference is routed through externally supplied
    per-parcel outlet fractions; this is *not* in-stream transport modeling.
    """

    def __init__(self, domain: AnnualModelDomain, baseline: np.ndarray,
                 weights: np.ndarray, delivery: np.ndarray):
        self.domain = domain
        self._source_baseline = baseline.copy()
        self._weights = weights.copy()
        self._delivery = delivery.copy()
        self._baseline: np.ndarray | None = None
        self._current: np.ndarray | None = None
        self._outlet: np.ndarray | None = None
        self._pending: dict[str, Any] | None = None
        self._finished = False

    def initialize(self, rng: np.random.Generator) -> None:
        if self._baseline is not None:
            raise RuntimeError("Reference backend already initialized")
        self._baseline = self._source_baseline.copy()
        self._current = self._baseline.copy()
        self._outlet = np.zeros((len(self.domain.pollutants), len(self.domain.outlet_oids)))
        self._pending = None
        self._finished = False
        # The reference law is intentionally deterministic *conditional on*
        # generator draws. No unrecorded baseline stochastic inputs exist.

    def snapshot(self) -> AnnualLoadSnapshot:
        if self._baseline is None or self._finished:
            raise RuntimeError("Reference backend has no active annual realization")
        return AnnualLoadSnapshot(self._baseline, self._current, self._outlet)

    def apply_bmp(self, placement: AnnualBMPPlacement, scenario_id: int,
                  rng: np.random.Generator) -> tuple[()]:
        if self._baseline is None or self._finished or self._pending is not None:
            raise RuntimeError("Reference backend scientific response is not ready")
        if not isinstance(placement, AnnualBMPPlacement) or scenario_id < 1:
            raise ValueError("Invalid reference backend placement or scenario identity")
        if placement.recorded_recovery_gains:
            raise ValueError("Reference backend does not implement AFO recovery")
        # Calculate then commit; invalid requests leave state unchanged.
        next_rates = self._current.copy()
        next_outlet = self._outlet.copy()
        weighted = (placement.efficiency * self._weights).sum(axis=1)
        for idx, fraction in zip(placement.affected_indices, placement.affected_fractions):
            old = next_rates[idx].copy()
            delta = np.zeros_like(old) if placement.failed else old * fraction * weighted
            next_rates[idx] = old - delta
            next_outlet += np.outer(delta * self.domain.parcel_area_ha[idx],
                                    self._delivery[idx])
        if not (np.isfinite(next_rates).all() and np.isfinite(next_outlet).all()):
            raise ValueError("Reference backend nonfinite response")
        self._current = next_rates
        self._outlet = next_outlet
        self._pending = {
            "cps": int(placement.cps),
            "host_parcel_index": int(placement.host_parcel_index),
            "scenario_id": int(scenario_id),
            "failed": bool(placement.failed),
            "affected_parcel_count": len(placement.affected_indices),
        }
        return ()  # This backend does not model manure recovery.

    def complete_bmp(self, cost_usd: float) -> Mapping[str, Any]:
        if self._pending is None:
            raise RuntimeError("No pending reference BMP response")
        if not np.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("Invalid reference BMP cost")
        record = {**self._pending, "cost_usd": float(cost_usd)}
        self._pending = None
        return record

    def finalize(self) -> None:
        self._finished = True
        self._pending = None


def make_backend(*, startup: Any, options: Mapping[str, Any]) -> ExternalBackendBundle:
    """Importable Phase 11 factory; requires its own file, never native contexts.

    ``domain_file`` is relative to the startup YAML's directory, or absolute.
    No private driver or PLET object is imported by this module. The
    *generator* still independently reads its existing project YAML.
    """
    if not isinstance(options, dict) or set(options) != {"domain_file"}:
        raise ValueError("Reference backend requires only backend_options.domain_file")
    path_value = options["domain_file"]
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("Reference backend domain_file must be a nonempty path")
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(startup.model_config).parent / candidate
    path = candidate.resolve(strict=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(raw, dict) or set(raw) != _DOMAIN_FIELDS or
            type(raw.get("schema_version")) is not int or raw["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Reference backend domain file has invalid schema or fields")
    domain = AnnualModelDomain(
        tuple(raw["parcel_ids"]), tuple(raw["outlet_oids"]),
        tuple(raw["pollutants"]), tuple(raw["pathway_names"]),
        tuple(raw["bmp_cps"]), raw["parcel_area_ha"],
        raw["parcel_perim_m"], raw["upstream_offset"],
        raw["upstream_index"], raw["selection_source_idxs"],
        raw["parcel_selection_probs"],
    )
    n = len(domain.parcel_ids)
    p = len(domain.pollutants)
    k = len(domain.pathway_names)
    o = len(domain.outlet_oids)
    baseline = _matrix(raw["baseline_rates_kg_ha_yr"], (n, p),
                       "baseline_rates_kg_ha_yr", nonnegative=True)
    weights = _matrix(raw["pathway_weights"], (p, k),
                      "pathway_weights", nonnegative=True)
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Reference backend pathway_weights must sum to one per pollutant")
    delivery = _matrix(raw["outlet_delivery_fraction"], (n, o),
                       "outlet_delivery_fraction", nonnegative=True)
    if np.any(delivery.sum(axis=1) > 1.0 + 1e-12):
        raise ValueError("Reference backend outlet_delivery_fraction row sums exceed one")
    bundle = ExternalBackendBundle(
        domain=domain,
        backend=LinearAnnualBackend(domain, baseline, weights, delivery),
        capabilities=AnnualBackendCapabilities(
            supported_cps=domain.bmp_cps,
            pollutants=domain.pollutants,
            pathways=domain.pathway_names,
            features=frozenset({"signed_efficiency", "multi_parcel_treatment"}),
        ),
    )
    return bundle
