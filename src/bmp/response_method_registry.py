"""Scientific BMP-response dispatch for approved formulation methods.

Practice/CPS resolution lives in bmp_response_registry. This registry selects
how a practice changes scientific model state, without embedding CPS branches
in the watershed model or allowing arbitrary method mixing.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping

from src.bmp.bmp_response_registry import response_family


def _plet_wetland(ctx, parcel_idx, efficiencies, loads, record, mass_rates, design):
    ctx._simulate_wetland(parcel_idx, efficiencies, loads, record, mass_rates, design=design)


def _plet_grassed(ctx, parcel_idx, efficiencies, loads, record, mass_rates, design):
    ctx._simulate_grassed(parcel_idx, efficiencies, loads, record, mass_rates, design=design)


def _plet_infield(ctx, parcel_idx, efficiencies, loads, record, mass_rates, design):
    # The legacy in-field scientific method has no separate design argument.
    ctx._simulate_infield(parcel_idx, efficiencies, loads, record, mass_rates)


@dataclass(frozen=True)
class BMPResponseMethod:
    family: str
    identifier: str
    apply: Callable


RESPONSE_IMPLEMENTATIONS: Mapping[str, Mapping[str, BMPResponseMethod]] = {
    "wetland": {
        "plet_pathway_efficiency": BMPResponseMethod(
            "wetland", "plet_pathway_efficiency", _plet_wetland),
    },
    "grassed": {
        "plet_pathway_efficiency": BMPResponseMethod(
            "grassed", "plet_pathway_efficiency", _plet_grassed),
    },
    "infield": {
        "plet_pathway_efficiency": BMPResponseMethod(
            "infield", "plet_pathway_efficiency", _plet_infield),
    },
}


def resolve_response_method(family: str, identifier: str) -> BMPResponseMethod:
    """Resolve an actual registered response implementation, or fail fast."""
    method = RESPONSE_IMPLEMENTATIONS.get(family, {}).get(identifier)
    if method is None:
        raise ValueError(
            f"Unregistered BMP response method {family}={identifier!r}; "
            f"available: {sorted(RESPONSE_IMPLEMENTATIONS.get(family, {}))}"
        )
    return method


def response_method_for_cps(cps: int, methods: Mapping[str, str]) -> BMPResponseMethod:
    """Resolve the formulation-approved response for a concrete CPS code."""
    family = response_family(cps)
    if family not in methods:
        raise ValueError(f"Formulation does not support BMP response family {family!r} (CPS {cps})")
    return resolve_response_method(family, methods[family])
