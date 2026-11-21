"""Explicit scientific capability declarations for optional annual BMI backends.

This is a *preflight declaration*, not a scientific certification. A backend
owns its equations, mass balance, units and BMP response validity. Declaring
support does not make two watershed formulations scientifically equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.interfaces.annual_model_contract import (
    ANNUAL_MODEL_CONTRACT_VERSION, AnnualModelDomain, AnnualWatershedBackend,
)
from src.interfaces.exchange_variables import EXCHANGE_SCHEMA_VERSION

SUPPORTED_FEATURES = frozenset({
    "signed_efficiency", "multi_parcel_treatment", "afo_recovery",
    "upstream_routing",
})
# The existing live generator may emit signed efficiencies and multi-parcel
# placements. A model must understand both even when a particular seed does
# not happen to exercise those features.
REQUIRED_LIVE_FEATURES = frozenset({"signed_efficiency", "multi_parcel_treatment"})


def _identifiers(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    if (not isinstance(values, tuple) or not values or
            any(not isinstance(v, str) or not v for v in values) or
            len(set(values)) != len(values)):
        raise ValueError(f"Backend {field} must be a nonempty tuple of unique strings")
    return values


@dataclass(frozen=True)
class AnnualBackendCapabilities:
    """What a third-party annual model explicitly claims it can interpret.

    A supplier must list *all* supported CPS codes and identifiers, not only
    those exercised by a hand-picked example. Capabilities are checked before
    initializing either coupled component or consuming baseline RNG draws.
    """

    supported_cps: tuple[int, ...]
    pollutants: tuple[str, ...]
    pathways: tuple[str, ...]
    features: frozenset[str]
    annual_contract_version: int = ANNUAL_MODEL_CONTRACT_VERSION
    exchange_schema_version: int = EXCHANGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (type(self.annual_contract_version) is not int or
                self.annual_contract_version != ANNUAL_MODEL_CONTRACT_VERSION):
            raise ValueError("Unsupported annual backend contract version")
        if (type(self.exchange_schema_version) is not int or
                self.exchange_schema_version != EXCHANGE_SCHEMA_VERSION):
            raise ValueError("Unsupported backend BMI exchange schema version")
        if (not isinstance(self.supported_cps, tuple) or not self.supported_cps or
                any(type(cps) is not int or cps <= 0 for cps in self.supported_cps) or
                len(set(self.supported_cps)) != len(self.supported_cps)):
            raise ValueError("Backend supported_cps must be unique positive integer codes")
        _identifiers(self.pollutants, "pollutants")
        _identifiers(self.pathways, "pathways")
        if (not isinstance(self.features, frozenset) or
                not all(isinstance(v, str) and v in SUPPORTED_FEATURES for v in self.features)):
            raise ValueError("Backend features contain unsupported or invalid declarations")

    def negotiate(self, domain: AnnualModelDomain, required_features: frozenset[str]) -> None:
        """Validate generator requirements before scientific initialization.

        The actual index identities/order and BMI units are independently
        checked against the two initialized components by the coupler.
        """
        if not isinstance(domain, AnnualModelDomain):
            raise TypeError("External backend must supply an AnnualModelDomain")
        missing_cps = set(domain.bmp_cps) - set(self.supported_cps)
        if missing_cps:
            raise ValueError(f"Backend does not support configured BMP CPS codes: {sorted(missing_cps)}")
        if tuple(domain.pollutants) != self.pollutants:
            raise ValueError("Backend pollutant identities/order disagree with the generator domain")
        if tuple(domain.pathway_names) != self.pathways:
            raise ValueError("Backend pathway identities/order disagree with the generator domain")
        required = REQUIRED_LIVE_FEATURES | required_features
        missing = required - self.features
        if missing:
            raise ValueError(f"Backend missing required scientific capabilities: {sorted(missing)}")


@dataclass(frozen=True)
class ExternalBackendBundle:
    """Result of a trusted third-party backend factory."""

    domain: AnnualModelDomain
    backend: AnnualWatershedBackend
    capabilities: AnnualBackendCapabilities

    def validate(self, required_features: frozenset[str]) -> None:
        if not isinstance(self.capabilities, AnnualBackendCapabilities):
            raise TypeError("External backend factory must supply AnnualBackendCapabilities")
        self.capabilities.negotiate(self.domain, required_features)
        for name in ("initialize", "snapshot", "apply_bmp", "complete_bmp", "finalize"):
            if not callable(getattr(self.backend, name, None)):
                raise TypeError(f"External backend is missing callable {name}()")
