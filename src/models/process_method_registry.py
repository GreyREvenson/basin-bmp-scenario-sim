"""Approved process implementations used by annual watershed formulations.

This is a *runtime* registry: each identifier resolves to the actual scientific
callable executed by the PLET/RUSLE assembly. It does not authorize mixing
algorithms; allowed combinations remain defined in the formulation registry.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping

from src.models.hydrology.runoff_plet_curve_number import plet_annual_surface_runoff_in
from src.models.hydrology.infiltration_plet_balance import plet_annual_infiltration_in
from src.models.sediment.soil_loss_rusle import rusle_sediment_load_rate_kg_ha_yr
from src.models.nutrients.surface_load_plet_emc import calculate_plet_pathway_load_rates


@dataclass(frozen=True)
class ProcessImplementation:
    """Scientific function plus its persistent configuration identifier."""
    process: str
    identifier: str
    compute: Callable


def _initialize_statistical_pathway_loads(ctx, logger):
    from src.models.assemblies.statistical_annual import initialize_statistical_baseline
    return initialize_statistical_baseline(ctx, len(ctx.pollutants), logger)


# Distinct process categories prevent accidentally using a runoff function as
# an infiltration or nutrient implementation merely because it is callable.
PROCESS_IMPLEMENTATIONS: Mapping[str, Mapping[str, ProcessImplementation]] = {
    "runoff": {
        "plet_curve_number": ProcessImplementation(
            "runoff", "plet_curve_number", plet_annual_surface_runoff_in),
    },
    "infiltration": {
        "plet_balance": ProcessImplementation(
            "infiltration", "plet_balance", plet_annual_infiltration_in),
    },
    "sediment": {
        "rusle": ProcessImplementation(
            "sediment", "rusle", rusle_sediment_load_rate_kg_ha_yr),
    },
    "nutrients": {
        "plet_emc": ProcessImplementation(
            "nutrients", "plet_emc", calculate_plet_pathway_load_rates),
    },
    "baseline_loads": {
        # Statistical sampling is an independently callable model method.
        "statistical_pathway_loads": ProcessImplementation(
            "baseline_loads", "statistical_pathway_loads", _initialize_statistical_pathway_loads),
    },
}


def resolve_process_method(process: str, identifier: str) -> ProcessImplementation:
    """Resolve a registered algorithm, rejecting unknown processes and IDs."""
    impl = PROCESS_IMPLEMENTATIONS.get(process, {}).get(identifier)
    if impl is None:
        raise ValueError(
            f"Unregistered scientific process method {process}={identifier!r}; "
            f"available: {sorted(PROCESS_IMPLEMENTATIONS.get(process, {}))}"
        )
    return impl


@dataclass(frozen=True)
class PletProcessMethods:
    """Callable scientific kernels comprising an approved PLET formulation."""
    runoff: Callable
    infiltration: Callable
    sediment: Callable
    nutrients: Callable


def resolve_plet_process_methods(methods: Mapping[str, str]) -> PletProcessMethods:
    """Resolve and validate every required PLET/RUSLE process algorithm."""
    required = {"runoff", "infiltration", "sediment", "nutrients"}
    if set(methods) != required:
        raise ValueError(
            "PLET/RUSLE formulation process set must contain exactly "
            f"{sorted(required)}; got {sorted(methods)}"
        )
    return PletProcessMethods(**{
        name: resolve_process_method(name, methods[name]).compute
        for name in ("runoff", "infiltration", "sediment", "nutrients")
    })
