"""Conservative, versioned scientific model formulations.

Method choices represent reviewed combinations, not arbitrary composable
modules. This first extraction only registers scientific behavior that already
exists in the input branch; future alternatives require explicit review.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Formulation:
    identifier: str
    baseline_mode: str
    process_methods: Mapping[str, str]
    bmp_response_methods: Mapping[str, str]


FORMULATIONS: dict[str, Formulation] = {
    "plet_rusle_annual_v1": Formulation(
        "plet_rusle_annual_v1", "plet_rusle",
        {"runoff":"plet_curve_number", "infiltration":"plet_balance",
         "sediment":"rusle", "nutrients":"plet_emc"},
        {"wetland":"plet_pathway_efficiency", "grassed":"plet_pathway_efficiency",
         "infield":"plet_pathway_efficiency"},
    ),
    "statistical_annual_v1": Formulation(
        "statistical_annual_v1", "statistical",
        {"baseline_loads":"statistical_pathway_loads"},
        {"wetland":"plet_pathway_efficiency", "grassed":"plet_pathway_efficiency",
         "infield":"plet_pathway_efficiency"},
    ),
}


def validate_formulation_selection(cfg: Mapping[str, Any]) -> Formulation:
    """Fail fast for incompatible/unsupported model methods in YAML.

    Absence of an explicit formulation preserves the existing load_generation
    behavior for backwards compatibility.
    """
    simulation = cfg.get("simulation") or {}
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be a mapping")
    unknown = set(simulation) - {"formulation", "bmp_methods"}
    if unknown:
        raise ValueError(f"Unknown simulation options: {sorted(unknown)}")
    loading = cfg.get("load_generation") or {}
    if not isinstance(loading, dict):
        raise ValueError("load_generation must be a mapping")
    mode = str(loading.get("mode", "statistical")).strip().lower()
    requested = simulation.get("formulation") or {
        "plet_rusle": "plet_rusle_annual_v1", "statistical": "statistical_annual_v1"
    }.get(mode)
    if requested not in FORMULATIONS:
        raise ValueError(
            f"Unsupported model formulation {requested!r}; "
            f"supported: {sorted(FORMULATIONS)}"
        )
    form = FORMULATIONS[requested]
    if mode != form.baseline_mode:
        raise ValueError(
            f"Formulation {requested!r} requires load_generation.mode="
            f"{form.baseline_mode!r}; got {mode!r}"
        )
    methods = simulation.get("bmp_methods") or {}
    if not isinstance(methods, dict):
        raise ValueError("simulation.bmp_methods must be a mapping")
    for kind, method in methods.items():
        if kind not in form.bmp_response_methods:
            raise ValueError(f"Unsupported BMP response family: {kind!r}")
        approved = form.bmp_response_methods[kind]
        if method != approved:
            raise ValueError(
                f"Unapproved BMP response method {method!r} for {kind!r} "
                f"in formulation {requested!r}; currently approved: {approved!r}"
            )
    return form
