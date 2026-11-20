"""Resolve a reviewed formulation to executable scientific methods.

The formulation registry alone establishes scientific approval. This builder
checks that every approved method is actually implemented, and constructs
per-realization bound method sets. It does not permit arbitrary YAML mixes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from src.formulations.formulation_registry import Formulation, validate_formulation_selection
from src.bmp.response_method_registry import (
    BMPResponseMethod, resolve_response_method, response_method_for_cps,
)
from src.models.process_method_registry import (
    PletProcessMethods, resolve_process_method, resolve_plet_process_methods,
)


@dataclass(frozen=True)
class ScientificMethodPlan:
    """Fully resolved executable methods for one approved formulation."""
    formulation: Formulation
    initialize_baseline: Callable
    process_methods: PletProcessMethods | None

    def bmp_response(self, cps: int) -> BMPResponseMethod:
        return response_method_for_cps(cps, self.formulation.bmp_response_methods)


def validate_registered_formulation(formulation: Formulation) -> None:
    """Ensure approved methods have concrete, compatible implementations."""
    for process, identifier in formulation.process_methods.items():
        resolve_process_method(process, identifier)
    for family, identifier in formulation.bmp_response_methods.items():
        resolve_response_method(family, identifier)
    if formulation.baseline_mode == "plet_rusle":
        resolve_plet_process_methods(formulation.process_methods)
    elif formulation.baseline_mode == "statistical":
        if dict(formulation.process_methods) != {"baseline_loads": "statistical_pathway_loads"}:
            raise ValueError("Statistical formulation requires statistical pathway baseline method")
    else:
        raise ValueError(f"Unsupported formulation baseline mode {formulation.baseline_mode!r}")


def build_method_plan(cfg: Mapping[str, Any], mode: str) -> ScientificMethodPlan:
    """Resolve the worker's actual mode and optional user formulation choice.

    Direct Python callers often pass minimal configs without a
    ``load_generation`` block; the worker's validated mode is authoritative.
    An *explicitly supplied*, conflicting load-generation mode is rejected.
    """
    if mode not in ("plet_rusle", "statistical"):
        raise ValueError(f"Unsupported load-generation mode: {mode!r}")
    selection_cfg = dict(cfg)
    loading = dict(selection_cfg.get("load_generation") or {})
    loading.setdefault("mode", mode)
    selection_cfg["load_generation"] = loading
    form = validate_formulation_selection(selection_cfg)
    if form.baseline_mode != mode:
        raise ValueError(
            f"Formulation {form.identifier!r} does not match worker baseline mode {mode!r}"
        )
    validate_registered_formulation(form)
    if mode == "plet_rusle":
        from src.models.assemblies.plet_rusle_annual import initialize_plet_rusle_baseline
        methods = resolve_plet_process_methods(form.process_methods)

        def initialize(ctx, logger):
            return initialize_plet_rusle_baseline(ctx, logger, process_methods=methods)
    else:
        methods = None
        initialize = resolve_process_method(
            "baseline_loads", "statistical_pathway_loads"
        ).compute
    return ScientificMethodPlan(form, initialize, methods)
