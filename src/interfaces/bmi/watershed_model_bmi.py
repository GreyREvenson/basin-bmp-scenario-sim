"""BMI 2.0 facade for an annual PLET/RUSLE or statistical realization.

A prepared, validated ScenarioModelContext is injected because constructing
it requires project-specific geospatial and stochastic inputs.  The native
Python extension ``set_placement``/``complete_pending`` carries the rich BMP
metadata and preserves AFO-recovery-before-cost draw ordering.  Converting
those extensions to pure BMI numeric inputs is a subsequent coupling task.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.interfaces.bmi.array_component import BmiArrayComponent
from src.interfaces.exchange_variables import MODEL_VARIABLES
from src.models.assemblies.annual_model_realization import AnnualModelRealization


class WatershedModelBmi(BmiArrayComponent):
    """Expose scientific baseline, treated loads, and delivered mass changes."""

    def __init__(self, ctx: Any, logger: Any):
        super().__init__(MODEL_VARIABLES, name="Basin BMP annual watershed model")
        self._ctx = ctx
        self._logger = logger
        self._scientific: AnnualModelRealization | None = None
        self._queued = None
        self._pending = None
        self._scenario_id = 1

    @property
    def native_model(self) -> AnnualModelRealization:
        if self._scientific is None:
            raise RuntimeError("Call initialize() before accessing the watershed model")
        return self._scientific

    def initialize(self, config_file: str = "") -> None:
        if config_file:
            raise ValueError("This prepared-context adapter takes no external config file")
        if self._initialized:
            raise RuntimeError("BMI watershed model is already initialized")
        self._scientific = AnnualModelRealization.initialize(self._ctx, self._logger)
        self._initialized = True
        self._time = 0.0
        self._end_time = 0.0
        self._publish_model_state()

    def _publish_model_state(self) -> None:
        model = self.native_model
        reduced = np.array([
            model.delivered_reductions[pol][oid]
            for pol in self._ctx.pollutants for oid in self._ctx.outlet_oids
        ], dtype=np.float64)
        self._publish({
            "model__parcel_count": len(self._ctx.parcel_ids),
            "model__pollutant_count": len(self._ctx.pollutants),
            "model__outlet_count": len(self._ctx.outlet_oids),
            "model__parcel_baseline_load_rate": model.baseline_load_rates.flatten(order="C"),
            "model__parcel_current_load_rate": model.load_rates.flatten(order="C"),
            "model__outlet_removed_mass": reduced,
        })

    def set_placement(self, placement, scenario_id: int = 1) -> None:
        """Queue one native placement; caller retains ownership of cost RNG."""
        if not self._initialized:
            raise RuntimeError("Call initialize() before queueing a placement")
        if self._queued is not None or self._pending is not None:
            raise RuntimeError("Complete the previous placement before queueing another")
        self._queued = placement
        self._scenario_id = int(scenario_id)
        self._end_time = self._time + 1.0

    def update(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")
        if self._queued is None or self._pending is not None:
            raise RuntimeError("Queue a placement and complete prior response before update")
        self._pending = self.native_model.apply_placement(self._queued, self._scenario_id)
        self._queued = None
        self._time += 1.0
        self._publish_model_state()

    @property
    def pending_recovery_gains(self) -> tuple[float, ...]:
        if self._pending is None:
            raise RuntimeError("No pending BMP response")
        return self._pending.recovery_gains

    def complete_pending(self, cost_usd: float) -> dict:
        if self._pending is None:
            raise RuntimeError("No pending BMP response to complete")
        if not np.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("Cost must be finite and nonnegative")
        record = self.native_model.complete_placement(self._pending, float(cost_usd))
        self._pending = None
        self._publish_model_state()
        return record

    def finalize(self) -> None:
        super().finalize()
        self._scientific = None
        self._queued = None
        self._pending = None
