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
from src.interfaces.domain_index_contract import index_contract_values
from src.interfaces.domain_exchange import export_placement_domain
from src.interfaces.rng_exchange import encode_rng_state, decode_rng_state
from src.interfaces.placement_exchange import PLACEMENT_FIELDS, placement_from_arrays
from src.interfaces.bmi.record_exchange import encode_bmp_record
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
        self._received_placement_fields: set[str] = set()
        self._cost_received = False

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
        # A fully numeric, self-describing input buffer for each new BMP.
        # Variable-width vectors may be resized by standard BMI set_value().
        self._publish({
            **index_contract_values(self._ctx),
            **export_placement_domain(self._ctx),
            "coupling__model_rng_state_utf8": encode_rng_state(self._ctx.rng),
            "coupling__generator_rng_state_utf8": [],
            **{name: ([] if meta.rank == 1 else 0)
               for name, meta in self._metadata.items() if meta.intent == "input"},
            "placement__recovery_effect_scale": 1.0,
            "scenario__identifier": 1,
            "placement__cost": 0.0,
            "placement__recovery_gains": [],
            "model__last_bmp_record_utf8": [],
        })

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

    def set_value(self, name: str, src: np.ndarray) -> None:
        """Accept BMI placement fields; reject partial/late state mutations."""
        if name == "coupling__generator_rng_state_utf8":
            decode_rng_state(src, self._ctx.rng)
        if name in PLACEMENT_FIELDS and (self._pending is not None or self._queued is not None):
            raise RuntimeError("Cannot overwrite a queued or pending BMI placement")
        if name == "placement__cost" and self._pending is None:
            raise RuntimeError("BMP cost must be supplied after scientific model update")
        if name == "scenario__identifier" and (self._pending is not None or self._queued is not None):
            raise RuntimeError("Cannot change scenario identifier during BMP response")
        if name == "placement__cost":
            cost_value = self._source(src, "float64")
            if cost_value.size != 1 or cost_value[0] < 0:
                raise ValueError("BMI BMP cost must be finite and nonnegative")
        super().set_value(name, src)
        if name in PLACEMENT_FIELDS:
            self._received_placement_fields.add(name)
            if len(self._received_placement_fields) == len(PLACEMENT_FIELDS):
                self._end_time = self._time + 1.0
        if name == "coupling__generator_rng_state_utf8":
            self._ctx.rng.bit_generator.state = decode_rng_state(src, self._ctx.rng)
        if name == "placement__cost":
            cost = float(self._read(name)[0])
            if not np.isfinite(cost) or cost < 0:
                raise ValueError("BMI BMP cost must be finite and nonnegative")
            self._cost_received = True

    def update(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")
        if self._pending is not None:
            raise RuntimeError("Queue a placement and complete prior response before update")
        if self._queued is None:
            missing = set(PLACEMENT_FIELDS) - self._received_placement_fields
            if missing:
                raise RuntimeError(f"Queue a placement: BMI inputs are incomplete: {sorted(missing)}")
            self._queued = placement_from_arrays(self._values, self._ctx)
            self._scenario_id = int(self._read("scenario__identifier")[0])
            if self._scenario_id < 1:
                raise ValueError("BMI scenario identifier must be positive")
        self._pending = self.native_model.apply_placement(self._queued, self._scenario_id)
        self._queued = None
        self._received_placement_fields.clear()
        self._cost_received = False
        self._time += 1.0
        self._publish_model_state()
        self._publish({
            "placement__recovery_gains": self._pending.recovery_gains,
            "coupling__model_rng_state_utf8": encode_rng_state(self._ctx.rng),
        })

    @property
    def pending_recovery_gains(self) -> tuple[float, ...]:
        if self._pending is None:
            raise RuntimeError("No pending BMP response")
        return self._pending.recovery_gains

    def complete_pending(self, cost_usd: float | None = None) -> dict:
        """Finalize mass accounting after cost arrives through BMI set_value().

        The Phase 5 positional cost argument remains temporarily supported for
        native callers. The Phase 6 driver transfers cost only through BMI.
        """
        if self._pending is None:
            raise RuntimeError("No pending BMP response to complete")
        if cost_usd is None:
            if not self._cost_received:
                raise RuntimeError("Supply placement__cost via BMI before completing response")
            cost_usd = float(self._read("placement__cost")[0])
        if not np.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("Cost must be finite and nonnegative")
        record = self.native_model.complete_placement(self._pending, float(cost_usd))
        self._pending = None
        self._cost_received = False
        self._publish_model_state()
        self._publish({
            "placement__recovery_gains": [],
            "model__last_bmp_record_utf8": encode_bmp_record(record),
            "coupling__model_rng_state_utf8": encode_rng_state(self._ctx.rng),
        })
        return record

    def finalize(self) -> None:
        super().finalize()
        self._scientific = None
        self._queued = None
        self._pending = None
        self._received_placement_fields.clear()
        self._cost_received = False
