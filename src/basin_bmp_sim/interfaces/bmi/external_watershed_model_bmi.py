"""BMI adapter for an independently implemented *annual* watershed backend.

Only the explicit annual-model contract and the published BMI variable catalog
are required; no PLET, RUSLE, AFO, native Model, or context classes are imported.
The backend is responsible for scientific validity and its own mass balance.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from basin_bmp_sim.interfaces.annual_model_contract import (
    AnnualBMPPlacement, AnnualLoadSnapshot, AnnualModelDomain, AnnualWatershedBackend,
    PLACEMENT_FIELDS,
)
from basin_bmp_sim.interfaces.bmi.array_component import BmiArrayComponent
from basin_bmp_sim.interfaces.coupling_lifecycle import LIFECYCLE_SCHEMA_VERSION
from basin_bmp_sim.interfaces.bmi.record_exchange import encode_bmp_record
from basin_bmp_sim.interfaces.domain_index_contract import index_contract_values
from basin_bmp_sim.interfaces.exchange_variables import MODEL_VARIABLES
from basin_bmp_sim.interfaces.rng_exchange import decode_rng_state, encode_rng_state


class ExternalWatershedModelBmi(BmiArrayComponent):
    """Exchange-model facade implementing the same BMI model variables.

    Construct with an explicit domain, backend and *scenario worker RNG*.
    No source YAML, prepared driver payload or project-internal scientific
    context is required. The live generator must publish the *same* ordered
    index manifest, and the coupler must preserve RNG exchange boundaries.
    """

    def __init__(self, domain: AnnualModelDomain, backend: AnnualWatershedBackend,
                 rng: np.random.Generator):
        super().__init__(MODEL_VARIABLES, name="External annual watershed model")
        self.domain = domain
        self.backend = backend
        self.rng = rng
        self._received: set[str] = set()
        self._pending = False
        self._cost_received = False

    def _publish_snapshot(self) -> None:
        snapshot = self.backend.snapshot()
        if not isinstance(snapshot, AnnualLoadSnapshot):
            raise ValueError("Annual watershed backend snapshot must be AnnualLoadSnapshot")
        state = snapshot.validated(self.domain)
        self._publish({
            "model__parcel_count": len(self.domain.parcel_ids),
            "model__pollutant_count": len(self.domain.pollutants),
            "model__outlet_count": len(self.domain.outlet_oids),
            "model__parcel_baseline_load_rate": state.baseline_rates.ravel(order="C").copy(),
            "model__parcel_current_load_rate": state.current_rates.ravel(order="C").copy(),
            "model__outlet_removed_mass": state.outlet_removed_kg.ravel(order="C").copy(),
        })

    def initialize(self, config_file: str = "") -> None:
        if self._initialized:
            raise RuntimeError("External BMI watershed model is already initialized")
        if config_file:
            raise ValueError("External backend is configured at construction, not through a project YAML")
        self.backend.initialize(self.rng)
        # Validate the scientific baseline before advertising a live interface.
        snapshot = self.backend.snapshot()
        if not isinstance(snapshot, AnnualLoadSnapshot):
            raise ValueError("Annual watershed backend snapshot must be AnnualLoadSnapshot")
        snapshot.validated(self.domain)
        self._initialized = True
        self._time = self._end_time = 0.0
        self._received.clear()
        self._pending = False
        self._cost_received = False
        self._publish({
            **index_contract_values(self.domain),
            "coupling__lifecycle_schema_version": LIFECYCLE_SCHEMA_VERSION,
            **{k: v.copy() for k, v in self.domain.domain_values().items()},
            "coupling__model_rng_state_utf8": encode_rng_state(self.rng),
            "coupling__generator_rng_state_utf8": [],
            **{name: ([] if meta.rank else 0) for name, meta in self._metadata.items()
               if meta.intent == "input"},
            "scenario__identifier": 1,
            "placement__recovery_effect_scale": 1.0,
            "placement__cost": 0.0,
            "placement__recovery_gains": [],
            "model__last_bmp_record_utf8": [],
        })
        self._publish_snapshot()

    def set_value(self, name: str, src: np.ndarray) -> None:
        if not self._initialized:
            raise RuntimeError("Call initialize() before setting BMI variables")
        if name in PLACEMENT_FIELDS and self._pending:
            raise RuntimeError("Finish pending BMP response before transferring another placement")
        if name == "scenario__identifier" and self._pending:
            raise RuntimeError("Cannot change scenario identifier during BMP response")
        if name == "placement__cost":
            if not self._pending:
                raise RuntimeError("BMP cost must follow scientific response")
            cost = self._source(src, "float64")
            if cost.size != 1 or cost[0] < 0:
                raise ValueError("Annual BMI BMP cost must be finite and nonnegative")
        if name == "coupling__generator_rng_state_utf8":
            # Validate before mutating either BMI buffers or the RNG.
            state = decode_rng_state(src, self.rng)
        super().set_value(name, src)
        if name in PLACEMENT_FIELDS:
            self._received.add(name)
            if len(self._received) == len(PLACEMENT_FIELDS):
                self._end_time = self._time + 1.0
        elif name == "coupling__generator_rng_state_utf8":
            self.rng.bit_generator.state = state
        elif name == "placement__cost":
            self._cost_received = True

    def update(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")
        if self._pending:
            raise RuntimeError("Complete the current BMP response before update()")
        missing = set(PLACEMENT_FIELDS) - self._received
        if missing:
            raise RuntimeError(f"Annual BMI BMP request incomplete: {sorted(missing)}")
        placement = AnnualBMPPlacement.from_bmi(self._values, self.domain)
        sid = int(self._read("scenario__identifier")[0])
        if sid < 1:
            raise ValueError("Annual BMI scenario identifier must be positive")
        gains = np.asarray(self.backend.apply_bmp(placement, sid, self.rng), dtype=np.float64)
        if (gains.ndim != 1 or not np.isfinite(gains).all() or
                (gains < 0).any() or (gains > 1).any()):
            raise ValueError("Annual watershed backend recovery gains must lie in [0, 1]")
        self._pending = True
        self._cost_received = False
        self._received.clear()
        self._time += 1.0
        self._publish_snapshot()
        self._publish({
            "placement__recovery_gains": gains,
            "coupling__model_rng_state_utf8": encode_rng_state(self.rng),
        })

    def complete_pending(self, cost_usd: float | None = None) -> dict:
        if not self._pending:
            raise RuntimeError("No pending annual BMI scientific response")
        if cost_usd is None:
            if not self._cost_received:
                raise RuntimeError("Transfer placement__cost before completing the model response")
            cost_usd = float(self._read("placement__cost")[0])
        if not np.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("Annual BMI BMP cost must be finite and nonnegative")
        record = dict(self.backend.complete_bmp(float(cost_usd)))
        encoded = encode_bmp_record(record)
        self._publish_snapshot()
        self._pending = False
        self._cost_received = False
        self._publish({
            "placement__recovery_gains": [],
            "model__last_bmp_record_utf8": encoded,
            "coupling__model_rng_state_utf8": encode_rng_state(self.rng),
        })
        return record

    def finalize(self) -> None:
        if self._initialized:
            self.backend.finalize()
        super().finalize()
        self._received.clear()
        self._pending = False
        self._cost_received = False
