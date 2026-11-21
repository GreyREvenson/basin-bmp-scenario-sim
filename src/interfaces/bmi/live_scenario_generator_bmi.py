"""BMI facade for the *live* BMP scenario sampler.

One update samples exactly one BMP placement design. The scientific model must
apply its response before ``complete_pending`` draws the associated BMP cost;
this preserves the legacy AFO -> cost random-number order. Native access to
rich design/response attributes remains an explicit extension, not yet a
pure BMI exchange contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.interfaces.bmi.array_component import BmiArrayComponent
from src.interfaces.exchange_variables import LIVE_GENERATOR_VARIABLES
from src.interfaces.placement_exchange import placement_arrays
from src.scenarios.stochastic_scenario_generator import StochasticScenarioGenerator


class NoEligiblePlacements(RuntimeError):
    """The approved generator has exhausted its parcel/CPS candidates."""


class LiveScenarioGeneratorBmi(BmiArrayComponent):
    """Expose sampled BMP placements without pre-generating complete scenarios."""

    def __init__(self, ctx: Any, cfg: dict, *, scenario_id: int = 1,
                 seed: int | None = None, replay=None):
        super().__init__(LIVE_GENERATOR_VARIABLES,
                         name="Basin BMP live stochastic scenario generator")
        self._ctx = ctx
        self._cfg = cfg
        self._seed = seed
        self._scenario_id = int(scenario_id)
        self._replay = replay
        self._generator: StochasticScenarioGenerator | None = None
        self._pending = None
        self._completed = False
        self._recovery_received = False

    def initialize(self, config_file: str = "") -> None:
        if config_file:
            raise ValueError("Live generator requires a prepared context, not a config file")
        if self._initialized:
            raise RuntimeError("BMI live generator is already initialized")
        self._generator = StochasticScenarioGenerator(self._ctx, self._cfg,
                                                       replay=self._replay)
        # The generator's worker seed is controlled by its prepared context.
        if self._replay is not None:
            if self._seed is None:
                raise ValueError("Replay requires an explicit worker seed")
            self._generator.validate_replay(self._ctx, self._scenario_id, self._seed)
        self._time = 0.0
        # Maximum possible number of host/CPS pairs. Budget and other
        # stopping criteria may terminate earlier; do not interpret this as
        # a promised number of completed BMPs.
        self._end_time = float(len(self._ctx.parcel_selection_ids) * len(self._ctx.bmp_cps))
        self._pending = None
        self._completed = False
        self._recovery_received = False
        self._initialized = True
        self._publish({
            "placement__cps_code": 0,
            "placement__host_parcel_index": 0,
            "placement__affected_parcel_index": [],
            "placement__affected_parcel_fraction": [],
            "placement__efficiency": [],
            "placement__failed": 0,
            "placement__recovery_effect_scale": 1.0,
            "placement__cost_quantity_ha": 0.0,
            "placement__wetland_area_ha": 0.0,
            "placement__catchment_area_ratio": 0.0,
            "placement__buffer_length_m": 0.0,
            "placement__buffer_area_ha": 0.0,
            "placement__portion_treated": 0.0,
            "placement__recorded_recovery_gains": [],
            "placement__recovery_gains": [],
            "placement__completed": 0,
            "placement__cost": 0.0,
            "scenario__bmp_count": 0,
            "scenario__cumulative_cost": 0.0,
        })

    @property
    def pending_placement(self):
        if self._pending is None:
            raise RuntimeError("No pending BMP placement")
        return self._pending

    def update(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")
        if self._pending is not None:
            raise RuntimeError("Complete the pending placement before the next update")
        if self._time >= self._end_time:
            raise NoEligiblePlacements("No remaining BMP placements")
        placement = self._generator.next_realization(self._ctx)
        if placement is None:
            self._end_time = self._time
            raise NoEligiblePlacements("No eligible BMP placements remain")
        self._pending = placement
        self._time += 1.0
        self._recovery_received = False
        self._publish({
            **placement_arrays(placement, self._ctx),
            "placement__recovery_gains": [],
            "placement__completed": 0,
            "placement__cost": 0.0,
        })

    def set_value(self, name: str, src: np.ndarray) -> None:
        if name == "placement__recovery_gains" and self._pending is None:
            raise RuntimeError("Queue a placement before supplying AFO recovery gains")
        super().set_value(name, src)
        if name == "placement__recovery_gains":
            gains = self._read(name)
            if np.any(gains < 0) or np.any(gains > 1):
                raise ValueError("BMI AFO recovery gains must lie in [0, 1]")
            self._recovery_received = True

    def complete_pending(self, *, recovery_gains: tuple[float, ...] | None = None):
        """Complete after BMI recovery transfer; retain Phase 5 legacy override."""
        if self._pending is None:
            raise RuntimeError("No pending BMP placement")
        if recovery_gains is None:
            if not self._recovery_received:
                raise RuntimeError("Supply placement__recovery_gains before completing a BMI placement")
            recovery_gains = tuple(float(v) for v in self._read("placement__recovery_gains"))
        result = self._generator.complete_placement(
            self._ctx, self._pending, recovery_gains=tuple(recovery_gains),
        )
        self._pending = None
        self._publish({
            "placement__completed": 1,
            "placement__cost": result.cost_usd,
            "scenario__bmp_count": self._generator.total_bmp,
            "scenario__cumulative_cost": self._generator.total_cost,
        })
        return result

    def completed_scenario(self, seed: int):
        if self._pending is not None:
            raise RuntimeError("Cannot complete scenario with a pending BMP")
        if not self._initialized:
            raise RuntimeError("Call initialize() before completing a scenario")
        return self._generator.complete_scenario(
            self._ctx, self._scenario_id, int(seed),
        )

    def finalize(self) -> None:
        if self._pending is not None:
            raise RuntimeError("Cannot finalize with pending BMP response")
        super().finalize()
        self._generator = None
