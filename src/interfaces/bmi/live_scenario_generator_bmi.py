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
from src.scenarios.stochastic_scenario_generator import StochasticScenarioGenerator


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
        self._initialized = True
        self._publish({
            "placement__cps_code": 0,
            "placement__host_parcel_index": 0,
            "placement__affected_parcel_index": [],
            "placement__affected_parcel_fraction": [],
            "placement__efficiency": [],
            "placement__failed": 0,
            "placement__recovery_effect_scale": 1.0,
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
            raise RuntimeError("No remaining BMP placements")
        placement = self._generator.next_realization(self._ctx)
        if placement is None:
            self._end_time = self._time
            raise RuntimeError("No eligible BMP placements remain")
        self._pending = placement
        self._time += 1.0
        self._publish({
            "placement__cps_code": placement.choice.cps,
            "placement__host_parcel_index": placement.choice.parcel_index,
            "placement__affected_parcel_index": [i for i, f in placement.design.treated],
            "placement__affected_parcel_fraction": [f for i, f in placement.design.treated],
            "placement__efficiency": [float(v) for pol in placement.efficiency_by_pollutant
                                      for _, v in pol],
            "placement__failed": int(placement.failed),
            "placement__recovery_effect_scale": placement.recovery_effect_scale,
            "placement__completed": 0,
            "placement__cost": 0.0,
        })

    def complete_pending(self, *, recovery_gains: tuple[float, ...] = ()):
        if self._pending is None:
            raise RuntimeError("No pending BMP placement")
        result = self._generator.complete_placement(
            self._ctx, self._pending, recovery_gains=recovery_gains,
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
