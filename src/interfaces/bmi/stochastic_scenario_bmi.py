"""BMI 2.0 facade for completed stochastic BMP scenario definitions.

This facade exposes the persisted/replayable *scenario stream*, not the
interleaved live placement sampler.  Generating complete scenarios ahead of
model execution would perturb the legacy random-number stream (especially for
AFO recovery followed by cost sampling).  The live generator remains the
engine's native Python component until a later coupling phase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.interfaces.bmi.array_component import BmiArrayComponent
from src.interfaces.exchange_variables import SCENARIO_VARIABLES, scenario_arrays
from src.scenarios.stochastic_scenario_generator import BMPScenario


class StochasticScenarioBmi(BmiArrayComponent):
    """Expose one completed BMPScenario per BMI logical update."""

    def __init__(self, scenarios: Sequence[BMPScenario] | None = None):
        super().__init__(SCENARIO_VARIABLES, name="Basin BMP recorded stochastic scenario generator")
        self._provided = None if scenarios is None else tuple(scenarios)
        self._scenarios: tuple[BMPScenario, ...] = ()
        self.current_scenario: BMPScenario | None = None

    def initialize(self, config_file: str = "") -> None:
        if self._initialized:
            raise RuntimeError("BMI scenario component is already initialized")
        if self._provided is not None:
            if config_file:
                raise ValueError("In-memory scenarios cannot be combined with a scenario path")
            scenarios = self._provided
        else:
            if not config_file:
                raise ValueError("Initialize with a saved BMPScenario JSON file or directory")
            path = Path(config_file)
            if path.is_dir():
                paths = sorted(path.glob("s[0-9]*.json"), key=lambda p: int(p.stem[1:]))
            else:
                paths = [path]
            scenarios = tuple(BMPScenario.load(p) for p in paths)
        self._scenarios = tuple(scenarios)
        self._time = 0.0
        self._end_time = float(len(scenarios))
        self.current_scenario = None
        self._initialized = True
        # Valid output metadata also before the first realization is read.
        self._publish({
            "scenario__identifier": 0,
            "scenario__random_seed": 0,
            "scenario__bmp_count": 0,
            "bmp__cps_code": [],
            "bmp__host_parcel_index": [],
            "bmp__implementation_cost": [],
            "bmp__affected_parcel_offset": [0],
            "bmp__affected_parcel_index": [],
            "bmp__affected_parcel_fraction": [],
        })

    def update(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")
        if self._time >= self._end_time:
            raise RuntimeError("No remaining saved scenarios")
        self.current_scenario = self._scenarios[int(self._time)]
        self._publish(scenario_arrays(self.current_scenario))
        self._time += 1.0

    def finalize(self) -> None:
        super().finalize()
        self.current_scenario = None
        self._scenarios = ()
