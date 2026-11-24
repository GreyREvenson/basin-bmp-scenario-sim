"""BMI 2.0 facade for the complete stochastic Monte Carlo driver.

``update`` executes one independent scenario using the existing driver
worker and writes its regular output artifacts. This adapter is deliberately
serial; the native driver retains parallel execution.  Scenario position is
logical, not physical time (BMI time units are ``none``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import SeedSequence

from basin_bmp_sim.domain.constants import DATA_N_SCENARIOS, CFG_OUTPUTS, XAXIS_COUNT, YAXIS_TOTAL
from basin_bmp_sim.interfaces.bmi.array_component import BmiArrayComponent
from basin_bmp_sim.interfaces.exchange_variables import DRIVER_VARIABLES
from basin_bmp_sim.interfaces.domain_index_contract import index_contract_values
from basin_bmp_sim.scenarios.generator import BMPScenario


class MonteCarloDriverBmi(BmiArrayComponent):
    """Public BMI over the existing stochastic execution engine."""

    def __init__(self, prepared_model: Any = None, *, replay_dir: str | Path | None = None):
        super().__init__(DRIVER_VARIABLES, name="Basin BMP stochastic Monte Carlo driver")
        self._prepared_model = prepared_model
        self._model = None
        self._replay_dir = None if replay_dir is None else Path(replay_dir)
        self._child_seeds = []
        self._results = []
        self._outputs_dir: Path | None = None
        self._owns_logger = False

    def initialize(self, config_file: str = "") -> None:
        if self._initialized:
            raise RuntimeError("BMI driver already initialized")
        if self._prepared_model is not None:
            if config_file:
                raise ValueError("Do not provide a config file with prepared_model")
            self._model = self._prepared_model
        else:
            if not config_file:
                raise ValueError("Initialize the BMI driver with a YAML config path")
            from basin_bmp_sim.common.io import read_config
            from basin_bmp_sim.config.loading import load_and_validate_all
            from basin_bmp_sim.config.settings import normalize_config, resolve_config_paths
            from basin_bmp_sim.config.validation.configuration import validate_config
            from basin_bmp_sim.common.logging import make_logger
            from basin_bmp_sim.execution.monte_carlo_driver import Model
            cfg_path = Path(config_file)
            cfg = normalize_config(read_config(cfg_path))
            resolve_config_paths(cfg, cfg_path)
            validate_config(cfg)
            output = Path(cfg[CFG_OUTPUTS])
            output.mkdir(parents=True, exist_ok=True)
            logger, _ = make_logger(output, verbose=bool(cfg.get("verbose", False)), console=False)
            data = load_and_validate_all(cfg, logger)
            self._model = Model(cfg, data, logger)
            self._owns_logger = True
        model = self._model
        self._outputs_dir = Path(model.cfg[CFG_OUTPUTS])
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        n_scenarios = int(model.data[DATA_N_SCENARIOS])
        if n_scenarios < 0:
            raise ValueError("Scenario count must be nonnegative")
        base_seed = model.data.get("random_seed")
        children = SeedSequence(base_seed if base_seed is not None else None).spawn(n_scenarios)
        self._child_seeds = [int(child.generate_state(1)[0]) for child in children]
        self._time = 0.0
        self._end_time = float(n_scenarios)
        self._initialized = True
        self._results = []
        self._publish({
            **index_contract_values(model),
            "ensemble__completed_scenario_count": 0,
            "ensemble__requested_scenario_count": n_scenarios,
            "scenario__bmp_count": 0,
            "scenario__total_cost": 0.0,
            "scenario__outlet_removed_mass": np.zeros(
                len(model.pollutants) * len(model.outlet_oids), dtype=float
            ),
        })

    def update(self) -> None:
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")
        if self._time >= self._end_time:
            raise RuntimeError("All scenarios have already been executed")
        from basin_bmp_sim.execution.monte_carlo_driver import _run_one_scenario
        model = self._model
        sidx = int(self._time)
        sid = sidx + 1
        replay = (BMPScenario.load(self._replay_dir / f"s{sid}.json")
                  if self._replay_dir is not None else None)
        result = _run_one_scenario(
            model._shared_payload(), model.cfg, sidx, self._child_seeds[sidx],
            self._outputs_dir, replay,
        )
        self._results.append(result)
        scenario = BMPScenario.load(self._outputs_dir / "scenario_definitions" / f"s{sid}.json")
        delivered = [
            float(result.get((pol, oid, XAXIS_COUNT, YAXIS_TOTAL), [(sid, 0, 0)])[-1][2])
            for pol in model.pollutants for oid in model.outlet_oids
        ]
        self._time += 1.0
        self._publish({
            "ensemble__completed_scenario_count": sid,
            "scenario__bmp_count": len(scenario.placements),
            "scenario__total_cost": sum(float(p.cost_usd) for p in scenario.placements),
            "scenario__outlet_removed_mass": delivered,
        })

    def finalize(self) -> None:
        if self._initialized and self._results:
            from basin_bmp_sim.results.trajectory_output_manager import collect_and_write_trajectories
            from basin_bmp_sim.common.io import _write_parquet_atomic
            collect_and_write_trajectories(
                self._results, self._outputs_dir, self._model.logger,
                write_parquet=_write_parquet_atomic,
            )
        super().finalize()
        self._results = []
        self._model = None
        self._child_seeds = []
