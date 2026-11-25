"""Run the simulation workflow and write scenario output files.

This module coordinates scenario execution, applies BMPs, collects summary
outputs, and writes the per-scenario and cross-scenario result tables used by
the rest of the application.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import SeedSequence, default_rng

from basin_bmp_sim.bmp.pathways.plet_pathway_efficiency import _apply_pathway_reduction, _get_current_total_load_rate, _get_pathway_load_rates
from basin_bmp_sim.bmp.sampling.selection import _get_bmp_name, _get_bmp_selection_probs, _select_bmp_type
from basin_bmp_sim.bmp.sampling.efficiency import _sample_efficiency_map
from basin_bmp_sim.bmp.grassed_waterway.response.plet_pathway_efficiency import _simulate_grassed
from basin_bmp_sim.bmp.infield.response.plet_pathway_efficiency import _simulate_infield
from basin_bmp_sim.bmp.wetland.response.plet_pathway_efficiency import _simulate_wetland
from basin_bmp_sim.scenarios.generator import (
    BMPScenario, StochasticScenarioGenerator,
)
from basin_bmp_sim.science.formulations.formulation_registry import validate_formulation_selection
from basin_bmp_sim.scenarios.costs import _estimate_costs_for_probabilities, _get_bmp_cost, _select_cost_rate_median
from basin_bmp_sim.science.afo.feedlot import feedlot_bmp_cost_quantities
from basin_bmp_sim.common.io import _write_parquet_atomic, _flatten_plot_records
from basin_bmp_sim.common.distributions import sample_group_key
from basin_bmp_sim.common.logging import make_worker_logger, log_scope
from basin_bmp_sim.domain.parcel import (
    _get_delivery_coeffs,
    _get_parcel_metadata,
    _get_parcel_out_oids,
    _get_parcel_up_list,
    _sample_parcel_index,
    _sample_load_rate,
)
from basin_bmp_sim.scenarios.sampling import _piecewise_quantile_sample, _sample_from_stats, _trunc_normal
from basin_bmp_sim.results.bmp_summary import BMPSummaryCollector
from basin_bmp_sim.execution.bmi_coupling import AnnualBmiScenarioCoupler, get_bmi_value
from basin_bmp_sim.results.trajectory_output_manager import collect_and_write_trajectories
from basin_bmp_sim.science.model_context import ScenarioModelContext
from basin_bmp_sim.science.prepared_inputs import PreparedModelInputs
from basin_bmp_sim.science.assemblies.annual_model_realization import AnnualModelRealization
from basin_bmp_sim.science.accounting.afo_accounting import (
    build_operation_budget_rows, build_load_component_rows,
)
from basin_bmp_sim.science.diagnostics.load_parameter_records import model_parameter_records
from basin_bmp_sim.domain.constants import (
    CURRENT_TIMESTEP_YEARS as _CURRENT_TIMESTEP_YEARS,
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_DETERMINISTIC_BMP_TYPE_ORDER,
    CFG_DETERMINISTIC_PARCEL_ORDER,
    CFG_OUTPUTS,
    CFG_PARALLEL,
    CFG_VERBOSE,  # pass verbose to worker loggers
    # Failure config keys
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    DEFAULT_BMP_FAIL_REDUCTION,
    # Outputs
    OUTPUT_PORTION_TREATED,
    OUTPUT_BMP_FAILED,
    COL_POLLUTANT,
    COL_PATHWAY,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    DATA_AVG_AREA_HA,
    DATA_AVG_PERIM_M,
    DATA_BMP_COST,
    DATA_BMP_EFFICIENCY,
    DATA_BMP_LIMIT_N,
    DATA_BMP_LIMIT_USD,
    DATA_CPS,
    DATA_DELIVERY_RATIOS,
    DATA_N_SCENARIOS,
    DATA_RANDOM_SEED,
    DATA_OUTLET_LOC,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_PARCEL_OUT_MAP,
    DATA_PARCEL_P,
    DATA_PARCEL_DOWN_MAP,
    DATA_PARCELS,
    DATA_POLLUTANT_LOAD_RATE,
    DATA_LOAD_GENERATION,
    DATA_PLET_INPUTS,
    DATA_RUSLE_INPUTS,
    DATA_POLLUTANT_CONCENTRATIONS,
    DATA_GROUNDWATER_CONCENTRATIONS,
    DATA_PATHWAYS,
    DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE,
    LOAD_GROUNDWATER_LOADS,
    LOAD_MODE_PLET_RUSLE,
    LOAD_MODE_STATISTICAL,
    DATA_POLLUTANTS,
    DIR_OUTLET_TRAJECTORIES,
    DIR_SCENARIO_METRICS,
    FILE_ALL_SCENARIOS_PARQUET,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_COST_USD,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
    OUTPUT_WETLAND_AREA,
    XAXIS_COST,
    XAXIS_COUNT,
    YAXIS_MEAN,
    YAXIS_TARGET,
    YAXIS_TOTAL,
)


# Scientific accounting symbols used by this driver.
from basin_bmp_sim.science.accounting.bmp_mass import (
    _safe_mass_ratio, _bmp_impacted_parcel_indices,
    _removed_mass_by_parcel_kg, _baseline_mass_before_bmp_kg,
    _add_mass_metrics_to_bmp_record,
)
from basin_bmp_sim.science.routing.outlet_delivery import _accumulate_delivered_bmp_reductions
from basin_bmp_sim.scenarios.deterministic_selection import _next_deterministic_bmp_type



class Model(PreparedModelInputs):
    """Monte Carlo driver: inherits validated lookup and worker input preparation."""

    def run_all_scenarios(self, *, replay_dir: Path | str | None = None) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
        """Run all scenarios and return cross-scenario trajectory data.

        The method spawns workers when configured, executes each scenario,
        merges their plot records, and writes the canonical outlet trajectory
        parquet file.

        Returns
        -------
        dict[tuple[str, str, str, str], list[tuple[int, float, float]]]
            Plot records keyed by pollutant, outlet, x-axis, and y-axis.
        """
        outputs_dir = Path(self.cfg[CFG_OUTPUTS])
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir

        n_scenarios = int(self.data[DATA_N_SCENARIOS])
        parallel = dict(self.cfg.get(CFG_PARALLEL) or {})
        n_jobs = int(parallel["n_jobs"])

        shared = self._shared_payload()
        base_seed = self.data.get("random_seed")
        ss = SeedSequence(base_seed if base_seed is not None else None)
        child_seeds = ss.spawn(n_scenarios)

        # Load and validate every requested replay artifact before launching
        # workers; no scenario is silently regenerated if a file is missing.
        if replay_dir is None:
            replay_scenarios = [None] * n_scenarios
        else:
            replay_root = Path(replay_dir)
            replay_scenarios = [
                BMPScenario.load(replay_root / f"s{sidx + 1}.json")
                for sidx in range(n_scenarios)
            ]
        self.logger.info(f"Running {n_scenarios} scenario(s) with n_jobs={n_jobs}")
        func = delayed(_run_one_scenario)
        results = Parallel(n_jobs=n_jobs)(
            func(shared, self.cfg, sidx, int(child_seeds[sidx].generate_state(1)[0]),
                 outputs_dir, replay_scenarios[sidx])
            for sidx in range(n_scenarios)
        )

        # Results are collected and written by the driver's output manager.
        return collect_and_write_trajectories(
            results, outputs_dir, self.logger, write_parquet=_write_parquet_atomic
        )


# Historical name retained for external callers and regression fixtures.
_ScenarioContext = ScenarioModelContext


def _run_one_scenario(
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
    sidx: int,
    seed: int,
    outputs_dir: Path,
    scenario_definition: BMPScenario | None = None,
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Run one scenario and write its output files.

    Parameters
    ----------
    shared : dict[str, Any]
        Shared model payload used to construct the worker context.
    cfg : dict[str, Any]
        Scenario configuration mapping.
    sidx : int
        Zero-based scenario index.
    seed : int
        Seed for the worker-specific random number generator.
    outputs_dir : pathlib.Path
        Directory where scenario outputs should be written.

    Returns
    -------
    dict[tuple[str, str, str, str], list[tuple[int, float, float]]]
        Plot records for the scenario.
    """
    sid = sidx + 1
    logger = make_worker_logger(
        outputs_dir,
        scenario_id=sid,
        verbose=bool(cfg[CFG_VERBOSE]),
    )
    ctx = _ScenarioContext(cfg, shared, logger, seed)

    with log_scope(label=f"scenario {sid}", logger=logger, level=logging.INFO):
        logger.info(f"=== scenario {sid} start ===")

        # Phase 6: scientific inputs/outputs exchange through BMI numerical
        # buffers. The native model object remains available only for the
        # existing scientific diagnostic/output-record writers.
        coupler = AnnualBmiScenarioCoupler(
            ctx, cfg, logger, scenario_id=sid, seed=seed, replay=scenario_definition,
        )
        coupler.initialize()
        scientific_model = coupler.model.native_model
        baseline_load_rates = get_bmi_value(
            coupler.model, "model__parcel_baseline_load_rate"
        ).reshape(len(ctx.parcel_ids), len(ctx.pollutants))
        load_state = scientific_model.load_state
        afo_assignments = scientific_model.afo_assignments
        manure_mass_ledger = scientific_model.manure_mass_ledger

        # Limits
        limit_n = cfg.get("bmp_limit_n",None)
        limit_usd = cfg.get("bmp_limit_usd",None)
        if limit_n is None and limit_usd is None:
            logger.warning("No BMP limits specified; setting bmp limit to 1 to prevent indefinite run")
            limit_n = 1
        total_cost = 0.0
        total_bmp = 0

        # Axes and record buffers
        x_axes: List[str] = [XAXIS_COUNT]
        if cfg.get(CFG_BMP_COST):
            x_axes.append(XAXIS_COST)
        y_axes: List[str] = [YAXIS_TOTAL]
        if ctx.outlet_target_map:
            y_axes.append(YAXIS_TARGET)
        if ctx.outlet_mean_map:
            y_axes.append(YAXIS_MEAN)

        records: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
        scenario_bmps: List[Dict[str, Any]] = []
        scenario_parcels: List[Dict[str, Any]] = []

        # Initialize summary collector once per scenario
        collector = BMPSummaryCollector(ctx.pollutants, scenario_id=sid)

        # The generator has already checked replay eligibility during BMI
        # initialization. Its placement values cross the model boundary only
        # through named, checked numerical BMI arrays.

        # Main loop: a parcel is sampled only while at least one CPS is selectable.
        while True:
            if limit_usd is not None and total_cost >= limit_usd:
                logger.info("BMP placement stopped: cost limit reached.")
                break
            if limit_n is not None and total_bmp >= limit_n:
                logger.info("BMP placement stopped: count limit reached.")
                break

            completed = coupler.next_bmp()
            if completed is None:
                logger.info("BMP placement stopped: no selectable parcel/CPS combinations remain.")
                break
            parcel_idx = completed.parcel_index
            pid = str(ctx.parcel_ids[parcel_idx])
            cps = completed.cps
            with log_scope(label=f"apply bmp cps={cps} pid={pid}", logger=logger):
                # AFO gains, physical designs and implementation cost have
                # already crossed the BMI boundary in the scientifically
                # required order.  Collection remains driver-owned.
                total_bmp = completed.completed_count
                total_cost = completed.total_cost_usd
                bmp_rec = completed.bmp_record
                scenario_bmps.append(bmp_rec)
                collector.add_bmp_record(bmp_rec)
                cumul = completed.delivered_reductions
                for pol_idx, pol in enumerate(ctx.pollutants):
                    for oid_idx, oid in enumerate(ctx.outlet_oids):
                        for xax in x_axes:
                            for yax in y_axes:
                                xval = total_bmp if xax == XAXIS_COUNT else total_cost
                                if yax == YAXIS_TOTAL:
                                    yval = cumul[pol_idx, oid_idx]
                                elif yax == YAXIS_TARGET:
                                    tgt = ctx.outlet_target_map.get((str(oid), pol))
                                    yval = (cumul[pol_idx, oid_idx] / tgt * 100.0) if tgt is not None and tgt > 0 else 0.0
                                elif yax == YAXIS_MEAN:
                                    mu = ctx.outlet_mean_map.get((str(oid), pol))
                                    yval = (cumul[pol_idx, oid_idx] / mu * 100.0) if mu is not None and mu > 0 else 0.0
                                else:
                                    yval = 0.0
                                records[(pol, oid, xax, yax)].append((sid, xval, yval))

        # The fully realized scenario is a model-independent, replayable artifact.
        # Writing the JSON never samples RNGs or modifies modeled loads.
        coupler.completed_scenario().save(
            outputs_dir / "scenario_definitions" / f"s{sid}.json"
        )

        # Realized operation-to-field allocations, independently auditable for
        # deterministic and stochastic location assumptions.
        if afo_assignments:
            allocation_dir = outputs_dir / "afo_manure_assignments"
            allocation_dir.mkdir(parents=True, exist_ok=True)
            _write_parquet_atomic(
                pd.DataFrame([dict(scenario=sid, **rec) for rec in afo_assignments]),
                allocation_dir / f"s{sid}.parquet", logger=logger,
            )

        if manure_mass_ledger:
            manure_dir = outputs_dir / "afo_mass_ledger"
            manure_dir.mkdir(parents=True, exist_ok=True)
            manure_df = pd.DataFrame([dict(scenario=sid, **rec) for rec in manure_mass_ledger])
            _write_parquet_atomic(manure_df, manure_dir / f"s{sid}.parquet", logger=logger)
            budget_rows = build_operation_budget_rows(ctx, sid, manure_df)
            budgets_dir = outputs_dir / "afo_operation_budgets"
            budgets_dir.mkdir(parents=True, exist_ok=True)
            _write_parquet_atomic(pd.DataFrame(budget_rows), budgets_dir / f"s{sid}.parquet", logger=logger)
            effects = getattr(ctx, "afo_bmp_effect_records", [])
            if effects:
                effects_dir = outputs_dir / "afo_manure_bmp_effects"
                effects_dir.mkdir(parents=True, exist_ok=True)
                _write_parquet_atomic(pd.DataFrame([dict(scenario=sid, **rec) for rec in effects]),
                                      effects_dir / f"s{sid}.parquet", logger=logger)

        # Direct feedlot and manure-field components retained independently
        # through BMP application (kg/year) to enable source-specific auditing.
        if afo_assignments:
            component_rows = build_load_component_rows(ctx, sid)
            component_dir = outputs_dir / "afo_load_components"
            component_dir.mkdir(parents=True, exist_ok=True)
            _write_parquet_atomic(pd.DataFrame(component_rows), component_dir / f"s{sid}.parquet", logger=logger)

        # Parcel-level outputs use the same published BMI snapshots as the
        # public watershed-model adapter; never reach into mutable model
        # arrays to get current scientific values.
        load_rates = get_bmi_value(
            coupler.model, "model__parcel_current_load_rate"
        ).reshape(len(ctx.parcel_ids), len(ctx.pollutants))
        # Parcel-level before/after
        for parcel_idx, pid_i in enumerate(ctx.parcel_ids):
            row = dict(scenario=sid, pid=str(pid_i))
            for pol_idx, pol in enumerate(ctx.pollutants):
                # Explicit areal-load-rate columns are canonical.
                row[f"baseline_load_rate_{pol}_kg_ha_yr"] = float(baseline_load_rates[parcel_idx, pol_idx])
                row[f"final_load_rate_{pol}_kg_ha_yr"] = float(load_rates[parcel_idx, pol_idx])
            scenario_parcels.append(row)

        # The scientific model owns derived load diagnostics; the driver
        # merely collects and persists the model-supplied records.
        scenario_load_parameters = model_parameter_records(ctx, sid, load_state)

        # Build summary tables once, then write parquet outputs.
        summary_df = collector.generate_summary_dataframe()
        rollup = collector.generate_rollup_summary()
        summary_with_rollup = pd.concat([summary_df, pd.DataFrame([rollup])], ignore_index=True)

        # Write per-scenario parquet tables.
        bmps_dir = outputs_dir / "bmps"
        parcels_dir = outputs_dir / "parcels"
        load_parameters_dir = outputs_dir / "load_parameters"
        metrics_dir = outputs_dir / DIR_SCENARIO_METRICS
        output_dirs = [bmps_dir, parcels_dir, metrics_dir]
        if scenario_load_parameters:
            output_dirs.append(load_parameters_dir)
        for d in output_dirs:
            d.mkdir(parents=True, exist_ok=True)

        bmps_parquet = bmps_dir / f"s{sid}.parquet"
        parcels_parquet = parcels_dir / f"s{sid}.parquet"
        metrics_parquet = metrics_dir / f"s{sid}.parquet"
        _write_parquet_atomic(pd.DataFrame(scenario_bmps), bmps_parquet, logger=logger)
        _write_parquet_atomic(pd.DataFrame(scenario_parcels), parcels_parquet, logger=logger)
        _write_parquet_atomic(summary_with_rollup, metrics_parquet, logger=logger)
        if scenario_load_parameters:
            load_parameters_parquet = load_parameters_dir / f"s{sid}.parquet"
            _write_parquet_atomic(pd.DataFrame(scenario_load_parameters), load_parameters_parquet, logger=logger)

        logger.info(f"Wrote per-scenario BMPs parquet: {bmps_parquet}")
        logger.info(f"Wrote per-scenario parcels parquet: {parcels_parquet}")
        if scenario_load_parameters:
            logger.info(f"Wrote realized PLET/RUSLE parameters parquet: {load_parameters_parquet}")
        logger.info(f"Wrote canonical scenario metrics parquet: {metrics_parquet}")
        logger.info(f"=== scenario {sid} end (cost={total_cost:.2f}, bmp={total_bmp}) ===")
    return records