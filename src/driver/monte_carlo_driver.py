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

from src.bmp import (
    _apply_pathway_reduction,
    _get_bmp_name,
    _get_current_total_load_rate,
    _get_pathway_load_rates,
    _get_bmp_selection_probs,
    _sample_efficiency_map,      # per-pathway sampler
    _select_bmp_type,
    _simulate_grassed,
    _simulate_infield,
    _simulate_wetland,
)
from src.scenarios.stochastic_scenario_generator import (
    BMPScenario, StochasticScenarioGenerator,
)
from src.formulations.formulation_registry import validate_formulation_selection
from src.cost import _estimate_costs_for_probabilities, _get_bmp_cost, _select_cost_rate_median
from src.afo import feedlot_bmp_cost_quantities
from src.io_utils import _write_parquet_atomic, _flatten_plot_records
from src.input_distributions import sample_group_key
from src.logging_utils import make_worker_logger, log_scope
from src.parcel import (
    _get_delivery_coeffs,
    _get_parcel_metadata,
    _get_parcel_out_oids,
    _get_parcel_up_list,
    _sample_parcel_index,
    _sample_load_rate,
)
from src.sampling import _piecewise_quantile_sample, _sample_from_stats, _trunc_normal
from src.driver.results.bmp_summary import BMPSummaryCollector
from src.driver.bmi_coupling import AnnualBmiScenarioCoupler, get_bmi_value
from src.driver.results.trajectory_output_manager import collect_and_write_trajectories
from src.models.model_context import ModelContextServices, ScenarioModelContext
from src.models.assemblies.annual_model_realization import AnnualModelRealization
from src.models.accounting.afo_accounting import (
    build_operation_budget_rows, build_load_component_rows,
)
from src.models.diagnostics.load_parameter_records import model_parameter_records
from src.constants import (
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


# Compatibility aliases for older direct imports through ``src.model``.
from src.models.accounting.bmp_mass import (
    _safe_mass_ratio, _bmp_impacted_parcel_indices,
    _removed_mass_by_parcel_kg, _baseline_mass_before_bmp_kg,
    _add_mass_metrics_to_bmp_record,
)
from src.models.routing.outlet_delivery import _accumulate_delivered_bmp_reductions
from src.scenarios.deterministic_selection import _next_deterministic_bmp_type



class Model(ModelContextServices):
    """Main controller for the simulation.

    The model prepares data, binds helper methods, runs one or more scenarios,
    and writes the output files consumed by summary and plotting workflows.
    """

    def __init__(self, cfg: Dict[str, Any], data: Dict[str, Any], logger: logging.Logger) -> None:
        """Initialize the model and bind helper methods.

        Parameters
        ----------
        cfg : dict[str, Any]
            Scenario configuration mapping.
        data : dict[str, Any]
            Validated input data bundle.
        logger : logging.Logger
            Logger used for model-level messages.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If configured pathway fractions are invalid.
        """
        if cfg.get("simulation") is not None:
            # Direct Python callers receive the same fail-fast formulation check.
            selection_cfg = dict(cfg)
            generation = dict(selection_cfg.get("load_generation") or {})
            generation.setdefault("mode", (data.get("load_generation") or {}).get(
                "mode", data.get("load_generation_mode", "statistical")))
            selection_cfg["load_generation"] = generation
            validate_formulation_selection(selection_cfg)
        self.cfg = cfg
        self.data = data
        self.logger = logger
        seed = data[DATA_RANDOM_SEED]
        self.rng = np.random.default_rng(seed)
        self.outputs_dir: Optional[Path] = None

        # Pathway definitions and aggregate-load-rate fractions are validated in
        # input_config because statistical mode may use arbitrary pathway names.

        self._prepare_lookup_tables()

    def _prepare_lookup_tables(self) -> None:
        """Build lookup tables used during scenario execution.

        The method derives parcel indices, outlet mappings, selection
        probabilities, BMP effectiveness structures, and load-generation
        configuration values from the validated input data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If parcel IDs are duplicated or required input tables are missing.
        """
        with log_scope(label="prepare_lookup_tables", logger=self.logger):
            parcels = self.data[DATA_PARCELS]
            self.parcel_ids = parcels["pid"].astype(str).tolist()
            if len(set(self.parcel_ids)) != len(self.parcel_ids):
                pid_series = pd.Series(self.parcel_ids)
                dup_pids = sorted(pid_series[pid_series.duplicated()].unique().tolist())
                raise ValueError(f"Duplicate parcel IDs found in loaded parcels: {dup_pids}")
            self.pid_to_index = {pid: idx for idx, pid in enumerate(self.parcel_ids)}
            self.pollutants = list(self.data[DATA_POLLUTANTS])
            self.pollutant_to_index = {p: i for i, p in enumerate(self.pollutants)}
            self.parcel_area_ha = parcels["area_ha"].astype(float).tolist()
            self.parcel_perim_m = parcels["perim_m"].astype(float).tolist()

            # Parcel outlet and up-gradient mappings
            po_map = self.data[DATA_PARCEL_OUT_MAP]
            self.parcel_out_oids = [[str(x) for x in po_map[pid]] for pid in self.parcel_ids]
            pd_map = self.data[DATA_PARCEL_DOWN_MAP]
            # parcel_down stores source -> at most one downstream parcel.
            # Reverse these edges in memory to support upstream wetland traversal.
            self.upstream_by_pid = {pid: [] for pid in self.parcel_ids}
            for pid, downstream_pid in pd_map.items():
                if (
                    downstream_pid is not None
                    and pid in self.pid_to_index
                    and downstream_pid in self.upstream_by_pid
                ):
                    self.upstream_by_pid[downstream_pid].append(pid)
            self.parcel_up_idxs = [
                [self.pid_to_index[u] for u in self.upstream_by_pid[pid]]
                for pid in self.parcel_ids
            ]

            # Parcel selection probabilities
            sel = self.data[DATA_PARCEL_P]
            self.parcel_selection_ids = sel["pid"].astype(str).tolist()
            self.parcel_selection_probs = sel["probability"].astype(float).values
            self.selection_source_idxs = [self.pid_to_index[pid] for pid in self.parcel_selection_ids]
            # parcel selection weights define only the BMP-placement universe.  Hydrologic
            # state remains indexed to every modeled parcel so upstream parcels
            # continue to contribute loads to downstream structural practices
            # even when those upstream parcels are not themselves BMP-eligible.

            # Outlet IDs and optional targets/means
            self.outlet_oids = list(self.data[DATA_OUTLET_LOC]["oid"].astype(str).tolist())
            self.outlet_target_map = {}
            if self.data.get(DATA_OUTLET_TARGET) is not None:
                for _, r in self.data[DATA_OUTLET_TARGET].iterrows():
                    self.outlet_target_map[(str(r["oid"]), str(r[COL_POLLUTANT]))] = float(r["target"])
            self.outlet_mean_map = {}
            if self.data.get(DATA_OUTLET_MEAN) is not None:
                for _, r in self.data[DATA_OUTLET_MEAN].iterrows():
                    self.outlet_mean_map[(str(r["oid"]), str(r[COL_POLLUTANT]))] = float(r["mean"])

            # Delivery coeffs
            self.delivery_coeffs = {}
            if self.data.get("delivery_ratios") is not None:
                for _, r in self.data["delivery_ratios"].iterrows():
                    self.delivery_coeffs[(str(r["pid"]), str(r["oid"]))] = dict(
                        sdr_f_to_s=float(r["sdr_f_to_s"]),
                        sdr_s_to_o=float(r["sdr_s_to_o"]),
                        ndr_f_to_s=float(r["ndr_f_to_s"]),
                        ndr_s_to_o=float(r["ndr_s_to_o"]),
                    )

            # Active pathways are mode-specific and were validated during input loading.
            self.pathway_names = list(self.data.get(DATA_PATHWAYS) or ["surface"])
            self.pollutant_load_rate_pathway_fractions = dict(
                self.data.get(DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS) or {}
            )
            self.pollutant_load_rate_is_aggregate = bool(
                self.data[DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE]
            )

            # Efficiency stats by CPS x pollutant x active pathway.
            self.bmp_cps = sorted(int(c) for c in self.data[DATA_CPS])
            self.bmp_efficiency_stats = {int(c): [None] * len(self.pollutants) for c in self.bmp_cps}
            eff = self.data[DATA_BMP_EFFICIENCY]
            for _, row in eff.iterrows():
                cps_key = int(row["cps"])
                pol_key = self.pollutant_to_index[str(row[COL_POLLUTANT])]
                path = str(row[COL_PATHWAY]).strip().lower()
                stats = {
                    k: row[k]
                    for k in row.index
                    if k not in ("cps", COL_POLLUTANT, COL_PATHWAY) and not pd.isna(row[k])
                }
                entry = self.bmp_efficiency_stats[cps_key][pol_key]
                if entry is None:
                    entry = {}
                    self.bmp_efficiency_stats[cps_key][pol_key] = entry
                entry[path] = stats  # type: ignore[index]

            # Load-generation mode and optional PLET/RUSLE inputs
            self.load_generation = dict(self.data.get(DATA_LOAD_GENERATION) or {})
            self.load_generation_mode = str(
                self.load_generation["mode"]
            ).strip().lower()
            self.plet_inputs = self.data.get(DATA_PLET_INPUTS)
            self.rusle_inputs = self.data.get(DATA_RUSLE_INPUTS)
            self.pollutant_concentrations = self.data.get(DATA_POLLUTANT_CONCENTRATIONS)
            self.groundwater_concentrations = self.data.get(DATA_GROUNDWATER_CONCENTRATIONS)
            self.groundwater_loads = self.load_generation_mode == LOAD_MODE_PLET_RUSLE
            self.afo_inputs = self.data.get("afo_inputs")
            self.feedlot_bmp_cps = set(int(c) for c in (self.data.get("feedlot_bmp_cps") or []))

            # Statistical mode supports either explicit pathway-specific load rates
            # distributions or one aggregate_load_rate distribution split by validated
            # user-defined pathway fractions.
            self.pollutant_load_rate_stats = [[None] * len(self.pollutants) for _ in range(len(self.parcel_ids))]
            pollutant_load_rate_table = self.data.get(DATA_POLLUTANT_LOAD_RATE)
            if pollutant_load_rate_table is not None:
                for _, row in pollutant_load_rate_table.iterrows():
                    i = self.pid_to_index[str(row["pid"])]
                    j = self.pollutant_to_index[str(row[COL_POLLUTANT])]
                    stats = {
                        k: row[k]
                        for k in row.index
                        if k not in ("pid", COL_POLLUTANT, COL_PATHWAY) and not pd.isna(row[k])
                    }
                    if self.pollutant_load_rate_is_aggregate:
                        self.pollutant_load_rate_stats[i][j] = stats
                    else:
                        path = str(row[COL_PATHWAY]).strip().lower()
                        entry = self.pollutant_load_rate_stats[i][j]
                        if entry is None:
                            entry = {}
                            self.pollutant_load_rate_stats[i][j] = entry
                        entry[path] = stats  # type: ignore[index]
            elif self.load_generation_mode != LOAD_MODE_PLET_RUSLE:
                raise ValueError("pollutant_load_rate is required in statistical load-generation mode")


            # BMP selection probabilities (possibly via costs or explicit table)
            bmp_probs = self._get_bmp_selection_probs(self.cfg.get(CFG_BMP_SEL))
            self.bmp_cps = bmp_probs["cps"].astype(int).tolist()
            self.bmp_selection_probs = bmp_probs["probability"].astype(float).to_numpy()

            # One-line summary (now VERBOSE)
            self.logger.verbose(
                f"Prepared lookup tables: parcels={len(self.parcel_ids)}, "
                f"pollutants={len(self.pollutants)}, bmp_types={len(self.bmp_cps)}"
            )

    def _shared_payload(self) -> Dict[str, Any]:
        """Package shared data for worker processes.

        Returns
        -------
        dict[str, Any]
            Serializable data bundle containing the model state needed by
            worker processes.
        """
        pollutant_load_rate_stats = self.pollutant_load_rate_stats
        pollutant_load_rate_pathway_fractions = dict(self.pollutant_load_rate_pathway_fractions or {})
        pollutant_load_rate_is_aggregate = bool(self.pollutant_load_rate_is_aggregate)

        return dict(
            cfg=self.cfg,
            data=self.data,
            # Full hydrologic parcel universe.  Scenario load arrays, routing,
            # and upstream relationships use these indices.
            parcel_ids=self.parcel_ids,
            pid_to_index=self.pid_to_index,
            pollutants=self.pollutants,
            parcel_area_ha=np.asarray(self.parcel_area_ha, dtype=float),
            parcel_perim_m=np.asarray(self.parcel_perim_m, dtype=float),
            parcel_out_oids=self.parcel_out_oids,
            parcel_up_idxs=self.parcel_up_idxs,
            # BMP placement remains restricted to the parcel-selection universe.  The
            # selection position is translated to a full parcel index through
            # selection_source_idxs inside the scenario loop.
            parcel_selection_ids=self.parcel_selection_ids,
            parcel_selection_probs=np.asarray(self.parcel_selection_probs, dtype=float),
            selection_source_idxs=np.asarray(self.selection_source_idxs, dtype=int),
            outlet_oids=self.outlet_oids,
            outlet_target_map=self.outlet_target_map,
            outlet_mean_map=self.outlet_mean_map,
            delivery_coeffs=self.delivery_coeffs,
            bmp_efficiency_stats=self.bmp_efficiency_stats,
            pollutant_load_rate_stats=pollutant_load_rate_stats,
            load_generation=self.load_generation,
            load_generation_mode=self.load_generation_mode,
            plet_inputs=self.plet_inputs,
            rusle_inputs=self.rusle_inputs,
            pollutant_concentrations=self.pollutant_concentrations,
            groundwater_concentrations=self.groundwater_concentrations,
            pathway_names=self.pathway_names,
            pollutant_load_rate_pathway_fractions=pollutant_load_rate_pathway_fractions,
            pollutant_load_rate_is_aggregate=pollutant_load_rate_is_aggregate,
            groundwater_loads=self.groundwater_loads,
            # AFO support is optional.  Some lightweight Model objects (including
            # hydrologic-universe test fixtures) are constructed with __new__ and
            # intentionally omit optional attributes.  Match _ScenarioContext's
            # defaults instead of requiring an AFO input when none is configured.
            afo_inputs=getattr(self, "afo_inputs", None),
            feedlot_bmp_cps=set(getattr(self, "feedlot_bmp_cps", None) or ()),
            bmp_cps=self.bmp_cps,
            bmp_selection_probs=self.bmp_selection_probs,
            avg_area_ha=self.data[DATA_AVG_AREA_HA],
            avg_perim_m=self.data[DATA_AVG_PERIM_M],
            random_seed=self.data.get("random_seed"),
        )

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