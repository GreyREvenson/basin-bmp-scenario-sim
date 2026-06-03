"""
Scenario orchestration (parallel execution and per-scenario outputs).

Coordinates:
- Preparing lookup structures for fast scenario execution
- Running scenarios in parallel
- Writing per-scenario CSVs and logs
- Producing transposed per-scenario summaries with an "All CPS" column
"""

from __future__ import annotations

import logging
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import SeedSequence, default_rng

from src.bmp import (
    _get_bmp_name,
    _get_bmp_selection_probs,
    _sample_efficiency,          # legacy scalar sampler (kept for backward-compat)
    _sample_efficiency_map,      # per-pathway sampler
    _select_bmp_type,
    _simulate_grassed,
    _simulate_infield,
    _simulate_wetland,
)
from src.cost import _estimate_costs_for_probabilities, _get_bmp_cost, _select_cost_rate_median
from src.logging_utils import make_worker_logger
from src.parcel import (
    _get_delivery_coeffs,
    _get_parcel_metadata,
    _get_parcel_out_oids,
    _get_parcel_up_list,
    _sample_parcel_index,
    _sample_yield,
)
from src.sampling import _piecewise_quantile_sample, _sample_from_stats, _trunc_normal
from src.summaries import BMPSummaryCollector
from src.constants import (
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_OUTPUTS,
    CFG_PARALLEL,
    CFG_POLLUTANT_YIELD_FRAC_SURFACE,
    CFG_POLLUTANT_YIELD_FRAC_SHALLOW,
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
    DATA_OUTLET_LOC,
    DATA_OUTLET_MEAN,
    DATA_OUTLET_TARGET,
    DATA_PARCEL_OUT_MAP,
    DATA_PARCEL_P,
    DATA_PARCEL_UP_MAP,
    DATA_PARCELS,
    DATA_POLLUTANT_YIELD,
    DATA_POLLUTANTS,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_COST_USD,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_REMOVED,
    OUTPUT_REMOVED_PREFIX,
    OUTPUT_TREATED,
    OUTPUT_TREATED_PREFIX,
    OUTPUT_WETLAND_AREA,
    XAXIS_COST,
    XAXIS_COUNT,
    YAXIS_MEAN,
    YAXIS_TARGET,
    YAXIS_TOTAL,
)


class Model:
    """Main simulation orchestrator for running multiple scenarios."""

    def __init__(self, cfg: Dict[str, Any], data: Dict[str, Any], logger: logging.Logger) -> None:
        self.cfg = cfg
        self.data = data
        self.logger = logger
        seed = data.get("random_seed", None)
        self.rng = np.random.default_rng(seed)
        self.outputs_dir: Optional[Path] = None

        # Bind helper functions
        self._sample_from_stats = types.MethodType(_sample_from_stats, self)
        self._piecewise_quantile_sample = types.MethodType(_piecewise_quantile_sample, self)
        self._trunc_normal = types.MethodType(_trunc_normal, self)

        self._select_bmp_type = types.MethodType(_select_bmp_type, self)
        self._get_bmp_name = types.MethodType(_get_bmp_name, self)
        self._sample_efficiency = types.MethodType(_sample_efficiency, self)          # legacy
        self._sample_efficiency_map = types.MethodType(_sample_efficiency_map, self)  # pathway-aware
        self._simulate_wetland = types.MethodType(_simulate_wetland, self)
        self._simulate_grassed = types.MethodType(_simulate_grassed, self)
        self._simulate_infield = types.MethodType(_simulate_infield, self)
        self._get_bmp_selection_probs = types.MethodType(_get_bmp_selection_probs, self)
        self._get_bmp_cost = types.MethodType(_get_bmp_cost, self)

        self._sample_parcel_index = types.MethodType(_sample_parcel_index, self)
        self._sample_yield = types.MethodType(_sample_yield, self)
        self._get_parcel_metadata = types.MethodType(_get_parcel_metadata, self)
        self._get_parcel_up_list = types.MethodType(_get_parcel_up_list, self)
        self._get_parcel_out_oids = types.MethodType(_get_parcel_out_oids, self)
        self._delivery_coeffs = types.MethodType(_get_delivery_coeffs, self)

        self._estimate_costs_for_probabilities = types.MethodType(_estimate_costs_for_probabilities, self)
        self._select_cost_rate_median = types.MethodType(_select_cost_rate_median, self)

        # Validate and store pathway fractions
        surf_frac = float(self.cfg.get(CFG_POLLUTANT_YIELD_FRAC_SURFACE, 0.0))
        shal_frac = float(self.cfg.get(CFG_POLLUTANT_YIELD_FRAC_SHALLOW, 0.0))
        if not (0.0 <= surf_frac <= 1.0 and 0.0 <= shal_frac <= 1.0):
            raise ValueError("pollutant_yield_frac_surface and pollutant_yield_frac_shallow must be in [0,1]")
        if surf_frac + shal_frac > 1.0:
            raise ValueError("pollutant_yield_frac_surface + pollutant_yield_frac_shallow must be <= 1.0")
        self.pollutant_yield_frac_surface = surf_frac
        self.pollutant_yield_frac_shallow = shal_frac

        self._prepare_lookup_tables()

    def _prepare_lookup_tables(self) -> None:
        """Assemble arrays and mappings used during scenario execution."""
        parcels = self.data[DATA_PARCELS]
        self.parcel_ids = parcels["pid"].astype(str).tolist()
        self.pid_to_index = {pid: idx for idx, pid in enumerate(self.parcel_ids)}
        self.pollutants = list(self.data[DATA_POLLUTANTS])
        self.pollutant_to_index = {p: i for i, p in enumerate(self.pollutants)}
        self.parcel_area_ha = parcels["area_ha"].astype(float).tolist()
        self.parcel_perim_m = parcels["perim_m"].astype(float).tolist()

        # Parcel outlet and up-gradient mappings
        po_map = self.data[DATA_PARCEL_OUT_MAP]
        self.parcel_out_oids = [[str(x) for x in po_map.get(pid, [])] for pid in self.parcel_ids]
        pu_map = self.data[DATA_PARCEL_UP_MAP]
        self.parcel_up_idxs = [[self.pid_to_index[u] for u in pu_map.get(pid, []) if u in self.pid_to_index] for pid in self.parcel_ids]

        # Parcel selection probabilities
        sel = self.data[DATA_PARCEL_P]
        self.parcel_selection_ids = sel["pid"].astype(str).tolist()
        self.parcel_selection_probs = sel["probability"].astype(float).values

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

        # Efficiency stats by CPS x pollutant (optionally per pathway)
        self.bmp_cps = sorted(int(c) for c in self.data[DATA_CPS])
        self.bmp_efficiency_stats = {int(c): [None] * len(self.pollutants) for c in self.bmp_cps}
        eff = self.data[DATA_BMP_EFFICIENCY]
        has_pathway = (COL_PATHWAY in eff.columns)

        for _, row in eff.iterrows():
            cps_key = int(row["cps"])
            pol_key = self.pollutant_to_index[str(row[COL_POLLUTANT])]
            stats = {k: row[k] for k in row.index if k not in ("cps", COL_POLLUTANT, COL_PATHWAY)}
            if has_pathway:
                path = str(row.get(COL_PATHWAY, "surface")).strip().lower()
                if self.bmp_efficiency_stats[cps_key][pol_key] is None or not isinstance(self.bmp_efficiency_stats[cps_key][pol_key], dict):
                    self.bmp_efficiency_stats[cps_key][pol_key] = {}
                self.bmp_efficiency_stats[cps_key][pol_key][path] = stats  # type: ignore[index]
            else:
                self.bmp_efficiency_stats[cps_key][pol_key] = stats

        # Yield stats per parcel x pollutant
        pol_y = self.data[DATA_POLLUTANT_YIELD]
        self.pollutant_yield_stats = [[None] * len(self.pollutants) for _ in range(len(self.parcel_ids))]
        for _, row in pol_y.iterrows():
            i = self.pid_to_index[str(row["pid"])]
            j = self.pollutant_to_index[str(row[COL_POLLUTANT])]
            self.pollutant_yield_stats[i][j] = {k: row[k] for k in row.index if k not in ("pid", COL_POLLUTANT)}

        # BMP selection probabilities
        if self.cfg.get(CFG_BMP_SEL):
            probs_df = self._get_bmp_selection_probs(self.cfg.get(CFG_BMP_SEL))
        else:
            if self.data.get(DATA_BMP_COST) is not None:
                probs_df = self._estimate_costs_for_probabilities()
            else:
                probs_df = pd.DataFrame(
                    {"cps": self.bmp_cps, "probability": np.full(len(self.bmp_cps), 1.0 / len(self.bmp_cps))}
                )
        probs_df = probs_df[probs_df["cps"].astype(int).isin(self.bmp_cps)]
        self.bmp_cps = probs_df["cps"].astype(int).tolist()
        self.bmp_selection_probs = probs_df["probability"].astype(float).values

    def _shared_payload(self) -> Dict[str, Any]:
        """Create a read-only payload for worker processes."""
        return dict(
            cfg=self.cfg,
            data=self.data,
            parcel_ids=self.parcel_ids,
            pid_to_index=self.pid_to_index,  # ensure workers have PID->index mapping
            pollutants=self.pollutants,
            parcel_area_ha=np.asarray(self.parcel_area_ha, dtype=float),
            parcel_perim_m=np.asarray(self.parcel_perim_m, dtype=float),
            parcel_out_oids=self.parcel_out_oids,
            parcel_up_idxs=self.parcel_up_idxs,
            parcel_selection_ids=self.parcel_selection_ids,
            parcel_selection_probs=np.asarray(self.parcel_selection_probs, dtype=float),
            outlet_oids=self.outlet_oids,
            outlet_target_map=self.outlet_target_map,
            outlet_mean_map=self.outlet_mean_map,
            delivery_coeffs=self.delivery_coeffs,
            bmp_efficiency_stats=self.bmp_efficiency_stats,
            pollutant_yield_stats=self.pollutant_yield_stats,
            bmp_cps=self.bmp_cps,
            bmp_selection_probs=self.bmp_selection_probs,
            avg_area_ha=self.data.get(DATA_AVG_AREA_HA, 0.0),
            avg_perim_m=self.data.get(DATA_AVG_PERIM_M, 0.0),
            random_seed=self.data.get("random_seed"),
            # Pathway fractions for simulators
            pollutant_yield_frac_surface=self.pollutant_yield_frac_surface,
            pollutant_yield_frac_shallow=self.pollutant_yield_frac_shallow,
        )

    def run_all_scenarios(self) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
        """Run all scenarios (possibly in parallel) and return plotting records."""
        outputs_dir = Path(self.cfg.get(CFG_OUTPUTS, "./outputs"))
        outputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = outputs_dir

        n_scenarios = int(self.data[DATA_N_SCENARIOS])
        parallel = dict(self.cfg.get(CFG_PARALLEL) or {})
        n_jobs = int(parallel.get("n_jobs", 1))

        shared = self._shared_payload()
        base_seed = self.data.get("random_seed")
        ss = SeedSequence(base_seed if base_seed is not None else None)
        child_seeds = ss.spawn(n_scenarios)

        self.logger.info(f"Running {n_scenarios} scenario(s) with n_jobs={n_jobs}")
        func = delayed(_run_one_scenario)
        results = Parallel(n_jobs=n_jobs)(
            func(shared, self.cfg, sidx, int(child_seeds[sidx].generate_state(1)[0]), outputs_dir)
            for sidx in range(n_scenarios)
        )

        # Merge plotting records
        merged: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
        for recs in results:
            for k, v in recs.items():
                merged[k].extend(v)
        return merged


class _ScenarioContext:
    """Lightweight worker context object for a single scenario."""

    def __init__(self, cfg: Dict[str, Any], shared: Dict[str, Any], logger, seed: int) -> None:
        self.cfg = cfg
        self.logger = logger
        self.rng = default_rng(seed)

        # Unpack shared, then alias for getattr to work like the Model instance
        for k, v in shared.items():
            setattr(self, k, v)

        # Bind helpers with self as first arg
        self._sample_from_stats = types.MethodType(_sample_from_stats, self)
        self._piecewise_quantile_sample = types.MethodType(_piecewise_quantile_sample, self)
        self._trunc_normal = types.MethodType(_trunc_normal, self)

        self._select_bmp_type = types.MethodType(_select_bmp_type, self)
        self._get_bmp_name = types.MethodType(_get_bmp_name, self)
        self._sample_efficiency = types.MethodType(_sample_efficiency, self)            # legacy
        self._sample_efficiency_map = types.MethodType(_sample_efficiency_map, self)    # pathway-aware
        self._simulate_wetland = types.MethodType(_simulate_wetland, self)
        self._simulate_grassed = types.MethodType(_simulate_grassed, self)
        self._simulate_infield = types.MethodType(_simulate_infield, self)
        self._get_bmp_selection_probs = types.MethodType(_get_bmp_selection_probs, self)
        self._get_bmp_cost = types.MethodType(_get_bmp_cost, self)

        self._sample_parcel_index = types.MethodType(_sample_parcel_index, self)
        self._sample_yield = types.MethodType(_sample_yield, self)
        self._get_parcel_metadata = types.MethodType(_get_parcel_metadata, self)
        self._get_parcel_up_list = types.MethodType(_get_parcel_up_list, self)
        self._get_parcel_out_oids = types.MethodType(_get_parcel_out_oids, self)
        self._delivery_coeffs = types.MethodType(_get_delivery_coeffs, self)


def _run_one_scenario(
    shared: Dict[str, Any],
    cfg: Dict[str, Any],
    sidx: int,
    seed: int,
    outputs_dir: Path,
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Execute one scenario and write its outputs."""
    sid = sidx + 1
    logger = make_worker_logger(outputs_dir, scenario_id=sid)
    ctx = _ScenarioContext(cfg, shared, logger, seed)

    logger.info(f"=== scenario {sid} start ===")

    n_pol = len(ctx.pollutants)
    baseline = np.zeros((len(ctx.parcel_selection_ids), n_pol), dtype=float)
    yields = np.zeros_like(baseline)

    # Sample baseline parcel yields for selection set
    pid_to_parcel_idx = {str(pid): ctx.pid_to_index[str(pid)] for pid in ctx.parcel_selection_ids}
    for i, pid in enumerate(ctx.parcel_selection_ids):
        parcel_idx = pid_to_parcel_idx[str(pid)]
        for pol_idx, pol in enumerate(ctx.pollutants):
            y = ctx._sample_yield(parcel_idx, pol_idx)
            baseline[i, pol_idx] = y
            yields[i, pol_idx] = y

    # Limits
    limit_n = cfg.get("bmp_limit_n")
    limit_usd = cfg.get("bmp_limit_usd")
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
    cumul: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Initialize summary collector once per scenario
    collector = BMPSummaryCollector(ctx.pollutants, scenario_id=sid)

    # Track CPS applied per parcel ID to prevent duplicates
    applied_by_pid: Dict[str, set] = defaultdict(set)

    # Main loop
    idle_tries = 0
    max_idle_tries = max(100, len(ctx.parcel_selection_ids) * len(ctx.bmp_cps))
    while True:
        if limit_usd is not None and total_cost >= limit_usd:
            break
        if limit_n is not None and total_bmp >= limit_n:
            break

        parcel_idx = ctx._sample_parcel_index()
        pid = ctx.parcel_selection_ids[parcel_idx]

        # Filter CPS by what has already been applied to this parcel
        already = applied_by_pid[str(pid)]
        allowed_idx = [i for i, c in enumerate(ctx.bmp_cps) if int(c) not in already]

        # If no CPS remain for this parcel, try another parcel; if globally exhausted, stop
        if not allowed_idx:
            idle_tries += 1
            if idle_tries >= max_idle_tries:
                logger.info("No remaining CPS options across parcels; stopping early to avoid infinite loop.")
                break
            continue
        idle_tries = 0  # reset on a viable placement

        # Renormalize probabilities over the allowed CPS subset
        probs_sub = ctx.bmp_selection_probs[allowed_idx]
        probs_sub = probs_sub / probs_sub.sum()
        sel = ctx.rng.choice(len(allowed_idx), p=probs_sub)
        cps = int(ctx.bmp_cps[allowed_idx[sel]])

        # Sample per-pathway efficiencies per pollutant
        eff_maps = [ctx._sample_efficiency_map(cps, pol_idx) for pol_idx in range(n_pol)]

        # Optional BMP failure draw and efficiency scaling
        failed_flag = False
        fr_cfg = ctx.cfg.get(CFG_BMP_FAIL_RATE, 0.0)
        fail_rate = float(fr_cfg if fr_cfg is not None else 0.0)
        if fail_rate > 0.0:
            fail_rate = max(0.0, min(1.0, fail_rate))
            failed = int(ctx.rng.choice([0, 1], p=[1.0 - fail_rate, fail_rate]))
            if failed == 1:
                red_cfg = ctx.cfg.get(CFG_BMP_FAIL_REDUCTION, DEFAULT_BMP_FAIL_REDUCTION)
                reduction = float(red_cfg if red_cfg is not None else DEFAULT_BMP_FAIL_REDUCTION)
                reduction = max(0.0, min(1.0, reduction))
                eff_maps = [{k: float(v) * reduction for k, v in emap.items()} for emap in eff_maps]
                failed_flag = True
                ctx.logger.debug(f"BMP failure triggered for cps={cps}; scaling efficiencies by {reduction:.2f}")

        # Per-BMP record
        bmp_rec: Dict[str, Any] = dict(
            scenario=sid,
            cps=cps,
            cps_name=ctx._get_bmp_name(cps),
            pid=str(pid),
            **{
                OUTPUT_IMPACTED_PIDS: "",
                OUTPUT_LINEAR_LENGTH: None,
                OUTPUT_BUFFER_AREA: None,
                OUTPUT_PORTION_TREATED: None,
                OUTPUT_WETLAND_AREA: None,
                OUTPUT_CATCHMENT_RATIO: None,
            },
        )
        # Record failure flag for CSVs and summaries
        bmp_rec[OUTPUT_BMP_FAILED] = bool(failed_flag)

        bmp_outputs = {OUTPUT_TREATED: np.zeros(n_pol, dtype=float), OUTPUT_REMOVED: np.zeros(n_pol, dtype=float)}

        # Apply BMP
        if cps in (656, 657):
            ctx._simulate_wetland(parcel_idx, eff_maps, yields, bmp_rec, bmp_outputs)
            quantity = float(bmp_rec[OUTPUT_WETLAND_AREA])
        elif cps in (412,):
            ctx._simulate_grassed(parcel_idx, eff_maps, yields, bmp_rec, bmp_outputs)
            quantity = float(bmp_rec[OUTPUT_BUFFER_AREA]) if bmp_rec[OUTPUT_BUFFER_AREA] else 0.0
        else:
            ctx._simulate_infield(parcel_idx, eff_maps, yields, bmp_rec, bmp_outputs)
            quantity = float(ctx.parcel_area_ha[parcel_idx])

        # Costing and totals
        cost_this = ctx._get_bmp_cost(cps, quantity)
        total_cost += cost_this
        total_bmp += 1

        # Mark CPS as applied for this parcel
        applied_by_pid[str(pid)].add(int(cps))

        # Finalize the BMP record
        bmp_rec[OUTPUT_COST_USD] = cost_this
        for pol_idx, pol in enumerate(ctx.pollutants):
            bmp_rec[f"{OUTPUT_TREATED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_TREATED][pol_idx])
            bmp_rec[f"{OUTPUT_REMOVED_PREFIX}{pol}"] = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
        scenario_bmps.append(bmp_rec)

        # Add to summary collector
        pidx_base = pid_to_parcel_idx.get(str(pid), parcel_idx)
        pid_baseline_yields = {pol: float(baseline[pidx_base, i]) for i, pol in enumerate(ctx.pollutants)}
        collector.add_bmp_record(bmp_rec, pid_baseline_yields)

        # Delivered reductions for plots
        oids = ctx._get_parcel_out_oids(parcel_idx)
        for pol_idx, pol in enumerate(ctx.pollutants):
            removed = float(bmp_outputs[OUTPUT_REMOVED][pol_idx])
            for oid in oids:
                dr = ctx._delivery_coeffs(pid, oid)
                deliver = (
                    removed * dr[COL_SDR_F_TO_S] * dr[COL_SDR_S_TO_O]
                    if pol == "TSS"
                    else removed * dr[COL_NDR_F_TO_S] * dr[COL_NDR_S_TO_O]
                )
                cumul[pol][oid] += deliver

        # Record current cumulative for each axis choice
        for pol in ctx.pollutants:
            for oid in ctx.outlet_oids:
                for xax in x_axes:
                    for yax in y_axes:
                        xval = total_bmp if xax == XAXIS_COUNT else total_cost
                        if yax == YAXIS_TOTAL:
                            yval = cumul[pol][oid]
                        elif yax == YAXIS_TARGET:
                            tgt = ctx.outlet_target_map.get((str(oid), pol), 0.0)
                            yval = (cumul[pol][oid] / tgt * 100.0) if tgt > 0 else 0.0
                        elif yax == YAXIS_MEAN:
                            mu = ctx.outlet_mean_map.get((str(oid), pol), 0.0)
                            yval = (cumul[pol][oid] / mu * 100.0) if mu > 0 else 0.0
                        else:
                            yval = 0.0
                        records[(pol, oid, xax, yax)].append((sid, xval, yval))

    # Parcel-level before/after
    for parcel_idx, pid_i in enumerate(ctx.parcel_selection_ids):
        row = dict(scenario=sid, pid=str(pid_i))
        for pol_idx, pol in enumerate(ctx.pollutants):
            row[f"baseline_{pol}"] = float(baseline[parcel_idx, pol_idx])
            row[f"final_{pol}"] = float(yields[parcel_idx, pol_idx])
        scenario_parcels.append(row)

    # Write CSVs and the transposed summary with “All CPS” roll‑up
    bmps_dir = outputs_dir / "bmps"
    parcels_dir = outputs_dir / "parcels"
    summaries_dir = outputs_dir / "summaries"
    for d in (bmps_dir, parcels_dir, summaries_dir):
        d.mkdir(parents=True, exist_ok=True)

    bmps_path = bmps_dir / f"s{sid}.csv"
    parcels_path = parcels_dir / f"s{sid}.csv"
    summary_path = summaries_dir / f"s{sid}.csv"

    pd.DataFrame(scenario_bmps).to_csv(bmps_path, index=False)
    pd.DataFrame(scenario_parcels).to_csv(parcels_path, index=False)

    summary_df = collector.generate_summary_dataframe()
    rollup = collector.generate_rollup_summary()
    summary_with_rollup = pd.concat([summary_df, pd.DataFrame([rollup])], ignore_index=True)

    col_labels = []
    for _, r in summary_with_rollup.iterrows():
        if str(r["cps_name"]) == "All CPS":
            col_labels.append(f"s{int(r['scenario'])}-All CPS")
        else:
            col_labels.append(f"s{int(r['scenario'])}-{str(r['cps_name'])}({int(r['cps'])})")

    tdf = summary_with_rollup.T
    tdf.columns = col_labels
    tdf = tdf.reset_index().rename(columns={"index": "field"})
    tdf.to_csv(summary_path, index=False)

    logger.info(f"Wrote per-scenario BMPs: {bmps_path}")
    logger.info(f"Wrote per-scenario parcels: {parcels_path}")
    logger.info(f"Wrote transposed BMP summary with All CPS: {summary_path}")
    logger.info(f"=== scenario {sid} end (cost={total_cost:.2f}, bmp={total_bmp}) ===")
    return records