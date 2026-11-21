"""Prepared watershed input builder shared by native and BMI execution.

This module owns the scientific/index lookup construction and the validated
worker payload. It deliberately does not schedule Monte Carlo workers, write
outputs or instantiate BMI components. Native Model inherits its behavior.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.models.model_context import ModelContextServices
from src.formulations.formulation_registry import validate_formulation_selection
from src.logging_utils import log_scope
from src.constants import (
    CFG_BMP_SEL, DATA_RANDOM_SEED, DATA_PARCELS, DATA_POLLUTANTS,
    DATA_PARCEL_OUT_MAP, DATA_PARCEL_DOWN_MAP, DATA_PARCEL_P,
    DATA_OUTLET_LOC, DATA_OUTLET_TARGET, DATA_OUTLET_MEAN,
    DATA_PATHWAYS, DATA_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    DATA_POLLUTANT_LOAD_RATE_IS_AGGREGATE, DATA_CPS, DATA_BMP_EFFICIENCY,
    DATA_LOAD_GENERATION, DATA_PLET_INPUTS, DATA_RUSLE_INPUTS,
    DATA_POLLUTANT_CONCENTRATIONS, DATA_GROUNDWATER_CONCENTRATIONS,
    DATA_POLLUTANT_LOAD_RATE, DATA_AVG_AREA_HA, DATA_AVG_PERIM_M,
    COL_POLLUTANT, COL_PATHWAY, LOAD_MODE_PLET_RUSLE,
)

class PreparedModelInputs(ModelContextServices):
    """Reusable validated model-input preparation, independent of the driver.

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
