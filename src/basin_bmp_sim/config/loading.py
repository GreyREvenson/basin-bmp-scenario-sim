"""Orchestrate construction of the validated watershed simulation input bundle."""

from __future__ import annotations

from basin_bmp_sim.science.plet_rows import _rows_for_pid

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from basin_bmp_sim.domain.constants import (
    CFG_BMP_EFFICIENCY,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_CPS,
    CFG_N_SCENARIOS,
    CFG_PARALLEL,
    CFG_PARCELS,
    CFG_POLLUTANTS,
    CFG_RANDOM_SEED,
    CFG_LOAD_GENERATION,
    LOAD_MODE_STATISTICAL,
    LOAD_MODE_PLET_RUSLE,
    LOAD_RUSLE_INPUTS,
    LOAD_GROUNDWATER_LOADS,
    LOAD_TREAT_GROUNDWATER_WITH_BMPS,
    COL_MEAN,
    COL_OID,
    COL_PID,
    COL_TARGET,
    COL_PATHWAY,
    PLET_PATHWAY_VALUES,
    GPKG_PARCEL_OUTLETS_TABLE,
    RUSLE_PARAMETER_NAMES,
)
from basin_bmp_sim.common.utils import ci_get, normalize_pollutant_label
from basin_bmp_sim.common.logging import log_scope
from basin_bmp_sim.config.validation.statistics import validate_unique_rows
from basin_bmp_sim.config.validation.configuration import validate_config
from basin_bmp_sim.config.validation.plet import validate_plet_runtime_inputs
from basin_bmp_sim.config.validation.statistical import (
    validate_statistical_efficiency_coverage,
    validate_statistical_load_rates,
)
from basin_bmp_sim.config.plet_tables import _append_parameter_defaults
from basin_bmp_sim.config.settings import normalize_config
from basin_bmp_sim.config.table_sources import (
    _validate_explicit_references,
    _validate_input_package_schema,
    _read_gpkg_input_table,
    _assemble_parcel_parameter_source,
    _assemble_plet_hydrology_source,
    _assemble_plet_concentration_source,
)
from basin_bmp_sim.config.plet_loading import (
    _effective_plet_classification_pairs,
    _load_parameter_stats_table,
    _load_plet_parameter_table,
    _load_plet_hydrology_lookup,
    _load_pollutant_concentrations,
    _split_plet_concentrations,
)

from basin_bmp_sim.config import spatial_loading, bmp_loading, statistical_loading


def _load_pollutants(cfg: Dict[str, Any]) -> List[str]:
    """Load pollutant names from configuration.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.

    Returns
    -------
    list[str]
        Normalized pollutant names.
    """
    pols = ci_get(cfg, CFG_POLLUTANTS)
    if isinstance(pols, str):
        pols = [pols]
    if not pols:
        raise ValueError(f"At least one {CFG_POLLUTANTS} value must be specified")
    return [normalize_pollutant_label(p) for p in pols]


def _load_cps(cfg: Dict[str, Any]) -> List[int]:
    """Load CPS codes while rejecting booleans, fractions, and non-finite values."""
    cps = ci_get(cfg, CFG_CPS)
    if isinstance(cps, (int, float, np.integer, np.floating)) and not isinstance(cps, (bool, np.bool_)):
        cps = [cps]
    if not isinstance(cps, (list, tuple)) or not cps:
        raise ValueError("At least one cps code must be specified")
    result: List[int] = []
    for index, value in enumerate(cps):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"cps[{index}] must be a finite integer, not a boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cps[{index}]={value!r} must be a finite integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"cps[{index}]={value!r} must be a finite integer")
        result.append(int(numeric))
    if len(set(result)) != len(result):
        raise ValueError("cps contains duplicate codes")
    return result


def load_and_validate_all(cfg: Dict[str, Any], logger: Any) -> Dict[str, Any]:
    """Load, validate, and assemble all scenario inputs.

    Parameters
    ----------
    cfg : dict[str, Any]
        Scenario configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    dict[str, Any]
        Data bundle containing the validated inputs and derived lookup
        structures required by the model.

    Raises
    ------
    ValueError
        If configuration values are invalid or required inputs are missing.
    FileNotFoundError
        If a configured input file does not exist.
    """
    normalize_config(cfg)
    validate_config(cfg)
    logger.info("Loading and validating input datasets")
    with log_scope(logger=logger):
        domain = spatial_loading._load_domain(cfg, logger)
        parcels = spatial_loading._load_parcels(cfg, domain, logger)
        from basin_bmp_sim.config.scopes import load_scoped_input_tables
        input_tables = load_scoped_input_tables(cfg, parcels, logger, _read_gpkg_input_table)
        input_kwargs = {"input_tables": input_tables} if input_tables is not None else {}

        up = spatial_loading._load_parcel_graph(cfg, logger)
        out = spatial_loading._load_parcel_outlets(cfg, logger)
        sel = spatial_loading._load_parcel_selection(cfg, parcels, logger, **input_kwargs)

        parcel_ids = [int(pid) for pid in parcels[COL_PID].tolist()]
        source_parcel_ids = [
            int(pid) for pid in parcels.attrs.get("source_pid_universe", parcel_ids)
        ]
        parcel_down_map = spatial_loading._build_parcel_down_map(
            up, parcel_ids, source_parcel_ids=source_parcel_ids, logger=logger
        )

        out = out.copy()
        out[COL_PID] = out[COL_PID].astype("int64")
        out[COL_OID] = out[COL_OID].astype(str).str.strip()
        validate_unique_rows(out, [COL_PID, COL_OID], GPKG_PARCEL_OUTLETS_TABLE)
        unknown_out_pids = sorted(set(out[COL_PID]) - set(source_parcel_ids))
        if unknown_out_pids:
            raise ValueError(
                "parcel_outlets references parcel IDs not found in the source parcels layer: "
                f"{unknown_out_pids[:10]}"
            )
        clipped_out_rows = ~out[COL_PID].isin(parcel_ids)
        if clipped_out_rows.any():
            logger.verbose(
                f"Filtered {int(clipped_out_rows.sum())} parcel_outlets relationship(s) for parcels outside the modeled domain"
            )
            out = out.loc[~clipped_out_rows].copy()
        parcel_out_map: Dict[str, List[str]] = {str(pid): [] for pid in parcel_ids}
        for row in out.itertuples(index=False):
            pid = str(int(getattr(row, COL_PID)))
            oid = str(getattr(row, COL_OID))
            parcel_out_map[pid].append(oid)

        spatial_loading._validate_parcel_outlet_topology(parcel_down_map, parcel_out_map)

        pollutants = _load_pollutants(cfg)
        cps = _load_cps(cfg)

        load_generation = ci_get(cfg, CFG_LOAD_GENERATION) or {}
        if not isinstance(load_generation, dict):
            raise ValueError("load_generation must be a mapping")
        load_generation = {str(k).lower(): v for k, v in load_generation.items()}
        if "mode" not in load_generation or load_generation["mode"] is None:
            load_generation["mode"] = LOAD_MODE_STATISTICAL
        load_mode = str(load_generation["mode"]).strip().lower()
        if load_mode not in {LOAD_MODE_STATISTICAL, LOAD_MODE_PLET_RUSLE}:
            raise ValueError(f"Unsupported load_generation mode: {load_mode}")
        load_generation["mode"] = load_mode
        if "pathway_mode" in load_generation:
            raise ValueError(
                "load_generation.pathway_mode has been removed; "
                "plet_rusle mode always derives pathway loads from PLET/RUSLE inputs"
            )
        _validate_input_package_schema(cfg, load_mode, logger, **input_kwargs)


        outlet_loc = spatial_loading._load_outlet_loc(cfg, domain, logger)
        valid_oids = set(outlet_loc[COL_OID].astype(str))
        referenced_oids = {oid for values in parcel_out_map.values() for oid in values}
        unknown_oids = sorted(referenced_oids - valid_oids)
        if unknown_oids:
            raise ValueError(
                "parcel_outlets references outlet IDs not found in outlets layer: "
                f"{unknown_oids[:10]}"
            )
        outlet_target = spatial_loading._load_optional_outlet_stats(cfg, COL_TARGET, logger)
        outlet_mean = spatial_loading._load_optional_outlet_stats(cfg, COL_MEAN, logger)
        for label, table in (("outlet_stats.target", outlet_target), ("outlet_stats.mean", outlet_mean)):
            if table is not None:
                _validate_explicit_references(
                    table[COL_OID].tolist(), valid_oids, label=label, allow_default=False
                )

        supplied_legacy_groundwater_keys = (
            LOAD_GROUNDWATER_LOADS in load_generation
            or LOAD_TREAT_GROUNDWATER_WITH_BMPS in load_generation
        )
        if LOAD_GROUNDWATER_LOADS not in load_generation or load_generation[LOAD_GROUNDWATER_LOADS] is None:
            load_generation[LOAD_GROUNDWATER_LOADS] = False
        groundwater_loads = bool(load_generation[LOAD_GROUNDWATER_LOADS])
        if LOAD_TREAT_GROUNDWATER_WITH_BMPS not in load_generation or load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] is None:
            load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] = False
        treat_groundwater_with_bmps = bool(load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS])
        load_generation[LOAD_GROUNDWATER_LOADS] = groundwater_loads
        load_generation[LOAD_TREAT_GROUNDWATER_WITH_BMPS] = treat_groundwater_with_bmps
        if load_mode == LOAD_MODE_PLET_RUSLE and supplied_legacy_groundwater_keys:
            logger.verbose(
                "plet_rusle now always estimates lookup-derived subsurface loads; "
                "groundwater_loads/treat_groundwater_with_bmps do not alter pathway generation. "
                "Subsurface BMP treatment is controlled by the subsurface efficiency."
            )

        if ci_get(cfg, CFG_BMP_EFFICIENCY) is None:
            raise ValueError("bmp_efficiency is required")
        bmp_eff = bmp_loading._load_bmp_efficiency(
            cfg, cps, pollutants, logger, complete_legacy=False,
        )
        bmp_cost = bmp_loading._load_bmp_cost(cfg, cps, logger)

        plet_inputs = None
        rusle_inputs = None
        pollutant_concentrations = None
        groundwater_concentrations = None
        if load_mode == LOAD_MODE_PLET_RUSLE:
            package_parameters = _assemble_parcel_parameter_source(cfg, logger, **input_kwargs)
            if package_parameters is None:
                raise ValueError(
                    "plet_rusle mode requires dedicated input_* parameter tables in parcels.gpkg"
                )
            from basin_bmp_sim.science.plet_classification import canonical_parameter_name
            canonical_names = package_parameters["parameter"].map(canonical_parameter_name)
            rusle_mask = canonical_names.isin(set(RUSLE_PARAMETER_NAMES))
            plet_source = package_parameters.loc[~rusle_mask].copy()
            rusle_rows = package_parameters.loc[rusle_mask].copy()
            rusle_source = None if rusle_rows.empty else rusle_rows

            plet_inputs = _load_plet_parameter_table(
                plet_source,
                [int(pid) for pid in sel[COL_PID].tolist()],
                logger,
            )
            if plet_inputs is None:
                raise ValueError(
                    "plet_rusle mode requires dedicated PLET input_* tables in parcels.gpkg"
                )
            required_hydrology_pairs = _effective_plet_classification_pairs(
                plet_inputs, parcel_ids
            )
            hydrology_source = _assemble_plet_hydrology_source(cfg, logger, **input_kwargs)
            parcel_pairs = {}
            if COL_PID in hydrology_source:
                for pid in parcel_ids:
                    effective = {row["parameter"]: row["value"] for row in _rows_for_pid(plet_inputs, pid)}
                    parcel_pairs[pid] = (effective["land_cover"], effective["hsg"])
            plet_hydrology_lookup = _load_plet_hydrology_lookup(
                hydrology_source,
                logger,
                required_pairs=required_hydrology_pairs,
                **({"required_parcel_pairs": parcel_pairs} if parcel_pairs else {}),
            )
            load_generation["_hydrology_lookup_table"] = plet_hydrology_lookup
            rusle_inputs = _load_parameter_stats_table(
                rusle_source, LOAD_RUSLE_INPUTS, logger,
                parcel_ids=parcel_ids,
            )
            concentration_source = _assemble_plet_concentration_source(cfg, logger, **input_kwargs)
            unified_concentrations = _load_pollutant_concentrations(
                concentration_source,
                pollutants,
                logger,
                parcel_ids=parcel_ids,
            )
            pollutant_concentrations, groundwater_concentrations = _split_plet_concentrations(
                unified_concentrations
            )
            # Validate required coverage before optional defaults are appended.
            # This prevents missing required inputs from silently becoming
            # plausible-looking zero/default model values.
            validate_plet_runtime_inputs(
                plet_inputs,
                rusle_inputs,
                pollutant_concentrations,
                groundwater_concentrations,
                parcel_ids,
                pollutants,
            )
            plet_inputs = _append_parameter_defaults(plet_inputs, pollutants)
            pollutant_load_rate = None
            pathways = list(PLET_PATHWAY_VALUES)
            pollutant_load_rate_is_aggregate = False
            pollutant_load_rate_pathway_fractions: Dict[str, float] = {}
            bmp_eff = bmp_loading._complete_plet_bmp_efficiency_coverage(bmp_eff, cps, pollutants, logger)
        else:
            pollutant_load_rate = statistical_loading._load_pollutant_load_rate(
                cfg, parcels, pollutants, logger, **input_kwargs,
            )
            load_rate_pathways, pollutant_load_rate_is_aggregate = validate_statistical_load_rates(
                pollutant_load_rate, parcels, pollutants
            )
            if pollutant_load_rate_is_aggregate:
                if COL_PATHWAY in bmp_eff.columns:
                    pathways = list(dict.fromkeys(bmp_eff[COL_PATHWAY].astype(str).tolist()))
                else:
                    pathways = ["surface"]
                pollutant_load_rate_pathway_fractions = statistical_loading._resolve_aggregate_pathway_fractions(
                    cfg, load_generation, pathways
                )
            else:
                pathways = load_rate_pathways
                pollutant_load_rate_pathway_fractions = {}
            bmp_eff = validate_statistical_efficiency_coverage(
                bmp_eff, cps, pollutants, pathways
            )

        # Optional dedicated AFO GeoPackage keeps operation-to-field relations
        # separate from canonical per-variable parcel input_* tables.
        afo_inputs = None
        feedlot_bmp_cps = []
        afo_path = ci_get(cfg, "animal_operations")
        if afo_path is not None:
            manure_cfg = ci_get(cfg, "manure_loading") or {}
            if not isinstance(manure_cfg, dict):
                raise ValueError("manure_loading must be a mapping")
            method = str(ci_get(manure_cfg, "method") or "concentration_increment").lower()
            surface_pathway = str(ci_get(manure_cfg, "surface_pathway") or "surface")
            if method == "concentration_increment" and load_mode != LOAD_MODE_PLET_RUSLE:
                raise ValueError("concentration_increment manure loading is PLET-only; use mass_based with statistical mode")
            if method == "mass_based" and not {"TN", "TP"}.issubset(set(pollutants)):
                raise ValueError("mass_based manure loading requires both TN and TP pollutants")
            excretion_path = ci_get(manure_cfg, "excretion_factors")
            if excretion_path is not None and method != "mass_based":
                raise ValueError("manure_loading.excretion_factors requires method: mass_based")
            if excretion_path is not None and (
                not isinstance(excretion_path, (str, Path)) or not str(excretion_path).strip()
            ):
                raise ValueError("manure_loading.excretion_factors must be a nonblank CSV path")
            feedlot_bmp_cps = [int(c) for c in (ci_get(cfg, "feedlot_bmp_cps") or [])]
            if len(set(feedlot_bmp_cps)) != len(feedlot_bmp_cps) or not set(feedlot_bmp_cps).issubset(set(cps)):
                raise ValueError("feedlot_bmp_cps must be distinct CPS codes present in cps")
            if set(feedlot_bmp_cps).intersection({412, 656, 657}):
                raise ValueError("feedlot_bmp_cps cannot include buffer or wetland CPS")
            from basin_bmp_sim.science.afo.inputs import load_afo_inputs
            afo_inputs = load_afo_inputs(
                afo_path, parcels, pollutants, method=method, mode=load_mode,
                pathways=pathways, surface_pathway=surface_pathway,
                parcel_path=ci_get(cfg, CFG_PARCELS),
                excretion_factors_path=excretion_path,
                runoff_adjusted_export=ci_get(manure_cfg, "runoff_adjusted_export")
                if "runoff_adjusted_export" in manure_cfg else False,
                reference_runoff_mm=ci_get(manure_cfg, "reference_runoff_mm"),
                feedlot_bmp_cps=feedlot_bmp_cps,
            )
            # Reject incomplete feedlot length-based costing at input load time,
            # before a Monte Carlo worker ever selects the practice.
            if bmp_cost is not None and not bmp_cost.empty and feedlot_bmp_cps:
                from basin_bmp_sim.scenarios.costs import _canonical_cost_unit
                for c in feedlot_bmp_cps:
                    cost_rows = bmp_cost[bmp_cost["cps"].astype(int) == c]
                    if not any(_canonical_cost_unit(row.get("unit")) == "usd/m"
                               for _, row in cost_rows.iterrows()):
                        continue
                    dims = afo_inputs.get("feedlot_cost_dimensions", {})
                    for operation in afo_inputs["operations"]:
                        if "cost_length_m" not in dims.get((str(operation["afo_id"]), c), {}):
                            raise ValueError(
                                f"Feedlot CPS {c} uses USD/m costs: "
                                f"afo_feedlot_bmp_cost_dimensions requires cost_length_m "
                                f"for operation {operation['afo_id']!r}"
                            )
        elif ci_get(cfg, "feedlot_bmp_cps") or ci_get(cfg, "manure_loading"):
            raise ValueError("feedlot_bmp_cps and manure_loading require animal_operations")

        delivery_ratios = spatial_loading._load_delivery_ratios(cfg, logger, **input_kwargs)
        delivery_ratios = spatial_loading._complete_delivery_ratio_defaults(delivery_ratios, parcel_out_map)

        avg_area_ha = float(parcels["area_ha"].mean())
        avg_perim_m = float(parcels["perim_m"].mean())

        logger.info("Input validation complete; assembling data payload")

    return dict(
        parcels=parcels,
        afo_inputs=afo_inputs,
        feedlot_bmp_cps=feedlot_bmp_cps,
        parcel_p=sel,
        parcel_down_map=parcel_down_map,
        parcel_out_map=parcel_out_map,
        pollutants=pollutants,
        cps=cps,
        outlet_loc=outlet_loc,
        outlet_target=outlet_target,
        outlet_mean=outlet_mean,
        bmp_eff=bmp_eff,
        bmp_cost=bmp_cost,
        pollutant_load_rate=pollutant_load_rate,
        delivery_ratios=delivery_ratios,
        load_generation=load_generation,
        plet_inputs=plet_inputs,
        rusle_inputs=rusle_inputs,
        pollutant_concentrations=pollutant_concentrations,
        groundwater_concentrations=groundwater_concentrations,
        pathways=pathways,
        pollutant_load_rate_pathway_fractions=pollutant_load_rate_pathway_fractions,
        pollutant_load_rate_is_aggregate=pollutant_load_rate_is_aggregate,
        bmp_limit_n=ci_get(cfg, CFG_BMP_LIMIT_N),
        bmp_limit_usd=ci_get(cfg, CFG_BMP_LIMIT_USD),
        n_scenarios=int(ci_get(cfg, CFG_N_SCENARIOS)),
        random_seed=ci_get(cfg, CFG_RANDOM_SEED),
        avg_area_ha=avg_area_ha,
        avg_perim_m=avg_perim_m,
        parallel=ci_get(cfg, CFG_PARALLEL),
    )
