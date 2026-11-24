"""Configuration for the configuration."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from basin_bmp_sim.domain.constants import (
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_BUFFER_DEPTH_FT,
    CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
    CFG_DETERMINISTIC_BMP_TYPE_ORDER,
    CFG_DETERMINISTIC_PARCEL_ORDER,
    CFG_N_SCENARIOS,
    CFG_PARALLEL,
    FRACTION_DOMAIN,
    NONNEGATIVE_DOMAIN,
    POSITIVE_DOMAIN,
)
from basin_bmp_sim.common.utils import ci_get
from basin_bmp_sim.config.validation.domains import validate_scalar_in_domain


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate configuration values after defaults and normalization are applied.

    Parameters
    ----------
    cfg : dict[str, Any]
        Normalized model configuration mapping.

    Raises
    ------
    ValueError
        If a configuration value is non-finite, non-integral where an integer
        is required, or outside its allowed physical range.
    """
    legacy_top_level = {
        "outlet_loc", "parcel_out", "parcel_down", "parcel_p",
        "pollutant_load_rate", "delivery_ratios", "outlet_target", "outlet_mean",
    }
    supplied_legacy = sorted(key for key in legacy_top_level if ci_get(cfg, key) is not None)
    if supplied_legacy:
        raise ValueError(
            "Legacy per-file parcel/outlet configuration keys are no longer supported: "
            f"{supplied_legacy}. Use the consolidated 'parcels' and 'outlets' GeoPackages."
        )
    normalized_top_keys = {str(key).lower() for key in cfg}
    if "input_distributions" in normalized_top_keys:
        raise ValueError(
            "input_distributions is no longer supported; define each fixed value or "
            "distribution directly on its model input row"
        )

    load_generation = ci_get(cfg, "load_generation")
    if isinstance(load_generation, dict):
        if any(str(key).lower() == "input_distributions" for key in load_generation):
            raise ValueError(
                "load_generation.input_distributions is no longer supported; define each "
                "distribution directly on its model input row"
            )
        legacy_load_keys = {"plet_inputs", "rusle_inputs", "pollutant_concentrations", "groundwater_concentrations", "hydrology_lookup"}
        supplied_load_legacy = sorted(
            key for key in legacy_load_keys if ci_get(load_generation, key) is not None
        )
        if supplied_load_legacy:
            raise ValueError(
                "Legacy load_generation file keys are no longer supported: "
                f"{supplied_load_legacy}. Store each variable in its dedicated input_* table in parcels.gpkg."
            )

    allowed_top_level = {
        "domain", "parcels", "plet_forcing", "huc_parameters", "outlets", "pollutants", "cps",
        "animal_operations", "feedlot_bmp_cps", "manure_loading",
        "bmp_efficiency", "bmp_cost", "bmp_sel", "n_scenarios",
        "bmp_limit_n", "bmp_limit_usd", "parallel", "random_seed",
        "outputs", "verbose", "buffer_depth_ft", "bmp_sel_prob_via_costs",
        CFG_DETERMINISTIC_PARCEL_ORDER, CFG_DETERMINISTIC_BMP_TYPE_ORDER,
        CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
        "load_generation", "simulation", "bmp_fail_rate",
        "bmp_fail_reduction", "pollutant_load_rate_pathway_fractions",
        "pollutant_load_rate_frac_surface", "pollutant_load_rate_frac_shallow",
    }
    unknown_top = sorted(str(key) for key in cfg if str(key).lower() not in allowed_top_level and str(key).lower() not in legacy_top_level)
    if unknown_top:
        raise ValueError(f"Unknown configuration key(s): {unknown_top}")
    from basin_bmp_sim.science.formulations.formulation_registry import validate_formulation_selection
    validate_formulation_selection(cfg)

    if ci_get(cfg, "plet_forcing") is not None and ci_get(cfg, "huc_parameters") is not None:
        raise ValueError("Specify huc_parameters or the deprecated plet_forcing alias, not both")
    for key in ("huc_parameters", "plet_forcing"):
        value = ci_get(cfg, key)
        if value is not None and (not isinstance(value, (str, Path)) or not str(value).strip()):
            raise ValueError(f"{key} must be one GeoPackage path")

    if isinstance(load_generation, dict):
        allowed_load_generation = {
            "mode", "groundwater_loads", "treat_groundwater_with_bmps",
            "pollutant_load_rate_pathway_fractions", "pathway_mode",
        }
        unknown_load = sorted(
            str(key) for key in load_generation
            if str(key).lower() not in allowed_load_generation
            and str(key).lower() not in {"plet_inputs", "rusle_inputs", "pollutant_concentrations", "groundwater_concentrations", "hydrology_lookup"}
        )
        if unknown_load:
            raise ValueError(f"Unknown load_generation key(s): {unknown_load}")

    n_scenarios_raw = ci_get(cfg, CFG_N_SCENARIOS)
    n_scenarios = validate_scalar_in_domain(
        n_scenarios_raw, POSITIVE_DOMAIN, CFG_N_SCENARIOS
    )
    if not float(n_scenarios).is_integer():
        raise ValueError(f"{CFG_N_SCENARIOS} must be an integer >= 1")

    validate_scalar_in_domain(
        ci_get(cfg, CFG_BUFFER_DEPTH_FT), POSITIVE_DOMAIN, CFG_BUFFER_DEPTH_FT
    )
    validate_scalar_in_domain(
        ci_get(cfg, CFG_BMP_FAIL_RATE), FRACTION_DOMAIN, CFG_BMP_FAIL_RATE
    )
    validate_scalar_in_domain(
        ci_get(cfg, CFG_BMP_FAIL_REDUCTION), FRACTION_DOMAIN, CFG_BMP_FAIL_REDUCTION
    )
    deterministic_parcel_order = ci_get(cfg, CFG_DETERMINISTIC_PARCEL_ORDER)
    if (
        deterministic_parcel_order is not None
        and not isinstance(deterministic_parcel_order, bool)
    ):
        raise ValueError(f"{CFG_DETERMINISTIC_PARCEL_ORDER} must be a boolean")
    deterministic_bmp_type_order = ci_get(cfg, CFG_DETERMINISTIC_BMP_TYPE_ORDER)
    if (
        deterministic_bmp_type_order is not None
        and not isinstance(deterministic_bmp_type_order, bool)
    ):
        raise ValueError(f"{CFG_DETERMINISTIC_BMP_TYPE_ORDER} must be a boolean")
    deterministic_bmp_design_parameters = ci_get(cfg, CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS)
    if (
        deterministic_bmp_design_parameters is not None
        and not isinstance(deterministic_bmp_design_parameters, bool)
    ):
        raise ValueError(f"{CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS} must be a boolean")

    limit_n = ci_get(cfg, CFG_BMP_LIMIT_N)
    if limit_n is not None:
        numeric_limit_n = validate_scalar_in_domain(
            limit_n, NONNEGATIVE_DOMAIN, CFG_BMP_LIMIT_N
        )
        if not numeric_limit_n.is_integer():
            raise ValueError(f"{CFG_BMP_LIMIT_N} must be an integer >= 0")

    limit_usd = ci_get(cfg, CFG_BMP_LIMIT_USD)
    if limit_usd is not None:
        validate_scalar_in_domain(
            limit_usd, NONNEGATIVE_DOMAIN, CFG_BMP_LIMIT_USD
        )

    parallel = ci_get(cfg, CFG_PARALLEL)
    if not isinstance(parallel, dict):
        raise ValueError(f"{CFG_PARALLEL} must be a mapping")
    if "n_jobs" not in parallel or parallel["n_jobs"] is None:
        raise ValueError("parallel.n_jobs must be specified after configuration normalization")
    n_jobs = validate_scalar_in_domain(
        parallel["n_jobs"], POSITIVE_DOMAIN, "parallel.n_jobs"
    )
    if not n_jobs.is_integer():
        raise ValueError("parallel.n_jobs must be an integer >= 1")
