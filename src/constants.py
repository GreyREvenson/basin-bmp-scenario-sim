"""Central constants used throughout the simulation.

This module defines the canonical configuration keys, validated data payload
keys, CSV column names, output labels, pollutant aliases, and BMP names used by
the rest of the application.

Notes
-----
Units are normalized as follows:

* lengths are in meters (m)
* areas are in hectares (ha)
* costs are in USD

Canonical pollutant labels are defined by ``POLLUTANT_CANONICAL`` and aliases
map through ``POLLUTANT_ALIAS_MAP``. Output prefixes such as ``treated_`` and
``removed_`` denote per-pollutant loads, while ``cost_usd`` and
``total_cost_usd`` capture BMP costing.
"""

# Config keys
CFG_DOMAIN = "domain"
CFG_PARCELS = "parcels"
CFG_OUTLET_LOC = "outlet_loc"
CFG_PARCEL_OUT = "parcel_out"
CFG_PARCEL_UP = "parcel_up"
CFG_PARCEL_P = "parcel_p"
CFG_POLLUTANTS = "pollutants"
CFG_CPS = "cps"
CFG_POLLUTANT_YIELD = "pollutant_yield"
CFG_BMP_EFFICIENCY = "bmp_efficiency"
CFG_BMP_COST = "bmp_cost"
CFG_DELIVERY_RATIOS = "delivery_ratios"
CFG_OUTLET_TARGET = "outlet_target"
CFG_OUTLET_MEAN = "outlet_mean"
CFG_N_SCENARIOS = "n_scenarios"
CFG_BMP_LIMIT_N = "bmp_limit_n"
CFG_BMP_LIMIT_USD = "bmp_limit_usd"
CFG_BMP_SEL = "bmp_sel"
CFG_PARALLEL = "parallel"
CFG_RANDOM_SEED = "random_seed"
CFG_OUTPUTS = "outputs"
CFG_VERBOSE = "verbose"
CFG_BUFFER_DEPTH_FT = "buffer_depth_ft"
CFG_BMP_SEL_PROB_VIA_COSTS = "bmp_sel_prob_via_costs"
CFG_INPUT_DISTRIBUTIONS = "input_distributions"

# Canonical output folders/files
DIR_SCENARIO_METRICS = "scenario_metrics"
DIR_OUTLET_TRAJECTORIES = "outlet_trajectories"
FILE_ALL_SCENARIOS_PARQUET = "all_scenarios.parquet"

# Optional PLET/RUSLE load-generation block
CFG_LOAD_GENERATION = "load_generation"
LOAD_MODE_STATISTICAL = "statistical"
LOAD_MODE_PLET_RUSLE = "plet_rusle"
LOAD_PLET_INPUTS = "plet_inputs"
LOAD_HYDROLOGY_LOOKUP = "hydrology_lookup"
LOAD_RUSLE_INPUTS = "rusle_inputs"
LOAD_CONCENTRATIONS = "pollutant_concentrations"
LOAD_GROUNDWATER_CONCENTRATIONS = "groundwater_concentrations"
LOAD_GROUNDWATER_LOADS = "groundwater_loads"
LOAD_TREAT_GROUNDWATER_WITH_BMPS = "treat_groundwater_with_bmps"

# New: BMP failure configuration
CFG_BMP_FAIL_RATE = "bmp_fail_rate"            # probability [0,1] a BMP fails
CFG_BMP_FAIL_REDUCTION = "bmp_fail_reduction"  # efficiency scale [0,1] on failure

# New: fractions to split parcel yields by pathway
CFG_POLLUTANT_YIELD_FRAC_SURFACE = "pollutant_yield_frac_surface"
CFG_POLLUTANT_YIELD_FRAC_SHALLOW = "pollutant_yield_frac_shallow"
CFG_POLLUTANT_YIELD_PATHWAY_FRACTIONS = "pollutant_yield_pathway_fractions"

# Data payload keys (used in the validated data dict passed to Model)
DATA_PARCELS = "parcels"
DATA_PARCEL_P = "parcel_p"
DATA_PARCEL_UP_MAP = "parcel_up_map"
DATA_PARCEL_OUT_MAP = "parcel_out_map"
DATA_POLLUTANTS = "pollutants"
DATA_CPS = "cps"
DATA_OUTLET_LOC = "outlet_loc"
DATA_OUTLET_TARGET = "outlet_target"
DATA_OUTLET_MEAN = "outlet_mean"
DATA_BMP_EFFICIENCY = "bmp_eff"
DATA_BMP_COST = "bmp_cost"
DATA_POLLUTANT_YIELD = "pollutant_yield"
DATA_DELIVERY_RATIOS = "delivery_ratios"
DATA_BMP_LIMIT_N = "bmp_limit_n"
DATA_BMP_LIMIT_USD = "bmp_limit_usd"
DATA_N_SCENARIOS = "n_scenarios"
DATA_RANDOM_SEED = "random_seed"
DATA_AVG_AREA_HA = "avg_area_ha"
DATA_AVG_PERIM_M = "avg_perim_m"
DATA_LOAD_GENERATION = "load_generation"
DATA_PLET_INPUTS = "plet_inputs"
DATA_RUSLE_INPUTS = "rusle_inputs"
DATA_POLLUTANT_CONCENTRATIONS = "pollutant_concentrations"
DATA_GROUNDWATER_CONCENTRATIONS = "groundwater_concentrations"
DATA_PATHWAYS = "pathways"
DATA_POLLUTANT_YIELD_PATHWAY_FRACTIONS = "pollutant_yield_pathway_fractions"
DATA_POLLUTANT_YIELD_IS_AGGREGATE = "pollutant_yield_is_aggregate"

# Common column names
COL_PID = "pid"
COL_OID = "oid"
COL_CPS = "cps"
COL_POLLUTANT = "pollutant"
COL_OIDS = "oids"
COL_PID_UP = "pid_up"
COL_PROBABILITY = "probability"
COL_UNIT = "unit"
COL_AREA_M2 = "area_m2"
COL_AREA_HA = "area_ha"
COL_PERIM_M = "perim_m"
COL_TARGET = "target"
COL_MEAN = "mean"
COL_SD = "sd"
COL_MIN = "min"
COL_MAX = "max"
COL_SDR_F_TO_S = "sdr_f_to_s"
COL_SDR_S_TO_O = "sdr_s_to_o"
COL_NDR_F_TO_S = "ndr_f_to_s"
COL_NDR_S_TO_O = "ndr_s_to_o"
PERCENTILE_PREFIX = "p"

# New: optional pathway column
COL_PATHWAY = "pathway"
COL_DISTRIBUTION_ID = "distribution_id"
COL_SAMPLE_GROUP = "sample_group"

# Output and axis constants
XAXIS_COST = "cost"
XAXIS_COUNT = "count"
YAXIS_TOTAL = "total"
YAXIS_TARGET = "target"
YAXIS_MEAN = "mean"

# Default values
DEFAULT_BUFFER_DEPTH_FT = 35.0
DEFAULT_BMP_FAIL_REDUCTION = 0.25  # used when a failure occurs but reduction not provided

OUTPUT_TREATED_PREFIX = "treated_"
OUTPUT_REMOVED_PREFIX = "removed_"
OUTPUT_BASELINE_PREFIX = "baseline_"
OUTPUT_FINAL_PREFIX = "final_"

# Output record suffixes
OUTPUT_EFFICIENCY_JSON = "efficiency_json"
OUTPUT_LINEAR_LENGTH = "linear_length_m"
OUTPUT_BUFFER_AREA = "buffer_area_ha"
OUTPUT_PORTION_TREATED = "portion_treated"
OUTPUT_WETLAND_AREA = "wetland_area_ha"
OUTPUT_CATCHMENT_RATIO = "catchment_to_wetland_ratio"
OUTPUT_IMPACTED_PIDS = "impacted_pids"
OUTPUT_TREATED = "treated"
OUTPUT_REMOVED = "removed"
OUTPUT_COST_USD = "cost_usd"
OUTPUT_TOTAL_COST_USD = "total_cost_usd"
OUTPUT_EFFICIENCY = "efficiency"
OUTPUT_BMP_FAILED = "failed"  # New: per-BMP failure flag in bmps CSVs

# Pollutant canonical labels and alias mapping
POLLUTANT_CANONICAL = ("TN", "TP", "TSS")
POLLUTANT_ALIAS_MAP = {
    "tn": "TN",
    "tp": "TP",
    "tss": "TSS",
    "nitrogen": "TN",
    "phosphorus": "TP",
    "sediment": "TSS",
}

# BMP CPS code name mapping
BMP_CPS_NAME_MAP = {
    329: "Residue Management (No-Till)",
    340: "Cover Crop",
    412: "Grassed Waterway",
    590: "Nutrient Management",
    656: "Constructed Wetland",
}

# Canonical pathway labels
PATHWAY_VALUES = ("surface", "shallow subsurface", "deep subsurface")
PLET_PATHWAY_VALUES = ("surface", "subsurface")