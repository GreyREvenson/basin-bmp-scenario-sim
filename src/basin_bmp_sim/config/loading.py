"""Load and assemble validated model input tables.

Filesystem and serialization operations live in :mod:`basin_bmp_sim.common.io`, while
validation policy lives in :mod:`basin_bmp_sim.config.validation`. Configuration normalization and defaults live in ``settings`` and
``plet_defaults``. This module prepares the validated input bundle consumed by
the simulation.
"""

from __future__ import annotations

from pathlib import Path
import difflib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, Tuple
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd

from basin_bmp_sim.domain.parcel_graph import validate_acyclic_downstream_map
from basin_bmp_sim.domain.constants import (
    CFG_BMP_COST,
    CFG_BMP_SEL,
    CFG_BMP_EFFICIENCY,
    CFG_BMP_LIMIT_N,
    CFG_BMP_LIMIT_USD,
    CFG_BMP_FAIL_RATE,
    CFG_BMP_FAIL_REDUCTION,
    CFG_BMP_SEL_PROB_VIA_COSTS,
    CFG_BUFFER_DEPTH_FT,
    CFG_DETERMINISTIC_BMP_DESIGN_PARAMETERS,
    CFG_DETERMINISTIC_BMP_TYPE_ORDER,
    CFG_DETERMINISTIC_PARCEL_ORDER,
    CFG_OUTPUTS,
    CFG_VERBOSE,
    CFG_CPS,
    CFG_DELIVERY_RATIOS,
    CFG_DOMAIN,
    CFG_N_SCENARIOS,
    CFG_OUTLETS,
    CFG_PARALLEL,
    CFG_PARCELS,
    CFG_PLET_FORCING,
    CFG_HUC_PARAMETERS,
    CFG_POLLUTANT_LOAD_RATE,
    CFG_POLLUTANTS,
    CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE,
    CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW,
    CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS,
    CFG_RANDOM_SEED,
    CFG_LOAD_GENERATION,
    LOAD_MODE_STATISTICAL,
    LOAD_MODE_PLET_RUSLE,
    LOAD_PLET_INPUTS,
    LOAD_RUSLE_INPUTS,
    LOAD_CONCENTRATIONS,
    LOAD_GROUNDWATER_CONCENTRATIONS,
    LOAD_GROUNDWATER_LOADS,
    LOAD_GENERATION_PATH_KEYS as _LOAD_GENERATION_PATH_KEYS,
    LOAD_TREAT_GROUNDWATER_WITH_BMPS,
    COL_CPS,
    COL_COST_COMPONENT,
    COL_MEAN,
    COL_OID,
    COL_PID,
    COL_PID_DOWN,
    COL_POLLUTANT,
    COL_PROBABILITY,
    COL_TARGET,
    COL_UNIT,
    COL_PATHWAY,
    PATHWAY_VALUES,
    PLET_PATHWAY_VALUES,
    PLET_HYDROLOGY_LABEL as _PLET_HYDROLOGY_LABEL,
    DEFAULT_BMP_FAIL_REDUCTION,
    DEFAULT_BUFFER_DEPTH_FT,
    COL_SDR_F_TO_S,
    COL_SDR_S_TO_O,
    COL_NDR_F_TO_S,
    COL_NDR_S_TO_O,
    GPKG_PARCELS_LAYER,
    GPKG_PARCEL_DOWN_TABLE,
    GPKG_PARCEL_OUTLETS_TABLE,
    GPKG_PARCEL_HUC12_TABLE,
    GPKG_PLET_HUC12_LAYER,
    GPKG_OUTLETS_LAYER,
    GPKG_OUTLET_STATS_TABLE,
    PARCEL_PARAMETER_INPUT_TABLES,
    HUC12_PLET_PARAMETER_INPUT_TABLES,
    GPKG_INPUT_SELECTION_WEIGHT,
    GPKG_INPUT_CURVE_NUMBER,
    GPKG_INPUT_INFILTRATION_FRACTION,
    GPKG_INPUT_POLLUTANT_LOAD_RATE,
    GPKG_INPUT_SURFACE_CONCENTRATION,
    GPKG_INPUT_SUBSURFACE_CONCENTRATION,
    GPKG_DELIVERY_RATIO_TABLES,
    CONFIG_PATH_KEYS as _CONFIG_PATH_KEYS,
    RUSLE_PARAMETER_NAMES,
    COL_HUC12,
    COL_AREA_FRACTION,
)
from basin_bmp_sim.common.io import (
    MissingInputTableError,
    list_geopackage_tables,
    read_csv_table,
    read_geodataframe,
    read_geopackage_table,
    read_geopackage_table_info,
    read_geopackage_integer_ids,
    read_parquet_table,
)
from basin_bmp_sim.common.utils import ci_get, normalize_columns, normalize_pollutant_label
from basin_bmp_sim.common.logging import log_scope
from basin_bmp_sim.config.distributions import (
    statistic_columns,
    stats_from_row,
)
from basin_bmp_sim.domain.constants import FRACTION_DOMAIN, NONNEGATIVE_DOMAIN, POSITIVE_DOMAIN
from basin_bmp_sim.config.validation.statistics import require_columns, validate_distribution_bounds, validate_numeric_distribution_rows, validate_stats_rows, validate_stats_table, validate_unique_rows
from basin_bmp_sim.config.validation.configuration import validate_config
from basin_bmp_sim.config.validation.domains import validate_numeric_columns_in_domain
from basin_bmp_sim.config.validation.plet import validate_plet_input_table, validate_plet_runtime_inputs
from basin_bmp_sim.config.validation.statistical import validate_statistical_efficiency_coverage, validate_statistical_load_rates
from basin_bmp_sim.config.validation.selection import validate_bmp_selection_table, validate_trajectory_table

from basin_bmp_sim.config.plet_tables import _append_parameter_defaults
from basin_bmp_sim.config.settings import normalize_config
from basin_bmp_sim.config.schema import (
    INPUT_SCHEMA_TABLE,
    INPUT_SCHEMA_VERSION,
    INPUT_VARIABLE_SPECS,
    KNOWN_INPUT_TABLES,
    required_input_tables,
)

































from basin_bmp_sim.config.table_sources import (
    _normalize_integer_identifier_value,
    _normalize_text_identifier_value,
    _normalize_identifier_columns,
    _validate_explicit_references,
    _normalize_cps_column,
    _sort_input_table,
    _validate_input_package_schema,
    _merge_csvs,
    _read_gpkg_input_table,
    _try_read_gpkg_input_table,
    _load_fixed_numeric_variable_table,
    _read_model_input_table,
    _assemble_parcel_parameter_source,
    _assemble_plet_hydrology_source,
    _assemble_plet_concentration_source,
    _load_table_source,
    _normalize_pathway_label,
    _normalize_pathway_column,
    _normalize_pollutant_column,
    _nonblank,
    _row_stats_raw,
    _ensure_projected,
    _normalize_huc12,
    _rows_for_pid
)
from basin_bmp_sim.config.plet_loading import (
    _effective_plet_classification_pairs,
    _load_plet_hydrology_records,
    _load_parameter_stats_table,
    _load_plet_parameter_table,
    _load_plet_hydrology_lookup,
    _load_pollutant_concentrations,
    _split_plet_concentrations,
    _load_parcel_huc12,
    _load_plet_huc12_layer,
    _assemble_huc12_plet_parameter_source,
    _merge_huc12_plet_parameters,
    _merge_huc12_plet_concentrations
)

def load_bmp_selection_probabilities(
    path: Union[str, Path],
    cps: Sequence[int],
    logger: Any,
) -> pd.DataFrame:
    """Read and normalize explicit BMP-selection probabilities.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the BMP-selection probability CSV file.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        Normalized BMP-selection probability table.
    """
    logger.verbose(f"Reading BMP selection probabilities from {path}")
    df = normalize_columns(read_csv_table(path))

    if COL_PROBABILITY not in df.columns:
        alias = next((name for name in ("pr", "p") if name in df.columns), None)
        if alias is not None:
            df[COL_PROBABILITY] = df[alias]

    if COL_CPS in df.columns:
        cps_numeric = pd.to_numeric(df[COL_CPS], errors="coerce")
        finite_integer = np.isfinite(cps_numeric) & (cps_numeric % 1 == 0)
        df.loc[finite_integer, COL_CPS] = cps_numeric.loc[finite_integer].astype(int)

    if COL_PROBABILITY in df.columns:
        df[COL_PROBABILITY] = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")

    configured_cps = {int(value) for value in cps}
    if COL_CPS in df.columns:
        df = df[df[COL_CPS].isin(configured_cps)].copy()

    validate_bmp_selection_table(df, cps)
    df[COL_CPS] = df[COL_CPS].astype(int)
    df[COL_PROBABILITY] = df[COL_PROBABILITY].astype(float)
    df[COL_PROBABILITY] = df[COL_PROBABILITY] / float(df[COL_PROBABILITY].sum())

    result = df[[COL_CPS, COL_PROBABILITY]].reset_index(drop=True)
    logger.verbose(
        f"Loaded explicit BMP selection probabilities from {path}: "
        f"{result.to_dict(orient='records')}"
    )
    return result


def load_trajectory_records(
    path: Union[str, Path],
) -> Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]:
    """Read and normalize canonical outlet-trajectory records for plotting.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the canonical trajectory Parquet file.

    Returns
    -------
    Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
        Trajectory records keyed by pollutant, outlet, x-axis, and y-axis definitions.
    """
    df = read_parquet_table(path)
    validate_trajectory_table(df)

    for column in ("scenario", "step", "x_value", "y_value"):
        df[column] = pd.to_numeric(df[column], errors="raise")

    out: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]] = defaultdict(list)
    df = df.sort_values(["scenario", "pollutant", "oid", "x_axis", "y_axis", "step"])
    for _, row in df.iterrows():
        key = (str(row["pollutant"]), str(row["oid"]), str(row["x_axis"]), str(row["y_axis"]))
        out[key].append((int(row["scenario"]), float(row["x_value"]), float(row["y_value"])))
    return out














































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


def _load_domain(cfg: Dict[str, Any], logger: Any) -> gpd.GeoDataFrame:
    """Load and normalize the model domain boundary.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    logger : Any
        Logger used for progress reporting.

    Returns
    -------
    geopandas.GeoDataFrame
        Domain boundary in a projected CRS with lowercase columns.
    """
    raw_domain = ci_get(cfg, CFG_DOMAIN)
    if raw_domain is None:
        raise ValueError("domain is required")
    domain_path = Path(raw_domain)
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_path}")
    layer = None
    if domain_path.suffix.lower() == ".gpkg" and "domain" in list_geopackage_tables(domain_path):
        layer = "domain"
    domain = read_geodataframe(domain_path, layer=layer) if layer else read_geodataframe(domain_path)
    domain = _ensure_projected(domain, logger)
    return domain.rename(columns={c: c.lower() for c in domain.columns})


def _load_parcels(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load the spatial ``parcels`` layer from the consolidated parcel GeoPackage."""
    raw_parcels = ci_get(cfg, CFG_PARCELS)
    if raw_parcels is None:
        raise ValueError("parcels is required and must point to the consolidated parcels GeoPackage")
    parcels_path = Path(raw_parcels)
    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcels GeoPackage not found: {parcels_path}")
    try:
        parcels = read_geodataframe(
            parcels_path, layer=GPKG_PARCELS_LAYER, fid_as_index=True
        )
    except TypeError:
        # Older GeoPandas/Fiona combinations may not accept ``fid_as_index``.
        parcels = read_geodataframe(parcels_path, layer=GPKG_PARCELS_LAYER)
    except Exception as exc:
        raise ValueError(
            f"Parcels GeoPackage must contain spatial layer '{GPKG_PARCELS_LAYER}': {parcels_path}"
        ) from exc

    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if COL_PID in parcels.columns:
        parcels = parcels.reset_index(drop=True)
    else:
        # In GeoPackage schema v3, ``pid`` is the layer's INTEGER PRIMARY KEY.
        # OGR/GDAL usually exposes that field as the feature ID rather than as a
        # normal attribute.  Prefer the FID index when the reader provides it.
        index_name = str(parcels.index.name or "").lower()
        if index_name in {"fid", COL_PID}:
            feature_ids = parcels.index.to_numpy(copy=True)
            parcels = parcels.reset_index(drop=True)
            parcels.insert(0, COL_PID, feature_ids)
        else:
            # Some GeoPandas/Fiona versions ignore ``fid_as_index`` and return a
            # zero-based RangeIndex. Recover the real GeoPackage feature IDs
            # directly from SQLite instead of mistaking row numbers for parcel IDs.
            feature_ids = read_geopackage_integer_ids(
                parcels_path, GPKG_PARCELS_LAYER, COL_PID
            )
            if len(feature_ids) != len(parcels):
                raise ValueError(
                    "Could not recover parcel IDs from the GeoPackage: the number "
                    "of pid values does not match the number of parcel features"
                )
            parcels = parcels.reset_index(drop=True)
            parcels.insert(0, COL_PID, feature_ids)
    parcels = _normalize_identifier_columns(parcels, [COL_PID], CFG_PARCELS)
    if parcels[COL_PID].duplicated().any():
        dup_pids = sorted(parcels.loc[parcels[COL_PID].duplicated(), COL_PID].unique().tolist())
        raise ValueError(f"Parcel IDs must be unique in the source parcels layer; duplicates found: {dup_pids}")
    source_pid_universe = set(parcels[COL_PID].tolist())
    if parcels.crs is None:
        raise ValueError("Parcels layer must declare a CRS")
    # The domain establishes the one analysis CRS used for all geometry math.
    parcels = parcels.to_crs(domain.crs)
    parcels = gpd.overlay(parcels, domain, how="intersection")
    parcels = parcels.rename(columns={c: c.lower() for c in parcels.columns})
    if parcels.empty:
        raise ValueError("No parcels remain after clipping to the domain")
    parcels = _normalize_identifier_columns(parcels, [COL_PID], CFG_PARCELS)
    if parcels[COL_PID].duplicated().any():
        dup_pids = sorted(parcels.loc[parcels[COL_PID].duplicated(), COL_PID].unique().tolist())
        raise ValueError(f"Parcel IDs must be unique after clipping; duplicates found: {dup_pids}")
    if parcels.geometry.isna().any() or parcels.geometry.is_empty.any():
        raise ValueError("Parcels contain null or empty geometry after clipping")
    if (~parcels.geometry.is_valid).any():
        raise ValueError("Parcels contain invalid geometry after clipping")
    parcels["area_m2"] = parcels.geometry.area
    parcels["perim_m"] = parcels.geometry.length
    parcels["area_ha"] = parcels["area_m2"] / 10000.0
    validate_numeric_columns_in_domain(
        parcels, ["area_m2", "area_ha", "perim_m"], POSITIVE_DOMAIN, CFG_PARCELS
    )
    parcels.attrs["source_pid_universe"] = sorted(source_pid_universe)
    return parcels













def _load_parcel_graph(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load required ``parcel_down(pid,pid_down)``; no legacy fallback."""
    package = ci_get(cfg, CFG_PARCELS)
    try:
        return _read_gpkg_input_table(
            package,
            GPKG_PARCEL_DOWN_TABLE,
            [COL_PID, COL_PID_DOWN],
            GPKG_PARCEL_DOWN_TABLE,
            logger,
        )
    except MissingInputTableError as exc:
        raise ValueError(
            f"Required 'parcel_down(pid,pid_down)' table missing from {package}. "
            "Legacy parcel_up is not supported. Regenerate the drainage table "
            "using utils/create_parcel_down.py, or migrate the GeoPackage. "
            "An empty parcel_down table is valid when all parcels are outlets."
        ) from exc


def _build_parcel_down_map(
    downstream_rows: pd.DataFrame,
    parcel_ids: Sequence[str],
    source_parcel_ids: Optional[Sequence[str]] = None,
    logger: Any = None,
) -> Dict[str, Optional[str]]:
    """Validate one immediate downstream PID per source, filtering clipped edges.

    The GeoPackage stores ``source pid -> pid_down``. The model derives the
    reverse immediate-upstream adjacency in memory for wetland catchments.
    Unmodeled-domain links are filtered only *after* all source IDs have been
    checked, so missing/typo IDs do not masquerade as domain clipping.
    """
    ordered_pids = [str(pid).strip() for pid in parcel_ids]
    modeled = set(ordered_pids)
    source = ({str(pid).strip() for pid in source_parcel_ids}
              if source_parcel_ids is not None else modeled)
    downstream: Dict[str, Optional[str]] = {pid: None for pid in ordered_pids}
    source_to_target: dict[str,str] = {}
    unknown: set[str] = set()
    clipped_edges = 0

    def normalize_pid(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
            text = str(int(value))
        return text

    for row_idx, row in downstream_rows.iterrows():
        pid = normalize_pid(row[COL_PID])
        pid_down = normalize_pid(row[COL_PID_DOWN])
        if not pid or not pid_down:
            raise ValueError(f"parcel_down row {row_idx} requires nonblank pid and pid_down")
        if (pid == '*' or pid_down == '*' or ',' in pid or ',' in pid_down):
            raise ValueError("parcel_down requires a single integer pid_down per row; "
                             "wildcards and comma-separated values are invalid")
        if pid == pid_down:
            raise ValueError(f"parcel_down contains a self-link / directed cycle: {pid} -> {pid}")
        if pid not in source:
            unknown.add(pid)
        if pid_down not in source:
            unknown.add(pid_down)
        if pid in source_to_target:
            raise ValueError(f"parcel_down contains more than one downstream "
                             f"relationship for pid {pid}")
        source_to_target[pid] = pid_down
        if pid in modeled and pid_down in modeled:
            downstream[pid] = pid_down
        else:
            clipped_edges += 1
    if unknown:
        raise ValueError("parcel_down references parcel IDs absent from source "
                         f"parcels: {sorted(unknown)[:10]}")
    if clipped_edges and logger is not None:
        logger.verbose(f"Filtered {clipped_edges} parcel_down link(s) outside modeled domain")

    # Check acyclicity after domain clipping. The reverse graph is used only
    # for validation and wetland traversal; the stored table is downstream.
    upstream: Dict[str, List[str]] = {pid: [] for pid in ordered_pids}
    for pid, to in downstream.items():
        if to is not None:
            upstream[to].append(pid)
    validate_acyclic_downstream_map(downstream)
    return downstream



def _validate_parcel_outlet_topology(
    parcel_down_map: Mapping[str, Optional[str]],
    parcel_out_map: Mapping[str, Sequence[str]],
) -> None:
    """Check complete downstream outlet membership for the representative graph.

    If p drains immediately to q, every modeled outlet downstream of q must
    also receive p's load. Extra outlets for p are valid (nested gauges).
    Parcels with no downstream neighbor may have any outlet membership.
    The check uses the post-domain-clipping mappings, not source-wide edges.
    """
    violations: List[str] = []
    for pid, downstream_pid in parcel_down_map.items():
        if downstream_pid is None:
            continue
        source_outlets = set(map(str, parcel_out_map.get(str(pid), ())))
        downstream_outlets = set(map(str, parcel_out_map.get(str(downstream_pid), ())))
        missing = sorted(downstream_outlets - source_outlets)
        if missing:
            violations.append(
                f'pid={pid} -> pid_down={downstream_pid}: missing downstream outlet(s) {missing}'
            )
    if violations:
        raise ValueError(
            'parcel_outlets is inconsistent with parcel_down: each upstream '
            'parcel must list every modeled outlet associated with its '
            'immediate downstream parcel. Example violation(s): '
            + '; '.join(violations[:10])
        )


def _load_parcel_outlets(cfg: Dict[str, Any], logger: Any) -> pd.DataFrame:
    """Load normalized parcel-to-outlet edges from ``parcels.gpkg``."""
    return _read_gpkg_input_table(
        ci_get(cfg, CFG_PARCELS),
        GPKG_PARCEL_OUTLETS_TABLE,
        [COL_PID, COL_OID],
        GPKG_PARCEL_OUTLETS_TABLE,
        logger,
    )

def _load_parcel_selection(
    cfg: Dict[str, Any], parcels: pd.DataFrame, logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Load optional parcel selection weights from ``input_selection_weight``.

    A row with ``pid IS NULL`` supplies the default weight; exact integer PID
    rows override it.
    """
    if parcels.empty:
        raise ValueError("No parcels available for selection")
    table = _load_fixed_numeric_variable_table(
        cfg, GPKG_INPUT_SELECTION_WEIGHT, [COL_PID], logger,
        required=False, domain=NONNEGATIVE_DOMAIN,
        input_tables=input_tables,
    )
    parcel_ids = [int(pid) for pid in parcels[COL_PID].tolist()]
    if table is None or table.empty:
        return pd.DataFrame({
            COL_PID: parcel_ids,
            COL_PROBABILITY: np.full(len(parcels), 1.0 / len(parcels)),
        })
    table = table.copy()
    default_rows = table[table[COL_PID].isna()]
    if len(default_rows) > 1:
        raise ValueError(
            f"{GPKG_INPUT_SELECTION_WEIGHT} may contain at most one NULL-pid default row"
        )
    default_value = None if default_rows.empty else float(default_rows.iloc[0]["value"])
    exact = {
        int(row[COL_PID]): float(row["value"])
        for _, row in table[table[COL_PID].notna()].iterrows()
    }
    unknown = sorted(set(exact) - set(parcel_ids))
    if unknown:
        raise ValueError(
            f"{GPKG_INPUT_SELECTION_WEIGHT} references parcel IDs not found after clipping: {unknown[:10]}"
        )
    weights = []
    for pid in parcel_ids:
        if pid in exact:
            weights.append(exact[pid])
        elif default_value is not None:
            weights.append(default_value)
        else:
            raise ValueError(
                f"{GPKG_INPUT_SELECTION_WEIGHT} has no value for pid={pid}; "
                "supply the parcel explicitly or add one row with pid=NULL"
            )
    weights_arr = np.asarray(weights, dtype=float)
    if ((~np.isfinite(weights_arr)) | (weights_arr < 0.0)).any():
        raise ValueError(
            f"{GPKG_INPUT_SELECTION_WEIGHT} values must be finite and >= 0"
        )
    total = float(weights_arr.sum())
    if total <= 0.0:
        raise ValueError(f"{GPKG_INPUT_SELECTION_WEIGHT} values sum to zero or negative")
    return pd.DataFrame({
        COL_PID: parcel_ids,
        COL_PROBABILITY: weights_arr / total,
    })

def _load_outlet_loc(cfg: Dict[str, Any], domain: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Load the spatial ``outlets`` layer from the consolidated outlet GeoPackage."""
    raw_outlets = ci_get(cfg, CFG_OUTLETS)
    if raw_outlets is None:
        raise ValueError("outlets is required and must point to the consolidated outlets GeoPackage")
    outlet_path = Path(raw_outlets)
    if not outlet_path.exists():
        raise FileNotFoundError(f"Outlets GeoPackage not found: {outlet_path}")
    try:
        outlet_loc = read_geodataframe(outlet_path, layer=GPKG_OUTLETS_LAYER).to_crs(domain.crs)
    except Exception as exc:
        raise ValueError(
            f"Outlets GeoPackage must contain spatial layer '{GPKG_OUTLETS_LAYER}': {outlet_path}"
        ) from exc
    outlet_loc = outlet_loc.rename(columns={c: c.lower() for c in outlet_loc.columns})
    require_columns(outlet_loc, [COL_OID], CFG_OUTLETS, logger)
    outlet_loc = _normalize_identifier_columns(outlet_loc, [COL_OID], CFG_OUTLETS)
    validate_unique_rows(outlet_loc, [COL_OID], CFG_OUTLETS)
    if outlet_loc.geometry.isna().any() or outlet_loc.geometry.is_empty.any():
        raise ValueError("Outlets contain null or empty geometry")
    if (~outlet_loc.geometry.is_valid).any():
        raise ValueError("Outlets contain invalid geometry")
    non_points = ~outlet_loc.geometry.geom_type.isin(["Point", "MultiPoint"])
    if non_points.any():
        raise ValueError("Outlets layer must contain point geometry")
    return outlet_loc.reset_index(drop=True)

def _load_optional_outlet_stats(
    cfg: Dict[str, Any],
    value_col: str,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Load one optional statistic column from the consolidated ``outlet_stats`` table."""
    if value_col not in {COL_TARGET, COL_MEAN}:
        raise ValueError(f"Unsupported outlet statistic column: {value_col}")
    label = f"{GPKG_OUTLET_STATS_TABLE}.{value_col}"
    required_cols = [COL_OID, COL_POLLUTANT, value_col]
    package = ci_get(cfg, CFG_OUTLETS)
    base_required = [COL_OID, COL_POLLUTANT]
    try:
        df = _read_gpkg_input_table(
            package, GPKG_OUTLET_STATS_TABLE, base_required, "outlet_stats", logger
        )
    except MissingInputTableError:
        logger.verbose("outlet_stats table not present; skipping optional outlet statistics")
        return None
    if value_col not in df.columns:
        logger.verbose(f"outlet_stats does not contain {value_col}; skipping {label}")
        return None
    df = df[df[value_col].notna()].copy()
    if df.empty:
        return None
    require_columns(df, required_cols, label, logger)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, label, logger)
    validate_numeric_columns_in_domain(df, [value_col], NONNEGATIVE_DOMAIN, label)
    validate_unique_rows(df, [COL_OID, COL_POLLUTANT], label)
    return df[list(required_cols)].reset_index(drop=True)

def _load_delivery_ratios(
    cfg: Dict[str, Any], logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """Load the four parcel-to-outlet delivery variables from dedicated tables.

    A global default row is represented by ``pid IS NULL AND oid IS NULL``.
    Partial-NULL keys are invalid.
    """
    base = _load_parcel_outlets(cfg, logger)[[COL_PID, COL_OID]].copy()
    base[COL_PID] = base[COL_PID].astype("int64")
    base[COL_OID] = base[COL_OID].astype(str).str.strip()
    result = base.copy()
    any_supplied = False
    valid_pairs = set(zip(base[COL_PID], base[COL_OID]))

    for column, table_name in GPKG_DELIVERY_RATIO_TABLES.items():
        table = _load_fixed_numeric_variable_table(
            cfg, table_name, [COL_PID, COL_OID], logger,
            required=False, domain=FRACTION_DOMAIN,
            input_tables=input_tables,
        )
        if table is None or table.empty:
            result[column] = 1.0
            continue
        any_supplied = True
        table = table.copy()
        default_mask = table[COL_PID].isna() & table[COL_OID].isna()
        defaults = table[default_mask]
        if len(defaults) > 1:
            raise ValueError(f"{table_name} may contain at most one NULL/NULL default row")
        partial_null = table[COL_PID].isna() ^ table[COL_OID].isna()
        if partial_null.any():
            raise ValueError(
                f"{table_name} supports exact pid/oid rows or one row with both pid and oid NULL"
            )
        default = 1.0 if defaults.empty else float(defaults.iloc[0]["value"])
        exact_rows = table[~default_mask]
        exact = {
            (int(row[COL_PID]), str(row[COL_OID])): float(row["value"])
            for _, row in exact_rows.iterrows()
        }
        unknown = sorted(set(exact) - valid_pairs, key=lambda pair: (pair[0], pair[1]))
        if unknown:
            raise ValueError(
                f"{table_name} references parcel/outlet relationships not present in parcel_outlets: {unknown[:10]}"
            )
        result[column] = [exact.get(pair, default) for pair in zip(result[COL_PID], result[COL_OID])]

    if not any_supplied:
        logger.verbose("No delivery-ratio input tables present; using neutral defaults")
    return result.reset_index(drop=True)

def _efficiency_stat_columns(df: pd.DataFrame) -> List[str]:
    """Return columns that can define an efficiency distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        BMP efficiency input table.

    Returns
    -------
    list[str]
        Statistic columns present in the table.
    """
    named_stats = {
        "value",
        "mean",
        "average",
        "avg",
        "sd",
        "std",
        "min",
        "minimum",
        "max",
        "maximum",
        "p0",
        "p100",
    }
    return [
        col
        for col in df.columns
        if str(col).lower() in named_stats
        or (
            str(col).lower().startswith("p")
            and str(col).lower()[1:].isdigit()
        )
    ]


def _plet_numeric_spec_columns(df: pd.DataFrame) -> List[str]:
    """Return standardized numeric-specification columns present in a BMP-efficiency table.

    Parameters
    ----------
    df : pandas.DataFrame
        BMP-efficiency input table.

    Returns
    -------
    list[str]
        Numeric-specification columns present in the table.
    """
    candidates = [
        "value",
        "mean",
        "sd",
        "min",
        "p05",
        "p50",
        "p95",
        "max",
    ]
    return [col for col in candidates if col in df.columns]


def _complete_bmp_efficiency_coverage(
    df: pd.DataFrame,
    cps: Sequence[int],
    pollutants: Sequence[str],
    logger: Any,
) -> pd.DataFrame:
    """Legacy three-path completion used by the public loader API.

    Surface is required. Missing shallow/deep subsurface values are completed
    as fixed zero distributions with verbose logging. Production mode-specific
    validation bypasses this compatibility layer.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : Sequence[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        BMP efficiency table with complete legacy pathway coverage.

    Raises
    ------
    ValueError
        If required surface-efficiency coverage is missing for a configured CPS/pollutant combination.
    """
    completed = df.copy()
    completed[COL_CPS] = completed[COL_CPS].astype(int)
    if COL_PATHWAY not in completed.columns:
        completed[COL_PATHWAY] = PATHWAY_VALUES[0]
        logger.verbose(
            "bmp_efficiency has no pathway column; treating supplied "
            "CPS x pollutant efficiencies as surface efficiencies"
        )

    configured_cps = [int(cps_code) for cps_code in cps]
    configured_pollutants = [str(pollutant) for pollutant in pollutants]
    surface_pathway = PATHWAY_VALUES[0]
    subsurface_pathways = PATHWAY_VALUES[1:]
    missing_surface: List[Tuple[int, str]] = []
    for cps_code in configured_cps:
        for pollutant in configured_pollutants:
            mask = (
                (completed[COL_CPS] == cps_code)
                & (completed[COL_POLLUTANT] == pollutant)
                & (completed[COL_PATHWAY] == surface_pathway)
            )
            if not mask.any():
                missing_surface.append((cps_code, pollutant))
    if missing_surface:
        details = ", ".join(
            f"cps={cps_code}, pollutant={pollutant}"
            for cps_code, pollutant in missing_surface
        )
        raise ValueError(
            "bmp_efficiency is missing required surface efficiency coverage "
            f"for configured CPS x pollutant combinations: {details}"
        )

    stat_columns = _efficiency_stat_columns(completed)
    added_rows: List[pd.Series] = []
    for cps_code in configured_cps:
        for pollutant in configured_pollutants:
            pair_mask = (
                (completed[COL_CPS] == cps_code)
                & (completed[COL_POLLUTANT] == pollutant)
            )
            surface_row = completed[
                pair_mask & (completed[COL_PATHWAY] == surface_pathway)
            ].iloc[0]
            for pathway in subsurface_pathways:
                pathway_mask = pair_mask & (completed[COL_PATHWAY] == pathway)
                if pathway_mask.any():
                    row_index = completed[pathway_mask].index[0]
                    has_values = any(
                        not pd.isna(completed.at[row_index, column])
                        for column in stat_columns
                    )
                    if has_values:
                        continue
                    for column in stat_columns:
                        completed.at[row_index, column] = 0.0
                else:
                    default_row = surface_row.copy()
                    default_row[COL_PATHWAY] = pathway
                    for column in stat_columns:
                        default_row[column] = 0.0
                    added_rows.append(default_row)
                logger.verbose(
                    "No bmp_efficiency value specified for "
                    f"cps={cps_code}, pollutant={pollutant}, pathway='{pathway}'; "
                    "assuming efficiency=0"
                )

    if added_rows:
        completed = pd.concat(
            [completed, pd.DataFrame(added_rows, columns=completed.columns)],
            ignore_index=True,
        )

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)
    pathway_order = {pathway: idx for idx, pathway in enumerate(PATHWAY_VALUES)}
    completed["_pathway_order"] = completed[COL_PATHWAY].map(pathway_order)
    completed = completed.sort_values(
        [COL_CPS, COL_POLLUTANT, "_pathway_order"], kind="stable"
    ).drop(columns="_pathway_order")
    return completed.reset_index(drop=True)


def _complete_plet_bmp_efficiency_coverage(
    df: pd.DataFrame, cps: Sequence[int], pollutants: Sequence[str], logger: Any
) -> pd.DataFrame:
    """Require PLET surface efficiencies and default missing subsurface to zero.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    cps : Sequence[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : Sequence[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        BMP efficiency table with required PLET pathway coverage.

    Raises
    ------
    ValueError
        If PLET/RUSLE surface-efficiency coverage is missing for a configured CPS/pollutant combination.
    """
    completed = df.copy()
    completed[COL_CPS] = completed[COL_CPS].astype(int)

    if COL_PATHWAY not in completed.columns:
        completed[COL_PATHWAY] = "surface"
        logger.verbose(
            "plet_rusle bmp_efficiency has no pathway column; treating supplied "
            "CPS x pollutant rows as surface efficiencies"
        )
    else:
        completed[COL_PATHWAY] = completed[COL_PATHWAY].astype(str).str.strip().str.lower()
        unexpected = sorted(
            set(completed[COL_PATHWAY].astype(str)) - set(PLET_PATHWAY_VALUES)
        )
        if unexpected:
            logger.warning(
                "plet_rusle recognizes only pathway labels 'surface' and "
                f"'subsurface'. Ignoring unexpected bmp_efficiency pathway labels: {unexpected}. "
                "If no correctly labeled subsurface efficiency remains for a CPS/pollutant, "
                "subsurface efficiency will be assumed to be 0."
            )
        completed = completed[completed[COL_PATHWAY].isin(PLET_PATHWAY_VALUES)].copy()

    validate_unique_rows(
        completed, [COL_CPS, COL_POLLUTANT, COL_PATHWAY], CFG_BMP_EFFICIENCY
    )

    numeric_cols = _plet_numeric_spec_columns(completed)
    added_rows: List[pd.Series] = []
    missing_surface: List[Tuple[int, str]] = []

    for cps_code in [int(x) for x in cps]:
        for pollutant in [str(x) for x in pollutants]:
            pair = (completed[COL_CPS] == cps_code) & (completed[COL_POLLUTANT] == pollutant)
            surf = completed[pair & (completed[COL_PATHWAY] == "surface")]
            if surf.empty:
                missing_surface.append((cps_code, pollutant))
                continue

            sub = completed[pair & (completed[COL_PATHWAY] == "subsurface")]
            sub_has_stats = False
            if not sub.empty:
                row = sub.iloc[0]
                sub_has_stats = any(not pd.isna(row.get(c)) for c in numeric_cols)

            if not sub_has_stats:
                template = surf.iloc[0].copy()
                template[COL_PATHWAY] = "subsurface"
                for c in numeric_cols:
                    template[c] = pd.NA
                if "value" in numeric_cols:
                    template["value"] = 0.0
                if COL_UNIT in template.index:
                    template[COL_UNIT] = "fraction"
                if "notes" in template.index:
                    prior = template["notes"]
                    auto_note = (
                        "Auto-added in plet_rusle: missing subsurface efficiency defaults to 0"
                    )
                    template["notes"] = (
                        auto_note
                        if pd.isna(prior) or str(prior).strip() == ""
                        else f"{prior}; {auto_note}"
                    )
                if not sub.empty:
                    completed = completed.drop(index=sub.index)
                added_rows.append(template)
                logger.warning(
                    "plet_rusle: no correctly labeled subsurface BMP efficiency was "
                    f"defined for cps={cps_code}, pollutant={pollutant}; assuming efficiency=0"
                )

    if missing_surface:
        details = ", ".join(f"cps={c}, pollutant={p}" for c, p in missing_surface)
        raise ValueError(
            "plet_rusle requires a surface bmp_efficiency for every configured "
            f"CPS x pollutant combination; missing: {details}"
        )

    if added_rows:
        completed = pd.concat([completed, pd.DataFrame(added_rows, columns=completed.columns)], ignore_index=True)

    validate_stats_rows(completed, CFG_BMP_EFFICIENCY)

    order = {"surface": 0, "subsurface": 1}
    completed["_pathway_order"] = completed[COL_PATHWAY].map(order)
    return completed.sort_values(
        [COL_CPS, COL_POLLUTANT, "_pathway_order"], kind="stable"
    ).drop(columns="_pathway_order").reset_index(drop=True)


def _load_bmp_efficiency(
    cfg: Dict[str, Any],
    cps: List[int],
    pollutants: List[str],
    logger: Any,
    *,
    complete_legacy: bool = True,
) -> pd.DataFrame:
    """Load and normalize BMP effectiveness inputs.

    ``complete_legacy=True`` preserves the public three-path loader behavior
    used by existing callers/tests. The main model loader passes ``False`` and
    then performs mode-specific validation for PLET/RUSLE or statistical mode.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Normalized model configuration mapping.
    cps : List[int]
        Conservation Practice Standard (CPS) code or codes.
    pollutants : List[str]
        Pollutant names in model order.
    logger : Any
        Logger used for diagnostic and progress messages.
    complete_legacy : bool
        Whether to complete legacy three-pathway BMP efficiency coverage.

    Returns
    -------
    pd.DataFrame
        Normalized BMP efficiency table.

    Raises
    ------
    ValueError
        If no BMP-efficiency records remain for the configured CPS codes and pollutants.
    """
    df = _merge_csvs(ci_get(cfg, CFG_BMP_EFFICIENCY), [COL_CPS, COL_POLLUTANT], CFG_BMP_EFFICIENCY, logger)
    df = _normalize_cps_column(df, CFG_BMP_EFFICIENCY)
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_BMP_EFFICIENCY, logger)
    df = _normalize_pathway_column(df, CFG_BMP_EFFICIENCY, logger)
    df = df[df[COL_CPS].isin(cps) & df[COL_POLLUTANT].isin(pollutants)].copy()
    if df.empty:
        raise ValueError("bmp_efficiency has no records for specified cps+pollutants")

    if complete_legacy:
        return _complete_bmp_efficiency_coverage(df, cps, pollutants, logger)
    return df


def _load_bmp_cost(cfg: Dict[str, Any], cps: List[int], logger: Any) -> Optional[pd.DataFrame]:
    """Optionally load BMP cost inputs.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration mapping.
    cps : list[int]
        BMP CPS codes to retain.
    logger : Any
        Logger used for progress and warning messages.

    Returns
    -------
    pandas.DataFrame or None
        BMP cost table filtered to the requested BMPs, or ``None`` when no
        usable cost table is configured.
    """
    path = ci_get(cfg, CFG_BMP_COST)
    if path is None:
        return None
    # BMP cost tables allow more than one row per CPS. Each row represents an
    # additive cost component (for example seed + establishment + termination).
    # Legacy one-row-per-CPS files remain valid: when ``cost_component`` is
    # absent, the loader assigns the component name ``total``.
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    frames: List[pd.DataFrame] = []
    for p in paths:
        logger.verbose(f"Reading {CFG_BMP_COST} from {p}")
        frame = normalize_columns(read_csv_table(p))
        require_columns(frame, [COL_CPS, COL_UNIT], f"{CFG_BMP_COST} ({p})", logger)
        if COL_COST_COMPONENT not in frame.columns:
            frame[COL_COST_COMPONENT] = "total"
        else:
            component = frame[COL_COST_COMPONENT].astype("string").str.strip()
            frame[COL_COST_COMPONENT] = component.mask(component.isna() | (component == ""), "total")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True).drop_duplicates(keep="first")
    logical_key = [COL_CPS, COL_COST_COMPONENT]
    dup = df.duplicated(subset=logical_key, keep=False)
    if dup.any():
        preview = df.loc[dup, logical_key].head(10).to_dict(orient="records")
        raise ValueError(
            f"{CFG_BMP_COST} contains conflicting duplicate rows for logical key "
            f"{logical_key}: {preview}"
        )
    df = df.reset_index(drop=True)
    df = _normalize_cps_column(df, CFG_BMP_COST)
    validate_stats_table(df, CFG_BMP_COST)
    df = df[df[COL_CPS].isin(cps)].copy()
    if df.empty:
        logger.warning("bmp_cost has no records for specified cps; proceeding without costing")
        return None
    return df


def _expand_pollutant_load_rate_defaults(
    df: pd.DataFrame,
    parcel_ids: Sequence[Any],
    pollutants: Sequence[str],
) -> pd.DataFrame:
    """Expand NULL-pid load-rate defaults while preserving exact overrides.

    External GeoPackage inputs have already been normalized to integer PIDs.
    Keeping this helper tolerant of programmatic string IDs makes isolated
    model/unit-test contexts usable without weakening the file-input contract.
    """
    def pid_key(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return str(value)
        text = str(value).strip()
        try:
            numeric = float(text)
            if np.isfinite(numeric) and numeric.is_integer():
                return int(numeric)
        except ValueError:
            pass
        return text

    out = df.copy()
    valid_pids = {pid_key(pid) for pid in parcel_ids}
    normalized_pid = out[COL_PID].map(pid_key)
    out = out[
        (out[COL_PID].isna() | normalized_pid.isin(valid_pids))
        & out[COL_POLLUTANT].isin(list(pollutants))
    ].copy()
    normalized_pid = out[COL_PID].map(pid_key)
    if out.empty or not out[COL_PID].isna().any():
        out = out[out[COL_PID].notna()].copy()
        out[COL_PID] = out[COL_PID].map(pid_key)
        return out.reset_index(drop=True)

    explicit = COL_PATHWAY in out.columns
    keys = [COL_PID, COL_POLLUTANT] + ([COL_PATHWAY] if explicit else [])
    validate_unique_rows(out, keys, CFG_POLLUTANT_LOAD_RATE)
    pathways: List[Optional[str]] = (
        list(dict.fromkeys(out[COL_PATHWAY].astype(str).tolist()))
        if explicit else [None]
    )

    defaults: Dict[Tuple[str, Optional[str]], pd.Series] = {}
    exact: Dict[Tuple[Any, str, Optional[str]], pd.Series] = {}
    for _, row in out.iterrows():
        path = str(row[COL_PATHWAY]) if explicit else None
        pollutant = str(row[COL_POLLUTANT])
        key = pid_key(row[COL_PID])
        if key is None:
            defaults[(pollutant, path)] = row
        else:
            exact[(key, pollutant, path)] = row

    expanded: List[Dict[str, Any]] = []
    for raw_pid in parcel_ids:
        pid = pid_key(raw_pid)
        for pollutant in map(str, pollutants):
            for path in pathways:
                row = exact.get((pid, pollutant, path))
                if row is None:
                    row = defaults.get((pollutant, path))
                if row is None:
                    continue
                copied = row.to_dict()
                copied[COL_PID] = pid
                expanded.append(copied)
    if not expanded:
        return out.iloc[0:0].copy()
    return pd.DataFrame(expanded).reset_index(drop=True)

def _load_pollutant_load_rate(
    cfg: Dict[str, Any],
    parcels: pd.DataFrame,
    pollutants: List[str],
    logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Load statistical parcel pollutant load rates from ``parcels.gpkg``."""
    df = _read_model_input_table(
        cfg, GPKG_INPUT_POLLUTANT_LOAD_RATE, [COL_PID, COL_POLLUTANT], logger,
        required=True, input_tables=input_tables,
    )
    df = _normalize_pollutant_column(df, COL_POLLUTANT, CFG_POLLUTANT_LOAD_RATE, logger)
    df = _normalize_pathway_column(df, CFG_POLLUTANT_LOAD_RATE, logger)
    validate_stats_table(df, CFG_POLLUTANT_LOAD_RATE)
    df = _normalize_identifier_columns(df, [COL_PID], CFG_POLLUTANT_LOAD_RATE, allow_null_pid=True)
    parcel_ids = [int(pid) for pid in parcels[COL_PID].tolist()]
    _validate_explicit_references(
        df[COL_PID].tolist(), parcel_ids, label=CFG_POLLUTANT_LOAD_RATE, allow_default=True
    )
    df = _expand_pollutant_load_rate_defaults(df, parcel_ids, pollutants)
    if df.empty:
        raise ValueError(f"{GPKG_INPUT_POLLUTANT_LOAD_RATE} has no records for specified parcels+pollutants")
    validate_stats_rows(df, CFG_POLLUTANT_LOAD_RATE)
    return df

def _resolve_aggregate_pathway_fractions(
    cfg: Dict[str, Any], load_generation: Dict[str, Any], pathways: Sequence[str]
) -> Dict[str, float]:
    """Resolve fractions used to split one sampled aggregate parcel load rate.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Normalized model configuration mapping.
    load_generation : Dict[str, Any]
        Load-generation configuration mapping.
    pathways : Sequence[str]
        Pollutant transport pathway names.

    Returns
    -------
    Dict[str, float]
        Normalized pathway fractions keyed by pathway name.

    Raises
    ------
    ValueError
        If pathway fractions are malformed, reference unknown pathways, fall
        outside ``[0, 1]``, or do not sum to one.
    """
    pathways = list(pathways)
    if len(pathways) == 1:
        return {pathways[0]: 1.0}
    raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS)
    if raw is None:
        raw = load_generation.get(CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS)
    fractions: Dict[str, float] = {}
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError(f"{CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS} must be a mapping")
        fractions = {_normalize_pathway_label(k): float(v) for k, v in raw.items()}
    else:
        surf_raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_FRAC_SURFACE)
        shallow_raw = ci_get(cfg, CFG_POLLUTANT_LOAD_RATE_FRAC_SHALLOW)
        if surf_raw is not None:
            fractions["surface"] = float(surf_raw)
        if shallow_raw is not None:
            fractions["shallow subsurface"] = float(shallow_raw)
        if shallow_raw is not None and surf_raw is None and "surface" in pathways:
            fractions["surface"] = 1.0 - float(shallow_raw)
        elif surf_raw is not None and shallow_raw is not None and "deep subsurface" in pathways:
            fractions["deep subsurface"] = 1.0 - float(surf_raw) - float(shallow_raw)
    unknown = set(fractions) - set(pathways)
    if unknown:
        raise ValueError(
            f"Pathway fractions refer to pathways not defined by bmp_efficiency: {sorted(unknown)}"
        )
    if not fractions:
        raise ValueError(
            "Statistical mode uses one aggregate pollutant_load_rate per parcel but multiple BMP pathways. "
            f"Define {CFG_POLLUTANT_LOAD_RATE_PATHWAY_FRACTIONS}, e.g. {{'shallow subsurface': 0.2, 'surface': 0.8}}."
        )
    for path in pathways:
        fractions.setdefault(path, 0.0)
    vals = np.asarray(list(fractions.values()), dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("pollutant load rate pathway fractions must be finite")
    if (vals < 0.0).any() or (vals > 1.0).any():
        raise ValueError("pollutant load rate pathway fractions must each be in [0,1]")
    total = float(vals.sum())
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError(f"pollutant load rate pathway fractions must sum to 1.0; got {total:.12g}")
    return {path: float(fractions[path]) for path in pathways}


def _complete_delivery_ratio_defaults(
    delivery_ratios: Optional[pd.DataFrame],
    parcel_out_map: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Return a complete parcel/outlet delivery table with neutral defaults.

    Parameters
    ----------
    delivery_ratios : Optional[pd.DataFrame]
        Parcel-to-outlet delivery-ratio table, if configured.
    parcel_out_map : Mapping[str, Sequence[str]]
        Mapping from parcel IDs to connected outlet IDs.

    Returns
    -------
    pd.DataFrame
        Complete parcel-to-outlet delivery-ratio table.

    Raises
    ------
    ValueError
        If any configured delivery-ratio value lies outside ``[0, 1]``.
    """
    columns = [COL_PID, COL_OID, COL_SDR_F_TO_S, COL_SDR_S_TO_O, COL_NDR_F_TO_S, COL_NDR_S_TO_O]
    if delivery_ratios is None:
        out = pd.DataFrame(columns=columns)
    else:
        out = delivery_ratios.copy()
    existing = {(str(row[COL_PID]), str(row[COL_OID])) for _, row in out.iterrows()}
    rows: List[Dict[str, Any]] = []
    for pid, outlet_ids in parcel_out_map.items():
        for oid in outlet_ids:
            key = (str(pid), str(oid))
            if key in existing:
                continue
            rows.append({
                COL_PID: pid,
                COL_OID: key[1],
                COL_SDR_F_TO_S: 1.0,
                COL_SDR_S_TO_O: 1.0,
                COL_NDR_F_TO_S: 1.0,
                COL_NDR_S_TO_O: 1.0,
            })
    if rows:
        defaults = pd.DataFrame(rows, columns=columns)
        out = defaults if out.empty else pd.concat([out, defaults], ignore_index=True)
    for column in (COL_SDR_F_TO_S, COL_SDR_S_TO_O, COL_NDR_F_TO_S, COL_NDR_S_TO_O):
        if column not in out.columns:
            out[column] = pd.Series(dtype=float)
        out[column] = pd.to_numeric(out[column], errors="raise")
        validate_numeric_columns_in_domain(
            out, [column], FRACTION_DOMAIN, CFG_DELIVERY_RATIOS
        )
    return out.reset_index(drop=True)


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
        domain = _load_domain(cfg, logger)
        parcels = _load_parcels(cfg, domain, logger)
        from basin_bmp_sim.config.scopes import load_scoped_input_tables
        input_tables = load_scoped_input_tables(cfg, parcels, logger, _read_gpkg_input_table)
        input_kwargs = {"input_tables": input_tables} if input_tables is not None else {}

        up = _load_parcel_graph(cfg, logger)
        out = _load_parcel_outlets(cfg, logger)
        sel = _load_parcel_selection(cfg, parcels, logger, **input_kwargs)

        parcel_ids = [int(pid) for pid in parcels[COL_PID].tolist()]
        source_parcel_ids = [
            int(pid) for pid in parcels.attrs.get("source_pid_universe", parcel_ids)
        ]
        parcel_down_map = _build_parcel_down_map(
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

        _validate_parcel_outlet_topology(parcel_down_map, parcel_out_map)

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


        outlet_loc = _load_outlet_loc(cfg, domain, logger)
        valid_oids = set(outlet_loc[COL_OID].astype(str))
        referenced_oids = {oid for values in parcel_out_map.values() for oid in values}
        unknown_oids = sorted(referenced_oids - valid_oids)
        if unknown_oids:
            raise ValueError(
                "parcel_outlets references outlet IDs not found in outlets layer: "
                f"{unknown_oids[:10]}"
            )
        outlet_target = _load_optional_outlet_stats(cfg, COL_TARGET, logger)
        outlet_mean = _load_optional_outlet_stats(cfg, COL_MEAN, logger)
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
        bmp_eff = _load_bmp_efficiency(
            cfg, cps, pollutants, logger, complete_legacy=False,
        )
        bmp_cost = _load_bmp_cost(cfg, cps, logger)

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
            bmp_eff = _complete_plet_bmp_efficiency_coverage(bmp_eff, cps, pollutants, logger)
        else:
            pollutant_load_rate = _load_pollutant_load_rate(
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
                pollutant_load_rate_pathway_fractions = _resolve_aggregate_pathway_fractions(
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

        delivery_ratios = _load_delivery_ratios(cfg, logger, **input_kwargs)
        delivery_ratios = _complete_delivery_ratio_defaults(delivery_ratios, parcel_out_map)

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
