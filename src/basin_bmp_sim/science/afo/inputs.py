"""Read and validate animal feeding operation data."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence
import sqlite3
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.constants import AFO_FEEDLOT_MAX_CONCENTRATIONS as _FEEDLOT_MAX_CONC
from basin_bmp_sim.config.distributions import stats_from_row, statistic_columns
from basin_bmp_sim.config.validation.statistics import validate_named_parameter_stats, validate_stats_rows
from basin_bmp_sim.science.plet_classification import normalize_plet_land_cover

from .shared import _animal_species, _need_columns, _finite
from .excretion import reference_excretion_factors, _EXCRETION_PARAMETERS


def _afo_id(value: Any, label: str) -> str:
    """Normalize operation IDs once, rejecting missing and whitespace-only IDs."""
    if value is None or pd.isna(value):
        raise ValueError(f"{label} must contain nonblank AFO IDs")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{label} must contain nonblank AFO IDs")
    return normalized


def _normalize_afo_ids(frame: pd.DataFrame, label: str) -> None:
    if "afo_id" in frame.columns:
        frame["afo_id"] = frame["afo_id"].map(lambda value: _afo_id(value, label))


def _read_table(conn: sqlite3.Connection, name: str, *, optional: bool = False) -> pd.DataFrame:
    names = {str(x[0]) for x in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if name not in names:
        if optional:
            return pd.DataFrame()
        raise ValueError(f"AFO GeoPackage is missing table {name!r}")
    # AFO example GeoPackages expose operation point geometry and explanatory
    # per-record notes in QGIS. Neither is a numerical model input; in
    # particular, the notes field must not be mistaken for a distribution
    # statistic in legacy concentration-increment mode.
    model_columns = [col[1] for col in conn.execute(f'PRAGMA table_info("{name}")')
                     if str(col[1]).strip().lower() not in {"fid", "geom", "notes"}]
    projection = ", ".join('"' + col.replace('"', '""') + '"' for col in model_columns)
    frame = pd.read_sql_query(f'SELECT {projection} FROM "{name}"', conn)
    frame.columns = [str(col).strip().lower() for col in frame.columns]
    return frame


def _integer_pid(value: Any, name: str) -> str:
    if value is None or pd.isna(value):
        raise ValueError(f"{name} must be an integer parcel ID")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer parcel ID: {value!r}") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be a finite integer parcel ID: {value!r}")
    return str(int(number))


def _require_fixed_parameter(
    row: Mapping[str, Any], name: str, label: str,
    *, low: float = 0., high: float | None = None, exclusive_low: bool = False,
) -> float:
    """Reject ambiguous/distribution columns for explicitly fixed AFO fields."""
    stats = validate_named_parameter_stats(
        row, name, label, required=True, low=low, high=high,
        exclusive_low=exclusive_low,
    )
    if set(stats) != {"value"} or pd.isna(row.get(name)):
        raise ValueError(f"{label}: {name} must be one fixed scalar, not a distribution")
    return float(stats["value"])


def load_afo_inputs(path: str | Path, parcels: pd.DataFrame, pollutants: Sequence[str],
                    *, method: str = "concentration_increment", mode: str = "plet_rusle",
                    pathways: Sequence[str] = ("surface",), surface_pathway: str = "surface",
                    parcel_path: str | Path | None = None,
                    excretion_factors_path: str | Path | None = None,
                    runoff_adjusted_export: bool = False,
                    reference_runoff_mm: float | None = None,
                    feedlot_bmp_cps: Sequence[int] = ()) -> dict[str, Any]:
    """Load and validate AFO inputs and the optional mass-based runoff adjustment."""
    if not isinstance(runoff_adjusted_export, bool):
        raise ValueError("manure_loading.runoff_adjusted_export must be boolean")
    if method == "mass_based":
        if runoff_adjusted_export:
            if reference_runoff_mm is None:
                raise ValueError("runoff-adjusted manure export requires reference_runoff_mm")
            reference_runoff_mm = _finite(
                reference_runoff_mm, "manure_loading.reference_runoff_mm", low=1e-12
            )
        elif reference_runoff_mm is not None:
            raise ValueError("reference_runoff_mm requires runoff_adjusted_export: true")
    elif runoff_adjusted_export or reference_runoff_mm is not None:
        raise ValueError("runoff-adjusted manure export requires method: mass_based")
    package = Path(path)
    if not package.is_file():
        raise FileNotFoundError(f"AFO GeoPackage not found: {package}")
    with sqlite3.connect(package) as conn:
        operations = _read_table(conn, "afo_operations")
        animals = _read_table(conn, "afo_animals")
        fields = _read_table(conn, "afo_manure_fields", optional=True)
        increments = _read_table(conn, "afo_manure_concentration_increments", optional=method == "mass_based")
        management = _read_table(conn, "afo_manure_management", optional=method != "mass_based")
        eligibility = _read_table(conn, "afo_field_eligibility", optional=mode != "statistical")
        hydrology = _read_table(conn, "afo_hydrology", optional=mode != "statistical")
        recovery = _read_table(conn, "afo_feedlot_manure_effects", optional=True)
        cost_dimensions = _read_table(conn, "afo_feedlot_bmp_cost_dimensions", optional=True)
    operation_columns = ["afo_id", "pid", "feedlot_area_ha", "paved_pct", "allocation"]
    excretion_factors = {}
    if method == "concentration_increment":
        operation_columns.append("manure_months")
    _need_columns(operations, "afo_operations", operation_columns)
    if "manure_months" not in operations.columns:
        operations["manure_months"] = 12.0
    _need_columns(animals, "afo_animals", ["afo_id", "species"])
    if method not in {"mass_based", "concentration_increment"}:
        raise ValueError("manure_loading.method must be mass_based or concentration_increment")
    if method == "concentration_increment":
        _need_columns(increments, "afo_manure_concentration_increments", ["land_cover", "pollutant"])
    else:
        _need_columns(management, "afo_manure_management", ["afo_id"])
        if mode == "statistical":
            _need_columns(eligibility, "afo_field_eligibility", ["pid", "eligible"])
            _need_columns(hydrology, "afo_hydrology", ["pid"])
            if surface_pathway not in pathways:
                raise ValueError(f"Configured surface pathway {surface_pathway!r} is not an active statistical pathway")
    if not fields.empty:
        _need_columns(fields, "afo_manure_fields", ["afo_id", "pid"])
    if not recovery.empty:
        if method != "mass_based":
            raise ValueError("afo_feedlot_manure_effects requires mass_based manure loading")
        _need_columns(recovery, "afo_feedlot_manure_effects", ["afo_id", "cps"])
    if operations.empty:
        raise ValueError("AFO package requires at least one operation")
    for table, label in ((operations, "afo_operations.afo_id"),
                         (animals, "afo_animals.afo_id"),
                         (fields, "afo_manure_fields.afo_id"),
                         (management, "afo_manure_management.afo_id"),
                         (recovery, "afo_feedlot_manure_effects.afo_id"),
                         (cost_dimensions, "afo_feedlot_bmp_cost_dimensions.afo_id")):
        _normalize_afo_ids(table, label)
    operations["pid"] = operations["pid"].astype(object)
    if operations["afo_id"].duplicated().any():
        raise ValueError("afo_operations.afo_id must contain distinct nonblank IDs")
    op_ids = set(operations["afo_id"])
    if not recovery.empty:
        cps_allowed = {int(c) for c in feedlot_bmp_cps}
        cps_normalized = []
        for _, row in recovery.iterrows():
            op_id = row["afo_id"]
            if op_id not in op_ids:
                raise ValueError(f"afo_feedlot_manure_effects references unknown operation {op_id!r}")
            cps = _integer_pid(row["cps"], "afo_feedlot_manure_effects.cps")
            cps = int(cps)
            if cps not in cps_allowed:
                raise ValueError(f"afo_feedlot_manure_effects CPS {cps} must be in feedlot_bmp_cps")
            cps_normalized.append(cps)
            validate_named_parameter_stats(
                row, "additional_collection_fraction",
                f"afo_feedlot_manure_effects[{op_id}/{cps}]",
                required=True, low=0., high=1.,
            )
        recovery["cps"] = cps_normalized
        if recovery.duplicated(["afo_id", "cps"]).any():
            raise ValueError("Duplicate afo_feedlot_manure_effects operation/CPS pair")
    # Optional operation/CPS-specific cost dimensions. A feedlot BMP acts on
    # the host parcel but its cost area is the treated feedlot footprint, not
    # the parcel's entire area. Length-based costs have no defensible default.
    cost_dimensions_by_op_cps: dict[tuple[str, int], dict[str, float]] = {}
    if not cost_dimensions.empty:
        _need_columns(cost_dimensions, "afo_feedlot_bmp_cost_dimensions", ["afo_id", "cps"])
        allowed = {int(c) for c in feedlot_bmp_cps}
        for index, row in cost_dimensions.iterrows():
            op_id = str(row["afo_id"]).strip()
            if op_id not in op_ids:
                raise ValueError(
                    f"afo_feedlot_bmp_cost_dimensions row {index}: unknown AFO {op_id!r}"
                )
            cps = int(_integer_pid(row["cps"], "afo_feedlot_bmp_cost_dimensions.cps"))
            if cps not in allowed:
                raise ValueError(
                    f"afo_feedlot_bmp_cost_dimensions[{op_id}/{cps}]: "
                    "CPS must be configured in feedlot_bmp_cps"
                )
            key = (op_id, cps)
            if key in cost_dimensions_by_op_cps:
                raise ValueError(f"Duplicate AFO feedlot cost dimensions for {op_id}/CPS {cps}")
            dims = {}
            for field in ("cost_area_ha", "cost_length_m"):
                label = f"afo_feedlot_bmp_cost_dimensions[{op_id}/{cps}]"
                supplied = validate_named_parameter_stats(
                    row, field, label, low=0., exclusive_low=True,
                )
                if supplied:
                    if set(supplied) != {"value"} or pd.isna(row.get(field)):
                        raise ValueError(f"{label}: {field} must be a fixed scalar")
                    dims[field] = float(supplied["value"])
            if not dims:
                raise ValueError(
                    f"afo_feedlot_bmp_cost_dimensions[{op_id}/{cps}] "
                    "requires cost_area_ha and/or cost_length_m"
                )
            cost_dimensions_by_op_cps[key] = dims
    pid_set = set(parcels["pid"].astype(str))
    area_by_pid = dict(zip(parcels["pid"].astype(str), parcels["area_ha"].astype(float)))
    for k, row in operations.iterrows():
        op_id = row["afo_id"]
        pid = _integer_pid(row["pid"], f"afo_operations[{op_id}].pid")
        if pid not in pid_set:
            raise ValueError(f"Operation {op_id} refers to unknown host pid {pid}")
        operations.at[k, "pid"] = pid
        _require_fixed_parameter(row, "feedlot_area_ha", f"afo_operations[{op_id}]",
                                 exclusive_low=True)
        area = _finite(row["feedlot_area_ha"], f"{op_id}.feedlot_area_ha", low=1e-12)
        if area > area_by_pid[pid] + 1e-8:
            raise ValueError(f"Operation {op_id} feedlot area exceeds host parcel area")
        _require_fixed_parameter(row, "paved_pct", f"afo_operations[{op_id}]", high=100.)
        if method == "concentration_increment":
            _require_fixed_parameter(row, "manure_months", f"afo_operations[{op_id}]",
                                     high=12.)
        allocation_mode = str(row["allocation"]).strip().lower()
        if allocation_mode not in {"explicit", "sampled"}:
            raise ValueError(f"Operation {op_id} allocation must be explicit or sampled")
        operations.at[k, "allocation"] = allocation_mode
        if allocation_mode == "sampled":
            selection = str(row.get("selection", "nearest") or "nearest").strip().lower()
            if selection not in {"nearest", "random"}:
                raise ValueError(f"Operation {op_id} selection must be nearest or random")
            operations.at[k, "selection"] = selection
            validate_named_parameter_stats(
                row, "n_fields", f"afo_operations[{op_id}]",
                required=True, low=1.,
            )
    host_area = {}
    for _, row in operations.iterrows():
        host_area[row["pid"]] = host_area.get(row["pid"], 0.) + float(row["feedlot_area_ha"])
    for pid, total in host_area.items():
        if total > area_by_pid[pid] + 1e-8:
            raise ValueError(f"Combined feedlot footprints exceed parcel area for pid {pid}")
    for op_id in op_ids:
        if not animals["afo_id"].eq(op_id).any():
            raise ValueError(f"Operation {op_id} has no animal inventory")
    for _, row in animals.iterrows():
        op_id = str(row["afo_id"]).strip()
        if op_id not in op_ids:
            raise ValueError(f"afo_animals references unknown AFO {op_id}")
        if not str(row["species"]).strip():
            raise ValueError("afo_animals.species must not be blank")
        validate_named_parameter_stats(
            row, "count", f"afo_animals[{op_id}/{row['species']}]",
            required=True, low=0.,
        )
        validate_named_parameter_stats(
            row, "animal_days", f"afo_animals[{op_id}/{row['species']}]",
            low=0., high=366.,
        )
        for pol in set(p.upper() for p in pollutants).intersection(_FEEDLOT_MAX_CONC):
            factor = f"factor_{pol.lower()}"
            if factor not in animals or pd.isna(row[factor]):
                raise ValueError(f"afo_animals requires {factor} for {op_id}; PLET equivalent animal factor")
            _require_fixed_parameter(row, factor, f"afo_animals[{op_id}/{row['species']}]")
    if fields.empty and "afo_id" not in fields:
        fields = pd.DataFrame(columns=["afo_id", "pid"])
    fields["pid"] = fields["pid"].astype(object)
    # Check relations after canonical integer PID and AFO-ID conversion: 1,
    # 1.0, and " 1 " all represent the same parcel, not distinct receivers.
    for k, row in fields.iterrows():
        op_id = row["afo_id"]
        pid = _integer_pid(row["pid"], "afo_manure_fields.pid")
        if op_id not in op_ids or pid not in pid_set:
            raise ValueError(f"Invalid manure field reference {op_id}/{pid}")
        fields.at[k, "pid"] = pid
    if fields.duplicated(["afo_id", "pid"]).any():
        raise ValueError("Duplicate operation-to-receiving-parcel relation")
    for _, row in fields.iterrows():
        op_id, pid = row["afo_id"], row["pid"]
        if validate_named_parameter_stats(
            row, "weight", f"afo_manure_fields[{op_id}/{pid}]",
            low=0., exclusive_low=True,
        ):
            _require_fixed_parameter(row, "weight", f"afo_manure_fields[{op_id}/{pid}]",
                                     exclusive_low=True)
    if fields["afo_id"].isin(set(operations.loc[operations.allocation == "sampled", "afo_id"])).any():
        raise ValueError("Sampled operations must not also specify explicit manure fields")
    for op_id in operations.loc[operations.allocation == "explicit", "afo_id"]:
        op_fields = fields.loc[fields["afo_id"].eq(op_id)]
        if op_fields.empty:
            raise ValueError(f"Explicit operation {op_id} has no manure-receiving fields")
        # Custom weights are an all-or-none choice for each mass-based AFO.
        # Reject incomplete assignments before any Monte Carlo workers start.
        if method == "mass_based" and "weight" in op_fields:
            has_weight = op_fields["weight"].notna()
            if has_weight.any() and not has_weight.all():
                raise ValueError(
                    f"afo_manure_fields[{op_id}]: all explicit receiving fields "
                    "must provide a positive weight if any field has a weight"
                )
    if method == "concentration_increment":
        increments["land_cover"] = increments["land_cover"].map(normalize_plet_land_cover)
        increments["pollutant"] = increments["pollutant"].astype(str).str.upper().str.strip()
        if increments.duplicated(["land_cover", "pollutant"]).any():
            raise ValueError("Duplicate manure concentration increments by land_cover/pollutant")
        if not set(increments["land_cover"]).issubset({"cropland", "pastureland"}):
            raise ValueError("Manure concentration increments apply only to cropland/pastureland")
        if not set(increments["pollutant"]).issubset(set(map(str.upper, pollutants))):
            raise ValueError("Manure concentration increments reference unknown pollutants")
        # PLET land cover can vary between realizations. Before simulation,
        # require coverage for both possible agricultural receiver classes and
        # every active pollutant with a modeled manure concentration increment.
        incremental_pollutants = {str(pol).upper() for pol in pollutants} & {"TN", "TP", "BOD"}
        required_pairs = {(cover, pol) for cover in ("cropland", "pastureland")
                          for pol in incremental_pollutants}
        present_pairs = set(zip(increments["land_cover"], increments["pollutant"]))
        missing_pairs = sorted(required_pairs - present_pairs)
        if missing_pairs:
            raise ValueError(
                "afo_manure_concentration_increments is missing required "
                f"land_cover/pollutant combinations: {missing_pairs}"
            )
        for index, row in increments.iterrows():
            stats = stats_from_row(row, {"land_cover", "pollutant", "sample_group", "units"})
            if not stats:
                raise ValueError("Each manure concentration increment needs a value or distribution")
            validate_stats_rows(
                pd.DataFrame([{"parameter": "increment", **stats}]),
                f"afo_manure_concentration_increments row {index}",
            )
            for name, value in stats.items():
                if name != "sd":
                    _finite(value, f"afo_manure_concentration_increments.{name}")
    else:
        excretion_factors = reference_excretion_factors(excretion_factors_path)
        if management["afo_id"].astype(str).duplicated().any() or set(management["afo_id"].astype(str)) != op_ids:
            raise ValueError("afo_manure_management requires exactly one row per operation")
        # APLE-specific coefficients are not interchangeable with annual
        # export fractions. Fail rather than silently ignoring old inputs.
        deprecated = (
            "f_n", "alpha_n", "wep_fraction", "solids_pct", "season",
            "incorporated_fraction",
        )
        for _, row in management.iterrows():
            op_id = str(row["afo_id"])
            old = [name for name in deprecated if any(
                (column == name or column.startswith(name + "_")) and pd.notna(value)
                for column, value in row.items()
            )]
            if old:
                raise ValueError(
                    f"afo_manure_management[{op_id}] uses obsolete APLE parameters {old}; "
                    "replace with tn_surface_export_fraction and tp_surface_export_fraction "
                    "(do not convert coefficients mechanically)"
                )
            for name in ("tn_surface_export_fraction", "tp_surface_export_fraction"):
                validate_named_parameter_stats(
                    row, name, f"afo_manure_management[{op_id}]",
                    required=True, low=0., high=1.,
                )
            for name in ("collection_fraction", "export_fraction",
                         "n_retention_fraction", "p_retention_fraction"):
                validate_named_parameter_stats(
                    row, name, f"afo_manure_management[{op_id}]",
                    low=0., high=1.,
                )
        for _, row in animals.iterrows():
            species = _animal_species(row["species"])
            unsupported = [
                column for column in animals.columns
                if any(column == name or column.startswith(name + "_")
                       for name in ("manure_kg_head_day", "tn_kg_head_day", "tp_kg_head_day",
                                    "manure_kg_1000kg_day"))
                and pd.notna(row[column])
            ]
            if unsupported:
                raise ValueError(
                    f"afo_animals contains unsupported excretion inputs {unsupported}; "
                    f"supported parameters are {', '.join(_EXCRETION_PARAMETERS)}"
                )
            for parameter in _EXCRETION_PARAMETERS:
                stats = validate_named_parameter_stats(
                    row, parameter, f"afo_animals[{row['afo_id']}/{species}]",
                    low=0., exclusive_low=(parameter == "representative_mass_kg"),
                )
                if not stats and parameter not in excretion_factors.get(species, {}):
                    raise ValueError(
                        f"afo_animals[{row['afo_id']}/{species}] requires {parameter}; "
                        "supply an animal-row override or a species row in the excretion CSV"
                    )
    eligible_by_pid = {}
    hydrology_by_pid = {}
    if method == "mass_based" and mode == "statistical":
        for _, row in eligibility.iterrows():
            pid = _integer_pid(row["pid"], "afo_field_eligibility.pid")
            if pid not in pid_set or pid in eligible_by_pid:
                raise ValueError(f"Unknown/duplicate AFO eligibility pid {pid}")
            if float(row["eligible"]) not in (0., 1.):
                raise ValueError("AFO eligibility must be 0 or 1")
            eligible_by_pid[pid] = bool(float(row["eligible"]))
        # Inheritance: parcel-specific -> HUC12 -> NULL-pid domain default.
        # HUC12 keys are resolved from the canonical parcel_huc12 table.
        mapping = {}
        if parcel_path is not None:
            with sqlite3.connect(parcel_path) as con:
                names = {x[0] for x in con.execute("SELECT name FROM sqlite_master")}
                if "parcel_huc12" in names:
                    cols = {info[1] for info in con.execute("PRAGMA table_info(parcel_huc12)")}
                    huc_col = "huc_id" if "huc_id" in cols else "huc12"
                    mapping = {str(int(pid)): str(huc) for pid, huc in
                               con.execute(f"SELECT pid,{huc_col} FROM parcel_huc12")}
        # The legacy runoff override has been removed. Reject even a NULL
        # column to prevent silently retaining an obsolete GeoPackage schema.
        unsupported = [column for column in hydrology.columns
                       if column == "feedlot_runoff_mm" or column.startswith("feedlot_runoff_mm_")]
        if unsupported:
            raise ValueError(
                "afo_hydrology no longer accepts feedlot_runoff_mm; remove "
                f"obsolete column(s) {unsupported}. Feedlot runoff is derived "
                "from annual_precip_mm with the PLET curve-number method."
            )
        by_pid, by_huc, default = {}, {}, None
        for _, row in hydrology.iterrows():
            rec = row.to_dict()
            pid_value = rec.get("pid")
            huc_value = rec.get("huc12")
            has_pid = pid_value is not None and not pd.isna(pid_value)
            has_huc = huc_value is not None and not pd.isna(huc_value)
            if has_pid and has_huc:
                raise ValueError("afo_hydrology row cannot specify both pid and huc12")
            if has_pid:
                pid = _integer_pid(pid_value, "afo_hydrology.pid")
                if pid not in pid_set or pid in by_pid:
                    raise ValueError(f"Unknown/duplicate AFO hydrology pid {pid}")
                by_pid[pid] = rec
            elif has_huc:
                key = str(huc_value).strip()
                if key in by_huc:
                    raise ValueError(f"Duplicate AFO hydrology huc12 {key}")
                by_huc[key] = rec
            else:
                if default is not None:
                    raise ValueError("Duplicate default AFO hydrology row")
                default = rec
            for name in ("annual_runoff_mm", "annual_precip_mm"):
                validate_named_parameter_stats(
                    row, name, f"afo_hydrology row {row.name}", low=0.,
                )
            validate_named_parameter_stats(
                row, "rain_days", f"afo_hydrology row {row.name}", low=1., high=366.,
            )
        for pid in pid_set:
            rec = by_pid.get(pid) or by_huc.get(mapping.get(pid, "")) or default
            if rec is not None:
                hydrology_by_pid[pid] = rec
        active_hosts = {str(op["pid"]) for op in operations.to_dict("records")}
        # Constant-export mode needs no receiving-field hydrology. Direct
        # feedlot loading only needs precipitation at operation hosts.
        required_pids = active_hosts | ({pid for pid, yes in eligible_by_pid.items() if yes} if runoff_adjusted_export else set())
        for pid in required_pids:
            if pid not in hydrology_by_pid:
                raise ValueError(f"Missing statistical AFO hydrology for pid {pid}")
            row = hydrology_by_pid[pid]
            required = ["annual_runoff_mm"] if runoff_adjusted_export and eligible_by_pid.get(pid) else []
            if pid in active_hosts:
                required.append("annual_precip_mm")
            for name in required:
                if not (name in row and pd.notna(row[name])) and not any(
                    k.startswith(name + "_") and pd.notna(v) for k,v in row.items()
                ):
                    raise ValueError(f"Statistical AFO pid {pid} missing {name} or distribution")
        # Eligibility and feedlot footprints are fixed in statistical mode,
        # so the number of usable receiving parcels is known before sampling.
        available_receivers = sum(
            1 for pid, permitted in eligible_by_pid.items()
            if permitted and area_by_pid[pid] - host_area.get(pid, 0.) > 1e-12
        )
        if available_receivers == 0:
            raise ValueError("Statistical AFO requires at least one eligible field with positive available area")
        for _, op in operations.loc[operations.allocation == "sampled"].iterrows():
            validate_named_parameter_stats(
                op, "n_fields", f"afo_operations[{op['afo_id']}]",
                required=True, low=1., high=float(available_receivers),
            )
    return {"operations": operations.to_dict("records"), "animals": animals.to_dict("records"),
            "excretion_factors": excretion_factors,
            "fields": fields.to_dict("records"),
            "manure_concentration_increments": increments.to_dict("records") if not increments.empty else [],
            "management": management.to_dict("records") if not management.empty else [],
            "feedlot_manure_effects": recovery.to_dict("records") if not recovery.empty else [],
            "feedlot_cost_dimensions": cost_dimensions_by_op_cps,
            "eligible": eligible_by_pid, "hydrology": hydrology_by_pid,
            "surface_pathway": surface_pathway, "method": method, "mode": mode,
            "runoff_adjusted_export": runoff_adjusted_export,
            "reference_runoff_mm": reference_runoff_mm,
            "geometry": {str(r.pid): r.geometry for r in parcels.itertuples()},
            "area_ha": area_by_pid}
