"""GeoPackage/CSV input-table preparation and normalization."""

from __future__ import annotations

from basin_bmp_sim.common.utils import is_nonblank
from pathlib import Path
import difflib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import geopandas as gpd
import numpy as np
import pandas as pd
from basin_bmp_sim.domain.constants import (
    CFG_PARCELS,
    COL_CPS,
    COL_OID,
    COL_PID,
    COL_PID_DOWN,
    COL_POLLUTANT,
    COL_PATHWAY,
    GPKG_PARCELS_LAYER,
    GPKG_PARCEL_DOWN_TABLE,
    GPKG_PARCEL_OUTLETS_TABLE,
    GPKG_PARCEL_HUC12_TABLE,
    PARCEL_PARAMETER_INPUT_TABLES,
    GPKG_INPUT_CURVE_NUMBER,
    GPKG_INPUT_INFILTRATION_FRACTION,
    GPKG_INPUT_SURFACE_CONCENTRATION,
    GPKG_INPUT_SUBSURFACE_CONCENTRATION,
)
from basin_bmp_sim.common.io import (
    MissingInputTableError,
    list_geopackage_tables,
    read_csv_table,
    read_geopackage_table,
    read_geopackage_table_info,
)
from basin_bmp_sim.common.utils import ci_get, normalize_columns, normalize_pollutant_label
from basin_bmp_sim.common.distributions import statistic_columns, stats_from_row
from basin_bmp_sim.config.validation.statistics import require_columns, validate_unique_rows
from basin_bmp_sim.domain.parameter_validation import validate_numeric_columns_in_domain
from basin_bmp_sim.config.schema import (
    INPUT_SCHEMA_TABLE,
    INPUT_SCHEMA_VERSION,
    INPUT_VARIABLE_SPECS,
    KNOWN_INPUT_TABLES,
    required_input_tables,
)



def _normalize_integer_identifier_value(
    value: Any, label: str, *, allow_null: bool = False
) -> Optional[int]:
    """Normalize a parcel identifier to an integer.

    ``NULL`` is reserved for package-wide default rows in tables whose schema
    explicitly permits defaults. The legacy ``"*"`` sentinel is rejected so
    every non-null PID is always a real parcel identifier.
    """
    if value is None or pd.isna(value):
        if allow_null:
            return None
        raise ValueError(f"{label} must not be null")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an integer parcel identifier, not a boolean")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{label} must not be blank")
        if text == "*":
            raise ValueError(
                f"{label} uses the legacy '*' parcel default; use NULL pid for a default row"
            )
        value = text
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}={value!r} must be an integer parcel identifier") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label}={value!r} must be a finite integer parcel identifier")
    return int(numeric)


def _normalize_text_identifier_value(
    value: Any, label: str, *, allow_null: bool = False
) -> Optional[str]:
    """Normalize an outlet-style identifier while optionally permitting NULL defaults."""
    if value is None or pd.isna(value):
        if allow_null:
            return None
        raise ValueError(f"{label} must not be null")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be an identifier, not a boolean")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be blank")
    if text == "*":
        raise ValueError(
            f"{label} uses the legacy '*' default sentinel; use NULL in all default key columns"
        )
    return text


def _normalize_identifier_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    label: str,
    *,
    allow_null_pid: bool = False,
    allow_null_oid: bool = False,
) -> pd.DataFrame:
    """Return a copy with canonical integer PIDs and normalized outlet IDs."""
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        if column in {COL_PID, COL_PID_DOWN}:
            allow_null = allow_null_pid and column == COL_PID
            values = [
                _normalize_integer_identifier_value(
                    value, f"{label} row {index} {column}", allow_null=allow_null
                )
                for index, value in out[column].items()
            ]
            out[column] = pd.array(values, dtype="Int64" if allow_null else "int64")
        else:
            values = [
                _normalize_text_identifier_value(
                    value, f"{label} row {index} {column}", allow_null=allow_null_oid
                )
                for index, value in out[column].items()
            ]
            out[column] = values
    return out


def _validate_explicit_references(
    values: Sequence[Any],
    valid_values: Sequence[Any],
    *,
    label: str,
    allow_default: bool = False,
) -> None:
    """Reject explicit foreign-key values that do not exist in the target universe."""
    valid = set(valid_values)
    supplied = set()
    saw_null = False
    for value in values:
        if value is None or pd.isna(value):
            saw_null = True
            continue
        supplied.add(value)
    if saw_null and not allow_default:
        raise ValueError(f"{label} contains NULL identifiers where defaults are not allowed")
    unknown = sorted(supplied - valid, key=lambda value: str(value))
    if unknown:
        raise ValueError(f"{label} references unknown identifiers: {unknown[:10]}")


def _normalize_cps_column(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Normalize a CPS column and reject booleans, fractions, and non-finite codes."""
    if COL_CPS not in df.columns:
        return df
    out = df.copy()
    normalized: List[int] = []
    for index, value in out[COL_CPS].items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{label} row {index} cps must be a finite integer, not a boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} row {index} cps={value!r} must be a finite integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{label} row {index} cps={value!r} must be a finite integer")
        normalized.append(int(numeric))
    out[COL_CPS] = normalized
    return out


def _sort_input_table(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Return deterministic row ordering without changing stored key values."""
    if df.empty or not keys:
        return df.reset_index(drop=True)
    out = df.copy()
    temp_cols: List[str] = []
    for index, key in enumerate(keys):
        if key not in out.columns:
            continue
        temp = f"__sort_key_{index}"
        out[temp] = out[key].map(lambda value: "" if pd.isna(value) else str(value).strip())
        temp_cols.append(temp)
    if temp_cols:
        out = out.sort_values(temp_cols, kind="stable").drop(columns=temp_cols)
    return out.reset_index(drop=True)


def _validate_input_package_schema(
    cfg: Dict[str, Any], load_mode: str, logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> None:
    """Reject unknown input tables and require mode-specific tables."""
    raw_package = ci_get(cfg, CFG_PARCELS)
    # The spatial parcel loader owns path existence/requiredness. Keeping this
    # helper focused on schema discovery also makes it easy to unit-test loader
    # stages independently with mocked spatial inputs.
    if raw_package is None:
        return
    package = Path(raw_package)
    if not package.exists():
        return
    table_names = set(list_geopackage_tables(package))
    supplied_input_tables = {name for name in table_names if name.startswith("input_")}
    unknown = sorted(supplied_input_tables - set(KNOWN_INPUT_TABLES))
    if unknown:
        details = []
        known = sorted(KNOWN_INPUT_TABLES)
        for name in unknown:
            match = difflib.get_close_matches(name, known, n=1, cutoff=0.65)
            details.append(f"{name!r}" + (f" (did you mean {match[0]!r}?)" if match else ""))
        raise ValueError("Unknown input_* table(s) in parcels GeoPackage: " + ", ".join(details))

    available = table_names | set(input_tables or {})
    missing = sorted(set(required_input_tables(load_mode)) - available)
    if missing:
        raise ValueError(
            f"load_generation.mode={load_mode!r} requires input table(s) missing from parcels/HUC/domain GeoPackages: {missing}"
        )

    if INPUT_SCHEMA_TABLE in table_names:
        schema = read_geopackage_table(package, INPUT_SCHEMA_TABLE)
        schema = normalize_columns(schema)
        require_columns(schema, ["schema_version"], INPUT_SCHEMA_TABLE, logger)
        versions = pd.to_numeric(schema["schema_version"], errors="coerce")
        if len(schema) != 1 or versions.isna().any() or int(versions.iloc[0]) != INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"{INPUT_SCHEMA_TABLE} must contain exactly one row with schema_version={INPUT_SCHEMA_VERSION}"
            )

        parcel_info = read_geopackage_table_info(package, GPKG_PARCELS_LAYER)
        parcel_columns = {row["name"].lower(): row for row in parcel_info}
        pid_meta = parcel_columns.get(COL_PID)
        if (
            pid_meta is None
            or "INT" not in pid_meta["type"].upper()
            or int(pid_meta["pk"]) != 1
            or "fid" in parcel_columns
        ):
            raise ValueError(
                "GeoPackage schema v3 requires parcels.pid to be the INTEGER PRIMARY KEY "
                "and does not use a separate fid column"
            )

        editable_tables = (set(KNOWN_INPUT_TABLES) | {GPKG_PARCEL_DOWN_TABLE, GPKG_PARCEL_OUTLETS_TABLE, GPKG_PARCEL_HUC12_TABLE, "parcel_huc"}) & table_names
        for table_name in sorted(editable_tables):
            info = read_geopackage_table_info(package, table_name)
            metadata = {row["name"].lower(): row for row in info}
            if not any(int(row["pk"]) > 0 for row in info):
                raise ValueError(
                    f"GeoPackage schema v3 requires {table_name!r} to have an integer primary-key row ID so it is editable in QGIS"
                )
            for pid_column in (COL_PID, COL_PID_DOWN):
                if pid_column in metadata and "INT" not in metadata[pid_column]["type"].upper():
                    raise ValueError(
                        f"GeoPackage schema v3 requires {table_name}.{pid_column} to use INTEGER storage"
                    )
    else:
        logger.warning(
            f"{INPUT_SCHEMA_TABLE} is not present in {package.name}; accepting the current table layout, "
            f"but future input packages should declare schema_version={INPUT_SCHEMA_VERSION}"
        )


def _merge_csvs(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one or more CSV files and combine them into one table.

    Parameters
    ----------
    paths : str, pathlib.Path, or sequence of str or pathlib.Path
        One or more CSV file paths.
    required_cols : sequence of str
        Columns that must be present in every file.
    label : str
        Human-readable dataset name used in logs and errors.
    logger : Any
        Logger used for progress and duplicate warnings.

    Returns
    -------
    pandas.DataFrame
        Concatenated dataframe with duplicates removed on the required key
        columns.
    """
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames: List[pd.DataFrame] = []
    for p in paths:
        logger.verbose(f"Reading {label} from {p}")
        df = read_csv_table(p)
        df = normalize_columns(df)
        require_columns(df, required_cols, f"{label} ({p})", logger)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    dedup_subset = list(required_cols)
    if COL_PATHWAY in out.columns and COL_PATHWAY not in dedup_subset:
        dedup_subset.append(COL_PATHWAY)

    # Exact duplicate rows are harmless and may arise when users concatenate
    # identical source files. Conflicting rows for the same logical key are not:
    # accepting them would make file order determine model behavior.
    out = out.drop_duplicates(keep="first")
    dup = out.duplicated(subset=dedup_subset, keep=False)
    if dup.any():
        preview = out.loc[dup, dedup_subset].head(10).to_dict(orient="records")
        raise ValueError(
            f"{label} contains conflicting duplicate rows for logical key {dedup_subset}: {preview}"
        )
    return out.reset_index(drop=True)


def _read_gpkg_input_table(
    package_path: Union[str, Path],
    table_name: str,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Read one named attribute table from a consolidated GeoPackage input.

    The GeoPackage is a physical container only: each logical dataset retains
    its own normalized relational table and downstream model contracts remain
    unchanged.
    """
    package = Path(package_path)
    if not package.exists():
        raise FileNotFoundError(f"{label} GeoPackage not found: {package}")
    logger.verbose(f"Reading {label} from {package} layer={table_name}")
    try:
        frame = read_geopackage_table(package, table_name)
    except MissingInputTableError as exc:
        raise MissingInputTableError(
            f"{label} requires table/layer '{table_name}' in {package}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Failed to read {label} table/layer '{table_name}' from {package}: {exc}"
        ) from exc
    frame = normalize_columns(pd.DataFrame(frame))
    # ``id``/``fid`` are technical SQLite/QGIS row identifiers for attribute
    # tables; they are deliberately not part of the model's logical schema.
    technical_ids = [column for column in ("fid", "id", "row_id") if column in frame.columns and column not in required_cols]
    if technical_ids:
        frame = frame.drop(columns=technical_ids)
    require_columns(frame, required_cols, f"{label} ({package}:{table_name})", logger)
    identifier_cols = [column for column in (COL_PID, COL_PID_DOWN, COL_OID) if column in frame.columns]
    if identifier_cols:
        spec = INPUT_VARIABLE_SPECS.get(table_name)
        # Schema-v3 renamed ``wildcard_allowed`` to ``default_row_allowed``
        # when parcel defaults changed from pid="*" to pid IS NULL. Accept
        # either registry attribute so a partially updated working tree does
        # not crash with AttributeError.
        allow_default = bool(
            spec
            and getattr(
                spec,
                "default_row_allowed",
                getattr(spec, "wildcard_allowed", False),
            )
        )
        frame = _normalize_identifier_columns(
            frame,
            identifier_cols,
            label,
            allow_null_pid=allow_default or table_name in {
                GPKG_INPUT_CURVE_NUMBER, GPKG_INPUT_INFILTRATION_FRACTION,
            },
            allow_null_oid=allow_default and COL_OID in getattr(spec, "key_columns", ()),
        )
    sort_cols = list(required_cols)
    if COL_PATHWAY in frame.columns and COL_PATHWAY not in sort_cols:
        sort_cols.append(COL_PATHWAY)
    return _sort_input_table(frame, sort_cols)


def _try_read_gpkg_input_table(
    package_path: Union[str, Path],
    table_name: str,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> Optional[pd.DataFrame]:
    """Read an optional GeoPackage attribute table, returning ``None`` if absent."""
    try:
        return _read_gpkg_input_table(
            package_path, table_name, required_cols, label, logger
        )
    except MissingInputTableError:
        return None


def _load_fixed_numeric_variable_table(
    cfg: Dict[str, Any],
    table_name: str,
    key_cols: Sequence[str],
    logger: Any,
    *,
    required: bool = False,
    domain: Any = None,
    input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """Load a deterministic per-variable table using the standard input schema."""
    table = _read_model_input_table(
        cfg, table_name, key_cols, logger, required=required, input_tables=input_tables,
    )
    if table is None:
        return None
    if "value" not in table.columns:
        raise ValueError(f"{table_name} requires a value column")
    other_stats = [c for c in statistic_columns(table.columns) if str(c).lower() != "value"]
    if other_stats and table[other_stats].notna().any(axis=1).any():
        raise ValueError(f"{table_name} is deterministic and only allows fixed value")
    table = table.copy()
    table["value"] = pd.to_numeric(table["value"], errors="raise")
    validate_unique_rows(table, list(key_cols), table_name)
    if domain is not None:
        validate_numeric_columns_in_domain(table, ["value"], domain, table_name)
    return table.reset_index(drop=True)


def _read_model_input_table(
    cfg: Mapping[str, Any], table_name: str, keys: Sequence[str], logger: Any,
    *, required: bool = False,
    input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    if input_tables is not None and table_name in input_tables:
        return input_tables[table_name].copy()
    reader = _read_gpkg_input_table if required else _try_read_gpkg_input_table
    return reader(ci_get(cfg, CFG_PARCELS), table_name, keys, table_name, logger)


def _assemble_parcel_parameter_source(
    cfg: Dict[str, Any],
    logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Optional[pd.DataFrame]:
    """Assemble dedicated ``input_*`` parameter tables into the runtime long form."""
    frames: List[pd.DataFrame] = []
    for parameter, table_name in PARCEL_PARAMETER_INPUT_TABLES.items():
        table = _read_model_input_table(
            cfg, table_name, [COL_PID], logger, input_tables=input_tables,
        )
        if table is None:
            continue
        table = table.copy()
        table.insert(1, "parameter", parameter)
        frames.append(table)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


def _assemble_plet_hydrology_source(
    cfg: Dict[str, Any],
    logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Assemble CN and infiltration tables into the existing hydrology long form."""
    specs = (
        ("cn", GPKG_INPUT_CURVE_NUMBER),
        ("infiltration_fraction", GPKG_INPUT_INFILTRATION_FRACTION),
    )
    frames: List[pd.DataFrame] = []
    for parameter, table_name in specs:
        table = _read_model_input_table(
            cfg, table_name, ["land_cover", "hsg"], logger,
            required=True, input_tables=input_tables,
        )
        table.insert(2, "parameter", parameter)
        frames.append(table)
    return pd.concat(frames, ignore_index=True, sort=False)


def _assemble_plet_concentration_source(
    cfg: Dict[str, Any],
    logger: Any,
    *, input_tables: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Assemble dedicated surface/subsurface concentration variable tables."""
    frames: List[pd.DataFrame] = []
    for pathway, table_name in (
        ("surface", GPKG_INPUT_SURFACE_CONCENTRATION),
        ("subsurface", GPKG_INPUT_SUBSURFACE_CONCENTRATION),
    ):
        table = _read_model_input_table(
            cfg, table_name, [COL_PID, COL_POLLUTANT], logger, input_tables=input_tables,
        )
        if table is None:
            continue
        table = table.copy()
        table.insert(2, COL_PATHWAY, pathway)
        frames.append(table)
    if not frames:
        return pd.DataFrame(columns=[COL_PID, COL_POLLUTANT, COL_PATHWAY, "value"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_table_source(
    source: Any,
    required_cols: Sequence[str],
    label: str,
    logger: Any,
) -> pd.DataFrame:
    """Load a logical input table from either an in-memory frame or CSV path(s)."""
    if isinstance(source, pd.DataFrame):
        frame = normalize_columns(source.copy())
        require_columns(frame, required_cols, label, logger)
        return _sort_input_table(frame, required_cols)
    return _merge_csvs(source, required_cols, label, logger)


def _normalize_pathway_label(value: Any) -> str:
    """Return a stable, user-extensible pathway label.

    Parameters
    ----------
    value : Any
        Input value to normalize or evaluate.

    Returns
    -------
    str
        Normalized pathway label.
    """
    label = str(value).strip().lower().replace("_", " ")
    return " ".join(label.split())


def _normalize_pathway_column(df: pd.DataFrame, label: str, logger: Any) -> pd.DataFrame:
    """Normalize pathway labels without restricting user-defined pathways.

    Statistical mode may use any non-empty pathway labels shared by the parcel
    load-rate and BMP efficiency inputs. PLET/RUSLE-specific pathway restrictions
    are applied later, after the load-generation mode is known.

    Parameters
    ----------
    df : pd.DataFrame
        Input table to process.
    label : str
        Context label used in diagnostics and validation errors.
    logger : Any
        Logger used for diagnostic and progress messages.

    Returns
    -------
    pd.DataFrame
        Copy of the table with normalized pathway labels.

    Raises
    ------
    ValueError
        If any pathway label is blank after normalization.
    """
    del logger
    if COL_PATHWAY not in df.columns:
        return df
    df[COL_PATHWAY] = df[COL_PATHWAY].map(_normalize_pathway_label)
    bad = df[COL_PATHWAY].eq("")
    if bad.any():
        raise ValueError(f"{label} contains blank pathway labels")
    return df


def _normalize_pollutant_column(df: pd.DataFrame, col: str, label: str, logger: Any) -> pd.DataFrame:
    """Normalize pollutant labels in a dataframe column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    col : str
        Name of the pollutant column.
    label : str
        Dataset label used in error messages.
    logger : Any
        Logger retained for interface consistency.

    Returns
    -------
    pandas.DataFrame
        Dataframe with standardized pollutant labels.

    Raises
    ------
    ValueError
        If the pollutant column is missing or cannot be normalized.
    """
    if col not in df.columns:
        raise ValueError(f"{label} missing required column '{col}'")
    try:
        df[col] = [normalize_pollutant_label(x) for x in df[col]]
    except Exception as ex:
        raise ValueError(f"Failed to normalize pollutant labels in {label}: {ex}") from ex
    return df






def _ensure_projected(gdf: gpd.GeoDataFrame, logger: Any) -> gpd.GeoDataFrame:
    """Return geometry in a projected CRS whose linear units are meters.

    Area/perimeter values are model inputs, so merely being projected is not
    sufficient: State Plane feet, for example, must not be interpreted as
    meters.  The model therefore establishes a metric analysis CRS once from
    the domain and transforms all other spatial inputs into that CRS.
    """
    if gdf.crs is None:
        raise ValueError("Spatial inputs must declare a CRS; a missing CRS cannot be safely inferred")

    crs = gdf.crs
    axis_info = list(getattr(crs, "axis_info", ()) or ())
    uses_meters = bool(
        crs.is_projected
        and axis_info
        and all(
            getattr(axis, "unit_conversion_factor", None) is not None
            and abs(float(axis.unit_conversion_factor) - 1.0) < 1e-12
            for axis in axis_info[:2]
        )
    )
    if uses_meters:
        return gdf

    est = gdf.estimate_utm_crs()
    if est is None:
        raise ValueError(f"Could not determine a metric analysis CRS from input CRS {crs}")
    logger.info(f"Reprojecting analysis geometry from {crs} to metric CRS: {est}")
    return gdf.to_crs(est)


def _normalize_huc12(value: Any, *, label: str) -> str:
    """Normalize and validate one HUC12 code as a 12-character text value."""
    if value is None or pd.isna(value):
        raise ValueError(f"{label} must not be null")
    text = str(value).strip()
    if not (len(text) == 12 and text.isdigit()):
        raise ValueError(
            f"{label}={value!r} must be a 12-digit HUC stored as text "
            "(for example '050902021001')"
        )
    return text
