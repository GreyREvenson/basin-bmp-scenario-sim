"""Prepare HUC12-scale PLET forcing and parcel assignments from USGS WBD.

Downloads threshold-qualified USGS WBD HUC12 polygons, stores WBD geometry and
metadata in the spatial ``huc12`` layer, stores each HUC12-scale PLET variable
in its own ``input_*`` attribute table using the same fixed/distribution schema
as parcel inputs, and writes ``parcel_huc12`` assignments to the parcel GPKG.

On initialization, ``--initial-plet-export`` can import AVG_RAIN, RAIN_DAYS,
and ANNUAL_RAINFALL from an EPA PLET Excel export as fixed values. Subsequent
WBD refreshes preserve the HUC12 ``input_*`` tables.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import fiona
import geopandas as gpd
import pandas as pd

try:
    # Import path when loaded as ``utils.download_wbd_huc12`` (tests/modules).
    from utils.create_parcel_huc12 import (
        PARCEL_LAYER,
        _normalize_huc12,
        _read_parcels,
        assign_parcels_to_huc12,
        write_parcel_huc12,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    # ``python utils/download_wbd_huc12.py`` places ``utils`` on sys.path.
    from create_parcel_huc12 import (
        PARCEL_LAYER,
        _normalize_huc12,
        _read_parcels,
        assign_parcels_to_huc12,
        write_parcel_huc12,
    )

WBD_HUC12_QUERY_URL = (
    "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query"
)
HUC12_LAYER = "huc12"
DEFAULT_PAGE_SIZE = 2000
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_AREA_FRACTION_THRESHOLD = 0.75

WBD_FIELDS = ("huc12", "name", "states", "areaacres", "areasqkm")
PLET_PARAMETER_TABLES = {
    "avg_rain_in": "input_avg_rain_in",
    "annual_precip_in": "input_annual_precip_in",
    "rain_days": "input_rain_days",
    "rain_correction_fraction": "input_rain_correction_fraction",
    "runoff_day_fraction": "input_runoff_day_fraction",
    "hsg": "input_hsg",
    "r": "input_rusle_r",
    "k": "input_rusle_k",
    "ls": "input_rusle_ls",
    "c": "input_rusle_c",
    "p": "input_rusle_p",
    "sediment_n_pct": "input_sediment_n_pct",
    "sediment_p_pct": "input_sediment_p_pct",
    "surface_concentration": "input_surface_concentration",
    "subsurface_concentration": "input_subsurface_concentration",
}
LAND_COVER_PARAMETERS = frozenset(("r", "k", "ls", "c", "p", "surface_concentration", "subsurface_concentration"))
CONCENTRATION_PARAMETERS = frozenset(("surface_concentration", "subsurface_concentration"))
PLET_FORCING_FIELDS = ("avg_rain_in", "annual_precip_in", "rain_days", "rain_correction_fraction", "runoff_day_fraction")
PLET_PARAMETER_UNITS = {
    "avg_rain_in": "in/event",
    "annual_precip_in": "in/year",
    "rain_days": "days/year",
    "rain_correction_fraction": "fraction",
    "runoff_day_fraction": "fraction",
}
STAT_COLUMNS = (
    "value", "mean", "sd", "min", "p05", "p10", "p25", "p50",
    "p75", "p90", "p95", "max",
)


def parcel_footprint_wgs84(
    parcel_path: Path,
    *,
    parcel_layer: str = PARCEL_LAYER,
) -> Tuple[gpd.GeoDataFrame, Tuple[float, float, float, float]]:
    """Return parcel polygons in WGS84 and their total bounds."""
    parcels = _read_parcels(parcel_path, parcel_layer)
    parcels_wgs84 = parcels.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = map(float, parcels_wgs84.total_bounds)
    if not (minx < maxx and miny < maxy):
        raise ValueError("Parcel extent is empty or degenerate")
    return parcels_wgs84, (minx, miny, maxx, maxy)


def _request_json(
    url: str,
    params: Dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Issue one HTTP GET request and decode a JSON/GeoJSON object."""
    query_url = f"{url}?{urlencode(params)}"
    request = Request(
        query_url,
        headers={"User-Agent": "basin-bmp-scenario-sim/0.1 (+USGS-WBD-input-helper)"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as exc:  # pragma: no cover - environment/network specific
        raise RuntimeError(
            f"Could not download USGS WBD HUC12 data from {url}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("USGS WBD service returned an unexpected non-object response")
    if "error" in payload:
        err = payload.get("error") or {}
        message = err.get("message") if isinstance(err, dict) else str(err)
        details = err.get("details") if isinstance(err, dict) else None
        suffix = f"; details={details}" if details else ""
        raise RuntimeError(f"USGS WBD service error: {message}{suffix}")
    return payload


def download_wbd_huc12(
    bounds_wgs84: Sequence[float],
    *,
    service_url: str = WBD_HUC12_QUERY_URL,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> gpd.GeoDataFrame:
    """Download WBD HUC12 polygons intersecting a WGS84 bounding box."""
    if len(bounds_wgs84) != 4:
        raise ValueError("bounds_wgs84 must be (minx, miny, maxx, maxy)")
    minx, miny, maxx, maxy = map(float, bounds_wgs84)
    if not (minx < maxx and miny < maxy):
        raise ValueError("WGS84 bounds are empty or degenerate")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    features: list[Dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "where": "1=1",
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join(WBD_FIELDS),
            "returnGeometry": "true",
            "outSR": 4326,
            "orderByFields": "objectid",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "geojson",
        }
        payload = _request_json(service_url, params, timeout=timeout)
        page = payload.get("features", [])
        if not isinstance(page, list):
            raise RuntimeError("USGS WBD GeoJSON response does not contain a feature list")
        features.extend(page)

        # ArcGIS may omit exceededTransferLimit from GeoJSON, so a full page
        # is sufficient reason to request the next page. An empty/short page
        # terminates pagination.
        if len(page) < page_size:
            break
        offset += len(page)

    if not features:
        raise ValueError("USGS WBD returned no HUC12 polygons for the parcel extent")

    hucs = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    hucs.columns = [str(column).strip().lower() for column in hucs.columns]
    if "huc12" not in hucs.columns:
        raise RuntimeError(
            f"USGS WBD response is missing HUC12; returned fields: {list(hucs.columns)}"
        )
    hucs["huc12"] = hucs["huc12"].map(_normalize_huc12)
    hucs = hucs.drop_duplicates(subset=["huc12"], keep="last").copy()
    if hucs.geometry.isna().any() or hucs.geometry.is_empty.any():
        raise ValueError("USGS WBD response contains null or empty HUC12 geometry")
    if (~hucs.geometry.is_valid).any():
        hucs["geometry"] = hucs.geometry.buffer(0)
    if (~hucs.geometry.is_valid).any():
        raise ValueError("USGS WBD response contains invalid HUC12 geometry")
    return hucs.reset_index(drop=True)


def retain_hucs_by_parcel_overlap_threshold(
    hucs: gpd.GeoDataFrame,
    parcels_wgs84: gpd.GeoDataFrame,
    *,
    threshold: float,
) -> gpd.GeoDataFrame:
    """Retain HUC12s containing >= ``threshold`` of at least one parcel.

    ``threshold`` uses the same parcel-area fraction definition as
    ``parcel_huc12.area_fraction``. Therefore a threshold of 0.75 keeps a
    HUC12 only when at least one parcel has 75% or more of its area inside
    that HUC12.
    """
    if hucs.crs is None or parcels_wgs84.crs is None:
        raise ValueError("Both HUC12 and parcel geometries must declare a CRS")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    hucs = hucs.to_crs(parcels_wgs84.crs)
    analysis_crs = parcels_wgs84.estimate_utm_crs()
    if analysis_crs is None:
        raise ValueError("Could not determine a projected CRS for overlap calculations")

    parcels_m = parcels_wgs84[["pid", "geometry"]].to_crs(analysis_crs)
    hucs_m = hucs[["huc12", "geometry"]].to_crs(analysis_crs)
    parcel_area = parcels_m.set_index("pid").geometry.area

    intersections = gpd.overlay(
        parcels_m,
        hucs_m,
        how="intersection",
        keep_geom_type=False,
    )
    if intersections.empty:
        raise ValueError("Downloaded HUC12 polygons do not intersect the parcels")
    intersections["intersection_area"] = intersections.geometry.area
    intersections = intersections[intersections["intersection_area"] > 0].copy()
    intersections["area_fraction"] = intersections.apply(
        lambda row: float(
            row["intersection_area"] / parcel_area.loc[int(row["pid"])]
        ),
        axis=1,
    ).clip(lower=0.0, upper=1.0)

    qualifying = intersections.loc[
        intersections["area_fraction"] >= threshold, "huc12"
    ].astype(str)
    keep_hucs = set(qualifying)
    if not keep_hucs:
        raise ValueError(
            "No HUC12 contains a parcel meeting the area-fraction threshold "
            f"({threshold:.2f})"
        )

    out = hucs[hucs["huc12"].isin(keep_hucs)].copy().reset_index(drop=True)

    # Ensure every parcel still intersects at least one retained HUC. If not,
    # assignment would be incomplete and the forcing relationship invalid.
    covered = set(
        intersections.loc[
            intersections["huc12"].isin(keep_hucs), "pid"
        ].astype(int)
    )
    missing = sorted(set(parcels_m["pid"].astype(int)) - covered)
    if missing:
        raise ValueError(
            "The HUC12 overlap threshold removes every HUC12 intersecting some "
            "parcels; lower --area-fraction-threshold or inspect parcel/HUC geometry. "
            f"Missing pid values: {missing[:20]}"
        )
    return out

def _normalize_initial_export(table: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Normalize the HUC12-keyed values extracted from an EPA PLET export."""
    table = table.copy()
    table.columns = [str(column).strip().lower() for column in table.columns]
    if "huc12" not in table.columns:
        raise ValueError(f"{label} must contain a huc12 column")
    table["huc12"] = table["huc12"].map(_normalize_huc12)
    if table["huc12"].duplicated().any():
        duplicated = table.loc[table["huc12"].duplicated(), "huc12"].tolist()
        raise ValueError(f"Duplicate HUC12 rows in {label}: {duplicated[:10]}")
    for field in PLET_FORCING_FIELDS:
        if field not in table.columns:
            continue
        raw_nonnull = table[field].notna()
        converted = pd.to_numeric(table[field], errors="coerce")
        invalid = raw_nonnull & converted.isna()
        if invalid.any():
            bad = table.loc[invalid, ["huc12", field]].head(10).to_dict(orient="records")
            raise ValueError(f"{label}.{field} must be numeric or blank; invalid rows: {bad}")
        table[field] = converted
    return table


def read_initial_plet_export(path: Path) -> pd.DataFrame:
    """Extract HUC12 climate forcing from an EPA PLET Excel export."""
    try:
        table = pd.read_excel(path, sheet_name="1. Watershed Land Use", engine="calamine")
    except Exception as exc:
        raise ValueError(f"Could not read PLET export {path}: {exc}") from exc

    table = table.copy()
    table.columns = [str(column).strip().upper() for column in table.columns]
    required = {"WATERSHED", "AVG_RAIN", "RAIN_DAYS", "ANNUAL_RAINFALL"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            f"PLET export sheet '1. Watershed Land Use' is missing columns: {missing}"
        )

    watershed = table["WATERSHED"].astype("string")
    huc12 = watershed.str.extract(r"^\s*(\d{12})(?:\s*-.*)?$", expand=False)
    invalid = huc12.isna()
    if invalid.any():
        bad = watershed.loc[invalid].head(10).tolist()
        raise ValueError(
            "PLET export WATERSHED values must begin with a 12-digit HUC12; "
            f"invalid examples: {bad}"
        )

    out = pd.DataFrame(
        {
            "huc12": huc12.map(_normalize_huc12),
            "avg_rain_in": table["AVG_RAIN"],
            "rain_days": table["RAIN_DAYS"],
            "annual_precip_in": table["ANNUAL_RAINFALL"],
        }
    )
    return _normalize_initial_export(out, label=str(path))


def _parameter_keys(parameter: str) -> list[str]:
    return ["huc12", *(["land_cover"] if parameter in LAND_COVER_PARAMETERS else []),
            *(["pollutant"] if parameter in CONCENTRATION_PARAMETERS else [])]


def _empty_parameter_table(parameter: str = "avg_rain_in") -> pd.DataFrame:
    columns = [*_parameter_keys(parameter), *STAT_COLUMNS, "sample_group", "units", "notes"]
    return pd.DataFrame(columns=columns)


def parameter_tables_from_initial_export(
    export: pd.DataFrame,
    *,
    source_label: str,
) -> Dict[str, pd.DataFrame]:
    """Convert a wide PLET export to one fixed-value table per parameter."""
    out: Dict[str, pd.DataFrame] = {
        parameter: _empty_parameter_table(parameter) for parameter in PLET_PARAMETER_TABLES
    }
    template_columns = list(_empty_parameter_table().columns)
    for parameter in ("avg_rain_in", "annual_precip_in", "rain_days"):
        if parameter not in export.columns:
            continue
        rows = export[["huc12", parameter]].dropna(subset=[parameter]).copy()
        rows = rows.rename(columns={parameter: "value"})
        rows["units"] = PLET_PARAMETER_UNITS[parameter]
        rows["notes"] = f"Initialized from EPA PLET Input Data Server export: {source_label}"
        for column in template_columns:
            if column not in rows.columns:
                rows[column] = None
        out[parameter] = rows[template_columns].reset_index(drop=True)
    return out


def read_existing_parameter_tables(package: Path) -> Dict[str, pd.DataFrame]:
    """Read existing HUC12 ``input_*`` tables for preservation on WBD refresh."""
    out: Dict[str, pd.DataFrame] = {}
    if not package.exists():
        return out
    # Do not use sqlite3.Connection as a context manager here.  That context
    # manager commits/rolls back but does not close the connection, which can
    # leave the existing GeoPackage locked on Windows when a refresh later
    # replaces the file.  Read the registered tables through SQLite itself and
    # close the connection explicitly before returning.
    con = sqlite3.connect(package)
    try:
        rows = con.execute(
            "SELECT table_name FROM gpkg_contents"
        ).fetchall()
        layers = {str(row[0]) for row in rows}
        for parameter, table_name in PLET_PARAMETER_TABLES.items():
            if table_name not in layers:
                continue
            table = pd.read_sql_query(f'SELECT * FROM "{table_name}" ORDER BY id', con)
            table = table.drop(columns=[c for c in ("id", "fid", "row_id") if c in table.columns])
            if "huc12" not in table.columns:
                raise ValueError(f"{package}:{table_name} must contain huc12")
            table["huc12"] = table["huc12"].map(_normalize_huc12)
            keys = _parameter_keys(parameter)
            if not set(keys).issubset(table):
                raise ValueError(f"{package}:{table_name} must contain {keys}")
            if table.duplicated(subset=keys).any():
                raise ValueError(f"Duplicate {keys} rows in {package}:{table_name}")
            out[parameter] = table
    except sqlite3.DatabaseError:
        return {}
    finally:
        con.close()
    return out


def _normalize_parameter_table_for_write(
    table: Optional[pd.DataFrame],
    *,
    valid_hucs: set[str],
    parameter: str = "avg_rain_in",
) -> pd.DataFrame:
    """Normalize one input table to the standard HUC12 distribution schema."""
    base = _empty_parameter_table(parameter)
    if table is None or table.empty:
        return base
    out = table.copy()
    out.columns = [str(column).strip().lower() for column in out.columns]
    if "huc12" not in out.columns:
        raise ValueError("HUC12 parameter table must contain huc12")
    out["huc12"] = out["huc12"].map(_normalize_huc12)
    out = out[out["huc12"].isin(valid_hucs)].copy()
    keys = _parameter_keys(parameter)
    missing = set(keys) - set(out)
    if missing:
        raise ValueError(f"{parameter} missing key columns: {sorted(missing)}")
    if out.duplicated(subset=keys).any():
        raise ValueError(f"Duplicate {parameter} rows for keys {keys}")
    for column in base.columns:
        if column not in out.columns:
            out[column] = None
    return out[list(base.columns)].reset_index(drop=True)


def _create_parameter_table(
    con: sqlite3.Connection,
    table_name: str,
    rows: pd.DataFrame,
    parameter: str = "avg_rain_in",
) -> None:
    """Create/register one QGIS-editable HUC12 input table and insert rows."""
    stat_defs = ",\n                ".join(
        f'"{column}" {"TEXT" if parameter == "hsg" and column == "value" else "REAL"}'
        for column in STAT_COLUMNS
    )
    other_keys = (
        ('                land_cover TEXT NOT NULL,\n' if parameter in LAND_COVER_PARAMETERS else '')
        + ('                pollutant TEXT NOT NULL,\n' if parameter in CONCENTRATION_PARAMETERS else '')
    )
    keys = ", ".join(_parameter_keys(parameter))
    sql = (
        f'CREATE TABLE "{table_name}" (\n'
        '                id INTEGER PRIMARY KEY AUTOINCREMENT,\n'
        '                huc12 TEXT NOT NULL CHECK(length(huc12)=12),\n'
        f'{other_keys}'
        f'                {stat_defs},\n'
        '                sample_group TEXT,\n'
        '                units TEXT,\n'
        '                notes TEXT,\n'
        f'                UNIQUE ({keys})\n'
        '            )'
    )
    con.execute(sql)
    con.execute(
        "INSERT INTO gpkg_contents(table_name, data_type, identifier, description, srs_id) "
        "VALUES (?, 'attributes', ?, ?, NULL)",
        (table_name, table_name, "HUC12-scale PLET input; fixed value or distribution"),
    )
    if rows.empty:
        return
    columns = list(_empty_parameter_table(parameter).columns)
    placeholders = ",".join("?" for _ in columns)
    quoted = ",".join(f'"{column}"' for column in columns)
    values = []
    for row in rows.itertuples(index=False, name=None):
        values.append(tuple(None if pd.isna(value) else value for value in row))
    con.executemany(
        f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})', values
    )


def write_huc12_package(
    package: Path,
    hucs: gpd.GeoDataFrame,
    parameter_tables: Dict[str, pd.DataFrame],
    *,
    layer: str = HUC12_LAYER,
) -> None:
    """Write WBD geometry plus HUC12 parameter tables as one GeoPackage."""
    package.parent.mkdir(parents=True, exist_ok=True)
    temp = package.with_name(f".{package.stem}.tmp.gpkg")
    if temp.exists():
        temp.unlink()

    structural = hucs.drop(
        columns=[c for c in PLET_FORCING_FIELDS if c in hucs.columns],
        errors="ignore",
    )
    structural.to_file(temp, layer=layer, driver="GPKG")
    valid_hucs = set(structural["huc12"].astype(str))

    # sqlite3.Connection's context manager commits/rolls back transactions but
    # does not close the connection.  Keeping ``con`` alive through the
    # subsequent Path.replace() leaves the GeoPackage locked on Windows and
    # causes WinError 32.  Close it explicitly before replacing the target.
    con = sqlite3.connect(temp)
    try:
        for parameter, table_name in PLET_PARAMETER_TABLES.items():
            rows = _normalize_parameter_table_for_write(
                parameter_tables.get(parameter), valid_hucs=valid_hucs, parameter=parameter
            )
            _create_parameter_table(con, table_name, rows, parameter)
        con.commit()
    finally:
        con.close()

    temp.replace(package)


def prepare_huc12_forcing(
    parcel_path: Path,
    plet_forcing_path: Path,
    *,
    area_fraction_threshold: float = DEFAULT_AREA_FRACTION_THRESHOLD,
    initial_plet_export: Optional[Path] = None,
    parcel_layer: str = PARCEL_LAYER,
    huc12_layer: str = HUC12_LAYER,
    service_url: str = WBD_HUC12_QUERY_URL,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Download/store HUC12 geometry, preserve inputs, and assign parcels."""
    parcels_wgs84, bounds = parcel_footprint_wgs84(parcel_path, parcel_layer=parcel_layer)

    preserved = read_existing_parameter_tables(plet_forcing_path)
    has_existing_values = any(not table.empty for table in preserved.values())
    if has_existing_values and initial_plet_export is not None:
        raise ValueError(
            "--initial-plet-export is only allowed before HUC12 input tables contain values; "
            "edit the existing input_* tables directly instead"
        )
    if initial_plet_export is not None:
        initial = read_initial_plet_export(initial_plet_export)
        parameter_tables = parameter_tables_from_initial_export(
            initial, source_label=initial_plet_export.name
        )
    else:
        parameter_tables = preserved

    hucs = download_wbd_huc12(
        bounds,
        service_url=service_url,
        page_size=page_size,
        timeout=timeout,
    )
    hucs = retain_hucs_by_parcel_overlap_threshold(
        hucs, parcels_wgs84, threshold=area_fraction_threshold
    )
    write_huc12_package(
        plet_forcing_path, hucs, parameter_tables, layer=huc12_layer
    )

    assignments = assign_parcels_to_huc12(
        parcel_path,
        plet_forcing_path,
        parcel_layer=parcel_layer,
        huc_layer=huc12_layer,
        huc12_field="huc12",
    )
    write_parcel_huc12(parcel_path, assignments)
    return hucs, assignments

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parcels", type=Path, help="Parcel GeoPackage to update")
    parser.add_argument(
        "plet_forcing",
        type=Path,
        help="Dedicated PLET forcing GeoPackage to create/refresh",
    )
    parser.add_argument(
        "--initial-plet-export",
        type=Path,
        default=None,
        help=(
            "Initialization only: EPA PLET Excel export used to populate "
            "fixed AVG_RAIN, RAIN_DAYS, and ANNUAL_RAINFALL rows in HUC12 input_* tables"
        ),
    )
    parser.add_argument("--parcel-layer", default=PARCEL_LAYER)
    parser.add_argument("--huc12-layer", default=HUC12_LAYER)
    parser.add_argument(
        "--service-url",
        default=WBD_HUC12_QUERY_URL,
        help="USGS ArcGIS REST HUC12 query endpoint",
    )
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--area-fraction-threshold",
        type=float,
        default=DEFAULT_AREA_FRACTION_THRESHOLD,
        help="Retain HUC12s with at least one parcel meeting this area fraction and warn on dominant assignments below it (default: 0.75)",
    )
    args = parser.parse_args()

    if not args.parcels.exists():
        parser.error(f"Parcel GeoPackage not found: {args.parcels}")
    if args.initial_plet_export is not None and not args.initial_plet_export.exists():
        parser.error(f"PLET export not found: {args.initial_plet_export}")
    if args.page_size <= 0:
        parser.error("--page-size must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not 0 <= args.area_fraction_threshold <= 1:
        parser.error("--area-fraction-threshold must be between 0 and 1")

    hucs, assignments = prepare_huc12_forcing(
        args.parcels,
        args.plet_forcing,
        parcel_layer=args.parcel_layer,
        huc12_layer=args.huc12_layer,
        service_url=args.service_url,
        page_size=args.page_size,
        timeout=args.timeout,
        area_fraction_threshold=args.area_fraction_threshold,
        initial_plet_export=args.initial_plet_export,
    )

    low = assignments[assignments["area_fraction"] < args.area_fraction_threshold]
    print(
        f"Stored {len(hucs):,} threshold-qualified HUC12 polygon(s) in "
        f"{args.plet_forcing}:{args.huc12_layer}"
    )
    tables = read_existing_parameter_tables(args.plet_forcing)
    counts = {PLET_PARAMETER_TABLES[p]: len(tables.get(p, ())) for p in PLET_PARAMETER_TABLES}
    print("HUC12 PLET input rows: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    print(f"Wrote {len(assignments):,} parcel assignments to {args.parcels}:parcel_huc12")
    if not low.empty:
        print(
            f"Warning: {len(low):,} parcel(s) have dominant-HUC area_fraction "
            f"below {args.area_fraction_threshold:.2f}."
        )
        print(low.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
