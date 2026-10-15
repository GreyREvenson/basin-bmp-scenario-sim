"""Build independent parcel and HUC12 PLET GeoPackages from an EPA export.

Usage: python utils/create_plet_model_inputs.py parcels.gpkg parcels export.xlsx

The parcel output is a SQLite backup of the input, so unrelated input tables,
parcel overrides, feature IDs and GeoPackage metadata are preserved. The two
output files must not already exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
import requests
from rasterio.io import MemoryFile
from shapely.geometry import box
from shapely.prepared import prep

try:
    from utils.create_parcel_huc12 import assign_parcels_to_huc12, write_parcel_huc12, _read_parcels
    from utils.download_wbd_huc12 import (
        download_wbd_huc12,
        parameter_tables_from_initial_export,
        parcel_footprint_wgs84,
        read_initial_plet_export,
        retain_hucs_by_parcel_overlap_threshold,
        write_huc12_package,
    )
    from utils.download_ssurgo_hsg import (
        _copy_sqlite_database,
        _parcel_hsg_values,
        clip_to_domain,
        download_ssurgo,
        normalize_wfs_axis_order,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from create_parcel_huc12 import assign_parcels_to_huc12, write_parcel_huc12, _read_parcels
    from download_wbd_huc12 import (
        download_wbd_huc12,
        parameter_tables_from_initial_export,
        parcel_footprint_wgs84,
        read_initial_plet_export,
        retain_hucs_by_parcel_overlap_threshold,
        write_huc12_package,
    )
    from download_ssurgo_hsg import (
        _copy_sqlite_database,
        _parcel_hsg_values,
        clip_to_domain,
        download_ssurgo,
        normalize_wfs_axis_order,
    )

LOGGER = logging.getLogger(__name__)
LAND_COVERS = ("urban", "cropland", "pastureland", "forest", "user_defined")
HSG_VALUES = ("A", "B", "C", "D")
RUSLE_COLUMNS = {"cropland": "CROP", "pastureland": "PAST", "forest": "FOREST",
                 "user_defined": "USER_DEFINED"}
NLCD_CLASSES = {
    21: "urban", 22: "urban", 23: "urban", 24: "urban",
    31: "pastureland",
    41: "forest", 42: "forest", 43: "forest",
    71: "pastureland", 81: "pastureland", 82: "cropland",
    90: "forest",
}
ASSUMED_NLCD_CLASSES = {31: "barren", 90: "woody wetlands"}
STATS = ("value", "mean", "sd", "min", "p05", "p10", "p25", "p50",
         "p75", "p90", "p95", "max")
NLCD_SERVICE_URL = (
    "https://di-nlcd.img.arcgis.com/arcgis/rest/services/"
    "USA_NLCD_Annual_LandCover/ImageServer"
)


@dataclass(frozen=True)
class NLCDImage:
    data: bytes
    year: int
    source: str


def _sheet(workbook: pd.ExcelFile, name: str, columns: set[str]) -> pd.DataFrame:
    table = workbook.parse(sheet_name=name)
    table.columns = [str(column).strip().upper() for column in table.columns]
    missing = columns - set(table)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    return table


def _numeric(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _cover(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    if text not in LAND_COVERS:
        raise ValueError(f"Unmapped PLET land cover {value!r}; expected {LAND_COVERS}")
    return text


def _huc_sheet(workbook: pd.ExcelFile, name: str, fields: set[str]) -> pd.DataFrame:
    table = _sheet(workbook, name, {"WATERSHED", *fields})
    hucs = table["WATERSHED"].astype("string").str.extract(
        r"^\s*(\d{12})(?:\s*-.*)?$", expand=False
    )
    if hucs.isna().any() or hucs.duplicated().any():
        raise ValueError(f"{name} has missing/invalid or duplicate WATERSHED HUC12 values")
    table["huc12"] = hucs
    return table.set_index("huc12")


def _rows(records: list[dict], label: str) -> pd.DataFrame:
    if not records:
        raise ValueError(f"No usable {label} values in PLET workbook")
    return pd.DataFrame.from_records(records)


def workbook_tables(
    path: Path, hucs: set[str], *, export: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Extract numeric PLET inputs; intentionally ignore exported land-use areas."""
    if export is None:
        export = read_initial_plet_export(path)
    missing = hucs - set(export["huc12"])
    if missing:
        raise ValueError(f"PLET workbook has no climate values for HUC12s: {sorted(missing)}")
    export = export.loc[export["huc12"].isin(hucs)].copy()
    tables = parameter_tables_from_initial_export(export, source_label=path.name)
    with pd.ExcelFile(path, engine="calamine") as workbook:
        watershed = _huc_sheet(workbook, "1. Watershed Land Use", {"SHG"})
        soil = _huc_sheet(
            workbook, "5. Universal Soil Loss Equation",
            {f"{prefix}_{parameter.upper()}" for prefix in RUSLE_COLUMNS.values()
             for parameter in ("r", "k", "ls", "c", "p")},
        )
        sediment_nutrients = _huc_sheet(
            workbook, "4. Nutrient and E.Coli Content",
            {"SOIL_N_CONC", "SOIL_P_CONC"},
        )
        nutrients = _sheet(workbook, "7a. Nutrient Concentration", {"LANDUSE", "N", "P"})
        hydro_sheets = {
            name: _sheet(workbook, name, {"SHG", *HSG_VALUES})
            for name in ("6. Reference Runoff Curve", "Soil Infiltration")
        }
    if hucs - set(soil.index):
        raise ValueError(f"PLET USLE sheet missing HUC12s: {sorted(hucs - set(soil.index))}")
    if hucs - set(sediment_nutrients.index):
        raise ValueError(
            f"PLET sheet 4 missing HUC12s: {sorted(hucs - set(sediment_nutrients.index))}"
        )

    for parameter, field in (("sediment_n_pct", "SOIL_N_CONC"),
                             ("sediment_p_pct", "SOIL_P_CONC")):
        records = []
        for huc in sorted(hucs):
            value = _numeric(sediment_nutrients.loc[huc, field], f"{huc} {field}")
            if not 0 <= value <= 100:
                raise ValueError(f"{huc} {field} must be a percent between 0 and 100")
            records.append({
                "huc12": huc, "value": value, "units": "%",
                "notes": f"PLET sheet 4, {field}; soil nutrient concentration in percent",
            })
        tables[parameter] = _rows(records, parameter)

    for parameter in ("r", "k", "ls", "c", "p"):
        records = []
        for huc in sorted(hucs):
            for cover, prefix in RUSLE_COLUMNS.items():
                value = _numeric(soil.loc[huc, f"{prefix}_{parameter.upper()}"],
                                 f"{huc} {prefix}_{parameter.upper()}")
                records.append({"huc12": huc, "land_cover": cover, "value": value,
                                "notes": f"PLET sheet 5, {prefix}_{parameter.upper()}"})
        tables[parameter] = _rows(records, parameter)

    hsg_rows = []
    for huc in sorted(hucs):
        hsg = str(watershed.loc[huc, "SHG"]).strip().upper()
        if hsg not in HSG_VALUES:
            raise ValueError(f"Unsupported PLET SHG {hsg!r} in HUC12 {huc}")
        hsg_rows.append({"huc12": huc, "value": hsg, "notes": "PLET sheet 1, SHG"})
    tables["hsg"] = _rows(hsg_rows, "hsg")

    nutrients["land_cover"] = nutrients["LANDUSE"].map(
        lambda v: None if str(v).strip().lower() == "feedlots" else _cover(v)
    )
    nutrients = nutrients.dropna(subset=["land_cover"])
    if nutrients["land_cover"].duplicated().any() or set(nutrients["land_cover"]) != set(LAND_COVERS):
        raise ValueError("PLET nutrient sheet must have one row per PLET land cover")
    for kind in ("surface", "subsurface"):
        records = []
        for huc in sorted(hucs):
            for row in nutrients.itertuples(index=False):
                for source, pollutant in (("N", "TN"), ("P", "TP")):
                    value = _numeric(getattr(row, source), f"{row.land_cover} {source}")
                    records.append({
                        "huc12": huc, "land_cover": row.land_cover, "pollutant": pollutant,
                        "value": value if kind == "surface" else 0.0,
                        "units": "mg/L",
                        "notes": "PLET sheet 7a" if kind == "surface" else
                                 "Zero subsurface concentration; PLET sheet 7a supplies surface only",
                    })
            if kind == "surface":
                for cover in LAND_COVERS:
                    records.append({
                        "huc12": huc, "land_cover": cover, "pollutant": "TSS",
                        "value": 0.0, "units": "mg/L",
                        "notes": "PLACEHOLDER: PLET workbook has no TSS concentration; "
                                 "review and replace or override at parcel scale",
                    })
        tables[f"{kind}_concentration"] = _rows(records, f"{kind} concentrations")
    LOGGER.warning(
        "TSS PLACEHOLDER: PLET workbook has no TSS concentration; generated zero "
        "surface TSS concentrations for %d HUC12s x %d land covers. Review and "
        "replace these values or provide parcel TSS overrides before modeling. "
        "Workbook also has no urban RUSLE factors; review urban TSS/RUSLE inputs.",
        len(hucs), len(LAND_COVERS),
    )

    hydrology = {}
    for name, parameter in (("6. Reference Runoff Curve", "cn"),
                            ("Soil Infiltration", "infiltration_fraction")):
        sheet = hydro_sheets[name]
        sheet["land_cover"] = sheet["SHG"].map(_cover)
        if sheet["land_cover"].duplicated().any() or set(sheet["land_cover"]) != set(LAND_COVERS):
            raise ValueError(f"{name} must have exactly one row per PLET land cover")
        records = []
        for row in sheet.itertuples(index=False):
            for hsg in HSG_VALUES:
                value = _numeric(getattr(row, hsg), f"{name} {row.land_cover}/{hsg}")
                if parameter == "cn" and value == 0:
                    LOGGER.warning("Omitting undefined zero PLET curve number for %s/%s; "
                                   "supply a positive curve number before modeling this class",
                                   row.land_cover, hsg)
                    continue
                if parameter == "cn" and not 0 < value <= 100:
                    raise ValueError(f"{name} {row.land_cover}/{hsg} curve number must be in (0, 100]")
                if parameter == "infiltration_fraction" and not 0 <= value <= 1:
                    raise ValueError(f"{name} {row.land_cover}/{hsg} infiltration must be in [0, 1]")
                records.append({"land_cover": row.land_cover, "hsg": hsg,
                                "value": value,
                                "notes": f"PLET {name}"})
        hydrology[parameter] = _rows(records, parameter)
    return tables, hydrology


def reconstruct_rain_factors(
    export: pd.DataFrame, hucs: gpd.GeoDataFrame, *, tolerance: float = 1e-7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimate grid factors from rainfall and nearby uniquely solved HUCs."""
    required = {"huc12", "avg_rain_in", "rain_days", "annual_precip_in"}
    if not required <= set(export):
        raise ValueError(f"Climate export missing {sorted(required - set(export))}")
    candidates: dict[str, list[tuple[float, float, float]]] = {}
    observations = {}
    for row in export.itertuples(index=False):
        huc = str(row.huc12)
        avg = _numeric(row.avg_rain_in, f"{huc} AVG_RAIN")
        rain_days = _numeric(row.rain_days, f"{huc} RAIN_DAYS")
        annual = _numeric(row.annual_precip_in, f"{huc} ANNUAL_RAINFALL")
        if min(avg, rain_days, annual) <= 0:
            raise ValueError(f"{huc} climate values must be positive")
        matches = []
        for r in range(840, 901):
            for rd in range(350, 501):
                predicted = annual * r / (rain_days * rd)
                error = abs(predicted - avg) / avg
                if error <= tolerance:
                    matches.append((r / 1000, rd / 1000, error))
        if not matches:
            raise ValueError(f"{huc} has no Rcor/RDcor grid solution within {tolerance:g} "
                             f"relative error; AVG_RAIN={avg}")
        candidates[huc] = matches
        observations[huc] = (avg, rain_days, annual)

    geometry = hucs.set_index("huc12")
    if not set(candidates) <= set(geometry.index):
        raise ValueError(f"Missing WBD geometry for HUC12s: {sorted(set(candidates) - set(geometry.index))}")
    projected = geometry.to_crs(geometry.estimate_utm_crs())
    unique = {huc: options[0] for huc, options in candidates.items() if len(options) == 1}
    result = {"rain_correction_fraction": [], "runoff_day_fraction": []}
    audit = []
    for huc, options in candidates.items():
        status = "unique grid solution; reconstructed estimate, not independently verified"
        if len(options) == 1:
            choice = options[0]
        else:
            if not unique:
                raise ValueError("Ambiguous rainfall factor reconstruction: no uniquely "
                                 "reconstructed HUC12 exists to resolve neighboring HUCs")
            # Adjacent polygons have zero boundary distance, so use their
            # centroids to rank genuinely nearby independent reconstructions.
            centroid = projected.loc[huc].geometry.centroid
            nearest = sorted(
                unique,
                key=lambda other: (
                    centroid.distance(projected.loc[other].geometry.centroid), other),
            )[:4]
            distances = [
                centroid.distance(projected.loc[other].geometry.centroid)
                for other in nearest
            ]
            if any(distance <= 0 for distance in distances):
                raise ValueError(f"{huc} has coincident centroids with a unique HUC12; "
                                 "cannot distance-weight rainfall factor estimates")
            total_weight = sum(1 / distance for distance in distances)
            target_r = sum(unique[other][0] / distance for other, distance in zip(nearest, distances)) / total_weight
            target_rd = sum(unique[other][1] / distance for other, distance in zip(nearest, distances)) / total_weight
            choice = min(options, key=lambda option: (
                (option[0] - target_r) ** 2 + (option[1] - target_rd) ** 2,
                option[2], option[0], option[1]))
            status = (f"ambiguous; estimated from nearest unique HUC12s {', '.join(nearest)}; "
                      f"{len(options)} grid solutions; not independently verified")
        avg, days, annual = observations[huc]
        predicted = annual * choice[0] / (days * choice[1])
        absolute_error = abs(predicted - avg)
        audit.append({"huc12": huc, "annual_rainfall": annual, "rain_days": days,
                      "exported_avg_rain": avg, "predicted_avg_rain": predicted,
                      "absolute_error": absolute_error, "relative_error": absolute_error / avg, "rcor": choice[0],
                      "rdcor": choice[1], "status": status})
        for parameter, value in (("rain_correction_fraction", choice[0]),
                                 ("runoff_day_fraction", choice[1])):
            result[parameter].append({"huc12": huc, "value": value,
                                      "units": "fraction", "notes": status})
        LOGGER.info("Rain audit HUC12 %s annual_rainfall=%.15g rain_days=%.15g "
                    "exported_avg_rain=%.15g reconstructed_avg_rain=%.15g "
                    "absolute_error=%.3g relative_error=%.3g "
                    "Rcor=%.3f RDcor=%.3f status=%s",
                    huc, annual, days, avg, predicted, absolute_error,
                    absolute_error / avg, *choice[:2], status)
    return _rows(result["rain_correction_fraction"], "Rcor"), _rows(
        result["runoff_day_fraction"], "RDcor"), pd.DataFrame(audit)


def _arcgis_result(response: requests.Response, operation: str) -> dict:
    response.raise_for_status()
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(f"NLCD {operation} did not return JSON") from exc
    if not isinstance(payload, dict) or "error" in payload:
        raise RuntimeError(f"NLCD {operation} failed: {payload}")
    return payload


def download_nlcd(bounds: tuple[float, float, float, float]) -> NLCDImage:
    """Fetch the newest published annual NLCD raster from ArcGIS ImageServer."""
    minx, miny, maxx, maxy = bounds
    if not all(math.isfinite(v) for v in bounds) or minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid EPSG:5070 NLCD bounds: {bounds}")
    query = _arcgis_result(requests.get(
        f"{NLCD_SERVICE_URL}/query",
        params={"f": "json", "where": "1=1", "outFields": "Year",
                "returnDistinctValues": "true", "returnGeometry": "false",
                "orderByFields": "Year DESC"}, timeout=60,
    ), "year query")
    features = query.get("features")
    if not isinstance(features, list) or not features:
        raise RuntimeError(f"NLCD year query has no features: {query}")
    years = []
    for feature in features:
        if not isinstance(feature, dict) or not isinstance(feature.get("attributes"), dict):
            raise RuntimeError(f"NLCD year query has malformed feature: {feature}")
        year = feature["attributes"].get("Year")
        if type(year) is not int or not 1985 <= year <= 2100:
            raise RuntimeError(f"NLCD year query has invalid Year: {year!r}")
        years.append(year)
    year = max(years)
    width = max(1, math.ceil((maxx - minx) / 30))
    height = max(1, math.ceil((maxy - miny) / 30))
    mosaic_rule = {"mosaicMethod": "esriMosaicAttribute", "where": f"Year = {year}",
                   "sortField": "Year", "sortValue": year}
    export = _arcgis_result(requests.get(
        f"{NLCD_SERVICE_URL}/exportImage",
        params={"f": "json", "bbox": ",".join(map(str, bounds)), "bboxSR": 5070,
                "imageSR": 5070, "size": f"{width},{height}", "format": "tiff",
                "pixelType": "U8", "interpolation": "RSP_NearestNeighbor",
                "mosaicRule": json.dumps(mosaic_rule)}, timeout=180,
    ), "image export")
    href = export.get("href")
    parsed = urlparse(href) if isinstance(href, str) else None
    if parsed is None or parsed.scheme != "https" or parsed.hostname != urlparse(NLCD_SERVICE_URL).hostname:
        raise RuntimeError(f"NLCD image export returned an unexpected href: {href!r}")
    response = requests.get(href, timeout=180)
    response.raise_for_status()
    if not response.content.startswith((b"II*\x00", b"MM\x00*")):
        raise RuntimeError(f"NLCD {year} image export did not return a GeoTIFF")
    LOGGER.info("NLCD source=%s year=%d", NLCD_SERVICE_URL, year)
    return NLCDImage(response.content, year, NLCD_SERVICE_URL)


def parcel_nlcd_land_cover(
    parcels: gpd.GeoDataFrame, *,
    source: Callable[[tuple[float, float, float, float]], NLCDImage] = download_nlcd,
    verbose: bool = False,
) -> pd.DataFrame:
    """Classify parcels by area-weighted NLCD pixel overlap, including tiny parcels."""
    projected = parcels.to_crs("EPSG:5070")
    image = source(tuple(map(float, projected.total_bounds)))
    if (not isinstance(image, NLCDImage) or not isinstance(image.data, bytes)
            or type(image.year) is not int or not 1985 <= image.year <= 2100
            or not image.source):
        raise ValueError("NLCD source must return NLCDImage(data, year, source)")
    LOGGER.info("Classifying parcels from NLCD %d (%s)", image.year, image.source)
    with MemoryFile(image.data) as memory:
        with memory.open() as raster:
            if raster.crs is None or raster.count != 1 or raster.dtypes[0] != "uint8":
                raise ValueError("NLCD image must be a georeferenced single-band U8 raster")
            affine = raster.transform
            if affine.b != 0 or affine.d != 0 or affine.a <= 0 or affine.e >= 0:
                raise ValueError("NLCD coverage must be a north-up raster")
            shapes = projected.to_crs(raster.crs)
            pixels = raster.read([1])[0]
            classified = []
            assumed_parcels: dict[int, list[int]] = {code: [] for code in ASSUMED_NLCD_CLASSES}
            unmapped_parcels: dict[int, list[int]] = {}
            for row in shapes.itertuples():
                bounds = row.geometry.bounds
                if row.geometry.is_empty or not row.geometry.is_valid or row.geometry.area <= 0:
                    raise ValueError(f"Parcel {row.pid} must be a valid nonempty polygon")
                left = max(0, math.floor((bounds[0] - affine.c) / affine.a))
                right = min(raster.width, math.ceil((bounds[2] - affine.c) / affine.a))
                top = max(0, math.floor((bounds[3] - affine.f) / affine.e))
                bottom = min(raster.height, math.ceil((bounds[1] - affine.f) / affine.e))
                if left >= right or top >= bottom:
                    raise ValueError(f"Parcel {row.pid} falls outside NLCD coverage")
                areas: dict[int, float] = {}
                valid_area = 0.0
                touched = 0
                prepared = prep(row.geometry)
                for y in range(top, bottom):
                    y_upper = affine.f + y * affine.e
                    for x in range(left, right):
                        x_left = affine.c + x * affine.a
                        cell = box(x_left, y_upper + affine.e, x_left + affine.a, y_upper)
                        if not prepared.intersects(cell):
                            continue
                        area = row.geometry.intersection(cell).area
                        if area <= 0:
                            continue
                        code = int(pixels[y, x])
                        if raster.nodata is not None and code == raster.nodata:
                            continue
                        if raster.nodata is None and code == 0:
                            raise ValueError(
                                f"NLCD code 0 intersects parcel {row.pid} but raster "
                                "does not declare a nodata value"
                            )
                        valid_area += area
                        areas[code] = areas.get(code, 0.0) + area
                        touched += 1
                if valid_area <= 0:
                    raise ValueError(f"No valid NLCD pixel area inside parcel {row.pid}; "
                                     "check raster coverage and nodata")
                coverage = min(1.0, valid_area / row.geometry.area)
                if coverage < 0.999:
                    LOGGER.warning("NLCD parcel pid=%d has only %.2f%% valid raster coverage",
                                   row.pid, 100 * coverage)
                dominant = min(areas, key=lambda code: (-areas[code], code))
                if dominant not in NLCD_CLASSES:
                    unmapped_parcels.setdefault(dominant, []).append(int(row.pid))
                    continue
                if dominant in assumed_parcels:
                    assumed_parcels[dominant].append(int(row.pid))
                if verbose:
                    details = ", ".join(f"{code}={100 * area / valid_area:.2f}%"
                                        for code, area in sorted(areas.items()))
                    categories: dict[str, float] = {}
                    for code, area in areas.items():
                        category = NLCD_CLASSES.get(code, f"unmapped:{code}")
                        categories[category] = categories.get(category, 0.0) + area
                    category_details = ", ".join(
                        f"{category}={100 * area / valid_area:.2f}%"
                        for category, area in sorted(categories.items())
                    )
                    LOGGER.info("NLCD parcel pid=%d year=%d dominant=%d land_cover=%s "
                                "intersected_pixels=%d valid_coverage=%.2f%% "
                                "code_percentages=[%s] category_percentages=[%s]",
                                row.pid, image.year, dominant, NLCD_CLASSES[dominant],
                                touched, 100 * coverage, details, category_details)
                note = f"NLCD {image.year} dominant class {dominant}; {image.source}"
                if dominant in assumed_parcels:
                    note += (f"; ASSUMPTION: {ASSUMED_NLCD_CLASSES[dominant]} ({dominant}) "
                             f"treated as PLET {NLCD_CLASSES[dominant]}")
                classified.append({"pid": int(row.pid), "value": NLCD_CLASSES[dominant],
                                   "notes": note})
    for code, pids in assumed_parcels.items():
        if not pids:
            continue
        LOGGER.warning(
            "NLCD ASSUMPTION: treating dominant %s class %d as PLET %s "
            "for %d parcel(s) (first 20 pids: %s). Review these classifications "
            "before modeling; each generated input_land_cover row records the assumption.",
            ASSUMED_NLCD_CLASSES[code], code, NLCD_CLASSES[code], len(pids), pids[:20],
        )
    if unmapped_parcels:
        details = "; ".join(
            f"{code}: {len(pids)} parcel(s) (first 20 pids: {pids[:20]})"
            for code, pids in sorted(unmapped_parcels.items())
        )
        raise ValueError(f"Unmapped dominant NLCD codes: {details}; "
                         "classify these parcels explicitly before using the output")
    counts = Counter(row["value"] for row in classified)
    for cover, count in sorted(counts.items()):
        LOGGER.info("NLCD %d %s: %d parcels (%.2f%%)",
                    image.year, cover, count, 100 * count / len(classified))
    return _rows(classified, "NLCD parcel classifications")


def _replace_input_table(con: sqlite3.Connection, name: str, keys: tuple[str, ...],
                         rows: pd.DataFrame) -> None:
    """Replace one generated attribute table without touching other input tables."""
    if rows.duplicated(subset=list(keys)).any():
        raise ValueError(f"Duplicate {name} keys")
    con.execute(f'DROP TABLE IF EXISTS "{name}"')
    key_defs = ", ".join(f'"{key}" {"INTEGER" if key == "pid" else "TEXT"} NOT NULL'
                         for key in keys)
    con.execute(f'CREATE TABLE "{name}" (id INTEGER PRIMARY KEY, {key_defs}, '
                + ", ".join(f'"{stat}" {"TEXT" if name in ("input_land_cover", "input_hsg") and stat == "value" else "REAL"}'
                            for stat in STATS)
                + ', sample_group TEXT, units TEXT, notes TEXT, '
                + f'UNIQUE ({", ".join(keys)}))')
    con.execute("DELETE FROM gpkg_contents WHERE table_name=?", (name,))
    con.execute(
        "INSERT INTO gpkg_contents(table_name,data_type,identifier,description,srs_id) "
        "VALUES (?,'attributes',?,?,NULL)", (name, name, "Generated PLET input"),
    )
    columns = [*keys, *STATS, "sample_group", "units", "notes"]
    quoted = ", ".join(f'"{column}"' for column in columns)
    con.executemany(
        f'INSERT INTO "{name}" ({quoted}) VALUES ({", ".join("?" for _ in columns)})',
        [tuple(None if pd.isna(row.get(col)) else row.get(col) for col in columns)
         for row in rows.to_dict("records")],
    )


def _remove_input_table(con: sqlite3.Connection, name: str) -> None:
    con.execute(f'DROP TABLE IF EXISTS "{name}"')
    con.execute("DELETE FROM gpkg_contents WHERE table_name=?", (name,))


def _merge_hydrology_overrides(
    con: sqlite3.Connection, name: str, generated: pd.DataFrame,
) -> pd.DataFrame:
    """Keep existing land-cover/HSG rows, including their distributions."""
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    if not exists:
        return generated
    existing = pd.read_sql_query(f'SELECT * FROM "{name}"', con)
    required = {"land_cover", "hsg", "value"}
    if not required <= set(existing):
        raise ValueError(f"{name} is missing columns {sorted(required - set(existing))}")
    if existing[["land_cover", "hsg"]].isna().any().any():
        raise ValueError(f"{name} has null land-cover/HSG override keys")
    existing["land_cover"] = existing["land_cover"].map(_cover)
    existing["hsg"] = existing["hsg"].map(lambda value: str(value).strip().upper())
    if not set(existing["hsg"]) <= set(HSG_VALUES):
        raise ValueError(f"{name} has invalid HSG override values")
    if existing.duplicated(subset=["land_cover", "hsg"]).any():
        raise ValueError(f"{name} has duplicate hydrology overrides")
    existing = existing.drop(columns=["id", "fid"], errors="ignore")
    missing = generated.merge(
        existing[["land_cover", "hsg"]], on=["land_cover", "hsg"], how="left",
        indicator=True,
    )
    missing = missing.loc[missing["_merge"] == "left_only"].drop(columns="_merge")
    LOGGER.info("%s: retaining %d existing override rows, adding %d workbook rows",
                name, len(existing), len(missing))
    return pd.concat([existing, missing], ignore_index=True)


def _check_parcel_layer_conflict(path: Path, layer: str) -> None:
    """Reject conflicts and require schema-v3 parcel feature IDs for renaming."""
    if layer == "parcels":
        return
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as con:
        if con.execute(
            "SELECT 1 FROM sqlite_master WHERE lower(name)='parcels'"
        ).fetchone():
            raise ValueError(
                f"{path} already contains a parcels table; cannot rename {layer!r} "
                "without replacing the existing canonical layer"
            )
        quoted = '"' + layer.replace('"', '""') + '"'
        primary_keys = [
            (str(column[1]).lower(), str(column[2]).upper())
            for column in con.execute(f"PRAGMA table_info({quoted})")
            if column[5]
        ]
        if primary_keys != [("pid", "INTEGER")]:
            raise ValueError(
                f"Parcel layer {layer!r} must use pid INTEGER PRIMARY KEY to "
                "preserve feature IDs and parcel relationships on rename"
            )


def _rename_parcel_layer(path: Path, layer: str) -> None:
    """Use GDAL's GeoPackage SQL handler to rename layer and spatial metadata."""
    if layer == "parcels":
        return
    executable = shutil.which("ogrinfo")
    if executable is None:
        candidate = Path(sys.executable).parent / "Library" / "bin" / "ogrinfo.exe"
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        raise RuntimeError("GDAL ogrinfo is required to safely rename a GeoPackage layer")
    quoted = '"' + layer.replace('"', '""') + '"'
    result = subprocess.run(
        [executable, "-q", "-sql", f"ALTER TABLE {quoted} RENAME TO parcels", str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Could not rename parcel layer {layer!r}: {result.stderr or result.stdout}")
    with sqlite3.connect(path) as con:
        content = con.execute(
            "SELECT identifier FROM gpkg_contents WHERE table_name='parcels' AND data_type='features'"
        ).fetchone()
        geometry = con.execute(
            "SELECT 1 FROM gpkg_geometry_columns WHERE table_name='parcels'"
        ).fetchone()
        old = con.execute(
            "SELECT 1 FROM gpkg_contents WHERE table_name=?", (layer,)
        ).fetchone()
        integrity = con.execute("PRAGMA foreign_key_check").fetchall()
    if content is None or geometry is None or old or integrity:
        raise RuntimeError(f"GDAL did not produce a valid canonical parcels layer in {path}")
    LOGGER.info("Renamed copied GeoPackage layer %r to 'parcels' (identifier %r)",
                layer, content[0])


def prepare_plet_inputs(
    parcel_path: Path, parcel_layer: str, workbook: Path, *, sssurgo_hsg_true: bool = False,
    nlcd_source: Callable[[tuple[float, float, float, float]], NLCDImage] = download_nlcd,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Build both outputs; injectable NLCD source enables offline tests."""
    if not parcel_path.is_file() or not workbook.is_file():
        raise FileNotFoundError(f"Parcel GeoPackage and workbook must exist: {parcel_path}, {workbook}")
    output = parcel_path.with_name(f"{parcel_path.stem}_plet.gpkg")
    forcing = parcel_path.with_name(f"{parcel_path.stem}_plet_huc12.gpkg")
    temp_forcing = forcing.with_name(f".{forcing.stem}.tmp.gpkg")
    if output.exists() or forcing.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output} or {forcing}")
    if temp_forcing.exists():
        raise FileExistsError(f"Refusing to overwrite existing HUC12 temporary package: {temp_forcing}")
    _check_parcel_layer_conflict(parcel_path, parcel_layer)

    parcels_wgs84, bounds = parcel_footprint_wgs84(parcel_path, parcel_layer=parcel_layer)
    hucs = retain_hucs_by_parcel_overlap_threshold(
        download_wbd_huc12(bounds), parcels_wgs84, threshold=0.75)
    retained = set(hucs["huc12"].astype(str))
    export = read_initial_plet_export(workbook)
    tables, hydrology = workbook_tables(workbook, retained, export=export)
    if sssurgo_hsg_true:
        tables["hsg"] = pd.DataFrame()
    export = export.loc[export["huc12"].isin(retained)]
    tables["rain_correction_fraction"], tables["runoff_day_fraction"], _ = (
        reconstruct_rain_factors(export, hucs))
    parcels = _read_parcels(parcel_path, parcel_layer)
    land_cover = parcel_nlcd_land_cover(parcels, source=nlcd_source, verbose=verbose)

    hsg_rows = None
    if sssurgo_hsg_true:
        domain = parcels_wgs84.geometry.union_all()
        soil = clip_to_domain(
            normalize_wfs_axis_order(download_ssurgo(domain, 0.1, 120), domain), domain)
        soil_parcels = parcels[["pid", "geometry"]].rename(columns={"pid": "_fid"})
        soil_values, _ = _parcel_hsg_values(soil_parcels, soil, domain)
        hsg_rows = []
        for pid in parcels["pid"]:
            dominant = soil_values.get(int(pid), (None, None))[0]
            if dominant not in HSG_VALUES:
                raise ValueError(f"Parcel {pid} has no valid SSURGO HSG A/B/C/D: {dominant!r}")
            hsg_rows.append({"pid": int(pid), "value": dominant, "notes": "Dominant SSURGO HSG"})
        for hsg, count in sorted(Counter(row["value"] for row in hsg_rows).items()):
            LOGGER.info("SSURGO HSG %s: %d parcels (%.2f%%)",
                        hsg, count, 100 * count / len(hsg_rows))

    created: list[Path] = []
    try:
        for path in (output, forcing):
            with path.open("xb"):
                pass
            created.append(path)
        _copy_sqlite_database(parcel_path, output)
        _rename_parcel_layer(output, parcel_layer)
        write_huc12_package(forcing, hucs, tables)
        assignments = assign_parcels_to_huc12(
            output, forcing, parcel_layer="parcels", huc_layer="huc12")
        write_parcel_huc12(output, assignments)
        with sqlite3.connect(output) as con:
            _replace_input_table(con, "input_land_cover", ("pid",), land_cover)
            if hsg_rows is None:
                _remove_input_table(con, "input_hsg")
            else:
                _replace_input_table(con, "input_hsg", ("pid",), pd.DataFrame(hsg_rows))
            for parameter, name in (("cn", "input_curve_number"),
                                    ("infiltration_fraction", "input_infiltration_fraction")):
                merged = _merge_hydrology_overrides(con, name, hydrology[parameter])
                _replace_input_table(con, name, ("land_cover", "hsg"), merged)
        LOGGER.info("HUC12 assignment: %d parcels, %d retained HUC12s; %.2f%% "
                    "of parcels have >=75%% overlap",
                    len(assignments), len(retained),
                    100 * (assignments["area_fraction"] >= 0.75).mean())
    except Exception:
        for path in (*created, temp_forcing):
            if path.exists():
                path.unlink()
        raise
    return output, forcing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parcels", type=Path, help="Input parcel GeoPackage")
    parser.add_argument("parcel_layer", help="Parcel polygon layer name")
    parser.add_argument("workbook", type=Path, help="EPA PLET XLSX export")
    parser.add_argument("--sssurgo_hsg_true", action="store_true",
                        help="Use dominant USDA SSURGO HSG per parcel instead of workbook HUC12 SHG")
    parser.add_argument("--verbose", action="store_true",
                        help="Log each parcel's NLCD class percentages")
    parser.add_argument("--log-path", type=Path,
                        help="Audit log path (default: <parcel-stem>_plet.log beside input)")
    args = parser.parse_args()
    log_path = args.log_path or args.parcels.with_name(f"{args.parcels.stem}_plet.log")
    if log_path.resolve() in {
        args.parcels.resolve(), args.workbook.resolve(),
        args.parcels.with_name(f"{args.parcels.stem}_plet.gpkg").resolve(),
        args.parcels.with_name(f"{args.parcels.stem}_plet_huc12.gpkg").resolve(),
    }:
        parser.error("--log-path must not point to an input or GeoPackage output")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    LOGGER.addHandler(console)
    try:
        LOGGER.info("Starting PLET input preparation; audit log: %s", log_path)
        out, forcing = prepare_plet_inputs(
            args.parcels, args.parcel_layer, args.workbook,
            sssurgo_hsg_true=args.sssurgo_hsg_true, verbose=args.verbose)
        LOGGER.info("Created %s and %s", out, forcing)
    finally:
        LOGGER.removeHandler(handler)
        LOGGER.removeHandler(console)
        handler.close()
        console.close()
    print(f"PLET audit log: {log_path}")
    print(f"Created {out} and {forcing}")


if __name__ == "__main__":
    main()
