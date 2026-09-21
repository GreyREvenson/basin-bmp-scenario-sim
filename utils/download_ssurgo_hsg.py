#!/usr/bin/env python3
"""
Download USDA-NRCS SSURGO Hydrologic Soil Group (HSG) polygons for a
GeoPackage domain and, optionally, annotate a parcel GeoPackage with HSG
information.

Domain behavior
---------------
1. Reads every spatial layer in the input domain GeoPackage.
2. Treats every valid Polygon/MultiPolygon feature as part of the domain.
3. Dissolves those polygons into one domain.
4. Downloads USDA-NRCS SSURGO map-unit polygons from Soil Data Access WFS.
5. Uses the map-unit "Hydrologic Group - Dominant Conditions" attribute.
6. Clips the SSURGO layer to the dissolved domain.
7. Writes:
      - domain
      - ssurgo_hsg

Optional parcel behavior
------------------------
When --parcels is supplied:
1. Treats every valid Polygon/MultiPolygon feature in every spatial layer of
   the parcel GeoPackage as a parcel.
2. Requires at least one parcel to overlap the domain by positive area.
3. For each parcel, computes:
      hsg_dom : hydrologic soil group covering the largest area of the parcel
                (within the downloaded/clipped domain)
      hsg_all : all hydrologic soil groups occurring in the parcel, separated
                by semicolons
4. Creates a SECOND output GeoPackage as a database-level copy of the parcel
   GeoPackage, preserving its existing spatial tables, nonspatial tables,
   relations, indexes, triggers, and metadata.
5. Leaves each spatial parcel table unchanged and creates a nonspatial HSG
   attribute table for each polygon parcel layer. For a parcel layer named
   "parcels", the new table is "parcels_hsg". The HSG table uses the same
   primary-key column name/value as the spatial parcel table and stores:
      hsg_dom : areal-dominant hydrologic soil group
      hsg_all : all hydrologic soil groups occurring in the parcel
   The shared primary key is also declared as a SQLite foreign key back to
   the spatial parcel table, providing a one-to-one link.

Parcels with no positive-area overlap with the clipped SSURGO HSG layer are
retained in the HSG table and receive NULL for hsg_dom and hsg_all.

Existing-HSG reuse mode
-----------------------
When --parcels and --hsg-input are both supplied, the script DOES NOT contact
USDA or create a new whole-domain HSG GeoPackage. Instead it reads the
"ssurgo_hsg" layer from the existing GeoPackage previously created by this
script, validates that it overlaps the current domain, and uses it to annotate
the parcel output.

Examples
--------
    python download_ssurgo_hsg.py domain.gpkg

    python download_ssurgo_hsg.py domain.gpkg \
        --parcels parcels.gpkg

    python download_ssurgo_hsg.py domain.gpkg \
        --parcels parcels.gpkg \
        --parcels-output parcels_with_hsg.gpkg \
        --overwrite

    # Reuse an existing whole-domain HSG file; no USDA download occurs.
    python download_ssurgo_hsg.py domain.gpkg \
        --parcels parcels.gpkg \
        --hsg-input domain_ssurgo_hsg.gpkg

Dependencies
------------
    pip install geopandas pyogrio shapely requests
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import requests
from requests.adapters import HTTPAdapter
from shapely import make_valid, union_all
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import transform as shapely_transform
from urllib3.util.retry import Retry


WFS_URL = (
    "https://SDMDataAccess.sc.egov.usda.gov/"
    "Spatial/SDMWGS84Geographic.wfs"
)
WFS_TYPENAME = "mapunitpolyextended"
WGS84 = "EPSG:4326"

HSG_DOM_FIELD = "hsg_dom"
HSG_ALL_FIELD = "hsg_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dissolve polygon features in a GeoPackage into a domain, "
            "download USDA-NRCS SSURGO hydrologic soil groups (or reuse an "
            "existing HSG GeoPackage), and optionally create linked "
            "nonspatial per-parcel HSG attribute tables in a copied parcel "
            "GeoPackage."
        )
    )
    parser.add_argument(
        "gpkg",
        type=Path,
        help="Input GeoPackage whose polygon features define the domain.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "SSURGO output GeoPackage. Default: "
            "<domain_stem>_ssurgo_hsg.gpkg beside the domain GeoPackage."
        ),
    )
    parser.add_argument(
        "--parcels",
        type=Path,
        default=None,
        help=(
            "Optional parcel GeoPackage. Every valid polygon feature in its "
            "spatial layers is treated as a parcel."
        ),
    )
    parser.add_argument(
        "--hsg-input",
        "--existing-hsg",
        dest="hsg_input",
        type=Path,
        default=None,
        help=(
            "Optional whole-domain HSG GeoPackage previously created by this "
            "script. Requires --parcels. When supplied, the USDA download is "
            "skipped and the existing 'ssurgo_hsg' layer is used."
        ),
    )
    parser.add_argument(
        "--parcels-output",
        type=Path,
        default=None,
        help=(
            "Output copy of --parcels with nonspatial per-parcel HSG table(s). "
            "The spatial parcel table(s) are left unchanged. Default: "
            "<parcels_stem>_ssurgo_hsg.gpkg beside the parcel file."
        ),
    )
    parser.add_argument(
        "--tile-degrees",
        type=float,
        default=0.5,
        help=(
            "Maximum WGS84 width/height of each WFS request tile in degrees. "
            "Smaller values make more requests. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout per USDA request in seconds. Default: 180.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output GeoPackages.",
    )
    return parser.parse_args()


def polygonal_part(geom):
    """Return only polygonal area from a geometry, or None."""
    if geom is None or geom.is_empty:
        return None

    geom = make_valid(geom)

    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return geom

    if isinstance(geom, GeometryCollection):
        pieces = []
        for part in geom.geoms:
            poly = polygonal_part(part)
            if poly is None or poly.is_empty:
                continue
            if isinstance(poly, MultiPolygon):
                pieces.extend(poly.geoms)
            else:
                pieces.append(poly)

        if not pieces:
            return None
        return polygonal_part(union_all(pieces))

    return None


def as_multipolygon(geom):
    """Normalize a polygonal geometry to MultiPolygon."""
    geom = polygonal_part(geom)
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return geom


def build_domain(gpkg: Path):
    """
    Read every spatial layer and dissolve all valid polygonal features.

    Returns
    -------
    domain_native : shapely geometry
        Dissolved domain in the CRS of the first polygonal layer.
    native_crs : pyproj CRS
        CRS used for the SSURGO output GeoPackage.
    layer_names : list[str]
        Polygonal layers contributing to the domain.
    """
    if not gpkg.exists():
        raise FileNotFoundError(f"Input GeoPackage does not exist: {gpkg}")

    layers = gpd.list_layers(gpkg)
    if layers.empty:
        raise ValueError(f"No layers found in GeoPackage: {gpkg}")

    native_crs = None
    all_polygons = []
    contributing_layers = []

    for row in layers.itertuples(index=False):
        layer_name = row.name
        geometry_type = getattr(row, "geometry_type", None)

        if geometry_type is None or pd.isna(geometry_type):
            continue

        gdf = gpd.read_file(gpkg, layer=layer_name)
        if gdf.empty or "geometry" not in gdf.columns:
            continue
        if gdf.crs is None:
            raise ValueError(
                f"Layer '{layer_name}' in {gpkg} has no CRS."
            )

        polygon_geoms = []
        for geom in gdf.geometry:
            poly = polygonal_part(geom)
            if poly is not None and not poly.is_empty:
                polygon_geoms.append(poly)

        if not polygon_geoms:
            print(
                f"Skipping layer '{layer_name}': no valid polygonal area.",
                file=sys.stderr,
            )
            continue

        if native_crs is None:
            native_crs = gdf.crs

        if gdf.crs != native_crs:
            polygon_geoms = list(
                gpd.GeoSeries(polygon_geoms, crs=gdf.crs)
                .to_crs(native_crs)
            )

        all_polygons.extend(polygon_geoms)
        contributing_layers.append(layer_name)

    if not all_polygons or native_crs is None:
        raise ValueError(
            "The domain GeoPackage contains no valid Polygon/MultiPolygon "
            "features."
        )

    domain = polygonal_part(union_all(all_polygons))
    if domain is None or domain.is_empty:
        raise ValueError("The dissolved GeoPackage domain is empty.")

    return domain, native_crs, contributing_layers


def make_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "ssurgo-hsg-downloader/1.1 "
                "(USDA-NRCS Soil Data Access client)"
            )
        }
    )
    return session


def iter_tiles(domain_wgs84, tile_degrees: float):
    """Yield WGS84 bounding boxes that intersect the domain."""
    if tile_degrees <= 0:
        raise ValueError("--tile-degrees must be greater than zero.")

    minx, miny, maxx, maxy = domain_wgs84.bounds
    width = max(maxx - minx, 1e-12)
    height = max(maxy - miny, 1e-12)

    nx = max(1, math.ceil(width / tile_degrees))
    ny = max(1, math.ceil(height / tile_degrees))
    dx = width / nx
    dy = height / ny

    for ix in range(nx):
        x0 = minx + ix * dx
        x1 = maxx if ix == nx - 1 else minx + (ix + 1) * dx

        for iy in range(ny):
            y0 = miny + iy * dy
            y1 = maxy if iy == ny - 1 else miny + (iy + 1) * dy

            tile = box(x0, y0, x1, y1)
            if tile.intersects(domain_wgs84):
                yield (x0, y0, x1, y1)


def download_wfs_tile(
    session: requests.Session,
    bbox_tuple,
    timeout: int,
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox_tuple

    params = {
        "SERVICE": "WFS",
        "VERSION": "1.1.0",
        "REQUEST": "GetFeature",
        "TYPENAME": WFS_TYPENAME,
        "SRSNAME": WGS84,
        # SDA's WFS expects a four-value BBOX here. SRSNAME supplies the CRS.
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "OUTPUTFORMAT": "GML2",
        "MAXFEATURES": "250000",
    }

    response = session.get(WFS_URL, params=params, timeout=timeout)

    if not response.ok:
        detail = response.text[:4000].strip()
        raise RuntimeError(
            f"USDA Soil Data Access WFS returned HTTP {response.status_code}.\n"
            f"Request URL: {response.url}\n"
            f"Response:\n{detail}"
        )

    head = response.content[:4000].lower()
    if b"exceptionreport" in head or b"serviceexception" in head:
        raise RuntimeError(
            "USDA WFS returned an exception:\n"
            + response.text[:4000]
        )

    # On Windows a NamedTemporaryFile cannot reliably be reopened by
    # GDAL/pyogrio while its original handle remains open. Write to a temp
    # directory so the file is closed before GeoPandas reads it.
    with tempfile.TemporaryDirectory() as tmpdir:
        gml_path = Path(tmpdir) / "ssurgo_tile.gml"
        gml_path.write_bytes(response.content)

        try:
            gdf = gpd.read_file(gml_path)
        except Exception as exc:
            if (
                b"featuremember" not in response.content.lower()
                and b"member>" not in response.content.lower()
            ):
                return gpd.GeoDataFrame(geometry=[], crs=WGS84)

            content_type = response.headers.get("Content-Type", "")
            preview = response.text[:1000].strip()
            raise RuntimeError(
                "Could not parse USDA WFS GML response.\n"
                f"Content-Type: {content_type}\n"
                f"Response preview:\n{preview}"
            ) from exc

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    else:
        gdf = gdf.to_crs(WGS84)

    return gdf


def normalize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Lowercase WFS field names while preserving the geometry column."""
    geometry_name = gdf.geometry.name
    rename = {}
    used = set()

    for col in gdf.columns:
        if col == geometry_name:
            continue

        base = str(col).split(":")[-1].lower()
        candidate = base
        i = 2
        while candidate in used:
            candidate = f"{base}_{i}"
            i += 1
        used.add(candidate)
        rename[col] = candidate

    gdf = gdf.rename(columns=rename)
    if geometry_name != "geometry":
        gdf = gdf.rename_geometry("geometry")
    return gdf


def download_ssurgo(domain_wgs84, tile_degrees: float, timeout: int):
    tiles = list(iter_tiles(domain_wgs84, tile_degrees))
    print(f"USDA WFS request tiles: {len(tiles)}")

    session = make_session()
    frames = []

    for i, tile_bbox in enumerate(tiles, start=1):
        print(f"Downloading SSURGO tile {i}/{len(tiles)} ...")
        tile_gdf = download_wfs_tile(session, tile_bbox, timeout)
        if not tile_gdf.empty:
            frames.append(tile_gdf)

    if not frames:
        raise RuntimeError(
            "USDA Soil Data Access returned no SSURGO map-unit polygons "
            "for the input domain."
        )

    soil = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=WGS84,
    )
    soil = normalize_columns(soil)

    if "hydgrpdcd" not in soil.columns:
        available = ", ".join(
            sorted(c for c in soil.columns if c != "geometry")
        )
        raise RuntimeError(
            "The USDA mapunitpolyextended response did not contain "
            "'hydgrpdcd'. Available fields were:\n"
            f"{available}"
        )

    # Remove full duplicate polygons returned in adjacent request tiles.
    soil["_geom_wkb"] = soil.geometry.to_wkb(hex=True)
    dedup_fields = ["_geom_wkb"]
    if "mukey" in soil.columns:
        dedup_fields.insert(0, "mukey")
    soil = (
        soil.drop_duplicates(subset=dedup_fields)
        .drop(columns="_geom_wkb")
        .reset_index(drop=True)
    )

    return soil


def _swap_xy_geometry(geom):
    """Swap X/Y coordinates in a Shapely geometry."""
    if geom is None or geom.is_empty:
        return geom

    def swapper(x, y, z=None):
        if z is None:
            return y, x
        return y, x, z

    return shapely_transform(swapper, geom)


def normalize_wfs_axis_order(
    soil: gpd.GeoDataFrame,
    domain_wgs84,
) -> gpd.GeoDataFrame:
    """
    Detect and repair the WFS 1.1 / EPSG:4326 latitude-longitude axis swap.
    """
    if soil.empty:
        return soil

    direct_hits = int(soil.geometry.intersects(domain_wgs84).sum())
    direct_bounds = tuple(float(v) for v in soil.total_bounds)

    print(
        "Downloaded SSURGO bounds as read: "
        f"{direct_bounds[0]:.6f}, {direct_bounds[1]:.6f}, "
        f"{direct_bounds[2]:.6f}, {direct_bounds[3]:.6f}"
    )
    print(f"Direct intersections with domain: {direct_hits:,}")

    if direct_hits > 0:
        return soil

    swapped = soil.copy()
    swapped["geometry"] = swapped.geometry.map(_swap_xy_geometry)
    swapped = swapped.set_crs(WGS84, allow_override=True)

    swapped_hits = int(swapped.geometry.intersects(domain_wgs84).sum())
    swapped_bounds = tuple(float(v) for v in swapped.total_bounds)

    print(
        "SSURGO bounds after X/Y swap test: "
        f"{swapped_bounds[0]:.6f}, {swapped_bounds[1]:.6f}, "
        f"{swapped_bounds[2]:.6f}, {swapped_bounds[3]:.6f}"
    )
    print(f"Intersections after X/Y swap test: {swapped_hits:,}")

    if swapped_hits > 0:
        print(
            "Detected WFS/GML latitude-longitude axis reversal; "
            "using longitude-latitude coordinates."
        )
        return swapped

    dminx, dminy, dmaxx, dmaxy = domain_wgs84.bounds
    raise RuntimeError(
        "USDA returned SSURGO features, but neither their original nor "
        "X/Y-swapped coordinates intersect the requested domain.\n"
        f"Domain bounds: {dminx:.6f}, {dminy:.6f}, "
        f"{dmaxx:.6f}, {dmaxy:.6f}\n"
        f"Downloaded bounds: {direct_bounds}\n"
        f"X/Y-swapped bounds: {swapped_bounds}"
    )


def clip_to_domain(
    soil: gpd.GeoDataFrame,
    domain_wgs84,
) -> gpd.GeoDataFrame:
    soil = soil.loc[
        soil.geometry.notna()
        & ~soil.geometry.is_empty
        & soil.geometry.intersects(domain_wgs84)
    ].copy()

    if soil.empty:
        raise RuntimeError(
            "SSURGO polygons were downloaded, but none intersect the domain."
        )

    soil["geometry"] = soil.geometry.intersection(domain_wgs84)
    soil["geometry"] = soil.geometry.map(as_multipolygon)
    soil = soil.loc[
        soil.geometry.notna() & ~soil.geometry.is_empty
    ].copy()

    preferred = [
        "areasymbol",
        "spatialver",
        "spatialversion",
        "musym",
        "muname",
        "mukey",
        "hydgrpdcd",
    ]
    keep = [c for c in preferred if c in soil.columns]
    soil = soil[keep + ["geometry"]].copy()
    soil = soil.rename(columns={"hydgrpdcd": "hsg"})
    soil["hsg"] = soil["hsg"].astype("string")

    return soil.reset_index(drop=True)



def load_existing_hsg(
    hsg_gpkg: Path,
    domain_wgs84,
) -> gpd.GeoDataFrame:
    """
    Load and validate an existing whole-domain HSG GeoPackage.

    The file must contain the ``ssurgo_hsg`` layer written by this script and
    that layer must contain an ``hsg`` field. Only valid polygonal features
    having positive-area overlap with the current domain are retained.
    """
    if not hsg_gpkg.exists():
        raise FileNotFoundError(
            f"Existing HSG GeoPackage does not exist: {hsg_gpkg}"
        )

    layers = gpd.list_layers(hsg_gpkg)
    if layers.empty or "ssurgo_hsg" not in set(layers["name"].astype(str)):
        available = (
            ", ".join(layers["name"].astype(str).tolist())
            if not layers.empty
            else "(none)"
        )
        raise ValueError(
            f"Existing HSG GeoPackage must contain a 'ssurgo_hsg' layer. "
            f"Available layers: {available}"
        )

    soil = gpd.read_file(hsg_gpkg, layer="ssurgo_hsg")
    if soil.empty:
        raise ValueError(
            f"The 'ssurgo_hsg' layer is empty in: {hsg_gpkg}"
        )
    if soil.crs is None:
        raise ValueError(
            f"The 'ssurgo_hsg' layer has no CRS in: {hsg_gpkg}"
        )
    if "hsg" not in soil.columns:
        raise ValueError(
            f"The 'ssurgo_hsg' layer in {hsg_gpkg} does not contain "
            "the required 'hsg' field."
        )

    # Keep only valid polygonal geometry.
    soil = soil.copy()
    soil["geometry"] = soil.geometry.map(polygonal_part)
    soil = soil.loc[
        soil.geometry.notna() & ~soil.geometry.is_empty
    ].copy()
    if soil.empty:
        raise ValueError(
            f"The 'ssurgo_hsg' layer contains no valid polygon features: "
            f"{hsg_gpkg}"
        )

    soil = soil.to_crs(WGS84)

    # Restrict to the current domain. This also verifies that the supplied HSG
    # file actually corresponds to (or at least overlaps) this domain.
    intersects = soil.geometry.intersects(domain_wgs84)
    if not bool(intersects.any()):
        bounds = tuple(float(v) for v in soil.total_bounds)
        dminx, dminy, dmaxx, dmaxy = domain_wgs84.bounds
        raise ValueError(
            "The existing HSG file has no polygon features intersecting the "
            "current domain.\n"
            f"Domain bounds: {dminx:.6f}, {dminy:.6f}, "
            f"{dmaxx:.6f}, {dmaxy:.6f}\n"
            f"HSG bounds: {bounds}"
        )

    soil = soil.loc[intersects].copy()
    soil["geometry"] = soil.geometry.intersection(domain_wgs84)
    soil["geometry"] = soil.geometry.map(as_multipolygon)
    soil = soil.loc[
        soil.geometry.notna() & ~soil.geometry.is_empty
    ].copy()

    # Normalize HSG values the same way as the download path.
    soil["hsg"] = soil["hsg"].astype("string").str.strip()

    if soil.empty:
        raise ValueError(
            "The existing HSG file has no positive-area polygon overlap with "
            "the current domain."
        )

    return soil.reset_index(drop=True)

def write_ssurgo_output(
    output: Path,
    domain_native,
    native_crs,
    soil_wgs84: gpd.GeoDataFrame,
):
    output.parent.mkdir(parents=True, exist_ok=True)

    domain_gdf = gpd.GeoDataFrame(
        {"source": ["dissolved input GeoPackage polygon features"]},
        geometry=[as_multipolygon(domain_native)],
        crs=native_crs,
    )
    soil_out = soil_wgs84.to_crs(native_crs)

    domain_gdf.to_file(
        output,
        layer="domain",
        driver="GPKG",
        mode="w",
        index=False,
    )
    soil_out.to_file(
        output,
        layer="ssurgo_hsg",
        driver="GPKG",
        mode="a",
        index=False,
    )


def _quote_ident(name: str) -> str:
    """Quote a SQLite identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def _copy_sqlite_database(source: Path, destination: Path):
    """
    Copy a GeoPackage using SQLite's backup API.

    This preserves the database contents, including nonspatial tables,
    relationship metadata, indexes, and triggers, and includes committed WAL
    content if the source database happens to be in WAL mode.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    src_uri = source.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(src_uri, uri=True) as src_conn:
        with sqlite3.connect(destination) as dst_conn:
            src_conn.backup(dst_conn)


def _primary_key_info(gpkg: Path, layer_name: str):
    """Return (name, declared SQLite type) for a feature table primary key."""
    with sqlite3.connect(gpkg) as conn:
        rows = conn.execute(
            f"PRAGMA table_info({_quote_ident(layer_name)})"
        ).fetchall()

    pk_rows = [row for row in rows if int(row[5]) > 0]
    if len(pk_rows) != 1:
        raise RuntimeError(
            f"Parcel layer '{layer_name}' must have exactly one primary-key "
            "column so parcel results can be linked safely."
        )

    name = str(pk_rows[0][1])
    declared_type = str(pk_rows[0][2] or "INTEGER").strip() or "INTEGER"
    return name, declared_type


def _primary_key_column(gpkg: Path, layer_name: str) -> str:
    """Return the GeoPackage feature table primary-key/FID column name."""
    return _primary_key_info(gpkg, layer_name)[0]


def _choose_area_crs(gdf: gpd.GeoDataFrame):
    """
    Choose a CRS suitable for comparing intersection areas.

    If the parcel CRS is projected, use it. Otherwise estimate a local UTM
    zone, with EPSG:6933 as a global equal-area fallback.
    """
    if gdf.crs is None:
        raise ValueError("Parcel layer has no CRS.")

    if getattr(gdf.crs, "is_projected", False):
        return gdf.crs

    try:
        estimated = gdf.estimate_utm_crs()
    except Exception:
        estimated = None

    return estimated or "EPSG:6933"


def _hsg_sort_key(value: str):
    """Provide a stable, human-friendly ordering of common HSG labels."""
    order = {
        "A": 0,
        "A/D": 1,
        "B": 2,
        "B/D": 3,
        "C": 4,
        "C/D": 5,
        "D": 6,
    }
    text = str(value).strip()
    return (order.get(text.upper(), 100), text.upper())


def _valid_parcel_frame(
    parcel_gpkg: Path,
    layer_name: str,
) -> gpd.GeoDataFrame:
    """
    Read a parcel feature table with actual GeoPackage FIDs as the index and
    retain only rows that have valid polygonal area for calculation.
    """
    gdf = pyogrio.read_dataframe(
        parcel_gpkg,
        layer=layer_name,
        fid_as_index=True,
    )

    # pyogrio returns a plain pandas DataFrame for non-spatial GeoPackage
    # tables. Those tables must be preserved in the copied GeoPackage, but
    # they are not parcel layers and should not enter the HSG calculation.
    if not isinstance(gdf, gpd.GeoDataFrame) or "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(
            {"_fid": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs=None),
        )

    if gdf.empty:
        return gpd.GeoDataFrame(
            {"_fid": pd.Series(dtype="int64")},
            geometry=gpd.GeoSeries([], crs=gdf.crs),
            crs=gdf.crs,
        )

    if gdf.crs is None:
        raise ValueError(
            f"Parcel layer '{layer_name}' has no CRS. Assign a CRS first."
        )

    fids = []
    geoms = []

    for fid, geom in gdf.geometry.items():
        poly = polygonal_part(geom)
        if poly is None or poly.is_empty:
            continue
        fids.append(int(fid))
        geoms.append(poly)

    return gpd.GeoDataFrame(
        {"_fid": fids},
        geometry=geoms,
        crs=gdf.crs,
    )


def _parcel_hsg_values(
    parcels: gpd.GeoDataFrame,
    soil_wgs84: gpd.GeoDataFrame,
    domain_wgs84,
):
    """
    Calculate hsg_dom and hsg_all for one polygon feature table.

    Returns
    -------
    results : dict[int, tuple[str|None, str|None]]
        Mapping of GeoPackage FID to (hsg_dom, hsg_all).
    overlap_count : int
        Number of valid parcel polygons with positive-area domain overlap.
    """
    if parcels.empty:
        return {}, 0

    domain_layer = (
        gpd.GeoSeries([domain_wgs84], crs=WGS84)
        .to_crs(parcels.crs)
        .iloc[0]
    )

    overlaps_domain = []
    for geom in parcels.geometry:
        inter = geom.intersection(domain_layer)
        overlaps_domain.append(
            not inter.is_empty and polygonal_part(inter) is not None
            and inter.area > 0
        )

    overlap_count = int(sum(overlaps_domain))

    area_crs = _choose_area_crs(parcels)
    parcels_area = parcels.to_crs(area_crs)
    soil_area = soil_wgs84.to_crs(area_crs)

    # Avoid treating missing/blank HSG values as classifications.
    hsg_text = soil_area["hsg"].astype("string").str.strip()
    rated = hsg_text.notna() & (hsg_text != "")
    soil_area = soil_area.loc[rated].copy()
    soil_area["hsg"] = hsg_text.loc[rated]

    results = {}

    if soil_area.empty:
        for fid in parcels["_fid"]:
            results[int(fid)] = (None, None)
        return results, overlap_count

    sindex = soil_area.sindex

    for fid_value, parcel_geom in zip(
        parcels_area["_fid"].tolist(),
        parcels_area.geometry,
    ):
        fid = int(fid_value)

        totals = defaultdict(float)
        candidate_positions = list(
            sindex.query(parcel_geom, predicate="intersects")
        )

        for pos in candidate_positions:
            soil_row = soil_area.iloc[int(pos)]
            inter = parcel_geom.intersection(soil_row.geometry)
            if inter.is_empty:
                continue
            area = float(inter.area)
            if area <= 0:
                continue

            hsg = str(soil_row["hsg"]).strip()
            if hsg:
                totals[hsg] += area

        if not totals:
            results[fid] = (None, None)
            continue

        # Largest total intersection area wins. A deterministic HSG sort key
        # breaks an exact numerical tie.
        dominant = sorted(
            totals.items(),
            key=lambda item: (-item[1], _hsg_sort_key(item[0])),
        )[0][0]

        all_hsg = ";".join(sorted(totals, key=_hsg_sort_key))
        results[fid] = (dominant, all_hsg)

    return results, overlap_count


def _hsg_table_name(layer_name: str) -> str:
    """Name of the nonspatial HSG table associated with a parcel layer."""
    return f"{layer_name}_hsg"


def _create_hsg_attribute_table(
    conn: sqlite3.Connection,
    parcel_layer: str,
    pk_col: str,
    pk_type: str,
):
    """
    Create a GeoPackage nonspatial attribute table linked one-to-one to a
    spatial parcel feature table by the parcel table's primary key.

    The HSG table primary key uses the same column name and values as the
    parcel feature table and also carries a SQLite FOREIGN KEY reference to
    the parcel feature table primary key.
    """
    table_name = _hsg_table_name(parcel_layer)
    table_q = _quote_ident(table_name)
    parcel_q = _quote_ident(parcel_layer)
    pk_q = _quote_ident(pk_col)

    # This output is a fresh database copy. If the input already contains a
    # prior table created by this script, replace that specific derived table
    # so results cannot become stale.
    conn.execute(
        "DELETE FROM gpkg_contents WHERE table_name = ?",
        (table_name,),
    )
    conn.execute(f"DROP TABLE IF EXISTS {table_q}")

    conn.execute(
        f"""
        CREATE TABLE {table_q} (
            {pk_q} {pk_type} NOT NULL PRIMARY KEY,
            {_quote_ident(HSG_DOM_FIELD)} TEXT,
            {_quote_ident(HSG_ALL_FIELD)} TEXT,
            FOREIGN KEY ({pk_q})
                REFERENCES {parcel_q} ({pk_q})
                ON UPDATE CASCADE
                ON DELETE CASCADE
        )
        """
    )

    # Register this as a nonspatial GeoPackage attributes table so desktop GIS
    # software such as QGIS exposes it as a regular table.
    conn.execute(
        """
        INSERT INTO gpkg_contents (
            table_name,
            data_type,
            identifier,
            description,
            last_change,
            min_x,
            min_y,
            max_x,
            max_y,
            srs_id
        )
        VALUES (
            ?,
            'attributes',
            ?,
            ?,
            strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
            NULL, NULL, NULL, NULL, NULL
        )
        """,
        (
            table_name,
            table_name,
            f"SSURGO hydrologic soil group attributes for {parcel_layer}",
        ),
    )

    return table_name


def annotate_parcel_gpkg(
    parcel_gpkg: Path,
    parcel_output: Path,
    soil_wgs84: gpd.GeoDataFrame,
    domain_wgs84,
):
    """
    Create a preserved copy of a parcel GeoPackage and write per-parcel HSG
    results to nonspatial attribute table(s), leaving parcel feature tables
    unchanged.
    """
    if not parcel_gpkg.exists():
        raise FileNotFoundError(
            f"Parcel GeoPackage does not exist: {parcel_gpkg}"
        )

    layers = gpd.list_layers(parcel_gpkg)
    if layers.empty:
        raise ValueError(
            f"No layers found in parcel GeoPackage: {parcel_gpkg}"
        )

    layer_results = {}
    total_valid_parcels = 0
    total_domain_overlaps = 0
    polygon_layers = []

    for row in layers.itertuples(index=False):
        layer_name = row.name
        geometry_type = getattr(row, "geometry_type", None)

        # Preserve nonspatial tables in the database copy, but do not attempt
        # to interpret them as parcels.
        if geometry_type is None or pd.isna(geometry_type):
            continue

        parcels = _valid_parcel_frame(parcel_gpkg, layer_name)
        if parcels.empty:
            continue

        # Verify the spatial layer has a single usable primary key before
        # spending time on the overlay calculation.
        _primary_key_info(parcel_gpkg, layer_name)

        polygon_layers.append(layer_name)
        total_valid_parcels += len(parcels)

        results, overlap_count = _parcel_hsg_values(
            parcels=parcels,
            soil_wgs84=soil_wgs84,
            domain_wgs84=domain_wgs84,
        )
        total_domain_overlaps += overlap_count
        layer_results[layer_name] = results

        classified = sum(
            1 for dominant, _ in results.values()
            if dominant is not None
        )
        print(
            f"Parcel layer '{layer_name}': "
            f"{len(parcels):,} valid polygon parcels, "
            f"{overlap_count:,} overlap domain, "
            f"{classified:,} assigned HSG."
        )

    if total_valid_parcels == 0:
        raise ValueError(
            "The parcel GeoPackage contains no valid polygon features."
        )

    if total_domain_overlaps == 0:
        raise ValueError(
            "The parcel GeoPackage has valid polygon features, but none "
            "overlap the dissolved domain by positive area."
        )

    _copy_sqlite_database(parcel_gpkg, parcel_output)

    created_tables = []
    with sqlite3.connect(parcel_output) as conn:
        # Enabling FK enforcement here validates inserts against the copied
        # spatial parcel table. The constraint remains part of the schema for
        # clients that also enable SQLite foreign keys.
        conn.execute("PRAGMA foreign_keys = ON")

        for layer_name in polygon_layers:
            pk_col, pk_type = _primary_key_info(parcel_output, layer_name)
            hsg_table = _create_hsg_attribute_table(
                conn=conn,
                parcel_layer=layer_name,
                pk_col=pk_col,
                pk_type=pk_type,
            )
            created_tables.append(hsg_table)

            table_q = _quote_ident(hsg_table)
            pk_q = _quote_ident(pk_col)
            dom_q = _quote_ident(HSG_DOM_FIELD)
            all_q = _quote_ident(HSG_ALL_FIELD)

            insert_sql = (
                f"INSERT INTO {table_q} "
                f"({pk_q}, {dom_q}, {all_q}) VALUES (?, ?, ?)"
            )

            rows = [
                (fid, dominant, all_hsg)
                for fid, (dominant, all_hsg)
                in layer_results[layer_name].items()
            ]
            conn.executemany(insert_sql, rows)

        conn.commit()

    print(f"Parcel GeoPackage copy written: {parcel_output}")
    print(
        "Spatial parcel table(s) left unchanged. "
        "Created nonspatial HSG table(s): " + ", ".join(created_tables)
    )
    for layer_name in polygon_layers:
        pk_col = _primary_key_column(parcel_output, layer_name)
        print(
            f"  {_hsg_table_name(layer_name)}.{pk_col} -> "
            f"{layer_name}.{pk_col}"
        )


def _prepare_output(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}\n"
                "Use --overwrite to replace it."
            )
        path.unlink()


def main() -> int:
    args = parse_args()

    input_gpkg = args.gpkg.resolve()

    parcel_gpkg = (
        args.parcels.resolve()
        if args.parcels is not None
        else None
    )
    hsg_input = (
        args.hsg_input.resolve()
        if args.hsg_input is not None
        else None
    )

    if args.parcels_output is not None and parcel_gpkg is None:
        raise ValueError(
            "--parcels-output requires --parcels."
        )

    if hsg_input is not None and parcel_gpkg is None:
        raise ValueError(
            "--hsg-input requires --parcels because its purpose is to reuse "
            "an existing HSG layer for parcel annotation."
        )

    reuse_hsg = hsg_input is not None

    if reuse_hsg and args.output is not None:
        raise ValueError(
            "--output cannot be used with --hsg-input because reuse mode "
            "does not create a new whole-domain HSG output."
        )

    output = None
    if not reuse_hsg:
        output = (
            args.output.resolve()
            if args.output is not None
            else input_gpkg.with_name(
                f"{input_gpkg.stem}_ssurgo_hsg.gpkg"
            )
        )

    parcel_output = None
    if parcel_gpkg is not None:
        parcel_output = (
            args.parcels_output.resolve()
            if args.parcels_output is not None
            else parcel_gpkg.with_name(
                f"{parcel_gpkg.stem}_ssurgo_hsg.gpkg"
            )
        )

    if output is not None and output == input_gpkg:
        raise ValueError(
            "SSURGO output must be different from the domain input GeoPackage."
        )

    if parcel_gpkg is not None and parcel_output == parcel_gpkg:
        raise ValueError(
            "Parcel output must be different from the parcel input GeoPackage."
        )

    if (
        parcel_output is not None
        and output is not None
        and parcel_output == output
    ):
        raise ValueError(
            "SSURGO output and parcel output must be different files."
        )

    if hsg_input is not None:
        if hsg_input == input_gpkg:
            # This is technically readable, but almost certainly indicates a
            # mistaken path because the domain input and prior HSG output are
            # expected to be distinct files.
            print(
                "WARNING: --hsg-input is the same file as the domain input.",
                file=sys.stderr,
            )
        if parcel_output is not None and parcel_output == hsg_input:
            raise ValueError(
                "Parcel output must be different from the existing HSG input."
            )

    # In reuse mode, deliberately do not prepare/delete/create the whole-domain
    # HSG output. The supplied HSG file is read-only input.
    if output is not None:
        _prepare_output(output, args.overwrite)
    if parcel_output is not None:
        _prepare_output(parcel_output, args.overwrite)

    print(f"Domain input:   {input_gpkg}")
    if reuse_hsg:
        print(f"Existing HSG:   {hsg_input}")
        print("USDA download:  skipped")
    else:
        print(f"SSURGO output:  {output}")

    if parcel_gpkg is not None:
        print(f"Parcel input:   {parcel_gpkg}")
        print(f"Parcel output:  {parcel_output}")

    domain_native, native_crs, layers = build_domain(input_gpkg)
    print(
        "Polygon layers contributing to domain: "
        + ", ".join(layers)
    )

    domain_wgs84 = (
        gpd.GeoSeries([domain_native], crs=native_crs)
        .to_crs(WGS84)
        .iloc[0]
    )
    domain_wgs84 = polygonal_part(domain_wgs84)

    if domain_wgs84 is None or domain_wgs84.is_empty:
        raise ValueError(
            "Domain became empty after reprojection to WGS84."
        )

    minx, miny, maxx, maxy = domain_wgs84.bounds
    print(
        "Domain WGS84 bounds: "
        f"{minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}"
    )

    if reuse_hsg:
        soil = load_existing_hsg(
            hsg_gpkg=hsg_input,
            domain_wgs84=domain_wgs84,
        )
        print(
            f"Loaded existing HSG polygons intersecting domain: {len(soil):,}"
        )
    else:
        soil = download_ssurgo(
            domain_wgs84=domain_wgs84,
            tile_degrees=args.tile_degrees,
            timeout=args.timeout,
        )
        soil = normalize_wfs_axis_order(soil, domain_wgs84)
        soil = clip_to_domain(soil, domain_wgs84)

        print(f"Clipped SSURGO polygons: {len(soil):,}")

    values = sorted(
        (
            str(v)
            for v in soil["hsg"].dropna().unique()
            if str(v).strip()
        ),
        key=_hsg_sort_key,
    )
    print("HSG values: " + ", ".join(values))

    if not reuse_hsg:
        write_ssurgo_output(
            output=output,
            domain_native=domain_native,
            native_crs=native_crs,
            soil_wgs84=soil,
        )
        print(f"SSURGO output finished: {output}")
        print("Layers written: domain, ssurgo_hsg")

    if parcel_gpkg is not None:
        annotate_parcel_gpkg(
            parcel_gpkg=parcel_gpkg,
            parcel_output=parcel_output,
            soil_wgs84=soil,
            domain_wgs84=domain_wgs84,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
