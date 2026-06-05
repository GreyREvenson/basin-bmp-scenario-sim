#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Determine upgradient parcels from a DEM using a D8 flow network. "
                    "Outputs a CSV with columns: pid, pid_up (comma-separated)."
    )
    parser.add_argument("parcels", help="Path to parcel boundaries (GeoPackage/GeoJSON/Shapefile). Must have an ID field.")
    parser.add_argument("dem", help="Path to DEM raster (GeoTIFF).")
    parser.add_argument("--id-field", default="pid", help="Name of the parcel ID field (default: pid).")
    parser.add_argument("--out", default=None, help="Path to CSV output. Default: <parcels_basename>_upgradient.csv")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-filled-dem", action="store_true",
                        help="If richdem is installed, fill depressions before computing flow directions.")
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def read_parcels(parcels_path: str, id_field: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(parcels_path)
    if id_field not in gdf.columns:
        raise ValueError(f"Parcel ID field '{id_field}' not found in {parcels_path}. Columns: {list(gdf.columns)}")
    # Drop null/empty geometries
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise ValueError("No valid parcel geometries found.")
    # Attempt to fix invalid geometries
    if not gdf.geometry.is_valid.all():
        logging.info("Fixing invalid parcel geometries with buffer(0)")
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def reproject_to_match(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Parcels have no CRS. Please define a CRS before running.")
    if target_crs is None:
        raise ValueError("DEM has no CRS. DEM must have a valid CRS.")
    if gdf.crs != target_crs:
        logging.info(f"Reprojecting parcels from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf


def rasterize_parcels(
    gdf: gpd.GeoDataFrame,
    id_field: str,
    out_shape: Tuple[int, int],
    transform
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, int]]:
    """
    Rasterize parcels to match DEM grid.
    Returns:
      - pid_raster: int array where each cell is code for parcel id, 0 for background
      - code_to_pid: dict mapping int code -> original pid (string repr)
      - pid_to_code: dict mapping original pid (string repr) -> int code
    """
    pids = gdf[id_field].astype(str).tolist()
    unique_pids = list(dict.fromkeys(pids))  # preserve order as in gdf
    pid_to_code = {pid: (i + 1) for i, pid in enumerate(unique_pids)}  # 0 is background
    code_to_pid = {v: k for k, v in pid_to_code.items()}

    shapes = ((geom, pid_to_code[str(pid)]) for geom, pid in zip(gdf.geometry, gdf[id_field]))
    pid_raster = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.int32
    )
    return pid_raster, code_to_pid, pid_to_code


def maybe_fill_depressions(dem_arr: np.ndarray, use_fill: bool) -> np.ndarray:
    """
    If use_fill and richdem is available, fill depressions.
    Otherwise, return the input DEM unchanged.
    """
    if not use_fill:
        return dem_arr
    try:
        import richdem as rd
    except ImportError:
        logging.warning("richdem not installed; proceeding without depression filling.")
        return dem_arr

    logging.info("Filling depressions in DEM (richdem)...")
    dem_rd = rd.rdarray(dem_arr.copy(), no_data=np.nan)
    rd.FillDepressions(dem_rd, in_place=True)
    return np.asarray(dem_rd, dtype=np.float32)


def compute_d8_pointers(dem: np.ndarray, transform) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute D8 flow pointers (steepest descent) for each cell.
    Returns (next_row, next_col) arrays of shape dem.shape. Cells with no valid downslope neighbor
    will point to the lowest neighbor (to break flats). Nodata cells have next_row/next_col = -1.
    """
    nrows, ncols = dem.shape
    next_r = np.full(dem.shape, -1, dtype=np.int32)
    next_c = np.full(dem.shape, -1, dtype=np.int32)

    # Pixel sizes (assumes north-up affine)
    dx = abs(transform.a)
    dy = abs(transform.e)
    diag = (dx**2 + dy**2) ** 0.5

    # Neighbor offsets and their distances
    neighbors = [
        (-1, -1, diag), (-1, 0, dy), (-1, 1, diag),
        (0, -1, dx),               (0, 1, dx),
        (1, -1, diag),  (1, 0, dy), (1, 1, diag),
    ]

    # Precompute nodata mask (NaN indicates nodata)
    nodata = np.isnan(dem)

    logging.info("Computing D8 flow pointers (steepest descent)...")
    for r in range(nrows):
        for c in range(ncols):
            if nodata[r, c]:
                continue
            z0 = dem[r, c]
            best_drop = -np.inf
            best_rc = None

            # Track minimal neighbor elevation to break flats if no descent found
            min_z = np.inf
            min_rc = None

            for dr, dc, dist in neighbors:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                    continue
                if nodata[rr, cc]:
                    continue
                zn = dem[rr, cc]
                # Track minimal neighbor
                if zn < min_z:
                    min_z = zn
                    min_rc = (rr, cc)
                # Steepest descent
                drop = (z0 - zn) / dist
                if drop > best_drop:
                    best_drop = drop
                    best_rc = (rr, cc)

            if best_rc is not None and best_drop > 0:
                next_r[r, c], next_c[r, c] = best_rc
            elif min_rc is not None:
                # No lower neighbor; move toward the lowest neighbor (flat resolution heuristic)
                next_r[r, c], next_c[r, c] = min_rc
            # else remain -1 for isolated cells surrounded by nodata

    return next_r, next_c


def pick_pour_point(dem_arr: np.ndarray, mask: np.ndarray) -> Tuple[int, int]:
    """
    Pick the pour point as the lowest-elevation cell within the parcel's cells.
    Returns (row, col). Raises if no valid cell is found.
    """
    candidate_mask = mask & (~np.isnan(dem_arr))
    if not np.any(candidate_mask):
        raise ValueError("Parcel has no valid DEM cells after masking/nodata.")
    rows, cols = np.where(candidate_mask)
    vals = dem_arr[rows, cols]
    min_idx = np.argmin(vals)
    return int(rows[min_idx]), int(cols[min_idx])


def delineate_upgradient_from_pointers(
    pour_row: int,
    pour_col: int,
    next_r: np.ndarray,
    next_c: np.ndarray,
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Delineate upstream catchment using D8 pointers:
    Include all cells whose next pointer leads to the pour point.
    """
    nrows, ncols = next_r.shape
    visited = np.zeros(next_r.shape, dtype=bool)
    # Queue for BFS
    from collections import deque
    q = deque()
    q.append((pour_row, pour_col))
    visited[pour_row, pour_col] = True

    # Neighbor offsets for scanning "who flows into me"
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    while q:
        r, c = q.popleft()
        for dr, dc in neighbors:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue
            if not valid_mask[rr, cc] or visited[rr, cc]:
                continue
            # Does neighbor flow into (r, c)?
            if next_r[rr, cc] == r and next_c[rr, cc] == c:
                visited[rr, cc] = True
                q.append((rr, cc))

    return visited


def main():
    args = parse_args()
    setup_logging(args.log_level)

    if args.out is None:
        base = os.path.splitext(os.path.basename(args.parcels))[0]
        args.out = f"{base}_upgradient.csv"

    # Read DEM
    logging.info(f"Reading DEM: {args.dem}")
    with rasterio.open(args.dem) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        dem_crs = src.crs
        dem_nodata = src.nodata
        dem_shape = (src.height, src.width)

    # Normalize nodata -> NaN
    if dem_nodata is not None:
        dem = np.where(dem == dem_nodata, np.nan, dem)

    # Optional depression filling
    dem_for_flow = maybe_fill_depressions(dem, args.use_filled_dem)

    # Compute D8 pointers
    next_r, next_c = compute_d8_pointers(dem_for_flow, transform)

    # Valid cell mask (not NaN)
    valid_mask = ~np.isnan(dem_for_flow)

    # Read parcels and reproject
    logging.info(f"Reading parcels: {args.parcels}")
    gdf = read_parcels(args.parcels, args.id_field)
    gdf = reproject_to_match(gdf, dem_crs)

    # Rasterize parcels to DEM grid
    logging.info("Rasterizing parcels to DEM grid...")
    pid_raster, code_to_pid, pid_to_code = rasterize_parcels(
        gdf, args.id_field, out_shape=dem_shape, transform=transform
    )

    ordered_codes = [pid_to_code[str(pid)] for pid in gdf[args.id_field].astype(str)]

    results: List[Tuple[str, List[str]]] = []
    total = len(ordered_codes)
    logging.info(f"Processing {total} parcels to determine upstream parcels...")

    for idx, code in enumerate(ordered_codes, start=1):
        parcel_mask = (pid_raster == code)
        if not np.any(parcel_mask):
            logging.warning(f"Parcel {code_to_pid[code]} has no DEM-covered cells; recording empty upstream.")
            results.append((code_to_pid[code], []))
            continue

        # Pick pour point (lowest cell within parcel)
        try:
            pr, pc = pick_pour_point(dem_for_flow, parcel_mask)
        except ValueError as e:
            logging.warning(f"Parcel {code_to_pid[code]}: {e}; recording empty upstream.")
            results.append((code_to_pid[code], []))
            continue

        # Delineate upstream catchment via BFS on D8 pointers
        catch_mask = delineate_upgradient_from_pointers(pr, pc, next_r, next_c, valid_mask)

        # Find upstream parcels intersecting the catchment
        upstream_codes = np.unique(pid_raster[catch_mask])
        upstream_codes = upstream_codes[(upstream_codes != 0) & (upstream_codes != code)]
        upstream_pids = [code_to_pid[c] for c in upstream_codes]

        results.append((code_to_pid[code], upstream_pids))

        if idx % 10 == 0 or idx == total:
            logging.info(f"Processed {idx}/{total} parcels.")

    # Write CSV output: pid, pid_up (comma-separated string, empty if none)
    logging.info(f"Writing results to {args.out}")
    import csv
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pid', 'pid_up'])
        for pid, upstream_list in results:
            pid_up_str = ",".join(map(str, upstream_list)) if upstream_list else ""
            writer.writerow([pid, pid_up_str])

    logging.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)