"""Assign each parcel to its dominant HUC12 by polygon intersection.

The utility writes/updates the non-spatial ``parcel_huc12`` table inside the
parcel GeoPackage. Each parcel receives exactly one HUC12: the polygon covering
the greatest share of the parcel's area. ``area_fraction`` records that share
for QA at HUC boundaries.

Example
-------
python utils/create_parcel_huc12.py \
    examples/east_fork/inputs/parcels/parcels.gpkg \
    path/to/wbd_huc12.gpkg --huc12-layer WBDHU12
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

PARCEL_LAYER = "parcels"
OUTPUT_TABLE = "parcel_huc12"


def _normalize_huc12(value: object) -> str:
    if value is None or pd.isna(value):
        raise ValueError("HUC12 values must not be null")
    text = str(value).strip()
    if not (len(text) == 12 and text.isdigit()):
        raise ValueError(
            f"Invalid HUC12 {value!r}; HUC12 must be stored as 12-digit text"
        )
    return text


def _read_parcels(path: Path, layer: str) -> gpd.GeoDataFrame:
    parcels = gpd.read_file(path, layer=layer, fid_as_index=True)
    if parcels.empty:
        raise ValueError(f"Parcel layer {layer!r} is empty")
    if parcels.crs is None:
        raise ValueError("Parcel layer must declare a CRS")
    if parcels.geometry.isna().any() or parcels.geometry.is_empty.any():
        raise ValueError("Parcel layer contains null or empty geometry")
    if (~parcels.geometry.is_valid).any():
        raise ValueError("Parcel layer contains invalid geometry")

    # Schema-v3 uses pid itself as the GeoPackage INTEGER PRIMARY KEY. OGR
    # exposes that key as the feature ID, so fid_as_index=True recovers it.
    pids = pd.to_numeric(pd.Index(parcels.index), errors="raise")
    if not np.all(np.isfinite(pids)) or not np.all(np.equal(pids, np.floor(pids))):
        raise ValueError("Parcel feature IDs must be finite integers (schema-v3 pid)")
    parcels = parcels.copy()
    parcels["pid"] = pids.astype("int64")
    if parcels["pid"].duplicated().any():
        raise ValueError("Parcel pid values must be unique")
    return parcels[["pid", "geometry"]]


def assign_parcels_to_huc12(
    parcel_path: Path,
    huc_path: Path,
    *,
    parcel_layer: str = PARCEL_LAYER,
    huc_layer: Optional[str] = None,
    huc12_field: str = "huc12",
) -> pd.DataFrame:
    """Return dominant-HUC12 assignments and overlap fractions."""
    parcels = _read_parcels(parcel_path, parcel_layer)
    hucs = gpd.read_file(huc_path, layer=huc_layer) if huc_layer else gpd.read_file(huc_path)
    if hucs.empty:
        raise ValueError("HUC12 polygon input is empty")
    if hucs.crs is None:
        raise ValueError("HUC12 polygon input must declare a CRS")

    normalized_columns = {str(c).lower(): c for c in hucs.columns}
    source_field = normalized_columns.get(huc12_field.lower())
    if source_field is None:
        raise ValueError(
            f"HUC12 polygon input does not contain field {huc12_field!r}; "
            f"available fields: {list(hucs.columns)}"
        )
    hucs = hucs[[source_field, "geometry"]].rename(columns={source_field: "huc12"}).copy()
    hucs["huc12"] = hucs["huc12"].map(_normalize_huc12)
    if hucs["huc12"].duplicated().any():
        # Dissolve permits tiled/source datasets that repeat a HUC12 while
        # preserving the logical one-polygon-per-HUC relationship.
        hucs = hucs.dissolve(by="huc12", as_index=False)
    if hucs.geometry.isna().any() or hucs.geometry.is_empty.any():
        raise ValueError("HUC12 polygon input contains null or empty geometry")
    if (~hucs.geometry.is_valid).any():
        raise ValueError("HUC12 polygon input contains invalid geometry")

    hucs = hucs.to_crs(parcels.crs)
    analysis_crs = parcels.estimate_utm_crs()
    if analysis_crs is None:
        raise ValueError("Could not determine a projected CRS for area calculations")
    parcels_m = parcels.to_crs(analysis_crs)
    hucs_m = hucs.to_crs(analysis_crs)
    parcel_area = parcels_m.set_index("pid").geometry.area

    intersections = gpd.overlay(
        parcels_m[["pid", "geometry"]],
        hucs_m[["huc12", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if intersections.empty:
        raise ValueError("No parcel/HUC12 intersections were found")
    intersections["intersection_area"] = intersections.geometry.area
    intersections = intersections[intersections["intersection_area"] > 0].copy()

    dominant = (
        intersections.sort_values(
            ["pid", "intersection_area", "huc12"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates(subset=["pid"], keep="first")
        .copy()
    )
    dominant["area_fraction"] = dominant.apply(
        lambda row: float(row["intersection_area"] / parcel_area.loc[int(row["pid"])]),
        axis=1,
    )
    dominant["area_fraction"] = dominant["area_fraction"].clip(lower=0.0, upper=1.0)
    out = dominant[["pid", "huc12", "area_fraction"]].sort_values("pid").reset_index(drop=True)

    missing = sorted(set(parcels_m["pid"].astype(int)) - set(out["pid"].astype(int)))
    if missing:
        raise ValueError(
            "Some parcels do not intersect any supplied HUC12 polygon; "
            f"missing pid values: {missing[:20]}"
        )
    return out


def write_parcel_huc12(parcel_path: Path, assignments: pd.DataFrame) -> None:
    """Replace ``parcel_huc12`` in an existing GeoPackage and register it."""
    with sqlite3.connect(parcel_path) as con:
        con.execute(f'DROP TABLE IF EXISTS "{OUTPUT_TABLE}"')
        con.execute(
            f'''CREATE TABLE "{OUTPUT_TABLE}" (
                id INTEGER PRIMARY KEY,
                pid INTEGER NOT NULL UNIQUE,
                huc12 TEXT NOT NULL CHECK(length(huc12)=12),
                area_fraction REAL NOT NULL CHECK(area_fraction>=0 AND area_fraction<=1),
                FOREIGN KEY(pid) REFERENCES parcels(pid)
            )'''
        )
        con.executemany(
            f'INSERT INTO "{OUTPUT_TABLE}" (pid, huc12, area_fraction) VALUES (?, ?, ?)',
            [
                (int(row.pid), str(row.huc12), float(row.area_fraction))
                for row in assignments.itertuples(index=False)
            ],
        )
        # GeoPackage attribute-table registration makes the table visible and
        # editable in GIS applications such as QGIS.
        con.execute("DELETE FROM gpkg_contents WHERE table_name=?", (OUTPUT_TABLE,))
        con.execute(
            """INSERT INTO gpkg_contents
               (table_name, data_type, identifier, description, last_change, srs_id)
               VALUES (?, 'attributes', ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'), NULL)""",
            (
                OUTPUT_TABLE,
                OUTPUT_TABLE,
                "Dominant HUC12 assignment for each parcel; area_fraction is the parcel share in the assigned HUC12",
            ),
        )
        con.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parcels", type=Path, help="Parcel GeoPackage to update")
    parser.add_argument("huc12_polygons", type=Path, help="Polygon dataset containing HUC12 boundaries")
    parser.add_argument("--parcel-layer", default=PARCEL_LAYER)
    parser.add_argument("--huc12-layer", default=None)
    parser.add_argument("--huc12-field", default="huc12")
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=0.75,
        help="Report parcels whose dominant HUC covers less than this fraction (default: 0.75)",
    )
    args = parser.parse_args()

    if not args.parcels.exists():
        parser.error(f"Parcel GeoPackage not found: {args.parcels}")
    if not args.huc12_polygons.exists():
        parser.error(f"HUC12 polygon dataset not found: {args.huc12_polygons}")
    if not 0 <= args.warning_threshold <= 1:
        parser.error("--warning-threshold must be between 0 and 1")

    assignments = assign_parcels_to_huc12(
        args.parcels,
        args.huc12_polygons,
        parcel_layer=args.parcel_layer,
        huc_layer=args.huc12_layer,
        huc12_field=args.huc12_field,
    )
    write_parcel_huc12(args.parcels, assignments)

    low = assignments[assignments["area_fraction"] < args.warning_threshold]
    print(f"Wrote {len(assignments):,} parcel assignments to {args.parcels}:{OUTPUT_TABLE}")
    print(f"HUC12s represented: {assignments['huc12'].nunique():,}")
    if not low.empty:
        print(
            f"Warning: {len(low):,} parcel(s) have dominant-HUC area_fraction "
            f"below {args.warning_threshold:.2f}."
        )
        print(low.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
