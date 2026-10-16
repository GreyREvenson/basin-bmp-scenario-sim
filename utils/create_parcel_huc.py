"""Write dominant parcel_huc assignments using local polygons at any HUC level."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.input_scopes import assign_parcels_to_hucs
from utils.create_parcel_huc12 import _read_parcels


def write_parcel_huc(parcel_path: Path, assignments: pd.DataFrame) -> None:
    """Replace only the registered parcel_huc relationship table."""
    with sqlite3.connect(parcel_path) as con:
        con.execute("DROP TABLE IF EXISTS parcel_huc")
        con.execute(
            """CREATE TABLE parcel_huc (
                id INTEGER PRIMARY KEY,
                pid INTEGER NOT NULL UNIQUE,
                huc_id TEXT NOT NULL,
                area_fraction REAL NOT NULL CHECK(area_fraction>=0 AND area_fraction<=1),
                FOREIGN KEY(pid) REFERENCES parcels(pid)
            )"""
        )
        con.executemany(
            "INSERT INTO parcel_huc (pid, huc_id, area_fraction) VALUES (?, ?, ?)",
            [(int(r.pid), str(r.huc_id), float(r.area_fraction)) for r in assignments.itertuples()],
        )
        con.execute("DELETE FROM gpkg_contents WHERE table_name='parcel_huc'")
        con.execute(
            """INSERT INTO gpkg_contents
            (table_name, data_type, identifier, description, last_change, srs_id)
            VALUES ('parcel_huc', 'attributes', 'parcel_huc',
                    'Dominant HUC assignment and parcel overlap fraction',
                    strftime('%Y-%m-%dT%H:%M:%fZ','now'), NULL)"""
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parcels", type=Path)
    parser.add_argument("hucs", type=Path)
    parser.add_argument("--parcel-layer", default="parcels")
    parser.add_argument("--huc-layer", default="hucs")
    parser.add_argument("--huc-field", default="huc_id")
    args = parser.parse_args()
    parcels = _read_parcels(args.parcels, args.parcel_layer)
    hucs = gpd.read_file(args.hucs, layer=args.huc_layer)
    hucs = hucs.rename(columns={args.huc_field: "huc_id"})
    assignments = assign_parcels_to_hucs(parcels, hucs)
    write_parcel_huc(args.parcels, assignments)
    print(f"Wrote {len(assignments)} parcel_huc assignments to {args.parcels}")
    low = assignments[assignments.area_fraction < 0.75]
    if not low.empty:
        print(f"Warning: {len(low)} parcels have less than 75% overlap with their assigned HUC")


if __name__ == "__main__":
    main()
