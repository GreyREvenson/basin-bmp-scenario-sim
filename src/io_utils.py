"""Filesystem and serialization helpers for model inputs and outputs.

This module deliberately avoids model-specific defaults, aliases, and domain
validation. Defaults and normalization live in :mod:`src.input_config`;
validation policy lives in :mod:`src.input_validation`.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd
import yaml


class MissingInputTableError(ValueError):
    """Raised when a requested GeoPackage attribute table does not exist."""


def list_geopackage_tables(path: Union[str, Path]) -> List[str]:
    """Return ordinary table names present in a GeoPackage/SQLite container."""
    package = Path(path)
    if not package.exists():
        raise FileNotFoundError(f"GeoPackage not found: {package}")
    with sqlite3.connect(package) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    return [str(row[0]) for row in rows]


def read_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Deserialize a YAML configuration file without applying model semantics.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the YAML configuration file.

        Returns
        -------
        Dict[str, Any]
            Deserialized configuration mapping.

        Raises
        ------
        FileNotFoundError
            If the YAML configuration file does not exist.
        ValueError
            If the YAML document root is not a mapping.
        
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Input YAML file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Configuration YAML root must be a mapping")
    return dict(loaded)


def read_csv_table(path: Union[str, Path]) -> pd.DataFrame:
    """Deserialize one CSV file without model-specific normalization.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Deserialized CSV table.
        
    """
    return pd.read_csv(path)


def read_csv_tables(
    paths: Union[str, Path, Sequence[Union[str, Path]]],
) -> List[pd.DataFrame]:
    """Deserialize one or more CSV files, preserving one frame per file.

        Parameters
        ----------
        paths : Union[str, Path, Sequence[Union[str, Path]]]
            One or more input file paths.

        Returns
        -------
        List[pd.DataFrame]
            Deserialized CSV tables in input order.
        
    """
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    return [read_csv_table(item) for item in items]


def read_geodataframe(
    path: Union[str, Path], *, layer: str | None = None, fid_as_index: bool = False
) -> gpd.GeoDataFrame:
    """Deserialize a geospatial vector dataset without domain normalization.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the geospatial vector dataset.
        layer : str or None
            Optional layer/table name for multi-layer containers such as GeoPackage.
        fid_as_index : bool
            When true, preserve the vector driver's feature ID as the dataframe
            index. This is used for the parcels layer because schema v2 names
            the GeoPackage integer feature ID column ``pid``.

        Returns
        -------
        gpd.GeoDataFrame
            Deserialized geospatial dataset.
        
    """
    kwargs = {"fid_as_index": True} if fid_as_index else {}
    return (
        gpd.read_file(path, layer=layer, **kwargs)
        if layer is not None
        else gpd.read_file(path, **kwargs)
    )


def read_geopackage_table(path: Union[str, Path], table: str) -> pd.DataFrame:
    """Read one non-spatial attribute table from a GeoPackage.

    Attribute tables are stored as ordinary SQLite tables inside the GeoPackage.
    Reading them with SQLite avoids imposing geospatial-layer semantics on data
    such as parcel relationships, parameter rows, and outlet statistics.
    """
    package = Path(path)
    if not package.exists():
        raise FileNotFoundError(f"GeoPackage not found: {package}")
    safe_table = str(table).replace('"', '""')
    with sqlite3.connect(package) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (str(table),)
        ).fetchone()
        if exists is None:
            raise MissingInputTableError(
                f"GeoPackage table not found: {table} in {package}"
            )
        return pd.read_sql_query(f'SELECT * FROM "{safe_table}"', conn)




def read_geopackage_integer_ids(
    path: Union[str, Path], table: str, id_column: str
) -> List[int]:
    """Read an INTEGER PRIMARY KEY column from a GeoPackage in feature-ID order.

    GeoPackage/OGR drivers commonly expose an INTEGER PRIMARY KEY as the feature
    ID rather than as an ordinary attribute.  This helper reads that identifier
    directly from SQLite so callers can recover it when a geospatial reader hides
    the field.
    """
    package = Path(path)
    if not package.exists():
        raise FileNotFoundError(f"GeoPackage not found: {package}")
    safe_table = str(table).replace('"', '""')
    safe_column = str(id_column).replace('"', '""')
    with sqlite3.connect(package) as conn:
        info = conn.execute(f'PRAGMA table_info("{safe_table}")').fetchall()
        if not info:
            raise MissingInputTableError(
                f"GeoPackage table not found: {table} in {package}"
            )
        match = next(
            (row for row in info if str(row[1]).lower() == str(id_column).lower()),
            None,
        )
        if match is None:
            raise ValueError(
                f"GeoPackage table '{table}' does not contain identifier column '{id_column}'"
            )
        declared_type = str(match[2] or "").upper()
        if int(match[5]) != 1 or "INT" not in declared_type:
            raise ValueError(
                f"GeoPackage table '{table}' column '{id_column}' must be an INTEGER PRIMARY KEY"
            )
        rows = conn.execute(
            f'SELECT "{safe_column}" FROM "{safe_table}" '
            f'ORDER BY "{safe_column}"'
        ).fetchall()
    return [int(row[0]) for row in rows]


def read_geopackage_table_info(path: Union[str, Path], table: str) -> List[Dict[str, Any]]:
    """Return SQLite column metadata for one GeoPackage table.

    Each row contains ``name``, declared ``type``, ``notnull``, ``default``,
    and ``pk`` fields from ``PRAGMA table_info``.
    """
    package = Path(path)
    if not package.exists():
        raise FileNotFoundError(f"GeoPackage not found: {package}")
    safe_table = str(table).replace('"', '""')
    with sqlite3.connect(package) as conn:
        rows = conn.execute(f'PRAGMA table_info("{safe_table}")').fetchall()
    if not rows:
        raise MissingInputTableError(f"GeoPackage table not found: {table} in {package}")
    return [
        {
            "cid": int(row[0]),
            "name": str(row[1]),
            "type": str(row[2] or ""),
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": int(row[5]),
        }
        for row in rows
    ]


def read_parquet_table(path: Union[str, Path]) -> pd.DataFrame:
    """Deserialize a Parquet table without model-specific validation.

        Parameters
        ----------
        path : Union[str, Path]
            Path to the Parquet file.

        Returns
        -------
        pd.DataFrame
            Deserialized Parquet table.
        
    """
    return pd.read_parquet(Path(path))


def _write_parquet_atomic(
    df: pd.DataFrame, path: Path, *, logger: logging.Logger
) -> None:
    """Write a parquet file atomically.

        Parameters
        ----------
        df : pd.DataFrame
            Table to serialize.
        path : Path
            Destination path for the Parquet file.
        logger : logging.Logger
            Logger used for diagnostic and progress messages.

        Raises
        ------
        RuntimeError
            If no supported Parquet engine is installed.
        
    """
    del logger
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    except (ImportError, ModuleNotFoundError) as ex:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            "Parquet output is required but no parquet engine is installed. "
            "Install pyarrow or fastparquet in the active environment."
        ) from ex


def _flatten_plot_records(
    merged: Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
) -> pd.DataFrame:
    """Convert plot records into the normalized serialized trajectory table.

        Parameters
        ----------
        merged : Dict[Tuple[str, str, str, str], List[Tuple[int, float, float]]]
            Merged trajectory records keyed by pollutant, outlet, and axis definitions.

        Returns
        -------
        pd.DataFrame
            Normalized trajectory table suitable for serialization.
        
    """
    rows: List[Dict[str, Any]] = []
    step_counters: Dict[Tuple[int, str, str, str, str], int] = defaultdict(int)
    for (pol, oid, xax, yax), points in merged.items():
        for sid, xval, yval in points:
            counter_key = (int(sid), str(pol), str(oid), str(xax), str(yax))
            step_counters[counter_key] += 1
            rows.append(
                {
                    "scenario": int(sid),
                    "pollutant": str(pol),
                    "oid": str(oid),
                    "x_axis": str(xax),
                    "y_axis": str(yax),
                    "step": int(step_counters[counter_key]),
                    "x_value": float(xval),
                    "y_value": float(yval),
                }
            )
    columns = [
        "scenario", "pollutant", "oid", "x_axis", "y_axis",
        "step", "x_value", "y_value",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["scenario", "pollutant", "oid", "x_axis", "y_axis", "step"]
    ).reset_index(drop=True)
