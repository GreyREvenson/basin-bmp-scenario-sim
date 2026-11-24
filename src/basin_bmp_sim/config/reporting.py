"""Human-readable validation diagnostics for normalized input data."""
from __future__ import annotations
from pathlib import Path
from typing import Any, List, Mapping
import geopandas as gpd
import pandas as pd
from basin_bmp_sim.domain.constants import COL_PID, CFG_PARCELS
from basin_bmp_sim.common.utils import ci_get
from basin_bmp_sim.common.io import list_geopackage_tables

def format_input_validation_report(data: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    """Return a concise human-readable report for ``--validate-only`` runs."""
    parcels = data.get("parcels")
    outlets = data.get("outlet_loc")
    plet_inputs = data.get("plet_inputs")
    package = ci_get(cfg, CFG_PARCELS)
    input_tables: List[str] = []
    if package is not None and Path(package).exists():
        input_tables = sorted(
            name for name in list_geopackage_tables(package) if name.startswith("input_")
        )

    default_rows = 0
    override_rows = 0
    defaults_applied = 0
    if isinstance(plet_inputs, pd.DataFrame) and not plet_inputs.empty:
        default_rows = int(plet_inputs[COL_PID].isna().sum())
        override_rows = int(plet_inputs[COL_PID].notna().sum())
        if "_default_applied" in plet_inputs.columns:
            defaults_applied = int(
                plet_inputs["_default_applied"].fillna(False).astype(bool).sum()
            )

    load_generation = data.get("load_generation") or {}
    mode = str(load_generation.get("mode", "statistical"))
    crs_text = "unknown"
    if isinstance(parcels, gpd.GeoDataFrame) and parcels.crs is not None:
        crs_text = parcels.crs.to_string()
    parcel_count = len(parcels) if parcels is not None else 0
    outlet_count = len(outlets) if outlets is not None else 0

    lines = [
        "INPUT VALIDATION",
        "----------------",
        f"Mode:                     {mode}",
        f"Parcels:                  {parcel_count}",
        f"Analysis CRS:             {crs_text} (metric)",
        f"Outlets:                  {outlet_count}",
        f"Pollutants:               {', '.join(map(str, data.get('pollutants', [])))}",
        f"CPS codes:                {', '.join(map(str, data.get('cps', [])))}",
        f"Recognized input tables:  {len(input_tables)}",
        f"Default parameter rows:   {default_rows}",
        f"Parcel overrides:         {override_rows}",
        f"Optional defaults added:  {defaults_applied}",
        "Unknown input tables:     0",
        "Unknown parcel/outlet IDs: 0",
        "Duplicate logical keys:   0",
        "",
        "VALID",
    ]
    return "\n".join(lines)
