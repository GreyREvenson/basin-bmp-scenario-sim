"""Create stochastic East Fork PLET GeoPackages from the constant-input files.

    python utils/create_plet_vary_inputs.py
    python utils/create_plet_vary_inputs.py --overwrite
    python utils/create_plet_vary_inputs.py --help

Creates, in the corresponding input directories:
    parcels_plet_stochastic.gpkg
    huc12s_plet_stochastic.gpkg

The sources are backed up with SQLite, retaining geometries, spatial indexes,
parcel/HUC12 IDs, relation tables, input-table schemas and metadata. Numeric
``value`` entries selected for uncertainty are replaced with ``min``, ``p10``,
``p50``, ``p90``, ``max``; everything else is left as in the source. The model
samples these as piecewise-linear quantile functions. The p50 is the source
value, and each envelope is bounded to the parameter's physical domain.

IMPORTANT: These are illustrative, expert-judgment *sensitivity* envelopes,
NOT locally calibrated parameter distributions, confidence intervals or
validated interannual weather variability. Widths are explicit below and can
be replaced by field observations, model calibration, or literature-based
parameter estimates. EPA recommends refining initial PLET inputs using local
information: https://www.epa.gov/nps/plet ; USDA RUSLE2 background:
https://www.nrcs.usda.gov/sites/default/files/2022-10/RUSLE2%20Handbook_0.pdf

Scientific/modeling limitations:
* ``input_avg_rain_in`` is derived in PLET from annual precipitation, rainfall
  days and correction fractions. The current stochastic input model samples
  these as *independent* parameters and cannot enforce that algebraic identity
  per realization. Because the model uses avg_rain_in when supplied, variations
  in annual_precip_in / rain_correction_fraction do NOT affect runoff in these
  HUC12-driven runs; they are intentionally left fixed. To vary only independent
  climate inputs, remove HUC12 avg_rain_in rows and model the annual-precip form
  in a separate, deliberately configured experiment.
* HUC12-sourced climate/RUSLE/concentration distributions are shared across
  parcels in the same HUC12 per model scenario. Hydrology-lookup distributions
  are sampled independently for each parcel unless the user adds a group.
* The current sampler does not constrain correlation between curve number and
  infiltration, nor between different climate variables or neighboring HUC12s.
* Zero subsurface and TSS concentrations in the HUC12 source are deliberately
  preserved: they are placeholders/assumptions, not evidence of zero pollution.
* Neutral routing/transport coefficients (all 1), selection weights, fixed
  classes and deliberately zero initial-abstraction factors remain unchanged.

No geospatial libraries or network access are required.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARCELS = REPO_ROOT / "examples/east_fork/inputs/parcels/parcels_plet_deterministic.gpkg"
DEFAULT_HUCS = REPO_ROOT / "examples/east_fork/inputs/plet/huc12s_plet_deterministic.gpkg"
STATISTICS = ("mean", "sd", "min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max")


@dataclass(frozen=True)
class Envelope:
    """Widths are fractions of the original value (not claimed empirical SDs).

    ``inner`` controls p10/p90; ``outer`` controls min/max. These two pairs
    produce a more informative peaked distribution than a uniform min/max.
    """

    inner: float
    outer: float
    floor: float = 0.0
    ceiling: float | None = None
    rationale: str = "illustrative relative uncertainty"


# Heuristic spread fractions, centered on local original PLET reference values.
# Tighter K and rainfall than C/LS and measured nutrient concentrations; wider
# for concentrated/heterogeneous pollutant transport assumptions. No source
# establishes these particular percentiles as calibrated East Fork estimates.
RULES: dict[str, Envelope] = {
    "annual_precip_in": Envelope(.10, .22, rationale="annual precipitation sensitivity"),
    "avg_rain_in": Envelope(.12, .25, rationale="effective event-depth sensitivity"),
    "rain_days": Envelope(.08, .18, floor=1, ceiling=366, rationale="wet-day count sensitivity"),
    "rain_correction_fraction": Envelope(.03, .07, ceiling=1, rationale="rain correction uncertainty"),
    "runoff_day_fraction": Envelope(.12, .28, ceiling=1, rationale="runoff-day fraction uncertainty"),
    "enrichment_ratio": Envelope(.25, .50, rationale="sediment enrichment variability"),
    "rusle_r": Envelope(.10, .23, rationale="rainfall erosivity variation"),
    "rusle_k": Envelope(.12, .27, rationale="soil erodibility and soil mix variation"),
    "rusle_ls": Envelope(.30, .65, rationale="slope length and terrain representation"),
    "rusle_c": Envelope(.45, .80, ceiling=1, rationale="land-cover management heterogeneity"),
    "rusle_p": Envelope(.10, .30, ceiling=1, rationale="conservation management uncertainty"),
    "sediment_n_pct": Envelope(.25, .55, ceiling=100, rationale="sediment nitrogen composition"),
    "sediment_p_pct": Envelope(.25, .55, ceiling=100, rationale="sediment phosphorus composition"),
    "surface_concentration": Envelope(.35, .75, rationale="runoff water-quality heterogeneity"),
    "subsurface_concentration": Envelope(.40, .80, rationale="shallow subsurface nutrient variability"),
}


def _rule_for(table: str, *, location: str, value: float, row: sqlite3.Row) -> Envelope | None:
    name = table.removeprefix("input_")
    if name in {"selection_weight", "land_cover", "hsg", "ia_ratio", "pollutant_load_rate"}:
        return None
    # Spatial routing/delivery coefficients == 1 currently encode no losses.
    # Changing them would introduce an uncalibrated transport model on top of
    # load-generation uncertainty, so those coefficients stay as configured.
    if name in {"ndr_f_to_s", "ndr_s_to_o", "sdr_f_to_s", "sdr_s_to_o", "sediment_delivery_ratio"}:
        return None
    if location == "huc12" and name in {"annual_precip_in", "rain_correction_fraction"}:
        return None  # redundant with the supplied HUC12 AVG_RAIN forcing
    if location == "parcel" and name in {"annual_precip_in", "rain_days", "rain_correction_fraction", "runoff_day_fraction"}:
        # Parcel-wide defaults are overridden by HUC12 forcing in the standard
        # East Fork configuration. Preserve these fallback reference values.
        return None
    if name in {"surface_concentration", "subsurface_concentration"}:
        if value == 0:
            return None  # structural zero / an unfilled placeholder
        if row["pollutant"] == "TSS":
            return None  # placeholder if nonzero values appear unexpectedly
    if name == "rusle_p" and value >= .999999:
        return None  # P=1 is the explicit no-practice default
    if value == 0:
        return None  # preserve deliberate zeros; never invent a positive median
    if name == "curve_number":
        # Model accepts CN in (0,100]. Spread is in *CN points*, not percent.
        # For e.g. CN=90, p10=86, p90=94 and support [82,98].
        return Envelope(4 / value, 8 / value, floor=1, ceiling=100,
                        rationale="CN class/management uncertainty (±4/8 CN points)")
    if name == "infiltration_fraction":
        return Envelope(.20, .42, ceiling=1, rationale="infiltration class uncertainty")
    return RULES.get(name)


def _envelope(center: float, rule: Envelope) -> tuple[float, float, float, float, float]:
    """Return monotonic physically bounded (min,p10,p50,p90,max)."""
    if center < rule.floor or (rule.ceiling is not None and center > rule.ceiling):
        raise ValueError(f"Reference value {center} outside {rule} bounds")
    lo = max(rule.floor, center * (1 - rule.outer))
    p10 = max(rule.floor, center * (1 - rule.inner))
    p90 = center * (1 + rule.inner)
    hi = center * (1 + rule.outer)
    if rule.ceiling is not None:
        p90 = min(p90, rule.ceiling)
        hi = min(hi, rule.ceiling)
    if not lo <= p10 <= center <= p90 <= hi:
        raise ValueError(f"Invalid distribution support around {center}: {(lo, p10, center, p90, hi)}")
    return lo, p10, center, p90, hi


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _populate(connection: sqlite3.Connection, location: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for table in sorted(t for t in _table_names(connection) if t.startswith("input_")):
        cols = {column[1] for column in connection.execute(f'PRAGMA table_info("{table}")')}
        if not {"id", "value", "min", "p10", "p50", "p90", "max"}.issubset(cols):
            continue
        connection.row_factory = sqlite3.Row
        records = connection.execute(f'SELECT * FROM "{table}"').fetchall()
        for row in records:
            raw = row["value"]
            if raw is None:
                continue
            try:
                value = float(raw)
            except (ValueError, TypeError):
                continue  # e.g. land cover or HSG categorical input
            if not math.isfinite(value):
                raise ValueError(f"{table} id={row['id']} has a non-finite value")
            rule = _rule_for(table, location=location, value=value, row=row)
            if rule is None:
                continue
            if any(row[name] is not None for name in STATISTICS):
                raise ValueError(f"{table} id={row['id']} mixes value with existing distribution fields")
            lo, p10, med, p90, hi = _envelope(value, rule)
            if lo == hi:
                continue
            old_notes = (row["notes"] or "").strip() if "notes" in cols else ""
            provenance = ("ILLUSTRATIVE sensitivity only, NOT calibrated quantiles; "
                          f"{rule.rationale}; p50=original constant {value:g}")
            notes = f"{old_notes}; {provenance}" if old_notes else provenance
            connection.execute(
                f'UPDATE "{table}" SET value=NULL, mean=NULL, sd=NULL, '
                'min=?,p05=NULL,p10=?,p25=NULL,p50=?,p75=NULL,p90=?,p95=NULL,max=?,notes=? WHERE id=?',
                (lo, p10, med, p90, hi, notes, row["id"]),
            )
            # The PLET R index is identical for all land covers per HUC12;
            # these should draw one R value per HUC12, not four independent Rs.
            if location == "huc12" and table == "input_rusle_r" and "sample_group" in cols:
                connection.execute(
                    f'UPDATE "{table}" SET sample_group=? WHERE id=?',
                    (f"plet:huc12:{row['huc12']}:rusle_r", row["id"]),
                )
            counts[table] += 1
    return counts


def _create_temp(source: Path, destination: Path, *, location: str) -> tuple[Path, Counter[str]]:
    if not source.is_file():
        raise FileNotFoundError(f"Missing input GeoPackage: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".gpkg", dir=destination.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with closing(sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)) as src:
            with closing(sqlite3.connect(temp)) as out:
                src.backup(out)
                with out:
                    changed = _populate(out, location)
                integrity = out.execute("PRAGMA quick_check").fetchone()[0]
                if integrity != "ok":
                    raise ValueError(f"GeoPackage integrity check failed: {integrity}")
        return temp, changed
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def create_vary_inputs(
    parcel_source: Path = DEFAULT_PARCELS,
    huc_source: Path = DEFAULT_HUCS,
    parcel_output: Path | None = None,
    huc_output: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Counter[str]]:
    """Create both outputs without modifying originals. Raises if outputs exist."""
    parcel_source = Path(parcel_source).resolve()
    huc_source = Path(huc_source).resolve()
    parcel_output = Path(parcel_output or parcel_source.with_name("parcels_plet_stochastic.gpkg")).resolve()
    huc_output = Path(huc_output or huc_source.with_name("huc12s_plet_stochastic.gpkg")).resolve()
    destinations = (parcel_output, huc_output)
    sources = (parcel_source, huc_source)
    if len(set(destinations + sources)) != 4:
        raise ValueError("Input and output paths must be four distinct paths")
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError(f"Output exists (use --overwrite): {[str(p) for p in destinations if p.exists()]}")
    temps: list[Path] = []
    try:
        parcel_temp, parcel_counts = _create_temp(parcel_source, parcel_output, location="parcel")
        temps.append(parcel_temp)
        huc_temp, huc_counts = _create_temp(huc_source, huc_output, location="huc12")
        temps.append(huc_temp)
        # Both inputs are fully prepared before either output is published.
        for temporary, destination in zip(temps, destinations):
            os.replace(temporary, destination)
        return {"parcels": parcel_counts, "huc12s": huc_counts}
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parcel-source", type=Path, default=DEFAULT_PARCELS)
    parser.add_argument("--huc12-source", type=Path, default=DEFAULT_HUCS)
    parser.add_argument("--parcel-output", type=Path)
    parser.add_argument("--huc12-output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing _vary.gpkg outputs")
    args = parser.parse_args()
    results = create_vary_inputs(args.parcel_source, args.huc12_source,
                                 args.parcel_output, args.huc12_output,
                                 overwrite=args.overwrite)
    for location, counts in results.items():
        print(f"{location}: converted {sum(counts.values())} input rows in {len(counts)} tables")
        for table, n in sorted(counts.items()):
            print(f"  {table}: {n}")
    print(f"Parcel output: {args.parcel_output or args.parcel_source.with_name('parcels_plet_stochastic.gpkg')}")
    print(f"HUC12 output: {args.huc12_output or args.huc12_source.with_name('huc12s_plet_stochastic.gpkg')}")


if __name__ == "__main__":
    main()
