"""Animal nutrient excretion rates and production inventories."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
from basin_bmp_sim.common.distributions import sample_group_key, stats_from_row
from basin_bmp_sim.domain.statistics_validation import validate_named_parameter_stats, validate_stats_rows, validate_unique_rows
from basin_bmp_sim.common.io import read_csv_table
from basin_bmp_sim.common.utils import normalize_columns

_EXCRETION_PARAMETERS = (
    "representative_mass_kg", "tn_kg_1000kg_day", "tp_kg_1000kg_day",
)

from .shared import _animal_species, _need_columns, _finite, _draw, _sample_stats


def reference_excretion_factors(
    path: str | Path | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load native D384.1 rate statistics and required representative masses."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "afo" / "manure_excretion_reference.csv"
    label = f"manure excretion factors ({path})"
    table = normalize_columns(read_csv_table(path))
    _need_columns(table, label, ["species", "parameter"])
    if table.empty:
        raise ValueError(f"{label} requires at least one species")
    for column in ("species", "parameter"):
        if table[column].isna().any() or table[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{label}.{column} must not be blank")
        table[column] = table[column].astype(str).str.strip().str.lower()
    table["species"] = table["species"].map(_animal_species)
    unexpected = set(table["parameter"]) - set(_EXCRETION_PARAMETERS)
    if unexpected:
        raise ValueError(f"{label} contains unsupported parameters: {sorted(unexpected)}")
    validate_unique_rows(table, ["species", "parameter"], label)
    validate_stats_rows(table, label)
    factors = {}
    for species, rows in table.groupby("species", sort=False):
        missing = set(_EXCRETION_PARAMETERS) - set(rows["parameter"])
        if missing:
            raise ValueError(f"{label} species {species!r} is missing parameters: {sorted(missing)}")
        factors[species] = {}
        for _, row in rows.iterrows():
            stats = stats_from_row(row)
            group = row.get("sample_group")
            record: dict[str, Any] = dict(stats)
            if group is not None and pd.notna(group) and str(group).strip():
                record["sample_group"] = str(group).strip()
            factors[species][row["parameter"]] = record
    return factors


def _excretion_override_stats(animal: Mapping[str, Any], parameter: str) -> dict[str, float]:
    """Extract and validate an animal-row override without merging reference rates."""
    return validate_named_parameter_stats(
        animal, parameter, f"afo_animals[{animal.get('afo_id', '?')}/{animal.get('species', '?')}]",
        low=0., exclusive_low=(parameter == "representative_mass_kg"),
    )


def animal_production(
    ctx: Any, animals: list[dict[str, Any]], op_id: str,
    *, excretion_cache: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    """Calculate as-excreted annual nutrient N/P inventories.

    Availability/handling losses are applied separately; animal-days is the
    number of days each enumerated head is present in the modeled year.
    """
    factors = ctx.afo_inputs["excretion_factors"]
    if excretion_cache is None:
        excretion_cache = {}
    totals = {"TN": 0.0, "TP": 0.0}
    for animal in animals:
        if str(animal["afo_id"]) != op_id:
            continue
        species = _animal_species(animal["species"])
        known = factors.get(species, {})
        number = _draw(ctx, animal, "count")
        days = _draw(ctx, animal, "animal_days", default=365., maximum=366.)
        if days > 366:
            raise ValueError(f"{op_id}: animal_days cannot exceed 366")
        realized = {}
        for field in _EXCRETION_PARAMETERS:
            kind = "positive" if field == "representative_mass_kg" else "nonnegative"
            stats = _excretion_override_stats(animal, field)
            if stats:
                rate = float(ctx._sample_from_stats(stats, kind=kind))
            else:
                if field not in known:
                    raise ValueError(f"{op_id}/{species}: missing manure excretion parameter {field}")
                reference = known[field]
                key = sample_group_key(reference, pid=repr((op_id, species)), variable=field)
                if key not in excretion_cache:
                    excretion_cache[key] = _sample_stats(ctx, reference, kind=kind)
                rate = excretion_cache[key]
            rate = _finite(rate, f"{op_id}/{species}.{field}")
            if field == "representative_mass_kg" and rate <= 0:
                raise ValueError(f"{op_id}/{species}.representative_mass_kg must be positive")
            realized[field] = rate
        scale = number * days * realized["representative_mass_kg"] / 1000.
        for nutrient, field in zip(("TN", "TP"), _EXCRETION_PARAMETERS[1:]):
            totals[nutrient] += scale * realized[field]
    return totals
