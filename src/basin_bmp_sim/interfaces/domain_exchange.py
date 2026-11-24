"""Versioned, numeric parcel-domain transfer for live BMI scenario generation.

Arrays are indexed by the ordered parcel identity manifest (Phase 7).  The
reverse upstream adjacency is serialized in CSR form; it is *not* a raster
or a physical BMI grid.  Only data required for placement/design are sent;
the scientific model retains load generation, AFO physics and outlet routing.
"""
from __future__ import annotations

import numpy as np

DOMAIN_SCHEMA_VERSION = 1
DOMAIN_FIELDS = (
    "domain__schema_version",
    "domain__parcel_area_ha",
    "domain__parcel_perimeter_m",
    "domain__upstream_offset",
    "domain__upstream_index",
    "domain__selection_source_index",
    "domain__selection_probability",
)


def export_placement_domain(ctx) -> dict[str, np.ndarray]:
    offsets = [0]
    upstream: list[int] = []
    for upstream_ids in ctx.parcel_up_idxs:
        upstream.extend(int(v) for v in upstream_ids)
        offsets.append(len(upstream))
    return {
        "domain__schema_version": np.array([DOMAIN_SCHEMA_VERSION], dtype=np.int64),
        "domain__parcel_area_ha": np.asarray(ctx.parcel_area_ha, dtype=np.float64).copy(),
        "domain__parcel_perimeter_m": np.asarray(ctx.parcel_perim_m, dtype=np.float64).copy(),
        "domain__upstream_offset": np.asarray(offsets, dtype=np.int64),
        "domain__upstream_index": np.asarray(upstream, dtype=np.int64),
        "domain__selection_source_index": np.asarray(ctx.selection_source_idxs, dtype=np.int64).copy(),
        "domain__selection_probability": np.asarray(ctx.parcel_selection_probs, dtype=np.float64).copy(),
    }


def validate_placement_domain(values, *, n_parcels: int, n_selection: int) -> dict[str, np.ndarray]:
    """Validate the whole packet before modifying a consumer's domain."""
    missing = set(DOMAIN_FIELDS) - set(values)
    if missing:
        raise ValueError(f"Incomplete BMI parcel domain: {sorted(missing)}")
    data = {name: np.asarray(values[name]).copy() for name in DOMAIN_FIELDS}
    version = data["domain__schema_version"]
    if version.dtype != np.int64 or version.shape != (1,) or version[0] != DOMAIN_SCHEMA_VERSION:
        raise ValueError("Unsupported BMI parcel domain schema version")
    for name in ("domain__parcel_area_ha", "domain__parcel_perimeter_m"):
        x = data[name]
        if x.dtype != np.float64 or x.shape != (n_parcels,) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError(f"Invalid BMI parcel domain {name} geometry")
    for name in ("domain__upstream_offset", "domain__upstream_index", "domain__selection_source_index"):
        if data[name].dtype != np.int64 or data[name].ndim != 1:
            raise ValueError(f"Invalid BMI parcel domain {name} index type")
    offsets, upstream = data["domain__upstream_offset"], data["domain__upstream_index"]
    if (offsets.shape != (n_parcels + 1,) or offsets[0] != 0 or
            offsets[-1] != upstream.size or np.any(np.diff(offsets) < 0) or
            np.any(upstream < 0) or np.any(upstream >= n_parcels)):
        raise ValueError("Invalid BMI parcel domain upstream connectivity")
    # Source topology must be acyclic: each row lists *immediate* upstream
    # parcels. Kahn's algorithm makes this check O(parcels + connections).
    in_degree = np.zeros(n_parcels, dtype=np.int64)
    for idx in range(n_parcels):
        neighbors = upstream[offsets[idx]:offsets[idx + 1]]
        if idx in neighbors or np.unique(neighbors).size != neighbors.size:
            raise ValueError("Invalid BMI parcel domain upstream self-link or duplicate")
        np.add.at(in_degree, neighbors, 1)
    ready = [int(i) for i in np.flatnonzero(in_degree == 0)]
    visited = 0
    while ready:
        idx = ready.pop()
        visited += 1
        for upstream_idx in upstream[offsets[idx]:offsets[idx + 1]]:
            in_degree[upstream_idx] -= 1
            if in_degree[upstream_idx] == 0:
                ready.append(int(upstream_idx))
    if visited != n_parcels:
        raise ValueError("Invalid BMI parcel domain upstream graph contains a cycle")
    selection = data["domain__selection_source_index"]
    probabilities = data["domain__selection_probability"]
    if (selection.shape != (n_selection,) or np.any(selection < 0) or
            np.any(selection >= n_parcels) or np.unique(selection).size != n_selection):
        raise ValueError("Invalid BMI parcel domain selection indices")
    if (probabilities.dtype != np.float64 or probabilities.shape != (n_selection,) or
            not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
            not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-8)):
        raise ValueError("Invalid BMI parcel domain selection probabilities")
    return data


def install_placement_domain(ctx, values) -> None:
    """Atomically replace the generator's ordered placement-domain arrays."""
    data = validate_placement_domain(values, n_parcels=len(ctx.parcel_ids),
                                     n_selection=len(ctx.parcel_selection_ids))
    if any(str(ctx.parcel_selection_ids[i]) != str(ctx.parcel_ids[int(idx)])
           for i, idx in enumerate(data["domain__selection_source_index"])):
        raise ValueError("BMI parcel domain selection IDs disagree with transferred indices")
    offsets, upstream = data["domain__upstream_offset"], data["domain__upstream_index"]
    ctx.parcel_area_ha = data["domain__parcel_area_ha"]
    ctx.parcel_perim_m = data["domain__parcel_perimeter_m"]
    ctx.parcel_up_idxs = [list(map(int, upstream[offsets[i]:offsets[i + 1]]))
                          for i in range(len(ctx.parcel_ids))]
    ctx.selection_source_idxs = data["domain__selection_source_index"]
    ctx.parcel_selection_probs = data["domain__selection_probability"]
