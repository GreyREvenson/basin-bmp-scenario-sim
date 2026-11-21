"""Explicit BMI coupling identity for ordered *index domains*.

No geographic coordinates, connectivity graph, or scientific equation is
inferred from these identifiers. Both BMI components must agree on the exact
order of parcels, outlets, pollutants, pathways and approved BMP codes before
any integer index or flattened load array is transferred.

The byte vectors are BMI-compatible sidecars: schema-versioned UTF-8 JSON and
its SHA-256 digest, not CSDMS Standard Names or geospatial grids.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from src.interfaces.exchange_variables import EXCHANGE_SCHEMA_VERSION


DOMAIN_INDEX_SCHEMA_VERSION = 1


def index_manifest(context) -> bytes:
    """Serialize the index ordering shared by a live generator and model."""
    # Prepared-model test doubles expose a validated shared payload rather
    # than duplicating every attribute on the outer driver object.
    required = ("parcel_ids", "outlet_oids", "pollutants", "pathway_names", "bmp_cps")
    shared = (context._shared_payload()
              if any(not hasattr(context, key) for key in required)
              else {})
    def field(key):
        return getattr(context, key) if hasattr(context, key) else shared[key]

    manifest = {
        "schema_version": DOMAIN_INDEX_SCHEMA_VERSION,
        "exchange_schema_version": EXCHANGE_SCHEMA_VERSION,
        "parcel_ids": [str(x) for x in field("parcel_ids")],
        "outlet_oids": [str(x) for x in field("outlet_oids")],
        "pollutants": [str(x) for x in field("pollutants")],
        "pathway_names": [str(x) for x in field("pathway_names")],
        "bmp_cps": [int(x) for x in field("bmp_cps")],
    }
    if len(set(manifest["parcel_ids"])) != len(manifest["parcel_ids"]):
        raise ValueError("BMI index contract contains duplicate parcel identifiers")
    if len(set(manifest["outlet_oids"])) != len(manifest["outlet_oids"]):
        raise ValueError("BMI index contract contains duplicate outlet identifiers")
    if len(set(manifest["pollutants"])) != len(manifest["pollutants"]):
        raise ValueError("BMI index contract contains duplicate pollutant identifiers")
    if len(set(manifest["pathway_names"])) != len(manifest["pathway_names"]):
        raise ValueError("BMI index contract contains duplicate pathway names")
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def index_contract_values(context) -> dict[str, np.ndarray]:
    """Publish a readable manifest plus its fixed 32-byte SHA-256 digest."""
    manifest = index_manifest(context)
    return {
        "coupling__exchange_schema_version": np.array([EXCHANGE_SCHEMA_VERSION], dtype=np.int64),
        "coupling__index_manifest_utf8": np.frombuffer(manifest, dtype=np.uint8).copy(),
        "coupling__index_fingerprint": np.frombuffer(hashlib.sha256(manifest).digest(), dtype=np.uint8).copy(),
    }
