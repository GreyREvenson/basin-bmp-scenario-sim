"""Lossless BMI uint8 sidecar for heterogeneous per-BMP output records.

BMI numeric fields remain authoritative for scientific coupling.  A BMP
record contains variable-length text such as CPS names and parcel identifiers,
which are not BMI numeric variables.  Encode these *output records only* as
UTF-8 JSON so that the driver can collect model outputs without depending on
private native result objects.  Consumers must not treat this as a spatial
scientific variable or a CSDMS Standard Name.
"""
from __future__ import annotations

import json

import numpy as np


def encode_bmp_record(record: dict) -> np.ndarray:
    def adapt(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Unsupported BMI BMP-record value type: {type(obj).__name__}")
    # Preserve legacy numeric behaviour (including any non-finite diagnostics)
    # through the initial interface extraction. Validation is a separate phase.
    text = json.dumps(record, default=adapt, ensure_ascii=False, allow_nan=True)
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def decode_bmp_record(payload: np.ndarray) -> dict:
    arr = np.asarray(payload)
    if arr.ndim != 1 or arr.dtype != np.uint8 or not len(arr):
        raise ValueError("BMI BMP output record must be a nonempty uint8 vector")
    value = json.loads(arr.tobytes().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("BMI BMP output record must contain a JSON object")
    return value
