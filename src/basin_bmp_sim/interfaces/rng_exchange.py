"""Lossless JSON transport of the shared *logical* RNG stream via BMI uint8.

The legacy implementation deliberately interleaves RNG draws across baseline,
placement, AFO recovery and costs. Separate component RNGs must therefore
handoff state at each of those existing boundaries. No random numbers are
sampled by this serializer. This is a project-local coupling protocol, not a
CSDMS standard and not a generally interoperable random-stream specification.
"""
from __future__ import annotations

import json
import numpy as np

RNG_SCHEMA_VERSION = 1


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported random-state type {type(value).__name__}")


def encode_rng_state(rng) -> np.ndarray:
    payload = {"schema_version": RNG_SCHEMA_VERSION, "state": rng.bit_generator.state}
    data = json.dumps(payload, default=_json_default, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    return np.frombuffer(data, dtype=np.uint8).copy()


def decode_rng_state(basin_bmp_sim, rng) -> dict:
    values = np.asarray(basin_bmp_sim)
    if values.ndim != 1 or values.dtype != np.uint8 or values.size > 1_000_000:
        raise ValueError("BMI RNG state requires a bounded uint8 vector")
    try:
        data = json.loads(values.tobytes().decode("utf-8"))
        if (not isinstance(data, dict) or data.get("schema_version") != RNG_SCHEMA_VERSION
                or not isinstance(data.get("state"), dict)):
            raise ValueError("Unsupported BMI RNG state schema")
        state = data["state"]
        if state.get("bit_generator") != type(rng.bit_generator).__name__:
            raise ValueError("BMI RNG bit-generator mismatch")
        # Validate with a throwaway bit generator first so a malformed packet
        # cannot partially mutate the live scientific stream.
        trial = type(rng.bit_generator)(0)
        trial.state = state
    except (ValueError, UnicodeError, json.JSONDecodeError, KeyError, TypeError,
            OverflowError, AttributeError) as exc:
        raise ValueError("Invalid BMI RNG state payload") from exc
    return state


def restore_rng_state(rng, basin_bmp_sim) -> None:
    rng.bit_generator.state = decode_rng_state(basin_bmp_sim, rng)
