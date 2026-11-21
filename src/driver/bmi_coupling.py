"""BMI variable-transfer mediator between generator and scientific model.

The driver never passes a RealizedBMPPlacement, PendingPlacementResponse, or
scientific load array directly between components.  Each BMP is first exposed
as flattened numeric generator outputs, copied via BMI ``get_value``, and
received through watershed-model BMI ``set_value``.  Model-derived AFO gains
then flow back to the generator through the same method pair, *before* the
original cost RNG draw; realized cost travels back to the model afterward.

Scientific model ownership, the shared read-only prepared domain context, and
output-record/persistence orchestration are not yet generalized as independent
CSDMS-executable processes.  No geospatial grid or physical time is invented.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from src.interfaces.bmi.live_scenario_generator_bmi import (
    LiveScenarioGeneratorBmi, NoEligiblePlacements,
)
from src.interfaces.bmi.watershed_model_bmi import WatershedModelBmi
from src.interfaces.placement_exchange import PLACEMENT_FIELDS
from src.interfaces.bmi.record_exchange import decode_bmp_record
from src.interfaces.exchange_variables import EXCHANGE_SCHEMA_VERSION


def get_bmi_value(component, name: str) -> np.ndarray:
    """Return a defensive copy through BMI, not a get_value_ptr alias."""
    dtype = np.dtype(component.get_var_type(name))
    size = component.get_grid_size(component.get_var_grid(name))
    arr = np.empty(size, dtype=dtype)
    component.get_value(name, arr)
    return arr


def transfer_bmi_value(source, target, name: str) -> None:
    """Check contract and copy the complete numeric variable through BMI."""
    if name not in source.get_output_var_names():
        raise ValueError(f"BMI source does not advertise output {name}")
    if name not in target.get_input_var_names():
        raise ValueError(f"BMI target does not advertise input {name}")
    if source.get_var_type(name) != target.get_var_type(name):
        raise ValueError(f"BMI type mismatch for {name}")
    if source.get_var_units(name) != target.get_var_units(name):
        raise ValueError(f"BMI unit mismatch for {name}")
    if source.get_grid_rank(source.get_var_grid(name)) != target.get_grid_rank(target.get_var_grid(name)):
        raise ValueError(f"BMI rank mismatch for {name}")
    target.set_value(name, get_bmi_value(source, name))


def validate_bmi_index_contract(generator, model) -> None:
    """Fail before any BMP is simulated if BMI index domains are incompatible.

    Index equality is necessary but not sufficient for scientific-model
    compatibility; approved formulations and preflight checks remain required.
    """
    for component in (generator, model):
        version = get_bmi_value(component, "coupling__exchange_schema_version")
        if version.size != 1 or int(version[0]) != EXCHANGE_SCHEMA_VERSION:
            raise ValueError("Unsupported BMI exchange schema version")
        fingerprint = get_bmi_value(component, "coupling__index_fingerprint")
        if fingerprint.dtype != np.uint8 or fingerprint.size != 32:
            raise ValueError("BMI index fingerprint must contain 32 SHA-256 bytes")
        manifest = get_bmi_value(component, "coupling__index_manifest_utf8")
        if manifest.dtype != np.uint8:
            raise ValueError("BMI index manifest must contain UTF-8 bytes")
        if not np.array_equal(np.frombuffer(hashlib.sha256(manifest.tobytes()).digest(),
                                            dtype=np.uint8), fingerprint):
            raise ValueError("BMI index fingerprint does not match its published manifest")
        try:
            parsed = json.loads(manifest.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid BMI index manifest JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("BMI index manifest must be a JSON object")
        if parsed.get("exchange_schema_version") != EXCHANGE_SCHEMA_VERSION:
            raise ValueError("BMI manifest exchange schema version mismatch")
    if not np.array_equal(
        get_bmi_value(generator, "coupling__index_fingerprint"),
        get_bmi_value(model, "coupling__index_fingerprint"),
    ):
        raise ValueError("BMI index-domain mismatch: parcel/outlet/pollutant/pathway/CPS ordering differs")
    # Digests are transport checks; retain a complete, inspectable manifest.
    if not np.array_equal(
        get_bmi_value(generator, "coupling__index_manifest_utf8"),
        get_bmi_value(model, "coupling__index_manifest_utf8"),
    ):
        raise ValueError("BMI index-domain manifest mismatch")


@dataclass(frozen=True)
class CompletedBMPResponse:
    """Driver-facing record and standardized numeric diagnostics."""
    cps: int
    parcel_index: int
    cost_usd: float
    completed_count: int
    total_cost_usd: float
    delivered_reductions: np.ndarray
    bmp_record: dict


class AnnualBmiScenarioCoupler:
    """Coordinate one realization using BMI to exchange live scientific data."""

    def __init__(self, ctx, cfg, logger, *, scenario_id: int, seed: int, replay=None):
        self.ctx = ctx
        self.cfg = cfg
        self.logger = logger
        self.scenario_id = int(scenario_id)
        self.seed = int(seed)
        self.generator = LiveScenarioGeneratorBmi(
            ctx, cfg, scenario_id=self.scenario_id, seed=self.seed, replay=replay,
        )
        self.model = WatershedModelBmi(ctx, logger)
        self.initialized = False

    def initialize(self):
        if self.initialized:
            raise RuntimeError("BMI scenario coupler already initialized")
        # Baseline simulation must use the worker RNG *before* the first
        # placement is sampled; initialize in precisely the original order.
        self.model.initialize()
        self.generator.initialize()
        validate_bmi_index_contract(self.generator, self.model)
        self.model.set_value("scenario__identifier", np.array([self.scenario_id], dtype=np.int64))
        self.initialized = True

    def next_bmp(self) -> CompletedBMPResponse | None:
        if not self.initialized:
            raise RuntimeError("Initialize the BMI coupler before execution")
        try:
            self.generator.update()
        except NoEligiblePlacements:
            return None

        cps = int(get_bmi_value(self.generator, "placement__cps_code")[0])
        idx = int(get_bmi_value(self.generator, "placement__host_parcel_index")[0])
        # The transfer is entirely numeric and includes wetland physical
        # catchments, grassed design, effectiveness and replay gains.
        for name in PLACEMENT_FIELDS:
            transfer_bmi_value(self.generator, self.model, name)
        self.model.update()

        # Crucial RNG ordering: model AFO recovery -> generator cost draw ->
        # model mass accounting. Both exchanged values use standard BMI.
        transfer_bmi_value(self.model, self.generator, "placement__recovery_gains")
        self.generator.complete_pending()
        transfer_bmi_value(self.generator, self.model, "placement__cost")
        self.model.complete_pending()
        bmp_record = decode_bmp_record(get_bmi_value(self.model, "model__last_bmp_record_utf8"))
        return CompletedBMPResponse(
            cps=cps,
            parcel_index=idx,
            cost_usd=float(get_bmi_value(self.generator, "placement__cost")[0]),
            completed_count=int(get_bmi_value(self.generator, "scenario__bmp_count")[0]),
            total_cost_usd=float(get_bmi_value(self.generator, "scenario__cumulative_cost")[0]),
            delivered_reductions=get_bmi_value(self.model, "model__outlet_removed_mass").reshape(
                len(self.ctx.pollutants), len(self.ctx.outlet_oids)
            ),
            bmp_record=bmp_record,
        )

    def completed_scenario(self):
        """Persist replayable scenario definitions via the generator owner."""
        return self.generator.completed_scenario(self.seed)

    def finalize(self):
        self.generator.finalize()
        self.model.finalize()
        self.initialized = False
