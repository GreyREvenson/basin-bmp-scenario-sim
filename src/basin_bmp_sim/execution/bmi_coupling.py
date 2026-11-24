"""BMI variable-transfer mediator between generator and scientific model.

The driver never passes a RealizedBMPPlacement, PendingPlacementResponse, or
scientific load array directly between components.  Each BMP is first exposed
as flattened numeric generator outputs, copied via BMI ``get_value``, and
received through watershed-model BMI ``set_value``.  Model-derived AFO gains
then flow back to the generator through the same method pair, *before* the
original cost RNG draw; realized cost travels back to the model afterward.

The watershed model and generator now own distinct contexts. The scientific
model publishes ordered physical domain arrays, transferred through BMI into
stochastic generation. Versioned RNG-state handoffs preserve the single legacy
stream across both contexts. The driver still owns preparation of the shared
input data, diagnostics and persistence; this is not yet independent process
initialization or general third-party BMI model interoperability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from basin_bmp_sim.interfaces.bmi.live_scenario_generator_bmi import (
    LiveScenarioGeneratorBmi, NoEligiblePlacements,
)
from basin_bmp_sim.interfaces.bmi.watershed_model_bmi import WatershedModelBmi
from basin_bmp_sim.interfaces.placement_exchange import PLACEMENT_FIELDS
from basin_bmp_sim.interfaces.bmi.record_exchange import decode_bmp_record
from basin_bmp_sim.interfaces.coupling_lifecycle import CouplerPhase, negotiate_lifecycle
from basin_bmp_sim.scenarios.generator import BMPScenario
from basin_bmp_sim.interfaces.exchange_variables import EXCHANGE_SCHEMA_VERSION, MODEL_VARIABLES
from basin_bmp_sim.interfaces.domain_exchange import DOMAIN_FIELDS
from basin_bmp_sim.scenarios.generator_context import GeneratorModelContext


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


def validate_annual_model_bmi(model) -> None:
    """Preflight the complete *model-neutral* annual BMI input/output surface.

    Only public BMI methods are used; an alternative model need not expose a
    PLET context or inherit from the bundled watershed adapter. The checks
    prevent silent pollutant/outlet reshaping or unit reinterpretation.
    """
    inputs = set(model.get_input_var_names())
    outputs = set(model.get_output_var_names())
    for var in MODEL_VARIABLES:
        if var.name not in (inputs if var.intent == "input" else outputs):
            raise ValueError(f"Annual BMI model missing {var.intent} variable {var.name}")
        if model.get_var_type(var.name) != str(np.dtype(var.dtype)):
            raise ValueError(f"Annual BMI model dtype mismatch for {var.name}")
        if model.get_var_units(var.name) != var.units:
            raise ValueError(f"Annual BMI model unit mismatch for {var.name}")
        if model.get_grid_rank(model.get_var_grid(var.name)) != var.rank:
            raise ValueError(f"Annual BMI model rank mismatch for {var.name}")
    sizes = [int(get_bmi_value(model, name)[0]) for name in (
        "model__parcel_count", "model__pollutant_count", "model__outlet_count"
    )]
    if any(n < 1 for n in sizes):
        raise ValueError("Annual BMI parcel/pollutant/outlet counts must be positive")
    parcels, pollutants, outlets = sizes
    for name, expected in (
        ("model__parcel_baseline_load_rate", parcels * pollutants),
        ("model__parcel_current_load_rate", parcels * pollutants),
        ("model__outlet_removed_mass", pollutants * outlets),
    ):
        value = get_bmi_value(model, name)
        if value.size != expected or not np.isfinite(value).all():
            raise ValueError(f"Annual BMI model has invalid layout or values for {name}")
    # Verify actual manifest identities, not merely numeric counts.
    manifest = json.loads(get_bmi_value(model, "coupling__index_manifest_utf8").tobytes().decode("utf-8"))
    for key, expected in (("parcel_ids", parcels), ("pollutants", pollutants),
                          ("outlet_oids", outlets)):
        if len(manifest[key]) != expected:
            raise ValueError(f"Annual BMI model {key} manifest disagrees with load layout")


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
        self.generator_context = GeneratorModelContext(ctx, self.seed)
        self.generator = LiveScenarioGeneratorBmi(
            self.generator_context, cfg, scenario_id=self.scenario_id,
            seed=self.seed, replay=replay, require_domain_import=True,
        )
        self.model = WatershedModelBmi(ctx, logger)
        self.initialized = False
        self._phase = CouplerPhase.CREATED
        self._startup_manifest = None
        self._generator_config_file = ""
        self._model_config_file = ""

    @classmethod
    def from_manifest(cls, manifest_path):
        """Build a file-initialized coupler with two independently prepared BMI owners.

        Both adapters read their own validated source inputs during
        initialize(); no driver Model, Python shared payload or mutable
        generator/scientific context is handed between them.
        """
        from pathlib import Path
        from basin_bmp_sim.interfaces.component_bootstrap import ComponentStartup
        spec = ComponentStartup.read(manifest_path)
        coupled = cls.__new__(cls)
        coupled.ctx = None
        coupled.cfg = None
        coupled.logger = None
        coupled.scenario_id = spec.scenario_id
        coupled.seed = spec.worker_seed
        coupled.generator_context = None
        coupled.generator = LiveScenarioGeneratorBmi(require_domain_import=True)
        coupled.model = WatershedModelBmi()
        coupled.initialized = False
        coupled._phase = CouplerPhase.CREATED
        coupled._startup_manifest = str(Path(manifest_path).resolve())
        coupled._generator_config_file = ""
        coupled._model_config_file = ""
        return coupled

    @classmethod
    def from_components(cls, generator, model, *, seed: int, scenario_id: int = 1,
                        generator_config_file: str = "", model_config_file: str = ""):
        """Couple two independently created BMI components through public variables.

        This does not instantiate the native driver or assume either component
        has a project-specific Python context. Components must publish exactly
        matching ordered index manifests and support annual contract variables.
        """
        coupled = cls.__new__(cls)
        coupled.ctx = None
        coupled.cfg = None
        coupled.logger = None
        coupled.scenario_id = int(scenario_id)
        coupled.seed = int(seed)
        coupled.generator_context = None
        coupled.generator = generator
        coupled.model = model
        coupled.initialized = False
        coupled._phase = CouplerPhase.CREATED
        coupled._startup_manifest = None
        coupled._generator_config_file = str(generator_config_file)
        coupled._model_config_file = str(model_config_file)
        return coupled

    @property
    def lifecycle_phase(self) -> str:
        """Public logical state of this coupled realization."""
        return self._phase.value

    def initialize(self):
        if self._phase is not CouplerPhase.CREATED:
            raise RuntimeError("BMI scenario coupler already initialized or finalized")
        # The baseline model consumes its draws *before* the generator samples.
        # No private _ctx or other implementation fields are read across the
        # boundary, including for separately file-initialized components.
        initialized = []
        try:
            self.model.initialize(self._model_config_file or self._startup_manifest or "")
            initialized.append(self.model)
            self.generator.initialize(self._generator_config_file or self._startup_manifest or "")
            initialized.append(self.generator)
            negotiate_lifecycle(self.generator, self.model)
            validate_bmi_index_contract(self.generator, self.model)
            validate_annual_model_bmi(self.model)
            for name in DOMAIN_FIELDS:
                transfer_bmi_value(self.model, self.generator, name)
            self.generator.accept_domain()
            self.model.set_value("scenario__identifier", np.array([self.scenario_id], dtype=np.int64))
        except Exception:
            # Keep independently initialized components usable for retry only
            # through a *new* coupler; avoid leaking owned resources if any
            # negotiation or domain validation fails.
            for component in reversed(initialized):
                try:
                    component.finalize()
                except Exception:
                    pass
            self._phase = CouplerPhase.FINALIZED
            self.initialized = False
            raise
        self.initialized = True
        self._phase = CouplerPhase.READY

    def next_bmp(self) -> CompletedBMPResponse | None:
        if self._phase is CouplerPhase.EXHAUSTED:
            return None
        if self._phase is not CouplerPhase.READY:
            raise RuntimeError("Initialize the BMI coupler before execution")
        # Worker baseline may have consumed stochastic draws. Synchronize
        # the generator's separate stream before any placement is sampled.
        transfer_bmi_value(self.model, self.generator, "coupling__model_rng_state_utf8")
        try:
            self.generator.update()
        except NoEligiblePlacements:
            self._phase = CouplerPhase.EXHAUSTED
            return None

        cps = int(get_bmi_value(self.generator, "placement__cps_code")[0])
        idx = int(get_bmi_value(self.generator, "placement__host_parcel_index")[0])
        # The transfer is entirely numeric and includes wetland physical
        # catchments, grassed design, effectiveness and replay gains.
        for name in PLACEMENT_FIELDS:
            transfer_bmi_value(self.generator, self.model, name)
        transfer_bmi_value(self.generator, self.model, "coupling__generator_rng_state_utf8")
        self.model.update()

        # Crucial RNG ordering: model AFO recovery -> generator cost draw ->
        # model mass accounting. Both exchanged values use standard BMI.
        transfer_bmi_value(self.model, self.generator, "coupling__model_rng_state_utf8")
        transfer_bmi_value(self.model, self.generator, "placement__recovery_gains")
        self.generator.complete_pending()
        transfer_bmi_value(self.generator, self.model, "placement__cost")
        transfer_bmi_value(self.generator, self.model, "coupling__generator_rng_state_utf8")
        self.model.complete_pending()
        bmp_record = decode_bmp_record(get_bmi_value(self.model, "model__last_bmp_record_utf8"))
        return CompletedBMPResponse(
            cps=cps,
            parcel_index=idx,
            cost_usd=float(get_bmi_value(self.generator, "placement__cost")[0]),
            completed_count=int(get_bmi_value(self.generator, "scenario__bmp_count")[0]),
            total_cost_usd=float(get_bmi_value(self.generator, "scenario__cumulative_cost")[0]),
            delivered_reductions=get_bmi_value(self.model, "model__outlet_removed_mass").reshape(
                int(get_bmi_value(self.model, "model__pollutant_count")[0]),
                int(get_bmi_value(self.model, "model__outlet_count")[0]),
            ),
            bmp_record=bmp_record,
        )

    def completed_scenario(self):
        """Read the replay through the generator's advertised BMI byte array.

        ``export_replay`` is the versioned public lifecycle extension; the
        coupler never receives a generator-owned Python scenario instance.
        """
        if self._phase not in (CouplerPhase.READY, CouplerPhase.EXHAUSTED):
            raise RuntimeError("Initialize the BMI coupler before exporting replay")
        self.generator.export_replay(self.seed)
        raw = get_bmi_value(self.generator, "scenario__replay_utf8").tobytes()
        try:
            data = json.loads(raw.decode("utf-8"))
            scenario = BMPScenario.from_dict(data)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid BMI replay JSON payload") from exc
        if scenario.scenario_id != self.scenario_id or scenario.seed != self.seed:
            raise ValueError("BMI replay scenario identity or seed mismatch")
        return scenario

    def finalize(self):
        if self._phase is CouplerPhase.FINALIZED:
            return
        try:
            self.generator.finalize()
        finally:
            # Always release the independent scientific model owner even if
            # the generator's cleanup fails.
            try:
                self.model.finalize()
            finally:
                self.initialized = False
                self._phase = CouplerPhase.FINALIZED
