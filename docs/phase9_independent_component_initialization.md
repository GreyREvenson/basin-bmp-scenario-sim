# Phase 9 — independent source-file initialization of BMI components

## Purpose and boundaries

The Phase 8 generator and scientific model had separate *mutable states*, but
both were constructed by the Monte Carlo driver from the same in-memory
`_shared_payload`. Phase 9 permits each adapter to initialize itself, in its
own process if desired, from a **versioned JSON startup manifest** pointing to
the existing validated source-model YAML. The adapters load, validate and
prepare their own input data independently. No Python context object, NumPy
array, pandas dataframe or RNG instance is passed from one adapter to the
other during this startup path.

This is a new **optional** initialization path. The current native Monte Carlo
driver still uses its existing seeded worker flow, and prepared-context adapter
construction remains supported. Phase 9 changes **no** scientific equation,
stochastic distribution, BMP response method, RNG draw boundary, BMI exchange
variable, parcel ordering, output file schema or replay format.

## Source manifest

Save a UTF-8 JSON file, for example `scenario_1_bmi.json`:

```json
{
  "schema_version": 1,
  "model_config": "../model_config.yaml",
  "worker_seed": 12345,
  "scenario_id": 1
}
```

- `model_config` is a required path to the existing project model YAML;
  relative paths resolve **from the manifest directory**, not the process
  working directory. Paths inside the model YAML continue to resolve relative
  to the YAML according to existing input-config rules.
- `worker_seed` is required and is the **individual scenario's worker seed**
  (nonnegative integer less than 2^64), **not** necessarily the project's base
  seed. To reproduce a native Monte Carlo scenario, use the seed produced by
  the existing `SeedSequence(base_seed).spawn(...)` scheme.
- `scenario_id` is a required, positive, one-based integer.
- `replay_file` is an optional path to an existing completed BMP-scenario JSON
  file, resolved relative to the manifest; its identity and worker seed are
  validated by the live generator.
- `schema_version: 1` is required. Missing/unknown fields, booleans in place
  of integer fields, missing source files and incompatible versions fail before
  input preparation. A JSON manifest is **not** a pickle and contains no
  executable Python expressions.

## Standard BMI component lifecycle

```python
from basin_bmp_sim.interfaces.bmi import LiveScenarioGeneratorBmi, WatershedModelBmi
from basin_bmp_sim.execution.bmi_coupling import (
    get_bmi_value, transfer_bmi_value, validate_bmi_index_contract,
)
from basin_bmp_sim.interfaces.domain_exchange import DOMAIN_FIELDS

model = WatershedModelBmi()
generator = LiveScenarioGeneratorBmi()
model.initialize("scenario_1_bmi.json")  # independently prepares its context
generator.initialize("scenario_1_bmi.json")  # independently prepares its own
validate_bmi_index_contract(generator, model)
for name in DOMAIN_FIELDS:
    transfer_bmi_value(model, generator, name)
generator.accept_domain()
# Live placement sampling still requires the Phase 8 BMI RNG and placement
# transfer steps before a cost is drawn. See the coupler example below.
# Call finalize() after completing pending placements.
```

For a complete one-scenario application of the approved BMI-mediated protocol,
use the lightweight coupler, which coordinates the existing explicit exchange:

```python
from basin_bmp_sim.execution.bmi_coupling import AnnualBmiScenarioCoupler

coupler = AnnualBmiScenarioCoupler.from_manifest("scenario_1_bmi.json")
coupler.initialize()
try:
    responses = []
    while (response := coupler.next_bmp()) is not None:
        responses.append(response)
    replayable = coupler.completed_scenario()
finally:
    coupler.finalize()
```

The coupler instantiates two **unprepared adapters**. On `initialize`, each
adapter reads its own input YAML and source tables, constructs its own
`ScenarioModelContext` or `GeneratorModelContext`, publishes the same ordered
index manifest, and imports the model's physical placement domain through BMI.
The coupler never constructs a driver `Model`, nor does it inject the model's
scientific context into the generator. Its `CompletedBMPResponse` records and
replayable `BMPScenario` are returned to the caller; this example does **not**
write the Monte Carlo driver's usual Parquet tables.

## Preparation code ownership

`basin_bmp_sim/models/prepared_inputs.py` now owns model lookup preparation and the
validated worker payload used by `ScenarioModelContext`. The original
`basin_bmp_sim.execution.monte_carlo_driver.Model` inherits these methods, preserving native
imports, lookup semantics, and outputs. The independent loader in
`basin_bmp_sim/interfaces/component_bootstrap.py` constructs `PreparedModelInputs`
directly; it does not import or construct `Model` from the driver. Input
validation remains in the existing project input-config/validation modules.

## Scientific and architectural limitations

- The independent path currently **loads all project data twice** if the model
  and generator initialize separately; it is intentionally simple and does not
  yet provide a compact, model-neutral generator settings file. On large GIS
  inputs this is more expensive than the native worker path.
- Both components still need the **same validated project YAML and compatible
  scientific formulation**. This phase does not make an arbitrary outside BMI
  watershed model compatible with the existing BMP generator. The Phase 7/8
  index and physical-domain contracts and Phase 8 RNG exchanges are still
  required.
- The config manifest and RNG byte sidecars are project-specific, not CSDMS
  standardized variable names or a BMI wire protocol. The adapter method
  catalog is BMI 2.0-shaped, but complete official compliance and
  interoperability still require external certification.
- The native driver continues to own ensemble scheduling and result
  persistence. Standalone BMI component execution returns scientific results
  and replay metadata; a future phase can formalize persistence and
  diagnostic-output exchange without crossing native Python contexts.

## Testing and before/after equivalence

- `tests/test_phase9_independent_bmi_initialization.py` checks strict manifest
  parsing, path resolution, independent adapter initialization, source-context
  separation, live BMI coupling in PLET/RUSLE and statistical configurations,
  AFO cost/recovery ordering and completed-scenario equality, and preservation
  of the driver's inherited preparation behavior. Component execution tests
  use small already-validated fixtures and stub only *file-loading* machinery;
  actual end-to-end file loading with a real project YAML is not exercised by
  these unit tests.
- Eight seeded cases from the **untouched Phase 8 uploaded ZIP** and the
  refactored code produced identical captured per-scenario output tables and
  outlet trajectories, SHA-256
  `1b63916915012a41120cf61a2e33ff76bc4372e5904a8a3d1f5d7a142cb7c208`.
- The broad scientific subset includes the Phase 1–9 architectural tests,
  PLET/RUSLE and statistical modes, wetlands, AFO recovery and accounting,
  parcel routing and BMP stacking. The full suite cannot be established here
  because the environment lacks the Parquet engine `pyarrow`; it also lacks
  `python-calamine`. Official `bmipy` checks are skipped until the optional
  `bmi` extra is installed.

Run `python -m pytest tests` in the dependency-complete environment before
merging, and exercise the manifest against a real user YAML for full file-based
integration verification.
