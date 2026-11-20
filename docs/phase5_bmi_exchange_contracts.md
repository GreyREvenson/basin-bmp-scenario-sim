# Phase 5: BMI 2.0 adapters and numerical exchange contracts

This phase applies to the supplied `dev_arch_refac (4)` ZIP, after Phase 4.
It adds an **opt-in Basic Model Interface (BMI) method surface** without
replacing any existing scenario execution, numerical kernel, scientific
formulation, or CLI. Both existing approved formulations continue to be
selected and validated through the Phase 4 formulation registry.

## Public modules

| Python class | Location | Purpose |
| --- | --- | --- |
| `StochasticScenarioBmi` | `src/interfaces/bmi/stochastic_scenario_bmi.py` | Iterate **completed** saved `BMPScenario` JSON definitions without resampling |
| `LiveScenarioGeneratorBmi` | `src/interfaces/bmi/live_scenario_generator_bmi.py` | Expose one live pre-response BMP placement per update and the committed cost after scientific response |
| `WatershedModelBmi` | `src/interfaces/bmi/watershed_model_bmi.py` | Wrap one existing `AnnualModelRealization`, exposing baseline and treated load-rate vectors and delivered mass change |
| `MonteCarloDriverBmi` | `src/interfaces/bmi/monte_carlo_driver_bmi.py` | Run one full stochastic realization per logical BMI `update()`, exporting its latest result |
| `BmiArrayComponent` | `src/interfaces/bmi/array_component.py` | Implement array, variable metadata, logical time, and index-space grid methods |
| `ExchangeVariable`, `scenario_arrays` | `src/interfaces/exchange_variables.py` | Local variable catalog and CSR exchange encoding |

All four public adapters provide the BMI 2.0 lifecycle, info, time,
getter/setter, and grid method signatures. This patch does **not** add a
`bmipy` dependency or claim certification by the CSDMS BMI test harness;
the interface must be checked against the installed `bmipy` specification
in a dependency-equipped environment before asserting formal conformity.

## Execution and time

- `initialize(config_file)` is required before `update` or reading values.
- Driver: `initialize("path/to/settings.yaml")` loads/validates inputs like the
  CLI; or inject a prepared `Model` at construction and call `initialize()`.
  `update()` runs and writes one scenario **serially** using the existing
  native worker; it uses the same `SeedSequence.spawn()` child seeds as the
  native parallel/serial driver. `finalize()` writes the collected outlet
  trajectory table. The existing CLI and native driver's parallel execution
  remain unchanged.
- Recorded scenario stream: `initialize("path/to/scenario_definitions/")`
  or supply an in-memory sequence of completed `BMPScenario` objects and
  `initialize()`; one `update()` exposes one complete definition.
  `get_end_time()` equals the number of saved definitions.
- Live generator: inject a prepared context, configuration, and (for replay)
  the worker seed. Initialize **after** the scientific baseline has been
  sampled. Each `update()` samples one pending placement and exposes its
  numeric design and efficiency fields; call `complete_pending(recovery_gains=...)`
  **after** the scientific model processes the placement, so cost is sampled
  in the original RNG order. `pending_placement` and `completed_scenario(seed)`
  are currently native Python extension methods, not standardized BMI inputs.
  `get_end_time()` is an initial **upper bound** on the number of placements,
  since cost exhaustion and other stopping rules are determined dynamically.
  This BMI facade does not itself enforce the driver's `bmp_limit_n` or
  `bmp_limit_usd` policy; the driver remains responsible for those rules.
- Watershed: inject a prepared `ScenarioModelContext` and logger, then
  `initialize()` to sample the baseline exactly as the current model does.
  `set_placement(plan)` queues a **native Python** placement object,
  `update()` applies its scientific response, and `complete_pending(cost_usd)`
  completes mass/outlet accounting. The sampler must call
  `generator.complete_placement(..., recovery_gains=bmi.pending_recovery_gains)`
  **between** `update()` and `complete_pending()` to preserve legacy AFO
  recovery/cost RNG order. These two bridging methods are explicitly **not**
  BMI methods; a purely BMI-coupled live driver is future work.
- Time units are `none`: logical Monte Carlo scenarios or BMP placements
  are **not** simulated physical time. `update_until(n)` accepts only integer
  positions. A future physically time-stepped model will need distinct
  semantics and an explicit time-unit mediator.

## Exchange variables

`src/interfaces/exchange_variables.py` is the canonical catalog. Names
listed below are **project-local provisional names**, *not* reviewed CSDMS
Standard Names. A future published coupling contract needs name review,
standardization, and compatibility testing.

### Scenario stream (completed realizations)

| Variable | Type | Units | Shape / meaning |
| --- | --- | --- | --- |
| `scenario__identifier` | int64 | 1 | scalar, one-based |
| `scenario__random_seed` | int64 | 1 | scalar, worker seed |
| `scenario__bmp_count` | int64 | 1 | scalar |
| `bmp__cps_code` | int64 | 1 | one entry per BMP |
| `bmp__host_parcel_index` | int64 | 1 | zero-based full hydrologic parcel-universe index |
| `bmp__implementation_cost` | float64 | USD | one realized value per BMP |
| `bmp__affected_parcel_offset` | int64 | 1 | CSR offsets of length `bmp_count+1` |
| `bmp__affected_parcel_index` | int64 | 1 | flattened indices into full parcel universe |
| `bmp__affected_parcel_fraction` | float64 | 1 | flattened fractions paired with affected indices |

For BMP `k`, the affected parcel range is
`[offset[k]:offset[k+1]]`. Values preserve BMP implementation order. The
original `BMPScenario` JSON remains authoritative for variable-length
metadata, efficiency draws, AFO recovery, family, and scenario replay. **The
numeric arrays are not sufficient to recreate all scientific inputs.**

### Live placement generator

`placement__cps_code`, `placement__host_parcel_index`,
`placement__failed`, `placement__recovery_effect_scale`,
`placement__completed`, `placement__cost`, `scenario__bmp_count`, and
`scenario__cumulative_cost` are scalar outputs.
`placement__affected_parcel_index`, `placement__affected_parcel_fraction`,
and `placement__efficiency` are vectors. The efficiency vector uses the
prepared context's pollutant/pathway ordering; the affected-parcel arrays
use the full hydrologic parcel universe. `placement__completed` is 0 while
the response is pending, 1 after the cost is realized. `placement__cost` is
zero until committed, **not** an estimate. A full transferable design object,
including metadata and arbitrary BMP-specific parameters, is not yet
represented entirely by BMI variables.

### Watershed model

`model__parcel_count`, `model__pollutant_count`, `model__outlet_count`
are scalar int64 variables. `model__parcel_baseline_load_rate` and
`model__parcel_current_load_rate` are float64, units `kg ha-1 yr-1`,
row-major `(parcel, pollutant)` vectors. `model__outlet_removed_mass` is
float64, units `kg`, row-major `(pollutant, outlet)` signed cumulative
delivered mass change within the current modeled one-year step. Negative
values are valid (e.g., induced manure effects).

### Full Monte Carlo driver

`ensemble__completed_scenario_count`, `ensemble__requested_scenario_count`,
`scenario__bmp_count` are scalar int64 outputs;
`scenario__total_cost` is scalar float64 USD;
`scenario__outlet_removed_mass` is float64 kg, row-major `(pollutant,
outlet)` values for the most recently completed scenario (not an
across-ensemble mean). Zero BMPs produce zero reduction, not an error.

## Numerical arrays, grids, and state ownership

- `get_value` copies into a preallocated, equal-sized NumPy array of the
  advertised dtype. `get_value_at_indices` copies selected indices. All
  outputs are read-only; `set_value` and `set_value_at_indices` reject them.
- Scalar variables have grid ID 0, rank 0. Each vector has a distinct,
  deterministic, rank-1 **index-space** `uniform_rectilinear` grid, origin
  zero and spacing one. These are *logical array positions*, **not parcel
  geometry**, geographic x coordinates, or hydrologic routing edges.
- A vector retains its logical grid identity even if its length becomes one.
  Dynamically sized arrays can change between scenario updates; clients must
  query the current `get_grid_size` / `get_var_nbytes` and reallocate buffers.
  `get_value_ptr` can be invalidated by an update that resizes a vector.
- The current project can have string parcel IDs and outlet IDs. The adapter
  preserves the existing order in the prepared model, but no geographical
  coordinates, polygon topology, or string ID mapping are currently exposed
  via BMI. Those require a domain contract/sidecar or a carefully designed
  spatial BMI grid before external geographic coupling is supported.
- The scientific model and the generator retain sole ownership of their
  underlying state. The BMI arrays represent exposed **snapshots**, not a
  way to mutate core model state behind the native interface.

## Scientific safeguards and remaining work

The approved formulation registry remains authoritative. The existence of
numeric variables and a BMI method surface does **not** imply that arbitrary
algorithm combinations are scientifically valid. No new algorithm, YAML
option, or model composition is authorized by this phase.

Important future work:

1. Formalize a model-independent physical scenario definition distinct from
   sampled model response, while preserving the legacy random stream.
2. Extend the partial live-generator and watershed BMI facades into a
   complete BMI-only placement/design/response exchange, including
   pollutant/pathway identifiers, spatial relationships, BMP-specific
   metadata, AFO effects, and model capabilities. The driver must then
   use the BMI getters/setters **exclusively** for its inter-component
   coupling; this patch deliberately retains native extension calls.
3. Standardize variable names and units; expose a validated spatial domain
   and stable parcel/outlet identifier mappings.
4. Run the official `bmipy` contract/test harness once available, then
   add independent driver-versus-native Parquet/parallel comparisons in a
   complete dependency environment.

## Verification

The eight fixed-seed reference cases in the permanent Phase 1/2 fixtures
were independently captured before and after this patch using the exact
supplied `(4)` ZIP, and have identical serialized result bytes and SHA-256
hashes: `1b63916915012a41120cf61aee2a33ff76bc4372e5904a8a3d1f5d7a142cb7c208`.
These comparisons cover in-field/wetland/grassed PLET/RUSLE, statistical,
AFO mass-based, multi-BMP budget, statistical multi-BMP, and AFO recovery.

The adapter tests in `tests/test_phase5_bmi_contracts.py` exercise the BMI
methods, scalar/vector grid metadata, CSR representation, model state,
seeded driver execution and recorded-scenario replay. They do not replace
the full repository regression suite:

```console
python -m pytest tests
```

The patch-development environment lacks `pyarrow` and `python-calamine`;
examples requiring Parquet or Excel parsing cannot be certified here.
