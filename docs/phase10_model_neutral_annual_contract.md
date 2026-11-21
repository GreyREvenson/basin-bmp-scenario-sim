# Phase 10 — model-neutral annual watershed exchange contract

## Scope

Phase 10 adds a **separate, opt-in interoperability path** for a scientific
backend that is *not* the built-in PLET/RUSLE or statistical engine. It does
not change any existing numerical kernel, approved formulation, driver
configuration, scenario-replay file, BMP sampling algorithm, cost algorithm,
AFO implementation, or Monte Carlo output format. The native driver and
`WatershedModelBmi` are retained unchanged. For the first time, the
`AnnualBmiScenarioCoupler` can execute the existing live generator against
an independently implemented annual scientific model that has **no native
PLET/RUSLE context**.

The contract is **annual**, not a time-stepped hydrology BMI standard. The
current project-specific variable names and the index-manifest/RNG sidecars
are *not* recognized CSDMS Standard Names. Third-party `bmipy` conformance
remains a separate optional check.

## New modules

- `src/interfaces/annual_model_contract.py`: immutable, validated physical
  `AnnualModelDomain`, annual `AnnualLoadSnapshot`, generic
  `AnnualBMPPlacement` (constructed only from numeric BMI input arrays), and
  `AnnualWatershedBackend` Python protocol. **No project-specific science
  modules are imported** by this contract.
- `src/interfaces/bmi/external_watershed_model_bmi.py`:
  `ExternalWatershedModelBmi(domain, backend, rng)` wraps any implementation
  of the backend protocol and exposes the exact existing `MODEL_VARIABLES`
  BMI interface. No native scenario/model context is required.
- `src/driver/bmi_coupling.py`:
  `AnnualBmiScenarioCoupler.from_components(...)` accepts separate generator
  and model BMI instances. `validate_annual_model_bmi` checks all variable
  names, directions, units, dtypes, grid ranks, finite flattened load
  matrices and manifest dimensions **before any BMP execution**. The coupler
  now determines outlet-array shape exclusively via model BMI counts,
  rather than reading its native `ctx`.

The existing `AnnualBmiScenarioCoupler(ctx, cfg, ...)` and
`AnnualBmiScenarioCoupler.from_manifest(...)` continue to work unchanged.

## Required physical domain

`AnnualModelDomain` contains these **ordered** identities:

- `parcel_ids`: all hydrologic parcels (not just eligible BMP hosts)
- `outlet_oids`: outlet IDs
- `pollutants`: names/ordering of annual load columns
- `pathway_names`: ordering of efficiency columns, including surface and
  subsurface pathways where present
- `bmp_cps`: allowed CPS codes, matching the generator's approved options

It also contains `parcel_area_ha`, `parcel_perim_m`, immediate upstream
connectivity as CSR (`upstream_offset` and `upstream_index`), and
`selection_source_idxs` with `parcel_selection_probs`. The same existing
Phase 7/8 exchange schema (`EXCHANGE_SCHEMA_VERSION = 2`) validates domain
identity and transfers geometry. Domain arrays are copied and read-only;
duplicate/invalid identities, out-of-range indices, invalid probabilities,
upstream cycles, and silent float-to-integer index coercion are rejected.

Both components must publish identical versioned ordered-index manifests.
**Matching row counts alone is not enough**: pollutant, outlet, parcel,
pathway and CPS identities/order must agree before simulation.

## Annual scientific input/output contract

The scientific backend implements these methods:

```python
class MyBackend:
    def initialize(self, rng: np.random.Generator) -> None: ...
    def snapshot(self) -> AnnualLoadSnapshot: ...
    def apply_bmp(self, placement: AnnualBMPPlacement, scenario_id: int,
                  rng: np.random.Generator) -> Sequence[float]: ...
    def complete_bmp(self, cost_usd: float) -> Mapping[str, Any]: ...
    def finalize(self) -> None: ...
```

- `initialize` samples/calculates a baseline using the supplied scenario
  worker RNG; it does **not** need to read a project YAML.
- `snapshot` returns a `AnnualLoadSnapshot` with
  `baseline_rates[parcel, pollutant]` and
  `current_rates[parcel, pollutant]` in **kg ha⁻¹ yr⁻¹**, and
  `outlet_removed_kg[pollutant, outlet]` in **kg**. Outlet removal is cumulative
  *within this scenario's annual realization*; negative values (a net
  increase) are valid and are never clipped. All values must be finite.
- `apply_bmp` receives one immutable model-neutral placement request. It
  owns all scientific response, mass accounting, routing and AFO physics.
  It returns an ordered vector of AFO recovery fractions, all in `[0, 1]`,
  **before** the generator draws the BMP implementation cost. It may advance
  the supplied RNG for model-specific stochastic response.
- `complete_bmp` receives the realized BMI `placement__cost`, finalizes any
  remaining bookkeeping, and returns a JSON-encodable BMP record dictionary.
  `snapshot` must then reflect the completed response.
- `finalize` releases backend resources.

The backend owns its scientific state and the validity of its equations,
mass balances, routing, and AFO calculations. The generic adapter validates
numeric interfaces and lifecycle ordering **but cannot scientifically
validate an arbitrary backend**. Backends should make their own methods
atomic on error; a failed method may have changed native backend state.

`AnnualBMPPlacement` retains host/affected parcel indices and treatment
fractions; `efficiency[pollutant, pathway]` (signed efficiencies permitted);
failed flag; recovery scale; design area, wetland area/catchment ratio,
grassed-waterway length/area/fraction; and recorded AFO gains. The generic
contract does **not** impose the built-in model's CPS-family response rules.
Each external backend must explicitly confirm which practice codes and design
variables it can interpret. New practice-specific variables require a
versioned contract extension, not ad-hoc Python metadata exchange.

## Example: independent components

```python
import numpy as np
from src.driver.bmi_coupling import AnnualBmiScenarioCoupler
from src.interfaces.bmi import ExternalWatershedModelBmi, LiveScenarioGeneratorBmi

# `domain` is a validated AnnualModelDomain supplied by the new model;
# `backend` implements AnnualWatershedBackend with its own scientific rules.
model = ExternalWatershedModelBmi(domain, backend, np.random.default_rng(worker_seed))
generator = LiveScenarioGeneratorBmi(require_domain_import=True)
coupler = AnnualBmiScenarioCoupler.from_components(
    generator, model, scenario_id=1, seed=worker_seed,
    generator_config_file="scenario_1_bmi.json",  # Phase 9 generator manifest
)
coupler.initialize()
try:
    rows = []
    while (response := coupler.next_bmp()) is not None:
        rows.append(response)
    replay = coupler.completed_scenario()
finally:
    coupler.finalize()
```

The coupler exchanges the original Phase 8 variable sets: index manifests,
physical domain, bidirectional RNG snapshots, complete BMP placement arrays,
AFO recovery vector, cost and signed cumulative outlet mass changes. The
model and generator can be configured and constructed independently; the
coupler **does not read the external model's Python context**. An example
independent scientific backend that does *not* use PLET/RUSLE is implemented
in `tests/test_phase10_model_neutral_exchange.py` and runs a complete live
coupling. It is a contract demonstration, **not a scientifically calibrated
watershed model**.

## Important limitations

1. The **generator** still uses the current project's validated YAML,
   selection rules, BMP design samplers, and approved pollutant/pathway/CPS
   ordering; it does not yet read a fully model-neutral generator settings
   file. The external model must support the matching variables and practices.
2. The native CLI driver still uses the built-in scientific backend. The
   external path is accessible via the standalone `from_components` coupler;
   complete custom-model ensemble scheduling, user-facing configuration and
   Parquet persistence are deferred.
3. The exchange retains the Phase 8 project-local RNG protocol, which assumes
   compatible NumPy bit generators and synchronizes state at the baseline,
   placement, recovery and cost boundaries. It is not a generic distributed
   stochastic-process standard.
4. The backend's JSON BMP record need not have the legacy driver record schema.
   Users of external backends must decide how to store and interpret their
   records; the generic coupler returns them unchanged.
5. `get_end_time()` counts BMI placements, **not** physical hydrologic time.
   A daily or multi-year model requires a separate interface agreement.
6. Arbitrary science methods and process chains are **not** certified by this
   patch. Formal BMI conformance and scientific compatibility testing with a
   real independent model remain necessary.

## Tests and equivalence

`tests/test_phase10_model_neutral_exchange.py` verifies the new backend's
full live BMI execution, index identity preflight, domain validation,
negative/signed values, malformed placement rejection, and immutable input
snapshots. Existing formulation and BMI regression tests exercise the native
execution path.

The eight original fixed-seed in-memory output captures (five Phase 1 and
three extended Phase 2 cases) were generated before and after Phase 10 using
identical inputs. Their combined serialized output (including replayable
scenario definitions) is **byte-for-byte identical**. SHA-256:
`bbe7aee7d989e44b912bc8bc4306ba770e9d965006b45ce6038967156108e188`.
These results cover the built-in scientific workflow; they do **not** assert
numerical equivalence between different scientific backends.

The development environment lacks a working Parquet engine and the optional
`bmipy` dependency. Run the full suite in a fully provisioned environment:

```console
python -m pytest tests
```
