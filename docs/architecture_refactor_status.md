# Staged architectural refactor: numerical-equivalence checkpoint

This branch patch is **phase 1**, not the completed BMI framework. It establishes
real package boundaries and a numerical regression gate while preserving the
public `src.model`, `src.bmp`, `src.plet_rusle`, `src.summaries` and
`src.plotting` import paths. The legacy driver uses the newly separated modules
for core calculations, but still performs BMP-response application and expense
sampling in its monolithic scenario loop.

## Implemented in this patch

- `src/bmp/` contains separate PLET-style response implementations for wetlands,
  grassed waterways and generic in-field BMPs; sampling and pathway calculations
  have their own modules. Files with explicit CPS names describe the currently
  supported practices; a registry resolves their response family. Existing
  unrecognized CPS codes retain the original generic in-field fallback for now.
- `src/models/assemblies/` contains the existing annual statistical and
  PLET/RUSLE baseline assemblies. PLET hydrology, infiltration, RUSLE sediment
  and concentration-based nutrient equations have separate modules in
  `src/models/`; original `src.plet_rusle` names remain available.
- `src/scenarios/stochastic_scenario_generator.py` owns the eligibility and
  placement selection state, currently preserving the original shared RNG draw
  order. It **does not yet generate a fully independent, serializable scenario**.
- `src/driver/monte_carlo_driver.py` owns orchestration and the
  `src/driver/results/` package now owns plot preparation, BMP summaries and
  outlet-trajectory collection. Physical parcel/outlet calculations remain in
  the scientific execution loop until further extraction.
- `src/formulations/formulation_registry.py` defines two initial approved
  formulation IDs: `plet_rusle_annual_v1` and `statistical_annual_v1`.
  The optional YAML `simulation.formulation` and `simulation.bmp_methods`
  selections are preflight-validated. Unsupported overrides are rejected,
  rather than ignored. Legacy configuration remains supported.
- Package discovery is updated for nested `src.*` Python packages.

## Not yet completed (separate, incremental patches recommended)

1. Move remaining BMP placement/design/cost sampling completely out of the
   driver, introduce immutable `BMPScenario` and replayable RNG streams.
2. Replace dynamic `types.MethodType` helper binding and the `_ScenarioContext`
   shared payload with explicit state objects and model-neutral contracts.
3. Decouple BMP *design* and *effect* for wetlands, grassed waterways and
   AFOs, preserving upstream parcel accounting and field manure recovery.
4. Extract full model-state updates, pollutant delivery and all baseline-vs-
   scenario mass-accounting calculations from the driver into model assemblies.
5. Complete a BMI variable catalog and CSDMS 2.0 adapters for the generator,
   watershed model and driver, with explicit logical-scenario steps, spatial
   indexing, array dimensions, and metadata. **This patch does not claim BMI
   compliance yet.**
6. Extend results output management to all per-scenario tables, add buffered
   ensemble summaries and stronger serial/parallel output tests.
7. Add alternative, scientifically reviewed process/BMP response algorithms
   **only after** required-input and capability compatibility checks exist.

## Evidence of unchanged numerical results

The baseline for `tests/golden/pre_refactor_seeded_results.json` was captured
from the *unmodified uploaded* `dev_arch_refac` branch, before moving code.
Five scenarios use the same fixed random seed and capture every per-scenario
output table and the returned outlet trajectory records in memory:

- PLET/RUSLE + in-field BMP (CPS 340)
- PLET/RUSLE + wetland BMP (CPS 656)
- PLET/RUSLE + grassed waterway (CPS 412)
- Statistical loads + in-field BMP
- Mass-based AFO + PLET/RUSLE and BMP

After the extraction all five captured outputs compared **exactly**, including
numerical cells and record ordering. The persisted golden test uses
`rtol=1e-12, atol=1e-12` to accommodate small platform-dependent floating
point differences. Run:

```bash
python -m pytest tests/test_architecture_result_equivalence.py -v
python -m pytest tests/test_formulation_compatibility.py -v
python -m pytest tests
```

The ChatGPT execution environment lacked `pyarrow` and outbound package
installation failed. Numerical golden tests capture output DataFrames before
Parquet serialization, so they *do* test the model mathematics. The full
Parquet-backed, serial/parallel tests cannot be certified in that environment.
**Install `pyarrow` in your project's environment and run the entire suite.**

For a full end-to-end before/after test on the *same environment* with the same
config, seed, scenario count and worker settings:

```bash
python run_model.py path/to/config.yaml --outputs before --seed 43812
# Apply patch, install updated code, run in another directory:
python run_model.py path/to/config.yaml --outputs after --seed 43812
python scripts/compare_model_results.py before after
```

Run this for both PLET/RUSLE and statistical configs, and representative AFO
and wetland cases. The comparison reports missing files, schema changes, null
mask differences, categorical changes, and numerical discrepancies. Compare
additional plots manually as needed; log files are intentionally excluded.

## Scientific safeguards

- Preserve existing annual timestep units, source-specific BMP load impacts,
  signed efficiencies, and upstream parcel contributions.
- Preserve sequential stacking and cost-limit stopping until they can be
  explicitly represented in replayable scenario definitions.
- Do not infer scientific correctness from BMI interoperability: the approved
  formulation registry, required-input checks and runtime mass-accounting
  invariants are separate and complementary controls.
- Avoid introducing or activating speculative scientific models solely to
  increase the apparent modularity of the directory structure.

### Additional East Fork example comparison

The existing East Fork statistical example was also executed on the original
and the refactored code, with `random_seed=43812`, `n_scenarios=2`,
`bmp_limit_n=4`, and `n_jobs=1`. All **seven** captured output tables compared
**exactly**, including two BMP tables (four placements each), two parcel
tables (2,563 parcels each), two scenario-metric tables, and the 144-row
outlet-trajectory table. The environment used a temporary in-memory/pickle
serialization substitution because `pyarrow` was missing. This confirms
scientific outputs and dataframe values, **not** native Parquet I/O.

---

# Phase 2: incremental, replayable stochastic scenario realization

This second patch applies to the uploaded `dev_arch_refac (1)` branch **after
Phase 1**. It adds an explicit scenario-generation contract without changing
PLET/RUSLE, statistical, wetland-routing or AFO-response equations.

## What changed

- `src/scenarios/bmp_designs.py` owns model-independent *physical* design
  sampling for wetlands, grassed waterways, and in-field practices. Each
  frozen `BMPDesign` records affected hydrologic parcel indices, treatment
  fractions, cost geometry, and associated output metadata. Wetland designs
  still use the full upstream graph and preserve its deterministic tie order.
- `src/scenarios/stochastic_scenario_generator.py` owns eligible placement
  selection, stochastic BMP efficiencies, BMP failure draws, sampled physical
  designs, per-placement cost draws and stopping bookkeeping. It now provides
  frozen `RealizedBMPPlacement` and `BMPScenario` data classes.
- BMP response methods consume explicit physical designs and efficiency maps;
  for older direct-call tests, a missing design is still sampled using the
  same design helper. This compatibility path can be removed in a later phase.
- AFO recovery effects remain in the scientific model. The realized recovery
  gains are recorded and can be injected during replay without resampling.
- The driver saves each completed realization to
  `outputs/scenario_definitions/s<id>.json` using a versioned schema.
- Replay is available with the same seed, domain, formulation and validated
  inputs. From the command line:

  ```console
  python run_model.py path/to/config.yaml --outputs before --seed 43812
  python run_model.py path/to/config.yaml --outputs after --seed 43812 --scenario-replay-dir before/scenario_definitions
  python scripts/compare_model_results.py before after
  ```

  The scenario count must match, because each input JSON file is loaded by its
  numbered scenario ID. Missing artifacts and incompatible scenario metadata
  cause errors, not silent resampling. In Python, call
  `Model.run_all_scenarios(replay_dir=...)` or
  `_run_one_scenario(..., scenario_definition=BMPScenario.load(path))`.

## Why generation is incremental, not up front

The **existing** RNG is shared between baseline loads, selected BMPs, design,
AFO recovery, and cost. Generating all designs and costs before simulating
loads would reorder draws and change seeded results. Phase 2 therefore
realizes and applies one placement at a time; the resulting **complete**
scenario definition is replayable. A later phase can introduce independently
seeded random streams, a fully model-independent scenario definition and a
separate model-response uncertainty realization. Such a change should be
intentional and documented as potentially changing existing seed-to-output
identity.

`BMPScenario` currently includes both model-independent physical design and
**model-specific response efficiencies/recovery gains** to ensure exact replay.
Do not yet treat it as a portable formulation-neutral interchange schema.
There is **no CSDMS BMI implementation** in this phase; the three BMI adapters,
variable catalog, and model-state independence remain upcoming work. The
model driver still owns some scientific response and mass-accounting logic.

## Before/after verification

- The permanent Phase 1 golden-result tests still pass.
- `tests/golden/pre_phase2_seeded_extended_results.json` was captured from
  the unmodified new uploaded branch (before Phase 2 edits), using the same
  controlled fixtures used by `tests/test_phase2_scenario_generation.py`.
  The three extended cases include multiple BMP types with cost- and count-
  constrained placement and stochastic failure, a statistical multi-BMP run,
  and mass-based AFO recovery. **All output table values, BMP records and
  outlet trajectories matched byte-for-byte** in JSON capture between the
  original and refactored versions.
- Every extended case is also replayed from its saved JSON realization, and
  the replayed outputs are compared to the generated outputs; attempted
  placement/design resampling during replay fails the test.
- The full suite requires `pyarrow` and `python-calamine`, which are absent
  from the patch-construction environment. Tests requiring those optional
  dependencies must be rerun in the project's normal environment. Passing
  the available tests does not prove full native Parquet equivalence.

---

# Phase 3: scientific model state and driver separation

This patch applies to the uploaded **`dev_arch_refac (2)` ZIP**, which already
contains Phase 2. It does **not** alter input schemas, user-facing CLI flags,
BMP methods, PLET/RUSLE equations, or statistical sampling algorithms.

## Separation of responsibilities

- `src/models/assemblies/annual_model_realization.py` owns one independent
  realization of annual scientific model state, using an explicit dataclass.
  `initialize()` selects the supported PLET/RUSLE or statistical baseline
  assembly. `apply_placement()` performs scientific BMP and AFO responses;
  `complete_placement()` reconciles signed parcel mass changes and updates
  pollutant delivery to linked outlets. These two placement calls straddle
  generator-owned cost sampling **on purpose** to preserve the original RNG
  stream and exact Phase 2 replay behavior.
- `src/models/accounting/bmp_mass.py` owns physical mass calculations and
  before/after parcel deltas. `src/models/routing/outlet_delivery.py` owns
  scientific parcel-to-outlet delivery. The driver no longer applies their
  equations itself; it reads accumulated outlet-reduction values.
- `src/models/accounting/afo_accounting.py` owns manure operation budget
  calculations, conservation checks, and AFO component-load records.
  `src/models/diagnostics/load_parameter_records.py` owns the PLET/RUSLE
  numerical diagnostics. The **driver** still writes all output files.
- `src/models/model_context.py` replaces repeated per-instance
  `types.MethodType` binding with statically declared Python helper methods.
  It offers an isolated `ScenarioModelContext` and validates required worker
  fields; `src.model._ScenarioContext` remains a backwards-compatible alias.
- `src/scenarios/deterministic_selection.py` owns the legacy deterministic
  BMP-selection helper. Existing `src.model` imports of old private helpers
  are preserved for downstream tools and unit tests.
- `src/driver/monte_carlo_driver.py` now coordinates the generator and model,
  requests cost draws, manages output records and delegates to its
  `driver/results/` collectors. It does not implement BMP response equations,
  AFO-induced loads, or scientific delivery calculations.

## Before/after numerical evidence

The exact supplied `dev_arch_refac (2)` ZIP was extracted twice. Eight
controlled scenarios were run on the **untouched** and **refactored** source,
using identical seeds and in-memory output capture (not Parquet serialization).
Both full JSON snapshots have identical sizes and SHA-256 digests:

`7008d89edd41fc2433775ac836d8c8a80b7250fd9f671b6b93f114200ac4867a`

The cases are PLET/RUSLE in-field, wetland and grassed-waterway treatments;
statistical in-field loads; AFO mass-based loads; budget-constrained
multi-BMP execution; statistical multiple-BMP execution; and manure recovery.
This establishes **exact equality of the captured tables and outlet
trajectories** for these scenarios, not universal model equivalence.

The prior golden and replay tests remain in place. New tests in
`tests/test_phase3_model_driver_separation.py` exercise the model contract
independently of the driver, validate worker-context requirements and check
that scientific helpers are not attached per instance.

A broad scientific regression selection passed (**346 passed**, one
deselected) with one environment-only exception: `test_custom_csv_config_and_parallel_mass_simulations` requires
`pyarrow`, which is absent in the patch-construction environment. The same
failure was reproduced on the untouched uploaded branch. Full native Parquet
I/O, and the complete test suite, must still be run in the user's environment:

```console
python -m pytest tests
```

## Explicit limits / future phases

This is **not yet a BMI implementation**. `src/interfaces/bmi/` is unchanged.
`AnnualModelRealization` provides the first concrete native Python scientific
execution boundary; a future BMI adapter should wrap it without copying its
scientific logic. Historical process kernels still accept a mutable context
with domain and scenario attributes, and the parallel worker payload remains
mapping-based for backwards compatibility. Do not claim full cross-model
scenario independence: the Phase 2 scenario schema still contains sampled
model-response efficiencies and AFO recovery values. Changing the RNG streams
or replacing the remaining context mapping requires a separate, explicitly
validated migration.

---

# Phase 4: approved scientific method dispatch

The fourth patch adds executable method registries and a resolved formulation
plan. Baseline assembly selection, PLET/RUSLE process calculation, and BMP
response dispatch now use registered callable implementations associated with
the selected, preflight-approved formulation, rather than selection of BMP
response families inside the annual scientific model. No new numerical model
is authorized. See `docs/phase4_formulation_method_contracts.md` for the method
contracts, scientific safeguards, and tests. This remains pre-BMI work.

---

# Phase 5: BMI 2.0 facade and exchange-variable catalog

This phase adds BMI 2.0 *method-compatible adapters* for the driver, annual
scientific model, live placement generator, and recorded scenario definitions, plus a versioned catalog
of numeric exchange variables and tests. **No scientific calculation or
existing driver execution code changes.** The 3 interfaces do not yet form
a purely BMI-coupled live workflow; the Phase 2 generator's interleaved
sampling and the scientific model's rich placement objects remain native.
See `docs/phase5_bmi_exchange_contracts.md` for full contracts, limitations,
validation commands, and the exact before/after numerical evidence.
