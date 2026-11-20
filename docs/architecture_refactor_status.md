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
