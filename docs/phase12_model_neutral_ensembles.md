# Phase 12 — Deterministic model-neutral BMI ensembles

## Scope and compatibility

Phase 12 extends the **opt-in** selected-backend path introduced in Phase 11
from one realization to a deterministic, **serial ensemble**. Each realization
initializes a fresh stochastic generator and a fresh native or external
watershed-model adapter. The existing native `basin-bmp-sim model.yaml`
execution path, scientific formulations, native ensemble worker seeds,
Parquet/plot outputs, and Phase 1–11 replay formats are unchanged.

The Phase 11 selection JSON **schema version 1 still runs exactly one
realization** and writes the unchanged Phase 11 result schema version 1.
**Schema version 2** adds exactly one field, `ensemble`, and requires
`ensemble.n_scenarios` to be a positive integer. No new YAML option silently
switches the backend. The extra BMI ensemble path writes a distinct JSON
schema; it does *not* pretend that external-model outputs are native PLET
accounting or Parquet tables.

## Configuring and running an ensemble

Create the existing Phase 9 startup manifest, with **base seed** and **first
scenario ID**:

```json
{
  "schema_version": 1,
  "model_config": "model.yaml",
  "worker_seed": 12345,
  "scenario_id": 1
}
```

Select the backend and number of scenarios:

```json
{
  "schema_version": 2,
  "backend": "external",
  "component_manifest": "startup.json",
  "external_factory": "my_model.provider:create_annual_backend",
  "backend_options": {"parameter_file": "my_inputs.json"},
  "required_features": ["afo_recovery"],
  "ensemble": {"n_scenarios": 100}
}
```

For the built-in BMI scientific model, set `"backend": "native"` and omit
`external_factory`, `backend_options` and `required_features`. All Phase 11
backend capability negotiations remain active for **every** realization;
third-party factories must construct fresh model objects for each call.

Run either mode through the existing opt-in CLI:

```bash
basin-bmp-sim --backend-selection ensemble_selection.json --coupled-result outputs/ensemble.json
```

The generator still initializes from the **native model YAML**, using the
approved BMP selection/design/cost inputs and the same scientific domain
exchange. External model providers must support those generator-defined BMP
rules; this phase does not introduce an independently configurable generator.

## Identity and random-number policy

The manifest `worker_seed` is interpreted as a **base seed** in schema 2. For
`n_scenarios=N`, worker seeds are generated in stable order using:

```python
np.random.SeedSequence(base_seed).spawn(N)
# Each child is converted to an integer via:
int(child.generate_state(1, dtype=np.uint64)[0])
```

The result explicitly records policy identifier
`numpy-seedsequence-spawn-uint64-v1`, base seed, each worker seed, and each
scenario ID. IDs are consecutive, starting with the manifest `scenario_id`.
Adding more scenarios does not change earlier scenarios' worker seeds. Each
worker's baseline, placement, AFO recovery and implementation-cost draws
continue in the existing Phase 8 BMI handoff order. The original manifest
and model YAML are not modified. A temporary file-based manifest for each
worker is discarded after execution.

**This is a newly defined BMI-ensemble seed policy:** the native Monte Carlo
CLI currently uses 32-bit child-state seeds, and a Phase 11 single scenario
uses the startup manifest's seed directly. Therefore the same `base_seed`
need not produce identical placements in these **three different modes**.
Numerical equivalence refers to the **unchanged native execution path**;
Phase 12 reproducibility refers to identical repeated BMI-ensemble runs.

Input startup manifests containing `replay_file` are rejected in ensemble
mode to avoid applying the same recorded scenario to multiple independent
seeds. Each saved scenario can be replayed separately (see below).

## Result schema version 2

`--coupled-result` writes **one self-contained JSON file** only after *all*
realizations complete. It contains:

- `result_schema_version: 2`, `run_type: "ensemble"`, `backend`, seed policy,
  root seed, first scenario ID and count.
- `index_manifest`: the **identical ordered index domain** shared by all
  scenarios. A mismatch in parcel, outlet, pollutant, pathway or CPS ordering
  rejects the entire ensemble before output publication.
- `statistics`: cross-scenario descriptive statistics for BMP count, total
  cost (USD), final cumulative outlet removal (kg), baseline parcel load rates
  (kg ha⁻¹ yr⁻¹), and final parcel load rates (kg ha⁻¹ yr⁻¹). Each numerical
  quantity has `mean`, `std_population`, `min`, `max`, `p05`, `median`, and
  `p95`. The standard deviation uses denominator `N` and percentiles use
  NumPy's linear interpolation. **Negative signed removals are not clipped.**
- `scenario_summaries`: one record per realization with scenario ID, seed,
  BMP count, total cost and final cumulative outlet removal.
- `scenarios`: the complete original Phase 11 single-scenario result for
  each realization, including its `replay_scenario`, BMP response records,
  baseline and treated load arrays, and cumulative outlet changes.

Flat arrays retain the Phase 11 conventions: parcel load rates are in
`[parcel, pollutant]` row-major order and outlet masses are
`[pollutant, outlet]` row-major order. `statistics` declares the 2-D shape;
rows remain flattened for direct compatibility. All estimates describe
**variation across annual stochastic realizations**, not temporal variance,
confidence intervals, or independently verified scientific uncertainty.
An external backend's declared capabilities and outputs remain the scientific
responsibility of its provider.

Results are built in memory and written as a temporary file followed by an
atomic replacement, so a failed realization cannot overwrite the previous
complete result. This initial implementation is **serial and memory-backed**:
it does not offer `n_jobs`, distributed runs, streaming summary calculation,
resume/checkpoint, or separate Parquet tables. Large ensembles may need a
future streaming/persistence extension.

## Replay export and independent reruns

Every scenario result contains its complete original Phase 2 replay definition.
To export individual `BMPScenario.load()`-compatible files:

```bash
python -m basin_bmp_sim.execution.ensemble_runner outputs/ensemble.json outputs/replays
```

The command writes `s1.json`, `s2.json`, etc., with the actual scenario IDs.
It validates all replay identities before exporting and refuses to overwrite
existing files. To replay one, create a Phase 9 **single-scenario** startup
manifest with the exact `scenario_id` and `worker_seed` from its
`scenario_summaries` row, the same original `model_config`, and
`"replay_file": "outputs/replays/s1.json"`. Use a **version 1** selected
backend JSON pointing to that manifest and run the Phase 11 single-scenario
CLI. Preserve the original provider and scientific domain: Phase 2 replay
definitions include model-specific efficiency and AFO response parameters and
cannot be assumed scientifically interchangeable among external backends.

## Verification and next steps

`tests/test_phase12_backend_ensembles.py` covers strict schema validation,
version 1 compatibility, deterministic seed derivation, multi-scenario native
and third-party runs, separate backend instances, repeatability, descriptive
statistics and signed removal, rejection of incompatible domain indices,
per-scenario replay equivalence, and failure atomicity. Existing Phase 1–11
regression tests verify the unchanged default scientific path.

Future steps may add parallel workers, streaming/columnar results, ensemble
resume/retry policies, scientific target-exceedance summaries, and fully
independent generator configuration. None is implicitly claimed here.
