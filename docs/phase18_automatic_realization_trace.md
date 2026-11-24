# Phase 18 — automatic, replay-grounded BMP input traces

## Scope and scientific meaning

Phase 17 accepted a *manually supplied* realization-aligned feature matrix. This
phase adds an opt-in exporter that extracts the **actual realized values already
recorded** in the Phase 2 `BMPScenario` replay definitions embedded in Phase
12/13 BMI ensembles. The exporter does **not** run a scientific model, generate
new random draws, change RNG state, or alter the ensemble/checkpoint/replay files.
Its output is deterministic for an unchanged ensemble.

**Scope deliberately limited to saved BMP realization data.** The replay
records contain placement identities and practice codes, sampled/fixed physical
designs and treatment fractions, signed pathway efficiencies, failures, AFO
recovery gains, and realized costs. They do **not** contain every separately
sampled baseline PLET hydrology, nutrient concentration, soil-erosion, AFO
excretion, or statistical load-distribution input. Phase 18 **does not pretend
these unsaved inputs can be recovered** from a worker seed or resampled
distributions. No formal variance attribution or Sobol index is produced.

Two optional artifacts are available:

1. **Phase 17-compatible summary trace** (exact trace schema version 1): one
   row per scenario aligned by scenario ID and worker seed, with explicit units
   and source provenance. All features have `role: realized_diagnostic`, even
   when sourced from actual sampled efficiencies, because **aggregated sums**
   are downstream of stochastic BMP selection/placement. They cannot be
   interpreted as independent input effects.
2. **Detailed realization ledger** (schema version 1): a self-contained copy
   of the actual recorded placement values, with the complete index manifest
   and the SHA-256 digest of the original ensemble. These values may reflect
   either fixed or stochastic inputs. This ledger avoids reconstructing draws
   from the distributions and supports independent inspection.

Every row reports:

- Total realized BMP count and failed BMP count (`count`).
- Sum of BMP physical-design `cost_quantity_ha` (`ha`), and sum of treated
  parcel fractions (`1`); fractions are summed across *placements* so an
  upstream parcel treated by multiple BMPs may contribute multiple times.
- Sum of recorded AFO recovery gain fractions (`1`), across all placements.
- Realized placement count for each CPS code (`count`).
- Signed sum of recorded BMP efficiency values for each ordered pollutant and
  pathway (`1`). These are efficiencies as **drawn/recorded**, not achieved
  whole-watershed effectiveness or pollutant mass reduction. Negative values
  are retained. A zero for an unselected practice means **no realized draw**,
  not a scientific zero-effect draw.

The exporter intentionally does **not** use observed costs or outlet mass
reductions as predictor features for those same outcomes. The detailed ledger
still retains realized BMP costs because those are part of the original replay
and needed for audit/replay, not to claim independent cost predictors.

The Phase 17 analysis accepts up to 100 features. An unusually large
CPS×pollutant×pathway domain exceeding 100 trace columns is rejected with an
explanation; the original ensemble remains unaffected.

## Export after an existing BMI ensemble

Use a completed Phase 12 or Phase 13 model-neutral ensemble JSON:

```bash
python -m basin_bmp_sim.analysis.realization_trace \
    outputs/ensemble.json \
    outputs/bmp_realization_trace.json \
    --ledger outputs/bmp_realization_ledger.json
```

Or write the files automatically when the selected BMI ensemble finishes:

```bash
basin-bmp-sim \
    --backend-selection ensemble_selection.json \
    --coupled-result outputs/ensemble.json \
    --input-trace-result outputs/bmp_realization_trace.json \
    --input-ledger-result outputs/bmp_realization_ledger.json
```

`--input-ledger-result` is optional, but it requires `--input-trace-result`.
These flags are **only** accepted for selected BMI ensembles; they are not
part of the legacy native YAML/Parquet CLI. Existing backend-selection
schema versions 1, 2 and 3 remain valid, without new required keys.

The Phase 12/13 BMI generator uses the documented independent worker seed
schedule, which is the basis for setting `sampling_design_confirmed: true`
for the native backend. For an **external** scientific backend, the exporter
cannot confirm independence of any additional model-specific draws. It
requires the analyst to explicitly attest to an IID ensemble design by adding
`--confirm-external-independent-draws`. This assertion is recorded in the
trace provenance. It is not a statistical independence test.

## Use the trace with Phase 17

```bash
python -m basin_bmp_sim.analysis.input_sensitivity \
    outputs/ensemble.json \
    outputs/bmp_realization_trace.json \
    decision_support.json \
    sensitivity.json \
    outputs/bmp_input_associations.json \
    --chart outputs/bmp_input_associations.svg
```

The resulting Spearman correlations quantify **descriptive association of
realized BMP choices/efficiencies with costs or signed mass reduction**. Because
features reflect both selection and effect draws, they are not independent
model drivers. Formally attributing baseline or parameter uncertainty requires
instrumenting scientific samplers with **named, scoped input draw events**,
including parcel, pollutant, model parameter, sample group and distribution.
That is a separate next step, not accomplished by replay extraction alone.

## Validation and reproducibility

- The exporter validates Phase 12/13 ensemble IDs, seeds, domain ordering,
  compact/original results consistency, replay placement counts, CPS and host
  identity alignment, per-placement cost agreement, finite numeric inputs,
  efficiency pathway keys, AFO gains and treated parcel fractions.
- Input files are read-only. The trace and optional ledger must not overwrite
  the source or each other. Validation and JSON serializability checks happen
  before either file is published; each output uses a separate atomic rename.
  **The two output files are not a multi-file transaction.**
- Serial Phase 12 and checkpointed Phase 13 ensembles with identical replay
  results produce identical exported traces. This also holds across resume,
  since export reads the *completed* ensemble, not partial worker memory.
- The existing Phase 17 manual trace format remains supported, and older
  native-model Parquet outputs are not modified.
