# Phase 15 — strategy comparison, attainment curves and Pareto frontiers

## Scope

Phase 15 is an **optional, model-neutral post-processing layer** on completed
Phase 12/13 BMI ensemble **result-schema-v2 JSON**. It does not initialize
models, generate scenarios, change BMP practices, change AFO accounting, or
alter original ensemble, checkpoint, replay, native CLI or Parquet results.
It builds on Phase 14's strict ensemble validation and signed annual outlet
mass-removal target definitions.

Inputs are two separate JSON files:

1. A **Phase 14 decision-support configuration** declaring one or more
   `minimum_removed_mass_kg` targets, and optionally `budget_usd`.
2. A **Phase 15 comparison configuration** that names at least two existing
   ensemble JSON files and the target to visualize.

The comparison produces a separate, deterministic JSON report and, optionally,
two standalone SVG visualizations. The SVGs use only Python standard-library
code and require no plotting dependency.

## Example

First, use the same target definitions for all strategies
(`decision_support.json`):

```json
{
  "schema_version": 1,
  "targets": [
    {
      "name": "Outlet 1 TP",
      "pollutant": "TP",
      "outlet_oid": "outlet-1",
      "minimum_removed_mass_kg": 500.0
    }
  ],
  "budget_usd": 100000.0
}
```

Then create `strategies.json`. Paths to ensembles resolve **relative to the
comparison configuration file**:

```json
{
  "schema_version": 1,
  "primary_target": "Outlet 1 TP",
  "comparison_mode": "descriptive",
  "strategies": [
    {"name": "Current strategy", "ensemble_result": "results/current.json"},
    {"name": "Riparian BMP strategy", "ensemble_result": "results/riparian.json"},
    {"name": "Wetland strategy", "ensemble_result": "results/wetlands.json"}
  ]
}
```

Run **without rerunning scientific simulations**:

```bash
python -m basin_bmp_sim.analysis.ensemble_comparison \
    strategies.json \
    decision_support.json \
    outputs/strategy_comparison.json \
    --charts-dir outputs/charts
```

The optional `--charts-dir` creates:

- `cost_vs_reduction.svg`: one marker per strategy; X = mean implementation
  cost (USD), Y = mean signed annual mass removed (kg) for `primary_target`.
  The dashed line marks the target mass; a black ring highlights Pareto-efficient
  points **only** when comparable mode has been explicitly enabled.
- `attainment_curves.svg`: empirical *fraction of realizations meeting or
  exceeding* each annual mass-reduction threshold for `primary_target`.
  The dashed line marks the configured target. Rendering samples 121
  thresholds across the observed data range; the exact target probabilities
  are retained in the JSON report.

The report contains input-file SHA-256 hashes, per-strategy costs, budgets,
per-target means and empirical probabilities, Wilson 95% **Monte Carlo
sampling-precision** intervals, conditional/within-budget outcomes, and joint
target attainment. It preserves signed negative reductions and the Phase 14
null/undefined convention for USD/kg with zero or negative annual removal.

## Cross-model comparison vs Pareto eligibility

**Default mode is `descriptive`.** It checks that all ensembles use identical
ordered pollutant and outlet names, then presents each strategy's results
alongside the others. It permits native and external backends, and different
parcel-domain manifests; these are *descriptive contrasts only*. It **does
not** produce frontiers, rankings, or claims about scientific equivalence.
Matching name strings alone are not proof that models represent the same
watershed, outlets, nutrient constituents, baselines, forcing or uncertainty.

An analyst who has **independently verified** that two or more runs share the
same scientific model, domain, forcing and outcome definitions, and differ
only in meaningful management strategies, may opt into Pareto analysis:

```json
{
  "schema_version": 1,
  "primary_target": "Outlet 1 TP",
  "comparison_mode": "comparable",
  "scientific_comparability_confirmed": true,
  "scientific_model_id": "validated-annual-model-v1",
  "strategies": [
    {"name": "Current strategy", "ensemble_result": "results/current.json"},
    {"name": "Wetland strategy", "ensemble_result": "results/wetlands.json"}
  ]
}
```

`comparable` mode *also* requires identical `backend` values and an identical
**complete** BMI index manifest, including ordered parcel IDs, pollutants,
outlets, pathways, and supported BMP identities. A model ID is an
**analyst-provided assertion**, not a verified model/version fingerprint.
No software check establishes scientific equivalence across external backends;
frontiers are deliberately refused for mixed backend types or different
manifests.

For each target, the report gives two non-dominated sets of **strategy means**:

- **Mean implementation cost vs mean signed annual mass removed**, minimizing
  cost and maximizing annual mass reduction.
- **Mean implementation cost vs empirical target-attainment probability**,
  minimizing cost and maximizing probability.

A strategy is dominated only if another strategy costs **no more** and
performs **at least as well**, with at least one strict improvement. Exact
ties are both non-dominated. These are *descriptive* frontiers of estimated
means, **not** robust/uncertainty-aware Pareto optimization or significance
tests. No confidence intervals for *differences between strategies* are
calculated. There is no automatic matching/paired inference, **even when two
ensembles happen to share worker seeds**. Differences in scenario count are
allowed; each strategy's probability uses its own denominator.

## Scientific caveats

- All mass quantities are **signed annual outlet mass removed (kg)**, not
  absolute loads, concentrations, compliance with water-quality standards,
  or a multi-year time series.
- Costs are **total BMP implementation USD**, not annualized or discounted
  lifecycle costs; USD per positive annual kg removed is a proxy.
- Attainment probability is conditional on the model, scientific assumptions,
  input data, and uncertainty distributions, not a regulatory guarantee.
- Wilson intervals represent only finite-sample Monte Carlo precision under
  independent realization assumptions; they do not capture structural model
  error or imply statistically significant strategy differences.
- Before enabling comparable mode, verify the BMP strategies share the
  relevant baseline, forcing, geography, outlet definitions, pollutant units,
  assessment period, and cost conventions. BMI metadata alone does not prove
  these conditions.
- Chart appearance may be cluttered for very large strategy sets. Chart lines
  sample thresholds for display only; report probabilities use exact scenario
  counts with no resampling.

## Validation, publication and reproducibility

All source ensembles must be complete and internally consistent with their
Phase 12/13 seed schedule, index manifest and original scenario summaries.
Unknown, duplicated, or missing strategies and mismatched target identities
are rejected before output publication. Input files are **read-only** and
output destinations cannot equal an input/configuration file. Comparisons
store source-file SHA-256 digests so exact inputs are traceable. Repeated
execution with unchanged inputs produces byte-identical report and SVG files.

Reports and SVGs are each published by same-directory atomic replacement.
**They are not published as one multi-file transaction**: an I/O failure
while publishing a later SVG can leave earlier outputs updated, though
scientific source files remain unchanged. The ensemble JSONs are fully read
into memory; large ensembles may require substantial RAM.

## Testing

`tests/test_phase15_strategy_comparison.py` covers signed negative values,
probabilities and budgets, explicit-Pareto requirements, dominated/equal and
cost-vs-benefit tradeoffs, mixed backends, mismatched manifests, corrupt
sources, XML escaping, repeatability, CLI execution, source immutability, and
report preservation on validation failure. The original numerical golden and
Phase 2–14 regression tests remain authoritative for the scientific engine.

## Follow-on analysis

Phase 16 adds optional uncertainty-aware *pairwise* comparisons using
independent or explicitly confirmed paired bootstrap designs. The Phase 15
report and SVGs remain unchanged. See
[Phase 16 — strategy uncertainty](phase16_strategy_uncertainty.md).
