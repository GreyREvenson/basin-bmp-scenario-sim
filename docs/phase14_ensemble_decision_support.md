# Phase 14 — Model-neutral ensemble decision support

## Scope and scientific interpretation

Phase 14 is an **optional post-processing layer** for the completed
Phase 12/13 model-neutral BMI ensemble **result schema version 2**. It adds no
random draws and **does not change** the native Monte Carlo CLI, model
formulations, BMP selection, AFO mass accounting, seed policy, replay,
checkpointing or the existing ensemble output JSON. No scientific model is
initialized when analyzing a previously completed result.

The Phase 11–13 model-neutral interface publishes the *signed annual mass
removed at each modeled outlet* (`outlet_removed_mass_kg`), ordered as
`[pollutant, outlet]`. A positive value represents modeled reduction relative
to baseline, a negative value an increase. **It does not provide baseline
absolute outlet mass or concentration.** Phase 14 therefore supports only
**minimum signed annual mass-removal targets in kilograms**, not attainment of
absolute water-quality standards, loading limits, concentration thresholds,
or time-series compliance. A zero target is attainable with exactly zero
removal; negative removals never count as having met a nonnegative target.

The proportion of simulated realizations meeting a target is an **empirical
Monte Carlo probability conditional on the model, forcing, practices,
uncertainties and selected backend**. A Wilson 95% binomial interval
quantifies finite-sample Monte Carlo precision *if realizations are
independent*; it is **not** a scientific-model uncertainty bound, calibration
interval, regulatory confidence level or guarantee of attaining a target.
External backend capability negotiation does not imply cross-model scientific
equivalence.

The reported cost-effectiveness ratio is **implementation USD divided by the
modeled annual kg of removal**, evaluated separately for each realization.
It is not a discounted lifecycle cost, recurring cost, or marginal abatement
cost curve. Ratios are **undefined** for zero or negative removal and are
excluded only from the conditional ratio distribution; all scenarios remain
included in mass-reduction and target probabilities. The report gives counts
of excluded ratios. Cost comparisons use the *same total implementation
cost* associated with each realization; no annualization or lifetime
benefit is inferred.

## Configuration and usage

Create `decision_support.json`:

```json
{
  "schema_version": 1,
  "targets": [
    {
      "name": "Outlet 1 total phosphorus reduction",
      "pollutant": "TP",
      "outlet_oid": "outlet-1",
      "minimum_removed_mass_kg": 500.0
    },
    {
      "name": "Outlet 1 total nitrogen reduction",
      "pollutant": "TN",
      "outlet_oid": "outlet-1",
      "minimum_removed_mass_kg": 1000.0
    }
  ],
  "budget_usd": 100000.0
}
```

**Replace** `TP`, `TN` and `outlet-1` with exact strings from your
ensemble's `index_manifest.pollutants` and `index_manifest.outlet_oids`.
These IDs are case-sensitive and validated. `targets` must be nonempty,
with unique names and nonnegative finite `minimum_removed_mass_kg`; unknown
fields and wrong JSON types are rejected. `budget_usd` is optional, finite,
and nonnegative. Omitting it removes budget-based statistics rather than
assuming zero cost.

### Analyze an ensemble already on disk (no model rerun)

```bash
python -m basin_bmp_sim.analysis.ensemble_analysis outputs/ensemble.json decision_support.json outputs/decision_support.json
```

### Generate an ensemble and analysis together

With an existing Phase 12/13 `--backend-selection` (schema v2/v3):

```bash
basin-bmp-sim \
  --backend-selection ensemble_selection.json \
  --coupled-result outputs/ensemble.json \
  --analysis-config decision_support.json \
  --analysis-result outputs/decision_support.json
```

The two analysis arguments are both required when either is supplied, and
are rejected for single-scenario backend selections or for native YAML runs.
The configuration is validated *before* ensemble execution. The scientific
ensemble is published normally, then the separate analysis is computed and
published. If analysis fails after a successful simulation, the ensemble
result is still valid and preserved; there is **no atomic two-file
transaction**. Use the post-processing command to fix/retry analysis without
rerunning science. Resume directories and replay definitions are read-only to
this analyzer.

## Result schema and definitions

Analysis JSON is **schema version 1**, separate from the unchanged ensemble
result schema v2. It records a SHA-256 digest of the exact input ensemble
file; backend, seeds and index manifest; scenario count; distribution of
implementation costs; optional budget outcomes; outcomes for each target;
and the joint probability of attaining *all* targets.

- `attainment`: number and fraction satisfying `removed_kg >= target_kg`,
  with a Wilson 95% finite-sample Monte Carlo interval.
- `signed_removed_mass_kg`: across-scenario mean, population standard
  deviation, extrema and p05/median/p95. **Negative values are retained.**
- `cost_effectiveness`: ratio summaries only for strictly positive removal,
  with counts of valid and undefined observations. It can be `null` when
  *no* scenario has a positive removal.
- `within_budget` (optional): fraction with total implementation cost
  `<= budget_usd`, inclusive of equality.
- `attainment_within_budget` (optional): joint fraction satisfying a target
  **and** the budget, with the original all-scenario denominator.
- `conditional_attainment_given_within_budget` (optional): fraction satisfying
  target **among** within-budget scenarios; `null` when none qualifies.
- `joint_all_targets`: fraction satisfying every specified target in the
  *same realization*; optional `all_targets_within_budget` evaluates all
  targets *and* the budget jointly.

Percentiles use NumPy's linear interpolation and population standard
deviation, matching the Phase 12 aggregate convention. All event
probabilities use the full realization count, including scenarios with no
BMPs, negative pollutant removal, and costs of zero. A target meeting the
threshold *exactly* counts as attained.

The analysis requires a complete Phase 12/13 result with the original
per-scenario records present, and validates ordered pollutant/outlet IDs,
scenario IDs, the deterministic seed schedule, nonnegative finite costs,
outlet array dimensions and consistency of original scenario results with
compact summaries. It refuses a destination path equal to the ensemble's
source and writes the analysis JSON atomically by same-directory replacement.
This implementation loads the completed JSON into RAM; it does not analyze
partially written checkpoint directories or implement streaming aggregation.

### Comparing different backends

The same `decision_support.json` can be applied separately to native and
external ensembles **only when the declared pollutant/outlet identities and
units correspond**. Differences in reported probabilities are conditional on
models that may have different baseline definitions, structural biases,
uncertainties and scientific scopes. The software does not treat these as
paired, calibrated or interchangeable scientific results, and does not rank
unvalidated backends.

## Verification

`tests/test_phase14_ensemble_decision_support.py` checks exact target and
joint event counts, signed negative/zero removal, budget conditioning,
undefined cost ratios, target/manifest errors, malformed metadata and
summaries, output atomicity and source provenance. It also runs selected
native and independent external BMI ensembles with analysis enabled, verifies
the ensemble's serialized bytes remain identical, and verifies Phase 12
serial versus Phase 13 checkpointed results produce the same analysis.
Existing Phase 1–13 numerical regression tests remain authoritative for the
scientific model.
