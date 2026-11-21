# Phase 16 — uncertainty-aware comparison of stochastic BMP strategies

## Scope and scientific interpretation

This is **optional post-processing** of completed model-neutral Phase 12/13
ensemble JSON files. It does not rerun the watershed model, change BMP designs,
change the stochastic seed schedule, or modify any existing ensemble, replay,
checkpoint, Phase 14 analysis, or Phase 15 comparison file.

It extends Phase 15 with *pairwise* confidence intervals for differences in:

- Mean implementation cost, in USD.
- Mean **signed annual outlet pollutant mass removed**, in kg.
- Probability of meeting each specified minimum annual mass-removal target.
- Probability of simultaneously meeting all targets.
- When a budget is configured: probability of being within budget and
  probability of both meeting all targets and being within budget.

The report also estimates how frequently strategy B would outperform strategy A
on **one randomly drawn realization**. This is a distinct quantity from a
confidence interval on their difference in *ensemble means*.

**All differences are `B - A`.** Higher mass removal and target probabilities
favor B; lower implementation cost favors B. Negative pollutant-removal values
are retained (the BMP strategy may increase load). A target of zero means
removal `>= 0`, *not* necessarily a positive reduction.

These intervals quantify **finite Monte Carlo sampling precision** conditional
on the scientific model and input distributions. They **do not** quantify
structural model discrepancy, calibration error, data uncertainty, external
validity, or the probability that one policy is scientifically better.

## Required comparability and design declarations

Phase 16 refuses a Phase 15 comparison in `descriptive` mode. You must first
assert a scientifically comparable set of runs using the existing Phase 15
fields:

```json
{
  "schema_version": 1,
  "primary_target": "Outlet 1 TN",
  "comparison_mode": "comparable",
  "scientific_comparability_confirmed": true,
  "scientific_model_id": "reviewed-science-and-domain-v1",
  "strategies": [
    {"name": "Baseline", "ensemble_result": "results/baseline.json"},
    {"name": "Expanded buffers", "ensemble_result": "results/buffers.json"}
  ]
}
```

As in Phase 15, `comparable` mode requires the same backend type and exactly
the same full BMI index manifest (pollutants, outlets, parcels and other domain
indices). These checks are **necessary but insufficient**: the analyst must
also verify the same scientific equations, inputs/forcing distribution,
model-domain definitions, baseline and costs/uncertainty conventions. The
`scientific_model_id` remains an analyst assertion, not a verified fingerprint.

Phase 16 additionally requires one of two explicitly confirmed sampling
designs (a separate `uncertainty.json` file):

**Independent ensembles:**

```json
{
  "schema_version": 1,
  "sampling_design": "independent",
  "sampling_design_confirmed": true,
  "n_bootstrap": 2000,
  "confidence_level": 0.95,
  "random_seed": 91
}
```

This resamples scenario realizations **independently within each strategy**.
The software rejects overlapping *worker seeds* across the supplied
ensembles, to guard against accidentally treating shared random streams as
independent data. Different seeds alone do **not** prove probabilistic
independence: the analyst must verify the actual stochastic design. Scenario
counts can differ. There must be at least two realizations per strategy.

**Paired / verified common-random-number ensembles:**

```json
{
  "schema_version": 1,
  "sampling_design": "paired",
  "sampling_design_confirmed": true,
  "pairing_id": "documented-aligned-input-draws-v1",
  "n_bootstrap": 2000,
  "confidence_level": 0.95,
  "random_seed": 91
}
```

This resamples **the same scenario indices for both strategies** within each
bootstrap replicate, retaining the correlation induced by their shared
underlying uncertainty draws. The software requires matching ordered scenario
IDs and worker seeds, equal scenario counts, and a nonempty `pairing_id`.
These are *metadata checks only*. Matching random seeds cannot establish
actual pairing, because strategy-specific random draws can branch differently
and may consume random numbers in a different order (for example, BMP costs or
AFO recovery). **Only use `paired` if you intentionally designed, verified,
and documented the alignment of the scientific uncertainty realizations.**

If two ensembles have overlapping worker seeds but no verified matched-draw
design, **neither independent nor paired inference is justified by metadata**.
Regenerate with verified independent streams, or redesign them to support
actual paired common-random-number sampling. No automatic pairing occurs.

## Run after Phase 15

Your Phase 14 `decision_support.json` supplies the same named pollutant/outlet
mass targets and optional total implementation budget to each strategy.

Run:

```bash
python -m src.driver.strategy_uncertainty \
    strategies.json \
    decision_support.json \
    uncertainty.json \
    outputs/strategy_uncertainty.json \
    --chart outputs/strategy_uncertainty_intervals.svg
```

The optional chart shows **B-minus-A differences in mean signed annual mass
removed**, with percentile intervals for the primary target. The dashed line
marks zero difference. It supports up to 60 strategy pairs to avoid unreadable
SVGs; omit `--chart` for larger sets. The original Phase 15 SVGs and JSON are
not modified.

The uncertainty config accepts the required `schema_version`,
`sampling_design`, `sampling_design_confirmed`, optional `pairing_id` (paired
only), `n_bootstrap` (integer 200–50,000; default 2,000), `confidence_level`
(finite number from 0.5 to less than 1; default 0.95), and `random_seed`
(unsigned 64-bit integer; default 0). Unknown fields and ambiguous design
choices are rejected before publication.

## Output interpretation

For each pair, the `difference_b_minus_a` is computed directly from the
original scenario arrays. `bootstrap_percentile_interval` is the specified
central percentile interval of resampled differences. `bootstrap_fraction_favoring_b_with_half_ties`
is the fraction of bootstrap *mean differences* in the favorable direction,
awarding half credit for exact ties. **This is not a Bayesian posterior
probability or a calibrated probability that a management strategy is better.**

`single_realization_probability_b_greater_with_half_ties` for mass reduction,
and `single_realization_probability_b_cheaper_with_half_ties` for costs,
answer a separate question: the empirical chance of winning for one draw.
With independent ensembles this compares **all cross-product pairs** of draws
without assuming matched indices. With a paired design it compares only
**matched pairs**. Exact ties each count as one half. These are conditional
empirical probabilities, not model-validation measures.

The bootstrap uses a seeded NumPy random generator, percentile intervals,
and bounded-memory resampling batches. Different seeds generally lead to
slightly different interval endpoints; fixed inputs/configuration/seed produce
byte-for-byte repeatable output. Confidence limits are **not adjusted** for
multiple strategies, targets or comparisons. With small scenario counts, skew,
rare events or probabilities near 0/1, percentile intervals may be too narrow
or have poor coverage; this tool does not offer BCa, clustered/block bootstrap,
importance weighting or formal hypothesis tests. Do not interpret a single
unadjusted interval excluding zero as a definitive decision rule.

Uncertainty output includes each exact ensemble-source SHA-256 digest, the
scientific model assertion, target definitions, selected design and `pairing_id`,
resample count, seed, level, and calculation/assumption notes. No model
calibration or inferential conclusions are inferred from these fields.

## Validation and reproducibility

- Original ensemble files, scenario-definition files and checkpoints are
  **read-only**. The report and optional SVG are newly generated files.
- The report refuses to overwrite any input ensemble or configuration file,
  or to use the same destination for the report and SVG.
- JSON and SVG are each written through an atomic, same-directory replacement;
  **they are not a single multi-file transaction**.
- Phase 14's complete-ensemble and seed-schedule checks run before analysis;
  mismatched model backends or index manifests fail validation.
- Test using `python -m pytest tests/test_phase16_strategy_uncertainty.py -v`,
  followed by `python -m pytest tests` in the project's fully equipped Python
  environment. Existing numerical-golden and replay tests remain authoritative
  for model calculations.
