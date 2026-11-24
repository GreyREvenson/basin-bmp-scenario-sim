# Phase 17 — Realization-aligned input sensitivity diagnostics

## Scope and scientific validity

Phase 17 adds **optional, post-processing** diagnostics on a completed model-neutral
Phase 12/13 BMI ensemble. It accepts a **separately captured input/diagnostic
trace** with exactly one row per scenario. It does not change the native model,
scenario generator, BMI exchanges, random seeds, replay files, checkpoints, or
Phase 14–16 analysis results.

The ensemble result retains costs and signed annual outlet mass reductions, but
**does not systematically record every sampled uncertain scientific parameter**.
Thus a scientifically defensible input-sensitivity study cannot reconstruct
all the necessary parameter values from the saved ensemble alone. Phase 17
**does not invent them**. The analyst must record and supply the actual
realization-specific values externally; in particular, do not fill them in
from distributions or a replay file unless they are provably the values
used for that specific scenario.

Phase 17 calculates **exploratory, unconditional Spearman rank correlations**
for each documented feature with:

- The implementation cost in USD.
- Signed annual mass removed at each configured pollutant/outlet target, in kg.
- The corresponding *binary* annual mass-removal target-attainment indicator.

Spearman correlation uses average ranks for ties and lies in [-1, +1].
A negative sign for mass removal is retained; the analysis does not treat all
BMPs as beneficial. The result is **null**, not zero, when an input or outcome
is constant. An event with all successes or all failures also has undefined
rank correlation. Pairwise input/input rank correlations are included so the
analyst can recognize collinear sampled factors.

**This is not uncertainty attribution or a global variance decomposition.**
Spearman rho is not a Sobol index, an independent parameter effect, a fraction
of output variance, an explained-variance measure, or a causal coefficient.
Variables may be correlated; multiple parameters may vary together; BMP design,
selection and cost decisions can respond to other sampled values. The
`realized_diagnostic` feature role explicitly permits downstream quantities
that should **never** be called independent input effects. In particular,
**do not rank scientific causes solely by absolute correlation magnitude**.
Future formal uncertainty-attribution work will need explicit variance-based
experimental designs (and verified input trajectories), not simply more
postprocessing of unconstrained stochastic realizations.

Bootstrap percentile intervals quantify precision of each empirical association
across the existing scenario draws. They do **not** include input uncertainty
beyond the assumed sampling distribution, model structural error, or
observation uncertainty. No confidence interval or multiple-comparison
correction constitutes proof of a causal effect. Bootstrapping resamples
entire aligned input/output rows together; it never independently shuffles
input and output values. This assumes independent, identically distributed
realizations. Conditional sampling/importance weights, dependent simulation
runs, adaptive ensemble designs, and time-correlated runs require other methods.

## 1. Record actual features for each realization

Supply a JSON file matching schema version 1. The example values below are
**illustrative only**; replace both values and identity fields with the values
actually recorded during each BMI scenario. They must correspond exactly to
the `scenario_summaries` rows in the ensemble, in order.

```json
{
  "schema_version": 1,
  "sampling_design": "independent",
  "sampling_design_confirmed": true,
  "trace_provenance": "Versioned sampling script and run log, with source commit and variable definitions",
  "features": [
    {"name": "precipitation_multiplier", "units": "1", "role": "sampled_input"},
    {"name": "baseline_n_kg_ha_yr_multiplier", "units": "1", "role": "sampled_input"},
    {"name": "number_of_bmps_selected", "units": "count", "role": "realized_diagnostic"}
  ],
  "realizations": [
    {"scenario_id": 100, "worker_seed": 123, "values": [0.9, 1.1, 4]},
    {"scenario_id": 101, "worker_seed": 456, "values": [1.1, 0.8, 6]}
  ]
}
```

In actual use there must be one row for **every** ensemble scenario, matching
both `scenario_id` and `worker_seed`. The trace requires an explicit assertion
that realizations are independent; matching IDs/seeds alone cannot verify
independence. If your ensemble has dependent or weighted draws, the Phase 17
IID bootstrap is not appropriate. The application validates identity fields
against the Phase 12/13 ensemble and also validates the ensemble's own seed
schedule and original scenario results. It checks finite, numeric values;
unique named features; and explicit units and roles. It rejects missing or
extra fields, non-finite values, booleans masquerading as measurements,
corrupted ensemble summaries and re-ordered trace rows. These checks cannot
independently verify that a logged input was truly used by the scientific
model: `trace_provenance` is an analyst statement.

Feature `role` may be `sampled_input` or `realized_diagnostic`. A diagnostic
could be selected BMP count, a realized application fraction, or another
quantity derived during execution. Such a diagnostic can be informative but
cannot be interpreted as a parameter intervention. **Do not use costs or
outlet reductions as sampled inputs to explain themselves.** When input data
are unavailable, the responsible result is to report that sensitivity cannot
yet be evaluated, rather than guessing reconstructed draws.

## 2. Specify pollutant targets

Use the existing Phase 14 `decision_support.json` definition. Example:

```json
{
  "schema_version": 1,
  "targets": [
    {
      "name": "Outlet 1 nitrogen",
      "pollutant": "TN",
      "outlet_oid": "1",
      "minimum_removed_mass_kg": 100
    }
  ]
}
```

The labels and outlet identifiers must match the BMI index manifest in the
ensemble. The signed reduction threshold is inclusive (`removed_kg >= target`)
and **not** an absolute concentration standard. The Phase 14 budget, if any,
is not used as a predictor; implementation cost is analyzed separately.

## 3. Configure bootstrap sampling

Create `sensitivity.json`:

```json
{
  "schema_version": 1,
  "n_bootstrap": 1000,
  "confidence_level": 0.95,
  "random_seed": 17,
  "primary_target": "Outlet 1 nitrogen"
}
```

- `n_bootstrap`: integer 200–5,000 (default 400).
- `confidence_level`: a finite number in [0.5, 1) (default 0.95).
- `random_seed`: unsigned 64-bit integer (default 0).
- `primary_target`: one of the named targets; defaults to the first target.

The bootstrap uses the same row indices for each feature/outcome within each
resample. Each feature/outcome pair uses its own deterministic slice of the
seeded generator. If fewer than 95% of resamples yield defined rank
correlations (e.g. due to rare binary attainment events), the interval is
`null`; the report gives counts of valid and undefined resamples. There must
be at least four realizations. With small ensembles, rare events or a large
number of candidate features, correlations and intervals are unstable; no
multiplicity adjustment is attempted.

## 4. Run the optional analysis

```bash
python -m basin_bmp_sim.analysis.input_sensitivity \
    outputs/ensemble.json \
    input_trace.json \
    decision_support.json \
    sensitivity.json \
    outputs/input_sensitivity.json \
    --chart outputs/input_sensitivity.svg
```

`--chart` is optional. It creates a deterministic, XML-escaped SVG showing
signed rank correlations with the **primary target's annual mass removal**.
The chart supports up to 40 features to remain legible; JSON supports 100.
Charts are labelled descriptive and deliberately omit claims of variance
attribution. The report also contains the cost, binary-attainment and
all-target associations.

The report embeds SHA-256 hashes for the complete input ensemble and trace,
the documented trace provenance, source roles/units, the seed and confidence
settings, complete association estimates and intervals, and methodological
limitations. The original files are read-only. The output JSON and optional
SVG are atomically replaced **independently** (not as a multi-file transaction),
and no output may collide with any input file.

## Future designed sensitivity studies

To estimate Sobol or other first/total-order variance contributions, build a
controlled experimental design, preserve input draw matrices and forcing,
execute the required model evaluations, and separately validate the assumptions
about independent inputs. For correlated variables or dependent scientific
inputs, use a suitable conditional or dependent-input sensitivity method rather
than reporting standard independent-input Sobol indices. Those designed model
experiments are **not** implemented in Phase 17.
