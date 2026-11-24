# Phase 19 — Independent annual backend acceptance milestone

## Scope

This phase introduces a **file-configured, importable external scientific backend**
(`reference_backend.annual_linear:make_backend`) and a **public-BMI-method
acceptance probe** (`basin_bmp_sim.certification.acceptance`). It exercises the already
published Phase 10–13 external backend path without importing any private
PLET/RUSLE, AFO, Monte Carlo driver, or scenario-generator context into the
reference scientific backend. It does **not** alter scientific methods,
random-number generation, existing native or external result schemas, replay
format, or the default CLI.

**The reference model is intentionally not a calibrated/validated watershed
model.** It demonstrates the contract and is not evidence that a real external
hydrology implementation is scientifically compatible with the native model.

## External reference backend

The factory is selected in the normal Phase 11 backend-selection JSON:

```json
{
  "schema_version": 1,
  "backend": "external",
  "component_manifest": "startup.json",
  "external_factory": "reference_backend.annual_linear:make_backend",
  "backend_options": {"domain_file": "reference_domain.json"},
  "required_features": ["signed_efficiency", "multi_parcel_treatment"]
}
```

`domain_file` resolves relative to the **startup model YAML directory** (or
is an absolute path). It is owned by the external backend and is never read
by the generic coupler. It must match the generator's ordered identities,
selected CPS codes, and complete parcel domain. Its strict version 1 schema
consists of the following fields:

- `schema_version`: integer 1.
- `parcel_ids`, `outlet_oids`, `pollutants`, `pathway_names`, `bmp_cps`:
  ordered identity lists matching the generator.
- `parcel_area_ha`, `parcel_perim_m`: per-parcel physical geometry.
- `upstream_offset`, `upstream_index`: immediate-upgradient CSR graph.
- `selection_source_idxs`, `parcel_selection_probs`: ordered eligible hosts
  and their normalized selection probabilities.
- `baseline_rates_kg_ha_yr`: matrix `[parcel, pollutant]`, finite nonnegative.
- `pathway_weights`: matrix `[pollutant, pathway]`, finite nonnegative,
  each pollutant's row sums to one.
- `outlet_delivery_fraction`: matrix `[parcel, outlet]`, finite nonnegative;
  each parcel's row sum is <= 1. These are *assumed* routed mass fractions,
  not derived from hydrologic or stream transport equations.

For an affected parcel `i` and pollutant `p`, the reference law evaluates
signed annual mass-rate removal as

```
current_rate[i,p] * treatment_fraction[i]
    * sum(pathway_weights[p,k] * efficiency[p,k] for k)
```

The current rate is updated sequentially for BMP stacking. A failed practice
produces zero change. Each signed rate difference is multiplied by the parcel
area and the configured outlet fractions to obtain outlet removed kg per year.
Negative efficiency yields a negative reduction, **never clipped to zero**.
The backend does not model runoff, sediment routing, manure production, AFO
recovery, nitrogen/phosphorus conservation across different media, or physical
wetland retention. It therefore advertises only `signed_efficiency` and
`multi_parcel_treatment`, **not** `afo_recovery` or `upstream_routing`.
The generic linear response to configured CPS codes is an interface exercise,
not a practice-specific scientific claim.

For native scenario generation, the model YAML and startup manifest still
follow their existing Phase 9 formats. The reference backend does not infer
those inputs, and a domain with mismatched ordering fails at the BMI index
contract check before any BMP executes.

## Runnable black-box acceptance check

```bash
python -m basin_bmp_sim.certification.acceptance \
    external_selection.json outputs/bmi_acceptance.json
```

This runs a **single-scenario selection twice**, initializing fresh components
and checking only public BMI methods for each adapter:

- input/output names, types, units, scalar/vector ranks, array sizes,
  `get_value`, `get_value_at_indices` and virtual index-grid shape;
- successful BMI-to-BMI generator/model coupling and scientific output shape;
- cumulative signed outlet mass returned by BMI versus driver-facing results;
- reproducible final load snapshots and replay definitions (SHA-256 hash).

The JSON report records the check scope, both components' variable catalogs,
placement count, deterministic-repeat flag, and final-result digest. The
probe is **read-only** apart from ordinary model execution. It does not inspect
or depend on private component state. A failing validation does not publish a
partial report. Version 2/3 ensemble selections are intentionally rejected by
this command because it probes a *single component realization*; ensemble
execution and checkpointing are tested separately.

**This is a project-level public-interface acceptance test, not official CSDMS
BMI conformance certification.** The annual pseudo-time is a placement count,
not physical time. The index-space grids are not geospatial coordinates and
project-local variable names are not CSDMS Standard Names. The existing
`complete_pending()` and replay APIs also remain project-specific extensions.

## Verification tests

`tests/test_phase19_public_interoperability.py` runs the importable backend
through the actual external-selection CLI route, Phase 12 serial ensembles,
Phase 13 checkpoint-and-resume, positive and negative signed mass accounting,
strict JSON validation, incompatible index ordering, and deterministic public
BMI probes. Tests generate their matching generator fixture separately; the
reference backend itself receives **only a validated file** through its
published factory arguments.

## Remaining completion milestones

1. Run this acceptance path with an **actual third-party model** adapted
   without native Python-context imports, and evaluate its response semantics,
   including AFO/manure and wetland routing where supported.
2. Exercise the optional official `bmipy` conformance comparison in a
   dependency-complete environment, then address documented BMI deviations.
3. Replace remaining project-specific method calls (including
   `complete_pending`) with formally versioned extension protocols where
   genuinely independent third-party components require it.
4. Validate full native/external process-parallel execution (including on
   Windows), schema evolution, and full Parquet output regression suites.
5. Define a deliberate deprecation/compatibility plan for legacy imports
   and scientific-context adapters, rather than removing them abruptly.

Additional sensitivity/visualization features are **not required** to close
the core architectural refactor.
