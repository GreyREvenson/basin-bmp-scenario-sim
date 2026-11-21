# Architecture status after Phase 18

The central separation-of-concerns refactor is largely achieved. Scientific
PLET/RUSLE and statistical assemblies, BMP response processes and routing,
scenario sampling/replay, driver/results, formulation selection and versioned
BMI exchanges now have distinct package responsibilities. Model-neutral
annual contract, independent component startup, capability negotiation and
configurable external backends support opted-in BMI coupling. Serializable
ensembles, deterministic seeds, parallel checkpoint/restart and exploratory
decision-support tools have been added in later phases.

Phase 18 addresses a gap in Phase 17 by exporting a realization-aligned BMP
input diagnostic trace and detailed ledger directly from **saved** scenario
replay definitions. No scientific/RNG code or existing result schema changes.
This records actual BMP parameter realizations but **does not record unsaved
baseline scientific input draws**, nor does it implement Sobol designs.

## Remaining architectural completion criteria

1. **Standards interoperability:** verify the public BMI variable/time/grid
   interface with a real third-party BMI harness and independent model; fix
   deviations without weakening domain/unit identity checks.
2. **True model-neutral science contracts:** specify and test cross-model
   semantics for BMP responses, process compatibility, AFO/manure and outlet
   routing rather than merely matching array shapes or CPS names. Scientific
   equivalence between different models must never be assumed.
3. **Remove residual private coupling:** reduce adapter dependence on native
   contexts, model-specific scenario serialization and legacy compatibility
   helpers. Keep a controlled deprecation path for stable imports.
4. **Full uncertainty input provenance:** add named, scoped draw logging for
   baseline hydrology, concentrations, erosion, AFOs and costs; distinguish
   fixed parameters, draws, model outputs and derived diagnostics. Add
   designed experiments only after this provenance is validated.
5. **Production integration:** build end-to-end cross-backend tests, real
   process-parallel/Windows checks, full native Parquet/equivalence tests,
   failure and output schema compatibility tests, and a published interface
   stability/versioning policy.

The refactor does not need more decision-support features to be considered
architecturally mature. A practical completion milestone is independently
initializing, running and validating a genuinely separate watershed backend
through the versioned public contracts, with rigorous compatibility tests and
no dependence on internal PLET Python objects.
