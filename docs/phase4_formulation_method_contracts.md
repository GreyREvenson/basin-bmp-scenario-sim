# Phase 4: formulation-backed scientific method dispatch

This phase applies to the `dev_arch_refac (3)` branch after Phase 3. **No new
scientific algorithm is introduced.** It establishes real callable registries
for process methods and BMP response methods, and uses the approved formulation
to select scientific implementations at runtime.

## Boundary and method selection

- `src/formulations/formulation_registry.py` remains the scientific approval
  authority. There are still exactly two approved formulations:
  `plet_rusle_annual_v1` and `statistical_annual_v1`.
- `src/formulations/formulation_builder.py` validates the worker's mode and
  config, checks that each formulation's methods exist in executable registries,
  and builds an immutable per-realization `ScientificMethodPlan`.
- `src/models/process_method_registry.py` binds PLET runoff, infiltration,
  RUSLE sediment, and PLET concentration-based loading to their actual callable
  kernels. `calculate_plet_pathway_load_rates` accepts an optional resolved
  process-method bundle; its legacy direct-call behavior is unchanged.
- `src/bmp/response_method_registry.py` resolves a CPS to its practice family
  and then to the approved response implementation. The scientific model no
  longer branches on wetland, grassed-waterway and in-field response families.
- The statistical assembly remains an approved, distinct baseline assembly and
  preserves its per-pathway sampling logic. All existing PLET/RUSLE and
  statistical BMP response methods remain unchanged numerically.

## Scientific compatibility policy

The new registries are *implementation catalogs*, not blanket approval to
combine methods. YAML may select an existing approved formulation and may
repeat its approved BMP response identifiers. Unknown and unapproved
identifiers still fail during input validation. In addition, formulation
preflight now checks that the approved identifiers have runnable code, before
Monte Carlo execution. No arbitrary per-process YAML overrides are supported.

New scientific alternatives must first be independently implemented and
tested, then explicitly reviewed and added to an approved formulation with
required-input, spatial, temporal, and mass-accounting compatibility checks.
The existence of a callable or matching function signature does *not* imply
scientific validity.

## Numerical and compatibility requirements

The phase preserves the original parameter sampling order and uses exactly the
same PLET/RUSLE kernels, BMP response functions, and annual mass accounting.
Run the repository-wide regression suite including the Phase 1/2/3 golden
fixtures and the new Phase 4 registry tests:

```console
python -m pytest tests/test_phase4_method_dispatch.py tests/test_architecture_result_equivalence.py tests/test_phase2_scenario_generation.py tests/test_phase3_model_driver_separation.py
python -m pytest tests
```

The model driver and its output subsystem remain responsible for orchestration
and writing. This is **not yet a CSDMS BMI implementation**. The next phases
will formalize external generator/model/driver BMI interfaces and data-transfer
contracts without changing the accepted scientific methods.

## Phase 4 patch verification

Eight seeded before/after scenarios were captured independently from the
untouched uploaded `(3)` ZIP and from this patch. Each snapshot has 98,530
JSON characters and identical SHA-256 digest:

`1a42d7b6607e4ab02940921e01cb12f515b82eaa9aadb2145a19fee2a52d3014`

This covers five prior architecture golden scenarios and three extended
placement/replay scenarios; all captured per-scenario tables and outlet
trajectories matched exactly. A scientific regression subset with AFO,
wetlands, PLET/RUSLE, statistical, and other modeling tests passed
(**275 tests, five nonfatal warnings**).

The entire repository suite could not be certified in this environment because
`pyarrow` and `python-calamine` are absent. The broad-suite failures observed
were dependency related. In the normal project environment, run:

```console
python -m pytest tests
```

Passing these comparisons does not certify every possible numerical input or
model option. No alternative scientific algorithm has been activated.
