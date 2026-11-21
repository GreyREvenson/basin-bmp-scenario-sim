# Phase 21 — Annual BMI public-contract freeze and black-box audit

## Purpose

Phase 21 adds a **machine-checkable compatibility gate** to the existing public
annual-model coupling API. It does **not** change the stochastic simulation,
scientific formulations, RNG order, scenario definitions, checkpoint files,
or the Phase 14–18 decision-support reports.

The executable public contract is `src/interfaces/bmi/public_contract.py`,
with a deliberately frozen JSON snapshot at
`src/interfaces/bmi/public_contract_v1.json`. The snapshot is included in the
installed Python package. It records, in declared order, the published BMI
variable names, intent, data type, units, rank, resizable-input designation,
CSDMS BMI 2.0 method names, and the separately versioned local protocol.

The contract covers four distinct roles:

| Role | Adapter(s) | Scope |
| --- | --- | --- |
| `annual_model` | `WatershedModelBmi`, `ExternalWatershedModelBmi` | Annual loads and live BMP response |
| `live_generator` | `LiveScenarioGeneratorBmi` | Pending BMP realization, recovery/cost exchange, replay export |
| `recorded_scenario` | `StochasticScenarioBmi` | Read-only saved-scenario stream |
| `monte_carlo_driver` | `MonteCarloDriverBmi` | Native scenario orchestration and diagnostics |

An independent annual watershed model need only implement the `annual_model`
role, together with the backend capability declaration and published extension
methods. It does not inherit a PLET class or import a native model context.

## Commands

From the repository root:

```bash
# Verify the published API still matches the frozen v1 contract.
python -m src.interfaces.bmi.public_contract

# Inspect the current public contract, with variable order and units.
python -m src.interfaces.bmi.public_contract --show

# Test a complete independent backend through the public interface.
python -m src.interfaces.bmi.acceptance backend_selection.json acceptance_report.json

# Run the contract and interoperability regression tests.
python -m pytest tests/test_phase21_bmi_public_contract.py tests/test_phase19_public_interoperability.py

# Optional: compare the adapter method catalog to the official bmipy interface.
python -m pip install -e ".[bmi]"
python -m pytest tests/test_phase7_bmipy_optional.py -v
```

`accept_backend` now runs the strict public-variable audit before the existing
black-box scenario/response/reproducibility checks. The audit checks only
public BMI getters and does not consume RNG draws, advance logical time,
write a file, or change an exchange variable. It tests:

- advertised input/output sets and required project extension methods;
- exact declared names, dtype, units, scalar/vector rank, and variable intent;
- byte counts, scalar shape and virtual 1-D index-space grid shape;
- index coordinates (`origin=0`, `spacing=1`) rather than fake geospatial coordinates;
- `get_value`, `get_value_ptr`, and indexed reads against each other;
- finite, monotonic logical-time metadata with unit `none`.

Extra BMI variables are permitted, provided they obey the existing 1-D
index-space contract. An audit failure raises a `ValueError` and prevents the
acceptance report from being published. Existing acceptance-report JSON keys
remain unchanged.

## Compatibility and version policy

1. **Stable public surface:** The v1 snapshot is the published contract for
   the four roles above. Changes to private modules, PLET scientific kernels,
   or non-BMI diagnostic writers do not require a contract change.
2. **Additive changes:** Adding a declared BMI variable, extension method, or
   optional capability still requires a reviewed snapshot update and tests.
   Compatible extensions should preserve existing variable meaning and order.
3. **Breaking changes:** Renaming a variable, changing units/dtype/rank/intent,
   changing ordering semantics, altering the logical timestep meaning, or
   changing the two-stage BMP response protocol requires a documented
   migration and an appropriate increment to the BMI exchange or lifecycle
   schema. Retain older adapters or provide an explicit converter if
   backward compatibility is required.
4. **Identity and interpretation:** Parcel, pollutant, pathway, outlet, and
   CPS ordering remain governed by the hashed index manifest. These virtual
   index coordinates are not real-world geographic coordinates. `get_time_units`
   returns `none` because a BMI update is a *logical placement/scenario step*,
   not a simulation year.
5. **Independent validation:** In addition to the frozen static contract,
   acceptance tests should run against at least one *external* implementation
   without accessing private contexts. The reference backend is a contract
   test fixture, not a scientifically validated replacement for PLET/RUSLE.

Do **not** rewrite the old snapshot in response to a failing test without
first investigating the compatibility impact and updating the versioned
migration documentation. A new snapshot alone does not make an incompatible
scientific or BMI implementation valid.

## Scope and remaining work

This is a project-level public annual-BMI contract and **structural/runtime
conformance audit**, **not CSDMS certification**. In particular, the project
has BMI-compliant *method names*, but it also needs project-specific
`accept_domain`, `complete_pending`, and `export_replay` extensions to couple
interleaved BMP/AFO calculations. Grid metadata describes virtual index space
only; the model does not expose physical geospatial grids or physical-time
semantics through BMI. The optional official `bmipy` abstract-method test is
skipped when `bmipy` is not installed.

Still outstanding for an architecture-complete release: exercise the official
BMI dependency in CI, test Windows `spawn` parallel ensembles end-to-end,
finish or explicitly isolate the native-specific diagnostic-output path,
and publish a supported external-backend developer example and release policy.
