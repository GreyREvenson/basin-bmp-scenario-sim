# Phase 20 — Versioned annual BMI component lifecycle

## Purpose and scope

The annual BMI mediator now negotiates an explicit **project extension protocol**
(version `1`) before importing the domain or sampling any BMP. The CSDMS BMI
2.0 interface itself has no concept of a two-stage BMP response, a stochastic
physical-domain import, or a completed replay definition. Those are **local
extensions**, not CSDMS Standard Names or methods. A separate protocol version
avoids conflating lifecycle changes with the numeric BMI exchange schema
(currently version `2`).

Phase 20 also removes the file-startup coupler's introspection of private
`model._ctx` and `generator._ctx`. Both component owners continue to load their
own project inputs, and the mediator interacts only through public methods,
versioned extensions, and BMI variables. The prepared-context constructor
retains its `ctx`, `cfg`, and `logger` references for backward compatibility.

This phase **does not** alter native PLET/RUSLE/scientific calculations, BMP
placement selection, the AFO-recovery-before-cost RNG handoff, replay contents,
or the existing CLI and ensemble result schemas.

## Negotiated protocol, v1

At coupled initialization, both the generator and watershed model publish:

| Output | Type | Interpretation |
| --- | --- | --- |
| `coupling__lifecycle_schema_version` | scalar `int64`, unit `1` | Must be `1` on both components |

The generator additionally publishes:

| Output | Type | Interpretation |
| --- | --- | --- |
| `scenario__replay_utf8` | variable-length `uint8` vector, unit `1` | UTF-8 JSON containing the **complete** `BMPScenario` schema-v1 definition. Initially empty; populated by `export_replay(seed)`. |

The mediator validates both components' published version, dtype, units,
rank, and the required extension callables before transferring the parcel
domain. It rejects mismatches before any BMP sample is drawn. It also
validates the replay payload's JSON/schema and the scenario ID and seed.

Required *public project extensions* beyond standard BMI 2.0:

1. Generator `accept_domain()`: consume the complete received physical-domain
   BMI input arrays before the first `update()`.
2. Generator `complete_pending()`: after model recovery gains have arrived,
   realize cost and conclude the pending placement.
3. Model `complete_pending()`: after generator cost has arrived, finalize
   scientific mass accounting and expose the new BMI outputs.
4. Generator `export_replay(seed)`: publish the completed scenario JSON into
   `scenario__replay_utf8` without sampling further random numbers.

The mediator does **not** use a generator-native `BMPScenario` object for replay
transfer. The older `generator.completed_scenario(seed)` remains available to
existing Python callers, but is not called by the mediator.

The mediator exposes a `lifecycle_phase` read-only property with logical states
`created` → `ready` → `exhausted` → `finalized`. Finalization is safe to repeat.
An initialization failure after one or both components have started triggers
best-effort cleanup and marks the coupler finalized rather than leaving a
half-initialized coupling ready for use.

`exhausted` is entered after `NoEligiblePlacements`; it is safe to call
`next_bmp()` again (returns `None`). Replay export remains available in both
`ready` and `exhausted` states, preserving the legacy ability to inspect a
partially generated live scenario when there is no pending placement.

## Migration and compatibility

- **Native CLI and native diagnostics:** unaffected, including legacy
  `basin_bmp_sim.execution.monte_carlo_driver` imports and the native context used by result writers.
- **Startup manifests and backend-selection JSON:** unchanged.
- **Replay JSON schema:** unchanged. The BMI byte output is only a transport.
- **Independent backend adapters:** `ExternalWatershedModelBmi` publishes the
  lifecycle version by default, so the shipped reference backend works
  without modifications.
- **Third-party BMI generator/model adapters:** an adapter previously exposing
  the Phase 19 annual numeric surface but **not** these new lifecycle
  declarations must add the version and generator replay extension. Otherwise
  the coupler deliberately fails during preflight; there is no silent fallback
  to private Python types.
- **Numeric exchange schema:** remains `2`; this is an additive public variable
  and a separately versioned lifecycle extension, not a reordering of
  pollutant, parcel, pathway, or outlet arrays.

## Verification

Tests cover negotiation mismatch, missing extensions, cleanup after failure,
public replay validation and idempotence, preservation of generator and model
RNG states during replay export, file-startup context isolation, native golden
results, and reference-backend end-to-end integration.

## Still to complete

This is **not official CSDMS BMI interoperability certification**. In
particular, the extensions above are not standard BMI methods, and logical
placement counts remain the BMI time coordinate rather than physical years.
The native Monte Carlo result writer still accesses native scientific
objects for PLET-specific diagnostics. A real third-party scientific model,
Windows multi-process execution, optional `bmipy` conformance, and a
public deprecation/stability policy still require separate validation.
