# Phase 32: Simplify internal context and BMI interfaces

## Design decisions

- Replace the 21-method `ModelContextServices` interface with one six-operation
  `SamplingContext`, shared by the prepared, worker and detached generator
  contexts. Fixed response, routing, selection-probability and lookup functions
  are imported from their implementations and called directly. Tests for those
  operations use the actual state contract instead of binding methods onto mocks.
  Existing stochastic overrides remain deliberate test/extension seams.
- Keep distinct generator and scientific state and RNG ownership: merging them
  would break independently initialized BMI components and replay semantics.
- `AnnualBMPPlacement.from_bmi` is the sole validator of the common placement
  wire format. Native placement conversion consumes its validated request and
  adds only native practice-family checks. The encoder emits flat arrays using
  the canonical exchange-variable metadata, including the dtype of empty arrays.
  Negative efficiencies remain valid. Native callers that passed untyped scalar
  packets directly to the private decoder must now use the declared BMI arrays.
- Generic distribution/table validation has one canonical owner,
  `domain.statistics_validation`, not `config.validation.statistics`. AFO and
  configuration callers use the same implementation; rules are unchanged. No
  compatibility stub is provided for the previous internal import path.
- Restore BMI `src` keyword argument names and avoid duplicate RNG decoding.
  The public exchange variable names, metadata and versions are unchanged.

## Scope and verification

No scientific equations, distribution algorithms, RNG draws, replay schemas or
numerical golden fixtures were changed. Contexts remain attribute-based; this
phase does not attempt to replace all mutable state with a large dataclass or
introduce another general-purpose interface. Remaining work includes removing
legacy driver reexports, moving all file ingestion out of scientific preparation,
normalizing AFO parameter records before realization, and assessing whether
native and external BMI lifecycle orchestration can share more code without
conflating their distinct backend responsibilities. Checkpoint/output integrity
issues are separate from this simplification.

Run `python -m pytest tests` in the complete development environment.
