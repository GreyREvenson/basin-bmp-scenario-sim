# Phase 24 — Architecture release-readiness evidence and compatibility policy

## Scope and interpretation

The architecture consists of separately initialized stochastic scenario
and scientific-model components, BMI variable exchange, an approved annual
model contract, configurable external scientific backends, reproducible
ensembles, checkpoints, and model-neutral result/analysis formats.

Phase 24 **does not modify scientific equations, random draws, BMI variables,
or ensemble results**. It closes an evidence-management gap: previously a
public interface audit and a process portability command existed, but their
results were not tied to one another or checked in a single release gate.

The new `basin_bmp_sim.certification.release_gate` checks that a public-contract audit,
external-backend acceptance report, and **real** Windows and Linux process-
spawn portability reports have been completed for the same source
configuration and public contract. Without reports, the gate says
`pending_evidence`, **not** `ready` or certified.

## How to gather evidence

1. Install the project's BMI extra: `python -m pip install -e ".[bmi]"`.
2. Select a small, file-backed **independently installed** annual watershed
   model in a Phase 11/12/13 `backend_selection.json`. The backend must have
   its own scientific implementation and an importable factory.
3. Generate the independent-model public-interface report:

   ```bash
   python -m basin_bmp_sim.certification.acceptance backend_selection.json acceptance.json
   ```

4. On **Windows**, run the two-scenario spawned-process check:

   ```bash
   python -m basin_bmp_sim.certification.portability backend_selection.json windows.json --n-scenarios 2 --n-jobs 2
   ```

5. Run the identical selection on **Linux**, saving as `linux.json`.
6. Put the evidence files together and evaluate:

   ```bash
   python -m basin_bmp_sim.certification.release_gate --acceptance acceptance.json --windows windows.json --linux linux.json --report release_readiness.json --require-ready
   ```

Without all three reports, omit the corresponding arguments to get an honest
`pending_evidence` report. `--require-ready` exits nonzero if any report is
missing. Malformed, inconsistent or failed reports are rejected; an existing
readiness file is not overwritten on validation failure.

Acceptance and portability reports now record four content hashes:
backend-selection JSON, component startup manifest, model configuration YAML,
and the frozen public BMI contract JSON. The release gate checks that all
three reports use identical hashes and the same external factory. Each
platform must independently demonstrate serial/spawn/resume identity and
have actually checked the installed `bmipy` abstract interface. **Identical
cross-platform scientific result bytes are not required**: the evidence
shows reproducibility within each operating system, not bitwise equivalence
between operating systems.

The bundled `reference_backend.annual_linear` imports only the published
annual-model and backend-capability contracts. The release gate audits that
source file for private EPA model imports. This does not inspect arbitrary
third-party model source trees.

## Supported compatibility boundary

- **Published annual BMI v1:** `public_contract_v1.json`; exchange schema v2,
  lifecycle schema v1. Renaming, changing dtype/units/ordering, or modifying
  the two-stage BMP protocol is a breaking change requiring versioning and a
  documented migration.
- **Backend-selection schemas 1, 2, 3** remain supported for, respectively,
  single runs, serial ensembles, and checkpointed ensembles.
- **Checkpoint envelope v2:** incomplete checkpoint directories from older
  schemas are not resumable safely. Completed ensemble/replay JSON outputs
  remain readable. Scientific input hash preservation is required on resume.
- **Model-neutral annual interface:** a compatible external backend must
  satisfy the public variable contract and declare supported CPS codes,
  pollutant order, pathways and required scientific capabilities. Native
  Parquet diagnostics remain part of the original PLET/RUSLE execution path,
  not an external-backend interface promise.
- **Maintenance policy:** retain regression fixtures and black-box tests for
  public API changes. Additive fields still require contract review; breaking
  changes require a new schema, migration notes, and compatibility tests.

## Release decision and outstanding risks

The gate's successful status is `ready_for_architecture_release_review`,
**not** formal CSDMS certification or endorsement of an external backend's
scientific validity. Evidence JSON is unsigned and operator supplied; a
release owner must corroborate successful CI jobs, operating-system runners,
installed package versions, and the backend's scientific suitability.

The source ZIP supplied for Phase 24 did not include a `.github` directory,
so the previously described Phase 22 workflow cannot be audited from that
archive. Check that the actual Git repository contains the expected workflow
and review its execution history; a workflow file by itself is not proof of
Windows process-level execution.

Other accepted limitations are: project-specific BMI lifecycle extensions,
virtual index grids and logical time rather than physical grids/timestamps,
undeclared third-party scientific files or remote inputs outside checkpoint
provenance, dependency version drift, and native-only diagnostic tables. These
are documented boundaries, not evidence that other watershed models are
scientifically interchangeable.

The refactoring can be declared **architecture-complete** after evidence is
collected and any resulting defects are fixed. Future analytical features
(sensitivity measures, scientific calibration, visualization) are optional
model development, not prerequisites for the architecture boundary.
