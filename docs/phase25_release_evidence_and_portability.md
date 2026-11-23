# Phase 25 — Content-bound release evidence and portable CI

## Purpose and scope

Phase 25 strengthens the release-readiness procedure introduced in Phase 24.
It does **not** change the BMI variable contract, watershed equations, stochastic
sampling, scenario JSON, checkpoints, or the native driver. It does not declare
this repository scientifically or formally CSDMS certified.

The previous evidence identity included the backend selection, startup manifest,
model YAML and published BMI contract. Those hashes alone did **not** detect
changed scientific data files or an altered external backend entrypoint.

## New evidence identity (schema version 2)

Acceptance and portability reports now include:

- `scientific_inputs_sha256`: content digest of scientific inputs referenced by
  supported YAML keys, shapefile sidecars and explicitly declared backend assets.
  The digest uses semantic labels, file suffix, content hashes and sizes, **not
  absolute machine paths**, so copies of the same data can match across Windows
  and Linux.
- `external_backend_source_sha256`: SHA-256 hash of the configured factory
  **entrypoint module file**. This does not cover transitive imports, installed
  dependencies, or arbitrary remote scientific services.
- `additional_scientific_input_count`: count of explicitly declared backend
  assets. At least **one** is required by the release gate for an external-model
  review. The asset list must include *all* external scientific data files,
  including the reference backend's domain JSON.

Both evidence commands verify these identities **before and after** executing
scientific calculations and refuse to publish a success report if input files
change during execution. Version 3 ensemble `additional_input_files` are also
honored during certification and preserved in the temporary checkpoint plans.

The release gate now consumes evidence schema version 2 and rejects older
version 1 reports. This is an **evidence-report migration**, not a change to
scientific result or replay schemas. Regenerate all acceptance, Windows and
Linux reports after applying this patch. Old locally saved evidence remains
readable as JSON but cannot qualify a release candidate.

## Run the evidence checks

Install the optional interface package and use a **single external-backend
selection JSON** that points to a small, file-backed, representative model.
For a backend whose scientific data includes `backend_domain.json`:

```bash
python -m pip install -e ".[bmi]"
python -m src.interfaces.bmi.acceptance selection.json acceptance.json --input-file backend_domain.json
python -m src.interfaces.bmi.portability selection.json windows.json --n-scenarios 2 --n-jobs 2 --input-file backend_domain.json
```

Run the same portability command on Linux, producing `linux.json`. The
`--input-file` option can be repeated, and paths are resolved **relative to
`selection.json`**, not to the shell's current directory. The exact ordered
asset list and file contents must agree across platforms. Use the identical
selection, startup and model YAML bytes on both platforms.

Finally:

```bash
python -m src.interfaces.bmi.release_gate --acceptance acceptance.json --windows windows.json --linux linux.json --report readiness.json --require-ready
```

The gate still reports `pending_evidence` when inputs are absent. Its highest
positive result is `ready_for_architecture_release_review`, not a formal
external BMI certification or scientific model endorsement. Reports are
**unsigned local files**: a reviewer must independently corroborate their
provenance against CI job logs and immutable source revisions.

## GitHub Actions coverage

This patch includes `.github/workflows/bmi-portability.yml`, which was absent
from the supplied ZIP even though earlier Phase 22 notes described it. It
runs selected API, BMI and evidence tests on Windows and Ubuntu with `bmipy`
installed. A manual `workflow_dispatch` run can select a checked-in,
file-backed *external* model and one backend asset to produce actual acceptance
and spawned-process reports as downloadable CI artifacts.

A passing unit-test job or uploaded report **does not by itself complete**
release qualification. It requires a real scientific two-worker run on both
systems for the same model and a reviewer-verified independent model. The
workflow cannot access arbitrary files on a developer's laptop; its selection
and input files must be available on the runner.

## Remaining evidence limits

- The YAML inventory recognizes explicitly modeled configuration path keys,
  not arbitrary plugin inputs; extra backend assets must be supplied.
- The backend source hash identifies the factory entrypoint only, not its
  imported scientific modules or package distribution build.
- SHA-256 hashes protect content identity, **not evidence authenticity**.
- Equal serial/spawn/resume results on each OS show repeatability, not
  numerical agreement between different OSes or scientific correctness.
- Native-only diagnostic Parquet output remains intentionally outside the
  model-neutral external backend contract.
