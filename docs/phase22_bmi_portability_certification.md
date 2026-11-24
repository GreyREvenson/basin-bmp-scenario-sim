# Phase 22 — BMI portability and integration-certification gate

## Intent and scope

This is an **architecture-hardening** phase, not a change to the PLET/RUSLE,
statistical-load, BMP, or AFO equations. It does not change the existing
stochastic generator, random-number sequence, backend-selection schema,
checkpoint format, or native Parquet output path. The previous public BMI v1
contract and all exchange variable names remain unchanged.

The new tool, `basin_bmp_sim.certification.portability`, checks three conditions:

1. The frozen Phase 21 project contract is still present and unchanged.
2. All five bundled BMI adapters satisfy the abstract-method requirements of
   the **installed** `bmipy.Bmi` Python interface. This requires the optional
   `bmipy` dependency unless explicitly disabled.
3. A **real** process-spawned (`n_jobs >= 2`) Phase 13 ensemble yields
   byte-for-byte the same JSON output as a serial checkpointed ensemble,
   including scenario replay definitions; a resumed run also matches.

It uses the actual configured generator and watershed model. No production
worker is substituted by a thread or test fixture. Each run uses a fresh
component per realization and the unchanged Phase 12 seed schedule. Checkpoints
and comparison results live in temporary directories and are removed after the
check. An existing user output or checkpoint directory is **never written**.

The returned JSON report records the run environment, seed-schedule-related
scenario count, model/config source-file hashes, result SHA-256, and which
checks were completed. A report is published atomically **only on success**.

## Commands

Install the official Python BMI abstract interface alongside the project:

```bash
python -m pip install -e ".[bmi]"
```

Use a **small, valid, file-backed** Phase 11–13 backend selection, with an
independently accessible startup YAML and required source data. The selection
may be native or external, and may be single-scenario or ensemble: the
certification gate creates its own two-realization test schedule.

```bash
python -m basin_bmp_sim.certification.portability \
    path/to/backend_selection.json \
    outputs/bmi_portability_report.json \
    --n-scenarios 2 --n-jobs 2
```

On Windows, use the same command on one line or PowerShell line continuations.
To check process portability when `bmipy` cannot be installed, add
`--without-bmipy`: the report will mark `official_bmipy` as `not_checked`,
**not** claim official conformance.

A configured external factory must be an importable, module-level factory
visible to freshly spawned processes; closures, live model instances, monkeypatch
fixtures, and other parent-process state are not transferred. This is also a
useful check on third-party backend packaging.

To run the integration test separately from the unit suite:

```bash
# bash / Linux / macOS
export BASIN_BMP_BMI_SPAWN_SELECTION=path/to/backend_selection.json
python -m pytest -q tests_cfg/test_bmi_spawn_integration.py
```

For PowerShell:

```powershell
$env:BASIN_BMP_BMI_SPAWN_SELECTION = "path/to/backend_selection.json"
python -m pytest -q tests_cfg/test_bmi_spawn_integration.py
```

The ordinary `python -m pytest tests` suite contains unit tests for report
atomicity, failure handling, the official-ABC checker, and a *real* spawned
process pool executing importable worker code. The full scientific spawn test
is deliberately placed outside `tests/` because watershed file I/O and
scientific workloads can be expensive.

## CI / cross-platform coverage

`.github/workflows/bmi-portability.yml` runs the public BMI tests with
`bmipy` installed on **Ubuntu and Windows** (Python 3.11), so the formerly
skipped optional BMI comparison is exercised by CI. The workflow can also be
triggered manually with a checked-in `spawn_selection` path to run the
file-backed two-worker scientific acceptance test on both operating systems.

The workflow file itself is **not evidence of a successful Windows execution**.
A clean Windows CI run and the generated report are required before claiming
that the selected external backend and scientific dataset have passed Windows
process-level interoperability. The heavy example watershed may take longer
than the lightweight unit tests; supply a representative small configuration.

## Scientific and interoperability limitations

- Matching serial, parallel and resumed bytes demonstrates repeatability for
  the chosen configuration, Python/NumPy environment and backend version. It
  does not prove agreement between different scientific models or different
  dependency versions.
- `bmipy.Bmi` abstract-method coverage and the project's runtime audit are
  **not formal CSDMS certification**. The published BMI uses logical steps and
  virtual 1-D index grids, not physical-year or geospatial grid semantics.
- External-backend scientific validity, mass-balance appropriateness, and
  support for each BMP/AFO feature still require separate domain review.
- The portability report hashes its startup manifest, model YAML and selection
  file. **Separately**, the Phase 23 checkpoint runner hashes native YAML-
  referenced scientific input files and any explicitly declared additional
  backend assets before allowing resume. Unlisted plugin data, installed
  backend source revisions, and dependency versions are not automatically
  covered. Freeze or declare them for defensible repeatability.
- Native diagnostic Parquet tables remain available only through the original
  native driver. The model-neutral CLI continues to publish neutral JSON and
  does **not** assume native diagnostic schema. A regression test protects
  this output boundary.

## Release-gate interpretation

Phase 22 establishes an executable portability acceptance **procedure** and
adds the Windows/Linux CI workflow. Before declaring architecture completion,
collect actual successful CI results (including an independently installed
external backend and a small scientific spawn selection), review remaining
native-only diagnostic dependencies, and document a supported API/versioning
and release process. The new test utility does not automatically certify all
third-party models or hardware configurations.
