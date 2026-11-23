# Phase 23 — Scientific-input provenance for resumable BMI ensembles

## Purpose

An ensemble resumed from checkpoints must not silently mix scientific inputs
from different revisions. Previously, the checkpoint identity hashed the YAML
configuration but **not the GeoPackages, CSVs and other scientific input files
referenced by that YAML**. Changing the contents of an existing input file
without changing its path or the YAML could leave an old checkpoint eligible
for reuse. Phase 23 closes this specific reproducibility gap.

Only schema-3 checkpointed BMI ensembles are affected. Native Monte Carlo
executions, schema-1 single scenarios and schema-2 noncheckpointed ensembles
retain their behavior. No modeling equation, random draw, worker-seed policy,
replay definition, or published ensemble JSON schema was changed.

## Inventory and verification

Before checkpoint creation or resume, the runner builds an input manifest with
an absolute path, referencing YAML key(s), byte size and streamed SHA-256 hash
for every discovered file. It hashes:

- The model YAML itself (already tracked by the checkpoint runner).
- Scalar or list-valued scientific file paths recognized by the native YAML
  loader: `domain`, `parcels`, `plet_forcing`, `huc_parameters`, `outlets`,
  `animal_operations`, `bmp_efficiency`, `bmp_cost`, and `bmp_sel`.
- `manure_loading.excretion_factors` when explicitly configured, or the bundled
  AFO excretion-factor CSV if mass-based AFO loading uses the default.
- All files in a configured directory, and existing `.dbf`, `.shx`, `.prj`,
  `.cpg`, and `.qpj` companions for a `.shp` file.
- Each file or directory listed in optional `additional_input_files` (below).

Outputs are **not** included. Duplicate references to the same file share a
single digest with all their origin labels. Values in YAML that are not file
paths are already covered by the YAML digest. Referenced but missing files
cause an error before a checkpoint directory is initialized. Hashes compare
content, not timestamps or names alone. All existing records are checked
before new scenarios run, as before.

On resume, the newly computed manifest must equal the recorded run identity.
The runner also recomputes the scientific-input fingerprint before publishing
results, rejecting input mutation during execution and leaving an already
published ensemble untouched. It does not track transient mutations that are
reverted between checks; operators must avoid editing inputs during a run.

## Backend-specific input assets

External watershed backends can open files absent from the native YAML schema.
Declare each such data file (or source directory) in the version-3 backend
selection's **optional** `additional_input_files` list:

```json
{
  "schema_version": 3,
  "backend": "external",
  "component_manifest": "startup.json",
  "external_factory": "my_backend.plugin:make_backend",
  "backend_options": {"coefficients_file": "coefficients.csv"},
  "ensemble": {
    "n_scenarios": 10,
    "n_jobs": 2,
    "checkpoint_dir": "outputs/checkpoints",
    "resume": false,
    "additional_input_files": ["coefficients.csv", "my_backend/plugin.py"]
  }
}
```

Paths in `additional_input_files` resolve **relative to the backend-selection
JSON**, not the model YAML. The list must contain distinct nonempty path
strings. This declaration is an operator responsibility; the framework cannot
infer arbitrary file I/O performed by a third-party backend. Declare backend
code and custom reference datasets where practical. The declared files must
exist and be readable. External package versions and transitive libraries are
not discovered or hashed automatically.

## Checkpoint migration

The checkpoint envelope version changes from **1 to 2**. Prior Phase 13–22
checkpoint directories do not prove which source data produced their saved
realizations; they cannot be safely resumed under the new guarantee. Resume
returns a specific error instructing the user to **start a new checkpoint
directory**. Do not rewrite `run.json` or edit stored digests to bypass this
check. Completed **published ensemble results** and saved replay definitions
remain readable and unchanged.

Changing YAML contents, referenced data-file contents, extra declared files,
backend settings, or the seed schedule requires a fresh checkpoint directory.
Adjusting `resume` or `n_jobs` alone does not change the fingerprint.

## Scope and limitations

- This is a provenance and input-immutability guard, **not** scientific
  validation or full environment reproducibility. Source-code revisions,
  environment/dependency versions, backend side effects, remote data and
  undeclared plugin-specific dependencies remain the operator's responsibility.
- Paths are recorded as absolute filesystem locations. Resume is intended
  within a stable installation, not across moved datasets or operating systems.
- The inventory is based on explicit native YAML file keys; it does not guess
  file paths from arbitrary free-form strings or unsupported plugin options.
- Large datasets are streamed in bounded chunks but fully read for hashing
  at run start and before publishing. Expect additional I/O proportional to
  the combined source-data size. Source files must remain immutable during
  worker execution.

## Validation

```bash
python -m pytest tests/test_phase23_scientific_provenance.py tests/test_phase13_checkpointed_ensembles.py
python -m pytest tests
```

The new tests use only temporary, synthetic configuration files; they test
changed content with restored timestamps, shapefile components, AFO reference
coverage, declared external files, resume rejection, schema migration and
atomic output preservation. Existing seeded golden-result tests remain the
scientific behavior gate.
