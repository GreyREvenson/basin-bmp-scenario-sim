# Phase 13 — Parallel, checkpointed and resumable model-neutral BMI ensembles

## Scope and backward compatibility

Phase 13 adds an **opt-in** schema version 3 to the Phase 11/12 selected-backend
path. The legacy native YAML CLI, PLET/RUSLE and statistical science, AFO and
cost sampling, and native Parquet results are unchanged. Backend selections
using schema 1 (one scenario) or schema 2 (serial in-memory ensemble) are
unchanged and continue to use their original code paths.

Version 3 uses the **same** Phase 12 `numpy-seedsequence-spawn-uint64-v1`
worker-seed schedule, the same independently initialized BMI components per
scenario, the same scenario replay definitions, the same Phase 12
**result-schema-v2 JSON**, and the same across-realization statistics.
Neither `n_jobs` nor checkpoint resumption changes realization identity or
sampling order *inside* a realization. Statistical results are sorted in
scenario-index order, not job completion order. This does not make a
third-party backend scientifically equivalent to native PLET/RUSLE.

## Selection configuration

Create a Phase 9 startup manifest specifying the existing source model YAML,
base seed and first scenario ID. In a separate backend selection JSON:

```json
{
  "schema_version": 3,
  "backend": "native",
  "component_manifest": "startup.json",
  "ensemble": {
    "n_scenarios": 100,
    "n_jobs": 4,
    "checkpoint_dir": "outputs/my_ensemble_checkpoints",
    "resume": false
  }
}
```

To use a trusted external annual backend, set `backend: "external"` and add
`external_factory`, `backend_options` and any `required_features` exactly as
in Phase 11. Component initialization and capability negotiation occur **for
every realization**, inside the worker. The external factory must be importable
in each worker process; in-memory monkeypatched modules will not transfer
across the `spawn` boundary. The `checkpoint_dir` is resolved relative to the
selection JSON file; it must be **unique to this ensemble** and initially empty.
`n_jobs` defaults to 1 and accepts integers 1 through 64; each parallel job
is a separate process. The parallel pool uses `spawn` for portability across
Windows, macOS and Linux. Memory and I/O requirements scale with the selected
job count. When running on a small machine, start with `n_jobs: 2`.

Invoke the existing CLI:

```bash
basin-bmp-sim --backend-selection selection_v3.json --coupled-result outputs/ensemble.json
```

If a worker fails or execution is interrupted, a previously published result
is **not overwritten**. Successfully committed scenario checkpoints remain
in the checkpoint directory. For a subsequent run, edit only:

```json
"resume": true
```

and run the same command. You may also adjust `n_jobs` between attempts.
Resuming validates the run fingerprint, completed-scenario seed and replay
identities, SHA-256 record checksums and the complete Phase 12 cross-scenario
contract before publishing the final output. Only missing scenarios run again.
`resume: false` refuses a nonempty directory to prevent accidental overwrite;
`resume: true` requires an existing `run.json`.

The checkpoint directory holds a versioned `run.json` and one atomic JSON file
per finished scenario (`realization_00000000.json`, etc.). Only the parent
process writes these files. Each worker returns a Phase 11 record; the number
of in-flight tasks is limited to twice `n_jobs`. A directory-level
`.runner.lock` prevents two ensemble runners from writing to the same
checkpoint directory concurrently. If a process is force-killed and leaves a
stale lock, verify no previous run is active before manually deleting it.

## Replay, validation and memory characteristics

The final result remains a standard Phase 12 ensemble JSON, compatible with:

```bash
python -m basin_bmp_sim.execution.ensemble_runner outputs/ensemble.json outputs/replays
```

Final assembly reads full BMP histories **one scenario at a time** from disk,
not as one list in RAM. Statistics still retain compact numeric arrays
across all realizations to compute exact percentiles; it is *not* fully
constant-memory numerical aggregation. The final JSON is written to a
same-directory temporary file and published by atomic replacement only after
all checkpoints and summary statistics have passed validation. Errors can
leave temporary checkpoint-write files; restart ignores these incomplete
files and checks only committed records.

**Updated by Phase 23:** Resume identity now also includes SHA-256 hashes of the
native scientific files named in the model YAML, shapefile component files,
and (when applicable) the bundled default AFO excretion-factor table. The
optional `ensemble.additional_input_files` list covers external plugin files
and other assets not described by the native YAML schema. The runner checks
input content again before publishing. See
`docs/phase23_scientific_input_provenance.md` for coverage, migration, and
scientific limitations.

Phase 13 checkpoint directories predate this stronger identity and **cannot
be safely resumed** after Phase 23. Start a new checkpoint directory; normal
Phase 12/13 ensemble results and seed schedules are unchanged.

## Verification

- Phase 13 tests check strict version-3 validation; version-1/2 compatibility;
  identical v2 versus v3 serial JSON bytes for PLET/statistical, mixed-budget,
  and AFO-recovery fixtures; resumed scenarios are not rerun; interrupted
  external-model runs resume only missing scenarios; cost/outlet/replay
  identities and SHA-256 checks; changed YAML/corrupted checkpoint rejection;
  preservation of existing results on error; thread-executor validation of
  concurrent completion ordering; and real process-`spawn` worker import.
- Existing Phase 1–12 architecture, golden-science and BMI tests remain the
  required numerical-regression gate.
- The real spawned-process model example could not be completed in the
  constrained verification environment; the user should run the full suite
  and a small real `n_jobs: 2` ensemble with their installed dependencies.

For a truly large ensemble or when external backend initialization is costly,
future improvements may include more efficient numeric percentile aggregation,
remote checkpoint storage and explicit input-data provenance hashes.
