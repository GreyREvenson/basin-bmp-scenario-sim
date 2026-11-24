# Phase 7: BMI method coverage and ordered-domain coupling safeguards

**Target:** the supplied `dev_arch_refac (6)` repository, after Phase 6. This
phase hardens the public BMI adapters and the generator-to-model variable
contract **without changing** any stochastic draw sequence, PLET/RUSLE or
statistical numerical kernel, wetland routing, AFO process, output schema, or
approved scientific formulation.

## BMI 2.0 methods

`BmiArrayComponent` now implements `get_input_item_count()`,
`get_output_item_count()`, and `get_grid_face_edges()` (the last raises
`NotImplementedError` because no edge/face topology is represented). These
complete the BMI 2.0 *method-name surface* on the driver, live generator,
recorded-scenario generator, and watershed adapters. The existing grid IDs
remain **logical 1-D index-space grids**, not true geographic parcel or
outlet grids. Time remains a logical scenario/placement counter, `none` units.

Data transfer has stronger safety guarantees:

- `get_value` and `get_value_at_indices` require writable, contiguous,
  correctly typed 1-D NumPy destination arrays. Previously a reshape of a
  strided destination could write into a temporary without updating the
  caller's buffer.
- `set_value` and `set_value_at_indices` now reject implicit dtype conversion
  rather than silently truncating floating values or wrapping integer
  values. Values must match the advertised dtype exactly.
- Indexed writes validate bounds and reject duplicate write positions. They
  call the public `set_value()` method with a complete proposed vector;
  scientific and lifecycle checks cannot be bypassed through indexed writes.
  Repeated **read** indices remain legal.
- Live AFO recovery gains and watershed BMP cost values are checked **before**
  the corresponding BMI input buffer is modified, so invalid inputs cannot
  poison the pending transaction.

The official BMI method catalog and conventions are documented at:

- https://bmi.csdms.io/en/stable/bmi.spec.html
- https://bmi.csdms.io/en/stable/bmi.best_practices.html
- https://github.com/csdms/bmi-python

## New versioned index-domain handshake

Both the live stochastic generator and watershed model publish three
additional **output-only** BMI variables. The external driver BMI publishes
the same variables from its prepared model.

| Variable | Type/shape | Meaning |
| --- | --- | --- |
| `coupling__exchange_schema_version` | int64 scalar | Version of this project's exchange variable schema |
| `coupling__index_manifest_utf8` | uint8 vector | Canonical UTF-8 JSON manifest of ordered parcel IDs, outlet IDs, pollutant names, pathway names and BMP CPS codes |
| `coupling__index_fingerprint` | uint8[32] | SHA-256 digest of the manifest bytes |

The existing live coupler validates both components' versions, checks that
each fingerprint matches its own manifest, and then requires identical
fingerprints and manifests *before* applying any BMP. Incompatible ordering
now raises a `ValueError` rather than producing a plausible but incorrectly
indexed simulation.

The manifest contract is in `basin_bmp_sim/interfaces/domain_index_contract.py`.
It does **not** describe polygons, parcel hydrologic connectivity, pollutant
mass-transfer equations, or model capabilities. The current components still
share the same validated in-memory watershed domain. Matching index manifests
are necessary for coupling but do not prove scientific compatibility;
formulation registry and input preflight checks remain authoritative.
The new names are project-local, not CSDMS Standard Names.

## Optional reference interface check

`pyproject.toml` now includes an optional `bmi` extra for `bmipy`.
An opt-in `tests/test_phase7_bmipy_optional.py` compares the adapter method
catalog to the installed official `bmipy.Bmi` abstract methods:

```bash
python -m pip install -e ".[bmi]"
python -m pytest tests/test_phase7_bmipy_optional.py -v
```

This checks **method coverage**, not conformance certification or third-party
framework interoperability. The development environment did not have
`bmipy`, so this optional comparison was **skipped**, not passed.
`tests/test_phase7_bmi_conformance.py` checks the required BMI 2.0 method
surface independently and exercises inputs/outputs, grid metadata, safe
buffer behavior, dynamic array extents, and index-domain mismatches.

## Verification and scientific equivalence

Before modifying the exact `(6)` ZIP, eight seeded outputs were independently
captured using the permanent golden fixture harness. The same cases were
captured after this patch. Their canonical serialized output (tables and
outlet trajectories) matches **byte for byte**, SHA-256:

`1b63916915012a41120cf61a2e33ff76bc4372e5904a8a3d1f5d7a142cb7c208`

Cases: PLET/RUSLE in-field, constructed wetland, grassed waterway, statistical
loading, mass-based AFO, multi-BMP budget, statistical multi-BMP, AFO manure
recovery. New tests cover failure conditions without altering scientific
formulations.

The patch-development runtime lacks `pyarrow` and `python-calamine`. Tests
that invoke actual Parquet or PLET workbook I/O fail for those missing
optional dependencies. The wide scientific regression selection excluding
those dependency-bound tests passed. Run the full suite in your dependency-
equipped local environment:

```bash
python -m pytest tests
```

## Remaining work

1. Test the adapters against a full external CSDMS BMI harness, including
   lifecycle and grid metadata semantics; this phase does not claim full
   external certification.
2. Decide whether the optional `complete_pending()` commands should remain
   documented BMI extensions or be represented by a more general component
   event protocol. They preserve AFO-recovery-before-cost RNG sequencing.
3. Transfer the validated *geospatial/hydrologic domain* independently, not
   only an index manifest, before coupling models that do not share memory.
4. Separate stochastic scenario **implementation** from model-response
   parameter uncertainty for cross-model comparisons, without silently
   changing current seeded results.
5. Add user-environment Parquet and serial/parallel before/after checks.
