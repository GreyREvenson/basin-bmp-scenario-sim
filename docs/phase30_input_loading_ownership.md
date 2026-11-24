# Phase 30 — input-loading ownership

Phase 30 makes the central configuration loader an **orchestrator**, rather than a
repository-wide collection of file readers and table normalizers. It is a
clean-break internal import migration: there are **no compatibility wrappers**.

| Canonical module | Responsibility |
|---|---|
| `basin_bmp_sim.config.loading` | Validate the run configuration and assemble the final model input bundle (`load_and_validate_all`). Also owns basic CPS/pollutant-list normalization. |
| `basin_bmp_sim.config.spatial_loading` | Load domains, parcels, directed parcel connectivity, outlet locations/targets, selection weights, and delivery ratios. |
| `basin_bmp_sim.config.bmp_loading` | Normalize selection-probability, effectiveness/efficiency, and additive BMP-cost tables, and validate coverage. |
| `basin_bmp_sim.config.statistical_loading` | Prepare statistical pollutant load-rate rows, expand parcel defaults, and resolve aggregate pathway fractions. |
| `basin_bmp_sim.results.trajectory_loading` | Read and validate saved outlet trajectories used in plots. |
| `basin_bmp_sim.config.table_sources` | Read normalized, schema-aware source tables (unchanged in this phase). |
| `basin_bmp_sim.config.plet_loading` | Prepare PLET/RUSLE and HUC-specific input tables (unchanged in this phase). |

`config.loading` now calls the spatial, BMP, and statistical modules through
module-qualified names, so importing the orchestrator does **not** provide an
obsolete re-export API for the moved helpers. Internal tests patch the owning
module directly when isolating I/O or validation logic. Other runtime callers
of BMP selection and result plotting import their canonical modules.

## Scientific invariants

The 22 relocated helper-function ASTs are identical to the original functions
in the uploaded Phase 29 baseline. The input assembly sequence, pathway
validation, output dictionary keys, stochastic sampling, and scientific methods
are not intentionally modified. Golden scenario/output and replay tests still
use the same fixtures. The changed loader only delegates to the same helpers in
their new locations.

## Migration

Reinstall the editable distribution if needed: `python -m pip install -e .`.
Replace imports from `basin_bmp_sim.config.loading` with the canonical module
above; the sole public entrypoint remaining there is `load_and_validate_all`.
No old import path is retained solely for compatibility.

## Verification

The Phase 30 run in the available Linux environment passed 977 selected tests,
with one optional `bmipy` skip. Eight baseline/golden case tests and the new
module-boundary checks are included. The package wheel builds, and contains
all four new modules. Tests requiring `pyarrow`/Parquet or `python-calamine`
were excluded from that broad run because these packages are absent here.
The entire suite must be run in the full Windows development environment.

Some warnings from newer Pandas/pyproj versions were present before this phase;
this module-movement patch does not suppress or change them.
