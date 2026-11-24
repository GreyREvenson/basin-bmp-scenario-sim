# Phase 28: AFO decomposition, independent scientific defaults, and warning fixes

This is a clean-break source refactor. No backward-compatibility imports have been retained for the former monolithic `basin_bmp_sim.science.afo` module or moved configuration functions. Update callers to use the canonical imports below.

## Canonical module owners

| Functionality | Module |
|---|---|
| Load and validate AFO input tables | `basin_bmp_sim.science.afo.inputs` |
| Excretion factors and animal nutrient inventories | `basin_bmp_sim.science.afo.excretion` |
| Feedlot runoff, baseline loading and BMP cost quantities | `basin_bmp_sim.science.afo.feedlot` |
| Receiving-field selection | `basin_bmp_sim.science.afo.allocation` |
| Manure mass allocation and annual surface export | `basin_bmp_sim.science.afo.manure` |
| Feedlot/manure BMP recovery | `basin_bmp_sim.science.afo.recovery` |
| Shared AFO numerical validation and draws | `basin_bmp_sim.science.afo.shared` |
| Scientific PLET/RUSLE parameter defaults | `basin_bmp_sim.science.plet_defaults` |
| Insert defaults as parcel-input table rows | `basin_bmp_sim.config.plet_tables` |
| Model configuration defaults, normalization and path resolution | `basin_bmp_sim.config.settings` |
| Human-readable validation report | `basin_bmp_sim.config.reporting` |
| Geospatial and tabular model input preparation | `basin_bmp_sim.config.loading` |

All affected callers and tests now import directly from the owning module. The AFO package root deliberately does not re-export old functions. The bundled excretion-reference CSV remains under `basin_bmp_sim/data/afo/` and is read from the correct package-relative location.

## Avoiding the warnings observed on Windows

- `utils/create_plet_model_inputs.py` uses `pd.read_excel(workbook, sheet_name=...)` rather than the deprecated `ExcelFile.parse()` method. The existing synthetic-workbook tests still exercise the workbook processing, and an additional narrow test checks the read with the deprecation message promoted to an error.
- `utils/drainage_validation.py` calculates transformed DEM pixel corners directly from the six affine coefficients. This removes deprecated `Affine * (x, y)` use, and avoids assuming that older `affine` installations support `Affine @ (x, y)`. Rotated and sheared transforms retain their geometry.

There are no new warning filters or globally silenced categories. No PLET/RUSLE equations, feedlot calculations, nutrient-mass accounting, RNG draws, or scenario-output schemas were changed.

## Deliberately deferred work

`config/loading.py`, `config/validation.py`, and `science/plet_rusle.py` still contain large independent sections. Splitting them by subdomain should be the next *separate*, regression-gated patch. Phase 28 intentionally does not mix that higher-risk decomposition with the AFO module migration.

## Verification

After applying the patch, reinstall editable code if needed and execute:

```bash
python -m pip install -e .
python -m pytest tests
```

The full dependency-equipped suite should pass on the same scientific fixtures and golden results as Phase 27. To check the specifically reported deprecations, run:

```bash
python -m pytest tests/test_phase28_module_boundaries.py tests/test_parcel_down_contract.py -W error::PendingDeprecationWarning
python -m pytest tests/test_create_plet_model_inputs.py
```

The second command requires `python-calamine`. The first verifies package ownership and DEM transform behavior without that dependency.
