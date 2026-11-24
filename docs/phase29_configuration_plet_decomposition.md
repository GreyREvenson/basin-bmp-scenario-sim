# Phase 29 — configuration and PLET/RUSLE module boundaries

This is a **clean-break internal API change**. No legacy module aliases or re-export-only adapters are provided. All callers, tests, and utilities must use the new module locations. The equations, annual load units, parameter distributions, input schemas, random-number sampling sequence, and public BMI exchange contract are unchanged.

## Module ownership

| Canonical module | Owns |
| --- | --- |
| `config.table_sources` | File/table access, identifier normalization, CSV/GeoPackage table assembly and generic row selection |
| `config.plet_loading` | PLET hydrology lookups, pollutant concentration tables, HUC12 forcing and inheritance |
| `config.loading` | Top-level loading orchestration, remaining spatial/BMP/statistical loading, validated input-bundle construction |
| `domain.parameter_validation` | Physical domains, numeric value validation and sampling domains |
| `config.validation.statistics` | Distribution statistics, bounds, row uniqueness and column requirements |
| `config.validation.selection` | BMP-selection and trajectory-table validation |
| `config.validation.plet` | PLET input and runtime coverage validation |
| `config.validation.configuration` | Model configuration validation |
| `config.validation.statistical` | Statistical load and BMP-efficiency coverage checks |
| `science.plet_state` | Per-parcel load state data structure |
| `science.plet_classification` | Pure PLET parameter, land-cover and HSG classification normalization |
| `science.plet_sampling` | Parcel parameter, concentration and hydrology sampling |
| `science.plet_initialization` | Baseline pathway-load initialization with registered scientific process methods |

The file-based helper `plet_hydrology_from_classifications` is now under `config.plet_loading`, not the scientific classification module. Existing process equations continue to live in `science.hydrology`, `science.sediment`, and `science.nutrients`; code that previously imported those methods indirectly through `science.plet_rusle` now imports their canonical modules.

## Removed locations

- `basin_bmp_sim.config.validation` **as a flat module** is replaced by the `config.validation` package. Import concrete functions from its submodules; the package root does not re-export functions for compatibility.
- `basin_bmp_sim.science.plet_rusle` is removed. Load state, sampling, initialization and classification each have a single canonical home.
- Generic and PLET-specific table loaders are no longer defined in `config.loading`.

`config.loading` is still substantial because it orchestrates validated input assembly and includes spatial, BMP, and statistical loader code. A subsequent scoped cleanup can extract those groups without simultaneously changing numerical science.

## Verification

Run the numerical regression fixtures, PLET lookup tests, configuration/HUC12 tests and whole test suite after applying the patch:

```bash
python -m pip install -e .
python -m pytest tests
```

The tests include an import-ownership check preventing the removed monolithic paths from being silently reintroduced. The migration does not alter output/replay schemas or the bundled BMI contract.
