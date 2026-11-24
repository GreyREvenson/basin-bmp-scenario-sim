# Phase 27: canonical package and source ownership

## Scope

The Python distribution is still `basin-bmp-sim`, but the only importable
scientific framework package is **`basin_bmp_sim`**, implemented in the standard
`src/basin_bmp_sim/` source layout. The independently implemented reference
backend remains a separate top-level Python package, now also installed from
`src/reference_backend/`.

This is an intentional **breaking** source migration: no legacy `src.*` import
aliases, wrapper modules, or former `run_model.py` entry point are shipped.
Install this branch again after applying the patch.

## Module ownership

| Package | Owns |
|---|---|
| `config` | Model input loading, units, scopes, distribution schemas, validation |
| `domain` | Stable parcel, network and scientific domain definitions |
| `scenarios` | BMP selection, sampling, physical designs and costing |
| `bmp` | Practice definitions and practice-family response implementations |
| `science` | PLET/RUSLE, AFO, annual model assemblies, routing and mass accounting |
| `interfaces` | BMI components, public exchange contracts and lifecycle |
| `execution` | Native Monte Carlo, model-neutral coupling, ensembles, checkpoints, CLI |
| `results` | Native result aggregation, plotting and serialization |
| `analysis` | Model-neutral decision support and sensitivity post-processing |
| `certification` | External-backend acceptance, BMI portability, release evidence and gates |
| `common` | Reusable non-scientific logging, I/O and normalization helpers |

The old source root (`src/`) now contains only installed package directories.
Explicit module names replace the former `src.model`, `src.summaries` and
`src.plotting` aliases, and the shadowed `src/bmp.py` is removed. Imports of
BMP pathway helpers target their defining modules rather than legacy package
reexports.

## Key commands

```bash
python -m pip install -e .
basin-bmp-sim --help
python -m basin_bmp_sim.execution.cli config.yaml
python -m pytest tests
python -m basin_bmp_sim.interfaces.bmi.public_contract
```

The test suite configures the source directory for child Python processes when
executing directly from a checkout. Installation is still required for normal
CLI usage and for importing the package outside the source checkout.

## What stays fixed

* No PLET/RUSLE/BMP/AFO scientific formulas or constants were changed.
* Scenario RNG, replay definitions, stored ensemble schemas, and public BMI
  exchange variable names and numeric contracts remain unchanged.
* Golden regression fixtures are not updated to hide changes.

## Future work, deliberately excluded from this package move

The large input-loading, input-validation, AFO and PLET/RUSLE modules still
contain several responsibilities. A subsequent scientific/input decomposition
should split them **without** any temporary import aliases, while keeping
numeric outputs and RNG ordering under strict regression tests. Review
any leftover reexports and redundant scientific helper interfaces at that point.
Similarly, dependency direction (e.g. scientific process imports from input
parsing) needs to be made explicit and mechanically enforced after those
modules have been decomposed.
