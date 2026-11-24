# Phase 31: Source dependency and redundancy cleanup

This phase follows the clean-break refactoring policy: there are **no backward import aliases**. The scientific equations, RNG call sites and BMI exchange schema are unchanged.

## Single-owner implementations

| Responsibility | Canonical owner | Old duplicates removed |
|---|---|---|
| Numeric distribution parsing and sampling | `common/distributions.py` | `config/distributions.py` relocated |
| Physical units/conversion | `common/units.py` | `config/units.py` relocated |
| Physical domains and parameter validation | `domain/parameter_validation.py` | `config/validation/domains.py` relocated |
| PLET NULL-key default and parcel-override resolution | `science/plet_rows.py` | Two copies in input modules |
| Blank scalar input check | `common/utils.py` | Four copies |
| Safe signed mass ratio | `common/numerics.py` | Two copies |
| Numeric/statistical row validation | `config/validation/statistics.py:validate_stats_rows` | Duplicate `validate_stats_table` function |

The scientific model and scenario sampler no longer import low-level numeric sampling and physical-domain logic from configuration-loading modules. The six BMP response/sampling modules also drop unused imports copied from the pre-refactor monolith.

## Dependency policy

- `science`, `bmp`, `domain` and `scenarios` must not import the execution driver or input loading/table I/O implementation.
- Model-specific scientific methods may use validation and sampling policies, but their physical domains are owned by `domain/` and reusable numeric utilities by `common/`.
- `config/` handles file and table ingestion, resolves defaults, and assembles validated inputs.
- `execution/` owns sequencing, checkpoints and output publication.
- Internal import paths that moved in this phase are intentionally unsupported. Update dependent code to use the canonical modules.

New `tests/test_phase31_dependency_cleanup.py` enforces these boundaries and exercises representative semantic equivalence.

## Deferred cleanup

Some AFO scientific implementation modules still call generic statistics validators under `config.validation.statistics`. Separating all input validation from simulation-time scientific checks would require a larger validation architecture change and should be tested independently. The scientific and BMI component context APIs remain dynamically typed; this phase removes their type-only import dependency on the driver without rewriting the context contract.

## Verification

Run `python -m pip install -e .` followed by `python -m pytest tests` in the complete Windows environment. The numerical golden fixtures and scenario replay tests must continue to match exactly.
