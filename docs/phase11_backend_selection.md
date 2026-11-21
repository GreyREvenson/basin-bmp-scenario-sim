# Phase 11 — Explicit model selection and capability negotiation

## Scope

Phase 11 adds an **opt-in, user-selected** annual BMI execution path. It
selects one of two model implementations for a *single* stochastic BMP
scenario:

- `native`: the separately initialized native `WatershedModelBmi` with the
  existing PLET/RUSLE or statistical approved formulation.
- `external`: a trusted installed Python factory constructs an independent
  annual watershed backend, a validated domain, and a declaration of
  scientific capabilities. The generic `ExternalWatershedModelBmi` wraps it.

The *existing* `python run_model.py config.yaml` CLI continues to use the
native Monte Carlo driver and writes its original Parquet results, diagnostic
files and plots. Phase 11 adds `--backend-selection` and `--coupled-result` to
that same CLI; this optional mode writes a **separate model-neutral JSON
result**, not the native output tables. Its scientifically neutral output
cannot be substituted for the original PLET/RUSLE table schema.

Phase 11 makes no changes to scientific kernels, input preparation, approved
model formulations, random-number ordering, BMI variable schemas, or replay
serialization. **No existing YAML setting silently switches the backend.**

## Configuring an opt-in execution

First prepare the Phase 9 per-scenario component startup manifest:

```json
{
  "schema_version": 1,
  "model_config": "model.yaml",
  "worker_seed": 12345,
  "scenario_id": 1
}
```

Then make `annual_backend.json` in the same folder:

```json
{
  "schema_version": 1,
  "backend": "external",
  "component_manifest": "scenario_1_bmi.json",
  "external_factory": "my_model.provider:create_annual_backend",
  "backend_options": {"parameter_file": "my_science_inputs.json"},
  "required_features": ["afo_recovery", "upstream_routing"]
}
```

`component_manifest` resolves relative to `annual_backend.json`; startup
manifest paths resolve relative to the startup manifest. The selection file
is strictly versioned and disallows unknown keys, invalid feature names, and
native selections carrying external-model settings. It must reference an
existing validated Phase 9 startup manifest (including its existing model
YAML, seed and scenario ID). This allows the live **existing generator** to
initialize without receiving native driver contexts.

For native BMI execution, instead specify:

```json
{
  "schema_version": 1,
  "backend": "native",
  "component_manifest": "scenario_1_bmi.json"
}
```

Run either configuration using:

```bash
python run_model.py --backend-selection annual_backend.json --coupled-result outputs/one_scenario.json
```

The old CLI remains:

```bash
python run_model.py model.yaml
```

`--backend-selection` is deliberately incompatible with passing a native
positional YAML config or `--seed`, `--outputs`, and
`--scenario-replay-dir`: the manifests supply the selected component's
identity, RNG seed and optional Phase 9 replay artifact.

## External factory API

The external import `module.path:callable` is **trusted executable Python**.
Select only an installed, reviewed module; importing it executes Python code.
There is no plugin sandbox or model-certification service.

```python
import numpy as np
from src.interfaces.backend_capabilities import (
    AnnualBackendCapabilities, ExternalBackendBundle,
)

def create_annual_backend(*, startup, options):
    # Implement these with your own science and validated model data.
    domain = load_annual_domain(options["parameter_file"])
    backend = YourIndependentAnnualWatershedBackend(domain, options)
    capabilities = AnnualBackendCapabilities(
        supported_cps=tuple(domain.bmp_cps),
        pollutants=tuple(domain.pollutants),
        pathways=tuple(domain.pathway_names),
        features=frozenset({
            "signed_efficiency", "multi_parcel_treatment",
            "afo_recovery", "upstream_routing",
        }),
    )
    return ExternalBackendBundle(domain, backend, capabilities)
```

`load_annual_domain` and `YourIndependentAnnualWatershedBackend` are **your
model's implementations**, not bundled functions. The backend implements
the Phase 10 five-method `AnnualWatershedBackend` protocol. The runtime
supplies `np.random.default_rng(startup.worker_seed)` to the external BMI
adapter; the backend receives this generator during `initialize()` and
`apply_bmp()`. It should **not** create its own uncoordinated RNG.

The factory is called **before scientific initialization**, with keyword
arguments `startup: ComponentStartup` and `options: dict`. It returns
`ExternalBackendBundle(domain, backend, capabilities)`. The capability
preflight verifies:

1. Supported annual-contract and BMI exchange schema versions.
2. Every configured CPS code is included in the backend's supported CPS codes.
3. The ordered pollutant and pathway identities match the declared domain.
4. `signed_efficiency` and `multi_parcel_treatment` are always declared,
   since the current live generator may exercise either.
5. All additional `required_features` selected by the user are declared:
   `afo_recovery` and/or `upstream_routing`.
6. The scientific backend exposes callable `initialize`, `snapshot`,
   `apply_bmp`, `complete_bmp`, and `finalize` methods.

After initialization, the Phase 10 BMI coupler additionally checks the
actual array units, types and dimensions, ordered parcel/outlet/pollutant/
pathway/CPS index manifest, and model/generator domain exchange **before
sampling BMP placements**. These checks cannot establish scientific
accuracy. Features are **claims by the external model provider**, not a
certification of mass conservation, sediment/nutrient equations, wetland
routing, or AFO behavior. Declare any active model requirements under
`required_features`; do not omit `afo_recovery` or `upstream_routing` merely
to bypass an incompatibility.

## Opt-in JSON output

The JSON output includes the selected backend type, scenario ID and worker
seed, the actual ordered index manifest, flat row-major baseline and current
parcel load-rate arrays (`kg ha-1 yr-1`), flat row-major outlet change (`kg`),
per-BMP records, outlet mass change after each BMP, and the replayable
completed scenario. The index manifest supplies dimensions and identities.
Negative signed efficiencies and negative outlet improvements are preserved.
The file is written atomically (temporary file + rename) after the entire
scenario succeeds; a failing validation does not write a partial result.

The new path runs **one scenario per selection/startup manifest**. It does not
yet schedule cross-scenario ensembles of arbitrary models, recreate native
Parquet tables for external models, produce native summary figures, or provide
scientific calibration/validation for arbitrary providers. Multi-scenario
scheduling and a more general generator configuration contract are future
work. In particular, the Phase 9 generator still loads and validates the
native project YAML and uses its current approved sampling and CPS design
rules. Alternative models must match that domain; they do not automatically
become selectable by editing the existing science-formulation field.

## Tests

`tests/test_phase11_backend_selection.py` verifies strict selection parsing,
relative path resolution, missing provider errors, version checks, missing
CPS or feature preflight, native and external selection, JSON output and
repeatability with a non-PLET backend, plus end-to-end CLI routing. Existing
Phase 1/2 golden outputs and Phase 3–10 BMI and scientific regression tests
continue to cover the unchanged default CLI and scientific numerics.
