# Phase 6: BMI-mediated live scenario and annual watershed execution

This phase changes the **runtime driver path**, not merely the availability of
BMI-shaped facades. Its purpose is to transfer each realized BMP's design and
scientific inputs from the live generator to the watershed-model component
through BMI `get_value` and `set_value`. Model-computed AFO recovery gains are
returned through BMI *before* the generator samples cost; realized cost is
sent back to the model through BMI before the model completes mass accounting.
The driver also receives delivered pollutant reductions and parcel baseline/
current loads from the BMI output buffers. The original driver owns result
persistence, plotting trajectories, and ensemble outputs.

## Versioned exchange and stable ordering

`src/interfaces/placement_exchange.py` is the transport schema. The generator
encodes one placement into typed arrays and the watershed adapter reconstructs
a native `RealizedBMPPlacement` **internally**, for the existing scientific
kernels. The coupler never passes such Python objects across the component
boundary. Source and target declare **the same variable name, unit, rank and
numeric dtype**, verified before every transfer. Transfers use BMI `get_value`
(copy), not `get_value_ptr` (mutable view). The exchange is part of the
project-local version-1 contract and does not claim CSDMS Standard Names.

| Generator output / model input | Units | Meaning |
| --- | --- | --- |
| `placement__cps_code` | 1 | Supported CPS identifier |
| `placement__host_parcel_index` | 1 | Full hydrologic-universe index |
| `placement__affected_parcel_index` | 1 | Ordered parcel indices (variable size) |
| `placement__affected_parcel_fraction` | 1 | Matching fractions; host first |
| `placement__efficiency` | 1 | Flattened `(pollutant, pathway)` efficiencies |
| `placement__failed` | 1 | Zero or one |
| `placement__recovery_effect_scale` | 1 | Zero to one |
| `placement__cost_quantity_ha` | ha | Sampled practice cost quantity |
| `placement__wetland_area_ha` | ha | Wetland-specific design |
| `placement__catchment_area_ratio` | 1 | Wetland-specific design |
| `placement__buffer_length_m` | m | Grassed-waterway length |
| `placement__buffer_area_ha` | ha | Grassed-waterway area |
| `placement__portion_treated` | 1 | Grassed-waterway treatment fraction |
| `placement__recorded_recovery_gains` | 1 | Previous AFO response for exact replay; empty otherwise |

For other BMP families, the inapplicable design scalars are zero. The wetland
`impacted_pids` CSV diagnostic is rebuilt from the ordered affected parcel
indices and the prepared domain's stable parcel-ID mapping, not inferred from
parcel geography. Each scientific model and generator instance currently gets
the **same validated context** at construction; transport does not independently
transfer the domain's geospatial topology, metadata, or model configuration.

### Two-stage response and cost exchange

1. The model adapter `initialize()` draws the baseline exactly once, **before**
   the generator samples placements. Both components then expose BMI arrays.
2. Generator `update()` exposes one pending BMP. The coupler transfers all
   placement variables via BMI into the model and calls model `update()`.
3. The model performs BMP response and any AFO recovery draws; it publishes
   `placement__recovery_gains` (float64). The coupler transfers that BMI output
   to the generator's BMI input of the same name.
4. Generator `complete_pending()` samples the cost *after* response and
   publishes `placement__cost` (float64). The coupler transfers cost to the
   watershed BMI input and calls its `complete_pending()`.
5. The model finishes load/mass accounting and publishes parcel and outlet
   numeric results plus `model__last_bmp_record_utf8` (uint8). The latter is
   a version-1 **output-record sidecar**, containing UTF-8 JSON of the existing
   variable-field BMP output record. It is for faithful output collection, not
   a scientific spatial variable or a replacement for named numeric BMI fields.
6. The driver writes existing output tables, trajectory data and the original
   versioned scenario JSON. The generator's native `completed_scenario()`
   command remains the artifact-persistence extension.

`complete_pending` on both adapters is a native **lifecycle command**, not a
standard BMI method. All *data exchanged between components* during live
placement execution travel through BMI values; the internal context and the
artifact/diagnostic extension APIs still exist. The public Monte Carlo driver
BMI now invokes the same BMI-mediated worker, so native CLI, parallel workers,
replay and external driver-BMI entry points share this execution path.

## Validation and safety

The model BMI refuses incomplete placement transfers, invalid CPS identifiers,
out-of-range or repeated parcel indices, mismatched affected fractions,
wrong efficiency dimensions, invalid wetland/grassed geometry, and cost values
outside the finite nonnegative domain. The coupler refuses mismatched units,
dtypes, ranks or unadvertised variables. The model adapter only permits
`placement__cost` after a response is pending. Variable-width input vectors
can be resized with ordinary BMI `set_value`; previously acquired pointers to
such vectors are invalid after resizing. Published output vectors can also
change shape between updates. `get_time_units()` remains `none`, not years.

The formulation registry and preflight compatibility policy from Phase 4 are
unchanged. This phase **does not approve** new scientific-method combinations
or alter any stochastic distributions, BMP-response formulas, AFO accounting,
wetland routing, pollutant delivery, or outputs.

## Limitations / future phases

- The BMI façade has not been independently certified against the official
  `bmipy` implementation/test harness in this environment.
- The generator/model share a validated prepared context. Full domain transfer
  (including real geospatial topology and ID lookups), model capabilities,
  initialization from external BMI configuration, and physical-time coupling
  remain future tasks.
- Generator and model still use their small native `complete_pending` lifecycle
  extensions to honor the legacy interleaved AFO/cost RNG sequence. Saved
  scenario definitions and optional AFO diagnostics also use native output
  extensions. The result JSON sidecar is numeric transport, but not a CSDMS
  Standard Name. Do not market this as arbitrary third-party model coupling.
- Cross-model comparisons need an explicit separation between stochastic BMP
  implementation parameters and model-response random draws; this version
  preserves the latter in recorded scenarios for exact legacy replay.
- Input `set_value_at_indices` is not used for variable-size transfers.
- Independent end-to-end Parquet and parallel checks need `pyarrow` or an
  equivalent Parquet engine. Workbook tests need `python-calamine`.

## Verification

Before editing, the preceding Phase 5 source state was recreated from the
last visible Phase 4 ZIP and the distributed Phase 5 Git patch. Eight seeded
scenarios were captured using the existing repo's fixture harness. The same
capture was run after this phase. The serialized JSON result, including all
BMP, parcel, AFO, summary and outlet trajectory records, matches **byte for
byte** (SHA-256 `1b63916915012a41120cf61a2e33ff76bc4372e5904a8a3d1f5d7a142cb7c208`).

`tests/test_phase6_bmi_coupling.py` includes numerical transport round-trips,
BMI-only transfer guards, incomplete/invalid input rejection, unit mismatch
checks and replay. Run the full suite in the intended environment:

```console
python -m pytest tests
```
