# Phase 8 — Detached scenario-generator context and BMI parcel-domain transfer

## Scope and rationale

Before Phase 8, `AnnualBmiScenarioCoupler` constructed the live scenario
sampler and the scientific model around **the same mutable worker context**.
The generator depended indirectly on scientific-model baseline state and on
its live `np.random.Generator` object. Its wetland and grassed-waterway
placement heuristics read parcel areas, perimeters, and immediate upstream
adjacency from that shared object. An index manifest alone did **not**
transfer that physical placement domain.

This patch separates the **component-owned state** while retaining the
approved scientific methods and the exact legacy stochastic sequence:

- `GeneratorModelContext` is a new, small placement/cost/efficiency sampling
  context. It has its own RNG, copies of required configuration and cost/AFO
  inputs, and private parcel-domain/selection arrays. It imports only
  sampling/selection/cost helpers; it has no scientific load-state object.
- `WatershedModelBmi` retains ownership of the scientific `ScenarioModelContext`.
  That context stays attached to the driver so existing load diagnostics,
  AFO audit tables, and output code need no numerical or file-schema changes.
- `WatershedModelBmi` **publishes** a versioned parcel-domain packet through
  standard BMI `get_value`; `LiveScenarioGeneratorBmi` **receives** it through
  BMI `set_value`, validates the packet as a whole, and replaces its own
  physical-domain arrays **before the first placement draw**.
- The two components exchange their separate NumPy RNG bit-generator states
  as UTF-8 JSON in BMI `uint8` arrays at the *same ordering boundaries* as
  the old shared generator: baseline -> design -> AFO response -> cost ->
  next design. This preserves the single logical random stream without a
  shared mutable RNG reference.

## Protocol version and fields

`EXCHANGE_SCHEMA_VERSION` is now **2**. The index manifest still has its own
version 1; the new parcel-domain packet has `DOMAIN_SCHEMA_VERSION = 1`.
This is a **breaking coupling-contract revision**: a Phase 7 generator that
lacks the required new domain and RNG interfaces must not be coupled to a
Phase 8 model. The driver verifies the index manifest and schema before
transferring the physical domain.

| Scientific-model output -> generator input | Units | Shape and meaning |
| --- | --- | --- |
| `domain__schema_version` | 1 | Scalar packet version |
| `domain__parcel_area_ha` | ha | Full hydrologic parcel-universe area, ordered |
| `domain__parcel_perimeter_m` | m | Matching parcel perimeter |
| `domain__upstream_offset` | 1 | CSR offsets, parcel count plus one |
| `domain__upstream_index` | 1 | Flattened immediate-upstream indices in the original order |
| `domain__selection_source_index` | 1 | BMP-eligible selection positions -> full parcel-universe indices |
| `domain__selection_probability` | 1 | Matching placement-selection probability weights |

The **already existing** `coupling__index_manifest_utf8` sidecar provides
parcel and outlet IDs, pollutant/pathway order, and configured CPS IDs. The
new domain packet does not assert geographic coordinates or polygon geometry:
areas, perimeters and topological connectivity are sufficient for the
**current** BMP-placement heuristics only. This is **not** a physical CSDMS
raster, an irregular-grid definition, or a complete hydrologic model input.

| BMI value | Direction | Meaning |
| --- | --- | --- |
| `coupling__model_rng_state_utf8` | model output -> generator input | RNG state after baseline/AFO draws |
| `coupling__generator_rng_state_utf8` | generator output -> model input | RNG state after design/cost draws |

RNG sidecars are project-local and use a constrained versioned JSON encoding
of NumPy bit-generator state; they are **not** portable stochastic standards.
Malformed states, wrong bit-generator types, and incompatible schema versions
are rejected before mutating the target stream.

## Live execution order

1. Construct two contexts from the prepared worker configuration, one for the
   scientific model and one for placement generation. Do not share their
   mutable arrays, cost dataframes, or RNG objects.
2. Initialize the scientific model and sample annual baseline exactly once.
3. Initialize the live generator, check the Phase 8 index/exchange contract,
   transfer all physical-domain fields via BMI, and install the domain only
   after atomic validation.
4. **Every placement:** transfer model RNG -> generator; generator `update()`
   draws placement, efficiency, failure, and geometry.
5. Transfer placement arrays and generator RNG -> scientific model; model
   `update()` applies BMP and AFO recovery.
6. Transfer model RNG and numeric recovery gains -> generator; generator
   `complete_pending()` draws or reuses implementation cost.
7. Transfer cost and generator RNG -> scientific model; model
   `complete_pending()` finalizes signed mass accounting and published loads.
8. Repeat without changing existing annual model outputs or persisted scenario
   JSON; the driver still handles output writing and ensemble orchestration.

## Validation

- All packet dimensions, dtypes, schema versions, geometry nonnegativity,
  finite values, CSR offsets, in-range/unique selection indices, matching
  selection IDs and normalized weights are checked before generator mutation.
- Immediate upstream adjacency rejects self-links, repeated neighbors and
  directed cycles (no sorting/reindexing of valid connections).
- Component RNGs never reference the same Python generator object; explicit
  BMI transfers restore the same logical draw state at existing draw
  boundaries. RNG payload validation uses a throwaway bit generator before
  applying state to the live one.
- New tests verify context/array isolation, domain packet transfer, validation
  failure before placement, bad RNG rejection without state mutation, normal
  AFO recovery, exact replay and version enforcement.

The eight seeded scientific-output captures taken **before changing the exact
`(7)` ZIP** were compared byte for byte after the patch, including all
per-scenario tables and outlet trajectories. Both canonical JSON snapshots
have SHA-256:

`1b63916915012a41120cf61a2e33ff76bc4372e5904a8a3d1f5d7a142cb7c208`

The original golden-output and Phase 2 replay test suites also pass after the
change. Please run the complete test suite in an environment with `pyarrow`
and `python-calamine` installed; these dependencies were not available when
this patch was developed. The optional `bmipy` comparison remains skipped
unless `pip install -e '.[bmi]'` is used.

## Deliberate limits / future phase

**This is independent component *state*, not independent stand-alone process
initialization.** The driver still validates the input files and constructs
both contexts from one prepared worker payload. BPM efficiency priors, cost
curves, AFO inputs and approved scientific formulation are still injected
through project-native Python configuration, not through an independently
serialized BMI component-init contract. Driver-owned output diagnostics also
remain native. The Phase 8 domain arrays are the physical subset needed by
current placement heuristics, not complete geospatial polygons, runoff
routing, or an external environmental model.

A future phase should introduce explicit, validated independent configuration
and capability manifests for both components, with component-specific input
loading, no implicit access to the driver's shared payload, and third-party
BMI harness tests. Splitting **BMP implementation uncertainty** from the
stochastic model response is a separate scientific decision; this patch does
not invent or change those distributions.
