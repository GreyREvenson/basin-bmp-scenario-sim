# Input and output reference

[← Back to main README](../readme.md)

## Physical input files

A typical model run uses these physical inputs:

```text
inputs/
  config/
    model.yaml
  domain/
    domain.gpkg
  parcels/
    parcels.gpkg
  outlets/
    outlets.gpkg
  plet/
    plet_hydrology_lookup.csv       # plet_rusle only
  bmps/
    bmp_efficiency.csv
    bmp_cost.csv                    # optional
  misc/
    input_distributions.csv         # optional
```

Parcel and outlet data are consolidated into GeoPackages. The model does not accept separate `parcel_out`, `parcel_up`, `parcel_p`, `pollutant_load_rate`, `outlet_loc`, `outlet_target`, `outlet_mean`, or delivery-ratio file paths.

## `parcels.gpkg`

The file path is configured with:

```yaml
parcels: ../parcels/parcels.gpkg
```

The filename and location are arbitrary. The internal table/layer names below are the model schema.

### `parcels` — required spatial layer

One feature per modeled parcel.

Required columns:

- `pid` — unique parcel identifier.
- geometry — parcel polygon/multipolygon geometry.

Optional columns:

- `selection_weight` — finite nonnegative weight used to derive parcel-selection probabilities. If absent, all modeled parcels receive equal selection probability. At least one weight must be positive when the column is present.

Parcel area and perimeter are derived from geometry after clipping to the domain.

### `parcel_up` — optional attribute table

Normalized directed parcel connectivity:

```text
pid | pid_up
P3  | P1
P3  | P2
```

Each row is one edge. Wildcards and comma-separated lists are not supported. If the table is absent, all parcels are treated as having no upstream parcels.

### `parcel_outlets` — required attribute table

Normalized parcel-to-outlet relationships:

```text
pid | oid
P1  | O1
P1  | O2
P2  | O2
```

Each row is one relationship. Every referenced `pid` must exist in `parcels`; every referenced `oid` must exist in the `outlets` layer.

Optional delivery-ratio columns may be included directly on the same relationship table:

- `sdr_f_to_s`
- `sdr_s_to_o`
- `ndr_f_to_s`
- `ndr_s_to_o`

If none are present, neutral values of 1.0 are used. If delivery ratios are supplied, all four columns are required and values must lie in `[0, 1]`.

### `pollutant_load_rates` — required in statistical mode

Long-form parcel pollutant load-rate rows using the common numeric/distribution schema.

Keys:

- `pid`
- `pollutant`
- optional `pathway`

`pid="*"` may define defaults for all parcels; exact parcel rows override matching defaults.

### `parcel_parameters` — required in `plet_rusle` mode

One combined long-form table for parcel PLET and RUSLE parameters.

Keys:

- `pid`
- `parameter`

The table uses the common numeric/distribution columns. `pid="*"` defaults and parcel-specific overrides are supported.

PLET parameters include climate terms and the fixed `land_cover` and `hsg` classifications. Curve number (`cn`) and `infiltration_fraction` do **not** belong here; they come from the separately configured hydrology lookup.

RUSLE parameters (`r`, `k`, `ls`, `c`, `p`, `sdr`, `sediment_n_pct`, `sediment_p_pct`, and `enrichment_ratio`) live in this same table. A parcel using RUSLE must have the required complete factor set.

### `pollutant_concentrations` — required in `plet_rusle` mode

Unified PLET concentration table.

Keys:

- `pid`
- `pollutant`
- `pathway`

`pathway` must be either `surface` or `subsurface`. Numeric rows use the common fixed-value/distribution schema. `pid="*"` defaults and parcel-specific overrides are resolved independently by pollutant and pathway.

## `outlets.gpkg`

Configured with:

```yaml
outlets: ../outlets/outlets.gpkg
```

### `outlets` — required spatial layer

Required columns:

- `oid` — unique outlet identifier.
- geometry — outlet point geometry.

Other descriptive metadata columns are allowed.

### `outlet_stats` — optional attribute table

One row per outlet × pollutant:

```text
oid | pollutant | target | mean
O1  | TN        | ...    | ...
O1  | TP        | ...    | ...
```

`target` and `mean` are optional columns. The model uses whichever are present and nonblank. Values must be nonnegative.

## User-supplied PLET hydrology lookup

`load_generation.hydrology_lookup` remains a separate CSV and is required in `plet_rusle` mode.

It defines `cn` and `infiltration_fraction` for each supported land-cover × HSG pairing. Either parameter may be fixed or stochastic using the common distribution schema. There is no source-code fallback; the active configured table controls the values used in the run.

## Other CSV inputs

### `input_distributions`

Optional reusable distribution catalog keyed by `distribution_id`.

### `bmp_efficiency`

BMP efficiency values/distributions by CPS, pollutant, and pathway where applicable.

### `bmp_cost`

Optional BMP cost values/distributions. The cost unit remains required because cost scaling depends on it.

## Common numeric schema

Numeric rows use some combination of:

```text
value, distribution_id, mean, sd, min, p05, p50, p95, max
```

Additional percentile columns such as `p10`, `p25`, `p75`, and `p90` are supported where valid. See [Standardized numeric inputs and distributions](input_distributions.md).

## Output directory

Canonical outputs below the configured `outputs` directory include:

- `bmps/s{scenario}.parquet`
- `parcels/s{scenario}.parquet`
- `load_parameters/s{scenario}.parquet` for PLET/RUSLE diagnostics
- `scenario_metrics/s{scenario}.parquet`
- `outlet_trajectories/all_scenarios.parquet`
- `log.txt` and scenario logs
- summary plot files
