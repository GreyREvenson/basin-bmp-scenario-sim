# Input and output reference

[← Back to main README](../readme.md)

## Physical input files

A typical run uses:

```text
inputs/
  config/model.yaml
  domain/domain.gpkg
  parcels/parcels.gpkg
  outlets/outlets.gpkg
  bmps/bmp_efficiency.csv
  bmps/bmp_cost.csv              # optional
  misc/input_distributions.csv   # optional
```

Parcel-side model variables are stored inside `parcels.gpkg`. The GeoPackage file may be renamed or moved; update only the YAML `parcels` path. Internal table names are the model schema.

## `parcels.gpkg`

### Structural tables

`parcels` is the required spatial layer. It contains only parcel identity and geometry:

- `pid` — unique parcel identifier.
- geometry — polygon/multipolygon geometry.

Parcel area and perimeter are derived from geometry after clipping to the domain.

`parcel_up` is an optional attribute table with one upstream edge per row:

```text
pid | pid_up
P3  | P1
P3  | P2
```

`parcel_outlets` is required and contains one parcel-to-outlet relationship per row:

```text
pid | oid
P1  | O1
P1  | O2
```

The structural tables do not contain modeled input variables.

## One variable per `input_*` table

Every parcel-side user input variable has its own GeoPackage attribute table. Numeric tables use the same standardized value/distribution columns:

```text
value, distribution_id, mean, sd, min,
p05, p10, p25, p50, p75, p90, p95, max,
sample_group, units, notes
```

Only the columns needed to define a row need values. A row must define either a fixed `value`, a `distribution_id`, or a valid inline distribution. See [Standardized numeric inputs and distributions](input_distributions.md).

Most parcel-specific variable tables are keyed by `pid`. `pid="*"` supplies a default; an exact `pid` row overrides the default.

### Parcel selection

`input_selection_weight`

```text
pid | value | units | notes
```

Weights are deterministic, finite, and nonnegative. If the table is absent, parcel selection is uniform. A `pid="*"` row may supply a default.

### PLET parcel variables

Each of these tables is keyed by `pid`:

- `input_annual_precip_in`
- `input_rain_days`
- `input_rain_correction_fraction`
- `input_runoff_day_fraction`
- `input_land_cover`
- `input_hsg`
- `input_ia_ratio`

`input_land_cover` and `input_hsg` are categorical and therefore require fixed `value` entries; distributions are not allowed.

### Curve number and infiltration fraction

PLET hydrology assumptions are user inputs stored in the same parcel GeoPackage:

- `input_curve_number`
- `input_infiltration_fraction`

These two tables are keyed by:

```text
land_cover | hsg
```

and then use the standard numeric/distribution columns. Every supported land-cover × HSG pair must have both a curve-number definition and an infiltration-fraction definition. Either variable may be fixed or stochastic.

There is no separate `hydrology_lookup` file or source-code fallback.

### RUSLE and sediment variables

Each table is keyed by `pid`:

- `input_rusle_r`
- `input_rusle_k`
- `input_rusle_ls`
- `input_rusle_c`
- `input_rusle_p`
- `input_sediment_delivery_ratio`
- `input_sediment_n_pct`
- `input_sediment_p_pct`
- `input_enrichment_ratio`

The loader assembles these tables into the existing runtime RUSLE parameter representation before scenario execution.

### Statistical load rate

`input_pollutant_load_rate` is required in statistical mode.

Keys:

- `pid`
- `pollutant`
- optional `pathway`

The table uses the common numeric/distribution schema. `pid="*"` defaults and parcel-specific overrides are supported.

### PLET pollutant concentrations

PLET/RUSLE uses two separate input variables:

- `input_surface_concentration`
- `input_subsurface_concentration`

Both are keyed by:

```text
pid | pollutant
```

and use the common numeric/distribution schema. The pathway is implied by the table name rather than stored in a `pathway` column.

### Delivery-ratio variables

The parcel-to-outlet relationship remains in `parcel_outlets`. Each optional delivery variable is stored separately:

- `input_sdr_f_to_s`
- `input_sdr_s_to_o`
- `input_ndr_f_to_s`
- `input_ndr_s_to_o`

Keys:

```text
pid | oid
```

These are deterministic fractions in `[0, 1]`. An exact parcel/outlet row overrides an optional global `pid="*", oid="*"` default. If a table is absent, the neutral value `1.0` is used.

## `outlets.gpkg`

Configured with:

```yaml
outlets: ../outlets/outlets.gpkg
```

`outlets` is the required spatial layer with unique `oid` and point geometry.

`outlet_stats` is an optional attribute table keyed by `oid × pollutant` with optional nonnegative `target` and `mean` columns.

## Other CSV inputs

`input_distributions` is an optional reusable distribution catalog keyed by `distribution_id`.

`bmp_efficiency` defines BMP efficiency values/distributions by CPS, pollutant, and pathway where applicable.

`bmp_cost` is optional and defines BMP cost inputs.

## Canonical outputs

Below the configured `outputs` directory, the model writes:

- `bmps/s{scenario}.parquet`
- `parcels/s{scenario}.parquet`
- `load_parameters/s{scenario}.parquet` for PLET/RUSLE diagnostics
- `scenario_metrics/s{scenario}.parquet`
- `outlet_trajectories/all_scenarios.parquet`
- logs and summary plots
