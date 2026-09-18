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
```

Parcel-side model variables are stored inside `parcels.gpkg`. The GeoPackage file may be renamed or moved; update only the YAML `parcels` path. Internal table names are the model schema.

## `parcels.gpkg`

### Structural tables

`parcels` is the required spatial layer. It contains only parcel identity and geometry:

- `pid` — unique integer parcel identifier and the GeoPackage feature/primary key (`INTEGER PRIMARY KEY`). There is no separate `fid` parcel identifier.
- geometry — polygon/multipolygon geometry.

All parcel references in other GeoPackage tables (`pid`, `pid_up`) are integers. Non-spatial attribute tables use a technical integer `id` primary key where QGIS needs a stable editable row identifier; the model ignores that `id`.

Parcel area and perimeter are derived from geometry after clipping to the domain.

`parcel_up` is an optional attribute table with one upstream edge per row:

```text
pid | pid_up
3   | 1
3   | 2
```

`parcel_outlets` is required and contains one parcel-to-outlet relationship per row:

```text
pid | oid
1   | O1
1   | O2
```

The structural tables do not contain modeled input variables.

A parcel GeoPackage may also contain the metadata table:

```text
model_input_schema
schema_version
3
```

Schema version `3` identifies the integer-PID/NULL-default one-variable-per-table layout with inline-only distributions (no named distribution IDs). Unknown
`input_*` tables are rejected so misspelled variable names cannot be silently
ignored.

### Editing in QGIS

Schema-v3 GeoPackages are designed to be editable directly in QGIS. The spatial `parcels` layer uses `pid` itself as its integer feature/primary key. Attribute-only tables have a separate technical `id INTEGER PRIMARY KEY` so QGIS can identify and edit individual rows; this `id` is not a parcel identifier and is ignored by the model.

For a default row, leave `pid` empty/NULL in QGIS. Do not enter `*`. For delivery-ratio defaults, leave both `pid` and `oid` NULL. Parcel-specific rows must use an existing integer `pid`. Because relationship and input tables reference `parcels.pid`, changing an existing parcel PID should be treated as a schema/data migration rather than an ordinary attribute edit.

## One variable per `input_*` table

Every parcel-side user input variable has its own GeoPackage attribute table. Numeric tables use the same standardized value/distribution columns:

```text
value, mean, sd, min,
p05, p10, p25, p50, p75, p90, p95, max,
sample_group, units, notes
```


Most parcel-specific variable tables are keyed by integer `pid`. Where a table supports a package-wide default, leave `pid` as SQL `NULL`; an exact integer `pid` row overrides that default. The legacy string `"*"` is invalid in PID fields.

### Parcel selection

`input_selection_weight`

```text
pid | value | units | notes
```

Weights are deterministic, finite, and nonnegative. If the table is absent, parcel selection is uniform. A row with `pid IS NULL` may supply a default.

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

and then use the standard numeric/distribution columns. Every land-cover × HSG pair actually used by modeled parcels must have both a curve-number definition and an infiltration-fraction definition. Reusable tables may contain additional valid pairings. Either variable may be fixed or stochastic.

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

The table uses the common numeric/distribution schema. `pid IS NULL` defaults and parcel-specific integer-PID overrides are supported.

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

These are deterministic fractions in `[0, 1]`. An exact parcel/outlet row overrides an optional global row where both `pid` and `oid` are SQL `NULL`. Partial defaults (only one key NULL) are invalid. If a table is absent, the neutral value `1.0` is used.

## `outlets.gpkg`

Configured with:

```yaml
outlets: ../outlets/outlets.gpkg
```

`outlets` is the required spatial layer with unique `oid` and point geometry.

`outlet_stats` is an optional attribute table keyed by `oid × pollutant` with optional nonnegative `target` and `mean` columns.

## Other CSV inputs


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
