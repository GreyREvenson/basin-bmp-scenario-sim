# Standardized numeric inputs and distributions

[← Back to main README](../readme.md)

## Purpose

Numeric model inputs use one common row-level convention. The convention is used by:

- statistical-mode parcel pollutant yields;
- PLET numeric parameters;
- the PLET land-cover/HSG Curve Number and infiltration table;
- RUSLE parameters;
- runoff concentrations;
- groundwater concentrations;
- BMP efficiencies; and
- BMP costs.

The goal is to make a fixed value and an uncertain input interchangeable without changing the structure of the model, while keeping large parcel datasets manageable.

## Canonical distribution columns

New input files should use these names where applicable:

```text
value, distribution_id, mean, sd, min, p05, p50, p95, max
```

Other percentile levels (`p10`, `p25`, `p75`, `p90`, etc.) are also supported. Metadata columns such as `units`, `unit`, `notes`, and identifiers such as `pid`, `pollutant`, `parameter`, `pathway`, or `cps` depend on the input table.

Recognized legacy aliases remain accepted:

- `average` / `avg` → `mean`
- `std` → `sd`
- `minimum` → `min`
- `maximum` → `max`
- `p0` → `min`
- `p100` → `max`

Use the canonical names for new files.

## Accepted numeric forms

Each numeric row must use exactly one coherent specification.

| Form | Required statistics | Sampling behavior |
|---|---|---|
| Fixed | `value` | Always returns the supplied value |
| Normal | `mean`, `sd` | Normal sampling, truncated when semantic/physical bounds apply |
| Bounded normal | `mean`, `sd`, `min`, `max` | Truncated normal within the stated bounds |
| Legacy bounded normal | `min`, `mean`, `max` | Truncated normal with inferred `sd = (max - min) / 4` |
| Uniform | `min`, `max` | Uniform between the endpoints |
| Percentile distribution | `min`, one or more `pXX`, `max` | Piecewise-linear inverse-CDF sampling |
| Reusable named distribution | `distribution_id` | Uses a definition from `input_distributions.csv` |

Examples:

```text
Fixed:                value=8.5
Normal:               mean=8.5, sd=1.2
Bounded normal:       mean=8.5, sd=1.2, min=5, max=12
Legacy bounded normal: mean=8.5, min=5, max=12
Uniform:              min=5, max=12
Percentile:           min=5, p10=6, p50=8.5, p90=11, max=12
```

### Invalid/ambiguous combinations

The loader rejects ambiguous rows. In particular:

- do not mix `value` with distribution statistics;
- do not combine `distribution_id` with inline distribution statistics;
- do not provide only one of `min` or `max`;
- do not combine percentile statistics with `mean`/`sd`; and
- percentile distributions require both `min` and `max` endpoints.

The loader also checks finite numeric values, nonnegative `sd`, ordered bounds, and monotonic percentile values.

## Reusable distribution catalog

A configuration may define an optional catalog:

```yaml
input_distributions: ./path/to/input_distributions.csv
```

The catalog contains one row per `distribution_id` and uses the same distribution columns:

```csv
distribution_id,value,mean,sd,min,p05,p50,p95,max,units,notes
annual_precip_default,,42,3,34,,,,50,in/year,Example bounded normal
runoff_tn_default,3,,,,,,,,mg/L,Fixed value
cover_crop_surface_tn,,0.35,,0.15,,,,0.55,fraction,Legacy min/mean/max form
```

A use-site row can then be short:

```csv
pid,parameter,distribution_id,units
*,annual_precip_in,annual_precip_default,in/year
```

`distribution_id` reuses the **distribution definition**. It does **not** mean that different parcels receive the same sampled number.

The catalog is most useful when many rows share the same uncertainty assumption. If each parcel genuinely has a unique distribution, put that distribution directly on the parcel row rather than creating thousands of one-use catalog IDs.

The East Fork statistical example demonstrates the intended scaling pattern. Its previous `pollutant_yield.csv` repeated the same three pollutant distributions across 2,681 parcels (8,043 data rows). The standardized example stores those three distributions once in `input_distributions.csv` and uses three `pid=*` rows in `pollutant_yield.csv`; each parcel still receives its own independent sample unless a shared draw is explicitly requested.

## Parcel defaults and overrides

For parcel-indexed inputs, `pid="*"` means “use this row as the default for every parcel.” An exact parcel row overrides the wildcard row for the same variable.

Example:

```csv
pid,parameter,value,distribution_id,mean,sd,min,max,units
*,annual_precip_in,,,42,3,34,50,in/year
P104,annual_precip_in,,,46,2,40,52,in/year
```

All parcels except `P104` sample from the first distribution. `P104` samples from the second.

This pattern is recommended when a watershed contains thousands of parcels and many share assumptions. It avoids repeating identical rows while retaining explicit parcel-level overrides.

Statistical-mode `pollutant_yield` supports the same default/override pattern. For pathway-specific yields, the override key is parcel + pollutant + pathway. For aggregate yields, it is parcel + pollutant.

## Independent versus shared draws

A wildcard row or `distribution_id` does not create correlation by itself. The normal behavior is an independent draw for each parcel.

For the parcel-indexed PLET parameter/concentration tables and the PLET hydrology table, an explicit `sample_group` can request a shared draw:

```csv
pid,parameter,mean,sd,sample_group
*,annual_precip_in,42,3,watershed_year
```

All affected parcels receive the same annual-precipitation draw within that scenario.

A sample group is variable-specific. Assigning the same text label to different variables does not create a multivariate/correlated distribution between those variables.

## Required PLET hydrology input

`plet_rusle` mode requires:

```yaml
load_generation:
  mode: plet_rusle
  hydrology_lookup: ./path/to/plet_hydrology_lookup.csv
```

The table is a user input, not a source-code lookup. It uses long form:

```csv
land_cover,hsg,parameter,value,distribution_id,mean,sd,min,p05,p50,p95,max,sample_group,units
cropland,B,cn,78,,,,,,,,,,,dimensionless
cropland,B,infiltration_fraction,0.30,,,,,,,,,,,fraction
```

For every supported land-cover/HSG pairing the file must contain exactly one `cn` row and exactly one `infiltration_fraction` row. Supported land covers are:

- `urban`
- `cropland`
- `pastureland`
- `forest`
- `user_defined`

Supported HSG values are `A`, `B`, `C`, and `D`. This produces 40 required parameter rows: 5 land covers × 4 HSGs × 2 parameters.

CN must remain in `(0, 100]`. Infiltration fraction must remain in `[0, 1]`. A fixed value or a distribution can be supplied for either parameter.

Example stochastic pair:

```csv
land_cover,hsg,parameter,value,distribution_id,mean,sd,min,max,units
cropland,B,cn,,,78,2,70,86,dimensionless
cropland,B,infiltration_fraction,,,0.30,0.03,0.20,0.40,fraction
```

Each parcel classified as cropland/HSG B samples those pair-specific distributions independently unless `sample_group` is supplied.

## Recommended file organization

For a PLET/RUSLE project:

```text
inputs/
  plet/
    input_distributions.csv       # optional reusable distributions
    plet_inputs.csv               # parcel PLET variables + land_cover/HSG
    plet_hydrology_lookup.csv     # REQUIRED CN/infiltration by class pair
    rusle_inputs.csv              # RUSLE variables
    pollutant_concentrations.csv  # surface-runoff concentrations
    groundwater_concentrations.csv
    bmp_efficiency.csv
    bmp_cost.csv
    ...spatial/routing inputs...
```

The organization deliberately separates different scientific variable families while keeping the numeric specification syntax the same in every file.