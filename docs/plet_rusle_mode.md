# PLET/RUSLE load-generation mode

[← Back to main README](../readme.md)

## Purpose

`plet_rusle` generates annual parcel pollutant yields internally using PLET-style runoff and infiltration calculations, pollutant concentrations, and optional RUSLE sediment calculations. It replaces the statistical-mode `pollutant_yield` input.

The production model has exactly two pathways:

1. `surface`
2. `subsurface`

Earlier shallow/deep subsurface splits are not part of the production `plet_rusle` pathway representation.

## Required PLET parcel inputs

Every modeled parcel must resolve these PLET inputs:

- `annual_precip_in`;
- `rain_days`;
- `rain_correction_fraction`;
- `runoff_day_fraction`;
- `land_cover`; and
- `hsg`.

`land_cover` and `hsg` are classifications and must be supplied as fixed values rather than sampled statistical distributions.

Supported land-cover classes are:

- `urban`;
- `cropland`;
- `pastureland`;
- `forest`; and
- `user_defined`.

HSG must be one of `A`, `B`, `C`, or `D`.

Example long-form PLET table:

```csv
pid,parameter,value,distribution_id,mean,sd,min,max,units
*,annual_precip_in,,annual_precip_default,,,,,in/year
*,rain_days,,,100,10,80,120,days/year
*,rain_correction_fraction,0.90,,,,,,fraction
*,runoff_day_fraction,0.25,,,,,,fraction
*,land_cover,cropland,,,,,,classification
*,hsg,B,,,,,,classification
P205,land_cover,pastureland,,,,,,classification
P205,hsg,C,,,,,,classification
```

Rows with `pid="*"` provide defaults. Parcel-specific rows override the default for the same parameter. Numeric wildcard rows are sampled independently for each parcel unless an explicit `sample_group` requests a shared draw.

## Required land-cover/HSG hydrology input

Curve Number and infiltration fraction are no longer hidden source-code constants. `plet_rusle` requires a user-supplied hydrology input table:

```yaml
load_generation:
  mode: plet_rusle
  plet_inputs: ./inputs/plet/plet_inputs.csv
  hydrology_lookup: ./inputs/plet/plet_hydrology_lookup.csv
```

The table must contain exactly one `cn` row and one `infiltration_fraction` row for every supported land-cover × HSG pairing. With five supported land-cover classes and four HSG classes, the complete table has 40 parameter rows.

```csv
land_cover,hsg,parameter,value,distribution_id,mean,sd,min,max,units
cropland,B,cn,78,,,,,,dimensionless
cropland,B,infiltration_fraction,0.30,,,,,,fraction
pastureland,C,cn,,,79,2,72,86,dimensionless
pastureland,C,infiltration_fraction,,,0.15,0.02,0.10,0.22,fraction
```

Both quantities may therefore be deterministic or uncertain:

```text
parcel land_cover + HSG
          ↓
required hydrology input row
          ↓
sample CN + infiltration fraction
          ↓
PLET runoff + infiltration calculations
```

CN must remain in `(0, 100]`. Infiltration fraction must remain in `[0, 1]`. The loader validates each distribution's stated support/statistics against these physical limits.

A fixed or uncertain class-pair definition is sampled independently for each parcel that uses that pairing. If a scenario should intentionally use one shared hydrologic draw for multiple parcels, set `sample_group` explicitly on the corresponding row.

The old source file:

```text
src/data/plet_hydrology_lookup.csv
```

is no longer a production model input and should be removed. The East Fork example places the table at:

```text
examples/east_fork/inputs/plet/plet_hydrology_lookup.csv
```

The example preserves the previous PLET reference values as fixed `value` entries, but users can replace any CN or infiltration-fraction row with a distribution.

Supplying `cn` or `infiltration_fraction` directly in `plet_inputs.csv` is rejected. The hydrology table is the single source of those parameters in `plet_rusle` mode.

## Surface runoff

The PLET-style calculation derives the number of runoff-producing days and representative event precipitation from annual precipitation, rain-day frequency, rainfall correction, and runoff-day fraction. The sampled Curve Number is then used in the SCS/NRCS runoff relationship.

Conceptually:

```text
runoff days = rain_days × runoff_day_fraction

event rainfall = annual_precip_in × rain_correction_fraction / runoff_days
```

Potential retention is:

```text
S = 1000 / CN - 10
```

The event runoff calculation uses the configured initial-abstraction ratio. The default `ia_ratio` is 0. Annual runoff is event runoff multiplied by runoff days.

## Infiltration and subsurface load

Annual infiltration is calculated from annual precipitation and the sampled land-cover/HSG infiltration fraction:

```text
annual infiltration
  = annual_precip_in
  × rain_correction_fraction
  × infiltration_fraction
  × groundwater_multiplier
```

The default `groundwater_multiplier` is 1.0.

For every modeled non-TSS pollutant, `groundwater_concentrations` is required. Subsurface yield is calculated from groundwater concentration and annual infiltration volume:

```text
subsurface load
  = groundwater concentration × annual infiltration volume
```

TSS has no subsurface component.

## Surface pollutant loads

`pollutant_concentrations` supplies runoff concentrations and is required when TN or TP is modeled.

The concentration-derived component is:

```text
surface runoff load
  = runoff concentration × annual runoff volume
```

For TSS, a runoff concentration may be used when complete RUSLE inputs are not available.

## RUSLE sediment

RUSLE input is optional, but if any RUSLE values are supplied for a parcel the required factor set must be complete:

- `r`;
- `k`;
- `ls`;
- `c`; and
- `p`.

The parcel must also provide either `sdr` or `watershed_area_mi2`, allowing a sediment delivery ratio to be supplied or derived according to the current model formulation.

Gross erosion is based on:

```text
R × K × LS × C × P
```

and the model applies sediment delivery and configured sediment multipliers before converting to annual mass per area.

When complete RUSLE inputs exist:

```text
surface TSS = delivered RUSLE sediment yield
```

For TN and TP, sediment-associated nutrient mass can be added using:

- `sediment_n_pct`;
- `sediment_p_pct`; and
- `enrichment_ratio`.

Conceptually:

```text
TN_surface
  = runoff TN load
  + sediment yield × sediment N fraction × enrichment ratio

TP_surface
  = runoff TP load
  + sediment yield × sediment P fraction × enrichment ratio
```

Therefore:

```text
TN_total = TN_surface + TN_subsurface
TP_total = TP_surface + TP_subsurface
TSS_total = TSS_surface
```

## BMP efficiencies in `plet_rusle`

BMP efficiencies operate on the two production pathways independently and use the same standardized fixed-value/distribution columns as other numeric inputs.

```csv
cps,pollutant,pathway,value,distribution_id,mean,sd,min,max
340,TN,surface,,,0.35,,0.20,0.50
340,TN,subsurface,0,,,,,
340,TP,surface,,,0.25,,0.10,0.40
```

### Surface efficiency

A correctly labeled `surface` efficiency is required for every configured CPS × pollutant combination. Missing surface coverage is an error.

### Subsurface efficiency

A correctly labeled `subsurface` efficiency is optional. If it is absent or contains no usable statistics, the model inserts:

```text
subsurface efficiency = 0
```

and logs a warning.

Unexpected pathway labels such as `shallow subsurface` or `deep subsurface` are not remapped to the PLET `subsurface` pathway. They are ignored in this mode, a warning is logged, and the subsurface efficiency defaults to zero if no valid `subsurface` row remains.

This prevents an incorrectly labeled row from silently changing infiltration-derived nutrient treatment.

## Standardized values and distributions

PLET parameters, RUSLE parameters, runoff concentrations, groundwater concentrations, hydrology inputs, BMP efficiencies, and BMP costs can all use the common fixed-value/distribution schema where those inputs are numeric. Reusable definitions can be placed in `input_distributions.csv` and referenced with `distribution_id`.

See [Standardized numeric inputs and distributions](input_distributions.md).

## Removed/legacy configuration concepts

`load_generation.pathway_mode` has been removed. `plet_rusle` always derives its two production pathways from PLET/RUSLE inputs.

Legacy `groundwater_loads` and `treat_groundwater_with_bmps` keys do not determine production pathway generation. The current production calculation always estimates subsurface load from groundwater concentration and sampled infiltration. Whether a BMP reduces that load is determined by the BMP's `subsurface` efficiency.

`pollutant_yield` and statistical pathway-fraction settings are not used to generate baseline PLET/RUSLE loads.

## Main distinction from statistical mode

In statistical mode, the user supplies the baseline yields and defines the pathways. In `plet_rusle`, the model calculates baseline yields and the pathway meanings are fixed by the hydrologic formulation.
