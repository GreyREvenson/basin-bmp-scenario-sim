# PLET/RUSLE load-generation mode

[← Back to main README](../readme.md)

`plet_rusle` mode derives annual surface and subsurface loads from user-supplied parcel parameters, hydrology assumptions, pollutant concentrations, and optional RUSLE sediment parameters.

## Configuration

```yaml
domain: ../domain/domain.gpkg
parcels: ../parcels/parcels.gpkg
outlets: ../outlets/outlets.gpkg

pollutants: [TN, TP, TSS]
cps: [340, 329, 590, 412, 656]

bmp_efficiency: ../bmps/bmp_efficiency.csv
input_distributions: ../misc/input_distributions.csv

load_generation:
  mode: plet_rusle
```

All parcel-side PLET/RUSLE inputs live in dedicated `input_*` tables inside `parcels.gpkg`. No PLET parcel-input path or hydrology-lookup path is configured in YAML.

## Parcel PLET tables

Each variable gets its own table keyed by `pid`:

- `input_annual_precip_in`
- `input_rain_days`
- `input_rain_correction_fraction`
- `input_runoff_day_fraction`
- `input_land_cover`
- `input_hsg`
- `input_ia_ratio`

Numeric tables use the standard fixed-value/distribution schema. `input_land_cover` and `input_hsg` require fixed categorical values. `pid="*"` defaults are supported.

## User-controlled hydrology

Curve number and infiltration fraction are also separate user-input tables inside `parcels.gpkg`:

```text
input_curve_number
input_infiltration_fraction
```

Both are keyed by `land_cover × hsg` and use the same numeric/distribution schema as other numeric inputs. Every supported pair must be covered by both tables.

These are the values actually used by PLET. They may be fixed or distributions. There is no source-code fallback and no external `hydrology_lookup` configuration key.

## RUSLE/sediment tables

- `input_rusle_r`
- `input_rusle_k`
- `input_rusle_ls`
- `input_rusle_c`
- `input_rusle_p`
- `input_sediment_delivery_ratio`
- `input_sediment_n_pct`
- `input_sediment_p_pct`
- `input_enrichment_ratio`

The input layer assembles these dedicated tables into the existing runtime RUSLE parameter table, so the simulation equations and parallel scenario execution remain unchanged.

## Pollutant concentrations

Surface and subsurface concentrations are distinct variables:

```text
input_surface_concentration
input_subsurface_concentration
```

Each table is keyed by `pid × pollutant` and uses the common numeric/distribution schema. `pid="*"` supplies defaults.

Surface concentrations are required for TN and TP and for TSS when complete RUSLE inputs are unavailable. Subsurface concentrations are required for modeled non-TSS pollutants.

## BMP pathways

PLET/RUSLE production pathways are fixed to `surface` and `subsurface`. Surface BMP efficiencies are required. Missing correctly labeled subsurface efficiencies are completed with zero and logged.
