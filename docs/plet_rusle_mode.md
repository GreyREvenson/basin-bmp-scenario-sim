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
  hydrology_lookup: ../plet/plet_hydrology_lookup.csv
```

PLET/RUSLE parcel inputs are stored in the consolidated parcel GeoPackage rather than configured as separate files.

## `parcel_parameters`

The `parcel_parameters` table combines the former PLET and RUSLE long-form parameter tables. Each row is keyed by `pid` and `parameter` and uses the standard fixed-value/distribution columns.

PLET inputs include:

- `annual_precip_in`
- `rain_days`
- `rain_correction_fraction`
- `runoff_day_fraction`
- `land_cover`
- `hsg`
- optional supported PLET parameters such as `ia_ratio`

`land_cover` and `hsg` are fixed classifications. `cn` and `infiltration_fraction` must not be supplied as parcel parameters; they are resolved from the hydrology lookup.

RUSLE parameters live in the same table:

- `r`
- `k`
- `ls`
- `c`
- `p`
- `sdr`
- `sediment_n_pct`
- `sediment_p_pct`
- `enrichment_ratio`

`pid="*"` may supply defaults, with parcel-specific rows overriding defaults.

## Hydrology lookup

`load_generation.hydrology_lookup` is a separate, required user input. It defines both `cn` and `infiltration_fraction` for supported land-cover × HSG pairs.

Both may be fixed values or probability distributions. This table deliberately remains outside the source tree and outside the parcel GeoPackage so a user can replace hydrologic assumptions independently for a run.

## `pollutant_concentrations`

The parcel GeoPackage contains one unified concentration table with:

```text
pid | pollutant | pathway | ...numeric/distribution columns...
```

`pathway` must be:

- `surface` — combined with PLET runoff volume.
- `subsurface` — combined with PLET infiltration volume.

Surface concentrations are required for TN and TP and for TSS when complete RUSLE sediment inputs are unavailable. Subsurface concentrations are required for modeled non-TSS pollutants.

## BMP pathways

Production pathways are fixed to `surface` and `subsurface`. Surface BMP efficiencies are required. Missing correctly labeled subsurface efficiencies are completed with zero and logged.

## Removed inputs

The following are not part of the consolidated interface:

- separate `plet_inputs` path
- separate `rusle_inputs` path
- separate `pollutant_concentrations` path
- `pathway_mode`
- `watershed_area_mi2` and equivalent spellings

The user-controlled hydrology lookup remains the sole source of curve-number and infiltration-fraction assumptions.
