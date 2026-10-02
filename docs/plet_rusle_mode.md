# PLET/RUSLE load-generation mode

[← Back to main README](../readme.md)

`plet_rusle` mode derives annual surface and subsurface loads from user-supplied parcel parameters, hydrology assumptions, pollutant concentrations, and optional RUSLE sediment parameters.

## Configuration

```yaml
domain: ../domain/domain.gpkg
parcels: ../parcels/parcels.gpkg
# Optional HUC12-scale climate forcing:
plet_forcing: ../plet/plet_inputs.gpkg
outlets: ../outlets/outlets.gpkg

pollutants: [TN, TP, TSS]
cps: [340, 329, 590, 412, 656]

bmp_efficiency: ../bmps/bmp_efficiency.csv

load_generation:
  mode: plet_rusle
```

Parcel-specific PLET/RUSLE inputs and overrides live in dedicated `input_*` tables inside `parcels.gpkg`. HUC12-scale precipitation forcing may optionally be supplied through a separate `plet_forcing` GeoPackage. Curve-number and infiltration inputs remain in `parcels.gpkg`; there is no separate hydrology-lookup path.

## HUC12 climate forcing

When `plet_forcing` is configured, `parcels.gpkg` must contain a non-spatial `parcel_huc12` relationship table:

```text
pid | huc12 | area_fraction
```

`huc12` is stored as 12-character text. The preparation utility writes `area_fraction`, the fraction of parcel area inside its assigned (dominant) HUC12; when present, the loader checks that it lies between 0 and 1 and records values below 0.75 in the model's `log.txt` without printing the warning to the console. Every modeled parcel must have exactly one assignment. Forcing uses the dominant HUC12 only; it does not blend values across HUC12 boundaries.

The forcing GeoPackage separates spatial structure from model variables. The spatial `huc12` layer contains only WBD geometry/metadata:

```text
huc12 | name | states | areaacres | areasqkm | geometry
```

HUC12-scale PLET variables are stored in dedicated attribute tables in the same GeoPackage:

- `input_avg_rain_in`
- `input_annual_precip_in`
- `input_rain_days`
- `input_rain_correction_fraction`
- `input_runoff_day_fraction`

Each table is keyed by `huc12` and uses the same numeric uncertainty schema as parcel inputs:

```text
huc12 | value | mean | sd | min | p05 | p10 | p25 | p50 | p75 | p90 | p95 | max | sample_group | units | notes
```

A row may therefore be a fixed value or a distribution. By default, a distribution is sampled **once per HUC12 per scenario** and that draw is shared by all parcels assigned to the HUC12. An explicit `sample_group` can intentionally share a draw across multiple HUC12 rows for the same parameter.

If any HUC12 forcing rows are populated, every assigned HUC12 must have at least one `input_*` forcing row. Individual HUC12 variables may be omitted when parcel-specific rows or `pid=NULL` parcel defaults supply them; if all HUC12 tables are empty, climate inputs must come from parcel tables. Parcel `input_land_cover`, `input_hsg`, `input_curve_number`, and `input_infiltration_fraction` remain required in PLET/RUSLE mode; HUC12 forcing does not supply them.

For climate variables, precedence is:

```text
parcel-specific input_* override > HUC12 input_* row > pid=NULL parcel default
```

Thus parcel tables remain useful for defaults and local overrides without duplicating HUC12 values across every parcel. If `runoff_day_fraction` is absent from the HUC12 tables, it must resolve from a parcel-specific or `pid=NULL` parcel row because PLET annualizes event runoff using corrected rain days.

### Build or refresh the HUC12 forcing input

`utils/download_wbd_huc12.py` reads the parcel extent, downloads candidate 12-digit hydrologic-unit polygons from the USGS WBD service, filters them using a parcel-overlap threshold, writes/refreshes the spatial `huc12` layer, preserves the HUC12 `input_*` tables, and populates `parcels.gpkg:parcel_huc12` using greatest parcel-area overlap.

For the East Fork example:

```bash
python utils/download_wbd_huc12.py \
  examples/east_fork/inputs/parcels/parcels.gpkg \
  examples/east_fork/inputs/plet/plet_inputs_per_huc12.gpkg \
  --initial-plet-export examples/east_fork/inputs/misc/EastFork_plet_inputs.xlsx
```

`--initial-plet-export` reads the `1. Watershed Land Use` sheet and can initialize `AVG_RAIN`, `RAIN_DAYS`, and `ANNUAL_RAINFALL` as fixed rows in `input_avg_rain_in`, `input_rain_days`, and `input_annual_precip_in`. It cannot be used again once any HUC12 input table contains values; edit those tables directly in the GeoPackage. Re-running the utility refreshes WBD geometry/metadata and preserves input rows for HUC12s still retained by the refreshed layer; rows for HUC12s no longer retained are dropped, so back up edited inputs before refreshing. The default `--area-fraction-threshold 0.75` keeps a HUC12 only if at least one parcel has 75% or more of its area inside that HUC12. A refresh fails if the retained HUC12s do not cover every parcel; lower the threshold or inspect the geometries. The utility reports dominant assignments below the chosen threshold, while the model loader warns below its fixed 0.75 QA threshold.

`utils/create_parcel_huc12.py` remains available as a lower-level utility when HUC12 polygons are already available locally.

## Parcel PLET tables

Each variable gets its own table keyed by `pid`:

- `input_avg_rain_in` (PLET `AVG_RAIN`, inches/event; optional direct event-rainfall forcing)
- `input_annual_precip_in` (legacy/derived forcing)
- `input_rain_days`
- `input_rain_correction_fraction` (required only when `input_avg_rain_in` is not supplied)
- `input_runoff_day_fraction`
- `input_land_cover`
- `input_hsg`
- `input_ia_ratio`

Numeric tables use the standard fixed-value/distribution schema. `input_land_cover` and `input_hsg` require fixed categorical values. Where defaults are supported, `pid IS NULL` supplies the default and an exact integer `pid` row overrides it.

PLET precipitation forcing may be supplied in either of two forms:

1. **Direct PLET form:** `input_avg_rain_in` + `input_rain_days` + `input_runoff_day_fraction`. `avg_rain_in` is the PLET Input Data Server `AVG_RAIN` value and is used directly as event rainfall `P`.
2. **Derived form:** `input_annual_precip_in` + `input_rain_correction_fraction` + `input_rain_days` + `input_runoff_day_fraction`. The model derives `P = annual_precip_in * rain_correction_fraction / (rain_days * runoff_day_fraction)`.

If `input_avg_rain_in` is present for a parcel, it is authoritative. Annual runoff is still multiplied by the corrected number of runoff-producing rain days, `rain_days * runoff_day_fraction`. For subsurface infiltration, corrected annual precipitation is reconstructed as `avg_rain_in * rain_days * runoff_day_fraction`, which is algebraically equivalent to `annual_precip_in * rain_correction_fraction` when `AVG_RAIN` comes from PLET Equation 3.


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

Each table is keyed by `pid × pollutant` and uses the common numeric/distribution schema. A row with `pid IS NULL` supplies the pollutant default; an exact integer `pid` overrides it.

Surface concentrations are required for TN and TP and for TSS when complete RUSLE inputs are unavailable. Subsurface concentrations are required for modeled non-TSS pollutants.

## BMP pathways

PLET/RUSLE production pathways are fixed to `surface` and `subsurface`. Surface BMP efficiencies are required. Missing correctly labeled subsurface efficiencies are completed with zero and logged.
