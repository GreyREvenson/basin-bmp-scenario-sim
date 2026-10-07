# Configuration reference

[← Back to main README](../readme.md)

## Configuration structure

A YAML configuration points to a small set of physical input packages and selects one of two load-generation modes:

- `statistical` (default)
- `plet_rusle`

Parcel-related inputs are consolidated in one GeoPackage and outlet-related inputs in a second GeoPackage. Within `parcels.gpkg`, every modeled user input variable has its own `input_*` table.

## Common configuration

```yaml
domain: ../domain/domain.gpkg
parcels: ../parcels/parcels.gpkg
outlets: ../outlets/outlets.gpkg

pollutants: [TN, TP, TSS]
cps: [340, 329, 590, 412, 656]

bmp_efficiency: ../bmps/bmp_efficiency.csv
bmp_cost: ../bmps/bmp_cost.csv                         # optional

outputs: ../../outputs/run_1
random_seed: 42
n_scenarios: 100
bmp_limit_n: 50
```

## Path resolution

Relative filesystem paths are resolved relative to the YAML file, not the process working directory. The GeoPackage files themselves can therefore be renamed or moved by changing only the corresponding YAML path.

Internal GeoPackage table names are schema names. Structural tables are `parcels`, `parcel_up`, and `parcel_outlets`; variable tables use the `input_*` prefix.

## Statistical mode

```yaml
load_generation:
  mode: statistical
```

`parcels.gpkg` must contain `input_pollutant_load_rate`.

## PLET/RUSLE mode

```yaml
load_generation:
  mode: plet_rusle
```

PLET/RUSLE parcel inputs—including curve number and infiltration fraction—are read from dedicated `input_*` tables in `parcels.gpkg`. HUC12-scale precipitation forcing may optionally be supplied with a top-level path:

```yaml
plet_forcing: ../plet/plet_inputs_per_huc12.gpkg
```

In that case `parcels.gpkg` also contains `parcel_huc12`, while the forcing GeoPackage contains a spatial `huc12` WBD layer plus separate HUC12-keyed `input_*` tables for `avg_rain_in`, `annual_precip_in`, `rain_days`, `rain_correction_fraction`, and `runoff_day_fraction`. It can also hold HUC12 `input_hsg` and HUC12 × land-cover RUSLE factors and pollutant concentrations. Numeric tables use the same fixed/distribution schema as parcel inputs, while HSG is a fixed classification. Parcel-specific values override HUC12 rows, which override `pid=NULL` parcel defaults. If any HUC12 parameter rows are supplied, every assigned HUC12 must have at least one; other variables may fall back to parcel defaults. Empty forcing tables leave climate variables to parcel inputs. There is no separate `hydrology_lookup` path. See [PLET/RUSLE mode](plet_rusle_mode.md) for preparation and coverage rules.

## Standard numeric input schema

Numeric variable tables share the same uncertainty columns:

```text
value, mean, sd, min,
p05, p10, p25, p50, p75, p90, p95, max,
sample_group, units, notes
```


## Parallel configuration

```yaml
parallel:
  n_jobs: -1
  max_nbytes: "1M"
  temp_folder: "/tmp/bmp-loky"
```

All input paths are resolved and all input tables are loaded before parallel scenario workers are launched.
