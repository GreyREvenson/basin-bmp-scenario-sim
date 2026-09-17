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
input_distributions: ../misc/input_distributions.csv   # optional
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

PLET/RUSLE inputs—including curve number and infiltration fraction—are read from the dedicated `input_*` tables in `parcels.gpkg`. There is no separate `hydrology_lookup` path.

## Standard numeric input schema

Numeric variable tables share the same uncertainty columns:

```text
value, distribution_id, mean, sd, min,
p05, p10, p25, p50, p75, p90, p95, max,
sample_group, units, notes
```

Only the fields needed for a row need values. See [Standardized numeric inputs and distributions](input_distributions.md).

## Parallel configuration

```yaml
parallel:
  n_jobs: -1
  max_nbytes: "1M"
  temp_folder: "/tmp/bmp-loky"
```

All input paths are resolved and all input tables are loaded before parallel scenario workers are launched.
