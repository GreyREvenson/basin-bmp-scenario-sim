# Configuration reference

[← Back to main README](../readme.md)

## Configuration structure

A YAML configuration points to a small set of physical input packages and selects one of two load-generation modes:

- `statistical` (the default)
- `plet_rusle`

Parcel-related inputs are consolidated in one GeoPackage and outlet-related inputs are consolidated in a second GeoPackage. The internal table names are part of the model input schema; users may rename or move the GeoPackage files themselves and update only the YAML paths.

## Common required configuration

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

The two consolidated packages are:

- `parcels` — path to a GeoPackage containing the spatial `parcels` layer and parcel-related attribute tables.
- `outlets` — path to a GeoPackage containing the spatial `outlets` layer and optional `outlet_stats` table.

See [Input and output reference](input_output_reference.md) for the required internal tables and columns.

## Path resolution

Relative paths are resolved relative to the YAML file, not the process working directory. Absolute paths are preserved. Paths are resolved once in the parent process before inputs are loaded and before parallel scenario workers are launched.

For example, a YAML stored in `inputs/config/model.yaml` can use:

```yaml
parcels: ../parcels/parcels.gpkg
outlets: ../outlets/outlets.gpkg
```

The physical GeoPackage files may be renamed or moved; only the YAML entries need to change. Internal table names such as `parcels`, `parcel_outlets`, and `outlet_stats` are schema names and remain fixed.

## Statistical mode

```yaml
load_generation:
  mode: statistical
```

The `parcels` GeoPackage must contain a `pollutant_load_rates` attribute table. No separate pollutant-load-rate path is configured.

If an aggregate parcel × pollutant load rate is supplied while BMP efficiencies define multiple pathways, configure the split in YAML:

```yaml
pollutant_load_rate_pathway_fractions:
  surface: 0.70
  shallow subsurface: 0.20
  tile: 0.10
```

The fractions must correspond to active BMP-efficiency pathways and sum to 1.0.

## `plet_rusle` mode

```yaml
load_generation:
  mode: plet_rusle
  hydrology_lookup: ../plet/plet_hydrology_lookup.csv
```

The `parcels` GeoPackage must contain:

- `parcel_parameters` — combined PLET and RUSLE parameter rows.
- `pollutant_concentrations` — unified `surface` and `subsurface` concentration rows.

The PLET hydrology lookup remains a separate user-controlled CSV because users may change curve numbers, infiltration fractions, and their distributions independently of parcel geometry and parcel-specific parameters.

`plet_rusle` production pathways are fixed to `surface` and `subsurface`. `pathway_mode` is not supported.

## Standard numeric input schema

Numeric attribute tables and CSV inputs use the common fixed-value/distribution convention:

```text
value, distribution_id, mean, sd, min, p05, p50, p95, max
```

Only the fields needed for a row need values. See [Standardized numeric inputs and distributions](input_distributions.md).

## Scenario stopping conditions

A scenario may use `bmp_limit_n`, `bmp_limit_usd`, or both. If both are supplied, the scenario stops before another BMP is placed as soon as either limit has been met.

## Parallel configuration

```yaml
parallel:
  n_jobs: -1
  max_nbytes: "1M"
  temp_folder: "/tmp/bmp-loky"
```

- `n_jobs` controls worker processes; `-1` uses all available CPUs.
- `max_nbytes` controls the joblib memmap threshold.
- `temp_folder` optionally sets the worker temporary directory.

Input files are fully located and loaded before scenario parallelization begins.
