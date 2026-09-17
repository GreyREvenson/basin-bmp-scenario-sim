# Statistical load-generation mode

[← Back to main README](../readme.md)

Statistical mode uses parcel load-rate distributions directly rather than deriving loads from PLET/RUSLE hydrology.

## Configuration

```yaml
domain: ../domain/domain.gpkg
parcels: ../parcels/parcels.gpkg
outlets: ../outlets/outlets.gpkg

pollutants: [TN, TP, TSS]
cps: [340, 329, 590]

bmp_efficiency: ../bmps/bmp_efficiency.csv
input_distributions: ../misc/input_distributions.csv

load_generation:
  mode: statistical
```

The load-rate data themselves are stored in the `pollutant_load_rates` table inside the configured parcel GeoPackage. There is no separate pollutant-load-rate path in YAML.

## `pollutant_load_rates` table

The natural key is parcel × pollutant, optionally extended by pathway.

Example aggregate rows:

```text
pid | pollutant | distribution_id | units
*   | TN        | yield_tn        | kg/ha/year
*   | TP        | yield_tp        | kg/ha/year
```

Example pathway-aware rows:

```text
pid | pollutant | pathway            | mean | sd | units
*   | TN        | surface            | 8.0  | 1.5| kg/ha/year
*   | TN        | shallow subsurface | 2.0  | 0.5| kg/ha/year
```

`pid="*"` provides defaults; exact parcel rows override matching wildcard rows.

## Aggregate inputs and pathway fractions

If load rates are aggregate but BMP efficiencies distinguish multiple pathways, define the split in YAML:

```yaml
pollutant_load_rate_pathway_fractions:
  surface: 0.70
  shallow subsurface: 0.20
  tile: 0.10
```

Fractions must match the active efficiency pathways and sum to 1.0.

## Parcel selection

Parcel-selection weighting is no longer a separate file. Put optional `selection_weight` values directly on the spatial `parcels` layer. If that column is absent, selection is uniform across modeled parcels.
