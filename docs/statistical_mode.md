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

## `input_pollutant_load_rate`

The statistical load-rate variable is stored in the `input_pollutant_load_rate` table inside `parcels.gpkg`.

The natural key is parcel × pollutant, optionally extended by pathway.

Example aggregate rows:

```text
pid | pollutant | distribution_id | units
*   | TN        | yield_tn        | kg/ha/year
*   | TP        | yield_tp        | kg/ha/year
```

Example pathway-aware rows:

```text
pid | pollutant | pathway            | mean | sd  | units
*   | TN        | surface            | 8.0  | 1.5 | kg/ha/year
*   | TN        | shallow subsurface | 2.0  | 0.5 | kg/ha/year
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

Fractions must match active efficiency pathways and sum to `1.0`.

## Parcel selection

Optional parcel-selection weights are stored in `input_selection_weight`, not on the spatial `parcels` layer. The table is keyed by `pid` and uses deterministic `value` entries. If the table is absent, parcel selection is uniform.
