# Statistical load-generation mode

[← Back to main README](../readme.md)

## Purpose

Statistical mode is the default load-generation approach. It is designed for applications in which annual parcel pollutant yields are available directly from monitoring, another model, literature, calibration, or expert judgment.

Statistical mode does **not** calculate runoff, infiltration, erosion, or groundwater flow. Those processes may be represented implicitly in the yield distributions supplied by the user.

## Selecting the mode

The mode can be omitted because statistical mode is the default, or specified explicitly:

```yaml
load_generation:
  mode: statistical
```

`pollutant_yield` is required.

## Option 1: explicit pathway-specific yields

If `pollutant_yield.csv` contains a `pathway` column, supply a separate distribution for every parcel × pollutant × pathway combination.

Example:

```csv
pid,pollutant,pathway,mean,min,max
1,TN,surface,8.0,5.0,11.0
1,TN,shallow subsurface,3.0,1.0,5.0
1,TN,tile,5.0,2.0,8.0
1,TP,surface,1.2,0.7,1.8
1,TP,shallow subsurface,0.1,0.02,0.2
1,TP,tile,0.2,0.05,0.4
```

Pathway names are user-defined. Examples include:

- `surface`;
- `shallow subsurface`;
- `deep subsurface`;
- `tile`;
- `groundwater`; or
- another non-empty pathway label.

The model samples each pathway yield from its supplied statistics and sums pathway yields to obtain total parcel yield.

### Coverage requirements

When pathway-specific yields are used:

- every modeled parcel must have complete pollutant × pathway yield coverage;
- `bmp_efficiency.csv` must use the same pathway set; and
- every configured CPS × pollutant × pathway combination must have a BMP efficiency.

Missing yield or efficiency coverage is an error rather than an implicit zero.

## Option 2: one aggregate yield split among pathways

A user may instead provide one distribution per parcel × pollutant with no `pathway` column:

```csv
pid,pollutant,mean,min,max
1,TN,16.0,10.0,22.0
1,TP,1.5,0.9,2.2
1,TSS,850.0,500.0,1200.0
```

The aggregate yield is sampled once. If only one BMP-efficiency pathway is active, the entire sampled yield is assigned to it.

If multiple pathways are active, define pathway fractions:

```yaml
pollutant_yield_pathway_fractions:
  surface: 0.60
  shallow subsurface: 0.20
  tile: 0.20
```

For sampled aggregate yield `Y` and pathway fraction `f_k`:

```text
Y_k = Y × f_k
```

Fractions must:

- refer only to active pathways;
- each be between 0 and 1; and
- sum to 1.0.

This design preserves the interpretation of a **single uncertain aggregate yield** that is partitioned deterministically after sampling, rather than sampling independent scaled versions of the same total distribution.

## BMP efficiencies

A typical pathway-aware efficiency table is:

```csv
cps,pollutant,pathway,mean,min,max
340,TN,surface,0.35,0.20,0.50
340,TN,shallow subsurface,0.15,0.05,0.25
340,TN,tile,0.10,0.00,0.20
```

Every active pathway requires complete efficiency coverage for every configured CPS and pollutant.

For pathway yield `Y`, treated fraction `f_t`, and sampled efficiency `e`:

```text
Y_new = Y_old × (1 - f_t × e)
```

If an efficiency is negative, the model preserves the signed effect; the pathway load can increase.

## When to use statistical mode

Statistical mode is useful when:

- another model already provides parcel-scale yields;
- monitoring or literature distributions are preferred to simplified PLET/RUSLE load generation;
- user-defined pathways such as tile drainage are important;
- a watershed-specific empirical model is being coupled to the scenario simulator; or
- baseline uncertainty is most naturally expressed directly as yield distributions.

## Main distinction from `plet_rusle`

Statistical pathways are user-defined categories and carry no automatically imposed hydrologic meaning. In `plet_rusle`, the model itself derives the fixed `surface` and `subsurface` components from hydrologic calculations.
