# Model concepts and scientific formulation

[← Back to main README](../readme.md)

## Purpose

`basin-bmp-scenario-sim` is a Monte Carlo framework for connecting uncertainty in parcel-scale pollutant generation and BMP implementation to basin-outlet pollutant outcomes. It operates on annualized pollutant yields and loads rather than simulating a continuous hydrograph or water-quality time series.

## Core quantities

The model distinguishes several related quantities.

### Concentration

Concentration is pollutant mass per unit water volume, for example `mg/L`. Concentrations are used explicitly by `plet_rusle` to calculate runoff-derived and infiltration-derived nutrient loads.

### Yield

Yield is pollutant mass normalized by land area and time, for example:

```text
kg/ha/year
```

Parcel pollutant generation is represented internally as pathway-specific annual yields.

### Load

Load is total pollutant mass over a period. For parcel area `A` and annual yield `Y`:

```text
parcel load = Y × A
```

### Pathway

A pathway partitions a parcel's pollutant yield into components that may respond differently to BMPs.

In **statistical mode**, pathway labels are user-defined bookkeeping categories and may represent concepts such as surface runoff, shallow subsurface flow, tile drainage, groundwater, or another user-defined transport category.

In **`plet_rusle` mode**, pathway meanings are fixed:

- `surface` — runoff-derived load plus sediment-associated pollutant load where applicable;
- `subsurface` — infiltration/groundwater-derived non-TSS load.

## Monte Carlo representation

Uncertain quantities may be represented using fixed values or statistical information supported by the input table. Across scenarios, the model samples uncertain values and thereby produces a distribution of implementation outcomes rather than one deterministic answer.

Uncertainty may include:

- parcel selection;
- BMP type;
- parcel pollutant yields or physical load-generation inputs;
- BMP pollutant-removal efficiencies;
- BMP-specific treatment characteristics;
- cost; and
- BMP failure.

The resulting distribution of outlet loads or target attainment can be interpreted as conditional on those user-defined uncertainty distributions and the model's structural assumptions.

## Scenario sequence

For each scenario, the model first establishes baseline pathway yields for every selected parcel and pollutant. It then repeatedly selects BMP placements and updates the current pathway loads. Later BMPs operate on loads remaining after earlier BMPs.

Conceptually, for pathway yield `Y`, treated fraction `f`, and sampled BMP efficiency `e`:

```text
Y_new = Y_old × (1 - f × e)
```

If `e` is negative, the same equation represents an increase in load rather than removal.

## Parcel-to-outlet evaluation

`parcel_out` associates parcels with modeled outlets. Optional delivery ratios can attenuate parcel loads before outlet evaluation. Outlet target and mean-load inputs allow scenario trajectories to be expressed relative to decision-relevant reference values.

The load-generation mode determines how parcel baseline loads are established. It does not change the overall parcel-selection, BMP-selection, routing, and outlet-evaluation framework.

## What the model does not represent directly

Unless supplied indirectly through input distributions or delivery factors, the model does not explicitly simulate:

- sub-daily or daily hydrology;
- channel hydraulics;
- stream temperature;
- in-stream nutrient transformation;
- groundwater travel-time distributions;
- sediment storage and remobilization within the channel network; or
- mechanistic BMP performance through time.

These processes may matter in particular applications and should be addressed with complementary models, empirical data, or uncertainty assumptions where appropriate.
