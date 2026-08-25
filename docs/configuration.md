# Configuration reference

[← Back to main README](../readme.md)

## Configuration structure

A YAML configuration combines common scenario settings with one of two load-generation configurations:

- default/statistical mode; or
- `plet_rusle` mode.

If `load_generation.mode` is omitted, the model uses statistical mode.

## Common required configuration

Typical common requirements are:

- `domain` — watershed boundary file;
- `parcels` — parcel polygon file;
- `outlet_loc` — outlet location file;
- `parcel_out` — CSV mapping parcels to outlet IDs;
- `pollutants` — modeled pollutant labels;
- `cps` — BMP/conservation-practice CPS codes;
- `bmp_efficiency` — BMP efficiency statistics;
- `n_scenarios` — number of Monte Carlo scenarios; and
- at least one of `bmp_limit_n` or `bmp_limit_usd`.

Common optional settings include:

- `parcel_up` — upstream parcel relationships;
- `parcel_p` — parcel-selection probability weights;
- `bmp_cost` — BMP cost statistics;
- `delivery_ratios` — parcel-to-outlet delivery ratios;
- `outlet_target` — outlet pollutant reduction targets;
- `outlet_mean` — outlet mean-load reference values;
- `buffer_depth_ft` — buffer-depth assumption used by applicable grassed BMP calculations;
- `bmp_sel_prob_via_costs` — allow cost to influence BMP-selection probabilities;
- `bmp_fail_rate` — probability that a BMP placement fails;
- `bmp_fail_reduction` — efficiency multiplier applied after failure;
- `random_seed` — base random seed;
- `outputs` — output directory;
- `verbose` — verbose logging; and
- `parallel` — parallel execution settings.

Pollutant aliases such as `nitrogen`, `phosphorus`, and `sediment` are normalized to canonical pollutant labels where supported.

## Statistical mode configuration

Statistical mode can be explicit:

```yaml
load_generation:
  mode: statistical
```

or `load_generation` can be omitted.

A basic configuration is:

```yaml
verbose: true
outputs: ./outputs
random_seed: 42

domain: ./inputs/domain.gpkg
parcels: ./inputs/parcels.gpkg
outlet_loc: ./inputs/outlet_loc.gpkg
parcel_out: ./inputs/parcel_out.csv
parcel_up: ./inputs/parcel_up.csv
parcel_p: ./inputs/parcel_p.csv

pollutants: [TN, TP, TSS]
cps: [340, 329, 590, 412, 656]

pollutant_yield: ./inputs/pollutant_yield.csv
bmp_efficiency: ./inputs/bmp_efficiency.csv
bmp_cost: ./inputs/bmp_cost.csv

outlet_target: ./inputs/outlet_target.csv
outlet_mean: ./inputs/outlet_mean.csv

n_scenarios: 1000
bmp_limit_n: 200

bmp_fail_rate: 0.25
bmp_fail_reduction: 0.25
```

### Statistical mode with aggregate yields

If `pollutant_yield.csv` contains one aggregate yield per parcel × pollutant while `bmp_efficiency.csv` defines multiple pathways, define the split explicitly:

```yaml
pollutant_yield_pathway_fractions:
  surface: 0.70
  shallow subsurface: 0.20
  tile: 0.10
```

The fractions must correspond to active BMP-efficiency pathways and sum to 1.0.

Legacy surface/shallow fraction keys are retained for compatible configurations, but the general mapping above is recommended for new applications.

See [Statistical load-generation mode](statistical_mode.md).

## `plet_rusle` configuration

A typical configuration is:

```yaml
verbose: true
outputs: ./outputs
random_seed: 42

domain: ./inputs/domain.gpkg
parcels: ./inputs/parcels.gpkg
outlet_loc: ./inputs/outlet_loc.gpkg
parcel_out: ./inputs/parcel_out.csv
parcel_up: ./inputs/parcel_up.csv
parcel_p: ./inputs/parcel_p.csv

pollutants: [TN, TP, TSS]
cps: [340, 329, 590, 412, 656]

bmp_efficiency: ./inputs/bmp_efficiency_plet.csv
bmp_cost: ./inputs/bmp_cost.csv

load_generation:
  mode: plet_rusle
  plet_inputs: ./inputs/plet/plet_inputs.csv
  rusle_inputs: ./inputs/plet/rusle_inputs.csv
  pollutant_concentrations: ./inputs/plet/pollutant_concentrations.csv
  groundwater_concentrations: ./inputs/plet/groundwater_concentrations.csv

outlet_target: ./inputs/outlet_target.csv
outlet_mean: ./inputs/outlet_mean.csv

n_scenarios: 1000
bmp_limit_n: 200

bmp_fail_rate: 0.25
bmp_fail_reduction: 0.25
```

### Mode-specific rules

In `plet_rusle` mode:

- `pollutant_yield` is not used;
- production pathways are fixed to `surface` and `subsurface`;
- `plet_inputs` is required;
- `pollutant_concentrations` is required when TN or TP is modeled;
- `groundwater_concentrations` is required for every modeled non-TSS pollutant;
- `rusle_inputs` is optional, but a parcel that supplies RUSLE inputs must supply a complete RUSLE factor set and either `sdr` or `watershed_area_mi2`;
- `pathway_mode` has been removed and is an error if supplied; and
- statistical pathway-fraction settings do not control PLET/RUSLE pathways.

Legacy `groundwater_loads` and `treat_groundwater_with_bmps` settings do not control production PLET/RUSLE pathway generation. The production calculation always estimates the lookup-derived subsurface load, and BMP treatment is controlled by the `subsurface` BMP efficiency.

See [PLET/RUSLE load-generation mode](plet_rusle_mode.md).

## Scenario stopping conditions

The model supports a BMP-count limit, a cost limit, or both. When both are configured, the scenario terminates when a stopping condition is reached according to the scenario engine's current implementation.

## Parallel configuration

```yaml
parallel:
  n_jobs: -1
  max_nbytes: "1M"
  temp_folder: "/tmp/bmp-loky"
```

- `n_jobs` controls worker processes; `-1` uses all available CPUs.
- `max_nbytes` controls the memmap threshold for objects passed to workers.
- `temp_folder` optionally sets the `loky` temporary directory.
