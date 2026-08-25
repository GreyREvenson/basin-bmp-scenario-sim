# Input and output reference

[← Back to main README](../readme.md)

## Common input files

### `domain`

Watershed/domain geometry used to define the modeled spatial extent.

### `parcels`

Parcel or field polygons. Parcel IDs must be unique after any domain clipping performed by the model.

### `outlet_loc`

Modeled outlet locations.

### `parcel_out`

Maps parcel IDs to one or more outlet IDs. Referenced outlet IDs must exist in `outlet_loc`.

### `parcel_up`

Optional upstream parcel connectivity. This is used by BMP calculations that require contributing-area relationships.

### `parcel_p`

Optional parcel-selection probability information. Probabilities must be valid for the parcels used by the scenario engine.

### `bmp_efficiency`

BMP efficiency distributions by CPS, pollutant, and—when pathway-aware—pathway.

Statistical mode requires complete coverage for every active pathway. `plet_rusle` requires surface coverage and defaults missing correctly labeled subsurface efficiency to zero with logging.

### `bmp_cost`

Optional BMP cost distributions used for cost accounting and, when configured, BMP-selection weighting.

### `delivery_ratios`

Optional parcel-to-outlet delivery ratios used to attenuate loads before outlet evaluation.

### `outlet_target`

Optional outlet pollutant reduction targets.

### `outlet_mean`

Optional outlet mean-load reference values.

## Statistical-mode load inputs

### `pollutant_yield`

Required in statistical mode. It may contain:

- explicit parcel × pollutant × pathway distributions; or
- one aggregate parcel × pollutant distribution that is subsequently split with configured pathway fractions.

See [Statistical load-generation mode](statistical_mode.md).

## `plet_rusle` load inputs

### `load_generation.plet_inputs`

Required long-form PLET parameter table. It supplies climate and classification inputs used to resolve Curve Number and infiltration fraction and calculate annual runoff/infiltration.

### `load_generation.rusle_inputs`

Optional long-form RUSLE parameter table. A parcel with RUSLE data must have a complete factor set and an SDR or watershed area value required by the current delivery formulation.

### `load_generation.pollutant_concentrations`

Runoff concentrations. Required when TN or TP is modeled. TSS concentration is also needed when RUSLE is not available for a parcel and TSS is modeled.

### `load_generation.groundwater_concentrations`

Required for each modeled non-TSS pollutant. Used with PLET infiltration volume to calculate the `subsurface` pathway.

## Statistical input formats

Input tables may provide a fixed `value` or statistical information supported by the model, such as combinations of:

- `mean` and `sd`;
- `min` and `max`; or
- percentile fields such as `p05`, `p50`, and `p95`.

The applicable loader validates required key columns and supported statistical information for each table.

## Output directory

Current canonical outputs are written below the configured `outputs` directory.

### Per-BMP records

```text
bmps/s{scenario}.parquet
```

Contains individual BMP placement records and associated pollutant-treatment/removal information.

### Per-parcel records

```text
parcels/s{scenario}.parquet
```

Contains parcel baseline/final pollutant information for each scenario.

### PLET/RUSLE load diagnostics

```text
load_parameters/s{scenario}.parquet
```

Written when PLET/RUSLE load-generation diagnostics are available. The production PLET pathways are `surface` and `subsurface`; compatibility diagnostic fields may also be retained for older callers/tests and should not be confused with additional production pathways.

### Scenario metrics

```text
scenario_metrics/s{scenario}.parquet
```

Canonical per-scenario metrics.

### Outlet trajectories

```text
outlet_trajectories/all_scenarios.parquet
```

Aggregated outlet trajectory data used for downstream plotting and scenario comparison.

### Logs

```text
log.txt
logs/s{scenario}.txt
```

The driver writes the overall log, while scenario workers can write scenario-specific logs. PLET/RUSLE logs also report cases where subsurface BMP efficiency defaults to zero or unexpected pathway labels are ignored.

### Plots

Summary `plot_*` outputs visualize scenario trajectories such as implementation cost or BMP count versus outlet load/target metrics.

## Interpreting outputs

Each scenario is one possible implementation realization. Scientific interpretation should focus on the **distribution** of outcomes across scenarios rather than treating a single scenario as a prediction.

Useful summaries include:

- median and percentile pollutant reductions;
- probability of meeting a target;
- cost distributions;
- BMP portfolio composition;
- sensitivity to baseline-load assumptions; and
- differences between configuration alternatives.
