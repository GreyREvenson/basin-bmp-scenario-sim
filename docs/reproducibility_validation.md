# Reproducibility, testing, and limitations

[← Back to main README](../readme.md)

## Reproducibility

Set a base seed in YAML:

```yaml
random_seed: 12345
```

or override it on the command line:

```bash
python run_model.py config.yaml --seed 12345
```

The model uses the base seed to create scenario-specific child seeds. Repeating a run with the same code, input files, configuration, seed, and compatible software environment should reproduce the stochastic sequence used by the scenario simulation.

For scientific analyses, archive:

- repository commit/release identifier;
- configuration YAML;
- all input tables and geospatial files;
- random seed;
- Python and dependency versions;
- logs; and
- canonical output parquet files.

## Parallel execution

The model uses `joblib` for scenario-level parallelism.

```yaml
parallel:
  n_jobs: -1
  max_nbytes: "1M"
  temp_folder: "/tmp/bmp-loky"
```

Parallel execution should not be treated as a substitute for recording the seed and software environment. Reproducibility claims should be verified for the exact runtime environment used in an analysis.

## Testing

Scientific model development should maintain tests covering at least:

- input schema and coverage validation;
- parcel/outlet and upstream parsing;
- statistical pathway behavior;
- PLET land-cover/HSG lookup behavior;
- PLET runoff and infiltration calculations;
- RUSLE sediment calculations;
- surface/subsurface mass balance in `plet_rusle`;
- missing/mislabeled BMP efficiency handling;
- signed BMP efficiencies;
- BMP failure behavior; and
- output/reporting compatibility.

Code-level unit tests demonstrate implementation consistency; they do not by themselves establish scientific validity for a particular watershed.

## Scientific validation

Validation should be matched to the selected load-generation mode.

### Statistical mode

Evaluate whether supplied yield and BMP-efficiency distributions are supported by the best available monitoring data, literature, watershed model results, or expert elicitation. Where possible, compare implied baseline loads with observed or independently modeled loads.

### `plet_rusle` mode

Evaluate PLET/RUSLE inputs and outputs against appropriate benchmarks, including:

- PLET reference calculations;
- observed or independently modeled runoff;
- observed/estimated sediment export;
- nutrient concentrations and loads; and
- parcel- or watershed-scale mass balance.

The bundled land-cover/HSG lookup constrains hydrologic parameters consistently, but that does not guarantee that the resulting annual runoff or subsurface load is accurate for every local setting.

## Structural assumptions and limitations

Important assumptions include:

- annualized rather than continuous load simulation;
- serial BMP stacking;
- pathway-specific efficiencies applied multiplicatively to current remaining load;
- optional delivery-ratio routing rather than explicit in-stream fate/transport;
- user-defined statistical pathways have no automatically inferred hydrologic meaning;
- PLET/RUSLE production pathways are limited to `surface` and `subsurface`;
- PLET subsurface load is represented as groundwater concentration multiplied by lookup-derived infiltration volume; and
- BMP efficiencies represent net effects supplied by the user rather than mechanistically simulated BMP processes.

## Parameter uncertainty versus structural uncertainty

Monte Carlo sampling quantifies uncertainty represented in the supplied distributions. It does **not** automatically quantify structural uncertainty associated with omitted processes, choice of equations, spatial aggregation, or transferability of BMP-effectiveness studies.

Users should consider scenario comparisons or alternative model structures where structural uncertainty could materially affect conclusions.

## Appropriate interpretation

Model results are conditional statements of the form:

> Given these baseline-load assumptions, BMP-effect distributions, placement rules, costs, failure assumptions, pathway definitions, and routing assumptions, this is the simulated distribution of basin outcomes.

That is different from claiming that the output distribution captures every source of real-world uncertainty.

## Recommended reporting

A scientific or technical report using the model should document:

- model version/commit;
- selected load-generation mode;
- pathway definitions;
- all major input data sources;
- probability distributions and their rationale;
- BMP interaction/failure assumptions;
- number of Monte Carlo scenarios;
- random seed;
- outlet routing/delivery assumptions;
- calibration or validation comparisons; and
- key limitations or unsupported processes.
