# BASIN-BMP-SCENARIO-SIMulator

`basin-bmp-scenario-sim` is a probabilistic, basin-scale best management practice (BMP) scenario simulator for evaluating how uncertainty in pollutant generation, BMP placement, BMP effectiveness, BMP cost, and BMP failure affects the likelihood of meeting pollutant-load reduction targets.

## Scientific purpose

The model is intended for watershed planning and uncertainty analysis. It uses Monte Carlo simulation to generate many plausible basin-wide BMP implementation scenarios, propagate uncertain inputs through each scenario, and evaluate the resulting parcel- and outlet-scale pollutant loads, costs, and target attainment.

The model is designed to answer questions such as:

- How variable are expected pollutant reductions across plausible BMP portfolios?
- How often does a given implementation strategy meet an outlet load-reduction target?
- How do uncertainty in baseline pollutant generation and BMP performance affect predicted outcomes?
- How do implementation limits such as BMP count or cost change the distribution of outcomes?

The simulator is **not** a continuous hydrologic or water-quality model. It represents annual pollutant generation and BMP effects at the parcel scale and routes those loads to configured outlets.

## Model workflow

At a high level, each Monte Carlo scenario:

1. establishes baseline parcel pollutant yields;
2. selects parcels and eligible BMPs according to configured probabilities;
3. samples uncertain input values such as yields, hydrologic parameters, BMP efficiencies, and costs;
4. applies pathway-specific BMP effects to the current parcel loads;
5. optionally simulates BMP failure;
6. routes remaining and removed loads to configured outlets; and
7. records scenario trajectories and summary metrics until the configured BMP-count or cost stopping condition is reached.

## Two load-generation modes

The model provides two alternative ways to establish baseline parcel pollutant yields.

| Feature | Default statistical mode | `plet_rusle` mode |
|---|---|---|
| Baseline pollutant generation | User supplies parcel yield values/distributions | Model calculates loads from PLET-style hydrology, concentrations, and optional RUSLE |
| `pollutant_yield` | Required | Not used |
| Pollutant pathways | User-defined | Fixed to `surface` and `subsurface` |
| Runoff/infiltration modeled | No | Yes |
| Curve Number | Not used | **Required user input by land-cover × HSG pairing**; fixed value or distribution |
| Infiltration fraction | Not used | **Required user input by land-cover × HSG pairing**; fixed value or distribution |
| Sediment | User-supplied yield | RUSLE when complete RUSLE inputs are supplied, otherwise concentration-based TSS may be used |
| BMP efficiency coverage | Required for every active pathway | Surface required; missing correctly labeled subsurface defaults to 0 with logging |

The **default statistical mode** is appropriate when parcel pollutant yields are already available from monitoring, another model, literature, calibration, or expert judgment. The **`plet_rusle` mode** is appropriate when baseline loads should be generated internally from PLET-style runoff/infiltration calculations and, optionally, RUSLE sediment generation.

Detailed descriptions are provided in [Statistical load-generation mode](docs/statistical_mode.md) and [PLET/RUSLE load-generation mode](docs/plet_rusle_mode.md).

## Standardized uncertain-input format

Numeric inputs use a common value/distribution convention. Depending on the table, an input may be specified as:

- a fixed `value`;
- `mean` + `sd` (optionally with `min`/`max` bounds);
- the legacy/common `min` + `mean` + `max` form;
- `min` + `max` for a uniform distribution;
- `min` + percentile columns such as `p05`, `p50`, `p95` + `max`; or
- a reusable `distribution_id` defined once in an optional `input_distributions.csv` catalog.

Parcel-indexed inputs may use `pid: "*"` as a default and provide only parcel-specific overrides. This can greatly reduce duplication when thousands of parcels share assumptions. If every parcel has a unique distribution, it can still be supplied directly in long form—one row per parcel/variable (or parcel/pollutant/pathway, as appropriate).

See [Standardized numeric inputs and distributions](docs/input_distributions.md) for the complete schema and sampling rules.

## Quick start

A model run requires common spatial/routing inputs, pollutant and BMP definitions, a BMP-efficiency table, a load-generation configuration, and at least one scenario stopping condition.

Example command:

```bash
python run_model.py config.yaml
```

Statistical mode is the default:

```yaml
pollutants: [TN, TP, TSS]
cps: [340, 329, 590]

# Optional reusable catalog for numeric assumptions.
input_distributions: ./inputs/input_distributions.csv

pollutant_yield: ./inputs/pollutant_yield.csv
bmp_efficiency: ./inputs/bmp_efficiency.csv

n_scenarios: 1000
bmp_limit_n: 200
```

PLET/RUSLE mode is enabled explicitly and requires a hydrology table:

```yaml
input_distributions: ./inputs/plet/input_distributions.csv  # optional

load_generation:
  mode: plet_rusle
  plet_inputs: ./inputs/plet/plet_inputs.csv
  hydrology_lookup: ./inputs/plet/plet_hydrology_lookup.csv  # required
  rusle_inputs: ./inputs/plet/rusle_inputs.csv
  pollutant_concentrations: ./inputs/plet/pollutant_concentrations.csv
  groundwater_concentrations: ./inputs/plet/groundwater_concentrations.csv
```

See [Configuration reference](docs/configuration.md) for complete examples and mode-specific requirements.

## Documentation

- [Model concepts and scientific formulation](docs/model_overview.md) — yields, loads, pathways, Monte Carlo structure, and outlet routing.
- [Configuration reference](docs/configuration.md) — common YAML settings and complete examples for both modes.
- [Standardized numeric inputs and distributions](docs/input_distributions.md) — fixed values, distributions, reusable distribution IDs, parcel defaults/overrides, and shared draws.
- [Statistical load-generation mode](docs/statistical_mode.md) — direct parcel yields, arbitrary pathways, aggregate-yield splitting, and coverage rules.
- [PLET/RUSLE load-generation mode](docs/plet_rusle_mode.md) — required land-cover/HSG hydrology inputs, runoff, infiltration, groundwater loads, RUSLE, and two-pathway BMP treatment.
- [BMP simulation](docs/bmp_simulation.md) — BMP selection, efficiencies, treatment fractions, failure, signed effects, and serial stacking.
- [Input and output reference](docs/input_output_reference.md) — input file roles, output files, and interpretation.
- [Reproducibility, testing, and limitations](docs/reproducibility_validation.md) — seeds, parallel execution, validation expectations, assumptions, and appropriate interpretation.

## Required inputs at a glance

Inputs shared by both modes generally include:

- watershed/domain geometry;
- parcel polygons;
- outlet locations;
- parcel-to-outlet mapping;
- modeled pollutants;
- configured BMP/conservation-practice CPS codes;
- BMP-efficiency statistics;
- number of Monte Carlo scenarios; and
- a BMP-count limit, cost limit, or both.

Load-generation-specific inputs differ substantially between the two modes. See [Configuration reference](docs/configuration.md).

## Outputs

The model writes scenario-level and aggregated results to the configured `outputs` directory. Current canonical outputs include per-BMP records, per-parcel results, scenario metrics, outlet trajectories, logs, plots, and—when `plet_rusle` is used—load-generation diagnostic records.

See [Input and output reference](docs/input_output_reference.md) for details.

## Reproducibility

Set `random_seed` in the YAML configuration or use the CLI `--seed` option. The base seed is used to spawn scenario-specific child seeds so a run can be reproduced when the same code, inputs, configuration, and seed are used.

For reproducible scientific analyses, archive or record:

- the repository commit or release;
- the configuration YAML;
- all input data files, including any distribution catalog and PLET hydrology lookup;
- the random seed;
- the Python environment/dependency versions; and
- the generated logs and canonical outputs.

See [Reproducibility, testing, and limitations](docs/reproducibility_validation.md).

## Scientific scope and limitations

The model is a scenario and uncertainty framework, not a substitute for a calibrated process-based watershed model where detailed temporal hydrology, water-quality transformation, or in-stream processes are required. Results depend on the validity of the supplied probability distributions, pathway definitions, BMP efficiencies, routing assumptions, and—in `plet_rusle` mode—the PLET/RUSLE parameterization, including the user-supplied Curve Number and infiltration-fraction assumptions.

BMP efficiencies are applied serially to the current remaining load. Parcel-to-outlet routing may use optional delivery ratios, but the simulator does not independently resolve all physical fate and transport processes between a parcel and an outlet.

Model outputs should therefore be interpreted as conditional on the selected model structure and input assumptions.

## Citation

When using the model in scientific or technical work, identify the repository and the exact version, release, or Git commit used for the analysis. If a formal publication or DOI is established for the model, that citation should be added here and used in preference to the repository-only citation.

## License

This project is distributed under the [MIT License](license.md).

## Contact

evenson.grey@epa.gov
