"""Run the model from the command line using a YAML settings file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from basin_bmp_sim.domain.constants import (
    CFG_BMP_LIMIT_N,
    CFG_N_SCENARIOS,
    CFG_OUTPUTS,
    CFG_PARALLEL,
    CFG_RANDOM_SEED,
    CFG_VERBOSE,
)
from basin_bmp_sim.common.io import read_config
from basin_bmp_sim.config.loading import load_and_validate_all
from basin_bmp_sim.config.settings import normalize_config, resolve_config_paths
from basin_bmp_sim.config.validation.configuration import validate_config
from basin_bmp_sim.common.logging import make_logger
from basin_bmp_sim.execution.monte_carlo_driver import Model
from basin_bmp_sim.execution.backend_selection import run_backend_selection
from basin_bmp_sim.results.plotting import make_summary_plots


def parse_args() -> argparse.Namespace:
    """Read command-line options for this script.

        Returns
        -------
        argparse.Namespace
            Parsed command-line arguments.
        
    """
    parser = argparse.ArgumentParser(
        description="Run the BMP scenario model using a YAML configuration file."
    )
    parser.add_argument("config", nargs="?", help="Path to the YAML configuration file (native mode)")
    parser.add_argument(
        "--backend-selection", type=Path,
        help="Opt-in Phase 11/12 JSON selection for a single or ensemble annual BMI backend",
    )
    parser.add_argument(
        "--coupled-result", type=Path,
        help="Destination JSON file for --backend-selection (not native Parquet output)",
    )
    parser.add_argument(
        "--analysis-config", type=Path,
        help="Optional Phase 14 targets/budget JSON; requires BMI ensemble selection",
    )
    parser.add_argument(
        "--analysis-result", type=Path,
        help="Destination decision-support JSON; requires --analysis-config",
    )
    parser.add_argument(
        "--input-trace-result", type=Path,
        help="Optional Phase 18 recorded BMP realization diagnostics; requires BMI ensemble",
    )
    parser.add_argument(
        "--input-ledger-result", type=Path,
        help="Optional detailed realized BMP values; requires --input-trace-result",
    )
    parser.add_argument(
        "--confirm-external-independent-draws", action="store_true",
        help="Affirm external BMI backend's independently sampled ensemble design for Phase 17",
    )
    parser.add_argument("--outputs", help="Override the outputs directory from config")
    parser.add_argument("--seed", type=int, help="Override random seed from config")
    parser.add_argument("--quiet", action="store_true", help="Disable console logging")
    parser.add_argument(
        "--scenario-replay-dir", type=Path,
        help="Replay recorded BMP realizations from a scenario_definitions directory; "
             "requires the same validated inputs, formulation, and seed",
    )
    parser.add_argument("--version", action="version", version="basin-bmp-sim 0.1.0")
    args = parser.parse_args()
    if args.config is None and args.backend_selection is None:
        parser.error("Supply a native YAML config or --backend-selection")
    return args


def run_from_config(
    config_path: str | Path,
    *,
    outputs: str | Path | None = None,
    seed: int | None = None,
    n_scenarios: int | None = None,
    n_jobs: int | None = None,
    bmp_limit_n: int | None = None,
    verbose: bool | None = None,
    console: bool = True,
    make_plots: bool = True,
    scenario_replay_dir: str | Path | None = None,
) -> Path:
    """Run the model from one configuration file.

    Relative filesystem paths in the YAML are resolved against the YAML file's
    own directory before any inputs are loaded or parallel workers are started.
    Programmatic overrides are applied after path resolution.
    """
    cfg_path = Path(config_path)
    cfg = normalize_config(read_config(cfg_path))
    resolve_config_paths(cfg, cfg_path)

    if outputs is not None:
        cfg[CFG_OUTPUTS] = str(Path(outputs).expanduser().resolve())
    if seed is not None:
        cfg[CFG_RANDOM_SEED] = int(seed)
    if n_scenarios is not None:
        cfg[CFG_N_SCENARIOS] = int(n_scenarios)
    if bmp_limit_n is not None:
        cfg[CFG_BMP_LIMIT_N] = int(bmp_limit_n)
    if n_jobs is not None:
        parallel = dict(cfg.get(CFG_PARALLEL) or {})
        parallel["n_jobs"] = int(n_jobs)
        cfg[CFG_PARALLEL] = parallel
    if verbose is not None:
        cfg[CFG_VERBOSE] = bool(verbose)

    validate_config(cfg)

    outputs_dir = Path(cfg[CFG_OUTPUTS])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    file_verbose = bool(cfg[CFG_VERBOSE])
    logger, log_path = make_logger(outputs_dir, verbose=file_verbose, console=console)
    logger.info("Starting model run")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Logging to: {log_path}")

    # All configured input paths are absolute and all input datasets are loaded
    # before Model.run_all_scenarios() launches any parallel workers.
    data = load_and_validate_all(cfg, logger)

    sim = Model(cfg, data, logger)
    if scenario_replay_dir is None:
        scenario_records = sim.run_all_scenarios()
    else:
        scenario_records = sim.run_all_scenarios(replay_dir=scenario_replay_dir)

    if make_plots:
        make_summary_plots(cfg, data, scenario_records, outputs_dir, logger)

    logger.info("Model run complete")
    return outputs_dir


def main() -> None:
    """Parse CLI arguments and run the configured model."""
    args = parse_args()
    try:
        if args.backend_selection is not None:
            if (args.config is not None or args.outputs is not None or
                    args.seed is not None or args.scenario_replay_dir is not None):
                raise ValueError(
                    "--backend-selection cannot be combined with native YAML, "
                    "--outputs, --seed, or --scenario-replay-dir; "
                    "use the selection and startup manifests instead"
                )
            if args.coupled_result is None:
                raise ValueError("--backend-selection requires --coupled-result")
            if (args.analysis_config is None) != (args.analysis_result is None):
                raise ValueError("--analysis-config and --analysis-result must be supplied together")
            if args.input_ledger_result is not None and args.input_trace_result is None:
                raise ValueError("--input-ledger-result requires --input-trace-result")
            if args.confirm_external_independent_draws and args.input_trace_result is None:
                raise ValueError("--confirm-external-independent-draws requires --input-trace-result")
            # Check output collisions before a potentially expensive ensemble
            # run, rather than discovering one only during post-processing.
            trace_paths = [p.expanduser().resolve() for p in
                           (args.input_trace_result, args.input_ledger_result) if p is not None]
            other_paths = [p.expanduser().resolve() for p in
                           (args.coupled_result, args.backend_selection,
                            args.analysis_config, args.analysis_result) if p is not None]
            if len(set(trace_paths)) != len(trace_paths) or any(p in other_paths for p in trace_paths):
                raise ValueError("BMI trace outputs must not overwrite any input or other output")
            if args.analysis_config is not None or args.input_trace_result is not None:
                from basin_bmp_sim.execution.backend_selection import BackendSelection
                selection = BackendSelection.read(args.backend_selection)
                if selection.ensemble is None:
                    raise ValueError("Analysis or input trace requires a selected BMI ensemble")
                if args.input_trace_result is not None and selection.backend == "external" and not args.confirm_external_independent_draws:
                    raise ValueError("External traces require --confirm-external-independent-draws")
                if args.analysis_config is not None:
                    from basin_bmp_sim.analysis.ensemble_analysis import DecisionSupportConfig
                    # Reject malformed targets before spending time on models.
                    DecisionSupportConfig.read(args.analysis_config)
            run_backend_selection(args.backend_selection, args.coupled_result)
            if args.input_trace_result is not None:
                from basin_bmp_sim.analysis.realization_trace import write_realization_trace
                write_realization_trace(args.coupled_result, args.input_trace_result,
                                        ledger_file=args.input_ledger_result,
                                        confirm_external_independence=args.confirm_external_independent_draws)
            if args.analysis_config is not None:
                from basin_bmp_sim.analysis.ensemble_analysis import write_ensemble_analysis
                write_ensemble_analysis(args.coupled_result, args.analysis_config, args.analysis_result)
        else:
            if args.coupled_result is not None:
                raise ValueError("--coupled-result requires --backend-selection")
            if args.analysis_config is not None or args.analysis_result is not None:
                raise ValueError("--analysis-config and --analysis-result require --backend-selection")
            if (args.input_trace_result is not None or args.input_ledger_result is not None or
                    args.confirm_external_independent_draws):
                raise ValueError("Input realization traces require a BMI ensemble backend selection")
            run_from_config(
                args.config,
                outputs=args.outputs,
                seed=args.seed,
                console=not args.quiet,
                scenario_replay_dir=args.scenario_replay_dir,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
