"""Run the model from the command line using a YAML settings file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.constants import (
    CFG_BMP_LIMIT_N,
    CFG_N_SCENARIOS,
    CFG_OUTPUTS,
    CFG_PARALLEL,
    CFG_RANDOM_SEED,
    CFG_VERBOSE,
)
from src.io_utils import read_config
from src.input_config import load_and_validate_all, normalize_config, resolve_config_paths
from src.input_validation import validate_config
from src.logging_utils import make_logger
from src.model import Model
from src.plotting import make_summary_plots


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
    parser.add_argument("config", help="Path to the YAML configuration file")
    parser.add_argument("--outputs", help="Override the outputs directory from config")
    parser.add_argument("--seed", type=int, help="Override random seed from config")
    parser.add_argument("--quiet", action="store_true", help="Disable console logging")
    parser.add_argument("--version", action="version", version="basin-bmp-sim 0.1.0")
    return parser.parse_args()


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
    scenario_records = sim.run_all_scenarios()

    if make_plots:
        make_summary_plots(cfg, data, scenario_records, outputs_dir, logger)

    logger.info("Model run complete")
    return outputs_dir


def main() -> None:
    """Parse CLI arguments and run the configured model."""
    args = parse_args()
    try:
        run_from_config(
            args.config,
            outputs=args.outputs,
            seed=args.seed,
            console=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
