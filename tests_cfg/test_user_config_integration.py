"""Optional end-to-end validation for any user-supplied model YAML.

Invoke explicitly, for example::

    python -m pytest integration_tests/test_user_config_integration.py \\
        --model-config path/to/model.yaml

This file is intentionally outside ``tests/`` so ordinary test execution does
not depend on any real example dataset or YAML file.
"""

from __future__ import annotations

from pathlib import Path

import run_model


def test_user_supplied_yaml_runs_end_to_end(model_config_path: Path, tmp_path: Path) -> None:
    outputs = tmp_path / "integration_outputs"

    result = run_model.run_from_config(
        model_config_path,
        outputs=outputs,
        n_jobs=1,
        n_scenarios=1,
        bmp_limit_n=1,
        verbose=False,
        console=False,
        make_plots=False,
    )

    assert result == outputs.resolve()
    assert (outputs / "bmps" / "s1.parquet").is_file()
    assert (outputs / "parcels" / "s1.parquet").is_file()
    assert (outputs / "scenario_metrics" / "s1.parquet").is_file()
    assert (outputs / "outlet_trajectories" / "all_scenarios.parquet").is_file()
