from __future__ import annotations

from pathlib import Path

import yaml

import run_model


def test_example_config_runs_end_to_end(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_cfg = repo_root / "examples" / "east_fork" / "east_fork.yaml"
    cfg = yaml.safe_load(example_cfg.read_text(encoding="utf-8"))

    cfg["outputs"] = str(tmp_path / "outputs")
    cfg["n_scenarios"] = 1
    cfg["bmp_limit_n"] = 5
    cfg["parallel"] = {"n_jobs": 1}
    cfg["verbose"] = False

    smoke_cfg = tmp_path / "smoke.yaml"
    smoke_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(run_model, "make_summary_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr("sys.argv", ["run_model.py", str(smoke_cfg)])

    run_model.main()

    outputs = tmp_path / "outputs"
    assert (outputs / "log.txt").exists()
    assert (outputs / "logs" / "s1.txt").exists()
    assert (outputs / "bmps" / "s1.parquet").exists()
    assert (outputs / "parcels" / "s1.parquet").exists()
    assert (outputs / "scenario_metrics" / "s1.parquet").exists()
    assert (outputs / "outlet_trajectories" / "all_scenarios.parquet").exists()
