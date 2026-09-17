from __future__ import annotations

from pathlib import Path

import yaml

from src.constants import (
    CFG_BMP_EFFICIENCY,
    CFG_DOMAIN,
    CFG_LOAD_GENERATION,
    CFG_OUTPUTS,
    CFG_PARCELS,
    CFG_OUTLETS,
    LOAD_HYDROLOGY_LOOKUP,
)
from src.input_config import normalize_config, resolve_config_paths
from src.io_utils import read_config


def test_resolve_config_paths_uses_yaml_directory_not_cwd(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / "project" / "config"
    input_dir = tmp_path / "project" / "inputs"
    config_dir.mkdir(parents=True)
    input_dir.mkdir()

    config_path = config_dir / "model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "domain": "../inputs/domain.gpkg",
                "parcels": "../inputs/parcels.gpkg",
                "outlets": "../inputs/outlets.gpkg",
                "bmp_efficiency": "../inputs/bmp.csv",
                "outputs": "../outputs/run_1",
                "load_generation": {
                    "mode": "plet_rusle",
                    "hydrology_lookup": "../inputs/hydrology.csv",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    unrelated = tmp_path / "somewhere_else"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    cfg = normalize_config(read_config(config_path))
    resolve_config_paths(cfg, config_path)

    assert Path(cfg[CFG_DOMAIN]) == (input_dir / "domain.gpkg").resolve()
    assert Path(cfg[CFG_PARCELS]) == (input_dir / "parcels.gpkg").resolve()
    assert Path(cfg[CFG_OUTLETS]) == (input_dir / "outlets.gpkg").resolve()
    assert Path(cfg[CFG_BMP_EFFICIENCY]) == (input_dir / "bmp.csv").resolve()
    assert Path(cfg[CFG_OUTPUTS]) == (tmp_path / "project" / "outputs" / "run_1").resolve()
    assert Path(cfg[CFG_LOAD_GENERATION][LOAD_HYDROLOGY_LOOKUP]) == (
        input_dir / "hydrology.csv"
    ).resolve()


def test_moving_input_only_requires_yaml_path_change(tmp_path) -> None:
    config_dir = tmp_path / "config"
    original_dir = tmp_path / "inputs"
    moved_dir = tmp_path / "renamed_inputs"
    config_dir.mkdir()
    original_dir.mkdir()
    moved_dir.mkdir()

    original = original_dir / "old_name.csv"
    moved = moved_dir / "new_name.csv"
    original.write_text("x\n1\n", encoding="utf-8")
    moved.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")

    config_path = config_dir / "model.yaml"
    config_path.write_text(
        yaml.safe_dump({"bmp_efficiency": "../renamed_inputs/new_name.csv"}),
        encoding="utf-8",
    )

    cfg = normalize_config(read_config(config_path))
    resolve_config_paths(cfg, config_path)

    assert Path(cfg[CFG_BMP_EFFICIENCY]) == moved.resolve()
    assert Path(cfg[CFG_BMP_EFFICIENCY]).exists()


def test_resolve_config_paths_preserves_absolute_paths(tmp_path) -> None:
    absolute = (tmp_path / "already_absolute.csv").resolve()
    cfg = normalize_config({"bmp_efficiency": str(absolute)})

    resolve_config_paths(cfg, tmp_path / "config" / "model.yaml")

    assert Path(cfg[CFG_BMP_EFFICIENCY]) == absolute


def test_run_from_config_resolves_paths_before_model_is_constructed(tmp_path, monkeypatch) -> None:
    import run_model

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "domain": "../inputs/domain.gpkg",
                "outputs": "../outputs",
                "random_seed": 42,
                "verbose": False,
            }
        ),
        encoding="utf-8",
    )

    seen = {}

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

    class DummyModel:
        def __init__(self, cfg, data, logger):
            seen["model_cfg"] = dict(cfg)

        def run_all_scenarios(self):
            return []

    monkeypatch.setattr(run_model, "validate_config", lambda cfg: None)
    monkeypatch.setattr(
        run_model,
        "make_logger",
        lambda outputs_dir, *, verbose, console: (DummyLogger(), Path(outputs_dir) / "log.txt"),
    )
    monkeypatch.setattr(
        run_model,
        "load_and_validate_all",
        lambda cfg, logger: seen.setdefault("load_cfg", dict(cfg)) or {},
    )
    monkeypatch.setattr(run_model, "Model", DummyModel)

    run_model.run_from_config(config_path, make_plots=False)

    expected_domain = (tmp_path / "inputs" / "domain.gpkg").resolve()
    assert Path(seen["load_cfg"][CFG_DOMAIN]) == expected_domain
    assert Path(seen["model_cfg"][CFG_DOMAIN]) == expected_domain
    assert Path(seen["load_cfg"][CFG_DOMAIN]).is_absolute()
