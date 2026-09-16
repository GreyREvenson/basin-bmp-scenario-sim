from __future__ import annotations

import logging

from model_test_cases import make_plet_rusle_case, make_statistical_case
from test_reproducibility_helpers import (
    assert_cross_output_consistency,
    assert_expected_scenarios_present,
    assert_outputs_equal,
    read_canonical_outputs,
)
from src.model import Model


def _run_case(builder, outputs_dir, *, n_jobs: int, random_seed: int) -> None:
    cfg, data = builder(
        outputs_dir,
        n_jobs=n_jobs,
        n_scenarios=2,
        random_seed=random_seed,
        bmp_limit_n=3,
    )
    logger = logging.getLogger(f"reproducibility.{outputs_dir.name}")
    Model(cfg, data, logger).run_all_scenarios()


def test_statistical_mode_repeat_run_reproducible_serial(tmp_path) -> None:
    outputs_a = tmp_path / "serial_a"
    outputs_b = tmp_path / "serial_b"

    _run_case(make_statistical_case, outputs_a, n_jobs=1, random_seed=202501)
    _run_case(make_statistical_case, outputs_b, n_jobs=1, random_seed=202501)

    a = read_canonical_outputs(outputs_a, expect_load_parameters=False)
    b = read_canonical_outputs(outputs_b, expect_load_parameters=False)

    assert_expected_scenarios_present(a, n_scenarios=2)
    assert_expected_scenarios_present(b, n_scenarios=2)
    assert_cross_output_consistency(a, expect_load_parameters=False)
    assert_cross_output_consistency(b, expect_load_parameters=False)
    assert_outputs_equal(a, b)


def test_statistical_mode_serial_equals_parallel(tmp_path) -> None:
    outputs_serial = tmp_path / "serial"
    outputs_parallel = tmp_path / "parallel"

    _run_case(make_statistical_case, outputs_serial, n_jobs=1, random_seed=202502)
    _run_case(make_statistical_case, outputs_parallel, n_jobs=2, random_seed=202502)

    serial = read_canonical_outputs(outputs_serial, expect_load_parameters=False)
    parallel = read_canonical_outputs(outputs_parallel, expect_load_parameters=False)

    assert_expected_scenarios_present(serial, n_scenarios=2)
    assert_expected_scenarios_present(parallel, n_scenarios=2)
    assert_cross_output_consistency(serial, expect_load_parameters=False)
    assert_cross_output_consistency(parallel, expect_load_parameters=False)
    assert_outputs_equal(serial, parallel)


def test_plet_mode_repeat_run_reproducible_serial(tmp_path) -> None:
    outputs_a = tmp_path / "plet_serial_a"
    outputs_b = tmp_path / "plet_serial_b"

    _run_case(make_plet_rusle_case, outputs_a, n_jobs=1, random_seed=202503)
    _run_case(make_plet_rusle_case, outputs_b, n_jobs=1, random_seed=202503)

    a = read_canonical_outputs(outputs_a, expect_load_parameters=True)
    b = read_canonical_outputs(outputs_b, expect_load_parameters=True)

    assert_expected_scenarios_present(a, n_scenarios=2)
    assert_expected_scenarios_present(b, n_scenarios=2)
    assert_cross_output_consistency(a, expect_load_parameters=True)
    assert_cross_output_consistency(b, expect_load_parameters=True)
    assert_outputs_equal(a, b)


def test_plet_mode_serial_equals_parallel(tmp_path) -> None:
    outputs_serial = tmp_path / "plet_serial"
    outputs_parallel = tmp_path / "plet_parallel"

    _run_case(make_plet_rusle_case, outputs_serial, n_jobs=1, random_seed=202504)
    _run_case(make_plet_rusle_case, outputs_parallel, n_jobs=2, random_seed=202504)

    serial = read_canonical_outputs(outputs_serial, expect_load_parameters=True)
    parallel = read_canonical_outputs(outputs_parallel, expect_load_parameters=True)

    assert_expected_scenarios_present(serial, n_scenarios=2)
    assert_expected_scenarios_present(parallel, n_scenarios=2)
    assert_cross_output_consistency(serial, expect_load_parameters=True)
    assert_cross_output_consistency(parallel, expect_load_parameters=True)
    assert_outputs_equal(serial, parallel)
