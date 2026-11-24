from __future__ import annotations

from basin_bmp_sim.common.logging import make_logger


def test_make_logger_uses_logs_subdirectory(tmp_path) -> None:
    logger, log_path = make_logger(tmp_path, verbose=False, scenario_id=1, console=False)

    assert log_path == tmp_path / "logs" / "s1.txt"
    logger.handlers[0].flush()
