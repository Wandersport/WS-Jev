"""Tests verifying exact reproducibility of seed-demo and cycle execution across runs."""

from __future__ import annotations

import argparse

from pm_research.calibration.metrics import CalibrationEngine
from pm_research.cli import cmd_seed_demo
from pm_research.config import SystemConfig
from pm_research.portfolio.tess import TessPortfolio
from pm_research.storage.db import Database


def test_seed_demo_exact_reproducibility(tmp_path):
    """Verify that running seed-demo repeatedly produces identical states, metrics, and calibration data."""
    db_file = str(tmp_path / "demo_repro.db")
    dashboard_file = str(tmp_path / "dashboard.html")
    fixed_base_time = "2026-01-01T12:00:00Z"

    args = argparse.Namespace(
        db=db_file,
        dashboard_out=dashboard_file,
        no_reset=False,
        base_time=fixed_base_time,
    )

    # 1. First run
    ret_1 = cmd_seed_demo(args)
    assert ret_1 == 0

    db = Database(db_file)
    cycles_1 = db.get_cycles()
    assert len(cycles_1) == 2, "Demo must produce exactly 2 cycles."

    config = SystemConfig()
    tess_1 = TessPortfolio(config, db)
    cash_1 = tess_1.virtual_cash
    equity_1 = tess_1.equity
    realized_1 = tess_1.realized_pnl
    positions_1 = len(tess_1.open_positions)

    calib_engine = CalibrationEngine()
    obs_1 = db.get_all_calibration_observations()
    metrics_1 = calib_engine.compute_metrics(obs_1)
    assert metrics_1.sample_size == 2, f"Expected exactly 2 calibration records, got {metrics_1.sample_size}"

    # 2. Second run on the same DB (should reset and produce identical result)
    ret_2 = cmd_seed_demo(args)
    assert ret_2 == 0

    cycles_2 = db.get_cycles()
    assert len(cycles_2) == 2, "Second run must also have exactly 2 cycles without accumulation"

    tess_2 = TessPortfolio(config, db)
    assert tess_2.virtual_cash == cash_1
    assert tess_2.equity == equity_1
    assert tess_2.realized_pnl == realized_1
    assert len(tess_2.open_positions) == positions_1

    obs_2 = db.get_all_calibration_observations()
    metrics_2 = calib_engine.compute_metrics(obs_2)
    assert metrics_2.sample_size == metrics_1.sample_size
    assert abs(metrics_2.brier_score - metrics_1.brier_score) < 1e-9
    assert abs(metrics_2.log_loss - metrics_1.log_loss) < 1e-9
    assert abs(metrics_2.expected_calibration_error - metrics_1.expected_calibration_error) < 1e-9
    assert abs(metrics_2.forecast_bias - metrics_1.forecast_bias) < 1e-9
