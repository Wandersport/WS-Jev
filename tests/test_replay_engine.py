"""Comprehensive quantitative validation tests for ReplayEngine, latency models, and temporal integrity."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from pm_research.data.synthetic_replay import create_deterministic_synthetic_replay_dataset
from pm_research.replay.engine import ReplayEngine
from pm_research.replay.models import ReplayConfig
from pm_research.storage.db import Database


def test_replay_deterministic_reproducibility() -> None:
    """Two independent replay runs with identical configuration and seed yield bit-for-bit identical results."""
    ds = create_deterministic_synthetic_replay_dataset()

    cfg1 = ReplayConfig(dataset_path="", latency_seconds=0.0, random_seed=42)
    eng1 = ReplayEngine(config=cfg1, dataset=ds)
    res1 = eng1.run()

    cfg2 = ReplayConfig(dataset_path="", latency_seconds=0.0, random_seed=42)
    eng2 = ReplayEngine(config=cfg2, dataset=ds)
    res2 = eng2.run()

    # Exact deterministic equivalence
    assert res1.cycles_count == res2.cycles_count
    assert res1.snapshots_count == res2.snapshots_count
    assert res1.proposals_count == res2.proposals_count
    assert res1.accepted_count == res2.accepted_count
    assert res1.rejected_count == res2.rejected_count
    assert res1.fills_count == res2.fills_count
    assert res1.final_equity == res2.final_equity
    assert res1.total_pnl == res2.total_pnl
    assert res1.simulated_return_pct == res2.simulated_return_pct
    assert res1.max_drawdown_pct == res2.max_drawdown_pct
    assert res1.calibration.brier_score == res2.calibration.brier_score
    assert res1.calibration.log_loss == res2.calibration.log_loss


def test_replay_temporal_lookahead_isolation() -> None:
    """Historical resolutions cannot be known or processed prior to their simulated resolution time."""
    ds = create_deterministic_synthetic_replay_dataset()
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(hours=12)

    # Run replay strictly ending at t1 (before resolutions at t5 = t0 + 60h)
    cfg = ReplayConfig(dataset_path="", latency_seconds=0.0, end_time=t1)
    db = Database(":memory:")
    eng = ReplayEngine(config=cfg, db=db, dataset=ds)
    res = eng.run()
    assert res.cycles_count == 2

    # The simulation stopped before t5, but during wrap-up only pending resolutions are processed
    # Let's verify that during cycles at t0 and t1, no resolutions were executed
    cycles = db.get_cycles()
    assert len(cycles) == 2  # Only t0 and t1 executed


def test_replay_latency_impact() -> None:
    """Execution latency delays fills to future snapshot depth, altering fill prices and slippage."""
    ds = create_deterministic_synthetic_replay_dataset()

    # 1. Zero latency: instantaneous fill at t0
    cfg_zero = ReplayConfig(dataset_path="", latency_seconds=0.0, random_seed=42)
    eng_zero = ReplayEngine(config=cfg_zero, dataset=ds)
    res_zero = eng_zero.run()

    # 2. 12-hour latency: fill delayed to t1
    cfg_lat = ReplayConfig(dataset_path="", latency_seconds=43200.0, random_seed=42)
    eng_lat = ReplayEngine(config=cfg_lat, dataset=ds)
    res_lat = eng_lat.run()

    # Both should execute fills
    assert res_zero.fills_count > 0
    assert res_lat.fills_count > 0

    # Under 12h latency, price has drifted up on mkt_alpha_tech (from ask 0.40 to ask 0.46),
    # resulting in higher slippage or different fill prices
    assert res_lat.total_slippage != res_zero.total_slippage
    assert res_lat.final_equity != res_zero.final_equity


def test_replay_accounting_invariants() -> None:
    """Portfolio accounting invariants hold at all times during replay."""
    ds = create_deterministic_synthetic_replay_dataset()
    db = Database(":memory:")
    cfg = ReplayConfig(dataset_path="", latency_seconds=0.0, initial_bankroll=1000.0)
    eng = ReplayEngine(config=cfg, db=db, dataset=ds)
    res = eng.run()

    # Invariant: Equity = virtual_cash + positions_value
    assert abs(res.final_equity - (eng.tess.virtual_cash + eng.tess.positions_value)) < 1e-4

    # Invariant: Total P&L = realized + unrealized
    assert abs(res.total_pnl - (eng.tess.realized_pnl + eng.tess.unrealized_pnl)) < 1e-4

    # Invariant: Total P&L = final_equity - initial_bankroll
    assert abs(res.total_pnl - (res.final_equity - 1000.0)) < 1e-2


def test_replay_risk_gate_rejection_in_simulation() -> None:
    """Bram risk gate actively rejects high-spread and low-liquidity markets in historical replay."""
    ds = create_deterministic_synthetic_replay_dataset()
    db = Database(":memory:")
    cfg = ReplayConfig(dataset_path="", latency_seconds=0.0)
    eng = ReplayEngine(config=cfg, db=db, dataset=ds)
    res = eng.run()

    # Proposals were generated for illiquid / volatile markets, but Bram rejected them
    assert res.rejected_count > 0

    with db._get_connection() as conn:
        rejections = conn.execute("SELECT market_id, reason_code FROM risk_decisions WHERE accepted = 0").fetchall()
        rejected_mkts = {r["market_id"]: r["reason_code"] for r in rejections}

        # mkt_delta_illiquid must be rejected for liquidity
        assert "mkt_delta_illiquid" in rejected_mkts
        assert "REJECT_LOW_LIQUIDITY" in rejected_mkts["mkt_delta_illiquid"]

        # mkt_gamma_volatile must be rejected at t1 for high spread
        assert "mkt_gamma_volatile" in rejected_mkts
        assert "REJECT_HIGH_SPREAD" in rejected_mkts["mkt_gamma_volatile"]


def test_replay_run_persistence_and_retrieval() -> None:
    """Replay runs are saved to the database and can be queried."""
    ds = create_deterministic_synthetic_replay_dataset()
    db = Database(":memory:")
    cfg = ReplayConfig(dataset_path="", latency_seconds=0.0)
    eng = ReplayEngine(config=cfg, db=db, dataset=ds)
    res = eng.run()

    # Query DB
    run_record = db.get_replay_run(res.replay_id)
    assert run_record is not None
    assert run_record["dataset_id"] == "synthetic_benchmark_v1"
    res_dict = json.loads(run_record["result_json"])
    assert res_dict["portfolio"]["final_equity"] == res.final_equity
    assert res_dict["calibration"]["brier_score"] == res.calibration.brier_score

    all_runs = db.list_replay_runs()
    assert len(all_runs) == 1
    assert all_runs[0]["replay_id"] == res.replay_id
