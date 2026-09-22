"""Tests verifying idempotency, duplicate protection, and atomic transaction rollback safety."""

from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    Market,
    MarketQuote,
    MarketSnapshot,
    MarketStatus,
    PaperFill,
    Side,
    TradeProposal,
)
from pm_research.pipeline.bram import BramRiskGate
from pm_research.portfolio.tess import TessPortfolio
from pm_research.storage.db import Database
from pm_research.utils import now_utc


def test_double_settlement_idempotency(tmp_path):
    """Verify resolving the same market twice does not double-credit cash or duplicate calibration records."""
    db_file = tmp_path / "idempotency.db"
    db = Database(db_file)
    config = SystemConfig(initial_bankroll=1000.0)
    tess = TessPortfolio(config, db)

    now = now_utc()

    # Buy 10 YES contracts @ 0.50 = $5.00
    fill = PaperFill(
        fill_id="f1",
        paper_order_id="o1",
        proposal_id="p1",
        market_id="mkt_idem",
        side=Side.YES,
        requested_quantity=10.0,
        filled_quantity=10.0,
        unfilled_quantity=0.0,
        average_fill_price=0.50,
        slippage=0.0,
        fees=0.0,
        total_cost=5.00,
        snapshot_id="s1",
        timestamp=now,
    )
    tess.process_fill(fill, category="TECH")
    assert abs(tess.virtual_cash - 995.00) < 1e-5

    # First resolution: YES -> pays 10 * $1.00 = $10.00
    res_1 = tess.resolve_market("mkt_idem", Side.YES, cycle_id="c1", resolved_at=now + timedelta(days=1))
    cash_after_1 = tess.virtual_cash
    assert abs(cash_after_1 - 1005.00) < 1e-5
    assert len(res_1) == 0  # No estimate was pre-recorded for this market

    # Second resolution (duplicate call): should be completely idempotent
    res_2 = tess.resolve_market("mkt_idem", Side.YES, cycle_id="c2", resolved_at=now + timedelta(days=2))
    assert abs(tess.virtual_cash - cash_after_1) < 1e-5, "Cash must not be credited a second time on duplicate settlement"
    assert len(res_2) == 0


def test_atomic_transaction_rollback_on_error(tmp_path):
    """Verify db.transaction() performs automatic atomic rollback on exception."""
    db_file = tmp_path / "rollback.db"
    db = Database(db_file)
    now = now_utc()

    # Verify transaction rolls back when exception raised inside block
    with pytest.raises(RuntimeError, match="Simulated crash"):
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO cycles (cycle_id, started_at, finished_at, config_hash, notes)
                VALUES ('c_crash', ?, ?, 'hash_test', 'will rollback')
                """,
                (now.isoformat(), now.isoformat()),
            )
            # Simulate sudden crash
            raise RuntimeError("Simulated crash")

    # Verify record was rolled back and is not in database
    cycles = db.get_cycles()
    assert len(cycles) == 0, "Failed transaction must be rolled back completely"


def test_unique_constraints_prevent_duplicates(tmp_path):
    """Verify SQLite UNIQUE constraints prevent duplicate snapshot and calibration observation insertions."""
    db_file = tmp_path / "constraints.db"
    db = Database(db_file)
    now = now_utc()

    with db.transaction() as conn:
        # 1. market_snapshots UNIQUE(cycle_id, market_id)
        conn.execute(
            """
            INSERT INTO market_snapshots (
                snapshot_id, cycle_id, market_id, timestamp, status, category, question,
                resolution_time, liquidity, volume_24h
            ) VALUES ('s1', 'c_same', 'm_dup', ?, 'ACTIVE', 'TECH', 'Q', ?, 100.0, 100.0)
            """,
            (now.isoformat(), (now + timedelta(days=2)).isoformat()),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO market_snapshots (
                    snapshot_id, cycle_id, market_id, timestamp, status, category, question,
                    resolution_time, liquidity, volume_24h
                ) VALUES ('s2', 'c_same', 'm_dup', ?, 'ACTIVE', 'TECH', 'Q', ?, 100.0, 100.0)
                """,
                (now.isoformat(), (now + timedelta(days=2)).isoformat()),
            )

        # 2. calibration_observations UNIQUE(estimate_id, market_id)
        conn.execute(
            """
            INSERT INTO calibration_observations (
                observation_id, estimate_id, market_id, model_version,
                predicted_probability, actual_outcome, category, brier_score, log_loss, resolved_at
            ) VALUES ('obs_1', 'e_same', 'm_dup', 'v1', 0.7, 1.0, 'TECH', 0.09, 0.35, ?)
            """,
            (now.isoformat(),),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO calibration_observations (
                    observation_id, estimate_id, market_id, model_version,
                    predicted_probability, actual_outcome, category, brier_score, log_loss, resolved_at
                ) VALUES ('obs_2', 'e_same', 'm_dup', 'v1', 0.7, 1.0, 'TECH', 0.09, 0.35, ?)
                """,
                (now.isoformat(),),
            )


def test_bram_risk_gate_rejects_duplicate_position():
    """Verify Bram risk gate rejects orders if position in that market is already open."""
    config = SystemConfig()
    gate = BramRiskGate(config)
    now = now_utc()

    m = Market(
        market_id="mkt_open",
        question="Open?",
        category="TECH",
        status=MarketStatus.ACTIVE,
        resolution_time=now + timedelta(days=5),
        quote=MarketQuote(yes_bid=0.48, yes_ask=0.52, spread=0.04, liquidity=5000.0),
    )
    snap = MarketSnapshot("s1", "c1", m.market_id, now, m)

    prop = TradeProposal(
        proposal_id="p1",
        cycle_id="c1",
        market_id=m.market_id,
        side=Side.YES,
        quote_price=0.52,
        q_hat=0.70,
        sigma_q=0.02,
        model_version="ilsa-v1",
        raw_edge=0.18,
        robust_edge=0.15,
        proposed_size_contracts=96.15,
        proposed_capital=50.0,
        reason_codes=["EDGE_POSITIVE"],
        timestamp=now,
    )

    from pm_research.domain.models import PaperPosition, RiskState

    dummy_pos = PaperPosition(
        position_id="pos_1",
        market_id=m.market_id,
        side=Side.YES,
        quantity=50.0,
        average_entry_price=0.50,
        total_cost=25.0,
        current_price=0.50,
        current_value=25.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        category="TECH",
        status="OPEN",
        opened_at=now,
    )

    decision = gate.evaluate(
        proposal=prop,
        snapshot=snap,
        open_positions=[dummy_pos],
        equity=1000.0,
        virtual_cash=500.0,
        current_risk_state=RiskState.NORMAL,
        current_time=now,
    )

    assert not decision.accepted
    assert "DUPLICATE_POSITION" in decision.reason_code
