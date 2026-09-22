"""Rigorous verification of portfolio accounting invariants and settlement math."""

from __future__ import annotations

from datetime import timedelta

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    Market,
    MarketQuote,
    MarketSnapshot,
    MarketStatus,
    PaperFill,
    Side,
)
from pm_research.portfolio.tess import TessPortfolio
from pm_research.storage.db import Database
from pm_research.utils import now_utc


def test_accounting_invariant_all_scenarios(tmp_path):
    """Verify equity == cash + positions_value == initial_bankroll + realized_pnl + unrealized_pnl across buys, marks, and resolutions."""
    db_file = tmp_path / "accounting_inv.db"
    db = Database(db_file)
    initial_bankroll = 5000.0
    config = SystemConfig(initial_bankroll=initial_bankroll)
    tess = TessPortfolio(config, db)

    now = now_utc()

    def assert_invariants(label: str):
        # 1. Equity definition
        expected_equity = round(tess.virtual_cash + tess.positions_value, 6)
        assert abs(tess.equity - expected_equity) < 1e-5, f"[{label}] Equity mismatch: {tess.equity} != {expected_equity}"

        # 2. Bankroll reconciliation
        unrealized = sum(p.unrealized_pnl for p in tess.open_positions)
        expected_from_pnl = round(initial_bankroll + tess.realized_pnl + unrealized, 6)
        assert abs(tess.equity - expected_from_pnl) < 1e-5, (
            f"[{label}] Bankroll mismatch: equity={tess.equity}, from_pnl={expected_from_pnl} "
            f"(cash={tess.virtual_cash}, pos_val={tess.positions_value}, realized={tess.realized_pnl}, unrealized={unrealized})"
        )

    # Invariant holds at start
    assert_invariants("Initial state")

    # 1. Buy YES contracts in Market A: 100 contracts @ 0.40, fee $0.10 -> total cost $40.10
    fill_a = PaperFill(
        fill_id="f_a",
        paper_order_id="o_a",
        proposal_id="p_a",
        market_id="m_a",
        side=Side.YES,
        requested_quantity=100.0,
        filled_quantity=100.0,
        unfilled_quantity=0.0,
        average_fill_price=0.40,
        slippage=0.0,
        fees=0.10,
        total_cost=40.10,
        snapshot_id="s_a",
        timestamp=now,
    )
    tess.process_fill(fill_a, category="TECH")
    assert abs(tess.virtual_cash - (5000.0 - 40.10)) < 1e-5
    assert_invariants("Post Buy YES Market A")

    # 2. Buy NO contracts in Market B: 200 contracts @ 0.35, fee $0.20 -> total cost $70.20
    fill_b = PaperFill(
        fill_id="f_b",
        paper_order_id="o_b",
        proposal_id="p_b",
        market_id="m_b",
        side=Side.NO,
        requested_quantity=200.0,
        filled_quantity=200.0,
        unfilled_quantity=0.0,
        average_fill_price=0.35,
        slippage=0.0,
        fees=0.20,
        total_cost=70.20,
        snapshot_id="s_b",
        timestamp=now,
    )
    tess.process_fill(fill_b, category="MACRO")
    assert abs(tess.virtual_cash - (5000.0 - 40.10 - 70.20)) < 1e-5
    assert_invariants("Post Buy NO Market B")

    # 3. Mark to market:
    # Market A price moves from 0.40 -> 0.55 (YES position gain: 100 * 0.55 = $55.00, gain +$14.90)
    # Market B YES price moves from 0.65 -> 0.70 (so NO price is 1.0 - 0.70 = 0.30; NO position loss: 200 * 0.30 = $60.00, loss -$10.20)
    m_a = Market(
        market_id="m_a",
        question="Market A?",
        category="TECH",
        status=MarketStatus.ACTIVE,
        resolution_time=now + timedelta(days=5),
        quote=MarketQuote(yes_bid=0.54, yes_ask=0.56, last_price=0.55, midpoint=0.55, spread=0.02, liquidity=10000.0),
    )
    m_b = Market(
        market_id="m_b",
        question="Market B?",
        category="MACRO",
        status=MarketStatus.ACTIVE,
        resolution_time=now + timedelta(days=5),
        quote=MarketQuote(yes_bid=0.69, yes_ask=0.71, last_price=0.70, midpoint=0.70, spread=0.02, liquidity=10000.0),
    )
    snap_a = MarketSnapshot("s_a2", "c2", "m_a", now, m_a)
    snap_b = MarketSnapshot("s_b2", "c2", "m_b", now, m_b)
    tess.mark_to_market([snap_a, snap_b], cycle_id="c2", current_time=now)

    assert abs(tess.positions_value - (55.00 + 60.00)) < 1e-5
    assert_invariants("Post Mark-to-Market")

    # 4. Settle Market A as YES (WINNING YES outcome):
    # Payout = 100 * $1.00 = $100.00. Cash increases by $100.00.
    # Realized PnL from A = $100.00 - $40.10 = +$59.90.
    tess.resolve_market("m_a", Side.YES, cycle_id="c3", resolved_at=now + timedelta(days=1))
    assert not any(p.market_id == "m_a" for p in tess.open_positions)
    assert abs(tess.realized_pnl - 59.90) < 1e-5
    assert_invariants("Post Settlement Market A (WIN)")

    # 5. Settle Market B as YES (so NO position LOSES):
    # Payout = 200 * $0.00 = $0.00. Cash increases by $0.00.
    # Realized PnL from B = 0 - $70.20 = -$70.20.
    # Total realized PnL = 59.90 - 70.20 = -$10.30.
    tess.resolve_market("m_b", Side.YES, cycle_id="c4", resolved_at=now + timedelta(days=2))
    assert not any(p.market_id == "m_b" for p in tess.open_positions)
    assert len(tess.open_positions) == 0
    assert abs(tess.positions_value - 0.0) < 1e-5
    assert abs(tess.realized_pnl - (-10.30)) < 1e-5
    assert abs(tess.virtual_cash - (5000.0 - 10.30)) < 1e-5
    assert abs(tess.equity - (5000.0 - 10.30)) < 1e-5
    assert_invariants("Post Settlement Market B (LOSS) - All closed")
