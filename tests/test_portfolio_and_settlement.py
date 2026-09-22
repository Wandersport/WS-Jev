"""Portfolio and settlement tests verifying Tess accounting, marking, and resolutions."""

from __future__ import annotations

from datetime import timedelta

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    Market,
    MarketQuote,
    MarketSnapshot,
    MarketStatus,
    PaperFill,
    ProbabilityEstimate,
    Side,
)
from pm_research.portfolio.tess import TessPortfolio
from pm_research.storage.db import Database
from pm_research.utils import now_utc


def test_portfolio_lifecycle_and_accounting(tmp_path):
    """Test full portfolio lifecycle: cash deduction, mark-to-market, resolution win/loss, and PnL."""
    db_file = tmp_path / "test_tess.db"
    db = Database(db_file)
    config = SystemConfig(initial_bankroll=1000.0)
    tess = TessPortfolio(config, db)

    now = now_utc()
    assert tess.virtual_cash == 1000.0
    assert tess.equity == 1000.0
    assert tess.high_water_mark == 1000.0

    # 1. Process paper fill: buy 50 YES contracts @ 0.40 = $20.00 + $0.02 fees = $20.02
    fill = PaperFill(
        fill_id="f1",
        paper_order_id="o1",
        proposal_id="pr1",
        market_id="m_win",
        side=Side.YES,
        requested_quantity=50.0,
        filled_quantity=50.0,
        unfilled_quantity=0.0,
        average_fill_price=0.40,
        slippage=0.005,
        fees=0.02,
        total_cost=20.02,
        snapshot_id="s1",
        timestamp=now,
    )
    tess.process_fill(fill, category="TECH")

    assert math_close(tess.virtual_cash, 979.98)
    assert len(tess.open_positions) == 1

    # 2. Mark to Market: price moves up to 0.50
    m_win = Market(
        market_id="m_win",
        question="Win?",
        category="TECH",
        status=MarketStatus.ACTIVE,
        resolution_time=now + timedelta(days=1),
        quote=MarketQuote(yes_bid=0.49, yes_ask=0.51, last_price=0.50, midpoint=0.50, spread=0.02, liquidity=5000.0),
    )
    snap_win = MarketSnapshot("s_m1", "c1", "m_win", now, m_win)
    tess.mark_to_market([snap_win], cycle_id="c1", current_time=now)

    # Position value = 50 * 0.50 = $25.00
    # Unrealized PnL = $25.00 - $20.02 = +$4.98
    # Equity = $979.98 + $25.00 = $1,004.98
    assert math_close(tess.positions_value, 25.00)
    assert math_close(tess.equity, 1004.98)
    assert math_close(tess.high_water_mark, 1004.98)
    assert tess.current_drawdown == 0.0

    # 3. Record a probability estimate for calibration observation linking
    est = ProbabilityEstimate(
        estimate_id="e1",
        cycle_id="c1",
        market_id="m_win",
        snapshot_id="s1",
        model_version="ilsa-v1",
        q_hat=0.75,
        reference_probability=0.50,
        uncertainty=0.03,
        timestamp=now,
    )
    db.save_probability_estimate(est)

    # 4. Settle / Resolve Market as YES (Winning outcome!)
    t_res = now + timedelta(days=1)
    observations = tess.resolve_market("m_win", Side.YES, cycle_id="c2", resolved_at=t_res)

    # Winning payout: 50 contracts * $1.00 = $50.00 payout
    # Virtual cash = $979.98 + $50.00 = $1,029.98
    # Realized PnL = $50.00 - $20.02 = +$29.98
    assert math_close(tess.virtual_cash, 1029.98)
    assert math_close(tess.realized_pnl, 29.98)
    assert len(tess.open_positions) == 0

    # Check calibration record
    assert len(observations) == 1
    obs = observations[0]
    assert obs.actual_outcome == 1.0
    assert math_close(obs.brier_score, (0.75 - 1.0) ** 2)


def math_close(a: float, b: float, tol: float = 1e-2) -> bool:
    return abs(a - b) <= tol
