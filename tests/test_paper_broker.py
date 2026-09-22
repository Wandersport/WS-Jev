"""PaperBroker unit tests verifying order-book walking, slippage, and paper fill simulation."""

from __future__ import annotations

import pytest

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    Market,
    MarketQuote,
    MarketSnapshot,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    RiskDecision,
    RiskState,
    Side,
    TradeProposal,
)
from pm_research.execution.paper_broker import PaperBroker
from pm_research.utils import now_utc


@pytest.fixture
def broker():
    return PaperBroker(SystemConfig(simulated_fee_rate=0.001))


def test_order_book_walking_multi_level(broker):
    """Verify PaperBroker correctly consumes multiple depth levels and calculates weighted average price and slippage."""
    now = now_utc()
    # Book with 3 ask levels:
    # Level 1: 10 contracts @ 0.40
    # Level 2: 20 contracts @ 0.42
    # Level 3: 50 contracts @ 0.45
    ob = OrderBook(
        bids=[OrderBookLevel(0.38, 50.0)],
        asks=[
            OrderBookLevel(0.40, 10.0),
            OrderBookLevel(0.42, 20.0),
            OrderBookLevel(0.45, 50.0),
        ],
    )
    market = Market(
        market_id="mkt_ob_test",
        question="Test?",
        category="TEST",
        status=MarketStatus.ACTIVE,
        resolution_time=now,
        quote=MarketQuote(yes_bid=0.38, yes_ask=0.40, spread=0.02, liquidity=5000.0),
        order_book=ob,
    )
    snap = MarketSnapshot("s1", "c1", market.market_id, now, market)

    prop = TradeProposal(
        proposal_id="p1",
        cycle_id="c1",
        market_id=market.market_id,
        side=Side.YES,
        quote_price=0.40,
        q_hat=0.55,
        sigma_q=0.02,
        model_version="v1",
        raw_edge=0.15,
        robust_edge=0.10,
        proposed_size_contracts=25.0,  # Needs 10 from level 1 and 15 from level 2
        proposed_capital=10.3,
    )

    dec = RiskDecision(
        decision_id="d1",
        proposal_id=prop.proposal_id,
        cycle_id="c1",
        market_id=market.market_id,
        accepted=True,
        allocated_contracts=25.0,
        allocated_capital=10.3,
        reason_code="ACCEPT",
        risk_state=RiskState.NORMAL,
    )

    res = broker.execute_decision(dec, prop, snap)
    assert res is not None
    order, fill = res

    # Expected:
    # 10 @ 0.40 = $4.00
    # 15 @ 0.42 = $6.30
    # Total cost before fees = $10.30
    # Average fill price = 10.30 / 25 = 0.4120
    assert fill.filled_quantity == 25.0
    assert fill.unfilled_quantity == 0.0
    assert math_close(fill.average_fill_price, 0.4120)
    assert math_close(fill.slippage, 0.4120 - 0.40)  # slippage = 0.0120
    assert fill.fees > 0.0


def test_order_book_partial_fill_on_thin_depth(broker):
    """Verify partial fill when requested quantity exceeds total order book asks."""
    now = now_utc()
    ob = OrderBook(
        bids=[OrderBookLevel(0.38, 50.0)],
        asks=[OrderBookLevel(0.40, 10.0)],  # Only 10 contracts available
    )
    market = Market(
        market_id="mkt_thin",
        question="Test?",
        category="TEST",
        status=MarketStatus.ACTIVE,
        resolution_time=now,
        quote=MarketQuote(yes_bid=0.38, yes_ask=0.40, spread=0.02, liquidity=500.0),
        order_book=ob,
    )
    snap = MarketSnapshot("s1", "c1", market.market_id, now, market)

    prop = TradeProposal(
        proposal_id="p1",
        cycle_id="c1",
        market_id=market.market_id,
        side=Side.YES,
        quote_price=0.40,
        q_hat=0.55,
        sigma_q=0.02,
        model_version="v1",
        raw_edge=0.15,
        robust_edge=0.10,
        proposed_size_contracts=30.0,
        proposed_capital=12.0,
    )
    dec = RiskDecision("d1", "p1", "c1", market.market_id, True, 30.0, 12.0, "ACCEPT", RiskState.NORMAL)

    res = broker.execute_decision(dec, prop, snap)
    assert res is not None
    order, fill = res

    assert fill.filled_quantity == 10.0
    assert fill.unfilled_quantity == 20.0
    assert fill.average_fill_price == 0.40


def test_fallback_conservative_fill_model(broker):
    """Verify fallback model when order book is absent: never assumes midpoint, adds slippage penalty."""
    now = now_utc()
    market = Market(
        market_id="mkt_no_ob",
        question="Test?",
        category="TEST",
        status=MarketStatus.ACTIVE,
        resolution_time=now,
        quote=MarketQuote(yes_bid=0.46, yes_ask=0.54, spread=0.08, liquidity=2000.0),
        order_book=None,
    )
    snap = MarketSnapshot("s1", "c1", market.market_id, now, market)

    prop = TradeProposal(
        proposal_id="p1",
        cycle_id="c1",
        market_id=market.market_id,
        side=Side.YES,
        quote_price=0.54,
        q_hat=0.65,
        sigma_q=0.02,
        model_version="v1",
        raw_edge=0.11,
        robust_edge=0.07,
        proposed_size_contracts=20.0,
        proposed_capital=10.8,
    )
    dec = RiskDecision("d1", "p1", "c1", market.market_id, True, 20.0, 10.8, "ACCEPT", RiskState.NORMAL)

    res = broker.execute_decision(dec, prop, snap)
    assert res is not None
    order, fill = res

    # Average fill must be strictly greater than quote price (conservative slippage penalty)
    assert fill.average_fill_price > prop.quote_price
    assert fill.slippage > 0.0
    assert fill.filled_quantity == 20.0


def math_close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol
