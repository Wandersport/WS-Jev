"""Risk and survival tests verifying Bram constraints, drawdown states, and monotonic risk reduction."""

from __future__ import annotations

from datetime import timedelta

import pytest

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    Market,
    MarketQuote,
    MarketSnapshot,
    MarketStatus,
    PaperPosition,
    RiskState,
    Side,
    TradeProposal,
)
from pm_research.pipeline.bram import BramRiskGate
from pm_research.portfolio.survival import SurvivalManager
from pm_research.utils import now_utc


@pytest.fixture
def base_config():
    return SystemConfig(
        min_liquidity=500.0,
        max_spread=0.10,
        min_time_to_resolution_hours=1.0,
        min_robust_edge=0.03,
        max_uncertainty=0.15,
        max_total_exposure_pct=0.40,
        max_category_exposure_pct=0.20,
    )


@pytest.fixture
def active_snapshot():
    now = now_utc()
    m = Market(
        market_id="m_test",
        question="Test Question?",
        category="POLITICS",
        status=MarketStatus.ACTIVE,
        resolution_time=now + timedelta(days=5),
        quote=MarketQuote(
            yes_bid=0.48,
            yes_ask=0.52,
            spread=0.04,
            liquidity=5000.0,
            volume_24h=10000.0,
        ),
    )
    return MarketSnapshot(
        snapshot_id="snap_1",
        cycle_id="cyc_1",
        market_id=m.market_id,
        timestamp=now,
        market=m,
    )


def test_monotonic_risk_reduction(base_config):
    """Verify strictly that risk parameters scale monotonically as survival state worsens."""
    manager = SurvivalManager(base_config)
    passed, violations = manager.verify_monotonic_risk_reduction()
    assert passed, f"Monotonic risk reduction failed: {violations}"


def test_survival_state_transitions(base_config):
    """Verify deterministic drawdown state mapping."""
    manager = SurvivalManager(base_config)

    assert manager.determine_risk_state(0.00) == RiskState.NORMAL
    assert manager.determine_risk_state(0.04) == RiskState.NORMAL
    assert manager.determine_risk_state(0.05) == RiskState.CAUTION
    assert manager.determine_risk_state(0.08) == RiskState.CAUTION
    assert manager.determine_risk_state(0.10) == RiskState.SURVIVAL
    assert manager.determine_risk_state(0.19) == RiskState.SURVIVAL
    assert manager.determine_risk_state(0.20) == RiskState.CRITICAL
    assert manager.determine_risk_state(0.29) == RiskState.CRITICAL
    assert manager.determine_risk_state(0.30) == RiskState.HALTED
    assert manager.determine_risk_state(0.50) == RiskState.HALTED


def test_halted_mode_blocks_all_proposals(base_config, active_snapshot):
    """Verify that in HALTED mode Bram immediately rejects with REJECT_HALTED_MODE."""
    bram = BramRiskGate(base_config)
    prop = TradeProposal(
        proposal_id="p1",
        cycle_id="c1",
        market_id=active_snapshot.market_id,
        side=Side.YES,
        quote_price=0.52,
        q_hat=0.65,
        sigma_q=0.03,
        model_version="v1",
        raw_edge=0.13,
        robust_edge=0.08,
        proposed_size_contracts=20.0,
        proposed_capital=10.4,
    )

    decision = bram.evaluate(
        proposal=prop,
        snapshot=active_snapshot,
        open_positions=[],
        equity=1000.0,
        virtual_cash=1000.0,
        current_risk_state=RiskState.HALTED,
    )
    assert not decision.accepted
    assert decision.reason_code == "REJECT_HALTED_MODE"


def test_rejections_stale_liquidity_spread_edge(base_config, active_snapshot):
    """Verify individual risk gates."""
    bram = BramRiskGate(base_config)

    # 1. Low liquidity rejection
    active_snapshot.market.quote.liquidity = 100.0  # min is 500
    prop = TradeProposal(
        proposal_id="p1",
        cycle_id="c1",
        market_id=active_snapshot.market_id,
        side=Side.YES,
        quote_price=0.52,
        q_hat=0.65,
        sigma_q=0.03,
        model_version="v1",
        raw_edge=0.13,
        robust_edge=0.08,
        proposed_size_contracts=20.0,
        proposed_capital=10.4,
    )
    dec = bram.evaluate(prop, active_snapshot, [], 1000.0, 1000.0, RiskState.NORMAL)
    assert not dec.accepted
    assert dec.reason_code == "REJECT_LOW_LIQUIDITY"

    # Reset liquidity, test high spread
    active_snapshot.market.quote.liquidity = 5000.0
    active_snapshot.market.quote.spread = 0.15  # max is 0.10
    dec = bram.evaluate(prop, active_snapshot, [], 1000.0, 1000.0, RiskState.NORMAL)
    assert not dec.accepted
    assert dec.reason_code == "REJECT_HIGH_SPREAD"

    # Reset spread, test low robust edge
    active_snapshot.market.quote.spread = 0.04
    prop.robust_edge = 0.01  # min is 0.03
    dec = bram.evaluate(prop, active_snapshot, [], 1000.0, 1000.0, RiskState.NORMAL)
    assert not dec.accepted
    assert dec.reason_code == "REJECT_ROBUST_EDGE_TOO_LOW"

    # Reset edge, test stale quote
    prop.robust_edge = 0.05
    stale_time = now_utc() + timedelta(seconds=600)  # evaluated 600s after snapshot
    dec = bram.evaluate(prop, active_snapshot, [], 1000.0, 1000.0, RiskState.NORMAL, current_time=stale_time)
    assert not dec.accepted
    assert dec.reason_code == "REJECT_STALE_DATA"


def test_exposure_and_concentration_rejections(base_config, active_snapshot):
    """Verify total exposure, category concentration, and duplicate protection."""
    bram = BramRiskGate(base_config)
    now = now_utc()

    prop = TradeProposal(
        proposal_id="p1",
        cycle_id="c1",
        market_id=active_snapshot.market_id,
        side=Side.YES,
        quote_price=0.52,
        q_hat=0.65,
        sigma_q=0.03,
        model_version="v1",
        raw_edge=0.13,
        robust_edge=0.08,
        proposed_size_contracts=20.0,
        proposed_capital=30.0,
    )

    # 1. Duplicate position protection
    existing_pos = PaperPosition(
        position_id="pos_dup",
        market_id=active_snapshot.market_id,
        side=Side.YES,
        quantity=50.0,
        average_entry_price=0.50,
        total_cost=25.0,
        current_price=0.50,
        current_value=25.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        category="POLITICS",
        opened_at=now,
    )
    dec = bram.evaluate(prop, active_snapshot, [existing_pos], 1000.0, 1000.0, RiskState.NORMAL)
    assert not dec.accepted
    assert dec.reason_code == "REJECT_DUPLICATE_POSITION"

    # 2. Total exposure cap (max 40% = $400 of $1000 equity)
    huge_other_pos = PaperPosition(
        position_id="pos_other",
        market_id="m_other",
        side=Side.YES,
        quantity=800.0,
        average_entry_price=0.50,
        total_cost=400.0,
        current_price=0.50,
        current_value=390.0,  # $390 exposed
        unrealized_pnl=-10.0,
        realized_pnl=0.0,
        category="OTHER",
        opened_at=now,
    )
    dec = bram.evaluate(prop, active_snapshot, [huge_other_pos], 1000.0, 1000.0, RiskState.NORMAL)
    assert not dec.accepted
    assert dec.reason_code == "REJECT_MAX_EXPOSURE"
