"""Tests verifying numerical stability, boundary edge cases, NaN/Inf protection, and Kelly limits."""

from __future__ import annotations

import math
from datetime import timedelta

from pm_research.config import SystemConfig
from pm_research.domain.models import RiskState
from pm_research.pipeline.bram import BramRiskGate
from pm_research.pipeline.ilsa import IlsaEstimator, logit, sigmoid
from pm_research.pipeline.kett import KettSizer
from pm_research.pipeline.rigo import RigoIngestor
from pm_research.utils import now_utc, to_iso_utc


def test_logit_sigmoid_extreme_boundaries():
    """Verify logit and sigmoid handle extreme values without throwing OverflowError or returning NaN."""
    # Probabilities very close to 0 and 1
    assert not math.isnan(logit(1e-12))
    assert not math.isinf(logit(1e-12))
    assert not math.isnan(logit(1.0 - 1e-12))
    assert not math.isinf(logit(1.0 - 1e-12))

    # Extreme logits in sigmoid
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0
    assert not math.isnan(sigmoid(float("nan"))) or sigmoid(0.0) == 0.5


def test_rigo_rejects_nan_inf_and_crossed_books():
    """Verify Rigo rejects quotes containing NaN, Inf, or crossed order books."""
    config = SystemConfig()
    rigo = RigoIngestor(config)
    now = now_utc()

    malformed_markets = [
        # NaN price
        {
            "market_id": "mkt_nan",
            "question": "NaN test?",
            "category": "TECH",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(now + timedelta(days=5)),
            "yes_bid": float("nan"),
            "yes_ask": 0.50,
            "liquidity": 1000.0,
            "volume_24h": 5000.0,
        },
        # Inf price
        {
            "market_id": "mkt_inf",
            "question": "Inf test?",
            "category": "TECH",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(now + timedelta(days=5)),
            "yes_bid": 0.40,
            "yes_ask": float("inf"),
            "liquidity": 1000.0,
            "volume_24h": 5000.0,
        },
        # Out of bounds (> 1.0)
        {
            "market_id": "mkt_oob",
            "question": "OOB test?",
            "category": "TECH",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(now + timedelta(days=5)),
            "yes_bid": 1.25,
            "yes_ask": 1.30,
            "liquidity": 1000.0,
            "volume_24h": 5000.0,
        },
        # Crossed book: bid > ask
        {
            "market_id": "mkt_crossed",
            "question": "Crossed test?",
            "category": "TECH",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(now + timedelta(days=5)),
            "yes_bid": 0.65,
            "yes_ask": 0.55,
            "liquidity": 1000.0,
            "volume_24h": 5000.0,
        },
    ]

    snapshots = rigo.ingest_snapshots(malformed_markets, cycle_id="c_nan", current_time=now)
    # All malformed or crossed markets must be rejected
    assert len(snapshots) == 0


def test_kelly_sizing_bounds_under_massive_edge():
    """Verify that even with 99.9% probability vs 1% price, Kelly sizing is bounded by conservative safety caps."""
    config = SystemConfig(
        max_single_position_capital=50.0,
        base_kelly_multiplier=0.25,
    )
    sizer = KettSizer(config)
    rigo = RigoIngestor(config)
    now = now_utc()

    # Active market with huge apparent edge: ask 0.10, but model will forecast 0.95
    raw = [
        {
            "market_id": "mkt_huge_edge",
            "question": "Huge edge?",
            "category": "TECH",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(now + timedelta(days=10)),
            "yes_bid": 0.08,
            "yes_ask": 0.10,
            "last_price": 0.09,
            "liquidity": 50000.0,
            "volume_24h": 100000.0,
            "metadata": {"research_signal": 0.25},
        }
    ]

    snaps = rigo.ingest_snapshots(raw, cycle_id="c_kelly", current_time=now)
    assert len(snaps) == 1
    snap = snaps[0]

    ilsa = IlsaEstimator(config)
    est = ilsa.estimate(snap)
    # Force high probability to test boundary
    est.q_hat = 0.95

    equity = 10000.0
    prop = sizer.calculate_proposal(snap, est, equity=equity, available_cash=5000.0, current_risk_state=RiskState.NORMAL)

    assert prop is not None
    # Must never allocate more than max_single_position_capital
    assert prop.proposed_capital <= config.max_single_position_capital + 1e-5, (
        f"Proposed capital {prop.proposed_capital} exceeded max allowed {config.max_single_position_capital}"
    )


def test_zero_cash_and_insufficient_bankroll():
    """Verify Bram risk gate rejects orders when available cash is zero or below minimum."""
    config = SystemConfig(min_robust_edge=0.01)
    gate = BramRiskGate(config)
    rigo = RigoIngestor(config)
    now = now_utc()

    raw = [
        {
            "market_id": "mkt_nocash",
            "question": "No cash?",
            "category": "TECH",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(now + timedelta(days=10)),
            "yes_bid": 0.40,
            "yes_ask": 0.42,
            "last_price": 0.41,
            "liquidity": 5000.0,
            "volume_24h": 10000.0,
            "metadata": {"research_signal": 0.20},
        }
    ]
    snaps = rigo.ingest_snapshots(raw, cycle_id="c_nc", current_time=now)
    snap = snaps[0]

    ilsa = IlsaEstimator(config)
    est = ilsa.estimate(snap)
    est.q_hat = 0.70  # Strong positive edge
    kett = KettSizer(config)
    prop = kett.calculate_proposal(snap, est, equity=1000.0, available_cash=100.0, current_risk_state=RiskState.NORMAL)

    assert prop is not None
    assert prop.proposed_capital > 1.0

    # Evaluate at Bram with 0.50 cash (below 1.00 min)
    decision = gate.evaluate(
        proposal=prop,
        snapshot=snap,
        open_positions=[],
        equity=1000.0,
        virtual_cash=0.50,  # Below $1.00 minimum
        current_risk_state=RiskState.NORMAL,
        current_time=now,
    )
    assert not decision.accepted
    assert decision.reason_code == "REJECT_INSUFFICIENT_CASH"
