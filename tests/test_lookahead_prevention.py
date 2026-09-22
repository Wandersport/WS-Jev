"""Tests verifying strict prevention of look-ahead bias and future data leakage in the pipeline."""

from __future__ import annotations

from datetime import timedelta

from pm_research.config import SystemConfig
from pm_research.domain.models import MarketStatus, RiskState
from pm_research.pipeline.ilsa import IlsaEstimator
from pm_research.pipeline.kett import KettSizer
from pm_research.pipeline.rigo import RigoIngestor
from pm_research.utils import now_utc, to_iso_utc


def test_rigo_strips_premature_resolution_on_active_markets():
    """Verify Rigo strips any premature resolution fields attached to active market raw feeds."""
    config = SystemConfig()
    rigo = RigoIngestor(config)
    now = now_utc()

    # Raw market payload maliciously or accidentally containing resolved outcome while active
    raw_feed = [
        {
            "market_id": "mkt_leak_test",
            "question": "Will event E happen?",
            "category": "TECH",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(now + timedelta(days=7)),
            "yes_bid": 0.40,
            "yes_ask": 0.42,
            "no_bid": 0.58,
            "no_ask": 0.60,
            "last_price": 0.41,
            "liquidity": 10000.0,
            "volume_24h": 25000.0,
            # Leaked future outcome fields:
            "resolved_outcome": "YES",
            "resolution_time_actual": to_iso_utc(now + timedelta(days=7)),
        }
    ]

    snapshots = rigo.ingest_snapshots(raw_feed, cycle_id="c_leak", current_time=now)
    assert len(snapshots) == 1
    market = snapshots[0].market

    # Critical invariant: Active markets must have NO resolved outcome
    assert market.status == MarketStatus.ACTIVE
    assert market.resolved_outcome is None, "Active market must have resolved_outcome set to None"
    assert market.resolution_time_actual is None, "Active market must have resolution_time_actual set to None"


def test_pipeline_decision_at_t0_isolated_from_future_resolution():
    """Verify Ilsa and Kett decisions at t0 depend strictly on prior signals, not future resolution."""
    config = SystemConfig()
    rigo = RigoIngestor(config)
    ilsa = IlsaEstimator(config)
    kett = KettSizer(config)

    t0 = now_utc()

    # Two identical active markets at t0 with same prices and signals, but one will eventually resolve YES and one NO
    raw_feed = [
        {
            "market_id": "mkt_will_win",
            "question": "Question 1?",
            "category": "POLITICS",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=10)),
            "yes_bid": 0.45,
            "yes_ask": 0.47,
            "last_price": 0.46,
            "liquidity": 5000.0,
            "volume_24h": 10000.0,
            "metadata": {"research_signal": 0.08},
        },
        {
            "market_id": "mkt_will_lose",
            "question": "Question 2?",
            "category": "POLITICS",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=10)),
            "yes_bid": 0.45,
            "yes_ask": 0.47,
            "last_price": 0.46,
            "liquidity": 5000.0,
            "volume_24h": 10000.0,
            "metadata": {"research_signal": 0.08},
        },
    ]

    snaps = rigo.ingest_snapshots(raw_feed, cycle_id="c_t0", current_time=t0)
    assert len(snaps) == 2

    snap_win = next(s for s in snaps if s.market_id == "mkt_will_win")
    snap_lose = next(s for s in snaps if s.market_id == "mkt_will_lose")

    # Run Ilsa
    est_win = ilsa.estimate(snap_win)
    est_lose = ilsa.estimate(snap_lose)

    # Invariant: identical inputs at t0 must yield identical probability forecasts
    assert abs(est_win.q_hat - est_lose.q_hat) < 1e-9
    assert abs(est_win.uncertainty - est_lose.uncertainty) < 1e-9

    # Run Kett
    prop_win = kett.calculate_proposal(snap_win, est_win, equity=1000.0, available_cash=1000.0, current_risk_state=RiskState.NORMAL)
    prop_lose = kett.calculate_proposal(snap_lose, est_lose, equity=1000.0, available_cash=1000.0, current_risk_state=RiskState.NORMAL)

    assert prop_win is not None
    assert prop_lose is not None

    # Invariant: identical sizing and edge at t0
    assert abs(prop_win.raw_edge - prop_lose.raw_edge) < 1e-9
    assert abs(prop_win.robust_edge - prop_lose.robust_edge) < 1e-9
    assert abs(prop_win.proposed_capital - prop_lose.proposed_capital) < 1e-9
