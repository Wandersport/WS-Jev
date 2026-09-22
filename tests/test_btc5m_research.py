"""Unit and integration tests for BTC 5-minute prospective forecasting laboratory.

Tests:
1. Slug derivation, 300s duration check, outcome mapping by label, priceToBeat policies
2. Chainlink reference feed: anchor detection, return calculations, stale/future tick rejection
3. Polymarket CLOB book: sorting, crossed book rejection, binary cross-outcome synthesis, market_q
4. Binance perp microstructure: depth imbalances, microprice, taker flow coverage (None on missing data)
5. BTC5mFeatureSnapshot: timing deviation, data quality validation, JSON round-tripping
6. 4 Ablation conditions: state isolation, filtering, payload formatting (zero trading questions)
7. Scoring & Round-clustered bootstrap: Brier score, log loss, delta calculations, confidence intervals
8. Safety invariants: Zero order placement, zero execution classes, pure shadow research
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from pm_research.research.btc5m.ablation import (
    COND_A_REF_ONLY,
    COND_B_REF_PERP,
    COND_C_MARKET_AWARE,
    COND_D_FULL,
    build_ablation_state,
    build_btc5m_request_payload,
)
from pm_research.research.btc5m.binance_feed import BinancePerpFeed
from pm_research.research.btc5m.contract import (
    SKIP_NON_300S_DURATION,
    BTC5mContractManager,
    BTC5mOfficialResolution,
)
from pm_research.research.btc5m.lab import (
    BTC5mShadowLab,
    compute_brier,
    compute_log_loss,
)
from pm_research.research.btc5m.poly_book import (
    PolymarketBookCollector,
    parse_and_validate_book,
)
from pm_research.research.btc5m.reference_feed import (
    SOURCE_TWAP_60S,
    ChainlinkReferenceFeed,
    ReferenceFeatures,
)
from pm_research.research.btc5m.snapshot import (
    SKIP_NO_EXACT_ANCHOR,
    BTC5mFeatureSnapshot,
    build_feature_snapshot,
)
from pm_research.research.jev_openrouter import JEV_MODEL_PIN
from pm_research.storage.db import Database


# 1. Contract & Slug Invariant Tests
def test_slug_derivation() -> None:
    mgr = BTC5mContractManager()
    # At epoch 1700000123: 1700000123 // 300 * 300 = 1700000100
    slug = mgr.derive_round_slug(1700000123)
    assert slug == "btc-updown-5m-1700000100"

    # Exactly on boundary 1700000400 (divisible by 300)
    slug_boundary = mgr.derive_round_slug(1700000400)
    assert slug_boundary == "btc-updown-5m-1700000400"


def test_contract_duration_and_outcome_mapping() -> None:
    mgr = BTC5mContractManager()

    # Valid event data with Up/Down outcomes and 300s window
    mock_event = {
        "id": "event_123",
        "title": "Bitcoin Up/Down 5m",
        "markets": [
            {
                "id": "mkt_1",
                "conditionId": "0x" + "a" * 64,
                "question": "Will BTC be up?",
                "resolutionSource": "https://data-api.chain.link/streams/btc-usd-twap-60s-streams",
                "outcomes": json.dumps(["Down", "Up"]),  # Reversed order to test label mapping
                "clobTokenIds": json.dumps(["tok_down", "tok_up"]),
                "acceptingOrders": True,
                "eventMetadata": {"priceToBeat": 85000.50},
            }
        ],
    }

    round_info, skip = mgr.parse_and_validate_round(mock_event, "btc-updown-5m-1700000000")
    assert skip is None
    assert round_info is not None
    assert round_info.up_token_id == "tok_up"
    assert round_info.down_token_id == "tok_down"
    assert round_info.duration_seconds == 300
    assert round_info.price_to_beat == 85000.50
    assert round_info.round_slug == "btc-updown-5m-1700000000"


def test_contract_rejects_non_300s_duration() -> None:
    mgr = BTC5mContractManager()
    mock_event = {
        "markets": [
            {
                "conditionId": "0x" + "b" * 64,
                "resolutionSource": "chainlink-twap",
                "outcomes": ["Up", "Down"],
                "clobTokenIds": ["t1", "t2"],
                "startDate": "2026-01-01T00:00:00Z",
                "endDate": "2026-01-01T00:10:00Z",  # 600s
            }
        ]
    }
    round_info, skip = mgr.parse_and_validate_round(mock_event, "non_standard_slug")
    assert round_info is None
    assert skip == SKIP_NON_300S_DURATION


def test_official_resolution_parsing() -> None:
    res = BTC5mOfficialResolution(
        condition_id="0x123",
        status="resolved",
        winning_outcome="UP",
        payout_up=1.0,
        payout_down=0.0,
        resolved_at_utc=datetime.now(timezone.utc),
        resolution_source="polymarket_data_api",
        raw_payload={"resolution_price": 85100.25},
    )
    assert res.is_resolved is True
    assert res.resolved_outcome == "UP"
    assert res.resolution_price == 85100.25


# 2. Reference Feed Tests
def test_reference_feed_ticks_and_returns() -> None:
    feed = ChainlinkReferenceFeed()
    base_t = 1700000000000  # round start: 1700000000

    # Inject opening anchor tick
    feed.add_tick(SOURCE_TWAP_60S, base_t, 85000.0, received_at_ms=base_t)
    # Inject tick at +30s
    feed.add_tick(SOURCE_TWAP_60S, base_t + 30000, 85085.0, received_at_ms=base_t + 30000)
    # Inject tick at +60s
    feed.add_tick(SOURCE_TWAP_60S, base_t + 60000, 85170.0, received_at_ms=base_t + 60000)

    features = feed.compute_features(
        now_ms=base_t + 60000,
        round_start_epoch=1700000000,
    )
    assert features.current_price == 85170.0
    assert features.price_to_beat == 85000.0
    assert features.price_to_beat_source == "opening-twap-anchor"

    # Distance to beat: (85170 - 85000) / 85000 * 10000 = 20.0 bps
    assert features.distance_to_beat_bps == pytest.approx(20.0, abs=0.1)

    # 30s return: (85170 - 85085) / 85085 * 10000 = 10.0 bps
    assert features.return_30s_bps == pytest.approx(10.0, abs=0.2)


def test_reference_feed_future_and_stale_rejection() -> None:
    feed = ChainlinkReferenceFeed()
    now_ms = 1700000000000

    # Tick too far in future (> 2000ms)
    assert not feed.add_tick(SOURCE_TWAP_60S, now_ms + 5000, 85000.0, received_at_ms=now_ms)
    # Tick too old (> 30 min old)
    assert not feed.add_tick(SOURCE_TWAP_60S, now_ms - 2_000_000, 85000.0, received_at_ms=now_ms)


# 3. Polymarket CLOB Book Tests
def test_clob_sorting_and_crossed_book_rejection() -> None:
    # Crossed book: best bid (0.60) >= best ask (0.50)
    raw = {
        "bids": [{"price": "0.60", "size": "100"}],
        "asks": [{"price": "0.50", "size": "100"}],
        "timestamp": 1700000000000,
    }
    book = parse_and_validate_book(raw, "tok_1", 1700000000000)
    assert book.is_crossed is True
    assert book.is_valid is False

    # Valid book
    raw_valid = {
        "bids": [{"price": "0.45", "size": "50"}, {"price": "0.48", "size": "100"}],
        "asks": [{"price": "0.55", "size": "80"}, {"price": "0.52", "size": "60"}],
        "timestamp": 1700000000000,
    }
    v_book = parse_and_validate_book(raw_valid, "tok_1", 1700000000000)
    assert v_book.is_valid is True
    assert v_book.best_bid == 0.48  # Sorted descending
    assert v_book.best_ask == 0.52  # Sorted ascending
    assert v_book.midpoint == 0.50
    assert v_book.spread == 0.04


def test_clob_binary_complementarity_synthesis() -> None:
    collector = PolymarketBookCollector()
    now_ms = 1700000000000

    # UP book has only asks at 0.02
    up_raw = {
        "bids": [],
        "asks": [{"price": "0.02", "size": "500"}],
        "timestamp": now_ms,
    }
    # DOWN book has only bids at 0.98
    down_raw = {
        "bids": [{"price": "0.98", "size": "500"}],
        "asks": [],
        "timestamp": now_ms,
    }

    # Manually inject parsed books
    up_book = parse_and_validate_book(up_raw, "tok_up", now_ms)
    down_book = parse_and_validate_book(down_raw, "tok_down", now_ms)

    collector.inject_book("tok_up", up_book)
    collector.inject_book("tok_down", down_book)

    state = collector.fetch_market_state("tok_up", "tok_down", now_ms=now_ms)
    assert state.is_valid is True
    # Synthetic effective UP ask is 0.02 (from UP ask 0.02 and DOWN bid 0.98 -> 1.0 - 0.98 = 0.02)
    assert state.market_q == 0.02


# 4. Binance Perpetual Microstructure Tests
def test_binance_depth_imbalance_and_microprice() -> None:
    feed = BinancePerpFeed()
    bids = [(85000.0, 10.0), (84990.0, 15.0)]
    asks = [(85010.0, 5.0), (85020.0, 5.0)]
    feed.add_depth(bids, asks, timestamp_ms=1700000000000)

    feats = feed.compute_features(now_ms=1700000000000, ref_price=85000.0, round_start_price=85000.0)
    assert feats.is_valid is True
    assert feats.best_bid == 85000.0
    assert feats.best_ask == 85010.0
    assert feats.mid_price == 85005.0

    # Top-5 imbalance: (25 - 10) / 35 = 15 / 35 = 0.4286
    assert feats.top5_depth_imbalance == pytest.approx(0.4286, abs=0.001)

    # Microprice: (85000 * 5 + 85010 * 10) / 15 = (425000 + 850100) / 15 = 85006.67
    assert feats.microprice == pytest.approx(85006.67, abs=0.05)


def test_binance_taker_flow_coverage_requirement() -> None:
    feed = BinancePerpFeed()
    now_ms = 1700000060000  # 60 seconds in

    # Only 5 seconds of trade history
    feed.add_trade(timestamp_ms=now_ms - 4000, price=85000.0, qty=1.0, is_buyer_maker=False)
    feed.add_trade(timestamp_ms=now_ms - 2000, price=85001.0, qty=2.0, is_buyer_maker=True)

    feed.add_depth([(85000.0, 1.0)], [(85001.0, 1.0)], now_ms)
    feats = feed.compute_features(now_ms=now_ms)

    # 10s, 30s, 60s taker flow MUST be None because oldest trade is only 4s old (insufficient coverage)
    assert feats.taker_flow_10s_imbalance is None
    assert feats.taker_flow_30s_imbalance is None
    assert feats.taker_flow_60s_imbalance is None

    # Now test feed with full chronological trade coverage (> 60s)
    feed_cov = BinancePerpFeed()
    feed_cov.add_trade(timestamp_ms=now_ms - 65000, price=84995.0, qty=5.0, is_buyer_maker=False)
    feed_cov.add_trade(timestamp_ms=now_ms - 4000, price=85000.0, qty=1.0, is_buyer_maker=False)
    feed_cov.add_trade(timestamp_ms=now_ms - 2000, price=85001.0, qty=2.0, is_buyer_maker=True)
    feed_cov.add_depth([(85000.0, 1.0)], [(85001.0, 1.0)], now_ms)
    feats_with_coverage = feed_cov.compute_features(now_ms=now_ms)
    assert feats_with_coverage.taker_flow_60s_imbalance is not None


# 5. Snapshot Construction & Data Quality Tests
def test_snapshot_builder_and_quality_skip_reasons() -> None:
    now_ms = 1700000060000
    ref_feats = ReferenceFeatures(
        source=SOURCE_TWAP_60S,
        current_price=85000.0,
        timestamp_ms=now_ms,
        age_ms=100,
        anchor_price=None,  # Missing anchor!
        anchor_source=None,
        distance_to_anchor_bps=None,
        return_10s_bps=None,
        return_30s_bps=None,
        return_60s_bps=None,
        since_round_open_bps=None,
    )

    collector = PolymarketBookCollector()
    mock_state = collector.fetch_market_state("u", "d", now_ms=now_ms)

    snap = build_feature_snapshot(
        round_slug="btc-updown-5m-1700000000",
        round_start_epoch=1700000000,
        round_end_epoch=1700000300,
        target_horizon_sec=240,
        captured_at_ms=now_ms,
        ref_features=ref_feats,
        poly_state=mock_state,
        binance_features=None,
    )

    assert snap.is_valid is False
    assert snap.skip_reason == SKIP_NO_EXACT_ANCHOR


# 6. Ablation State Builder & Prompt Invariants
def test_ablation_state_isolation() -> None:
    now_ms = 1700000060000
    snap = BTC5mFeatureSnapshot(
        snapshot_id="snap_1",
        round_slug="btc-updown-5m-1700000000",
        round_start_epoch=1700000000,
        round_end_epoch=1700000300,
        target_horizon_sec=180,
        captured_at_ms=now_ms,
        target_scheduled_ms=now_ms,
        timing_deviation_ms=0,
        seconds_remaining=180.0,
        reference_source="chainlink-twap-60s",
        price_to_beat=85000.0,
        price_to_beat_source="gamma-metadata",
        current_reference_price=85020.0,
        ref_distance_to_beat_bps=2.35,
        ref_return_10s_bps=1.0,
        ref_return_30s_bps=2.0,
        ref_return_60s_bps=3.0,
        ref_data_age_ms=200,
        market_q=0.55,
        up_best_bid=0.54,
        up_best_ask=0.56,
        down_best_bid=0.44,
        down_best_ask=0.46,
        poly_spread=0.02,
        poly_data_age_ms=100,
        poly_book_valid=True,
        binance_perp_mid=85025.0,
        binance_microprice=85026.0,
        binance_microprice_offset_bps=0.12,
        binance_top5_depth_imbalance=0.25,
        binance_top20_depth_imbalance=0.18,
        binance_spread_bps=0.015,
        binance_taker_flow_10s_imbalance=0.10,
        binance_taker_flow_30s_imbalance=0.15,
        binance_taker_flow_60s_imbalance=0.20,
        binance_taker_buy_qty_60s=50.0,
        binance_taker_sell_qty_60s=33.3,
        binance_return_10s_bps=0.5,
        binance_return_30s_bps=1.2,
        binance_return_60s_bps=2.1,
        binance_return_since_open_bps=2.9,
        binance_basis_bps=0.58,
        binance_data_age_ms=150,
        binance_valid=True,
        is_valid=True,
    )

    # Condition A: Reference only. NO perp, NO polymarket odds
    state_a = build_ablation_state(snap, COND_A_REF_ONLY)
    assert "current_reference_price" in state_a
    assert "binance_perp_microstructure" not in state_a
    assert "polymarket_order_book" not in state_a

    # Condition B: Reference + Perp. NO polymarket odds
    state_b = build_ablation_state(snap, COND_B_REF_PERP)
    assert "binance_perp_microstructure" in state_b
    assert "polymarket_order_book" not in state_b

    # Condition C: Market aware. NO binance perp
    state_c = build_ablation_state(snap, COND_C_MARKET_AWARE)
    assert "polymarket_order_book" in state_c
    assert "binance_perp_microstructure" not in state_c

    # Condition D: Full. All features present
    state_d = build_ablation_state(snap, COND_D_FULL)
    assert "binance_perp_microstructure" in state_d
    assert "polymarket_order_book" in state_d


def test_request_payload_asks_only_probability() -> None:
    state = {"market_type": "BTC 5-Minute Up/Down"}
    payload = build_btc5m_request_payload(state)
    assert payload["model"] == JEV_MODEL_PIN
    q = payload["questions"]["will_resolve_up"]
    assert "probability" in q["instructions"].lower()
    # Invariant: Zero trading instructions
    for word in ("trade", "order", "buy", "sell", "bet", "wager", "leverage", "size"):
        assert word not in q["instructions"].lower()


# 7. Scoring and Round-Clustered Bootstrap Tests
def test_brier_and_log_loss_math() -> None:
    # Perfect forecast
    assert compute_brier(1.0, 1) == 0.0
    assert compute_brier(0.0, 0) == 0.0
    # Complete miss
    assert compute_brier(0.0, 1) == 1.0
    assert compute_brier(1.0, 0) == 1.0
    # 50/50 forecast
    assert compute_brier(0.5, 1) == 0.25

    # Log loss bounds
    loss_certain = compute_log_loss(0.999999, 1)
    assert loss_certain < 0.001
    loss_wrong = compute_log_loss(0.000001, 1)
    assert loss_wrong > 10.0


def test_clustered_bootstrap_evaluation() -> None:
    db = Database(":memory:")
    lab = BTC5mShadowLab(db=db)

    # Insert 3 rounds of mock scores
    for r_idx in range(3):
        slug = f"round_{r_idx}"
        for h in (180, 60):
            score = {
                "score_id": f"s_{slug}_{h}",
                "round_slug": slug,
                "target_horizon_sec": h,
                "snapshot_id": f"snap_{slug}_{h}",
                "resolved_outcome": "UP",
                "resolution_price": 85100.0,
                "resolved_at": "2026-01-01T00:05:00Z",
                "scored_at": "2026-01-01T00:06:00Z",
                "market_q": 0.60,
                "market_brier": compute_brier(0.60, 1),
                "market_log_loss": compute_log_loss(0.60, 1),
                "cond_a_prob": 0.70,
                "cond_a_brier": compute_brier(0.70, 1),
                "cond_a_log_loss": compute_log_loss(0.70, 1),
                "cond_b_prob": 0.75,
                "cond_b_brier": compute_brier(0.75, 1),
                "cond_b_log_loss": compute_log_loss(0.75, 1),
                "cond_c_prob": 0.65,
                "cond_c_brier": compute_brier(0.65, 1),
                "cond_c_log_loss": compute_log_loss(0.65, 1),
                "cond_d_prob": 0.80,
                "cond_d_brier": compute_brier(0.80, 1),
                "cond_d_log_loss": compute_log_loss(0.80, 1),
                "metadata": {},
            }
            db.save_btc5m_resolution_score(score)

    summary = lab.compute_evaluation_summary(n_boot=200, seed=123)
    assert summary["total_rounds"] == 3
    assert summary["total_scores"] == 6

    # Forecaster had p=0.70 vs market p=0.60 when outcome was 1 -> forecaster should have lower Brier score
    m_a = summary["metrics_by_condition"][COND_A_REF_ONLY]
    assert m_a["delta_brier"] < 0.0  # negative delta means forecaster outperformed market
    assert m_a["delta_brier_95ci"] is not None


# 8. Structural Safety Invariant Tests
def test_pure_shadow_research_safety() -> None:
    # Ensure lab does not import or contain live broker/order execution
    lab = BTC5mShadowLab(db=Database(":memory:"))
    assert not hasattr(lab, "broker")
    assert not hasattr(lab, "live_broker")
    assert not hasattr(lab, "execute_order")
    assert not hasattr(lab, "submit_order")
    assert not hasattr(lab, "wallet")
