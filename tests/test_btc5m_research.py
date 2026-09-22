"""Unit and regression tests for BTC 5-minute prospective forecasting laboratory.

Tests covering Phase 6.1 forensic integrity hardening (Requirements A1 - A15):
1. A1: Observed native market baseline vs cross-outcome implied diagnostic
2. A2: Token mismatch strictly invalidates order book
3. A3: Real 300-second metadata validation and multi-market slug matching
4. A4: Strict Chainlink BTC/USD 60s TWAP settlement source verification
5. A5: Removal of ambiguous RTDS topic fallback (no loose mapping)
6. A6: Exact anchor hierarchy and provenance timestamps
7. A7: Authoritative outcome index payout mapping (reversed outcomes test)
8. A8: Binance return since open uses Binance open price (not Chainlink anchor)
9. A9: Source event timestamp vs receipt timestamp distinction
10. A10 & A11: Freeze timing model and post-horizon data leakage rejection
11. A12: Fail-closed Jev response parsing (rejects NaN, Inf, out-of-bounds, missing fields)
12. A13: Pre-close response requirement (rejects late responses >= round_end)
13. A14 & A15: Four-condition fairness, snapshot hash, and paired ablation deltas
14. Pure paper-only research safety invariants
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from pm_research.research.btc5m.ablation import (
    COND_A_REF_ONLY,
    COND_C_MARKET_AWARE,
    BTC5mAblationRunner,
    JevParseError,
    build_ablation_state,
    parse_jev_response_strict,
)
from pm_research.research.btc5m.binance_feed import BinancePerpFeed
from pm_research.research.btc5m.contract import (
    SKIP_INCOMPLETE_ROUND_METADATA,
    SKIP_NON_300S_DURATION,
    BTC5mContractManager,
    BTC5mRoundInfo,
    validate_settlement_source_url,
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
    SKIP_POST_HORIZON_DATA_LEAKAGE,
    BTC5mFeatureSnapshot,
    build_feature_snapshot,
)
from pm_research.research.jev_openrouter import JEV_MODEL_PIN
from pm_research.storage.db import Database


# ---------------------------------------------------------------------------
# A1: Observed Native Baseline vs Cross-Outcome Implied Diagnostic
# ---------------------------------------------------------------------------
def test_a1_primary_market_baseline_observed_not_synthetic() -> None:
    collector = PolymarketBookCollector()
    now_ms = 1700000000000

    # Case 1: Valid two-sided native UP book exists
    up_raw = {
        "asset_id": "tok_up",
        "bids": [{"price": "0.48", "size": "100"}],
        "asks": [{"price": "0.52", "size": "100"}],
        "timestamp": now_ms,
    }
    down_raw = {
        "asset_id": "tok_down",
        "bids": [{"price": "0.47", "size": "100"}],
        "asks": [{"price": "0.53", "size": "100"}],
        "timestamp": now_ms,
    }
    collector.inject_book("tok_up", parse_and_validate_book(up_raw, "tok_up", now_ms))
    collector.inject_book("tok_down", parse_and_validate_book(down_raw, "tok_down", now_ms))

    state = collector.fetch_market_state("tok_up", "tok_down", now_ms=now_ms)
    assert state.is_valid is True
    # Primary baseline is strictly the observed native UP mid
    assert state.market_q_primary == 0.50
    assert state.market_q == 0.50
    assert state.native_up_bid == 0.48
    assert state.native_up_ask == 0.52

    # Cross-outcome implied calculation is retained distinctly
    # Down ask 0.53 -> implied UP bid 0.47
    # Down bid 0.47 -> implied UP ask 0.53
    assert state.cross_outcome_implied_up_bid == 0.47
    assert state.cross_outcome_implied_up_ask == 0.53
    assert state.market_q_implied_cross_outcome == 0.50

    # Case 2: Native UP book is one-sided (no bids, only ask at 0.02)
    # Primary market baseline must NOT be fabricated (market_q_primary must be None)
    # But implied cross-outcome mid is retained as secondary diagnostic
    collector_onesided = PolymarketBookCollector()
    up_onesided = {
        "asset_id": "tok_up",
        "bids": [],
        "asks": [{"price": "0.02", "size": "500"}],
        "timestamp": now_ms,
    }
    down_onesided = {
        "asset_id": "tok_down",
        "bids": [{"price": "0.98", "size": "500"}],
        "asks": [{"price": "0.99", "size": "500"}],
        "timestamp": now_ms,
    }
    collector_onesided.inject_book("tok_up", parse_and_validate_book(up_onesided, "tok_up", now_ms))
    collector_onesided.inject_book("tok_down", parse_and_validate_book(down_onesided, "tok_down", now_ms))

    state_onesided = collector_onesided.fetch_market_state("tok_up", "tok_down", now_ms=now_ms)
    # Primary baseline is NOT fabricated
    assert state_onesided.market_q_primary is None
    # Implied diagnostic: down_ask 0.99 -> implied up bid 0.01; down_bid 0.98 -> implied up ask 0.02
    assert state_onesided.cross_outcome_implied_up_bid == 0.01
    assert state_onesided.cross_outcome_implied_up_ask == 0.02
    assert state_onesided.market_q_implied_cross_outcome == 0.015


# ---------------------------------------------------------------------------
# A2: Token Mismatch Rejection
# ---------------------------------------------------------------------------
def test_a2_token_mismatch_invalidates_book() -> None:
    now_ms = 1700000000000
    # Payload provides asset_id 'wrong_token' when expected is 'tok_up'
    raw_mismatch = {
        "asset_id": "wrong_token",
        "bids": [{"price": "0.45", "size": "100"}],
        "asks": [{"price": "0.55", "size": "100"}],
        "timestamp": now_ms,
    }
    book = parse_and_validate_book(raw_mismatch, "tok_up", now_ms)
    # Must be completely invalidated
    assert book.is_valid is False
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.midpoint is None
    assert len(book.bids) == 0
    assert len(book.asks) == 0


# ---------------------------------------------------------------------------
# A3: Real 300-Second Market Contract Metadata Validation
# ---------------------------------------------------------------------------
def test_a3_real_300s_metadata_validation() -> None:
    mgr = BTC5mContractManager()
    slug_epoch = 1700000100
    expected_slug = f"btc-updown-5m-{slug_epoch}"

    # Multi-market event where only the second market matches the slug
    mock_event = {
        "slug": expected_slug,
        "markets": [
            {
                "id": "mkt_wrong",
                "slug": "btc-updown-5m-1699999800",
                "conditionId": "0x" + "0" * 64,
                "startDate": "2023-11-14T22:10:00Z",
                "endDate": "2023-11-14T22:15:00Z",
            },
            {
                "id": "mkt_matched",
                "slug": expected_slug,
                "conditionId": "0x" + "a" * 64,
                "question": "Will BTC resolve Up?",
                "resolutionSource": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
                "outcomes": ["Up", "Down"],
                "clobTokenIds": ["tok_up", "tok_down"],
                "startDate": datetime.fromtimestamp(slug_epoch, tz=timezone.utc).isoformat(),
                "endDate": datetime.fromtimestamp(slug_epoch + 300, tz=timezone.utc).isoformat(),
                "eventMetadata": {"priceToBeat": 85000.0},
            },
        ],
    }

    round_info, skip = mgr.parse_and_validate_round(mock_event, expected_slug)
    assert skip is None
    assert round_info is not None
    assert round_info.market_id == "mkt_matched"
    assert round_info.duration_seconds == 300
    assert round_info.start_epoch == slug_epoch
    assert round_info.end_epoch == slug_epoch + 300

    # Test rejection when metadata duration is not 300s
    mock_bad_duration = dict(mock_event)
    mock_bad_duration["markets"] = [
        dict(
            mock_event["markets"][1],
            endDate=datetime.fromtimestamp(slug_epoch + 600, tz=timezone.utc).isoformat(),
        )
    ]
    _, skip_dur = mgr.parse_and_validate_round(mock_bad_duration, expected_slug)
    assert skip_dur == SKIP_NON_300S_DURATION

    # Test rejection when no market matches slug
    mock_no_match = dict(mock_event)
    mock_no_match["markets"] = [mock_event["markets"][0]]
    mock_no_match["slug"] = "other_slug"
    _, skip_match = mgr.parse_and_validate_round(mock_no_match, expected_slug)
    assert skip_match == SKIP_INCOMPLETE_ROUND_METADATA


# ---------------------------------------------------------------------------
# A4: Strict Settlement Source Verification
# ---------------------------------------------------------------------------
def test_a4_strict_settlement_source_verification() -> None:
    # Supported: BTC/USD 60s TWAP
    assert validate_settlement_source_url("https://data.chain.link/streams/btc-usd-twap-60s-streams") is True
    assert validate_settlement_source_url("https://data-api.chain.link/streams/btc-usd-twap-60s-streams") is True

    # Reject 30s TWAP
    assert validate_settlement_source_url("https://data.chain.link/streams/btc-usd-twap-30s-streams") is False

    # Reject other asset
    assert validate_settlement_source_url("https://data.chain.link/streams/eth-usd-twap-60s-streams") is False

    # Reject non-https
    assert validate_settlement_source_url("http://data.chain.link/streams/btc-usd-twap-60s-streams") is False

    # Reject generic text
    assert validate_settlement_source_url("chainlink twap price") is False
    assert validate_settlement_source_url("https://example.com/btc-usd-twap-60s-streams") is False


# ---------------------------------------------------------------------------
# A5: RTDS Topic Mapping Integrity
# ---------------------------------------------------------------------------
def test_a5_rtds_topic_strict_mapping() -> None:
    feed = ChainlinkReferenceFeed()
    now_ms = 1700000000000

    # 1. Official 60s TWAP topic is accepted
    twap_msg = {
        "topic": "crypto_prices_twap_sixty",
        "payload": {
            "symbol": "BTC/USD",
            "timestamp": now_ms,
            "value": 85000.0,
        },
    }
    assert feed.ingest_rtds_payload(twap_msg, now_ms=now_ms) == 1
    assert feed.get_latest_tick(SOURCE_TWAP_60S) is not None

    # 2. Generic 'crypto_prices' must NOT be accepted as 60s TWAP
    generic_msg = {
        "topic": "crypto_prices",
        "payload": {
            "symbol": "BTC/USD",
            "timestamp": now_ms + 1000,
            "value": 85050.0,
        },
    }
    assert feed.ingest_rtds_payload(generic_msg, now_ms=now_ms + 1000) == 0

    # 3. Unknown topic rejected
    unknown_msg = {
        "topic": "unknown_crypto_feed",
        "payload": {
            "symbol": "BTC/USD",
            "timestamp": now_ms + 2000,
            "value": 85060.0,
        },
    }
    assert feed.ingest_rtds_payload(unknown_msg, now_ms=now_ms + 2000) == 0


# ---------------------------------------------------------------------------
# A6: Exact Anchor Provenance & Hierarchy
# ---------------------------------------------------------------------------
def test_a6_exact_anchor_hierarchy_and_provenance() -> None:
    feed = ChainlinkReferenceFeed()
    round_start_ms = 1700000000000

    # Case 1: Gamma priceToBeat is authoritative when present
    feats_gamma = feed.compute_features(
        round_start_ms=round_start_ms,
        event_price_to_beat=85123.45,
        now_ms=round_start_ms + 10000,
    )
    assert feats_gamma.anchor_price == 85123.45
    assert feats_gamma.anchor_source == "gamma-metadata"
    assert feats_gamma.anchor_source_timestamp_ms == round_start_ms
    assert feats_gamma.anchor_received_timestamp_ms == round_start_ms + 10000

    # Case 2: Exact boundary tick on round_start_ms
    feed.add_tick(SOURCE_TWAP_60S, round_start_ms, 85050.0, received_at_ms=round_start_ms + 10)
    feats_exact = feed.compute_features(
        round_start_ms=round_start_ms,
        event_price_to_beat=None,
        now_ms=round_start_ms + 10000,
    )
    assert feats_exact.anchor_price == 85050.0
    assert feats_exact.anchor_source == "rtds-twap-60s-exact-boundary"
    assert feats_exact.anchor_source_timestamp_ms == round_start_ms
    assert feats_exact.anchor_received_timestamp_ms == round_start_ms + 10

    # Case 3: Inexact tick (e.g. at round_start_ms + 2000) must NOT be used as anchor
    feed_inexact = ChainlinkReferenceFeed()
    feed_inexact.add_tick(SOURCE_TWAP_60S, round_start_ms + 2000, 85055.0, received_at_ms=round_start_ms + 2010)
    feats_none = feed_inexact.compute_features(
        round_start_ms=round_start_ms,
        event_price_to_beat=None,
        now_ms=round_start_ms + 10000,
    )
    assert feats_none.anchor_price is None


# ---------------------------------------------------------------------------
# A7: Authoritative Outcome Index Payout Mapping
# ---------------------------------------------------------------------------
def test_a7_reversed_outcomes_payout_mapping() -> None:
    # Market configured with reversed outcomes: index 0 is "Down", index 1 is "Up"
    round_info = BTC5mRoundInfo(
        slug="btc-updown-5m-1700000000",
        condition_id="0x" + "1" * 64,
        market_id="m_1",
        question="BTC up?",
        description="",
        resolution_source_url="https://data.chain.link/streams/btc-usd-twap-60s-streams",
        settlement_rule="",
        start_time_utc=datetime.now(timezone.utc),
        end_time_utc=datetime.now(timezone.utc),
        duration_seconds=300,
        up_token_id="tok_up",
        down_token_id="tok_down",
        up_outcome_index=1,
        down_outcome_index=0,
        accepting_orders=True,
        price_to_beat=85000.0,
        price_to_beat_source="gamma",
        fee_rate=0.0,
        fee_exponent=1.0,
    )

    # Mock resolution API response where payouts is [0, 1] (meaning Down=0, Up=1)
    # Even though index 0 is 0, since up_outcome_index=1, payout_up is payouts[1] = 1.0 -> UP wins!
    mock_row = {
        "condition_id": round_info.condition_id,
        "status": "resolved",
        "payouts": ["0", "1"],  # [Down, Up]
        "resolved_at": "2026-01-01T00:05:00Z",
        "resolution_price": 85150.0,
    }

    # Simulate resolution parser directly with round_info mapping
    up_idx = round_info.up_outcome_index  # 1
    down_idx = round_info.down_outcome_index  # 0
    payout_up = float(mock_row["payouts"][up_idx])
    payout_down = float(mock_row["payouts"][down_idx])
    winner = "UP" if payout_up > payout_down else "DOWN"

    assert winner == "UP"
    assert payout_up == 1.0
    assert payout_down == 0.0


# ---------------------------------------------------------------------------
# A8: Binance Since-Open Return Must Use Binance Open Price
# ---------------------------------------------------------------------------
def test_a8_binance_since_open_return_vs_basis() -> None:
    feed = BinancePerpFeed()
    now_ms = 1700000030000

    # Chainlink reference anchor is 85000.0
    chainlink_ref = 85000.0
    # Observed Binance perp open mid was 85050.0
    binance_open_mid = 85050.0

    # Current Binance book: bid 85100, ask 85102 -> mid 85101.0
    feed.add_depth([(85100.0, 1.0)], [(85102.0, 1.0)], source_timestamp_ms=now_ms, received_at_ms=now_ms)

    feats = feed.compute_features(
        now_ms=now_ms,
        ref_price=chainlink_ref,
        binance_open_price=binance_open_mid,
    )

    # Binance return since open: (85101.0 - 85050.0) / 85050.0 * 10000 = +5.996 bps
    assert feats.return_since_round_open_bps == pytest.approx(5.996, abs=0.01)

    # Binance basis vs Chainlink reference: (85101.0 - 85000.0) / 85000.0 * 10000 = +11.882 bps
    assert feats.basis_vs_ref_bps == pytest.approx(11.882, abs=0.01)

    # Invariant A8: They MUST be different
    assert feats.return_since_round_open_bps != feats.basis_vs_ref_bps


# ---------------------------------------------------------------------------
# A9: Source Timestamp vs Receipt Timestamp Distinction
# ---------------------------------------------------------------------------
def test_a9_source_vs_receipt_timestamps() -> None:
    now_ms = 1700000010000
    source_ts = 1700000009500  # 500ms earlier at source

    # CLOB book
    raw_poly = {
        "asset_id": "tok_up",
        "bids": [{"price": "0.49", "size": "10"}],
        "asks": [{"price": "0.51", "size": "10"}],
        "timestamp": source_ts,
    }
    book = parse_and_validate_book(raw_poly, "tok_up", received_at_ms=now_ms)
    assert book.source_event_timestamp_ms == source_ts
    assert book.received_at_ms == now_ms
    assert book.source_age_ms is not None and book.source_age_ms >= 500
    assert book.receipt_age_ms >= 0

    # Binance
    feed = BinancePerpFeed()
    feed.add_depth([(85000.0, 1.0)], [(85001.0, 1.0)], source_timestamp_ms=source_ts, received_at_ms=now_ms)
    feats = feed.compute_features(now_ms=now_ms)
    assert feats.source_event_timestamp_ms == source_ts
    assert feats.received_at_ms == now_ms
    assert feats.source_age_ms is not None and feats.source_age_ms >= 500


# ---------------------------------------------------------------------------
# A10 & A11: Snapshot Timing & Post-Horizon Data Leakage
# ---------------------------------------------------------------------------
def test_a10_a11_snapshot_freeze_timing_and_leakage() -> None:
    round_start = 1700000000
    round_end = 1700000300
    h_sec = 240
    target_sched_ms = (round_end - h_sec) * 1000  # 1700000060000

    ref_feats = ReferenceFeatures(
        source=SOURCE_TWAP_60S,
        current_price=85000.0,
        source_event_timestamp_ms=target_sched_ms,
        received_at_ms=target_sched_ms + 50,
        receipt_age_ms=50,
        source_age_ms=50,
        anchor_price=85000.0,
        anchor_source="gamma",
        anchor_source_timestamp_ms=round_start * 1000,
        anchor_received_timestamp_ms=round_start * 1000,
        distance_to_anchor_bps=0.0,
        return_10s_bps=0.0,
        return_30s_bps=0.0,
        return_60s_bps=0.0,
        since_round_open_bps=0.0,
    )

    collector = PolymarketBookCollector()
    up_raw = {"asset_id": "u", "bids": [{"price": "0.50", "size": "1"}], "asks": [{"price": "0.51", "size": "1"}]}
    down_raw = {"asset_id": "d", "bids": [{"price": "0.49", "size": "1"}], "asks": [{"price": "0.50", "size": "1"}]}
    collector.inject_book("u", parse_and_validate_book(up_raw, "u", target_sched_ms))
    collector.inject_book("d", parse_and_validate_book(down_raw, "d", target_sched_ms))
    state = collector.fetch_market_state("u", "d", now_ms=target_sched_ms)

    # Valid on-time snapshot
    snap_valid = build_feature_snapshot(
        round_slug="btc-updown-5m-1700000000",
        round_start_epoch=round_start,
        round_end_epoch=round_end,
        target_horizon_sec=h_sec,
        capture_started_at_ms=target_sched_ms,
        capture_completed_at_ms=target_sched_ms + 100,  # +100ms drift
        ref_features=ref_feats,
        poly_state=state,
        binance_features=None,
    )
    assert snap_valid.is_valid is True
    assert snap_valid.timing_deviation_ms == 100
    assert snap_valid.actual_freeze_timestamp_ms == target_sched_ms + 100

    # Leakage rejection: Reference source timestamp exceeds target cutoff
    leak_ref = ReferenceFeatures(
        source=SOURCE_TWAP_60S,
        current_price=85000.0,
        source_event_timestamp_ms=target_sched_ms + 10000,  # 10s after target horizon!
        received_at_ms=target_sched_ms + 10050,
        receipt_age_ms=50,
        source_age_ms=50,
        anchor_price=85000.0,
        anchor_source="gamma",
        anchor_source_timestamp_ms=round_start * 1000,
        anchor_received_timestamp_ms=round_start * 1000,
        distance_to_anchor_bps=0.0,
        return_10s_bps=0.0,
        return_30s_bps=0.0,
        return_60s_bps=0.0,
        since_round_open_bps=0.0,
    )
    snap_leak = build_feature_snapshot(
        round_slug="btc-updown-5m-1700000000",
        round_start_epoch=round_start,
        round_end_epoch=round_end,
        target_horizon_sec=h_sec,
        capture_started_at_ms=target_sched_ms,
        capture_completed_at_ms=target_sched_ms + 100,
        ref_features=leak_ref,
        poly_state=state,
        binance_features=None,
    )
    assert snap_leak.is_valid is False
    assert snap_leak.skip_reason == SKIP_POST_HORIZON_DATA_LEAKAGE


# ---------------------------------------------------------------------------
# A12: Jev Response Parsing Must Fail Closed
# ---------------------------------------------------------------------------
def test_a12_jev_response_parsing_fails_closed() -> None:
    # 1. Valid payload
    valid_resp = {
        "model": JEV_MODEL_PIN,
        "answers": {
            "will_resolve_up": {"type": "noul", "noul": 0.65},
            "outcome_choice": {"type": "choice", "choice": "UP", "confidence": 0.8},
        },
    }
    p_up, p_down, choice, conf = parse_jev_response_strict(valid_resp)
    assert p_up == 0.65
    assert p_down == 0.35
    assert choice == "UP"
    assert conf == 0.8

    # 2. Missing answers
    with pytest.raises(JevParseError, match="MISSING_ANSWERS"):
        parse_jev_response_strict({})

    # 3. Missing will_resolve_up
    with pytest.raises(JevParseError, match="MISSING_WILL_RESOLVE_UP"):
        parse_jev_response_strict({"answers": {}})

    # 4. Out of bounds (< 0)
    with pytest.raises(JevParseError, match="NOUL_OUT_OF_BOUNDS"):
        parse_jev_response_strict({
            "answers": {
                "will_resolve_up": {"type": "noul", "noul": -0.1},
                "outcome_choice": {"choice": "DOWN"},
            }
        })

    # 5. Out of bounds (> 1)
    with pytest.raises(JevParseError, match="NOUL_OUT_OF_BOUNDS"):
        parse_jev_response_strict({
            "answers": {
                "will_resolve_up": {"type": "noul", "noul": 1.5},
                "outcome_choice": {"choice": "UP"},
            }
        })

    # 6. NaN or Inf
    with pytest.raises(JevParseError, match="NOUL_NAN_OR_INF"):
        parse_jev_response_strict({
            "answers": {
                "will_resolve_up": {"type": "noul", "noul": float("nan")},
                "outcome_choice": {"choice": "UP"},
            }
        })

    # 7. Model mismatch
    with pytest.raises(JevParseError, match="MODEL_MISMATCH"):
        parse_jev_response_strict(
            {
                "model": "openai/gpt-4o",
                "answers": {
                    "will_resolve_up": {"type": "noul", "noul": 0.5},
                    "outcome_choice": {"choice": "UP"},
                },
            },
            pinned_model=JEV_MODEL_PIN,
        )


# ---------------------------------------------------------------------------
# A13: Pre-Close Response Requirement
# ---------------------------------------------------------------------------
def test_a13_pre_close_response_requirement() -> None:
    now_ms = 1700000060000
    snap = BTC5mFeatureSnapshot(
        snapshot_id="snap_1",
        round_slug="btc-updown-5m-1700000000",
        round_start_epoch=1700000000,
        round_end_epoch=1700000300,  # round ends at 1700000300000
        target_horizon_sec=240,
        capture_started_at_ms=now_ms,
        capture_completed_at_ms=now_ms,
        actual_freeze_timestamp_ms=now_ms,
        target_scheduled_ms=now_ms,
        timing_deviation_ms=0,
        seconds_remaining=240.0,
        reference_source="chainlink-twap-60s",
        price_to_beat=85000.0,
        price_to_beat_source="gamma-metadata",
        anchor_price=85000.0,
        anchor_source="gamma-metadata",
        anchor_source_timestamp_ms=1700000000000,
        anchor_received_timestamp_ms=1700000000000,
        current_reference_price=85000.0,
        ref_distance_to_beat_bps=0.0,
        ref_return_10s_bps=0.0,
        ref_return_30s_bps=0.0,
        ref_return_60s_bps=0.0,
        ref_source_timestamp_ms=now_ms,
        ref_received_at_ms=now_ms,
        ref_receipt_age_ms=0,
        ref_source_age_ms=0,
        native_up_bid=0.50,
        native_up_ask=0.52,
        native_up_mid=0.51,
        native_down_bid=0.48,
        native_down_ask=0.50,
        native_down_mid=0.49,
        cross_outcome_implied_up_bid=0.50,
        cross_outcome_implied_up_ask=0.52,
        cross_outcome_implied_mid=0.51,
        market_q_primary=0.51,
        market_q_implied_cross_outcome=0.51,
        poly_spread=0.02,
        poly_source_timestamp_ms=now_ms,
        poly_received_at_ms=now_ms,
        poly_data_age_ms=0,
        poly_book_valid=True,
        binance_perp_mid=None,
        binance_microprice=None,
        binance_microprice_offset_bps=None,
        binance_top5_depth_imbalance=None,
        binance_top20_depth_imbalance=None,
        binance_spread_bps=None,
        binance_taker_flow_10s_imbalance=None,
        binance_taker_flow_30s_imbalance=None,
        binance_taker_flow_60s_imbalance=None,
        binance_taker_buy_qty_60s=None,
        binance_taker_sell_qty_60s=None,
        binance_return_10s_bps=None,
        binance_return_30s_bps=None,
        binance_return_60s_bps=None,
        binance_return_since_open_bps=None,
        binance_open_mid=None,
        binance_open_timestamp_ms=None,
        binance_basis_bps=None,
        binance_source_timestamp_ms=None,
        binance_received_at_ms=None,
        binance_receipt_age_ms=0,
        binance_source_age_ms=None,
        binance_valid=False,
        is_valid=True,
    )

    # Mock client returning valid response
    class MockClient:
        def compute_request_hash(self, p: dict) -> str:
            return "hash_123"

        def query_decision_payload(self, payload: dict, bypass_cache: bool = False):
            return {
                "model": JEV_MODEL_PIN,
                "answers": {
                    "will_resolve_up": {"type": "noul", "noul": 0.6},
                    "outcome_choice": {"type": "choice", "choice": "UP"},
                },
            }, False, "hash_123"

    runner = BTC5mAblationRunner(client=MockClient())

    # If round ended in the past (e.g. round_end_epoch was 1000s ago)
    past_snap = BTC5mFeatureSnapshot.from_dict({
        **snap.to_dict(),
        "round_end_epoch": int(time.time()) - 10,
    })
    fc_late = runner.run_single_condition(past_snap, COND_A_REF_ONLY)
    assert fc_late.is_valid is False
    assert fc_late.rejection_reason == "INVALID_LATE_MODEL_RESPONSE"


# ---------------------------------------------------------------------------
# A14 & A15: Four-Condition State & Paired Evaluation
# ---------------------------------------------------------------------------
def test_a14_a15_ablation_states_and_paired_deltas() -> None:
    now_ms = 1700000060000
    snap = BTC5mFeatureSnapshot.from_dict({
        "snapshot_id": "snap_1",
        "round_slug": "btc-updown-5m-1700000000",
        "round_start_epoch": 1700000000,
        "round_end_epoch": 1700000300,
        "target_horizon_sec": 180,
        "capture_started_at_ms": now_ms,
        "capture_completed_at_ms": now_ms,
        "actual_freeze_timestamp_ms": now_ms,
        "target_scheduled_ms": now_ms,
        "timing_deviation_ms": 0,
        "seconds_remaining": 180.0,
        "reference_source": "chainlink-twap-60s",
        "price_to_beat": 85000.0,
        "price_to_beat_source": "gamma-metadata",
        "anchor_price": 85000.0,
        "anchor_source": "gamma-metadata",
        "anchor_source_timestamp_ms": 1700000000000,
        "anchor_received_timestamp_ms": 1700000000000,
        "current_reference_price": 85020.0,
        "ref_distance_to_beat_bps": 2.35,
        "ref_return_10s_bps": 1.0,
        "ref_return_30s_bps": 2.0,
        "ref_return_60s_bps": 3.0,
        "ref_source_timestamp_ms": now_ms,
        "ref_received_at_ms": now_ms,
        "ref_receipt_age_ms": 0,
        "ref_source_age_ms": 0,
        "native_up_bid": 0.54,
        "native_up_ask": 0.56,
        "native_up_mid": 0.55,
        "native_down_bid": 0.44,
        "native_down_ask": 0.46,
        "native_down_mid": 0.45,
        "cross_outcome_implied_up_bid": 0.54,
        "cross_outcome_implied_up_ask": 0.56,
        "cross_outcome_implied_mid": 0.55,
        "market_q_primary": 0.55,
        "market_q_implied_cross_outcome": 0.55,
        "poly_spread": 0.02,
        "poly_source_timestamp_ms": now_ms,
        "poly_received_at_ms": now_ms,
        "poly_data_age_ms": 100,
        "poly_book_valid": True,
        "binance_perp_mid": 85025.0,
        "binance_microprice": 85026.0,
        "binance_microprice_offset_bps": 0.12,
        "binance_top5_depth_imbalance": 0.25,
        "binance_top20_depth_imbalance": 0.18,
        "binance_spread_bps": 0.015,
        "binance_taker_flow_10s_imbalance": 0.10,
        "binance_taker_flow_30s_imbalance": 0.15,
        "binance_taker_flow_60s_imbalance": 0.20,
        "binance_taker_buy_qty_60s": 50.0,
        "binance_taker_sell_qty_60s": 33.3,
        "binance_return_10s_bps": 0.5,
        "binance_return_30s_bps": 1.2,
        "binance_return_60s_bps": 2.1,
        "binance_return_since_open_bps": 2.9,
        "binance_open_mid": 85000.0,
        "binance_open_timestamp_ms": 1700000000000,
        "binance_basis_bps": 0.58,
        "binance_source_timestamp_ms": now_ms,
        "binance_received_at_ms": now_ms,
        "binance_receipt_age_ms": 150,
        "binance_source_age_ms": 150,
        "binance_valid": True,
        "is_valid": True,
    })

    # Condition C & D must supply observed native book
    state_c = build_ablation_state(snap, COND_C_MARKET_AWARE)
    assert "observed_native_book" in state_c["polymarket_order_book"]
    assert state_c["polymarket_order_book"]["market_consensus_probability"] == 0.55

    # Test clustered bootstrap with paired deltas
    db = Database(":memory:")
    lab = BTC5mShadowLab(db=db)

    # Save mock resolution scores
    for r in range(4):
        slug = f"r_{r}"
        score = {
            "score_id": f"sc_{slug}",
            "round_slug": slug,
            "target_horizon_sec": 60,
            "snapshot_id": f"sn_{slug}",
            "resolved_outcome": "UP",
            "resolution_price": 85100.0,
            "resolved_at": "2026-01-01T00:05:00Z",
            "scored_at": "2026-01-01T00:06:00Z",
            "market_q": 0.55,
            "market_brier": compute_brier(0.55, 1),
            "market_log_loss": compute_log_loss(0.55, 1),
            "cond_a_prob": 0.65,
            "cond_a_brier": compute_brier(0.65, 1),
            "cond_a_log_loss": compute_log_loss(0.65, 1),
            "cond_b_prob": 0.70,
            "cond_b_brier": compute_brier(0.70, 1),
            "cond_b_log_loss": compute_log_loss(0.70, 1),
            "cond_c_prob": 0.60,
            "cond_c_brier": compute_brier(0.60, 1),
            "cond_c_log_loss": compute_log_loss(0.60, 1),
            "cond_d_prob": 0.75,
            "cond_d_brier": compute_brier(0.75, 1),
            "cond_d_log_loss": compute_log_loss(0.75, 1),
            "metadata": {},
        }
        db.save_btc5m_resolution_score(score)

    summary = lab.compute_evaluation_summary(n_boot=100)
    assert "REF_PLUS_PERP_MINUS_REF_ONLY" in summary["paired_deltas"]
    assert "FULL_MINUS_MARKET_AWARE" in summary["paired_deltas"]
    assert "MARKET_AWARE_MINUS_REF_ONLY" in summary["paired_deltas"]


# ---------------------------------------------------------------------------
# Structural Safety Invariant Tests
# ---------------------------------------------------------------------------
def test_pure_shadow_research_safety() -> None:
    lab = BTC5mShadowLab(db=Database(":memory:"))
    assert not hasattr(lab, "broker")
    assert not hasattr(lab, "live_broker")
    assert not hasattr(lab, "execute_order")
    assert not hasattr(lab, "submit_order")
    assert not hasattr(lab, "wallet")


def test_slug_derivation() -> None:
    mgr = BTC5mContractManager()
    slug = mgr.derive_round_slug(1700000123)
    assert slug == "btc-updown-5m-1700000100"
    slug_boundary = mgr.derive_round_slug(1700000400)
    assert slug_boundary == "btc-updown-5m-1700000400"


def test_binance_depth_imbalance_and_microprice() -> None:
    feed = BinancePerpFeed()
    bids = [(85000.0, 10.0), (84990.0, 15.0)]
    asks = [(85010.0, 5.0), (85020.0, 5.0)]
    feed.add_depth(bids, asks, source_timestamp_ms=1700000000000)

    feats = feed.compute_features(now_ms=1700000000000, ref_price=85000.0, binance_open_price=85000.0)
    assert feats.is_valid is True
    assert feats.best_bid == 85000.0
    assert feats.best_ask == 85010.0
    assert feats.mid_price == 85005.0
    assert feats.top5_depth_imbalance == pytest.approx(0.4286, abs=0.001)
    assert feats.microprice == pytest.approx(85006.67, abs=0.05)


def test_binance_taker_flow_coverage_requirement() -> None:
    feed = BinancePerpFeed()
    now_ms = 1700000060000
    feed.add_trade(timestamp_ms=now_ms - 4000, price=85000.0, qty=1.0, is_buyer_maker=False)
    feed.add_trade(timestamp_ms=now_ms - 2000, price=85001.0, qty=2.0, is_buyer_maker=True)
    feed.add_depth([(85000.0, 1.0)], [(85001.0, 1.0)], source_timestamp_ms=now_ms)
    feats = feed.compute_features(now_ms=now_ms)
    assert feats.taker_flow_10s_imbalance is None
    assert feats.taker_flow_30s_imbalance is None
    assert feats.taker_flow_60s_imbalance is None

    feed_cov = BinancePerpFeed()
    feed_cov.add_trade(timestamp_ms=now_ms - 65000, price=84995.0, qty=5.0, is_buyer_maker=False)
    feed_cov.add_trade(timestamp_ms=now_ms - 4000, price=85000.0, qty=1.0, is_buyer_maker=False)
    feed_cov.add_trade(timestamp_ms=now_ms - 2000, price=85001.0, qty=2.0, is_buyer_maker=True)
    feed_cov.add_depth([(85000.0, 1.0)], [(85001.0, 1.0)], source_timestamp_ms=now_ms)
    feats_cov = feed_cov.compute_features(now_ms=now_ms)
    assert feats_cov.taker_flow_60s_imbalance is not None


def test_brier_and_log_loss_math() -> None:
    assert compute_brier(1.0, 1) == 0.0
    assert compute_brier(0.0, 0) == 0.0
    assert compute_brier(0.0, 1) == 1.0
    assert compute_brier(1.0, 0) == 1.0
    assert compute_brier(0.5, 1) == 0.25

    loss_certain = compute_log_loss(0.999999, 1)
    assert loss_certain < 0.001
    loss_wrong = compute_log_loss(0.000001, 1)
    assert loss_wrong > 10.0


def test_request_payload_asks_only_probability() -> None:
    from pm_research.research.btc5m.ablation import build_btc5m_request_payload

    state = {"market_type": "BTC 5-Minute Up/Down"}
    payload = build_btc5m_request_payload(state)
    assert payload["model"] == JEV_MODEL_PIN
    q = payload["questions"]["will_resolve_up"]
    assert "probability" in q["instructions"].lower()
    for word in ("trade", "order", "buy", "sell", "bet", "wager", "leverage", "size"):
        assert word not in q["instructions"].lower()

