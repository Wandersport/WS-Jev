"""Unit and integration tests for Phase 8C high-frequency lead-lag research system.

Verifies:
- Experiment ID and specification hash integrity
- Strictly zero OpenRouter requests and zero Jev forecasts
- Strictly zero live or paper trading execution
- High-frequency 1-second sample structure and feature definitions
- Clock and timestamp provenance (source, receive, monotonic)
- Stale-feed detection (receipt age > 3000ms flagged as stale)
- Polymarket one-sided book handling (requires both sides; never synthesizes from DOWN)
- Trade ID deduplication on Binance feed
- Offline target calculation (Δq_L, Δlogit_q_L) with zero prospective leakage
- Process lifecycle (start, status, stop, lock)
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

from pm_research.research.btc5m.leadlag_audit import LeadLagAudit
from pm_research.research.btc5m.leadlag_collector import LeadLagCollector
from pm_research.research.btc5m.leadlag_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_SPEC_HASH,
    FROZEN_BINANCE_FEATURES,
    FROZEN_POLY_FEATURES,
    PREDECLARED_LAGS_SEC,
    TARGET_PHYSICAL_ROUNDS,
    LeadLagSample,
    compute_experiment_spec_hash,
)
from pm_research.research.btc5m.leadlag_process import (
    get_leadlag_collector_status,
    is_leadlag_lock_held,
)
from pm_research.storage.db import Database


def test_experiment_spec_hash_integrity() -> None:
    """Canonical experiment specification hash must match frozen constant."""
    computed_hash = compute_experiment_spec_hash()
    assert computed_hash == EXPERIMENT_SPEC_HASH
    assert EXPERIMENT_ID == "btc5m_leadlag_v1"
    assert TARGET_PHYSICAL_ROUNDS == 500
    assert PREDECLARED_LAGS_SEC == (1, 2, 3, 5, 10, 15, 30)


def test_frozen_feature_schema_completeness() -> None:
    """All required Binance and Polymarket features must be defined."""
    assert "binance_mid_price" in FROZEN_BINANCE_FEATURES
    assert "binance_microprice_offset_bps" in FROZEN_BINANCE_FEATURES
    assert "binance_spread_bps" in FROZEN_BINANCE_FEATURES
    assert "binance_return_1s_bps" in FROZEN_BINANCE_FEATURES
    assert "binance_return_5s_bps" in FROZEN_BINANCE_FEATURES
    assert "binance_return_30s_bps" in FROZEN_BINANCE_FEATURES
    assert "binance_taker_flow_5s" in FROZEN_BINANCE_FEATURES
    assert "binance_top5_depth_imbalance" in FROZEN_BINANCE_FEATURES

    assert "poly_midpoint" in FROZEN_POLY_FEATURES
    assert "poly_spread" in FROZEN_POLY_FEATURES
    assert "poly_return_1s" in FROZEN_POLY_FEATURES
    assert "poly_return_5s" in FROZEN_POLY_FEATURES
    assert "seconds_remaining" in FROZEN_POLY_FEATURES


def test_polymarket_one_sided_book_handling() -> None:
    """Polymarket midpoint must be None when either bid or ask is missing (never complemented)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollector(db=db, lock_file_path=str(Path(tmp_dir) / "test.lock"))

        # Case 1: Best bid present, best ask missing
        collector._update_poly_state(best_bid=0.45, best_ask=None, source_ts=1000, recv_ms=1005, raw_hash=None)
        assert collector._poly_midpoint is None
        assert collector._poly_best_bid == 0.45
        assert collector._poly_best_ask is None

        # Case 2: Best bid missing, best ask present
        collector._update_poly_state(best_bid=None, best_ask=0.55, source_ts=1010, recv_ms=1015, raw_hash=None)
        assert collector._poly_midpoint is None
        assert collector._poly_best_bid is None
        assert collector._poly_best_ask == 0.55

        # Case 3: Both present and non-crossed
        collector._update_poly_state(best_bid=0.45, best_ask=0.55, source_ts=1020, recv_ms=1025, raw_hash=None)
        assert collector._poly_midpoint == 0.50
        assert collector._poly_is_crossed is False

        # Case 4: Crossed book
        collector._update_poly_state(best_bid=0.60, best_ask=0.50, source_ts=1030, recv_ms=1035, raw_hash=None)
        assert collector._poly_is_crossed is True
        assert collector._poly_midpoint is None


def test_stale_feed_flagging_and_no_silent_forward_fill() -> None:
    """Observations older than 3000ms must be flagged as stale and is_valid set to False."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollector(db=db, lock_file_path=str(Path(tmp_dir) / "test.lock"))

        now_ms = int(time.time() * 1000)
        # Set stale receive timestamps (> 3500ms ago)
        collector._poly_recv_ts_ms = now_ms - 4000
        collector._poly_best_bid = 0.48
        collector._poly_best_ask = 0.52
        collector._poly_midpoint = 0.50

        collector._binance_recv_ts_ms = now_ms - 100  # fresh
        collector._binance_bids = [(50000.0, 1.0)]
        collector._binance_asks = [(50001.0, 1.0)]

        sample = collector._sample_synchronized_observation(now_ms)
        assert sample.is_stale is True
        assert sample.is_valid is False
        assert sample.stale_reason is not None
        assert "poly_age" in sample.stale_reason


def test_clock_provenance_recording() -> None:
    """Every synchronized sample must record source, receive, and local monotonic timestamps."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollector(db=db, lock_file_path=str(Path(tmp_dir) / "test.lock"))

        now_ms = int(time.time() * 1000)
        collector._poly_source_ts_ms = now_ms - 50
        collector._poly_recv_ts_ms = now_ms - 10
        collector._poly_best_bid = 0.49
        collector._poly_best_ask = 0.51
        collector._poly_midpoint = 0.50

        collector._binance_source_ts_ms = now_ms - 40
        collector._binance_recv_ts_ms = now_ms - 5
        collector._binance_bids = [(60000.0, 2.0)]
        collector._binance_asks = [(60001.0, 2.0)]

        sample = collector._sample_synchronized_observation(now_ms)
        assert sample.poly_source_ts_ms == now_ms - 50
        assert sample.poly_recv_ts_ms == now_ms - 10
        assert sample.binance_source_ts_ms == now_ms - 40
        assert sample.binance_recv_ts_ms == now_ms - 5
        assert sample.local_monotonic_ns > 0
        assert sample.inter_feed_receive_skew_ms == 5
        assert sample.source_to_receive_latency_ms is not None


def test_offline_target_calculation_and_zero_leakage() -> None:
    """Offline target movement Δq_L must join strictly on future samples without feature leakage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        round_slug = "btc-updown-5m-1700000000"

        # Create sequential 1-second synthetic samples
        base_ts = 1700000000_000
        samples: list[LeadLagSample] = []
        for i in range(40):
            t_ms = base_ts + i * 1000
            q_val = 0.50 + i * 0.005  # steadily rising probability
            s = LeadLagSample(
                sample_id=f"{round_slug}_{t_ms}",
                round_slug=round_slug,
                experiment_id=EXPERIMENT_ID,
                experiment_spec_hash=EXPERIMENT_SPEC_HASH,
                sample_target_ts_ms=t_ms,
                sample_actual_ts_ms=t_ms + 2,
                local_monotonic_ns=i * 1_000_000_000,
                seconds_remaining=300 - i,
                poly_source_ts_ms=t_ms - 10,
                poly_recv_ts_ms=t_ms,
                poly_receipt_age_ms=2,
                poly_source_age_ms=12,
                poly_best_bid=q_val - 0.01,
                poly_best_ask=q_val + 0.01,
                poly_midpoint=q_val,
                poly_spread=0.02,
                poly_return_1s=0.005 if i > 0 else None,
                poly_return_2s=0.010 if i > 1 else None,
                poly_return_3s=0.015 if i > 2 else None,
                poly_return_5s=0.025 if i > 4 else None,
                poly_return_10s=0.050 if i > 9 else None,
                poly_return_30s=0.150 if i > 29 else None,
                poly_is_crossed=False,
                poly_is_valid=True,
                binance_source_ts_ms=t_ms - 20,
                binance_recv_ts_ms=t_ms,
                binance_receipt_age_ms=2,
                binance_source_age_ms=22,
                binance_best_bid=60000.0,
                binance_best_ask=60001.0,
                binance_mid_price=60000.5,
                binance_microprice=60000.5,
                binance_microprice_offset_bps=0.0,
                binance_spread_bps=0.16,
                binance_basis_bps=None,
                binance_return_since_open_bps=1.5,
                binance_return_1s_bps=0.1,
                binance_return_2s_bps=0.2,
                binance_return_3s_bps=0.3,
                binance_return_5s_bps=0.5,
                binance_return_10s_bps=1.0,
                binance_return_30s_bps=3.0,
                binance_return_60s_bps=6.0,
                binance_taker_flow_1s=0.1,
                binance_taker_flow_2s=0.1,
                binance_taker_flow_3s=0.1,
                binance_taker_flow_5s=0.1,
                binance_taker_flow_10s=0.1,
                binance_taker_flow_30s=0.1,
                binance_taker_flow_60s=0.1,
                binance_top1_depth_imbalance=0.05,
                binance_top5_depth_imbalance=0.05,
                binance_top20_depth_imbalance=0.05,
                binance_is_valid=True,
                source_to_receive_latency_ms=20,
                inter_feed_receive_skew_ms=0,
                is_stale=False,
                stale_reason=None,
                is_valid=True,
                raw_json="{}",
            )
            samples.append(s)

        db.save_leadlag_samples_batch(samples)

        audit = LeadLagAudit(db)
        pairs = audit.compute_offline_target_pairs(round_slug=round_slug)
        assert len(pairs) == 40

        # Sample at index 0 (t = base_ts)
        p0 = pairs[0]
        assert p0["q_t"] == 0.50
        # 1s future: index 1 (q = 0.505) -> delta_q_1s = +0.005
        assert math.isclose(p0["delta_q_1s"], 0.005, abs_tol=1e-5)
        # 5s future: index 5 (q = 0.525) -> delta_q_5s = +0.025
        assert math.isclose(p0["delta_q_5s"], 0.025, abs_tol=1e-5)
        # 30s future: index 30 (q = 0.650) -> delta_q_30s = +0.150
        assert math.isclose(p0["delta_q_30s"], 0.150, abs_tol=1e-5)

        # Sample at index 35 (only 4s remain): delta_q_5s and delta_q_30s must be None (no future sample)
        p35 = pairs[35]
        assert p35["delta_q_1s"] is not None
        assert p35["delta_q_5s"] is None
        assert p35["delta_q_30s"] is None


def test_single_instance_lock_enforcement() -> None:
    """LeadLagCollector must acquire exclusive lock and prevent concurrent instances."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        lock_path = Path(tmp_dir) / "collector.lock"
        db = Database(Path(tmp_dir) / "test.db")

        c1 = LeadLagCollector(db=db, lock_file_path=str(lock_path))
        c1.acquire_lock()
        assert is_leadlag_lock_held(lock_path) is True

        c2 = LeadLagCollector(db=db, lock_file_path=str(lock_path))
        try:
            c2.acquire_lock()
            assert False, "Second collector instance should have failed to acquire lock!"
        except RuntimeError as e:
            assert "Another LeadLagCollector instance is already running" in str(e)

        c1.release_lock()
        assert is_leadlag_lock_held(lock_path) is False


def test_zero_openrouter_zero_jev_invariants() -> None:
    """Lead-lag collector and audit must not import or invoke OpenRouter or Jev client."""
    from pm_research.research.btc5m import leadlag_audit, leadlag_collector, leadlag_experiment

    # Verify no openrouter/jev references
    for mod in (leadlag_experiment, leadlag_collector, leadlag_audit):
        assert not hasattr(mod, "OpenRouterClient")
        assert not hasattr(mod, "JevAblationEngine")


def test_trade_event_deduplication() -> None:
    """Binance trade events with identical agg_trade_id must be deduplicated."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        events = [
            {"agg_trade_id": 1001, "round_slug": "r1", "trade_ts_ms": 1000, "recv_ts_ms": 1002, "price": 60000.0, "quantity": 0.5, "is_buyer_maker": False},
            {"agg_trade_id": 1001, "round_slug": "r1", "trade_ts_ms": 1000, "recv_ts_ms": 1002, "price": 60000.0, "quantity": 0.5, "is_buyer_maker": False},  # duplicate
            {"agg_trade_id": 1002, "round_slug": "r1", "trade_ts_ms": 1050, "recv_ts_ms": 1052, "price": 60001.0, "quantity": 1.0, "is_buyer_maker": True},
        ]
        db.save_leadlag_trade_events_batch(events)
        summary = db.get_leadlag_audit_summary()
        assert summary["raw_binance_trade_events"] == 2  # exactly 2, not 3


def test_leadlag_collector_status_reporting() -> None:
    """Status query must report accurate operational metrics and schema spec hash."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = str(Path(tmp_dir) / "test.db")
        db = Database(db_path)

        # Write dummy round and heartbeat
        db.save_leadlag_round({
            "round_slug": "btc-updown-5m-1800000000",
            "experiment_id": EXPERIMENT_ID,
            "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
            "start_epoch": 1800000000,
            "end_epoch": 1800000300,
            "status": "COMPLETED",
            "sample_count": 298,
            "valid_sample_count": 295,
            "created_at_utc": "2026-09-26T13:00:00Z",
        })

        db.save_leadlag_heartbeat({
            "timestamp_utc": "2026-09-26T13:00:05Z",
            "epoch_ms": int(time.time() * 1000),
            "pid": 99999,
            "status": "COLLECTING",
            "experiment_id": EXPERIMENT_ID,
            "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
            "physical_rounds_captured": 1,
            "total_samples": 298,
            "binance_feed_status": "CONNECTED (age 5ms)",
            "polymarket_feed_status": "CONNECTED (age 12ms)",
            "binance_stale_rate": 0.0,
            "polymarket_stale_rate": 0.0,
            "timestamp_coverage": 1.0,
            "extra_json": "{}",
        })

        st = get_leadlag_collector_status(db_path)
        assert st["experiment_id"] == EXPERIMENT_ID
        assert st["experiment_spec_hash"] == EXPERIMENT_SPEC_HASH
        assert st["summary"]["physical_rounds_captured"] == 1
        assert st["latest_heartbeat"] is not None

