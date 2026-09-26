"""Concurrency and thread-safety tests for BinancePerpFeed and collector.

Phase 7.3 Operational Hardening:
1. Midpoint history concurrent mutation regression (RuntimeError: deque mutated during iteration)
2. Trade history concurrent mutation during window and feature calculations
3. Order book depth concurrent mutation and coherent point-in-time snapshots
4. Round-open dictionary thread safety under concurrent read/write
5. Round-level operational error boundary resilience
"""

from __future__ import annotations

import math
import threading
from unittest.mock import MagicMock

from pm_research.research.btc5m.binance_feed import (
    BINANCE_OPEN_TIMING_TOLERANCE_MS,
    BinancePerpFeed,
)
from pm_research.research.btc5m.collector import BTC5mAutonomousCollector
from pm_research.research.btc5m.contract import BTC5mRoundInfo
from pm_research.storage.db import Database


def test_midpoint_history_concurrent_mutation_regression() -> None:
    """Deterministic regression test for original Phase 7.3 failure:

    'RuntimeError: deque mutated during iteration' in find_boundary_open_mid.
    A writer thread continuously appends midpoint observations while reader threads
    repeatedly search for opening boundary midpoints.
    """
    feed = BinancePerpFeed()
    round_start_ms = 1_700_000_000_000

    # Pre-populate some history
    for i in range(-50, 50):
        ts = round_start_ms + i * 50
        feed.record_midpoint(source_ts=ts, recv_ts=ts + 10, mid_price=68000.0 + i)

    stop_event = threading.Event()
    exceptions: list[Exception] = []

    def writer() -> None:
        idx = 0
        while not stop_event.is_set():
            ts = round_start_ms + (idx % 200 - 100) * 20
            feed.record_midpoint(source_ts=ts, recv_ts=ts + 5, mid_price=68000.0 + (idx % 100))
            idx += 1
            if idx > 3000:
                break

    def reader() -> None:
        try:
            for _ in range(1000):
                res = feed.find_boundary_open_mid(
                    round_start_ms=round_start_ms,
                    tolerance_ms=BINANCE_OPEN_TIMING_TOLERANCE_MS,
                )
                if res is not None:
                    mid, s_ts, r_ts, offset_ms = res
                    assert mid > 0
                    assert abs(offset_ms) <= BINANCE_OPEN_TIMING_TOLERANCE_MS
                    assert r_ts > 0
        except Exception as exc:
            exceptions.append(exc)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    stop_event.set()

    assert not exceptions, f"Concurrency failure in midpoint history: {exceptions}"


def test_trade_history_concurrent_mutation() -> None:
    """Verify that continuous trade ingestion via add_trade concurrently with:

    - _compute_taker_flow_window()
    - _compute_price_return_window()
    - compute_features()
    does not raise RuntimeError, produce corrupted data, or return NaN/Inf.
    """
    feed = BinancePerpFeed(max_trade_history=2000)
    now_ms = 1_700_000_100_000

    # Seed trade history covering 120 seconds
    for i in range(120):
        t_ms = now_ms - (120 - i) * 1000
        feed.add_trade(timestamp_ms=t_ms, price=68000.0 + i * 0.1, qty=0.5, is_buyer_maker=(i % 2 == 0))

    # Add depth so compute_features has valid book
    feed.add_depth(
        bids=[(68010.0, 1.0), (68005.0, 2.0)],
        asks=[(68015.0, 1.0), (68020.0, 2.0)],
        source_timestamp_ms=now_ms,
        received_at_ms=now_ms,
    )

    stop_event = threading.Event()
    exceptions: list[Exception] = []

    def writer() -> None:
        idx = 0
        while not stop_event.is_set():
            t_ms = now_ms + idx * 5
            feed.add_trade(
                timestamp_ms=t_ms,
                price=68010.0 + (idx % 20) * 0.5,
                qty=0.1 * ((idx % 10) + 1),
                is_buyer_maker=(idx % 3 == 0),
            )
            idx += 1
            if idx > 2500:
                break

    def reader() -> None:
        try:
            for _ in range(500):
                # 1. Taker flow windows
                tf10 = feed._compute_taker_flow_window(now_ms=now_ms, window_sec=10)
                if tf10 is not None:
                    imb, b_qty, s_qty = tf10
                    assert not math.isnan(imb) and not math.isinf(imb)
                    assert -1.0 <= imb <= 1.0
                    assert b_qty >= 0.0 and s_qty >= 0.0

                tf60 = feed._compute_taker_flow_window(now_ms=now_ms, window_sec=60)
                if tf60 is not None:
                    imb, b_qty, s_qty = tf60
                    assert not math.isnan(imb) and not math.isinf(imb)
                    assert -1.0 <= imb <= 1.0

                # 2. Price return windows
                ret10 = feed._compute_price_return_window(now_ms=now_ms, current_mid=68012.5, window_sec=10)
                if ret10 is not None:
                    assert not math.isnan(ret10) and not math.isinf(ret10)

                # 3. Full feature extraction
                feats = feed.compute_features(
                    now_ms=now_ms,
                    ref_price=68012.0,
                    binance_open_price=68000.0,
                )
                assert feats.is_valid is True
                assert feats.mid_price > 0
                if feats.taker_flow_10s_imbalance is not None:
                    assert -1.0 <= feats.taker_flow_10s_imbalance <= 1.0
        except Exception as exc:
            exceptions.append(exc)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    stop_event.set()

    assert not exceptions, f"Concurrency failure in trade history: {exceptions}"


def test_depth_order_book_concurrent_mutation() -> None:
    """Verify concurrent depth updates and feature computations:

    - Coherent point-in-time snapshot (bids and asks taken together)
    - Crossed book validation is preserved under rapid concurrent book updates
    - No index errors or corrupted depth levels.
    """
    feed = BinancePerpFeed()
    now_ms = 1_700_000_000_000

    stop_event = threading.Event()
    exceptions: list[Exception] = []

    def writer() -> None:
        idx = 0
        while not stop_event.is_set():
            # Alternate between valid book and crossed book
            if idx % 10 == 0:
                # Crossed book
                feed.add_depth(
                    bids=[(68020.0, 1.0)],
                    asks=[(68010.0, 1.0)],
                    source_timestamp_ms=now_ms + idx,
                    received_at_ms=now_ms + idx,
                )
            else:
                # Valid uncrossed book
                mid = 68000.0 + (idx % 50)
                feed.add_depth(
                    bids=[(mid - 1.0, 1.0), (mid - 2.0, 2.0)],
                    asks=[(mid + 1.0, 1.0), (mid + 2.0, 2.0)],
                    source_timestamp_ms=now_ms + idx,
                    received_at_ms=now_ms + idx,
                )
            idx += 1
            if idx > 2000:
                break

    def reader() -> None:
        try:
            for i in range(500):
                feats = feed.compute_features(now_ms=now_ms + i, ref_price=68000.0)
                if feats.is_valid:
                    # Invariant: valid features MUST have uncrossed book
                    assert feats.best_bid < feats.best_ask
                    assert feats.spread_bps > 0
                    assert feats.mid_price == round((feats.best_bid + feats.best_ask) / 2.0, 2)
                    assert -1.0 <= feats.top5_depth_imbalance <= 1.0
                else:
                    # Invariant: invalid must specify rejection reason
                    assert feats.rejection_reason is not None
        except Exception as exc:
            exceptions.append(exc)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    stop_event.set()

    assert not exceptions, f"Concurrency failure in depth order book: {exceptions}"


def test_round_open_concurrent_read_write() -> None:
    """Verify thread-safe concurrent recording and querying of round opens."""
    feed = BinancePerpFeed()
    stop_event = threading.Event()
    exceptions: list[Exception] = []

    def writer() -> None:
        idx = 0
        while not stop_event.is_set():
            slug = f"round_{idx % 20}"
            feed.record_round_open(
                round_slug=slug,
                mid_price=68000.0 + idx,
                timestamp_ms=1000 + idx,
                source_timestamp_ms=990 + idx,
                offset_ms=10,
            )
            idx += 1
            if idx > 2000:
                break

    def reader() -> None:
        try:
            for i in range(1000):
                slug = f"round_{i % 20}"
                rec = feed.get_round_open(slug)
                if rec is not None:
                    mid, recv_ts, s_ts, offset_ms = rec
                    assert mid >= 68000.0
                    assert recv_ts >= 1000
                    assert offset_ms == 10
        except Exception as exc:
            exceptions.append(exc)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=10.0)

    stop_event.set()

    assert not exceptions, f"Concurrency failure in round opens: {exceptions}"


def test_round_level_operational_error_resilience(tmp_path: any) -> None:
    """Verify that an unexpected round-level operational exception is caught and logged

    allowing the collector loop to cleanly continue to subsequent rounds, while fatal
    invariants (such as spec hash tampering or database corruption) immediately raise.
    """
    import pytest

    db = Database(db_path=tmp_path / "test_resilience.db")
    collector = BTC5mAutonomousCollector(
        target_valid_rounds=10,
        cost_ceiling_usd=10.0,
        db=db,
        stop_file_path=tmp_path / "collector.stop",
    )

    round_info = MagicMock(spec=BTC5mRoundInfo)
    round_info.round_slug = "btc-updown-5m-1700000000"
    round_info.slug = "btc-updown-5m-1700000000"
    round_info.start_epoch = 1700000000
    round_info.end_epoch = 1700000300
    round_info.duration_sec = 300
    round_info.seconds_remaining = 240.0
    round_info.price_to_beat = 68000.0
    round_info.price_to_beat_source = "chainlink"
    round_info.up_token_id = "up_tok"
    round_info.down_token_id = "down_tok"
    round_info.up_outcome_index = 0
    round_info.down_outcome_index = 1
    round_info.condition_id = "cond_1"
    round_info.question = "Will BTC be above 68000?"

    # 1. Operational error in horizon processing: should be caught, not crash collector
    collector._process_round_horizons = MagicMock(side_effect=RuntimeError("Simulated transient socket error"))
    collector.lab.poll_round_resolution = MagicMock(return_value=None)
    collector.is_stop_requested = MagicMock(side_effect=[False, True])  # stop after 1 round
    collector.lab.discover_active_round = MagicMock(return_value=round_info)
    collector.lab.contract_mgr.discover_active_round = MagicMock(return_value=round_info)

    # run() should complete gracefully despite the round error
    collector.run()

    # Heartbeat was emitted noting operational failure
    with db._get_connection() as conn:
        rows = conn.execute("SELECT notes FROM btc5m_collector_heartbeat").fetchall()
        notes = [r[0] for r in rows if r[0]]
    assert any("operational failure" in n.lower() for n in notes), f"Notes recorded: {notes}"

    # 2. Fatal invariant error: spec hash mismatch MUST raise
    collector._process_round_horizons = MagicMock(side_effect=ValueError("FATAL: experiment_spec_hash mismatch"))
    collector.is_stop_requested = MagicMock(return_value=False)
    with pytest.raises(ValueError, match="experiment_spec_hash"):
        collector.run()
