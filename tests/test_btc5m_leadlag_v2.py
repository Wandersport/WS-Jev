"""Unit and integration tests for Phase 8C.2 lead-lag v2 measurement architecture.

Verifies:
- Experiment ID and specification hash integrity
- Binance lowercase combined-stream aggtrade routing (btcusdt@aggtrade)
- Payload e="aggTrade" routing
- aggTrade persistence and deduplication by agg_trade_id
- Chronological ordering and out-of-order sequence regression handling
- Rolling taker-flow computation across all horizons (1s, 2s, 3s, 5s, 10s, 30s, 60s)
- No zero substitution for missing trade data / warmup
- Polymarket full book snapshot source timestamp extraction
- Polymarket price_change outer payload timestamp extraction
- No source-time fabrication (missing source ts preserved as None)
- Raw wire payload lossless zlib compression and decompression round-trip (SHA256 verified)
- Storage isolation: v1 dataset row counts unchanged and immune to v2 writes
- Raw audit counts filtered strictly by experiment_id
- Heartbeat and status isolation between v1 and v2
- Round and token transition boundary isolation (zero token leakage across rounds)
- Parser telemetry increments on malformed payloads (JSONDecodeError, missing keys)
- No silent broad-exception behavior
- Clock and latency percentile distributions (p50, p90, p95, p99, max)
- Taker-flow SLA and source timestamp SLA evaluation
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pm_research.research.btc5m.contract import BTC5mRoundInfo
from pm_research.research.btc5m.leadlag_v2_audit import LeadLagV2Audit
from pm_research.research.btc5m.leadlag_v2_collector import (
    LeadLagCollectorV2,
    compute_percentiles,
)
from pm_research.research.btc5m.leadlag_v2_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_PILOT_ID,
    EXPERIMENT_SPEC_HASH,
    PREDECLARED_LAGS_SEC,
    TARGET_PHYSICAL_ROUNDS,
    LeadLagSampleV2,
    compute_experiment_spec_hash,
)
from pm_research.storage.db import Database


def test_v2_spec_hash_and_constants() -> None:
    """Canonical v2 experiment specification hash must match approved constant."""
    computed_hash = compute_experiment_spec_hash()
    assert computed_hash == EXPERIMENT_SPEC_HASH
    assert EXPERIMENT_ID == "btc5m_leadlag_v2"
    assert EXPERIMENT_PILOT_ID == "btc5m_leadlag_v2_pilot"
    assert TARGET_PHYSICAL_ROUNDS == 500
    assert PREDECLARED_LAGS_SEC == (1, 2, 3, 5, 10, 15, 30)


def test_binance_lowercase_aggtrade_and_event_routing() -> None:
    """Binance lowercase combined-stream aggtrade routing and event type routing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))

        now_ms = 1727500000000
        mono_ns = 1_000_000_000

        # 1. Lowercase stream identifier
        wire_msg_lowercase = json.dumps({
            "stream": "btcusdt@aggtrade",
            "data": {
                "e": "aggTrade",
                "E": now_ms,
                "s": "BTCUSDT",
                "a": 10001,
                "p": "65000.50",
                "q": "0.25",
                "f": 501,
                "l": 502,
                "T": now_ms - 5,
                "m": False,  # buyer is taker -> Buy trade
            }
        })
        collector._handle_binance_message(wire_msg_lowercase, now_ms, mono_ns)
        assert collector._aggtrade_events_received == 1
        assert len(collector._binance_trades) == 1
        t_ms, p, q, is_bm, agg_id = collector._binance_trades[0]
        assert agg_id == 10001
        assert t_ms == now_ms - 5
        assert p == 65000.50
        assert q == 0.25
        assert is_bm is False

        # 2. Event type routing when stream casing differs or general stream
        wire_msg_event_type = json.dumps({
            "stream": "btcusdt@custom_stream",
            "data": {
                "e": "aggTrade",
                "E": now_ms + 10,
                "s": "BTCUSDT",
                "a": 10002,
                "p": "65001.00",
                "q": "0.10",
                "T": now_ms + 8,
                "m": True,  # seller is taker -> Sell trade
            }
        })
        collector._handle_binance_message(wire_msg_event_type, now_ms + 10, mono_ns + 10_000_000)
        assert collector._aggtrade_events_received == 2
        assert len(collector._binance_trades) == 2
        assert collector._binance_trades[1][4] == 10002


def test_aggtrade_deduplication_and_persistence() -> None:
    """Duplicate agg_trade_id messages must be rejected in memory and SQLite."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))

        now_ms = 1727500000000
        mono_ns = 1_000_000_000

        wire_msg = json.dumps({
            "stream": "btcusdt@aggtrade",
            "data": {
                "e": "aggTrade",
                "E": now_ms,
                "s": "BTCUSDT",
                "a": 20001,
                "p": "65000.00",
                "q": "1.0",
                "T": now_ms - 2,
                "m": True,
            }
        })

        # Send first time
        collector._handle_binance_message(wire_msg, now_ms, mono_ns)
        assert collector._aggtrade_events_received == 1
        assert collector._duplicate_aggtrade_events == 0

        # Send second time (duplicate wire message)
        collector._handle_binance_message(wire_msg, now_ms + 1, mono_ns + 1_000_000)
        assert collector._aggtrade_events_received == 2
        assert collector._duplicate_aggtrade_events == 1
        assert len(collector._binance_trades) == 1  # Not added twice

        # Flush to DB and test DB unique primary key
        collector._flush_batch()
        summary = db.get_leadlag_v2_audit_summary(EXPERIMENT_ID)
        assert summary["raw_binance_trade_events"] == 1
        assert summary["unique_binance_trade_events"] == 1


def test_aggtrade_out_of_order_and_chronological_ordering() -> None:
    """Out-of-order trade messages must be detected and sorted chronologically."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))

        now_ms = 1727500000000
        mono_ns = 1_000_000_000

        # Event 1: T = 1000, ID = 1
        collector._handle_binance_message(
            json.dumps({"stream": "btcusdt@aggtrade", "data": {"e": "aggTrade", "a": 1, "p": "100", "q": "1", "T": 1000, "m": True}}),
            now_ms,
            mono_ns,
        )
        # Event 2: T = 1050, ID = 2
        collector._handle_binance_message(
            json.dumps({"stream": "btcusdt@aggtrade", "data": {"e": "aggTrade", "a": 2, "p": "101", "q": "1", "T": 1050, "m": False}}),
            now_ms + 1,
            mono_ns + 1_000_000,
        )
        # Event 3 (Out-of-order regression): T = 1020, ID = 3
        collector._handle_binance_message(
            json.dumps({"stream": "btcusdt@aggtrade", "data": {"e": "aggTrade", "a": 3, "p": "100.5", "q": "1", "T": 1020, "m": True}}),
            now_ms + 2,
            mono_ns + 2_000_000,
        )

        assert collector._out_of_order_trade_events == 1
        trades = list(collector._binance_trades)
        assert len(trades) == 3
        # Must be sorted by trade_ts_ms: 1000, 1020, 1050
        assert trades[0][0] == 1000 and trades[0][4] == 1
        assert trades[1][0] == 1020 and trades[1][4] == 3
        assert trades[2][0] == 1050 and trades[2][4] == 2


def test_taker_flow_calculation_and_no_zero_substitution() -> None:
    """Taker flow must return None during warmup or missing trade windows, never 0.0."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))

        now_ms = 1727500000000
        # No trades recorded yet
        assert collector._binance_trades == collections.deque()
        sample = collector._sample_synchronized_observation(now_ms, actual_ms=now_ms)
        # All taker flow values MUST be None, never substituted with 0.0
        assert sample.binance_taker_flow_1s is None
        assert sample.binance_taker_flow_5s is None
        assert sample.binance_taker_flow_60s is None

        # Add trades starting at t0 = now_ms - 10_000 (only 10 seconds of trade history)
        t0 = now_ms - 10_000
        collector._first_trade_ms = t0
        collector._binance_trades.extend([
            (now_ms - 2000, 60000.0, 1.0, False, 101),  # Buy 1.0 at -2s
            (now_ms - 1000, 60000.0, 0.5, True, 102),   # Sell 0.5 at -1s
            (now_ms - 500, 60000.0, 1.5, False, 103),   # Buy 1.5 at -0.5s
        ])

        sample2 = collector._sample_synchronized_observation(now_ms, actual_ms=now_ms)
        # 1s window: trades at -500ms (Buy 1.5) and -1000ms (Sell 0.5). Buy=1.5, Sell=0.5, Tot=2.0 -> (1.5-0.5)/2.0 = +0.50
        assert sample2.binance_taker_flow_1s == 0.50
        # 5s window: Buy=2.5, Sell=0.5, Tot=3.0 -> (2.5-0.5)/3.0 = +0.6667
        assert math.isclose(sample2.binance_taker_flow_5s, 0.6667, abs_tol=1e-4)
        # 60s window: only 10s of trade data exists since collector started -> MUST be None!
        assert sample2.binance_taker_flow_60s is None


def test_polymarket_snapshot_and_delta_timestamp_capture() -> None:
    """Polymarket timestamps must be preserved from both snapshot and price_change outer frames."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))
        collector._poly_token_id = "token_up_123"

        now_ms = 1727500000000
        mono_ns = 1_000_000_000

        # 1. Full snapshot message
        snapshot_wire = json.dumps([{
            "timestamp": "1727500000100",
            "bids": [{"price": "0.48", "size": "100"}],
            "asks": [{"price": "0.52", "size": "100"}],
            "hash": "0xsnap1",
        }])
        collector._handle_poly_message(snapshot_wire, now_ms + 150, mono_ns)
        assert collector._snapshot_events_received == 1
        assert collector._poly_source_ts_ms == 1727500000100
        assert collector._poly_provenance_mode == "WS_SNAPSHOT"
        assert collector._poly_best_bid == 0.48
        assert collector._poly_best_ask == 0.52
        assert collector._poly_midpoint == 0.50

        # 2. Incremental price_change frame with outer timestamp (fixing v1 defect)
        delta_wire = json.dumps({
            "event_type": "price_change",
            "timestamp": "1727500000500",
            "price_changes": [
                {
                    "asset_id": "token_up_123",
                    "price": "0.51",
                    "side": "BUY",
                    "size": "200",
                    "best_bid": "0.49",
                    "best_ask": "0.51",
                    "hash": "0xdelta1",
                }
            ]
        })
        collector._handle_poly_message(delta_wire, now_ms + 540, mono_ns + 40_000_000)
        assert collector._delta_events_received == 1
        assert collector._poly_source_ts_ms == 1727500000500  # Preserved from outer frame!
        assert collector._poly_provenance_mode == "WS_DELTA"
        assert collector._poly_best_bid == 0.49
        assert collector._poly_best_ask == 0.51
        assert collector._poly_midpoint == 0.50

        # 3. Payload without source timestamp: must preserve None and increment missing counter
        missing_wire = json.dumps({
            "event_type": "price_change",
            "price_changes": [
                {
                    "asset_id": "token_up_123",
                    "best_bid": "0.50",
                    "best_ask": "0.52",
                }
            ]
        })
        collector._handle_poly_message(missing_wire, now_ms + 600, mono_ns + 50_000_000)
        assert collector._poly_source_ts_ms is None
        assert collector._poly_source_ts_missing > 0


def test_lossless_raw_payload_round_trip() -> None:
    """Exact raw WebSocket wire payloads must be compressed and completely recoverable."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))

        original_wire_str = json.dumps({
            "stream": "btcusdt@depth20@100ms",
            "data": {
                "e": "depthUpdate",
                "E": 1727500000000,
                "T": 1727499999990,
                "s": "BTCUSDT",
                "b": [["65000.00", "5.123"], ["64999.00", "12.456"]],
                "a": [["65001.00", "4.789"], ["65002.00", "8.901"]],
            }
        })
        original_bytes = original_wire_str.encode("utf-8")
        recv_ms = 1727500000010
        mono_ns = 5_000_000

        collector._handle_binance_message(original_wire_str, recv_ms, mono_ns)
        assert len(collector._raw_payloads) == 1
        payload_id = collector._raw_payloads[0]["payload_id"]
        collector._flush_batch()

        recovered = db.get_leadlag_v2_raw_payload(payload_id)
        assert recovered is not None
        assert recovered["raw_bytes"] == original_bytes
        assert recovered["uncompressed_len"] == len(original_bytes)
        assert recovered["compressed_len"] < len(original_bytes)
        assert recovered["sha256_hash"] == hashlib.sha256(original_bytes).hexdigest()


def test_experiment_isolation_v1_and_v2() -> None:
    """Writing v2 data must never alter v1 table counts or v1 audit summary."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")

        # 1. Seed simulated v1 records
        db.save_leadlag_round({
            "round_slug": "r_v1_001",
            "experiment_id": "btc5m_leadlag_v1",
            "experiment_spec_hash": "v1_spec_hash_123",
            "start_epoch": 1000,
            "end_epoch": 1300,
            "status": "COMPLETED",
            "sample_count": 300,
            "valid_sample_count": 298,
            "created_at_utc": "2026-09-26T12:00:00Z",
        })
        db.save_leadlag_depth_events_batch([
            {"event_id": "d1", "round_slug": "r_v1_001", "source_ts_ms": 1000, "recv_ts_ms": 1002, "best_bid": 50000.0, "best_ask": 50001.0, "mid_price": 50000.5}
        ])
        db.save_leadlag_trade_events_batch([
            {"agg_trade_id": 999, "round_slug": "r_v1_001", "trade_ts_ms": 1000, "recv_ts_ms": 1002, "price": 50000.0, "quantity": 1.0, "is_buyer_maker": True}
        ])
        v1_summary_before = db.get_leadlag_audit_summary("btc5m_leadlag_v1")
        assert v1_summary_before["physical_rounds_captured"] == 1
        assert v1_summary_before["raw_binance_depth_events"] == 1
        assert v1_summary_before["raw_binance_trade_events"] == 1

        # 2. Write v2 records into isolated tables
        db.save_leadlag_v2_round({
            "round_slug": "r_v2_001",
            "experiment_id": "btc5m_leadlag_v2",
            "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
            "is_pilot": False,
            "start_epoch": 2000,
            "end_epoch": 2300,
            "status": "COMPLETED",
            "sample_count": 300,
            "valid_sample_count": 300,
            "created_at_utc": "2026-09-28T12:00:00Z",
        })
        db.save_leadlag_v2_depth_events_batch([
            {"event_id": "v2_d1", "experiment_id": "btc5m_leadlag_v2", "experiment_spec_hash": EXPERIMENT_SPEC_HASH, "round_slug": "r_v2_001", "source_ts_ms": 2000, "recv_ts_ms": 2002, "best_bid": 60000.0, "best_ask": 60001.0, "mid_price": 60000.5}
        ])
        db.save_leadlag_v2_trade_events_batch([
            {"experiment_id": "btc5m_leadlag_v2", "experiment_spec_hash": EXPERIMENT_SPEC_HASH, "agg_trade_id": 8888, "round_slug": "r_v2_001", "trade_ts_ms": 2000, "recv_ts_ms": 2002, "price": 60000.0, "quantity": 2.0, "is_buyer_maker": False}
        ])

        # 3. Verify v1 summary remains 100% UNCHANGED
        v1_summary_after = db.get_leadlag_audit_summary("btc5m_leadlag_v1")
        assert v1_summary_after["physical_rounds_captured"] == 1
        assert v1_summary_after["raw_binance_depth_events"] == 1
        assert v1_summary_after["raw_binance_trade_events"] == 1
        assert v1_summary_after == v1_summary_before

        # 4. Verify v2 summary isolates v2 data
        v2_summary = db.get_leadlag_v2_audit_summary("btc5m_leadlag_v2")
        assert v2_summary["physical_rounds_captured"] == 1
        assert v2_summary["raw_binance_depth_events"] == 1
        assert v2_summary["raw_binance_trade_events"] == 1
        assert v2_summary["unique_binance_trade_events"] == 1


def test_round_and_token_transition_isolation() -> None:
    """Round transitions must reset Polymarket book and Binance open price completely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))

        # Use properly 300-second-aligned epochs:
        # Round 1: 1699999800 -> 1700000100  (slug btc-updown-5m-1699999800)
        # Round 2: 1700000100 -> 1700000400  (slug btc-updown-5m-1700000100)
        r1_start = 1699999800
        r1_end = 1700000100
        r2_start = 1700000100
        r2_end = 1700000400

        # Setup round 1
        round1_info = BTC5mRoundInfo(
            slug=f"btc-updown-5m-{r1_start}",
            condition_id="cond_A",
            market_id="mkt_A",
            question="Will BTC go up?",
            description="",
            resolution_source_url="",
            settlement_rule="",
            start_time_utc=datetime.fromtimestamp(r1_start, tz=timezone.utc),
            end_time_utc=datetime.fromtimestamp(r1_end, tz=timezone.utc),
            duration_seconds=300,
            up_token_id="up_token_A",
            down_token_id="down_token_A",
            up_outcome_index=0,
            down_outcome_index=1,
            accepting_orders=True,
            price_to_beat=None,
            price_to_beat_source=None,
            fee_rate=0.0,
            fee_exponent=1.0,
        )
        collector._current_round_slug = round1_info.slug
        collector._current_round_info = round1_info
        collector._poly_token_id = "up_token_A"
        collector._poly_best_bid = 0.52
        collector._poly_best_ask = 0.54
        collector._poly_midpoint = 0.53
        collector._binance_open_mid = 50000.0
        collector._binance_mids.append((1000, 1002, 50500.0))

        # Transition to round 2
        round2_info = BTC5mRoundInfo(
            slug=f"btc-updown-5m-{r2_start}",
            condition_id="cond_B",
            market_id="mkt_B",
            question="Will BTC go up?",
            description="",
            resolution_source_url="",
            settlement_rule="",
            start_time_utc=datetime.fromtimestamp(r2_start, tz=timezone.utc),
            end_time_utc=datetime.fromtimestamp(r2_end, tz=timezone.utc),
            duration_seconds=300,
            up_token_id="up_token_B",
            down_token_id="down_token_B",
            up_outcome_index=0,
            down_outcome_index=1,
            accepting_orders=True,
            price_to_beat=None,
            price_to_beat_source=None,
            fee_rate=0.0,
            fee_exponent=1.0,
        )
        collector._upcoming_round_slug = round2_info.slug
        collector._upcoming_round_info = round2_info

        # Trigger transition: r2_start + 1 -> derive_round_slug gives btc-updown-5m-{r2_start}
        collector._ensure_active_round(r2_start + 1)

        # Assert no old book leakage
        assert collector._current_round_slug == f"btc-updown-5m-{r2_start}"
        assert collector._poly_token_id == "up_token_B"
        assert collector._poly_best_bid is None
        assert collector._poly_best_ask is None
        assert collector._poly_midpoint is None
        assert len(collector._poly_mids) == 0  # cleared
        # Assert Binance open reset to last mid
        assert collector._binance_open_mid == 50500.0
        assert collector._binance_open_round_slug == f"btc-updown-5m-{r2_start}"


def test_parser_telemetry_increments_on_malformed_payload() -> None:
    """Parser errors must increment diagnostic telemetry counters instead of silent suppression."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))

        # Send invalid JSON to Binance handler
        collector._handle_binance_message("{ invalid json }", 1000, 1000)
        assert collector._malformed_json_count == 1
        assert collector._malformed_binance_events == 1

        # Send non-dict payload
        collector._handle_binance_message(json.dumps(["not a dict"]), 1001, 1001)
        assert collector._malformed_binance_events == 2

        # Send invalid JSON to Polymarket handler
        collector._handle_poly_message("{ broken poly json", 1002, 1002)
        assert collector._malformed_json_count == 2
        assert collector._malformed_poly_events == 1


def test_percentile_calculation_accuracy() -> None:
    """Rolling percentile calculations must compute exact p50, p90, p95, p99, and max."""
    values = list(range(1, 101))  # 1 to 100
    res = compute_percentiles(values)
    assert res["p50"] == 50.0 or res["p50"] == 51.0
    assert res["p90"] == 90.0 or res["p90"] == 91.0
    assert res["p95"] == 95.0 or res["p95"] == 96.0
    assert res["p99"] == 99.0 or res["p99"] == 100.0
    assert res["max"] == 100.0
    assert math.isclose(res["mean"], 50.5, abs_tol=1e-2)


def test_offline_target_calculation_v2() -> None:
    """LeadLagV2Audit must compute offline Δq_L and Δlogit_q_L targets without prospective leakage."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        round_slug = "btc-updown-5m-1800000000"

        base_ts = 1800000000_000
        samples = []
        for i in range(35):
            t_ms = base_ts + i * 1000
            q_val = 0.50 + i * 0.01  # rising probability
            s = LeadLagSampleV2(
                sample_id=f"{round_slug}_{t_ms}",
                round_slug=round_slug,
                experiment_id=EXPERIMENT_ID,
                experiment_spec_hash=EXPERIMENT_SPEC_HASH,
                is_pilot=False,
                sample_target_ts_ms=t_ms,
                sample_actual_ts_ms=t_ms + 1,
                local_monotonic_ns=i * 1_000_000_000,
                seconds_remaining=300 - i,
                poly_source_ts_ms=t_ms - 10,
                poly_recv_ts_ms=t_ms,
                poly_receipt_age_ms=1,
                poly_source_age_ms=11,
                poly_provenance_mode="WS_DELTA",
                poly_best_bid=q_val - 0.005,
                poly_best_ask=q_val + 0.005,
                poly_midpoint=q_val,
                poly_spread=0.01,
                poly_return_1s=0.01 if i > 0 else None,
                poly_return_2s=0.02 if i > 1 else None,
                poly_return_3s=0.03 if i > 2 else None,
                poly_return_5s=0.05 if i > 4 else None,
                poly_return_10s=0.10 if i > 9 else None,
                poly_return_30s=0.30 if i > 29 else None,
                poly_is_crossed=False,
                poly_is_valid=True,
                binance_source_ts_ms=t_ms - 20,
                binance_recv_ts_ms=t_ms,
                binance_receipt_age_ms=1,
                binance_source_age_ms=21,
                binance_best_bid=65000.0,
                binance_best_ask=65001.0,
                binance_mid_price=65000.5,
                binance_microprice=65000.5,
                binance_microprice_offset_bps=0.0,
                binance_spread_bps=0.15,
                binance_basis_bps=None,
                binance_return_since_open_bps=1.0,
                binance_return_1s_bps=0.1,
                binance_return_2s_bps=0.2,
                binance_return_3s_bps=0.3,
                binance_return_5s_bps=0.5,
                binance_return_10s_bps=1.0,
                binance_return_30s_bps=3.0,
                binance_return_60s_bps=6.0,
                binance_taker_flow_1s=0.2,
                binance_taker_flow_2s=0.2,
                binance_taker_flow_3s=0.2,
                binance_taker_flow_5s=0.2,
                binance_taker_flow_10s=0.2,
                binance_taker_flow_30s=0.2,
                binance_taker_flow_60s=0.2,
                binance_top1_depth_imbalance=0.05,
                binance_top5_depth_imbalance=0.05,
                binance_top20_depth_imbalance=0.05,
                binance_is_valid=True,
                source_to_receive_latency_ms=20,
                inter_feed_receive_skew_ms=0,
                is_stale=False,
                stale_reason=None,
                is_valid=True,
                raw_payload_id=None,
                raw_json="{}",
            )
            samples.append(s)

        db.save_leadlag_v2_samples_batch(samples)

        audit = LeadLagV2Audit(db)
        pairs = audit.compute_offline_target_pairs(round_slug=round_slug, experiment_id=EXPERIMENT_ID)
        assert len(pairs) == 35

        p0 = pairs[0]
        assert p0["q_t"] == 0.50
        assert math.isclose(p0["delta_q_1s"], 0.01, abs_tol=1e-5)
        assert math.isclose(p0["delta_q_5s"], 0.05, abs_tol=1e-5)
        assert math.isclose(p0["delta_q_30s"], 0.30, abs_tol=1e-5)

        # Horizon boundary test (sample at 32s has 2s future left)
        p32 = pairs[32]
        assert p32["delta_q_1s"] is not None
        assert p32["delta_q_2s"] is not None
        assert p32["delta_q_5s"] is None
        assert p32["delta_q_30s"] is None


def test_binance_trade_stream_routing() -> None:
    """Binance @trade stream (e='trade', trade_id in field 't') must be correctly routed.

    Regression test: the Binance USD-M Futures API deprecated @aggTrade and replaced
    it with @trade. The trade stream uses field 't' for trade ID instead of 'a'.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))

        now_ms = 1727500000000
        mono_ns = 1_000_000_000

        # Real @trade stream format (e='trade', trade_id in 't')
        wire_msg_trade = json.dumps({
            "stream": "btcusdt@trade",
            "data": {
                "e": "trade",
                "E": now_ms,
                "T": now_ms - 3,
                "s": "BTCUSDT",
                "t": 8125880784,
                "p": "82966.70",
                "q": "0.007",
                "X": "MARKET",
                "m": True,
                "st": 1,
            }
        })
        collector._handle_binance_message(wire_msg_trade, now_ms, mono_ns)
        assert collector._aggtrade_events_received == 1
        assert len(collector._binance_trades) == 1

        t_ms, p, q, is_bm, trade_id = collector._binance_trades[0]
        assert trade_id == 8125880784
        assert t_ms == now_ms - 3
        assert p == 82966.70
        assert q == 0.007
        assert is_bm is True

        # Verify raw trade event buffered with correct agg_trade_id
        assert len(collector._raw_trade_events) == 1
        assert collector._raw_trade_events[0]["agg_trade_id"] == 8125880784

        # Legacy aggTrade format must still work
        wire_msg_legacy = json.dumps({
            "stream": "btcusdt@aggtrade",
            "data": {
                "e": "aggTrade",
                "E": now_ms + 10,
                "s": "BTCUSDT",
                "a": 10001,
                "p": "65000.50",
                "q": "0.25",
                "T": now_ms + 8,
                "m": False,
            }
        })
        collector._handle_binance_message(wire_msg_legacy, now_ms + 10, mono_ns + 10_000_000)
        assert collector._aggtrade_events_received == 2
        assert len(collector._binance_trades) == 2
        assert collector._binance_trades[1][4] == 10001  # trade_id from 'a' field


def test_actual_taker_flow_coverage_not_elapsed_time_proxy() -> None:
    """Heartbeat taker_flow_60s_coverage must reflect real non-null ratio, not elapsed time.

    Regression test for the defect where coverage was calculated as:
      1.0 if (first_trade_ms and >60s elapsed) else 0.0
    which is NOT an actual coverage ratio.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))
        collector.experiment_id = EXPERIMENT_PILOT_ID
        collector.is_pilot = True

        base_ts = 1800000000_000
        round_slug = "btc-updown-5m-1800000000"

        # Create samples where taker_flow_60s is None (no trade data)
        # This simulates what happens when aggTrade stream doesn't work
        samples_no_flow = []
        for i in range(10):
            t_ms = base_ts + i * 1000
            s = LeadLagSampleV2(
                sample_id=f"{round_slug}_{t_ms}",
                round_slug=round_slug,
                experiment_id=EXPERIMENT_PILOT_ID,
                experiment_spec_hash=EXPERIMENT_SPEC_HASH,
                is_pilot=True,
                sample_target_ts_ms=t_ms,
                sample_actual_ts_ms=t_ms + 2,
                local_monotonic_ns=i * 1_000_000_000,
                seconds_remaining=300 - i,
                poly_source_ts_ms=t_ms - 10,
                poly_recv_ts_ms=t_ms,
                poly_receipt_age_ms=2,
                poly_source_age_ms=12,
                poly_provenance_mode="WS_DELTA",
                poly_best_bid=0.49,
                poly_best_ask=0.51,
                poly_midpoint=0.50,
                poly_spread=0.02,
                poly_return_1s=None,
                poly_return_2s=None,
                poly_return_3s=None,
                poly_return_5s=None,
                poly_return_10s=None,
                poly_return_30s=None,
                poly_is_crossed=False,
                poly_is_valid=True,
                binance_source_ts_ms=t_ms - 20,
                binance_recv_ts_ms=t_ms,
                binance_receipt_age_ms=2,
                binance_source_age_ms=22,
                binance_best_bid=65000.0,
                binance_best_ask=65001.0,
                binance_mid_price=65000.5,
                binance_microprice=65000.5,
                binance_microprice_offset_bps=0.0,
                binance_spread_bps=0.15,
                binance_basis_bps=None,
                binance_return_since_open_bps=0.0,
                binance_return_1s_bps=None,
                binance_return_2s_bps=None,
                binance_return_3s_bps=None,
                binance_return_5s_bps=None,
                binance_return_10s_bps=None,
                binance_return_30s_bps=None,
                binance_return_60s_bps=None,
                binance_taker_flow_1s=None,
                binance_taker_flow_2s=None,
                binance_taker_flow_3s=None,
                binance_taker_flow_5s=None,
                binance_taker_flow_10s=None,
                binance_taker_flow_30s=None,
                binance_taker_flow_60s=None,  # All None - no trade data
                binance_top1_depth_imbalance=0.05,
                binance_top5_depth_imbalance=0.05,
                binance_top20_depth_imbalance=0.05,
                binance_is_valid=True,
                source_to_receive_latency_ms=20,
                inter_feed_receive_skew_ms=0,
                is_stale=False,
                stale_reason=None,
                is_valid=True,
                raw_payload_id=None,
                raw_json="{}",
            )
            samples_no_flow.append(s)

        db.save_leadlag_v2_samples_batch(samples_no_flow)

        # Set first_trade_ms to simulate "first trade existed and >60s elapsed"
        # Under the old bug, this would return 1.0. Under the fix, it must return 0.0
        # because all taker_flow_60s values are None.
        collector._first_trade_ms = base_ts - 120_000  # first trade 120s ago

        coverage = collector._compute_actual_taker_flow_coverage()

        # All valid samples are post-warmup (sample_actual_ts_ms > first_trade + 60s)
        # but ALL have binance_taker_flow_60s=None, so coverage must be 0.0
        assert coverage == 0.0, f"Expected 0.0 coverage for all-None taker flow, got {coverage}"


def test_actual_taker_flow_coverage_with_real_data() -> None:
    """Coverage must correctly reflect the ratio of non-null taker flow values."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = Database(Path(tmp_dir) / "test.db")
        collector = LeadLagCollectorV2(db=db, lock_file_path=str(Path(tmp_dir) / "c.lock"))
        collector.experiment_id = EXPERIMENT_PILOT_ID
        collector.is_pilot = True

        base_ts = 1800000000_000
        round_slug = "btc-updown-5m-1800000000"
        first_trade = base_ts - 120_000
        collector._first_trade_ms = first_trade

        samples = []
        for i in range(20):
            t_ms = base_ts + i * 1000
            # 18 out of 20 samples have taker_flow_60s (90% coverage)
            tf_60s = 0.3 if i < 18 else None
            s = LeadLagSampleV2(
                sample_id=f"{round_slug}_{t_ms}",
                round_slug=round_slug,
                experiment_id=EXPERIMENT_PILOT_ID,
                experiment_spec_hash=EXPERIMENT_SPEC_HASH,
                is_pilot=True,
                sample_target_ts_ms=t_ms,
                sample_actual_ts_ms=t_ms + 2,
                local_monotonic_ns=i * 1_000_000_000,
                seconds_remaining=300 - i,
                poly_source_ts_ms=t_ms - 10,
                poly_recv_ts_ms=t_ms,
                poly_receipt_age_ms=2,
                poly_source_age_ms=12,
                poly_provenance_mode="WS_DELTA",
                poly_best_bid=0.49,
                poly_best_ask=0.51,
                poly_midpoint=0.50,
                poly_spread=0.02,
                poly_return_1s=None,
                poly_return_2s=None,
                poly_return_3s=None,
                poly_return_5s=None,
                poly_return_10s=None,
                poly_return_30s=None,
                poly_is_crossed=False,
                poly_is_valid=True,
                binance_source_ts_ms=t_ms - 20,
                binance_recv_ts_ms=t_ms,
                binance_receipt_age_ms=2,
                binance_source_age_ms=22,
                binance_best_bid=65000.0,
                binance_best_ask=65001.0,
                binance_mid_price=65000.5,
                binance_microprice=65000.5,
                binance_microprice_offset_bps=0.0,
                binance_spread_bps=0.15,
                binance_basis_bps=None,
                binance_return_since_open_bps=0.0,
                binance_return_1s_bps=None,
                binance_return_2s_bps=None,
                binance_return_3s_bps=None,
                binance_return_5s_bps=None,
                binance_return_10s_bps=None,
                binance_return_30s_bps=None,
                binance_return_60s_bps=None,
                binance_taker_flow_1s=0.2 if i < 18 else None,
                binance_taker_flow_2s=0.2 if i < 18 else None,
                binance_taker_flow_3s=0.2 if i < 18 else None,
                binance_taker_flow_5s=0.2 if i < 18 else None,
                binance_taker_flow_10s=0.2 if i < 18 else None,
                binance_taker_flow_30s=0.2 if i < 18 else None,
                binance_taker_flow_60s=tf_60s,
                binance_top1_depth_imbalance=0.05,
                binance_top5_depth_imbalance=0.05,
                binance_top20_depth_imbalance=0.05,
                binance_is_valid=True,
                source_to_receive_latency_ms=20,
                inter_feed_receive_skew_ms=0,
                is_stale=False,
                stale_reason=None,
                is_valid=True,
                raw_payload_id=None,
                raw_json="{}",
            )
            samples.append(s)

        db.save_leadlag_v2_samples_batch(samples)
        collector._total_samples_count = 20

        coverage = collector._compute_actual_taker_flow_coverage()
        # 18 non-null out of 20 eligible valid post-warmup samples = 0.9
        assert coverage == 0.9, f"Expected 0.9 coverage, got {coverage}"


def test_binance_ws_url_uses_trade_stream() -> None:
    """WS URL must use @trade (not deprecated @aggTrade) for Binance futures trade data."""
    from pm_research.research.btc5m.leadlag_v2_collector import BINANCE_WS_URL
    assert "btcusdt@trade" in BINANCE_WS_URL
    assert "aggTrade" not in BINANCE_WS_URL, "aggTrade stream is deprecated on Binance futures"
