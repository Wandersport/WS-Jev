"""Unit and regression tests for btc5m leadlag v3 replication architecture."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pm_research.research.btc5m.leadlag_v3_collector import (
    LeadLagCollectorV3,
)
from pm_research.research.btc5m.leadlag_v3_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_PILOT_ID,
    EXPERIMENT_SPEC_HASH,
    FROZEN_TRANSPORT_MODEL_HASH,
    FROZEN_TRANSPORT_MODEL_PATH,
    PILOT_PHYSICAL_ROUNDS,
    PREDECLARED_LAGS_SEC,
    TARGET_PHYSICAL_ROUNDS,
    compute_experiment_spec_hash,
)
from pm_research.storage.db import Database


def test_v3_spec_constants_and_hash():
    """Verify frozen spec constants, SHA256 checksums, and model hashes."""
    assert EXPERIMENT_ID == "btc5m_leadlag_v3_replication_2s"
    assert EXPERIMENT_PILOT_ID == "btc5m_leadlag_v3_replication_2s_pilot"
    assert TARGET_PHYSICAL_ROUNDS == 250
    assert PILOT_PHYSICAL_ROUNDS == 10
    assert PREDECLARED_LAGS_SEC == (2,)

    computed_spec_hash = compute_experiment_spec_hash()
    assert computed_spec_hash == EXPERIMENT_SPEC_HASH
    assert computed_spec_hash == "bfe5b0553a7143dd8941239dd481cf0e86499b54338dc6717565d1d9dc28fa53"

    assert FROZEN_TRANSPORT_MODEL_PATH.exists()
    import hashlib

    model_bytes = FROZEN_TRANSPORT_MODEL_PATH.read_bytes()
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    assert model_sha == FROZEN_TRANSPORT_MODEL_HASH
    assert model_sha == "ee914a59072320d142944530957036c325632f941432d934f472e689dc7f1c77"


def test_v3_collector_init_and_safety(tmp_path: Path):
    """Verify collector initializes safely with isolated DB and paper constraints."""
    db_file = tmp_path / "test_replication.db"
    db = Database(db_file)
    collector = LeadLagCollectorV3(
        db=db,
        is_pilot=True,
        lock_file_path=str(tmp_path / "collector.lock"),
    )

    assert collector.is_pilot is True
    assert collector.experiment_id == EXPERIMENT_PILOT_ID
    assert collector.target_physical_rounds == 10
    assert collector._write_queue.qsize() == 0

    # Verify absence of live execution attributes
    for forbidden in ["wallet", "private_key", "sign_transaction", "broker_url", "live"]:
        assert not hasattr(collector, forbidden)


def test_v3_fail_closed_on_queue_backlog(tmp_path: Path):
    """Verify collector fails closed if persistence worker queue exceeds max limit."""
    db_file = tmp_path / "test_backlog.db"
    db = Database(db_file)
    collector = LeadLagCollectorV3(
        db=db,
        max_queue_depth=5,
        lock_file_path=str(tmp_path / "collector.lock"),
    )

    # Enqueue items up to limit
    for i in range(5):
        collector._enqueue_write("test_op", {"idx": i})

    assert collector._write_queue.qsize() == 5

    # 6th item must trigger fail-closed exception
    with pytest.raises(RuntimeError, match="Persistence queue exceeded safety limit"):
        collector._enqueue_write("test_op", {"idx": 6})

    assert collector._stop_event.is_set()


def test_v3_async_persistence_drain(tmp_path: Path):
    """Verify background persistence worker batches and writes to SQLite cleanly."""
    db_file = tmp_path / "test_async_drain.db"
    db = Database(db_file)
    collector = LeadLagCollectorV3(
        db=db,
        lock_file_path=str(tmp_path / "collector.lock"),
    )

    collector._start_persistence_worker()

    # Enqueue sample, raw payload, and heartbeat
    now_ms = int(time.time() * 1000)
    sample_data = {
        "sample_id": "round1_1000",
        "round_slug": "round1",
        "experiment_id": EXPERIMENT_ID,
        "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
        "is_pilot": 0,
        "sample_target_ts_ms": now_ms,
        "sample_actual_ts_ms": now_ms + 2,
        "local_monotonic_ns": time.monotonic_ns(),
        "seconds_remaining": 298,
        "poly_recv_ts_ms": now_ms - 10,
        "poly_receipt_age_ms": 12,
        "poly_best_bid": 0.51,
        "poly_best_ask": 0.52,
        "poly_midpoint": 0.515,
        "poly_spread": 0.01,
        "poly_is_crossed": 0,
        "poly_is_valid": 1,
        "binance_recv_ts_ms": now_ms - 5,
        "binance_receipt_age_ms": 7,
        "binance_best_bid": 65000.0,
        "binance_best_ask": 65000.5,
        "binance_mid_price": 65000.25,
        "binance_is_valid": 1,
        "is_stale": 0,
        "is_valid": 1,
    }
    collector._enqueue_write("sample", sample_data)

    payload_bytes = b'{"e":"depthUpdate","b":[["65000","1.0"]],"a":[["65001","1.0"]]}'
    payload_id = collector._persist_lossless_raw_payload("binance", payload_bytes, now_ms)
    assert payload_id is not None

    collector._emit_heartbeat()

    # Wait for queue to drain
    timeout = time.time() + 3.0
    while not collector._write_queue.empty() and time.time() < timeout:
        time.sleep(0.05)

    collector._stop_event.set()
    if collector._writer_thread:
        collector._writer_thread.join(timeout=2.0)

    # Verify data landed in database
    samples = db.get_leadlag_v2_samples("round1", experiment_id=EXPERIMENT_ID)
    assert len(samples) == 1
    assert samples[0]["sample_id"] == "round1_1000"

    hb = db.get_leadlag_v2_latest_heartbeat(experiment_id=EXPERIMENT_ID)
    assert hb is not None
    assert hb["experiment_id"] == EXPERIMENT_ID

    # Verify lossless raw payload roundtrip
    raw_record = db.get_leadlag_v2_raw_payload(payload_id)
    assert raw_record is not None
    assert raw_record["raw_bytes"] == payload_bytes


def test_v3_in_memory_telemetry_accuracy(tmp_path: Path):
    """Verify O(1) in-memory telemetry calculation without full-table queries."""
    db_file = tmp_path / "test_telemetry.db"
    db = Database(db_file)
    collector = LeadLagCollectorV3(
        db=db,
        lock_file_path=str(tmp_path / "collector.lock"),
    )

    # Simulate tick capture
    t0 = int(time.time() * 1000)
    collector._poly_best_bid = 0.50
    collector._poly_best_ask = 0.51
    collector._poly_midpoint = 0.505
    collector._poly_recv_ts_ms = t0
    collector._bn_bids = {65000.0: 1.5}
    collector._bn_asks = {65000.5: 2.0}
    collector._bn_recv_ts_ms = t0
    collector._current_round_slug = "btc-updown-5m-1700000000"

    s1 = collector._capture_sample(t0)
    assert s1.sample_id == f"btc-updown-5m-1700000000_{t0}"
    assert collector._captured_ticks_count == 1
    assert collector._intended_ticks_count == 1

    snap = collector.get_telemetry_snapshot()
    assert snap["captured_ticks"] == 1
    assert snap["capture_ratio"] == 1.0
    assert snap["writer_queue_depth"] == 1


def test_v3_fail_fast_checks(tmp_path: Path):
    """Verify pilot fail-fast triggers immediately on SLA violations."""
    db = Database(tmp_path / "test_ff.db")
    collector = LeadLagCollectorV3(db=db, is_pilot=True, lock_file_path=str(tmp_path / "collector.lock"))

    # 1. Zero trades after 30s
    collector._aggtrade_events_received = 0
    with pytest.raises(RuntimeError, match="0 trades received after 30s"):
        collector._check_pilot_fail_fast(35.0)

    # 2. Polymarket delta missing source ts
    collector._aggtrade_events_received = 50
    collector._delta_events_received = 20
    collector._poly_source_ts_present = 0
    with pytest.raises(RuntimeError, match="0 source timestamps on Polymarket delta"):
        collector._check_pilot_fail_fast(35.0)

    # 3. Excessive parser exceptions
    collector._poly_source_ts_present = 20
    collector._parser_exception_count = 35
    with pytest.raises(RuntimeError, match="Excessive parser exceptions"):
        collector._check_pilot_fail_fast(35.0)

    # 4. Binance TS coverage < 99% after 60s
    collector._parser_exception_count = 0
    collector._binance_source_ts_present = 90
    collector._binance_source_ts_missing = 10  # 90% < 99%
    with pytest.raises(RuntimeError, match="Binance TS coverage"):
        collector._check_pilot_fail_fast(65.0)

    # 5. Poly TS coverage < 95% after 60s
    collector._binance_source_ts_present = 100
    collector._binance_source_ts_missing = 0
    collector._poly_source_ts_present = 90
    collector._poly_source_ts_missing = 10  # 90% < 95%
    with pytest.raises(RuntimeError, match="Poly TS coverage"):
        collector._check_pilot_fail_fast(65.0)

    # 6. Taker flow coverage < 95% after 120s
    collector._poly_source_ts_present = 100
    collector._poly_source_ts_missing = 0
    collector._valid_post_warmup_samples_count = 100
    collector._non_null_tf_60s_count = 90  # 90% < 95%
    with pytest.raises(RuntimeError, match="Taker-flow 60s coverage"):
        collector._check_pilot_fail_fast(125.0)


def test_v3_round_lifecycle_in_memory(tmp_path: Path):
    """Verify physical round transitions and completion without SQLite blocking."""
    db = Database(tmp_path / "test_lifecycle.db")
    collector = LeadLagCollectorV3(db=db, lock_file_path=str(tmp_path / "collector.lock"))

    # Transition to round
    collector._current_round_slug = "btc-updown-5m-1700000000"
    collector._current_round_sample_count = 300
    collector._current_round_valid_count = 295

    collector._close_current_round()
    assert collector._current_round_sample_count == 0
    assert collector._current_round_valid_count == 0
    assert collector._write_queue.qsize() == 1

    item = collector._write_queue.get_nowait()
    assert item[0] == "round_complete"
    assert item[1]["round_slug"] == "btc-updown-5m-1700000000"
    assert item[1]["total_samples"] == 300
    assert item[1]["valid_samples"] == 295


def test_v3_pilot_audit_percentile():
    """Verify statistical percentile helper in pilot audit."""
    from pm_research.research.btc5m.leadlag_v3_pilot_audit import _percentile

    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(data, 50) == 30.0
    assert _percentile(data, 0) == 10.0
    assert _percentile(data, 100) == 50.0
    assert _percentile([], 50) == 0.0


def test_v3_confirmatory_unblinding_artifacts():
    """Verify Phase 8E confirmatory replication artifacts and verdict integrity."""
    import json
    from pm_research.research.btc5m.leadlag_v3_confirmatory_unblinding import (
        OUTPUT_DIR,
        student_t_two_sided_p,
        compute_percentiles,
    )

    # 1. Statistical helper tests
    p_val_2 = student_t_two_sided_p(2.0, 249)
    assert 0.045 < p_val_2 < 0.050
    p_val_zero = student_t_two_sided_p(0.0, 249)
    assert abs(p_val_zero - 1.0) < 1e-4

    pcts = compute_percentiles([10.0, 20.0, 30.0, 40.0, 50.0], [50.0])
    assert pcts[50.0] == 30.0

    # 2. Artifact file existence
    expected_files = [
        "README.md",
        "final_replication_verdict.json",
        "preregistration_verification.json",
        "primary_transport_results.csv",
        "secondary_refit_results.csv",
        "timing_sensitivity.csv",
        "round_robustness.csv",
        "volatility_heterogeneity.csv",
        "temporal_stability.csv",
    ]
    for fname in expected_files:
        p = OUTPUT_DIR / fname
        assert p.exists(), f"Missing artifact: {p}"

    # 3. Verdict contents
    verdict_path = OUTPUT_DIR / "final_replication_verdict.json"
    with open(verdict_path) as f:
        v = json.load(f)

    assert v["canonical_experiment_id"] == "btc5m_leadlag_v3_replication_2s"
    assert v["final_replication_verdict"] == "NOT_REPLICATED"
    assert v["primary_results"]["n_pairs"] == 65436
    assert v["primary_results"]["n_rounds"] == 250
    assert v["preregistered_gates"]["gate_integrity"] is True


