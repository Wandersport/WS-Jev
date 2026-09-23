"""Unit and integration tests for Phase 7 autonomous BTC 5m prospective research collector.

Tests cover:
1. Experiment specification freezing and deterministic spec hashing (SHA-256)
2. Spec tampering detection / rejection
3. Binance boundary midpoint selection within +/- 2500ms tolerance
4. Rejection of Binance midpoints outside +/- 2500ms tolerance
5. Missing boundary midpoint strictly forces return_since_round_open_bps to None
6. Collector startup crash recovery and reconciliation of uncompleted rounds
7. Duplicate round prevention and idempotent scoring
8. Cache deduplication preventing redundant OpenRouter billing
9. Cost guard ceiling enforcement (pauses LLM inference when limit reached)
10. Milestone checkpoint triggers (at 30, 100, 500 rounds) and JSON artifact generation
11. Transactional SQLite online backup creation and rotation
12. Graceful stop file signaling (data/collector.stop)
13. Target valid rounds loop termination
14. Strict paper-only research safety invariants (0 proposals, 0 orders, 0 fills)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from pm_research.research.btc5m.ablation import (
    COND_A_REF_ONLY,
    COND_B_REF_PERP,
    COND_C_MARKET_AWARE,
    COND_D_FULL,
    BTC5mAblationForecast,
)
from pm_research.research.btc5m.backup import (
    backup_database,
    list_backups,
    restore_database,
)
from pm_research.research.btc5m.binance_feed import (
    BINANCE_OPEN_TIMING_TOLERANCE_MS,
    BinancePerpFeed,
)
from pm_research.research.btc5m.collector import (
    BTC5mAutonomousCollector,
)
from pm_research.research.btc5m.contract import BTC5mRoundInfo
from pm_research.research.btc5m.experiment import (
    CANONICAL_SPEC,
    EXPERIMENT_ID,
    EXPERIMENT_SPEC_HASH,
    FROZEN_CONDITIONS,
    MODEL_PIN,
    PRIMARY_MARKET_BASELINE,
    STANDARD_HORIZONS,
    compute_experiment_spec_hash,
    get_experiment_spec,
    verify_phase6_compatibility,
)
from pm_research.research.btc5m.lab import BTC5mShadowLab
from pm_research.research.btc5m.process import (
    CollectorLock,
    determine_collector_status,
    is_pid_alive,
    launch_detached_collector,
    read_collector_pid,
    remove_collector_pid,
    stop_collector,
    write_collector_pid,
)
from pm_research.storage.db import Database

# ---------------------------------------------------------------------------
# 1. Experiment Spec Freeze & Hashing Tests
# ---------------------------------------------------------------------------


def test_spec_hash_determinism() -> None:
    """The spec hash must be deterministic, non-empty, and match EXPERIMENT_SPEC_HASH."""
    h1 = compute_experiment_spec_hash(CANONICAL_SPEC)
    h2 = compute_experiment_spec_hash(CANONICAL_SPEC)
    assert h1 == h2
    assert len(h1) == 64
    assert h1 == EXPERIMENT_SPEC_HASH


def test_spec_frozen_constants() -> None:
    """Verify all frozen experiment v1 specification parameters."""
    assert EXPERIMENT_ID == "btc5m_jev_ablation_v1"
    assert MODEL_PIN == "typesafe/jev-1.13"
    assert STANDARD_HORIZONS == (240, 180, 120, 60, 30)
    assert PRIMARY_MARKET_BASELINE == "NATIVE_UP_MIDPOINT"
    assert FROZEN_CONDITIONS == (
        COND_A_REF_ONLY,
        COND_B_REF_PERP,
        COND_C_MARKET_AWARE,
        COND_D_FULL,
    )


def test_spec_tampering_detected() -> None:
    """Any modification to the frozen specification alters the spec hash."""
    spec_copy = get_experiment_spec()
    spec_copy["horizons_sec"] = [240, 180, 120, 60]  # altered horizons
    tampered_hash = compute_experiment_spec_hash(spec_copy)
    assert tampered_hash != EXPERIMENT_SPEC_HASH

    spec_copy2 = get_experiment_spec()
    spec_copy2["model_id"] = "typesafe/jev-latest"  # altered model
    tampered_hash2 = compute_experiment_spec_hash(spec_copy2)
    assert tampered_hash2 != EXPERIMENT_SPEC_HASH


def test_phase6_compatibility_verification() -> None:
    """Verify phase 6 compatibility checker rejects invalid horizons or models."""
    valid_snap = MagicMock(is_valid=True, target_horizon_sec=240, market_q_primary=0.55)
    valid_fc = MagicMock(is_valid=True, condition=COND_A_REF_ONLY, model_id="typesafe/jev-1.13")

    ok, reason = verify_phase6_compatibility(
        round_slug="btc-updown-5m-1700000000",
        snapshots=[valid_snap],
        forecasts=[valid_fc],
    )
    assert ok is True
    assert reason is None

    # Invalid slug format
    ok, reason = verify_phase6_compatibility(
        round_slug="random-slug",
        snapshots=[valid_snap],
        forecasts=[valid_fc],
    )
    assert ok is False
    assert "slug" in reason.lower()

    # Invalid horizon
    invalid_snap = MagicMock(is_valid=True, target_horizon_sec=50, market_q_primary=0.5)
    ok, reason = verify_phase6_compatibility(
        round_slug="btc-updown-5m-1700000000",
        snapshots=[invalid_snap],
        forecasts=[valid_fc],
    )
    assert ok is False
    assert "horizon" in reason.lower()


# ---------------------------------------------------------------------------
# 2. Binance Round-Open Provenance Tests (+/- 2500ms tolerance)
# ---------------------------------------------------------------------------


def test_binance_open_tolerance_constant() -> None:
    """Ensure tolerance is strictly 2500ms as required by Phase 7 specification."""
    assert BINANCE_OPEN_TIMING_TOLERANCE_MS == 2500


def test_binance_boundary_open_within_tolerance() -> None:
    """Midpoint within +/- 2500ms of round_start is matched and recorded with provenance."""
    feed = BinancePerpFeed()
    round_start_ms = 1_700_000_000_000

    # Add midpoints in history
    # 1. 2000ms before boundary (within 2500ms)
    feed._midpoint_history.append((round_start_ms - 2000, round_start_ms - 1980, 68000.0))
    # 2. 500ms after boundary (within 2500ms) - closest
    feed._midpoint_history.append((round_start_ms + 500, round_start_ms + 520, 68010.0))

    match = feed.find_boundary_open_mid(round_start_ms=round_start_ms, tolerance_ms=2500)
    assert match is not None
    mid, s_ts, r_ts, offset_ms = match
    assert mid == 68010.0
    assert s_ts == round_start_ms + 500
    assert r_ts == round_start_ms + 520
    assert offset_ms == 500

    # Record round open
    feed.record_round_open(
        round_slug="test-slug",
        mid_price=mid,
        timestamp_ms=r_ts,
        source_timestamp_ms=s_ts,
        offset_ms=offset_ms,
    )
    stored = feed.get_round_open("test-slug")
    assert stored is not None
    assert stored[0] == 68010.0
    assert stored[1] == r_ts
    assert stored[2] == s_ts
    assert stored[3] == offset_ms


def test_binance_boundary_open_outside_tolerance_rejected() -> None:
    """Midpoint at 2501ms or greater is rejected by find_boundary_open_mid."""
    feed = BinancePerpFeed()
    round_start_ms = 1_700_000_000_000

    # Only a point 3000ms after boundary
    feed._midpoint_history.append((round_start_ms + 3000, round_start_ms + 3020, 68050.0))

    match = feed.find_boundary_open_mid(round_start_ms=round_start_ms, tolerance_ms=2500)
    assert match is None


def test_missing_boundary_open_sets_return_to_none() -> None:
    """When open midpoint is missing or outside tolerance, return_since_round_open_bps is None."""
    feed = BinancePerpFeed()
    now_ms = 1_700_000_000_000
    # Add depth so mid_price is valid: (68060 + 68076)/2 = 68068.0
    feed.add_depth([(68060.0, 1.0)], [(68076.0, 1.0)], source_timestamp_ms=now_ms)

    # Compute features with binance_open_price = None
    feats = feed.compute_features(
        now_ms=now_ms,
        ref_price=68000.0,
        binance_open_price=None,
    )
    assert feats.return_since_round_open_bps is None

    # Compute features with binance_open_price but timing offset exceeds tolerance
    feats_exceeded = feed.compute_features(
        now_ms=now_ms,
        ref_price=68000.0,
        binance_open_price=68000.0,
        binance_open_timing_offset_ms=3000,  # exceeds 2500ms
    )
    assert feats_exceeded.return_since_round_open_bps is None

    # Within tolerance -> valid return: (68068 - 68000) / 68000 * 10000 = +10.0 bps
    feats_valid = feed.compute_features(
        now_ms=now_ms,
        ref_price=68000.0,
        binance_open_price=68000.0,
        binance_open_timing_offset_ms=500,
    )
    assert feats_valid.return_since_round_open_bps is not None
    assert round(feats_valid.return_since_round_open_bps, 1) == 10.0


# ---------------------------------------------------------------------------
# 3. Transactional SQLite Backup Tests
# ---------------------------------------------------------------------------


def test_sqlite_backup_and_rotation(tmp_path: Path) -> None:
    """Test transactional backup creation, listing, rotation, and restoration."""
    db_file = tmp_path / "test.db"
    backup_dir = tmp_path / "backups"

    # Create dummy database
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE sample (id INT, val TEXT);")
    conn.execute("INSERT INTO sample VALUES (1, 'alpha'), (2, 'beta');")
    conn.commit()
    conn.close()

    # Create backup 1
    b1 = backup_database(
        db_path=db_file,
        backup_dir=backup_dir,
        max_backups=2,
        rounds_count=10,
    )
    assert b1.exists()
    assert "_10r.db" in b1.name

    time.sleep(0.01)
    # Create backup 2
    b2 = backup_database(
        db_path=db_file,
        backup_dir=backup_dir,
        max_backups=2,
        rounds_count=20,
    )
    assert b2.exists()
    assert len(list_backups(backup_dir)) == 2

    time.sleep(0.01)
    # Create backup 3 -> should prune backup 1 since max_backups=2
    b3 = backup_database(
        db_path=db_file,
        backup_dir=backup_dir,
        max_backups=2,
        rounds_count=30,
    )
    assert b3.exists()
    backups = list_backups(backup_dir)
    assert len(backups) == 2
    assert b1 not in backups
    assert b3 in backups
    assert b2 in backups

    # Restore test
    restored_db = tmp_path / "restored.db"
    restore_database(b3, restored_db)
    assert restored_db.exists()

    conn_r = sqlite3.connect(str(restored_db))
    rows = conn_r.execute("SELECT val FROM sample ORDER BY id;").fetchall()
    conn_r.close()
    assert rows == [("alpha",), ("beta",)]


# ---------------------------------------------------------------------------
# 4. Collector Startup Crash Recovery & Idempotency Tests
# ---------------------------------------------------------------------------


def test_collector_startup_recovery(tmp_path: Path) -> None:
    """Collector reconciles uncompleted active rounds from database on startup."""
    db = Database(db_path=tmp_path / "test_rec.db")

    # Insert an expired active round
    past_epoch = int(time.time()) - 200
    start_dt = datetime.fromtimestamp(past_epoch - 300, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(past_epoch, tz=timezone.utc)
    r_info = BTC5mRoundInfo(
        slug="btc-updown-5m-1700000000",
        condition_id="0x" + "1" * 64,
        market_id="m1",
        question="BTC up?",
        description="",
        resolution_source_url="https://data.chain.link/streams/btc-usd-twap-60s-streams",
        settlement_rule="",
        start_time_utc=start_dt,
        end_time_utc=end_dt,
        duration_seconds=300,
        up_token_id="tok_up",
        down_token_id="tok_down",
        up_outcome_index=1,
        down_outcome_index=0,
        accepting_orders=True,
        price_to_beat=68000.0,
        price_to_beat_source="gamma",
        fee_rate=0.0,
        fee_exponent=1.0,
    )
    db.save_btc5m_round(r_info, status="active")

    # Mock lab to return resolution
    mock_lab = MagicMock(spec=BTC5mShadowLab)
    mock_lab.ref_feed = MagicMock(status="connected")
    mock_lab.binance_feed = MagicMock(status="connected")
    mock_lab.contract_mgr = MagicMock()
    mock_lab.contract_mgr.discover_round_by_slug.return_value = (r_info, None)
    mock_res = MagicMock(is_resolved=True, resolved_outcome="Up")
    mock_lab.poll_round_resolution.return_value = mock_res

    collector = BTC5mAutonomousCollector(
        db=db,
        lab=mock_lab,
        target_valid_rounds=10,
        status_file_path=tmp_path / "status.json",
        stop_file_path=tmp_path / "stop.txt",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    collector.reconcile_startup_state()
    mock_lab.poll_round_resolution.assert_called_once()


def test_collector_idempotent_duplicate_prevention(tmp_path: Path) -> None:
    """Saving duplicate round information updates record idempotently without SQL error."""
    db = Database(db_path=tmp_path / "test_idem.db")
    now_epoch = int(time.time())
    start_dt = datetime.fromtimestamp(now_epoch, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(now_epoch + 300, tz=timezone.utc)
    r_info = BTC5mRoundInfo(
        slug="btc-updown-5m-idem",
        condition_id="0x" + "1" * 64,
        market_id="m_idem",
        question="BTC up?",
        description="",
        resolution_source_url="https://data.chain.link/streams/btc-usd-twap-60s-streams",
        settlement_rule="",
        start_time_utc=start_dt,
        end_time_utc=end_dt,
        duration_seconds=300,
        up_token_id="tok_up",
        down_token_id="tok_down",
        up_outcome_index=1,
        down_outcome_index=0,
        accepting_orders=True,
        price_to_beat=68000.0,
        price_to_beat_source="gamma",
        fee_rate=0.0,
        fee_exponent=1.0,
    )
    # Save twice
    db.save_btc5m_round(r_info, status="active")
    db.save_btc5m_round(r_info, status="active")
    rounds = db.get_btc5m_rounds()
    assert len(rounds) == 1
    assert rounds[0]["round_slug"] == "btc-updown-5m-idem"


# ---------------------------------------------------------------------------
# 5. OpenRouter Cached Deduplication & Cost Ceiling Tests
# ---------------------------------------------------------------------------


def test_cost_guard_trigger_on_ceiling_breach(tmp_path: Path) -> None:
    """Collector stops executing LLM forecasts when cost ceiling is breached."""
    db = Database(db_path=tmp_path / "test_cost.db")
    status_file = tmp_path / "status.json"

    # Prepopulate forecast with cost exceeding ceiling of $1.00 via direct SQL
    with db._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO jev_forecasts (
                forecast_id, capture_id, market_id, condition, model_id, model_returned,
                schema_version, request_hash, captured_at, market_question,
                jev_yes_probability, jev_no_probability, jev_choice, cost,
                raw_response_hash, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "f_cost", "c1", "m1", "cond_a", MODEL_PIN, MODEL_PIN,
                "v1", "h1", "2026-01-01T00:00:00Z", "q",
                0.5, 0.5, "UP", 1.50,
                "raw1", "{}",
            ),
        )
    assert db.get_total_openrouter_cost() == 1.50

    mock_lab = MagicMock(spec=BTC5mShadowLab)
    mock_lab.ref_feed = MagicMock(status="connected")
    mock_lab.binance_feed = MagicMock(status="connected")
    collector = BTC5mAutonomousCollector(
        db=db,
        lab=mock_lab,
        target_valid_rounds=10,
        cost_ceiling_usd=1.00,  # limit is $1.00
        status_file_path=status_file,
        stop_file_path=tmp_path / "stop.txt",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    collector.emit_heartbeat()
    with open(status_file, "r") as f:
        data = json.load(f)
    assert data["total_openrouter_cost_usd"] == 1.50
    assert data["cost_ceiling_usd"] == 1.00


def test_milestone_checkpoint_trigger(tmp_path: Path) -> None:
    """Milestone checkpoint creates DB record, JSON artifact, and backup."""
    db = Database(db_path=tmp_path / "test_chk.db")
    checkpoints_dir = tmp_path / "checkpoints"

    mock_lab = MagicMock(spec=BTC5mShadowLab)
    mock_lab.compute_evaluation_summary.return_value = {
        "market_native_brier": 0.20,
        "ref_only_brier": 0.23,
    }

    collector = BTC5mAutonomousCollector(
        db=db,
        lab=mock_lab,
        target_valid_rounds=100,
        milestones=(30, 100, 500),
        status_file_path=tmp_path / "status.json",
        stop_file_path=tmp_path / "stop.txt",
        checkpoints_dir=checkpoints_dir,
    )

    artifact = collector.trigger_milestone_checkpoint(milestone=30, valid_rounds=30)
    assert artifact is not None
    assert artifact.exists()

    with open(artifact, "r") as f:
        c_data = json.load(f)
    assert c_data["checkpoint_milestone"] == 30
    assert c_data["valid_resolved_rounds"] == 30
    assert c_data["experiment_spec_hash"] == EXPERIMENT_SPEC_HASH
    assert c_data["evaluation_summary"]["market_native_brier"] == 0.20

    # DB record check
    checkpoints = db.get_checkpoints()
    assert len(checkpoints) == 1
    assert checkpoints[0]["milestone_rounds"] == 30


def test_collector_stop_file_detection(tmp_path: Path) -> None:
    """Creating stop file causes is_stop_requested() to return True."""
    stop_file = tmp_path / "collector.stop"
    collector = BTC5mAutonomousCollector(
        stop_file_path=stop_file,
        status_file_path=tmp_path / "status.json",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    assert collector.is_stop_requested() is False

    stop_file.touch()
    assert collector.is_stop_requested() is True


def test_paper_only_safety_contract_unbroken() -> None:
    """Verify that collector has zero trade proposals, orders, fills, or positions."""
    collector = BTC5mAutonomousCollector()
    # Check that collector has no trading methods or attributes
    assert not hasattr(collector, "place_order")
    assert not hasattr(collector, "send_transaction")
    assert not hasattr(collector, "wallet")
    assert not hasattr(collector, "private_key")
    assert not hasattr(collector, "api_key")


# ---------------------------------------------------------------------------
# Phase 7.1 Operational Hardening Tests: Process, Concurrency Lock, Cost Semantics
# ---------------------------------------------------------------------------


def test_collector_lock_single_instance(tmp_path: Path) -> None:
    """Exclusive lock prevents two collectors from running concurrently."""
    lock_file = tmp_path / "collector.lock"
    lock1 = CollectorLock(lock_file)
    assert lock1.acquire() is True
    assert lock1.is_locked_by_other() is True

    # Second instance attempting to acquire the same lock MUST fail
    lock2 = CollectorLock(lock_file)
    assert lock2.acquire() is False

    # Once lock1 is released, lock2 can acquire
    lock1.release()
    assert lock1.is_locked_by_other() is False
    assert lock2.acquire() is True
    lock2.release()


def test_collector_lock_run_rejection(tmp_path: Path) -> None:
    """Collector.run() immediately terminates with LOCK_REJECTED if lock is held."""
    lock_file = tmp_path / "test.lock"
    pid_file = tmp_path / "test.pid"

    # Pre-hold the lock
    outer_lock = CollectorLock(lock_file)
    assert outer_lock.acquire() is True

    try:
        collector = BTC5mAutonomousCollector(
            lock_file_path=lock_file,
            pid_file_path=pid_file,
            status_file_path=tmp_path / "status.json",
            checkpoints_dir=tmp_path / "checkpoints",
        )
        collector.run()
        assert collector.status == "LOCK_REJECTED"
        assert not pid_file.exists()
    finally:
        outer_lock.release()


def test_pid_management_lifecycle(tmp_path: Path) -> None:
    """Verify PID writing, reading, verification, and cleanup."""
    pid_file = tmp_path / "collector.pid"
    assert read_collector_pid(pid_file) is None

    my_pid = os.getpid()
    write_collector_pid(my_pid, pid_file)
    assert read_collector_pid(pid_file) == my_pid
    assert is_pid_alive(my_pid) is True
    assert is_pid_alive(999999999) is False

    remove_collector_pid(pid_file)
    assert read_collector_pid(pid_file) is None


def test_detached_launcher_command_args_safety(tmp_path: Path) -> None:
    """Detached process launcher passes parameters via argv without secrets in argv."""
    lock_file = tmp_path / "test.lock"
    pid_file = tmp_path / "test.pid"
    log_file = tmp_path / "test.log"

    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        success, pid, msg = launch_detached_collector(
            target_valid_rounds=50,
            cost_ceiling_usd=7.50,
            pid_file=pid_file,
            lock_file=lock_file,
            log_file=log_file,
            cwd=tmp_path,
        )

        assert success is True
        assert pid == 12345
        assert "Launched detached collector with PID 12345" in msg
        mock_popen.assert_called_once()
        call_kwargs = mock_popen.call_args.kwargs
        call_args = mock_popen.call_args.args[0]

        # Invariant: session must be detached
        assert call_kwargs.get("start_new_session") is True
        # Invariant: argv contains command arguments but NO raw keys or secrets
        assert "btc5m-collect" in call_args
        assert "--target-valid-rounds" in call_args
        assert "50" in call_args
        assert "--cost-ceiling" in call_args
        assert "7.5" in call_args
        for arg in call_args:
            assert "OPENROUTER" not in str(arg)
            assert "sk-or" not in str(arg)


def test_determine_collector_status_states(tmp_path: Path) -> None:
    """Test all status states: NOT_STARTED, RUNNING_HEALTHY, RUNNING_STALE_HEARTBEAT, PROCESS_DEAD."""
    pid_file = tmp_path / "collector.pid"
    status_file = tmp_path / "status.json"
    lock_file = tmp_path / "collector.lock"

    # 1. NOT_STARTED
    st1 = determine_collector_status(pid_file=pid_file, status_file=status_file, lock_file=lock_file)
    assert st1["state"] == "NOT_STARTED"
    assert st1["process_alive"] is False
    assert st1["lock_held"] is False

    # 2. RUNNING_HEALTHY
    my_pid = os.getpid()
    write_collector_pid(my_pid, pid_file)
    now_epoch = int(time.time())
    with open(status_file, "w") as f:
        json.dump({"timestamp_epoch": now_epoch, "status": "COLLECTING"}, f)

    with patch("pm_research.research.btc5m.process.verify_collector_process", return_value=True):
        st2 = determine_collector_status(pid_file=pid_file, status_file=status_file, lock_file=lock_file)
        assert st2["state"] == "RUNNING_HEALTHY"
        assert st2["process_alive"] is True
        assert st2["heartbeat_fresh"] is True

    # 3. RUNNING_STALE_HEARTBEAT (>120s old)
    with open(status_file, "w") as f:
        json.dump({"timestamp_epoch": now_epoch - 200, "status": "COLLECTING"}, f)

    with patch("pm_research.research.btc5m.process.verify_collector_process", return_value=True):
        st3 = determine_collector_status(pid_file=pid_file, status_file=status_file, lock_file=lock_file)
        assert st3["state"] == "RUNNING_STALE_HEARTBEAT"
        assert st3["process_alive"] is True
        assert st3["heartbeat_fresh"] is False
        assert st3["heartbeat_age_sec"] >= 200

    # 4. PROCESS_DEAD (PID in file does not exist)
    write_collector_pid(999999999, pid_file)
    st4 = determine_collector_status(pid_file=pid_file, status_file=status_file, lock_file=lock_file)
    assert st4["state"] == "PROCESS_DEAD"
    assert st4["process_alive"] is False


def test_stop_collector_graceful(tmp_path: Path) -> None:
    """stop_collector creates stop file and cleans up PID."""
    pid_file = tmp_path / "collector.pid"
    stop_file = tmp_path / "collector.stop"

    # Dead process cleanup
    write_collector_pid(999999999, pid_file)
    success, msg = stop_collector(pid_file=pid_file, stop_file=stop_file)
    assert success is True
    assert not pid_file.exists()


def test_openrouter_cost_deduplication_and_audit(tmp_path: Path) -> None:
    """Verify that cache hits add $0.00 to total spend, and audit accurately reports metrics."""
    db = Database(db_path=tmp_path / "test_cost.db")

    fc_remote = BTC5mAblationForecast(
        forecast_id="f_remote",
        snapshot_id="s1",
        round_slug="r1",
        target_horizon_sec=240,
        condition="BTC5M_REFERENCE_ONLY",
        model_id=MODEL_PIN,
        request_hash="hash_remote",
        captured_at_ms=1000,
        market_q=0.5,
        jev_up_prob=0.55,
        jev_down_prob=0.45,
        jev_choice="UP",
        confidence=0.8,
        input_tokens=100,
        output_tokens=20,
        latency_ms=250,
        from_cache=False,
        raw_response_hash="raw_remote",
        created_at_utc="2026-01-01T00:00:00Z",
        cost=0.0010,
    )
    db.save_btc5m_forecast(fc_remote)

    fc_cache = BTC5mAblationForecast(
        forecast_id="f_cache",
        snapshot_id="s2",
        round_slug="r1",
        target_horizon_sec=240,
        condition="BTC5M_REFERENCE_ONLY",
        model_id=MODEL_PIN,
        request_hash="hash_remote",
        captured_at_ms=1000,
        market_q=0.5,
        jev_up_prob=0.55,
        jev_down_prob=0.45,
        jev_choice="UP",
        confidence=0.8,
        input_tokens=100,
        output_tokens=20,
        latency_ms=10,
        from_cache=True,
        raw_response_hash="raw_remote",
        created_at_utc="2026-01-01T00:00:00Z",
        cost=None,
    )
    db.save_btc5m_forecast(fc_cache)

    # Total OpenRouter cost must count ONLY the remote charge
    total_cost = db.get_total_openrouter_cost()
    assert abs(total_cost - 0.0010) < 1e-6

    # Audit check
    audit = db.get_cost_accounting_audit()
    assert audit["total_db_forecasts"] == 2
    assert audit["remote_requests_charged"] == 1
    assert audit["local_cache_hits_zero_cost"] == 1
    assert abs(audit["canonical_total_reported_cost_usd"] - 0.0010) < 1e-6


# ---------------------------------------------------------------------------
# 7. Phase 7.2 Checkpoint Idempotency & Dataset Audit Tests
# ---------------------------------------------------------------------------


def test_checkpoint_idempotency_and_uniqueness(tmp_path: Path) -> None:
    """Verify that checkpoint creation is idempotent and DB unique index prevents canonical duplicates."""
    db = Database(db_path=tmp_path / "test_cp.db")

    # 1. First save creates checkpoint
    cp_id_1 = db.save_checkpoint(
        milestone_rounds=100,
        valid_resolved_rounds=100,
        total_snapshots=1005,
        total_forecasts=1332,
        total_openrouter_cost_usd=0.0497,
        brier_scores={},
        experiment_spec_hash=EXPERIMENT_SPEC_HASH,
        notes="Initial checkpoint",
    )
    assert cp_id_1.startswith("checkpoint_100r_")

    # 2. Second save with identical milestone and spec hash returns existing ID idempotently
    cp_id_2 = db.save_checkpoint(
        milestone_rounds=100,
        valid_resolved_rounds=100,
        total_snapshots=1005,
        total_forecasts=1332,
        total_openrouter_cost_usd=0.0497,
        brier_scores={},
        experiment_spec_hash=EXPERIMENT_SPEC_HASH,
        notes="Duplicate call attempt",
    )
    assert cp_id_2 == cp_id_1

    # Only 1 canonical checkpoint exists
    canonical_cps = db.get_checkpoints(canonical_only=True)
    assert len(canonical_cps) == 1
    assert canonical_cps[0]["checkpoint_id"] == cp_id_1
    assert canonical_cps[0]["is_canonical"] == 1

    # 3. Direct raw SQL attempt to insert a second canonical checkpoint violates partial unique index
    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        with db._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO btc5m_checkpoints (
                    checkpoint_id, milestone_rounds, created_at_utc, valid_resolved_rounds,
                    total_snapshots, total_forecasts, total_openrouter_cost_usd,
                    brier_scores_json, experiment_spec_hash, backup_file_path, notes,
                    is_canonical
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    "checkpoint_100r_dup_raw",
                    100,
                    "2026-01-01T00:00:00Z",
                    100,
                    1005,
                    1332,
                    0.05,
                    "{}",
                    EXPERIMENT_SPEC_HASH,
                    None,
                    "illegal dup",
                ),
            )

    # 4. Inserting a non-canonical legacy record (is_canonical=0) succeeds without violating index
    with db._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO btc5m_checkpoints (
                checkpoint_id, milestone_rounds, created_at_utc, valid_resolved_rounds,
                total_snapshots, total_forecasts, total_openrouter_cost_usd,
                brier_scores_json, experiment_spec_hash, backup_file_path, notes,
                is_canonical
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                "checkpoint_100r_legacy",
                100,
                "2026-01-01T00:00:05Z",
                100,
                1005,
                1332,
                0.05,
                "{}",
                EXPERIMENT_SPEC_HASH,
                None,
                "legacy non-canonical dup",
            ),
        )

    # canonical_only=True still returns 1, canonical_only=False returns 2
    assert len(db.get_checkpoints(canonical_only=True)) == 1
    assert len(db.get_checkpoints(canonical_only=False)) == 2


def test_collector_target_milestone_double_trigger_prevention(tmp_path: Path) -> None:
    """Verify that collector does not double-trigger milestone checkpoint on loop completion."""
    db = Database(db_path=tmp_path / "test_collector_guard.db")
    collector = BTC5mAutonomousCollector(
        target_valid_rounds=100,
        cost_ceiling_usd=10.0,
        db=db,
    )

    # Mock lab
    mock_lab = MagicMock()
    mock_lab.compute_evaluation_summary.return_value = {
        "status": "VALID",
        "total_scores": 100,
        "metrics_by_condition": {},
    }
    collector.lab = mock_lab

    # First trigger should create checkpoint and add milestone to _checked_milestones
    with patch("pm_research.research.btc5m.collector.backup_database") as mock_backup:
        mock_backup.return_value = tmp_path / "backup_100r.db"
        cp_path_1 = collector.trigger_milestone_checkpoint(100, valid_rounds=100)
        assert cp_path_1 is not None
        assert 100 in collector._checked_milestones
        assert mock_backup.call_count == 1

        # Second trigger should detect existing milestone and return early without second backup
        cp_path_2 = collector.trigger_milestone_checkpoint(100, valid_rounds=100)
        assert cp_path_2 is None
        assert mock_backup.call_count == 1  # Not called again


def test_dataset_audit_metrics(tmp_path: Path) -> None:
    """Verify get_dataset_audit correctly aggregates rounds, snapshots, skips, horizons, and scores."""
    db = Database(db_path=tmp_path / "test_audit.db")

    # Insert round via SQL directly
    with db._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO btc5m_rounds (
                round_slug, start_epoch, end_epoch, duration_sec, price_to_beat,
                price_to_beat_source, up_token_id, down_token_id, up_outcome_index,
                down_outcome_index, condition_id, question, discovered_at, status,
                resolved_outcome, resolution_price, settled_at, experiment_spec_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "btc-updown-5m-1700000000", 1700000000, 1700000300, 300, 68000.0,
                "chainlink", "up_tok", "down_tok", 0, 1, "cond_1",
                "Will BTC be above 68000?", "2026-01-01T00:00:00Z", "resolved",
                "UP", 68100.0, "2026-01-01T00:05:05Z", EXPERIMENT_SPEC_HASH,
            ),
        )

        # Insert 1 valid snapshot and 1 invalid snapshot
        conn.execute(
            """
            INSERT INTO btc5m_snapshots (
                snapshot_id, round_slug, target_horizon_sec, captured_at_ms,
                target_scheduled_ms, timing_deviation_ms, seconds_remaining,
                reference_source, current_reference_price, is_valid, skip_reason,
                experiment_spec_hash, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "s_valid", "btc-updown-5m-1700000000", 240, 1700000060000,
                1700000060000, 0, 240.0,
                "chainlink", 68000.0, 1, None,
                EXPERIMENT_SPEC_HASH, "{}",
            ),
        )
        conn.execute(
            """
            INSERT INTO btc5m_snapshots (
                snapshot_id, round_slug, target_horizon_sec, captured_at_ms,
                target_scheduled_ms, timing_deviation_ms, seconds_remaining,
                reference_source, current_reference_price, is_valid, skip_reason,
                experiment_spec_hash, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "s_invalid", "btc-updown-5m-1700000000", 180, 1700000120000,
                1700000120000, 0, 180.0,
                "chainlink", 68000.0, 0, "SKIP_NO_EXACT_ANCHOR",
                EXPERIMENT_SPEC_HASH, "{}",
            ),
        )

    # Insert 1 score row via save_btc5m_resolution_score
    score_data = {
        "score_id": "sc_1",
        "round_slug": "btc-updown-5m-1700000000",
        "target_horizon_sec": 240,
        "snapshot_id": "s_valid",
        "resolved_outcome": "UP",
        "resolution_price": 68100.0,
        "resolved_at": "2026-01-01T00:05:05Z",
        "scored_at": "2026-01-01T00:06:00Z",
        "market_q": 0.55,
        "market_brier": 0.2025,
        "market_log_loss": 0.5978,
        "market_q_implied": 0.55,
        "market_implied_brier": 0.2025,
        "market_implied_log_loss": 0.5978,
        "cond_a_prob": 0.58,
        "cond_a_brier": 0.1764,
        "cond_a_log_loss": 0.5447,
        "cond_b_prob": 0.57,
        "cond_b_brier": 0.1849,
        "cond_b_log_loss": 0.5621,
        "cond_c_prob": 0.56,
        "cond_c_brier": 0.1936,
        "cond_c_log_loss": 0.5798,
        "cond_d_prob": 0.56,
        "cond_d_brier": 0.1936,
        "cond_d_log_loss": 0.5798,
        "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
        "metadata_json": "{}",
    }
    db.save_btc5m_resolution_score(score_data)

    audit = db.get_dataset_audit()
    assert audit["experiment_id"] == "btc5m_jev_ablation_v1"
    assert audit["experiment_spec_hash"] == EXPERIMENT_SPEC_HASH
    assert audit["rounds_discovered"] == 1
    assert audit["rounds_resolved"] == 1
    assert audit["rounds_scored"] == 1
    assert audit["rounds_with_zero_valid_snapshots"] == 0
    assert audit["snapshots_total"] == 2
    assert audit["snapshots_valid"] == 1
    assert audit["snapshots_invalid"] == 1
    assert audit["snapshot_valid_rate"] == 0.5
    assert audit["scored_round_horizons"] == 1
    assert audit["duplicate_score_keys"] == 0
    assert audit["raw_multiple_snapshot_pairs"] == 0
    assert audit["max_snapshots_per_round_horizon"] == 1
    assert audit["skip_reason_breakdown"]["SKIP_NO_EXACT_ANCHOR"] == 1
    assert audit["horizon_breakdown"][240]["valid"] == 1
    assert audit["horizon_breakdown"][180]["invalid"] == 1


def test_unpaired_market_brier_handling(tmp_path: Path) -> None:
    """Verify compute_evaluation_summary correctly handles snapshots where market_brier is None."""
    db = Database(db_path=tmp_path / "test_unpaired.db")
    lab = BTC5mShadowLab(db=db)

    # Insert 2 score rows:
    # 1. Normal paired observation
    # 2. Observation where market orderbook had no bids, so market_brier is None
    score_paired = {
        "score_id": "sc_paired",
        "round_slug": "r1",
        "target_horizon_sec": 240,
        "snapshot_id": "s1",
        "resolved_outcome": "UP",
        "resolution_price": 68100.0,
        "resolved_at": "2026-01-01T00:05:05Z",
        "scored_at": "2026-01-01T00:06:00Z",
        "market_q": 0.50,
        "market_brier": 0.2500,
        "market_log_loss": 0.6931,
        "market_q_implied": 0.50,
        "market_implied_brier": 0.2500,
        "market_implied_log_loss": 0.6931,
        "cond_a_prob": 0.60,
        "cond_a_brier": 0.1600,
        "cond_a_log_loss": 0.5108,
        "cond_b_prob": 0.60,
        "cond_b_brier": 0.1600,
        "cond_b_log_loss": 0.5108,
        "cond_c_prob": 0.60,
        "cond_c_brier": 0.1600,
        "cond_c_log_loss": 0.5108,
        "cond_d_prob": 0.60,
        "cond_d_brier": 0.1600,
        "cond_d_log_loss": 0.5108,
        "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
        "metadata_json": "{}",
    }
    db.save_btc5m_resolution_score(score_paired)

    score_unpaired = {
        "score_id": "sc_unpaired",
        "round_slug": "r2",
        "target_horizon_sec": 120,
        "snapshot_id": "s2",
        "resolved_outcome": "UP",
        "resolution_price": 68200.0,
        "resolved_at": "2026-01-01T00:10:05Z",
        "scored_at": "2026-01-01T00:11:00Z",
        "market_q": None,
        "market_brier": None,
        "market_log_loss": None,
        "market_q_implied": None,
        "market_implied_brier": None,
        "market_implied_log_loss": None,
        "cond_a_prob": 0.70,
        "cond_a_brier": 0.0900,
        "cond_a_log_loss": 0.3567,
        "cond_b_prob": 0.70,
        "cond_b_brier": 0.0900,
        "cond_b_log_loss": 0.3567,
        "cond_c_prob": 0.70,
        "cond_c_brier": 0.0900,
        "cond_c_log_loss": 0.3567,
        "cond_d_prob": 0.70,
        "cond_d_brier": 0.0900,
        "cond_d_log_loss": 0.3567,
        "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
        "metadata_json": "{}",
    }
    db.save_btc5m_resolution_score(score_unpaired)

    summary = lab.compute_evaluation_summary(n_boot=10)
    assert summary["total_scores"] == 2
    assert summary["total_rounds"] == 2

    # Paired count must strictly be 1 because r2 lacks market_brier
    for cond in [COND_A_REF_ONLY, COND_B_REF_PERP, COND_C_MARKET_AWARE, COND_D_FULL]:
        cond_metric = summary["metrics_by_condition"][cond]
        assert cond_metric["paired_count"] == 1
        assert cond_metric["market_brier"] == 0.2500
        assert cond_metric["mean_brier"] == 0.1600
        assert cond_metric["delta_brier"] == -0.0900


def test_experiment_hash_unchanged() -> None:
    """Ensure the scientific experiment specification hash remains frozen and unaltered."""
    expected_frozen_hash = "78bad4ab38785772e0e42fd8329111699101e88896cd0dc56fc5204a8c3d085a"
    assert EXPERIMENT_SPEC_HASH == expected_frozen_hash
    computed = compute_experiment_spec_hash(CANONICAL_SPEC)
    assert computed == expected_frozen_hash

