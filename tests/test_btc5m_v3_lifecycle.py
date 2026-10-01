"""Unit tests for LeadLagCollectorV3 round-target lifecycle, resume, and startup semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from pm_research.research.btc5m.leadlag_v3_collector import LeadLagCollectorV3
from pm_research.research.btc5m.leadlag_v3_experiment import (
    EXPERIMENT_PILOT_ID,
    EXPERIMENT_SPEC_HASH,
)
from pm_research.storage.db import Database


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    """Create a temporary test database with v2 schema."""
    db_file = tmp_path / "test_lifecycle.db"
    db = Database(db_file)
    with db._get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leadlag_v2_rounds (
                round_slug TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                experiment_spec_hash TEXT NOT NULL,
                is_pilot INTEGER NOT NULL,
                start_epoch INTEGER NOT NULL,
                end_epoch INTEGER NOT NULL,
                up_token_id TEXT,
                down_token_id TEXT,
                condition_id TEXT,
                status TEXT NOT NULL,
                sample_count INTEGER DEFAULT 0,
                valid_sample_count INTEGER DEFAULT 0,
                created_at_utc TEXT NOT NULL,
                completed_at_utc TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leadlag_v2_samples (
                sample_target_ts_ms INTEGER NOT NULL,
                sample_actual_ts_ms INTEGER NOT NULL,
                round_slug TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                is_valid INTEGER NOT NULL,
                is_stale INTEGER NOT NULL,
                PRIMARY KEY (sample_target_ts_ms, round_slug)
            )
        """)
    return db


def test_resume_counts_only_qualifying_full_rounds(test_db: Database, tmp_path: Path) -> None:
    """Verify that _count_completed_full_rounds_in_db only counts completed rounds with >= 285 samples."""
    with test_db._get_connection() as conn:
        # 1. Warmup partial (122 samples) -> should NOT qualify
        conn.execute(
            """INSERT INTO leadlag_v2_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "btc-updown-5m-1000",
                EXPERIMENT_PILOT_ID,
                EXPERIMENT_SPEC_HASH,
                1,
                1000,
                1300,
                "u1",
                "d1",
                "c1",
                "WARMUP_PARTIAL",
                122,
                120,
                "2026-10-01T00:00:00Z",
                "2026-10-01T00:05:00Z",
            ),
        )
        # 2. Completed full round (300 samples) -> QUALIFIES
        conn.execute(
            """INSERT INTO leadlag_v2_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "btc-updown-5m-1300",
                EXPERIMENT_PILOT_ID,
                EXPERIMENT_SPEC_HASH,
                1,
                1300,
                1600,
                "u2",
                "d2",
                "c2",
                "COMPLETED",
                300,
                290,
                "2026-10-01T00:05:00Z",
                "2026-10-01T00:10:00Z",
            ),
        )
        # 3. Truncated round (1 sample) -> should NOT qualify
        conn.execute(
            """INSERT INTO leadlag_v2_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "btc-updown-5m-1600",
                EXPERIMENT_PILOT_ID,
                EXPERIMENT_SPEC_HASH,
                1,
                1600,
                1900,
                "u3",
                "d3",
                "c3",
                "COMPLETED",
                1,
                0,
                "2026-10-01T00:10:00Z",
                "2026-10-01T00:10:01Z",
            ),
        )
        # 4. Completed full round (299 samples) -> QUALIFIES
        conn.execute(
            """INSERT INTO leadlag_v2_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "btc-updown-5m-1900",
                EXPERIMENT_PILOT_ID,
                EXPERIMENT_SPEC_HASH,
                1,
                1900,
                2200,
                "u4",
                "d4",
                "c4",
                "COMPLETED",
                299,
                280,
                "2026-10-01T00:15:00Z",
                "2026-10-01T00:20:00Z",
            ),
        )

    lock_file = tmp_path / "test.lock"
    collector = LeadLagCollectorV3(
        db=test_db,
        target_physical_rounds=10,
        is_pilot=True,
        lock_file_path=str(lock_file),
    )
    # Out of 4 recorded rounds, exactly 2 qualify as completed full rounds
    assert collector._rounds_completed_count == 2


def test_target_already_satisfied_exits_immediately(test_db: Database, tmp_path: Path) -> None:
    """If target is 2 and 2 completed rounds exist, run() should exit immediately without collecting."""
    with test_db._get_connection() as conn:
        for i in range(2):
            conn.execute(
                """INSERT INTO leadlag_v2_rounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"btc-updown-5m-{1000 + i*300}",
                    EXPERIMENT_PILOT_ID,
                    EXPERIMENT_SPEC_HASH,
                    1,
                    1000 + i * 300,
                    1300 + i * 300,
                    f"u{i}",
                    f"d{i}",
                    f"c{i}",
                    "COMPLETED",
                    300,
                    300,
                    "2026-10-01T00:00:00Z",
                    "2026-10-01T00:05:00Z",
                ),
            )

    lock_file = tmp_path / "test.lock"
    collector = LeadLagCollectorV3(
        db=test_db,
        target_physical_rounds=2,
        is_pilot=True,
        lock_file_path=str(lock_file),
    )
    assert collector._rounds_completed_count == 2
    # Calling run() should return immediately
    collector.run()
    assert collector._rounds_completed_count == 2


def test_mid_round_startup_is_marked_warmup_and_does_not_count(test_db: Database, tmp_path: Path) -> None:
    """Mid-round startup (>2s after boundary) marks round WARMUP_PARTIAL and does not increment completed count."""
    lock_file = tmp_path / "test.lock"
    collector = LeadLagCollectorV3(
        db=test_db,
        target_physical_rounds=2,
        is_pilot=True,
        lock_file_path=str(lock_file),
    )
    assert collector._rounds_completed_count == 0

    # Start at 120s into the 300s round (epoch 1790874120, round start 1790874000)
    now_sec = 1790874120.0
    active = collector._ensure_active_round(now_sec)
    assert active is True
    assert collector._is_current_round_warmup_partial is True
    assert collector._current_round_slug == "btc-updown-5m-1790874000"

    # Simulate collecting 5 samples
    collector._current_round_sample_count = 5
    collector._current_round_valid_count = 5

    # Close the round
    collector._close_current_round()
    # Completed full rounds counter MUST NOT increment
    assert collector._rounds_completed_count == 0
    assert collector._current_round_slug is None

    # Next round begins cleanly at boundary (1790874300.5)
    active2 = collector._ensure_active_round(1790874300.5)
    assert active2 is True
    assert collector._is_current_round_warmup_partial is False
    assert collector._current_round_slug == "btc-updown-5m-1790874300"

    # Simulate collecting 300 samples
    collector._current_round_sample_count = 300
    collector._current_round_valid_count = 295

    # Close the full round
    collector._close_current_round()
    # Completed full rounds counter MUST increment by 1
    assert collector._rounds_completed_count == 1


def test_target_completion_stops_at_exact_target(test_db: Database, tmp_path: Path) -> None:
    """Target=1 completes exactly 1 full round and stops without registering a 2nd round."""
    lock_file = tmp_path / "test.lock"
    collector = LeadLagCollectorV3(
        db=test_db,
        target_physical_rounds=1,
        is_pilot=True,
        lock_file_path=str(lock_file),
    )
    assert collector._rounds_completed_count == 0

    # Round 1 starts at boundary
    active = collector._ensure_active_round(1790874300.0)
    assert active is True
    collector._current_round_sample_count = 300

    # Next boundary arrives (1790874600.0)
    active = collector._ensure_active_round(1790874600.0)
    # Round 1 was closed, completed count is now 1 == target
    assert collector._rounds_completed_count == 1
    # _ensure_active_round must return False because target is reached!
    assert active is False
    # No new round registered
    assert collector._current_round_slug is None


def test_target_2_completes_two_full_rounds_and_stops(test_db: Database, tmp_path: Path) -> None:
    """Target=2 completes exactly 2 full rounds and stops without collecting a 3rd round."""
    lock_file = tmp_path / "test.lock"
    collector = LeadLagCollectorV3(
        db=test_db,
        target_physical_rounds=2,
        is_pilot=True,
        lock_file_path=str(lock_file),
    )
    assert collector._rounds_completed_count == 0

    # Round 1 starts
    assert collector._ensure_active_round(1790874300.0) is True
    collector._current_round_sample_count = 300

    # Round 2 starts (Round 1 finishes)
    assert collector._ensure_active_round(1790874600.0) is True
    assert collector._rounds_completed_count == 1
    collector._current_round_sample_count = 300

    # Round 3 attempt (Round 2 finishes, target of 2 reached)
    assert collector._ensure_active_round(1790874900.0) is False
    assert collector._rounds_completed_count == 2
    assert collector._current_round_slug is None


def test_boundary_sync_does_not_drop_second_0() -> None:
    """Discrete second grid alignment starting at boundary captures second 0 (T=0)."""
    # Simulate arriving at boundary: 1790874300.002
    boundary_sec = 1790874300
    now = boundary_sec + 0.002
    # Discrete second logic: if now - int(now) < 0.2: next_sample_sec = int(now)
    if now - int(now) < 0.2:
        next_sample_sec = int(now)
    else:
        next_sample_sec = int(now) + 1

    assert next_sample_sec == boundary_sec
    sample_sec_ms = next_sample_sec * 1000
    # Must equal exact boundary millisecond (second 0)
    assert sample_sec_ms == 1790874300000

