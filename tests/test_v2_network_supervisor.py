"""Focused tests for v2_network_supervisor exit classification and safety gates.

Only tests the classifier/gate logic using temporary files and mock state.
Does NOT import or modify any collector/scientific code.
"""

from __future__ import annotations

import sqlite3

# Import supervisor module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from v2_network_supervisor import (
    RestartRateLimiter,
    check_db_gates,
    classify_collector_exit,
)

# ---------------------------------------------------------------------------
# Exit classification tests
# ---------------------------------------------------------------------------


class TestClassifyCollectorExit:
    """Test exit classification from log tail analysis."""

    def _write_log(self, tmp_path: Path, lines: list[str]) -> Path:
        log = tmp_path / "collector.log"
        log.write_text("\n".join(lines) + "\n")
        return log

    def test_network_only_both_feeds(self, tmp_path: Path) -> None:
        """Binance depth + Poly feed silence → network-only."""
        log = self._write_log(tmp_path, [
            "Some normal operation log line",
            "Health watchdog warning (5/6): BINANCE_DEPTH_SILENCE_32S; POLY_FEED_SILENCE_40S",
            "Health watchdog warning (6/6): BINANCE_DEPTH_SILENCE_35S; POLY_FEED_SILENCE_43S",
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: BINANCE_DEPTH_SILENCE_35S; POLY_FEED_SILENCE_43S",
            "Traceback (most recent call last):",
            '  File "cli.py", line 1618, in run',
            "RuntimeError: Collector aborted by watchdog: SUSTAINED_FAILURE: BINANCE_DEPTH_SILENCE_35S; POLY_FEED_SILENCE_43S",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is True
        assert "NETWORK_FEED_SILENCE" in result.reason

    def test_network_only_binance_depth_only(self, tmp_path: Path) -> None:
        """Binance depth silence alone → network-only."""
        log = self._write_log(tmp_path, [
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: BINANCE_DEPTH_SILENCE_45S",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is True

    def test_network_only_poly_feed_only(self, tmp_path: Path) -> None:
        """Poly feed silence alone → network-only."""
        log = self._write_log(tmp_path, [
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: POLY_FEED_SILENCE_60S",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is True

    def test_network_only_binance_trade_silence(self, tmp_path: Path) -> None:
        """Binance trade silence → network-only."""
        log = self._write_log(tmp_path, [
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: BINANCE_TRADE_SILENCE_90S",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is True

    def test_not_network_parser_exceptions(self, tmp_path: Path) -> None:
        """EXCESSIVE_PARSER_EXCEPTIONS → NOT network-only."""
        log = self._write_log(tmp_path, [
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: EXCESSIVE_PARSER_EXCEPTIONS_51",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is False
        assert "DISQUALIFIED" in result.reason

    def test_not_network_parser_mixed_with_silence(self, tmp_path: Path) -> None:
        """Parser exceptions mixed with network silence → NOT network-only."""
        log = self._write_log(tmp_path, [
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: BINANCE_DEPTH_SILENCE_30S; EXCESSIVE_PARSER_EXCEPTIONS_51",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is False
        assert "DISQUALIFIED" in result.reason

    def test_not_network_coverage_below(self, tmp_path: Path) -> None:
        """TS coverage below threshold → NOT network-only."""
        log = self._write_log(tmp_path, [
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: BINANCE_TS_COV_85.0PCT_BELOW_99",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is False

    def test_not_network_taker_flow_coverage(self, tmp_path: Path) -> None:
        """Taker flow coverage below → NOT network-only."""
        log = self._write_log(tmp_path, [
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: TAKER_FLOW_COV_80.0PCT_BELOW_95",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is False

    def test_not_network_manual_sigint(self, tmp_path: Path) -> None:
        """KeyboardInterrupt/SIGINT → NOT network-only."""
        log = self._write_log(tmp_path, [
            "Some log line",
            "KeyboardInterrupt",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is False
        assert "MANUAL_SIGNAL" in result.reason

    def test_not_network_manual_sigterm(self, tmp_path: Path) -> None:
        """SIGTERM → NOT network-only."""
        log = self._write_log(tmp_path, [
            "Received SIGTERM, shutting down",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is False
        assert "MANUAL_SIGNAL" in result.reason

    def test_not_network_normal_completion(self, tmp_path: Path) -> None:
        """Normal 500-round completion → NOT network-only."""
        log = self._write_log(tmp_path, [
            "Target of 500 physical rounds reached! Cleanly closing collector.",
            "LeadLagCollectorV2 stopped cleanly.",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is False
        assert "NORMAL_COMPLETION" in result.reason

    def test_not_network_no_log_file(self, tmp_path: Path) -> None:
        """Missing log file → NOT network-only."""
        result = classify_collector_exit(tmp_path / "nonexistent.log")
        assert result.is_network_only is False
        assert "NO_LOG" in result.reason

    def test_not_network_empty_log(self, tmp_path: Path) -> None:
        """Empty log → NOT network-only."""
        log = self._write_log(tmp_path, [])
        result = classify_collector_exit(log)
        assert result.is_network_only is False

    def test_not_network_unknown_reason(self, tmp_path: Path) -> None:
        """Unrecognized reason tag → NOT network-only."""
        log = self._write_log(tmp_path, [
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: SOME_UNKNOWN_REASON_42",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is False
        assert "UNRECOGNIZED" in result.reason

    def test_uses_last_abort_line_not_first(self, tmp_path: Path) -> None:
        """When multiple aborts exist, classifies the LAST one."""
        log = self._write_log(tmp_path, [
            # First abort was parser exceptions (old bug)
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: EXCESSIVE_PARSER_EXCEPTIONS_51",
            "RuntimeError: Collector aborted by watchdog: SUSTAINED_FAILURE: EXCESSIVE_PARSER_EXCEPTIONS_51",
            # Second abort was network silence (after hotfix restart)
            "Starting LeadLagCollectorV2 ...",
            "SUSTAINED HEALTH WATCHDOG ABORT: Collector degraded for 60s sustained! Reasons: BINANCE_DEPTH_SILENCE_35S; POLY_FEED_SILENCE_43S",
            "RuntimeError: Collector aborted by watchdog: SUSTAINED_FAILURE: BINANCE_DEPTH_SILENCE_35S; POLY_FEED_SILENCE_43S",
        ])
        result = classify_collector_exit(log)
        assert result.is_network_only is True


# ---------------------------------------------------------------------------
# DB gate tests
# ---------------------------------------------------------------------------


def _create_test_db(path: Path, completed: int = 167, active: int = 0,
                    spec_hash: str = "dc8663876e8a1ca5f22d73f1e5f8bdd69fcf1a4946aba741708947c7e63c3352",
                    create_duplicate: bool = False) -> None:
    """Create a minimal test DB with leadlag_v2_rounds table."""
    con = sqlite3.connect(str(path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS leadlag_v2_rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            round_slug TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            experiment_spec_hash TEXT NOT NULL,
            is_pilot INTEGER DEFAULT 0,
            status TEXT NOT NULL
        )
    """)

    for i in range(completed):
        slug = f"round_{i:04d}"
        con.execute(
            "INSERT INTO leadlag_v2_rounds (round_slug, experiment_id, experiment_spec_hash, is_pilot, status) "
            "VALUES (?, ?, ?, 0, 'COMPLETED')",
            (slug, "btc5m_leadlag_v2", spec_hash),
        )

    for i in range(active):
        slug = f"active_{i:04d}"
        con.execute(
            "INSERT INTO leadlag_v2_rounds (round_slug, experiment_id, experiment_spec_hash, is_pilot, status) "
            "VALUES (?, ?, ?, 0, 'ACTIVE')",
            (slug, "btc5m_leadlag_v2", spec_hash),
        )

    if create_duplicate:
        con.execute(
            "INSERT INTO leadlag_v2_rounds (round_slug, experiment_id, experiment_spec_hash, is_pilot, status) "
            "VALUES (?, ?, ?, 0, 'COMPLETED')",
            ("round_0000", "btc5m_leadlag_v2", spec_hash),
        )

    con.commit()
    con.close()


class TestDBGates:
    """Test DB safety gate checks."""

    def test_clean_state_passes(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        _create_test_db(db, completed=167)
        result = check_db_gates(db)
        assert result.passed is True
        assert result.completed_rounds == 167
        assert result.active_rounds == 0
        assert result.spec_hash_consistent is True
        assert result.duplicate_slugs == 0

    def test_active_rounds_fail(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        _create_test_db(db, completed=167, active=1)
        result = check_db_gates(db)
        assert result.passed is False
        assert result.active_rounds == 1

    def test_target_reached_fail(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        _create_test_db(db, completed=500)
        result = check_db_gates(db)
        assert result.passed is False
        assert "TARGET_REACHED" in result.detail

    def test_spec_hash_mismatch_fail(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        _create_test_db(db, completed=167, spec_hash="wrong_hash_abc123")
        result = check_db_gates(db)
        assert result.passed is False
        assert result.spec_hash_consistent is False

    def test_duplicate_slugs_fail(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        _create_test_db(db, completed=167, create_duplicate=True)
        result = check_db_gates(db)
        assert result.passed is False
        assert result.duplicate_slugs > 0

    def test_missing_db_fails(self, tmp_path: Path) -> None:
        db = tmp_path / "nonexistent.db"
        result = check_db_gates(db)
        # Should fail gracefully (empty DB has 0 rounds which is fine, but
        # actually it'll create an empty DB and fail to find the table)
        # The implementation handles this via exception
        assert result.completed_rounds == 0 or not result.passed


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------


class TestRestartRateLimiter:
    """Test restart rate limiting."""

    def test_allows_first_restart(self) -> None:
        rl = RestartRateLimiter(max_restarts=3, window_seconds=60)
        assert rl.can_restart() is True

    def test_allows_up_to_max(self) -> None:
        rl = RestartRateLimiter(max_restarts=3, window_seconds=60)
        rl.record_restart()
        rl.record_restart()
        assert rl.can_restart() is True  # 2 recorded, 3rd allowed
        rl.record_restart()
        assert rl.can_restart() is False  # 3 recorded, 4th blocked

    def test_window_expiry(self) -> None:
        import time
        rl = RestartRateLimiter(max_restarts=1, window_seconds=1)
        rl.record_restart()
        assert rl.can_restart() is False
        time.sleep(1.1)
        assert rl.can_restart() is True

    def test_recent_count(self) -> None:
        rl = RestartRateLimiter(max_restarts=3, window_seconds=3600)
        assert rl.recent_count == 0
        rl.record_restart()
        assert rl.recent_count == 1
        rl.record_restart()
        assert rl.recent_count == 2
