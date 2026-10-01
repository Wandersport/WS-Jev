#!/usr/bin/env python3
"""External operational supervisor for btc5m_leadlag_v2 network-only restarts.

This is a STANDALONE operational wrapper. It does NOT modify the collector,
experiment spec, or any scientific code. It monitors the collector PID,
classifies exit reasons, and auto-restarts ONLY on transient network/feed
silence watchdog aborts — after verifying all safety gates.

Usage:
    uv run python scripts/v2_network_supervisor.py --pid 16311

Safety properties:
    - NEVER restarts on parser exceptions, DB corruption, spec mismatch, etc.
    - Bounded restart rate: max 3 restarts within 60 minutes
    - Requires DNS/connectivity recovery before restart
    - Requires DB quick_check=ok, 0 ACTIVE rounds, consistent spec hash
    - Requires free disk >= 10 GiB above projected remaining requirement
    - Logs all decisions to data/logs/v2_supervisor.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "pm_research.db"
COLLECTOR_LOG = PROJECT_ROOT / "data" / "logs" / "btc5m_leadlag_v2_collector.log"
SUPERVISOR_LOG = PROJECT_ROOT / "data" / "logs" / "v2_supervisor.log"
PID_FILE = PROJECT_ROOT / "data" / "btc5m_leadlag_v2_collector.pid"

EXPERIMENT_ID = "btc5m_leadlag_v2"
FROZEN_SPEC_HASH = "dc8663876e8a1ca5f22d73f1e5f8bdd69fcf1a4946aba741708947c7e63c3352"
TARGET_TOTAL_ROUNDS = 500

MAX_RESTARTS_PER_WINDOW = 3
RESTART_WINDOW_SECONDS = 3600  # 60 minutes

# Network check endpoints (public, read-only, unauthenticated)
BINANCE_CHECK_HOST = "api.binance.com"
POLY_CHECK_HOST = "gamma-api.polymarket.com"

# Backoff schedule for network recovery waiting (seconds)
NETWORK_BACKOFF_SCHEDULE = [10, 20, 40, 60]

# Warmup seconds to wait after restart before checking health
WARMUP_SECONDS = 90

# How often to poll for process exit (seconds)
POLL_INTERVAL = 10.0

# Minimum free disk above projected remaining requirement (GiB)
MIN_DISK_HEADROOM_GIB = 10.0

# Estimated MiB per round (from observed ~135 MiB/round)
ESTIMATED_MIB_PER_ROUND = 140

# ---------------------------------------------------------------------------
# Network-only abort classification
# ---------------------------------------------------------------------------

# Regex patterns for NETWORK-ONLY watchdog abort reasons
_NETWORK_ONLY_PATTERNS = [
    re.compile(r"BINANCE_DEPTH_SILENCE_\d+S"),
    re.compile(r"BINANCE_TRADE_SILENCE_\d+S"),
    re.compile(r"POLY_FEED_SILENCE_\d+S"),
]

# Patterns that MUST NOT be present for network-only classification
_DISQUALIFYING_PATTERNS = [
    re.compile(r"EXCESSIVE_PARSER_EXCEPTIONS"),
    re.compile(r"BINANCE_TS_COV_.*BELOW"),
    re.compile(r"POLY_TS_COV_.*BELOW"),
    re.compile(r"TAKER_FLOW_COV_.*BELOW"),
]


class ExitClassification(NamedTuple):
    """Result of classifying a collector exit."""

    is_network_only: bool
    reason: str
    raw_failure_line: str


def classify_collector_exit(log_path: Path) -> ExitClassification:
    """Classify the most recent collector exit from its log file.

    Returns an ExitClassification indicating whether the exit was a
    network-only watchdog abort (restartable) or not.

    Classification rules:
    - Must find 'SUSTAINED HEALTH WATCHDOG ABORT' as the last abort line
    - All reasons in that line must match network-silence patterns
    - No disqualifying patterns (parser exceptions, coverage, etc.)
    - SIGINT/SIGTERM/KeyboardInterrupt are NOT network-only
    - Normal completion is NOT network-only
    - Unknown RuntimeError is NOT network-only
    """
    if not log_path.exists():
        return ExitClassification(
            is_network_only=False,
            reason="NO_LOG_FILE",
            raw_failure_line="",
        )

    # Read last 200 lines of the log to find the exit reason
    try:
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        tail_lines = lines[-200:] if len(lines) > 200 else lines
    except Exception as e:
        return ExitClassification(
            is_network_only=False,
            reason=f"LOG_READ_ERROR: {e}",
            raw_failure_line="",
        )

    # Check for manual stop signals
    for line in tail_lines:
        if "KeyboardInterrupt" in line or "SIGINT" in line or "SIGTERM" in line:
            return ExitClassification(
                is_network_only=False,
                reason="MANUAL_SIGNAL_STOP",
                raw_failure_line=line.strip(),
            )

    # Find the LAST watchdog abort line
    last_abort_line = ""
    for line in tail_lines:
        if "SUSTAINED HEALTH WATCHDOG ABORT" in line:
            last_abort_line = line.strip()

    if not last_abort_line:
        # Check if it was normal completion
        for line in tail_lines:
            if "physical rounds reached" in line.lower() or "cleanly closing collector" in line.lower():
                return ExitClassification(
                    is_network_only=False,
                    reason="NORMAL_COMPLETION",
                    raw_failure_line=line.strip(),
                )
        return ExitClassification(
            is_network_only=False,
            reason="UNKNOWN_EXIT_NO_ABORT_LINE",
            raw_failure_line="",
        )

    # Extract the reasons part after "Reasons: "
    reasons_match = re.search(r"Reasons:\s*(.+)$", last_abort_line)
    if not reasons_match:
        return ExitClassification(
            is_network_only=False,
            reason="ABORT_LINE_UNPARSEABLE",
            raw_failure_line=last_abort_line,
        )

    reasons_str = reasons_match.group(1).strip()
    reason_parts = [r.strip() for r in reasons_str.split(";") if r.strip()]

    # Check for disqualifying patterns first
    for part in reason_parts:
        for dq_pat in _DISQUALIFYING_PATTERNS:
            if dq_pat.search(part):
                return ExitClassification(
                    is_network_only=False,
                    reason=f"DISQUALIFIED_BY_{part}",
                    raw_failure_line=last_abort_line,
                )

    # Verify ALL reason parts match network-only patterns
    for part in reason_parts:
        matched = any(pat.search(part) for pat in _NETWORK_ONLY_PATTERNS)
        if not matched:
            return ExitClassification(
                is_network_only=False,
                reason=f"UNRECOGNIZED_REASON_{part}",
                raw_failure_line=last_abort_line,
            )

    return ExitClassification(
        is_network_only=True,
        reason=f"NETWORK_FEED_SILENCE: {reasons_str}",
        raw_failure_line=last_abort_line,
    )


# ---------------------------------------------------------------------------
# DB safety gates
# ---------------------------------------------------------------------------


class DBGateResult(NamedTuple):
    """Result of DB safety gate checks."""

    passed: bool
    quick_check_ok: bool
    completed_rounds: int
    active_rounds: int
    spec_hash_consistent: bool
    duplicate_slugs: int
    detail: str


def check_db_gates(db_path: Path) -> DBGateResult:
    """Verify all DB safety gates for restart eligibility.

    Gates:
    1. PRAGMA quick_check = ok
    2. 0 ACTIVE/incomplete rounds
    3. Consistent spec hash
    4. 0 duplicate round slugs
    5. Target not already reached
    """
    try:
        con = sqlite3.connect(str(db_path), timeout=30.0)
        con.execute("PRAGMA journal_mode=WAL")

        # quick_check
        qc = con.execute("PRAGMA quick_check").fetchone()
        quick_check_ok = qc is not None and qc[0] == "ok"
        if not quick_check_ok:
            return DBGateResult(False, False, 0, 0, False, 0, f"quick_check={qc}")

        # Round status counts
        rows = con.execute(
            "SELECT status, count(*) FROM leadlag_v2_rounds "
            "WHERE experiment_id = ? AND is_pilot = 0 GROUP BY status",
            (EXPERIMENT_ID,),
        ).fetchall()
        status_map = dict(rows)
        completed = status_map.get("COMPLETED", 0)
        active = status_map.get("ACTIVE", 0)

        if active > 0:
            con.close()
            return DBGateResult(False, True, completed, active, False, 0, f"ACTIVE_ROUNDS={active}")

        if completed >= TARGET_TOTAL_ROUNDS:
            con.close()
            return DBGateResult(False, True, completed, 0, False, 0, f"TARGET_REACHED={completed}")

        # Spec hash consistency
        hashes = con.execute(
            "SELECT DISTINCT experiment_spec_hash FROM leadlag_v2_rounds "
            "WHERE experiment_id = ? AND is_pilot = 0",
            (EXPERIMENT_ID,),
        ).fetchall()
        spec_consistent = len(hashes) <= 1 and (len(hashes) == 0 or hashes[0][0] == FROZEN_SPEC_HASH)

        if not spec_consistent:
            con.close()
            return DBGateResult(False, True, completed, 0, False, 0, f"SPEC_HASH_MISMATCH={hashes}")

        # Duplicate slugs
        dupes = con.execute(
            "SELECT count(*) FROM (SELECT round_slug FROM leadlag_v2_rounds "
            "WHERE experiment_id = ? AND is_pilot = 0 "
            "GROUP BY round_slug HAVING count(*) > 1)",
            (EXPERIMENT_ID,),
        ).fetchone()[0]

        if dupes > 0:
            con.close()
            return DBGateResult(False, True, completed, 0, True, dupes, f"DUPLICATE_SLUGS={dupes}")

        con.close()
        return DBGateResult(True, True, completed, 0, True, 0, f"COMPLETED={completed}")

    except Exception as e:
        return DBGateResult(False, False, 0, 0, False, 0, f"DB_ERROR: {e}")


# ---------------------------------------------------------------------------
# Disk space gate
# ---------------------------------------------------------------------------


def check_disk_gate(db_path: Path) -> tuple[bool, float, float, str]:
    """Check if free disk is sufficient for remaining rounds.

    Returns (passed, free_gib, required_gib, detail).
    """
    try:
        stat = shutil.disk_usage(str(db_path.parent))
        free_gib = stat.free / (1024**3)

        # Get completed rounds from DB
        con = sqlite3.connect(str(db_path), timeout=10.0)
        completed = con.execute(
            "SELECT count(*) FROM leadlag_v2_rounds "
            "WHERE experiment_id = ? AND is_pilot = 0 AND status = 'COMPLETED'",
            (EXPERIMENT_ID,),
        ).fetchone()[0]
        con.close()

        remaining = max(0, TARGET_TOTAL_ROUNDS - completed)
        projected_gib = (remaining * ESTIMATED_MIB_PER_ROUND) / 1024
        required_gib = projected_gib + MIN_DISK_HEADROOM_GIB

        passed = free_gib >= required_gib
        detail = f"free={free_gib:.1f}GiB remaining_rounds={remaining} projected={projected_gib:.1f}GiB required={required_gib:.1f}GiB"
        return passed, free_gib, required_gib, detail

    except Exception as e:
        return False, 0.0, 0.0, f"DISK_CHECK_ERROR: {e}"


# ---------------------------------------------------------------------------
# Network connectivity check
# ---------------------------------------------------------------------------


def check_network_connectivity() -> tuple[bool, str]:
    """Verify DNS resolution and basic TCP connectivity to both endpoints.

    Returns (all_reachable, detail).
    """
    results = {}
    for host in [BINANCE_CHECK_HOST, POLY_CHECK_HOST]:
        try:
            # DNS resolution
            addr = socket.getaddrinfo(host, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
            if not addr:
                results[host] = "DNS_NO_RESULTS"
                continue

            # TCP connect with 10s timeout
            sock = socket.create_connection((host, 443), timeout=10)
            sock.close()
            results[host] = "OK"

        except socket.gaierror as e:
            results[host] = f"DNS_FAIL: {e}"
        except socket.timeout:
            results[host] = "TCP_TIMEOUT"
        except OSError as e:
            results[host] = f"TCP_ERROR: {e}"

    all_ok = all(v == "OK" for v in results.values())
    detail = "; ".join(f"{h}={v}" for h, v in results.items())
    return all_ok, detail


def wait_for_network_recovery(logger: logging.Logger) -> bool:
    """Wait with bounded backoff until network connectivity recovers.

    Returns True if recovered, False if all retries exhausted.
    """
    for delay in NETWORK_BACKOFF_SCHEDULE:
        ok, detail = check_network_connectivity()
        if ok:
            logger.info(f"Network connectivity confirmed: {detail}")
            return True
        logger.info(f"Network not ready ({detail}), waiting {delay}s...")
        time.sleep(delay)

    # Final check
    ok, detail = check_network_connectivity()
    if ok:
        logger.info(f"Network connectivity confirmed on final check: {detail}")
        return True

    logger.error(f"Network recovery failed after full backoff: {detail}")
    return False


# ---------------------------------------------------------------------------
# Restart rate limiter
# ---------------------------------------------------------------------------


class RestartRateLimiter:
    """Tracks restart timestamps and enforces max-per-window limit."""

    def __init__(self, max_restarts: int = MAX_RESTARTS_PER_WINDOW,
                 window_seconds: int = RESTART_WINDOW_SECONDS) -> None:
        self.max_restarts = max_restarts
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def can_restart(self) -> bool:
        """Check if a restart is allowed under the rate limit."""
        now = time.time()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        return len(self._timestamps) < self.max_restarts

    def record_restart(self) -> None:
        """Record a restart event."""
        self._timestamps.append(time.time())

    @property
    def recent_count(self) -> int:
        """Number of restarts in the current window."""
        now = time.time()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        return len(self._timestamps)


# ---------------------------------------------------------------------------
# Collector restart
# ---------------------------------------------------------------------------


def restart_collector(logger: logging.Logger) -> tuple[bool, int | None, int | None]:
    """Restart the collector via the standard CLI entry point.

    Returns (success, new_collector_pid, caffeinate_pid).
    """
    cmd = [
        sys.executable, "-m", "pm_research.cli",
        "btc5m-leadlag-v2-run",
        "--target-physical-rounds", str(TARGET_TOTAL_ROUNDS),
    ]

    log_fd = open(str(COLLECTOR_LOG), "a")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            cwd=str(PROJECT_ROOT),
        )
    except Exception as e:
        logger.error(f"Failed to launch collector: {e}")
        log_fd.close()
        return False, None, None

    new_pid = proc.pid
    PID_FILE.write_text(f"{new_pid}\n")
    time.sleep(1.0)

    # Check it's still alive
    try:
        os.kill(new_pid, 0)
    except ProcessLookupError:
        logger.error(f"Collector PID {new_pid} died immediately after launch")
        return False, None, None

    # Attach caffeinate
    caff_pid: int | None = None
    try:
        caff_proc = subprocess.Popen(
            ["caffeinate", "-i", "-w", str(new_pid)],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        caff_pid = caff_proc.pid
        logger.info(f"Caffeinate attached: PID {caff_pid} -> collector PID {new_pid}")
    except Exception as e:
        logger.warning(f"caffeinate attach failed (non-fatal): {e}")

    return True, new_pid, caff_pid


def verify_post_restart_health(new_pid: int, logger: logging.Logger) -> tuple[bool, str]:
    """Wait for warmup and verify the restarted collector is healthy.

    Returns (healthy, detail).
    """
    logger.info(f"Waiting {WARMUP_SECONDS}s for collector warmup (PID {new_pid})...")
    time.sleep(WARMUP_SECONDS)

    # Check process is still alive
    try:
        os.kill(new_pid, 0)
    except ProcessLookupError:
        return False, f"Collector PID {new_pid} died during warmup"

    # Check heartbeat freshness from DB
    try:
        con = sqlite3.connect(str(DB_PATH), timeout=10.0)
        row = con.execute(
            "SELECT epoch_ms, extra_json FROM leadlag_v2_collector_heartbeat "
            "WHERE experiment_id = ? ORDER BY epoch_ms DESC LIMIT 1",
            (EXPERIMENT_ID,),
        ).fetchone()
        con.close()

        if row is None:
            return False, "No heartbeat found after warmup"

        age_sec = (time.time() * 1000 - row[0]) / 1000.0
        if age_sec > 30.0:
            return False, f"Heartbeat stale: {age_sec:.1f}s old"

        # Check watchdog status from extra_json
        if row[1]:
            extra = json.loads(row[1])
            watchdog = extra.get("health_watchdog_status", "UNKNOWN")
            parser_exc = extra.get("parser_exception_count", -1)
            if watchdog != "HEALTHY":
                return False, f"Watchdog={watchdog} after warmup"
            if parser_exc > 0:
                return False, f"Parser exceptions={parser_exc} after warmup"

        return True, f"HEALTHY (heartbeat age={age_sec:.1f}s)"

    except Exception as e:
        return False, f"Health check error: {e}"


# ---------------------------------------------------------------------------
# Supervisor event logging
# ---------------------------------------------------------------------------


def setup_supervisor_logger() -> logging.Logger:
    """Create a dedicated logger for supervisor events."""
    SUPERVISOR_LOG.parent.mkdir(parents=True, exist_ok=True)

    slog = logging.getLogger("v2_supervisor")
    slog.setLevel(logging.INFO)
    slog.propagate = False

    # File handler
    fh = logging.FileHandler(str(SUPERVISOR_LOG), mode="a")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    ))
    slog.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [SUPERVISOR] %(message)s",
        datefmt="%H:%M:%S",
    ))
    slog.addHandler(ch)

    return slog


def log_event(logger: logging.Logger, event: dict) -> None:
    """Write a structured JSON event to the supervisor log."""
    event["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    logger.info(json.dumps(event, default=str))


# ---------------------------------------------------------------------------
# Process monitoring
# ---------------------------------------------------------------------------


def is_pid_alive(pid: int) -> bool:
    """Check if a process is alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wait_for_exit(pid: int, logger: logging.Logger) -> None:
    """Poll until the given PID exits."""
    logger.info(f"Monitoring collector PID {pid} (polling every {POLL_INTERVAL}s)...")
    while is_pid_alive(pid):
        time.sleep(POLL_INTERVAL)
    logger.info(f"Collector PID {pid} has exited.")


# ---------------------------------------------------------------------------
# Main supervisor loop
# ---------------------------------------------------------------------------


def run_supervisor(initial_pid: int) -> int:
    """Main supervisor entry point.

    Monitors the given collector PID. On exit, classifies the reason and
    auto-restarts only if ALL safety gates pass for a network-only abort.

    Returns 0 on normal/clean exit, 1 on error requiring manual intervention.
    """
    slog = setup_supervisor_logger()
    rate_limiter = RestartRateLimiter()
    current_pid = initial_pid

    log_event(slog, {
        "event": "SUPERVISOR_START",
        "initial_pid": initial_pid,
        "target_rounds": TARGET_TOTAL_ROUNDS,
        "max_restarts": MAX_RESTARTS_PER_WINDOW,
        "restart_window_sec": RESTART_WINDOW_SECONDS,
    })

    # Verify initial PID is alive
    if not is_pid_alive(current_pid):
        slog.error(f"Initial PID {current_pid} is not alive. Exiting.")
        log_event(slog, {"event": "SUPERVISOR_EXIT", "reason": "INITIAL_PID_DEAD"})
        return 1

    while True:
        # Wait for current collector to exit
        wait_for_exit(current_pid, slog)

        log_event(slog, {"event": "COLLECTOR_EXIT_DETECTED", "pid": current_pid})

        # Small delay to let log flush
        time.sleep(2.0)

        # 1. Classify exit reason
        classification = classify_collector_exit(COLLECTOR_LOG)
        log_event(slog, {
            "event": "EXIT_CLASSIFIED",
            "is_network_only": classification.is_network_only,
            "reason": classification.reason,
            "raw_line": classification.raw_failure_line,
        })

        if not classification.is_network_only:
            slog.info(
                f"Exit is NOT network-only ({classification.reason}). "
                f"Supervisor stopping — manual intervention required."
            )
            log_event(slog, {"event": "SUPERVISOR_EXIT", "reason": f"NON_NETWORK_EXIT: {classification.reason}"})
            return 1

        # 2. Check restart rate limit
        if not rate_limiter.can_restart():
            slog.error(
                f"Restart rate limit reached ({rate_limiter.recent_count}/{MAX_RESTARTS_PER_WINDOW} "
                f"in {RESTART_WINDOW_SECONDS}s). Manual intervention required."
            )
            log_event(slog, {"event": "SUPERVISOR_EXIT", "reason": "RATE_LIMIT_EXCEEDED"})
            return 1

        # 3. DB safety gates
        db_result = check_db_gates(DB_PATH)
        log_event(slog, {
            "event": "DB_GATES_CHECK",
            "passed": db_result.passed,
            "completed_rounds": db_result.completed_rounds,
            "active_rounds": db_result.active_rounds,
            "spec_hash_consistent": db_result.spec_hash_consistent,
            "duplicate_slugs": db_result.duplicate_slugs,
            "detail": db_result.detail,
        })

        if not db_result.passed:
            slog.error(f"DB gates FAILED: {db_result.detail}. Manual intervention required.")
            log_event(slog, {"event": "SUPERVISOR_EXIT", "reason": f"DB_GATE_FAIL: {db_result.detail}"})
            return 1

        # 4. Disk space gate
        disk_ok, free_gib, required_gib, disk_detail = check_disk_gate(DB_PATH)
        log_event(slog, {
            "event": "DISK_GATE_CHECK",
            "passed": disk_ok,
            "free_gib": round(free_gib, 2),
            "required_gib": round(required_gib, 2),
            "detail": disk_detail,
        })

        if not disk_ok:
            slog.error(f"Disk gate FAILED: {disk_detail}. Manual intervention required.")
            log_event(slog, {"event": "SUPERVISOR_EXIT", "reason": f"DISK_GATE_FAIL: {disk_detail}"})
            return 1

        # 5. Wait for network recovery
        slog.info("Waiting for network connectivity to recover...")
        if not wait_for_network_recovery(slog):
            slog.error("Network recovery failed. Manual intervention required.")
            log_event(slog, {"event": "SUPERVISOR_EXIT", "reason": "NETWORK_RECOVERY_FAILED"})
            return 1

        # 6. All gates passed — restart
        rate_limiter.record_restart()
        old_pid = current_pid

        slog.info(
            f"All gates passed. Restarting collector "
            f"(restart {rate_limiter.recent_count}/{MAX_RESTARTS_PER_WINDOW}, "
            f"from round {db_result.completed_rounds})..."
        )

        success, new_pid, caff_pid = restart_collector(slog)
        if not success or new_pid is None:
            slog.error("Collector restart failed. Manual intervention required.")
            log_event(slog, {"event": "SUPERVISOR_EXIT", "reason": "RESTART_LAUNCH_FAILED"})
            return 1

        log_event(slog, {
            "event": "COLLECTOR_RESTARTED",
            "old_pid": old_pid,
            "new_pid": new_pid,
            "caffeinate_pid": caff_pid,
            "persisted_rounds": db_result.completed_rounds,
            "failure_reason": classification.reason,
            "restart_count": rate_limiter.recent_count,
        })

        # 7. Verify post-restart health
        healthy, health_detail = verify_post_restart_health(new_pid, slog)
        log_event(slog, {
            "event": "POST_RESTART_HEALTH",
            "healthy": healthy,
            "detail": health_detail,
        })

        if not healthy:
            slog.error(f"Post-restart health check failed: {health_detail}. Manual intervention required.")
            # Kill the unhealthy collector
            try:
                os.kill(new_pid, signal.SIGTERM)
            except Exception:
                pass
            log_event(slog, {"event": "SUPERVISOR_EXIT", "reason": f"POST_RESTART_UNHEALTHY: {health_detail}"})
            return 1

        slog.info(f"Collector restarted successfully: PID {new_pid}, {health_detail}")
        current_pid = new_pid
        # Continue monitoring loop


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="External network-only supervisor for btc5m_leadlag_v2 collector"
    )
    parser.add_argument(
        "--pid", type=int, required=True,
        help="PID of the currently running collector to monitor"
    )
    args = parser.parse_args()

    if args.pid <= 0:
        print(f"Error: invalid PID {args.pid}", file=sys.stderr)
        return 1

    return run_supervisor(args.pid)


if __name__ == "__main__":
    sys.exit(main())
