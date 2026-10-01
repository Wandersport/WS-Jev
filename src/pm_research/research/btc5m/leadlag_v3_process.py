"""Process lifecycle management for detached Phase 8E lead-lag v3 replication collector.

Provides:
- Detached process launching via subprocess.Popen(..., start_new_session=True)
- Single-instance locking via fcntl.flock on lock file
- PID tracking, verification, and termination
- Heartbeat freshness and collector health status
"""

from __future__ import annotations

import fcntl
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pm_research.research.btc5m.leadlag_v3_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_PILOT_ID,
    PILOT_PHYSICAL_ROUNDS,
    REPLICATION_DB_PATH,
    TARGET_PHYSICAL_ROUNDS,
)
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILE: Path = Path("data/btc5m_leadlag_v3_replication.lock")
DEFAULT_PID_FILE: Path = Path("data/btc5m_leadlag_v3_replication.pid")
DEFAULT_LOG_FILE: Path = Path("data/logs/btc5m_leadlag_v3_replication.log")

DEFAULT_PILOT_LOCK_FILE: Path = Path("data/btc5m_leadlag_v3_pilot.lock")
DEFAULT_PILOT_PID_FILE: Path = Path("data/btc5m_leadlag_v3_pilot.pid")
DEFAULT_PILOT_LOG_FILE: Path = Path("data/logs/btc5m_leadlag_v3_pilot.log")

HEARTBEAT_STALE_THRESHOLD_SEC: float = 30.0


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID is alive and accessible."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def is_leadlag_v3_lock_held(lock_path: Path = DEFAULT_LOCK_FILE) -> bool:
    """Check whether the collector lock file is currently held by an active process."""
    if not lock_path.exists():
        return False
    try:
        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except (BlockingIOError, OSError):
            return True
        finally:
            os.close(fd)
    except Exception:
        return False


def start_leadlag_v3_collector(
    target_physical_rounds: int = TARGET_PHYSICAL_ROUNDS,
    is_pilot: bool = False,
    db_path: str | None = None,
    log_file: Path | None = None,
    pid_file: Path | None = None,
    lock_file: Path | None = None,
) -> tuple[bool, str, int | None]:
    """Launch detached OS background lead-lag v3 replication collector process."""
    target_lock = lock_file or (DEFAULT_PILOT_LOCK_FILE if is_pilot else DEFAULT_LOCK_FILE)
    target_pid = pid_file or (DEFAULT_PILOT_PID_FILE if is_pilot else DEFAULT_PID_FILE)
    target_log = log_file or (DEFAULT_PILOT_LOG_FILE if is_pilot else DEFAULT_LOG_FILE)
    target_rounds = PILOT_PHYSICAL_ROUNDS if is_pilot else target_physical_rounds

    if is_leadlag_v3_lock_held(target_lock):
        pid_held: int | None = None
        if target_pid.exists():
            try:
                pid_held = int(target_pid.read_text().strip())
            except Exception:
                pass
        return False, f"Collector v3 already running (lock held by PID {pid_held})", pid_held

    target_log.parent.mkdir(parents=True, exist_ok=True)
    target_pid.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pm_research.cli",
        "btc5m-leadlag-v3-run",
        "--target-physical-rounds",
        str(target_rounds),
    ]
    if is_pilot:
        cmd.append("--pilot")
    if db_path:
        cmd.extend(["--db", db_path])

    log_fd = open(str(target_log), "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Fully detached from Antigravity session
        close_fds=True,
    )

    pid = proc.pid
    target_pid.write_text(f"{pid}\n")
    time.sleep(0.5)

    if not is_pid_alive(pid):
        return False, f"Collector process failed to launch (PID {pid} died immediately)", None

    mode_str = f"pilot mode ({target_rounds} rounds)" if is_pilot else f"production mode ({target_rounds} rounds)"
    return True, f"Collector v3 launched successfully in background {mode_str} (PID {pid})", pid


def stop_leadlag_v3_collector(
    is_pilot: bool = False,
    pid_file: Path | None = None,
    lock_file: Path | None = None,
    timeout_sec: float = 10.0,
) -> tuple[bool, str]:
    """Gracefully terminate background lead-lag v3 collector process via SIGTERM."""
    target_pid = pid_file or (DEFAULT_PILOT_PID_FILE if is_pilot else DEFAULT_PID_FILE)
    target_lock = lock_file or (DEFAULT_PILOT_LOCK_FILE if is_pilot else DEFAULT_LOCK_FILE)

    if not target_pid.exists():
        return False, "PID file not found; collector does not appear to be running."

    try:
        pid = int(target_pid.read_text().strip())
    except Exception as e:
        return False, f"Could not read PID file: {e}"

    if not is_pid_alive(pid):
        try:
            target_pid.unlink(missing_ok=True)
        except Exception:
            pass
        return True, f"Process {pid} is not running. Cleaned up stale PID file."

    logger.info("Sending SIGTERM to lead-lag v3 collector (PID %d)...", pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as e:
        return False, f"Failed to send SIGTERM to PID {pid}: {e}"

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not is_pid_alive(pid):
            break
        time.sleep(0.25)

    if is_pid_alive(pid):
        return False, f"Collector PID {pid} did not exit within {timeout_sec}s of SIGTERM."

    try:
        target_pid.unlink(missing_ok=True)
        target_lock.unlink(missing_ok=True)
    except Exception:
        pass

    return True, f"LeadLagCollectorV3 (PID {pid}) terminated gracefully."


def get_leadlag_v3_collector_status(
    db_path: str | Path | None = None,
    is_pilot: bool = False,
) -> dict[str, Any]:
    """Return comprehensive health status for the lead-lag v3 collector."""
    target_pid = DEFAULT_PILOT_PID_FILE if is_pilot else DEFAULT_PID_FILE
    target_lock = DEFAULT_PILOT_LOCK_FILE if is_pilot else DEFAULT_LOCK_FILE
    exp_id = EXPERIMENT_PILOT_ID if is_pilot else EXPERIMENT_ID
    target_db_path = Path(db_path) if db_path else REPLICATION_DB_PATH

    pid: int | None = None
    if target_pid.exists():
        try:
            pid = int(target_pid.read_text().strip())
        except Exception:
            pass

    pid_alive = is_pid_alive(pid) if pid else False
    lock_held = is_leadlag_v3_lock_held(target_lock)

    hb: dict[str, Any] | None = None
    if target_db_path.exists():
        try:
            db = Database(target_db_path)
            hb = db.get_leadlag_v2_latest_heartbeat(experiment_id=exp_id)
        except Exception as e:
            logger.warning(f"Could not read heartbeat from {target_db_path}: {e}")

    now_sec = time.time()
    hb_epoch = (hb.get("epoch_ms", 0) / 1000.0) if hb else 0.0
    hb_age_sec = (now_sec - hb_epoch) if hb_epoch > 0 else 999999.0
    hb_fresh = hb_age_sec <= HEARTBEAT_STALE_THRESHOLD_SEC

    return {
        "pid": pid,
        "pid_alive": pid_alive,
        "lock_held": lock_held,
        "heartbeat_age_sec": round(hb_age_sec, 1),
        "heartbeat_fresh": hb_fresh,
        "is_healthy": pid_alive and lock_held and hb_fresh,
        "latest_heartbeat": hb,
    }
