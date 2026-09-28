"""Process lifecycle management for the detached Phase 8C.2 lead-lag v2 collector.

Provides:
- Detached process launching via subprocess.Popen(..., start_new_session=True)
- Single-instance locking via fcntl.flock on data/btc5m_leadlag_v2_collector.lock (or pilot lock)
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

from pm_research.research.btc5m.leadlag_v2_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_PILOT_ID,
    EXPERIMENT_SPEC_HASH,
)
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILE: Path = Path("data/btc5m_leadlag_v2_collector.lock")
DEFAULT_PID_FILE: Path = Path("data/btc5m_leadlag_v2_collector.pid")
DEFAULT_LOG_FILE: Path = Path("data/logs/btc5m_leadlag_v2_collector.log")

DEFAULT_PILOT_LOCK_FILE: Path = Path("data/btc5m_leadlag_v2_pilot.lock")
DEFAULT_PILOT_PID_FILE: Path = Path("data/btc5m_leadlag_v2_pilot.pid")
DEFAULT_PILOT_LOG_FILE: Path = Path("data/logs/btc5m_leadlag_v2_pilot.log")

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


def is_leadlag_v2_lock_held(lock_path: Path = DEFAULT_LOCK_FILE) -> bool:
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


def start_leadlag_v2_collector(
    target_physical_rounds: int = 500,
    is_pilot: bool = False,
    db_path: str | None = None,
    log_file: Path | None = None,
    pid_file: Path | None = None,
    lock_file: Path | None = None,
) -> tuple[bool, str, int | None]:
    """Launch detached OS background lead-lag v2 collector process."""
    target_lock = lock_file or (DEFAULT_PILOT_LOCK_FILE if is_pilot else DEFAULT_LOCK_FILE)
    target_pid = pid_file or (DEFAULT_PILOT_PID_FILE if is_pilot else DEFAULT_PID_FILE)
    target_log = log_file or (DEFAULT_PILOT_LOG_FILE if is_pilot else DEFAULT_LOG_FILE)

    if is_leadlag_v2_lock_held(target_lock):
        pid_held: int | None = None
        if target_pid.exists():
            try:
                pid_held = int(target_pid.read_text().strip())
            except Exception:
                pass
        return False, f"Collector already running (lock held by PID {pid_held})", pid_held

    target_log.parent.mkdir(parents=True, exist_ok=True)
    target_pid.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pm_research.cli",
        "btc5m-leadlag-v2-run",
        "--target-physical-rounds",
        str(target_physical_rounds),
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

    mode_str = "pilot mode (3 rounds)" if is_pilot else f"production mode ({target_physical_rounds} rounds)"
    return True, f"Collector v2 launched successfully in background {mode_str} (PID {pid})", pid


def stop_leadlag_v2_collector(
    is_pilot: bool = False,
    pid_file: Path | None = None,
    lock_file: Path | None = None,
    timeout_sec: float = 10.0,
) -> tuple[bool, str]:
    """Gracefully terminate running lead-lag v2 collector process."""
    target_pid = pid_file or (DEFAULT_PILOT_PID_FILE if is_pilot else DEFAULT_PID_FILE)
    target_lock = lock_file or (DEFAULT_PILOT_LOCK_FILE if is_pilot else DEFAULT_LOCK_FILE)

    # If neither is specified, check both
    if pid_file is None:
        if not target_pid.exists() and DEFAULT_PID_FILE.exists():
            target_pid = DEFAULT_PID_FILE
            target_lock = DEFAULT_LOCK_FILE
        elif not target_pid.exists() and DEFAULT_PILOT_PID_FILE.exists():
            target_pid = DEFAULT_PILOT_PID_FILE
            target_lock = DEFAULT_PILOT_LOCK_FILE

    if not target_pid.exists():
        return False, "No PID file found. Collector does not appear to be running."

    try:
        pid = int(target_pid.read_text().strip())
    except Exception:
        return False, "Failed to read PID from file."

    if not is_pid_alive(pid):
        try:
            target_pid.unlink(missing_ok=True)
            target_lock.unlink(missing_ok=True)
        except Exception:
            pass
        return True, f"Process {pid} was already dead. Cleaned up stale files."

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True, f"Process {pid} already exited."
    except Exception as e:
        return False, f"Failed to send SIGTERM to process {pid}: {e}"

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not is_pid_alive(pid):
            try:
                target_pid.unlink(missing_ok=True)
                target_lock.unlink(missing_ok=True)
            except Exception:
                pass
            return True, f"Collector v2 (PID {pid}) terminated cleanly."
        time.sleep(0.5)

    # Fallback to SIGKILL
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        target_pid.unlink(missing_ok=True)
        target_lock.unlink(missing_ok=True)
        return True, f"Collector v2 (PID {pid}) forced killed after timeout."
    except Exception as e:
        return False, f"Failed to kill process {pid}: {e}"


def get_leadlag_v2_collector_status(
    db_path: str | None = None,
    is_pilot: bool = False,
) -> dict[str, Any]:
    """Get full operational and data status of the lead-lag v2 collector."""
    db = Database(db_path or "data/pm_research.db")
    exp_id = EXPERIMENT_PILOT_ID if is_pilot else EXPERIMENT_ID
    pid_file = DEFAULT_PILOT_PID_FILE if is_pilot else DEFAULT_PID_FILE
    lock_file = DEFAULT_PILOT_LOCK_FILE if is_pilot else DEFAULT_LOCK_FILE

    pid: int | None = None
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except Exception:
            pass

    proc_alive = is_pid_alive(pid) if pid else False
    lock_held = is_leadlag_v2_lock_held(lock_file)

    latest_hb = db.get_leadlag_v2_latest_heartbeat(exp_id)
    hb_fresh = False
    now_epoch_ms = int(time.time() * 1000)
    hb_age_sec: float | None = None

    if latest_hb:
        hb_age_sec = (now_epoch_ms - latest_hb["epoch_ms"]) / 1000.0
        hb_fresh = hb_age_sec <= HEARTBEAT_STALE_THRESHOLD_SEC

    # Status classification
    if proc_alive and lock_held and hb_fresh:
        state = "RUNNING_HEALTHY"
    elif proc_alive and lock_held and not hb_fresh:
        state = "RUNNING_STALE_HEARTBEAT"
    elif not proc_alive and lock_held:
        state = "CRASHED_LOCK_HELD"
    elif not proc_alive and not lock_held:
        state = "STOPPED"
    else:
        state = "UNKNOWN"

    summary = db.get_leadlag_v2_audit_summary(exp_id)

    return {
        "collector_state": state,
        "process_alive": proc_alive,
        "pid": pid,
        "file_lock_held": lock_held,
        "heartbeat_fresh": hb_fresh,
        "heartbeat_age_sec": round(hb_age_sec, 1) if hb_age_sec is not None else None,
        "latest_heartbeat": latest_hb,
        "experiment_id": exp_id,
        "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
        "is_pilot": is_pilot,
        "summary": summary,
    }
