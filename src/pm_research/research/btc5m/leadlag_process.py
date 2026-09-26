"""Process lifecycle management for the detached Phase 8C lead-lag collector.

Provides:
- Detached process launching via subprocess.Popen(..., start_new_session=True)
- Single-instance locking via fcntl.flock on data/btc5m_leadlag_collector.lock
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

from pm_research.research.btc5m.leadlag_experiment import EXPERIMENT_ID, EXPERIMENT_SPEC_HASH
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILE: Path = Path("data/btc5m_leadlag_collector.lock")
DEFAULT_PID_FILE: Path = Path("data/btc5m_leadlag_collector.pid")
DEFAULT_LOG_FILE: Path = Path("data/logs/btc5m_leadlag_collector.log")

DEFAULT_LEADLAG_LOCK_FILE: Path = DEFAULT_LOCK_FILE
DEFAULT_LEADLAG_PID_FILE: Path = DEFAULT_PID_FILE
DEFAULT_LEADLAG_LOG_FILE: Path = DEFAULT_LOG_FILE

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


def is_leadlag_lock_held(lock_path: Path = DEFAULT_LOCK_FILE) -> bool:
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


def start_leadlag_collector(
    target_physical_rounds: int = 500,
    db_path: str | None = None,
    log_file: Path = DEFAULT_LOG_FILE,
    pid_file: Path = DEFAULT_PID_FILE,
) -> tuple[bool, str, int | None]:
    """Launch detached OS background lead-lag collector process."""
    if is_leadlag_lock_held():
        pid_held: int | None = None
        if pid_file.exists():
            try:
                pid_held = int(pid_file.read_text().strip())
            except Exception:
                pass
        return False, f"Collector already running (lock held by PID {pid_held})", pid_held

    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pm_research.cli",
        "btc5m-leadlag-run",
        "--target-physical-rounds",
        str(target_physical_rounds),
    ]
    if db_path:
        cmd.extend(["--db", db_path])

    log_fd = open(str(log_file), "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Fully detached from Antigravity session
        close_fds=True,
    )

    pid = proc.pid
    pid_file.write_text(f"{pid}\n")
    time.sleep(0.5)

    if not is_pid_alive(pid):
        return False, f"Collector process failed to launch (PID {pid} died immediately)", None

    return True, f"Collector launched successfully in background (PID {pid})", pid


def stop_leadlag_collector(
    pid_file: Path = DEFAULT_PID_FILE,
    lock_file: Path = DEFAULT_LOCK_FILE,
    timeout_sec: float = 10.0,
) -> tuple[bool, str]:
    """Gracefully terminate running lead-lag collector process."""
    if not pid_file.exists():
        return False, "No PID file found. Collector does not appear to be running."

    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        return False, "Failed to read PID from file."

    if not is_pid_alive(pid):
        try:
            pid_file.unlink(missing_ok=True)
            lock_file.unlink(missing_ok=True)
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
                pid_file.unlink(missing_ok=True)
                lock_file.unlink(missing_ok=True)
            except Exception:
                pass
            return True, f"Collector (PID {pid}) terminated cleanly."
        time.sleep(0.5)

    # Fallback to SIGKILL if still alive
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        pid_file.unlink(missing_ok=True)
        lock_file.unlink(missing_ok=True)
        return True, f"Collector (PID {pid}) forced killed after timeout."
    except Exception as e:
        return False, f"Failed to kill process {pid}: {e}"


def get_leadlag_collector_status(db_path: str | None = None) -> dict[str, Any]:
    """Get full operational and data status of the lead-lag collector."""
    db = Database(db_path or "data/pm_research.db")
    pid_file = DEFAULT_PID_FILE
    lock_file = DEFAULT_LOCK_FILE

    pid: int | None = None
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except Exception:
            pass

    proc_alive = is_pid_alive(pid) if pid else False
    lock_held = is_leadlag_lock_held(lock_file)

    latest_hb = db.get_leadlag_latest_heartbeat(EXPERIMENT_ID)
    hb_fresh = False
    now_epoch_ms = int(time.time() * 1000)
    hb_age_sec: float | None = None

    if latest_hb:
        hb_age_sec = (now_epoch_ms - latest_hb["epoch_ms"]) / 1000.0
        hb_fresh = hb_age_sec <= HEARTBEAT_STALE_THRESHOLD_SEC

    # Determine status categorization
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

    summary = db.get_leadlag_audit_summary(EXPERIMENT_ID)

    return {
        "collector_state": state,
        "process_alive": proc_alive,
        "pid": pid,
        "file_lock_held": lock_held,
        "heartbeat_fresh": hb_fresh,
        "heartbeat_age_sec": round(hb_age_sec, 1) if hb_age_sec is not None else None,
        "latest_heartbeat": latest_hb,
        "experiment_id": EXPERIMENT_ID,
        "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
        "summary": summary,
    }
