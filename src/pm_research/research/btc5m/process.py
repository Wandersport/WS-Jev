"""Process management and single-instance concurrency lock for BTC 5-minute collector.

Provides:
- Non-blocking exclusive file lock via fcntl.flock on data/collector.lock
- Project-owned detached process launcher (subprocess.Popen, start_new_session=True)
- PID tracking and verification (matching PID with collector command)
- Heartbeat freshness evaluation (stale threshold: 120 seconds)
- Comprehensive status categorization:
  RUNNING_HEALTHY, RUNNING_STALE_HEARTBEAT, PROCESS_DEAD, NOT_STARTED,
  TARGET_REACHED, COST_GUARD_TRIGGERED
- Two-tier graceful shutdown (stop file signal -> bounded wait -> SIGTERM fallback)
- Safe logging redirection to Git-ignored data/logs/btc5m_collector.log
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

DEFAULT_LOCK_FILE: Path = Path("data/collector.lock")
DEFAULT_PID_FILE: Path = Path("data/collector.pid")
DEFAULT_STATUS_FILE: Path = Path("data/collector_status.json")
DEFAULT_STOP_FILE: Path = Path("data/collector.stop")
DEFAULT_LOG_FILE: Path = Path("data/logs/btc5m_collector.log")

HEARTBEAT_STALE_THRESHOLD_SEC: float = 120.0


class CollectorLock:
    """Operating-system level exclusive file lock preventing multiple collector instances.

    Uses non-blocking fcntl.flock (LOCK_EX | LOCK_NB).
    Releases automatically upon process exit or crash via OS file table closure.
    """

    def __init__(self, lock_file: Path | str = DEFAULT_LOCK_FILE) -> None:
        self.lock_file = Path(lock_file)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Attempt to acquire exclusive non-blocking lock.

        Returns True if acquired, False if already held by another process.
        """
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.lock_file), os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, OSError):
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except Exception:
                    pass
                self._fd = None
            return False

    def release(self) -> None:
        """Release the file lock and close file descriptor."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None

    def is_locked_by_other(self) -> bool:
        """Check if lock is currently held by another process without acquiring it permanently."""
        if not self.lock_file.exists():
            return False
        try:
            test_fd = os.open(str(self.lock_file), os.O_RDWR, 0o644)
            try:
                fcntl.flock(test_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(test_fd, fcntl.LOCK_UN)
                return False
            except (BlockingIOError, OSError):
                return True
            finally:
                os.close(test_fd)
        except Exception:
            return False


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID exists in the OS process table."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def verify_collector_process(pid: int) -> bool:
    """Verify that a given PID is alive and corresponds to the WS-Jev collector."""
    if not is_pid_alive(pid):
        return False
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        # Verify that process contains pm_research or btc5m
        if any(token in output for token in ("pm_research", "btc5m", "collector")):
            return True
        return False
    except Exception:
        # Fallback to liveness check if ps query fails
        return is_pid_alive(pid)


def read_collector_pid(pid_file: Path | str = DEFAULT_PID_FILE) -> int | None:
    """Read the stored collector PID if file exists and contains a valid integer."""
    p_path = Path(pid_file)
    if not p_path.exists():
        return None
    try:
        content = p_path.read_text(encoding="utf-8").strip()
        # Content format: PID or PID\ntimestamp
        first_line = content.split("\n")[0].strip()
        return int(first_line)
    except Exception:
        return None


def write_collector_pid(
    pid: int,
    pid_file: Path | str = DEFAULT_PID_FILE,
) -> None:
    """Write current PID and ISO timestamp to PID file."""
    p_path = Path(pid_file)
    p_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    p_path.write_text(f"{pid}\nstarted_at={now_iso}\n", encoding="utf-8")


def remove_collector_pid(pid_file: Path | str = DEFAULT_PID_FILE) -> None:
    """Safely remove PID file if it exists."""
    p_path = Path(pid_file)
    if p_path.exists():
        try:
            p_path.unlink()
        except Exception:
            pass


def determine_collector_status(
    pid_file: Path | str = DEFAULT_PID_FILE,
    status_file: Path | str = DEFAULT_STATUS_FILE,
    lock_file: Path | str = DEFAULT_LOCK_FILE,
    db: Database | None = None,
) -> dict[str, Any]:
    """Inspect PID, lock, status file, and DB heartbeat to categorize collector status.

    Returns structured status dictionary distinguishing PROCESS_ALIVE and HEARTBEAT_FRESH.
    """
    pid = read_collector_pid(pid_file)
    proc_alive = False
    proc_verified = False

    if pid is not None:
        proc_alive = is_pid_alive(pid)
        if proc_alive:
            proc_verified = verify_collector_process(pid)

    lock = CollectorLock(lock_file)
    is_locked = lock.is_locked_by_other()

    # Load status file
    s_path = Path(status_file)
    status_data: dict[str, Any] = {}
    if s_path.exists():
        try:
            with open(s_path, "r", encoding="utf-8") as f:
                status_data = json.load(f)
        except Exception:
            pass

    # Check heartbeat freshness
    heartbeat_epoch = status_data.get("timestamp_epoch")
    now_epoch = int(time.time())
    heartbeat_age_sec: float | None = None
    heartbeat_fresh = False

    if heartbeat_epoch is not None:
        heartbeat_age_sec = max(0.0, float(now_epoch - heartbeat_epoch))
        heartbeat_fresh = heartbeat_age_sec <= HEARTBEAT_STALE_THRESHOLD_SEC

    # Evaluate high-level status state
    state: str = "NOT_STARTED"
    raw_status = status_data.get("status", "UNKNOWN")

    if raw_status == "TARGET_REACHED" and not proc_alive:
        state = "TARGET_REACHED"
    elif raw_status == "COST_GUARD_TRIGGERED":
        state = "COST_GUARD_TRIGGERED"
    elif proc_alive and proc_verified:
        if heartbeat_fresh:
            state = "RUNNING_HEALTHY"
        else:
            state = "RUNNING_STALE_HEARTBEAT"
    elif pid is not None and (not proc_alive or not proc_verified):
        state = "PROCESS_DEAD"
    elif is_locked and not proc_alive:
        state = "RUNNING_HEALTHY"  # Lock held by detached child whose PID file might not be read yet
    elif status_data:
        state = "STOPPED"
    else:
        state = "NOT_STARTED"

    return {
        "state": state,
        "pid": pid,
        "process_alive": proc_alive,
        "process_verified": proc_verified,
        "lock_held": is_locked,
        "heartbeat_fresh": heartbeat_fresh,
        "heartbeat_age_sec": heartbeat_age_sec,
        "status_data": status_data,
    }


def launch_detached_collector(
    target_valid_rounds: int = 100,
    cost_ceiling_usd: float = 10.0,
    pid_file: Path | str = DEFAULT_PID_FILE,
    lock_file: Path | str = DEFAULT_LOCK_FILE,
    log_file: Path | str = DEFAULT_LOG_FILE,
    cwd: Path | str | None = None,
) -> tuple[bool, int | None, str]:
    """Launch the autonomous collector as a detached OS child process.

    Returns (success, pid, message).
    """
    lock = CollectorLock(lock_file)
    if lock.is_locked_by_other():
        existing_pid = read_collector_pid(pid_file)
        return (
            False,
            existing_pid,
            f"Collector already running (lock held, PID: {existing_pid or 'unknown'}).",
        )

    # Verify existing PID isn't already alive
    existing_pid = read_collector_pid(pid_file)
    if existing_pid and is_pid_alive(existing_pid) and verify_collector_process(existing_pid):
        return (
            False,
            existing_pid,
            f"Collector already running with verified PID: {existing_pid}.",
        )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    working_dir = Path(cwd) if cwd else Path.cwd()

    # Form command without shell interpolation and WITHOUT exposing credentials in argv
    cmd = [
        sys.executable,
        "-m",
        "pm_research.cli",
        "btc5m-collect",
        "--target-valid-rounds",
        str(target_valid_rounds),
        "--cost-ceiling",
        str(cost_ceiling_usd),
    ]

    # Open log file for child output redirection
    log_fp = open(log_path, "a", encoding="utf-8")

    # Inherit parent environment variables cleanly without CLI exposure
    child_env = dict(os.environ)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(working_dir.resolve()),
            env=child_env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from controlling terminal/session
        )
        pid = proc.pid
        write_collector_pid(pid, pid_file)

        # Brief pause to verify child started cleanly
        time.sleep(0.5)
        if proc.poll() is not None:
            # Child exited prematurely
            exit_code = proc.poll()
            remove_collector_pid(pid_file)
            return (
                False,
                None,
                f"Collector child exited immediately with code {exit_code}. Check {log_path}.",
            )

        logger.info(f"Detached collector process launched successfully with PID {pid}")
        return True, pid, f"Launched detached collector with PID {pid} (logs: {log_path})"
    except Exception as e:
        remove_collector_pid(pid_file)
        return False, None, f"Failed to spawn collector process: {e}"
    finally:
        log_fp.close()


def stop_collector(
    pid_file: Path | str = DEFAULT_PID_FILE,
    stop_file: Path | str = DEFAULT_STOP_FILE,
    timeout_sec: float = 10.0,
) -> tuple[bool, str]:
    """Request graceful stop of the collector and verify its termination.

    1. Writes stop file (data/collector.stop)
    2. Waits for process to exit
    3. If process remains alive after timeout, sends verified SIGTERM
    4. Cleans up PID file upon verified exit.
    """
    s_path = Path(stop_file)
    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.touch()

    pid = read_collector_pid(pid_file)
    if pid is None:
        return True, "Stop signal file written. No active PID file found."

    if not is_pid_alive(pid):
        remove_collector_pid(pid_file)
        if s_path.exists():
            s_path.unlink()
        return True, f"Process {pid} was not running. Cleaned up stale PID file."

    # Verify process identity before interacting
    if not verify_collector_process(pid):
        return False, f"PID {pid} belongs to an unrelated process. Refusing to signal."

    # Wait for graceful shutdown via stop file
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_sec:
        if not is_pid_alive(pid):
            remove_collector_pid(pid_file)
            if s_path.exists():
                try:
                    s_path.unlink()
                except Exception:
                    pass
            return True, f"Collector process {pid} stopped gracefully."
        time.sleep(0.5)

    # Process still alive after timeout: send SIGTERM
    logger.info(f"Process {pid} did not stop gracefully within {timeout_sec}s. Sending SIGTERM...")
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as e:
        return False, f"Error sending SIGTERM to PID {pid}: {e}"

    # Wait up to 5s for SIGTERM termination
    t1 = time.monotonic()
    while time.monotonic() - t1 < 5.0:
        if not is_pid_alive(pid):
            remove_collector_pid(pid_file)
            if s_path.exists():
                try:
                    s_path.unlink()
                except Exception:
                    pass
            return True, f"Collector process {pid} terminated after SIGTERM."
        time.sleep(0.5)

    return False, f"Process {pid} did not terminate after SIGTERM."
