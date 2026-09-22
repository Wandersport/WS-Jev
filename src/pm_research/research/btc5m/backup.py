"""Transactional SQLite backup subsystem for BTC 5-minute research persistence.

Uses Python's native sqlite3 online backup API (Connection.backup) to create
consistent, non-blocking snapshot copies of the research database without
interfering with active reader/writer threads.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR: Path = Path("data/backups")
DEFAULT_MAX_BACKUPS: int = 5


def backup_database(
    db_path: Path | str = "data/pm_research.db",
    backup_dir: Path | str = DEFAULT_BACKUP_DIR,
    max_backups: int = DEFAULT_MAX_BACKUPS,
    rounds_count: int | None = None,
) -> Path:
    """Create an atomic transactional backup of the SQLite database.

    Args:
        db_path: Path to the active SQLite database file.
        backup_dir: Directory where backup files will be stored.
        max_backups: Maximum number of recent backups to retain.
        rounds_count: Optional number of valid resolved rounds to embed in filename.

    Returns:
        Path to the created backup file.
    """
    src_path = Path(db_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source database file not found: {src_path}")

    b_dir = Path(backup_dir)
    b_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rounds_tag = f"_{rounds_count}r" if rounds_count is not None else ""
    backup_filename = f"pm_research_backup_{timestamp_str}{rounds_tag}.db"
    dest_path = b_dir / backup_filename

    logger.info(f"Creating transactional database backup: {src_path} -> {dest_path}")
    t0 = time.monotonic()

    # Use native online backup API
    src_conn = sqlite3.connect(f"file:{src_path.resolve()}?mode=ro", uri=True)
    dest_conn = sqlite3.connect(str(dest_path.resolve()))
    try:
        src_conn.backup(dest_conn, pages=100, progress=None)
    finally:
        dest_conn.close()
        src_conn.close()

    elapsed = time.monotonic() - t0
    size_bytes = dest_path.stat().st_size
    logger.info(
        f"Backup completed in {elapsed:.3f}s: {dest_path} ({size_bytes:,} bytes)"
    )

    # Rotate old backups
    rotate_backups(backup_dir=b_dir, max_backups=max_backups)

    return dest_path


def rotate_backups(
    backup_dir: Path | str = DEFAULT_BACKUP_DIR,
    max_backups: int = DEFAULT_MAX_BACKUPS,
) -> list[Path]:
    """Prune older backups beyond max_backups. Returns list of retained backups."""
    b_dir = Path(backup_dir)
    if not b_dir.exists():
        return []

    backups = sorted(
        b_dir.glob("pm_research_backup_*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    retained: list[Path] = []
    for i, b in enumerate(backups):
        if i < max_backups:
            retained.append(b)
        else:
            try:
                b.unlink()
                logger.debug(f"Pruned older backup: {b}")
            except Exception as e:
                logger.warning(f"Could not prune backup {b}: {e}")

    return retained


def list_backups(backup_dir: Path | str = DEFAULT_BACKUP_DIR) -> list[Path]:
    """List existing backup files sorted by modification time (most recent first)."""
    b_dir = Path(backup_dir)
    if not b_dir.exists():
        return []
    return sorted(
        b_dir.glob("pm_research_backup_*.db"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def restore_database(
    backup_path: Path | str,
    target_path: Path | str = "data/pm_research.db",
) -> None:
    """Restore database from a backup file using online backup copy."""
    src = Path(backup_path)
    if not src.exists():
        raise FileNotFoundError(f"Backup file not found: {src}")

    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(f"file:{src.resolve()}?mode=ro", uri=True)
    target_conn = sqlite3.connect(str(target.resolve()))
    try:
        src_conn.backup(target_conn)
    finally:
        target_conn.close()
        src_conn.close()
    logger.info(f"Database successfully restored from {src} to {target}")
