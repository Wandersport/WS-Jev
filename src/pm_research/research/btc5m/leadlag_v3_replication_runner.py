"""Runner for Phase 8E Lead-Lag v3 250-Round Prospective Confirmatory Replication."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pm_research.research.btc5m.leadlag_v3_collector import LeadLagCollectorV3
from pm_research.research.btc5m.leadlag_v3_experiment import (
    EXPERIMENT_ID,
    REPLICATION_DB_PATH,
    TARGET_PHYSICAL_ROUNDS,
)
from pm_research.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("replication_runner")

DEFAULT_LOCK_FILE: Path = Path("data/btc5m_leadlag_v3_replication.lock")
DEFAULT_PID_FILE: Path = Path("data/btc5m_leadlag_v3_replication.pid")
DEFAULT_LOG_FILE: Path = Path("data/logs/btc5m_leadlag_v3_replication.log")


def run_replication(target_rounds: int = TARGET_PHYSICAL_ROUNDS) -> None:
    DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_PID_FILE.parent.mkdir(parents=True, exist_ok=True)

    pid = os.getpid()
    DEFAULT_PID_FILE.write_text(f"{pid}\n")
    logger.info(f"Replication runner started with PID {pid} (PID file: {DEFAULT_PID_FILE})")

    db = Database(REPLICATION_DB_PATH)
    collector = LeadLagCollectorV3(
        db=db,
        target_physical_rounds=target_rounds,
        is_pilot=False,
        wait_for_boundary=True,
        lock_file_path=str(DEFAULT_LOCK_FILE),
    )
    logger.info(
        f"Starting LeadLagCollectorV3 for confirmatory replication experiment '{EXPERIMENT_ID}' "
        f"(target: {target_rounds} full rounds, wait_for_boundary=True)..."
    )
    try:
        collector.run()
        logger.info(f"Replication collector completed {target_rounds} rounds cleanly.")
    finally:
        try:
            DEFAULT_PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_PHYSICAL_ROUNDS
    run_replication(rounds)
