"""Single clean full round collector runner for pilot completion."""

from __future__ import annotations

import logging
import sys
import time

from pm_research.research.btc5m.leadlag_v3_collector import LeadLagCollectorV3
from pm_research.research.btc5m.leadlag_v3_experiment import REPLICATION_DB_PATH
from pm_research.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("single_round_runner")

TARGET_BOUNDARY_EPOCH: int = 1790877300  # 17:55:00 UTC / 19:55:00 local


def run_single_clean_round(target_epoch: int | None = None) -> None:
    now = time.time()
    if target_epoch is None:
        target_epoch = ((int(now) // 300) + 1) * 300

    logger.info(f"Target boundary epoch: {target_epoch} (in {target_epoch - now:.2f}s). Initializing collector with wait_for_boundary=True...")
    db = Database(REPLICATION_DB_PATH)
    collector = LeadLagCollectorV3(
        db=db,
        target_physical_rounds=10,
        is_pilot=True,
        wait_for_boundary=True,
    )
    collector.run()
    logger.info("Single clean round collector finished successfully.")


if __name__ == "__main__":
    boundary = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_single_clean_round(boundary)
