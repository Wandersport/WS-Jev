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


def run_single_clean_round(target_epoch: int = TARGET_BOUNDARY_EPOCH) -> None:
    now = time.time()
    if now < target_epoch:
        sleep_sec = target_epoch - now
        logger.info(f"Waiting {sleep_sec:.2f}s until exact 5-minute boundary ({target_epoch})...")
        time.sleep(sleep_sec)

    # Ensure we are at or past the boundary
    while time.time() < target_epoch:
        time.sleep(0.05)

    logger.info(f"Boundary reached at {time.time():.2f}. Starting LeadLagCollectorV3 for 1 clean full round...")
    db = Database(REPLICATION_DB_PATH)
    collector = LeadLagCollectorV3(
        db=db,
        target_physical_rounds=1,
        is_pilot=True,
    )
    collector.run()
    logger.info("Single clean round collector finished successfully.")


if __name__ == "__main__":
    boundary = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_BOUNDARY_EPOCH
    run_single_clean_round(boundary)
