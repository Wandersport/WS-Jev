"""Autonomous, crash-recoverable prospective data collector for BTC 5-minute forecasting research.

Coordinates:
- Public WebSocket streams (Chainlink RTDS settlement reference & Binance Perp microstructure)
- Discovery of active BTC 5-minute prediction rounds
- Strict round-open boundary midpoint provenance (+/- 2500ms tolerance)
- Multi-horizon point-in-time snapshot freezing (240s, 180s, 120s, 60s, 30s)
- 4-condition Jev ablation queries (pinned typesafe/jev-1.13)
- Cost ceiling guard enforcement (default $10.00)
- Official settlement resolution ingestion and scoring
- Milestone checkpoints (30, 100, 500 valid resolved rounds)
- Periodic heartbeats and status reporting (data/collector_status.json)
- Transactional SQLite backups via sqlite3.Connection.backup()
- Graceful crash recovery and clean shutdown handling

Safety Invariants:
- FORECAST RESEARCH ONLY.
- Zero trade proposals (Kett), zero risk decisions (Bram), zero paper broker orders/fills.
- Zero wallets, private keys, live credentials, or transaction signing.
- Public read-only data endpoints only.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pm_research.research.btc5m.backup import backup_database
from pm_research.research.btc5m.experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_SPEC_HASH,
)
from pm_research.research.btc5m.lab import BTC5mShadowLab
from pm_research.research.btc5m.snapshot import STANDARD_HORIZONS_SEC
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

DEFAULT_TARGET_VALID_ROUNDS: int = 100
DEFAULT_COST_CEILING_USD: float = 10.0
DEFAULT_MILESTONES: tuple[int, ...] = (30, 100, 500)
DEFAULT_STATUS_FILE: Path = Path("data/collector_status.json")
DEFAULT_STOP_FILE: Path = Path("data/collector.stop")
DEFAULT_CHECKPOINTS_DIR: Path = Path("data/checkpoints")


class BTC5mAutonomousCollector:
    """Long-running autonomous collector for the frozen BTC 5-minute Jev prospective experiment."""

    def __init__(
        self,
        db: Database | None = None,
        lab: BTC5mShadowLab | None = None,
        target_valid_rounds: int = DEFAULT_TARGET_VALID_ROUNDS,
        cost_ceiling_usd: float = DEFAULT_COST_CEILING_USD,
        milestones: Sequence[int] = DEFAULT_MILESTONES,
        status_file_path: Path | str = DEFAULT_STATUS_FILE,
        stop_file_path: Path | str = DEFAULT_STOP_FILE,
        checkpoints_dir: Path | str = DEFAULT_CHECKPOINTS_DIR,
        poll_interval_sec: float = 3.0,
    ) -> None:
        self.db = db or Database()
        self.lab = lab or BTC5mShadowLab(db=self.db)
        self.target_valid_rounds = target_valid_rounds
        self.cost_ceiling_usd = cost_ceiling_usd
        self.milestones = sorted(milestones)
        self.status_file_path = Path(status_file_path)
        self.stop_file_path = Path(stop_file_path)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.poll_interval_sec = poll_interval_sec

        self.status: str = "INITIALIZING"
        self.current_round_slug: str | None = None
        self.stop_requested: bool = False
        self.cost_guard_triggered: bool = False
        self._checked_milestones: set[int] = set()

        # Ensure directory structures exist
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.status_file_path.parent.mkdir(parents=True, exist_ok=True)

    def request_stop(self) -> None:
        """Signal collector to shut down gracefully after current operation."""
        self.stop_requested = True
        logger.info("Graceful stop requested.")

    def is_stop_requested(self) -> bool:
        """Check if stop was requested via signal, file, or target reached."""
        if self.stop_requested:
            return True
        if self.stop_file_path.exists():
            logger.info(f"Stop signal file detected: {self.stop_file_path}")
            self.stop_requested = True
            return True
        return False

    def emit_heartbeat(
        self,
        status: str | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Record collector heartbeat to database and write human-readable status JSON."""
        current_status = status or self.status
        valid_rounds = self.db.count_valid_resolved_rounds()
        cost_usd = self.db.get_total_openrouter_cost()

        try:
            snaps_count = len(self.db.get_btc5m_snapshots())
            fcs_count = len(self.db.get_btc5m_forecasts())
        except Exception:
            snaps_count = 0
            fcs_count = 0

        rtds_status = getattr(self.lab.ref_feed, "status", "unknown")
        binance_status = getattr(self.lab.binance_feed, "status", "unknown")

        now_utc = datetime.now(timezone.utc).isoformat()
        now_epoch = int(time.time())

        # Save to DB
        try:
            self.db.save_collector_heartbeat(
                status=current_status,
                current_round_slug=self.current_round_slug,
                valid_resolved_rounds=valid_rounds,
                total_snapshots=snaps_count,
                total_forecasts=fcs_count,
                total_openrouter_cost_usd=cost_usd,
                rtds_status=rtds_status,
                binance_ws_status=binance_status,
                experiment_spec_hash=EXPERIMENT_SPEC_HASH,
                notes=notes,
            )
        except Exception as e:
            logger.warning(f"Could not persist heartbeat to DB: {e}")

        payload: dict[str, Any] = {
            "timestamp_utc": now_utc,
            "timestamp_epoch": now_epoch,
            "status": current_status,
            "experiment_id": EXPERIMENT_ID,
            "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
            "current_round_slug": self.current_round_slug,
            "valid_resolved_rounds": valid_rounds,
            "target_valid_rounds": self.target_valid_rounds,
            "total_snapshots": snaps_count,
            "total_forecasts": fcs_count,
            "total_openrouter_cost_usd": cost_usd,
            "cost_ceiling_usd": self.cost_ceiling_usd,
            "cost_guard_triggered": self.cost_guard_triggered,
            "feeds": {
                "chainlink_rtds": rtds_status,
                "binance_perp_ws": binance_status,
            },
            "notes": notes,
        }

        # Write atomic status file
        try:
            tmp_file = self.status_file_path.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp_file.replace(self.status_file_path)
        except Exception as e:
            logger.warning(f"Could not write status file {self.status_file_path}: {e}")

        return payload

    def reconcile_startup_state(self) -> None:
        """Inspect database for uncompleted/active rounds and recover or resolve them."""
        logger.info("Checking database state for startup crash recovery...")
        now_epoch = int(time.time())

        try:
            active_rounds = self.db.get_btc5m_rounds(status="active")
        except Exception as e:
            logger.warning(f"Could not query active rounds: {e}")
            active_rounds = []

        for r_dict in active_rounds:
            slug = r_dict["round_slug"]
            end_epoch = int(r_dict["end_epoch"])
            logger.info(f"Recovering active round: {slug} (ended {now_epoch - end_epoch}s ago)")

            if now_epoch > end_epoch + 15:
                # Round has concluded, attempt official resolution poll
                try:
                    r_obj, _ = self.lab.contract_mgr.discover_round_by_slug(slug)
                    if r_obj is not None:
                        official_res = self.lab.poll_round_resolution(
                            round_info=r_obj,
                            poll_interval_sec=2.0,
                            max_wait_sec=15,
                        )
                        if official_res and official_res.is_resolved:
                            logger.info(f"Successfully recovered and resolved {slug} to {official_res.resolved_outcome}")
                except Exception as e:
                    logger.warning(f"Could not resolve recovered round {slug}: {e}")

        # Mark previously reached milestones as checked
        current_valid = self.db.count_valid_resolved_rounds()
        for m in self.milestones:
            if current_valid >= m:
                self._checked_milestones.add(m)

        logger.info(
            f"Startup recovery complete. Valid resolved rounds: {current_valid}/{self.target_valid_rounds}. "
            f"Total spend: ${self.db.get_total_openrouter_cost():.4f}"
        )

    def trigger_milestone_checkpoint(
        self,
        milestone: int,
        valid_rounds: int,
    ) -> Path | None:
        """Generate evaluation summary, create DB backup, and write checkpoint artifact."""
        logger.info(f"Triggering milestone checkpoint for {milestone} rounds (current: {valid_rounds})...")

        # 1. Transactional SQLite backup
        backup_path: Path | None = None
        try:
            backup_path = backup_database(rounds_count=valid_rounds)
        except Exception as e:
            logger.warning(f"Milestone backup creation failed: {e}")

        # 2. Compute evaluation metrics
        eval_summary = self.lab.compute_evaluation_summary()

        # 3. Persist checkpoint in DB
        cost_usd = self.db.get_total_openrouter_cost()
        snapshots_count = len(self.db.get_btc5m_snapshots())
        forecasts_count = len(self.db.get_btc5m_forecasts())
        notes = f"Milestone checkpoint at {valid_rounds} valid resolved rounds"

        try:
            self.db.save_checkpoint(
                milestone_rounds=milestone,
                valid_resolved_rounds=valid_rounds,
                total_snapshots=snapshots_count,
                total_forecasts=forecasts_count,
                total_openrouter_cost_usd=cost_usd,
                brier_scores=eval_summary,
                experiment_spec_hash=EXPERIMENT_SPEC_HASH,
                backup_file_path=str(backup_path) if backup_path else None,
                notes=notes,
            )
        except Exception as e:
            logger.warning(f"Could not save checkpoint to DB: {e}")

        # 4. Write milestone JSON artifact
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoints_dir / f"checkpoint_{milestone}r_{timestamp_str}.json"
        checkpoint_data: dict[str, Any] = {
            "checkpoint_milestone": milestone,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "valid_resolved_rounds": valid_rounds,
            "target_valid_rounds": self.target_valid_rounds,
            "total_snapshots": snapshots_count,
            "total_forecasts": forecasts_count,
            "total_openrouter_cost_usd": cost_usd,
            "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
            "backup_file": str(backup_path) if backup_path else None,
            "evaluation_summary": eval_summary,
        }
        try:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Checkpoint artifact written: {checkpoint_file}")
            self._checked_milestones.add(milestone)
            return checkpoint_file
        except Exception as e:
            logger.warning(f"Could not write checkpoint file {checkpoint_file}: {e}")
            return None

    def run(self) -> None:
        """Main autonomous collection loop."""
        logger.info("=== Starting BTC 5-minute Autonomous Prospective Collector ===")
        logger.info(f"Target valid rounds: {self.target_valid_rounds}")
        logger.info(f"Cost ceiling: ${self.cost_ceiling_usd:.2f}")
        logger.info(f"Frozen Spec Hash: {EXPERIMENT_SPEC_HASH}")

        # Setup OS signal handlers for graceful shutdown
        def _sig_handler(signum: int, _frame: Any) -> None:
            logger.info(f"Received signal {signum}. Requesting graceful shutdown...")
            self.request_stop()

        try:
            signal.signal(signal.SIGINT, _sig_handler)
            signal.signal(signal.SIGTERM, _sig_handler)
        except Exception:
            pass  # May fail if not in main thread

        # Start background feeds
        try:
            self.lab.ref_feed.start_background_listener()
            self.lab.binance_feed.start_background_listener()
            logger.info("Started background streaming feeds (Chainlink RTDS & Binance Perp WS)")
        except Exception as e:
            logger.warning(f"Error starting background feeds: {e}")

        # Reconcile on startup
        self.status = "STARTUP_RECOVERY"
        self.emit_heartbeat()
        self.reconcile_startup_state()

        completed_slugs: set[str] = {
            r["round_slug"]
            for r in self.db.get_btc5m_rounds()
            if r.get("status") == "resolved"
        }

        self.status = "COLLECTING"

        while not self.is_stop_requested():
            valid_rounds = self.db.count_valid_resolved_rounds()
            if valid_rounds >= self.target_valid_rounds:
                logger.info(
                    f"Reached target valid resolved rounds ({valid_rounds}/{self.target_valid_rounds})."
                )
                self.status = "TARGET_REACHED"
                self.trigger_milestone_checkpoint(
                    milestone=self.target_valid_rounds,
                    valid_rounds=valid_rounds,
                )
                self.emit_heartbeat(status="TARGET_REACHED")
                break

            # Check cost ceiling guard
            total_spend = self.db.get_total_openrouter_cost()
            if total_spend >= self.cost_ceiling_usd:
                if not self.cost_guard_triggered:
                    logger.warning(
                        f"Cost ceiling reached: spent ${total_spend:.4f} >= limit ${self.cost_ceiling_usd:.2f}. "
                        "Triggering COST_GUARD_TRIGGERED mode (pausing LLM forecasts)."
                    )
                    self.cost_guard_triggered = True
                self.emit_heartbeat(status="COST_GUARD_TRIGGERED")

            # Discover active round
            try:
                round_info = self.lab.contract_mgr.discover_active_round()
            except Exception as e:
                logger.debug(f"Discovery poll failed: {e}")
                self.emit_heartbeat(notes=f"Discovery poll: {e}")
                time.sleep(self.poll_interval_sec)
                continue

            slug = round_info.round_slug
            self.current_round_slug = slug

            if slug in completed_slugs:
                # Current round is already resolved; wait for the next 5m cycle
                seconds_to_end = max(1.0, round_info.seconds_remaining)
                logger.debug(
                    f"Round {slug} already resolved. Waiting {min(15.0, seconds_to_end):.1f}s for next cycle..."
                )
                self.emit_heartbeat()
                time.sleep(min(10.0, seconds_to_end))
                continue

            # Check if round has sufficient time remaining
            if round_info.seconds_remaining < 35.0:
                logger.info(
                    f"Discovered round {slug} with only {round_info.seconds_remaining:.1f}s remaining. "
                    "Skipping forecasting to avoid partial capture."
                )
                self.emit_heartbeat(notes=f"Skipping round {slug} (late discovery)")
                time.sleep(max(2.0, round_info.seconds_remaining + 3.0))
                continue

            # Persist discovered round
            self.db.save_btc5m_round(round_info, status="active")
            logger.info(
                f"=== Monitoring Round {slug} | Remaining: {round_info.seconds_remaining:.1f}s "
                f"| Target valid: {valid_rounds}/{self.target_valid_rounds} ==="
            )
            self.emit_heartbeat()

            # Record Binance boundary open midpoint if within tolerance
            self._ensure_round_open_provenance(round_info)

            # Monitor horizons
            self._process_round_horizons(round_info)

            if self.is_stop_requested():
                break

            # Round close: wait for settlement and score
            logger.info(f"Round {slug} completed horizons. Awaiting official settlement...")
            self.status = "AWAITING_RESOLUTION"
            self.emit_heartbeat()

            official_res = self.lab.poll_round_resolution(
                round_info=round_info,
                poll_interval_sec=4.0,
                max_wait_sec=300,
            )

            if official_res and official_res.is_resolved:
                completed_slugs.add(slug)
                new_valid_count = self.db.count_valid_resolved_rounds()
                logger.info(
                    f"Round {slug} scored successfully. Total valid resolved rounds: {new_valid_count}"
                )

                # Check milestones
                for m in self.milestones:
                    if new_valid_count >= m and m not in self._checked_milestones:
                        self.trigger_milestone_checkpoint(milestone=m, valid_rounds=new_valid_count)

            self.current_round_slug = None
            self.status = "COLLECTING"
            self.emit_heartbeat()
            time.sleep(self.poll_interval_sec)

        # Clean shutdown handling
        logger.info("Collector loop terminated. Cleaning up...")
        try:
            self.lab.ref_feed.stop_background_listener()
            self.lab.binance_feed.stop_background_listener()
        except Exception:
            pass

        final_status = "STOPPED" if not (self.status == "TARGET_REACHED") else "TARGET_REACHED"
        self.emit_heartbeat(status=final_status, notes="Collector shut down cleanly")

        if self.stop_file_path.exists():
            try:
                self.stop_file_path.unlink()
                logger.info(f"Removed stop file: {self.stop_file_path}")
            except Exception:
                pass

        logger.info(f"Collector shutdown complete. Final status: {final_status}")

    def _ensure_round_open_provenance(self, round_info: Any) -> None:
        """Attempt to find and record the exact boundary midpoint from Binance feed within tolerance."""
        start_ms = round_info.start_epoch * 1000
        boundary_mid = self.lab.binance_feed.find_boundary_open_mid(
            round_start_ms=start_ms,
            tolerance_ms=2500,
        )
        if boundary_mid is not None:
            mid, s_ts, r_ts, offset_ms = boundary_mid
            self.lab.binance_feed.record_round_open(
                round_slug=round_info.round_slug,
                mid_price=mid,
                timestamp_ms=r_ts,
                source_timestamp_ms=s_ts,
                offset_ms=offset_ms,
            )
            logger.info(
                f"Recorded Binance boundary open mid for {round_info.round_slug}: {mid} "
                f"(timing offset: {offset_ms}ms)"
            )
        else:
            # If round started very recently, check REST snapshot once as fallback
            now_ms = int(time.time() * 1000)
            if abs(now_ms - start_ms) <= 2500:
                try:
                    b_snap = self.lab.binance_feed.fetch_rest_snapshot(ref_price=round_info.price_to_beat)
                    if b_snap.is_valid and b_snap.mid_price > 0:
                        offset = b_snap.received_at_ms - start_ms
                        if abs(offset) <= 2500:
                            self.lab.binance_feed.record_round_open(
                                round_slug=round_info.round_slug,
                                mid_price=b_snap.mid_price,
                                timestamp_ms=b_snap.received_at_ms,
                                source_timestamp_ms=b_snap.source_event_timestamp_ms,
                                offset_ms=offset,
                            )
                            logger.info(
                                f"Recorded fallback Binance open mid for {round_info.round_slug}: "
                                f"{b_snap.mid_price} (offset: {offset}ms)"
                            )
                except Exception as e:
                    logger.debug(f"REST fallback open mid capture failed: {e}")

    def _process_round_horizons(self, round_info: Any) -> None:
        """Iterate through the 5 standard horizons and execute snapshots/forecasts."""
        horizons = sorted(STANDARD_HORIZONS_SEC, reverse=True)

        for h in horizons:
            if self.is_stop_requested():
                break

            target_time = round_info.end_epoch - h
            now = time.time()
            wait_sec = target_time - now

            if wait_sec < -3.0:
                logger.info(f"Horizon {h}s already elapsed ({wait_sec:.1f}s ago). Skipping.")
                continue

            if wait_sec > 0:
                logger.debug(f"Waiting {wait_sec:.1f}s for horizon {h}s...")
                # Sleep in short increments to remain responsive to stop requests
                while wait_sec > 0 and not self.is_stop_requested():
                    step = min(wait_sec, 1.0)
                    time.sleep(step)
                    wait_sec = target_time - time.time()

            if self.is_stop_requested():
                break

            # Capture snapshot at horizon
            snapshot = self.lab.capture_horizon_snapshot(
                round_info=round_info,
                target_horizon_sec=h,
            )
            self.emit_heartbeat()

            if not snapshot.is_valid:
                logger.info(
                    f"Horizon {h}s snapshot invalid for {round_info.round_slug} "
                    f"({snapshot.skip_reason}). Skipping ablation."
                )
                continue

            # Check cost ceiling before executing OpenRouter LLM inference
            total_spend = self.db.get_total_openrouter_cost()
            if total_spend >= self.cost_ceiling_usd:
                logger.warning(
                    f"Cost ceiling reached (${total_spend:.4f} >= ${self.cost_ceiling_usd:.2f}). "
                    f"Skipping Jev ablation for horizon {h}s."
                )
                self.cost_guard_triggered = True
                continue

            # Execute 4 ablation conditions
            logger.info(
                f"Executing Jev ablation (4 conditions) for {round_info.round_slug} @ {h}s..."
            )
            fcs = self.lab.execute_horizon_ablation(snapshot)
            logger.info(f"Produced {len(fcs)} valid forecasts for {snapshot.snapshot_id}")
            self.emit_heartbeat()
