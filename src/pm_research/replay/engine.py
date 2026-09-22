"""Deterministic historical replay engine with simulated clock and execution latency."""

from __future__ import annotations

import json
import logging
import random
import subprocess
from datetime import timedelta

from pm_research.calibration.metrics import CalibrationEngine
from pm_research.config import SystemConfig
from pm_research.domain.models import (
    PaperFill,
)
from pm_research.execution.paper_broker import PaperBroker
from pm_research.pipeline.bram import BramRiskGate
from pm_research.pipeline.holt import HoltResearcher
from pm_research.pipeline.ilsa import IlsaEstimator
from pm_research.pipeline.kett import KettSizer
from pm_research.pipeline.rigo import RigoIngestor
from pm_research.portfolio.tess import TessPortfolio
from pm_research.replay.clock import SimulatedClock
from pm_research.replay.dataset import DatasetManager, ReplayDataset
from pm_research.replay.models import (
    HistoricalResolution,
    QueuedOrder,
    ReplayConfig,
    ReplayResult,
)
from pm_research.storage.db import Database
from pm_research.utils import ensure_utc, generate_id, now_utc, to_iso_utc

logger = logging.getLogger(__name__)


def get_git_commit_sha() -> str:
    """Retrieve the current local Git commit SHA, or 'unknown' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3.0,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


class ReplayEngine:
    """Simulates historical prediction-market quantitative research with temporal integrity."""

    def __init__(
        self,
        config: ReplayConfig,
        db: Database | None = None,
        dataset: ReplayDataset | None = None,
    ) -> None:
        self.replay_config = config
        self.sys_config = config.system_config or SystemConfig(
            initial_bankroll=config.initial_bankroll
        )
        # Ensure bankroll alignment
        self.sys_config.initial_bankroll = config.initial_bankroll

        # Set deterministic random seed
        random.seed(config.random_seed)

        # Load dataset
        if dataset is not None:
            self.dataset = dataset
        else:
            manager = DatasetManager()
            self.dataset = manager.get_dataset(config.dataset_path)

        # Initialize local database (in-memory or specified path)
        self.db = db or Database(":memory:")
        self.clock: SimulatedClock | None = None

        # Core pipeline components
        self.rigo = RigoIngestor(self.sys_config)
        self.holt = HoltResearcher(self.sys_config)
        self.ilsa = IlsaEstimator(self.sys_config)
        self.kett = KettSizer(self.sys_config)
        self.bram = BramRiskGate(self.sys_config)
        self.paper_broker = PaperBroker(self.sys_config)
        self.tess = TessPortfolio(self.sys_config, self.db)

        # State tracking
        self.queued_orders: list[QueuedOrder] = []
        self.resolved_market_ids: set[str] = set()

    def run(self) -> ReplayResult:
        """Execute the deterministic historical replay end-to-end."""
        started_at = now_utc()
        replay_id = generate_id("replay")
        logger.info(f"Starting Replay {replay_id} on dataset '{self.dataset.manifest.dataset_id}'")

        all_snapshots = self.dataset.get_snapshots()
        if not all_snapshots:
            raise ValueError(f"Dataset '{self.dataset.manifest.dataset_id}' contains no snapshots.")

        # Determine simulation window
        sim_start = self.replay_config.start_time or all_snapshots[0].timestamp
        sim_end = self.replay_config.end_time or all_snapshots[-1].timestamp
        sim_start = ensure_utc(sim_start)
        sim_end = ensure_utc(sim_end)

        # Initialize SimulatedClock strictly at sim_start
        self.clock = SimulatedClock(sim_start)

        # Track resolutions: strictly isolated until clock reaches resolution time
        pending_resolutions = sorted(self.dataset.get_resolutions(), key=lambda r: r.resolved_at)

        cycles_processed = 0
        snapshots_processed = 0
        total_proposals = 0
        total_accepted = 0
        total_rejected = 0
        executed_fills: list[PaperFill] = []
        raw_edges: list[float] = []
        robust_edges: list[float] = []

        # Iterate chronologically over timestamp batches (cycles)
        for cycle_time, raw_snapshots in self.dataset.stream_cycles():
            cycle_time = ensure_utc(cycle_time)
            if cycle_time < sim_start:
                continue
            if cycle_time > sim_end:
                break

            # 1. Advance simulated clock forward to exact cycle timestamp
            self.clock.set_time(cycle_time)
            t_k = self.clock.now()
            cycle_id = generate_id("cyc")
            cycles_processed += 1
            snapshots_processed += len(raw_snapshots)

            with self.db.transaction() as conn:
                self.db.save_cycle(
                    cycle_id=cycle_id,
                    started_at=to_iso_utc(t_k),
                    config_hash=self.sys_config.config_hash,
                    notes=f"Replay {replay_id} cycle at {t_k.isoformat()}",
                    conn=conn,
                )

                # 2. Check and settle any resolutions where resolved_at <= t_k
                still_pending: list[HistoricalResolution] = []
                for res in pending_resolutions:
                    if res.resolved_at <= t_k:
                        if res.market_id not in self.resolved_market_ids:
                            logger.info(f"Resolving market {res.market_id} to {res.resolved_outcome.value} at simulated time {t_k.isoformat()}")
                            self.tess.resolve_market(
                                market_id=res.market_id,
                                resolved_outcome=res.resolved_outcome,
                                cycle_id=cycle_id,
                                resolved_at=res.resolved_at,
                            )
                            self.resolved_market_ids.add(res.market_id)
                    else:
                        still_pending.append(res)
                pending_resolutions = still_pending

                # 3. Ingest and normalize current raw snapshots at t_k
                raw_dicts = [s.to_dict() for s in raw_snapshots]
                current_market_snapshots = self.rigo.ingest_snapshots(
                    raw_markets=raw_dicts,
                    cycle_id=cycle_id,
                    current_time=t_k,
                )
                for m_snap in current_market_snapshots:
                    self.db.save_market_snapshot(m_snap, conn=conn)

                snap_by_market_id = {s.market_id: s for s in current_market_snapshots}

                # 4. Check Queued Latency Orders: attempt fill against fresh depth if earliest_fill_time <= t_k
                still_queued: list[QueuedOrder] = []
                for queued in self.queued_orders:
                    if queued.earliest_fill_time <= t_k:
                        fresh_snap = snap_by_market_id.get(queued.market_id)
                        # Order can fill only if market is still actively quoted and not settled
                        if fresh_snap and fresh_snap.market_id not in self.resolved_market_ids:
                            exec_res = self.paper_broker.execute_decision(
                                decision=queued.decision,
                                proposal=queued.proposal,
                                snapshot=fresh_snap,
                            )
                            if exec_res is not None:
                                order, fill = exec_res
                                executed_fills.append(fill)
                                self.db.save_paper_order(order, conn=conn)
                                self.db.save_paper_fill(fill, conn=conn)
                                self.tess.process_fill(fill, category=fresh_snap.market.category)
                                logger.info(
                                    f"Latency fill executed for {queued.market_id}: requested {order.requested_quantity} contracts, filled at {fill.average_fill_price} (latency: {self.replay_config.latency_seconds}s)"
                                )
                        else:
                            logger.info(
                                f"Queued order for {queued.market_id} expired without fill (market no longer active or resolved)"
                            )
                    else:
                        still_queued.append(queued)
                self.queued_orders = still_queued

                # 5. Mark existing open positions to market
                self.tess.mark_to_market(current_market_snapshots, cycle_id=cycle_id, current_time=t_k)

                # 6. Execute Research Pipeline on active markets
                for snap in current_market_snapshots:
                    mkt = snap.market
                    if mkt.market_id in self.resolved_market_ids:
                        continue

                    # HOLT: Extract research features
                    feat = self.holt.extract_features(snap, current_time=t_k)
                    self.db.save_research_features(feat, conn=conn)

                    # ILSA: Estimate probability
                    est = self.ilsa.estimate(snap, features=feat)
                    self.db.save_probability_estimate(est, conn=conn)

                    # KETT: Calculate sizing proposal
                    cat_exp = self.tess.category_exposure.get(mkt.category, 0.0)
                    prop = self.kett.calculate_proposal(
                        snapshot=snap,
                        estimate=est,
                        equity=self.tess.equity,
                        available_cash=self.tess.virtual_cash,
                        current_risk_state=self.tess.risk_state,
                        category_exposure=cat_exp,
                        total_exposure=self.tess.total_exposure,
                    )

                    if prop is None:
                        continue

                    total_proposals += 1
                    raw_edges.append(prop.raw_edge)
                    robust_edges.append(prop.robust_edge)
                    self.db.save_trade_proposal(prop, conn=conn)

                    # BRAM: Authoritative risk gate
                    decision = self.bram.evaluate(
                        proposal=prop,
                        snapshot=snap,
                        open_positions=self.tess.open_positions,
                        equity=self.tess.equity,
                        virtual_cash=self.tess.virtual_cash,
                        current_risk_state=self.tess.risk_state,
                        current_time=t_k,
                    )
                    self.db.save_risk_decision(decision, conn=conn)

                    if decision.accepted:
                        total_accepted += 1
                        # Latency routing
                        if self.replay_config.latency_seconds <= 0.0:
                            # Immediate fill at t_k
                            exec_res = self.paper_broker.execute_decision(decision, prop, snap)
                            if exec_res is not None:
                                order, fill = exec_res
                                executed_fills.append(fill)
                                self.db.save_paper_order(order, conn=conn)
                                self.db.save_paper_fill(fill, conn=conn)
                                self.tess.process_fill(fill, category=mkt.category)
                        else:
                            # Queue order for delayed fill after latency
                            queued_order = QueuedOrder(
                                order_id=generate_id("qord"),
                                market_id=prop.market_id,
                                side=prop.side,
                                decision=decision,
                                proposal=prop,
                                decision_time=t_k,
                                earliest_fill_time=t_k + timedelta(seconds=self.replay_config.latency_seconds),
                                decision_snapshot=snap,
                            )
                            self.queued_orders.append(queued_order)
                    else:
                        total_rejected += 1

                # 7. Final cycle mark-to-market and snapshot
                self.tess.mark_to_market(current_market_snapshots, cycle_id=cycle_id, current_time=t_k)
                self.db.finish_cycle(cycle_id, to_iso_utc(now_utc()), conn=conn)

        # -------------------------------------------------------------
        # Post-Replay Final Resolution & Accounting Settlement
        # Settle any resolutions that occurred up to sim_end or resolution horizon
        # -------------------------------------------------------------
        final_resolution_time = sim_end
        if pending_resolutions:
            for res in pending_resolutions:
                if res.market_id not in self.resolved_market_ids:
                    # Advance simulated clock to resolution time if forward
                    if res.resolved_at > self.clock.now():
                        self.clock.set_time(res.resolved_at)
                    self.tess.resolve_market(
                        market_id=res.market_id,
                        resolved_outcome=res.resolved_outcome,
                        cycle_id=generate_id("cyc_res"),
                        resolved_at=res.resolved_at,
                    )
                    self.resolved_market_ids.add(res.market_id)
            final_resolution_time = self.clock.now()

        # Final mark-to-market
        self.tess.mark_to_market([], cycle_id="final", current_time=self.clock.now())

        # -------------------------------------------------------------
        # Compile Metrics & Calibration Analytics
        # -------------------------------------------------------------
        calib_obs = self.db.get_all_calibration_observations()
        calib_engine = CalibrationEngine()
        calibration_report = calib_engine.compute_metrics(calib_obs)

        # Portfolio metrics
        init_bankroll = self.tess.bankroll_initial
        final_eq = self.tess.equity
        total_pnl = self.tess.total_pnl
        ret_pct = round(((final_eq - init_bankroll) / init_bankroll) * 100.0, 2)
        max_dd_pct = round(self.tess.max_drawdown * 100.0, 2)
        turnover = round(sum(f.total_cost for f in executed_fills), 2)
        total_slippage = round(sum(f.slippage for f in executed_fills), 4)
        total_fees = round(sum(f.fees for f in executed_fills), 4)
        avg_raw_edge = round(sum(raw_edges) / len(raw_edges), 4) if raw_edges else 0.0
        avg_robust_edge = round(sum(robust_edges) / len(robust_edges), 4) if robust_edges else 0.0

        finished_at = now_utc()
        git_commit = get_git_commit_sha()

        result = ReplayResult(
            replay_id=replay_id,
            dataset_id=self.dataset.manifest.dataset_id,
            model_version=self.sys_config.model_version,
            config_hash=self.sys_config.config_hash,
            random_seed=self.replay_config.random_seed,
            git_commit_sha=git_commit,
            started_at=started_at,
            finished_at=finished_at,
            simulated_start=sim_start,
            simulated_end=final_resolution_time,
            cycles_count=cycles_processed,
            snapshots_count=snapshots_processed,
            latency_seconds=self.replay_config.latency_seconds,
            calibration=calibration_report,
            initial_bankroll=init_bankroll,
            final_equity=round(final_eq, 2),
            total_pnl=round(total_pnl, 2),
            simulated_return_pct=ret_pct,
            max_drawdown_pct=max_dd_pct,
            turnover=turnover,
            total_slippage=total_slippage,
            total_fees=total_fees,
            proposals_count=total_proposals,
            accepted_count=total_accepted,
            rejected_count=total_rejected,
            fills_count=len(executed_fills),
            avg_raw_edge=avg_raw_edge,
            avg_robust_edge=avg_robust_edge,
            final_open_positions_count=len(self.tess.open_positions),
        )

        # Persist replay result to database
        self.db.save_replay_run(
            replay_id=replay_id,
            dataset_id=self.dataset.manifest.dataset_id,
            created_at=to_iso_utc(finished_at),
            config_json=json.dumps({
                "dataset_path": str(self.replay_config.dataset_path),
                "latency_seconds": self.replay_config.latency_seconds,
                "initial_bankroll": self.replay_config.initial_bankroll,
                "random_seed": self.replay_config.random_seed,
            }),
            result_json=json.dumps(result.to_dict()),
        )

        logger.info(
            f"Replay {replay_id} finished: Return={ret_pct}%, MaxDD={max_dd_pct}%, Brier={calibration_report.brier_score}, Fills={len(executed_fills)}"
        )
        return result
