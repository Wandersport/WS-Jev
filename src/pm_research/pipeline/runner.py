"""Pipeline runner executing the research cycle."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    PaperFill,
    PaperOrder,
    ProbabilityEstimate,
    ResearchFeatures,
    RiskDecision,
    TradeProposal,
)
from pm_research.execution.paper_broker import PaperBroker
from pm_research.pipeline.bram import BramRiskGate
from pm_research.pipeline.holt import HoltResearcher
from pm_research.pipeline.ilsa import IlsaEstimator
from pm_research.pipeline.kett import KettSizer
from pm_research.pipeline.rigo import RigoIngestor
from pm_research.portfolio.tess import TessPortfolio
from pm_research.storage.db import Database
from pm_research.utils import ensure_utc, generate_id, now_utc, to_iso_utc

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the quantitative research pipeline:

    Rigo -> Holt -> Ilsa -> Kett -> Bram -> PaperBroker -> Tess.
    """

    def __init__(
        self,
        config: SystemConfig,
        db: Database,
        rigo: RigoIngestor | None = None,
        holt: HoltResearcher | None = None,
        ilsa: IlsaEstimator | None = None,
        kett: KettSizer | None = None,
        bram: BramRiskGate | None = None,
        paper_broker: PaperBroker | None = None,
        tess: TessPortfolio | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.rigo = rigo or RigoIngestor(config)
        self.holt = holt or HoltResearcher(config)
        self.ilsa = ilsa or IlsaEstimator(config)
        self.kett = kett or KettSizer(config)
        self.bram = bram or BramRiskGate(config)
        self.paper_broker = paper_broker or PaperBroker(config)
        self.tess = tess or TessPortfolio(config, db)

    def run_cycle(
        self,
        raw_markets: list[dict[str, Any]],
        features_by_market: dict[str, ResearchFeatures] | None = None,
        cycle_time: datetime | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Execute a full paper research cycle end-to-end inside an atomic transaction."""
        now = ensure_utc(cycle_time or now_utc())
        cycle_id = generate_id("cyc")
        started_at_str = to_iso_utc(now)

        with self.db.transaction() as conn:
            # 1. Start cycle in DB
            self.db.save_cycle(cycle_id, started_at_str, self.config.config_hash, notes=notes, conn=conn)

            # 2. RIGO: Ingestion and normalization
            snapshots = self.rigo.ingest_snapshots(raw_markets, cycle_id=cycle_id, current_time=now)
            for snap in snapshots:
                self.db.save_market_snapshot(snap, conn=conn)

            # 3. Mark existing portfolio positions to market with fresh snapshots
            self.tess.mark_to_market(snapshots, cycle_id=cycle_id, current_time=now)

            estimates: list[ProbabilityEstimate] = []
            proposals: list[TradeProposal] = []
            decisions: list[RiskDecision] = []
            executed_orders: list[PaperOrder] = []
            executed_fills: list[PaperFill] = []

            # 4. HOLT -> ILSA -> KETT -> BRAM -> PAPERBROKER -> TESS for each market snapshot
            for snap in snapshots:
                market = snap.market

                # HOLT: Extract research features (or use provided override)
                if features_by_market and market.market_id in features_by_market:
                    feat = features_by_market[market.market_id]
                else:
                    feat = self.holt.extract_features(snap, current_time=now)
                self.db.save_research_features(feat, conn=conn)

                # ILSA: Probability estimation
                estimate = self.ilsa.estimate(snap, features=feat)
                estimates.append(estimate)
                self.db.save_probability_estimate(estimate, conn=conn)

                # KETT: Edge calculation and sizing proposal
                cat_exp = self.tess.category_exposure.get(market.category, 0.0)
                proposal = self.kett.calculate_proposal(
                    snapshot=snap,
                    estimate=estimate,
                    equity=self.tess.equity,
                    available_cash=self.tess.virtual_cash,
                    current_risk_state=self.tess.risk_state,
                    category_exposure=cat_exp,
                    total_exposure=self.tess.total_exposure,
                )

                if proposal is None:
                    continue

                proposals.append(proposal)
                self.db.save_trade_proposal(proposal, conn=conn)

                # BRAM: Authoritative risk gate
                decision = self.bram.evaluate(
                    proposal=proposal,
                    snapshot=snap,
                    open_positions=self.tess.open_positions,
                    equity=self.tess.equity,
                    virtual_cash=self.tess.virtual_cash,
                    current_risk_state=self.tess.risk_state,
                    current_time=now,
                )
                decisions.append(decision)
                self.db.save_risk_decision(decision, conn=conn)

                # PAPERBROKER: Paper execution simulation
                if decision.accepted:
                    exec_result = self.paper_broker.execute_decision(decision, proposal, snap)
                    if exec_result is not None:
                        order, fill = exec_result
                        executed_orders.append(order)
                        executed_fills.append(fill)

                        self.db.save_paper_order(order, conn=conn)
                        self.db.save_paper_fill(fill, conn=conn)

                        # TESS: Process fill and update portfolio
                        self.tess.process_fill(fill, category=market.category)

            # 5. Final mark-to-market and portfolio snapshot update
            final_portfolio_snap = self.tess.mark_to_market(snapshots, cycle_id=cycle_id, current_time=now)
            finished_at_str = to_iso_utc(now_utc())
            self.db.finish_cycle(cycle_id, finished_at_str, conn=conn)

        return {
            "cycle_id": cycle_id,
            "snapshots_count": len(snapshots),
            "estimates_count": len(estimates),
            "proposals_count": len(proposals),
            "decisions_count": len(decisions),
            "accepted_count": sum(1 for d in decisions if d.accepted),
            "rejected_count": sum(1 for d in decisions if not d.accepted),
            "orders_count": len(executed_orders),
            "fills_count": len(executed_fills),
            "portfolio_equity": final_portfolio_snap.equity,
            "virtual_cash": final_portfolio_snap.virtual_cash,
            "current_drawdown": final_portfolio_snap.current_drawdown,
            "risk_state": final_portfolio_snap.risk_state.value,
            "open_positions_count": final_portfolio_snap.open_position_count,
        }
