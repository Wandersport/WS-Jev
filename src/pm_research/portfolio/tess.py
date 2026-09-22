"""TESS: Paper portfolio accounting and risk management.

STRUCTURALLY PAPER-ONLY: Contains no external execution capability.
Tracks virtual bankroll, open paper positions, mark-to-market valuations,
realized/unrealized P&L, drawdown state, and calibration observations upon resolution.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    CalibrationObservation,
    MarketSnapshot,
    PaperFill,
    PaperPosition,
    PortfolioSnapshot,
    RiskState,
    Side,
)
from pm_research.portfolio.survival import SurvivalManager
from pm_research.storage.db import Database
from pm_research.utils import ensure_utc, generate_id, now_utc

logger = logging.getLogger(__name__)


class TessPortfolio:
    """Paper portfolio manager maintaining hypothetical capital, positions, and risk states."""

    def __init__(self, config: SystemConfig, db: Database) -> None:
        self.config = config
        self.db = db
        self.survival_manager = SurvivalManager(config)

        # Restore from database or initialize fresh
        latest_snap = self.db.get_latest_portfolio_snapshot()
        if latest_snap:
            self.bankroll_initial = latest_snap.bankroll_initial
            self.virtual_cash = latest_snap.virtual_cash
            self.realized_pnl = latest_snap.realized_pnl
            self.high_water_mark = latest_snap.high_water_mark
            self.max_drawdown = latest_snap.max_drawdown
            self.risk_state = latest_snap.risk_state
        else:
            self.bankroll_initial = config.initial_bankroll
            self.virtual_cash = config.initial_bankroll
            self.realized_pnl = 0.0
            self.high_water_mark = config.initial_bankroll
            self.max_drawdown = 0.0
            self.risk_state = RiskState.NORMAL

        self.positions: dict[str, PaperPosition] = {
            pos.position_id: pos for pos in self.db.get_open_positions()
        }

    @property
    def open_positions(self) -> list[PaperPosition]:
        return [p for p in self.positions.values() if p.status == "OPEN"]

    @property
    def positions_value(self) -> float:
        return sum(p.current_value for p in self.open_positions)

    @property
    def equity(self) -> float:
        return self.virtual_cash + self.positions_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.open_positions)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def current_drawdown(self) -> float:
        if self.high_water_mark <= 0.0:
            return 0.0
        return max(0.0, (self.high_water_mark - self.equity) / self.high_water_mark)

    @property
    def total_exposure(self) -> float:
        return self.positions_value

    @property
    def category_exposure(self) -> dict[str, float]:
        cat_exp: dict[str, float] = {}
        for p in self.open_positions:
            cat_exp[p.category] = cat_exp.get(p.category, 0.0) + p.current_value
        return cat_exp

    def process_fill(self, fill: PaperFill, category: str) -> PaperPosition:
        """Process a simulated fill from PaperBroker, deduct cash, and create/update a position."""
        # Deduct total cash (cost of contracts + fees)
        self.virtual_cash -= fill.total_cost

        pos_id = generate_id("pos")
        position = PaperPosition(
            position_id=pos_id,
            market_id=fill.market_id,
            side=fill.side,
            quantity=fill.filled_quantity,
            average_entry_price=fill.average_fill_price,
            total_cost=fill.total_cost,
            current_price=fill.average_fill_price,
            current_value=fill.filled_quantity * fill.average_fill_price,
            unrealized_pnl=-fill.fees,  # Immediate unrealized pnl reflects fees paid
            realized_pnl=0.0,
            category=category,
            status="OPEN",
            opened_at=fill.timestamp,
        )

        self.positions[pos_id] = position
        self.db.save_paper_position(position)
        return position

    def mark_to_market(
        self,
        snapshots: list[MarketSnapshot],
        cycle_id: str,
        current_time: datetime | None = None,
    ) -> PortfolioSnapshot:
        """Mark open positions against current market snapshot prices and update drawdown/survival states."""
        now = ensure_utc(current_time or now_utc())
        snapshot_by_market = {s.market_id: s for s in snapshots}

        for pos in self.open_positions:
            snap = snapshot_by_market.get(pos.market_id)
            if not snap:
                continue

            quote = snap.market.quote
            # Determine mark price for the contract with proper side handling
            if pos.side == Side.YES:
                mark_price = quote.midpoint or quote.last_price or quote.yes_bid or pos.current_price
            else:
                if quote.no_bid is not None:
                    mark_price = quote.no_bid
                elif quote.midpoint is not None:
                    mark_price = 1.0 - quote.midpoint
                elif quote.last_price is not None:
                    mark_price = 1.0 - quote.last_price
                else:
                    mark_price = pos.current_price

            if not math.isfinite(mark_price):
                mark_price = pos.current_price

            mark_price = max(0.0, min(1.0, mark_price))
            pos.current_price = mark_price
            pos.current_value = pos.quantity * mark_price
            pos.unrealized_pnl = pos.current_value - pos.total_cost
            self.db.save_paper_position(pos)

        # Update high-water mark and drawdown
        eq = self.equity
        if eq > self.high_water_mark:
            self.high_water_mark = eq

        curr_dd = self.current_drawdown
        if curr_dd > self.max_drawdown:
            self.max_drawdown = curr_dd

        # Deterministic risk state transition
        self.risk_state = self.survival_manager.determine_risk_state(curr_dd)

        snap = PortfolioSnapshot(
            snapshot_id=generate_id("psnap"),
            cycle_id=cycle_id,
            timestamp=now,
            bankroll_initial=self.bankroll_initial,
            virtual_cash=round(self.virtual_cash, 2),
            positions_value=round(self.positions_value, 2),
            equity=round(self.equity, 2),
            realized_pnl=round(self.realized_pnl, 2),
            unrealized_pnl=round(self.unrealized_pnl, 2),
            total_pnl=round(self.total_pnl, 2),
            high_water_mark=round(self.high_water_mark, 2),
            current_drawdown=round(curr_dd, 4),
            max_drawdown=round(self.max_drawdown, 4),
            total_exposure=round(self.total_exposure, 2),
            category_exposure={k: round(v, 2) for k, v in self.category_exposure.items()},
            open_position_count=len(self.open_positions),
            risk_state=self.risk_state,
        )

        self.db.save_portfolio_snapshot(snap)
        return snap

    def resolve_market(
        self,
        market_id: str,
        resolved_outcome: Side,
        cycle_id: str,
        resolved_at: datetime | None = None,
    ) -> list[CalibrationObservation]:
        """Settle all paper positions for a resolved market and compute calibration observations idempotently."""
        now = ensure_utc(resolved_at or now_utc())

        # Idempotency check: Guard against double-settlement
        prior_res = self.db.get_resolution(market_id)
        if prior_res is not None:
            logger.warning(
                f"Market {market_id} is already resolved to {prior_res[0]}. "
                "Idempotency guard active: returning existing calibration observations without duplicate payout."
            )
            return [
                obs for obs in self.db.get_all_calibration_observations()
                if obs.market_id == market_id
            ]

        # Execute resolution atomically
        with self.db.transaction() as conn:
            self.db.save_resolution(market_id, resolved_outcome, now.isoformat(), conn=conn)

            # 1. Close open positions for this market
            for pos in list(self.open_positions):
                if pos.market_id != market_id:
                    continue

                if pos.side == resolved_outcome:
                    # Won: each contract pays 1.0 virtual dollar
                    payout = pos.quantity * 1.0
                    pnl = payout - pos.total_cost
                else:
                    # Lost: payout is 0.0
                    payout = 0.0
                    pnl = -pos.total_cost

                self.virtual_cash += payout
                self.realized_pnl += pnl

                pos.status = "CLOSED"
                pos.closed_at = now
                pos.resolved_outcome = resolved_outcome
                pos.current_value = 0.0
                pos.unrealized_pnl = 0.0
                pos.realized_pnl = round(pnl, 2)
                self.db.save_paper_position(pos, conn=conn)

            # 2. Calibration observations for all estimates made for this market
            actual_val = 1.0 if resolved_outcome == Side.YES else 0.0
            observations: list[CalibrationObservation] = []
            estimates = self.db.get_market_estimates(market_id)

            for est in estimates:
                q = est.q_hat
                brier = (q - actual_val) ** 2

                # Bounded log loss
                eps = 1e-6
                q_clamped = max(eps, min(1.0 - eps, q))
                logloss = -(actual_val * math.log(q_clamped) + (1.0 - actual_val) * math.log(1.0 - q_clamped))

                # Edge bucket categorization
                edge = abs(q - est.reference_probability)
                if edge < 0.03:
                    edge_bucket = "0.00-0.03"
                elif edge < 0.06:
                    edge_bucket = "0.03-0.06"
                elif edge < 0.10:
                    edge_bucket = "0.06-0.10"
                else:
                    edge_bucket = "0.10+"

                # Deterministic observation ID to guarantee idempotency
                obs_id = f"calib_{est.estimate_id}"
                obs = CalibrationObservation(
                    observation_id=obs_id,
                    estimate_id=est.estimate_id,
                    market_id=market_id,
                    model_version=est.model_version,
                    predicted_probability=round(q, 4),
                    actual_outcome=actual_val,
                    category="MARKET",
                    edge_bucket=edge_bucket,
                    risk_state=self.risk_state.value,
                    brier_score=round(brier, 4),
                    log_loss=round(logloss, 4),
                    resolved_at=now,
                )
                self.db.save_calibration_observation(obs, conn=conn)
                observations.append(obs)

        # Re-mark equity after settlement
        self.mark_to_market([], cycle_id=cycle_id, current_time=now)
        return observations
