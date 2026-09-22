"""PaperBroker: Paper-only local execution simulation.

THIS COMPONENT IS STRUCTURALLY PAPER-ONLY.
It simulates order execution against local order-book depth or conservative slippage models.
It contains NO live trading endpoints, NO external API calls, and NO wallet integrations.
"""

from __future__ import annotations

import logging

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    MarketSnapshot,
    PaperFill,
    PaperOrder,
    RiskDecision,
    TradeProposal,
)
from pm_research.utils import ensure_utc, generate_id, now_utc

logger = logging.getLogger(__name__)


class PaperBroker:
    """Simulates paper order execution and fills against hypothetical market liquidity."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def execute_decision(
        self,
        decision: RiskDecision,
        proposal: TradeProposal,
        snapshot: MarketSnapshot,
    ) -> tuple[PaperOrder, PaperFill] | None:
        """Simulate hypothetical execution for an approved RiskDecision."""
        if not decision.accepted or decision.allocated_contracts <= 0:
            return None

        now = ensure_utc(now_utc())
        order_id = generate_id("porder")
        market = snapshot.market

        # 1. Record Paper Order
        paper_order = PaperOrder(
            paper_order_id=order_id,
            proposal_id=proposal.proposal_id,
            decision_id=decision.decision_id,
            market_id=proposal.market_id,
            side=proposal.side,
            requested_quantity=decision.allocated_contracts,
            limit_price=proposal.quote_price,
            snapshot_id=snapshot.snapshot_id,
            timestamp=now,
        )

        requested_qty = decision.allocated_contracts

        # 2. Simulate Fill
        if market.order_book and market.order_book.asks:
            # Walk order book depth level by level
            # For YES we buy YES asks; for NO we buy NO asks
            # In order book, asks are sorted ascending by price
            filled_qty = 0.0
            accumulated_cost = 0.0
            remaining_to_fill = requested_qty

            for level in market.order_book.asks:
                if remaining_to_fill <= 0:
                    break
                fillable_at_level = min(remaining_to_fill, level.quantity)
                accumulated_cost += fillable_at_level * level.price
                filled_qty += fillable_at_level
                remaining_to_fill -= fillable_at_level

            if filled_qty <= 0.0:
                # No liquidity available in book
                logger.warning(f"PaperBroker: Zero book depth available for {market.market_id}")
                return None

            unfilled_qty = requested_qty - filled_qty
            avg_fill_price = accumulated_cost / filled_qty
            slippage = max(0.0, avg_fill_price - proposal.quote_price)

        else:
            # Conservative documented fallback fill model
            # Never assumes midpoint. Adds positive slippage penalty based on spread.
            spread = market.quote.spread or 0.04
            slippage = max(0.002, spread * 0.20)
            avg_fill_price = min(0.99, proposal.quote_price + slippage)
            filled_qty = requested_qty
            unfilled_qty = 0.0
            accumulated_cost = filled_qty * avg_fill_price

        # Calculate simulated transaction fees
        fees = accumulated_cost * self.config.simulated_fee_rate
        total_cost = accumulated_cost + fees

        fill_id = generate_id("pfill")
        paper_fill = PaperFill(
            fill_id=fill_id,
            paper_order_id=paper_order.paper_order_id,
            proposal_id=proposal.proposal_id,
            market_id=proposal.market_id,
            side=proposal.side,
            requested_quantity=round(requested_qty, 2),
            filled_quantity=round(filled_qty, 2),
            unfilled_quantity=round(unfilled_qty, 2),
            average_fill_price=round(avg_fill_price, 4),
            slippage=round(slippage, 4),
            fees=round(fees, 4),
            total_cost=round(total_cost, 2),
            snapshot_id=snapshot.snapshot_id,
            timestamp=now,
        )

        return paper_order, paper_fill
