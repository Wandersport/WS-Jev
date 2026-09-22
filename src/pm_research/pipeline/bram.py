"""BRAM: Authoritative risk gatekeeper."""

from __future__ import annotations

from datetime import datetime

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    MarketSnapshot,
    MarketStatus,
    PaperPosition,
    RiskDecision,
    RiskState,
    TradeProposal,
)
from pm_research.utils import ensure_utc, generate_id, now_utc


class BramRiskGate:
    """Evaluates proposals against strict risk constraints and outputs an authoritative RiskDecision."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def evaluate(
        self,
        proposal: TradeProposal,
        snapshot: MarketSnapshot,
        open_positions: list[PaperPosition],
        equity: float,
        virtual_cash: float,
        current_risk_state: RiskState,
        current_time: datetime | None = None,
    ) -> RiskDecision:
        """Enforce strict risk limits and decide whether to ACCEPT or REJECT proposal."""
        now = ensure_utc(current_time or now_utc())
        market = snapshot.market
        quote = market.quote

        def reject(code: str) -> RiskDecision:
            return RiskDecision(
                decision_id=generate_id("dec"),
                proposal_id=proposal.proposal_id,
                cycle_id=proposal.cycle_id,
                market_id=proposal.market_id,
                accepted=False,
                allocated_contracts=0.0,
                allocated_capital=0.0,
                reason_code=code,
                risk_state=current_risk_state,
                timestamp=now,
            )

        # 1. Survival State: HALTED blocks all new positions
        if current_risk_state == RiskState.HALTED:
            return reject("REJECT_HALTED_MODE")

        # 2. Market status check: must be ACTIVE
        if market.status != MarketStatus.ACTIVE:
            return reject("REJECT_MARKET_NOT_ACTIVE")

        # 3. Data staleness check
        quote_age = (now - snapshot.timestamp).total_seconds()
        if quote_age > self.config.max_quote_age_seconds:
            return reject("REJECT_STALE_DATA")

        # 4. Valid quote checks
        if quote.yes_bid is not None and quote.yes_ask is not None:
            if quote.yes_bid >= quote.yes_ask:
                return reject("REJECT_INVALID_QUOTE_CROSSED")
        if proposal.quote_price <= 0.0 or proposal.quote_price >= 1.0:
            return reject("REJECT_INVALID_QUOTE_BOUNDS")

        # 5. Minimum liquidity check
        if quote.liquidity < self.config.min_liquidity:
            return reject("REJECT_LOW_LIQUIDITY")

        # 6. Maximum spread check
        if quote.spread is not None and quote.spread > self.config.max_spread:
            return reject("REJECT_HIGH_SPREAD")

        # 7. Minimum time to resolution
        hours_to_res = (market.resolution_time - now).total_seconds() / 3600.0
        if hours_to_res < self.config.min_time_to_resolution_hours:
            return reject("REJECT_EXPIRING_SOON")

        # 8. Uncertainty threshold check
        if proposal.sigma_q > self.config.max_uncertainty:
            return reject("REJECT_HIGH_UNCERTAINTY")

        # 9. Robust edge threshold check (scaled by survival tier)
        tier_cfg = getattr(self.config, f"tier_{current_risk_state.value.lower()}", self.config.tier_normal)
        effective_min_edge = self.config.min_robust_edge * tier_cfg.min_robust_edge_scale
        if proposal.robust_edge < effective_min_edge:
            return reject("REJECT_ROBUST_EDGE_TOO_LOW")

        # 10. Maximum open positions check
        if len(open_positions) >= self.config.max_open_positions:
            return reject("REJECT_MAX_OPEN_POSITIONS")

        # 11. Duplicate position protection
        for pos in open_positions:
            if pos.market_id == proposal.market_id:
                return reject("REJECT_DUPLICATE_POSITION")

        # 12. Virtual cash check
        min_order_capital = 1.0
        if virtual_cash < min_order_capital or proposal.proposed_capital > virtual_cash:
            if virtual_cash < min_order_capital:
                return reject("REJECT_INSUFFICIENT_CASH")
            # If partial cash is available, we cap it rather than full reject, provided it meets min_order_capital
            pass

        # 13. Total exposure check
        current_total_exposure = sum(pos.current_value for pos in open_positions)
        max_total_exposure = self.config.max_total_exposure_pct * equity * tier_cfg.max_total_exposure_scale
        if current_total_exposure + proposal.proposed_capital > max_total_exposure:
            return reject("REJECT_MAX_EXPOSURE")

        # 14. Category concentration check
        category_exposure = sum(pos.current_value for pos in open_positions if pos.category == market.category)
        max_cat_exposure = self.config.max_category_exposure_pct * equity * tier_cfg.max_total_exposure_scale
        if category_exposure + proposal.proposed_capital > max_cat_exposure:
            return reject("REJECT_CATEGORY_CONCENTRATION")

        # 15. Single position capital cap
        effective_max_cap = self.config.max_single_position_capital * tier_cfg.max_position_capital_scale
        allocated_capital = min(proposal.proposed_capital, virtual_cash, effective_max_cap)
        if allocated_capital < min_order_capital:
            return reject("REJECT_CAPITAL_BELOW_MINIMUM")

        allocated_contracts = allocated_capital / proposal.quote_price

        # Approved!
        return RiskDecision(
            decision_id=generate_id("dec"),
            proposal_id=proposal.proposal_id,
            cycle_id=proposal.cycle_id,
            market_id=proposal.market_id,
            accepted=True,
            allocated_contracts=round(allocated_contracts, 2),
            allocated_capital=round(allocated_capital, 2),
            reason_code="ACCEPT",
            risk_state=current_risk_state,
            timestamp=now,
        )
