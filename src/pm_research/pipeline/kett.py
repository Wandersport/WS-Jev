"""KETT: Edge detection and hypothetical Kelly position sizing stage."""

from __future__ import annotations

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    MarketSnapshot,
    ProbabilityEstimate,
    RiskState,
    Side,
    TradeProposal,
)
from pm_research.utils import ensure_utc, generate_id, now_utc


class KettSizer:
    """Calculates executable raw edge, robust edge (with uncertainty and cost penalties), and conservative fractional Kelly sizing."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def calculate_proposal(
        self,
        snapshot: MarketSnapshot,
        estimate: ProbabilityEstimate,
        equity: float,
        available_cash: float,
        current_risk_state: RiskState = RiskState.NORMAL,
        category_exposure: float = 0.0,
        total_exposure: float = 0.0,
    ) -> TradeProposal | None:
        """Evaluate executable edge for both YES and NO sides and generate a sizing proposal."""
        market = snapshot.market
        quote = market.quote

        # Get executable ask quotes
        yes_ask = quote.yes_ask
        no_ask = quote.no_ask

        # If executable quotes are not available, cannot calculate actionable executable edge
        if yes_ask is None and no_ask is None:
            return None

        # Determine effective survival tier configuration
        tier_cfg = getattr(self.config, f"tier_{current_risk_state.value.lower()}", self.config.tier_normal)
        effective_kelly_mult = self.config.base_kelly_multiplier * tier_cfg.kelly_multiplier_scale
        effective_max_position_capital = self.config.max_single_position_capital * tier_cfg.max_position_capital_scale

        # Evaluate YES side edge
        edge_yes = -999.0
        robust_edge_yes = -999.0
        if yes_ask is not None and 0.0 < yes_ask < 1.0:
            edge_yes = estimate.q_hat - yes_ask
            robust_edge_yes = (
                edge_yes
                - (self.config.z_uncertainty_multiplier * estimate.uncertainty)
                - self.config.estimated_cost_rate
            )

        # Evaluate NO side edge
        edge_no = -999.0
        robust_edge_no = -999.0
        if no_ask is not None and 0.0 < no_ask < 1.0:
            q_no = 1.0 - estimate.q_hat
            edge_no = q_no - no_ask
            robust_edge_no = (
                edge_no
                - (self.config.z_uncertainty_multiplier * estimate.uncertainty)
                - self.config.estimated_cost_rate
            )

        # Pick the side with higher robust edge
        if robust_edge_yes >= robust_edge_no and yes_ask is not None:
            chosen_side = Side.YES
            quote_price = yes_ask
            q_for_side = estimate.q_hat
            raw_edge = edge_yes
            robust_edge = robust_edge_yes
        elif no_ask is not None:
            chosen_side = Side.NO
            quote_price = no_ask
            q_for_side = 1.0 - estimate.q_hat
            raw_edge = edge_no
            robust_edge = robust_edge_no
        else:
            return None

        p = quote_price
        if p >= 1.0 or p <= 0.0:
            return None

        # Fractional Kelly sizing:
        # Full Kelly fraction: f* = (q - p) / (1 - p) for binary contracts
        # If q <= p, full kelly is 0
        if q_for_side > p:
            f_full = (q_for_side - p) / (1.0 - p)
            f_candidate = effective_kelly_mult * f_full
            unconstrained_capital = f_candidate * max(0.0, equity)
        else:
            f_full = 0.0
            unconstrained_capital = 0.0

        # Sizing constraints and caps:
        reasons: list[str] = [f"ROBUST_EDGE_{robust_edge:.4f}"]

        # 1. Single position capital cap
        capped_capital = min(unconstrained_capital, effective_max_position_capital)
        if unconstrained_capital > 0 and capped_capital < unconstrained_capital:
            reasons.append("CAPPED_BY_SINGLE_POSITION_LIMIT")

        # 2. Available cash cap
        if capped_capital > available_cash:
            capped_capital = max(0.0, available_cash)
            reasons.append("CAPPED_BY_AVAILABLE_CASH")

        # 3. Category exposure cap
        max_cat_cap = self.config.max_category_exposure_pct * equity * tier_cfg.max_total_exposure_scale
        remaining_cat_capacity = max(0.0, max_cat_cap - category_exposure)
        if capped_capital > remaining_cat_capacity:
            capped_capital = remaining_cat_capacity
            reasons.append("CAPPED_BY_CATEGORY_LIMIT")

        # 4. Total exposure cap
        max_tot_cap = self.config.max_total_exposure_pct * equity * tier_cfg.max_total_exposure_scale
        remaining_tot_capacity = max(0.0, max_tot_cap - total_exposure)
        if capped_capital > remaining_tot_capacity:
            capped_capital = remaining_tot_capacity
            reasons.append("CAPPED_BY_TOTAL_EXPOSURE_LIMIT")

        # 5. Order book depth cap (if order book exists)
        if market.order_book:
            total_ask_depth = market.order_book.total_ask_depth
            if total_ask_depth > 0:
                max_contracts_from_depth = total_ask_depth * self.config.max_single_position_depth_pct
                depth_cap_capital = max_contracts_from_depth * p
                if capped_capital > depth_cap_capital:
                    capped_capital = depth_cap_capital
                    reasons.append("CAPPED_BY_ORDERBOOK_DEPTH")

        proposed_contracts = (capped_capital / p) if p > 0 else 0.0

        return TradeProposal(
            proposal_id=generate_id("prop"),
            cycle_id=snapshot.cycle_id,
            market_id=market.market_id,
            side=chosen_side,
            quote_price=round(quote_price, 4),
            q_hat=round(estimate.q_hat, 4),
            sigma_q=round(estimate.uncertainty, 4),
            model_version=estimate.model_version,
            raw_edge=round(raw_edge, 4),
            robust_edge=round(robust_edge, 4),
            proposed_size_contracts=round(proposed_contracts, 2),
            proposed_capital=round(capped_capital, 2),
            reason_codes=reasons,
            timestamp=ensure_utc(now_utc()),
        )
