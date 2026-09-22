"""Uncertainty estimation for probability forecasts."""

from __future__ import annotations

import math

from pm_research.config import SystemConfig
from pm_research.domain.models import Market


class UncertaintyModel:
    """Estimates explicit epistemic/aleatoric uncertainty sigma_q on probability estimates.

    Uncertainty penalties stem from:
    1. Low liquidity penalty: Thin markets have noisy price discovery and higher manipulation risk.
    2. Wide spread penalty: High bid-ask spreads indicate market maker uncertainty.
    3. Stale data penalty: Older quotes carry stale information.
    4. Time to resolution penalty: Very long horizon predictions carry higher variance.
    5. Order book asymmetry: Highly skewed books imply liquidity fragility.
    """

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def estimate_uncertainty(
        self,
        market: Market,
        age_seconds: float = 0.0,
        missing_features_count: int = 0,
    ) -> tuple[float, dict[str, float]]:
        """Calculate sigma_q and return component breakdown."""
        penalties: dict[str, float] = {}

        # Base uncertainty floor
        base_sigma = 0.02
        penalties["base"] = base_sigma

        # 1. Spread penalty: wider spread -> higher uncertainty
        spread = market.quote.spread or 0.05
        # Spread above 0.02 adds uncertainty
        spread_penalty = min(0.08, max(0.0, (spread - 0.02) * 0.5))
        penalties["spread"] = spread_penalty

        # 2. Liquidity penalty: liquidity below reference (e.g. $5,000) adds penalty
        liq = max(1.0, market.quote.liquidity)
        ref_liq = 5000.0
        if liq < ref_liq:
            # Logarithmic penalty for thin markets
            liq_penalty = min(0.08, 0.02 * math.log(ref_liq / liq))
        else:
            liq_penalty = 0.0
        penalties["liquidity"] = liq_penalty

        # 3. Staleness penalty
        if age_seconds > 60.0:
            stale_penalty = min(0.06, (age_seconds - 60.0) / 600.0 * 0.05)
        else:
            stale_penalty = 0.0
        penalties["staleness"] = stale_penalty

        # 4. Missing features penalty
        missing_penalty = min(0.05, missing_features_count * 0.015)
        penalties["missing_features"] = missing_penalty

        # Total uncertainty sigma_q
        total_sigma = sum(penalties.values())
        # Cap total uncertainty at 0.35 to avoid unbounded penalties
        total_sigma = min(0.35, max(0.01, total_sigma))

        return total_sigma, penalties
