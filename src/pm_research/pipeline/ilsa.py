"""ILSA: Transparent probability estimation engine."""

from __future__ import annotations

import math

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    MarketSnapshot,
    ProbabilityEstimate,
    ResearchFeatures,
)
from pm_research.pipeline.uncertainty import UncertaintyModel
from pm_research.utils import ensure_utc, generate_id, now_utc


def logit(p: float, eps: float = 1e-4) -> float:
    """Compute the logit of probability p, bounded away from 0 and 1."""
    p_clamped = max(eps, min(1.0 - eps, p))
    return math.log(p_clamped / (1.0 - p_clamped))


def sigmoid(z: float) -> float:
    """Compute standard logistic sigmoid."""
    if z < -40.0:
        return 0.0
    if z > 40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class IlsaEstimator:
    """Estimates calibrated probability q_hat = P(YES) using a transparent logit baseline and bounded feature adjustments."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.uncertainty_model = UncertaintyModel(config)

    def estimate(
        self,
        snapshot: MarketSnapshot,
        features: ResearchFeatures | None = None,
    ) -> ProbabilityEstimate:
        """Produce a ProbabilityEstimate for a given market snapshot."""
        market = snapshot.market
        quote = market.quote

        # 1. Determine reference probability (midpoint preferred, fallback to last price or 0.5)
        if quote.midpoint is not None:
            p_ref = quote.midpoint
        elif quote.last_price is not None:
            p_ref = quote.last_price
        else:
            p_ref = 0.50

        # Clamp reference probability to valid range
        p_ref = max(0.001, min(0.999, p_ref))

        # 2. Baseline logit-adjusted probability:
        # q_base = sigmoid(slope * logit(p_ref))
        z = self.config.probability_slope * logit(p_ref)
        q_base = sigmoid(z)

        contributions: dict[str, float] = {
            "p_ref": round(p_ref, 4),
            "q_base": round(q_base, 4),
            "slope": self.config.probability_slope,
        }

        # 3. Bounded feature adjustments from Holt
        adjustment_total = 0.0
        missing_count = 0
        if features and features.values:
            # A. Fundamental research signal adjustment: bounded in [-0.20, +0.20]
            signal = features.values.get("research_signal", 0.0)
            sig_adj = max(-0.20, min(0.20, signal))
            contributions["adj_research_signal"] = round(sig_adj, 4)
            adjustment_total += sig_adj

            # B. Momentum feature adjustment: bounded in [-0.08, +0.08]
            mom = features.values.get("momentum_24h", 0.0)
            mom_adj = max(-0.08, min(0.08, mom * 0.20))
            contributions["adj_momentum"] = round(mom_adj, 4)
            adjustment_total += mom_adj

            # C. Order book imbalance adjustment: bounded in [-0.04, +0.04]
            ob_imb = features.values.get("order_book_imbalance", 0.0)
            ob_adj = max(-0.04, min(0.04, ob_imb * 0.06))
            contributions["adj_order_book_imbalance"] = round(ob_adj, 4)
            adjustment_total += ob_adj

            # Time to resolution
            t_res = features.values.get("hours_to_resolution", 100.0)
            contributions["hours_to_resolution"] = round(t_res, 1)
        else:
            missing_count = 2

        # 4. Final clamped probability q_hat
        q_raw = q_base + adjustment_total
        q_hat = max(self.config.q_min, min(self.config.q_max, q_raw))
        contributions["q_raw"] = round(q_raw, 4)
        contributions["q_final"] = round(q_hat, 4)

        # 5. Uncertainty sigma_q estimation
        age_sec = (now_utc() - snapshot.timestamp).total_seconds()
        sigma_q, uncertainty_breakdown = self.uncertainty_model.estimate_uncertainty(
            market,
            age_seconds=max(0.0, age_sec),
            missing_features_count=missing_count,
        )
        for k, v in uncertainty_breakdown.items():
            contributions[f"sigma_{k}"] = round(v, 4)

        return ProbabilityEstimate(
            estimate_id=generate_id("est"),
            cycle_id=snapshot.cycle_id,
            market_id=market.market_id,
            snapshot_id=snapshot.snapshot_id,
            model_version=self.config.model_version,
            q_hat=round(q_hat, 4),
            reference_probability=round(p_ref, 4),
            uncertainty=round(sigma_q, 4),
            feature_contributions=contributions,
            timestamp=ensure_utc(snapshot.timestamp),
        )
