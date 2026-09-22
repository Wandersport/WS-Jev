"""Validated system configuration for paper trading research."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pm_research.utils import compute_config_hash


@dataclass
class SurvivalTierConfig:
    kelly_multiplier_scale: float
    min_robust_edge_scale: float
    max_position_capital_scale: float
    max_total_exposure_scale: float


@dataclass
class SystemConfig:
    # Model configuration
    model_version: str = "ilsa-base-v1.0"
    probability_slope: float = 1.05
    q_min: float = 0.01
    q_max: float = 0.99

    # Sizing and edge
    base_kelly_multiplier: float = 0.08  # 8% Kelly
    z_uncertainty_multiplier: float = 1.5  # z in: raw_edge - z*sigma_q - costs
    min_robust_edge: float = 0.03  # 3% robust edge required
    estimated_cost_rate: float = 0.005  # 0.5% round-trip cost/slippage penalty
    simulated_fee_rate: float = 0.001  # 0.1% simulated fee on fill

    # Position & portfolio limits
    initial_bankroll: float = 1000.0
    max_single_position_capital: float = 50.0  # Max $50 in one position
    max_single_position_depth_pct: float = 0.10  # Never take more than 10% of visible orderbook depth
    max_total_exposure_pct: float = 0.40  # Max 40% of equity deployed
    max_category_exposure_pct: float = 0.20  # Max 20% of equity in single category
    max_open_positions: int = 15

    # Bram risk gates
    min_liquidity: float = 500.0
    max_spread: float = 0.12
    min_time_to_resolution_hours: float = 1.0
    max_uncertainty: float = 0.18
    max_quote_age_seconds: float = 300.0  # 5 minutes stale check

    # Drawdown thresholds (percentage from high-water mark)
    drawdown_caution_pct: float = 0.05   # 5%
    drawdown_survival_pct: float = 0.10  # 10%
    drawdown_critical_pct: float = 0.20  # 20%
    drawdown_halted_pct: float = 0.30    # 30%

    # Drawdown scaling factors (guaranteeing monotonic risk reduction)
    tier_normal: SurvivalTierConfig = field(
        default_factory=lambda: SurvivalTierConfig(
            kelly_multiplier_scale=1.0,
            min_robust_edge_scale=1.0,
            max_position_capital_scale=1.0,
            max_total_exposure_scale=1.0,
        )
    )
    tier_caution: SurvivalTierConfig = field(
        default_factory=lambda: SurvivalTierConfig(
            kelly_multiplier_scale=0.70,
            min_robust_edge_scale=1.25,
            max_position_capital_scale=0.70,
            max_total_exposure_scale=0.75,
        )
    )
    tier_survival: SurvivalTierConfig = field(
        default_factory=lambda: SurvivalTierConfig(
            kelly_multiplier_scale=0.40,
            min_robust_edge_scale=1.75,
            max_position_capital_scale=0.40,
            max_total_exposure_scale=0.50,
        )
    )
    tier_critical: SurvivalTierConfig = field(
        default_factory=lambda: SurvivalTierConfig(
            kelly_multiplier_scale=0.15,
            min_robust_edge_scale=2.50,
            max_position_capital_scale=0.20,
            max_total_exposure_scale=0.25,
        )
    )
    tier_halted: SurvivalTierConfig = field(
        default_factory=lambda: SurvivalTierConfig(
            kelly_multiplier_scale=0.0,
            min_robust_edge_scale=100.0,
            max_position_capital_scale=0.0,
            max_total_exposure_scale=0.0,
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self.to_dict())
