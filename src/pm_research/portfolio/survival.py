"""Survival and drawdown state management with monotonic risk reduction guarantees."""

from __future__ import annotations

from pm_research.config import SystemConfig
from pm_research.domain.models import RiskState


class SurvivalManager:
    """Manages risk state transitions based on equity drawdown and enforces monotonic risk reduction."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def determine_risk_state(self, current_drawdown: float) -> RiskState:
        """Determine risk state deterministically from peak-to-trough equity drawdown fraction."""
        dd = max(0.0, current_drawdown)

        if dd >= self.config.drawdown_halted_pct:
            return RiskState.HALTED
        elif dd >= self.config.drawdown_critical_pct:
            return RiskState.CRITICAL
        elif dd >= self.config.drawdown_survival_pct:
            return RiskState.SURVIVAL
        elif dd >= self.config.drawdown_caution_pct:
            return RiskState.CAUTION
        else:
            return RiskState.NORMAL

    def get_tier_config(self, state: RiskState):
        """Retrieve tier scaling configuration for a given risk state."""
        return getattr(self.config, f"tier_{state.value.lower()}", self.config.tier_normal)

    def verify_monotonic_risk_reduction(self) -> tuple[bool, list[str]]:
        """Verify that risk reduction is strictly monotonic across deteriorating risk states.

        As state worsens (NORMAL -> CAUTION -> SURVIVAL -> CRITICAL -> HALTED):
        1. Kelly multiplier scale must NOT increase.
        2. Minimum robust edge scale must NOT decrease.
        3. Max position capital scale must NOT increase.
        4. Max total exposure scale must NOT increase.
        """
        states_order = [
            RiskState.NORMAL,
            RiskState.CAUTION,
            RiskState.SURVIVAL,
            RiskState.CRITICAL,
            RiskState.HALTED,
        ]

        violations: list[str] = []

        for i in range(len(states_order) - 1):
            s_curr = states_order[i]
            s_next = states_order[i + 1]
            cfg_curr = self.get_tier_config(s_curr)
            cfg_next = self.get_tier_config(s_next)

            # 1. Kelly multiplier must not increase
            if cfg_next.kelly_multiplier_scale > cfg_curr.kelly_multiplier_scale:
                violations.append(
                    f"Kelly multiplier increased from {s_curr.value} ({cfg_curr.kelly_multiplier_scale}) "
                    f"to {s_next.value} ({cfg_next.kelly_multiplier_scale})"
                )

            # 2. Min edge must not decrease
            if cfg_next.min_robust_edge_scale < cfg_curr.min_robust_edge_scale:
                violations.append(
                    f"Min robust edge decreased from {s_curr.value} ({cfg_curr.min_robust_edge_scale}) "
                    f"to {s_next.value} ({cfg_next.min_robust_edge_scale})"
                )

            # 3. Max position capital must not increase
            if cfg_next.max_position_capital_scale > cfg_curr.max_position_capital_scale:
                violations.append(
                    f"Max position capital increased from {s_curr.value} ({cfg_curr.max_position_capital_scale}) "
                    f"to {s_next.value} ({cfg_next.max_position_capital_scale})"
                )

            # 4. Max total exposure must not increase
            if cfg_next.max_total_exposure_scale > cfg_curr.max_total_exposure_scale:
                violations.append(
                    f"Max total exposure increased from {s_curr.value} ({cfg_curr.max_total_exposure_scale}) "
                    f"to {s_next.value} ({cfg_next.max_total_exposure_scale})"
                )

        return len(violations) == 0, violations
