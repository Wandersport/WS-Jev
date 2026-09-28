"""Candidate research tasks for Phase 9A Jev Decision Research Lab.

RESEARCH HYPOTHESES ONLY:
All tasks defined herein represent non-validated scientific hypotheses.
None of these tasks are claimed to be predictive, profitable, or validated.
Every task explicitly specifies its objective label construction formula
or is marked NON_SCORED_EXPLORATORY.
"""

from __future__ import annotations

from pm_research.research.jev_lab.contract import (
    STATUS_DEV_HYPOTHESIS,
    DecisionTask,
)

# ------------------------------------------------------------------------------
# Task A: Setup Gating / Abstention
# ------------------------------------------------------------------------------
TASK_SETUP_GATING = DecisionTask(
    task_id="setup_gating_v1",
    task_version="1.0.0",
    task_type="SETUP_GATING",
    question=(
        "Given current market microstructure, orderbook depth imbalance, and recent volatility, "
        "is the current 5-minute observation a high-quality decision setup, an unfavorable setup, "
        "or should the system abstain from acting?"
    ),
    allowed_choices=("GOOD_SETUP", "BAD_SETUP"),
    optional_abstain=True,
    state_schema_version="jev-lab-microstructure-v1",
    label_definition=(
        "Objective construction: Computed at round close T_end. If realized spread-adjusted "
        "volatility exceeds noise threshold and price discovery is continuous without gaps, "
        "label is GOOD_SETUP; if spread-to-depth ratio is dislocated or toxic adverse selection "
        "occurs, label is BAD_SETUP."
    ),
    information_cutoff="t_observation",
    target_horizon="5m",
    status=STATUS_DEV_HYPOTHESIS,
    description="Evaluate Jev as a gatekeeper that filters out low-conviction or noisy rounds.",
)

# ------------------------------------------------------------------------------
# Task B: Market Regime Classification
# ------------------------------------------------------------------------------
TASK_MARKET_REGIME = DecisionTask(
    task_id="market_regime_v1",
    task_version="1.0.0",
    task_type="REGIME",
    question=(
        "Classify the prevailing market microstructure regime for the upcoming 5-minute window: "
        "TRENDING, MEAN_REVERTING, or NEUTRAL?"
    ),
    allowed_choices=("TRENDING", "MEAN_REVERTING", "NEUTRAL"),
    optional_abstain=True,
    state_schema_version="jev-lab-microstructure-v1",
    label_definition=(
        "Objective construction: Evaluated over round duration [t_0, t_end]. Computed from "
        "variance-ratio test VR(q) = Var(r_q) / (q * Var(r_1)) and autocorrelation of 1-second "
        "log-returns. If VR > 1.2 with p < 0.05 -> TRENDING; if VR < 0.8 with p < 0.05 -> MEAN_REVERTING; "
        "otherwise NEUTRAL."
    ),
    information_cutoff="t_observation",
    target_horizon="5m",
    status=STATUS_DEV_HYPOTHESIS,
    description="Regime detection to condition secondary statistical models without predicting direction.",
)

# ------------------------------------------------------------------------------
# Task C: Cross-Venue Signal Agreement
# ------------------------------------------------------------------------------
TASK_SIGNAL_AGREEMENT = DecisionTask(
    task_id="signal_agreement_v1",
    task_version="1.0.0",
    task_type="SIGNAL_AGREEMENT",
    question=(
        "Do the Binance high-frequency orderbook/flow signals and Polymarket book prices "
        "exhibit coherent alignment, conflicted pressure, or absence of meaningful signal?"
    ),
    allowed_choices=("EXTERNAL_SIGNAL_COHERENT", "EXTERNAL_SIGNAL_CONFLICTED", "NO_SIGNAL"),
    optional_abstain=True,
    state_schema_version="jev-lab-leadlag-v1",
    label_definition=(
        "Objective construction: Compares signed Binance 60s taker flow sign(TF_60s) with "
        "signed Polymarket 30s price return sign(R_poly_30s). If sign(TF_60s) == sign(R_poly_30s) "
        "and |TF_60s| > 0.1 -> EXTERNAL_SIGNAL_COHERENT; if opposite signs -> EXTERNAL_SIGNAL_CONFLICTED; "
        "if |TF_60s| <= 0.1 -> NO_SIGNAL."
    ),
    information_cutoff="t_observation",
    target_horizon="30s",
    status=STATUS_DEV_HYPOTHESIS,
    description="Detect lead-lag agreement between fast Binance flow and prediction-market adjustment.",
)

# ------------------------------------------------------------------------------
# Task D: Order Flow Toxicity State
# ------------------------------------------------------------------------------
TASK_FLOW_TOXICITY = DecisionTask(
    task_id="flow_toxicity_v1",
    task_version="1.0.0",
    task_type="FLOW_STATE",
    question=(
        "Is the recent aggressive trade flow on Binance and Polymarket TOXIC (informed/adverse), "
        "NON_TOXIC (balanced/noise), or UNCERTAIN?"
    ),
    allowed_choices=("TOXIC", "NON_TOXIC", "UNCERTAIN"),
    optional_abstain=True,
    state_schema_version="jev-lab-microstructure-v1",
    label_definition=(
        "Objective construction: Evaluated from empirical VPIN (Volume-Synchronized Probability of "
        "Toxicity) and realized post-trade price impact over 60 seconds. If adverse price impact "
        "> 3x baseline spread -> TOXIC; if post-trade reversal occurs -> NON_TOXIC; otherwise UNCERTAIN."
    ),
    information_cutoff="t_observation",
    target_horizon="60s",
    status=STATUS_DEV_HYPOTHESIS,
    description="Flow toxicity judgment to prevent adverse selection in passive liquidity placement.",
)

# Canonical task catalog
CANONICAL_TASKS: dict[str, DecisionTask] = {
    TASK_SETUP_GATING.task_id: TASK_SETUP_GATING,
    TASK_MARKET_REGIME.task_id: TASK_MARKET_REGIME,
    TASK_SIGNAL_AGREEMENT.task_id: TASK_SIGNAL_AGREEMENT,
    TASK_FLOW_TOXICITY.task_id: TASK_FLOW_TOXICITY,
}


class TaskRegistry:
    """Registry for managing active, exploratory, and archived research tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, DecisionTask] = dict(CANONICAL_TASKS)

    def register_task(self, task: DecisionTask) -> None:
        """Register a new research task specification."""
        if task.task_id in self._tasks:
            existing = self._tasks[task.task_id]
            if existing.task_version != task.task_version:
                raise ValueError(
                    f"Task {task.task_id} already exists with version {existing.task_version}. "
                    "Cannot overwrite with different version without unregistering."
                )
        self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> DecisionTask:
        """Retrieve registered task by ID."""
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in TaskRegistry. Available: {list(self._tasks.keys())}")
        return self._tasks[task_id]

    def list_tasks(self) -> list[DecisionTask]:
        """List all registered tasks."""
        return list(self._tasks.values())
