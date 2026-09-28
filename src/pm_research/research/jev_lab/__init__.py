"""Phase 9A Jev Decision Research Lab.

Fast typed decision, regime classification, setup gating, and strategy governance.
Strictly Paper-Only / Zero live execution components.
"""

from __future__ import annotations

from pm_research.research.jev_lab.contract import (
    ABSTAIN_CHOICE,
    STATUS_DEV_HYPOTHESIS,
    STATUS_NON_SCORED_EXPLORATORY,
    STATUS_NOT_VALIDATED,
    STATUS_VALIDATED,
    DecisionObservation,
    DecisionResponse,
    DecisionScore,
    DecisionTask,
)
from pm_research.research.jev_lab.evaluator import (
    AggregateEvaluation,
    evaluate_cohort,
    score_single_decision,
)
from pm_research.research.jev_lab.governance import (
    HypothesisAttempt,
    MultipleTestingLedger,
)
from pm_research.research.jev_lab.providers import (
    AlwaysAbstainProvider,
    BaseDecisionProvider,
    DeterministicRuleProvider,
    JevDecisionProvider,
    MajorityBaseRateProvider,
    RandomBaselineProvider,
)
from pm_research.research.jev_lab.storage import JevLabStorage
from pm_research.research.jev_lab.strategy_contract import (
    StrategyCandidate,
    compute_strategy_hash,
)
from pm_research.research.jev_lab.tasks import (
    CANONICAL_TASKS,
    TASK_FLOW_TOXICITY,
    TASK_MARKET_REGIME,
    TASK_SETUP_GATING,
    TASK_SIGNAL_AGREEMENT,
    TaskRegistry,
)

__all__ = [
    "ABSTAIN_CHOICE",
    "STATUS_DEV_HYPOTHESIS",
    "STATUS_NOT_VALIDATED",
    "STATUS_NON_SCORED_EXPLORATORY",
    "STATUS_VALIDATED",
    "DecisionTask",
    "DecisionObservation",
    "DecisionResponse",
    "DecisionScore",
    "AggregateEvaluation",
    "score_single_decision",
    "evaluate_cohort",
    "HypothesisAttempt",
    "MultipleTestingLedger",
    "BaseDecisionProvider",
    "JevDecisionProvider",
    "RandomBaselineProvider",
    "MajorityBaseRateProvider",
    "AlwaysAbstainProvider",
    "DeterministicRuleProvider",
    "JevLabStorage",
    "StrategyCandidate",
    "compute_strategy_hash",
    "TASK_SETUP_GATING",
    "TASK_MARKET_REGIME",
    "TASK_SIGNAL_AGREEMENT",
    "TASK_FLOW_TOXICITY",
    "CANONICAL_TASKS",
    "TaskRegistry",
]
