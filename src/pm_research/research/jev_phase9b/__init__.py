"""Phase 9B: Meaningful Polymarket Move Setup-Gating Benchmark package."""

from __future__ import annotations

from pm_research.research.jev_phase9b.baselines import (
    AlwaysAbstainBaseline,
    DeterministicRuleBaseline,
    MajorityBaseline,
    RandomBaseline,
    StandardizedLogisticRegression,
)
from pm_research.research.jev_phase9b.dataset import Phase9bDataset, Phase9bSample
from pm_research.research.jev_phase9b.evaluator import (
    EvaluationMetrics,
    compute_metrics,
    compute_selective_prediction_curve,
    run_round_bootstrap,
)
from pm_research.research.jev_phase9b.reporting import generate_phase9b_reports
from pm_research.research.jev_phase9b.runner import Phase9bRunner
from pm_research.research.jev_phase9b.spec import (
    ANCHOR_SECONDS_REMAINING,
    CHOICE_MEANINGFUL_MOVE,
    CHOICE_QUIET,
    TARGET_HORIZON,
    TASK_ID,
    TASK_VERSION,
    compute_task_spec_hash,
    get_phase9b_decision_task,
)

__all__ = [
    "TASK_ID",
    "TASK_VERSION",
    "TARGET_HORIZON",
    "ANCHOR_SECONDS_REMAINING",
    "CHOICE_MEANINGFUL_MOVE",
    "CHOICE_QUIET",
    "compute_task_spec_hash",
    "get_phase9b_decision_task",
    "Phase9bSample",
    "Phase9bDataset",
    "MajorityBaseline",
    "DeterministicRuleBaseline",
    "StandardizedLogisticRegression",
    "RandomBaseline",
    "AlwaysAbstainBaseline",
    "EvaluationMetrics",
    "compute_metrics",
    "compute_selective_prediction_curve",
    "run_round_bootstrap",
    "Phase9bRunner",
    "generate_phase9b_reports",
]
