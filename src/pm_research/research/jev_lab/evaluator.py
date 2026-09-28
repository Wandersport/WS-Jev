"""Evaluation and scoring framework with first-class abstention support.

RESEARCH METRICS ONLY:
Scoring accounts for:
- Coverage rate (N_acted / N_total)
- Abstention rate (N_abstained / N_total)
- Conditional accuracy (accuracy on instances where the model acted)
- Effective overall accuracy (correct / total, preventing high scores through near-total abstention)
- Multiclass calibration & Brier scores where appropriate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pm_research.research.jev_lab.contract import (
    DecisionResponse,
    DecisionScore,
    DecisionTask,
)
from pm_research.utils import generate_id


@dataclass(frozen=True)
class AggregateEvaluation:
    """Comprehensive evaluation summary across a cohort of decision responses."""

    task_id: str
    provider: str
    model: str
    total_observations: int
    total_acted: int
    total_abstained: int
    coverage_rate: float
    abstention_rate: float
    conditional_accuracy: float
    effective_accuracy: float
    mean_brier_score: float | None
    mean_log_loss: float | None
    choice_distribution: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation to dictionary representation."""
        return {
            "task_id": self.task_id,
            "provider": self.provider,
            "model": self.model,
            "total_observations": self.total_observations,
            "total_acted": self.total_acted,
            "total_abstained": self.total_abstained,
            "coverage_rate": self.coverage_rate,
            "abstention_rate": self.abstention_rate,
            "conditional_accuracy": self.conditional_accuracy,
            "effective_accuracy": self.effective_accuracy,
            "mean_brier_score": self.mean_brier_score,
            "mean_log_loss": self.mean_log_loss,
            "choice_distribution": self.choice_distribution,
            "metadata": self.metadata,
        }


def score_single_decision(
    task: DecisionTask,
    response: DecisionResponse,
    objective_label: str | None,
) -> DecisionScore:
    """Score a single response against its objective label.

    Handles ABSTAIN cleanly: if the model abstains, is_abstained=True and
    is_correct=None (or False if scored against an action-required standard).
    """
    if response.choice not in task.valid_choices_set:
        raise ValueError(
            f"Response choice '{response.choice}' is invalid for task {task.task_id}. "
            f"Allowed: {task.valid_choices_set}"
        )

    score_id = generate_id("score")
    now_utc = datetime.now(timezone.utc)

    if response.is_abstained:
        return DecisionScore(
            score_id=score_id,
            response_id=response.response_id,
            task_id=task.task_id,
            objective_label=objective_label,
            is_correct=None,
            is_abstained=True,
            brier_score=None,
            log_loss=None,
            scored_at_utc=now_utc,
            metrics={"status": "ABSTAINED"},
        )

    # Model acted
    if objective_label is None:
        # Exploratory or unlabelled evaluation
        return DecisionScore(
            score_id=score_id,
            response_id=response.response_id,
            task_id=task.task_id,
            objective_label=None,
            is_correct=None,
            is_abstained=False,
            brier_score=None,
            log_loss=None,
            scored_at_utc=now_utc,
            metrics={"status": "UNLABELLED_OBSERVATION"},
        )

    is_correct = response.choice == objective_label

    # Compute multiclass Brier component if confidence is provided
    brier: float | None = None
    log_loss: float | None = None
    if response.confidence is not None:
        conf = max(0.0001, min(0.9999, float(response.confidence)))
        # Brier score: (p_chosen - y)^2 where y=1 if correct, 0 if incorrect
        y = 1.0 if is_correct else 0.0
        brier = round((conf - y) ** 2, 4)
        log_loss = round(-math.log(conf) if is_correct else -math.log(1.0 - conf), 4)

    return DecisionScore(
        score_id=score_id,
        response_id=response.response_id,
        task_id=task.task_id,
        objective_label=objective_label,
        is_correct=is_correct,
        is_abstained=False,
        brier_score=brier,
        log_loss=log_loss,
        scored_at_utc=now_utc,
        metrics={"status": "EVALUATED"},
    )


def evaluate_cohort(
    task: DecisionTask,
    responses_with_labels: list[tuple[DecisionResponse, str | None]],
) -> AggregateEvaluation:
    """Evaluate an entire cohort of decision responses against objective labels.

    Computes coverage, abstention, conditional accuracy, and effective accuracy.
    """
    total = len(responses_with_labels)
    if total == 0:
        return AggregateEvaluation(
            task_id=task.task_id,
            provider="UNKNOWN",
            model="UNKNOWN",
            total_observations=0,
            total_acted=0,
            total_abstained=0,
            coverage_rate=0.0,
            abstention_rate=0.0,
            conditional_accuracy=0.0,
            effective_accuracy=0.0,
            mean_brier_score=None,
            mean_log_loss=None,
        )

    provider = responses_with_labels[0][0].provider
    model = responses_with_labels[0][0].returned_model or responses_with_labels[0][0].requested_model

    choice_dist: dict[str, int] = {}
    acted = 0
    abstained = 0
    correct_when_acted = 0
    brier_scores: list[float] = []
    log_losses: list[float] = []

    for resp, label in responses_with_labels:
        choice_dist[resp.choice] = choice_dist.get(resp.choice, 0) + 1
        score = score_single_decision(task, resp, label)

        if score.is_abstained:
            abstained += 1
        else:
            acted += 1
            if score.is_correct is True:
                correct_when_acted += 1
            if score.brier_score is not None:
                brier_scores.append(score.brier_score)
            if score.log_loss is not None:
                log_losses.append(score.log_loss)

    coverage_rate = round(acted / total, 4)
    abstention_rate = round(abstained / total, 4)
    conditional_accuracy = round(correct_when_acted / acted, 4) if acted > 0 else 0.0
    # Effective accuracy penalizes high abstention: correct / total
    effective_accuracy = round(correct_when_acted / total, 4)
    mean_brier = round(sum(brier_scores) / len(brier_scores), 4) if brier_scores else None
    mean_log_loss = round(sum(log_losses) / len(log_losses), 4) if log_losses else None

    return AggregateEvaluation(
        task_id=task.task_id,
        provider=provider,
        model=model,
        total_observations=total,
        total_acted=acted,
        total_abstained=abstained,
        coverage_rate=coverage_rate,
        abstention_rate=abstention_rate,
        conditional_accuracy=conditional_accuracy,
        effective_accuracy=effective_accuracy,
        mean_brier_score=mean_brier,
        mean_log_loss=mean_log_loss,
        choice_distribution=choice_dist,
    )
