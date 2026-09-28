"""Generic typed decision contract for Phase 9A Jev Decision Research Lab.

RESEARCH USE ONLY:
This module defines an immutable, typed schema for evaluating fast judgment,
regime classification, signal agreement, and setup gating.
It is structurally isolated from order routing, execution, wallets, and trading.
ZERO execution semantics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pm_research.utils import ensure_utc, to_iso_utc

# Canonical abstention token
ABSTAIN_CHOICE: str = "ABSTAIN"

# Allowed task statuses
STATUS_DEV_HYPOTHESIS: str = "DEVELOPMENT_HYPOTHESIS"
STATUS_NOT_VALIDATED: str = "NOT_VALIDATED"
STATUS_NON_SCORED_EXPLORATORY: str = "NON_SCORED_EXPLORATORY"
STATUS_VALIDATED: str = "VALIDATED"


@dataclass(frozen=True)
class DecisionTask:
    """Immutable definition of a discrete scientific judgment or gating task."""

    task_id: str
    task_version: str
    task_type: str
    question: str
    allowed_choices: tuple[str, ...]
    optional_abstain: bool = True
    state_schema_version: str = "jev-lab-v1"
    label_definition: str = ""
    information_cutoff: str = "t_minus_0"
    target_horizon: str = "5m"
    status: str = STATUS_DEV_HYPOTHESIS
    description: str = ""

    def __post_init__(self) -> None:
        if not self.task_id or not self.task_version:
            raise ValueError("task_id and task_version must be non-empty")
        if not self.allowed_choices:
            raise ValueError("allowed_choices must contain at least one discrete option")
        if ABSTAIN_CHOICE in self.allowed_choices and not self.optional_abstain:
            raise ValueError("If ABSTAIN is in allowed_choices, optional_abstain must be True")

    @property
    def valid_choices_set(self) -> frozenset[str]:
        """All permissible response choices including ABSTAIN if enabled."""
        choices = set(self.allowed_choices)
        if self.optional_abstain:
            choices.add(ABSTAIN_CHOICE)
        return frozenset(choices)

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_version": self.task_version,
            "task_type": self.task_type,
            "question": self.question,
            "allowed_choices": list(self.allowed_choices),
            "optional_abstain": self.optional_abstain,
            "state_schema_version": self.state_schema_version,
            "label_definition": self.label_definition,
            "information_cutoff": self.information_cutoff,
            "target_horizon": self.target_horizon,
            "status": self.status,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionTask:
        """Construct DecisionTask from dictionary."""
        return cls(
            task_id=str(data["task_id"]),
            task_version=str(data["task_version"]),
            task_type=str(data.get("task_type", "GENERAL")),
            question=str(data["question"]),
            allowed_choices=tuple(data["allowed_choices"]),
            optional_abstain=bool(data.get("optional_abstain", True)),
            state_schema_version=str(data.get("state_schema_version", "jev-lab-v1")),
            label_definition=str(data.get("label_definition", "")),
            information_cutoff=str(data.get("information_cutoff", "t_minus_0")),
            target_horizon=str(data.get("target_horizon", "5m")),
            status=str(data.get("status", STATUS_DEV_HYPOTHESIS)),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class DecisionObservation:
    """Immutable input state bundle submitted to a decision provider."""

    observation_id: str
    task_id: str
    market_round_id: str
    as_of_ts_utc: datetime
    state_payload: dict[str, Any]
    provenance: str
    state_hash: str = ""

    def __post_init__(self) -> None:
        if not self.state_hash:
            computed = self.compute_state_hash(self.state_payload)
            object.__setattr__(self, "state_hash", computed)

    @staticmethod
    def compute_state_hash(payload: dict[str, Any]) -> str:
        """Compute canonical SHA-256 fingerprint for deterministic state matching."""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert observation to dictionary."""
        return {
            "observation_id": self.observation_id,
            "task_id": self.task_id,
            "market_round_id": self.market_round_id,
            "as_of_ts": to_iso_utc(ensure_utc(self.as_of_ts_utc)),
            "state_payload": self.state_payload,
            "provenance": self.provenance,
            "state_hash": self.state_hash,
        }


@dataclass(frozen=True)
class DecisionResponse:
    """Immutable response generated by a judgment provider (Jev, baseline, or challenger)."""

    response_id: str
    observation_id: str
    task_id: str
    provider: str
    requested_model: str
    returned_model: str
    choice: str
    confidence: float | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost: float | None
    raw_response_hash: str
    request_hash: str
    timestamp_utc: datetime
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def is_abstained(self) -> bool:
        """True if the model chose to abstain from making an active judgment."""
        return self.choice == ABSTAIN_CHOICE

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "response_id": self.response_id,
            "observation_id": self.observation_id,
            "task_id": self.task_id,
            "provider": self.provider,
            "requested_model": self.requested_model,
            "returned_model": self.returned_model,
            "choice": self.choice,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "raw_response_hash": self.raw_response_hash,
            "request_hash": self.request_hash,
            "timestamp": to_iso_utc(ensure_utc(self.timestamp_utc)),
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class DecisionScore:
    """Scientific scoring of a single decision observation against an objective label."""

    score_id: str
    response_id: str
    task_id: str
    objective_label: str | None
    is_correct: bool | None
    is_abstained: bool
    brier_score: float | None
    log_loss: float | None
    scored_at_utc: datetime
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert score to dictionary."""
        return {
            "score_id": self.score_id,
            "response_id": self.response_id,
            "task_id": self.task_id,
            "objective_label": self.objective_label,
            "is_correct": self.is_correct,
            "is_abstained": self.is_abstained,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "scored_at": to_iso_utc(ensure_utc(self.scored_at_utc)),
            "metrics": self.metrics,
        }
