"""Decision judgment provider interfaces, OpenRouter Jev adapter, and baselines.

RESEARCH USE ONLY:
- Supports Jev (typesafe/jev-1.13) via OpenRouter /api/alpha/decisions through dedicated client.
- Transparent deterministic baselines (Random, Majority, Rule-based, AlwaysAbstain).
- Model-neutral architecture prepared for future challengers (Laya, Kev, CLM).
- ZERO live execution components. ZERO trading authentication.
"""

from __future__ import annotations

import abc
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable

from pm_research.research.jev_lab.contract import (
    ABSTAIN_CHOICE,
    DecisionObservation,
    DecisionResponse,
    DecisionTask,
)
from pm_research.research.jev_openrouter import JEV_MODEL_PIN, JevOpenRouterClient
from pm_research.utils import generate_id

logger = logging.getLogger(__name__)

PINNED_JEV_MODEL: str = JEV_MODEL_PIN


class BaseDecisionProvider(abc.ABC):
    """Abstract model-neutral provider interface for fast typed judgments."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Name identifying the provider subsystem."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Name of the model or algorithm."""
        ...

    @abc.abstractmethod
    def evaluate(
        self,
        task: DecisionTask,
        observation: DecisionObservation,
    ) -> DecisionResponse:
        """Execute judgment on an observation. Returns DecisionResponse."""
        ...


class JevDecisionProvider(BaseDecisionProvider):
    """Production research provider connecting to OpenRouter decisions endpoint with TypeSafe Jev."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.requested_model_pin = PINNED_JEV_MODEL
        self._client = JevOpenRouterClient(
            api_key=api_key,
            request_timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def provider_name(self) -> str:
        return "openrouter_jev"

    @property
    def model_name(self) -> str:
        return self.requested_model_pin

    def build_request_payload(
        self,
        task: DecisionTask,
        observation: DecisionObservation,
    ) -> dict[str, Any]:
        """Construct canonical OpenRouter /api/alpha/decisions request payload."""
        criteria_dict: dict[str, str] = {}
        for c in task.allowed_choices:
            criteria_dict[c] = f"Condition evaluates to {c} under the specified criteria."
        if task.optional_abstain:
            criteria_dict[ABSTAIN_CHOICE] = (
                "Abstain from choosing an active label due to high uncertainty, "
                "conflicting signals, or noisy microstructure."
            )

        payload = {
            "model": self.requested_model_pin,
            "state": observation.state_payload,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": task.question,
                    "criteria": criteria_dict,
                }
            },
        }
        return payload

    def evaluate(
        self,
        task: DecisionTask,
        observation: DecisionObservation,
    ) -> DecisionResponse:
        """Submit observation to Jev via OpenRouter decisions endpoint."""
        payload = self.build_request_payload(task, observation)
        data, latency_ms, raw_hash, request_hash = self._client.query_typed_decision(payload)

        # Validate response structure
        returned_model = str(data.get("model", self.requested_model_pin))
        answers_map = data.get("answers") or data.get("choices") or data.get("decisions") or {}
        decision_obj = answers_map.get("decision") or {}

        raw_choice: str | None = None
        confidence: float | None = None

        if isinstance(decision_obj, dict):
            raw_choice = decision_obj.get("choice") or decision_obj.get("value")
            confidence = decision_obj.get("confidence")
        elif isinstance(decision_obj, str):
            raw_choice = decision_obj

        if raw_choice is None:
            # Check top-level result formats
            if "result" in data and isinstance(data["result"], dict):
                raw_choice = data["result"].get("choice")
                confidence = data["result"].get("confidence")

        if raw_choice is None:
            raise ValueError(f"Malformed Jev response: no choice found in payload: {data}")

        choice = str(raw_choice).strip()
        if choice not in task.valid_choices_set:
            raise ValueError(
                f"Jev returned choice '{choice}' which is not in task valid choices: {task.valid_choices_set}"
            )

        # Token and cost accounting
        usage = data.get("usage", {})
        input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
        output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
        cost = (
            float(usage.get("cost", usage.get("total_cost", 0.0)))
            if ("cost" in usage or "total_cost" in usage)
            else None
        )

        return DecisionResponse(
            response_id=generate_id("resp_jev"),
            observation_id=observation.observation_id,
            task_id=task.task_id,
            provider=self.provider_name,
            requested_model=self.requested_model_pin,
            returned_model=returned_model,
            choice=choice,
            confidence=float(confidence) if confidence is not None else None,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            raw_response_hash=raw_hash,
            request_hash=request_hash,
            timestamp_utc=datetime.now(timezone.utc),
            raw_response=data,
        )


class RandomBaselineProvider(BaseDecisionProvider):
    """Deterministic random baseline with fixed seed for statistical control."""

    def __init__(self, seed: int = 42, abstain_probability: float = 0.1) -> None:
        self.seed = seed
        self.abstain_probability = abstain_probability
        self._rng = random.Random(seed)

    @property
    def provider_name(self) -> str:
        return "baseline_random"

    @property
    def model_name(self) -> str:
        return f"uniform_random_seed_{self.seed}"

    def evaluate(
        self,
        task: DecisionTask,
        observation: DecisionObservation,
    ) -> DecisionResponse:
        t0 = time.monotonic()
        if task.optional_abstain and self._rng.random() < self.abstain_probability:
            choice = ABSTAIN_CHOICE
        else:
            choice = self._rng.choice(list(task.allowed_choices))

        latency_ms = max(1, int((time.monotonic() - t0) * 1000))
        return DecisionResponse(
            response_id=generate_id("resp_rand"),
            observation_id=observation.observation_id,
            task_id=task.task_id,
            provider=self.provider_name,
            requested_model=self.model_name,
            returned_model=self.model_name,
            choice=choice,
            confidence=round(1.0 / len(task.allowed_choices), 4),
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            raw_response_hash="baseline_deterministic_hash",
            request_hash=observation.state_hash,
            timestamp_utc=datetime.now(timezone.utc),
        )


class MajorityBaseRateProvider(BaseDecisionProvider):
    """Majority class / base-rate baseline that always predicts the most frequent choice."""

    def __init__(self, dominant_choice: str) -> None:
        self.dominant_choice = dominant_choice

    @property
    def provider_name(self) -> str:
        return "baseline_majority"

    @property
    def model_name(self) -> str:
        return f"majority_{self.dominant_choice}"

    def evaluate(
        self,
        task: DecisionTask,
        observation: DecisionObservation,
    ) -> DecisionResponse:
        t0 = time.monotonic()
        choice = (
            self.dominant_choice
            if self.dominant_choice in task.allowed_choices
            else task.allowed_choices[0]
        )
        latency_ms = max(1, int((time.monotonic() - t0) * 1000))
        return DecisionResponse(
            response_id=generate_id("resp_maj"),
            observation_id=observation.observation_id,
            task_id=task.task_id,
            provider=self.provider_name,
            requested_model=self.model_name,
            returned_model=self.model_name,
            choice=choice,
            confidence=1.0,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            raw_response_hash="baseline_majority_hash",
            request_hash=observation.state_hash,
            timestamp_utc=datetime.now(timezone.utc),
        )


class AlwaysAbstainProvider(BaseDecisionProvider):
    """Control provider that always abstains, establishing baseline abstention metrics."""

    @property
    def provider_name(self) -> str:
        return "baseline_always_abstain"

    @property
    def model_name(self) -> str:
        return "always_abstain"

    def evaluate(
        self,
        task: DecisionTask,
        observation: DecisionObservation,
    ) -> DecisionResponse:
        t0 = time.monotonic()
        if not task.optional_abstain:
            raise ValueError(f"Task {task.task_id} does not permit abstention")

        latency_ms = max(1, int((time.monotonic() - t0) * 1000))
        return DecisionResponse(
            response_id=generate_id("resp_abs"),
            observation_id=observation.observation_id,
            task_id=task.task_id,
            provider=self.provider_name,
            requested_model=self.model_name,
            returned_model=self.model_name,
            choice=ABSTAIN_CHOICE,
            confidence=None,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            raw_response_hash="baseline_abstain_hash",
            request_hash=observation.state_hash,
            timestamp_utc=datetime.now(timezone.utc),
        )


class DeterministicRuleProvider(BaseDecisionProvider):
    """Transparent heuristic rule baseline executing a pure function over state."""

    def __init__(
        self,
        rule_name: str,
        rule_fn: Callable[[dict[str, Any]], str],
    ) -> None:
        self.rule_name = rule_name
        self.rule_fn = rule_fn

    @property
    def provider_name(self) -> str:
        return "baseline_rule"

    @property
    def model_name(self) -> str:
        return self.rule_name

    def evaluate(
        self,
        task: DecisionTask,
        observation: DecisionObservation,
    ) -> DecisionResponse:
        t0 = time.monotonic()
        choice = self.rule_fn(observation.state_payload)
        if choice not in task.valid_choices_set:
            raise ValueError(f"Rule returned choice '{choice}' not valid for task {task.task_id}")

        latency_ms = max(1, int((time.monotonic() - t0) * 1000))
        return DecisionResponse(
            response_id=generate_id("resp_rule"),
            observation_id=observation.observation_id,
            task_id=task.task_id,
            provider=self.provider_name,
            requested_model=self.model_name,
            returned_model=self.model_name,
            choice=choice,
            confidence=0.8,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            raw_response_hash="baseline_rule_hash",
            request_hash=observation.state_hash,
            timestamp_utc=datetime.now(timezone.utc),
        )


class SecurityError(Exception):
    """Raised when request target violates safety constraints."""
    pass
