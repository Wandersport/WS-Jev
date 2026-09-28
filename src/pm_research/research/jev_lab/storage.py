"""Lightweight database persistence for Phase 9A Jev Decision Research Lab.

RESEARCH USE ONLY:
Creates isolated, lightweight relational schemas for decision tasks, observations,
responses, scores, strategy candidates, and hypothesis attempts.
Minimal footprint: strictly designed to consume <1 MB for development/smoke testing.
"""

from __future__ import annotations

import json
from typing import Any

from pm_research.research.jev_lab.contract import (
    DecisionObservation,
    DecisionResponse,
    DecisionScore,
    DecisionTask,
)
from pm_research.research.jev_lab.governance import HypothesisAttempt
from pm_research.research.jev_lab.strategy_contract import StrategyCandidate
from pm_research.storage.db import Database
from pm_research.utils import ensure_utc, to_iso_utc


class JevLabStorage:
    """Manages isolated lightweight storage for the Jev Decision Lab."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.init_schema()

    def init_schema(self) -> None:
        """Create isolated jev_lab tables if they do not exist."""
        sql = """
            CREATE TABLE IF NOT EXISTS jev_lab_tasks (
                task_id TEXT PRIMARY KEY,
                task_version TEXT NOT NULL,
                task_type TEXT NOT NULL,
                question TEXT NOT NULL,
                allowed_choices_json TEXT NOT NULL,
                optional_abstain INTEGER NOT NULL,
                state_schema_version TEXT NOT NULL,
                label_definition TEXT NOT NULL,
                information_cutoff TEXT NOT NULL,
                target_horizon TEXT NOT NULL,
                status TEXT NOT NULL,
                description TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jev_lab_observations (
                observation_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                market_round_id TEXT NOT NULL,
                as_of_ts TEXT NOT NULL,
                state_payload_json TEXT NOT NULL,
                provenance TEXT NOT NULL,
                state_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jev_lab_responses (
                response_id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                requested_model TEXT NOT NULL,
                returned_model TEXT NOT NULL,
                choice TEXT NOT NULL,
                confidence REAL,
                latency_ms INTEGER NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost REAL,
                raw_response_hash TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                raw_response_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jev_lab_scores (
                score_id TEXT PRIMARY KEY,
                response_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                objective_label TEXT,
                is_correct INTEGER,
                is_abstained INTEGER NOT NULL,
                brier_score REAL,
                log_loss REAL,
                scored_at_utc TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jev_lab_strategy_candidates (
                strategy_id TEXT PRIMARY KEY,
                strategy_version TEXT NOT NULL,
                source_reference TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                required_features_json TEXT NOT NULL,
                market TEXT NOT NULL,
                horizon TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                parameter_search_budget INTEGER NOT NULL,
                cost_model_json TEXT NOT NULL,
                validation_protocol TEXT NOT NULL,
                rejection_conditions_json TEXT NOT NULL,
                frozen_at_utc TEXT NOT NULL,
                strategy_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS jev_lab_hypothesis_attempts (
                attempt_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                strategy_hash TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                parameter_values_json TEXT NOT NULL,
                status TEXT NOT NULL,
                rejection_reason TEXT,
                timestamp_utc TEXT NOT NULL,
                metrics_json TEXT NOT NULL
            );
        """
        with self.db._get_connection() as conn:
            conn.executescript(sql)

    def save_task(self, task: DecisionTask) -> None:
        """Persist or update a DecisionTask record."""
        sql = """
            INSERT OR REPLACE INTO jev_lab_tasks (
                task_id, task_version, task_type, question,
                allowed_choices_json, optional_abstain, state_schema_version,
                label_definition, information_cutoff, target_horizon,
                status, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db._get_connection() as conn:
            conn.execute(
                sql,
                (
                    task.task_id,
                    task.task_version,
                    task.task_type,
                    task.question,
                    json.dumps(list(task.allowed_choices)),
                    1 if task.optional_abstain else 0,
                    task.state_schema_version,
                    task.label_definition,
                    task.information_cutoff,
                    task.target_horizon,
                    task.status,
                    task.description,
                ),
            )

    def save_observation(self, observation: DecisionObservation) -> None:
        """Persist a DecisionObservation record."""
        sql = """
            INSERT OR REPLACE INTO jev_lab_observations (
                observation_id, task_id, market_round_id,
                as_of_ts, state_payload_json, provenance, state_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.db._get_connection() as conn:
            conn.execute(
                sql,
                (
                    observation.observation_id,
                    observation.task_id,
                    observation.market_round_id,
                    to_iso_utc(ensure_utc(observation.as_of_ts_utc)),
                    json.dumps(observation.state_payload),
                    observation.provenance,
                    observation.state_hash,
                ),
            )

    def save_response(self, response: DecisionResponse) -> None:
        """Persist a DecisionResponse record."""
        sql = """
            INSERT OR REPLACE INTO jev_lab_responses (
                response_id, observation_id, task_id, provider,
                requested_model, returned_model, choice, confidence,
                latency_ms, input_tokens, output_tokens, cost,
                raw_response_hash, request_hash, timestamp_utc,
                raw_response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db._get_connection() as conn:
            conn.execute(
                sql,
                (
                    response.response_id,
                    response.observation_id,
                    response.task_id,
                    response.provider,
                    response.requested_model,
                    response.returned_model,
                    response.choice,
                    response.confidence,
                    response.latency_ms,
                    response.input_tokens,
                    response.output_tokens,
                    response.cost,
                    response.raw_response_hash,
                    response.request_hash,
                    to_iso_utc(ensure_utc(response.timestamp_utc)),
                    json.dumps(response.raw_response),
                ),
            )

    def save_score(self, score: DecisionScore) -> None:
        """Persist a DecisionScore record."""
        sql = """
            INSERT OR REPLACE INTO jev_lab_scores (
                score_id, response_id, task_id, objective_label,
                is_correct, is_abstained, brier_score, log_loss,
                scored_at_utc, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db._get_connection() as conn:
            conn.execute(
                sql,
                (
                    score.score_id,
                    score.response_id,
                    score.task_id,
                    score.objective_label,
                    1 if score.is_correct is True else (0 if score.is_correct is False else None),
                    1 if score.is_abstained else 0,
                    score.brier_score,
                    score.log_loss,
                    to_iso_utc(ensure_utc(score.scored_at_utc)),
                    json.dumps(score.metrics),
                ),
            )

    def save_strategy_candidate(self, candidate: StrategyCandidate) -> None:
        """Persist an immutable StrategyCandidate record."""
        sql = """
            INSERT OR REPLACE INTO jev_lab_strategy_candidates (
                strategy_id, strategy_version, source_reference, hypothesis,
                required_features_json, market, horizon, parameters_json,
                parameter_search_budget, cost_model_json, validation_protocol,
                rejection_conditions_json, frozen_at_utc, strategy_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db._get_connection() as conn:
            conn.execute(
                sql,
                (
                    candidate.strategy_id,
                    candidate.strategy_version,
                    candidate.source_reference,
                    candidate.hypothesis,
                    json.dumps(list(candidate.required_features)),
                    candidate.market,
                    candidate.horizon,
                    json.dumps(candidate.parameters),
                    candidate.parameter_search_budget,
                    json.dumps(candidate.cost_model),
                    candidate.validation_protocol,
                    json.dumps(list(candidate.rejection_conditions)),
                    to_iso_utc(ensure_utc(candidate.frozen_at_utc)),
                    candidate.strategy_hash,
                ),
            )

    def save_hypothesis_attempt(self, attempt: HypothesisAttempt) -> None:
        """Persist a HypothesisAttempt record."""
        sql = """
            INSERT OR REPLACE INTO jev_lab_hypothesis_attempts (
                attempt_id, strategy_id, strategy_hash, variant_id,
                parameter_values_json, status, rejection_reason,
                timestamp_utc, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db._get_connection() as conn:
            conn.execute(
                sql,
                (
                    attempt.attempt_id,
                    attempt.strategy_id,
                    attempt.strategy_hash,
                    attempt.variant_id,
                    json.dumps(attempt.parameter_values),
                    attempt.status,
                    attempt.rejection_reason,
                    to_iso_utc(ensure_utc(attempt.timestamp_utc)),
                    json.dumps(attempt.metrics),
                ),
            )

    def get_audit_summary(self) -> dict[str, Any]:
        """Return counts and statistics for all Jev Lab tables."""
        with self.db._get_connection() as conn:
            tasks_cnt = conn.execute("SELECT count(*) FROM jev_lab_tasks").fetchone()[0]
            obs_cnt = conn.execute("SELECT count(*) FROM jev_lab_observations").fetchone()[0]
            resp_cnt = conn.execute("SELECT count(*) FROM jev_lab_responses").fetchone()[0]
            score_cnt = conn.execute("SELECT count(*) FROM jev_lab_scores").fetchone()[0]
            strat_cnt = conn.execute("SELECT count(*) FROM jev_lab_strategy_candidates").fetchone()[0]
            hyp_cnt = conn.execute("SELECT count(*) FROM jev_lab_hypothesis_attempts").fetchone()[0]

            providers = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT provider FROM jev_lab_responses"
                ).fetchall()
            ]

        return {
            "registered_tasks": tasks_cnt,
            "observations_recorded": obs_cnt,
            "responses_recorded": resp_cnt,
            "scores_recorded": score_cnt,
            "strategy_candidates": strat_cnt,
            "hypothesis_attempts": hyp_cnt,
            "active_providers": providers,
        }
