"""Execution runner and API budget controller for Phase 9B.

Enforces:
1. Hard budget ceiling: <= 500 requests for primary task.
2. Hard cost ceiling: Stops immediately if cumulative cost exceeds $1.00.
3. Idempotent execution: Existing responses in JevLabStorage are replayed without redundant API calls.
4. Pre-evaluation registration in MultipleTestingLedger.
5. Zero execution semantics (no order placement, no wallet interaction).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pm_research.research.jev_lab import (
    JevDecisionProvider,
    JevLabStorage,
    MultipleTestingLedger,
)
from pm_research.research.jev_lab.contract import (
    ABSTAIN_CHOICE,
    DecisionResponse,
)
from pm_research.research.jev_openrouter import has_openrouter_api_key
from pm_research.research.jev_phase9b.baselines import (
    AlwaysAbstainBaseline,
    DeterministicRuleBaseline,
    MajorityBaseline,
    RandomBaseline,
    StandardizedLogisticRegression,
)
from pm_research.research.jev_phase9b.dataset import Phase9bDataset
from pm_research.research.jev_phase9b.spec import (
    TASK_ID,
    compute_task_spec_hash,
    get_phase9b_decision_task,
)
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

MAX_PRIMARY_REQUESTS: int = 500
MAX_COST_CEILING_USD: float = 1.00


class Phase9bRunner:
    """Orchestrates Jev evaluation and baseline comparisons for Phase 9B."""

    def __init__(
        self,
        db_path: str = "data/pm_research.db",
        timeout_seconds: float = 25.0,
        max_retries: int = 1,
    ) -> None:
        self.db = Database(db_path)
        self.storage = JevLabStorage(self.db)
        self.ledger = MultipleTestingLedger()
        self.task = get_phase9b_decision_task()
        self.spec_hash = compute_task_spec_hash()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Register task in storage
        self.storage.save_task(self.task)

    def run_benchmark(
        self,
        dataset: Phase9bDataset,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute full benchmark across Jev and all baselines on the eligible dataset."""
        # 1. Multiple-Testing Pre-Registration
        attempt_record = self.ledger.record_attempt(
            strategy_id="jev_setup_gating_phase9b",
            strategy_hash=self.spec_hash,
            variant_id="primary_30s_120s_anchor",
            parameter_values={
                "horizon": "30s",
                "anchor": 120,
                "n_eligible": dataset.eligible_count,
            },
            status="REGISTERED_PRE_EVALUATION",
            metrics={"task_id": TASK_ID, "spec_hash": self.spec_hash},
        )
        self.storage.save_hypothesis_attempt(attempt_record)

        # 2. Fit learned baselines strictly on training split
        train_samples = dataset.train_samples
        logit_baseline = StandardizedLogisticRegression(
            l2_penalty=0.01, learning_rate=0.05, max_epochs=300
        )
        logit_baseline.fit(train_samples)

        ridge_baseline = StandardizedLogisticRegression(
            l2_penalty=0.50, learning_rate=0.05, max_epochs=300,
            model_name="baseline_ridge_logistic_l2_0.5"
        )
        ridge_baseline.fit(train_samples)

        maj_baseline = MajorityBaseline()
        rule_baseline = DeterministicRuleBaseline()
        rand_baseline = RandomBaseline(seed=42)
        abs_baseline = AlwaysAbstainBaseline()

        # 3. Evaluate Baselines across all eligible samples
        baseline_preds: dict[str, list[tuple[str, str, float | None, float | None]]] = {
            maj_baseline.name: [],
            rule_baseline.name: [],
            logit_baseline.name: [],
            ridge_baseline.name: [],
            rand_baseline.name: [],
            abs_baseline.name: [],
        }

        for sample in dataset.samples:
            true_label = sample.objective_label

            p_maj = maj_baseline.predict(sample)
            baseline_preds[maj_baseline.name].append(
                (p_maj.choice, true_label, p_maj.probability_meaningful, None)
            )

            p_rule = rule_baseline.predict(sample)
            baseline_preds[rule_baseline.name].append(
                (p_rule.choice, true_label, p_rule.probability_meaningful, None)
            )

            p_logit = logit_baseline.predict(sample)
            baseline_preds[logit_baseline.name].append(
                (p_logit.choice, true_label, p_logit.probability_meaningful, None)
            )

            p_ridge = ridge_baseline.predict(sample)
            baseline_preds[ridge_baseline.name].append(
                (p_ridge.choice, true_label, p_ridge.probability_meaningful, None)
            )

            p_rand = rand_baseline.predict(sample)
            baseline_preds[rand_baseline.name].append(
                (p_rand.choice, true_label, p_rand.probability_meaningful, None)
            )

            p_abs = abs_baseline.predict(sample)
            baseline_preds[abs_baseline.name].append(
                (p_abs.choice, true_label, p_abs.probability_meaningful, None)
            )

        # 4. Evaluate Jev Model (with strict cost and count limits)
        jev_predictions: list[tuple[str, str, float | None, float | None]] = []
        jev_latencies: list[int] = []
        jev_raw_responses: list[DecisionResponse] = []
        cumulative_cost = 0.0
        n_api_calls = 0
        n_cached_replays = 0
        returned_model_id = "NONE"

        api_available = has_openrouter_api_key()
        jev_provider = (
            JevDecisionProvider(
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
            )
            if (api_available and not dry_run)
            else None
        )

        for sample in dataset.samples:
            obs = sample.to_decision_observation()
            self.storage.save_observation(obs)

            if dry_run or not api_available:
                # Simulated placeholder if API key missing or dry run
                choice = ABSTAIN_CHOICE
                conf = None
                jev_predictions.append((choice, sample.objective_label, conf, None))
                continue

            # Check if response already exists in storage for idempotency
            with self.db._get_connection() as conn:
                existing = conn.execute(
                    """
                    SELECT choice, confidence, latency_ms, cost, returned_model
                    FROM jev_lab_responses
                    WHERE observation_id = ? AND task_id = ?
                    LIMIT 1
                    """,
                    (obs.observation_id, self.task.task_id),
                ).fetchone()

            if existing:
                n_cached_replays += 1
                c_choice, c_conf, c_lat, c_cost, c_mod = existing
                returned_model_id = c_mod
                jev_latencies.append(c_lat)
                if c_cost:
                    cumulative_cost += c_cost
                jev_predictions.append(
                    (c_choice, sample.objective_label, c_conf, c_cost)
                )
                continue

            # Check budget limits before calling remote API
            if n_api_calls >= MAX_PRIMARY_REQUESTS:
                logger.warning(f"Exceeded max primary requests ({MAX_PRIMARY_REQUESTS}). Stopping.")
                break
            if cumulative_cost >= MAX_COST_CEILING_USD:
                logger.warning(f"Exceeded cost ceiling (${MAX_COST_CEILING_USD:.2f}). Stopping.")
                break

            # Execute Jev Inference
            try:
                assert jev_provider is not None
                resp = jev_provider.evaluate(self.task, obs)
                self.storage.save_response(resp)
                jev_raw_responses.append(resp)

                n_api_calls += 1
                jev_latencies.append(resp.latency_ms)
                returned_model_id = resp.returned_model
                if resp.cost:
                    cumulative_cost += resp.cost

                jev_predictions.append(
                    (resp.choice, sample.objective_label, resp.confidence, resp.cost)
                )

                # Light pacing to respect rate limits
                time.sleep(0.05)

                if n_api_calls % 25 == 0 or n_api_calls == len(dataset.samples):
                    print(
                        f"  [Jev Eval] Queried {n_api_calls}/{len(dataset.samples)} samples | "
                        f"Cumulative Cost: ${cumulative_cost:.4f} | Latency: {resp.latency_ms}ms",
                        flush=True,
                    )

            except Exception as e:
                logger.error(f"Error querying Jev for {obs.observation_id}: {e}")
                # Fail closed on individual request error
                jev_predictions.append((ABSTAIN_CHOICE, sample.objective_label, None, None))

        return {
            "n_eligible": dataset.eligible_count,
            "n_api_calls": n_api_calls,
            "n_cached_replays": n_cached_replays,
            "cumulative_cost": round(cumulative_cost, 6),
            "returned_model_id": returned_model_id,
            "avg_latency_ms": (
                round(sum(jev_latencies) / len(jev_latencies), 2)
                if jev_latencies
                else 0.0
            ),
            "jev_latencies": jev_latencies,
            "jev_predictions": jev_predictions,
            "baseline_predictions": baseline_preds,
        }
