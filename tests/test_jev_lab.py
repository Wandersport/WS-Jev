"""Unit and integration tests for Phase 9A Jev Decision Research Lab.

Verifies:
1. Typed-choice schema and contract validation.
2. First-class abstention support in tasks, providers, and evaluators.
3. Invalid choice rejection and fail-closed semantics.
4. Deterministic request hashing and response fingerprinting.
5. Pinned model identifier and returned model persistence.
6. Token and cost accounting.
7. Deterministic baselines (Random with fixed seed, Majority, AlwaysAbstain, Rule-based).
8. Strategy candidate contract and deterministic hashing.
9. Multiple-testing governance and hypothesis-attempt accounting (Bonferroni / Holm).
10. Lightweight SQLite storage and audit ledger.
11. Structural safety invariants: zero live execution, zero order routing, zero wallets.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from pm_research.research.jev_lab.contract import (
    ABSTAIN_CHOICE,
    STATUS_DEV_HYPOTHESIS,
    DecisionObservation,
    DecisionResponse,
    DecisionScore,
    DecisionTask,
)
from pm_research.research.jev_lab.evaluator import (
    evaluate_cohort,
    score_single_decision,
)
from pm_research.research.jev_lab.governance import (
    MultipleTestingLedger,
)
from pm_research.research.jev_lab.providers import (
    PINNED_JEV_MODEL,
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
from pm_research.storage.db import Database

# ---------------------------------------------------------------------------
# 1. Typed-Choice Schema & Contract Tests
# ---------------------------------------------------------------------------


def test_decision_task_contract_valid_choices():
    """Verify task contract validates allowed choices and includes ABSTAIN if requested."""
    task = DecisionTask(
        task_id="test_task_1",
        task_version="1.0",
        task_type="BINARY_GATE",
        question="Is setup valid?",
        allowed_choices=("VALID", "INVALID"),
        optional_abstain=True,
        state_schema_version="test-schema-v1",
        label_definition="Objective definition",
        information_cutoff="t_minus_0",
        target_horizon="5m",
        description="A test task",
    )
    assert "VALID" in task.valid_choices_set
    assert "INVALID" in task.valid_choices_set
    assert ABSTAIN_CHOICE in task.valid_choices_set
    assert task.status == STATUS_DEV_HYPOTHESIS

    # Duplicate or empty choices rejected
    with pytest.raises(ValueError, match="allowed_choices must contain at least one discrete option"):
        DecisionTask(
            task_id="invalid",
            task_version="1.0",
            task_type="EMPTY",
            question="",
            allowed_choices=(),
        )

    with pytest.raises(ValueError, match="task_id and task_version must be non-empty"):
        DecisionTask(
            task_id="",
            task_version="1.0",
            task_type="EMPTY",
            question="",
            allowed_choices=("A",),
        )


def test_decision_observation_contract():
    """Verify DecisionObservation hashing and immutability."""
    now = datetime.now(timezone.utc)
    obs = DecisionObservation(
        observation_id="obs_001",
        task_id="test_task_1",
        market_round_id="round_123",
        as_of_ts_utc=now,
        state_payload={"feature_a": 1.5, "feature_b": 0.2},
        provenance="TEST_FIXTURE",
    )
    assert obs.state_hash is not None
    assert len(obs.state_hash) == 64

    # Identical state payload produces identical state_hash
    obs2 = DecisionObservation(
        observation_id="obs_002",
        task_id="test_task_1",
        market_round_id="round_124",
        as_of_ts_utc=now,
        state_payload={"feature_b": 0.2, "feature_a": 1.5},  # reversed key order
        provenance="TEST_FIXTURE",
    )
    assert obs.state_hash == obs2.state_hash


def test_canonical_tasks_registry():
    """Verify canonical tasks are well-defined and flagged as development hypotheses."""
    assert len(CANONICAL_TASKS) >= 4
    for task in [
        TASK_SETUP_GATING,
        TASK_MARKET_REGIME,
        TASK_SIGNAL_AGREEMENT,
        TASK_FLOW_TOXICITY,
    ]:
        assert task.status == STATUS_DEV_HYPOTHESIS
        assert task.optional_abstain is True
        assert ABSTAIN_CHOICE in task.valid_choices_set
        assert task.label_definition != ""

    reg = TaskRegistry()
    assert reg.get_task(TASK_SETUP_GATING.task_id) == TASK_SETUP_GATING
    with pytest.raises(KeyError):
        reg.get_task("nonexistent_task")


# ---------------------------------------------------------------------------
# 2. Abstention Support & Evaluator Metrics
# ---------------------------------------------------------------------------


def test_evaluator_abstention_and_metrics():
    """Verify scoring handles coverage, abstention, and conditional vs effective accuracy."""
    now = datetime.now(timezone.utc)
    task = TASK_SETUP_GATING

    # 4 observations, 1 true label each
    labels = ["GOOD_SETUP", "GOOD_SETUP", "BAD_SETUP", "BAD_SETUP"]

    # Model predictions:
    # 0: GOOD_SETUP (correct)
    # 1: BAD_SETUP  (incorrect)
    # 2: ABSTAIN    (abstained)
    # 3: BAD_SETUP  (correct)
    responses = [
        DecisionResponse(
            response_id="r0",
            observation_id="obs_0",
            task_id=task.task_id,
            provider="test",
            requested_model="m",
            returned_model="m",
            choice="GOOD_SETUP",
            confidence=0.8,
            latency_ms=10,
            input_tokens=100,
            output_tokens=5,
            cost=0.0001,
            raw_response_hash="h0",
            request_hash="req0",
            timestamp_utc=now,
        ),
        DecisionResponse(
            response_id="r1",
            observation_id="obs_1",
            task_id=task.task_id,
            provider="test",
            requested_model="m",
            returned_model="m",
            choice="BAD_SETUP",
            confidence=0.7,
            latency_ms=10,
            input_tokens=100,
            output_tokens=5,
            cost=0.0001,
            raw_response_hash="h1",
            request_hash="req1",
            timestamp_utc=now,
        ),
        DecisionResponse(
            response_id="r2",
            observation_id="obs_2",
            task_id=task.task_id,
            provider="test",
            requested_model="m",
            returned_model="m",
            choice=ABSTAIN_CHOICE,
            confidence=None,
            latency_ms=10,
            input_tokens=100,
            output_tokens=5,
            cost=0.0001,
            raw_response_hash="h2",
            request_hash="req2",
            timestamp_utc=now,
        ),
        DecisionResponse(
            response_id="r3",
            observation_id="obs_3",
            task_id=task.task_id,
            provider="test",
            requested_model="m",
            returned_model="m",
            choice="BAD_SETUP",
            confidence=0.9,
            latency_ms=10,
            input_tokens=100,
            output_tokens=5,
            cost=0.0001,
            raw_response_hash="h3",
            request_hash="req3",
            timestamp_utc=now,
        ),
    ]

    responses_with_labels = list(zip(responses, labels, strict=True))
    cohort = evaluate_cohort(task, responses_with_labels)

    assert cohort.total_observations == 4
    assert cohort.total_abstained == 1
    assert cohort.total_acted == 3
    assert cohort.coverage_rate == 0.75
    assert cohort.abstention_rate == 0.25
    # Out of 3 active decisions: 2 correct (obs 0 and 3), 1 wrong (obs 1) -> 2/3 ≈ 0.6667
    assert pytest.approx(cohort.conditional_accuracy, rel=1e-3) == 2 / 3
    # Effective accuracy across all 4: 2 / 4 = 0.50
    assert pytest.approx(cohort.effective_accuracy, rel=1e-3) == 0.50


def test_score_single_decision_invalid_choice():
    """Verify scoring rejects response with choice not in task valid set."""
    now = datetime.now(timezone.utc)
    task = TASK_SETUP_GATING
    resp = DecisionResponse(
        response_id="r_inv",
        observation_id="obs_0",
        task_id=task.task_id,
        provider="test",
        requested_model="m",
        returned_model="m",
        choice="BUY_CALLS",  # Illegal choice
        confidence=0.9,
        latency_ms=10,
        input_tokens=10,
        output_tokens=2,
        cost=None,
        raw_response_hash="h",
        request_hash="req",
        timestamp_utc=now,
    )
    with pytest.raises(ValueError, match="is invalid for task"):
        score_single_decision(task, resp, "GOOD_SETUP")


# ---------------------------------------------------------------------------
# 3. Deterministic Baselines Tests
# ---------------------------------------------------------------------------


def test_random_baseline_provider_reproducibility():
    """Verify RandomBaselineProvider is 100% reproducible with fixed seed."""
    task = TASK_SETUP_GATING
    now = datetime.now(timezone.utc)
    obs = DecisionObservation(
        observation_id="obs_test",
        task_id=task.task_id,
        market_round_id="rnd_test",
        as_of_ts_utc=now,
        state_payload={"x": 1},
        provenance="TEST",
    )

    prov1 = RandomBaselineProvider(seed=12345, abstain_probability=0.2)
    prov2 = RandomBaselineProvider(seed=12345, abstain_probability=0.2)

    choices1 = [prov1.evaluate(task, obs).choice for _ in range(20)]
    choices2 = [prov2.evaluate(task, obs).choice for _ in range(20)]

    assert choices1 == choices2
    assert ABSTAIN_CHOICE in choices1
    assert any(c in task.allowed_choices for c in choices1)


def test_majority_baseline_provider():
    """Verify MajorityBaseRateProvider returns dominant choice."""
    task = TASK_SETUP_GATING
    now = datetime.now(timezone.utc)
    obs = DecisionObservation(
        observation_id="obs_maj",
        task_id=task.task_id,
        market_round_id="rnd_maj",
        as_of_ts_utc=now,
        state_payload={},
        provenance="TEST",
    )
    prov = MajorityBaseRateProvider(dominant_choice="GOOD_SETUP")
    resp = prov.evaluate(task, obs)
    assert resp.choice == "GOOD_SETUP"
    assert resp.provider == "baseline_majority"


def test_always_abstain_provider():
    """Verify AlwaysAbstainProvider reliably yields ABSTAIN."""
    task = TASK_SETUP_GATING
    now = datetime.now(timezone.utc)
    obs = DecisionObservation(
        observation_id="obs_abs",
        task_id=task.task_id,
        market_round_id="rnd_abs",
        as_of_ts_utc=now,
        state_payload={},
        provenance="TEST",
    )
    prov = AlwaysAbstainProvider()
    resp = prov.evaluate(task, obs)
    assert resp.choice == ABSTAIN_CHOICE


def test_deterministic_rule_provider():
    """Verify DeterministicRuleProvider executes custom deterministic rules."""
    task = TASK_SETUP_GATING
    now = datetime.now(timezone.utc)

    def gating_rule(state: dict[str, Any]) -> str:
        spread = state.get("spread_bps", 10.0)
        return "GOOD_SETUP" if spread < 2.0 else "BAD_SETUP"

    prov = DeterministicRuleProvider("tight_spread_rule", gating_rule)

    obs_tight = DecisionObservation(
        observation_id="obs_t",
        task_id=task.task_id,
        market_round_id="r1",
        as_of_ts_utc=now,
        state_payload={"spread_bps": 1.2},
        provenance="TEST",
    )
    obs_wide = DecisionObservation(
        observation_id="obs_w",
        task_id=task.task_id,
        market_round_id="r2",
        as_of_ts_utc=now,
        state_payload={"spread_bps": 4.5},
        provenance="TEST",
    )

    assert prov.evaluate(task, obs_tight).choice == "GOOD_SETUP"
    assert prov.evaluate(task, obs_wide).choice == "BAD_SETUP"


# ---------------------------------------------------------------------------
# 4. Strategy Candidate Contract & Deterministic Hashing
# ---------------------------------------------------------------------------


def test_strategy_candidate_contract_and_hashing():
    """Verify StrategyCandidate hashing determinism and parameter budget validation."""
    now = datetime.now(timezone.utc)
    strat = StrategyCandidate(
        strategy_id="strat_regime_vol_filter_v1",
        strategy_version="1.0.0",
        source_reference="The-Quant-Trading-Vault/volatility/filter_01",
        hypothesis="Gating trades on 60s realized volatility prevents toxic execution",
        required_features=("spread_bps", "realized_volatility_60s"),
        market="BTC-5M-PM",
        horizon="5m",
        parameters={"max_spread_bps": 2.5, "max_vol": 0.003},
        parameter_search_budget=10,
        cost_model={"model": "zero_commission_fixed_half_spread"},
        validation_protocol="temporal_split_500_rounds",
        rejection_conditions=(
            "conditional_accuracy <= 0.50",
            "coverage_rate < 0.20",
        ),
        frozen_at_utc=now,
    )

    h1 = compute_strategy_hash(
        strategy_id=strat.strategy_id,
        strategy_version=strat.strategy_version,
        source_reference=strat.source_reference,
        hypothesis=strat.hypothesis,
        required_features=strat.required_features,
        market=strat.market,
        horizon=strat.horizon,
        parameters=strat.parameters,
        parameter_search_budget=strat.parameter_search_budget,
        cost_model=strat.cost_model,
        validation_protocol=strat.validation_protocol,
        rejection_conditions=strat.rejection_conditions,
    )
    assert strat.strategy_hash == h1
    assert len(strat.strategy_hash) == 64

    # Different parameters produce different hash
    strat_diff = StrategyCandidate(
        strategy_id="strat_regime_vol_filter_v1",
        strategy_version="1.0.0",
        source_reference="The-Quant-Trading-Vault/volatility/filter_01",
        hypothesis="Gating trades on 60s realized volatility prevents toxic execution",
        required_features=("spread_bps", "realized_volatility_60s"),
        market="BTC-5M-PM",
        horizon="5m",
        parameters={"max_spread_bps": 3.0, "max_vol": 0.003},  # changed
        parameter_search_budget=10,
        cost_model={"model": "zero_commission_fixed_half_spread"},
        validation_protocol="temporal_split_500_rounds",
        rejection_conditions=(
            "conditional_accuracy <= 0.50",
            "coverage_rate < 0.20",
        ),
        frozen_at_utc=now,
    )
    assert strat_diff.strategy_hash != h1

    # Budget violation rejection
    with pytest.raises(ValueError, match="parameter_search_budget must be strictly positive"):
        StrategyCandidate(
            strategy_id="bad",
            strategy_version="1",
            source_reference="ref",
            hypothesis="hyp",
            required_features=(),
            market="M",
            horizon="H",
            parameters={},
            parameter_search_budget=0,
            cost_model={},
            validation_protocol="split",
            rejection_conditions=("cond",),
            frozen_at_utc=now,
        )


# ---------------------------------------------------------------------------
# 5. Multiple-Testing Governance & Accounting
# ---------------------------------------------------------------------------


def test_multiple_testing_ledger():
    """Verify MultipleTestingLedger computes Bonferroni and Holm corrections."""
    ledger = MultipleTestingLedger()

    # Record 4 hypothesis attempts
    for i in range(4):
        ledger.record_attempt(
            strategy_id="strat_1",
            strategy_hash="f" * 64,
            variant_id=f"var_{i}",
            parameter_values={"threshold": i * 0.5},
            status="REJECTED" if i < 3 else "SURVIVED",
            rejection_reason="low_accuracy" if i < 3 else None,
            metrics={"p_value": 0.01 * (i + 1)},
        )

    assert ledger.total_variants_attempted == 4
    assert ledger.total_hypotheses_attempted == 1
    assert ledger.total_rejected == 3
    assert ledger.total_surviving == 1

    # Bonferroni threshold for alpha=0.05 across 4 tests: 0.05 / 4 = 0.0125
    assert pytest.approx(ledger.bonferroni_alpha(0.05), rel=1e-4) == 0.0125

    # Holm threshold for rank 1 (1-indexed) out of 4: 0.05 / (4 - 1 + 1) = 0.0125
    assert pytest.approx(ledger.holm_bonferroni_threshold(rank=1, nominal_alpha=0.05), rel=1e-4) == 0.0125
    # rank 2: 0.05 / (4 - 2 + 1) = 0.05 / 3 ≈ 0.01667
    assert pytest.approx(ledger.holm_bonferroni_threshold(rank=2, nominal_alpha=0.05), rel=1e-4) == 0.05 / 3
    # rank 4: 0.05 / 1 = 0.05
    assert pytest.approx(ledger.holm_bonferroni_threshold(rank=4, nominal_alpha=0.05), rel=1e-4) == 0.05


# ---------------------------------------------------------------------------
# 6. Storage & Database Operations
# ---------------------------------------------------------------------------


def test_jev_lab_storage_roundtrip():
    """Verify tasks, observations, responses, and scores persist and retrieve cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_research.db"
        db = Database(str(db_path))
        storage = JevLabStorage(db)

        # 1. Task persistence
        storage.save_task(TASK_SETUP_GATING)

        # 2. Observation persistence
        now = datetime.now(timezone.utc)
        obs = DecisionObservation(
            observation_id="obs_roundtrip_1",
            task_id=TASK_SETUP_GATING.task_id,
            market_round_id="round_1",
            as_of_ts_utc=now,
            state_payload={"spread": 1.2, "imbalance": 0.4},
            provenance="FIXTURE",
        )
        storage.save_observation(obs)

        # 3. Response persistence
        resp = DecisionResponse(
            response_id="resp_roundtrip_1",
            observation_id=obs.observation_id,
            task_id=TASK_SETUP_GATING.task_id,
            provider="test_prov",
            requested_model=PINNED_JEV_MODEL,
            returned_model="typesafe/jev-1.13-2026",
            choice="GOOD_SETUP",
            confidence=0.88,
            latency_ms=120,
            input_tokens=250,
            output_tokens=15,
            cost=0.00025,
            raw_response_hash="hash_raw",
            request_hash="hash_req",
            timestamp_utc=now,
            raw_response={"test": "payload"},
        )
        storage.save_response(resp)

        # 4. Score persistence
        score = DecisionScore(
            score_id="score_roundtrip_1",
            response_id=resp.response_id,
            task_id=TASK_SETUP_GATING.task_id,
            objective_label="GOOD_SETUP",
            is_correct=True,
            is_abstained=False,
            brier_score=0.0144,
            log_loss=0.128,
            scored_at_utc=now,
        )
        storage.save_score(score)

        # 5. Audit summary
        summary = storage.get_audit_summary()
        assert summary["registered_tasks"] == 1
        assert summary["observations_recorded"] == 1
        assert summary["responses_recorded"] == 1
        assert summary["scores_recorded"] == 1


# ---------------------------------------------------------------------------
# 7. Structural Safety Invariants
# ---------------------------------------------------------------------------


def test_safety_invariants_no_execution_semantics():
    """Verify DecisionTask, Response, and Providers have zero live execution capability."""
    for cls in [
        DecisionTask,
        DecisionObservation,
        DecisionResponse,
        DecisionScore,
        BaseDecisionProvider,
        JevDecisionProvider,
        RandomBaselineProvider,
        MajorityBaseRateProvider,
        AlwaysAbstainProvider,
        DeterministicRuleProvider,
    ]:
        forbidden_attributes = [
            "buy",
            "sell",
            "place_order",
            "execute_order",
            "submit_order",
            "wallet",
            "private_key",
            "api_secret",
            "keystore",
            "sign_transaction",
            "broadcast",
        ]
        for attr in forbidden_attributes:
            assert not hasattr(cls, attr), f"Forbidden attribute '{attr}' detected on {cls.__name__}"
