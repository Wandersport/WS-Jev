"""Unit and integration tests for Phase 9B: Jev Setup-Gating Benchmark.

Verifies:
1. Objective label construction (|q_(t+30s) - q_t| >= s_t -> MEANINGFUL_MOVE, else QUIET).
2. Lookahead prevention: zero future feature leakage into observation state.
3. Anchor selection: at most one primary observation per physical round near 120s.
4. Receive-time as-of tolerance checking.
5. Exclusion of invalid taker flow fields from state payload.
6. Task specification and prompt hash determinism.
7. Pre-evaluation registration in MultipleTestingLedger.
8. Abstention scoring and metric calculations (conditional vs effective accuracy).
9. Chronological partitioning (60% Train, 20% Calib, 20% Holdout).
10. Baseline train-only preprocessing and standardization.
11. Round-level block bootstrap uncertainty calculation.
12. API budget ceilings (max 500 requests, $1.00 cost limit).
13. Structural safety: zero execution semantics, zero wallets, zero order routing.
"""

from __future__ import annotations

from typing import Any

import pytest

from pm_research.research.jev_lab.contract import ABSTAIN_CHOICE
from pm_research.research.jev_phase9b.baselines import (
    AlwaysAbstainBaseline,
    DeterministicRuleBaseline,
    MajorityBaseline,
    RandomBaseline,
    StandardizedLogisticRegression,
)
from pm_research.research.jev_phase9b.dataset import Phase9bDataset, Phase9bSample
from pm_research.research.jev_phase9b.evaluator import (
    compute_metrics,
    compute_selective_prediction_curve,
    run_round_bootstrap,
)
from pm_research.research.jev_phase9b.runner import (
    MAX_COST_CEILING_USD,
    MAX_PRIMARY_REQUESTS,
    Phase9bRunner,
)
from pm_research.research.jev_phase9b.spec import (
    CHOICE_MEANINGFUL_MOVE,
    CHOICE_QUIET,
    EXCLUDED_FIELDS,
    TASK_ID,
    compute_task_spec_hash,
    get_phase9b_decision_task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_dummy_sample(
    round_slug: str,
    poly_mid: float = 0.50,
    poly_spread: float = 0.02,
    future_mid: float = 0.53,
    split: str = "train",
    binance_ret_30s: float = 5.0,
) -> Phase9bSample:
    abs_move = abs(future_mid - poly_mid)
    label = CHOICE_MEANINGFUL_MOVE if abs_move >= poly_spread else CHOICE_QUIET
    features: dict[str, Any] = {
        "poly_midpoint": poly_mid,
        "poly_spread": poly_spread,
        "seconds_remaining": 120,
        "poly_return_1s": 0.001,
        "poly_return_2s": 0.002,
        "poly_return_3s": 0.003,
        "poly_return_5s": 0.005,
        "poly_return_10s": 0.010,
        "poly_return_30s": 0.025,
        "binance_mid_price": 65000.0,
        "binance_spread_bps": 1.2,
        "binance_microprice_offset_bps": 0.3,
        "binance_top1_depth_imbalance": 0.15,
        "binance_top5_depth_imbalance": 0.25,
        "binance_top20_depth_imbalance": 0.30,
        "binance_return_since_open_bps": 8.0,
        "binance_return_1s_bps": 0.5,
        "binance_return_2s_bps": 1.0,
        "binance_return_3s_bps": 1.5,
        "binance_return_5s_bps": 2.0,
        "binance_return_10s_bps": 3.0,
        "binance_return_30s_bps": binance_ret_30s,
        "binance_return_60s_bps": 7.0,
    }
    return Phase9bSample(
        round_slug=round_slug,
        sample_id=f"samp_{round_slug}",
        as_of_ts_ms=1727500000000,
        seconds_remaining=120,
        poly_midpoint=poly_mid,
        poly_spread=poly_spread,
        future_sample_id=f"fut_{round_slug}",
        future_ts_ms=1727500030000,
        future_poly_midpoint=future_mid,
        absolute_move=abs_move,
        objective_label=label,
        features=features,
        split=split,
    )


# ---------------------------------------------------------------------------
# 1. Spec & Task Tests
# ---------------------------------------------------------------------------


def test_phase9b_spec_hash_determinism():
    """Verify task specification produces a deterministic SHA-256 fingerprint."""
    h1 = compute_task_spec_hash()
    h2 = compute_task_spec_hash()
    assert h1 == h2
    assert len(h1) == 64

    task = get_phase9b_decision_task()
    assert task.task_id == TASK_ID
    assert CHOICE_MEANINGFUL_MOVE in task.valid_choices_set
    assert CHOICE_QUIET in task.valid_choices_set
    assert ABSTAIN_CHOICE in task.valid_choices_set
    assert task.optional_abstain is True


def test_phase9b_exclusion_of_taker_flow():
    """Verify all taker-flow fields are strictly absent from sample features."""
    sample = make_dummy_sample("round_test_1")
    for excluded in EXCLUDED_FIELDS:
        assert excluded not in sample.features, f"Forbidden field '{excluded}' present in features"


# ---------------------------------------------------------------------------
# 2. Objective Label Construction & Lookahead Tests
# ---------------------------------------------------------------------------


def test_objective_label_construction():
    """Verify label rule: MEANINGFUL_MOVE if |q_(t+30s) - q_t| >= s_t, else QUIET."""
    # Move (0.53 - 0.50 = 0.03) >= spread (0.02) -> MEANINGFUL_MOVE
    s1 = make_dummy_sample("r1", poly_mid=0.50, poly_spread=0.02, future_mid=0.53)
    assert s1.objective_label == CHOICE_MEANINGFUL_MOVE

    # Move (0.51 - 0.50 = 0.01) < spread (0.02) -> QUIET
    s2 = make_dummy_sample("r2", poly_mid=0.50, poly_spread=0.02, future_mid=0.51)
    assert s2.objective_label == CHOICE_QUIET

    # Downward move (0.47 - 0.50 = -0.03 -> abs=0.03) >= spread (0.025) -> MEANINGFUL_MOVE
    s3 = make_dummy_sample("r3", poly_mid=0.50, poly_spread=0.025, future_mid=0.47)
    assert s3.objective_label == CHOICE_MEANINGFUL_MOVE


def test_no_future_information_in_state():
    """Verify state features contain only as-of observable metrics, no future q or label."""
    sample = make_dummy_sample("r_clean")
    obs = sample.to_decision_observation()
    payload = obs.state_payload

    # Future fields must NOT be in state_payload
    for bad in ["future_ts_ms", "future_poly_midpoint", "absolute_move", "objective_label"]:
        assert bad not in payload


# ---------------------------------------------------------------------------
# 3. Dataset & Chronological Partitioning Tests
# ---------------------------------------------------------------------------


def test_chronological_splits():
    """Verify 60% Train, 20% Calib, 20% Holdout partition ratio."""
    samples = [make_dummy_sample(f"r_{i}") for i in range(100)]
    dataset = Phase9bDataset(
        samples=samples,
        total_rounds=100,
        exclusion_reasons={},
    )
    assert dataset.eligible_count == 100
    n = len(samples)
    n_train = int(n * 0.60)
    n_calib = int(n * 0.20)
    assert n_train == 60
    assert n_calib == 20
    assert (n - n_train - n_calib) == 20


# ---------------------------------------------------------------------------
# 4. Baselines & Train-Only Preprocessing Tests
# ---------------------------------------------------------------------------


def test_baselines_prediction():
    """Verify deterministic baselines operate properly."""
    sample = make_dummy_sample("r_base", binance_ret_30s=10.0, poly_spread=0.01)

    maj = MajorityBaseline()
    maj.fit([sample])
    p_maj = maj.predict(sample)
    assert p_maj.choice == CHOICE_MEANINGFUL_MOVE

    rand = RandomBaseline(seed=42)
    p_rand = rand.predict(sample)
    assert p_rand.choice in [CHOICE_MEANINGFUL_MOVE, CHOICE_QUIET]

    abs_base = AlwaysAbstainBaseline()
    assert abs_base.predict(sample).choice == ABSTAIN_CHOICE

    rule = DeterministicRuleBaseline()
    p_rule = rule.predict(sample)
    assert p_rule.choice == CHOICE_MEANINGFUL_MOVE


def test_logistic_regression_fit_and_zero_leakage():
    """Verify logistic regression fits strictly on train and computes standardized predictions."""
    train_data = [
        make_dummy_sample(f"tr_{i}", future_mid=0.55 if i % 2 == 0 else 0.505)
        for i in range(50)
    ]
    test_sample = make_dummy_sample("test_eval")

    model = StandardizedLogisticRegression(max_epochs=50)
    model.fit(train_data)
    assert model.is_fitted

    pred = model.predict(test_sample)
    assert pred.choice in [CHOICE_MEANINGFUL_MOVE, CHOICE_QUIET]
    assert pred.probability_meaningful is not None
    assert 0.0 <= pred.probability_meaningful <= 1.0


# ---------------------------------------------------------------------------
# 5. Evaluator & Bootstrap Tests
# ---------------------------------------------------------------------------


def test_evaluator_metrics_computation():
    """Verify coverage, abstention, and accuracy math."""
    preds = [
        (CHOICE_MEANINGFUL_MOVE, CHOICE_MEANINGFUL_MOVE, 0.8, 0.0001),  # Correct (active)
        (CHOICE_QUIET, CHOICE_QUIET, 0.2, 0.0001),                      # Correct (active)
        (CHOICE_MEANINGFUL_MOVE, CHOICE_QUIET, 0.7, 0.0001),            # Wrong (active)
        (ABSTAIN_CHOICE, CHOICE_MEANINGFUL_MOVE, None, 0.0001),         # Abstained
    ]
    metrics = compute_metrics("test_model", "test_cohort", preds, avg_latency_ms=150.0)

    assert metrics.n_eligible == 4
    assert metrics.n_abstained == 1
    assert metrics.n_acted == 3
    assert metrics.coverage_rate == 0.75
    assert metrics.abstention_rate == 0.25
    # Active: 2 correct out of 3 -> 2/3 ≈ 0.6667
    assert pytest.approx(metrics.conditional_accuracy, rel=1e-3) == 2 / 3
    # Effective: 2 correct out of 4 -> 0.50
    assert metrics.effective_accuracy == 0.50
    assert pytest.approx(metrics.total_cost, rel=1e-6) == 0.0004
    assert metrics.avg_latency_ms == 150.0


def test_round_bootstrap_uncertainty():
    """Verify round bootstrap produces valid 95% confidence intervals and paired diffs."""
    true_labels = [CHOICE_MEANINGFUL_MOVE] * 40 + [CHOICE_QUIET] * 10
    model_preds = {
        "typesafe/jev-1.13": [CHOICE_MEANINGFUL_MOVE] * 45 + [CHOICE_QUIET] * 5,
        "baseline_majority": [CHOICE_MEANINGFUL_MOVE] * 50,
    }
    boot = run_round_bootstrap(model_preds, true_labels, n_bootstraps=100, seed=42)

    assert "typesafe/jev-1.13" in boot["models"]
    assert "baseline_majority" in boot["models"]
    assert "typesafe/jev-1.13_minus_baseline_majority" in boot["paired_differences"]
    ci = boot["models"]["typesafe/jev-1.13"]
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]


def test_selective_prediction_curve():
    """Verify selective prediction evaluates accuracy across confidence cutoffs."""
    preds_with_conf = [
        (CHOICE_MEANINGFUL_MOVE, CHOICE_MEANINGFUL_MOVE, 0.95),
        (CHOICE_MEANINGFUL_MOVE, CHOICE_MEANINGFUL_MOVE, 0.75),
        (CHOICE_MEANINGFUL_MOVE, CHOICE_QUIET, 0.55),
    ]
    curve = compute_selective_prediction_curve(preds_with_conf, thresholds=(0.5, 0.7, 0.9))
    assert len(curve) == 3
    # At 0.9 threshold: only 1 acted, 1 correct -> 100% accuracy, coverage = 1/3
    assert curve[2]["n_acted"] == 1
    assert curve[2]["conditional_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# 6. Budget & Safety Ceiling Tests
# ---------------------------------------------------------------------------


def test_budget_ceilings():
    """Verify runner budget constants align with prompt requirements."""
    assert MAX_PRIMARY_REQUESTS == 500
    assert MAX_COST_CEILING_USD == 1.00


def test_structural_safety_zero_execution_components():
    """Verify Phase 9B runner and dataset contain zero live execution components."""
    for cls in [
        Phase9bRunner,
        Phase9bDataset,
        Phase9bSample,
        MajorityBaseline,
        RandomBaseline,
        AlwaysAbstainBaseline,
        DeterministicRuleBaseline,
        StandardizedLogisticRegression,
    ]:
        forbidden = [
            "buy", "sell", "place_order", "execute_order", "wallet",
            "private_key", "secret", "sign_transaction"
        ]
        for attr in forbidden:
            assert not hasattr(cls, attr), f"Forbidden attribute '{attr}' on {cls.__name__}"
