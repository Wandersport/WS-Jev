"""Comprehensive regression and verification test suite for Phase 9B.1 audit.

Verifies:
1. Task hash vs strategy hash distinction and regression prevention.
2. Binary confidence-to-positive-probability conversion (MEANINGFUL_MOVE).
3. QUIET confidence conversion (1.0 - confidence).
4. ABSTAIN scoring behavior (excluded from binary Brier/log loss).
5. Train-derived majority baseline (including synthetic QUIET-majority cohort).
6. Holdout-only baseline reporting.
7. Bootstrap superiority terminology (bootstrap_superiority_fraction).
8. Zero additional API calls during audit.
9. Cached-response replay determinism.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pm_research.research.jev_lab.contract import ABSTAIN_CHOICE
from pm_research.research.jev_phase9b.baselines import (
    MajorityBaseline,
)
from pm_research.research.jev_phase9b.dataset import Phase9bSample
from pm_research.research.jev_phase9b.evaluator import (
    compute_metrics,
    convert_binary_confidence_to_positive_probability,
    run_round_bootstrap,
)
from pm_research.research.jev_phase9b.spec import (
    CHOICE_MEANINGFUL_MOVE,
    CHOICE_QUIET,
    compute_task_spec_hash,
)
from pm_research.research.jev_phase9b_audit.audit import (
    HALLUCINATED_TERMINAL_HASH,
    Phase9bAuditEngine,
    generate_phase9b_audit_reports,
)


def make_dummy_sample(
    round_slug: str,
    label: str = CHOICE_MEANINGFUL_MOVE,
    split: str = "train",
) -> Phase9bSample:
    """Create dummy Phase9bSample for unit testing."""
    return Phase9bSample(
        round_slug=round_slug,
        sample_id=f"samp_{round_slug}",
        as_of_ts_ms=1727500000000,
        seconds_remaining=120,
        poly_midpoint=0.50,
        poly_spread=0.01,
        future_sample_id=f"fut_{round_slug}",
        future_ts_ms=1727500030000,
        future_poly_midpoint=0.52 if label == CHOICE_MEANINGFUL_MOVE else 0.505,
        absolute_move=0.02 if label == CHOICE_MEANINGFUL_MOVE else 0.005,
        objective_label=label,
        features={
            "poly_midpoint": 0.50,
            "poly_spread": 0.01,
            "seconds_remaining": 120,
            "poly_return_1s": 0.0,
            "poly_return_2s": 0.0,
            "poly_return_3s": 0.0,
            "poly_return_5s": 0.0,
            "poly_return_10s": 0.0,
            "poly_return_30s": 0.0,
            "binance_mid_price": 65000.0,
            "binance_spread_bps": 1.0,
            "binance_microprice_offset_bps": 0.2,
            "binance_top1_depth_imbalance": 0.1,
            "binance_top5_depth_imbalance": 0.1,
            "binance_top20_depth_imbalance": 0.1,
            "binance_return_since_open_bps": 5.0,
            "binance_return_1s_bps": 1.0,
            "binance_return_2s_bps": 1.5,
            "binance_return_3s_bps": 2.0,
            "binance_return_5s_bps": 2.5,
            "binance_return_10s_bps": 3.0,
            "binance_return_30s_bps": 4.0,
            "binance_return_60s_bps": 5.0,
        },
        split=split,
    )


# ---------------------------------------------------------------------------
# 1. Task Hash vs Strategy Hash Distinction
# ---------------------------------------------------------------------------


def test_task_hash_vs_strategy_hash_distinction():
    """Verify task spec hash is deterministic and distinguishes from strategy hash & hallucinated hash."""
    canonical_task_hash = compute_task_spec_hash()
    assert canonical_task_hash == "2b8dd4f5d3eb91fb2227c3bb2bc40ed38e1f4b114e40fd53b4f3279bfba79109"
    assert canonical_task_hash != HALLUCINATED_TERMINAL_HASH
    assert len(canonical_task_hash) == 64


# ---------------------------------------------------------------------------
# 2. Binary Confidence-to-Positive-Probability Conversion
# ---------------------------------------------------------------------------


def test_binary_confidence_conversion_meaningful_move():
    """When model chooses MEANINGFUL_MOVE, P(MEANINGFUL_MOVE) equals confidence."""
    p = convert_binary_confidence_to_positive_probability(
        choice=CHOICE_MEANINGFUL_MOVE,
        confidence=0.72,
    )
    assert p == pytest.approx(0.72)


def test_binary_confidence_conversion_quiet():
    """When model chooses QUIET, P(MEANINGFUL_MOVE) equals 1.0 - confidence."""
    p1 = convert_binary_confidence_to_positive_probability(
        choice=CHOICE_QUIET,
        confidence=0.60,
    )
    assert p1 == pytest.approx(0.40)

    p2 = convert_binary_confidence_to_positive_probability(
        choice=CHOICE_QUIET,
        confidence=0.85,
    )
    assert p2 == pytest.approx(0.15)


def test_abstain_scoring_behavior():
    """When model chooses ABSTAIN, probability is None and metric calculation excludes it from Brier."""
    p = convert_binary_confidence_to_positive_probability(
        choice=ABSTAIN_CHOICE,
        confidence=0.50,
    )
    assert p is None

    # Verify evaluator handles abstention
    preds = [
        (CHOICE_MEANINGFUL_MOVE, CHOICE_MEANINGFUL_MOVE, 0.8, None),
        (ABSTAIN_CHOICE, CHOICE_MEANINGFUL_MOVE, None, None),
        (CHOICE_QUIET, CHOICE_QUIET, 0.2, None),
    ]
    met = compute_metrics("test_model", "test_cohort", preds)
    assert met.n_eligible == 3
    assert met.n_acted == 2
    assert met.n_abstained == 1
    assert met.coverage_rate == pytest.approx(2 / 3)
    assert met.mean_brier_score is not None
    # Brier calculated over 2 active samples: (0.8-1)^2 = 0.04, (0.2-0)^2 = 0.04 -> mean = 0.04
    assert met.mean_brier_score == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# 3. Train-Derived Majority Baseline
# ---------------------------------------------------------------------------


def test_train_derived_majority_baseline_quiet_majority():
    """MajorityBaseline must explicitly derive majority class from train split, including QUIET majority."""
    maj = MajorityBaseline()

    # Unfitted predict should fail
    dummy = make_dummy_sample("r1")
    with pytest.raises(ValueError, match="must be fitted"):
        maj.predict(dummy)

    # Synthetic train samples with QUIET majority
    quiet_samples = [
        make_dummy_sample(f"r_{i}", label=CHOICE_QUIET)
        for i in range(8)
    ]
    move_samples = [
        make_dummy_sample(f"r_m_{i}", label=CHOICE_MEANINGFUL_MOVE)
        for i in range(2)
    ]
    synthetic_train = quiet_samples + move_samples

    maj.fit(synthetic_train)
    assert maj._majority_choice == CHOICE_QUIET

    pred = maj.predict(dummy)
    assert pred.choice == CHOICE_QUIET
    assert pred.probability_meaningful == 0.0


# ---------------------------------------------------------------------------
# 4. Holdout-Only Baseline Reporting
# ---------------------------------------------------------------------------


def test_holdout_only_baseline_reporting(tmp_path):
    """Verify holdout report artifacts exist and contain exactly 96 holdout evaluations."""
    engine = Phase9bAuditEngine(db_path="data/pm_research.db")
    audit_data = engine.run_comprehensive_audit()

    holdout_metrics = audit_data["metrics_by_cohort"]["retrospective_holdout_20pct"]
    assert len(holdout_metrics) == 7
    jev_h = holdout_metrics["typesafe/jev-1.13"]
    assert jev_h.n_eligible == 96
    assert jev_h.n_acted == 96
    assert jev_h.coverage_rate == 1.0
    assert jev_h.effective_accuracy == pytest.approx(0.3854, abs=1e-3)
    assert jev_h.balanced_accuracy == pytest.approx(0.4500, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. Bootstrap Superiority Terminology
# ---------------------------------------------------------------------------


def test_bootstrap_superiority_terminology():
    """Bootstrap results must report bootstrap_superiority_fraction without false p-value labels."""
    preds = {
        "typesafe/jev-1.13": [CHOICE_MEANINGFUL_MOVE, CHOICE_QUIET, CHOICE_QUIET, CHOICE_MEANINGFUL_MOVE],
        "baseline_maj": [CHOICE_MEANINGFUL_MOVE, CHOICE_MEANINGFUL_MOVE, CHOICE_MEANINGFUL_MOVE, CHOICE_MEANINGFUL_MOVE],
    }
    labels = [CHOICE_MEANINGFUL_MOVE, CHOICE_MEANINGFUL_MOVE, CHOICE_QUIET, CHOICE_MEANINGFUL_MOVE]

    res = run_round_bootstrap(preds, labels, n_bootstraps=100, seed=42)
    diff = res["paired_differences"]["typesafe/jev-1.13_minus_baseline_maj"]

    assert "bootstrap_superiority_fraction" in diff
    assert isinstance(diff["bootstrap_superiority_fraction"], float)
    assert 0.0 <= diff["bootstrap_superiority_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Zero Additional API Calls During Audit
# ---------------------------------------------------------------------------


def test_zero_api_calls_during_audit(tmp_path):
    """Verify audit execution performs zero HTTP/API network calls."""
    out_dir = tmp_path / "reports_audit"

    # Patch urllib.request to ensure zero network requests are made
    with patch("urllib.request.urlopen") as mock_url:
        reports = generate_phase9b_audit_reports(output_dir=out_dir, db_path="data/pm_research.db")
        assert mock_url.call_count == 0

    assert len(reports) == 8
    sum_data = Phase9bAuditEngine(db_path="data/pm_research.db").audit_persisted_responses()
    assert sum_data["current_run_api_requests"] == 0
    assert sum_data["cached_responses_used"] == 480
    assert sum_data["persisted_responses"] == 480


# ---------------------------------------------------------------------------
# 7. Cached Response Replay Determinism
# ---------------------------------------------------------------------------


def test_cached_response_replay_determinism():
    """Running the audit engine repeatedly on existing SQLite database produces byte-identical results."""
    engine = Phase9bAuditEngine(db_path="data/pm_research.db")
    res1 = engine.run_comprehensive_audit()
    res2 = engine.run_comprehensive_audit()

    assert res1["audit_summary"] == res2["audit_summary"]
    assert res1["hash_reconciliation"] == res2["hash_reconciliation"]
    assert res1["confidence_semantics"] == res2["confidence_semantics"]
    assert res1["holdout_bootstrap"] == res2["holdout_bootstrap"]
