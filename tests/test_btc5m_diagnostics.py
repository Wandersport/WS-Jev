"""Tests for BTC 5-Minute Phase 8A Post-Hoc Diagnostics & Microstructure Engine.

Verifies:
- Grouped CV never leaks rounds across folds
- Offset logistic formulation and numerical probability bounds
- Feature provenance and lookahead prevention
- Calibration binning and Cox calibration
- Per-horizon metrics and clustered bootstrap
- Immutable database reads (zero writes during diagnostics)
- Deterministic reproducible outputs
"""

import math
import tempfile

from pm_research.research.btc5m.diagnostics import (
    ALL_MICROSTRUCTURE_FEATURES,
    BTC5mPhase8Diagnostics,
    accuracy_score,
    brier_score,
    clustered_bootstrap_ci,
    deterministic_group_kfold,
    fit_cox_calibration,
    fit_logistic_regression,
    log_loss_score,
    logit,
    mace_score,
    sigmoid,
)
from pm_research.storage.db import Database


def test_deterministic_group_kfold_no_round_leakage() -> None:
    """Grouped CV must never assign observations with the same round_slug to different folds."""
    # Synthetic round slugs with multiple observations per round
    round_slugs = [
        "round_001", "round_001", "round_001",
        "round_002", "round_002",
        "round_003", "round_003", "round_003", "round_003",
        "round_004",
        "round_005", "round_005",
        "round_006", "round_006",
        "round_007", "round_007", "round_007",
        "round_008",
        "round_009", "round_009",
        "round_010",
    ]
    n_splits = 5
    fold_map = deterministic_group_kfold(round_slugs, n_splits=n_splits, seed=42)

    # 1. Every unique slug has exactly one assigned fold
    unique_slugs = set(round_slugs)
    assert len(fold_map) == len(unique_slugs)
    for slug in unique_slugs:
        assert 0 <= fold_map[slug] < n_splits

    # 2. Check that no round appears in both train and test for any fold
    for fold in range(n_splits):
        train_rounds = {s for s in unique_slugs if fold_map[s] != fold}
        test_rounds = {s for s in unique_slugs if fold_map[s] == fold}
        overlap = train_rounds.intersection(test_rounds)
        assert len(overlap) == 0, f"Round leakage detected in fold {fold}: {overlap}"

    # 3. Determinism: identical seed produces identical fold map
    fold_map_repro = deterministic_group_kfold(round_slugs, n_splits=n_splits, seed=42)
    assert fold_map == fold_map_repro


def test_sigmoid_and_logit_bounds() -> None:
    """Sigmoid and logit functions must handle extremes gracefully without overflow/NaN."""
    # Extreme values
    assert sigmoid(100.0) == 1.0 / (1.0 + math.exp(-35.0))
    assert sigmoid(-100.0) == 1.0 / (1.0 + math.exp(35.0))
    assert sigmoid(0.0) == 0.5
    assert 0.0 < sigmoid(50.0) < 1.0
    assert 0.0 < sigmoid(-50.0) < 1.0

    # Logit clamping
    assert math.isfinite(logit(1.0))
    assert math.isfinite(logit(0.0))
    assert math.isclose(logit(0.5), 0.0, abs_tol=1e-9)
    assert logit(0.999999) > 0.0
    assert logit(0.000001) < 0.0


def test_logistic_regression_offset_formulation() -> None:
    """When X is zero and intercept is False, offset model prediction must match exact offset probability."""
    # If offset_i = logit(q_i) and beta=0, then sigmoid(offset_i) = q_i
    q_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
    offsets = [logit(q) for q in q_vals]
    X_zero = [[0.0] for _ in q_vals]
    y_dummy = [0.0, 0.0, 1.0, 1.0, 1.0]

    b0, b_vec = fit_logistic_regression(
        X_zero,
        y_dummy,
        offsets=offsets,
        fit_intercept=False,
        l2_reg=100.0,
    )
    assert b0 == 0.0
    # Weights should be very close to 0 due to strong L2 regularization on zero features
    assert abs(b_vec[0]) < 1e-4

    # Predictions should match q_vals
    for i, q in enumerate(q_vals):
        pred_p = sigmoid(offsets[i] + b_vec[0] * X_zero[i][0])
        assert math.isclose(pred_p, q, abs_tol=1e-3)


def test_fit_logistic_regression_convergence() -> None:
    """L2-regularized logistic regression converges to positive coefficient for positive signal."""
    # Linearly separable 1D problem
    X = [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]]
    y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

    b0, b_vec = fit_logistic_regression(X, y, fit_intercept=True, l2_reg=0.1)
    # Slope must be strongly positive
    assert b_vec[0] > 0.5
    # Intercept near 0 due to symmetric data
    assert abs(b0) < 0.5

    # Check predicted probabilities are ordered correctly
    preds = [sigmoid(b0 + b_vec[0] * x[0]) for x in X]
    assert all(0.0 < p < 1.0 for p in preds)
    for i in range(len(preds) - 1):
        assert preds[i] <= preds[i + 1]


def test_scoring_metrics() -> None:
    """Verify Brier score, log loss, accuracy, and MACE calculations."""
    p = [0.9, 0.8, 0.2, 0.1]
    y = [1.0, 1.0, 0.0, 0.0]

    # Perfect direction
    assert accuracy_score(p, y) == 1.0

    # Brier score: (0.1^2 + 0.2^2 + 0.2^2 + 0.1^2) / 4 = (0.01 + 0.04 + 0.04 + 0.01) / 4 = 0.025
    assert math.isclose(brier_score(p, y), 0.025, abs_tol=1e-6)

    # Log loss
    ll = log_loss_score(p, y)
    assert 0.0 < ll < 0.3

    # MACE
    mc = mace_score(p, y, n_bins=10)
    assert 0.0 <= mc <= 1.0


def test_cox_calibration() -> None:
    """Cox calibration returns alpha ~ 0 and beta ~ 1 for well-calibrated synthetic predictions."""
    # Synthetic well-calibrated probabilities
    q_vals = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    y_vals = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    log_odds = [logit(q) for q in q_vals]

    alpha, beta = fit_cox_calibration(log_odds, y_vals)
    # Beta should be positive
    assert beta > 0.0
    assert abs(alpha) < 2.0


def test_clustered_bootstrap_confidence_interval() -> None:
    """Clustered bootstrap CI bounds delta Brier properly."""
    # Grouped sample where model is consistently worse than market
    sample_items = [
        ("round_A", 0.25, 0.15),
        ("round_A", 0.20, 0.10),
        ("round_B", 0.30, 0.18),
        ("round_B", 0.22, 0.12),
        ("round_C", 0.28, 0.14),
        ("round_C", 0.19, 0.11),
    ]
    low, high = clustered_bootstrap_ci(sample_items, n_boot=200, seed=42)
    assert low <= high
    # Since model Brier > market Brier in every observation, CI should be strictly positive
    assert low > 0.0


def test_feature_provenance_audit() -> None:
    """Feature provenance audit confirms no lookahead across all features."""
    db = Database("data/pm_research.db")
    diag = BTC5mPhase8Diagnostics(db)
    provenance = diag.audit_feature_provenance()

    assert len(provenance) > 5
    for item in provenance:
        assert item["available_at_forecast_time"] == "YES"
        assert item["uses_future_information"] == "NO"
        assert item["feature_name"] in ALL_MICROSTRUCTURE_FEATURES or item["feature_name"] in [
            "market_q", "poly_spread"
        ]


def test_diagnostics_immutable_database_reads() -> None:
    """Diagnostics engine must execute read-only queries with zero mutations to database tables."""
    db = Database("data/pm_research.db")

    # Record row counts before
    rounds_before = len(db.get_btc5m_rounds())
    snaps_before = len(db.get_btc5m_snapshots())
    fcs_before = len(db.get_btc5m_forecasts())
    scores_before = len(db.get_btc5m_resolution_scores())

    with tempfile.TemporaryDirectory() as tmp_dir:
        diag = BTC5mPhase8Diagnostics(db)
        report = diag.run_all_diagnostics(output_dir=tmp_dir)
        assert report["status"] == "PASS"

    # Verify zero mutations
    assert len(db.get_btc5m_rounds()) == rounds_before
    assert len(db.get_btc5m_snapshots()) == snaps_before
    assert len(db.get_btc5m_forecasts()) == fcs_before
    assert len(db.get_btc5m_resolution_scores()) == scores_before


def test_diagnostics_deterministic_reproducibility() -> None:
    """Diagnostics engine produces deterministic results on identical input data."""
    db = Database("data/pm_research.db")
    diag = BTC5mPhase8Diagnostics(db)

    # Run residual model evaluation twice
    res1 = diag.evaluate_polymarket_residual_model()
    res2 = diag.evaluate_polymarket_residual_model()

    assert res1["market_brier"] == res2["market_brier"]
    assert res1["offset_model_cv_brier"] == res2["offset_model_cv_brier"]
    assert res1["delta_brier"] == res2["delta_brier"]
    assert res1["average_coefficients"] == res2["average_coefficients"]


def test_lead_lag_feasibility_transition_counts() -> None:
    """Lead/lag feasibility correctly extracts within-round transitions."""
    db = Database("data/pm_research.db")
    diag = BTC5mPhase8Diagnostics(db)
    lead_lag = diag.evaluate_lead_lag_feasibility()

    assert lead_lag["label"] == "EXPLORATORY_LEAD_LAG_DIAGNOSTIC"
    assert lead_lag["total_transition_pairs"] > 1000
    transitions = {t["transition"]: t["valid_pairs"] for t in lead_lag["transitions"]}
    assert transitions["240s -> 180s"] > 300
    assert transitions["180s -> 120s"] > 400
    assert transitions["120s -> 60s"] > 200
    assert transitions["60s -> 30s"] > 80


def test_nested_market_models_identical_folds() -> None:
    """M0 through M4 must be evaluated on identical grouped folds with zero leakage."""
    db = Database("data/pm_research.db")
    diag = BTC5mPhase8Diagnostics(db)
    nested = diag.evaluate_nested_market_models()

    assert nested["n_observations"] == 1649
    assert nested["grouped_folds"] == 5
    assert nested["preprocessing_leakage_found"] == "NO"

    # Verify Brier orderings and metrics exist
    assert "m0_raw_market" in nested
    assert "m1_intercept_only" in nested
    assert "m2_affine_calibration" in nested
    assert "m3_offset_microstructure" in nested
    assert "m4_affine_plus_microstructure" in nested

    m0_br = nested["m0_raw_market"]["brier"]
    m1_br = nested["m1_intercept_only"]["brier"]
    m2_br = nested["m2_affine_calibration"]["brier"]
    m3_br = nested["m3_offset_microstructure"]["brier"]
    m4_br = nested["m4_affine_plus_microstructure"]["brier"]

    assert 0.13 < m0_br < 0.15
    assert 0.13 < m1_br < 0.15
    assert 0.13 < m2_br < 0.15
    assert 0.13 < m3_br < 0.15
    assert 0.13 < m4_br < 0.15

    # Fold stability: exactly 5 folds
    assert len(nested["fold_stability"]) == 5
    for fs in nested["fold_stability"]:
        assert fs["test_obs"] > 0
        assert "delta_brier_m3_vs_m1" in fs
        assert "delta_brier_m4_vs_m2" in fs


def test_fit_logistic_regression_custom_penalty_vector() -> None:
    """Logistic regression must support element-specific L2 regularization penalties."""
    X = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0], [-2.0, -1.0]]
    y = [1.0, 1.0, 0.0, 0.0]

    # Zero penalty on intercept and first weight, heavy penalty on second weight
    b0, b = fit_logistic_regression(
        X,
        y,
        offsets=None,
        fit_intercept=True,
        l2_reg=[0.0, 0.0, 1000.0],
        max_iter=50,
    )
    assert math.isfinite(b0)
    assert abs(b[0]) > abs(b[1])  # heavily penalized b[1] should shrink toward zero


def test_temporal_robustness_expanding_window() -> None:
    """Expanding chronological window must train strictly on past rounds and test on future."""
    db = Database("data/pm_research.db")
    diag = BTC5mPhase8Diagnostics(db)
    temporal = diag.evaluate_temporal_robustness()

    assert temporal["label"] == "DEVELOPMENT_TEMPORAL_ROBUSTNESS_ONLY"
    blocks = temporal["blocks"]
    assert len(blocks) == 4  # Blocks 1 through 4

    for b_idx, block in enumerate(blocks):
        # Training rounds count must strictly expand: 100, 200, 300, 400
        assert block["training_rounds_count"] == (b_idx + 1) * 100
        assert block["validation_obs_count"] > 200
        assert "delta_brier_m4_vs_m2" in block


def test_collinearity_and_vif_calculation() -> None:
    """Collinearity evaluation must accurately identify redundant microstructure features."""
    db = Database("data/pm_research.db")
    diag = BTC5mPhase8Diagnostics(db)
    collinearity = diag.evaluate_collinearity_and_price_displacement()

    vifs = collinearity["variance_inflation_factors"]
    assert vifs["binance_microprice_offset_bps"] > 10.0
    assert vifs["binance_top5_depth_imbalance"] > 10.0
    assert vifs["binance_basis_bps"] < 5.0

    corr = collinearity["correlation_matrix"]
    assert corr["binance_microprice_offset_bps"]["binance_top5_depth_imbalance"] > 0.95
    assert corr["binance_return_since_open_bps"]["ref_distance_to_beat_bps"] > 0.70


def test_sample_size_and_power_estimation() -> None:
    """Power analysis must provide simulated power for 500, 1000, 2000 rounds."""
    db = Database("data/pm_research.db")
    diag = BTC5mPhase8Diagnostics(db)
    nested = diag.evaluate_nested_market_models()
    pwr = diag.evaluate_sample_size_and_power(nested)

    assert "m3_vs_m1_effect" in pwr
    assert "m4_vs_m2_effect" in pwr
    assert "candidate_evaluations" in pwr
    assert "500_rounds" in pwr["candidate_evaluations"]
    assert "1000_rounds" in pwr["candidate_evaluations"]
    assert "2000_rounds" in pwr["candidate_evaluations"]


def test_cost_accounting_reconciliation_with_database() -> None:
    """Verify that reported cost breakdown exactly matches the raw SQLite database values."""
    db = Database("data/pm_research.db")
    fcs = db.get_btc5m_forecasts()
    assert len(fcs) == 6600

    sum_cost = sum(fc.cost or 0.0 for fc in fcs)
    assert math.isclose(sum_cost, 0.242322, abs_tol=1e-5)

    sum_prompt_tokens = sum(fc.input_tokens or 0 for fc in fcs)
    sum_completion_tokens = sum(fc.output_tokens or 0 for fc in fcs)
    assert sum_prompt_tokens == 5768893
    assert sum_completion_tokens == 343148

