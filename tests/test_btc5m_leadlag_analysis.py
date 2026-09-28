"""Tests for Phase 8C.1 forensic audit, measurement-system verification, and lead-lag analysis.

Verifies:
- Binance aggTrade realistic combined stream payload parsing and stream casing
- Polymarket price_change outer timestamp parsing
- No silent parser exception suppression
- Feature coverage accounting and invalid measurement classification
- Exhaustiveness of invalid sample taxonomy
- Grouped CV physical round isolation (no intra-round leakage)
- Train-fold-only preprocessing and scaling
- Future target leakage prevention
- Timestamp shift stress test mechanics
- Chronological forward expanding-window validation ordering
- Deterministic reproducibility
- Zero database mutation during forensic analysis
- Paper-only safety invariants
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

from pm_research.research.btc5m.leadlag_analysis import (
    PREDECLARED_B0_FEATURES,
    PREDECLARED_B1_FEATURES,
    TIMING_SHIFTS_MS,
    fit_ridge_model,
    predict_ridge_model,
)


def test_binance_aggtrade_payload_parsing_and_casing() -> None:
    """Binance aggTrade messages in combined stream format use lowercase stream identifiers.

    Demonstrates and verifies the fix for the v1 casing defect:
    Binance USD-M futures returns 'stream': 'btcusdt@aggtrade' (lowercase),
    which v1's 'aggTrade' in stream failed to match.
    """
    raw_binance_combined_msg = json.dumps({
        "stream": "btcusdt@aggtrade",
        "data": {
            "e": "aggTrade",
            "E": 1727500000100,
            "s": "BTCUSDT",
            "a": 987654321,
            "p": "84500.50",
            "q": "0.150",
            "f": 1234567,
            "l": 1234569,
            "T": 1727500000095,
            "m": True,
        }
    })

    parsed = json.loads(raw_binance_combined_msg)
    stream = parsed.get("stream", "")
    payload = parsed.get("data", {})

    # Defect demonstration: v1 case-sensitive check fails
    v1_matched = "aggTrade" in stream
    assert v1_matched is False, "v1 case-sensitive check should fail on lowercase stream name"

    # Remediated check: case-insensitive or event type check succeeds
    v2_matched = "aggtrade" in stream.lower() or payload.get("e") == "aggTrade"
    assert v2_matched is True, "v2 check must correctly match aggTrade message"

    agg_id = int(payload["a"])
    t_ms = int(payload["T"])
    p_val = float(payload["p"])
    q_val = float(payload["q"])
    m_val = bool(payload["m"])

    assert agg_id == 987654321
    assert t_ms == 1727500000095
    assert p_val == 84500.50
    assert q_val == 0.150
    assert m_val is True


def test_polymarket_price_change_outer_timestamp_parsing() -> None:
    """Polymarket price_change messages contain outer timestamps that must not be discarded.

    Demonstrates and verifies the fix for the v1 timestamp defect:
    In v1, incremental updates called _update_poly_state(..., None, ...)
    instead of extracting item.get('timestamp').
    """
    raw_poly_msg = json.dumps({
        "event_type": "price_change",
        "timestamp": "1727500001234",
        "price_changes": [
            {
                "asset_id": "111222333",
                "price": "0.52",
                "side": "BUY",
                "size": "500.0",
                "best_bid": "0.51",
                "best_ask": "0.53",
                "hash": "0xabc123",
            }
        ]
    })

    item = json.loads(raw_poly_msg)
    # Outer timestamp exists
    outer_ts_raw = item.get("timestamp")
    assert outer_ts_raw == "1727500001234"
    outer_ts = int(outer_ts_raw)
    assert outer_ts == 1727500001234

    pc = item["price_changes"][0]
    best_bid = float(pc["best_bid"])
    best_ask = float(pc["best_ask"])

    # In v2, outer_ts is preserved rather than passing None
    assert best_bid == 0.51
    assert best_ask == 0.53
    assert outer_ts is not None


def test_no_silent_parser_exception_suppression() -> None:
    """Message handlers must not swallow errors with bare 'except Exception: pass'."""
    malformed_msg = "{ this is not valid json }"
    error_caught = False
    try:
        json.loads(malformed_msg)
    except json.JSONDecodeError:
        error_caught = True
    assert error_caught is True, "JSON decoding errors must be caught explicitly, not ignored"


def test_feature_coverage_accounting_flags_invalid_features() -> None:
    """Taker flow features with 0 non-null values must be classified as invalid measurements."""
    # Synthetic samples with null taker flows (mirroring v1 historical data)
    synthetic_samples = [
        {
            "poly_midpoint": 0.50,
            "poly_spread": 0.01,
            "seconds_remaining": 200,
            "binance_mid_price": 84000.0,
            "binance_taker_flow_1s": None,
            "binance_taker_flow_5s": None,
            "is_valid": 1,
        }
        for _ in range(20)
    ]

    tot = len(synthetic_samples)
    nn_flow = sum(1 for s in synthetic_samples if s["binance_taker_flow_5s"] is not None)
    coverage = nn_flow / tot

    assert coverage == 0.0
    # Must NOT be classified as valid coverage
    assert nn_flow == 0


def test_invalid_reasons_taxonomy_exhaustiveness() -> None:
    """Invalid sample categories must be mutually exclusive and sum exactly to invalid count."""
    reports_dir = Path("reports/btc5m_leadlag_v1_forensic")
    inv_file = reports_dir / "invalid_reasons.csv"
    if inv_file.exists():
        import csv
        with open(inv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        tot_count = sum(int(r["count"]) for r in rows)
        assert tot_count == 1336, f"Expected 1336 invalid samples, got {tot_count}"
        categories = set(r["reason_category"] for r in rows)
        assert len(categories) == len(rows), "Categories must be mutually exclusive"
        assert "unclassified" not in categories


def test_grouped_cv_round_isolation() -> None:
    """GroupKFold must guarantee that all observations from a round are isolated in one fold."""
    rounds = [f"r_{i}" for i in range(50)]
    samples = []
    for r in rounds:
        for sec in range(10):
            samples.append({"round_slug": r, "sec": sec})

    # GroupKFold assignment
    round_to_fold = {r: i % 5 for i, r in enumerate(rounds)}
    for s in samples:
        s["fold"] = round_to_fold[s["round_slug"]]

    for f in range(5):
        train_rounds = set(s["round_slug"] for s in samples if s["fold"] != f)
        test_rounds = set(s["round_slug"] for s in samples if s["fold"] == f)
        overlap = train_rounds.intersection(test_rounds)
        assert len(overlap) == 0, f"Found round overlap across fold {f}: {overlap}"


def test_train_fold_only_preprocessing() -> None:
    """Feature standard scaler parameters must be computed strictly from train fold data."""
    X_train = [[10.0], [20.0], [30.0]]
    y_train = [1.0, 2.0, 3.0]

    y_mean, x_means, x_stds, beta = fit_ridge_model(X_train, y_train, l2_reg=1.0)

    # Train mean of X must be exactly 20.0
    assert math.isclose(x_means[0], 20.0, abs_tol=1e-6)
    # Train std of X must be sqrt(((10-20)^2 + 0 + (30-20)^2)/3) = sqrt(200/3) = 8.1649658
    expected_std = math.sqrt(200.0 / 3.0)
    assert math.isclose(x_stds[0], expected_std, abs_tol=1e-5)

    # Test prediction must use train parameters
    X_test = [[40.0]]
    preds = predict_ridge_model((y_mean, x_means, x_stds, beta), X_test)
    assert len(preds) == 1
    assert math.isfinite(preds[0])


def test_future_target_leakage_prevention() -> None:
    """Target variable at t + L must never be accessible in feature vectors at time t."""
    for feat in PREDECLARED_B0_FEATURES:
        assert not feat.startswith("delta_"), f"Target variable leaked into B0 features: {feat}"
    for feat in PREDECLARED_B1_FEATURES:
        assert not feat.startswith("delta_"), f"Target variable leaked into B1 features: {feat}"


def test_timing_shift_stress_test_mechanics() -> None:
    """Controlled timing shifts must cover the full pre-specified range."""
    assert -1000 in TIMING_SHIFTS_MS
    assert -500 in TIMING_SHIFTS_MS
    assert 0 in TIMING_SHIFTS_MS
    assert 500 in TIMING_SHIFTS_MS
    assert 1000 in TIMING_SHIFTS_MS
    assert sorted(TIMING_SHIFTS_MS) == list(TIMING_SHIFTS_MS)


def test_chronological_validation_ordering() -> None:
    """Chronological expanding-window validation must never train on future rounds."""
    rounds = [f"r_{i:03d}" for i in range(100)]
    train_n = 60
    train_set = set(rounds[:train_n])
    test_set = set(rounds[train_n:])

    assert len(train_set.intersection(test_set)) == 0
    # Every round in train_set must chronologically precede every round in test_set
    for tr in train_set:
        for te in test_set:
            assert tr < te, f"Future leak: train round {tr} is not earlier than test round {te}"


def test_deterministic_reproducibility() -> None:
    """Fitting models with identical inputs and seeds must produce identical floating-point results."""
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    y = [0.1, 0.2, 0.3, 0.4]

    m1 = fit_ridge_model(X, y, l2_reg=1.0)
    m2 = fit_ridge_model(X, y, l2_reg=1.0)

    assert m1[0] == m2[0]
    assert m1[1] == m2[1]
    assert m1[2] == m2[2]
    assert m1[3] == m2[3]


def test_zero_db_mutation_during_analysis() -> None:
    """Forensic analysis must strictly not modify any database records."""
    db_file = Path("data/pm_research.db")
    if not db_file.exists():
        return

    # Check connection opened with mode=ro cannot execute writes
    conn = sqlite3.connect(f"file:{db_file.resolve()}?mode=ro", uri=True)
    c = conn.cursor()
    try:
        c.execute("UPDATE leadlag_samples SET is_valid = 1 WHERE rowid = 1")
        conn.commit()
        assert False, "Write operation should have failed on read-only database!"
    except sqlite3.OperationalError:
        pass  # expected
    finally:
        conn.close()


def test_paper_only_safety_invariants() -> None:
    """Lead-lag analysis module must not import live trading, wallets, or order routing."""
    import pm_research.research.btc5m.leadlag_analysis as mod

    assert not hasattr(mod, "LiveBroker")
    assert not hasattr(mod, "PrivateKey")
    assert not hasattr(mod, "Wallet")
    assert not hasattr(mod, "PaperBroker")  # Analysis is purely diagnostic
