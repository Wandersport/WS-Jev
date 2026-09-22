"""Deep mathematical verification of probability calibration metrics, brier scores, log loss, and ECE."""

from __future__ import annotations

import math

from pm_research.calibration.metrics import CalibrationEngine
from pm_research.domain.models import CalibrationObservation
from pm_research.utils import now_utc


def test_calibration_hand_calculated_benchmark():
    """Verify Brier score, log loss, forecast bias, and ECE against exact manual mathematical calculations."""
    now = now_utc()
    # 3 hand-calculated observations:
    # 1. p=0.8, y=1.0 -> brier = (0.8-1.0)^2 = 0.04, log_loss = -ln(0.8) ~= 0.22314355
    # 2. p=0.2, y=0.0 -> brier = (0.2-0.0)^2 = 0.04, log_loss = -ln(1-0.2) ~= 0.22314355
    # 3. p=0.6, y=0.0 -> brier = (0.6-0.0)^2 = 0.36, log_loss = -ln(1-0.6) ~= 0.91629073
    obs = [
        CalibrationObservation(
            observation_id="o1",
            estimate_id="e1",
            market_id="m1",
            model_version="model_a",
            predicted_probability=0.8,
            actual_outcome=1.0,
            category="TECH",
            edge_bucket="HIGH",
            risk_state="NORMAL",
            brier_score=0.04,
            log_loss=0.22314355,
            resolved_at=now,
        ),
        CalibrationObservation(
            observation_id="o2",
            estimate_id="e2",
            market_id="m2",
            model_version="model_a",
            predicted_probability=0.2,
            actual_outcome=0.0,
            category="TECH",
            edge_bucket="LOW",
            risk_state="NORMAL",
            brier_score=0.04,
            log_loss=0.22314355,
            resolved_at=now,
        ),
        CalibrationObservation(
            observation_id="o3",
            estimate_id="e3",
            market_id="m3",
            model_version="model_b",
            predicted_probability=0.6,
            actual_outcome=0.0,
            category="MACRO",
            edge_bucket="MED",
            risk_state="CAUTION",
            brier_score=0.36,
            log_loss=0.91629073,
            resolved_at=now,
        ),
    ]

    engine = CalibrationEngine(num_bins=10)
    report = engine.compute_metrics(obs)

    # 1. Sample size
    assert report.sample_size == 3

    # 2. Brier score: (0.04 + 0.04 + 0.36) / 3 = 0.44 / 3 = 0.1466666... -> rounded to 4 decimals = 0.1467
    expected_brier = 0.44 / 3.0
    assert abs(report.brier_score - round(expected_brier, 4)) < 1e-4

    # 3. Log loss: (-ln(0.8) - ln(0.8) - ln(0.4)) / 3 = (0.22314355 + 0.22314355 + 0.91629073) / 3 = 0.4541926... -> 0.4542
    expected_log_loss = (-2.0 * math.log(0.8) - math.log(0.4)) / 3.0
    assert abs(report.log_loss - round(expected_log_loss, 4)) < 1e-4

    # 4. Forecast bias: mean(predicted) - mean(actual) = ((0.8+0.2+0.6)/3) - ((1.0+0.0+0.0)/3) = (1.6 - 1.0) / 3 = 0.20
    expected_bias = (1.6 - 1.0) / 3.0
    assert abs(report.forecast_bias - round(expected_bias, 4)) < 1e-4

    # 5. Sliced breakdowns
    assert "model_a" in report.by_model_version
    assert "model_b" in report.by_model_version
    assert report.by_model_version["model_a"]["count"] == 2
    assert report.by_model_version["model_b"]["count"] == 1
    # model_a brier: (0.04 + 0.04) / 2 = 0.04
    assert abs(report.by_model_version["model_a"]["brier_score"] - 0.04) < 1e-4

    assert "TECH" in report.by_category
    assert "MACRO" in report.by_category
    assert report.by_category["TECH"]["count"] == 2
    assert report.by_category["MACRO"]["count"] == 1


def test_calibration_empty_and_perfect_forecasts():
    """Verify calibration engine handles empty sequence and perfect 1.0/0.0 forecasts safely."""
    engine = CalibrationEngine()
    now = now_utc()

    # Empty
    empty_report = engine.compute_metrics([])
    assert empty_report.sample_size == 0
    assert empty_report.brier_score == 0.0
    assert empty_report.log_loss == 0.0
    assert empty_report.expected_calibration_error == 0.0

    # Perfect forecasts
    perfect_obs = [
        CalibrationObservation(
            observation_id="o1",
            estimate_id="e1",
            market_id="m1",
            model_version="v1",
            predicted_probability=0.999,
            actual_outcome=1.0,
            category="TECH",
            edge_bucket="HIGH",
            risk_state="NORMAL",
            brier_score=0.0,
            log_loss=0.0,
            resolved_at=now,
        ),
        CalibrationObservation(
            observation_id="o2",
            estimate_id="e2",
            market_id="m2",
            model_version="v1",
            predicted_probability=0.001,
            actual_outcome=0.0,
            category="TECH",
            edge_bucket="LOW",
            risk_state="NORMAL",
            brier_score=0.0,
            log_loss=0.0,
            resolved_at=now,
        ),
    ]
    perf_report = engine.compute_metrics(perfect_obs)
    assert perf_report.brier_score < 0.001
    assert perf_report.log_loss < 0.005
    assert abs(perf_report.forecast_bias) < 0.002
