"""Unit tests for mathematical models, UTC time, uncertainty, and calibration metrics."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from pm_research.calibration.metrics import (
    CalibrationEngine,
    CalibrationObservation,
)
from pm_research.config import SystemConfig
from pm_research.domain.models import Market, MarketQuote, MarketStatus
from pm_research.pipeline.ilsa import logit, sigmoid
from pm_research.pipeline.uncertainty import UncertaintyModel
from pm_research.utils import ensure_utc, now_utc, parse_iso_utc, to_iso_utc


def test_utc_handling():
    """Verify strict UTC compliance and serialization."""
    now = now_utc()
    assert now.tzinfo == timezone.utc

    # Naive datetime must raise ValueError
    naive = datetime(2026, 9, 22, 12, 0, 0)
    with pytest.raises(ValueError, match="Naive datetime provided"):
        ensure_utc(naive)

    # ISO roundtrip
    iso_str = to_iso_utc(now)
    assert iso_str.endswith("Z")
    parsed = parse_iso_utc(iso_str)
    assert parsed.tzinfo is not None
    assert parsed.astimezone(timezone.utc) == now


def test_probability_math_and_clipping():
    """Verify logit, sigmoid, and edge bounds."""
    assert math.isclose(sigmoid(0.0), 0.5, abs_tol=1e-6)
    assert math.isclose(logit(0.5), 0.0, abs_tol=1e-6)

    # Monotonicity
    assert sigmoid(-5.0) < sigmoid(0.0) < sigmoid(5.0)
    assert logit(0.1) < logit(0.5) < logit(0.9)

    # Large values don't overflow
    assert sigmoid(100.0) == 1.0
    assert sigmoid(-100.0) == 0.0


def test_uncertainty_model_penalties():
    """Verify uncertainty penalties for spread, liquidity, and staleness."""
    config = SystemConfig()
    model = UncertaintyModel(config)

    now = now_utc()
    # Healthy market
    healthy_market = Market(
        market_id="m_healthy",
        question="Q?",
        category="TEST",
        status=MarketStatus.ACTIVE,
        resolution_time=now,
        quote=MarketQuote(
            yes_bid=0.49,
            yes_ask=0.51,
            spread=0.02,
            liquidity=10000.0,
            volume_24h=5000.0,
        ),
    )
    sigma_healthy, penalties_healthy = model.estimate_uncertainty(healthy_market, age_seconds=10.0)

    # Thin & wide spread market
    stressed_market = Market(
        market_id="m_stressed",
        question="Q?",
        category="TEST",
        status=MarketStatus.ACTIVE,
        resolution_time=now,
        quote=MarketQuote(
            yes_bid=0.40,
            yes_ask=0.55,
            spread=0.15,
            liquidity=200.0,
            volume_24h=100.0,
        ),
    )
    sigma_stressed, penalties_stressed = model.estimate_uncertainty(stressed_market, age_seconds=300.0)

    # Stressed market must have strictly higher uncertainty
    assert sigma_stressed > sigma_healthy
    assert penalties_stressed["spread"] > penalties_healthy["spread"]
    assert penalties_stressed["liquidity"] > penalties_healthy["liquidity"]
    assert penalties_stressed["staleness"] > penalties_healthy["staleness"]


def test_calibration_formulas():
    """Verify Brier score, log loss, forecast bias, and ECE formulas."""
    engine = CalibrationEngine(num_bins=5)

    now = now_utc()
    observations = [
        # Predicted 0.8, outcome 1 (good forecast)
        CalibrationObservation(
            observation_id="c1",
            estimate_id="e1",
            market_id="m1",
            model_version="v1",
            predicted_probability=0.8,
            actual_outcome=1.0,
            category="TECH",
            edge_bucket="0.06-0.10",
            risk_state="NORMAL",
            brier_score=(0.8 - 1.0) ** 2,  # 0.04
            log_loss=-math.log(0.8),
            resolved_at=now,
        ),
        # Predicted 0.2, outcome 0 (good forecast)
        CalibrationObservation(
            observation_id="c2",
            estimate_id="e2",
            market_id="m2",
            model_version="v1",
            predicted_probability=0.2,
            actual_outcome=0.0,
            category="TECH",
            edge_bucket="0.06-0.10",
            risk_state="NORMAL",
            brier_score=(0.2 - 0.0) ** 2,  # 0.04
            log_loss=-math.log(0.8),
            resolved_at=now,
        ),
        # Predicted 0.7, outcome 0 (poor forecast)
        CalibrationObservation(
            observation_id="c3",
            estimate_id="e3",
            market_id="m3",
            model_version="v1",
            predicted_probability=0.7,
            actual_outcome=0.0,
            category="TECH",
            edge_bucket="0.06-0.10",
            risk_state="NORMAL",
            brier_score=(0.7 - 0.0) ** 2,  # 0.49
            log_loss=-math.log(0.3),
            resolved_at=now,
        ),
    ]

    report = engine.compute_metrics(observations)
    expected_brier = (0.04 + 0.04 + 0.49) / 3.0
    assert math.isclose(report.brier_score, expected_brier, abs_tol=1e-4)

    expected_bias = ((0.8 + 0.2 + 0.7) - (1.0 + 0.0 + 0.0)) / 3.0
    assert math.isclose(report.forecast_bias, expected_bias, abs_tol=1e-4)

    assert report.sample_size == 3
    assert report.expected_calibration_error >= 0.0
    assert report.maximum_calibration_error >= 0.0
