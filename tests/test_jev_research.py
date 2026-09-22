"""Unit and integration tests for TypeSafe Jev prospective shadow research infrastructure."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from pm_research.research.jev_openrouter import (
    CONDITION_BLIND,
    CONDITION_MARKET_AWARE,
    JEV_MODEL_PIN,
    JEV_SCHEMA_VERSION,
    JevCaptureRecord,
    JevForecast,
    JevOpenRouterClient,
    JevResolutionScore,
)
from pm_research.research.jev_shadow import (
    JevResolutionScorer,
    JevShadowRunner,
)
from pm_research.storage.db import Database


def _make_mock_jev_response(
    p_yes: float = 0.42,
    choice: str = "NO",
    model: str = "typesafe/jev-1.13-20260917",
) -> dict[str, Any]:
    """Helper to generate a mock OpenRouter Jev decision response."""
    return {
        "id": "dec_mock_123456",
        "provider": "TypeSafe",
        "model": model,
        "answers": {
            "will_resolve_yes": {
                "type": "noul",
                "noul": p_yes,
            },
            "outcome_choice": {
                "type": "choice",
                "choice": choice,
                "confidence": 0.85,
            },
        },
        "usage": {
            "input_tokens": 450,
            "output_tokens": 48,
            "cost": 0.000018,
        },
    }


# ==============================================================================
# 1. JevOpenRouterClient Unit Tests
# ==============================================================================


def test_jev_state_builder_blind_and_aware():
    """Verify state builder produces sanitized payloads and respects condition parameters."""
    client = JevOpenRouterClient(api_key="mock_test_key")
    now = datetime(2026, 9, 22, 12, 0, 0, tzinfo=timezone.utc)
    res_time = datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

    # Condition A: JEV_BLIND
    long_question = "Will candidate X win? " * 100
    state_blind = client.build_state(
        question=long_question,
        criteria="Must be confirmed by official election authority.",
        category="POLITICS",
        captured_at=now,
        resolution_time=res_time,
        condition=CONDITION_BLIND,
    )
    assert state_blind["condition"] == CONDITION_BLIND
    assert len(state_blind["market_question"]) <= 500
    assert "market_consensus_probability" not in state_blind
    assert state_blind["category"] == "POLITICS"

    # Condition B: JEV_MARKET_AWARE
    state_aware = client.build_state(
        question="Will inflation drop below 2%?",
        criteria=None,
        category="ECONOMICS",
        captured_at=now,
        resolution_time=res_time,
        condition=CONDITION_MARKET_AWARE,
        market_prob=0.35,
    )
    assert state_aware["condition"] == CONDITION_MARKET_AWARE
    assert state_aware["market_consensus_probability"] == 0.35
    assert "market_consensus_notice" in state_aware

    # Error cases
    with pytest.raises(ValueError, match="Unknown Jev research condition"):
        client.build_state("Q", "C", "CAT", now, res_time, condition="INVALID_CONDITION")

    with pytest.raises(ValueError, match="market_prob is required"):
        client.build_state(
            "Q", "C", "CAT", now, res_time, condition=CONDITION_MARKET_AWARE, market_prob=None
        )


def test_jev_payload_and_request_hash():
    """Verify request schema assembly and deterministic hash calculation."""
    client = JevOpenRouterClient(api_key="mock_test_key")
    state = {"market_question": "Test question", "condition": CONDITION_BLIND}
    payload = client.build_request_payload(state)

    assert payload["model"] == JEV_MODEL_PIN
    assert "will_resolve_yes" in payload["questions"]
    assert payload["questions"]["will_resolve_yes"]["type"] == "noul"
    assert "outcome_choice" in payload["questions"]
    assert payload["questions"]["outcome_choice"]["type"] == "choice"

    # Deterministic hash
    h1 = client.compute_request_hash(payload)
    h2 = client.compute_request_hash(payload)
    assert h1 == h2
    assert len(h1) == 64


def test_jev_exact_model_pinning_enforcement(monkeypatch, tmp_path):
    """Verify that any returned model not matching the pinned prefix raises ValueError."""
    client = JevOpenRouterClient(api_key="mock_test_key", cache_dir=tmp_path / "cache")
    state = {"market_question": "Test", "condition": CONDITION_BLIND}

    # Case 1: Unapproved foreign model returned
    bad_resp_foreign = _make_mock_jev_response(model="openai/gpt-4o")

    class MockHTTPResponse:
        status = 200

        def read(self):
            return json.dumps(bad_resp_foreign).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: MockHTTPResponse())

    with pytest.raises(ValueError, match="Model mismatch"):
        client.query_decision(state, bypass_cache=True)

    # Case 2: Unpinned 'jev-latest' returned
    bad_resp_latest = _make_mock_jev_response(model="typesafe/jev-latest")

    class MockLatestResponse:
        status = 200

        def read(self):
            return json.dumps(bad_resp_latest).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: MockLatestResponse())

    with pytest.raises(ValueError, match="Model mismatch"):
        client.query_decision(state, bypass_cache=True)

    # Case 3: Pinned model variant matching typesafe/jev-1.13
    good_resp = _make_mock_jev_response(model="typesafe/jev-1.13-20260917")

    class MockGoodResponse:
        status = 200

        def read(self):
            return json.dumps(good_resp).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: MockGoodResponse())

    resp, is_cached, req_hash = client.query_decision(state, bypass_cache=True)
    assert resp["model"] == "typesafe/jev-1.13-20260917"
    assert not is_cached


def test_jev_missing_api_key_raises(monkeypatch):
    """Verify that calling remote API without an available key raises a clear ValueError."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = JevOpenRouterClient(api_key=None)

    with pytest.raises(ValueError, match="required for Jev remote research inference"):
        client._get_api_key()


def test_jev_local_response_caching(tmp_path):
    """Verify that locally cached responses bypass remote calls."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    client = JevOpenRouterClient(api_key="mock_key", cache_dir=cache_dir)

    state = {"market_question": "Will solar power reach 30%?", "condition": CONDITION_BLIND}
    payload = client.build_request_payload(state)
    req_hash = client.compute_request_hash(payload)

    # Pre-populate cache file
    cached_data = _make_mock_jev_response(p_yes=0.77, choice="YES")
    cache_file = cache_dir / f"{req_hash}.json"
    cache_file.write_text(json.dumps(cached_data), encoding="utf-8")

    # query_decision should retrieve cached data without making an HTTP request
    resp, is_cached, returned_hash = client.query_decision(state, bypass_cache=False)
    assert is_cached is True
    assert returned_hash == req_hash
    assert resp["answers"]["will_resolve_yes"]["noul"] == 0.77


def test_jev_forecast_parsing():
    """Verify parsing of valid responses into immutable JevForecast objects."""
    client = JevOpenRouterClient(api_key="mock_key")
    now = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
    res_time = datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)
    mock_resp = _make_mock_jev_response(p_yes=0.68, choice="YES")

    forecast = client.parse_forecast(
        response_data=mock_resp,
        capture_id="cap_001",
        market_id="mkt_poly_101",
        condition=CONDITION_BLIND,
        question="Will AI pass benchmark Z?",
        criteria="Standard benchmark rules.",
        resolution_time=res_time,
        captured_at=now,
        request_hash="abc123hash",
    )

    assert forecast.market_id == "mkt_poly_101"
    assert forecast.jev_yes_probability == 0.68
    assert forecast.jev_no_probability == 0.32
    assert forecast.jev_choice == "YES"
    assert forecast.confidence == 0.85
    assert forecast.input_tokens == 450
    assert forecast.output_tokens == 48
    assert forecast.cost == 0.000018
    assert forecast.model_id == JEV_MODEL_PIN
    assert forecast.schema_version == JEV_SCHEMA_VERSION

    # Boundary verification: out of bounds probability must fail
    invalid_resp = _make_mock_jev_response(p_yes=1.45)
    with pytest.raises(ValueError, match="out of bounds"):
        client.parse_forecast(
            invalid_resp, "cap_1", "m_1", CONDITION_BLIND, "Q", None, None, now, "hash"
        )


# ==============================================================================
# 2. Database Persistence Tests
# ==============================================================================


def test_jev_database_persistence(tmp_path):
    """Verify SQLite persistence for Jev captures, forecasts, and resolution scores."""
    db_path = tmp_path / "test_jev.db"
    db = Database(str(db_path))
    now = datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)
    res_dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)

    # 1. Save and query JevCaptureRecord
    capture = JevCaptureRecord(
        capture_id="cap_test_001",
        market_id="pub_poly_42",
        captured_at=now,
        market_question="Will project ship on time?",
        category="TECH",
        resolution_time=res_dt,
        market_prob=0.55,
        ilsa_prob=0.51,
        jev_blind_prob=0.60,
        jev_market_aware_prob=0.58,
    )
    db.save_jev_capture(capture)

    captures = db.get_jev_captures()
    assert len(captures) == 1
    loaded_cap = captures[0]
    assert loaded_cap.capture_id == "cap_test_001"
    assert loaded_cap.market_prob == 0.55
    assert loaded_cap.ilsa_prob == 0.51
    assert loaded_cap.jev_blind_prob == 0.60
    assert loaded_cap.jev_market_aware_prob == 0.58
    assert loaded_cap.resolved_outcome is None

    # Verify unresolved query
    unresolved = db.get_unresolved_jev_captures()
    assert len(unresolved) == 1
    assert unresolved[0].capture_id == "cap_test_001"

    # 2. Save and query JevForecast
    fc_blind = JevForecast(
        forecast_id="fc_001",
        capture_id="cap_test_001",
        market_id="pub_poly_42",
        condition=CONDITION_BLIND,
        model_id=JEV_MODEL_PIN,
        model_returned="typesafe/jev-1.13-20260917",
        schema_version=JEV_SCHEMA_VERSION,
        request_hash="reqhash_blind",
        captured_at_utc=now,
        market_question="Will project ship on time?",
        resolution_criteria="Official announcement.",
        market_resolution_time=res_dt,
        jev_yes_probability=0.60,
        jev_no_probability=0.40,
        jev_choice="YES",
        confidence=0.80,
        input_tokens=420,
        output_tokens=50,
        cost=0.000017,
        raw_response_hash="resphash_blind",
    )
    db.save_jev_forecast(fc_blind)

    forecasts = db.get_jev_forecasts(market_id="pub_poly_42")
    assert len(forecasts) == 1
    assert forecasts[0].condition == CONDITION_BLIND
    assert forecasts[0].jev_yes_probability == 0.60

    # 3. Update resolution and save resolution score
    resolved_time = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    db.update_jev_capture_resolution("cap_test_001", "YES", resolved_time)

    # Now unresolved count should be zero
    assert len(db.get_unresolved_jev_captures()) == 0

    score = JevResolutionScore(
        score_id="sc_001",
        capture_id="cap_test_001",
        market_id="pub_poly_42",
        resolved_outcome="YES",
        resolved_at=resolved_time,
        scored_at=now,
        market_prob=0.55,
        ilsa_prob=0.51,
        jev_blind_prob=0.60,
        jev_market_aware_prob=0.58,
        market_brier=0.2025,
        ilsa_brier=0.2401,
        jev_blind_brier=0.1600,
        jev_market_aware_brier=0.1764,
        market_log_loss=0.5978,
        ilsa_log_loss=0.6733,
        jev_blind_log_loss=0.5108,
        jev_market_aware_log_loss=0.5447,
    )
    db.save_jev_resolution_score(score)

    scores = db.get_jev_resolution_scores()
    assert len(scores) == 1
    assert scores[0].score_id == "sc_001"
    assert scores[0].jev_blind_brier == 0.1600


# ==============================================================================
# 3. JevShadowRunner & Research Isolation Invariants
# ==============================================================================


def test_jev_shadow_runner_preserves_safety_invariants(tmp_path, monkeypatch):
    """CRITICAL: Verify Jev shadow runner records forecasts WITHOUT creating orders or modifying portfolio."""
    db_path = tmp_path / "shadow_test.db"
    db = Database(str(db_path))

    # Mock public market adapter to return 2 active unresolved markets
    mock_public_markets = [
        {
            "market_id": "pub_1001",
            "question": "Will NASA launch the mission in 2026?",
            "category": "SCIENCE",
            "description": "Must launch before Dec 31, 2026.",
            "resolution_time": "2026-12-31T23:59:59Z",
            "last_price": 0.45,
            "liquidity": 15000.0,
            "volume_24h": 2500.0,
            "status": "ACTIVE",
        },
        {
            "market_id": "pub_1002",
            "question": "Will Fed cut rates at the next meeting?",
            "category": "ECONOMICS",
            "description": "Standard FOMC rate decision.",
            "resolution_time": "2026-10-31T23:59:59Z",
            "last_price": 0.70,
            "liquidity": 50000.0,
            "volume_24h": 8000.0,
            "status": "ACTIVE",
        },
    ]

    mock_adapter = MagicMock()
    mock_adapter.fetch_public_markets.return_value = mock_public_markets

    # Mock Jev OpenRouter client
    client = JevOpenRouterClient(api_key="mock_key", cache_dir=tmp_path / "cache")

    def mock_query(state, bypass_cache=False):
        cond = state.get("condition", CONDITION_BLIND)
        p = 0.50 if cond == CONDITION_BLIND else state.get("market_consensus_probability", 0.50)
        return _make_mock_jev_response(p_yes=p), False, "mock_hash"

    monkeypatch.setattr(client, "query_decision", mock_query)

    runner = JevShadowRunner(db=db, jev_client=client, market_adapter=mock_adapter)
    summary = runner.run_prospective_cycle(max_markets=5, min_liquidity=1000.0)

    # Verify cycle execution
    assert summary.markets_discovered == 2
    assert summary.markets_eligible == 2
    assert summary.markets_captured == 2
    assert summary.jev_requests_sent == 4  # 2 markets * 2 conditions (blind + aware)

    # Verify forecasts and captures in DB
    captures = db.get_jev_captures()
    assert len(captures) == 2
    forecasts = db.get_jev_forecasts()
    assert len(forecasts) == 4

    # STRICT INVARIANTS: Zero proposals, zero orders, zero fills, zero portfolio execution
    act_summary = db.get_activity_summary()
    assert act_summary["proposals"] == 0, "SHADOW VIOLATION: Trade proposals must never be created by Jev"
    assert act_summary["orders"] == 0, "SHADOW VIOLATION: Orders must never be submitted by Jev"
    assert act_summary["fills"] == 0, "SHADOW VIOLATION: Fills must never occur during Jev shadow cycle"
    assert len(db.get_all_positions()) == 0, "SHADOW VIOLATION: Positions must never be opened by Jev"


# ==============================================================================
# 4. JevResolutionScorer Unit Tests
# ==============================================================================


def test_jev_resolution_scorer_states(tmp_path):
    """Verify JevResolutionScorer behavior across empty, pending, and scored states."""
    db_path = tmp_path / "scorer_test.db"
    db = Database(str(db_path))
    scorer = JevResolutionScorer(db=db)

    # 1. Empty DB
    summary_empty = scorer.evaluate_pending_resolutions()
    assert summary_empty.scoring_status == "NO_CAPTURES_FOUND"
    assert summary_empty.total_captures == 0

    # 2. Add unresolved capture
    now = datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc)
    res_time = datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)
    cap = JevCaptureRecord(
        capture_id="cap_sc_01",
        market_id="pub_poly_99",
        captured_at=now,
        market_question="Will breakthrough X happen?",
        category="SCIENCE",
        resolution_time=res_time,
        market_prob=0.30,
        ilsa_prob=0.25,
        jev_blind_prob=0.20,
        jev_market_aware_prob=0.22,
    )
    db.save_jev_capture(cap)

    summary_pending = scorer.evaluate_pending_resolutions()
    assert summary_pending.scoring_status == "WAITING_FOR_FUTURE_RESOLUTIONS"
    assert summary_pending.total_captures == 1
    assert summary_pending.unresolved_captures == 1
    assert summary_pending.resolved_captures == 0

    # 3. Simulate genuine resolution to NO (actual outcome = 0.0)
    resolved_at = datetime(2026, 12, 1, 1, 0, tzinfo=timezone.utc)
    score = scorer.score_capture(cap, resolved_outcome="NO", resolved_at=resolved_at)

    assert score.market_id == "pub_poly_99"
    assert score.resolved_outcome == "NO"
    # Actual outcome = 0.0:
    # market_brier = (0.30 - 0)^2 = 0.09
    assert pytest.approx(score.market_brier, abs=1e-4) == 0.09
    # ilsa_brier = (0.25 - 0)^2 = 0.0625
    assert pytest.approx(score.ilsa_brier, abs=1e-4) == 0.0625
    # jev_blind_brier = (0.20 - 0)^2 = 0.04
    assert pytest.approx(score.jev_blind_brier, abs=1e-4) == 0.04

    # 4. Evaluate scored state
    summary_scored = scorer.evaluate_pending_resolutions()
    assert summary_scored.scoring_status == "SCORED"
    assert summary_scored.resolved_captures == 1
    assert summary_scored.unresolved_captures == 0
    assert summary_scored.market_brier is not None
    assert summary_scored.jev_blind_brier is not None
    # Jev blind brier (0.04) vs Market brier (0.09): delta = -0.05
    assert pytest.approx(summary_scored.delta_jev_blind_vs_market, abs=1e-4) == -0.05
