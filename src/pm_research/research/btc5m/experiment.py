"""Frozen scientific experiment specification for BTC 5-minute Jev prospective ablation (v1).

Defines canonical immutable experimental parameters, feature schema, decision schema,
timing tolerances, skip policies, and deterministic specification hashing.

EXPERIMENT_ID: btc5m_jev_ablation_v1
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

EXPERIMENT_ID: str = "btc5m_jev_ablation_v1"
EXPERIMENT_VERSION: str = "1.0.0"
MODEL_PIN: str = "typesafe/jev-1.13"
STANDARD_HORIZONS: tuple[int, ...] = (240, 180, 120, 60, 30)
PRIMARY_MARKET_BASELINE: str = "NATIVE_UP_MIDPOINT"

COND_A_REF_ONLY: str = "BTC5M_REFERENCE_ONLY"
COND_B_REF_PERP: str = "BTC5M_REFERENCE_PLUS_PERP"
COND_C_MARKET_AWARE: str = "BTC5M_MARKET_AWARE"
COND_D_FULL: str = "BTC5M_FULL"

FROZEN_CONDITIONS: tuple[str, ...] = (
    COND_A_REF_ONLY,
    COND_B_REF_PERP,
    COND_C_MARKET_AWARE,
    COND_D_FULL,
)

# Canonical frozen decision schema
FROZEN_DECISION_QUESTIONS: dict[str, Any] = {
    "will_resolve_up": {
        "type": "noul",
        "instructions": (
            "Will this BTC 5-minute prediction market resolve UP under the official resolution rule? "
            "Evaluate the probability that the official reference price at round settlement is >= priceToBeat. "
            "Focus solely on objective, calibrated probability estimation without external bias."
        ),
        "criteria": {
            "true": "The official Chainlink settlement price at round close is >= priceToBeat, so the market resolves UP.",
            "false": "The official Chainlink settlement price at round close is < priceToBeat, so the market resolves DOWN.",
        },
    },
    "outcome_choice": {
        "type": "choice",
        "instructions": "Which discrete outcome will occur under the resolution criteria?",
        "criteria": {
            "UP": "The market resolves to UP.",
            "DOWN": "The market resolves to DOWN.",
        },
    },
}

# Frozen timing tolerances
MAX_ACCEPTABLE_TIMING_DRIFT_MS: int = 3000
BINANCE_OPEN_TIMING_TOLERANCE_MS: int = 2500
METADATA_DURATION_TOLERANCE_SEC: float = 2.0
MAX_REFERENCE_DATA_AGE_MS: int = 30000

# Canonical Experiment Specification Dictionary
CANONICAL_SPEC: dict[str, Any] = {
    "experiment_id": EXPERIMENT_ID,
    "version": EXPERIMENT_VERSION,
    "model_id": MODEL_PIN,
    "horizons_sec": list(STANDARD_HORIZONS),
    "primary_market_baseline": PRIMARY_MARKET_BASELINE,
    "conditions": list(FROZEN_CONDITIONS),
    "market_eligibility": {
        "slug_pattern": r"^btc-updown-5m-\d+$",
        "duration_seconds": 300,
        "duration_tolerance_seconds": METADATA_DURATION_TOLERANCE_SEC,
        "outcomes": ["Up", "Down"],
        "settlement_source_domain": "data.chain.link",
        "settlement_source_stream": "btc-usd-twap-60s-streams",
    },
    "reference_feed": {
        "source": "chainlink-twap-60s",
        "topic": "crypto_prices_twap_sixty",
        "anchor_hierarchy": [
            "gamma_metadata_priceToBeat",
            "exact_boundary_tick_timestamp_ms",
        ],
        "max_age_ms": MAX_REFERENCE_DATA_AGE_MS,
    },
    "market_baseline_policy": {
        "primary": "NATIVE_UP_MIDPOINT",
        "synthetic_complement_as_primary": False,
        "cross_outcome_implied_diagnostic": True,
        "require_two_sided_book": True,
    },
    "binance_provenance": {
        "symbol": "BTCUSDT",
        "contract": "USD-M Perpetual",
        "boundary_tolerance_ms": BINANCE_OPEN_TIMING_TOLERANCE_MS,
        "require_boundary_observation_for_open_return": True,
    },
    "timing_and_leakage": {
        "authoritative_freeze_point": "capture_completed_at_ms",
        "max_timing_drift_ms": MAX_ACCEPTABLE_TIMING_DRIFT_MS,
        "reject_post_horizon_source_timestamps": True,
        "reject_late_model_responses": True,
    },
    "decision_schema": FROZEN_DECISION_QUESTIONS,
    "scoring_methodology": {
        "primary_metric": "brier_score",
        "secondary_metrics": ["log_loss", "forecast_bias", "directional_accuracy"],
        "paired_deltas": [
            "REF_ONLY - MARKET",
            "REF_PLUS_PERP - MARKET",
            "MARKET_AWARE - MARKET",
            "FULL - MARKET",
            "REF_PLUS_PERP - REF_ONLY",
            "FULL - MARKET_AWARE",
            "MARKET_AWARE - REF_ONLY",
        ],
        "uncertainty": "round_clustered_bootstrap",
        "bootstrap_resamples": 10000,
        "confidence_level": 0.95,
    },
    "safety_contract": {
        "live_trading": False,
        "wallets_or_keys": False,
        "unauthenticated_market_data_only": True,
        "trade_proposals_generated": 0,
        "paper_orders_generated": 0,
    },
}


def compute_experiment_spec_hash(spec: dict[str, Any] | None = None) -> str:
    """Compute the deterministic SHA-256 hash of the canonical experiment specification."""
    target = spec or CANONICAL_SPEC
    canonical_json = json.dumps(target, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


EXPERIMENT_SPEC_HASH: str = compute_experiment_spec_hash()


def get_experiment_spec() -> dict[str, Any]:
    """Return a deep copy of the canonical experiment specification."""
    return json.loads(json.dumps(CANONICAL_SPEC))


def verify_phase6_compatibility(
    round_slug: str,
    snapshots: list[Any],
    forecasts: list[Any],
) -> tuple[bool, str | None]:
    """Verify whether a Phase 6.1 observation conforms to the frozen v1 specification."""
    if not round_slug.startswith("btc-updown-5m-"):
        return False, "Non-conforming round slug format"

    for s in snapshots:
        if s.is_valid:
            # Check horizons
            if s.target_horizon_sec not in STANDARD_HORIZONS:
                return False, f"Non-conforming horizon {s.target_horizon_sec}"
            # Check market_q_primary
            if s.market_q_primary is None:
                return False, "Valid snapshot has None market_q_primary"

    for f in forecasts:
        if f.is_valid:
            if f.condition not in FROZEN_CONDITIONS:
                return False, f"Non-conforming condition {f.condition}"
            if not f.model_id.startswith(MODEL_PIN):
                return False, f"Model mismatch {f.model_id}"

    return True, None
