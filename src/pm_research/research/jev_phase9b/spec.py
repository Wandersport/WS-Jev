"""Specification for Phase 9B: Meaningful Polymarket Move Setup-Gating Benchmark.

SCIENTIFIC HYPOTHESIS ONLY:
Investigates whether TypeSafe Jev (typesafe/jev-1.13) can identify market states
where a meaningful future Polymarket midpoint move is more likely, compared to
simple transparent baselines.

TASK_ID: meaningful_poly_move_30s_v1
Choices: MEANINGFUL_MOVE, QUIET, ABSTAIN
Target Horizon: 30 seconds
Primary Anchor: ~120s remaining in physical BTC-5m round.
"""

from __future__ import annotations

import hashlib
import json

from pm_research.research.jev_lab.contract import (
    ABSTAIN_CHOICE,
    STATUS_DEV_HYPOTHESIS,
    DecisionTask,
)

TASK_ID: str = "meaningful_poly_move_30s_v1"
TASK_VERSION: str = "1.0.0"
TARGET_HORIZON: str = "30s"
ANCHOR_SECONDS_REMAINING: int = 120
ANCHOR_TOLERANCE_SECONDS: int = 10
FUTURE_HORIZON_MS: int = 30_000
FUTURE_TOLERANCE_MS: int = 3_000

CHOICE_MEANINGFUL_MOVE: str = "MEANINGFUL_MOVE"
CHOICE_QUIET: str = "QUIET"
ALLOWED_ACTIVE_CHOICES: tuple[str, ...] = (CHOICE_MEANINGFUL_MOVE, CHOICE_QUIET)
ALL_PERMISSIBLE_CHOICES: tuple[str, ...] = (
    CHOICE_MEANINGFUL_MOVE,
    CHOICE_QUIET,
    ABSTAIN_CHOICE,
)

PROMPT_INSTRUCTIONS: str = (
    "Given only the current market state, classify whether the native "
    "Polymarket UP midpoint will make an absolute move at least as large as "
    "its current spread within the next 30 seconds."
)

CRITERIA: dict[str, str] = {
    CHOICE_MEANINGFUL_MOVE: (
        "The absolute change in Polymarket UP midpoint over the next 30 seconds "
        "is greater than or equal to the current bid-ask spread: |q_(t+30s) - q_t| >= s_t."
    ),
    CHOICE_QUIET: (
        "The absolute change in Polymarket UP midpoint over the next 30 seconds "
        "is strictly less than the current bid-ask spread: |q_(t+30s) - q_t| < s_t."
    ),
    ABSTAIN_CHOICE: (
        "Abstain from choosing an active label due to high uncertainty, "
        "ambiguity, or noisy microstructure."
    ),
}

FEATURE_FIELDS_POLY: tuple[str, ...] = (
    "poly_midpoint",
    "poly_spread",
    "seconds_remaining",
    "poly_return_1s",
    "poly_return_2s",
    "poly_return_3s",
    "poly_return_5s",
    "poly_return_10s",
    "poly_return_30s",
)

FEATURE_FIELDS_BINANCE: tuple[str, ...] = (
    "binance_mid_price",
    "binance_spread_bps",
    "binance_microprice_offset_bps",
    "binance_top1_depth_imbalance",
    "binance_top5_depth_imbalance",
    "binance_top20_depth_imbalance",
    "binance_return_since_open_bps",
    "binance_return_1s_bps",
    "binance_return_2s_bps",
    "binance_return_3s_bps",
    "binance_return_5s_bps",
    "binance_return_10s_bps",
    "binance_return_30s_bps",
    "binance_return_60s_bps",
)

# Explicitly excluded fields (v1 measurement defect or target leakage)
EXCLUDED_FIELDS: tuple[str, ...] = (
    "binance_taker_flow_1s",
    "binance_taker_flow_2s",
    "binance_taker_flow_3s",
    "binance_taker_flow_5s",
    "binance_taker_flow_10s",
    "binance_taker_flow_30s",
    "binance_taker_flow_60s",
)


def compute_task_spec_hash() -> str:
    """Compute deterministic SHA-256 fingerprint for the frozen Phase 9B task specification."""
    spec_dict = {
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "target_horizon": TARGET_HORIZON,
        "anchor_seconds_remaining": ANCHOR_SECONDS_REMAINING,
        "anchor_tolerance_seconds": ANCHOR_TOLERANCE_SECONDS,
        "future_horizon_ms": FUTURE_HORIZON_MS,
        "future_tolerance_ms": FUTURE_TOLERANCE_MS,
        "choices": list(ALLOWED_ACTIVE_CHOICES),
        "prompt_instructions": PROMPT_INSTRUCTIONS,
        "criteria": CRITERIA,
        "poly_features": list(FEATURE_FIELDS_POLY),
        "binance_features": list(FEATURE_FIELDS_BINANCE),
        "excluded_features": list(EXCLUDED_FIELDS),
    }
    canonical_json = json.dumps(spec_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_phase9b_decision_task() -> DecisionTask:
    """Instantiate the canonical DecisionTask contract for Phase 9B."""
    return DecisionTask(
        task_id=TASK_ID,
        task_version=TASK_VERSION,
        task_type="SETUP_GATING",
        question=PROMPT_INSTRUCTIONS,
        allowed_choices=ALLOWED_ACTIVE_CHOICES,
        optional_abstain=True,
        state_schema_version="jev-phase9b-microstructure-v1",
        label_definition=(
            "Objective construction: at t (~120s remaining), q_t = poly_midpoint, s_t = poly_spread. "
            "At t+30s, q_(t+30s) = poly_midpoint. MEANINGFUL_MOVE if |q_(t+30s) - q_t| >= s_t; "
            "QUIET otherwise."
        ),
        information_cutoff="t_observation",
        target_horizon=TARGET_HORIZON,
        status=STATUS_DEV_HYPOTHESIS,
        description="Retrospective setup-gating development benchmark on btc5m_leadlag_v1 data.",
    )
