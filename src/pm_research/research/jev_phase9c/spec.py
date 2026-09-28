"""Phase 9C: Prospective Jev Decision Protocol Specifications.

Defines the three candidate typed tasks for prospective evaluation on btc5m_leadlag_v2:
1. Candidate Task A: poly_displacement_regime_30s_v2
2. Candidate Task B: cross_venue_lead_coherent_10s_v2
3. Candidate Task C: short_horizon_regime_60s_v2

STRICT PROTOCOL MANDATES:
- Pure statistical / probabilistic decision research.
- ZERO execution: positions=0, orders=0, fills=0, PnL=0, PaperBroker calls=0.
- Maximum remote LLM calls: <= 300 holdout-only calls (0 on train/calibration).
- Cost ceiling: <= $1.00 USD.
- Bonferroni multiple testing control: alpha = 0.05 / 3 ~= 0.0166667.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pm_research.research.jev_lab.contract import ABSTAIN_CHOICE

# Task IDs
TASK_A_ID: str = "poly_displacement_regime_30s_v2"
TASK_B_ID: str = "cross_venue_lead_coherent_10s_v2"
TASK_C_ID: str = "short_horizon_regime_60s_v2"

# Task A Choices
TASK_A_UP: str = "UP_DISPLACEMENT"
TASK_A_DOWN: str = "DOWN_DISPLACEMENT"
TASK_A_BENIGN: str = "BENIGN"
TASK_A_ACTIVE_CHOICES: tuple[str, ...] = (TASK_A_UP, TASK_A_DOWN, TASK_A_BENIGN)
TASK_A_ALL_CHOICES: tuple[str, ...] = (*TASK_A_ACTIVE_CHOICES, ABSTAIN_CHOICE)

# Task B Choices
TASK_B_COHERENT: str = "COHERENT"
TASK_B_NOT_COHERENT: str = "NOT_COHERENT"
TASK_B_ACTIVE_CHOICES: tuple[str, ...] = (TASK_B_COHERENT, TASK_B_NOT_COHERENT)
TASK_B_ALL_CHOICES: tuple[str, ...] = (*TASK_B_ACTIVE_CHOICES, ABSTAIN_CHOICE)

# Task C Choices
TASK_C_HIGH_VOL: str = "HIGH_VOLATILITY"
TASK_C_LOW_VOL: str = "LOW_VOLATILITY"
TASK_C_ACTIVE_CHOICES: tuple[str, ...] = (TASK_C_HIGH_VOL, TASK_C_LOW_VOL)
TASK_C_ALL_CHOICES: tuple[str, ...] = (*TASK_C_ACTIVE_CHOICES, ABSTAIN_CHOICE)

# Common High-Frequency Feature Fields available in btc5m_leadlag_v2
COMMON_V2_FEATURES: tuple[str, ...] = (
    "poly_midpoint",
    "poly_spread",
    "seconds_remaining",
    "poly_return_1s",
    "poly_return_2s",
    "poly_return_3s",
    "poly_return_5s",
    "poly_return_10s",
    "poly_return_30s",
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
    # Real Binance USD-M taker flow validated in Phase 8C.2
    "binance_taker_flow_1s",
    "binance_taker_flow_2s",
    "binance_taker_flow_3s",
    "binance_taker_flow_5s",
    "binance_taker_flow_10s",
    "binance_taker_flow_30s",
    "binance_taker_flow_60s",
)


def get_task_a_spec() -> dict[str, Any]:
    """Candidate Task A: Polymarket Displacement Regime over 30s."""
    spec = {
        "task_id": TASK_A_ID,
        "task_version": "2.0.0",
        "task_type": "MICROSTRUCTURE_REGIME_CLASSIFICATION",
        "description": "Predict whether Polymarket midpoint moves by at least the half-spread over next 30s.",
        "anchor_seconds_remaining": 120,
        "anchor_tolerance_seconds": 10,
        "target_horizon_seconds": 30,
        "active_choices": list(TASK_A_ACTIVE_CHOICES),
        "all_choices": list(TASK_A_ALL_CHOICES),
        "abstain_allowed": True,
        "label_formula": "Displacement = (q_(t+30s) - q_t) / (s_t / 2)",
        "label_criteria": {
            TASK_A_UP: "Displacement >= +1.0 (midpoint moves up by >= half-spread)",
            TASK_A_DOWN: "Displacement <= -1.0 (midpoint moves down by >= half-spread)",
            TASK_A_BENIGN: "|Displacement| < 1.0 (midpoint stays within half-spread)",
            ABSTAIN_CHOICE: "Abstain due to high ambiguity or noisy microstructure",
        },
        "feature_fields": list(COMMON_V2_FEATURES),
        "zero_execution_constraints": {
            "positions": 0,
            "orders": 0,
            "fills": 0,
            "pnl": 0.0,
            "paper_broker_calls": 0,
        },
    }
    canonical_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec["sha256"] = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return spec


def get_task_b_spec() -> dict[str, Any]:
    """Candidate Task B: Cross-Venue Lead Coherence over 10s."""
    spec = {
        "task_id": TASK_B_ID,
        "task_version": "2.0.0",
        "task_type": "CROSS_VENUE_EVENT_COHERENCE",
        "description": "Predict whether Polymarket responds coherently within 10s to a Binance dislocation >= 3 bps.",
        "trigger_condition": "First event between 180s and 60s remaining with |binance_5s_return| >= 3 bps",
        "trigger_min_seconds_remaining": 60,
        "trigger_max_seconds_remaining": 180,
        "dislocation_threshold_bps": 3.0,
        "reaction_horizon_seconds": 10,
        "active_choices": list(TASK_B_ACTIVE_CHOICES),
        "all_choices": list(TASK_B_ALL_CHOICES),
        "abstain_allowed": True,
        "label_formula": "COHERENT iff sign(delta_q_poly_10s) == sign(delta_p_binance_5s) and |delta_q_poly_10s| > 0",
        "label_criteria": {
            TASK_B_COHERENT: "Polymarket midpoint adjusts in same direction within 10s (|delta_q| > 0)",
            TASK_B_NOT_COHERENT: "Polymarket midpoint moves opposite or fails to move within 10s",
            ABSTAIN_CHOICE: "Abstain due to high ambiguity or noisy microstructure",
        },
        "feature_fields": list(COMMON_V2_FEATURES),
        "zero_execution_constraints": {
            "positions": 0,
            "orders": 0,
            "fills": 0,
            "pnl": 0.0,
            "paper_broker_calls": 0,
        },
    }
    canonical_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec["sha256"] = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return spec


def get_task_c_spec() -> dict[str, Any]:
    """Candidate Task C: Short-Horizon Realized Volatility Regime over 60s."""
    spec = {
        "task_id": TASK_C_ID,
        "task_version": "2.0.0",
        "task_type": "VOLATILITY_REGIME_CLASSIFICATION",
        "description": "Predict whether the next 60s will exhibit high or low realized volatility.",
        "anchor_seconds_remaining": 120,
        "anchor_tolerance_seconds": 10,
        "horizon_seconds": 60,
        "metric": "Synchronized 1-second Binance mid return standard deviation over 60s",
        "threshold": "Frozen empirical median of development split (rounds 1-300)",
        "active_choices": list(TASK_C_ACTIVE_CHOICES),
        "all_choices": list(TASK_C_ALL_CHOICES),
        "abstain_allowed": True,
        "label_criteria": {
            TASK_C_HIGH_VOL: "sigma_60s > Median_train(sigma_60s)",
            TASK_C_LOW_VOL: "sigma_60s <= Median_train(sigma_60s)",
            ABSTAIN_CHOICE: "Abstain due to high ambiguity or noisy microstructure",
        },
        "feature_fields": list(COMMON_V2_FEATURES),
        "zero_execution_constraints": {
            "positions": 0,
            "orders": 0,
            "fills": 0,
            "pnl": 0.0,
            "paper_broker_calls": 0,
        },
    }
    canonical_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    spec["sha256"] = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return spec
