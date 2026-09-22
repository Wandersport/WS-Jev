"""Ablation condition state builders and Jev inference runner for BTC 5-minute markets.

4 Experimental Conditions:
- Condition A: BTC5M_REFERENCE_ONLY (Rule, anchor/priceToBeat, current ref, distance, returns)
- Condition B: BTC5M_REFERENCE_PLUS_PERP (A + Binance perp microstructure, taker flow, basis)
- Condition C: BTC5M_MARKET_AWARE (A + observed Polymarket native book and consensus)
- Condition D: BTC5M_FULL (All features combined)

Strict Research Invariants (Requirements A1, A12, A13, A14):
- Only probability estimation is requested (zero trading/order recommendations)
- Condition C/D explicitly separates observed native quotes from cross-outcome implied diagnostic
- Jev response parsing strictly FAILS CLOSED (never defaults to 0.5, never clamps, never invents choice)
- Pre-close response requirement: response_received_at_ms >= round_end_ms marks INVALID_LATE_MODEL_RESPONSE
- Four-condition fairness: immutable feature snapshot hash tracked, execution sequence logged
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from pm_research.research.btc5m.snapshot import BTC5mFeatureSnapshot
from pm_research.research.jev_openrouter import JEV_MODEL_PIN, JevOpenRouterClient

logger = logging.getLogger(__name__)

COND_A_REF_ONLY: str = "BTC5M_REFERENCE_ONLY"
COND_B_REF_PERP: str = "BTC5M_REFERENCE_PLUS_PERP"
COND_C_MARKET_AWARE: str = "BTC5M_MARKET_AWARE"
COND_D_FULL: str = "BTC5M_FULL"

ALL_BTC5M_CONDITIONS: tuple[str, ...] = (
    COND_A_REF_ONLY,
    COND_B_REF_PERP,
    COND_C_MARKET_AWARE,
    COND_D_FULL,
)


class JevParseError(Exception):
    """Raised when Jev API response is malformed or invalid."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class BTC5mAblationForecast:
    """Immutable probabilistic forecast for a specific ablation condition on a frozen snapshot."""

    forecast_id: str
    snapshot_id: str
    round_slug: str
    target_horizon_sec: int
    condition: str
    model_id: str
    request_hash: str
    captured_at_ms: int
    market_q: float | None
    jev_up_prob: float
    jev_down_prob: float
    jev_choice: str  # "UP" or "DOWN"
    confidence: float | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    from_cache: bool
    raw_response_hash: str
    created_at_utc: str

    # Timing & integrity metadata (Requirements A13, A14)
    request_started_at_utc: str | None = None
    response_received_at_utc: str | None = None
    response_received_at_ms: int | None = None
    round_end_ms: int | None = None
    snapshot_hash: str | None = None
    request_order: int = 0
    is_valid: bool = True
    rejection_reason: str | None = None
    cost: float | None = None  # Actual reported OpenRouter cost (usage.cost)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BTC5mAblationForecast:
        """Construct from dictionary."""
        return cls(**data)


def parse_jev_response_strict(
    resp_json: dict[str, Any],
    pinned_model: str = JEV_MODEL_PIN,
) -> tuple[float, float, str, float | None]:
    """Strictly parse Jev decision response.

    Invariant A12: Fail closed.
    - answers exists
    - will_resolve_up exists
    - type == 'noul'
    - noul is numeric, not NaN, not Inf, 0.0 <= noul <= 1.0
    - outcome_choice exists and choice in ('UP', 'DOWN')
    - Reject if model mismatch
    """
    if not isinstance(resp_json, dict):
        raise JevParseError("RESP_NOT_DICT")

    # Model pinning check
    model_ret = str(resp_json.get("model", "")).strip()
    if model_ret and (pinned_model not in model_ret and model_ret not in pinned_model):
        raise JevParseError(f"MODEL_MISMATCH_{model_ret}")

    answers = resp_json.get("answers")
    if not isinstance(answers, dict):
        raise JevParseError("MISSING_ANSWERS")

    up_ans = answers.get("will_resolve_up")
    if not isinstance(up_ans, dict):
        raise JevParseError("MISSING_WILL_RESOLVE_UP")

    if up_ans.get("type") != "noul":
        raise JevParseError(f"INVALID_ANSWER_TYPE_{up_ans.get('type')}")

    raw_p = up_ans.get("noul")
    if raw_p is None or not isinstance(raw_p, (int, float)):
        raise JevParseError("NOUL_NOT_NUMERIC")

    if math.isnan(raw_p) or math.isinf(raw_p):
        raise JevParseError("NOUL_NAN_OR_INF")

    if raw_p < 0.0 or raw_p > 1.0:
        raise JevParseError(f"NOUL_OUT_OF_BOUNDS_{raw_p}")

    p_up = round(float(raw_p), 4)
    p_down = round(1.0 - p_up, 4)

    choice_ans = answers.get("outcome_choice")
    if not isinstance(choice_ans, dict):
        raise JevParseError("MISSING_OUTCOME_CHOICE")

    raw_choice = choice_ans.get("choice")
    if raw_choice is None:
        raise JevParseError("MISSING_CHOICE_VALUE")

    choice = str(raw_choice).strip().upper()
    if choice not in ("UP", "DOWN"):
        raise JevParseError(f"INVALID_CHOICE_{choice}")

    conf = choice_ans.get("confidence")
    confidence: float | None = None
    if conf is not None:
        try:
            cf = float(conf)
            if 0.0 <= cf <= 1.0 and not math.isnan(cf) and not math.isinf(cf):
                confidence = round(cf, 4)
        except (ValueError, TypeError):
            pass

    return p_up, p_down, choice, confidence


def build_ablation_state(
    snapshot: BTC5mFeatureSnapshot,
    condition: str,
) -> dict[str, Any]:
    """Construct the strictly filtered input state dictionary for the given ablation condition.

    Requirement A1: Distinguish observed native quotes from cross-outcome implied diagnostic.
    """
    if condition not in ALL_BTC5M_CONDITIONS:
        raise ValueError(f"Unknown ablation condition: {condition}")

    # Base reference state (Condition A)
    base_state: dict[str, Any] = {
        "market_type": "BTC 5-Minute Up/Down",
        "round_slug": snapshot.round_slug,
        "seconds_remaining": snapshot.seconds_remaining,
        "settlement_feed": snapshot.reference_source,
        "price_to_beat": snapshot.price_to_beat,
        "price_to_beat_source": snapshot.price_to_beat_source,
        "current_reference_price": snapshot.current_reference_price,
        "ref_distance_to_beat_bps": snapshot.ref_distance_to_beat_bps,
        "ref_returns_bps": {
            "10s": snapshot.ref_return_10s_bps,
            "30s": snapshot.ref_return_30s_bps,
            "60s": snapshot.ref_return_60s_bps,
        },
        "resolution_rule": (
            "Market resolves UP if official Chainlink reference price at round end >= priceToBeat, "
            "otherwise DOWN."
        ),
        "ablation_condition": condition,
    }

    if condition == COND_A_REF_ONLY:
        return base_state

    if condition == COND_B_REF_PERP:
        state = dict(base_state)
        state["binance_perp_microstructure"] = {
            "mid_price": snapshot.binance_perp_mid,
            "microprice_offset_bps": snapshot.binance_microprice_offset_bps,
            "top5_depth_imbalance": snapshot.binance_top5_depth_imbalance,
            "top20_depth_imbalance": snapshot.binance_top20_depth_imbalance,
            "spread_bps": snapshot.binance_spread_bps,
            "taker_flow_imbalance": {
                "10s": snapshot.binance_taker_flow_10s_imbalance,
                "30s": snapshot.binance_taker_flow_30s_imbalance,
                "60s": snapshot.binance_taker_flow_60s_imbalance,
            },
            "perp_returns_bps": {
                "10s": snapshot.binance_return_10s_bps,
                "30s": snapshot.binance_return_30s_bps,
                "60s": snapshot.binance_return_60s_bps,
                "since_open": snapshot.binance_return_since_open_bps,
            },
            "basis_vs_ref_bps": snapshot.binance_basis_bps,
        }
        return state

    if condition == COND_C_MARKET_AWARE:
        state = dict(base_state)
        # Requirement A1: Supply observed native book and explicit distinction from implied
        state["polymarket_order_book"] = {
            "observed_native_book": {
                "up_best_bid": snapshot.native_up_bid,
                "up_best_ask": snapshot.native_up_ask,
                "up_midpoint": snapshot.native_up_mid,
                "down_best_bid": snapshot.native_down_bid,
                "down_best_ask": snapshot.native_down_ask,
                "down_midpoint": snapshot.native_down_mid,
            },
            "market_consensus_probability": snapshot.market_q_primary,
            "cross_outcome_implied_probability": snapshot.market_q_implied_cross_outcome,
            "poly_spread": snapshot.poly_spread,
            "provenance_notice": (
                "The market consensus probability is the directly observed native UP midpoint from "
                "the Polymarket prediction market. The implied probability is derived via cross-outcome "
                "binary complementarity. Neither is ground truth."
            ),
        }
        return state

    if condition == COND_D_FULL:
        state = dict(base_state)
        state["binance_perp_microstructure"] = {
            "mid_price": snapshot.binance_perp_mid,
            "microprice_offset_bps": snapshot.binance_microprice_offset_bps,
            "top5_depth_imbalance": snapshot.binance_top5_depth_imbalance,
            "top20_depth_imbalance": snapshot.binance_top20_depth_imbalance,
            "spread_bps": snapshot.binance_spread_bps,
            "taker_flow_imbalance": {
                "10s": snapshot.binance_taker_flow_10s_imbalance,
                "30s": snapshot.binance_taker_flow_30s_imbalance,
                "60s": snapshot.binance_taker_flow_60s_imbalance,
            },
            "perp_returns_bps": {
                "10s": snapshot.binance_return_10s_bps,
                "30s": snapshot.binance_return_30s_bps,
                "60s": snapshot.binance_return_60s_bps,
                "since_open": snapshot.binance_return_since_open_bps,
            },
            "basis_vs_ref_bps": snapshot.binance_basis_bps,
        }
        state["polymarket_order_book"] = {
            "observed_native_book": {
                "up_best_bid": snapshot.native_up_bid,
                "up_best_ask": snapshot.native_up_ask,
                "up_midpoint": snapshot.native_up_mid,
                "down_best_bid": snapshot.native_down_bid,
                "down_best_ask": snapshot.native_down_ask,
                "down_midpoint": snapshot.native_down_mid,
            },
            "market_consensus_probability": snapshot.market_q_primary,
            "cross_outcome_implied_probability": snapshot.market_q_implied_cross_outcome,
            "poly_spread": snapshot.poly_spread,
            "provenance_notice": (
                "The market consensus probability is the directly observed native UP midpoint from "
                "the Polymarket prediction market. The implied probability is derived via cross-outcome "
                "binary complementarity. Neither is ground truth."
            ),
        }
        return state

    raise ValueError(f"Unhandled condition: {condition}")


def build_btc5m_request_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble structured OpenRouter /api/alpha/decisions request schema for BTC 5m."""
    return {
        "model": JEV_MODEL_PIN,
        "state": state,
        "questions": {
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
        },
    }


class BTC5mAblationRunner:
    """Executes the 4-condition prospective forecasting ablation on frozen snapshots."""

    def __init__(self, client: JevOpenRouterClient | None = None) -> None:
        self.client = client or JevOpenRouterClient()

    def run_single_condition(
        self,
        snapshot: BTC5mFeatureSnapshot,
        condition: str,
        request_order: int = 1,
        bypass_cache: bool = False,
    ) -> BTC5mAblationForecast:
        """Run inference for a single condition on a snapshot.

        Requirements A12, A13, A14:
        - Fails closed on malformed output
        - Excludes late responses received after round close
        - Tracks snapshot hash and execution sequence
        """
        state = build_ablation_state(snapshot, condition)
        payload = build_btc5m_request_payload(state)
        req_hash = self.client.compute_request_hash(payload)

        # Snapshot hash for four-condition fairness tracking
        snap_hash = hashlib.sha256(snapshot.to_json().encode("utf-8")).hexdigest()

        req_started_utc = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        try:
            resp_json, from_cache, req_hash = self.client.query_decision_payload(
                payload=payload,
                bypass_cache=bypass_cache,
            )
            raw_hash = hashlib.sha256(json.dumps(resp_json, sort_keys=True).encode("utf-8")).hexdigest()
        except Exception as e:
            latency_ms = int((time.monotonic() - t0) * 1000)
            req_finished_utc = datetime.now(timezone.utc).isoformat()
            return BTC5mAblationForecast(
                forecast_id=f"{snapshot.snapshot_id}_{condition}",
                snapshot_id=snapshot.snapshot_id,
                round_slug=snapshot.round_slug,
                target_horizon_sec=snapshot.target_horizon_sec,
                condition=condition,
                model_id=JEV_MODEL_PIN,
                request_hash=req_hash,
                captured_at_ms=snapshot.captured_at_ms,
                market_q=snapshot.market_q_primary,
                jev_up_prob=-1.0,
                jev_down_prob=-1.0,
                jev_choice="UNKNOWN",
                confidence=None,
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
                from_cache=False,
                raw_response_hash="",
                created_at_utc=req_finished_utc,
                request_started_at_utc=req_started_utc,
                response_received_at_utc=req_finished_utc,
                response_received_at_ms=int(time.time() * 1000),
                round_end_ms=snapshot.round_end_epoch * 1000,
                snapshot_hash=snap_hash,
                request_order=request_order,
                is_valid=False,
                rejection_reason=f"HTTP_OR_TRANSPORT_ERROR: {e}",
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        req_finished_utc = datetime.now(timezone.utc).isoformat()
        resp_recv_ms = int(time.time() * 1000)
        round_end_ms = snapshot.round_end_epoch * 1000

        # Requirement A12: Parse strictly and fail closed
        is_valid = True
        rejection_reason = None
        try:
            p_up, p_down, choice, confidence = parse_jev_response_strict(resp_json, JEV_MODEL_PIN)
        except JevParseError as e:
            is_valid = False
            rejection_reason = f"PARSE_ERROR: {e.reason}"
            p_up = -1.0
            p_down = -1.0
            choice = "UNKNOWN"
            confidence = None

        # Requirement A13: Pre-close response requirement
        if is_valid and resp_recv_ms >= round_end_ms:
            is_valid = False
            rejection_reason = "INVALID_LATE_MODEL_RESPONSE"

        # Token usage & reported cost
        usage = resp_json.get("usage", {})
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens", 0))
        reported_cost = float(usage["cost"]) if usage.get("cost") is not None else None

        forecast_id = f"{snapshot.snapshot_id}_{condition}"

        return BTC5mAblationForecast(
            forecast_id=forecast_id,
            snapshot_id=snapshot.snapshot_id,
            round_slug=snapshot.round_slug,
            target_horizon_sec=snapshot.target_horizon_sec,
            condition=condition,
            model_id=JEV_MODEL_PIN,
            request_hash=req_hash,
            captured_at_ms=snapshot.captured_at_ms,
            market_q=snapshot.market_q_primary,
            jev_up_prob=p_up,
            jev_down_prob=p_down,
            jev_choice=choice,
            confidence=confidence,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            from_cache=from_cache,
            raw_response_hash=raw_hash,
            created_at_utc=req_finished_utc,
            request_started_at_utc=req_started_utc,
            response_received_at_utc=req_finished_utc,
            response_received_at_ms=resp_recv_ms,
            round_end_ms=round_end_ms,
            snapshot_hash=snap_hash,
            request_order=request_order,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
            cost=reported_cost,
        )

    def run_all_conditions(
        self,
        snapshot: BTC5mFeatureSnapshot,
        conditions: Sequence[str] = ALL_BTC5M_CONDITIONS,
        bypass_cache: bool = False,
    ) -> dict[str, BTC5mAblationForecast]:
        """Run all 4 ablation conditions sequentially on the same frozen snapshot."""
        results: dict[str, BTC5mAblationForecast] = {}
        for order, cond in enumerate(conditions, start=1):
            fc = self.run_single_condition(
                snapshot=snapshot,
                condition=cond,
                request_order=order,
                bypass_cache=bypass_cache,
            )
            results[cond] = fc
        return results
