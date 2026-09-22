"""Ablation condition state builders and Jev inference runner for BTC 5-minute markets.

4 Experimental Conditions:
- Condition A: BTC5M_REFERENCE_ONLY (Rule, anchor/priceToBeat, current ref, distance, returns)
- Condition B: BTC5M_REFERENCE_PLUS_PERP (A + Binance perp microstructure, taker flow, basis)
- Condition C: BTC5M_MARKET_AWARE (A + Polymarket book consensus, market_q)
- Condition D: BTC5M_FULL (All features combined)

Strict Research Invariant:
Only probability estimation is requested.
Zero trading recommendations, zero order sizing, zero live/paper execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BTC5mAblationForecast:
        """Construct from dictionary."""
        return cls(**data)


def build_ablation_state(
    snapshot: BTC5mFeatureSnapshot,
    condition: str,
) -> dict[str, Any]:
    """Construct the strictly filtered input state dictionary for the given ablation condition."""
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
        state["polymarket_order_book"] = {
            "market_consensus_probability": snapshot.market_q,
            "up_best_bid": snapshot.up_best_bid,
            "up_best_ask": snapshot.up_best_ask,
            "down_best_bid": snapshot.down_best_bid,
            "down_best_ask": snapshot.down_best_ask,
            "poly_spread": snapshot.poly_spread,
            "market_consensus_notice": (
                "The market consensus probability is an observed prediction-market price, "
                "not ground truth."
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
            "market_consensus_probability": snapshot.market_q,
            "up_best_bid": snapshot.up_best_bid,
            "up_best_ask": snapshot.up_best_ask,
            "down_best_bid": snapshot.down_best_bid,
            "down_best_ask": snapshot.down_best_ask,
            "poly_spread": snapshot.poly_spread,
            "market_consensus_notice": (
                "The market consensus probability is an observed prediction-market price, "
                "not ground truth."
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
        bypass_cache: bool = False,
    ) -> BTC5mAblationForecast:
        """Run inference for a single condition on a snapshot."""
        state = build_ablation_state(snapshot, condition)
        payload = build_btc5m_request_payload(state)
        req_hash = self.client.compute_request_hash(payload)

        t0 = time.monotonic()
        resp_json, from_cache, req_hash = self.client.query_decision_payload(
            payload=payload,
            bypass_cache=bypass_cache,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Parse response
        answers = resp_json.get("answers", {})
        up_ans = answers.get("will_resolve_up", {})
        p_up_raw = float(up_ans.get("noul", 0.5))
        p_up = round(max(0.0, min(1.0, p_up_raw)), 4)
        p_down = round(1.0 - p_up, 4)

        choice_ans = answers.get("outcome_choice", {})
        choice = str(choice_ans.get("choice", "UP" if p_up >= 0.5 else "DOWN")).upper()
        confidence = choice_ans.get("confidence")
        if confidence is not None:
            try:
                confidence = round(float(confidence), 4)
            except (ValueError, TypeError):
                confidence = None

        usage = resp_json.get("usage", {})
        input_tok = int(usage.get("input_tokens", 0))
        output_tok = int(usage.get("output_tokens", 0))

        raw_bytes = json.dumps(resp_json, sort_keys=True).encode("utf-8")
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()

        now_utc = datetime.now(timezone.utc).isoformat()
        forecast_id = f"btc5m_fc_{snapshot.snapshot_id}_{condition}"

        return BTC5mAblationForecast(
            forecast_id=forecast_id,
            snapshot_id=snapshot.snapshot_id,
            round_slug=snapshot.round_slug,
            target_horizon_sec=snapshot.target_horizon_sec,
            condition=condition,
            model_id=JEV_MODEL_PIN,
            request_hash=req_hash,
            captured_at_ms=snapshot.captured_at_ms,
            market_q=snapshot.market_q,
            jev_up_prob=p_up,
            jev_down_prob=p_down,
            jev_choice=choice,
            confidence=confidence,
            input_tokens=input_tok,
            output_tokens=output_tok,
            latency_ms=latency_ms,
            from_cache=from_cache,
            raw_response_hash=raw_hash,
            created_at_utc=now_utc,
        )

    def run_all_conditions(
        self,
        snapshot: BTC5mFeatureSnapshot,
        conditions: Sequence[str] = ALL_BTC5M_CONDITIONS,
        bypass_cache: bool = False,
    ) -> dict[str, BTC5mAblationForecast]:
        """Execute all specified ablation conditions against the single frozen snapshot."""
        results: dict[str, BTC5mAblationForecast] = {}
        for cond in conditions:
            try:
                fc = self.run_single_condition(
                    snapshot=snapshot,
                    condition=cond,
                    bypass_cache=bypass_cache,
                )
                results[cond] = fc
            except Exception as e:
                logger.error(
                    f"Ablation condition {cond} failed for snapshot {snapshot.snapshot_id}: {e}"
                )
        return results
