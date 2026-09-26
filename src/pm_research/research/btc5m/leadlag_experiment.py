"""Frozen scientific specification for BTC 5-minute Lead-Lag Observational Experiment (v1).

Defines canonical immutable experimental parameters, feature sets, predeclared lags,
and deterministic specification hashing.

EXPERIMENT_ID: btc5m_leadlag_v1
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_ID: str = "btc5m_leadlag_v1"
EXPERIMENT_VERSION: str = "1.0.0"
EXPERIMENT_SPEC_HASH: str = "454b3752fb1d6ca229958c770f27ba9acf6a07dd0b8500e24c299becc86a1c33"

TARGET_PHYSICAL_ROUNDS: int = 500
SAMPLING_CADENCE_SEC: int = 1
PREDECLARED_LAGS_SEC: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30)

MAX_STALE_AGE_MS: int = 3000  # Feeds older than 3 seconds are flagged stale

# Frozen feature definitions
FROZEN_BINANCE_FEATURES: tuple[str, ...] = (
    "binance_best_bid",
    "binance_best_ask",
    "binance_mid_price",
    "binance_microprice",
    "binance_microprice_offset_bps",
    "binance_spread_bps",
    "binance_basis_bps",
    "binance_return_since_open_bps",
    "binance_return_1s_bps",
    "binance_return_2s_bps",
    "binance_return_3s_bps",
    "binance_return_5s_bps",
    "binance_return_10s_bps",
    "binance_return_30s_bps",
    "binance_return_60s_bps",
    "binance_taker_flow_1s",
    "binance_taker_flow_2s",
    "binance_taker_flow_3s",
    "binance_taker_flow_5s",
    "binance_taker_flow_10s",
    "binance_taker_flow_30s",
    "binance_taker_flow_60s",
    "binance_top1_depth_imbalance",
    "binance_top5_depth_imbalance",
    "binance_top20_depth_imbalance",
)

FROZEN_POLY_FEATURES: tuple[str, ...] = (
    "poly_best_bid",
    "poly_best_ask",
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


@dataclass(frozen=True)
class LeadLagSample:
    """Synchronized 1-second observation of Binance perpetual and Polymarket UP orderbook."""

    sample_id: str
    round_slug: str
    experiment_id: str
    experiment_spec_hash: str
    sample_target_ts_ms: int
    sample_actual_ts_ms: int
    local_monotonic_ns: int
    seconds_remaining: int

    # Polymarket state
    poly_source_ts_ms: int | None
    poly_recv_ts_ms: int
    poly_receipt_age_ms: int
    poly_source_age_ms: int | None
    poly_best_bid: float | None
    poly_best_ask: float | None
    poly_midpoint: float | None
    poly_spread: float | None
    poly_return_1s: float | None
    poly_return_2s: float | None
    poly_return_3s: float | None
    poly_return_5s: float | None
    poly_return_10s: float | None
    poly_return_30s: float | None
    poly_is_crossed: bool
    poly_is_valid: bool

    # Binance state
    binance_source_ts_ms: int | None
    binance_recv_ts_ms: int
    binance_receipt_age_ms: int
    binance_source_age_ms: int | None
    binance_best_bid: float | None
    binance_best_ask: float | None
    binance_mid_price: float | None
    binance_microprice: float | None
    binance_microprice_offset_bps: float | None
    binance_spread_bps: float | None
    binance_basis_bps: float | None
    binance_return_since_open_bps: float | None
    binance_return_1s_bps: float | None
    binance_return_2s_bps: float | None
    binance_return_3s_bps: float | None
    binance_return_5s_bps: float | None
    binance_return_10s_bps: float | None
    binance_return_30s_bps: float | None
    binance_return_60s_bps: float | None
    binance_taker_flow_1s: float | None
    binance_taker_flow_2s: float | None
    binance_taker_flow_3s: float | None
    binance_taker_flow_5s: float | None
    binance_taker_flow_10s: float | None
    binance_taker_flow_30s: float | None
    binance_taker_flow_60s: float | None
    binance_top1_depth_imbalance: float | None
    binance_top5_depth_imbalance: float | None
    binance_top20_depth_imbalance: float | None
    binance_is_valid: bool

    # Provenance & Quality
    source_to_receive_latency_ms: int | None
    inter_feed_receive_skew_ms: int
    is_stale: bool
    stale_reason: str | None
    is_valid: bool
    raw_json: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_experiment_spec_hash(spec_path: str | Path = "docs/BTC5M_LEADLAG_EXPERIMENT_SPEC.md") -> str:
    """Compute SHA256 checksum of the canonical experiment specification."""
    p = Path(spec_path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment spec not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()
