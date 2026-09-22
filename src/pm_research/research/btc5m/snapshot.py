"""Immutable point-in-time snapshot for BTC 5-minute prospective forecasting.

Captures all input data at a frozen instant before round settlement:
- Chainlink reference stream (anchor, priceToBeat, current, distance, returns)
- Polymarket CLOB state (UP/DOWN best bid/ask, market_q, spread)
- Binance USD-M Perpetual microstructure (mid, microprice, depth imbalance, taker flow, basis)
- Timing metadata (target horizon, captured timestamp, deviation from schedule)
- Quality control and skip reason tracking
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from pm_research.research.btc5m.binance_feed import BinancePerpFeatures
from pm_research.research.btc5m.poly_book import PolymarketMarketState
from pm_research.research.btc5m.reference_feed import ReferenceFeatures

# Standardized forecasting horizons (seconds remaining before round close)
STANDARD_HORIZONS_SEC: tuple[int, ...] = (240, 180, 120, 60, 30)
MAX_ACCEPTABLE_TIMING_DRIFT_MS: int = 3500

# Standardized skip/filter reasons
SKIP_NO_EXACT_ANCHOR: str = "SKIP_NO_EXACT_ANCHOR"
SKIP_STALE_REFERENCE: str = "SKIP_STALE_REFERENCE"
SKIP_CROSSED_BOOK: str = "SKIP_CROSSED_BOOK"
SKIP_EXCESSIVE_TIMING_DRIFT: str = "SKIP_EXCESSIVE_TIMING_DRIFT"
SKIP_INVALID_PERP: str = "SKIP_INVALID_PERP"
SKIP_MISSING_MARKET_Q: str = "SKIP_MISSING_MARKET_Q"


@dataclass(frozen=True)
class BTC5mFeatureSnapshot:
    """Frozen point-in-time observation of all input sources for a single forecast horizon."""

    snapshot_id: str
    round_slug: str
    round_start_epoch: int
    round_end_epoch: int
    target_horizon_sec: int
    captured_at_ms: int
    target_scheduled_ms: int
    timing_deviation_ms: int
    seconds_remaining: float

    # Reference features
    reference_source: str
    price_to_beat: float | None
    price_to_beat_source: str
    current_reference_price: float
    ref_distance_to_beat_bps: float | None
    ref_return_10s_bps: float | None
    ref_return_30s_bps: float | None
    ref_return_60s_bps: float | None
    ref_data_age_ms: int

    # Polymarket book consensus
    market_q: float | None
    up_best_bid: float | None
    up_best_ask: float | None
    down_best_bid: float | None
    down_best_ask: float | None
    poly_spread: float | None
    poly_data_age_ms: int
    poly_book_valid: bool

    # Binance Perpetual features
    binance_perp_mid: float | None
    binance_microprice: float | None
    binance_microprice_offset_bps: float | None
    binance_top5_depth_imbalance: float | None
    binance_top20_depth_imbalance: float | None
    binance_spread_bps: float | None
    binance_taker_flow_10s_imbalance: float | None
    binance_taker_flow_30s_imbalance: float | None
    binance_taker_flow_60s_imbalance: float | None
    binance_taker_buy_qty_60s: float | None
    binance_taker_sell_qty_60s: float | None
    binance_return_10s_bps: float | None
    binance_return_30s_bps: float | None
    binance_return_60s_bps: float | None
    binance_return_since_open_bps: float | None
    binance_basis_bps: float | None
    binance_data_age_ms: int
    binance_valid: bool

    # Quality control
    is_valid: bool
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BTC5mFeatureSnapshot:
        """Construct from dictionary."""
        return cls(**data)


def build_feature_snapshot(
    round_slug: str,
    round_start_epoch: int,
    round_end_epoch: int,
    target_horizon_sec: int,
    captured_at_ms: int,
    ref_features: ReferenceFeatures,
    poly_state: PolymarketMarketState,
    binance_features: BinancePerpFeatures | None,
) -> BTC5mFeatureSnapshot:
    """Combine individual subsystem feeds into a validated, frozen BTC5mFeatureSnapshot."""
    target_scheduled_ms = (round_end_epoch - target_horizon_sec) * 1000
    timing_dev_ms = captured_at_ms - target_scheduled_ms
    seconds_remaining = max(0.0, round_end_epoch - (captured_at_ms / 1000.0))

    # Evaluate validation conditions
    skip_reason: str | None = None

    if ref_features.price_to_beat is None or ref_features.price_to_beat <= 0:
        skip_reason = SKIP_NO_EXACT_ANCHOR
    elif ref_features.data_age_ms > 30_000:
        skip_reason = SKIP_STALE_REFERENCE
    elif not poly_state.is_valid or poly_state.market_q is None:
        skip_reason = SKIP_MISSING_MARKET_Q
    elif abs(timing_dev_ms) > MAX_ACCEPTABLE_TIMING_DRIFT_MS:
        skip_reason = SKIP_EXCESSIVE_TIMING_DRIFT
    elif binance_features is not None and not binance_features.is_valid:
        skip_reason = SKIP_INVALID_PERP

    is_valid = skip_reason is None

    snapshot_id = f"{round_slug}_{target_horizon_sec}s_{captured_at_ms}"

    return BTC5mFeatureSnapshot(
        snapshot_id=snapshot_id,
        round_slug=round_slug,
        round_start_epoch=round_start_epoch,
        round_end_epoch=round_end_epoch,
        target_horizon_sec=target_horizon_sec,
        captured_at_ms=captured_at_ms,
        target_scheduled_ms=target_scheduled_ms,
        timing_deviation_ms=timing_dev_ms,
        seconds_remaining=round(seconds_remaining, 2),
        reference_source=ref_features.source,
        price_to_beat=ref_features.price_to_beat,
        price_to_beat_source=ref_features.price_to_beat_source,
        current_reference_price=ref_features.current_price,
        ref_distance_to_beat_bps=ref_features.distance_to_beat_bps,
        ref_return_10s_bps=ref_features.return_10s_bps,
        ref_return_30s_bps=ref_features.return_30s_bps,
        ref_return_60s_bps=ref_features.return_60s_bps,
        ref_data_age_ms=ref_features.data_age_ms,
        market_q=poly_state.market_q,
        up_best_bid=poly_state.up_best_bid,
        up_best_ask=poly_state.up_best_ask,
        down_best_bid=poly_state.down_best_bid,
        down_best_ask=poly_state.down_best_ask,
        poly_spread=poly_state.spread,
        poly_data_age_ms=max(0, captured_at_ms - poly_state.captured_at_ms),
        poly_book_valid=poly_state.is_valid,
        binance_perp_mid=binance_features.mid_price if binance_features else None,
        binance_microprice=binance_features.microprice if binance_features else None,
        binance_microprice_offset_bps=binance_features.microprice_offset_bps
        if binance_features
        else None,
        binance_top5_depth_imbalance=binance_features.top5_depth_imbalance
        if binance_features
        else None,
        binance_top20_depth_imbalance=binance_features.top20_depth_imbalance
        if binance_features
        else None,
        binance_spread_bps=binance_features.spread_bps if binance_features else None,
        binance_taker_flow_10s_imbalance=binance_features.taker_flow_10s_imbalance
        if binance_features
        else None,
        binance_taker_flow_30s_imbalance=binance_features.taker_flow_30s_imbalance
        if binance_features
        else None,
        binance_taker_flow_60s_imbalance=binance_features.taker_flow_60s_imbalance
        if binance_features
        else None,
        binance_taker_buy_qty_60s=binance_features.taker_buy_qty_60s
        if binance_features
        else None,
        binance_taker_sell_qty_60s=binance_features.taker_sell_qty_60s
        if binance_features
        else None,
        binance_return_10s_bps=binance_features.return_10s_bps if binance_features else None,
        binance_return_30s_bps=binance_features.return_30s_bps if binance_features else None,
        binance_return_60s_bps=binance_features.return_60s_bps if binance_features else None,
        binance_return_since_open_bps=binance_features.return_since_round_open_bps
        if binance_features
        else None,
        binance_basis_bps=binance_features.basis_vs_ref_bps if binance_features else None,
        binance_data_age_ms=binance_features.data_age_ms if binance_features else 0,
        binance_valid=binance_features.is_valid if binance_features else False,
        is_valid=is_valid,
        skip_reason=skip_reason,
    )
