"""Immutable point-in-time snapshot for BTC 5-minute prospective forecasting.

Captures all input data at a frozen instant before round settlement:
- Chainlink reference stream (anchor, priceToBeat, current, distance, returns, source vs receipt timestamps)
- Polymarket CLOB state:
    * Native observed: up_bid, up_ask, up_mid, down_bid, down_ask, down_mid, market_q_primary
    * Cross-outcome implied: implied_up_bid, implied_up_ask, implied_mid, market_q_implied_cross_outcome
- Binance USD-M Perpetual microstructure (mid, microprice, depth imbalance, taker flow, basis, Binance-only return since open)
- Strict timing model (Requirements A10, A11):
    * capture_started_at_ms
    * capture_completed_at_ms
    * actual_freeze_timestamp_ms (= capture_completed_at_ms)
    * target_scheduled_ms
    * timing_deviation_ms (evaluated against freeze completion)
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
SKIP_POST_HORIZON_DATA_LEAKAGE: str = "SKIP_POST_HORIZON_DATA_LEAKAGE"


@dataclass(frozen=True)
class BTC5mFeatureSnapshot:
    """Frozen point-in-time observation of all input sources for a single forecast horizon."""

    snapshot_id: str
    round_slug: str
    round_start_epoch: int
    round_end_epoch: int
    target_horizon_sec: int

    # Timing model (Requirements A10, A11)
    capture_started_at_ms: int
    capture_completed_at_ms: int
    actual_freeze_timestamp_ms: int
    target_scheduled_ms: int
    timing_deviation_ms: int
    seconds_remaining: float

    # Reference features & anchor provenance (Requirement A6)
    reference_source: str
    price_to_beat: float | None
    price_to_beat_source: str
    anchor_price: float | None
    anchor_source: str | None
    anchor_source_timestamp_ms: int | None
    anchor_received_timestamp_ms: int | None
    current_reference_price: float
    ref_distance_to_beat_bps: float | None
    ref_return_10s_bps: float | None
    ref_return_30s_bps: float | None
    ref_return_60s_bps: float | None
    ref_source_timestamp_ms: int
    ref_received_at_ms: int
    ref_receipt_age_ms: int
    ref_source_age_ms: int

    # Polymarket book consensus (Requirement A1)
    native_up_bid: float | None
    native_up_ask: float | None
    native_up_mid: float | None
    native_down_bid: float | None
    native_down_ask: float | None
    native_down_mid: float | None
    cross_outcome_implied_up_bid: float | None
    cross_outcome_implied_up_ask: float | None
    cross_outcome_implied_mid: float | None
    market_q_primary: float | None  # Primary market baseline (observed native UP mid)
    market_q_implied_cross_outcome: float | None  # Secondary diagnostic
    poly_spread: float | None
    poly_source_timestamp_ms: int | None
    poly_received_at_ms: int
    poly_data_age_ms: int
    poly_book_valid: bool

    # Binance Perpetual features (Requirements A8, A9)
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
    binance_return_since_open_bps: float | None  # Binance-only open return
    binance_open_mid: float | None
    binance_open_timestamp_ms: int | None
    binance_basis_bps: float | None  # Legitimate cross-source basis
    binance_source_timestamp_ms: int | None
    binance_received_at_ms: int | None
    binance_receipt_age_ms: int
    binance_source_age_ms: int | None
    binance_valid: bool

    # Quality control
    is_valid: bool
    skip_reason: str | None = None
    experiment_spec_hash: str | None = None
    binance_open_source_timestamp_ms: int | None = None
    binance_open_received_at_ms: int | None = None
    binance_open_timing_offset_ms: int | None = None

    # Backward compatibility properties
    @property
    def captured_at_ms(self) -> int:
        """Authoritative completion freeze timestamp."""
        return self.actual_freeze_timestamp_ms

    @property
    def market_q(self) -> float | None:
        """Primary market baseline benchmark."""
        return self.market_q_primary

    @property
    def up_best_bid(self) -> float | None:
        return self.native_up_bid

    @property
    def up_best_ask(self) -> float | None:
        return self.native_up_ask

    @property
    def down_best_bid(self) -> float | None:
        return self.native_down_bid

    @property
    def down_best_ask(self) -> float | None:
        return self.native_down_ask

    @property
    def ref_data_age_ms(self) -> int:
        return self.ref_receipt_age_ms

    @property
    def binance_data_age_ms(self) -> int:
        return self.binance_receipt_age_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BTC5mFeatureSnapshot:
        """Construct from dictionary, handling backward compatibility with legacy fields."""
        data_clean = dict(data)
        # Handle capture timestamp backward compatibility
        freeze_ts = data_clean.get("actual_freeze_timestamp_ms") or data_clean.get("captured_at_ms") or 0
        data_clean.setdefault("actual_freeze_timestamp_ms", freeze_ts)
        data_clean.setdefault("capture_started_at_ms", freeze_ts)
        data_clean.setdefault("capture_completed_at_ms", freeze_ts)

        # Handle market_q baseline backward compatibility
        m_q = data_clean.get("market_q")
        data_clean.setdefault("market_q_primary", data_clean.get("native_up_mid", m_q))
        data_clean.setdefault("market_q_implied_cross_outcome", None)
        data_clean.setdefault("native_up_bid", data_clean.get("up_best_bid"))
        data_clean.setdefault("native_up_ask", data_clean.get("up_best_ask"))
        data_clean.setdefault("native_up_mid", data_clean.get("market_q_primary"))
        data_clean.setdefault("native_down_bid", data_clean.get("down_best_bid"))
        data_clean.setdefault("native_down_ask", data_clean.get("down_best_ask"))
        data_clean.setdefault("native_down_mid", None)
        data_clean.setdefault("cross_outcome_implied_up_bid", None)
        data_clean.setdefault("cross_outcome_implied_up_ask", None)
        data_clean.setdefault("cross_outcome_implied_mid", None)

        # Handle anchor provenance backward compatibility
        data_clean.setdefault("anchor_price", data_clean.get("price_to_beat"))
        data_clean.setdefault("anchor_source", data_clean.get("price_to_beat_source"))
        data_clean.setdefault("anchor_source_timestamp_ms", None)
        data_clean.setdefault("anchor_received_timestamp_ms", None)

        # Handle timestamps
        data_clean.setdefault("ref_source_timestamp_ms", freeze_ts)
        data_clean.setdefault("ref_received_at_ms", freeze_ts)
        data_clean.setdefault("ref_receipt_age_ms", data_clean.get("ref_data_age_ms", 0))
        data_clean.setdefault("ref_source_age_ms", data_clean.get("ref_data_age_ms", 0))
        data_clean.setdefault("poly_source_timestamp_ms", None)
        data_clean.setdefault("poly_received_at_ms", freeze_ts)
        data_clean.setdefault("binance_source_timestamp_ms", None)
        data_clean.setdefault("binance_received_at_ms", freeze_ts)
        data_clean.setdefault("binance_receipt_age_ms", data_clean.get("binance_data_age_ms", 0))
        data_clean.setdefault("binance_source_age_ms", None)
        data_clean.setdefault("binance_open_mid", None)
        data_clean.setdefault("binance_open_timestamp_ms", None)
        data_clean.setdefault("binance_open_source_timestamp_ms", None)
        data_clean.setdefault("binance_open_received_at_ms", None)
        data_clean.setdefault("binance_open_timing_offset_ms", None)
        data_clean.setdefault("experiment_spec_hash", None)

        # Remove keys that aren't in dataclass
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data_clean.items() if k in valid_fields}
        return cls(**filtered)


def build_feature_snapshot(
    round_slug: str,
    round_start_epoch: int,
    round_end_epoch: int,
    target_horizon_sec: int,
    capture_started_at_ms: int,
    capture_completed_at_ms: int,
    ref_features: ReferenceFeatures,
    poly_state: PolymarketMarketState,
    binance_features: BinancePerpFeatures | None,
    captured_at_ms: int | None = None,  # For backward compatibility
    experiment_spec_hash: str | None = None,
) -> BTC5mFeatureSnapshot:
    """Combine individual subsystem feeds into a validated, frozen BTC5mFeatureSnapshot.

    Requirements A10, A11:
    - Authoritative frozen timestamp is capture_completed_at_ms
    - Timing deviation evaluated against freeze completion
    - Enforces post-horizon data leakage check
    """
    freeze_ts_ms = capture_completed_at_ms if capture_completed_at_ms > 0 else (captured_at_ms or capture_started_at_ms)
    start_ts_ms = capture_started_at_ms if capture_started_at_ms > 0 else freeze_ts_ms

    target_scheduled_ms = (round_end_epoch - target_horizon_sec) * 1000
    timing_dev_ms = freeze_ts_ms - target_scheduled_ms
    seconds_remaining = max(0.0, round_end_epoch - (freeze_ts_ms / 1000.0))

    # Evaluate validation conditions
    skip_reason: str | None = None

    # Invariant A6: Exact anchor required
    if ref_features.price_to_beat is None or ref_features.price_to_beat <= 0:
        skip_reason = SKIP_NO_EXACT_ANCHOR
    elif ref_features.receipt_age_ms > 30_000:
        skip_reason = SKIP_STALE_REFERENCE
    # Invariant A1: Book validity
    elif not poly_state.is_valid:
        skip_reason = SKIP_MISSING_MARKET_Q
    # Invariant A10: Timing deviation check against freeze time
    elif abs(timing_dev_ms) > MAX_ACCEPTABLE_TIMING_DRIFT_MS:
        skip_reason = SKIP_EXCESSIVE_TIMING_DRIFT
    # Invariant A11: Post-horizon data leakage check
    # Check that no source datum has an exchange timestamp beyond the target cutoff
    elif ref_features.source_event_timestamp_ms > target_scheduled_ms + MAX_ACCEPTABLE_TIMING_DRIFT_MS:
        skip_reason = SKIP_POST_HORIZON_DATA_LEAKAGE
    elif binance_features is not None and not binance_features.is_valid:
        skip_reason = SKIP_INVALID_PERP

    is_valid = skip_reason is None
    snapshot_id = f"{round_slug}_{target_horizon_sec}s_{freeze_ts_ms}"

    return BTC5mFeatureSnapshot(
        snapshot_id=snapshot_id,
        round_slug=round_slug,
        round_start_epoch=round_start_epoch,
        round_end_epoch=round_end_epoch,
        target_horizon_sec=target_horizon_sec,
        capture_started_at_ms=start_ts_ms,
        capture_completed_at_ms=freeze_ts_ms,
        actual_freeze_timestamp_ms=freeze_ts_ms,
        target_scheduled_ms=target_scheduled_ms,
        timing_deviation_ms=timing_dev_ms,
        seconds_remaining=round(seconds_remaining, 2),
        reference_source=ref_features.source,
        price_to_beat=ref_features.price_to_beat,
        price_to_beat_source=ref_features.price_to_beat_source,
        anchor_price=ref_features.anchor_price,
        anchor_source=ref_features.anchor_source,
        anchor_source_timestamp_ms=ref_features.anchor_source_timestamp_ms,
        anchor_received_timestamp_ms=ref_features.anchor_received_timestamp_ms,
        current_reference_price=ref_features.current_price,
        ref_distance_to_beat_bps=ref_features.distance_to_beat_bps,
        ref_return_10s_bps=ref_features.return_10s_bps,
        ref_return_30s_bps=ref_features.return_30s_bps,
        ref_return_60s_bps=ref_features.return_60s_bps,
        ref_source_timestamp_ms=ref_features.source_event_timestamp_ms,
        ref_received_at_ms=ref_features.received_at_ms,
        ref_receipt_age_ms=ref_features.receipt_age_ms,
        ref_source_age_ms=ref_features.source_age_ms,
        native_up_bid=poly_state.native_up_bid,
        native_up_ask=poly_state.native_up_ask,
        native_up_mid=poly_state.native_up_mid,
        native_down_bid=poly_state.native_down_bid,
        native_down_ask=poly_state.native_down_ask,
        native_down_mid=poly_state.native_down_mid,
        cross_outcome_implied_up_bid=poly_state.cross_outcome_implied_up_bid,
        cross_outcome_implied_up_ask=poly_state.cross_outcome_implied_up_ask,
        cross_outcome_implied_mid=poly_state.cross_outcome_implied_mid,
        market_q_primary=poly_state.market_q_primary,
        market_q_implied_cross_outcome=poly_state.market_q_implied_cross_outcome,
        poly_spread=poly_state.spread,
        poly_source_timestamp_ms=poly_state.up_book.source_event_timestamp_ms,
        poly_received_at_ms=poly_state.up_book.received_at_ms,
        poly_data_age_ms=max(0, freeze_ts_ms - poly_state.captured_at_ms),
        poly_book_valid=poly_state.is_valid,
        binance_perp_mid=binance_features.mid_price if binance_features else None,
        binance_microprice=binance_features.microprice if binance_features else None,
        binance_microprice_offset_bps=binance_features.microprice_offset_bps if binance_features else None,
        binance_top5_depth_imbalance=binance_features.top5_depth_imbalance if binance_features else None,
        binance_top20_depth_imbalance=binance_features.top20_depth_imbalance if binance_features else None,
        binance_spread_bps=binance_features.spread_bps if binance_features else None,
        binance_taker_flow_10s_imbalance=binance_features.taker_flow_10s_imbalance if binance_features else None,
        binance_taker_flow_30s_imbalance=binance_features.taker_flow_30s_imbalance if binance_features else None,
        binance_taker_flow_60s_imbalance=binance_features.taker_flow_60s_imbalance if binance_features else None,
        binance_taker_buy_qty_60s=binance_features.taker_buy_qty_60s if binance_features else None,
        binance_taker_sell_qty_60s=binance_features.taker_sell_qty_60s if binance_features else None,
        binance_return_10s_bps=binance_features.return_10s_bps if binance_features else None,
        binance_return_30s_bps=binance_features.return_30s_bps if binance_features else None,
        binance_return_60s_bps=binance_features.return_60s_bps if binance_features else None,
        binance_return_since_open_bps=binance_features.return_since_round_open_bps if binance_features else None,
        binance_open_mid=binance_features.binance_open_mid if binance_features else None,
        binance_open_timestamp_ms=binance_features.binance_open_timestamp_ms if binance_features else None,
        binance_open_source_timestamp_ms=binance_features.binance_open_source_timestamp_ms if binance_features else None,
        binance_open_received_at_ms=binance_features.binance_open_received_at_ms if binance_features else None,
        binance_open_timing_offset_ms=binance_features.binance_open_timing_offset_ms if binance_features else None,
        binance_basis_bps=binance_features.basis_vs_ref_bps if binance_features else None,
        binance_source_timestamp_ms=binance_features.source_event_timestamp_ms if binance_features else None,
        binance_received_at_ms=binance_features.received_at_ms if binance_features else None,
        binance_receipt_age_ms=binance_features.receipt_age_ms if binance_features else 0,
        binance_source_age_ms=binance_features.source_age_ms if binance_features else None,
        binance_valid=binance_features.is_valid if binance_features else False,
        is_valid=is_valid,
        skip_reason=skip_reason,
        experiment_spec_hash=experiment_spec_hash,
    )
