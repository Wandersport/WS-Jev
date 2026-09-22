"""Public read-only Binance USD-M Perpetual BTC/USDT microstructure feed.

Strictly unauthenticated read-only access:
- REST: GET https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=20
- REST: GET https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&limit=1000
- WebSocket (optional/streaming): wss://fstream.binance.com/ws/btcusdt@depth20@100ms
- ZERO credentials, API keys, or signed endpoints.
- Rejects crossed order books.
- Computes:
  * Top-5 and Top-20 depth imbalance
  * Microprice and microprice offset (bps)
  * Taker flow buy/sell volume and flow imbalance over 10s, 30s, 60s windows
    (returns None if data coverage is less than the window, never fabricating zero)
  * Price returns over 10s, 30s, 60s windows and since round start (bps)
  * Basis vs Chainlink reference (bps)
"""

from __future__ import annotations

import collections
import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BINANCE_FAPI_BASE: str = "https://fapi.binance.com"
ALLOWED_BINANCE_HOSTS: frozenset[str] = frozenset({"fapi.binance.com", "fstream.binance.com"})


@dataclass(frozen=True)
class BinancePerpFeatures:
    """Microstructure features extracted from Binance BTCUSDT perpetual."""

    best_bid: float
    best_ask: float
    mid_price: float
    microprice: float
    microprice_offset_bps: float
    top5_depth_imbalance: float
    top20_depth_imbalance: float
    total_bid_qty_top20: float
    total_ask_qty_top20: float
    spread_bps: float
    taker_flow_10s_imbalance: float | None
    taker_flow_30s_imbalance: float | None
    taker_flow_60s_imbalance: float | None
    taker_buy_qty_60s: float | None
    taker_sell_qty_60s: float | None
    return_10s_bps: float | None
    return_30s_bps: float | None
    return_60s_bps: float | None
    return_since_round_open_bps: float | None
    basis_vs_ref_bps: float | None
    snapshot_time_ms: int
    data_age_ms: int
    is_valid: bool
    rejection_reason: str | None = None


class BinancePerpFeed:
    """Manages Binance BTC/USDT perpetual depth and trade history to compute microstructure features."""

    def __init__(self, max_trade_history: int = 5000) -> None:
        self._bids: list[tuple[float, float]] = []  # sorted descending by price: (price, qty)
        self._asks: list[tuple[float, float]] = []  # sorted ascending by price: (price, qty)
        self._depth_timestamp_ms: int = 0
        # trades stored as: (timestamp_ms, price, qty, is_buyer_maker)
        # is_buyer_maker == True -> maker was buyer -> taker was seller (taker sell)
        # is_buyer_maker == False -> maker was seller -> taker was buyer (taker buy)
        self._trades: collections.deque[tuple[int, float, float, bool]] = collections.deque(
            maxlen=max_trade_history
        )

    def add_depth(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        timestamp_ms: int,
    ) -> None:
        """Inject or update depth levels."""
        # Sort bids descending, asks ascending
        self._bids = sorted(bids, key=lambda x: x[0], reverse=True)
        self._asks = sorted(asks, key=lambda x: x[0])
        self._depth_timestamp_ms = timestamp_ms

    def add_trade(
        self,
        timestamp_ms: int,
        price: float,
        qty: float,
        is_buyer_maker: bool,
    ) -> None:
        """Inject a single executed trade."""
        self._trades.append((timestamp_ms, price, qty, is_buyer_maker))

    def add_trades(
        self,
        trades: list[tuple[int, float, float, bool]],
    ) -> None:
        """Batch inject trades."""
        for t in trades:
            self.add_trade(t[0], t[1], t[2], t[3])

    def fetch_rest_snapshot(
        self,
        ref_price: float | None = None,
        round_start_price: float | None = None,
        timeout_sec: float = 5.0,
    ) -> BinancePerpFeatures:
        """Fetch depth and aggregate trades via public read-only REST endpoints."""
        now_ms = int(time.time() * 1000)

        # 1. Fetch depth
        depth_url = f"{BINANCE_FAPI_BASE}/fapi/v1/depth?symbol=BTCUSDT&limit=20"
        parsed_depth = urllib.parse.urlparse(depth_url)
        if parsed_depth.hostname not in ALLOWED_BINANCE_HOSTS:
            raise ValueError(f"Prohibited host: {parsed_depth.hostname}")

        req_depth = urllib.request.Request(
            depth_url,
            headers={"User-Agent": "WS-Jev-Research/1.0 (Public Read-Only)"},
        )
        with urllib.request.urlopen(req_depth, timeout=timeout_sec) as resp:
            raw_depth = json.loads(resp.read().decode("utf-8"))

        bids = [(float(p), float(q)) for p, q in raw_depth.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in raw_depth.get("asks", [])]
        depth_ts = int(raw_depth.get("T", now_ms))
        self.add_depth(bids, asks, depth_ts)

        # 2. Fetch recent aggTrades
        trades_url = f"{BINANCE_FAPI_BASE}/fapi/v1/aggTrades?symbol=BTCUSDT&limit=1000"
        parsed_trades = urllib.parse.urlparse(trades_url)
        if parsed_trades.hostname not in ALLOWED_BINANCE_HOSTS:
            raise ValueError(f"Prohibited host: {parsed_trades.hostname}")

        req_trades = urllib.request.Request(
            trades_url,
            headers={"User-Agent": "WS-Jev-Research/1.0 (Public Read-Only)"},
        )
        with urllib.request.urlopen(req_trades, timeout=timeout_sec) as resp:
            raw_trades = json.loads(resp.read().decode("utf-8"))

        for item in raw_trades:
            # item: {'a': agg_id, 'p': price, 'q': qty, 'f': first_id, 'l': last_id, 'T': time, 'm': is_buyer_maker}
            try:
                t_ms = int(item["T"])
                p_val = float(item["p"])
                q_val = float(item["q"])
                m_val = bool(item["m"])
                self.add_trade(t_ms, p_val, q_val, m_val)
            except (KeyError, ValueError, TypeError):
                continue

        return self.compute_features(
            now_ms=now_ms,
            ref_price=ref_price,
            round_start_price=round_start_price,
        )

    def compute_features(
        self,
        now_ms: int,
        ref_price: float | None = None,
        round_start_price: float | None = None,
    ) -> BinancePerpFeatures:
        """Extract all microstructure features from current state."""
        if not self._bids or not self._asks:
            return BinancePerpFeatures(
                best_bid=0.0,
                best_ask=0.0,
                mid_price=0.0,
                microprice=0.0,
                microprice_offset_bps=0.0,
                top5_depth_imbalance=0.0,
                top20_depth_imbalance=0.0,
                total_bid_qty_top20=0.0,
                total_ask_qty_top20=0.0,
                spread_bps=0.0,
                taker_flow_10s_imbalance=None,
                taker_flow_30s_imbalance=None,
                taker_flow_60s_imbalance=None,
                taker_buy_qty_60s=None,
                taker_sell_qty_60s=None,
                return_10s_bps=None,
                return_30s_bps=None,
                return_60s_bps=None,
                return_since_round_open_bps=None,
                basis_vs_ref_bps=None,
                snapshot_time_ms=now_ms,
                data_age_ms=0,
                is_valid=False,
                rejection_reason="Empty bids or asks in order book",
            )

        best_bid = self._bids[0][0]
        bid_qty_1 = self._bids[0][1]
        best_ask = self._asks[0][0]
        ask_qty_1 = self._asks[0][1]

        # Invariant: reject crossed order books
        if best_bid >= best_ask:
            return BinancePerpFeatures(
                best_bid=best_bid,
                best_ask=best_ask,
                mid_price=round((best_bid + best_ask) / 2.0, 4),
                microprice=0.0,
                microprice_offset_bps=0.0,
                top5_depth_imbalance=0.0,
                top20_depth_imbalance=0.0,
                total_bid_qty_top20=0.0,
                total_ask_qty_top20=0.0,
                spread_bps=0.0,
                taker_flow_10s_imbalance=None,
                taker_flow_30s_imbalance=None,
                taker_flow_60s_imbalance=None,
                taker_buy_qty_60s=None,
                taker_sell_qty_60s=None,
                return_10s_bps=None,
                return_30s_bps=None,
                return_60s_bps=None,
                return_since_round_open_bps=None,
                basis_vs_ref_bps=None,
                snapshot_time_ms=now_ms,
                data_age_ms=max(0, now_ms - self._depth_timestamp_ms),
                is_valid=False,
                rejection_reason=f"Crossed order book: bid {best_bid} >= ask {best_ask}",
            )

        mid_price = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / mid_price) * 10_000.0

        # Microprice calculation: (best_bid * ask_qty_1 + best_ask * bid_qty_1) / (bid_qty_1 + ask_qty_1)
        tot_top1_qty = bid_qty_1 + ask_qty_1
        if tot_top1_qty > 0:
            microprice = (best_bid * ask_qty_1 + best_ask * bid_qty_1) / tot_top1_qty
            microprice_offset_bps = ((microprice - mid_price) / mid_price) * 10_000.0
        else:
            microprice = mid_price
            microprice_offset_bps = 0.0

        # Depth imbalances: top 5 and top 20
        top5_bid_qty = sum(q for _, q in self._bids[:5])
        top5_ask_qty = sum(q for _, q in self._asks[:5])
        tot5 = top5_bid_qty + top5_ask_qty
        top5_imbalance = (top5_bid_qty - top5_ask_qty) / tot5 if tot5 > 0 else 0.0

        top20_bid_qty = sum(q for _, q in self._bids[:20])
        top20_ask_qty = sum(q for _, q in self._asks[:20])
        tot20 = top20_bid_qty + top20_ask_qty
        top20_imbalance = (top20_bid_qty - top20_ask_qty) / tot20 if tot20 > 0 else 0.0

        # Basis vs reference feed
        basis_vs_ref_bps: float | None = None
        if ref_price is not None and ref_price > 0:
            basis_vs_ref_bps = round(((mid_price - ref_price) / ref_price) * 10_000.0, 3)

        # Return since round open
        return_since_round_open_bps: float | None = None
        if round_start_price is not None and round_start_price > 0:
            return_since_round_open_bps = round(
                ((mid_price - round_start_price) / round_start_price) * 10_000.0, 3
            )

        # Taker flow and price returns over rolling windows: 10s, 30s, 60s
        taker_10s = self._compute_taker_flow_window(now_ms, window_sec=10)
        taker_30s = self._compute_taker_flow_window(now_ms, window_sec=30)
        taker_60s = self._compute_taker_flow_window(now_ms, window_sec=60)

        ret_10s = self._compute_price_return_window(now_ms, mid_price, window_sec=10)
        ret_30s = self._compute_price_return_window(now_ms, mid_price, window_sec=30)
        ret_60s = self._compute_price_return_window(now_ms, mid_price, window_sec=60)

        data_age_ms = max(0, now_ms - self._depth_timestamp_ms) if self._depth_timestamp_ms else 0

        return BinancePerpFeatures(
            best_bid=round(best_bid, 2),
            best_ask=round(best_ask, 2),
            mid_price=round(mid_price, 2),
            microprice=round(microprice, 2),
            microprice_offset_bps=round(microprice_offset_bps, 3),
            top5_depth_imbalance=round(top5_imbalance, 4),
            top20_depth_imbalance=round(top20_imbalance, 4),
            total_bid_qty_top20=round(top20_bid_qty, 4),
            total_ask_qty_top20=round(top20_ask_qty, 4),
            spread_bps=round(spread_bps, 3),
            taker_flow_10s_imbalance=round(taker_10s[0], 4) if taker_10s is not None else None,
            taker_flow_30s_imbalance=round(taker_30s[0], 4) if taker_30s is not None else None,
            taker_flow_60s_imbalance=round(taker_60s[0], 4) if taker_60s is not None else None,
            taker_buy_qty_60s=round(taker_60s[1], 4) if taker_60s is not None else None,
            taker_sell_qty_60s=round(taker_60s[2], 4) if taker_60s is not None else None,
            return_10s_bps=round(ret_10s, 3) if ret_10s is not None else None,
            return_30s_bps=round(ret_30s, 3) if ret_30s is not None else None,
            return_60s_bps=round(ret_60s, 3) if ret_60s is not None else None,
            return_since_round_open_bps=return_since_round_open_bps,
            basis_vs_ref_bps=basis_vs_ref_bps,
            snapshot_time_ms=now_ms,
            data_age_ms=data_age_ms,
            is_valid=True,
            rejection_reason=None,
        )

    def _compute_taker_flow_window(
        self,
        now_ms: int,
        window_sec: int,
    ) -> tuple[float, float, float] | None:
        """Compute (imbalance, buy_qty, sell_qty) for window.

        Returns None if trade history does NOT cover at least window_sec.
        Never fabricate zero imbalance when coverage is missing.
        """
        if not self._trades:
            return None

        window_ms = window_sec * 1000
        cutoff_ms = now_ms - window_ms

        oldest_ts = self._trades[0][0]
        # Invariant: data coverage requirement
        # Oldest trade must be at or before cutoff_ms, otherwise we don't have the full window
        if oldest_ts > cutoff_ms:
            return None

        buy_qty = 0.0
        sell_qty = 0.0

        for ts, _price, qty, is_buyer_maker in reversed(self._trades):
            if ts < cutoff_ms:
                break
            if is_buyer_maker:
                # Maker was buyer -> taker was seller -> taker sell
                sell_qty += qty
            else:
                # Maker was seller -> taker was buyer -> taker buy
                buy_qty += qty

        tot = buy_qty + sell_qty
        imbalance = (buy_qty - sell_qty) / tot if tot > 0 else 0.0
        return (imbalance, buy_qty, sell_qty)

    def _compute_price_return_window(
        self,
        now_ms: int,
        current_mid: float,
        window_sec: int,
    ) -> float | None:
        """Compute price return over window_sec in basis points.

        Returns None if trade history does not span back to window_sec.
        """
        if not self._trades or current_mid <= 0:
            return None

        window_ms = window_sec * 1000
        target_ts = now_ms - window_ms

        oldest_ts = self._trades[0][0]
        if oldest_ts > target_ts:
            return None

        # Find trade closest to target_ts (searching backwards)
        past_price: float | None = None
        for ts, price, _qty, _m in reversed(self._trades):
            if ts <= target_ts:
                past_price = price
                break

        if past_price is None or past_price <= 0:
            return None

        return ((current_mid - past_price) / past_price) * 10_000.0
