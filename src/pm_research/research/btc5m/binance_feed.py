"""Public read-only Binance USD-M Perpetual BTC/USDT microstructure feed.

Strictly unauthenticated read-only access:
- REST: GET https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=20
- REST: GET https://fapi.binance.com/fapi/v1/aggTrades?symbol=BTCUSDT&limit=1000
- WebSocket (optional/streaming): wss://fstream.binance.com/ws/btcusdt@depth20@100ms
- ZERO credentials, API keys, or signed endpoints.
- Rejects crossed order books.
- Strict provenance (Requirements A8, A9):
    * Distinguishes source_event_timestamp vs received_at_ms
    * Explicitly separates receipt_age_ms from source_age_ms
    * Binance return since open uses ONLY observed Binance perp opening mid, NEVER Chainlink anchor
    * Basis vs Chainlink reference is kept as the legitimate cross-source measure
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BINANCE_FAPI_BASE: str = "https://fapi.binance.com"
BINANCE_WS_STREAM_URL: str = (
    "wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@aggTrade"
)
ALLOWED_BINANCE_HOSTS: frozenset[str] = frozenset({"fapi.binance.com", "fstream.binance.com"})
BINANCE_OPEN_TIMING_TOLERANCE_MS: int = 2500


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
    binance_open_mid: float | None
    binance_open_timestamp_ms: int | None

    # Legitimate cross-source basis vs Chainlink reference
    basis_vs_ref_bps: float | None

    # Requirement A9: Source vs received timestamps
    source_event_timestamp_ms: int | None
    received_at_ms: int
    receipt_age_ms: int
    source_age_ms: int | None

    is_valid: bool
    binance_open_source_timestamp_ms: int | None = None
    binance_open_received_at_ms: int | None = None
    binance_open_timing_offset_ms: int | None = None
    rejection_reason: str | None = None

    @property
    def snapshot_time_ms(self) -> int:
        return self.received_at_ms

    @property
    def data_age_ms(self) -> int:
        """Backward compatibility: return receipt age."""
        return self.receipt_age_ms


class BinancePerpFeed:
    """Manages Binance BTC/USDT perpetual depth and trade history to compute microstructure features."""

    def __init__(self, max_trade_history: int = 5000) -> None:
        self._state_lock: threading.RLock = threading.RLock()
        self._bids: list[tuple[float, float]] = []  # sorted descending by price: (price, qty)
        self._asks: list[tuple[float, float]] = []  # sorted ascending by price: (price, qty)
        self._depth_source_timestamp_ms: int | None = None
        self._depth_received_at_ms: int = 0
        # trades stored as: (timestamp_ms, price, qty, is_buyer_maker)
        self._trades: collections.deque[tuple[int, float, float, bool]] = collections.deque(
            maxlen=max_trade_history
        )
        # Observed Binance midpoints at round boundaries: round_slug -> (mid, recv_ts, source_ts, offset_ms)
        self._recorded_round_opens: dict[str, tuple[float, int, int | None, int | None]] = {}
        # Circular buffer of observed midpoints: (source_ts, recv_ts, mid_price)
        self._midpoint_history: collections.deque[tuple[int | None, int, float]] = collections.deque(
            maxlen=50000
        )
        self._running: bool = False
        self.status: str = "Initialized (disconnected)"
        self._listener_thread: threading.Thread | None = None

    def record_midpoint(self, source_ts: int | None, recv_ts: int, mid_price: float) -> None:
        """Record an observed midpoint in the circular history buffer."""
        if mid_price > 0:
            with self._state_lock:
                self._midpoint_history.append((source_ts, recv_ts, mid_price))

    def find_boundary_open_mid(
        self,
        round_start_ms: int,
        tolerance_ms: int = BINANCE_OPEN_TIMING_TOLERANCE_MS,
    ) -> tuple[float, int | None, int, int] | None:
        """Find the closest observed Binance perpetual midpoint to the round opening boundary.

        Returns (mid_price, source_timestamp_ms, received_at_ms, timing_offset_ms)
        if an observation exists within +/- tolerance_ms of round_start_ms.
        Returns None if no observation falls within the tolerance window.
        """
        with self._state_lock:
            if not self._midpoint_history:
                return None
            midpoint_history = list(self._midpoint_history)

        best_obs: tuple[int | None, int, float] | None = None
        best_diff: float = float("inf")
        best_offset: int = 0

        for source_ts, recv_ts, mid in midpoint_history:
            eval_ts = source_ts if source_ts is not None else recv_ts
            diff = abs(eval_ts - round_start_ms)
            if diff < best_diff:
                best_diff = diff
                best_obs = (source_ts, recv_ts, mid)
                best_offset = eval_ts - round_start_ms

        if best_obs is not None and best_diff <= tolerance_ms:
            source_ts, recv_ts, mid = best_obs
            return (mid, source_ts, recv_ts, best_offset)

        return None

    def record_round_open(
        self,
        round_slug: str,
        mid_price: float,
        timestamp_ms: int,
        source_timestamp_ms: int | None = None,
        offset_ms: int | None = None,
    ) -> None:
        """Record the observed Binance perpetual midpoint at the round opening boundary."""
        if mid_price > 0:
            with self._state_lock:
                self._recorded_round_opens[round_slug] = (
                    mid_price,
                    timestamp_ms,
                    source_timestamp_ms,
                    offset_ms,
                )

    def get_round_open(
        self, round_slug: str
    ) -> tuple[float, int, int | None, int | None] | None:
        """Retrieve recorded Binance opening mid and provenance for a round."""
        with self._state_lock:
            rec = self._recorded_round_opens.get(round_slug)
        if rec is None:
            return None
        if len(rec) == 2:
            return (rec[0], rec[1], None, None)
        return rec

    def add_depth(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        source_timestamp_ms: int | None,
        received_at_ms: int | None = None,
    ) -> None:
        """Inject or update depth levels."""
        sorted_bids = sorted(bids, key=lambda x: x[0], reverse=True)
        sorted_asks = sorted(asks, key=lambda x: x[0])
        recv_ms = received_at_ms or int(time.time() * 1000)

        with self._state_lock:
            self._bids = sorted_bids
            self._asks = sorted_asks
            self._depth_source_timestamp_ms = source_timestamp_ms
            self._depth_received_at_ms = recv_ms

            # Record midpoint in circular buffer if valid order book
            if sorted_bids and sorted_asks:
                bb = sorted_bids[0][0]
                ba = sorted_asks[0][0]
                if bb < ba:
                    mid = (bb + ba) / 2.0
                    self._midpoint_history.append((source_timestamp_ms, recv_ms, mid))

    def add_trade(
        self,
        timestamp_ms: int,
        price: float,
        qty: float,
        is_buyer_maker: bool,
    ) -> None:
        """Inject a single executed trade."""
        with self._state_lock:
            self._trades.append((timestamp_ms, price, qty, is_buyer_maker))

    def add_trades(
        self,
        trades: list[tuple[int, float, float, bool]],
    ) -> None:
        """Batch inject trades."""
        with self._state_lock:
            for t in trades:
                self._trades.append((t[0], t[1], t[2], t[3]))

    def start_background_listener(self) -> None:
        """Start non-blocking daemon thread to maintain live Binance perp WebSocket connection."""
        with self._state_lock:
            if self._running:
                return
            self._running = True
            self.status = "Connecting..."

        def _worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run() -> None:
                while self._running:
                    try:
                        import websockets

                        async with websockets.connect(
                            BINANCE_WS_STREAM_URL,
                            ping_interval=20,
                            ping_timeout=10,
                            close_timeout=5,
                        ) as ws:
                            with self._state_lock:
                                self.status = "Connected"
                            while self._running:
                                msg = await ws.recv()
                                now_ms = int(time.time() * 1000)
                                try:
                                    parsed = json.loads(msg)
                                    stream = parsed.get("stream", "")
                                    payload = parsed.get("data", {})
                                    if "depth" in stream:
                                        bids = [(float(p), float(q)) for p, q in payload.get("b", [])]
                                        asks = [(float(p), float(q)) for p, q in payload.get("a", [])]
                                        source_ts = payload.get("T") or payload.get("E")
                                        s_ts = int(source_ts) if source_ts is not None else None
                                        self.add_depth(bids, asks, s_ts, now_ms)
                                    elif "aggTrade" in stream:
                                        t_ms = int(payload["T"])
                                        p_val = float(payload["p"])
                                        q_val = float(payload["q"])
                                        m_val = bool(payload["m"])
                                        self.add_trade(t_ms, p_val, q_val, m_val)
                                except Exception:
                                    continue
                    except Exception as e:
                        with self._state_lock:
                            self.status = f"Disconnected ({e})"
                        await asyncio.sleep(2.0)

            try:
                loop.run_until_complete(_run())
            except Exception:
                pass
            finally:
                loop.close()

        self._listener_thread = threading.Thread(target=_worker, daemon=True, name="binance_perp_ws")
        self._listener_thread.start()

    def stop_background_listener(self) -> None:
        """Stop background WebSocket listener."""
        with self._state_lock:
            self._running = False
            self.status = "Stopped"

    def is_connected(self) -> bool:
        """Check if WebSocket stream is currently connected."""
        with self._state_lock:
            return self.status == "Connected"

    def fetch_rest_snapshot(
        self,
        ref_price: float | None = None,
        binance_open_price: float | None = None,
        binance_open_timestamp_ms: int | None = None,
        round_slug: str | None = None,
        timeout_sec: float = 5.0,
    ) -> BinancePerpFeatures:
        """Fetch depth and aggregate trades via public read-only REST endpoints."""
        recv_ms = int(time.time() * 1000)

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

        # Requirement A9: Extract source exchange timestamp if present, otherwise None
        raw_ts = raw_depth.get("T") or raw_depth.get("E")
        source_ts: int | None = int(raw_ts) if raw_ts is not None else None
        self.add_depth(bids, asks, source_ts, recv_ms)

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
            try:
                t_ms = int(item["T"])
                p_val = float(item["p"])
                q_val = float(item["q"])
                m_val = bool(item["m"])
                self.add_trade(t_ms, p_val, q_val, m_val)
            except (KeyError, ValueError, TypeError):
                continue

        # Lookup recorded open price if round_slug provided
        open_source_ts: int | None = None
        open_recv_ts: int | None = None
        open_offset_ms: int | None = None
        if round_slug and binance_open_price is None:
            rec = self.get_round_open(round_slug)
            if rec:
                binance_open_price = rec[0]
                binance_open_timestamp_ms = rec[1]
                open_recv_ts = rec[1]
                open_source_ts = rec[2]
                open_offset_ms = rec[3]

        return self.compute_features(
            now_ms=recv_ms,
            ref_price=ref_price,
            binance_open_price=binance_open_price,
            binance_open_timestamp_ms=binance_open_timestamp_ms,
            binance_open_source_timestamp_ms=open_source_ts,
            binance_open_received_at_ms=open_recv_ts,
            binance_open_timing_offset_ms=open_offset_ms,
        )

    def get_live_features(
        self,
        ref_price: float | None = None,
        round_slug: str | None = None,
        now_ms: int | None = None,
    ) -> BinancePerpFeatures:
        """Extract features from memory if streaming, fallback to REST snapshot if empty or stale."""
        current_time = now_ms or int(time.time() * 1000)
        with self._state_lock:
            has_depth = bool(self._bids and self._asks)
            depth_recv_ms = self._depth_received_at_ms

        # If we have bids/asks and data is fresh (< 5s old), compute from live memory
        if has_depth and (current_time - depth_recv_ms < 5000):
            open_mid: float | None = None
            open_ts: int | None = None
            open_source_ts: int | None = None
            open_offset_ms: int | None = None
            if round_slug:
                rec = self.get_round_open(round_slug)
                if rec:
                    open_mid, open_ts, open_source_ts, open_offset_ms = rec
            return self.compute_features(
                now_ms=current_time,
                ref_price=ref_price,
                binance_open_price=open_mid,
                binance_open_timestamp_ms=open_ts,
                binance_open_source_timestamp_ms=open_source_ts,
                binance_open_received_at_ms=open_ts,
                binance_open_timing_offset_ms=open_offset_ms,
            )
        # Fallback to REST snapshot
        return self.fetch_rest_snapshot(ref_price=ref_price, round_slug=round_slug)

    def compute_features(
        self,
        now_ms: int,
        ref_price: float | None = None,
        binance_open_price: float | None = None,
        binance_open_timestamp_ms: int | None = None,
        round_start_price: float | None = None,  # Deprecated parameter; mapped to binance_open_price
        binance_open_source_timestamp_ms: int | None = None,
        binance_open_received_at_ms: int | None = None,
        binance_open_timing_offset_ms: int | None = None,
    ) -> BinancePerpFeatures:
        """Extract all microstructure features from current state.

        Requirement A8: Binance return since open is calculated strictly from
        binance_open_price (never from Chainlink anchor).
        Requirement A9: Distinguish source event timestamp from receipt timestamp.
        """
        # Fallback for API compatibility if caller passed round_start_price
        if binance_open_price is None and round_start_price is not None:
            binance_open_price = round_start_price

        # Atomic memory snapshot of order book and trade history under lock
        with self._state_lock:
            bids = list(self._bids)
            asks = list(self._asks)
            source_ts = self._depth_source_timestamp_ms
            recv_ts = self._depth_received_at_ms or now_ms
            trade_list = list(self._trades)

        receipt_age = max(0, now_ms - recv_ts)
        source_age = max(0, now_ms - source_ts) if source_ts is not None else None

        if not bids or not asks:
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
                binance_open_mid=binance_open_price,
                binance_open_timestamp_ms=binance_open_timestamp_ms,
                binance_open_source_timestamp_ms=binance_open_source_timestamp_ms,
                binance_open_received_at_ms=binance_open_received_at_ms,
                binance_open_timing_offset_ms=binance_open_timing_offset_ms,
                basis_vs_ref_bps=None,
                source_event_timestamp_ms=source_ts,
                received_at_ms=recv_ts,
                receipt_age_ms=receipt_age,
                source_age_ms=source_age,
                is_valid=False,
                rejection_reason="Empty bids or asks in order book",
            )

        best_bid = bids[0][0]
        bid_qty_1 = bids[0][1]
        best_ask = asks[0][0]
        ask_qty_1 = asks[0][1]

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
                binance_open_mid=binance_open_price,
                binance_open_timestamp_ms=binance_open_timestamp_ms,
                binance_open_source_timestamp_ms=binance_open_source_timestamp_ms,
                binance_open_received_at_ms=binance_open_received_at_ms,
                binance_open_timing_offset_ms=binance_open_timing_offset_ms,
                basis_vs_ref_bps=None,
                source_event_timestamp_ms=source_ts,
                received_at_ms=recv_ts,
                receipt_age_ms=receipt_age,
                source_age_ms=source_age,
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
        top5_bid_qty = sum(q for _, q in bids[:5])
        top5_ask_qty = sum(q for _, q in asks[:5])
        tot5 = top5_bid_qty + top5_ask_qty
        top5_imbalance = (top5_bid_qty - top5_ask_qty) / tot5 if tot5 > 0 else 0.0

        top20_bid_qty = sum(q for _, q in bids[:20])
        top20_ask_qty = sum(q for _, q in asks[:20])
        tot20 = top20_bid_qty + top20_ask_qty
        top20_imbalance = (top20_bid_qty - top20_ask_qty) / tot20 if tot20 > 0 else 0.0

        # Basis vs reference feed (legitimate cross-source measure)
        basis_vs_ref_bps: float | None = None
        if ref_price is not None and ref_price > 0:
            basis_vs_ref_bps = round(((mid_price - ref_price) / ref_price) * 10_000.0, 3)

        # Requirement: Return since round open uses ONLY observed Binance open price.
        # If open price missing or not positive, or if timing offset exceeds tolerance, return None.
        return_since_round_open_bps: float | None = None
        if binance_open_price is not None and binance_open_price > 0:
            if binance_open_timing_offset_ms is None or abs(binance_open_timing_offset_ms) <= BINANCE_OPEN_TIMING_TOLERANCE_MS:
                return_since_round_open_bps = round(
                    ((mid_price - binance_open_price) / binance_open_price) * 10_000.0, 3
                )

        # Taker flow and price returns over rolling windows: 10s, 30s, 60s using snapshot
        taker_10s = self._compute_taker_flow_window(now_ms, window_sec=10, trades=trade_list)
        taker_30s = self._compute_taker_flow_window(now_ms, window_sec=30, trades=trade_list)
        taker_60s = self._compute_taker_flow_window(now_ms, window_sec=60, trades=trade_list)

        ret_10s = self._compute_price_return_window(now_ms, mid_price, window_sec=10, trades=trade_list)
        ret_30s = self._compute_price_return_window(now_ms, mid_price, window_sec=30, trades=trade_list)
        ret_60s = self._compute_price_return_window(now_ms, mid_price, window_sec=60, trades=trade_list)

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
            binance_open_mid=binance_open_price,
            binance_open_timestamp_ms=binance_open_timestamp_ms,
            binance_open_source_timestamp_ms=binance_open_source_timestamp_ms,
            binance_open_received_at_ms=binance_open_received_at_ms,
            binance_open_timing_offset_ms=binance_open_timing_offset_ms,
            basis_vs_ref_bps=basis_vs_ref_bps,
            source_event_timestamp_ms=source_ts,
            received_at_ms=recv_ts,
            receipt_age_ms=receipt_age,
            source_age_ms=source_age,
            is_valid=True,
            rejection_reason=None,
        )

    def _compute_taker_flow_window(
        self,
        now_ms: int,
        window_sec: int,
        trades: list[tuple[int, float, float, bool]] | None = None,
    ) -> tuple[float, float, float] | None:
        """Compute (imbalance, buy_qty, sell_qty) for window.

        Returns None if trade history does NOT cover at least window_sec.
        Never fabricate zero imbalance when coverage is missing.
        """
        if trades is None:
            with self._state_lock:
                if not self._trades:
                    return None
                trade_list = list(self._trades)
        else:
            trade_list = trades

        if not trade_list:
            return None

        window_ms = window_sec * 1000
        cutoff_ms = now_ms - window_ms

        oldest_ts = trade_list[0][0]
        # Invariant: data coverage requirement
        if oldest_ts > cutoff_ms:
            return None

        buy_qty = 0.0
        sell_qty = 0.0

        for ts, _price, qty, is_buyer_maker in reversed(trade_list):
            if ts < cutoff_ms:
                break
            if is_buyer_maker:
                sell_qty += qty
            else:
                buy_qty += qty

        tot = buy_qty + sell_qty
        imbalance = (buy_qty - sell_qty) / tot if tot > 0 else 0.0
        return (imbalance, buy_qty, sell_qty)

    def _compute_price_return_window(
        self,
        now_ms: int,
        current_mid: float,
        window_sec: int,
        trades: list[tuple[int, float, float, bool]] | None = None,
    ) -> float | None:
        """Compute price return over window_sec in basis points.

        Returns None if trade history does not span back to window_sec.
        """
        if current_mid <= 0:
            return None

        if trades is None:
            with self._state_lock:
                if not self._trades:
                    return None
                trade_list = list(self._trades)
        else:
            trade_list = trades

        if not trade_list:
            return None

        window_ms = window_sec * 1000
        target_ts = now_ms - window_ms

        oldest_ts = trade_list[0][0]
        if oldest_ts > target_ts:
            return None

        past_price: float | None = None
        for ts, price, _qty, _m in reversed(trade_list):
            if ts <= target_ts:
                past_price = price
                break

        if past_price is None or past_price <= 0:
            return None

        return ((current_mid - past_price) / past_price) * 10_000.0
