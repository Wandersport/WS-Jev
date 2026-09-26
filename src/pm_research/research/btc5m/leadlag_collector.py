"""High-frequency 1-second synchronized lead-lag observational collector.

Continuous public read-only collection:
- Binance USD-M Perpetual BTC/USDT (orderbook depth + aggTrades)
- Polymarket CLOB BTC 5-minute native UP orderbook (streaming WebSocket + REST fallback)

ZERO live trading. ZERO paper broker orders. ZERO OpenRouter/Jev requests.
"""

from __future__ import annotations

import asyncio
import collections
import fcntl
import json
import logging
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from pm_research.research.btc5m.contract import BTC5mContractManager, BTC5mRoundInfo
from pm_research.research.btc5m.leadlag_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_SPEC_HASH,
    MAX_STALE_AGE_MS,
    LeadLagSample,
)
from pm_research.research.btc5m.poly_book import PolymarketBookCollector
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

BINANCE_WS_URL: str = "wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@aggTrade"
POLY_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DEFAULT_LOCK_FILE: str = "data/btc5m_leadlag_collector.lock"


class LeadLagCollector:
    """Synchronized 1-second lead-lag observational collector."""

    def __init__(
        self,
        db: Database,
        target_physical_rounds: int = 500,
        lock_file_path: str = DEFAULT_LOCK_FILE,
    ) -> None:
        self.db = db
        self.target_physical_rounds = target_physical_rounds
        self.lock_file_path = Path(lock_file_path)
        self.contract_mgr = BTC5mContractManager()
        self.poly_rest_collector = PolymarketBookCollector()

        self._lock_fd: int | None = None
        self._running: bool = False
        self._stop_event = threading.Event()

        # Thread synchronization locks
        self._binance_lock = threading.RLock()
        self._poly_lock = threading.RLock()

        # Binance state
        self._binance_bids: list[tuple[float, float]] = []  # [(price, qty)]
        self._binance_asks: list[tuple[float, float]] = []
        self._binance_source_ts_ms: int | None = None
        self._binance_recv_ts_ms: int = 0
        self._binance_trades: collections.deque[tuple[int, float, float, bool, int]] = collections.deque(maxlen=10000)
        # (source_ts, recv_ts, mid)
        self._binance_mids: collections.deque[tuple[int | None, int, float]] = collections.deque(maxlen=3600)
        self._binance_open_mid: float | None = None
        self._binance_open_round_slug: str | None = None
        self._binance_status: str = "INITIALIZING"

        # Polymarket state
        self._poly_token_id: str | None = None
        self._poly_round_slug: str | None = None
        self._poly_best_bid: float | None = None
        self._poly_best_ask: float | None = None
        self._poly_midpoint: float | None = None
        self._poly_source_ts_ms: int | None = None
        self._poly_recv_ts_ms: int = 0
        self._poly_is_crossed: bool = False
        self._poly_mids: collections.deque[tuple[int | None, int, float]] = collections.deque(maxlen=1800)
        self._poly_status: str = "INITIALIZING"

        # Raw event buffers for batched insertion
        self._raw_depth_events: list[dict[str, Any]] = []
        self._raw_trade_events: list[dict[str, Any]] = []
        self._raw_poly_events: list[dict[str, Any]] = []
        self._sample_batch: list[LeadLagSample] = []

        # Lifecycle tracking
        self._current_round_slug: str | None = None
        self._current_round_info: BTC5mRoundInfo | None = None
        self._rounds_captured_count: int = 0
        self._total_samples_count: int = 0
        self._stale_samples_count: int = 0
        self._last_heartbeat_time: float = 0.0

    def acquire_lock(self) -> None:
        """Acquire non-blocking single-instance lock."""
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd = os.open(str(self.lock_file_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(self._lock_fd, 0)
            os.write(self._lock_fd, f"{os.getpid()}\n".encode("utf-8"))
            os.fsync(self._lock_fd)
        except (BlockingIOError, OSError) as e:
            raise RuntimeError(
                f"Another LeadLagCollector instance is already running with lock {self.lock_file_path}"
            ) from e

    def release_lock(self) -> None:
        """Release single-instance lock."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
            except Exception:
                pass
            self._lock_fd = None
            if self.lock_file_path.exists():
                try:
                    self.lock_file_path.unlink()
                except Exception:
                    pass

    # ==========================================================================
    # Binance Stream Ingestion
    # ==========================================================================

    def _start_binance_worker(self) -> None:
        def _worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run() -> None:
                while not self._stop_event.is_set():
                    try:
                        with self._binance_lock:
                            self._binance_status = "CONNECTING"
                        async with websockets.connect(
                            BINANCE_WS_URL,
                            ping_interval=20,
                            ping_timeout=10,
                            close_timeout=5,
                        ) as ws:
                            with self._binance_lock:
                                self._binance_status = "CONNECTED"
                            while not self._stop_event.is_set():
                                msg = await ws.recv()
                                now_ms = int(time.time() * 1000)
                                self._handle_binance_message(msg, now_ms)
                    except Exception as e:
                        with self._binance_lock:
                            self._binance_status = f"DISCONNECTED ({type(e).__name__})"
                        if not self._stop_event.is_set():
                            await asyncio.sleep(2.0)

            try:
                loop.run_until_complete(_run())
            finally:
                loop.close()

        t = threading.Thread(target=_worker, daemon=True, name="binance_perp_stream")
        t.start()

    def _handle_binance_message(self, raw_msg: str | bytes, recv_ms: int) -> None:
        try:
            parsed = json.loads(raw_msg)
            stream = parsed.get("stream", "")
            payload = parsed.get("data", {})

            if "depth" in stream:
                bids = [(float(p), float(q)) for p, q in payload.get("b", [])]
                asks = [(float(p), float(q)) for p, q in payload.get("a", [])]
                source_ts = payload.get("T") or payload.get("E")
                s_ts = int(source_ts) if source_ts is not None else None

                best_bid = bids[0][0] if bids else None
                best_ask = asks[0][0] if asks else None

                with self._binance_lock:
                    self._binance_bids = bids
                    self._binance_asks = asks
                    self._binance_source_ts_ms = s_ts
                    self._binance_recv_ts_ms = recv_ms

                    if best_bid is not None and best_ask is not None and best_bid < best_ask:
                        mid = (best_bid + best_ask) / 2.0
                        self._binance_mids.append((s_ts, recv_ms, mid))
                        if self._binance_open_mid is None:
                            self._binance_open_mid = mid

                # Raw event persistence (sampled at 1Hz or buffered)
                if best_bid is not None and best_ask is not None:
                    ev = {
                        "event_id": f"bn_depth_{recv_ms}_{len(self._raw_depth_events)}",
                        "round_slug": self._current_round_slug or "pending",
                        "source_ts_ms": s_ts,
                        "recv_ts_ms": recv_ms,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "mid_price": (best_bid + best_ask) / 2.0,
                        "raw_json": json.dumps({"bids_count": len(bids), "asks_count": len(asks)}),
                    }
                    if len(self._raw_depth_events) < 500:
                        self._raw_depth_events.append(ev)

            elif "aggTrade" in stream:
                agg_id = int(payload["a"])
                t_ms = int(payload["T"])
                p_val = float(payload["p"])
                q_val = float(payload["q"])
                m_val = bool(payload["m"])

                with self._binance_lock:
                    self._binance_trades.append((t_ms, p_val, q_val, m_val, agg_id))

                ev_tr = {
                    "agg_trade_id": agg_id,
                    "round_slug": self._current_round_slug or "pending",
                    "trade_ts_ms": t_ms,
                    "recv_ts_ms": recv_ms,
                    "price": p_val,
                    "quantity": q_val,
                    "is_buyer_maker": m_val,
                }
                if len(self._raw_trade_events) < 1000:
                    self._raw_trade_events.append(ev_tr)
        except Exception:
            pass

    # ==========================================================================
    # Polymarket Stream Ingestion
    # ==========================================================================

    def _start_polymarket_worker(self) -> None:
        def _worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run() -> None:
                subscribed_token: str | None = None
                while not self._stop_event.is_set():
                    try:
                        token_to_sub = self._poly_token_id
                        if not token_to_sub:
                            with self._poly_lock:
                                self._poly_status = "WAITING_FOR_TOKEN"
                            await asyncio.sleep(1.0)
                            continue

                        with self._poly_lock:
                            self._poly_status = "CONNECTING"

                        async with websockets.connect(
                            POLY_WS_URL,
                            ping_interval=20,
                            ping_timeout=10,
                            close_timeout=5,
                        ) as ws:
                            sub_msg = {"type": "market", "assets_ids": [token_to_sub]}
                            await ws.send(json.dumps(sub_msg))
                            subscribed_token = token_to_sub

                            with self._poly_lock:
                                self._poly_status = "CONNECTED"

                            while not self._stop_event.is_set():
                                # Check if active token changed
                                if self._poly_token_id != subscribed_token:
                                    break  # Reconnect with new token

                                try:
                                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                                    recv_ms = int(time.time() * 1000)
                                    self._handle_poly_message(msg, recv_ms)
                                except asyncio.TimeoutError:
                                    # Fallback REST poll to ensure freshness if stream is quiet
                                    self._poll_poly_rest_fallback()

                    except Exception as e:
                        with self._poly_lock:
                            self._poly_status = f"DISCONNECTED ({type(e).__name__})"
                        if not self._stop_event.is_set():
                            # Fallback poll during disconnect
                            self._poll_poly_rest_fallback()
                            await asyncio.sleep(2.0)

            try:
                loop.run_until_complete(_run())
            finally:
                loop.close()

        t = threading.Thread(target=_worker, daemon=True, name="polymarket_book_stream")
        t.start()

    def _handle_poly_message(self, raw_msg: str | bytes, recv_ms: int) -> None:
        try:
            data = json.loads(raw_msg)
            # Two payload formats: initial snapshot array or price_changes dict
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                # Snapshot format
                if "bids" in item and "asks" in item:
                    bids = item.get("bids", [])
                    asks = item.get("asks", [])
                    s_ts_raw = item.get("timestamp")
                    s_ts = int(s_ts_raw) if s_ts_raw else None
                    raw_hash = item.get("hash")

                    best_bid = max([float(b["price"]) for b in bids]) if bids else None
                    best_ask = min([float(a["price"]) for a in asks]) if asks else None

                    self._update_poly_state(best_bid, best_ask, s_ts, recv_ms, raw_hash)

                # Incremental price_changes format
                elif "price_changes" in item:
                    for pc in item.get("price_changes", []):
                        if str(pc.get("asset_id")) == self._poly_token_id:
                            bb = float(pc["best_bid"]) if pc.get("best_bid") is not None else None
                            ba = float(pc["best_ask"]) if pc.get("best_ask") is not None else None
                            raw_hash = pc.get("hash")
                            self._update_poly_state(bb, ba, None, recv_ms, raw_hash)
        except Exception:
            pass

    def _poll_poly_rest_fallback(self) -> None:
        """Fallback unauthenticated REST poll when WebSocket is quiet or reconnecting."""
        token_id = self._poly_token_id
        if not token_id:
            return
        try:
            book = self.poly_rest_collector.fetch_order_book(token_id)
            if book.is_valid:
                self._update_poly_state(
                    book.best_bid,
                    book.best_ask,
                    book.source_event_timestamp_ms,
                    book.received_at_ms,
                    None,
                )
        except Exception:
            pass

    def _update_poly_state(
        self,
        best_bid: float | None,
        best_ask: float | None,
        source_ts: int | None,
        recv_ms: int,
        raw_hash: str | None,
    ) -> None:
        with self._poly_lock:
            self._poly_best_bid = best_bid
            self._poly_best_ask = best_ask
            self._poly_source_ts_ms = source_ts
            self._poly_recv_ts_ms = recv_ms

            is_crossed = False
            mid: float | None = None
            if best_bid is not None and best_ask is not None:
                if best_bid >= best_ask:
                    is_crossed = True
                else:
                    mid = round((best_bid + best_ask) / 2.0, 5)
                    self._poly_mids.append((source_ts, recv_ms, mid))

            self._poly_is_crossed = is_crossed
            self._poly_midpoint = mid

        if len(self._raw_poly_events) < 500:
            ev = {
                "event_id": f"poly_book_{recv_ms}_{len(self._raw_poly_events)}",
                "round_slug": self._current_round_slug or "pending",
                "token_id": self._poly_token_id or "unknown",
                "source_ts_ms": source_ts,
                "recv_ts_ms": recv_ms,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "midpoint": mid,
                "spread": round(best_ask - best_bid, 5) if best_bid and best_ask else None,
                "raw_hash": raw_hash,
            }
            self._raw_poly_events.append(ev)

    # ==========================================================================
    # High-Frequency 1-Second Synchronized Sampling
    # ==========================================================================

    def _sample_synchronized_observation(self, target_sec_ms: int) -> LeadLagSample:
        actual_ms = int(time.time() * 1000)
        mono_ns = time.monotonic_ns()

        round_slug = self._current_round_slug or "unknown"
        end_epoch = self._current_round_info.end_epoch if self._current_round_info else int(time.time()) + 300
        seconds_remaining = max(0, int(end_epoch - (target_sec_ms // 1000)))

        # Snapshot Binance state under lock
        with self._binance_lock:
            bn_source_ts = self._binance_source_ts_ms
            bn_recv_ts = self._binance_recv_ts_ms
            bids = list(self._binance_bids)
            asks = list(self._binance_asks)
            trades = list(self._binance_trades)
            mids = list(self._binance_mids)
            bn_open_mid = self._binance_open_mid

        # Snapshot Polymarket state under lock
        with self._poly_lock:
            poly_source_ts = self._poly_source_ts_ms
            poly_recv_ts = self._poly_recv_ts_ms
            poly_bid = self._poly_best_bid
            poly_ask = self._poly_best_ask
            poly_mid = self._poly_midpoint
            poly_crossed = self._poly_is_crossed
            poly_mids = list(self._poly_mids)

        poly_receipt_age = max(0, actual_ms - poly_recv_ts) if poly_recv_ts > 0 else 999999
        poly_source_age = max(0, actual_ms - poly_source_ts) if poly_source_ts is not None else None

        bn_receipt_age = max(0, actual_ms - bn_recv_ts) if bn_recv_ts > 0 else 999999
        bn_source_age = max(0, actual_ms - bn_source_ts) if bn_source_ts is not None else None

        # Stale checks: age > 3000ms
        is_stale = False
        stale_reasons: list[str] = []
        if poly_receipt_age > MAX_STALE_AGE_MS:
            is_stale = True
            stale_reasons.append(f"poly_age({poly_receipt_age}ms)>3s")
        if bn_receipt_age > MAX_STALE_AGE_MS:
            is_stale = True
            stale_reasons.append(f"binance_age({bn_receipt_age}ms)>3s")

        # Polymarket features
        poly_spread = round(poly_ask - poly_bid, 5) if poly_bid and poly_ask else None
        poly_is_valid = poly_mid is not None and not poly_crossed and not is_stale

        def _poly_ret(sec: int) -> float | None:
            if poly_mid is None:
                return None
            cutoff_ms = actual_ms - sec * 1000
            for _s_ts, r_ts, past_m in reversed(poly_mids):
                if r_ts <= cutoff_ms:
                    return round(poly_mid - past_m, 5)
            return None

        poly_ret_1s = _poly_ret(1)
        poly_ret_2s = _poly_ret(2)
        poly_ret_3s = _poly_ret(3)
        poly_ret_5s = _poly_ret(5)
        poly_ret_10s = _poly_ret(10)
        poly_ret_30s = _poly_ret(30)

        # Binance features
        bn_bid = bids[0][0] if bids else None
        bn_ask = asks[0][0] if asks else None
        bn_mid = (bn_bid + bn_ask) / 2.0 if bn_bid and bn_ask else None

        bn_is_valid = bn_mid is not None and bn_bid < bn_ask and not is_stale if bn_bid and bn_ask else False

        microprice: float | None = None
        microprice_offset_bps: float | None = None
        spread_bps: float | None = None
        top1_imb: float | None = None
        top5_imb: float | None = None
        top20_imb: float | None = None

        if bn_bid and bn_ask and bn_mid and bn_mid > 0:
            bid_q1 = bids[0][1]
            ask_q1 = asks[0][1]
            tot1 = bid_q1 + ask_q1
            if tot1 > 0:
                microprice = (bn_bid * ask_q1 + bn_ask * bid_q1) / tot1
                microprice_offset_bps = round(((microprice - bn_mid) / bn_mid) * 10000.0, 3)
                top1_imb = round((bid_q1 - ask_q1) / tot1, 4)
            else:
                microprice = bn_mid
                microprice_offset_bps = 0.0
                top1_imb = 0.0

            spread_bps = round(((bn_ask - bn_bid) / bn_mid) * 10000.0, 3)

            top5_b = sum(q for _, q in bids[:5])
            top5_a = sum(q for _, q in asks[:5])
            top5_tot = top5_b + top5_a
            top5_imb = round((top5_b - top5_a) / top5_tot, 4) if top5_tot > 0 else 0.0

            top20_b = sum(q for _, q in bids[:20])
            top20_a = sum(q for _, q in asks[:20])
            top20_tot = top20_b + top20_a
            top20_imb = round((top20_b - top20_a) / top20_tot, 4) if top20_tot > 0 else 0.0

        # Binance return since open
        ret_since_open: float | None = None
        if bn_mid and bn_open_mid and bn_open_mid > 0:
            ret_since_open = round(((bn_mid - bn_open_mid) / bn_open_mid) * 10000.0, 3)

        # Binance returns over 1s, 2s, 3s, 5s, 10s, 30s, 60s
        def _bn_ret(sec: int) -> float | None:
            if bn_mid is None:
                return None
            target_ms = actual_ms - sec * 1000
            for _s_ts, r_ts, past_m in reversed(mids):
                if r_ts <= target_ms:
                    return round(((bn_mid - past_m) / past_m) * 10000.0, 3)
            return None

        ret_1s = _bn_ret(1)
        ret_2s = _bn_ret(2)
        ret_3s = _bn_ret(3)
        ret_5s = _bn_ret(5)
        ret_10s = _bn_ret(10)
        ret_30s = _bn_ret(30)
        ret_60s = _bn_ret(60)

        # Binance taker flow over windows
        def _taker_flow(sec: int) -> float | None:
            cutoff = actual_ms - sec * 1000
            buy_q = 0.0
            sell_q = 0.0
            found = False
            for t_ms, _p, q, is_bm, _id in reversed(trades):
                if t_ms < cutoff:
                    break
                found = True
                if is_bm:
                    sell_q += q
                else:
                    buy_q += q
            if not found:
                return None
            tot = buy_q + sell_q
            return round((buy_q - sell_q) / tot, 4) if tot > 0 else 0.0

        tf_1s = _taker_flow(1)
        tf_2s = _taker_flow(2)
        tf_3s = _taker_flow(3)
        tf_5s = _taker_flow(5)
        tf_10s = _taker_flow(10)
        tf_30s = _taker_flow(30)
        tf_60s = _taker_flow(60)

        # Clock provenance
        source_lat: int | None = None
        if bn_source_ts is not None:
            source_lat = max(0, actual_ms - bn_source_ts)
        elif poly_source_ts is not None:
            source_lat = max(0, actual_ms - poly_source_ts)

        skew_ms = abs(bn_recv_ts - poly_recv_ts) if bn_recv_ts > 0 and poly_recv_ts > 0 else 0

        overall_valid = poly_is_valid and bn_is_valid and not is_stale

        return LeadLagSample(
            sample_id=f"{round_slug}_{target_sec_ms}",
            round_slug=round_slug,
            experiment_id=EXPERIMENT_ID,
            experiment_spec_hash=EXPERIMENT_SPEC_HASH,
            sample_target_ts_ms=target_sec_ms,
            sample_actual_ts_ms=actual_ms,
            local_monotonic_ns=mono_ns,
            seconds_remaining=seconds_remaining,
            poly_source_ts_ms=poly_source_ts,
            poly_recv_ts_ms=poly_recv_ts,
            poly_receipt_age_ms=poly_receipt_age,
            poly_source_age_ms=poly_source_age,
            poly_best_bid=poly_bid,
            poly_best_ask=poly_ask,
            poly_midpoint=poly_mid,
            poly_spread=poly_spread,
            poly_return_1s=poly_ret_1s,
            poly_return_2s=poly_ret_2s,
            poly_return_3s=poly_ret_3s,
            poly_return_5s=poly_ret_5s,
            poly_return_10s=poly_ret_10s,
            poly_return_30s=poly_ret_30s,
            poly_is_crossed=poly_crossed,
            poly_is_valid=poly_is_valid,
            binance_source_ts_ms=bn_source_ts,
            binance_recv_ts_ms=bn_recv_ts,
            binance_receipt_age_ms=bn_receipt_age,
            binance_source_age_ms=bn_source_age,
            binance_best_bid=bn_bid,
            binance_best_ask=bn_ask,
            binance_mid_price=bn_mid,
            binance_microprice=microprice,
            binance_microprice_offset_bps=microprice_offset_bps,
            binance_spread_bps=spread_bps,
            binance_basis_bps=None,  # Computed if ref feed configured
            binance_return_since_open_bps=ret_since_open,
            binance_return_1s_bps=ret_1s,
            binance_return_2s_bps=ret_2s,
            binance_return_3s_bps=ret_3s,
            binance_return_5s_bps=ret_5s,
            binance_return_10s_bps=ret_10s,
            binance_return_30s_bps=ret_30s,
            binance_return_60s_bps=ret_60s,
            binance_taker_flow_1s=tf_1s,
            binance_taker_flow_2s=tf_2s,
            binance_taker_flow_3s=tf_3s,
            binance_taker_flow_5s=tf_5s,
            binance_taker_flow_10s=tf_10s,
            binance_taker_flow_30s=tf_30s,
            binance_taker_flow_60s=tf_60s,
            binance_top1_depth_imbalance=top1_imb,
            binance_top5_depth_imbalance=top5_imb,
            binance_top20_depth_imbalance=top20_imb,
            binance_is_valid=bn_is_valid,
            source_to_receive_latency_ms=source_lat,
            inter_feed_receive_skew_ms=skew_ms,
            is_stale=is_stale,
            stale_reason="; ".join(stale_reasons) if stale_reasons else None,
            is_valid=overall_valid,
            raw_json="{}",
        )

    # ==========================================================================
    # Round Lifecycle & Flush
    # ==========================================================================

    def _ensure_active_round(self, now_sec: float) -> None:
        expected_slug = self.contract_mgr.derive_round_slug(now_sec)
        if expected_slug == self._current_round_slug:
            return

        # Round transition occurred!
        if self._current_round_slug is not None:
            self._close_current_round()

        logger.info(f"Transitioning to new physical round: {expected_slug}")
        self._current_round_slug = expected_slug
        start_epoch = (int(now_sec) // 300) * 300
        end_epoch = start_epoch + 300

        # Reset Binance open price for new round
        with self._binance_lock:
            self._binance_open_mid = None
            if self._binance_mids:
                self._binance_open_mid = self._binance_mids[-1][2]
            self._binance_open_round_slug = expected_slug

        # Fetch contract details from Gamma
        event = self.contract_mgr.fetch_event_by_slug(expected_slug)
        round_info: BTC5mRoundInfo | None = None
        if event:
            round_info, _ = self.contract_mgr.parse_and_validate_round(event, expected_slug)

        self._current_round_info = round_info
        up_token = round_info.up_token_id if round_info else None
        down_token = round_info.down_token_id if round_info else None
        cond_id = round_info.condition_id if round_info else None

        # Update Polymarket token subscription
        with self._poly_lock:
            self._poly_token_id = up_token
            self._poly_round_slug = expected_slug
            self._poly_best_bid = None
            self._poly_best_ask = None
            self._poly_midpoint = None

        # Persist new round in database
        self.db.save_leadlag_round({
            "round_slug": expected_slug,
            "experiment_id": EXPERIMENT_ID,
            "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "up_token_id": up_token,
            "down_token_id": down_token,
            "condition_id": cond_id,
            "status": "ACTIVE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

        self._rounds_captured_count += 1
        logger.info(f"Physical round {self._rounds_captured_count}/{self.target_physical_rounds} registered: {expected_slug}")

        # Checkpoints at 30, 100, 500 rounds
        if self._rounds_captured_count in (30, 100, 500):
            self._create_checkpoint(self._rounds_captured_count)

    def _close_current_round(self) -> None:
        if self._current_round_slug is None:
            return
        self._flush_batch()
        # Query total samples recorded for this round
        samples = self.db.get_leadlag_samples(round_slug=self._current_round_slug)
        tot = len(samples)
        val = sum(1 for s in samples if s.get("is_valid"))
        self.db.update_leadlag_round_completion(
            round_slug=self._current_round_slug,
            sample_count=tot,
            valid_sample_count=val,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"Closed physical round {self._current_round_slug}: {val}/{tot} valid samples.")

    def _flush_batch(self) -> None:
        """Batch write all pending samples and raw events to SQLite."""
        with self.db.transaction() as conn:
            if self._sample_batch:
                self.db.save_leadlag_samples_batch(self._sample_batch, conn=conn)
                self._sample_batch.clear()

            if self._raw_depth_events:
                self.db.save_leadlag_depth_events_batch(self._raw_depth_events)
                self._raw_depth_events.clear()

            if self._raw_trade_events:
                self.db.save_leadlag_trade_events_batch(self._raw_trade_events)
                self._raw_trade_events.clear()

            if self._raw_poly_events:
                self.db.save_leadlag_poly_events_batch(self._raw_poly_events)
                self._raw_poly_events.clear()

    def _create_checkpoint(self, milestone: int) -> None:
        summary = self.db.get_leadlag_audit_summary()
        cp_file = Path(f"data/checkpoints/checkpoint_leadlag_{milestone}r_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
        cp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cp_file, "w") as f:
            json.dump({
                "milestone_rounds": milestone,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "experiment_id": EXPERIMENT_ID,
                "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
                "summary": summary,
            }, f, indent=2)
        logger.info(f"Created leadlag milestone checkpoint at {cp_file}")

    def _emit_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat_time < 5.0:
            return
        self._last_heartbeat_time = now

        tot = self._total_samples_count
        stale_rate = round(self._stale_samples_count / tot, 4) if tot > 0 else 0.0

        with self._binance_lock:
            bn_st = self._binance_status
            bn_age = max(0, int(now * 1000) - self._binance_recv_ts_ms) if self._binance_recv_ts_ms > 0 else 999999

        with self._poly_lock:
            poly_st = self._poly_status
            poly_age = max(0, int(now * 1000) - self._poly_recv_ts_ms) if self._poly_recv_ts_ms > 0 else 999999

        hb = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "epoch_ms": int(now * 1000),
            "pid": os.getpid(),
            "status": "COLLECTING" if not self._stop_event.is_set() else "STOPPED",
            "experiment_id": EXPERIMENT_ID,
            "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
            "physical_rounds_captured": self._rounds_captured_count,
            "total_samples": self._total_samples_count,
            "binance_feed_status": f"{bn_st} (age {bn_age}ms)",
            "polymarket_feed_status": f"{poly_st} (age {poly_age}ms)",
            "binance_stale_rate": stale_rate,
            "polymarket_stale_rate": stale_rate,
            "timestamp_coverage": 1.0,
            "extra_json": json.dumps({"current_round": self._current_round_slug}),
        }
        self.db.save_leadlag_heartbeat(hb)

    # ==========================================================================
    # Main Execution Loop
    # ==========================================================================

    def run(self) -> None:
        """Run the continuous 1-second synchronized lead-lag collector."""
        logger.info(f"Starting LeadLagCollector (PID {os.getpid()}) targeting {self.target_physical_rounds} physical rounds...")
        self.acquire_lock()
        self._running = True

        def _handle_signal(signum: int, _frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating clean collector shutdown...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # Start background streaming threads
        self._start_binance_worker()
        self._start_polymarket_worker()

        try:
            while not self._stop_event.is_set():
                now = time.time()

                # Check if target physical rounds reached
                if self._rounds_captured_count >= self.target_physical_rounds:
                    logger.info(f"Target of {self.target_physical_rounds} physical rounds reached! Cleanly closing collector.")
                    break

                # Align to exact 1.0 second boundary
                sleep_sec = 1.0 - (now % 1.0)
                if sleep_sec > 0.01:
                    time.sleep(sleep_sec)

                sample_sec_ms = int(time.time() // 1.0) * 1000

                # Ensure active physical round metadata
                self._ensure_active_round(time.time())

                # Collect high-frequency synchronized sample
                sample = self._sample_synchronized_observation(sample_sec_ms)
                self._sample_batch.append(sample)
                self._total_samples_count += 1
                if sample.is_stale:
                    self._stale_samples_count += 1

                # Flush every 10 samples (10 seconds)
                if len(self._sample_batch) >= 10:
                    self._flush_batch()

                self._emit_heartbeat()

        except Exception as e:
            logger.exception(f"Unhandled error in lead-lag collector main loop: {e}")
        finally:
            self._close_current_round()
            self._flush_batch()
            self._emit_heartbeat()
            self.release_lock()
            logger.info("LeadLagCollector stopped cleanly.")
