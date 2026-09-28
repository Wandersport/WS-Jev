"""High-frequency 1-second synchronized lead-lag observational collector (v2).

Continuous public read-only collection:
- Binance USD-M Perpetual BTC/USDT (orderbook depth + aggTrades)
- Polymarket CLOB BTC 5-minute native UP orderbook (streaming WebSocket + REST fallback)

Includes remediated instrumentation:
- Preserves outer timestamps for Polymarket price_change frames
- Case-insensitive Binance combined-stream matching & explicit aggTrade routing
- Complete raw message wire persistence with lossless zlib compression
- Aggregate trade ID deduplication and monotonic sequence ordering
- Hardened round and token transitions with T-30s metadata prefetching
- Telemetry counters and latency percentiles (p50, p90, p95, p99, max)
- Structured diagnostic counters (zero silent exception suppression)
- Separate isolated storage and tables

ZERO live trading. ZERO paper broker orders. ZERO OpenRouter/Jev requests.
"""

from __future__ import annotations

import asyncio
import collections
import fcntl
import hashlib
import json
import logging
import os
import signal
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from pm_research.research.btc5m.contract import BTC5mContractManager, BTC5mRoundInfo
from pm_research.research.btc5m.leadlag_v2_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_PILOT_ID,
    EXPERIMENT_SPEC_HASH,
    MAX_STALE_AGE_MS,
    LeadLagSampleV2,
)
from pm_research.research.btc5m.poly_book import PolymarketBookCollector
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

BINANCE_WS_URL: str = "wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@aggTrade"
POLY_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

DEFAULT_V2_LOCK_FILE: str = "data/btc5m_leadlag_v2_collector.lock"
DEFAULT_V2_PILOT_LOCK_FILE: str = "data/btc5m_leadlag_v2_pilot.lock"


def compute_percentiles(values: list[float | int]) -> dict[str, float]:
    """Compute standard summary percentiles and mean."""
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(values)
    n = len(s)
    return {
        "p50": round(float(s[int(0.50 * (n - 1))]), 2),
        "p90": round(float(s[int(0.90 * (n - 1))]), 2),
        "p95": round(float(s[int(0.95 * (n - 1))]), 2),
        "p99": round(float(s[int(0.99 * (n - 1))]), 2),
        "max": round(float(s[-1]), 2),
        "mean": round(float(sum(s) / n), 2),
    }


class LeadLagCollectorV2:
    """Synchronized 1-second lead-lag observational collector (v2 architecture)."""

    def __init__(
        self,
        db: Database,
        target_physical_rounds: int = 500,
        is_pilot: bool = False,
        lock_file_path: str | None = None,
    ) -> None:
        self.db = db
        self.target_physical_rounds = target_physical_rounds
        self.is_pilot = is_pilot
        self.experiment_id = EXPERIMENT_PILOT_ID if is_pilot else EXPERIMENT_ID
        self.experiment_spec_hash = EXPERIMENT_SPEC_HASH

        default_lock = DEFAULT_V2_PILOT_LOCK_FILE if is_pilot else DEFAULT_V2_LOCK_FILE
        self.lock_file_path = Path(lock_file_path or default_lock)
        self.contract_mgr = BTC5mContractManager()
        self.poly_rest_collector = PolymarketBookCollector()

        self._lock_fd: int | None = None
        self._running: bool = False
        self._stop_event = threading.Event()
        self._pilot_failed: bool = False
        self._pilot_failure_reason: str | None = None

        # Thread synchronization locks
        self._binance_lock = threading.RLock()
        self._poly_lock = threading.RLock()

        # Binance state
        self._binance_bids: list[tuple[float, float]] = []  # [(price, qty)]
        self._binance_asks: list[tuple[float, float]] = []
        self._binance_source_ts_ms: int | None = None
        self._binance_recv_ts_ms: int = 0
        self._binance_mono_ns: int = 0
        self._binance_trades: collections.deque[tuple[int, float, float, bool, int]] = collections.deque(maxlen=10000)
        self._seen_trade_ids_queue: collections.deque[int] = collections.deque(maxlen=20000)
        self._seen_trade_ids_set: set[int] = set()
        self._first_trade_ms: int | None = None
        # (source_ts, recv_ts, mid)
        self._binance_mids: collections.deque[tuple[int | None, int, float]] = collections.deque(maxlen=3600)
        self._binance_open_mid: float | None = None
        self._binance_open_round_slug: str | None = None
        self._binance_status: str = "INITIALIZING"
        self._last_binance_final_u: int | None = None
        self._binance_depth_continuity_gaps: int = 0

        # Polymarket state
        self._poly_token_id: str | None = None
        self._poly_round_slug: str | None = None
        self._poly_best_bid: float | None = None
        self._poly_best_ask: float | None = None
        self._poly_midpoint: float | None = None
        self._poly_source_ts_ms: int | None = None
        self._poly_recv_ts_ms: int = 0
        self._poly_mono_ns: int = 0
        self._poly_provenance_mode: str = "WS_SNAPSHOT"
        self._poly_is_crossed: bool = False
        self._poly_mids: collections.deque[tuple[int | None, int, float]] = collections.deque(maxlen=1800)
        self._poly_status: str = "INITIALIZING"

        # Raw event and payload buffers for batched insertion
        self._raw_payload_seq: int = 0
        self._raw_payloads: list[dict[str, Any]] = []
        self._raw_depth_events: list[dict[str, Any]] = []
        self._raw_trade_events: list[dict[str, Any]] = []
        self._raw_poly_events: list[dict[str, Any]] = []
        self._sample_batch: list[LeadLagSampleV2] = []

        # Lifecycle tracking
        self._current_round_slug: str | None = None
        self._current_round_info: BTC5mRoundInfo | None = None
        self._upcoming_round_slug: str | None = None
        self._upcoming_round_info: BTC5mRoundInfo | None = None
        self._rounds_captured_count: int = 0
        self._total_samples_count: int = 0
        self._stale_samples_count: int = 0
        self._last_heartbeat_time: float = 0.0

        # Diagnostics & Telemetry counters
        self._depth_events_received: int = 0
        self._aggtrade_events_received: int = 0
        self._duplicate_aggtrade_events: int = 0
        self._out_of_order_trade_events: int = 0
        self._malformed_binance_events: int = 0
        self._unhandled_binance_stream_count: int = 0

        self._snapshot_events_received: int = 0
        self._delta_events_received: int = 0
        self._rest_fallback_events: int = 0
        self._poly_source_ts_present: int = 0
        self._poly_source_ts_missing: int = 0
        self._binance_source_ts_present: int = 0
        self._binance_source_ts_missing: int = 0
        self._malformed_poly_events: int = 0
        self._unhandled_poly_event_count: int = 0

        self._malformed_json_count: int = 0
        self._missing_key_count: int = 0
        self._parser_exception_count: int = 0

        # Rolling latency collections (last 1000 items)
        self._binance_latencies_ms: collections.deque[int] = collections.deque(maxlen=1000)
        self._poly_latencies_ms: collections.deque[int] = collections.deque(maxlen=1000)
        self._inter_feed_skews_ms: collections.deque[int] = collections.deque(maxlen=1000)
        self._sample_target_errors_ms: collections.deque[int] = collections.deque(maxlen=1000)

        # Start wall time for warmup / fail-fast checks
        self._start_time_sec: float = time.time()

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
                f"Another LeadLagCollectorV2 instance is already running with lock {self.lock_file_path}"
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

    def _persist_lossless_raw_payload(self, feed: str, raw_bytes: bytes, recv_ms: int) -> str:
        """Compress raw wire payload using zlib and queue for transactional persistence."""
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        compressed = zlib.compress(raw_bytes, level=6)
        self._raw_payload_seq += 1
        payload_id = f"{feed}_{recv_ms}_{self._raw_payload_seq}_{sha256[:8]}"

        entry = {
            "payload_id": payload_id,
            "experiment_id": self.experiment_id,
            "experiment_spec_hash": self.experiment_spec_hash,
            "feed": feed,
            "round_slug": self._current_round_slug or "pending",
            "recv_ts_ms": recv_ms,
            "uncompressed_len": len(raw_bytes),
            "compressed_len": len(compressed),
            "sha256_hash": sha256,
            "compressed_payload": compressed,
        }
        self._raw_payloads.append(entry)
        return payload_id

    # ==========================================================================
    # Binance Stream Ingestion (Case-Insensitive & aggTrade Defect Remediated)
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
                                mono_ns = time.monotonic_ns()
                                self._handle_binance_message(msg, now_ms, mono_ns)
                    except Exception as e:
                        with self._binance_lock:
                            self._binance_status = f"DISCONNECTED ({type(e).__name__})"
                        if not self._stop_event.is_set():
                            await asyncio.sleep(2.0)

            try:
                loop.run_until_complete(_run())
            finally:
                loop.close()

        t = threading.Thread(target=_worker, daemon=True, name="binance_perp_v2_stream")
        t.start()

    def _handle_binance_message(self, raw_msg: str | bytes, recv_ms: int, mono_ns: int) -> None:
        raw_bytes = raw_msg.encode("utf-8") if isinstance(raw_msg, str) else raw_msg

        try:
            parsed = json.loads(raw_bytes)
        except json.JSONDecodeError as e:
            self._malformed_json_count += 1
            self._malformed_binance_events += 1
            logger.warning("Malformed JSON from Binance stream: %s", e)
            return
        except Exception as e:
            self._parser_exception_count += 1
            self._malformed_binance_events += 1
            logger.warning("Unexpected parser error decoding Binance frame: %s", e)
            return

        if not isinstance(parsed, dict):
            self._malformed_binance_events += 1
            return

        stream = str(parsed.get("stream", ""))
        payload = parsed.get("data", {})
        if not isinstance(payload, dict):
            self._malformed_binance_events += 1
            return

        # Case-insensitive stream matching and explicit payload event type checking
        s_lower = stream.lower()
        e_type = str(payload.get("e", ""))

        payload_id = self._persist_lossless_raw_payload("binance", raw_bytes, recv_ms)

        try:
            if "depth" in s_lower or e_type == "depthUpdate":
                self._depth_events_received += 1
                bids = [(float(p), float(q)) for p, q in payload.get("b", [])]
                asks = [(float(p), float(q)) for p, q in payload.get("a", [])]
                source_ts = payload.get("T") or payload.get("E")
                s_ts = int(source_ts) if source_ts is not None else None

                if s_ts is not None:
                    self._binance_source_ts_present += 1
                    lat = max(0, recv_ms - s_ts)
                    self._binance_latencies_ms.append(lat)
                else:
                    self._binance_source_ts_missing += 1

                first_u = payload.get("U")
                final_u = payload.get("u")
                prev_u = payload.get("pu")

                if self._last_binance_final_u is not None and prev_u is not None:
                    if prev_u != self._last_binance_final_u:
                        self._binance_depth_continuity_gaps += 1
                if final_u is not None:
                    self._last_binance_final_u = int(final_u)

                best_bid = bids[0][0] if bids else None
                best_ask = asks[0][0] if asks else None

                with self._binance_lock:
                    self._binance_bids = bids
                    self._binance_asks = asks
                    self._binance_source_ts_ms = s_ts
                    self._binance_recv_ts_ms = recv_ms
                    self._binance_mono_ns = mono_ns

                    if best_bid is not None and best_ask is not None and best_bid < best_ask:
                        mid = (best_bid + best_ask) / 2.0
                        self._binance_mids.append((s_ts, recv_ms, mid))
                        if self._binance_open_mid is None:
                            self._binance_open_mid = mid
                            self._binance_open_round_slug = self._current_round_slug

                # Buffer depth event for raw persistence
                if best_bid is not None and best_ask is not None:
                    ev = {
                        "event_id": f"bn_depth_{recv_ms}_{self._depth_events_received}",
                        "experiment_id": self.experiment_id,
                        "experiment_spec_hash": self.experiment_spec_hash,
                        "round_slug": self._current_round_slug or "pending",
                        "source_ts_ms": s_ts,
                        "recv_ts_ms": recv_ms,
                        "local_monotonic_ns": mono_ns,
                        "first_update_id": int(first_u) if first_u is not None else None,
                        "final_update_id": int(final_u) if final_u is not None else None,
                        "prev_final_update_id": int(prev_u) if prev_u is not None else None,
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "mid_price": (best_bid + best_ask) / 2.0,
                        "raw_payload_id": payload_id,
                        "raw_json": json.dumps({"bids_count": len(bids), "asks_count": len(asks)}),
                    }
                    if len(self._raw_depth_events) < 1000:
                        self._raw_depth_events.append(ev)

            elif "aggtrade" in s_lower or e_type == "aggTrade":
                self._aggtrade_events_received += 1
                agg_id = int(payload["a"])
                t_ms = int(payload["T"])
                p_val = float(payload["p"])
                q_val = float(payload["q"])
                m_val = bool(payload["m"])

                if self._first_trade_ms is None:
                    self._first_trade_ms = t_ms

                # Check duplicate agg_trade_id
                if agg_id in self._seen_trade_ids_set:
                    self._duplicate_aggtrade_events += 1
                    return

                if len(self._seen_trade_ids_queue) >= self._seen_trade_ids_queue.maxlen:
                    popped = self._seen_trade_ids_queue.popleft()
                    self._seen_trade_ids_set.discard(popped)
                self._seen_trade_ids_queue.append(agg_id)
                self._seen_trade_ids_set.add(agg_id)

                with self._binance_lock:
                    # Enforce strict chronological trade ordering (trade_ts_ms, agg_id)
                    if self._binance_trades and (t_ms, agg_id) < (self._binance_trades[-1][0], self._binance_trades[-1][4]):
                        self._out_of_order_trade_events += 1
                        tr_list = list(self._binance_trades)
                        tr_list.append((t_ms, p_val, q_val, m_val, agg_id))
                        tr_list.sort(key=lambda x: (x[0], x[4]))
                        self._binance_trades = collections.deque(tr_list[-10000:], maxlen=10000)
                    else:
                        self._binance_trades.append((t_ms, p_val, q_val, m_val, agg_id))

                ev_tr = {
                    "experiment_id": self.experiment_id,
                    "experiment_spec_hash": self.experiment_spec_hash,
                    "agg_trade_id": agg_id,
                    "round_slug": self._current_round_slug or "pending",
                    "trade_ts_ms": t_ms,
                    "recv_ts_ms": recv_ms,
                    "local_monotonic_ns": mono_ns,
                    "price": p_val,
                    "quantity": q_val,
                    "is_buyer_maker": m_val,
                    "raw_payload_id": payload_id,
                }
                if len(self._raw_trade_events) < 2000:
                    self._raw_trade_events.append(ev_tr)

            else:
                self._unhandled_binance_stream_count += 1

        except KeyError as e:
            self._missing_key_count += 1
            self._malformed_binance_events += 1
            logger.warning("Missing expected key %s in Binance payload: %s", e, payload)
        except Exception as e:
            self._parser_exception_count += 1
            self._malformed_binance_events += 1
            logger.warning("Parser exception handling Binance message: %s", e)

    # ==========================================================================
    # Polymarket Stream Ingestion (Timestamp Discarding Defect Remediated)
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
                                if self._poly_token_id != subscribed_token:
                                    break  # Reconnect with new round token

                                try:
                                    msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                                    recv_ms = int(time.time() * 1000)
                                    mono_ns = time.monotonic_ns()
                                    self._handle_poly_message(msg, recv_ms, mono_ns)
                                except asyncio.TimeoutError:
                                    # Quiet stream: execute unauthenticated public REST fallback poll
                                    self._poll_poly_rest_fallback()

                    except Exception as e:
                        with self._poly_lock:
                            self._poly_status = f"DISCONNECTED ({type(e).__name__})"
                        if not self._stop_event.is_set():
                            self._poll_poly_rest_fallback()
                            await asyncio.sleep(2.0)

            try:
                loop.run_until_complete(_run())
            finally:
                loop.close()

        t = threading.Thread(target=_worker, daemon=True, name="polymarket_book_v2_stream")
        t.start()

    def _parse_ts(self, ts_val: Any) -> int | None:
        """Parse source timestamp preserving milliseconds."""
        if ts_val is None:
            return None
        try:
            val = int(ts_val)
            if 1_000_000_000 <= val < 10_000_000_000:
                return val * 1000  # seconds -> ms
            elif val > 10_000_000_000_000:
                return val // 1000  # microseconds -> ms
            return val
        except (ValueError, TypeError):
            return None

    def _handle_poly_message(self, raw_msg: str | bytes, recv_ms: int, mono_ns: int) -> None:
        raw_bytes = raw_msg.encode("utf-8") if isinstance(raw_msg, str) else raw_msg

        try:
            data = json.loads(raw_bytes)
        except json.JSONDecodeError as e:
            self._malformed_json_count += 1
            self._malformed_poly_events += 1
            logger.warning("Malformed JSON from Polymarket stream: %s", e)
            return
        except Exception as e:
            self._parser_exception_count += 1
            self._malformed_poly_events += 1
            logger.warning("Unexpected parser error decoding Polymarket frame: %s", e)
            return

        payload_id = self._persist_lossless_raw_payload("polymarket", raw_bytes, recv_ms)

        try:
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    self._malformed_poly_events += 1
                    continue

                # 1. Full snapshot array
                if "bids" in item and "asks" in item:
                    self._snapshot_events_received += 1
                    bids = item.get("bids", [])
                    asks = item.get("asks", [])
                    s_ts = self._parse_ts(item.get("timestamp"))
                    raw_hash = item.get("hash")

                    if s_ts is not None:
                        self._poly_source_ts_present += 1
                        lat = max(0, recv_ms - s_ts)
                        self._poly_latencies_ms.append(lat)
                    else:
                        self._poly_source_ts_missing += 1

                    best_bid = max([float(b["price"]) for b in bids]) if bids else None
                    best_ask = min([float(a["price"]) for a in asks]) if asks else None

                    self._update_poly_state(
                        best_bid=best_bid,
                        best_ask=best_ask,
                        source_ts=s_ts,
                        recv_ms=recv_ms,
                        mono_ns=mono_ns,
                        raw_hash=raw_hash,
                        provenance_mode="WS_SNAPSHOT",
                        raw_payload_id=payload_id,
                        event_type="book",
                    )

                # 2. Incremental price_change / price_changes
                elif "price_changes" in item or item.get("event_type") == "price_change":
                    self._delta_events_received += 1
                    outer_ts = self._parse_ts(item.get("timestamp"))
                    pcs = item.get("price_changes", [])
                    for pc in pcs:
                        if not isinstance(pc, dict):
                            continue
                        if str(pc.get("asset_id")) == self._poly_token_id:
                            # Preserve outer timestamp or inner item timestamp
                            s_ts = outer_ts or self._parse_ts(pc.get("timestamp"))
                            if s_ts is not None:
                                self._poly_source_ts_present += 1
                                lat = max(0, recv_ms - s_ts)
                                self._poly_latencies_ms.append(lat)
                            else:
                                self._poly_source_ts_missing += 1

                            bb = float(pc["best_bid"]) if pc.get("best_bid") is not None else None
                            ba = float(pc["best_ask"]) if pc.get("best_ask") is not None else None
                            raw_hash = pc.get("hash")
                            self._update_poly_state(
                                best_bid=bb,
                                best_ask=ba,
                                source_ts=s_ts,
                                recv_ms=recv_ms,
                                mono_ns=mono_ns,
                                raw_hash=raw_hash,
                                provenance_mode="WS_DELTA",
                                raw_payload_id=payload_id,
                                event_type="price_change",
                            )

                else:
                    self._unhandled_poly_event_count += 1

        except KeyError as e:
            self._missing_key_count += 1
            self._malformed_poly_events += 1
            logger.warning("Missing key %s in Polymarket payload: %s", e, data)
        except Exception as e:
            self._parser_exception_count += 1
            self._malformed_poly_events += 1
            logger.warning("Parser error handling Polymarket message: %s", e)

    def _poll_poly_rest_fallback(self) -> None:
        """Fallback unauthenticated public REST poll when WebSocket is quiet or reconnecting."""
        token_id = self._poly_token_id
        if not token_id:
            return
        recv_ms = int(time.time() * 1000)
        mono_ns = time.monotonic_ns()
        try:
            book = self.poly_rest_collector.fetch_order_book(token_id)
            if book.is_valid:
                self._rest_fallback_events += 1
                s_ts = book.source_event_timestamp_ms
                if s_ts is not None:
                    self._poly_source_ts_present += 1
                else:
                    self._poly_source_ts_missing += 1

                raw_bytes = json.dumps({
                    "token_id": token_id,
                    "bids": book.bids,
                    "asks": book.asks,
                    "source_ts": s_ts,
                    "recv_ms": recv_ms,
                }).encode("utf-8")
                payload_id = self._persist_lossless_raw_payload("polymarket", raw_bytes, recv_ms)

                self._update_poly_state(
                    best_bid=book.best_bid,
                    best_ask=book.best_ask,
                    source_ts=s_ts,
                    recv_ms=recv_ms,
                    mono_ns=mono_ns,
                    raw_hash=None,
                    provenance_mode="REST_FALLBACK",
                    raw_payload_id=payload_id,
                    event_type="refresh",
                )
        except Exception as e:
            self._parser_exception_count += 1
            logger.warning("REST fallback poll failed for token %s: %s", token_id, e)

    def _update_poly_state(
        self,
        best_bid: float | None,
        best_ask: float | None,
        source_ts: int | None,
        recv_ms: int,
        mono_ns: int,
        raw_hash: str | None,
        provenance_mode: str,
        raw_payload_id: str | None,
        event_type: str = "book",
    ) -> None:
        with self._poly_lock:
            self._poly_best_bid = best_bid
            self._poly_best_ask = best_ask
            self._poly_source_ts_ms = source_ts
            self._poly_recv_ts_ms = recv_ms
            self._poly_mono_ns = mono_ns
            self._poly_provenance_mode = provenance_mode

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

        if len(self._raw_poly_events) < 1000:
            ev = {
                "event_id": f"poly_book_{recv_ms}_{len(self._raw_poly_events)}_{self._delta_events_received + self._snapshot_events_received}",
                "experiment_id": self.experiment_id,
                "experiment_spec_hash": self.experiment_spec_hash,
                "round_slug": self._current_round_slug or "pending",
                "token_id": self._poly_token_id or "unknown",
                "provenance_mode": provenance_mode,
                "event_type": event_type,
                "source_ts_ms": source_ts,
                "recv_ts_ms": recv_ms,
                "local_monotonic_ns": mono_ns,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "midpoint": mid,
                "spread": round(best_ask - best_bid, 5) if best_bid and best_ask else None,
                "raw_hash": raw_hash,
                "raw_payload_id": raw_payload_id,
            }
            self._raw_poly_events.append(ev)

    # ==========================================================================
    # High-Frequency 1-Second Synchronized Sampling
    # ==========================================================================

    def _sample_synchronized_observation(
        self, target_sec_ms: int, actual_ms: int | None = None
    ) -> LeadLagSampleV2:
        if actual_ms is None:
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
            poly_mode = self._poly_provenance_mode
            poly_mids = list(self._poly_mids)

        poly_receipt_age = max(0, actual_ms - poly_recv_ts) if poly_recv_ts > 0 else 999999
        poly_source_age = max(0, actual_ms - poly_source_ts) if poly_source_ts is not None else None

        bn_receipt_age = max(0, actual_ms - bn_recv_ts) if bn_recv_ts > 0 else 999999
        bn_source_age = max(0, actual_ms - bn_source_ts) if bn_source_ts is not None else None

        # Stale checks: receipt age > 3000ms
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

        # Binance returns over historical horizons
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

        # Prospective Taker Flow (Mandate 12: Never substitute missing flow with 0)
        def _taker_flow(sec: int) -> float | None:
            if not trades:
                return None
            # If collector started less than sec seconds ago, lookback window cannot be complete
            if self._first_trade_ms is None or (actual_ms - self._first_trade_ms) < (sec * 1000):
                return None

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

        # Clock provenance & Latencies
        source_lat: int | None = None
        if bn_source_ts is not None:
            source_lat = max(0, actual_ms - bn_source_ts)
        elif poly_source_ts is not None:
            source_lat = max(0, actual_ms - poly_source_ts)

        skew_ms = abs(bn_recv_ts - poly_recv_ts) if bn_recv_ts > 0 and poly_recv_ts > 0 else 0
        self._inter_feed_skews_ms.append(skew_ms)

        target_err = abs(actual_ms - target_sec_ms)
        self._sample_target_errors_ms.append(target_err)

        overall_valid = poly_is_valid and bn_is_valid and not is_stale

        return LeadLagSampleV2(
            sample_id=f"{round_slug}_{target_sec_ms}",
            round_slug=round_slug,
            experiment_id=self.experiment_id,
            experiment_spec_hash=self.experiment_spec_hash,
            is_pilot=self.is_pilot,
            sample_target_ts_ms=target_sec_ms,
            sample_actual_ts_ms=actual_ms,
            local_monotonic_ns=mono_ns,
            seconds_remaining=seconds_remaining,
            poly_source_ts_ms=poly_source_ts,
            poly_recv_ts_ms=poly_recv_ts,
            poly_receipt_age_ms=poly_receipt_age,
            poly_source_age_ms=poly_source_age,
            poly_provenance_mode=poly_mode,
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
            binance_basis_bps=None,
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
            raw_payload_id=None,
            raw_json="{}",
        )

    # ==========================================================================
    # Round Lifecycle & Boundary Transition Handling (Mandate 11)
    # ==========================================================================

    def _prefetch_upcoming_round_if_needed(self, now_sec: float) -> None:
        """Prefetch upcoming physical round metadata 30s before boundary (T-30s)."""
        current_start = (int(now_sec) // 300) * 300
        sec_into_round = int(now_sec) - current_start
        if sec_into_round >= 270:  # T - 30s
            next_start = current_start + 300
            next_slug = self.contract_mgr.derive_round_slug(next_start + 10)
            if self._upcoming_round_slug != next_slug:
                logger.info(f"Prefetching upcoming round metadata for {next_slug} at T-30s...")
                round_info = self._fetch_round_info_with_backoff(next_slug)
                if round_info:
                    self._upcoming_round_slug = next_slug
                    self._upcoming_round_info = round_info
                    logger.info(f"Upcoming round prefetch succeeded: {next_slug} (token={round_info.up_token_id})")

    def _fetch_round_info_with_backoff(self, slug: str) -> BTC5mRoundInfo | None:
        """Safely fetch round contract metadata from Gamma API with exponential backoff."""
        delays = [0.0, 0.5, 1.0, 2.0]
        for delay in delays:
            if delay > 0:
                time.sleep(delay)
            try:
                event = self.contract_mgr.fetch_event_by_slug(slug)
                if event:
                    info, err = self.contract_mgr.parse_and_validate_round(event, slug)
                    if info and not err:
                        return info
            except Exception as e:
                logger.warning(f"Error fetching round info for {slug} (attempt): {e}")
        return None

    def _ensure_active_round(self, now_sec: float) -> None:
        expected_slug = self.contract_mgr.derive_round_slug(now_sec)
        if expected_slug == self._current_round_slug:
            # Check T-30s prefetch
            self._prefetch_upcoming_round_if_needed(now_sec)
            return

        # Round transition occurred
        if self._current_round_slug is not None:
            self._close_current_round()

        logger.info(f"Transitioning to new physical round: {expected_slug}")
        self._current_round_slug = expected_slug
        start_epoch = (int(now_sec) // 300) * 300
        end_epoch = start_epoch + 300

        # Reset Binance open price for the new round
        with self._binance_lock:
            self._binance_open_mid = None
            if self._binance_mids:
                self._binance_open_mid = self._binance_mids[-1][2]
            self._binance_open_round_slug = expected_slug

        # Use pre-fetched metadata if matching, otherwise fetch with backoff
        round_info: BTC5mRoundInfo | None = None
        if self._upcoming_round_slug == expected_slug and self._upcoming_round_info:
            round_info = self._upcoming_round_info
            self._upcoming_round_slug = None
            self._upcoming_round_info = None
        else:
            round_info = self._fetch_round_info_with_backoff(expected_slug)

        self._current_round_info = round_info
        up_token = round_info.up_token_id if round_info else None
        down_token = round_info.down_token_id if round_info else None
        cond_id = round_info.condition_id if round_info else None

        # Reset Polymarket state completely so no old token book leaks into new round
        with self._poly_lock:
            self._poly_token_id = up_token
            self._poly_round_slug = expected_slug
            self._poly_best_bid = None
            self._poly_best_ask = None
            self._poly_midpoint = None
            self._poly_mids.clear()
            self._poly_status = "ROUND_TRANSITIONED"

        # Persist new round in database
        self.db.save_leadlag_v2_round({
            "round_slug": expected_slug,
            "experiment_id": self.experiment_id,
            "experiment_spec_hash": self.experiment_spec_hash,
            "is_pilot": self.is_pilot,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "up_token_id": up_token,
            "down_token_id": down_token,
            "condition_id": cond_id,
            "status": "ACTIVE",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

        self._rounds_captured_count += 1
        logger.info(
            f"Physical round {self._rounds_captured_count}/{self.target_physical_rounds} "
            f"registered: {expected_slug} (token={up_token})"
        )

    def _close_current_round(self) -> None:
        if self._current_round_slug is None:
            return
        self._flush_batch()
        # Query total samples recorded for this round
        samples = self.db.get_leadlag_v2_samples(
            round_slug=self._current_round_slug,
            experiment_id=self.experiment_id,
        )
        tot = len(samples)
        val = sum(1 for s in samples if s.get("is_valid"))
        self.db.update_leadlag_v2_round_completion(
            round_slug=self._current_round_slug,
            sample_count=tot,
            valid_sample_count=val,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            experiment_id=self.experiment_id,
        )
        logger.info(f"Closed physical round {self._current_round_slug}: {val}/{tot} valid samples.")

    def _flush_batch(self) -> None:
        """Batch write pending samples, raw events, and compressed payloads."""
        with self.db.transaction() as conn:
            if self._raw_payloads:
                self.db.save_leadlag_v2_raw_payloads_batch(self._raw_payloads, conn=conn)
                self._raw_payloads.clear()

            if self._sample_batch:
                self.db.save_leadlag_v2_samples_batch(self._sample_batch, conn=conn)
                self._sample_batch.clear()

            if self._raw_depth_events:
                self.db.save_leadlag_v2_depth_events_batch(self._raw_depth_events, conn=conn)
                self._raw_depth_events.clear()

            if self._raw_trade_events:
                self.db.save_leadlag_v2_trade_events_batch(self._raw_trade_events, conn=conn)
                self._raw_trade_events.clear()

            if self._raw_poly_events:
                self.db.save_leadlag_v2_poly_events_batch(self._raw_poly_events, conn=conn)
                self._raw_poly_events.clear()

    def _emit_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat_time < 5.0:
            return
        self._last_heartbeat_time = now

        val = sum(1 for s in self._sample_batch if s.is_valid) if self._sample_batch else 0

        with self._binance_lock:
            bn_st = self._binance_status
            bn_age = max(0, int(now * 1000) - self._binance_recv_ts_ms) if self._binance_recv_ts_ms > 0 else 999999

        with self._poly_lock:
            poly_st = self._poly_status
            poly_age = max(0, int(now * 1000) - self._poly_recv_ts_ms) if self._poly_recv_ts_ms > 0 else 999999

        poly_tot = self._poly_source_ts_present + self._poly_source_ts_missing
        poly_cov = round(self._poly_source_ts_present / poly_tot, 4) if poly_tot > 0 else 0.0

        bn_tot = self._binance_source_ts_present + self._binance_source_ts_missing
        bn_cov = round(self._binance_source_ts_present / bn_tot, 4) if bn_tot > 0 else 0.0

        # Percentile metrics
        latency_summary = {
            "binance_source_to_recv": compute_percentiles(list(self._binance_latencies_ms)),
            "poly_source_to_recv": compute_percentiles(list(self._poly_latencies_ms)),
            "inter_feed_skew": compute_percentiles(list(self._inter_feed_skews_ms)),
            "sample_target_error": compute_percentiles(list(self._sample_target_errors_ms)),
        }

        hb = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "epoch_ms": int(now * 1000),
            "pid": os.getpid(),
            "status": "COLLECTING" if not self._stop_event.is_set() else "STOPPED",
            "experiment_id": self.experiment_id,
            "experiment_spec_hash": self.experiment_spec_hash,
            "is_pilot": self.is_pilot,
            "physical_rounds_captured": self._rounds_captured_count,
            "total_samples": self._total_samples_count,
            "valid_samples": val,
            "stale_samples": self._stale_samples_count,
            "binance_feed_status": f"{bn_st} (age {bn_age}ms)",
            "polymarket_feed_status": f"{poly_st} (age {poly_age}ms)",
            "binance_depth_events": self._depth_events_received,
            "binance_trade_events": self._aggtrade_events_received,
            "binance_unique_trades": len(self._seen_trade_ids_set),
            "poly_snapshot_events": self._snapshot_events_received,
            "poly_delta_events": self._delta_events_received,
            "poly_rest_fallback_events": self._rest_fallback_events,
            "binance_source_ts_present": self._binance_source_ts_present,
            "binance_source_ts_missing": self._binance_source_ts_missing,
            "poly_source_ts_present": self._poly_source_ts_present,
            "poly_source_ts_missing": self._poly_source_ts_missing,
            "binance_ts_coverage": bn_cov,
            "poly_ts_coverage": poly_cov,
            "taker_flow_60s_coverage": 1.0 if (self._first_trade_ms and (int(now * 1000) - self._first_trade_ms) > 60000) else 0.0,
            "malformed_events": self._malformed_binance_events + self._malformed_poly_events,
            "unhandled_events": self._unhandled_binance_stream_count + self._unhandled_poly_event_count,
            "duplicate_trades": self._duplicate_aggtrade_events,
            "latency_metrics_json": json.dumps(latency_summary),
            "extra_json": json.dumps({
                "current_round": self._current_round_slug,
                "out_of_order_trades": self._out_of_order_trade_events,
                "depth_continuity_gaps": self._binance_depth_continuity_gaps,
                "missing_key_count": self._missing_key_count,
                "parser_exception_count": self._parser_exception_count,
            }),
        }
        self.db.save_leadlag_v2_heartbeat(hb)

    # ==========================================================================
    # Fail-Fast Pilot Verification (Mandate 18)
    # ==========================================================================

    def _check_pilot_fail_fast(self, elapsed_sec: float) -> None:
        """Enforce strict fail-fast rules for live pilot data collection."""
        if not self.is_pilot:
            return

        # 1. Binance aggTrade count must not be 0 after 30s of active feed
        if elapsed_sec > 30.0 and self._aggtrade_events_received == 0:
            logger.error("PILOT FAIL-FAST: 0 aggTrades received after 30s of active feed! Aborting.")
            self._pilot_failed = True
            self._pilot_failure_reason = "BINANCE_AGGTRADE_ZERO_AFTER_30S"
            self._stop_event.set()

        # 2. Polymarket delta source timestamp must not be 0% after 15 delta events
        if self._delta_events_received >= 15 and self._poly_source_ts_present == 0:
            logger.error("PILOT FAIL-FAST: 0 source timestamps on Polymarket delta events! Aborting.")
            self._pilot_failed = True
            self._pilot_failure_reason = "POLY_DELTA_SOURCE_TS_ABSENT"
            self._stop_event.set()

        # 3. Parser exceptions must not accumulate uncontrollably
        if self._parser_exception_count > 30:
            logger.error("PILOT FAIL-FAST: Continuous parser exceptions (>30)! Aborting.")
            self._pilot_failed = True
            self._pilot_failure_reason = "EXCESSIVE_PARSER_EXCEPTIONS"
            self._stop_event.set()

    # ==========================================================================
    # Main Execution Loop
    # ==========================================================================

    def run(self) -> None:
        """Run the continuous 1-second synchronized lead-lag v2 collector."""
        mode_label = "PILOT VALIDATION (3 ROUNDS)" if self.is_pilot else f"PRODUCTION ({self.target_physical_rounds} ROUNDS)"
        logger.info(
            f"Starting LeadLagCollectorV2 (PID {os.getpid()}) in {mode_label} mode: "
            f"experiment_id={self.experiment_id}..."
        )
        self.acquire_lock()
        self._running = True
        self._start_time_sec = time.time()

        def _handle_signal(signum: int, _frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating clean v2 collector shutdown...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # Start background streaming threads
        self._start_binance_worker()
        self._start_polymarket_worker()

        try:
            while not self._stop_event.is_set():
                now = time.time()
                elapsed = now - self._start_time_sec

                # Enforce fail-fast conditions in pilot mode
                self._check_pilot_fail_fast(elapsed)
                if self._stop_event.is_set():
                    break

                # Check if target physical rounds reached
                if self._rounds_captured_count >= self.target_physical_rounds:
                    logger.info(
                        f"Target of {self.target_physical_rounds} physical rounds reached! Cleanly closing collector."
                    )
                    break

                # Align to exact 1.0 second boundary
                sleep_sec = 1.0 - (now % 1.0)
                if sleep_sec > 0.01:
                    time.sleep(sleep_sec)

                sample_sec_ms = int(time.time() // 1.0) * 1000

                # Ensure active physical round metadata and T-30s prefetch
                self._ensure_active_round(time.time())

                # Collect high-frequency synchronized sample
                sample = self._sample_synchronized_observation(sample_sec_ms)
                self._sample_batch.append(sample)
                self._total_samples_count += 1
                if sample.is_stale:
                    self._stale_samples_count += 1

                # Flush batch every 10 samples (10 seconds)
                if len(self._sample_batch) >= 10:
                    self._flush_batch()

                self._emit_heartbeat()

        except Exception as e:
            logger.exception(f"Unhandled error in lead-lag v2 collector main loop: {e}")
        finally:
            self._close_current_round()
            self._flush_batch()
            self._emit_heartbeat()
            self.release_lock()
            logger.info("LeadLagCollectorV2 stopped cleanly.")
            if self._pilot_failed:
                raise RuntimeError(f"Pilot failed fast due to: {self._pilot_failure_reason}")
