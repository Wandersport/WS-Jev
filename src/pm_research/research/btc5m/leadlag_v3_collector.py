"""High-frequency 1-second synchronized lead-lag replication collector (v3).

Remediated measurement architecture for prospective replication:
- In-memory tick capture with zero SQLite blocking on sampling thread
- Asynchronous persistence worker thread with bounded queue
- Fail-closed queue monitoring (aborts if persistence backlog exceeds safety threshold)
- In-memory O(1) heartbeat telemetry (zero full-table scans)
- Isolated replication database: data/pm_research_v3_replication.db
- Strict 1.0-second cadence preservation

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
import queue
import shutil
import signal
import threading
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from pm_research.research.btc5m.contract import BTC5mContractManager, BTC5mRoundInfo
from pm_research.research.btc5m.leadlag_v2_collector import compute_percentiles
from pm_research.research.btc5m.leadlag_v2_experiment import MAX_STALE_AGE_MS
from pm_research.research.btc5m.leadlag_v3_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_PILOT_ID,
    EXPERIMENT_SPEC_HASH,
    PILOT_PHYSICAL_ROUNDS,
    REPLICATION_DB_PATH,
    TARGET_PHYSICAL_ROUNDS,
    LeadLagSampleV3,
)
from pm_research.research.btc5m.poly_book import PolymarketBookCollector
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)

BINANCE_WS_URL: str = "wss://fstream.binance.com/stream?streams=btcusdt@depth20@100ms/btcusdt@trade"
POLY_WS_URL: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

DEFAULT_V3_LOCK_FILE: str = "data/btc5m_leadlag_v3_replication.lock"
DEFAULT_V3_PILOT_LOCK_FILE: str = "data/btc5m_leadlag_v3_pilot.lock"

DEFAULT_MIN_DISK_FREE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2.0 GiB minimum safe floor
DEFAULT_MAX_WRITER_QUEUE_SIZE: int = 2000  # Fail-closed threshold for persistence backlog


class LeadLagCollectorV3:
    """Cadence-fixed 1-second lead-lag replication collector (v3 architecture)."""

    def __init__(
        self,
        db: Database | None = None,
        target_physical_rounds: int | None = None,
        is_pilot: bool = False,
        wait_for_boundary: bool = False,
        lock_file_path: str | None = None,
        min_disk_free_bytes: int = DEFAULT_MIN_DISK_FREE_BYTES,
        max_queue_depth: int = DEFAULT_MAX_WRITER_QUEUE_SIZE,
    ) -> None:
        self.db = db or Database(REPLICATION_DB_PATH)
        self.is_pilot = is_pilot
        if target_physical_rounds is not None:
            self.target_physical_rounds = target_physical_rounds
        else:
            self.target_physical_rounds = PILOT_PHYSICAL_ROUNDS if is_pilot else TARGET_PHYSICAL_ROUNDS
        self.wait_for_boundary = wait_for_boundary
        self.experiment_id = EXPERIMENT_PILOT_ID if is_pilot else EXPERIMENT_ID
        self.experiment_spec_hash = EXPERIMENT_SPEC_HASH
        self.max_queue_depth = max_queue_depth

        default_lock = DEFAULT_V3_PILOT_LOCK_FILE if is_pilot else DEFAULT_V3_LOCK_FILE
        self.lock_file_path = Path(lock_file_path or default_lock)
        self.contract_mgr = BTC5mContractManager()
        self.poly_rest_collector = PolymarketBookCollector()

        self._lock_file_fd: int | None = None
        self._stop_event = threading.Event()
        self._shutdown_initiated = False

        # In-memory Async Persistence Queue
        self._write_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=10000)
        self._writer_thread: threading.Thread | None = None
        self._writer_queue_high_water: int = 0
        self._writer_commit_latencies_ms: collections.deque[float] = collections.deque(maxlen=100)
        self._writer_errors_count: int = 0

        # Physical round & scheduling state
        self._current_round_slug: str | None = None
        self._current_round_info: BTC5mRoundInfo | None = None
        self._upcoming_round_slug: str | None = None
        self._upcoming_round_info: BTC5mRoundInfo | None = None
        self._rounds_captured_count: int = 0
        self._rounds_completed_count: int = self._count_completed_full_rounds_in_db()
        self._first_round_of_run: bool = True
        self._is_current_round_warmup_partial: bool = False
        self._current_round_sample_count: int = 0
        self._current_round_valid_count: int = 0

        # In-memory tick cadence telemetry
        self._intended_ticks_count: int = 0
        self._captured_ticks_count: int = 0
        self._missed_ticks_count: int = 0
        self._interarrival_times_ms: collections.deque[float] = collections.deque(maxlen=500)
        self._last_sample_actual_ms: int | None = None
        self._sample_target_errors_ms: collections.deque[int] = collections.deque(maxlen=500)

        # In-memory Taker-flow SLA accounting (O(1) tracking, zero SQLite queries)
        self._first_trade_ms: int | None = None
        self._valid_post_warmup_samples_count: int = 0
        self._non_null_tf_60s_count: int = 0

        # Memory buffers for Binance feeds
        self._bn_lock = threading.Lock()
        self._bn_bids: dict[float, float] = {}
        self._bn_asks: dict[float, float] = {}
        self._bn_source_ts_ms: int | None = None
        self._bn_recv_ts_ms: int = 0
        self._bn_last_u: int | None = None
        self._bn_status: str = "DISCONNECTED"
        self._binance_trades: collections.deque[tuple[int, float, float, bool, int]] = collections.deque(maxlen=50000)
        self._seen_trade_ids_set: set[int] = set()
        self._bn_mid_history: collections.deque[tuple[int | None, int, float]] = collections.deque(maxlen=1000)
        self._bn_open_mid: float | None = None

        # Memory buffers for Polymarket feeds
        self._poly_lock = threading.Lock()
        self._poly_token_id: str | None = None
        self._poly_round_slug: str | None = None
        self._poly_best_bid: float | None = None
        self._poly_best_ask: float | None = None
        self._poly_midpoint: float | None = None
        self._poly_source_ts_ms: int | None = None
        self._poly_recv_ts_ms: int = 0
        self._poly_status: str = "DISCONNECTED"
        self._poly_mids: collections.deque[tuple[int | None, int, float]] = collections.deque(maxlen=1000)

        # Structured diagnostic counters
        self._depth_events_received: int = 0
        self._aggtrade_events_received: int = 0
        self._duplicate_aggtrade_events: int = 0
        self._out_of_order_trade_events: int = 0
        self._binance_depth_continuity_gaps: int = 0
        self._snapshot_events_received: int = 0
        self._delta_events_received: int = 0
        self._rest_fallback_events: int = 0
        self._unhandled_binance_stream_count: int = 0
        self._unhandled_poly_event_count: int = 0
        self._malformed_binance_events: int = 0
        self._malformed_poly_events: int = 0
        self._missing_key_count: int = 0
        self._parser_exception_count: int = 0

        # Feed timestamp SLA tracking
        self._binance_source_ts_present: int = 0
        self._binance_source_ts_missing: int = 0
        self._poly_source_ts_present: int = 0
        self._poly_source_ts_missing: int = 0

        # Latencies & Skew
        self._binance_latencies_ms: collections.deque[int] = collections.deque(maxlen=500)
        self._poly_latencies_ms: collections.deque[int] = collections.deque(maxlen=500)
        self._inter_feed_skews_ms: collections.deque[int] = collections.deque(maxlen=500)

        # Storage guard
        self._min_disk_free_bytes = min_disk_free_bytes
        self._disk_free_bytes: int | None = None

    def _count_completed_full_rounds_in_db(self) -> int:
        """Count already-completed full rounds matching qualification criteria in DB."""
        try:
            with self.db._get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) FROM leadlag_v2_rounds
                    WHERE experiment_id = ?
                      AND status = 'COMPLETED'
                      AND sample_count >= 285
                      AND (end_epoch - start_epoch) >= 300
                    """,
                    (self.experiment_id,),
                ).fetchone()
            count = int(row[0]) if row and row[0] is not None else 0
            logger.info(
                f"DB resume check: {count} completed full rounds (status=COMPLETED, samples>=285) "
                f"found for {self.experiment_id}."
            )
            return count
        except Exception as e:
            logger.warning(f"Could not query completed rounds from DB: {e}")
            return 0

    # ==========================================================================
    # Asynchronous Persistence Worker
    # ==========================================================================

    def _start_persistence_worker(self) -> None:
        """Start the background persistence worker thread."""
        self._writer_thread = threading.Thread(
            target=self._persistence_worker_loop,
            name="v3_persistence_worker",
            daemon=True,
        )
        self._writer_thread.start()
        logger.info("Asynchronous persistence worker started.")

    def _persistence_worker_loop(self) -> None:
        """Background thread consuming write batches and executing transactions."""
        while not self._stop_event.is_set() or not self._write_queue.empty():
            batch: list[tuple[str, Any]] = []
            try:
                # Grab first item blocking up to 250ms
                item = self._write_queue.get(timeout=0.25)
                batch.append(item)
                # Drain remaining available items in queue up to batch size 500
                while len(batch) < 500 and not self._write_queue.empty():
                    batch.append(self._write_queue.get_nowait())
            except queue.Empty:
                continue

            if not batch:
                continue

            # Update high water mark
            q_depth = self._write_queue.qsize()
            if q_depth > self._writer_queue_high_water:
                self._writer_queue_high_water = q_depth

            t0 = time.time()
            try:
                samples_batch: list[Any] = []
                raw_payloads_batch: list[dict[str, Any]] = []
                trades_batch: list[dict[str, Any]] = []

                with self.db.transaction() as writer_conn:
                    for op_type, data in batch:
                        if op_type == "sample":
                            samples_batch.append(data)
                        elif op_type == "raw_payload":
                            raw_payloads_batch.append(data)
                        elif op_type == "trade":
                            trades_batch.append(data)
                        elif op_type == "round_save":
                            self.db.save_leadlag_v2_round(data, conn=writer_conn)
                        elif op_type == "round_complete":
                            self.db.update_leadlag_v2_round_completion(
                                round_slug=data["round_slug"],
                                sample_count=data["total_samples"],
                                valid_sample_count=data["valid_samples"],
                                completed_at_utc=data.get("completed_at_utc", datetime.now(timezone.utc).isoformat()),
                                status=data.get("status", "COMPLETED"),
                                experiment_id=data["experiment_id"],
                                conn=writer_conn,
                            )
                        elif op_type == "heartbeat":
                            self.db.save_leadlag_v2_heartbeat(data, conn=writer_conn)

                    if samples_batch:
                        self.db.save_leadlag_v2_samples_batch(samples_batch, conn=writer_conn)
                    if raw_payloads_batch:
                        self.db.save_leadlag_v2_raw_payloads_batch(raw_payloads_batch, conn=writer_conn)
                    if trades_batch:
                        self.db.save_leadlag_v2_trade_events_batch(trades_batch, conn=writer_conn)

                commit_ms = (time.time() - t0) * 1000.0
                self._writer_commit_latencies_ms.append(commit_ms)
                for _ in batch:
                    self._write_queue.task_done()
            except Exception as e:
                self._writer_errors_count += 1
                logger.error(f"Persistence worker error committing batch of {len(batch)}: {e}")
                for _ in batch:
                    self._write_queue.task_done()

        logger.info("Persistence worker exited cleanly.")

    def _enqueue_write(self, op_type: str, data: Any) -> None:
        """Enqueue an item for asynchronous persistence with fail-closed protection."""
        q_depth = self._write_queue.qsize()
        if q_depth >= self.max_queue_depth:
            logger.critical(
                f"Persistence queue backlog ({q_depth}) exceeded safety limit ({self.max_queue_depth})! "
                "Failing closed to prevent sampling cadence degradation."
            )
            self._shutdown_initiated = True
            self._stop_event.set()
            raise RuntimeError(f"Persistence queue exceeded safety limit ({q_depth})")

        try:
            self._write_queue.put_nowait((op_type, data))
        except queue.Full:
            self._shutdown_initiated = True
            self._stop_event.set()
            raise RuntimeError("Persistence queue full")

    # ==========================================================================
    # File Locking & Safety Guard
    # ==========================================================================

    def _acquire_lock(self) -> None:
        """Acquire exclusive advisory file lock."""
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file_fd = os.open(str(self.lock_file_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(self._lock_file_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(self._lock_file_fd, f"{os.getpid()}\n".encode("utf-8"))
        except (BlockingIOError, OSError) as e:
            raise RuntimeError(f"Could not acquire lock on {self.lock_file_path}: another collector running?") from e

    def _release_lock(self) -> None:
        """Release advisory file lock."""
        if self._lock_file_fd is not None:
            try:
                fcntl.flock(self._lock_file_fd, fcntl.LOCK_UN)
                os.close(self._lock_file_fd)
            except Exception:
                pass
            self._lock_file_fd = None
        if self.lock_file_path.exists():
            try:
                self.lock_file_path.unlink()
            except Exception:
                pass

    def _check_disk_space(self) -> None:
        """Evaluate free disk storage against minimum operating floor."""
        try:
            total, used, free = shutil.disk_usage(self.db.db_path.parent)
            self._disk_free_bytes = free
            if free < self._min_disk_free_bytes:
                free_gb = round(free / (1024 ** 3), 2)
                floor_gb = round(self._min_disk_free_bytes / (1024 ** 3), 2)
                raise RuntimeError(f"Disk free space ({free_gb} GiB) fell below minimum safe floor ({floor_gb} GiB)")
        except Exception as e:
            if "below minimum" in str(e):
                raise
            logger.warning(f"Could not check disk usage: {e}")

    # ==========================================================================
    # Feed Parsing (Identical to v2 for exact scientific equivalence)
    # ==========================================================================

    def _parse_ts(self, raw_val: Any) -> int | None:
        if raw_val is None:
            return None
        try:
            val = float(raw_val)
            if val <= 0:
                return None
            if val < 1e11:  # seconds
                return int(val * 1000)
            if val < 1e14:  # milliseconds
                return int(val)
            if val < 1e17:  # microseconds
                return int(val / 1000)
            return int(val / 1_000_000)  # nanoseconds
        except (ValueError, TypeError):
            return None

    def _persist_lossless_raw_payload(self, feed_name: str, payload_bytes: bytes, recv_ms: int) -> str:
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
        compressed = zlib.compress(payload_bytes, level=6)
        payload_id = f"{feed_name}_{recv_ms}_{payload_hash[:12]}"
        record = {
            "payload_id": payload_id,
            "experiment_id": self.experiment_id,
            "experiment_spec_hash": self.experiment_spec_hash,
            "feed": feed_name,
            "round_slug": self._current_round_slug or "pending",
            "recv_ts_ms": recv_ms,
            "uncompressed_len": len(payload_bytes),
            "compressed_len": len(compressed),
            "sha256_hash": payload_hash,
            "compressed_payload": compressed,
        }
        self._enqueue_write("raw_payload", record)
        return payload_id

    def _handle_binance_message(self, data: str, recv_ms: int, mono_ns: int) -> None:
        """Parse Binance WebSocket stream frames."""
        try:
            msg = json.loads(data)
        except Exception:
            self._malformed_binance_events += 1
            self._parser_exception_count += 1
            return

        payload_bytes = data.encode("utf-8")
        self._persist_lossless_raw_payload("binance", payload_bytes, recv_ms)

        stream = str(msg.get("stream", "")).lower()
        payload = msg.get("data", msg)
        event_type = payload.get("e")

        if "depth20" in stream or event_type == "depthUpdate":
            self._depth_events_received += 1
            s_ts = self._parse_ts(payload.get("E") or payload.get("T"))
            if s_ts is not None:
                self._binance_source_ts_present += 1
                lat = max(0, recv_ms - s_ts)
                self._binance_latencies_ms.append(lat)
            else:
                self._binance_source_ts_missing += 1

            u = payload.get("u")
            if self._bn_last_u is not None and u is not None:
                if u > self._bn_last_u + 1:
                    self._binance_depth_continuity_gaps += 1
            self._bn_last_u = u

            bids_raw = payload.get("b", [])
            asks_raw = payload.get("a", [])
            with self._bn_lock:
                self._bn_bids = {float(p): float(q) for p, q in bids_raw}
                self._bn_asks = {float(p): float(q) for p, q in asks_raw}
                self._bn_source_ts_ms = s_ts
                self._bn_recv_ts_ms = recv_ms
                self._bn_status = "CONNECTED"

                if self._bn_bids and self._bn_asks:
                    best_b = max(self._bn_bids.keys())
                    best_a = min(self._bn_asks.keys())
                    mid = (best_b + best_a) / 2.0
                    self._bn_mid_history.append((s_ts, recv_ms, mid))
                    if self._bn_open_mid is None:
                        self._bn_open_mid = mid

        elif "aggtrade" in stream or "trade" in stream or event_type in ("aggTrade", "trade"):
            self._aggtrade_events_received += 1
            trade_id = payload.get("a") or payload.get("t")
            if trade_id is not None:
                if trade_id in self._seen_trade_ids_set:
                    self._duplicate_aggtrade_events += 1
                    return
                self._seen_trade_ids_set.add(trade_id)
                if len(self._seen_trade_ids_set) > 100000:
                    self._seen_trade_ids_set.clear()

            s_ts = self._parse_ts(payload.get("T") or payload.get("E"))
            price = float(payload["p"])
            qty = float(payload["q"])
            is_buyer_maker = bool(payload.get("m", False))

            with self._bn_lock:
                if self._binance_trades and s_ts is not None:
                    last_t = self._binance_trades[-1][0]
                    if s_ts < last_t:
                        self._out_of_order_trade_events += 1

                self._binance_trades.append((s_ts or recv_ms, price, qty, is_buyer_maker, int(trade_id or 0)))
                if self._first_trade_ms is None:
                    self._first_trade_ms = recv_ms

    def _handle_poly_message(self, data: str, recv_ms: int, mono_ns: int) -> None:
        """Parse Polymarket WebSocket stream frames."""
        try:
            msg = json.loads(data)
        except Exception:
            self._malformed_poly_events += 1
            self._parser_exception_count += 1
            return

        payload_bytes = data.encode("utf-8")
        self._persist_lossless_raw_payload("polymarket", payload_bytes, recv_ms)

        items = msg if isinstance(msg, list) else [msg]
        for item in items:
            if not isinstance(item, dict):
                continue
            event_type = item.get("event_type")
            if "bids" in item or "asks" in item or event_type == "book":
                self._snapshot_events_received += 1
                s_ts = self._parse_ts(item.get("timestamp") or item.get("source_ts"))
                if s_ts is not None:
                    self._poly_source_ts_present += 1
                    lat = max(0, recv_ms - s_ts)
                    self._poly_latencies_ms.append(lat)
                else:
                    self._poly_source_ts_missing += 1

                bids_raw = item.get("bids", [])
                asks_raw = item.get("asks", [])
                b_prices = [float(b["price"]) for b in bids_raw if isinstance(b, dict) and "price" in b]
                a_prices = [float(a["price"]) for a in asks_raw if isinstance(a, dict) and "price" in a]
                best_b = max(b_prices) if b_prices else None
                best_a = min(a_prices) if a_prices else None
                self._update_poly_state(best_b, best_a, s_ts, recv_ms)

            elif "price_changes" in item or event_type == "price_change":
                self._delta_events_received += 1
                outer_ts = self._parse_ts(item.get("timestamp"))
                pcs = item.get("price_changes", [])
                for pc in pcs:
                    if isinstance(pc, dict) and str(pc.get("asset_id")) == self._poly_token_id:
                        s_ts = outer_ts or self._parse_ts(pc.get("timestamp"))
                        if s_ts is not None:
                            self._poly_source_ts_present += 1
                            lat = max(0, recv_ms - s_ts)
                            self._poly_latencies_ms.append(lat)
                        else:
                            self._poly_source_ts_missing += 1
                        bb = float(pc["best_bid"]) if pc.get("best_bid") is not None else None
                        ba = float(pc["best_ask"]) if pc.get("best_ask") is not None else None
                        self._update_poly_state(bb, ba, s_ts, recv_ms)

    def _update_poly_state(self, best_bid: float | None, best_ask: float | None, source_ts: int | None, recv_ms: int) -> None:
        with self._poly_lock:
            if best_bid is not None:
                self._poly_best_bid = best_bid
            if best_ask is not None:
                self._poly_best_ask = best_ask
            self._poly_source_ts_ms = source_ts
            self._poly_recv_ts_ms = recv_ms
            self._poly_status = "CONNECTED"

            bb = self._poly_best_bid
            ba = self._poly_best_ask
            if bb is not None and ba is not None and bb < ba:
                mid = (bb + ba) / 2.0
                self._poly_midpoint = mid
                self._poly_mids.append((source_ts, recv_ms, mid))

    # ==========================================================================
    # Synchronized 1-Second Sampling Tick
    # ==========================================================================

    def _capture_sample(self, target_sec_ms: int) -> LeadLagSampleV3:
        """Capture prospective 1-second state with ZERO synchronous SQLite blocking."""
        self._intended_ticks_count += 1
        now = time.time()
        actual_ms = int(now * 1000)
        mono_ns = time.monotonic_ns()

        # Interarrival tracking
        if self._last_sample_actual_ms is not None:
            interarrival = actual_ms - self._last_sample_actual_ms
            self._interarrival_times_ms.append(interarrival)
        self._last_sample_actual_ms = actual_ms
        self._captured_ticks_count += 1

        target_err = abs(actual_ms - target_sec_ms)
        self._sample_target_errors_ms.append(target_err)

        round_slug = self._current_round_slug or "unassigned_round"
        start_epoch = int(round_slug.split("-")[-1]) if "-" in round_slug else int(now)
        end_epoch = start_epoch + 300
        seconds_remaining = max(0, end_epoch - int(now))

        # Snapshot Polymarket in-memory state
        with self._poly_lock:
            poly_bid = self._poly_best_bid
            poly_ask = self._poly_best_ask
            poly_mid = self._poly_midpoint
            poly_source_ts = self._poly_source_ts_ms
            poly_recv_ts = self._poly_recv_ts_ms
            poly_mids = list(self._poly_mids)

        poly_receipt_age = max(0, actual_ms - poly_recv_ts) if poly_recv_ts > 0 else 999999
        poly_source_age = max(0, actual_ms - poly_source_ts) if poly_source_ts is not None else None
        poly_spread = round(poly_ask - poly_bid, 4) if poly_bid and poly_ask else None
        poly_crossed = bool(poly_bid and poly_ask and poly_bid >= poly_ask)
        poly_is_valid = bool(poly_bid and poly_ask and not poly_crossed and poly_receipt_age <= MAX_STALE_AGE_MS)

        # Snapshot Binance in-memory state
        with self._bn_lock:
            bn_bids_dict = dict(self._bn_bids)
            bn_asks_dict = dict(self._bn_asks)
            bn_source_ts = self._bn_source_ts_ms
            bn_recv_ts = self._bn_recv_ts_ms
            mids = list(self._bn_mid_history)
            trades = list(self._binance_trades)
            bn_open_mid = self._bn_open_mid

        bn_receipt_age = max(0, actual_ms - bn_recv_ts) if bn_recv_ts > 0 else 999999
        bn_source_age = max(0, actual_ms - bn_source_ts) if bn_source_ts is not None else None

        bids = sorted(bn_bids_dict.items(), key=lambda x: x[0], reverse=True)
        asks = sorted(bn_asks_dict.items(), key=lambda x: x[0])
        bn_bid = bids[0][0] if bids else None
        bn_ask = asks[0][0] if asks else None
        bn_mid = (bn_bid + bn_ask) / 2.0 if bn_bid and bn_ask else None
        bn_is_valid = bool(bn_bid and bn_ask and bn_bid < bn_ask and bn_receipt_age <= MAX_STALE_AGE_MS)

        is_stale = (poly_receipt_age > MAX_STALE_AGE_MS) or (bn_receipt_age > MAX_STALE_AGE_MS)
        overall_valid = poly_is_valid and bn_is_valid and not is_stale

        # Feature calculations
        microprice = None
        microprice_offset_bps = None
        top1_imb = None
        top5_imb = None
        top20_imb = None
        spread_bps = None
        if bn_bid and bn_ask and bn_mid:
            bid_q1 = bids[0][1]
            ask_q1 = asks[0][1]
            tot1 = bid_q1 + ask_q1
            if tot1 > 0:
                microprice = (bn_bid * ask_q1 + bn_ask * bid_q1) / tot1
                microprice_offset_bps = round(((microprice - bn_mid) / bn_mid) * 10000.0, 3)
                top1_imb = round((bid_q1 - ask_q1) / tot1, 4)
            spread_bps = round(((bn_ask - bn_bid) / bn_mid) * 10000.0, 3)
            top5_b = sum(q for _, q in bids[:5])
            top5_a = sum(q for _, q in asks[:5])
            top5_imb = round((top5_b - top5_a) / (top5_b + top5_a), 4) if (top5_b + top5_a) > 0 else 0.0
            top20_b = sum(q for _, q in bids[:20])
            top20_a = sum(q for _, q in asks[:20])
            top20_imb = round((top20_b - top20_a) / (top20_b + top20_a), 4) if (top20_b + top20_a) > 0 else 0.0

        def _poly_ret(sec: int) -> float | None:
            if poly_mid is None:
                return None
            target_t = actual_ms - sec * 1000
            for _s, r_t, past_m in reversed(poly_mids):
                if r_t <= target_t:
                    return round(poly_mid - past_m, 5)
            return None

        def _bn_ret(sec: int) -> float | None:
            if bn_mid is None:
                return None
            target_t = actual_ms - sec * 1000
            for _s, r_t, past_m in reversed(mids):
                if r_t <= target_t:
                    return round(((bn_mid - past_m) / past_m) * 10000.0, 3)
            return None

        def _taker_flow(sec: int) -> float | None:
            if not trades or self._first_trade_ms is None or (actual_ms - self._first_trade_ms) < (sec * 1000):
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

        tf_60s = _taker_flow(60)

        # O(1) in-memory tracker for taker-flow 60s coverage
        if overall_valid and self._first_trade_ms and (actual_ms - self._first_trade_ms) >= 60000:
            self._valid_post_warmup_samples_count += 1
            if tf_60s is not None:
                self._non_null_tf_60s_count += 1

        skew_ms = abs(bn_recv_ts - poly_recv_ts) if bn_recv_ts > 0 and poly_recv_ts > 0 else 0
        self._inter_feed_skews_ms.append(skew_ms)

        sample = LeadLagSampleV3(
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
            poly_provenance_mode="WS_STREAM",
            poly_best_bid=poly_bid,
            poly_best_ask=poly_ask,
            poly_midpoint=poly_mid,
            poly_spread=poly_spread,
            poly_return_1s=_poly_ret(1),
            poly_return_2s=_poly_ret(2),
            poly_return_3s=_poly_ret(3),
            poly_return_5s=_poly_ret(5),
            poly_return_10s=_poly_ret(10),
            poly_return_30s=_poly_ret(30),
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
            binance_return_since_open_bps=round(((bn_mid - bn_open_mid) / bn_open_mid) * 10000.0, 3) if bn_mid and bn_open_mid else None,
            binance_return_1s_bps=_bn_ret(1),
            binance_return_2s_bps=_bn_ret(2),
            binance_return_3s_bps=_bn_ret(3),
            binance_return_5s_bps=_bn_ret(5),
            binance_return_10s_bps=_bn_ret(10),
            binance_return_30s_bps=_bn_ret(30),
            binance_return_60s_bps=_bn_ret(60),
            binance_taker_flow_1s=_taker_flow(1),
            binance_taker_flow_2s=_taker_flow(2),
            binance_taker_flow_3s=_taker_flow(3),
            binance_taker_flow_5s=_taker_flow(5),
            binance_taker_flow_10s=_taker_flow(10),
            binance_taker_flow_30s=_taker_flow(30),
            binance_taker_flow_60s=tf_60s,
            binance_top1_depth_imbalance=top1_imb,
            binance_top5_depth_imbalance=top5_imb,
            binance_top20_depth_imbalance=top20_imb,
            binance_is_valid=bn_is_valid,
            source_to_receive_latency_ms=max(0, actual_ms - bn_source_ts) if bn_source_ts else None,
            inter_feed_receive_skew_ms=skew_ms,
            is_stale=is_stale,
            stale_reason="stale_feed" if is_stale else None,
            is_valid=overall_valid,
            raw_payload_id=None,
            raw_json="",
        )

        self._current_round_sample_count += 1
        if overall_valid:
            self._current_round_valid_count += 1

        # Enqueue write asynchronously (Zero disk blocking on sampling thread!)
        self._enqueue_write("sample", sample.to_dict())
        return sample

    # ==========================================================================
    # In-Memory Heartbeat Telemetry (O(1), Zero Full-Table Scans!)
    # ==========================================================================

    def _emit_heartbeat(self) -> None:
        """Emit telemetry heartbeat using O(1) in-memory counters."""
        now = time.time()
        self._check_disk_space()

        poly_tot = self._poly_source_ts_present + self._poly_source_ts_missing
        poly_cov = round(self._poly_source_ts_present / poly_tot, 4) if poly_tot > 0 else 0.0

        bn_tot = self._binance_source_ts_present + self._binance_source_ts_missing
        bn_cov = round(self._binance_source_ts_present / bn_tot, 4) if bn_tot > 0 else 0.0

        tf_cov = (
            round(self._non_null_tf_60s_count / self._valid_post_warmup_samples_count, 4)
            if self._valid_post_warmup_samples_count > 0
            else 0.0
        )

        q_depth = self._write_queue.qsize()
        commit_lat = compute_percentiles(list(self._writer_commit_latencies_ms))
        inter_stats = compute_percentiles(list(self._interarrival_times_ms))

        telemetry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "epoch_ms": int(now * 1000),
            "pid": os.getpid(),
            "status": "COLLECTING" if not self._stop_event.is_set() else "STOPPED",
            "experiment_id": self.experiment_id,
            "experiment_spec_hash": self.experiment_spec_hash,
            "is_pilot": self.is_pilot,
            "physical_rounds_captured": self._rounds_captured_count,
            "total_samples": self._captured_ticks_count,
            "valid_samples": self._current_round_valid_count,
            "intended_ticks": self._intended_ticks_count,
            "captured_ticks": self._captured_ticks_count,
            "missed_ticks": self._missed_ticks_count,
            "capture_ratio": round(self._captured_ticks_count / self._intended_ticks_count, 4) if self._intended_ticks_count > 0 else 1.0,
            "interarrival_ms_p50": inter_stats["p50"],
            "interarrival_ms_p95": inter_stats["p95"],
            "interarrival_ms_p99": inter_stats["p99"],
            "interarrival_ms_mean": inter_stats["mean"],
            "writer_queue_depth": q_depth,
            "writer_queue_high_water": self._writer_queue_high_water,
            "writer_commit_latency_p50_ms": commit_lat["p50"],
            "writer_commit_latency_p95_ms": commit_lat["p95"],
            "writer_errors_count": self._writer_errors_count,
            "binance_ts_coverage": bn_cov,
            "poly_ts_coverage": poly_cov,
            "taker_flow_60s_coverage": tf_cov,
            "malformed_events": self._malformed_binance_events + self._malformed_poly_events,
            "parser_exceptions": self._parser_exception_count,
            "disk_free_gb": round(self._disk_free_bytes / (1024 ** 3), 2) if self._disk_free_bytes else None,
        }

        # Enqueue heartbeat to persistence worker
        hb_record = {
            "timestamp_utc": telemetry["timestamp_utc"],
            "epoch_ms": telemetry["epoch_ms"],
            "pid": telemetry["pid"],
            "status": telemetry["status"],
            "experiment_id": self.experiment_id,
            "experiment_spec_hash": self.experiment_spec_hash,
            "is_pilot": self.is_pilot,
            "physical_rounds_captured": self._rounds_captured_count,
            "total_samples": self._captured_ticks_count,
            "valid_samples": self._current_round_valid_count,
            "stale_samples": 0,
            "binance_feed_status": self._bn_status,
            "polymarket_feed_status": self._poly_status,
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
            "taker_flow_60s_coverage": tf_cov,
            "malformed_events": telemetry["malformed_events"],
            "unhandled_events": self._unhandled_binance_stream_count + self._unhandled_poly_event_count,
            "duplicate_trades": self._duplicate_aggtrade_events,
            "latency_metrics_json": json.dumps(telemetry),
            "extra_json": json.dumps({"writer_queue_depth": q_depth}),
        }
        self._enqueue_write("heartbeat", hb_record)
        logger.debug(f"Heartbeat emitted: captured={self._captured_ticks_count}, queue={q_depth}, tf_cov={tf_cov}")

    def get_telemetry_snapshot(self) -> dict[str, Any]:
        """Return instantaneous in-memory telemetry snapshot."""
        inter_stats = compute_percentiles(list(self._interarrival_times_ms))
        poly_tot = self._poly_source_ts_present + self._poly_source_ts_missing
        poly_cov = round(self._poly_source_ts_present / poly_tot, 4) if poly_tot > 0 else 0.0
        bn_tot = self._binance_source_ts_present + self._binance_source_ts_missing
        bn_cov = round(self._binance_source_ts_present / bn_tot, 4) if bn_tot > 0 else 0.0
        tf_cov = (
            round(self._non_null_tf_60s_count / self._valid_post_warmup_samples_count, 4)
            if self._valid_post_warmup_samples_count > 0
            else 0.0
        )
        return {
            "intended_ticks": self._intended_ticks_count,
            "captured_ticks": self._captured_ticks_count,
            "missed_ticks": self._missed_ticks_count,
            "capture_ratio": round(self._captured_ticks_count / self._intended_ticks_count, 4) if self._intended_ticks_count > 0 else 1.0,
            "interarrival_ms_p50": inter_stats["p50"],
            "interarrival_ms_p95": inter_stats["p95"],
            "interarrival_ms_p99": inter_stats["p99"],
            "writer_queue_depth": self._write_queue.qsize(),
            "writer_queue_high_water": self._writer_queue_high_water,
            "binance_ts_coverage": bn_cov,
            "poly_ts_coverage": poly_cov,
            "taker_flow_60s_coverage": tf_cov,
            "malformed_events": self._malformed_binance_events + self._malformed_poly_events,
            "parser_exceptions": self._parser_exception_count,
        }

    # ==========================================================================
    # Background Feed Workers
    # ==========================================================================

    def _start_binance_worker(self) -> None:
        def _worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def _run() -> None:
                while not self._stop_event.is_set():
                    try:
                        with self._bn_lock:
                            self._bn_status = "CONNECTING"
                        async with websockets.connect(
                            BINANCE_WS_URL,
                            ping_interval=20,
                            ping_timeout=10,
                            close_timeout=5,
                        ) as ws:
                            with self._bn_lock:
                                self._bn_status = "CONNECTED"
                            while not self._stop_event.is_set():
                                msg = await ws.recv()
                                now_ms = int(time.time() * 1000)
                                mono_ns = time.monotonic_ns()
                                self._handle_binance_message(msg, now_ms, mono_ns)
                    except Exception as e:
                        with self._bn_lock:
                            self._bn_status = f"DISCONNECTED ({type(e).__name__})"
                        if not self._stop_event.is_set():
                            await asyncio.sleep(2.0)

            try:
                loop.run_until_complete(_run())
            finally:
                loop.close()

        t = threading.Thread(target=_worker, daemon=True, name="v3_binance_stream")
        t.start()

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

        t = threading.Thread(target=_worker, daemon=True, name="v3_polymarket_stream")
        t.start()

    def _poll_poly_rest_fallback(self) -> None:
        """Fallback unauthenticated public REST poll when WebSocket is quiet."""
        token_id = self._poly_token_id
        if not token_id:
            return
        recv_ms = int(time.time() * 1000)
        try:
            book = self.poly_rest_collector.fetch_book(token_id)
            if book.is_valid:
                self._rest_fallback_events += 1
                s_ts = book.source_event_timestamp_ms
                if s_ts is not None:
                    self._poly_source_ts_present += 1
                else:
                    self._poly_source_ts_missing += 1

                raw_bytes = json.dumps({
                    "token_id": token_id,
                    "bids": [{"price": b.price, "size": b.size} for b in book.bids],
                    "asks": [{"price": a.price, "size": a.size} for a in book.asks],
                    "source_ts": s_ts,
                    "recv_ms": recv_ms,
                }).encode("utf-8")
                self._persist_lossless_raw_payload("polymarket", raw_bytes, recv_ms)
                self._update_poly_state(book.best_bid, book.best_ask, s_ts, recv_ms)
        except Exception as e:
            logger.warning(f"Error polling Polymarket REST fallback: {e}")

    # ==========================================================================
    # Round Lifecycle & Scheduling
    # ==========================================================================

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
                logger.warning(f"Error fetching round info for {slug}: {e}")
        return None

    def _prefetch_upcoming_round_if_needed(self, now_sec: float) -> None:
        """Prefetch upcoming physical round metadata 30s before boundary (T-30s)."""
        current_start = (int(now_sec) // 300) * 300
        sec_into_round = int(now_sec) - current_start
        if sec_into_round >= 270:
            next_start = current_start + 300
            next_slug = self.contract_mgr.derive_round_slug(next_start + 10)
            if self._upcoming_round_slug != next_slug:
                round_info = self._fetch_round_info_with_backoff(next_slug)
                if round_info:
                    self._upcoming_round_slug = next_slug
                    self._upcoming_round_info = round_info
                    logger.info(f"Upcoming round prefetch succeeded: {next_slug}")

    def _ensure_active_round(self, now_sec: float) -> bool:
        """Ensure current active physical round is registered without DB blocking.

        Returns True if a round is actively registered, False if target full rounds reached.
        """
        expected_slug = self.contract_mgr.derive_round_slug(now_sec)
        if expected_slug == self._current_round_slug:
            self._prefetch_upcoming_round_if_needed(now_sec)
            return True

        if self._current_round_slug is not None:
            self._close_current_round()

        if self._rounds_completed_count >= self.target_physical_rounds:
            return False

        self._current_round_slug = expected_slug
        start_epoch = (int(now_sec) // 300) * 300
        end_epoch = start_epoch + 300

        # Check if startup is mid-round:
        # If started >2.0s into the 5-minute round and this is the first round of the run, mark WARMUP_PARTIAL
        sec_into_round = now_sec - start_epoch
        if self._first_round_of_run and sec_into_round > 2.0:
            self._is_current_round_warmup_partial = True
            round_status = "WARMUP_PARTIAL"
            logger.warning(
                f"Collector started mid-round ({sec_into_round:.1f}s after boundary). "
                f"Marking round {expected_slug} as WARMUP_PARTIAL (will NOT count toward target)."
            )
        else:
            self._is_current_round_warmup_partial = False
            round_status = "ACTIVE"

        self._first_round_of_run = False

        with self._bn_lock:
            self._bn_open_mid = None
            if self._bn_mid_history:
                self._bn_open_mid = self._bn_mid_history[-1][2]

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

        with self._poly_lock:
            self._poly_token_id = up_token
            self._poly_round_slug = expected_slug
            self._poly_best_bid = None
            self._poly_best_ask = None
            self._poly_midpoint = None
            self._poly_mids.clear()
            self._poly_status = "ROUND_TRANSITIONED"

        self._enqueue_write("round_save", {
            "round_slug": expected_slug,
            "experiment_id": self.experiment_id,
            "experiment_spec_hash": self.experiment_spec_hash,
            "is_pilot": self.is_pilot,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "up_token_id": up_token,
            "down_token_id": down_token,
            "condition_id": cond_id,
            "status": round_status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })

        self._rounds_captured_count += 1
        logger.info(
            f"Physical round registered: {expected_slug} (status={round_status}, token={up_token}) "
            f"[{self._rounds_completed_count}/{self.target_physical_rounds} full rounds completed]"
        )
        return True

    def _close_current_round(self) -> None:
        """Close physical round asynchronously using in-memory counts (zero full-table queries!)."""
        if self._current_round_slug is None:
            return

        is_partial = self._is_current_round_warmup_partial
        final_status = "WARMUP_PARTIAL" if is_partial else "COMPLETED"

        self._enqueue_write("round_complete", {
            "round_slug": self._current_round_slug,
            "experiment_id": self.experiment_id,
            "total_samples": self._current_round_sample_count,
            "valid_samples": self._current_round_valid_count,
            "status": final_status,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })

        if is_partial:
            logger.info(
                f"Enqueued completion for WARMUP_PARTIAL round {self._current_round_slug}: "
                f"{self._current_round_valid_count}/{self._current_round_sample_count} valid samples. "
                "Does NOT increment completed full rounds count."
            )
        else:
            self._rounds_completed_count += 1
            logger.info(
                f"Enqueued completion for full round {self._current_round_slug}: "
                f"{self._current_round_valid_count}/{self._current_round_sample_count} valid samples. "
                f"Completed full rounds: {self._rounds_completed_count}/{self.target_physical_rounds}."
            )

        self._current_round_slug = None
        self._is_current_round_warmup_partial = False
        self._current_round_sample_count = 0
        self._current_round_valid_count = 0

    # ==========================================================================
    # Pilot SLA Fail-Fast Enforcement
    # ==========================================================================

    def _check_pilot_fail_fast(self, elapsed_sec: float) -> None:
        """Verify strict SLA compliance during pilot execution."""
        if not self.is_pilot:
            return

        if elapsed_sec > 30.0 and self._aggtrade_events_received == 0:
            raise RuntimeError("PILOT FAIL-FAST: 0 trades received after 30s!")

        if self._delta_events_received >= 15 and self._poly_source_ts_present == 0:
            raise RuntimeError("PILOT FAIL-FAST: 0 source timestamps on Polymarket delta events!")

        if self._parser_exception_count > 30:
            raise RuntimeError("PILOT FAIL-FAST: Excessive parser exceptions (>30)!")

        if elapsed_sec > 60.0:
            bn_tot = self._binance_source_ts_present + self._binance_source_ts_missing
            if bn_tot >= 50:
                bn_cov = self._binance_source_ts_present / bn_tot
                if bn_cov < 0.99:
                    raise RuntimeError(f"PILOT FAIL-FAST: Binance TS coverage {bn_cov*100:.1f}% < 99%!")

            poly_tot = self._poly_source_ts_present + self._poly_source_ts_missing
            if poly_tot >= 50:
                poly_cov = self._poly_source_ts_present / poly_tot
                if poly_cov < 0.95:
                    raise RuntimeError(f"PILOT FAIL-FAST: Poly TS coverage {poly_cov*100:.1f}% < 95%!")

        if elapsed_sec > 120.0 and self._valid_post_warmup_samples_count >= 30:
            tf_cov = self._non_null_tf_60s_count / self._valid_post_warmup_samples_count
            if tf_cov < 0.95:
                raise RuntimeError(f"PILOT FAIL-FAST: Taker-flow 60s coverage {tf_cov*100:.1f}% < 95%!")

    # ==========================================================================
    # Main Execution Loop
    # ==========================================================================

    def run(self) -> None:
        """Execute the cadence-fixed 1-second lead-lag replication collection loop."""
        mode_label = f"PILOT ({PILOT_PHYSICAL_ROUNDS} ROUNDS)" if self.is_pilot else f"PRODUCTION ({self.target_physical_rounds} ROUNDS)"
        logger.info(f"Starting LeadLagCollectorV3 in {mode_label} mode (PID {os.getpid()})...")

        self._acquire_lock()
        self._start_time_sec = time.time()
        self._start_persistence_worker()

        def _handle_signal(signum: int, _frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating clean v3 shutdown...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        self._start_binance_worker()
        self._start_polymarket_worker()

        if self._rounds_completed_count >= self.target_physical_rounds:
            logger.info(
                f"Target of {self.target_physical_rounds} full rounds already satisfied in DB "
                f"({self._rounds_completed_count} existing). Nothing to collect."
            )
            self._release_lock()
            return

        if self.wait_for_boundary:
            now = time.time()
            start_epoch = (int(now) // 300) * 300
            sec_into_round = now - start_epoch
            if sec_into_round > 1.0:
                next_boundary = start_epoch + 300
                logger.info(
                    f"wait_for_boundary=True: feeds warming up. "
                    f"Waiting {next_boundary - now:.2f}s until boundary {next_boundary}..."
                )
                upcoming_slug = self.contract_mgr.derive_round_slug(next_boundary + 10)
                round_info = self._fetch_round_info_with_backoff(upcoming_slug)
                if round_info:
                    self._upcoming_round_slug = upcoming_slug
                    self._upcoming_round_info = round_info
                    logger.info(f"Prefetched boundary round info: {upcoming_slug}")

                while not self._stop_event.is_set() and time.time() < next_boundary:
                    remain = next_boundary - time.time()
                    if remain > 0.05:
                        time.sleep(min(remain - 0.01, 0.25))
                    else:
                        time.sleep(0.001)

        try:
            now = time.time()
            if now - int(now) < 0.2:
                next_sample_sec = int(now)
            else:
                next_sample_sec = int(now) + 1

            while not self._stop_event.is_set():
                now = time.time()
                elapsed = now - self._start_time_sec

                self._check_pilot_fail_fast(elapsed)
                self._check_disk_space()

                if self._rounds_completed_count >= self.target_physical_rounds:
                    logger.info(f"Target of {self.target_physical_rounds} physical rounds reached! Closing cleanly.")
                    break

                sleep_sec = next_sample_sec - now
                if sleep_sec > 0.002:
                    time.sleep(sleep_sec)

                sample_sec_ms = next_sample_sec * 1000
                active = self._ensure_active_round(time.time())
                if not active:
                    logger.info(f"Target of {self.target_physical_rounds} full rounds reached! Exiting loop.")
                    break

                self._capture_sample(sample_sec_ms)
                self._emit_heartbeat()
                next_sample_sec += 1

        except Exception as e:
            logger.exception(f"Unhandled error in lead-lag v3 collector main loop: {e}")
            raise
        finally:
            if self._current_round_slug is not None:
                self._close_current_round()
            self._emit_heartbeat()
            self._stop_event.set()

            # Drain queue and wait for worker thread to finish
            if self._writer_thread and self._writer_thread.is_alive():
                logger.info("Waiting for persistence worker to finish draining write queue...")
                self._writer_thread.join(timeout=10.0)

            self._release_lock()
            logger.info("LeadLagCollectorV3 stopped cleanly.")
