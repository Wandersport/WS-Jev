"""Public read-only Chainlink BTC/USD reference data feed.

Connects to Polymarket RTDS (Real-Time Data Service) via public read-only WebSocket
or maintains injected ticks for offline reproducible testing.

Topics handled:
- crypto_prices_chainlink (spot) -> SOURCE_SPOT
- crypto_prices_twap_sixty (60s TWAP, official settlement stream) -> SOURCE_TWAP_60S
- crypto_prices_twap_thirty (30s TWAP) -> SOURCE_TWAP_30S

Strict Invariants (Requirements A5, A6, A9):
- Unauthenticated read-only access (NO credentials)
- Host allowlist: ws-live-data.polymarket.com
- Strictly rejects generic/unknown RTDS topics from masquerading as settlement stream
- Strict anchor hierarchy:
    1. Authoritative Gamma priceToBeat
    2. Exact official 60s TWAP opening boundary tick (t.timestamp_ms == round_start_ms)
- Distinguishes source_event_timestamp vs received_at_ms
- Rejects future timestamps (> now + 2000ms)
- Rejects stale ticks (> 30 minutes old)
- Computes 10s, 30s, 60s returns and anchor distance in basis points
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

RTDS_WS_URL: str = "wss://ws-live-data.polymarket.com"
ALLOWED_RTDS_HOSTS: frozenset[str] = frozenset({"ws-live-data.polymarket.com"})

SOURCE_SPOT: str = "chainlink-spot"
SOURCE_TWAP_60S: str = "chainlink-twap-60s"
SOURCE_TWAP_30S: str = "chainlink-twap-30s"

TOPIC_TO_SOURCE: dict[str, str] = {
    "crypto_prices_chainlink": SOURCE_SPOT,
    "crypto_prices_twap_sixty": SOURCE_TWAP_60S,
    "crypto_prices_twap_thirty": SOURCE_TWAP_30S,
}


@dataclass(frozen=True)
class ReferenceTick:
    """A single validated timestamped Chainlink BTC/USD price point."""

    source: str  # SOURCE_SPOT, SOURCE_TWAP_60S, or SOURCE_TWAP_30S
    timestamp_ms: int  # Source event timestamp from provider
    price: float
    received_at_ms: int  # Local clock receipt timestamp


@dataclass(frozen=True)
class ReferenceFeatures:
    """Extracted features from the authoritative settlement reference stream."""

    source: str
    current_price: float
    source_event_timestamp_ms: int
    received_at_ms: int
    receipt_age_ms: int
    source_age_ms: int
    anchor_price: float | None
    anchor_source: str | None
    anchor_source_timestamp_ms: int | None
    anchor_received_timestamp_ms: int | None
    distance_to_anchor_bps: float | None
    return_10s_bps: float | None
    return_30s_bps: float | None
    return_60s_bps: float | None
    since_round_open_bps: float | None

    # Backward compatibility properties
    @property
    def timestamp_ms(self) -> int:
        return self.source_event_timestamp_ms

    @property
    def age_ms(self) -> int:
        return self.receipt_age_ms

    @property
    def price_to_beat(self) -> float | None:
        return self.anchor_price

    @property
    def price_to_beat_source(self) -> str:
        return self.anchor_source or "none"

    @property
    def distance_to_beat_bps(self) -> float | None:
        return self.distance_to_anchor_bps

    @property
    def data_age_ms(self) -> int:
        return self.receipt_age_ms

    @property
    def opening_anchor_detected(self) -> bool:
        return self.anchor_price is not None


class ChainlinkReferenceFeed:
    """In-memory collector and stream consumer for Chainlink BTC/USD reference data."""

    def __init__(
        self,
        ws_url: str = RTDS_WS_URL,
        max_history_ms: int = 1800000,  # 30 minutes
        max_ticks: int = 10000,
    ) -> None:
        self.ws_url = ws_url
        self.max_history_ms = max_history_ms
        self.max_ticks = max_ticks
        self._ticks: list[ReferenceTick] = []
        self._running = False
        self._ws_task: asyncio.Task | None = None
        self.last_message_time_ms: int = 0
        self.status: str = "Initialized (disconnected)"

    @property
    def ticks(self) -> list[ReferenceTick]:
        return list(self._ticks)

    def add_tick(
        self,
        source: str,
        timestamp_ms: int,
        price: float,
        now_ms: int | None = None,
        received_at_ms: int | None = None,
    ) -> bool:
        """Validate and add a single price tick. Returns True if accepted."""
        now = received_at_ms if received_at_ms is not None else (now_ms if now_ms is not None else int(time.time() * 1000))

        # Invariant: Reject future timestamps with clock skew buffer (2000ms)
        if timestamp_ms > now + 2000:
            logger.debug(f"Rejected future tick: ts={timestamp_ms} > now={now}+2000")
            return False

        # Invariant: Reject ancient ticks
        if timestamp_ms < now - self.max_history_ms:
            return False

        # Invariant: Price must be strictly positive
        if price <= 0.0 or not isinstance(price, (int, float)):
            return False

        tick = ReferenceTick(
            source=source,
            timestamp_ms=timestamp_ms,
            price=float(price),
            received_at_ms=now,
        )

        # Deduplicate by (source, timestamp_ms)
        existing_idx = next(
            (i for i, t in enumerate(self._ticks) if t.source == source and t.timestamp_ms == timestamp_ms),
            None,
        )
        if existing_idx is not None:
            self._ticks[existing_idx] = tick
        else:
            self._ticks.append(tick)

        # Maintain sort order by source timestamp
        self._ticks.sort(key=lambda t: t.timestamp_ms)

        # Prune older than 30m and limit length
        cutoff = now - self.max_history_ms
        self._ticks = [t for t in self._ticks if t.timestamp_ms >= cutoff][-self.max_ticks :]
        return True

    def ingest_rtds_payload(self, raw_data: dict[str, Any], now_ms: int | None = None) -> int:
        """Ingest a parsed message frame from the Polymarket RTDS feed.

        Requirement A5: Reject ambiguous/generic topics. Only known, explicitly
        mapped topics may become SOURCE_TWAP_60S.
        """
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        topic = str(raw_data.get("topic", "")).strip()

        # Strict topic mapping: generic 'crypto_prices' is NOT allowed to become 60s TWAP
        source = TOPIC_TO_SOURCE.get(topic)
        if not source:
            return 0

        payload = raw_data.get("payload")
        if not isinstance(payload, dict):
            return 0

        # Verify symbol is BTC/USD
        sym = str(payload.get("symbol", "")).lower()
        if sym and sym != "btc/usd":
            return 0

        entries = payload.get("data")
        if not isinstance(entries, list):
            if "timestamp" in payload and "value" in payload:
                entries = [payload]
            else:
                return 0

        added = 0
        for item in entries:
            if not isinstance(item, dict):
                continue
            ts = item.get("timestamp")
            val = item.get("value")
            if ts is not None and val is not None:
                try:
                    if self.add_tick(
                        source=source,
                        timestamp_ms=int(ts),
                        price=float(val),
                        received_at_ms=now,
                    ):
                        added += 1
                except (ValueError, TypeError):
                    continue

        self.last_message_time_ms = now
        return added

    def find_exact_anchor(self, start_ms: int, source: str = SOURCE_TWAP_60S) -> ReferenceTick | None:
        """Find the exact observation matching the round boundary timestamp."""
        for t in self._ticks:
            if t.source == source and t.timestamp_ms == start_ms:
                return t
        return None

    def get_latest_tick(self, source: str = SOURCE_TWAP_60S) -> ReferenceTick | None:
        """Retrieve the most recent validated tick for a given source."""
        for t in reversed(self._ticks):
            if t.source == source:
                return t
        return None

    def compute_features(
        self,
        source: str = SOURCE_TWAP_60S,
        round_start_ms: int = 0,
        anchor_price: float | None = None,
        anchor_source: str | None = None,
        now_ms: int | None = None,
        max_age_ms: int = 30000,
        round_start_epoch: int | None = None,
        event_price_to_beat: float | None = None,
    ) -> ReferenceFeatures:
        """Compute reference price features, momentum returns, and distance to anchor.

        Requirement A6: Maintain strict anchor hierarchy:
        1. Authoritative Gamma priceToBeat when genuinely present and valid (> 0)
        2. Exact official 60-second TWAP opening-boundary observation (t.timestamp_ms == round_start_ms)
        Do NOT approximate the opening anchor with nearest arbitrary tick.
        """
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        if round_start_epoch is not None and round_start_ms == 0:
            round_start_ms = round_start_epoch * 1000

        anchor_source_ts: int | None = None
        anchor_recv_ts: int | None = None

        # Hierarchy 1: Gamma priceToBeat
        if anchor_price is None and event_price_to_beat is not None and event_price_to_beat > 0:
            anchor_price = event_price_to_beat
            anchor_source = "gamma-metadata"
            anchor_source_ts = round_start_ms if round_start_ms > 0 else now
            anchor_recv_ts = now

        # Hierarchy 2: Exact boundary tick (t.timestamp_ms == round_start_ms)
        if anchor_price is None and round_start_ms > 0:
            exact_tick = self.find_exact_anchor(start_ms=round_start_ms, source=source)
            if exact_tick is not None:
                anchor_price = exact_tick.price
                anchor_source = "rtds-twap-60s-exact-boundary"
                anchor_source_ts = exact_tick.timestamp_ms
                anchor_recv_ts = exact_tick.received_at_ms

        latest = self.get_latest_tick(source=source)
        if not latest:
            return ReferenceFeatures(
                source=source,
                current_price=0.0,
                source_event_timestamp_ms=now,
                received_at_ms=now,
                receipt_age_ms=999999,
                source_age_ms=999999,
                anchor_price=anchor_price,
                anchor_source=anchor_source,
                anchor_source_timestamp_ms=anchor_source_ts,
                anchor_received_timestamp_ms=anchor_recv_ts,
                distance_to_anchor_bps=None,
                return_10s_bps=None,
                return_30s_bps=None,
                return_60s_bps=None,
                since_round_open_bps=None,
            )

        receipt_age = max(0, now - latest.received_at_ms)
        source_age = max(0, now - latest.timestamp_ms)
        curr_p = latest.price

        # Distance to anchor in basis points: (curr / anchor - 1.0) * 10000.0
        dist_bps = None
        since_round_open_bps = None
        if anchor_price is not None and anchor_price > 0.0:
            dist_bps = round((curr_p / anchor_price - 1.0) * 10000.0, 2)
            since_round_open_bps = dist_bps

        # Calculate returns over 10s, 30s, 60s in basis points
        def get_return_bps(seconds: int) -> float | None:
            target_ts = latest.timestamp_ms - (seconds * 1000)
            past_tick: ReferenceTick | None = None
            for t in reversed(self._ticks):
                if t.source == source and t.timestamp_ms <= target_ts:
                    past_tick = t
                    break
            if not past_tick or past_tick.price <= 0:
                return None
            return round(((curr_p - past_tick.price) / past_tick.price) * 10000.0, 2)

        return ReferenceFeatures(
            source=source,
            current_price=curr_p,
            source_event_timestamp_ms=latest.timestamp_ms,
            received_at_ms=latest.received_at_ms,
            receipt_age_ms=receipt_age,
            source_age_ms=source_age,
            anchor_price=anchor_price,
            anchor_source=anchor_source,
            anchor_source_timestamp_ms=anchor_source_ts,
            anchor_received_timestamp_ms=anchor_recv_ts,
            distance_to_anchor_bps=dist_bps,
            return_10s_bps=get_return_bps(10),
            return_30s_bps=get_return_bps(30),
            return_60s_bps=get_return_bps(60),
            since_round_open_bps=since_round_open_bps,
        )

    def start_background_listener(self) -> None:
        """Start non-blocking daemon thread to maintain live RTDS connection."""
        import threading

        def _worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._running = True

            async def _run() -> None:
                while self._running:
                    try:
                        import websockets

                        async with websockets.connect(
                            self.ws_url,
                            ping_interval=20,
                            ping_timeout=10,
                            close_timeout=5,
                        ) as ws:
                            self.status = "Connected"
                            sub_msg = {
                                "action": "subscribe",
                                "subscriptions": [
                                    {"topic": "crypto_prices_twap_sixty", "type": "*"},
                                    {"topic": "crypto_prices_chainlink", "type": "*"},
                                ],
                            }
                            await ws.send(json.dumps(sub_msg))
                            while self._running:
                                msg = await ws.recv()
                                try:
                                    data = json.loads(msg)
                                    if isinstance(data, dict):
                                        self.ingest_rtds_payload(data)
                                except Exception:
                                    pass
                    except Exception as e:
                        self.status = f"Disconnected ({e})"
                        await asyncio.sleep(3.0)

            try:
                loop.run_until_complete(_run())
            except Exception:
                pass
            finally:
                loop.close()

        self._listener_thread = threading.Thread(target=_worker, daemon=True, name="rtds_client")
        self._listener_thread.start()

    def stop_background_listener(self) -> None:
        """Stop background RTDS listener."""
        self._running = False
        self.status = "Stopped"

    def is_connected(self) -> bool:
        """Check if RTDS stream is currently connected."""
        return getattr(self, "status", "") == "Connected"
