"""RIGO: Market ingestion, validation, and normalization stage."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pm_research.config import SystemConfig
from pm_research.domain.models import (
    Market,
    MarketQuote,
    MarketSnapshot,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Side,
)
from pm_research.utils import ensure_utc, generate_id, now_utc, parse_iso_utc

logger = logging.getLogger(__name__)


class RigoIngestor:
    """Ingests raw prediction market data, validates integrity, and normalizes to UTC domain models."""

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def normalize_market(self, raw: dict[str, Any], current_time: datetime | None = None) -> Market | None:
        """Validate and normalize a raw market payload into a Market domain object.

        Returns None if data is malformed or invalid.
        """
        now = current_time or now_utc()
        now = ensure_utc(now)

        try:
            market_id = str(raw["market_id"]).strip()
            question = str(raw.get("question", "")).strip()
            category = str(raw.get("category", "GENERAL")).strip().upper()
            status_str = str(raw.get("status", "ACTIVE")).strip().upper()

            if not market_id or not question:
                logger.warning("Rejecting market: missing market_id or question.")
                return None

            try:
                status = MarketStatus(status_str)
            except ValueError:
                logger.warning(f"Rejecting market {market_id}: invalid status {status_str}")
                return None

            # Parse resolution time
            raw_res_time = raw.get("resolution_time")
            if isinstance(raw_res_time, datetime):
                resolution_time = ensure_utc(raw_res_time)
            elif isinstance(raw_res_time, str):
                resolution_time = parse_iso_utc(raw_res_time)
            else:
                logger.warning(f"Rejecting market {market_id}: invalid resolution_time format.")
                return None

            # Parse quotes
            yes_bid = float(raw["yes_bid"]) if raw.get("yes_bid") is not None else None
            yes_ask = float(raw["yes_ask"]) if raw.get("yes_ask") is not None else None
            no_bid = float(raw["no_bid"]) if raw.get("no_bid") is not None else None
            no_ask = float(raw["no_ask"]) if raw.get("no_ask") is not None else None
            last_price = float(raw["last_price"]) if raw.get("last_price") is not None else None

            # Validate price bounds in [0.0, 1.0]
            for p, name in [
                (yes_bid, "yes_bid"),
                (yes_ask, "yes_ask"),
                (no_bid, "no_bid"),
                (no_ask, "no_ask"),
                (last_price, "last_price"),
            ]:
                if p is not None and not (0.0 <= p <= 1.0):
                    logger.warning(f"Rejecting market {market_id}: {name}={p} out of bounds [0, 1].")
                    return None

            # Validate bid <= ask if both exist
            if yes_bid is not None and yes_ask is not None and yes_bid > yes_ask:
                logger.warning(f"Rejecting market {market_id}: crossed book yes_bid {yes_bid} > yes_ask {yes_ask}")
                return None

            if no_bid is not None and no_ask is not None and no_bid > no_ask:
                logger.warning(f"Rejecting market {market_id}: crossed book no_bid {no_bid} > no_ask {no_ask}")
                return None

            # If no_ask is not explicitly given, in binary markets no_ask = 1 - yes_bid
            if no_ask is None and yes_bid is not None:
                no_ask = round(1.0 - yes_bid, 4)
            if no_bid is None and yes_ask is not None:
                no_bid = round(1.0 - yes_ask, 4)

            # Compute midpoint and spread
            midpoint = None
            spread = float(raw["spread"]) if raw.get("spread") is not None else None
            if yes_bid is not None and yes_ask is not None:
                midpoint = (yes_bid + yes_ask) / 2.0
                if spread is None:
                    spread = max(0.0, yes_ask - yes_bid)
            elif last_price is not None:
                midpoint = last_price

            liquidity = float(raw.get("liquidity", 0.0))
            volume_24h = float(raw.get("volume_24h", 0.0))

            if liquidity < 0.0 or volume_24h < 0.0:
                logger.warning(f"Rejecting market {market_id}: negative liquidity or volume.")
                return None

            quote = MarketQuote(
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                last_price=last_price,
                midpoint=midpoint,
                spread=spread,
                liquidity=liquidity,
                volume_24h=volume_24h,
            )

            # Parse order book if present
            order_book = None
            if "order_book" in raw and raw["order_book"]:
                ob_raw = raw["order_book"]
                bids = [
                    OrderBookLevel(float(lvl["price"]), float(lvl["quantity"]))
                    for lvl in ob_raw.get("bids", [])
                ]
                asks = [
                    OrderBookLevel(float(lvl["price"]), float(lvl["quantity"]))
                    for lvl in ob_raw.get("asks", [])
                ]
                # Sort bids descending, asks ascending
                bids.sort(key=lambda x: x.price, reverse=True)
                asks.sort(key=lambda x: x.price)
                order_book = OrderBook(bids=bids, asks=asks)

            # Parse optional resolved outcome strictly for non-ACTIVE markets
            resolved_outcome = None
            res_time_actual = None
            if status != MarketStatus.ACTIVE:
                if raw.get("resolved_outcome"):
                    res_str = str(raw["resolved_outcome"]).upper()
                    if res_str in (Side.YES.value, Side.NO.value):
                        resolved_outcome = Side(res_str)
                if raw.get("resolution_time_actual"):
                    res_time_actual = parse_iso_utc(str(raw["resolution_time_actual"]))

            created_at = (
                parse_iso_utc(str(raw["created_at"]))
                if "created_at" in raw
                else now
            )
            updated_at = (
                parse_iso_utc(str(raw["updated_at"]))
                if "updated_at" in raw
                else now
            )

            return Market(
                market_id=market_id,
                question=question,
                category=category,
                status=status,
                resolution_time=resolution_time,
                quote=quote,
                order_book=order_book,
                created_at=created_at,
                updated_at=updated_at,
                resolved_outcome=resolved_outcome,
                resolution_time_actual=res_time_actual,
                metadata=raw.get("metadata", {}),
            )

        except Exception as e:
            logger.warning(f"Error normalizing market data: {e}")
            return None

    def ingest_snapshots(
        self,
        raw_markets: list[dict[str, Any]],
        cycle_id: str,
        current_time: datetime | None = None,
    ) -> list[MarketSnapshot]:
        """Normalize a batch of raw markets into MarketSnapshot objects for a cycle."""
        now = ensure_utc(current_time or now_utc())
        snapshots: list[MarketSnapshot] = []

        for raw in raw_markets:
            market = self.normalize_market(raw, current_time=now)
            if market is None:
                continue

            snapshot = MarketSnapshot(
                snapshot_id=generate_id("snap"),
                cycle_id=cycle_id,
                market_id=market.market_id,
                timestamp=now,
                market=market,
            )
            snapshots.append(snapshot)

        return snapshots

    def ingest_snapshot(
        self,
        raw_market: dict[str, Any],
        cycle_id: str,
        current_time: datetime | None = None,
    ) -> MarketSnapshot | None:
        """Normalize a single raw market into a MarketSnapshot object."""
        res = self.ingest_snapshots([raw_market], cycle_id=cycle_id, current_time=current_time)
        return res[0] if res else None
