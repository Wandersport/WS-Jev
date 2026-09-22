"""Public unauthenticated order book fetcher and parser for Polymarket CLOB.

Strictly read-only GET operations:
- Endpoint: GET https://clob.polymarket.com/book?token_id={token_id}
- Zero API keys or authentication headers
- Validates token ID, prices, sizes, and timestamps
- Rejects crossed order books (best_bid >= best_ask)
- Sorts bids descending and asks ascending
- Computes best bid, best ask, midpoints, spreads, and market_q
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CLOB_API_BASE: str = "https://clob.polymarket.com"


@dataclass(frozen=True)
class BookLevel:
    """A single price-level in the order book."""

    price: float
    size: float


@dataclass(frozen=True)
class ValidatedOrderBook:
    """A fully validated, sorted public order book for a single outcome token."""

    token_id: str
    timestamp_ms: int
    received_at_ms: int
    bids: tuple[BookLevel, ...]  # Sorted descending by price
    asks: tuple[BookLevel, ...]  # Sorted ascending by price
    best_bid: float | None
    best_ask: float | None
    midpoint: float | None
    spread: float | None
    is_crossed: bool
    is_valid: bool
    min_order_size: float | None


@dataclass(frozen=True)
class PolymarketMarketState:
    """Combined order book state for UP and DOWN tokens with market consensus probability."""

    captured_at_ms: int
    up_book: ValidatedOrderBook
    down_book: ValidatedOrderBook
    up_best_bid: float | None
    up_best_ask: float | None
    down_best_bid: float | None
    down_best_ask: float | None
    market_q: float | None  # Primary market probability: (up_best_bid + up_best_ask) / 2
    spread: float | None
    is_valid: bool
    rejection_reason: str | None = None


def parse_and_validate_book(
    raw_payload: dict[str, Any],
    expected_token_id: str,
    received_at_ms: int,
) -> ValidatedOrderBook:
    """Sort, validate, and check invariants for a raw order book payload."""
    if not isinstance(raw_payload, dict):
        return ValidatedOrderBook(
            token_id=expected_token_id,
            timestamp_ms=received_at_ms,
            received_at_ms=received_at_ms,
            bids=(),
            asks=(),
            best_bid=None,
            best_ask=None,
            midpoint=None,
            spread=None,
            is_crossed=False,
            is_valid=False,
            min_order_size=None,
        )

    asset_id = str(raw_payload.get("asset_id", "")).strip()
    if asset_id != expected_token_id:
        logger.warning(f"Order book asset_id mismatch: {asset_id} != {expected_token_id}")

    try:
        ts_val = int(raw_payload.get("timestamp", received_at_ms))
    except (ValueError, TypeError):
        ts_val = received_at_ms

    try:
        min_sz = float(raw_payload.get("min_order_size", 0.0))
    except (ValueError, TypeError):
        min_sz = None

    # Parse and filter bids (0.0 < price < 1.0, size > 0.0)
    raw_bids = raw_payload.get("bids") or []
    parsed_bids: list[BookLevel] = []
    for item in raw_bids:
        if isinstance(item, dict):
            try:
                p = float(item["price"])
                sz = float(item["size"])
                if 0.0 < p < 1.0 and sz > 0.0:
                    parsed_bids.append(BookLevel(price=round(p, 4), size=round(sz, 4)))
            except (KeyError, ValueError, TypeError):
                continue

    # Parse and filter asks (0.0 < price < 1.0, size > 0.0)
    raw_asks = raw_payload.get("asks") or []
    parsed_asks: list[BookLevel] = []
    for item in raw_asks:
        if isinstance(item, dict):
            try:
                p = float(item["price"])
                sz = float(item["size"])
                if 0.0 < p < 1.0 and sz > 0.0:
                    parsed_asks.append(BookLevel(price=round(p, 4), size=round(sz, 4)))
            except (KeyError, ValueError, TypeError):
                continue

    # Sort bids strictly descending by price
    parsed_bids.sort(key=lambda b: b.price, reverse=True)
    # Sort asks strictly ascending by price
    parsed_asks.sort(key=lambda a: a.price)

    best_bid = parsed_bids[0].price if parsed_bids else None
    best_ask = parsed_asks[0].price if parsed_asks else None

    # Invariant: crossed book detection
    is_crossed = False
    if best_bid is not None and best_ask is not None and best_bid >= best_ask:
        is_crossed = True

    midpoint = None
    spread = None
    if best_bid is not None and best_ask is not None and not is_crossed:
        midpoint = round((best_bid + best_ask) / 2.0, 4)
        spread = round(best_ask - best_bid, 4)

    is_valid = bool(best_bid is not None and best_ask is not None and not is_crossed)

    return ValidatedOrderBook(
        token_id=expected_token_id,
        timestamp_ms=ts_val,
        received_at_ms=received_at_ms,
        bids=tuple(parsed_bids),
        asks=tuple(parsed_asks),
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        spread=spread,
        is_crossed=is_crossed,
        is_valid=is_valid,
        min_order_size=min_sz,
    )


class PolymarketBookCollector:
    """Fetches public order books via unauthenticated GET /book?token_id=..."""

    def __init__(
        self,
        clob_base: str = CLOB_API_BASE,
        request_timeout: float = 5.0,
    ) -> None:
        self.clob_base = clob_base.rstrip("/")
        self.request_timeout = request_timeout
        self._injected_books: dict[str, ValidatedOrderBook] = {}

    def inject_book(self, token_id: str, book: ValidatedOrderBook) -> None:
        """Inject an in-memory order book for deterministic testing."""
        self._injected_books[token_id] = book

    def fetch_book(self, token_id: str) -> ValidatedOrderBook:
        """Fetch and validate the order book for a single outcome token."""
        if token_id in self._injected_books:
            return self._injected_books[token_id]

        url = f"{self.clob_base}/book?token_id={urllib.parse.quote(token_id)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pm-research-btc5m/1.0",
                "Accept": "application/json",
            },
            method="GET",
        )
        t_recv = int(time.time() * 1000)
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                if resp.status == 200:
                    raw_data = json.loads(resp.read().decode("utf-8"))
                    return parse_and_validate_book(raw_data, token_id, t_recv)
                logger.warning(f"CLOB /book returned HTTP {resp.status} for token {token_id}")
        except Exception as e:
            logger.warning(f"Error fetching order book for {token_id}: {e}")

        # Return empty invalid book on failure
        return ValidatedOrderBook(
            token_id=token_id,
            timestamp_ms=t_recv,
            received_at_ms=t_recv,
            bids=(),
            asks=(),
            best_bid=None,
            best_ask=None,
            midpoint=None,
            spread=None,
            is_crossed=False,
            is_valid=False,
            min_order_size=None,
        )

    def fetch_market_state(
        self,
        up_token_id: str,
        down_token_id: str,
        now_ms: int | None = None,
        max_age_ms: int = 15000,
    ) -> PolymarketMarketState:
        """Fetch order books for both UP and DOWN tokens and compute market_q."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        up_book = self.fetch_book(up_token_id)
        down_book = self.fetch_book(down_token_id)

        # Freshness check
        if now - up_book.received_at_ms > max_age_ms or now - down_book.received_at_ms > max_age_ms:
            return PolymarketMarketState(
                captured_at_ms=now,
                up_book=up_book,
                down_book=down_book,
                up_best_bid=up_book.best_bid,
                up_best_ask=up_book.best_ask,
                down_best_bid=down_book.best_bid,
                down_best_ask=down_book.best_ask,
                market_q=None,
                spread=up_book.spread,
                is_valid=False,
                rejection_reason="SKIP_STALE_POLYMARKET_BOOK",
            )

        # Crossed book check
        if up_book.is_crossed or down_book.is_crossed:
            return PolymarketMarketState(
                captured_at_ms=now,
                up_book=up_book,
                down_book=down_book,
                up_best_bid=up_book.best_bid,
                up_best_ask=up_book.best_ask,
                down_best_bid=down_book.best_bid,
                down_best_ask=down_book.best_ask,
                market_q=None,
                spread=up_book.spread,
                is_valid=False,
                rejection_reason="CROSSED_ORDER_BOOK",
            )

        # Synthesize effective UP best bid and ask using binary market complementarity:
        # Ask on DOWN at P_D is equivalent to Bid on UP at (1.0 - P_D)
        # Bid on DOWN at P_D is equivalent to Ask on UP at (1.0 - P_D)
        cand_up_bids: list[float] = []
        cand_up_asks: list[float] = []

        if up_book.best_bid is not None:
            cand_up_bids.append(up_book.best_bid)
        if down_book.best_ask is not None:
            cand_up_bids.append(round(1.0 - down_book.best_ask, 4))

        if up_book.best_ask is not None:
            cand_up_asks.append(up_book.best_ask)
        if down_book.best_bid is not None:
            cand_up_asks.append(round(1.0 - down_book.best_bid, 4))

        eff_up_bid = max(cand_up_bids) if cand_up_bids else None
        eff_up_ask = min(cand_up_asks) if cand_up_asks else None

        # Primary market consensus probability market_q
        market_q: float | None = None
        spread: float | None = None

        if eff_up_bid is not None and eff_up_ask is not None:
            market_q = round((eff_up_bid + eff_up_ask) / 2.0, 4)
            spread = round(max(0.0, eff_up_ask - eff_up_bid), 4)
        elif eff_up_ask is not None and eff_up_ask <= 0.02:
            # Bound at lower extreme (market heavily favoring DOWN)
            market_q = eff_up_ask
            spread = eff_up_ask
        elif eff_up_bid is not None and eff_up_bid >= 0.98:
            # Bound at upper extreme (market heavily favoring UP)
            market_q = eff_up_bid
            spread = round(1.0 - eff_up_bid, 4)

        is_valid = bool(market_q is not None and (0.001 <= market_q <= 0.999))
        rejection_reason = None if is_valid else "NO_VALID_POLY_MARKET_ODDS"

        return PolymarketMarketState(
            captured_at_ms=now,
            up_book=up_book,
            down_book=down_book,
            up_best_bid=eff_up_bid,
            up_best_ask=eff_up_ask,
            down_best_bid=down_book.best_bid,
            down_best_ask=down_book.best_ask,
            market_q=market_q,
            spread=spread or up_book.spread,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        )

    # Alias for API compatibility
    collect_market_state = fetch_market_state

