"""Public unauthenticated order book fetcher and parser for Polymarket CLOB.

Strictly read-only GET operations:
- Endpoint: GET https://clob.polymarket.com/book?token_id={token_id}
- Zero API keys or authentication headers
- Validates token ID (token mismatch strictly invalidates book)
- Distinguishes source_event_timestamp vs received_at_ms
- Rejects crossed order books (best_bid >= best_ask)
- Sorts bids descending and asks ascending
- Distinguishes observed native market consensus from cross-outcome implied diagnostic:
    * Primary market baseline: native_up_mid (None if native UP book unavailable)
    * Secondary diagnostic: cross_outcome_implied_mid (never confused with observed)
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
    source_event_timestamp_ms: int | None
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

    @property
    def timestamp_ms(self) -> int:
        """Backward compatibility: return source timestamp if available, else received."""
        return self.source_event_timestamp_ms if self.source_event_timestamp_ms is not None else self.received_at_ms

    @property
    def receipt_age_ms(self) -> int:
        return max(0, int(time.time() * 1000) - self.received_at_ms)

    @property
    def source_age_ms(self) -> int | None:
        if self.source_event_timestamp_ms is None:
            return None
        return max(0, int(time.time() * 1000) - self.source_event_timestamp_ms)


@dataclass(frozen=True)
class PolymarketMarketState:
    """Combined order book state for UP and DOWN tokens.

    Preserves strict separation between observed native quotes and cross-outcome
    implied diagnostic calculations.
    """

    captured_at_ms: int
    up_book: ValidatedOrderBook
    down_book: ValidatedOrderBook

    # Native observed quotes (ground truth market baseline)
    native_up_bid: float | None
    native_up_ask: float | None
    native_up_mid: float | None

    native_down_bid: float | None
    native_down_ask: float | None
    native_down_mid: float | None

    # Cross-outcome implied calculations (secondary diagnostic only)
    cross_outcome_implied_up_bid: float | None
    cross_outcome_implied_up_ask: float | None
    cross_outcome_implied_mid: float | None

    # Primary baseline benchmark: observed native_up_mid (None if unavailable)
    market_q_primary: float | None
    # Secondary diagnostic: implied from DOWN book complementarity
    market_q_implied_cross_outcome: float | None

    spread: float | None
    is_valid: bool
    rejection_reason: str | None = None

    # Backward compatibility properties
    @property
    def market_q(self) -> float | None:
        """Primary market probability benchmark (observed native UP midpoint)."""
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


def parse_and_validate_book(
    raw_payload: dict[str, Any],
    expected_token_id: str,
    received_at_ms: int,
) -> ValidatedOrderBook:
    """Sort, validate, and check invariants for a raw order book payload."""
    if not isinstance(raw_payload, dict):
        return ValidatedOrderBook(
            token_id=expected_token_id,
            source_event_timestamp_ms=None,
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

    # Invariant A2: Token mismatch strictly invalidates the book
    asset_id = str(raw_payload.get("asset_id", "")).strip()
    if asset_id and expected_token_id and asset_id != expected_token_id:
        logger.warning(
            f"Order book asset_id mismatch: received '{asset_id}' != expected '{expected_token_id}'. Invalidating book."
        )
        return ValidatedOrderBook(
            token_id=expected_token_id,
            source_event_timestamp_ms=None,
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

    # Invariant A9: Distinguish source event timestamp from receipt timestamp
    source_ts: int | None = None
    if "timestamp" in raw_payload:
        try:
            source_ts = int(raw_payload["timestamp"])
        except (ValueError, TypeError):
            source_ts = None

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
        source_event_timestamp_ms=source_ts,
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
            source_event_timestamp_ms=None,
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
        """Fetch order books for both UP and DOWN tokens and compute primary and implied baselines."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        up_book = self.fetch_book(up_token_id)
        down_book = self.fetch_book(down_token_id)

        # Freshness check based on receipt time (receipt_age)
        if now - up_book.received_at_ms > max_age_ms or now - down_book.received_at_ms > max_age_ms:
            return PolymarketMarketState(
                captured_at_ms=now,
                up_book=up_book,
                down_book=down_book,
                native_up_bid=up_book.best_bid,
                native_up_ask=up_book.best_ask,
                native_up_mid=up_book.midpoint,
                native_down_bid=down_book.best_bid,
                native_down_ask=down_book.best_ask,
                native_down_mid=down_book.midpoint,
                cross_outcome_implied_up_bid=None,
                cross_outcome_implied_up_ask=None,
                cross_outcome_implied_mid=None,
                market_q_primary=None,
                market_q_implied_cross_outcome=None,
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
                native_up_bid=up_book.best_bid,
                native_up_ask=up_book.best_ask,
                native_up_mid=up_book.midpoint,
                native_down_bid=down_book.best_bid,
                native_down_ask=down_book.best_ask,
                native_down_mid=down_book.midpoint,
                cross_outcome_implied_up_bid=None,
                cross_outcome_implied_up_ask=None,
                cross_outcome_implied_mid=None,
                market_q_primary=None,
                market_q_implied_cross_outcome=None,
                spread=up_book.spread,
                is_valid=False,
                rejection_reason="CROSSED_ORDER_BOOK",
            )

        # 1. Native Observed Market Baseline (Requirement A1)
        # Primary benchmark is strictly the observed native UP midpoint when a valid
        # two-sided native UP book exists. If unavailable, do not fabricate: set to None.
        native_up_mid = up_book.midpoint if (up_book.is_valid and up_book.midpoint is not None) else None
        market_q_primary = native_up_mid

        # 2. Cross-Outcome Implied Diagnostic (Requirement A1)
        # An ask on DOWN at P_D implies a bid on UP at (1.0 - P_D)
        # A bid on DOWN at P_D implies an ask on UP at (1.0 - P_D)
        implied_up_bid: float | None = None
        implied_up_ask: float | None = None
        if down_book.best_ask is not None:
            implied_up_bid = round(1.0 - down_book.best_ask, 4)
        if down_book.best_bid is not None:
            implied_up_ask = round(1.0 - down_book.best_bid, 4)

        implied_mid: float | None = None
        if implied_up_bid is not None and implied_up_ask is not None and implied_up_bid < implied_up_ask:
            implied_mid = round((implied_up_bid + implied_up_ask) / 2.0, 4)

        # Validity: book is valid if we have at least one valid probability measure
        # But market_q_primary strictly reflects native observed mid
        is_valid = bool(
            (market_q_primary is not None and 0.001 <= market_q_primary <= 0.999)
            or (implied_mid is not None and 0.001 <= implied_mid <= 0.999)
        )
        rejection_reason = None if is_valid else "NO_VALID_POLY_MARKET_ODDS"

        return PolymarketMarketState(
            captured_at_ms=now,
            up_book=up_book,
            down_book=down_book,
            native_up_bid=up_book.best_bid,
            native_up_ask=up_book.best_ask,
            native_up_mid=up_book.midpoint,
            native_down_bid=down_book.best_bid,
            native_down_ask=down_book.best_ask,
            native_down_mid=down_book.midpoint,
            cross_outcome_implied_up_bid=implied_up_bid,
            cross_outcome_implied_up_ask=implied_up_ask,
            cross_outcome_implied_mid=implied_mid,
            market_q_primary=market_q_primary,
            market_q_implied_cross_outcome=implied_mid,
            spread=up_book.spread,
            is_valid=is_valid,
            rejection_reason=rejection_reason,
        )

    # Alias for API compatibility
    collect_market_state = fetch_market_state
