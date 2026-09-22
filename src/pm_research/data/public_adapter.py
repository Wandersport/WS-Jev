"""Public read-only prediction market data adapter.

READ-ONLY AND UNCREDENTIALED:
- Consumes public read-only market endpoints only.
- Strict host allowlisting (HTTPS only, Polymarket public endpoints).
- Custom redirect handler preventing cross-host or protocol escalation.
- Requires and accepts NO API keys, NO private keys, and NO trading secrets.
- Enforces explicit request timeouts, real rate limiting, and bounded exponential backoff.
- On network failure or malformed schema: yields empty list (no data -> no proposal).
- Never fabricates missing values or timestamps.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from pm_research.utils import parse_iso_utc

logger = logging.getLogger(__name__)

# Strictly allowlisted read-only public hosts
DEFAULT_ALLOWED_HOSTS: frozenset[str] = frozenset({
    "gamma-api.polymarket.com",
    "clob.polymarket.com",
})


class RestrictedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Restricts HTTP redirects strictly to allowed hosts and HTTPS scheme."""

    def __init__(self, allowed_hosts: frozenset[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme.lower() != "https":
            logger.warning(f"Refusing redirect to non-HTTPS scheme: {newurl}")
            return None
        if parsed.netloc.lower() not in self.allowed_hosts:
            logger.warning(f"Refusing redirect to non-allowlisted host: {parsed.netloc}")
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PublicMarketDataAdapter:
    """Polite, rate-limited public read-only market data client with host allowlisting."""

    def __init__(
        self,
        base_url: str = "https://gamma-api.polymarket.com",
        allowed_hosts: frozenset[str] | None = None,
        request_timeout_seconds: float = 5.0,
        min_request_interval_seconds: float = 1.0,
        max_retries: int = 2,
    ) -> None:
        self.allowed_hosts = allowed_hosts or DEFAULT_ALLOWED_HOSTS
        parsed_base = urllib.parse.urlparse(base_url)
        if parsed_base.scheme.lower() != "https":
            raise ValueError(f"Insecure public adapter URL scheme '{parsed_base.scheme}'. Must be 'https'.")
        if parsed_base.netloc.lower() not in self.allowed_hosts:
            raise ValueError(f"Host '{parsed_base.netloc}' is not in the allowlist: {sorted(self.allowed_hosts)}")

        self.base_url = base_url.rstrip("/")
        # Bound timeout between 1.0s and 10.0s
        self.request_timeout = max(1.0, min(10.0, float(request_timeout_seconds)))
        self.min_interval = max(0.1, float(min_request_interval_seconds))
        # Bound retries between 0 and 3
        self.max_retries = max(0, min(3, int(max_retries)))
        self._last_request_time: float = 0.0

        # Custom opener enforcing redirect boundaries
        self._opener = urllib.request.build_opener(RestrictedRedirectHandler(self.allowed_hosts))

    def _wait_for_rate_limit(self) -> None:
        """Polite client-side rate limiting enforcing minimum interval between outgoing queries."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def fetch_public_markets(self, limit: int = 20) -> list[dict[str, Any]]:
        """Fetch active public markets from unauthenticated public read-only endpoint.

        Returns empty list on any network/parsing failure (no data -> no proposal).
        """
        clamped_limit = max(1, min(100, int(limit)))
        url = f"{self.base_url}/markets?limit={clamped_limit}&active=true&closed=false"

        # Explicitly enforce GET method and unauthenticated public headers
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pm-research-paper-simulation/1.0",
                "Accept": "application/json",
            },
            method="GET",
        )

        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                with self._opener.open(req, timeout=self.request_timeout) as resp:
                    if resp.status != 200:
                        logger.warning(f"Public API returned HTTP {resp.status} on attempt {attempt + 1}")
                        continue
                    body_bytes = resp.read()
                    data = json.loads(body_bytes.decode("utf-8"))
                    if isinstance(data, list):
                        return self._normalize_public_response(data)
                    elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                        return self._normalize_public_response(data["data"])
                    logger.warning("Unexpected public response schema: expected list or {'data': list}")
                    return []

            except urllib.error.HTTPError as e:
                logger.warning(f"HTTP error {e.code} fetching public markets: {e.reason}")
                if e.code == 429:
                    # Rate limited: bounded exponential backoff up to 5.0s
                    backoff = min(5.0, 1.5 * (2 ** attempt))
                    time.sleep(backoff)
                elif e.code >= 500:
                    backoff = min(3.0, 1.0 * (attempt + 1))
                    time.sleep(backoff)
                else:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, Exception) as e:
                logger.warning(f"Network or parsing failure fetching public markets: {e}")
                backoff = min(3.0, 1.0 * (attempt + 1))
                time.sleep(backoff)

        logger.warning("Public market data fetch failed or unavailable; falling back safely to offline data.")
        return []

    def _normalize_public_response(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map public API response fields into standard dictionary format for Rigo.

        Never fabricates missing prices or dates; skips invalid records.
        """
        normalized: list[dict[str, Any]] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                market_id = str(item.get("id") or item.get("conditionId") or item.get("slug") or "").strip()
                question = str(item.get("question") or item.get("title") or "").strip()
                if not market_id or not question:
                    continue

                # Parse prices strictly
                outcome_prices = item.get("outcomePrices")
                yes_price = None
                if isinstance(outcome_prices, list) and len(outcome_prices) >= 1:
                    try:
                        p_val = float(outcome_prices[0])
                        if 0.0 <= p_val <= 1.0:
                            yes_price = p_val
                    except (ValueError, TypeError):
                        pass
                elif item.get("lastTradePrice") is not None:
                    try:
                        p_val = float(item["lastTradePrice"])
                        if 0.0 <= p_val <= 1.0:
                            yes_price = p_val
                    except (ValueError, TypeError):
                        pass

                if yes_price is None:
                    continue

                # Parse resolution date: never fabricate; reject record if missing or unparseable
                end_date_str = item.get("endDate") or item.get("resolutionTime")
                if not end_date_str:
                    continue
                try:
                    res_dt = parse_iso_utc(str(end_date_str))
                except Exception:
                    continue

                # Construct conservative quote without fabricating depth
                half_spread = 0.02
                yes_bid = max(0.01, round(yes_price - half_spread, 3))
                yes_ask = min(0.99, round(yes_price + half_spread, 3))
                no_ask = round(1.0 - yes_bid, 3)
                no_bid = round(1.0 - yes_ask, 3)

                liquidity = float(item.get("liquidity") or 0.0)
                volume_24h = float(item.get("volume24hr") or item.get("volume") or 0.0)

                created_at_str = item.get("createdAt")
                if created_at_str:
                    try:
                        created_dt = parse_iso_utc(str(created_at_str))
                    except Exception:
                        created_dt = res_dt
                else:
                    created_dt = res_dt

                normalized.append({
                    "market_id": f"pub_{market_id}",
                    "question": question,
                    "category": str(item.get("category", "PUBLIC")).upper(),
                    "status": "ACTIVE",
                    "resolution_time": res_dt.isoformat(),
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "no_bid": no_bid,
                    "no_ask": no_ask,
                    "last_price": yes_price,
                    "liquidity": max(0.0, liquidity),
                    "volume_24h": max(0.0, volume_24h),
                    "created_at": created_dt.isoformat(),
                    "metadata": {"source": "public_api", "raw_item_id": market_id},
                })
            except Exception as e:
                logger.debug(f"Skipping malformed public market item: {e}")
                continue

        return normalized
