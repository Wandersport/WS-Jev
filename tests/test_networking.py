"""Unit and integration tests for network access and public read-only market data adapter."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from pm_research.data.public_adapter import (
    PublicMarketDataAdapter,
    RestrictedRedirectHandler,
)


def test_allowed_hosts_and_https_enforcement():
    """Verify adapter enforces HTTPS scheme and strictly allows only approved hosts during initialization."""
    # Non-HTTPS should raise ValueError
    with pytest.raises(ValueError, match="Insecure public adapter URL scheme"):
        PublicMarketDataAdapter(base_url="http://gamma-api.polymarket.com")

    # Disallowed host should raise ValueError
    with pytest.raises(ValueError, match="not in the allowlist"):
        PublicMarketDataAdapter(base_url="https://malicious-site.com")

    # Allowed host should succeed
    adapter = PublicMarketDataAdapter(base_url="https://gamma-api.polymarket.com")
    assert adapter.base_url == "https://gamma-api.polymarket.com"


def test_restricted_redirect_handler_blocks_escapes():
    """Verify redirect handler refuses protocol downgrade or redirection to unallowed hosts."""
    handler = RestrictedRedirectHandler(allowed_hosts=frozenset({"gamma-api.polymarket.com", "clob.polymarket.com"}))
    dummy_req = MagicMock()

    # Redirect to http -> returns None (refused)
    res_http = handler.redirect_request(
        req=dummy_req, fp=None, code=302, msg="Found", headers={}, newurl="http://gamma-api.polymarket.com/api"
    )
    assert res_http is None

    # Redirect to disallowed host -> returns None (refused)
    res_evil = handler.redirect_request(
        req=dummy_req, fp=None, code=302, msg="Found", headers={}, newurl="https://evil.com/phish"
    )
    assert res_evil is None

    # Redirect to allowed host HTTPS -> allowed
    with patch("urllib.request.HTTPRedirectHandler.redirect_request", return_value="OK"):
        res_ok = handler.redirect_request(
            req=dummy_req, fp=None, code=302, msg="Found", headers={}, newurl="https://clob.polymarket.com/orderbook"
        )
        assert res_ok == "OK"


def test_fetch_public_markets_zero_credentials():
    """Verify that requests contain zero Authorization or trading API headers and use GET."""
    adapter = PublicMarketDataAdapter(min_request_interval_seconds=0.0)

    with patch.object(adapter._opener, "open") as mock_open:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"[]"
        mock_resp.__enter__.return_value = mock_resp
        mock_open.return_value = mock_resp

        data = adapter.fetch_public_markets(limit=5)
        assert data == []

        assert mock_open.called
        req = mock_open.call_args[0][0]
        # Verify method is strictly GET
        assert req.get_method() == "GET"
        # Verify no auth headers
        headers_lower = {k.lower(): v for k, v in req.headers.items()}
        assert "authorization" not in headers_lower
        assert "x-api-key" not in headers_lower
        assert "api-key" not in headers_lower
        assert "private-key" not in headers_lower


def test_fetch_public_markets_timeout_bounded_retry():
    """Verify that timeouts are caught and retried bounded times without infinite looping."""
    adapter = PublicMarketDataAdapter(max_retries=2, min_request_interval_seconds=0.0)

    with patch.object(adapter._opener, "open") as mock_open:
        mock_open.side_effect = urllib.error.URLError("timed out")

        with patch("time.sleep"):  # fast test
            res = adapter.fetch_public_markets(limit=5)
            assert res == []
            # 1 initial try + 2 retries = 3 calls
            assert mock_open.call_count == 3


def test_fetch_public_markets_429_rate_limit():
    """Verify that HTTP 429 rate limit responses are handled with bounded backoff."""
    adapter = PublicMarketDataAdapter(max_retries=1, min_request_interval_seconds=0.0)

    with patch.object(adapter._opener, "open") as mock_open:
        fp_mock = MagicMock()
        fp_mock.read.return_value = b"Rate limit exceeded"
        err_429 = urllib.error.HTTPError(
            url="https://gamma-api.polymarket.com/markets",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "0.02"},
            fp=fp_mock,
        )
        mock_open.side_effect = err_429

        with patch("time.sleep"):
            res = adapter.fetch_public_markets(limit=5)
            assert res == []
            assert mock_open.call_count == 2


def test_fetch_public_markets_malformed_json():
    """Verify that malformed/truncated JSON fails safely and returns empty list."""
    adapter = PublicMarketDataAdapter(min_request_interval_seconds=0.0)

    with patch.object(adapter._opener, "open") as mock_open:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"truncated": true, "unclosed'
        mock_resp.__enter__.return_value = mock_resp
        mock_open.return_value = mock_resp

        with patch("time.sleep"):
            res = adapter.fetch_public_markets(limit=5)
            assert res == []


def test_fetch_public_markets_500_server_error():
    """Verify that HTTP 500 server error responses retry and return empty list safely."""
    adapter = PublicMarketDataAdapter(max_retries=1, min_request_interval_seconds=0.0)

    with patch.object(adapter._opener, "open") as mock_open:
        fp_mock = MagicMock()
        fp_mock.read.return_value = b"Internal Server Error"
        err_500 = urllib.error.HTTPError(
            url="https://gamma-api.polymarket.com/markets",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=fp_mock,
        )
        mock_open.side_effect = err_500

        with patch("time.sleep"):
            res = adapter.fetch_public_markets(limit=5)
            assert res == []
            assert mock_open.call_count == 2


def test_fetch_public_markets_normalization():
    """Verify that valid public payload is correctly normalized to domain models with zero fabrication."""
    adapter = PublicMarketDataAdapter(min_request_interval_seconds=0.0)

    mock_gamma_payload = [
        {
            "id": "poly_123",
            "question": "Will candidate X win the election?",
            "category": "POLITICS",
            "active": True,
            "closed": False,
            "endDate": "2026-11-05T00:00:00Z",
            "outcomePrices": ["0.52", "0.48"],
            "liquidity": "25000.50",
            "volume24hr": "80000.00",
            "createdAt": "2026-01-01T00:00:00Z",
        },
        {
            # Malformed item: missing question -> should be safely skipped
            "id": "bad_item",
        },
    ]

    with patch.object(adapter._opener, "open") as mock_open:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(mock_gamma_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_open.return_value = mock_resp

        markets = adapter.fetch_public_markets(limit=5)
        assert len(markets) == 1
        m = markets[0]
        assert m["market_id"] == "pub_poly_123"
        assert m["question"] == "Will candidate X win the election?"
        assert m["yes_bid"] == 0.50
        assert m["yes_ask"] == 0.54
        assert m["last_price"] == 0.52
        assert m["liquidity"] == 25000.50
        assert m["status"] == "ACTIVE"
        assert m["resolution_time"] == "2026-11-05T00:00:00+00:00"
