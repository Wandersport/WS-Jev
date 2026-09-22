"""Public read-only historical data collector for resolved prediction markets."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pm_research.data.public_adapter import DEFAULT_ALLOWED_HOSTS, RestrictedRedirectHandler
from pm_research.domain.models import Side
from pm_research.replay.dataset import DatasetManifest, ReplayDataset
from pm_research.replay.models import HistoricalResolution, HistoricalSnapshot
from pm_research.utils import ensure_utc, parse_iso_utc, to_iso_utc

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Detailed summary of discovered, included, and excluded historical markets."""

    markets_discovered: int = 0
    markets_included: int = 0
    markets_excluded: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)
    price_points_downloaded: int = 0
    earliest_resolution: str | None = None
    latest_resolution: str | None = None
    categories: dict[str, int] = field(default_factory=dict)

    def record_exclusion(self, reason: str) -> None:
        self.markets_excluded += 1
        self.exclusion_reasons[reason] = self.exclusion_reasons.get(reason, 0) + 1


class HistoricalCollector:
    """Collects resolved prediction market data from public unauthenticated endpoints."""

    def __init__(
        self,
        gamma_base_url: str = "https://gamma-api.polymarket.com",
        clob_base_url: str = "https://clob.polymarket.com",
        allowed_hosts: frozenset[str] | None = None,
        cache_dir: str | Path = "data/historical_raw",
        request_timeout: float = 10.0,
        rate_limit_delay: float = 0.35,
        max_retries: int = 2,
    ) -> None:
        self.allowed_hosts = allowed_hosts or DEFAULT_ALLOWED_HOSTS
        self.gamma_base = gamma_base_url.rstrip("/")
        self.clob_base = clob_base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_timeout = max(1.0, float(request_timeout))
        self.rate_limit_delay = max(0.1, float(rate_limit_delay))
        self.max_retries = max(0, min(3, int(max_retries)))
        self._last_request_time: float = 0.0

        self._opener = urllib.request.build_opener(RestrictedRedirectHandler(self.allowed_hosts))

    def _wait_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    def _http_get_json(self, url: str) -> Any | None:
        """Execute unauthenticated GET request with timeouts and rate limits."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() != "https":
            raise ValueError(f"Insecure scheme '{parsed.scheme}'. Must be 'https'.")
        if parsed.netloc.lower() not in self.allowed_hosts:
            raise ValueError(f"Host '{parsed.netloc}' not in allowlist.")

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pm-research-historical-study/1.0",
                "Accept": "application/json",
            },
            method="GET",
        )

        for attempt in range(self.max_retries + 1):
            self._wait_rate_limit()
            try:
                with self._opener.open(req, timeout=self.request_timeout) as resp:
                    if resp.status == 200:
                        return json.loads(resp.read().decode("utf-8"))
                    logger.warning(f"HTTP {resp.status} fetching {url}")
            except urllib.error.HTTPError as e:
                logger.warning(f"HTTPError {e.code} for {url}: {e.reason}")
                if e.code == 429:
                    time.sleep(min(5.0, 1.5 * (2**attempt)))
                elif e.code >= 500:
                    time.sleep(min(3.0, 1.0 * (attempt + 1)))
                else:
                    return None
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                time.sleep(min(2.0, 0.5 * (attempt + 1)))

        return None

    def discover_resolved_markets(
        self,
        max_candidates: int = 100,
        start_date: str = "2024-01-01T00:00:00Z",
        end_date: str = "2024-12-31T23:59:59Z",
        min_volume: float = 20000.0,
    ) -> list[dict[str, Any]]:
        """Paginate Gamma API to discover resolved markets matching volume and date filters."""
        discovered: list[dict[str, Any]] = []
        batch_size = 50
        offset = 0

        while len(discovered) < max_candidates:
            query = urllib.parse.urlencode({
                "closed": "true",
                "limit": str(min(batch_size, max_candidates - len(discovered))),
                "offset": str(offset),
                "volume_num_min": str(int(min_volume)),
                "end_date_min": start_date,
                "end_date_max": end_date,
            })
            url = f"{self.gamma_base}/markets?{query}"
            data = self._http_get_json(url)

            if not data or not isinstance(data, list) or len(data) == 0:
                break

            discovered.extend(data)
            offset += len(data)

            if len(data) < batch_size:
                break

        return discovered

    def fetch_token_prices_history(
        self,
        token_id: str,
        end_ts: int,
        days_back: int = 30,
        fidelity: int = 60,
    ) -> list[dict[str, float | int]]:
        """Fetch historical CLOB price series using 14-day chunks with local caching."""
        cache_file = self.cache_dir / f"clob_{token_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if isinstance(cached, list) and len(cached) > 0:
                        return cached
            except Exception:
                pass

        chunk_size = 14 * 86400
        start_ts = end_ts - (days_back * 86400)
        t_cursor = end_ts
        all_points: list[dict[str, float | int]] = []
        seen_t: set[int] = set()

        while t_cursor > start_ts:
            t_chunk_start = max(start_ts, t_cursor - chunk_size)
            query = urllib.parse.urlencode({
                "market": token_id,
                "startTs": str(t_chunk_start),
                "endTs": str(t_cursor),
                "fidelity": str(fidelity),
            })
            url = f"{self.clob_base}/prices-history?{query}"
            data = self._http_get_json(url)

            if data and isinstance(data, dict) and "history" in data:
                for pt in data["history"]:
                    t_val = int(pt.get("t", 0))
                    p_val = float(pt.get("p", 0.0))
                    if t_val > 0 and 0.0 <= p_val <= 1.0 and t_val not in seen_t:
                        seen_t.add(t_val)
                        all_points.append({"t": t_val, "p": p_val})

            t_cursor = t_chunk_start

        all_points.sort(key=lambda x: x["t"])

        # Cache immutable raw response if non-empty
        if all_points:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(all_points, f)
            except Exception as e:
                logger.warning(f"Failed to cache CLOB data for {token_id}: {e}")

        return all_points

    def filter_and_collect_market(
        self,
        raw_mkt: dict[str, Any],
        report: QualityReport,
        days_back: int = 30,
        min_history_points: int = 5,
    ) -> tuple[list[HistoricalSnapshot], HistoricalResolution] | None:
        """Validate market eligibility, retrieve CLOB history, and normalize."""
        # 1. Binary check
        outcomes_raw = raw_mkt.get("outcomes")
        if isinstance(outcomes_raw, str):
            try:
                outcomes = json.loads(outcomes_raw)
            except Exception:
                outcomes = []
        elif isinstance(outcomes_raw, list):
            outcomes = outcomes_raw
        else:
            outcomes = []

        if len(outcomes) != 2:
            report.record_exclusion("EXCLUDE_NON_BINARY")
            return None

        # 2. Resolution check
        prices_raw = raw_mkt.get("outcomePrices")
        if isinstance(prices_raw, str):
            try:
                prices = json.loads(prices_raw)
            except Exception:
                prices = []
        elif isinstance(prices_raw, list):
            prices = prices_raw
        else:
            prices = []

        if len(prices) != 2:
            report.record_exclusion("EXCLUDE_NO_RESOLUTION")
            return None

        try:
            p0 = float(prices[0])
            p1 = float(prices[1])
        except (ValueError, TypeError):
            report.record_exclusion("EXCLUDE_NO_RESOLUTION")
            return None

        if p0 >= 0.99 and p1 <= 0.01:
            winning_side = Side.YES
        elif p1 >= 0.99 and p0 <= 0.01:
            winning_side = Side.NO
        else:
            report.record_exclusion("EXCLUDE_NO_RESOLUTION")
            return None

        # 3. Token ID check
        tokens_raw = raw_mkt.get("clobTokenIds")
        if isinstance(tokens_raw, str):
            try:
                tokens = json.loads(tokens_raw)
            except Exception:
                tokens = []
        elif isinstance(tokens_raw, list):
            tokens = tokens_raw
        else:
            tokens = []

        if not tokens or len(tokens) < 1 or not str(tokens[0]).strip():
            report.record_exclusion("EXCLUDE_NO_TOKEN_ID")
            return None

        yes_token_id = str(tokens[0]).strip()

        # 4. Resolution timestamp check
        res_time_str = raw_mkt.get("closedTime") or raw_mkt.get("endDate")
        if not res_time_str:
            report.record_exclusion("EXCLUDE_MALFORMED_HISTORY")
            return None
        try:
            res_dt = parse_iso_utc(str(res_time_str))
        except Exception:
            report.record_exclusion("EXCLUDE_MALFORMED_HISTORY")
            return None

        # 5. Fetch CLOB price points
        end_ts = int(res_dt.timestamp())
        history_pts = self.fetch_token_prices_history(
            token_id=yes_token_id,
            end_ts=end_ts,
            days_back=days_back,
            fidelity=60,
        )

        if len(history_pts) < min_history_points:
            report.record_exclusion("EXCLUDE_INSUFFICIENT_HISTORY")
            return None

        market_id = f"poly_{raw_mkt.get('id') or raw_mkt.get('slug')}"
        question = str(raw_mkt.get("question", "")).strip()
        category = str(raw_mkt.get("category", "PUBLIC")).strip().upper()
        liquidity = float(raw_mkt.get("liquidity", 0.0) or 0.0)
        volume_24h = float(raw_mkt.get("volume24hr", 0.0) or 0.0)

        # 6. Normalize to HistoricalSnapshots
        snapshots: list[HistoricalSnapshot] = []
        # Pre-compute historical lookup for momentum
        # Map: timestamp (int) -> price
        t_to_p = {int(pt["t"]): float(pt["p"]) for pt in history_pts}
        sorted_ts = sorted(t_to_p.keys())

        for pt in history_pts:
            t_sec = int(pt["t"])
            p_val = float(pt["p"])
            pt_dt = datetime.fromtimestamp(t_sec, tz=timezone.utc)

            # Strictly historical 24h momentum:
            # Look for price at or before t_sec - 86400 (NEVER after t_sec!)
            t_24h_prior = t_sec - 86400
            prior_p = None
            for past_t in reversed(sorted_ts):
                if past_t <= t_24h_prior:
                    prior_p = t_to_p[past_t]
                    break

            mom_24h = round(p_val - prior_p, 4) if prior_p is not None else 0.0

            snap = HistoricalSnapshot(
                market_id=market_id,
                timestamp=pt_dt,
                status="ACTIVE",
                question=question,
                category=category,
                resolution_time=res_dt,
                yes_bid=None,  # Price-only history: never fabricate bid
                yes_ask=None,  # Price-only history: never fabricate ask
                no_bid=None,
                no_ask=None,
                last_price=round(p_val, 4),
                midpoint=round(p_val, 4),  # Midpoint aligns with observed price in forecast mode
                spread=None,  # Spread is unavailable in CLOB prices-history
                liquidity=liquidity,
                volume_24h=volume_24h,
                order_book=None,
                metadata={
                    "source": "polymarket_public_archive",
                    "price_source": "clob_prices_history",
                    "quote_source": "unavailable",
                    "orderbook_source": "unavailable",
                    "token_id": yes_token_id,
                    "momentum_24h": mom_24h,
                },
            )
            snapshots.append(snap)

        resolution = HistoricalResolution(
            market_id=market_id,
            resolved_outcome=winning_side,
            resolved_at=res_dt,
        )

        report.markets_included += 1
        report.price_points_downloaded += len(snapshots)
        report.categories[category] = report.categories.get(category, 0) + 1

        if report.earliest_resolution is None or res_time_str < report.earliest_resolution:
            report.earliest_resolution = to_iso_utc(res_dt)
        if report.latest_resolution is None or res_time_str > report.latest_resolution:
            report.latest_resolution = to_iso_utc(res_dt)

        return snapshots, resolution

    def build_dataset(
        self,
        dataset_id: str,
        name: str,
        target_dir: str | Path,
        max_markets: int = 50,
        start_date: str = "2024-01-01T00:00:00Z",
        end_date: str = "2024-12-31T23:59:59Z",
        min_volume: float = 20000.0,
        days_back: int = 30,
    ) -> tuple[ReplayDataset, QualityReport]:
        """Orchestrate discovery, download, and packaging of a genuine historical ReplayDataset."""
        report = QualityReport()
        logger.info(f"Discovering resolved markets between {start_date} and {end_date}...")
        raw_candidates = self.discover_resolved_markets(
            max_candidates=max_markets * 3,  # Buffer for non-binary/missing history exclusions
            start_date=start_date,
            end_date=end_date,
            min_volume=min_volume,
        )
        report.markets_discovered = len(raw_candidates)

        all_snapshots: list[HistoricalSnapshot] = []
        all_resolutions: list[HistoricalResolution] = []

        for raw_mkt in raw_candidates:
            if report.markets_included >= max_markets:
                break

            res_tuple = self.filter_and_collect_market(
                raw_mkt=raw_mkt,
                report=report,
                days_back=days_back,
            )
            if res_tuple is not None:
                snaps, resolution = res_tuple
                all_snapshots.extend(snaps)
                all_resolutions.append(resolution)

        if not all_snapshots:
            raise ValueError("No eligible markets could be collected.")

        all_snapshots.sort(key=lambda s: s.timestamp)
        all_resolutions.sort(key=lambda r: r.resolved_at)

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            name=name,
            source="polymarket_public_archive",
            created_at=ensure_utc(datetime.now(timezone.utc)),
            start_time=all_snapshots[0].timestamp,
            end_time=all_snapshots[-1].timestamp,
            market_count=len(all_resolutions),
            snapshot_count=len(all_snapshots),
            resolution_count=len(all_resolutions),
            schema_version="1.0.0",
            is_synthetic=False,
            notes="Real historical Polymarket resolved markets. Forecast evaluation mode (price-only).",
        )

        dataset = ReplayDataset(
            manifest=manifest,
            snapshots=all_snapshots,
            resolutions=all_resolutions,
        )
        dataset.save(target_dir)

        # Save quality report alongside dataset
        report_file = Path(target_dir) / "quality_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "markets_discovered": report.markets_discovered,
                    "markets_included": report.markets_included,
                    "markets_excluded": report.markets_excluded,
                    "exclusion_reasons": report.exclusion_reasons,
                    "price_points_downloaded": report.price_points_downloaded,
                    "date_range": [report.earliest_resolution, report.latest_resolution],
                    "categories": report.categories,
                },
                f,
                indent=2,
            )

        return dataset, report
