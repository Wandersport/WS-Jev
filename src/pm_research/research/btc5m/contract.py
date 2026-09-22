"""Polymarket BTC 5-minute market contract discovery, parsing, and resolution verification.

Strictly read-only and unauthenticated.
Validates:
- 300-second duration invariant
- Outcome token mapping by label ("Up", "Down")
- Authoritative resolution source verification (Chainlink TWAP stream)
- Authoritative Price-to-Beat (anchor) from metadata or exact start boundary
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pm_research.utils import ensure_utc, parse_iso_utc

logger = logging.getLogger(__name__)

GAMMA_API_BASE: str = "https://gamma-api.polymarket.com"
DATA_API_BASE: str = "https://data-api.polymarket.com"
CLOB_API_BASE: str = "https://clob.polymarket.com"

# Rejection reason codes
SKIP_UNSUPPORTED_SETTLEMENT_SOURCE = "SKIP_UNSUPPORTED_SETTLEMENT_SOURCE"
SKIP_INCOMPLETE_ROUND_METADATA = "SKIP_INCOMPLETE_ROUND_METADATA"
SKIP_NON_300S_DURATION = "SKIP_NON_300S_DURATION"
SKIP_NO_EXACT_ANCHOR = "SKIP_NO_EXACT_ANCHOR"
SKIP_MARKET_NOT_ACCEPTING_ORDERS = "SKIP_MARKET_NOT_ACCEPTING_ORDERS"


@dataclass(frozen=True)
class BTC5mRoundInfo:
    """Immutable verified metadata for a Polymarket BTC 5-minute round."""

    slug: str
    condition_id: str
    market_id: str
    question: str
    description: str
    resolution_source_url: str
    settlement_rule: str
    start_time_utc: datetime
    end_time_utc: datetime
    duration_seconds: int
    up_token_id: str
    down_token_id: str
    accepting_orders: bool
    price_to_beat: float | None
    price_to_beat_source: str | None
    fee_rate: float
    fee_exponent: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def round_slug(self) -> str:
        return self.slug

    @property
    def start_epoch(self) -> int:
        return int(self.start_time_utc.timestamp())

    @property
    def end_epoch(self) -> int:
        return int(self.end_time_utc.timestamp())

    @property
    def duration_sec(self) -> int:
        return self.duration_seconds

    @property
    def seconds_remaining(self) -> float:
        import time

        return max(0.0, self.end_time_utc.timestamp() - time.time())


@dataclass(frozen=True)
class BTC5mOfficialResolution:
    """Authoritative official resolution from Polymarket Data API."""

    condition_id: str
    status: str  # "resolved"
    winning_outcome: str  # "UP" or "DOWN"
    payout_up: float  # 1.0 or 0.0
    payout_down: float  # 0.0 or 1.0
    resolved_at_utc: datetime
    resolution_source: str
    raw_payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.status == "resolved"

    @property
    def resolved_outcome(self) -> str:
        return self.winning_outcome

    @property
    def resolution_price(self) -> float | None:
        val = self.raw_payload.get("resolution_price") or self.raw_payload.get("price")
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    @property
    def settled_at(self) -> str:
        return self.resolved_at_utc.isoformat()


class BTC5mContractManager:
    """Manages discovery and contract verification for Polymarket BTC 5m markets."""

    def __init__(
        self,
        gamma_base: str = GAMMA_API_BASE,
        data_api_base: str = DATA_API_BASE,
        request_timeout: float = 6.0,
    ) -> None:
        self.gamma_base = gamma_base.rstrip("/")
        self.data_api_base = data_api_base.rstrip("/")
        self.request_timeout = request_timeout

    @staticmethod
    def derive_round_slug(epoch_seconds: int | float) -> str:
        """Derive the deterministic Polymarket slug for a given epoch timestamp."""
        round_start = (int(epoch_seconds) // 300) * 300
        return f"btc-updown-5m-{round_start}"

    @staticmethod
    def derive_round_bounds(epoch_seconds: int | float) -> tuple[datetime, datetime]:
        """Derive start and end UTC datetimes for a 5-minute round."""
        round_start = (int(epoch_seconds) // 300) * 300
        start_dt = datetime.fromtimestamp(round_start, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(round_start + 300, tz=timezone.utc)
        return start_dt, end_dt

    def fetch_event_by_slug(self, slug: str) -> dict[str, Any] | None:
        """Fetch raw public market event data from Gamma API by slug."""
        url = f"{self.gamma_base}/events/slug/{urllib.parse.quote(slug)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pm-research-btc5m/1.0",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
                logger.warning(f"Gamma API returned status {resp.status} for slug {slug}")
                return None
        except Exception as e:
            logger.warning(f"Failed to fetch event for slug {slug}: {e}")
            return None

    def parse_and_validate_round(
        self,
        event_data: dict[str, Any],
        expected_slug: str,
    ) -> tuple[BTC5mRoundInfo | None, str | None]:
        """Parse raw Gamma event data and strictly validate all BTC 5m contract invariants."""
        if not isinstance(event_data, dict):
            return None, SKIP_INCOMPLETE_ROUND_METADATA

        markets = event_data.get("markets")
        if not isinstance(markets, list) or len(markets) == 0:
            return None, SKIP_INCOMPLETE_ROUND_METADATA

        market = markets[0]
        if not isinstance(market, dict):
            return None, SKIP_INCOMPLETE_ROUND_METADATA

        # 1. Validate condition ID
        condition_id = str(market.get("conditionId", "")).strip()
        if not condition_id.startswith("0x") or len(condition_id) != 66:
            return None, SKIP_INCOMPLETE_ROUND_METADATA

        market_id = str(market.get("id", "")).strip()
        question = str(market.get("question") or event_data.get("title") or "").strip()
        description = str(market.get("description") or event_data.get("description") or "").strip()

        # 2. Validate Settlement Source
        res_source = str(
            market.get("resolutionSource") or event_data.get("resolutionSource") or ""
        ).strip().lower()
        has_chainlink = "chainlink" in res_source or "chain.link" in res_source
        has_twap = "twap" in res_source
        if not (has_chainlink and has_twap):
            # Re-check description if resolutionSource URL was simplified
            desc_lower = description.lower()
            desc_chainlink = "chainlink" in desc_lower or "chain.link" in desc_lower
            desc_twap = "twap" in desc_lower
            if not (desc_chainlink and desc_twap):
                logger.warning(f"Unsupported settlement source '{res_source}' for {expected_slug}")
                return None, SKIP_UNSUPPORTED_SETTLEMENT_SOURCE

        # 3. Validate outcomes strictly by label ("Up" and "Down")
        outcomes_raw = market.get("outcomes")
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
            return None, SKIP_INCOMPLETE_ROUND_METADATA

        up_idx: int | None = None
        down_idx: int | None = None
        for i, o in enumerate(outcomes):
            o_clean = str(o).strip().lower()
            if o_clean == "up":
                up_idx = i
            elif o_clean == "down":
                down_idx = i

        if up_idx is None or down_idx is None:
            logger.warning(f"Market outcomes {outcomes} do not contain 'Up' and 'Down' labels")
            return None, SKIP_INCOMPLETE_ROUND_METADATA

        # Parse token IDs mapped by outcome index
        token_ids_raw = market.get("clobTokenIds")
        if isinstance(token_ids_raw, str):
            try:
                token_ids = json.loads(token_ids_raw)
            except Exception:
                token_ids = []
        elif isinstance(token_ids_raw, list):
            token_ids = token_ids_raw
        else:
            token_ids = []

        if len(token_ids) != 2:
            return None, SKIP_INCOMPLETE_ROUND_METADATA

        up_token_id = str(token_ids[up_idx])
        down_token_id = str(token_ids[down_idx])

        # 4. Validate exact 300-second duration
        # Derive epoch start from slug e.g. btc-updown-5m-1790091900
        try:
            slug_epoch = int(expected_slug.split("-")[-1])
            start_dt = datetime.fromtimestamp(slug_epoch, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(slug_epoch + 300, tz=timezone.utc)
            duration = 300
        except Exception:
            # Fall back to parsing startDate / endDate
            start_dt_raw = parse_iso_utc(market.get("startDate") or market.get("acceptingOrdersTimestamp"))
            end_dt_raw = parse_iso_utc(market.get("endDate") or market.get("resolutionTime"))
            if start_dt_raw is None or end_dt_raw is None:
                return None, SKIP_INCOMPLETE_ROUND_METADATA
            start_dt = ensure_utc(start_dt_raw)
            end_dt = ensure_utc(end_dt_raw)
            duration = int((end_dt - start_dt).total_seconds())

        if duration != 300:
            logger.warning(f"Market {expected_slug} duration is {duration}s != 300s")
            return None, SKIP_NON_300S_DURATION

        # 5. Extract Price-to-Beat (anchor) from eventMetadata
        event_metadata = event_data.get("eventMetadata") or market.get("eventMetadata") or {}
        price_to_beat: float | None = None
        price_to_beat_source: str | None = None

        if isinstance(event_metadata, dict) and event_metadata.get("priceToBeat") is not None:
            try:
                ptb_val = float(event_metadata["priceToBeat"])
                if ptb_val > 0.0:
                    price_to_beat = ptb_val
                    price_to_beat_source = "gamma-metadata"
            except (ValueError, TypeError):
                pass

        # Fee extraction (defaults to 0 if feesDisabled or feeSchedule missing)
        fee_rate = 0.0
        fee_exponent = 1.0
        fee_sched = market.get("feeSchedule")
        if isinstance(fee_sched, dict):
            try:
                fee_rate = float(fee_sched.get("rate", 0.0))
                fee_exponent = float(fee_sched.get("exponent", 1.0))
            except (ValueError, TypeError):
                pass

        accepting_orders = bool(
            market.get("active", True)
            and not market.get("closed", False)
            and market.get("acceptingOrders", True)
        )

        round_info = BTC5mRoundInfo(
            slug=expected_slug,
            condition_id=condition_id,
            market_id=market_id,
            question=question,
            description=description,
            resolution_source_url=res_source,
            settlement_rule="UP if final Chainlink reference price >= priceToBeat, otherwise DOWN",
            start_time_utc=start_dt,
            end_time_utc=end_dt,
            duration_seconds=duration,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            accepting_orders=accepting_orders,
            price_to_beat=price_to_beat,
            price_to_beat_source=price_to_beat_source,
            fee_rate=fee_rate,
            fee_exponent=fee_exponent,
            metadata={
                "event_id": str(event_data.get("id", "")),
                "slug_epoch": slug_epoch if "slug_epoch" in locals() else None,
                "event_metadata": event_metadata,
            },
        )
        return round_info, None

    def discover_round_by_slug(self, slug: str) -> tuple[BTC5mRoundInfo | None, str | None]:
        """Discover and validate a specific round by slug."""
        event_data = self.fetch_event_by_slug(slug)
        if not event_data:
            return None, SKIP_INCOMPLETE_ROUND_METADATA
        return self.parse_and_validate_round(event_data, slug)

    def discover_current_round(self, now_ts: float | None = None) -> tuple[BTC5mRoundInfo | None, str | None]:
        """Discover and validate the currently active BTC 5m round."""
        import time

        ts = now_ts if now_ts is not None else time.time()
        slug = self.derive_round_slug(ts)
        return self.discover_round_by_slug(slug)

    def discover_active_round(self, now_ts: float | None = None) -> BTC5mRoundInfo:
        """Discover and strictly validate the currently active round, raising if unavailable."""
        round_info, skip_reason = self.discover_current_round(now_ts)
        if round_info is None:
            raise RuntimeError(f"Could not discover active round: {skip_reason}")
        return round_info

    def fetch_official_resolution(
        self, target: BTC5mRoundInfo | str
    ) -> BTC5mOfficialResolution | None:
        """Query Polymarket Data API for authoritative resolution state."""
        condition_id = target.condition_id if isinstance(target, BTC5mRoundInfo) else target
        url = f"{self.data_api_base}/v2/resolutions?condition={urllib.parse.quote(condition_id)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "pm-research-btc5m/1.0",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
                rows = data.get("data") if isinstance(data, dict) else []
                if not isinstance(rows, list) or len(rows) == 0:
                    return None

                # Find matching row with status == 'resolved'
                row = None
                for r in rows:
                    cid = r.get("condition_id") or r.get("condition")
                    if str(cid).lower() == condition_id.lower() and r.get("status") == "resolved":
                        row = r
                        break

                if not row:
                    return None

                payouts_raw = row.get("payouts")
                if not isinstance(payouts_raw, list) or len(payouts_raw) != 2:
                    return None

                p_up_raw = float(payouts_raw[0])
                p_down_raw = float(payouts_raw[1])
                total = p_up_raw + p_down_raw
                if total <= 0:
                    return None

                p_up = round(p_up_raw / total, 4)
                p_down = round(p_down_raw / total, 4)
                winner = "UP" if p_up > p_down else "DOWN"

                res_at_raw = row.get("resolved_at") or row.get("last_update_timestamp")
                res_at = parse_iso_utc(str(res_at_raw)) if res_at_raw else ensure_utc(datetime.now(timezone.utc))

                return BTC5mOfficialResolution(
                    condition_id=condition_id,
                    status="resolved",
                    winning_outcome=winner,
                    payout_up=p_up,
                    payout_down=p_down,
                    resolved_at_utc=ensure_utc(res_at),
                    resolution_source=str(row.get("resolution_source", "polymarket_data_api")),
                    raw_payload=row,
                )
        except Exception as e:
            logger.warning(f"Error fetching resolution for {condition_id}: {e}")
            return None
