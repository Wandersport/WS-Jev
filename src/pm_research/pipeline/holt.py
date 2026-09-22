"""HOLT: Research feature extraction stage."""

from __future__ import annotations

import math
from datetime import datetime

from pm_research.config import SystemConfig
from pm_research.domain.models import MarketSnapshot, ResearchFeatures
from pm_research.utils import ensure_utc, generate_id, now_utc


class HoltResearcher:
    """Extracts transparent, fully auditable research features from market data and order books.

    Features computed:
    - momentum_24h: short-term price trend/velocity
    - order_book_imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
    - spread_pct: bid-ask spread normalized by midpoint
    - liquidity_log: log-transformed market depth
    - hours_to_resolution: time horizon until market settlement
    - volume_turnover_ratio: 24h volume relative to visible liquidity
    - signal_sentiment: exogenous research signal or categorical sentiment
    """

    def __init__(self, config: SystemConfig) -> None:
        self.config = config

    def extract_features(
        self,
        snapshot: MarketSnapshot,
        current_time: datetime | None = None,
    ) -> ResearchFeatures:
        """Extract domain features with complete provenance tracking."""
        now = ensure_utc(current_time or now_utc())
        market = snapshot.market
        quote = market.quote
        meta = market.metadata or {}

        features: dict[str, float] = {}
        provenance: dict[str, str] = {}

        # 1. Spread and normalized spread
        mid = quote.midpoint or quote.last_price or 0.50
        spread = quote.spread or 0.04
        features["midpoint"] = round(mid, 4)
        provenance["midpoint"] = "quote.midpoint or quote.last_price"

        spread_pct = spread / max(0.01, mid)
        features["spread_pct"] = round(spread_pct, 4)
        provenance["spread_pct"] = "spread / max(0.01, midpoint)"

        # 2. Liquidity log
        liq = max(1.0, quote.liquidity)
        features["liquidity_log"] = round(math.log10(liq), 3)
        provenance["liquidity_log"] = "log10(max(1.0, quote.liquidity))"

        # 3. Hours to resolution
        hours_to_res = max(0.0, (market.resolution_time - now).total_seconds() / 3600.0)
        features["hours_to_resolution"] = round(hours_to_res, 2)
        provenance["hours_to_resolution"] = "(market.resolution_time - now).total_seconds() / 3600"

        # 4. Volume turnover ratio
        vol = max(0.0, quote.volume_24h)
        turnover = vol / liq
        features["volume_turnover_ratio"] = round(turnover, 3)
        provenance["volume_turnover_ratio"] = "volume_24h / max(1.0, liquidity)"

        # 5. Order book imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
        if market.order_book and (market.order_book.total_bid_depth + market.order_book.total_ask_depth) > 0:
            bid_d = market.order_book.total_bid_depth
            ask_d = market.order_book.total_ask_depth
            ob_imbalance = (bid_d - ask_d) / (bid_d + ask_d)
            features["order_book_imbalance"] = round(ob_imbalance, 4)
            provenance["order_book_imbalance"] = "(bid_depth - ask_depth) / (bid_depth + ask_depth)"
        else:
            features["order_book_imbalance"] = 0.0
            provenance["order_book_imbalance"] = "default_zero (no order book)"

        # 6. Momentum (from metadata signal or price delta)
        mom = float(meta.get("momentum_24h", 0.0))
        features["momentum_24h"] = round(mom, 4)
        provenance["momentum_24h"] = "market.metadata.momentum_24h"

        # 7. Exogenous research / sentiment signal
        # Used for research priors (e.g. fundamental research, polling models)
        signal = float(meta.get("research_signal", 0.0))
        features["research_signal"] = round(signal, 4)
        provenance["research_signal"] = "market.metadata.research_signal"

        return ResearchFeatures(
            feature_id=generate_id("feat"),
            cycle_id=snapshot.cycle_id,
            market_id=market.market_id,
            timestamp=now,
            values=features,
            provenance=provenance,
        )
