"""Deterministic synthetic market fixtures for offline testing and demo."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pm_research.utils import ensure_utc, now_utc, to_iso_utc


def get_deterministic_synthetic_markets(base_time: datetime | None = None) -> list[dict[str, Any]]:
    """Generate a rich, deterministic set of synthetic prediction markets covering varied conditions."""
    t0 = ensure_utc(base_time or now_utc())

    markets: list[dict[str, Any]] = [
        # 1. Acceptable opportunity for YES: Mispriced favorite
        {
            "market_id": "mkt_alpha_tech",
            "question": "Will Company Alpha release their new open-weights AI model before Q4?",
            "category": "TECH",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=14)),
            "yes_bid": 0.38,
            "yes_ask": 0.40,
            "no_bid": 0.58,
            "no_ask": 0.62,
            "last_price": 0.39,
            "liquidity": 12500.0,
            "volume_24h": 45000.0,
            "order_book": {
                "bids": [{"price": 0.38, "quantity": 500.0}, {"price": 0.37, "quantity": 1000.0}],
                "asks": [{"price": 0.40, "quantity": 600.0}, {"price": 0.41, "quantity": 1200.0}],
            },
            "created_at": to_iso_utc(t0 - timedelta(days=2)),
            "metadata": {
                "source": "synthetic_seed",
                "target_outcome": "YES",
                "research_signal": 0.16,
                "momentum_24h": 0.25,
            },
        },
        # 2. Acceptable opportunity for NO: Overpriced hype
        {
            "market_id": "mkt_beta_macro",
            "question": "Will Global Central Bank cut benchmark interest rates by >50 bps in next meeting?",
            "category": "MACRO",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=21)),
            "yes_bid": 0.62,
            "yes_ask": 0.65,
            "no_bid": 0.33,
            "no_ask": 0.37,
            "last_price": 0.64,
            "liquidity": 18000.0,
            "volume_24h": 85000.0,
            "order_book": {
                "bids": [{"price": 0.62, "quantity": 800.0}],
                "asks": [{"price": 0.65, "quantity": 750.0}, {"price": 0.66, "quantity": 1500.0}],
            },
            "created_at": to_iso_utc(t0 - timedelta(days=5)),
            "metadata": {
                "source": "synthetic_seed",
                "target_outcome": "NO",
                "research_signal": -0.18,
                "momentum_24h": -0.30,
            },
        },
        # 3. Bram Rejection: Low liquidity ($120 < $500 threshold)
        {
            "market_id": "mkt_gamma_low_liq",
            "question": "Will Local Municipality pass Ordinance 42?",
            "category": "POLITICS",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=7)),
            "yes_bid": 0.40,
            "yes_ask": 0.45,
            "no_bid": 0.50,
            "no_ask": 0.58,
            "last_price": 0.42,
            "liquidity": 120.0,
            "volume_24h": 300.0,
            "created_at": to_iso_utc(t0 - timedelta(days=1)),
            "metadata": {
                "expected_reject": "REJECT_LOW_LIQUIDITY",
                "research_signal": 0.15,
            },
        },
        # 4. Bram Rejection: Wide spread (0.22 > 0.12 threshold)
        {
            "market_id": "mkt_delta_wide_spread",
            "question": "Will obscure treaty ratification occur this month?",
            "category": "INTERNATIONAL",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=10)),
            "yes_bid": 0.30,
            "yes_ask": 0.52,
            "no_bid": 0.40,
            "no_ask": 0.65,
            "last_price": 0.40,
            "liquidity": 6000.0,
            "volume_24h": 12000.0,
            "created_at": to_iso_utc(t0 - timedelta(days=3)),
            "metadata": {
                "expected_reject": "REJECT_HIGH_SPREAD",
                "research_signal": 0.15,
            },
        },
        # 5. Bram Rejection: Expiring in 20 minutes (< 1 hour threshold)
        {
            "market_id": "mkt_epsilon_expiring",
            "question": "Will emergency press conference occur in the next hour?",
            "category": "NEWS",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(minutes=20)),
            "yes_bid": 0.45,
            "yes_ask": 0.48,
            "no_bid": 0.48,
            "no_ask": 0.54,
            "last_price": 0.47,
            "liquidity": 5000.0,
            "volume_24h": 8000.0,
            "created_at": to_iso_utc(t0 - timedelta(hours=2)),
            "metadata": {
                "expected_reject": "REJECT_EXPIRING_SOON",
                "research_signal": 0.15,
            },
        },
        # 6. Bram Rejection: Fairly priced / robust edge too low (< 0.03 threshold)
        {
            "market_id": "mkt_zeta_fair",
            "question": "Will national coin toss result in heads?",
            "category": "GENERAL",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=3)),
            "yes_bid": 0.49,
            "yes_ask": 0.51,
            "no_bid": 0.49,
            "no_ask": 0.51,
            "last_price": 0.50,
            "liquidity": 25000.0,
            "volume_24h": 90000.0,
            "created_at": to_iso_utc(t0 - timedelta(days=1)),
            "metadata": {
                "expected_reject": "REJECT_ROBUST_EDGE_TOO_LOW",
                "research_signal": 0.0,
            },
        },
        # 7. Bram Rejection: Market status CLOSED
        {
            "market_id": "mkt_eta_closed",
            "question": "Did candidate X file campaign paperwork on time?",
            "category": "POLITICS",
            "status": "CLOSED",
            "resolution_time": to_iso_utc(t0 - timedelta(hours=1)),
            "yes_bid": 0.90,
            "yes_ask": 0.95,
            "last_price": 0.92,
            "liquidity": 15000.0,
            "volume_24h": 20000.0,
            "created_at": to_iso_utc(t0 - timedelta(days=10)),
            "metadata": {
                "expected_reject": "REJECT_MARKET_NOT_ACTIVE",
                "research_signal": 0.20,
            },
        },
        # 8. Acceptable opportunity with deep order-book for depth walking simulation
        {
            "market_id": "mkt_theta_depth",
            "question": "Will renewable energy generation exceed 35% in state grid this week?",
            "category": "ENERGY",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=5)),
            "yes_bid": 0.35,
            "yes_ask": 0.37,
            "no_bid": 0.60,
            "no_ask": 0.64,
            "last_price": 0.36,
            "liquidity": 35000.0,
            "volume_24h": 110000.0,
            "order_book": {
                "bids": [
                    {"price": 0.35, "quantity": 100.0},
                    {"price": 0.34, "quantity": 300.0},
                    {"price": 0.33, "quantity": 500.0},
                ],
                "asks": [
                    {"price": 0.37, "quantity": 40.0},   # Level 1
                    {"price": 0.38, "quantity": 80.0},   # Level 2
                    {"price": 0.39, "quantity": 200.0},  # Level 3
                ],
            },
            "created_at": to_iso_utc(t0 - timedelta(days=3)),
            "metadata": {
                "source": "synthetic_seed",
                "target_outcome": "YES",
                "research_signal": 0.14,
                "momentum_24h": 0.15,
            },
        },
        # 9. Rigo Rejection: Crossed book (malformed quote)
        {
            "market_id": "mkt_kappa_crossed",
            "question": "Malformed test: bid greater than ask",
            "category": "TEST",
            "status": "ACTIVE",
            "resolution_time": to_iso_utc(t0 + timedelta(days=1)),
            "yes_bid": 0.65,
            "yes_ask": 0.55,  # crossed!
            "liquidity": 2000.0,
            "volume_24h": 1000.0,
            "created_at": to_iso_utc(t0 - timedelta(hours=1)),
            "metadata": {"expected_reject": "RIGO_NORMALIZATION_FAILURE"},
        },
    ]

    return markets
