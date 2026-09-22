"""Deterministic rich multi-period synthetic replay fixture for backtesting and validation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pm_research.domain.models import OrderBook, OrderBookLevel, Side
from pm_research.replay.dataset import DatasetManifest, ReplayDataset
from pm_research.replay.models import HistoricalResolution, HistoricalSnapshot
from pm_research.utils import ensure_utc


def create_deterministic_synthetic_replay_dataset(
    base_time: datetime | None = None,
    target_dir: Path | None = None,
) -> ReplayDataset:
    """Construct a multi-period chronological dataset with varying spreads, depths, liquidity, latency events, and resolutions."""
    t0 = ensure_utc(base_time or datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc))

    # Time steps: 8 cycles spaced by 12 hours
    t1 = t0 + timedelta(hours=12)
    t2 = t0 + timedelta(hours=24)
    t3 = t0 + timedelta(hours=36)
    t4 = t0 + timedelta(hours=48)
    t5 = t0 + timedelta(hours=60)

    snapshots: list[HistoricalSnapshot] = []

    # -------------------------------------------------------------
    # Market 1: mkt_alpha_tech (Underpriced favorite, trending up, resolves YES at t5)
    # -------------------------------------------------------------
    res_time_alpha = t0 + timedelta(days=14)
    # Cycle 0 (t0): Underpriced at 0.38 / 0.40 -> Model signals YES conviction (q_hat ~ 0.58), positive edge
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_alpha_tech",
            timestamp=t0,
            status="ACTIVE",
            question="Will Model Alpha exceed MMLU 90 before Q2?",
            category="TECH",
            resolution_time=res_time_alpha,
            yes_bid=0.38,
            yes_ask=0.40,
            no_bid=0.58,
            no_ask=0.62,
            last_price=0.39,
            midpoint=0.39,
            spread=0.02,
            liquidity=15000.0,
            volume_24h=50000.0,
            order_book=OrderBook(
                bids=[OrderBookLevel(0.38, 500.0), OrderBookLevel(0.37, 1000.0)],
                asks=[OrderBookLevel(0.40, 200.0), OrderBookLevel(0.42, 500.0)],
            ),
            metadata={"research_signal": 0.18, "momentum_24h": 0.20},
        )
    )
    # Cycle 1 (t1): Price advances to 0.44 / 0.46 (if latency = 12h, order fills here with slippage)
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_alpha_tech",
            timestamp=t1,
            status="ACTIVE",
            question="Will Model Alpha exceed MMLU 90 before Q2?",
            category="TECH",
            resolution_time=res_time_alpha,
            yes_bid=0.44,
            yes_ask=0.46,
            no_bid=0.52,
            no_ask=0.56,
            last_price=0.45,
            midpoint=0.45,
            spread=0.02,
            liquidity=18000.0,
            volume_24h=65000.0,
            order_book=OrderBook(
                bids=[OrderBookLevel(0.44, 600.0)],
                asks=[OrderBookLevel(0.46, 300.0), OrderBookLevel(0.48, 800.0)],
            ),
            metadata={"research_signal": 0.15, "momentum_24h": 0.30},
        )
    )
    # Cycle 2 (t2): Price advances further to 0.55 / 0.58
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_alpha_tech",
            timestamp=t2,
            status="ACTIVE",
            question="Will Model Alpha exceed MMLU 90 before Q2?",
            category="TECH",
            resolution_time=res_time_alpha,
            yes_bid=0.55,
            yes_ask=0.58,
            no_bid=0.40,
            no_ask=0.45,
            last_price=0.56,
            midpoint=0.565,
            spread=0.03,
            liquidity=22000.0,
            volume_24h=80000.0,
            metadata={"research_signal": 0.12},
        )
    )
    # Cycle 3 (t3): Price advances to 0.70 / 0.73
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_alpha_tech",
            timestamp=t3,
            status="ACTIVE",
            question="Will Model Alpha exceed MMLU 90 before Q2?",
            category="TECH",
            resolution_time=res_time_alpha,
            yes_bid=0.70,
            yes_ask=0.73,
            no_bid=0.25,
            no_ask=0.30,
            last_price=0.71,
            midpoint=0.715,
            spread=0.03,
            liquidity=25000.0,
            volume_24h=95000.0,
            metadata={"research_signal": 0.08},
        )
    )
    # Cycle 4 (t4): Nearing resolution price 0.88 / 0.92
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_alpha_tech",
            timestamp=t4,
            status="ACTIVE",
            question="Will Model Alpha exceed MMLU 90 before Q2?",
            category="TECH",
            resolution_time=res_time_alpha,
            yes_bid=0.88,
            yes_ask=0.92,
            no_bid=0.06,
            no_ask=0.12,
            last_price=0.90,
            midpoint=0.90,
            spread=0.04,
            liquidity=30000.0,
            volume_24h=120000.0,
            metadata={"research_signal": 0.04},
        )
    )

    # -------------------------------------------------------------
    # Market 2: mkt_beta_macro (Overpriced hype, resolves NO at t5)
    # -------------------------------------------------------------
    res_time_beta = t0 + timedelta(days=21)
    # Cycle 0 (t0): High YES price 0.65 / 0.68, low NO price 0.32 / 0.35 -> Model signals NO conviction (buying NO)
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_beta_macro",
            timestamp=t0,
            status="ACTIVE",
            question="Will Central Bank hike rates by >50 bps?",
            category="MACRO",
            resolution_time=res_time_beta,
            yes_bid=0.65,
            yes_ask=0.68,
            no_bid=0.32,
            no_ask=0.35,
            last_price=0.66,
            midpoint=0.665,
            spread=0.03,
            liquidity=20000.0,
            volume_24h=80000.0,
            order_book=OrderBook(
                bids=[OrderBookLevel(0.65, 800.0)],
                asks=[OrderBookLevel(0.68, 400.0)],
            ),
            metadata={"research_signal": -0.15},
        )
    )
    # Cycle 1 (t1): YES price begins dropping to 0.58 / 0.62 (NO price rising to 0.38 / 0.42)
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_beta_macro",
            timestamp=t1,
            status="ACTIVE",
            question="Will Central Bank hike rates by >50 bps?",
            category="MACRO",
            resolution_time=res_time_beta,
            yes_bid=0.58,
            yes_ask=0.62,
            no_bid=0.38,
            no_ask=0.42,
            last_price=0.60,
            midpoint=0.60,
            spread=0.04,
            liquidity=22000.0,
            volume_24h=90000.0,
            metadata={"research_signal": -0.18},
        )
    )
    # Cycle 2 (t2): YES price drops to 0.45 / 0.49
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_beta_macro",
            timestamp=t2,
            status="ACTIVE",
            question="Will Central Bank hike rates by >50 bps?",
            category="MACRO",
            resolution_time=res_time_beta,
            yes_bid=0.45,
            yes_ask=0.49,
            no_bid=0.51,
            no_ask=0.55,
            last_price=0.47,
            midpoint=0.47,
            spread=0.04,
            liquidity=24000.0,
            volume_24h=110000.0,
            metadata={"research_signal": -0.22},
        )
    )
    # Cycle 3 (t3): YES price drops to 0.25 / 0.29
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_beta_macro",
            timestamp=t3,
            status="ACTIVE",
            question="Will Central Bank hike rates by >50 bps?",
            category="MACRO",
            resolution_time=res_time_beta,
            yes_bid=0.25,
            yes_ask=0.29,
            no_bid=0.71,
            no_ask=0.75,
            last_price=0.27,
            midpoint=0.27,
            spread=0.04,
            liquidity=26000.0,
            volume_24h=130000.0,
            metadata={"research_signal": -0.25},
        )
    )
    # Cycle 4 (t4): Collapsing to 0.08 / 0.12
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_beta_macro",
            timestamp=t4,
            status="ACTIVE",
            question="Will Central Bank hike rates by >50 bps?",
            category="MACRO",
            resolution_time=res_time_beta,
            yes_bid=0.08,
            yes_ask=0.12,
            no_bid=0.88,
            no_ask=0.92,
            last_price=0.10,
            midpoint=0.10,
            spread=0.04,
            liquidity=30000.0,
            volume_24h=150000.0,
            metadata={"research_signal": -0.28},
        )
    )

    # -------------------------------------------------------------
    # Market 3: mkt_gamma_spread_rejection (Triggers Bram REJECT_HIGH_SPREAD at t1)
    # -------------------------------------------------------------
    res_time_gamma = t0 + timedelta(days=30)
    # t0: Fair market
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_gamma_volatile",
            timestamp=t0,
            status="ACTIVE",
            question="Will Space Mission Gamma land safely?",
            category="SCIENCE",
            resolution_time=res_time_gamma,
            yes_bid=0.48,
            yes_ask=0.52,
            no_bid=0.48,
            no_ask=0.52,
            last_price=0.50,
            midpoint=0.50,
            spread=0.04,
            liquidity=10000.0,
            volume_24h=30000.0,
            metadata={"research_signal": 0.02},
        )
    )
    # t1: Massive spread widening (bid 0.30, ask 0.70 -> spread 0.40 > 0.12 limit) -> Bram MUST reject
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_gamma_volatile",
            timestamp=t1,
            status="ACTIVE",
            question="Will Space Mission Gamma land safely?",
            category="SCIENCE",
            resolution_time=res_time_gamma,
            yes_bid=0.30,
            yes_ask=0.70,
            no_bid=0.30,
            no_ask=0.70,
            last_price=0.50,
            midpoint=0.50,
            spread=0.40,  # Far above config.max_spread (0.12)
            liquidity=4000.0,
            volume_24h=32000.0,
            metadata={"research_signal": 0.15},
        )
    )

    # -------------------------------------------------------------
    # Market 4: mkt_delta_low_liquidity (Triggers Bram REJECT_LOW_LIQUIDITY)
    # -------------------------------------------------------------
    snapshots.append(
        HistoricalSnapshot(
            market_id="mkt_delta_illiquid",
            timestamp=t0,
            status="ACTIVE",
            question="Will obscure bill XYZ pass committee?",
            category="POLITICS",
            resolution_time=t0 + timedelta(days=20),
            yes_bid=0.20,
            yes_ask=0.24,
            no_bid=0.76,
            no_ask=0.80,
            last_price=0.22,
            midpoint=0.22,
            spread=0.04,
            liquidity=120.0,  # Below min_liquidity (500.0)
            volume_24h=300.0,
            metadata={"research_signal": 0.20},
        )
    )

    # -------------------------------------------------------------
    # Resolutions: Available strictly at t5
    # -------------------------------------------------------------
    resolutions = [
        HistoricalResolution(
            market_id="mkt_alpha_tech",
            resolved_outcome=Side.YES,
            resolved_at=t5,
        ),
        HistoricalResolution(
            market_id="mkt_beta_macro",
            resolved_outcome=Side.NO,
            resolved_at=t5,
        ),
    ]

    manifest = DatasetManifest(
        dataset_id="synthetic_benchmark_v1",
        name="Deterministic Synthetic Multi-Period Benchmark",
        source="synthetic_generator",
        created_at=t0,
        start_time=t0,
        end_time=t4,
        market_count=4,
        snapshot_count=len(snapshots),
        resolution_count=len(resolutions),
        schema_version="1.0.0",
        is_synthetic=True,
        notes="Deterministic benchmark with price trends, spread spikes, liquidity limits, and delayed resolutions.",
    )

    dataset = ReplayDataset(manifest=manifest, snapshots=snapshots, resolutions=resolutions)

    if target_dir:
        dataset.save(target_dir)

    return dataset
