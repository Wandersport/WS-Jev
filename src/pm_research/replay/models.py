"""Data models for historical replay, execution latency, and backtesting metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pm_research.calibration.metrics import CalibrationReport
from pm_research.config import SystemConfig
from pm_research.domain.models import (
    MarketSnapshot,
    OrderBook,
    OrderBookLevel,
    RiskDecision,
    Side,
    TradeProposal,
)
from pm_research.utils import ensure_utc, parse_iso_utc, to_iso_utc


@dataclass
class HistoricalSnapshot:
    """A point-in-time normalized market observation at historical timestamp t."""

    market_id: str
    timestamp: datetime
    status: str
    question: str
    category: str
    resolution_time: datetime
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    midpoint: float | None = None
    spread: float | None = None
    liquidity: float = 0.0
    volume_24h: float = 0.0
    order_book: OrderBook | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)
        self.resolution_time = ensure_utc(self.resolution_time)

    def to_dict(self) -> dict[str, Any]:
        ob_dict = None
        if self.order_book:
            ob_dict = {
                "bids": [{"price": lvl.price, "quantity": lvl.quantity} for lvl in self.order_book.bids],
                "asks": [{"price": lvl.price, "quantity": lvl.quantity} for lvl in self.order_book.asks],
            }
        return {
            "market_id": self.market_id,
            "timestamp": to_iso_utc(self.timestamp),
            "status": self.status,
            "question": self.question,
            "category": self.category,
            "resolution_time": to_iso_utc(self.resolution_time),
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "no_bid": self.no_bid,
            "no_ask": self.no_ask,
            "last_price": self.last_price,
            "midpoint": self.midpoint,
            "spread": self.spread,
            "liquidity": self.liquidity,
            "volume_24h": self.volume_24h,
            "order_book": ob_dict,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoricalSnapshot:
        ob = None
        if data.get("order_book"):
            bids = [OrderBookLevel(float(b["price"]), float(b["quantity"])) for b in data["order_book"].get("bids", [])]
            asks = [OrderBookLevel(float(a["price"]), float(a["quantity"])) for a in data["order_book"].get("asks", [])]
            ob = OrderBook(bids=bids, asks=asks)

        return cls(
            market_id=str(data["market_id"]),
            timestamp=parse_iso_utc(str(data["timestamp"])),
            status=str(data.get("status", "ACTIVE")),
            question=str(data.get("question", "")),
            category=str(data.get("category", "GENERAL")),
            resolution_time=parse_iso_utc(str(data["resolution_time"])),
            yes_bid=float(data["yes_bid"]) if data.get("yes_bid") is not None else None,
            yes_ask=float(data["yes_ask"]) if data.get("yes_ask") is not None else None,
            no_bid=float(data["no_bid"]) if data.get("no_bid") is not None else None,
            no_ask=float(data["no_ask"]) if data.get("no_ask") is not None else None,
            last_price=float(data["last_price"]) if data.get("last_price") is not None else None,
            midpoint=float(data["midpoint"]) if data.get("midpoint") is not None else None,
            spread=float(data["spread"]) if data.get("spread") is not None else None,
            liquidity=float(data.get("liquidity", 0.0)),
            volume_24h=float(data.get("volume_24h", 0.0)),
            order_book=ob,
            metadata=data.get("metadata", {}),
        )


@dataclass
class HistoricalResolution:
    """Chronologically isolated event resolution record revealed only when clock >= resolved_at."""

    market_id: str
    resolved_outcome: Side
    resolved_at: datetime
    payout_rate: float = 1.0

    def __post_init__(self) -> None:
        self.resolved_at = ensure_utc(self.resolved_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_id": self.market_id,
            "resolved_outcome": self.resolved_outcome.value,
            "resolved_at": to_iso_utc(self.resolved_at),
            "payout_rate": self.payout_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoricalResolution:
        return cls(
            market_id=str(data["market_id"]),
            resolved_outcome=Side(data["resolved_outcome"]),
            resolved_at=parse_iso_utc(str(data["resolved_at"])),
            payout_rate=float(data.get("payout_rate", 1.0)),
        )


@dataclass
class QueuedOrder:
    """An approved order waiting for simulated execution latency to elapse before filling against fresh depth."""

    order_id: str
    market_id: str
    side: Side
    decision: RiskDecision
    proposal: TradeProposal
    decision_time: datetime
    earliest_fill_time: datetime
    decision_snapshot: MarketSnapshot


@dataclass
class ReplayConfig:
    """Configuration parameters for a historical simulation run."""

    dataset_path: str
    latency_seconds: float = 0.0  # Simulated decision-to-fill latency
    initial_bankroll: float = 1000.0
    random_seed: int = 42
    start_time: datetime | None = None
    end_time: datetime | None = None
    system_config: SystemConfig | None = None


@dataclass
class ReplayResult:
    """Comprehensive statistical and portfolio outcome of a historical replay run."""

    # 1. Traceability
    replay_id: str
    dataset_id: str
    model_version: str
    config_hash: str
    random_seed: int
    git_commit_sha: str
    started_at: datetime
    finished_at: datetime
    simulated_start: datetime
    simulated_end: datetime
    cycles_count: int
    snapshots_count: int
    latency_seconds: float

    # 2. Forecast Calibration & Quality
    calibration: CalibrationReport

    # 3. Simulated Paper Portfolio Results
    initial_bankroll: float
    final_equity: float
    total_pnl: float
    simulated_return_pct: float
    max_drawdown_pct: float
    turnover: float
    total_slippage: float
    total_fees: float
    proposals_count: int
    accepted_count: int
    rejected_count: int
    fills_count: int
    avg_raw_edge: float
    avg_robust_edge: float
    final_open_positions_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "traceability": {
                "replay_id": self.replay_id,
                "dataset_id": self.dataset_id,
                "model_version": self.model_version,
                "config_hash": self.config_hash,
                "random_seed": self.random_seed,
                "git_commit_sha": self.git_commit_sha,
                "started_at": to_iso_utc(self.started_at),
                "finished_at": to_iso_utc(self.finished_at),
                "simulated_start": to_iso_utc(self.simulated_start),
                "simulated_end": to_iso_utc(self.simulated_end),
                "cycles_count": self.cycles_count,
                "snapshots_count": self.snapshots_count,
                "latency_seconds": self.latency_seconds,
            },
            "calibration": {
                "sample_size": self.calibration.sample_size,
                "brier_score": self.calibration.brier_score,
                "log_loss": self.calibration.log_loss,
                "forecast_bias": self.calibration.forecast_bias,
                "expected_calibration_error": self.calibration.expected_calibration_error,
                "maximum_calibration_error": self.calibration.maximum_calibration_error,
                "by_category": self.calibration.by_category,
                "by_model_version": self.calibration.by_model_version,
                "by_edge_bucket": self.calibration.by_edge_bucket,
                "by_risk_state": self.calibration.by_risk_state,
            },
            "portfolio": {
                "initial_bankroll": self.initial_bankroll,
                "final_equity": self.final_equity,
                "total_pnl": self.total_pnl,
                "simulated_return_pct": self.simulated_return_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
                "turnover": self.turnover,
                "total_slippage": self.total_slippage,
                "total_fees": self.total_fees,
                "proposals_count": self.proposals_count,
                "accepted_count": self.accepted_count,
                "rejected_count": self.rejected_count,
                "fills_count": self.fills_count,
                "avg_raw_edge": self.avg_raw_edge,
                "avg_robust_edge": self.avg_robust_edge,
                "final_open_positions_count": self.final_open_positions_count,
            },
        }
