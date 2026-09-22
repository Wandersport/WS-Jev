"""Domain models for the prediction market research system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pm_research.utils import ensure_utc, now_utc


class Side(str, Enum):
    YES = "YES"
    NO = "NO"


class MarketStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    RESOLVED = "RESOLVED"


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    SURVIVAL = "SURVIVAL"
    CRITICAL = "CRITICAL"
    HALTED = "HALTED"


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.price <= 1.0):
            raise ValueError(f"OrderBookLevel price must be in [0, 1], got {self.price}")
        if self.quantity < 0:
            raise ValueError(f"OrderBookLevel quantity cannot be negative, got {self.quantity}")


@dataclass
class OrderBook:
    bids: list[OrderBookLevel] = field(default_factory=list)  # sorted descending by price
    asks: list[OrderBookLevel] = field(default_factory=list)  # sorted ascending by price

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> float | None:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return bb or ba

    @property
    def spread(self) -> float | None:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return max(0.0, ba - bb)
        return None

    @property
    def total_bid_depth(self) -> float:
        return sum(lvl.quantity for lvl in self.bids)

    @property
    def total_ask_depth(self) -> float:
        return sum(lvl.quantity for lvl in self.asks)


@dataclass
class MarketQuote:
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    last_price: float | None = None
    midpoint: float | None = None
    spread: float | None = None
    liquidity: float = 0.0
    volume_24h: float = 0.0


@dataclass
class Market:
    market_id: str
    question: str
    category: str
    status: MarketStatus
    resolution_time: datetime
    quote: MarketQuote
    order_book: OrderBook | None = None
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)
    resolved_outcome: Side | None = None
    resolution_time_actual: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.resolution_time = ensure_utc(self.resolution_time)
        self.created_at = ensure_utc(self.created_at)
        self.updated_at = ensure_utc(self.updated_at)
        if self.resolution_time_actual is not None:
            self.resolution_time_actual = ensure_utc(self.resolution_time_actual)


@dataclass
class MarketSnapshot:
    snapshot_id: str
    cycle_id: str
    market_id: str
    timestamp: datetime
    market: Market

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)


@dataclass
class ResearchFeatures:
    feature_id: str
    cycle_id: str
    market_id: str
    timestamp: datetime
    values: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)


@dataclass
class ProbabilityEstimate:
    estimate_id: str
    cycle_id: str
    market_id: str
    snapshot_id: str
    model_version: str
    q_hat: float  # P(YES) in [0.0, 1.0]
    reference_probability: float  # market implied price / probability
    uncertainty: float  # sigma_q >= 0
    feature_contributions: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)
        if not (0.0 <= self.q_hat <= 1.0):
            raise ValueError(f"q_hat must be in [0, 1], got {self.q_hat}")
        if not (0.0 <= self.reference_probability <= 1.0):
            raise ValueError(f"reference_probability must be in [0, 1], got {self.reference_probability}")
        if self.uncertainty < 0.0:
            raise ValueError(f"uncertainty must be non-negative, got {self.uncertainty}")


@dataclass
class TradeProposal:
    proposal_id: str
    cycle_id: str
    market_id: str
    side: Side
    quote_price: float
    q_hat: float
    sigma_q: float
    model_version: str
    raw_edge: float
    robust_edge: float
    proposed_size_contracts: float
    proposed_capital: float
    reason_codes: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)


@dataclass
class RiskDecision:
    decision_id: str
    proposal_id: str
    cycle_id: str
    market_id: str
    accepted: bool
    allocated_contracts: float
    allocated_capital: float
    reason_code: str
    risk_state: RiskState
    timestamp: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)


@dataclass
class PaperOrder:
    paper_order_id: str
    proposal_id: str
    decision_id: str
    market_id: str
    side: Side
    requested_quantity: float
    limit_price: float
    snapshot_id: str
    timestamp: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)


@dataclass
class PaperFill:
    fill_id: str
    paper_order_id: str
    proposal_id: str
    market_id: str
    side: Side
    requested_quantity: float
    filled_quantity: float
    unfilled_quantity: float
    average_fill_price: float
    slippage: float
    fees: float
    total_cost: float
    snapshot_id: str
    timestamp: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)


@dataclass
class PaperPosition:
    position_id: str
    market_id: str
    side: Side
    quantity: float
    average_entry_price: float
    total_cost: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    realized_pnl: float
    category: str
    status: str = "OPEN"  # OPEN or CLOSED
    opened_at: datetime = field(default_factory=now_utc)
    closed_at: datetime | None = None
    resolved_outcome: Side | None = None

    def __post_init__(self) -> None:
        self.opened_at = ensure_utc(self.opened_at)
        if self.closed_at is not None:
            self.closed_at = ensure_utc(self.closed_at)


@dataclass
class PortfolioSnapshot:
    snapshot_id: str
    cycle_id: str
    timestamp: datetime
    bankroll_initial: float
    virtual_cash: float
    positions_value: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    high_water_mark: float
    current_drawdown: float
    max_drawdown: float
    total_exposure: float
    category_exposure: dict[str, float]
    open_position_count: int
    risk_state: RiskState

    def __post_init__(self) -> None:
        self.timestamp = ensure_utc(self.timestamp)


@dataclass
class CalibrationObservation:
    observation_id: str
    estimate_id: str
    market_id: str
    model_version: str
    predicted_probability: float  # q_hat
    actual_outcome: float  # 1.0 for YES, 0.0 for NO
    category: str
    edge_bucket: str
    risk_state: str
    brier_score: float
    log_loss: float
    resolved_at: datetime

    def __post_init__(self) -> None:
        self.resolved_at = ensure_utc(self.resolved_at)
