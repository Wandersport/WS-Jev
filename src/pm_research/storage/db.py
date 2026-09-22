"""Local SQLite transactional storage for audit trail and research reproducibility."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from pm_research.domain.models import (
    CalibrationObservation,
    MarketSnapshot,
    PaperFill,
    PaperOrder,
    PaperPosition,
    PortfolioSnapshot,
    ProbabilityEstimate,
    ResearchFeatures,
    RiskDecision,
    RiskState,
    Side,
    TradeProposal,
)
from pm_research.research.btc5m.ablation import BTC5mAblationForecast
from pm_research.research.btc5m.contract import BTC5mRoundInfo
from pm_research.research.btc5m.snapshot import BTC5mFeatureSnapshot
from pm_research.research.jev_openrouter import (
    JevCaptureRecord,
    JevForecast,
    JevResolutionScore,
)
from pm_research.utils import parse_iso_utc, to_iso_utc


class _ConnContext:
    """Connection context manager that manages transaction isolation and closure."""

    def __init__(self, conn: sqlite3.Connection, should_close: bool = False) -> None:
        self.conn = conn
        self.should_close = should_close

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            if self.conn.in_transaction:
                self.conn.commit()
        else:
            if self.conn.in_transaction:
                self.conn.rollback()
        if self.should_close:
            self.conn.close()


class Database:
    """Transactional SQLite database for paper trading research."""

    def __init__(self, db_path: str | Path = "data/pm_research.db") -> None:
        self.is_memory = str(db_path) == ":memory:"
        if self.is_memory:
            self.db_path = Path(":memory:")
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
            self._shared_conn.execute("PRAGMA foreign_keys = ON")
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._shared_conn = None
        self._local = threading.local()
        self._init_tables()

    def _create_connection(self) -> sqlite3.Connection:
        if self.is_memory and self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _get_connection(self) -> _ConnContext:
        if self.is_memory and self._shared_conn is not None:
            return _ConnContext(self._shared_conn, should_close=False)
        active = getattr(self._local, "active_conn", None)
        if active is not None:
            return _ConnContext(active, should_close=False)
        return _ConnContext(self._create_connection(), should_close=True)

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing an atomic transaction with automatic commit/rollback."""
        if getattr(self._local, "active_conn", None) is not None:
            yield self._local.active_conn
            return

        if self.is_memory and self._shared_conn is not None:
            conn = self._shared_conn
            self._local.active_conn = conn
            try:
                if not conn.in_transaction:
                    conn.execute("BEGIN IMMEDIATE")
                yield conn
                if conn.in_transaction:
                    conn.commit()
            except Exception:
                if conn.in_transaction:
                    conn.rollback()
                raise
            finally:
                self._local.active_conn = None
            return

        conn = self._create_connection()
        self._local.active_conn = conn
        try:
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if conn.in_transaction:
                conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            self._local.active_conn = None
            conn.close()

    def reset_database(self) -> None:
        """Cleanly reset all tables to an empty state for deterministic demo and test runs."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                DELETE FROM calibration_observations;
                DELETE FROM market_resolutions;
                DELETE FROM portfolio_snapshots;
                DELETE FROM paper_positions;
                DELETE FROM paper_fills;
                DELETE FROM paper_orders;
                DELETE FROM risk_decisions;
                DELETE FROM trade_proposals;
                DELETE FROM probability_estimates;
                DELETE FROM research_features;
                DELETE FROM market_snapshots;
                DELETE FROM cycles;
                DELETE FROM jev_resolution_scores;
                DELETE FROM jev_forecasts;
                DELETE FROM jev_captures;
                """
            )

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cycles (
                    cycle_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    config_hash TEXT NOT NULL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS market_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    category TEXT NOT NULL,
                    question TEXT NOT NULL,
                    resolution_time TEXT NOT NULL,
                    yes_bid REAL,
                    yes_ask REAL,
                    no_bid REAL,
                    no_ask REAL,
                    last_price REAL,
                    midpoint REAL,
                    spread REAL,
                    liquidity REAL NOT NULL,
                    volume_24h REAL NOT NULL,
                    order_book_json TEXT,
                    raw_json TEXT,
                    UNIQUE(cycle_id, market_id)
                );

                CREATE TABLE IF NOT EXISTS research_features (
                    feature_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    values_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    UNIQUE(cycle_id, market_id)
                );

                CREATE TABLE IF NOT EXISTS probability_estimates (
                    estimate_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    q_hat REAL NOT NULL,
                    reference_probability REAL NOT NULL,
                    uncertainty REAL NOT NULL,
                    contributions_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    UNIQUE(cycle_id, market_id, model_version)
                );

                CREATE TABLE IF NOT EXISTS trade_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quote_price REAL NOT NULL,
                    q_hat REAL NOT NULL,
                    sigma_q REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    raw_edge REAL NOT NULL,
                    robust_edge REAL NOT NULL,
                    proposed_size_contracts REAL NOT NULL,
                    proposed_capital REAL NOT NULL,
                    reason_codes_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    UNIQUE(cycle_id, market_id)
                );

                CREATE TABLE IF NOT EXISTS risk_decisions (
                    decision_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL UNIQUE,
                    cycle_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    allocated_contracts REAL NOT NULL,
                    allocated_capital REAL NOT NULL,
                    reason_code TEXT NOT NULL,
                    risk_state TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_orders (
                    paper_order_id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL UNIQUE,
                    decision_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    requested_quantity REAL NOT NULL,
                    limit_price REAL NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_fills (
                    fill_id TEXT PRIMARY KEY,
                    paper_order_id TEXT NOT NULL UNIQUE,
                    proposal_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    requested_quantity REAL NOT NULL,
                    filled_quantity REAL NOT NULL,
                    unfilled_quantity REAL NOT NULL,
                    average_fill_price REAL NOT NULL,
                    slippage REAL NOT NULL,
                    fees REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS paper_positions (
                    position_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    average_entry_price REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    current_price REAL NOT NULL,
                    current_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT,
                    resolved_outcome TEXT
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    bankroll_initial REAL NOT NULL,
                    virtual_cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    equity REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    high_water_mark REAL NOT NULL,
                    current_drawdown REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    total_exposure REAL NOT NULL,
                    category_exposure_json TEXT NOT NULL,
                    open_position_count INTEGER NOT NULL,
                    risk_state TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS market_resolutions (
                    market_id TEXT PRIMARY KEY,
                    resolved_outcome TEXT NOT NULL,
                    resolved_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS calibration_observations (
                    observation_id TEXT PRIMARY KEY,
                    estimate_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    predicted_probability REAL NOT NULL,
                    actual_outcome REAL NOT NULL,
                    category TEXT NOT NULL,
                    edge_bucket TEXT,
                    risk_state TEXT,
                    brier_score REAL NOT NULL,
                    log_loss REAL NOT NULL,
                    resolved_at TEXT NOT NULL,
                    UNIQUE(estimate_id, market_id)
                );

                CREATE TABLE IF NOT EXISTS replay_runs (
                    replay_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jev_captures (
                    capture_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    market_question TEXT NOT NULL,
                    category TEXT NOT NULL,
                    resolution_time TEXT,
                    market_prob REAL NOT NULL,
                    ilsa_prob REAL NOT NULL,
                    jev_blind_prob REAL,
                    jev_market_aware_prob REAL,
                    resolved_outcome TEXT,
                    resolved_at TEXT,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jev_forecasts (
                    forecast_id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_returned TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    market_question TEXT NOT NULL,
                    resolution_criteria TEXT,
                    market_resolution_time TEXT,
                    jev_yes_probability REAL NOT NULL,
                    jev_no_probability REAL NOT NULL,
                    jev_choice TEXT NOT NULL,
                    confidence REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost REAL,
                    raw_response_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(market_id, captured_at, model_id, condition, schema_version)
                );

                CREATE TABLE IF NOT EXISTS jev_resolution_scores (
                    score_id TEXT PRIMARY KEY,
                    capture_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    resolved_outcome TEXT NOT NULL,
                    resolved_at TEXT NOT NULL,
                    scored_at TEXT NOT NULL,
                    market_prob REAL NOT NULL,
                    ilsa_prob REAL NOT NULL,
                    jev_blind_prob REAL,
                    jev_market_aware_prob REAL,
                    market_brier REAL NOT NULL,
                    ilsa_brier REAL NOT NULL,
                    jev_blind_brier REAL,
                    jev_market_aware_brier REAL,
                    market_log_loss REAL NOT NULL,
                    ilsa_log_loss REAL NOT NULL,
                    jev_blind_log_loss REAL,
                    jev_market_aware_log_loss REAL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(capture_id)
                );

                -- BTC 5-minute prospective research tables
                CREATE TABLE IF NOT EXISTS btc5m_rounds (
                    round_slug TEXT PRIMARY KEY,
                    start_epoch INTEGER NOT NULL,
                    end_epoch INTEGER NOT NULL,
                    duration_sec INTEGER NOT NULL,
                    price_to_beat REAL,
                    price_to_beat_source TEXT,
                    up_token_id TEXT,
                    down_token_id TEXT,
                    up_outcome_index INTEGER NOT NULL DEFAULT 0,
                    down_outcome_index INTEGER NOT NULL DEFAULT 1,
                    condition_id TEXT,
                    question TEXT,
                    discovered_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resolved_outcome TEXT,
                    resolution_price REAL,
                    settled_at TEXT
                );

                CREATE TABLE IF NOT EXISTS btc5m_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    round_slug TEXT NOT NULL,
                    target_horizon_sec INTEGER NOT NULL,
                    captured_at_ms INTEGER NOT NULL,
                    capture_started_at_ms INTEGER,
                    capture_completed_at_ms INTEGER,
                    actual_freeze_timestamp_ms INTEGER,
                    target_scheduled_ms INTEGER NOT NULL,
                    timing_deviation_ms INTEGER NOT NULL,
                    seconds_remaining REAL NOT NULL,
                    reference_source TEXT NOT NULL,
                    price_to_beat REAL,
                    price_to_beat_source TEXT,
                    anchor_price REAL,
                    anchor_source TEXT,
                    anchor_source_timestamp_ms INTEGER,
                    anchor_received_timestamp_ms INTEGER,
                    current_reference_price REAL NOT NULL,
                    ref_distance_to_beat_bps REAL,
                    ref_return_10s_bps REAL,
                    ref_return_30s_bps REAL,
                    ref_return_60s_bps REAL,
                    market_q REAL,
                    market_q_primary REAL,
                    market_q_implied_cross_outcome REAL,
                    up_best_bid REAL,
                    up_best_ask REAL,
                    down_best_bid REAL,
                    down_best_ask REAL,
                    poly_spread REAL,
                    binance_perp_mid REAL,
                    binance_microprice REAL,
                    binance_microprice_offset_bps REAL,
                    binance_top5_depth_imbalance REAL,
                    binance_top20_depth_imbalance REAL,
                    binance_spread_bps REAL,
                    binance_taker_flow_10s_imbalance REAL,
                    binance_taker_flow_30s_imbalance REAL,
                    binance_taker_flow_60s_imbalance REAL,
                    binance_return_10s_bps REAL,
                    binance_return_30s_bps REAL,
                    binance_return_60s_bps REAL,
                    binance_return_since_open_bps REAL,
                    binance_open_mid REAL,
                    binance_open_timestamp_ms INTEGER,
                    binance_basis_bps REAL,
                    is_valid INTEGER NOT NULL,
                    skip_reason TEXT,
                    raw_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS btc5m_forecasts (
                    forecast_id TEXT PRIMARY KEY,
                    snapshot_id TEXT NOT NULL,
                    round_slug TEXT NOT NULL,
                    target_horizon_sec INTEGER NOT NULL,
                    condition TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    captured_at_ms INTEGER NOT NULL,
                    market_q REAL,
                    jev_up_prob REAL NOT NULL,
                    jev_down_prob REAL NOT NULL,
                    jev_choice TEXT NOT NULL,
                    confidence REAL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    from_cache INTEGER NOT NULL,
                    raw_response_hash TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    request_started_at_utc TEXT,
                    response_received_at_utc TEXT,
                    response_received_at_ms INTEGER,
                    round_end_ms INTEGER,
                    snapshot_hash TEXT,
                    request_order INTEGER NOT NULL DEFAULT 1,
                    is_valid INTEGER NOT NULL DEFAULT 1,
                    rejection_reason TEXT,
                    UNIQUE(snapshot_id, condition)
                );

                CREATE TABLE IF NOT EXISTS btc5m_resolution_scores (
                    score_id TEXT PRIMARY KEY,
                    round_slug TEXT NOT NULL,
                    target_horizon_sec INTEGER NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    resolved_outcome TEXT NOT NULL,
                    resolution_price REAL,
                    resolved_at TEXT NOT NULL,
                    scored_at TEXT NOT NULL,
                    market_q REAL,
                    market_brier REAL,
                    market_log_loss REAL,
                    cond_a_prob REAL,
                    cond_a_brier REAL,
                    cond_a_log_loss REAL,
                    cond_b_prob REAL,
                    cond_b_brier REAL,
                    cond_b_log_loss REAL,
                    cond_c_prob REAL,
                    cond_c_brier REAL,
                    cond_c_log_loss REAL,
                    cond_d_prob REAL,
                    cond_d_brier REAL,
                    cond_d_log_loss REAL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(round_slug, target_horizon_sec)
                );

                -- Indexes for fast query and integrity verification
                CREATE INDEX IF NOT EXISTS idx_snapshots_market ON market_snapshots(market_id);
                CREATE INDEX IF NOT EXISTS idx_estimates_market ON probability_estimates(market_id);
                CREATE INDEX IF NOT EXISTS idx_positions_status ON paper_positions(status);
                CREATE INDEX IF NOT EXISTS idx_positions_market ON paper_positions(market_id);
                CREATE INDEX IF NOT EXISTS idx_calib_model ON calibration_observations(model_version);
                CREATE INDEX IF NOT EXISTS idx_decisions_accepted ON risk_decisions(accepted);
                CREATE INDEX IF NOT EXISTS idx_replay_dataset ON replay_runs(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_jev_captures_market ON jev_captures(market_id);
                CREATE INDEX IF NOT EXISTS idx_jev_forecasts_capture ON jev_forecasts(capture_id);
                CREATE INDEX IF NOT EXISTS idx_jev_forecasts_market ON jev_forecasts(market_id);
                CREATE INDEX IF NOT EXISTS idx_jev_scores_market ON jev_resolution_scores(market_id);
                CREATE INDEX IF NOT EXISTS idx_btc5m_rounds_status ON btc5m_rounds(status);
                CREATE INDEX IF NOT EXISTS idx_btc5m_snapshots_round ON btc5m_snapshots(round_slug);
                CREATE INDEX IF NOT EXISTS idx_btc5m_forecasts_round ON btc5m_forecasts(round_slug);
                CREATE INDEX IF NOT EXISTS idx_btc5m_forecasts_condition ON btc5m_forecasts(condition);
                CREATE INDEX IF NOT EXISTS idx_btc5m_scores_round ON btc5m_resolution_scores(round_slug);
                """
            )

    # Persistence methods
    def save_cycle(self, cycle_id: str, started_at: str, config_hash: str, notes: str = "", conn: sqlite3.Connection | None = None) -> None:
        sql = "INSERT OR REPLACE INTO cycles (cycle_id, started_at, config_hash, notes) VALUES (?, ?, ?, ?)"
        params = (cycle_id, started_at, config_hash, notes)
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def finish_cycle(self, cycle_id: str, finished_at: str, conn: sqlite3.Connection | None = None) -> None:
        sql = "UPDATE cycles SET finished_at = ? WHERE cycle_id = ?"
        params = (finished_at, cycle_id)
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_market_snapshot(self, snapshot: MarketSnapshot, conn: sqlite3.Connection | None = None) -> None:
        m = snapshot.market
        q = m.quote
        ob_json = None
        if m.order_book:
            ob_json = json.dumps({
                "bids": [{"price": lvl.price, "quantity": lvl.quantity} for lvl in m.order_book.bids],
                "asks": [{"price": lvl.price, "quantity": lvl.quantity} for lvl in m.order_book.asks],
            })

        sql = """
        INSERT OR REPLACE INTO market_snapshots (
            snapshot_id, cycle_id, market_id, timestamp, status, category, question,
            resolution_time, yes_bid, yes_ask, no_bid, no_ask, last_price, midpoint,
            spread, liquidity, volume_24h, order_book_json, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            snapshot.snapshot_id,
            snapshot.cycle_id,
            snapshot.market_id,
            to_iso_utc(snapshot.timestamp),
            m.status.value,
            m.category,
            m.question,
            to_iso_utc(m.resolution_time),
            q.yes_bid,
            q.yes_ask,
            q.no_bid,
            q.no_ask,
            q.last_price,
            q.midpoint,
            q.spread,
            q.liquidity,
            q.volume_24h,
            ob_json,
            json.dumps(m.metadata),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_research_features(self, features: ResearchFeatures, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT OR REPLACE INTO research_features (
            feature_id, cycle_id, market_id, timestamp, values_json, provenance_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            features.feature_id,
            features.cycle_id,
            features.market_id,
            to_iso_utc(features.timestamp),
            json.dumps(features.values),
            json.dumps(features.provenance),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_probability_estimate(self, estimate: ProbabilityEstimate, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT OR REPLACE INTO probability_estimates (
            estimate_id, cycle_id, market_id, snapshot_id, model_version,
            q_hat, reference_probability, uncertainty, contributions_json, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            estimate.estimate_id,
            estimate.cycle_id,
            estimate.market_id,
            estimate.snapshot_id,
            estimate.model_version,
            estimate.q_hat,
            estimate.reference_probability,
            estimate.uncertainty,
            json.dumps(estimate.feature_contributions),
            to_iso_utc(estimate.timestamp),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_trade_proposal(self, proposal: TradeProposal, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT OR REPLACE INTO trade_proposals (
            proposal_id, cycle_id, market_id, side, quote_price, q_hat, sigma_q,
            model_version, raw_edge, robust_edge, proposed_size_contracts,
            proposed_capital, reason_codes_json, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            proposal.proposal_id,
            proposal.cycle_id,
            proposal.market_id,
            proposal.side.value,
            proposal.quote_price,
            proposal.q_hat,
            proposal.sigma_q,
            proposal.model_version,
            proposal.raw_edge,
            proposal.robust_edge,
            proposal.proposed_size_contracts,
            proposal.proposed_capital,
            json.dumps(proposal.reason_codes),
            to_iso_utc(proposal.timestamp),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_risk_decision(self, decision: RiskDecision, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT OR REPLACE INTO risk_decisions (
            decision_id, proposal_id, cycle_id, market_id, accepted,
            allocated_contracts, allocated_capital, reason_code, risk_state, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            decision.decision_id,
            decision.proposal_id,
            decision.cycle_id,
            decision.market_id,
            1 if decision.accepted else 0,
            decision.allocated_contracts,
            decision.allocated_capital,
            decision.reason_code,
            decision.risk_state.value,
            to_iso_utc(decision.timestamp),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_paper_order(self, order: PaperOrder, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT OR REPLACE INTO paper_orders (
            paper_order_id, proposal_id, decision_id, market_id, side,
            requested_quantity, limit_price, snapshot_id, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            order.paper_order_id,
            order.proposal_id,
            order.decision_id,
            order.market_id,
            order.side.value,
            order.requested_quantity,
            order.limit_price,
            order.snapshot_id,
            to_iso_utc(order.timestamp),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_paper_fill(self, fill: PaperFill, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT OR REPLACE INTO paper_fills (
            fill_id, paper_order_id, proposal_id, market_id, side,
            requested_quantity, filled_quantity, unfilled_quantity,
            average_fill_price, slippage, fees, total_cost, snapshot_id, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            fill.fill_id,
            fill.paper_order_id,
            fill.proposal_id,
            fill.market_id,
            fill.side.value,
            fill.requested_quantity,
            fill.filled_quantity,
            fill.unfilled_quantity,
            fill.average_fill_price,
            fill.slippage,
            fill.fees,
            fill.total_cost,
            fill.snapshot_id,
            to_iso_utc(fill.timestamp),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_paper_position(self, pos: PaperPosition, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT OR REPLACE INTO paper_positions (
            position_id, market_id, side, quantity, average_entry_price,
            total_cost, current_price, current_value, unrealized_pnl,
            realized_pnl, category, status, opened_at, closed_at, resolved_outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            pos.position_id,
            pos.market_id,
            pos.side.value,
            pos.quantity,
            pos.average_entry_price,
            pos.total_cost,
            pos.current_price,
            pos.current_value,
            pos.unrealized_pnl,
            pos.realized_pnl,
            pos.category,
            pos.status,
            to_iso_utc(pos.opened_at),
            to_iso_utc(pos.closed_at) if pos.closed_at else None,
            pos.resolved_outcome.value if pos.resolved_outcome else None,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_portfolio_snapshot(self, snap: PortfolioSnapshot, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT INTO portfolio_snapshots (
            snapshot_id, cycle_id, timestamp, bankroll_initial, virtual_cash,
            positions_value, equity, realized_pnl, unrealized_pnl, total_pnl,
            high_water_mark, current_drawdown, max_drawdown, total_exposure,
            category_exposure_json, open_position_count, risk_state
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            snap.snapshot_id,
            snap.cycle_id,
            to_iso_utc(snap.timestamp),
            snap.bankroll_initial,
            snap.virtual_cash,
            snap.positions_value,
            snap.equity,
            snap.realized_pnl,
            snap.unrealized_pnl,
            snap.total_pnl,
            snap.high_water_mark,
            snap.current_drawdown,
            snap.max_drawdown,
            snap.total_exposure,
            json.dumps(snap.category_exposure),
            snap.open_position_count,
            snap.risk_state.value,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_resolution(self, market_id: str, resolved_outcome: Side, resolved_at: str, conn: sqlite3.Connection | None = None) -> None:
        sql = "INSERT OR REPLACE INTO market_resolutions (market_id, resolved_outcome, resolved_at) VALUES (?, ?, ?)"
        params = (market_id, resolved_outcome.value, resolved_at)
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_calibration_observation(self, obs: CalibrationObservation, conn: sqlite3.Connection | None = None) -> None:
        sql = """
        INSERT OR REPLACE INTO calibration_observations (
            observation_id, estimate_id, market_id, model_version,
            predicted_probability, actual_outcome, category, edge_bucket,
            risk_state, brier_score, log_loss, resolved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            obs.observation_id,
            obs.estimate_id,
            obs.market_id,
            obs.model_version,
            obs.predicted_probability,
            obs.actual_outcome,
            obs.category,
            obs.edge_bucket,
            obs.risk_state,
            obs.brier_score,
            obs.log_loss,
            to_iso_utc(obs.resolved_at),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def get_open_positions(self) -> list[PaperPosition]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM paper_positions WHERE status = 'OPEN'").fetchall()
            positions = []
            for r in rows:
                positions.append(
                    PaperPosition(
                        position_id=r["position_id"],
                        market_id=r["market_id"],
                        side=Side(r["side"]),
                        quantity=r["quantity"],
                        average_entry_price=r["average_entry_price"],
                        total_cost=r["total_cost"],
                        current_price=r["current_price"],
                        current_value=r["current_value"],
                        unrealized_pnl=r["unrealized_pnl"],
                        realized_pnl=r["realized_pnl"],
                        category=r["category"],
                        status=r["status"],
                        opened_at=parse_iso_utc(r["opened_at"]),
                        closed_at=parse_iso_utc(r["closed_at"]) if r["closed_at"] else None,
                        resolved_outcome=Side(r["resolved_outcome"]) if r["resolved_outcome"] else None,
                    )
                )
            return positions

    def get_all_positions(self) -> list[PaperPosition]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM paper_positions ORDER BY opened_at ASC").fetchall()
            positions = []
            for r in rows:
                positions.append(
                    PaperPosition(
                        position_id=r["position_id"],
                        market_id=r["market_id"],
                        side=Side(r["side"]),
                        quantity=r["quantity"],
                        average_entry_price=r["average_entry_price"],
                        total_cost=r["total_cost"],
                        current_price=r["current_price"],
                        current_value=r["current_value"],
                        unrealized_pnl=r["unrealized_pnl"],
                        realized_pnl=r["realized_pnl"],
                        category=r["category"],
                        status=r["status"],
                        opened_at=parse_iso_utc(r["opened_at"]),
                        closed_at=parse_iso_utc(r["closed_at"]) if r["closed_at"] else None,
                        resolved_outcome=Side(r["resolved_outcome"]) if r["resolved_outcome"] else None,
                    )
                )
            return positions

    def get_latest_portfolio_snapshot(self) -> PortfolioSnapshot | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1").fetchone()
            if not row:
                return None
            return PortfolioSnapshot(
                snapshot_id=row["snapshot_id"],
                cycle_id=row["cycle_id"],
                timestamp=parse_iso_utc(row["timestamp"]),
                bankroll_initial=row["bankroll_initial"],
                virtual_cash=row["virtual_cash"],
                positions_value=row["positions_value"],
                equity=row["equity"],
                realized_pnl=row["realized_pnl"],
                unrealized_pnl=row["unrealized_pnl"],
                total_pnl=row["total_pnl"],
                high_water_mark=row["high_water_mark"],
                current_drawdown=row["current_drawdown"],
                max_drawdown=row["max_drawdown"],
                total_exposure=row["total_exposure"],
                category_exposure=json.loads(row["category_exposure_json"]),
                open_position_count=row["open_position_count"],
                risk_state=RiskState(row["risk_state"]),
            )

    def get_all_calibration_observations(self) -> list[CalibrationObservation]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM calibration_observations ORDER BY resolved_at ASC").fetchall()
            obs_list = []
            for r in rows:
                obs_list.append(
                    CalibrationObservation(
                        observation_id=r["observation_id"],
                        estimate_id=r["estimate_id"],
                        market_id=r["market_id"],
                        model_version=r["model_version"],
                        predicted_probability=r["predicted_probability"],
                        actual_outcome=r["actual_outcome"],
                        category=r["category"],
                        edge_bucket=r["edge_bucket"] or "UNKNOWN",
                        risk_state=r["risk_state"] or "NORMAL",
                        brier_score=r["brier_score"],
                        log_loss=r["log_loss"],
                        resolved_at=parse_iso_utc(r["resolved_at"]),
                    )
                )
            return obs_list

    def get_market_estimates(self, market_id: str) -> list[ProbabilityEstimate]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM probability_estimates WHERE market_id = ? ORDER BY timestamp DESC",
                (market_id,),
            ).fetchall()
            estimates = []
            for r in rows:
                estimates.append(
                    ProbabilityEstimate(
                        estimate_id=r["estimate_id"],
                        cycle_id=r["cycle_id"],
                        market_id=r["market_id"],
                        snapshot_id=r["snapshot_id"],
                        model_version=r["model_version"],
                        q_hat=r["q_hat"],
                        reference_probability=r["reference_probability"],
                        uncertainty=r["uncertainty"],
                        feature_contributions=json.loads(r["contributions_json"]),
                        timestamp=parse_iso_utc(r["timestamp"]),
                    )
                )
            return estimates

    def get_resolution(self, market_id: str) -> tuple[Side, str] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT resolved_outcome, resolved_at FROM market_resolutions WHERE market_id = ?", (market_id,)).fetchone()
            if row:
                return Side(row["resolved_outcome"]), row["resolved_at"]
            return None

    def get_cycle_count(self) -> int:
        with self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM cycles").fetchone()
            return row["count"] if row else 0

    def get_cycles(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM cycles ORDER BY started_at ASC").fetchall()
            return [dict(r) for r in rows]

    def get_activity_summary(self) -> dict[str, Any]:
        with self._get_connection() as conn:
            cycles = conn.execute("SELECT COUNT(*) as c FROM cycles").fetchone()["c"]
            proposals = conn.execute("SELECT COUNT(*) as c FROM trade_proposals").fetchone()["c"]
            accepted = conn.execute("SELECT COUNT(*) as c FROM risk_decisions WHERE accepted = 1").fetchone()["c"]
            rejected = conn.execute("SELECT COUNT(*) as c FROM risk_decisions WHERE accepted = 0").fetchone()["c"]
            orders = conn.execute("SELECT COUNT(*) as c FROM paper_orders").fetchone()["c"]
            fills = conn.execute("SELECT COUNT(*) as c FROM paper_fills").fetchone()["c"]
            total_slippage = conn.execute("SELECT COALESCE(SUM(slippage), 0.0) as s FROM paper_fills").fetchone()["s"]
            last_cycle = conn.execute("SELECT cycle_id, started_at FROM cycles ORDER BY started_at DESC LIMIT 1").fetchone()
            return {
                "cycles": cycles,
                "proposals": proposals,
                "accepted": accepted,
                "rejected": rejected,
                "orders": orders,
                "fills": fills,
                "total_slippage": total_slippage,
                "last_cycle_id": last_cycle["cycle_id"] if last_cycle else None,
                "last_cycle_time": last_cycle["started_at"] if last_cycle else None,
            }

    def save_replay_run(
        self,
        replay_id: str,
        dataset_id: str,
        created_at: str,
        config_json: str,
        result_json: str,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        sql = "INSERT OR REPLACE INTO replay_runs (replay_id, dataset_id, created_at, config_json, result_json) VALUES (?, ?, ?, ?, ?)"
        params = (replay_id, dataset_id, created_at, config_json, result_json)
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def get_replay_run(self, replay_id: str) -> dict[str, Any] | None:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM replay_runs WHERE replay_id = ?", (replay_id,)).fetchone()
            if not row:
                return None
            return dict(row)

    def list_replay_runs(self) -> list[dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT replay_id, dataset_id, created_at FROM replay_runs ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def save_jev_capture(
        self,
        capture: JevCaptureRecord,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Persist or update a prospective Jev capture uniting market, Ilsa, and Jev forecasts."""
        sql = """
            INSERT OR REPLACE INTO jev_captures (
                capture_id, market_id, captured_at, market_question, category,
                resolution_time, market_prob, ilsa_prob, jev_blind_prob,
                jev_market_aware_prob, resolved_outcome, resolved_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            capture.capture_id,
            capture.market_id,
            to_iso_utc(capture.captured_at),
            capture.market_question,
            capture.category,
            to_iso_utc(capture.resolution_time) if capture.resolution_time else None,
            capture.market_prob,
            capture.ilsa_prob,
            capture.jev_blind_prob,
            capture.jev_market_aware_prob,
            capture.resolved_outcome,
            to_iso_utc(capture.resolved_at) if capture.resolved_at else None,
            json.dumps(capture.metadata),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_jev_forecast(
        self,
        forecast: JevForecast,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Persist an immutable JevForecast record."""
        sql = """
            INSERT OR REPLACE INTO jev_forecasts (
                forecast_id, capture_id, market_id, condition, model_id,
                model_returned, schema_version, request_hash, captured_at,
                market_question, resolution_criteria, market_resolution_time,
                jev_yes_probability, jev_no_probability, jev_choice,
                confidence, input_tokens, output_tokens, cost,
                raw_response_hash, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            forecast.forecast_id,
            forecast.capture_id,
            forecast.market_id,
            forecast.condition,
            forecast.model_id,
            forecast.model_returned,
            forecast.schema_version,
            forecast.request_hash,
            to_iso_utc(forecast.captured_at_utc),
            forecast.market_question,
            forecast.resolution_criteria,
            to_iso_utc(forecast.market_resolution_time) if forecast.market_resolution_time else None,
            forecast.jev_yes_probability,
            forecast.jev_no_probability,
            forecast.jev_choice,
            forecast.confidence,
            forecast.input_tokens,
            forecast.output_tokens,
            forecast.cost,
            forecast.raw_response_hash,
            json.dumps(forecast.metadata),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def get_jev_captures(self, limit: int = 100) -> list[JevCaptureRecord]:
        """Retrieve recent prospective Jev capture records."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jev_captures ORDER BY captured_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            captures: list[JevCaptureRecord] = []
            for r in rows:
                captures.append(
                    JevCaptureRecord(
                        capture_id=r["capture_id"],
                        market_id=r["market_id"],
                        captured_at=parse_iso_utc(r["captured_at"]),
                        market_question=r["market_question"],
                        category=r["category"],
                        resolution_time=(
                            parse_iso_utc(r["resolution_time"])
                            if r["resolution_time"]
                            else None
                        ),
                        market_prob=float(r["market_prob"]),
                        ilsa_prob=float(r["ilsa_prob"]),
                        jev_blind_prob=(
                            float(r["jev_blind_prob"])
                            if r["jev_blind_prob"] is not None
                            else None
                        ),
                        jev_market_aware_prob=(
                            float(r["jev_market_aware_prob"])
                            if r["jev_market_aware_prob"] is not None
                            else None
                        ),
                        resolved_outcome=r["resolved_outcome"],
                        resolved_at=(
                            parse_iso_utc(r["resolved_at"])
                            if r["resolved_at"]
                            else None
                        ),
                        metadata=json.loads(r["metadata_json"]),
                    )
                )
            return captures

    def get_jev_forecasts(
        self,
        capture_id: str | None = None,
        market_id: str | None = None,
    ) -> list[JevForecast]:
        """Retrieve JevForecast records with optional filtering."""
        with self._get_connection() as conn:
            query = "SELECT * FROM jev_forecasts WHERE 1=1"
            params: list[Any] = []
            if capture_id:
                query += " AND capture_id = ?"
                params.append(capture_id)
            if market_id:
                query += " AND market_id = ?"
                params.append(market_id)
            query += " ORDER BY captured_at DESC"
            rows = conn.execute(query, params).fetchall()
            forecasts: list[JevForecast] = []
            for r in rows:
                forecasts.append(
                    JevForecast(
                        forecast_id=r["forecast_id"],
                        capture_id=r["capture_id"],
                        market_id=r["market_id"],
                        condition=r["condition"],
                        model_id=r["model_id"],
                        model_returned=r["model_returned"],
                        schema_version=r["schema_version"],
                        request_hash=r["request_hash"],
                        captured_at_utc=parse_iso_utc(r["captured_at"]),
                        market_question=r["market_question"],
                        resolution_criteria=r["resolution_criteria"],
                        market_resolution_time=(
                            parse_iso_utc(r["market_resolution_time"])
                            if r["market_resolution_time"]
                            else None
                        ),
                        jev_yes_probability=float(r["jev_yes_probability"]),
                        jev_no_probability=float(r["jev_no_probability"]),
                        jev_choice=r["jev_choice"],
                        confidence=(
                            float(r["confidence"])
                            if r["confidence"] is not None
                            else None
                        ),
                        input_tokens=int(r["input_tokens"]),
                        output_tokens=int(r["output_tokens"]),
                        cost=float(r["cost"]) if r["cost"] is not None else None,
                        raw_response_hash=r["raw_response_hash"],
                        metadata=json.loads(r["metadata_json"]),
                    )
                )
            return forecasts

    def get_unresolved_jev_captures(self) -> list[JevCaptureRecord]:
        """Retrieve prospective captures awaiting market resolution."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jev_captures WHERE resolved_outcome IS NULL ORDER BY captured_at ASC"
            ).fetchall()
            captures: list[JevCaptureRecord] = []
            for r in rows:
                captures.append(
                    JevCaptureRecord(
                        capture_id=r["capture_id"],
                        market_id=r["market_id"],
                        captured_at=parse_iso_utc(r["captured_at"]),
                        market_question=r["market_question"],
                        category=r["category"],
                        resolution_time=(
                            parse_iso_utc(r["resolution_time"])
                            if r["resolution_time"]
                            else None
                        ),
                        market_prob=float(r["market_prob"]),
                        ilsa_prob=float(r["ilsa_prob"]),
                        jev_blind_prob=(
                            float(r["jev_blind_prob"])
                            if r["jev_blind_prob"] is not None
                            else None
                        ),
                        jev_market_aware_prob=(
                            float(r["jev_market_aware_prob"])
                            if r["jev_market_aware_prob"] is not None
                            else None
                        ),
                        resolved_outcome=None,
                        resolved_at=None,
                        metadata=json.loads(r["metadata_json"]),
                    )
                )
            return captures

    def update_jev_capture_resolution(
        self,
        capture_id: str,
        resolved_outcome: str,
        resolved_at: datetime | str,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Record the actual resolution for a prospective capture."""
        res_at_str = to_iso_utc(resolved_at) if isinstance(resolved_at, datetime) else str(resolved_at)
        sql = "UPDATE jev_captures SET resolved_outcome = ?, resolved_at = ? WHERE capture_id = ?"
        params = (resolved_outcome, res_at_str, capture_id)
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def save_jev_resolution_score(
        self,
        score: JevResolutionScore,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Persist resolution score evaluation comparing forecasters against ground truth."""
        sql = """
            INSERT OR REPLACE INTO jev_resolution_scores (
                score_id, capture_id, market_id, resolved_outcome, resolved_at,
                scored_at, market_prob, ilsa_prob, jev_blind_prob, jev_market_aware_prob,
                market_brier, ilsa_brier, jev_blind_brier, jev_market_aware_brier,
                market_log_loss, ilsa_log_loss, jev_blind_log_loss,
                jev_market_aware_log_loss, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            score.score_id,
            score.capture_id,
            score.market_id,
            score.resolved_outcome,
            to_iso_utc(score.resolved_at),
            to_iso_utc(score.scored_at),
            score.market_prob,
            score.ilsa_prob,
            score.jev_blind_prob,
            score.jev_market_aware_prob,
            score.market_brier,
            score.ilsa_brier,
            score.jev_blind_brier,
            score.jev_market_aware_brier,
            score.market_log_loss,
            score.ilsa_log_loss,
            score.jev_blind_log_loss,
            score.jev_market_aware_log_loss,
            json.dumps(score.metadata),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def get_jev_resolution_scores(self) -> list[JevResolutionScore]:
        """Retrieve all evaluated prospective Jev resolution scores."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM jev_resolution_scores ORDER BY scored_at DESC").fetchall()
            scores: list[JevResolutionScore] = []
            for r in rows:
                scores.append(
                    JevResolutionScore(
                        score_id=r["score_id"],
                        capture_id=r["capture_id"],
                        market_id=r["market_id"],
                        resolved_outcome=r["resolved_outcome"],
                        resolved_at=parse_iso_utc(r["resolved_at"]),
                        scored_at=parse_iso_utc(r["scored_at"]),
                        market_prob=float(r["market_prob"]),
                        ilsa_prob=float(r["ilsa_prob"]),
                        jev_blind_prob=(
                            float(r["jev_blind_prob"])
                            if r["jev_blind_prob"] is not None
                            else None
                        ),
                        jev_market_aware_prob=(
                            float(r["jev_market_aware_prob"])
                            if r["jev_market_aware_prob"] is not None
                            else None
                        ),
                        market_brier=float(r["market_brier"]),
                        ilsa_brier=float(r["ilsa_brier"]),
                        jev_blind_brier=(
                            float(r["jev_blind_brier"])
                            if r["jev_blind_brier"] is not None
                            else None
                        ),
                        jev_market_aware_brier=(
                            float(r["jev_market_aware_brier"])
                            if r["jev_market_aware_brier"] is not None
                            else None
                        ),
                        market_log_loss=float(r["market_log_loss"]),
                        ilsa_log_loss=float(r["ilsa_log_loss"]),
                        jev_blind_log_loss=(
                            float(r["jev_blind_log_loss"])
                            if r["jev_blind_log_loss"] is not None
                            else None
                        ),
                        jev_market_aware_log_loss=(
                            float(r["jev_market_aware_log_loss"])
                            if r["jev_market_aware_log_loss"] is not None
                            else None
                        ),
                        metadata=json.loads(r["metadata_json"]),
                    )
                )
            return scores

    # BTC 5-minute prospective lab persistence methods
    def save_btc5m_round(
        self,
        round_info: BTC5mRoundInfo,
        status: str = "active",
        resolved_outcome: str | None = None,
        resolution_price: float | None = None,
        settled_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Persist or update a discovered BTC 5m round contract."""
        sql = """
            INSERT OR REPLACE INTO btc5m_rounds (
                round_slug, start_epoch, end_epoch, duration_sec, price_to_beat,
                price_to_beat_source, up_token_id, down_token_id, up_outcome_index,
                down_outcome_index, condition_id, question, discovered_at, status,
                resolved_outcome, resolution_price, settled_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now_utc = datetime.now().isoformat()
        params = (
            round_info.round_slug,
            round_info.start_epoch,
            round_info.end_epoch,
            round_info.duration_sec,
            round_info.price_to_beat,
            round_info.price_to_beat_source,
            round_info.up_token_id,
            round_info.down_token_id,
            round_info.up_outcome_index,
            round_info.down_outcome_index,
            round_info.condition_id,
            round_info.question,
            now_utc,
            status,
            resolved_outcome,
            resolution_price,
            settled_at,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def update_btc5m_round_resolution(
        self,
        round_slug: str,
        resolved_outcome: str,
        resolution_price: float | None = None,
        settled_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Record the official settlement outcome for a round."""
        sql = """
            UPDATE btc5m_rounds
            SET status = 'resolved', resolved_outcome = ?, resolution_price = ?, settled_at = ?
            WHERE round_slug = ?
        """
        settled_str = settled_at or datetime.now().isoformat()
        params = (resolved_outcome, resolution_price, settled_str, round_slug)
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def get_btc5m_rounds(self, status: str | None = None) -> list[dict[str, Any]]:
        """Retrieve stored BTC 5m rounds, optionally filtered by status."""
        sql = "SELECT * FROM btc5m_rounds"
        params: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY start_epoch DESC"

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def save_btc5m_snapshot(
        self,
        snapshot: BTC5mFeatureSnapshot,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Persist a frozen point-in-time BTC 5m feature snapshot."""
        sql = """
            INSERT OR REPLACE INTO btc5m_snapshots (
                snapshot_id, round_slug, target_horizon_sec, captured_at_ms,
                target_scheduled_ms, timing_deviation_ms, seconds_remaining,
                reference_source, price_to_beat, price_to_beat_source,
                current_reference_price, ref_distance_to_beat_bps, ref_return_10s_bps,
                ref_return_30s_bps, ref_return_60s_bps, market_q, up_best_bid,
                up_best_ask, down_best_bid, down_best_ask, poly_spread,
                binance_perp_mid, binance_microprice, binance_microprice_offset_bps,
                binance_top5_depth_imbalance, binance_top20_depth_imbalance,
                binance_spread_bps, binance_taker_flow_10s_imbalance,
                binance_taker_flow_30s_imbalance, binance_taker_flow_60s_imbalance,
                binance_return_10s_bps, binance_return_30s_bps, binance_return_60s_bps,
                binance_return_since_open_bps, binance_basis_bps, is_valid,
                skip_reason, raw_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """
        params = (
            snapshot.snapshot_id,
            snapshot.round_slug,
            snapshot.target_horizon_sec,
            snapshot.captured_at_ms,
            snapshot.target_scheduled_ms,
            snapshot.timing_deviation_ms,
            snapshot.seconds_remaining,
            snapshot.reference_source,
            snapshot.price_to_beat,
            snapshot.price_to_beat_source,
            snapshot.current_reference_price,
            snapshot.ref_distance_to_beat_bps,
            snapshot.ref_return_10s_bps,
            snapshot.ref_return_30s_bps,
            snapshot.ref_return_60s_bps,
            snapshot.market_q,
            snapshot.up_best_bid,
            snapshot.up_best_ask,
            snapshot.down_best_bid,
            snapshot.down_best_ask,
            snapshot.poly_spread,
            snapshot.binance_perp_mid,
            snapshot.binance_microprice,
            snapshot.binance_microprice_offset_bps,
            snapshot.binance_top5_depth_imbalance,
            snapshot.binance_top20_depth_imbalance,
            snapshot.binance_spread_bps,
            snapshot.binance_taker_flow_10s_imbalance,
            snapshot.binance_taker_flow_30s_imbalance,
            snapshot.binance_taker_flow_60s_imbalance,
            snapshot.binance_return_10s_bps,
            snapshot.binance_return_30s_bps,
            snapshot.binance_return_60s_bps,
            snapshot.binance_return_since_open_bps,
            snapshot.binance_basis_bps,
            1 if snapshot.is_valid else 0,
            snapshot.skip_reason,
            snapshot.to_json(),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def get_btc5m_snapshots(
        self,
        round_slug: str | None = None,
        valid_only: bool = False,
    ) -> list[BTC5mFeatureSnapshot]:
        """Retrieve feature snapshots."""
        sql = "SELECT raw_json FROM btc5m_snapshots"
        clauses: list[str] = []
        params: list[Any] = []
        if round_slug:
            clauses.append("round_slug = ?")
            params.append(round_slug)
        if valid_only:
            clauses.append("is_valid = 1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY captured_at_ms ASC"

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [BTC5mFeatureSnapshot.from_dict(json.loads(r["raw_json"])) for r in rows]

    def save_btc5m_forecast(
        self,
        forecast: BTC5mAblationForecast,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Persist a single ablation forecast."""
        sql = """
            INSERT OR REPLACE INTO btc5m_forecasts (
                forecast_id, snapshot_id, round_slug, target_horizon_sec,
                condition, model_id, request_hash, captured_at_ms,
                market_q, jev_up_prob, jev_down_prob, jev_choice,
                confidence, input_tokens, output_tokens, latency_ms,
                from_cache, raw_response_hash, created_at_utc,
                request_started_at_utc, response_received_at_utc,
                response_received_at_ms, round_end_ms, snapshot_hash,
                request_order, is_valid, rejection_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            forecast.forecast_id,
            forecast.snapshot_id,
            forecast.round_slug,
            forecast.target_horizon_sec,
            forecast.condition,
            forecast.model_id,
            forecast.request_hash,
            forecast.captured_at_ms,
            forecast.market_q,
            forecast.jev_up_prob,
            forecast.jev_down_prob,
            forecast.jev_choice,
            forecast.confidence,
            forecast.input_tokens,
            forecast.output_tokens,
            forecast.latency_ms,
            1 if forecast.from_cache else 0,
            forecast.raw_response_hash,
            forecast.created_at_utc,
            forecast.request_started_at_utc,
            forecast.response_received_at_utc,
            forecast.response_received_at_ms,
            forecast.round_end_ms,
            forecast.snapshot_hash,
            forecast.request_order,
            1 if forecast.is_valid else 0,
            forecast.rejection_reason,
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def get_btc5m_forecasts(
        self,
        round_slug: str | None = None,
        condition: str | None = None,
        valid_only: bool = False,
    ) -> list[BTC5mAblationForecast]:
        """Retrieve ablation forecasts."""
        sql = "SELECT * FROM btc5m_forecasts"
        clauses: list[str] = []
        params: list[Any] = []
        if round_slug:
            clauses.append("round_slug = ?")
            params.append(round_slug)
        if condition:
            clauses.append("condition = ?")
            params.append(condition)
        if valid_only:
            clauses.append("is_valid = 1")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY captured_at_ms ASC"

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                BTC5mAblationForecast(
                    forecast_id=r["forecast_id"],
                    snapshot_id=r["snapshot_id"],
                    round_slug=r["round_slug"],
                    target_horizon_sec=int(r["target_horizon_sec"]),
                    condition=r["condition"],
                    model_id=r["model_id"],
                    request_hash=r["request_hash"],
                    captured_at_ms=int(r["captured_at_ms"]),
                    market_q=float(r["market_q"]) if r["market_q"] is not None else None,
                    jev_up_prob=float(r["jev_up_prob"]),
                    jev_down_prob=float(r["jev_down_prob"]),
                    jev_choice=r["jev_choice"],
                    confidence=float(r["confidence"]) if r["confidence"] is not None else None,
                    input_tokens=int(r["input_tokens"]),
                    output_tokens=int(r["output_tokens"]),
                    latency_ms=int(r["latency_ms"]),
                    from_cache=bool(r["from_cache"]),
                    raw_response_hash=r["raw_response_hash"],
                    created_at_utc=r["created_at_utc"],
                    request_started_at_utc=r["request_started_at_utc"] if "request_started_at_utc" in r.keys() else None,
                    response_received_at_utc=r["response_received_at_utc"] if "response_received_at_utc" in r.keys() else None,
                    response_received_at_ms=int(r["response_received_at_ms"]) if ("response_received_at_ms" in r.keys() and r["response_received_at_ms"] is not None) else None,
                    round_end_ms=int(r["round_end_ms"]) if ("round_end_ms" in r.keys() and r["round_end_ms"] is not None) else None,
                    snapshot_hash=r["snapshot_hash"] if "snapshot_hash" in r.keys() else None,
                    request_order=int(r["request_order"]) if "request_order" in r.keys() else 1,
                    is_valid=bool(r["is_valid"]) if "is_valid" in r.keys() else True,
                    rejection_reason=r["rejection_reason"] if "rejection_reason" in r.keys() else None,
                )
                for r in rows
            ]

    def save_btc5m_resolution_score(
        self,
        score: dict[str, Any],
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Persist comparative score record for a resolved round."""
        sql = """
            INSERT OR REPLACE INTO btc5m_resolution_scores (
                score_id, round_slug, target_horizon_sec, snapshot_id,
                resolved_outcome, resolution_price, resolved_at, scored_at,
                market_q, market_brier, market_log_loss,
                cond_a_prob, cond_a_brier, cond_a_log_loss,
                cond_b_prob, cond_b_brier, cond_b_log_loss,
                cond_c_prob, cond_c_brier, cond_c_log_loss,
                cond_d_prob, cond_d_brier, cond_d_log_loss,
                metadata_json
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """
        params = (
            score["score_id"],
            score["round_slug"],
            score["target_horizon_sec"],
            score["snapshot_id"],
            score["resolved_outcome"],
            score.get("resolution_price"),
            score["resolved_at"],
            score["scored_at"],
            score.get("market_q"),
            score.get("market_brier"),
            score.get("market_log_loss"),
            score.get("cond_a_prob"),
            score.get("cond_a_brier"),
            score.get("cond_a_log_loss"),
            score.get("cond_b_prob"),
            score.get("cond_b_brier"),
            score.get("cond_b_log_loss"),
            score.get("cond_c_prob"),
            score.get("cond_c_brier"),
            score.get("cond_c_log_loss"),
            score.get("cond_d_prob"),
            score.get("cond_d_brier"),
            score.get("cond_d_log_loss"),
            json.dumps(score.get("metadata", {})),
        )
        if conn is not None:
            conn.execute(sql, params)
        else:
            with self._get_connection() as c:
                c.execute(sql, params)

    def get_btc5m_resolution_scores(
        self,
        round_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve evaluated resolution scores."""
        sql = "SELECT * FROM btc5m_resolution_scores"
        params: list[Any] = []
        if round_slug:
            sql += " WHERE round_slug = ?"
            params.append(round_slug)
        sql += " ORDER BY scored_at DESC"

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]


