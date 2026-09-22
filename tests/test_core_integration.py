"""Core integration test for Milestone 1: Rigo -> Ilsa -> Kett -> Bram -> PaperBroker -> Tess."""

from datetime import datetime, timezone

from pm_research.config import SystemConfig
from pm_research.data.synthetic import get_deterministic_synthetic_markets
from pm_research.domain.models import RiskState, Side
from pm_research.pipeline.runner import PipelineRunner
from pm_research.storage.db import Database


def test_core_pipeline_milestone_1(tmp_path):
    """Test the complete core end-to-end research pipeline on deterministic synthetic data."""
    db_file = tmp_path / "test_research.db"
    db = Database(db_file)
    config = SystemConfig(
        initial_bankroll=1000.0,
        min_robust_edge=0.02,
        model_version="ilsa-test-v1.0",
    )

    runner = PipelineRunner(config=config, db=db)
    raw_markets = get_deterministic_synthetic_markets()

    # Run cycle 1
    summary = runner.run_cycle(raw_markets=raw_markets, notes="Milestone 1 Test Run")

    # 1. Assertions on ingestion
    assert summary["snapshots_count"] > 0
    # Crossed quote market (mkt_kappa_crossed) must have been rejected by Rigo
    assert summary["snapshots_count"] == len(raw_markets) - 1

    # 2. Assertions on estimates & proposals
    assert summary["estimates_count"] == summary["snapshots_count"]
    assert summary["proposals_count"] >= 2

    # 3. Assertions on Bram decisions
    assert summary["accepted_count"] >= 2
    assert summary["rejected_count"] >= 1

    # Check reason codes in database
    with db._get_connection() as conn:
        rejections = conn.execute("SELECT market_id, reason_code FROM risk_decisions WHERE accepted = 0").fetchall()
        reject_codes = {r["market_id"]: r["reason_code"] for r in rejections}

        # Check expected rejections
        assert reject_codes.get("mkt_gamma_low_liq") == "REJECT_LOW_LIQUIDITY"
        assert reject_codes.get("mkt_delta_wide_spread") == "REJECT_HIGH_SPREAD"
        assert reject_codes.get("mkt_epsilon_expiring") == "REJECT_EXPIRING_SOON"
        assert reject_codes.get("mkt_eta_closed") == "REJECT_MARKET_NOT_ACTIVE"

    # 4. Assertions on PaperBroker execution & Tess portfolio
    assert summary["orders_count"] == summary["accepted_count"]
    assert summary["fills_count"] == summary["accepted_count"]
    assert summary["virtual_cash"] < 1000.0  # Cash was spent on contracts + fees
    assert summary["open_positions_count"] == summary["fills_count"]
    assert summary["risk_state"] == RiskState.NORMAL.value

    # 5. Check order book depth walking for mkt_theta_depth
    with db._get_connection() as conn:
        fill_theta = conn.execute(
            "SELECT * FROM paper_fills WHERE market_id = 'mkt_theta_depth'"
        ).fetchone()
        if fill_theta:
            # Slippage must have been recorded
            assert fill_theta["slippage"] >= 0.0
            assert fill_theta["filled_quantity"] > 0.0
            assert fill_theta["average_fill_price"] > 0.0

    # 6. Settle/Resolve a market (e.g., mkt_alpha_tech resolves to YES)
    res_time = datetime.now(timezone.utc)
    observations = runner.tess.resolve_market(
        market_id="mkt_alpha_tech",
        resolved_outcome=Side.YES,
        cycle_id=summary["cycle_id"],
        resolved_at=res_time,
    )

    assert len(observations) >= 1
    obs = observations[0]
    assert obs.market_id == "mkt_alpha_tech"
    assert obs.actual_outcome == 1.0
    assert 0.0 <= obs.brier_score <= 1.0
    assert obs.log_loss >= 0.0
    assert obs.model_version == "ilsa-test-v1.0"

    # Verify positions in Tess: alpha tech is closed, realized PnL recorded
    all_positions = db.get_all_positions()
    alpha_pos = [p for p in all_positions if p.market_id == "mkt_alpha_tech"][0]
    assert alpha_pos.status == "CLOSED"
    assert alpha_pos.resolved_outcome == Side.YES
    assert alpha_pos.realized_pnl > 0.0  # We bought YES at ~0.40 and it paid out 1.0

    # Verify audit persistence: check cycle in DB
    cycle_count = db.get_cycle_count()
    assert cycle_count == 1
