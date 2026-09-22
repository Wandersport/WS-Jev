"""Full pipeline integration test: Rigo -> Holt -> Ilsa -> Kett -> Bram -> PaperBroker -> Tess -> Reporting."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from pm_research.config import SystemConfig
from pm_research.data.synthetic import get_deterministic_synthetic_markets
from pm_research.domain.models import RiskState, Side
from pm_research.pipeline.runner import PipelineRunner
from pm_research.reporting.report import ReportGenerator
from pm_research.storage.db import Database
from pm_research.utils import now_utc


def test_full_research_pipeline_lifecycle(tmp_path):
    """Verify complete quantitative research lifecycle:

    Multi-cycle ingestion, feature extraction, estimation, sizing, risk gating,
    order execution, portfolio marking, market resolution, calibration computation,
    and report dashboard generation.
    """
    db_file = tmp_path / "full_pipeline.db"
    db = Database(db_file)
    config = SystemConfig(
        initial_bankroll=1000.0,
        min_robust_edge=0.02,
        model_version="ilsa-full-v1.0",
    )
    runner = PipelineRunner(config=config, db=db)

    t0 = now_utc() - timedelta(days=5)

    # --- CYCLE 1 ---
    raw_markets = get_deterministic_synthetic_markets(base_time=t0)
    summary_c1 = runner.run_cycle(raw_markets, cycle_time=t0, notes="Cycle 1: Baseline")

    assert summary_c1["snapshots_count"] == 8
    assert summary_c1["estimates_count"] == 8
    assert summary_c1["proposals_count"] >= 2
    assert summary_c1["accepted_count"] >= 2
    assert summary_c1["fills_count"] >= 2
    assert summary_c1["virtual_cash"] < 1000.0
    assert summary_c1["risk_state"] == RiskState.NORMAL.value

    # Verify features extracted and stored in DB
    with db._get_connection() as conn:
        feat_rows = conn.execute("SELECT * FROM research_features").fetchall()
        assert len(feat_rows) == 8

    # --- RESOLUTIONS ---
    t_res = t0 + timedelta(days=2)
    obs_alpha = runner.tess.resolve_market("mkt_alpha_tech", Side.YES, cycle_id=summary_c1["cycle_id"], resolved_at=t_res)
    obs_beta = runner.tess.resolve_market("mkt_beta_macro", Side.NO, cycle_id=summary_c1["cycle_id"], resolved_at=t_res)

    assert len(obs_alpha) >= 1
    assert len(obs_beta) >= 1

    # --- CYCLE 2 ---
    t1 = t0 + timedelta(days=3)
    summary_c2 = runner.run_cycle(raw_markets, cycle_time=t1, notes="Cycle 2: Post-Resolution")

    assert summary_c2["portfolio_equity"] > 0
    assert summary_c2["open_positions_count"] >= 1  # mkt_theta_depth still open

    # --- REPORTING ---
    rep_gen = ReportGenerator(config, db)
    cli_rep = rep_gen.generate_cli_report()
    assert "PAPER TRADING / SIMULATION ONLY" in cli_rep
    assert "Brier Score" in cli_rep

    html_out = tmp_path / "dashboard.html"
    dashboard_path = rep_gen.generate_html_dashboard(str(html_out))
    assert Path(dashboard_path).exists()
    content = Path(dashboard_path).read_text(encoding="utf-8")
    assert "PAPER TRADING / SIMULATION ONLY" in content
    assert "Probability Calibration Buckets" in content
