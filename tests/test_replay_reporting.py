"""Unit tests for replay reporting generation, simulation notices, and diagnostics."""

from __future__ import annotations

from pathlib import Path

from pm_research.data.synthetic_replay import create_deterministic_synthetic_replay_dataset
from pm_research.replay.engine import ReplayEngine
from pm_research.replay.models import ReplayConfig
from pm_research.replay.report import ReplayReportGenerator


def test_replay_terminal_report_contains_safety_disclaimer() -> None:
    """Terminal report must display prominent simulation and paper-only disclaimer banners."""
    ds = create_deterministic_synthetic_replay_dataset()
    cfg = ReplayConfig(dataset_path="", latency_seconds=0.0)
    eng = ReplayEngine(config=cfg, dataset=ds)
    res = eng.run()

    report_str = ReplayReportGenerator.format_terminal_report(res)

    # Must contain unambiguous research simulation notice
    assert "RESEARCH SIMULATION NOTICE: PAPER-TRADING AND HISTORICAL REPLAY ONLY" in report_str
    assert "THIS IS NOT LIVE EXECUTION" in report_str
    assert "Brier Score:" in report_str
    assert "Expected Calib. Error:" in report_str
    assert "Total Simulated P&L:" in report_str


def test_replay_html_report_generation(tmp_path: Path) -> None:
    """HTML report is generated with valid structure, safety banners, and metric tables."""
    ds = create_deterministic_synthetic_replay_dataset()
    cfg = ReplayConfig(dataset_path="", latency_seconds=0.0)
    eng = ReplayEngine(config=cfg, dataset=ds)
    res = eng.run()

    html_path = tmp_path / "test_report.html"
    out = ReplayReportGenerator.generate_html_report(res, html_path)

    assert out.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "RESEARCH SIMULATION NOTICE" in content
    assert res.replay_id in content
    assert str(res.calibration.brier_score) in content
    assert f"${res.final_equity:,.2f}" in content
