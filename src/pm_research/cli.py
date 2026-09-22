"""Command-line interface for the paper prediction market research system.

THIS SYSTEM IS STRUCTURALLY INCAPABLE OF LIVE TRADING.
Available subcommands:
  seed-demo      Seed deterministic synthetic scenario with resolutions & calibration
  run-once       Execute a single research pipeline cycle
  run-loop       Execute multiple research cycles
  portfolio      Display current paper portfolio state
  calibration    Display forecast calibration analytics
  report         Generate CLI and HTML dashboard report
  verify-safety  Run automated safety checks proving no live trading capability
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

from pm_research.calibration.metrics import CalibrationEngine
from pm_research.config import SystemConfig
from pm_research.data.public_adapter import PublicMarketDataAdapter
from pm_research.data.synthetic import get_deterministic_synthetic_markets
from pm_research.domain.models import Side
from pm_research.pipeline.runner import PipelineRunner
from pm_research.reporting.report import ReportGenerator
from pm_research.safety.verifier import SafetyVerifier
from pm_research.storage.db import Database
from pm_research.utils import parse_iso_utc


def get_db(db_path: str | None = None) -> Database:
    path = db_path or "data/pm_research.db"
    return Database(path)


def cmd_seed_demo(args: argparse.Namespace) -> int:
    """Seed a rich deterministic synthetic demonstration with orders, fills, resolutions, and calibration."""
    db = get_db(args.db)
    if not getattr(args, "no_reset", False):
        print("    [!] Resetting database for clean deterministic execution...")
        db.reset_database()

    config = SystemConfig(min_robust_edge=0.02)
    runner = PipelineRunner(config=config, db=db)

    print("\n[+] Seeding deterministic synthetic demo...")
    if getattr(args, "base_time", None):
        base_t = parse_iso_utc(args.base_time)
    else:
        base_t = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Cycle 1: Ingestion & Paper Trading
    raw_markets = get_deterministic_synthetic_markets(base_time=base_t)
    summary_1 = runner.run_cycle(
        raw_markets=raw_markets,
        cycle_time=base_t,
        notes="Demo Cycle 1: Initial Discovery & Paper Sizing",
    )
    print(f"    Cycle 1 executed: {summary_1['snapshots_count']} markets ingested, "
          f"{summary_1['accepted_count']} accepted, {summary_1['rejected_count']} rejected, "
          f"{summary_1['fills_count']} paper fills.")

    # 2. Market Resolutions: mkt_alpha_tech resolves to YES, mkt_beta_macro resolves to NO
    print("\n[+] Simulating market resolutions and calibration recording...")
    t_res1 = base_t + timedelta(days=3)
    obs_alpha = runner.tess.resolve_market(
        market_id="mkt_alpha_tech",
        resolved_outcome=Side.YES,
        cycle_id=summary_1["cycle_id"],
        resolved_at=t_res1,
    )
    print(f"    Market 'mkt_alpha_tech' resolved to YES -> {len(obs_alpha)} calibration record(s).")

    obs_beta = runner.tess.resolve_market(
        market_id="mkt_beta_macro",
        resolved_outcome=Side.NO,
        cycle_id=summary_1["cycle_id"],
        resolved_at=t_res1,
    )
    print(f"    Market 'mkt_beta_macro' resolved to NO -> {len(obs_beta)} calibration record(s).")

    # 3. Cycle 2: Follow-up cycle marking open positions and evaluating updated markets
    t_cyc2 = base_t + timedelta(days=4)
    summary_2 = runner.run_cycle(
        raw_markets=raw_markets,
        cycle_time=t_cyc2,
        notes="Demo Cycle 2: Post-resolution re-marking and risk check",
    )
    print(f"    Cycle 2 executed: Portfolio equity=${summary_2['portfolio_equity']:,.2f}, "
          f"drawdown={summary_2['current_drawdown']*100:.2f}%, state={summary_2['risk_state']}.")

    # 4. Generate Reports
    rep_gen = ReportGenerator(config, db)
    html_path = rep_gen.generate_html_dashboard(args.dashboard_out)
    print(f"\n[+] Static HTML dashboard generated at: {html_path}")
    print(rep_gen.generate_cli_report())

    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    """Execute a single paper research cycle."""
    db = get_db(args.db)
    config = SystemConfig()
    runner = PipelineRunner(config=config, db=db)

    if args.public:
        print("[*] Ingesting public read-only market data (unauthenticated)...")
        adapter = PublicMarketDataAdapter()
        raw_markets = adapter.fetch_public_markets(limit=20)
        if not raw_markets:
            print("[!] Public market data unavailable or empty; falling back to synthetic fixtures.")
            raw_markets = get_deterministic_synthetic_markets()
    else:
        raw_markets = get_deterministic_synthetic_markets()

    summary = runner.run_cycle(raw_markets=raw_markets, notes="Manual run-once cycle")

    print(f"\nCycle Completed: {summary['cycle_id']}")
    print(f"  Ingested Snapshots:   {summary['snapshots_count']}")
    print(f"  Probability Estimates:{summary['estimates_count']}")
    print(f"  Trade Proposals:      {summary['proposals_count']}")
    print(f"  Bram Accepted:        {summary['accepted_count']}")
    print(f"  Bram Rejected:        {summary['rejected_count']}")
    print(f"  Paper Fills:          {summary['fills_count']}")
    print(f"  Virtual Cash:         ${summary['virtual_cash']:,.2f}")
    print(f"  Portfolio Equity:     ${summary['portfolio_equity']:,.2f}")
    print(f"  Current Drawdown:     {summary['current_drawdown'] * 100:.2f}%")
    print(f"  Risk State:           {summary['risk_state']}")

    return 0


def cmd_run_loop(args: argparse.Namespace) -> int:
    """Execute a loop of paper research cycles."""
    db = get_db(args.db)
    config = SystemConfig()
    runner = PipelineRunner(config=config, db=db)

    print(f"\n[*] Starting research loop: {args.cycles} cycles, {args.interval}s interval...")
    for i in range(1, args.cycles + 1):
        raw_markets = get_deterministic_synthetic_markets()
        summary = runner.run_cycle(raw_markets=raw_markets, notes=f"Loop Cycle {i}/{args.cycles}")
        print(f"  [Cycle {i}/{args.cycles}] Fills: {summary['fills_count']}, "
              f"Equity: ${summary['portfolio_equity']:,.2f}, Drawdown: {summary['current_drawdown']*100:.2f}%")
        if i < args.cycles:
            time.sleep(args.interval)

    print("[+] Loop completed.")
    return 0


def cmd_portfolio(args: argparse.Namespace) -> int:
    """Print current paper portfolio state."""
    db = get_db(args.db)
    config = SystemConfig()
    rep_gen = ReportGenerator(config, db)
    print(rep_gen.generate_cli_report())
    return 0


def cmd_calibration(args: argparse.Namespace) -> int:
    """Print forecast calibration metrics."""
    db = get_db(args.db)
    engine = CalibrationEngine()
    obs = db.get_all_calibration_observations()
    rep = engine.compute_metrics(obs)

    print("\n" + "=" * 65)
    print("  FORECAST CALIBRATION ANALYSIS")
    print("=" * 65)
    print(f"  Observations:     {rep.sample_size}")
    if rep.sample_size == 0:
        print("  (No resolved market observations recorded yet)")
        print("=" * 65 + "\n")
        return 0

    print(f"  Brier Score:      {rep.brier_score:.4f} (0.0 = perfect calibration)")
    print(f"  Log Loss:         {rep.log_loss:.4f}")
    print(f"  Forecast Bias:    {rep.forecast_bias:+.4f}")
    print(f"  ECE:              {rep.expected_calibration_error:.4f}")
    print(f"  MCE:              {rep.maximum_calibration_error:.4f}")

    print("\n  Calibration Buckets:")
    print("  -------------------------------------------------------------")
    print("  Range        | Count | Mean Predicted | Empirical Rate | Error")
    print("  -------------------------------------------------------------")
    for b in rep.buckets:
        if b.count > 0:
            print(f"  [{b.bin_lower:.2f}, {b.bin_upper:.2f}) | {b.count:5d} | {b.mean_predicted:14.4f} | {b.empirical_frequency:14.4f} | {b.bin_error:6.4f}")
    print("  -------------------------------------------------------------\n")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate text and HTML reports."""
    db = get_db(args.db)
    config = SystemConfig()
    rep_gen = ReportGenerator(config, db)

    html_path = rep_gen.generate_html_dashboard(args.dashboard_out)
    print(rep_gen.generate_cli_report())
    print(f"[+] Static HTML dashboard written to: {html_path}\n")
    return 0


def cmd_verify_safety(args: argparse.Namespace) -> int:
    """Execute automated safety scans verifying structural paper-only compliance."""
    print("\n" + "=" * 70)
    print("  AUTOMATED SAFETY VERIFICATION: PAPER-ONLY ENFORCEMENT")
    print("=" * 70)

    verifier = SafetyVerifier()
    res = verifier.verify_all()

    print(f"\nScanned {res.scanned_files_count} source files across repository.\n")
    print("Rules Checked:")
    for r in res.checked_rules:
        print(f"  [OK] {r}")

    if res.passed:
        print("\n" + "=" * 70)
        print("  SAFETY VERIFICATION RESULT: PASS")
        print("  - Live trading capability: ZERO")
        print("  - Wallets / Private Keys: ZERO")
        print("  - Live execution stubs/ABCs: ZERO")
        print("  - Sole execution module: PaperBroker")
        print("=" * 70 + "\n")
        return 0
    else:
        print("\n" + "!" * 70)
        print("  SAFETY VERIFICATION RESULT: FAIL")
        for v in res.violations:
            print(f"  [VIOLATION] {v}")
        print("!" * 70 + "\n")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pmr",
        description="Paper-trading-only prediction market quantitative research pipeline.",
        epilog="SIMULATION ONLY - STRUCTURALLY INCAPABLE OF LIVE TRADING.",
    )
    parser.add_argument("--db", default="data/pm_research.db", help="Path to SQLite database")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # seed-demo
    p_seed = subparsers.add_parser("seed-demo", help="Seed deterministic synthetic demonstration")
    p_seed.add_argument("--dashboard-out", default="reports/dashboard.html", help="Path for HTML dashboard")
    p_seed.add_argument("--no-reset", action="store_true", help="Do not reset database before seeding demo")
    p_seed.add_argument("--base-time", default=None, help="Base time for demo in ISO UTC format")

    # run-once
    p_run = subparsers.add_parser("run-once", help="Execute single paper research cycle")
    p_run.add_argument("--public", action="store_true", help="Fetch public read-only markets (unauthenticated)")

    # run-loop
    p_loop = subparsers.add_parser("run-loop", help="Execute loop of paper research cycles")
    p_loop.add_argument("--cycles", type=int, default=3, help="Number of cycles to run")
    p_loop.add_argument("--interval", type=float, default=1.0, help="Interval in seconds between cycles")

    # portfolio
    subparsers.add_parser("portfolio", help="Show current paper portfolio")

    # calibration
    subparsers.add_parser("calibration", help="Show forecast calibration metrics")

    # report
    p_rep = subparsers.add_parser("report", help="Generate report and HTML dashboard")
    p_rep.add_argument("--dashboard-out", default="reports/dashboard.html", help="Path for HTML dashboard")

    # verify-safety
    subparsers.add_parser("verify-safety", help="Run automated paper-only safety checks")

    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    dispatch = {
        "seed-demo": cmd_seed_demo,
        "run-once": cmd_run_once,
        "run-loop": cmd_run_loop,
        "portfolio": cmd_portfolio,
        "calibration": cmd_calibration,
        "report": cmd_report,
        "verify-safety": cmd_verify_safety,
    }

    handler = dispatch.get(args.subcommand)
    if handler:
        return handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
