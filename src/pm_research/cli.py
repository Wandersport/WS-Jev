"""Command-line interface for the paper prediction market research system.

THIS SYSTEM IS STRUCTURALLY INCAPABLE OF LIVE TRADING.
Available subcommands:
  seed-demo         Seed deterministic synthetic scenario with resolutions & calibration
  run-once          Execute a single research pipeline cycle
  run-loop          Execute multiple research cycles
  replay            Execute historical replay simulation
  replay-report     Inspect a recorded historical replay run
  datasets          List registered historical replay datasets
  history-probe     Probe public read-only market data endpoints
  history-collect   Collect and normalize historical resolved market datasets
  history-validate  Benchmark probability model against historical market baseline
  portfolio         Display current paper portfolio state
  calibration       Display forecast calibration analytics
  report            Generate CLI and HTML dashboard report
  verify-safety     Run automated safety checks proving no live trading capability
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pm_research.calibration.market_baseline import StandardizedHorizonEvaluator
from pm_research.calibration.metrics import CalibrationEngine
from pm_research.config import SystemConfig
from pm_research.data.historical_collector import HistoricalCollector
from pm_research.data.public_adapter import PublicMarketDataAdapter
from pm_research.data.synthetic import get_deterministic_synthetic_markets
from pm_research.domain.models import Side
from pm_research.pipeline.runner import PipelineRunner
from pm_research.replay.dataset import DatasetManager, ReplayDataset
from pm_research.replay.engine import ReplayEngine
from pm_research.replay.models import ReplayConfig
from pm_research.replay.report import ReplayReportGenerator
from pm_research.reporting.history_report import HistoryReportGenerator
from pm_research.reporting.report import ReportGenerator
from pm_research.safety.verifier import SafetyVerifier
from pm_research.storage.db import Database
from pm_research.utils import parse_iso_utc, to_iso_utc


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


def cmd_replay(args: argparse.Namespace) -> int:
    """Execute a historical replay simulation."""
    import json
    sim_db = Database(args.db) if args.db else Database(":memory:")
    cfg = ReplayConfig(
        dataset_path=args.dataset,
        latency_seconds=float(args.latency),
        initial_bankroll=float(args.bankroll),
        random_seed=int(args.seed),
    )
    engine = ReplayEngine(config=cfg, db=sim_db)
    result = engine.run()

    # Also persist run record to master database for replay-report retrieval
    try:
        master_db = Database("data/pm_research.db")
        master_db.save_replay_run(
            replay_id=result.replay_id,
            dataset_id=result.dataset_id,
            created_at=to_iso_utc(result.finished_at),
            config_json=json.dumps({
                "dataset_path": str(cfg.dataset_path),
                "latency_seconds": cfg.latency_seconds,
                "initial_bankroll": cfg.initial_bankroll,
                "random_seed": cfg.random_seed,
            }),
            result_json=json.dumps(result.to_dict()),
        )
    except Exception:
        pass

    report_str = ReplayReportGenerator.format_terminal_report(result)
    print(report_str)

    if args.html_out:
        html_p = ReplayReportGenerator.generate_html_report(result, args.html_out)
        print(f"[+] HTML replay report saved to: {html_p}\n")
    return 0


def cmd_replay_report(args: argparse.Namespace) -> int:
    """Display or export a previously executed replay run."""
    import json
    db = get_db(args.db)
    run_record = db.get_replay_run(args.replay_id)
    if not run_record:
        print(f"[!] Replay run '{args.replay_id}' not found in database.")
        runs = db.list_replay_runs()
        if runs:
            print("Available replay runs:")
            for r in runs:
                print(f"  - {r['replay_id']} ({r['dataset_id']} at {r['created_at']})")
        return 1

    res_dict = json.loads(run_record["result_json"])
    print(f"\nReplay Run: {run_record['replay_id']} (Dataset: {run_record['dataset_id']})")
    print(f"Created: {run_record['created_at']}")
    calib = res_dict.get("calibration", {})
    port = res_dict.get("portfolio", {})
    trace = res_dict.get("traceability", {})
    print(f"Brier Score: {calib.get('brier_score')} | Log Loss: {calib.get('log_loss')} | Bias: {calib.get('forecast_bias')}")
    print(f"Initial: ${port.get('initial_bankroll')} | Final Equity: ${port.get('final_equity')} | Return: {port.get('simulated_return_pct')}%")
    print(f"Max Drawdown: {port.get('max_drawdown_pct')}% | Latency: {trace.get('latency_seconds')}s | Fills: {port.get('fills_count')}\n")
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    """List all registered historical replay datasets and verify checksums."""
    manager = DatasetManager()
    manifests = manager.list_datasets()
    print("\n" + "=" * 80)
    print("  REGISTERED HISTORICAL REPLAY DATASETS")
    print("=" * 80)
    if not manifests:
        print("  No registered datasets found in data/datasets/.")
        print("=" * 80 + "\n")
        return 0

    for m in manifests:
        print(f"  Dataset ID:       {m.dataset_id}")
        print(f"    Name:           {m.name}")
        print(f"    Source:         {m.source} (Synthetic: {m.is_synthetic})")
        print(f"    Time Span:      {to_iso_utc(m.start_time)} to {to_iso_utc(m.end_time)}")
        print(f"    Markets:        {m.market_count} | Snapshots: {m.snapshot_count} | Resolutions: {m.resolution_count}")
        print(f"    SHA256:         {m.checksum_sha256[:16]}...{m.checksum_sha256[-8:] if m.checksum_sha256 else ''}")
        print("  " + "-" * 76)
    print("=" * 80 + "\n")
def cmd_history_probe(args: argparse.Namespace) -> int:
    """Probe public unauthenticated Gamma and CLOB API endpoints and print diagnostics."""
    print("\n" + "=" * 80)
    print("  [!] PUBLIC READ-ONLY MARKET DATA PROBE")
    print("=" * 80)
    print("  Testing unauthenticated read-only GET requests to allowlisted hosts...")

    collector = HistoricalCollector()
    t0 = time.monotonic()
    markets = collector.discover_resolved_markets(max_candidates=3)
    elapsed_gamma = time.monotonic() - t0

    print(f"\n  [1] Gamma API Probe: {collector.gamma_base}/markets?closed=true")
    print(f"      Latency: {elapsed_gamma:.3f}s")
    print(f"      Markets Returned: {len(markets)}")

    if not markets:
        print("      [!] Failed to retrieve sample markets from Gamma API.")
        return 1

    sample = markets[0]
    print(f"      Sample Market ID: {sample.get('id')}")
    print(f"      Question: {sample.get('question')}")
    print(f"      Outcomes: {sample.get('outcomes')}")
    print(f"      Outcome Prices: {sample.get('outcomePrices')}")
    print(f"      Closed Time: {sample.get('closedTime') or sample.get('endDate')}")

    tokens = sample.get("clobTokenIds")
    if isinstance(tokens, str):
        try:
            tokens = json.loads(tokens)
        except Exception:
            tokens = []

    if tokens and len(tokens) > 0:
        token_id = str(tokens[0])
        print(f"\n  [2] CLOB API Probe: {collector.clob_base}/prices-history")
        print(f"      Testing Token ID: {token_id[:16]}...")
        t0 = time.monotonic()
        closed_str = sample.get("closedTime") or sample.get("endDate")
        try:
            end_ts = int(parse_iso_utc(str(closed_str)).timestamp())
        except Exception:
            end_ts = int(time.time())
        pts = collector.fetch_token_prices_history(
            token_id=token_id, end_ts=end_ts, days_back=7, fidelity=60
        )
        elapsed_clob = time.monotonic() - t0
        print(f"      Latency: {elapsed_clob:.3f}s")
        print(f"      Price Points Returned: {len(pts)}")
        if pts:
            print(f"      First Point: ts={pts[0].get('t')} price={pts[0].get('p')}")
            print(f"      Last Point:  ts={pts[-1].get('t')} price={pts[-1].get('p')}")
    else:
        print("\n  [2] CLOB API Probe: Skipped (no token ID on sample market)")

    print("\n" + "=" * 80)
    print("  PROBE COMPLETE - READ-ONLY NETWORKING OPERATIONAL")
    print("=" * 80 + "\n")
    return 0


def cmd_history_collect(args: argparse.Namespace) -> int:
    """Download, filter, and normalize historical prediction market data into a ReplayDataset."""
    print("\n" + "=" * 80)
    print("  [!] HISTORICAL RESOLVED MARKET COLLECTOR")
    print("      Strictly public read-only unauthenticated endpoints")
    print("=" * 80)

    collector = HistoricalCollector()
    target_dir = Path(args.out)

    print(f"  Target Directory:  {target_dir}")
    print(f"  Max Markets:       {args.max_markets}")
    print(f"  Days History:      {args.days}")
    print(f"  Start Date:        {args.start_date}")
    print(f"  End Date:          {args.end_date}")
    print(f"  Min Volume:        ${args.min_volume:,.0f}")
    print("\n  Collecting markets (this may take a few moments)...")

    dataset, report = collector.build_dataset(
        dataset_id=args.dataset_id,
        name=args.dataset_name,
        target_dir=target_dir,
        max_markets=args.max_markets,
        start_date=args.start_date,
        end_date=args.end_date,
        min_volume=args.min_volume,
        days_back=args.days,
    )

    print("\n" + "-" * 80)
    print("  COLLECTION & QUALITY REPORT")
    print("-" * 80)
    print(f"  Markets Discovered:        {report.markets_discovered}")
    print(f"  Markets Included:          {report.markets_included}")
    print(f"  Markets Excluded:          {report.markets_excluded}")
    print("  Exclusion Breakdown:")
    for reason, count in sorted(report.exclusion_reasons.items()):
        print(f"    - {reason:<28}: {count}")
    print(f"  Total Snapshots Packaged:  {len(dataset.snapshots)}")
    print(f"  Total Resolutions:         {len(dataset.resolutions)}")
    print(f"  Date Range:                {report.earliest_resolution} to {report.latest_resolution}")
    print(f"  Dataset SHA256:            {dataset.manifest.checksum_sha256}")
    print("=" * 80 + "\n")
    return 0


def cmd_history_validate(args: argparse.Namespace) -> int:
    """Benchmark the deterministic probability model against historical market baseline."""
    manager = DatasetManager()
    dataset = manager.get_dataset(args.dataset)
    if dataset is None:
        p = Path(args.dataset)
        if p.exists() and (p / "manifest.json").exists():
            dataset = ReplayDataset.load(p)
        else:
            print(f"Error: Dataset '{args.dataset}' not found.")
            return 1

    evaluator = StandardizedHorizonEvaluator(
        bootstrap_samples=args.bootstrap_samples,
    )
    report = evaluator.evaluate_dataset(dataset)
    rep_gen = HistoryReportGenerator(report)

    # CLI report
    cli_text = rep_gen.generate_cli_report()
    print(cli_text)

    # HTML report if requested
    if args.html_out:
        out_path = rep_gen.generate_html_report(args.html_out)
        print(f"\n[+] Standalone HTML report generated: {out_path}\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pmr",
        description="Paper-trading-only prediction market quantitative research pipeline.",
        epilog="SIMULATION ONLY - STRUCTURALLY INCAPABLE OF LIVE TRADING.",
    )
    parser.add_argument("--db", default=None, help="Path to SQLite database (defaults to data/pm_research.db)")

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

    # replay
    p_replay = subparsers.add_parser("replay", help="Execute historical replay simulation")
    p_replay.add_argument("--dataset", default="synthetic_benchmark_v1", help="Dataset ID or directory path")
    p_replay.add_argument("--latency", type=float, default=0.0, help="Simulated execution latency in seconds")
    p_replay.add_argument("--bankroll", type=float, default=1000.0, help="Initial virtual capital")
    p_replay.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    p_replay.add_argument("--html-out", default=None, help="Path for HTML replay report output")

    # replay-report
    p_rep_report = subparsers.add_parser("replay-report", help="Inspect a recorded historical replay run")
    p_rep_report.add_argument("--replay-id", required=True, help="Replay ID to inspect")

    # datasets
    subparsers.add_parser("datasets", help="List registered historical replay datasets")

    # history-probe
    subparsers.add_parser("history-probe", help="Probe public read-only market data endpoints")

    # history-collect
    p_hcollect = subparsers.add_parser(
        "history-collect", help="Collect and normalize historical resolved market datasets"
    )
    p_hcollect.add_argument("--max-markets", type=int, default=50, help="Maximum markets to package")
    p_hcollect.add_argument("--days", type=int, default=30, help="Days of price history prior to resolution")
    p_hcollect.add_argument("--out", default="data/datasets/polymarket_resolved_v1", help="Output directory")
    p_hcollect.add_argument("--dataset-id", default="polymarket_resolved_v1", help="Dataset identifier")
    p_hcollect.add_argument("--dataset-name", default="Polymarket Resolved Markets 2024", help="Dataset name")
    p_hcollect.add_argument("--start-date", default="2024-01-01T00:00:00Z", help="Start date ISO UTC")
    p_hcollect.add_argument("--end-date", default="2024-12-31T23:59:59Z", help="End date ISO UTC")
    p_hcollect.add_argument("--min-volume", type=float, default=20000.0, help="Minimum market volume filter")

    # history-validate
    p_hval = subparsers.add_parser(
        "history-validate", help="Benchmark probability model against historical market baseline"
    )
    p_hval.add_argument("--dataset", required=True, help="Dataset ID or directory path")
    p_hval.add_argument("--bootstrap-samples", type=int, default=1000, help="Clustered bootstrap iterations")
    p_hval.add_argument("--html-out", default="reports/history_benchmark.html", help="Path for HTML report")

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
        "replay": cmd_replay,
        "replay-report": cmd_replay_report,
        "datasets": cmd_datasets,
        "history-probe": cmd_history_probe,
        "history-collect": cmd_history_collect,
        "history-validate": cmd_history_validate,
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

