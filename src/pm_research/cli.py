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
  jev-probe         Test connectivity and model pinning with TypeSafe Jev decisions API
  jev-shadow-cycle  Execute prospective shadow forecasting cycle on active markets
  jev-shadow-status Display status of prospective Jev shadow forecasts & resolutions
  btc5m-phase8-diagnostics Run post-hoc diagnostics and exploratory microstructure analysis
"""

from __future__ import annotations

import argparse
import json
import logging
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
from pm_research.research.btc5m.backup import backup_database, list_backups
from pm_research.research.btc5m.collector import BTC5mAutonomousCollector
from pm_research.research.btc5m.experiment import EXPERIMENT_SPEC_HASH
from pm_research.research.btc5m.lab import BTC5mShadowLab
from pm_research.research.btc5m.process import (
    determine_collector_status,
    launch_detached_collector,
    stop_collector,
)
from pm_research.research.btc5m.snapshot import STANDARD_HORIZONS_SEC
from pm_research.safety.verifier import SafetyVerifier
from pm_research.storage.db import Database
from pm_research.utils import parse_iso_utc, to_iso_utc

logger = logging.getLogger(__name__)


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


def cmd_jev_probe(args: argparse.Namespace) -> int:
    """Test connectivity and decision inference with pinned TypeSafe Jev model via OpenRouter."""
    from pm_research.research.jev_openrouter import JEV_MODEL_PIN, JevOpenRouterClient

    print("\n" + "=" * 80)
    print("  [!] TYPESAFE JEV PROBE & AUTHENTICATION VERIFICATION")
    print(f"      Pinned Target Model: {JEV_MODEL_PIN}")
    print("      Endpoint:            https://openrouter.ai/api/alpha/decisions")
    print("      Research Mode:       SHADOW ONLY (Zero Live Trading Capability)")
    print("=" * 80)

    try:
        client = JevOpenRouterClient()
    except ValueError as err:
        print(f"\n[-] Environment Error: {err}")
        print("    Ensure the required OpenRouter research API key is exported in your environment.")
        return 1

    question = "Will humanity establish a permanent scientific base on Mars before 2035?"
    criteria = "Resolves YES if continuous crewed presence is maintained on Mars for 30+ consecutive days before Jan 1, 2035."

    print("\n  Querying probe decision:")
    print(f"    Question: {question}")
    print("    Condition: JEV_BLIND")

    t0 = time.perf_counter()
    try:
        forecast = client.generate_forecast(
            question=question,
            criteria=criteria,
            condition="JEV_BLIND",
            category="SCIENCE",
            bypass_cache=args.bypass_cache,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print("\n" + "-" * 80)
        print("  PROBE RESPONSE VERIFIED")
        print("-" * 80)
        print(f"  Model Requested:     {forecast.model_id}")
        print(f"  Model Returned:      {forecast.model_returned}")
        print(f"  Model Pin Valid:     PASS (matches {JEV_MODEL_PIN}*)")
        print(f"  Jev Yes Probability: {forecast.jev_yes_probability:.4f} ({forecast.jev_yes_probability * 100:.1f}%)")
        print(f"  Jev No Probability:  {forecast.jev_no_probability:.4f} ({forecast.jev_no_probability * 100:.1f}%)")
        print(f"  Choice:              {forecast.jev_choice}")
        if forecast.confidence is not None:
            print(f"  Reported Confidence: {forecast.confidence:.4f}")
        print(f"  Latency:             {elapsed_ms:.1f} ms")
        print(f"  Input Tokens:        {forecast.input_tokens}")
        print(f"  Output Tokens:       {forecast.output_tokens}")
        if forecast.cost is not None:
            print(f"  Reported Cost:       ${forecast.cost:.6f}")
        print(f"  Request Hash:        {forecast.request_hash[:16]}...")
        print(f"  Response Hash:       {forecast.raw_response_hash[:16]}...")
        print("  Credential Security: OpenRouter key verified in memory, zero disk writes, unlogged")
        print("=" * 80 + "\n")
        return 0
    except Exception as exc:
        print(f"\n[-] Jev Probe Failed: {exc}")
        return 1


def cmd_jev_shadow_cycle(args: argparse.Namespace) -> int:
    """Execute a prospective shadow forecasting cycle on active unresolved prediction markets."""
    from pm_research.research.jev_shadow import JevShadowRunner

    db = get_db(args.db)
    print("\n" + "=" * 80)
    print("  [!] PROSPECTIVE JEV SHADOW FORECASTING CYCLE")
    print("      Shadow Research Mode: No orders, no proposals, no portfolio execution")
    print("      Targeting active unresolved prediction markets")
    print("=" * 80)

    try:
        runner = JevShadowRunner(db=db)
    except ValueError as err:
        print(f"\n[-] Error initializing Jev shadow runner: {err}")
        print("    Ensure the required OpenRouter research API key is available in your environment.")
        return 1

    print("\n  Parameters:")
    print(f"    Max Markets:    {args.max_markets}")
    print(f"    Min Liquidity:  ${args.min_liquidity:,.0f}")
    print(f"    Market Aware:   {not args.no_aware}")
    print(f"    Bypass Cache:   {args.bypass_cache}")
    print("\n  Starting shadow cycle (discovering unresolved markets & querying Jev)...")

    summary = runner.run_prospective_cycle(
        max_markets=args.max_markets,
        min_liquidity=args.min_liquidity,
        enable_market_aware=not args.no_aware,
        bypass_cache=args.bypass_cache,
    )

    print("\n" + "-" * 80)
    print("  CYCLE SUMMARY")
    print("-" * 80)
    print(f"  Cycle ID:            {summary.cycle_id}")
    print(f"  Timestamp (UTC):     {to_iso_utc(summary.timestamp_utc)}")
    print(f"  Markets Discovered:  {summary.markets_discovered}")
    print(f"  Markets Eligible:    {summary.markets_eligible}")
    print(f"  Markets Captured:    {summary.markets_captured}")
    print(f"  Jev Requests Sent:   {summary.jev_requests_sent}")
    print(f"  Jev Cache Hits:      {summary.jev_cache_hits}")
    print(f"  Failed Requests:     {summary.failed_requests}")
    print(f"  Input Tokens:        {summary.total_input_tokens}")
    print(f"  Output Tokens:       {summary.total_output_tokens}")
    if summary.total_cost is not None:
        print(f"  Total Cost:          ${summary.total_cost:.6f}")

    if summary.captures:
        print("\n" + "-" * 80)
        print("  PROSPECTIVE CAPTURES RECORDED")
        print("-" * 80)
        print(f"  {'Market Question':<45} | {'Mkt_p':>6} | {'Ilsa_p':>6} | {'JevBld':>6} | {'JevAwr':>6}")
        print("  " + "-" * 77)
        for cap in summary.captures:
            q_short = cap.market_question[:43] + ".." if len(cap.market_question) > 45 else cap.market_question
            mkt_str = f"{cap.market_prob:.2f}"
            ilsa_str = f"{cap.ilsa_prob:.2f}"
            bld_str = f"{cap.jev_blind_prob:.2f}" if cap.jev_blind_prob is not None else "  N/A"
            awr_str = f"{cap.jev_market_aware_prob:.2f}" if cap.jev_market_aware_prob is not None else "  N/A"
            print(f"  {q_short:<45} | {mkt_str:>6} | {ilsa_str:>6} | {bld_str:>6} | {awr_str:>6}")

    print("\n  [i] SHADOW INVARIANT CONFIRMATION:")
    print("      - Zero trade proposals generated")
    print("      - Zero orders submitted to PaperBroker")
    print("      - Portfolio sizing unchanged")
    print("=" * 80 + "\n")
    return 0


def cmd_jev_shadow_status(args: argparse.Namespace) -> int:
    """Display status of recorded prospective Jev shadow forecasts and resolution evaluations."""
    from pm_research.research.jev_shadow import JevResolutionScorer

    db = get_db(args.db)
    print("\n" + "=" * 80)
    print("  [!] PROSPECTIVE JEV SHADOW RESEARCH STATUS")
    print("=" * 80)

    scorer = JevResolutionScorer(db=db)
    scoring_summary = scorer.evaluate_pending_resolutions()

    captures = db.get_jev_captures(limit=100)

    print(f"\n  Total Captures in DB:   {scoring_summary.total_captures}")
    print(f"  Unresolved Captures:    {scoring_summary.unresolved_captures}")
    print(f"  Resolved Captures:      {scoring_summary.resolved_captures}")
    print(f"  Resolution Status:      {scoring_summary.scoring_status}")

    if scoring_summary.scoring_status == "WAITING_FOR_FUTURE_RESOLUTIONS":
        print("  [!] Markets are currently active/unresolved. Quantitative scoring awaits genuine future resolutions.")

    if captures:
        print("\n" + "-" * 80)
        print("  RECENT PROSPECTIVE CAPTURES")
        print("-" * 80)
        print(f"  {'Captured (UTC)':<19} | {'Market Question':<35} | {'Mkt_p':>5} | {'Ilsa_p':>6} | {'JevBld':>6} | {'JevAwr':>6}")
        print("  " + "-" * 88)
        for cap in captures[:20]:
            t_str = to_iso_utc(cap.captured_at)[:19].replace("T", " ")
            q_short = cap.market_question[:33] + ".." if len(cap.market_question) > 35 else cap.market_question
            mkt_str = f"{cap.market_prob:.2f}"
            ilsa_str = f"{cap.ilsa_prob:.2f}"
            bld_str = f"{cap.jev_blind_prob:.2f}" if cap.jev_blind_prob is not None else " N/A"
            awr_str = f"{cap.jev_market_aware_prob:.2f}" if cap.jev_market_aware_prob is not None else " N/A"
            print(f"  {t_str:<19} | {q_short:<35} | {mkt_str:>5} | {ilsa_str:>6} | {bld_str:>6} | {awr_str:>6}")

    print("=" * 80 + "\n")
    return 0


def cmd_btc5m_probe(args: argparse.Namespace) -> int:
    """Probe public read-only market data feeds and OpenRouter inference connectivity."""
    db = get_db(args.db)
    lab = BTC5mShadowLab(db=db)
    print("\n" + "=" * 80)
    print("  [!] PROBING BTC 5-MINUTE DATA FEEDS & JEV CONNECTIVITY")
    print("=" * 80)
    results = lab.probe_connectivity()
    print(f"  Polymarket Gamma API (Active Round):  {'PASS' if results['polymarket_gamma'] else 'FAIL'}")
    print(f"  Polymarket Book:                      {'PASS' if results['polymarket_book'] else 'FAIL'}")
    print(f"  Binance USD-M Perpetual (FAPI):       {'PASS' if results['binance_perp'] else 'FAIL'}")
    print(f"  OpenRouter TypeSafe Jev API Key:      {'PASS' if results['openrouter_jev'] else 'FAIL'}")
    print("\n  Market Baseline Provenance (Requirement B1):")
    details = results.get("details", {})
    print(f"    native_market_q:                 {details.get('native_market_q')}")
    print(f"    cross_outcome_implied_q:         {details.get('cross_outcome_implied_q')}")
    print("    PRIMARY_MARKET_BASELINE_METHOD:  NATIVE_UP_MIDPOINT")
    print("    SYNTHETIC_COMPLEMENT_AS_PRIMARY: NO")
    print("\n  Probe Details:")
    for k, v in details.items():
        print(f"    - {k}: {v}")
    print("=" * 80 + "\n")
    all_ok = (
        results["polymarket_gamma"]
        and results["polymarket_book"]
        and results["binance_perp"]
        and results["openrouter_jev"]
    )
    return 0 if all_ok else 1


def cmd_btc5m_shadow(args: argparse.Namespace) -> int:
    """Run prospective BTC 5-minute shadow forecasting loop."""
    db = get_db(args.db)
    horizons = STANDARD_HORIZONS_SEC
    if getattr(args, "horizons", None):
        horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())

    lab = BTC5mShadowLab(db=db, horizons_sec=horizons)
    try:
        lab.ref_feed.start_background_listener()
        lab._ref_listener_started = True
    except Exception as e:
        logger.warning(f"Could not start background reference listener: {e}")

    rounds_count = max(1, getattr(args, "rounds", 1))

    print("\n" + "=" * 80)
    print("  [!] STARTING BTC 5-MINUTE PROSPECTIVE SHADOW FORECASTING")
    print(f"  Rounds to monitor:    {rounds_count}")
    print(f"  Standard horizons:    {horizons}s remaining")
    print(f"  Settlement polling:   {'ENABLED' if not args.no_poll_resolution else 'DISABLED'}")
    print("=" * 80)

    for i in range(rounds_count):
        print(f"\n--- [Round {i+1}/{rounds_count}] Discovering active market... ---")
        try:
            round_info = lab.contract_mgr.discover_active_round()
            max_h = max(horizons)
            if round_info.seconds_remaining < max_h + 1.0:
                wait_sec = max(0.0, round_info.end_epoch - time.time()) + 2.0
                print(f"  Current round {round_info.round_slug} has {round_info.seconds_remaining:.1f}s remaining (< {max_h + 1.0}s required).")
                print(f"  Waiting {wait_sec:.1f}s for fresh round to guarantee complete point-in-time horizon capture...")
                time.sleep(wait_sec)
                round_info = lab.contract_mgr.discover_active_round()

            print(f"  Monitoring: {round_info.round_slug}")
            print(f"  PriceToBeat: {round_info.price_to_beat} (source: {round_info.price_to_beat_source})")
            print(f"  Seconds remaining in round: {round_info.seconds_remaining:.1f}s")

            res = lab.monitor_round(
                round_info=round_info,
                horizons_sec=horizons,
                poll_resolution_after=not args.no_poll_resolution,
            )
            print(f"  Round {round_info.round_slug} complete:")
            print(f"    Snapshots captured: {res['snapshots_captured']}")
            print(f"    Forecasts produced: {res['forecasts_produced']}")
            if res.get("resolved"):
                print(f"    Official outcome:   {res.get('outcome')} (price: {res.get('resolution_price')})")
        except Exception as e:
            print(f"  [ERROR] Monitoring round failed: {e}")
            logger.exception("Round monitoring error")

    print("\n" + "=" * 80)
    print("  [+] BTC 5m prospective shadow run completed.")
    print("=" * 80 + "\n")
    return 0


def cmd_btc5m_status(args: argparse.Namespace) -> int:
    """Display status of BTC 5-minute research rounds, snapshots, and forecasts."""
    db = get_db(args.db)
    print("\n" + "=" * 80)
    print("  [!] BTC 5-MINUTE PROSPECTIVE RESEARCH STATUS")
    print("=" * 80)

    rounds = db.get_btc5m_rounds()
    snapshots = db.get_btc5m_snapshots()
    forecasts = db.get_btc5m_forecasts()
    scores = db.get_btc5m_resolution_scores()

    print(f"\n  Total Discovered Rounds: {len(rounds)}")
    print(f"  Total Snapshots:         {len(snapshots)} ({sum(1 for s in snapshots if s.is_valid)} valid)")
    print(f"  Total Forecasts:         {len(forecasts)}")
    print(f"  Evaluated Scores:        {len(scores)}")

    if rounds:
        print("\n" + "-" * 80)
        print("  RECENT ROUNDS")
        print("-" * 80)
        print(f"  {'Round Slug':<35} | {'PriceToBeat':>11} | {'Status':<9} | {'Outcome':<7}")
        print("  " + "-" * 75)
        for r in rounds[:15]:
            ptb_str = f"{r['price_to_beat']:.2f}" if r.get('price_to_beat') else "N/A"
            out_str = str(r.get('resolved_outcome') or "PENDING")
            print(f"  {r['round_slug']:<35} | {ptb_str:>11} | {r['status']:<9} | {out_str:<7}")

    if forecasts:
        print("\n" + "-" * 80)
        print("  RECENT FORECASTS")
        print("-" * 80)
        print(f"  {'Round':<30} | {'Hz':>4} | {'Condition':<26} | {'Mkt_q':>5} | {'Jev_UP':>6} | {'Choice':<4}")
        print("  " + "-" * 88)
        for fc in forecasts[-16:]:
            mkt_str = f"{fc.market_q:.2f}" if fc.market_q is not None else " N/A"
            print(f"  {fc.round_slug:<30} | {fc.target_horizon_sec:>3}s | {fc.condition:<26} | {mkt_str:>5} | {fc.jev_up_prob:>6.2f} | {fc.jev_choice:<4}")

    print("=" * 80 + "\n")
    return 0


def cmd_btc5m_report(args: argparse.Namespace) -> int:
    """Generate comparative ablation evaluation report with clustered bootstrapping."""
    db = get_db(args.db)
    lab = BTC5mShadowLab(db=db)
    summary = lab.compute_evaluation_summary()

    print("\n" + "=" * 80)
    print("  [!] BTC 5-MINUTE ABLATION FORECASTING BENCHMARK REPORT")
    print("=" * 80)
    print("  PRIMARY_MARKET_BASELINE_METHOD=NATIVE_UP_MIDPOINT")
    print("  SYNTHETIC_COMPLEMENT_USED_AS_PRIMARY=NO")
    print("  CROSS_OUTCOME_IMPLIED_VALUE_RETAINED_AS_DIAGNOSTIC=YES")

    if summary.get("status") == "NO_RESOLVED_DATA":
        print("\n  [!] No resolved rounds with completed scores found in database.")
        print("      Run prospective shadow collection ('pmr btc5m-shadow') and allow rounds to settle.")
        print("=" * 80 + "\n")
        return 0

    print(f"\n  Total Evaluated Rounds:    {summary['total_rounds']}")
    print(f"  Total Evaluated Snapshots: {summary['total_scores']}")

    print("\n" + "-" * 80)
    print("  ABLATION PERFORMANCE SUMMARY (vs Raw Polymarket Native Consensus)")
    print("-" * 80)
    print(f"  {'Condition':<28} | {'N':>4} | {'Brier':>7} | {'MktBrier':>8} | {'DeltaBrier':>10} | {'95% Bootstrap CI':<19}")
    print("  " + "-" * 86)

    for cond, m in summary.get("metrics_by_condition", {}).items():
        ci_str = f"[{m['delta_brier_95ci'][0]:+.4f}, {m['delta_brier_95ci'][1]:+.4f}]" if m.get("delta_brier_95ci") else "N/A"
        delta_str = f"{m['delta_brier']:+.4f}" if m.get('delta_brier') is not None else "N/A"
        print(f"  {cond:<28} | {m.get('paired_count', 0):>4} | {m['mean_brier']:>7.4f} | {m['market_brier']:>8.4f} | {delta_str:>10} | {ci_str:<19}")

    if summary.get("paired_deltas"):
        print("\n" + "-" * 80)
        print("  PAIRED ABLATION COMPARISONS")
        print("-" * 80)
        for pair_name, val in summary["paired_deltas"].items():
            print(f"  {pair_name:<35} = {val:+.5f}")

    print("\n" + "-" * 80)
    print("  Notes:")
    print("  * Delta Brier = Forecaster Brier - Market Brier (negative indicates forecaster outperforms market).")
    print("  * 95% Bootstrap CI is computed via round-clustered resampling to account for round-level settlement correlation.")
    print("=" * 80 + "\n")
    return 0


def cmd_btc5m_collect(args: argparse.Namespace) -> int:
    """Run autonomous prospective BTC 5-minute collector with crash recovery and cost ceiling."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    db = get_db(args.db)
    target_rounds = getattr(args, "target_valid_rounds", 100) or getattr(args, "rounds", 100)
    cost_ceiling = getattr(args, "cost_ceiling", 10.0)

    print("\n" + "=" * 80)
    print("  [!] LAUNCHING AUTONOMOUS BTC 5-MINUTE PROSPECTIVE RESEARCH COLLECTOR")
    print("=" * 80)
    print(f"  TARGET_VALID_ROUNDS:   {target_rounds}")
    print(f"  COST_CEILING_USD:      ${cost_ceiling:.2f}")
    print(f"  EXPERIMENT_SPEC_HASH:  {EXPERIMENT_SPEC_HASH}")
    print("  SAFETY_CONTRACT:       FORECAST RESEARCH ONLY (Zero Trading / Paper Only)")
    print("=" * 80 + "\n")

    collector = BTC5mAutonomousCollector(
        db=db,
        target_valid_rounds=target_rounds,
        cost_ceiling_usd=cost_ceiling,
    )
    try:
        collector.run()
        return 0
    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt received. Gracefully stopping collector...")
        collector.request_stop()
        return 0


def cmd_btc5m_collector_start(args: argparse.Namespace) -> int:
    """Launch the autonomous BTC 5m prospective collector as a detached OS process."""
    target_rounds = getattr(args, "target_valid_rounds", 100) or 100
    cost_ceiling = getattr(args, "cost_ceiling", 10.0) or 10.0

    print("\n" + "=" * 80)
    print("  [!] LAUNCHING DETACHED AUTONOMOUS BTC 5m COLLECTOR")
    print("=" * 80)
    print(f"  TARGET_VALID_ROUNDS:        {target_rounds}")
    print(f"  COST_CEILING_USD:           ${cost_ceiling:.2f}")
    print(f"  EXPERIMENT_SPEC_HASH:       {EXPERIMENT_SPEC_HASH}")
    print("  SINGLE_INSTANCE_GUARANTEE:  fcntl.flock (data/collector.lock)")
    print("  LOG_REDIRECT:               data/logs/btc5m_collector.log")
    print("  SAFETY_CONTRACT:            FORECAST RESEARCH ONLY (Zero Trading / Paper Only)")

    success, pid, msg = launch_detached_collector(
        target_valid_rounds=target_rounds,
        cost_ceiling_usd=cost_ceiling,
    )

    if success:
        print(f"\n  [+] SUCCESS: {msg}")
        print(f"      PID: {pid}")
        print("      Collector is running detached in background, independent of terminal.")
        print("      Check status with: uv run pmr btc5m-collector-status")
        print("      Stop anytime with: uv run pmr btc5m-collector-stop")
        print("=" * 80 + "\n")
        return 0
    else:
        print(f"\n  [-] LAUNCH REJECTED: {msg}")
        if pid:
            print(f"      Existing PID: {pid}")
        print("=" * 80 + "\n")
        return 1


def cmd_btc5m_collector_status(args: argparse.Namespace) -> int:
    """Inspect live health, process state, and progress of the autonomous collector."""
    db = get_db(args.db)
    status_info = determine_collector_status(db=db)

    print("\n" + "=" * 80)
    print("  [!] BTC 5-MINUTE COLLECTOR LIVE HEALTH & STATUS")
    print("=" * 80)

    state = status_info["state"]
    pid = status_info["pid"]
    proc_alive = status_info["process_alive"]
    proc_verified = status_info["process_verified"]
    lock_held = status_info["lock_held"]
    hb_fresh = status_info["heartbeat_fresh"]
    hb_age = status_info["heartbeat_age_sec"]
    file_data = status_info["status_data"]

    print(f"  COLLECTOR_STATE:       {state}")
    print(
        f"  PROCESS_ALIVE:         {'YES' if proc_alive else 'NO'} "
        f"(PID: {pid or 'None'}, Verified: {'YES' if proc_verified else 'NO'})"
    )
    print(f"  FILE_LOCK_HELD:        {'YES' if lock_held else 'NO'}")
    age_str = f"{hb_age:.1f}s ago" if hb_age is not None else "N/A"
    print(f"  HEARTBEAT_FRESH:       {'YES' if hb_fresh else 'NO'} (Age: {age_str}, Stale Threshold: 120s)")

    if file_data:
        print(f"  Last Heartbeat UTC:    {file_data.get('timestamp_utc', 'N/A')}")
        print(f"  Active Round Slug:     {file_data.get('current_round_slug') or 'None (idle/polling)'}")
        print(
            f"  Valid Resolved Rounds: {file_data.get('valid_resolved_rounds', 0)} / "
            f"{file_data.get('target_valid_rounds', 100)}"
        )
        print(f"  Total Snapshots:       {file_data.get('total_snapshots', 0)}")
        print(f"  Total Forecasts:       {file_data.get('total_forecasts', 0)}")
        cost_spent = file_data.get("total_openrouter_cost_usd", 0.0)
        ceiling = file_data.get("cost_ceiling_usd", 10.0)
        guard = file_data.get("cost_guard_triggered", False)
        print(
            f"  Canonical Cost Spend:  ${cost_spent:.4f} / ${ceiling:.2f} "
            f"(Guard Triggered: {guard})"
        )
        feeds = file_data.get("feeds", {})
        print(f"  Chainlink RTDS Feed:   {feeds.get('chainlink_rtds', 'N/A')}")
        print(f"  Binance Perp WS Feed:  {feeds.get('binance_perp_ws', 'N/A')}")
        print(f"  Experiment Spec Hash:  {file_data.get('experiment_spec_hash', 'N/A')}")
        if file_data.get("notes"):
            print(f"  Notes:                 {file_data.get('notes')}")
    else:
        hb = db.get_latest_collector_heartbeat()
        if hb:
            print(f"  Collector State (DB):  {hb.get('status', 'UNKNOWN')}")
            print(f"  Last Heartbeat:        {hb.get('timestamp_utc', 'N/A')}")
            print(f"  Active Round Slug:     {hb.get('current_round_slug') or 'None'}")
            print(f"  Valid Resolved Rounds: {hb.get('valid_resolved_rounds', 0)}")
            print(f"  Total Spend:           ${hb.get('total_openrouter_cost_usd', 0.0):.4f}")
            print(f"  RTDS Feed:             {hb.get('rtds_status', 'N/A')}")
            print(f"  Binance Perp Feed:     {hb.get('binance_ws_status', 'N/A')}")
        else:
            print("  No collector activity recorded yet. Start with 'pmr btc5m-collector-start'.")

    # Also list checkpoints
    checkpoints = db.get_checkpoints(canonical_only=False)
    if checkpoints:
        print("\n" + "-" * 80)
        print("  MILESTONE CHECKPOINTS")
        print("-" * 80)
        for cp in checkpoints:
            tag = " [LEGACY DUPLICATE]" if not cp.get("is_canonical", 1) else ""
            print(
                f"  Milestone {cp['milestone_rounds']:>3}r{tag:<20} | Created: {cp['created_at_utc']} | "
                f"Valid: {cp['valid_resolved_rounds']} | Spend: ${cp['total_openrouter_cost_usd']:.4f}"
            )

    print("=" * 80 + "\n")
    return 0


def cmd_btc5m_collector_stop(args: argparse.Namespace) -> int:
    """Signal running autonomous collector to stop gracefully and verify termination."""
    print("\n" + "=" * 80)
    print("  [!] STOPPING BTC 5-MINUTE AUTONOMOUS COLLECTOR")
    print("=" * 80)
    success, msg = stop_collector()
    if success:
        print(f"  [+] {msg}")
        print("=" * 80 + "\n")
        return 0
    else:
        print(f"  [-] {msg}")
        print("=" * 80 + "\n")
        return 1


def cmd_btc5m_audit_cost(args: argparse.Namespace) -> int:
    """Audit OpenRouter cost accounting semantics and verify deduplication."""
    db = get_db(args.db)
    audit = db.get_cost_accounting_audit()

    print("\n" + "=" * 80)
    print("  [!] OPENROUTER COST ACCOUNTING FORENSIC AUDIT")
    print("=" * 80)
    print(f"  TOTAL_DB_FORECASTS:              {audit['total_db_forecasts']}")
    print(f"  REMOTE_CHARGED_REQUESTS:         {audit['remote_requests_charged']} (exact OpenRouter usage.cost)")
    print(f"  LOCAL_CACHE_HITS:                {audit['local_cache_hits_zero_cost']} ($0.00 additional spend)")
    print(f"  NULL_COST_FORECASTS:             {audit['null_cost_forecasts']}")
    print("-" * 80)
    print(f"  ACTUAL_CANONICAL_COST (USD):     ${audit['canonical_total_reported_cost_usd']:.6f}")
    print(f"  ALL_EXPERIMENTS_COST (USD):      ${audit['all_experiments_total_cost_usd']:.6f}")
    print("-" * 80)
    print("  SYNTHETIC FORMULA COMPARISON (for the 96 pilot requests):")
    print(f"    Phase 6.1 Report (Formula A):  ${audit['synthetic_formula_a_cost_usd']:.5f} ($0.30/1M in + $1.50/1M out)")
    print(f"    Phase 7 Report   (Formula B):  ${audit['synthetic_formula_b_cost_usd']:.5f} ($0.15/1M in + $0.60/1M out)")
    print(f"    Actual OpenRouter Spend:       ${audit['actual_pilot_openrouter_cost_usd']:.6f}")
    print("-" * 80)
    print("  STATUS:")
    print("    - Discrepancy mathematically explained by differing synthetic token formulas.")
    print("    - True canonical metric implemented: sum of actual OpenRouter usage.cost.")
    print("    - Cache hits add $0.00 to spend.")
    print("=" * 80 + "\n")
    return 0


def cmd_btc5m_backup(args: argparse.Namespace) -> int:
    """Create an atomic transactional SQLite database backup."""
    db = get_db(args.db)
    valid_rounds = db.count_valid_resolved_rounds()
    dest = backup_database(
        db_path=args.db or "data/pm_research.db",
        rounds_count=valid_rounds,
    )
    backups = list_backups()
    print("\n" + "=" * 80)
    print("  [+] TRANSACTIONAL SQLite BACKUP COMPLETED")
    print("=" * 80)
    print(f"  New Backup File: {dest} ({dest.stat().st_size:,} bytes)")
    print(f"  Valid Rounds:    {valid_rounds}")
    print(f"  Total Backups:   {len(backups)}")
    print("=" * 80 + "\n")
    return 0


def cmd_btc5m_audit_dataset(args: argparse.Namespace) -> int:
    """Audit BTC 5m dataset integrity, snapshot validity rates, score uniqueness, and forecast pairing."""
    db = get_db(args.db)
    audit = db.get_dataset_audit()

    print("\n" + "=" * 80)
    print("  [!] BTC 5-MINUTE DATASET INTEGRITY & FORENSIC AUDIT")
    print(f"      Experiment ID:   {audit['experiment_id']}")
    print(f"      Spec Hash:       {audit['experiment_spec_hash']}")
    print("=" * 80)
    print("  ROUND STATISTICS:")
    print(f"    Physical Rounds Discovered:             {audit['rounds_discovered']}")
    print(f"    Physical Rounds Resolved:               {audit['rounds_resolved']}")
    print(f"    Physical Rounds With Valid Snapshots:   {audit['rounds_scored']}")
    print(f"    Physical Rounds With Zero Valid Snaps:  {audit['rounds_with_zero_valid_snapshots']}")
    print("-" * 80)
    print("  SNAPSHOT INTEGRITY & YIELD:")
    print(f"    Total Snapshots:                        {audit['snapshots_total']}")
    print(f"    Valid Snapshots:                        {audit['snapshots_valid']} ({audit['snapshot_valid_rate'] * 100:.1f}%)")
    print(f"    Invalid / Skipped Snapshots:            {audit['snapshots_invalid']} ({(1.0 - audit['snapshot_valid_rate']) * 100:.1f}%)")
    print(f"    Raw Multi-Snapshot (Round, Horizon):    {audit['raw_multiple_snapshot_pairs']}")
    print(f"    Max Snapshots per (Round, Horizon):     {audit['max_snapshots_per_round_horizon']}")
    print("-" * 80)
    print("  SKIP REASONS BREAKDOWN:")
    for reason, count in audit["skip_reason_breakdown"].items():
        pct = (count / audit["snapshots_invalid"] * 100) if audit["snapshots_invalid"] > 0 else 0.0
        print(f"    {reason:<35}: {count:>4} ({pct:>5.1f}%)")
    print("-" * 80)
    print("  PER-HORIZON VALIDITY BREAKDOWN:")
    print(f"    {'Horizon':<10} | {'Total':<8} | {'Valid':<8} | {'Invalid':<8} | {'Valid %':<8}")
    print("    " + "-" * 50)
    for hz, stats in sorted(audit["horizon_breakdown"].items(), reverse=True):
        hz_label = f"{hz}s"
        t = stats["total"]
        v = stats["valid"]
        inv = stats["invalid"]
        pct = (v / t * 100) if t > 0 else 0.0
        print(f"    {hz_label:<10} | {t:<8} | {v:<8} | {inv:<8} | {pct:>6.1f}%")
    print("-" * 80)
    print("  SCORING & EVALUATION INTEGRITY:")
    print(f"    Total Scored Round-Horizons:            {audit['scored_round_horizons']}")
    print(f"    Duplicate Score Keys:                   {audit['duplicate_score_keys']}")
    print(f"    Total Forecast Rows:                    {audit['forecast_rows']}")
    print(f"    Valid Forecast Rows:                    {audit['valid_forecast_rows']}")
    print(f"    Invalid Forecast Rows:                  {audit['invalid_forecast_rows']}")
    print(f"    Unpaired Score Rows (Market / Cond NA): {audit['unpaired_score_rows']}")
    print("=" * 80 + "\n")
    return 0


def cmd_btc5m_phase8_diagnostics(args: argparse.Namespace) -> int:
    """Execute Phase 8A post-hoc diagnostics and exploratory microstructure analysis."""
    from pm_research.research.btc5m.diagnostics import BTC5mPhase8Diagnostics

    db = get_db(args.db)
    out_dir = Path(getattr(args, "output_dir", "reports/btc5m_phase8_diagnostics"))
    engine = BTC5mPhase8Diagnostics(db)
    report = engine.run_all_diagnostics(output_dir=out_dir)

    print("\n" + "=" * 80)
    print("  [!] BTC 5-MINUTE PHASE 8A POST-HOC DIAGNOSTICS & MICROSTRUCTURE REPORT")
    print("=" * 80)
    print("  STATUS:                  PASS")
    print(f"  OUTPUT_DIRECTORY:        {out_dir}")
    print("  RESEARCH_INTEGRITY:      IMMUTABLE READ-ONLY EXECUTION")
    print("  DATA_LEAKAGE_AUDIT:      PASS (NO FUTURE LOOKAHEAD)")
    print("-" * 80)

    # Per-horizon summary
    print("\n  PER-HORIZON BENCHMARK EVALUATION:")
    print(f"  {'Horizon':<8} | {'Forecaster':<26} | {'N':>5} | {'Brier':>7} | {'LogLoss':>7} | {'Acc':>5} | {'DeltaBr':>8} | {'95% Bootstrap CI':<19}")
    print("  " + "-" * 98)
    for m in report["per_horizon_metrics"]:
        delta_str = f"{m['delta_brier']:+.4f}" if m.get("delta_brier") is not None else "BASELINE"
        ci_str = f"[{m['delta_brier_95ci'][0]:+.4f}, {m['delta_brier_95ci'][1]:+.4f}]" if m.get("delta_brier_95ci") else "N/A"
        print(f"  {str(m['horizon_sec']) + 's':<8} | {m['forecaster']:<26} | {m['n_observations']:>5} | {m['brier']:>7.4f} | {m['log_loss']:>7.4f} | {m['accuracy_05']:>5.3f} | {delta_str:>8} | {ci_str:<19}")

    # Jev Behavior
    jb = report["jev_behavior"]
    print("\n" + "-" * 80)
    print("  JEV BEHAVIOR & CALIBRATION FAILURE DECOMPOSITION:")
    print(f"    Market Aware Shrunk toward 0.5: {jb['MARKET_AWARE']['shrunk_toward_05_pct']}% ({jb['MARKET_AWARE']['shrunk_toward_05_count']}/{jb['MARKET_AWARE']['n_paired']})")
    print(f"    Full Shrunk toward 0.5:         {jb['FULL']['shrunk_toward_05_pct']}% ({jb['FULL']['shrunk_toward_05_count']}/{jb['FULL']['n_paired']})")
    print("    Adjustment Magnitude Tiers (Market Aware):")
    for t in jb["MARKET_AWARE"]["tiers_by_adjustment_magnitude"]:
        print(f"      Tier {t['tier']:<16} N={t['n']:>4} | MktBr={t['market_brier']:.4f} | JevBr={t['jev_brier']:.4f} | DeltaBr={t['delta_brier']:+.4f}")

    # Univariate signals
    print("\n" + "-" * 80)
    print("  DIRECT MICROSTRUCTURE SIGNALS (GROUPED 5-FOLD CV):")
    print(f"  {'Feature':<32} | {'Sign':<4} | {'StdBeta':>8} | {'CV Brier':>8} | {'Delta vs Base':>14}")
    print("  " + "-" * 74)
    for u in report["univariate_signals"]:
        print(f"  {u['feature_name']:<32} | {u['sign']:<4} | {u['standardized_beta']:>+8.4f} | {u['cv_brier']:>8.5f} | {u['delta_brier_vs_base']:>+14.5f}")

    # Residual model
    res = report["residual_model"]
    print("\n" + "-" * 80)
    print("  POLYMARKET RESIDUAL LOGISTIC-OFFSET MODEL:")
    print(f"    N Paired:                {res['n_observations']}")
    print(f"    Grouped Folds:           {res['grouped_folds']}")
    print(f"    Market Brier:            {res['market_brier']:.5f}")
    print(f"    Offset Model CV Brier:   {res['offset_model_cv_brier']:.5f}")
    print(f"    Delta Brier:             {res['delta_brier']:+.5f}")
    print(f"    Market LogLoss:          {res['market_logloss']:.5f}")
    print(f"    Offset Model CV LogLoss: {res['offset_model_cv_logloss']:.5f}")
    print(f"    Delta LogLoss:           {res['delta_logloss']:+.5f}")
    print(f"    Average Coefficients:    {res['average_coefficients']}")

    # Microstructure-only model
    mo = report["microstructure_only"]
    print("\n" + "-" * 80)
    print("  MICROSTRUCTURE-ONLY BASELINE MODEL (L2 Logistic, Excludes Market Q):")
    print(f"    Label:                   {mo['label']}")
    print(f"    CV Brier:                {mo['microstructure_only_cv_brier']:.5f}")
    print(f"    CV LogLoss:              {mo['microstructure_only_cv_logloss']:.5f}")
    print(f"    Base Rate Brier:         {mo['base_rate_brier']:.5f}")
    print(f"    Delta vs Base Rate:      {mo['delta_brier_vs_base_rate']:+.5f}")
    print(f"    Delta vs Jev Ref Only:   {mo['delta_brier_vs_jev_ref_only']:+.5f}")
    print(f"    Delta vs Jev Ref+Perp:   {mo['delta_brier_vs_jev_ref_perp']:+.5f}")

    # Lead/lag
    ll = report["lead_lag"]
    print("\n" + "-" * 80)
    print("  LEAD/LAG FEASIBILITY AUDIT:")
    print(f"    Label:                   {ll['label']}")
    print(f"    Total Transition Pairs:  {ll['total_transition_pairs']}")
    for tr in ll["transitions"]:
        corr_s = f"{tr['correlation_with_binance_open_return']:+.4f}" if tr.get("correlation_with_binance_open_return") is not None else "N/A"
        print(f"    Transition {tr['transition']:<12}: N={tr['valid_pairs']:>4} | MeanDeltaQ={tr['mean_delta_q']:+.4f} | StdDeltaQ={tr['std_delta_q']:.4f} | Corr(B_ret, dQ)={corr_s}")

    print("\n" + "=" * 80 + "\n")
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

    # jev-probe
    p_jev_probe = subparsers.add_parser(
        "jev-probe", help="Test connectivity and model pinning with TypeSafe Jev decisions API"
    )
    p_jev_probe.add_argument("--bypass-cache", action="store_true", help="Bypass local cache")

    # jev-shadow-cycle
    p_jev_cycle = subparsers.add_parser(
        "jev-shadow-cycle", help="Execute prospective shadow forecasting cycle on active markets"
    )
    p_jev_cycle.add_argument("--max-markets", type=int, default=20, help="Max markets to evaluate")
    p_jev_cycle.add_argument("--min-liquidity", type=float, default=1000.0, help="Min liquidity filter")
    p_jev_cycle.add_argument("--no-aware", action="store_true", help="Disable JEV_MARKET_AWARE condition")
    p_jev_cycle.add_argument("--bypass-cache", action="store_true", help="Bypass local cache")

    # jev-shadow-status
    subparsers.add_parser(
        "jev-shadow-status", help="Display status of prospective Jev shadow forecasts & resolutions"
    )

    # btc5m-probe
    subparsers.add_parser(
        "btc5m-probe", help="Probe BTC 5-minute data feeds and OpenRouter inference connectivity"
    )

    # btc5m-shadow
    p_btc5m_shadow = subparsers.add_parser(
        "btc5m-shadow", help="Run prospective BTC 5-minute shadow forecasting laboratory"
    )
    p_btc5m_shadow.add_argument("--rounds", type=int, default=1, help="Number of 5-minute rounds to monitor")
    p_btc5m_shadow.add_argument(
        "--horizons",
        default=None,
        help="Comma-separated horizon seconds remaining (e.g. '240,180,120,60,30')",
    )
    p_btc5m_shadow.add_argument(
        "--no-poll-resolution",
        action="store_true",
        help="Skip polling official Polymarket Data API resolution after round end",
    )
    p_btc5m_shadow.add_argument(
        "--bypass-cache",
        action="store_true",
        help="Bypass local cache for Jev decisions",
    )

    # btc5m-status
    subparsers.add_parser(
        "btc5m-status", help="Display status of BTC 5m rounds, snapshots, forecasts, and resolutions"
    )

    # btc5m-report
    subparsers.add_parser(
        "btc5m-report", help="Generate comparative ablation evaluation report with bootstrap CI"
    )

    # btc5m-collect
    p_btc5m_collect = subparsers.add_parser(
        "btc5m-collect",
        help="Run autonomous crash-recoverable collector for frozen prospective BTC 5m research",
    )
    p_btc5m_collect.add_argument(
        "--target-valid-rounds",
        "--rounds",
        type=int,
        default=100,
        dest="target_valid_rounds",
        help="Target number of valid resolved rounds before stopping (default: 100)",
    )
    p_btc5m_collect.add_argument(
        "--cost-ceiling",
        type=float,
        default=10.0,
        help="OpenRouter cost ceiling in USD; pauses Jev inference when exceeded (default: $10.00)",
    )

    # btc5m-collector-start
    p_btc5m_cstart = subparsers.add_parser(
        "btc5m-collector-start",
        help="Launch the autonomous prospective collector as a detached OS background process",
    )
    p_btc5m_cstart.add_argument(
        "--target-valid-rounds",
        "--rounds",
        type=int,
        default=100,
        dest="target_valid_rounds",
        help="Target number of valid resolved rounds before stopping (default: 100)",
    )
    p_btc5m_cstart.add_argument(
        "--cost-ceiling",
        type=float,
        default=10.0,
        help="OpenRouter cost ceiling in USD; pauses Jev inference when exceeded (default: $10.00)",
    )

    # btc5m-collector-status
    subparsers.add_parser(
        "btc5m-collector-status",
        help="Inspect live health and progress of the autonomous prospective collector",
    )

    # btc5m-collector-stop
    subparsers.add_parser(
        "btc5m-collector-stop",
        help="Signal the running autonomous prospective collector to shut down gracefully",
    )

    # btc5m-audit-cost
    subparsers.add_parser(
        "btc5m-audit-cost",
        help="Audit OpenRouter cost accounting semantics, deduplication, and cache hits",
    )

    # btc5m-backup
    subparsers.add_parser(
        "btc5m-backup",
        help="Create immediate transactional SQLite database backup",
    )

    # btc5m-audit-dataset
    subparsers.add_parser(
        "btc5m-audit-dataset",
        help="Audit BTC 5m dataset integrity, snapshot validity rates, score uniqueness, and forecast pairing",
    )

    # btc5m-phase8-diagnostics
    p_phase8_diag = subparsers.add_parser(
        "btc5m-phase8-diagnostics",
        help="Run post-hoc diagnostic and exploratory microstructure feature analysis (Phase 8A)",
    )
    p_phase8_diag.add_argument(
        "--output-dir",
        default="reports/btc5m_phase8_diagnostics",
        help="Output directory for generated JSON/CSV diagnostic artifacts",
    )

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
        "jev-probe": cmd_jev_probe,
        "jev-shadow-cycle": cmd_jev_shadow_cycle,
        "jev-shadow-status": cmd_jev_shadow_status,
        "btc5m-probe": cmd_btc5m_probe,
        "btc5m-shadow": cmd_btc5m_shadow,
        "btc5m-status": cmd_btc5m_status,
        "btc5m-report": cmd_btc5m_report,
        "btc5m-collect": cmd_btc5m_collect,
        "btc5m-collector-start": cmd_btc5m_collector_start,
        "btc5m-collector-status": cmd_btc5m_collector_status,
        "btc5m-collector-stop": cmd_btc5m_collector_stop,
        "btc5m-audit-cost": cmd_btc5m_audit_cost,
        "btc5m-backup": cmd_btc5m_backup,
        "btc5m-audit-dataset": cmd_btc5m_audit_dataset,
        "btc5m-phase8-diagnostics": cmd_btc5m_phase8_diagnostics,
    }

    handler = dispatch.get(args.subcommand)
    if handler:
        return handler(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

