"""Research reporting for historical replay simulations and probability calibration."""

from __future__ import annotations

from pathlib import Path

from pm_research.replay.models import ReplayResult
from pm_research.utils import to_iso_utc


class ReplayReportGenerator:
    """Generates rigorous quantitative reports for historical replay runs."""

    DISCLAIMER = (
        "********************************************************************************\n"
        "[!] RESEARCH SIMULATION NOTICE: PAPER-TRADING AND HISTORICAL REPLAY ONLY.\n"
        "    ALL PERFORMANCE FIGURES ARE SIMULATED, HYPOTHETICAL, AND DERIVED UNDER\n"
        "    MODELED ASSUMPTIONS. THIS IS NOT LIVE EXECUTION, HAS NO FINANCIAL CLAIMS,\n"
        "    AND DOES NOT REPRESENT OR GUARANTEE FUTURE REAL-WORLD TRADING PERFORMANCE.\n"
        "********************************************************************************"
    )

    @classmethod
    def format_terminal_report(cls, result: ReplayResult) -> str:
        """Format an extensive terminal-ready quantitative replay report."""
        cal = result.calibration
        lines = [
            "",
            cls.DISCLAIMER,
            "=" * 80,
            f"  HISTORICAL REPLAY REPORT — {result.replay_id}",
            "=" * 80,
            "",
            "1. RUN METADATA & PROVENANCE",
            "-" * 80,
            f"  Dataset ID:              {result.dataset_id}",
            f"  Model Version:           {result.model_version}",
            f"  Git Commit SHA:          {result.git_commit_sha}",
            f"  Config Hash:             {result.config_hash[:16]}...",
            f"  Random Seed:             {result.random_seed}",
            f"  Simulated Latency:       {result.latency_seconds:.1f} seconds",
            f"  Simulated Start (UTC):   {to_iso_utc(result.simulated_start)}",
            f"  Simulated End (UTC):     {to_iso_utc(result.simulated_end)}",
            f"  Cycles Processed:        {result.cycles_count}",
            f"  Snapshots Processed:     {result.snapshots_count}",
            "",
            "2. STATISTICAL FORECAST CALIBRATION (PRIMARY RESEARCH GOAL)",
            "-" * 80,
            f"  Resolution Samples (N):  {cal.sample_size}",
            f"  Brier Score:             {cal.brier_score:.4f}  (0.0 = perfect, 0.25 = uninformative coin flip)",
            f"  Log Loss:                {cal.log_loss:.4f}",
            f"  Forecast Bias:           {cal.forecast_bias:+.4f}  (>0 = overconfident/optimistic, <0 = underconfident)",
            f"  Expected Calib. Error:   {cal.expected_calibration_error:.4f}  (ECE)",
            f"  Maximum Calib. Error:    {cal.maximum_calibration_error:.4f}  (MCE)",
            "",
        ]

        if cal.buckets:
            lines.extend([
                "  Calibration Buckets (Reliability Diagram):",
                "  " + "-" * 70,
                f"  {'Bin Range':<15} | {'Count':<6} | {'Mean Pred':<11} | {'Empirical Freq':<15} | {'Error':<8}",
                "  " + "-" * 70,
            ])
            for b in cal.buckets:
                if b.count > 0:
                    lines.append(
                        f"  [{b.bin_lower:.2f} - {b.bin_upper:.2f})  | {b.count:<6} | {b.mean_predicted:<11.4f} | {b.empirical_frequency:<15.4f} | {b.bin_error:<8.4f}"
                    )
            lines.append("  " + "-" * 70)
            lines.append("")

        if cal.by_category:
            lines.extend([
                "  Calibration by Category:",
                "  " + "-" * 70,
                f"  {'Category':<15} | {'Count':<6} | {'Brier':<8} | {'Mean Pred':<10} | {'Actual Rate':<12} | {'Bias':<8}",
                "  " + "-" * 70,
            ])
            for cat, data in sorted(cal.by_category.items()):
                lines.append(
                    f"  {cat:<15} | {data['count']:<6} | {data['brier_score']:<8.4f} | {data['mean_predicted']:<10.4f} | {data['empirical_rate']:<12.4f} | {data['bias']:<+8.4f}"
                )
            lines.append("  " + "-" * 70)
            lines.append("")

        lines.extend([
            "3. SIMULATED PAPER PORTFOLIO RESULTS",
            "-" * 80,
            f"  Initial Virtual Capital: ${result.initial_bankroll:,.2f}",
            f"  Final Virtual Equity:    ${result.final_equity:,.2f}",
            f"  Total Simulated P&L:     ${result.total_pnl:+,.2f}",
            f"  Simulated Return:        {result.simulated_return_pct:+.2f}%",
            f"  Maximum Drawdown:        {result.max_drawdown_pct:.2f}%",
            f"  Turnover:                ${result.turnover:,.2f}",
            f"  Total Slippage:          ${result.total_slippage:,.4f}",
            f"  Simulated Fees:          ${result.total_fees:,.4f}",
            f"  Open Positions at End:   {result.final_open_positions_count}",
            "",
            "4. EXECUTION ACTIVITY & SIZING FUNNEL",
            "-" * 80,
            f"  Proposals Generated:     {result.proposals_count}",
            f"  Approved by Bram:        {result.accepted_count}",
            f"  Rejected by Bram:        {result.rejected_count}",
            f"  Paper Fills Executed:    {result.fills_count}",
            f"  Average Raw Edge:        {result.avg_raw_edge:+.4f}",
            f"  Average Robust Edge:     {result.avg_robust_edge:+.4f}",
            "=" * 80,
            "",
        ])
        return "\n".join(lines)

    @classmethod
    def generate_html_report(cls, result: ReplayResult, output_path: str | Path) -> Path:
        """Generate a clean, standalone HTML report with tables and calibration diagnostics."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cal = result.calibration
        buckets_rows = ""
        for b in cal.buckets:
            if b.count > 0:
                buckets_rows += f"""
                <tr>
                    <td>[{b.bin_lower:.2f} - {b.bin_upper:.2f})</td>
                    <td>{b.count}</td>
                    <td>{b.mean_predicted:.4f}</td>
                    <td>{b.empirical_frequency:.4f}</td>
                    <td>{b.bin_error:.4f}</td>
                </tr>
                """

        category_rows = ""
        for cat, data in sorted(cal.by_category.items()):
            category_rows += f"""
            <tr>
                <td>{cat}</td>
                <td>{data['count']}</td>
                <td>{data['brier_score']:.4f}</td>
                <td>{data['mean_predicted']:.4f}</td>
                <td>{data['empirical_rate']:.4f}</td>
                <td>{data['bias']:+.4f}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Historical Replay Report — {result.replay_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            color: #1e293b;
            background-color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #ffffff;
            padding: 32px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}
        .banner {{
            background-color: #fee2e2;
            border-left: 4px solid #ef4444;
            color: #991b1b;
            padding: 16px;
            margin-bottom: 24px;
            font-size: 0.9em;
            font-weight: 500;
        }}
        h1, h2, h3 {{
            color: #0f172a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #f1f5f9;
            padding: 16px;
            border-radius: 6px;
        }}
        .card-title {{
            font-size: 0.85em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .card-value {{
            font-size: 1.6em;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 24px;
            font-size: 0.95em;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="banner">
            <strong>RESEARCH SIMULATION NOTICE:</strong> This report represents historical simulation and paper trading only.
            All metrics are calculated against historical or synthetic data under modeled assumptions.
            This system contains no live trading capability and these figures do not represent live financial performance.
        </div>

        <h1>Historical Replay Report</h1>
        <p><strong>Replay ID:</strong> {result.replay_id} | <strong>Dataset:</strong> {result.dataset_id} | <strong>Model:</strong> {result.model_version}</p>
        <p><strong>Simulated Span:</strong> {to_iso_utc(result.simulated_start)} to {to_iso_utc(result.simulated_end)} | <strong>Simulated Latency:</strong> {result.latency_seconds}s</p>

        <h2>Statistical Forecast Calibration</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Brier Score</div>
                <div class="card-value">{cal.brier_score:.4f}</div>
            </div>
            <div class="card">
                <div class="card-title">Log Loss</div>
                <div class="card-value">{cal.log_loss:.4f}</div>
            </div>
            <div class="card">
                <div class="card-title">Forecast Bias</div>
                <div class="card-value">{cal.forecast_bias:+.4f}</div>
            </div>
            <div class="card">
                <div class="card-title">ECE (Calibration Error)</div>
                <div class="card-value">{cal.expected_calibration_error:.4f}</div>
            </div>
        </div>

        <h3>Reliability Diagram / Calibration Buckets</h3>
        <table>
            <thead>
                <tr>
                    <th>Predicted Probability Bin</th>
                    <th>Count</th>
                    <th>Mean Predicted</th>
                    <th>Empirical Frequency</th>
                    <th>Calibration Error</th>
                </tr>
            </thead>
            <tbody>
                {buckets_rows}
            </tbody>
        </table>

        <h2>Simulated Paper Portfolio Results</h2>
        <div class="grid">
            <div class="card">
                <div class="card-title">Final Equity</div>
                <div class="card-value">${result.final_equity:,.2f}</div>
            </div>
            <div class="card">
                <div class="card-title">Simulated Return</div>
                <div class="card-value">{result.simulated_return_pct:+.2f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Max Drawdown</div>
                <div class="card-value">{result.max_drawdown_pct:.2f}%</div>
            </div>
            <div class="card">
                <div class="card-title">Paper Fills Executed</div>
                <div class="card-value">{result.fills_count}</div>
            </div>
        </div>

        <h3>Execution Activity & Sizing</h3>
        <table>
            <tr><td>Initial Capital</td><td>${result.initial_bankroll:,.2f}</td></tr>
            <tr><td>Total Simulated P&L</td><td>${result.total_pnl:+,.2f}</td></tr>
            <tr><td>Simulated Turnover</td><td>${result.turnover:,.2f}</td></tr>
            <tr><td>Total Slippage</td><td>${result.total_slippage:,.4f}</td></tr>
            <tr><td>Simulated Fees</td><td>${result.total_fees:,.4f}</td></tr>
            <tr><td>Trade Proposals</td><td>{result.proposals_count}</td></tr>
            <tr><td>Accepted by Bram</td><td>{result.accepted_count}</td></tr>
            <tr><td>Rejected by Bram</td><td>{result.rejected_count}</td></tr>
        </table>

        <h2>Calibration by Category</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Brier Score</th>
                    <th>Mean Predicted</th>
                    <th>Actual Rate</th>
                    <th>Bias</th>
                </tr>
            </thead>
            <tbody>
                {category_rows}
            </tbody>
        </table>

        <footer style="margin-top: 40px; font-size: 0.8em; color: #94a3b8; text-align: center;">
            Generated by Wandersport PM Research Replay Engine | Git Commit: {result.git_commit_sha}
        </footer>
    </div>
</body>
</html>
"""
        with open(out, "w", encoding="utf-8") as f:
            f.write(html_content)

        return out
