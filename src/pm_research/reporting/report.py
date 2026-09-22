"""Reporting and static HTML dashboard generator with prominent paper-only indicators."""

from __future__ import annotations

from pathlib import Path

from pm_research.calibration.metrics import CalibrationEngine
from pm_research.config import SystemConfig
from pm_research.storage.db import Database
from pm_research.utils import now_utc, to_iso_utc


class ReportGenerator:
    """Generates CLI terminal reports and static HTML dashboards for paper trading research."""

    def __init__(self, config: SystemConfig, db: Database) -> None:
        self.config = config
        self.db = db
        self.calib_engine = CalibrationEngine()

    def generate_cli_report(self) -> str:
        """Produce a clean, formatted text report for terminal output."""
        snap = self.db.get_latest_portfolio_snapshot()
        activity = self.db.get_activity_summary()
        obs = self.db.get_all_calibration_observations()
        calib = self.calib_engine.compute_metrics(obs)
        positions = self.db.get_open_positions()

        lines: list[str] = []
        sep = "=" * 78
        subsep = "-" * 78

        lines.append(sep)
        lines.append("  [!] PAPER TRADING / SIMULATION ONLY - NO LIVE EXECUTION CAPABILITY")
        lines.append("      Quantitative Prediction Market Research System")
        lines.append(sep)

        # Operations
        lines.append("\n[OPERATIONS & RESEARCH CYCLES]")
        lines.append(f"  Total Cycles Completed:  {activity['cycles']}")
        lines.append(f"  Last Cycle ID:           {activity['last_cycle_id'] or 'N/A'}")
        lines.append(f"  Last Cycle UTC:          {activity['last_cycle_time'] or 'N/A'}")
        lines.append(f"  Active Model Version:    {self.config.model_version}")

        # Portfolio
        lines.append("\n[PORTFOLIO STATE]")
        if snap:
            lines.append(f"  Virtual Cash:            ${snap.virtual_cash:,.2f}")
            lines.append(f"  Positions Value:         ${snap.positions_value:,.2f}")
            lines.append(f"  Total Equity:            ${snap.equity:,.2f}")
            lines.append(f"  Realized P&L:            ${snap.realized_pnl:+,.2f}")
            lines.append(f"  Unrealized P&L:          ${snap.unrealized_pnl:+,.2f}")
            lines.append(f"  Total P&L:               ${snap.total_pnl:+,.2f}")
            lines.append(f"  High-Water Mark:         ${snap.high_water_mark:,.2f}")
            lines.append(f"  Current Drawdown:        {snap.current_drawdown * 100:.2f}%")
            lines.append(f"  Max Peak-to-Trough DD:   {snap.max_drawdown * 100:.2f}%")
            lines.append(f"  Total Exposure:          ${snap.total_exposure:,.2f}")
            lines.append(f"  Risk State:              {snap.risk_state.value}")
        else:
            lines.append(f"  Initial Bankroll:        ${self.config.initial_bankroll:,.2f}")
            lines.append("  (No portfolio snapshots recorded yet)")

        # Paper activity
        lines.append("\n[PAPER ACTIVITY SUMMARY]")
        lines.append(f"  Total Trade Proposals:   {activity['proposals']}")
        lines.append(f"  Bram Accepted:           {activity['accepted']}")
        lines.append(f"  Bram Rejected:           {activity['rejected']}")
        lines.append(f"  Simulated Paper Fills:   {activity['fills']}")
        lines.append(f"  Cumulative Slippage:     ${activity['total_slippage']:.4f}")
        lines.append(f"  Open Paper Positions:    {len(positions)}")

        # Calibration
        lines.append("\n[FORECAST CALIBRATION METRICS]")
        lines.append(f"  Resolved Sample Size:    {calib.sample_size}")
        if calib.sample_size > 0:
            lines.append(f"  Overall Brier Score:     {calib.brier_score:.4f} (lower is better, 0.0 is perfect)")
            lines.append(f"  Overall Log Loss:        {calib.log_loss:.4f}")
            lines.append(f"  Forecast Bias:           {calib.forecast_bias:+.4f} (positive = overconfident)")
            lines.append(f"  Expected Calib Error:    {calib.expected_calibration_error:.4f} (ECE)")
            lines.append(f"  Max Calibration Error:   {calib.maximum_calibration_error:.4f} (MCE)")

            lines.append("\n  Calibration Curve Bins:")
            lines.append("  " + subsep[:65])
            lines.append("  Bin Range      | Count | Mean Predicted | Empirical Rate | Error")
            lines.append("  " + subsep[:65])
            for b in calib.buckets:
                if b.count > 0:
                    lines.append(
                        f"  [{b.bin_lower:.2f}, {b.bin_upper:.2f}) | {b.count:5d} | "
                        f"{b.mean_predicted:14.4f} | {b.empirical_frequency:14.4f} | {b.bin_error:6.4f}"
                    )
            lines.append("  " + subsep[:65])
        else:
            lines.append("  (Awaiting market resolutions to compute calibration metrics)")

        lines.append("\n" + sep)
        lines.append("  END REPORT - PAPER TRADING RESEARCH SYSTEM")
        lines.append(sep + "\n")

        return "\n".join(lines)

    def generate_html_dashboard(self, output_path: str = "reports/dashboard.html") -> Path:
        """Produce an interactive, modern HTML dashboard file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        snap = self.db.get_latest_portfolio_snapshot()
        activity = self.db.get_activity_summary()
        obs = self.db.get_all_calibration_observations()
        calib = self.calib_engine.compute_metrics(obs)
        all_positions = self.db.get_all_positions()

        # Precompute strings for HTML template
        equity_str = f"${snap.equity:,.2f}" if snap else f"${self.config.initial_bankroll:,.2f}"
        cash_str = f"${snap.virtual_cash:,.2f}" if snap else f"${self.config.initial_bankroll:,.2f}"
        realized_pnl_str = f"${snap.realized_pnl:+,.2f}" if snap else "$0.00"
        pnl_class = "positive" if (snap and snap.realized_pnl >= 0) else "negative"
        drawdown_str = f"{snap.current_drawdown * 100:.2f}%" if snap else "0.00%"
        dd_class = "negative" if (snap and snap.current_drawdown > 0.05) else ""
        risk_state_str = snap.risk_state.value if snap else "NORMAL"
        brier_str = f"{calib.brier_score:.4f}" if calib.sample_size > 0 else "N/A"
        ece_str = f"{calib.expected_calibration_error:.4f}" if calib.sample_size > 0 else "N/A"
        bias_str = f"{calib.forecast_bias:+.4f}" if calib.sample_size > 0 else "N/A"

        # Build calibration table rows
        calib_rows = ""
        for b in calib.buckets:
            if b.count > 0:
                calib_rows += f"""
                <tr>
                    <td>[{b.bin_lower:.2f}, {b.bin_upper:.2f})</td>
                    <td>{b.count}</td>
                    <td>{b.mean_predicted:.4f}</td>
                    <td>{b.empirical_frequency:.4f}</td>
                    <td>{b.bin_error:.4f}</td>
                </tr>
                """

        # Build position table rows
        pos_rows = ""
        for p in all_positions[:25]:
            status_badge = (
                '<span class="badge badge-open">OPEN</span>'
                if p.status == "OPEN"
                else '<span class="badge badge-closed">CLOSED</span>'
            )
            pnl = p.realized_pnl if p.status == "CLOSED" else p.unrealized_pnl
            pos_pnl_class = "positive" if pnl >= 0 else "negative"
            pos_rows += f"""
            <tr>
                <td><code>{p.market_id}</code></td>
                <td><strong>{p.side.value}</strong></td>
                <td>{p.quantity:.1f}</td>
                <td>${p.average_entry_price:.3f}</td>
                <td>${p.current_price:.3f}</td>
                <td class="{pos_pnl_class}">${pnl:+.2f}</td>
                <td>{p.category}</td>
                <td>{status_badge}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Market Research Dashboard</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #38bdf8;
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            margin: 0;
            padding: 24px;
        }}
        .banner {{
            background: linear-gradient(90deg, #b91c1c, #991b1b);
            color: white;
            padding: 16px 24px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1.1rem;
            text-align: center;
            letter-spacing: 0.05em;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
        }}
        .header {{
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 1.8rem;
        }}
        .header p {{
            margin: 0;
            color: var(--text-secondary);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }}
        .card-title {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}
        .card-value {{
            font-size: 1.6rem;
            font-weight: 700;
        }}
        .positive {{ color: var(--success); }}
        .negative {{ color: var(--danger); }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-open {{ background: rgba(56, 189, 248, 0.2); color: var(--accent); }}
        .badge-closed {{ background: rgba(148, 163, 184, 0.2); color: var(--text-secondary); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
        }}
        .section-title {{
            font-size: 1.2rem;
            font-weight: 600;
            margin: 32px 0 16px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        code {{
            background: rgba(0,0,0,0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: ui-monospace, monospace;
        }}
    </style>
</head>
<body>
    <div class="banner">
        [!] PAPER TRADING / SIMULATION ONLY &mdash; NO LIVE EXECUTION CAPABILITY
    </div>

    <div class="header">
        <h1>Quantitative Research Pipeline: Rigo &rarr; Holt &rarr; Ilsa &rarr; Kett &rarr; Bram &rarr; Tess</h1>
        <p>Audit trail and calibration report generated at {to_iso_utc(now_utc())}</p>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Total Equity</div>
            <div class="card-value">{equity_str}</div>
        </div>
        <div class="card">
            <div class="card-title">Virtual Cash</div>
            <div class="card-value">{cash_str}</div>
        </div>
        <div class="card">
            <div class="card-title">Realized P&L</div>
            <div class="card-value {pnl_class}">{realized_pnl_str}</div>
        </div>
        <div class="card">
            <div class="card-title">Current Drawdown</div>
            <div class="card-value {dd_class}">{drawdown_str}</div>
        </div>
        <div class="card">
            <div class="card-title">Risk State</div>
            <div class="card-value" style="color: var(--accent);">{risk_state_str}</div>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Brier Score</div>
            <div class="card-value">{brier_str}</div>
        </div>
        <div class="card">
            <div class="card-title">Expected Calib Error (ECE)</div>
            <div class="card-value">{ece_str}</div>
        </div>
        <div class="card">
            <div class="card-title">Forecast Bias</div>
            <div class="card-value">{bias_str}</div>
        </div>
        <div class="card">
            <div class="card-title">Resolved Forecasts</div>
            <div class="card-value">{calib.sample_size}</div>
        </div>
        <div class="card">
            <div class="card-title">Simulated Slippage</div>
            <div class="card-value">${activity['total_slippage']:.4f}</div>
        </div>
    </div>

    <div class="section-title">Probability Calibration Buckets</div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Bucket Range</th>
                    <th>Observations</th>
                    <th>Mean Predicted q̂</th>
                    <th>Empirical Outcome Frequency</th>
                    <th>Calibration Error |q̂ - y|</th>
                </tr>
            </thead>
            <tbody>
                {calib_rows if calib_rows else '<tr><td colspan="5" style="text-align: center; color: var(--text-secondary);">No resolved calibration observations yet.</td></tr>'}
            </tbody>
        </table>
    </div>

    <div class="section-title">Paper Positions Audit Trail</div>
    <div class="card">
        <table>
            <thead>
                <tr>
                    <th>Market ID</th>
                    <th>Side</th>
                    <th>Contracts</th>
                    <th>Entry Price</th>
                    <th>Mark Price</th>
                    <th>P&L</th>
                    <th>Category</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {pos_rows if pos_rows else '<tr><td colspan="8" style="text-align: center; color: var(--text-secondary);">No positions recorded yet.</td></tr>'}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        out.write_text(html_content, encoding="utf-8")
        return out
