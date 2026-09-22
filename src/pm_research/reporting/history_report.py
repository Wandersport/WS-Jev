"""Historical forecast evaluation reporting for prediction market research.

Renders terminal summaries and standalone HTML dashboards comparing the model
against historical prediction market baseline probabilities.
"""

from __future__ import annotations

import html
from pathlib import Path

from pm_research.calibration.market_baseline import MarketBaselineReport
from pm_research.utils import now_utc, to_iso_utc


class HistoryReportGenerator:
    """Renders research reports comparing the deterministic model against market baseline."""

    def __init__(self, report: MarketBaselineReport) -> None:
        self.report = report

    def generate_cli_report(self) -> str:
        """Produce a formatted terminal report with statistical significance tables."""
        lines: list[str] = []
        sep = "=" * 84
        subsep = "-" * 84

        lines.append(sep)
        lines.append("  [!] PAPER TRADING / RESEARCH ONLY - NO LIVE EXECUTION CAPABILITY")
        lines.append("      HISTORICAL PROBABILITY FORECAST BENCHMARK REPORT")
        lines.append(sep)

        lines.append("\n[EVALUATION CONTRACT & METHODOLOGY]")
        lines.append("  Execution Quality:       PRICE_ONLY (Historical settlement & transaction prices)")
        lines.append("  Simulated Paper P&L:     UNAVAILABLE (No order book / bid-ask depth in history)")
        lines.append("  Primary Benchmark:       Market Observed Price (The market is the benchmark)")
        lines.append("  Model Status:            FROZEN (No parameter tuning on historical data)")
        lines.append("  Temporal Integrity:      Strictly historical snapshots (zero lookahead)")
        lines.append("  Uncertainty Method:      Market-Clustered Bootstrap (1,000 iterations, 95% CI)")

        lines.append("\n[DATASET OVERVIEW]")
        lines.append(f"  Dataset Identifier:      {self.report.dataset_id}")
        lines.append(f"  Total Unique Markets:    {self.report.total_unique_markets}")
        lines.append(f"  Total Horizon Obs (N):   {self.report.total_observations}")
        lines.append(f"  Generated At (UTC):      {to_iso_utc(now_utc())}")

        ov = self.report.overall
        lines.append("\n[OVERALL FORECAST ACCURACY]")
        lines.append(f"  {'Metric':<22} | {'Market Baseline':<16} | {'Model (Ilsa)':<16} | {'Delta (Model - Mkt)':<24}")
        lines.append(subsep)
        brier_ci_str = f"[{ov.delta_brier_ci[0]:+.4f}, {ov.delta_brier_ci[1]:+.4f}]"
        ll_ci_str = f"[{ov.delta_log_loss_ci[0]:+.4f}, {ov.delta_log_loss_ci[1]:+.4f}]"
        lines.append(
            f"  {'Brier Score':<22} | {ov.market_brier:<16.4f} | {ov.model_brier:<16.4f} | {ov.delta_brier:+0.4f} {brier_ci_str}"
        )
        lines.append(
            f"  {'Binary Log Loss':<22} | {ov.market_log_loss:<16.4f} | {ov.model_log_loss:<16.4f} | {ov.delta_log_loss:+0.4f} {ll_ci_str}"
        )
        lines.append(
            f"  {'Forecast Bias':<22} | {ov.market_bias:<+16.4f} | {ov.model_bias:<+16.4f} | {ov.model_bias - ov.market_bias:+0.4f}"
        )
        lines.append(
            f"  {'ECE (10 bins)':<22} | {ov.market_ece:<16.4f} | {ov.model_ece:<16.4f} | {ov.model_ece - ov.market_ece:+0.4f}"
        )

        lines.append("\n[STANDARDIZED HORIZONS]")
        lines.append(
            f"  {'Horizon':<8} | {'Obs (N)':<7} | {'Mkts (M)':<8} | {'Mkt Brier':<9} | {'Mod Brier':<9} | {'Delta Brier':<11} | {'Delta LogLoss':<13}"
        )
        lines.append(subsep)
        for h_label, h_sum in self.report.by_horizon.items():
            if h_sum.observation_count == 0:
                continue
            lines.append(
                f"  {h_label:<8} | {h_sum.observation_count:<7} | {h_sum.market_count:<8} | "
                f"{h_sum.market_brier:<9.4f} | {h_sum.model_brier:<9.4f} | "
                f"{h_sum.delta_brier:<+11.4f} | {h_sum.delta_log_loss:<+13.4f}"
            )

        lines.append("\n[CHRONOLOGICAL HOLDOUT BREAKDOWN]")
        lines.append(
            f"  {'Split':<10} | {'Obs (N)':<7} | {'Mkts (M)':<8} | {'Mkt Brier':<9} | {'Mod Brier':<9} | {'Delta Brier':<11} | {'Delta LogLoss':<13}"
        )
        lines.append(subsep)
        for s_name, s_sum in self.report.by_split.items():
            if s_sum.observation_count == 0:
                continue
            lines.append(
                f"  {s_name.upper():<10} | {s_sum.observation_count:<7} | {s_sum.market_count:<8} | "
                f"{s_sum.market_brier:<9.4f} | {s_sum.model_brier:<9.4f} | "
                f"{s_sum.delta_brier:<+11.4f} | {s_sum.delta_log_loss:<+13.4f}"
            )

        if self.report.by_category:
            lines.append("\n[CATEGORY BREAKDOWN]")
            lines.append(
                f"  {'Category':<16} | {'Obs (N)':<7} | {'Mkts (M)':<8} | {'Mkt Brier':<9} | {'Mod Brier':<9} | {'Delta Brier':<11}"
            )
            lines.append(subsep)
            for c_name, c_sum in self.report.by_category.items():
                if c_sum.observation_count == 0:
                    continue
                lines.append(
                    f"  {c_name[:16]:<16} | {c_sum.observation_count:<7} | {c_sum.market_count:<8} | "
                    f"{c_sum.market_brier:<9.4f} | {c_sum.model_brier:<9.4f} | "
                    f"{c_sum.delta_brier:<+11.4f}"
                )

        # Statistical conclusion
        lines.append("\n[SCIENTIFIC CONCLUSION & INFERENCE]")
        ci_low, ci_high = ov.delta_brier_ci
        if ci_high < 0.0:
            conclusion = (
                "MODEL OUTPERFORMS MARKET: Delta Brier is strictly negative and the 95% "
                "market-clustered bootstrap confidence interval does NOT overlap zero."
            )
        elif ci_low > 0.0:
            conclusion = (
                "MARKET OUTPERFORMS MODEL: The raw market probability is strictly superior. "
                "The model introduces error relative to the market benchmark."
            )
        else:
            conclusion = (
                "NO STATISTICALLY SIGNIFICANT DIFFERENCE: The 95% confidence interval for Delta Brier "
                f"spans zero ({brier_ci_str}). The model's deterministic adjustments do not "
                "provide a statistically distinguishable edge over raw market pricing."
            )
        lines.append(f"  {conclusion}")
        lines.append(sep)

        return "\n".join(lines)

    def generate_html_report(self, output_path: str | Path) -> Path:
        """Produce a clean, standalone HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ov = self.report.overall

        horizon_rows: list[str] = []
        for h_label, h_sum in self.report.by_horizon.items():
            if h_sum.observation_count == 0:
                continue
            delta_color = "#16a34a" if h_sum.delta_brier < 0 else ("#dc2626" if h_sum.delta_brier > 0 else "#6b7280")
            horizon_rows.append(
                f"<tr>"
                f"<td style='font-weight:600;'>{html.escape(h_label)}</td>"
                f"<td>{h_sum.observation_count}</td>"
                f"<td>{h_sum.market_count}</td>"
                f"<td>{h_sum.market_brier:.4f}</td>"
                f"<td>{h_sum.model_brier:.4f}</td>"
                f"<td style='color:{delta_color}; font-weight:600;'>{h_sum.delta_brier:+.4f}</td>"
                f"<td>{h_sum.market_log_loss:.4f}</td>"
                f"<td>{h_sum.model_log_loss:.4f}</td>"
                f"<td style='color:{delta_color}; font-weight:600;'>{h_sum.delta_log_loss:+.4f}</td>"
                f"</tr>"
            )

        split_rows: list[str] = []
        for s_name, s_sum in self.report.by_split.items():
            if s_sum.observation_count == 0:
                continue
            delta_color = "#16a34a" if s_sum.delta_brier < 0 else ("#dc2626" if s_sum.delta_brier > 0 else "#6b7280")
            split_rows.append(
                f"<tr>"
                f"<td style='font-weight:600;'>{html.escape(s_name.upper())}</td>"
                f"<td>{s_sum.observation_count}</td>"
                f"<td>{s_sum.market_count}</td>"
                f"<td>{s_sum.market_brier:.4f}</td>"
                f"<td>{s_sum.model_brier:.4f}</td>"
                f"<td style='color:{delta_color}; font-weight:600;'>{s_sum.delta_brier:+.4f}</td>"
                f"<td>{s_sum.market_log_loss:.4f}</td>"
                f"<td>{s_sum.model_log_loss:.4f}</td>"
                f"<td style='color:{delta_color}; font-weight:600;'>{s_sum.delta_log_loss:+.4f}</td>"
                f"</tr>"
            )

        cat_rows: list[str] = []
        for c_name, c_sum in self.report.by_category.items():
            if c_sum.observation_count == 0:
                continue
            delta_color = "#16a34a" if c_sum.delta_brier < 0 else ("#dc2626" if c_sum.delta_brier > 0 else "#6b7280")
            cat_rows.append(
                f"<tr>"
                f"<td style='font-weight:600;'>{html.escape(c_name)}</td>"
                f"<td>{c_sum.observation_count}</td>"
                f"<td>{c_sum.market_count}</td>"
                f"<td>{c_sum.market_brier:.4f}</td>"
                f"<td>{c_sum.model_brier:.4f}</td>"
                f"<td style='color:{delta_color}; font-weight:600;'>{c_sum.delta_brier:+.4f}</td>"
                f"</tr>"
            )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Prediction Market Research - Historical Validation Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --warning: #f59e0b;
      --accent: #38bdf8;
      --success: #22c55e;
      --danger: #ef4444;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    .banner {{
      background: rgba(245, 158, 11, 0.15);
      border: 1px solid var(--warning);
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 24px;
      font-weight: 600;
      color: var(--warning);
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    h1, h2, h3 {{ margin-top: 0; }}
    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric-box {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
    }}
    .metric-label {{
      font-size: 0.85rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric-value {{
      font-size: 1.6rem;
      font-weight: 700;
      margin-top: 8px;
    }}
    .metric-sub {{
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 0.9rem;
    }}
    th {{
      background: rgba(15, 23, 42, 0.8);
      color: var(--text-muted);
      text-transform: uppercase;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
    }}
    tr:hover {{
      background: rgba(51, 65, 85, 0.3);
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.2);
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="banner">
      [!] STRICTLY PAPER TRADING / RESEARCH ONLY — NO LIVE EXECUTION CAPABILITY<br>
      <span style="font-size: 0.85rem; font-weight: normal; color: var(--text-muted);">
        Forecast Evaluation Mode (Price-Only History). Simulated P&L is unavailable due to absent historical order books.
      </span>
    </div>

    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 24px;">
      <div>
        <h1 style="margin-bottom: 4px;">Historical Forecast Benchmark Report</h1>
        <div style="color: var(--text-muted); font-size: 0.9rem;">
          Dataset: <span class="badge">{html.escape(self.report.dataset_id)}</span> |
          Generated: {to_iso_utc(now_utc())}
        </div>
      </div>
      <div>
        <span class="badge">Model Frozen (No Retraining)</span>
      </div>
    </div>

    <div class="grid-4">
      <div class="metric-box">
        <div class="metric-label">Unique Resolved Markets</div>
        <div class="metric-value">{self.report.total_unique_markets}</div>
        <div class="metric-sub">{self.report.total_observations} horizon observations</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Brier Score (Market vs Model)</div>
        <div class="metric-value">{ov.market_brier:.4f} <span style="font-size: 1rem; color: var(--text-muted);">vs</span> {ov.model_brier:.4f}</div>
        <div class="metric-sub">&Delta;Brier: {ov.delta_brier:+.4f} [{ov.delta_brier_ci[0]:+.4f}, {ov.delta_brier_ci[1]:+.4f}]</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Log Loss (Market vs Model)</div>
        <div class="metric-value">{ov.market_log_loss:.4f} <span style="font-size: 1rem; color: var(--text-muted);">vs</span> {ov.model_log_loss:.4f}</div>
        <div class="metric-sub">&Delta;LogLoss: {ov.delta_log_loss:+.4f} [{ov.delta_log_loss_ci[0]:+.4f}, {ov.delta_log_loss_ci[1]:+.4f}]</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Expected Calibration Error</div>
        <div class="metric-value">{ov.market_ece:.4f} <span style="font-size: 1rem; color: var(--text-muted);">vs</span> {ov.model_ece:.4f}</div>
        <div class="metric-sub">Bias: Mkt {ov.market_bias:+.4f} | Mod {ov.model_bias:+.4f}</div>
      </div>
    </div>

    <div class="card">
      <h2>Evaluation Across Standardized Horizons</h2>
      <p style="color: var(--text-muted); font-size: 0.85rem;">
        Snapshots are sampled strictly at or before t_resolve - horizon. Lookahead is impossible.
      </p>
      <table>
        <thead>
          <tr>
            <th>Horizon</th>
            <th>Obs (N)</th>
            <th>Mkts (M)</th>
            <th>Mkt Brier</th>
            <th>Mod Brier</th>
            <th>&Delta; Brier</th>
            <th>Mkt LogLoss</th>
            <th>Mod LogLoss</th>
            <th>&Delta; LogLoss</th>
          </tr>
        </thead>
        <tbody>
          {"".join(horizon_rows)}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Chronological Holdout Validation</h2>
      <p style="color: var(--text-muted); font-size: 0.85rem;">
        Markets partitioned chronologically by resolution timestamp (60% Dev, 20% Val, 20% Holdout).
      </p>
      <table>
        <thead>
          <tr>
            <th>Split</th>
            <th>Obs (N)</th>
            <th>Mkts (M)</th>
            <th>Mkt Brier</th>
            <th>Mod Brier</th>
            <th>&Delta; Brier</th>
            <th>Mkt LogLoss</th>
            <th>Mod LogLoss</th>
            <th>&Delta; LogLoss</th>
          </tr>
        </thead>
        <tbody>
          {"".join(split_rows)}
        </tbody>
      </table>
    </div>

    <div class="card">
      <h2>Category Slices</h2>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Obs (N)</th>
            <th>Mkts (M)</th>
            <th>Mkt Brier</th>
            <th>Mod Brier</th>
            <th>&Delta; Brier</th>
          </tr>
        </thead>
        <tbody>
          {"".join(cat_rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return path
