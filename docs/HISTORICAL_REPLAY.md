# Historical Replay & Quantitative Validation Engine

> **RESEARCH NOTICE: PAPER TRADING ONLY**
> All historical replay simulations are executed locally using hypothetical paper portfolios. The system contains **zero live execution capability**, no wallets, and no exchange order placement endpoints.

---

## 1. Overview & Objectives

The Historical Replay subsystem provides a deterministic, rigorous backtesting and simulation environment for quantitative research on prediction markets. It enables:

1. **Temporal Integrity**: Guarantees that no future market data or resolutions leak into decision pipelines.
2. **Forecast Calibration Slicing**: Evaluates the true statistical calibration (Brier score, binary log loss, ECE, MCE) of probability forecasts independently of trading returns.
3. **Execution Latency Simulation**: Models decision-to-fill latency ($0\text{s} \to N\text{s}$) where approved orders fill against future, updated order-book depth, realistically capturing price drift and adverse selection.
4. **Reproducibility & Audit Trail**: Every replay run is cryptographically bound to dataset SHA256 checksums, system configuration hashes, Git commit SHAs, and random seeds.

---

## 2. Simulated Clock & Temporal Progression

Historical replay operates under a dedicated, deterministic `SimulatedClock`:

- **UTC Enforced**: Every timestamp is explicitly timezone-aware in UTC (`datetime.timezone.utc`).
- **Strict Monotonicity**: Time only advances forward ($t_{k+1} \ge t_k$). Any attempt to move the clock backwards triggers a fatal `ValueError`.
- **Chronological Cycles**: Market snapshots are streamed in strictly sorted timestamp batches. All decision stages (`Holt` $\to$ `Ilsa` $\to$ `Kett` $\to$ `Bram`) observe only snapshots valid at $t_k$.

### Look-Ahead Bias Prevention

In real prediction markets, event resolutions ($Y \in \{0, 1\}$) occur at specific timestamps $t_{\text{resolve}}$. The replay engine enforces:
- Event resolutions are loaded into a chronologically isolated queue.
- No resolution is revealed, processed, or settled until `clock.now() >= resolved_at`.
- Positions remain open and marked to market until that exact simulated point in time.

---

## 3. Execution Latency Model

In fast or low-liquidity prediction markets, instantaneous execution is an unrealistic assumption. The replay engine introduces an explicit `latency_seconds` parameter:

```
t_k: Bram approves TradeProposal
      │
      ├─ latency == 0 ──► Instant PaperBroker fill at t_k snapshot depth
      │
      └─ latency > 0  ──► Order queued with earliest_fill_time = t_k + latency
                           │
                           ▼
                          Simulated clock advances to t_(k+m) >= earliest_fill_time
                           │
                           ▼
                          Attempt fill against fresh snapshot depth at t_(k+m)
```

### Consequences of Modeled Latency:
1. **Price Drift & Slippage**: If market price has advanced in the direction of the trade between $t_k$ and $t_{k+m}$, the fill price degrades (higher slippage).
2. **Unfilled Orders**: If the market spread widened beyond tolerance or the market resolved prior to $t_{k+m}$, the queued order expires unfilled.
3. **Realistic P&L Degradation**: Compares zero-latency theoretical returns against realistic execution latency, revealing whether strategy edge survives execution delay.

---

## 4. Dataset Management & Integrity

Historical replay datasets are structured as immutable directories containing:

```text
data/datasets/<dataset_id>/
├── manifest.json       # Metadata, timestamps, counts, and SHA256 checksum
├── snapshots.jsonl     # Chronologically sorted point-in-time market snapshots
└── resolutions.jsonl   # Chronologically sorted event settlement records
```

### Manifest Schema (`DatasetManifest`):
- `dataset_id`: Unique alphanumeric dataset identifier.
- `source`: Provenance tag (e.g. `synthetic_generator`, `polymarket_public_archive`).
- `start_time` / `end_time`: ISO UTC time boundaries.
- `market_count`, `snapshot_count`, `resolution_count`: Cardinality counts.
- `checksum_sha256`: Cryptographic SHA256 hash of `snapshots.jsonl` verifying dataset integrity.
- `is_synthetic`: Boolean distinguishing synthetic fixtures from real public archives.

---

## 5. Statistical Calibration vs Portfolio Accounting

The replay engine cleanly bifurcates analysis into two distinct categories:

### 1. Forecast Quality (Primary Quantitative Goal)
- **Brier Score**: $\frac{1}{N}\sum (\hat{q}_i - y_i)^2 \in [0, 1]$ (0.0 = perfect; 0.25 = uninformative baseline).
- **Binary Log Loss**: $-\frac{1}{N}\sum [y_i \ln(\hat{q}_i) + (1-y_i) \ln(1-\hat{q}_i)]$.
- **Forecast Bias**: Mean predicted probability minus empirical base rate ($\bar{q} - \bar{y}$).
- **Expected Calibration Error (ECE)**: Weighted average calibration error across probability bins.
- **Maximum Calibration Error (MCE)**: Worst-case deviation across reliability diagram bins.
- **Sliced Breakdowns**: Evaluated by market category, model version, and risk state.

### 2. Simulated Paper Portfolio Performance
- Virtual capital growth: Initial virtual bankroll vs. Final virtual equity.
- Max peak-to-trough drawdown percentage.
- Virtual turnover, simulated slippage, and modeled fees.
- Execution sizing funnel: Proposals $\to$ Bram approvals $\to$ Bram rejections $\to$ Paper fills.

---

## 6. CLI Commands

```bash
# 1. List registered replay datasets with checksums
uv run pmr datasets

# 2. Run historical replay simulation (zero latency)
uv run pmr replay --dataset synthetic_benchmark_v1 --latency 0.0

# 3. Run historical replay with 12-hour execution latency and HTML report export
uv run pmr replay --dataset synthetic_benchmark_v1 --latency 43200 --html-out reports/replay_run.html

# 4. Inspect recorded replay run from database
uv run pmr replay-report --replay-id <replay-id>
```
