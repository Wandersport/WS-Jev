# Prediction Market Research System (`pm-research`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Paper Trading Only](https://img.shields.io/badge/trading-paper--only-green.svg)](docs/SAFETY.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **IMPORTANT NOTICE: PAPER TRADING AND QUANTITATIVE RESEARCH ONLY.**
> This repository is **structurally incapable of live trading**. It contains no wallet integration, no private key handling, no transaction signing, and no live order execution endpoints. All orders and fills are simulated locally.
> Do not attempt to use this repository for real-money execution.

---

## Overview

`pm-research` is an offline-capable, reproducible quantitative research platform for binary prediction markets. Inspired by the conceptual pipeline:

$$\mathbf{Rigo} \longrightarrow \mathbf{Holt} \longrightarrow \mathbf{Ilsa} \longrightarrow \mathbf{Kett} \longrightarrow \mathbf{Bram} \longrightarrow \mathbf{PaperBroker} \longrightarrow \mathbf{Tess}$$

The system prioritizes:
1. **Correctness & Reproducibility**: Deterministic seeds, UTC-normalized timestamps, and complete SQLite audit logging.
2. **Safety & Capital Preservation**: Conservative fractional Kelly sizing, authoritative Bram risk gating, and peak-to-trough drawdown survival state machines.
3. **Probability Calibration**: Rigorous evaluation using Brier scores, binary log loss, and Expected Calibration Error (ECE) rather than superficial win rates.
4. **Zero Live Trading Capability**: Structural enforcement preventing live execution, proven by automated AST safety tests.

---

## Conceptual Pipeline

| Component | Role | Function |
|:---|:---|:---|
| **Rigo** | Market Ingestor | Ingests market feeds, checks quote validity (`bid <= ask`, price bounds), rejects stale data, and normalizes to UTC domain models. |
| **Holt** | Feature Extractor | Derives momentum, order book imbalance, spread percentages, liquidity metrics, and time-to-resolution with full provenance. |
| **Ilsa** | Probability Engine | Estimates calibrated probability $\hat{q} = P(\text{YES})$ via logit baseline and bounded feature adjustments, with explicit uncertainty $\sigma_q$. |
| **Kett** | Edge & Sizer | Detects executable raw edge, computes robust edge ($\text{raw} - z\sigma_q - \text{costs}$), and calculates conservative fractional Kelly size. |
| **Bram** | Authoritative Risk Gate | Evaluates hard limits (liquidity, spread, concentration, cash, exposure, and survival state); accepts or rejects with stable reason codes. |
| **PaperBroker** | Execution Simulator | The **sole execution engine** in the system. Walks order-book depth or applies conservative fallback slippage models. Paper-only. |
| **Tess** | Paper Portfolio Manager | Tracks virtual cash, mark-to-market positions, realized/unrealized P&L, high-water marks, drawdowns, and calibration observations. |

---

## Quickstart & Commands

This project uses modern Python tooling via **`uv`**.

### 1. Installation & Environment Setup

```bash
uv sync
```

### 2. Run Automated Safety Verification

Verify that no private keys, wallets, blockchain signing, or live broker classes exist in the codebase:

```bash
uv run pmr verify-safety
```

### 3. Run the Deterministic Synthetic Demo

Seed a multi-cycle scenario featuring varied market conditions, Bram rejections, simulated fills, market resolutions, and calibration analytics:

```bash
uv run pmr seed-demo
```

### 4. Run Research Cycles

Execute a single research pipeline cycle (offline fixtures or public read-only market data):

```bash
uv run pmr run-once
```

Run a multi-cycle simulation loop:

```bash
uv run pmr run-loop --cycles 3 --interval 1.0
```

### 5. Historical Replay & Quantitative Backtesting

Run deterministic multi-period simulations with simulated clock and execution latency:

```bash
# List available datasets and verify checksums
uv run pmr datasets

# Run replay with 0-second execution latency
uv run pmr replay --dataset synthetic_benchmark_v1 --latency 0.0

# Run replay with 12-hour latency and export HTML report
uv run pmr replay --dataset synthetic_benchmark_v1 --latency 43200 --html-out reports/replay_benchmark.html

# Inspect a recorded replay run
uv run pmr replay-report --replay-id <replay-id>
```

### 6. Inspect Paper Portfolio & Calibration

Display current portfolio state (virtual cash, equity, drawdown, open positions):

```bash
uv run pmr portfolio
```

Display forecast calibration metrics (Brier score, log loss, ECE, calibration bins):

```bash
uv run pmr calibration
```

### 7. Generate Reports & Static Dashboard

Generate a terminal summary and an interactive static HTML dashboard:

```bash
uv run pmr report --dashboard-out reports/dashboard.html
```

Open `reports/dashboard.html` in any web browser to view portfolio charts, calibration tables, and execution logs.

### 8. Run Test Suite

Run the full pytest suite (unit, risk, survival, PaperBroker, replay, latency, and integration tests):

```bash
uv run pytest
```

### 9. Run Linting and Code Quality Checks

```bash
uv run ruff check src tests
```

---

## Project Structure

```text
├── config/
│   └── default_config.json        # Reference system parameters
├── data/
│   └── datasets/                  # Immutable replay datasets with manifests & SHA256
│       └── synthetic_benchmark/   # Deterministic multi-period benchmark
├── docs/
│   ├── ARCHITECTURE.md            # Detailed pipeline component breakdown
│   ├── CALIBRATION.md             # Calibration math: Brier, log loss, ECE
│   ├── HISTORICAL_REPLAY.md       # Replay engine, simulated clock, & latency models
│   ├── RESEARCH_NOTES.md          # Market microstructure & research limitations
│   └── SAFETY.md                  # Non-negotiable paper-only safety policy
├── src/
│   └── pm_research/
│       ├── calibration/           # Brier score, log loss, ECE, calibration bins
│       ├── data/                  # Synthetic fixtures, dataset importer, & public adapter
│       ├── domain/                # Typed domain models (Market, OrderBook, Proposals, Fills)
│       ├── execution/             # PaperBroker: sole paper-only execution simulator
│       ├── pipeline/              # Rigo, Holt, Ilsa, Kett, Bram, and PipelineRunner
│       ├── portfolio/             # Tess paper portfolio and SurvivalManager state machine
│       ├── replay/                # Simulated clock, ReplayEngine, models, & reports
│       ├── reporting/             # CLI report formatter and HTML dashboard generator
│       ├── safety/                # AST and token safety verifier engine
│       ├── storage/               # Transactional SQLite audit logging
│       ├── cli.py                 # CLI entry point (seed-demo, replay, run-once, report)
│       ├── config.py              # Validated configuration dataclasses
│       └── utils.py               # Timezone-aware UTC helpers and hashing
├── tests/
│   ├── test_accounting_invariants.py # Strict virtual cash & equity accounting
│   ├── test_core_integration.py   # Core Rigo -> Ilsa -> Kett -> Bram -> Broker -> Tess
│   ├── test_full_pipeline_integration.py # Multi-cycle lifecycle & dashboard test
│   ├── test_paper_broker.py       # Order-book walking & slippage simulation tests
│   ├── test_portfolio_and_settlement.py # Marking, cash accounting, & settlement
│   ├── test_replay_dataset.py     # Manifests, SHA256 checksums, & dataset import
│   ├── test_replay_engine.py      # Replay engine, lookahead isolation, & latency impact
│   ├── test_replay_reporting.py   # Replay report generation & simulation warnings
│   ├── test_risk_and_survival.py  # Bram gates, drawdown states, & monotonic risk reduction
│   ├── test_safety.py             # Verification of zero live trading / zero wallets
│   ├── test_simulated_clock.py    # Deterministic UTC progression & monotonic time
│   └── test_units.py              # Math, logit/sigmoid, UTC, and calibration formulas
├── pyproject.toml                 # Modern package definition and script mappings
└── README.md
```

---

## Risk & Survival State Machine

To enforce strict capital preservation, the system monitors peak-to-trough equity drawdown:

$$\text{Drawdown} = \frac{\text{High Water Mark} - \text{Equity}}{\text{High Water Mark}}$$

The system transitions across five deterministic states:

```
[ NORMAL ] (Drawdown < 5%)
    │
    ▼
[ CAUTION ] (Drawdown 5% - 10%): Kelly scale 0.70x, Min edge 1.25x
    │
    ▼
[ SURVIVAL ] (Drawdown 10% - 20%): Kelly scale 0.40x, Min edge 1.75x
    │
    ▼
[ CRITICAL ] (Drawdown 20% - 30%): Kelly scale 0.15x, Min edge 2.50x
    │
    ▼
[ HALTED ] (Drawdown >= 30%): New positions completely blocked (Kelly = 0.0x)
```

**Monotonic Risk Reduction**: As drawdown worsens, Kelly multiplier and maximum position sizes strictly decrease, while required robust edge strictly increases. Risk never increases during losing streaks.

---

## Research Limitations & Disclaimers

1. **Simulated vs. Real Fills**: Real prediction markets suffer from severe adverse selection, queue priority dynamics, and transient order-book depth. Realized live performance will always underperform paper simulation.
2. **Oracle & Resolution Risk**: Real markets depend on centralized or decentralized governance (e.g., UMA, Polymarket resolvers). Ambiguity in question interpretation introduces catastrophic non-diversifiable risk.
3. **Capacity Constraints**: Prediction market liquidity is thin. Strategies that appear viable with $1,000 cannot scale to larger asset pools without inducing massive price impact.
4. **Extreme Return Claims**: Claims of turning $50 into thousands of dollars within hours represent high-variance gambling paths with near-certain long-term ruin ($P(\text{ruin}) \to 1$). They are rejected as an engineering or research objective.

---

## License

MIT License. Designed strictly for quantitative research and educational simulation.
