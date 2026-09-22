# External Reference Review: Jev / Poly Crypto Demo

```text
REFERENCE_REPOSITORY: https://github.com/frankda/jev-poly-crypto-demo
REVIEWED_COMMIT:      6d73b1b52cef3a8bf61e8a260ff5e7e4857af296
SOURCE_CODE_COPIED:   NO (Strictly Zero Source Code or Comments Copied)
REVIEW_PURPOSE:       Architectural & Quantitative Research Reference Only
LEGAL_NOTICE:         The reference repository contains no explicit LICENSE file at the root.
                      In strict compliance with intellectual property standards, WS-Jev has
                      not vendored, copied, or translated any source files or code blocks.
                      All components in WS-Jev are independently authored in Python.
```

---

## 1. Context & Review Scope

The external repository `frankda/jev-poly-crypto-demo` was examined at commit `6d73b1b52cef3a8bf61e8a260ff5e7e4857af296` to understand how high-frequency 5-minute prediction markets operate on Polymarket and how TypeSafe Jev is framed for binary crypto outcomes.

The review covered all core modules:
- `README.md`: System overview, round derivation, settlement rules, fee modeling, parameter defaults.
- `src/model.ts`: Jev input state formatting, question schema, class probability normalization, mock heuristics.
- `src/binance.ts`: Binance USDⓈ-M BTCUSDT perpetual WebSocket integration (`depth20@100ms`, `aggTrade`), book imbalance, microprice, taker flow windows.
- `src/polymarket.ts`: Gamma API event discovery, CLOB order-book parsing, Chainlink RTDS WebSocket consumption (`crypto_prices_chainlink`, `crypto_prices_twap_sixty`), settlement resolution lookup.
- `src/engine.ts`: Event loop, timing cadence, observation guards, inference lifecycle, post-inference quote refresh.
- `src/policy.ts`: Simulated order fill walking, entry edge evaluation, exit holding rules, daily loss limit guards.

---

## 2. Comprehensive Concept Matrix

| Architectural / Research Concept | Evaluation Status | WS-Jev Decision & Methodological Rationale |
| :--- | :--- | :--- |
| **Exact 5-Minute Round Slug Derivation** | **Adopted** | `btc-updown-5m-{epoch_start}` where `epoch_start = (now // 300) * 300` accurately maps to Polymarket's Gamma slug structure. |
| **300-Second Duration Invariant** | **Adopted** | Verify that `endMs - startMs == 300000`. Any round with deviating duration is rejected to prevent non-standard market distortion. |
| **Outcome Label Mapping by Name** | **Adopted** | Map outcome token IDs strictly by matching outcome labels `"Up"` and `"Down"`. Never assume positional index (e.g. index 0 is not guaranteed to be Up). |
| **Authoritative Price-to-Beat Hierarchy** | **Adopted** | Prefer `eventMetadata.priceToBeat` from Polymarket Gamma API. Fall back strictly to exact Chainlink tick where `timestamp == startMs`. If no exact anchor is available, skip round (`SKIP_NO_EXACT_ANCHOR`). Never approximate. |
| **Settlement Source Separation** | **Adopted** | Read `resolutionSource` from metadata (e.g., Chainlink TWAP 60s). Binance is strictly treated as an auxiliary feature feed, never as a settlement source. |
| **Binance Perp Microstructure Indicators** | **Adopted** | Top-5 and Top-20 depth imbalance, microprice offset from midpoint, and windowed taker flow imbalance (10s, 30s, 60s) provide standard microstructure features. |
| **Incomplete Taker Window -> None** | **Adopted** | When feed uptime is shorter than the requested flow window (e.g., 60s), report `None` rather than zero. Incomplete data is unknown, not zero. |
| **Cross-Source Basis Tracking** | **Adopted** | Track basis between Binance perpetual midpoint and observed Chainlink reference: `(binance_mid / chainlink_ref - 1) * 10000` (bps). |
| **Strict Data Staleness Guards** | **Adopted** | Define configurable staleness thresholds (`max_data_age_ms = 15000`). If reference price or order books exceed age limit, skip capture. |
| **Single-Flight Inference & Latency Tracking**| **Adopted** | Enforce single-flight requests with explicit timeout (`AbortController` / `asyncio.wait_for`). Measure exact latency (median, P90, P95, max). |
| **Public Unauthenticated Market Data** | **Adopted** | Polymarket metadata, order books, RTDS, and Binance data are fetched strictly via public read-only unauthenticated endpoints. Zero API keys. |
| **Model Pinning (`typesafe/jev-1.13`)** | **Adapted** | The external repo uses `@ai-sdk/typesafe-ai` with unpinned `jev-latest`. WS-Jev uses OpenRouter REST API pinned strictly to `typesafe/jev-1.13` for scientific reproducibility. |
| **Standardized Horizon Sampling** | **Adapted** | The external repo polls continuously every 6 seconds for trading triggers. WS-Jev adapts this into standardized pre-expiry horizons (240s, 180s, 120s, 60s, 30s) to enable clean comparative forecasting analysis across rounds. |
| **4-Condition Jev Ablation Matrix** | **Adapted** | Instead of a single model prompt with optional perp features, WS-Jev implements 4 controlled ablation conditions (Reference Only, Reference + Perp, Market Aware, Full) to isolate feature contributions. |
| **Offline Deterministic Testing** | **Adapted** | All feeds, parsers, and calculators are covered by deterministic Python fixtures without requiring live network sockets during tests. |
| **Trading Rules & Simulated P&L** | **REJECTED** | WS-Jev is strictly a research and calibration laboratory. All trade generation, order placement, and P&L tracking are excluded. |
| **Entry Thresholds (`MIN_EDGE = 0.05`)** | **REJECTED** | Trading thresholds presuppose forecasting edge. WS-Jev first measures whether an edge exists statistically before hypothesizing trading rules. |
| **Score Shrinkage (`SCORE_WEIGHT = 0.5`)** | **REJECTED** | Arbitrary shrinkage toward 0.5 distorts probability calibration. WS-Jev evaluates raw model calibration directly using Brier score and log loss. |
| **Jev-Decided Trade Actions (`entry` question)**| **REJECTED** | Prompting Jev to choose between `buy_up`, `buy_down`, and `wait` conflates forecasting with execution heuristics. WS-Jev prompts strictly for probability. |
| **Simulated Fill Walking & Slippage Buffers** | **REJECTED** | Simulated order book execution is irrelevant to forecasting evaluation. Order books are used strictly as observational features and market baselines (`market_q`). |
| **One-Position Policy & Exit Rules** | **REJECTED** | Position management is out of scope for Phase 6. |
| **Daily Loss Limits & Bankroll Management** | **REJECTED** | Portfolio capital controls are omitted; no virtual capital is committed. |
| **Live Trading Switch (`TRADING_MODE=live`)** | **REJECTED** | Live trading modes, stubs, or switches violate WS-Jev's foundational safety contract. |

---

## 3. Methodological & Scientific Rationale

### 3.1 Why Reject Trading Rules and Score Shrinkage?
The external repository implements an ad-hoc heuristic:
$$\text{score} = 0.5 + w \times (p - 0.5)$$
where $w = 0.5$. It then checks if $\text{score} - \text{cost} \ge 0.05$.

In quantitative research, artificially shrinking a probability toward 0.5 hides miscalibration rather than addressing it. If Jev outputs $p = 0.80$ when the true probability is $0.55$, shrinking $p$ to $0.65$ may accidentally improve simulated trade selectivity, but it leaves the underlying forecasting defect unquantified.

WS-Jev treats the market itself as the primary baseline:
$$\Delta \text{Brier} = \text{Brier}(\text{Jev}) - \text{Brier}(\text{Market})$$
$$\Delta \text{LogLoss} = \text{LogLoss}(\text{Jev}) - \text{LogLoss}(\text{Market})$$

Evaluating the raw output of `typesafe/jev-1.13` directly allows us to measure whether the model possesses genuine calibration and signal, rather than tuning ad-hoc shrinkage parameters to fit retrospective observations.

### 3.2 Why Standardized Within-Round Horizons?
High-frequency 6-second polling creates heavily autocorrelated, unevenly spaced time series that vary round-by-round. For rigorous statistical comparison:
- 240 seconds remaining (early round, maximum uncertainty)
- 180 seconds remaining
- 120 seconds remaining
- 60 seconds remaining
- 30 seconds remaining (late round, terminal convergence)

Standardizing horizons allows round-clustered bootstrapping across matched time points, revealing how model forecasting quality evolves as time-to-expiry decays.

---

## 4. Summary of Architectural Implementation in WS-Jev

All concepts selected for adoption and adaptation are implemented in clean, modular Python under `src/pm_research/research/btc5m/`:
- `contract.py`: Polymarket BTC 5m event discovery, exact 300s duration check, outcome token mapping, and resolution verification.
- `reference_feed.py`: Public WebSocket / REST consumer for Chainlink BTC/USD spot and TWAP streams with exact-anchor detection.
- `poly_book.py`: Public unauthenticated CLOB book fetcher (`GET /book?token_id=...`) with sorting and validation.
- `binance_feed.py`: Public Binance USDⓈ-M perpetual feed computing depth imbalance, microprice offset, taker flow, and returns.
- `snapshot.py`: Immutable `BTC5mFeatureSnapshot` dataclass uniting all features at each capture horizon.
- `ablation.py`: Four-condition payload builder and OpenRouter query runner (`BTC5M_REFERENCE_ONLY`, `BTC5M_REFERENCE_PLUS_PERP`, `BTC5M_MARKET_AWARE`, `BTC5M_FULL`).
- `lab.py`: Orchestrator for prospective monitoring, horizon scheduling, persistence, and resolution scoring.
