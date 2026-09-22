# System Architecture

**PAPER TRADING / RESEARCH SIMULATION ONLY &mdash; NO LIVE EXECUTION**

The system implements the conceptual quantitative research pipeline:

```
[ Market Feed / Fixtures ]
           │
           ▼
         RIGO   (Market Ingestion & UTC Normalization)
           │
           ▼
         HOLT   (Research Feature Extraction & Provenance)
           │
           ▼
         ILSA   (Probability Estimation q̂ & Uncertainty σ_q)
           │
           ▼
         KETT   (Robust Edge Calculation & Fractional Kelly Sizing)
           │
           ▼
         BRAM   (Authoritative Risk Gate & Concentration Limits)
           │
           ▼
      PAPERBROKER (Local Simulation: Order-Book Walking & Slippage)
           │
           ▼
         TESS   (Paper Portfolio Accounting, Drawdowns & Settlements)
           │
           ▼
   [ SQLite Audit Trail & HTML Dashboard ]
```

---

## Component Responsibilities

### 1. RIGO &mdash; Ingestion and Normalization
- **Purpose**: Ingest raw prediction market feeds, validate data integrity, reject malformed or crossed books, and normalize to UTC domain models.
- **Key Inputs**: Deterministic synthetic fixtures or unauthenticated public read-only market endpoints.
- **Key Outputs**: `MarketSnapshot` containing typed `Market`, `MarketQuote`, and optional `OrderBook`.
- **Validation**: Rejects invalid prices (outside `[0, 1]`), crossed books (`bid > ask`), negative liquidity/volume, or invalid resolution timestamps.

### 2. HOLT &mdash; Research Features
- **Purpose**: Compute transparent, auditable features for market dynamics and microstructure.
- **Features Extracted**:
  - `momentum_24h`: Short-term price trend/velocity.
  - `order_book_imbalance`: `(bid_depth - ask_depth) / (bid_depth + ask_depth)` in `[-1.0, 1.0]`.
  - `spread_pct`: Bid-ask spread relative to midpoint.
  - `liquidity_log`: `log10(liquidity + 1)`.
  - `hours_to_resolution`: Remaining time until contract maturity.
  - `volume_turnover_ratio`: 24h trading volume relative to visible liquidity.
  - `research_signal`: Fundamental exogenous signal / domain prior.
- **Provenance**: Every extracted feature retains explicit documentation of its source formula.

### 3. ILSA &mdash; Probability Estimation
- **Purpose**: Compute calibrated forecast probability $\hat{q} = P(\text{YES})$ alongside explicit uncertainty $\sigma_q$.
- **Mathematical Model**:
  $$\text{logit}(p_{\text{ref}}) = \ln\left(\frac{p_{\text{ref}}}{1 - p_{\text{ref}}}\right)$$
  $$q_{\text{base}} = \sigma\left(\text{slope} \cdot \text{logit}(p_{\text{ref}})\right)$$
  $$\hat{q} = \text{clip}\left(q_{\text{base}} + \sum \Delta_i, q_{\text{min}}, q_{\text{max}}\right)$$
- **Uncertainty Model**:
  Estimates $\sigma_q$ with additive penalties for wide spreads, thin liquidity, data staleness, and missing features.
- **Provenance**: Emits `ProbabilityEstimate` with unique estimate ID, cycle ID, model version, and exact component breakdown.

### 4. KETT &mdash; Edge Detection and Kelly Sizing
- **Purpose**: Compute executable raw edge, subtract safety margins and transaction costs, and size hypothetical positions using conservative fractional Kelly.
- **Edge Equations**:
  $$\text{Raw Edge (YES)} = \hat{q} - \text{Ask}_{\text{YES}}$$
  $$\text{Raw Edge (NO)} = (1 - \hat{q}) - \text{Ask}_{\text{NO}}$$
  $$\text{Robust Edge} = \text{Raw Edge} - z \cdot \sigma_q - \text{Costs}$$
- **Fractional Kelly Sizing**:
  $$f^* = \max\left(0, \frac{q - p}{1 - p}\right)$$
  $$f_{\text{candidate}} = \text{kelly\_multiplier} \cdot f^*$$
  Subject to single-position caps, order-book depth caps, total exposure caps, and category exposure limits.

### 5. BRAM &mdash; Authoritative Risk Gate
- **Purpose**: Chief Risk Officer. Evaluates proposals against rigid deterministic risk rules and issues an immutable `ACCEPT` or `REJECT_*` decision.
- **Stable Reason Codes**:
  - `ACCEPT`
  - `REJECT_HALTED_MODE`
  - `REJECT_MARKET_NOT_ACTIVE`
  - `REJECT_STALE_DATA`
  - `REJECT_LOW_LIQUIDITY`
  - `REJECT_HIGH_SPREAD`
  - `REJECT_EXPIRING_SOON`
  - `REJECT_HIGH_UNCERTAINTY`
  - `REJECT_ROBUST_EDGE_TOO_LOW`
  - `REJECT_MAX_EXPOSURE`
  - `REJECT_CATEGORY_CONCENTRATION`
  - `REJECT_MAX_OPEN_POSITIONS`
  - `REJECT_DUPLICATE_POSITION`
  - `REJECT_INSUFFICIENT_CASH`

### 6. PAPERBROKER &mdash; Simulation Only
- **Purpose**: The **sole execution implementation** in the repository. Structurally incapable of live trading.
- **Execution Logic**:
  - **Depth Walking**: If an order book is available, consumes asks level by level, calculating volume-weighted average fill price and slippage. Supports partial fills if depth is exhausted.
  - **Fallback Model**: When book depth is unavailable, applies a documented conservative slippage penalty `quote_price + 0.20 * spread`. Never assumes midpoint execution.
  - **Fee Modeling**: Deducts simulated exchange transaction fees on every fill.

### 7. TESS &mdash; Paper Portfolio Management
- **Purpose**: Virtual capital ledger, mark-to-market accounting, position tracking, and drawdown management.
- **Accounting**:
  - Cash deductions on fills, cash credits on winning resolutions ($1.00 per contract).
  - High-water mark tracking and peak-to-trough drawdown calculation:
    $$\text{Drawdown} = \frac{\text{HWM} - \text{Equity}}{\text{HWM}}$$
  - Feeds drawdown into `SurvivalManager` to trigger deterministic risk state transitions.
  - Computes Brier scores and log loss upon market settlement.
