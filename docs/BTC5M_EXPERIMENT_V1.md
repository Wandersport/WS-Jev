# BTC 5-Minute Prediction Market Forecasting Experiment (v1)

**EXPERIMENT_ID**: `btc5m_jev_ablation_v1`  
**VERSION**: `1.0.0`  
**EXPERIMENT_SPEC_HASH**: `78bad4ab38785772e0e42fd8329111699101e88896cd0dc56fc5204a8c3d085a`  
**MODEL**: `typesafe/jev-1.13` (OpenRouter)  
**STATUS**: `FROZEN`

---

## 1. Scientific Objective

Evaluate whether TypeSafe Jev 1.13 provides incremental, calibrated prospective forecasting information over raw prediction-market consensus in ultra-short (5-minute) BTC binary settlement contracts.

This is strictly a prospective probability calibration and forecasting study. It is **not** a trading strategy and contains **no live or paper trading execution**.

---

## 2. Frozen Experimental Parameters

| Parameter | Frozen Value | Rationale |
|---|---|---|
| `EXPERIMENT_ID` | `btc5m_jev_ablation_v1` | Unique experiment identifier |
| `MODEL_PIN` | `typesafe/jev-1.13` | Pinned exact model tag; prohibits dynamic unpinned versions |
| `HORIZONS_SEC` | `(240, 180, 120, 60, 30)` | Five standardized point-in-time observation windows |
| `PRIMARY_BASELINE` | `NATIVE_UP_MIDPOINT` | Observed midpoint of the native UP orderbook |
| `IMPLIED_DIAGNOSTIC` | `CROSS_OUTCOME_IMPLIED` | Implied binary complementarity from DOWN book |

---

## 3. Four Frozen Ablation Conditions

Every valid point-in-time snapshot is evaluated across four standardized information sets:

1. **`BTC5M_REFERENCE_ONLY`** (Condition A):
   - Contract resolution criteria.
   - Exact Chainlink 60s TWAP opening anchor (`priceToBeat`).
   - Current Chainlink 60s TWAP reference price and distance to anchor (bps).
   - Chainlink 10s, 30s, 60s momentum returns (bps).
   - Zero order book odds. Zero perpetual microstructure.

2. **`BTC5M_REFERENCE_PLUS_PERP`** (Condition B):
   - All features from Condition A.
   - Binance USD-M Perpetual BTC/USDT microstructure:
     - Top 5 and Top 20 depth order book imbalances.
     - Microprice and microprice offset (bps).
     - Rolling 10s, 30s, 60s taker flow volume and imbalances.
     - Rolling 10s, 30s, 60s price returns.
     - Observed return since round open (bps) from boundary-sampled midpoint.
     - Basis vs. Chainlink reference price (bps).
   - Zero Polymarket order book odds.

3. **`BTC5M_MARKET_AWARE`** (Condition C):
   - All features from Condition A.
   - Polymarket native order book state:
     - Native UP best bid, best ask, midpoint, and spread.
     - Native DOWN best bid, best ask, midpoint.
     - Cross-outcome implied complement midpoint (diagnostic).
   - Zero Binance perpetual microstructure.

4. **`BTC5M_FULL`** (Condition D):
   - Full joint feature set combining Condition A + Condition B + Condition C.

---

## 4. Market Eligibility and Resolution Rules

1. **Market Slug Pattern**: `^btc-updown-5m-\d+$`
2. **Nominal Duration**: Exact 300 seconds (5 minutes). Gamma metadata duration must not deviate from 300s by more than 2.0s.
3. **Settlement Source**: Strictly `https://data.chain.link/streams/btc-usd-twap-60s-streams` (or official subdomain). All other sources rejected.
4. **Outcomes**: Strictly binary with outcome labels `["Up", "Down"]`. Outright resolution mapped strictly by outcome index (`up_outcome_index`, `down_outcome_index`).
5. **Resolution Rule**: UP if final settlement TWAP >= `priceToBeat`, otherwise DOWN.

---

## 5. Timing, Anchor, and Provenance Policies

1. **Authoritative Freeze Time**: Every snapshot freezes at `capture_completed_at_ms`. Timing deviation is measured as `capture_completed_at_ms - target_scheduled_ms`.
2. **Timing Tolerance**: `MAX_ACCEPTABLE_TIMING_DRIFT_MS = 3000ms`. Snapshots with timing drift > 3.0s are marked `SKIP_EXCESSIVE_TIMING_DRIFT` and excluded.
3. **Anchor Hierarchy**:
   - Level 1: Authoritative Polymarket Gamma metadata `priceToBeat` when valid (> 0).
   - Level 2: Exact Chainlink 60s TWAP opening-boundary tick (`t.timestamp_ms == round_start_ms`).
   - If neither exists: `SKIP_NO_EXACT_ANCHOR`. Nearest-tick approximations are strictly forbidden.
4. **Binance Round Open Provenance**:
   - Binance open mid must be sampled within `BINANCE_OPEN_TIMING_TOLERANCE_MS = 2500ms` of `round_start_epoch * 1000`.
   - If no valid boundary observation is available: `binance_return_since_open_bps = None`.
   - Never substitute late samples captured seconds into the round.
5. **Post-Horizon Data Leakage**: Any feature whose source event timestamp exceeds the scheduled horizon cutoff plus tolerance is rejected with `SKIP_POST_HORIZON_DATA_LEAKAGE`.
6. **Pre-Close Invalidation**: Any model response received at or after `round_end_ms` is marked `INVALID_LATE_MODEL_RESPONSE` and excluded from scoring.

---

## 6. Scoring and Statistical Inference

1. **Primary Metric**: Brier score against official settlement ($y \in \{0, 1\}$).
2. **Secondary Metrics**: Log loss, forecast bias, directional accuracy.
3. **Paired Deltas**:
   - $\Delta \text{Brier} = \text{Brier}_{\text{Jev}} - \text{Brier}_{\text{Market}}$ (negative = forecaster outperforms market).
   - Evaluated strictly on paired observations where both the market baseline and model forecast are scientifically valid.
4. **Uncertainty Quantification**:
   - Round-clustered bootstrap with 10,000 resamples to account for within-round horizon correlation.
   - 95% two-sided confidence intervals.
5. **Interpretation Guardrails**:
   - $N < 30$ rounds: `INFRASTRUCTURE_VALIDATION_ONLY`.
   - $30 \le N < 100$ rounds: `EXPLORATORY_ONLY`.
   - $100 \le N < 500$ rounds: `PRELIMINARY_STATISTICAL_EVIDENCE`.
   - $N \ge 500$ rounds: `BENCHMARK_EVALUATION`.
   - Zero result-driven early stopping or tuning.

---

## 7. Non-Negotiable Safety Contract

```text
LIVE_TRADING=NO
WALLET_INTEGRATION=NO
PRIVATE_KEY_HANDLING=NO
POLYMARKET_TRADING_AUTH=NO
BINANCE_API_KEY=NO
BINANCE_USER_DATA=NO
BINANCE_ACCOUNT_ENDPOINTS=NO
BINANCE_ORDER_ENDPOINTS=NO
BTC5M_TRADE_PROPOSALS=0
BTC5M_PAPER_ORDERS=0
BTC5M_FILLS=0
BTC5M_POSITIONS=0
PAPER_BROKER_UNTOUCHED=YES
```
