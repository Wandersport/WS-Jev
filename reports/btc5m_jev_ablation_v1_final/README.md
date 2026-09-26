# BTC 5-Minute Jev Ablation Experiment (v1) — Final Research Archive

**EXPERIMENT_ID**: `btc5m_jev_ablation_v1`  
**EXPERIMENT_SPEC_HASH**: `78bad4ab38785772e0e42fd8329111699101e88896cd0dc56fc5204a8c3d085a`  
**MODEL**: `typesafe/jev-1.13` (via OpenRouter)  
**HORIZONS_SEC**: `(240, 180, 120, 60, 30)`  
**PRIMARY_BASELINE**: `NATIVE_UP_MIDPOINT`  
**STATUS**: `FROZEN_COMPLETED` (500 Valid Resolved Rounds Reached)  
**CLOSING_COMMIT**: `e9091a3cb4be2501f6e4b1f81f3e601e1bc2f07b`  
**ARCHIVAL_GIT_TAG**: `btc5m-jev-ablation-v1-500r`  

---

## 1. Formal Scientific Conclusions

The primary Phase 7 scientific hypothesis was whether LLM-based probability forecasts from TypeSafe Jev 1.13 provide incremental, calibrated information over the native Polymarket prediction-market consensus in ultra-short (5-minute) BTC binary settlement contracts.

Based on 500 prospective, validly resolved rounds and 1,650 scored point-in-time horizons (6,600 prospective forecasts), the empirical results establish three formal findings:

### Primary Finding
```text
JEV_UNDERPERFORMS_NATIVE_MARKET_BASELINE_IN_FROZEN_BTC5M_EXPERIMENT
```
Across all four ablation conditions (Reference Only, Reference + Perp, Market Aware, Full), TypeSafe Jev achieves higher (worse) Brier score and higher log loss than the native Polymarket midpoint consensus. All paired differences are statistically significant with round-clustered bootstrap 95% confidence intervals strictly excluding zero.

### Secondary Finding 1
```text
BINANCE_FEATURES_IMPROVE_REFERENCE_ONLY_JEV_BUT_DO_NOT_OVERCOME_MARKET_BASELINE
```
When Jev is blinded to the Polymarket orderbook, providing Binance USD-M Perpetual microstructure features improves Jev's forecast accuracy substantially:
- $\Delta \text{Brier}(\text{Ref+Perp} - \text{RefOnly}) = -0.01403$ (improves Brier from 0.20495 to 0.19096).
However, this enhanced forecast remains far inferior to the native Polymarket consensus ($0.19096$ vs. $0.14047$, $\Delta = +0.05049$).

### Secondary Finding 2
```text
BINANCE_ADDS_NO_OBSERVED_INCREMENTAL_VALUE_ONCE_JEV_SEES_MARKET_PRICE
```
When Jev is already conditioned on the Polymarket orderbook (`MARKET_AWARE`), adding Binance perpetual microstructure (`FULL`) provides zero incremental value and slightly degrades performance:
- $\Delta \text{Brier}(\text{Full} - \text{MarketAware}) = +0.00126$ ($0.16107$ vs. $0.15981$).
The native Polymarket price already aggregates or subsumes the relevant directional information accessible to the LLM.

---

## 2. Final Primary Aggregate Benchmark Results

Evaluated strictly on 1,649 paired scored observations across 500 unique physical rounds:

| Condition | N (Paired) | Mean Brier | Market Brier | Delta Brier | 95% Clustered Bootstrap CI | Mean Log Loss | Market Log Loss | Delta Log Loss |
|---|---|---|---|---|---|---|---|---|
| **Polymarket Native Baseline** | 1,649 | 0.14047 | 0.14047 | +0.00000 | [0.00000, 0.00000] | 0.42887 | 0.42887 | +0.00000 |
| **BTC5M_REFERENCE_ONLY** | 1,649 | 0.20495 | 0.14047 | **+0.06448** | [+0.05723, +0.07133] | 0.60089 | 0.42887 | +0.17202 |
| **BTC5M_REFERENCE_PLUS_PERP** | 1,649 | 0.19096 | 0.14047 | **+0.05049** | [+0.04439, +0.05661] | 0.56982 | 0.42887 | +0.14095 |
| **BTC5M_MARKET_AWARE** | 1,649 | 0.15981 | 0.14047 | **+0.01934** | [+0.01580, +0.02315] | 0.49235 | 0.42887 | +0.06348 |
| **BTC5M_FULL** | 1,649 | 0.16107 | 0.14047 | **+0.02060** | [+0.01708, +0.02416] | 0.49592 | 0.42887 | +0.06705 |

### Pairwise Ablation Deltas
- `REF_PLUS_PERP_MINUS_REF_ONLY`: **-0.01403** (Binance features improve blind reference forecasting)
- `FULL_MINUS_MARKET_AWARE`: **+0.00126** (Binance features do not improve market-aware forecasting)
- `MARKET_AWARE_MINUS_REF_ONLY`: **-0.04519** (Market price is overwhelmingly the strongest signal)

*Note: Round-clustered bootstrap with 1,000 resamples grouping by `round_slug` to account for within-round horizon correlation.*

---

## 3. Dataset Audit & Counts

```text
PHYSICAL_ROUNDS_DISCOVERED:   844
PHYSICAL_ROUNDS_RESOLVED:     844
ROUNDS_WITH_VALID_SNAPSHOTS:  500
TOTAL_SNAPSHOTS_PERSISTED:    3,925
VALID_SNAPSHOTS:              1,650
INVALID_SNAPSHOTS_EXCLUDED:   2,275

EXCLUSION_BREAKDOWN:
  SKIP_NO_EXACT_ANCHOR:          1,564
  SKIP_MISSING_MARKET_Q:           639
  SKIP_STALE_REFERENCE:             66
  SKIP_EXCESSIVE_TIMING_DRIFT:       6

VALID_SNAPSHOTS_BY_HORIZON:
  240s: 348
  180s: 483
  120s: 461
   60s: 265
   30s:  93

SCORED_ROUND_HORIZONS:        1,650
PAIRED_MARKET_SCORE_ROWS:     1,649
UNPAIRED_MARKET_SCORE_ROWS:   1  (due to transient empty best bid at 120s in round btc-updown-5m-1790162400)
DUPLICATE_SCORE_KEYS:         0
TOTAL_FORECAST_ROWS:          6,600
FAILED_JEV_REQUESTS:          0
LATE_JEV_RESPONSES:           0
```

---

## 4. OpenRouter Financial Accounting

- **CANONICAL_TOTAL_COST_USD**: `$0.243008` (from official OpenRouter API response metadata persisted in checkpoints)
- **RECORDED_FORECAST_COST_USD**: `$0.242322` (sum of individual forecast row cost attributes)
- **TOTAL_FORECAST_REQUESTS**: `6,600`
- **AVERAGE_COST_PER_FORECAST**: `$0.0000368`
- **BREAKDOWN BY CONDITION**:
  - `BTC5M_REFERENCE_ONLY`: 1,650 requests | $0.055610 | 1,034,220 input tokens | 77,550 output tokens
  - `BTC5M_REFERENCE_PLUS_PERP`: 1,650 requests | $0.063162 | 1,228,880 input tokens | 77,550 output tokens
  - `BTC5M_MARKET_AWARE`: 1,650 requests | $0.058348 | 1,105,440 input tokens | 77,550 output tokens
  - `BTC5M_FULL`: 1,650 requests | $0.065202 | 1,279,720 input tokens | 77,550 output tokens

---

## 5. Methodological Limitations

1. **Contract Specificity**:
   The empirical findings apply strictly to ultra-short (5-minute) BTC Up/Down binary prediction market contracts settled against the official Chainlink 60s TWAP feed. Findings cannot be generalized to longer duration contracts (hourly, daily, weekly) or other asset classes.
2. **Temporal Window**:
   Collection took place in late September 2026 across continuous regime variations in BTC spot and perpetual markets. Future market microstructure dynamics could shift.
3. **Model & Architecture Boundary**:
   The evaluation was pinned strictly to `typesafe/jev-1.13` through OpenRouter using zero-shot prompting with structured JSON outputs. Prompt engineering, fine-tuning, or alternative model families were not evaluated under this frozen spec.
4. **Anchor Gate Conservatism**:
   The strict anchor hierarchy rejected 1,564 snapshots that lacked a bit-exact Chainlink boundary timestamp or authoritative Gamma metadata `priceToBeat`. While preserving absolute data integrity, this excluded volatile or rapid-open rounds where anchor ticks drifted slightly.
5. **Structural Paper-Only Safety**:
   Zero live or paper trading execution was attempted. The experiment measured pure point-in-time prospective forecast calibration against market settlement.
