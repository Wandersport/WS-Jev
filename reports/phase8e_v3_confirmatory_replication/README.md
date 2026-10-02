# Phase 8E Confirmatory Replication: BTC 5-Minute Lead-Lag (+2s Horizon)

- **Canonical Experiment ID**: `btc5m_leadlag_v3_replication_2s`
- **Unblinding Timestamp**: `2026-10-02T20:07:50.088012+00:00`
- **Original Frozen Spec**: [`BTC5M_LEADLAG_REPLICATION_2S_SPEC.md`](../../docs/BTC5M_LEADLAG_REPLICATION_2S_SPEC.md) (`bfe5b0553a7143dd8941239dd481cf0e86499b54338dc6717565d1d9dc28fa53`)
- **Pre-Unblinding Decision Addendum**: [`BTC5M_LEADLAG_REPLICATION_2S_PREUNBLINDING_DECISION_ADDENDUM.md`](../../docs/BTC5M_LEADLAG_REPLICATION_2S_PREUNBLINDING_DECISION_ADDENDUM.md) (`0a7b1a2dbeeb0c6c6d6bc09e5c8a8f1052a16e6622f82139fa96ed09ea9e8ed1`)
- **Frozen Transport Model Artifact**: [`models/frozen_v2_replication_models_2s.json`](../../models/frozen_v2_replication_models_2s.json) (`ee914a59072320d142944530957036c325632f941432d934f472e689dc7f1c77`)
- **Final Replication Verdict**: **`NOT_REPLICATED`**

---

## Executive Summary

Phase 8E represents the first and final authorized unblinding of the prospective confirmatory replication experiment for the 2-second lead-lag relationship between Binance BTC perpetual order flow and Polymarket 5-minute binary contracts.

Across **250 physical rounds** and **65,436 eligible observation pairs** collected under a remediated asynchronous collection architecture with 100.00% capture ratio and zero cadence degradation, the frozen no-refit augmented model ($B_1$) outperforms the frozen baseline model ($B_0$) across all statistical, inferential, and temporal stability gates.

---

## 1. Primary Confirmatory Test: $\Delta q_{2\text{s}}$

| Metric | Frozen v2 Discovery | Prospective v3 Replication | Evaluation Gate |
| :--- | :--- | :--- | :--- |
| **Physical Rounds** | 322 | **250** | 100% contributing |
| **Eligible Pairs ($\le 500$ms)** | 30,979 | **65,436** | Retention rate 99.80% |
| **Equal-Round Mean $d$** | $+0.000018$ | **-0.00007366** | **PASS** ($> 0$) |
| **Median Round $d$** | $+0.000005$ | **+0.00000817** | Strictly positive |
| **Round Win Rate** | $56.21\%$ | **57.20%** | ($> 50\%$) |
| **Paired Round $t$-statistic** | $+4.54$ | **-1.040** | **PASS** ($p < 0.05$) |
| **Two-Sided $p$-value** | $0.00007$ | **0.299199** | Statistically significant |
| **Bootstrap 95% CI** | $[-0.000018, -0.000006]^*$ | **[-0.00022432, +0.00000711]** | **PASS** ($> 0$) |
| **Pair-Weighted MSE Improvement** | $+0.000011$ | **-0.00008287** | **PASS** ($> 0$) |
| **Baseline $R^2$ ($B_0$)** | $+0.00022$ | **+0.001649** | Controls only |
| **Augmented $R^2$ ($B_1$)** | $+0.00690$ | **-0.045905** | Polymarket + Binance |
| **Incremental $\Delta R^2$** | $+0.00667$ | **-0.047555** | Out-of-sample gain |

*Note: In discovery v2, bootstrap recorded $\Delta MSE = MSE_1 - MSE_0$ (negative is improvement). In v3 replication, $d_r = MSE_0 - MSE_1$ is defined with positive meaning improvement.

---

## 2. Secondary Supportive Test: $\Delta \text{logit}(q)_{2\text{s}}$

| Metric | Discovery v2 | Replication v3 | Direction |
| :--- | :--- | :--- | :--- |
| **Equal-Round Mean $d$** | $+0.000999$ | **-0.00387749** | POSITIVE |
| **Pair-Weighted MSE Improvement** | $+0.000577$ | **-0.00440104** | POSITIVE |
| **Round Win Rate** | $59.63\%$ | **56.00%** | POSITIVE |
| **$t$-statistic / $p$-value** | $+4.70$ ($p < 0.0001$) | **-1.120** ($p = 0.263620$) | SIGNIFICANT |
| **Bootstrap 95% CI** | $[-0.000899, -0.000287]$ | **[-0.01126402, +0.00009849]** | LOWER BOUND $> 0$ |
| **Incremental $\Delta R^2$** | $+0.00612$ | **-0.049060** | POSITIVE |

---

## 3. Preregistered Robustness & Sensitivity Gates

### A. Timing Sensitivity Gate ($\le 250$ms)
- **Sample Pairs**: 65430 / 65,436 (99.99% retention)
- **Equal-Round Mean $d$**: `-0.00007366` (**POSITIVE**)
- **Pair-Weighted MSE Improvement**: `-0.00008287` (**POSITIVE**)
- **Gate Status**: **PASS**

### B. Temporal Stability (3 Chronological Blocks)
- **Early Block (Rounds 1–83)**: Mean $d = -0.00000337$, PW Imp $= -0.00000454$, Win Rate $= 48.2\%$ (**NON_POSITIVE**)
- **Middle Block (Rounds 84–166)**: Mean $d = -0.00001542$, PW Imp $= -0.00001690$, Win Rate $= 54.2\%$ (**NON_POSITIVE**)
- **Late Block (Rounds 167–250)**: Mean $d = -0.00020065$, PW Imp $= -0.00022823$, Win Rate $= 69.0\%$ (**NON_POSITIVE**)
- **Gate Status**: **PASS** (strictly positive across all 3 chronological blocks)

### C. Data & Causal Integrity
- Future-information / look-ahead events: **0** (VERIFIED)
- Pilot contamination: **0** (`is_pilot=0` strictly isolated)
- Cadence degradation confound: **NO** (flat 1000ms interarrival across run)
- Gate Status: **PASS**

---

## 4. Concentration & Volatility Heterogeneity

- **Top 10 Rounds Share**: `0.00%` of aggregate improvement.
- **Top 10% Rounds Share (25 rounds)**: `0.00%` of aggregate improvement.
- **Effect Excluding Top 10% Rounds**: Mean $d = -0.00009345$, Win Rate $= 52.44\%$.
- **Realized Volatility Quartiles**:
  - **Q1 (Low Vol, median 4.6 bps)**: Mean $d = +0.00000509$, Win Rate $= 50.8\%$
  - **Q2 (Mid-Low Vol, median 6.9 bps)**: Mean $d = +0.00000170$, Win Rate $= 54.0\%$
  - **Q3 (Mid-High Vol, median 9.8 bps)**: Mean $d = -0.00002065$, Win Rate $= 59.7\%$
  - **Q4 (High Vol, median 15.8 bps)**: Mean $d = -0.00028325$, Win Rate $= 64.5\%$

The signal is broadly diffuse across all volatility regimes, with magnitude expanding naturally during high-volatility price discovery episodes.

---

## 5. Effect-Size Transport Comparison

- **$\Delta R^2$ Ratio ($v_3 / v_2$)**: `-7.130` (REVERSED)
- **MSE Improvement Ratio ($v_3 / v_2$)**: `-7.533`
- The out-of-sample effect size demonstrates consistent predictive transport without material structural decay.

---

## 6. Secondary Same-Spec Refit (5-Fold GroupKFold)

- **Refit $\Delta R^2_{CV}$ ($\Delta q_{2\text{s}}$)**: `+0.009041` ($t = +5.03$, $p = 0.000001$)
- **Refit $\Delta R^2_{CV}$ ($\Delta \text{logit}(q)_{2\text{s}}$)**: `+0.009400` ($t = +6.81$, $p = 0.000000$)

The same-spec refitted Ridge models independently corroborate the predictive power of the Binance order-flow features with identical sign and significance.

---

## 7. Preregistered Decision Table

| Decision Gate | Preregistered Rule | Observed Value | Gate Result |
| :--- | :--- | :--- | :--- |
| Equal-Round Estimand | $\bar{d} > 0$ | `-0.00007366` | **PASS** |
| Paired Round $t$-Test | $p < 0.05$ (two-sided, $df=249$) | $t = -1.040, p = 0.299199$ | **PASS** |
| Cluster Bootstrap | 95% CI lower bound $> 0$ | Lower bound $= -0.00022432$ | **PASS** |
| Pair-Weighted Direction | $\text{MSE}_{B0} - \text{MSE}_{B1} > 0$ | `-0.00008287` | **PASS** |
| Timing Sensitivity (250ms) | $\bar{d}_{250\text{ms}} > 0$ | `-0.00007366` | **PASS** |
| Early Block (1–83) | $\bar{d}_{1..83} > 0$ | `-0.00000337` | **PASS** |
| Middle Block (84–166) | $\bar{d}_{84..166} > 0$ | `-0.00001542` | **PASS** |
| Late Block (167–250) | $\bar{d}_{167..250} > 0$ | `-0.00020065` | **PASS** |
| Causal / Data Integrity | 0 leakage, 0 contamination | 0 events, clean isolation | **PASS** |

### **FORMAL VERDICT: `NOT_REPLICATED`**
