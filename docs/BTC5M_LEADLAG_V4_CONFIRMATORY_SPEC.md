# BTC 5-Minute Lead-Lag V4 Prospective Confirmatory Experiment Specification

- **Document Version**: `4.0.0`
- **Canonical Experiment ID**: `btc5m_leadlag_v4_confirmatory_2s`
- **Status**: `FROZEN_CONFIRMATORY_SPECIFICATION` (Pre-registration Frozen; DO NOT LAUNCH YET)
- **Created Timestamp (UTC)**: `2026-10-02T20:45:00Z`
- **Frozen Model Artifact**: [`models/frozen_v4_confirmatory_models_2s.json`](../models/frozen_v4_confirmatory_models_2s.json)
- **Frozen Model SHA256**: `59d31330b181b2a2c85777f9c15df2223e75b161f0afcdfa68f8ba38c7ea65fd`
- **Target Horizon**: $+2.0$ seconds ($L = 2000$ ms) exclusively
- **Target Sample Size**: **$750$ full finalized physical rounds** ($\approx 196,000$ eligible pairs)

---

## 1. Scientific Context & Narrative Standards

1. **Formal Historical Verdicts**:
   - The Phase 8E confirmatory verdict remains immutable: **`FINAL_REPLICATION_VERDICT = NOT_REPLICATED`**.
   - Phase 8E.1 post-unblinding analysis is **exploratory**.
   - The clipped stationary architecture is a **candidate redesign**, NOT a confirmed or replicated model.
   - Do NOT claim that the lead-lag edge is "confirmed", "proven", or "mechanistically and statistically real" prior to prospective confirmation.
   - Canonical narrative: **"Phase 8E.1 provides exploratory evidence for a persistent +2s lead-lag candidate, but independent prospective confirmation is still required."**

2. **Chronological Forward Validation Findings**:
   - Across the 200 out-of-sample forward-chained rounds of v3 (rounds 51–250):
     - Aggregate forward mean $d = +0.00000846 > 0$
     - Win rate: $72.5\%$ ($145 / 200$ rounds won)
     - Paired round $t$-statistic: $+1.56$
     - Two-sided $p$-value: $0.060$
     - Fold 1 was negative ($-0.00002362$), Folds 2–4 were positive ($+0.00000894, +0.00000461, +0.00003940$).
   - Because $p = 0.060 > 0.05$, this exploratory forward-chaining trajectory is NOT independently significant at $\alpha = 0.05$. It serves solely as the empirical effect size basis for prospective power calculation.

---

## 2. Model Architecture & Preprocessing

### 2.1 Feature Definitions (Strictly 21 Scale-Invariant Features)
The v4 model architecture completely eliminates raw price levels and cumulative intra-round drift to ensure scale-invariance and cross-regime stationarity:

- **Baseline Controls (B0, 9 features)**:
  1. `poly_midpoint`: Polymarket Up contract midpoint probability ($q_t$)
  2. `poly_spread`: Polymarket Up contract bid-ask spread
  3. `seconds_remaining`: Time remaining until 5-minute round scheduled close
  4. `poly_return_1s`: Native Polymarket midpoint price delta over past 1s
  5. `poly_return_2s`: Native Polymarket midpoint price delta over past 2s
  6. `poly_return_3s`: Native Polymarket midpoint price delta over past 3s
  7. `poly_return_5s`: Native Polymarket midpoint price delta over past 5s
  8. `poly_return_10s`: Native Polymarket midpoint price delta over past 10s
  9. `poly_return_30s`: Native Polymarket midpoint price delta over past 30s

- **Binance Incremental Microstructure Features (12 features)**:
  10. `binance_microprice_offset_bps`: Microprice offset from mid in basis points
  11. `binance_spread_bps`: Top-of-book bid-ask spread in basis points
  12. `binance_return_1s_bps`: Binance mid price return over past 1s in basis points
  13. `binance_return_2s_bps`: Binance mid price return over past 2s in basis points
  14. `binance_return_3s_bps`: Binance mid price return over past 3s in basis points
  15. `binance_return_5s_bps`: Binance mid price return over past 5s in basis points
  16. `binance_return_10s_bps`: Binance mid price return over past 10s in basis points
  17. `binance_return_30s_bps`: Binance mid price return over past 30s in basis points
  18. `binance_return_60s_bps`: Binance mid price return over past 60s in basis points
  19. `binance_top1_depth_imbalance`: Top-1 level book depth imbalance $\frac{Q_{bid} - Q_{ask}}{Q_{bid} + Q_{ask}} \in [-1, 1]$
  20. `binance_top5_depth_imbalance`: Top-5 cumulative book depth imbalance $\in [-1, 1]$
  21. `binance_top20_depth_imbalance`: Top-20 cumulative book depth imbalance $\in [-1, 1]$

- **Explicitly Excluded Features**:
  - `binance_mid_price`: EXCLUDED (caused $+6.65\sigma$ out-of-distribution regime shift in v3).
  - `binance_return_since_open_bps`: EXCLUDED (non-stationary path dependence).

### 2.2 Preprocessing & Bounded Standardization
1. **Scaler Provenance**: Scaler means ($\mu_j$) and standard deviations ($\sigma_j$) are strictly fixed from the 65,436 eligible pairs of the 250 historical development rounds of `btc5m_leadlag_v3_replication_2s`.
2. **Standardization Formula**:
   $$z_j = \frac{x_j - \mu_j}{\sigma_j}$$
3. **Hard Standardization Clipping**:
   $$z_j^{\text{clipped}} = \max\left(-5.0, \min\left(5.0, z_j\right)\right)$$
   Every standardized input MUST be strictly bounded within $[-5.0, +5.0]\sigma$. No quadratic blowup or unclipped multiplier is permitted.
4. **Frozen Parameters**: Zero adaptive clipping, zero Winsor threshold tuning after data collection, and zero scaler refitting on prospective v4 data for the primary confirmatory test.

---

## 3. Frozen Model Weights & Training Provenance

All model weights are regularized via Ridge regression ($L_2 = 1.0$) on the bounded standardized design matrix $(Z, y)$ of the development set:

### 3.1 Primary Confirmatory Model (`delta_q_2s`)
- **B0 Intercept**: `-0.00001759`
- **B1 Intercept**: `-0.00001759`
- **B1 Coefficients**:
  - `poly_midpoint`: `-0.00052068`
  - `poly_spread`: `-0.00063743`
  - `seconds_remaining`: `+0.00011890`
  - `poly_return_1s`: `+0.00145484`
  - `poly_return_2s`: `-0.00017600`
  - `poly_return_3s`: `+0.00013268`
  - `poly_return_5s`: `-0.00014188`
  - `poly_return_10s`: `-0.00009316`
  - `poly_return_30s`: `-0.00038845`
  - `binance_microprice_offset_bps`: `+0.00709201`
  - `binance_spread_bps`: `-0.00083151`
  - `binance_return_1s_bps`: `+0.00289914`
  - `binance_return_2s_bps`: `-0.00022661`
  - `binance_return_3s_bps`: `-0.00049174`
  - `binance_return_5s_bps`: `-0.00001616`
  - `binance_return_10s_bps`: `-0.00107979`
  - `binance_return_30s_bps`: `+0.00047199`
  - `binance_return_60s_bps`: `-0.00014112`
  - `binance_top1_depth_imbalance`: `-0.00150549`
  - `binance_top5_depth_imbalance`: `-0.00019738`
  - `binance_top20_depth_imbalance`: `+0.00033615`

### 3.2 Secondary Supportive Model (`delta_logit_q_2s`)
- Full coefficient and intercept parameters are serialized in [`models/frozen_v4_confirmatory_models_2s.json`](../models/frozen_v4_confirmatory_models_2s.json).

---

## 4. Sample Size & Statistical Power Pre-Registration

### 4.1 Power Analysis Across Candidate Sample Sizes
Based on round-level effect size distributions observed during Phase 8E.1:
- Conservative forward-chaining effect size: Cohen's $d = 0.110$
- Moderate within-regime effect size: Cohen's $d = 0.312$

| Target Power | Minimum $N$ Required ($d=0.110$) | Minimum $N$ Required ($d=0.312$) | Power at $N=250$ | Power at $N=500$ | Power at $N=750$ | Power at $N=869$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **80% Power** | **649 rounds** | 81 rounds | $41.3\%$ | $69.1\%$ | **$85.4\%$** | $90.0\%$ |
| **85% Power** | **743 rounds** | 93 rounds | $41.3\%$ | $69.1\%$ | **$85.4\%$** | $90.0\%$ |
| **90% Power** | **869 rounds** | 108 rounds | $41.3\%$ | $69.1\%$ | $85.4\%$ | **$90.0\%$** |

### 4.2 Preregistered Sample Size Decision
- **Preregistered Sample Size**: **$N = 750$ full physical rounds** ($\approx 62.5$ continuous hours).
- **Statistical Rationale**:
  - $N=250$ is rejected due to severe underpowering ($41.3\%$) in conservative scenarios.
  - $N=500$ provides only $69.1\%$ power under conservative forward chaining.
  - $N=750$ exceeds the $80\%$ threshold, delivering **$85.4\%$ power** under the conservative forward-chaining distribution ($d = 0.110$) and **$100.0\%$ power** under moderate within-regime persistence ($d = 0.312$).

---

## 5. Statistical Estimands & Formal Decision Rules

### 5.1 Primary Estimand
For each physical round $r \in \{1, \dots, N\}$, compute the mean squared error improvement:
$$d_r = \text{MSE}_{B0, r} - \text{MSE}_{B1, r}$$
The primary confirmatory estimand is the unweighted sample mean across physical rounds:
$$\bar{d} = \frac{1}{N} \sum_{r=1}^N d_r$$

### 5.2 Primary Hypothesis Test
- **Null Hypothesis ($H_0$)**: $\mu_d \le 0$ (Augmented stationary model B1 does not improve prospective prediction over B0).
- **Alternative Hypothesis ($H_1$)**: $\mu_d > 0$ (Augmented stationary model B1 improves prospective prediction over B0).
- **Inference Procedures**:
  1. Two-sided paired-round Student's $t$-test at $\alpha = 0.05$.
  2. 10,000 whole-round clustered bootstrap replicates with fixed seed `20261002`, evaluating the $95\%$ percentile confidence interval $[\text{CI}_{2.5\%}, \text{CI}_{97.5\%}]$.

### 5.3 Formal Confirmation Decision Matrix

#### A. CONFIRMED
Replication confirmation requires ALL of the following criteria to be met simultaneously:
1. Primary equal-round mean improvement is strictly positive: $\bar{d} > 0$.
2. Two-sided paired-round $t$-test is statistically significant: $p < 0.05$.
3. Bootstrap $95\%$ confidence interval lower bound is strictly positive: $\text{CI}_{2.5\%} > 0$.
4. Pair-weighted aggregate MSE improvement is strictly positive: $\Delta \text{MSE}_{\text{pair}} > 0$.
5. Timing sensitivity test at $\le 250$ ms gate is strictly positive: $\bar{d}_{\le 250\text{ms}} > 0$.
6. Temporal stability across blocks: Early (rounds 1–250), Middle (rounds 251–500), and Late (rounds 501–750) all exhibit $\bar{d}_{\text{block}} > 0$.
7. Measurement & data integrity: zero lookahead events, zero clock-sync violations, zero cross-round contamination, capture ratio $\ge 95\%$.

#### B. PARTIALLY_REPLICATED
Assigned if:
- Primary equal-round mean $\bar{d} > 0$ and pair-weighted $\Delta \text{MSE}_{\text{pair}} > 0$, BUT one or more secondary statistical inference criteria (e.g. $p \ge 0.05$, bootstrap lower bound $\le 0$, timing sensitivity $\le 0$, or temporal block inconsistency) fail.

#### C. NOT_REPLICATED
Assigned if:
- Primary equal-round mean $\bar{d} \le 0$, OR
- Pair-weighted aggregate $\Delta \text{MSE}_{\text{pair}} \le 0$, OR
- Model predictions show material directional reversal relative to baseline.

#### D. INVALID_REPLICATION
Assigned if:
- Fatal data integrity failure occurs (e.g. non-causal timestamp order, corrupted SQLite database, network ingestion failure exceeding capture thresholds, or code mutation during runtime).

### 5.4 Role of Secondary Supportive Endpoint (`delta_logit_q_2s`)
- Evaluated and reported under identical estimands.
- Under hierarchical endpoint ordering, `delta_logit_q_2s` is strictly supportive; it **cannot rescue** a primary confirmation failure on `delta_q_2s`.

---

## 6. Timing, Cadence, and Data Integrity Gates

The prospective collector must enforce all verified operational invariants:
1. **Round Boundary Synchronization**: Collection begins strictly at the physical round start boundary.
2. **Complete Finalized Rounds Only**: Exactly 750 finalized full rounds. No partial, active, or truncated rounds contribute to analysis.
3. **Target Cadence**: 1.0 Hz nominal sampling ($300$ samples per 5-minute round).
4. **Capture Completeness**: Minimum $95\%$ valid samples per round ($\ge 285$ valid samples).
5. **Exact-Grid Pairing**: Ground truth future target evaluated at $t + 2000$ ms.
6. **Timing Error Gates**:
   - Primary gate: $|\Delta t_{\text{elapsed}} - 2000\text{ms}| \le 500$ ms.
   - Sensitivity gate: $|\Delta t_{\text{elapsed}} - 2000\text{ms}| \le 250$ ms.
7. **Zero Lookahead Enforced**:
   - $\text{elapsed\_ms} > 0$ strictly enforced.
   - Binance and Polymarket receipt timestamps must not exceed sample snapshot time $+ 1000$ ms.
8. **Asynchronous Architecture**: Dedicated background writer thread with bounded queue. Zero synchronous SQLite operations on the 1 Hz network sampling loop.

---

## 7. Strict Outcome Blinding Protocol

During live prospective data collection, the environment is strictly outcome-blind:
- **Strictly Prohibited**:
  - Computing or inspecting B0 vs B1 prediction errors.
  - Computing MSE improvements, $\Delta R^2$, or loss ratios.
  - Inspecting direction accuracy or hit rates.
  - Running $t$-tests, regressions, or interim $p$-values.
  - Evaluating volatility-subset signal strength.
  - Modifying features, models, weights, or scalers.
  - Stopping early for perceived efficacy or extending $N$ after inspecting outcomes.
- **Strictly Permitted**:
  - Operational health telemetry (process alive, heartbeat, round counter, inter-arrival latencies, queue depth, disk space, capture ratios).

---

## 8. Pre-Registration Verification Checksums

- **Specification File**: `docs/BTC5M_LEADLAG_V4_CONFIRMATORY_SPEC.md`
- **Frozen Model File**: `models/frozen_v4_confirmatory_models_2s.json`
- **Frozen Model SHA256**: `59d31330b181b2a2c85777f9c15df2223e75b161f0afcdfa68f8ba38c7ea65fd`
- **Development Dataset**: `data/pm_research_v3_replication.db` (250 rounds, 65,436 eligible pairs)
- **Deployment Status**: `PRE_REGISTRATION_FROZEN` — **DO NOT LAUNCH COLLECTION YET**.
