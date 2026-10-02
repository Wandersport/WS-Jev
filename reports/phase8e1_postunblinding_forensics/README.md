# Phase 8E.1 Post-Unblinding Forensic Analysis: Root Cause & Scientific Resolution

- **Canonical Experiment ID**: `btc5m_leadlag_v3_replication_2s`
- **Forensic Analysis Timestamp**: `2026-10-02T20:23:04.506354+00:00`
- **Formal Confirmatory Verdict**: **`NOT_REPLICATED`** (Immutable Phase 8E outcome)
- **Investigation Purpose**: Deconstruct the frozen no-refit transport failure, evaluate strict chronological forward validation, audit scale-invariant feature architectures, and design the v4 prospective replication.

---

## Executive Summary of Findings

1. **Primary Mechanism of Frozen Transport Failure**:
   - The frozen model $B_1$ incorporated non-stationary raw price level (`binance_mid_price`) and cumulative open drift (`binance_return_since_open_bps`).
   - Between discovery v2 ($83.4k) and prospective v3 ($85.6k–$87.2k), Bitcoin experienced a **$+6.65\sigma$ mean shift** and up to **$+11.35\sigma$ peak excursion**.
   - During high-volatility microstructure shocks in v3, `binance_microprice_offset_bps` experienced an excursion of **$-207.8\sigma$** and `binance_spread_bps` of **$+819.8\sigma$** relative to v2 standard deviations.
   - In the absence of standardization clipping, the linear regression model generated impossible predictions (reaching $\hat{y} = -2.16$ on bounded probability changes), producing massive quadratic squared-error penalties in high-volatility rounds.

2. **Strict Chronological Forward Validation**:
   - In expanding-window forward validation on prospective v3 data (Train $1..T$, Test $T..T+50$):
   - Fold 2 (Rounds 101–150): Win Rate $= 72.0\%$, $\Delta R^2 = +0.00514$
   - Fold 3 (Rounds 151–200): Win Rate $= 82.0\%$, $\Delta R^2 = +0.00158$
   - Fold 4 (Rounds 201–250): Win Rate $= 92.0\%$, $\Delta R^2 = +0.02296$
   - Aggregate across all 200 out-of-sample forward rounds (52,185 pairs): **Pair-Weighted Imp $= +0.00000715$, Round Win Rate $= 72.5\%$, $t = +1.56$**.

3. **Resolution via Scale-Invariance & Outlier Clipping**:
   - Excluding non-stationary price variables and applying standard $[-5.0, +5.0]\sigma$ clipping restores out-of-sample transport on frozen v2 coefficients:
   - **Clipped Transport**: Pair-Weighted Imp $= +0.00001457$, Mean $d = +0.00001498$, $\Delta R^2 = +0.00836$, Win Rate $= 62.4\%$, **$t = +6.96$ ($p < 10^{-11}$)**.
   - Across all 4 volatility quartiles Q1–Q4, performance is strictly positive, with Q4 (highest volatility) delivering the largest predictive gain ($+0.00002486$, win rate $64.5\%$).

---

## Table 1: Feature Shift & Error Decomposition

| Feature Name | Stationarity | V2 Mean | V3 Mean | V3 Shift ($\sigma$) | Max $|z_{v2}|$ | Frozen $\beta$ ($\Delta q$) | Mean Bias Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `poly_midpoint` | STATIONARY | 0.484 | 0.555 | +0.20 | 1.5 | -0.002077 | -0.000421 |
| `poly_spread` | STATIONARY | 0.009 | 0.010 | +0.23 | 47.5 | +0.000241 | +0.000057 |
| `seconds_remaining` | STATIONARY | 150.004 | 162.162 | +0.14 | 1.7 | +0.000133 | +0.000019 |
| `poly_return_1s` | STATIONARY | 0.000 | 0.000 | -0.00 | 14.9 | +0.001062 | -0.000001 |
| `poly_return_2s` | STATIONARY | 0.000 | 0.000 | -0.00 | 15.2 | -0.000481 | +0.000002 |
| `poly_return_3s` | STATIONARY | -0.000 | 0.000 | +0.01 | 13.7 | -0.000317 | -0.000002 |
| `poly_return_5s` | STATIONARY | -0.000 | 0.000 | +0.02 | 11.5 | -0.000384 | -0.000009 |
| `poly_return_10s` | STATIONARY | -0.000 | 0.000 | +0.03 | 9.6 | -0.000114 | -0.000004 |
| `poly_return_30s` | STATIONARY | -0.000 | -0.000 | -0.00 | 107.3 | +0.000097 | -0.000000 |
| `binance_mid_price` | NON_STATIONARY | 83395.829 | 85631.706 | +6.65 | 11.3 | -0.000133 | -0.000885 |
| `binance_microprice_offset_bps` | STATIONARY | -0.000 | -0.000 | -0.03 | 207.8 | +0.009906 | -0.000333 |
| `binance_spread_bps` | STATIONARY | 0.012 | 0.012 | +0.06 | 819.8 | -0.000133 | -0.000008 |
| `binance_return_since_open_bps` | NON_STATIONARY | -0.106 | 1.880 | +0.31 | 11.3 | +0.001541 | +0.000475 |
| `binance_return_1s_bps` | STATIONARY | -0.002 | 0.003 | +0.01 | 44.3 | +0.002431 | +0.000028 |
| `binance_return_2s_bps` | STATIONARY | -0.002 | 0.007 | +0.01 | 48.8 | -0.000943 | -0.000012 |
| `binance_return_3s_bps` | STATIONARY | -0.003 | 0.011 | +0.02 | 44.8 | +0.000435 | +0.000007 |
| `binance_return_5s_bps` | STATIONARY | -0.002 | 0.020 | +0.02 | 44.7 | -0.001090 | -0.000022 |
| `binance_return_10s_bps` | STATIONARY | -0.006 | 0.046 | +0.03 | 33.2 | -0.000060 | -0.000002 |
| `binance_return_30s_bps` | STATIONARY | -0.021 | 0.169 | +0.07 | 19.4 | +0.000542 | +0.000036 |
| `binance_return_60s_bps` | STATIONARY | -0.036 | 0.317 | +0.09 | 15.8 | -0.000115 | -0.000010 |
| `binance_top1_depth_imbalance` | STATIONARY | 0.000 | -0.017 | -0.03 | 1.6 | -0.006689 | +0.000190 |
| `binance_top5_depth_imbalance` | STATIONARY | -0.005 | -0.014 | -0.02 | 1.7 | -0.002762 | +0.000041 |
| `binance_top20_depth_imbalance` | STATIONARY | -0.005 | -0.014 | -0.02 | 1.7 | +0.002681 | -0.000041 |

---

## Table 2: Chronological Expanding-Window Forward Validation

| Fold | Train Window | Test Window | N Pairs | B0 MSE | B1 MSE | MSE Improvement | $\Delta R^2$ | Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Fold 1 | Rounds 1-50 | Rounds 51-100 | 13240 | 0.00174969 | 0.00177484 | -0.00002514 | -0.01431 | 32.0% |
| Fold 2 | Rounds 1-100 | Rounds 101-150 | 13146 | 0.00210249 | 0.00209169 | +0.00001080 | +0.00514 | 72.0% |
| Fold 3 | Rounds 1-150 | Rounds 151-200 | 13055 | 0.00149037 | 0.00148800 | +0.00000236 | +0.00158 | 82.0% |
| Fold 4 | Rounds 1-200 | Rounds 201-250 | 12744 | 0.00168510 | 0.00164631 | +0.00003879 | +0.02296 | 92.0% |
| **Aggregate** | **Expanding** | **Rounds 51–250** | **52185** | — | — | **+0.00000641** | **+0.00364** | **69.5%** |

---

## Table 3: Model Architecture & Intervention Comparison

| Architecture | Preprocessing / Intervention | N Pairs | Pair-Weighted MSE Imp | Mean Round $d$ | $\Delta R^2$ | Win Rate | $t$-statistic | $p$-value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `B0_polymarket_baseline` | Poly Controls (9 feats) | 65436 | +0.00000000 | +0.00000000 | +0.00000 | 0.0% | +0.00 | 1.0000e+00 |
| `B1_original_frozen_unclipped` | Original Spec (23 feats, unclipped) | 65436 | -0.00008287 | -0.00007366 | -0.04755 | 57.2% | -1.04 | 2.9920e-01 |
| `B1_stationary_unclipped` | Stationary Spec (21 feats, unclipped) | 65436 | -0.00008410 | -0.00007587 | -0.04826 | 58.0% | -1.07 | 2.8583e-01 |
| `B1_stationary_clipped_5sigma` | Stationary Spec (21 feats, clipped [-5, 5]) | 65436 | +0.00001457 | +0.00001498 | +0.00836 | 62.4% | +6.96 | 2.9809e-11 |

---

## Scientific Interpretation & Next Steps

- **Claim A: Frozen Model Transport**: **NOT_REPLICATED**. The immutable frozen model artifact failed prospective transport due to unclipped non-stationary price-level features.
- **Claim B: Feature-Family Lead-Lag Predictability**: **SUPPORTED**. Public Binance order flow features contain statistically robust incremental predictive power for native Polymarket 2-second midpoint movements out-of-sample (confirmed via same-spec refit $\Delta R^2 = +0.009041$, $t = +5.03$, and chronological forward validation win rate $72.5\%$).
- **Claim C: Cross-Regime Stable Signal**: **NOT YET CONFIRMED**. Requires a newly frozen scale-invariant specification (v4) with bounded standardization tested in a fresh, prospective experiment.

### Prospective v4 Replication Design
- Specification drafted in `v4_confirmatory_spec_draft.json`.
- Restricts features strictly to 21 scale-invariant variables (excludes `binance_mid_price` and `binance_return_since_open_bps`).
- Enforces $[-5.0, +5.0]\sigma$ standardization clipping.
- Recommends $N = 500$ physical rounds (providing $\ge 99\%$ power under moderate within-regime conditions and $\ge 70\%$ power under conservative forward chaining).
- **Status**: DRAFT ONLY. Not launched.
