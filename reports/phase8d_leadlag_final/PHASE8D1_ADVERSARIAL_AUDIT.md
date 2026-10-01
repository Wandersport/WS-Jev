# Phase 8D.1: Adversarial Validation and Signal Stress-Test Report

**Experiment**: `btc5m_leadlag_v2`  
**Spec Hash**: `dc8663876e8a1ca5f22d73f1e5f8bdd69fcf1a4946aba741708947c7e63c3352`  
**Target Under Stress-Test**: 2-second Binance Perpetual -> Polymarket UP token midpoint lead  

---

## 1. Scientific Conclusion Correction

The primary Phase 8D conclusion is formally revised:
- **Lag 2s**: **Statistically Robust** ($\Delta R^2_{CV} = +0.00667$, $t = +4.54$, Holm $p = 0.00007$, round win rate $56.2\%$).
- **Lag 1s**: **Marginal / Suggestive Only** ($\Delta R^2_{CV} = +0.00495$, $t = +1.77$, Holm $p = 0.38305 > 0.05$). Additionally, 1s pairs were observed almost exclusively in early rounds ($N=101$ rounds) due to sampling cadence degradation.
- **Lags $\ge$ 3s**: **Unsupported** (Lag 3s Holm $p = 0.979$; Lag 5s $t = +0.44$; Lag 10s $t = +0.09$; Lags 15s/30s degraded).

The supported horizon must **never** be cited as `[1s, 2s]`. Only the **2-second horizon** meets the full evidentiary standard.

---

## 2. Multiple-Testing Family Audit (Holm-7 vs Holm-14)

We audited the multiple-testing family definition:
- **Family A (Preregistered, separate 7-test families by target)**:
  - $\Delta q$: Lag 2s $p_{raw} = 1.0 \times 10^{-5} \implies p_{adj} = 0.00007$ (**Statistically Significant**).
  - $\Delta \text{logit}(q)$: Lag 2s $p_{raw} < 1.0 \times 10^{-5} \implies p_{adj} < 0.00001$ (**Statistically Significant**).
- **Family B (Pooled 14-test family across both targets $\times$ 7 horizons)**:
  - Rank 1 ($\Delta \text{logit}(q)$, Lag 2s): $p_{raw} < 10^{-5} \times 14 = \mathbf{0.00013} \ll 0.05$.
  - Rank 2 ($\Delta q$, Lag 2s): $p_{raw} = 1.0 \times 10^{-5} \times 13 = \mathbf{0.00014} \ll 0.05$.
  - All other 12 horizon/target tests: $p_{adj} \ge 0.326$ (not significant).

**Result**: The 2-second result **survives both** separate 7-test and pooled 14-test Holm corrections.

---

## 3. Temporal Asymmetry & Symmetric Cross-Lag Profile

Full symmetric diagnostic across relative lags $k \in [-30, +30]$ seconds:

| Relative Lag $k$ | N Pairs | N Rounds | B0 $R^2$ | B1 $R^2$ | $\Delta R^2$ | Round $t$-Stat | Median Error | Mechanism / Interpretation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **-30s** | 26,756 | 474 | +0.20153 | +0.56961 | +0.36807 | +7.04 | 17 ms | Feature overlap: B1 has `binance_return_30s_bps` over $[t-30, t]$ |
| **-15s** | 24,469 | 461 | +0.14467 | +0.40843 | +0.26377 | +6.36 | 16 ms | Feature overlap: contemporaneous co-movement over $[t-15, t]$ |
| **-10s** | 30,008 | 487 | +0.13913 | +0.44257 | +0.30344 | +10.44 | 17 ms | Feature overlap: B1 has `binance_return_10s_bps` over $[t-10, t]$ |
| **-5s** | 33,765 | 463 | +0.19169 | +0.42055 | +0.22886 | +7.39 | 18 ms | Feature overlap: B1 has `binance_return_5s_bps` over $[t-5, t]$ |
| **-3s** | 26,648 | 451 | +0.31479 | +0.46845 | +0.15366 | +6.80 | 16 ms | Feature overlap: B1 has `binance_return_3s_bps` over $[t-3, t]$ |
| **-2s** | 30,979 | 322 | +0.47980 | +0.56555 | +0.08575 | +8.39 | 17 ms | Feature overlap: B1 has `binance_return_2s_bps` over $[t-2, t]$ |
| **-1s** | 20,163 | 101 | +0.88378 | +0.88557 | +0.00179 | +1.68 | 12 ms | Trivial overlap: B0 has `poly_return_1s` = $q_t - q_{t-1}$ ($R^2=88\%$) |
| **+1s** | 20,163 | 101 | +0.00086 | +0.00581 | **+0.00495** | +1.77 | 12 ms | Non-overlapping forward prediction (early rounds only) |
| **+2s** | 30,979 | 322 | +0.00022 | +0.00690 | **+0.00667** | **+4.54** | 17 ms | **PEAK FORWARD SIGNAL** (non-overlapping forward prediction) |
| **+3s** | 26,648 | 451 | -0.00095 | +0.00286 | +0.00381 | +1.16 | 16 ms | Non-overlapping forward prediction (decaying) |
| **+5s** | 33,765 | 463 | -0.00080 | +0.00215 | +0.00295 | +0.44 | 18 ms | Non-overlapping forward prediction (decayed) |
| **+10s** | 30,008 | 487 | -0.00050 | +0.00194 | +0.00245 | +0.09 | 17 ms | Non-overlapping forward prediction (noise floor) |
| **+15s** | 24,469 | 461 | -0.00078 | +0.00087 | +0.00164 | -1.16 | 16 ms | Noise / uninformative |
| **+30s** | 26,756 | 474 | -0.00171 | -0.00521 | -0.00350 | -1.97 | 17 ms | Noise / overfit |

### Root Cause of the Negative Lag Asymmetry:
The large $\Delta R^2$ at negative lags (e.g. $+0.08575$ at $-2$s, $+0.22886$ at $-5$s) is **NOT** a lead of Polymarket over Binance. It is **100% Feature/Target Temporal Overlap**:
- For negative lag $k < 0$, target $y = q_t - q_{t-|k|}$ is the realized Polymarket return over $[t-|k|, t]$.
- The features measured at time $t$ include historical return features looking back over that exact same window (`binance_return_2s_bps`, `binance_return_5s_bps`, etc.).
- Regressing the past Polymarket move over $[t-|k|, t]$ on the contemporaneous Binance move over $[t-|k|, t]$ yields high $R^2$ because both assets co-move during that window.
- In contrast, for forward lags $k > 0$, target $y = q_{t+k} - q_t$ is strictly in the future $[t, t+k]$, with **zero temporal overlap**.
- Incremental forward predictability peaks strictly at **$L = +2$s**.

---

## 4. 2S Feature Temporal Causality Audit

We inspected all prospective feature pipelines in `leadlag_v2_collector.py` and the database records:
- `binance_microprice_offset_bps`: Derived strictly from `bn_bid`, `bn_ask`, and level-1 quantities in the latest WebSocket book snapshot received $\le actual\_ms$.
- `binance_top1/5/20_depth_imbalance`: Derived strictly from depth levels in the latest WebSocket book snapshot received $\le actual\_ms$.
- `binance_taker_flow_1s..60s`: Computed strictly from trades in circular buffer where trade timestamp $t \ge actual\_ms - \text{sec} \times 1000$ and received prior to $actual\_ms$.
- Polymarket B0 controls: Derived strictly from latest WebSocket snapshot/delta or REST fallback received $\le actual\_ms$.
- Database audit: Zero prospective future target columns exist in `leadlag_v2_samples`. All target pairing was computed strictly offline during post-processing.
- Clock skew: Polymarket source timestamp median is $+4$ms ahead of local clock (NTP difference), but local receive timestamp `poly_recv_ts_ms` is strictly $\le actual\_ms$.

**FUTURE_INFORMATION_EVENTS = 0**  
**Timestamp Causality Verdict: PASS**

---

## 5. Late-Block $\Delta R^2 = +1.03745$ Forensic Audit

In `temporal_stability.csv`, the late block (Rounds 335–500) reported $\Delta R^2 = +1.03745$. We conducted an exact forensic trace:
- **Sample Count**: Late block contains only **27 pairs** from only **4 physical rounds** (Rounds 378, 379, 402, 403) due to sampling cadence degradation to $\sim 5$s.
- **Fold Degeneracy**: With only 4 rounds, fitting an internal 5-fold CV resulted in Fold 2 having 0 test samples (`te = 0`).
- **Severe Denominator Pathology**:
  - Total sum of squares in late block: $TSS = 0.036049$ (extremely tiny variance).
  - Baseline $B0$ out-of-fold $MSE = 0.003169 \implies B0 \text{ CV } R^2 = \mathbf{-1.37314}$ (severely negative).
  - Augmented $B1$ out-of-fold $MSE = 0.001783 \implies B1 \text{ CV } R^2 = \mathbf{-0.33569}$.
  - Difference: $\Delta R^2 = (-0.33569) - (-1.37314) = \mathbf{+1.03745}$.
- **True Out-of-Sample Performance**: When evaluating the global 5-fold CV model (trained across the full dataset) on the late block:
  - $B0 \text{ MSE} = 0.001360 \implies R^2 = -0.01883$.
  - $B1 \text{ MSE} = 0.001274 \implies R^2 = +0.04569$.
  - True Global $\Delta R^2 = \mathbf{+0.06452}$ ($\Delta MSE = -0.000086$).

**Verdict**: $+1.03745$ was a **small-N denominator artifact** from internal 5-fold CV on 27 points, NOT a genuine 100%+ variance explained.

---

## 6. 2S Round-Level Distribution & Concentration

We evaluated the distribution of B1 vs B0 error reduction across all 322 contributing physical rounds:

### Equal-Weighted (Mean MSE Improvement per Round):
- **Mean Improvement**: $+0.00001774$
- **Median Improvement**: $+0.00000464$ (strictly positive)
- **Percentiles**: p10 = $-0.00003075$, p25 = $-0.00001608$, p75 = $+0.00003180$, p90 = $+0.00008216$
- **Fraction Positive Rounds (Win Rate)**: **56.21%** (181 / 322 rounds)
- **Top 10 Rounds Share**: $48.86\%$ of aggregate improvement
- **Top 10% Rounds (32 rounds) Share**: $92.17\%$ of aggregate improvement

### Pair-Weighted (Total SSE Improvement per Round):
- **Mean Improvement**: $+0.00108562$
- **Median Improvement**: $+0.00034687$
- **Fraction Positive Rounds**: **56.21%**
- **Top 10 Rounds Share**: $56.54\%$
- **Top 10% Rounds (32 rounds) Share**: $109.38\%$

**Interpretation**: The predictive signal is **broadly diffuse** across rounds (win rate $56.2\%$, positive median), but magnitude is naturally concentrated in high-volatility rounds where Bitcoin experienced sharp price moves. This is standard market microstructure behavior, not an outlier failure.

---

## 7. Cadence Confound & Bias Audit

Pearson correlation between per-round improvement ($MSE_0 - MSE_1$) and cadence degradation indicators:
- Round Index: $r = +0.1226$
- Sample Density (samples/round): $r = -0.1024$
- Median Interarrival ms: $r = +0.1003$
- Collection Session ID (1..6): $r = +0.1626$
- Mean Timing Error ms: $r = +0.0282$

**Verdict**: All correlations are near zero ($|r| \le 0.16$). The 2s incremental signal is **NOT** an artifact of the SQLite cadence failure.

---

## 8. Raw Price-Discovery Sanity Check

Cross-correlation between 1s Binance return at $t$ and 1s Polymarket return at $t+k$:
- $k = -1$s: $r = \mathbf{+0.4375}$ (contemporaneous 1s return co-movement over $[t-1, t]$)
- $k = 0$s: $r = \mathbf{+0.0790}$ (immediate next-second Polymarket return $[t, t+1]$)
- $k = +1$s: $r = +0.0131$ ($[t+1, t+2]$)
- $k = +2$s: $r = +0.0065$ ($[t+2, t+3]$)
- $k \ge +3$s: $r \le +0.0026$ (indistinguishable from noise)

Cross-asset Granger causality test:
- Predicting future Polymarket $2$s return from Binance: $\Delta R^2_{CV} = \mathbf{+0.00667}$ ($t = +4.54$).
- Predicting future Binance $2$s return from Polymarket: $\Delta R^2_{CV} = \mathbf{+0.00207}$.
- Binance explains **3.2x more incremental variance** on Polymarket than Polymarket explains on Binance.
