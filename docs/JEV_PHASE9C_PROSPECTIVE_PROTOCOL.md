# Phase 9C: Prospective Jev Decision Protocol (`btc5m_leadlag_v2`)

**Document Status**: FROZEN PROSPECTIVE RESEARCH PROTOCOL  
**Target Environment**: Strictly Statistical & Probabilistic Microstructure Research (Zero Execution)  
**Dataset Target**: Future `btc5m_leadlag_v2` (500 Physical Rounds)  
**Execution Capability**: ZERO (Positions=0, Orders=0, Fills=0, PnL=0, PaperBroker Calls=0; Structurally Incapable of Live Trading)

---

## 1. Executive Summary & Scientific Motivation

Phase 9B evaluated **TypeSafe Jev (`typesafe/jev-1.13`)** on retrospective data from `btc5m_leadlag_v1` on a simple binary move task (`meaningful_poly_move_30s_v1`). That evaluation strongly rejected the setup-gating hypothesis:
1. **Severe Base-Rate Imbalance**: In BTC-5m fast markets, the Polymarket midpoint moves by at least the spread in 89.4% of 30-second intervals. A naive majority class predictor achieves 86.46% accuracy on the retrospective holdout, while Jev achieved only 38.54% (with a balanced accuracy of 45.00%, below 50% random chance).
2. **High-Confidence Error Concentration**: As Jev's reported confidence increased, its accuracy monotonically dropped (31.87% at $\tau=0.0 \to 0.0\%$ at $\tau=0.7$).
3. **v1 Measurement Defect**: The retrospective v1 dataset suffered from corrupted Binance taker flow (producing zero data) and elapsed-time receive approximations rather than native exchange timestamps.
4. **Probabilistic Scoring Flaw Resolved**: Phase 9B.2 proved Jev's scalar confidence is an affine re-centering of choice probability ($(p_{\text{chosen}} - 1/K)/(1 - 1/K)$ with $K=3$). All probabilistic evaluations must extract exact probability vectors directly from `answers.decision.probabilities`.

Phase 9C defines the **prospective protocol** to evaluate TypeSafe Jev once the 500-round `btc5m_leadlag_v2` physical collection completes. Phase 9C **must not repeat** the unbalanced setup-gating task on v1 and **must not make any API calls** until v2 data is fully collected and verified.

---

## 2. New Scientific Capabilities of `btc5m_leadlag_v2`

The prospective protocol leverages the validated instrumentation improvements in `leadlag_v2_collector`:
* **Real Binance Taker Flow**: High-frequency trade events ingested from USD-M Futures `btcusdt@trade` (`e="trade"`, trade ID `t`, buyer maker flag `m`). Provides genuine, non-zero aggressive taker order flow across rolling windows (1s, 2s, 3s, 5s, 10s, 30s, 60s).
* **Native Exchange Timestamps**: Preserves exchange matching-engine timestamps (`poly_source_ts_ms` $\ge 95\%$ coverage) and Binance trade event timestamps (`binance_source_ts_ms` $\ge 99\%$ coverage).
* **Lossless Raw Compression**: Every raw book and trade frame archived losslessly in zlib-compressed payloads for full auditability and zero lookahead verification.

---

## 3. Candidate Prospective Typed Decision Tasks

Rather than selecting a single prompt post-hoc, Phase 9C establishes three candidate typed tasks with **strictly objective, deterministic labels** based on high-frequency microstructure dynamics:

### Candidate Task A: Polymarket Displacement Regime (`poly_displacement_regime_30s_v2`)
* **Scientific Motivation**: Midpoint displacement over 30s without an executed fill is a displacement regime, not fill toxicity. Given rolling aggressive taker imbalance on Binance USD-M futures and Polymarket top-of-book depth, Jev predicts the direction and scale of Polymarket midpoint displacement relative to half-spread.
* **Anchor**: Observation at approximately 120s remaining in the physical round ($t \in [110\text{s}, 130\text{s}]$).
* **Objective Label**:
  $$\text{Displacement} = \frac{q_{t+30\text{s}} - q_t}{s_t / 2}$$
  $$\text{Label} = \begin{cases} 
  \text{UP\_DISPLACEMENT} & \text{if } \text{Displacement} \ge +1.0 \\
  \text{DOWN\_DISPLACEMENT} & \text{if } \text{Displacement} \le -1.0 \\
  \text{BENIGN} & \text{if } |\text{Displacement}| < 1.0 
  \end{cases}$$
* **Allowed Choices**: `UP_DISPLACEMENT`, `DOWN_DISPLACEMENT`, `BENIGN`, `ABSTAIN`.
* **Symmetry**: Balanced 3-class distribution avoiding the 90/10 single-class saturation observed in Phase 9B.

### Candidate Task B: Cross-Venue Lead Coherence (`cross_venue_lead_coherent_10s_v2`)
* **Scientific Motivation**: When Binance USD-M futures microprice makes an abrupt dislocation $\ge 3$ bps within 5 seconds, Polymarket midpoint will adjust in the same direction within 10 seconds.
* **Trigger Condition**: Conditioned on the **FIRST** event in the round occurring between 180s and 60s remaining where Binance microprice dislocation satisfies:
  $$|\Delta p_{\text{binance}, 5\text{s}}| = \left|\frac{p_{\text{micro}, t} - p_{\text{micro}, t-5\text{s}}}{p_{\text{micro}, t-5\text{s}}}\right| \ge 3 \text{ bps}$$
* **Objective Label**:
  Conditioned on the trigger event:
  $$\Delta q_{\text{poly}, 10\text{s}} = q_{t+10\text{s}} - q_t$$
  $$\text{Label} = \begin{cases}
  \text{COHERENT} & \text{if } \text{sign}(\Delta q_{\text{poly}, 10\text{s}}) == \text{sign}(\Delta p_{\text{binance}, 5\text{s}}) \text{ and } |\Delta q_{\text{poly}, 10\text{s}}| > 0 \\
  \text{NOT\_COHERENT} & \text{otherwise}
  \end{cases}$$
* **Allowed Choices**: `COHERENT`, `NOT_COHERENT`, `ABSTAIN`.

### Candidate Task C: Short-Horizon Volatility Regime (`short_horizon_regime_60s_v2`)
* **Scientific Motivation**: Synchronized 1-second Binance mid return standard deviation over 60 seconds predicts whether the immediate horizon will experience heightened volatility or calm.
* **Anchor**: Observation at approximately 120s remaining in the physical round ($t \in [110\text{s}, 130\text{s}]$).
* **Objective Label**:
  $$\sigma_{60\text{s}} = \sqrt{\frac{1}{59}\sum_{i=1}^{60} (r_i - \bar{r})^2}, \quad r_i = \ln(p_i / p_{i-1})$$
  Binarized against the **frozen empirical median of the training split** (first 300 rounds):
  $$\text{Label} = \begin{cases}
  \text{HIGH\_VOLATILITY} & \text{if } \sigma_{60\text{s}} > \text{Median}_{\text{train}}(\sigma_{60\text{s}}) \\
  \text{LOW\_VOLATILITY} & \text{if } \sigma_{60\text{s}} \le \text{Median}_{\text{train}}(\sigma_{60\text{s}})
  \end{cases}$$
* **Allowed Choices**: `HIGH_VOLATILITY`, `LOW_VOLATILITY`, `ABSTAIN`.

---

## 4. Prospective Preregistration & Governance Workflow

Before executing any remote LLM queries on the future v2 dataset:

1. **Pre-Freeze Specification**:
   * Compute canonical SHA-256 hash of each task definition, allowed choices, state payload schema, and prompt instructions.
   * Compute a composite Family Manifest SHA-256 hash across all 3 tasks.
   * Record tasks in `MultipleTestingLedger` with status `REGISTERED_PRE_EVALUATION`.
2. **Strict Chronological Data Split**:
   * **Development / In-Sample Cohort**: First 60% (300 rounds, rounds 1–300). ZERO remote LLM calls permitted.
   * **Calibration Cohort**: Next 20% (100 rounds, rounds 301–400). ZERO remote LLM calls permitted.
   * **Prospective Holdout Cohort**: Final 20% (100 rounds, rounds 401–500). Evaluated strictly once without parameter retuning.
3. **Multiple Testing Correction**:
   * Bonferroni family-wise error rate control: For $K=3$ tested candidate hypotheses, the nominal significance threshold is:
     $$\alpha_{\text{adj}} = \frac{0.05}{3} \approx 0.0166667$$
4. **Hard Budget & Cost Ceilings**:
   * Remote calls strictly restricted to the Prospective Holdout Cohort (rounds 401–500).
   * Request cap: $\le 100$ calls per task, total $\le 300$ calls across the 3-task family.
   * Hard cost ceiling: $\le \$1.00$ USD total. Immediate halt if exceeded.
   * Idempotent replay: Responses stored in `jev_lab_responses` with cryptographic raw hashes.
5. **Minimum Viability Rules (Pre-Evaluation Gates)**:
   * **Sample Size**: $\ge 200$ total valid observations across dataset for Tasks A and C; $\ge 100$ qualifying rounds for Task B.
   * **Class Balance**: No required class may comprise $< 10\%$ of observations in any cohort (preventing degenerate single-class dominance).
   * **Microstructure Quality**: $\ge 95\%$ source timestamp coverage on Polymarket and Binance; taker-flow coverage $\ge 95\%$ post-warmup.

---

## 5. Primary Evaluation Standards & Metrics

To guarantee probabilistic and statistical rigor:
1. **Primary Out-of-Sample Benchmark**: The primary comparative table MUST report the **Prospective Holdout Cohort (rounds 401–500)**, not the combined dataset.
2. **Balanced Metrics as Primary Discrete Metrics**:
   * Balanced Accuracy: $(\text{Sensitivity} + \text{Specificity}) / 2$
   * Macro F1
   * Matthews Correlation Coefficient (MCC)
   * Sensitivity & Specificity separately
3. **Canonical Probabilistic Metrics**:
   * Probabilities extracted directly from `answers.decision.probabilities`.
   * **Active-Class Conditional Probability** (Primary):
     $$p_{k,\text{cond}} = \frac{p_k}{\sum_{j \in \text{active}} p_j}$$
   * **Unconditional Probability** (Secondary): $p_k$ directly.
   * Verify scalar confidence conforms to:
     $$\text{confidence} = \frac{p_{\text{chosen}} - 1/K}{1 - 1/K}$$
   * Scalar confidence must NEVER be treated directly as class probability.
4. **Round-Level Block Bootstrap**:
   * Minimum 1,000 resamples blocked at the physical round level.
   * Report `bootstrap_superiority_fraction` and 95% bootstrap CI (not labeled as frequentist p-values).

---

## 6. Zero Execution & Safety Mandate

* **Pure Statistical Research**: Phase 9C is strictly decision research and probability calibration.
* **Zero Simulation Trading**: Positions=0, orders=0, fills=0, PnL=0.
* **PaperBroker Calls = 0**: The `PaperBroker` execution engine is NOT called during Phase 9C.
* **Zero Live Execution Components**: No wallets, private keys, live brokers, transaction signers, or API trading credentials.
