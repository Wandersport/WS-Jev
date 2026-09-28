# Phase 9C: Prospective Jev Decision Protocol (`btc5m_leadlag_v2`)

**Document Status**: FROZEN PROSPECTIVE RESEARCH PROTOCOL  
**Target Environment**: Strictly Paper-Only Simulation Engine  
**Dataset Target**: Future `btc5m_leadlag_v2` (500 Physical Rounds)  
**Live Execution Capability**: ZERO (Structurally incapable of live trading)

---

## 1. Executive Summary & Scientific Motivation

Phase 9B evaluated **TypeSafe Jev (`typesafe/jev-1.13`)** on retrospective data from `btc5m_leadlag_v1` on a simple binary move task (`meaningful_poly_move_30s_v1`). That evaluation strongly rejected the setup-gating hypothesis:
1. **Severe Base-Rate Imbalance**: In BTC-5m fast markets, the Polymarket midpoint moves by at least the spread in 89.4% of 30-second intervals. A naive majority class predictor achieves 86.46% accuracy on the retrospective holdout, while Jev achieved only 38.54% (with a balanced accuracy of 45.00%, below 50% random chance).
2. **High-Confidence Error Concentration**: As Jev's reported confidence increased, its accuracy monotonically dropped (31.87% at $\tau=0.0 \to 0.0\%$ at $\tau=0.7$).
3. **v1 Measurement Defect**: The retrospective v1 dataset suffered from corrupted Binance taker flow (producing zero data) and elapsed-time receive approximations rather than native exchange timestamps.

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

### Candidate Task A: Order Flow Toxicity (`flow_toxicity_30s_v2`)
* **Hypothesis**: Given rolling aggressive taker imbalance on Binance USD-M futures and Polymarket top-of-book depth, Jev can detect whether incoming taker flow is *toxic* (predicting adverse midpoint displacement greater than the half-spread).
* **Objective Label**:
  $$\text{Displacement} = \frac{q_{t+30\text{s}} - q_t}{s_t / 2}$$
  $$\text{Label} = \begin{cases} 
  \text{TOXIC\_UP} & \text{if } \text{Displacement} \ge +1.0 \\
  \text{TOXIC\_DOWN} & \text{if } \text{Displacement} \le -1.0 \\
  \text{BENIGN} & \text{if } |\text{Displacement}| < 1.0 
  \end{cases}$$
* **Symmetry**: Balanced 3-class distribution avoiding the 90/10 single-class saturation observed in Phase 9B.

### Candidate Task B: Cross-Venue Lead Coherence (`cross_venue_lead_coherent_10s_v2`)
* **Hypothesis**: When Binance USD-M futures microprice makes an abrupt dislocation $> 3$ bps within 5 seconds, Polymarket midpoint will adjust in the same direction within 10 seconds.
* **Objective Label**:
  Conditioned on $|\Delta p_{\text{binance}, t}| \ge 3 \text{ bps}$:
  $$\text{COHERENT} \iff \text{sign}(q_{t+10\text{s}} - q_t) == \text{sign}(\Delta p_{\text{binance}, t})$$
  $$\text{NOT\_COHERENT} \iff \text{sign}(q_{t+10\text{s}} - q_t) \ne \text{sign}(\Delta p_{\text{binance}, t})$$

### Candidate Task C: Short-Horizon Volatility Regime (`short_horizon_regime_60s_v2`)
* **Hypothesis**: Microstructure depth imbalance, spread dynamics, and trade arrival rates predict whether the next 60 seconds will be high-realized-volatility (active dislocation) or low-realized-volatility (stable mean-reversion).
* **Objective Label**:
  $$\sigma_{60\text{s}} = \sqrt{\sum (\Delta \ln p_i)^2}$$
  Binarized at the empirical median of the training split:
  $$\text{HIGH\_VOLATILITY} \iff \sigma_{60\text{s}} > \text{Median}_{\text{train}}(\sigma_{60\text{s}})$$
  $$\text{LOW\_VOLATILITY} \iff \sigma_{60\text{s}} \le \text{Median}_{\text{train}}(\sigma_{60\text{s}})$$

---

## 4. Prospective Preregistration & Governance Workflow

Before executing any remote LLM queries on the future v2 dataset:

1. **Pre-Freeze Specification**:
   * Compute canonical SHA-256 hash of task definition, allowed choices, state payload schema, and prompt instructions.
   * Record task in `MultipleTestingLedger` with status `REGISTERED_PRE_EVALUATION`.
2. **Strict Chronological Data Split**:
   * **Development / In-Sample Cohort**: First 60% (300 rounds).
   * **Calibration Cohort**: Next 20% (100 rounds).
   * **Prospective Holdout Cohort**: Final 20% (100 rounds). Strictly held out and evaluated once without parameter retuning.
3. **Multiple Testing Correction**:
   * Bonferroni family-wise error rate control: For $K$ tested candidate hypotheses, the nominal significance threshold is $\alpha_{\text{adj}} = 0.05 / K$.
4. **Hard Budget & Cost Ceilings**:
   * Max requests: $\le 500$ calls.
   * Hard cost ceiling: $\le \$1.00$ USD. Immediate halt if reached.
   * Idempotent replay: Responses stored in `jev_lab_responses` with cryptographic raw hashes.

---

## 5. Primary Evaluation Standards & Metrics

To avoid the evaluation defects discovered in Phase 9B:
1. **Primary Out-of-Sample Benchmark**: The primary comparative table MUST report the **Prospective Holdout Cohort (final 20%)**, not the combined dataset.
2. **Balanced Metrics as Primary**:
   * Balanced Accuracy: $(\text{Sensitivity} + \text{Specificity}) / 2$
   * Macro F1
   * Matthews Correlation Coefficient (MCC)
   * Sensitivity & Specificity separately
3. **Confidence Scoring Semantics**:
   * Model confidence must be explicitly mapped to the chosen option:
     $$P(\text{Positive}) = \text{confidence if choice == POSITIVE else } (1.0 - \text{confidence})$$
   * Abstentions must be excluded from binary Brier / LogLoss calculation.
4. **Round-Level Block Bootstrap**:
   * Minimum 1,000 resamples at the physical round level.
   * Report `bootstrap_superiority_fraction` and 95% bootstrap CI (not labeled as frequentist p-values).

---

## 6. Zero Execution & Safety Mandate

* Strictly paper-only: All forecasts feed the local `PaperBroker` simulation engine only.
* Zero wallet integrations, zero private keys, zero live execution code.
* Unauthenticated public read-only market data ingestion.
