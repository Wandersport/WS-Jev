# Phase 9A — Jev Decision Research Lab: Specification & Architecture

## 1. Executive Summary

Phase 9A establishes the **Jev Decision Research Lab**, a dedicated software and scientific harness designed to investigate the utility of TypeSafe Jev (`typesafe/jev-1.13`) as a **fast typed decision and gating model** rather than an unconstrained probability estimator.

This research environment is strictly isolated from live execution:
* **Paper-Only & Research-Only**: Structurally incapable of live order routing, order sizing, or key management.
* **Separation from Production**: Operates in an isolated namespace (`pm_research.research.jev_lab`) without altering the frozen Phase 8C v2 high-frequency measurement collector.

---

## 2. Scientific Foundations & Prior Results

### 2.1. Why Direct Jev P(UP) Replacement Failed in Phase 7
In prospective research across 500 valid resolved physical BTC-5m rounds in Phase 7, probability calibration analysis revealed:
* **Native Polymarket Orderbook Consensus**: Brier score $\approx \mathbf{0.1405}$
* **Jev Market-Aware**: Brier score $\approx \mathbf{0.1598}$
* **Jev Full**: Brier score $\approx \mathbf{0.1611}$

The native prediction market crowd forecast substantially outperformed Jev in predicting the binary settlement of BTC 5-minute contracts. Substituting raw model output $P(\text{UP})$ for the market consensus degraded probabilistic calibration.

### 2.2. The New Jev Hypothesis: Fast Typed Decision & Gating Layer
Rather than asking Jev to output a continuous numerical probability $\hat{q} \in [0, 1]$, Phase 9A investigates Jev for **discrete structured judgment**:
1. **Setup Gating**: `GOOD_SETUP` vs `BAD_SETUP` vs `ABSTAIN` (filtering noisy, illiquid, or dislocated states).
2. **Microstructure Regime**: `TRENDING` vs `MEAN_REVERTING` vs `NEUTRAL`.
3. **Cross-Venue Signal Agreement**: `EXTERNAL_SIGNAL_COHERENT` vs `EXTERNAL_SIGNAL_CONFLICTED` vs `NO_SIGNAL`.
4. **Order Flow Toxicity**: `TOXIC` vs `NON_TOXIC` vs `UNCERTAIN`.

Phase 9A makes **zero claims** of trading edge or profitability; it constructs the auditable scientific apparatus to test these hypotheses rigorously.

---

## 3. Core Architectural Principles

### 3.1. Why Abstention is Mandatory
Viral social-media trading bots commonly force a binary action every single block, inducing excessive churning, high fees, and low edge. 
In the Jev Decision Lab:
* **`ABSTAIN` is a first-class choice** in every compatible task contract.
* **Scoring Guard**: The evaluation engine computes both *conditional accuracy* (accuracy when acting) and *effective accuracy* ($\text{Correct} / N_{\text{total}}$). This mathematically prevents a model from achieving an artificially high score by abstaining on $99\%$ of observations.

### 3.2. Why Baselines are Mandatory
Jev must never be evaluated in isolation. Every scientific evaluation benchmark compares Jev against:
1. `RandomBaselineProvider`: Deterministic seeded uniform random baseline.
2. `MajorityBaseRateProvider`: Constant output of the empirical dominant class.
3. `DeterministicRuleProvider`: Simple heuristic rules based on observable thresholds (e.g. spread or imbalance).
4. `AlwaysAbstainProvider`: Establishes the boundary behavior of complete non-participation.

Any candidate model that does not statistically beat the base-rate and heuristic rules under Bonferroni multiple-testing correction is immediately rejected.

### 3.3. Quant Strategy Vault: Hypotheses, Not Alpha
The external quant strategy repositories (`brainbrick-trades/The-Quant-Trading-Vault` and `brainbrick-trades/Quant-Trading-Strategies`) are treated strictly as **hypothesis catalogs**. They are referenced externally via `docs/STRATEGY_REFERENCE_REGISTRY.md` and are **never** cloned en masse into the local workspace. No external strategy is assumed to be profitable without prospective out-of-sample verification.

### 3.4. Division of Labor: Opus Hypothesizes, Deterministic Code Evaluates
* **Role of LLMs (Opus / Claude)**: Code authoring, research critique, and drafting formal `StrategyCandidate` proposals.
* **Role of Deterministic Code**: Fixed mathematical evaluation, multiple-testing accounting (`MultipleTestingLedger`), and pass/fail gatekeeping.
* **Separation**: Opus is never permitted to evaluate its own hypotheses or adjust rejection thresholds post-hoc.

### 3.5. Separation from Phase 8C v2
Phase 8C v2 is a **frozen high-frequency observational measurement apparatus** designed to collect 500 physical BTC-5m rounds with microsecond-stamped Binance trades and Polymarket book deltas. 
Phase 9A development operates on an isolated branch (`research/jev-phase9a`) and adds separate tables (`jev_lab_*`). It does not alter Phase 8C v2 collector constants, schemas, or execution flows.
