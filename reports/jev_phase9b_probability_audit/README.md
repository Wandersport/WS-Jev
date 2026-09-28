# Phase 9B.2 — Final Jev Probability Integrity Audit Report

## 1. Executive Summary

This report delivers the final probabilistic integrity correction for the Phase 9B setup-gating benchmark (`meaningful_poly_move_30s_v1`).
All 480 persisted responses in `data/pm_research.db` were audited directly at the raw JSON level.

### Key Audit Findings
* **Raw Probability Vectors Verified**: 480/480 responses (100%) contain full, valid 3-choice probability vectors (`MEANINGFUL_MOVE`, `QUIET`, `ABSTAIN`).
* **Confidence Formula Confirmed**: Every response strictly matches:
  $$\text{confidence} = \frac{p_{\text{chosen}} - 1/3}{1 - 1/3} = 1.5 \cdot p_{\text{chosen}} - 0.5$$
  with a maximum formula error of 0.015 (median error 0.005, due entirely to 2-decimal scalar rounding).
* **Canonical Probability Scoring Defined**:
  Probabilities are extracted directly from `raw_response_json -> answers.decision.probabilities`.
  The canonical primary binary probabilistic metric evaluates the **Active-Class Conditional Probability**:
  $$p_{M\_cond} = \frac{p_M}{p_M + p_Q}$$
* **Discrete Classification Unchanged**: Frozen historical discrete choices remain completely unmodified.

---

## 2. Metric Version Comparison Across Cohorts

| Cohort | Metric | Phase 9B Naive (INVALID) | Phase 9B.1 Conf-Converted (SUPERSEDED) | Phase 9B.2 Canonical Conditional (CANONICAL) | Phase 9B.2 Unconditional (CANONICAL) |
|---|---|---|---|---|---|
| **Holdout (96)** | **Brier Score** | 0.416414 | 0.285164 | **0.348376** | 0.404816 |
| **Holdout (96)** | **Log Loss** | 1.091183 | 0.786852 | **0.934242** | 1.069447 |
| **Full (480)** | **Brier Score** | 0.396248 | 0.284581 | **0.395181** | 0.451803 |
| **Full (480)** | **Log Loss** | 1.038548 | 0.782828 | **1.046800** | 1.187580 |

---

## 3. Retrospective Holdout Discrete Classification Summary (N=96)

* **Accuracy**: 38.54% (37 / 96)
* **Balanced Accuracy**: 45.00% (Below 50% random chance)
* **Sensitivity (Meaningful Recall)**: 36.14% (30 / 83)
* **Specificity (Quiet Recall)**: 53.85% (7 / 13)
* **Macro F1**: 0.3480
* **MCC**: -0.0707

---

## 4. Final Scientific Decision

**`JEV_GATING_HYPOTHESIS_NOT_SUPPORTED_ON_V1_DEVELOPMENT_DATA`**
* The final probability audit establishes the true canonical Brier score on the retrospective holdout as **0.3484** (log loss: **0.9342**).
* Transparent baselines (Majority Base Rate at Brier 0.1354 and Standardized Logistic Regression at Brier 0.1178) substantially outperform TypeSafe Jev.
* Discrete choices remain frozen; no post-hoc label inversion or threshold tuning was permitted.
