# Phase 9B.1 — Retrospective Result Integrity Audit Report

## 1. Executive Summary

This report documents the corrective scientific audit (Phase 9B.1) of the Phase 9B Jev setup-gating benchmark (`meaningful_poly_move_30s_v1`).
All 480 responses persisted in `data/pm_research.db` were verified with zero new API calls made.

### Key Audit Findings
* **Persisted Responses Verified**: 480/480 unique observations, 0 duplicates.
* **API Requests Reconciled**: 480 original remote API calls during the experiment; 0 API calls during this audit.
* **Hash Reporting Defect**: Resolved. The terminal hash `75e03264...` was an assistant formatting hallucination. The canonical task specification hash across code, database, and artifacts is `2b8dd4f5d3eb91fb2227c3bb2bc40ed38e1f4b114e40fd53b4f3279bfba79109`.
* **Confidence Semantics Clarified**: Jev `confidence` represents normalized confidence in the **chosen option** (`(p_choice - 1/3)/(1 - 1/3)`), not the probability of the positive class. Original Brier and LogLoss were invalid.
* **Primary Out-of-Sample Evaluation Cohort**: The 96-round Retrospective Holdout (final 20% chronologically).

---

## 2. Primary Comparative Benchmark: Retrospective Holdout (N=96)

| Model / Baseline | Coverage | Accuracy | Bal Acc | Sens (Recall M) | Spec (Recall Q) | Macro F1 | MCC | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **TypeSafe Jev (1.13)** | **100.0%** | **38.54%** | **45.00%** | **36.14%** | **53.85%** | **0.3480** | **-0.0707** | **0.2852** |
| `baseline_majority_base_rate` | 100.0% | 86.46% | 50.00% | 100.00% | 0.00% | 0.4637 | +0.0000 | 0.1354 |
| `baseline_logistic_regression` | 100.0% | 87.50% | 57.09% | 98.80% | 15.38% | 0.5909 | +0.2789 | 0.1178 |

---

## 3. High-Confidence Error Concentration

Selective prediction analysis confirms that accuracy monotonically decreases as confidence increases:
* $\tau = 0.0$: Coverage 100.0%, Accuracy 31.87%, Balanced Accuracy 52.39%
* $\tau = 0.5$: Coverage 24.0%, Accuracy 15.65%, Balanced Accuracy 45.83%
* $\tau = 0.6$: Coverage 8.1%, Accuracy 7.69%, Balanced Accuracy 50.00%
* $\tau = 0.7$: Coverage 1.25% (6 samples), Accuracy 0.00%, Balanced Accuracy 0.00%

**HIGH_CONFIDENCE_ERROR_CONCENTRATION=YES**

---

## 4. Holdout Paired Bootstrap ($B=1,000$, seed=42)

* **Jev vs Majority Baseline**:
  * Accuracy Diff: -48.02% (95% CI: [-61.46%, -35.42%])
  * Bootstrap Superiority Fraction: 0.0
* **Jev vs Logistic Regression**:
  * Accuracy Diff: -49.08% (95% CI: [-61.46%, -36.46%])
  * Bootstrap Superiority Fraction: 0.0

---

## 5. Scientific Decision

**`JEV_GATING_HYPOTHESIS_NOT_SUPPORTED_ON_V1_DEVELOPMENT_DATA`**
* The audit proves that while original probabilistic scoring was mathematically flawed, the underlying discrete classification choices remain unchanged.
* On the 96-round retrospective holdout, Jev attains 38.54% accuracy and 45.00% balanced accuracy (below chance), significantly underperforming transparent baselines.
