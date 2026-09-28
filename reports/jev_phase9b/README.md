# Phase 9B — Jev Setup-Gating Development Benchmark Report

## 1. Executive Summary

This report documents the empirical evaluation of **TypeSafe Jev (`typesafe/jev-1.13`)** on the retrospective setup-gating task `meaningful_poly_move_30s_v1` using 500 physical rounds of `btc5m_leadlag_v1`.

### Key Findings
* **Total Physical Rounds**: 500
* **Eligible Single-Round Observations**: 480 (Exclusions: 20 due to non-positive spread or invalid sample sync)
* **Underlying Label Distribution**: 429 `MEANINGFUL_MOVE` (89.4%), 51 `QUIET`
* **Jev Coverage**: 100.0% (480/480)
* **Jev Effective Accuracy**: 31.87%
* **Majority Baseline Accuracy**: 89.38%
* **Logistic Regression Effective Accuracy**: 89.79%

---

## 2. Comparative Benchmark Table

| Model / Baseline | Coverage | Conditional Acc | Effective Acc | Balanced Acc | Brier Score | Latency (avg) |
|---|---|---|---|---|---|---|
| **TypeSafe Jev (1.13)** | **100.0%** | **31.87%** | **31.87%** | **52.39%** | **0.3962** | **556.06 ms** |
| `baseline_majority_base_rate` | 100.0% | 89.38% | 89.38% | 50.00% | 0.1062 | 0.0 ms |
| `baseline_logistic_regression` | 100.0% | 89.79% | 89.79% | 52.82% | 0.0948 | 0.0 ms |

---

## 3. Scientific Decision

**Decision**: `JEV_GATING_HYPOTHESIS_NOT_SUPPORTED_ON_V1_DEVELOPMENT_DATA` (or comparison summary)
* The base rate of meaningful moves in 30s is high (89.4%), and transparent baselines perform effectively.
* Round-level block bootstrap confirms the statistical boundaries of incremental value.
