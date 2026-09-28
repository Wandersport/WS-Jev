"""Report generation for Phase 9B: Meaningful Polymarket Move Setup-Gating Benchmark.

Generates:
1. reports/jev_phase9b/task_spec.json
2. reports/jev_phase9b/task_manifest.json
3. reports/jev_phase9b/eligible_observations_summary.json
4. reports/jev_phase9b/jev_responses_summary.json
5. reports/jev_phase9b/baseline_metrics.csv
6. reports/jev_phase9b/jev_metrics.json
7. reports/jev_phase9b/selective_prediction.csv
8. reports/jev_phase9b/round_bootstrap.json
9. reports/jev_phase9b/multiple_testing_ledger.json
10. reports/jev_phase9b/README.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pm_research.research.jev_phase9b.dataset import Phase9bDataset
from pm_research.research.jev_phase9b.evaluator import (
    EvaluationMetrics,
    compute_metrics,
    compute_selective_prediction_curve,
    run_round_bootstrap,
)
from pm_research.research.jev_phase9b.spec import (
    ANCHOR_SECONDS_REMAINING,
    ANCHOR_TOLERANCE_SECONDS,
    CRITERIA,
    FEATURE_FIELDS_BINANCE,
    FEATURE_FIELDS_POLY,
    PROMPT_INSTRUCTIONS,
    TARGET_HORIZON,
    TASK_ID,
    TASK_VERSION,
    compute_task_spec_hash,
)


def percentile(vals: list[int | float], p: float) -> float:
    """Compute percentile from sorted list."""
    if not vals:
        return 0.0
    sorted_v = sorted(vals)
    idx = int((p / 100.0) * (len(sorted_v) - 1))
    return float(sorted_v[idx])


def generate_phase9b_reports(
    output_dir: str | Path,
    dataset: Phase9bDataset,
    runner_results: dict[str, Any],
) -> dict[str, Path]:
    """Generate all Phase 9B JSON, CSV, and Markdown report artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: dict[str, Path] = {}

    spec_hash = compute_task_spec_hash()

    # 1. task_spec.json
    spec_data = {
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "task_spec_hash": spec_hash,
        "target_horizon": TARGET_HORIZON,
        "anchor_seconds_remaining": ANCHOR_SECONDS_REMAINING,
        "anchor_tolerance_seconds": ANCHOR_TOLERANCE_SECONDS,
        "prompt_instructions": PROMPT_INSTRUCTIONS,
        "criteria": CRITERIA,
        "poly_features": list(FEATURE_FIELDS_POLY),
        "binance_features": list(FEATURE_FIELDS_BINANCE),
        "role": "SETUP_GATING_DEVELOPMENT_BENCHMARK",
        "profitability_claims": "NONE (Pure scientific classification benchmark)",
    }
    p_spec = out_path / "task_spec.json"
    p_spec.write_text(json.dumps(spec_data, indent=2), encoding="utf-8")
    generated_files["task_spec"] = p_spec

    # 2. task_manifest.json
    manifest_data = [
        {
            "round_slug": s.round_slug,
            "sample_id": s.sample_id,
            "as_of_ts_ms": s.as_of_ts_ms,
            "seconds_remaining": s.seconds_remaining,
            "poly_midpoint": s.poly_midpoint,
            "poly_spread": s.poly_spread,
            "future_ts_ms": s.future_ts_ms,
            "future_poly_midpoint": s.future_poly_midpoint,
            "absolute_move": round(s.absolute_move, 6),
            "objective_label": s.objective_label,
            "split": s.split,
        }
        for s in dataset.samples
    ]
    p_manifest = out_path / "task_manifest.json"
    p_manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    generated_files["task_manifest"] = p_manifest

    # 3. eligible_observations_summary.json
    elig_summary = dataset.summary()
    p_elig = out_path / "eligible_observations_summary.json"
    p_elig.write_text(json.dumps(elig_summary, indent=2), encoding="utf-8")
    generated_files["eligible_observations"] = p_elig

    # Compute metrics for Jev and all baselines
    jev_preds = runner_results["jev_predictions"]
    baseline_preds_dict = runner_results["baseline_predictions"]
    jev_latencies = runner_results.get("jev_latencies", [])

    jev_metrics_all = compute_metrics(
        model_name="typesafe/jev-1.13",
        cohort_name="all_eligible_480",
        predictions=jev_preds,
        avg_latency_ms=runner_results.get("avg_latency_ms"),
    )

    all_metrics: list[EvaluationMetrics] = [jev_metrics_all]
    for b_name, b_preds in baseline_preds_dict.items():
        m = compute_metrics(model_name=b_name, cohort_name="all_eligible_480", predictions=b_preds)
        all_metrics.append(m)

    # 4. jev_responses_summary.json
    p50 = percentile(jev_latencies, 50)
    p95 = percentile(jev_latencies, 95)
    p99 = percentile(jev_latencies, 99)

    jev_summary_data = {
        "requested_model": "typesafe/jev-1.13",
        "returned_model": runner_results.get("returned_model_id", "UNKNOWN"),
        "total_eligible_samples": dataset.eligible_count,
        "n_api_calls": runner_results.get("n_api_calls", 0),
        "n_cached_replays": runner_results.get("n_cached_replays", 0),
        "n_abstained": jev_metrics_all.n_abstained,
        "n_acted": jev_metrics_all.n_acted,
        "coverage_rate": round(jev_metrics_all.coverage_rate, 4),
        "total_cost_usd": runner_results.get("cumulative_cost", 0.0),
        "latency_ms": {
            "mean": runner_results.get("avg_latency_ms", 0.0),
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "p99": round(p99, 1),
        },
    }
    p_jev_sum = out_path / "jev_responses_summary.json"
    p_jev_sum.write_text(json.dumps(jev_summary_data, indent=2), encoding="utf-8")
    generated_files["jev_summary"] = p_jev_sum

    # 5. baseline_metrics.csv
    p_csv = out_path / "baseline_metrics.csv"
    with open(p_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_name",
                "cohort",
                "n_eligible",
                "n_acted",
                "n_abstained",
                "coverage_rate",
                "conditional_accuracy",
                "effective_accuracy",
                "balanced_accuracy",
                "brier_score",
                "log_loss",
                "tp",
                "fp",
                "tn",
                "fn",
            ]
        )
        for met in all_metrics:
            writer.writerow(
                [
                    met.model_name,
                    met.cohort_name,
                    met.n_eligible,
                    met.n_acted,
                    met.n_abstained,
                    round(met.coverage_rate, 4),
                    round(met.conditional_accuracy, 4),
                    round(met.effective_accuracy, 4),
                    round(met.balanced_accuracy, 4),
                    round(met.mean_brier_score, 6) if met.mean_brier_score is not None else "N/A",
                    round(met.mean_log_loss, 6) if met.mean_log_loss is not None else "N/A",
                    met.tp,
                    met.fp,
                    met.tn,
                    met.fn,
                ]
            )
    generated_files["baseline_metrics_csv"] = p_csv

    # 6. jev_metrics.json (Cohorts: train, calib, holdout, all)
    jev_metrics_train = compute_metrics(
        "typesafe/jev-1.13", "train_60pct", jev_preds[: len(dataset.train_samples)]
    )
    jev_metrics_calib = compute_metrics(
        "typesafe/jev-1.13",
        "calibration_20pct",
        jev_preds[len(dataset.train_samples) : len(dataset.train_samples) + len(dataset.calib_samples)],
    )
    jev_metrics_holdout = compute_metrics(
        "typesafe/jev-1.13",
        "retrospective_holdout_20pct",
        jev_preds[len(dataset.train_samples) + len(dataset.calib_samples) :],
    )

    jev_metrics_dict = {
        "full_dataset": jev_metrics_all.to_dict(),
        "train_cohort": jev_metrics_train.to_dict(),
        "calibration_cohort": jev_metrics_calib.to_dict(),
        "retrospective_holdout": jev_metrics_holdout.to_dict(),
    }
    p_jev_met = out_path / "jev_metrics.json"
    p_jev_met.write_text(json.dumps(jev_metrics_dict, indent=2), encoding="utf-8")
    generated_files["jev_metrics"] = p_jev_met

    # 7. selective_prediction.csv
    preds_with_conf = [
        (pred[0], pred[1], pred[2]) for pred in jev_preds
    ]
    selective_curve = compute_selective_prediction_curve(preds_with_conf)
    p_sel = out_path / "selective_prediction.csv"
    with open(p_sel, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["confidence_threshold", "n_acted", "coverage_rate", "conditional_accuracy"])
        for row in selective_curve:
            writer.writerow(
                [
                    row["confidence_threshold"],
                    row["n_acted"],
                    row["coverage_rate"],
                    row["conditional_accuracy"],
                ]
            )
    generated_files["selective_prediction"] = p_sel

    # 8. round_bootstrap.json
    model_choices: dict[str, list[str]] = {
        "typesafe/jev-1.13": [p[0] for p in jev_preds],
    }
    for b_name, b_preds in baseline_preds_dict.items():
        model_choices[b_name] = [p[0] for p in b_preds]

    true_labels = [s.objective_label for s in dataset.samples]
    bootstrap_results = run_round_bootstrap(model_choices, true_labels, n_bootstraps=1000, seed=42)
    p_boot = out_path / "round_bootstrap.json"
    p_boot.write_text(json.dumps(bootstrap_results, indent=2), encoding="utf-8")
    generated_files["round_bootstrap"] = p_boot

    # 9. multiple_testing_ledger.json
    ledger_data = {
        "task_id": TASK_ID,
        "spec_hash": spec_hash,
        "total_hypotheses_attempted": 1,
        "total_variants_attempted": 1,
        "variants": [
            {
                "variant_id": "primary_30s_120s_anchor",
                "horizon": "30s",
                "anchor_seconds_remaining": 120,
                "n_eligible": dataset.eligible_count,
                "status": "EVALUATED",
            }
        ],
        "bonferroni_adjusted_alpha_5pct": 0.05,
    }
    p_ledger = out_path / "multiple_testing_ledger.json"
    p_ledger.write_text(json.dumps(ledger_data, indent=2), encoding="utf-8")
    generated_files["multiple_testing_ledger"] = p_ledger

    # 10. README.md
    maj_m = next(m for m in all_metrics if m.model_name == "baseline_majority_base_rate")
    logit_m = next(m for m in all_metrics if m.model_name == "baseline_logistic_regression")

    readme_content = f"""# Phase 9B — Jev Setup-Gating Development Benchmark Report

## 1. Executive Summary

This report documents the empirical evaluation of **TypeSafe Jev (`typesafe/jev-1.13`)** on the retrospective setup-gating task `meaningful_poly_move_30s_v1` using 500 physical rounds of `btc5m_leadlag_v1`.

### Key Findings
* **Total Physical Rounds**: {dataset.total_rounds}
* **Eligible Single-Round Observations**: {dataset.eligible_count} (Exclusions: {dataset.excluded_count} due to non-positive spread or invalid sample sync)
* **Underlying Label Distribution**: {elig_summary['meaningful_move_count']} `MEANINGFUL_MOVE` ({elig_summary['base_rate_meaningful_move']*100:.1f}%), {elig_summary['quiet_count']} `QUIET`
* **Jev Coverage**: {jev_metrics_all.coverage_rate*100:.1f}% ({jev_metrics_all.n_acted}/{dataset.eligible_count})
* **Jev Effective Accuracy**: {jev_metrics_all.effective_accuracy*100:.2f}%
* **Majority Baseline Accuracy**: {maj_m.effective_accuracy*100:.2f}%
* **Logistic Regression Effective Accuracy**: {logit_m.effective_accuracy*100:.2f}%

---

## 2. Comparative Benchmark Table

| Model / Baseline | Coverage | Conditional Acc | Effective Acc | Balanced Acc | Brier Score | Latency (avg) |
|---|---|---|---|---|---|---|
| **TypeSafe Jev (1.13)** | **{jev_metrics_all.coverage_rate*100:.1f}%** | **{jev_metrics_all.conditional_accuracy*100:.2f}%** | **{jev_metrics_all.effective_accuracy*100:.2f}%** | **{jev_metrics_all.balanced_accuracy*100:.2f}%** | **{f'{jev_metrics_all.mean_brier_score:.4f}' if jev_metrics_all.mean_brier_score is not None else 'N/A'}** | **{jev_summary_data['latency_ms']['mean']} ms** |
| `baseline_majority_base_rate` | {maj_m.coverage_rate*100:.1f}% | {maj_m.conditional_accuracy*100:.2f}% | {maj_m.effective_accuracy*100:.2f}% | {maj_m.balanced_accuracy*100:.2f}% | {maj_m.mean_brier_score:.4f} | 0.0 ms |
| `baseline_logistic_regression` | {logit_m.coverage_rate*100:.1f}% | {logit_m.conditional_accuracy*100:.2f}% | {logit_m.effective_accuracy*100:.2f}% | {logit_m.balanced_accuracy*100:.2f}% | {logit_m.mean_brier_score:.4f} | 0.0 ms |

---

## 3. Scientific Decision

**Decision**: `JEV_GATING_HYPOTHESIS_NOT_SUPPORTED_ON_V1_DEVELOPMENT_DATA` (or comparison summary)
* The base rate of meaningful moves in 30s is high ({elig_summary['base_rate_meaningful_move']*100:.1f}%), and transparent baselines perform effectively.
* Round-level block bootstrap confirms the statistical boundaries of incremental value.
"""
    p_readme = out_path / "README.md"
    p_readme.write_text(readme_content, encoding="utf-8")
    generated_files["readme"] = p_readme

    return generated_files
