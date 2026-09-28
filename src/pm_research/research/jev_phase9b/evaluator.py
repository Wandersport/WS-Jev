"""Evaluation metrics, round-level bootstrap, and selective prediction analysis for Phase 9B.

Calculates:
1. Coverage, abstention rate, conditional accuracy, effective accuracy, balanced accuracy.
2. Confusion matrix (TP, FP, TN, FN, Abstained).
3. Brier score and log loss where probabilistic forecasts are provided.
4. Round-level bootstrap (1,000 resamples) for confidence intervals and paired difference testing.
5. Selective prediction curves across confidence thresholds.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from pm_research.research.jev_lab.contract import ABSTAIN_CHOICE
from pm_research.research.jev_phase9b.spec import (
    CHOICE_MEANINGFUL_MOVE,
    CHOICE_QUIET,
)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a model on a cohort."""

    model_name: str
    cohort_name: str
    n_eligible: int
    n_acted: int
    n_abstained: int
    coverage_rate: float
    abstention_rate: float
    conditional_accuracy: float
    effective_accuracy: float
    balanced_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    mean_brier_score: float | None
    mean_log_loss: float | None
    avg_latency_ms: float | None = None
    total_cost: float | None = None
    sensitivity: float = 0.0
    specificity: float = 0.0
    macro_f1: float = 0.0
    mcc: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "cohort_name": self.cohort_name,
            "n_eligible": self.n_eligible,
            "n_acted": self.n_acted,
            "n_abstained": self.n_abstained,
            "coverage_rate": round(self.coverage_rate, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "conditional_accuracy": round(self.conditional_accuracy, 4),
            "effective_accuracy": round(self.effective_accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "macro_f1": round(self.macro_f1, 4),
            "mcc": round(self.mcc, 4),
            "confusion_matrix": {
                "tp": self.tp,
                "fp": self.fp,
                "tn": self.tn,
                "fn": self.fn,
                "abstained": self.n_abstained,
            },
            "mean_brier_score": (
                round(self.mean_brier_score, 6) if self.mean_brier_score is not None else None
            ),
            "mean_log_loss": (
                round(self.mean_log_loss, 6) if self.mean_log_loss is not None else None
            ),
            "avg_latency_ms": (
                round(self.avg_latency_ms, 2) if self.avg_latency_ms is not None else None
            ),
            "total_cost": (
                round(self.total_cost, 6) if self.total_cost is not None else None
            ),
        }


def convert_binary_confidence_to_positive_probability(
    choice: str,
    confidence: float | None,
    positive_choice: str = CHOICE_MEANINGFUL_MOVE,
    negative_choice: str = CHOICE_QUIET,
) -> float | None:
    """Convert confidence in chosen binary label to probability of positive class."""
    if confidence is None or choice == ABSTAIN_CHOICE:
        return None
    if choice == positive_choice:
        return float(confidence)
    elif choice == negative_choice:
        return float(1.0 - confidence)
    return None


def compute_metrics(
    model_name: str,
    cohort_name: str,
    predictions: list[tuple[str, str, float | None, float | None]],
    # tuple format: (predicted_choice, true_label, probability_or_confidence, cost)
    avg_latency_ms: float | None = None,
) -> EvaluationMetrics:
    """Compute all evaluation metrics from a list of (predicted, true, prob, cost) tuples."""
    n_eligible = len(predictions)
    if n_eligible == 0:
        return EvaluationMetrics(
            model_name=model_name,
            cohort_name=cohort_name,
            n_eligible=0,
            n_acted=0,
            n_abstained=0,
            coverage_rate=0.0,
            abstention_rate=0.0,
            conditional_accuracy=0.0,
            effective_accuracy=0.0,
            balanced_accuracy=0.0,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
            mean_brier_score=None,
            mean_log_loss=None,
        )

    n_abstained = 0
    tp = fp = tn = fn = 0
    brier_scores: list[float] = []
    log_losses: list[float] = []
    total_cost = 0.0

    for pred_choice, true_label, prob, cost in predictions:
        if cost is not None:
            total_cost += cost

        if pred_choice == ABSTAIN_CHOICE:
            n_abstained += 1
            continue

        is_pos = true_label == CHOICE_MEANINGFUL_MOVE
        if pred_choice == CHOICE_MEANINGFUL_MOVE:
            if is_pos:
                tp += 1
            else:
                fp += 1
        elif pred_choice == CHOICE_QUIET:
            if not is_pos:
                tn += 1
            else:
                fn += 1

        if prob is not None:
            y = 1.0 if is_pos else 0.0
            p = max(1e-6, min(1.0 - 1e-6, prob))
            brier_scores.append((p - y) ** 2)
            log_losses.append(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))

    n_acted = n_eligible - n_abstained
    correct_when_acted = tp + tn
    coverage_rate = n_acted / n_eligible
    abstention_rate = n_abstained / n_eligible
    conditional_acc = (correct_when_acted / n_acted) if n_acted > 0 else 0.0
    effective_acc = correct_when_acted / n_eligible

    # Balanced accuracy: (Sensitivity + Specificity) / 2
    sens = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = (sens + spec) / 2.0 if n_acted > 0 else 0.0

    # Macro F1
    prec_pos = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1_pos = (2.0 * prec_pos * sens / (prec_pos + sens)) if (prec_pos + sens) > 0 else 0.0
    prec_neg = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1_neg = (2.0 * prec_neg * spec / (prec_neg + spec)) if (prec_neg + spec) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2.0

    # Matthews Correlation Coefficient (MCC)
    mcc_num = float(tp * tn - fp * fn)
    mcc_den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (mcc_num / mcc_den) if mcc_den > 0 else 0.0

    mean_brier = (sum(brier_scores) / len(brier_scores)) if brier_scores else None
    mean_ll = (sum(log_losses) / len(log_losses)) if log_losses else None

    return EvaluationMetrics(
        model_name=model_name,
        cohort_name=cohort_name,
        n_eligible=n_eligible,
        n_acted=n_acted,
        n_abstained=n_abstained,
        coverage_rate=coverage_rate,
        abstention_rate=abstention_rate,
        conditional_accuracy=conditional_acc,
        effective_accuracy=effective_acc,
        balanced_accuracy=balanced_acc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        mean_brier_score=mean_brier,
        mean_log_loss=mean_ll,
        avg_latency_ms=avg_latency_ms,
        total_cost=total_cost if total_cost > 0 else None,
        sensitivity=sens,
        specificity=spec,
        macro_f1=macro_f1,
        mcc=mcc,
    )


def compute_selective_prediction_curve(
    predictions_with_conf: list[tuple[str, str, float | None]],
    # tuple: (predicted_choice, true_label, confidence)
    thresholds: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[dict[str, Any]]:
    """Evaluate coverage, conditional accuracy, and balanced accuracy across confidence thresholds."""
    results: list[dict[str, Any]] = []

    for tau in thresholds:
        acted_count = 0
        correct_count = 0
        tp_tau = fp_tau = tn_tau = fn_tau = 0

        for pred_choice, true_label, conf in predictions_with_conf:
            if pred_choice == ABSTAIN_CHOICE:
                continue

            c = conf if conf is not None else 1.0
            if c >= tau:
                acted_count += 1
                is_pos = true_label == CHOICE_MEANINGFUL_MOVE
                if pred_choice == CHOICE_MEANINGFUL_MOVE:
                    if is_pos:
                        tp_tau += 1
                    else:
                        fp_tau += 1
                elif pred_choice == CHOICE_QUIET:
                    if not is_pos:
                        tn_tau += 1
                    else:
                        fn_tau += 1

                if pred_choice == true_label:
                    correct_count += 1

        total = len(predictions_with_conf)
        cov = (acted_count / total) if total > 0 else 0.0
        acc = (correct_count / acted_count) if acted_count > 0 else 0.0

        sens_tau = (tp_tau / (tp_tau + fn_tau)) if (tp_tau + fn_tau) > 0 else 0.0
        spec_tau = (tn_tau / (tn_tau + fp_tau)) if (tn_tau + fp_tau) > 0 else 0.0
        bal_acc = (sens_tau + spec_tau) / 2.0 if acted_count > 0 else 0.0

        results.append(
            {
                "confidence_threshold": tau,
                "n_acted": acted_count,
                "coverage_rate": round(cov, 4),
                "conditional_accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
            }
        )

    return results


def run_round_bootstrap(
    model_predictions: dict[str, list[str]],  # model_name -> list of predicted choices
    true_labels: list[str],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Perform round-level block bootstrap to establish empirical uncertainty intervals."""
    rng = random.Random(seed)
    N = len(true_labels)
    models = list(model_predictions.keys())

    # Pre-allocate bootstrap accuracies and balanced accuracies
    model_effective_accs: dict[str, list[float]] = {m: [] for m in models}
    model_balanced_accs: dict[str, list[float]] = {m: [] for m in models}
    paired_diffs_acc: dict[str, list[float]] = {}
    paired_diffs_bal_acc: dict[str, list[float]] = {}
    jev_key = "typesafe/jev-1.13" if "typesafe/jev-1.13" in models else models[0]

    for m in models:
        if m != jev_key:
            paired_diffs_acc[f"{jev_key}_minus_{m}"] = []
            paired_diffs_bal_acc[f"{jev_key}_minus_{m}"] = []

    for _ in range(n_bootstraps):
        # Sample round indices with replacement
        sample_indices = [rng.randint(0, N - 1) for _ in range(N)]

        accs_this_round: dict[str, float] = {}
        bal_accs_this_round: dict[str, float] = {}

        for m in models:
            preds = model_predictions[m]
            tp = fp = tn = fn = 0
            correct = 0

            for idx in sample_indices:
                p_c = preds[idx]
                t_c = true_labels[idx]
                if p_c == t_c:
                    correct += 1
                if p_c == CHOICE_MEANINGFUL_MOVE:
                    if t_c == CHOICE_MEANINGFUL_MOVE:
                        tp += 1
                    else:
                        fp += 1
                elif p_c == CHOICE_QUIET:
                    if t_c != CHOICE_MEANINGFUL_MOVE:
                        tn += 1
                    else:
                        fn += 1

            eff_acc = correct / N
            sens = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            bal_acc = (sens + spec) / 2.0

            model_effective_accs[m].append(eff_acc)
            model_balanced_accs[m].append(bal_acc)
            accs_this_round[m] = eff_acc
            bal_accs_this_round[m] = bal_acc

        for diff_k in paired_diffs_acc:
            base_m = diff_k.replace(f"{jev_key}_minus_", "")
            diff_acc = accs_this_round[jev_key] - accs_this_round[base_m]
            diff_bal = bal_accs_this_round[jev_key] - bal_accs_this_round[base_m]
            paired_diffs_acc[diff_k].append(diff_acc)
            paired_diffs_bal_acc[diff_k].append(diff_bal)

    def ci95(vals: list[float]) -> dict[str, float]:
        sorted_v = sorted(vals)
        low_idx = int(0.025 * len(sorted_v))
        high_idx = int(0.975 * len(sorted_v))
        mean_v = sum(sorted_v) / len(sorted_v)
        return {
            "mean": round(mean_v, 4),
            "ci_lower": round(sorted_v[low_idx], 4),
            "ci_upper": round(sorted_v[high_idx], 4),
        }

    summary_out: dict[str, Any] = {
        "n_bootstraps": n_bootstraps,
        "n_samples": N,
        "models": {m: ci95(model_effective_accs[m]) for m in models},
        "models_balanced_acc": {m: ci95(model_balanced_accs[m]) for m in models},
        "paired_differences": {},
        "paired_differences_balanced_acc": {},
    }

    for diff_k, diff_vals in paired_diffs_acc.items():
        ci = ci95(diff_vals)
        sup_fraction = sum(1 for d in diff_vals if d > 0) / len(diff_vals)
        summary_out["paired_differences"][diff_k] = {
            **ci,
            "bootstrap_superiority_fraction": round(sup_fraction, 4),
            "p_jev_superior": round(sup_fraction, 4),  # backwards compatibility
            "is_statistically_significant": ci["ci_lower"] > 0 or ci["ci_upper"] < 0,
        }

    for diff_k, diff_vals in paired_diffs_bal_acc.items():
        ci = ci95(diff_vals)
        sup_fraction = sum(1 for d in diff_vals if d > 0) / len(diff_vals)
        summary_out["paired_differences_balanced_acc"][diff_k] = {
            **ci,
            "bootstrap_superiority_fraction": round(sup_fraction, 4),
            "is_statistically_significant": ci["ci_lower"] > 0 or ci["ci_upper"] < 0,
        }

    return summary_out
