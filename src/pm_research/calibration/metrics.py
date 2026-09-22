"""Calibration analytics: Brier score, Log Loss, ECE, calibration curves, and sliced metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from pm_research.domain.models import CalibrationObservation


@dataclass
class CalibrationBucket:
    bin_lower: float
    bin_upper: float
    count: int
    mean_predicted: float
    empirical_frequency: float
    bin_error: float


@dataclass
class CalibrationReport:
    sample_size: int
    brier_score: float
    log_loss: float
    forecast_bias: float
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float   # MCE
    buckets: list[CalibrationBucket] = field(default_factory=list)
    by_model_version: dict[str, dict[str, float]] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_edge_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    by_risk_state: dict[str, dict[str, float]] = field(default_factory=dict)


class CalibrationEngine:
    """Computes comprehensive statistical calibration metrics for probability forecasts."""

    def __init__(self, num_bins: int = 10, eps: float = 1e-6) -> None:
        self.num_bins = num_bins
        self.eps = eps

    def compute_metrics(
        self, observations: Sequence[CalibrationObservation]
    ) -> CalibrationReport:
        """Compute Brier score, log loss, ECE, bias, and sliced analytics across observations."""
        n = len(observations)
        if n == 0:
            return CalibrationReport(
                sample_size=0,
                brier_score=0.0,
                log_loss=0.0,
                forecast_bias=0.0,
                expected_calibration_error=0.0,
                maximum_calibration_error=0.0,
            )

        # 1. Overall Brier score and Log Loss
        brier_sum = 0.0
        logloss_sum = 0.0
        pred_sum = 0.0
        actual_sum = 0.0

        for obs in observations:
            q = obs.predicted_probability
            y = obs.actual_outcome
            brier_sum += (q - y) ** 2

            q_c = max(self.eps, min(1.0 - self.eps, q))
            ll = -(y * math.log(q_c) + (1.0 - y) * math.log(1.0 - q_c))
            logloss_sum += ll

            pred_sum += q
            actual_sum += y

        mean_brier = brier_sum / n
        mean_logloss = logloss_sum / n
        forecast_bias = (pred_sum - actual_sum) / n

        # 2. Calibration buckets and ECE
        bin_width = 1.0 / self.num_bins
        bucket_preds: list[list[float]] = [[] for _ in range(self.num_bins)]
        bucket_actuals: list[list[float]] = [[] for _ in range(self.num_bins)]

        for obs in observations:
            q = obs.predicted_probability
            y = obs.actual_outcome
            bin_idx = min(int(q / bin_width), self.num_bins - 1)
            bucket_preds[bin_idx].append(q)
            bucket_actuals[bin_idx].append(y)

        buckets: list[CalibrationBucket] = []
        ece = 0.0
        mce = 0.0

        for i in range(self.num_bins):
            lower = round(i * bin_width, 2)
            upper = round((i + 1) * bin_width, 2)
            k = len(bucket_preds[i])

            if k > 0:
                mean_p = sum(bucket_preds[i]) / k
                emp_f = sum(bucket_actuals[i]) / k
                err = abs(mean_p - emp_f)
                ece += (k / n) * err
                if err > mce:
                    mce = err
            else:
                mean_p = 0.0
                emp_f = 0.0
                err = 0.0

            buckets.append(
                CalibrationBucket(
                    bin_lower=lower,
                    bin_upper=upper,
                    count=k,
                    mean_predicted=round(mean_p, 4),
                    empirical_frequency=round(emp_f, 4),
                    bin_error=round(err, 4),
                )
            )

        # 3. Sliced metrics helper
        def slice_metrics(getter_fn) -> dict[str, dict[str, float]]:
            groups: dict[str, list[CalibrationObservation]] = {}
            for obs in observations:
                key = getter_fn(obs)
                groups.setdefault(key, []).append(obs)

            result: dict[str, dict[str, float]] = {}
            for key, group in groups.items():
                gk = len(group)
                gbrier = sum((o.predicted_probability - o.actual_outcome) ** 2 for o in group) / gk
                gpred = sum(o.predicted_probability for o in group) / gk
                gact = sum(o.actual_outcome for o in group) / gk
                result[key] = {
                    "count": gk,
                    "brier_score": round(gbrier, 4),
                    "mean_predicted": round(gpred, 4),
                    "empirical_rate": round(gact, 4),
                    "bias": round(gpred - gact, 4),
                }
            return result

        by_model = slice_metrics(lambda o: o.model_version)
        by_category = slice_metrics(lambda o: o.category)
        by_edge = slice_metrics(lambda o: o.edge_bucket)
        by_risk = slice_metrics(lambda o: o.risk_state)

        return CalibrationReport(
            sample_size=n,
            brier_score=round(mean_brier, 4),
            log_loss=round(mean_logloss, 4),
            forecast_bias=round(forecast_bias, 4),
            expected_calibration_error=round(ece, 4),
            maximum_calibration_error=round(mce, 4),
            buckets=buckets,
            by_model_version=by_model,
            by_category=by_category,
            by_edge_bucket=by_edge,
            by_risk_state=by_risk,
        )


def binary_log_loss(y: float, p: float, eps: float = 1e-6) -> float:
    """Compute binary cross-entropy loss with boundary clipping."""
    p_clamped = max(eps, min(1.0 - eps, p))
    return -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))

