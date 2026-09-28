"""Transparent baselines for Phase 9B setup-gating benchmark.

Implements:
1. Majority / Base-Rate Baseline.
2. Deterministic Simple-Rule Baseline (e.g., tight spread or high short-term reference return).
3. Standardized Logistic Regression (pure standard library, trained strictly on 60% train split).
4. Regularized Ridge Logistic Regression (L2 penalty, trained strictly on train split).
5. Seeded Uniform Random Baseline.
6. Always Abstain Boundary Control.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from pm_research.research.jev_lab.contract import ABSTAIN_CHOICE
from pm_research.research.jev_phase9b.dataset import Phase9bSample
from pm_research.research.jev_phase9b.spec import (
    CHOICE_MEANINGFUL_MOVE,
    CHOICE_QUIET,
)


@dataclass
class BaselinePrediction:
    """Standardized prediction output for a baseline model."""

    choice: str  # MEANINGFUL_MOVE, QUIET, or ABSTAIN
    probability_meaningful: float | None = None
    confidence: float | None = None


class BasePhase9bBaseline:
    """Abstract interface for Phase 9B baselines."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def predict(self, sample: Phase9bSample) -> BaselinePrediction:
        raise NotImplementedError


class MajorityBaseline(BasePhase9bBaseline):
    """Majority class predictor derived strictly from training split."""

    def __init__(self, majority_choice: str = CHOICE_MEANINGFUL_MOVE) -> None:
        self._majority_choice = majority_choice

    @property
    def name(self) -> str:
        return "baseline_majority_base_rate"

    def predict(self, sample: Phase9bSample) -> BaselinePrediction:
        return BaselinePrediction(
            choice=self._majority_choice,
            probability_meaningful=1.0 if self._majority_choice == CHOICE_MEANINGFUL_MOVE else 0.0,
            confidence=1.0,
        )


class RandomBaseline(BasePhase9bBaseline):
    """Deterministic seeded random baseline."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return f"baseline_random_seed_{self.seed}"

    def predict(self, sample: Phase9bSample) -> BaselinePrediction:
        choice = self._rng.choice([CHOICE_MEANINGFUL_MOVE, CHOICE_QUIET])
        return BaselinePrediction(
            choice=choice,
            probability_meaningful=0.5,
            confidence=0.5,
        )


class AlwaysAbstainBaseline(BasePhase9bBaseline):
    """Boundary control that always abstains."""

    @property
    def name(self) -> str:
        return "baseline_always_abstain"

    def predict(self, sample: Phase9bSample) -> BaselinePrediction:
        return BaselinePrediction(
            choice=ABSTAIN_CHOICE,
            probability_meaningful=None,
            confidence=None,
        )


class DeterministicRuleBaseline(BasePhase9bBaseline):
    """Simple transparent heuristic: high return or narrow spread triggers MEANINGFUL_MOVE."""

    def __init__(self, max_spread: float = 0.04, min_return_30s_bps: float = 3.0) -> None:
        self.max_spread = max_spread
        self.min_return_30s_bps = min_return_30s_bps

    @property
    def name(self) -> str:
        return "baseline_deterministic_rule"

    def predict(self, sample: Phase9bSample) -> BaselinePrediction:
        spread = sample.poly_spread
        ret_30s = abs(sample.features.get("binance_return_30s_bps") or 0.0)

        # If spread is tight or 30s Binance return is notable, expect meaningful move
        if spread <= self.max_spread or ret_30s >= self.min_return_30s_bps:
            return BaselinePrediction(
                choice=CHOICE_MEANINGFUL_MOVE,
                probability_meaningful=0.75,
                confidence=0.75,
            )
        return BaselinePrediction(
            choice=CHOICE_QUIET,
            probability_meaningful=0.25,
            confidence=0.75,
        )


class StandardizedLogisticRegression(BasePhase9bBaseline):
    """Pure standard-library logistic regression with optional L2 (Ridge) penalty.

    Trained strictly on training split with zero future leakage.
    """

    FEATURE_KEYS: tuple[str, ...] = (
        "poly_midpoint",
        "poly_spread",
        "poly_return_5s",
        "poly_return_10s",
        "poly_return_30s",
        "binance_spread_bps",
        "binance_microprice_offset_bps",
        "binance_top5_depth_imbalance",
        "binance_return_since_open_bps",
        "binance_return_5s_bps",
        "binance_return_10s_bps",
        "binance_return_30s_bps",
        "binance_return_60s_bps",
    )

    def __init__(
        self,
        l2_penalty: float = 0.01,
        learning_rate: float = 0.05,
        max_epochs: int = 400,
        model_name: str = "baseline_logistic_regression",
    ) -> None:
        self.l2_penalty = l2_penalty
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self._name = model_name

        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.weights: list[float] = [0.0] * len(self.FEATURE_KEYS)
        self.bias: float = 0.0
        self.is_fitted: bool = False

    @property
    def name(self) -> str:
        return self._name

    def fit(self, train_samples: list[Phase9bSample]) -> None:
        """Fit scaler and logistic weights strictly on training data."""
        if not train_samples:
            raise ValueError("train_samples cannot be empty")

        # 1. Compute means and standard deviations for imputation & standardization
        for k in self.FEATURE_KEYS:
            vals = [
                float(s.features[k])
                for s in train_samples
                if s.features.get(k) is not None
            ]
            if vals:
                m = sum(vals) / len(vals)
                var = sum((x - m) ** 2 for x in vals) / len(vals)
                std = math.sqrt(var) if var > 1e-9 else 1.0
            else:
                m, std = 0.0, 1.0
            self.means[k] = m
            self.stds[k] = std

        # 2. Extract feature matrix X and targets y (1 for MEANINGFUL_MOVE, 0 for QUIET)
        X: list[list[float]] = []
        y: list[float] = []

        for s in train_samples:
            row: list[float] = []
            for k in self.FEATURE_KEYS:
                val = s.features.get(k)
                raw = float(val) if val is not None else self.means[k]
                norm = (raw - self.means[k]) / self.stds[k]
                row.append(norm)
            X.append(row)
            y.append(1.0 if s.objective_label == CHOICE_MEANINGFUL_MOVE else 0.0)

        # 3. Batch Gradient Descent with L2 penalty
        N = len(X)
        P = len(self.FEATURE_KEYS)
        w = [0.0] * P
        b = 0.0

        for _ in range(self.max_epochs):
            # Forward pass
            dw = [0.0] * P
            db = 0.0

            for i in range(N):
                z = sum(w[j] * X[i][j] for j in range(P)) + b
                # Clamped sigmoid
                p_pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                err = p_pred - y[i]

                for j in range(P):
                    dw[j] += err * X[i][j]
                db += err

            # Gradient update with L2 regularization
            for j in range(P):
                grad_w = (dw[j] / N) + (self.l2_penalty * w[j])
                w[j] -= self.learning_rate * grad_w
            b -= self.learning_rate * (db / N)

        self.weights = w
        self.bias = b
        self.is_fitted = True

    def predict(self, sample: Phase9bSample) -> BaselinePrediction:
        """Compute standardized prediction on an observation."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        z = self.bias
        for j, k in enumerate(self.FEATURE_KEYS):
            val = sample.features.get(k)
            raw = float(val) if val is not None else self.means[k]
            norm = (raw - self.means[k]) / self.stds[k]
            z += self.weights[j] * norm

        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        choice = CHOICE_MEANINGFUL_MOVE if prob >= 0.5 else CHOICE_QUIET
        confidence = prob if choice == CHOICE_MEANINGFUL_MOVE else (1.0 - prob)

        return BaselinePrediction(
            choice=choice,
            probability_meaningful=round(prob, 4),
            confidence=round(confidence, 4),
        )
