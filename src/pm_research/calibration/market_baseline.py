"""Standardized horizon evaluation and market baseline benchmarking.

This module evaluates whether the deterministic probability model (Ilsa) provides
measurable forecasting improvement over the raw prediction-market price on genuine
historical resolved markets across standardized pre-resolution horizons.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

from pm_research.config import SystemConfig
from pm_research.domain.models import Side
from pm_research.pipeline.holt import HoltResearcher
from pm_research.pipeline.ilsa import IlsaEstimator
from pm_research.pipeline.rigo import RigoIngestor
from pm_research.replay.dataset import ReplayDataset
from pm_research.replay.models import HistoricalSnapshot


@dataclass(frozen=True)
class HorizonSpec:
    """Definition of a standardized pre-resolution evaluation horizon."""

    label: str
    seconds: int
    max_stale_seconds: int

    @property
    def hours(self) -> float:
        return self.seconds / 3600.0


STANDARDIZED_HORIZONS: tuple[HorizonSpec, ...] = (
    HorizonSpec(label="30d", seconds=30 * 86400, max_stale_seconds=7 * 86400),
    HorizonSpec(label="14d", seconds=14 * 86400, max_stale_seconds=3 * 86400),
    HorizonSpec(label="7d", seconds=7 * 86400, max_stale_seconds=2 * 86400),
    HorizonSpec(label="72h", seconds=72 * 3600, max_stale_seconds=24 * 3600),
    HorizonSpec(label="24h", seconds=24 * 3600, max_stale_seconds=12 * 3600),
    HorizonSpec(label="6h", seconds=6 * 3600, max_stale_seconds=4 * 3600),
)


@dataclass
class HorizonObservation:
    """A single matched prediction market snapshot observation at a fixed horizon."""

    market_id: str
    horizon: str
    snapshot_time: datetime
    resolution_time: datetime
    target_time: datetime
    hours_prior: float
    category: str
    market_prob: float
    model_prob: float
    actual_outcome: float
    split: str  # "dev", "val", "holdout"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Statistical summary comparing market baseline against the model."""

    observation_count: int
    market_count: int
    market_brier: float
    model_brier: float
    delta_brier: float
    delta_brier_ci: tuple[float, float]
    market_log_loss: float
    model_log_loss: float
    delta_log_loss: float
    delta_log_loss_ci: tuple[float, float]
    market_bias: float
    model_bias: float
    market_ece: float
    model_ece: float


@dataclass
class MarketBaselineReport:
    """Comprehensive evaluation report comparing model against market benchmark."""

    dataset_id: str
    total_unique_markets: int
    total_observations: int
    overall: MetricSummary
    by_horizon: dict[str, MetricSummary] = field(default_factory=dict)
    by_split: dict[str, MetricSummary] = field(default_factory=dict)
    by_category: dict[str, MetricSummary] = field(default_factory=dict)
    observations: list[HorizonObservation] = field(default_factory=list)


def binary_log_loss(y: float, p: float, eps: float = 1e-6) -> float:
    """Compute binary cross-entropy loss with boundary clipping."""
    p_clamped = max(eps, min(1.0 - eps, p))
    return -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))


def compute_ece(probs: Sequence[float], actuals: Sequence[float], num_bins: int = 10) -> float:
    """Calculate Expected Calibration Error across ten uniform bins."""
    n = len(probs)
    if n == 0:
        return 0.0

    bin_width = 1.0 / num_bins
    bin_preds: list[list[float]] = [[] for _ in range(num_bins)]
    bin_actuals: list[list[float]] = [[] for _ in range(num_bins)]

    for p, y in zip(probs, actuals, strict=True):
        idx = min(int(p / bin_width), num_bins - 1)
        bin_preds[idx].append(p)
        bin_actuals[idx].append(y)

    ece = 0.0
    for i in range(num_bins):
        k = len(bin_preds[i])
        if k > 0:
            mean_pred = sum(bin_preds[i]) / k
            mean_actual = sum(bin_actuals[i]) / k
            ece += (k / n) * abs(mean_pred - mean_actual)

    return ece


def compute_metrics_slice(
    obs_list: Sequence[HorizonObservation],
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> MetricSummary:
    """Compute all evaluation metrics and market-clustered bootstrap intervals."""
    n = len(obs_list)
    if n == 0:
        return MetricSummary(
            observation_count=0,
            market_count=0,
            market_brier=0.0,
            model_brier=0.0,
            delta_brier=0.0,
            delta_brier_ci=(0.0, 0.0),
            market_log_loss=0.0,
            model_log_loss=0.0,
            delta_log_loss=0.0,
            delta_log_loss_ci=(0.0, 0.0),
            market_bias=0.0,
            model_bias=0.0,
            market_ece=0.0,
            model_ece=0.0,
        )

    unique_markets = sorted({o.market_id for o in obs_list})
    m = len(unique_markets)

    mkt_brier_sum = sum((o.market_prob - o.actual_outcome) ** 2 for o in obs_list)
    mod_brier_sum = sum((o.model_prob - o.actual_outcome) ** 2 for o in obs_list)
    mkt_brier = mkt_brier_sum / n
    mod_brier = mod_brier_sum / n
    delta_brier = mod_brier - mkt_brier

    mkt_ll_sum = sum(binary_log_loss(o.actual_outcome, o.market_prob) for o in obs_list)
    mod_ll_sum = sum(binary_log_loss(o.actual_outcome, o.model_prob) for o in obs_list)
    mkt_ll = mkt_ll_sum / n
    mod_ll = mod_ll_sum / n
    delta_ll = mod_ll - mkt_ll

    mkt_bias = sum(o.market_prob - o.actual_outcome for o in obs_list) / n
    mod_bias = sum(o.model_prob - o.actual_outcome for o in obs_list) / n

    mkt_ece = compute_ece([o.market_prob for o in obs_list], [o.actual_outcome for o in obs_list])
    mod_ece = compute_ece([o.model_prob for o in obs_list], [o.actual_outcome for o in obs_list])

    # Market-clustered bootstrap: resample unique markets with replacement
    delta_brier_ci = (delta_brier, delta_brier)
    delta_log_loss_ci = (delta_ll, delta_ll)

    if m >= 2 and bootstrap_samples > 0:
        rng = random.Random(seed)
        obs_by_market: dict[str, list[HorizonObservation]] = {}
        for o in obs_list:
            obs_by_market.setdefault(o.market_id, []).append(o)

        sampled_delta_briers: list[float] = []
        sampled_delta_lls: list[float] = []

        for _ in range(bootstrap_samples):
            sampled_mkts = rng.choices(unique_markets, k=m)
            b_obs: list[HorizonObservation] = []
            for sm in sampled_mkts:
                b_obs.extend(obs_by_market[sm])

            bn = len(b_obs)
            if bn == 0:
                continue

            b_mkt_b = sum((o.market_prob - o.actual_outcome) ** 2 for o in b_obs) / bn
            b_mod_b = sum((o.model_prob - o.actual_outcome) ** 2 for o in b_obs) / bn
            sampled_delta_briers.append(b_mod_b - b_mkt_b)

            b_mkt_l = sum(binary_log_loss(o.actual_outcome, o.market_prob) for o in b_obs) / bn
            b_mod_l = sum(binary_log_loss(o.actual_outcome, o.model_prob) for o in b_obs) / bn
            sampled_delta_lls.append(b_mod_l - b_mkt_l)

        if sampled_delta_briers:
            sampled_delta_briers.sort()
            low_idx = int(0.025 * len(sampled_delta_briers))
            high_idx = int(0.975 * len(sampled_delta_briers))
            delta_brier_ci = (
                round(sampled_delta_briers[low_idx], 5),
                round(sampled_delta_briers[high_idx], 5),
            )

        if sampled_delta_lls:
            sampled_delta_lls.sort()
            low_idx = int(0.025 * len(sampled_delta_lls))
            high_idx = int(0.975 * len(sampled_delta_lls))
            delta_log_loss_ci = (
                round(sampled_delta_lls[low_idx], 5),
                round(sampled_delta_lls[high_idx], 5),
            )

    return MetricSummary(
        observation_count=n,
        market_count=m,
        market_brier=round(mkt_brier, 5),
        model_brier=round(mod_brier, 5),
        delta_brier=round(delta_brier, 5),
        delta_brier_ci=delta_brier_ci,
        market_log_loss=round(mkt_ll, 5),
        model_log_loss=round(mod_ll, 5),
        delta_log_loss=round(delta_ll, 5),
        delta_log_loss_ci=delta_log_loss_ci,
        market_bias=round(mkt_bias, 5),
        model_bias=round(mod_bias, 5),
        market_ece=round(mkt_ece, 5),
        model_ece=round(mod_ece, 5),
    )


class StandardizedHorizonEvaluator:
    """Evaluates prediction models against market baselines without lookahead bias."""

    def __init__(
        self,
        config: SystemConfig | None = None,
        horizons: tuple[HorizonSpec, ...] = STANDARDIZED_HORIZONS,
        bootstrap_samples: int = 1000,
        random_seed: int = 42,
    ) -> None:
        self.config = config or SystemConfig()
        self.horizons = horizons
        self.bootstrap_samples = bootstrap_samples
        self.random_seed = random_seed

        # Pipeline stages initialized with frozen parameters
        self.rigo = RigoIngestor(self.config)
        self.holt = HoltResearcher(self.config)
        self.ilsa = IlsaEstimator(self.config)

    def evaluate_dataset(self, dataset: ReplayDataset) -> MarketBaselineReport:
        """Run standardized horizon evaluation across all resolved markets in the dataset."""
        # 1. Group snapshots by market_id
        snaps_by_market: dict[str, list[HistoricalSnapshot]] = {}
        for snap in dataset.snapshots:
            snaps_by_market.setdefault(snap.market_id, []).append(snap)

        # Sort each market's snapshots chronologically
        for m_id in snaps_by_market:
            snaps_by_market[m_id].sort(key=lambda s: s.timestamp)

        # 2. Chronological split by resolution date
        # Sort resolutions chronologically
        sorted_resolutions = sorted(dataset.resolutions, key=lambda r: r.resolved_at)
        total_res = len(sorted_resolutions)
        dev_cutoff = int(total_res * 0.60)
        val_cutoff = int(total_res * 0.80)

        market_splits: dict[str, str] = {}
        for i, res in enumerate(sorted_resolutions):
            if i < dev_cutoff:
                market_splits[res.market_id] = "dev"
            elif i < val_cutoff:
                market_splits[res.market_id] = "val"
            else:
                market_splits[res.market_id] = "holdout"

        res_by_market = {r.market_id: r for r in dataset.resolutions}

        # 3. Match snapshots at each standardized horizon
        matched_observations: list[HorizonObservation] = []

        for m_id, resolution in res_by_market.items():
            mkt_snaps = snaps_by_market.get(m_id)
            if not mkt_snaps:
                continue

            split_tag = market_splits.get(m_id, "dev")
            y_actual = 1.0 if resolution.resolved_outcome == Side.YES else 0.0

            for h_spec in self.horizons:
                t_res = resolution.resolved_at
                t_target = t_res - timedelta(seconds=h_spec.seconds)

                # Find latest snapshot s where s.timestamp <= t_target
                # Binary search or scan
                chosen_snap: HistoricalSnapshot | None = None
                for s in reversed(mkt_snaps):
                    if s.timestamp <= t_target:
                        chosen_snap = s
                        break

                if chosen_snap is None:
                    # No snapshot recorded before or at target time
                    continue

                # Check staleness tolerance: snapshot must not be older than target - max_stale
                staleness_sec = (t_target - chosen_snap.timestamp).total_seconds()
                if staleness_sec > h_spec.max_stale_seconds:
                    # Observation too stale to represent this horizon
                    continue

                # Strict temporal assertion: snapshot MUST be strictly prior to resolution
                assert chosen_snap.timestamp <= t_target, "Lookahead violation: snap > target"
                assert chosen_snap.timestamp < resolution.resolved_at, "Lookahead violation: snap >= resolution"

                # Ingest through Rigo at snapshot simulated time
                m_snap = self.rigo.ingest_snapshot(
                    chosen_snap.to_dict(),
                    cycle_id=f"eval_{h_spec.label}",
                    current_time=chosen_snap.timestamp,
                )
                if m_snap is None:
                    continue

                # Extract features via Holt at snapshot simulated time
                features = self.holt.extract_features(
                    m_snap,
                    current_time=chosen_snap.timestamp,
                )

                # Estimate probability via frozen Ilsa
                est = self.ilsa.estimate(m_snap, features)

                # Market baseline probability: observed last_price
                p_mkt = (
                    chosen_snap.last_price
                    if chosen_snap.last_price is not None
                    else (chosen_snap.midpoint or 0.50)
                )
                p_model = est.q_hat

                hours_prior = (t_res - chosen_snap.timestamp).total_seconds() / 3600.0

                obs = HorizonObservation(
                    market_id=m_id,
                    horizon=h_spec.label,
                    snapshot_time=chosen_snap.timestamp,
                    resolution_time=t_res,
                    target_time=t_target,
                    hours_prior=round(hours_prior, 2),
                    category=chosen_snap.category,
                    market_prob=round(p_mkt, 4),
                    model_prob=round(p_model, 4),
                    actual_outcome=y_actual,
                    split=split_tag,
                    metadata={
                        "uncertainty": est.uncertainty,
                        "contributions": est.feature_contributions,
                    },
                )
                matched_observations.append(obs)

        # 4. Compute summaries
        overall_summary = compute_metrics_slice(
            matched_observations,
            bootstrap_samples=self.bootstrap_samples,
            seed=self.random_seed,
        )

        by_horizon: dict[str, MetricSummary] = {}
        for h_spec in self.horizons:
            h_obs = [o for o in matched_observations if o.horizon == h_spec.label]
            by_horizon[h_spec.label] = compute_metrics_slice(
                h_obs,
                bootstrap_samples=self.bootstrap_samples,
                seed=self.random_seed,
            )

        by_split: dict[str, MetricSummary] = {}
        for split_name in ("dev", "val", "holdout"):
            s_obs = [o for o in matched_observations if o.split == split_name]
            by_split[split_name] = compute_metrics_slice(
                s_obs,
                bootstrap_samples=self.bootstrap_samples,
                seed=self.random_seed,
            )

        by_cat: dict[str, MetricSummary] = {}
        categories = sorted({o.category for o in matched_observations})
        for cat in categories:
            c_obs = [o for o in matched_observations if o.category == cat]
            by_cat[cat] = compute_metrics_slice(
                c_obs,
                bootstrap_samples=self.bootstrap_samples,
                seed=self.random_seed,
            )

        unique_evaluated_mkts = len({o.market_id for o in matched_observations})

        return MarketBaselineReport(
            dataset_id=dataset.manifest.dataset_id,
            total_unique_markets=unique_evaluated_mkts,
            total_observations=len(matched_observations),
            overall=overall_summary,
            by_horizon=by_horizon,
            by_split=by_split,
            by_category=by_cat,
            observations=matched_observations,
        )
