"""Unit and statistical tests for standardized horizon evaluation and market baseline comparison."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pm_research.calibration.market_baseline import (
    HorizonObservation,
    HorizonSpec,
    StandardizedHorizonEvaluator,
    binary_log_loss,
    compute_ece,
    compute_metrics_slice,
)
from pm_research.domain.models import Side
from pm_research.replay.dataset import DatasetManifest, ReplayDataset
from pm_research.replay.models import HistoricalResolution, HistoricalSnapshot


def test_binary_log_loss_and_ece():
    """Verify log loss clamping and ECE calculation correctness."""
    # Perfect predictions: y=1, p=1.0 -> near 0
    ll_perf = binary_log_loss(1.0, 1.0)
    assert ll_perf < 1e-4

    # Terrible prediction: y=1, p=0.0 -> clamped
    ll_bad = binary_log_loss(1.0, 0.0)
    assert ll_bad > 10.0

    # ECE with perfectly calibrated 50/50
    probs = [0.1, 0.9]
    actuals = [0.0, 1.0]
    ece = compute_ece(probs, actuals, num_bins=10)
    # bin 1: pred 0.1, act 0.0 -> err 0.1
    # bin 9: pred 0.9, act 1.0 -> err 0.1
    assert abs(ece - 0.1) < 1e-4


def test_horizon_selection_strictly_no_lookahead():
    """Verify that horizon snapshot matching never selects snapshots occurring after target time."""
    t_res = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    market_id = "test_mkt_lookahead"

    # Target for 24h horizon is t_res - 24h = June 14, 12:00:00
    # Provide snapshots at:
    # 1. June 14, 10:00:00 (target - 2h)
    # 2. June 14, 11:59:00 (target - 1m) -> SHOULD BE CHOSEN
    # 3. June 14, 12:01:00 (target + 1m) -> FUTURE RELATIVE TO HORIZON TARGET!
    # 4. June 15, 11:00:00 (target + 23h) -> WAY IN FUTURE!
    snaps = [
        HistoricalSnapshot(
            market_id=market_id,
            timestamp=datetime(2024, 6, 14, 10, 0, 0, tzinfo=timezone.utc),
            status="ACTIVE",
            question="Will it rain?",
            category="WEATHER",
            resolution_time=t_res,
            last_price=0.30,
            midpoint=0.30,
        ),
        HistoricalSnapshot(
            market_id=market_id,
            timestamp=datetime(2024, 6, 14, 11, 59, 0, tzinfo=timezone.utc),
            status="ACTIVE",
            question="Will it rain?",
            category="WEATHER",
            resolution_time=t_res,
            last_price=0.35,
            midpoint=0.35,
        ),
        HistoricalSnapshot(
            market_id=market_id,
            timestamp=datetime(2024, 6, 14, 12, 1, 0, tzinfo=timezone.utc),
            status="ACTIVE",
            question="Will it rain?",
            category="WEATHER",
            resolution_time=t_res,
            last_price=0.80,  # Huge jump after horizon cutoff
            midpoint=0.80,
        ),
        HistoricalSnapshot(
            market_id=market_id,
            timestamp=datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc),
            status="ACTIVE",
            question="Will it rain?",
            category="WEATHER",
            resolution_time=t_res,
            last_price=0.99,
            midpoint=0.99,
        ),
    ]

    resolution = HistoricalResolution(
        market_id=market_id,
        resolved_outcome=Side.YES,
        resolved_at=t_res,
    )

    manifest = DatasetManifest(
        dataset_id="test_ds",
        name="Test",
        source="test",
        created_at=datetime.now(timezone.utc),
        start_time=snaps[0].timestamp,
        end_time=snaps[-1].timestamp,
        market_count=1,
        snapshot_count=len(snaps),
        resolution_count=1,
        is_synthetic=False,
    )

    dataset = ReplayDataset(
        manifest=manifest,
        snapshots=snaps,
        resolutions=[resolution],
    )

    evaluator = StandardizedHorizonEvaluator(
        horizons=(HorizonSpec(label="24h", seconds=24 * 3600, max_stale_seconds=12 * 3600),),
        bootstrap_samples=0,
    )
    report = evaluator.evaluate_dataset(dataset)

    assert len(report.observations) == 1
    obs = report.observations[0]
    assert obs.market_id == market_id
    assert obs.horizon == "24h"
    # Must have chosen snapshot 2 at 11:59:00 with price 0.35, NEVER 0.80 or 0.99!
    assert obs.snapshot_time == datetime(2024, 6, 14, 11, 59, 0, tzinfo=timezone.utc)
    assert obs.market_prob == 0.35
    assert obs.actual_outcome == 1.0


def test_stale_tolerance_rejection():
    """Verify that snapshots older than max_stale_seconds are rejected for that horizon."""
    t_res = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    market_id = "test_mkt_stale"

    # Only one snapshot at June 10 (5 days before resolution)
    snap = HistoricalSnapshot(
        market_id=market_id,
        timestamp=datetime(2024, 6, 10, 12, 0, 0, tzinfo=timezone.utc),
        status="ACTIVE",
        question="Question?",
        category="PUBLIC",
        resolution_time=t_res,
        last_price=0.50,
        midpoint=0.50,
    )
    resolution = HistoricalResolution(
        market_id=market_id,
        resolved_outcome=Side.NO,
        resolved_at=t_res,
    )
    manifest = DatasetManifest(
        dataset_id="test_stale",
        name="Test",
        source="test",
        created_at=datetime.now(timezone.utc),
        start_time=snap.timestamp,
        end_time=snap.timestamp,
        market_count=1,
        snapshot_count=1,
        resolution_count=1,
        is_synthetic=False,
    )
    dataset = ReplayDataset(manifest=manifest, snapshots=[snap], resolutions=[resolution])

    # Horizons: 7d (max stale 2d -> target June 8: snap is after target so candidate count = 0)
    # 6h (target June 15 06:00: snap is June 10, staleness = 4.75 days > 4 hours max stale -> rejected!)
    evaluator = StandardizedHorizonEvaluator(
        horizons=(HorizonSpec(label="6h", seconds=6 * 3600, max_stale_seconds=4 * 3600),),
        bootstrap_samples=0,
    )
    report = evaluator.evaluate_dataset(dataset)
    assert len(report.observations) == 0, "Stale observation should have been excluded."


def test_clustered_bootstrap_reproducibility():
    """Verify that clustered bootstrap CI calculation is reproducible with fixed random seed."""
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    obs = [
        HorizonObservation(
            market_id=f"mkt_{i % 5}",
            horizon="24h",
            snapshot_time=t0,
            resolution_time=t0 + timedelta(days=1),
            target_time=t0,
            hours_prior=24.0,
            category="PUBLIC",
            market_prob=0.30 + (i * 0.05),
            model_prob=0.35 + (i * 0.04),
            actual_outcome=1.0 if i % 2 == 0 else 0.0,
            split="dev",
        )
        for i in range(10)
    ]

    res1 = compute_metrics_slice(obs, bootstrap_samples=100, seed=42)
    res2 = compute_metrics_slice(obs, bootstrap_samples=100, seed=42)

    assert res1.delta_brier == res2.delta_brier
    assert res1.delta_brier_ci == res2.delta_brier_ci
    assert res1.delta_log_loss_ci == res2.delta_log_loss_ci


def test_chronological_holdout_splits():
    """Verify that chronological splits assign earlier markets to dev and latest to holdout."""
    base_t = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    resolutions: list[HistoricalResolution] = []
    snapshots: list[HistoricalSnapshot] = []

    for i in range(10):
        m_id = f"mkt_{i:02d}"
        res_t = base_t + timedelta(days=i * 10)
        resolutions.append(
            HistoricalResolution(market_id=m_id, resolved_outcome=Side.YES, resolved_at=res_t)
        )
        snapshots.append(
            HistoricalSnapshot(
                market_id=m_id,
                timestamp=res_t - timedelta(days=1),
                status="ACTIVE",
                question=f"Question {i}?",
                category="PUBLIC",
                resolution_time=res_t,
                last_price=0.50,
                midpoint=0.50,
            )
        )

    manifest = DatasetManifest(
        dataset_id="test_split_ds",
        name="Test",
        source="test",
        created_at=datetime.now(timezone.utc),
        start_time=snapshots[0].timestamp,
        end_time=snapshots[-1].timestamp,
        market_count=10,
        snapshot_count=10,
        resolution_count=10,
        is_synthetic=False,
    )
    dataset = ReplayDataset(manifest=manifest, snapshots=snapshots, resolutions=resolutions)

    evaluator = StandardizedHorizonEvaluator(
        horizons=(HorizonSpec(label="24h", seconds=24 * 3600, max_stale_seconds=12 * 3600),),
        bootstrap_samples=0,
    )
    report = evaluator.evaluate_dataset(dataset)

    # 10 markets: 6 dev (60%), 2 val (20%), 2 holdout (20%)
    dev_obs = [o for o in report.observations if o.split == "dev"]
    val_obs = [o for o in report.observations if o.split == "val"]
    holdout_obs = [o for o in report.observations if o.split == "holdout"]

    assert len(dev_obs) == 6
    assert len(val_obs) == 2
    assert len(holdout_obs) == 2

    # Verify chronological ordering
    max_dev_res = max(o.resolution_time for o in dev_obs)
    min_val_res = min(o.resolution_time for o in val_obs)
    max_val_res = max(o.resolution_time for o in val_obs)
    min_holdout_res = min(o.resolution_time for o in holdout_obs)

    assert max_dev_res <= min_val_res
    assert max_val_res <= min_holdout_res
