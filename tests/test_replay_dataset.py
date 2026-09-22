"""Unit and integration tests for historical replay dataset management, manifest verification, and import."""

from __future__ import annotations

from pathlib import Path

from pm_research.data.importer import DatasetImporter
from pm_research.data.synthetic_replay import create_deterministic_synthetic_replay_dataset
from pm_research.replay.dataset import DatasetManager, ReplayDataset


def test_synthetic_replay_dataset_creation_and_checksum(tmp_path: Path) -> None:
    """Verifies that synthetic replay datasets are generated with valid manifests and SHA256 checksums."""
    target_dir = tmp_path / "test_synth_ds"
    ds = create_deterministic_synthetic_replay_dataset(target_dir=target_dir)

    assert ds.manifest.dataset_id == "synthetic_benchmark_v1"
    assert ds.manifest.checksum_sha256 != ""
    assert len(ds.manifest.checksum_sha256) == 64  # Hex SHA256 length
    assert len(ds.get_snapshots()) == 13
    assert len(ds.get_resolutions()) == 2

    # Verify reload from disk
    reloaded = ReplayDataset.load(target_dir)
    assert reloaded.manifest.dataset_id == ds.manifest.dataset_id
    assert len(reloaded.get_snapshots()) == 13
    assert len(reloaded.get_resolutions()) == 2


def test_dataset_manager_discovery(tmp_path: Path) -> None:
    """DatasetManager locates registered datasets across directory trees."""
    ds_dir = tmp_path / "my_custom_ds"
    create_deterministic_synthetic_replay_dataset(target_dir=ds_dir)

    manager = DatasetManager(root_dir=tmp_path)
    datasets = manager.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].dataset_id == "synthetic_benchmark_v1"

    loaded = manager.get_dataset("synthetic_benchmark_v1")
    assert len(loaded.get_snapshots()) == 13


def test_dataset_stream_cycles() -> None:
    """stream_cycles yields snapshots strictly grouped by exact timestamp in chronological order."""
    ds = create_deterministic_synthetic_replay_dataset()
    cycles = list(ds.stream_cycles())

    assert len(cycles) == 5  # t0, t1, t2, t3, t4

    prev_time = None
    for ts, snaps in cycles:
        if prev_time is not None:
            assert ts > prev_time  # Strictly monotonic
        prev_time = ts
        assert len(snaps) > 0
        for s in snaps:
            assert s.timestamp == ts


def test_dataset_importer_csv(tmp_path: Path) -> None:
    """DatasetImporter parses tabular CSV files into valid ReplayDataset."""
    csv_file = tmp_path / "markets.csv"
    csv_content = """market_id,timestamp,status,question,category,resolution_time,yes_bid,yes_ask,no_bid,no_ask,last_price,midpoint,spread,liquidity,volume_24h
mkt_1,2026-01-01T12:00:00Z,ACTIVE,Test Question 1,TECH,2026-01-10T12:00:00Z,0.40,0.42,0.58,0.60,0.41,0.41,0.02,1000.0,5000.0
mkt_2,2026-01-01T12:00:00Z,ACTIVE,Test Question 2,MACRO,2026-01-10T12:00:00Z,0.60,0.64,0.36,0.40,0.62,0.62,0.04,2000.0,8000.0
"""
    csv_file.write_text(csv_content, encoding="utf-8")

    out_dir = tmp_path / "imported_ds"
    ds = DatasetImporter.import_from_csv(
        csv_path=csv_file,
        dataset_id="test_imported_csv",
        name="Test CSV Import",
        output_dir=out_dir,
    )

    assert ds.manifest.dataset_id == "test_imported_csv"
    assert ds.manifest.snapshot_count == 2
    assert ds.manifest.market_count == 2
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "snapshots.jsonl").exists()
