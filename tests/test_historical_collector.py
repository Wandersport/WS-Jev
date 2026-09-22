"""Unit and integration tests for public historical data collector."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pm_research.data.historical_collector import HistoricalCollector, QualityReport
from pm_research.domain.models import Side


def test_collector_exclusion_rules(tmp_path: Path):
    """Verify that market filtering strictly excludes non-binary, unresolved, or missing token markets."""
    collector = HistoricalCollector(cache_dir=tmp_path / "cache")
    report = QualityReport()

    # 1. Non-binary outcomes
    non_binary = {
        "id": "1",
        "outcomes": json.dumps(["A", "B", "C"]),
        "outcomePrices": json.dumps(["1", "0", "0"]),
        "clobTokenIds": json.dumps(["tok1", "tok2", "tok3"]),
        "closedTime": "2024-06-01T12:00:00Z",
    }
    res = collector.filter_and_collect_market(non_binary, report)
    assert res is None
    assert report.exclusion_reasons.get("EXCLUDE_NON_BINARY") == 1

    # 2. Unresolved / ambiguous outcome prices
    unresolved = {
        "id": "2",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.5", "0.5"]),
        "clobTokenIds": json.dumps(["tok1", "tok2"]),
        "closedTime": "2024-06-01T12:00:00Z",
    }
    res = collector.filter_and_collect_market(unresolved, report)
    assert res is None
    assert report.exclusion_reasons.get("EXCLUDE_NO_RESOLUTION") == 1

    # 3. Missing clob token IDs
    no_token = {
        "id": "3",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["1", "0"]),
        "clobTokenIds": json.dumps([]),
        "closedTime": "2024-06-01T12:00:00Z",
    }
    res = collector.filter_and_collect_market(no_token, report)
    assert res is None
    assert report.exclusion_reasons.get("EXCLUDE_NO_TOKEN_ID") == 1

    # 4. Insufficient history
    mock_history = []
    with patch.object(collector, "fetch_token_prices_history", return_value=mock_history):
        sparse_mkt = {
            "id": "4",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["1", "0"]),
            "clobTokenIds": json.dumps(["tok1", "tok2"]),
            "closedTime": "2024-06-01T12:00:00Z",
        }
        res = collector.filter_and_collect_market(sparse_mkt, report, min_history_points=5)
        assert res is None
        assert report.exclusion_reasons.get("EXCLUDE_INSUFFICIENT_HISTORY") == 1


def test_collector_normalization_no_fabricated_quotes(tmp_path: Path):
    """Verify that normalized historical snapshots contain observed prices and zero fabricated quotes."""
    collector = HistoricalCollector(cache_dir=tmp_path / "cache")
    report = QualityReport()

    mock_pts = [
        {"t": 1717200000, "p": 0.40},
        {"t": 1717286400, "p": 0.55},
        {"t": 1717372800, "p": 0.85},
        {"t": 1717459200, "p": 0.95},
        {"t": 1717545600, "p": 1.00},
    ]

    valid_mkt = {
        "id": "101",
        "slug": "sample-market",
        "question": "Will candidate win?",
        "category": "POLITICS",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["1", "0"]),
        "clobTokenIds": json.dumps(["tok_yes", "tok_no"]),
        "closedTime": "2024-06-05T00:00:00Z",
        "liquidity": "50000",
        "volume24hr": "12000",
    }

    with patch.object(collector, "fetch_token_prices_history", return_value=mock_pts):
        result = collector.filter_and_collect_market(valid_mkt, report, min_history_points=3)

    assert result is not None
    snapshots, resolution = result

    # Check resolution
    assert resolution.market_id == "poly_101"
    assert resolution.resolved_outcome == Side.YES
    assert resolution.resolved_at == datetime(2024, 6, 5, 0, 0, 0, tzinfo=timezone.utc)

    # Check snapshots
    assert len(snapshots) == 5
    for s in snapshots:
        assert s.market_id == "poly_101"
        assert s.category == "POLITICS"
        # Price-only history guarantees:
        assert s.yes_bid is None
        assert s.yes_ask is None
        assert s.no_bid is None
        assert s.no_ask is None
        assert s.spread is None
        assert s.order_book is None
        assert s.last_price is not None
        assert s.metadata["quote_source"] == "unavailable"
        assert s.metadata["price_source"] == "clob_prices_history"

    # Check strictly historical momentum:
    # Point 0: 0.40 -> no 24h prior -> mom = 0.0
    assert snapshots[0].metadata["momentum_24h"] == 0.0
    # Point 1 (t0 + 86400): 0.55 -> prior is 0.40 -> mom = 0.15
    assert snapshots[1].metadata["momentum_24h"] == 0.15


def test_collector_caching_and_dataset_packaging(tmp_path: Path):
    """Verify that collector caches raw responses and packages a valid ReplayDataset with checksum."""
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "dataset"
    collector = HistoricalCollector(cache_dir=cache_dir)

    raw_candidates = [
        {
            "id": "201",
            "question": "Market One?",
            "category": "CRYPTO",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["1", "0"]),
            "clobTokenIds": json.dumps(["token_201_yes"]),
            "closedTime": "2024-06-10T12:00:00Z",
            "liquidity": "10000",
            "volume24hr": "5000",
        },
        {
            "id": "202",
            "question": "Market Two?",
            "category": "TECH",
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0", "1"]),
            "clobTokenIds": json.dumps(["token_202_yes"]),
            "closedTime": "2024-06-12T12:00:00Z",
            "liquidity": "20000",
            "volume24hr": "8000",
        },
    ]

    mock_pts = [
        {"t": 1718000000 + i * 3600, "p": 0.20 + (i * 0.05)}
        for i in range(10)
    ]

    with patch.object(collector, "discover_resolved_markets", return_value=raw_candidates):
        with patch.object(collector, "_http_get_json", return_value={"history": mock_pts}):
            dataset, report = collector.build_dataset(
                dataset_id="test_pack_v1",
                name="Test Package",
                target_dir=out_dir,
                max_markets=2,
            )

    assert report.markets_included == 2
    assert report.markets_excluded == 0
    assert len(dataset.resolutions) == 2
    assert len(dataset.snapshots) == 20
    assert dataset.manifest.is_synthetic is False
    assert dataset.manifest.checksum_sha256 != ""

    # Verify cache files were written
    assert (cache_dir / "clob_token_201_yes.json").exists()
    assert (cache_dir / "clob_token_202_yes.json").exists()

    # Verify dataset files exist
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "snapshots.jsonl").exists()
    assert (out_dir / "resolutions.jsonl").exists()
    assert (out_dir / "quality_report.json").exists()
