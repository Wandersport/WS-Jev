"""Importer and converter for historical prediction market data into ReplayDataset format."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from pm_research.replay.dataset import DatasetManifest, ReplayDataset
from pm_research.replay.models import HistoricalResolution, HistoricalSnapshot
from pm_research.utils import now_utc, parse_iso_utc

logger = logging.getLogger(__name__)


class DatasetImporter:
    """Converts external historical prediction market files into standard ReplayDataset format."""

    @classmethod
    def import_from_jsonl(
        cls,
        snapshots_path: str | Path,
        resolutions_path: str | Path | None = None,
        dataset_id: str = "imported_dataset",
        name: str = "Imported Dataset",
        source: str = "external_jsonl",
        output_dir: str | Path | None = None,
        is_synthetic: bool = False,
        notes: str = "",
    ) -> ReplayDataset:
        """Import from JSONL files into an immutable ReplayDataset with manifest and checksum."""
        s_path = Path(snapshots_path)
        if not s_path.exists():
            raise FileNotFoundError(f"Snapshots file not found: {s_path}")

        snapshots: list[HistoricalSnapshot] = []
        market_ids: set[str] = set()

        with open(s_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    snap = HistoricalSnapshot.from_dict(data)
                    snapshots.append(snap)
                    market_ids.add(snap.market_id)
                except Exception as e:
                    logger.warning(f"Skipping malformed snapshot at line {line_no}: {e}")

        if not snapshots:
            raise ValueError(f"No valid snapshots found in {s_path}")

        # Ensure strict chronological ordering
        snapshots.sort(key=lambda s: s.timestamp)

        resolutions: list[HistoricalResolution] = []
        if resolutions_path:
            r_path = Path(resolutions_path)
            if r_path.exists():
                with open(r_path, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, start=1):
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            data = json.loads(line_str)
                            res = HistoricalResolution.from_dict(data)
                            resolutions.append(res)
                        except Exception as e:
                            logger.warning(f"Skipping malformed resolution at line {line_no}: {e}")
                resolutions.sort(key=lambda r: r.resolved_at)

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            name=name,
            source=source,
            created_at=now_utc(),
            start_time=snapshots[0].timestamp,
            end_time=snapshots[-1].timestamp,
            market_count=len(market_ids),
            snapshot_count=len(snapshots),
            resolution_count=len(resolutions),
            schema_version="1.0.0",
            is_synthetic=is_synthetic,
            notes=notes,
        )

        dataset = ReplayDataset(
            manifest=manifest,
            snapshots=snapshots,
            resolutions=resolutions,
        )

        if output_dir:
            dataset.save(output_dir)

        return dataset

    @classmethod
    def import_from_csv(
        cls,
        csv_path: str | Path,
        dataset_id: str,
        name: str,
        source: str = "external_csv",
        output_dir: str | Path | None = None,
        is_synthetic: bool = False,
    ) -> ReplayDataset:
        """Import standard tabular snapshot data from CSV."""
        c_path = Path(csv_path)
        if not c_path.exists():
            raise FileNotFoundError(f"CSV file not found: {c_path}")

        snapshots: list[HistoricalSnapshot] = []
        market_ids: set[str] = set()

        with open(c_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = parse_iso_utc(row["timestamp"])
                    res_time = parse_iso_utc(row["resolution_time"])
                    snap = HistoricalSnapshot(
                        market_id=str(row["market_id"]).strip(),
                        timestamp=ts,
                        status=str(row.get("status", "ACTIVE")).strip(),
                        question=str(row.get("question", "")).strip(),
                        category=str(row.get("category", "GENERAL")).strip().upper(),
                        resolution_time=res_time,
                        yes_bid=float(row["yes_bid"]) if row.get("yes_bid") else None,
                        yes_ask=float(row["yes_ask"]) if row.get("yes_ask") else None,
                        no_bid=float(row["no_bid"]) if row.get("no_bid") else None,
                        no_ask=float(row["no_ask"]) if row.get("no_ask") else None,
                        last_price=float(row["last_price"]) if row.get("last_price") else None,
                        midpoint=float(row["midpoint"]) if row.get("midpoint") else None,
                        spread=float(row["spread"]) if row.get("spread") else None,
                        liquidity=float(row.get("liquidity", 0.0)),
                        volume_24h=float(row.get("volume_24h", 0.0)),
                    )
                    snapshots.append(snap)
                    market_ids.add(snap.market_id)
                except Exception as e:
                    logger.warning(f"Error parsing CSV row: {e}")

        if not snapshots:
            raise ValueError(f"No valid snapshots found in CSV {c_path}")

        snapshots.sort(key=lambda s: s.timestamp)

        manifest = DatasetManifest(
            dataset_id=dataset_id,
            name=name,
            source=source,
            created_at=now_utc(),
            start_time=snapshots[0].timestamp,
            end_time=snapshots[-1].timestamp,
            market_count=len(market_ids),
            snapshot_count=len(snapshots),
            resolution_count=0,
            schema_version="1.0.0",
            is_synthetic=is_synthetic,
        )

        dataset = ReplayDataset(manifest=manifest, snapshots=snapshots, resolutions=[])
        if output_dir:
            dataset.save(output_dir)
        return dataset
