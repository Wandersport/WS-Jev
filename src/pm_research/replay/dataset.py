"""Historical replay dataset management, JSONL streaming, and manifest integrity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generator

from pm_research.replay.models import HistoricalResolution, HistoricalSnapshot
from pm_research.utils import ensure_utc, parse_iso_utc, to_iso_utc


@dataclass
class DatasetManifest:
    """Metadata specification and cryptographic checksum for a historical replay dataset."""

    dataset_id: str
    name: str
    source: str  # e.g., "synthetic_benchmark", "polymarket_public_archive"
    created_at: datetime
    start_time: datetime
    end_time: datetime
    market_count: int
    snapshot_count: int
    resolution_count: int
    schema_version: str = "1.0.0"
    checksum_sha256: str = ""
    is_synthetic: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        self.created_at = ensure_utc(self.created_at)
        self.start_time = ensure_utc(self.start_time)
        self.end_time = ensure_utc(self.end_time)

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "created_at": to_iso_utc(self.created_at),
            "start_time": to_iso_utc(self.start_time),
            "end_time": to_iso_utc(self.end_time),
            "market_count": self.market_count,
            "snapshot_count": self.snapshot_count,
            "resolution_count": self.resolution_count,
            "schema_version": self.schema_version,
            "checksum_sha256": self.checksum_sha256,
            "is_synthetic": self.is_synthetic,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | int | bool]) -> DatasetManifest:
        return cls(
            dataset_id=str(data["dataset_id"]),
            name=str(data["name"]),
            source=str(data["source"]),
            created_at=parse_iso_utc(str(data["created_at"])),
            start_time=parse_iso_utc(str(data["start_time"])),
            end_time=parse_iso_utc(str(data["end_time"])),
            market_count=int(data["market_count"]),
            snapshot_count=int(data["snapshot_count"]),
            resolution_count=int(data["resolution_count"]),
            schema_version=str(data.get("schema_version", "1.0.0")),
            checksum_sha256=str(data.get("checksum_sha256", "")),
            is_synthetic=bool(data.get("is_synthetic", True)),
            notes=str(data.get("notes", "")),
        )


class ReplayDataset:
    """Manages an immutable historical replay dataset stored in JSONL format."""

    def __init__(
        self,
        manifest: DatasetManifest,
        snapshots: list[HistoricalSnapshot] | None = None,
        resolutions: list[HistoricalResolution] | None = None,
        directory: Path | None = None,
    ) -> None:
        self.manifest = manifest
        self._snapshots: list[HistoricalSnapshot] | None = (
            sorted(snapshots, key=lambda s: s.timestamp) if snapshots else None
        )
        self._resolutions: list[HistoricalResolution] | None = (
            sorted(resolutions, key=lambda r: r.resolved_at) if resolutions else None
        )
        self.directory = directory

    @classmethod
    def load(cls, dataset_path: str | Path) -> ReplayDataset:
        """Load a replay dataset from a directory containing manifest.json, snapshots.jsonl, and resolutions.jsonl."""
        dir_path = Path(dataset_path)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dir_path}")

        manifest_file = dir_path / "manifest.json"
        snapshots_file = dir_path / "snapshots.jsonl"
        resolutions_file = dir_path / "resolutions.jsonl"

        if not manifest_file.exists() or not snapshots_file.exists():
            raise FileNotFoundError(f"Missing required dataset files in {dir_path}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = DatasetManifest.from_dict(json.load(f))

        snapshots: list[HistoricalSnapshot] = []
        with open(snapshots_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    snapshots.append(HistoricalSnapshot.from_dict(json.loads(line_str)))

        # Ensure snapshots are chronologically sorted
        snapshots.sort(key=lambda s: s.timestamp)

        resolutions: list[HistoricalResolution] = []
        if resolutions_file.exists():
            with open(resolutions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        resolutions.append(HistoricalResolution.from_dict(json.loads(line_str)))
        resolutions.sort(key=lambda r: r.resolved_at)

        return cls(manifest=manifest, snapshots=snapshots, resolutions=resolutions, directory=dir_path)

    def save(self, target_dir: str | Path) -> Path:
        """Save dataset to directory as manifest.json, snapshots.jsonl, and resolutions.jsonl with SHA256 checksum."""
        out_path = Path(target_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        snapshots = self.get_snapshots()
        resolutions = self.get_resolutions()

        snapshots_file = out_path / "snapshots.jsonl"
        hasher = hashlib.sha256()

        with open(snapshots_file, "w", encoding="utf-8") as f:
            for snap in sorted(snapshots, key=lambda s: s.timestamp):
                line = json.dumps(snap.to_dict()) + "\n"
                f.write(line)
                hasher.update(line.encode("utf-8"))

        resolutions_file = out_path / "resolutions.jsonl"
        with open(resolutions_file, "w", encoding="utf-8") as f:
            for res in sorted(resolutions, key=lambda r: r.resolved_at):
                line = json.dumps(res.to_dict()) + "\n"
                f.write(line)
                hasher.update(line.encode("utf-8"))

        # Update manifest checksum
        self.manifest.checksum_sha256 = hasher.hexdigest()
        manifest_file = out_path / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)

        self.directory = out_path
        return out_path

    def get_snapshots(self) -> list[HistoricalSnapshot]:
        """Return all historical snapshots sorted chronologically."""
        if self._snapshots is None and self.directory:
            loaded = ReplayDataset.load(self.directory)
            self._snapshots = loaded._snapshots
        return self._snapshots or []

    def get_resolutions(self) -> list[HistoricalResolution]:
        """Return all resolutions sorted chronologically."""
        if self._resolutions is None and self.directory:
            loaded = ReplayDataset.load(self.directory)
            self._resolutions = loaded._resolutions
        return self._resolutions or []

    def stream_cycles(self) -> Generator[tuple[datetime, list[HistoricalSnapshot]], None, None]:
        """Group chronological snapshots by exact timestamp batches (cycles)."""
        snapshots = self.get_snapshots()
        if not snapshots:
            return

        current_time = snapshots[0].timestamp
        batch: list[HistoricalSnapshot] = []

        for s in snapshots:
            if s.timestamp != current_time:
                yield current_time, batch
                current_time = s.timestamp
                batch = [s]
            else:
                batch.append(s)

        if batch:
            yield current_time, batch


class DatasetManager:
    """Discovers, registers, and inspects historical replay datasets."""

    DEFAULT_DATASETS_DIR = Path("data/datasets")

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = root_dir or self.DEFAULT_DATASETS_DIR

    def list_datasets(self) -> list[DatasetManifest]:
        """List all available datasets with valid manifests."""
        manifests: list[DatasetManifest] = []
        if not self.root_dir.exists():
            return manifests

        for d in sorted(self.root_dir.iterdir()):
            if d.is_dir():
                mf = d / "manifest.json"
                if mf.exists():
                    try:
                        with open(mf, "r", encoding="utf-8") as f:
                            manifests.append(DatasetManifest.from_dict(json.load(f)))
                    except Exception:
                        continue
        return manifests

    def get_dataset(self, dataset_id_or_path: str) -> ReplayDataset:
        """Find and load a dataset by ID or direct directory path."""
        p = Path(dataset_id_or_path)
        if p.exists() and p.is_dir():
            return ReplayDataset.load(p)

        # Search by dataset_id in root_dir
        candidate = self.root_dir / dataset_id_or_path
        if candidate.exists() and candidate.is_dir():
            return ReplayDataset.load(candidate)

        for d in self.root_dir.glob("*"):
            if d.is_dir() and (d / "manifest.json").exists():
                try:
                    with open(d / "manifest.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("dataset_id") == dataset_id_or_path:
                            return ReplayDataset.load(d)
                except Exception:
                    continue

        raise FileNotFoundError(f"Could not locate dataset '{dataset_id_or_path}' in {self.root_dir}")
