"""Utility functions for UTC timestamps, formatting, and hashing."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any


def now_utc() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware and normalized to UTC."""
    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime object, got {type(dt)}")
    if dt.tzinfo is None:
        raise ValueError(f"Naive datetime provided ({dt}); all timestamps must be timezone-aware UTC.")
    return dt.astimezone(timezone.utc)


def to_iso_utc(dt: datetime) -> str:
    """Serialize datetime to unambiguous ISO-8601 UTC string ending in 'Z'."""
    utc_dt = ensure_utc(dt)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_iso_utc(s: str) -> datetime:
    """Parse ISO-8601 string into timezone-aware UTC datetime.

    Accepts:
    - '2026-09-22T14:30:00Z'
    - '2026-09-22T16:30:00+02:00'
    - '2026-09-22T10:30:00-04:00'
    - '2026-09-22T14:30:00.123456+00:00'

    Raises ValueError on malformed or naive strings.
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected string timestamp, got {type(s)}")
    clean_s = s.strip()
    if not clean_s:
        raise ValueError("Empty timestamp string cannot be parsed.")

    # Convert 'Z' suffix to +00:00 for fromisoformat compatibility
    if clean_s.endswith("Z") or clean_s.endswith("z"):
        clean_s = clean_s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(clean_s)
    except Exception as e:
        raise ValueError(f"Malformed ISO-8601 timestamp '{s}': {e}") from e

    if dt.tzinfo is None:
        raise ValueError(f"Naive timestamp parsed from '{s}'. Explicit timezone offset is required.")

    return dt.astimezone(timezone.utc)


def generate_id(prefix: str) -> str:
    """Generate a unique prefixed identifier."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of configuration."""
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
