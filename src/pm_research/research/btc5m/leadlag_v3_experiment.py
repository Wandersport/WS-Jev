"""Frozen scientific specification for BTC 5-minute Lead-Lag Prospective Replication (v3).

Canonical Experiment ID: btc5m_leadlag_v3_replication_2s
Pilot Experiment ID: btc5m_leadlag_v3_replication_2s_pilot
Target Horizon: Exactly +2 seconds (L = 2s only)
Target Physical Rounds: 250
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pm_research.research.btc5m.leadlag_v2_experiment import (
    FROZEN_BINANCE_FEATURES,
    FROZEN_POLY_FEATURES,
    MAX_STALE_AGE_MS,
)
from pm_research.research.btc5m.leadlag_v2_experiment import (
    LeadLagSampleV2 as LeadLagSampleV3,
)

EXPERIMENT_ID: str = "btc5m_leadlag_v3_replication_2s"
EXPERIMENT_PILOT_ID: str = "btc5m_leadlag_v3_replication_2s_pilot"
EXPERIMENT_VERSION: str = "3.0.0"
EXPERIMENT_SPEC_HASH: str = "bfe5b0553a7143dd8941239dd481cf0e86499b54338dc6717565d1d9dc28fa53"

TARGET_PHYSICAL_ROUNDS: int = 250
PILOT_PHYSICAL_ROUNDS: int = 10
SAMPLING_CADENCE_SEC: int = 1
PREDECLARED_LAGS_SEC: tuple[int, ...] = (2,)  # Strictly +2s only

REPLICATION_DB_PATH: Path = Path("data/pm_research_v3_replication.db")
FROZEN_TRANSPORT_MODEL_PATH: Path = Path("models/frozen_v2_replication_models_2s.json")
FROZEN_TRANSPORT_MODEL_HASH: str = "ee914a59072320d142944530957036c325632f941432d934f472e689dc7f1c77"

__all__ = [
    "EXPERIMENT_ID",
    "EXPERIMENT_PILOT_ID",
    "EXPERIMENT_SPEC_HASH",
    "EXPERIMENT_VERSION",
    "FROZEN_BINANCE_FEATURES",
    "FROZEN_POLY_FEATURES",
    "FROZEN_TRANSPORT_MODEL_HASH",
    "FROZEN_TRANSPORT_MODEL_PATH",
    "LeadLagSampleV3",
    "MAX_STALE_AGE_MS",
    "PILOT_PHYSICAL_ROUNDS",
    "PREDECLARED_LAGS_SEC",
    "REPLICATION_DB_PATH",
    "TARGET_PHYSICAL_ROUNDS",
    "compute_experiment_spec_hash",
]


def compute_experiment_spec_hash(
    spec_path: str | Path = "docs/BTC5M_LEADLAG_REPLICATION_2S_SPEC.md",
) -> str:
    """Compute SHA256 checksum of the canonical v3 replication experiment specification."""
    p = Path(spec_path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment spec not found: {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()
