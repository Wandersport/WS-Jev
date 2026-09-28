"""Strategy candidate contract for Phase 9A governance.

RESEARCH GOVERNANCE ONLY:
Prevents unconstrained hypothesis fishing or iterative over-tuning by Opus.
Enforces that every strategy hypothesis is pre-declared, frozen with a
deterministic SHA-256 fingerprint, bound to a parameter search budget,
and associated with pre-committed rejection conditions before any backtesting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pm_research.utils import ensure_utc, to_iso_utc


def compute_strategy_hash(
    strategy_id: str,
    strategy_version: str,
    source_reference: str,
    hypothesis: str,
    required_features: tuple[str, ...],
    market: str,
    horizon: str,
    parameters: dict[str, Any],
    parameter_search_budget: int,
    cost_model: dict[str, Any],
    validation_protocol: str,
    rejection_conditions: tuple[str, ...],
) -> str:
    """Compute deterministic SHA-256 hash across all pre-declared strategy specifications."""
    spec_dict = {
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "source_reference": source_reference,
        "hypothesis": hypothesis.strip(),
        "required_features": sorted(required_features),
        "market": market.upper().strip(),
        "horizon": horizon.strip(),
        "parameters": parameters,
        "parameter_search_budget": parameter_search_budget,
        "cost_model": cost_model,
        "validation_protocol": validation_protocol.strip(),
        "rejection_conditions": sorted(rejection_conditions),
    }
    canonical_json = json.dumps(spec_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StrategyCandidate:
    """Pre-declared immutable strategy hypothesis contract."""

    strategy_id: str
    strategy_version: str
    source_reference: str
    hypothesis: str
    required_features: tuple[str, ...]
    market: str
    horizon: str
    parameters: dict[str, Any]
    parameter_search_budget: int
    cost_model: dict[str, Any]
    validation_protocol: str
    rejection_conditions: tuple[str, ...]
    frozen_at_utc: datetime
    strategy_hash: str = ""

    def __post_init__(self) -> None:
        if not self.strategy_id or not self.strategy_version:
            raise ValueError("strategy_id and strategy_version must be non-empty")
        if self.parameter_search_budget <= 0:
            raise ValueError("parameter_search_budget must be strictly positive")
        if not self.rejection_conditions:
            raise ValueError("Pre-committed rejection_conditions are mandatory")

        computed_hash = compute_strategy_hash(
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            source_reference=self.source_reference,
            hypothesis=self.hypothesis,
            required_features=self.required_features,
            market=self.market,
            horizon=self.horizon,
            parameters=self.parameters,
            parameter_search_budget=self.parameter_search_budget,
            cost_model=self.cost_model,
            validation_protocol=self.validation_protocol,
            rejection_conditions=self.rejection_conditions,
        )
        if not self.strategy_hash:
            object.__setattr__(self, "strategy_hash", computed_hash)
        elif self.strategy_hash != computed_hash:
            raise ValueError(
                f"Provided strategy_hash '{self.strategy_hash}' does not match "
                f"deterministically computed hash '{computed_hash}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert strategy candidate to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "source_reference": self.source_reference,
            "hypothesis": self.hypothesis,
            "required_features": list(self.required_features),
            "market": self.market,
            "horizon": self.horizon,
            "parameters": self.parameters,
            "parameter_search_budget": self.parameter_search_budget,
            "cost_model": self.cost_model,
            "validation_protocol": self.validation_protocol,
            "rejection_conditions": list(self.rejection_conditions),
            "frozen_at": to_iso_utc(ensure_utc(self.frozen_at_utc)),
            "strategy_hash": self.strategy_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyCandidate:
        """Reconstruct StrategyCandidate from dictionary."""
        return cls(
            strategy_id=str(data["strategy_id"]),
            strategy_version=str(data["strategy_version"]),
            source_reference=str(data["source_reference"]),
            hypothesis=str(data["hypothesis"]),
            required_features=tuple(data.get("required_features", [])),
            market=str(data.get("market", "BTC-5M")),
            horizon=str(data.get("horizon", "5m")),
            parameters=dict(data.get("parameters", {})),
            parameter_search_budget=int(data.get("parameter_search_budget", 10)),
            cost_model=dict(data.get("cost_model", {})),
            validation_protocol=str(data.get("validation_protocol", "PURGED_WALK_FORWARD")),
            rejection_conditions=tuple(data.get("rejection_conditions", [])),
            frozen_at_utc=datetime.fromisoformat(data["frozen_at"]),
            strategy_hash=str(data.get("strategy_hash", "")),
        )
