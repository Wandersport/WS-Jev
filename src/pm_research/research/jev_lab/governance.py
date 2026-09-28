"""Multiple-testing governance and hypothesis tracking for Phase 9A.

RESEARCH GOVERNANCE ONLY:
Ensures the research system maintains an immutable ledger of every
hypothesis and parameter variant evaluated. Prevents selection bias
and p-hacking by tracking trial counts and providing multiple-testing adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pm_research.utils import ensure_utc, generate_id, to_iso_utc


@dataclass(frozen=True)
class HypothesisAttempt:
    """Immutable record of an individual hypothesis or parameter variant evaluation."""

    attempt_id: str
    strategy_id: str
    strategy_hash: str
    variant_id: str
    parameter_values: dict[str, Any]
    status: str  # "SUBMITTED", "REJECTED", "SURVIVED"
    rejection_reason: str | None
    timestamp_utc: datetime
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert attempt to dictionary representation."""
        return {
            "attempt_id": self.attempt_id,
            "strategy_id": self.strategy_id,
            "strategy_hash": self.strategy_hash,
            "variant_id": self.variant_id,
            "parameter_values": self.parameter_values,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "timestamp": to_iso_utc(ensure_utc(self.timestamp_utc)),
            "metrics": self.metrics,
        }


class MultipleTestingLedger:
    """Audit ledger tracking all scientific hypothesis and parameter attempts."""

    def __init__(self) -> None:
        self._attempts: list[HypothesisAttempt] = []

    def record_attempt(
        self,
        strategy_id: str,
        strategy_hash: str,
        variant_id: str,
        parameter_values: dict[str, Any],
        status: str,
        rejection_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> HypothesisAttempt:
        """Record a new evaluation attempt into the permanent ledger."""
        attempt = HypothesisAttempt(
            attempt_id=generate_id("hyp"),
            strategy_id=strategy_id,
            strategy_hash=strategy_hash,
            variant_id=variant_id,
            parameter_values=parameter_values,
            status=status,
            rejection_reason=rejection_reason,
            timestamp_utc=datetime.now(timezone.utc),
            metrics=metrics or {},
        )
        self._attempts.append(attempt)
        return attempt

    @property
    def total_variants_attempted(self) -> int:
        """Total number of parameter variants evaluated across all strategies."""
        return len(self._attempts)

    @property
    def total_hypotheses_attempted(self) -> int:
        """Number of unique strategy hypotheses attempted."""
        return len({a.strategy_id for a in self._attempts})

    @property
    def total_rejected(self) -> int:
        """Number of attempts rejected by pre-committed rejection conditions."""
        return sum(1 for a in self._attempts if a.status == "REJECTED")

    @property
    def total_surviving(self) -> int:
        """Number of attempts that survived rejection criteria."""
        return sum(1 for a in self._attempts if a.status == "SURVIVED")

    def bonferroni_alpha(self, nominal_alpha: float = 0.05) -> float:
        """Compute conservative Bonferroni family-wise significance threshold."""
        n = max(1, self.total_variants_attempted)
        return nominal_alpha / n

    def holm_bonferroni_threshold(
        self,
        rank: int,
        nominal_alpha: float = 0.05,
    ) -> float:
        """Compute Holm-Bonferroni stepwise significance threshold for rank (1-indexed)."""
        m = max(1, self.total_variants_attempted)
        divisor = max(1, m - rank + 1)
        return nominal_alpha / divisor

    def summary(self) -> dict[str, Any]:
        """Generate audit report of multiple-testing governance metrics."""
        return {
            "total_hypotheses_attempted": self.total_hypotheses_attempted,
            "total_variants_attempted": self.total_variants_attempted,
            "total_rejected": self.total_rejected,
            "total_surviving": self.total_surviving,
            "rejection_rate": (
                round(self.total_rejected / self.total_variants_attempted, 4)
                if self.total_variants_attempted > 0
                else 0.0
            ),
            "bonferroni_adjusted_alpha_5pct": round(self.bonferroni_alpha(0.05), 8),
        }
