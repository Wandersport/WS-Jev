"""Offline data quality audit and target label generation for lead-lag research.

Strictly read-only post-processing.
Labels (Δq_1s, Δq_2s, Δq_3s, Δq_5s, Δq_10s, Δq_15s, Δq_30s) are constructed
strictly offline by joining samples with future samples at t + L.

Guarantees ZERO label leakage during prospective collection.
"""

from __future__ import annotations

import math
from typing import Any

from pm_research.research.btc5m.leadlag_experiment import (
    EXPERIMENT_ID,
    EXPERIMENT_SPEC_HASH,
    PREDECLARED_LAGS_SEC,
)
from pm_research.storage.db import Database


def logit(p: float, eps: float = 1e-6) -> float:
    p_clamped = max(eps, min(1.0 - eps, p))
    return math.log(p_clamped / (1.0 - p_clamped))


class LeadLagAudit:
    """Offline data quality audit and target variable calculation."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def compute_offline_target_pairs(
        self,
        round_slug: str | None = None,
    ) -> list[dict[str, Any]]:
        """Compute future delta movements Δq_L and Δlogit_q_L for each valid sample at time t.

        Strictly offline post-processing. Never accessible during prospective feature capture.
        """
        samples = self.db.get_leadlag_samples(round_slug=round_slug)
        if not samples:
            return []

        # Index samples by (round_slug, sample_target_ts_ms)
        sample_map: dict[tuple[str, int], dict[str, Any]] = {
            (s["round_slug"], s["sample_target_ts_ms"]): s for s in samples
        }

        augmented_records: list[dict[str, Any]] = []

        for s in samples:
            if not s.get("is_valid") or s.get("poly_midpoint") is None:
                continue

            q_t = float(s["poly_midpoint"])
            logit_q_t = logit(q_t)
            rd = s["round_slug"]
            t_ms = s["sample_target_ts_ms"]

            row: dict[str, Any] = {
                "sample_id": s["sample_id"],
                "round_slug": rd,
                "target_ts_ms": t_ms,
                "q_t": q_t,
                "logit_q_t": logit_q_t,
                "seconds_remaining": s["seconds_remaining"],
                # Contemporaneous features
                "poly_spread": s.get("poly_spread"),
                "poly_return_1s": s.get("poly_return_1s"),
                "poly_return_5s": s.get("poly_return_5s"),
                "binance_microprice_offset_bps": s.get("binance_microprice_offset_bps"),
                "binance_return_1s_bps": s.get("binance_return_1s_bps"),
                "binance_return_5s_bps": s.get("binance_return_5s_bps"),
                "binance_taker_flow_5s": s.get("binance_taker_flow_5s"),
                "binance_top5_depth_imbalance": s.get("binance_top5_depth_imbalance"),
            }

            # Calculate future deltas
            for lag_sec in PREDECLARED_LAGS_SEC:
                future_ts = t_ms + lag_sec * 1000
                future_sample = sample_map.get((rd, future_ts))
                if future_sample and future_sample.get("poly_midpoint") is not None:
                    q_future = float(future_sample["poly_midpoint"])
                    row[f"delta_q_{lag_sec}s"] = round(q_future - q_t, 5)
                    row[f"delta_logit_{lag_sec}s"] = round(logit(q_future) - logit_q_t, 5)
                else:
                    row[f"delta_q_{lag_sec}s"] = None
                    row[f"delta_logit_{lag_sec}s"] = None

            augmented_records.append(row)

        return augmented_records

    def run_quality_audit(self) -> dict[str, Any]:
        """Perform comprehensive statistical and timing provenance audit."""
        summary = self.db.get_leadlag_audit_summary()
        latest_hb = self.db.get_leadlag_latest_heartbeat()

        rounds = self.db.get_leadlag_rounds()

        return {
            "experiment_id": EXPERIMENT_ID,
            "experiment_spec_hash": EXPERIMENT_SPEC_HASH,
            "summary": summary,
            "latest_heartbeat": latest_hb,
            "physical_rounds_count": len(rounds),
            "rounds_list": [r["round_slug"] for r in rounds[-10:]],
            "predeclared_lags": list(PREDECLARED_LAGS_SEC),
        }
