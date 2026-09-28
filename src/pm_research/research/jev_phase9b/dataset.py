"""Dataset extractor and chronological cohort partitioner for Phase 9B.

RETROSPECTIVE DEVELOPMENT DATASET ONLY:
Extracts at most ONE observation per physical round from btc5m_leadlag_v1 data.
Enforces:
1. Primary anchor near 120 seconds remaining (within ±10s tolerance).
2. Future receive-time sample at t + 30s (within ±3000ms tolerance).
3. Zero taker-flow fields (excluded due to v1 measurement defect).
4. No future leakage into the observation state.
5. Strict chronological splits: 60% Train, 20% Calibration, 20% Retrospective Holdout.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pm_research.research.jev_lab.contract import DecisionObservation
from pm_research.research.jev_phase9b.spec import (
    ANCHOR_SECONDS_REMAINING,
    ANCHOR_TOLERANCE_SECONDS,
    CHOICE_MEANINGFUL_MOVE,
    CHOICE_QUIET,
    FUTURE_HORIZON_MS,
    FUTURE_TOLERANCE_MS,
    TASK_ID,
)


@dataclass(frozen=True)
class Phase9bSample:
    """Immutable single-round observation and future evaluation target."""

    round_slug: str
    sample_id: str
    as_of_ts_ms: int
    seconds_remaining: int
    poly_midpoint: float
    poly_spread: float
    future_sample_id: str
    future_ts_ms: int
    future_poly_midpoint: float
    absolute_move: float
    objective_label: str  # MEANINGFUL_MOVE or QUIET
    features: dict[str, Any]
    split: str  # "train", "calibration", "holdout"

    def to_decision_observation(self) -> DecisionObservation:
        """Construct canonical DecisionObservation for Jev Lab evaluation."""
        as_of_utc = datetime.fromtimestamp(self.as_of_ts_ms / 1000.0, tz=timezone.utc)
        return DecisionObservation(
            observation_id=f"obs_p9b_{self.round_slug}",
            task_id=TASK_ID,
            market_round_id=self.round_slug,
            as_of_ts_utc=as_of_utc,
            state_payload=self.features,
            provenance="BTC5M_LEADLAG_V1_RETROSPECTIVE_DEV",
        )


class Phase9bDataset:
    """Manages the extracted retrospective development dataset for Phase 9B."""

    def __init__(
        self,
        samples: list[Phase9bSample],
        total_rounds: int,
        exclusion_reasons: dict[str, int],
    ) -> None:
        self.samples = samples
        self.total_rounds = total_rounds
        self.exclusion_reasons = exclusion_reasons

    @property
    def eligible_count(self) -> int:
        return len(self.samples)

    @property
    def excluded_count(self) -> int:
        return self.total_rounds - self.eligible_count

    @property
    def train_samples(self) -> list[Phase9bSample]:
        return [s for s in self.samples if s.split == "train"]

    @property
    def calib_samples(self) -> list[Phase9bSample]:
        return [s for s in self.samples if s.split == "calibration"]

    @property
    def holdout_samples(self) -> list[Phase9bSample]:
        return [s for s in self.samples if s.split == "holdout"]

    @classmethod
    def load_from_db(
        cls,
        db_path: str | Path = "data/pm_research.db",
    ) -> Phase9bDataset:
        """Extract and validate eligible round-level observations from database."""
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        rounds = [
            r[0]
            for r in c.execute(
                "SELECT round_slug FROM leadlag_rounds ORDER BY start_epoch ASC"
            ).fetchall()
        ]
        total_rounds = len(rounds)

        extracted: list[dict[str, Any]] = []
        exclusion_reasons: dict[str, int] = {}

        for r_slug in rounds:
            # 1. Fetch nearest sample to anchor (~120s remaining)
            rows = c.execute(
                """
                SELECT
                    sample_id, seconds_remaining, sample_actual_ts_ms,
                    poly_midpoint, poly_spread,
                    poly_return_1s, poly_return_2s, poly_return_3s,
                    poly_return_5s, poly_return_10s, poly_return_30s,
                    binance_mid_price, binance_spread_bps, binance_microprice_offset_bps,
                    binance_top1_depth_imbalance, binance_top5_depth_imbalance, binance_top20_depth_imbalance,
                    binance_return_since_open_bps, binance_return_1s_bps, binance_return_2s_bps,
                    binance_return_3s_bps, binance_return_5s_bps, binance_return_10s_bps,
                    binance_return_30s_bps, binance_return_60s_bps,
                    is_valid, poly_is_valid, binance_is_valid
                FROM leadlag_samples
                WHERE round_slug = ?
                ORDER BY abs(seconds_remaining - ?) ASC
                LIMIT 1
                """,
                (r_slug, ANCHOR_SECONDS_REMAINING),
            ).fetchall()

            if not rows:
                exclusion_reasons["no_samples_in_round"] = (
                    exclusion_reasons.get("no_samples_in_round", 0) + 1
                )
                continue

            row = rows[0]
            sec_rem = row[1]
            ts_ms = row[2]
            poly_mid = row[3]
            poly_spread = row[4]
            is_valid = row[25]
            poly_valid = row[26]
            bin_valid = row[27]

            if abs(sec_rem - ANCHOR_SECONDS_REMAINING) > ANCHOR_TOLERANCE_SECONDS:
                exclusion_reasons["anchor_beyond_tolerance"] = (
                    exclusion_reasons.get("anchor_beyond_tolerance", 0) + 1
                )
                continue

            if not (is_valid and poly_valid and bin_valid):
                exclusion_reasons["microstructure_invalid_at_anchor"] = (
                    exclusion_reasons.get("microstructure_invalid_at_anchor", 0) + 1
                )
                continue

            if poly_mid is None or poly_spread is None or poly_spread <= 0.0:
                exclusion_reasons["invalid_midpoint_or_spread"] = (
                    exclusion_reasons.get("invalid_midpoint_or_spread", 0) + 1
                )
                continue

            # 2. Query future sample at approximately t + 30s
            target_future_ts = ts_ms + FUTURE_HORIZON_MS
            future_rows = c.execute(
                """
                SELECT sample_id, sample_actual_ts_ms, poly_midpoint, poly_is_valid
                FROM leadlag_samples
                WHERE round_slug = ? AND abs(sample_actual_ts_ms - ?) <= ?
                ORDER BY abs(sample_actual_ts_ms - ?) ASC
                LIMIT 1
                """,
                (r_slug, target_future_ts, FUTURE_TOLERANCE_MS, target_future_ts),
            ).fetchall()

            if not future_rows:
                exclusion_reasons["no_future_sample_within_tolerance"] = (
                    exclusion_reasons.get("no_future_sample_within_tolerance", 0) + 1
                )
                continue

            fut = future_rows[0]
            fut_sample_id = fut[0]
            fut_ts_ms = fut[1]
            fut_mid = fut[2]
            fut_valid = fut[3]

            if fut_mid is None or not fut_valid:
                exclusion_reasons["future_poly_midpoint_invalid"] = (
                    exclusion_reasons.get("future_poly_midpoint_invalid", 0) + 1
                )
                continue

            # 3. Objective Label Construction
            abs_move = abs(fut_mid - poly_mid)
            label = CHOICE_MEANINGFUL_MOVE if abs_move >= poly_spread else CHOICE_QUIET

            # 4. Assemble State Features (Zero Taker Flow, explicit None for missing)
            features: dict[str, Any] = {
                "poly_midpoint": round(float(poly_mid), 4),
                "poly_spread": round(float(poly_spread), 4),
                "seconds_remaining": int(sec_rem),
                "poly_return_1s": (
                    round(float(row[5]), 6) if row[5] is not None else None
                ),
                "poly_return_2s": (
                    round(float(row[6]), 6) if row[6] is not None else None
                ),
                "poly_return_3s": (
                    round(float(row[7]), 6) if row[7] is not None else None
                ),
                "poly_return_5s": (
                    round(float(row[8]), 6) if row[8] is not None else None
                ),
                "poly_return_10s": (
                    round(float(row[9]), 6) if row[9] is not None else None
                ),
                "poly_return_30s": (
                    round(float(row[10]), 6) if row[10] is not None else None
                ),
                "binance_mid_price": (
                    round(float(row[11]), 2) if row[11] is not None else None
                ),
                "binance_spread_bps": (
                    round(float(row[12]), 2) if row[12] is not None else None
                ),
                "binance_microprice_offset_bps": (
                    round(float(row[13]), 2) if row[13] is not None else None
                ),
                "binance_top1_depth_imbalance": (
                    round(float(row[14]), 4) if row[14] is not None else None
                ),
                "binance_top5_depth_imbalance": (
                    round(float(row[15]), 4) if row[15] is not None else None
                ),
                "binance_top20_depth_imbalance": (
                    round(float(row[16]), 4) if row[16] is not None else None
                ),
                "binance_return_since_open_bps": (
                    round(float(row[17]), 2) if row[17] is not None else None
                ),
                "binance_return_1s_bps": (
                    round(float(row[18]), 2) if row[18] is not None else None
                ),
                "binance_return_2s_bps": (
                    round(float(row[19]), 2) if row[19] is not None else None
                ),
                "binance_return_3s_bps": (
                    round(float(row[20]), 2) if row[20] is not None else None
                ),
                "binance_return_5s_bps": (
                    round(float(row[21]), 2) if row[21] is not None else None
                ),
                "binance_return_10s_bps": (
                    round(float(row[22]), 2) if row[22] is not None else None
                ),
                "binance_return_30s_bps": (
                    round(float(row[23]), 2) if row[23] is not None else None
                ),
                "binance_return_60s_bps": (
                    round(float(row[24]), 2) if row[24] is not None else None
                ),
            }

            extracted.append(
                {
                    "round_slug": r_slug,
                    "sample_id": row[0],
                    "as_of_ts_ms": ts_ms,
                    "seconds_remaining": sec_rem,
                    "poly_midpoint": poly_mid,
                    "poly_spread": poly_spread,
                    "future_sample_id": fut_sample_id,
                    "future_ts_ms": fut_ts_ms,
                    "future_poly_midpoint": fut_mid,
                    "absolute_move": abs_move,
                    "objective_label": label,
                    "features": features,
                }
            )

        conn.close()

        # 5. Chronological Partitioning (60% Train, 20% Calib, 20% Holdout)
        n = len(extracted)
        n_train = int(n * 0.60)
        n_calib = int(n * 0.20)

        final_samples: list[Phase9bSample] = []
        for i, item in enumerate(extracted):
            if i < n_train:
                split = "train"
            elif i < n_train + n_calib:
                split = "calibration"
            else:
                split = "holdout"

            final_samples.append(
                Phase9bSample(
                    round_slug=item["round_slug"],
                    sample_id=item["sample_id"],
                    as_of_ts_ms=item["as_of_ts_ms"],
                    seconds_remaining=item["seconds_remaining"],
                    poly_midpoint=item["poly_midpoint"],
                    poly_spread=item["poly_spread"],
                    future_sample_id=item["future_sample_id"],
                    future_ts_ms=item["future_ts_ms"],
                    future_poly_midpoint=item["future_poly_midpoint"],
                    absolute_move=item["absolute_move"],
                    objective_label=item["objective_label"],
                    features=item["features"],
                    split=split,
                )
            )

        return cls(
            samples=final_samples,
            total_rounds=total_rounds,
            exclusion_reasons=exclusion_reasons,
        )

    def summary(self) -> dict[str, Any]:
        """Generate high-level metadata summary of dataset eligibility and splits."""
        labels = [s.objective_label for s in self.samples]
        meaningful = sum(1 for lbl in labels if lbl == CHOICE_MEANINGFUL_MOVE)
        quiet = sum(1 for lbl in labels if lbl == CHOICE_QUIET)

        return {
            "total_rounds": self.total_rounds,
            "eligible_rounds": self.eligible_count,
            "excluded_rounds": self.excluded_count,
            "exclusion_reasons": self.exclusion_reasons,
            "meaningful_move_count": meaningful,
            "quiet_count": quiet,
            "base_rate_meaningful_move": (
                round(meaningful / self.eligible_count, 4) if self.eligible_count else 0.0
            ),
            "splits": {
                "train_count": len(self.train_samples),
                "calibration_count": len(self.calib_samples),
                "holdout_count": len(self.holdout_samples),
            },
        }
