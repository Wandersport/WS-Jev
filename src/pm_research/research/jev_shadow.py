"""Prospective shadow capture and resolution scoring runner for TypeSafe Jev.

SHADOW RESEARCH ONLY:
This module captures prospective forecasts from Jev alongside market baselines
and frozen Ilsa estimates. It never executes orders, never modifies portfolio sizing,
and never calls PaperBroker, Kett, or Bram.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pm_research.calibration.metrics import binary_log_loss
from pm_research.config import SystemConfig
from pm_research.data.public_adapter import PublicMarketDataAdapter
from pm_research.pipeline.holt import HoltResearcher
from pm_research.pipeline.ilsa import IlsaEstimator
from pm_research.pipeline.rigo import RigoIngestor
from pm_research.research.jev_openrouter import (
    CONDITION_BLIND,
    CONDITION_MARKET_AWARE,
    JevCaptureRecord,
    JevOpenRouterClient,
    JevResolutionScore,
)
from pm_research.utils import ensure_utc, generate_id, now_utc, parse_iso_utc, to_iso_utc

if TYPE_CHECKING:
    from pm_research.storage.db import Database

logger = logging.getLogger(__name__)


@dataclass
class JevCycleSummary:
    """Summary of a prospective Jev shadow forecasting cycle."""

    cycle_id: str
    timestamp_utc: datetime
    markets_discovered: int
    markets_eligible: int
    markets_captured: int
    jev_requests_sent: int
    jev_cache_hits: int
    failed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float | None
    captures: list[JevCaptureRecord] = field(default_factory=list)


@dataclass
class JevScoringSummary:
    """Summary of prospective forecast resolution scoring."""

    total_captures: int
    unresolved_captures: int
    resolved_captures: int
    scoring_status: str  # "WAITING_FOR_FUTURE_RESOLUTIONS" or "SCORED"
    market_brier: float | None = None
    ilsa_brier: float | None = None
    jev_blind_brier: float | None = None
    jev_market_aware_brier: float | None = None
    delta_jev_blind_vs_market: float | None = None
    delta_jev_aware_vs_market: float | None = None
    scores: list[JevResolutionScore] = field(default_factory=list)


class JevShadowRunner:
    """Orchestrates prospective shadow forecasting on active unresolved prediction markets."""

    def __init__(
        self,
        db: Database,
        jev_client: JevOpenRouterClient | None = None,
        market_adapter: PublicMarketDataAdapter | None = None,
        config: SystemConfig | None = None,
    ) -> None:
        self.db = db
        self.client = jev_client or JevOpenRouterClient()
        self.market_adapter = market_adapter or PublicMarketDataAdapter()
        self.config = config or SystemConfig()

        # Frozen deterministic pipeline stages (for comparison only)
        self.rigo = RigoIngestor(self.config)
        self.holt = HoltResearcher(self.config)
        self.ilsa = IlsaEstimator(self.config)

    def run_prospective_cycle(
        self,
        max_markets: int = 20,
        min_liquidity: float = 1000.0,
        enable_market_aware: bool = True,
        bypass_cache: bool = False,
    ) -> JevCycleSummary:
        """Discover active unresolved markets, obtain frozen baseline + Jev shadow forecasts, and persist."""
        cycle_id = generate_id("jev_cyc")
        now = ensure_utc(now_utc())

        logger.info(f"Beginning prospective Jev shadow cycle {cycle_id} at {to_iso_utc(now)}...")

        # 1. Discover public active unresolved markets via unauthenticated GET adapter
        raw_items = self.market_adapter.fetch_public_markets(limit=min(100, max_markets * 4))
        discovered_count = len(raw_items) if raw_items else 0

        if not raw_items:
            logger.warning("No active markets returned by Polymarket public endpoint.")
            return JevCycleSummary(
                cycle_id=cycle_id,
                timestamp_utc=now,
                markets_discovered=0,
                markets_eligible=0,
                markets_captured=0,
                jev_requests_sent=0,
                jev_cache_hits=0,
                failed_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost=0.0,
            )

        # 2. Filter for binary unresolved markets with valid price and future resolution
        eligible_candidates: list[dict[str, Any]] = []
        for item in raw_items:
            question = str(item.get("question", "")).strip()
            if not question:
                continue

            last_price = item.get("last_price")
            if last_price is None or not (0.01 <= float(last_price) <= 0.99):
                continue

            # Check future resolution time (must be at least 1 hour in the future)
            end_date_val = item.get("resolution_time") or item.get("endDate") or item.get("resolutionTime")
            if not end_date_val:
                continue
            try:
                res_dt = parse_iso_utc(str(end_date_val))
                if (res_dt - now).total_seconds() < 3600.0:
                    continue
            except Exception:
                continue

            # Liquidity check
            liq = float(item.get("liquidity", 0.0) or 0.0)
            if liq < min_liquidity:
                continue

            eligible_candidates.append(item)
            if len(eligible_candidates) >= max_markets:
                break

        eligible_count = len(eligible_candidates)
        logger.info(f"Discovered {discovered_count} markets; {eligible_count} eligible for prospective capture.")

        # 3. Capture forecasts for each eligible market
        captures: list[JevCaptureRecord] = []
        total_requests = 0
        cache_hits = 0
        failed_requests = 0
        total_in_tokens = 0
        total_out_tokens = 0
        total_cost = 0.0

        for mkt in eligible_candidates:
            raw_id = str(mkt.get("market_id") or mkt.get("id") or mkt.get("slug") or "")
            market_id = raw_id if raw_id.startswith("pub_") else f"pub_{raw_id}"
            question = str(mkt.get("question", "")).strip()
            category = str(mkt.get("category", "PUBLIC")).strip().upper()
            criteria = str(mkt.get("description", "")).strip() or None
            end_date_val = mkt.get("resolution_time") or mkt.get("endDate") or mkt.get("resolutionTime")
            res_dt = parse_iso_utc(str(end_date_val))
            p_market = float(mkt["last_price"])

            # Compute frozen Ilsa estimate via Rigo -> Holt -> Ilsa
            m_snap = self.rigo.ingest_snapshot(mkt, cycle_id=cycle_id, current_time=now)
            if m_snap is not None:
                features = self.holt.extract_features(m_snap, current_time=now)
                est = self.ilsa.estimate(m_snap, features)
                p_ilsa = est.q_hat
            else:
                p_ilsa = p_market

            capture_id = generate_id("jev_cap")

            # Condition A: JEV_BLIND (no market probability provided)
            p_blind: float | None = None
            try:
                state_blind = self.client.build_state(
                    question=question,
                    criteria=criteria,
                    category=category,
                    captured_at=now,
                    resolution_time=res_dt,
                    condition=CONDITION_BLIND,
                )
                resp_blind, is_cached, req_hash = self.client.query_decision(
                    state=state_blind, bypass_cache=bypass_cache
                )
                total_requests += 1
                if is_cached:
                    cache_hits += 1

                fc_blind = self.client.parse_forecast(
                    response_data=resp_blind,
                    capture_id=capture_id,
                    market_id=market_id,
                    condition=CONDITION_BLIND,
                    question=question,
                    criteria=criteria,
                    resolution_time=res_dt,
                    captured_at=now,
                    request_hash=req_hash,
                )
                self.db.save_jev_forecast(fc_blind)
                p_blind = fc_blind.jev_yes_probability
                total_in_tokens += fc_blind.input_tokens
                total_out_tokens += fc_blind.output_tokens
                if fc_blind.cost:
                    total_cost += fc_blind.cost
            except Exception as e:
                logger.warning(f"Failed Jev Blind forecast for {market_id}: {e}")
                failed_requests += 1

            # Condition B: JEV_MARKET_AWARE (market probability supplied as consensus)
            p_aware: float | None = None
            if enable_market_aware:
                try:
                    state_aware = self.client.build_state(
                        question=question,
                        criteria=criteria,
                        category=category,
                        captured_at=now,
                        resolution_time=res_dt,
                        condition=CONDITION_MARKET_AWARE,
                        market_prob=p_market,
                    )
                    resp_aware, is_cached_aware, req_hash_aware = self.client.query_decision(
                        state=state_aware, bypass_cache=bypass_cache
                    )
                    total_requests += 1
                    if is_cached_aware:
                        cache_hits += 1

                    fc_aware = self.client.parse_forecast(
                        response_data=resp_aware,
                        capture_id=capture_id,
                        market_id=market_id,
                        condition=CONDITION_MARKET_AWARE,
                        question=question,
                        criteria=criteria,
                        resolution_time=res_dt,
                        captured_at=now,
                        request_hash=req_hash_aware,
                    )
                    self.db.save_jev_forecast(fc_aware)
                    p_aware = fc_aware.jev_yes_probability
                    total_in_tokens += fc_aware.input_tokens
                    total_out_tokens += fc_aware.output_tokens
                    if fc_aware.cost:
                        total_cost += fc_aware.cost
                except Exception as e:
                    logger.warning(f"Failed Jev Market-Aware forecast for {market_id}: {e}")
                    failed_requests += 1

            # Save united prospective capture record
            cap_record = JevCaptureRecord(
                capture_id=capture_id,
                market_id=market_id,
                captured_at=now,
                market_question=question,
                category=category,
                resolution_time=res_dt,
                market_prob=round(p_market, 4),
                ilsa_prob=round(p_ilsa, 4),
                jev_blind_prob=p_blind,
                jev_market_aware_prob=p_aware,
                resolved_outcome=None,
                resolved_at=None,
                metadata={
                    "cycle_id": cycle_id,
                    "liquidity": float(mkt.get("liquidity", 0.0) or 0.0),
                    "volume_24h": float(mkt.get("volume24hr", 0.0) or 0.0),
                },
            )
            self.db.save_jev_capture(cap_record)
            captures.append(cap_record)

        return JevCycleSummary(
            cycle_id=cycle_id,
            timestamp_utc=now,
            markets_discovered=discovered_count,
            markets_eligible=eligible_count,
            markets_captured=len(captures),
            jev_requests_sent=total_requests,
            jev_cache_hits=cache_hits,
            failed_requests=failed_requests,
            total_input_tokens=total_in_tokens,
            total_output_tokens=total_out_tokens,
            total_cost=round(total_cost, 6) if total_cost > 0 else None,
            captures=captures,
        )


class JevResolutionScorer:
    """Evaluates prospective forecasts as markets resolve, without fabricating unobserved outcomes."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def evaluate_pending_resolutions(self) -> JevScoringSummary:
        """Check unresolved prospective captures against known resolutions and score completed markets."""
        all_captures = self.db.get_jev_captures(limit=1000)
        unresolved = [c for c in all_captures if c.resolved_outcome is None]
        resolved_count = len(all_captures) - len(unresolved)

        if not all_captures:
            return JevScoringSummary(
                total_captures=0,
                unresolved_captures=0,
                resolved_captures=0,
                scoring_status="NO_CAPTURES_FOUND",
            )

        if resolved_count == 0:
            return JevScoringSummary(
                total_captures=len(all_captures),
                unresolved_captures=len(unresolved),
                resolved_captures=0,
                scoring_status="WAITING_FOR_FUTURE_RESOLUTIONS",
            )

        scores = self.db.get_jev_resolution_scores()
        if not scores:
            return JevScoringSummary(
                total_captures=len(all_captures),
                unresolved_captures=len(unresolved),
                resolved_captures=resolved_count,
                scoring_status="WAITING_FOR_FUTURE_RESOLUTIONS",
            )

        mb = sum(s.market_brier for s in scores) / len(scores)
        ib = sum(s.ilsa_brier for s in scores) / len(scores)
        blind_scores = [s.jev_blind_brier for s in scores if s.jev_blind_brier is not None]
        jb = sum(blind_scores) / len(blind_scores) if blind_scores else None
        aware_scores = [s.jev_market_aware_brier for s in scores if s.jev_market_aware_brier is not None]
        ja = sum(aware_scores) / len(aware_scores) if aware_scores else None

        delta_b = (jb - mb) if jb is not None else None
        delta_a = (ja - mb) if ja is not None else None

        return JevScoringSummary(
            total_captures=len(all_captures),
            unresolved_captures=len(unresolved),
            resolved_captures=len(scores),
            scoring_status="SCORED",
            market_brier=round(mb, 4),
            ilsa_brier=round(ib, 4),
            jev_blind_brier=round(jb, 4) if jb is not None else None,
            jev_market_aware_brier=round(ja, 4) if ja is not None else None,
            delta_jev_blind_vs_market=round(delta_b, 4) if delta_b is not None else None,
            delta_jev_aware_vs_market=round(delta_a, 4) if delta_a is not None else None,
            scores=scores,
        )

    def score_capture(
        self,
        capture: JevCaptureRecord,
        resolved_outcome: str,
        resolved_at: datetime,
    ) -> JevResolutionScore:
        """Score a prospective capture against its ground-truth resolution."""
        outcome_val = 1.0 if resolved_outcome.upper() == "YES" else 0.0
        now = ensure_utc(datetime.now(timezone.utc))

        mkt_brier = (capture.market_prob - outcome_val) ** 2
        mkt_ll = binary_log_loss(outcome_val, capture.market_prob)

        ilsa_brier = (capture.ilsa_prob - outcome_val) ** 2
        ilsa_ll = binary_log_loss(outcome_val, capture.ilsa_prob)

        blind_brier = (
            (capture.jev_blind_prob - outcome_val) ** 2
            if capture.jev_blind_prob is not None
            else None
        )
        blind_ll = (
            binary_log_loss(outcome_val, capture.jev_blind_prob)
            if capture.jev_blind_prob is not None
            else None
        )

        aware_brier = (
            (capture.jev_market_aware_prob - outcome_val) ** 2
            if capture.jev_market_aware_prob is not None
            else None
        )
        aware_ll = (
            binary_log_loss(outcome_val, capture.jev_market_aware_prob)
            if capture.jev_market_aware_prob is not None
            else None
        )

        score = JevResolutionScore(
            score_id=generate_id("jev_sc"),
            capture_id=capture.capture_id,
            market_id=capture.market_id,
            resolved_outcome=resolved_outcome.upper(),
            resolved_at=ensure_utc(resolved_at),
            scored_at=now,
            market_prob=capture.market_prob,
            ilsa_prob=capture.ilsa_prob,
            jev_blind_prob=capture.jev_blind_prob,
            jev_market_aware_prob=capture.jev_market_aware_prob,
            market_brier=round(mkt_brier, 4),
            ilsa_brier=round(ilsa_brier, 4),
            jev_blind_brier=round(blind_brier, 4) if blind_brier is not None else None,
            jev_market_aware_brier=round(aware_brier, 4) if aware_brier is not None else None,
            market_log_loss=round(mkt_ll, 4),
            ilsa_log_loss=round(ilsa_ll, 4),
            jev_blind_log_loss=round(blind_ll, 4) if blind_ll is not None else None,
            jev_market_aware_log_loss=round(aware_ll, 4) if aware_ll is not None else None,
        )
        self.db.update_jev_capture_resolution(capture.capture_id, resolved_outcome.upper(), resolved_at)
        self.db.save_jev_resolution_score(score)
        return score

