"""BTC 5-minute prospective forecasting research laboratory.

Coordinates:
- Round discovery and contract invariant validation
- Reference stream monitoring and anchor capture
- Polymarket order book consensus capture (native vs implied)
- Binance USD-M Perpetual microstructure capture (observed open mid)
- Point-in-time immutable snapshot construction with strict freeze timing
- 4-condition Jev ablation forecasting (pinned typesafe/jev-1.13)
- Official resolution ingestion via Polymarket Data API
- Resolution scoring and round-clustered bootstrap evaluation

Strict Paper-Only Research Invariants:
- Zero order generation, zero trade execution, zero paper trading
- Zero wallet/private key functionality
- Completely isolated from Kett, Bram, and PaperBroker
"""

from __future__ import annotations

import logging
import math
import random
import time
from datetime import datetime, timezone
from typing import Any, Sequence

from pm_research.research.btc5m.ablation import (
    COND_A_REF_ONLY,
    COND_B_REF_PERP,
    COND_C_MARKET_AWARE,
    COND_D_FULL,
    BTC5mAblationForecast,
    BTC5mAblationRunner,
)
from pm_research.research.btc5m.binance_feed import BinancePerpFeed
from pm_research.research.btc5m.contract import (
    BTC5mContractManager,
    BTC5mOfficialResolution,
    BTC5mRoundInfo,
)
from pm_research.research.btc5m.poly_book import PolymarketBookCollector
from pm_research.research.btc5m.reference_feed import ChainlinkReferenceFeed
from pm_research.research.btc5m.snapshot import (
    STANDARD_HORIZONS_SEC,
    BTC5mFeatureSnapshot,
    build_feature_snapshot,
)
from pm_research.research.jev_openrouter import JevOpenRouterClient
from pm_research.storage.db import Database

logger = logging.getLogger(__name__)


def compute_brier(p: float, outcome_up: int) -> float:
    """Compute Brier score for a binary probability prediction."""
    return round((p - float(outcome_up)) ** 2, 6)


def compute_log_loss(p: float, outcome_up: int, eps: float = 1e-6) -> float:
    """Compute binary log loss with bounded probability clipping."""
    p_clipped = max(eps, min(1.0 - eps, p))
    y = float(outcome_up)
    loss = -(y * math.log(p_clipped) + (1.0 - y) * math.log(1.0 - p_clipped))
    return round(loss, 6)


class BTC5mShadowLab:
    """Research coordinator for prospective BTC 5-minute probability forecasting."""

    def __init__(
        self,
        db: Database | None = None,
        jev_client: JevOpenRouterClient | None = None,
        horizons_sec: Sequence[int] = STANDARD_HORIZONS_SEC,
    ) -> None:
        self.db = db or Database()
        self.jev_client = jev_client or JevOpenRouterClient()
        self.contract_mgr = BTC5mContractManager()
        self.ref_feed = ChainlinkReferenceFeed()
        self.poly_collector = PolymarketBookCollector()
        self.binance_feed = BinancePerpFeed()
        self.ablation_runner = BTC5mAblationRunner(client=self.jev_client)
        self.horizons_sec = sorted(horizons_sec, reverse=True)

    def probe_connectivity(self) -> dict[str, Any]:
        """Test public endpoints and OpenRouter connectivity without waiting for a full round."""
        results: dict[str, Any] = {
            "polymarket_gamma": False,
            "polymarket_book": False,
            "binance_perp": False,
            "openrouter_jev": False,
            "details": {},
        }

        # 1. Probe Gamma API for active BTC 5m round
        try:
            round_info = self.contract_mgr.discover_active_round()
            results["polymarket_gamma"] = True
            results["details"]["active_round"] = {
                "slug": round_info.round_slug,
                "price_to_beat": round_info.price_to_beat,
                "seconds_remaining": round(round_info.seconds_remaining, 1),
            }
        except Exception as e:
            results["details"]["gamma_error"] = str(e)
            return results

        # 2. Probe CLOB order book for active tokens (Requirement B1: distinguish native vs implied)
        try:
            state = self.poly_collector.collect_market_state(
                up_token_id=round_info.up_token_id,
                down_token_id=round_info.down_token_id,
            )
            results["polymarket_book"] = state.is_valid
            results["details"]["native_market_q"] = state.market_q_primary
            results["details"]["cross_outcome_implied_q"] = state.market_q_implied_cross_outcome
            results["details"]["native_up_bid"] = state.native_up_bid
            results["details"]["native_up_ask"] = state.native_up_ask
            results["details"]["poly_spread"] = state.spread
        except Exception as e:
            results["details"]["clob_error"] = str(e)

        # 3. Probe Binance FAPI
        try:
            binance_feats = self.binance_feed.fetch_rest_snapshot(
                ref_price=round_info.price_to_beat,
                round_slug=round_info.round_slug,
            )
            results["binance_perp"] = binance_feats.is_valid
            results["details"]["binance_mid"] = binance_feats.mid_price
            results["details"]["binance_spread_bps"] = binance_feats.spread_bps
            results["details"]["binance_microprice_offset"] = binance_feats.microprice_offset_bps
        except Exception as e:
            results["details"]["binance_error"] = str(e)

        # 4. Probe OpenRouter Jev decision inference
        try:
            _key = self.jev_client._get_api_key()
            results["openrouter_jev"] = True
            results["details"]["openrouter_key_present"] = True
        except Exception as e:
            results["details"]["openrouter_error"] = str(e)

        return results

    def capture_horizon_snapshot(
        self,
        round_info: BTC5mRoundInfo,
        target_horizon_sec: int,
    ) -> BTC5mFeatureSnapshot:
        """Capture and freeze input features for a single target horizon.

        Requirements A8, A10, A11:
        - capture_started_at_ms recorded before fetching inputs
        - capture_completed_at_ms recorded after all inputs acquired
        - Authoritative freeze timestamp is completion time
        - Binance uses round_slug for observed open price (never Chainlink anchor)
        """
        t_start_ms = int(time.time() * 1000)

        # Extract reference features (use priceToBeat from event metadata or exact opening boundary)
        ref_features = self.ref_feed.compute_features(
            now_ms=t_start_ms,
            round_start_epoch=round_info.start_epoch,
            event_price_to_beat=round_info.price_to_beat,
        )

        # Extract Polymarket CLOB state
        poly_state = self.poly_collector.collect_market_state(
            up_token_id=round_info.up_token_id,
            down_token_id=round_info.down_token_id,
        )

        # Extract Binance Perpetual microstructure (Requirement A8: use round_slug for observed open)
        binance_features: Any = None
        try:
            binance_features = self.binance_feed.fetch_rest_snapshot(
                ref_price=ref_features.current_price,
                round_slug=round_info.round_slug,
            )
        except Exception as e:
            logger.warning(f"Binance snapshot fetch error: {e}")

        t_complete_ms = int(time.time() * 1000)

        # Build point-in-time frozen snapshot
        snapshot = build_feature_snapshot(
            round_slug=round_info.round_slug,
            round_start_epoch=round_info.start_epoch,
            round_end_epoch=round_info.end_epoch,
            target_horizon_sec=target_horizon_sec,
            capture_started_at_ms=t_start_ms,
            capture_completed_at_ms=t_complete_ms,
            ref_features=ref_features,
            poly_state=poly_state,
            binance_features=binance_features,
        )

        # Persist snapshot
        self.db.save_btc5m_snapshot(snapshot)
        return snapshot

    def execute_horizon_ablation(
        self,
        snapshot: BTC5mFeatureSnapshot,
        bypass_cache: bool = False,
    ) -> dict[str, BTC5mAblationForecast]:
        """Run all 4 ablation conditions on a validated snapshot and save forecasts."""
        if not snapshot.is_valid:
            logger.info(
                f"Skipping ablation for invalid snapshot {snapshot.snapshot_id}: {snapshot.skip_reason}"
            )
            return {}

        forecasts = self.ablation_runner.run_all_conditions(
            snapshot=snapshot,
            bypass_cache=bypass_cache,
        )

        for fc in forecasts.values():
            self.db.save_btc5m_forecast(fc)

        return forecasts

    def monitor_round(
        self,
        round_info: BTC5mRoundInfo,
        horizons_sec: Sequence[int] | None = None,
        poll_resolution_after: bool = True,
        max_resolution_wait_sec: int = 300,
    ) -> dict[str, Any]:
        """Monitor an active round through its horizons, capture snapshots, and evaluate."""
        horizons = sorted(horizons_sec or self.horizons_sec, reverse=True)
        self.db.save_btc5m_round(round_info, status="active")

        if not getattr(self, "_ref_listener_started", False):
            try:
                self.ref_feed.start_background_listener()
                self._ref_listener_started = True
            except Exception as e:
                logger.warning(f"Could not start background reference listener: {e}")

        # Requirement A8: Record observed Binance open mid at start of round
        try:
            b_open = self.binance_feed.fetch_rest_snapshot(ref_price=round_info.price_to_beat)
            if b_open.is_valid and b_open.mid_price > 0:
                self.binance_feed.record_round_open(
                    round_slug=round_info.round_slug,
                    mid_price=b_open.mid_price,
                    timestamp_ms=b_open.received_at_ms,
                )
        except Exception as e:
            logger.debug(f"Could not capture Binance open mid at round start: {e}")

        snapshots_captured: list[str] = []
        forecasts_produced: int = 0

        for h in horizons:
            # Target scheduled freeze time: round_end_epoch - h
            target_time = round_info.end_epoch - h
            now = time.time()
            wait_sec = target_time - now

            if wait_sec < -3.0:
                logger.warning(
                    f"Horizon {h}s already elapsed ({wait_sec:.1f}s ago). Skipping to avoid excessive timing drift."
                )
                continue

            if wait_sec > 0:
                logger.info(
                    f"Waiting {wait_sec:.1f}s for horizon {h}s remaining (round {round_info.round_slug})..."
                )
                time.sleep(wait_sec)

            # Capture snapshot at horizon
            snapshot = self.capture_horizon_snapshot(
                round_info=round_info,
                target_horizon_sec=h,
            )
            snapshots_captured.append(snapshot.snapshot_id)

            if snapshot.is_valid:
                fcs = self.execute_horizon_ablation(snapshot)
                forecasts_produced += len(fcs)
            else:
                logger.warning(
                    f"Horizon {h}s snapshot invalid ({snapshot.skip_reason}). Skipping ablation."
                )

        round_result: dict[str, Any] = {
            "round_slug": round_info.round_slug,
            "snapshots_captured": len(snapshots_captured),
            "forecasts_produced": forecasts_produced,
            "resolved": False,
        }

        if poll_resolution_after:
            wait_for_end = max(0.0, round_info.end_epoch - time.time()) + 5.0
            logger.info(f"Waiting {wait_for_end:.1f}s for round close before polling resolution...")
            time.sleep(wait_for_end)

            official_res = self.poll_round_resolution(
                round_info=round_info,
                max_wait_sec=max_resolution_wait_sec,
            )
            if official_res:
                round_result["resolved"] = True
                round_result["outcome"] = official_res.resolved_outcome
                round_result["resolution_price"] = official_res.resolution_price

        return round_result

    def poll_round_resolution(
        self,
        round_info: BTC5mRoundInfo,
        poll_interval_sec: float = 10.0,
        max_wait_sec: int = 300,
    ) -> BTC5mOfficialResolution | None:
        """Poll the Polymarket Data API until official settlement is available."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait_sec:
            try:
                res = self.contract_mgr.fetch_official_resolution(round_info)
                if res and res.is_resolved:
                    logger.info(
                        f"Round {round_info.round_slug} resolved to {res.resolved_outcome} "
                        f"(settlement price: {res.resolution_price})"
                    )
                    self.db.update_btc5m_round_resolution(
                        round_slug=round_info.round_slug,
                        resolved_outcome=res.resolved_outcome,
                        resolution_price=res.resolution_price,
                        settled_at=res.settled_at,
                    )
                    self.score_resolved_round(round_info.round_slug, res)
                    return res
            except Exception as e:
                logger.warning(f"Resolution poll error: {e}")

            time.sleep(poll_interval_sec)

        logger.warning(
            f"Timed out after {max_wait_sec}s waiting for resolution of {round_info.round_slug}"
        )
        return None

    def score_resolved_round(
        self,
        round_slug: str,
        resolution: BTC5mOfficialResolution,
    ) -> list[dict[str, Any]]:
        """Score all valid snapshots and forecasts for a resolved round.

        Requirements A1, A13, A15:
        - Primary market baseline: native observed midpoint
        - Secondary diagnostic: cross-outcome implied midpoint
        - Excludes invalid or late forecasts (is_valid == False)
        """
        outcome_up = 1 if resolution.resolved_outcome.upper() == "UP" else 0
        snapshots = self.db.get_btc5m_snapshots(round_slug=round_slug, valid_only=True)
        forecasts = self.db.get_btc5m_forecasts(round_slug=round_slug, valid_only=True)

        # Group valid forecasts by snapshot_id and condition
        fc_map: dict[str, dict[str, BTC5mAblationForecast]] = {}
        for fc in forecasts:
            if not fc.is_valid:
                continue
            if fc.snapshot_id not in fc_map:
                fc_map[fc.snapshot_id] = {}
            fc_map[fc.snapshot_id][fc.condition] = fc

        scored_records: list[dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc).isoformat()

        for snap in snapshots:
            # Primary market baseline (Requirement A1, A15)
            market_q = snap.market_q_primary
            market_brier = compute_brier(market_q, outcome_up) if market_q is not None else None
            market_log_loss = (
                compute_log_loss(market_q, outcome_up) if market_q is not None else None
            )

            # Secondary cross-outcome implied diagnostic
            market_q_implied = snap.market_q_implied_cross_outcome
            implied_brier = (
                compute_brier(market_q_implied, outcome_up) if market_q_implied is not None else None
            )
            implied_log_loss = (
                compute_log_loss(market_q_implied, outcome_up) if market_q_implied is not None else None
            )

            cond_fcs = fc_map.get(snap.snapshot_id, {})
            fc_a = cond_fcs.get(COND_A_REF_ONLY)
            fc_b = cond_fcs.get(COND_B_REF_PERP)
            fc_c = cond_fcs.get(COND_C_MARKET_AWARE)
            fc_d = cond_fcs.get(COND_D_FULL)

            score_record: dict[str, Any] = {
                "score_id": f"score_{snap.snapshot_id}",
                "round_slug": round_slug,
                "target_horizon_sec": snap.target_horizon_sec,
                "snapshot_id": snap.snapshot_id,
                "resolved_outcome": resolution.resolved_outcome,
                "resolution_price": resolution.resolution_price,
                "resolved_at": resolution.settled_at or now_utc,
                "scored_at": now_utc,
                "market_q": market_q,
                "market_brier": market_brier,
                "market_log_loss": market_log_loss,
                "market_q_implied": market_q_implied,
                "market_implied_brier": implied_brier,
                "market_implied_log_loss": implied_log_loss,
                "cond_a_prob": fc_a.jev_up_prob if fc_a else None,
                "cond_a_brier": compute_brier(fc_a.jev_up_prob, outcome_up) if fc_a else None,
                "cond_a_log_loss": compute_log_loss(fc_a.jev_up_prob, outcome_up) if fc_a else None,
                "cond_b_prob": fc_b.jev_up_prob if fc_b else None,
                "cond_b_brier": compute_brier(fc_b.jev_up_prob, outcome_up) if fc_b else None,
                "cond_b_log_loss": compute_log_loss(fc_b.jev_up_prob, outcome_up) if fc_b else None,
                "cond_c_prob": fc_c.jev_up_prob if fc_c else None,
                "cond_c_brier": compute_brier(fc_c.jev_up_prob, outcome_up) if fc_c else None,
                "cond_c_log_loss": compute_log_loss(fc_c.jev_up_prob, outcome_up) if fc_c else None,
                "cond_d_prob": fc_d.jev_up_prob if fc_d else None,
                "cond_d_brier": compute_brier(fc_d.jev_up_prob, outcome_up) if fc_d else None,
                "cond_d_log_loss": compute_log_loss(fc_d.jev_up_prob, outcome_up) if fc_d else None,
                "metadata": {
                    "price_to_beat": snap.price_to_beat,
                    "seconds_remaining": snap.seconds_remaining,
                    "market_q_primary_method": "NATIVE_UP_MIDPOINT",
                },
            }

            self.db.save_btc5m_resolution_score(score_record)
            scored_records.append(score_record)

        return scored_records

    def compute_evaluation_summary(
        self,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Compute aggregated calibration metrics, strictly paired deltas, and clustered bootstrap CI."""
        scores = self.db.get_btc5m_resolution_scores()
        if not scores:
            return {"status": "NO_RESOLVED_DATA", "total_scores": 0}

        rounds_dict: dict[str, list[dict[str, Any]]] = {}
        for s in scores:
            r_slug = s["round_slug"]
            rounds_dict.setdefault(r_slug, []).append(s)

        unique_rounds = list(rounds_dict.keys())
        n_rounds = len(unique_rounds)

        conditions = [COND_A_REF_ONLY, COND_B_REF_PERP, COND_C_MARKET_AWARE, COND_D_FULL]
        cond_keys = {
            COND_A_REF_ONLY: ("cond_a_prob", "cond_a_brier", "cond_a_log_loss"),
            COND_B_REF_PERP: ("cond_b_prob", "cond_b_brier", "cond_b_log_loss"),
            COND_C_MARKET_AWARE: ("cond_c_prob", "cond_c_brier", "cond_c_log_loss"),
            COND_D_FULL: ("cond_d_prob", "cond_d_brier", "cond_d_log_loss"),
        }

        summary: dict[str, Any] = {
            "total_scores": len(scores),
            "total_rounds": n_rounds,
            "metrics_by_condition": {},
            "paired_deltas": {},
            "metrics_by_horizon": {},
        }

        # Clustered bootstrap over rounds
        rng = random.Random(seed)
        boot_diffs: dict[str, list[float]] = {c: [] for c in conditions}

        for _ in range(n_boot):
            sampled_slugs = rng.choices(unique_rounds, k=n_rounds)
            sample_scores: list[dict[str, Any]] = []
            for slug in sampled_slugs:
                sample_scores.extend(rounds_dict[slug])

            for cond in conditions:
                _p_key, brier_key, _ = cond_keys[cond]
                paired_m: list[float] = []
                paired_c: list[float] = []
                for s in sample_scores:
                    mb = s.get("market_brier")
                    cb = s.get(brier_key)
                    if mb is not None and cb is not None:
                        paired_m.append(float(mb))
                        paired_c.append(float(cb))

                if paired_m and paired_c:
                    diff = sum(paired_c) / len(paired_c) - sum(paired_m) / len(paired_m)
                    boot_diffs[cond].append(diff)

        # Calculate metrics using strictly paired observations (Requirement A14, A15)
        for cond in conditions:
            _p_key, brier_key, log_loss_key = cond_keys[cond]

            paired_scores = [
                s for s in scores
                if s.get(brier_key) is not None and s.get("market_brier") is not None
            ]
            c_briers = [float(s[brier_key]) for s in paired_scores]
            m_briers = [float(s["market_brier"]) for s in paired_scores]
            c_losses = [float(s[log_loss_key]) for s in paired_scores if s.get(log_loss_key) is not None]
            m_losses = [float(s["market_log_loss"]) for s in paired_scores if s.get("market_log_loss") is not None]

            mean_brier = sum(c_briers) / len(c_briers) if c_briers else None
            mean_m_brier = sum(m_briers) / len(m_briers) if m_briers else None
            delta_brier = (mean_brier - mean_m_brier) if (mean_brier is not None and mean_m_brier is not None) else None

            mean_loss = sum(c_losses) / len(c_losses) if c_losses else None
            mean_m_loss = sum(m_losses) / len(m_losses) if m_losses else None
            delta_loss = (mean_loss - mean_m_loss) if (mean_loss is not None and mean_m_loss is not None) else None

            diffs = sorted(boot_diffs[cond])
            ci_lower = diffs[int(0.025 * len(diffs))] if diffs else None
            ci_upper = diffs[int(0.975 * len(diffs))] if diffs else None

            summary["metrics_by_condition"][cond] = {
                "paired_count": len(paired_scores),
                "mean_brier": round(mean_brier, 5) if mean_brier is not None else None,
                "market_brier": round(mean_m_brier, 5) if mean_m_brier is not None else None,
                "delta_brier": round(delta_brier, 5) if delta_brier is not None else None,
                "delta_brier_95ci": (round(ci_lower, 5), round(ci_upper, 5))
                if (ci_lower is not None and ci_upper is not None)
                else None,
                "mean_log_loss": round(mean_loss, 5) if mean_loss is not None else None,
                "market_log_loss": round(mean_m_loss, 5) if mean_m_loss is not None else None,
                "delta_log_loss": round(delta_loss, 5) if delta_loss is not None else None,
            }

        # Additional pairwise ablation deltas (Requirement B5)
        # RefPlusPerp - RefOnly
        pair_b_a = [s for s in scores if s.get("cond_b_brier") is not None and s.get("cond_a_brier") is not None]
        if pair_b_a:
            diff_brier = (sum(float(s["cond_b_brier"]) for s in pair_b_a) - sum(float(s["cond_a_brier"]) for s in pair_b_a)) / len(pair_b_a)
            summary["paired_deltas"]["REF_PLUS_PERP_MINUS_REF_ONLY"] = round(diff_brier, 5)

        # Full - MarketAware
        pair_d_c = [s for s in scores if s.get("cond_d_brier") is not None and s.get("cond_c_brier") is not None]
        if pair_d_c:
            diff_brier = (sum(float(s["cond_d_brier"]) for s in pair_d_c) - sum(float(s["cond_c_brier"]) for s in pair_d_c)) / len(pair_d_c)
            summary["paired_deltas"]["FULL_MINUS_MARKET_AWARE"] = round(diff_brier, 5)

        # MarketAware - RefOnly
        pair_c_a = [s for s in scores if s.get("cond_c_brier") is not None and s.get("cond_a_brier") is not None]
        if pair_c_a:
            diff_brier = (sum(float(s["cond_c_brier"]) for s in pair_c_a) - sum(float(s["cond_a_brier"]) for s in pair_c_a)) / len(pair_c_a)
            summary["paired_deltas"]["MARKET_AWARE_MINUS_REF_ONLY"] = round(diff_brier, 5)

        return summary
