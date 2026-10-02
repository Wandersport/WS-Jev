"""Comprehensive final measurement and 2s-pair eligibility audit for Phase 8E Lead-Lag v3 replication.

STRICT RESEARCH INTEGRITY:
- Outcome-blind: zero predictive metrics, zero Delta R2, zero coefficients, zero direction accuracy.
- Counts, timing error, latencies, and feed integrity ONLY.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
import zlib
from pathlib import Path
from typing import Any

from pm_research.research.btc5m.leadlag_v3_experiment import (
    EXPERIMENT_ID,
    REPLICATION_DB_PATH,
)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return s[f] + (s[c] - s[f]) * d


def audit_final_replication() -> dict[str, Any]:
    db_p = Path(REPLICATION_DB_PATH)
    conn = sqlite3.connect(f"file:{db_p.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Rounds
    rounds = cur.execute(
        "SELECT * FROM leadlag_v2_rounds WHERE experiment_id = ? ORDER BY start_epoch ASC",
        (EXPERIMENT_ID,),
    ).fetchall()

    completed_rounds = [r for r in rounds if r["status"] == "COMPLETED"]
    active_rounds = [r for r in rounds if r["status"] == "ACTIVE"]
    sample_counts = [r["sample_count"] for r in completed_rounds]

    rounds_300 = sum(1 for c in sample_counts if c == 300)
    rounds_lt_300 = sum(1 for c in sample_counts if c < 300)

    # 2. Samples
    samples = cur.execute(
        """
        SELECT sample_id, sample_target_ts_ms, sample_actual_ts_ms, round_slug,
               binance_source_ts_ms, poly_source_ts_ms, binance_recv_ts_ms, poly_recv_ts_ms,
               binance_taker_flow_60s, is_valid, is_stale, seconds_remaining,
               poly_midpoint, poly_best_bid, poly_best_ask, poly_is_crossed,
               binance_best_bid, binance_best_ask, binance_is_valid, poly_is_valid
        FROM leadlag_v2_samples
        WHERE experiment_id = ?
        ORDER BY sample_target_ts_ms ASC
        """,
        (EXPERIMENT_ID,),
    ).fetchall()

    total_samples = len(samples)
    intended_ticks = len(completed_rounds) * 300
    capture_ratio = total_samples / intended_ticks if intended_ticks > 0 else 0.0

    valid_samples_count = sum(1 for s in samples if s["is_valid"] == 1)
    invalid_samples_count = sum(1 for s in samples if s["is_valid"] == 0)

    # Interarrival and target errors
    interarrivals: list[float] = []
    target_errors: list[float] = []
    samples_by_round: dict[str, list[dict[str, Any]]] = {}

    for s in samples:
        rd = s["round_slug"]
        if rd not in samples_by_round:
            samples_by_round[rd] = []
        samples_by_round[rd].append(dict(s))

    round_medians: list[float] = []
    for rd, r_samples in samples_by_round.items():
        r_samples.sort(key=lambda x: x["sample_target_ts_ms"])
        r_actuals = [x["sample_actual_ts_ms"] for x in r_samples]
        r_targets = [x["sample_target_ts_ms"] for x in r_samples]
        r_inter = []
        for i in range(1, len(r_actuals)):
            diff = float(r_actuals[i] - r_actuals[i - 1])
            r_inter.append(diff)
            interarrivals.append(diff)
        for a, t in zip(r_actuals, r_targets):
            target_errors.append(float(abs(a - t)))
        round_medians.append(statistics.median(r_inter) if r_inter else 1000.0)

    inter_p50 = statistics.median(interarrivals) if interarrivals else 1000.0
    inter_p95 = _percentile(interarrivals, 95)
    inter_p99 = _percentile(interarrivals, 99)
    inter_max = max(interarrivals) if interarrivals else 1000.0

    target_err_p50 = _percentile(target_errors, 50)
    target_err_p95 = _percentile(target_errors, 95)
    target_err_p99 = _percentile(target_errors, 99)

    # Feed coverage
    bn_valid = sum(1 for s in samples if s["binance_source_ts_ms"] is not None)
    bn_cov = bn_valid / total_samples if total_samples > 0 else 0.0

    poly_valid = sum(1 for s in samples if s["poly_source_ts_ms"] is not None)
    poly_cov = poly_valid / total_samples if total_samples > 0 else 0.0

    tf_valid = 0
    tf_total = 0
    for s in samples:
        if s["is_valid"] == 1 and s["seconds_remaining"] <= 240:
            tf_total += 1
            if s["binance_taker_flow_60s"] is not None:
                tf_valid += 1
    tf_cov = tf_valid / tf_total if tf_total > 0 else 0.0

    stale_cnt = sum(1 for s in samples if s["is_stale"] == 1)
    stale_rate = stale_cnt / total_samples if total_samples > 0 else 0.0

    # Heartbeats & Latencies
    hb_rows = cur.execute(
        "SELECT extra_json, latency_metrics_json FROM leadlag_v2_collector_heartbeat WHERE experiment_id = ? ORDER BY rowid ASC",
        (EXPERIMENT_ID,),
    ).fetchall()

    q_depths: list[int] = []
    commit_p50: list[float] = []
    commit_p95: list[float] = []
    commit_p99: list[float] = []
    for hb in hb_rows:
        e = json.loads(hb["extra_json"]) if hb["extra_json"] else {}
        lat = json.loads(hb["latency_metrics_json"]) if hb["latency_metrics_json"] else {}
        if "writer_queue_depth" in e:
            q_depths.append(e["writer_queue_depth"])
        if "writer_commit_latency_p50_ms" in lat:
            commit_p50.append(lat["writer_commit_latency_p50_ms"])
        if "writer_commit_latency_p95_ms" in lat:
            commit_p95.append(lat["writer_commit_latency_p95_ms"])
        if "writer_commit_latency_p99_ms" in lat:
            commit_p99.append(lat["writer_commit_latency_p99_ms"])

    q_max = max(q_depths) if q_depths else 0
    writer_lags = [q * lat for q, lat in zip(q_depths, commit_p50)]
    w_lag_p95 = _percentile(writer_lags, 95)
    w_lag_p99 = _percentile(writer_lags, 99)
    c_lat_p95 = _percentile(commit_p95, 95)
    c_lat_p99 = _percentile(commit_p99, 99)

    latest_lat = json.loads(hb_rows[-1]["latency_metrics_json"]) if hb_rows else {}
    parser_exceptions = latest_lat.get("parser_exceptions", 0)
    malformed_events = latest_lat.get("malformed_events", 0)

    # Raw roundtrip
    sample_payload = cur.execute(
        "SELECT compressed_payload, sha256_hash FROM leadlag_v2_raw_payloads WHERE experiment_id = ? ORDER BY rowid DESC LIMIT 1",
        (EXPERIMENT_ID,),
    ).fetchone()
    raw_roundtrip_ok = False
    if sample_payload:
        dec = zlib.decompress(sample_payload["compressed_payload"])
        raw_roundtrip_ok = hashlib.sha256(dec).hexdigest() == sample_payload["sha256_hash"]

    # Cadence degradation
    early_10 = statistics.mean(round_medians[:10]) if len(round_medians) >= 10 else 1000.0
    late_10 = statistics.mean(round_medians[-10:]) if len(round_medians) >= 10 else 1000.0
    cadence_degraded = late_10 > early_10 * 1.2

    # 3. 2-Second Pair Eligibility (Frozen Exact Grid)
    # Target horizon L = 2000 ms
    # abs(actual_elapsed_ms - 2000) <= 500 ms
    sample_map: dict[tuple[str, int], dict[str, Any]] = {
        (s["round_slug"], s["sample_target_ts_ms"]): s for s in samples
    }

    # Candidate samples: all valid samples with seconds_remaining >= 2 (so a +2s target exists in the round)
    candidate_samples = [s for s in samples if s["is_valid"] == 1 and s["seconds_remaining"] >= 2]
    total_candidate_count = len(candidate_samples)

    eligible_pairs = []
    contributing_rounds = set()
    lag_timing_errors: list[float] = []
    future_info_events = 0

    for s in candidate_samples:
        rd = s["round_slug"]
        t = s["sample_target_ts_ms"]
        future_t = t + 2000
        future_s = sample_map.get((rd, future_t))

        if future_s and future_s["poly_midpoint"] is not None and s["poly_midpoint"] is not None:
            actual_curr = s["sample_actual_ts_ms"]
            actual_future = future_s["sample_actual_ts_ms"]
            elapsed_ms = actual_future - actual_curr

            # Future information check:
            # 1. Does future happen after current? (elapsed_ms must be > 0)
            if elapsed_ms <= 0:
                future_info_events += 1

            # 2. Did s observe any feed timestamp after actual_curr?
            if s["binance_source_ts_ms"] and s["binance_source_ts_ms"] > actual_curr + 50:
                future_info_events += 1
            if s["poly_source_ts_ms"] and s["poly_source_ts_ms"] > actual_curr + 50:
                future_info_events += 1

            timing_error = abs(elapsed_ms - 2000.0)
            if timing_error <= 500.0:
                eligible_pairs.append({
                    "round_slug": rd,
                    "target_ts_ms": t,
                    "elapsed_ms": elapsed_ms,
                    "timing_error": timing_error,
                })
                contributing_rounds.add(rd)
                lag_timing_errors.append(timing_error)

    pair_retention_rate = len(eligible_pairs) / total_candidate_count if total_candidate_count > 0 else 0.0
    timing_err_p50 = _percentile(lag_timing_errors, 50)
    timing_err_p95 = _percentile(lag_timing_errors, 95)
    timing_err_p99 = _percentile(lag_timing_errors, 99)

    conn.close()

    return {
        "total_full_rounds": len(completed_rounds),
        "active_rounds": len(active_rounds),
        "total_samples": total_samples,
        "intended_ticks": intended_ticks,
        "capture_ratio": capture_ratio,
        "rounds_300": rounds_300,
        "rounds_lt_300": rounds_lt_300,
        "per_round_min": min(sample_counts) if sample_counts else 0,
        "per_round_median": statistics.median(sample_counts) if sample_counts else 0,
        "per_round_max": max(sample_counts) if sample_counts else 0,
        "valid_samples_count": valid_samples_count,
        "invalid_samples_count": invalid_samples_count,
        "binance_ts_coverage": bn_cov,
        "poly_ts_coverage": poly_cov,
        "taker_flow_60s_coverage": tf_cov,
        "stale_rate": stale_rate,
        "interarrival_p50": inter_p50,
        "interarrival_p95": inter_p95,
        "interarrival_p99": inter_p99,
        "interarrival_max": inter_max,
        "target_err_p50": target_err_p50,
        "target_err_p95": target_err_p95,
        "target_err_p99": target_err_p99,
        "parser_exceptions": parser_exceptions,
        "malformed_events": malformed_events,
        "writer_queue_max": q_max,
        "writer_lag_p95": w_lag_p95,
        "writer_lag_p99": w_lag_p99,
        "commit_latency_p95": c_lat_p95,
        "commit_latency_p99": c_lat_p99,
        "raw_roundtrip": "PASS" if raw_roundtrip_ok else "FAIL",
        "cadence_degraded": cadence_degraded,
        "candidate_samples": total_candidate_count,
        "eligible_2s_pairs": len(eligible_pairs),
        "contributing_rounds_2s": len(contributing_rounds),
        "pair_retention_rate": pair_retention_rate,
        "timing_err_p50": timing_err_p50,
        "timing_err_p95": timing_err_p95,
        "timing_err_p99": timing_err_p99,
        "future_info_events": future_info_events,
    }


if __name__ == "__main__":
    res = audit_final_replication()
    print("=== FINAL REPLICATION COMPREHENSIVE MEASUREMENT AUDIT ===")
    for k, v in res.items():
        print(f"{k}: {v}")
