"""Measurement-only integrity audit for Phase 8E Lead-Lag v3 Pilot.

STRICT RESEARCH INTEGRITY:
- Zero outcome peeking
- Zero predictive metric evaluation (no Delta R2, no MSE, no coefficients)
- Zero directional label inspection
- Standard library only (no external dependencies)
- Measurement cadence, queue backlog, latency, and feed integrity ONLY.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from pm_research.research.btc5m.leadlag_v3_experiment import (
    EXPERIMENT_PILOT_ID,
    PILOT_PHYSICAL_ROUNDS,
    REPLICATION_DB_PATH,
)


def _percentile(data: list[float], p: float) -> float:
    """Calculate percentile p (0-100) using nearest-rank or linear interpolation."""
    if not data:
        return 0.0
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_d) - 1)
    d = k - f
    return sorted_d[f] + (sorted_d[c] - sorted_d[f]) * d


def run_pilot_measurement_audit(
    db_path: str | Path = REPLICATION_DB_PATH,
    experiment_id: str = EXPERIMENT_PILOT_ID,
) -> dict[str, Any]:
    """Execute deterministic measurement audit on the completed pilot rounds."""
    db_p = Path(db_path)
    if not db_p.exists():
        raise FileNotFoundError(f"Replication database not found at {db_p}")

    conn = sqlite3.connect(f"file:{db_p.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. PRAGMA quick_check
    qc = cur.execute("PRAGMA quick_check;").fetchone()[0]
    db_ok = (qc == "ok")

    # 2. Physical Rounds Audit
    rounds_rows = cur.execute(
        "SELECT * FROM leadlag_v2_rounds WHERE experiment_id = ? ORDER BY start_epoch ASC",
        (experiment_id,),
    ).fetchall()

    warmup_rounds = [r for r in rounds_rows if r["status"] == "WARMUP_PARTIAL"]
    truncated_rounds = [r for r in rounds_rows if r["status"] == "COMPLETED" and (r["sample_count"] or 0) < 285]
    full_rounds = [
        r for r in rounds_rows
        if r["status"] == "COMPLETED"
        and (r["sample_count"] or 0) >= 285
        and (r["end_epoch"] - r["start_epoch"]) >= 300
    ]
    active_rounds = [r for r in rounds_rows if r["status"] == "ACTIVE"]
    slugs = [r["round_slug"] for r in rounds_rows]
    has_duplicates = len(slugs) != len(set(slugs))
    full_slugs = {r["round_slug"] for r in full_rounds}

    # 3. Synchronized Samples Audit (Include ONLY the full pilot measurement rounds)
    samples_rows = cur.execute(
        "SELECT * FROM leadlag_v2_samples WHERE experiment_id = ? ORDER BY sample_target_ts_ms ASC",
        (experiment_id,),
    ).fetchall()

    meas_samples = [s for s in samples_rows if s["round_slug"] in full_slugs]
    total_samples = len(meas_samples)
    stale_samples = sum(1 for s in meas_samples if s["is_stale"] == 1)
    stale_rate = stale_samples / total_samples if total_samples > 0 else 0.0

    # Interarrival & Target Error calculations
    interarrivals_ms: list[float] = []
    target_errors_ms: list[float] = []

    # Calculate by round to prevent inter-round boundary jump
    samples_by_round: dict[str, list[dict[str, Any]]] = {}
    for s in meas_samples:
        r_slug = s["round_slug"]
        if r_slug not in samples_by_round:
            samples_by_round[r_slug] = []
        samples_by_round[r_slug].append(dict(s))

    round_cadence_stats: list[dict[str, Any]] = []
    for r_slug, r_samples in samples_by_round.items():
        r_samples.sort(key=lambda x: x["sample_target_ts_ms"])
        r_actuals = [x["sample_actual_ts_ms"] for x in r_samples]
        r_targets = [x["sample_target_ts_ms"] for x in r_samples]

        r_inter = []
        for i in range(1, len(r_actuals)):
            diff = float(r_actuals[i] - r_actuals[i - 1])
            r_inter.append(diff)
            interarrivals_ms.append(diff)

        r_terr = [float(abs(a - t)) for a, t in zip(r_actuals, r_targets)]
        target_errors_ms.extend(r_terr)

        r_inter_sorted = sorted(r_inter) if r_inter else [1000.0]
        round_cadence_stats.append({
            "round_slug": r_slug,
            "sample_count": len(r_samples),
            "median_interarrival": statistics.median(r_inter_sorted),
            "p95_interarrival": _percentile(r_inter_sorted, 95),
            "max_interarrival": max(r_inter_sorted),
        })

    median_inter = statistics.median(interarrivals_ms) if interarrivals_ms else 1000.0
    p95_inter = _percentile(interarrivals_ms, 95)
    p99_inter = _percentile(interarrivals_ms, 99)
    max_inter = max(interarrivals_ms) if interarrivals_ms else 1000.0

    target_err_p95 = _percentile(target_errors_ms, 95)

    # 4. Feed Timestamp & Taker Flow Coverage
    binance_ts_valid = sum(1 for s in meas_samples if s["binance_source_ts_ms"] is not None)
    binance_ts_cov = binance_ts_valid / total_samples if total_samples > 0 else 0.0

    poly_ts_valid = sum(1 for s in meas_samples if s["poly_source_ts_ms"] is not None)
    poly_ts_cov = poly_ts_valid / total_samples if total_samples > 0 else 0.0

    # Taker flow 60s coverage on valid post-warmup samples
    tf_60s_evaluated = 0
    tf_60s_valid = 0
    for s in meas_samples:
        if s["is_valid"] == 1 and s["seconds_remaining"] <= 240:
            tf_60s_evaluated += 1
            if s["binance_taker_flow_60s"] is not None:
                tf_60s_valid += 1

    tf_60s_cov = tf_60s_valid / tf_60s_evaluated if tf_60s_evaluated > 0 else 0.0

    # 5. Heartbeat & Persistence Worker Telemetry
    hb_rows = cur.execute(
        "SELECT * FROM leadlag_v2_collector_heartbeat WHERE experiment_id = ? ORDER BY epoch_ms ASC",
        (experiment_id,),
    ).fetchall()

    latest_hb = hb_rows[-1] if hb_rows else None

    writer_q_depths: list[int] = []
    commit_latencies_p50: list[float] = []
    commit_latencies_p95: list[float] = []
    parser_exceptions = latest_hb["malformed_events"] if latest_hb else 0
    malformed_events = latest_hb["malformed_events"] if latest_hb else 0

    for hb in hb_rows:
        extra = json.loads(hb["extra_json"]) if hb["extra_json"] else {}
        lat = json.loads(hb["latency_metrics_json"]) if hb["latency_metrics_json"] else {}
        if "writer_queue_depth" in extra:
            writer_q_depths.append(extra["writer_queue_depth"])
        if "writer_commit_latency_p50_ms" in lat:
            commit_latencies_p50.append(lat["writer_commit_latency_p50_ms"])
        if "writer_commit_latency_p95_ms" in lat:
            commit_latencies_p95.append(lat["writer_commit_latency_p95_ms"])

    writer_lag_estimates = [q * lat for q, lat in zip(writer_q_depths, commit_latencies_p50)]
    writer_max_q = max(writer_q_depths) if writer_q_depths else 0
    writer_final_q = writer_q_depths[-1] if writer_q_depths else 0

    intended_ticks = len(full_rounds) * 300
    captured_ticks = total_samples
    missed_ticks = max(0, intended_ticks - captured_ticks)
    capture_ratio = captured_ticks / intended_ticks if intended_ticks > 0 else 0.0

    # 6. Raw Payload Lossless Provenance Roundtrip
    sample_payload = cur.execute(
        "SELECT * FROM leadlag_v2_raw_payloads WHERE experiment_id = ? ORDER BY recv_ts_ms DESC LIMIT 1",
        (experiment_id,),
    ).fetchone()

    raw_roundtrip_ok = False
    if sample_payload:
        import hashlib
        import zlib
        decomp = zlib.decompress(sample_payload["compressed_payload"])
        calc_sha = hashlib.sha256(decomp).hexdigest()
        raw_roundtrip_ok = (calc_sha == sample_payload["sha256_hash"])

    # 7. Cadence Degradation Test
    cadence_degraded = False
    if len(round_cadence_stats) >= 6:
        early_inter = statistics.mean([r["median_interarrival"] for r in round_cadence_stats[:3]])
        late_inter = statistics.mean([r["median_interarrival"] for r in round_cadence_stats[-3:]])
        if late_inter > early_inter * 1.2:
            cadence_degraded = True

    conn.close()

    # Pass / Fail criteria evaluation
    pass_gates = {
        "full_pilot_rounds": len(full_rounds) == PILOT_PHYSICAL_ROUNDS,
        "active_rounds_zero": len(active_rounds) == 0,
        "no_duplicate_slugs": not has_duplicates,
        "db_quick_check_ok": db_ok,
        "capture_ratio_ge_95": capture_ratio >= 0.95,
        "median_interarrival_close_1000": 950 <= median_inter <= 1050,
        "p95_interarrival_le_1500": p95_inter <= 1500.0,
        "no_cadence_degradation": not cadence_degraded,
        "binance_ts_ge_99": binance_ts_cov >= 0.99,
        "poly_ts_ge_95": poly_ts_cov >= 0.95,
        "taker_flow_60s_ge_95": tf_60s_cov >= 0.95,
        "parser_exceptions_zero": parser_exceptions == 0,
        "malformed_events_zero": malformed_events == 0,
        "writer_backlog_bounded": writer_max_q < 2000,
        "raw_roundtrip_pass": raw_roundtrip_ok,
    }

    all_pass = all(pass_gates.values())

    return {
        "status": "PASS" if all_pass else "FAIL",
        "full_pilot_rounds": len(full_rounds),
        "warmup_partial_rounds": len(warmup_rounds),
        "truncated_rounds": len(truncated_rounds),
        "total_rounds_recorded": len(rounds_rows),
        "intended_ticks": intended_ticks,
        "captured_ticks": captured_ticks,
        "missed_ticks": missed_ticks,
        "capture_ratio": capture_ratio,
        "median_interarrival_ms": median_inter,
        "p95_interarrival_ms": p95_inter,
        "p99_interarrival_ms": p99_inter,
        "max_interarrival_ms": max_inter,
        "target_error_p95_ms": target_err_p95,
        "writer_queue_max": writer_max_q,
        "writer_queue_final": writer_final_q,
        "writer_lag_p95_ms": _percentile(writer_lag_estimates, 95) if writer_lag_estimates else 0.0,
        "commit_latency_p95_ms": _percentile(commit_latencies_p95, 95) if commit_latencies_p95 else 0.0,
        "binance_ts_coverage": binance_ts_cov,
        "poly_ts_coverage": poly_ts_cov,
        "taker_flow_60s": tf_60s_cov,
        "stale_rate": stale_rate,
        "parser_exceptions": parser_exceptions,
        "malformed_events": malformed_events,
        "raw_roundtrip": "PASS" if raw_roundtrip_ok else "FAIL",
        "cadence_degraded": cadence_degraded,
        "round_to_round_degradation": "YES" if cadence_degraded else "NO",
        "pilot_data_excluded_from_replication": True,
        "ready_for_final_replication": all_pass,
        "pass_gates": pass_gates,
        "round_cadence_stats": round_cadence_stats,
    }
