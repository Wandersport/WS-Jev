"""Measurement-only integrity audit for Phase 8E Lead-Lag v3 250-Round Confirmatory Replication.

STRICT RESEARCH INTEGRITY:
- Zero outcome peeking
- Zero predictive metric evaluation (no Delta R2, no MSE, no coefficients)
- Zero directional label inspection
- Standard library only
- Cadence, queue backlog, latency, and feed integrity ONLY.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from pm_research.research.btc5m.leadlag_v3_experiment import (
    EXPERIMENT_ID,
    REPLICATION_DB_PATH,
    TARGET_PHYSICAL_ROUNDS,
)


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_d) - 1)
    d = k - f
    return sorted_d[f] + (sorted_d[c] - sorted_d[f]) * d


def run_confirmatory_measurement_audit(
    db_path: str | Path = REPLICATION_DB_PATH,
    experiment_id: str = EXPERIMENT_ID,
) -> dict[str, Any]:
    db_p = Path(db_path)
    if not db_p.exists():
        raise FileNotFoundError(f"Database not found at {db_p}")

    # Process and lock checks
    pid_file = Path("data/btc5m_leadlag_v3_replication.pid")
    pid_alive = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            pid_alive = True
        except OSError:
            pid_alive = False

    lock_file = Path("data/btc5m_leadlag_v3_replication.lock")
    lock_held = False
    if lock_file.exists():
        import fcntl

        try:
            fd = os.open(str(lock_file), os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                lock_held = False
            except (BlockingIOError, OSError):
                lock_held = True
            finally:
                os.close(fd)
        except Exception:
            lock_held = False

    # Read-only DB connection
    conn = sqlite3.connect(f"file:{db_p.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Quick check
    qc = cur.execute("PRAGMA quick_check;").fetchone()[0]

    # 2. Rounds
    rounds_rows = cur.execute(
        "SELECT * FROM leadlag_v2_rounds WHERE experiment_id = ? ORDER BY start_epoch ASC",
        (experiment_id,),
    ).fetchall()

    completed_rounds = [
        r
        for r in rounds_rows
        if r["status"] == "COMPLETED"
        and (r["sample_count"] or 0) >= 285
        and (r["end_epoch"] - r["start_epoch"]) >= 300
    ]
    active_rounds = [r for r in rounds_rows if r["status"] == "ACTIVE"]
    slugs = [r["round_slug"] for r in rounds_rows]
    has_duplicates = len(slugs) != len(set(slugs))
    completed_slugs = {r["round_slug"] for r in completed_rounds}

    # 3. Synchronized samples
    samples_rows = cur.execute(
        """
        SELECT sample_target_ts_ms, sample_actual_ts_ms, round_slug,
               binance_source_ts_ms, poly_source_ts_ms, binance_taker_flow_60s,
               is_valid, is_stale, seconds_remaining
        FROM leadlag_v2_samples
        WHERE experiment_id = ?
        ORDER BY sample_target_ts_ms ASC
        """,
        (experiment_id,),
    ).fetchall()

    meas_samples = [s for s in samples_rows if s["round_slug"] in completed_slugs]
    total_samples = len(meas_samples)
    intended_ticks = len(completed_rounds) * 300
    missed_ticks = max(0, intended_ticks - total_samples)
    capture_ratio = total_samples / intended_ticks if intended_ticks > 0 else 0.0

    stale_samples = sum(1 for s in meas_samples if s["is_stale"] == 1)
    stale_rate = stale_samples / total_samples if total_samples > 0 else 0.0

    # Group by round for interarrival and target error calculations
    interarrivals_ms: list[float] = []
    target_errors_ms: list[float] = []
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

        for a, t in zip(r_actuals, r_targets):
            target_errors_ms.append(float(abs(a - t)))

        r_inter_sorted = sorted(r_inter) if r_inter else [1000.0]
        round_cadence_stats.append({
            "round_slug": r_slug,
            "sample_count": len(r_samples),
            "median_inter": statistics.median(r_inter_sorted),
        })

    sorted_inter = sorted(interarrivals_ms)
    median_inter = statistics.median(sorted_inter) if sorted_inter else 1000.0
    p95_inter = _percentile(sorted_inter, 95)
    p99_inter = _percentile(sorted_inter, 99)
    max_inter = max(sorted_inter) if sorted_inter else 1000.0
    target_err_p95 = _percentile(target_errors_ms, 95)

    # 4. Coverage metrics
    bn_valid = sum(1 for s in meas_samples if s["binance_source_ts_ms"] is not None)
    bn_cov = bn_valid / total_samples if total_samples > 0 else 0.0

    poly_valid = sum(1 for s in meas_samples if s["poly_source_ts_ms"] is not None)
    poly_cov = poly_valid / total_samples if total_samples > 0 else 0.0

    tf_valid = 0
    tf_total = 0
    for s in meas_samples:
        if s["is_valid"] == 1 and s["seconds_remaining"] <= 240:
            tf_total += 1
            if s["binance_taker_flow_60s"] is not None:
                tf_valid += 1
    tf_cov = tf_valid / tf_total if tf_total > 0 else 0.0

    # 5. Heartbeat & Persistence Latencies
    hb_rows = cur.execute(
        "SELECT * FROM leadlag_v2_collector_heartbeat WHERE experiment_id = ? ORDER BY epoch_ms ASC",
        (experiment_id,),
    ).fetchall()

    writer_q_depths: list[int] = []
    commit_latencies_p50: list[float] = []
    commit_latencies_p95: list[float] = []
    latest_hb = hb_rows[-1] if hb_rows else None
    extra_latest = json.loads(latest_hb["extra_json"]) if latest_hb and latest_hb["extra_json"] else {}

    for hb in hb_rows:
        e = json.loads(hb["extra_json"]) if hb["extra_json"] else {}
        lat = json.loads(hb["latency_metrics_json"]) if hb["latency_metrics_json"] else {}
        if "writer_queue_depth" in e:
            writer_q_depths.append(e["writer_queue_depth"])
        if "writer_commit_latency_p50_ms" in lat:
            commit_latencies_p50.append(lat["writer_commit_latency_p50_ms"])
        if "writer_commit_latency_p95_ms" in lat:
            commit_latencies_p95.append(lat["writer_commit_latency_p95_ms"])

    writer_max_q = max(writer_q_depths) if writer_q_depths else 0
    writer_final_q = writer_q_depths[-1] if writer_q_depths else 0
    writer_lag_estimates = [q * lat for q, lat in zip(writer_q_depths, commit_latencies_p50)]
    writer_lag_p95 = _percentile(writer_lag_estimates, 95) if writer_lag_estimates else 0.0
    commit_lat_p95 = _percentile(commit_latencies_p95, 95) if commit_latencies_p95 else 0.0

    parser_exceptions = extra_latest.get("parser_exceptions", 0)
    malformed_events = extra_latest.get("malformed_events", 0)

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
        raw_roundtrip_ok = calc_sha == sample_payload["sha256_hash"]

    # 7. Cadence Degradation Test
    cadence_degraded = False
    if len(round_cadence_stats) >= 10:
        early_inter = statistics.mean([r["median_inter"] for r in round_cadence_stats[:10]])
        late_inter = statistics.mean([r["median_inter"] for r in round_cadence_stats[-10:]])
        if late_inter > early_inter * 1.2:
            cadence_degraded = True

    # 8. Disk Space
    free_bytes = shutil.disk_usage(".").free
    free_gib = free_bytes / (1024**3)

    conn.close()

    # Pass Gates
    pass_gates = {
        "completed_rounds_exact_250": len(completed_rounds) == TARGET_PHYSICAL_ROUNDS,
        "active_rounds_zero": len(active_rounds) == 0,
        "process_alive_false": not pid_alive,
        "lock_held_false": not lock_held,
        "no_duplicate_slugs": not has_duplicates,
        "db_quick_check_ok": qc == "ok",
        "capture_ratio_ge_95": capture_ratio >= 0.95,
        "median_interarrival_close_1000": 950.0 <= median_inter <= 1050.0,
        "p95_interarrival_le_1500": p95_inter <= 1500.0,
        "no_cadence_degradation": not cadence_degraded,
        "binance_ts_ge_99": bn_cov >= 0.99,
        "poly_ts_ge_95": poly_cov >= 0.95,
        "taker_flow_60s_ge_95": tf_cov >= 0.95,
        "stale_rate_le_05": stale_rate <= 0.05,
        "parser_exceptions_zero": parser_exceptions == 0,
        "malformed_events_zero": malformed_events == 0,
        "writer_backlog_bounded": writer_max_q < 2000,
        "raw_roundtrip_pass": raw_roundtrip_ok,
        "free_disk_ge_20gib": free_gib >= 20.0,
    }

    all_pass = all(pass_gates.values())

    return {
        "status": "PASS" if all_pass else "FAIL",
        "all_pass": all_pass,
        "pass_gates": pass_gates,
        "completed_full_rounds": len(completed_rounds),
        "active_rounds": len(active_rounds),
        "total_rounds_recorded": len(rounds_rows),
        "process_alive": pid_alive,
        "file_lock_held": lock_held,
        "pragma_quick_check": qc,
        "intended_ticks": intended_ticks,
        "captured_ticks": total_samples,
        "missed_ticks": missed_ticks,
        "capture_ratio": capture_ratio,
        "median_interarrival_ms": median_inter,
        "p95_interarrival_ms": p95_inter,
        "p99_interarrival_ms": p99_inter,
        "max_interarrival_ms": max_inter,
        "sample_target_err_p95_ms": target_err_p95,
        "writer_queue_max": writer_max_q,
        "writer_queue_final": writer_final_q,
        "writer_lag_p95_ms": writer_lag_p95,
        "commit_latency_p95_ms": commit_lat_p95,
        "binance_ts_coverage": bn_cov,
        "poly_ts_coverage": poly_cov,
        "taker_flow_60s_coverage": tf_cov,
        "stale_rate": stale_rate,
        "parser_exceptions": parser_exceptions,
        "malformed_events": malformed_events,
        "raw_roundtrip": "PASS" if raw_roundtrip_ok else "FAIL",
        "cadence_degraded": cadence_degraded,
        "free_disk_gib": free_gib,
    }


if __name__ == "__main__":
    res = run_confirmatory_measurement_audit()
    print("=== CONFIRMATORY REPLICATION MEASUREMENT INTEGRITY AUDIT ===")
    print(f"AUDIT_STATUS={res['status']}")
    for k, v in res.items():
        if k != "pass_gates":
            print(f"{k}: {v}")
    print("\nPASS GATES:")
    for k, v in res["pass_gates"].items():
        print(f"  {k}: {v}")
