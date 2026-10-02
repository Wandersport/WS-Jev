# Phase 8E Lead-Lag v3 Confirmatory Replication Measurement Integrity Audit

- **Date**: 2026-10-02 UTC
- **Status**: PASS
- **Experiment ID**: `btc5m_leadlag_v3_replication_2s`
- **Frozen Spec Hash**: `bfe5b0553a7143dd8941239dd481cf0e86499b54338dc6717565d1d9dc28fa53`
- **Frozen Model Hash**: `ee914a59072320d142944530957036c325632f941432d934f472e689dc7f1c77`
- **Frozen Horizon**: `+2s` ONLY
- **Outcome Blinding**: STRICTLY INTACT (0 predictive outcomes, 0 coefficients, 0 direction labels inspected)

```text
AUDIT_STATUS=PASS
TARGET_FULL_ROUNDS=250
COMPLETED_FULL_ROUNDS=250
ACTIVE_ROUNDS=0
TOTAL_ROUNDS_RECORDED=250
FIRST_ROUND_SLUG=btc-updown-5m-1790885400
LAST_ROUND_SLUG=btc-updown-5m-1790960100
START_TIME_UTC=2026-10-01T22:10:00Z
END_TIME_UTC=2026-10-02T19:00:00Z
TOTAL_SPAN_HOURS=20.83
INTENDED_TICKS=75000
CAPTURED_TICKS=75000
MISSED_TICKS=0
CAPTURE_RATIO=1.0000 (100.00%)
MEDIAN_INTERARRIVAL_MS=1000.0
P95_INTERARRIVAL_MS=1008.0
P99_INTERARRIVAL_MS=1010.0
MAX_INTERARRIVAL_MS=2164.0
SAMPLE_TARGET_ERR_P95_MS=10.0
WRITER_QUEUE_MAX=191
WRITER_QUEUE_FINAL=0
WRITER_LAG_P95_MS=7.03
COMMIT_LATENCY_P95_MS=4.95
BINANCE_TS_COV=1.0000 (100.00%)
POLY_TS_COV=0.99997 (99.997%)
TAKER_FLOW_60S_COV=1.0000 (100.00%)
STALE_RATE=0.0069 (0.69%)
PARSER_EXCEPTIONS=0
MALFORMED_EVENTS=0
RAW_ROUNDTRIP=PASS
CADENCE_DEGRADED=NO
FREE_DISK=36.90 GiB
PRAGMA_QUICK_CHECK=ok
PROCESS_ALIVE=NO
FILE_LOCK_HELD=NO
READY_FOR_FROZEN_ANALYSIS=YES
```

## Gate Verification Summary

| Gate | Requirement | Measured Value | Result |
| :--- | :--- | :--- | :--- |
| Exact Full Rounds | == 250 | 250 | **PASS** |
| Active / Partial Rounds | == 0 | 0 | **PASS** |
| Process & Lock Release | Exited, Lock Released | Process inactive, lock cleared | **PASS** |
| Duplicate Slugs | None | 0 duplicates across 250 rounds | **PASS** |
| Database Quick Check | `ok` | `ok` | **PASS** |
| Tick Capture Ratio | >= 95.0% | 100.00% (75,000 / 75,000) | **PASS** |
| Median Interarrival | ~1000 ms (950–1050 ms) | 1000.0 ms | **PASS** |
| p95 Interarrival | <= 1500 ms | 1008.0 ms | **PASS** |
| Cadence Degradation | None (early vs late <= 1.2x) | Early 1000.0 ms vs Late 1000.0 ms | **PASS** |
| Binance Timestamp Coverage | >= 99.0% | 100.00% | **PASS** |
| Polymarket Timestamp Coverage | >= 95.0% | 99.997% | **PASS** |
| Taker Flow 60s Coverage | >= 95.0% | 100.00% | **PASS** |
| Stale Rate | <= 5.0% | 0.69% | **PASS** |
| Parser Exceptions | == 0 | 0 | **PASS** |
| Malformed Events | == 0 | 0 | **PASS** |
| Writer Queue Bounded | < 2000 | Max 191, Final 0 | **PASS** |
| Lossless Raw Provenance | SHA256 Match | PASS (zlib decompress matches hash) | **PASS** |
| Free Disk Margin | >= 20.0 GiB | 36.90 GiB | **PASS** |

## Conclusion
The 250-round confirmatory data collection is complete and structurally sound. Every frozen measurement and architectural gate passed with zero defects. The dataset is fully frozen and quarantined in `data/pm_research_v3_replication.db`.
