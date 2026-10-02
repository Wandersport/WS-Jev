# Phase 8E Lead-Lag v3 Confirmatory Replication Measurement Integrity Audit

- **Date**: 2026-10-02 UTC
- **Status**: PASS
- **Experiment ID**: `btc5m_leadlag_v3_replication_2s`
- **Frozen Spec Hash**: `bfe5b0553a7143dd8941239dd481cf0e86499b54338dc6717565d1d9dc28fa53`
- **Frozen Model Hash**: `ee914a59072320d142944530957036c325632f941432d934f472e689dc7f1c77`
- **Frozen Horizon**: `+2s` ONLY
- **Outcome Blinding**: STRICTLY INTACT (0 predictive outcomes, 0 coefficients, 0 direction labels inspected)

```text
FINAL_COLLECTION_STATUS=PASS
NORMAL_TARGET_COMPLETION=YES
TOTAL_FULL_ROUNDS=250
ACTIVE_ROUNDS=0
TOTAL_SAMPLES=75000
EXPECTED_SAMPLES=75000
CAPTURE_RATIO=1.0000 (100.00%)
ROUNDS_300_OF_300=250
ROUNDS_LT_300=0
DB_QUICK_CHECK=ok
VALID_SAMPLES_STATUS_FIELD=0 (telemetry accounting artifact)
ACTUAL_ELIGIBLE_SAMPLE_COUNT=65687
INVALID_SAMPLE_COUNT=9313
VALID_SAMPLES_ZERO_ROOT_CAUSE=In leadlag_v3_collector.py line 782/815, telemetry recorded self._current_round_valid_count instead of a cumulative counter. At round 250 completion, _close_current_round() reset this counter to 0 immediately before final telemetry emission. Persisted database contains exactly 65,687 valid samples (is_valid=1).
BINANCE_TS_COVERAGE=1.0000 (100.00%)
POLY_TS_COVERAGE=0.99997 (99.997%)
TAKER_FLOW_60S=1.0000 (100.00%)
STALE_RATE=0.0069 (0.69%)
MEDIAN_INTERARRIVAL_MS=1000.0
P95_INTERARRIVAL_MS=1008.0
P99_INTERARRIVAL_MS=1010.0
PARSER_EXCEPTIONS=0
MALFORMED_EVENTS=0
WRITER_QUEUE_MAX=191
CADENCE_DEGRADATION=NO
RAW_ROUNDTRIP=PASS
ELIGIBLE_2S_PAIRS=65436
CONTRIBUTING_ROUNDS_2S=250
P95_2S_TIMING_ERROR_MS=9.0
FUTURE_INFORMATION_EVENTS=0
OUTCOME_BLINDING_INTACT=YES
READY_FOR_FINAL_REPLICATION_ANALYSIS=YES
```

---

## 1. Normal Completion Verification
- **Target Completion**: The collector cleanly executed all 250 physical rounds and terminated on schedule:
  `2026-10-02 19:00:00,002 [INFO] Enqueued completion for full round btc-updown-5m-1790960100: 245/300 valid samples. Completed full rounds: 250/250.`
  `2026-10-02 19:00:00,002 [INFO] Target of 250 full rounds reached! Exiting loop.`
- **Durable Persistence Drain**: Write queue drained cleanly (`Persistence worker exited cleanly`) at depth 0.
- **Process & Resource Release**: PID file `data/btc5m_leadlag_v3_replication.pid` and lock file `data/btc5m_leadlag_v3_replication.lock` removed; process and caffeinate terminated.
- **Round Inventory**: Exactly 250 finalized rounds with status `COMPLETED`, each lasting exactly 300 seconds. 0 `ACTIVE` or non-finalized rounds. Zero extra target+1 rounds captured.

---

## 2. Database Integrity & Hash Verification
- **PRAGMA quick_check**: Passed with status `ok` across all 36.9 GiB.
- **Sample Count**: Exactly 75,000 synchronized sample rows for `btc5m_leadlag_v3_replication_2s`.
- **Round Slugs**: Exactly 250 unique round slugs; 0 duplicate slugs.
- **Target Timestamp Uniqueness**: 0 duplicate target timestamps within any round.
- **Pilot Experiment Isolation**: Pilot dataset (`btc5m_leadlag_v3_replication_2s_pilot`, 12 rounds, 3,122 samples) strictly isolated with `is_pilot=1`; replication dataset strictly tagged with `is_pilot=0`.
- **Frozen Hashes**:
  - `docs/BTC5M_LEADLAG_REPLICATION_2S_SPEC.md` SHA256: `bfe5b0553a7143dd8941239dd481cf0e86499b54338dc6717565d1d9dc28fa53` (**MATCH**)
  - `models/frozen_v2_replication_models_2s.json` SHA256: `ee914a59072320d142944530957036c325632f941432d934f472e689dc7f1c77` (**MATCH**)

---

## 3. Validity Forensics
- **Root Cause of `Valid Samples: 0` in Heartbeat**:
  - In `src/pm_research/research/btc5m/leadlag_v3_collector.py`, lines 782 and 815:
    `"valid_samples": self._current_round_valid_count`
  - While `total_samples` tracked the cumulative run total (`self._captured_ticks_count = 75,000`), `valid_samples` mistakenly referenced `self._current_round_valid_count`, which is the *per-round* active accumulator.
  - When round 250 finished, `_close_current_round()` executed and reset `self._current_round_valid_count = 0` at line 1150.
  - The `finally` block in `run()` (line 1281) then called `self._emit_heartbeat()`, logging `0` to telemetry.
- **Persisted Validity in Database**:
  - In `leadlag_v2_samples`: `is_valid` is persisted directly for every row. Exactly **65,687** samples have `is_valid = 1` (87.58%), and **9,313** samples have `is_valid = 0` (12.42%).
  - In `leadlag_v2_rounds`: `valid_sample_count` is persisted directly for every round, summing to exactly 65,687 across all 250 rounds.
- **Invalid Samples Breakdown (9,313 total)**:
  - **Polymarket Zero Bid** (`poly_best_bid == 0.0`): **8,165** samples (87.7%). Occurs during terminal round resolution when the binary token probability approaches 0 or 1. `poly_is_valid = bool(poly_bid and poly_ask ...)` evaluates `bool(0.0) == False`.
  - **Polymarket Initial Book Formation** (`poly_best_bid IS NULL`): **1,119** samples (12.0%). Occurs during the first 1–5 seconds of round startup while the WebSocket awaits the initial delta/snapshot for the new token.
  - **Both Binance & Polymarket Invalid**: **18** samples (0.19%).
  - **Binance Invalid Only**: **10** samples (0.11%).
  - **Polymarket Stale Feed Only** (`receipt_age > 1500ms`): **1** sample (0.01%).
  - **Crossed Book**: **0** samples (0.00%). Zero crossed books occurred.

---

## 4. Final Measurement Quality
- **Intended Ticks**: 75,000
- **Captured Ticks**: 75,000 (Capture Ratio: 100.00%, 0 missed ticks)
- **Per-Round Sample Count**: Min: 300, Median: 300, Max: 300 (All 250 rounds captured exactly 300/300)
- **Interarrival Cadence**: Median: 1000.0 ms, p95: 1008.0 ms, p99: 1010.0 ms, Max: 2164.0 ms
- **Target Timing Error**: Median: 5.0 ms, p95: 10.0 ms, p99: 10.0 ms
- **Binance Timestamp Coverage**: 100.00% (75,000 / 75,000)
- **Polymarket Timestamp Coverage**: 99.997% (74,998 / 75,000)
- **Taker Flow 60s Coverage**: 100.00% (post-warmup valid samples)
- **Stale Rate**: 0.69% (519 / 75,000, well below 5.0% gate)
- **Parser Exceptions**: 0
- **Malformed Events**: 0
- **Persistence Latency & Queuing**: Writer Queue Max: 191 (Final: 0), Writer Lag p95: 7.03 ms, Commit Latency p95: 4.95 ms
- **Lossless Provenance Roundtrip**: PASS (decompressed payload SHA256 matches persisted hash)
- **Cadence Degradation**: NO (Early 10 rounds median: 1000.0 ms; Late 10 rounds median: 1000.0 ms)

---

## 5. 2-Second Pair Eligibility (Frozen Exact Grid)
- **Candidate Samples**: 65,564 valid samples with `seconds_remaining >= 2`
- **Eligible 2s Pairs**: **65,436** pairs satisfying exact grid matching ($t + 2000$ms) and timing gate $|(actual\_future - actual\_curr) - 2000| \le 500$ms
- **Contributing Physical Rounds**: **250 / 250** (100% of rounds contribute eligible pairs)
- **Pair Retention Rate**: **99.80%** (65,436 / 65,564)
- **Lag Timing Error**: Median: 3.0 ms, p95: 9.0 ms, p99: 10.0 ms
- **Future-Information Events**: **0** (Zero lookahead, zero backwards time travel, zero future leakage)
