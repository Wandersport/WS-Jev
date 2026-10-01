# Phase 8C.2 / BTC 5-Minute Lead-Lag V2 — Measurement Cadence Audit

**Audit Date:** `2026-10-01T10:03:23Z`
**Experiment ID:** `btc5m_leadlag_v2`
**Spec Hash:** `dc8663876e8a1ca5f22d73f1e5f8bdd69fcf1a4946aba741708947c7e63c3352`

---

## 1. Problem Statement

The final collection audit revealed that while all 500 physical rounds were captured and closed cleanly with a median observed round span of **296.6 seconds**, the total sample count was **58,617** rather than the theoretical target of **150,000** (500 rounds × 300 seconds). The median sample count per round was **95.5** instead of 300, and **423 / 500 rounds** contained fewer than 240 samples.

This audit evaluates the root cause, degradation trajectory, and econometric viability of the dataset for Phase 8D lead-lag analysis.

---

## 2. Root Cause Determination

The cadence slowdown was caused by **synchronous database operations blocking the single-threaded collector sampling loop**:

1. **O(N) Full-Table Scan in Heartbeat (`_compute_actual_taker_flow_coverage`)**:
   - In `src/pm_research/research/btc5m/leadlag_v2_collector.py` (lines 1296–1328), the collector emits a heartbeat every 5 seconds.
   - To compute real taker-flow coverage without relying on elapsed-time proxies, it calls `self.db.get_leadlag_v2_samples(experiment_id=self.experiment_id)`.
   - This executes `SELECT * FROM leadlag_v2_samples WHERE experiment_id = ?`, loading all accumulated samples into memory.
   - At Round 1 (0 samples), this query took **<2ms**.
   - By Round 250 (~35k samples), it took **~1.5s**.
   - By Round 500 (58k samples), it took **~3.34s**.
   - Because this ran in the main loop thread every ~5 seconds, the thread was blocked for multiple seconds on every heartbeat.

2. **Large-Volume B-Tree Writes in `_flush_batch()`**:
   - Every 10 samples, `_flush_batch()` synchronously inserts pending raw WebSocket wire payloads into `leadlag_v2_raw_payloads`.
   - By collection completion, `leadlag_v2_raw_payloads` had reached **83.6 million rows** (70.3 GB).
   - Each batch update required writing and balancing SQLite B-tree pages for both the primary key and the secondary index `idx_leadlag_v2_payload_exp`.

3. **Timer Realignment Behavior**:
   - In the collector main loop, after waking from a blocking operation, the timer calculates `sleep_sec = 1.0 - (now % 1.0)`.
   - It aligns forward to the **next** integer second and does **not** backfill missed seconds.
   - Consequently, when the thread blocked for 3.3 seconds, 3 to 4 one-second sampling ticks were skipped.
   - The round duration (5 minutes) remained intact, but intra-round sample ticks were omitted.

4. **Feed Integrity Confirmation**:
   - Stale sample rate was **0.46%**.
   - Binance timestamp coverage was **99.99%**.
   - Polymarket timestamp coverage was **99.99%**.
   - The external feeds were **not** silent or stalled; the sampling loop simply ticked at a slower rate.

---

## 3. Degradation Trajectory

The cadence degraded monotonically as cumulative samples accumulated in SQLite:

- **Correlation (`round_index` vs `sample_count`)**: Pearson = `-0.8534`, Spearman = `-0.9723`
- **Correlation (`round_index` vs `median_interval`)**: Pearson = `0.9510`, Spearman = `0.9607`

### 25-Round Block Cadence Evolution

| Round Range | Median Samples / Round | Median Inter-Arrival | P95 Inter-Arrival | Effective Samples / Min |
|---|---|---|---|---|
| 1..25 | 300 | 1.00s | 1.02s | 60.2 |
| 26..50 | 299 | 1.00s | 1.06s | 60.0 |
| 51..75 | 246 | 1.01s | 2.00s | 49.5 |
| 76..100 | 178 | 1.93s | 2.98s | 35.8 |
| 101..125 | 126 | 2.04s | 3.02s | 25.4 |
| 126..150 | 126 | 2.05s | 3.03s | 25.4 |
| 151..175 | 121 | 2.09s | 3.71s | 24.5 |
| 176..200 | 109 | 2.32s | 4.05s | 22.0 |
| 201..225 | 100 | 2.59s | 4.19s | 20.2 |
| 226..250 | 95 | 3.58s | 4.43s | 19.2 |
| 251..275 | 96 | 3.63s | 4.49s | 19.4 |
| 276..300 | 91 | 3.92s | 5.00s | 18.3 |
| 301..325 | 78 | 4.04s | 5.11s | 15.9 |
| 326..350 | 71 | 4.79s | 5.20s | 14.4 |
| 351..375 | 70 | 4.76s | 5.39s | 14.2 |
| 376..400 | 61 | 5.04s | 6.24s | 12.6 |
| 401..425 | 60 | 5.04s | 6.16s | 12.7 |
| 426..450 | 61 | 5.02s | 6.04s | 12.4 |
| 451..475 | 58 | 5.42s | 6.12s | 11.8 |
| 476..500 | 54 | 5.80s | 6.32s | 11.0 |

---

## 4. Frozen Lag Pairing Eligibility

In `src/pm_research/research/btc5m/leadlag_analysis.py`, target observation pairs are constructed using **Exact Target Grid Matching**:
```python
future_s = sample_map.get((round_slug, sample_target_ts_ms + lag_ms))
```

Because `sample_target_ts_ms` is strictly aligned to integer seconds, exact grid matching ensures that whenever a future observation exists on the grid, its nominal spacing is exactly $L$ seconds.

### Lag Pair Metrics Across 500 Physical Rounds (57,716 Valid Candidates)

| Lag | Valid Pairs | Capture Rate | P50 Error | P95 Error | P99 Error | % within ±100ms | % within ±250ms | % within ±500ms |
|---|---|---|---|---|---|---|---|---|
| **1s** | 20,372 | 35.3% | 12ms | 83ms | 743ms | 96.4% | 98.9% | 99.0% |
| **2s** | 31,878 | 55.2% | 17ms | 245ms | 925ms | 89.4% | 95.1% | 97.2% |
| **3s** | 28,050 | 48.6% | 17ms | 499ms | 934ms | 86.8% | 92.6% | 95.0% |
| **5s** | 35,163 | 60.9% | 19ms | 433ms | 883ms | 84.0% | 91.6% | 96.0% |
| **10s** | 30,981 | 53.7% | 18ms | 288ms | 924ms | 87.5% | 94.3% | 96.9% |
| **15s** | 25,675 | 44.5% | 17ms | 412ms | 947ms | 88.4% | 93.5% | 95.3% |
| **30s** | 27,903 | 48.4% | 18ms | 381ms | 932ms | 85.3% | 92.3% | 95.9% |

### Key Findings on Lag Eligibility
1. **Massive Statistical Power**: Across all 7 lags, the dataset provides **20,372 to 35,163 valid observation pairs per lag**.
2. **Sub-20ms Median Precision**: For every lag horizon, the median timing error is **12ms to 19ms**.
3. **P95 Error Under 500ms**: Over 95% of pairs across all lags fall within **±500ms** of the nominal horizon.

---

## 5. Phase 8D Econometric Decision

- **CADENCE_AUDIT_STATUS:** `AUDIT_COMPLETE`
- **FROZEN_LAGS_MEASURABLE:** `YES`
- **PHASE8D_CAN_PROCEED:** `YES`
- **REQUIRED_TIMING_GATE:** `EXACT_GRID_MATCHING_OR_MAX_LAG_ERROR_500MS`

### Methodological Rule for Phase 8D
1. **Predeclared Target Pairing**: Use Exact Grid Matching (`(round_slug, target_ts + lag_ms)`) with an explicit timing-tolerance gate:
   $$\left| (t_{\text{actual, fut}} - t_{\text{actual, curr}}) - L \right| \le 500\text{ms}$$
2. **Feature Integrity**: Features ($X_t$) are computed from raw event deques at the sample capture timestamp, guaranteeing no lookahead and exact temporal validity.
3. **No Series Distortion**: By restricting analysis to verified $(X_t, Y_{t+L})$ pairs, temporal gaps between skipped samples introduce no econometric distortion (unlike autoregressive models that assume equidistant steps without timestamp conditioning).
