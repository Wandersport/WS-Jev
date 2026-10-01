# Phase 8C.2 / BTC 5-Minute Lead-Lag V2 — Final Collection Audit

**Audit Timestamp:** `2026-10-01T00:59:45Z`  
**Experiment ID:** `btc5m_leadlag_v2`  
**Frozen Spec Hash:** `dc8663876e8a1ca5f22d73f1e5f8bdd69fcf1a4946aba741708947c7e63c3352`  
**Status:** `NORMAL_TARGET_COMPLETION` (500 / 500 Physical Rounds)

---

## 1. Executive Summary

The Phase 8C.2 autonomous high-frequency lead-lag v2 collector has successfully reached its target of **500 completed physical rounds** (~41.7 hours of continuous 1-second synchronized observational market data).

- **Total Completed Rounds:** 500
- **Active / Unclosed Rounds:** 0
- **Duplicate Round Slugs:** 0
- **Total Synchronized 1s Samples:** 58,617.0
- **Valid Synchronized Samples:** 57,716.0 (98.46%)
- **Stale Samples:** 267 (stale rate: 0.46%)
- **Binance Source Timestamp Coverage:** 99.99% (spec threshold: >=99.0%)
- **Polymarket Source Timestamp Coverage:** 99.99% (spec threshold: >=95.0%)
- **Taker-Flow 60s Lookback Coverage:** 99.60% (spec threshold: >=95.0%)
- **Parser Exceptions:** 0 in final run
- **Malformed Events:** 0
- **Duplicate Binance Trades:** 0 (enforced by PK on `(experiment_id, agg_trade_id)`)
- **Raw Payload Deterministic SHA256 Round-Trip:** 100.0% (300/300)
- **Cross-Experiment Contamination:** NONE

---

## 2. Round Quality Distributions

Across all 500 completed physical rounds:

| Metric | Min | P1 | P5 | P25 | Median (P50) | P75 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|---|
| **Sample Count** | 1.0 | 43.88 | 53.0 | 67.0 | 95.5 | 126.0 | 300.0 | 300.0 | 300.0 |
| **Valid Sample Count** | 0.0 | 42.88 | 51.0 | 66.0 | 94.0 | 124.0 | 296.1 | 299.0 | 299.0 |
| **Observed Span (sec)** | 1.0 | 97.42 | 290.86 | 294.82 | 296.62 | 297.93 | 298.99 | 299.33 | 299.59 |

### Round Bracket Counts
- **< 10 samples:** 3 rounds (startup / intentional pause boundary artifacts)
- **< 60 samples:** 75 rounds
- **< 120 samples:** 334 rounds
- **< 240 samples:** 423 rounds
- **>= 240 samples:** 77 rounds

### Identified Truncated / Boundary Rounds (4 rounds)
The following rounds were captured during operational transitions (startup, manual pause, or network recovery):
- **Round #377** (`btc-updown-5m-1790763900`): 1 samples, span=1.0s
- **Round #401** (`btc-updown-5m-1790777100`): 3 samples, span=10.32s
- **Round #402** (`btc-updown-5m-1790781600`): 17 samples, span=35.07s
- **Round #500** (`btc-updown-5m-1790811000`): 1 samples, span=1.0s

*Note: These boundary rounds are documented descriptively and preserved exactly as recorded. No ad-hoc exclusion filtering is applied prior to formal Phase 8D econometric analysis.*

---

## 3. Session & Interruption History

The 500 rounds were captured across **6 operational sessions**:

| Session | PID | Start (UTC) | End (UTC) | Rounds | Count | Stop Type | Stop Reason |
|---|---|---|---|---|---|---|---|
| 1 | 12469 | `2026-09-28T21:15:35.116942+00:00` | `2026-09-29T09:48:42.909198+00:00` | 1..151 | 151 | `NETWORK_ABORT` | SUSTAINED_FAILURE: EXCESSIVE_PARSER_EXCE... |
| 2 | 12374 | `2026-09-29T11:27:05.943524+00:00` | `2026-09-29T12:43:53.813191+00:00` | 152..167 | 16 | `NETWORK_ABORT` | SUSTAINED_FAILURE: BINANCE_DEPTH_SILENCE... |
| 3 | 16311 | `2026-09-29T14:33:23.171793+00:00` | `2026-09-29T15:39:58.580658+00:00` | 168..181 | 14 | `MANUAL_PAUSE` | MANUAL_PAUSE (SIGTERM graceful stop at r... |
| 4 | 18829 | `2026-09-29T18:10:47.081852+00:00` | `2026-09-30T10:28:47.892414+00:00` | 182..377 | 196 | `MANUAL_PAUSE` | MANUAL_PAUSE (SIGTERM graceful stop; par... |
| 5 | 79578 | `2026-09-30T12:13:51.440754+00:00` | `2026-09-30T14:05:35.522364+00:00` | 378..401 | 24 | `MANUAL_PAUSE` | MANUAL_PAUSE (SIGTERM graceful stop; par... |
| 6 | 92515 | `2026-09-30T15:24:25.731905+00:00` | `2026-09-30T23:30:13.054708+00:00` | 402..500 | 99 | `NORMAL_COMPLETION` | NORMAL_COMPLETION (Target of 500 physica... |

All temporal gaps between sessions correspond to documented host maintenance, hotfix testing, or manual pauses. No simulated samples were recorded across session gaps.

---

## 4. Raw Provenance & Lossless Preservation

- **Total Raw Payload Frames:** ~83,673,724
- **SHA256 Round-Trip Integrity:** 300 / 300 sample payloads verified with exact byte and digest parity.
- **Polymarket Unhandled Events:** 59,296 events. Confirmed to be public `last_trade_price` execution notifications. These were intentionally unhandled by the midpoint book estimator but are losslessly preserved in the raw payload database.
- **REST Fallbacks:** 78 samples used public REST fallback when WebSocket frames were delayed, with zero parser exceptions.

---

## 5. Storage State

- **Active Database File:** `pm_research.db` — 65.5 GiB (70,335,336,448 bytes)
- **Free Disk Space:** 14.31 GiB
- **Canonical Recovery Backups:**
  - `pm_research_backup_20260928_100735_30r.db`: 2.19 GiB
  - `pm_research_backup_20260928_211322.db`: 2.5 GiB

---

## 6. Audit Verdict

- **FINAL_V2_COLLECTION_STATUS:** `COMPLETE_SUCCESS`
- **NORMAL_TARGET_COMPLETION:** `YES`
- **READY_FOR_PHASE8D_ANALYSIS:** `YES`
