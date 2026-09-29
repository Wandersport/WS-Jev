# Operational Incident Manifest: BTC 5m Lead-Lag v2 Watchdog Abort & Recovery

- **Date / Timestamp**: 2026-09-29T09:48:46Z - 2026-09-29T09:49:02Z
- **Preceding Production HEAD**: `805160c010495c105f7b7f674a973002ac9c36e2`
- **Experiment ID**: `btc5m_leadlag_v2`
- **Scientific Specification Hash**: `dc8663876e8a1ca5f22d73f1e5f8bdd69fcf1a4946aba741708947c7e63c3352` (UNCHANGED)

---

## 1. Incident Description & Root Cause

During production collection of `btc5m_leadlag_v2` (target: 500 physical rounds), the background collector process aborted via its sustained health watchdog at round 151:
```text
RuntimeError: Collector aborted by watchdog: SUSTAINED_FAILURE: EXCESSIVE_PARSER_EXCEPTIONS_51
```

### Root Cause Analysis
1. During transient WebSocket reconnection / quiet feed intervals, the collector invokes `_poll_poly_rest_fallback()`.
2. The fallback method invoked `self.poly_rest_collector.fetch_order_book(token_id)`.
3. The `PolymarketBookCollector` class defines `fetch_book(token_id)` (not `fetch_order_book`).
4. This raised `AttributeError: 'PolymarketBookCollector' object has no attribute 'fetch_order_book'`, caught by the general parser exception block and incrementing `_parser_exception_count`.
5. When `_parser_exception_count > 50`, the production health watchdog flagged `EXCESSIVE_PARSER_EXCEPTIONS_51` for 6 consecutive evaluations (60s sustained), cleanly halting the collector.

---

## 2. Dataset & Database State Audit

- **SQLite Integrity (`PRAGMA quick_check`)**: `ok`
- **Total Persisted Rounds**: Exactly 151 completed physical rounds
- **Active / Incomplete Rounds**: 0 (all 151 rounds have `status = 'COMPLETED'`)
- **Total Synchronized 1s Samples**: 31,213 (Valid: 30,896, Stale: 92)
- **Stale Sample Rate**: 0.29% (comfortably below the 5.0% ceiling)
- **Affected Round**: Round 151 (`btc-updown-5m-1790675100`), started at `2026-09-29T09:45:00Z` and completed at `2026-09-29T09:48:46Z` with 93 samples (76 valid) upon graceful shutdown.
- **Predictive Confidentiality**: ZERO predictive modeling, label checking, or scientific outcome inspection was conducted on partial v2 data.

---

## 3. Engineering Hotfix & Crash-Safe Resume Semantics

1. **REST Fallback Correction**:
   - Replaced `self.poly_rest_collector.fetch_order_book(token_id)` with `self.poly_rest_collector.fetch_book(token_id)`.
   - Serialized `BookLevel` objects into standard JSON-serializable dictionaries `[{"price": ..., "size": ...}]` for lossless raw payload compression.
2. **Crash-Safe Resume Semantics**:
   - Target rounds (`target_physical_rounds = 500`) represents the TOTAL persisted completed rounds for the experiment.
   - At startup, `LeadLagCollectorV2` initializes progress from existing completed rounds in the database.
   - With 151 existing completed rounds, `_rounds_captured_count` initializes to 151.
   - The next captured round is registered as round 152 (`Physical round 152/500 registered`).
   - If started within an already completed round window, the collector waits for the next round transition without overwriting.
   - The collector cleanly terminates when total persisted rounds reaches 500.
3. **Fail-Closed Guarantees**:
   - Collector immediately aborts (`RuntimeError`) at startup if any unexpected ACTIVE or non-completed rounds exist in the database.
   - Collector immediately aborts if any persisted round has an experiment spec hash mismatch or cross-mode (`is_pilot`) inconsistency.

---

## 4. Verification Suite

- Regression tests added in `tests/test_btc5m_leadlag_v2.py`:
  - `test_poly_rest_fallback_hotfix_accepts_book_and_records_provenance`
  - `test_resume_from_151_to_target_500`
  - `test_resume_fail_closed_on_unexpected_active_round`
  - `test_resume_fail_closed_on_inconsistent_spec_hash`
- Full test suite: `uv run pytest` passes 100%.
- Code quality: `uv run ruff check src tests` passes cleanly.
- Public safety: `uv run pmr verify-safety` passes with 0 violations.
- Scientific spec hash: UNCHANGED.
