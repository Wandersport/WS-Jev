# BTC 5-Minute Lead-Lag Observational Experiment Specification (v2)

- **Experiment ID**: `btc5m_leadlag_v2`
- **Pilot Experiment ID**: `btc5m_leadlag_v2_pilot`
- **Status**: APPROVED EXPERIMENTAL SPECIFICATION
- **Predecessor Experiment**: `btc5m_leadlag_v1`
- **Design Objective**: Remediate measurement-system defects identified in the Phase 8C.1 forensic audit before executing any subsequent empirical lead-lag data collection.

---

## 1. Executive Summary of v1 Root Causes & Mandated v2 Fixes

The Phase 8C.1 forensic audit confirmed two critical measurement-system implementation defects in `btc5m_leadlag_v1`:

1. **Polymarket Timestamp Discarding**:
   - *v1 Defect*: In `_handle_poly_message()`, incremental `price_changes` messages explicitly called `_update_poly_state(..., source_ts=None, ...)` rather than extracting the outer message timestamp (`item.get("timestamp")`). Furthermore, `leadlag_polymarket_book_events` only stored a truncated `raw_hash` rather than the raw wire payload, making retroactive recovery mathematically impossible (97.58% missing source timestamps).
   - *v2 Mandate*: Full preservation of `timestamp` from both initial book snapshots and incremental `price_changes` outer payloads, plus lossless compression of complete raw WebSocket messages into `zlib` compressed binary blobs for retroactive provenance auditing.

2. **Binance aggTrade Combined Stream Casing Bug**:
   - *v1 Defect*: The combined stream subscription URL subscribed to `btcusdt@depth20@100ms/btcusdt@aggTrade`, but Binance USD-M futures WebSocket servers emit all combined stream identifiers in lowercase (`"stream": "btcusdt@aggtrade"`). The parser checked `elif "aggTrade" in stream:`. Because `"aggTrade"` has an uppercase `'T'`, the condition evaluated to `False` on 100% of incoming trade messages.
   - *v2 Mandate*: Case-insensitive stream matching (`stream.lower()`), explicit event type routing (`payload.get("e") == "aggTrade"`), removal of all silent `except Exception: pass` blocks, and explicit unhandled-event telemetry counters.

---

## 2. Quantitative Scope and Constants

- **Canonical Experiment ID**: `btc5m_leadlag_v2`
- **Pilot Experiment ID**: `btc5m_leadlag_v2_pilot`
- **Target Physical Rounds**: 500 rounds (~41.7 hours continuous observational collection)
- **Pilot Physical Rounds**: 3 rounds (~15 minutes observational validation)
- **Sampling Cadence**: 1.0 second ($\pm 10\text{ms}$ precision monotonic timer alignment)
- **Target Predeclared Lags**: $[1, 2, 3, 5, 10, 15, 30]$ seconds
- **Target Horizons**: $\Delta q_L$ and $\Delta \text{logit}(q_L)$
- **Stale Feed Threshold**: $\Delta t_{\text{receipt}} > 3000\text{ms}$

---

## 3. Required Instrumentation Enhancements (13 Mandates)

### Mandate 1: Polymarket Source Timestamp Preservation Across All WS Payloads
- Extract `timestamp` from all supported Polymarket WebSocket market channel frames:
  - Snapshot array: `item.get("timestamp")`
  - Price change event: `item.get("timestamp")` or `item.get("price_changes", [{}])[0].get("timestamp")`
- Fallback to REST only when stream is silent; record `provenance_mode: "WS_SNAPSHOT" | "WS_DELTA" | "REST_FALLBACK"`.

### Mandate 2: Complete Raw Message Persistence / Lossless Provenance
- Store raw wire payloads in `leadlag_v2_raw_payloads` compressed using Python standard-library `zlib`.
- Include SHA256 checksum and uncompressed length for deterministic round-trip verification.
- Guarantees that any future audit can reconstruct the exact wire payload.

### Mandate 3: Case-Insensitive Binance Combined Stream Parser
- Use case-folded stream comparison:
  ```python
  s_lower = stream.lower()
  if "depth" in s_lower:
      ...
  elif "aggtrade" in s_lower or payload.get("e") == "aggTrade":
      ...
  ```

### Mandate 4: Elimination of Broad Silent Exception Suppression
- Remove all `except Exception: pass` in network message handling.
- Replace with structured logging and increment explicit diagnostic counters.

### Mandate 5: Explicit Telemetry & Malformed Event Counters
- Track in collector heartbeat:
  - `depth_events_received`
  - `aggtrade_events_received`
  - `duplicate_aggtrade_events`
  - `out_of_order_trade_events`
  - `malformed_binance_events`
  - `snapshot_events_received`
  - `delta_events_received`
  - `rest_fallback_events`
  - `source_timestamp_present`
  - `source_timestamp_missing`
  - `malformed_poly_events`
  - `unhandled_stream_count`
  - `parser_exception_count`

### Mandate 6: Aggregate Trade ID Deduplication
- Preserve strict unique primary key on `(experiment_id, agg_trade_id)` in `leadlag_v2_binance_trade_events` using `INSERT OR IGNORE`.
- In memory, deduplicate rolling deque by `agg_trade_id`.

### Mandate 7: Strict Event Ordering Guarantees
- Order trades by `trade_ts_ms` ascending; if identical, tie-break by `agg_trade_id`.
- Rejects out-of-order sequence regressions.

### Mandate 8: Per-Feed Sequence & Continuity Diagnostics
- Monitor Binance depth event continuity (`E` event time and update IDs `U`, `u`, `pu`).
- Monitor Polymarket price change `hash` continuity where available.

### Mandate 9: Real-Time Source vs Receive Coverage Metrics
- Monitor live source timestamp coverage on both feeds:
  - Alert if `poly_source_coverage < 95%` after warmup.
  - Alert if `binance_source_coverage < 99%`.

### Mandate 10: Percentile Latency Metrics in Heartbeats
- Compute rolling window percentiles (p50, p90, p95, p99, max) for:
  - Binance wire latency: `recv_ts_ms - source_ts_ms`
  - Polymarket wire latency: `recv_ts_ms - source_ts_ms`
  - Inter-feed receive skew: $|t_{\text{recv, Binance}} - t_{\text{recv, Poly}}|$
  - Target error: $|t_{\text{actual}} - t_{\text{target}}|$

### Mandate 11: Robust Round & Token Transition Handling
- Pre-fetch next round Gamma event metadata 30 seconds prior to epoch boundary (`T - 30s`).
- Retry Gamma metadata fetch with exponential backoff (`0.5s, 1.0s, 2.0s`) to prevent round transition dropouts.
- Zero old token book or open price leakage across rounds.

### Mandate 12: Taker-Flow Coverage SLA (>95%)
- Require `binance_taker_flow_*` non-null coverage >95% on all valid samples after 60s warmup.
- Never substitute missing trade data with 0.

### Mandate 13: Payload Fixture Unit & Integration Tests
- Maintain full fixture test suite testing exact raw WebSocket payloads from both Binance and Polymarket.

---

## 4. Frozen Safety Constraints

- **STRICTLY PAPER-ONLY**: The v2 collector remains purely observational and analytical.
- **ZERO TRADING CAPABILITY**: No wallets, private keys, live brokers, execution adapters, or real order routing.
- **EXECUTION**: `btc5m_leadlag_v2` is a research specification only; execution shall not be initiated without explicit authorized command.
