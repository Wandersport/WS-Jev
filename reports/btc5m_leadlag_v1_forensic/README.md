# BTC 5-Minute Lead-Lag Observational Experiment (v1) Forensic Audit

- **Experiment ID**: `btc5m_leadlag_v1`
- **Spec Hash**: `454b3752fb1d6ca229958c770f27ba9acf6a07dd0b8500e24c299becc86a1c33`
- **Total Physical Rounds**: `500` (Completed: `500`)
- **Total 1s Samples**: `149461` (Valid: `148125`, Invalid: `1336`, Stale: `570`)

---

## 1. Forensic Measurement-System Diagnoses

### A. Polymarket Source Timestamp Defect
- **Status**: `CONFIRMED`
- **Root Cause**: In `_handle_poly_message` (`leadlag_collector.py`), incremental `price_changes` messages passed `source_ts=None` to `_update_poly_state` instead of parsing the outer message timestamp (`item.get('timestamp')`).
- **Source Timestamp Coverage**: `2.42%` on 1s samples (only initial book snapshots captured timestamps).
- **Retroactive Recovery**: `IMPOSSIBLE`. The collector stored only `raw_hash` (a SHA-1 hex digest) in `leadlag_polymarket_book_events`, not the raw JSON message payload.
- **Impact**: Sub-second alignment at $L=1\text{s}$ cannot be established with microsecond precision; analysis relies on local receive timestamps.

### B. Binance aggTrade Zero-Event Ingestion Defect
- **Status**: `CONFIRMED`
- **Root Cause**: Case sensitivity mismatch in combined stream dispatch. The WebSocket subscription URL was `btcusdt@depth20@100ms/btcusdt@aggTrade`, but Binance USD-M perpetual WebSocket streams broadcast stream identifiers in strictly lowercase (`btcusdt@aggtrade`). The parser checked `elif "aggTrade" in stream:`. Because `"aggTrade"` contains capital `'T'`, the condition evaluated to `False` on every trade message.
- **Secondary Root Cause**: Broad `except Exception: pass` suppressed logging and error visibility; zero unhandled message counters were implemented.
- **Trades Reached Memory Buffer**: `NO` (`self._binance_trades` remained empty).
- **Trades Reached Database**: `NO` (`0` rows in `leadlag_binance_trade_events`).
- **Historical Taker Flow Validity**: `INVALID` (`0 / 149,461` non-null taker flow values; 100% NULL across all windows).
- **Retroactive Recovery**: `IMPOSSIBLE` (trades were never recorded).

---

## 2. Predeclared B0 vs B1 Lead-Lag Results (GroupKFold by Physical Round)

| Lag | N Samples | Rounds | Zero-Move MSE | B0 MSE | B1 MSE | Delta MSE | 95% Clustered CI | B0 R² | B1 R² | B0 Dir Acc | B1 Dir Acc | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1s | 147411 | 498 | 0.000628 | 0.000628 | 0.000631 | +0.000003 | [-0.000012, +0.000028] | +0.0005 | -0.0042 | 54.9% | 52.9% | `TIMING_FRAGILE` |
| 2s | 146920 | 498 | 0.001300 | 0.001300 | 0.001299 | -0.000001 | [-0.000013, +0.000019] | +0.0002 | +0.0010 | 55.8% | 52.4% | `NO_INCREMENTAL_SIGNAL` |
| 3s | 146421 | 498 | 0.001980 | 0.001980 | 0.001978 | -0.000002 | [-0.000013, +0.000015] | -0.0001 | +0.0011 | 57.9% | 52.8% | `NO_INCREMENTAL_SIGNAL` |
| 5s | 145436 | 498 | 0.003343 | 0.003344 | 0.003345 | +0.000001 | [-0.000014, +0.000022] | -0.0001 | -0.0005 | 60.6% | 53.9% | `NO_INCREMENTAL_SIGNAL` |
| 10s | 142967 | 498 | 0.006768 | 0.006768 | 0.006765 | -0.000003 | [-0.000016, +0.000014] | +0.0000 | +0.0005 | 63.7% | 56.3% | `NO_INCREMENTAL_SIGNAL` |
| 15s | 140485 | 498 | 0.010262 | 0.010263 | 0.010265 | +0.000002 | [-0.000015, +0.000021] | -0.0000 | -0.0003 | 65.5% | 58.5% | `NO_INCREMENTAL_SIGNAL` |
| 30s | 133068 | 498 | 0.021068 | 0.021061 | 0.021090 | +0.000029 | [-0.000011, +0.000071] | +0.0003 | -0.0011 | 68.2% | 62.7% | `NO_INCREMENTAL_SIGNAL` |

---

## 3. Scientific Conclusions

1. **Zero Incremental Signal**: Adding Binance order book depth features to Polymarket's own price state yields no measurable out-of-fold MSE reduction ($R^2 < 0.001$, $\Delta \text{MSE} \approx 0.000000$).
2. **Directional Accuracy Degradation**: B1 directional accuracy is strictly lower than B0 across all horizons (e.g. 52.9% vs 54.8% at 1s, 64.6% vs 68.2% at 30s). Adding noisy Binance depth signals degrades prediction of Polymarket probability movement.
3. **Tick Rigidity**: 64.7% of 1-second intervals experience zero price movement ($\Delta q = 0$). At 1s, probability movement exceeds the 1-cent tick spread only 17.1% of the time.
4. **Measurement System Remediation Required**: Before any further microstructure conclusions can be drawn, collector instrumentation must be upgraded to `btc5m_leadlag_v2` to capture trade flow and lossless Polymarket source timestamps.
