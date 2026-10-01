# BTC 5-Minute Lead-Lag Observational Experiment Specification (v3 Replication)

**Canonical Experiment ID**: `btc5m_leadlag_v3_replication_2s`  
**Pilot Experiment ID**: `btc5m_leadlag_v3_replication_2s_pilot`  
**Version**: `3.0.0`  
**Target Horizon**: Exactly **+2.0 seconds** (`L = 2s` only)  
**Target Physical Rounds**: `250` (prospective power $\ge 85\%$ even under 25% effect shrinkage)  
**Pilot Physical Rounds**: `10` (strictly measurement-only validation)  
**Isolated Storage Path**: `data/pm_research_v3_replication.db`  

---

## 1. Scientific Objective & Hypotheses

The discovery phase (v2) established that public Binance BTC perpetual microprice offset and depth imbalance provide statistically robust incremental predictive power for native Polymarket UP token midpoint movement over a **2-second horizon** conditional on Polymarket's own autoregressive state ($\Delta R^2_{CV} = +0.00667$, round-clustered $t = +4.54$, Holm $p = 0.00007$, win rate $56.2\%$). Shorter horizons (1s) were marginal/cadence-constrained, and longer horizons ($\ge 3$s) decayed to noise.

This prospective experiment tests whether the 2-second incremental lead transports out-of-sample to an independently collected, prospective dataset with zero sampling cadence degradation.

### Predeclared Analyses:
1. **Primary: No-Refit Transport Test**:
   Directly evaluate the frozen v2 models (`models/frozen_v2_replication_models_2s.json`, SHA256 `ee914a59072320d142944530957036c325632f941432d934f472e689dc7f1c77`) on prospective replication data without refitting.
2. **Secondary: Same-Spec Refit**:
   Refit the identical Ridge regression specification ($L2=1.0$) exclusively within replication data using 5-fold GroupKFold by physical round.

---

## 2. Frozen Feature Sets & Target Definitions

### A. Polymarket Controls Baseline ($B_0$, 9 features):
1. `poly_midpoint`
2. `poly_spread`
3. `seconds_remaining`
4. `poly_return_1s`
5. `poly_return_2s`
6. `poly_return_3s`
7. `poly_return_5s`
8. `poly_return_10s`
9. `poly_return_30s`

### B. Binance Incremental Features (14 features):
1. `binance_mid_price`
2. `binance_microprice_offset_bps`
3. `binance_spread_bps`
4. `binance_return_since_open_bps`
5. `binance_return_1s_bps`
6. `binance_return_2s_bps`
7. `binance_return_3s_bps`
8. `binance_return_5s_bps`
9. `binance_return_10s_bps`
10. `binance_return_30s_bps`
11. `binance_return_60s_bps`
12. `binance_top1_depth_imbalance`
13. `binance_top5_depth_imbalance`
14. `binance_top20_depth_imbalance`

### C. Augmented Model ($B_1$, 23 features):
$B_1 = B_0 \cup \text{Binance Incremental}$.

### D. Targets:
- Primary Target: $\Delta q_{2s} = q_{t+2000\text{ms}} - q_t$
- Secondary Target: $\Delta \text{logit}(q)_{2s} = \text{logit}(q_{t+2000\text{ms}}) - \text{logit}(q_t)$ with clamping $\epsilon = 10^{-6}$.

### E. Pairing Rules:
- Pairing must use exact `sample_target_ts_ms` grid matching (`target_future - target_curr == 2000`).
- Primary timing gate: $|(actual\_future - actual\_curr) - 2000| \le 500\text{ms}$.
- Sensitivity timing gate: $|(actual\_future - actual\_curr) - 2000| \le 250\text{ms}$.
- Nearest-neighbor matching is strictly forbidden.

---

## 3. Remediated Measurement Architecture (Cadence-Fixed)

To eliminate the SQLite blocking that caused cadence degradation in v2:
1. **Isolated Asynchronous Persistence Thread**:
   - The tick sampling loop takes an instantaneous in-memory snapshot of feed state and enqueues it to `queue.Queue(maxsize=5000)`.
   - A dedicated background `PersistenceWorker` thread batches writes and executes SQLite transactions asynchronously.
   - The sampling thread executes zero SQLite queries and zero SQLite writes during tick collection.
2. **In-Memory Heartbeat & Telemetry**:
   - Heartbeat metrics (ticks intended, captured, missed, latency percentiles) are tracked in $O(1)$ in-memory circular buffers.
   - Zero full-table scans during runtime.
3. **Fail-Closed Backlog Protection**:
   - Writer queue backlog is continuously monitored. If queue depth exceeds 2,000 items, the collector stops and fails closed rather than degrading sampling cadence.
4. **Storage Isolation**:
   - All v3 replication data is persisted exclusively to `data/pm_research_v3_replication.db`. Zero reads or writes to `pm_research.db`.
5. **Paper-Only Invariance**:
   - Zero live execution capability, zero private keys, zero order routing.
