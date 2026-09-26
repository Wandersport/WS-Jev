# BTC 5-Minute Lead-Lag Observational Microstructure Experiment (v1)

**EXPERIMENT_ID**: `btc5m_leadlag_v1`  
**VERSION**: `1.0.0`  
**STATUS**: `FROZEN_SPECIFICATION`  
**RESEARCH_MODE**: `READ_ONLY_OBSERVATIONAL`  
**TRADING_CAPABILITY**: `ZERO (NO LIVE TRADING, NO PAPER BROKER ORDERS)`  
**OPENROUTER_REQUESTS**: `ZERO (NO JEV FORECASTS)`  

---

## 1. Scientific Objective

Evaluate whether public Binance USD-M BTC perpetual orderbook and trade microstructure at time $t$ contains predictive information about subsequent movements in the Polymarket native BTC 5-minute binary probability $q_{t+L}$ over ultra-short positive lags ($L \in \{1, 2, 3, 5, 10, 15, 30\}$ seconds), conditional on Polymarket's own contemporaneous state and recent history.

This is a market-dynamics observational study. Binary contract settlement outcome is **not** the primary prediction target.

---

## 2. Formal Hypotheses

- **Primary Hypothesis ($H_1$)**:
  Public Binance BTC perpetual market state and order flow at time $t$ predicts subsequent Polymarket native UP midpoint movement at short positive lags ($1\text{s}, 2\text{s}, 3\text{s}, 5\text{s}, 10\text{s}, 15\text{s}, 30\text{s}$), conditional on Polymarket's own contemporaneous quote state and autoregressive returns.

- **Null Hypothesis ($H_0$)**:
  After conditioning on Polymarket's own current price, spread, and recent returns, Binance perpetual features contain no reproducible incremental information about subsequent Polymarket probability changes.

---

## 3. Target Variables

For each synchronized observation time $t$, record native Polymarket UP probability:

$$q_t = \frac{\text{best\_bid}_{\text{UP}} + \text{best\_ask}_{\text{UP}}}{2}$$

Offline post-processing targets (computed strictly offline after persistence; never accessible at feature time $t$):

$$\Delta q_L = q_{t+L} - q_t \quad \text{for } L \in \{1, 2, 3, 5, 10, 15, 30\} \text{ seconds}$$
$$\Delta \text{logit}(q)_L = \text{logit}(q_{t+L}) - \text{logit}(q_t)$$

---

## 4. Frozen Observational Features (Computed Prospectively at Time $t$)

### Venue A: Binance USD-M Perpetual (BTCUSDT)
1. **Price State**:
   - `binance_best_bid`: Top-of-book bid price
   - `binance_best_ask`: Top-of-book ask price
   - `binance_mid_price`: $(P_{\text{bid}} + P_{\text{ask}}) / 2$
   - `binance_microprice`: Volume-weighted top-1 price: $(P_{\text{bid}} Q_{\text{ask}, 1} + P_{\text{ask}} Q_{\text{bid}, 1}) / (Q_{\text{bid}, 1} + Q_{\text{ask}, 1})$
   - `binance_microprice_offset_bps`: $(P_{\text{micro}} - P_{\text{mid}}) / P_{\text{mid}} \times 10,000$
   - `binance_spread_bps`: $(P_{\text{ask}} - P_{\text{bid}}) / P_{\text{mid}} \times 10,000$
   - `binance_basis_bps`: Contre-reference displacement vs Chainlink TWAP (bps)
   - `binance_return_since_open_bps`: $(P_{\text{mid}} - P_{\text{open}}) / P_{\text{open}} \times 10,000$
2. **Returns**:
   - `binance_return_1s_bps`, `binance_return_2s_bps`, `binance_return_3s_bps`, `binance_return_5s_bps`, `binance_return_10s_bps`, `binance_return_30s_bps`, `binance_return_60s_bps`
3. **Taker Flow**:
   - `binance_taker_flow_1s`, `binance_taker_flow_2s`, `binance_taker_flow_3s`, `binance_taker_flow_5s`, `binance_taker_flow_10s`, `binance_taker_flow_30s`, `binance_taker_flow_60s`
   - Defined as: $(Q_{\text{buy}} - Q_{\text{sell}}) / (Q_{\text{buy}} + Q_{\text{sell}})$ over rolling time windows.
4. **Depth Asymmetry**:
   - `binance_top1_depth_imbalance`: $(Q_{\text{bid}, 1} - Q_{\text{ask}, 1}) / (Q_{\text{bid}, 1} + Q_{\text{ask}, 1})$
   - `binance_top5_depth_imbalance`: $(\sum_1^5 Q_{\text{bid}} - \sum_1^5 Q_{\text{ask}}) / (\sum_1^5 Q_{\text{bid}} + \sum_1^5 Q_{\text{ask}})$
   - `binance_top20_depth_imbalance`: $(\sum_1^{20} Q_{\text{bid}} - \sum_1^{20} Q_{\text{ask}}) / (\sum_1^{20} Q_{\text{bid}} + \sum_1^{20} Q_{\text{ask}})$

### Venue B: Polymarket Native UP Orderbook
1. **Contemporaneous State**:
   - `poly_best_bid`: Native UP best bid
   - `poly_best_ask`: Native UP best ask
   - `poly_midpoint`: Native UP midpoint $q_t$ (null if one-sided)
   - `poly_spread`: Native UP spread
   - `seconds_remaining`: Time until 5-minute round expiration
2. **Autoregressive State**:
   - `poly_return_1s`: $q_t - q_{t-1\text{s}}$
   - `poly_return_2s`: $q_t - q_{t-2\text{s}}$
   - `poly_return_3s`: $q_t - q_{t-3\text{s}}$
   - `poly_return_5s`: $q_t - q_{t-5\text{s}}$
   - `poly_return_10s`: $q_t - q_{t-10\text{s}}$
   - `poly_return_30s`: $q_t - q_{t-30\text{s}}$

---

## 5. Timestamp Provenance & Clock Alignment Protocol

Every 1-second sample records three distinct time dimensions:
1. `source_event_timestamp`: Exchange-reported event/matching time.
2. `local_receive_timestamp`: Local POSIX epoch millisecond when packet arrived.
3. `local_monotonic_timestamp`: High-resolution monotonically increasing nanosecond counter.

Stale data policy:
- If receipt age $> 3,000\text{ms}$ on either venue, the record is flagged with `is_stale = 1` and rejected from clean modeling samples.
- Silent forward-filling of stale quotes is strictly prohibited.

---

## 6. Predeclared Analysis Protocol (Post-Collection Only)

Following the collection of the 500 physical development rounds:
1. **Model $B_0$ (Polymarket Autoregressive Baseline)**:
   Predict $\Delta q_L$ (and $\Delta \text{logit}(q)_L$) using only Polymarket own-state features ($q_t$, `poly_spread`, `seconds_remaining`, and `poly_return_*`).
2. **Model $B_1$ (Polymarket + Binance Full Lead-Lag)**:
   Predict $\Delta q_L$ using $B_0$ features plus predeclared Binance features.
3. **Statistical Test**:
   Paired clustered comparison ($R^2$, MSE, directional accuracy) to determine whether $B_1$ outperforms $B_0$ out-of-fold.
