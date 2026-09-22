# Statistical Probability Calibration

In quantitative prediction market research, **probability calibration** is the primary benchmark of forecast quality, not win rate or simulated paper return.

---

## Why Calibration Outweighs Win Rate

A trading system can achieve an 80% win rate simply by buying contracts priced at 90 cents on high-probability events; however, if those contracts were actually 75% likely to occur, the system has a negative expected value and will eventually suffer catastrophic drawdowns.

Conversely, a properly calibrated forecaster whose predicted probability $\hat{q} = P(\text{YES})$ aligns with the true conditional frequency of outcomes provides a reliable foundation for fractional Kelly sizing and sustainable capital preservation.

---

## Mathematical Metrics

### 1. Brier Score
For $N$ resolved forecasts where $q_i \in [0, 1]$ is the predicted probability and $y_i \in \{0, 1\}$ is the realized binary outcome:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^N (q_i - y_i)^2$$

- Range: $[0.0, 1.0]$.
- A score of $0.0$ indicates perfect forecasting.
- A baseline predicting $0.5$ for every event has a Brier score of $0.25$.

### 2. Log Loss (Binary Cross-Entropy)
Penalizes severe overconfidence on incorrect forecasts:

$$\text{LL} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{q}_i) + (1 - y_i) \ln(1 - \hat{q}_i) \right]$$

To prevent numerical instability, $\hat{q}_i$ is clamped to $[\epsilon, 1 - \epsilon]$ where $\epsilon = 10^{-6}$.

### 3. Forecast Bias
Measures systematic optimism or pessimism:

$$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (q_i - y_i) = \bar{q} - \bar{y}$$

- $\text{Bias} > 0$: The model systematically over-predicts YES occurrences.
- $\text{Bias} < 0$: The model systematically under-predicts YES occurrences.

### 4. Expected Calibration Error (ECE)
Observations are partitioned into $K$ equal-width probability bins $B_1, \dots, B_K \subset [0, 1]$:

$$\text{ECE} = \sum_{k=1}^K \frac{|B_k|}{N} \left| \bar{q}(B_k) - \bar{y}(B_k) \right|$$

where:
- $\bar{q}(B_k)$ is the mean predicted probability in bin $k$.
- $\bar{y}(B_k)$ is the empirical fraction of positive outcomes in bin $k$.

### 5. Maximum Calibration Error (MCE)
The worst-case bin deviation across the curve:

$$\text{MCE} = \max_{k \in \{1, \dots, K\}} \left| \bar{q}(B_k) - \bar{y}(B_k) \right|$$

---

## Reporting Slices
Calibration observations are stored immutably in SQLite and evaluated across four dimensions:
1. **By Model Version**: Comparing parameter iterations (e.g., `ilsa-base-v1.0` vs. future iterations).
2. **By Category**: Detecting domain-specific biases (e.g., TECH vs. POLITICS vs. MACRO).
3. **By Edge Bucket**: Evaluating whether larger perceived edges correspond to superior calibration.
4. **By Survival State**: Ensuring forecast quality is maintained during market stress and drawdown.
