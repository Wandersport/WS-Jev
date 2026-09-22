# Quantitative Research Notes & Limitations

**THIS SYSTEM IS A RESEARCH PROTOTYPE &mdash; NOT A PROFITABLE TRADING SYSTEM**

Quantitative research in prediction markets requires rigorous awareness of market microstructure realities, statistical biases, and structural limitations. This document outlines why simulated results differ from live trading and explains why claims of extreme short-term compounding are fundamentally invalid research targets.

---

## 1. Simulated vs. Real Fills

In live prediction markets, order fills are subject to adverse selection and competition:
- **Passive Fill Illusion**: Assuming limit orders get filled without moving the market is unrealistic. In reality, orders are filled primarily when informed counterparties know the quote is mispriced.
- **Queue Priority**: Real order books prioritize speed and queue position. A simulation walking visible depth ignores front-running and queue cancellation.
- **Liquidity Illusion**: Visible order book depth is frequently ephemeral. Market makers cancel quotes within milliseconds upon news arrival.

[`PaperBroker`](../src/pm_research/execution/paper_broker.py) mitigates this by walking conservative ask depth and adding adverse slippage penalties, but paper fills will always be more favorable than live execution.

---

## 2. Microstructure: Spread, Slippage, and Fees

- **Wide Effective Spreads**: In prediction markets, bid-ask spreads often range from 2% to 15% of contract value. A round-trip crossing of the spread eliminates theoretical edge.
- **Slippage on Scale**: Attempting to deploy meaningful capital immediately moves thin order books.
- **Fee Friction**: Exchange fees and gas costs create a high hurdle rate that erodes modest statistical edges.

---

## 3. Epistemic Uncertainty & Miscalibration

No model probability $\hat{q}$ represents ground truth.
- **Overconfidence**: Neural networks and heuristic estimators frequently produce probabilities clustered near 0 or 1 without proportional empirical accuracy.
- **Regime Shift**: Political and macro events exhibit fat-tailed non-ergodic distributions where historical frequencies do not reflect future likelihoods.
- **Uncertainty Margin**: The [`Kett`](../src/pm_research/pipeline/kett.py) module explicitly deducts $z \cdot \sigma_q$ from raw edge to demand higher conviction before proposing capital allocation.

---

## 4. Methodological Biases

1. **Look-Ahead Bias**: Utilizing information in feature extraction that was not strictly timestamped prior to decision time. The system enforces UTC normalization and cycle timestamps to prevent this.
2. **Survivorship & Selection Bias**: Analyzing only active or popular markets ignores unresolved, illiquid, or canceled markets where capital was permanently locked.
3. **Resolution Ambiguity**: Prediction markets rely on oracle resolution or human governance (e.g., UMA, Polymarket resolvers). Ambiguous question phrasing, subjective outcomes, or malicious disputes introduce catastrophic oracle risk that cannot be hedged mathematically.
4. **Correlation & Clustered Risk**: Contracts across geopolitical or macroeconomic themes (e.g., interest rate decisions across different maturities) share hidden common factors. A portfolio of seemingly distinct positions can experience simultaneous total wipeout.

---

## 5. Signal Decay & Scaling Limits

- **Capacity Constraints**: The total open interest in most prediction markets is small (often under $100,000 per market). A strategy with $1,000 in simulated capital cannot scale to $50,000 without destroying the market price.
- **Alpha Decay**: As automated participants enter prediction markets, obvious mispricings between related contracts vanish rapidly.

---

## 6. Why Extreme Short-Term Return Claims Are Invalid Benchmarks

Social media narratives occasionally claim extreme short-term returns (e.g., turning $50 into thousands of dollars within hours). In quantitative finance, such outcomes represent:
- **Maximum Kelly / Gambler's Ruin**: Taking highly leveraged, unhedged binary bets on longshots. Such paths have a high probability of total ruin ($P(\text{ruin}) \approx 100\%$ over long horizons).
- **Extreme Variance, Zero Alpha**: A coin-flipping strategy betting 100% of bankroll will occasionally win 6 times in a row, producing a $64\times$ return by pure chance while having negative expected value.
- **Non-Reproducibility**: Such results cannot be reproduced systematically, cannot manage risk, and violate capital preservation principles.

This research project firmly rejects aggressive compounding targets, prioritizing **capital preservation, probability calibration, risk gating (Bram), and reproducible scientific inquiry**.
