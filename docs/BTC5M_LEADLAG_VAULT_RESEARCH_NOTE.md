# Architecture & Methodological Governance Note: Opus, Jev, and the Strategy Vault

- **Date**: 2026-09-28
- **Status**: Future Research Design Note (Non-Operational)
- **Context**: Post-Phase 8C.1 Forensic Audit of `btc5m_leadlag_v1`

---

## 1. Separation of Responsibilities

In future quantitative research, four components must remain strictly separated by architectural firewalls:

```
+-------------------------------------------------------+
| 1. Strategy Vault (Hypothesis Source Only)            |
|    - 5,806 candidate strategies                       |
|    - Untested, unvalidated idea library               |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 2. Opus (Hypothesis Generation / Research / Coding)   |
|    - Formulates falsifiable mathematical hypotheses   |
|    - Writes clean evaluation & feature code           |
|    - MUST NOT evaluate or grade its own models        |
+-------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------+
| 3. Deterministic Evaluator (Independent Rejection)    |
|    - GroupKFold CV grouped by physical round          |
|    - Round-clustered bootstrap confidence intervals   |
|    - Rigid zero-leakage temporal forward testing      |
|    - Hard gate: rejects non-significant improvements  |
+-------------------------------------------------------+
                           | (Only if signal passes gate)
                           v
+-------------------------------------------------------+
| 4. Jev / LLM Typed Judgment Layer                     |
|    - Fast typed contextual adjustment                 |
|    - Strictly downstream of proven statistical signal |
|    - Never used as a substitute for raw price edge    |
+-------------------------------------------------------+
```

---

## 2. Critical Methodological Warnings

1. **Multiple Testing Hazards**:
   - Evaluating 5,806 candidate rules or features without Bonferroni or False Discovery Rate (FDR) control guarantees finding dozens of spurious "alphas" by pure random chance ($p < 0.05$ produces ~290 false discoveries out of 5,806 tests).
   - Any mining across the vault must pre-register hypothesis subsets and apply rigorous family-wise error rate corrections.

2. **Strategy-Selection Bias**:
   - Selecting the top performing strategies on historical data without out-of-sample forward testing selects for sample noise and regime luck rather than durable market microstructure dynamics.

3. **Backtest Overfitting**:
   - Microstructure features that show tiny in-sample correlations rapidly collapse when exposed to realistic transaction costs, spread crossing, and latency jitter.

4. **Self-Evaluation Bias (AI-Grading-AI)**:
   - The AI system that generates code or proposes hypotheses (e.g., Opus) must **never** be the authority that evaluates or validates the results.
   - Validation must be conducted exclusively by an unyielding, deterministic Python statistical test harness with pre-registered metrics, fixed random seeds, and clustered standard errors.

---

## 3. Operational Conclusion for Current Phase

Phase 8C.1 evidence demonstrates that:
1. Binance order book depth features currently provide **no statistically significant incremental predictive power** over Polymarket's own recent price state ($\Delta R^2 < 0.001$, $\Delta \text{MSE} \approx 0$).
2. Measurement-system defects (casing bug on aggTrades, dropped Polymarket timestamps) must be remediated in a future collector (`btc5m_leadlag_v2`) before any meaningful trade flow analysis can occur.
3. Therefore, **beginning Opus/Jev strategy discovery at this time is entirely unjustified and premature**.
4. Scientific integrity requires fixing the measurement apparatus first.
