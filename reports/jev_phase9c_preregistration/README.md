# Phase 9C: Prospective Jev Decision Protocol Preregistration

- **Protocol Status**: FROZEN PREREGISTRATION
- **Family ID**: `jev_phase9c_prospective_family_v2`
- **Composite Family Hash**: `3c773ae8c140be4a6af14be2be234e040ee94aa1c3c3e1baf07441edba1cc44a`
- **Dataset Target**: Future `btc5m_leadlag_v2` (500 Physical Rounds)
- **Structural Execution Capability**: ZERO (Positions=0, Orders=0, Fills=0, PaperBroker Calls=0)

---

## 1. Candidate Prospective Tasks

| Task | ID | Horizon | Choices | Task SHA-256 |
|---|---|---|---|---|
| **Task A** | `poly_displacement_regime_30s_v2` | 30s | UP, DOWN, BENIGN, ABSTAIN | `6f14dddc8ccb8b6d...` |
| **Task B** | `cross_venue_lead_coherent_10s_v2` | 10s | COHERENT, NOT_COHERENT, ABSTAIN | `cb051016b8fa3d97...` |
| **Task C** | `short_horizon_regime_60s_v2` | 60s | HIGH_VOL, LOW_VOL, ABSTAIN | `5a9ab2e88000d30e...` |

---

## 2. API Call Budget & Split Rules

* **Train / Development (Rounds 1–300)**: 0 remote LLM calls allowed.
* **Calibration (Rounds 301–400)**: 0 remote LLM calls allowed.
* **Prospective Holdout (Rounds 401–500)**: $\le 100$ calls/task, $\le 300$ calls total.
* **Hard Cost Ceiling**: $\le \$1.00$ USD total.

---

## 3. Multiple Testing & Viability Gates

* **Bonferroni Adjusted Significance**: $\alpha = 0.05 / 3 \approx 0.0166667$.
* **Pre-Evaluation Viability Gates**:
  * Minimum observations: $\ge 200$ (Tasks A & C), $\ge 100$ qualifying rounds (Task B).
  * Minimum class frequency: $\ge 10\%$ for every required active class.
  * Microstructure quality: $\ge 95\%$ Polymarket source TS, $\ge 99\%$ Binance source TS, $\ge 95\%$ taker flow post-warmup.

---

## 4. Probabilistic Scoring Standard

* Direct extraction of exact probability vectors from `answers.decision.probabilities`.
* Primary metric: Active-Class Conditional Probability ($p_{k,\text{cond}} = p_k / \sum_{j \in \text{active}} p_j$).
* Secondary metric: Unconditional Probability ($p_k$).
* Strict verification of confidence formula: $\text{confidence} = (p_{\text{chosen}} - 1/K) / (1 - 1/K)$.
