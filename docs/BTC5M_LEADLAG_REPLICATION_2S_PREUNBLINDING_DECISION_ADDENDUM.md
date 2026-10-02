# Phase 8E Lead-Lag v3 Confirmatory Replication: Pre-Unblinding Statistical Decision Addendum

- **Document Path**: `docs/BTC5M_LEADLAG_REPLICATION_2S_PREUNBLINDING_DECISION_ADDENDUM.md`
- **Creation Timestamp**: `2026-10-02T19:55:00Z`
- **Canonical Experiment ID**: `btc5m_leadlag_v3_replication_2s`
- **Original Frozen Spec**: `docs/BTC5M_LEADLAG_REPLICATION_2S_SPEC.md`
- **Original Frozen Spec Hash**: `bfe5b0553a7143dd8941239dd481cf0e86499b54338dc6717565d1d9dc28fa53`
- **Original Frozen Spec Status**: **STRICTLY UNCHANGED**
- **Frozen Model Artifact**: `models/frozen_v2_replication_models_2s.json`
- **Frozen Model Hash**: `ee914a59072320d142944530957036c325632f941432d934f472e689dc7f1c77`
- **Measurement Audit Reference**: `reports/BTC5M_LEADLAG_V3_CONFIRMATORY_MEASUREMENT_AUDIT.md` (Commit: `19a470b`)
- **Outcome Blinding Status**: **STRICTLY INTACT** (Zero predictive metrics, predictions, targets, or B0/B1 comparisons have been computed or inspected)

---

## Context & Purpose
This statistical decision addendum is established after the physical completion of prospective data collection (250/250 rounds, 75,000/75,000 ticks) and after successful measurement-integrity validation, but strictly **PRIOR** to the computation or unblinding of any replication predictive outcome. 

Because the original prospective specification (`docs/BTC5M_LEADLAG_REPLICATION_2S_SPEC.md`) specified the model features, targets, and physical collection rules but omitted formal inferential decision boundaries, aggregation weighting rules, and compound success criteria, this document formalizes those decision rules in an immutable addendum before unblinding to eliminate all post-hoc researcher degrees of freedom.

---

## 1. Endpoint Hierarchy
1. **Primary Confirmatory Endpoint**:
   - `delta_q_2s` ($\Delta q_{2s} = q_{t+2000\text{ms}} - q_t$, native Polymarket UP token midpoint change).
   - This endpoint alone governs formal replication confirmation.
2. **Secondary Supportive Endpoint**:
   - `delta_logit_q_2s` ($\Delta \text{logit}(q)_{2s} = \text{logit}(q_{t+2000\text{ms}}) - \text{logit}(q_t)$ with clamping $\epsilon = 10^{-6}$).
   - This endpoint is supportive and descriptive. It cannot rescue a failed primary endpoint.
3. **Multiplicity Treatment**:
   - Under this strict hierarchical endpoint ordering, no formal multiplicity adjustment (e.g., $\alpha$-splitting or Holm-Bonferroni across targets) is required. The primary confirmatory gate is tested at full $\alpha = 0.05$.
   - A co-primary interpretation is explicitly rejected.

---

## 2. Primary Estimand
1. For the frozen **NO-REFIT transport test**, for each physical round $r \in \{1, \dots, 250\}$, compute the round-level loss improvement:
   $$d_r = \text{MSE}_{B0, r} - \text{MSE}_{B1, r}$$
   where positive $d_r$ indicates that the frozen augmented model $B_1$ achieves lower squared error than the frozen baseline model $B_0$ on physical round $r$.
2. **Primary Estimand**:
   The equal-weight mean of $d_r$ across all 250 physical rounds:
   $$\bar{d} = \frac{1}{250} \sum_{r=1}^{250} d_r$$
   *Rationale*: The physical round is the independent sampling and clustering unit. Equal-round weighting prevents rounds with marginally higher sample counts from dominating inference.
3. **Role of Pair-Weighting**:
   Pair-weighted pooled MSE improvement ($\text{MSE}_{B0, \text{pooled}} - \text{MSE}_{B1, \text{pooled}}$) is a mandatory consistency diagnostic, but is not the primary inferential estimand.

---

## 3. Primary Inferential Rules
1. **Sample Unit & Parameters**:
   - Clusters: $N = 250$ physical rounds.
   - Significance level: $\alpha = 0.05$.
   - Confidence level: $95\%$.
   - Predeclared expected direction: $B_1 \text{ improvement} > 0$.
   - Two-sided test formulation used for conservatism despite predeclared directionality.
2. **Required Tests**:
   - **Test A: Paired One-Sample Round-Level t-Test**:
     Calculated on the round-level series $\{d_r\}_{r=1}^{250}$:
     $$t = \frac{\bar{d}}{s_d / \sqrt{250}}, \quad \text{two-sided } p < 0.05$$
   - **Test B: Deterministic Round-Cluster Bootstrap**:
     - Resampling unit: Entire physical round (cluster bootstrap).
     - Replicates: $B = 10,000$.
     - Random seed: `20261002`.
     - Confidence interval: Percentile $95\%$ CI of $\bar{d}$ ($[2.5\text{th percentile}, 97.5\text{th percentile}]$).
3. **Primary Inferential PASS Criteria**:
   Formal inferential pass requires that **all three** conditions hold:
   1. $\bar{d} > 0$
   2. Paired round t-test two-sided $p < 0.05$
   3. Bootstrap $95\%$ CI lower bound $> 0$

---

## 4. Required Directional & Robustness Conditions
In addition to primary statistical inference, replication confirmation requires all of the following:
1. **Pair-Weighted Consistency**: Pair-weighted pooled MSE improvement $> 0$.
2. **Equal-Round Consistency**: Equal-round mean MSE improvement $\bar{d} > 0$.
3. **Primary Timing Gate**: Frozen primary timing gate $\le 500$ ms passes.
4. **Timing Sensitivity Invariance**: Sensitivity gate $\le 250$ ms exhibits the same positive direction ($\bar{d}_{250\text{ms}} > 0$).
5. **Temporal Stability**:
   Chronological partition of the 250 rounds into three prospectively defined blocks:
   - Early: rounds 1–83
   - Middle: rounds 84–166
   - Late: rounds 167–250
   *Requirement*: $\text{MSE}_{B0} - \text{MSE}_{B1} > 0$ across all 3 chronological blocks (statistical significance within individual sub-blocks is not required).
6. **Causal & Measurement Integrity**:
   - Zero future-information leakage (verified).
   - Zero pilot or v2 contamination (verified).
   - Zero cadence confound or measurement-integrity failure (verified).

---

## 5. Effect Size Interpretation
1. No arbitrary minimum $\Delta R^2$ magnitude threshold is imposed.
2. No requirement that prospective $v_3$ effect size equals discovery $v_2$ effect size.
3. Descriptive reporting:
   - Frozen $v_2$ discovery effect
   - Prospective $v_3$ replication effect
   - Effect-size ratio ($v_3 / v_2$)
4. Effect shrinkage or attenuation alone does not fail replication if all confirmatory inferential and robustness criteria are met.

---

## 6. Formal Replication Verdict Rules
The replication outcome shall be mapped to one of the following mutually exclusive verdicts:

- **`CONFIRMED`**:
  All primary $\Delta q_{2s}$ criteria are satisfied:
  - $\bar{d} > 0$ (equal-round mean improvement is positive).
  - Paired round t-test $p < 0.05$.
  - Round-cluster bootstrap 95% CI lower bound $> 0$.
  - Pair-weighted pooled MSE improvement $> 0$.
  - Timing sensitivity ($\le 250$ ms) improvement $> 0$.
  - Early, Middle, and Late blocks all demonstrate positive improvement.
  - All causal, measurement, and leakage integrity gates pass.

- **`PARTIALLY_REPLICATED`**:
  Primary $\Delta q_{2s}$ direction is positive in both equal-round ($\bar{d} > 0$) and pair-weighted analyses, but one or more secondary inferential or robustness conditions (e.g., $p \ge 0.05$, bootstrap CI includes zero, or one temporal block is non-positive) fail.

- **`NOT_REPLICATED`**:
  Either:
  - Equal-round primary improvement $\bar{d} \le 0$, OR
  - Pair-weighted pooled improvement $\le 0$, OR
  - Effect direction reverses materially in prospective data.

- **`INVALID_REPLICATION`**:
  Assigned exclusively if an unresolvable data collection, leakage, or structural integrity defect invalidates the statistical interpretability of the dataset.

Secondary endpoint $\Delta \text{logit}(q)_{2s}$ is evaluated and reported under the identical procedure as supportive evidence, but does not alter the formal confirmatory verdict.

---

## 7. Role of $\Delta R^2$
$\Delta R^2$ ($R^2_{B1} - R^2_{B0}$) is an effect-size summary metric. The formal confirmatory decision is strictly governed by physical-round predictive loss improvement $d_r$. $\Delta R^2$ shall not be used as an additional un-preregistered screening gate.

---

## 8. Immutability Commitment
This addendum is committed to Git and pushed to remote before any script or function executes predictions or reads outcome labels from the $v_3$ replication database. Once committed, these decision rules are completely frozen.
