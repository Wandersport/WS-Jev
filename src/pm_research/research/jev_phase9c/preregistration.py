"""Phase 9C: Prospective Preregistration Generator.

Generates immutable, versioned preregistration JSON artifacts for Phase 9C
in reports/jev_phase9c_preregistration/ without executing any remote API calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pm_research.research.jev_phase9c.spec import (
    TASK_A_ID,
    TASK_B_ID,
    TASK_C_ID,
    get_task_a_spec,
    get_task_b_spec,
    get_task_c_spec,
)


def _compute_sha256_of_dict(data: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 of JSON-serializable dictionary."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_baseline_manifest() -> dict[str, Any]:
    """Preregistered baseline algorithms and training protocol."""
    manifest = {
        "manifest_type": "baseline_manifest",
        "version": "1.0.0",
        "description": "Preregistered transparent and statistical baselines for prospective evaluation",
        "training_policy": "Strict chronological training on Development cohort (rounds 1-300) only; zero leakage",
        "baselines": {
            TASK_A_ID: [
                {
                    "name": "MajorityBaseRate",
                    "type": "TRIVIAL_EMPIRICAL",
                    "description": "Predicts the empirical majority class from the 300 development rounds.",
                },
                {
                    "name": "StandardizedLogisticRegression",
                    "type": "PARAMETRIC_STATISTICAL",
                    "description": "L2-regularized multinomial logistic regression on standardized microstructure features.",
                },
                {
                    "name": "RollingTakerFlowHeuristic",
                    "type": "DOMAIN_HEURISTIC",
                    "description": "Signs and thresholds rolling 5s Binance USD-M aggressive taker imbalance.",
                },
            ],
            TASK_B_ID: [
                {
                    "name": "MajorityBaseRate",
                    "type": "TRIVIAL_EMPIRICAL",
                    "description": "Predicts empirical base rate of lead coherence from development trigger events.",
                },
                {
                    "name": "MicropriceDislocationSignMatch",
                    "type": "DOMAIN_HEURISTIC",
                    "description": "Predicts COHERENT whenever dislocation magnitude exceeds empirical 75th percentile.",
                },
                {
                    "name": "StandardizedLogisticRegression",
                    "type": "PARAMETRIC_STATISTICAL",
                    "description": "L2-regularized logistic regression using book depth imbalance and dislocation magnitude.",
                },
            ],
            TASK_C_ID: [
                {
                    "name": "MedianBaseRate",
                    "type": "TRIVIAL_EMPIRICAL",
                    "description": "Predicts uniform 50% probability based on median split construction.",
                },
                {
                    "name": "PriorWindowVolatilityPersistence",
                    "type": "AUTOREGRESSIVE_HEURISTIC",
                    "description": "Predicts high volatility if prior 60s volatility was above median.",
                },
                {
                    "name": "StandardizedLogisticRegression",
                    "type": "PARAMETRIC_STATISTICAL",
                    "description": "L2-regularized logistic regression on trade arrival count and top-book spread.",
                },
            ],
        },
    }
    manifest["sha256"] = _compute_sha256_of_dict(manifest)
    return manifest


def generate_api_budget() -> dict[str, Any]:
    """Preregistered remote LLM API budget constraints."""
    budget = {
        "manifest_type": "api_budget",
        "version": "1.0.0",
        "target_dataset": "btc5m_leadlag_v2",
        "total_dataset_rounds": 500,
        "splits": {
            "train_rounds": "1-300 (60% cohort)",
            "calibration_rounds": "301-400 (20% cohort)",
            "holdout_rounds": "401-500 (20% cohort)",
        },
        "remote_llm_call_policy": {
            "train_rounds_calls_allowed": 0,
            "calibration_rounds_calls_allowed": 0,
            "holdout_rounds_calls_allowed": 300,
            "max_calls_per_task": 100,
            "max_total_calls": 300,
            "cost_ceiling_usd": 1.00,
            "immediate_halt_on_budget_exceeded": True,
            "idempotent_sqlite_storage": True,
        },
        "model_target": "typesafe/jev-1.13",
    }
    budget["sha256"] = _compute_sha256_of_dict(budget)
    return budget


def generate_viability_rules() -> dict[str, Any]:
    """Preregistered pre-evaluation viability gates and execution constraints."""
    rules = {
        "manifest_type": "viability_rules",
        "version": "1.0.0",
        "description": "Minimum data quality, sample representation, and structural safety gates",
        "gates": {
            "sample_size": {
                "min_observations_task_a": 200,
                "min_qualifying_rounds_task_b": 100,
                "min_observations_task_c": 200,
            },
            "class_balance": {
                "min_class_representation_pct": 10.0,
                "rule": "If any required active class has < 10% representation in holdout, evaluation halts as DEGENERATE_BASE_RATE.",
            },
            "microstructure_integrity": {
                "min_polymarket_source_ts_pct": 95.0,
                "min_binance_source_ts_pct": 99.0,
                "min_taker_flow_coverage_post_warmup_pct": 95.0,
            },
        },
        "structural_execution_mandates": {
            "strictly_paper_research": True,
            "live_trading_capability": False,
            "paper_broker_calls": 0,
            "orders": 0,
            "fills": 0,
            "positions": 0,
            "pnl": 0.0,
            "note": "Phase 9C evaluates probabilistic forecasting quality only; it does not simulate order execution.",
        },
    }
    rules["sha256"] = _compute_sha256_of_dict(rules)
    return rules


def generate_evaluation_plan() -> dict[str, Any]:
    """Preregistered evaluation plan, metrics, and multiple-testing corrections."""
    plan = {
        "manifest_type": "evaluation_plan",
        "version": "1.0.0",
        "primary_benchmark_cohort": "prospective_holdout_rounds_401_to_500",
        "discrete_metrics": {
            "primary": "balanced_accuracy",
            "secondary": ["macro_f1", "matthews_correlation_coefficient", "sensitivity", "specificity", "accuracy"],
        },
        "probabilistic_metrics": {
            "primary": {
                "name": "canonical_active_class_conditional",
                "formula": "p_k_cond = p_k / sum_{j in active} p_j",
                "metrics": ["brier_score", "log_loss"],
            },
            "secondary": {
                "name": "unconditional_raw",
                "formula": "p_k",
                "metrics": ["brier_score", "log_loss"],
            },
            "confidence_validation": {
                "required_formula": "confidence = (p_chosen - 1/K) / (1 - 1/K)",
                "prohibition": "Scalar confidence must NEVER be treated directly as class probability.",
            },
        },
        "multiplicity_control": {
            "method": "Bonferroni",
            "family_hypotheses_k": 3,
            "family_alpha": 0.05,
            "adjusted_alpha": 0.05 / 3.0,
        },
        "bootstrap": {
            "resamples": 1000,
            "blocking_unit": "physical_round",
            "confidence_level": 0.95,
            "metric": "bootstrap_superiority_fraction",
        },
    }
    plan["sha256"] = _compute_sha256_of_dict(plan)
    return plan


def generate_phase9c_preregistration_artifacts(output_dir: str | Path = "reports/jev_phase9c_preregistration") -> dict[str, Path]:
    """Generate all frozen Phase 9C preregistration files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: dict[str, Path] = {}

    # 1. Task Specs
    spec_a = get_task_a_spec()
    spec_b = get_task_b_spec()
    spec_c = get_task_c_spec()

    p_a = out_path / "task_a_spec.json"
    p_a.write_text(json.dumps(spec_a, indent=2), encoding="utf-8")
    generated_files["task_a_spec"] = p_a

    p_b = out_path / "task_b_spec.json"
    p_b.write_text(json.dumps(spec_b, indent=2), encoding="utf-8")
    generated_files["task_b_spec"] = p_b

    p_c = out_path / "task_c_spec.json"
    p_c.write_text(json.dumps(spec_c, indent=2), encoding="utf-8")
    generated_files["task_c_spec"] = p_c

    # 2. Baseline Manifest
    base_man = generate_baseline_manifest()
    p_base = out_path / "baseline_manifest.json"
    p_base.write_text(json.dumps(base_man, indent=2), encoding="utf-8")
    generated_files["baseline_manifest"] = p_base

    # 3. API Budget
    api_bud = generate_api_budget()
    p_bud = out_path / "api_budget.json"
    p_bud.write_text(json.dumps(api_bud, indent=2), encoding="utf-8")
    generated_files["api_budget"] = p_bud

    # 4. Viability Rules
    viab = generate_viability_rules()
    p_viab = out_path / "viability_rules.json"
    p_viab.write_text(json.dumps(viab, indent=2), encoding="utf-8")
    generated_files["viability_rules"] = p_viab

    # 5. Evaluation Plan
    eval_pl = generate_evaluation_plan()
    p_eval = out_path / "evaluation_plan.json"
    p_eval.write_text(json.dumps(eval_pl, indent=2), encoding="utf-8")
    generated_files["evaluation_plan"] = p_eval

    # 6. Family Manifest (links all component artifacts and computes composite hash)
    component_hashes = {
        "task_a_spec": spec_a["sha256"],
        "task_b_spec": spec_b["sha256"],
        "task_c_spec": spec_c["sha256"],
        "baseline_manifest": base_man["sha256"],
        "api_budget": api_bud["sha256"],
        "viability_rules": viab["sha256"],
        "evaluation_plan": eval_pl["sha256"],
    }
    composite_raw = json.dumps(component_hashes, sort_keys=True, separators=(",", ":"))
    family_sha256 = hashlib.sha256(composite_raw.encode("utf-8")).hexdigest()

    family_manifest = {
        "family_id": "jev_phase9c_prospective_family_v2",
        "family_version": "1.0.0",
        "status": "PREREGISTERED_FROZEN",
        "target_dataset": "btc5m_leadlag_v2",
        "tasks": [TASK_A_ID, TASK_B_ID, TASK_C_ID],
        "component_hashes": component_hashes,
        "family_sha256": family_sha256,
        "execution_safety": {
            "positions": 0,
            "orders": 0,
            "fills": 0,
            "paper_broker_calls": 0,
            "live_trading_capability": False,
        },
    }
    p_fam = out_path / "family_manifest.json"
    p_fam.write_text(json.dumps(family_manifest, indent=2), encoding="utf-8")
    generated_files["family_manifest"] = p_fam

    # 7. README.md
    readme_content = f"""# Phase 9C: Prospective Jev Decision Protocol Preregistration

- **Protocol Status**: FROZEN PREREGISTRATION
- **Family ID**: `jev_phase9c_prospective_family_v2`
- **Composite Family Hash**: `{family_sha256}`
- **Dataset Target**: Future `btc5m_leadlag_v2` (500 Physical Rounds)
- **Structural Execution Capability**: ZERO (Positions=0, Orders=0, Fills=0, PaperBroker Calls=0)

---

## 1. Candidate Prospective Tasks

| Task | ID | Horizon | Choices | Task SHA-256 |
|---|---|---|---|---|
| **Task A** | `{TASK_A_ID}` | 30s | UP, DOWN, BENIGN, ABSTAIN | `{spec_a['sha256'][:16]}...` |
| **Task B** | `{TASK_B_ID}` | 10s | COHERENT, NOT_COHERENT, ABSTAIN | `{spec_b['sha256'][:16]}...` |
| **Task C** | `{TASK_C_ID}` | 60s | HIGH_VOL, LOW_VOL, ABSTAIN | `{spec_c['sha256'][:16]}...` |

---

## 2. API Call Budget & Split Rules

* **Train / Development (Rounds 1–300)**: 0 remote LLM calls allowed.
* **Calibration (Rounds 301–400)**: 0 remote LLM calls allowed.
* **Prospective Holdout (Rounds 401–500)**: $\\le 100$ calls/task, $\\le 300$ calls total.
* **Hard Cost Ceiling**: $\\le \\$1.00$ USD total.

---

## 3. Multiple Testing & Viability Gates

* **Bonferroni Adjusted Significance**: $\\alpha = 0.05 / 3 \\approx 0.0166667$.
* **Pre-Evaluation Viability Gates**:
  * Minimum observations: $\\ge 200$ (Tasks A & C), $\\ge 100$ qualifying rounds (Task B).
  * Minimum class frequency: $\\ge 10\\%$ for every required active class.
  * Microstructure quality: $\\ge 95\\%$ Polymarket source TS, $\\ge 99\\%$ Binance source TS, $\\ge 95\\%$ taker flow post-warmup.

---

## 4. Probabilistic Scoring Standard

* Direct extraction of exact probability vectors from `answers.decision.probabilities`.
* Primary metric: Active-Class Conditional Probability ($p_{{k,\\text{{cond}}}} = p_k / \\sum_{{j \\in \\text{{active}}}} p_j$).
* Secondary metric: Unconditional Probability ($p_k$).
* Strict verification of confidence formula: $\\text{{confidence}} = (p_{{\\text{{chosen}}}} - 1/K) / (1 - 1/K)$.
"""
    p_readme = out_path / "README.md"
    p_readme.write_text(readme_content, encoding="utf-8")
    generated_files["readme"] = p_readme

    return generated_files
