"""Tests for Phase 9B.2 Probability Integrity Audit and Phase 9C Prospective Protocol Preregistration.

Verifies:
1. Exact raw probability vector extraction and sum validation across 480 responses.
2. Jev confidence formula verification: confidence = (p_chosen - 1/3) / (1 - 1/3).
3. Canonical conditional Brier score (0.3484) and LogLoss (0.9342) on retrospective holdout.
4. Frozen discrete classification metrics matching historical audit (38.54% acc, 45.00% bal acc).
5. Phase 9C candidate task specs (Task A displacement, Task B lead coherence, Task C realized vol).
6. Phase 9C preregistration manifests, viability gates, and zero-execution guarantees.
7. Budget limits: 0 calls on train/calibration, <= 300 holdout calls, <= $1.00 ceiling.
8. CLI command execution for jev-phase9b-probability-audit and jev-phase9c-preregister.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from pm_research.cli import main
from pm_research.research.jev_phase9b_audit.probability_audit import (
    Phase9bProbabilityAuditEngine,
    generate_phase9b_probability_audit_reports,
)
from pm_research.research.jev_phase9c.preregistration import (
    generate_phase9c_preregistration_artifacts,
)
from pm_research.research.jev_phase9c.spec import (
    TASK_A_ALL_CHOICES,
    TASK_A_ID,
    TASK_B_ALL_CHOICES,
    TASK_B_ID,
    TASK_C_ALL_CHOICES,
    TASK_C_ID,
    get_task_a_spec,
    get_task_b_spec,
    get_task_c_spec,
)

# ---------------------------------------------------------------------------
# Phase 9B.2 Probability Integrity Audit Tests
# ---------------------------------------------------------------------------


def test_phase9b2_raw_probability_vectors_integrity():
    """Verify all 480 persisted responses have valid raw probability vectors summing to 1.0."""
    engine = Phase9bProbabilityAuditEngine(db_path="data/pm_research.db")
    res = engine.audit_raw_probability_vectors()

    assert res["status"] == "PASS"
    assert res["responses_checked"] == 480
    assert res["full_vectors_present"] == 480
    assert res["missing_vectors"] == 0
    assert res["invalid_vectors"] == 0
    assert res["choice_count_k"] == 3
    assert res["choices_present"] == ["ABSTAIN", "MEANINGFUL_MOVE", "QUIET"]
    assert res["max_vector_sum_error"] <= 0.01

    # Formula check
    f_val = res["formula_validation"]
    assert f_val["formula_confirmed"] is True
    assert f_val["k_value"] == 3
    assert f_val["max_formula_error"] <= 0.02
    assert f_val["scalar_confidence_used_as_probability"] is False


def test_phase9b2_canonical_probability_metrics():
    """Verify canonical probability metrics against frozen holdout ground truth."""
    engine = Phase9bProbabilityAuditEngine(db_path="data/pm_research.db")
    metrics = engine.compute_canonical_probability_metrics()

    holdout = metrics["retrospective_holdout_20pct"]
    assert holdout["n_samples"] == 96

    # Discrete metrics remain frozen
    disc = holdout["discrete_metrics"]
    assert disc["accuracy"] == pytest.approx(0.3854, abs=1e-3)
    assert disc["balanced_accuracy"] == pytest.approx(0.4500, abs=1e-3)
    assert disc["sensitivity_meaningful_recall"] == pytest.approx(0.3614, abs=1e-3)
    assert disc["specificity_quiet_recall"] == pytest.approx(0.5385, abs=1e-3)
    assert disc["macro_f1"] == pytest.approx(0.3480, abs=1e-3)
    assert disc["mcc"] == pytest.approx(-0.0707, abs=1e-3)

    # Probabilistic metrics
    pm = holdout["probabilistic_metrics"]
    cond = pm["canonical_active_class_conditional"]
    assert cond["brier_score"] == pytest.approx(0.348376, abs=1e-4)
    assert cond["log_loss"] == pytest.approx(0.934242, abs=1e-4)

    uncond = pm["unconditional_raw"]
    assert uncond["brier_score"] == pytest.approx(0.404816, abs=1e-4)
    assert uncond["log_loss"] == pytest.approx(1.069447, abs=1e-4)


def test_phase9b2_report_generation(tmp_path):
    """Verify generation of all Phase 9B.2 report artifacts without network calls."""
    out_dir = tmp_path / "p9b2_reports"
    with patch("urllib.request.urlopen") as mock_url:
        reports = generate_phase9b_probability_audit_reports(output_dir=out_dir, db_path="data/pm_research.db")
        assert mock_url.call_count == 0

    assert len(reports) == 5
    assert (out_dir / "raw_probability_integrity.json").is_file()
    assert (out_dir / "confidence_formula_validation.json").is_file()
    assert (out_dir / "corrected_probability_metrics.json").is_file()
    assert (out_dir / "comparison_of_metric_versions.json").is_file()
    assert (out_dir / "README.md").is_file()


# ---------------------------------------------------------------------------
# Phase 9C Prospective Protocol Specification Tests
# ---------------------------------------------------------------------------


def test_phase9c_candidate_task_specs():
    """Verify Phase 9C task specifications, choices, and cryptographic hashes."""
    spec_a = get_task_a_spec()
    assert spec_a["task_id"] == TASK_A_ID
    assert spec_a["target_horizon_seconds"] == 30
    assert set(spec_a["all_choices"]) == set(TASK_A_ALL_CHOICES)
    assert len(spec_a["sha256"]) == 64
    assert spec_a["zero_execution_constraints"]["paper_broker_calls"] == 0

    spec_b = get_task_b_spec()
    assert spec_b["task_id"] == TASK_B_ID
    assert spec_b["reaction_horizon_seconds"] == 10
    assert spec_b["dislocation_threshold_bps"] == 3.0
    assert set(spec_b["all_choices"]) == set(TASK_B_ALL_CHOICES)
    assert len(spec_b["sha256"]) == 64
    assert spec_b["zero_execution_constraints"]["paper_broker_calls"] == 0

    spec_c = get_task_c_spec()
    assert spec_c["task_id"] == TASK_C_ID
    assert spec_c["horizon_seconds"] == 60
    assert set(spec_c["all_choices"]) == set(TASK_C_ALL_CHOICES)
    assert len(spec_c["sha256"]) == 64
    assert spec_c["zero_execution_constraints"]["paper_broker_calls"] == 0


def test_phase9c_preregistration_artifacts_generation(tmp_path):
    """Verify Phase 9C preregistration files generation, composite hash, and constraints."""
    out_dir = tmp_path / "p9c_prereg"
    artifacts = generate_phase9c_preregistration_artifacts(output_dir=out_dir)

    assert len(artifacts) == 9
    for path in artifacts.values():
        assert path.is_file()

    # Verify family manifest
    fam_data = json.loads((out_dir / "family_manifest.json").read_text(encoding="utf-8"))
    assert fam_data["family_id"] == "jev_phase9c_prospective_family_v2"
    assert len(fam_data["family_sha256"]) == 64
    assert fam_data["execution_safety"]["paper_broker_calls"] == 0
    assert fam_data["execution_safety"]["live_trading_capability"] is False

    # Verify budget
    bud_data = json.loads((out_dir / "api_budget.json").read_text(encoding="utf-8"))
    assert bud_data["remote_llm_call_policy"]["train_rounds_calls_allowed"] == 0
    assert bud_data["remote_llm_call_policy"]["calibration_rounds_calls_allowed"] == 0
    assert bud_data["remote_llm_call_policy"]["holdout_rounds_calls_allowed"] == 300
    assert bud_data["remote_llm_call_policy"]["cost_ceiling_usd"] == 1.00

    # Verify viability rules
    viab_data = json.loads((out_dir / "viability_rules.json").read_text(encoding="utf-8"))
    assert viab_data["gates"]["sample_size"]["min_observations_task_a"] >= 200
    assert viab_data["gates"]["sample_size"]["min_qualifying_rounds_task_b"] >= 100
    assert viab_data["gates"]["class_balance"]["min_class_representation_pct"] == 10.0
    assert viab_data["structural_execution_mandates"]["paper_broker_calls"] == 0

    # Verify evaluation plan multiplicity control
    eval_data = json.loads((out_dir / "evaluation_plan.json").read_text(encoding="utf-8"))
    assert eval_data["multiplicity_control"]["family_hypotheses_k"] == 3
    assert eval_data["multiplicity_control"]["adjusted_alpha"] == pytest.approx(0.05 / 3.0)


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------


def test_cli_jev_phase9b_probability_audit(tmp_path):
    """Verify CLI subcommand jev-phase9b-probability-audit runs successfully."""
    out_dir = tmp_path / "cli_p9b2"
    ret = main(["--db", "data/pm_research.db", "jev-phase9b-probability-audit", "--output-dir", str(out_dir)])
    assert ret == 0
    assert (out_dir / "README.md").is_file()


def test_cli_jev_phase9c_preregister(tmp_path):
    """Verify CLI subcommand jev-phase9c-preregister runs successfully."""
    out_dir = tmp_path / "cli_p9c"
    ret = main(["jev-phase9c-preregister", "--output-dir", str(out_dir)])
    assert ret == 0
    assert (out_dir / "family_manifest.json").is_file()
