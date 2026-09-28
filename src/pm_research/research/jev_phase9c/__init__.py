"""Phase 9C: Prospective Jev Decision Research Protocol (`btc5m_leadlag_v2`).

Contains frozen prospective candidate task specifications, preregistration manifests,
and cryptographic integrity hashing for the future 500-round v2 dataset evaluation.
"""

from __future__ import annotations

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

__all__ = [
    "TASK_A_ID",
    "TASK_B_ID",
    "TASK_C_ID",
    "TASK_A_ALL_CHOICES",
    "TASK_B_ALL_CHOICES",
    "TASK_C_ALL_CHOICES",
    "get_task_a_spec",
    "get_task_b_spec",
    "get_task_c_spec",
    "generate_phase9c_preregistration_artifacts",
]
