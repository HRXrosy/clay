"""
Clay optimizer harness (v0.0.67 Phase 3 first slice).

Python-side scaffolding for the v0.0.62 roadmap §3 two-stage optimizer
pipeline (Stage 1 tSS-BO global explorer + Stage 2 Adjoint Network
local gradient convergence). This module ships ONLY the harness — the
actual tSS-BO library (github.com/PascalInn/tSS-BO) integration is
deferred to v0.0.68+; the Adjoint Network is even later.

Public surface (intentionally narrow):

  * :class:`harness.OptimizerInterface` — Protocol every optimizer
    implements. Two methods: ``propose(history) -> Sizing`` and
    ``update(history, sizing, result) -> None``. Pure Python so the
    real tSS-BO wrapper (v0.0.68+) and the deterministic stubs below
    share one contract.
  * :class:`harness.Sizing` — a frozen dataclass; ordered ``dict[str,
    float]`` of knob name → physical value. The cell builder consumes
    this as kwargs (``gm_id_diff=...``, ``W_diff=...``, etc.).
  * :class:`harness.OptimizerHistory` — append-only record of every
    iteration with sizing / measures / violations / feasibility / reward.
  * :class:`harness.OptimizerLoop` — orchestrator. Build cell → Phase 2
    pre-filter → ngspice → reward. Hard-fails closed: the loop never
    aborts on a single iteration's exception; it records the failure
    and moves on so the optimizer can learn from negative samples.
  * :class:`harness.SpecTarget` — the (name, op, value) tuple the
    reward function tries to satisfy. Multiple targets aggregate.
  * :func:`harness.default_reward` — clipped, dimensionless distance-to-
    target reward. The choice is documented inline; tSS-BO + the
    Adjoint path will both plug in their own reward shapers.

  * :mod:`stub_optimizers` — three deterministic optimizers shipped for
    contract validation and the CLI sanity path: ``RandomSearchStub``
    (uniform random within bounds, seeded), ``GridSearchStub`` (regular
    enumeration over knob bounds), ``GradientFreeStub`` (returns the
    same sizing every iteration — a no-op).

  * :mod:`run` — ``python -m tools.clay_optimize`` CLI. Builds a cell
    from registered template + target spec, runs the harness for
    ``--max-iters`` iterations, prints a history table.

What this slice does NOT ship:

  * The real tSS-BO library integration (v0.0.68+).
  * The Adjoint Network gradient extractor (v0.0.69+).
  * MDP / Gym-like wrapping (Phase 4 / v0.0.70+).
  * Any production sizing search — the stubs are NOT the optimizer,
    they're the smoke-test surface to validate the contract.
"""

from __future__ import annotations

__all__ = [
    "OptimizerInterface",
    "OptimizerHistory",
    "OptimizerHistoryEntry",
    "OptimizerLoop",
    "Sizing",
    "SizingBounds",
    "SpecTarget",
    "default_reward",
    "RandomSearchStub",
    "GridSearchStub",
    "GradientFreeStub",
    # v0.0.68 Phase 3 second slice — real tSS-BO adapter (opt-in).
    "TSsBoAdapter",
    "BoBudgetTracker",
    "BudgetExceededError",
    "build_tssbo_adapter",
    "build_tssbo_adapter_from_intent",
    "TSSBO_OPT_IN_ENV",
    "ExactCandidateFeedback",
    "PhysicalSearchVector",
    "classify_exact_search_outcome",
    "exact_search_trajectory_data",
    "refine_measured_candidates",
    "select_exact_candidate",
    "select_exact_replay_candidates",
    "select_routability_cohort",
    "PhysicalIntentCandidateEvidence",
    "PhysicalIntentInfeasible",
    "PhysicalIntentSearchResult",
    "physical_intent_state_sha256",
    "prepare_physical_intent_search",
    "PHYSICAL_SEARCH_TARGET_SCHEMA",
    "PHYSICAL_SEARCH_ADMISSION_SCHEMA",
    "PhysicalSearchAdmission",
    "PhysicalSearchRegion",
    "PhysicalSearchRelation",
    "PhysicalSearchTarget",
    "PhysicalSearchTargetError",
    "evaluate_physical_search_admission",
    "load_physical_search_target",
]

from .harness import (
    OptimizerHistory,
    OptimizerHistoryEntry,
    OptimizerInterface,
    OptimizerLoop,
    Sizing,
    SizingBounds,
    SpecTarget,
    default_reward,
)
from .stub_optimizers import (
    GradientFreeStub,
    GridSearchStub,
    RandomSearchStub,
)
from .budget import BoBudgetTracker, BudgetExceededError
from .tssbo_adapter import (
    OPT_IN_ENV as TSSBO_OPT_IN_ENV,
    TSsBoAdapter,
    build_tssbo_adapter,
    build_tssbo_adapter_from_intent,
)
from .physical_intent_search import (
    ExactCandidateFeedback,
    PhysicalSearchVector,
    PhysicalIntentCandidateEvidence,
    PhysicalIntentInfeasible,
    PhysicalIntentSearchResult,
    classify_exact_search_outcome,
    exact_search_trajectory_data,
    physical_intent_state_sha256,
    prepare_physical_intent_search,
    refine_measured_candidates,
    select_exact_candidate,
    select_exact_replay_candidates,
    select_routability_cohort,
)
from .physical_search_target import (
    PHYSICAL_SEARCH_ADMISSION_SCHEMA,
    PHYSICAL_SEARCH_TARGET_SCHEMA,
    PhysicalSearchAdmission,
    PhysicalSearchRegion,
    PhysicalSearchRelation,
    PhysicalSearchTarget,
    PhysicalSearchTargetError,
    evaluate_physical_search_admission,
    load_physical_search_target,
)
