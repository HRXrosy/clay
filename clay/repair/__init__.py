"""
clay.repair — repair-operator runtime (the REPAIR ALGEBRA).

Closes the "Repair completeness" gap from
paper/clay-tcad/clay_language_completeness_theory.html: CLAY already emits
structured :class:`~clay.invariants_modules._violation.InvariantViolation`
envelopes with gradient hints, but had no runtime that maps a violation to
a *legal repair operator*, sizes a bounded knob move, measures distance-to-
satisfaction, and drives a terminating repair loop.

This package is self-contained — it imports the invariant runtime (the
violation dataclass + the Newton-step convention) READ-ONLY and adds no
hooks to existing modules. Repair status lives on an axis strictly
separate from verification status; a verification FAIL stays FAIL even
when no repair exists, and no repair status — not even ``REPAIRED`` —
certifies a verification PASS.

Public surface:

  Status
    * :class:`RepairStatus` — REPAIRABLE / REPAIR_PENDING /
      REPAIR_UNSUPPORTED / NO_REPAIR_OPERATOR / REPAIRED.

  Registry
    * :class:`RepairClass`, :class:`RepairOperator`,
      :class:`RepairRegistry`, :func:`default_registry`.

  Proposal
    * :class:`RepairProposal`, :func:`propose_repair`,
      :data:`DEFAULT_TRUST_REGION`.

  Distance / safety
    * :func:`violation_distance`, :func:`repair_is_locally_safe`.

  Loop
    * :class:`RepairLoop`, :class:`RepairResult`, :class:`RepairStep`,
      :class:`StopReason`, :data:`DEFAULT_RETRY_BUDGET`.
"""

from __future__ import annotations

from ._distance import repair_is_locally_safe, violation_distance
from ._loop import (
    DEFAULT_RETRY_BUDGET,
    RepairLoop,
    RepairResult,
    RepairStep,
    StopReason,
)
from ._proposal import (
    DEFAULT_TRUST_REGION,
    RepairProposal,
    propose_repair,
)
from ._registry import (
    RepairClass,
    RepairOperator,
    RepairRegistry,
    default_registry,
)
from ._status import RepairStatus

__all__ = [
    # status
    "RepairStatus",
    # registry
    "RepairClass",
    "RepairOperator",
    "RepairRegistry",
    "default_registry",
    # proposal
    "RepairProposal",
    "propose_repair",
    "DEFAULT_TRUST_REGION",
    # distance / safety
    "violation_distance",
    "repair_is_locally_safe",
    # loop
    "RepairLoop",
    "RepairResult",
    "RepairStep",
    "StopReason",
    "DEFAULT_RETRY_BUDGET",
]
