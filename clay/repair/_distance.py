"""
Distance-to-satisfaction over a set of invariant violations, plus the
local-safety predicate used by the repair loop.

``violation_distance`` is the *repair-runtime* view of how far a design
is from satisfying its declared obligations: a weighted sum of per-
violation distances ``d_i``, where ``d_i`` is the relative error in
EXCESS of tolerance (zero when within tolerance). This is distinct from
the differentiable satisfaction-loss surface in
``clay._ir_bridge_distance_to_sat`` (which feeds the adjoint/gradient
path); here the input is the already-computed :class:`InvariantViolation`
set and the output is a single scalar progress metric the repair loop's
strict-descent check consumes.

``repair_is_locally_safe`` encodes the theory doc's local-safety guard:
a repair step is locally safe iff the new distance does not exceed the
old AND no preserved obligation regressed (no previously-satisfied
obligation became unsatisfied).
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional

from clay.invariants_modules._violation import InvariantViolation


def _violation_distance_one(v: InvariantViolation) -> float:
    """Per-violation distance ``d_i`` >= 0.

    Contract:
      * ``skipped`` / passing entries contribute 0.
      * ``mismatch`` entries contribute ``max(0, |relative_error| -
        tolerance)`` — the amount by which the entry exceeds its band.
        A finite relative_error is required; if it is missing/non-finite
        the entry falls back to the ``measure_missing`` treatment below.
      * ``measure_missing`` / ``pending`` entries (no actual value) carry
        a fixed unit penalty of 1.0 so an unobservable obligation still
        registers nonzero distance (it is NOT satisfied), but cannot be
        sized by a gradient — the repair loop treats it as a floor it
        cannot descend without new observation.
    """
    if v.passed or v.category == "skipped":
        return 0.0
    rel = v.relative_error
    tol = float(v.tolerance) if isinstance(v.tolerance, (int, float)) else 0.0
    if isinstance(rel, (int, float)) and math.isfinite(float(rel)):
        excess = abs(float(rel)) - tol
        return excess if excess > 0.0 else 0.0
    # No usable relative error: treat as an unobservable / pending defect.
    return 1.0


def violation_distance(
    violations: Iterable[InvariantViolation],
    weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Weighted distance-to-satisfaction for a violation set.

    Args:
      * `violations` — the InvariantViolation set from one verifier run.
      * `weights`    — optional ``{role_kind: weight}`` map. A missing
                       kind defaults to weight 1.0. Weights let a caller
                       prioritise (e.g.) stability obligations over a
                       cosmetic gain margin during the repair search.

    Returns a non-negative scalar; 0.0 means every violation is within
    tolerance (or skipped). The metric is monotone in each per-violation
    distance: adding a violation or worsening one never decreases the
    total.
    """
    total = 0.0
    for v in violations:
        w = 1.0
        if weights is not None:
            wv = weights.get(v.role_kind, 1.0)
            if isinstance(wv, (int, float)) and math.isfinite(float(wv)):
                w = float(wv)
        total += w * _violation_distance_one(v)
    return total


def _satisfied_keys(violations: Iterable[InvariantViolation]) -> set[tuple[str, str]]:
    """Set of (role_name, target_attr) keys currently within tolerance.

    Used by :func:`repair_is_locally_safe` to detect a regressed
    preserved obligation: a key satisfied before that is no longer
    satisfied after a repair step.
    """
    out: set[tuple[str, str]] = set()
    for v in violations:
        if _violation_distance_one(v) == 0.0:
            out.add((v.role_name, v.target_attr))
    return out


def repair_is_locally_safe(
    before_dist: float,
    after_dist: float,
    *,
    before_violations: Optional[Iterable[InvariantViolation]] = None,
    after_violations: Optional[Iterable[InvariantViolation]] = None,
    atol: float = 1e-12,
) -> bool:
    """Local-safety predicate for one repair step.

    Returns True iff BOTH hold:
      1. Non-increasing distance: ``after_dist <= before_dist + atol``.
      2. No preserved obligation regressed: every (role_name,
         target_attr) that was satisfied before is still satisfied after.

    Condition 2 is only checked when both violation sets are supplied;
    with distances alone the predicate degrades to the non-increase test.
    The ``atol`` slack absorbs floating-point noise so a numerically-equal
    distance is not rejected.
    """
    if not (math.isfinite(before_dist) and math.isfinite(after_dist)):
        return False
    if after_dist > before_dist + atol:
        return False
    if before_violations is not None and after_violations is not None:
        before_ok = _satisfied_keys(before_violations)
        after_ok = _satisfied_keys(after_violations)
        # A regression is any previously-satisfied obligation that is no
        # longer satisfied.
        if not before_ok.issubset(after_ok):
            return False
    return True
