"""
Repair proposal — turn one :class:`InvariantViolation` into either a
concrete :class:`RepairProposal` or a :class:`RepairStatus` explaining
why no actionable proposal could be made.

Decision order (theory doc, "Repair completeness theorem shape"):

  1. No operator registered for (role_kind, target_attr)
        -> RepairStatus.NO_REPAIR_OPERATOR
  2. Operator exists but its only repair class is deferred
     (topology_changing) -> RepairStatus.REPAIR_UNSUPPORTED
  3. Operator exists, class supported, but the local evidence needed to
     size a bounded move is missing -> RepairStatus.REPAIR_PENDING
  4. Operator exists + evidence present -> RepairProposal (status
     embedded as RepairStatus.REPAIRABLE).

A `skipped` / passing violation yields no proposal — there is nothing to
repair. This is reported as REPAIR_PENDING with reason ``"no_violation"``
so the caller never mistakes "already satisfied" for "broken algebra".

Evidence preference:
  * If the violation carries a finite ``gradient_value``, the proposed
    knob delta is a trust-region-bounded Newton step on the primary knob,
    reusing the SAME step convention as
    ``clay.invariants_modules._gradient._newton_step_target`` (signed
    distance / |gradient|, clipped to a relative trust region).
  * Else, if the violation carries a finite signed ``relative_error``
    beyond tolerance and the operator's local response is monotone, the
    proposed delta is a bounded *relative* knob move whose sign comes from
    the operator's ``monotone_sign`` and the sign of the error.

This module imports the violation dataclass + the gradient Newton helper
read-only; it does not edit any existing invariant module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from clay.invariants_modules._violation import InvariantViolation

from ._registry import RepairOperator, RepairRegistry, default_registry
from ._status import RepairStatus

# Default trust-region: a single proposed knob move may not exceed this
# relative fraction of the current knob (or, for the gradient path, of the
# current target value). Keeps a degenerate-OP gradient from proposing an
# order-of-magnitude jump on one retry.
DEFAULT_TRUST_REGION: float = 0.5

# Floor for a gradient magnitude before it is treated as "no direction".
_GRAD_EPS: float = 1e-30


@dataclass
class RepairProposal:
    """A concrete, bounded repair move for one violation.

    Fields:
      * `status`        — always :attr:`RepairStatus.REPAIRABLE` (a
                          proposal is only constructed on the actionable
                          path).
      * `role_kind` / `role_name` / `target_attr` — echoed from the
                          violation for traceability (Diagnostic Locality).
      * `operator`      — the :class:`RepairOperator` that produced the
                          move.
      * `knob`          — the primary knob name being adjusted.
      * `knob_delta`    — signed relative change to apply to the knob
                          (e.g. +0.12 means "increase the knob by 12%").
                          Relative because the runtime does not know the
                          knob's absolute current value here; the apply
                          step resolves it against the live device.
      * `suggested_target_value` — when the evidence was a gradient and
                          the violation already carried a Newton-step
                          target suggestion, it is echoed for the caller.
      * `evidence`      — ``"gradient"`` or ``"monotone_error"`` — which
                          path sized the delta.
      * `trust_region`  — the relative bound the delta was clipped to.
      * `preserved_obligations` — human-readable list of what the operator
                          class must NOT break (topology, ports, role
                          identity, ...). Surfaced so a downstream safety
                          check can verify none regressed.
      * `rationale`     — one-line text hint (re-uses the violation's
                          existing ``suggested_adjustment`` when present).
    """

    status: RepairStatus
    role_kind: str
    role_name: str
    target_attr: str
    operator: RepairOperator
    knob: str
    knob_delta: float
    evidence: str
    trust_region: float
    suggested_target_value: Optional[float] = None
    preserved_obligations: tuple[str, ...] = ()
    rationale: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def knob_direction(self) -> int:
        """+1 to increase the knob, -1 to decrease, 0 for a null move."""
        if self.knob_delta > 0:
            return 1
        if self.knob_delta < 0:
            return -1
        return 0


# Preserved-obligation text per repair class (theory doc, Repair Algebra
# "Preserved Obligations" column). Surfaced on every proposal so a safety
# check can reason about what must stay intact.
_PRESERVED_BY_CLASS: dict[str, tuple[str, ...]] = {
    "sizing": ("topology", "ports", "role_identity", "pdk_legality"),
    "bias": ("functional_topology", "supply_rails"),
    "compensation": ("stage_topology", "dc_operating_point_within_tolerance"),
    "topology_preserving": ("external_port_contract", "declared_function"),
    "layout_aware": ("schematic_connectivity", "measurement_obligations"),
    "topology_changing": (),
}


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _bounded_gradient_delta(
    violation: InvariantViolation,
    trust_region: float,
) -> Optional[float]:
    """Relative knob delta from the violation's gradient evidence.

    Reuses the signed-distance / |gradient| Newton scale from
    ``_gradient._newton_step_target`` but expresses the move as a
    *relative* knob change (dimensionless), clipped to the trust region.

    Returns None when the gradient is missing / degenerate or the
    target/actual pair is not finite.
    """
    grad = violation.gradient_value
    tgt = violation.target_value
    act = violation.actual_value
    if not (_is_finite_number(grad) and _is_finite_number(tgt)
            and _is_finite_number(act)):
        return None
    grad = float(grad)
    if abs(grad) < _GRAD_EPS:
        return None
    tgt = float(tgt)
    act = float(act)
    spec_distance = abs(tgt - act)
    # Newton scale: how far the knob must move (in target-units) to close
    # the gap at the local slope. Direction follows the gradient sign, as
    # in _newton_step_target.
    direction = 1.0 if grad > 0 else -1.0
    step_in_target_units = direction * spec_distance / max(abs(grad), _GRAD_EPS)
    # Express relative to the current target magnitude so the move is a
    # dimensionless knob fraction the apply step can scale.
    denom = max(abs(tgt), _GRAD_EPS)
    rel = step_in_target_units / denom
    return _clip(rel, trust_region)


def _bounded_monotone_delta(
    violation: InvariantViolation,
    operator: RepairOperator,
    trust_region: float,
) -> Optional[float]:
    """Relative knob delta from a monotone signed relative_error.

    When no gradient is available but the operator's local response is
    monotone and the violation carries a finite ``relative_error`` beyond
    tolerance, move the primary knob in the direction the monotone sign
    dictates. ``relative_error = (actual - target)/|target|``; a positive
    error means "actual too high", so the knob must move to LOWER the
    measure: ``-sign(error) * monotone_sign``.

    Magnitude is the excess error beyond tolerance, clipped to the trust
    region — a deliberately conservative one-step move (the response is
    only assumed locally monotone, not linear).
    """
    rel_err = violation.relative_error
    if not _is_finite_number(rel_err):
        return None
    rel_err = float(rel_err)
    tol = float(violation.tolerance) if _is_finite_number(violation.tolerance) else 0.0
    excess = abs(rel_err) - tol
    if excess <= 0.0:
        # Within tolerance on this attribute — nothing to move.
        return None
    err_sign = 1.0 if rel_err > 0 else -1.0
    # To reduce the measure when actual is too high (err_sign>0), move the
    # knob opposite to its monotone sign; and vice-versa.
    knob_sign = -err_sign * float(operator.monotone_sign)
    return _clip(knob_sign * excess, trust_region)


def _clip(rel: float, trust_region: float) -> float:
    """Clip a relative delta into [-trust_region, +trust_region]."""
    if not math.isfinite(rel):
        return 0.0
    if rel > trust_region:
        return trust_region
    if rel < -trust_region:
        return -trust_region
    return rel


def propose_repair(
    violation: InvariantViolation,
    registry: Optional[RepairRegistry] = None,
    *,
    trust_region: float = DEFAULT_TRUST_REGION,
) -> Union[RepairProposal, RepairStatus]:
    """Propose a bounded repair move for a single invariant violation.

    Returns a :class:`RepairProposal` (status REPAIRABLE) when an operator
    exists, its class is supported, and the local evidence is sufficient
    to size a bounded knob delta. Otherwise returns the explaining
    :class:`RepairStatus`:

      * ``NO_REPAIR_OPERATOR`` — no operator for (role_kind, target_attr).
      * ``REPAIR_UNSUPPORTED`` — operator's only class is deferred.
      * ``REPAIR_PENDING``     — operator exists but evidence is missing
                                 (or the violation is already satisfied).

    The function never mutates the violation and never touches the
    verification status.
    """
    reg = registry if registry is not None else default_registry()

    # An already-passing / skipped violation has nothing to repair. This
    # is a PENDING-with-reason, NOT a missing operator: the algebra is
    # fine, there is simply no active defect.
    if violation.passed or violation.category == "skipped":
        return RepairStatus.REPAIR_PENDING

    op = reg.lookup(violation.role_kind, violation.target_attr)
    if op is None:
        return RepairStatus.NO_REPAIR_OPERATOR

    if not op.is_supported:
        # Only a deferred (topology-changing) class could close this.
        return RepairStatus.REPAIR_UNSUPPORTED

    # Try the gradient path first (highest-fidelity evidence), then the
    # monotone-error fallback.
    delta = _bounded_gradient_delta(violation, trust_region)
    evidence = "gradient"
    if delta is None:
        delta = _bounded_monotone_delta(violation, op, trust_region)
        evidence = "monotone_error"

    if delta is None or delta == 0.0:
        # Operator exists, class supported, but neither a usable gradient
        # nor a finite over-tolerance error is present. Repair is blocked
        # on missing evidence, not on missing algebra.
        return RepairStatus.REPAIR_PENDING

    preserved = _PRESERVED_BY_CLASS.get(op.repair_class.value, ())
    rationale = violation.suggested_adjustment or (
        f"adjust {op.primary_knob} to close {violation.target_attr} "
        f"on role {violation.role_name!r}"
    )

    return RepairProposal(
        status=RepairStatus.REPAIRABLE,
        role_kind=violation.role_kind,
        role_name=violation.role_name,
        target_attr=violation.target_attr,
        operator=op,
        knob=op.primary_knob,
        knob_delta=float(delta),
        evidence=evidence,
        trust_region=trust_region,
        suggested_target_value=violation.suggested_target_value,
        preserved_obligations=preserved,
        rationale=rationale,
    )
