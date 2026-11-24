"""
Repair status enum — strictly separate from verification status.

The completeness theory (paper/clay-tcad/clay_language_completeness_theory.html,
"Status Semantics" + "Repair completeness theorem shape") requires that a
circuit's *verification* status and *repair* status live on independent
axes. A design can be verification-FAIL while its repair status is
REPAIRABLE, REPAIR_PENDING, REPAIR_UNSUPPORTED or NO_REPAIR_OPERATOR. The
inability to repair must never masquerade as "unobservable" or as a
verification PASS.

Concretely:

  * `RepairStatus` is a closed set distinct from
    :class:`clay.invariants_modules._violation.VerificationStatus`.
  * A repair status never rewrites a verification status. The two enums
    are not interconvertible; the only relationship is the invariant that
    `RepairStatus.REPAIRED` does NOT imply `VerificationStatus.PASS` until
    a fresh deterministic oracle run certifies the candidate.
"""

from __future__ import annotations

from enum import Enum


class RepairStatus(Enum):
    """Repair-axis status, orthogonal to verification status.

    Values:
      * `REPAIRABLE`        — a legal repair operator exists for the
                              violation AND the local evidence (gradient
                              or signed target distance) is sufficient to
                              propose a bounded knob delta.
      * `REPAIR_PENDING`    — a legal repair knob exists for the
                              (role_kind, target_attr) pair, but the
                              evidence needed to size the move (gradient
                              and/or a finite target/actual pair) is not
                              yet available. The repair is blocked on
                              missing observation, not on missing algebra.
      * `REPAIR_UNSUPPORTED` — the only repair class that could close the
                              violation is currently out of scope (e.g. a
                              topology-changing move). The operator exists
                              in the algebra but is deferred.
      * `NO_REPAIR_OPERATOR` — no repair operator is registered for the
                              (role_kind, target_attr) pair at all. The
                              theory doc names this `FAIL_NO_REPAIR_OPERATOR`;
                              the runtime enum keeps the shorter token.
      * `REPAIRED`          — a proposed repair was applied AND a fresh
                              oracle run reduced the violation distance to
                              within tolerance. This is a repair-axis fact
                              only; it does NOT certify verification PASS —
                              that remains the verifier's exclusive call.
    """

    REPAIRABLE = "repairable"
    REPAIR_PENDING = "repair_pending"
    REPAIR_UNSUPPORTED = "repair_unsupported"
    NO_REPAIR_OPERATOR = "no_repair_operator"
    REPAIRED = "repaired"

    @property
    def is_actionable(self) -> bool:
        """True iff a concrete repair move can be proposed right now."""
        return self is RepairStatus.REPAIRABLE

    @property
    def certifies_verification_pass(self) -> bool:
        """Always False.

        No repair status — not even ``REPAIRED`` — may be read as a
        verification PASS. PASS is produced only by the deterministic
        oracle (see Oracle Soundness / Non-Probabilistic Certification in
        the theory doc). This property exists so callers can assert the
        separation invariant explicitly in their own guards.
        """
        return False
