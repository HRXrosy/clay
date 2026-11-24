"""Unified checker verdict + blocking policy for the Intent->Obligation->Evidence->Verdict
chain.

A *checker* runs on an *artifact* and returns a :class:`CheckerVerdict`: one of the five
explicit states (pass / fail / unmeasurable / unsupported / advisory), the *evidence* it
read, a *blocker reason*, and whether the obligation is ``required``. This is the
"Evidence + Verdict" pair the architecture names; it projects into the existing
certificate spine (:class:`clay.core_contract.ObservationRow`) via
:func:`to_observation_row`, so checkers feed the SAME chain as the rest of Clay rather
than a parallel result model.

Two invariants are enforced at construction so a gate cannot lie:

* a ``PASS`` MUST carry evidence -- an empty-evidence pass is a rubber stamp
  (``gate must not silent-pass``);
* a blocking non-pass (``FAIL``/``UNSUPPORTED``/``UNMEASURABLE``/...) MUST carry a
  ``blocker_reason`` -- a gate that rejects must say why.

:func:`is_blocking` is the SINGLE place that decides which states stop sign-off. The
``required`` flag (the ``quality @require`` decision) is what turns a tolerated
incompleteness (unsupported/unmeasurable on an optional obligation) into a hard block.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from clay.core_contract import ObservationRow, SourceObligation, VerificationStatus

__all__ = ["CheckerVerdict", "is_blocking", "to_observation_row", "ADVISORY_STATES",
           "ALWAYS_BLOCKING", "BLOCK_IF_REQUIRED"]

# states that block sign-off REGARDLESS of `required` -- a real, observed defect or a
# checker crash. There is no design in which these are acceptable.
ALWAYS_BLOCKING = frozenset({
    VerificationStatus.FAIL,
    VerificationStatus.ERROR,
    VerificationStatus.NO_CONVERGENCE,
    # An unresolvable priority conflict (the obligation set is self-contradictory:
    # two same-tier obligations, no declared superiority, equal specificity). It
    # blocks unconditionally until a designer declares superiority or changes an
    # intent -- the system NEVER silently picks a winner (no-silent-priority, the
    # dual of no-silent-pass). Only the priority resolver emits CONFLICT, so this
    # is purely additive: no pre-existing checker returns it.
    VerificationStatus.CONFLICT,
})

# states that block ONLY when the obligation is `required` -- a tolerated incompleteness
# (not implemented / not measurable / not yet run) that becomes a hard block exactly when
# the design @require(s) it. This is how Clay tolerates partial capability without ever
# silently passing a required obligation.
BLOCK_IF_REQUIRED = frozenset({
    VerificationStatus.UNSUPPORTED,
    VerificationStatus.UNMEASURABLE,
    VerificationStatus.PENDING,
    VerificationStatus.SKIP,
    VerificationStatus.UNKNOWN,
})

# states that NEVER block -- a pass, or an informational advisory.
ADVISORY_STATES = frozenset({
    VerificationStatus.PASS,
    VerificationStatus.ADVISORY,
})


@dataclass(frozen=True)
class CheckerVerdict:
    """One checker's verdict on one obligation, carrying its evidence + blocker reason."""

    checker: str
    obligation_id: str
    status: VerificationStatus
    evidence: Mapping[str, Any] = field(default_factory=dict)
    blocker_reason: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        if self.status == VerificationStatus.PASS and not self.evidence:
            raise ValueError(
                f"silent-pass violation: checker {self.checker!r} returned PASS on "
                f"obligation {self.obligation_id!r} with NO evidence -- a pass must cite "
                "the evidence it read (an empty-evidence pass is a rubber stamp)."
            )
        if self.status in (ALWAYS_BLOCKING | BLOCK_IF_REQUIRED) and not self.blocker_reason:
            raise ValueError(
                f"checker {self.checker!r} returned {self.status.value!r} on obligation "
                f"{self.obligation_id!r} with NO blocker_reason -- a gate that rejects (or "
                "reports an incompleteness) must say why."
            )


def is_blocking(verdict: CheckerVerdict) -> bool:
    """Does this verdict BLOCK sign-off? ``ALWAYS_BLOCKING`` states block unconditionally;
    ``BLOCK_IF_REQUIRED`` states block iff the obligation is ``required``; advisory/pass
    never block."""
    if verdict.status in ALWAYS_BLOCKING:
        return True
    if verdict.status in BLOCK_IF_REQUIRED:
        return verdict.required
    return False


def to_observation_row(verdict: CheckerVerdict, obligation: SourceObligation) -> ObservationRow:
    """Project a checker verdict into the certificate spine's evidence row, linked to its
    obligation. Lets a checker feed :class:`clay.core_contract.ContractCertificate`
    (Intent->Obligation->Evidence->Verdict) directly."""
    if verdict.obligation_id != obligation.id:
        raise ValueError(
            f"verdict obligation_id {verdict.obligation_id!r} != obligation.id "
            f"{obligation.id!r}; a verdict only certifies the obligation it ran for."
        )
    return ObservationRow(
        source_obligation_id=obligation.id,
        observation_id=obligation.observation_id,
        observation_kind=obligation.observation_kind,
        unit=obligation.unit,
        status=verdict.status,
        required=obligation.required,
    )
