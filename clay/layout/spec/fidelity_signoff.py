"""PHASE F1 -- the geometric-fidelity GATE + the obligation-class COVERAGE gate.

ROOT (established with the user): Clay gated only LEGALITY (DRC / LVS), never
geometric QUALITY, so "green" meant LEGAL, not GOOD -- the user's eye was the
only geometric oracle. F0 (clay.layout.spec.geometric_fidelity) built the
measure-only fidelity CHECKERS. This module makes them GATE and -- the heart of
the root-cure -- adds an OBLIGATION-CLASS COVERAGE gate so a signoff that
measured ONLY {DRC, LVS} is judged INCOMPLETE (blocked), never green. This
extends "no silent pass" to "no silent legality-only pass".

Two mechanisms, both feeding the SAME signoff blocking policy
(:func:`clay.signoff_verdict.is_blocking`) -- no parallel result model:

  1. fidelity_verdicts(report)
     Bridge an F0 :class:`~clay.layout.spec.geometric_fidelity.FidelityReport`
     into the canonical :class:`~clay.signoff_verdict.CheckerVerdict` set. A RED
     metric (satisfied is False) -> a BLOCKING ``FAIL``; a GREEN metric -> a
     ``PASS`` carrying its measured evidence; an unobservable metric ->
     ``UNMEASURABLE`` (required => blocks; never a silent pass). The metric was
     already measured from the DESIGN topology + the EMITTED GDS (the single
     canonical artifact); this module only verdicts it -- it reads no coordinate
     and no reference.

  2. required_obligation_classes(artifact_type) + coverage_verdicts(measured, ...)
     The required-obligation-CLASS registry: an ``AsBuiltLayout`` MUST carry
     {DRC, LVS, geometric_fidelity}. The coverage gate compares the set of
     classes a signoff actually MEASURED against the required set; a REQUIRED
     class that is ABSENT yields a BLOCKING ``UNMEASURABLE`` verdict (the
     INCOMPLETE state -- it was never measured, distinct from a measured FAIL).
     So DRC=0 + LVS matched with geometric_fidelity unmeasured is NOT green: the
     coverage gate blocks. This is the structural reason a legality-only signoff
     can no longer be reported green -- enforcement, not discipline.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Set

from clay.core_contract import VerificationStatus
from clay.signoff_verdict import CheckerVerdict

__all__ = [
    "fidelity_verdicts",
    "electrical_intent_verdicts",
    "required_obligation_classes",
    "coverage_verdicts",
    "REQUIRED_OBLIGATION_CLASSES",
    "GEOMETRIC_FIDELITY_CLASS",
    "ELECTRICAL_INTENT_CLASS",
]

# The obligation-CLASS name the fidelity gate registers under. A signoff that
# produces fidelity verdicts has MEASURED this class; one that did not has an
# ABSENT required class (the legality-only failure mode).
GEOMETRIC_FIDELITY_CLASS = "geometric_fidelity"

# The electrical-intent obligation CLASS (the E1/E2 topology-derived checkers --
# symmetry placement/deviation, em_width, parasitic balance, sensitive coupling).
# These are QUALITY-tier: a RED electrical obligation (e.g. the diff axis is
# off-centre / the per-pair axes scatter) is a BLOCKING fail, so a layout that is
# DRC/LVS-legal but electrically-asymmetric cannot be reported green.
ELECTRICAL_INTENT_CLASS = "electrical_intent"


# --------------------------------------------------------------------------- #
# The required-obligation-CLASS registry (per artifact type).
# --------------------------------------------------------------------------- #
#
# This is the structural heart of "no silent legality-only pass". It is a TABLE
# keyed by artifact type -- not a per-circuit special case (it keys on the kind
# of ARTIFACT, never on a named circuit or spec). Every artifact type that can be
# signed off declares the obligation CLASSES that MUST be measured for its
# signoff to be complete. A class absent from a signoff's measured set is an
# INCOMPLETE signoff, blocked -- the same honesty as a required obligation that
# is un-run.
REQUIRED_OBLIGATION_CLASSES: Dict[str, frozenset] = {
    # An as-built layout is legal AND good only when all three classes are
    # measured: DRC (design-rule legality), LVS (connectivity correctness), and
    # geometric_fidelity (the QUALITY gate F0/F1 mechanise). Measuring only the
    # first two is the exact failure mode the user caught with their eye.
    "AsBuiltLayout": frozenset({"DRC", "LVS", GEOMETRIC_FIDELITY_CLASS}),
}


def required_obligation_classes(artifact_type: str) -> frozenset:
    """The obligation CLASSES required for ``artifact_type``'s signoff to be
    complete. Raises ``KeyError`` for an unknown artifact type -- there is NO
    silent empty-set fallback (that would make every signoff trivially complete,
    re-opening the legality-only hole). The registry is the single source of
    truth for what "complete coverage" means."""
    if artifact_type not in REQUIRED_OBLIGATION_CLASSES:
        raise KeyError(
            f"unknown artifact type {artifact_type!r}; no required-obligation-"
            f"class set is registered for it. Known: "
            f"{sorted(REQUIRED_OBLIGATION_CLASSES)}. Refusing a silent empty "
            "coverage set (that would make the signoff trivially complete)."
        )
    return REQUIRED_OBLIGATION_CLASSES[artifact_type]


# --------------------------------------------------------------------------- #
# (1) fidelity gate: FidelityReport -> CheckerVerdicts (FAIL when RED, blocking).
# --------------------------------------------------------------------------- #
def _fidelity_verdict_for_metric(metric: Any) -> CheckerVerdict:
    """One :class:`CheckerVerdict` for one F0 ``MetricVerdict``.

    The mapping is the ``fidelity`` hardness mapping made explicit in signoff
    terms:

      * satisfied is False  -> FAIL  (blocking -- a RED geometric metric gates);
      * satisfied is True   -> PASS  (carries the measured evidence);
      * not observable      -> UNMEASURABLE (required => blocks; never silent);
      * observable but None -> ADVISORY (a structural metric not yet decided --
        reported, never a fake pass).
    """
    oid = f"fidelity.{metric.name}"
    evidence = {
        "value": metric.value,
        "expected": metric.expected,
        "threshold": metric.threshold,
        "detail": metric.detail,
    }

    if not metric.observable:
        return CheckerVerdict(
            checker="geometric_fidelity",
            obligation_id=oid,
            status=VerificationStatus.UNMEASURABLE,
            evidence=evidence,
            blocker_reason=(
                f"{metric.name} unmeasurable from emitted geometry: "
                f"{metric.detail}"
            ),
            required=True,
        )
    if metric.satisfied is True:
        return CheckerVerdict(
            checker="geometric_fidelity",
            obligation_id=oid,
            status=VerificationStatus.PASS,
            evidence=evidence,
            blocker_reason="",
            required=True,
        )
    if metric.satisfied is False:
        return CheckerVerdict(
            checker="geometric_fidelity",
            obligation_id=oid,
            status=VerificationStatus.FAIL,
            evidence=evidence,
            blocker_reason=(
                f"geometric fidelity RED: {metric.name} measured "
                f"{metric.value} vs band {metric.threshold} ({metric.detail})"
            ),
            required=True,
        )
    # observable but not quantitatively decided (structural placeholder).
    return CheckerVerdict(
        checker="geometric_fidelity",
        obligation_id=oid,
        status=VerificationStatus.ADVISORY,
        evidence=evidence,
        blocker_reason="",
        required=True,
    )


def fidelity_verdicts(report: Any) -> Dict[str, CheckerVerdict]:
    """Bridge an F0 :class:`FidelityReport` into the canonical signoff verdict
    set. One :class:`CheckerVerdict` per measured metric, keyed by
    ``fidelity.<metric>``. A RED metric is a BLOCKING FAIL -- the mechanism that
    makes a legal-but-geometrically-wrong layout un-greenable.

    The report was measured by :func:`measure_fidelity` from the design topology
    + the emitted GDS; this function only verdicts it (reads no coordinate, no
    reference).
    """
    out: Dict[str, CheckerVerdict] = {}
    for metric in getattr(report, "metrics", ()) or ():
        v = _fidelity_verdict_for_metric(metric)
        out[v.obligation_id] = v
    return out


# --------------------------------------------------------------------------- #
# (1b) electrical-intent gate: ElecVerdict (5-state) -> CheckerVerdicts.
# --------------------------------------------------------------------------- #
# The QUALITY-tier obligations whose RED state BLOCKS signoff (a DRC/LVS-legal but
# electrically-asymmetric layout is not green). The remaining electrical verdicts
# (parasitic balance / sensitive coupling) are reported but kept ADVISORY here --
# they ride a calibrated-PEX dependency and are not yet a hard gate.
_ELECTRICAL_BLOCKING_OBLIGATIONS = frozenset({
    "symmetry_placement",   # the diff axis must be ONE shared, centred axis
    "symmetry_deviation",   # the routed mirror metal must be symmetric
    "em_width",             # power nets at the EM width class
})


def _electrical_status(state: str) -> "VerificationStatus":
    """Map the ElecVerdict 5-state string to the canonical VerificationStatus
    (the two share the same five names, so this is a direct, total mapping --
    never a silent default)."""
    return VerificationStatus(state)


def electrical_intent_verdicts(elec_verdicts: Any) -> Dict[str, CheckerVerdict]:
    """Bridge the E1/E2 :class:`ElecVerdict` set (from
    :func:`clay.layout.spec.electrical_obligations.measure_electrical`) into the
    canonical signoff verdict set -- one :class:`CheckerVerdict` per obligation,
    keyed by ``electrical.<name>``. A RED obligation in
    ``_ELECTRICAL_BLOCKING_OBLIGATIONS`` is REQUIRED (so a FAIL blocks signoff:
    the layout is electrically wrong even if DRC/LVS-legal); the PEX-dependent
    balance/coupling obligations are reported non-required (advisory-strength).

    Each ElecVerdict was measured from the design TOPOLOGY + the as-built route /
    emitted GDS (never a reference coordinate); this function only verdicts it.
    """
    out: Dict[str, CheckerVerdict] = {}
    for ev in (elec_verdicts or ()):
        name = getattr(ev, "name", None)
        if name is None:
            continue
        oid = f"electrical.{name}"
        required = name in _ELECTRICAL_BLOCKING_OBLIGATIONS
        evidence = {
            "value": getattr(ev, "value", None),
            "threshold": getattr(ev, "threshold", ""),
            "members": list(getattr(ev, "members", ()) or ()),
            "detail": getattr(ev, "detail", ""),
            "backend": getattr(ev, "backend", ""),
        }
        status = _electrical_status(getattr(ev, "state"))
        # a blocking non-pass MUST carry a reason (CheckerVerdict invariant); a
        # PASS/ADVISORY carries none. UNMEASURABLE/UNSUPPORTED block iff required.
        if status in (VerificationStatus.PASS, VerificationStatus.ADVISORY):
            reason = ""
        else:
            reason = (
                f"electrical intent {status.value}: {name} = "
                f"{getattr(ev, 'value', None)} vs {getattr(ev, 'threshold', '')} "
                f"({getattr(ev, 'detail', '')})"
            )
        out[oid] = CheckerVerdict(
            checker="electrical_intent",
            obligation_id=oid,
            status=status,
            evidence=evidence,
            blocker_reason=reason,
            required=required,
        )
    return out


# --------------------------------------------------------------------------- #
# (2) the COVERAGE gate: required class absent -> blocking INCOMPLETE.
# --------------------------------------------------------------------------- #
def coverage_verdicts(
    measured_classes: Set[str] | Mapping[str, Any],
    *,
    artifact_type: str = "AsBuiltLayout",
) -> Dict[str, CheckerVerdict]:
    """The obligation-class COVERAGE gate -- "no silent legality-only pass".

    Compare the obligation CLASSES a signoff actually measured
    (``measured_classes``) against :func:`required_obligation_classes` for the
    artifact type. For every REQUIRED class that is ABSENT, emit a BLOCKING
    ``UNMEASURABLE`` verdict (the INCOMPLETE state: the class was never measured,
    distinct from a measured FAIL). A required class can never be silently
    omitted: its absence blocks signoff exactly like an un-run required
    obligation.

    Parameters
    ----------
    measured_classes : set or mapping
        The obligation CLASSES this signoff produced verdicts for (e.g.
        ``{"DRC", "LVS", "geometric_fidelity"}``). A mapping (e.g. the full
        verdict dict keyed by class) is accepted; only its keys are read.
    artifact_type : str
        Which artifact's required-class set to enforce (default
        ``"AsBuiltLayout"``). An unknown artifact type raises (no silent
        empty-set), via :func:`required_obligation_classes`.

    Returns
    -------
    dict of obligation_id -> CheckerVerdict
        One ``coverage.<class>`` verdict per MISSING required class (each
        blocking). Empty when every required class is present (the coverage gate
        fires ONLY on the silent-omission failure mode -- it does not re-verdict
        a class that WAS measured; that class's own checkers do).
    """
    required = required_obligation_classes(artifact_type)
    measured = set(measured_classes)  # set() of keys for a mapping; identity for a set
    missing = sorted(required - measured)

    out: Dict[str, CheckerVerdict] = {}
    for cls in missing:
        oid = f"coverage.{cls}"
        out[oid] = CheckerVerdict(
            checker="coverage",
            obligation_id=oid,
            status=VerificationStatus.UNMEASURABLE,
            evidence={
                "artifact_type": artifact_type,
                "required": sorted(required),
                "measured": sorted(measured),
                "missing_class": cls,
            },
            blocker_reason=(
                f"signoff INCOMPLETE: required obligation class {cls!r} was not "
                f"measured for a {artifact_type} (measured only "
                f"{sorted(measured)}). A legality-only signoff is NOT green -- "
                "no silent legality-only pass."
            ),
            required=True,
        )
    return out
