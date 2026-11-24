"""SP3 -- hardness -> 5-state verdict layer (one independent checker per rule).

Each obligation emitted by a spec rule (SP2) is verdicted by its NAMED checker
against the AS-BUILT placement geometry, then mapped to Clay's explicit 5-state
verdict model by the rule's declared hardness.

Why the checker is independent of the obligation's self-report
-------------------------------------------------------------
The obligation dict only carries *symbols* (metric/effect/object/hardness/
checker/rule/members) -- it never carries a coordinate, a measured value, or a
self-asserted pass/fail. A checker here is a function ``checker(members,
placement) -> Measurement`` whose ONLY inputs are (a) the design-sourced member
names and (b) the as-built ``placement`` geometry. It re-derives the measured
quantity from geometry every time; it cannot read what the emitter "intended".
So "native indication" (the emitter that added the hyperedge) and "oracle"
(this checker, reading geometry) are structurally separate code paths over the
same canonical inputs -- if they ever disagree, that is a bug in one of them,
not a thing this layer papers over.

The 5 states (Clay's explicit-incompleteness model)
---------------------------------------------------
    pass         -- oracle observed the geometry and the obligation holds.
    fail         -- oracle observed the geometry and the obligation is violated.
    unmeasurable -- oracle could not observe the relevant geometry (e.g. a
                    member is absent from the placement). NEVER a silent pass.
    unsupported  -- the obligation's effect has no model in this checker/PDK
                    (capability absent). Explicit, NEVER a silent pass.
    advisory     -- a structural / qualitative obligation that is reported but
                    never gates (and is reported when the underlying numeric
                    check is not yet implemented -- honest, not a fake pass).

Hardness -> reachable-state mapping
-----------------------------------
    hard             -> pass | fail            (| unmeasurable if unobservable)
    quantified_soft  -> pass | fail            (| unmeasurable | advisory when
                        the named checker is a structural placeholder)
    qualitative_soft -> advisory               (| unmeasurable)
    unsupported      -> unsupported

This module is pure stdlib + reads the zero-coordinate obligation ledger plus a
plain ``placement`` mapping ``{device_name: (x, y)}``. Deterministic.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


# The five explicit verdict states. Anything outside this set is a contract
# violation, never a silently-tolerated value.
FIVE_STATES = frozenset(
    {"pass", "fail", "unmeasurable", "unsupported", "advisory"}
)

# Centroid-coincidence threshold for the quantified_soft common-centroid check.
# Common-centroid demands the members' centroids coincide; in a coordinate-free
# obligation we have no per-design footprint scale to normalise against here, so
# we gate on an absolute coincidence tolerance: a satisfied common-centroid has
# its member points effectively on top of each other (spread ~ 0). A real
# placement that spreads members microns apart is a broken centroid. The band
# is deliberately tight; anything beyond it fails rather than silently passing.
_CENTROID_COINCIDENCE_TOL = 1e-6


class Measurement:
    """The independent oracle's observation for one obligation.

    A checker returns this; the verdict mapper turns it into a 5-state verdict
    by the rule's hardness. ``observable`` is False when the oracle could not
    read the relevant geometry (-> unmeasurable). ``supported`` is False when
    the checker has no model for the effect (-> unsupported). ``satisfied`` is
    the boolean outcome when observable and supported and quantitatively
    decidable; it is None for a purely structural (advisory) check.
    """

    __slots__ = ("observable", "supported", "satisfied", "value", "detail")

    def __init__(
        self,
        observable: bool = True,
        supported: bool = True,
        satisfied: Optional[bool] = None,
        value: Optional[float] = None,
        detail: str = "",
    ) -> None:
        self.observable = observable
        self.supported = supported
        self.satisfied = satisfied
        self.value = value
        self.detail = detail


# --------------------------------------------------------------------------- #
# Geometry helpers (read placement only -- the as-built canonical input).      #
# --------------------------------------------------------------------------- #
def _member_points(
    members: Sequence[str], placement: Mapping[str, Tuple[float, float]]
) -> Optional[List[Tuple[float, float]]]:
    """Return each member's placed point, or None if ANY member is absent.

    Absence -> the oracle cannot observe the obligation -> unmeasurable. We do
    not invent a default coordinate (that would be a silent pass)."""
    pts: List[Tuple[float, float]] = []
    for m in members:
        if m not in placement:
            return None
        x, y = placement[m]
        pts.append((float(x), float(y)))
    return pts


def _max_pairwise_distance(pts: Sequence[Tuple[float, float]]) -> float:
    """Largest distance between any two member points (centroid spread proxy).

    For a pair this is just the point-to-point distance; for >2 members it is
    the diameter of the point set, which is 0 iff every member coincides."""
    worst = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = math.hypot(dx, dy)
            if d > worst:
                worst = d
    return worst


# --------------------------------------------------------------------------- #
# Named checkers (independent oracles). Each reads members + placement only.   #
# --------------------------------------------------------------------------- #
def _check_centroid_coincidence(
    members: Sequence[str], placement: Mapping[str, Tuple[float, float]]
) -> Measurement:
    """Common-centroid oracle: members' placed centroids must coincide.

    Reads geometry directly (the member points in ``placement``); does not read
    the emitter's hyperedge or any self-reported value. Quantitatively decides
    by the max pairwise spread against ``_CENTROID_COINCIDENCE_TOL``."""
    pts = _member_points(members, placement)
    if pts is None:
        return Measurement(observable=False, detail="member absent from placement")
    if len(pts) < 2:
        # Nothing to coincide against -- not observable as a centroid relation.
        return Measurement(observable=False, detail="fewer than two members placed")
    spread = _max_pairwise_distance(pts)
    return Measurement(
        satisfied=(spread <= _CENTROID_COINCIDENCE_TOL),
        value=spread,
        detail=f"centroid spread {spread:g}",
    )


def _structural_placeholder(
    members: Sequence[str], placement: Mapping[str, Tuple[float, float]]
) -> Measurement:
    """Honest placeholder oracle for checks not yet quantitatively implemented.

    It can confirm the members are observable (present in placement) but does
    NOT yet verify the structural property (interdigitation balance, orientation
    match). It therefore returns satisfied=None -> the mapper renders it as
    ``advisory`` (reported, never gates) when observable, or ``unmeasurable``
    when a member is absent. It NEVER returns a (fake) pass."""
    pts = _member_points(members, placement)
    if pts is None:
        return Measurement(observable=False, detail="member absent from placement")
    return Measurement(
        satisfied=None,
        detail="structural check not yet quantitatively verified (advisory)",
    )


# Registry: named checker -> independent oracle function. Adding a rule with an
# unknown checker name is surfaced (unsupported), never silently passed.
_CHECKERS: Dict[str, Callable[..., Measurement]] = {
    "centroid_coincidence": _check_centroid_coincidence,
    # Structural placeholders -- honest advisory until SP follow-ups give them a
    # real geometric metric. They are NOT silent passes.
    "interdigitation_balance": _structural_placeholder,
    "orientation_match": _structural_placeholder,
}


def register_checker(name: str, fn: Callable[..., Measurement]) -> None:
    """Register (or deterministically overwrite) a named checker oracle."""
    if not isinstance(name, str) or not name:
        raise ValueError("checker name must be a non-empty string")
    if not callable(fn):
        raise ValueError("checker must be callable")
    _CHECKERS[name] = fn


# --------------------------------------------------------------------------- #
# Hardness -> 5-state mapping.                                                 #
# --------------------------------------------------------------------------- #
def _map_to_state(hardness: str, m: Measurement) -> str:
    """Map a checker Measurement to one of the 5 states by the rule's hardness.

    Incompleteness is always explicit: an unobservable geometry is
    ``unmeasurable``, an absent capability/effect-model is ``unsupported``, a
    structural-only check is ``advisory``. There is no code path that turns an
    unknown or unverified obligation into ``pass``."""
    # Capability gate first: an unsupported hardness, or a checker that reports
    # no model for the effect, is explicitly unsupported (never silent pass).
    if hardness == "unsupported" or not m.supported:
        return "unsupported"

    # Observability gate: if the oracle could not read the geometry, the only
    # honest answer is unmeasurable -- regardless of hardness.
    if not m.observable:
        return "unmeasurable"

    # Qualitative-soft never gates -> advisory (when observable).
    if hardness == "qualitative_soft":
        return "advisory"

    # Hard / quantified_soft / fidelity: a quantitatively decided obligation
    # passes or fails on the oracle's measurement. ``fidelity`` is the F1 gating
    # GEOMETRIC-QUALITY class -- a RED measurement is a blocking FAIL (like
    # DRC/LVS), NOT an advisory. That is the whole point: a legal-but-
    # geometrically-wrong layout cannot be reported green. If the named checker
    # is a structural placeholder (satisfied is None) we report advisory --
    # reported, never a silent pass.
    if hardness in ("hard", "quantified_soft", "fidelity"):
        if m.satisfied is True:
            return "pass"
        if m.satisfied is False:
            return "fail"
        return "advisory"  # structural placeholder: observable but not decided

    # Unknown hardness is a contract gap, surfaced as unsupported (the rule
    # constructor already rejects unknown hardness, so this is defence in depth).
    return "unsupported"


# --------------------------------------------------------------------------- #
# Public API.                                                                  #
# --------------------------------------------------------------------------- #
def verdict_obligations(intent, net, placement) -> List[dict]:
    """Verdict every obligation in ``net`` against the as-built ``placement``.

    For each obligation in ``net.obligations`` (the SP2 ledger), look up its
    named checker, run that independent oracle over the obligation's design-
    sourced members and the as-built ``placement`` geometry, and map the
    Measurement to one of Clay's 5 verdict states by the obligation's declared
    hardness. The returned list mirrors the obligation order (deterministic) and
    transparently re-carries the obligation's provenance fields plus the
    resulting ``state``, the measured ``value`` (if any), and a human ``detail``.

    A missing checker name is surfaced as ``unsupported`` -- never a silent pass.
    A missing ``placement`` (members absent) is ``unmeasurable`` -- never a
    silent pass.
    """
    placement = placement or {}
    obligations = list(getattr(net, "obligations", None) or [])
    verdicts: List[dict] = []

    for ob in obligations:
        hardness = ob.get("hardness")
        checker_name = ob.get("checker")
        members = list(ob.get("members") or [])

        checker = _CHECKERS.get(checker_name)
        if checker is None:
            # No oracle registered for this named checker: explicit unsupported.
            m = Measurement(
                supported=False,
                detail=f"no checker registered for {checker_name!r}",
            )
        else:
            m = checker(members, placement)

        state = _map_to_state(hardness, m)
        if state not in FIVE_STATES:  # defence in depth; must never happen.
            raise AssertionError(f"non-5-state verdict produced: {state!r}")

        verdicts.append(
            {
                "state": state,
                "rule": ob.get("rule"),
                "pack": ob.get("pack"),
                "metric": ob.get("metric"),
                "effect": ob.get("effect"),
                "object": ob.get("object"),
                "hardness": hardness,
                "checker": checker_name,
                "members": members,
                "value": m.value,
                "detail": m.detail,
            }
        )

    return verdicts


__all__ = [
    "FIVE_STATES",
    "Measurement",
    "verdict_obligations",
    "register_checker",
]
