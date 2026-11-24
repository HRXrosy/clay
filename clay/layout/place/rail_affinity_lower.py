"""Lower @rail_affinity / @anchor to the placement EDGE-AFFINITY term (Wave B4).

The materializer contract gap the spec names: ``rail_affinities`` is a first-class
``LayoutIntent`` field (``intent.py``) produced by ``craft_infer`` but has NO placement
lowering -- it is PARSED, not consumed. This module closes that gap on the CLOSED
placement vocabulary: a ``RailAffinityIntent(group, side)`` (or an
``AnchorIntent(target, edge)``) lowers to a
:class:`clay.layout.place.constraints.EdgeAffinity` -- a SOFT pull that draws the group
toward its declared supply-rail edge -- and to the ``rail_affinities`` kwarg
``place_envelope_min`` consumes (a ``(group_key, side)`` pull, empty-default byte-stable).

A placement CHECKER (:func:`check_rail_affinity`) verifies on the realised placement that
the pulled group actually lands on the correct side (a real geometric measurement, the
Wave B4 oracle) -- so the intent is not merely lowered but VERIFIABLE.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from clay.core_contract import VerificationStatus
from clay.layout.place.constraints import EdgeAffinity
from clay.signoff_verdict import CheckerVerdict

__all__ = [
    "lower_rail_affinities",
    "rail_affinity_kwargs",
    "check_rail_affinity",
]

# the two supply-rail edges an @rail_affinity can name (the vertical axis a rail band
# hugs). @anchor additionally allows left/right; those lower to EdgeAffinity too but the
# CP y-pull term consumes only top/bottom (a left/right pull is a reported gap today).
_TOP = frozenset({"top", "t", "vdd", "up"})
_BOTTOM = frozenset({"bottom", "bot", "b", "vss", "gnd", "down"})


def lower_rail_affinities(intent: Any) -> Tuple[EdgeAffinity, ...]:
    """Lower every ``RailAffinityIntent`` (and edge-affinity ``AnchorIntent``) on
    ``intent`` to a closed-vocabulary :class:`EdgeAffinity`. Pure; no coordinate. An
    intent that declares neither returns ``()`` (byte-stable)."""
    from clay.layout.intent import replay_rail_affinity_inventory

    out: List[EdgeAffinity] = []
    for ra in replay_rail_affinity_inventory(
        getattr(intent, "rail_affinities", ())
    ):
        edge = _norm_edge(ra.side)
        if edge:
            out.append(EdgeAffinity(id=ra.group, edge=edge))
    for an in getattr(intent, "anchors", ()) or ():
        edge = _norm_edge(getattr(an, "edge", ""))
        if edge:
            out.append(EdgeAffinity(id=str(getattr(an, "target", "")), edge=edge))
    return tuple(out)


def rail_affinity_kwargs(
    intent: Any, group_of_member: Mapping[str, str] | None = None
) -> Tuple[Tuple[str, str], ...]:
    """The ``rail_affinities`` kwarg ``place_envelope_min`` consumes: ``(group_key,
    side)`` pulls, one per lowered top/bottom EdgeAffinity. ``group_of_member`` maps a
    device/member label to its covering GROUP key (the placer places groups); an
    identity map is used when a rail-affinity already names a group key. Only top/bottom
    edges produce a y-pull (the CP term's axis); left/right are reported gaps."""
    gmap = dict(group_of_member or {})
    kwargs: List[Tuple[str, str]] = []
    for ea in lower_rail_affinities(intent):
        if ea.edge not in ("top", "bottom"):
            continue
        key = gmap.get(ea.id, ea.id)
        kwargs.append((key, ea.edge))
    return tuple(kwargs)


def check_rail_affinity(
    placement: Mapping[str, Tuple[float, float]],
    sizes: Mapping[str, Tuple[float, float]],
    rail_kwargs: Sequence[Tuple[str, str]],
    *,
    required: bool = False,
) -> CheckerVerdict:
    """Verify on the REALISED placement that each rail-affinity group lands on the
    correct half of the cell (a real geometric measurement, the Wave B4 oracle).

    ``placement`` maps group key -> centroid ``(x, y)``; ``sizes`` maps key ->
    ``(w, h)``. A 'bottom' group's centroid must be below the cell's vertical midline,
    a 'top' group's above it. Returns a 5-state verdict (PASS with evidence /
    ADVISORY when nothing to check / FAIL with the offenders)."""
    keys = [k for (k, _s) in rail_kwargs if k in placement]
    if not keys:
        return CheckerVerdict(
            checker="rail_affinity", obligation_id="place.rail_affinity",
            status=VerificationStatus.ADVISORY,
            evidence={"checked": 0},
            blocker_reason="", required=False,
        )
    ys = [placement[k][1] for k in placement]
    if not ys:
        return CheckerVerdict(
            checker="rail_affinity", obligation_id="place.rail_affinity",
            status=VerificationStatus.ADVISORY, evidence={"checked": 0}, required=False,
        )
    mid = (min(ys) + max(ys)) / 2.0
    offenders: Dict[str, str] = {}
    checked: Dict[str, float] = {}
    for k, side in rail_kwargs:
        if k not in placement:
            continue
        y = placement[k][1]
        checked[k] = y
        if side == "bottom" and y > mid:
            offenders[k] = f"bottom-affinity group at y={y:.2f} > midline {mid:.2f}"
        if side == "top" and y < mid:
            offenders[k] = f"top-affinity group at y={y:.2f} < midline {mid:.2f}"
    if offenders:
        return CheckerVerdict(
            checker="rail_affinity", obligation_id="place.rail_affinity",
            status=VerificationStatus.FAIL,
            evidence={"midline": mid, "checked": checked, "offenders": offenders},
            blocker_reason=(
                "rail-affinity group(s) on the wrong side of the cell midline: "
                + "; ".join(offenders.values())
            ),
            required=required,
        )
    return CheckerVerdict(
        checker="rail_affinity", obligation_id="place.rail_affinity",
        status=VerificationStatus.PASS,
        evidence={"midline": mid, "checked": checked},
        required=required,
    )


def _norm_edge(side: Any) -> str:
    s = str(side).lower().strip()
    if s in _TOP:
        return "top"
    if s in _BOTTOM:
        return "bottom"
    if s in ("left", "l"):
        return "left"
    if s in ("right", "r"):
        return "right"
    return ""
