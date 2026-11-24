"""LAYOUT VALIDITY ORACLE -- the accept/reject test the Phase-B perturbation
classifier needs.

WHAT THIS IS
============
The no-data-leakage degradation experiment classifies every RESIDUAL reference
coordinate as either a MISSING PRIMITIVE (a declared relation pins its value) or
a DON'T-CARE (genuine design freedom) by PERTURBING the coordinate and asking:
did the layout STAY VALID?  This module is the validity test that question runs
against.

  perturb a residual coordinate
    -> ``is_valid`` STAYS True  (and design quality preserved) -> DON'T-CARE
    -> ``is_valid`` GOES False  (only the original value was valid)
                                -> a relation pins it -> MISSING PRIMITIVE

VALIDITY (three independent criteria)
=====================================
A layout (device positions + sizes) is VALID iff ALL THREE hold:

  (a) OVERLAP-FREE
        no two device bounding boxes overlap in their INTERIORS. Edge-sharing
        (an @abut face touching at gap 0) is NOT an interior overlap, so abutted
        devices stay valid. A small ``overlap_eps_um`` lets a face touch / a
        sub-micron grid-snap residue count as "not overlapping".

  (b) DRC-SPACING-LEGAL
        every inter-device gap between NON-abutting devices is >= the PDK
        minimum spacing. We use a ``min_gap_um`` proxy for that minimum (a real
        DRC deck would supersede it -- see ``drc_deck`` note on the report).
        Devices a declared @abut joins are EXEMPT: abut intent says they touch.

  (c) DECLARED RELATIONS HOLD
        every structural edge the intent declares still holds on the given
        positions:
          * abut    -- each @abut pair still touches (face gap within ``abut_eps_um``)
          * align   -- each align group still shares its axis (x / y / centroid)
          * symmetric -- each declared symmetric pair still mirrors about the axis
          * order   -- each order edge (left_of / above) still holds

The oracle reads the SAME structural fields the geometric-intent extractor fills
(``abuts`` / ``aligns`` / ``orders`` / ``symmetry_axis`` + the topology pair
lists ``matched_pairs`` / ``diff_pairs`` / ``mirror_pairs``), so what the
extractor DECLARED is exactly what the oracle CHECKS -- agreement by construction.

REPORT
======
``is_valid`` returns a :class:`ValidityReport`: an overall ``valid`` bool, a
``per_criterion`` {name -> bool} summary, and a ``violations`` list naming each
specific failure (kind + the devices + the measured vs required quantity), so a
caller can see WHY a perturbation broke validity, not just THAT it did.

UNITS: every coordinate / size is in MICRONS (the shared ``device_bboxes``
contract ``(x0,y0,x1,y1) um``). ``positions`` are device CENTROIDS ``(cx, cy)``;
``sizes`` are footprints ``(w, h)``; a device's bbox is reconstructed as
``(cx-w/2, cy-h/2, cx+w/2, cy+h/2)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Defaults (microns). These are deliberately permissive proxies for this oracle
# layer; a wired DRC deck supplies the real numbers (noted on the report).
DEFAULT_MIN_GAP_UM: float = 0.20      # min-spacing proxy between non-abutting devices
DEFAULT_OVERLAP_EPS_UM: float = 0.05  # face-touch / grid-snap slack: interiors may
#                                       overlap by up to this and still count clean
DEFAULT_ABUT_EPS_UM: float = 0.05     # an @abut face counts as touching within this

_Bbox = Tuple[float, float, float, float]
_Pos = Tuple[float, float]
_Size = Tuple[float, float]


# --------------------------------------------------------------------------- #
# report types                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class Violation:
    """One specific way the layout is invalid.

    ``kind`` is the criterion (``overlap`` / ``spacing`` / ``abut`` / ``align`` /
    ``symmetry`` / ``order``); ``devices`` names the device(s) involved;
    ``measured`` / ``required`` carry the offending quantity (um) when applicable;
    ``detail`` is a human string.
    """

    kind: str
    devices: Tuple[str, ...]
    detail: str
    measured: Optional[float] = None
    required: Optional[float] = None


@dataclass
class ValidityReport:
    """Result of :func:`is_valid`.

    ``valid`` is the AND of the three criteria. ``per_criterion`` maps each
    criterion name -> bool (``overlap_free`` / ``drc_spacing_legal`` /
    ``relations_hold``). ``violations`` lists every specific failure. ``notes``
    records honest caveats (e.g. the min-gap PROXY in use, or that a real DRC
    deck would supersede criterion (b)).
    """

    valid: bool
    per_criterion: Dict[str, bool]
    violations: List[Violation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # so ``if is_valid(...)`` reads naturally
        return self.valid

    def violations_of(self, kind: str) -> List[Violation]:
        return [v for v in self.violations if v.kind == kind]


# --------------------------------------------------------------------------- #
# geometry helpers                                                            #
# --------------------------------------------------------------------------- #


def _bbox_of(name: str, positions: Dict[str, _Pos], sizes: Dict[str, _Size]) -> _Bbox:
    cx, cy = positions[name]
    w, h = sizes.get(name, (0.0, 0.0))
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _all_bboxes(
    positions: Dict[str, _Pos], sizes: Dict[str, _Size]
) -> Dict[str, _Bbox]:
    return {n: _bbox_of(n, positions, sizes) for n in positions}


def _center(bb: _Bbox) -> _Pos:
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def _interior_overlap(a: _Bbox, b: _Bbox, eps: float) -> float:
    """Positive interior overlap depth (min of x / y overlap), shrunk by ``eps``.

    Two bboxes overlap in their INTERIORS iff they overlap on BOTH axes by more
    than ``eps``. Returns the overlap depth on the lesser axis (>0 means overlap,
    <=0 means clean / merely touching). Shrinking by ``eps`` lets an abut face
    (gap 0) or a sub-um grid residue read as not-overlapping.
    """
    ox = min(a[2], b[2]) - max(a[0], b[0])  # >0 means they overlap on x
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return min(ox, oy) - eps


def _edge_gap(a: _Bbox, b: _Bbox) -> Tuple[float, float]:
    """(gap_x, gap_y): edge-to-edge separation on each axis (negative = overlap)."""
    gx = max(a[0] - b[2], b[0] - a[2])
    gy = max(a[1] - b[3], b[1] - a[3])
    return gx, gy


def _touch_gap(a: _Bbox, b: _Bbox) -> float:
    """Face gap for an @abut pair: the separation on the touching axis.

    Abutting bboxes share a face on one axis (gap ~ 0 there) while overlapping on
    the other (gap < 0 there). The face gap is therefore the LARGER (closer to 0
    or positive) of the two axis gaps -- if they truly abut it is ~0; if they were
    pulled apart it becomes the positive distance between them.
    """
    gx, gy = _edge_gap(a, b)
    return max(gx, gy)


# --------------------------------------------------------------------------- #
# read declared structural edges off the intent                              #
# --------------------------------------------------------------------------- #


def _declared_symmetric_pairs(intent: Any) -> List[Tuple[str, str]]:
    """The intent's declared symmetric pairs (matched / diff / mirror), deduped.

    These are the topology pairs the netlist half fills; criterion (c) checks each
    still mirrors about ``symmetry_axis``.
    """
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for attr in ("matched_pairs", "diff_pairs", "mirror_pairs"):
        for ab in getattr(intent, attr, ()) or ():
            if len(ab) != 2:
                continue
            a, b = str(ab[0]), str(ab[1])
            key = (a, b) if a <= b else (b, a)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


# --------------------------------------------------------------------------- #
# the public oracle                                                           #
# --------------------------------------------------------------------------- #


def is_valid(
    positions: Dict[str, _Pos],
    sizes: Dict[str, _Size],
    intent: Any,
    *,
    min_gap_um: float = DEFAULT_MIN_GAP_UM,
    overlap_eps_um: float = DEFAULT_OVERLAP_EPS_UM,
    abut_eps_um: float = DEFAULT_ABUT_EPS_UM,
    align_eps_um: float = 0.5,
    order_eps_um: float = 1e-6,
    drc_deck: Any = None,
    baseline_positions: Optional[Dict[str, _Pos]] = None,
) -> ValidityReport:
    """Is the layout (``positions`` + ``sizes``) VALID under ``intent``?

    See the module docstring for the three criteria. ``positions`` are device
    centroids ``{name -> (cx, cy)}``; ``sizes`` are footprints ``{name -> (w, h)}``
    (a missing size defaults to a point footprint). Only devices present in
    ``positions`` are considered; a declared edge naming an absent device is
    skipped (it cannot be checked, not a violation).

    ``min_gap_um`` is the min-spacing PROXY for criterion (b). If ``drc_deck`` is
    given we note it on the report (a real deck would supersede the proxy); this
    oracle layer still uses the min-gap check -- enough to tell a relation-pinned
    coordinate from a free one.

    SYMMETRY SUB-CHECK OF (c) -- BASELINE GATING. A declared symmetric pair is a
    matched/mirror ELECTRICAL role; in a folded/interdigitated reference such a
    pair need not be a GEOMETRIC reflection (the RF reference here is exactly that
    case -- its topology pairs sit side-by-side, not mirrored). Checking geometric
    reflection on a pair that never mirrored would flag the reference itself. So
    the symmetry sub-check only enforces "still mirrors about the axis" for pairs
    that DID mirror in ``baseline_positions`` (the reference). When
    ``baseline_positions`` is None the GIVEN layout is its own baseline (a pair is
    enforced only if it already mirrors there) -- so ``is_valid`` never rejects a
    layout for a symmetry the layout itself does not exhibit. The perturbation
    classifier passes the reference as ``baseline_positions`` so a perturbed
    layout is judged against the relations the reference actually had.

    Returns a :class:`ValidityReport` (truthy iff valid).
    """
    bboxes = _all_bboxes(positions, sizes)
    base_pos = baseline_positions if baseline_positions is not None else positions
    base_bboxes = _all_bboxes(base_pos, sizes)
    names = sorted(bboxes)
    present = set(names)
    violations: List[Violation] = []
    notes: List[str] = []

    # abut pairs (both present) -- exempt from the spacing rule and checked in (c).
    abut_pairs = [
        (str(a), str(b))
        for (a, b) in (getattr(intent, "abuts", None) or [])
        if str(a) in present and str(b) in present
    ]
    abut_set = {(a, b) if a <= b else (b, a) for a, b in abut_pairs}

    # --- (a) OVERLAP-FREE ------------------------------------------------- #
    for a, b in combinations(names, 2):
        depth = _interior_overlap(bboxes[a], bboxes[b], overlap_eps_um)
        if depth > 0:
            violations.append(
                Violation(
                    kind="overlap",
                    devices=(a, b),
                    detail=f"{a} and {b} overlap in interior by {depth:.4f} um",
                    measured=round(depth, 4),
                    required=0.0,
                )
            )

    # --- (b) DRC-SPACING-LEGAL -------------------------------------------- #
    if drc_deck is not None:
        notes.append(
            "drc_deck supplied: a real DRC deck would supersede the min-gap proxy; "
            "this layer still applies the min_gap_um spacing check."
        )
    else:
        notes.append(
            f"criterion (b) uses a min-gap PROXY ({min_gap_um:.3f} um); no DRC "
            "deck wired -- a real deck would supersede it."
        )
    for a, b in combinations(names, 2):
        key = (a, b) if a <= b else (b, a)
        if key in abut_set:
            continue  # abut intent says they touch -- spacing rule does not apply
        if _interior_overlap(bboxes[a], bboxes[b], overlap_eps_um) > 0:
            continue  # already an overlap violation; spacing is undefined
        gx, gy = _edge_gap(bboxes[a], bboxes[b])
        sep = max(gx, gy)  # separation on the axis that actually separates them
        if 0.0 <= sep < min_gap_um:
            violations.append(
                Violation(
                    kind="spacing",
                    devices=(a, b),
                    detail=f"{a}-{b} gap {sep:.4f} um < min {min_gap_um:.3f} um",
                    measured=round(sep, 4),
                    required=min_gap_um,
                )
            )

    # --- (c) DECLARED RELATIONS HOLD -------------------------------------- #
    # abut: each declared abut pair still touches (face gap within abut_eps).
    for a, b in abut_pairs:
        gap = _touch_gap(bboxes[a], bboxes[b])
        if gap > abut_eps_um:
            violations.append(
                Violation(
                    kind="abut",
                    devices=(a, b),
                    detail=f"abut {a}-{b} pulled apart: face gap {gap:.4f} um "
                    f"> eps {abut_eps_um:.3f} um",
                    measured=round(gap, 4),
                    required=abut_eps_um,
                )
            )

    # align: each align group still shares its axis.
    for grp in getattr(intent, "aligns", None) or []:
        axis, members = grp[0], [str(m) for m in grp[1]]
        members = [m for m in members if m in present]
        if len(members) < 2:
            continue
        cxs = [_center(bboxes[m])[0] for m in members]
        cys = [_center(bboxes[m])[1] for m in members]
        if axis in ("x", "centroid"):  # shared x-centre (a column)
            spread = max(cxs) - min(cxs)
            if spread > align_eps_um:
                violations.append(
                    Violation(
                        kind="align",
                        devices=tuple(members),
                        detail=f"x-align broken: x-centre spread {spread:.4f} um "
                        f"> eps {align_eps_um:.3f} um",
                        measured=round(spread, 4),
                        required=align_eps_um,
                    )
                )
        if axis in ("y", "centroid"):  # shared y-centre (a row)
            spread = max(cys) - min(cys)
            if spread > align_eps_um:
                violations.append(
                    Violation(
                        kind="align",
                        devices=tuple(members),
                        detail=f"y-align broken: y-centre spread {spread:.4f} um "
                        f"> eps {align_eps_um:.3f} um",
                        measured=round(spread, 4),
                        required=align_eps_um,
                    )
                )

    # symmetry: each declared symmetric pair that MIRRORED in the baseline still
    # mirrors about symmetry_axis. Baseline-gating (see docstring): a topology pair
    # that is not a geometric reflection in the reference is NOT enforced here, so
    # a folded/interdigitated reference is not flagged for a symmetry it never had.
    axis_x = getattr(intent, "symmetry_axis", None)

    def _mirror_residual(
        bb: Dict[str, _Bbox], a: str, b: str
    ) -> Tuple[float, float, bool]:
        xa, ya = _center(bb[a])
        xb, yb = _center(bb[b])
        arm = abs(abs(xa - axis_x) - abs(xb - axis_x))
        same = (xa - axis_x) * (xb - axis_x) > align_eps_um * align_eps_um
        return arm, abs(ya - yb), same

    if axis_x is not None:
        for a, b in _declared_symmetric_pairs(intent):
            if a not in present or b not in present:
                continue
            if a not in base_bboxes or b not in base_bboxes:
                continue
            # gate: only enforce pairs that actually mirror in the baseline.
            barm, bymis, bsame = _mirror_residual(base_bboxes, a, b)
            mirrored_in_baseline = (
                barm <= align_eps_um and bymis <= align_eps_um and not bsame
            )
            if not mirrored_in_baseline:
                continue
            arm_mismatch, y_mismatch, same_side = _mirror_residual(bboxes, a, b)
            if arm_mismatch > align_eps_um or same_side or y_mismatch > align_eps_um:
                violations.append(
                    Violation(
                        kind="symmetry",
                        devices=(a, b),
                        detail=f"symmetry {a}-{b} broken about x={axis_x:.3f}: "
                        f"arm mismatch {arm_mismatch:.4f}, y mismatch {y_mismatch:.4f}"
                        + (", same side of axis" if same_side else ""),
                        measured=round(max(arm_mismatch, y_mismatch), 4),
                        required=align_eps_um,
                    )
                )

    # order: each order edge still holds (left_of / above).
    for edge in getattr(intent, "orders", None) or []:
        if len(edge) != 3:
            continue
        a, rel, b = str(edge[0]), str(edge[1]), str(edge[2])
        if a not in present or b not in present:
            continue
        ca, cb = _center(bboxes[a]), _center(bboxes[b])
        if rel == "left_of":
            ok = ca[0] <= cb[0] + order_eps_um
            axis_desc = f"x({a})={ca[0]:.3f} vs x({b})={cb[0]:.3f}"
        elif rel == "above":
            ok = ca[1] >= cb[1] - order_eps_um
            axis_desc = f"y({a})={ca[1]:.3f} vs y({b})={cb[1]:.3f}"
        else:
            continue  # unknown relation -> not checked
        if not ok:
            violations.append(
                Violation(
                    kind="order",
                    devices=(a, b),
                    detail=f"order {a} {rel} {b} broken: {axis_desc}",
                )
            )

    overlap_free = not any(v.kind == "overlap" for v in violations)
    spacing_legal = not any(v.kind == "spacing" for v in violations)
    relations_hold = not any(
        v.kind in ("abut", "align", "symmetry", "order") for v in violations
    )
    per_criterion = {
        "overlap_free": overlap_free,
        "drc_spacing_legal": spacing_legal,
        "relations_hold": relations_hold,
    }
    valid = overlap_free and spacing_legal and relations_hold
    return ValidityReport(
        valid=valid,
        per_criterion=per_criterion,
        violations=violations,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# validity delta -- did a perturbation break validity?                       #
# --------------------------------------------------------------------------- #


@dataclass
class ValidityDelta:
    """Comparison of two layouts' validity (``a`` = before, ``b`` = after).

    ``broke`` is True iff ``a`` was valid and ``b`` is not (the perturbation
    BROKE validity -> a relation pinned the moved coordinate). ``fixed`` is the
    reverse. ``stayed_valid`` / ``stayed_invalid`` cover the no-change cases.
    ``newly_violated`` lists the violation kinds present in ``b`` but not ``a``
    (the relations the perturbation broke) and ``newly_satisfied`` the reverse.
    """

    report_a: ValidityReport
    report_b: ValidityReport
    broke: bool
    fixed: bool
    stayed_valid: bool
    stayed_invalid: bool
    newly_violated: List[Tuple[str, Tuple[str, ...]]]
    newly_satisfied: List[Tuple[str, Tuple[str, ...]]]


def _violation_keys(rep: ValidityReport) -> set:
    return {(v.kind, v.devices) for v in rep.violations}


def validity_delta(
    pos_a: Dict[str, _Pos],
    pos_b: Dict[str, _Pos],
    sizes: Dict[str, _Size],
    intent: Any,
    **kwargs: Any,
) -> ValidityDelta:
    """Compare validity of layout ``pos_a`` vs ``pos_b`` (same sizes / intent).

    Used by the perturbation classifier: ``pos_a`` is the reference, ``pos_b`` is
    the reference with ONE residual coordinate perturbed. If validity went
    True -> False (``broke``), only the original value was valid -> a relation
    pins it -> MISSING PRIMITIVE. If it stayed valid, the coordinate is a
    DON'T-CARE (no relation constrains it). Extra ``kwargs`` pass through to
    :func:`is_valid` (e.g. ``min_gap_um``).
    """
    rep_a = is_valid(pos_a, sizes, intent, **kwargs)
    rep_b = is_valid(pos_b, sizes, intent, **kwargs)
    keys_a = _violation_keys(rep_a)
    keys_b = _violation_keys(rep_b)
    newly_violated = sorted(keys_b - keys_a)
    newly_satisfied = sorted(keys_a - keys_b)
    return ValidityDelta(
        report_a=rep_a,
        report_b=rep_b,
        broke=rep_a.valid and not rep_b.valid,
        fixed=(not rep_a.valid) and rep_b.valid,
        stayed_valid=rep_a.valid and rep_b.valid,
        stayed_invalid=(not rep_a.valid) and (not rep_b.valid),
        newly_violated=[(k, d) for k, d in newly_violated],
        newly_satisfied=[(k, d) for k, d in newly_satisfied],
    )


# --------------------------------------------------------------------------- #
# convenience: derive positions + sizes from an intent's device_bboxes        #
# --------------------------------------------------------------------------- #


def positions_sizes_from_bboxes(
    device_bboxes: Dict[str, _Bbox],
) -> Tuple[Dict[str, _Pos], Dict[str, _Size]]:
    """Split ``{name -> (x0,y0,x1,y1)}`` into centroid positions + footprint sizes.

    The reference extractor yields ``device_bboxes``; the oracle takes centroids +
    sizes, so this adapter feeds the reference straight in.
    """
    positions: Dict[str, _Pos] = {}
    sizes: Dict[str, _Size] = {}
    for name, bb in device_bboxes.items():
        x0, y0, x1, y1 = (float(v) for v in bb)
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        positions[name] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        sizes[name] = (x1 - x0, y1 - y0)
    return positions, sizes


__all__ = [
    "is_valid",
    "validity_delta",
    "positions_sizes_from_bboxes",
    "ValidityReport",
    "ValidityDelta",
    "Violation",
    "DEFAULT_MIN_GAP_UM",
    "DEFAULT_OVERLAP_EPS_UM",
    "DEFAULT_ABUT_EPS_UM",
]
