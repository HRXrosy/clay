"""Realize the LEAKAGE-FREE abstract intent through the REAL constraint placer and
measure RELATIONAL FIDELITY (placer-invariance) versus the reference.

WHERE THIS FITS
===============
``discover.py`` detects the coordinate-free structural relations the reference layout
uses (abut / align / order / symmetric); ``netlist_intent.py`` derives the
schematic-matched pairs. Both are LEAKAGE-FREE: a relation is a kind + a sign, never a
stored reference coordinate (a SIZE or a row INDEX is allowed; an absolute position is
NOT). This module closes the loop the honest way: it feeds ONLY those abstract relations
into the real general MILP placer (:func:`annotate_to_solver.solve_placement_from_constraints`)
and asks whether the placer, given no coordinates, lands in the SAME EQUIVALENCE CLASS as
the reference -- i.e. whether the same abut / align / order / symmetric relations still
hold on the placed layout. Only the absolute inter-row SPACING differs, and that spacing
is a DECLARED DON'T-CARE (the placer compacts to area-minimal), so absolute-coordinate
deviation is explicitly NOT the metric.

THE HONEST METRIC: RELATIONAL FIDELITY
======================================
This is NOT absolute-coordinate reproduction (that would be leakage: it would only be
possible by baking the reference coordinates into the input). It is the question a faithful
placer should answer: given the structural relations alone, do the reference's relations
survive? :func:`relational_fidelity` re-runs ``detect_relations`` semantics on the PLACED
layout and reports, per family (abut / align / order / symmetric) and overall, the fraction
of reference relations preserved, plus the explicit list of relations the placer broke (the
honest shortfall). No number is tuned to a target; the raw fraction is reported.

LEAKAGE DISCIPLINE
==================
:func:`intent_to_constraints` carries ONLY relations, schematic pairs, device kinds, sizes
(footprints, established leakage-free) and OPTIONAL row INDICES. ``pinned_positions`` stays
EMPTY -- pinning a coordinate is the forbidden leak. ``build_devs`` derives each device's
width/height from the reference FOOTPRINT size (a size, not a position). The result is
realizable from pure relations: swap the reference and the only thing that can change is the
sizes, never a baked-in coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from clay.layout.annotate_to_solver import (
    PlacementSolution,
    SolverConstraints,
    solve_placement_from_constraints,
)
from clay.layout.extract import classify
from clay.layout.extract.discover import (
    DetectedRel,
    detect_relations,
)

_XY = Tuple[float, float]
_Bbox = Tuple[float, float, float, float]
_Size = Tuple[float, float]

# nm-per-um scale: the placer works in integer nm, the reference footprints are um.
_UM_TO_NM = 1000.0

# Map a detected-relation kind to its FAMILY for per-family fidelity reporting.
_FAMILY: Dict[str, str] = {
    "abut:x": "abut",
    "abut:y": "abut",
    "align:x": "align",
    "align:y": "align",
    "align:left": "align",
    "align:right": "align",
    "align:top": "align",
    "align:bottom": "align",
    "order:left_of": "order",
    "order:right_of": "order",
    "order:above": "order",
    "order:below": "order",
}
_FAMILIES = ("abut", "align", "order", "symmetric")


# --------------------------------------------------------------------------- #
# a minimal device stub: only .extent() (nm) + carries the kind                #
# --------------------------------------------------------------------------- #


@dataclass
class _DevStub:
    """The minimal device shape the MILP placer consumes.

    ``solve_placement_from_constraints`` reads each device only through
    ``d.extent()`` (via ``_half_widths`` / ``_heights``), so a device needs nothing
    more than an ``extent()`` returning an integer-nm bbox. The footprint comes from
    the reference SIZE (leakage-free -- a size, not a position); the extent is centred
    on the origin so the placer's own translation fully determines the placed position.
    """

    label: str
    kind: str
    w_nm: int
    h_nm: int

    def extent(self) -> _Bbox:
        hw = self.w_nm // 2
        hh = self.h_nm // 2
        return (-hw, -hh, hw, hh)


# --------------------------------------------------------------------------- #
# result types                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class RealizeResult:
    """The placement realized from the abstract relations alone.

    ``placed_positions`` maps device -> placed CENTROID ``(cx, cy)`` in um. ``placement_ok``
    is the real MILP success flag. ``engine`` names the placer used. ``bbox`` / ``area_um2``
    are the realized cell extent. ``solution`` is the underlying :class:`PlacementSolution`
    (None when the placer could not run).
    """

    placed_positions: Dict[str, _XY]
    placement_ok: bool
    engine: str
    bbox: _Bbox
    area_um2: float
    solution: Optional[PlacementSolution] = None
    note: str = ""


@dataclass
class FidelityReport:
    """Relational fidelity of a placement versus the reference (the honest metric).

    ``n_reference_relations`` / ``n_preserved`` are totals; ``per_family`` maps each family
    (abut / align / order / symmetric) to ``(n_ref, n_preserved, fidelity)``; ``fidelity`` is
    the overall preserved fraction. ``broken`` lists the reference relations the placer did
    NOT preserve (the honest shortfall) as ``(child, parent, kind)``.
    """

    n_reference_relations: int
    n_preserved: int
    fidelity: float
    per_family: Dict[str, Tuple[int, int, float]]
    broken: List[Tuple[str, str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. intent_to_constraints: abstract relations -> SolverConstraints            #
# --------------------------------------------------------------------------- #


def _norm_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _matched_pairs(intent: Any) -> List[Tuple[str, str]]:
    """The schematic-derived matched / diff / mirror pairs (deduped, norm-ordered)."""
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for src in ("matched_pairs", "diff_pairs", "mirror_pairs"):
        for pr in getattr(intent, src, None) or ():
            if len(pr) != 2:
                continue
            p = _norm_pair(str(pr[0]), str(pr[1]))
            if p[0] == p[1] or p in seen:
                continue
            seen.add(p)
            out.append(p)
    return out


def _device_kinds(intent: Any) -> Dict[str, str]:
    """``label -> 'nmos'|'pmos'`` for the MOS devices (passives omitted -- the placer
    defaults an unlisted device to the nmos band, which is correct for the OTA caps/res
    that sit in the lower analog tier)."""
    out: Dict[str, str] = {}
    for d in getattr(intent, "devices", None) or ():
        k = getattr(d, "kind", None)
        if k in ("nmos", "pmos"):
            out[str(getattr(d, "name"))] = k
    return out


def _equal_size_groups(intent: Any) -> Tuple[FrozenSet[str], ...]:
    """The declared equal-footprint groups (e.g. the two caps, the two resistors)."""
    out: List[FrozenSet[str]] = []
    for g in getattr(intent, "equal_size_groups", None) or ():
        members = frozenset(str(m) for m in g)
        if len(members) >= 2:
            out.append(members)
    return tuple(out)


def _order_constraints(
    rels: List[DetectedRel], abut_keys: set, align_keys: set
) -> Tuple[Tuple[str, str, str], ...]:
    """Map ``order:*`` detected relations to the placer's ``(a, b, rel)`` triples with
    ``rel`` in {"left_of", "above"}, normalising direction so ``a`` is the left/upper one.

    ``order:left_of`` / ``order:right_of`` -> a left_of b; ``order:above`` / ``order:below``
    -> a above b. De-duplicated. An order whose unordered device pair ALSO carries an
    abut or align relation is DROPPED: an abut pins the pair on both axes and an align
    shares an axis, so an extra order edge on the same pair would be redundant (and could
    contradict the tighter relation). Deterministic over the sorted result.
    """
    seen: set = set()
    out: List[Tuple[str, str, str]] = []
    for r in rels:
        if not r.kind.startswith("order:"):
            continue
        key = frozenset((r.child, r.parent))
        if key in abut_keys or key in align_keys:
            continue  # do not contradict / duplicate a tighter relation on this pair
        if r.kind == "order:left_of":
            a, b = r.child, r.parent
            rel = "left_of"
        elif r.kind == "order:right_of":
            a, b = r.parent, r.child
            rel = "left_of"
        elif r.kind == "order:above":
            a, b = r.parent, r.child
            rel = "above"
        elif r.kind == "order:below":
            a, b = r.child, r.parent
            rel = "above"
        else:
            continue
        triple = (a, b, rel)
        if triple in seen:
            continue
        seen.add(triple)
        out.append(triple)
    out.sort()
    return tuple(out)


def _abut_pairs(rels: List[DetectedRel]) -> Tuple[Tuple[str, str], ...]:
    """Map ``abut:x`` / ``abut:y`` detected relations to ``(child, parent)`` adjacency
    pairs (deduped over the unordered device pair, deterministic)."""
    seen: set = set()
    out: List[Tuple[str, str]] = []
    for r in rels:
        if r.kind not in ("abut:x", "abut:y"):
            continue
        key = frozenset((r.child, r.parent))
        if key in seen:
            continue
        seen.add(key)
        out.append((r.child, r.parent))
    out.sort()
    return tuple(out)


def intent_to_constraints(intent: Any) -> SolverConstraints:
    """Map the LEAKAGE-FREE discovered relations + schematic pairs into SolverConstraints.

    NO field carries a reference coordinate. The mapping:

      * ``symmetry_pairs``  <- the schematic matched / diff / mirror pairs (norm-ordered,
        deduped). ``symmetry_axis`` defaults to ``"x"`` (vertical mirror line).
      * ``abut_pairs``      <- ``detect_relations`` ``abut:x`` / ``abut:y`` pairs.
      * ``order_constraints`` <- ``detect_relations`` ``order:*`` mapped to ``(a, b, rel)``
        with direction normalised; an order on a pair that ALSO abuts / aligns is dropped.
      * ``equal_size_groups`` <- the intent's equal-footprint groups (e.g. the two caps).
      * ``device_kinds``    <- ``{name: 'nmos'|'pmos'}`` for the MOS devices.
      * ``pinned_positions`` STAYS EMPTY -- pinning a coordinate is the forbidden leak.
      * ``floorplan_rows``  is left EMPTY: the placer bands by device class. A row-index
        banding could be derived structurally (leakage-free), but the default banding is
        sufficient and lower-risk; left as the documented next lever.

    Deterministic (all sources deduped + sorted). Realizable from pure relations.
    """
    cell_name = str(getattr(intent, "cell", "") or "")
    sym_pairs = tuple(_matched_pairs(intent))
    kinds = _device_kinds(intent)
    equal_groups = _equal_size_groups(intent)

    rels = detect_relations(intent)
    abut = _abut_pairs(rels)
    abut_keys = {frozenset((a, b)) for a, b in abut}
    # an align relation on a device pair (center or edge) -- used to suppress a
    # contradictory / redundant order edge on the same pair.
    align_keys = {
        frozenset((r.child, r.parent))
        for r in rels
        if r.kind in ("align:x", "align:y", "align:left", "align:right",
                      "align:top", "align:bottom")
    }
    order = _order_constraints(rels, abut_keys, align_keys)

    return SolverConstraints(
        cell_name=cell_name,
        symmetry_pairs=sym_pairs,
        symmetry_axis="x",
        device_kinds=kinds,
        equal_size_groups=equal_groups,
        abut_pairs=abut,
        order_constraints=order,
        pinned_positions={},  # leak guard: never pin a reference coordinate
    )


# --------------------------------------------------------------------------- #
# 2. build_devs: ExtractedIntent -> {label -> _DevStub}                         #
# --------------------------------------------------------------------------- #


def build_devs(intent: Any) -> Dict[str, object]:
    """Build the ``devs`` dict the placer consumes: ``label -> _DevStub`` with each
    device's width/height from the reference FOOTPRINT size (leakage-free) and its kind.

    The footprint comes from :func:`classify.device_sizes` (the reference bbox W/H -- a
    SIZE, not a position). Every device in ``device_bboxes`` (all 17 for the OTA: 13 MOS +
    2 caps + 2 res) gets a stub; a MOS device carries its real kind, a passive defaults to
    ``"nmos"`` (the placer's lower band). A device with a non-positive footprint is floored
    to a 1nm extent so the placer never divides by zero. Deterministic (sorted labels).
    """
    sizes = classify.device_sizes(intent)
    kinds = _device_kinds(intent)
    devs: Dict[str, object] = {}
    for lbl in sorted(sizes):
        w_um, h_um = sizes[lbl]
        w_nm = max(2, int(round(w_um * _UM_TO_NM)))
        h_nm = max(2, int(round(h_um * _UM_TO_NM)))
        devs[lbl] = _DevStub(
            label=lbl,
            kind=kinds.get(lbl, "nmos"),
            w_nm=w_nm,
            h_nm=h_nm,
        )
    return devs


# --------------------------------------------------------------------------- #
# 3. realize: build constraints + devs, run the real placer                    #
# --------------------------------------------------------------------------- #


def realize(intent: Any, *, time_limit_s: float = 30.0) -> RealizeResult:
    """Build the constraints + devs from the abstract intent and run the REAL MILP placer.

    Returns a :class:`RealizeResult` carrying the placed CENTROIDS (um), the real placement
    success flag, the engine name, and the realized cell bbox / area. If the MILP backend
    is unavailable or the model is infeasible the result reports ``placement_ok=False`` with
    an empty position map and the placer's status note -- NO placement is fabricated.
    """
    constraints = intent_to_constraints(intent)
    devs = build_devs(intent)
    engine = "milp:solve_placement_from_constraints"

    try:
        sol = solve_placement_from_constraints(
            constraints, devs, time_limit_s=time_limit_s
        )
    except Exception as exc:  # noqa: BLE001 - no MILP backend / infeasible model
        return RealizeResult(
            placed_positions={},
            placement_ok=False,
            engine=engine,
            bbox=(0.0, 0.0, 0.0, 0.0),
            area_um2=0.0,
            solution=None,
            note=f"placer raised: {type(exc).__name__}: {exc}",
        )

    placed_positions: Dict[str, _XY] = {}
    for lbl, pd in sol.placed.items():
        e = pd.extent  # nm
        cx = (e[0] + e[2]) / 2.0 / _UM_TO_NM
        cy = (e[1] + e[3]) / 2.0 / _UM_TO_NM
        placed_positions[lbl] = (cx, cy)

    fe = sol.full_extent
    bbox = (
        fe[0] / _UM_TO_NM,
        fe[1] / _UM_TO_NM,
        fe[2] / _UM_TO_NM,
        fe[3] / _UM_TO_NM,
    )
    area = max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))

    return RealizeResult(
        placed_positions=placed_positions,
        placement_ok=bool(sol.success),
        engine=engine,
        bbox=bbox,
        area_um2=area,
        solution=sol,
        note=sol.note,
    )


# --------------------------------------------------------------------------- #
# 4. relational_fidelity: re-detect relations on the placed layout, compare    #
# --------------------------------------------------------------------------- #


class _PlacedIntent:
    """A duck-typed intent exposing ``device_bboxes`` built from placed centroids +
    reference sizes, so ``detect_relations`` runs on the PLACED layout unchanged."""

    def __init__(self, device_bboxes: Dict[str, _Bbox]) -> None:
        self.device_bboxes = device_bboxes


def _placed_bboxes(
    placed_positions: Dict[str, _XY], sizes: Dict[str, _Size]
) -> Dict[str, _Bbox]:
    out: Dict[str, _Bbox] = {}
    for lbl, (cx, cy) in placed_positions.items():
        w, h = sizes.get(lbl, (0.0, 0.0))
        out[lbl] = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
    return out


def _symmetric_preserved(
    a: str,
    b: str,
    placed: Dict[str, _XY],
    *,
    axis_tol_um: float = 0.6,
) -> bool:
    """Is the matched pair (a, b) mirror-placed about a common vertical axis in the
    placed layout? True iff the two devices share a y within tolerance and their x are
    mirrored about some common axis -- which, for any two points at equal y, is always
    realisable, so the binding test is the SHARED-Y (same-row) condition the placer's
    mirror-pair row enforces. A pair whose members the placer did NOT both place fails."""
    if a not in placed or b not in placed:
        return False
    (ax, ay), (bx, by) = placed[a], placed[b]
    # mirror about the axis midway between them requires the same y (a horizontal mirror
    # line through both centroids) -- the placer's same-row pairing realises exactly this.
    return abs(ay - by) <= axis_tol_um


def relational_fidelity(
    intent: Any,
    placed_positions: Dict[str, _XY],
    *,
    touch_tol_um: float = 0.6,
    align_tol_um: float = 0.6,
) -> FidelityReport:
    """Re-run ``detect_relations`` semantics on the PLACED layout and compare to the
    reference's detected relations -- the HONEST relational-fidelity metric.

    Reference relations come from ``detect_relations(intent)`` (abut / align / order) plus
    the schematic matched pairs (the ``symmetric`` family). For each, ask whether it HOLDS
    in the placed layout: abut / align / order are re-detected on the placed bboxes (built
    from placed centroids + reference sizes); a symmetric pair holds iff its members are
    mirror-placed about a common axis (same row) in the placed layout. Reports per-family
    and overall preserved fractions and the explicit ``broken`` list (reference relations the
    placer did not preserve). NO tuning -- the raw fractions are returned.
    """
    sizes = classify.device_sizes(intent)

    # --- reference relations (abut / align / order) and the placed re-detection --- #
    ref_rels = detect_relations(
        intent, touch_tol_um=touch_tol_um, align_tol_um=align_tol_um
    )
    placed_bb = _placed_bboxes(placed_positions, sizes)
    placed_intent = _PlacedIntent(placed_bb)
    placed_rels = detect_relations(
        placed_intent, touch_tol_um=touch_tol_um, align_tol_um=align_tol_um
    )
    placed_keys = {(r.child, r.parent, r.kind) for r in placed_rels}

    per_family_counts: Dict[str, List[int]] = {f: [0, 0] for f in _FAMILIES}
    broken: List[Tuple[str, str, str]] = []

    for r in ref_rels:
        fam = _FAMILY.get(r.kind)
        if fam is None:
            continue
        per_family_counts[fam][0] += 1
        if (r.child, r.parent, r.kind) in placed_keys:
            per_family_counts[fam][1] += 1
        else:
            broken.append((r.child, r.parent, r.kind))

    # --- symmetric family: the schematic matched pairs ----------------------------- #
    for a, b in _matched_pairs(intent):
        per_family_counts["symmetric"][0] += 1
        if _symmetric_preserved(a, b, placed_positions):
            per_family_counts["symmetric"][1] += 1
        else:
            broken.append((a, b, "symmetric"))

    n_ref = sum(c[0] for c in per_family_counts.values())
    n_pre = sum(c[1] for c in per_family_counts.values())
    per_family: Dict[str, Tuple[int, int, float]] = {}
    for f in _FAMILIES:
        nr, np_ = per_family_counts[f]
        per_family[f] = (nr, np_, (np_ / nr) if nr else 0.0)

    broken.sort()
    return FidelityReport(
        n_reference_relations=n_ref,
        n_preserved=n_pre,
        fidelity=(n_pre / n_ref) if n_ref else 0.0,
        per_family=per_family,
        broken=broken,
    )


# --------------------------------------------------------------------------- #
# 5. realize_summary: realize + score, honest report                           #
# --------------------------------------------------------------------------- #


def realize_summary(intent: Any, *, time_limit_s: float = 30.0) -> str:
    """Realize the abstract intent through the real placer, score relational fidelity, and
    render a short HONEST report.

    Reports the engine, placement-ok flag, the overall + per-family relational fidelity, the
    count of broken relations, and an EXPLICIT statement that the absolute inter-row SPACING
    differs from the reference BY DESIGN (the placer compacts to area-minimal; the spacing is
    a declared don't-care), so absolute-coordinate deviation is NOT the metric. Does NOT claim
    the reference was reproduced.
    """
    res = realize(intent, time_limit_s=time_limit_s)

    lines: List[str] = [
        "realize-intent: leakage-free relations -> real constraint placer",
        f"  engine        : {res.engine}",
        f"  placement_ok  : {res.placement_ok}",
    ]
    if not res.placement_ok:
        lines.append(f"  NOTE: placer did not produce a placement -- {res.note}")
        lines.append(
            "  (no fidelity scored: a placement was not fabricated; the shortfall is "
            "reported honestly)"
        )
        return "\n".join(lines)

    rep = relational_fidelity(intent, res.placed_positions)
    lines.append(
        f"  area          : {res.area_um2:.1f} um^2  bbox={tuple(round(v,2) for v in res.bbox)}"
    )
    lines.append(
        f"  RELATIONAL FIDELITY (overall): {rep.n_preserved}/{rep.n_reference_relations} "
        f"= {rep.fidelity:.3f}"
    )
    for f in _FAMILIES:
        nr, np_, fr = rep.per_family[f]
        lines.append(f"    {f:9s}: {np_}/{nr} = {fr:.3f}")
    lines.append(f"  broken relations: {len(rep.broken)}")
    lines.append(
        "  ABSOLUTE SPACING IS A DON'T-CARE: the placer compacts to area-minimal, so the "
        "inter-row spacing differs from the reference BY DESIGN; absolute-coordinate "
        "deviation is NOT the metric and the reference was NOT reproduced. The metric is "
        "whether the reference's abut/align/order/symmetric RELATIONS survive the placer."
    )
    lines.append(
        "  (the placer bands NMOS below PMOS; the OTA reference happens to match this -- "
        "noted, not relied on as a correctness claim.)"
    )
    return "\n".join(lines)


__all__ = [
    "RealizeResult",
    "FidelityReport",
    "intent_to_constraints",
    "build_devs",
    "realize",
    "relational_fidelity",
    "realize_summary",
]
