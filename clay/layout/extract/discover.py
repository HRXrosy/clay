"""PHASE-C COMPREHENSIVE LEAKAGE-FREE relation DISCOVERY (drive FLOATING down).

WHERE THIS FITS
===============
``leakfree`` strips the raw-coordinate offset bridges and HONESTLY reports how many
devices float (are NOT in the root's structural component) once the leak is gone --
on the real OTA, 13 of 17. ``power_intent`` re-connects the supply rows with
coordinate-free ``align:y`` rails. This module is the convergence driver above both:
it DISCOVERS the remaining structural primitives the reference layout actually uses
(edge-alignment, abut, stacking order) and adds each one ONLY when it reduces the
floating count -- driving FLOATING down toward the irreducible set (the genuine
don't-cares the perturbation classifier flags).

THE NEW LEAKAGE-FREE PRIMITIVE: EDGE ALIGNMENT
==============================================
The existing structural vocabulary pins a pair's CENTER (``align:x`` / ``align:y``).
But the OTA's MOS clusters do not share a center with the cap/res root cluster -- they
share an EDGE. MM13/MM14's LEFT face sits at the same x as resistor RR0's left face;
MM15's LEFT face coincides with RR5's. That is a structural relation, yet center-align
misses it (the centers differ by the half-width difference).

The KEY INSIGHT that keeps edge-align LEAKAGE-FREE: when two devices share a LEFT edge,
their center-to-center offset is exactly ``-(w_parent - w_child) / 2`` -- a value computed
from the two devices' SIZES, NOT from a stored reference coordinate. So an ``align:left``
edge pins one axis EXACTLY even for unequal-width devices, while persisting no position.
``align:right`` is ``+(w_parent - w_child)/2``; ``align:top`` / ``align:bottom`` are the
height analogues (the full convention is documented at the realization site below).

THE LEAKAGE DISCIPLINE (the whole scientific point)
===================================================
Reading ``intent.device_bboxes`` (the reference positions) is allowed ONLY to DETECT
whether a structural relation holds (the degradation START). Every EMITTED edge is
coordinate-free: a kind + a SIGN, never a stored reference coordinate. The realized
offset of an edge-align edge comes from SIZES at solve time. The final augmented graph
MUST pass :func:`leakfree.assert_no_coordinates` (zero ``offset`` edges); ``converge``
asserts ``leakage == 0`` on its result.

WHAT REMAINS FLOATING IS HONEST
===============================
A device the discovery cannot connect to the root by ANY validated structural relation
stays floating -- and that is the correct answer when the device is a genuine DON'T-CARE
(the perturbation classifier in ``classify.py`` labels it free). The ideal convergence
criterion is ``floating_final`` being a SUBSET of the classifier's don't-care set: every
still-floating device is genuine design freedom, not a primitive we failed to discover.
``convergence_summary`` reports that comparison honestly, naming the gap if any survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from clay.layout.extract import leakfree
from clay.layout.extract import power_intent
from clay.layout.extract.relgraph import (
    ConstraintGraph,
    RelEdge,
    build_constraint_graph,
)

_Bbox = Tuple[float, float, float, float]
_XY = Tuple[float, float]

# Default geometry tolerances (um) for the structural detectors. A facing-edge gap or a
# shared-edge / shared-center delta within tolerance is treated as the relation holding.
_TOUCH_TOL_UM: float = 0.6
_ALIGN_TOL_UM: float = 0.6
# Minimum perpendicular overlap (um) for an abut to count (a real shared face, not a
# corner kiss). Matches the spirit of relgraph._detect_abut's overlap > eps test.
_ABUT_OVERLAP_MIN_UM: float = 0.5

# The new edge-alignment structural kinds (leakage-free: realized from SIZES). They join
# the base leakfree structural kinds {abut, align:x, align:y, symmetric, order:*} to form
# the DISCOVER superset used for component-finding and the leak-guard analog below. We do
# NOT mutate leakfree._STRUCTURAL_KINDS -- we define our own superset.
_EDGE_ALIGN_KINDS = frozenset(
    {"align:left", "align:right", "align:top", "align:bottom"}
)
_DISCOVER_STRUCTURAL_KINDS = frozenset(
    {"abut", "align:x", "align:y", "symmetric"}
) | _EDGE_ALIGN_KINDS

# Priority order for the greedy convergence driver (lower = tried first). Abut pins both
# axes (strongest), then an edge-align pins one axis exactly, then a center-align, then a
# coarse stacking order. Deterministic over (priority, child, parent).
_KIND_PRIORITY: Dict[str, int] = {
    "abut": 0,
    "align:left": 1,
    "align:right": 1,
    "align:top": 1,
    "align:bottom": 1,
    "align:x": 2,
    "align:y": 2,
    "order:left_of": 3,
    "order:right_of": 3,
    "order:above": 3,
    "order:below": 3,
}


# --------------------------------------------------------------------------- #
# small geometry helpers (reference-read, DETECTION only)                     #
# --------------------------------------------------------------------------- #


def _bboxes_of(intent: Any) -> Dict[str, _Bbox]:
    raw = getattr(intent, "device_bboxes", None)
    if raw is None and isinstance(intent, dict):
        raw = intent.get("device_bboxes")
    out: Dict[str, _Bbox] = {}
    for k, v in (raw or {}).items():
        x0, y0, x1, y1 = (float(t) for t in v)
        out[str(k)] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    return out


def _center(bb: _Bbox) -> _XY:
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def _wh(bb: _Bbox) -> _XY:
    return (bb[2] - bb[0], bb[3] - bb[1])


def _sizes_from_intent(intent: Any) -> Dict[str, _XY]:
    """Footprints ``{name -> (w, h)}`` from ``intent.device_bboxes`` (um)."""
    return {n: _wh(bb) for n, bb in _bboxes_of(intent).items()}


# --------------------------------------------------------------------------- #
# DetectedRel: a coordinate-free structural relation                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetectedRel:
    """ONE coordinate-free structural relation detected in the reference.

    ``child`` is pinned relative to ``parent`` by a relation of ``kind`` (one of the
    abut / align-center / align-edge / order kinds). ``sign`` is ``+1`` or ``-1`` where
    a direction is meaningful (which face touches for abut; which way the stack goes for
    order) and ``+1`` (unused) otherwise. NO reference coordinate is stored -- the
    relation realizes its offset from SIZES (edge-align / abut) at solve time, so this
    object is leakage-free by construction.
    """

    child: str
    parent: str
    kind: str
    sign: int = 1


# --------------------------------------------------------------------------- #
# 1. detect_relations: scan every device pair, emit coordinate-free relations #
# --------------------------------------------------------------------------- #


def _detect_pair(
    par: str,
    child: str,
    bb_p: _Bbox,
    bb_c: _Bbox,
    *,
    touch_tol_um: float,
    align_tol_um: float,
) -> List[DetectedRel]:
    """Detect every structural relation that holds for ONE ordered pair (par < child).

    Emits abut (touching faces), align-center (shared centroid axis), align-edge (shared
    left/right/top/bottom face), and -- only when NONE of those abut/align relations
    fired -- a coarse stacking order from the centroid comparison. All coordinate-free.
    """
    out: List[DetectedRel] = []
    px0, py0, px1, py1 = bb_p
    cx0, cy0, cx1, cy1 = bb_c
    cpx, cpy = _center(bb_p)
    ccx, ccy = _center(bb_c)

    y_overlap = min(py1, cy1) - max(py0, cy0)
    x_overlap = min(px1, cx1) - max(px0, cx0)

    structural = False  # any abut/align fired -> suppress the coarse order fallback

    # --- abut: a facing pair of perpendicular-overlapping faces touch ------- #
    # abut:x -> a right face meets a left face (sign = which face of the child touches).
    if y_overlap > _ABUT_OVERLAP_MIN_UM:
        if abs(px1 - cx0) <= touch_tol_um:
            # parent's right meets child's left -> child sits to the RIGHT of parent.
            out.append(DetectedRel(child, par, "abut:x", +1))
            structural = True
        elif abs(cx1 - px0) <= touch_tol_um:
            # child's right meets parent's left -> child sits to the LEFT of parent.
            out.append(DetectedRel(child, par, "abut:x", -1))
            structural = True
    # abut:y -> a top face meets a bottom face.
    if x_overlap > _ABUT_OVERLAP_MIN_UM:
        if abs(py1 - cy0) <= touch_tol_um:
            # parent's top meets child's bottom -> child sits ABOVE parent.
            out.append(DetectedRel(child, par, "abut:y", +1))
            structural = True
        elif abs(cy1 - py0) <= touch_tol_um:
            out.append(DetectedRel(child, par, "abut:y", -1))
            structural = True

    # --- align: shared centroid axis (center) ------------------------------- #
    if abs(ccx - cpx) <= align_tol_um and abs(ccy - cpy) > align_tol_um:
        out.append(DetectedRel(child, par, "align:x", +1))
        structural = True
    if abs(ccy - cpy) <= align_tol_um and abs(ccx - cpx) > align_tol_um:
        out.append(DetectedRel(child, par, "align:y", +1))
        structural = True

    # --- align: shared EDGE (the new leakage-free edge-align kinds) --------- #
    # A shared left/right face pins the x axis EXACTLY; a shared top/bottom face pins y.
    # Detected from the reference faces; REALIZED from SIZES (no coordinate persisted).
    if abs(cx0 - px0) <= align_tol_um:
        out.append(DetectedRel(child, par, "align:left", +1))
        structural = True
    if abs(cx1 - px1) <= align_tol_um:
        out.append(DetectedRel(child, par, "align:right", +1))
        structural = True
    if abs(cy1 - py1) <= align_tol_um:
        out.append(DetectedRel(child, par, "align:top", +1))
        structural = True
    if abs(cy0 - py0) <= align_tol_um:
        out.append(DetectedRel(child, par, "align:bottom", +1))
        structural = True

    # --- order: coarse stacking sequence (only if no abut/align fired) ------ #
    # Emitted only between devices NOT already abut/aligned, to give an inter-row
    # sequence. Direction from the dominant centroid delta; sign-only, no magnitude.
    if not structural:
        dx = ccx - cpx
        dy = ccy - cpy
        if abs(dx) >= abs(dy):
            if dx >= 0:
                out.append(DetectedRel(child, par, "order:right_of", +1))
            else:
                out.append(DetectedRel(child, par, "order:left_of", -1))
        else:
            if dy >= 0:
                out.append(DetectedRel(child, par, "order:above", +1))
            else:
                out.append(DetectedRel(child, par, "order:below", -1))

    return out


def detect_relations(
    intent: Any,
    *,
    touch_tol_um: float = _TOUCH_TOL_UM,
    align_tol_um: float = _ALIGN_TOL_UM,
) -> List[DetectedRel]:
    """Detect every coordinate-free structural relation across all device pairs.

    For every unordered device pair (parent = the smaller name), run the structural
    detectors on the reference geometry and emit the relations that hold:

      * ``abut:x`` / ``abut:y``    -- facing faces touch (gap within ``touch_tol_um`` and
        perpendicular overlap > 0.5um). Sign = which face touches. Pins both axes.
      * ``align:x`` / ``align:y``  -- shared CENTER x (resp. y) within ``align_tol_um``.
      * ``align:left`` / ``align:right`` / ``align:top`` / ``align:bottom`` -- shared
        LEFT / RIGHT / TOP / BOTTOM face within ``align_tol_um`` (the new leakage-free
        edge-align kinds; the realize offset comes from SIZES).
      * ``order:*``                -- coarse stacking from the centroid comparison, emitted
        ONLY between devices that fired no abut/align relation (inter-row sequence).

    DETERMINISTIC: pairs are visited in sorted-name order; the per-pair detector order is
    fixed. Reads ``intent.device_bboxes`` ONLY to DETECT (degradation start); every
    emitted :class:`DetectedRel` is coordinate-free (a kind + a sign).
    """
    bboxes = _bboxes_of(intent)
    names = sorted(bboxes)
    out: List[DetectedRel] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            par, child = names[i], names[j]
            out.extend(
                _detect_pair(
                    par,
                    child,
                    bboxes[par],
                    bboxes[child],
                    touch_tol_um=touch_tol_um,
                    align_tol_um=align_tol_um,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# 2. to_edges: DetectedRel -> RelEdge (structural kinds, no offset edge)       #
# --------------------------------------------------------------------------- #


def to_edges(rels: List[DetectedRel]) -> List[RelEdge]:
    """Convert detected relations to :class:`RelEdge`, all STRUCTURAL (no ``offset``).

    The RelEdge ``dx, dy`` carry only the SIGN needed to realize the relation -- never a
    reference magnitude:

      * ``abut:x`` -> ``kind='abut'``, ``dx = sign``, ``dy = 0`` (horizontal abut).
      * ``abut:y`` -> ``kind='abut'``, ``dx = 0``, ``dy = sign`` (vertical abut).
      * ``align:x`` / ``align:y``                  -> ``dx = dy = 0`` (the solver shares the
        center coordinate; no magnitude used).
      * ``align:left`` / ``align:right`` / ``align:top`` / ``align:bottom`` -> ``dx = dy =
        0``; the offset is computed in the solver from SIZES.
      * ``order:*`` -> ``dx / dy`` carry the sign of the stacking direction.

    ``kind`` is the DetectedRel kind verbatim EXCEPT abut, which collapses ``abut:x`` /
    ``abut:y`` to the single ``'abut'`` kind leakfree understands (the axis is encoded in
    which of dx/dy is non-zero). NO ``offset`` edge is ever produced.
    """
    out: List[RelEdge] = []
    for r in rels:
        s = float(r.sign)
        if r.kind == "abut:x":
            out.append(RelEdge(r.child, r.parent, dx=s, dy=0.0, kind="abut"))
        elif r.kind == "abut:y":
            out.append(RelEdge(r.child, r.parent, dx=0.0, dy=s, kind="abut"))
        elif r.kind in ("align:x", "align:y") or r.kind in _EDGE_ALIGN_KINDS:
            out.append(RelEdge(r.child, r.parent, dx=0.0, dy=0.0, kind=r.kind))
        elif r.kind == "order:left_of":
            out.append(RelEdge(r.child, r.parent, dx=-1.0, dy=0.0, kind="order:left_of"))
        elif r.kind == "order:right_of":
            out.append(RelEdge(r.child, r.parent, dx=+1.0, dy=0.0, kind="order:right_of"))
        elif r.kind == "order:above":
            out.append(RelEdge(r.child, r.parent, dx=0.0, dy=+1.0, kind="order:above"))
        elif r.kind == "order:below":
            out.append(RelEdge(r.child, r.parent, dx=0.0, dy=-1.0, kind="order:below"))
    return out


# --------------------------------------------------------------------------- #
# 3. solve_discovered: leakfree solver EXTENDED with the edge-align kinds      #
# --------------------------------------------------------------------------- #


def _is_discover_structural(edge: RelEdge) -> bool:
    """True iff ``edge`` is a DISCOVER structural relation (base kinds + edge-align)."""
    k = edge.kind
    return k in _DISCOVER_STRUCTURAL_KINDS or k.startswith("order")


def assert_no_coordinates_discovered(graph: ConstraintGraph) -> None:
    """Leak guard for the DISCOVER path: raise if any ``offset`` edge survives.

    Identical contract to :func:`leakfree.assert_no_coordinates` (which only forbids the
    ``offset`` kind) -- the new edge-align kinds are coordinate-free, so the base guard is
    sufficient. Provided as a named DISCOVER-side entry point for symmetry with
    :data:`_DISCOVER_STRUCTURAL_KINDS`.
    """
    leakfree.assert_no_coordinates(graph)


def _solve_component_discovered(
    comp: List[str],
    edges: List[RelEdge],
    sizes: Dict[str, _XY],
    root: str,
    origin: _XY,
) -> Dict[str, _XY]:
    """Place ONE structural component, handling base + edge-align kinds.

    Propagates from a seed (the root if present, else the lowest-named member). The base
    kinds delegate to the exact leakfree placement (abut flush / align shared-center /
    symmetric mirror / order min-sep); the edge-align kinds use the SIZE-derived offset.
    Both directions of every edge are handled so a member reached via the child end is
    still pinned.

    Edge-align convention (offsets from SIZES, never from a reference coordinate):
      * ``align:left``   shared LEFT face  -> ``cx = px - (w_parent - w_child)/2``.
      * ``align:right``  shared RIGHT face -> ``cx = px + (w_parent - w_child)/2``.
      * ``align:top``    shared TOP face   -> ``cy = py + (h_parent - h_child)/2``.
      * ``align:bottom`` shared BOTTOM face-> ``cy = py - (h_parent - h_child)/2``.
    """
    cset = set(comp)
    rel = [e for e in edges if e.child in cset and e.parent in cset]
    seed = root if root in cset else comp[0]
    pos: Dict[str, _XY] = {seed: (float(origin[0]), float(origin[1]))}

    def child_from_parent(e: RelEdge, px: float, py: float) -> _XY:
        wp, hp = sizes.get(e.parent, (0.0, 0.0))
        wc, hc = sizes.get(e.child, (0.0, 0.0))
        if e.kind == "align:left":
            return (px - (wp - wc) / 2.0, py)
        if e.kind == "align:right":
            return (px + (wp - wc) / 2.0, py)
        if e.kind == "align:top":
            return (px, py + (hp - hc) / 2.0)
        if e.kind == "align:bottom":
            return (px, py - (hp - hc) / 2.0)
        # base kinds: same arithmetic as leakfree._solve_component's place_child.
        if e.kind == "abut":
            if abs(e.dx) >= abs(e.dy):
                sep = (wp + wc) / 2.0
                return (px + (sep if e.dx >= 0 else -sep), py)
            sep = (hp + hc) / 2.0
            return (px, py + (sep if e.dy >= 0 else -sep))
        if e.kind in ("align:x", "align:y"):
            return (px, py)
        if e.kind == "symmetric":
            mid = px + (e.dx / 2.0)
            return (2.0 * mid - px, py)
        if e.kind == "order:left_of":
            return (px + (wp + wc) / 2.0, py)
        if e.kind == "order:right_of":
            return (px - (wp + wc) / 2.0, py)
        if e.kind == "order:above":
            return (px, py + (hp + hc) / 2.0)
        if e.kind == "order:below":
            return (px, py - (hp + hc) / 2.0)
        return (px, py)

    def parent_from_child(e: RelEdge, cx: float, cy: float) -> _XY:
        wp, hp = sizes.get(e.parent, (0.0, 0.0))
        wc, hc = sizes.get(e.child, (0.0, 0.0))
        if e.kind == "align:left":
            return (cx + (wp - wc) / 2.0, cy)
        if e.kind == "align:right":
            return (cx - (wp - wc) / 2.0, cy)
        if e.kind == "align:top":
            return (cx, cy - (hp - hc) / 2.0)
        if e.kind == "align:bottom":
            return (cx, cy + (hp - hc) / 2.0)
        if e.kind == "abut":
            if abs(e.dx) >= abs(e.dy):
                sep = (wp + wc) / 2.0
                return (cx - (sep if e.dx >= 0 else -sep), cy)
            sep = (hp + hc) / 2.0
            return (cx, cy - (sep if e.dy >= 0 else -sep))
        if e.kind in ("align:x", "align:y"):
            return (cx, cy)
        if e.kind == "symmetric":
            mid = cx - (e.dx / 2.0)
            return (2.0 * mid - cx, cy)
        if e.kind == "order:left_of":
            return (cx - (wp + wc) / 2.0, cy)
        if e.kind == "order:right_of":
            return (cx + (wp + wc) / 2.0, cy)
        if e.kind == "order:above":
            return (cx, cy - (hp + hc) / 2.0)
        if e.kind == "order:below":
            return (cx, cy + (hp + hc) / 2.0)
        return (cx, cy)

    progress = True
    while progress:
        progress = False
        for e in rel:
            if e.parent in pos and e.child not in pos:
                px, py = pos[e.parent]
                pos[e.child] = child_from_parent(e, px, py)
                progress = True
            elif e.child in pos and e.parent not in pos:
                cx, cy = pos[e.child]
                pos[e.parent] = parent_from_child(e, cx, cy)
                progress = True
    for n in comp:
        if n not in pos:
            pos[n] = (float(origin[0]), float(origin[1]))
    return pos


def solve_discovered(
    graph: ConstraintGraph,
    intent: Any,
    sizes: Optional[Dict[str, _XY]] = None,
) -> Tuple[Dict[str, _XY], leakfree.LeakfreeReport]:
    """Solve from STRUCTURAL relations ONLY, handling base + edge-align kinds.

    Like :func:`leakfree.solve_leakfree` but the DISCOVER structural-kind set includes the
    edge-align kinds, so an ``align:left`` / ``align:right`` / ``align:top`` /
    ``align:bottom`` edge counts as CONNECTING for component-finding and is realized from
    SIZES. HARD GUARD FIRST: :func:`assert_no_coordinates_discovered` raises if any
    ``offset`` edge sneaked in.

    Method mirrors ``solve_leakfree``: find connected components under the DISCOVER
    structural edges (union-find), solve each component by propagation, pack components
    left-to-right into default slots, and FLAG as floating every device NOT in the root's
    component. ``intent`` is accepted for signature symmetry with the rest of the Phase-C
    surface; ``sizes`` defaults to the graph's reference sizes.

    Returns ``(positions, LeakfreeReport)``; ``n_floating`` is the honest incompleteness.
    """
    assert_no_coordinates_discovered(graph)

    if sizes is None:
        sizes = graph.sizes
    sizes = sizes or {}

    names = list(graph.names)
    if not names:
        return ({}, leakfree.LeakfreeReport(n_devices=0, n_components=0, n_floating=0))

    struct_edges = [e for e in graph.edges if _is_discover_structural(e)]
    comps = leakfree._components(names, struct_edges)

    root = graph.root
    root_comp_idx = 0
    for i, c in enumerate(comps):
        if root in c:
            root_comp_idx = i
            break

    positions: Dict[str, _XY] = {}
    slot_x = 0.0
    for comp in comps:
        comp_pos = _solve_component_discovered(
            comp, struct_edges, sizes, root, (0.0, 0.0)
        )
        xs = [p[0] for p in comp_pos.values()]
        widths = [sizes.get(n, (0.0, 0.0))[0] for n in comp]
        min_x = min(xs) if xs else 0.0
        comp_w = (max(xs) - min(xs)) if xs else 0.0
        half_max = max(widths) / 2.0 if widths else 0.0
        shift = slot_x - min_x + half_max
        for n, (cx, cy) in comp_pos.items():
            positions[n] = (cx + shift, cy)
        slot_x += comp_w + 2.0 * half_max + leakfree._DEFAULT_SLOT_GAP_UM

    floating: List[str] = []
    for i, comp in enumerate(comps):
        if i == root_comp_idx:
            continue
        floating.extend(comp)
    floating.sort()

    report = leakfree.LeakfreeReport(
        n_devices=len(names),
        n_components=len(comps),
        n_floating=len(floating),
        floating=floating,
        components=comps,
        leakage_in=0,
    )
    return (positions, report)


# --------------------------------------------------------------------------- #
# 4. converge: greedily add relations that reduce the floating count          #
# --------------------------------------------------------------------------- #


@dataclass
class ConvergeReport:
    """The convergence result: how far discovery drove the floating count down.

    * ``floating_start`` -- floating count of the geometry-only (+ power) base graph.
    * ``floating_end``   -- floating count after all reducing relations were added.
    * ``trajectory``     -- the floating count after each accepted relation (start first).
    * ``floating_final`` -- the sorted names still floating at the end (the residual).
    * ``n_relations_added`` -- how many discovered relations were accepted.
    * ``leakage``        -- ``offset``-edge count of the final graph (MUST be 0).
    """

    floating_start: int
    floating_end: int
    trajectory: List[int] = field(default_factory=list)
    floating_final: List[str] = field(default_factory=list)
    n_relations_added: int = 0
    leakage: int = 0


def _ranked_edges(rels: List[DetectedRel]) -> List[RelEdge]:
    """Detected relations -> RelEdges in deterministic convergence priority order.

    Priority: abut (both axes) > align edge > align center > order, ties by (child,
    parent, kind). The RelEdge carries the structural kind; abut keeps its axis via the
    dx/dy sign set in :func:`to_edges`.
    """
    paired = []
    for r in rels:
        prio = _KIND_PRIORITY.get(r.kind, 9)
        paired.append((prio, r.child, r.parent, r.kind, r))
    paired.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return to_edges([t[4] for t in paired])


def converge(
    intent: Any,
    *,
    base_graph: Optional[ConstraintGraph] = None,
) -> ConvergeReport:
    """Greedily add discovered relations that REDUCE the floating count.

    Start graph: the leak-stripped constraint graph augmented with the validated power
    rails (``power_intent.augment_graph`` over ``base_graph`` or a freshly built one).
    Then, in deterministic priority order (abut > align edge > align center > order),
    try each discovered relation: re-solve with :func:`solve_discovered` and KEEP the
    edge iff it STRICTLY reduces the floating count (which is exactly the event of a new
    device joining the root component without breaking validity). Edges that do not help
    are discarded, so the final graph carries only relations that earned their place.

    Records the floating-count ``trajectory`` (start first, then after each accepted
    edge) and the final floating set. Asserts ``leakage == 0`` on the result.
    """
    if base_graph is None:
        base_graph = build_constraint_graph(intent)

    graph = power_intent.augment_graph(base_graph, intent)
    sizes = _sizes_from_intent(intent)

    _p0, rep0 = solve_discovered(graph, intent, sizes)
    floating_start = rep0.n_floating
    trajectory = [floating_start]

    edges = list(graph.edges)
    cur_floating = floating_start
    added = 0

    existing = {(e.kind, e.child, e.parent) for e in edges}
    rels = detect_relations(intent)
    for cand in _ranked_edges(rels):
        if cur_floating == 0:
            break
        key = (cand.kind, cand.child, cand.parent)
        if key in existing:
            continue
        trial = ConstraintGraph(
            root=graph.root,
            edges=edges + [cand],
            sizes=dict(graph.sizes),
            axis_x=graph.axis_x,
            names=list(graph.names),
        )
        _p, rep = solve_discovered(trial, intent, sizes)
        if rep.n_floating < cur_floating:
            edges.append(cand)
            existing.add(key)
            cur_floating = rep.n_floating
            added += 1
            trajectory.append(cur_floating)

    final_graph = ConstraintGraph(
        root=graph.root,
        edges=edges,
        sizes=dict(graph.sizes),
        axis_x=graph.axis_x,
        names=list(graph.names),
    )
    _pf, repf = solve_discovered(final_graph, intent, sizes)
    leak = leakfree.leakage_count(final_graph)

    return ConvergeReport(
        floating_start=floating_start,
        floating_end=repf.n_floating,
        trajectory=trajectory,
        floating_final=list(repf.floating),
        n_relations_added=added,
        leakage=leak,
    )


# --------------------------------------------------------------------------- #
# 5. convergence_summary: converge + classify, report the floating-vs-dontcare #
# --------------------------------------------------------------------------- #


def convergence_summary(intent: Any) -> str:
    """Run convergence + perturbation classification and report the honest outcome.

    Reports ``floating_start -> floating_end``, the trajectory, the final floating set,
    the classifier's DON'T-CARE set, and -- the ideal convergence criterion -- whether
    ``floating_final`` is a SUBSET of the don't-care set (every still-floating device is
    genuine design freedom, not a missing primitive). If not a subset, the residual
    floating devices that are NOT don't-cares are named as the honest gap (candidate
    still-missing primitives Phase C has yet to discover).
    """
    from clay.layout.extract import classify

    rep = converge(intent)

    positions = classify.reference_positions(intent)
    sizes = classify.device_sizes(intent)
    results = classify.classify_all(positions, sizes, intent)
    dont_care = sorted(
        n for n, r in results.items() if r.label == "dont_care"
    )

    floating_final = sorted(rep.floating_final)
    dc_set = set(dont_care)
    not_dont_care = sorted(n for n in floating_final if n not in dc_set)
    subset = not not_dont_care

    lines = [
        f"convergence: floating {rep.floating_start} -> {rep.floating_end} "
        f"({rep.n_relations_added} relation(s) added, leakage={rep.leakage})",
        f"  trajectory       : {rep.trajectory}",
        f"  floating_final   : {floating_final}",
        f"  dont_care (Phase B): {dont_care}",
    ]
    if subset:
        lines.append(
            "  CONVERGED: floating_final is a SUBSET of the don't-care set -- every "
            "still-floating device is genuine design freedom (no missing primitive)."
        )
    else:
        lines.append(
            "  GAP: floating_final is NOT a subset of the don't-care set; still-floating "
            f"and NOT a don't-care: {not_dont_care} (candidate still-missing primitives)."
        )
    return "\n".join(lines)


__all__ = [
    "DetectedRel",
    "ConvergeReport",
    "detect_relations",
    "to_edges",
    "solve_discovered",
    "assert_no_coordinates_discovered",
    "converge",
    "convergence_summary",
]
