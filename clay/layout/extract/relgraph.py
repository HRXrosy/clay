"""RELATIVE-GRAPH degradation of a layout's ABSOLUTE coordinates.

STRUCTURE-SOURCE CONTRACT (leakage discipline)
==============================================
Everything this module derives (band membership / within-band order / cross-band
X / constraint edges) comes from the bboxes of the INTENT VIEW it is fed. Fed a
``correspond_to_gds``-bboxed intent, the structure is REFERENCE-derived -- that
is the MIGRATION use (legitimately reference-consuming, as the migrate/realize
paths are named). A LEAKAGE-FREE placement must instead feed the DESIGN-sourced
sketch view (:func:`clay.layout.extract.topo_sketch.sketch_intent_view`), as
``external_pdk`` does; the obligation is pinned by
``tests/test_placement_leakage.py`` (reference-perturbation independence).

WHAT THIS IS (the corrected convergence foundation)
===================================================
A reference layout hands us ``intent.device_bboxes = {name -> (x0,y0,x1,y1) um}``:
N devices pinned at N absolute centroid positions. Those absolute coordinates carry
no design intent on their own -- a layout translated 100um to the right is the SAME
layout. This module DEGRADES the absolute coordinates into a RELATIVE constraint
graph that is solved EXACTLY (constraint-graph compaction in the spirit of IPRAIL
retargeting), NOT re-derived with a heuristic placer.

The degradation:

    N absolute coordinates  ->  (N-1) exact local offsets  +  1 root

The (N-1) edges of a spanning tree pin every device relative to its parent; the
root's absolute position is the SINGLE residual -- and that residual is itself the
global-translation DON'T-CARE (where you put the root is the where-on-the-die choice,
not a design fact). So after degradation the absolute coordinates are GONE, yet the
layout reproduces 1:1 (max deviation 0 modulo the global translation).

WHY A SPANNING TREE OF *LOCAL* OFFSETS
======================================
We connect each device to its NEAREST already-connected neighbour (a minimum-spanning
-tree on centroid distance). That makes every offset LOCAL and small -- a child sits
a few um from its parent, not 90um from a far-corner root. Local offsets are what a
sibling abstraction pass can later recognise as ``abut`` (offset == half-w sum),
``align`` (one component == 0), ``symmetric`` (mirrored arms) or ``order`` (sign of a
component). A tree of global-from-root offsets would hide all of that structure.

THE EDGE ``kind`` FIELD
=======================
Every edge starts ``kind='offset'`` -- a raw exact centroid delta. A sibling pass
abstracts a subset of edges to ``'abut' / 'align' / 'symmetric' / 'order'`` (replacing
the exact delta with a structural relation that still solves to the same position
within the care mask). This module only produces the EXACT 'offset' foundation; the
abstraction lives elsewhere.

UNITS / DETERMINISM
===================
Centroids and offsets are in MICRONS (the shared ``(x0,y0,x1,y1) um`` contract). The
tree is built from SORTED candidate edges (distance, then name) so the same bboxes
yield a byte-identical graph. Ties in distance break by name -> deterministic.

THE EXACT ORACLE (``positions_equal``)
======================================
``positions_equal`` is the equivalence test the abstraction search will use to accept
or reject an abstracted edge: it quotients out the global translation (subtract each
set's centroid) and optionally a global mirror, then compares centroid-by-centroid
against a per-coordinate care mask. With ``care_mask_um=0`` it is the EXACT oracle
that proves the L0->L1 degradation is loss-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_Bbox = Tuple[float, float, float, float]
_XY = Tuple[float, float]

# Floating-point noise floor for the EXACT oracle. Recentering N coordinates of
# magnitude ~1e2 um subtracts a ~1e2 mean from each, so the result carries ~1e-13 um
# of unavoidable double-rounding. "Exact" (care_mask_um == 0) therefore means "equal
# to the float noise floor", not "literally bit-identical" -- the residual is rounding
# in the quotient, not a real degradation loss. 1e-9 um = 1e-15 m is far below any
# layout grid (the finest is ~1e-4 um), so this never masks a genuine deviation.
_EXACT_EPS_UM: float = 1e-9


def _norm_bbox(bb: Any) -> _Bbox:
    x0, y0, x1, y1 = (float(v) for v in bb)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _center(bb: _Bbox) -> _XY:
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def _wh(bb: _Bbox) -> _XY:
    return (bb[2] - bb[0], bb[3] - bb[1])


# --------------------------------------------------------------------------- #
# data structures                                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RelEdge:
    """One spanning-tree edge: ``child`` is pinned relative to ``parent``.

    ``dx, dy`` are the EXACT centroid offset (um) of the child from the parent:
    ``centroid(child) = centroid(parent) + (dx, dy)``. ``kind`` starts ``'offset'``
    (a raw exact delta); a sibling abstraction pass may later replace it with
    ``'abut' / 'align' / 'symmetric' / 'order'`` (a structural relation that still
    solves to the same position within the care mask).
    """

    child: str
    parent: str
    dx: float
    dy: float
    kind: str = "offset"


@dataclass
class RelGraph:
    """A layout expressed as a spanning tree of EXACT relative offsets.

    ``root`` is the one device whose absolute position is the single residual (the
    global-translation don't-care). ``edges`` are the (N-1) parent->child offsets
    that pin every other device. ``sizes`` carries each device's bbox footprint
    ``(w, h)`` um (the abstraction pass needs it to recognise abut/align). ``order``
    is the deterministic tree-build order (root first), handy for traversal/printing.
    """

    root: str
    edges: List[RelEdge] = field(default_factory=list)
    sizes: Dict[str, _XY] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        """All device names this graph determines (root + every child)."""
        out = [self.root]
        out.extend(e.child for e in self.edges)
        return out

    def children_of(self) -> Dict[str, List[RelEdge]]:
        """``{parent -> [edges]}`` adjacency for traversal."""
        adj: Dict[str, List[RelEdge]] = {}
        for e in self.edges:
            adj.setdefault(e.parent, []).append(e)
        return adj


# --------------------------------------------------------------------------- #
# build: absolute centroids -> spanning tree of LOCAL exact offsets           #
# --------------------------------------------------------------------------- #


def _bboxes_of(intent: Any) -> Dict[str, _Bbox]:
    raw = getattr(intent, "device_bboxes", None)
    if raw is None and isinstance(intent, dict):
        raw = intent.get("device_bboxes")
    raw = raw or {}
    return {str(k): _norm_bbox(v) for k, v in raw.items()}


def _choose_root(centroids: Dict[str, _XY]) -> str:
    """The device nearest the centroid-of-centroids.

    A central root keeps offsets short on average (the tree grows outward from the
    middle, not from a corner). Ties break by name for determinism.
    """
    names = sorted(centroids)
    mx = sum(centroids[n][0] for n in names) / len(names)
    my = sum(centroids[n][1] for n in names) / len(names)

    def d2_to_mid(n: str) -> Tuple[float, str]:
        cx, cy = centroids[n]
        return ((cx - mx) ** 2 + (cy - my) ** 2, n)

    return min(names, key=d2_to_mid)


def build_rel_graph(intent: Any) -> RelGraph:
    """Degrade absolute centroids into a spanning tree of EXACT LOCAL offsets.

    Algorithm (a Prim minimum-spanning-tree on centroid Euclidean distance):

    1. Compute each device's centroid + footprint from ``intent.device_bboxes``.
    2. Choose the root = the device nearest the centroid-of-centroids.
    3. Grow the tree: repeatedly attach the UNconnected device whose centroid is
       nearest ANY already-connected device, recording the EXACT centroid offset
       ``(dx, dy)`` from that nearest connected neighbour as a ``RelEdge`` (kind
       'offset'). Ties (distance, then child name, then parent name) break
       deterministically.

    The result has exactly ``N-1`` edges spanning all ``N`` devices from one root.
    Because each child attaches to its NEAREST connected neighbour, every offset is
    LOCAL (small) -- the property the later abstraction pass relies on. The root's
    absolute position is never stored: it is the single global-translation residual.

    Empty / single-device inputs return a graph with the appropriate root and no
    edges (a single device is already fully determined by the root residual).
    """
    bboxes = _bboxes_of(intent)
    if not bboxes:
        return RelGraph(root="", edges=[], sizes={}, order=[])

    centroids = {n: _center(bb) for n, bb in bboxes.items()}
    sizes = {n: _wh(bb) for n, bb in bboxes.items()}

    root = _choose_root(centroids)
    connected: List[str] = [root]
    connected_set = {root}
    edges: List[RelEdge] = []
    remaining = set(centroids) - connected_set

    def dist2(a: str, b: str) -> float:
        ax, ay = centroids[a]
        bx, by = centroids[b]
        return (ax - bx) ** 2 + (ay - by) ** 2

    while remaining:
        # find the (child, parent) pair with the minimum centroid distance where
        # child is unconnected and parent is connected. Deterministic tie-break:
        # (distance, child name, parent name).
        best: Optional[Tuple[float, str, str]] = None
        for child in sorted(remaining):
            for parent in connected:  # connected is grown in deterministic order
                d = dist2(child, parent)
                cand = (d, child, parent)
                if best is None or cand < best:
                    best = cand
        assert best is not None
        _d, child, parent = best
        cx, cy = centroids[child]
        px, py = centroids[parent]
        edges.append(RelEdge(child=child, parent=parent, dx=cx - px, dy=cy - py))
        connected.append(child)
        connected_set.add(child)
        remaining.discard(child)

    order = list(connected)
    return RelGraph(root=root, edges=edges, sizes=sizes, order=order)


# --------------------------------------------------------------------------- #
# solve: spanning tree + root position -> every device's centroid             #
# --------------------------------------------------------------------------- #


def solve_rel_graph(
    relgraph: RelGraph, root_xy: _XY = (0.0, 0.0)
) -> Dict[str, _XY]:
    """Accumulate EXACT offsets from the root -> every device's centroid (um).

    Place the root at ``root_xy`` (the global-translation choice), then walk the tree
    from the root adding each edge's exact ``(dx, dy)`` to the parent's already-solved
    centroid. Because the tree spans all devices from one root, every centroid is
    determined. The returned positions reproduce the input centroids EXACTLY up to the
    constant translation ``root_xy - centroid(root)``.

    Traversal is iterative (no recursion-depth limit) and processes children in
    deterministic edge order.
    """
    if not relgraph.root and not relgraph.edges:
        return {}

    adj = relgraph.children_of()
    pos: Dict[str, _XY] = {relgraph.root: (float(root_xy[0]), float(root_xy[1]))}

    stack = [relgraph.root]
    while stack:
        parent = stack.pop()
        px, py = pos[parent]
        for e in adj.get(parent, []):
            pos[e.child] = (px + e.dx, py + e.dy)
            stack.append(e.child)
    return pos


# --------------------------------------------------------------------------- #
# the EXACT equivalence oracle (quotient global translation [+ mirror])        #
# --------------------------------------------------------------------------- #


def _centroid_of(pos: Dict[str, _XY]) -> _XY:
    n = len(pos)
    if n == 0:
        return (0.0, 0.0)
    sx = sum(p[0] for p in pos.values())
    sy = sum(p[1] for p in pos.values())
    return (sx / n, sy / n)


def _recenter(pos: Dict[str, _XY]) -> Dict[str, _XY]:
    cx, cy = _centroid_of(pos)
    return {n: (p[0] - cx, p[1] - cy) for n, p in pos.items()}


def positions_equal(
    pos_a: Dict[str, _XY],
    pos_b: Dict[str, _XY],
    care_mask_um: float = 0.0,
    allow_mirror: bool = False,
) -> Tuple[bool, float]:
    """Equivalence oracle quotienting global translation (and optional mirror).

    Both position sets are RECENTERED (each set's centroid subtracted) so a pure
    global translation between them is quotiented away -- the don't-care. Then the two
    sets are compared name-by-name; the result is ``(equal, max_dev)`` where
    ``max_dev`` is the largest per-coordinate absolute deviation (um) over the shared
    names, and ``equal`` is ``max_dev <= care_mask_um``.

    With ``care_mask_um == 0`` this is the EXACT oracle: positions must coincide to the
    floating-point noise floor (``_EXACT_EPS_UM``, the unavoidable rounding of the
    recentering quotient -- 1e-9 um, far below any layout grid). It proves the
    degradation is loss-free. A positive mask is the tolerance the abstraction search
    grants an abstracted edge.

    ``allow_mirror``: if set, we ALSO try a global x-mirror (negate x after
    recentering) and a global y-mirror, and report the SMALLEST max deviation over
    {identity, x-mirror, y-mirror}. This quotients a whole-layout flip (the other
    placement don't-care for a symmetric cell). Default off (translation-only).

    Names present in only one set are ignored for the deviation but make the sets
    non-equal only if the shared-name comparison itself exceeds the mask is false; we
    require the name SETS to match for ``equal`` to be True (a missing device is a real
    difference). ``max_dev`` is computed over the intersection.
    """
    names = sorted(set(pos_a) & set(pos_b))
    same_names = set(pos_a) == set(pos_b)
    if not names:
        # nothing to compare; equal iff both empty
        return (same_names and not pos_a, 0.0)

    a = _recenter({n: pos_a[n] for n in names})
    b = _recenter({n: pos_b[n] for n in names})

    def max_dev_under(transform) -> float:
        m = 0.0
        for n in names:
            ax, ay = transform(a[n])
            bx, by = b[n]
            m = max(m, abs(ax - bx), abs(ay - by))
        return m

    candidates = [lambda p: p]  # identity
    if allow_mirror:
        candidates.append(lambda p: (-p[0], p[1]))  # x-mirror
        candidates.append(lambda p: (p[0], -p[1]))  # y-mirror

    max_dev = min(max_dev_under(t) for t in candidates)
    # the effective threshold floors the requested mask at the float noise floor so a
    # care_mask_um of exactly 0 still accepts a deviation that is pure quotient
    # rounding (see _EXACT_EPS_UM) -- this is the honest "exact" semantics.
    threshold = max(care_mask_um, _EXACT_EPS_UM)
    equal = same_names and (max_dev <= threshold)
    return (equal, max_dev)


# --------------------------------------------------------------------------- #
# CONSTRAINT GRAPH (not a spanning tree).                                       #
#                                                                              #
# WHY A CONSTRAINT GRAPH, NOT AN MST                                            #
# =================================                                            #
# ``build_rel_graph`` builds a minimum spanning TREE: exactly N-1 edges. A tree #
# is loss-free for ABSOLUTE reproduction (it pins every centroid), but it DROPS #
# every structural adjacency that is not on the tree. Concrete failure on the   #
# real OTA: the two MIM caps CC0 / CR1 are EXACTLY edge-adjacent (CC0 right edge #
# x=61.015 == CR1 left edge x=61.015, same y-extent, same 32.8x32.8 size) -- a  #
# textbook @abut + @align:y + @equal_size. But the MST has NO direct CC0<->CR1  #
# edge; it routes them through a 6-edge offset chain CC0->RR5->...->RR0->CR1.    #
# The cap-cap abut is therefore never captured: the caps' relative position      #
# accumulates through six non-retargetable external_pdk offsets, so when the devices grow #
# to gf180 sizes the two caps OVERLAP each other (and crash into the MOS row).   #
#                                                                              #
# A CONSTRAINT graph captures ALL structural adjacencies (multiple edges per     #
# device -- over-determined). The reference satisfies all of them consistently   #
# (it is a real layout), so the over-determined system is solvable; the residual #
# numeric content shrinks to the MINIMAL set of offset bridges needed to pin the #
# clusters that structure alone leaves under-determined -- the genuine            #
# incompressible cross-cluster residual.                                         #
# --------------------------------------------------------------------------- #

# Geometry tolerance (um) for the structural detectors: a gap / shared-axis delta
# within this of zero is treated as structural. Matches abstract_ops._EPS_UM.
_CG_EPS_UM: float = 0.05


@dataclass
class ConstraintGraph:
    """A layout expressed as an OVER-DETERMINED set of structural constraints.

    Unlike :class:`RelGraph` (a spanning tree of exactly N-1 offset edges), this
    carries an edge for EVERY structural adjacency between devices (abut / align /
    symmetric) PLUS the minimal offset bridges needed to fully determine every
    position. There are therefore MORE than N-1 edges in general -- the system is
    over-determined but CONSISTENT (the reference layout satisfies all of them).

    ``edges`` are :class:`RelEdge` (reused): each carries the EXACT reference
    ``dx, dy`` (so any edge can fall back to exact offset and the system is provably
    consistent at reference sizes) plus a structural ``kind``
    ('abut' / 'align:x' / 'align:y' / 'symmetric' / 'offset'). ``root`` is the
    translation don't-care pin. ``sizes`` is each device's reference ``(w, h)`` um.
    ``axis_x`` is the vertical symmetry axis x (um) if the intent declared one.
    """

    root: str
    edges: List[RelEdge] = field(default_factory=list)
    sizes: Dict[str, _XY] = field(default_factory=dict)
    axis_x: Optional[float] = None
    names: List[str] = field(default_factory=list)

    def structural_edges(self) -> List[RelEdge]:
        return [e for e in self.edges if e.kind != "offset"]

    def offset_edges(self) -> List[RelEdge]:
        return [e for e in self.edges if e.kind == "offset"]


def _pairs_from_intent(intent: Any, attr: str) -> List[Tuple[str, str]]:
    """Collect declared unordered pairs from a list-of-pairs intent attribute."""
    out: List[Tuple[str, str]] = []
    for ab in getattr(intent, attr, ()) or ():
        if len(ab) == 2:
            out.append((str(ab[0]), str(ab[1])))
    return out


def _declared_pair_set(intent: Any) -> set:
    """Frozenset-keyed set of declared matched/mirror pairs (for symmetric edges)."""
    s = set()
    for attr in ("matched_pairs", "mirror_pairs"):
        for a, b in _pairs_from_intent(intent, attr):
            s.add(frozenset((a, b)))
    return s


def _detect_abut(bb_p: _Bbox, bb_c: _Bbox, eps: float) -> Optional[str]:
    """Return 'x' if the two bboxes are edge-adjacent (touching) along x (one's right
    face == the other's left face, with overlapping y-extent), 'y' if along y, else
    None. Touching means the facing faces coincide within ``eps`` AND the boxes
    overlap on the perpendicular axis (a real shared face, not a corner kiss)."""
    px0, py0, px1, py1 = bb_p
    cx0, cy0, cx1, cy1 = bb_c
    y_overlap = min(py1, cy1) - max(py0, cy0)
    x_overlap = min(px1, cx1) - max(px0, cx0)
    # horizontal abut: a right face meets a left face, with shared y span.
    if y_overlap > eps and (
        abs(px1 - cx0) <= eps or abs(cx1 - px0) <= eps
    ):
        return "x"
    # vertical abut: a top face meets a bottom face, with shared x span.
    if x_overlap > eps and (
        abs(py1 - cy0) <= eps or abs(cy1 - py0) <= eps
    ):
        return "y"
    return None


def _detect_align(c_p: _XY, c_c: _XY, eps: float) -> Optional[str]:
    """Return 'x' if the two centroids share an x coordinate (vertical alignment /
    same column), 'y' if they share a y coordinate (horizontal alignment / same row),
    else None. If both, the closer-to-zero axis wins (an exact stack)."""
    dx = c_c[0] - c_p[0]
    dy = c_c[1] - c_p[1]
    ax = abs(dx) <= eps
    ay = abs(dy) <= eps
    if ax and ay:
        return None  # coincident centroids -- not an alignment relation
    if ax:
        return "x"
    if ay:
        return "y"
    return None


def build_constraint_graph(intent: Any, eps: float = _CG_EPS_UM) -> ConstraintGraph:
    """Degrade absolute centroids into an OVER-DETERMINED CONSTRAINT GRAPH that
    captures ALL structural adjacencies (NOT a spanning tree).

    Algorithm:

    1. STRUCTURAL edges -- for EVERY unordered device pair, run the structural
       detectors on the reference geometry:
         * ABUT      : bboxes are edge-adjacent (a shared face within ``eps``).
                       kind 'abut'; the touching axis is encoded in dominant dx/dy.
         * ALIGN     : centroids share an x (-> 'align:x') or y (-> 'align:y').
         * SYMMETRIC : the pair is DECLARED matched/mirror AND mirrors about the
                       cell's vertical axis (same y, opposite x). kind 'symmetric'.
       A pair may match several detectors -- every matching structural relation is
       added as its own edge (the graph is over-determined ON PURPOSE; the reference
       satisfies them all). Each edge also stores the EXACT reference ``dx, dy`` so
       the system is provably consistent at reference sizes.

    2. CONNECTIVITY / DETERMINACY -- structural edges alone may leave whole clusters
       disconnected, or pin a cluster on only one axis. Using union-find over the
       FULLY-determining edges (abut + symmetric pin both axes; align pins one), add
       the MINIMAL set of exact OFFSET bridges (nearest-neighbour, deterministic) to
       (a) connect disconnected components and (b) pin any residual free axis. These
       minimal offsets are the GENUINE incompressible cross-cluster residual.

    Returns a :class:`ConstraintGraph`. Empty / single-device inputs return a graph
    with the appropriate root and no edges.
    """
    bboxes = _bboxes_of(intent)
    if not bboxes:
        return ConstraintGraph(root="", edges=[], sizes={}, names=[])

    names = sorted(bboxes)
    centroids = {n: _center(bb) for n, bb in bboxes.items()}
    sizes = {n: _wh(bb) for n, bb in bboxes.items()}
    root = _choose_root(centroids)

    axis_x = getattr(intent, "symmetry_axis", None)
    axis_x = float(axis_x) if isinstance(axis_x, (int, float)) else None
    declared = _declared_pair_set(intent)

    edges: List[RelEdge] = []
    # Track, per device, which axes are PINNED to the connected structure so the
    # determinacy pass adds only the bridges that are actually missing. We use a
    # union-find PER AXIS: x-pin connectivity and y-pin connectivity are separate.
    parent_x: Dict[str, str] = {n: n for n in names}
    parent_y: Dict[str, str] = {n: n for n in names}

    def _find(pp: Dict[str, str], n: str) -> str:
        while pp[n] != n:
            pp[n] = pp[pp[n]]
            n = pp[n]
        return n

    def _union(pp: Dict[str, str], a: str, b: str) -> None:
        ra, rb = _find(pp, a), _find(pp, b)
        if ra != rb:
            # deterministic: smaller name is the canonical root
            if ra < rb:
                pp[rb] = ra
            else:
                pp[ra] = rb

    def _add(child: str, par: str, kind: str) -> None:
        cx, cy = centroids[child]
        px, py = centroids[par]
        edges.append(RelEdge(child=child, parent=par, dx=cx - px, dy=cy - py, kind=kind))
        # update per-axis pin connectivity by what this edge determines
        if kind in ("abut", "symmetric", "offset"):
            _union(parent_x, child, par)
            _union(parent_y, child, par)
        elif kind == "align:x":
            _union(parent_x, child, par)  # shared x pins the x axis
        elif kind == "align:y":
            _union(parent_y, child, par)  # shared y pins the y axis

    # 1. STRUCTURAL edges over every ordered-by-name pair (parent = smaller name).
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            par, child = names[i], names[j]  # par < child by sort
            bb_p, bb_c = bboxes[par], bboxes[child]
            c_p, c_c = centroids[par], centroids[child]

            # symmetric (declared mirror about the vertical axis)
            if frozenset((par, child)) in declared:
                dy = c_c[1] - c_p[1]
                dx = c_c[0] - c_p[0]
                if abs(dy) <= eps and abs(dx) > eps:
                    _add(child, par, "symmetric")

            # abut (shared face)
            ax = _detect_abut(bb_p, bb_c, eps)
            if ax is not None:
                _add(child, par, "abut")

            # align (shared centroid axis)
            al = _detect_align(c_p, c_c, eps)
            if al == "x":
                _add(child, par, "align:x")
            elif al == "y":
                _add(child, par, "align:y")

    # 2. DETERMINACY: add minimal exact-offset bridges. A device's position is fully
    # pinned iff it shares an x-component AND a y-component with the root's structure.
    # We bridge nearest-neighbour deterministically until both per-axis forests are a
    # single tree rooted at the root's component.
    def _need_bridge() -> Optional[Tuple[str, str, str]]:
        """Return (child, parent, why) for the nearest pair that would merge two
        distinct per-axis components, or None when both axes are fully connected."""
        root_rx = _find(parent_x, root)
        root_ry = _find(parent_y, root)
        x_done = all(_find(parent_x, n) == root_rx for n in names)
        y_done = all(_find(parent_y, n) == root_ry for n in names)
        if x_done and y_done:
            return None
        best: Optional[Tuple[float, str, str]] = None
        for a in names:
            for b in names:
                if a >= b:
                    continue
                # only useful if this offset bridge merges some axis component
                merges_x = _find(parent_x, a) != _find(parent_x, b)
                merges_y = _find(parent_y, a) != _find(parent_y, b)
                if not (merges_x or merges_y):
                    continue
                ca, cb = centroids[a], centroids[b]
                d = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
                cand = (d, a, b)
                if best is None or cand < best:
                    best = cand
        if best is None:
            return None
        _d, a, b = best
        return (b, a, "bridge")  # parent = smaller name (a < b)

    guard = 0
    while True:
        nb = _need_bridge()
        if nb is None:
            break
        child, par, _why = nb
        _add(child, par, "offset")
        guard += 1
        if guard > len(names) + 1:
            break  # safety: never loop forever

    return ConstraintGraph(
        root=root, edges=edges, sizes=sizes, axis_x=axis_x, names=names
    )


# --------------------------------------------------------------------------- #
# solve the over-determined-but-consistent constraint graph                    #
# --------------------------------------------------------------------------- #


def solve_constraint_graph(
    graph: ConstraintGraph,
    sizes: Optional[Dict[str, _XY]] = None,
    root_xy: _XY = (0.0, 0.0),
    *,
    grow_align_to_sizes: bool = False,
) -> Tuple[Dict[str, _XY], float]:
    """Solve the over-determined constraint system -> ``({name -> (cx, cy)}, max_resid)``.

    ``sizes`` (name -> (w, h) um) supplies the device footprints used to realise the
    SEPARATION of structural edges:
      * abut       -> facing bboxes touch: centre-to-centre on the dominant axis =
                      (size_parent + size_child)/2 (sign from the reference delta);
                      the off-axis component takes the reference value.
      * align:x    -> child shares parent's x; y from the reference dy.
        align:y    -> child shares parent's y; x from the reference dx.
      * symmetric  -> child = mirror of parent about the cell's vertical axis; y kept.
      * offset     -> exact reference dx, dy.
    Defaults to the graph's REFERENCE sizes -- the consistency case (MUST reproduce
    the reference EXACTLY, max_resid ~ 0).

    Because the system is over-determined, an edge may try to place an already-placed
    node. We solve by iterative propagation (a node is placed by the FIRST edge that
    reaches it from the placed set, deterministic edge order), then we measure the
    MAX RESIDUAL: for every edge, the disagreement between where it WOULD place its
    child and where the child actually sits. At reference sizes the reference satisfies
    every edge, so the residual is ~0; we report it honestly rather than hide it.
    """
    if not graph.root and not graph.edges:
        return ({}, 0.0)
    if sizes is None:
        sizes = graph.sizes
    sizes = sizes or {}

    # The symmetry axis is given in the REFERENCE frame; after pinning the root at
    # root_xy the whole layout translates by (root_xy - centroid(root)). We do not
    # know centroid(root) in the solved frame yet, so we resolve symmetric edges by
    # mirroring about the PAIR's own mid-x (parent_x + dx/2 from the reference), which
    # is translation-covariant and needs no global axis -- equivalent at ref sizes.

    pos: Dict[str, _XY] = {graph.root: (float(root_xy[0]), float(root_xy[1]))}

    def child_from_parent(e: RelEdge, px: float, py: float) -> _XY:
        """Where the CHILD goes given the PARENT is at (px, py)."""
        wp, hp = sizes.get(e.parent, (0.0, 0.0))
        wc, hc = sizes.get(e.child, (0.0, 0.0))
        if e.kind == "offset":
            return (px + e.dx, py + e.dy)
        if e.kind == "abut":
            if abs(e.dx) >= abs(e.dy):
                sep = (wp + wc) / 2.0
                cx = px + (sep if e.dx >= 0 else -sep)
                return (cx, py + e.dy)
            sep = (hp + hc) / 2.0
            cy = py + (sep if e.dy >= 0 else -sep)
            return (px + e.dx, cy)
        if e.kind == "align:x":
            # ``grow_align_to_sizes``: the MIGRATION semantics (opt-in). The
            # shared-x pair's free-axis gap is the reference dy GROWN to at least
            # face-to-face contact at the GIVEN sizes, closing the documented
            # align hole (reference gap kept verbatim -> grown devices overlap,
            # e.g. the gf180 MM18/MM19 4.35 um^2 the structural-overlap oracle
            # caught). DEFAULT OFF: the exact solve must reproduce the REFERENCE
            # verbatim, and a reference align pair may legitimately overlap
            # (interdigitated MM15/MM16) -- unconditional growth broke the
            # reference-reproduction oracles, so the growth is an explicit
            # migration semantics, never the default.
            if grow_align_to_sizes and e.dy != 0.0:
                sep = max(abs(e.dy), (hp + hc) / 2.0)
                return (px, py + (sep if e.dy >= 0 else -sep))
            return (px, py + e.dy)
        if e.kind == "align:y":
            if grow_align_to_sizes and e.dx != 0.0:
                sep = max(abs(e.dx), (wp + wc) / 2.0)
                return (px + (sep if e.dx >= 0 else -sep), py)
            return (px + e.dx, py)
        if e.kind == "symmetric":
            # mirror about the pair mid-x (parent_x + dx/2 in the reference frame ->
            # translation-covariant after pinning the parent). y preserved.
            mid = px + e.dx / 2.0
            return (2.0 * mid - px, py)
        return (px + e.dx, py + e.dy)

    def parent_from_child(e: RelEdge, cx: float, cy: float) -> _XY:
        """Where the PARENT goes given the CHILD is at (cx, cy) -- the inverse of
        ``child_from_parent`` (the constraint graph is UNDIRECTED for solving, so a
        node may be reached via the child end of an edge whose parent is the larger
        name)."""
        wp, hp = sizes.get(e.parent, (0.0, 0.0))
        wc, hc = sizes.get(e.child, (0.0, 0.0))
        if e.kind == "offset":
            return (cx - e.dx, cy - e.dy)
        if e.kind == "abut":
            if abs(e.dx) >= abs(e.dy):
                sep = (wp + wc) / 2.0
                px = cx - (sep if e.dx >= 0 else -sep)
                return (px, cy - e.dy)
            sep = (hp + hc) / 2.0
            py = cy - (sep if e.dy >= 0 else -sep)
            return (cx - e.dx, py)
        if e.kind == "align:x":
            return (cx, cy - e.dy)
        if e.kind == "align:y":
            return (cx - e.dx, cy)
        if e.kind == "symmetric":
            # child = 2*mid - parent, mid = parent + dx/2 -> child = parent + dx, so
            # parent = child - dx (x); y preserved.
            return (cx - e.dx, cy)
        return (cx - e.dx, cy - e.dy)

    edges = list(graph.edges)

    # The system is OVER-DETERMINED and, at NON-reference sizes, INCONSISTENT: when
    # devices grow non-uniformly to gf180, no placement honours every structural edge
    # AND every offset bridge at once. We resolve this by STRUCTURAL PRIORITY: build a
    # spanning tree that prefers structural edges (abut > symmetric > align > offset),
    # propagate along THAT tree (so every chosen structural separation is realised
    # exactly -> those pairs are overlap-free by construction), and treat the remaining
    # NON-tree edges as the residual (reported, not enforced). This keeps the cap-cap
    # abut -- and as many other structural adjacencies as the tree can carry -- as
    # honoured flush relations, and pushes the irreducible inconsistency onto the
    # offset bridges (the genuine cross-cluster residual), exactly where it belongs.
    #
    # Priority order for the spanning tree (lower = chosen first):
    _PRIO = {"abut": 0, "symmetric": 1, "align:x": 2, "align:y": 2, "offset": 3}

    # Kruskal-style: pick edges in (priority, child, parent) order that connect two
    # not-yet-connected nodes -> a maximum-structural spanning tree (a forest if the
    # graph is disconnected, but build_constraint_graph guarantees connectivity).
    uf: Dict[str, str] = {n: n for n in graph.names}

    def _find(n: str) -> str:
        while uf[n] != n:
            uf[n] = uf[uf[n]]
            n = uf[n]
        return n

    ranked = sorted(
        edges, key=lambda e: (_PRIO.get(e.kind, 3), e.child, e.parent)
    )
    tree_edges: List[RelEdge] = []
    for e in ranked:
        ra, rb = _find(e.child), _find(e.parent)
        if ra != rb:
            uf[ra] = rb
            tree_edges.append(e)

    # propagate along the spanning tree (undirected: place a node from whichever END is
    # already known). Deterministic -> deterministic positions.
    progress = True
    while progress:
        progress = False
        for e in tree_edges:
            if e.parent in pos and e.child not in pos:
                px, py = pos[e.parent]
                pos[e.child] = child_from_parent(e, px, py)
                progress = True
            elif e.child in pos and e.parent not in pos:
                cx, cy = pos[e.child]
                pos[e.parent] = parent_from_child(e, cx, cy)
                progress = True
    # any name never reached (disconnected component): pin at the root (last resort).
    for n in graph.names:
        if n not in pos:
            pos[n] = (float(root_xy[0]), float(root_xy[1]))

    # residual: the disagreement, over EVERY edge (tree + non-tree), between where the
    # edge wants the child and where the child sits. Tree edges are satisfied (~0); the
    # max over NON-tree edges is the irreducible over-determined residual -- reported
    # honestly, never hidden. At reference sizes every edge is consistent -> ~0.
    max_resid = 0.0
    for e in edges:
        if e.parent not in pos or e.child not in pos:
            continue
        px, py = pos[e.parent]
        wx, wy = child_from_parent(e, px, py)
        gx, gy = pos[e.child]
        max_resid = max(max_resid, abs(gx - wx), abs(gy - wy))

    return (pos, max_resid)


# --------------------------------------------------------------------------- #
# HIERARCHICAL / BAND layer.                                                    #
#                                                                              #
# WHY A BAND LEVEL ABOVE THE DEVICE CONSTRAINT GRAPH                            #
# =================================================                            #
# ``build_constraint_graph`` captures every WITHIN-cluster structural adjacency #
# (the cap-cap abut, every align/symmetric pair) and bridges the clusters with  #
# the MINIMAL set of exact OFFSET edges. Those cross-cluster offset bridges are  #
# the only residual -- and on the real OTA they are ALL the surviving overlaps   #
# (18 pairs / 1252 um2): the reference placed the CAP cluster a fixed external_pdk       #
# distance BELOW the PMOS row, an offset that does NOT retarget -- gf180 devices #
# are bigger and crash up into the row. The offset bridge encodes "caps sit      #
# 22.95um below the PMOS row centroids", a external_pdk magnitude, not a relation.        #
#                                                                              #
# The fix: a STRUCTURAL primitive one level up. The reference is TIERED -- a      #
# PMOS row on top, caps+resistors in the middle, an NMOS row on the bottom        #
# (verify by y-clustering the centroids: two big gaps, 18.21um and 22.95um,       #
# cleanly cut three bands out of within-band gaps that never exceed 6.26um).      #
# Adjacent bands have a STACKING relation: band B sits below band A, abutting or  #
# min-spaced. That relation RETARGETS -- realized at gf180 sizes the bands stack  #
# with gf180 min spacing, so no cross-band device can crash into the band above.  #
# The band-level edges REPLACE the device offset bridges that crossed between      #
# adjacent bands; the only offsets left are genuinely WITHIN-band or               #
# non-adjacent-cluster.                                                            #
# --------------------------------------------------------------------------- #

# Band detection: the gap (um) on the sorted y-centroid axis is a band boundary
# when it exceeds this MULTIPLE of the median within-run gap (a relative, not
# absolute, cut so it scales across PDKs). Floored by an absolute minimum so a
# degenerate all-equal-y run does not split on float noise.
_BAND_GAP_FACTOR: float = 3.0
_BAND_GAP_MIN_UM: float = 8.0


@dataclass
class Band:
    """One horizontal band (tier) of the layout.

    ``band_id`` is the deterministic index (0 = bottom-most, increasing upward).
    ``devices`` is the sorted list of device names in the band. ``y_lo`` / ``y_hi``
    are the band's centroid-y extent (um) in the reference frame (informational --
    the structural stacking uses solved band bboxes, not these reference values).
    """

    band_id: int
    devices: List[str]
    y_lo: float
    y_hi: float
    x_lo: float = 0.0
    x_hi: float = 0.0
    ref_cx: float = 0.0
    ref_cy: float = 0.0


@dataclass
class BandEdge:
    """A band-level STACKING edge: ``lower`` band sits directly below ``upper``.

    Meaning: ``upper``'s bottom face is above ``lower``'s top face, separated by a
    channel >= the min spacing for the kinds involved (abutting when min spacing is
    0). ``ref_gap`` is the reference centroid-to-centroid band gap (um) -- kept for
    the consistency check (at reference sizes the stack must reproduce it); at gf180
    sizes the solver uses gf180 min spacing instead, so the bands retarget.
    """

    lower: int
    upper: int
    ref_gap: float
    kind: str = "band_abut"


@dataclass
class HierGraph:
    """A layout expressed HIERARCHICALLY: per-band device constraint graphs PLUS
    band-level stacking edges between vertically-adjacent bands.

    ``bands`` is the ordered list of :class:`Band` (bottom-up). ``band_graphs`` maps
    band_id -> the :class:`ConstraintGraph` over THAT band's devices (built by
    reusing ``build_constraint_graph`` on the band sub-intent). ``band_edges`` are the
    :class:`BandEdge` stacking relations between adjacent bands -- the structural
    primitive that REPLACES the cross-band device offset bridges. ``sizes`` is each
    device's reference ``(w, h)`` um. ``root_band`` is the band placed first (the
    bottom band, the translation pin for the stack).
    """

    bands: List[Band] = field(default_factory=list)
    band_graphs: Dict[int, ConstraintGraph] = field(default_factory=dict)
    band_edges: List[BandEdge] = field(default_factory=list)
    sizes: Dict[str, _XY] = field(default_factory=dict)
    root_band: int = 0
    # P0 compaction (opt-in): the design-derived CompactionSpec carried by a
    # ``derive_design_sketch(compact=True)`` sketch view -- consumed by
    # ``solve_hierarchical_nonoverlap`` as a POST-stack outline press. None
    # (the default, and every non-sketch intent) = no press, byte-identical.
    compaction: Optional[Any] = None
    # mode='2d' embed-search self-declaration: one note per ACCEPTED
    # cross-row x-repack move (empty for mode='press'/no spec/no move) --
    # filled by ``solve_hierarchical_nonoverlap``.
    compaction_notes: Tuple[str, ...] = ()
    # P1 relax engine (opt-in): the intent view carried by a
    # ``derive_design_sketch(engine='relax')`` sketch view -- consumed by
    # ``solve_hierarchical_nonoverlap`` as a POST-press refinement stage
    # (:mod:`clay.layout.extract.place_relax`), which parks its honest
    # self-declaration on ``relax_report``. None (the default, and every
    # non-sketch intent) = no refinement, byte-identical.
    relax_intent: Optional[Any] = None
    relax_report: Optional[Any] = None


class _BandIntentView:
    """A lightweight intent VIEW restricted to one band's devices.

    ``build_constraint_graph`` reads ``device_bboxes`` + ``matched_pairs`` /
    ``mirror_pairs`` + ``symmetry_axis`` off the intent. This view filters all of
    those to the band's device set so the per-band constraint graph sees only the
    band's devices (and only the matched/mirror pairs fully inside the band)."""

    def __init__(self, intent: Any, devices: List[str]) -> None:
        dset = set(devices)
        all_bb = _bboxes_of(intent)
        self.device_bboxes = {n: all_bb[n] for n in devices if n in all_bb}

        def _filt(attr: str) -> List[Tuple[str, str]]:
            out: List[Tuple[str, str]] = []
            for ab in getattr(intent, attr, ()) or ():
                if len(ab) == 2 and str(ab[0]) in dset and str(ab[1]) in dset:
                    out.append((str(ab[0]), str(ab[1])))
            return out

        self.matched_pairs = _filt("matched_pairs")
        self.mirror_pairs = _filt("mirror_pairs")
        ax = getattr(intent, "symmetry_axis", None)
        self.symmetry_axis = ax


def cluster_bands(
    intent: Any,
    gap_factor: float = _BAND_GAP_FACTOR,
    gap_min_um: float = _BAND_GAP_MIN_UM,
) -> Dict[int, List[str]]:
    """Cluster devices into horizontal BANDS by centroid-y -> ``{band_id -> [names]}``.

    DETERMINISTIC gap-based clustering: sort the devices by centroid-y; a gap between
    consecutive y-centroids STARTS A NEW BAND when it exceeds ``max(gap_factor *
    median-within-band-gap, gap_min_um)`` -- a RELATIVE cut (so it scales across PDKs)
    floored by an absolute minimum (so a degenerate all-equal-y run does not split on
    float noise). ``band_id`` is assigned bottom-up (0 = smallest y / bottom band).

    On the real OTA this DETECTS ~3 macro bands (PMOS row on top, caps+resistors in a
    distinct middle band, NMOS row on the bottom) -- the two band-separating gaps
    (~18 and ~23 um) dwarf the within-band gaps (<= ~6 um). Empty input -> ``{}``;
    single device -> one band {0: [name]}.
    """
    bboxes = _bboxes_of(intent)
    if not bboxes:
        return {}
    # (centroid_y, name), sorted by y then name (deterministic on y ties).
    cys = sorted(((_center(bb)[1], n) for n, bb in bboxes.items()))
    if len(cys) == 1:
        return {0: [cys[0][1]]}

    gaps = [cys[i + 1][0] - cys[i][0] for i in range(len(cys) - 1)]
    # median of the consecutive gaps as the within-band scale.
    sg = sorted(gaps)
    m = len(sg)
    median_gap = sg[m // 2] if m % 2 == 1 else (sg[m // 2 - 1] + sg[m // 2]) / 2.0
    threshold = max(gap_factor * median_gap, gap_min_um)

    bands_dev: List[List[str]] = [[cys[0][1]]]
    for i in range(len(gaps)):
        if gaps[i] > threshold:
            bands_dev.append([])
        bands_dev[-1].append(cys[i + 1][1])

    return {bid: sorted(devs) for bid, devs in enumerate(bands_dev)}


def _band_bbox_ref(intent: Any, devices: List[str]) -> _Bbox:
    """Reference bounding box (x0,y0,x1,y1) um over a band's device bboxes."""
    bb = _bboxes_of(intent)
    xs0 = [bb[n][0] for n in devices if n in bb]
    ys0 = [bb[n][1] for n in devices if n in bb]
    xs1 = [bb[n][2] for n in devices if n in bb]
    ys1 = [bb[n][3] for n in devices if n in bb]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def build_hierarchical_graph(
    intent: Any,
    gap_factor: float = _BAND_GAP_FACTOR,
    gap_min_um: float = _BAND_GAP_MIN_UM,
    eps: float = _CG_EPS_UM,
) -> HierGraph:
    """Build the HIERARCHICAL graph: per-band device constraint graphs PLUS band-level
    stacking edges between vertically-adjacent bands.

    Steps:

    1. ``cluster_bands`` -> the macro bands (bottom-up).
    2. For EACH band, build the device-level CONSTRAINT GRAPH over ONLY that band's
       devices (reuse ``build_constraint_graph`` on a band-restricted intent view).
       Within-band structural adjacencies (abut/align/symmetric) + the minimal
       within-band offset bridges are captured exactly as before.
    3. BAND-LEVEL edges: for each pair of vertically-adjacent bands (band i below band
       i+1) add a :class:`BandEdge` ``band_abut`` meaning band i.top is below band
       i+1.bottom (stacked, min-spaced). This REPLACES the device offset bridges that
       previously crossed between those two bands -- the cross-cluster residual that
       did not retarget.

    Returns a :class:`HierGraph`. Empty / single-band inputs degrade gracefully (one
    band, no band edges).
    """
    bands_map = cluster_bands(intent, gap_factor, gap_min_um)
    sizes = {n: _wh(bb) for n, bb in _bboxes_of(intent).items()}

    bands: List[Band] = []
    band_graphs: Dict[int, ConstraintGraph] = {}
    all_bb = _bboxes_of(intent)
    for bid in sorted(bands_map):
        devs = bands_map[bid]
        bb = _band_bbox_ref(intent, devs)
        # reference mean centroid over the band's devices (the consistency anchor).
        cxs = [_center(all_bb[n])[0] for n in devs if n in all_bb]
        cys = [_center(all_bb[n])[1] for n in devs if n in all_bb]
        ref_cx = sum(cxs) / len(cxs) if cxs else 0.0
        ref_cy = sum(cys) / len(cys) if cys else 0.0
        bands.append(Band(
            band_id=bid, devices=devs, y_lo=bb[1], y_hi=bb[3],
            x_lo=bb[0], x_hi=bb[2], ref_cx=ref_cx, ref_cy=ref_cy,
        ))
        view = _BandIntentView(intent, devs)
        band_graphs[bid] = build_constraint_graph(view, eps=eps)

    band_edges: List[BandEdge] = []
    for i in range(len(bands) - 1):
        lo, up = bands[i], bands[i + 1]
        # reference mean-centroid-y band gap (consistency reproduction target).
        band_edges.append(
            BandEdge(lower=lo.band_id, upper=up.band_id, ref_gap=up.ref_cy - lo.ref_cy)
        )

    return HierGraph(
        bands=bands,
        band_graphs=band_graphs,
        band_edges=band_edges,
        sizes=sizes,
        root_band=bands[0].band_id if bands else 0,
        compaction=getattr(intent, "compaction_spec", None),
        relax_intent=(
            intent
            if str(getattr(intent, "placement_engine", "bands")) == "relax"
            else None
        ),
    )


# Minimum vertical channel (um) between two stacked bands at gf180 sizes. A small
# positive gap keeps the band boundary devices from kissing the band above. (Pure
# band ABUT would use 0; we keep a conservative channel so the realized stack never
# regresses to a cross-band touch under float rounding.)
_BAND_MIN_CHANNEL_UM: float = 1.0

# Name prefix of a RESERVATION pseudo row (the topo_sketch rail-row contract,
# re-exported there as ``RAIL_ROW_PREFIX``): a device-free band whose HEIGHT
# already includes its wall clearance (the consumer attaches ``width_um`` =
# trunk width + 2 x metal min-space). The stacker therefore stacks a
# reservation band FLUSH to its neighbours -- adding ``band_channel_um`` on a
# reservation seam would COUNT THE CLEARANCE TWICE (the P2 corridor
# double-count: 4.46 + 4.46 + 3 x 1.0 = 11.92 um where the two trunk
# corridors honestly need 8.92 um).
RESERVATION_ROW_PREFIX: str = "__rail__"


def _is_reservation_band(band: "Band") -> bool:
    """True iff every member of the band is a reservation pseudo row."""
    return bool(band.devices) and all(
        str(n).startswith(RESERVATION_ROW_PREFIX) for n in band.devices
    )

# Per-PDK minimum device-to-device CHANNEL (um) used by the within-band non-overlap
# solver: two devices separated on an axis are pushed center-to-center to
# (size_i+size_j)/2 + this gap, so their faces are at least this far apart. This is the
# only PDK-DEPENDENT quantity in the order+min-spacing realization; the ORDER itself is
# PDK-independent structure (derived from the reference relative positions).
_DEV_MIN_GAP_UM: float = 1.0


# --------------------------------------------------------------------------- #
# WITHIN-BAND NON-OVERLAP solve (order + min spacing, NOT raw external_pdk offsets).    #
#                                                                              #
# WHY ORDER + MIN SPACING, NOT THE EXACT OFFSET                                #
# ============================================                                #
# The exact-offset solve reproduces the reference EXACTLY (the offset magnitudes #
# ARE the reference), but at gf180 sizes those magnitudes do NOT separate the     #
# grown devices: an ``align:x`` pair (shared column) keeps the external_pdk y-gap (2.13um) #
# between two devices that at gf180 are 41.6um and 21.6um tall -> they overlap by   #
# 85um^2. The fix realizes the UN-PINNED axis from the reference's relative ORDER  #
# (a PDK-independent structural relation: which device is above/left of which) +    #
# the gf180 MIN SPACING, so every pair is separated on at least one axis BY          #
# CONSTRUCTION. The pinned axis (align/symmetric/abut) is still honoured exactly.    #
# --------------------------------------------------------------------------- #


def _band_orders(band_centroids: Dict[str, _XY]) -> Tuple[List[str], List[str]]:
    """Derive the per-axis total ORDER of a band's devices from reference centroids.

    Returns ``(x_order, y_order)``: the device names sorted by reference centroid-x
    (left -> right) and by reference centroid-y (bottom -> top). Ties break by name so
    the order is DETERMINISTIC. This ORDER is the PDK-independent structural relation
    (who is left-of / above whom in the reference); the non-overlap solver realizes the
    separation in this same order at the target PDK's sizes + min spacing.
    """
    names = sorted(band_centroids)
    x_order = sorted(names, key=lambda n: (band_centroids[n][0], n))
    y_order = sorted(names, key=lambda n: (band_centroids[n][1], n))
    return (x_order, y_order)


def _structural_axis_constraints(
    edges: List[RelEdge],
) -> Tuple[Dict[frozenset, str], Dict[frozenset, RelEdge]]:
    """From a band's constraint edges, decide for each pair which axis is PINNED.

    Returns ``(pinned, repr_edge)`` where ``pinned[frozenset(a,b)]`` is:
      * ``'x'``  the pair shares an x-coordinate (same column) -> separate on Y;
      * ``'y'``  the pair shares a y-coordinate (same row)    -> separate on X;
      * ``'abut_x'`` the pair abuts along x (flush, faces touch on x);
      * ``'abut_y'`` the pair abuts along y (flush, faces touch on y);
    and ``repr_edge`` carries one representative edge per pair (for the reference dx/dy
    sign). A pair with NO structural edge is absent from both dicts (free pair --
    separated purely by order on its dominant reference axis).

    align:x PINs the x-axis (shared column -> separate on y). align:y AND symmetric both
    PIN the y-axis (a symmetric pair mirrors about the cell's VERTICAL axis -> SAME y,
    OPPOSITE x, i.e. a shared row -> separate on x). abut pins the dominant separating
    axis to FLUSH. abut dominates (it is the most specific flush relation).
    """
    pinned: Dict[frozenset, str] = {}
    repr_edge: Dict[frozenset, RelEdge] = {}
    for e in edges:
        key = frozenset((e.child, e.parent))
        repr_edge.setdefault(key, e)
        if e.kind == "abut":
            ax = "abut_x" if abs(e.dx) >= abs(e.dy) else "abut_y"
            pinned[key] = ax  # abut dominates (most specific flush relation)
        elif e.kind == "align:x":
            # shared column (x pinned) -- only set if no abut already chosen.
            pinned.setdefault(key, "x")
        elif e.kind in ("align:y", "symmetric"):
            # shared row (y pinned): align:y directly, symmetric via mirror-about-x-axis
            # (same y, opposite x). Only set if no abut already chosen.
            pinned.setdefault(key, "y")
    return (pinned, repr_edge)


def solve_band_nonoverlap(
    band_devices: List[str],
    edges: List[RelEdge],
    sizes: Dict[str, _XY],
    centroids: Dict[str, _XY],
    min_gap: float = _DEV_MIN_GAP_UM,
    tight_sizes: Optional[Dict[str, _XY]] = None,
    tight_pairs: Optional[set] = None,
) -> Dict[str, _XY]:
    """Place a band's devices OVERLAP-FREE BY CONSTRUCTION at the GIVEN sizes.

    INTRA-GROUP ABUT (P-A, opt-in): ``tight_pairs`` is a set of
    ``frozenset((a, b))`` pairs that share ONE enclosure (a compact matched
    group ring), so the per-device ring PADDING baked into ``sizes`` must NOT be
    counted BETWEEN them (the ring pad belongs at the group's outer boundary,
    not between members). For such a pair the SEPARATION half-extent is read
    from ``tight_sizes`` (the RAW, unpadded stamp extents) instead of ``sizes``,
    so the two members place edge-to-edge at their true device spacing. Every
    OTHER pair (inter-group / inter-kind) keeps the padded ``sizes`` separation,
    so the per-group rings stay disjoint. The drawn stamp geometry is untouched;
    only the reserved placement footprint between tight members shrinks.
    ``tight_pairs=None`` (the default) is byte-identical to the historical
    padded-everywhere separation.

    The reference offset MAGNITUDES are NOT used. Instead the un-pinned axis of every
    pair is realized from the reference's relative ORDER + the target PDK's min spacing,
    while the structural relations are honoured:

      * ``align:x`` / ``symmetric`` -> the pair shares an x column (x pinned); they are
        separated on Y by the reference y-order (>= half+half+min_gap).
      * ``align:y``                 -> the pair shares a y row (y pinned); separated on X
        by the reference x-order.
      * ``abut``                    -> the pair is FLUSH on its dominant axis
        (center-to-center == (size+size)/2, sign from the reference delta); on the
        other axis they share a coord (so they are adjacent, no gap).
      * no structural edge          -> separated on the axis where their reference
        centroids differ MORE (their dominant reference order) by half+half+min_gap.

    Method: a longest-path / COMPACTION per axis. We build directed separation
    constraints ``pos[j] - pos[i] >= sep`` (i before j in the reference order on that
    axis) for every pair that must be separated on that axis, plus EQUALITY constraints
    for shared coords (align / abut-shared-face). Longest-path from a virtual source
    assigns each device the minimal coordinate satisfying every separation -> no two
    devices share area (every pair separated on at least one axis). Deterministic.

    ROW ALIGNMENT (P3): a y-equality group (a shared row) aligns its members'
    BOTTOM edges, not their centroids -- mixed-height rows sit on one bottom
    line with no (h_max - h_i)/2 steps; all edge-to-edge clearances are
    unchanged (see ``solve_axis``). X-equality groups stay centroid-shared
    (a shared column IS a shared centroid x).

    Returns ``{name -> (cx, cy)}`` centroids (um), one band, local frame.
    """
    devs = sorted(band_devices)
    if not devs:
        return {}
    if len(devs) == 1:
        return {devs[0]: (0.0, 0.0)}

    def half(n: str, axis: int) -> float:
        return sizes.get(n, (0.0, 0.0))[axis] / 2.0

    _tight_pairs = tight_pairs or frozenset()
    _tight_sizes = tight_sizes or {}

    def sep_half(n: str, axis: int, tight: bool) -> float:
        """Half-extent used for SEPARATION. For a tight (same-ring) pair, the
        RAW (unpadded) extent so the members abut at their true device spacing;
        otherwise the padded ``sizes`` extent (ring room between groups)."""
        if tight and n in _tight_sizes:
            return _tight_sizes[n][axis] / 2.0
        return half(n, axis)

    pinned, repr_edge = _structural_axis_constraints(edges)
    x_order = sorted(devs, key=lambda n: (centroids[n][0], n))
    y_order = sorted(devs, key=lambda n: (centroids[n][1], n))
    x_rank = {n: i for i, n in enumerate(x_order)}
    y_rank = {n: i for i, n in enumerate(y_order)}

    # Per-axis constraint lists. Each entry: (lo, hi, sep) meaning pos[hi]-pos[lo] >= sep
    # where lo precedes hi in the axis order. Also EQUALITY groups (union-find) for
    # shared coords (align pinned axis / abut shared face) so the pinned axis collapses.
    x_sep: List[Tuple[str, str, float]] = []
    y_sep: List[Tuple[str, str, float]] = []
    # equality (shared-coordinate) union-find per axis.
    eqx: Dict[str, str] = {n: n for n in devs}
    eqy: Dict[str, str] = {n: n for n in devs}

    def _find(uf: Dict[str, str], n: str) -> str:
        while uf[n] != n:
            uf[n] = uf[uf[n]]
            n = uf[n]
        return n

    # deterministic union (smaller name canonical) so two devices sharing a pinned
    # coordinate collapse to one position on that axis.
    def _union_det(uf: Dict[str, str], a: str, b: str) -> None:
        ra, rb = _find(uf, a), _find(uf, b)
        if ra != rb:
            if ra < rb:
                uf[rb] = ra
            else:
                uf[ra] = rb

    def order_pair(a: str, b: str, rank: Dict[str, int]) -> Tuple[str, str]:
        return (a, b) if rank[a] <= rank[b] else (b, a)

    def add_sep(a: str, b: str, axis: int) -> None:
        """Separate a,b on ``axis`` (0=x,1=y) in the reference order by half+half+gap."""
        rank = x_rank if axis == 0 else y_rank
        lo, hi = order_pair(a, b, rank)
        tight = frozenset((a, b)) in _tight_pairs
        sep = sep_half(lo, axis, tight) + sep_half(hi, axis, tight) + min_gap
        (x_sep if axis == 0 else y_sep).append((lo, hi, sep))

    # 1. apply structural pins + separate every pair on its non-pinned axis.
    for i in range(len(devs)):
        for j in range(i + 1, len(devs)):
            a, b = devs[i], devs[j]
            key = frozenset((a, b))
            kind = pinned.get(key)
            if kind == "x":
                _union_det(eqx, a, b)      # share x column
                add_sep(a, b, 1)           # separate on y
            elif kind == "y":
                _union_det(eqy, a, b)      # share y row
                add_sep(a, b, 0)           # separate on x
            elif kind == "abut_x":
                # flush on x (faces touch): center-to-center x = (w+w)/2, no gap; share
                # y coord (the abut keeps the perpendicular axis aligned).
                rank = x_rank
                lo, hi = order_pair(a, b, rank)
                tight = key in _tight_pairs
                sep = sep_half(lo, 0, tight) + sep_half(hi, 0, tight)  # flush
                x_sep.append((lo, hi, sep))
                _union_det(eqy, a, b)
            elif kind == "abut_y":
                rank = y_rank
                lo, hi = order_pair(a, b, rank)
                tight = key in _tight_pairs
                sep = sep_half(lo, 1, tight) + sep_half(hi, 1, tight)
                y_sep.append((lo, hi, sep))
                _union_det(eqx, a, b)
            else:
                # free pair: separate on the dominant REFERENCE axis (where centroids
                # differ more) so the reference relative order is honoured.
                ca, cb = centroids[a], centroids[b]
                if abs(ca[0] - cb[0]) >= abs(ca[1] - cb[1]):
                    add_sep(a, b, 0)
                else:
                    add_sep(a, b, 1)

    def solve_axis(
        sep: List[Tuple[str, str, float]], eq: Dict[str, str], axis: int
    ) -> Dict[str, float]:
        """Longest-path compaction on the equality-collapsed separation DAG -> coords.

        ANCHOR CONVENTION (P3 row alignment): on the X axis the group variable
        is the shared CENTROID (an equality group is a shared column). On the
        Y axis the group variable is the shared BOTTOM edge -- members of one
        row (align:y / symmetric / abut-perpendicular groups) sit on ONE
        bottom line instead of one centroid line, so mixed-height rows carry
        no (h_max - h_i)/2 bottom steps. Each device's centroid is the group
        anchor plus its own half-height offset; every centroid separation
        constraint converts to the anchor frame (s' = s + off(lo) - off(hi)),
        so the edge-to-edge clearances are EXACTLY the ones the centroid form
        guaranteed (for a y separation: bottom_hi - top_lo >= min_gap, by
        construction)."""
        # collapse equality groups: each device maps to its group rep; constraints are
        # between groups. A self-loop (lo,hi in same group with sep>eps) would be an
        # inconsistency -- we drop sep==0 self-edges (abut shared face is fine) and keep
        # the order otherwise (the reference never makes two same-coord devices need a
        # positive same-axis gap on the PINNED axis, by construction).
        def off(n: str) -> float:
            return half(n, 1) if axis == 1 else 0.0

        reps = {n: _find(eq, n) for n in devs}
        groups = sorted(set(reps.values()))
        adj: Dict[str, List[Tuple[str, float]]] = {g: [] for g in groups}
        indeg: Dict[str, int] = {g: 0 for g in groups}
        seen_edges = set()
        for lo, hi, s in sep:
            glo, ghi = reps[lo], reps[hi]
            if glo == ghi:
                continue  # same group on this axis -> no separation on it
            ek = (glo, ghi)
            # keep the MAX separation between two groups (deterministic), dedup direction
            adj[glo].append((ghi, s + off(lo) - off(hi)))
            indeg[ghi] += 1
            seen_edges.add(ek)
        # longest path via topological order (the DAG is acyclic: edges follow the axis
        # rank order, lo-rank -> hi-rank, so never cyclic).
        from collections import deque

        pos_g: Dict[str, float] = {g: 0.0 for g in groups}
        q = deque(sorted(g for g in groups if indeg[g] == 0))
        topo: List[str] = []
        indeg_w = dict(indeg)
        while q:
            g = q.popleft()
            topo.append(g)
            for h, _s in sorted(adj[g]):
                indeg_w[h] -= 1
                if indeg_w[h] == 0:
                    q.append(h)
            # keep queue deterministic
            q = deque(sorted(q))
        for g in topo:
            for h, s in adj[g]:
                if pos_g[h] < pos_g[g] + s:
                    pos_g[h] = pos_g[g] + s
        return {n: pos_g[reps[n]] + off(n) for n in devs}

    xpos = solve_axis(x_sep, eqx, 0)
    ypos = solve_axis(y_sep, eqy, 1)
    return {n: (xpos[n], ypos[n]) for n in devs}


def solve_hierarchical(
    hier: HierGraph,
    sizes: Optional[Dict[str, _XY]] = None,
    root_xy: _XY = (0.0, 0.0),
    band_channel_um: float = _BAND_MIN_CHANNEL_UM,
) -> Tuple[Dict[str, _XY], float]:
    """Solve the hierarchical graph -> ``({name -> (cx, cy)}, max_resid)``.

    (a) Solve EACH band's internal constraint graph (``solve_constraint_graph`` with
        ``sizes``) -> the band's devices' relative positions + the band's bbox.
    (b) STACK the bands by the band-level edges: the bottom (root) band is pinned at
        ``root_xy``; each next band up is translated so its BOTTOM sits a channel
        above the band below's TOP. At REFERENCE sizes the channel is taken to
        REPRODUCE the reference band gap (consistency); at NON-reference (gf180) sizes
        a fixed ``band_channel_um`` min channel is used so the bands stack with gf180
        spacing (no cross-band overlap by construction).
    (c) Compose -> absolute device centroids.

    The x-translation of each band is chosen to preserve the reference band-to-band
    x-centroid offset (the within-tier columns line up as the reference intended; x
    is not the colliding axis -- the bands collide vertically).

    ``max_resid`` is the max over all bands of each band's internal
    over-determined residual (reported honestly). At reference sizes with reference
    band gaps the whole stack reproduces the reference exactly (mod translation).
    """
    if not hier.bands:
        return ({}, 0.0)
    use_sizes = sizes if sizes is not None else hier.sizes
    use_sizes = use_sizes or {}

    # detect reference vs retarget: reference iff the passed sizes equal hier.sizes
    # (the consistency path reproduces the reference band gaps exactly).
    is_reference = sizes is None or sizes is hier.sizes or sizes == hier.sizes

    # (a) solve each band internally, centered at its own origin.
    band_local: Dict[int, Dict[str, _XY]] = {}
    band_bbox: Dict[int, _Bbox] = {}
    max_resid = 0.0
    for b in hier.bands:
        cg = hier.band_graphs[b.band_id]
        pos, resid = solve_constraint_graph(cg, use_sizes, (0.0, 0.0))
        max_resid = max(max_resid, resid)
        band_local[b.band_id] = pos
        # band bbox from solved device centroids + their sizes.
        xs0 = []; ys0 = []; xs1 = []; ys1 = []
        for n, (cx, cy) in pos.items():
            w, h = use_sizes.get(n, (0.0, 0.0))
            xs0.append(cx - w / 2.0); xs1.append(cx + w / 2.0)
            ys0.append(cy - h / 2.0); ys1.append(cy + h / 2.0)
        if xs0:
            band_bbox[b.band_id] = (min(xs0), min(ys0), max(xs1), max(ys1))
        else:
            band_bbox[b.band_id] = (0.0, 0.0, 0.0, 0.0)

    # Each band's solved-local centroid (mean of the band's solved device centroids).
    def band_local_centroid(bid: int) -> _XY:
        pos = band_local[bid]
        if not pos:
            return (0.0, 0.0)
        sx = sum(p[0] for p in pos.values()) / len(pos)
        sy = sum(p[1] for p in pos.values()) / len(pos)
        return (sx, sy)

    # (b) stack the bands bottom-up.
    #
    # X (NOT the colliding axis): every band is translated so its solved-local centroid
    # -x maps onto the band's REFERENCE centroid-x (shifted by root_xy[0]). This keeps
    # the reference cross-band x relation -- at reference sizes the band CG reproduces
    # the within-band x exactly, so mapping the centroids reproduces the reference x;
    # at gf180 sizes it is the honest x anchor (x is not where the bands collide).
    #
    # Y (the colliding axis): the bottom (root) band anchors so its solved centroid-y
    # maps onto its reference centroid-y (+ root_xy[1]). At REFERENCE sizes each higher
    # band reproduces the reference centroid-y GAP exactly (consistency). At gf180 sizes
    # each higher band's BOTTOM is placed a fixed min channel above the band below's
    # TOP -- the bands stack with gf180 spacing, so no device can cross into the band
    # above (the cross-band overlap is removed BY CONSTRUCTION).
    band_translate: Dict[int, _XY] = {}
    ref0_cx = hier.bands[0].ref_cx
    ref0_cy = hier.bands[0].ref_cy

    prev_top_abs: Optional[float] = None
    prev_abs_cy: Optional[float] = None
    for idx, b in enumerate(hier.bands):
        bid = b.band_id
        bx = band_bbox[bid]
        loc_cx, loc_cy = band_local_centroid(bid)
        loc_top = bx[3]
        loc_cy_box = (bx[1] + bx[3]) / 2.0

        # x: map local centroid-x onto reference centroid-x (relative to root).
        tx = (float(root_xy[0]) + (b.ref_cx - ref0_cx)) - loc_cx

        if idx == 0:
            # bottom band: map local centroid-y onto reference centroid-y.
            ty = float(root_xy[1]) - loc_cy
        elif is_reference:
            # reproduce the reference centroid-y gap from the band below.
            below = hier.bands[idx - 1]
            ref_gap = b.ref_cy - below.ref_cy
            this_abs_cy = prev_abs_cy + ref_gap
            ty = this_abs_cy - loc_cy
        else:
            # gf180: this band's bottom = band-below top + min channel.
            this_bottom_abs = prev_top_abs + band_channel_um
            ty = this_bottom_abs - bx[1]

        band_translate[bid] = (tx, ty)
        prev_top_abs = ty + loc_top
        prev_abs_cy = ty + loc_cy

    # (c) compose: absolute centroids = local + band translation.
    out: Dict[str, _XY] = {}
    for b in hier.bands:
        tx, ty = band_translate[b.band_id]
        for n, (cx, cy) in band_local[b.band_id].items():
            out[n] = (cx + tx, cy + ty)
    return (out, max_resid)


def _stack_bands(
    hier: HierGraph,
    band_local: Dict[int, Dict[str, _XY]],
    band_bbox: Dict[int, _Bbox],
    use_sizes: Dict[str, _XY],
    root_xy: _XY,
    band_channel_um: float,
    is_reference: bool,
    channel_um_per_gap: Optional[Dict[int, float]] = None,
) -> Dict[str, _XY]:
    """Stack per-band local placements bottom-up -> absolute centroids.

    Shared by ``solve_hierarchical`` (exact) and ``solve_hierarchical_nonoverlap``: the
    bottom band anchors at ``root_xy``; at reference sizes each higher band reproduces
    the reference centroid-y gap, at gf180 sizes each higher band's BOTTOM sits a min
    channel above the band below's TOP (no cross-band overlap by construction).

    X (P3 row centering): each band is translated so its solved BBOX CENTER
    maps onto the band's input-frame BBOX CENTER (relative to the root
    band's). For sketch-derived inputs every row is emitted centered on x=0,
    so every solved band centers on the outline center EXACTLY (the previous
    mean-centroid mapping drifted on bands with internal 2D structure, where
    the mean centroid is not the box center).

    RESERVATION SEAMS (P2 corridor accounting): a gap whose lower OR upper
    band is a reservation pseudo row (``RESERVATION_ROW_PREFIX``) stacks
    FLUSH (channel 0) -- the reservation's height already contains its wall
    clearance by the corridor-width contract (trunk + 2 x metal min-space,
    attached by the consumer), so adding ``band_channel_um`` there would
    double-count the clearance. ``channel_um_per_gap`` requests are also
    ignored on reservation seams: a corridor is a trunk reservation, not a
    crossing-net routing channel (the S6 demands ride the device-row gaps).

    ``channel_um_per_gap`` (plan S6, optional): per-band-gap channel widths
    (um), keyed by gap index (gap k = between bands k and k+1 bottom-up). The
    values come from the CALLER's design-sourced demand lowering
    (``topo_sketch.sketch_channel_um_per_gap``) -- this module stays pure
    geometry and looks up no PDK. Every per-gap value is FLOORED by
    ``band_channel_um`` (the channel never narrows below the historical min);
    a missing key falls back to ``band_channel_um``. ``None`` (the default) is
    byte-identical to the historical uniform channel.
    """
    def band_local_centroid(bid: int) -> _XY:
        pos = band_local[bid]
        if not pos:
            return (0.0, 0.0)
        sx = sum(p[0] for p in pos.values()) / len(pos)
        sy = sum(p[1] for p in pos.values()) / len(pos)
        return (sx, sy)

    band_translate: Dict[int, _XY] = {}
    ref0_bcx = (hier.bands[0].x_lo + hier.bands[0].x_hi) / 2.0

    prev_top_abs: Optional[float] = None
    prev_abs_cy: Optional[float] = None
    for idx, b in enumerate(hier.bands):
        bid = b.band_id
        bx = band_bbox[bid]
        _loc_cx, loc_cy = band_local_centroid(bid)
        loc_top = bx[3]

        # x: map the solved band bbox center onto the input-frame band bbox
        # center (relative to the root band) -- rows center on one axis.
        loc_bcx = (bx[0] + bx[2]) / 2.0
        ref_bcx = (b.x_lo + b.x_hi) / 2.0
        tx = (float(root_xy[0]) + (ref_bcx - ref0_bcx)) - loc_bcx

        if idx == 0:
            ty = float(root_xy[1]) - loc_cy
        elif is_reference:
            below = hier.bands[idx - 1]
            ref_gap = b.ref_cy - below.ref_cy
            this_abs_cy = prev_abs_cy + ref_gap
            ty = this_abs_cy - loc_cy
        elif _is_reservation_band(b) or _is_reservation_band(hier.bands[idx - 1]):
            # reservation seam: flush (the corridor row carries its own
            # clearance; no band channel, no S6 widening here).
            ty = prev_top_abs - bx[1]
        else:
            ch = band_channel_um
            if channel_um_per_gap is not None:
                # gap idx-1 = between band idx-1 and band idx; floored by the
                # historical min channel (never narrower than before).
                ch = max(ch, float(channel_um_per_gap.get(idx - 1,
                                                          band_channel_um)))
            this_bottom_abs = prev_top_abs + ch
            ty = this_bottom_abs - bx[1]

        band_translate[bid] = (tx, ty)
        prev_top_abs = ty + loc_top
        prev_abs_cy = ty + loc_cy

    out: Dict[str, _XY] = {}
    for b in hier.bands:
        tx, ty = band_translate[b.band_id]
        for n, (cx, cy) in band_local[b.band_id].items():
            out[n] = (cx + tx, cy + ty)
    return out


def _band_bbox_from_local(
    pos: Dict[str, _XY], sizes: Dict[str, _XY]
) -> _Bbox:
    xs0 = []; ys0 = []; xs1 = []; ys1 = []
    for n, (cx, cy) in pos.items():
        w, h = sizes.get(n, (0.0, 0.0))
        xs0.append(cx - w / 2.0); xs1.append(cx + w / 2.0)
        ys0.append(cy - h / 2.0); ys1.append(cy + h / 2.0)
    if not xs0:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def solve_hierarchical_nonoverlap(
    hier: HierGraph,
    sizes: Dict[str, _XY],
    root_xy: _XY = (0.0, 0.0),
    band_channel_um: float = _BAND_MIN_CHANNEL_UM,
    min_gap: float = _DEV_MIN_GAP_UM,
    channel_um_per_gap: Optional[Dict[int, float]] = None,
    tight_sizes: Optional[Dict[str, _XY]] = None,
    tight_pairs: Optional[set] = None,
) -> Tuple[Dict[str, _XY], float]:
    """Solve the hierarchical graph with WITHIN-BAND NON-OVERLAP realization.

    Unlike :func:`solve_hierarchical` (which realizes each band's internal positions by
    the EXACT reference-offset constraint solve -- correct at reference sizes but NOT
    overlap-free at non-reference sizes), this solves EACH band internally with
    :func:`solve_band_nonoverlap`: the un-pinned axis of every within-band pair is set by
    the reference ORDER + ``min_gap`` (gf180 spacing), so every band is overlap-free BY
    CONSTRUCTION at the given ``sizes``. Bands then stack with gf180 spacing (the
    band-level relation) exactly as :func:`solve_hierarchical` does at non-reference
    sizes. The result has ZERO overlapping pairs at ``sizes``.

    ``sizes`` MUST be the target (gf180) device sizes -- this is the migration path.
    Returns ``({name -> (cx, cy)}, 0.0)`` (residual is 0: non-overlap is by construction,
    there is no offset residual to report).

    ``channel_um_per_gap`` (plan S6, optional, passed through to
    :func:`_stack_bands`): per-band-gap channel widths (um) from the caller's
    design-sourced demand lowering, each floored by ``band_channel_um``.
    ``None`` (the default) keeps the uniform channel byte-identical.
    """
    if not hier.bands:
        return ({}, 0.0)
    sizes = sizes or {}

    band_local: Dict[int, Dict[str, _XY]] = {}
    band_bbox: Dict[int, _Bbox] = {}
    for b in hier.bands:
        cg = hier.band_graphs[b.band_id]
        # reference centroids = the band CG's EXACT solve at reference sizes (the ORDER
        # source: who is left-of / above whom in the reference). This is the PDK-
        # independent structural relation the non-overlap realization honours.
        ref_cents, _ = solve_constraint_graph(cg, cg.sizes, (0.0, 0.0))
        pos = solve_band_nonoverlap(
            b.devices, list(cg.edges), sizes, ref_cents, min_gap=min_gap,
            tight_sizes=tight_sizes, tight_pairs=tight_pairs,
        )
        band_local[b.band_id] = pos
        band_bbox[b.band_id] = _band_bbox_from_local(pos, sizes)

    out = _stack_bands(
        hier, band_local, band_bbox, sizes, root_xy, band_channel_um,
        is_reference=False, channel_um_per_gap=channel_um_per_gap,
    )
    # P0 OUTLINE COMPACTION (opt-in, carried by the sketch's compaction_spec
    # through the intent view): a POST-stack x/y alternating constraint-graph
    # press. The press receives EXACTLY the parameters this solve enforced
    # (min_gap / band channel / per-gap channels -- single source), so it can
    # never undercut a clearance the stack guaranteed; intent edges (order,
    # channels, rail flush seams, partner offsets) are in the press graph by
    # construction. None (the default) skips the pass byte-identically.
    spec = getattr(hier, "compaction", None)
    if spec is not None:
        from clay.layout.extract.compaction import compact_placement

        out = compact_placement(
            out, sizes, spec, min_gap=min_gap, band_channel=band_channel_um,
            channel_um_per_gap=channel_um_per_gap,
        )
        # FULL-2D FACE (compact='2d'): the deterministic embed-search stage
        # over the pressed output -- cross-row x-repack moves of
        # partner-rigid groups, every trial legalized by the SAME press,
        # accepted only on strict device-outline gain. The accepted moves
        # are self-declared on ``hier.compaction_notes`` (honest record of
        # the within-row x order given up). mode='press' (the historical
        # compact=True) skips this byte-identically.
        if getattr(spec, "mode", "press") == "2d":
            from clay.layout.extract.compaction import embed_search

            out, hier.compaction_notes = embed_search(
                out, sizes, spec, min_gap=min_gap,
                band_channel=band_channel_um,
                channel_um_per_gap=channel_um_per_gap,
            )
    # P1 RELAX REFINEMENT (opt-in, carried by the sketch's engine='relax'
    # request through the intent view): the continuous-relaxation stage over
    # THIS solve's own output (the design-self-produced seed) -- soft style
    # bands with honest off-band deviation notes, rigid symmetry bodies,
    # legalized back through the SAME compaction press. The report rides on
    # ``hier.relax_report``; the relaxer asserts no legality. None (the
    # default) skips the stage byte-identically.
    if getattr(hier, "relax_intent", None) is not None:
        from clay.layout.extract.place_relax import refine_banded_placement

        out, report = refine_banded_placement(
            hier, out, sizes, min_gap=min_gap, band_channel=band_channel_um,
            channel_um_per_gap=channel_um_per_gap,
        )
        hier.relax_report = report
    return (out, 0.0)


# --------------------------------------------------------------------------- #
# PATTERN-PRESERVING solve (minimal-perturbation legalization).                 #
#                                                                              #
# WHY THIS, NOT ORDER + MIN-SPACING                                            #
# =================================                                            #
# ``solve_band_nonoverlap`` / ``solve_hierarchical_nonoverlap`` realize the     #
# un-pinned axis of every pair from ORDER + the target PDK min spacing. That is  #
# overlap-free, but it DISCARDS the reference's relative GEOMETRY: every gap      #
# collapses to the minimum, so the recognizable proportions of the reference     #
# (the big offsets that define the arrangement) are lost -- the migrated layout  #
# no longer LOOKS like the reference (mangled aspect / tier proportions). 0      #
# overlaps is necessary but NOT sufficient for a faithful migration.            #
#                                                                              #
# ``solve_pattern_preserving`` instead STARTS from the reference relative        #
# arrangement (the EXACT offsets, which carry the reference pattern) and         #
# EXPANDS an offset ONLY where the bigger gf180 devices would otherwise overlap  #
# on that offset's dominant axis. Offsets already big enough are kept UNCHANGED  #
# -- that is what preserves the pattern: the large reference proportions stay,    #
# only the too-tight gaps grow to the minimum clearing distance (keeping the      #
# SIGN/direction and the other component's reference value). Structural edges     #
# (abut / align / symmetric) are honoured exactly. At reference sizes nothing     #
# needs expanding, so the solve reproduces the reference (max_dev ~ 0).          #
# --------------------------------------------------------------------------- #

# Minimum device-to-device CHANNEL (um) the pattern-preserving solver clears a
# too-tight offset to: an offset that puts two devices closer than
# (half_a + half_b + this) on its dominant axis is expanded to exactly that
# clearance (keeping the reference sign). The same per-PDK quantity as
# _DEV_MIN_GAP_UM (the only PDK-dependent number; the offsets themselves are the
# reference's recognizable geometry).
_PP_MIN_GAP_UM: float = _DEV_MIN_GAP_UM


def _flatten_to_constraint_graph(graph_or_hier: Any) -> "ConstraintGraph":
    """Return a flat :class:`ConstraintGraph` from a ConstraintGraph (identity) or a
    :class:`HierGraph` (union of every band's edges + the cross-band offset bridges
    rebuilt from the reference band centroids).

    A HierGraph carries per-band constraint graphs but DROPS the cross-band relation
    into band-stacking edges (order + min-spacing). To preserve the reference PATTERN
    we instead need the cross-band EXACT offsets back -- so we re-derive a single
    nearest-neighbour offset bridge between adjacent bands from the reference band
    centroids. The within-band structural + offset edges are taken verbatim.
    """
    if isinstance(graph_or_hier, ConstraintGraph):
        return graph_or_hier
    hier = graph_or_hier
    edges: List[RelEdge] = []
    names: List[str] = []
    sizes: Dict[str, _XY] = dict(getattr(hier, "sizes", {}) or {})
    axis_x: Optional[float] = None
    for b in getattr(hier, "bands", []):
        cg = hier.band_graphs[b.band_id]
        edges.extend(cg.edges)
        names.extend(cg.names)
        if cg.axis_x is not None and axis_x is None:
            axis_x = cg.axis_x
    # cross-band offset bridges: connect each band's reference-centroid-nearest device
    # to the band below's nearest device with an EXACT reference offset (so the
    # cross-band reference geometry -- the recognizable tier spacing -- is preserved
    # and only expanded if too tight).
    bands = list(getattr(hier, "bands", []))
    for i in range(len(bands) - 1):
        lo, up = bands[i], bands[i + 1]
        if not lo.devices or not up.devices:
            continue
        # nearest reference-centroid pair across the two bands.
        best = None
        for a in lo.devices:
            for c in up.devices:
                # reference centroids reconstructed from band ref means is not exact per
                # device; instead use the band ref centroid as the anchor offset so the
                # bridge carries the reference TIER gap (the pattern-defining proportion).
                cand = (abs(hash((a, c))), a, c)  # deterministic, value unused below
                if best is None or (a, c) < (best[1], best[2]):
                    best = cand
        if best is not None:
            _h, a, c = best
            # exact reference offset between the band ref centroids (tier proportion).
            dx = up.ref_cx - lo.ref_cx
            dy = up.ref_cy - lo.ref_cy
            edges.append(RelEdge(child=c, parent=a, dx=dx, dy=dy, kind="offset"))
    root = bands[0].devices[0] if bands and bands[0].devices else (
        getattr(hier, "root_band", "") and ""
    )
    if not root and names:
        root = sorted(names)[0]
    return ConstraintGraph(
        root=root, edges=edges, sizes=sizes, axis_x=axis_x, names=sorted(set(names))
    )


def solve_pattern_preserving(
    graph_or_hier: Any,
    sizes: Dict[str, _XY],
    *,
    min_gap_um: float = _PP_MIN_GAP_UM,
    root_xy: _XY = (0.0, 0.0),
    max_iter: int = 64,
) -> Tuple[Dict[str, _XY], float]:
    """Realize a layout PRESERVING the reference relative arrangement, expanding ONLY
    the offsets that the bigger ``sizes`` would otherwise make overlap.

    Accepts a :class:`ConstraintGraph` (preferred -- it carries every structural
    adjacency + the minimal reference offset bridges) or a :class:`HierGraph` (flattened
    via :func:`_flatten_to_constraint_graph`, which re-derives the cross-band reference
    offsets so the tier proportions are kept rather than collapsed to min-spacing).

    Per edge:
      * ``offset`` (child <- parent, dx, dy): the child goes at parent + (dx, dy) -- the
        EXACT reference offset -- UNLESS at ``sizes`` that puts the two closer than
        ``half_a + half_b + min_gap`` on the offset's DOMINANT axis. Then the dominant
        component is EXPANDED to exactly that clearance, keeping its SIGN; the other
        component keeps the reference value. A reference offset already big enough is
        kept UNCHANGED (the pattern-preserving property).
      * ``abut`` -> touching at ``sizes`` (center-to-center = (size+size)/2, ref sign).
      * ``align:x`` -> shared x (child.x = parent.x); ``align:y`` -> shared y.
      * ``symmetric`` -> child mirrors parent about the pair mid-x (y preserved).

    Realization: an iterative relaxation. Start every device at its exact-reference
    position (from ``solve_constraint_graph`` at REFERENCE sizes). Then repeatedly apply
    every edge as a directed lower-bound on the child's dominant coordinate from the
    parent (expansions only ever push devices APART, never together), honouring the
    structural equalities exactly, until no constraint is violated. Because expansions
    are monotone (push-apart only), the relaxation terminates OVERLAP-FREE on the
    expanded offsets + structural relations.

    Returns ``({name -> (cx, cy)}, max_dev)`` where ``max_dev`` is the max per-device
    deviation (um, translation-quotiented) from the EXACT reference solve -- a direct
    measure of how much perturbation the legalization introduced. At REFERENCE sizes no
    offset needs expanding, so ``max_dev ~ 0`` (the solve reproduces the reference).
    """
    cg = _flatten_to_constraint_graph(graph_or_hier)
    if not cg.names and not cg.edges:
        return ({}, 0.0)
    sizes = sizes or {}
    ref_sizes = cg.sizes or {}

    def half(n: str, axis: int, sz: Dict[str, _XY]) -> float:
        return sz.get(n, (0.0, 0.0))[axis] / 2.0

    # Reference detection: when the passed sizes ARE the reference sizes there is NO
    # device growth, so NOTHING is expanded and NOTHING is pushed apart -- the solve
    # reproduces the reference EXACTLY (bar: max_dev ~ 0). The reference layout itself
    # may have bbox-overlapping stacked pairs (shared-diffusion align pairs); at
    # reference sizes we faithfully reproduce that, we do not "fix" it. Expansion /
    # legalization only ever ADD separation relative to the reference, so they fire
    # only when devices actually grew.
    is_reference = sizes is ref_sizes or sizes == ref_sizes

    # 1. EXACT reference positions (the arrangement we measure perturbation against).
    ref_pos, _ = solve_constraint_graph(cg, ref_sizes, root_xy)

    # 2. STRUCTURAL-PRIORITY SPANNING TREE -- the SAME maximum-structural tree
    # ``solve_constraint_graph`` builds (abut > symmetric > align > offset). Propagating
    # along EXACTLY this tree is what makes the pattern-preserving solve reduce to the
    # exact reference at reference sizes (identical tree, identical realization when no
    # offset needs expanding). Each node is placed by ONE tree edge -> no over-determined
    # thrashing; the non-tree structural edges are honoured as residual (as the reference
    # already satisfies them, and expansions only push apart).
    _PRIO = {"abut": 0, "symmetric": 1, "align:x": 2, "align:y": 2, "offset": 3}
    uf: Dict[str, str] = {n: n for n in cg.names}

    def _find(n: str) -> str:
        while uf[n] != n:
            uf[n] = uf[uf[n]]
            n = uf[n]
        return n

    ranked = sorted(cg.edges, key=lambda e: (_PRIO.get(e.kind, 3), e.child, e.parent))
    tree_edges: List[RelEdge] = []
    for e in ranked:
        ra, rb = _find(e.child), _find(e.parent)
        if ra != rb:
            uf[ra] = rb
            tree_edges.append(e)

    # 3. Per tree edge, the realized child-from-parent delta (signed, target sizes).
    #    offset: exact reference (dx, dy) UNLESS the dominant axis is too tight at
    #            ``sizes`` -> expand that component to the clearing distance (sign kept,
    #            other component unchanged). This is the ONLY place a magnitude changes,
    #            and it only ever GROWS -- minimal perturbation, push-apart only.
    #    abut:   flush (center-to-center = (size+size)/2 on the dominant axis, ref sign);
    #            the perpendicular axis keeps the reference delta.
    #    align:x: child shares parent x (dx -> 0); align:y: child shares parent y.
    #    symmetric: child = parent + dx (mirror about the pair mid-x), y preserved.
    def realized_delta(e: RelEdge) -> _XY:
        p, c = e.parent, e.child
        if e.kind == "abut":
            if abs(e.dx) >= abs(e.dy):
                sep = half(p, 0, sizes) + half(c, 0, sizes)
                return ((sep if e.dx >= 0 else -sep), e.dy)
            sep = half(p, 1, sizes) + half(c, 1, sizes)
            return (e.dx, (sep if e.dy >= 0 else -sep))
        if e.kind == "align:x":
            # shared x column; keep the reference free-axis (y) offset. Free-axis
            # clearance for a too-tight grown stack is handled by the push-apart pass
            # (expanding it HERE would over-constrain a device that ALSO shares a row
            # with a third device via align:y -- the two structural edges then conflict).
            return (0.0, e.dy)
        if e.kind == "align:y":
            return (e.dx, 0.0)
        if e.kind == "symmetric":
            return (e.dx, e.dy)
        # offset: expand the dominant axis if too tight (only when devices grew).
        dx, dy = e.dx, e.dy
        if not is_reference:
            if abs(e.dx) >= abs(e.dy):
                clear = half(p, 0, sizes) + half(c, 0, sizes) + min_gap_um
                if abs(e.dx) < clear:
                    dx = clear if e.dx >= 0 else -clear
            else:
                clear = half(p, 1, sizes) + half(c, 1, sizes) + min_gap_um
                if abs(e.dy) < clear:
                    dy = clear if e.dy >= 0 else -clear
        return (dx, dy)

    # 4. Propagate along the spanning tree (undirected: place a node from whichever end
    # is already known). The root anchors at root_xy. Deterministic.
    pos: Dict[str, _XY] = {cg.root: (float(root_xy[0]), float(root_xy[1]))}
    if not pos[cg.root] and cg.names:  # root empty-string guard
        pass
    progress = True
    while progress:
        progress = False
        for e in tree_edges:
            dx, dy = realized_delta(e)
            if e.parent in pos and e.child not in pos:
                px, py = pos[e.parent]
                pos[e.child] = (px + dx, py + dy)
                progress = True
            elif e.child in pos and e.parent not in pos:
                cx, cy = pos[e.child]
                pos[e.parent] = (cx - dx, cy - dy)
                progress = True
    # any device the tree never reached: pin at the root (disconnected last resort).
    for n in cg.names:
        if n not in pos:
            pos[n] = (float(root_xy[0]), float(root_xy[1]))

    # 5. A second push-apart pass for residual overlaps NOT resolved by the tree.
    #
    # Even with every tree separation honoured, pairs can interpenetrate at target
    # sizes: (a) a pair with no structural edge whose clusters were pushed together by a
    # tree expansion elsewhere; (b) an ALIGN-stacked pair (shared column / row) whose
    # grown devices now overlap on the FREE axis -- the tree could not pre-expand it
    # without breaking a conflicting structural edge on the same device. We legalize both.
    #
    # Per pair, the AXES we are ALLOWED to push on (pushing on a pinned axis would
    # destroy the relation):
    #   * abut / symmetric : EXEMPT entirely (true flush / mirror adjacency).
    #   * align:x          : push only on Y (the free axis); keep the shared column.
    #   * align:y          : push only on X (the free axis); keep the shared row.
    #   * no structural edge: push on the reference's own dominant separating axis.
    # At reference sizes no new overlap exists -> the pass is a no-op (bar #3 exact).
    pair_kinds: Dict[frozenset, set] = {}
    for e in cg.edges:
        pair_kinds.setdefault(frozenset((e.child, e.parent)), set()).add(e.kind)

    def _allowed_axes(a: str, b: str) -> Tuple[bool, bool]:
        """(may_push_x, may_push_y) for the pair, honouring its structural relation."""
        kinds = pair_kinds.get(frozenset((a, b)), set())
        if "abut" in kinds or "symmetric" in kinds:
            return (False, False)  # flush / mirror -- never push
        if "align:x" in kinds:
            return (False, True)   # shared column -> push only on Y
        if "align:y" in kinds:
            return (True, False)   # shared row -> push only on X
        return (True, True)        # free pair -> either axis

    # cluster centroid (reference frame) -- a squeezed small device is pushed OUTWARD
    # from it (toward open space) rather than back into the pack.
    if pos:
        ccx = sum(p[0] for p in pos.values()) / len(pos)
        ccy = sum(p[1] for p in pos.values()) / len(pos)
    else:
        ccx = ccy = 0.0

    def _area(n: str) -> float:
        w, h = sizes.get(n, (0.0, 0.0))
        return w * h

    names_sorted = sorted(cg.names)
    for _ in range(max_iter if not is_reference else 0):
        moved = False
        for ii in range(len(names_sorted)):
            for jj in range(ii + 1, len(names_sorted)):
                a0, b0 = names_sorted[ii], names_sorted[jj]
                if a0 not in pos or b0 not in pos:
                    continue
                may_x, may_y = _allowed_axes(a0, b0)
                if not (may_x or may_y):
                    continue  # fully pinned (abut / symmetric)
                # MOVE the smaller-area device (least pattern perturbation); the bigger
                # device defines the macro arrangement and stays put. A structural pair
                # (align) keeps its mover constrained to the free axis.
                kinds = pair_kinds.get(frozenset((a0, b0)), set())
                if "align:x" in kinds or "align:y" in kinds:
                    mover, anchor = b0, a0  # align: child (b0) is the dependent device
                elif _area(a0) <= _area(b0):
                    mover, anchor = a0, b0
                else:
                    mover, anchor = b0, a0
                anx, any_ = pos[anchor]
                mvx, mvy = pos[mover]
                face_x = half(anchor, 0, sizes) + half(mover, 0, sizes)
                face_y = half(anchor, 1, sizes) + half(mover, 1, sizes)
                ovx = face_x - abs(mvx - anx)
                ovy = face_y - abs(mvy - any_)
                if ovx <= 1e-9 or ovy <= 1e-9:
                    continue  # disjoint on at least one axis -> no real overlap
                need_x = face_x + min_gap_um
                need_y = face_y + min_gap_um
                rax, ray = ref_pos.get(anchor, (anx, any_))
                rbx, rby = ref_pos.get(mover, (mvx, mvy))
                can = []
                if may_x:
                    can.append(("x", ovx))
                if may_y:
                    can.append(("y", ovy))
                if len(can) == 2 and abs(can[0][1] - can[1][1]) < 1e-9:
                    axis = "x" if abs(rbx - rax) >= abs(rby - ray) else "y"
                else:
                    axis = min(can, key=lambda t: t[1])[0]
                if axis == "x":
                    # push the mover OUTWARD from the cluster centroid (away from the
                    # pack) so a device squeezed between two big neighbours escapes
                    # rather than ping-ponging back in.
                    sign = 1.0 if (mvx - ccx) >= 0 else -1.0
                    if (anx + sign * need_x - mvx) * sign < 0:
                        sign = 1.0 if mvx >= anx else -1.0  # fall back to direct side
                    pos[mover] = (anx + sign * need_x, mvy)
                else:
                    sign = 1.0 if (mvy - ccy) >= 0 else -1.0
                    if (any_ + sign * need_y - mvy) * sign < 0:
                        sign = 1.0 if mvy >= any_ else -1.0
                    pos[mover] = (mvx, any_ + sign * need_y)
                moved = True
        if not moved:
            break

    # deviation from the exact reference (translation-quotiented) -- the perturbation.
    _, max_dev = positions_equal(pos, ref_pos, care_mask_um=1e9)
    return (pos, max_dev)


__all__ = [
    "RelEdge",
    "RelGraph",
    "build_rel_graph",
    "solve_rel_graph",
    "positions_equal",
    "ConstraintGraph",
    "build_constraint_graph",
    "solve_constraint_graph",
    "Band",
    "BandEdge",
    "HierGraph",
    "cluster_bands",
    "build_hierarchical_graph",
    "solve_hierarchical",
    "solve_band_nonoverlap",
    "solve_hierarchical_nonoverlap",
    "solve_pattern_preserving",
]
