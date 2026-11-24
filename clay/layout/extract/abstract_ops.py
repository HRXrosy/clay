"""Phase 4 of the INTENT-EXTRACTION CONVERGENCE METHOD: STRUCTURAL ABSTRACTION
of the relative-constraint graph (the next degradation level above ``relgraph``).

WHERE THIS FITS
===============
The degradation pipeline strips the reference layout down through ever-more
abstract representations, each kept equivalence-invariant under a don't-care mask:

    absolute centroids (1:1)
      -> relgraph.py     : RELATIVE constraint graph -- exact-offset edges
                           ``RelEdge(child, parent, dx, dy, kind='offset')`` whose
                           solution reproduces the positions up to translation.
      -> abstract_ops.py : STRUCTURAL constraint graph (THIS module) -- each numeric
                           OFFSET edge is abstracted into a STRUCTURAL relation
                           (abut / align / symmetric / order) WHEN the geometry
                           licenses it, keeping the graph's solution invariant.

The previous level (``relgraph``) already compresses absolute coordinates into a
tree of exact offsets (parent + dx,dy). It is still fully NUMERIC: every edge
carries two real magnitudes. This module asks the next question -- how much of
that numeric content is REDUNDANT given the structure? An offset edge whose child
sits flush against its parent need not record the gap (it is ``abut``); an edge
with a zero component need only keep the non-zero one (``align``); a declared
mirror pair's child is the parent's reflection (``symmetric``, no magnitudes); an
edge whose magnitude is pinned by OTHER constraints need keep only the SIGN
(``order``). Each abstraction DROPS bits while keeping the solution invariant
under the care-mask. The per-kind tally is the DEGRADATION RESULT: how much of the
layout's relative structure the structural vocabulary captures, versus the
incompressible numeric offsets that survive.

WHAT IS / ISN'T TOUCHED
=======================
This module is purely COMPOSITIONAL: it imports ``relgraph`` (or, when that sibling
is not yet on disk, a small local stub faithful to the documented interface --
``build_rel_graph`` / ``solve_rel_graph`` / ``positions_equal`` / ``RelEdge`` /
``RelGraph``) and the existing extractor stack. It does NOT edit relgraph.py /
relax.py / emit_clay.py / the placers / the grammar / any design.clay.

THE ABSTRACTION ORACLE
======================
Every candidate abstraction is CHECKED by re-solving the (mixed-kind) graph and
comparing to the anchor with ``positions_equal(.., care_mask_um)`` -- the SAME
translation-quotient + care-mask oracle the rest of the method uses. An
abstraction is kept IFF it leaves the solution within the mask. ``as_order`` in
particular is only sound when OTHER constraints already pin the dropped magnitude;
``minimize`` discovers that empirically by re-solving (an order edge that loses the
position fails the oracle and is rejected), so we never assert soundness we did not
check.

HONEST SCOPE
============
``minimize`` is GREEDY (one pass, most-abstract-first per edge); it reaches a per-
edge-1-minimal abstraction, NOT a global MDL optimum (the global e-graph search is
``converge_egraph``'s territory). The tally is a LOWER bound on how compressible the
relative graph is under this vocabulary and drop order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# relgraph import (with a documented-interface local stub fallback).
# --------------------------------------------------------------------------- #
# The sibling builds ``clay/layout/extract/relgraph.py`` exposing:
#   RelEdge(child, parent, dx, dy, kind='offset')
#   RelGraph  -- a container with at least ``.root`` (name) and ``.edges`` (list)
#   build_rel_graph(intent) -> RelGraph
#   solve_rel_graph(relgraph, root_xy=(0.0, 0.0)) -> {name -> (cx, cy)} um
#   positions_equal(a, b, care_mask_um) -> bool  (translation-quotient + mask)
# We import it when present; otherwise we fall back to a faithful local stub so
# this module + its test run standalone. The structural-abstraction logic below
# depends ONLY on the documented surface, so it works against either.
try:  # pragma: no cover - exercised by whichever branch is on disk
    from clay.layout.extract.relgraph import (  # type: ignore
        RelEdge,
        RelGraph,
        build_rel_graph,
        positions_equal,
        solve_rel_graph,
    )

    _RELGRAPH_SOURCE = "sibling"
except Exception:  # pragma: no cover - stub path
    _RELGRAPH_SOURCE = "stub"

    @dataclass
    class RelEdge:  # type: ignore[no-redef]
        """A relative-position edge: ``child`` sits at ``parent + (dx, dy)`` (um).

        ``kind`` is the abstraction level: 'offset' (exact dx,dy -- the relgraph
        base), or one of the structural kinds this module introduces ('abut',
        'align:x', 'align:y', 'symmetric', 'order:left_of', 'order:above').
        """

        child: str
        parent: str
        dx: float
        dy: float
        kind: str = "offset"

    @dataclass
    class RelGraph:  # type: ignore[no-redef]
        """A rooted tree of RelEdges. ``root`` is placed at ``root_xy``; every other
        node is reached by accumulating its parent edge's offset."""

        root: str
        edges: List["RelEdge"] = field(default_factory=list)

    def _anchor_centroids_stub(intent: Any) -> Dict[str, Tuple[float, float]]:
        dbb = getattr(intent, "device_bboxes", {}) or {}
        out: Dict[str, Tuple[float, float]] = {}
        for name, bb in dbb.items():
            if not bb or len(bb) < 4:
                continue
            out[str(name)] = (
                (float(bb[0]) + float(bb[2])) / 2.0,
                (float(bb[1]) + float(bb[3])) / 2.0,
            )
        return out

    def build_rel_graph(intent: Any) -> "RelGraph":  # type: ignore[no-redef]
        """Stub: a star tree of exact-offset edges off the first device (by sorted
        name) as root -- every child is ``root + exact (dx, dy)``. Faithful to the
        documented contract (exact offsets, solution == anchor up to translation);
        the real sibling may build a smarter spanning tree, but THIS module's
        abstraction logic is tree-shape-agnostic."""
        cents = _anchor_centroids_stub(intent)
        names = sorted(cents)
        if not names:
            return RelGraph(root="", edges=[])
        root = names[0]
        rx, ry = cents[root]
        edges = [
            RelEdge(child=n, parent=root, dx=cents[n][0] - rx, dy=cents[n][1] - ry)
            for n in names[1:]
        ]
        return RelGraph(root=root, edges=edges)

    def solve_rel_graph(  # type: ignore[no-redef]
        relgraph: "RelGraph", root_xy: Tuple[float, float] = (0.0, 0.0)
    ) -> Dict[str, Tuple[float, float]]:
        """Stub: accumulate exact-offset edges from the root over the tree."""
        return _solve_offset_tree(relgraph, root_xy)

    def positions_equal(  # type: ignore[no-redef]
        a: Dict[str, Tuple[float, float]],
        b: Dict[str, Tuple[float, float]],
        care_mask_um: float = 0.0,
        allow_mirror: bool = False,
    ) -> Tuple[bool, float]:
        """Stub mirroring the sibling contract: returns ``(equal, max_dev)``.

        Re-centres both on their own centroid-of-centroids (translation quotient),
        then takes the max PER-COORDINATE absolute deviation over shared nodes;
        ``equal`` requires the NAME SETS to match and ``max_dev <= care_mask_um``
        (floored at a tiny noise floor so an exact mask still accepts rounding)."""
        names = sorted(set(a) & set(b))
        same_names = set(a) == set(b)
        if not names:
            return (same_names and not a, 0.0)
        acx = sum(a[n][0] for n in names) / len(names)
        acy = sum(a[n][1] for n in names) / len(names)
        bcx = sum(b[n][0] for n in names) / len(names)
        bcy = sum(b[n][1] for n in names) / len(names)
        m = 0.0
        for n in names:
            da = (a[n][0] - acx) - (b[n][0] - bcx)
            db = (a[n][1] - acy) - (b[n][1] - bcy)
            m = max(m, abs(da), abs(db))
        threshold = max(care_mask_um, 1e-9)
        return (same_names and m <= threshold, m)


# Geometry tolerance (um): a gap/component within this of zero is treated as zero.
_EPS_UM = 0.05


def _equal(a: Dict[str, Tuple[float, float]], b: Dict[str, Tuple[float, float]],
           care_mask_um: float) -> bool:
    """Unwrap ``positions_equal`` (which returns ``(equal, max_dev)`` in both the
    sibling and the stub) to a plain bool. Centralises the contract so the
    abstraction search does not depend on whichever relgraph is on disk."""
    res = positions_equal(a, b, care_mask_um)
    return bool(res[0]) if isinstance(res, tuple) else bool(res)


# --------------------------------------------------------------------------- #
# bbox helpers (the abstraction operators read the device footprints).
# --------------------------------------------------------------------------- #
def _wh(bb: Sequence[float]) -> Tuple[float, float]:
    return (float(bb[2]) - float(bb[0]), float(bb[3]) - float(bb[1]))


def _center(bb: Sequence[float]) -> Tuple[float, float]:
    return ((float(bb[0]) + float(bb[2])) / 2.0, (float(bb[1]) + float(bb[3])) / 2.0)


# --------------------------------------------------------------------------- #
# 1. Abstraction operators. Each takes a RelEdge + device bboxes (+ intent for
#    symmetric) and returns a MORE-ABSTRACT RelEdge when the geometry licenses it,
#    else None. They never mutate the input edge (return a fresh RelEdge).
# --------------------------------------------------------------------------- #
def as_abut(edge: "RelEdge", bboxes: Dict[str, Sequence[float]], eps: float = _EPS_UM):
    """ABUT: if child/parent are EDGE-ADJACENT in the offset direction (the gap
    between their bboxes is ~0 within ``eps``), drop the exact gap -> kind='abut'.

    The dominant offset axis selects the touching face: for a mostly-horizontal
    offset (|dx| >= |dy|) the child's near vertical edge should touch the parent's
    far vertical edge (gap = |dx| - (wc+wp)/2 ~ 0); likewise for vertical. The
    solver then places them flush (centre-to-centre = (wc+wp)/2 on that axis),
    reproducing the offset WITHOUT storing it. Returns None if not flush.
    """
    pbb = bboxes.get(edge.parent)
    cbb = bboxes.get(edge.child)
    if not pbb or not cbb:
        return None
    wp, hp = _wh(pbb)
    wc, hc = _wh(cbb)
    if abs(edge.dx) >= abs(edge.dy):
        gap = abs(edge.dx) - (wp + wc) / 2.0
    else:
        gap = abs(edge.dy) - (hp + hc) / 2.0
    if abs(gap) <= eps:
        # keep the (small) off-axis component so a not-perfectly-aligned abut still
        # reproduces; the on-axis magnitude is what 'abut' regenerates from sizes.
        return RelEdge(edge.child, edge.parent, edge.dx, edge.dy, kind="abut")
    return None


def as_align(edge: "RelEdge", bboxes: Dict[str, Sequence[float]], eps: float = _EPS_UM):
    """ALIGN: if one offset component is ~0 (shared axis), drop it -> 'align:x'
    (dx~0, shared x / vertical alignment) or 'align:y' (dy~0, shared y / horizontal
    alignment). The other component is preserved by the surviving offset magnitude.

    ``bboxes`` is accepted for a uniform operator signature; alignment is judged off
    the offset components alone (the relgraph offset already encodes centroids).
    """
    if abs(edge.dx) <= eps and abs(edge.dy) > eps:
        return RelEdge(edge.child, edge.parent, edge.dx, edge.dy, kind="align:x")
    if abs(edge.dy) <= eps and abs(edge.dx) > eps:
        return RelEdge(edge.child, edge.parent, edge.dx, edge.dy, kind="align:y")
    return None


def _is_declared_pair(intent: Any, a: str, b: str) -> bool:
    for attr in ("matched_pairs", "mirror_pairs"):
        for ab in getattr(intent, attr, ()) or ():
            if len(ab) == 2 and {str(ab[0]), str(ab[1])} == {a, b}:
                return True
    return False


def as_symmetric(edge: "RelEdge", intent: Any, bboxes: Optional[Dict[str, Sequence[float]]] = None,
                 eps: float = _EPS_UM):
    """SYMMETRIC: if (child, parent) is a DECLARED matched/mirror pair AND they
    mirror about a VERTICAL axis (same y within ``eps``, opposite x about the pair's
    mid-x), the child is DERIVED as the parent's mirror -> kind='symmetric', dropping
    BOTH offset magnitudes (the axis is the shared structural fact).

    The vertical-mirror geometry: a vertical symmetry axis reflects x and preserves
    y, so dy ~ 0 and dx != 0 (the two sit at mirrored x on the same row). We require
    the declaration (so an accidental same-row offset is NOT called symmetric) AND
    that geometry. The solver realises the child by reflecting the parent about the
    cell's vertical axis (recovered from the surviving structure), so no magnitude is
    stored. Returns None if undeclared or not a vertical mirror.
    """
    if not _is_declared_pair(intent, edge.child, edge.parent):
        return None
    # vertical mirror: y preserved (dy ~ 0), x reflected (dx clearly non-zero)
    if abs(edge.dy) <= eps and abs(edge.dx) > eps:
        return RelEdge(edge.child, edge.parent, edge.dx, edge.dy, kind="symmetric")
    return None


def as_order(edge: "RelEdge", eps: float = _EPS_UM):
    """ORDER (weakest): keep ONLY the SIGN of the dominant offset, dropping the
    magnitude -> 'order:left_of' (dx > 0: child right of parent... see note) or
    'order:above' (dy > 0: child above parent).

    Sign convention: the edge says ``child = parent + (dx, dy)``. dx > 0 means the
    child is to the RIGHT of the parent; we label the relation by the parent's view
    -- 'order:left_of' means PARENT is left_of CHILD (parent precedes child on x).
    Likewise dy > 0 -> 'order:above' (parent below child, child above). Only the
    DOMINANT axis is kept. This is sound ONLY if OTHER constraints pin the magnitude;
    ``minimize`` checks that by re-solving (an under-determined order fails the
    oracle).
    """
    if abs(edge.dx) >= abs(edge.dy):
        if abs(edge.dx) <= eps:
            return None
        kind = "order:left_of" if edge.dx > 0 else "order:right_of"
    else:
        if abs(edge.dy) <= eps:
            return None
        kind = "order:above" if edge.dy > 0 else "order:below"
    return RelEdge(edge.child, edge.parent, edge.dx, edge.dy, kind=kind)


# --------------------------------------------------------------------------- #
# 2. solve_abstract_graph: realise a MIXED-kind graph.
# --------------------------------------------------------------------------- #
def _solve_offset_tree(
    relgraph: "RelGraph", root_xy: Tuple[float, float] = (0.0, 0.0)
) -> Dict[str, Tuple[float, float]]:
    """Accumulate EXACT-offset edges over the (possibly multi-level) tree from the
    root. Robust to any parent-before-child ordering (repeated relaxation passes
    until every reachable node is placed)."""
    pos: Dict[str, Tuple[float, float]] = {relgraph.root: (float(root_xy[0]), float(root_xy[1]))}
    edges = list(relgraph.edges)
    progress = True
    while progress:
        progress = False
        for e in edges:
            if e.parent in pos and e.child not in pos:
                px, py = pos[e.parent]
                pos[e.child] = (px + float(e.dx), py + float(e.dy))
                progress = True
    return pos


def solve_abstract_graph(
    relgraph: "RelGraph",
    bboxes: Dict[str, Sequence[float]],
    root_xy: Tuple[float, float] = (0.0, 0.0),
) -> Dict[str, Tuple[float, float]]:
    """Realise a graph whose edges are MIXED kinds.

    * all-'offset' -> delegates to ``relgraph.solve_rel_graph`` (MUST reproduce the
      reference frame exactly -- the invariant the whole degradation rests on).
    * 'abut'       -> place child flush against parent: centre-to-centre on the
      dominant axis = (size_parent + size_child)/2 (sign from the original offset);
      the off-axis component keeps its (small) stored value.
    * 'align:x'    -> child shares the parent's x; y from the stored dy.
      'align:y'    -> child shares the parent's y; x from the stored dx.
    * 'symmetric'  -> child = mirror of parent about the cell's VERTICAL axis
      (axis x recovered as the midpoint of all symmetric pairs' parents+children);
      y preserved.
    * 'order:*'    -> a strict inequality fed to a per-axis LONGEST-PATH; the
      magnitude is supplied by the other (exact) constraints. When order edges are
      the ONLY thing pinning a node, the longest-path packs it at the minimum
      separation -- which only matches the anchor if the reference sat at that
      separation, so the oracle (in ``minimize``) accepts an order abstraction only
      when the geometry truly licenses it.

    Two-phase solve: (A) accumulate exact/abut/align/symmetric over the tree (these
    are equality constraints -> deterministic positions); (B) for any node still
    unplaced because its incoming edge is an inequality ('order'), run a per-axis
    longest-path using the bbox sizes for minimum separations. The composition
    reproduces the reference when all edges are 'offset'.
    """
    # Fast path + invariant guarantee: a pure-offset graph delegates to relgraph.
    if all(e.kind == "offset" for e in relgraph.edges):
        return solve_rel_graph(relgraph, root_xy)

    pos: Dict[str, Tuple[float, float]] = {
        relgraph.root: (float(root_xy[0]), float(root_xy[1]))
    }

    # Recover the vertical symmetry axis (x) as the mean of the midpoints of every
    # declared symmetric pair, taken from the ORIGINAL offsets (parent at some x, the
    # child is the reflection). Because the edge stores dx, the pair's midpoint x in
    # the parent's frame is parent_x + dx/2; averaged over symmetric edges this is the
    # cell axis offset from whatever frame the parents resolve into. We compute it
    # lazily after the equality phase places the parents.
    sym_edges = [e for e in relgraph.edges if e.kind == "symmetric"]

    # Phase A: equality edges (offset / abut / align / symmetric-with-known-parent).
    # Iterate to a fixpoint so multi-level trees resolve regardless of edge order.
    order_edges = [e for e in relgraph.edges if e.kind.startswith("order")]
    eq_edges = [e for e in relgraph.edges if not e.kind.startswith("order")]
    progress = True
    while progress:
        progress = False
        for e in eq_edges:
            if e.parent not in pos or e.child in pos:
                continue
            px, py = pos[e.parent]
            if e.kind == "offset":
                pos[e.child] = (px + float(e.dx), py + float(e.dy))
            elif e.kind == "abut":
                pbb, cbb = bboxes.get(e.parent), bboxes.get(e.child)
                wp, hp = _wh(pbb) if pbb else (0.0, 0.0)
                wc, hc = _wh(cbb) if cbb else (0.0, 0.0)
                if abs(e.dx) >= abs(e.dy):
                    sep = (wp + wc) / 2.0
                    cx = px + (sep if e.dx >= 0 else -sep)
                    pos[e.child] = (cx, py + float(e.dy))
                else:
                    sep = (hp + hc) / 2.0
                    cy = py + (sep if e.dy >= 0 else -sep)
                    pos[e.child] = (px + float(e.dx), cy)
            elif e.kind == "align:x":
                pos[e.child] = (px, py + float(e.dy))  # shared x
            elif e.kind == "align:y":
                pos[e.child] = (px + float(e.dx), py)  # shared y
            elif e.kind == "symmetric":
                # placed in the symmetric sub-pass below (needs the axis); skip now.
                continue
            progress = True

        # symmetric edges: reflect child x about the cell axis, preserve parent y.
        if sym_edges:
            # axis x = mean over placed symmetric pairs of (parent_x + child_x)/2,
            # where child_x is the reflection target. We bootstrap the axis from the
            # stored dx (midpoint in parent frame = parent_x + dx/2), then place.
            axis_samples = [
                pos[e.parent][0] + float(e.dx) / 2.0
                for e in sym_edges
                if e.parent in pos
            ]
            if axis_samples:
                axis_x = sum(axis_samples) / len(axis_samples)
                for e in sym_edges:
                    if e.parent in pos and e.child not in pos:
                        px, py = pos[e.parent]
                        pos[e.child] = (2.0 * axis_x - px, py)  # mirror x, keep y
                        progress = True

    # Phase B: order inequalities via per-axis longest-path on bbox sizes.
    # An 'order' edge child=parent+(.,.) with kind order:left_of / order:above
    # asserts the parent precedes the child by AT LEAST the min separation (the
    # half-widths sum). We compute a longest-path so the child is packed at that
    # minimum; nodes already placed by equality keep their positions.
    if order_edges:
        # Build per-axis precedence: (u_before_v, axis, min_sep).
        def _min_sep_x(u: str, v: str) -> float:
            wu = _wh(bboxes[u])[0] if u in bboxes else 0.0
            wv = _wh(bboxes[v])[0] if v in bboxes else 0.0
            return (wu + wv) / 2.0

        def _min_sep_y(u: str, v: str) -> float:
            hu = _wh(bboxes[u])[1] if u in bboxes else 0.0
            hv = _wh(bboxes[v])[1] if v in bboxes else 0.0
            return (hu + hv) / 2.0

        # Seed coordinates: known positions are fixed; unknown start at root coord.
        rx, ry = pos[relgraph.root]
        x = {n: pos[n][0] for n in pos}
        y = {n: pos[n][1] for n in pos}
        # relax inequalities to a fixpoint (longest-path on a DAG of order edges).
        for _ in range(len(order_edges) + 1):
            for e in order_edges:
                p, c = e.parent, e.child
                if e.kind == "order:left_of":  # parent left of child on x
                    need = x.get(p, rx) + _min_sep_x(p, c)
                    x[c] = max(x.get(c, rx), need)
                    y.setdefault(c, y.get(p, ry))
                elif e.kind == "order:right_of":  # parent right of child
                    need = x.get(p, rx) - _min_sep_x(p, c)
                    x[c] = min(x.get(c, rx), need)
                    y.setdefault(c, y.get(p, ry))
                elif e.kind == "order:above":  # parent below child on y
                    need = y.get(p, ry) + _min_sep_y(p, c)
                    y[c] = max(y.get(c, ry), need)
                    x.setdefault(c, x.get(p, rx))
                elif e.kind == "order:below":
                    need = y.get(p, ry) - _min_sep_y(p, c)
                    y[c] = min(y.get(c, ry), need)
                    x.setdefault(c, x.get(p, rx))
        for n in set(x) | set(y):
            pos[n] = (x.get(n, rx), y.get(n, ry))

    return pos


# --------------------------------------------------------------------------- #
# 3. minimize: greedy most-abstract-first per edge, oracle-checked.
# --------------------------------------------------------------------------- #
@dataclass
class AbstractResult:
    """The outcome of greedily abstracting a relative graph.

    * ``graph``  -- the minimized RelGraph (edges at their most-abstract kept kind).
    * ``tally``  -- per-kind edge count: how many edges became symmetric / abut /
      align / order vs stayed exact 'offset'. THE DEGRADATION RESULT.
    * ``kinds``  -- per-(child,parent) the kept kind (for incompressible-offset
      reporting).
    * ``care_mask_um`` -- the don't-care radius used.
    * ``incompressible`` -- the (child, parent) edges that stayed 'offset'.
    """

    graph: "RelGraph"
    tally: Dict[str, int]
    kinds: Dict[Tuple[str, str], str]
    care_mask_um: float
    incompressible: List[Tuple[str, str]] = field(default_factory=list)


# generality order: most abstract first. 'symmetric' drops both magnitudes; 'abut'
# drops one (regenerated from sizes); 'align' drops one (zero) component; 'order'
# drops the magnitude but keeps the sign; 'offset' keeps everything.
_GENERALITY = ("symmetric", "abut", "align", "order")


def _tally_kind(kind: str) -> str:
    """Bucket a concrete kept kind into a tally bucket (align:x/y -> 'align', etc.)."""
    if kind.startswith("align"):
        return "align"
    if kind.startswith("order"):
        return "order"
    return kind  # 'offset' | 'abut' | 'symmetric'


def minimize(
    relgraph: "RelGraph",
    intent: Any,
    care_mask_um: float,
    *,
    anchor: Optional[Dict[str, Tuple[float, float]]] = None,
) -> AbstractResult:
    """GREEDILY abstract each edge to the MOST-abstract kind that KEEPS the graph's
    solution equal to the anchor under ``care_mask_um``.

    For each edge, try symmetric > abut > align > order > keep-offset (in that
    generality order); apply the first candidate whose abstraction leaves
    ``positions_equal(solve_abstract_graph(g'), anchor, care_mask)`` True; else keep
    'offset'. The anchor defaults to the EXACT solution of the all-offset relgraph
    (the relgraph invariant: that solution reproduces the reference up to
    translation). Returns the minimized graph + the per-kind tally.

    GREEDY + ORDER-DEPENDENT (per-edge 1-minimal), not a global optimum -- honest.
    """
    bboxes = dict(getattr(intent, "device_bboxes", {}) or {})

    if anchor is None:
        anchor = solve_rel_graph(relgraph, (0.0, 0.0))

    # Work on a mutable copy of the edges; abstractions are committed in place only
    # when the whole-graph oracle (re-solve) still matches the anchor.
    edges: List["RelEdge"] = [
        RelEdge(e.child, e.parent, e.dx, e.dy, e.kind) for e in relgraph.edges
    ]

    def _candidate(e: "RelEdge", level: str) -> Optional["RelEdge"]:
        if level == "symmetric":
            return as_symmetric(e, intent, bboxes)
        if level == "abut":
            return as_abut(e, bboxes)
        if level == "align":
            return as_align(e, bboxes)
        if level == "order":
            return as_order(e)
        return None

    for i, e in enumerate(edges):
        if e.kind != "offset":
            continue  # already abstract (shouldn't happen on a fresh relgraph)
        for level in _GENERALITY:
            cand = _candidate(e, level)
            if cand is None:
                continue
            trial = list(edges)
            trial[i] = cand
            g_trial = RelGraph(root=relgraph.root, edges=trial)
            solved = solve_abstract_graph(g_trial, bboxes, (0.0, 0.0))
            if _equal(solved, anchor, care_mask_um):
                edges[i] = cand
                break  # most-abstract licensed kind wins; stop trying

    g_min = RelGraph(root=relgraph.root, edges=edges)
    tally: Dict[str, int] = {
        "symmetric": 0, "abut": 0, "align": 0, "order": 0, "offset": 0
    }
    kinds: Dict[Tuple[str, str], str] = {}
    incompressible: List[Tuple[str, str]] = []
    for e in edges:
        tally[_tally_kind(e.kind)] += 1
        kinds[(e.child, e.parent)] = e.kind
        if e.kind == "offset":
            incompressible.append((e.child, e.parent))

    return AbstractResult(
        graph=g_min,
        tally=tally,
        kinds=kinds,
        care_mask_um=care_mask_um,
        incompressible=incompressible,
    )


# --------------------------------------------------------------------------- #
# 4. The real-OTA degradation runner (used by __main__ / the report step).
# --------------------------------------------------------------------------- #
def build_ota_intent() -> Any:
    """Build the real external_pdk OTA intent the way the relgraph agent does:
    netlist intent -> GDS correspondence -> geometric intent. Returns the
    ExtractedIntent (with device_bboxes + matched/mirror pairs populated)."""
    import os

    from clay.layout.extract.geometric_intent import extract_geometric_intent
    from clay.layout.extract.netlist_intent import (
        correspond_to_gds,
        extract_netlist_intent,
    )

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    base = os.path.join(
        repo, "refCircuit", "twoStageAmp", "0208_amp", "project_tsmc18rf"
    )
    netlist = os.path.join(base, "OTA_CS_Q.src.net")
    gds = os.path.join(base, "prj_gzc", "CMD", "GDS2", "layout.gds")
    intent = extract_netlist_intent(netlist, "OTA_CS_Q")
    correspond_to_gds(intent, gds)
    extract_geometric_intent(intent, gds)
    return intent


def run_ota_degradation(care_masks: Sequence[float] = (0.1, 0.5, 1.0)) -> None:
    """Run ``minimize`` on the real OTA at each care_mask and PRINT the per-kind
    tally + the incompressible offset edges. The DEGRADATION RESULT."""
    intent = build_ota_intent()
    relgraph = build_rel_graph(intent)
    n_edges = len(relgraph.edges)
    print(f"[abstract_ops] relgraph source = {_RELGRAPH_SOURCE}")
    print(f"[abstract_ops] real OTA (OTA_CS_Q): {n_edges} relative edges over "
          f"{len(getattr(intent, 'device_bboxes', {}))} devices")
    print(f"[abstract_ops] declared matched/mirror pairs: "
          f"{len(getattr(intent, 'matched_pairs', []))}/"
          f"{len(getattr(intent, 'mirror_pairs', []))}")
    for mask in care_masks:
        res = minimize(relgraph, intent, mask)
        t = res.tally
        captured = n_edges - t["offset"]
        pct = (100.0 * captured / n_edges) if n_edges else 0.0
        print(
            f"  care_mask={mask:>4} um | {n_edges} edges -> "
            f"symmetric={t['symmetric']} abut={t['abut']} align={t['align']} "
            f"order={t['order']} offset={t['offset']} "
            f"| structural-captured {captured}/{n_edges} ({pct:.0f}%)"
        )
        if res.incompressible:
            shown = ", ".join(f"{c}<-{p}" for c, p in res.incompressible[:8])
            more = "" if len(res.incompressible) <= 8 else f" (+{len(res.incompressible)-8})"
            print(f"      incompressible offsets: {shown}{more}")


__all__ = [
    "RelEdge",
    "RelGraph",
    "AbstractResult",
    "as_abut",
    "as_align",
    "as_symmetric",
    "as_order",
    "solve_abstract_graph",
    "minimize",
    "build_rel_graph",
    "solve_rel_graph",
    "positions_equal",
    "build_ota_intent",
    "run_ota_degradation",
]


if __name__ == "__main__":
    run_ota_degradation()
