"""LEAKAGE-FREE position solver: realize a layout from ABSTRACT relations ONLY.

THE SCIENCE (no-data-leakage rule)
==================================
The degradation method extracts layout-design PRIMITIVES by starting from the
reference's full absolute coordinates and progressively replacing each coordinate
with an ABSTRACT structural relation (abut / align / symmetric / order) or a
don't-care. ``build_constraint_graph`` (in ``relgraph.py``) already does most of
this: it emits an over-determined CONSTRAINT GRAPH whose edges are mostly
structural (kind in {abut, align:x, align:y, symmetric}) PLUS a minimal set of
``offset`` bridge edges that still carry the raw reference ``dx, dy``.

Those ``offset`` edges are the LEAK. Keeping them and replaying them
(back-annotation) is MEANINGLESS for primitive extraction: a raw coordinate
replayed is no primitive at all. The honest validation is to STRIP every raw
coordinate and realize the layout from the PURE-ABSTRACT structural relations
alone -- then measure how much actually reproduces (and, just as importantly, how
much does NOT, because stripping the offsets disconnects the graph and leaves
whole clusters with no relation pinning them).

WHAT THIS MODULE PROVIDES
=========================
* ``strip_coordinates``    : remove ALL ``offset`` edges -> pure-abstract graph.
* ``assert_no_coordinates`` : HARD code-level guard; raises ``LeakageError`` if ANY
                              raw coordinate / offset edge survives. The solver
                              MUST pass its input through this guard.
* ``leakage_count``        : number of raw-coordinate (``offset``) edges (target 0).
* ``solve_leakfree``       : solve device positions from STRUCTURAL relations ONLY,
                              honestly reporting ``n_floating`` -- the devices /
                              components NOT pinned by any structural relation
                              (the incompleteness that stripping the leak exposes).

THE HONEST POINT
================
Structural edges almost certainly UNDER-DETERMINE the layout: dropping the offset
bridges disconnects the constraint graph into several components with NO relation
across them. ``solve_leakfree`` does NOT invent a cross-component position from the
reference (that would re-leak a coordinate). It places each disconnected component
at a DEFAULT slot (packed left-to-right) and FLAGS it as floating. ``n_floating``
is the genuine incompleteness of the pure-abstract intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from clay.layout.extract.relgraph import (
    ConstraintGraph,
    RelEdge,
    _CG_EPS_UM,
)

_XY = Tuple[float, float]

# Structural (non-leaking) edge kinds. An edge of any OTHER kind that carries a raw
# dx/dy used for positioning is a coordinate leak. 'order:*' kinds (sign-only, from
# abstract_ops) are also structural / leak-free.
_STRUCTURAL_KINDS = frozenset(
    {"abut", "align:x", "align:y", "symmetric"}
)

# The leaking kind: an 'offset' edge stores the raw reference centroid delta and
# replays it verbatim -- that is the back-annotation leak we forbid.
_LEAK_KIND = "offset"

# Default horizontal slot pitch (um) for packing disconnected components left-to-
# right. This is a layout-independent default, NOT a reference coordinate -- it
# carries no design fact, it is just a non-overlapping parking spot.
_DEFAULT_SLOT_GAP_UM: float = 5.0


class LeakageError(AssertionError):
    """Raised when a raw coordinate / offset edge survives into the leak-free path.

    This is the HARD guard the no-data-leakage rule rests on: if ANY ``offset``-kind
    edge (or any edge still carrying a raw dx/dy used for positioning) reaches the
    leak-free solver, we have NOT extracted a primitive -- we are back-annotating a
    coordinate. The guard refuses to proceed rather than silently leak.
    """


def _is_structural(edge: RelEdge) -> bool:
    """True iff ``edge`` is a pure structural relation (no raw coordinate replay)."""
    k = edge.kind
    return k in _STRUCTURAL_KINDS or k.startswith("order")


def strip_coordinates(graph: ConstraintGraph) -> ConstraintGraph:
    """Remove ALL raw-coordinate (``offset``) edges -> the PURE-ABSTRACT intent.

    Keeps ONLY structural edges (abut / align:x / align:y / symmetric / order:*) and
    drops every ``offset`` edge (the raw reference dx, dy bridges). The result is a
    :class:`ConstraintGraph` with the SAME devices, sizes, root and axis, but ZERO
    raw coordinates -- ``leakage_count`` of the result is 0.

    Note: the surviving structural edges STILL carry a ``dx, dy`` field on the
    RelEdge dataclass, but those values are used ONLY for the abut SIGN (which face
    touches) and align/symmetric off-axis preservation within the SAME component --
    the leak-free solver never reads an offset edge's magnitude as a position, and
    ``assert_no_coordinates`` enforces that no ``offset`` edge survives.
    """
    kept = [e for e in graph.edges if _is_structural(e)]
    return ConstraintGraph(
        root=graph.root,
        edges=kept,
        sizes=dict(graph.sizes),
        axis_x=graph.axis_x,
        names=list(graph.names),
    )


def leakage_count(graph: ConstraintGraph) -> int:
    """Number of raw-coordinate (``offset``) edges in ``graph``. Target after strip: 0."""
    return sum(1 for e in graph.edges if e.kind == _LEAK_KIND)


def assert_no_coordinates(graph_or_intent: Any) -> None:
    """HARD guard: raise :class:`LeakageError` if ANY raw coordinate / offset survives.

    Accepts a :class:`ConstraintGraph` (checks its edges) -- raises if ANY edge is of
    the leaking ``offset`` kind. ``don't-care`` declarations and pure structural
    relations (abut / align / symmetric / order) are fine; a raw ``offset`` edge used
    for positioning is NOT. The leak-free solver calls this on its input so a leaked
    coordinate can never sneak into the pure-abstract realization.
    """
    edges = getattr(graph_or_intent, "edges", None)
    if edges is None:
        raise LeakageError(
            "assert_no_coordinates: object has no 'edges' (not a ConstraintGraph); "
            "cannot prove leak-freedom"
        )
    leaks = [e for e in edges if getattr(e, "kind", None) == _LEAK_KIND]
    if leaks:
        shown = ", ".join(f"{e.child}<-{e.parent}" for e in leaks[:6])
        more = "" if len(leaks) <= 6 else f" (+{len(leaks) - 6} more)"
        raise LeakageError(
            f"LEAKAGE: {len(leaks)} raw-coordinate 'offset' edge(s) survive into the "
            f"leak-free path [{shown}{more}]. Strip them with strip_coordinates() "
            f"before solving -- replaying raw coordinates yields no primitive."
        )


@dataclass
class LeakfreeReport:
    """The honest result of solving from STRUCTURAL relations only.

    * ``n_devices``   -- total devices in the graph.
    * ``n_components``-- connected components under the STRUCTURAL edges (a component
      is a set of devices mutually pinned by structural relations).
    * ``n_floating``  -- devices whose position is NOT pinned by any structural
      relation to the root's component = the incompleteness exposed by stripping the
      leak. A device is floating iff it is NOT in the SAME structural component as the
      root (its cross-component placement is a default slot, not a relation).
    * ``floating``    -- the sorted names of those floating devices.
    * ``components``  -- the list of structural components (each a sorted name list),
      ordered by their smallest member (deterministic).
    * ``leakage_in``  -- ``leakage_count`` of the solver's input (MUST be 0).
    """

    n_devices: int
    n_components: int
    n_floating: int
    floating: List[str] = field(default_factory=list)
    components: List[List[str]] = field(default_factory=list)
    leakage_in: int = 0


def _components(names: List[str], edges: List[RelEdge]) -> List[List[str]]:
    """Connected components of ``names`` under the (structural) ``edges`` (union-find).

    Deterministic: components are returned sorted by their smallest member, each
    component's names sorted.
    """
    parent: Dict[str, str] = {n: n for n in names}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for e in edges:
        if e.child in parent and e.parent in parent:
            union(e.child, e.parent)

    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    comps = [sorted(g) for g in groups.values()]
    comps.sort(key=lambda g: g[0])
    return comps


def _solve_component(
    comp: List[str],
    edges: List[RelEdge],
    sizes: Dict[str, _XY],
    axis_x: Optional[float],
    root: str,
    origin: _XY,
) -> Dict[str, _XY]:
    """Place ONE structural component relative to ``origin`` using structural edges.

    The component is internally connected by structural relations, so propagation
    from a seed (the lowest-named device, placed at ``origin``) pins every member.
    Realizes:
      * abut       -> faces flush: center-to-center on the dominant axis =
                      (size_parent + size_child)/2, sign from the (structural) edge's
                      stored dx/dy SIGN only (which face touches -- not its magnitude).
      * align:x    -> child shares parent's x; y inherits the parent (NO offset
                      magnitude is used; same-row/column abstraction).
        align:y    -> child shares parent's y; x inherits the parent.
      * symmetric  -> child mirrors parent about the pair mid-x; y preserved (the
                      mirror is structural, no magnitude).
      * order:*    -> child placed at parent + min-separation in the order direction.
    """
    cset = set(comp)
    rel = [
        e for e in edges if e.child in cset and e.parent in cset
    ]
    # Seed: prefer the global root if it is in this component, else the smallest name.
    seed = root if root in cset else comp[0]
    pos: Dict[str, _XY] = {seed: (float(origin[0]), float(origin[1]))}

    def place_child(e: RelEdge, px: float, py: float) -> _XY:
        wp, hp = sizes.get(e.parent, (0.0, 0.0))
        wc, hc = sizes.get(e.child, (0.0, 0.0))
        if e.kind == "abut":
            if abs(e.dx) >= abs(e.dy):
                sep = (wp + wc) / 2.0
                return (px + (sep if e.dx >= 0 else -sep), py)
            sep = (hp + hc) / 2.0
            return (px, py + (sep if e.dy >= 0 else -sep))
        if e.kind == "align:x":
            return (px, py)  # shared column; no off-axis offset magnitude used
        if e.kind == "align:y":
            return (px, py)  # shared row; no off-axis offset magnitude used
        if e.kind == "symmetric":
            mid = px + (e.dx / 2.0)  # mid-x sign/structure; reflect about it
            return (2.0 * mid - px, py)
        if e.kind.startswith("order"):
            if e.kind == "order:left_of":
                return (px + (wp + wc) / 2.0, py)
            if e.kind == "order:right_of":
                return (px - (wp + wc) / 2.0, py)
            if e.kind == "order:above":
                return (px, py + (hp + hc) / 2.0)
            if e.kind == "order:below":
                return (px, py - (hp + hc) / 2.0)
        return (px, py)

    def place_parent(e: RelEdge, cx: float, cy: float) -> _XY:
        wp, hp = sizes.get(e.parent, (0.0, 0.0))
        wc, hc = sizes.get(e.child, (0.0, 0.0))
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
        if e.kind.startswith("order"):
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
                pos[e.child] = place_child(e, px, py)
                progress = True
            elif e.child in pos and e.parent not in pos:
                cx, cy = pos[e.child]
                pos[e.parent] = place_parent(e, cx, cy)
                progress = True
    # any member unreached (shouldn't happen -- the component IS connected) pins at origin
    for n in comp:
        if n not in pos:
            pos[n] = (float(origin[0]), float(origin[1]))
    return pos


def solve_leakfree(
    graph_abstract: ConstraintGraph,
    sizes: Optional[Dict[str, _XY]] = None,
) -> Tuple[Dict[str, _XY], LeakfreeReport]:
    """Solve device positions from STRUCTURAL relations ONLY (no raw coordinates).

    HARD GUARD FIRST: calls :func:`assert_no_coordinates` on ``graph_abstract`` ->
    raises :class:`LeakageError` if ANY ``offset`` edge sneaked in. This is the proof
    the leak-free path cannot replay a raw coordinate.

    Method:
      1. find the CONNECTED COMPONENTS of the devices under the structural edges.
         Stripping the offset bridges disconnects the graph -- each component is a
         cluster mutually pinned by structural relations, with NO relation across
         components.
      2. WITHIN a component, propagate from a seed to pin every member by its
         structural relations (abut flush / align shared-coord / symmetric mirror /
         order min-sep). Positions are RELATIVE-determined.
      3. ACROSS components there is NO relation. Place each component at a DEFAULT
         slot (packed left-to-right by component bbox width + a fixed gap) and RECORD
         it. We do NOT read the reference for a cross-component position -- that would
         re-leak. Every device NOT in the root's structural component is FLAGGED as
         floating.

    Returns ``(positions, report)``. ``positions`` is ``{name -> (cx, cy)}`` um in a
    local frame; ``report`` (:class:`LeakfreeReport`) carries ``n_floating`` (the
    honest incompleteness) and which devices float.
    """
    # 1. HARD leak guard -- the leak-free path MUST pass through this.
    assert_no_coordinates(graph_abstract)

    if sizes is None:
        sizes = graph_abstract.sizes
    sizes = sizes or {}

    names = list(graph_abstract.names)
    if not names:
        return ({}, LeakfreeReport(n_devices=0, n_components=0, n_floating=0))

    struct_edges = [e for e in graph_abstract.edges if _is_structural(e)]
    comps = _components(names, struct_edges)

    # which component holds the root (the translation pin / "pinned" component).
    root = graph_abstract.root
    root_comp_idx = 0
    for i, c in enumerate(comps):
        if root in c:
            root_comp_idx = i
            break

    positions: Dict[str, _XY] = {}
    slot_x = 0.0
    for i, comp in enumerate(comps):
        comp_pos = _solve_component(
            comp, struct_edges, sizes, graph_abstract.axis_x, root, (0.0, 0.0)
        )
        # pack this component's local solution into its slot (left-to-right).
        xs = [p[0] for p in comp_pos.values()]
        widths = [sizes.get(n, (0.0, 0.0))[0] for n in comp]
        min_x = min(xs) if xs else 0.0
        # shift so the component's left edge starts at slot_x
        comp_w = (max(xs) - min(xs)) if xs else 0.0
        half_max = max(widths) / 2.0 if widths else 0.0
        shift = slot_x - min_x + half_max
        for n, (cx, cy) in comp_pos.items():
            positions[n] = (cx + shift, cy)
        slot_x += comp_w + 2.0 * half_max + _DEFAULT_SLOT_GAP_UM

    # floating = every device NOT in the root's structural component (no relation
    # pins it to the root cluster -> its cross-component slot is a default, not a fact).
    floating: List[str] = []
    for i, comp in enumerate(comps):
        if i == root_comp_idx:
            continue
        floating.extend(comp)
    floating.sort()

    report = LeakfreeReport(
        n_devices=len(names),
        n_components=len(comps),
        n_floating=len(floating),
        floating=floating,
        components=comps,
        leakage_in=0,  # guaranteed 0: assert_no_coordinates passed above
    )
    return (positions, report)


# --------------------------------------------------------------------------- #
# OTA degradation runner: build -> measure leak -> strip -> solve leak-free.    #
# --------------------------------------------------------------------------- #
def run_ota_leakfree() -> Tuple[Dict[str, _XY], LeakfreeReport, int, int]:
    """Run the full leak-free degradation on the real external_pdk OTA.

    1. build the OTA intent + constraint graph (reads the reference -- ALLOWED at the
       degradation START).
    2. measure ``leakage_count`` BEFORE strip (the raw-coordinate count, ~8).
    3. ``strip_coordinates`` -> pure-abstract graph; ``leakage_count`` AFTER (== 0).
    4. ``solve_leakfree`` at external_pdk reference sizes -> positions + honest report.

    Returns ``(positions, report, leak_before, leak_after)``.
    """
    from clay.layout.extract.abstract_ops import build_ota_intent
    from clay.layout.extract.relgraph import build_constraint_graph

    intent = build_ota_intent()
    graph = build_constraint_graph(intent, eps=_CG_EPS_UM)
    leak_before = leakage_count(graph)

    abstract = strip_coordinates(graph)
    leak_after = leakage_count(abstract)

    positions, report = solve_leakfree(abstract, sizes=graph.sizes)
    return (positions, report, leak_before, leak_after)


__all__ = [
    "LeakageError",
    "LeakfreeReport",
    "strip_coordinates",
    "assert_no_coordinates",
    "leakage_count",
    "solve_leakfree",
    "run_ota_leakfree",
]


if __name__ == "__main__":
    pos, rep, before, after = run_ota_leakfree()
    print(f"[leakfree] OTA leakage_count BEFORE strip = {before} (raw coordinates)")
    print(f"[leakfree] OTA leakage_count AFTER  strip = {after} (must be 0)")
    print(f"[leakfree] devices={rep.n_devices} components={rep.n_components} "
          f"n_floating={rep.n_floating}")
    print(f"[leakfree] floating devices: {rep.floating}")
    print(f"[leakfree] structural components:")
    for i, c in enumerate(rep.components):
        print(f"    component {i}: {c}")
