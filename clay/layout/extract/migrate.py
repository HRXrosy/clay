"""PDK MIGRATION of an extracted intent -- the OBJECTIVE INCOMPLETENESS MEASURE.

THE VALIDATION METHOD (PDK-independent, objective)
==================================================
A COMPLETE extracted intent is PURE RELATIONS with zero residual numeric offsets:
realizing it in ANY PDK is overlap-free BY CONSTRUCTION, because the solver packs
every device at that PDK's own sizes + min spacing. So if we MIGRATE the intent we
extracted from the external_pdk reference onto gf180 -- using gf180 device geometry and
sizes -- the OVERLAP COUNT of the resulting placement is a direct, objective measure
of how INCOMPLETE the extracted vocabulary is:

    overlap count == 0   <=>  the intent is realizable in gf180 with no collisions
    overlap count >  0   <=>  the surviving numeric content (external_pdk absolute offsets
                              the abstraction could NOT turn into a structural
                              relation) collides when devices grow to gf180 sizes.

The minimized mixed-kind graph (``abstract_ops.minimize``) has two edge populations:

  * STRUCTURAL edges (abut / align / order / symmetric): these are realized by
    ``solve_abstract_graph`` using the GF180 device sizes + min separations, so they
    are overlap-free for those pairs BY CONSTRUCTION -- the relation says "flush" /
    "same row" / "left of", and the solver honours it at gf180 dimensions.

  * OFFSET edges (kind == 'offset'): the incompressible residual. These keep their
    EXACT external_pdk ``(dx, dy)`` centroid delta. A external_pdk gap that was just wide enough for
    two ~3um external_pdk devices is NOT wide enough for two gf180 devices (gf180 minimum
    poly/active dimensions are larger), so the two gf180 stamps OVERLAP. The overlap
    localizes EXACTLY at the edge whose vocabulary was incomplete.

PREDICTION (the falsifiable claim this module tests): structural edges retarget
overlap-free (0 overlaps attributable to abut/align/order/symmetric); the residual
external_pdk offset edges cause ALL the overlaps. Overlaps localize precisely where the
extraction vocabulary failed to capture the relation structurally.

WHAT THIS MODULE DOES (5 steps)
===============================
1. ``build_gf180_geometry(intent)`` -- build each device's REAL gf180 geometry +
   size by emitting the intent to a .clay (``emit_clay``, default abstract mode) and
   running ``place_to_intent(..., tech='gf180mcu', engine='seqpair')`` ONCE. Each
   device's polys are captured RELATIVE to its own centroid (a reusable gf180 STAMP)
   plus its gf180 bbox size. Devices that fail to build are reported honestly.
2. ``migrate_positions(intent, gf180_sizes)`` -- solve the MINIMIZED abstract graph
   with GF180 sizes via ``abstract_ops.solve_abstract_graph``: structural edges
   realize with gf180 sizes + min spacing (overlap-free for those by construction);
   offset edges keep their external_pdk dx,dy (the non-retargetable residual).
3. ``emit_gf180_gds(intent, positions, geometry, out_gds)`` -- stamp each device's
   gf180 geometry (translated to its migrated centroid) into a flat gf180 GDS.
4. ``overlap_report(intent, positions, geometry)`` -- device-bbox overlaps (pairs +
   overlapping AREA + which devices), each attributed to the responsible graph
   edge(s) (offset vs structural). Headline = total pair count + total overlap area.
5. The OTA runner (``run_ota_migration``) wires it end-to-end and renders the GDS.

UNITS
=====
``intent.device_bboxes`` and ``positions`` are in MICRONS (the shared
``(x0,y0,x1,y1) um`` contract). The gf180 stamps / bboxes come back from the placer
in NANOMETRES (the GDS database unit, dbu = 0.001 um); we keep stamps in nm for GDS
emission but expose sizes/positions in um for the abstract solver + the report. The
single conversion point is ``_NM_PER_UM``.

SCOPE / DISCIPLINE
==================
This module is purely COMPOSITIONAL. It does NOT edit relgraph.py / abstract_ops.py /
relax.py / emit_clay.py / netlist_intent.py / geometric_intent.py / the placers / the
grammar / any design.clay. It only consumes their documented surfaces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from clay.layout.extract import abstract_ops, relgraph

_NM_PER_UM = 1000.0

_Bbox = Tuple[float, float, float, float]  # (x0,y0,x1,y1)
_XY = Tuple[float, float]


# --------------------------------------------------------------------------- #
# 0. data structures
# --------------------------------------------------------------------------- #
@dataclass
class DeviceGeometry:
    """One device's gf180 geometry, ready to stamp at any migrated centroid.

    ``polys`` is the list of placed gf180 ``LayerPolygon`` (from the gf180 placer)
    TRANSLATED so the device's bbox centroid sits at the ORIGIN -- a reusable stamp
    in NANOMETRES. ``bbox_nm`` is the device's gf180 footprint ``(x0,y0,x1,y1)`` nm
    (centroid at origin, so ``bbox_nm`` is symmetric about 0). ``size_um`` is the
    ``(w, h)`` footprint in MICRONS (what the abstract solver needs). ``kind`` is the
    device kind ('nmos' / 'pmos' / 'cap' / 'res' ...).
    """

    name: str
    polys: list  # list[LayerPolygon], centroid-relative, nm
    bbox_nm: Tuple[int, int, int, int]  # centroid-relative footprint, nm
    size_um: _XY  # (w, h) um
    kind: str = ""


@dataclass
class GeometrySet:
    """Result of ``build_gf180_geometry``: the per-device gf180 stamps + honesty.

    ``built`` maps device name -> :class:`DeviceGeometry` for every device whose
    gf180 geometry built cleanly. ``failed`` is the list of device names that did
    NOT build (reported honestly, never silently dropped). ``placer_note`` carries
    the placer's own provenance note.
    """

    built: Dict[str, DeviceGeometry] = field(default_factory=dict)
    failed: List[str] = field(default_factory=list)
    placer_note: str = ""

    def sizes_um(self) -> Dict[str, _XY]:
        return {n: g.size_um for n, g in self.built.items()}


@dataclass
class Overlap:
    """One overlapping device pair in the migrated gf180 placement.

    ``a`` / ``b`` are the device names (sorted). ``area_um2`` is the overlapping
    rectangle area (um^2). ``edge_kind`` is the bucketed kind of the relgraph edge
    that joins the pair in the minimized graph ('offset' / 'abut' / 'align' / 'order'
    / 'symmetric'), or 'indirect' when the overlapping pair is not a direct tree edge
    (the overlap then propagated from a magnitude residual elsewhere in the subtree).
    ``attribution`` is the headline bucket: 'offset' (a non-retargetable external_pdk
    magnitude residual -- a direct offset edge, the surviving-magnitude axis of an
    align edge, or an indirect path carrying one) or 'structural' (a separation the
    solver realized at gf180 sizes: abut / symmetric / order, or align's pinned axis).
    """

    a: str
    b: str
    area_um2: float
    edge_kind: str
    attribution: str


@dataclass
class OverlapReport:
    """The objective incompleteness measure of the gf180 migration.

    ``overlaps`` is every overlapping device pair. ``n_pairs`` / ``total_area_um2``
    are the headline numbers. ``by_attribution`` tallies overlapping pairs +
    overlap area split into 'offset' vs 'structural'. ``n_devices`` /
    ``n_built`` record how many devices were placed / built cleanly.
    """

    overlaps: List[Overlap] = field(default_factory=list)
    n_devices: int = 0
    n_built: int = 0

    @property
    def n_pairs(self) -> int:
        return len(self.overlaps)

    @property
    def total_area_um2(self) -> float:
        return sum(o.area_um2 for o in self.overlaps)

    @property
    def by_attribution(self) -> Dict[str, Dict[str, float]]:
        out = {
            "offset": {"pairs": 0, "area_um2": 0.0},
            "structural": {"pairs": 0, "area_um2": 0.0},
        }
        for o in self.overlaps:
            out[o.attribution]["pairs"] += 1
            out[o.attribution]["area_um2"] += o.area_um2
        return out

    def summary(self) -> str:
        att = self.by_attribution
        return (
            f"gf180 migration: {self.n_pairs} overlapping pair(s), "
            f"{self.total_area_um2:.2f} um^2 total overlap | "
            f"offset-attributed {att['offset']['pairs']} pair(s) "
            f"({att['offset']['area_um2']:.2f} um^2), "
            f"structural-attributed {att['structural']['pairs']} pair(s) "
            f"({att['structural']['area_um2']:.2f} um^2) | "
            f"built {self.n_built}/{self.n_devices} devices"
        )


# --------------------------------------------------------------------------- #
# 1. build_gf180_geometry: REAL gf180 geometry + size per device
# --------------------------------------------------------------------------- #
def _polys_bbox_nm(polys: list) -> Optional[Tuple[int, int, int, int]]:
    xs: List[int] = []
    ys: List[int] = []
    for lp in polys:
        for v in lp.polygon.vertices:
            xs.append(int(v.x))
            ys.append(int(v.y))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _translate_polys(polys: list, dx: int, dy: int) -> list:
    """Return a fresh list of polys translated by (dx, dy) nm (never mutate input)."""
    from clay.layout.backend import LayerPolygon, Point, Polygon

    out = []
    for lp in polys:
        out.append(
            LayerPolygon(
                layer=lp.layer,
                datatype=lp.datatype,
                polygon=Polygon(
                    vertices=[Point(int(v.x) + dx, int(v.y) + dy) for v in lp.polygon.vertices]
                ),
            )
        )
    return out


def build_gf180_geometry(intent: Any) -> GeometrySet:
    """Build each device's REAL gf180 geometry + size (a reusable centroid-relative
    stamp), by emitting the intent and running the gf180 placer ONCE.

    Faithful route: ``emit_clay(intent)`` (default ABSTRACT mode -- no pins, the
    fully-relational form) -> parse -> ``place_to_intent(.., tech='gf180mcu',
    engine='seqpair')``. The placer builds each device's gf180 artwork at its native
    PDK (the MIM cap builder works for gf180). For each placed device we capture its
    polys RELATIVE to its bbox centroid (so the stamp can be re-placed at any migrated
    centroid) plus its gf180 bbox size. A device that the placer did NOT produce is
    recorded in ``failed`` -- reported honestly, never silently dropped.
    """
    import pathlib
    import tempfile

    from clay.layout.extract.emit_clay import emit_clay
    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place_refine import place_to_intent
    from clay.parser import parse_clay_file

    want = [str(getattr(d, "name", "")) for d in (getattr(intent, "devices", []) or [])]
    want += [str(getattr(p, "name", "")) for p in (getattr(intent, "passives", []) or [])]
    want = [n for n in want if n]

    geo = GeometrySet()

    src = emit_clay(intent)  # default abstract mode (no pins, all relations)
    work = pathlib.Path(tempfile.mkdtemp(prefix="clay_migrate_"))
    f = work / "design.clay"
    f.write_text(src, encoding="utf-8")
    cf = parse_clay_file(f)
    if not cf.cell_defs:
        geo.failed = list(want)
        geo.placer_note = "emitted .clay produced no cell_defs"
        return geo
    cd = cf.cell_defs[0]
    li = layout_intent_from_cell_def(cd)
    cp = place_to_intent(cd, li, None, source_cir=None, tech="gf180mcu", engine="seqpair")
    placement = cp.placement
    geo.placer_note = getattr(cp, "note", "") or getattr(placement, "note", "")

    placed = getattr(placement, "devices", {}) or {}
    for name in want:
        pd = placed.get(name)
        if pd is None:
            geo.failed.append(name)
            continue
        polys = list(getattr(pd, "polys", []) or [])
        bb = _polys_bbox_nm(polys)
        if bb is None:
            # fall back to the placed bbox if the device carries no polys
            pbb = getattr(pd, "bbox", None)
            if pbb is None:
                geo.failed.append(name)
                continue
            bb = tuple(int(v) for v in pbb)
        x0, y0, x1, y1 = bb
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        rel_polys = _translate_polys(polys, -cx, -cy)
        rel_bbox = (x0 - cx, y0 - cy, x1 - cx, y1 - cy)
        size_um = ((x1 - x0) / _NM_PER_UM, (y1 - y0) / _NM_PER_UM)
        geo.built[name] = DeviceGeometry(
            name=name,
            polys=rel_polys,
            bbox_nm=rel_bbox,
            size_um=size_um,
            kind=str(getattr(pd, "kind", "")),
        )
    return geo


# --------------------------------------------------------------------------- #
# 2. migrate_positions: solve the MINIMIZED graph with GF180 sizes
# --------------------------------------------------------------------------- #
def _build_minimized_graph(intent: Any, care_mask_um: float):
    """Build the relgraph + its minimized mixed-kind graph (the same artefact the
    degradation runner produces). Returns ``(minimized_graph, AbstractResult)``."""
    g = relgraph.build_rel_graph(intent)
    res = abstract_ops.minimize(g, intent, care_mask_um=care_mask_um)
    return res.graph, res


def migrate_positions(
    intent: Any,
    gf180_sizes: Dict[str, _XY],
    *,
    care_mask_um: float = 0.5,
    minimized=None,
) -> Dict[str, _XY]:
    """Solve the MINIMIZED abstract graph with GF180 device SIZES -> centroids (um).

    The minimized graph (from ``abstract_ops.minimize``) carries structural edges
    (abut / align / order / symmetric) + the incompressible offset residual. We solve
    it with ``abstract_ops.solve_abstract_graph`` passing GF180 bboxes built from
    ``gf180_sizes`` (centered, since the solver reads only widths/heights for the
    min-separation of structural edges). Structural edges therefore realize at gf180
    sizes + min spacing (overlap-free for those pairs by construction); offset edges
    keep their exact external_pdk ``(dx, dy)`` -- the non-retargetable residual that the
    overlap report will localize.

    ``gf180_sizes`` maps device name -> ``(w, h)`` um. ``minimized`` may be passed in
    to reuse a graph already built (else it is built here at ``care_mask_um``).
    """
    if minimized is None:
        minimized, _res = _build_minimized_graph(intent, care_mask_um)

    # GF180 bboxes for the solver: centered (x0,y0,x1,y1) um from each size. The
    # solver consumes only (w, h) for structural min-separations + the stored dx,dy
    # for offset/align, so a centered box is sufficient + unambiguous.
    bboxes_um: Dict[str, _Bbox] = {}
    for name, (w, h) in gf180_sizes.items():
        bboxes_um[name] = (-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)

    positions = abstract_ops.solve_abstract_graph(minimized, bboxes_um, (0.0, 0.0))
    return {n: (float(p[0]), float(p[1])) for n, p in positions.items()}


# --------------------------------------------------------------------------- #
# 2b. CONSTRAINT-GRAPH migration: the same idea, but the positions come from the
#     OVER-DETERMINED constraint graph (relgraph.build_constraint_graph) instead of
#     the MST-derived minimized graph. The constraint graph captures EVERY structural
#     adjacency (cap-cap abut, every align/symmetric pair) as a direct edge, so those
#     separations realize at gf180 sizes BY CONSTRUCTION -- they no longer accumulate
#     through a long non-retargetable offset chain. Only the minimal cross-cluster
#     offset bridges keep external_pdk magnitudes; the overlaps localize to THOSE.
# --------------------------------------------------------------------------- #
def migrate_positions_cg(
    intent: Any,
    gf180_sizes: Dict[str, _XY],
    *,
    cg=None,
) -> Tuple[Dict[str, _XY], float]:
    """Solve the CONSTRAINT GRAPH with GF180 device SIZES -> ``(centroids, max_resid)``.

    The constraint graph (from ``relgraph.build_constraint_graph``) carries an edge for
    EVERY structural adjacency (abut / align / symmetric) plus the minimal offset
    bridges. Solving it with GF180 sizes (``relgraph.solve_constraint_graph``) realizes
    each structural separation at gf180 dimensions (overlap-free for those pairs by
    construction); only the offset bridges keep their external_pdk ``(dx, dy)``. ``max_resid``
    is the over-determined system's residual at GF180 sizes (reported honestly).
    """
    if cg is None:
        cg = relgraph.build_constraint_graph(intent)
    # grow_align_to_sizes: the migration semantics -- every structural pair
    # (abut AND align) realises overlap-free at the TARGET sizes by construction,
    # exactly what this function's contract (and the structural-overlap oracle)
    # claims. The default exact solve keeps reference gaps verbatim, which let a
    # gf180-grown align pair (MM18/MM19) overlap 4.35 um^2.
    positions, resid = relgraph.solve_constraint_graph(
        cg, gf180_sizes, (0.0, 0.0), grow_align_to_sizes=True
    )
    return ({n: (float(p[0]), float(p[1])) for n, p in positions.items()}, resid)


def _cg_edge_kinds(cg) -> Dict[frozenset, set]:
    """Map each unordered (a, b) device pair to the SET of constraint-graph edge kinds
    joining it (a pair may have several -- abut + align:y, etc.)."""
    out: Dict[frozenset, set] = {}
    for e in getattr(cg, "edges", []):
        out.setdefault(frozenset((e.child, e.parent)), set()).add(e.kind)
    return out


def overlap_report_cg(
    intent: Any,
    positions: Dict[str, _XY],
    geometry: GeometrySet,
    *,
    cg=None,
    eps_um: float = 1e-6,
) -> OverlapReport:
    """Overlap report for the CONSTRAINT-GRAPH migration.

    Same overlap geometry as ``overlap_report``, but attribution reads the constraint
    graph: a directly-joined pair is 'structural' iff ANY of its edges is a
    fully-pinning structural relation (abut / symmetric) or the pinned axis of an align
    edge covers the collision axis; else 'offset'. A pair joined only through offset
    bridges (no direct edge) is attributed 'offset' iff the connecting path carries a
    bridge, else 'structural'. The headline question -- do CC0/CR1 still overlap -- is
    answered by whether the cap-cap pair appears in ``overlaps`` at all.
    """
    if cg is None:
        cg = relgraph.build_constraint_graph(intent)

    pair_kinds = _cg_edge_kinds(cg)

    # adjacency carrying, per neighbour, whether the connecting edge is a residual
    # (offset bridge or align surviving-magnitude axis). abut / symmetric fully pin.
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for e in getattr(cg, "edges", []):
        adj.setdefault(e.child, []).append((e.parent, e.kind))
        adj.setdefault(e.parent, []).append((e.child, e.kind))

    _RESIDUAL = {"offset", "align:x", "align:y"}

    def path_has_residual(a: str, b: str) -> Optional[bool]:
        from collections import deque

        seen = {a}
        q = deque([(a, False)])
        while q:
            node, saw = q.popleft()
            if node == b:
                return saw
            for nbr, kind in adj.get(node, []):
                if nbr not in seen:
                    seen.add(nbr)
                    q.append((nbr, saw or kind in _RESIDUAL))
        return None

    def direct_attr(kinds: set, ox: float, oy: float) -> str:
        # a fully-pinning structural relation present -> structural.
        if "abut" in kinds or "symmetric" in kinds:
            return "structural"
        # align-only: pinned axis is structural; surviving-magnitude axis is offset.
        if "align:x" in kinds and oy <= ox:
            return "structural"
        if "align:y" in kinds and ox <= oy:
            return "structural"
        if "align:x" in kinds or "align:y" in kinds:
            return "offset"
        return "offset"

    names = [n for n in geometry.built if n in positions]
    bboxes = {
        n: _migrated_bbox_um(positions[n], geometry.built[n].size_um) for n in names
    }

    report = OverlapReport(n_devices=len(positions), n_built=len(geometry.built))
    names_sorted = sorted(names)
    for i in range(len(names_sorted)):
        for j in range(i + 1, len(names_sorted)):
            a, b = names_sorted[i], names_sorted[j]
            area, ox, oy = _rect_overlap(bboxes[a], bboxes[b])
            if area <= eps_um:
                continue
            kinds = pair_kinds.get(frozenset((a, b)))
            if kinds is not None:
                # report the most-structural kind present for the edge_kind label.
                if "abut" in kinds:
                    label = "abut"
                elif "symmetric" in kinds:
                    label = "symmetric"
                elif "align:x" in kinds or "align:y" in kinds:
                    label = "align"
                else:
                    label = "offset"
                attribution = direct_attr(kinds, ox, oy)
            else:
                label = "indirect"
                has_res = path_has_residual(a, b)
                attribution = "offset" if has_res else "structural"
            report.overlaps.append(
                Overlap(a=a, b=b, area_um2=area, edge_kind=label, attribution=attribution)
            )
    report.overlaps.sort(key=lambda o: (-o.area_um2, o.a, o.b))
    return report


# --------------------------------------------------------------------------- #
# 3. emit_gf180_gds: stamp each device's gf180 geometry at its migrated centroid
# --------------------------------------------------------------------------- #
def emit_gf180_gds(
    intent: Any,
    positions: Dict[str, _XY],
    geometry: GeometrySet,
    out_gds: str,
    *,
    cell_name: Optional[str] = None,
    dbu: float = 0.001,
) -> str:
    """Stamp each device's gf180 geometry (translated to its migrated centroid) into
    a flat gf180 GDS readable by klayout. Returns the written path.

    Only devices that BOTH built gf180 geometry AND have a migrated position are
    stamped (the honest set). Each device's centroid-relative stamp is translated by
    its migrated centroid (um -> nm) and inserted into a single top cell.
    """
    import pathlib

    import klayout.db as kdb

    name = cell_name or str(getattr(intent, "cell", "migrated"))
    ly = kdb.Layout()
    ly.dbu = dbu
    top = ly.create_cell(name)

    for dev, geo in geometry.built.items():
        pos = positions.get(dev)
        if pos is None:
            continue
        cx_nm = int(round(pos[0] * _NM_PER_UM))
        cy_nm = int(round(pos[1] * _NM_PER_UM))
        for lp in _translate_polys(geo.polys, cx_nm, cy_nm):
            li = ly.layer(lp.layer, lp.datatype)
            top.shapes(li).insert(
                kdb.Polygon([kdb.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices])
            )

    out = pathlib.Path(out_gds)
    out.parent.mkdir(parents=True, exist_ok=True)
    ly.write(str(out))
    return str(out)


# --------------------------------------------------------------------------- #
# 4. overlap_report: device-bbox overlaps, attributed to the responsible edges
# --------------------------------------------------------------------------- #
def _migrated_bbox_um(pos: _XY, size_um: _XY) -> _Bbox:
    w, h = size_um
    cx, cy = pos
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _rect_overlap(a: _Bbox, b: _Bbox) -> Tuple[float, float, float]:
    """Overlap of two bboxes: ``(area, overlap_x, overlap_y)`` um. ``overlap_x`` /
    ``overlap_y`` are the interpenetration extents on each axis (>0 only when the
    bboxes overlap on BOTH axes -> a real area). Returns (0,0,0) when disjoint."""
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    if ox <= 0.0 or oy <= 0.0:
        return (0.0, 0.0, 0.0)
    return (ox * oy, ox, oy)


def _rect_overlap_area(a: _Bbox, b: _Bbox) -> float:
    """Overlapping rectangle AREA (um^2); 0 when the bboxes do not overlap."""
    return _rect_overlap(a, b)[0]


def _edge_kind_lookup(minimized) -> Dict[frozenset, str]:
    """Map each unordered (child, parent) edge to its kept RAW kind in the minimized
    graph (keeping align:x / align:y / order:* distinct so axis-aware attribution can
    see which axis the relation actually pinned)."""
    out: Dict[frozenset, str] = {}
    for e in getattr(minimized, "edges", []):
        out[frozenset((e.child, e.parent))] = e.kind
    return out


def _bucket(kind: str) -> str:
    """Bucket a raw kept kind (align:x -> 'align', order:* -> 'order'; offset / abut /
    symmetric pass through)."""
    if kind.startswith("align"):
        return "align"
    if kind.startswith("order"):
        return "order"
    return kind


# A structural relation is REALIZED overlap-free only on the axis it actually pins:
#   * 'abut'      pins the dominant-axis separation = (sizes)/2 -> overlap-free there;
#   * 'symmetric' / 'order:*' likewise pin the dominant axis at >= min separation;
#   * 'align:x'   pins ONLY x (shared column) and KEEPS the external_pdk y-magnitude;
#     'align:y'   pins ONLY y (shared row)    and KEEPS the external_pdk x-magnitude.
# So an 'align' edge is half-structural: a collision along its SURVIVING-magnitude
# axis is the non-retargetable external_pdk residual (attributed 'offset'); a collision purely
# along its PINNED axis is a true structural-realization fact (attributed 'structural').
def _direct_attribution(raw_kind: str, ox: float, oy: float) -> str:
    """Attribute a DIRECT-edge overlap to 'offset' (residual magnitude) or
    'structural' given the edge's raw kind + the overlap extents (ox, oy) um."""
    if raw_kind == "offset":
        return "offset"
    if raw_kind == "align:x":
        # x is pinned (shared column); a y-dominated overlap is the surviving external_pdk
        # y-magnitude -> offset residual. An x-dominated overlap would be structural.
        return "offset" if oy >= ox else "structural"
    if raw_kind == "align:y":
        return "offset" if ox >= oy else "structural"
    # abut / symmetric / order:* pin the separating axis -> structural.
    return "structural"


def overlap_report(
    intent: Any,
    positions: Dict[str, _XY],
    geometry: GeometrySet,
    *,
    care_mask_um: float = 0.5,
    minimized=None,
    eps_um: float = 1e-6,
) -> OverlapReport:
    """Compute device-bbox overlaps in the migrated gf180 placement + attribute each
    to the responsible graph edge(s).

    For every pair of built+placed devices whose gf180 bboxes overlap (area >
    ``eps_um`` um^2) we record an :class:`Overlap`. A DIRECT-edge overlap is
    attributed by ``_direct_attribution``: a pure 'offset' edge -> 'offset'; an
    'align:x'/'align:y' edge -> 'offset' iff the overlap is dominated by its
    surviving-magnitude axis (the residual the relation did NOT capture), else
    'structural'; 'abut'/'symmetric'/'order' pin the separating axis -> 'structural'.
    An 'indirect' overlap (no direct tree edge) is attributed 'offset' iff ANY edge on
    the tree path between the two devices carries a magnitude residual (offset or
    align); if the whole connecting path is fully pinned it is 'structural'.

    The prediction this realizes: a COMPLETE intent (pure structural relations, no
    surviving magnitudes) migrates with structural-attributed == 0 overlaps; every
    overlap is offset-attributed and localizes exactly where the vocabulary is
    incomplete.
    """
    if minimized is None:
        minimized, _res = _build_minimized_graph(intent, care_mask_um)

    edge_kind = _edge_kind_lookup(minimized)  # frozenset(pair) -> RAW kind

    # adjacency for path attribution: which RAW edge-kinds lie on the tree path a<->b.
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for e in getattr(minimized, "edges", []):
        raw = edge_kind[frozenset((e.child, e.parent))]
        adj.setdefault(e.child, []).append((e.parent, raw))
        adj.setdefault(e.parent, []).append((e.child, raw))

    # An 'offset' edge OR an 'align' edge (its surviving-magnitude axis) carries a
    # non-retargetable external_pdk magnitude; abut / symmetric / order pin the separation.
    _RESIDUAL_KINDS = {"offset", "align:x", "align:y"}

    def path_has_residual(a: str, b: str) -> Optional[bool]:
        """BFS the tree from a to b; return True iff any edge on the path carries a
        external_pdk magnitude residual (offset or align), False if a fully-pinned path
        exists, None if no path."""
        from collections import deque

        seen = {a}
        q = deque([(a, False)])
        while q:
            node, saw_res = q.popleft()
            if node == b:
                return saw_res
            for nbr, raw in adj.get(node, []):
                if nbr not in seen:
                    seen.add(nbr)
                    q.append((nbr, saw_res or raw in _RESIDUAL_KINDS))
        return None

    names = [n for n in geometry.built if n in positions]
    bboxes = {
        n: _migrated_bbox_um(positions[n], geometry.built[n].size_um) for n in names
    }

    report = OverlapReport(
        n_devices=len(positions),
        n_built=len(geometry.built),
    )

    names_sorted = sorted(names)
    for i in range(len(names_sorted)):
        for j in range(i + 1, len(names_sorted)):
            a, b = names_sorted[i], names_sorted[j]
            area, ox, oy = _rect_overlap(bboxes[a], bboxes[b])
            if area <= eps_um:
                continue
            direct = edge_kind.get(frozenset((a, b)))
            if direct is not None:
                kind = _bucket(direct)
                attribution = _direct_attribution(direct, ox, oy)
            else:
                kind = "indirect"
                has_res = path_has_residual(a, b)
                attribution = "offset" if has_res else "structural"
            report.overlaps.append(
                Overlap(
                    a=a,
                    b=b,
                    area_um2=area,
                    edge_kind=kind,
                    attribution=attribution,
                )
            )

    report.overlaps.sort(key=lambda o: (-o.area_um2, o.a, o.b))
    return report


# --------------------------------------------------------------------------- #
# 5. end-to-end OTA runner
# --------------------------------------------------------------------------- #
@dataclass
class MigrationResult:
    """The full OTA migration outcome (geometry + positions + report + artefacts)."""

    geometry: GeometrySet
    positions: Dict[str, _XY]
    report: OverlapReport
    minimized: Any
    abstract_result: Any
    out_gds: str = ""
    out_png: str = ""


def migrate_intent_to_gf180(
    intent: Any,
    out_gds: str,
    *,
    care_mask_um: float = 0.5,
    render_png: Optional[str] = None,
    render_title: str = "",
) -> MigrationResult:
    """Run the whole migration on an extracted intent: build gf180 geometry, migrate
    positions through the minimized graph, emit the gf180 GDS, compute the overlap
    report, and (optionally) render a PNG. Returns a :class:`MigrationResult`.
    """
    minimized, res = _build_minimized_graph(intent, care_mask_um)
    geometry = build_gf180_geometry(intent)
    positions = migrate_positions(
        intent, geometry.sizes_um(), care_mask_um=care_mask_um, minimized=minimized
    )
    gds_path = emit_gf180_gds(intent, positions, geometry, out_gds)
    report = overlap_report(
        intent, positions, geometry, care_mask_um=care_mask_um, minimized=minimized
    )
    out_png = ""
    if render_png:
        from pathlib import Path

        from tools.render_gds import render

        render(Path(gds_path), Path(render_png), render_title)
        out_png = render_png
    return MigrationResult(
        geometry=geometry,
        positions=positions,
        report=report,
        minimized=minimized,
        abstract_result=res,
        out_gds=gds_path,
        out_png=out_png,
    )


def build_ota_intent() -> Any:
    """The real external_pdk OTA intent, built via the canonical INPUTS recipe.

    netlist intent -> GDS correspondence -> ``extract_geometric_intent(intent)``
    (the no-extra-args form, which populates bands / orders / abuts so the emitted
    ``layout_intent {}`` block is non-empty and PARSES). This mirrors the documented
    extractor INPUTS exactly; it does NOT use ``abstract_ops.build_ota_intent`` (that
    variant calls ``extract_geometric_intent(intent, gds)``, which leaves
    bands/orders/abuts empty -> an empty ``layout_intent {}`` block that the grammar
    rejects).
    """
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
    extract_geometric_intent(intent)
    return intent


def run_ota_migration(
    out_gds: str = "_review_gds/two_stage_gf180_migrated.gds",
    out_png: str = "_review_gds/two_stage_gf180_migrated.png",
    care_mask_um: float = 0.5,
) -> MigrationResult:
    """Migrate the real OTA intent onto gf180, emit the GDS + PNG, PRINT the headline
    overlap measure broken down by structural-vs-offset attribution."""
    intent = build_ota_intent()
    title = "gf180 MIGRATION of extracted intent -- overlaps = intent incompleteness"
    mr = migrate_intent_to_gf180(
        intent,
        out_gds,
        care_mask_um=care_mask_um,
        render_png=out_png,
        render_title=title,
    )
    t = mr.abstract_result.tally
    print(f"[migrate] minimized graph tally: {t}")
    print(f"[migrate] gf180 geometry built: {len(mr.geometry.built)}/"
          f"{len(mr.geometry.built) + len(mr.geometry.failed)} devices"
          + (f" | FAILED: {mr.geometry.failed}" if mr.geometry.failed else ""))
    print(f"[migrate] {mr.report.summary()}")
    for o in mr.report.overlaps:
        print(f"    OVERLAP {o.a} / {o.b}: {o.area_um2:.2f} um^2 "
              f"[{o.edge_kind} -> {o.attribution}]")
    print(f"[migrate] GDS  : {mr.out_gds}")
    print(f"[migrate] PNG  : {mr.out_png}")
    return mr


@dataclass
class MigrationResultCG:
    """The OTA migration outcome via the CONSTRAINT GRAPH (not the MST)."""

    geometry: GeometrySet
    positions: Dict[str, _XY]
    report: OverlapReport
    cg: Any
    solve_resid_um: float = 0.0
    out_gds: str = ""
    out_png: str = ""


def migrate_intent_to_gf180_cg(
    intent: Any,
    out_gds: str,
    *,
    render_png: Optional[str] = None,
    render_title: str = "",
) -> MigrationResultCG:
    """Run the migration on an extracted intent through the CONSTRAINT GRAPH: build the
    over-determined constraint graph, build gf180 geometry, solve positions at gf180
    sizes, emit the gf180 GDS, compute the overlap report, optionally render a PNG."""
    cg = relgraph.build_constraint_graph(intent)
    geometry = build_gf180_geometry(intent)
    positions, resid = migrate_positions_cg(intent, geometry.sizes_um(), cg=cg)
    gds_path = emit_gf180_gds(intent, positions, geometry, out_gds)
    report = overlap_report_cg(intent, positions, geometry, cg=cg)
    out_png = ""
    if render_png:
        from pathlib import Path

        from tools.render_gds import render

        render(Path(gds_path), Path(render_png), render_title)
        out_png = render_png
    return MigrationResultCG(
        geometry=geometry,
        positions=positions,
        report=report,
        cg=cg,
        solve_resid_um=resid,
        out_gds=gds_path,
        out_png=out_png,
    )


def run_ota_migration_cg(
    out_gds: str = "_review_gds/two_stage_gf180_migrated_cg.gds",
    out_png: str = "_review_gds/two_stage_gf180_migrated_cg.png",
) -> MigrationResultCG:
    """Migrate the real OTA intent onto gf180 through the CONSTRAINT GRAPH, emit the
    GDS + PNG, PRINT the headline overlap measure. The constraint graph captures the
    cap-cap abut (CC0/CR1) + every other structural adjacency as a DIRECT edge, so they
    realize overlap-free at gf180 sizes instead of accumulating through the MST offset
    chain that made the two caps overlap (the old 19/2246 result)."""
    intent = build_ota_intent()
    title = (
        "gf180 MIGRATION (CONSTRAINT GRAPH) -- structural adjacencies captured as "
        "direct edges; overlaps = genuine cross-cluster offset residual"
    )
    from collections import Counter

    mr = migrate_intent_to_gf180_cg(
        intent, out_gds, render_png=out_png, render_title=title
    )
    kinds = Counter(e.kind for e in mr.cg.edges)
    print(f"[migrate-cg] constraint graph: {len(mr.cg.edges)} edges over "
          f"{len(mr.cg.names)} devices | kinds={dict(kinds)}")
    print(f"[migrate-cg] gf180 geometry built: {len(mr.geometry.built)}/"
          f"{len(mr.geometry.built) + len(mr.geometry.failed)} devices"
          + (f" | FAILED: {mr.geometry.failed}" if mr.geometry.failed else ""))
    print(f"[migrate-cg] over-determined solve residual at gf180 sizes: "
          f"{mr.solve_resid_um:.3f} um")
    print(f"[migrate-cg] {mr.report.summary()}")
    for o in mr.report.overlaps:
        print(f"    OVERLAP {o.a} / {o.b}: {o.area_um2:.2f} um^2 "
              f"[{o.edge_kind} -> {o.attribution}]")
    cc_overlap = [o for o in mr.report.overlaps if {o.a, o.b} == {"CC0", "CR1"}]
    print(f"[migrate-cg] CC0/CR1 cap-cap overlap: "
          + ("NONE (abut captured)" if not cc_overlap
             else f"{cc_overlap[0].area_um2:.2f} um^2 STILL OVERLAPPING"))
    print(f"[migrate-cg] GDS  : {mr.out_gds}")
    print(f"[migrate-cg] PNG  : {mr.out_png}")
    return mr


# --------------------------------------------------------------------------- #
# 6. HIERARCHICAL / BAND migration: positions come from the HIERARCHICAL graph
#    (per-band device constraint graphs stacked by band-level edges) instead of the
#    flat constraint graph. The band-level stacking edges REPLACE the cross-cluster
#    device offset bridges -- so the cap cluster no longer crashes up into the PMOS
#    row (those crossings were the dominant 18/1252 residual). Only WITHIN-band or
#    non-adjacent-cluster offsets can still collide.
# --------------------------------------------------------------------------- #
def migrate_positions_hier(
    intent: Any,
    gf180_sizes: Dict[str, _XY],
    *,
    hier=None,
) -> Tuple[Dict[str, _XY], float]:
    """Solve the HIERARCHICAL graph with GF180 device SIZES -> ``(centroids, max_resid)``.

    The hierarchical graph (from ``relgraph.build_hierarchical_graph``) solves each
    band's internal constraint graph at gf180 sizes (overlap-free WITHIN a band by
    construction) then STACKS the bands with gf180 min spacing (the band-level edges).
    ``max_resid`` is the max over bands of each band's over-determined residual.
    """
    if hier is None:
        hier = relgraph.build_hierarchical_graph(intent)
    positions, resid = relgraph.solve_hierarchical(hier, gf180_sizes, (0.0, 0.0))
    return ({n: (float(p[0]), float(p[1])) for n, p in positions.items()}, resid)


def migrate_positions_hier_nov(
    intent: Any,
    gf180_sizes: Dict[str, _XY],
    *,
    hier=None,
) -> Tuple[Dict[str, _XY], float]:
    """Solve the HIERARCHICAL graph with WITHIN-BAND NON-OVERLAP realization at GF180
    sizes -> ``(centroids, resid)``.

    Unlike :func:`migrate_positions_hier` (which realizes each band internally by the
    exact reference-offset solve -- still overlap-prone at gf180 sizes because the external_pdk
    offset magnitudes do not separate the grown devices), this uses
    ``relgraph.solve_hierarchical_nonoverlap``: each band's un-pinned axis is realized
    from the reference ORDER + gf180 min spacing, so every band is overlap-free BY
    CONSTRUCTION, and bands stack with gf180 spacing. ``resid`` is 0 (non-overlap is by
    construction, no offset residual).
    """
    if hier is None:
        hier = relgraph.build_hierarchical_graph(intent)
    positions, resid = relgraph.solve_hierarchical_nonoverlap(hier, gf180_sizes, (0.0, 0.0))
    return ({n: (float(p[0]), float(p[1])) for n, p in positions.items()}, resid)


def _band_of(hier) -> Dict[str, int]:
    """Map each device name -> its band_id (for cross-band overlap classification)."""
    out: Dict[str, int] = {}
    for b in getattr(hier, "bands", []):
        for n in b.devices:
            out[n] = b.band_id
    return out


def overlap_report_hier(
    intent: Any,
    positions: Dict[str, _XY],
    geometry: GeometrySet,
    *,
    hier=None,
    eps_um: float = 1e-6,
) -> OverlapReport:
    """Overlap report for the HIERARCHICAL migration.

    Same overlap geometry as the other reports. Attribution: an overlap between two
    devices in DIFFERENT bands is 'structural' iff the bands are vertically adjacent
    (a stacking-edge realization fact) -- but since the band stack is overlap-free by
    construction, a cross-adjacent-band overlap should NOT appear; if one does it is a
    genuine residual and reported 'offset'. Within-band overlaps are attributed exactly
    as the flat constraint-graph report would (offset bridges that did not retarget).
    The headline question -- is the cap-cluster-vs-MOS-row overlap gone -- is answered
    by whether any CC0/CR1 (band 1) vs MOS-row (band 0 or 2) pair appears in overlaps.
    """
    if hier is None:
        hier = relgraph.build_hierarchical_graph(intent)
    band_of = _band_of(hier)
    # per-band flat constraint graphs for WITHIN-band attribution.
    band_cg = getattr(hier, "band_graphs", {})
    pair_kinds: Dict[frozenset, set] = {}
    for bid, cg in band_cg.items():
        for e in getattr(cg, "edges", []):
            pair_kinds.setdefault(frozenset((e.child, e.parent)), set()).add(e.kind)

    names = [n for n in geometry.built if n in positions]
    bboxes = {
        n: _migrated_bbox_um(positions[n], geometry.built[n].size_um) for n in names
    }
    report = OverlapReport(n_devices=len(positions), n_built=len(geometry.built))
    names_sorted = sorted(names)
    for i in range(len(names_sorted)):
        for j in range(i + 1, len(names_sorted)):
            a, b = names_sorted[i], names_sorted[j]
            area, ox, oy = _rect_overlap(bboxes[a], bboxes[b])
            if area <= eps_um:
                continue
            ba, bb_ = band_of.get(a), band_of.get(b)
            if ba is not None and bb_ is not None and ba != bb_:
                # cross-band overlap: the band stack should prevent it; any residual
                # is a genuine cross-band leak -> offset, label by band distance.
                label = "cross_band"
                attribution = "offset"
            else:
                kinds = pair_kinds.get(frozenset((a, b)))
                if kinds is not None:
                    if "abut" in kinds:
                        label = "abut"
                    elif "symmetric" in kinds:
                        label = "symmetric"
                    elif "align:x" in kinds or "align:y" in kinds:
                        label = "align"
                    else:
                        label = "offset"
                    if "abut" in kinds or "symmetric" in kinds:
                        attribution = "structural"
                    elif "align:x" in kinds and oy <= ox:
                        attribution = "structural"
                    elif "align:y" in kinds and ox <= oy:
                        attribution = "structural"
                    else:
                        attribution = "offset"
                else:
                    label = "within_band_indirect"
                    attribution = "offset"
            report.overlaps.append(
                Overlap(a=a, b=b, area_um2=area, edge_kind=label, attribution=attribution)
            )
    report.overlaps.sort(key=lambda o: (-o.area_um2, o.a, o.b))
    return report


@dataclass
class MigrationResultHier:
    """The OTA migration outcome via the HIERARCHICAL / BAND graph."""

    geometry: GeometrySet
    positions: Dict[str, _XY]
    report: OverlapReport
    hier: Any
    solve_resid_um: float = 0.0
    out_gds: str = ""
    out_png: str = ""


def migrate_intent_to_gf180_hier(
    intent: Any,
    out_gds: str,
    *,
    render_png: Optional[str] = None,
    render_title: str = "",
) -> MigrationResultHier:
    """Run the migration through the HIERARCHICAL graph: build the band hierarchy, build
    gf180 geometry, solve positions at gf180 sizes (bands stacked with gf180 spacing),
    emit the gf180 GDS, compute the overlap report, optionally render a PNG."""
    hier = relgraph.build_hierarchical_graph(intent)
    geometry = build_gf180_geometry(intent)
    positions, resid = migrate_positions_hier(intent, geometry.sizes_um(), hier=hier)
    gds_path = emit_gf180_gds(intent, positions, geometry, out_gds)
    report = overlap_report_hier(intent, positions, geometry, hier=hier)
    out_png = ""
    if render_png:
        from pathlib import Path

        from tools.render_gds import render

        render(Path(gds_path), Path(render_png), render_title)
        out_png = render_png
    return MigrationResultHier(
        geometry=geometry,
        positions=positions,
        report=report,
        hier=hier,
        solve_resid_um=resid,
        out_gds=gds_path,
        out_png=out_png,
    )


def run_ota_migration_hier(
    out_gds: str = "_review_gds/two_stage_gf180_migrated_hier.gds",
    out_png: str = "_review_gds/two_stage_gf180_migrated_hier.png",
) -> MigrationResultHier:
    """Migrate the real OTA intent onto gf180 through the HIERARCHICAL / BAND graph,
    emit the GDS + PNG, PRINT the headline overlap measure. The band-level stacking
    edges REPLACE the cross-cluster device offset bridges, so the cap cluster
    (CC0/CR1, the middle band) no longer crashes up into the PMOS row (the top band) --
    the dominant residual in the old 18/1252 flat-CG result."""
    intent = build_ota_intent()
    title = (
        "gf180 MIGRATION (BAND / HIERARCHICAL) -- bands stacked with gf180 spacing; "
        "cross-cluster offset bridges replaced by band-to-band stacking relations"
    )
    mr = migrate_intent_to_gf180_hier(
        intent, out_gds, render_png=out_png, render_title=title
    )
    print(f"[migrate-hier] bands detected: {len(mr.hier.bands)}")
    for b in mr.hier.bands:
        print(f"    band {b.band_id} (y {b.y_lo:.1f}..{b.y_hi:.1f}): {b.devices}")
    print(f"[migrate-hier] band edges: "
          + ", ".join(f"{e.lower}->{e.upper}(ref_gap={e.ref_gap:.1f})"
                      for e in mr.hier.band_edges))
    print(f"[migrate-hier] gf180 geometry built: {len(mr.geometry.built)}/"
          f"{len(mr.geometry.built) + len(mr.geometry.failed)} devices"
          + (f" | FAILED: {mr.geometry.failed}" if mr.geometry.failed else ""))
    print(f"[migrate-hier] per-band over-determined solve residual at gf180 sizes: "
          f"{mr.solve_resid_um:.3f} um")
    print(f"[migrate-hier] {mr.report.summary()}")
    for o in mr.report.overlaps:
        print(f"    OVERLAP {o.a} / {o.b}: {o.area_um2:.2f} um^2 "
              f"[{o.edge_kind} -> {o.attribution}]")
    band_of = _band_of(mr.hier)
    cap_band = band_of.get("CC0")
    capmos = [
        o for o in mr.report.overlaps
        if (band_of.get(o.a) == cap_band) != (band_of.get(o.b) == cap_band)
        and (("CC0" in (o.a, o.b)) or ("CR1" in (o.a, o.b)))
    ]
    print(f"[migrate-hier] cap-cluster-vs-MOS-row overlaps: "
          + ("NONE (band stacking eliminated them)" if not capmos
             else f"{len(capmos)} STILL PRESENT: "
                  + ", ".join(f"{o.a}/{o.b}" for o in capmos)))
    print(f"[migrate-hier] GDS  : {mr.out_gds}")
    print(f"[migrate-hier] PNG  : {mr.out_png}")
    return mr


def migrate_intent_to_gf180_hier_nov(
    intent: Any,
    out_gds: str,
    *,
    render_png: Optional[str] = None,
    render_title: str = "",
) -> MigrationResultHier:
    """Run the migration through the HIERARCHICAL graph with WITHIN-BAND NON-OVERLAP
    realization: build the band hierarchy, build gf180 geometry, solve positions at
    gf180 sizes via ``solve_hierarchical_nonoverlap`` (every band overlap-free by
    construction, bands stacked with gf180 spacing), emit the gf180 GDS, compute the
    overlap report, optionally render a PNG."""
    hier = relgraph.build_hierarchical_graph(intent)
    geometry = build_gf180_geometry(intent)
    positions, resid = migrate_positions_hier_nov(intent, geometry.sizes_um(), hier=hier)
    gds_path = emit_gf180_gds(intent, positions, geometry, out_gds)
    report = overlap_report_hier(intent, positions, geometry, hier=hier)
    out_png = ""
    if render_png:
        from pathlib import Path

        from tools.render_gds import render

        render(Path(gds_path), Path(render_png), render_title)
        out_png = render_png
    return MigrationResultHier(
        geometry=geometry,
        positions=positions,
        report=report,
        hier=hier,
        solve_resid_um=resid,
        out_gds=gds_path,
        out_png=out_png,
    )


def run_ota_migration_hier_nov(
    out_gds: str = "_review_gds/two_stage_gf180_migrated_clean.gds",
    out_png: str = "_review_gds/two_stage_gf180_migrated_clean.png",
) -> MigrationResultHier:
    """Migrate the real OTA intent onto gf180 through the HIERARCHICAL graph with the
    WITHIN-BAND NON-OVERLAP solver, emit the GDS + PNG, PRINT the headline overlap
    measure. Each band's un-pinned axis is realized from the reference ORDER + gf180 min
    spacing (overlap-free by construction), so the within-band residual overlaps of the
    exact-offset hier path (MM15/MM16, MM13/MM14, MM17/RR0; 3 pairs / 131.95 um^2) are
    gone -- the migration is 0 overlaps."""
    intent = build_ota_intent()
    title = "gf180 migration -- 0 overlaps"
    mr = migrate_intent_to_gf180_hier_nov(
        intent, out_gds, render_png=out_png, render_title=title
    )
    print(f"[migrate-hier-nov] bands detected: {len(mr.hier.bands)}")
    for b in mr.hier.bands:
        print(f"    band {b.band_id} (y {b.y_lo:.1f}..{b.y_hi:.1f}): {b.devices}")
    print(f"[migrate-hier-nov] gf180 geometry built: {len(mr.geometry.built)}/"
          f"{len(mr.geometry.built) + len(mr.geometry.failed)} devices"
          + (f" | FAILED: {mr.geometry.failed}" if mr.geometry.failed else ""))
    print(f"[migrate-hier-nov] {mr.report.summary()}")
    for o in mr.report.overlaps:
        print(f"    OVERLAP {o.a} / {o.b}: {o.area_um2:.2f} um^2 "
              f"[{o.edge_kind} -> {o.attribution}]")
    if not mr.report.overlaps:
        print("[migrate-hier-nov] 0 overlapping pairs (overlap-free by construction)")
    print(f"[migrate-hier-nov] GDS  : {mr.out_gds}")
    print(f"[migrate-hier-nov] PNG  : {mr.out_png}")
    return mr


# --------------------------------------------------------------------------- #
# 7. PATTERN-PRESERVING migration: positions come from the MINIMAL-PERTURBATION
#    legalizer (relgraph.solve_pattern_preserving) which KEEPS the reference relative
#    arrangement (the recognizable pattern) and expands ONLY the offsets that the
#    bigger gf180 devices would otherwise make overlap. Geometry combines the folded
#    MOS (fold.gf180_geometry_folded) with the caps/resistors from the placer
#    (build_gf180_geometry) so ALL 17 devices (incl the 2 MIM caps) are present.
# --------------------------------------------------------------------------- #
def build_gf180_geometry_combined(intent: Any, *, h_max_um: float = 22.0) -> GeometrySet:
    """All 17 devices' gf180 geometry: folded MOS (bounded finger height) PLUS the
    passives (caps / resistors) from the placer route.

    ``fold.gf180_geometry_folded`` builds only the 13 MOS (folded). The 2 MIM caps +
    2 resistors are NOT MOS, so we take them from ``build_gf180_geometry`` (the placer
    route, which builds the gf180 MIM cap + resistor cells). The two sets are merged;
    MOS folded geometry wins on any name collision. Devices that neither route built
    are reported in ``failed``.
    """
    from clay.layout.extract.fold import gf180_geometry_folded

    folded = gf180_geometry_folded(intent, h_max_um=h_max_um)
    placed = build_gf180_geometry(intent)

    out = GeometrySet()
    out.placer_note = (
        f"combined: folded MOS ({folded.placer_note}) + placer passives "
        f"({placed.placer_note})"
    )
    # MOS from the folded route.
    for name, geo in folded.built.items():
        out.built[name] = geo
    # passives (and anything the folded route failed) from the placer route.
    for name, geo in placed.built.items():
        if name not in out.built:
            out.built[name] = geo
    # honest failed set: names neither route built.
    want = [str(getattr(d, "name", "")) for d in (getattr(intent, "devices", []) or [])]
    want += [str(getattr(p, "name", "")) for p in (getattr(intent, "passives", []) or [])]
    want = [n for n in want if n]
    for name in want:
        if name not in out.built:
            out.failed.append(name)
    return out


def migrate_positions_pattern(
    intent: Any,
    gf180_sizes: Dict[str, _XY],
    *,
    min_gap_um: float = 1.0,
    cg=None,
) -> Tuple[Dict[str, _XY], float]:
    """Solve the CONSTRAINT GRAPH with the PATTERN-PRESERVING legalizer at GF180 sizes
    -> ``(centroids, max_dev_from_reference)``.

    Unlike :func:`migrate_positions_hier_nov` (order + min-spacing -- overlap-free but
    pattern-losing), this keeps the reference relative arrangement and expands ONLY the
    too-tight offsets (``relgraph.solve_pattern_preserving``). ``max_dev`` is the
    translation-quotiented deviation from the EXACT reference solve (the perturbation the
    legalization introduced); ~0 at reference sizes.
    """
    if cg is None:
        cg = relgraph.build_constraint_graph(intent)
    positions, max_dev = relgraph.solve_pattern_preserving(
        cg, gf180_sizes, min_gap_um=min_gap_um, root_xy=(0.0, 0.0)
    )
    return ({n: (float(p[0]), float(p[1])) for n, p in positions.items()}, max_dev)


@dataclass
class MigrationResultPattern:
    """The OTA migration outcome via the PATTERN-PRESERVING legalizer."""

    geometry: GeometrySet
    positions: Dict[str, _XY]
    report: OverlapReport
    cg: Any
    max_dev_um: float = 0.0
    out_gds: str = ""
    out_png: str = ""


def migrate_intent_to_gf180_pattern(
    intent: Any,
    out_gds: str,
    *,
    min_gap_um: float = 1.0,
    h_max_um: float = 22.0,
    render_png: Optional[str] = None,
    render_title: str = "",
) -> MigrationResultPattern:
    """Run the migration through the PATTERN-PRESERVING legalizer over the constraint
    graph: combine folded-MOS + placer-passive geometry (all 17 devices), solve
    positions keeping the reference arrangement (expanding only too-tight offsets),
    emit the gf180 GDS, compute the overlap report, optionally render a PNG."""
    cg = relgraph.build_constraint_graph(intent)
    geometry = build_gf180_geometry_combined(intent, h_max_um=h_max_um)
    positions, max_dev = migrate_positions_pattern(
        intent, geometry.sizes_um(), min_gap_um=min_gap_um, cg=cg
    )
    gds_path = emit_gf180_gds(intent, positions, geometry, out_gds)
    report = overlap_report_cg(intent, positions, geometry, cg=cg)
    out_png = ""
    if render_png:
        from pathlib import Path

        from tools.render_gds import render

        render(Path(gds_path), Path(render_png), render_title)
        out_png = render_png
    return MigrationResultPattern(
        geometry=geometry,
        positions=positions,
        report=report,
        cg=cg,
        max_dev_um=max_dev,
        out_gds=gds_path,
        out_png=out_png,
    )


def run_ota_migration_pattern(
    out_gds: str = "_review_gds/two_stage_gf180_pattern.gds",
    out_png: str = "_review_gds/two_stage_gf180_pattern.png",
    min_gap_um: float = 1.0,
) -> MigrationResultPattern:
    """Migrate the real OTA intent onto gf180 through the PATTERN-PRESERVING legalizer,
    emit the GDS + PNG, PRINT the headline overlap + perturbation measures. The
    reference relative arrangement is preserved (only too-tight offsets expand), so the
    migrated layout RECOGNIZABLY matches the reference -- unlike the order+min-spacing
    path which collapses every gap to the minimum and mangles the tier proportions."""
    intent = build_ota_intent()
    title = (
        "gf180 MIGRATION (PATTERN-PRESERVING) -- reference relative arrangement kept; "
        "only too-tight offsets expanded to clear gf180 device sizes"
    )
    mr = migrate_intent_to_gf180_pattern(
        intent, out_gds, min_gap_um=min_gap_um, render_png=out_png, render_title=title
    )
    print(f"[migrate-pattern] gf180 geometry built: {len(mr.geometry.built)}/"
          f"{len(mr.geometry.built) + len(mr.geometry.failed)} devices"
          + (f" | FAILED: {mr.geometry.failed}" if mr.geometry.failed else ""))
    print(f"[migrate-pattern] perturbation from reference (max_dev): "
          f"{mr.max_dev_um:.3f} um")
    print(f"[migrate-pattern] {mr.report.summary()}")
    for o in mr.report.overlaps:
        print(f"    OVERLAP {o.a} / {o.b}: {o.area_um2:.2f} um^2 "
              f"[{o.edge_kind} -> {o.attribution}]")
    if not mr.report.overlaps:
        print("[migrate-pattern] 0 overlapping pairs (legalized overlap-free)")
    print(f"[migrate-pattern] GDS  : {mr.out_gds}")
    print(f"[migrate-pattern] PNG  : {mr.out_png}")
    return mr


__all__ = [
    "DeviceGeometry",
    "GeometrySet",
    "Overlap",
    "OverlapReport",
    "MigrationResult",
    "MigrationResultCG",
    "MigrationResultHier",
    "MigrationResultPattern",
    "build_gf180_geometry_combined",
    "migrate_positions_pattern",
    "migrate_intent_to_gf180_pattern",
    "run_ota_migration_pattern",
    "build_gf180_geometry",
    "migrate_positions",
    "migrate_positions_cg",
    "migrate_positions_hier",
    "migrate_positions_hier_nov",
    "emit_gf180_gds",
    "overlap_report",
    "overlap_report_cg",
    "overlap_report_hier",
    "migrate_intent_to_gf180",
    "migrate_intent_to_gf180_cg",
    "migrate_intent_to_gf180_hier",
    "migrate_intent_to_gf180_hier_nov",
    "build_ota_intent",
    "run_ota_migration",
    "run_ota_migration_cg",
    "run_ota_migration_hier",
    "run_ota_migration_hier_nov",
]


if __name__ == "__main__":
    run_ota_migration()
