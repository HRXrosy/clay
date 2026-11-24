"""R140 (BL-140C) — eliminate gf180mcu ``PL.6`` (90-degree poly bends on COMP).

DIAGNOSIS (read off the real foundry deck, verbatim)
----------------------------------------------------
The real gf180mcu KLayout DRC deck encodes ``PL.6`` as (deck line 1550, also
``rule_decks/poly2.drc`` line 325)::

    pl6_l1 = poly2.corners(90.0).sized(0.1)
                 .or(poly2.corners(-90.0).sized(0.1))
                 .not(ymtp_mk)
                 .inside(comp.not(ymtp_mk))
    pl6_l1.output("PL.6", "PL.6 : 90 degree bends on the COMP are not allowed.")

So ``PL.6`` does NOT forbid bends on the COMP polygon itself — it forbids any
**Poly2 (30/0) 90-degree corner** (a convex ``corners(90.0)`` OR a concave
``corners(-90.0)`` corner), grown by 0.1um, whose box lands *inside* the COMP
(22/0) active region. Physically: the gate poly crossing the channel must be a
straight strip; an L-bend / step in the poly *while it is over the diffusion*
is illegal because the bend point has no well-defined channel length.

WHAT PRODUCED THE 40 INSTANCES (R139 real signoff)
--------------------------------------------------
The R139 signoff GDS was built by ``route_annotated_cell_sane`` (R136), which
lowers each device with the legacy ``backend.lower_mos_finger_geometry``. There
each gate poly finger is::

    Polygon.rectangle(x_min, 0, x_max, w_nm)          # poly finger: y in [0, w]

while the surrounding COMP/NDIFF rectangle is::

    Polygon.rectangle(ndiff_x_min, -fin_pitch,
                      ndiff_x_max, w_nm + fin_pitch)   # COMP: y in [-fp, w+fp]

Because the COMP rectangle extends ``fin_pitch`` (280 nm) *beyond* the poly in
both +y and -y (and wider in x), all **four** convex 90-degree corners of every
poly finger rectangle (at ``(x_min,0)``, ``(x_max,0)``, ``(x_min,w)``,
``(x_max,w)``) sit strictly *inside* COMP. The 5T-OTA has five fingered devices
contributing ~10 poly fingers, i.e. ~40 poly corners inside COMP -> ``PL.6 x40``.

The real device geometry (``gf180_device_geom.build_gf180_mos_device``, R139 A)
already avoids this by giving each poly finger a ``PL.4`` end-cap so the finger
spans ``y in [-endcap, w+endcap]`` while COMP spans ``y in [0, w]`` — there the
poly corners are *above/below* COMP, not inside it. This module is the
*normalizer* that turns any already-built cell's geometry (including the legacy
fingered lowering still used by the routed flow) PL.6-clean without depending on
the device builder being switched over.

THE FIX
-------
A poly 90-degree corner inside COMP is removed by **chamfering** it: the single
right-angle vertex is replaced by two vertices that cut a 45-degree bevel across
the corner. The deck explicitly permits 45-degree poly on COMP (rule PL.7), and
a 45-degree edge has corner turn angles of +-45, never +-90, so the chamfered
corner no longer matches ``poly2.corners(90.0|-90.0)`` and PL.6 cannot fire on
it. The chamfer is small (default 100 nm leg, == the rule's own ``.sized(0.1)``
probe) so device electricals are essentially unchanged; chamfering only removes
material at the corner, never adds it, so it cannot create new overlaps.

This is pure geometry. It imports the committed ``backend`` (Point / Polygon /
LayerPolygon / NamedCell / GdsLayer) read-only and fabricates no coordinates —
every output vertex is derived from an input vertex.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .backend import (
    GdsLayer,
    LayerPolygon,
    NamedCell,
    Point,
    Polygon,
)

# Real gf180mcu GDS keys (from gf180_layermap.GF180_VERIFIED_SOURCES, cited to
# gf180mcu.lyp). Re-declared here as plain tuples so this module has no import
# cycle with the layer map; the values are checked against it in the tests.
GF180_POLY2_GDS: Tuple[int, int] = (30, 0)  # Poly2 (gate) — PL.6 target layer
GF180_COMP_GDS: Tuple[int, int] = (22, 0)  # COMP (diffusion / active)

# The deck's own PL.6 corner probe size is ``.sized(0.1)`` = 100 nm. We use the
# same magnitude for the default chamfer leg so the geometric perturbation is no
# larger than the rule's own detection box.
DEFAULT_CHAMFER_NM = 100


# --- small geometry helpers (axis-aligned + general polygon) -----------------


def _signed_area_2x(verts: Sequence[Point]) -> int:
    """Twice the signed area (shoelace). >0 => CCW, <0 => CW."""
    n = len(verts)
    acc = 0
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        acc += a.x * b.y - b.x * a.y
    return acc


def _is_ccw(verts: Sequence[Point]) -> bool:
    return _signed_area_2x(verts) > 0


@dataclass(frozen=True)
class CornerInfo:
    """A polygon corner classified for PL.6 purposes.

    index:    vertex index within the polygon.
    point:    the corner vertex.
    turn:     +90 for a convex right-angle (``corners(90.0)``),
              -90 for a concave right-angle (``corners(-90.0)``),
              ``None`` if the corner is not a 90-degree bend.
    inside_comp: True iff the corner's sized probe box lies inside COMP.
    """

    index: int
    point: Point
    turn: Optional[int]
    inside_comp: bool


def _bbox(verts: Sequence[Point]) -> Tuple[int, int, int, int]:
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def classify_corner(prev: Point, cur: Point, nxt: Point, ccw: bool) -> Optional[int]:
    """Return +90 (convex), -90 (concave), or None for a polygon corner.

    Mirrors KLayout's ``corners(90.0)`` / ``corners(-90.0)`` which select
    right-angle convex / concave corners. The two incoming/outgoing edges must
    be axis-aligned and orthogonal; the convex/concave sense is taken from the
    z-component of the edge cross product, flipped for CW polygons so the result
    matches KLayout's interior-angle convention regardless of vertex order.
    """
    e1x, e1y = cur.x - prev.x, cur.y - prev.y
    e2x, e2y = nxt.x - cur.x, nxt.y - cur.y
    # both edges axis-aligned?
    if not ((e1x == 0) ^ (e1y == 0)) or not ((e2x == 0) ^ (e2y == 0)):
        return None
    # orthogonal turn? (one edge horizontal, the other vertical)
    if (e1x == 0) == (e2x == 0):
        return None  # collinear or anti-parallel — not a 90-degree bend
    cross = e1x * e2y - e1y * e2x  # >0 left turn, <0 right turn (CCW frame)
    if cross == 0:
        return None
    left_turn = cross > 0
    if not ccw:
        left_turn = not left_turn
    # In a CCW outer boundary, a left turn is a convex (+90) corner.
    return 90 if left_turn else -90


def _probe_box(p: Point, half: int) -> Tuple[int, int, int, int]:
    """The ``.sized(half)`` probe box (x_min, y_min, x_max, y_max) around p."""
    return (p.x - half, p.y - half, p.x + half, p.y + half)


def _box_inside_rect(
    box: Tuple[int, int, int, int], rect: Tuple[int, int, int, int]
) -> bool:
    bx0, by0, bx1, by1 = box
    rx0, ry0, rx1, ry1 = rect
    return bx0 >= rx0 and by0 >= ry0 and bx1 <= rx1 and by1 <= ry1


def corner_inside_any_rect(
    p: Point, comp_rects: Sequence[Tuple[int, int, int, int]], half: int
) -> bool:
    """True iff the corner probe box around ``p`` is inside any COMP rect."""
    box = _probe_box(p, half)
    return any(_box_inside_rect(box, r) for r in comp_rects)


def classify_polygon_corners(
    poly: Polygon,
    comp_rects: Sequence[Tuple[int, int, int, int]],
    probe_half_nm: int = DEFAULT_CHAMFER_NM,
) -> List[CornerInfo]:
    """Classify every corner of ``poly`` for the PL.6 condition."""
    verts = poly.vertices
    n = len(verts)
    if n < 3:
        return []
    ccw = _is_ccw(verts)
    out: List[CornerInfo] = []
    for i in range(n):
        prev = verts[(i - 1) % n]
        cur = verts[i]
        nxt = verts[(i + 1) % n]
        turn = classify_corner(prev, cur, nxt, ccw)
        inside = (
            corner_inside_any_rect(cur, comp_rects, probe_half_nm)
            if turn is not None
            else False
        )
        out.append(CornerInfo(index=i, point=cur, turn=turn, inside_comp=inside))
    return out


# --- public predicate: count PL.6-offending corners on a layer ----------------


def comp_rects_from_polygons(
    polygons: Iterable[LayerPolygon], comp_gds: Tuple[int, int] = GF180_COMP_GDS
) -> List[Tuple[int, int, int, int]]:
    """Axis-aligned bounding rects of every COMP polygon in a cell.

    The legacy and real device builders both emit COMP as plain rectangles, so
    a bbox per COMP polygon is exact. (Non-rectangular COMP would over-approx
    its interior with the bbox; for the rectilinear COMP this PDK uses, the bbox
    *is* the polygon.)
    """
    rects: List[Tuple[int, int, int, int]] = []
    for lp in polygons:
        if (lp.layer, lp.datatype) == comp_gds and lp.polygon.vertices:
            rects.append(_bbox(lp.polygon.vertices))
    return rects


def count_pl6_corners(
    polygons: Sequence[LayerPolygon],
    *,
    target_gds: Tuple[int, int] = GF180_POLY2_GDS,
    comp_gds: Tuple[int, int] = GF180_COMP_GDS,
    probe_half_nm: int = DEFAULT_CHAMFER_NM,
) -> int:
    """Count target-layer 90-degree corners (convex+concave) inside COMP.

    This is a faithful Python mirror of the deck's ``PL.6`` selection
    (``poly2.corners(90.0|-90.0).sized(0.1).inside(comp)``) for axis-aligned
    geometry, so a result of 0 here predicts a clean PL.6 from the real deck.
    """
    comp_rects = comp_rects_from_polygons(polygons, comp_gds)
    if not comp_rects:
        return 0
    total = 0
    for lp in polygons:
        if (lp.layer, lp.datatype) != target_gds:
            continue
        for ci in classify_polygon_corners(lp.polygon, comp_rects, probe_half_nm):
            if ci.turn is not None and ci.inside_comp:
                total += 1
    return total


# --- the normalizer: chamfer offending 90-degree corners ----------------------


def _chamfer_corner_vertices(
    prev: Point, cur: Point, nxt: Point, leg: int
) -> List[Point]:
    """Replace the right-angle vertex ``cur`` with a 45-degree bevel.

    Returns the two new vertices that cut across the corner: one a distance
    ``leg`` back along the incoming edge (prev->cur) and one a distance ``leg``
    forward along the outgoing edge (cur->nxt). The bevel always removes the
    corner material (cuts inward), so for both convex and concave right angles
    the polygon stays simple and never grows.

    ``leg`` is clamped so it never exceeds half of either adjacent edge, keeping
    the chamfer inside both edges (no self-intersection for short fingers).
    """
    e1x, e1y = cur.x - prev.x, cur.y - prev.y
    e2x, e2y = nxt.x - cur.x, nxt.y - cur.y
    len1 = abs(e1x) + abs(e1y)  # axis-aligned -> manhattan == euclidean length
    len2 = abs(e2x) + abs(e2y)
    if len1 == 0 or len2 == 0:
        return [cur]
    leg = max(1, min(leg, len1 // 2, len2 // 2))
    # unit step along each axis-aligned edge
    s1x = (e1x // len1) if e1x else 0
    s1y = (e1y // len1) if e1y else 0
    s2x = (e2x // len2) if e2x else 0
    s2y = (e2y // len2) if e2y else 0
    # point a distance `leg` BEFORE the corner (back along incoming edge)
    p_in = Point(cur.x - s1x * leg, cur.y - s1y * leg)
    # point a distance `leg` AFTER the corner (forward along outgoing edge)
    p_out = Point(cur.x + s2x * leg, cur.y + s2y * leg)
    return [p_in, p_out]


def chamfer_polygon_pl6(
    poly: Polygon,
    comp_rects: Sequence[Tuple[int, int, int, int]],
    *,
    leg_nm: int = DEFAULT_CHAMFER_NM,
    probe_half_nm: int = DEFAULT_CHAMFER_NM,
) -> Polygon:
    """Chamfer every 90-degree corner of ``poly`` that lies inside COMP.

    Non-offending corners (not 90-degree, or outside COMP) pass through
    unchanged. The vertex order is preserved.
    """
    verts = poly.vertices
    n = len(verts)
    if n < 3:
        return Polygon(vertices=list(verts))
    infos = classify_polygon_corners(poly, comp_rects, probe_half_nm)
    new_verts: List[Point] = []
    for i in range(n):
        ci = infos[i]
        if ci.turn is not None and ci.inside_comp:
            prev = verts[(i - 1) % n]
            cur = verts[i]
            nxt = verts[(i + 1) % n]
            new_verts.extend(_chamfer_corner_vertices(prev, cur, nxt, leg_nm))
        else:
            new_verts.append(verts[i])
    return Polygon(vertices=new_verts)


# --- the PREFERRED fix: extend poly endcaps so corners exit COMP -------------
#
# Chamfering removes the 90-degree bend but, on a rectangular poly finger whose
# corners are inside COMP because COMP over-extends past the poly in y, the
# bevel eats into the channel and trips DF.2a (min channel width) instead.
# The structurally-correct fix is to push the poly finger's top/bottom edges
# *out* past the COMP boundary (a PL.4 end-cap), so the corners land OUTSIDE
# COMP and PL.6 cannot match — with the channel width fully preserved. This is
# exactly what ``gf180_device_geom.build_gf180_mos_device`` does by
# construction; here we apply it as a post-process to an already-laid finger.

# poly2.drc PL.4_3.3V "Extension beyond COMP to form Poly2 end cap. : 0.22um"
GF180_POLY_ENDCAP_NM = 220


def _is_axis_rectangle(poly: Polygon) -> Optional[Tuple[int, int, int, int]]:
    """Return (x0,y0,x1,y1) if ``poly`` is an axis-aligned rectangle, else None."""
    verts = poly.vertices
    if len(verts) != 4:
        return None
    xs = sorted({v.x for v in verts})
    ys = sorted({v.y for v in verts})
    if len(xs) != 2 or len(ys) != 2:
        return None
    return (xs[0], ys[0], xs[1], ys[1])


def extend_poly_endcap_pl6(
    poly: Polygon,
    comp_rects: Sequence[Tuple[int, int, int, int]],
    *,
    endcap_nm: int = GF180_POLY_ENDCAP_NM,
    probe_half_nm: int = DEFAULT_CHAMFER_NM,
) -> Optional[Polygon]:
    """Extend a rectangular poly finger in y so its corners exit COMP.

    Only applies when ``poly`` is an axis-aligned rectangle that crosses (is
    contained in y by) some COMP rect and currently has corners inside COMP.
    The rectangle's y-range is grown to one ``endcap_nm`` *beyond* the tallest
    overlapping COMP rect on each side, so all four corners are clear of COMP
    by the PL.4 end-cap. The channel width (poly ∩ COMP in y) is unchanged.

    Returns the extended rectangle, or ``None`` if the simple-rectangle endcap
    fix does not apply (caller should fall back to chamfering).
    """
    rect = _is_axis_rectangle(poly)
    if rect is None:
        return None
    px0, py0, px1, py1 = rect
    # find COMP rects this finger overlaps in x (the ones it could cross)
    crossing = [
        (cx0, cy0, cx1, cy1)
        for (cx0, cy0, cx1, cy1) in comp_rects
        if cx0 < px1 and cx1 > px0
    ]
    if not crossing:
        return None
    comp_top = max(c[3] for c in crossing)
    comp_bot = min(c[1] for c in crossing)
    new_y1 = max(py1, comp_top + endcap_nm)
    new_y0 = min(py0, comp_bot - endcap_nm)
    return Polygon.rectangle(px0, new_y0, px1, new_y1)


def normalize_cell_pl6(
    polygons: Sequence[LayerPolygon],
    *,
    target_gds: Tuple[int, int] = GF180_POLY2_GDS,
    comp_gds: Tuple[int, int] = GF180_COMP_GDS,
    leg_nm: int = DEFAULT_CHAMFER_NM,
    probe_half_nm: int = DEFAULT_CHAMFER_NM,
    prefer_endcap: bool = True,
    endcap_nm: int = GF180_POLY_ENDCAP_NM,
) -> List[LayerPolygon]:
    """Return a new polygon list with all PL.6-offending corners removed.

    Only ``target_gds`` polygons are touched; every other layer (COMP, contact,
    metal, ...) is copied through byte-for-byte. The COMP rects used for the
    inside test are read from the SAME polygon list, so the normalizer is
    self-contained.

    Strategy: for a rectangular poly finger whose corners are inside COMP, the
    default (``prefer_endcap=True``) is to *extend* the finger past COMP (PL.4
    end-cap) — this clears PL.6 without narrowing the channel (no DF.2a). For
    any poly that is not a simple endcap-fixable rectangle, fall back to
    chamfering the offending corners. Set ``prefer_endcap=False`` to always
    chamfer.
    """
    comp_rects = comp_rects_from_polygons(polygons, comp_gds)
    out: List[LayerPolygon] = []
    for lp in polygons:
        if (lp.layer, lp.datatype) == target_gds and comp_rects:
            # does this poly currently have any PL.6-offending corner?
            offends = any(
                ci.turn is not None and ci.inside_comp
                for ci in classify_polygon_corners(lp.polygon, comp_rects, probe_half_nm)
            )
            if not offends:
                new_poly = Polygon(vertices=list(lp.polygon.vertices))
            else:
                new_poly = None
                if prefer_endcap:
                    new_poly = extend_poly_endcap_pl6(
                        lp.polygon,
                        comp_rects,
                        endcap_nm=endcap_nm,
                        probe_half_nm=probe_half_nm,
                    )
                if new_poly is None:
                    new_poly = chamfer_polygon_pl6(
                        lp.polygon,
                        comp_rects,
                        leg_nm=leg_nm,
                        probe_half_nm=probe_half_nm,
                    )
            out.append(
                LayerPolygon(layer=lp.layer, datatype=lp.datatype, polygon=new_poly)
            )
        else:
            out.append(
                LayerPolygon(
                    layer=lp.layer,
                    datatype=lp.datatype,
                    polygon=Polygon(vertices=list(lp.polygon.vertices)),
                )
            )
    return out


def normalize_named_cell_pl6(
    cell: NamedCell,
    *,
    target_gds: Tuple[int, int] = GF180_POLY2_GDS,
    comp_gds: Tuple[int, int] = GF180_COMP_GDS,
    leg_nm: int = DEFAULT_CHAMFER_NM,
    probe_half_nm: int = DEFAULT_CHAMFER_NM,
) -> NamedCell:
    """Convenience wrapper: normalize a :class:`NamedCell` in place-of-copy."""
    return NamedCell(
        name=cell.name,
        polygons=normalize_cell_pl6(
            cell.polygons,
            target_gds=target_gds,
            comp_gds=comp_gds,
            leg_nm=leg_nm,
            probe_half_nm=probe_half_nm,
        ),
    )


# --- L-shape -> rectangle decomposition (for COMP itself, if ever needed) -----


def decompose_rectilinear_to_rects(poly: Polygon) -> List[Polygon]:
    """Decompose an axis-aligned rectilinear polygon into disjoint rectangles.

    A simple vertical-slab sweep: cut at every distinct x of a vertex, and for
    each x-slab emit the maximal rectangles covering the polygon interior in
    that slab. Used to turn an L-shaped / stepped COMP into a set of plain
    rectangles (each of which has only convex corners and, when re-merged for
    DRC, presents no concave 90-degree notch). For an input that is already a
    rectangle this returns a single equivalent rectangle.

    This is a coarse, scan-line decomposition adequate for the rectilinear
    device geometry Clay emits; it is NOT a general polygon triangulator.
    """
    verts = poly.vertices
    if len(verts) < 3:
        return [Polygon(vertices=list(verts))]
    # Already a rectangle (4 axis-aligned corners) — return as-is.
    if len(verts) == 4:
        x0, y0, x1, y1 = _bbox(verts)
        # confirm it's an axis-aligned rectangle
        xs = sorted({v.x for v in verts})
        ys = sorted({v.y for v in verts})
        if len(xs) == 2 and len(ys) == 2:
            return [Polygon.rectangle(x0, y0, x1, y1)]

    xs = sorted({v.x for v in verts})
    rects: List[Polygon] = []
    for i in range(len(xs) - 1):
        xa, xb = xs[i], xs[i + 1]
        xm = (xa + xb) // 2
        # collect y-intervals where the vertical line x=xm is inside the polygon
        crossings: List[int] = []
        n = len(verts)
        for j in range(n):
            a = verts[j]
            b = verts[(j + 1) % n]
            if a.x == b.x:
                continue  # vertical edge — ignore for x-ray crossing
            lo, hi = sorted((a.x, b.x))
            if lo <= xm < hi:
                crossings.append(a.y)  # horizontal edge at y=a.y spans this x
        crossings.sort()
        for k in range(0, len(crossings) - 1, 2):
            y0, y1 = crossings[k], crossings[k + 1]
            if y1 > y0:
                rects.append(Polygon.rectangle(xa, y0, xb, y1))
    return rects if rects else [Polygon(vertices=list(verts))]


__all__ = [
    "GF180_POLY2_GDS",
    "GF180_COMP_GDS",
    "DEFAULT_CHAMFER_NM",
    "CornerInfo",
    "classify_corner",
    "classify_polygon_corners",
    "comp_rects_from_polygons",
    "corner_inside_any_rect",
    "count_pl6_corners",
    "chamfer_polygon_pl6",
    "extend_poly_endcap_pl6",
    "GF180_POLY_ENDCAP_NM",
    "normalize_cell_pl6",
    "normalize_named_cell_pl6",
    "decompose_rectilinear_to_rects",
]
