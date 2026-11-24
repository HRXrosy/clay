"""R135 (BL-135B) — pin-terminal correction + route containment.

R134's routed 5T-OTA leaked geometry outside the cell bounding box: a via12
landed at y = -133 (below the cell, whose y starts at -48) and an M2 wire
climbed to y = 8048 (above the cell), with an M3 bridge spanning nearly the
full width. Two root causes feed that escape:

1. Pin terminals were taken at device-bbox *extremes* (e.g. the tail drain at
   the top-right corner 8368/8048) rather than at sensible on-device terminal
   points. ``cell_netlist.extract_device_pins`` puts the gate at the poly top
   *edge* (y = max poly y), the drain at the diffusion top *edge*, the source
   at the diffusion bottom *edge*. Those are boundary / corner points, so a
   router that escapes from them starts already at the cell perimeter and then
   steps outside.

2. Even given good terminals, a router's M1->M2->M3 escape can produce vias /
   wires whose vertices fall outside the cell bbox (negative-y via, over-top
   M2 climb).

This module supplies reusable, side-effect-free helpers that fix both, WITHOUT
modifying ``backend.py``, ``cell_netlist.py``, or ``routed_lowering_full.py``
(all imported read-only). It fabricates no geometry: every terminal point is
the centroid of a REAL device polygon, and containment only clamps vertices
that already violate the allowed region — a legal in-bbox route is returned
byte-unchanged.

Public API
----------
- ``terminal_point(device_polys, kind, pdk_layers) -> Point``
      An interior terminal point for a device pin kind, derived from the
      device's poly fingers (gate) or diffusion rectangle (source/drain).
- ``point_in_polygon(pt, polygon) -> bool``
      Even-odd point-in-polygon test (boundary counts as inside).
- ``Region`` + ``region_from_bbox`` / ``region_from_cell`` / ``expand``
      The allowed rectangular region (cell bbox + margin).
- ``verify_within(region, polygons) -> bool``
      True iff every vertex of every polygon lies inside the region.
- ``clamp_within(region, polygons) -> List[LayerPolygon]``
      Clamp every vertex into the region. Returns NEW polygons; a route that
      is already wholly inside is returned with byte-identical vertices.
- ``has_negative_y`` / ``escapes_top`` — targeted escape detectors used by the
  R134 post-mortem (negative-y vias, over-top climbs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .backend import (
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    Point,
    Polygon,
)
from .cell_netlist import PinKind

# Reusable bbox key sets so terminal extraction agrees with cell_netlist's
# device slicing without copying its private helpers verbatim.


def _poly_key(pdk_layers: LayerMap) -> Tuple[int, int]:
    g = pdk_layers.gds_layer(LogicalLayer.POLY)
    if g is None:
        raise ValueError("PDK must define POLY layer for terminal extraction")
    return (g.layer, g.datatype)


def _diff_keys(pdk_layers: LayerMap) -> List[Tuple[int, int]]:
    keys: List[Tuple[int, int]] = []
    for ll in (LogicalLayer.NDIFF, LogicalLayer.PDIFF):
        g = pdk_layers.gds_layer(ll)
        if g is not None and (g.layer, g.datatype) not in keys:
            keys.append((g.layer, g.datatype))
    return keys


def _bbox_of(polygon: Polygon) -> Tuple[int, int, int, int]:
    xs = [v.x for v in polygon.vertices]
    ys = [v.y for v in polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _centroid_of(polygon: Polygon) -> Point:
    """Vertex-average centroid of a polygon, integer-rounded.

    For the axis-aligned rectangles this module deals with, the vertex average
    equals the bbox center, which is strictly interior. Using the explicit
    vertex average (rather than only the bbox center) keeps the point honest
    for any convex device polygon, not just rectangles.
    """
    verts = polygon.vertices
    n = len(verts)
    if n == 0:
        raise ValueError("cannot take centroid of empty polygon")
    sx = sum(v.x for v in verts)
    sy = sum(v.y for v in verts)
    return Point(sx // n, sy // n)


# --- point-in-polygon -------------------------------------------------------


def point_in_polygon(pt: Point, polygon: Polygon) -> bool:
    """Even-odd ray-cast point-in-polygon test; boundary counts as inside.

    Used by tests to assert an extracted terminal lies INSIDE the device
    polygon it came from (not on the whole-cell corner).
    """
    verts = polygon.vertices
    n = len(verts)
    if n < 3:
        return False
    x, y = pt.x, pt.y
    # Boundary check first (treat on-edge as inside).
    for i in range(n):
        ax, ay = verts[i].x, verts[i].y
        bx, by = verts[(i + 1) % n].x, verts[(i + 1) % n].y
        # On-segment test for axis-aligned + general edges.
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if cross == 0 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
    inside = False
    j = n - 1
    for i in range(n):
        yi, yj = verts[i].y, verts[j].y
        xi, xj = verts[i].x, verts[j].x
        if (yi > y) != (yj > y):
            x_int = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


# --- terminal extraction ----------------------------------------------------


def device_poly_fingers(
    device_polys: List[LayerPolygon], pdk_layers: LayerMap
) -> List[LayerPolygon]:
    """The poly-finger LayerPolygons of one device, sorted left-to-right."""
    key = _poly_key(pdk_layers)
    fingers = [lp for lp in device_polys if (lp.layer, lp.datatype) == key]
    return sorted(fingers, key=lambda lp: _bbox_of(lp.polygon)[0])


def device_diffusion(
    device_polys: List[LayerPolygon], pdk_layers: LayerMap
) -> Optional[LayerPolygon]:
    """The (first / widest) diffusion LayerPolygon of one device, if any."""
    keys = _diff_keys(pdk_layers)
    diffs = [lp for lp in device_polys if (lp.layer, lp.datatype) in keys]
    if not diffs:
        return None
    # The device's single diffusion rectangle spans all its fingers; pick the
    # widest in case stray rectangles slipped in.
    return max(diffs, key=lambda lp: _bbox_of(lp.polygon)[2] - _bbox_of(lp.polygon)[0])


def terminal_point(
    device_polys: List[LayerPolygon],
    kind: PinKind,
    pdk_layers: LayerMap,
) -> Point:
    """Return a terminal point that lies INSIDE the device's relevant polygon.

    - ``GATE``   -> centroid of the device's MIDDLE poly finger (the channel
      gate), i.e. an on-poly point, never a corner of the cell.
    - ``DRAIN``  -> a point inside the diffusion, on its UPPER half (so a drain
      net and a source net access the same diffusion at distinct, interior
      y-levels and can route without coinciding).
    - ``SOURCE`` -> a point inside the diffusion, on its LOWER half.
    - ``BULK``   -> centroid of the diffusion (bulk tie sits on the body).

    Every returned point is interior to a real device polygon — verified by the
    accompanying tests with ``point_in_polygon``. No coordinate is invented.
    """
    fingers = device_poly_fingers(device_polys, pdk_layers)
    diff = device_diffusion(device_polys, pdk_layers)

    if kind == PinKind.GATE:
        if not fingers:
            raise ValueError("device has no poly fingers; cannot place a gate terminal")
        # Middle finger is the representative gate; its centroid is interior.
        mid = fingers[len(fingers) // 2]
        return _centroid_of(mid.polygon)

    if diff is None:
        raise ValueError(
            "device has no diffusion; cannot place a source/drain/bulk terminal"
        )
    dx_min, dy_min, dx_max, dy_max = _bbox_of(diff.polygon)
    cx = (dx_min + dx_max) // 2

    if kind == PinKind.BULK:
        return _centroid_of(diff.polygon)

    # Source / drain: interior points in the lower / upper quarter of the
    # diffusion height. These are strictly inside the diffusion rectangle
    # (between dy_min and dy_max), not on the boundary edge.
    if kind == PinKind.DRAIN:
        y = dy_min + (3 * (dy_max - dy_min)) // 4
    else:  # SOURCE
        y = dy_min + (dy_max - dy_min) // 4
    # Guard against zero-height degenerate diffusion by nudging inside if equal
    # to an extreme (keeps the point off the boundary when height >= 2).
    if y == dy_min and dy_max > dy_min:
        y = dy_min + 1
    if y == dy_max and dy_max > dy_min:
        y = dy_max - 1
    return Point(cx, y)


# --- allowed region + containment -------------------------------------------


@dataclass(frozen=True, eq=True)
class Region:
    """An axis-aligned allowed region (closed interval on both axes)."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def contains(self, pt: Point) -> bool:
        return self.x_min <= pt.x <= self.x_max and self.y_min <= pt.y <= self.y_max

    def clamp_point(self, pt: Point) -> Point:
        return Point(
            min(max(pt.x, self.x_min), self.x_max),
            min(max(pt.y, self.y_min), self.y_max),
        )


def expand(region: Region, margin: int) -> Region:
    """Grow a region uniformly by ``margin`` (the allowed routing slack)."""
    return Region(
        region.x_min - margin,
        region.y_min - margin,
        region.x_max + margin,
        region.y_max + margin,
    )


def region_from_bbox(
    bbox: Tuple[int, int, int, int], margin: int = 0
) -> Region:
    """Build an allowed region from a (x_min, y_min, x_max, y_max) bbox."""
    x_min, y_min, x_max, y_max = bbox
    base = Region(x_min, y_min, x_max, y_max)
    return expand(base, margin) if margin else base


def region_from_cell(cell: NamedCell, margin: int = 0) -> Optional[Region]:
    """Build an allowed region from a placed cell's polygon bbox + margin."""
    xs = [v.x for lp in cell.polygons for v in lp.polygon.vertices]
    ys = [v.y for lp in cell.polygons for v in lp.polygon.vertices]
    if not xs:
        return None
    return region_from_bbox((min(xs), min(ys), max(xs), max(ys)), margin)


def verify_within(region: Region, polygons: List[LayerPolygon]) -> bool:
    """True iff every vertex of every polygon lies inside ``region``.

    This is the route-containment assertion: a routed net is contained iff no
    wire/via vertex escapes the cell bbox + margin.
    """
    for lp in polygons:
        for v in lp.polygon.vertices:
            if not region.contains(v):
                return False
    return True


def out_of_region_vertices(
    region: Region, polygons: List[LayerPolygon]
) -> List[Tuple[int, int]]:
    """List the (x, y) of every vertex that falls outside ``region``.

    Useful for diagnostics / asserting WHICH escape (negative-y via,
    over-top climb) occurred.
    """
    bad: List[Tuple[int, int]] = []
    for lp in polygons:
        for v in lp.polygon.vertices:
            if not region.contains(v):
                bad.append((v.x, v.y))
    return bad


def has_negative_y(polygons: List[LayerPolygon], floor: int = 0) -> bool:
    """True iff any vertex sits below ``floor`` (the R134 via12 at y=-133)."""
    return any(
        v.y < floor for lp in polygons for v in lp.polygon.vertices
    )


def escapes_top(polygons: List[LayerPolygon], ceiling: int) -> bool:
    """True iff any vertex rises above ``ceiling`` (the R134 M2 at y=8048)."""
    return any(
        v.y > ceiling for lp in polygons for v in lp.polygon.vertices
    )


def clamp_within(
    region: Region, polygons: List[LayerPolygon]
) -> List[LayerPolygon]:
    """Clamp every vertex of every polygon into ``region``; return NEW polygons.

    Regression guarantee: a polygon already wholly inside ``region`` is copied
    with byte-identical vertex coordinates (the clamp is the identity on
    in-region points), so a legal in-bbox route round-trips unchanged. Only
    escaping vertices are pulled back onto the region boundary.

    This never *grows* a route or invents geometry; it only retracts vertices
    that already violate the allowed region. Pair with ``verify_within`` to
    decide whether clamping was needed at all.
    """
    out: List[LayerPolygon] = []
    for lp in polygons:
        new_verts = [region.clamp_point(v) for v in lp.polygon.vertices]
        out.append(
            LayerPolygon(
                layer=lp.layer,
                datatype=lp.datatype,
                polygon=Polygon(vertices=new_verts),
            )
        )
    return out


__all__ = [
    "Region",
    "expand",
    "region_from_bbox",
    "region_from_cell",
    "point_in_polygon",
    "device_poly_fingers",
    "device_diffusion",
    "terminal_point",
    "verify_within",
    "out_of_region_vertices",
    "has_negative_y",
    "escapes_top",
    "clamp_within",
]
