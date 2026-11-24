"""R137 (BL-137A) — reroute-to-contain instead of clamp-to-truncate.

R135's :func:`clay.layout.route_containment.clamp_within` keeps a route inside
the cell's allowed region by pulling every *vertex* back onto the region
boundary. That is correct for containment but destroys wire shape: when an
escaping wire's two end-vertices are clamped onto different region edges, the
rectangle collapses to a thin sliver. R136's layout DRC
(:func:`clay.layout.layout_drc.run_layout_drc`) caught exactly that — two 48 nm
M1 stubs against the real 140 nm sky130 M1 min-width — because clamping a wire
that escapes (e.g. an x-traveling wire pushed back in x) leaves only the
half-width band, which is below the layer minimum.

This module routes the *same* pin-to-pin connection so that every emitted wire
lies entirely inside the region AT FULL WIDTH. Instead of truncating vertices,
it chooses a Manhattan path whose centre-lines are kept far enough from the
region edges that the half-width rectangle inflation never crosses them. The
narrow dimension of every returned wire is therefore exactly ``wire_width_nm``
(never a clamped sub-width stub).

Design
------
- A wire centre-line at ``x`` inflates to ``[x - w//2, x + w//2]``. For the
  rectangle to sit inside the region the centre-line must stay within an
  *inset* region shrunk by ``half_w = width_nm // 2`` on every side. So
  containment of full-width wires == containment of centre-lines in the inset
  region. We route entirely on centre-lines inside that inset region.
- Endpoints are assumed routable: if a raw endpoint sits in the boundary band
  (within ``half_w`` of an edge) its centre-line is nudged onto the inset edge
  so its own wire stub stays inside. This moves the connection point by at most
  ``half_w`` — it does not fabricate a wire outside the region.
- The path is a single-bend L. Because the inset region is a convex
  axis-aligned rectangle that contains both (clamped) endpoints, at least one
  of the two L-bend corners ``(sb.x, sa.y)`` / ``(sa.x, sb.y)`` is also inside
  it — both coordinates of each corner come from an interior endpoint. So a
  contained single-bend route always exists; we pick the preferred bend when it
  is interior (reproducing the plain router byte-for-byte in the common
  in-region case) and otherwise the alternate bend.

The actual rectangles are emitted by the unmodified
:func:`clay.layout.routing.route_m1_segment` /
:func:`clay.layout.routing.route_m1_manhattan` so layer/datatype and full-width
geometry come from the committed router, not reinvented here. ``route_m1_*``
is imported read-only; ``Region`` / ``verify_within`` come from the committed
``route_containment`` module read-only.
"""
from __future__ import annotations

from typing import List, Optional

from .backend import (
    LayerMap,
    LayerPolygon,
    Point,
)
from .route_containment import Region, verify_within
from .routing import route_m1_manhattan, route_m1_segment


def _inset(region: Region, half_w: int) -> Region:
    """Shrink ``region`` by ``half_w`` on every side (centre-line region).

    A full-width wire whose centre-line lies inside this inset region inflates
    to a rectangle that is still inside the original ``region``.
    """
    return Region(
        region.x_min + half_w,
        region.y_min + half_w,
        region.x_max - half_w,
        region.y_max - half_w,
    )


def _clamp_to(region: Region, pt: Point) -> Point:
    return region.clamp_point(pt)


def _bend_inside(inset: Region, bend: Point) -> bool:
    return inset.contains(bend)


def route_contained(
    a: Point,
    b: Point,
    width_nm: int,
    region: Region,
    pdk_layers: LayerMap,
    *,
    prefer_horizontal_first: bool = True,
) -> List[LayerPolygon]:
    """Route ``a`` -> ``b`` as full-width M1 wires entirely inside ``region``.

    Unlike clamp-to-region (which truncates escaping vertices into sub-width
    stubs), this re-routes the connection so EVERY returned wire is inside
    ``region`` and keeps its full ``width_nm`` narrow dimension.

    Args:
        a, b: the two pin terminal points to connect.
        width_nm: the routed wire width (the narrow dimension of every wire).
        region: the allowed rectangular region (cell bbox + margin).
        pdk_layers: the PDK LayerMap (for the M1 GDS layer).
        prefer_horizontal_first: bend preference, matching ``route_m1_manhattan``.
            When the preferred single-bend route is already fully contained, the
            output is byte-identical to ``route_m1_manhattan`` (no detour).

    Returns:
        A list of M1 LayerPolygons whose union connects ``a`` to ``b``, every
        one contained in ``region`` at full width.

    Raises:
        ValueError: if ``width_nm`` <= 0, or the region is too small to hold a
            full-width wire (inset region empty on an axis).
    """
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    half_w = width_nm // 2
    inset = _inset(region, half_w)
    if inset.x_min > inset.x_max or inset.y_min > inset.y_max:
        raise ValueError(
            f"region {region} too small to hold a {width_nm} nm full-width wire "
            f"(inset collapses to empty); widen the region or narrow the wire"
        )

    # Pull each endpoint's centre-line into the inset region so its own wire
    # stays inside. For an in-region pair this is the identity.
    sa = _clamp_to(inset, a)
    sb = _clamp_to(inset, b)

    # Degenerate: same point -> a single dot-segment (full width) at sa.
    if sa == sb:
        return route_m1_segment(sa, sb, width_nm, pdk_layers)

    # Axis-aligned: a single straight full-width segment, always inside inset.
    if sa.x == sb.x or sa.y == sb.y:
        return route_m1_segment(sa, sb, width_nm, pdk_layers)

    # Try the two single-bend L-routes; use one whose bend corner is inside the
    # inset region. The preferred order reproduces route_m1_manhattan exactly
    # when that bend is already contained (the no-needless-detour guarantee).
    pref_bend = (
        Point(sb.x, sa.y) if prefer_horizontal_first else Point(sa.x, sb.y)
    )
    if _bend_inside(inset, pref_bend):
        return route_m1_manhattan(
            sa, sb, width_nm, pdk_layers,
            prefer_horizontal_first=prefer_horizontal_first,
        )
    # The alternate bend is interior whenever the preferred one is not: both
    # corner coordinates are drawn from sa/sb, which both lie in the convex
    # inset rectangle, so one of the two corners is always contained.
    return route_m1_manhattan(
        sa, sb, width_nm, pdk_layers,
        prefer_horizontal_first=not prefer_horizontal_first,
    )


def route_contained_or_none(
    a: Point,
    b: Point,
    width_nm: int,
    region: Region,
    pdk_layers: LayerMap,
    **kwargs,
) -> Optional[List[LayerPolygon]]:
    """Like :func:`route_contained` but returns ``None`` instead of raising when
    the region is too small to hold a full-width wire.

    Useful for callers that want to skip an unroutable net rather than fail the
    whole lowering. A successful route is always fully contained
    (``verify_within`` True at full width).
    """
    try:
        return route_contained(a, b, width_nm, region, pdk_layers, **kwargs)
    except ValueError:
        return None


__all__ = [
    "route_contained",
    "route_contained_or_none",
]
