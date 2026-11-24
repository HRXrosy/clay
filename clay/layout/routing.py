"""Phase 8a — M1 wire segment routing primitive.

Builds M1 wire segments connecting two points on a layout. Phase 8a is
a baseline: orthogonal Manhattan routing (one bend max), single layer
(M1 only), no via insertion, no congestion check. Layer-correct
geometry only.

API:
- WireSegment dataclass: (start_x, start_y, end_x, end_y, width_nm)
- route_m1_segment(start, end, width_nm, pdk_layers) -> List[LayerPolygon]
- route_m1_manhattan(start, end, width_nm, pdk_layers, prefer_horizontal_first=True)
  -> List[LayerPolygon] (one bend, up to 2 rectangles)

Each polygon is a rectangle on the M1 layer; layer/datatype looked up
via pdk_layers.gds_layer(LogicalLayer.METAL1).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

from .backend import (
    Point, Polygon, LayerPolygon, LayerMap, LogicalLayer,
)


@dataclass(frozen=True, eq=True)
class WireSegment:
    """An axis-aligned M1 wire segment from (start_x, start_y) to (end_x, end_y)
    with the given width in nm DBU. Must be horizontal or vertical (one
    coordinate identical)."""
    start_x: int
    start_y: int
    end_x: int
    end_y: int
    width_nm: int

    def is_horizontal(self) -> bool:
        return self.start_y == self.end_y

    def is_vertical(self) -> bool:
        return self.start_x == self.end_x


def _segment_to_polygon(seg: WireSegment) -> Polygon:
    """Convert a wire segment to a polygon rectangle.

    Horizontal: width is in y-direction. Vertical: width is in x-direction.
    Raises ValueError on diagonal segments.
    """
    half_w = seg.width_nm // 2
    if seg.is_horizontal():
        x_min = min(seg.start_x, seg.end_x)
        x_max = max(seg.start_x, seg.end_x)
        y = seg.start_y
        return Polygon.rectangle(x_min, y - half_w, x_max, y + half_w)
    if seg.is_vertical():
        y_min = min(seg.start_y, seg.end_y)
        y_max = max(seg.start_y, seg.end_y)
        x = seg.start_x
        return Polygon.rectangle(x - half_w, y_min, x + half_w, y_max)
    raise ValueError(
        f"WireSegment must be axis-aligned (horizontal or vertical), got "
        f"diagonal from ({seg.start_x},{seg.start_y}) to ({seg.end_x},{seg.end_y})"
    )


def route_m1_segment(
    start: Point,
    end: Point,
    width_nm: int,
    pdk_layers: LayerMap,
) -> List[LayerPolygon]:
    """Route a single axis-aligned M1 wire segment from start to end.

    Raises ValueError if start and end are not on the same x or y axis.
    """
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    seg = WireSegment(start.x, start.y, end.x, end.y, width_nm)
    if not (seg.is_horizontal() or seg.is_vertical()):
        raise ValueError(
            f"route_m1_segment requires axis-aligned endpoints; got "
            f"({start.x},{start.y}) -> ({end.x},{end.y}) — use route_m1_manhattan instead"
        )
    metal1 = pdk_layers.gds_layer(LogicalLayer.METAL1)
    if metal1 is None:
        raise ValueError("PDK does not define METAL1 layer")
    return [LayerPolygon(
        layer=metal1.layer, datatype=metal1.datatype,
        polygon=_segment_to_polygon(seg),
    )]


def route_m1_manhattan(
    start: Point,
    end: Point,
    width_nm: int,
    pdk_layers: LayerMap,
    prefer_horizontal_first: bool = True,
) -> List[LayerPolygon]:
    """Route a Manhattan (single-bend) M1 path from start to end.

    Two rectangles: one horizontal + one vertical. If start and end are
    already axis-aligned, returns a single rectangle (degenerate to
    route_m1_segment behavior).

    Args:
        prefer_horizontal_first: if True, route horizontal then vertical
            (bend at (end.x, start.y)); if False, vertical then horizontal
            (bend at (start.x, end.y))
    """
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    metal1 = pdk_layers.gds_layer(LogicalLayer.METAL1)
    if metal1 is None:
        raise ValueError("PDK does not define METAL1 layer")

    # Degenerate: already axis-aligned
    if start.x == end.x or start.y == end.y:
        return route_m1_segment(start, end, width_nm, pdk_layers)

    bend = (
        Point(end.x, start.y) if prefer_horizontal_first
        else Point(start.x, end.y)
    )
    seg1 = WireSegment(start.x, start.y, bend.x, bend.y, width_nm)
    seg2 = WireSegment(bend.x, bend.y, end.x, end.y, width_nm)
    return [
        LayerPolygon(
            layer=metal1.layer, datatype=metal1.datatype,
            polygon=_segment_to_polygon(seg1),
        ),
        LayerPolygon(
            layer=metal1.layer, datatype=metal1.datatype,
            polygon=_segment_to_polygon(seg2),
        ),
    ]
