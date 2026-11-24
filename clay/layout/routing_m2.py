"""Phase 8c - M2 routing primitives (mirror of Phase 8a/8b for M2 layer).

Used when M1 routing is congested or for vertical escape over multiple
M1 wires. Identical semantics to M1 routing but emits on M2 layer.
"""
from __future__ import annotations
from typing import List

from .backend import Point, Polygon, LayerPolygon, LayerMap
from .routing import WireSegment, _segment_to_polygon as _seg_to_poly
from .routing_advanced import metal2_layer_for_pdk_layers


def route_m2_segment(
    start: Point,
    end: Point,
    width_nm: int,
    pdk_layers: LayerMap,
) -> List[LayerPolygon]:
    """Route a single axis-aligned M2 wire segment from start to end."""
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    seg = WireSegment(start.x, start.y, end.x, end.y, width_nm)
    if not (seg.is_horizontal() or seg.is_vertical()):
        raise ValueError(
            f"route_m2_segment requires axis-aligned endpoints; got "
            f"({start.x},{start.y}) -> ({end.x},{end.y})"
        )
    m2_layer, m2_dt = metal2_layer_for_pdk_layers(pdk_layers)
    return [LayerPolygon(
        layer=m2_layer, datatype=m2_dt,
        polygon=_seg_to_poly(seg),
    )]


def route_m2_manhattan(
    start: Point,
    end: Point,
    width_nm: int,
    pdk_layers: LayerMap,
    prefer_horizontal_first: bool = True,
) -> List[LayerPolygon]:
    """Route a single-bend Manhattan M2 path from start to end."""
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    m2_layer, m2_dt = metal2_layer_for_pdk_layers(pdk_layers)

    if start.x == end.x or start.y == end.y:
        return route_m2_segment(start, end, width_nm, pdk_layers)

    bend = (
        Point(end.x, start.y) if prefer_horizontal_first
        else Point(start.x, end.y)
    )
    seg1 = WireSegment(start.x, start.y, bend.x, bend.y, width_nm)
    seg2 = WireSegment(bend.x, bend.y, end.x, end.y, width_nm)
    return [
        LayerPolygon(layer=m2_layer, datatype=m2_dt, polygon=_seg_to_poly(seg1)),
        LayerPolygon(layer=m2_layer, datatype=m2_dt, polygon=_seg_to_poly(seg2)),
    ]


def route_m2_multibend(
    waypoints: List[Point],
    width_nm: int,
    pdk_layers: LayerMap,
) -> List[LayerPolygon]:
    """Route an M2 polyline along axis-aligned waypoints."""
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    if len(waypoints) < 2:
        raise ValueError(
            f"route_m2_multibend requires >=2 waypoints, got {len(waypoints)}"
        )
    m2_layer, m2_dt = metal2_layer_for_pdk_layers(pdk_layers)

    polys: List[LayerPolygon] = []
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        seg = WireSegment(a.x, a.y, b.x, b.y, width_nm)
        if not (seg.is_horizontal() or seg.is_vertical()):
            raise ValueError(
                f"waypoints[{i}]->({i+1}) is diagonal: "
                f"({a.x},{a.y})->({b.x},{b.y})"
            )
        polys.append(LayerPolygon(
            layer=m2_layer, datatype=m2_dt,
            polygon=_seg_to_poly(seg),
        ))
    return polys
