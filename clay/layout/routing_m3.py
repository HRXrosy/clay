"""Phase 10 — M3 routing + via23 primitives.

Extends Phase 8 routing stack (M1 in 8a/8b, M2 in 8c) to M3 for
power/clock distribution networks. via23 stack connects M2 to M3.

Hardcoded sky130/gf180mcu layer numbers (LogicalLayer enum is fixed
at backend.py level; not extended per Sec 3.1 preservation).
"""
from __future__ import annotations
from typing import List, Tuple

from .backend import Point, Polygon, LayerPolygon, LayerMap
from .routing import WireSegment, _segment_to_polygon as _seg_to_poly
from .routing_advanced import metal2_layer_for_pdk_layers


def metal3_layer_for_pdk_layers(pdk_layers: LayerMap) -> Tuple[int, int]:
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return (70, 20)
    if "gf180" in name:
        return (38, 0)
    raise ValueError(
        f"unknown PDK LayerMap type: {type(pdk_layers).__name__} for metal3 lookup"
    )


def via23_layer_for_pdk_layers(pdk_layers: LayerMap) -> Tuple[int, int]:
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return (70, 44)
    if "gf180" in name:
        return (37, 0)
    raise ValueError(
        f"unknown PDK LayerMap type: {type(pdk_layers).__name__} for via23 lookup"
    )


def route_m3_segment(
    start: Point,
    end: Point,
    width_nm: int,
    pdk_layers: LayerMap,
) -> List[LayerPolygon]:
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    seg = WireSegment(start.x, start.y, end.x, end.y, width_nm)
    if not (seg.is_horizontal() or seg.is_vertical()):
        raise ValueError(
            f"route_m3_segment requires axis-aligned endpoints; got "
            f"({start.x},{start.y}) -> ({end.x},{end.y})"
        )
    m3_layer, m3_dt = metal3_layer_for_pdk_layers(pdk_layers)
    return [LayerPolygon(layer=m3_layer, datatype=m3_dt, polygon=_seg_to_poly(seg))]


def route_m3_manhattan(
    start: Point,
    end: Point,
    width_nm: int,
    pdk_layers: LayerMap,
    prefer_horizontal_first: bool = True,
) -> List[LayerPolygon]:
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    m3_layer, m3_dt = metal3_layer_for_pdk_layers(pdk_layers)
    if start.x == end.x or start.y == end.y:
        return route_m3_segment(start, end, width_nm, pdk_layers)
    bend = Point(end.x, start.y) if prefer_horizontal_first else Point(start.x, end.y)
    seg1 = WireSegment(start.x, start.y, bend.x, bend.y, width_nm)
    seg2 = WireSegment(bend.x, bend.y, end.x, end.y, width_nm)
    return [
        LayerPolygon(layer=m3_layer, datatype=m3_dt, polygon=_seg_to_poly(seg1)),
        LayerPolygon(layer=m3_layer, datatype=m3_dt, polygon=_seg_to_poly(seg2)),
    ]


def route_m3_multibend(
    waypoints: List[Point],
    width_nm: int,
    pdk_layers: LayerMap,
) -> List[LayerPolygon]:
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    if len(waypoints) < 2:
        raise ValueError(
            f"route_m3_multibend requires >=2 waypoints, got {len(waypoints)}"
        )
    m3_layer, m3_dt = metal3_layer_for_pdk_layers(pdk_layers)
    polys: List[LayerPolygon] = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        seg = WireSegment(a.x, a.y, b.x, b.y, width_nm)
        if not (seg.is_horizontal() or seg.is_vertical()):
            raise ValueError(
                f"waypoints[{i}]->({i+1}) is diagonal: "
                f"({a.x},{a.y})->({b.x},{b.y})"
            )
        polys.append(LayerPolygon(
            layer=m3_layer, datatype=m3_dt, polygon=_seg_to_poly(seg),
        ))
    return polys


def insert_via23(
    center: Point,
    size_nm: int,
    pdk_layers: LayerMap,
    landing_pad_nm: int = None,
) -> List[LayerPolygon]:
    """3-polygon M2+via23+M3 stack at the given point."""
    if size_nm <= 0:
        raise ValueError(f"size_nm must be positive, got {size_nm}")
    if landing_pad_nm is None:
        landing_pad_nm = size_nm + 80
    if landing_pad_nm < size_nm:
        raise ValueError(
            f"landing_pad_nm ({landing_pad_nm}) must be >= size_nm ({size_nm})"
        )

    m2_layer, m2_dt = metal2_layer_for_pdk_layers(pdk_layers)
    via23_layer, via23_dt = via23_layer_for_pdk_layers(pdk_layers)
    m3_layer, m3_dt = metal3_layer_for_pdk_layers(pdk_layers)

    half_via = size_nm // 2
    half_pad = landing_pad_nm // 2

    via = Polygon.rectangle(
        center.x - half_via, center.y - half_via,
        center.x + half_via, center.y + half_via,
    )
    m2_pad = Polygon.rectangle(
        center.x - half_pad, center.y - half_pad,
        center.x + half_pad, center.y + half_pad,
    )
    m3_pad = Polygon.rectangle(
        center.x - half_pad, center.y - half_pad,
        center.x + half_pad, center.y + half_pad,
    )

    return [
        LayerPolygon(layer=m2_layer, datatype=m2_dt, polygon=m2_pad),
        LayerPolygon(layer=via23_layer, datatype=via23_dt, polygon=via),
        LayerPolygon(layer=m3_layer, datatype=m3_dt, polygon=m3_pad),
    ]


def route_m2_to_m3(
    start: Point,
    end: Point,
    width_nm: int,
    via_size_nm: int,
    pdk_layers: LayerMap,
    landing_pad_nm: int = None,
) -> List[LayerPolygon]:
    """Cross-layer route: via23 stack at start + M3 segment to end."""
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    m3_layer, m3_dt = metal3_layer_for_pdk_layers(pdk_layers)
    stack = insert_via23(start, via_size_nm, pdk_layers, landing_pad_nm)
    seg = WireSegment(start.x, start.y, end.x, end.y, width_nm)
    if not (seg.is_horizontal() or seg.is_vertical()):
        raise ValueError(
            f"route_m2_to_m3 requires axis-aligned start/end; got "
            f"({start.x},{start.y})->({end.x},{end.y})"
        )
    m3_seg = LayerPolygon(layer=m3_layer, datatype=m3_dt, polygon=_seg_to_poly(seg))
    return stack + [m3_seg]
