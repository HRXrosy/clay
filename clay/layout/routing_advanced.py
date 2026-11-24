"""Phase 8b — multi-bend M1 routing + via12 insertion + M1<->M2 cross-layer routes.

Extends Phase 8a single/one-bend primitives with:
- route_m1_multibend: arbitrary-length axis-aligned polyline route
- insert_via12: M1+via+M2 stack at a point
- route_m1_to_m2: cross-layer route via via12

PDK-specific layer numbers (via12 + M2) are hardcoded here for sky130 +
gf180mcu since the LogicalLayer enum is fixed at backend.py level
(Phase 1-3a). Adding new LogicalLayer variants is a layout-side change
that goes through future major-version Phase 8.x.
"""
from __future__ import annotations
from typing import List

from .backend import (
    Point, Polygon, LayerPolygon, LayerMap, LogicalLayer,
)
from .routing import WireSegment, _segment_to_polygon as _phase8a_seg_to_poly


# === Phase 8b-specific layer lookups (sky130 + gf180mcu) ===

def via12_layer_for_pdk_layers(pdk_layers: LayerMap):
    """Return (layer, datatype) tuple for via12 on sky130/gf180mcu.

    Raises ValueError on unknown PDK.
    """
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return (68, 44)
    if "gf180" in name:
        return (35, 0)
    raise ValueError(
        f"unknown PDK LayerMap type: {type(pdk_layers).__name__} for via12 lookup"
    )


def metal2_layer_for_pdk_layers(pdk_layers: LayerMap):
    """Return (layer, datatype) tuple for M2 on sky130/gf180mcu."""
    name = type(pdk_layers).__name__.lower()
    if "sky130" in name:
        return (68, 20)
    if "gf180" in name:
        return (36, 0)
    raise ValueError(
        f"unknown PDK LayerMap type: {type(pdk_layers).__name__} for metal2 lookup"
    )


# === Multi-bend M1 polyline route ===

def route_m1_multibend(
    waypoints: List[Point],
    width_nm: int,
    pdk_layers: LayerMap,
) -> List[LayerPolygon]:
    """Route an M1 polyline along axis-aligned waypoints.

    Each consecutive pair (waypoints[i], waypoints[i+1]) must share an
    x or y coordinate (axis-aligned). Returns one rectangle per segment.

    Raises:
        ValueError if width_nm <= 0, waypoints has < 2 points, or any
        consecutive pair is diagonal.
    """
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    if len(waypoints) < 2:
        raise ValueError(
            f"route_m1_multibend requires >=2 waypoints, got {len(waypoints)}"
        )
    metal1 = pdk_layers.gds_layer(LogicalLayer.METAL1)
    if metal1 is None:
        raise ValueError("PDK does not define METAL1 layer")

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
            layer=metal1.layer, datatype=metal1.datatype,
            polygon=_phase8a_seg_to_poly(seg),
        ))
    return polys


# === Via12 insertion ===

def insert_via12(
    center: Point,
    size_nm: int,
    pdk_layers: LayerMap,
    landing_pad_nm: int = None,
) -> List[LayerPolygon]:
    """Insert a M1+via+M2 stack at the given point.

    Args:
        center: center point of the via in DBU
        size_nm: square via side length (e.g. 170 for sky130 typical)
        pdk_layers: PDK LayerMap
        landing_pad_nm: M1/M2 landing pad side length (defaults to
            size_nm + 80 to give 40 nm enclosure on each side, which is
            sky130 typical M1/M2 enclosure of via)

    Returns 3 polygons: M1 pad + via12 + M2 pad.

    Raises ValueError on non-positive size_nm.
    """
    if size_nm <= 0:
        raise ValueError(f"size_nm must be positive, got {size_nm}")
    if landing_pad_nm is None:
        landing_pad_nm = size_nm + 80
    if landing_pad_nm < size_nm:
        raise ValueError(
            f"landing_pad_nm ({landing_pad_nm}) must be >= size_nm ({size_nm})"
        )

    metal1 = pdk_layers.gds_layer(LogicalLayer.METAL1)
    if metal1 is None:
        raise ValueError("PDK does not define METAL1 layer")
    via12_layer, via12_dt = via12_layer_for_pdk_layers(pdk_layers)
    m2_layer, m2_dt = metal2_layer_for_pdk_layers(pdk_layers)

    half_via = size_nm // 2
    half_pad = landing_pad_nm // 2

    via = Polygon.rectangle(
        center.x - half_via, center.y - half_via,
        center.x + half_via, center.y + half_via,
    )
    m1_pad = Polygon.rectangle(
        center.x - half_pad, center.y - half_pad,
        center.x + half_pad, center.y + half_pad,
    )
    m2_pad = Polygon.rectangle(
        center.x - half_pad, center.y - half_pad,
        center.x + half_pad, center.y + half_pad,
    )

    return [
        LayerPolygon(layer=metal1.layer, datatype=metal1.datatype, polygon=m1_pad),
        LayerPolygon(layer=via12_layer, datatype=via12_dt, polygon=via),
        LayerPolygon(layer=m2_layer, datatype=m2_dt, polygon=m2_pad),
    ]


# === M1<->M2 cross-layer route ===

def route_m1_to_m2(
    start: Point,
    end: Point,
    width_nm: int,
    via_size_nm: int,
    pdk_layers: LayerMap,
    landing_pad_nm: int = None,
) -> List[LayerPolygon]:
    """Cross-layer route from M1 start to M2 end via via12 transition.

    Strategy: place via12 stack at start (M1+via12+M2 pad), then route
    M2 wire from via12 center to end. If end is axis-aligned with start
    only one M2 segment; otherwise one M2 segment (currently single
    segment; multi-bend M2 routing is Phase 8c future).

    Raises ValueError if end is not axis-aligned with start (i.e. diagonal).
    """
    if width_nm <= 0:
        raise ValueError(f"width_nm must be positive, got {width_nm}")
    m2_layer, m2_dt = metal2_layer_for_pdk_layers(pdk_layers)

    # via12 stack at start
    stack = insert_via12(start, via_size_nm, pdk_layers, landing_pad_nm)

    # M2 segment from start to end (axis-aligned)
    seg = WireSegment(start.x, start.y, end.x, end.y, width_nm)
    if not (seg.is_horizontal() or seg.is_vertical()):
        raise ValueError(
            f"route_m1_to_m2 requires axis-aligned start/end; got "
            f"({start.x},{start.y})->({end.x},{end.y})"
        )
    m2_seg = LayerPolygon(
        layer=m2_layer, datatype=m2_dt,
        polygon=_phase8a_seg_to_poly(seg),
    )

    return stack + [m2_seg]
