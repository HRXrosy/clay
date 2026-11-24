"""R134 (BL-134A) — complete multi-net routed lowering.

R133's ``routed_lowering`` routed exactly ONE net (the current mirror's shared
gate) because ``AnnotatedCell`` had no net model. ``cell_netlist`` (R134)
supplies that model: gate / source / drain pins per device plus the cell's
real net topology. This module consumes it to route ALL of a cell's routable
nets — and, where nets must cross at the same planar level, escapes to M2/M3
through real via stacks so the crossing is layer-correct rather than a short.

Pipeline (opt-in; nothing existing is modified):

    cell --lower_fn--> placement-only ParameterizedLibrary   (byte-identical
                                                              to the non-routed
                                                              path)
         --build_cell_netlist--> explicit Net list (pins from placed geometry)
         --route each routable Net (REAL routers)--> wire + via LayerPolygons
         --merge into a COPY of the top cell--> routed ParameterizedLibrary

Every polygon merged into the routed library comes from a real router call:
``route_m1_manhattan`` / ``route_m1_multibend`` for in-plane M1, and
``route_m1_to_m2`` / ``route_m2_to_m3`` for layer-changing crossings. No wire
or via rectangle is hand-fabricated here, and no net connectivity is invented
— it all comes from ``cell_netlist``.

Discipline: this module imports ``routed_lowering`` (R133) and reuses its
``_deepcopy_library`` + ``RoutedNet`` types so the merge semantics stay
identical; it does NOT modify ``routed_lowering.py``, ``backend.py``, or any
router module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .annotations import AnnotatedCell
from .backend import (
    DeviceSizing,
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    ParameterizedLibrary,
    Point,
    lower_annotated_cell_to_parameterized,
)
from .cell_netlist import CellNetlist, Net, Pin, build_cell_netlist
from .routing import route_m1_manhattan
from .routing_advanced import (
    metal2_layer_for_pdk_layers,
    route_m1_multibend,
    route_m1_to_m2,
)
from .routing_m3 import metal3_layer_for_pdk_layers, route_m2_to_m3
from .routed_lowering import _deepcopy_library


# --- routed-net record (one per routable net) ------------------------------


@dataclass
class FullRoutedNet:
    """A fully routed net: its name, the pins it connected, the wires/vias the
    real routers produced, and which routing layers those polygons landed on."""

    name: str
    devices: Tuple[str, ...]
    pin_count: int
    wires: List[LayerPolygon] = field(default_factory=list)
    used_layers: List[Tuple[int, int]] = field(default_factory=list)
    crossed: bool = False  # whether a layer-change (via) was used

    def layer_set(self) -> List[Tuple[int, int]]:
        seen: List[Tuple[int, int]] = []
        for w in self.wires:
            key = (w.layer, w.datatype)
            if key not in seen:
                seen.append(key)
        return seen


@dataclass
class FullRoutedResult:
    """Result of full routed lowering."""

    library: ParameterizedLibrary
    placement_only: ParameterizedLibrary
    netlist: CellNetlist
    routed_nets: List[FullRoutedNet] = field(default_factory=list)

    def routed_net_names(self) -> List[str]:
        return [n.name for n in self.routed_nets]

    def total_wire_polygons(self) -> int:
        return sum(len(n.wires) for n in self.routed_nets)

    def routing_layer_polygon_count(self, layer: int, datatype: int) -> int:
        top = self.library.cells[-1] if self.library.cells else None
        if top is None:
            return 0
        return sum(
            1 for lp in top.polygons if lp.layer == layer and lp.datatype == datatype
        )


# --- geometric helpers ------------------------------------------------------


def _hspan_of_wires(wires: List[LayerPolygon]) -> List[Tuple[int, int, int]]:
    """For each wire polygon, return (y, x_min, x_max) if it is a horizontal
    run (constant y band). Used for crossing detection against rails."""
    out: List[Tuple[int, int, int]] = []
    for w in wires:
        xs = [v.x for v in w.polygon.vertices]
        ys = [v.y for v in w.polygon.vertices]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # Treat a thin band (height small relative to width) as horizontal.
        if (x_max - x_min) >= (y_max - y_min):
            out.append(((y_min + y_max) // 2, x_min, x_max))
    return out


def _segment_crosses_band(
    sx: int, sy: int, ex: int, ey: int, bands: List[Tuple[int, int, int]]
) -> bool:
    """Does the axis-aligned segment (sx,sy)->(ex,ey) cross any horizontal
    band (y, x_min, x_max)? Only vertical segments can cross a horizontal
    rail at a different y, which is the case we must lift onto another layer."""
    if sx != ex:
        return False  # horizontal segment: handled by routing on its own y
    y_lo, y_hi = (sy, ey) if sy <= ey else (ey, sy)
    for (by, bx_min, bx_max) in bands:
        if bx_min <= sx <= bx_max and y_lo < by < y_hi:
            return True
    return False


# --- per-net routing --------------------------------------------------------


def _route_in_plane_segment(
    a: Point, b: Point, width_nm: int, pdk_layers: LayerMap
) -> List[LayerPolygon]:
    """Route one pin-to-pin connection on M1 (single-bend Manhattan).

    If a and b are axis-aligned this is one rectangle; otherwise two (the
    real ``route_m1_manhattan`` output). Vertical-first when the two pins
    differ in y so the vertical run is explicit (needed for crossing checks).
    """
    if a.x == b.x or a.y == b.y:
        return route_m1_manhattan(a, b, width_nm, pdk_layers)
    # vertical-then-horizontal: bend at (a.x, b.y) keeps the vertical leg on
    # a.x so a later crossing check sees an explicit vertical segment there.
    return route_m1_manhattan(
        a, b, width_nm, pdk_layers, prefer_horizontal_first=False
    )


def _route_crossing_segment(
    a: Point,
    b: Point,
    width_nm: int,
    via_size_nm: int,
    pdk_layers: LayerMap,
) -> List[LayerPolygon]:
    """Route a pin-to-pin connection that must cross a rail, by lifting onto
    M2 (and, for a second crossing, M3) through real via stacks.

    Strategy: go vertical on M1 from ``a`` up to ``b.y`` would short the rail,
    so instead jump M1->M2 at ``a`` (``route_m1_to_m2`` gives via12 + an M2
    run along a's column to b.y), then jump M2->M3 (``route_m2_to_m3``) for the
    final horizontal hop to ``b`` so the M3 wire bridges over everything below.
    All geometry is produced by the real cross-layer routers.
    """
    # Leg 1: M1 -> M2 vertical escape along a.x from a.y to b.y.
    via_corner = Point(a.x, b.y)
    leg1 = route_m1_to_m2(a, via_corner, width_nm, via_size_nm, pdk_layers)
    # Leg 2: M2 -> M3 horizontal hop along b.y from a.x to b.x, bridging over.
    leg2 = route_m2_to_m3(via_corner, b, width_nm, via_size_nm, pdk_layers)
    return leg1 + leg2


def _cell_bbox(top: NamedCell) -> Optional[Tuple[int, int, int, int]]:
    xs = [v.x for lp in top.polygons for v in lp.polygon.vertices]
    ys = [v.y for lp in top.polygons for v in lp.polygon.vertices]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _route_rail_strap(
    net: Net,
    width_nm: int,
    pdk_layers: LayerMap,
    strap_y: int,
    x_min: int,
    x_max: int,
) -> List[LayerPolygon]:
    """Route a power rail as a continuous full-width M1 strap at ``strap_y``,
    plus M1 drops from each source pin up/down to the strap.

    The strap x-span (x_min..x_max) and y-level come from the placed cell's
    bbox + source-pin geometry — not invented. Real router calls:
    ``route_m1_multibend`` for the strap spine, ``route_m1_manhattan`` for the
    per-pin drops.
    """
    wires: List[LayerPolygon] = []
    # Continuous strap spine.
    wires.extend(
        route_m1_multibend(
            [Point(x_min, strap_y), Point(x_max, strap_y)], width_nm, pdk_layers
        )
    )
    # Drop from each source pin to the strap.
    for p in net.pins:
        sp = p.point
        if sp.y != strap_y:
            wires.extend(
                route_m1_manhattan(
                    sp, Point(sp.x, strap_y), width_nm, pdk_layers
                )
            )
    return wires


def _route_net(
    net: Net,
    width_nm: int,
    via_size_nm: int,
    pdk_layers: LayerMap,
    rail_bands: List[Tuple[int, int, int]],
) -> Tuple[List[LayerPolygon], bool]:
    """Route a single multi-pin net as a chain of pin-to-pin connections.

    Pins are ordered left-to-right (then bottom-to-top) and connected as a
    spanning chain (pin[i] -> pin[i+1]). A connection whose vertical leg would
    cross a rail band is routed through a via stack onto M2/M3 instead of
    shorting through M1; all other connections stay on M1.

    Returns (wires, crossed) where ``crossed`` is True iff any via was used.
    """
    pts = sorted(
        {(p.point.x, p.point.y) for p in net.pins}
    )
    if len(pts) < 2:
        return [], False
    ordered = [Point(x, y) for (x, y) in pts]

    wires: List[LayerPolygon] = []
    crossed = False
    for i in range(len(ordered) - 1):
        a = ordered[i]
        b = ordered[i + 1]
        # Does the vertical leg of this connection cross a rail at a.x?
        needs_lift = _segment_crosses_band(a.x, a.y, a.x, b.y, rail_bands) or (
            _segment_crosses_band(b.x, a.y, b.x, b.y, rail_bands)
        )
        if needs_lift and not net.is_rail:
            wires.extend(
                _route_crossing_segment(a, b, width_nm, via_size_nm, pdk_layers)
            )
            crossed = True
        else:
            wires.extend(_route_in_plane_segment(a, b, width_nm, pdk_layers))
    return wires, crossed


# --- public opt-in entry point ---------------------------------------------


def route_annotated_cell_full(
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
    sizing: DeviceSizing,
    *,
    lower_fn: Callable[..., ParameterizedLibrary] = lower_annotated_cell_to_parameterized,
    wire_width_nm: int = 140,
    via_size_nm: int = 170,
    power_strap: bool = False,
) -> FullRoutedResult:
    """Opt-in complete routed lowering: place + build net model + route ALL
    routable nets + merge into one library.

    Unlike R133's ``route_annotated_cell`` (which routed only the shared-gate
    net), this routes every net the cell's topology declares — gate, source
    rail, diode tie, output ports — and lifts crossing nets onto M2/M3 through
    real via stacks.

    Args:
        power_strap: when True, rails are routed as continuous full-width M1
            straps spanning the placed cell's bbox at a y-level between the
            source row and the gate/drain row (real power-distribution
            geometry). Any signal net whose vertical leg crosses that strap
            band is then lifted onto M2/M3 through real via stacks instead of
            shorting — this exercises the multi-layer/via path non-trivially.
            When False (default), rails are routed point-to-point between
            source pins (planar M1), which the side-by-side device placement
            keeps naturally crossing-free.

    The placement-only library in ``FullRoutedResult.placement_only`` is an
    unmodified ``lower_fn`` output (byte-identical to the non-routed path).
    The routed library is a deep copy with wires/vias added only to the top
    cell.
    """
    placement = lower_fn(cell, pdk_layers, fin_pitch_nm, sizing)
    routed_lib = _deepcopy_library(placement)
    netlist = build_cell_netlist(placement, cell, pdk_layers)

    top = routed_lib.cells[-1] if routed_lib.cells else None
    cell_bb = _cell_bbox(top) if top is not None else None

    # Rail bands: horizontal source-rail wires that other nets must cross over.
    # We route the rails first (so their geometry exists), record their bands,
    # then route the rest, lifting crossing nets onto M2/M3.
    rail_bands: List[Tuple[int, int, int]] = []
    routed_nets: List[FullRoutedNet] = []

    # Order: rails first, then the rest (so rail_bands is populated for
    # crossing detection on the remaining nets).
    routable = netlist.routable_nets()
    rails = [n for n in routable if n.is_rail]
    others = [n for n in routable if not n.is_rail]

    # Strap y-level: midway between the source row (cell bbox bottom) and the
    # device top (gate/drain row). Derived from real placed geometry.
    strap_y = None
    if power_strap and cell_bb is not None:
        bx_min, by_min, bx_max, by_max = cell_bb
        strap_y = (by_min + by_max) // 4  # low band, above sources, below gates

    for net in rails:
        if power_strap and strap_y is not None and cell_bb is not None:
            bx_min, by_min, bx_max, by_max = cell_bb
            wires = _route_rail_strap(
                net, wire_width_nm, pdk_layers, strap_y, bx_min, bx_max
            )
            crossed = False
        else:
            wires, crossed = _route_net(
                net, wire_width_nm, via_size_nm, pdk_layers, rail_bands
            )
        rail_bands.extend(_hspan_of_wires(wires))
        fr = FullRoutedNet(
            name=net.name,
            devices=tuple(net.devices()),
            pin_count=len(net.pins),
            wires=wires,
            crossed=crossed,
        )
        fr.used_layers = fr.layer_set()
        routed_nets.append(fr)
        if top is not None:
            top.polygons.extend(wires)

    for net in others:
        wires, crossed = _route_net(
            net, wire_width_nm, via_size_nm, pdk_layers, rail_bands
        )
        fr = FullRoutedNet(
            name=net.name,
            devices=tuple(net.devices()),
            pin_count=len(net.pins),
            wires=wires,
            crossed=crossed,
        )
        fr.used_layers = fr.layer_set()
        routed_nets.append(fr)
        if top is not None:
            top.polygons.extend(wires)

    return FullRoutedResult(
        library=routed_lib,
        placement_only=placement,
        netlist=netlist,
        routed_nets=routed_nets,
    )


def m2_layer(pdk_layers: LayerMap) -> Tuple[int, int]:
    """Convenience re-export of the M2 (layer, datatype) for a PDK."""
    return metal2_layer_for_pdk_layers(pdk_layers)


def m3_layer(pdk_layers: LayerMap) -> Tuple[int, int]:
    """Convenience re-export of the M3 (layer, datatype) for a PDK."""
    return metal3_layer_for_pdk_layers(pdk_layers)


def m1_layer(pdk_layers: LayerMap) -> Tuple[int, int]:
    g = pdk_layers.gds_layer(LogicalLayer.METAL1)
    return (g.layer, g.datatype)


__all__ = [
    "FullRoutedNet",
    "FullRoutedResult",
    "route_annotated_cell_full",
    "m1_layer",
    "m2_layer",
    "m3_layer",
]
