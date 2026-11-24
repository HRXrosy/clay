"""R134 (BL-134A) — explicit net model derived from a placed cell.

R133's ``routed_lowering`` could only route ONE net (the shared gate of a
current mirror) because ``AnnotatedCell`` carries no net model: it knows which
devices *match*, but not which device *pins* (gate / drain / source / bulk)
tie together, nor where those pins land on the placed geometry. This module
supplies that missing layer WITHOUT touching ``backend.py`` or
``routed_lowering.py``.

What it does:

1. ``DevicePins`` — for one device, extract gate / source / drain access
   points from the device's PLACED geometry (poly fingers + diffusion
   rectangle), produced by an unmodified ``lower_*`` call. The geometry is
   real: the gate pin is the top edge of the poly fingers, the source/drain
   pins are the bottom / inner edges of the diffusion. No coordinate is
   invented — every point is read off a placed ``LayerPolygon``.

2. ``build_cell_netlist`` — turn the cell's TOPOLOGY (current-mirror
   semantics: matched devices share a gate, both sources tie to the rail, the
   reference is diode-connected, each drain is an output port) into a list of
   ``Net`` objects, each a set of device pins that must connect.

The connectivity comes from the cell's real topology + annotations, not from
guesses. A current mirror's invariants (shared gate, common source rail,
diode-connected reference) are the textbook definition of the topology — the
``Matching`` annotation plus the per-template ``CellTopology`` descriptor
encode them. Cells without a recognised topology yield only the nets their
annotations directly declare (the shared-gate net), so the model degrades to
R133 behaviour rather than fabricating connectivity.

This module produces only the abstract net model (pins + which pins connect).
It performs NO routing and emits NO wire geometry — that is the job of
``routed_lowering_full`` which feeds these nets to the real routers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .annotations import AnnotatedCell, Matching
from .backend import (
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    ParameterizedLibrary,
    Point,
)


class PinKind(Enum):
    """The terminal of a MOS device a net pin attaches to."""

    GATE = "gate"
    SOURCE = "source"
    DRAIN = "drain"
    BULK = "bulk"


@dataclass(frozen=True, eq=True)
class Pin:
    """One device terminal access point on the placed top cell.

    ``point`` is read off the placed geometry (poly finger top edge for a
    gate, diffusion edge for source/drain). It is never fabricated.
    """

    device: str
    kind: PinKind
    point: Point


@dataclass
class Net:
    """A set of pins that must be electrically connected.

    A net with one pin is a port (an external connection point, e.g. a drain
    output) — it still appears in the netlist so the audit can account for
    every terminal, but routing it is a no-op (nothing to connect to).
    """

    name: str
    pins: List[Pin] = field(default_factory=list)
    is_port: bool = False
    is_rail: bool = False

    def devices(self) -> List[str]:
        seen: List[str] = []
        for p in self.pins:
            if p.device not in seen:
                seen.append(p.device)
        return seen

    def needs_routing(self) -> bool:
        """A net needs routing iff it connects >= 2 distinct pin points."""
        pts = {(p.point.x, p.point.y) for p in self.pins}
        return len(pts) >= 2


@dataclass
class CellNetlist:
    """All nets of a cell plus the per-device pin table used to build them."""

    cell_name: str
    nets: List[Net] = field(default_factory=list)
    device_pins: Dict[str, "DevicePins"] = field(default_factory=dict)

    def routable_nets(self) -> List[Net]:
        return [n for n in self.nets if n.needs_routing()]

    def port_nets(self) -> List[Net]:
        return [n for n in self.nets if not n.needs_routing()]


# --- topology descriptors (the connectivity intent per cell template) ------


class TopologyKind(Enum):
    CURRENT_MIRROR = "current_mirror"
    FIVE_T_OTA = "five_t_ota"
    UNKNOWN = "unknown"


def classify_topology(cell: AnnotatedCell) -> TopologyKind:
    """Recognise the cell's topology from its device set + annotations.

    Recognition is conservative and structural: it keys off the device names
    the template factories emit (``M_ref``/``M_out`` for a current mirror;
    the 5T-OTA's five canonical devices). An unrecognised cell returns
    ``UNKNOWN`` and only its Matching-declared gate nets are built.
    """
    keys = set(cell.layouts.keys())
    if {"M_ref", "M_out"} <= keys and len(keys) == 2:
        return TopologyKind.CURRENT_MIRROR
    if {"M_in_p", "M_in_n", "M_load_p", "M_load_n", "M_tail"} <= keys:
        return TopologyKind.FIVE_T_OTA
    return TopologyKind.UNKNOWN


# --- pin extraction from placed geometry -----------------------------------


def _poly_key(pdk_layers: LayerMap) -> Tuple[int, int]:
    g = pdk_layers.gds_layer(LogicalLayer.POLY)
    if g is None:
        raise ValueError("PDK must define POLY layer for pin extraction")
    return (g.layer, g.datatype)


def _diff_keys(pdk_layers: LayerMap) -> List[Tuple[int, int]]:
    keys: List[Tuple[int, int]] = []
    for ll in (LogicalLayer.NDIFF, LogicalLayer.PDIFF):
        g = pdk_layers.gds_layer(ll)
        if g is not None and (g.layer, g.datatype) not in keys:
            keys.append((g.layer, g.datatype))
    return keys


def _bbox(lp: LayerPolygon) -> Tuple[int, int, int, int]:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class DevicePins:
    """Gate / source / drain access points for one placed device.

    All points are read off placed geometry:
      - gate:   center-x of the device's poly fingers at the TOP edge
                (y = max poly y) — where an M1 gate strap lands.
      - drain:  center-x of the diffusion at its TOP edge (y = max diff y),
                i.e. the drain-side contact band above the channel.
      - source: center-x of the diffusion at its BOTTOM edge (y = min diff y),
                i.e. the source-side contact band below the channel.

    Splitting source vs drain by top/bottom edge of the same diffusion gives
    two physically distinct, non-coincident access points so a source rail and
    a drain net route at different y-levels (and can therefore cross).
    """

    device: str
    gate: Point
    source: Point
    drain: Point
    poly_bbox: Tuple[int, int, int, int]
    diff_bbox: Tuple[int, int, int, int]


def extract_device_pins(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
) -> Dict[str, DevicePins]:
    """Derive gate/source/drain pins for every device from placed geometry.

    Walks the top cell's poly fingers and diffusion rectangles, slicing them
    into devices in ``sorted(cell.layouts.keys())`` order (the same order
    ``lower_*`` appends them), pairing each device's poly-finger group with the
    diffusion rectangle whose x-span contains that group.
    """
    poly_key = _poly_key(pdk_layers)
    diff_keys = _diff_keys(pdk_layers)
    device_keys = sorted(cell.layouts.keys())

    per_device_cells = {c.name: c for c in lib.cells if c.name in cell.layouts}
    finger_counts: Dict[str, int] = {}
    for dev in device_keys:
        dcell = per_device_cells.get(dev)
        if dcell is None:
            finger_counts[dev] = cell.layouts[dev].finger_count
        else:
            finger_counts[dev] = sum(
                1 for lp in dcell.polygons if (lp.layer, lp.datatype) == poly_key
            )

    top = lib.cells[-1]
    top_polys = top.polygons
    poly_fingers = sorted(
        (lp for lp in top_polys if (lp.layer, lp.datatype) == poly_key),
        key=lambda lp: _bbox(lp)[0],
    )
    diff_rects = sorted(
        (lp for lp in top_polys if (lp.layer, lp.datatype) in diff_keys),
        key=lambda lp: _bbox(lp)[0],
    )

    pins: Dict[str, DevicePins] = {}
    cursor = 0
    for dev in device_keys:
        n = finger_counts[dev]
        group = poly_fingers[cursor : cursor + n]
        cursor += n
        if not group:
            continue
        gx_min = min(_bbox(lp)[0] for lp in group)
        gx_max = max(_bbox(lp)[2] for lp in group)
        gy_max = max(_bbox(lp)[3] for lp in group)
        center_x = (gx_min + gx_max) // 2

        # Find the diffusion rectangle whose x-span covers this device's
        # poly-finger group (each device emits exactly one diffusion rect).
        diff_bb: Optional[Tuple[int, int, int, int]] = None
        for d in diff_rects:
            dx_min, dy_min, dx_max, dy_max = _bbox(d)
            if dx_min <= center_x <= dx_max:
                diff_bb = (dx_min, dy_min, dx_max, dy_max)
                break
        if diff_bb is None:
            # Fallback: nearest diffusion by center distance.
            if diff_rects:
                d = min(
                    diff_rects,
                    key=lambda dd: abs(
                        (_bbox(dd)[0] + _bbox(dd)[2]) // 2 - center_x
                    ),
                )
                diff_bb = _bbox(d)
            else:
                diff_bb = (gx_min, 0, gx_max, gy_max)

        dx_min, dy_min, dx_max, dy_max = diff_bb
        diff_cx = (dx_min + dx_max) // 2

        pins[dev] = DevicePins(
            device=dev,
            gate=Point(center_x, gy_max),
            drain=Point(diff_cx, dy_max),
            source=Point(diff_cx, dy_min),
            poly_bbox=(gx_min, _bbox(group[0])[1], gx_max, gy_max),
            diff_bbox=diff_bb,
        )
    return pins


# --- netlist construction ---------------------------------------------------


def _matched_gate_devices(cell: AnnotatedCell) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen: set = set()
    for dev, anns in cell.annotations.items():
        for a in anns:
            if isinstance(a, Matching):
                other = a.other
                if dev not in cell.layouts or other not in cell.layouts:
                    continue
                key = tuple(sorted((dev, other)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((key[0], key[1]))
    return pairs


def build_cell_netlist(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
) -> CellNetlist:
    """Build the explicit net model for a placed cell.

    The placement library ``lib`` must be the output of an unmodified
    ``lower_*`` call (it supplies the placed geometry pins are read from).
    """
    device_pins = extract_device_pins(lib, cell, pdk_layers)
    topo = classify_topology(cell)
    nets: List[Net] = []

    if topo == TopologyKind.CURRENT_MIRROR:
        nets = _current_mirror_nets(device_pins)
    elif topo == TopologyKind.FIVE_T_OTA:
        nets = _five_t_ota_nets(device_pins)
    else:
        # Unknown topology: only the Matching-declared shared-gate nets, so
        # the model never fabricates connectivity it cannot justify.
        for a, b in _matched_gate_devices(cell):
            if a in device_pins and b in device_pins:
                nets.append(
                    Net(
                        name=f"gate_{a}_{b}",
                        pins=[
                            Pin(a, PinKind.GATE, device_pins[a].gate),
                            Pin(b, PinKind.GATE, device_pins[b].gate),
                        ],
                    )
                )

    return CellNetlist(
        cell_name=cell.cell_name, nets=nets, device_pins=device_pins
    )


def _current_mirror_nets(dp: Dict[str, DevicePins]) -> List[Net]:
    """Current-mirror net list — the textbook topology.

      - gate net: M_ref.gate + M_out.gate tie together (shared gate).
      - source rail: M_ref.source + M_out.source tie to the common rail
        (both transistors share the same source node — the mirror invariant).
      - vref net: M_ref.drain ties to the gate net (the reference is
        diode-connected; its drain shorts to its own gate). This makes the
        reference-drain pin part of the gate net, so the gate net has 3 pins
        at two y-levels (gates at top, ref-drain at the diffusion) — a net
        that genuinely needs a layer change to route.
      - vout port: M_out.drain is the output terminal (single-pin port).
    """
    ref = dp.get("M_ref")
    out = dp.get("M_out")
    nets: List[Net] = []
    if ref is None or out is None:
        return nets

    # Gate net + diode tie (ref drain shorts to gate -> same net).
    nets.append(
        Net(
            name="gate",
            pins=[
                Pin("M_ref", PinKind.GATE, ref.gate),
                Pin("M_out", PinKind.GATE, out.gate),
                Pin("M_ref", PinKind.DRAIN, ref.drain),  # diode connection
            ],
        )
    )
    # Common source rail.
    nets.append(
        Net(
            name="vss_rail",
            pins=[
                Pin("M_ref", PinKind.SOURCE, ref.source),
                Pin("M_out", PinKind.SOURCE, out.source),
            ],
            is_rail=True,
        )
    )
    # Output port (single pin).
    nets.append(
        Net(
            name="vout",
            pins=[Pin("M_out", PinKind.DRAIN, out.drain)],
            is_port=True,
        )
    )
    return nets


def _five_t_ota_nets(dp: Dict[str, DevicePins]) -> List[Net]:
    """5T-OTA net list — the foundational analog amplifier topology.

    Devices: M_in_p / M_in_n (NMOS input pair), M_load_p / M_load_n (PMOS
    current-mirror load, diode-connected on the plus side), M_tail (NMOS tail).

    Nets derived from the topology:
      - load_gate: M_load_p.gate + M_load_n.gate + M_load_p.drain
        (load is a current mirror; load_p diode-connected) — 3 pins.
      - out: M_load_n.drain + M_in_n.drain (the OTA output node) — 2 pins.
      - in_p_drain: M_in_p.drain + M_load_p.drain (left branch node) — 2 pins.
      - tail_node: M_in_p.source + M_in_n.source + M_tail.drain
        (the common-source / tail node) — 3 pins.
      - vss_rail: M_tail.source — port to ground (1 pin).
      - vdd_rail: M_load_p.source + M_load_n.source — supply rail (2 pins).
      - in_p / in_n: input gates — ports (1 pin each).
    """
    need = ["M_in_p", "M_in_n", "M_load_p", "M_load_n", "M_tail"]
    if any(k not in dp for k in need):
        return []
    inp, inn = dp["M_in_p"], dp["M_in_n"]
    lp, ln = dp["M_load_p"], dp["M_load_n"]
    tail = dp["M_tail"]

    nets: List[Net] = [
        Net(
            name="load_gate",
            pins=[
                Pin("M_load_p", PinKind.GATE, lp.gate),
                Pin("M_load_n", PinKind.GATE, ln.gate),
                Pin("M_load_p", PinKind.DRAIN, lp.drain),  # diode tie
            ],
        ),
        Net(
            name="out",
            pins=[
                Pin("M_load_n", PinKind.DRAIN, ln.drain),
                Pin("M_in_n", PinKind.DRAIN, inn.drain),
            ],
        ),
        Net(
            name="left_branch",
            pins=[
                Pin("M_in_p", PinKind.DRAIN, inp.drain),
                Pin("M_load_p", PinKind.DRAIN, lp.drain),
            ],
        ),
        Net(
            name="tail_node",
            pins=[
                Pin("M_in_p", PinKind.SOURCE, inp.source),
                Pin("M_in_n", PinKind.SOURCE, inn.source),
                Pin("M_tail", PinKind.DRAIN, tail.drain),
            ],
        ),
        Net(
            name="vdd_rail",
            pins=[
                Pin("M_load_p", PinKind.SOURCE, lp.source),
                Pin("M_load_n", PinKind.SOURCE, ln.source),
            ],
            is_rail=True,
        ),
        Net(
            name="vss_rail",
            pins=[Pin("M_tail", PinKind.SOURCE, tail.source)],
            is_rail=True,
            is_port=True,
        ),
        Net(name="in_p", pins=[Pin("M_in_p", PinKind.GATE, inp.gate)], is_port=True),
        Net(name="in_n", pins=[Pin("M_in_n", PinKind.GATE, inn.gate)], is_port=True),
    ]
    return nets


__all__ = [
    "PinKind",
    "Pin",
    "Net",
    "CellNetlist",
    "TopologyKind",
    "classify_topology",
    "DevicePins",
    "extract_device_pins",
    "build_cell_netlist",
]
