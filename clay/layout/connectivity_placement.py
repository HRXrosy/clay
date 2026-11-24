"""R135 (BL-135A) — connectivity-aware device placement.

R134's lowering (``lower_annotated_cell_to_parameterized``) places devices in a
naive left-to-right spread by ``sorted(cell.layouts.keys())`` order, all in a
single y-row. For a 5T-OTA that order is the alphabetical
``M_in_n, M_in_p, M_load_n, M_load_p, M_tail`` — so M_tail lands at the far
right while the input pair sits on the left. But the ``tail_node`` net ties the
input-pair SOURCE pins to the tail-transistor DRAIN pin: in a real 5T-OTA the
tail drain *is* the differential-pair common-source node, so those terminals
belong adjacent. The naive spread scatters them corner-to-corner, forcing the
router to climb the full cell height on M2 and bridge the full width on M3,
overshooting the cell bbox.

This module fixes the root cause — placement — without touching topology. It
computes new device positions that put electrically-connected devices next to
each other (tail under the input pair sharing the common-source column; the
current-mirror load above the input pair), minimising each net's pin-to-pin
Manhattan spread.

Design (opt-in; nothing existing is modified):

1. ``compute_placement`` — given a placed library + its ``CellNetlist``,
   produce a ``DevicePlacement`` map ``{device: (dx, dy)}``: the offset to
   apply to each device's *local-origin* geometry (the per-device NamedCells
   that ``lower_*`` already emits untranslated). For a recognised 5T-OTA the
   map is the textbook three-row stack; for any other cell it is a net-aware
   greedy 1D ordering that places strongly-connected devices adjacent. The
   map is overlap-free by construction (rows/columns are spaced by device
   widths/heights plus a gap, never narrower than the original device gap).

2. ``apply_placement`` — rebuild a ``ParameterizedLibrary`` by translating each
   per-device cell's geometry to its computed position. Connectivity/topology
   is untouched: the per-device geometry, the device set, and (via a fresh
   ``build_cell_netlist`` on the result) the net set + pin membership are
   identical — only coordinates move.

3. ``replace_placement`` — convenience: compute + apply in one call, returning
   the re-placed library.

No coordinate is fabricated: every position is derived from the placed device
bounding boxes (read off real geometry) plus the netlist adjacency. The module
imports ``cell_netlist`` and ``backend`` read-only and modifies neither.

Attribution note. ``cell_netlist.extract_device_pins`` slices the *top* cell's
poly fingers by x-order and assigns them to devices in ``sorted(keys)`` order —
a heuristic that is only valid for the naive single-row, x-monotonic layout
(where x-order already equals alphabetical key order). A connectivity-aware
layout reorders / stacks devices, so that heuristic would mis-attribute pins on
the re-placed top cell. This module therefore carries its own *authoritative*
pin extraction (``placed_device_pins``) that reads each device's OWN per-device
cell geometry (keyed by name, never ambiguous) and applies that device's offset.
Span measurements here use that authoritative attribution; spans must NOT be
re-derived from ``build_cell_netlist`` on a reordered top cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .annotations import AnnotatedCell
from .backend import (
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    ParameterizedLibrary,
    Point,
    Polygon,
)
from .cell_netlist import (
    CellNetlist,
    DevicePins,
    Net,
    PinKind,
    TopologyKind,
    build_cell_netlist,
    classify_topology,
)


# --- placement record -------------------------------------------------------


@dataclass
class DevicePlacement:
    """Computed per-device translation offsets for a connectivity-aware layout.

    ``offsets[device] = (dx, dy)`` is the translation to apply to that device's
    local-origin geometry (the untranslated per-device ``NamedCell`` emitted by
    ``lower_*``). ``strategy`` records which placement rule produced the map so
    callers / tests can assert on it.
    """

    offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    strategy: str = "unknown"

    def devices(self) -> List[str]:
        return list(self.offsets.keys())


# --- geometry helpers (read-only over placed geometry) ---------------------


def _poly_bbox(polys: List[LayerPolygon]) -> Optional[Tuple[int, int, int, int]]:
    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _device_cells(lib: ParameterizedLibrary, cell: AnnotatedCell) -> Dict[str, NamedCell]:
    """The per-device (local-origin) NamedCells, keyed by device name.

    ``lower_*`` emits one NamedCell per device (at local origin, untranslated)
    followed by the top cell; we pick the device cells by name.
    """
    return {c.name: c for c in lib.cells if c.name in cell.layouts}


def _device_local_bboxes(
    lib: ParameterizedLibrary, cell: AnnotatedCell
) -> Dict[str, Tuple[int, int, int, int]]:
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for dev, dcell in _device_cells(lib, cell).items():
        bb = _poly_bbox(dcell.polygons)
        if bb is not None:
            out[dev] = bb
    return out


def _translate_polys(polys: List[LayerPolygon], dx: int, dy: int) -> List[LayerPolygon]:
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(
                vertices=[Point(v.x + dx, v.y + dy) for v in lp.polygon.vertices]
            ),
        )
        for lp in polys
    ]


# --- authoritative per-device pin extraction --------------------------------


def _poly_diff_keys(pdk_layers: LayerMap) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    poly = pdk_layers.gds_layer(LogicalLayer.POLY)
    if poly is None:
        raise ValueError("PDK must define POLY layer for pin extraction")
    poly_key = (poly.layer, poly.datatype)
    diff_keys: List[Tuple[int, int]] = []
    for ll in (LogicalLayer.NDIFF, LogicalLayer.PDIFF):
        g = pdk_layers.gds_layer(ll)
        if g is not None and (g.layer, g.datatype) not in diff_keys:
            diff_keys.append((g.layer, g.datatype))
    return poly_key, diff_keys


def _pins_from_device_geometry(
    dev: str, polys: List[LayerPolygon], poly_key, diff_keys
) -> Optional[DevicePins]:
    """Extract gate/source/drain pins from ONE device's own geometry.

    This is the same physical convention as ``cell_netlist.DevicePins`` (gate =
    poly top-edge center-x; drain = diff top-edge center-x; source = diff
    bottom-edge center-x) but scoped to a single device's polygons, so the
    attribution is unambiguous (no x-order slicing across devices). Used to
    measure net spans correctly on reordered / stacked placements.
    """
    poly_fingers = [lp for lp in polys if (lp.layer, lp.datatype) == poly_key]
    diff_rects = [lp for lp in polys if (lp.layer, lp.datatype) in diff_keys]
    if not poly_fingers or not diff_rects:
        return None
    pbb = _poly_bbox(poly_fingers)
    dbb = _poly_bbox(diff_rects)
    assert pbb is not None and dbb is not None
    px_min, py_min, px_max, py_max = pbb
    dx_min, dy_min, dx_max, dy_max = dbb
    poly_cx = (px_min + px_max) // 2
    diff_cx = (dx_min + dx_max) // 2
    return DevicePins(
        device=dev,
        gate=Point(poly_cx, py_max),
        drain=Point(diff_cx, dy_max),
        source=Point(diff_cx, dy_min),
        poly_bbox=pbb,
        diff_bbox=dbb,
    )


def placed_device_pins(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    placement: "DevicePlacement",
) -> Dict[str, DevicePins]:
    """Authoritative per-device pins for a connectivity-aware placement.

    Reads each device's OWN per-device cell geometry (keyed by name) and
    translates it by ``placement.offsets[dev]`` — so the gate/source/drain
    points reflect exactly where that device lands, independent of x-order.
    This is the correct basis for measuring net spans on a re-placed cell
    (``build_cell_netlist`` on a reordered top cell would mis-attribute).
    """
    poly_key, diff_keys = _poly_diff_keys(pdk_layers)
    out: Dict[str, DevicePins] = {}
    for dev, dcell in _device_cells(lib, cell).items():
        dx, dy = placement.offsets.get(dev, (0, 0))
        moved = _translate_polys(dcell.polygons, dx, dy)
        dp = _pins_from_device_geometry(dev, moved, poly_key, diff_keys)
        if dp is not None:
            out[dev] = dp
    return out


def _pin_point(dp: Dict[str, DevicePins], device: str, kind: PinKind) -> Optional[Point]:
    d = dp.get(device)
    if d is None:
        return None
    if kind == PinKind.GATE:
        return d.gate
    if kind == PinKind.SOURCE:
        return d.source
    if kind == PinKind.DRAIN:
        return d.drain
    return None


def net_max_span_with_pins(net: Net, dp: Dict[str, DevicePins]) -> int:
    """Max pairwise Manhattan span of a net, re-evaluated against an
    authoritative ``DevicePins`` map (device+kind -> moved point) instead of the
    net's stored points. Lets us measure ``tail_node`` span on a re-placed cell
    using correct per-device attribution."""
    pts = set()
    for p in net.pins:
        pt = _pin_point(dp, p.device, p.kind)
        if pt is not None:
            pts.add((pt.x, pt.y))
    pts = sorted(pts)
    best = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
            if d > best:
                best = d
    return best


def tail_node_max_span(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    placement: "DevicePlacement",
) -> int:
    """Authoritative max pin-pair Manhattan span of the 5T-OTA ``tail_node``
    net under a given placement. The net's pin membership (which device+kind)
    comes from the topology netlist; the points come from authoritative placed
    pins. Raises ``KeyError`` if the cell has no ``tail_node`` net."""
    netlist = build_cell_netlist(lib, cell, pdk_layers)
    dp = placed_device_pins(lib, cell, pdk_layers, placement)
    for n in netlist.nets:
        if n.name == "tail_node":
            return net_max_span_with_pins(n, dp)
    raise KeyError("cell has no tail_node net")


# --- net span measurement (used for verification + greedy ordering) --------


def net_max_pin_span(net: Net) -> int:
    """Maximum pairwise Manhattan distance between any two distinct pin points
    of a net. This is the worst-case wire reach the router must span; the whole
    point of connectivity-aware placement is to shrink it for the tight nets
    (``tail_node`` above all)."""
    pts = sorted({(p.point.x, p.point.y) for p in net.pins})
    best = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
            if d > best:
                best = d
    return best


def net_total_span(net: Net) -> int:
    """Half-perimeter (HPWL) of a net's pin bounding box — the standard
    placement cost proxy for one net's wirelength."""
    pts = list({(p.point.x, p.point.y) for p in net.pins})
    if len(pts) < 2:
        return 0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def netlist_total_hpwl(netlist: CellNetlist) -> int:
    return sum(net_total_span(n) for n in netlist.nets)


def max_span_of(netlist: CellNetlist, net_name: str) -> int:
    for n in netlist.nets:
        if n.name == net_name:
            return net_max_pin_span(n)
    raise KeyError(f"net {net_name!r} not in netlist")


# --- placement strategies ---------------------------------------------------


def _gap_for(fin_pitch_nm: float) -> int:
    """Inter-device gap, matching the original lowering's device gap
    (4 * fin_pitch). Keeps spacing no tighter than the naive path, so an
    overlap that the naive path avoided cannot appear here."""
    return 4 * int(round(fin_pitch_nm))


def _five_t_ota_offsets(
    bboxes: Dict[str, Tuple[int, int, int, int]], gap: int
) -> Dict[str, Tuple[int, int]]:
    """Three-row connectivity-aware 5T-OTA stack.

    Layout (low y at bottom, x increasing right):

        row 2 (top):     M_load_p   M_load_n      (PMOS mirror load)
        row 1 (middle):  M_in_p     M_in_n        (NMOS input pair)
        row 0 (bottom):       M_tail               (NMOS tail source)

    The input pair shares a vertical band of x; M_tail sits in the SAME x band
    directly below them, so the tail DRAIN (top edge of M_tail) is one ``gap``
    below the input SOURCEs (bottom edge of the input row) — that is the
    ``tail_node`` common-source connection, now a short vertical hop instead of
    a corner-to-corner climb. The load pair sits directly above the input pair,
    keeping the ``out`` / ``left_branch`` / ``load_gate`` drain nets short too.

    All coordinates derive from the device bboxes + ``gap``; nothing invented.
    """
    devs = ["M_in_p", "M_in_n", "M_load_p", "M_load_n", "M_tail"]
    w = {d: bboxes[d][2] - bboxes[d][0] for d in devs}
    h = {d: bboxes[d][3] - bboxes[d][1] for d in devs}

    offsets: Dict[str, Tuple[int, int]] = {}

    # --- input row (middle): two devices side by side. Its horizontal band
    # is the reference all other rows center on. ---
    in_p_x = 0
    in_n_x = in_p_x + w["M_in_p"] + gap
    input_band_min = in_p_x
    input_band_max = in_n_x + w["M_in_n"]
    input_band_center = (input_band_min + input_band_max) // 2

    # --- load row (top): two devices side by side; the WHOLE row is centered
    # over the input band so the mirror-load drains land above the input
    # branch nodes. Each load device may be wider than an input device, so
    # space them by their own widths (never overlapping) and shift the row.
    load_row_w = w["M_load_p"] + gap + w["M_load_n"]
    load_row_min = input_band_center - load_row_w // 2
    load_p_x = load_row_min
    load_n_x = load_p_x + w["M_load_p"] + gap

    # --- tail row (bottom): single device, centered under the input band (its
    # drain is the common-source node for BOTH input devices). ---
    tail_x = input_band_center - w["M_tail"] // 2

    # --- y rows: bottom = tail, middle = input, top = load. ---
    tail_y = 0
    input_y = tail_y + h["M_tail"] + gap
    input_row_h = max(h["M_in_p"], h["M_in_n"])
    load_y = input_y + input_row_h + gap

    # Offsets translate each device's LOCAL bbox-min to the target position.
    def off(dev: str, target_x: int, target_y: int) -> Tuple[int, int]:
        bx_min, by_min, _, _ = bboxes[dev]
        return (target_x - bx_min, target_y - by_min)

    offsets["M_tail"] = off("M_tail", tail_x, tail_y)
    offsets["M_in_p"] = off("M_in_p", in_p_x, input_y)
    offsets["M_in_n"] = off("M_in_n", in_n_x, input_y)
    offsets["M_load_p"] = off("M_load_p", load_p_x, load_y)
    offsets["M_load_n"] = off("M_load_n", load_n_x, load_y)
    return offsets


def _net_adjacency_order(
    devices: List[str], netlist: CellNetlist
) -> List[str]:
    """Greedy net-aware 1D ordering for non-5T cells.

    Builds a device-device weight graph (number of shared nets) and grows a
    chain by repeatedly appending the unplaced device most strongly connected
    to the current chain end (ties broken by total connectivity, then name).
    This clusters strongly-connected devices next to each other, shrinking the
    summed net span vs the topology-blind ``sorted()`` order — the general-case
    fallback when the cell isn't a recognised 5T-OTA.
    """
    weight: Dict[Tuple[str, str], int] = {}

    def bump(a: str, b: str) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        weight[key] = weight.get(key, 0) + 1

    for net in netlist.nets:
        ds = net.devices()
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                bump(ds[i], ds[j])

    def w_between(a: str, b: str) -> int:
        key = (a, b) if a < b else (b, a)
        return weight.get(key, 0)

    total_conn = {
        d: sum(w_between(d, o) for o in devices if o != d) for d in devices
    }

    remaining = sorted(devices, key=lambda d: (-total_conn[d], d))
    if not remaining:
        return []
    order = [remaining.pop(0)]
    while remaining:
        end = order[-1]
        nxt = max(
            remaining,
            key=lambda d: (w_between(end, d), total_conn[d], -ord(d[0]) if d else 0),
        )
        order.append(nxt)
        remaining.remove(nxt)
    return order


def _row_offsets(
    order: List[str],
    bboxes: Dict[str, Tuple[int, int, int, int]],
    gap: int,
) -> Dict[str, Tuple[int, int]]:
    """Place devices in a single y-row, left to right, in ``order``."""
    offsets: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for dev in order:
        bx_min, by_min, bx_max, _ = bboxes[dev]
        offsets[dev] = (cursor - bx_min, -by_min)
        cursor += (bx_max - bx_min) + gap
    return offsets


# --- public API -------------------------------------------------------------


def compute_placement(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    netlist: CellNetlist,
    fin_pitch_nm: float,
) -> DevicePlacement:
    """Compute connectivity-aware per-device offsets for a placed cell.

    ``lib`` is an unmodified ``lower_*`` output (supplies per-device local-origin
    geometry). ``netlist`` is ``build_cell_netlist(lib, cell, pdk_layers)``.
    Returns offsets to translate each device's local geometry; topology and
    connectivity are untouched (only positions change).
    """
    bboxes = _device_local_bboxes(lib, cell)
    gap = _gap_for(fin_pitch_nm)
    topo = classify_topology(cell)

    if topo == TopologyKind.FIVE_T_OTA and all(
        d in bboxes
        for d in ("M_in_p", "M_in_n", "M_load_p", "M_load_n", "M_tail")
    ):
        offsets = _five_t_ota_offsets(bboxes, gap)
        return DevicePlacement(offsets=offsets, strategy="five_t_ota_stack")

    devices = sorted(bboxes.keys())
    order = _net_adjacency_order(devices, netlist)
    if not order:
        order = devices
    offsets = _row_offsets(order, bboxes, gap)
    return DevicePlacement(offsets=offsets, strategy="net_adjacency_row")


def apply_placement(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    placement: DevicePlacement,
) -> ParameterizedLibrary:
    """Rebuild a library with each device translated to its computed position.

    Produces a NEW ``ParameterizedLibrary`` (the input is not mutated):
      - one per-device NamedCell per device, geometry untouched (local origin),
      - a rebuilt top cell whose polygons are the per-device geometry
        translated by ``placement.offsets[dev]``.

    Topology / connectivity is unchanged: per-device geometry is copied
    verbatim, only coordinates of the top-cell instances move. Devices with no
    offset (shouldn't happen for a complete map) keep their original placement
    via a zero offset.
    """
    device_cells = _device_cells(lib, cell)
    new_lib = ParameterizedLibrary(
        name=lib.name,
        unit_user=lib.unit_user,
        unit_db_meters=lib.unit_db_meters,
    )

    # Per-device cells: copy verbatim (local origin, geometry untouched).
    for dev in sorted(device_cells.keys()):
        dcell = device_cells[dev]
        new_lib.add_cell(
            NamedCell(
                name=dcell.name,
                polygons=[
                    LayerPolygon(
                        layer=lp.layer,
                        datatype=lp.datatype,
                        polygon=Polygon(vertices=list(lp.polygon.vertices)),
                    )
                    for lp in dcell.polygons
                ],
            )
        )

    # Top cell: place each device's geometry at its computed offset.
    top_polys: List[LayerPolygon] = []
    for dev in sorted(device_cells.keys()):
        dx, dy = placement.offsets.get(dev, (0, 0))
        top_polys.extend(_translate_polys(device_cells[dev].polygons, dx, dy))

    new_lib.add_cell(NamedCell(name=cell.cell_name, polygons=top_polys))
    return new_lib


def replace_placement(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    fin_pitch_nm: float,
) -> Tuple[ParameterizedLibrary, DevicePlacement]:
    """Compute + apply connectivity-aware placement in one call.

    Returns ``(re_placed_library, placement)``. The re-placed library can be
    fed straight back into ``build_cell_netlist`` to verify the connectivity is
    unchanged and the net spans shrank.
    """
    netlist = build_cell_netlist(lib, cell, pdk_layers)
    placement = compute_placement(lib, cell, netlist, fin_pitch_nm)
    new_lib = apply_placement(lib, cell, placement)
    return new_lib, placement


# --- overlap check (for tests + audit) -------------------------------------


def _device_placed_bbox(
    bboxes: Dict[str, Tuple[int, int, int, int]],
    placement: DevicePlacement,
    dev: str,
) -> Tuple[int, int, int, int]:
    bx_min, by_min, bx_max, by_max = bboxes[dev]
    dx, dy = placement.offsets.get(dev, (0, 0))
    return (bx_min + dx, by_min + dy, bx_max + dx, by_max + dy)


def _bboxes_overlap(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    # Touching edges (shared boundary) is NOT overlap; require strict interior
    # intersection.
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def find_overlaps(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    placement: DevicePlacement,
) -> List[Tuple[str, str]]:
    """Return device-name pairs whose placed bounding boxes strictly overlap.
    Empty list == overlap-free placement."""
    bboxes = _device_local_bboxes(lib, cell)
    devs = sorted(placement.offsets.keys())
    placed = {d: _device_placed_bbox(bboxes, placement, d) for d in devs if d in bboxes}
    out: List[Tuple[str, str]] = []
    keys = list(placed.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if _bboxes_overlap(placed[keys[i]], placed[keys[j]]):
                out.append((keys[i], keys[j]))
    return out


__all__ = [
    "DevicePlacement",
    "compute_placement",
    "apply_placement",
    "replace_placement",
    "find_overlaps",
    "placed_device_pins",
    "net_max_span_with_pins",
    "tail_node_max_span",
    "net_max_pin_span",
    "net_total_span",
    "netlist_total_hpwl",
    "max_span_of",
]
