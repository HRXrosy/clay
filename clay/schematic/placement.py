"""Deterministic, role/stage-driven placement for a :class:`SchematicModel`.

The placement is a pure function of the model (no randomness, no I/O), so
re-running it byte-reproduces the same coordinates. The readable layout idea:

* Columns left -> right follow the signal stage:
  tail/input -> 1st-stage precharge -> regen latch -> 2nd-stage precharge ->
  output inverters -> reset/bleed.
* NMOS sit in a lower band, PMOS in an upper band (mirroring a real schematic
  with the supply at the top and ground at the bottom).
* Differential pairs are mirrored about a vertical centre axis: the ``*n`` /
  left member to the left, the ``*p`` / right member to the right.
* A vdd rail spans the top, a gnd rail spans the bottom.

DEVICE ORIENTATION (vertical flip) is a real degree of freedom. Each device's
``flip`` drives the actual pin Y (see :func:`~clay.schematic.symbols.pin_offsets`):
``flip=True`` swaps which of drain / source sits at the top vs bottom pin. The
supply-side convention is the default/anchor -- PMOS source UP toward the vdd
rail (drain faces DOWN into the channel), NMOS source DOWN toward the gnd rail
(drain faces UP). Orientation is then OPTIMISED under symmetry: for each
symmetry group the two partners must keep CONSISTENT (mirrored) orientation, and
subject to that the per-group flip that MINIMISES total Manhattan wire length is
chosen (ties -> the supply-side convention).

GEOMETRY CONTRACT (shared with the viewer):

* Integer units on a shared grid (:data:`~clay.schematic.symbols.GRID` = 20).
  Every device CENTER ``(x, y)`` and every pin ``(x, y)`` is grid-snapped.
* ``device.x`` / ``device.y`` are the symbol CENTER. Per-kind symbol box + pin
  offsets come from :data:`~clay.schematic.symbols.SYMBOLS`; absolute pin coords
  are ``center + offset`` (then snapped).
* Net routing is orthogonal trunk-and-stub: one horizontal trunk at the net's
  assigned channel-y, vertical stubs from the trunk down/up to each pin. Trunk
  y values are distinct per signal net so trunks never collide. Differential
  pairs are routed mirror-symmetric about the vertical centre axis. vdd stubs go
  UP to the top rail, gnd stubs go DOWN to the bottom rail; clk gets its own
  labelled trunk. Every wire endpoint lands exactly on a pin it connects.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .model import SchematicModel
from .symbols import GRID, default_flip, pin_offsets, snap, symbol_box

# --- layout constants (abstract schematic units, grid multiples) -------------
COL_PITCH = 200        # horizontal pitch between stage columns (>= 200)
ROW_PITCH = 240        # vertical pitch within a band (>= 240)
MARGIN = 80            # viewbox margin (grid multiple)

# Band centre-y for each kind. The inter-row channel between PMOS (upper) and
# NMOS (lower) carries the signal trunks. Picked so boxes (88 tall) clear it.
PMOS_BAND_Y = 240      # top band centre (PMOS row 0); sub-row 1 at +ROW_PITCH
NMOS_BAND_Y = 1040     # lower band centre (NMOS row 0)
# Signal trunks ride the clear inter-row channel: below the lowest PMOS box
# bottom (sub-row 1 box bottom = 480 + 88/2 = 524) and above the NMOS box top
# (1040 - 88/2 = 996). The band [560, 996) holds every trunk without crossing a
# device box.
CHANNEL_TOP = 560      # first signal-trunk channel y (clear of the PMOS band)
CHANNEL_STEP = 20      # vertical spacing between distinct net trunks (= GRID)

RAIL_VDD_Y = 80        # vdd rail y (above the PMOS band)
RAIL_GND_Y = 1320      # gnd rail y (below the NMOS band)

# Length of the short horizontal INPUT STUB drawn for a single-pin input gate
# net (net0 / net1): the gate pin extends outward to a small input-port point
# that carries the net label. Grid multiple. 5*GRID so the mirrored (right)
# port clears its device box and lands clear of the body pin (body pin sits at
# gate_x + 4*GRID, so a 4*GRID right stub would collide with it).
INPUT_STUB_LEN = 5 * GRID  # 100

# Length of the short OUTPUT stub branched off an output net's channel point so
# the output (outm / outp) carries a port pin. Mirror-symmetric: the left output
# (outm) branches LEFT, the right output (outp) branches RIGHT. 2*GRID keeps it
# short and clear of neighbouring trunks.
OUTPUT_STUB_LEN = 2 * GRID  # 40

# Length of the short clk INPUT stub branched off the clk trunk's left end so
# the clock carries a labelled input pin (mirror-neutral: it extends LEFT of the
# trunk, away from every device column).
CLK_PORT_STUB_LEN = 5 * GRID  # 100

# How far a supply (vdd / gnd) NET LABEL is pushed PERPENDICULARLY off its rail
# into clear space (vdd up, gnd down) so the label no longer overlaps the rail
# line or any wire. Grid multiple.
RAIL_LABEL_OFFSET = 2 * GRID  # 40

# Stage ordering left -> right by role. Each role maps to a stage column index.
ROLE_STAGE: Dict[str, int] = {
    "tail_clk_switch": 0,
    "input_pair": 1,
    "precharge1": 2,
    "regen_nmos": 3,
    "regen_pmos": 3,
    "precharge2": 4,
    "out_inv_m": 5,
    "out_inv_p": 5,
    "reset_bleed": 6,
    "unknown": 7,
}

# Within a stage, the differential "side" of a device. -1 = left (n-side),
# +1 = right (p-side), 0 = centred (single / shared device). Keyed by device id
# for the comparator; generic models fall back to net-name heuristics.
DEVICE_SIDE: Dict[str, int] = {
    "M7": 0,     # tail switch -- centred
    "M5": -1, "M6": +1,     # input pair (net0 left / net1 right)
    "M18": -1, "M2": +1,    # precharge1 (intern left / interp right)
    "M4": -1, "M3": +1,     # regen nmos (crossn left / crossp right)
    "M14": -1, "M13": +1,   # regen pmos (crossn left / crossp right)
    "M1": -1, "M12": +1,    # precharge2 (crossn left / crossp right)
    "M16": -1, "M8": -1,    # out_inv_m (outm side -- left)
    "M17": +1, "M15": +1,   # out_inv_p (outp side -- right)
    "M0": -1, "M22": +1,    # reset/bleed (intern left / interp right)
}


def symmetry_groups(model: SchematicModel) -> List[List[str]]:
    """Independent symmetry groups whose members must keep CONSISTENT (mirrored)
    orientation. A group is one ``(role, kind)`` bucket: a differential pair
    (``M5``/``M6`` ...), the two transistors of a mirrored output inverter, or a
    centred singleton (the tail switch). Returned in a deterministic order
    (first device-appearance order of each bucket).
    """
    order: List[Tuple[str, str]] = []
    members: Dict[Tuple[str, str], List[str]] = {}
    for d in model.devices:
        key = (d.role, d.kind)
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(d.id)
    return [members[k] for k in order]


@dataclass
class DevicePlacement:
    x: int                              # symbol CENTER x (grid-snapped)
    y: int                              # symbol CENTER y (grid-snapped)
    flip: bool
    box: Tuple[int, int]                # (w, h) symbol bounding box
    pins: Dict[str, Tuple[int, int]]    # term -> absolute (x, y), grid-snapped


@dataclass
class Rail:
    net: str
    x0: int
    x1: int
    y: int


@dataclass
class Port:
    """A top-level I/O port pin. ``kind`` in {"in", "out"}; ``dir`` is the
    direction the port glyph faces ("left"/"right"/"up"/"down")."""

    net: str
    kind: str
    x: int
    y: int
    dir: str


@dataclass
class Junction:
    """A real T/+ connection node: a point where THREE OR MORE wire-segment
    endpoints of the SAME net coincide."""

    net: str
    x: int
    y: int


@dataclass
class Placement:
    devices: Dict[str, DevicePlacement]
    rails: List[Rail]
    net_labels: Dict[str, Tuple[int, int]]
    net_wires: Dict[str, List[List[Tuple[int, int]]]]
    viewbox: Tuple[int, int, int, int]
    pin_index: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    ports: List[Port] = field(default_factory=list)
    junctions: List[Junction] = field(default_factory=list)


def _side_for(model: SchematicModel, dev_id: str) -> int:
    """Differential side of a device: explicit table first, then a net-name
    heuristic (``*n`` -> left, ``*p`` -> right) on the drain net."""
    if dev_id in DEVICE_SIDE:
        return DEVICE_SIDE[dev_id]
    drain = model.device(dev_id).terminals.get("drain", "")
    if drain.endswith("n") or drain.endswith("m") or drain.endswith("0"):
        return -1
    if drain.endswith("p") or drain.endswith("1"):
        return +1
    return 0


# Differential-pair net partners (left member, right member). Used to drive
# mirror-symmetric trunk routing about the centre axis.
DIFF_PAIR_PARTNERS: Dict[str, Tuple[str, int]] = {
    # net -> (partner net, side): -1 left, +1 right
    "net0": ("net1", -1), "net1": ("net0", +1),
    "intern": ("interp", -1), "interp": ("intern", +1),
    "crossn": ("crossp", -1), "crossp": ("crossn", +1),
    "outm": ("outp", -1), "outp": ("outm", +1),
}


# --- routing geometry: segment normalisation, junctions, merge ---------------

Seg = Tuple[Tuple[int, int], Tuple[int, int]]


def _net_segments(wires: List[List[Tuple[int, int]]]) -> List[Seg]:
    """Flatten a net's polylines into 2-point segments, dropping zero-length
    ones."""
    segs: List[Seg] = []
    for poly in wires:
        for a, b in zip(poly, poly[1:]):
            pa, pb = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
            if pa != pb:
                segs.append((pa, pb))
    return segs


def _net_junctions(segs: List[Seg]) -> List[Tuple[int, int]]:
    """Points where THREE OR MORE segment ENDPOINTS coincide (a real T/+ node).

    A simple 2-segment corner (exactly two endpoints meet) is NOT a junction; a
    segment merely passing THROUGH a point (no endpoint there) is not counted.
    """
    c: Counter = Counter()
    for a, b in segs:
        c[a] += 1
        c[b] += 1
    return sorted(pt for pt, n in c.items() if n >= 3)


def _merge_collinear(segs: List[Seg], keep_points: set) -> List[Seg]:
    """Collapse redundant geometry within ONE net, electrically unchanged.

    Drops zero-length segments (already dropped upstream), de-duplicates
    identical segments, and merges collinear segments that OVERLAP (share more
    than a single point). Two collinear segments that merely TOUCH at one point
    are merged ONLY when that point is not in ``keep_points`` (a junction or a
    pin we must preserve as an explicit node). Output is deterministic.
    """
    vert: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    horiz: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for a, b in segs:
        if a == b:
            continue
        if a[0] == b[0]:  # vertical
            lo, hi = sorted((a[1], b[1]))
            vert[a[0]].append((lo, hi))
        elif a[1] == b[1]:  # horizontal
            lo, hi = sorted((a[0], b[0]))
            horiz[a[1]].append((lo, hi))
        else:  # should not happen (orthogonal contract); keep verbatim
            horiz[None].append((a, b))  # type: ignore[arg-type]

    def _merge_1d(intervals: List[Tuple[int, int]], fixed: int, is_vert: bool):
        merged: List[Tuple[int, int]] = []
        for lo, hi in sorted(set(intervals)):
            if merged and lo < merged[-1][1]:
                # strict overlap -> always merge.
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            elif merged and lo == merged[-1][1]:
                # end-to-end touch: merge unless the touch point is a node to keep.
                pt = (fixed, lo) if is_vert else (lo, fixed)
                if pt in keep_points:
                    merged.append((lo, hi))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    out: List[Seg] = []
    for x in sorted(vert):
        for lo, hi in _merge_1d(vert[x], x, True):
            out.append(((x, lo), (x, hi)))
    for y in sorted(k for k in horiz if k is not None):
        for lo, hi in _merge_1d(horiz[y], y, False):
            out.append(((lo, y), (hi, y)))
    if None in horiz:  # any non-orthogonal leftovers, verbatim
        out.extend(horiz[None])  # type: ignore[arg-type]
    return out


def total_wirelength(placement: Placement) -> int:
    """Total Manhattan length of all routed net wires in a placement (the
    objective the orientation search minimises). Supply stubs to the rails are
    included; the local body tie is not a routed wire so it does not count.
    """
    total = 0
    for polys in placement.net_wires.values():
        for poly in polys:
            for (x0, y0), (x1, y1) in zip(poly, poly[1:]):
                total += abs(x1 - x0) + abs(y1 - y0)
    return int(total)


def _place_with_orientation(
    model: SchematicModel,
    flips: Dict[str, bool],
    *,
    vdd: str = "vdd",
    gnd: str = "gnd",
) -> Placement:
    """Place + route the model for a GIVEN per-device orientation map ``flips``.

    A pure function of ``(model, flips, vdd, gnd)``: each device's ``flip`` drives
    its actual pin Y (drain/source swap), then the trunk-and-stub routing,
    ports, junctions, merge, and viewbox are computed exactly as before. Used
    both for the final placement and for the symmetry-constrained orientation
    search (which calls it once per candidate orientation).
    """
    # Group devices by (stage, side) so same-stage same-side devices stack over
    # distinct sub-rows (e.g. out_inv NMOS+PMOS share a side).
    by_slot: Dict[Tuple[int, int], List[str]] = {}
    for d in model.devices:
        stage = ROLE_STAGE.get(d.role, ROLE_STAGE["unknown"])
        side = _side_for(model, d.id)
        by_slot.setdefault((stage, side), []).append(d.id)

    stages_present = {ROLE_STAGE.get(d.role, ROLE_STAGE["unknown"]) for d in model.devices}
    max_stage = max(stages_present | {0})
    # Centre axis: enough columns on each half to host the deepest stage mirror.
    center_x = snap(MARGIN + (max_stage + 1) * COL_PITCH)

    placements: Dict[str, DevicePlacement] = {}
    pin_index: Dict[str, Tuple[int, int]] = {}

    for d in model.devices:
        stage = ROLE_STAGE.get(d.role, ROLE_STAGE["unknown"])
        side = _side_for(model, d.id)
        siblings = by_slot[(stage, side)]
        sub = siblings.index(d.id)  # stable order within the slot

        # x: stage column, mirrored left/right of the centre axis by side.
        # Deeper stages sit closer to the axis (so the signal spine fans out).
        depth = (max_stage - stage + 1)
        if side < 0:
            cx = center_x - depth * COL_PITCH
        elif side > 0:
            cx = center_x + depth * COL_PITCH
        else:
            cx = center_x
        cx = snap(max(MARGIN + symbol_box(d.kind)[0] // 2, cx))

        # y: band centre by kind; sub-row offset stacks co-located devices.
        band = PMOS_BAND_Y if d.kind == "pmos" else NMOS_BAND_Y
        cy = snap(band + sub * ROW_PITCH)

        flip = flips[d.id]  # orientation DOF: drives which pin sits top/bottom
        box = symbol_box(d.kind)
        offs = pin_offsets(d.kind, flip)
        pins = {
            term: (snap(cx + ox), snap(cy + oy))
            for term, (ox, oy) in offs.items()
            if term in d.terminals
        }
        placements[d.id] = DevicePlacement(x=cx, y=cy, flip=flip, box=box, pins=pins)
        for term, xy in pins.items():
            pin_index[f"{d.id}.{term}"] = xy

    # --- rails: span the full device x-extent ---
    half_ws = [symbol_box(model.device(i).kind)[0] // 2 for i in placements]
    left_edges = [p.x - hw for p, hw in zip(placements.values(), half_ws)]
    right_edges = [p.x + hw for p, hw in zip(placements.values(), half_ws)]
    rail_x0 = snap(min(left_edges) - MARGIN)
    rail_x1 = snap(max(right_edges) + MARGIN)
    rails = [
        Rail(net=vdd, x0=rail_x0, x1=rail_x1, y=RAIL_VDD_Y),
        Rail(net=gnd, x0=rail_x0, x1=rail_x1, y=RAIL_GND_Y),
    ]

    # --- net labels + orthogonal trunk-and-stub routing -------------------
    net_labels: Dict[str, Tuple[int, int]] = {}
    net_wires: Dict[str, List[List[Tuple[int, int]]]] = {}
    # Per-net channel y (where this net's stubs converge) + trunk x-extent, kept
    # so output / clk PORT stubs can branch off the routed geometry afterwards.
    net_channel_y: Dict[str, int] = {}
    net_trunk_x: Dict[str, Tuple[int, int]] = {}

    # Assign a distinct trunk-y to each signal/clk net, deterministically, so
    # trunks never collide. Differential-pair partners SHARE a trunk-y (their
    # trunks are mirror images about the centre axis, equal length).
    trunk_y: Dict[str, int] = {}
    next_channel = [CHANNEL_TOP]

    def _channel_for(name: str) -> int:
        """Distinct trunk-y per signal net. Differential-pair partners get
        ADJACENT mirror channels (the left member just above the right member)
        so the pair reads as a symmetric stack and trunks never collide."""
        if name in trunk_y:
            return trunk_y[name]
        partner, side = DIFF_PAIR_PARTNERS.get(name, (None, 0))
        if partner is not None and partner in trunk_y:
            # mirror about the partner's channel by one step (deterministic).
            trunk_y[name] = snap(trunk_y[partner] + (CHANNEL_STEP if side > 0 else -CHANNEL_STEP))
            return trunk_y[name]
        ch = snap(next_channel[0])
        # leave room for the partner's adjacent channel.
        next_channel[0] += 2 * CHANNEL_STEP
        trunk_y[name] = ch
        return ch

    def _input_stub_net(refs: List[str]) -> Tuple[str, int] | None:
        """If ``refs`` is a single gate pin (a primary INPUT net with no other
        routed terminal), return ``(devId, side)`` for an outward input stub;
        else ``None``. The stub extends AWAY from the device (mirror-symmetric:
        left member -> left, right member -> right)."""
        if len(refs) != 1:
            return None
        dev_id, term = refs[0].rsplit(".", 1)
        if term != "gate":
            return None
        return dev_id, _side_for(model, dev_id)

    # Deterministic net visit order so channel assignment is reproducible.
    for net in model.nets:
        refs = model.terminals_of_net(net.name)
        pts = [pin_index[r] for r in refs if r in pin_index]
        if not pts:
            net_labels[net.name] = (rail_x0, RAIL_VDD_Y)
            net_wires[net.name] = []
            continue

        xs = sorted({p[0] for p in pts})
        wires: List[List[Tuple[int, int]]] = []

        stub_info = _input_stub_net(refs) if net.klass != "supply" else None
        if stub_info is not None:
            # Primary INPUT net: a short horizontal stub from the gate pin
            # extending OUTWARD to a small input-port point that carries the
            # net label. Orthogonal + grid-aligned, mirror-symmetric about the
            # centre axis (left member extends left, right member extends right)
            # so the signal-flow animation can start at the inputs.
            _dev_id, side = stub_info
            gx, gy = pts[0]
            # outward = away from the device's centre side. right member (+1)
            # mirrors to the right; everything else extends left.
            direction = +1 if side > 0 else -1
            port_x = snap(gx + direction * INPUT_STUB_LEN)
            wires.append([(gx, gy), (port_x, gy)])
            net_labels[net.name] = (port_x, gy)
            net_wires[net.name] = wires
            continue

        if net.klass == "supply":
            # Supply nets ride their rail; vdd stubs go UP, gnd stubs go DOWN.
            ty = RAIL_VDD_Y if net.name == vdd else RAIL_GND_Y
            if len(xs) > 1:
                wires.append([(xs[0], ty), (xs[-1], ty)])
            for px, py in sorted(pts):
                if py != ty:
                    wires.append([(px, py), (px, ty)])
            label_x = snap(xs[0])
            net_labels[net.name] = (label_x, ty)
        else:
            # signal / clk: one horizontal trunk in an assigned channel spanning
            # this net's own pin extent (so both far endpoints coincide with a
            # real pin), with a vertical stub from the trunk to each pin. The
            # trunk base of every stub lies on the trunk, so the viewer's symbol
            # leads meet the wires exactly.
            ty = _channel_for(net.name)
            tx0, tx1 = xs[0], xs[-1]
            if tx0 != tx1:
                wires.append([(tx0, ty), (tx1, ty)])
            for px, py in sorted(pts):
                if py != ty:
                    wires.append([(px, py), (px, ty)])
            # Label near the trunk's left end.
            label_x = snap(tx0)
            net_labels[net.name] = (label_x, ty)
            net_channel_y[net.name] = ty
            net_trunk_x[net.name] = (tx0, tx1)

        net_wires[net.name] = wires

    # --- I/O PORTS -----------------------------------------------------------
    # Build the top-level port pins: the three INPUTS (net0 / net1 / clk) and the
    # two OUTPUTS (outm / outp). net0 / net1 already terminate at an input-port
    # point (the far end of their gate input stub); clk gets its OWN short input
    # stub branched off its trunk; outm / outp each get a short output stub so
    # the output carries a port pin.
    ports: List[Port] = []

    # net0 / net1 inputs: the port is the far end of their gate input stub.
    for in_net in ("net0", "net1"):
        if in_net not in net_wires or not net_wires[in_net]:
            continue
        lx, ly = net_labels[in_net]
        # the stub starts at the gate pin; the port faces away from the device.
        stub = net_wires[in_net][0]
        gx = stub[0][0] if (stub[0][0], stub[0][1]) != (lx, ly) else stub[1][0]
        facing = "left" if lx < gx else "right"
        ports.append(Port(net=in_net, kind="in", x=lx, y=ly, dir=facing))

    # clk input: a short stub branched LEFT off the clk trunk's left end (mirror
    # neutral, clear of every device column), with a labelled input port pin.
    clk_net = "clk"
    if clk_net in net_channel_y:
        cty = net_channel_y[clk_net]
        ctx0 = net_trunk_x[clk_net][0]
        clk_port_x = snap(ctx0 - CLK_PORT_STUB_LEN)
        net_wires[clk_net].append([(clk_port_x, cty), (ctx0, cty)])
        net_labels[clk_net] = (clk_port_x, cty)
        ports.append(Port(net=clk_net, kind="in", x=clk_port_x, y=cty, dir="left"))

    # outm / outp outputs: a short stub branched off the channel point, mirrored
    # (left output -> left, right output -> right), with a labelled output port.
    for out_net in ("outm", "outp"):
        if out_net not in net_channel_y:
            continue
        oty = net_channel_y[out_net]
        # the channel point sits at the common pin x of the output stubs.
        pin_xs = [poly[0][0] for poly in net_wires[out_net]] or [net_trunk_x[out_net][0]]
        ox = pin_xs[0]
        _, side = DIFF_PAIR_PARTNERS.get(out_net, (None, -1))
        direction = +1 if side > 0 else -1
        port_x = snap(ox + direction * OUTPUT_STUB_LEN)
        net_wires[out_net].append([(ox, oty), (port_x, oty)])
        net_labels[out_net] = (port_x, oty)
        facing = "right" if direction > 0 else "left"
        ports.append(Port(net=out_net, kind="out", x=port_x, y=oty, dir=facing))

    # --- vdd / gnd NET LABELS off the rail -----------------------------------
    # Push each supply label PERPENDICULARLY off its rail into clear space (vdd
    # up, gnd down) so it no longer overlaps the rail line or any wire.
    for sup_net, off in ((vdd, -RAIL_LABEL_OFFSET), (gnd, +RAIL_LABEL_OFFSET)):
        if sup_net in net_labels:
            lx, ly = net_labels[sup_net]
            net_labels[sup_net] = (lx, snap(ly + off))

    # --- ROUTING OPTIMIZATION: merge redundant geometry per net --------------
    # Preserve pin points and junction nodes as explicit endpoints; collapse
    # zero-length / duplicate / overlapping collinear segments. Electrically
    # identical (same pin coverage, still orthogonal). Then compute junctions on
    # the FINAL merged geometry.
    junctions: List[Junction] = []
    segments_removed = 0
    for net in model.nets:
        raw = net_wires.get(net.name, [])
        segs = _net_segments(raw)
        before = len(segs)
        # points to keep as explicit nodes: every pin on this net + every junction.
        net_pin_pts = {pin_index[r] for r in model.terminals_of_net(net.name) if r in pin_index}
        pre_junctions = set(_net_junctions(segs))
        keep = net_pin_pts | pre_junctions
        merged = _merge_collinear(segs, keep)
        segments_removed += before - len(merged)
        net_wires[net.name] = [[a, b] for a, b in merged]
        for jx, jy in _net_junctions(merged):
            junctions.append(Junction(net=net.name, x=jx, y=jy))

    _place_with_orientation.segments_removed = segments_removed  # type: ignore[attr-defined]

    # --- viewbox: fit everything with a margin --------------------------------
    ys_all: List[int] = [RAIL_VDD_Y, RAIL_GND_Y]
    xs_all: List[int] = [rail_x0, rail_x1]
    for i, p in placements.items():
        _, h = symbol_box(model.device(i).kind)
        ys_all.append(p.y - h // 2)
        ys_all.append(p.y + h // 2)
    for ty in trunk_y.values():
        ys_all.append(ty)
    # ports + (offset) labels can sit outside the device band -> include them.
    for prt in ports:
        xs_all.append(prt.x)
        ys_all.append(prt.y)
    for lx, ly in net_labels.values():
        xs_all.append(lx)
        ys_all.append(ly)
    vb_x = snap(min(xs_all) - MARGIN)
    vb_y = snap(min(ys_all) - MARGIN)
    vb_w = snap((max(xs_all) + MARGIN) - vb_x)
    vb_h = snap((max(ys_all) + MARGIN) - vb_y)
    viewbox = (vb_x, vb_y, vb_w, vb_h)

    return Placement(
        devices=placements,
        rails=rails,
        net_labels=net_labels,
        net_wires=net_wires,
        viewbox=viewbox,
        pin_index=pin_index,
        ports=ports,
        junctions=junctions,
    )


def _default_flips(model: SchematicModel) -> Dict[str, bool]:
    """The supply-side-convention orientation for every device (PMOS source-up,
    NMOS source-down). The anchor the orientation search starts from."""
    return {d.id: default_flip(d.kind) for d in model.devices}


def place(model: SchematicModel, *, vdd: str = "vdd", gnd: str = "gnd") -> Placement:
    """Deterministically place every device on the grid (CENTER coords) +
    compute every terminal pin (x, y) = center + symbol offset, plus a vdd rail
    (top) and a gnd rail (bottom). Routing is mirror-symmetric trunk-and-stub.

    DEVICE ORIENTATION is optimised under symmetry. Starting from the supply-side
    convention (PMOS source-up, NMOS source-down -- :func:`_default_flips`), each
    independent :func:`symmetry_groups` group is tried in BOTH consistent
    (mirrored) orientations; the lower total-Manhattan-wirelength option is kept
    (ties -> the convention). The whole group flips together, so a differential
    pair / mirrored inverter never splits orientation. The before / after total
    wirelength is recorded on ``place.wirelength_before`` / ``place.wirelength_after``
    (and ``place.wirelength_saved``); ``place.flips`` is the chosen orientation
    map. The search NEVER increases wirelength versus the all-default orientation.
    """
    groups = symmetry_groups(model)

    # Anchor orientation = supply-side convention; baseline wirelength.
    flips = _default_flips(model)
    best = _place_with_orientation(model, flips, vdd=vdd, gnd=gnd)
    best_wl = total_wirelength(best)
    baseline_wl = best_wl

    # Symmetry-constrained greedy search: for each independent group, try
    # flipping the WHOLE group's orientation consistently and keep it only if it
    # strictly reduces total wirelength. Deterministic (groups in model order;
    # strict-improvement keep so a tie stays on the convention).
    for group in groups:
        trial = dict(flips)
        for dev_id in group:
            trial[dev_id] = not trial[dev_id]
        cand = _place_with_orientation(model, trial, vdd=vdd, gnd=gnd)
        cand_wl = total_wirelength(cand)
        if cand_wl < best_wl:
            flips = trial
            best = cand
            best_wl = cand_wl

    # Re-place once at the chosen orientation so segments_removed reflects it.
    final = _place_with_orientation(model, flips, vdd=vdd, gnd=gnd)
    place.segments_removed = _place_with_orientation.segments_removed  # type: ignore[attr-defined]
    place.wirelength_before = int(baseline_wl)  # type: ignore[attr-defined]
    place.wirelength_after = int(best_wl)  # type: ignore[attr-defined]
    place.wirelength_saved = int(baseline_wl - best_wl)  # type: ignore[attr-defined]
    place.flips = dict(flips)  # type: ignore[attr-defined]
    return final


__all__ = [
    "DevicePlacement",
    "Rail",
    "Port",
    "Junction",
    "Placement",
    "place",
    "total_wirelength",
    "symmetry_groups",
    "ROLE_STAGE",
    "DEVICE_SIDE",
    "DIFF_PAIR_PARTNERS",
]
