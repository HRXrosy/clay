"""Full 12-net mirror-symmetric routing of the 17-device comparator (two-row NMOS).

Places every device at its TRUE (nf, W, L) from ``COMP_DEVICE_PLAN`` in a TWO-ROW-
NMOS floorplan that halves the cell WIDTH vs a single transistor row: the four
widest NMOS pairs no longer share one row. Stack bottom->top is "N N P" -- a TUCK-N
band (output-inverter pull-down + reset, gnd-tied) just above the gnd floor, the
on-axis tail M7, a CORE-N band (input pair + cross-coupled latch) bordering the
signal channel, then the pulled-in PMOS row bordering it from above, vdd ceiling.
This module adds the COMPLETE signal routing so all 12 nets connect per schematic:

  * device terminals stay on metal1 (34/0); CORE-N (drains/gates up) and PMOS
    (flipped, drains/gates down) BORDER the inter-row signal channel; a gnd rail on
    the floor + a vdd rail on the ceiling, so no signal crosses a rail;
  * each SIGNAL net runs on a horizontal metal2 (36/0) trunk; channel-bordering
    terminals rise on M3 (intern/interp, outm/outp, clk) or M4 (cross-coupling
    crossn/crossp); the TUCK-N band's six signal terminals (out-inv drains, cross
    gates, reset gates) escape UP on dedicated jogged M4 lanes clear of the
    intern/interp trunk -- the short-mode the single-band router hit;
  * a global tap planner gives every riser a clear x-lane (same-net risers clear by
    the M3.2a gap or fully merge); a grounded M2 clk shield with SPAN-RELATIVE gnd
    feeds (derived from the narrower cell's half-width) balances clk into the regen
    pair without colliding with the out-inverter tap column;
  * the differential pairs sit on mirror-x tracks/lanes, so they stay parasitically
    balanced (the property v183's density-first router loses).

Verified on the v173 union-find (every schematic net one component, no two distinct
nets merge) and on real gf180 DRC (0 violations) + Magic (all 12 nets extract
distinct; differential / coupling balance held). ~42.9x25.4um (~1089um2), ~19%
smaller than the single-row 52.5x25.7um (~1348um2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .backend import LayerPolygon, Point, Polygon, _mirror_x, _translate
from .tech_geometry import M2_KEY, VIA1_KEY, via12_stack
from .tech_geometry import M3_KEY, VIA2_KEY, via23_stack
from .tech_geometry import M4_KEY, VIA3_KEY, via34_stack
from .full_symmetric_comparator import (
    _DEV, _PAIR_LEFT, _build_dev, _rect, _mirror_box,
    HALF_GAP_NM, ROW_PITCH_NM, RAIL_W_NM, M1, M1_LABEL,
)

Bbox = Tuple[int, int, int, int]

# pair name -> RIGHT (mirror) device label (p-side); LEFT is _PAIR_LEFT (n-side).
_PAIR_RIGHT = {
    "M5_M6": "M6", "M3_M4": "M3", "M16_M17": "M17", "M0_M22": "M22",
    "M13_M14": "M13", "M18_M2": "M2", "M1_M12": "M12", "M8_M15": "M15",
}
# TWO-ROW-NMOS, two-row-PMOS floorplan. Splitting each transistor row in two
# halves the cell WIDTH (the four widest pairs no longer share one row), at the
# cost of two extra y-bands. Net: ~33.8x31.3um (~1058um2) vs the single-row
# 52.5x25.7um (~1348um2). Engineer's intent stacked bottom->top is "N N P P":
#   gnd floor | TUCK-N (out-pd + reset, all gnd sources) | tail M7 on-axis |
#   CORE-N (input pair + latch-N, borders the channel) | signal channel |
#   PMOS-A (latch-P + out-pu, borders the channel) | PMOS-B (precharge) | vdd.
# The cross-coupling pair (crossn/crossp) touches CORE-N, PMOS-A (both border the
# channel -> short risers) and the far bands PMOS-B + TUCK-N (long risers that
# must be steered to dedicated lanes clear of the intern/interp trunk).
_CORE_N_ORDER = ["M5_M6", "M3_M4"]     # input pair + cross-coupled NMOS latch
_TUCK_N_ORDER = ["M16_M17", "M0_M22"]  # output-inverter NMOS + reset (gnd-tied)
_PMOS_ORDER = ["M13_M14", "M18_M2", "M1_M12", "M8_M15"]   # single PMOS row
_TUCK_LABELS = {"M16", "M17", "M0", "M22"}
PMOS_X_OFFSET_NM = 0        # PMOS pulled in directly over the core (no x de-align)
CHAN_GAP_NM = 1300          # inter-device spacing -> vertical routing channels
CHANNEL_EXTRA_NM = 9000     # inter-row signal-channel height (fits 5 trunk tracks)
TRACK_PITCH_NM = 1500       # vertical spacing between signal trunks in the channel


def _sh(b: Bbox, dx: int, dy: int) -> Bbox:
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def _mb(b: Bbox) -> Bbox:               # x-mirror a bbox about x=0
    return (-b[2], b[1], -b[0], b[3])


def _mirror_y(polys: List[LayerPolygon], axis_y: int) -> List[LayerPolygon]:
    return [LayerPolygon(layer=lp.layer, datatype=lp.datatype,
                         polygon=Polygon(vertices=list(reversed(
                             [Point(v.x, 2 * axis_y - v.y) for v in lp.polygon.vertices]))))
            for lp in polys]


def _mby(b: Bbox, axis_y: int) -> Bbox:     # y-mirror a bbox about axis_y
    return (b[0], 2 * axis_y - b[3], b[2], 2 * axis_y - b[1])


def _ctr(b: Bbox) -> Tuple[int, int]:
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


@dataclass
class Placement:
    polys: List[LayerPolygon] = field(default_factory=list)
    term: Dict[str, Dict[str, Bbox]] = field(default_factory=dict)   # label -> {s,d,g}
    geom: Dict[str, int] = field(default_factory=dict)


def _band_height(order) -> int:
    """Total vertical footprint (source rail bottom .. drain/gate top) of a band's
    devices when their comp bottoms sit at y=0 -- used to stack bands clear."""
    lo = hi = 0
    for name in order:
        dev, _ = _build_dev(_PAIR_LEFT[name])
        for b in (dev.source_rail_bbox, dev.drain_rail_bbox,
                  dev.gate_contact_bbox, dev.comp_bbox):
            lo = min(lo, b[1]); hi = max(hi, b[3])
    return hi - lo


def place_devices() -> Placement:
    """TWO-ROW-NMOS floorplan with a pulled-in single PMOS row (cuts the WIDTH:
    the four widest NMOS pairs no longer share one row).

    Stack bottom->top: gnd floor | TUCK-N band (output-inverter pull-down M16/M17 +
    reset M0/M22, every source on gnd -> a straight drop to the floor) | on-axis
    tail M7 | CORE-N band (input pair M5/M6 + cross-coupled NMOS latch M3/M4,
    drains/gates face UP into the channel) | signal channel | PMOS row (vertically
    flipped so sources face the vdd ceiling and drains/gates face DOWN into the
    channel, pulled in directly over the core) | vdd ceiling.

    Records per-label source/drain/gate bbox. CORE-N (from below) and PMOS (from
    above) BORDER the channel -> short risers to the trunks. The TUCK-N band sits
    below the core: its gnd terminals drop to the floor, and its four SIGNAL
    terminals (out-inv drains outm/outp, cross-gates crossp/crossn) escape UP
    through dedicated vertical corridors between the core devices, in lanes clear of
    the intern/interp trunk (the two short-modes the single-band router hit)."""
    polys: List[LayerPolygon] = []
    term: Dict[str, Dict[str, Bbox]] = {}
    vss_y = 0

    def place_one(label, mirror_label, dev, right_edge, row_y, flip=False) -> int:
        dx = right_edge - dev.comp_bbox[2]
        dy = row_y - dev.comp_bbox[1]
        left = _translate(list(dev.polygons), dx, dy)
        s = _sh(dev.source_rail_bbox, dx, dy)
        d = _sh(dev.drain_rail_bbox, dx, dy)
        g = _sh(dev.gate_contact_bbox, dx, dy)
        comp = _sh(dev.comp_bbox, dx, dy)
        if flip:                                    # PMOS: source->ceiling, drain->channel
            ay = (comp[1] + comp[3]) // 2
            left = _mirror_y(left, ay)
            s, d, g = _mby(s, ay), _mby(d, ay), _mby(g, ay)
        polys.extend(left)
        polys.extend(_mirror_x(list(left), 0))
        term[label] = {"s": s, "d": d, "g": g}
        term[mirror_label] = {"s": _mb(s), "d": _mb(d), "g": _mb(g)}
        return comp[0]

    def place_band(order, row_y, flip=False):
        right_edge = -HALF_GAP_NM
        for name in order:
            dev, _ = _build_dev(_PAIR_LEFT[name])
            cl = place_one(_PAIR_LEFT[name], _PAIR_RIGHT[name], dev, right_edge, row_y, flip=flip)
            right_edge = cl - CHAN_GAP_NM

    # --- y-band plan (comp-bottom of each band) -------------------------------
    # TUCK-N comp bottom: the device source rail sits ~490 below the comp, so a
    # comp bottom of 1100 puts the source bottom at ~610 -- >= the 230 nm M1.2a
    # spacing above the gnd rail top (y=0) so the source rails clear the rail.
    tuck_base = vss_y + 1100                       # TUCK-N just above the gnd floor
    tuck_h = _band_height(_TUCK_N_ORDER)
    tail_band = tuck_base + tuck_h + 700          # on-axis tail above the tuck row
    tail_h = _band_height(["M5_M6"])              # tail sized like the input pair band
    nmos_base = tail_band + tail_h + 500          # CORE-N borders the channel
    core_h = _band_height(_CORE_N_ORDER)
    pmos_base = nmos_base + core_h + CHANNEL_EXTRA_NM    # PMOS across the channel

    # CORE-N + TUCK-N (NMOS, not flipped: drains/gates face up).
    place_band(_TUCK_N_ORDER, tuck_base)
    place_band(_CORE_N_ORDER, nmos_base)

    tail, _ = _build_dev("M7")
    tcx = (tail.comp_bbox[0] + tail.comp_bbox[2]) // 2
    tdy = tail_band - tail.comp_bbox[1]
    polys.extend(_translate(list(tail.polygons), -tcx, tdy))
    term["M7"] = {
        "s": _sh(tail.source_rail_bbox, -tcx, tdy),
        "d": _sh(tail.drain_rail_bbox, -tcx, tdy),
        "g": _sh(tail.gate_contact_bbox, -tcx, tdy),
    }

    # PMOS row (flipped: sources face the vdd ceiling, drains/gates face down).
    place_band(_PMOS_ORDER, pmos_base, flip=True)

    # merge the per-PMOS N-wells into ONE continuous well (all PMOS share a well --
    # standard, and it removes the NW.2b inter-well spacing rule that otherwise forces
    # the PMOS devices far apart). Cover the union of the per-device (21,0) wells.
    nw = [lp for lp in polys if (lp.layer, lp.datatype) == (21, 0)]
    if nw:
        wx0 = min(v.x for lp in nw for v in lp.polygon.vertices)
        wy0 = min(v.y for lp in nw for v in lp.polygon.vertices)
        wx1 = max(v.x for lp in nw for v in lp.polygon.vertices)
        wy1 = max(v.y for lp in nw for v in lp.polygon.vertices)
        polys.append(_rect((21, 0), (wx0, wy0, wx1, wy1)))

    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    span = max(abs(min(xs)), abs(max(xs))) + 1200
    vdd_y = max(ys) + 1500                           # vdd rail on the ceiling
    # signal channel bounds: top of CORE-N drains .. bottom of (flipped) PMOS drains
    channel_lo = max(term[l]["d"][3] for l in ("M5", "M4"))
    channel_hi = min(term[l]["d"][1] for l in ("M14", "M18", "M1", "M8"))
    return Placement(polys=polys, term=term,
                     geom=dict(vss_y=vss_y, tail_band=tail_band, nmos_base=nmos_base,
                               pmos_base=pmos_base, vdd_y=vdd_y, span=span,
                               tuck_base=tuck_base,
                               channel_lo=channel_lo, channel_hi=channel_hi))


# === net membership (authoritative, from COMP_DEVICE_PLAN) ====================
def _net_terminals() -> Dict[str, List[Tuple[str, str]]]:
    nets: Dict[str, List[Tuple[str, str]]] = {}
    for label, (lbl, kind, d, g, s, nf, wf, lu) in _DEV.items():
        for net, tk in ((d, "d"), (g, "g"), (s, "s")):
            nets.setdefault(net, []).append((label, tk))
    return nets


NET_TERMINALS = _net_terminals()
MW = 400            # routing metal width (nm); half-width below
_HMW = MW // 2


def _lp_to_tagged(lp: LayerPolygon, net: str):
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (net, (lp.layer, lp.datatype), (min(xs), min(ys), max(xs), max(ys)))


def _vrect(layer, cx, y0, y1) -> LayerPolygon:
    return _rect(layer, (cx - _HMW, min(y0, y1), cx + _HMW, max(y0, y1)))


def _hrect(layer, x0, x1, cy) -> LayerPolygon:
    return _rect(layer, (min(x0, x1), cy - _HMW, max(x0, x1), cy + _HMW))


def _full_via(cx: int, cy: int) -> List[LayerPolygon]:
    """M1<->M2<->M3 via stack at a point (taps a device M1 terminal up to M3)."""
    return via12_stack(Point(cx, cy)) + via23_stack(Point(cx, cy))


def route_comb(net: str, centers: List[Tuple[int, int]], track_y: int):
    """2-layer Manhattan comb (no same-layer crossings between different nets):
    a HORIZONTAL trunk on M2 at ``track_y``, a VERTICAL riser on M3 to each
    terminal, a via23 at each trunk<->riser junction, and an M1<->M2<->M3 full via
    stack tapping each terminal's M1. Because horizontals are M2 and verticals are
    M3, a different net's trunk crossing this net's riser never shorts."""
    polys: List[LayerPolygon] = []
    tagged: List = []
    xs = [c[0] for c in centers]
    trunk = _hrect(M2_KEY, min(xs) - _HMW, max(xs) + _HMW, track_y)
    polys.append(trunk); tagged.append(_lp_to_tagged(trunk, net))
    for (cx, cy) in centers:
        riser = _vrect(M3_KEY, cx, track_y, cy)
        polys.append(riser); tagged.append(_lp_to_tagged(riser, net))
        for vp in via23_stack(Point(cx, track_y)) + _full_via(cx, cy):
            polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
    return polys, tagged


def route_clk(net: str, centers: List[Tuple[int, int]], track_y: int,
              axis_terminals=()):
    """Clk comb routed MIRROR-SYMMETRIC about x=0 (so clk couples EQUALLY into the
    two regen nodes and into net0/net1 -> its kick cancels). Same 2-layer Manhattan
    construction as ``route_comb`` (M2 trunk + M3 risers), except the on-axis tail
    gate (``axis_terminals``: terminals near x=0 whose long riser would otherwise run
    asymmetrically past one input) is brought up on the AXIS: a short M3 jog at the
    terminal's y to x=0, then the long vertical riser at x=0 -- equidistant from the
    mirror-paired victims, so the parallel-run coupling is symmetric."""
    polys: List[LayerPolygon] = []
    tagged: List = []
    axis_set = set(axis_terminals)
    xs = [c[0] for c in centers]
    trunk = _hrect(M2_KEY, min(xs) - _HMW, max(xs) + _HMW, track_y)
    polys.append(trunk); tagged.append(_lp_to_tagged(trunk, net))
    for (cx, cy) in centers:
        if (cx, cy) in axis_set and cx != 0:
            # tap the terminal M1 up to M3, jog on M3 to the axis, rise on the axis.
            for vp in _full_via(cx, cy):
                polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
            jog = _hrect(M3_KEY, cx, 0, cy)
            polys.append(jog); tagged.append(_lp_to_tagged(jog, net))
            riser = _vrect(M3_KEY, 0, track_y, cy)
            polys.append(riser); tagged.append(_lp_to_tagged(riser, net))
            for vp in via23_stack(Point(0, track_y)):
                polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
            continue
        riser = _vrect(M3_KEY, cx, track_y, cy)
        polys.append(riser); tagged.append(_lp_to_tagged(riser, net))
        for vp in via23_stack(Point(cx, track_y)) + _full_via(cx, cy):
            polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
    return polys, tagged


def _plan_lane(net, ty, track_y, occupied, near_x, reach=4200):
    """Pick a free vertical M4 lane x within ``reach`` of ``near_x`` (ANY x, not
    confined to a terminal rail) clear of every different-net occupied lane over
    [ty, track_y]. The jog lets a cross-coupling riser sidestep the immovable clk/
    intern gate-via columns; register the chosen lane so later risers avoid it."""
    y0, y1 = min(ty, track_y), max(ty, track_y)
    step = 150
    cands = [near_x] + [near_x + s * step * k
                        for k in range(1, reach // step + 1) for s in (1, -1)]
    for x in cands:
        probe = (x - _HMW - _SPACE, x + _HMW + _SPACE, y0 - _SPACE, y1 + _SPACE)
        if any(o[4] != net and _overlap(probe, o[:4]) for o in occupied):
            continue
        occupied.append((x - _HMW, x + _HMW, y0, y1, net))
        return x
    occupied.append((near_x - _HMW, near_x + _HMW, y0, y1, net))
    return near_x


def _plan_lane_mirror(ty, cn_track, cp_track, occupied, near_x, reach=4200):
    """Pick a crossn lane x such that BOTH the crossn riser (lane x over [ty,
    cn_track]) AND its exact crossp mirror (lane -x over [ty, cp_track]) are clear
    of every different-net occupied lane. crossp is built as the x-mirror of crossn
    for parasitic balance, but the two cross trunks now ride DIFFERENT y-tracks, so
    a crossn lane whose mirror is fine against other crossn lanes can still drive a
    crossp via23 M3 pad into a (lower) intern/interp M3 gate riser. Checking both
    images here keeps the mirror exact while clearing both M4 risers and their M3
    via pads. Registers both images so later lanes avoid them."""
    cy0, cy1 = min(ty, cn_track), max(ty, cn_track)
    py0, py1 = min(ty, cp_track), max(ty, cp_track)
    step = 150
    cands = [near_x] + [near_x + s * step * k
                        for k in range(1, reach // step + 1) for s in (1, -1)]
    for x in cands:
        cn_probe = (x - _HMW - _SPACE, x + _HMW + _SPACE, cy0 - _SPACE, cy1 + _SPACE)
        cp_probe = (-x - _HMW - _SPACE, -x + _HMW + _SPACE, py0 - _SPACE, py1 + _SPACE)
        blocked = any(o[4] != "crossn" and _overlap(cn_probe, o[:4]) for o in occupied) \
            or any(o[4] != "crossp" and _overlap(cp_probe, o[:4]) for o in occupied)
        if blocked:
            continue
        occupied.append((x - _HMW, x + _HMW, cy0, cy1, "crossn"))
        occupied.append((-x - _HMW, -x + _HMW, py0, py1, "crossp"))
        return x
    occupied.append((near_x - _HMW, near_x + _HMW, cy0, cy1, "crossn"))
    occupied.append((-near_x - _HMW, -near_x + _HMW, py0, py1, "crossp"))
    return near_x


def route_comb_m4(net: str, taps, track_y: int):
    """Cross-coupling comb: verticals on M4 with a short horizontal JOG. Each
    terminal is tapped (M1..M4 stack) at (tx, ty), jogs on M4 to a nearby clear lane
    x, then rises on M4 to the M2 trunk (M2..M4 junction). ``taps`` = (tx, ty, lane).
    M4 verticals never share the M3 layer with intern/outm/clk; the jog clears the
    immovable clk gate-via columns; crossn/crossp ride mirror-x + offset tracks."""
    polys: List[LayerPolygon] = []
    tagged: List = []
    xs = [lx for (_tx, _ty, lx) in taps]
    trunk = _hrect(M2_KEY, min(xs) - _HMW, max(xs) + _HMW, track_y)
    polys.append(trunk); tagged.append(_lp_to_tagged(trunk, net))
    for (tx, ty, lx) in taps:
        for vp in _full_via(tx, ty) + via34_stack(Point(tx, ty)):   # terminal M1..M4
            polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
        if lx != tx:
            jog = _hrect(M4_KEY, min(tx, lx) - _HMW, max(tx, lx) + _HMW, ty)
            polys.append(jog); tagged.append(_lp_to_tagged(jog, net))
        riser = _vrect(M4_KEY, lx, ty, track_y)
        polys.append(riser); tagged.append(_lp_to_tagged(riser, net))
        for vp in via23_stack(Point(lx, track_y)) + via34_stack(Point(lx, track_y)):
            polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
    return polys, tagged


def _plan_tuck_lane(net, mnet, ty, track_y, m_track_y, occupied, near_x, reach=12000):
    """Pick an x-lane for a TUCK-row signal riser (a long M4 climb from a terminal
    below the core row up to the net's channel trunk) such that BOTH the lane x over
    [ty, track_y] for ``net`` AND its exact mirror -x over [ty, m_track_y] for the
    mirror net ``mnet`` clear every different-net occupied lane. Returns the chosen
    x; registers both images so later lanes avoid them. The mirror images keep the
    differential pair (e.g. outm/outp, crossn/crossp gates) parasitically matched."""
    y0, y1 = min(ty, track_y), max(ty, track_y)
    my0, my1 = min(ty, m_track_y), max(ty, m_track_y)
    step = 150
    cands = [near_x] + [near_x + s * step * k
                        for k in range(1, reach // step + 1) for s in (1, -1)]
    for x in cands:
        probe = (x - _HMW - _SPACE, x + _HMW + _SPACE, y0 - _SPACE, y1 + _SPACE)
        mprobe = (-x - _HMW - _SPACE, -x + _HMW + _SPACE, my0 - _SPACE, my1 + _SPACE)
        # clear EVERY lane (same-net included): a tuck riser parallel to but not
        # merged with its own net's main M3 riser still violates M3.2a (0.28um).
        # The same-net trunk join happens via the M2 connector on a different layer.
        blocked = any(_overlap(probe, o[:4]) for o in occupied) \
            or any(_overlap(mprobe, o[:4]) for o in occupied)
        if blocked:
            continue
        occupied.append((x - _HMW, x + _HMW, y0, y1, net))
        occupied.append((-x - _HMW, -x + _HMW, my0, my1, mnet))
        return x
    occupied.append((near_x - _HMW, near_x + _HMW, y0, y1, net))
    occupied.append((-near_x - _HMW, -near_x + _HMW, my0, my1, mnet))
    return near_x


def route_tuck_riser(net, tx, ty, lx, track_y, trunk_reach_x=None):
    """Climb ONE tuck-row signal terminal up to its channel trunk on M4: tap the
    terminal M1 up the via stack to M4, jog on M4 from the terminal x ``tx`` to the
    clear lane ``lx``, rise on M4 to the trunk y, land on M2, then run an M2 segment
    from the lane to ``trunk_reach_x`` so it OVERLAPS the (already-built) M2 trunk and
    joins the net (the trunk's own taps may be far from the tuck lane). M4 verticals
    cross the M2 trunks and M3 risers without shorting (different layers, no via at
    the crossing); the only contacts are net-local via stacks + the same-net M2."""
    polys: List[LayerPolygon] = []
    tagged: List = []
    for vp in _full_via(tx, ty) + via34_stack(Point(tx, ty)):       # terminal M1..M4
        polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
    if lx != tx:
        jog = _hrect(M4_KEY, min(tx, lx) - _HMW, max(tx, lx) + _HMW, ty)
        polys.append(jog); tagged.append(_lp_to_tagged(jog, net))
        for vp in via34_stack(Point(lx, ty)):                       # M3<->M4 at the lane foot
            polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
    riser = _vrect(M4_KEY, lx, ty, track_y)
    polys.append(riser); tagged.append(_lp_to_tagged(riser, net))
    for vp in via23_stack(Point(lx, track_y)) + via34_stack(Point(lx, track_y)):
        polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
    # same-net M2 connector from the lane to the existing trunk (so the riser joins
    # the net even when the trunk's nearest tap is far from the tuck lane).
    if trunk_reach_x is not None and trunk_reach_x != lx:
        conn = _hrect(M2_KEY, min(lx, trunk_reach_x), max(lx, trunk_reach_x), track_y)
        polys.append(conn); tagged.append(_lp_to_tagged(conn, net))
    return polys, tagged


def _tap(p: Placement, label: str, tk: str) -> Tuple[int, int]:
    """Tap point for a device terminal: a SOURCE is tapped at its axis-facing edge
    (the wide source rail extends past the centered drain, so its riser never
    collides with the same device's drain riser); drain/gate are tapped at centre."""
    b = p.term[label][tk]
    cy = (b[1] + b[3]) // 2
    cx = _inner_x(b) if tk == "s" else (b[0] + b[2]) // 2
    return (cx, cy)


def _inner_x(b: Bbox, margin: int = 350) -> int:
    """x near the axis-facing edge of a terminal rail (clear of the centered drain
    and the left-edge gate). Symmetric: left device -> right edge, right -> left."""
    cx = (b[0] + b[2]) // 2
    if cx < -50:
        return b[2] - margin
    if cx > 50:
        return b[0] + margin
    return cx


def _device_terminal_tags(p: Placement):
    """Tag every device terminal rail (M1) with its schematic net for union-find."""
    tags = []
    for net, terms in NET_TERMINALS.items():
        for (label, tk) in terms:
            tags.append((net, M1, p.term[label][tk]))
    return tags


def _shield_feed_x(p: Placement) -> int:
    """SPAN-RELATIVE x for the gnd shield feed risers: just OUTSIDE the outermost
    device/routing column, inside the cell span. Hard-coding it (the old +/-20000)
    drove the M1 feed straight through the out-inverter PMOS tap column once the
    two-row cell narrowed -- the feed must be derived from the actual half-width."""
    xs = [v.x for lp in p.polys for v in lp.polygon.vertices]
    dev_edge = max(abs(min(xs)), abs(max(xs)))    # outermost device/well x
    feed = dev_edge + 900                          # clear column beyond every device
    return min(feed, p.geom["span"] - _HMW - 100)  # but inside the rail span


def route_clk_shield(p: Placement):
    """Grounded shield between the high clk M2 trunk and the cross-coupling M2
    trunks. The clk trunk sits near the PMOS gates, so it is closer to the (higher)
    crossp trunk than to the (lower) crossn trunk -> clk couples unequally into the
    regen pair. A full-width grounded M2 strip placed in that gap intercepts the
    clk->crossp field on the near side, pulling clk~crossp down toward clk~crossn.

    The strip ties to the floor gnd rail through two MIRROR-SYMMETRIC M1 feed risers
    at SPAN-RELATIVE columns (``_shield_feed_x``: just beyond the outermost device,
    inside the span -- not the old hard-coded +/-20000 that collided with the
    narrower two-row cell's out-inverter tap), so the shield is itself x-symmetric
    and extracts as gnd (one component with the rail -- not floating, not a short)."""
    g = p.geom
    polys: List[LayerPolygon] = []
    tagged = []
    # midway between the cross-pair band top (crossp track) and the clk track.
    m3_tracks, m4_tracks = _tracks(g)
    sh_y = (max(m4_tracks.values()) + m3_tracks["clk"]) // 2
    span = g["span"]
    strip = _hrect(M2_KEY, -span, span, sh_y)
    polys.append(strip); tagged.append(_lp_to_tagged(strip, "gnd"))
    feed_x = _shield_feed_x(p)
    for fx in (-feed_x, feed_x):
        feed = _rect(M1, (fx - _HMW, g["vss_y"] - RAIL_W_NM, fx + _HMW, sh_y + _HMW))
        polys.append(feed); tagged.append(_lp_to_tagged(feed, "gnd"))
        for vp in via12_stack(Point(fx, sh_y)):
            polys.append(vp); tagged.append(_lp_to_tagged(vp, "gnd"))
    return polys, tagged


def route_supply(p: Placement):
    """gnd / vdd M1 rails + a per-terminal M1 drop (inner-edge tap) to the rail."""
    g = p.geom
    polys: List[LayerPolygon] = []
    tagged = []
    gnd_rail = _rect(M1, (-g["span"], g["vss_y"] - RAIL_W_NM, g["span"], g["vss_y"]))
    vdd_rail = _rect(M1, (-g["span"], g["vdd_y"], g["span"], g["vdd_y"] + RAIL_W_NM))
    polys += [gnd_rail, vdd_rail]
    tagged += [_lp_to_tagged(gnd_rail, "gnd"), _lp_to_tagged(vdd_rail, "vdd")]
    lanes = []     # vertical M1 drop lanes, for the signal planner to avoid
    for net, rail_y, down in (("gnd", g["vss_y"], True), ("vdd", g["vdd_y"], False)):
        for (label, tk) in NET_TERMINALS[net]:
            b = p.term[label][tk]
            x = _inner_x(b)
            if down:
                drop = _rect(M1, (x - _HMW, rail_y, x + _HMW, b[3]))
            else:
                drop = _rect(M1, (x - _HMW, b[1], x + _HMW, rail_y + RAIL_W_NM))
            polys.append(drop); tagged.append(_lp_to_tagged(drop, net))
            t = _lp_to_tagged(drop, net)[2]
            lanes.append((t[0], t[2], t[1], t[3], net))
    return polys, tagged, lanes


def route_net069(p: Placement):
    """Tail node: M5.s & M6.s meet on-axis, connected down to M7.d (all M1)."""
    polys: List[LayerPolygon] = []
    tagged = []
    m5s = p.term["M5"]["s"]; m7d = p.term["M7"]["d"]
    y0, y1 = m5s[1], m5s[3]
    # M5.s right edge -> axis (mirror brings M6.s -> meet)
    strip = _rect(M1, (m5s[2] - MW, y0, 0, y1))
    polys.append(strip); tagged.append(_lp_to_tagged(strip, "net069"))
    polys.append(_rect(M1, _mirror_box((m5s[2] - MW, y0, 0, y1))))
    tagged.append(("net069", M1, _mb((m5s[2] - MW, y0, 0, y1))))
    # axis riser down to the tail drain
    riser = _rect(M1, (-_HMW, m7d[3], _HMW, y1))
    polys.append(riser); tagged.append(_lp_to_tagged(riser, "net069"))
    return polys, tagged




def _overlap(a, b) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _m3_lanes(tagged):
    """Occupied vertical M3 lanes (x0, x1, y0, y1, net) from a tagged-shape list."""
    return [(b[0], b[2], b[1], b[3], net) for (net, lk, b) in tagged if lk == M3_KEY]


_SPACE = 600          # min edge-to-edge clearance between different-net routing


def _plan_tap(p: Placement, net, label, tk, track_y, occupied):
    """Pick a tap x within the terminal rail whose M3 riser lane (track_y..term)
    keeps ``_SPACE`` clearance from every different-net occupied lane (the union-
    find counts edge-TOUCHING as a short, and a via pad bridges any metal it
    touches, so a hard gap is required); register and return (x, y)."""
    b = p.term[label][tk]
    cy = (b[1] + b[3]) // 2
    y0, y1 = min(track_y, cy), max(track_y, cy)
    default = _inner_x(b) if tk == "s" else (b[0] + b[2]) // 2
    lo, hi = b[0] + _HMW, b[2] - _HMW
    cands = [default] + [default + d for d in range(150, 3001, 150)] \
                      + [default - d for d in range(150, 3001, 150)]
    msp = 350      # same-net min gap (>= M3.2a 0.28um) when two risers don't merge
    for x in cands:
        if x < lo or x > hi:
            continue
        probe = (x - _HMW - _SPACE, x + _HMW + _SPACE, y0 - _SPACE, y1 + _SPACE)
        # different nets: full _SPACE clearance. SAME net: either clear by the M3.2a
        # gap (msp) or fully overlap (the risers merge into one polygon) -- a parallel
        # same-net riser 0.22um away still violates M3.2a.
        sprobe = (x - _HMW - msp, x + _HMW + msp, y0 - msp, y1 + msp)
        bad = False
        for o in occupied:
            if o[4] != net:
                if _overlap(probe, o[:4]):
                    bad = True; break
            else:
                # same net: ok if fully clear (msp) OR x-overlapping enough to merge.
                merged = (o[0] <= x - _HMW and x + _HMW <= o[1]) or \
                         (x - _HMW <= o[0] and o[1] <= x + _HMW)
                if _overlap(sprobe, o[:4]) and not merged:
                    bad = True; break
        if bad:
            continue
        occupied.append((x - _HMW, x + _HMW, y0, y1, net))
        return (x, cy)
    occupied.append((default - _HMW, default + _HMW, y0, y1, net))
    return (default, cy)


# M2 horizontal trunk track (y) per signal net. Differential pairs SHARE a track
# (left/right halves separated by the axis -> no overlap); the cross-coupling pair
# rides offset tracks (it spans the axis and would otherwise overlap).
def _tracks(geom):
    """Signal trunk y-levels in the inter-row channel.

    The cross-coupling pair (crossn/crossp) is the dominant aggressor onto the
    pre-amp nodes (intern/interp). Both crossn and crossp M2 trunks span the whole
    width (the latch topology forces each regen node to reach both sides), so they
    run parallel to and over the intern/interp trunk and couple to it by overlap.
    Two parallel aggressor trunks on the SAME side of a victim trunk can never
    couple to it equally -- the nearer one wins -- which is what made
    crossn~intern (nearer) >> crossp~interp (farther). The only symmetric geometry
    is to put the intern/interp trunk MIDWAY BETWEEN the two cross trunks, so it is
    equidistant from crossn and crossp and the regen->pre-amp coupling mirrors.

    So intern/interp ride a centre track with crossn one ``cross_gap`` below and
    crossp one ``cross_gap`` above it (symmetric about the intern/interp level);
    outm/outp keep their low track near the NMOS drains and clk rides high near the
    PMOS gates. A tighter ``cross_gap`` shrinks the crossn/crossp y-separation (the
    last residual: the two trunks sit at slightly different heights, so the x-mirror
    is not a perfect y-mirror) while staying DRC-legal (400 nm metal, M2.2a min 280
    nm spacing). Differential pairs share a track."""
    lo = geom["channel_lo"] + 800
    tp = TRACK_PITCH_NM
    cross_gap = 900                       # metal 400 + 500 gap (DRC M2.2a min 280)
    outm_y = lo + tp                      # near the NMOS drains (unchanged)
    crossn_y = outm_y + cross_gap         # first cross trunk above outm
    inter_y = crossn_y + cross_gap        # intern/interp centred between the pair
    crossp_y = inter_y + cross_gap        # second cross trunk, equidistant
    # clk rides high near the PMOS gates, with a clear gap above the crossp trunk so
    # it never abuts intern/interp (the taller two-row channel makes the room).
    clk_y = max(crossp_y + tp, geom["channel_hi"] - 700)
    m3 = {"intern": inter_y, "interp": inter_y, "outm": outm_y, "outp": outm_y,
          "clk": clk_y}
    m4 = {"crossn": crossn_y, "crossp": crossp_y}
    return m3, m4


_DEGRADE_NET = "outm"          # the matched-pair net whose routing is desymmetrized


def _degrade_matched_pair(p: Placement, polys, tagged, m3_tracks):
    """Break the outm/outp routing mirror by adding a one-sided EXTRA M2 plate on
    the outm trunk's left, so outm carries metal area + coupling its untouched outp
    mirror does not. The plate's right edge overlaps the existing outm trunk, so it
    JOINS the outm net (no short, the net stays one component); it sits in the free
    channel to the LEFT of the outm trunk, kept ``_SPACE`` clear of every other-net
    M2 shape (chiefly the crossp trunk above and the span gnd feed column to the
    left) so the degraded cell is still DRC-legal and shorts-free.

    Effect: the outm/outp ``matched_cc_balance`` total-cap imbalance grows well past
    the 10% budget on real PEX, while outp's mirror is bit-for-bit unchanged. Only
    the loop's degrade->fix demonstration builds this; the committed default
    (``symmetric_matched_routing=True``) never desymmetrizes -- it is byte-identical."""
    track_y = m3_tracks[_DEGRADE_NET]
    # right anchor: the existing outm trunk left edge (the plate overlaps it to join).
    m2_left = min((b[0] for (n, lk, b) in tagged
                   if n == _DEGRADE_NET and lk == M2_KEY), default=None)
    if m2_left is None:
        return
    plate_right = m2_left + _HMW                 # overlap the trunk so the plate joins
    # Fixed target band, deliberately kept BELOW the crossp trunk (which rides above
    # the outm track) so the only neighbour the plate must clear on the left is the
    # span gnd feed M1 column (a different layer -> no short, just coupling). Top sits
    # well under the lowest higher-track M2; bottom sits at the channel floor.
    plate_top = track_y + 1500                   # clear of the crossp trunk above
    plate_bot = p.geom["channel_lo"]             # down to the NMOS-drain channel floor
    # Left edge: extend toward the rail span but keep _SPACE from any NON-outm M2 that
    # would overlap the plate's y-band (none today below crossp, but checked anyway).
    x_left_limit = -p.geom["span"] + _HMW
    for (n, lk, b) in tagged:
        if n == _DEGRADE_NET or lk != M2_KEY:
            continue
        if b[2] <= plate_right and not (b[3] + _SPACE <= plate_bot or
                                        b[1] - _SPACE >= plate_top):
            x_left_limit = max(x_left_limit, b[2] + _SPACE)
    plate_left = max(m2_left - 4400, x_left_limit)
    if plate_left >= plate_right or plate_bot >= plate_top:
        return                                   # no room -> leave geometry untouched
    plate = _rect(M2_KEY, (plate_left, plate_bot, plate_right, plate_top))
    polys.append(plate); tagged.append(_lp_to_tagged(plate, _DEGRADE_NET))


def route_signals(p: Placement, seed_occupied=None, *, symmetric_matched_routing=True):
    """Route every signal net with ONE global, collision-free tap planner: each
    terminal's M3 riser gets a clear x-lane (thin gates claim their column first;
    wide drains/sources then step aside within their rail). Horizontals are M2,
    verticals M3, so a trunk crossing another net's riser never shorts. ``seed_
    occupied`` (the supply M1 drop lanes) is avoided too, so a signal via never
    taps onto a vdd/gnd drop.

    ``symmetric_matched_routing`` (default True) keeps every differential pair's
    routing an EXACT x-mirror (the committed DRC-clean layout: outm/outp,
    intern/interp, crossn/crossp etc. are built left-then-mirror, so each pair's
    two nets carry matched parasitics). Setting it False introduces a DELIBERATE
    routing asymmetry into ONE matched pair (the outm trunk is lengthened on the
    left only), so that net picks up extra metal area + coupling its mirror partner
    does not -- the outm/outp ``matched_cc_balance`` imbalance then worsens
    measurably on a real PEX extraction (past the 10% budget). It is the degraded
    starting point of the layout-intent repair APPLY loop's symmetrize knob and is
    NEVER taken by the committed default; the symmetric path is byte-identical."""
    polys: List[LayerPolygon] = []
    tagged = []
    labels: Dict[str, Tuple[int, int]] = {}
    occupied: List = list(seed_occupied or [])
    # net0 / net1 input-gate ports first (registered so every riser avoids them).
    for net in ("net0", "net1"):
        (l, tk) = NET_TERMINALS[net][0]
        cx, cy = _tap(p, l, tk)
        for vp in via12_stack(Point(cx, cy)):
            polys.append(vp); t = _lp_to_tagged(vp, net); tagged.append(t)
            occupied.append((t[2][0], t[2][2], t[2][1], t[2][3], net))
        labels[net] = (cx, cy)
    m3_tracks, m4_tracks = _tracks(p.geom)
    # Pre-seed the cross-coupling lane columns (each crossn terminal tap x AND its
    # crossp mirror -x) over the cross-track y-band, so the M3 risers planned below
    # steer clear of where the crossn/crossp M4 risers + their via23 M3 pads will
    # land. Without this, moving intern/interp up between the cross tracks makes an
    # intern gate riser abut a crossp via23 pad (M3.2a). The cross-pad layer is M3
    # at the track end, so the seed must be in the same (M3) planning space.
    cn_track, cp_track = m4_tracks["crossn"], m4_tracks["crossp"]
    cy_lo = min(cn_track, cp_track) - 250
    cy_hi = max(cn_track, cp_track) + 250
    # crossn channel-bordering terminals (the TUCK cross-gate M17.g is escaped in a
    # dedicated pass below, so it is excluded here -- its mirror M16.g drops out of
    # the crossp mirror automatically, keeping the cross pair exact).
    cn_terms = [(l, tk) for (l, tk) in NET_TERMINALS["crossn"] if l not in _TUCK_LABELS]
    for (l, tk) in cn_terms:
        tx, _ty = _tap(p, l, tk)
        for x, who in ((tx, "crossn"), (-tx, "crossp")):
            occupied.append((x - _HMW, x + _HMW, cy_lo, cy_hi, who))
    # M3 nets: rail-constrained risers, planned narrow-gates-first. EXCLUDE the TUCK
    # signal terminals (M0.g/M22.g on intern/interp, M16.d/M17.d on outm/outp): their
    # long risers cannot escape within their rail, so they get a dedicated jogged-M4
    # pass (route_tuck_riser) instead -- the fix for the tuck-riser-over-trunk short.
    reqs = [(net, l, tk, track) for net, track in m3_tracks.items()
            for (l, tk) in NET_TERMINALS[net] if l not in _TUCK_LABELS]
    reqs.sort(key=lambda r: p.term[r[1]][r[2]][2] - p.term[r[1]][r[2]][0])
    taps: Dict[str, List[Tuple[int, int]]] = {}
    clk_axis_taps: List[Tuple[int, int]] = []
    for (net, l, tk, track) in reqs:
        tap = _plan_tap(p, net, l, tk, track, occupied)
        taps.setdefault(net, []).append(tap)
        # the tail gate (M7.g) is the on-axis clk terminal: its long riser must run
        # on the axis so clk couples equally into net0/net1 and crossn/crossp.
        if net == "clk" and l == "M7":
            clk_axis_taps.append(tap)
    trunk_x: Dict[str, Tuple[int, int]] = {}     # net -> (min, max) trunk M2 x-span
    for net in m3_tracks:
        xs = [c[0] for c in taps[net]]
        trunk_x[net] = (min(xs) - _HMW, max(xs) + _HMW)
        if net == "clk":
            pl, tg = route_clk(net, taps[net], m3_tracks[net],
                               axis_terminals=clk_axis_taps)
        else:
            pl, tg = route_comb(net, taps[net], m3_tracks[net])
        polys += pl; tagged += tg
    # cross-coupling on M4 with jogged risers. Plan crossn's lanes, then route crossp
    # as the EXACT x-MIRROR of crossn (its terminals are crossn's mirror) so the two
    # latch nodes stay parasitically balanced -- the only asymmetry is the small track
    # offset (crossn and crossp ride different tracks so they don't overlap at the axis).
    cn_taps = []
    for (l, tk) in cn_terms:
        tx, ty = _tap(p, l, tk)
        # mirror-aware: both the crossn riser and its crossp mirror must clear every
        # occupied lane (incl. the intern/interp M3 risers now between the tracks).
        lx = _plan_lane_mirror(ty, cn_track, cp_track, occupied, tx)
        cn_taps.append((tx, ty, lx))
    pl, tg = route_comb_m4("crossn", cn_taps, cn_track)
    polys += pl; tagged += tg
    cn_lanes = [lx for (_tx, _ty, lx) in cn_taps]
    trunk_x["crossn"] = (min(cn_lanes) - _HMW, max(cn_lanes) + _HMW)
    cp_taps = [(-tx, ty, -lx) for (tx, ty, lx) in cn_taps]      # exact mirror
    pl, tg = route_comb_m4("crossp", cp_taps, cp_track)
    polys += pl; tagged += tg
    trunk_x["crossp"] = (-trunk_x["crossn"][1], -trunk_x["crossn"][0])
    # Seed the tuck-lane planner with the ACTUAL routed vertical M3/M4 geometry (and
    # via pads) of every channel net, so a tuck riser clears the real risers by
    # >= _SPACE (the tap-lane registration alone misses where a wide riser/pad
    # finally landed -- the residual M3.2a between a tuck riser and its own net's
    # main riser). M2 trunks are horizontal/own-layer joins and are not seeded.
    for (net, lk, b) in tagged:
        if lk in (M3_KEY, M4_KEY, VIA2_KEY, VIA3_KEY) and (b[3] - b[1]) > (b[2] - b[0]):
            occupied.append((b[0], b[2], b[1], b[3], net))
    # --- dedicated TUCK-row signal escapes (the two short-modes solved) ----------
    # Each of the three mirror pairs of tuck signal terminals climbs on M4 to its
    # net's channel trunk in a CLEAR lane (mirror-aware), so a tuck riser never
    # overlaps the intern/interp M2 trunk and crossn/crossp stay matched. Routed
    # left-then-mirror so the differential pair is parasitically balanced.
    for lnet, rnet, lterm, rterm in (
        ("outm", "outp", ("M16", "d"), ("M17", "d")),
        ("crossp", "crossn", ("M16", "g"), ("M17", "g")),
        ("intern", "interp", ("M0", "g"), ("M22", "g")),
    ):
        ltrack = m3_tracks.get(lnet, m4_tracks.get(lnet))
        rtrack = m3_tracks.get(rnet, m4_tracks.get(rnet))
        tx, ty = _tap(p, lterm[0], lterm[1])
        lx = _plan_tuck_lane(lnet, rnet, ty, ltrack, rtrack, occupied, tx)
        # connect the riser's M2 landing to the existing trunk's NEAR endpoint.
        lx0, lx1 = trunk_x[lnet]
        l_reach = lx0 if abs(lx0 - lx) <= abs(lx1 - lx) else lx1
        pl, tg = route_tuck_riser(lnet, tx, ty, lx, ltrack, trunk_reach_x=l_reach)
        polys += pl; tagged += tg
        # exact x-mirror for the differential partner.
        pl, tg = route_tuck_riser(rnet, -tx, ty, -lx, rtrack, trunk_reach_x=-l_reach)
        polys += pl; tagged += tg
    labels.update({"intern": _ctr(p.term["M5"]["d"]), "interp": _ctr(p.term["M6"]["d"]),
                   "outm": _ctr(p.term["M16"]["d"]), "outp": _ctr(p.term["M17"]["d"]),
                   "clk": _ctr(p.term["M18"]["g"]), "crossn": _ctr(p.term["M4"]["d"]),
                   "crossp": _ctr(p.term["M3"]["d"])})
    if not symmetric_matched_routing:
        _degrade_matched_pair(p, polys, tagged, m3_tracks)
    return polys, tagged, labels


@dataclass
class RoutedComparator:
    polys: List[LayerPolygon]
    tagged: List
    labels: Dict[str, Tuple[int, int]]
    placement: Placement


def build_routed_comparator(
    *, shield: bool = True, symmetric_matched_routing: bool = True
) -> RoutedComparator:
    """Assemble the fully-routed 17-device comparator: placement + supply rails +
    net069 tail + all signal nets. Returns geometry, per-shape net tags (for the
    union-find short check) and net label points (for Magic naming).

    ``shield`` (default True) draws the grounded clk shield strip
    (``route_clk_shield``) that balances clk coupling into the regen pair. The
    default keeps the committed DRC-clean / 6-of-6-obligation-pass layout
    byte-for-byte. Setting ``shield=False`` OMITS the shield strip, producing a
    degraded layout whose clk_regen obligation fails -- the starting point of the
    layout-intent repair APPLY loop (``clay.repair.layout_apply``).

    ``symmetric_matched_routing`` (default True) keeps every differential pair's
    routing an EXACT x-mirror, so each matched pair (outm/outp, intern/interp,
    crossn/crossp, net0/net1) carries balanced parasitics -- the byte-identical
    committed layout. Setting it False introduces a DELIBERATE one-sided asymmetry
    into the outm/outp pair (it lengthens the outm trunk on the left only), so that
    pair's ``matched_cc_balance`` imbalance worsens past budget on real PEX. It is
    the degraded starting point of the repair loop's ``symmetric_matched_routing``
    knob (operator ``symmetrize_routing``); the default is NEVER desymmetrized.

    Both knobs default to the committed geometry, so ``build_routed_comparator()``,
    ``build_routed_comparator(shield=True)`` and
    ``build_routed_comparator(symmetric_matched_routing=True)`` are byte-identical."""
    p = place_devices()
    polys = list(p.polys)
    tagged = list(_device_terminal_tags(p))
    labels: Dict[str, Tuple[int, int]] = {}
    sp, tg, lanes = route_supply(p); polys += sp; tagged += tg
    if shield:
        shp, shtg = route_clk_shield(p); polys += shp; tagged += shtg
    n0, nt = route_net069(p); polys += n0; tagged += nt
    sg, stg, slab = route_signals(
        p, seed_occupied=lanes, symmetric_matched_routing=symmetric_matched_routing
    )
    polys += sg; tagged += stg; labels.update(slab)
    labels["gnd"] = (0, p.geom["vss_y"] - RAIL_W_NM // 2)
    labels["vdd"] = (0, p.geom["vdd_y"] + RAIL_W_NM // 2)
    labels["net069"] = (0, (p.term["M7"]["d"][1] + p.term["M5"]["s"][1]) // 2)
    return RoutedComparator(polys=polys, tagged=tagged, labels=labels, placement=p)


def emit_routed_comparator(cell: RoutedComparator, gds_path: str, *,
                           cell_name: str = "routed_comparator", dbu: float = 0.001) -> str:
    """Emit all routed polygons (every layer) + net text labels (on M1) via klayout."""
    import klayout.db as db
    ly = db.Layout(); ly.dbu = dbu
    top = ly.create_cell(cell_name)
    for lp in cell.polys:
        li = ly.layer(lp.layer, lp.datatype)
        top.shapes(li).insert(db.Polygon([db.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices]))
    lbl = ly.layer(M1_LABEL[0], M1_LABEL[1])
    for net, (x, y) in cell.labels.items():
        top.shapes(lbl).insert(db.Text(net, db.Trans(db.Vector(int(x), int(y)))))
    ly.write(str(gds_path))
    return str(gds_path)
