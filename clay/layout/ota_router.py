"""Mirror-symmetric routed 5T-OTA -- a SECOND real layout for the layout-intent
contract (the comparator was the first).

Why this lane exists
====================
``clay.layout.intent.derive_layout_intent("five_t_ota")`` can already DERIVE a
5T-OTA's net-level layout intent (input pair ``inp``/``inn`` offset-critical;
mirror-load drains ``outp``/``outn`` a matched differential output pair; the
input gates' kickback must balance) from the real ``build_5t_ota`` cell template.
But until now there was NO routed 5T-OTA layout to CHECK that intent against --
only the comparator carried a routed, Magic-extractable cell. This module closes
that: it produces a Magic-extractable GDS of a 5-transistor OTA whose nets line
up with the derived intent, routed with the SAME symmetry discipline the
comparator uses (the differential halves are placed and wired as exact x-mirrors
about the cell axis, so the matched pairs stay parasitically balanced).

Topology (textbook 5T-OTA, 7 nets)
----------------------------------
       vdd
        |
   +----+----+
   |         |
  M_load_p  M_load_n     PMOS current-mirror load (gates commoned; the outp
   | (diode) |           side is diode-connected -> sets the mirror reference)
  outp      outn         <- differential output pair (mirror-load drains ==
   |         |              input-pair drains)
  M_in_p   M_in_n        NMOS differential input pair (gates inp / inn)
   |         |
   +----tail-+
        |
      M_tail              NMOS tail current source (gate vbias, on-axis)
        |
       gnd

Net classes that map to ``derive_layout_intent("five_t_ota")``:
  * ``inp`` / ``inn``  -- input-pair gates, the offset-critical SENSITIVE
    victims AND a matched ``matched_cc_balance`` diff pair + the ``kickback`` pair.
  * ``outp`` / ``outn`` -- the input/load shared drains, the matched output diff
    pair (a ``matched_cc_balance`` budget). ``outp`` is the diode-connected
    reference side, ``outn`` is the high-impedance output.
  * ``tail`` -- the common-mode source node (input-pair sources meet on-axis at
    the tail drain). ``vbias`` -- the tail gate. ``vdd`` / ``gnd`` -- supplies.

REUSE (built on the comparator's proven building blocks)
--------------------------------------------------------
  * device bodies: ``coclean_5tota_v163.build_foldable_device`` (the same
    DRC-legal-by-construction foldable gf180 NMOS / PMOS the comparator floorplan
    and the co-clean 5T-OTA both use) + ``build_substrate_tap`` for the gnd tie;
  * mirror construction: ``backend._mirror_x`` / ``_translate`` (the comparator's
    exact-mirror placement primitive);
  * layer keys + the supply-rail / inner-edge-tap idioms from
    ``full_symmetric_comparator`` (M1, M1_LABEL, RAIL_W_NM, _rect, _mirror_box);
  * routing: M2 horizontal trunks + M1 risers (the comparator ``route_comb``
    layer-separation idea: horizontals on M2, verticals/taps on M1, joined by a
    via12 stack, so a foreign trunk crossing a riser never shorts), with the
    differential halves routed as exact x-mirrors.

Honesty / scope (CRITICAL)
--------------------------
This is a routed, Magic-EXTRACTABLE layout whose purpose is to run the real
layout-intent contract on a SECOND cell. Every net is connected on real gf180
layers and the differential halves are exact x-mirrors, so the matched-pair caps
balance. DRC-cleanliness is NOT asserted by this module (a fresh router is not
claimed DRC-clean without the real ``gf180mcu.drc`` deck's own verdict); the
accompanying test reports the REAL Magic-measured imbalance and states plainly
whether the cell is DRC-clean or merely extractable. No PEX number is fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .backend import LayerPolygon, Point, _mirror_x, _translate
from .coclean_5tota import (
    GF180_METAL1_KEY,
    GF180_METAL1_LABEL_KEY,
    build_foldable_device,
    build_substrate_tap,
)
from .full_symmetric_comparator import _rect, _mirror_box
from .tech_geometry import M2_KEY, via12_stack

M1 = GF180_METAL1_KEY
M1_LABEL = GF180_METAL1_LABEL_KEY

Bbox = Tuple[int, int, int, int]

# --- 5T-OTA per-device sizing (mirrors the R151 / R163 folded template) -------
# input pair  Wf=4U  L=0.5U  nf=4   (matched differential NMOS)
# load pair   Wf=4U  L=1.0U  nf=4   (matched PMOS current-mirror)
# tail        Wf=8U  L=1.0U  nf=4   (on-axis NMOS current source)
_INPUT_NF = 4
_INPUT_WF_UM = 4.0
_INPUT_L_UM = 0.5
_LOAD_NF = 4
_LOAD_WF_UM = 4.0
_LOAD_L_UM = 1.0
_TAIL_NF = 4
_TAIL_WF_UM = 8.0
_TAIL_L_UM = 1.0

# Floorplan constants (nm). The cell axis is x=0; the LEFT half is built then
# x-mirrored, so the two differential halves are exact images.
HALF_GAP_NM = 1800        # axis -> inner edge of each half's device comp
ROW_GAP_NM = 3200         # vertical gap between device rows (clears rails/stubs)
RAIL_W_NM = 900           # supply-rail metal1 width
MW = 400                  # routing metal width
_HMW = MW // 2
_SPACE = 400              # min different-net clearance for the trunk planner


def _sh(b: Bbox, dx: int, dy: int) -> Bbox:
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def _mb(b: Bbox) -> Bbox:               # x-mirror a bbox about x=0
    return (-b[2], b[1], -b[0], b[3])


def _ctr(b: Bbox) -> Tuple[int, int]:
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def _inner_x(b: Bbox, margin: int = 350) -> int:
    """x near the axis-facing edge of a terminal rail (clear of the centred
    drain / gate column). LEFT device -> right edge; RIGHT (mirror) -> left edge."""
    cx = (b[0] + b[2]) // 2
    if cx < -50:
        return b[2] - margin
    if cx > 50:
        return b[0] + margin
    return cx


@dataclass
class Placement:
    polys: List[LayerPolygon] = field(default_factory=list)
    term: Dict[str, Dict[str, Bbox]] = field(default_factory=dict)   # label -> {s,d,g}
    geom: Dict[str, int] = field(default_factory=dict)


def place_devices() -> Placement:
    """Mirror-symmetric 5T-OTA floorplan.

    Build the LEFT half (one input NMOS + one load PMOS, drains facing the
    inter-row channel) plus the on-axis tail, then x-mirror the half so the RIGHT
    half is an exact image. Stack bottom->top:

        gnd floor | on-axis tail M_tail | NMOS input row (sources -> tail,
        drains/gates up) | PMOS load row (vertically flipped: sources -> vdd,
        drains/gates down) | vdd ceiling.

    Records every device terminal (source/drain/gate) bbox under its label
    (``M_in_p`` / ``M_in_n`` / ``M_load_p`` / ``M_load_n`` / ``M_tail``)."""
    polys: List[LayerPolygon] = []
    term: Dict[str, Dict[str, Bbox]] = {}
    gnd_y = 0

    # Build the foldable device bodies once.
    in_dev = build_foldable_device(
        kind="nmos", finger_count=_INPUT_NF,
        gate_length_um=_INPUT_L_UM, channel_width_um=_INPUT_WF_UM,
    )
    load_dev = build_foldable_device(
        kind="pmos", finger_count=_LOAD_NF,
        gate_length_um=_LOAD_L_UM, channel_width_um=_LOAD_WF_UM,
    )
    tail_dev = build_foldable_device(
        kind="nmos", finger_count=_TAIL_NF,
        gate_length_um=_TAIL_L_UM, channel_width_um=_TAIL_WF_UM,
    )

    def _band_h(dev) -> int:
        boxes = [dev.source_rail_bbox, dev.drain_rail_bbox,
                 dev.gate_contact_bbox, dev.comp_bbox]
        if dev.nwell_bbox is not None:
            boxes.append(dev.nwell_bbox)
        lo = min(b[1] for b in boxes)
        hi = max(b[3] for b in boxes)
        return hi - lo, lo

    # --- y-band plan -----------------------------------------------------------
    tail_h, tail_lo = _band_h(tail_dev)
    tail_base = gnd_y + 1100 - tail_lo            # tail comp band just above floor
    in_h, in_lo = _band_h(in_dev)
    in_base = tail_base + tail_h + ROW_GAP_NM - in_lo
    load_h, load_lo = _band_h(load_dev)
    # the PMOS load row is flipped (mirror-y), so reserve its band above the
    # NMOS input row with the channel gap between the two drain rails.
    load_base = in_base + in_h + ROW_GAP_NM

    def place_left(dev, label, mirror_label, row_y, *, flip=False,
                   source_to_axis=False):
        """Place ``dev`` as the LEFT half device with its comp bottom at ``row_y``
        and its comp right edge at -HALF_GAP_NM; record both the left terminal
        bboxes (``label``) and the x-mirror (``mirror_label``)."""
        dx = -HALF_GAP_NM - dev.comp_bbox[2]
        dy = row_y - dev.comp_bbox[1]
        left = _translate(list(dev.polygons), dx, dy)
        s = _sh(dev.source_rail_bbox, dx, dy)
        d = _sh(dev.drain_rail_bbox, dx, dy)
        g = _sh(dev.gate_contact_bbox, dx, dy)
        comp = _sh(dev.comp_bbox, dx, dy)
        if flip:
            # PMOS: mirror about the comp y-centre so sources face vdd (up) and
            # drains/gates face DOWN into the channel.
            ay = (comp[1] + comp[3]) // 2
            left = _mirror_y(left, ay)
            s, d, g = _mby(s, ay), _mby(d, ay), _mby(g, ay)
        polys.extend(left)
        polys.extend(_mirror_x(list(left), 0))
        term[label] = {"s": s, "d": d, "g": g, "comp": comp}
        term[mirror_label] = {"s": _mb(s), "d": _mb(d), "g": _mb(g), "comp": _mb(comp)}

    # NMOS input pair: left = M_in_p (gate inp), right (mirror) = M_in_n (gate inn).
    place_left(in_dev, "M_in_p", "M_in_n", in_base)
    # PMOS load pair: left = M_load_p (drain outp), right (mirror) = M_load_n.
    place_left(load_dev, "M_load_p", "M_load_n", load_base, flip=True)

    # On-axis tail (centred at x=0).
    tcx = (tail_dev.comp_bbox[0] + tail_dev.comp_bbox[2]) // 2
    tdy = tail_base - tail_dev.comp_bbox[1]
    polys.extend(_translate(list(tail_dev.polygons), -tcx, tdy))
    term["M_tail"] = {
        "s": _sh(tail_dev.source_rail_bbox, -tcx, tdy),
        "d": _sh(tail_dev.drain_rail_bbox, -tcx, tdy),
        "g": _sh(tail_dev.gate_contact_bbox, -tcx, tdy),
        "comp": _sh(tail_dev.comp_bbox, -tcx, tdy),
    }

    # Merge the two per-PMOS Nwells into ONE continuous well (standard; removes the
    # NW.2b inter-well different-potential spacing rule between the two load PMOS).
    nw = [lp for lp in polys if (lp.layer, lp.datatype) == (21, 0)]
    if nw:
        wx0 = min(v.x for lp in nw for v in lp.polygon.vertices)
        wy0 = min(v.y for lp in nw for v in lp.polygon.vertices)
        wx1 = max(v.x for lp in nw for v in lp.polygon.vertices)
        wy1 = max(v.y for lp in nw for v in lp.polygon.vertices)
        polys.append(_rect((21, 0), (wx0, wy0, wx1, wy1)))

    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    span = max(abs(min(xs)), abs(max(xs))) + 1200
    vdd_y = max(ys) + 1500
    # inter-row signal channel: top of input drains .. bottom of (flipped) load drains
    channel_lo = max(term["M_in_p"]["d"][3], term["M_in_n"]["d"][3])
    channel_hi = min(term["M_load_p"]["d"][1], term["M_load_n"]["d"][1])
    return Placement(polys=polys, term=term,
                     geom=dict(gnd_y=gnd_y, tail_base=tail_base, in_base=in_base,
                               load_base=load_base, vdd_y=vdd_y, span=span,
                               channel_lo=channel_lo, channel_hi=channel_hi))


def _mirror_y(polys: List[LayerPolygon], axis_y: int) -> List[LayerPolygon]:
    from .backend import Polygon
    out = []
    for lp in polys:
        verts = list(reversed([Point(v.x, 2 * axis_y - v.y) for v in lp.polygon.vertices]))
        out.append(LayerPolygon(layer=lp.layer, datatype=lp.datatype,
                                polygon=Polygon(vertices=verts)))
    return out


def _mby(b: Bbox, axis_y: int) -> Bbox:
    return (b[0], 2 * axis_y - b[3], b[2], 2 * axis_y - b[1])


# === routing ==================================================================

def _lp_to_tagged(lp: LayerPolygon, net: str):
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (net, (lp.layer, lp.datatype), (min(xs), min(ys), max(xs), max(ys)))


def _vrect(layer, cx, y0, y1) -> LayerPolygon:
    return _rect(layer, (cx - _HMW, min(y0, y1), cx + _HMW, max(y0, y1)))


def _hrect(layer, x0, x1, cy) -> LayerPolygon:
    return _rect(layer, (min(x0, x1), cy - _HMW, max(x0, x1), cy + _HMW))


def route_supply(p: Placement):
    """gnd / vdd M1 rails + per-source M1 drop to the rail.

    NMOS sources (input pair drains current FROM the tail node, but the input
    sources go to ``tail``, not gnd) -- so the gnd rail serves the TAIL source +
    the substrate tap. The PMOS load sources drop UP to the vdd rail."""
    g = p.geom
    polys: List[LayerPolygon] = []
    tagged = []
    gnd_rail = _rect(M1, (-g["span"], g["gnd_y"] - RAIL_W_NM, g["span"], g["gnd_y"]))
    vdd_rail = _rect(M1, (-g["span"], g["vdd_y"], g["span"], g["vdd_y"] + RAIL_W_NM))
    polys += [gnd_rail, vdd_rail]
    tagged += [_lp_to_tagged(gnd_rail, "gnd"), _lp_to_tagged(vdd_rail, "vdd")]

    # tail source -> gnd rail (on-axis drop).
    ts = p.term["M_tail"]["s"]
    tcx = (ts[0] + ts[2]) // 2
    drop = _rect(M1, (tcx - _HMW, g["gnd_y"] - RAIL_W_NM, tcx + _HMW, ts[3]))
    polys.append(drop); tagged.append(_lp_to_tagged(drop, "gnd"))
    # widen the on-axis tie so the whole tail source rail reaches the gnd drop.
    tie = _rect(M1, (ts[0], g["gnd_y"], ts[2], ts[3]))
    polys.append(tie); tagged.append(_lp_to_tagged(tie, "gnd"))

    # P+ substrate tap below the gnd rail (ties gnd to the global substrate so the
    # NMOS bulk extracts on gnd, like the co-clean 5T-OTA).
    sub = build_substrate_tap((-600, g["gnd_y"] - RAIL_W_NM, 600, g["gnd_y"]))
    for lp in sub.polygons:
        polys.append(lp); tagged.append(_lp_to_tagged(lp, "gnd"))

    # PMOS load sources -> vdd rail (each half, mirror-symmetric inner-edge drop).
    for label in ("M_load_p", "M_load_n"):
        s = p.term[label]["s"]
        x = _inner_x(s)
        drop = _rect(M1, (x - _HMW, s[1], x + _HMW, g["vdd_y"] + RAIL_W_NM))
        polys.append(drop); tagged.append(_lp_to_tagged(drop, "vdd"))
        # also bridge the full source rail up so the whole source reaches vdd.
        bridge = _rect(M1, (s[0], s[1], s[2], g["vdd_y"]))
        polys.append(bridge); tagged.append(_lp_to_tagged(bridge, "vdd"))
    return polys, tagged


def route_tail_node(p: Placement):
    """Tail node: the two input-pair SOURCES meet on-axis and tie down to the
    tail DRAIN (all M1). Built left + exact mirror so the node is x-symmetric."""
    polys: List[LayerPolygon] = []
    tagged = []
    s_in = p.term["M_in_p"]["s"]          # left input source
    td = p.term["M_tail"]["d"]            # on-axis tail drain
    y0, y1 = s_in[1], s_in[3]
    # left input source inner edge -> axis (mirror brings the right source to meet)
    strip = (s_in[2] - MW, y0, 0, y1)
    polys.append(_rect(M1, strip)); tagged.append(_lp_to_tagged(_rect(M1, strip), "tail"))
    mstrip = _mirror_box(strip)
    polys.append(_rect(M1, mstrip)); tagged.append(("tail", M1, mstrip))
    # axis riser from the meet band down to the tail drain
    riser = (-_HMW, td[3], _HMW, y1) if td[3] < y1 else (-_HMW, y0, _HMW, td[1])
    polys.append(_rect(M1, riser)); tagged.append(_lp_to_tagged(_rect(M1, riser), "tail"))
    # widen across the tail drain rail so it joins fully
    join = _rect(M1, (td[0], min(td[1], y1), td[2], max(td[3], y0)))
    polys.append(join); tagged.append(_lp_to_tagged(join, "tail"))
    return polys, tagged


def _tap_up(net, cx, cy, track_y):
    """An M1 riser from a terminal at (cx, cy) up/down to ``track_y`` + a via12
    stack at both ends, returning the polygons + a via-foot M2 pad at the track.
    Verticals are M1, the trunk is M2: a foreign trunk crossing this riser never
    shorts (different layers, no via at the crossing)."""
    polys: List[LayerPolygon] = []
    tagged = []
    riser = _vrect(M1, cx, cy, track_y)
    polys.append(riser); tagged.append(_lp_to_tagged(riser, net))
    for vp in via12_stack(Point(cx, track_y)) + via12_stack(Point(cx, cy)):
        polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
    return polys, tagged


def route_comb(net: str, centers: List[Tuple[int, int]], track_y: int):
    """2-layer Manhattan comb: an M2 horizontal trunk at ``track_y`` + an M1
    vertical riser to each terminal, joined by via12 stacks. Mirrors the
    comparator ``route_comb`` layer-separation discipline (horizontals M2,
    verticals M1) so a different net's trunk over this net's riser never shorts."""
    polys: List[LayerPolygon] = []
    tagged: List = []
    xs = [c[0] for c in centers]
    trunk = _hrect(M2_KEY, min(xs) - _HMW, max(xs) + _HMW, track_y)
    polys.append(trunk); tagged.append(_lp_to_tagged(trunk, net))
    for (cx, cy) in centers:
        pl, tg = _tap_up(net, cx, cy, track_y)
        polys += pl; tagged += tg
    return polys, tagged


def _terminal_tags(p: Placement):
    """Tag every device terminal rail (M1) with its schematic net.

    Net membership (textbook 5T-OTA, mirror-symmetric):
      inp  = M_in_p.g                 (left input gate)
      inn  = M_in_n.g                 (right input gate, mirror)
      outp = M_in_p.d + M_load_p.d + M_load_p.g + M_load_n.g  (diode ref side)
      outn = M_in_n.d + M_load_n.d    (high-impedance output side)
      tail = M_in_p.s + M_in_n.s + M_tail.d
      vbias= M_tail.g
      vdd  = M_load_p.s + M_load_n.s
      gnd  = M_tail.s (+ substrate)"""
    membership = {
        "inp": [("M_in_p", "g")],
        "inn": [("M_in_n", "g")],
        "outp": [("M_in_p", "d"), ("M_load_p", "d"), ("M_load_p", "g"), ("M_load_n", "g")],
        "outn": [("M_in_n", "d"), ("M_load_n", "d")],
        "vbias": [("M_tail", "g")],
    }
    tags = []
    for net, terms in membership.items():
        for (label, tk) in terms:
            tags.append((net, M1, p.term[label][tk]))
    return tags, membership


def _tracks(geom):
    """M2 trunk y-levels in the inter-row channel.

    outp/outn share ONE low track: each is a half-width comb confined to its own
    side of the axis (outp left, outn right), so the two trunks never reach the
    axis and never overlap -- the comparator's "differential pairs share a track"
    discipline. The current-mirror diode tie (the two load gates commoned to the
    outp/diode node) is the one inherently axis-crossing net in a mirror-load OTA;
    it rides its OWN higher track and drops to outp on the LEFT side only (clear of
    outn on the right). vbias (tail gate) rides the top track."""
    lo = geom["channel_lo"] + 700
    out_y = lo
    diode_y = lo + 900
    bias_y = lo + 1800
    return {"out": out_y, "diode": diode_y, "bias": bias_y}


def route_signals(p: Placement):
    """Route outp / outn (matched mirror combs) + the current-mirror diode tie +
    vbias. inp / inn are input-gate PORTS (a via12 stack at the gate), labelled
    but not trunk-routed (they are externally driven)."""
    polys: List[LayerPolygon] = []
    tagged = []
    labels: Dict[str, Tuple[int, int]] = {}
    tracks = _tracks(p.geom)

    def _term_ctr(label, tk):
        return _ctr(p.term[label][tk])

    # --- outp / outn : EXACT mirror combs, each confined to its own side ------
    # outn (right): input-n drain + load-n drain. outp (left): the x-mirror
    # (input-p drain + load-p drain). Each comb stays on its side of the axis, so
    # the two M2 trunks on the shared `out` track never reach x=0 -> no overlap.
    outn_centers = [_term_ctr("M_in_n", "d"), _term_ctr("M_load_n", "d")]
    pl, tg = route_comb("outn", outn_centers, tracks["out"])
    polys += pl; tagged += tg
    labels["outn"] = _term_ctr("M_in_n", "d")

    outp_centers = [_term_ctr("M_in_p", "d"), _term_ctr("M_load_p", "d")]
    pl, tg = route_comb("outp", outp_centers, tracks["out"])
    polys += pl; tagged += tg
    labels["outp"] = _term_ctr("M_in_p", "d")

    # --- current-mirror diode tie: load_p.g == load_n.g == outp (diode node) --
    # The one inherently axis-crossing net of a mirror-load OTA. It rides its OWN
    # higher `diode` track (M2) gathering both load-gate contacts, and drops to
    # the outp drain on the LEFT side only (clear of outn on the right). This net
    # is PART of outp electrically, so it is tagged "outp".
    glx = _term_ctr("M_load_p", "g")          # left load gate (-8060, ~)
    grx = _term_ctr("M_load_n", "g")          # right load gate (+8060, ~)
    dy = tracks["diode"]
    oy = tracks["out"]
    # M2 tie across both load gates on the diode track + an M1 riser tapping each
    # gate's poly contact up to it.
    for (gx, gy) in (glx, grx):
        pl, tg = _tap_up("outp", gx, gy, dy)
        polys += pl; tagged += tg
    tie = _hrect(M2_KEY, glx[0] - _HMW, grx[0] + _HMW, dy)
    polys.append(tie); tagged.append(_lp_to_tagged(tie, "outp"))
    # Bridge the diode track (dy) down to the out track (oy) at the LEFT load-drain
    # column (clear of outn on the right), so the diode tie and the out-track outp
    # trunk are ONE net (outp). An M1 riser between the two tracks + via12 at each.
    lpx = _term_ctr("M_load_p", "d")
    bridge = _vrect(M1, lpx[0], oy, dy)
    polys.append(bridge); tagged.append(_lp_to_tagged(bridge, "outp"))
    for vp in via12_stack(Point(lpx[0], dy)) + via12_stack(Point(lpx[0], oy)):
        polys.append(vp); tagged.append(_lp_to_tagged(vp, "outp"))

    # --- vbias : tail gate PORT (externally driven, like inp/inn) -------------
    # A via12 stack + a short M2 stub local to the tail gate contact, so vbias
    # extracts as its own named net without a long channel-crossing riser.
    vb = _ctr(p.term["M_tail"]["g"])
    for vp in via12_stack(Point(vb[0], vb[1])):
        polys.append(vp); tagged.append(_lp_to_tagged(vp, "vbias"))
    stub = _hrect(M2_KEY, vb[0] - 600, vb[0] + 600, vb[1])
    polys.append(stub); tagged.append(_lp_to_tagged(stub, "vbias"))
    labels["vbias"] = vb

    # --- inp / inn : input gate ports (via12 stack so the label lands on a
    # connected net; the gates are externally driven, not trunk-routed) --------
    for net, label in (("inp", "M_in_p"), ("inn", "M_in_n")):
        g = _ctr(p.term[label]["g"])
        for vp in via12_stack(Point(g[0], g[1])):
            polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
        labels[net] = g
    return polys, tagged, labels


@dataclass
class RoutedOta:
    polys: List[LayerPolygon]
    tagged: List
    labels: Dict[str, Tuple[int, int]]
    placement: Placement


def build_routed_5t_ota() -> RoutedOta:
    """Assemble the fully-routed mirror-symmetric 5T-OTA: placement + supply
    rails + tail node + the output/bias signal nets, with labels for every net
    (``inp`` / ``inn`` / ``outp`` / ``outn`` / ``tail`` / ``vbias`` / ``vdd`` /
    ``gnd``) so Magic extracts named, distinct nets that line up with
    ``derive_layout_intent("five_t_ota")``."""
    p = place_devices()
    polys = list(p.polys)
    term_tags, _membership = _terminal_tags(p)
    tagged = list(term_tags)
    labels: Dict[str, Tuple[int, int]] = {}

    sp, stg = route_supply(p); polys += sp; tagged += stg
    tn, ttg = route_tail_node(p); polys += tn; tagged += ttg
    sg, sgt, slab = route_signals(p); polys += sg; tagged += sgt; labels.update(slab)

    labels["gnd"] = (0, p.geom["gnd_y"] - RAIL_W_NM // 2)
    labels["vdd"] = (0, p.geom["vdd_y"] + RAIL_W_NM // 2)
    labels["tail"] = (0, (p.term["M_tail"]["d"][1] + p.term["M_in_p"]["s"][1]) // 2)
    return RoutedOta(polys=polys, tagged=tagged, labels=labels, placement=p)


def emit_routed_5t_ota(cell: RoutedOta, gds_path: str, *,
                       cell_name: str = "routed_5t_ota", dbu: float = 0.001) -> str:
    """Emit all routed polygons (every layer) + net text labels (on M1 34/10) via
    klayout, exactly like ``comparator_router.emit_routed_comparator``."""
    import klayout.db as db
    ly = db.Layout(); ly.dbu = dbu
    top = ly.create_cell(cell_name)
    for lp in cell.polys:
        li = ly.layer(lp.layer, lp.datatype)
        top.shapes(li).insert(
            db.Polygon([db.Point(int(v.x), int(v.y)) for v in lp.polygon.vertices])
        )
    lbl = ly.layer(M1_LABEL[0], M1_LABEL[1])
    for net, (x, y) in cell.labels.items():
        top.shapes(lbl).insert(db.Text(net, db.Trans(db.Vector(int(x), int(y)))))
    ly.write(str(gds_path))
    return str(gds_path)
