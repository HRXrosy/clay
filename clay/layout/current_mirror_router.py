"""Mirror-symmetric routed NMOS current mirror -- a THIRD real layout for the
layout-intent contract (the comparator was the first, the 5T-OTA the second).

Why this lane exists
====================
``clay.layout.intent.derive_layout_intent("current_mirror")`` can DERIVE a simple
current mirror's layout intent. Crucially that intent is the PLACEMENT-ONLY path:
a current mirror is a single-ended-output block, NOT a differential one, so it
declares ZERO ``matched_cc_balance`` node-cap budgets (``budgets == ()``) and the
mirror node pair ``(outp, outn)`` lands on ``mirror_pairs`` rather than
``diff_pairs``. Its real, checkable intent is therefore that the two mirror
DEVICES (the diode/reference leg and the mirrored-output leg) are placed
MIRROR-SYMMETRICALLY about the cell axis -- a device placement/sizing concern that
``clay.layout.placement_symmetry.mirror_placement_obligations`` consumes.

The comparator and the 5T-OTA both carried routed, Magic-extractable cells that
the parasitic side of the contract could check. This module adds a cell that
exercises the OTHER side of the contract -- the PLACEMENT-ONLY mirror path -- with
no cap budget at all, a deliberate vacuous parasitic pass whose substance is the
mirror device-placement symmetry.

Topology (textbook NMOS current mirror, 3 nets)
-----------------------------------------------
       outp (diode / reference node)          outn (mirrored output)
        |                                       |
    +---+---- gate bus (commoned to outp) ------+
    |                                           |
   M_ref  (diode-connected:                   M_out
    |      gate tied to its own drain)          |
    +------------------ gnd --------------------+
                        |
                       gnd (sources commoned to the gnd floor)

Net classes that map to ``derive_layout_intent("current_mirror")``:
  * ``outp`` -- the diode/reference leg: M_ref's drain, commoned to BOTH device
    gates (the diode tie that IS the current-mirror signature). The reference
    current is forced into this node.
  * ``outn`` -- the mirrored output leg: M_out's drain (high-impedance output).
  * ``gnd``  -- the common source node (both NMOS sources to the gnd floor) +
    substrate tie.

``(outp, outn)`` are the mirror NODES, asymmetric BY DESIGN (one diode-tied, one
high-impedance) -- so NO cap budget. The two DEVICES ``M_ref`` / ``M_out`` are
placed as exact x-mirrors about the axis so the mirror matches.

REUSE (built on the OTA / comparator proven building blocks)
------------------------------------------------------------
  * device bodies: ``coclean_5tota_v163.build_foldable_device`` (the same
    DRC-legal-by-construction foldable gf180 NMOS the OTA / comparator floorplan
    use) + ``build_substrate_tap`` for the gnd tie -- the OTA's exact device +
    substrate primitives.
  * mirror construction: ``backend._mirror_x`` / ``_translate`` (the OTA's
    exact-mirror placement primitive) -- the LEFT half (``M_ref``) is built then
    x-mirrored to the RIGHT half (``M_out``), so the placement is an exact image.
  * net tagging: the OTA's ``_lp_to_tagged`` + per-terminal-bbox tagging idiom, so
    ``placement_symmetry._device_net_map_from_cell`` recovers
    ``label -> {tk: net}`` straight off this cell's ``.tagged`` list (the cell is
    not in ``COMP_DEVICE_PLAN`` and exposes no ``_PAIR_LEFT`` annotation, exactly
    like the OTA) and ``_mirror_load_pairs_for`` picks ``(M_ref, M_out)`` by the
    commoned-gate diode-tie signature.
  * layer keys + the supply-rail / via12 idioms from the OTA router (M1, M1_LABEL,
    RAIL_W_NM, _rect, via12_stack), routing horizontals on M2 and verticals on M1
    so a foreign trunk crossing a riser never shorts.

Honesty / scope (CRITICAL)
--------------------------
This is a routed, Magic-EXTRACTABLE layout whose purpose is to run the PLACEMENT
side of the layout-intent contract on a THIRD cell. Every net is connected on real
gf180 layers and the two devices are exact x-mirrors. DRC-cleanliness is NOT
asserted by this module (a fresh router is not claimed DRC-clean without the real
``gf180mcu.drc`` deck's own verdict); the accompanying test reports the REAL
Magic-measured placement symmetry and states plainly that the cell is extractable,
not necessarily DRC-clean. No PEX or placement number is fabricated.
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
from .full_symmetric_comparator import _rect
from .tech_geometry import M2_KEY, via12_stack

M1 = GF180_METAL1_KEY
M1_LABEL = GF180_METAL1_LABEL_KEY

Bbox = Tuple[int, int, int, int]

# --- per-device sizing (a matched NMOS mirror pair; same size by construction) -
# Both legs identical (Wf=4U, L=1.0U, nf=4) -- a 1:1 mirror. The two devices are
# the matched pair the mirror methodology cares about.
_MIRROR_NF = 4
_MIRROR_WF_UM = 4.0
_MIRROR_L_UM = 1.0

# Floorplan constants (nm). The cell axis is x=0; the LEFT half (M_ref) is built
# then x-mirrored to the RIGHT half (M_out), so the two legs are exact images.
HALF_GAP_NM = 1800        # axis -> inner edge of each leg's device comp
RAIL_W_NM = 900           # gnd-rail metal1 width
MW = 400                  # routing metal width
_HMW = MW // 2


def _sh(b: Bbox, dx: int, dy: int) -> Bbox:
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def _mb(b: Bbox) -> Bbox:               # x-mirror a bbox about x=0
    return (-b[2], b[1], -b[0], b[3])


def _ctr(b: Bbox) -> Tuple[int, int]:
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)


def _inner_x(b: Bbox, margin: int = 350) -> int:
    """x near the axis-facing edge of a terminal rail. LEFT device -> right edge;
    RIGHT (mirror) -> left edge; on-axis -> centre."""
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
    """Mirror-symmetric current-mirror floorplan.

    Build the LEFT half (``M_ref``, the diode-connected reference leg) with its
    comp right edge at ``-HALF_GAP_NM`` and its comp bottom just above the gnd
    floor, then x-mirror it so the RIGHT half (``M_out``) is an exact image. Both
    NMOS sources face DOWN to the gnd floor; drains / gates face UP into the
    inter-device channel where the diode tie + the two outputs route.

    Records every device terminal (source/drain/gate) bbox under its label
    (``M_ref`` / ``M_out``)."""
    polys: List[LayerPolygon] = []
    term: Dict[str, Dict[str, Bbox]] = {}
    gnd_y = 0

    dev = build_foldable_device(
        kind="nmos", finger_count=_MIRROR_NF,
        gate_length_um=_MIRROR_L_UM, channel_width_um=_MIRROR_WF_UM,
    )

    # comp bottom sits a little above the gnd floor.
    row_y = gnd_y + 1100 - dev.comp_bbox[1]

    # LEFT half = M_ref; its x-mirror = M_out.
    dx = -HALF_GAP_NM - dev.comp_bbox[2]
    dy = row_y - dev.comp_bbox[1]
    left = _translate(list(dev.polygons), dx, dy)
    s = _sh(dev.source_rail_bbox, dx, dy)
    d = _sh(dev.drain_rail_bbox, dx, dy)
    g = _sh(dev.gate_contact_bbox, dx, dy)
    comp = _sh(dev.comp_bbox, dx, dy)
    polys.extend(left)
    polys.extend(_mirror_x(list(left), 0))
    term["M_ref"] = {"s": s, "d": d, "g": g, "comp": comp}
    term["M_out"] = {"s": _mb(s), "d": _mb(d), "g": _mb(g), "comp": _mb(comp)}

    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    span = max(abs(min(xs)), abs(max(xs))) + 1200
    chan_y = max(ys) + 1500           # routing channel above the device drains
    return Placement(
        polys=polys, term=term,
        geom=dict(gnd_y=gnd_y, span=span, chan_y=chan_y),
    )


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
    """gnd M1 floor rail + each NMOS source drop down to it + a P+ substrate tap.

    Both NMOS sources are commoned on ``gnd`` (the current-mirror common-source
    node). Built per-leg from the (mirror-symmetric) inner-edge of each source rail
    so the two drops are exact images, plus a substrate tap so the NMOS bulk
    extracts on gnd."""
    g = p.geom
    polys: List[LayerPolygon] = []
    tagged = []
    gnd_rail = _rect(M1, (-g["span"], g["gnd_y"] - RAIL_W_NM, g["span"], g["gnd_y"]))
    polys.append(gnd_rail); tagged.append(_lp_to_tagged(gnd_rail, "gnd"))

    for label in ("M_ref", "M_out"):
        s = p.term[label]["s"]
        x = _inner_x(s)
        drop = _rect(M1, (x - _HMW, g["gnd_y"] - RAIL_W_NM, x + _HMW, s[3]))
        polys.append(drop); tagged.append(_lp_to_tagged(drop, "gnd"))
        # bridge the whole source rail down to the gnd rail.
        bridge = _rect(M1, (s[0], g["gnd_y"], s[2], s[3]))
        polys.append(bridge); tagged.append(_lp_to_tagged(bridge, "gnd"))

    # P+ substrate tap below the gnd rail (NMOS bulk extracts on gnd).
    sub = build_substrate_tap((-600, g["gnd_y"] - RAIL_W_NM, 600, g["gnd_y"]))
    for lp in sub.polygons:
        polys.append(lp); tagged.append(_lp_to_tagged(lp, "gnd"))
    return polys, tagged


def _tap_up(net, cx, cy, track_y):
    """An M1 riser from a terminal at (cx, cy) to ``track_y`` + via12 stacks at
    both ends, joining an M1 vertical to an M2 trunk. A foreign trunk crossing
    this riser never shorts (different layers, no via at the crossing)."""
    polys: List[LayerPolygon] = []
    tagged = []
    riser = _vrect(M1, cx, cy, track_y)
    polys.append(riser); tagged.append(_lp_to_tagged(riser, net))
    for vp in via12_stack(Point(cx, track_y)) + via12_stack(Point(cx, cy)):
        polys.append(vp); tagged.append(_lp_to_tagged(vp, net))
    return polys, tagged


def route_signals(p: Placement):
    """Route the diode tie (gates commoned to outp = M_ref's own drain) + the two
    output ports.

    The current-mirror signature: BOTH gates are commoned and tied to the
    reference (diode) drain. So:
      * ``outp`` (diode/reference node) = M_ref.drain + M_ref.gate + M_out.gate,
        gathered on an M2 tie across both gate contacts and dropped to M_ref's
        drain on the LEFT side; this is the inherently axis-crossing net (the
        gate bus spans both legs), so it rides the channel track on M2 and taps
        each terminal on M1 (no metal1-metal1 short).
      * ``outn`` (mirrored output) = M_out.drain, a local via12 port (externally
        loaded, not trunk-routed across the axis).
    """
    polys: List[LayerPolygon] = []
    tagged = []
    labels: Dict[str, Tuple[int, int]] = {}
    chan = p.geom["chan_y"]

    def _term_ctr(label, tk):
        return _ctr(p.term[label][tk])

    # --- outp : diode tie (M_ref.d + M_ref.g + M_out.g), M2 trunk + M1 risers ---
    gl = _term_ctr("M_ref", "g")          # left gate contact
    gr = _term_ctr("M_out", "g")          # right gate contact (mirror)
    dl = _term_ctr("M_ref", "d")          # left drain (the diode/reference drain)
    for (cx, cy) in (gl, gr, dl):
        pl, tg = _tap_up("outp", cx, cy, chan)
        polys += pl; tagged += tg
    tie = _hrect(M2_KEY, gl[0] - _HMW, gr[0] + _HMW, chan)
    polys.append(tie); tagged.append(_lp_to_tagged(tie, "outp"))
    labels["outp"] = dl

    # --- outn : mirrored output port (M_out.drain), local via12 stack ----------
    dr = _term_ctr("M_out", "d")
    for vp in via12_stack(Point(dr[0], dr[1])):
        polys.append(vp); tagged.append(_lp_to_tagged(vp, "outn"))
    stub = _hrect(M2_KEY, dr[0] - 600, dr[0] + 600, dr[1])
    polys.append(stub); tagged.append(_lp_to_tagged(stub, "outn"))
    labels["outn"] = dr
    return polys, tagged, labels


def _terminal_tags(p: Placement):
    """Tag every device terminal rail (M1) with its schematic net.

    Net membership (textbook NMOS current mirror, mirror-symmetric):
      outp = M_ref.d + M_ref.g + M_out.g   (diode/reference node + commoned gates)
      outn = M_out.d                       (mirrored output)
      gnd  = M_ref.s + M_out.s (+ substrate)

    This is the authoritative per-terminal membership the placement checker reads
    off ``.tagged`` (by bbox identity): both gates carry ``outp`` (commoned, the
    diode-tie signature ``_mirror_load_pairs_for`` keys on), the two drains carry
    ``outp`` / ``outn`` (the mirror node pair), the two sources carry ``gnd``."""
    membership = {
        "outp": [("M_ref", "d"), ("M_ref", "g"), ("M_out", "g")],
        "outn": [("M_out", "d")],
        "gnd": [("M_ref", "s"), ("M_out", "s")],
    }
    tags = []
    for net, terms in membership.items():
        for (label, tk) in terms:
            tags.append((net, M1, p.term[label][tk]))
    return tags, membership


@dataclass
class RoutedCurrentMirror:
    polys: List[LayerPolygon]
    tagged: List
    labels: Dict[str, Tuple[int, int]]
    placement: Placement


def build_routed_current_mirror() -> RoutedCurrentMirror:
    """Assemble the fully-routed mirror-symmetric NMOS current mirror: placement +
    gnd rail / source drops + the diode tie + the two output ports, with labels for
    every net (``outp`` / ``outn`` / ``gnd``) so Magic extracts named, distinct
    nets that line up with ``derive_layout_intent("current_mirror")``.

    The two devices ``M_ref`` / ``M_out`` are placed as exact x-mirrors about x=0
    (``M_out == mirror_x(M_ref)``), so the mirror device-placement symmetry the
    contract checks holds by construction."""
    p = place_devices()
    polys = list(p.polys)
    term_tags, _membership = _terminal_tags(p)
    tagged = list(term_tags)
    labels: Dict[str, Tuple[int, int]] = {}

    sp, stg = route_supply(p); polys += sp; tagged += stg
    sg, sgt, slab = route_signals(p); polys += sg; tagged += sgt; labels.update(slab)

    labels["gnd"] = (0, p.geom["gnd_y"] - RAIL_W_NM // 2)
    return RoutedCurrentMirror(polys=polys, tagged=tagged, labels=labels, placement=p)


def emit_routed_current_mirror(cell: RoutedCurrentMirror, gds_path: str, *,
                               cell_name: str = "routed_current_mirror",
                               dbu: float = 0.001) -> str:
    """Emit all routed polygons (every layer) + net text labels (on M1 34/10) via
    klayout, exactly like ``ota_router.emit_routed_5t_ota``."""
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
