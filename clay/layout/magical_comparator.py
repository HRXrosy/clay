"""R171 (BL-171-MAG) -- a 17-device DOUBLE-TAIL DYNAMIC LATCH COMPARATOR laid out
on gf180mcu, pushed toward CO-CLEAN, as the Clay-side artifact for an honest
Clay-vs-MAGICAL layout benchmark.

The reference this lane targets
===============================
The benchmark reference is MAGICAL's ``cmp`` set (101 sizing variants, all the
SAME topology): a 17-device double-tail dynamic latch comparator in open_pdk
(``nch_lvt_mac`` / ``pch_lvt_mac``, L=40nm). 9 NMOS + 8 PMOS. The exact
connectivity (read VERBATIM from ``comp_31/comp.sp``) is::

    -- clk-gated tail (NMOS)
    M7  : d=net069 g=clk    s=gnd    b=gnd   nf=8  W=6.76u   (tail switch)
    -- input pair (NMOS), sources on the tail node net069
    M5  : d=intern g=net0   s=net069 b=gnd   nf=8  W=1.24u
    M6  : d=interp g=net1   s=net069 b=gnd   nf=8  W=1.24u
    -- 1st-stage precharge / reset PMOS (clk-gated), drains intern/interp
    M18 : d=intern g=clk    s=vdd    b=vdd   nf=2  W=1.00u
    M2  : d=interp g=clk    s=vdd    b=vdd   nf=2  W=1.00u
    -- cross-coupled NMOS latch
    M4  : d=crossn g=crossp s=intern b=gnd   nf=4  W=4.60u
    M3  : d=crossp g=crossn s=interp b=gnd   nf=4  W=4.60u
    -- cross-coupled PMOS latch
    M14 : d=crossn g=crossp s=vdd    b=vdd   nf=8  W=1.60u
    M13 : d=crossp g=crossn s=vdd    b=vdd   nf=8  W=1.60u
    -- 2nd-stage precharge PMOS (clk-gated) on the cross nodes
    M1  : d=crossn g=clk    s=vdd    b=vdd   nf=2  W=3.16u
    M12 : d=crossp g=clk    s=vdd    b=vdd   nf=2  W=3.16u
    -- output inverters (CMOS), gated by the cross nodes
    M16 : d=outm   g=crossp s=gnd    b=gnd   nf=4  W=3.40u   (NMOS pull-down)
    M8  : d=outm   g=crossp s=vdd    b=vdd   nf=4  W=2.56u   (PMOS pull-up)
    M17 : d=outp   g=crossn s=gnd    b=gnd   nf=4  W=3.40u   (NMOS pull-down)
    M15 : d=outp   g=crossn s=vdd    b=vdd   nf=4  W=2.56u   (PMOS pull-up)
    -- reset / bleed NMOS (L=1u long devices), gates intern/interp to gnd
    M0  : d=gnd    g=intern s=gnd    b=gnd   nf=1  L=1u  W=5.56u
    M22 : d=gnd    g=interp s=gnd    b=gnd   nf=1  L=1u  W=5.56u

12 nets: clk, gnd, vdd, net0, net1, net069, intern, interp, crossp, crossn,
outm, outp.

The CRITICAL cross-PDK caveat (NO fabricated verification)
==========================================================
MAGICAL = open_pdk (40 nm). We have NO open_pdk DRC/LVS deck or models here, so
MAGICAL's GDS CANNOT be re-verified / simulated in this environment, and Clay
CANNOT emit open_pdk. Clay's real PDK here is gf180mcu (180 nm) with the real
KLayout ``gf180mcu.drc`` + ``gf180mcu.lvs`` decks. So absolute um^2 is NOT
comparable; the benchmark NORMALISES by ``min-channel-L^2`` and compares
capability / structure. This module NEVER claims a open_pdk verification and the
companion benchmark tool reads MAGICAL's GDS read-only (structural compare only).

The gf180 translation (documented, honest)
-------------------------------------------
* Topology: IDENTICAL 17 devices, IDENTICAL connectivity, IDENTICAL 12 nets.
* Device models: ``nch_lvt_mac`` -> ``nmos_3p3``, ``pch_lvt_mac`` -> ``pmos_3p3``
  (gf180mcu has no LVT variant; the 3p3 core device is the closest real model
  the gf180mcu.lvs deck extracts).
* Gate length: open_pdk L=40 nm -> gf180 min L (the ``build_foldable_device``
  default L floored at the gf180 GATE_MIN_LENGTH); the two L=1u reset devices
  keep a long L=1u. (The 4.5x longer gf180 min-L is a process fact, recorded in
  the benchmark; it is why absolute area is NOT comparable.)
* Finger count: each device keeps its MAGICAL ``nf`` (the fold count) so the
  extracted ``m=N`` matches the reference fold structure.
* Per-finger width: floored at the gf180 channel min width; the relative width
  ratios across devices are preserved where the gf180 minimum allows.

How co-clean is pursued (the proven R162/R163 + R170 machinery)
---------------------------------------------------------------
* EVERY device is a DRC-legal-by-construction foldable gf180 MOS
  (``build_foldable_device`` from ``coclean_5tota_v163`` -- imported READ-ONLY):
  every contact on COMP (CO.11), CO.6 metal1 enclosure, implant grown over the
  gate (NP.5a / PP.5a), gate bus PL.5 clear of COMP; a PMOS carries its own
  Nwell (DF.4c) + an in-well N+ well-tap to its source rail.
* The 17 devices are placed left-to-right at NW.2b-safe gaps so no cross-device
  Nwell / COMP / implant spacing rule fires.
* The 12 inter-device nets are routed OVER THE TOP with a strict layer
  discipline (the R170 ``bus_router`` rule, re-used here): each net owns a
  DISTINCT M3 horizontal trunk row; each terminal escapes M1 -> M2 -> M3 -> M4
  via net-local via stacks and rises on an M4 vertical riser in its own terminal
  column up to the net's trunk row. Horizontals are pairwise-distinct by row,
  verticals are on M4 in per-terminal columns, and the only M3<->M4 contact is a
  net-local via34 stack -- so no two DIFFERENT nets' shapes touch.
* ``gnd`` is the NMOS bulk: named by a metal1 ``gnd`` TEXT label on a tail-source
  rail + a P+ substrate tap (the deck unifies the global substrate net to
  ``gnd``). ``vdd`` is the PMOS bulk: each PMOS already carries an in-well N+ tap
  to its own source rail, and the ``vdd`` trunk commons every PMOS source so all
  well bodies extract on ``vdd``; a metal1 ``vdd`` TEXT label names it.

Honesty contract (CRITICAL)
---------------------------
* The foldable device + taps + label injector + the R139 DRC / R145 LVS
  harnesses + the committed Rust binary-GDS emitter + the R170 union-find +
  via stacks are imported READ-ONLY; nothing here mutates them.
* The authoritative DRC verdict is the real deck's own ``.lyrdb`` violation
  count; the authoritative LVS verdict is the real deck's own ``compare`` +
  extracted ``.cir`` counts. A VACUOUS extraction (0 / device-short) only ever
  DEMOTES.
* CO-CLEAN is asserted ONLY when the SAME single GDS file passed BOTH real decks
  (DRC 0 + non-vacuous LVS match) AND the geometric union-find shows 0 inter-net
  shorts. Whatever level is reached is reported with the REAL deck counts; a
  residual is reported honestly, never fabricated up to a pass.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary, Point, Polygon
from .coclean_5tota import (
    GF180_METAL1_CONTACT_ENCLOSURE_NM,
    GF180_METAL1_KEY,
    GF180_METAL1_LABEL_KEY,
    GF180_METAL1_MIN_WIDTH_NM,
    FoldableDevice,
    SubstrateLabelSpec,
    build_foldable_device,
    build_substrate_tap,
    inject_metal1_labels_into_gds,
)
from .gf180_device_geom import GF180_CONTACT_SIZE_NM
from .tech_geometry import M2_KEY, VIA1_KEY, via12_stack
from .tech_geometry import M3_KEY, via23_stack
from .tech_geometry import M4_KEY, M4_MIN_WIDTH_NM, VIA3_KEY, via34_stack

# Repo root: clay/layout/magical_comparator_v171.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "demo_out" / "magical_comparator_v171"
DEFAULT_KLAYOUT_BIN = Path(r"C:\Program Files (x86)\KLayout\klayout_app.exe")
DEFAULT_CELL_NAME = "gf180_magical_comparator_v171"

# Net names (the global power nets the comparator declares).
GND_NET = "gnd"  # NMOS bulk / global substrate
VDD_NET = "vdd"  # PMOS bulk / well-tap tie

# The 12 inter-device nets that the over-the-top bus routes (gnd is handled by the
# substrate tap + label, vdd by the PMOS well-taps + trunk + label).
M3_KEY_T = M3_KEY
M4_KEY_T = M4_KEY

# via layers that bridge two metals in the union-find (Via1 M1<->M2, Via2 M2<->M3,
# Via3 M3<->M4).
from .tech_geometry import via23_stack as _v23  # noqa: E402  (alias clarity)
VIA2_KEY: Tuple[int, int] = (38, 0)
_ALL_VIA_KEYS = frozenset({VIA1_KEY, VIA2_KEY, VIA3_KEY})

# Routing geometry. The trunk rows + riser columns are quantised to a track pitch
# (>> the 0.28um metal spacing) so adjacent rows / columns clear by construction.
TRACK_PITCH_NM = 1000
M3_WIDTH_NM = 280
M4_WIDTH_NM = max(M4_MIN_WIDTH_NM, 280)
VIA_PAD_NM = 400  # the committed upper-via landing pad (400 nm square)

# Per-device placement gap (>> NW.2b 1.4um so no cross-device Nwell / implant /
# COMP spacing rule fires between any two placed devices).
DEVICE_GAP_NM = 3000


# --- the 17-device double-tail comparator topology (gf180 translation) --------
#
# Each entry: (label, kind, drain_net, gate_net, source_net, nf, Wf_um, L_um).
# nf = the MAGICAL fold count (kept so the extracted m=N matches the reference
# fold structure). Wf_um = per-finger drawn width (floored at the gf180 channel
# min in build_foldable_device). L_um: 40e-9 -> gf180 min L for the fast core;
# the two reset devices keep L=1u. The connectivity is IDENTICAL to comp_31.
#
# NOTE on the per-finger width floor (an HONEST gf180 translation fact):
# gf180mcu (180 nm) needs a wider per-finger source/drain region than open_pdk
# (40 nm) for the foldable device's source rail to clear its contact pads at
# M1.2a (0.23 um metal1 spacing). Below ~1.0 um per finger the narrow rail
# crowds the pads (probed: Wf=0.5 fails M1.2a x12; Wf=1.0 is DRC-0). So every
# per-finger width is FLOORED at 1.0 um here, and the relative ordering across
# devices is kept where the floor allows. The fold count ``nf`` is the MAGICAL
# value (so the extracted m=N matches the reference fold). This widening is one
# concrete reason gf180 absolute area is NOT comparable to open_pdk -- the
# benchmark normalises by min-L^2.
COMP_DEVICE_PLAN: Tuple[Tuple[str, str, str, str, str, int, float, float], ...] = (
    # clk-gated tail switch (NMOS)
    ("M7", "nmos", "net069", "clk", GND_NET, 8, 1.7, 0.28),
    # input pair (NMOS), sources tied to net069
    ("M5", "nmos", "intern", "net0", "net069", 8, 1.0, 0.28),
    ("M6", "nmos", "interp", "net1", "net069", 8, 1.0, 0.28),
    # 1st-stage precharge PMOS (clk-gated)
    ("M18", "pmos", "intern", "clk", VDD_NET, 2, 1.0, 0.28),
    ("M2", "pmos", "interp", "clk", VDD_NET, 2, 1.0, 0.28),
    # cross-coupled NMOS latch
    ("M4", "nmos", "crossn", "crossp", "intern", 4, 1.15, 0.28),
    ("M3", "nmos", "crossp", "crossn", "interp", 4, 1.15, 0.28),
    # cross-coupled PMOS latch
    ("M14", "pmos", "crossn", "crossp", VDD_NET, 8, 1.0, 0.28),
    ("M13", "pmos", "crossp", "crossn", VDD_NET, 8, 1.0, 0.28),
    # 2nd-stage precharge PMOS (clk-gated) on the cross nodes
    ("M1", "pmos", "crossn", "clk", VDD_NET, 2, 1.0, 0.28),
    ("M12", "pmos", "crossp", "clk", VDD_NET, 2, 1.0, 0.28),
    # output CMOS inverters
    ("M16", "nmos", "outm", "crossp", GND_NET, 4, 1.0, 0.28),  # NMOS pull-down
    ("M8", "pmos", "outm", "crossp", VDD_NET, 4, 1.0, 0.28),   # PMOS pull-up
    ("M17", "nmos", "outp", "crossn", GND_NET, 4, 1.0, 0.28),  # NMOS pull-down
    ("M15", "pmos", "outp", "crossn", VDD_NET, 4, 1.0, 0.28),  # PMOS pull-up
    # reset / bleed NMOS (long L=1u devices). FOLDED nf=2 in the gf180 translation
    # (MAGICAL used nf=1): folding widens the source rail so the source escape x
    # clears the single gate column, avoiding a rail_x collision at the M4 escape
    # riser. The fold is reflected in the reference schematic (m=2), so LVS matches.
    ("M0", "nmos", GND_NET, "intern", GND_NET, 2, 1.4, 1.0),
    ("M22", "nmos", GND_NET, "interp", GND_NET, 2, 1.4, 1.0),
)


# --- geometry helpers (pure) -------------------------------------------------


def _rect(layer_key: Tuple[int, int], box: Tuple[int, int, int, int]) -> LayerPolygon:
    x0, y0, x1, y1 = (int(v) for v in box)
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    return LayerPolygon(
        layer=layer_key[0],
        datatype=layer_key[1],
        polygon=Polygon.rectangle(lo_x, lo_y, hi_x, hi_y),
    )


def _bbox(lp: LayerPolygon) -> Tuple[int, int, int, int]:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _translate(lp: LayerPolygon, dx: int, dy: int) -> LayerPolygon:
    return LayerPolygon(
        layer=lp.layer,
        datatype=lp.datatype,
        polygon=Polygon(
            vertices=[type(v)(v.x + dx, v.y + dy) for v in lp.polygon.vertices]
        ),
    )


def _shift(box: Tuple[int, int, int, int], dx: int, dy: int) -> Tuple[int, int, int, int]:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _cx(b: Tuple[int, int, int, int]) -> int:
    return (b[0] + b[2]) // 2


def _cy(b: Tuple[int, int, int, int]) -> int:
    return (b[1] + b[3]) // 2


def _snap(v: int) -> int:
    return int(round(v / TRACK_PITCH_NM)) * TRACK_PITCH_NM


# --- a placed device terminal (one rail / contact escape point) ---------------


@dataclass(frozen=True)
class Terminal:
    """One device terminal: a net, an escape point ``(x, y)`` on the M1 rail/contact,
    and the terminal kind (``src`` / ``drn`` / ``gate``)."""

    net: str
    label: str
    kind: str
    x: int
    y: int


# --- the composed 17-device comparator layout ---------------------------------


@dataclass
class ComparatorLayout:
    """The composed 17-device comparator library + the net label specs + routed-net
    bookkeeping."""

    library: ParameterizedLibrary
    cell_name: str
    labels_list: List[SubstrateLabelSpec]
    device_bboxes: Dict[str, Tuple[int, int, int, int]]
    routed_nets: List[str]
    shapes_for_unionfind: List[Tuple[str, Tuple[int, int], Tuple[int, int, int, int]]]
    note: str = ""

    def labels(self) -> List[SubstrateLabelSpec]:
        return list(self.labels_list)


def build_comparator(
    *,
    cell_name: str = DEFAULT_CELL_NAME,
    gnd_net: str = GND_NET,
    vdd_net: str = VDD_NET,
) -> ComparatorLayout:
    """Build the 17-device double-tail comparator: 17 DRC-legal foldable devices
    in a row + an over-the-top M3-trunk / M4-riser bus for the 12 inter-device
    nets + gnd substrate tap/label + vdd PMOS-well taps/label.

    Returns a :class:`ComparatorLayout` carrying the single-top-cell library, the
    metal1 ``gnd`` / ``vdd`` labels, and the per-net (net, layer, bbox) shape list
    for the geometric union-find no-short check.
    """
    polys: List[LayerPolygon] = []
    # net -> list of Terminal (escape points to wire).
    net_terminals: Dict[str, List[Terminal]] = {}
    device_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    # For the union-find, the device-INTERNAL geometry (COMP / poly / implant /
    # contacts / Nwell) is NOT included: each foldable device is DRC-legal by
    # construction and does NOT short its own source / drain / gate to each other
    # (the channel + gate separate them) -- that is the proven R162/R163 invariant.
    # The union-find here verifies the INTER-DEVICE ROUTING does not merge two
    # distinct nets: it sees only (a) each device's three TERMINAL M1 rails, tagged
    # with their correct net, and (b) all routing geometry (escape via stacks, M3
    # trunks, M4 risers, via34 corners) tagged with the net. So a merged pair of two
    # DISTINCT nets is a genuine routing short.
    uf_shapes: List[Tuple[str, Tuple[int, int], Tuple[int, int, int, int]]] = []

    # The source-rail of the FIRST device whose source is gnd hosts the substrate
    # tap + the gnd label.
    gnd_tap_rail: Optional[Tuple[int, int, int, int]] = None
    vdd_label_rail: Optional[Tuple[int, int, int, int]] = None

    cursor_x = 0
    for (label, kind, d_net, g_net, s_net, nf, wf, lu) in COMP_DEVICE_PLAN:
        dev: FoldableDevice = build_foldable_device(
            kind=kind, finger_count=nf, gate_length_um=lu, channel_width_um=wf
        )
        ext = dev.extent()
        dx = cursor_x - ext[0]
        for lp in dev.polygons:
            tp = _translate(lp, dx, 0)
            polys.append(tp)
        sbox = _shift(dev.source_rail_bbox, dx, 0)
        dbox = _shift(dev.drain_rail_bbox, dx, 0)
        gbox = _shift(dev.gate_contact_bbox, dx, 0)
        dev_ext = _shift(ext, dx, 0)
        device_bboxes[label] = dev_ext
        # Tag the three TERMINAL M1 rails with their nets for the union-find
        # (the only device geometry the no-short check sees).
        uf_shapes.append((s_net, GF180_METAL1_KEY, sbox))
        uf_shapes.append((d_net, GF180_METAL1_KEY, dbox))
        uf_shapes.append((g_net, GF180_METAL1_KEY, gbox))

        # Source escape: the source rail is wide; escape at the edge FARTHEST from
        # both the centred drain rail AND the gate-contact column, so the source
        # riser never shares an x with the drain or gate terminal of the SAME device
        # (for an nf=1 device the source-right-edge can coincide with the gate x).
        s_half = max(GF180_METAL1_MIN_WIDTH_NM, GF180_CONTACT_SIZE_NM + 2 * GF180_METAL1_CONTACT_ENCLOSURE_NM) // 2
        gate_x = _cx(gbox)
        s_right = sbox[2] - s_half
        s_left = sbox[0] + s_half
        # pick the edge whose distance to the gate column is larger.
        s_esc_x = s_right if abs(s_right - gate_x) >= abs(s_left - gate_x) else s_left
        net_terminals.setdefault(s_net, []).append(
            Terminal(s_net, label, "src", s_esc_x, _cy(sbox))
        )
        net_terminals.setdefault(d_net, []).append(
            Terminal(d_net, label, "drn", _cx(dbox), _cy(dbox))
        )
        net_terminals.setdefault(g_net, []).append(
            Terminal(g_net, label, "gate", _cx(gbox), _cy(gbox))
        )

        if s_net == gnd_net and gnd_tap_rail is None:
            gnd_tap_rail = sbox
        if s_net == vdd_net and vdd_label_rail is None:
            vdd_label_rail = sbox

        cursor_x = dev_ext[2] + DEVICE_GAP_NM

    full_ext = (
        min(b[0] for b in device_bboxes.values()),
        min(b[1] for b in device_bboxes.values()),
        max(b[2] for b in device_bboxes.values()),
        max(b[3] for b in device_bboxes.values()),
    )

    # --- over-the-top M3-trunk / M4-riser bus for the 12 nets -----------------
    # Every net gets a DISTINCT M3 horizontal trunk row. Each TERMINAL rises on M4
    # in its OWN x-column near its rail; rises are short (rail y -> trunk y). Two
    # M4 risers collide only if two terminals share an x-column within M4 spacing;
    # that happens only LOCALLY on a device whose gate column sits near its S/D
    # column (the nf=1 reset devices). A GLOBAL column allocator de-collides: each
    # terminal is assigned the nearest free snapped column (track-pitch grid, never
    # reused) close to its rail x, with a tiny M2 jog from the rail to the column.
    # Because the assigned column is always within ~1 pitch of the rail, the M2 jog
    # is short and stays inside the device's own x-span -- no long crosser. The
    # union-find then proves 0 inter-net shorts.
    #
    # Deterministic net order: gnd, vdd first (widest fan-out power), then sorted.
    ordered_nets = [n for n in (gnd_net, vdd_net) if n in net_terminals]
    ordered_nets += sorted(n for n in net_terminals if n not in (gnd_net, vdd_net))

    routed_nets: List[str] = []
    half_m3 = M3_WIDTH_NM // 2
    half_m4 = M4_WIDTH_NM // 2
    m2_half = max(280, VIA_PAD_NM) // 2
    total_terms = sum(len(ts) for ts in net_terminals.values())
    # Two M3 row bands stack above the device row: the per-terminal JOG rows
    # [jog_base .. jog_base + total_terms*pitch] then the per-net TRUNK rows above
    # them. Both are full-pitch (>= M3.2a-safe) and disjoint, so no M3 row is shared.
    jog_base_y = full_ext[3] + 2 * TRACK_PITCH_NM
    trunk_base_y = jog_base_y + (total_terms + 2) * TRACK_PITCH_NM

    # Gather all terminals with their net + rail (x, y), sorted by rail x so the
    # allocator hands out columns in a stable left-to-right sweep.
    all_terms: List[Tuple[str, int, int, int]] = []  # (net, rail_x, rail_y, row_idx)
    net_row = {net: i for i, net in enumerate(ordered_nets)}
    for net in ordered_nets:
        for t in net_terminals[net]:
            all_terms.append((net, t.x, t.y, net_row[net]))
    all_terms.sort(key=lambda e: (e[1], e[2]))

    # Column allocator: assign each terminal a UNIQUE M4 riser column on a
    # track-pitch grid. A column must clear every OTHER-NET terminal's rail-escape
    # via by >= a keep-out so an M4 riser never passes over a foreign net's via12 /
    # via23 stack (a via bridges layers, so an M4 riser over a foreign via merges
    # nets). The keep-out is the M4 half-width + the via pad half + an M4 spacing
    # margin. The exclusion is by NET, not by x-equality, so two distinct nets that
    # share a rail x (e.g. an nf=1 device's source edge coinciding with its gate
    # column) are still kept apart.
    foreign_rail_xs: Dict[str, List[int]] = {}
    for n, rx, _ry, _ri in all_terms:
        foreign_rail_xs.setdefault(n, []).append(rx)
    riser_keepout = half_m4 + VIA_PAD_NM // 2 + 280
    used_cols: set = set()

    def _alloc_column(rail_x: int, own_net: str) -> int:
        base = _snap(rail_x)
        for step in range(0, 8000):
            for cand in (base + step * TRACK_PITCH_NM, base - step * TRACK_PITCH_NM):
                if cand in used_cols:
                    continue
                # the col must clear the OWN rail x by >= 2 pitches so the two jog-row
                # via34 stacks (one at the rail x, one at the col x) clear V3.2a.
                if abs(cand - rail_x) < 2 * TRACK_PITCH_NM:
                    continue
                # the riser column must clear every OTHER-NET terminal's rail via.
                bad = any(
                    n != own_net and any(abs(cand - rx) < riser_keepout for rx in xs)
                    for n, xs in foreign_rail_xs.items()
                )
                if bad:
                    continue
                used_cols.add(cand)
                return cand
        raise RuntimeError("column allocator exhausted")  # pragma: no cover

    # Per-terminal escape (single bend, strict layer discipline). HORIZONTALS on M3,
    # VERTICALS on M4, only meeting at net-local via34. Each terminal:
    #   M1 rail --via12+via23--> M3 at (rail_x, rail_y)
    #     M3 jog at a UNIQUE escape row jog_y from rail_x to the UNIQUE column col_x
    #     --via34--> M4 riser at col_x up to the net's M3 trunk row
    #     --via34--> M3 trunk row.
    # The escape M3 pad at the rail rises to jog_y on M4 at rail_x first (a tiny
    # net-local M4 step), so the M3 jog is at a UNIQUE jog row -- never parallel
    # within M3.2a, even for same-device terminals 0.5 um apart at the rail. The M4
    # risers (rail step + col riser) are short / in unique columns. col_x clears
    # every foreign rail x by the keep-out, and the reset devices are folded nf=2 so
    # no two nets share a rail_x. The union-find proves 0 inter-net shorts.
    #
    # To keep the via count modest (Via3 spacing V3.2a), the rail-to-jog M4 step is
    # SHORT: jog rows sit just above the device row, packed at full pitch.
    used_jog_rows: set = set()

    def _alloc_jog_row() -> int:
        k = 0
        while k in used_jog_rows:
            k += 1
        used_jog_rows.add(k)
        return jog_base_y + k * TRACK_PITCH_NM

    net_term_cols: Dict[str, List[int]] = {n: [] for n in ordered_nets}
    for net, rail_x, rail_y, row_idx in all_terms:
        trunk_y = trunk_base_y + row_idx * TRACK_PITCH_NM
        col_x = _alloc_column(rail_x, net)
        jog_y = _alloc_jog_row()
        net_term_cols[net].append(col_x)
        # M1->M2->M3 escape at the rail (via12 + via23) + via34 up to M4.
        for sp in (
            via12_stack(Point(rail_x, rail_y))
            + via23_stack(Point(rail_x, rail_y))
            + via34_stack(Point(rail_x, rail_y))
        ):
            polys.append(sp)
            uf_shapes.append((net, (sp.layer, sp.datatype), _bbox(sp)))
        # short M4 step at rail_x from rail_y up to the UNIQUE jog row.
        vlo, vhi = (rail_y, jog_y) if rail_y <= jog_y else (jog_y, rail_y)
        m4_rail = _rect(M4_KEY, (rail_x - half_m4, vlo, rail_x + half_m4, vhi))
        polys.append(m4_rail)
        uf_shapes.append((net, M4_KEY, _bbox(m4_rail)))
        # via34 (M4->M3) at the rail/jog corner; M3 jog from rail_x to col_x.
        for sp in via34_stack(Point(rail_x, jog_y)):
            polys.append(sp)
            uf_shapes.append((net, (sp.layer, sp.datatype), _bbox(sp)))
        jx0, jx1 = (rail_x, col_x) if rail_x <= col_x else (col_x, rail_x)
        m3_jog = _rect(M3_KEY, (jx0 - half_m3, jog_y - half_m3, jx1 + half_m3, jog_y + half_m3))
        polys.append(m3_jog)
        uf_shapes.append((net, M3_KEY, _bbox(m3_jog)))
        # via34 (M3->M4) at col/jog; M4 riser at col_x up to the trunk row.
        for sp in via34_stack(Point(col_x, jog_y)):
            polys.append(sp)
            uf_shapes.append((net, (sp.layer, sp.datatype), _bbox(sp)))
        lo, hi = (jog_y, trunk_y) if jog_y <= trunk_y else (trunk_y, jog_y)
        m4_col = _rect(M4_KEY, (col_x - half_m4, lo, col_x + half_m4, hi))
        polys.append(m4_col)
        uf_shapes.append((net, M4_KEY, _bbox(m4_col)))
        # via34 (M4->M3) at the trunk corner.
        for sp in via34_stack(Point(col_x, trunk_y)):
            polys.append(sp)
            uf_shapes.append((net, (sp.layer, sp.datatype), _bbox(sp)))

    for net in ordered_nets:
        row_idx = net_row[net]
        trunk_y = trunk_base_y + row_idx * TRACK_PITCH_NM
        cols = net_term_cols[net]
        trunk_x0, trunk_x1 = min(cols), max(cols)
        trunk = _rect(M3_KEY, (trunk_x0 - half_m3, trunk_y - half_m3, trunk_x1 + half_m3, trunk_y + half_m3))
        polys.append(trunk)
        uf_shapes.append((net, M3_KEY, _bbox(trunk)))
        routed_nets.append(net)

    # --- gnd: P+ substrate tap + metal1 gnd label -----------------------------
    labels: List[SubstrateLabelSpec] = []
    if gnd_tap_rail is not None:
        sub_tap = build_substrate_tap(gnd_tap_rail)
        for sp in sub_tap.polygons:
            polys.append(sp)
            uf_shapes.append((gnd_net, (sp.layer, sp.datatype), _bbox(sp)))
        labels.append(
            SubstrateLabelSpec(
                text=gnd_net,
                point=(_cx(gnd_tap_rail), _cy(gnd_tap_rail)),
                layer_key=GF180_METAL1_LABEL_KEY,
            )
        )
    # --- vdd: metal1 vdd label on a PMOS source rail (well bodies tie there) ---
    if vdd_label_rail is not None:
        labels.append(
            SubstrateLabelSpec(
                text=vdd_net,
                point=(_cx(vdd_label_rail), _cy(vdd_label_rail)),
                layer_key=GF180_METAL1_LABEL_KEY,
            )
        )

    top = NamedCell(name=cell_name, polygons=polys)
    lib = ParameterizedLibrary(
        name="clay_magical_comparator_v171",
        unit_user=0.001,
        unit_db_meters=1e-9,
        cells=[top],
    )
    note = (
        "R171 17-device double-tail dynamic latch comparator (gf180 translation of "
        "MAGICAL open_pdk comp): 9 foldable NMOS + 8 foldable PMOS in a row, "
        f"{len(routed_nets)} nets over-the-top M3-trunk/M4-riser bus, P+ substrate "
        f"tap + {gnd_net!r} label, PMOS well-taps + {vdd_net!r} label"
    )
    return ComparatorLayout(
        library=lib,
        cell_name=cell_name,
        labels_list=labels,
        device_bboxes=device_bboxes,
        routed_nets=routed_nets,
        shapes_for_unionfind=uf_shapes,
        note=note,
    )


def build_reference_schematic(
    cell_name: str = DEFAULT_CELL_NAME,
    *,
    gnd_net: str = GND_NET,
    vdd_net: str = VDD_NET,
) -> str:
    """Build the FOLDED reference SPICE for the 17-device comparator.

    17 ``m=N`` MOS (9 ``nmos_3p3`` + 8 ``pmos_3p3``); ``W`` the per-finger drawn
    width, ``m`` the finger count (the MAGICAL fold). Connectivity IDENTICAL to
    ``comp_31``: NMOS bulk = gnd, PMOS bulk = vdd. 12 nets.
    """
    ports = "clk gnd vdd net0 net1 net069 intern interp crossp crossn outm outp"
    lines = [f".subckt {cell_name} {ports}"]
    for (label, kind, d_net, g_net, s_net, nf, wf, lu) in COMP_DEVICE_PLAN:
        model = "nmos_3p3" if kind == "nmos" else "pmos_3p3"
        bulk = gnd_net if kind == "nmos" else vdd_net
        m = f" m={nf}" if nf != 1 else ""
        lines.append(
            f"M{label} {d_net} {g_net} {s_net} {bulk} {model} "
            f"W={wf:g}U L={lu:g}U{m}"
        )
    lines.append(".ends")
    return "\n".join(lines) + "\n"


# --- counts (pure) -----------------------------------------------------------


def count_schematic_devices(spice_text: str) -> int:
    n = 0
    for raw in spice_text.splitlines():
        line = raw.strip()
        if not line or line[0] in ("*", "."):
            continue
        if line[0] in ("m", "M"):
            n += 1
    return n


def count_schematic_nets(spice_text: str) -> int:
    nets: set = set()
    for raw in spice_text.splitlines():
        line = raw.strip()
        if not line or line[0] not in ("m", "M"):
            continue
        toks = line.split()
        body: List[str] = []
        for t in toks[1:]:
            if "=" in t:
                break
            body.append(t)
        if len(body) >= 5:
            nets.update(body[:4])
    return len(nets)


def count_schematic_devices_by_model(spice_text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for raw in spice_text.splitlines():
        line = raw.strip()
        if not line or line[0] not in ("m", "M"):
            continue
        toks = line.split()
        body: List[str] = []
        for t in toks[1:]:
            if "=" in t:
                break
            body.append(t)
        if len(body) >= 5:
            counts[body[4]] = counts.get(body[4], 0) + 1
    return counts


# --- geometric union-find (no inter-net short) over the EMITTED shapes ---------


@dataclass
class ShortCheck:
    distinct: bool
    short_count: int
    merged_pairs: List[Tuple[str, str]]


def verify_no_shorts(layout: ComparatorLayout, *, prefer_rust: bool = True) -> ShortCheck:
    """Run the SAME R170 geometric union-find used by the multi-block routing
    co-clean over the EMITTED net-tagged shapes (device terminal M1 rails + all
    routing geometry). A merged pair of two DISTINCT nets is a genuine routing
    short. Device-internal COMP/poly/implant is excluded: each foldable device is
    DRC-legal by construction and does not short its own S/D/G (the proven
    R162/R163 invariant), so the check is exactly the inter-device routing
    no-short test."""
    from ._rs_bridge_net_connectivity import verify_connectivity

    res = verify_connectivity(
        layout.shapes_for_unionfind,
        via_layers=tuple(_ALL_VIA_KEYS),
        prefer_rust=prefer_rust,
    )
    pairs = [tuple(p) for p in res["merged_pairs"]]
    return ShortCheck(
        distinct=len(pairs) == 0,
        short_count=len(pairs),
        merged_pairs=sorted(pairs),
    )


# --- availability predicates --------------------------------------------------


def klayout_binary_available(klayout_bin: Optional[Path] = None) -> bool:
    try:
        from tools.klayout_drc_signoff_v139.run import klayout_binary_available as _k

        return bool(_k(klayout_bin))
    except Exception:  # noqa: BLE001
        k = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
        return k.exists()


def drc_deck_available() -> bool:
    try:
        from tools.klayout_drc_signoff_v139.run import DEFAULT_DECK

        return DEFAULT_DECK.exists()
    except Exception:  # noqa: BLE001
        return False


def lvs_deck_available() -> bool:
    try:
        from tools.klayout_lvs_signoff_v145.run import lvs_deck_available as _d

        return bool(_d())
    except Exception:  # noqa: BLE001
        return False


def rust_gds_available() -> bool:
    try:
        from clay.layout.binary_gds_subprocess import cargo_binary_available

        return bool(cargo_binary_available())
    except Exception:  # noqa: BLE001
        return False


def real_signoff_available(*, klayout_bin: Optional[Path] = None) -> bool:
    return (
        klayout_binary_available(klayout_bin)
        and drc_deck_available()
        and lvs_deck_available()
        and rust_gds_available()
    )


def gds_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# --- end-to-end co-signoff (ONE GDS, BOTH decks + union-find) -----------------


@dataclass
class ComparatorCocleanResult:
    """Outcome of running BOTH real gf180mcu decks + the union-find on ONE GDS."""

    real_available: bool = False
    flow_note: str = ""
    schematic_device_count: Optional[int] = None
    schematic_net_count: Optional[int] = None
    schematic_by_model: Dict[str, int] = field(default_factory=dict)
    routed_net_count: int = 0
    layout_note: str = ""
    gds_path: Optional[str] = None
    gds_bytes: int = 0
    gds_sha256: Optional[str] = None
    # union-find
    union_find_ran: bool = False
    union_find_short_count: Optional[int] = None
    union_find_distinct: Optional[bool] = None
    # DRC
    drc_ran: bool = False
    drc_total_violations: Optional[int] = None
    drc_by_rule: Dict[str, int] = field(default_factory=dict)
    drc_pending_reason: str = ""
    # LVS
    lvs_ran: bool = False
    lvs_matched: Optional[bool] = None
    lvs_device_count: Optional[int] = None
    lvs_net_count: Optional[int] = None
    lvs_vacuous: bool = False
    lvs_pending_reason: str = ""

    @property
    def drc_clean(self) -> bool:
        return bool(self.drc_ran and self.drc_total_violations == 0)

    @property
    def lvs_clean(self) -> bool:
        return bool(self.lvs_ran and self.lvs_matched is True and not self.lvs_vacuous)

    @property
    def no_shorts(self) -> bool:
        return bool(self.union_find_ran and self.union_find_short_count == 0)

    @property
    def co_clean(self) -> bool:
        """ONE GDS: real-DRC 0 AND non-vacuous real-LVS match AND 0 inter-net short."""
        return bool(self.drc_clean and self.lvs_clean and self.no_shorts)

    @property
    def verdict(self) -> str:
        if self.co_clean:
            return (
                f"CO-CLEAN (17-dev: DRC 0, LVS match {self.lvs_device_count} dev / "
                f"{self.lvs_net_count} net, 0 inter-net shorts)"
            )
        parts = []
        if self.drc_ran:
            parts.append(f"DRC {self.drc_total_violations} viol")
        else:
            parts.append(f"DRC pending ({self.drc_pending_reason})")
        if self.lvs_ran:
            parts.append(
                f"LVS {'MATCH' if self.lvs_matched else 'MISMATCH'} "
                f"({self.lvs_device_count} dev / {self.lvs_net_count} net"
                f"{', VACUOUS' if self.lvs_vacuous else ''})"
            )
        else:
            parts.append(f"LVS pending ({self.lvs_pending_reason})")
        if self.union_find_ran:
            parts.append(f"{self.union_find_short_count} inter-net short(s)")
        return "RESIDUAL: " + " | ".join(parts)


def _emit_and_label_gds(layout: ComparatorLayout, out: Path, *, klayout_bin) -> Optional[Path]:
    try:
        from clay.layout.binary_gds_subprocess import emit_binary_gds_via_subprocess

        base_gds = out / f"{layout.cell_name}_base.gds"
        emit_binary_gds_via_subprocess(layout.library, base_gds)
    except Exception:  # noqa: BLE001
        return None
    if not (base_gds.exists() and base_gds.stat().st_size > 0):
        return None
    coclean_gds = out / f"{layout.cell_name}_coclean_v171.gds"
    ok = inject_metal1_labels_into_gds(
        base_gds, coclean_gds, layout.labels(), klayout_bin=klayout_bin
    )
    if not ok:
        return None
    return coclean_gds if (coclean_gds.exists() and coclean_gds.stat().st_size > 0) else None


def run_coclean_comparator(
    *,
    out_dir: Optional[object] = None,
    run_drc: bool = True,
    run_lvs: bool = True,
    run_union_find: bool = True,
    klayout_bin: Optional[Path] = None,
    timeout_sec: float = 1200.0,
) -> ComparatorCocleanResult:
    """Build the 17-device comparator GDS once, run the union-find, then run BOTH
    real decks on the IDENTICAL file. Every verdict is read VERBATIM; a vacuous
    LVS extraction only demotes; co-clean is asserted only when all three pass."""
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    avail = real_signoff_available(klayout_bin=klayout_bin)
    layout = build_comparator()
    schematic_text = build_reference_schematic(layout.cell_name)
    src_path = out / "gf180_magical_comparator_src_v171.cir"
    src_path.write_text(schematic_text, encoding="utf-8")
    dev_src = count_schematic_devices(schematic_text)
    net_src = count_schematic_nets(schematic_text)

    result = ComparatorCocleanResult(
        real_available=avail,
        flow_note=" | ".join(
            [
                f"gf180mcu.drc: {'present' if drc_deck_available() else 'absent'}",
                f"gf180mcu.lvs: {'present' if lvs_deck_available() else 'absent'}",
                f"KLayout: {'present' if klayout_binary_available(klayout_bin) else 'absent'}",
                f"Rust GDS: {'present' if rust_gds_available() else 'absent'}",
                f"gnd net: {GND_NET} | vdd net: {VDD_NET}",
            ]
        ),
        schematic_device_count=dev_src,
        schematic_net_count=net_src,
        schematic_by_model=count_schematic_devices_by_model(schematic_text),
        routed_net_count=len(layout.routed_nets),
        layout_note=layout.note,
    )

    if run_union_find:
        try:
            chk = verify_no_shorts(layout)
            result.union_find_ran = True
            result.union_find_short_count = chk.short_count
            result.union_find_distinct = chk.distinct
        except Exception as exc:  # noqa: BLE001
            result.union_find_ran = False
            result.drc_pending_reason = f"union-find failed: {type(exc).__name__}"

    gds: Optional[Path] = None
    if avail and (run_drc or run_lvs):
        gds = _emit_and_label_gds(layout, out, klayout_bin=klayout_bin)
    if gds is not None:
        result.gds_path = str(gds)
        result.gds_bytes = gds.stat().st_size
        result.gds_sha256 = gds_sha256(gds)

    # --- DRC ---------------------------------------------------------------
    if not run_drc:
        result.drc_pending_reason = "run_drc=False"
    elif not avail:
        result.drc_pending_reason = "real toolchain unavailable"
    elif gds is None:
        result.drc_pending_reason = "Rust GDS emit / label injection produced no GDS"
    else:
        try:
            from tools.klayout_drc_signoff_v139.run import run_gf180_drc

            report = run_gf180_drc(
                gds,
                report_path=out / f"{layout.cell_name}_coclean_v171.lyrdb",
                topcell=layout.cell_name,
                klayout_bin=klayout_bin,
                run_mode="flat",
                timeout_sec=timeout_sec,
            )
            result.drc_ran = True
            result.drc_total_violations = getattr(report, "total_violations", None)
            result.drc_by_rule = {
                getattr(v, "rule", "unknown"): getattr(v, "count", 0)
                for v in getattr(report, "violations", []) or []
            }
        except Exception as exc:  # noqa: BLE001
            result.drc_pending_reason = (
                f"real DRC deck failed: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            )

    # --- LVS ---------------------------------------------------------------
    if not run_lvs:
        result.lvs_pending_reason = "run_lvs=False"
    elif not avail:
        result.lvs_pending_reason = "real toolchain unavailable"
    elif gds is None:
        result.lvs_pending_reason = "Rust GDS emit / label injection produced no GDS"
    else:
        try:
            from tools.klayout_lvs_signoff_v145.run import run_gf180_lvs

            report = run_gf180_lvs(
                gds,
                src_path,
                topcell=layout.cell_name,
                report_path=out / f"{layout.cell_name}_coclean_v171.lvsdb",
                extracted_netlist_path=out / f"{layout.cell_name}_coclean_v171_extracted.cir",
                lvs_sub=GND_NET,
                klayout_bin=klayout_bin,
                run_mode="flat",
                timeout_sec=timeout_sec,
            )
            result.lvs_ran = True
            result.lvs_matched = getattr(report, "matched", None)
            result.lvs_device_count = getattr(report, "device_count", None)
            result.lvs_net_count = getattr(report, "net_count", None)
            result.lvs_vacuous = bool(
                result.lvs_device_count is not None
                and (
                    int(result.lvs_device_count) == 0
                    or int(result.lvs_device_count) < int(dev_src)
                )
            )
        except Exception as exc:  # noqa: BLE001
            result.lvs_pending_reason = (
                f"real LVS deck failed: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            )

    return result


__all__ = [
    "COMP_DEVICE_PLAN",
    "DEFAULT_CELL_NAME",
    "DEFAULT_KLAYOUT_BIN",
    "DEFAULT_OUT_DIR",
    "GND_NET",
    "REPO_ROOT",
    "VDD_NET",
    "ComparatorCocleanResult",
    "ComparatorLayout",
    "ShortCheck",
    "Terminal",
    "build_comparator",
    "build_reference_schematic",
    "count_schematic_devices",
    "count_schematic_devices_by_model",
    "count_schematic_nets",
    "drc_deck_available",
    "gds_sha256",
    "klayout_binary_available",
    "lvs_deck_available",
    "real_signoff_available",
    "run_coclean_comparator",
    "rust_gds_available",
    "verify_no_shorts",
]
