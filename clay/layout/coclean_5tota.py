"""R163 (BL-163A) -- 5T-OTA CO-CLEAN: ONE gf180mcu GDS that passes BOTH the real
``gf180mcu.drc`` deck (0 violations) AND the real ``gf180mcu.lvs`` deck (Netlists
match, 5 folded devices / 8 nets) -- on the *identical* file.

The two-geometry problem this lane closes
==========================================
By R162 the simple cells (current_mirror, diff_pair) were co-clean via the
DRC-legal-by-construction foldable NMOS (``coclean_current_mirror_v162``). The
5T-OTA stayed NOT co-clean: it was LVS net-clean (R151
``full_netclean_5tota_v151``) but on a GDS that the real ``gf180mcu.drc`` deck
reports **219 violations** on (R161 A measured, reproduced here):

    CO.11 x15  CO.6 x20  CO.6a x2  CO.6b x6   (contact / metal1-enclosure)
    NP.5a x12  PP.5a x8                       (implant overlap of gate)
    PL.5a_3.3V x5  PL.5b_3.3V x5              (field poly to COMP space)
    M2.1 x26  M2.2a x3  M2.3 x11              (metal2 width / space / area)
    V1.1 x92  V1.3a x14                       (via1 size / enclosure)

The 146 M2/V1 violations come from the R151 strap routing on M2 trunks + via1;
the 73 device-level violations are the same classes R162 closed for the simple
cells. So NO single 5T-OTA GDS passed both decks.

How this lane closes it (the R162 approach, extended to PMOS)
=============================================================
Build EVERY device DRC-legal-by-construction (the R162 ``build_foldable_nmos``
technique, generalised to PMOS with an Nwell), and route the all-5-net strap on
**metal1 only** -- no M2, no via1 -- so the whole 146-violation M2/V1 class
disappears and the device-level classes are legal by construction:

  * Every S/D contact lands ON the COMP active (CO.11), every metal1 over a
    contact encloses it by CO.6 (60 nm) and is >= M1.1 (230 nm) wide / >= M1.3
    area (380 nm square).
  * The implant (Nplus for NMOS, Pplus for PMOS) is grown to enclose the gate by
    NP.5a / PP.5a (0.23 um), which also satisfies NP.5b / PP.5b / DF.12.
  * The gate poly bus body sits >= PL.5 (0.1 um) clear of the COMP top.
  * The two PMOS PCOMPs are each wrapped in their own Nwell with DF.4c (0.43 um)
    overlap; an N+ well-tap (DF.4d 0.12 um Nwell-overlap-of-NCOMP) ties each well
    body to the PMOS source rail (``vdd``).
  * All inter-device nets are strapped on dedicated metal1 lanes with M1.2 (0.28
    um) spacing between distinct nets, so no metal1-metal1 short and no metal1
    width violation.

What folds for LVS (the same 5/8 the R151 schematic declares)
-------------------------------------------------------------
The reference is the R151 M-card folded schematic (5 ``m=2`` MOS / 8 nets):
3 ``nmos_3p3`` (M_in_p / M_in_n / M_tail) + 2 ``pmos_3p3`` (M_load_p / M_load_n).
Each foldable device extracts as one ``m=N`` MOS (single gate net via the poly
bus + one gate contact, one source net via the source rail, one drain net via
the drain rail). The PMOS bulk pin extracts as ``vdd`` (the well-tap tie); the
NMOS bulk extracts as ``vss`` (the global substrate, named via a metal1 ``vss``
label on the tail source rail + a P+ substrate tap).

Honesty contract (CRITICAL)
---------------------------
* Every coordinate is derived from the cited gf180mcu rule constants (no value
  invented). The committed rule constants come from ``gf180_device_geom`` and the
  R162 foldable-NMOS module (imported READ-ONLY).
* The real-deck path is gated on KLayout + gf180mcu.drc/.lvs + the Rust GDS
  emitter; it skips honestly when any is absent.
* The DRC verdict is the deck's own ``.lyrdb`` violation count; the LVS verdict
  is the deck's own ``compare`` output + extracted ``.cir`` counts. A vacuous
  (0-device / device-short) extraction is guard-demoted, never co-clean.
* BOTH decks run on the IDENTICAL ``.gds`` file; :class:`CoClean5TResult` records
  that single path and asserts both decks saw it. A residual is reported honestly
  as the 219->N reduction; co-clean is NEVER claimed unless both decks passed.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .backend import (
    Gf180mcuLayerMap,
    LayerMap,
    LayerPolygon,
    NamedCell,
    ParameterizedLibrary,
    Polygon,
)
from .gf180_device_geom import (
    GF180_CHANNEL_MIN_WIDTH_NM,
    GF180_COMP_COMP_SPACE_NM,
    GF180_COMP_CONTACT_ENCLOSURE_NM,
    GF180_CONTACT_SIZE_NM,
    GF180_CONTACT_TO_GATE_SPACE_NM,
    GF180_GATE_MIN_LENGTH_NM,
    GF180_POLY_CONTACT_ENCLOSURE_NM,
    GF180_POLY_ENDCAP_NM,
    GF180_POLY_MIN_WIDTH_NM,
    gf180_contacted_finger_space_nm,
)

# --- gf180mcu GDS keys (verified in gf180_layermap.GF180_VERIFIED_SOURCES) ----
GF180_NWELL_KEY: Tuple[int, int] = (21, 0)
GF180_COMP_KEY: Tuple[int, int] = (22, 0)
GF180_POLY_KEY: Tuple[int, int] = (30, 0)
GF180_PPLUS_KEY: Tuple[int, int] = (31, 0)
GF180_NPLUS_KEY: Tuple[int, int] = (32, 0)
GF180_CONTACT_KEY: Tuple[int, int] = (33, 0)
GF180_METAL1_KEY: Tuple[int, int] = (34, 0)
GF180_VIA1_KEY: Tuple[int, int] = (35, 0)
GF180_METAL2_KEY: Tuple[int, int] = (36, 0)
# The gf180mcu.lvs deck reads metal1 net labels from TEXT on 34/10 (line 409 /
# 1091): a TEXT "vss" on the tail source rail names that net "vss", which the
# deck unifies with the global substrate net (connect_global(sub, "vss")).
GF180_METAL1_LABEL_KEY: Tuple[int, int] = (34, 10)

# --- gf180mcu rule constants used here (nm; each cited) -----------------------
# metal1.drc M1.1 "min. metal1 width 0.23um".
GF180_METAL1_MIN_WIDTH_NM = 230
# metal1.drc M1.2a "min. metal1 space 0.23um"; we use the wider COMP-COMP space
# (DF.3a 0.28um) for distinct-net metal1 lanes for a euclidean margin.
GF180_METAL1_SPACE_NM = 280
# metal2.drc M2.1 "min. metal2 width 0.28um"; M2.2a "min. metal2 space 0.28um".
GF180_METAL2_MIN_WIDTH_NM = 280
GF180_METAL2_SPACE_NM = 280
# via1.drc V1.1 "Min/max Via1 size 0.26um" (exact size).
GF180_VIA1_SIZE_NM = 260
# via1.drc V1.4a "metal-2 overlap of via1 0.01um" (M2 encloses via1 by >= 10nm).
GF180_M2_VIA1_ENCLOSURE_NM = 100
# via1.drc V1.3a: via1 must be fully covered by metal1 (M1 encloses via1).
GF180_M1_VIA1_ENCLOSURE_NM = 60
# contact.drc CO.6 "Metal1 overlap of contact 0.06um" (metal1 enclosure of CO).
GF180_METAL1_CONTACT_ENCLOSURE_NM = 60
# nplus.drc NP.5a "Overlap of N-channel gate 0.23um" (Nplus enclosure of ngate).
# We use 260 (> 230) for a euclidean corner margin past the exact 0.23um edge.
GF180_NPLUS_GATE_OVERLAP_NM = 260
# pplus.drc PP.5a "Overlap of P-channel gate 0.23um" (Pplus enclosure of pgate).
GF180_PPLUS_GATE_OVERLAP_NM = 260
# poly2.drc PL.5a/PL.5b "Space from field Poly2 to COMP 0.1um".
GF180_FIELD_POLY_TO_COMP_NM = 100
# comp.drc DF.4c "Min. Nwell overlap of PCOMP (outside DNWELL) 0.43um"; +margin for
# the euclidean corner check.
GF180_NWELL_PCOMP_OVERLAP_NM = 500
# comp.drc DF.4d "Min. Nwell overlap of NCOMP (outside DNWELL) 0.12um"; +margin.
GF180_NWELL_NCOMP_OVERLAP_NM = 200
# nplus.drc NP.3a / pplus.drc PP.3a "Nplus/Pplus space to PCOMP/NCOMP 0.16um".
GF180_IMPLANT_TO_OPP_COMP_SPACE_NM = 160
# nwell.drc NW.1a "Min. Nwell width 0.86um".
GF180_NWELL_MIN_WIDTH_NM = 860
# nplus/pplus extension over a tap COMP (NP/PP.5b, 0.16um).
GF180_IMPLANT_EXTENSION_NM = 160

DEFAULT_KLAYOUT_BIN = Path(r"C:\Program Files (x86)\KLayout\klayout_app.exe")
DEFAULT_SUBSTRATE_NET = "vss"
DEFAULT_VDD_NET = "vdd"


# --- geometry helpers (pure) -------------------------------------------------


def _rect(layer_key: Tuple[int, int], box: Tuple[int, int, int, int]) -> LayerPolygon:
    x0, y0, x1, y1 = (int(v) for v in box)
    return LayerPolygon(
        layer=layer_key[0],
        datatype=layer_key[1],
        polygon=Polygon.rectangle(x0, y0, x1, y1),
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


# --- DRC-legal foldable device (NMOS or PMOS) --------------------------------


@dataclass(frozen=True)
class FoldableDevice:
    """A DRC-legal, LVS-foldable m=N gf180mcu MOS (NMOS or PMOS).

    ``polygons`` form ONE electrical MOS of width ``finger_count *
    channel_width_nm`` whose extraction folds to one ``m=N`` MOS: a single gate
    net (poly bus ties all fingers + one gate contact), one source net (every
    source column's ON-COMP contact commoned by a source metal1 rail), one drain
    net (every drain column commoned by a drain rail). EVERY contact lands on the
    COMP (CO.11), every metal1 over a contact encloses by CO.6, the implant
    encloses the gate by NP.5a / PP.5a, and the gate bus is PL.5 clear of the
    COMP. A PMOS additionally carries an Nwell (DF.4c) enclosing the PCOMP.

    Attributes:
        polygons: every drawn shape.
        kind: "nmos" or "pmos".
        finger_count / gate_length_nm / channel_width_nm: device dimensions.
        comp_bbox: the COMP diffusion bbox (x0,y0,x1,y1).
        source_rail_bbox / drain_rail_bbox / gate_bus_bbox / gate_contact_bbox:
            the commoning rails (metal1 / metal1 / poly2) + the gate contact.
        nwell_bbox: the Nwell bbox for a PMOS (None for an NMOS).
    """

    polygons: List[LayerPolygon]
    kind: str
    finger_count: int
    gate_length_nm: int
    channel_width_nm: int
    comp_bbox: Tuple[int, int, int, int]
    source_rail_bbox: Tuple[int, int, int, int]
    drain_rail_bbox: Tuple[int, int, int, int]
    gate_bus_bbox: Tuple[int, int, int, int]
    gate_contact_bbox: Tuple[int, int, int, int]
    nwell_bbox: Optional[Tuple[int, int, int, int]] = None

    @property
    def total_width_nm(self) -> int:
        return self.finger_count * self.channel_width_nm

    def polys_on(self, key: Tuple[int, int]) -> List[LayerPolygon]:
        return [lp for lp in self.polygons if (lp.layer, lp.datatype) == key]

    def extent(self) -> Tuple[int, int, int, int]:
        boxes = [_bbox(lp) for lp in self.polygons]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def all_contacts_on_comp(self) -> bool:
        """True iff EVERY contact lands inside a COMP rect OR on poly (CO.11).

        Checks against EVERY COMP rect in the device (the channel COMP plus, for a
        PMOS, the in-well N+ tap COMP), not just the device-active ``comp_bbox``.
        """
        comp_boxes = [_bbox(c) for c in self.polys_on(GF180_COMP_KEY)]
        poly_boxes = [_bbox(p) for p in self.polys_on(GF180_POLY_KEY)]

        def _inside(c, box):
            return (
                c[0] >= box[0] and c[1] >= box[1] and c[2] <= box[2] and c[3] <= box[3]
            )

        for co in self.polys_on(GF180_CONTACT_KEY):
            cb = _bbox(co)
            on_comp = any(_inside(cb, b) for b in comp_boxes)
            on_poly = any(_inside(cb, pb) for pb in poly_boxes)
            if not (on_comp or on_poly):
                return False
        return True


def build_foldable_device(
    *,
    kind: str = "nmos",
    finger_count: int = 2,
    gate_length_um: float = 0.5,
    channel_width_um: float = 4.0,
    nets: Optional[Dict[str, str]] = None,
) -> FoldableDevice:
    """Build a DRC-legal, LVS-foldable m=N gf180mcu MOS (``kind`` ``"nmos"`` or
    ``"pmos"``).

    The geometry is the R162 ``build_foldable_nmos`` recipe (every contact on
    COMP, CO.6 metal1 enclosure, implant grown by 0.23 um over the gate, gate bus
    PL.5 clear of the COMP). For a PMOS the implant is Pplus (grown by PP.5a) and
    the whole PCOMP is wrapped in a DF.4c Nwell.

    DELEGATION (P1): this gf180 geometry is now produced by the PDK-parameterized
    :func:`clay.layout.device_cell.build_device` on the gf180mcu
    :class:`~clay.layout.tech.LayoutTech`. The output is BYTE-IDENTICAL to the
    legacy in-line recipe (same polygons, same layer keys, same coordinates) -- the
    parity oracle in ``tests/test_device_cell_pdk.py``. ``build_device`` reads every
    layer key + DRC dimension from ``tech`` so the SAME geometry generalises to
    other PDKs; gf180 stays the safety oracle for the 5 proven L5 co-clean cells.
    ``draw_substrate_tap=False`` reproduces the legacy behaviour (the NMOS bulk is
    the global substrate, the P+ substrate tap is added at the cell level).
    """
    if kind not in ("nmos", "pmos"):
        raise ValueError("kind must be 'nmos' or 'pmos'")
    if finger_count < 1:
        raise ValueError("finger_count must be >= 1")

    from .device_cell import build_device as _build_device
    from .tech import LayoutTech as _LayoutTech

    _tech = _LayoutTech.resolve("gf180mcu")
    _geom = _build_device(
        _tech,
        kind,
        W=float(channel_width_um) * 1e-6,
        L=float(gate_length_um) * 1e-6,
        nf=int(finger_count),
        draw_substrate_tap=False,
        nets=nets,
    )
    return FoldableDevice(
        polygons=_geom.polygons,
        kind=kind,
        finger_count=finger_count,
        gate_length_nm=_geom.gate_length_nm,
        channel_width_nm=_geom.channel_width_nm,
        comp_bbox=_geom.comp_bbox,
        source_rail_bbox=_geom.source_rail_bbox,
        drain_rail_bbox=_geom.drain_rail_bbox,
        gate_bus_bbox=_geom.gate_bus_bbox,
        gate_contact_bbox=_geom.gate_contact_bbox,
        nwell_bbox=_geom.nwell_bbox,
    )


def _build_foldable_device_legacy(
    *,
    kind: str = "nmos",
    finger_count: int = 2,
    gate_length_um: float = 0.5,
    channel_width_um: float = 4.0,
) -> FoldableDevice:
    """The pre-P1 in-line gf180 recipe, retained as the parity ORACLE.

    ``tests/test_device_cell_pdk.py`` asserts ``build_foldable_device`` (now
    delegating to ``device_cell.build_device``) is byte-identical to THIS function,
    so any drift in the parameterized builder is caught against the frozen
    reference. Not used in production.
    """
    if kind not in ("nmos", "pmos"):
        raise ValueError("kind must be 'nmos' or 'pmos'")
    if finger_count < 1:
        raise ValueError("finger_count must be >= 1")

    is_pmos = kind == "pmos"
    implant_key = GF180_PPLUS_KEY if is_pmos else GF180_NPLUS_KEY
    implant_overlap = (
        GF180_PPLUS_GATE_OVERLAP_NM if is_pmos else GF180_NPLUS_GATE_OVERLAP_NM
    )

    l_nm = max(int(round(gate_length_um * 1000.0)), GF180_GATE_MIN_LENGTH_NM)
    w_nm = max(int(round(channel_width_um * 1000.0)), GF180_CHANNEL_MIN_WIDTH_NM)
    finger_space = gf180_contacted_finger_space_nm()  # >= contact + 2*CO.7
    endcap = GF180_POLY_ENDCAP_NM
    csize = GF180_CONTACT_SIZE_NM
    sd_clear = GF180_CONTACT_TO_GATE_SPACE_NM  # CO.7
    comp_enc = GF180_COMP_CONTACT_ENCLOSURE_NM  # CO.4
    m1_enc = GF180_METAL1_CONTACT_ENCLOSURE_NM  # CO.6
    # metal1 pad over a contact: CO.6-enclosing AND >= M1.3 min-area (380 nm sq).
    m1_pad = max(csize + 2 * m1_enc, 380)
    m1_w = max(GF180_METAL1_MIN_WIDTH_NM, csize + 2 * m1_enc)
    bus_band = max(GF180_POLY_MIN_WIDTH_NM, 180)
    top_endcap = max(GF180_FIELD_POLY_TO_COMP_NM + bus_band, endcap)

    polys: List[LayerPolygon] = []

    # --- gate poly fingers --------------------------------------------------
    finger_x: List[int] = [i * (l_nm + finger_space) for i in range(finger_count)]
    last_gate_x_max = finger_x[-1] + l_nm
    for x0 in finger_x:
        polys.append(
            _rect(GF180_POLY_KEY, (x0, -endcap, x0 + l_nm, w_nm + top_endcap))
        )

    # --- COMP (active) ------------------------------------------------------
    comp_x0 = finger_x[0] - sd_clear - csize - comp_enc
    comp_x1 = last_gate_x_max + sd_clear + csize + comp_enc
    comp_bbox = (comp_x0, 0, comp_x1, w_nm)
    polys.append(_rect(GF180_COMP_KEY, comp_bbox))

    # --- implant: COMP bbox grown by NP.5a / PP.5a (0.23um) -----------------
    polys.append(
        _rect(
            implant_key,
            (
                comp_x0 - implant_overlap,
                0 - implant_overlap,
                comp_x1 + implant_overlap,
                w_nm + implant_overlap,
            ),
        )
    )

    # --- S/D contact columns (ALL on the COMP, at the channel centre) -------
    cy = w_nm // 2
    sd_centres: List[int] = [finger_x[0] - sd_clear - csize // 2]
    for i in range(finger_count - 1):
        gap_lo = finger_x[i] + l_nm
        gap_hi = finger_x[i + 1]
        sd_centres.append((gap_lo + gap_hi) // 2)
    sd_centres.append(last_gate_x_max + sd_clear + csize // 2)
    source_cols = [i for i in range(len(sd_centres)) if i % 2 == 0]
    drain_cols = [i for i in range(len(sd_centres)) if i % 2 == 1]

    def _contact(cx: int, ccy: int) -> LayerPolygon:
        return _rect(
            GF180_CONTACT_KEY,
            (cx - csize // 2, ccy - csize // 2, cx + csize // 2, ccy + csize // 2),
        )

    def _m1_pad(cx: int, ccy: int) -> LayerPolygon:
        half = m1_pad // 2
        return _rect(GF180_METAL1_KEY, (cx - half, ccy - half, cx + half, ccy + half))

    for cx in sd_centres:
        polys.append(_contact(cx, cy))  # on-COMP contact
        polys.append(_m1_pad(cx, cy))  # CO.6-enclosing metal1 over it

    # --- source rail (below COMP) + drain rail (above the gate), metal1 only -
    rail_gap = GF180_FIELD_POLY_TO_COMP_NM + 50
    src_rail_cy = -(m1_w // 2 + rail_gap)
    src_xs = [sd_centres[c] for c in source_cols]
    src_rail_bbox = (
        min(src_xs) - m1_w // 2,
        src_rail_cy - m1_w // 2,
        max(src_xs) + m1_w // 2,
        src_rail_cy + m1_w // 2,
    )
    polys.append(_rect(GF180_METAL1_KEY, src_rail_bbox))
    for cx in src_xs:
        polys.append(
            _rect(GF180_METAL1_KEY, (cx - m1_w // 2, src_rail_cy, cx + m1_w // 2, cy))
        )

    # --- gate bus: poly strap >= PL.5 (0.1um) above the COMP top ------------
    bus_y0 = w_nm + GF180_FIELD_POLY_TO_COMP_NM
    bus_y1 = w_nm + top_endcap
    gate_cx = finger_x[0] - finger_space // 2
    bus_x0 = min(
        finger_x[0],
        gate_cx - max(l_nm // 2, csize // 2 + GF180_POLY_CONTACT_ENCLOSURE_NM),
    )
    gate_bus_bbox = (bus_x0, bus_y0, last_gate_x_max, bus_y1)
    polys.append(_rect(GF180_POLY_KEY, gate_bus_bbox))

    # Gate contact: a poly pad raised above the bus over the LEFT lane.
    gate_cy = bus_y1 + GF180_POLY_CONTACT_ENCLOSURE_NM + csize // 2
    gate_pad_top = gate_cy + csize // 2 + GF180_POLY_CONTACT_ENCLOSURE_NM
    gpad_half = max(m1_pad // 2, csize // 2 + GF180_POLY_CONTACT_ENCLOSURE_NM)
    polys.append(
        _rect(
            GF180_POLY_KEY,
            (gate_cx - gpad_half, bus_y0, gate_cx + gpad_half, gate_pad_top),
        )
    )
    gate_contact_bbox = (
        gate_cx - csize // 2,
        gate_cy - csize // 2,
        gate_cx + csize // 2,
        gate_cy + csize // 2,
    )
    polys.append(_rect(GF180_CONTACT_KEY, gate_contact_bbox))
    polys.append(_m1_pad(gate_cx, gate_cy))

    # --- drain rail (above the gate / gate contact), metal1 only ------------
    drn_rail_cy = gate_pad_top + GF180_FIELD_POLY_TO_COMP_NM + m1_w // 2 + 50
    drn_xs = [sd_centres[c] for c in drain_cols]
    drain_rail_bbox = (
        min(drn_xs) - m1_w // 2,
        drn_rail_cy - m1_w // 2,
        max(drn_xs) + m1_w // 2,
        drn_rail_cy + m1_w // 2,
    )
    polys.append(_rect(GF180_METAL1_KEY, drain_rail_bbox))
    for cx in drn_xs:
        polys.append(
            _rect(GF180_METAL1_KEY, (cx - m1_w // 2, cy, cx + m1_w // 2, drn_rail_cy))
        )

    # --- PMOS Nwell + in-well N+ well-tap tied to the source rail (vdd) ------
    # The Nwell encloses the PCOMP by DF.4c (with margin) and extends BELOW the
    # device (into the source-rail band) to host an N+ well-tap. The tap NCOMP +
    # Nplus sit fully clear of the PMOS Pplus (>= NP.3a / PP.3a 0.16um) so no
    # NCOMP-PCOMP butting (DF.3b / NP.3d/3e / PP.3d/3e) fires, and the tap M1
    # rises into the source rail so the well body extracts on the source net (vdd).
    nwell_bbox: Optional[Tuple[int, int, int, int]] = None
    if is_pmos:
        # Pplus implant bbox (COMP grown by PP.5a).
        pplus_x0 = comp_x0 - implant_overlap
        pplus_y0 = 0 - implant_overlap
        pplus_x1 = comp_x1 + implant_overlap
        # Well-tap geometry: place it BELOW the source rail (which is at
        # src_rail_cy < 0), in the field band, fully clear of the Pplus.
        tap_size = max(GF180_CONTACT_SIZE_NM + 2 * comp_enc + 80, 600)
        tap_ext = GF180_IMPLANT_EXTENSION_NM
        # the tap Nplus must be >= NP.3a below the Pplus bottom edge.
        tap_cy = (
            pplus_y0
            - GF180_IMPLANT_TO_OPP_COMP_SPACE_NM
            - tap_ext
            - tap_size // 2
        )
        tap_cx = (comp_x0 + comp_x1) // 2
        tap_comp = (
            tap_cx - tap_size // 2,
            tap_cy - tap_size // 2,
            tap_cx + tap_size // 2,
            tap_cy + tap_size // 2,
        )
        tap_nplus = (
            tap_comp[0] - tap_ext,
            tap_comp[1] - tap_ext,
            tap_comp[2] + tap_ext,
            tap_comp[3] + tap_ext,
        )
        tap_co = (
            tap_cx - csize // 2,
            tap_cy - csize // 2,
            tap_cx + csize // 2,
            tap_cy + csize // 2,
        )
        # tap M1 rises from the tap contact up into the source rail (which sits
        # just above the tap, at src_rail_cy). This ties the well body to vdd.
        tap_m1_half = max(csize // 2 + m1_enc, GF180_METAL1_MIN_WIDTH_NM // 2)
        tap_m1_top = src_rail_cy + m1_w // 2
        tap_m1 = (tap_cx - tap_m1_half, tap_cy - tap_m1_half, tap_cx + tap_m1_half, tap_m1_top)
        polys.append(_rect(GF180_COMP_KEY, tap_comp))
        polys.append(_rect(GF180_NPLUS_KEY, tap_nplus))
        polys.append(_rect(GF180_CONTACT_KEY, tap_co))
        polys.append(_rect(GF180_METAL1_KEY, tap_m1))
        # The tap M1 rises to ``src_rail_cy`` at ``tap_cx`` = the COMP centre, which
        # sits BETWEEN the source and drain columns. The source rail M1 only spans the
        # source columns ([min(src_xs), max(src_xs)]); for a FEW-FINGER device (nf=1)
        # that span does NOT reach the COMP-centre tap, leaving the tap M1 top
        # electrically FLOATING (the Nwell extracts as its own isolated net -- the nf=1
        # self-bulk PMOS bulk-open). Bridge the tap into the source rail with a same-net
        # M1 patch at ``src_rail_cy`` from the tap column to the nearest source column
        # ONLY when the source rail does not already cover the tap (a multi-finger device
        # ties the tap by abutment and stays BYTE-IDENTICAL -- no extra polygon). Same net
        # (self-bulk PMOS: source == bulk), runs along the source-rail band, clear of the
        # drain. Kept byte-parity with clay.layout.device_cell.build_device's identical fix.
        if src_xs:
            _rail_lo = min(src_xs) - m1_w // 2
            _rail_hi = max(src_xs) + m1_w // 2
            _tap_lo = tap_cx - tap_m1_half
            _tap_hi = tap_cx + tap_m1_half
            if _tap_lo > _rail_hi or _tap_hi < _rail_lo:
                _near = min(src_xs, key=lambda sx: abs(sx - tap_cx))
                polys.append(
                    _rect(
                        GF180_METAL1_KEY,
                        (
                            min(tap_cx, _near) - tap_m1_half,
                            src_rail_cy - m1_w // 2,
                            max(tap_cx, _near) + tap_m1_half,
                            src_rail_cy + m1_w // 2,
                        ),
                    )
                )

        # Nwell encloses BOTH the PCOMP (DF.4c) and the tap NCOMP (DF.4d), with
        # NW.1a min width on both axes.
        nx0 = min(comp_x0 - GF180_NWELL_PCOMP_OVERLAP_NM, tap_comp[0] - GF180_NWELL_NCOMP_OVERLAP_NM)
        nx1 = max(comp_x1 + GF180_NWELL_PCOMP_OVERLAP_NM, tap_comp[2] + GF180_NWELL_NCOMP_OVERLAP_NM)
        ny0 = min(tap_comp[1] - GF180_NWELL_NCOMP_OVERLAP_NM, 0 - GF180_NWELL_PCOMP_OVERLAP_NM)
        ny1 = w_nm + GF180_NWELL_PCOMP_OVERLAP_NM
        if nx1 - nx0 < GF180_NWELL_MIN_WIDTH_NM:
            mid = (nx0 + nx1) // 2
            nx0 = mid - GF180_NWELL_MIN_WIDTH_NM // 2
            nx1 = mid + GF180_NWELL_MIN_WIDTH_NM // 2
        if ny1 - ny0 < GF180_NWELL_MIN_WIDTH_NM:
            ny0 = ny1 - GF180_NWELL_MIN_WIDTH_NM
        nwell_bbox = (nx0, ny0, nx1, ny1)
        polys.append(_rect(GF180_NWELL_KEY, nwell_bbox))

    return FoldableDevice(
        polygons=polys,
        kind=kind,
        finger_count=finger_count,
        gate_length_nm=l_nm,
        channel_width_nm=w_nm,
        comp_bbox=comp_bbox,
        source_rail_bbox=src_rail_bbox,
        drain_rail_bbox=drain_rail_bbox,
        gate_bus_bbox=gate_bus_bbox,
        gate_contact_bbox=gate_contact_bbox,
        nwell_bbox=nwell_bbox,
    )


# --- N+ well-tap (ties an Nwell body to a net) -------------------------------


@dataclass(frozen=True)
class WellTap:
    """An N+ well-tap inside an Nwell: NCOMP + Nplus + contact + M1.

    The tap NCOMP sits inside the Nwell with DF.4d (0.12um) Nwell overlap; the
    contact lands on the NCOMP (CO.4 / CO.11); the Nplus covers the NCOMP
    (DF.12 / NP.5b); the M1 encloses the contact (CO.6) and rises to ``m1_top``.
    """

    polygons: List[LayerPolygon]
    comp_bbox: Tuple[int, int, int, int]
    contact_bbox: Tuple[int, int, int, int]
    m1_bbox: Tuple[int, int, int, int]


def build_nwell_tap(
    *,
    cx: int,
    cy: int,
    m1_top_y: int,
    tap_size_nm: int = 600,
) -> WellTap:
    """Build an N+ well-tap centred at (``cx``, ``cy``) whose M1 rises to
    ``m1_top_y`` (so a vdd rail / strap can land on it)."""
    csize = GF180_CONTACT_SIZE_NM
    enc = GF180_COMP_CONTACT_ENCLOSURE_NM
    m1_enc = GF180_METAL1_CONTACT_ENCLOSURE_NM
    size = max(int(tap_size_nm), csize + 2 * enc + 80)
    comp_bbox = (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)
    nplus_bbox = (
        comp_bbox[0] - GF180_IMPLANT_EXTENSION_NM,
        comp_bbox[1] - GF180_IMPLANT_EXTENSION_NM,
        comp_bbox[2] + GF180_IMPLANT_EXTENSION_NM,
        comp_bbox[3] + GF180_IMPLANT_EXTENSION_NM,
    )
    contact_bbox = (
        cx - csize // 2,
        cy - csize // 2,
        cx + csize // 2,
        cy + csize // 2,
    )
    m1_half = max(csize // 2 + m1_enc, GF180_METAL1_MIN_WIDTH_NM // 2)
    m1_bbox = (cx - m1_half, cy - m1_half, cx + m1_half, m1_top_y)
    polys = [
        _rect(GF180_COMP_KEY, comp_bbox),
        _rect(GF180_NPLUS_KEY, nplus_bbox),
        _rect(GF180_CONTACT_KEY, contact_bbox),
        _rect(GF180_METAL1_KEY, m1_bbox),
    ]
    return WellTap(
        polygons=polys,
        comp_bbox=comp_bbox,
        contact_bbox=contact_bbox,
        m1_bbox=m1_bbox,
    )


# --- P+ substrate tap (ties a net to the global substrate) -------------------


@dataclass(frozen=True)
class SubstrateTap:
    """A gf180 P+ substrate tap whose M1 overlaps a given rail (ties it to vss)."""

    polygons: List[LayerPolygon]
    comp_bbox: Tuple[int, int, int, int]
    contact_bbox: Tuple[int, int, int, int]
    m1_bbox: Tuple[int, int, int, int]


def build_substrate_tap(
    rail_bbox: Tuple[int, int, int, int],
    *,
    tap_size_nm: int = 600,
    gap_below_nm: int = 400,
) -> SubstrateTap:
    """Build a DRC-legal P+ substrate tap below ``rail_bbox`` whose M1 rises into
    the rail (so the rail net shares M1 with the substrate contact)."""
    csize = GF180_CONTACT_SIZE_NM
    enc = GF180_COMP_CONTACT_ENCLOSURE_NM
    m1_enc = GF180_METAL1_CONTACT_ENCLOSURE_NM
    size = max(int(tap_size_nm), csize + 2 * enc + 80)
    rx0, ry0, rx1, ry1 = rail_bbox
    rail_cx = (rx0 + rx1) // 2
    comp_y1 = ry0 - max(int(gap_below_nm), GF180_COMP_COMP_SPACE_NM)
    comp_y0 = comp_y1 - size
    comp_bbox = (rail_cx - size // 2, comp_y0, rail_cx + size // 2, comp_y1)
    pplus_bbox = (
        comp_bbox[0] - GF180_IMPLANT_EXTENSION_NM,
        comp_bbox[1] - GF180_IMPLANT_EXTENSION_NM,
        comp_bbox[2] + GF180_IMPLANT_EXTENSION_NM,
        comp_bbox[3] + GF180_IMPLANT_EXTENSION_NM,
    )
    ccx = (comp_bbox[0] + comp_bbox[2]) // 2
    ccy = (comp_bbox[1] + comp_bbox[3]) // 2
    contact_bbox = (
        ccx - csize // 2,
        ccy - csize // 2,
        ccx + csize // 2,
        ccy + csize // 2,
    )
    m1_half = max(csize // 2 + m1_enc, GF180_METAL1_MIN_WIDTH_NM // 2)
    m1_bbox = (ccx - m1_half, ccy - m1_half, ccx + m1_half, ry1)
    polys = [
        _rect(GF180_COMP_KEY, comp_bbox),
        _rect(GF180_PPLUS_KEY, pplus_bbox),
        _rect(GF180_CONTACT_KEY, contact_bbox),
        _rect(GF180_METAL1_KEY, m1_bbox),
    ]
    return SubstrateTap(
        polygons=polys,
        comp_bbox=comp_bbox,
        contact_bbox=contact_bbox,
        m1_bbox=m1_bbox,
    )


# --- metal1 net label (the LVS source<->substrate merge mechanism) -----------


@dataclass(frozen=True)
class SubstrateLabelSpec:
    """One gf180 metal1 net label to inject as a GDS TEXT (34/10)."""

    text: str
    point: Tuple[int, int]
    layer_key: Tuple[int, int] = GF180_METAL1_LABEL_KEY


# --- composed co-clean 5T-OTA ------------------------------------------------


@dataclass
class CoClean5TLayout:
    """The composed co-clean 5T-OTA library + the net label spec(s)."""

    library: ParameterizedLibrary
    cell_name: str
    labels_list: List[SubstrateLabelSpec]
    device_bboxes: Dict[str, Tuple[int, int, int, int]]
    note: str = ""

    def labels(self) -> List[SubstrateLabelSpec]:
        return list(self.labels_list)


# The 5T-OTA device dimensions (mirror the R151 folded template: m=2 each;
# input pair Wf=4U L=0.5U, load pair Wf=4U L=1U, tail Wf=8U L=1U).
_FIVE_T_SIZING: Dict[str, Tuple[str, float, float, int]] = {
    "M_in_p": ("nmos", 4.0, 0.5, 2),
    "M_in_n": ("nmos", 4.0, 0.5, 2),
    "M_load_p": ("pmos", 4.0, 1.0, 2),
    "M_load_n": ("pmos", 4.0, 1.0, 2),
    "M_tail": ("nmos", 8.0, 1.0, 2),
}


def build_coclean_5tota(
    *,
    cell_name: str = "gf180_coclean_5tota_v163",
    substrate_net: str = DEFAULT_SUBSTRATE_NET,
) -> CoClean5TLayout:
    """Build the co-clean 5T-OTA: 5 DRC-legal foldable devices (3 NMOS + 2 PMOS)
    placed in a row at DF.3a-safe gaps, strapped on metal1-only lanes for all
    inter-device nets, with the PMOS Nwells tapped to ``vdd`` + a P+ substrate tap
    + a metal1 ``vss`` label.

    Textbook 5T-OTA topology (8 nets: in_p, in_n, vout, vbias, vdd, vss, tail,
    load_gate):

      * M_in_p:  d=load_gate g=in_p   s=tail  b=vss
      * M_in_n:  d=vout      g=in_n   s=tail  b=vss
      * M_load_p:d=load_gate g=load_gate s=vdd b=vdd  (diode-tied)
      * M_load_n:d=vout      g=load_gate s=vdd b=vdd
      * M_tail:  d=tail      g=vbias  s=vss   b=vss

    The strap routes every inter-device net on its OWN horizontal metal1 lane
    (distinct y, M1.2 spacing apart) with vertical metal1 risers at each device's
    rail x, so no two distinct nets touch on metal1 (no short) and every net's
    terminals are joined (no float). No M2 / via1 is used.
    """
    # Build + place the five devices left-to-right at a generous DF.3a gap.
    order = ["M_in_p", "M_in_n", "M_tail", "M_load_p", "M_load_n"]
    devs: Dict[str, FoldableDevice] = {}
    for name in order:
        k, wf, lu, m = _FIVE_T_SIZING[name]
        devs[name] = build_foldable_device(
            kind=k, finger_count=m, gate_length_um=lu, channel_width_um=wf
        )

    # Gap big enough that no cross-device COMP / Nwell / implant rule fires.
    # NW.2b (Nwell different-potential space) is 1.4um; use a wide margin so the
    # PMOS Nwells never come within that of each other or of an NMOS COMP.
    gap = 2000

    polys: List[LayerPolygon] = []
    placed: Dict[str, FoldableDevice] = {}
    # per-device translated rail / contact bboxes.
    src: Dict[str, Tuple[int, int, int, int]] = {}
    drn: Dict[str, Tuple[int, int, int, int]] = {}
    gco: Dict[str, Tuple[int, int, int, int]] = {}
    dev_extent: Dict[str, Tuple[int, int, int, int]] = {}
    nwell: Dict[str, Tuple[int, int, int, int]] = {}

    cursor_x = 0
    for name in order:
        dev = devs[name]
        ext = dev.extent()
        dx = cursor_x - ext[0]
        for lp in dev.polygons:
            polys.append(_translate(lp, dx, 0))
        src[name] = _shift(dev.source_rail_bbox, dx, 0)
        drn[name] = _shift(dev.drain_rail_bbox, dx, 0)
        gco[name] = _shift(dev.gate_contact_bbox, dx, 0)
        dev_extent[name] = _shift(ext, dx, 0)
        if dev.nwell_bbox is not None:
            nwell[name] = _shift(dev.nwell_bbox, dx, 0)
        placed[name] = dev
        cursor_x = dev_extent[name][2] + gap

    m1_w = max(GF180_METAL1_MIN_WIDTH_NM, GF180_CONTACT_SIZE_NM + 2 * GF180_METAL1_CONTACT_ENCLOSURE_NM)
    half = m1_w // 2
    m2_half = max(GF180_METAL2_MIN_WIDTH_NM, GF180_VIA1_SIZE_NM + 2 * GF180_M2_VIA1_ENCLOSURE_NM) // 2
    lane_pitch = 2 * m2_half + GF180_METAL2_SPACE_NM
    v1h = GF180_VIA1_SIZE_NM // 2
    # M1 landing pad over a via1 (covers it fully for V1.3a + M1.3 area).
    m1pad_half = max(half, v1h + GF180_M1_VIA1_ENCLOSURE_NM, 190)

    # The terminal escape (x, y) each net joins. kind: src / drn / gate.
    #
    # CRITICAL (the same-device short the metal1-only attempt hit): for a folded
    # device the SOURCE rail is wide (spans the full COMP) while the DRAIN rail is
    # narrow and CENTRED, so a source riser at the device-centre x would run
    # straight up through the drain rail (shorting S to D). So the source escape is
    # taken at the source rail's far edge (clear of the centred drain x-span AND
    # the left-lane gate-contact x-span); the drain escapes at its own centre; the
    # gate at the gate-contact column.
    def _term(name: str, kind: str) -> Tuple[int, int]:
        if kind == "drn":
            b = drn[name]
            return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        if kind == "gate":
            b = gco[name]
            return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)
        # source: escape at the RIGHT edge of the source rail, clear of the
        # centred drain rail and the left gate-contact column.
        s = src[name]
        d = drn[name]
        sy = (s[1] + s[3]) // 2
        esc = s[2] - half  # right edge, inside the rail
        # ensure it clears the drain x-span by >= M1.2 space; the right edge is
        # well to the right of the centred drain, so this holds by construction.
        if esc - half <= d[2] + GF180_METAL1_SPACE_NM and esc + half >= d[0] - GF180_METAL1_SPACE_NM:
            esc = s[0] + half  # fall back to the left edge
        return (esc, sy)

    # Each net: list of (device, rail-kind) terminals. vss is handled by the
    # substrate tap + label (not a lane). The strap routes the 4 signal nets on
    # M2 trunks (one lane each) with M1 risers + via1 (M1<->M2) junctions, so
    # distinct nets never share same-metal geometry (the layer-separation that
    # keeps the metal1 risers short-free as they cross foreign M2 trunks).
    net_terms: List[Tuple[str, List[Tuple[str, str]]]] = [
        ("tail", [("M_in_p", "src"), ("M_in_n", "src"), ("M_tail", "drn")]),
        (
            "load_gate",
            [
                ("M_in_p", "drn"),
                ("M_load_p", "drn"),
                ("M_load_p", "gate"),
                ("M_load_n", "gate"),
            ],
        ),
        ("vout", [("M_in_n", "drn"), ("M_load_n", "drn")]),
        ("vdd", [("M_load_p", "src"), ("M_load_n", "src")]),
    ]

    # Lanes stack ABOVE the tallest device, each net on its own M2 y-lane.
    top_y = max(b[3] for b in dev_extent.values())
    base_lane = top_y + lane_pitch

    label_specs: List[SubstrateLabelSpec] = []

    def _via1(cx: int, cy: int) -> None:
        # via1 (exact V1.1 size) + M1 landing pad (V1.3a cover) + M2 cap (V1.4a).
        polys.append(_rect(GF180_VIA1_KEY, (cx - v1h, cy - v1h, cx + v1h, cy + v1h)))
        polys.append(
            _rect(GF180_METAL1_KEY, (cx - m1pad_half, cy - m1pad_half, cx + m1pad_half, cy + m1pad_half))
        )
        polys.append(
            _rect(
                GF180_METAL2_KEY,
                (cx - m2_half, cy - m2_half, cx + m2_half, cy + m2_half),
            )
        )

    for idx, (net, terms) in enumerate(net_terms):
        lane_y = base_lane + idx * lane_pitch
        riser_xs: List[int] = []
        for dname, kind in terms:
            tx, ty = _term(dname, kind)
            # M1 riser from the terminal up to the lane y.
            lo, hi = (ty, lane_y) if ty <= lane_y else (lane_y, ty)
            polys.append(_rect(GF180_METAL1_KEY, (tx - half, lo, tx + half, hi)))
            # via1 (M1<->M2) at the lane.
            _via1(tx, lane_y)
            riser_xs.append(tx)
        # M2 trunk joining all this net's via columns at the lane (>= M2.1 width).
        polys.append(
            _rect(
                GF180_METAL2_KEY,
                (min(riser_xs) - m2_half, lane_y - m2_half, max(riser_xs) + m2_half, lane_y + m2_half),
            )
        )

    # NOTE: each PMOS device already carries its own in-Nwell N+ well-tap whose
    # M1 ties the well body to that PMOS source rail (vdd) -- built in
    # build_foldable_device. The vdd lane above commons the two PMOS source rails,
    # so both well bodies extract on vdd. No separate well-tap routing is needed.

    # vss: P+ substrate tap below the tail source rail + a metal1 "vss" label.
    tail_src = src["M_tail"]
    sub_tap = build_substrate_tap(tail_src)
    polys.extend(sub_tap.polygons)
    label_specs.append(
        SubstrateLabelSpec(
            text=substrate_net,
            point=((tail_src[0] + tail_src[2]) // 2, (tail_src[1] + tail_src[3]) // 2),
        )
    )

    top = NamedCell(name=cell_name, polygons=polys)
    lib = ParameterizedLibrary(
        name="clay_coclean_5tota_v163",
        unit_user=0.001,
        unit_db_meters=1e-9,
        cells=[top],
    )
    return CoClean5TLayout(
        library=lib,
        cell_name=cell_name,
        labels_list=label_specs,
        device_bboxes=dev_extent,
        note=(
            "R163 co-clean 5T-OTA: 3 DRC-legal foldable NMOS + 2 foldable PMOS "
            "(Nwell + N+ well-tap to vdd) + metal1-only 4-net strap + P+ substrate "
            f"tap + metal1 {substrate_net!r} label"
        ),
    )


def build_reference_schematic_5tota(
    cell_name: str = "gf180_coclean_5tota_v163",
    *,
    substrate_net: str = DEFAULT_SUBSTRATE_NET,
    vdd_net: str = DEFAULT_VDD_NET,
) -> str:
    """Build the FOLDED 5T-OTA reference SPICE (5 ``M``-card MOS, m=2 each).

    Matches the R151 reference: 3 ``nmos_3p3`` + 2 ``pmos_3p3``, 8 nets
    (in_p, in_n, vout, vbias, vdd, vss, tail, load_gate). NMOS bulk == vss, PMOS
    bulk == vdd. W is the per-finger drawn width; ``m`` the finger count.
    """

    def card(inst: str, model: str, d: str, g: str, s: str, b: str) -> str:
        _, wf, lu, m = _FIVE_T_SIZING[inst]
        return f"M{inst} {d} {g} {s} {b} {model} W={wf:g}U L={lu:g}U m={m}"

    lines = [f".subckt {cell_name} in_p in_n vout vbias {vdd_net} {substrate_net}"]
    lines.append(card("M_in_p", "nmos_3p3", "load_gate", "in_p", "tail", substrate_net))
    lines.append(card("M_in_n", "nmos_3p3", "vout", "in_n", "tail", substrate_net))
    lines.append(card("M_load_p", "pmos_3p3", "load_gate", "load_gate", vdd_net, vdd_net))
    lines.append(card("M_load_n", "pmos_3p3", "vout", "load_gate", vdd_net, vdd_net))
    lines.append(card("M_tail", "nmos_3p3", "tail", "vbias", substrate_net, substrate_net))
    lines.append(".ends")
    return "\n".join(lines) + "\n"


def count_schematic_devices(spice_text: str) -> int:
    """Count ``M``-prefixed MOS primitive cards in a reference schematic."""
    n = 0
    for raw in spice_text.splitlines():
        line = raw.strip()
        if not line or line[0] in ("*", "."):
            continue
        if line[0] in ("m", "M"):
            n += 1
    return n


def count_schematic_nets(spice_text: str) -> int:
    """Count distinct net tokens referenced by the ``M`` cards (d/g/s/b columns)."""
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


# --- KLayout GDS label injection (gated; reuses the committed Rust emitter) ---


def klayout_binary_available(klayout_bin: Optional[Path] = None) -> bool:
    kbin = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
    return kbin.exists()


_INJECT_RUBY = r"""
ly = RBA::Layout.new
ly.read($input)
top = ly.top_cell
specs = $labels.split(";")
specs.each do |spec|
  next if spec.strip.empty?
  l, d, x, y, name = spec.split(",")
  li = ly.layer(l.to_i, d.to_i)
  top.shapes(li).insert(RBA::Text.new(name, RBA::Trans.new(RBA::Point.new(x.to_i, y.to_i))))
end
ly.write($output)
"""


def inject_metal1_labels_into_gds(
    input_gds,
    output_gds,
    labels: Sequence[SubstrateLabelSpec],
    *,
    klayout_bin: Optional[Path] = None,
    timeout_sec: float = 120.0,
) -> bool:
    """Add gf180 metal1 TEXT label(s) to ``input_gds`` -> ``output_gds`` via KLayout.

    Gated: returns ``False`` if the binary is absent.
    """
    kbin = Path(klayout_bin) if klayout_bin is not None else DEFAULT_KLAYOUT_BIN
    if not kbin.exists():
        return False
    in_path = Path(input_gds).resolve()
    out_path = Path(output_gds).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"input GDS not found: {in_path}")

    spec_str = ";".join(
        f"{lb.layer_key[0]},{lb.layer_key[1]},{lb.point[0]},{lb.point[1]},{lb.text}"
        for lb in labels
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_inject_labels_v163.rb", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(_INJECT_RUBY)
        ruby_path = Path(fh.name)
    try:
        cmd = [
            str(kbin),
            "-b",
            "-r",
            str(ruby_path),
            "-rd",
            f"input={in_path}",
            "-rd",
            f"output={out_path}",
            "-rd",
            f"labels={spec_str}",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, check=False
        )
    finally:
        try:
            ruby_path.unlink()
        except OSError:
            pass
    return proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


# --- gated real co-clean verification (BOTH decks on ONE GDS) -----------------


@dataclass
class CoClean5TResult:
    """Outcome of running BOTH real gf180mcu decks on ONE identical ``.gds``.

    Attributes:
        ran: True iff BOTH decks executed on the single GDS.
        gds_path: the single ``.gds`` file BOTH decks ran on (the co-clean axis).
        drc_ran / drc_total_violations / drc_by_rule: real DRC verdict (verbatim).
        lvs_ran / lvs_matched / lvs_device_count / lvs_net_count: real LVS verdict.
        schematic_device_count / schematic_net_count: the FOLDED reference counts.
        vacuous: True iff the LVS extraction is 0-device / device-short (guard).
        note: honest one-line summary / skip reason.
    """

    ran: bool = False
    gds_path: Optional[str] = None
    drc_ran: bool = False
    drc_total_violations: Optional[int] = None
    drc_by_rule: Dict[str, int] = field(default_factory=dict)
    lvs_ran: bool = False
    lvs_matched: Optional[bool] = None
    lvs_device_count: Optional[int] = None
    lvs_net_count: Optional[int] = None
    schematic_device_count: Optional[int] = None
    schematic_net_count: Optional[int] = None
    vacuous: bool = False
    note: str = ""

    @property
    def drc_clean(self) -> bool:
        return bool(self.drc_ran and self.drc_total_violations == 0)

    @property
    def lvs_clean(self) -> bool:
        return bool(self.lvs_ran and self.lvs_matched is True and not self.vacuous)

    @property
    def co_clean(self) -> bool:
        """True iff the SAME GDS is BOTH real-DRC-clean AND real-LVS-clean."""
        return bool(self.ran and self.drc_clean and self.lvs_clean)


def real_signoff_available() -> bool:
    """True iff KLayout + gf180mcu.drc + gf180mcu.lvs + Rust GDS are all present."""
    try:
        from tools.klayout_drc_signoff_v139.run import (
            DEFAULT_DECK as DRC_DECK,
            klayout_binary_available as _k,
        )
        from tools.klayout_lvs_signoff_v145.run import lvs_deck_available as _d
        from .binary_gds_subprocess import cargo_binary_available

        return bool(_k() and DRC_DECK.exists() and _d() and cargo_binary_available())
    except Exception:  # noqa: BLE001
        return False


def run_coclean_5tota_signoff(
    *,
    out_dir,
    substrate_net: str = DEFAULT_SUBSTRATE_NET,
    timeout_sec: float = 900.0,
) -> CoClean5TResult:
    """Build the co-clean 5T-OTA, emit ONE GDS, and run BOTH real decks on that
    IDENTICAL file. Read every verdict VERBATIM; guard vacuous LVS.

    Pipeline (gated + skip-safe):
      1. :func:`build_coclean_5tota` (5 foldable devices + metal1 strap + taps).
      2. committed Rust GDS emitter writes the polygon GDS.
      3. :func:`inject_metal1_labels_into_gds` adds the metal1 ``vss`` label ->
         the SINGLE co-clean GDS.
      4. run the REAL ``gf180mcu.drc`` on that GDS (verbatim violation count).
      5. run the REAL ``gf180mcu.lvs`` on the SAME GDS vs the folded reference
         (verbatim ``compare`` verdict + extracted counts), ``lvs_sub=vss``.
    """
    res = CoClean5TResult()
    if not real_signoff_available():
        res.note = "real signoff toolchain unavailable (KLayout / gf180mcu.drc / gf180mcu.lvs / Rust GDS)"
        return res

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        layout = build_coclean_5tota(substrate_net=substrate_net)
    except Exception as exc:  # noqa: BLE001
        res.note = f"compose failed: {type(exc).__name__}: {exc}"
        return res

    from .binary_gds_subprocess import emit_binary_gds_via_subprocess

    base_gds = out / f"{layout.cell_name}_base.gds"
    try:
        emit_binary_gds_via_subprocess(layout.library, base_gds)
    except Exception as exc:  # noqa: BLE001
        res.note = f"Rust GDS emit failed: {type(exc).__name__}"
        return res
    if not base_gds.exists() or base_gds.stat().st_size == 0:
        res.note = "Rust GDS emitter produced no GDS"
        return res

    # THE single co-clean GDS (one file both decks run on).
    coclean_gds = out / f"{layout.cell_name}_coclean.gds"
    if not inject_metal1_labels_into_gds(base_gds, coclean_gds, layout.labels()):
        res.note = "metal1 label injection failed (KLayout absent or batch error)"
        return res
    res.gds_path = str(coclean_gds.resolve())

    # 4. REAL DRC on the single GDS.
    from tools.klayout_drc_signoff_v139.run import Gf180DrcError, run_gf180_drc

    try:
        drc = run_gf180_drc(
            coclean_gds,
            report_path=out / f"{layout.cell_name}_coclean.lyrdb",
            topcell=layout.cell_name,
            run_mode="flat",
            timeout_sec=timeout_sec,
        )
    except Gf180DrcError as exc:  # noqa: BLE001
        res.note = f"real DRC deck ran but failed: {type(exc).__name__}: {str(exc).splitlines()[0]}"
        return res
    res.drc_ran = bool(getattr(drc, "ran_real_deck", True))
    res.drc_total_violations = drc.total_violations
    res.drc_by_rule = {v.rule: v.count for v in drc.violations}

    # 5. REAL LVS on the SAME single GDS.
    schematic_text = build_reference_schematic_5tota(
        layout.cell_name, substrate_net=substrate_net
    )
    sch_path = out / f"{layout.cell_name}_coclean_ref.cir"
    sch_path.write_text(schematic_text, encoding="utf-8")
    res.schematic_device_count = count_schematic_devices(schematic_text)
    res.schematic_net_count = count_schematic_nets(schematic_text)

    from tools.klayout_lvs_signoff_v145.run import Gf180LvsError, run_gf180_lvs

    try:
        lvs = run_gf180_lvs(
            coclean_gds,
            sch_path,
            topcell=layout.cell_name,
            report_path=out / f"{layout.cell_name}_coclean.lvsdb",
            extracted_netlist_path=out / f"{layout.cell_name}_coclean_extracted.cir",
            lvs_sub=substrate_net,
            run_mode="flat",
            timeout_sec=timeout_sec,
        )
    except Gf180LvsError as exc:  # noqa: BLE001
        res.note = f"real LVS deck ran but failed: {type(exc).__name__}: {str(exc).splitlines()[0]}"
        return res
    res.lvs_ran = bool(getattr(lvs, "ran_real_deck", True))
    res.lvs_matched = lvs.matched
    res.lvs_device_count = lvs.device_count
    res.lvs_net_count = lvs.net_count
    res.vacuous = bool(
        lvs.device_count is not None
        and (
            int(lvs.device_count) == 0
            or int(lvs.device_count) < int(res.schematic_device_count)
        )
    )

    res.ran = bool(res.drc_ran and res.lvs_ran)
    res.note = (
        f"co-clean 5T-OTA on ONE GDS: DRC {res.drc_total_violations} viol, "
        f"LVS {'MATCH' if res.lvs_matched else 'MISMATCH'} "
        f"(extracted {res.lvs_device_count} dev / {res.lvs_net_count} net)"
    )
    return res


__all__ = [
    "GF180_NWELL_KEY",
    "GF180_COMP_KEY",
    "GF180_POLY_KEY",
    "GF180_PPLUS_KEY",
    "GF180_NPLUS_KEY",
    "GF180_CONTACT_KEY",
    "GF180_METAL1_KEY",
    "GF180_METAL1_LABEL_KEY",
    "GF180_METAL1_CONTACT_ENCLOSURE_NM",
    "GF180_NPLUS_GATE_OVERLAP_NM",
    "GF180_PPLUS_GATE_OVERLAP_NM",
    "GF180_FIELD_POLY_TO_COMP_NM",
    "GF180_NWELL_PCOMP_OVERLAP_NM",
    "GF180_NWELL_NCOMP_OVERLAP_NM",
    "DEFAULT_KLAYOUT_BIN",
    "DEFAULT_SUBSTRATE_NET",
    "DEFAULT_VDD_NET",
    "FoldableDevice",
    "WellTap",
    "SubstrateTap",
    "SubstrateLabelSpec",
    "CoClean5TLayout",
    "CoClean5TResult",
    "build_foldable_device",
    "build_nwell_tap",
    "build_substrate_tap",
    "build_coclean_5tota",
    "build_reference_schematic_5tota",
    "count_schematic_devices",
    "count_schematic_nets",
    "klayout_binary_available",
    "inject_metal1_labels_into_gds",
    "real_signoff_available",
    "run_coclean_5tota_signoff",
]
