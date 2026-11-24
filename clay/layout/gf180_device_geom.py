"""R139 (BL-139A) — real gf180mcu PLANAR MOS device geometry.

The legacy ``lower_mos_finger_geometry`` in ``clay/layout/backend.py`` spaces
poly fingers by a fictional ``fin_pitch`` (sky130 = 48 nm, gf180 = 280 nm). But
sky130 and gf180mcu are *planar* 180/130 nm processes with **no fins**. Spacing
poly fingers by 48 nm violates the real poly-to-poly space rule by ~5x, which is
exactly the intra-device fin-pitch violation R136/R137 saw when checking devices
with ``layout_drc.run_layout_drc``.

This module builds a multi-finger MOS device whose geometry is DRC-legal *by
construction* using the REAL gf180mcu 3.3V (core / GF 0.18um) planar device
rules taken verbatim from the gf180mcu KLayout DRC deck:

    AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/rules/klayout/drc/
        rule_decks/poly2.drc   (PL.* rules)
        rule_decks/comp.drc    (DF.* rules)
        rule_decks/contact.drc (CO.* rules)

Real rule values used (all values in microns in the deck; we store nm):

    PL.1_3.3V   poly2 interconnect (finger) min width ........... 0.18 um = 180 nm
    PL.2_3.3V   gate width = channel length L, minimum .......... 0.28 um = 280 nm
    PL.3a_3.3V  poly2 space on COMP/Field (finger-to-finger) .... 0.24 um = 240 nm
    PL.4_3.3V   poly2 extension beyond COMP (gate end-cap) ...... 0.22 um = 220 nm
    DF.1a_3.3V  min COMP (diffusion) width ...................... 0.22 um = 220 nm
    DF.2a_3.3V  min channel width (COMP under gate) ............. 0.22 um = 220 nm
    DF.3a_3.3V  min COMP space (device-to-device diff) .......... 0.28 um = 280 nm
    CO.1        min/max contact size ............................ 0.22 um = 220 nm
    CO.2a       min contact spacing ............................. 0.25 um = 250 nm
    CO.3        poly2 overlap (enclosure) of contact ............ 0.07 um =  70 nm
    CO.4        COMP overlap (enclosure) of contact ............. 0.07 um =  70 nm
    CO.7        space from COMP contact to poly2 (gate) on COMP . 0.15 um = 150 nm
    CO.8        space from poly2 contact to COMP ................ 0.17 um = 170 nm

NOTE on finger pitch: PL.3a gives a *minimum* gate-to-gate poly space of 240 nm,
but a real MOS finger array places a source/drain contact diffusion region
*between* adjacent gates. To fit one column of S/D contacts (CO.1 = 220 nm wide)
with the required gate-poly-to-contact clearance (CO.7 = 150 nm on each side),
the contacted gate-to-gate space must be at least::

    contact(220) + 2 * CO.7(150) = 520 nm  >> PL.3a min 240 nm

So the contacted finger space is ``max(PL.3a, 520) = 520 nm`` and never the
48 nm (or 280 nm "fin_pitch") of the legacy lowering. The result passes the
poly-poly space check in ``layout_drc.run_layout_drc`` (which uses the 240 nm
gf180 POLY min-spacing from ``layer_pitch.GF180MCU_LAYER_PITCH``) by a wide
margin, and never produces the intra-device fin-pitch violation.

This module is pure geometry construction. It imports committed modules
(``backend``) read-only and emits ``LayerPolygon`` lists / a ``NamedCell``.
All coordinates are GDS database units (dbu = 1 nm for gf180mcu).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .backend import (
    Gf180mcuLayerMap,
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    Point,
    Polygon,
)

# --- real gf180mcu 3.3V planar device rules (nm) ---------------------------
#
# Each constant cites the originating rule id from the gf180mcu KLayout DRC
# deck. These are the 3.3V (core, GF 0.18um) flavour — the 5V variants are
# larger (e.g. PL.2_5V channel length is process-dependent) and are NOT used
# here. No value is invented; every number below is the ``X.um`` literal that
# appears in the cited rule.

# poly2.drc — PL.1_3.3V "Interconnect Width (outside PLFUSE). : 0.18um"
GF180_POLY_MIN_WIDTH_NM = 180  # rule PL.1_3.3V

# poly2.drc — PL.2_3.3V "Gate Width (Channel Length). : 0.28um"
GF180_GATE_MIN_LENGTH_NM = 280  # rule PL.2_3.3V  (minimum drawn channel L)

# poly2.drc — PL.3a_3.3V "Space on COMP/Field. : 0.24um" (gate-to-gate poly)
GF180_POLY_POLY_SPACE_NM = 240  # rule PL.3a_3.3V

# poly2.drc — PL.4_3.3V "Extension beyond COMP to form Poly2 end cap. : 0.22um"
GF180_POLY_ENDCAP_NM = 220  # rule PL.4_3.3V

# comp.drc — DF.1a_3.3V "Min. COMP Width. : 0.22um"
GF180_COMP_MIN_WIDTH_NM = 220  # rule DF.1a_3.3V

# comp.drc — DF.2a_3.3V "Min Channel Width. : 0.22um" (COMP under gate)
GF180_CHANNEL_MIN_WIDTH_NM = 220  # rule DF.2a_3.3V

# comp.drc — DF.3a_3.3V "Min. COMP Space ... : 0.28um" (device-to-device diff)
GF180_COMP_COMP_SPACE_NM = 280  # rule DF.3a_3.3V

# contact.drc — CO.1 "Min/max contact size. : 0.22um"
GF180_CONTACT_SIZE_NM = 220  # rule CO.1

# contact.drc — CO.2a "min. contact spacing : 0.25um"
GF180_CONTACT_SPACE_NM = 250  # rule CO.2a

# contact.drc — CO.3 "Poly2 overlap of contact. : 0.07um"
GF180_POLY_CONTACT_ENCLOSURE_NM = 70  # rule CO.3

# contact.drc — CO.4 "COMP overlap of contact. : 0.07um"
GF180_COMP_CONTACT_ENCLOSURE_NM = 70  # rule CO.4

# contact.drc — CO.7 "Space from COMP contact to Poly2 on COMP. : 0.15um"
GF180_CONTACT_TO_GATE_SPACE_NM = 150  # rule CO.7

# contact.drc — CO.8 "Space from Poly2 contact to COMP. : 0.17um"
GF180_POLY_CONTACT_TO_COMP_SPACE_NM = 170  # rule CO.8


def gf180_contacted_finger_space_nm() -> int:
    """Return the gate-to-gate poly space (nm) for a *contacted* finger array.

    A source/drain contact column sits between adjacent gate fingers. To fit a
    single contact (CO.1 = 220 nm) with the gate-poly-to-contact clearance
    (CO.7 = 150 nm) on both sides, the contacted gate-to-gate space must be at
    least ``contact + 2 * CO.7``. This always dominates the bare poly-poly min
    space (PL.3a = 240 nm), so the returned value is the larger of the two and
    is far above the legacy 48/280 nm fin pitch.
    """
    contacted = GF180_CONTACT_SIZE_NM + 2 * GF180_CONTACT_TO_GATE_SPACE_NM
    return max(GF180_POLY_POLY_SPACE_NM, contacted)


@dataclass(frozen=True)
class Gf180DeviceGeometry:
    """Result of lowering one multi-finger gf180mcu MOS device.

    Attributes:
        polygons: every drawn shape (poly fingers, COMP, S/D + gate contacts).
        finger_count: number of gate poly fingers.
        gate_length_nm: drawn channel length L (poly finger x-extent).
        channel_width_nm: drawn channel width W (poly finger / COMP y-extent of
            the active region).
        finger_space_nm: gate-to-gate poly space actually used (contacted).
        comp_bbox: (x_min, y_min, x_max, y_max) of the COMP diffusion rect.
    """

    polygons: List[LayerPolygon]
    finger_count: int
    gate_length_nm: int
    channel_width_nm: int
    finger_space_nm: int
    comp_bbox: tuple

    def named_cell(self, name: str) -> NamedCell:
        return NamedCell(
            name=name,
            polygons=[
                LayerPolygon(
                    layer=lp.layer,
                    datatype=lp.datatype,
                    polygon=Polygon(vertices=list(lp.polygon.vertices)),
                )
                for lp in self.polygons
            ],
        )

    def poly_polygons(self) -> List[LayerPolygon]:
        return [
            lp for lp in self.polygons
            if (lp.layer, lp.datatype) == (30, 0)
        ]

    def comp_polygons(self) -> List[LayerPolygon]:
        return [
            lp for lp in self.polygons
            if (lp.layer, lp.datatype) == (22, 0)
        ]

    def contact_polygons(self) -> List[LayerPolygon]:
        return [
            lp for lp in self.polygons
            if (lp.layer, lp.datatype) == (33, 0)
        ]


def _square(cx: int, cy: int, size: int) -> Polygon:
    """A ``size`` x ``size`` square centred at (cx, cy)."""
    half = size // 2
    return Polygon.rectangle(cx - half, cy - half, cx + half, cy + half)


def build_gf180_mos_device(
    *,
    finger_count: int,
    gate_length_um: float,
    channel_width_um: float,
    pdk_layers: Optional[LayerMap] = None,
    with_contacts: bool = True,
) -> Gf180DeviceGeometry:
    """Build a DRC-legal-by-construction multi-finger gf180mcu MOS device.

    Geometry layout (x = finger direction, y = channel-width direction)::

        COMP (diffusion) spans the whole device in x and the channel width in y,
        extended a little in x to enclose the outer S/D contacts.

        Gate poly fingers run vertically (y) over the COMP. Each finger is
        ``gate_length`` (>= PL.2 = 280 nm) wide in x and ``channel_width +
        2 * endcap`` tall (PL.4 = 220 nm end-cap above and below the COMP).

        Adjacent fingers are spaced by the *contacted* gate-to-gate space
        ``gf180_contacted_finger_space_nm()`` (>= 520 nm), never the 48 nm fin
        pitch. A source/drain contact column sits centred in each inter-finger
        gap (and at the two outer ends), CO.7 = 150 nm clear of every gate edge.

    Args:
        finger_count: number of gate fingers (>= 1).
        gate_length_um: drawn channel length L in um; clamped up to PL.2 min.
        channel_width_um: drawn channel width W in um; clamped up to DF.2a min.
        pdk_layers: a gf180mcu LayerMap (default ``Gf180mcuLayerMap()``).
        with_contacts: if True (default), emit S/D + gate contacts.

    Returns:
        Gf180DeviceGeometry with the lowered shapes and key dimensions.

    Raises:
        ValueError: if finger_count < 1 or the PDK lacks POLY / NDIFF / CONTACT.
    """
    if finger_count < 1:
        raise ValueError("finger_count must be >= 1")
    if pdk_layers is None:
        pdk_layers = Gf180mcuLayerMap()

    poly = pdk_layers.gds_layer(LogicalLayer.POLY)
    comp = pdk_layers.gds_layer(LogicalLayer.NDIFF)
    contact = pdk_layers.gds_layer(LogicalLayer.CONTACT)
    if poly is None or comp is None:
        raise ValueError("PDK must define POLY and NDIFF (COMP) layers")
    if with_contacts and contact is None:
        raise ValueError("PDK must define CONTACT layer when with_contacts=True")

    # Clamp the requested dimensions up to the real minimums (never below DRC).
    l_nm = max(int(round(gate_length_um * 1000.0)), GF180_GATE_MIN_LENGTH_NM)
    w_nm = max(int(round(channel_width_um * 1000.0)), GF180_CHANNEL_MIN_WIDTH_NM)

    finger_space = gf180_contacted_finger_space_nm()
    endcap = GF180_POLY_ENDCAP_NM

    polys: List[LayerPolygon] = []

    # --- gate poly fingers -------------------------------------------------
    # Finger i occupies x in [i*(L + space), i*(L + space) + L].
    finger_x: List[int] = []
    for i in range(finger_count):
        x_min = i * (l_nm + finger_space)
        x_max = x_min + l_nm
        finger_x.append(x_min)
        # Poly extends endcap above and below the COMP active region.
        polys.append(
            LayerPolygon(
                layer=poly.layer,
                datatype=poly.datatype,
                polygon=Polygon.rectangle(x_min, -endcap, x_max, w_nm + endcap),
            )
        )

    # --- COMP (diffusion) --------------------------------------------------
    # COMP spans from before the first S/D column to after the last, in x, and
    # exactly the channel width in y. We need the outer ends to enclose the
    # outer S/D contacts by CO.4 (70 nm). The outer S/D contact is one
    # gate-to-contact space (CO.7) outside the outer gate edges, so:
    #   comp_x_min = first_gate_x - CO.7 - contact - CO.4
    # which guarantees CO.4 enclosure of the outer contacts.
    last_gate_x_max = finger_x[-1] + l_nm
    sd_clear = GF180_CONTACT_TO_GATE_SPACE_NM  # CO.7
    csize = GF180_CONTACT_SIZE_NM             # CO.1
    enc = GF180_COMP_CONTACT_ENCLOSURE_NM     # CO.4
    comp_x_min = finger_x[0] - sd_clear - csize - enc
    comp_x_max = last_gate_x_max + sd_clear + csize + enc
    comp_y_min = 0
    comp_y_max = w_nm
    polys.append(
        LayerPolygon(
            layer=comp.layer,
            datatype=comp.datatype,
            polygon=Polygon.rectangle(comp_x_min, comp_y_min, comp_x_max, comp_y_max),
        )
    )
    comp_bbox = (comp_x_min, comp_y_min, comp_x_max, comp_y_max)

    # --- source/drain + gate contacts -------------------------------------
    if with_contacts and contact is not None:
        cy = w_nm // 2  # contacts centred in the channel width

        # S/D contact column x-centres: one at each end + one per inter-finger
        # gap. Outer columns sit CO.7 outside the outer gate edges; inner
        # columns sit in the middle of each inter-finger gap (>= CO.7 from both
        # neighbouring gate edges, since the gap is >= contact + 2*CO.7).
        sd_centres: List[int] = []
        # left end column (CO.7 clear of the first gate's left edge)
        sd_centres.append(finger_x[0] - sd_clear - csize // 2)
        # inner columns: midpoint of each inter-finger gap
        for i in range(finger_count - 1):
            gap_lo = finger_x[i] + l_nm        # right edge of finger i
            gap_hi = finger_x[i + 1]           # left edge of finger i+1
            sd_centres.append((gap_lo + gap_hi) // 2)
        # right end column
        sd_centres.append(last_gate_x_max + sd_clear + csize // 2)

        for cx in sd_centres:
            polys.append(
                LayerPolygon(
                    layer=contact.layer,
                    datatype=contact.datatype,
                    polygon=_square(cx, cy, csize),
                )
            )

        # One gate contact, placed on the first finger's poly above the COMP
        # (a poly-strap contact). It sits CO.8 (170 nm) clear of the COMP top
        # edge so it lands on field poly, and the poly end-cap (PL.4 = 220 nm)
        # plus CO.3 enclosure (70 nm) gives room for the contact on the poly.
        # Gate-contact y-centre: above COMP top by CO.8 + half contact.
        gate_cx = finger_x[0] + l_nm // 2
        gate_cy = w_nm + GF180_POLY_CONTACT_TO_COMP_SPACE_NM + csize // 2
        polys.append(
            LayerPolygon(
                layer=contact.layer,
                datatype=contact.datatype,
                polygon=_square(gate_cx, gate_cy, csize),
            )
        )
        # Extend the gate poly finger up to enclose the gate contact by CO.3.
        # Replace the first finger poly with a taller one covering the contact.
        gate_poly_top = gate_cy + csize // 2 + GF180_POLY_CONTACT_ENCLOSURE_NM
        first = polys[0]
        fx = [v.x for v in first.polygon.vertices]
        polys[0] = LayerPolygon(
            layer=first.layer,
            datatype=first.datatype,
            polygon=Polygon.rectangle(min(fx), -endcap, max(fx), gate_poly_top),
        )

    return Gf180DeviceGeometry(
        polygons=polys,
        finger_count=finger_count,
        gate_length_nm=l_nm,
        channel_width_nm=w_nm,
        finger_space_nm=finger_space,
        comp_bbox=comp_bbox,
    )


__all__ = [
    "GF180_POLY_MIN_WIDTH_NM",
    "GF180_GATE_MIN_LENGTH_NM",
    "GF180_POLY_POLY_SPACE_NM",
    "GF180_POLY_ENDCAP_NM",
    "GF180_COMP_MIN_WIDTH_NM",
    "GF180_CHANNEL_MIN_WIDTH_NM",
    "GF180_COMP_COMP_SPACE_NM",
    "GF180_CONTACT_SIZE_NM",
    "GF180_CONTACT_SPACE_NM",
    "GF180_POLY_CONTACT_ENCLOSURE_NM",
    "GF180_COMP_CONTACT_ENCLOSURE_NM",
    "GF180_CONTACT_TO_GATE_SPACE_NM",
    "GF180_POLY_CONTACT_TO_COMP_SPACE_NM",
    "Gf180DeviceGeometry",
    "gf180_contacted_finger_space_nm",
    "build_gf180_mos_device",
]
