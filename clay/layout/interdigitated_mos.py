"""R148 (BL-148B) — real interdigitated m=N multi-finger gf180mcu MOS.

Root cause (R147 A finding)
---------------------------
``gf180_device_geom.build_gf180_mos_device`` lays out N poly fingers but does
NOT wire them into ONE electrical transistor of width ``N*Wf``:

  * it drops only a SINGLE gate contact, on finger 0, so fingers 1..N-1 have no
    strap tying their gate poly to the others. With the foundry LVS connect
    stack (``connect(poly2_con, contact)`` + ``connect(contact, metal1_con)``,
    gf180mcu.lvs lines 1070-1074) each ungated finger's poly is its own net.
    There is no shared gate;
  * the per-gap source/drain contact columns are never commoned. Nothing ties
    the alternating drain columns to one drain net or the source columns to one
    source net, so adjacent fingers do not share an electrically-defined S/D.
    The extracted topology is N distinct devices (or a series chain), never one
    ``m=N`` device.

The standard fix — and what every analog layout actually draws — is the
INTERDIGITATED multi-finger device:

  * ALL N poly fingers connect to ONE gate net through a poly GATE BUS (a poly2
    strap that physically overlaps every finger's end-cap). One gate contact on
    the bus, plus the poly-poly merge, makes all fingers one poly net.
  * The N+1 source/drain diffusion contact columns ALTERNATE S D S D ... and are
    COMMONED by metal rails: every source column wires (contact -> M1 source
    rail) and every drain column wires (contact -> M1 drain rail). Finger i then
    has a source column on one side and a drain column on the other, sharing the
    diffusion between fingers. The result is ONE transistor of width ``N*Wf``,
    all drains commoned, all sources commoned.

KLayout's ``netlist.combine_devices`` (gf180mcu.lvs lines 1881-1883, enabled by
``-rd combine=true``) folds parallel MOS that share gate/source/drain nets into
a single ``m=N`` device. The geometry here is precisely the structure that
folds: the extraction sees N parallel single-finger MOS with identical
gate/source/drain nets, which combine to one ``m=N`` MOS (device count == 1, not
N), so a layout-vs-schematic against a clean ``m=N`` schematic device matches.

This module is pure geometry construction. It WRAPS ``gf180_device_geom``
(imported read-only) for the per-finger poly / COMP / S-D-contact geometry and
all real gf180mcu 3.3V planar rule constants — no rule value is re-invented
here — then ADDS the shared-gate poly bus + gate contact and the commoning
source/drain M1 rails + bridging contacts that turn the finger array into one
electrical m=N device. All coordinates are GDS dbu (1 nm for gf180mcu).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .backend import (
    Gf180mcuLayerMap,
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    Point,
    Polygon,
)
from .gf180_device_geom import (
    GF180_COMP_CONTACT_ENCLOSURE_NM,
    GF180_CONTACT_SIZE_NM,
    GF180_CONTACT_TO_GATE_SPACE_NM,
    GF180_POLY_CONTACT_ENCLOSURE_NM,
    GF180_POLY_CONTACT_TO_COMP_SPACE_NM,
    GF180_POLY_ENDCAP_NM,
    Gf180DeviceGeometry,
    build_gf180_mos_device,
    gf180_contacted_finger_space_nm,
)
from .gf180_implant import add_implant, implant_layer_for

# Metal1 minimum width (gf180mcu metal1.drc M1.1 "min width 0.23um = 230 nm").
# A commoning rail must be at least this wide; we cite the rule but never go
# below the contact size it must enclose either.
GF180_METAL1_MIN_WIDTH_NM = 230  # rule M1.1_3.3V


def _square(cx: int, cy: int, size: int) -> Polygon:
    """A ``size`` x ``size`` square centred at (cx, cy)."""
    half = size // 2
    return Polygon.rectangle(cx - half, cy - half, cx + half, cy + half)


def _rect_bbox(lp: LayerPolygon) -> Tuple[int, int, int, int]:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _overlap(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
    """True iff two axis-aligned bboxes overlap with positive area."""
    return (
        b1[0] < b2[2]
        and b2[0] < b1[2]
        and b1[1] < b2[3]
        and b2[1] < b1[3]
    )


def _touch_or_overlap(
    b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]
) -> bool:
    """True iff two axis-aligned bboxes overlap OR share an edge (touch)."""
    return b1[0] <= b2[2] and b2[0] <= b1[2] and b1[1] <= b2[3] and b2[1] <= b1[3]


@dataclass(frozen=True)
class InterdigitatedMosDevice:
    """A real m=N interdigitated multi-finger gf180mcu MOS.

    The ``polygons`` form ONE electrical transistor of width ``finger_count *
    channel_width_nm``: a single shared gate net (all poly fingers tied by the
    gate bus + gate contact) and alternating, commoned source/drain diffusion
    columns (every source column wired to the source M1 rail, every drain column
    to the drain M1 rail).

    Attributes
    ----------
    polygons:
        All drawn shapes (per-finger poly + COMP + S/D contacts from the wrapped
        ``gf180_device_geom`` device, PLUS the gate-bus poly, gate contact, and
        the source/drain commoning M1 rails and their bridging contacts).
    finger_count:
        Number of gate poly fingers N. The device width is ``N * channel_width``.
    gate_length_nm, channel_width_nm:
        Per-finger drawn channel length L and width Wf.
    sd_column_x:
        x-centres of the N+1 source/drain contact columns, left to right.
    source_column_idx, drain_column_idx:
        The S/D column indices (into ``sd_column_x``) commoned to the source and
        drain net respectively. Columns alternate S D S D ...; with N fingers
        there are N+1 columns. The convention here: even index -> source, odd
        index -> drain.
    gate_net, source_net, drain_net:
        The three logical net names this device's terminals merge into.
    device_type:
        ``"NMOS"`` or ``"PMOS"`` — selects the implant (Nplus 32/0 / Pplus
        31/0) drawn over the COMP so KLayout's MOS extraction classifies the
        device (and DF.12 "COMP must be covered by an implant" is satisfied).
    """

    polygons: List[LayerPolygon]
    finger_count: int
    gate_length_nm: int
    channel_width_nm: int
    sd_column_x: Tuple[int, ...]
    source_column_idx: Tuple[int, ...]
    drain_column_idx: Tuple[int, ...]
    gate_net: str = "gate"
    source_net: str = "source"
    drain_net: str = "drain"
    device_type: str = "NMOS"

    @property
    def total_width_nm(self) -> int:
        """The folded device's total channel width = N * Wf."""
        return self.finger_count * self.channel_width_nm

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

    def polys_on(self, key: Tuple[int, int]) -> List[LayerPolygon]:
        return [lp for lp in self.polygons if (lp.layer, lp.datatype) == key]

    # --- structural invariants (used by tests; pure geometry) ---------------

    def poly_fingers_share_one_net(self, poly_key: Tuple[int, int]) -> bool:
        """True iff all poly shapes form ONE connected net.

        Every finger must touch/overlap the gate bus, so the union of all poly
        rectangles is a single connected component. We verify by checking that
        the gate-bus rectangle touches every finger rectangle (a star topology
        through the bus is sufficient for one net).
        """
        polys = self.polys_on(poly_key)
        if len(polys) < 2:
            return len(polys) == 1  # a single finger trivially shares its net
        boxes = [_rect_bbox(p) for p in polys]
        # The gate bus is the poly rect spanning the widest x-extent.
        widths = [(b[2] - b[0], i) for i, b in enumerate(boxes)]
        bus_idx = max(widths)[1]
        bus = boxes[bus_idx]
        for i, b in enumerate(boxes):
            if i == bus_idx:
                continue
            if not _touch_or_overlap(bus, b):
                return False
        return True

    def gate_contact_on_bus(
        self, contact_key: Tuple[int, int], poly_key: Tuple[int, int]
    ) -> bool:
        """True iff there is a contact overlapping the gate bus poly."""
        polys = self.polys_on(poly_key)
        if not polys:
            return False
        boxes = [_rect_bbox(p) for p in polys]
        widths = [(b[2] - b[0], i) for i, b in enumerate(boxes)]
        bus = boxes[max(widths)[1]]
        return any(
            _overlap(bus, _rect_bbox(c)) for c in self.polys_on(contact_key)
        )

    def commoning_rail_overlaps_all_columns(
        self,
        rail_idx_columns: Tuple[int, ...],
        rail_bbox: Tuple[int, int, int, int],
    ) -> bool:
        """True iff ``rail_bbox`` spans the x-range of all given S/D columns."""
        csize = GF180_CONTACT_SIZE_NM
        for ci in rail_idx_columns:
            cx = self.sd_column_x[ci]
            col_lo, col_hi = cx - csize // 2, cx + csize // 2
            if not (rail_bbox[0] <= col_lo and rail_bbox[2] >= col_hi):
                return False
        return True


def build_interdigitated_mos(
    *,
    finger_count: int,
    gate_length_um: float,
    channel_width_um: float,
    pdk_layers: Optional[LayerMap] = None,
    gate_net: str = "gate",
    source_net: str = "source",
    drain_net: str = "drain",
    device_type: str = "NMOS",
) -> InterdigitatedMosDevice:
    """Build a real interdigitated m=N gf180mcu MOS (ONE electrical transistor).

    The per-finger poly / COMP / S-D-contact geometry comes from the committed
    ``gf180_device_geom.build_gf180_mos_device`` (imported read-only). On top of
    that this adds the two structural pieces the wrapped device is missing:

    1. SHARED GATE — a horizontal poly2 gate bus drawn above the COMP that
       overlaps the end-cap of EVERY finger (tying all N fingers into one poly
       net), plus a single gate contact landing on the bus.

    2. ALTERNATING + COMMONED SOURCE/DRAIN — the N+1 S/D contact columns are
       labelled S D S D ... (even index source, odd index drain). A source M1
       rail (below the COMP) spans all source columns and a drain M1 rail (above
       the COMP) spans all drain columns; each S/D contact is duplicated up onto
       its rail so the contact overlaps both the COMP and the M1 rail, which is
       the via the LVS connect stack needs to fuse the column into the
       source/drain net. Every source column then carries the same net, every
       drain column the same net, and adjacent fingers share their diffusion.

    Parameters
    ----------
    finger_count:
        Number of gate fingers N (>= 1). Device width = N * channel_width.
    gate_length_um, channel_width_um:
        Per-finger drawn L and Wf in um (clamped up to the gf180 minimums by the
        wrapped device builder).
    pdk_layers:
        A gf180mcu LayerMap (default ``Gf180mcuLayerMap()``).
    gate_net, source_net, drain_net:
        Logical net names the gate / source / drain terminals merge into.

    Returns
    -------
    InterdigitatedMosDevice
        Geometry of one m=N transistor.

    Raises
    ------
    ValueError
        If ``finger_count < 1`` or the PDK lacks POLY / COMP / CONTACT / METAL1.
    """
    if finger_count < 1:
        raise ValueError("finger_count must be >= 1")
    if pdk_layers is None:
        pdk_layers = Gf180mcuLayerMap()

    poly = pdk_layers.gds_layer(LogicalLayer.POLY)
    comp = pdk_layers.gds_layer(LogicalLayer.NDIFF)
    contact = pdk_layers.gds_layer(LogicalLayer.CONTACT)
    metal1 = pdk_layers.gds_layer(LogicalLayer.METAL1)
    if poly is None or comp is None or contact is None or metal1 is None:
        raise ValueError(
            "PDK must define POLY, NDIFF (COMP), CONTACT and METAL1 layers"
        )

    # --- per-finger base geometry (wrapped, read-only) ---------------------
    # We disable the wrapped builder's single asymmetric gate contact: this
    # module supplies a real shared gate bus + contact tying ALL fingers, not a
    # one-finger strap. We keep its DRC-legal poly fingers, COMP, and the S/D
    # contact columns.
    base: Gf180DeviceGeometry = build_gf180_mos_device(
        finger_count=finger_count,
        gate_length_um=gate_length_um,
        channel_width_um=channel_width_um,
        pdk_layers=pdk_layers,
        with_contacts=False,  # we add our own commoned S/D contacts below
    )

    poly_key = (poly.layer, poly.datatype)
    comp_key = (comp.layer, comp.datatype)
    contact_key = (contact.layer, contact.datatype)
    m1_key = (metal1.layer, metal1.datatype)

    l_nm = base.gate_length_nm
    w_nm = base.channel_width_nm
    finger_space = base.finger_space_nm
    endcap = GF180_POLY_ENDCAP_NM
    csize = GF180_CONTACT_SIZE_NM
    sd_clear = GF180_CONTACT_TO_GATE_SPACE_NM  # CO.7
    comp_enc = GF180_COMP_CONTACT_ENCLOSURE_NM  # CO.4

    # Finger i left-edge x (matches the wrapped builder's finger placement).
    finger_x: List[int] = [i * (l_nm + finger_space) for i in range(finger_count)]
    last_gate_x_max = finger_x[-1] + l_nm

    polys: List[LayerPolygon] = list(base.polygons)  # poly fingers + COMP

    # --- implant over the COMP (so MOS extraction classifies the device) ----
    # KLayout's gf180 MOS extraction (and DF.12) require the COMP to be covered
    # by Nplus (NMOS) / Pplus (PMOS). Without it the device extracts as no MOS
    # at all. ``add_implant`` (committed, read-only) grows the COMP bbox by the
    # NP.5b/PP.5b extension (160 nm) onto the correct implant layer.
    polys.extend(add_implant(base.comp_polygons(), device_type))

    # --- 1. SHARED GATE: poly bus tying every finger + one gate contact -----
    # The bus is a horizontal poly2 strap above the COMP top edge. It must
    # overlap each finger's end-cap (the fingers extend ``endcap`` above the
    # COMP, i.e. to y = w_nm + endcap), so the bus sits within that end-cap band
    # and spans the full finger array in x. Placing the bus from the COMP top
    # (y = w_nm) up to the finger top (y = w_nm + endcap) makes it overlap every
    # finger rectangle -> one connected poly net.
    bus_y_min = w_nm
    bus_y_max = w_nm + endcap
    bus_x_min = finger_x[0]
    bus_x_max = last_gate_x_max
    polys.append(
        LayerPolygon(
            layer=poly.layer,
            datatype=poly.datatype,
            polygon=Polygon.rectangle(bus_x_min, bus_y_min, bus_x_max, bus_y_max),
        )
    )
    # Raise the bus a little higher over the first finger so a CO.3-enclosed gate
    # contact fits on poly clear of the COMP (CO.8 = poly-contact-to-COMP space).
    gate_cx = finger_x[0] + l_nm // 2
    gate_cy = w_nm + GF180_POLY_CONTACT_TO_COMP_SPACE_NM + csize // 2
    gate_pad_top = gate_cy + csize // 2 + GF180_POLY_CONTACT_ENCLOSURE_NM
    # A poly pad on the bus over the first finger so the gate contact lands on
    # poly with CO.3 enclosure all around (and stays clear of COMP by CO.8).
    polys.append(
        LayerPolygon(
            layer=poly.layer,
            datatype=poly.datatype,
            polygon=Polygon.rectangle(
                finger_x[0],
                w_nm,
                finger_x[0] + l_nm,
                gate_pad_top,
            ),
        )
    )
    polys.append(
        LayerPolygon(
            layer=contact.layer,
            datatype=contact.datatype,
            polygon=_square(gate_cx, gate_cy, csize),
        )
    )

    # --- 2. ALTERNATING + COMMONED SOURCE/DRAIN -----------------------------
    # N+1 contact columns at the finger boundaries (one before finger 0, one in
    # each inter-finger gap, one after the last finger). Column centres match
    # the wrapped builder's S/D placement so the contacts land on the COMP with
    # CO.7 clearance from every gate edge.
    sd_centres: List[int] = []
    sd_centres.append(finger_x[0] - sd_clear - csize // 2)  # left end
    for i in range(finger_count - 1):
        gap_lo = finger_x[i] + l_nm
        gap_hi = finger_x[i + 1]
        sd_centres.append((gap_lo + gap_hi) // 2)
    sd_centres.append(last_gate_x_max + sd_clear + csize // 2)  # right end

    cy_mid = w_nm // 2
    source_cols = tuple(i for i in range(len(sd_centres)) if i % 2 == 0)
    drain_cols = tuple(i for i in range(len(sd_centres)) if i % 2 == 1)

    # COMP-level contact in every column (on the active diffusion, CO.4 enclosed
    # by the COMP rect from the wrapped builder).
    for cx in sd_centres:
        polys.append(
            LayerPolygon(
                layer=contact.layer,
                datatype=contact.datatype,
                polygon=_square(cx, cy_mid, csize),
            )
        )

    # Source commoning M1 rail BELOW the COMP, drain rail ABOVE the COMP (clear
    # of the gate bus). Each rail spans its columns in x and is M1-min-width tall
    # in y; a contact is duplicated onto the rail in each of that net's columns
    # so the via overlaps both the COMP-level contact's column and the rail.
    rail_h = max(GF180_METAL1_MIN_WIDTH_NM, csize + 2 * comp_enc)

    def _rail_and_vias(
        cols: Tuple[int, ...], rail_cy: int
    ) -> Tuple[Tuple[int, int, int, int], List[LayerPolygon]]:
        xs = [sd_centres[c] for c in cols]
        rail_x_min = min(xs) - csize // 2 - comp_enc
        rail_x_max = max(xs) + csize // 2 + comp_enc
        rail_bbox = (rail_x_min, rail_cy - rail_h // 2, rail_x_max, rail_cy + rail_h // 2)
        out: List[LayerPolygon] = [
            LayerPolygon(
                layer=metal1.layer,
                datatype=metal1.datatype,
                polygon=Polygon.rectangle(*rail_bbox),
            )
        ]
        # A bridging contact at every column on this rail's y, so the column's
        # diffusion -> contact -> M1 rail connect stack fuses it into the net.
        for cx in xs:
            out.append(
                LayerPolygon(
                    layer=contact.layer,
                    datatype=contact.datatype,
                    polygon=_square(cx, rail_cy, csize),
                )
            )
        return rail_bbox, out

    # Source rail sits below the COMP bottom (y=0) by its half-height + a clear
    # gap; drain rail above the COMP top, above the gate bus to stay clear.
    src_rail_cy = -(rail_h // 2 + sd_clear)
    drn_rail_cy = gate_pad_top + sd_clear + rail_h // 2
    src_bbox, src_polys = _rail_and_vias(source_cols, src_rail_cy)
    drn_bbox, drn_polys = _rail_and_vias(drain_cols, drn_rail_cy)
    polys.extend(src_polys)
    polys.extend(drn_polys)

    # Vertical M1 stubs joining each column's COMP-level contact to its rail, so
    # the per-column contact, the stub, and the rail-level contact share metal
    # (the rail is the commoning conductor; the stub gives a continuous M1 path).
    def _stub(cx: int, y_lo: int, y_hi: int) -> LayerPolygon:
        half = max(csize, GF180_METAL1_MIN_WIDTH_NM) // 2
        return LayerPolygon(
            layer=metal1.layer,
            datatype=metal1.datatype,
            polygon=Polygon.rectangle(cx - half, y_lo, cx + half, y_hi),
        )

    for c in source_cols:
        polys.append(_stub(sd_centres[c], src_rail_cy, cy_mid))
    for c in drain_cols:
        polys.append(_stub(sd_centres[c], cy_mid, drn_rail_cy))

    return InterdigitatedMosDevice(
        polygons=polys,
        finger_count=finger_count,
        gate_length_nm=l_nm,
        channel_width_nm=w_nm,
        sd_column_x=tuple(sd_centres),
        source_column_idx=source_cols,
        drain_column_idx=drain_cols,
        gate_net=gate_net,
        source_net=source_net,
        drain_net=drain_net,
        device_type=device_type,
    )


__all__ = [
    "GF180_METAL1_MIN_WIDTH_NM",
    "InterdigitatedMosDevice",
    "build_interdigitated_mos",
]
