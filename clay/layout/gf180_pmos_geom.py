"""R140 (BL-140E) — real gf180mcu PLANAR PMOS device geometry (in Nwell).

R139 (BL-139A) built the n-channel half of the device family in
``clay/layout/gf180_device_geom.py``: a DRC-legal-by-construction multi-finger
planar MOS (COMP diffusion + poly gate fingers + S/D / gate contacts) using the
REAL gf180mcu 3.3V planar device rules (PL.*, DF.*, CO.*), never the fictional
48 nm "fin pitch" of the legacy lowering.

This module completes the family with the p-channel device. A gf180mcu PMOS is
*exactly* the same planar finger construction — there is a single diffusion
layer (COMP, 22/0) shared by n- and p-channel devices; the only structural
difference is that the PMOS COMP (the "PCOMP") sits inside an **Nwell** (21/0),
and the Nwell must enclose that COMP by the well-overlap-of-PCOMP rule.

So this builder REUSES ``build_gf180_mos_device`` verbatim for the COMP, poly
fingers and contacts (imported read-only), then wraps the COMP in an enclosing
Nwell rectangle that overlaps the COMP on every side by the required margin.

Nwell enclosure rule used (real gf180mcu KLayout DRC deck, cited):

    DF.4c_3.3V  Min. (Nwell overlap of PCOMP) outside DNWELL. : 0.43um = 430 nm
        Source line: comp.drc rule deck, line 366 (the ``# Rule DF.4c_3.3V`` /
        ``df4c_l.output("DF.4c_3.3V", ...)`` block):
            AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/rules/
            klayout/drc/rule_decks/comp.drc
        Deck expression:
            df4c_l1 = nwell_n_dnwell.outside(sramcore)
                       .enclosing(pcomp.outside(dnwell), 0.43.um, euclidian)...

This is the canonical "Nwell encloses the PMOS diffusion" rule for a device
outside DNWELL (the baseline analog flow). The companion Pplus implant overlap
rules (PP.5a "Overlap of P-channel gate" = 0.23um, PP.5b "Extension beyond
COMP inside Nwell" = 0.16um, pplus.drc) are noted for reference; this geometry
emits the Nwell and the reused COMP/poly/contacts. The Pplus implant layer
(31/0) has no Clay LogicalLayer member and is not part of the intra-device DRC
subset (which checks POLY/COMP/NWELL min width + spacing), so it is left to a
later implant-layer pass.

The Nwell is large enough to clear its own min-width rule by construction:

    NW.1a_3.3V  Min. Nwell Width. : 0.86um = 860 nm   (nwell.drc, line 312)

With the COMP enclosed by 430 nm on every side, the Nwell's narrow dimension is
``comp_narrow + 2 * 430`` which is always >= 860 nm, so the Nwell never trips
the NW.1a min-width check in ``layout_drc.run_layout_drc``.

This module is pure geometry construction. It imports committed modules
(``backend``, ``gf180_device_geom``) read-only and emits ``LayerPolygon`` lists
/ a ``Gf180PmosGeometry``. All coordinates are GDS database units (dbu = 1 nm
for gf180mcu).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .backend import (
    Gf180mcuLayerMap,
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    Polygon,
)
from .gf180_device_geom import (
    Gf180DeviceGeometry,
    build_gf180_mos_device,
)


# --- real gf180mcu Nwell rules for a PMOS device (nm) ----------------------
#
# Every value below is a verbatim transcription of the gf180mcu KLayout DRC
# deck rule cited; no value is invented.

# comp.drc rule_deck — DF.4c_3.3V (line 366):
#   "Min. (Nwell overlap of PCOMP) outside DNWELL. : 0.43um"
# This is the Nwell enclosure of the PMOS diffusion (PCOMP) — the defining
# well-overlap rule that distinguishes a PMOS from the NMOS built in R139.
GF180_NWELL_PCOMP_ENCLOSURE_NM = 430  # rule DF.4c_3.3V

# nwell.drc rule_deck — NW.1a_3.3V (line 312): "Min. Nwell Width. : 0.86um"
# Used only to document/assert the by-construction min-width margin.
GF180_NWELL_MIN_WIDTH_NM = 860  # rule NW.1a_3.3V

# pplus.drc rule_deck — PP.5a (line 300): "Overlap of P-channel gate. : 0.23um"
#   PP.5b (line 310): "Extension beyond COMP ... Inside NWELL : 0.16um"
# Recorded for reference; the Pplus implant geometry is a later pass.
GF180_PPLUS_PGATE_OVERLAP_NM = 230  # rule PP.5a
GF180_PPLUS_COMP_EXTENSION_NM = 160  # rule PP.5b


@dataclass(frozen=True)
class Gf180PmosGeometry:
    """Result of lowering one multi-finger gf180mcu PMOS device.

    Wraps the reused n/p-shared core device geometry (COMP, poly fingers,
    contacts) and adds the enclosing Nwell rectangle required for a p-channel
    device.

    Attributes:
        polygons: every drawn shape — the enclosing Nwell first, then the
            reused COMP / poly fingers / contacts.
        core: the underlying ``Gf180DeviceGeometry`` (COMP/poly/contacts).
        nwell_bbox: (x_min, y_min, x_max, y_max) of the enclosing Nwell rect.
        nwell_layer / nwell_datatype: GDS (layer, datatype) of the Nwell
            (21 / 0 for gf180mcu).
        nwell_enclosure_nm: Nwell-overlap-of-PCOMP margin used (DF.4c = 430 nm).
    """

    polygons: List[LayerPolygon]
    core: Gf180DeviceGeometry
    nwell_bbox: tuple
    nwell_layer: int
    nwell_datatype: int
    nwell_enclosure_nm: int

    # --- convenience accessors mirroring Gf180DeviceGeometry -----------------

    @property
    def finger_count(self) -> int:
        return self.core.finger_count

    @property
    def gate_length_nm(self) -> int:
        return self.core.gate_length_nm

    @property
    def channel_width_nm(self) -> int:
        return self.core.channel_width_nm

    @property
    def finger_space_nm(self) -> int:
        return self.core.finger_space_nm

    @property
    def comp_bbox(self) -> tuple:
        return self.core.comp_bbox

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
        return self.core.poly_polygons()

    def comp_polygons(self) -> List[LayerPolygon]:
        return self.core.comp_polygons()

    def contact_polygons(self) -> List[LayerPolygon]:
        return self.core.contact_polygons()

    def nwell_polygons(self) -> List[LayerPolygon]:
        return [
            lp for lp in self.polygons
            if (lp.layer, lp.datatype) == (self.nwell_layer, self.nwell_datatype)
        ]


def build_gf180_pmos_device(
    *,
    finger_count: int,
    gate_length_um: float,
    channel_width_um: float,
    pdk_layers: Optional[LayerMap] = None,
    with_contacts: bool = True,
    nwell_enclosure_nm: int = GF180_NWELL_PCOMP_ENCLOSURE_NM,
) -> Gf180PmosGeometry:
    """Build a DRC-legal-by-construction multi-finger gf180mcu PMOS device.

    The COMP diffusion, poly gate fingers and S/D / gate contacts are produced
    by ``build_gf180_mos_device`` (the n/p-shared planar device builder from
    R139); a gf180mcu PMOS uses the identical finger construction. This builder
    then wraps the COMP in an enclosing **Nwell** (21/0) rectangle that
    overlaps the COMP on every side by ``nwell_enclosure_nm`` (default
    DF.4c_3.3V = 430 nm, the Nwell-overlap-of-PCOMP rule).

    Geometry layout (x = finger direction, y = channel-width direction)::

        Nwell encloses the entire COMP, extended outward by DF.4c (430 nm) on
        all four sides. The COMP, poly fingers and contacts are identical to
        the NMOS — only the surrounding Nwell distinguishes the p-channel
        device.

    Args:
        finger_count: number of gate fingers (>= 1).
        gate_length_um: drawn channel length L in um; clamped up to PL.2 min.
        channel_width_um: drawn channel width W in um; clamped up to DF.2a min.
        pdk_layers: a gf180mcu LayerMap (default ``Gf180mcuLayerMap()``). Must
            define the NWELL layer.
        with_contacts: if True (default), emit S/D + gate contacts.
        nwell_enclosure_nm: Nwell overlap of the COMP on each side (default
            DF.4c_3.3V = 430 nm). Must be >= the rule minimum.

    Returns:
        Gf180PmosGeometry with the enclosing Nwell plus the reused core shapes.

    Raises:
        ValueError: if finger_count < 1, the PDK lacks an NWELL layer, or
            ``nwell_enclosure_nm`` is below the DF.4c minimum.
    """
    if pdk_layers is None:
        pdk_layers = Gf180mcuLayerMap()
    if nwell_enclosure_nm < GF180_NWELL_PCOMP_ENCLOSURE_NM:
        raise ValueError(
            f"nwell_enclosure_nm ({nwell_enclosure_nm} nm) is below the "
            f"DF.4c_3.3V minimum Nwell-overlap-of-PCOMP rule "
            f"({GF180_NWELL_PCOMP_ENCLOSURE_NM} nm)"
        )

    nwell = pdk_layers.gds_layer(LogicalLayer.NWELL)
    if nwell is None:
        raise ValueError("PDK must define the NWELL layer for a PMOS device")

    # Reuse the n/p-shared planar device builder for COMP + poly + contacts.
    # build_gf180_mos_device raises on finger_count < 1, so we do not need to
    # re-check here.
    core = build_gf180_mos_device(
        finger_count=finger_count,
        gate_length_um=gate_length_um,
        channel_width_um=channel_width_um,
        pdk_layers=pdk_layers,
        with_contacts=with_contacts,
    )

    # Enclosing Nwell: extend the COMP bbox outward by the enclosure on all
    # sides. DF.4c is a *diffusion* (PCOMP) overlap rule, so the relevant
    # enclosed shape is the COMP rectangle, not the poly end-caps or the
    # off-COMP gate contact (those sit on POLY / CONTACT, not on the well
    # boundary). Enclosing the COMP by DF.4c is what the deck checks.
    comp_x_min, comp_y_min, comp_x_max, comp_y_max = core.comp_bbox
    nw_x_min = comp_x_min - nwell_enclosure_nm
    nw_y_min = comp_y_min - nwell_enclosure_nm
    nw_x_max = comp_x_max + nwell_enclosure_nm
    nw_y_max = comp_y_max + nwell_enclosure_nm
    nwell_bbox = (nw_x_min, nw_y_min, nw_x_max, nw_y_max)

    nwell_poly = LayerPolygon(
        layer=nwell.layer,
        datatype=nwell.datatype,
        polygon=Polygon.rectangle(nw_x_min, nw_y_min, nw_x_max, nw_y_max),
    )

    # Nwell first (drawn under the device), then the reused core shapes.
    polygons: List[LayerPolygon] = [nwell_poly]
    polygons.extend(
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(vertices=list(lp.polygon.vertices)),
        )
        for lp in core.polygons
    )

    return Gf180PmosGeometry(
        polygons=polygons,
        core=core,
        nwell_bbox=nwell_bbox,
        nwell_layer=nwell.layer,
        nwell_datatype=nwell.datatype,
        nwell_enclosure_nm=nwell_enclosure_nm,
    )


__all__ = [
    "GF180_NWELL_PCOMP_ENCLOSURE_NM",
    "GF180_NWELL_MIN_WIDTH_NM",
    "GF180_PPLUS_PGATE_OVERLAP_NM",
    "GF180_PPLUS_COMP_EXTENSION_NM",
    "Gf180PmosGeometry",
    "build_gf180_pmos_device",
]
