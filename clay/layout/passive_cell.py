"""PARAMETRIC PASSIVE-CELL BUILDER -- native target-PDK MIM cap + poly resistor.

WHY THIS EXISTS
===============
``clay.layout.device_cell.build_device`` synthesises MOS only. So
``clay.layout.extract.realize_render.real_device_geometry`` fills a target PDK's
PASSIVES (caps / resistors) by copying the external_pdk reference vendor capture
(``source='reference_passive'``) -- a stand-in that draws the WRONG process's
layers and plate areas for, say, external_pdk. This module replaces that proxy with REAL
synthesised target-PDK passive geometry:

  * :func:`build_mim_cap` -- a MIM capacitor whose plate area = value / cap density,
    drawn on the PDK's OWN MIM / metal / via layers.
  * :func:`build_poly_res` -- a poly resistor whose body length = value / sheet *
    width, drawn on the PDK's OWN poly / salicide-block / contact / metal layers.

Both return a :class:`~clay.layout.extract.library_device.CellGeometry` -- the
single geometry currency the emitter (``emit_library_gds``) stamps -- with
``source='passive_synth'``, centroid-relative um polygons, and ``clean=True``.

THE DISCIPLINE (enforced here)
==============================
  * GEOMETRY is PROCESS KNOWLEDGE: every dimension comes from a PDK fact -- the
    cap density (fF/um^2) / poly sheet resistance (ohm/sq) and the foundry DRC
    minima (plate min width, via size, contact size, salicide-block width ...).
    The plate area / resistor length is COMPUTED from the requested value and the
    density/sheet, NEVER copied from a reference coordinate.
  * POSITION is NOT here: a stamp is centroid-relative (its centroid is (0,0)),
    so it carries no floorplan position. The relation realise supplies positions.
  * We do NOT fabricate a clean-looking shape: the stamp is asserted MERGE-CLEAN
    (no same-layer self-overlap) by a real klayout boolean union-area gate, the
    same gate ``realize_render._merge_layer_rings`` uses. A degenerate request
    (non-positive value) raises rather than emit a fake cell.

LAYER + DENSITY/SHEET SOURCING (honesty)
========================================
The Clay :class:`~clay.layout.tech.LayoutTech` / PDK adapter surface for the
commercial PDKs exposes only the MOS layers (metal1/poly/contact/diff/implant/
nwell) and does NOT carry the MIM-cap plate layers, the cap density, the
resistor salicide-block / marker layers, or the poly sheet resistance. Those are
PROCESS facts that live in the foundry's own sign-off artifacts. For open_pdk
this module sources them from the REAL, in-repo external_pdk PDK decks/models and labels
each value's provenance in :data:`_SMIC18MMRF_PASSIVE` (the source line is cited
in the comment). When the tech / adapter DOES expose a value (e.g. a future
adapter that publishes ``poly_sheet_rho_ohm_sq`` or a MIM density) that wins, and
the resolution path records ``provenance='tech'``. A PDK with no resolvable
density/sheet falls back to a documented, clearly-labelled default and the
``CellGeometry.reason`` SAYS SO (``provenance='documented-default'``).

``open_pdk`` carries a MIM-density-ONLY row (:data:`_TSMC180RF_PASSIVE`, density
1.025 fF/um^2, cited to ``external_model.scs`` -- see the row's own comment for the full
citation set). open_pdk's poly sheet-rho is NOT tabulated here: it already
resolves off the PDK ADAPTER surface (``clay.pdk.open_pdk.ADAPTER
.poly_sheet_rho_ohm_sq``), which the adapter-first resolution order in
:func:`_resolve_poly_res_facts` / ``passive_device_cell._poly_sheet_rho`` always
consults before this table, so a open_pdk "res" row here would be dead data. The
open_pdk MIM row exists so the GENERIC resolver
(``passive_device_cell.mim_density_f_per_um2``) succeeds for open_pdk too (it
used to raise ``UnsupportedPdkFactError`` for every non-gf180 PDK this table did
not tabulate); the dedicated external_pdk MIM GEOMETRY builder
(``clay.layout.passive_device_cell_tsmc.build_mim_capacitor_tsmc``) is unaffected
-- it draws from its OWN cited constants, not this table.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from clay.layout.extract.library_device import CellGeometry

_LayerKey = Tuple[int, int]
_Ring = List[Tuple[float, float]]

# nm resolution the merge-clean gate works in (= emit_library_gds dbu).
_DBU: float = 0.001
# Union-area-preservation tolerance (um^2): the merge must remove only redundant
# overlap, never drawn area. < 1e-6 um^2 is well below a single dbu^2.
_UNION_AREA_EPS_UM2: float = 1e-6


# --------------------------------------------------------------------------- #
# Per-PDK passive process facts (layers + density/sheet + the DRC minima needed
# to draw a DRC-legal cell). Every value is sourced from the REAL in-repo PDK
# sign-off artifacts; the source file + line is cited so provenance is auditable.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _MimCapFacts:
    """MIM-capacitor process facts for one PDK (all lengths um, density fF/um^2)."""

    density_ff_um2: float       # parallel-plate cap density (fF/um^2)
    density_provenance: str     # where density came from (cited)
    top_plate_layer: _LayerKey  # MIM top plate GDS (layer, datatype)
    bot_plate_layer: _LayerKey  # bottom plate metal GDS
    via_layer: _LayerKey        # via from top plate up to the cover metal
    cover_layer: _LayerKey      # cover metal over the via (top-plate routing)
    plate_min_w_um: float       # MIM.* min top-plate width
    plate_max_w_um: float       # MIM.* max top-plate dimension (foundry cap)
    plate_space_um: float       # MIM.* min space between two top plates (a bank)
    bot_ext_um: float           # bottom-plate extension beyond the top plate
    via_size_um: float          # via cut size
    via_space_um: float         # via cut-to-cut space (generic, off-MIM)
    via_on_mim_space_um: float  # MIM.* min space between two vias ON a MIM plate
    via_to_plate_edge_um: float # min clearance of a via to the MIM plate edge
    mim_over_via_um: float      # min MIM extension beyond a via on it
    cover_ext_via_um: float     # cover-metal extension beyond the via
    marker_layers: Tuple[_LayerKey, ...] = ()
    marker_ext_um: float = 0.0


@dataclass(frozen=True)
class _PolyResFacts:
    """Poly-resistor process facts for one PDK (lengths um, sheet ohm/sq)."""

    sheet_ohm_sq: float         # poly sheet resistance (ohm/square)
    sheet_provenance: str       # where sheet came from (cited)
    sheet_model: str            # the resistor model the sheet is for
    body_layer: _LayerKey       # poly body GDS (layer, datatype)
    sab_layer: _LayerKey        # salicide-block over the body (non-silicide R)
    marker_layer: Optional[_LayerKey]  # resistor-recognition marker (or None)
    implant_layer: Optional[_LayerKey]  # S/D implant over the heads (or None)
    contact_layer: _LayerKey    # head contact cut GDS
    head_metal_layer: _LayerKey # head terminal metal GDS
    body_min_w_um: float        # poly min width
    contact_size_um: float      # contact cut size
    contact_space_um: float     # contact cut-to-cut space
    contact_enc_um: float       # poly/metal enclosure of a head contact
    sab_over_body_um: float     # SAB extension beyond the resistor body
    head_um: float              # silicided head length (contacts land here)
    metal_min_w_um: float       # head metal min width
    # S/D implant (pplus/nplus) MIN enclosure of the resistor body on ALL sides.
    # The implant must enclose pres_poly (poly2&sab&res_mk) -- and the sab-covered
    # poly2 strip -- by this margin or the foundry implant-enclosure rule fires.
    # 0.0 => legacy behaviour (implant = full poly bbox, no margin).
    implant_over_body_um: float = 0.0
    # MIN clearance a head CONTACT must keep from the SAB edge (the salicide-block
    # to contact spacing). The contact's body-side edge sits this far beyond the
    # SAB outer edge; the head is lengthened to enclose the pushed-out contact.
    # 0.0 => legacy behaviour (contact centred in the head, may straddle SAB).
    sab_to_contact_um: float = 0.0


# open_pdk -- sourced from the in-repo external_pdk 0.18 Mixed-Mode RF PDK
# (pdks/external_pdk). Every value cites the real file + the rule/line it
# traces to. This is process knowledge, not a guess.


_GF180MCU_MIM_DENSITIES_FF_UM2: Dict[str, float] = {
    "1": 1.0,
    "1.5": 1.5,
    "2": 2.0,
}

# Option-A (Metal2/Via2/Metal3) and 5LM Option-B (Metal4/Via4/Metal5) MIM
# process facts sourced from mim_capacitor_option_{a,b}.drc and gf180mcu.lvs.
_GF180MCU_MIM_OPTION_A = _MimCapFacts(
    density_ff_um2=1.0,
    density_provenance="GF180MCU gf180mcu.lvs MIM_OPTION=A mim_1p0fF (1.0 fF/um^2)",
    top_plate_layer=(75, 0),    # FuseTop / CAPM
    bot_plate_layer=(36, 0),    # Metal2
    via_layer=(38, 0),          # Via2
    cover_layer=(42, 0),        # Metal3
    plate_min_w_um=5.0,         # MIM.8a min area 25 um^2 -> 5um square
    plate_max_w_um=100.0,       # MIM.8b max area 10000 um^2 -> 100um square
    plate_space_um=0.6,         # MIM.6 min unrelated top-plate space
    bot_ext_um=0.6,             # MIM.3 bottom-plate overlap of FuseTop
    via_size_um=0.26,           # V2.1 cut size
    via_space_um=0.26,          # V2.2a generic space
    via_on_mim_space_um=0.5,    # MIM.9 sea-of-via spacing on FuseTop
    via_to_plate_edge_um=0.4,   # MIM.4 FuseTop overlap of Via2
    mim_over_via_um=0.4,        # MIM.4 FuseTop overlap of Via2
    cover_ext_via_um=0.06,      # V2.4b top metal enclosure of Via2
    marker_layers=((117, 5), (117, 10)),  # cap_mk + mim_l_mk
    marker_ext_um=0.20,
)

_GF180MCU_MIM_OPTION_B = _MimCapFacts(
    density_ff_um2=1.0,
    density_provenance="GF180MCU gf180mcu.lvs MIM_OPTION=B (5LM M4/V4/M5) 1.0 fF/um^2",
    top_plate_layer=(75, 0),    # FuseTop / CAPM
    bot_plate_layer=(46, 0),    # Metal4 (topmin1_metal under 5LM)
    via_layer=(41, 0),          # Via4 (top_via under 5LM)
    cover_layer=(81, 0),        # Metal5 (top_metal under 5LM)
    plate_min_w_um=5.0,         # MIMTM.8a min area 25 um^2 -> 5um square
    plate_max_w_um=100.0,       # MIMTM.8b max area 10000 um^2 -> 100um square
    plate_space_um=0.6,         # MIMTM.6 min unrelated top-plate space
    bot_ext_um=0.6,             # MIMTM.3 bottom-plate overlap of FuseTop
    via_size_um=0.26,           # V4.1 cut size
    via_space_um=0.26,          # V4.2a generic space
    via_on_mim_space_um=0.5,    # MIMTM.9 sea-of-via spacing on FuseTop
    via_to_plate_edge_um=0.4,   # MIMTM.4 FuseTop overlap of Via4
    mim_over_via_um=0.4,        # MIMTM.4 FuseTop overlap of Via4
    cover_ext_via_um=0.06,      # V4.4b top metal enclosure of Via4
    marker_layers=((117, 5), (117, 10)),  # cap_mk + mim_l_mk
    marker_ext_um=0.20,
)

# gf180mcu -- the poly-resistor facts sourced from the in-repo GF180MCU open PDK
# (gf180mcu_fd_pr klayout LVS + DRC rule decks). The standard PRECISION poly resistor is
# the unsilicided P+ poly ``ppolyf_u`` (350 ohm/sq), the natural fit for OTA bias /
# compensation resistors (hundreds-to-thousands of ohms). Every value cites the real deck.
_GF180MCU_PASSIVE: Dict[str, object] = {
    "mim": _GF180MCU_MIM_OPTION_A,
    "mim_options": {
        "A": _GF180MCU_MIM_OPTION_A,
        "B": _GF180MCU_MIM_OPTION_B,
    },
    "res": _PolyResFacts(
        # sheet: gf180mcu.lvs extract_devices(resistor_with_bulk("ppolyf_u", 350, ...)).
        sheet_ohm_sq=350.0,
        sheet_provenance="GF180MCU gf180mcu.lvs resistor_with_bulk(\"ppolyf_u\", 350) ohm/sq",
        sheet_model="ppolyf_u (unsilicided P+ poly precision resistor)",
        # body = poly2 (30/0); salicide block = SAB (49/0); resistor marker = res_mk
        # (110/5); head implant = pplus (31/0); head contact = contact (33/0); head metal
        # = metal1 (34/0). (gf180mcu.lvs layer block + p+_poly_resistor.drc.)
        body_layer=(30, 0),          # poly2
        sab_layer=(49, 0),           # SAB salicide block
        marker_layer=(110, 5),       # res_mk poly-resistor marker
        implant_layer=(31, 0),       # pplus (P+ S/D implant over the heads)
        contact_layer=(33, 0),       # contact
        head_metal_layer=(34, 0),    # metal1
        body_min_w_um=0.80,          # PRES.1 P+ poly resistor min width 0.80um
        contact_size_um=0.22,        # CO.1 contact size 0.22um
        contact_space_um=0.25,       # CO.2 contact space 0.25um
        contact_enc_um=0.10,         # poly enclosure of contact (conservative)
        sab_over_body_um=0.28,       # PRES.6 SAB min enclosure of resistor 0.28um
        head_um=0.50,                # silicided head length (>= contact + 2*enc, clean)
        metal_min_w_um=0.23,         # M1 min width 0.23um
        # PRES.5 pplus min enclosure of the P+ poly resistor body 0.30um (subsumes
        # PP.9 pplus enclosure of the sab-covered poly2 strip 0.18um).
        implant_over_body_um=0.30,
        # PRES.7 contact-inside-resistor to SAB separation 0.22um (subsumes SB.4
        # SAB-to-contact 0.15um). Pushes the head contact fully clear of the SAB.
        sab_to_contact_um=0.22,
    ),
}


# open_pdk -- external_pdk 0.18 Mixed-Signal Enhanced (the 1P6M MIM10 sibling of
# open_pdk). Same external_pdk 0.18 MIM10 process facts: the MIM density (0.971
# fF/um^2), the MIM top plate (58) / M5 bottom (65) / M6 cover (66) / V5 via
# (74), the salicide block (48), and the poly-resistor marker (350) are
# IDENTICAL to the open_pdk row -- both are the same external_pdk 0.18 1P6M MIM10
# stack. Every value here is also cited to open_pdk's OWN external_drc LVS deck
# (pdks/.../external_drc/LVS/external_pdk.11_0.lvs): MIM 58 (lvs:480),
# M5 65 (lvs:485), M6 66 (lvs:486), V5 74 (lvs:498), SAB 48 (lvs:478). The MIM
# density 0.971 fF/um^2 is the external_pdk 0.18 MIM10 process value the sibling row
# traces (COX = 9.71E-04 F/m^2 == 0.971 fF/um^2); RESP1 is streamed on GDS 96/0
# and remapped to private work layer 960 by the LVS deck (lvs:512-513). This row
# is DATA (the external_pdk 0.18 MIM10 facts), not a
# new builder -- it reuses the same _MimCapFacts / _PolyResFacts the sibling uses.
_POLY_RES_MODEL_SHEETS = {
    "gf180mcu": {
        "npolyf_u": (310., "sm141064.ngspice:399 rsh_npolyf_u"),
        "ppolyf_u": (350., "GF180 foundry LVS ppolyf_u sheet resistance"),
    },
}



# open_pdk -- MIM density only (task B3: closes the step.py silent-swallow seam
# where the GENERIC equal-size cap-plate resolver, `passive_device_cell.
# mim_density_f_per_um2` -> `_resolve_mim_facts`, raised UnsupportedPdkFactError
# for open_pdk because this table carried no external_pdk row at all -- swallowed by
# step.py's `_grow_equal_size_cap_plates` try/except, so an `@equal_size` group of
# open_pdk MIM caps silently never grew). NO "res" row is added here: open_pdk's
# poly sheet-rho already resolves via the ADAPTER surface (`clay.pdk.open_pdk.
# ADAPTER.poly_sheet_rho_ohm_sq = 292.0`), which `passive_device_cell._poly_sheet_rho`
# consults BEFORE ever falling through to this table (adapter override wins) -- so
# a open_pdk "res" entry here would never be read and would just be a second,
# inert declaration of the same number.
#
# density: external_pdk external_model.scs line 16764 (the ``mimcap`` HSPICE subckt, verbatim):
#   ``+ cmim_para=(w * l ) * 1.025e-3 + 2 * (w + l)* 2.425E-10``
# ``w,l`` in um -> the first (area) term is pF, so 1.025e-3 pF/um^2 ==
# 1.025 fF/um^2 (the HF model, lines 3967/3991, agrees). This is the SAME density
# `clay.layout.passive_device_cell_tsmc.external_pdk()` already uses
# to size the REAL external_pdk MIM builder (`build_mim_capacitor_tsmc`) -- this row does
# NOT replace that builder (a dedicated CTM/METAL-stack builder stays canonical
# for open_pdk; `passive_cell.build_mim_cap` is never dispatched to open_pdk,
# see `passive_device_cell.build_capacitor_for_tech`); the two declarations are
# bound equal by `tests/test_pdk_fact_parity.py::
# test_tsmc_mim_density_two_entry_points_agree`.
#
# The layer / DRC-minima fields below are UNUSED in production for open_pdk (only
# `density_ff_um2` / `density_provenance` are ever read, via `_resolve_mim_facts`);
# they are populated -- rather than left as placeholders -- with the SAME cited
# open_pdk deck values `passive_device_cell_tsmc` uses (level 4, the
# external_drc-implemented default), so this row is honest data, not a dummy:
#   top/bot/via/cover layers   -- external_pdk.layermap lines 44/47/48/100 (CTM4=67:4,
#                                  METAL4=31:0, VIA45=32:0, METAL5=33:0)
#   plate_min_w_um   4.0   -- CTM.W.1 top-plate width >= 4um   (external_drc.drc:2830)
#   plate_max_w_um  30.0   -- CTM.I.1 top-plate width max 30um (external_drc.drc:2892;
#                              found during this task's pre-read -- not previously
#                              cited/enforced anywhere in the external_pdk builder)
#   plate_space_um   1.2   -- CTM.S.1 top-plate space >= 1.2um (external_drc.drc:2837)
#   bot_ext_um       0.4   -- CTM.E.3 bottom plate encloses CTM >= 0.4um (:2857)
#   via_size_um     0.36   -- VIA4.W.1 exact cut size 0.36x0.36um (external_drc.drc:1369)
#   via_space_um    0.35   -- VIA4.S.1 generic cut spacing >= 0.35um (external_drc.drc:1374)
#   via_on_mim_space_um 2.0 -- CTM.S.4 via-on-CTM spacing >= 2um  (external_drc.drc:2873)
#   via_to_plate_edge_um / mim_over_via_um  0.24 -- CTM.E.1 CTM encloses its via
#                              by >= 0.24um (external_drc.drc:2850); external_pdk cites ONE
#                              enclosure rule here where external_pdk/gf180 cite two
#                              distinct ones (MIM.12 / MIM.3), so both fields
#                              take the same number.
#   cover_ext_via_um 0.12  -- nearest cited via/metal enclosure floor in this deck
#                              (CTM.E.2, bottom-plate-encloses-via, external_drc.drc:2853)
#                              -- no distinct cover-metal-encloses-via rule is cited
#                              for open_pdk's CTM.* block; used as the closest real
#                              floor since this field is inert for external_pdk production.


_PASSIVE_FACTS: Dict[str, Dict[str, object]] = {
    "gf180mcu": _GF180MCU_PASSIVE,
}

# Documented fallback density / sheet for a PDK whose passive facts are not yet
# tabulated AND whose tech/adapter exposes none. Clearly labelled so the caller
# knows it is NOT a real per-PDK value. 1.0 fF/um^2 is a typical 0.18um MIM
# density order of magnitude; 100 ohm/sq a typical precision-poly order. Used
# ONLY when nothing real resolves; the CellGeometry.reason records this.
_DEFAULT_MIM_DENSITY_FF_UM2: float = 1.0
_DEFAULT_POLY_SHEET_OHM_SQ: float = 100.0


# --------------------------------------------------------------------------- #
# merge-clean gate (mirror of realize_render._merge_layer_rings discipline)
# --------------------------------------------------------------------------- #
def _merge_layer_rings(rings_um: List[_Ring]) -> List[_Ring]:
    """UNION a layer's centroid-relative um rings into an overlap-free hole-free set.

    Collapses redundant same-layer overlap to the exact union (a klayout boolean
    merge), so the emitted cell has NO same-layer self-overlap. A merged polygon
    with a hole is split into hole-free trapezoids (the hole is preserved as the
    gap between pieces, NEVER hull-filled). Raises ``AssertionError`` if the merged
    union area differs from the original union area by more than
    :data:`_UNION_AREA_EPS_UM2` -- we only remove overlap, never change the shape.
    """
    import klayout.db as kdb

    reg = kdb.Region()
    for ring in rings_um:
        pts = [kdb.Point(round(x / _DBU), round(y / _DBU)) for (x, y) in ring]
        if len(pts) >= 3:
            reg.insert(kdb.Polygon(pts))
    orig_union_area = reg.merged().area() * _DBU * _DBU

    reg.merge()
    out_rings: List[_Ring] = []
    for poly in reg.each():
        if poly.holes() == 0:
            out_rings.append([(pt.x * _DBU, pt.y * _DBU) for pt in poly.each_point_hull()])
        else:
            for sp in poly.decompose_trapezoids():
                out_rings.append([(pt.x * _DBU, pt.y * _DBU) for pt in sp.each_point()])

    chk = kdb.Region()
    for ring in out_rings:
        pts = [kdb.Point(round(x / _DBU), round(y / _DBU)) for (x, y) in ring]
        if len(pts) >= 3:
            chk.insert(kdb.Polygon(pts))
    merged_union_area = chk.merged().area() * _DBU * _DBU
    assert abs(merged_union_area - orig_union_area) <= _UNION_AREA_EPS_UM2, (
        "passive stamp merge changed the drawn shape: union area "
        f"{merged_union_area:.9f} != original {orig_union_area:.9f} um^2"
    )
    return out_rings


def _self_overlap_area_um2(rings_um: List[_Ring]) -> float:
    """Total same-layer self-overlap area (um^2) among a layer's rings: the sum of
    raw ring areas minus the merged-union area. 0.0 means no self-overlap (the
    cell is merge-clean on this layer). A pure measurement (no mutation)."""
    import klayout.db as kdb

    raw = 0.0
    reg = kdb.Region()
    for ring in rings_um:
        pts = [kdb.Point(round(x / _DBU), round(y / _DBU)) for (x, y) in ring]
        if len(pts) >= 3:
            poly = kdb.Polygon(pts)
            reg.insert(poly)
            raw += poly.area() * _DBU * _DBU
    union = reg.merged().area() * _DBU * _DBU
    return max(0.0, raw - union)


def merge_clean_report(geom: CellGeometry) -> Dict[_LayerKey, float]:
    """``{(layer,dt): self_overlap_area_um2}`` for every layer of a stamp.

    A stamp is MERGE-CLEAN iff every value is ~0. Used by the tests (and any
    caller) to PROVE the synthesised cell has no same-layer self-overlap on real
    klayout polygons -- the honest check, not an assertion the geometry is clean.
    """
    return {
        key: _self_overlap_area_um2(rings)
        for key, rings in geom.polys_by_layer.items()
    }


# --------------------------------------------------------------------------- #
# fact resolution (tech / adapter first, then tabulated real PDK, then default)
# --------------------------------------------------------------------------- #
def _resolve_mim_facts(
    tech_id: str,
    *,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> Tuple[_MimCapFacts, str]:
    """Resolve MIM-cap facts for ``tech_id``: tabulated real-PDK facts when known.

    Returns ``(facts, provenance)``. (The Clay tech/adapter surface exposes no MIM
    density or plate layers for the commercial PDKs, so there is no tech override
    to consult yet; if a future adapter publishes one this is where it would win.)
    Raises ``KeyError`` for a PDK with no tabulated facts (a caller can then decide
    to skip rather than fabricate cap geometry on unknown layers).
    """
    entry = _PASSIVE_FACTS.get(tech_id.lower(), {})
    facts = entry.get("mim")
    if mim_option is None and tech_id.lower() == "gf180mcu" and str(pex_backend or "").strip().casefold() == "magic":
        from clay.pdk.gf180mcu import resolve_gf180_mim_config

        mim_option, _, _ = resolve_gf180_mim_config(pex_backend=pex_backend)
    if mim_option is not None:
        if tech_id.lower() == "gf180mcu":
            from clay.pdk.gf180mcu import normalize_gf180_mim_option

            opt_key = normalize_gf180_mim_option(mim_option)
        else:
            opt_key = str(mim_option).strip().upper()
        options_map = entry.get("mim_options")
        if isinstance(options_map, dict):
            if opt_key not in options_map:
                raise ValueError(
                    f"{tech_id!r}: unsupported mim_option {mim_option!r}; "
                    f"expected one of {sorted(options_map)}"
                )
            facts = options_map[opt_key]
        elif opt_key != "A":
            raise ValueError(
                f"{tech_id!r}: mim_option={mim_option!r} is only defined for "
                "multi-option MIM PDKs (e.g. gf180mcu)"
            )
    if facts is None:
        raise KeyError(
            f"no MIM-cap process facts tabulated for {tech_id!r}; passive_cell "
            f"knows: {sorted(k for k, v in _PASSIVE_FACTS.items() if 'mim' in v)}"
        )
    if cap_option is not None and tech_id.lower() == "gf180mcu":
        cap_key = str(cap_option).strip()
        if cap_key not in _GF180MCU_MIM_DENSITIES_FF_UM2:
            raise ValueError(
                f"gf180mcu: unsupported cap_option {cap_option!r}; "
                f"expected one of {sorted(_GF180MCU_MIM_DENSITIES_FF_UM2)}"
            )
        from dataclasses import replace

        density = _GF180MCU_MIM_DENSITIES_FF_UM2[cap_key]
        facts = replace(
            facts,
            density_ff_um2=density,
            density_provenance=(
                f"{facts.density_provenance.rsplit(' ', 2)[0]} {density:.1f} fF/um^2"
            ),
        )
    return facts, "real-pdk-deck"  # type: ignore[return-value]


def _resolve_poly_res_facts(tech_id: str) -> Tuple[_PolyResFacts, str]:
    """Resolve poly-resistor facts for ``tech_id``.

    If the registered PDK adapter publishes ``poly_sheet_rho_ohm_sq`` that REAL
    value overrides the tabulated sheet (provenance ``tech-adapter``); the layers +
    DRC minima still come from the tabulated real-PDK facts (the adapter does not
    carry them). Raises ``KeyError`` for a PDK with no tabulated facts.
    """
    facts = _PASSIVE_FACTS.get(tech_id.lower(), {}).get("res")
    if facts is None:
        raise KeyError(
            f"no poly-resistor process facts tabulated for {tech_id!r}; "
            f"passive_cell knows: "
            f"{sorted(k for k, v in _PASSIVE_FACTS.items() if 'res' in v)}"
        )
    assert isinstance(facts, _PolyResFacts)
    # adapter override for the sheet rho (the one passive value the adapter CAN
    # carry): a real published value wins over the tabulated default.
    try:
        from clay.pdk import get_pdk

        adapter = get_pdk(tech_id)
        rho = getattr(adapter, "poly_sheet_rho_ohm_sq", None)
        if rho:
            from dataclasses import replace

            facts = replace(
                facts,
                sheet_ohm_sq=float(rho),
                sheet_provenance=f"{tech_id} adapter poly_sheet_rho_ohm_sq={rho}",
                sheet_model=getattr(adapter, "poly_res_model", facts.sheet_model)
                or facts.sheet_model,
            )
            return facts, "tech-adapter"
    except Exception:
        pass
    return facts, "real-pdk-deck"


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _snap(v_um: float) -> float:
    """Round a um coordinate to the 1 nm database grid (avoids float drift)."""
    return round(round(v_um / _DBU) * _DBU, 4)


def _rect_centred(cx: float, cy: float, w: float, h: float) -> _Ring:
    """An axis-aligned rectangle ring centred at (cx, cy) with width w, height h."""
    x0, x1 = _snap(cx - w / 2.0), _snap(cx + w / 2.0)
    y0, y1 = _snap(cy - h / 2.0), _snap(cy + h / 2.0)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _via_array(
    cx: float,
    cy: float,
    field_w: float,
    field_h: float,
    via: float,
    pitch: float,
) -> List[_Ring]:
    """A centred grid of ``via``-sized square cuts on ``pitch``, filling a field of
    ``field_w`` x ``field_h`` (centred at cx, cy). At least one cut. Cuts are
    placed so the array is centred and every cut sits inside the field."""
    import math

    def _count(span: float) -> int:
        usable = span - via
        if usable <= 0:
            return 1
        return max(1, int(math.floor(usable / pitch)) + 1)

    nx = _count(field_w)
    ny = _count(field_h)
    out: List[_Ring] = []
    x_span = (nx - 1) * pitch
    y_span = (ny - 1) * pitch
    for ix in range(nx):
        for iy in range(ny):
            vx = cx - x_span / 2.0 + ix * pitch
            vy = cy - y_span / 2.0 + iy * pitch
            out.append(_rect_centred(vx, vy, via, via))
    return out


def _recentre(polys_by_layer: Dict[_LayerKey, List[_Ring]]) -> Dict[_LayerKey, List[_Ring]]:
    """Translate every ring so the cell's overall bbox centre is at (0, 0): the
    CellGeometry stamp contract (a stamp has centroid (0,0), carries no position)."""
    xs = [x for rings in polys_by_layer.values() for ring in rings for (x, _y) in ring]
    ys = [y for rings in polys_by_layer.values() for ring in rings for (_x, y) in ring]
    if not xs:
        return polys_by_layer
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    return {
        key: [[(_snap(x - cx), _snap(y - cy)) for (x, y) in ring] for ring in rings]
        for key, rings in polys_by_layer.items()
    }


def _finish(
    polys_by_layer: Dict[_LayerKey, List[_Ring]],
    *,
    kind: str,
    reason: str,
) -> CellGeometry:
    """Merge-clean every layer, recentre to centroid (0,0), and pack into a
    ``CellGeometry`` stamp tagged ``source='passive_synth'``."""
    merged: Dict[_LayerKey, List[_Ring]] = {}
    for key, rings in polys_by_layer.items():
        out_rings = _merge_layer_rings(rings)
        if out_rings:
            merged[key] = out_rings
    merged = _recentre(merged)

    cg = CellGeometry(
        name="",
        kind=kind,
        ref_centroid=(0.0, 0.0),  # a stamp has no reference position
        polys_by_layer=merged,
        n_instances=1,
        finger_cells=[],
        clean=True,
        reason=reason,
    )
    cg.source = "passive_synth"
    cg.merged_clean = True
    return cg


def _mim_bank_ties(
    facts: _MimCapFacts,
    centres: List[Tuple[float, float]],
    bot_w: float,
    bot_h: float,
    cover_w: float,
    cover_h: float,
    polys: Dict[_LayerKey, List[_Ring]],
) -> None:
    """Tie a banked cap's unit plates in PARALLEL so the bank is 2 nodes, not 2*N.

    Each unit plate stack drawn by :func:`_mim_unit_plate` is electrically isolated
    from its neighbours (the MIM.2 plate-to-plate space leaves a gap between every
    unit's bottom plate and between every unit's cover metal). A real banked MIM
    cap wires all top plates together and all bottom plates together. Here we draw,
    on the SAME real layers the unit stacks already use:

      * BOTTOM tie -- bars on the bottom-plate metal (``facts.bot_plate_layer``,
        external_pdk M5) that span every unit centre, so all bottom plates merge into ONE
        same-layer island (one bottom node).
      * TOP tie -- bars on the cover metal (``facts.cover_layer``, external_pdk M6) that
        span every unit centre, so all cover patches (each already joined to its
        MIM top plate by the V5 via array) merge into ONE same-layer island (one
        top node).

    The ties are drawn as a row-spine + column-spine grid through the unit centres:
    one horizontal bar per row (spanning that row's units) plus one vertical bar
    down a column (joining the rows). Each bar is centred on the unit centres it
    crosses and overlaps each unit's metal on that layer, so the same-layer merge
    unions them. No via is added (the tie is intra-layer); the two plates stay on
    two nodes because nothing joins the M5 bottom tie to the MIM/M6 top tie.
    """
    if len(centres) <= 1:
        return
    xs = sorted({cx for (cx, _cy) in centres})
    ys = sorted({cy for (_cx, cy) in centres})

    def _ties_on_layer(layer: _LayerKey, patch_w: float, patch_h: float) -> None:
        # bus widths: a fraction of the patch dimension so a bar through the unit
        # centres lands well inside each patch (overlaps it) -- the merge then ties
        # them. Bounded below so a small patch still gets a finite, snappable bar.
        bus_h = _snap(max(facts.via_size_um, patch_h * 0.5))
        bus_w = _snap(max(facts.via_size_um, patch_w * 0.5))
        by_row: Dict[float, List[float]] = {}
        for (cx, cy) in centres:
            by_row.setdefault(cy, []).append(cx)
        # one horizontal bar per row, spanning that row's unit centres.
        for cy, row_xs in by_row.items():
            x0, x1 = min(row_xs), max(row_xs)
            if x1 > x0:
                bar_w = _snap((x1 - x0) + patch_w)
                polys.setdefault(layer, []).append(
                    _rect_centred(_snap((x0 + x1) / 2.0), _snap(cy), bar_w, bus_h)
                )
        # one vertical bar joining the rows (down the first column that exists in
        # every row -- the leftmost x present, which each row's horizontal bar
        # reaches because every horizontal bar starts at its row's min x).
        if len(ys) > 1:
            spine_x = xs[0]
            y0, y1 = ys[0], ys[-1]
            bar_h = _snap((y1 - y0) + patch_h)
            polys.setdefault(layer, []).append(
                _rect_centred(_snap(spine_x), _snap((y0 + y1) / 2.0), bus_w, bar_h)
            )

    _ties_on_layer(facts.bot_plate_layer, bot_w, bot_h)   # bottom node tie (M5)
    _ties_on_layer(facts.cover_layer, cover_w, cover_h)    # top node tie (M6)


def _mim_unit_plate(
    facts: _MimCapFacts,
    cx: float,
    cy: float,
    w: float,
    h: float,
    polys: Dict[_LayerKey, List[_Ring]],
) -> None:
    """Append ONE MIM unit-plate stack (top plate + bottom plate + top-via array +
    cover metal) centred at ``(cx, cy)`` of top-plate size ``w x h`` to ``polys``.

    The full cap is one or more of these unit plates tiled into a bank (a big cap
    exceeds the foundry single-plate max, MIM.7/MIM.8, so a real flow banks unit
    plates wired in parallel). Each unit is DRC-legal on its own; the bank spacing
    (MIM.2) is handled by the tiler that calls this."""
    # MIM top plate.
    polys.setdefault(facts.top_plate_layer, []).append(_rect_centred(cx, cy, w, h))
    for marker_layer in facts.marker_layers:
        mw = _snap(w + 2.0 * facts.marker_ext_um)
        mh = _snap(h + 2.0 * facts.marker_ext_um)
        polys.setdefault(marker_layer, []).append(_rect_centred(cx, cy, mw, mh))
    # bottom plate: top plate grown by the bottom-extension rule on every side.
    bw = _snap(w + 2.0 * facts.bot_ext_um)
    bh = _snap(h + 2.0 * facts.bot_ext_um)
    polys.setdefault(facts.bot_plate_layer, []).append(_rect_centred(cx, cy, bw, bh))
    # top-via array on the plate, inset by the via-to-edge + MIM-over-via margins.
    # Vias landing ON the MIM plate use the MIM-specific via-on-plate spacing
    # (MIM.13), which is much larger than the generic via spacing, so the via field
    # is a sparse grid, not a dense one.
    inset = max(facts.via_to_plate_edge_um, facts.mim_over_via_um)
    field_w = max(facts.via_size_um, w - 2.0 * inset)
    field_h = max(facts.via_size_um, h - 2.0 * inset)
    pitch = facts.via_size_um + facts.via_on_mim_space_um
    polys.setdefault(facts.via_layer, []).extend(
        _via_array(cx, cy, field_w, field_h, facts.via_size_um, pitch)
    )
    # cover metal over the via array (grown by the cover-extension rule).
    cover_w = _snap(field_w + facts.via_size_um + 2.0 * facts.cover_ext_via_um)
    cover_h = _snap(field_h + facts.via_size_um + 2.0 * facts.cover_ext_via_um)
    polys.setdefault(facts.cover_layer, []).append(_rect_centred(cx, cy, cover_w, cover_h))


# --------------------------------------------------------------------------- #
# public builders
# --------------------------------------------------------------------------- #
def build_mim_cap(
    tech_id: str,
    value_farad: float,
    *,
    aspect: float = 1.0,
    mim_option: Optional[str] = None,
    cap_option: Optional[str] = None,
    pex_backend: Optional[str] = None,
) -> CellGeometry:
    """Synthesise a native MIM capacitor :class:`CellGeometry` for ``tech_id``.

    The plate AREA is computed from the requested capacitance and the PDK's REAL
    MIM density: ``area_um2 = value_farad / (density_fF_um2 * 1e-15)``. The plate is
    drawn (with the requested ``aspect`` = width/height) on the PDK's OWN MIM top
    plate layer, with a bottom plate extended by the foundry bottom-extension rule,
    a top-plate-to-cover-metal via ARRAY (sized + spaced + edge-cleared by the real
    MIM via rules), and a cover-metal patch over the via array. Coordinates are
    centroid-relative um; the stamp's centroid is (0, 0).

    Parameters
    ----------
    tech_id : str
        Target PDK id (e.g. ``"open_pdk"``).
    value_farad : float
        The capacitance in Farads (must be > 0).
    aspect : float
        Plate width / height ratio (default 1.0 = square). Clamped so neither plate
        dimension falls below the foundry MIM min width or rises above its max.

    Returns
    -------
    CellGeometry
        ``source='passive_synth'``, ``clean=True``, merge-clean, on the TARGET PDK's
        MIM / metal / via layers. ``reason`` records the density provenance + the
        computed plate area.

    Raises
    ------
    ValueError
        If ``value_farad`` is not positive.
    KeyError
        If the PDK has no tabulated MIM facts (the caller can skip rather than
        fabricate cap geometry on unknown layers).
    """
    if not (value_farad > 0.0):
        raise ValueError(f"value_farad must be > 0, got {value_farad!r}")
    import math

    facts, provenance = _resolve_mim_facts(
        tech_id, mim_option=mim_option, cap_option=cap_option, pex_backend=pex_backend
    )

    density = facts.density_ff_um2
    used_default = False
    if not (density > 0.0):
        density = _DEFAULT_MIM_DENSITY_FF_UM2
        used_default = True

    # plate area (um^2) = C / density.  density fF/um^2 -> F/um^2 = *1e-15.
    area_um2 = float(value_farad) / (density * 1e-15)

    # A single MIM plate is bounded to the foundry max (MIM.7) and min (MIM.1). If
    # the requested area fits in one plate we draw one plate of exactly that area
    # (with the requested aspect). If it exceeds the single-plate max area we BANK
    # identical unit plates wired in parallel (a real flow does this -- MIM.8 says a
    # cap larger than the single-plate max MUST be a combination of smaller caps),
    # so the TOTAL drawn plate area equals the requested area (it scales with C and
    # matches the density) instead of silently clamping to one over-max plate.
    max_w = facts.plate_max_w_um
    min_w = facts.plate_min_w_um
    single_max_area = max_w * max_w

    polys: Dict[_LayerKey, List[_Ring]] = {}

    if area_um2 <= single_max_area:
        # one plate of exactly the requested area, at the requested aspect, each
        # dimension clamped into [min_w, max_w].
        h = math.sqrt(area_um2 / max(aspect, 1e-9))
        w = area_um2 / h if h > 0 else math.sqrt(area_um2)
        w = min(max(w, min_w), max_w)
        h = min(max(h, min_w), max_w)
        w, h = _snap(w), _snap(h)
        _mim_unit_plate(facts, 0.0, 0.0, w, h, polys)
        n_units = 1
        unit_w, unit_h = w, h
        drawn_area = w * h
    else:
        # bank N identical square unit plates wired in parallel whose TOTAL area
        # equals the requested area. N is the fewest plates that keeps each plate's
        # side within the foundry band [min_w, max_w]: N = ceil(area / max_area),
        # and each plate side = sqrt(area / N) (so N*side^2 == area, area-exact to
        # grid snap -- the bank scales with C and matches the density, not a coarse
        # unit-quantized over-area). Tiled in the most-square grid with the MIM.2
        # plate-to-plate spacing between plates.
        n_units = max(1, math.ceil(area_um2 / single_max_area))
        side = math.sqrt(area_um2 / n_units)
        # if the per-plate side fell below the foundry min, drop to fewer plates so
        # each stays legal (rare: only for caps barely over the single-plate max).
        while n_units > 1 and side < min_w:
            n_units -= 1
            side = math.sqrt(area_um2 / n_units)
        side = min(max(side, min_w), max_w)
        unit = _snap(side)
        ncol = max(1, math.ceil(math.sqrt(n_units)))
        nrow = max(1, math.ceil(n_units / ncol))
        pitch = unit + facts.plate_space_um
        x_span = (ncol - 1) * pitch
        y_span = (nrow - 1) * pitch
        placed = 0
        centres: List[Tuple[float, float]] = []
        for r in range(nrow):
            for c in range(ncol):
                if placed >= n_units:
                    break
                cx = _snap(-x_span / 2.0 + c * pitch)
                cy = _snap(-y_span / 2.0 + r * pitch)
                _mim_unit_plate(facts, cx, cy, unit, unit, polys)
                centres.append((cx, cy))
                placed += 1
        # wire all top plates together (on the M6 cover) and all bottom plates
        # together (on M5) so the bank is ONE top node + ONE bottom node = 2 nodes,
        # not 2*N disjoint nodes. Bottom plate / cover metal extents (per unit) mirror
        # _mim_unit_plate's geometry, recomputed here to size the tie bars.
        bot_dim = _snap(unit + 2.0 * facts.bot_ext_um)
        inset = max(facts.via_to_plate_edge_um, facts.mim_over_via_um)
        field = max(facts.via_size_um, unit - 2.0 * inset)
        cover_dim = _snap(field + facts.via_size_um + 2.0 * facts.cover_ext_via_um)
        _mim_bank_ties(facts, centres, bot_dim, bot_dim, cover_dim, cover_dim, polys)
        unit_w = unit_h = unit
        drawn_area = n_units * unit * unit

    dens_note = (
        f"{density:.4g} fF/um^2 ({facts.density_provenance})"
        if not used_default
        else f"{density:.4g} fF/um^2 (DOCUMENTED DEFAULT -- no real density resolved)"
    )
    bank_note = (
        f"single plate {unit_w:.3f}x{unit_h:.3f} um"
        if n_units == 1
        else f"bank of {n_units} x {unit_w:.3f}x{unit_h:.3f} um unit plates"
    )
    reason = (
        f"MIM cap {value_farad:.4g} F @ density {dens_note}; {bank_note}; "
        f"total plate area {drawn_area:.4f} um^2 (requested {area_um2:.4f}); "
        f"layers top={facts.top_plate_layer} bot={facts.bot_plate_layer} "
        f"via={facts.via_layer} cover={facts.cover_layer}; "
        f"provenance={'documented-default' if used_default else provenance}; "
        f"merged-clean"
    )
    cg = _finish(polys, kind="cap", reason=reason)
    cg.density_ff_um2 = density
    cg.density_provenance = "documented-default" if used_default else provenance
    cg.plate_area_um2 = round(drawn_area, 6)
    cg.requested_area_um2 = round(area_um2, 6)
    cg.n_unit_plates = n_units
    return cg


def build_poly_res(
    tech_id: str,
    value_ohm: float,
    *,
    width_um: Optional[float] = None,
) -> CellGeometry:
    """Synthesise a native poly resistor :class:`CellGeometry` for ``tech_id``.

    The body LENGTH is computed from the requested resistance, the PDK's REAL poly
    sheet resistance, and the body width: ``length_um = value_ohm / sheet * width``
    (i.e. ``squares = value / sheet`` and ``length = squares * width``). The body is
    drawn on the PDK's OWN poly layer, covered by the salicide-block layer (the
    non-silicide resistor body) extended by the foundry SAB rule, with the resistor
    marker over it, two SILICIDED HEADS (no SAB) at the ends carrying a head-contact
    array + head metal + the head S/D implant. Coordinates are centroid-relative um.

    Parameters
    ----------
    tech_id : str
        Target PDK id (e.g. ``"open_pdk"``).
    value_ohm : float
        The resistance in ohms (must be > 0).
    width_um : float, optional
        The body width (default: the foundry poly min width). Wider bodies give a
        shorter bar for the same resistance. Clamped up to the min width.

    Returns
    -------
    CellGeometry
        ``source='passive_synth'``, ``clean=True``, merge-clean, on the TARGET PDK's
        poly / SAB / contact / metal layers. ``reason`` records the sheet provenance
        + the computed body length + #squares.

    Raises
    ------
    ValueError
        If ``value_ohm`` is not positive.
    KeyError
        If the PDK has no tabulated poly-resistor facts.
    """
    if not (value_ohm > 0.0):
        raise ValueError(f"value_ohm must be > 0, got {value_ohm!r}")

    facts, provenance = _resolve_poly_res_facts(tech_id)

    sheet = facts.sheet_ohm_sq
    used_default = False
    if not (sheet > 0.0):
        sheet = _DEFAULT_POLY_SHEET_OHM_SQ
        used_default = True

    width = facts.body_min_w_um if width_um is None else max(float(width_um), facts.body_min_w_um)
    width = _snap(width)

    # number of squares + the body length that realises the resistance.
    squares = float(value_ohm) / sheet
    # The EXTRACTED resistive length is the SALICIDE-BLOCK / marker-covered poly length
    # (the unsilicided span between the two silicided heads), which is the drawn body bar
    # PLUS the SAB extension (``sab_over_body_um`` on each end). So to make the extracted
    # R == value, the DRAWN body bar is shortened by ``2 * sab_over_body_um`` -- then the
    # marker-covered span equals ``squares * width`` and KLayout extracts R = value.
    target_marker_len = squares * width
    length = _snap(max(target_marker_len - 2.0 * facts.sab_over_body_um,
                       facts.body_min_w_um))

    # head length: long enough to land a head contact with the poly enclosure on
    # both sides (and >= the body min width). The heads are SILICIDED (no SAB).
    min_head = max(facts.head_um, facts.contact_size_um + 2.0 * facts.contact_enc_um)
    head_w = _snap(max(width, facts.contact_size_um + 2.0 * facts.contact_enc_um))

    # contact placement: when the PDK declares a SAB-to-contact clearance, the head
    # contact's body-side edge must sit that far BEYOND the SAB outer edge (which is
    # body_edge + sab_over_body), so the contact does not straddle the SAB (PRES.7 /
    # SB.4). The head is then lengthened to enclose the pushed-out contact with the
    # poly enclosure. clearance 0 => legacy contact centred in the head.
    contact_clear = facts.sab_to_contact_um
    if contact_clear > 0.0:
        contact_off = (facts.sab_over_body_um + contact_clear
                       + facts.contact_size_um / 2.0)
        head = _snap(max(min_head,
                         contact_off + facts.contact_size_um / 2.0
                         + facts.contact_enc_um))
    else:
        contact_off = None
        head = _snap(min_head)

    # the full poly bar spans head + body + head along x; body width along y.
    total_len = _snap(length + 2.0 * head)
    body_x0 = -length / 2.0
    body_x1 = length / 2.0
    left_head_cx = _snap(body_x0 - head / 2.0)
    right_head_cx = _snap(body_x1 + head / 2.0)
    # contact centres: pushed out by ``contact_off`` past the body edge when a SAB
    # clearance is declared, else the head centre (legacy).
    if contact_off is not None:
        left_contact_cx = _snap(body_x0 - contact_off)
        right_contact_cx = _snap(body_x1 + contact_off)
    else:
        left_contact_cx, right_contact_cx = left_head_cx, right_head_cx

    polys: Dict[_LayerKey, List[_Ring]] = {}

    # poly body bar (centred): the body region width, the heads at the head width.
    # Draw as three abutting rectangles (left head / body / right head) on the poly
    # layer; the merge unions them into one clean poly bar.
    polys.setdefault(facts.body_layer, []).append(_rect_centred(0.0, 0.0, length, width))
    polys[facts.body_layer].append(_rect_centred(left_head_cx, 0.0, head, head_w))
    polys[facts.body_layer].append(_rect_centred(right_head_cx, 0.0, head, head_w))

    # salicide block (SAB) over the BODY only (not the heads -- heads must silicide
    # to take contacts), extended beyond the body by the SAB rule.
    sab_len = _snap(length + 2.0 * facts.sab_over_body_um)
    sab_w = _snap(width + 2.0 * facts.sab_over_body_um)
    polys.setdefault(facts.sab_layer, []).append(_rect_centred(0.0, 0.0, sab_len, sab_w))

    # resistor-recognition marker over the BODY only (coincident with the SAB region,
    # NOT the heads). The LVS deck reads the unsilicided body as ``poly & sab & res_mk``
    # and the contactable terminals as ``poly NOT res_mk`` (the heads) -- so the marker
    # MUST stop at the body, or the heads have no terminal poly and the R cannot extract.
    if facts.marker_layer is not None:
        polys.setdefault(facts.marker_layer, []).append(
            _rect_centred(0.0, 0.0, sab_len, sab_w)
        )

    # head S/D implant over the heads + body. It MUST enclose the resistor body
    # (pres_poly = poly2&sab&res_mk) and the sab-covered poly2 strip by the foundry
    # implant-enclosure rule (PRES.5 / PP.9), so the implant rect is the full poly
    # bar bbox GROWN by ``implant_over_body_um`` on every side (0 => legacy bbox).
    if facts.implant_layer is not None:
        io = facts.implant_over_body_um
        polys.setdefault(facts.implant_layer, []).append(
            _rect_centred(0.0, 0.0,
                          _snap(total_len + 2.0 * io),
                          _snap(head_w + 2.0 * io))
        )

    # head contacts + head metal at each head (at the contact centre, which the SAB
    # clearance may push outboard of the head geometric centre).
    for hx in (left_contact_cx, right_contact_cx):
        polys.setdefault(facts.contact_layer, []).append(
            _rect_centred(hx, 0.0, facts.contact_size_um, facts.contact_size_um)
        )
        m_w = _snap(max(facts.metal_min_w_um, facts.contact_size_um + 2.0 * facts.contact_enc_um))
        polys.setdefault(facts.head_metal_layer, []).append(
            _rect_centred(hx, 0.0, m_w, m_w)
        )

    sheet_note = (
        f"{sheet:.4g} ohm/sq ({facts.sheet_provenance})"
        if not used_default
        else f"{sheet:.4g} ohm/sq (DOCUMENTED DEFAULT -- no real sheet resolved)"
    )
    reason = (
        f"poly resistor {value_ohm:.4g} ohm @ sheet {sheet_note} [{facts.sheet_model}]; "
        f"W={width:.3f} um, {squares:.3f} squares, body L={length:.3f} um; "
        f"layers body={facts.body_layer} sab={facts.sab_layer} "
        f"contact={facts.contact_layer} metal={facts.head_metal_layer}; "
        f"provenance={'documented-default' if used_default else provenance}; "
        f"merged-clean"
    )
    cg = _finish(polys, kind="res", reason=reason)
    cg.sheet_ohm_sq = sheet
    cg.sheet_provenance = "documented-default" if used_default else provenance
    cg.body_length_um = length
    cg.body_width_um = width
    cg.squares = round(squares, 6)
    return cg


__all__ = [
    "build_mim_cap",
    "build_poly_res",
    "merge_clean_report",
]
