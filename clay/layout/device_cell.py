"""PHASE P1 -- PDK-parameterized device-cell builder (Clay intent-driven P&R).

``build_device(tech, kind, W, L, nf, mirror=, axis_x=)`` builds ONE DRC-legal,
LVS-foldable m=N MOS for ANY PDK whose facts a :class:`~clay.layout.tech.LayoutTech`
aggregates -- replacing the gf180-only ``coclean_5tota_v163.build_foldable_device``
with a geometry that reads EVERY layer key + DRC dimension from ``tech``.

What this closes
================
The old builder hard-coded gf180 GDS keys (``GF180_*_KEY``) and gf180 rule
constants (``GF180_NPLUS_GATE_OVERLAP_NM`` ...). On open_pdk it emitted nothing
usable -- "0 devices / no implant layers" -- because the implant + well layers
and their rule values were never resolved from the PDK. ``build_device`` resolves
them all through ``tech``:

  * layer keys -- ``tech.build_layer("ndiff"/"poly"/"contact"/"metal1"/...)``,
    ``tech.contact_key()``, ``tech.metal_key(1)``, ``tech.label_key(1)``;
  * DRC dimensions -- ``tech.implant_well()`` (gate-implant overlap, nwell-pcomp /
    nwell-ncomp enclosure), ``tech.enclosure("contact", "comp"/"metal1")``,
    ``tech.via_size("contact")`` / ``tech.via_spacing("contact")``,
    ``tech.min_width("metal1"/"poly2"/"nwell")``;
  * bulk topology -- ``tech.bulk_topology()`` (a DATA branch:
    ``implant_distinguished`` PDKs draw Nplus/Pplus over the gate, whereas a
    ``diffusion_distinguished`` PDK like sky130 draws ndiff/pdiff active instead),
    ``tech.substrate_net()`` (the NMOS P+ substrate-tap net name).

Geometry is the SAME recipe as ``build_foldable_device`` (R162 foldable MOS):
folded poly fingers, every S/D contact on the COMP, commoned alternating S/D M1
rails, a poly gate bus + one gate contact, the gate implant grown over the gate,
and for a PMOS an enclosing Nwell + an in-well N+ well-tap tied to the bulk net.

Terminal reflection (the P1 crux)
=================================
:class:`DeviceGeom` carries a TERMINAL MAP ``{'s','d','g','b' -> bbox}`` with the
device's source / drain / gate / bulk terminals. When a member is placed
mirrored, ``build_device(mirror=True, axis_x=...)`` returns geometry AND terminals
already REFLECTED about ``axis_x`` -- so the *physical column* carrying ``'s'`` vs
``'d'`` is SWAPPED, exactly as the hand router does (``left`` then
``_mirror_x(left, axis)`` on real polygons, ``comparator_router``). The legacy
placer only TRANSLATED a mirrored member, leaving its source/drain rails at the
SAME offsets as the unmirrored partner, so the schematic s/d net attached to the
WRONG physical diffusion column and the extracted S/D diverged. Consuming this
reflected terminal map fixes that connectivity mismatch.

Parity contract (the safety oracle)
====================================
For gf180mcu ``build_device`` is BYTE-IDENTICAL to ``build_foldable_device``:
same polygons, same layer keys, same integer coordinates. The 5 proven gf180 L5
co-clean cells depend on that parity; ``tests/test_device_cell_pdk.py`` is the
oracle. ``build_foldable_device`` delegates here, so gf180 has exactly one
geometry source of truth.

Pure-function core
==================
The inner geometry (``_build_device_geom``) is a CLEAN PURE FUNCTION: integer
coordinates in, polygon + terminal data out, no I/O, no PDK lookups (every PDK
fact arrives pre-resolved in :class:`DeviceCellRules` + a :class:`_LayerKeys`).
A future Rust kernel (``_ir_bridge_build_device_geom_rs``, DEFERRED) can mirror it
byte-for-byte; this module keeps the byte-equal Python path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .backend import LayerPolygon, Point, Polygon
from .capacitor_unit_occurrences import CapacitorUnitOccurrenceGeometry
LayerKey = Tuple[int, int]
Bbox = Tuple[int, int, int, int]


# --- substrate/ground rail vocabulary ----------------------------------------
# The SAME classification the router's pin extractor uses
# (clay.layout.route.pins._POWER_BASES): a global supply/ground net,
# case-insensitive, with or without a trailing '!' global mark. The NMOS P+
# substrate tap ties the device body into the deck's ONE global substrate net
# (connect_global(ptap, substrate)); it is drawn ONLY when the declared body net
# IS such a rail. A private signal net (e.g. a source-degenerated NMOS body) is
# never tied down to the substrate. Kept as a small self-contained constant so
# this pure device-cell core takes no dependency on the router package.
_SUBSTRATE_RAIL_BASES = frozenset(
    {"vdd", "vss", "gnd", "vcc", "vee", "vpwr", "vgnd", "vnb", "vpb",
     "avdd", "avss"}
)


def _net_is_substrate_rail(
    net: str, declared_rails: Sequence[str] = ()
) -> bool:
    """True iff ``net`` is a recognized global supply/ground rail (the substrate
    an NMOS bulk tap ties into) -- mirrors the router's power-net vocabulary so
    the device cell and the pin extractor agree on rail vs private signal net."""
    if not net:
        return False
    base = net.strip().lower()
    if base.endswith("!"):
        base = base[:-1]
    declared = {
        str(item).strip().lower().rstrip("!")
        for item in declared_rails
        if str(item).strip()
    }
    return base in _SUBSTRATE_RAIL_BASES or base in declared


# --- pure geometry helpers (integer-only, no I/O) ----------------------------


def _rect(layer_key: LayerKey, box: Bbox) -> LayerPolygon:
    x0, y0, x1, y1 = (int(v) for v in box)
    return LayerPolygon(
        layer=layer_key[0],
        datatype=layer_key[1],
        polygon=Polygon.rectangle(x0, y0, x1, y1),
    )


def _bbox(lp: LayerPolygon) -> Bbox:
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _reflect_box_x(box: Bbox, axis_x: int) -> Bbox:
    """Reflect an axis-aligned bbox about the vertical line ``x = axis_x``.

    ``(x0, y0, x1, y1)`` -> ``(2*axis - x1, y0, 2*axis - x0, y1)``. The x-min/x-max
    swap is what makes a mirrored device's left column land where the partner's
    right column is -- so the physical column carrying 's' vs 'd' flips."""
    return (2 * axis_x - box[2], box[1], 2 * axis_x - box[0], box[3])


def _reflect_poly_x(lp: LayerPolygon, axis_x: int) -> LayerPolygon:
    """Reflect one polygon about ``x = axis_x`` (mirror of ``backend._mirror_x``
    for a single shape: negate x about the axis, reverse vertex winding)."""
    return LayerPolygon(
        layer=lp.layer,
        datatype=lp.datatype,
        polygon=Polygon(
            vertices=list(
                reversed([Point(2 * axis_x - v.x, v.y) for v in lp.polygon.vertices])
            )
        ),
    )


# --- device orientation (R0/R90/R180/R270) -----------------------------------
#
# A device is built ONCE at R0 (vertical-gate orientation) by the pure geometry
# core; ``orient`` rotates the WHOLE device about the origin by a 90-degree
# multiple, then normalizes it back into the first quadrant (so the device's
# lower-left corner lands where the R0 device's did). This is a clean affine
# transform applied to EVERY emitted polygon + EVERY feature bbox AFTER the
# unrotated device is built -- so an R90 device has its width/height swapped and
# its terminal map rotated consistently, while connectivity (the net each shape
# carries) is UNCHANGED. Rotation of an axis-aligned rectilinear device about a
# 90-degree multiple maps integer-on-grid coordinates to integer-on-grid
# coordinates and preserves every same-layer width + spacing, so the rotated
# device stays DRC-legal by construction (the grid re-snap below is a defensive
# guard, never a fix).

_ORIENTS = ("R0", "R90", "R180", "R270")


def _rotate_point(x: int, y: int, orient: str) -> Tuple[int, int]:
    """Rotate ``(x, y)`` about the origin by the 90-degree multiple ``orient``
    (counter-clockwise). R0 is identity; R90 (x,y)->(-y,x); R180 (x,y)->(-x,-y);
    R270 (x,y)->(y,-x). Integer-in / integer-out (the device coords are nm ints)."""
    if orient == "R0":
        return (x, y)
    if orient == "R90":
        return (-y, x)
    if orient == "R180":
        return (-x, -y)
    if orient == "R270":
        return (y, -x)
    raise ValueError(f"orient must be one of {_ORIENTS!r}, got {orient!r}")


def _rotate_box(box: Bbox, orient: str) -> Bbox:
    """Rotate an axis-aligned bbox about the origin by ``orient`` and renormalise
    to ``(x0<=x1, y0<=y1)``. The corners rotate to corners; the result is the
    bbox of the four rotated corners (for a 90-degree multiple this is exact, no
    area change -- a width becomes a height for R90/R270)."""
    x0, y0, x1, y1 = box
    pts = [
        _rotate_point(x0, y0, orient),
        _rotate_point(x1, y0, orient),
        _rotate_point(x1, y1, orient),
        _rotate_point(x0, y1, orient),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _rotate_poly(lp: LayerPolygon, orient: str) -> LayerPolygon:
    """Rotate one polygon about the origin by ``orient``. Vertex winding is
    preserved for R0/R180 and reversed for R90/R270 (an odd 90-degree rotation of
    a CCW ring flips its handedness), so the emitted ring stays consistently
    oriented -- matching what ``_reflect_poly_x`` does for the mirror seam."""
    if orient == "R0":
        return lp
    pts = [_rotate_point(v.x, v.y, orient) for v in lp.polygon.vertices]
    if orient in ("R90", "R270"):
        pts = list(reversed(pts))
    return LayerPolygon(
        layer=lp.layer,
        datatype=lp.datatype,
        polygon=Polygon(vertices=[Point(px, py) for (px, py) in pts]),
    )


def _device_grid_nm(tech) -> int:
    """The PDK manufacturing grid (nm) used to re-snap a rotated device. gf180 is
    5nm; other PDKs node-scale to the same 5nm floor (the rotation of an on-grid
    rectilinear device is already on-grid, so this only ever confirms it)."""
    return 5


def _m1_min_area_side_nm(area_nm2: int) -> int:
    """Area-square side surviving centered-pad division and D4 grid snapping."""
    area = max(0, int(area_nm2))
    side = math.isqrt(area)
    if side * side < area:
        side += 1
    # Both opposite pad edges share the same grid phase when its side is an
    # even grid multiple. Later orientation cannot round 448nm down to 445nm.
    quantum = 2 * _device_grid_nm(None)
    return ((side + quantum - 1) // quantum) * quantum


# --- resolved PDK facts the pure geometry core consumes ----------------------


@dataclass(frozen=True)
class _LayerKeys:
    """The GDS ``(layer, datatype)`` keys a device draws, resolved from a
    :class:`~clay.layout.tech.LayoutTech`. ``active`` is the diffusion/COMP layer
    (ndiff for an NMOS, pdiff for a PMOS on a diffusion-distinguished PDK; the one
    shared COMP on an implant-distinguished PDK). ``gate_implant`` is the implant
    grown over the gate (Nplus for NMOS, Pplus for PMOS) or ``None`` on a
    diffusion-distinguished PDK. ``tap_implant`` is the opposite-type implant a tap
    needs (Nplus for the PMOS in-well tap, Pplus for the NMOS substrate tap)."""

    active: LayerKey
    poly: LayerKey
    contact: LayerKey
    metal1: LayerKey
    metal1_label: LayerKey
    gate_implant: Optional[LayerKey]
    tap_implant: Optional[LayerKey]
    nwell: Optional[LayerKey]


@dataclass(frozen=True)
class DeviceCellRules:
    """Integer DRC dimensions (nm) the pure geometry core needs, resolved from a
    :class:`~clay.layout.tech.LayoutTech`. Every value traces to a ``tech``
    accessor (plus the documented euclidean-corner margin the gf180 foldable
    recipe used, carried as DATA so gf180 stays byte-parity)."""

    contact_size: int            # via_size("contact")  -- CO.1
    contact_space: int           # via_spacing("contact") -- CO.2 (cut-to-cut)
    contact_to_gate: int         # CO.7 (contact-to-gate spacing)
    comp_contact_enc: int        # enclosure("contact","comp")  -- CO.4
    metal1_contact_enc: int      # enclosure("contact","metal1")  -- CO.6
    metal1_line_end: int         # line_end_ext("metal1") -- M1.LE.1 (0 = none)
    poly_contact_enc: int        # enclosure("contact","poly2")  -- CO.3
    poly_endcap: int             # poly end-cap past the COMP edge
    poly_min_width: int          # min_width("poly2")
    gate_min_length: int         # min drawn gate length
    channel_min_width: int       # min_width("comp")
    metal1_min_width: int        # min_width("metal1")
    metal1_min_space: int        # min_space("metal1") -- M1.S.1 (abut clearance)
    field_poly_to_comp: int      # PL.5 (field poly to COMP space)
    implant_gate_overlap: int    # implant_well().{n,p}plus_gate_enclosure (+margin)
    implant_extension: int       # implant over a tap COMP (NP/PP.5b)
    implant_to_opp_comp: int     # NP/PP.3a (implant space to opposite COMP)
    nwell_pcomp_overlap: int     # implant_well().nwell_pcomp_enclosure (+margin)
    nwell_ncomp_overlap: int     # implant_well().nwell_ncomp_enclosure (+margin)
    nwell_min_width: int         # min_width("nwell")
    finger_space: int            # contacted inter-finger space (= csize + 2*CO.7)
    metal1_min_area: int = 0     # min_area("metal1") nm^2 -- M1.A.1 (0 = none);
    #                              the standalone substrate-tap M1 island must
    #                              satisfy it on its own (it no longer borrows
    #                              area from the source rail).


# Euclidean-corner margins the gf180 R162 foldable recipe added on top of the raw
# foundry edge rule (so the rotated/diagonal corner check passes). These are DATA
# (constants), not geometry logic: they keep the device-cell builder byte-parity
# with build_foldable_device while every BASE value still comes from tech.
_GF180_IMPLANT_GATE_MARGIN = 30     # 260 drawn vs NP.5a/PP.5a 230 edge rule
_GF180_NWELL_PCOMP_MARGIN = 70      # 500 drawn vs DF.4c 430 edge rule
_GF180_NWELL_NCOMP_MARGIN = 80      # 200 drawn vs DF.4d 120 edge rule

# OD2 (thick-gate-oxide / 3.3V selector) enclosure over the device, in nm. A THICK
# device draws OD2 over its channel; the foundry OD2 enclosure rules are OD2.E.1
# (OD2 over OD >= 0.32um) and OD2.E.2 (OD2 over the poly gate >= 0.4um). The R0
# device runs its poly fingers VERTICALLY, so the gate exits the active at the
# top/bottom COMP edges (needs the 0.4um gate enclosure) while the left/right COMP
# edges are pure OD (source/drain diffusion -> the 0.32um OD enclosure). Growing the
# COMP by these per-edge margins satisfies both rules by construction; the box
# rotates/reflects with the device (it is injected before the orient/mirror pass).
# These are DATA (foundry rule values), like the gf180 margins above -- OD2 is a
# open_pdk-only marker, so no other PDK ever grows it.
_OD2_ENCLOSE_OD_NM = 320       # OD2.E.1: OD2 over OD (source/drain diffusion) edges
_OD2_ENCLOSE_GATE_NM = 400     # OD2.E.2: OD2 over the poly gate (channel) exit edges
# CO.7 (contact-to-gate) + the poly end-cap are not same-layer width/space rules
# the rule container exposes by a single token; the contacted finger space the
# builder uses is csize + 2*CO.7. We source CO.7 + the end-cap from the gf180
# device-geom module's calibrated constants when present (gf180), else node-scale.
_CONTACT_TO_GATE_FALLBACK_FACTOR = 0.0   # see _rules_from_tech


def _dense_source_rail_gap(r: DeviceCellRules) -> int:
    """PDK-derived gap from COMP bottom to a dense-array source-rail centreline.

    Dense S/D M1 strips extend below COMP by
    ``metal1_line_end/contact_enclosure - comp_contact_enclosure``.  A shared
    source rail spans underneath foreign drain strips, so its top edge must clear
    that exact strip bottom by the M1 spacing rule.  All dense device/array
    builders consume this one translation from rules to geometry.
    """
    m1_cut_enc = max(r.metal1_contact_enc, r.metal1_line_end)
    dense_strip_bot = r.comp_contact_enc - m1_cut_enc
    return max(
        r.field_poly_to_comp + 50,
        r.metal1_min_space - dense_strip_bot,
    )


def _contact_via_name(tech) -> Optional[str]:
    """The rule-container via name for the diff/poly->metal1 contact cut (gf180
    'contact', sky130 'licon1'/'mcon'). Resolved from ``tech.via_names()`` since the
    name differs per PDK."""
    names = set(tech.via_names())
    for cand in ("contact", "licon1", "mcon", "co"):
        if cand in names:
            return cand
    return next(iter(tech.via_names()), None)


def _active_name(tech) -> str:
    """The rule-container name for the active/diffusion layer (gf180 'comp',
    external_pdk 'diff'). Names differ per PDK; resolve from the layer-name list."""
    names = set(tech.layer_names())
    for cand in ("comp", "diff", "active", "od"):
        if cand in names:
            return cand
    return "comp"


def _poly_name(tech) -> str:
    """The rule-container name for the gate-poly layer (gf180 'poly2', external_pdk
    'poly')."""
    names = set(tech.layer_names())
    for cand in ("poly2", "poly", "po", "gate"):
        if cand in names:
            return cand
    return "poly2"


def _enc_or(tech, inner: str, outer: str, fallback: int) -> int:
    try:
        v = tech.enclosure(inner, outer)
    except KeyError:
        v = None
    return int(v) if v is not None else int(fallback)


def _width_or(tech, name: str, fallback: int) -> int:
    try:
        v = tech.min_width(name)
    except KeyError:
        v = None
    return int(v) if v is not None else int(fallback)


def _space_or(tech, name: str, fallback: int) -> int:
    """``tech.min_space(name)`` (nm) or ``fallback`` when the PDK declares no
    same-layer spacing for ``name``. The fallback (the layer min width) is the
    conservative floor every target node satisfies (space >= width on all of
    gf180 / external_pdk / open_pdk metal1)."""
    try:
        v = tech.min_space(name)
    except KeyError:
        v = None
    return int(v) if v is not None else int(fallback)


def _area_or(tech, name: str, fallback: int) -> int:
    """``tech.min_area(name)`` (nm^2) or ``fallback`` when the PDK exposes no
    min-area datum for ``name``. 0 means "no rule" -- area floors stay inert."""
    try:
        v = tech.min_area(name)
    except Exception:
        v = None
    return int(v) if v else int(fallback)


def _line_end_or(tech, layer_name: str, fallback: int) -> int:
    """``tech.line_end_ext(layer_name)`` (nm) or ``fallback`` when the PDK declares
    no metal line-end rule. ``fallback=0`` means "no line-end" so the device M1
    geometry is unchanged for a PDK without the rule (gf180 -> byte-parity)."""
    try:
        v = tech.line_end_ext(layer_name)
    except Exception:
        v = None
    return int(v) if v else int(fallback)


def _rules_from_tech(tech) -> DeviceCellRules:
    """Resolve the integer DRC dimensions for ``tech``.

    Every BASE value comes from a ``tech`` accessor (the active/poly LAYER NAMES
    are resolved from ``tech.layer_names()`` because the rule container names them
    per-PDK -- gf180 'comp'/'poly2', external_pdk 'diff'/'poly'). Two gf180-recipe
    specifics (CO.7 contact-to-gate spacing and the poly end-cap) are not
    single-token rule container values; they are sourced from the gf180 device-geom
    calibrated constants for gf180 and node-scaled for other PDKs. The
    euclidean-corner margins (DATA, above) are added to the raw edge rule so gf180
    reproduces the foldable recipe byte-for-byte."""
    iw = tech.implant_well()
    contact_via = _contact_via_name(tech)
    csize = (tech.via_size(contact_via) if contact_via else None) or 0
    # CO.2 cut-to-cut spacing (for the contact_rows column pitch); falls back to
    # the cut size when the PDK exposes no spacing token.
    via_sp = (tech.via_spacing(contact_via) if contact_via else None) or csize
    active = _active_name(tech)
    poly = _poly_name(tech)

    # CO.7 contact-to-gate + poly end-cap: gf180 device-geom calibrated values when
    # this is gf180; otherwise a node-scaled fallback from the contact size.
    co7, endcap, gate_min_len, finger_space = _co7_endcap_for(tech, csize)

    comp_contact_enc = _enc_or(tech, "contact", active, csize // 4 or 50)
    metal1_contact_enc = _enc_or(tech, "contact", "metal1", csize // 4 or 50)
    poly_contact_enc = _enc_or(tech, "contact", poly, comp_contact_enc)

    # M1 end-of-line (M1.LE.1): how far an M1 wire END must overhang the cut it
    # lands on (the deck's ``metal.enclosing(cut, v, projection)`` rule). gf180 has
    # NO line-end rule -> 0 (the device M1-over-contact geometry is byte-unchanged);
    # open_pdk requires 0.06um (60nm) so the small M1-over-contact pads/connectors
    # (CO.6 = 5nm overhang) must extend past the cut to the line-end.
    metal1_line_end = _line_end_or(tech, "metal1", 0)

    # node-scaled fallbacks for widths the deck is silent about (keeps the device
    # legal-by-construction: a finger >= contact size, an nwell >= 2*contact).
    chan_min = _width_or(tech, active, csize)
    nwell_min = _width_or(tech, "nwell", 4 * csize)

    return DeviceCellRules(
        contact_size=csize,
        contact_space=via_sp,
        contact_to_gate=co7,
        comp_contact_enc=comp_contact_enc,
        metal1_contact_enc=metal1_contact_enc,
        metal1_line_end=metal1_line_end,
        poly_contact_enc=poly_contact_enc,
        poly_endcap=endcap,
        poly_min_width=_width_or(tech, poly, csize),
        gate_min_length=gate_min_len,
        channel_min_width=chan_min,
        metal1_min_width=_width_or(tech, "metal1", csize + 2 * metal1_contact_enc),
        metal1_min_space=_space_or(
            tech, "metal1", _width_or(tech, "metal1", csize + 2 * metal1_contact_enc)
        ),
        field_poly_to_comp=_field_poly_to_comp(tech),
        implant_gate_overlap=int(iw.nplus_gate_enclosure) + _implant_gate_margin(tech),
        implant_extension=_implant_extension(tech),
        implant_to_opp_comp=_implant_to_opp_comp(tech),
        nwell_pcomp_overlap=int(iw.nwell_pcomp_enclosure) + _nwell_pcomp_margin(tech),
        nwell_ncomp_overlap=int(iw.nwell_ncomp_enclosure) + _nwell_ncomp_margin(tech),
        nwell_min_width=nwell_min,
        finger_space=finger_space,
        metal1_min_area=_area_or(tech, "metal1", 0),
    )


def _is_gf180(tech) -> bool:
    return getattr(tech, "pdk_id", "").lower() == "gf180mcu"


def _od2_key(tech) -> Optional[LayerKey]:
    """The GDS key for the thick-gate-oxide (3.3V) OD2 marker, or None when the PDK
    does not declare one. Resolved through the rule-container marker token (the
    open_pdk adapter wires ``od2`` -> (4, 0) in its layermap); a PDK without a
    thick-ox domain returns None, so a thin build never touches OD2."""
    try:
        return tech.marker_key("od2")
    except Exception:
        return None


def _thick_oxide_is_default(tech) -> bool:
    """Whether this PDK's baseline MOS is already the thick-oxide class."""
    adapter = getattr(tech, "adapter", None)
    extra = getattr(adapter, "extra", {}) if adapter is not None else {}
    return bool(extra.get("thick_oxide_is_default", False))


def _od2_box_over_comp(comp_bbox: Bbox) -> Bbox:
    """The OD2 rectangle for a THICK device: the COMP grown by the per-edge OD2
    enclosure (OD2.E.1 over the OD sides, OD2.E.2 over the gate-exit sides). Built in
    the R0 frame (gate exits at the COMP top/bottom); it rotates/reflects with the
    device because it is injected before the orient/mirror pass."""
    x0, y0, x1, y1 = comp_bbox
    return (
        x0 - _OD2_ENCLOSE_OD_NM,
        y0 - _OD2_ENCLOSE_GATE_NM,
        x1 + _OD2_ENCLOSE_OD_NM,
        y1 + _OD2_ENCLOSE_GATE_NM,
    )


def _co7_endcap_for(tech, csize: int) -> Tuple[int, int, int, int]:
    """Return (contact_to_gate CO.7, poly_endcap, gate_min_length, finger_space).

    For gf180 these come from the calibrated ``gf180_device_geom`` constants (and
    ``gf180_contacted_finger_space_nm()``); for other PDKs a node-scaled fallback
    keeps the device legal-by-construction (csize-derived clearances)."""
    if _is_gf180(tech):
        from .gf180_device_geom import (
            GF180_CONTACT_TO_GATE_SPACE_NM,
            GF180_POLY_ENDCAP_NM,
            GF180_GATE_MIN_LENGTH_NM,
            gf180_contacted_finger_space_nm,
        )
        return (
            GF180_CONTACT_TO_GATE_SPACE_NM,
            GF180_POLY_ENDCAP_NM,
            GF180_GATE_MIN_LENGTH_NM,
            gf180_contacted_finger_space_nm(),
        )
    # Generic node-scaled fallback. CO.7 ~ contact spacing/2; end-cap ~ contact
    # size; min gate length from the adapter (lmin) when available; finger space =
    # contact + 2*CO.7 (a contacted inter-finger gap, the same construction gf180
    # uses).
    contact_via = _contact_via_name(tech)
    via_sp = (tech.via_spacing(contact_via) if contact_via else None) or csize
    co7 = via_sp // 2
    endcap = csize
    lmin_um = getattr(tech.adapter, "lmin_um", None)
    gate_min = int(round(float(lmin_um) * 1000.0)) if lmin_um else csize
    finger_space = csize + 2 * co7
    return (co7, endcap, gate_min, finger_space)


def _field_poly_to_comp(tech) -> int:
    """PL.5 field-poly-to-COMP space. gf180 calibrated constant, else node-scale."""
    if _is_gf180(tech):
        from .coclean_5tota import GF180_FIELD_POLY_TO_COMP_NM
        return GF180_FIELD_POLY_TO_COMP_NM
    # node-scaled: half the contact size is a safe field-poly clearance.
    cv = _contact_via_name(tech)
    csize = (tech.via_size(cv) if cv else None) or 200
    return max(1, csize // 2)


def _implant_extension(tech) -> int:
    """Implant extension over a tap COMP (NP/PP.5b)."""
    if _is_gf180(tech):
        from .coclean_5tota import GF180_IMPLANT_EXTENSION_NM
        return GF180_IMPLANT_EXTENSION_NM
    iw = tech.implant_well()
    # the smaller of the gate overlap is a safe tap extension; floor at contact enc.
    return min(int(iw.nplus_gate_enclosure), int(iw.pplus_gate_enclosure))


def _implant_to_opp_comp(tech) -> int:
    """Implant-to-opposite-COMP space (NP/PP.3a)."""
    if _is_gf180(tech):
        from .coclean_5tota import GF180_IMPLANT_TO_OPP_COMP_SPACE_NM
        return GF180_IMPLANT_TO_OPP_COMP_SPACE_NM
    return _implant_extension(tech)


def _implant_gate_margin(tech) -> int:
    return _GF180_IMPLANT_GATE_MARGIN if _is_gf180(tech) else 0


def _nwell_pcomp_margin(tech) -> int:
    return _GF180_NWELL_PCOMP_MARGIN if _is_gf180(tech) else 0


def _nwell_ncomp_margin(tech) -> int:
    return _GF180_NWELL_NCOMP_MARGIN if _is_gf180(tech) else 0


# --- DeviceGeom -- the P1 output contract ------------------------------------


@dataclass(frozen=True)
class Terminal:
    """One device terminal: a net role ('s'/'d'/'g'/'b'), its bbox, and the net
    name it binds to (so a router/extractor can attach the schematic net to the
    right physical shape). ``role`` is the device-local pin; ``net`` is the cell
    net the placer/router assigns ('' until bound)."""

    role: str
    bbox: Bbox
    net: str = ""

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)


@dataclass(frozen=True)
class DeviceGeom:
    """A PDK-parameterized m=N MOS device cell (the P1 output, consumed by P4/P5).

    ``polygons``       -- every drawn shape (LayerPolygon on tech-resolved keys).
    ``layers``         -- per-LayoutTech-layer-key polygon lists
                          (COMP/poly/contact/metal1/implant/well/tap), keyed by the
                          GDS ``(layer, datatype)`` tuple ``tech`` returns.
    ``terminals``      -- {'s','d','g','b' -> Terminal} terminal map. 's'/'d' are
                          the source/drain commoning M1 rails, 'g' the gate
                          contact, 'b' the bulk tie (PMOS nwell tap / NMOS
                          substrate tap). A MIRRORED device's 's'/'d' are reflected
                          so the physical column carrying each net is swapped.
    ``kind``           -- "nmos" or "pmos".
    ``mirror``/``axis_x`` -- whether this cell was built reflected, and about which
                          vertical axis (for router provenance).
    ``orient``         -- the device orientation it was built at ('R0'/'R90'/
                          'R180'/'R270'); R0 is the default vertical-gate cell.
    ``finger_count`` / ``gate_length_nm`` / ``channel_width_nm`` -- dimensions.
    ``comp_bbox`` / ``nwell_bbox`` -- the active COMP / PMOS nwell extents.
    """

    polygons: List[LayerPolygon]
    terminals: Dict[str, Terminal]
    kind: str
    finger_count: int
    gate_length_nm: int
    channel_width_nm: int
    comp_bbox: Bbox
    mirror: bool = False
    axis_x: Optional[int] = None
    orient: str = "R0"
    nwell_bbox: Optional[Bbox] = None
    # back-compat rail bboxes (the FoldableDevice surface the placer/router read)
    source_rail_bbox: Optional[Bbox] = None
    drain_rail_bbox: Optional[Bbox] = None
    gate_bus_bbox: Optional[Bbox] = None
    gate_contact_bbox: Optional[Bbox] = None
    # The extent the placement (MILP packing + router channel) should reserve. None
    # -> the drawn-polygon extent (``extent()``) is used (gf180 + every existing
    # caller unchanged). A SHARED-NWELL PMOS that suppressed its well tap with
    # ``stable_extent`` carries the TAP-INCLUSIVE extent here so the placement is
    # byte-identical to the tap version while the emitted artwork drops the tap.
    placement_extent: Optional[Bbox] = None
    # True when this PMOS drew a LOCAL (source != bulk) well tap on the body net
    # rather than welding the tap into the source rail. Such a tap is tied in-cell
    # through the contiguous well and exposes NO router 'b' terminal (a separate
    # body-net riser would land next to the different-net source rail and short it).
    well_tap_isolated: bool = False
    capacitor_unit_occurrences: CapacitorUnitOccurrenceGeometry | None = None
    @property
    def total_width_nm(self) -> int:
        return self.finger_count * self.channel_width_nm

    @property
    def layers(self) -> Dict[LayerKey, List[LayerPolygon]]:
        out: Dict[LayerKey, List[LayerPolygon]] = {}
        for lp in self.polygons:
            out.setdefault((lp.layer, lp.datatype), []).append(lp)
        return out

    def polys_on(self, key: LayerKey) -> List[LayerPolygon]:
        return [lp for lp in self.polygons if (lp.layer, lp.datatype) == key]

    def extent(self) -> Bbox:
        # A SHARED-NWELL PMOS that suppressed its well tap reports the reserved
        # TAP-INCLUSIVE extent so the placement is byte-identical to the tap version
        # (placement-neutral). Every other device reports its drawn-polygon extent.
        if self.placement_extent is not None:
            return self.placement_extent
        boxes = [_bbox(lp) for lp in self.polygons]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def drawn_extent(self) -> Bbox:
        """The extent of the DRAWN polygons (ignores ``placement_extent``)."""
        boxes = [_bbox(lp) for lp in self.polygons]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )


# --- the pure geometry core (integer-in / data-out; Rust-mirrorable) ----------


@dataclass(frozen=True)
class _RawGeom:
    """The pure-function geometry result before any reflection: a flat polygon
    list (each as ``(layer_key, bbox)``) plus the key terminal/feature bboxes.
    Integer coordinates only; no LayerPolygon objects, no I/O -- a Rust port
    mirrors exactly this."""

    rects: List[Tuple[LayerKey, Bbox]]
    comp_bbox: Bbox
    source_rail_bbox: Bbox
    drain_rail_bbox: Bbox
    gate_bus_bbox: Bbox
    gate_contact_bbox: Bbox
    nwell_bbox: Optional[Bbox]
    tap_bbox: Optional[Bbox]   # PMOS in-well N+ tap contact / NMOS substrate tap
    # The extent the placement (MILP packing + router channel) should reserve. When
    # a PMOS suppresses its well tap with ``stable_extent`` it is the TAP-INCLUSIVE
    # extent (so the placement is byte-identical to the tap version), else None (the
    # drawn-polygon extent is used) -> gf180 + every existing caller unchanged.
    placement_extent: Optional[Bbox] = None
    # True for a SOURCE != BULK PMOS whose well tap is a LOCAL well-tie on the body
    # net (not welded to the source rail). The tie is established IN-CELL through the
    # contiguous well; it is NOT a globally-routable terminal -- the router must NOT
    # rise a separate body-net pad to it (that pad, on the body net VDDA, would land
    # adjacent to the DIFFERENT-net source rail and short it, re-merging source onto
    # VDDA). So the device exposes NO router 'b' terminal for an isolated well tap.
    well_tap_isolated: bool = False


def _build_serpentine_geom(
    *,
    kind: str,
    l_nm: int,
    w_nm: int,
    nf: int,
    fold_rows: int,
    keys: _LayerKeys,
    r: DeviceCellRules,
    draw_substrate_tap: bool,
    substrate_implant: Optional[LayerKey],
    draw_well_tap: bool = True,
    band_tap_mode: str = "isolated",
    well_tap_mode: str = "source_tied",
) -> _RawGeom:
    """PURE geometry core for a SHARED-DIFFUSION (serpentine) multi-row MOS.

    The crux S1 capability: ONE device whose total channel width (the same
    ``nf * w_nm`` an ``nf``-finger single-row build would draw) is REDISTRIBUTED
    across ``fold_rows`` horizontal BANDS stacked vertically -- the ``nf`` fingers
    are dealt out evenly over the bands (``nf`` must be a multiple of
    ``fold_rows``; the caller guarantees this so the array is rectangular and the
    total channel width is conserved EXACTLY: ``(nf/fold_rows) fingers * w_nm *
    fold_rows == nf * w_nm``). The whole active is ONE CONTINUOUS COMP rectangle
    spanning all bands -- the bands ABUT with NO inter-row gap, so the diffusion
    is continuous top-to-bottom and a given S (or D) column is one diffusion node
    shared by every band's finger at that column. KLayout extracts each
    poly-crossing as a finger and folds the same-S/D-net fingers into ONE MOS of
    the conserved total width (NOT the ``fold_rows`` fragments that abutting
    independent ``build_device`` strips with a 1um gap produce -- the measured S1
    blocker).

    Construction (R0; integer-in, ``_RawGeom`` out -- no PDK lookups, no I/O):
      * COMP: one rectangle x = comp_x0..comp_x1, y = 0..H (H = fold_rows*w_nm),
        plus the per-x-end S/D landing margin the single-row recipe uses.
      * Poly fingers: each finger is ONE full-height vertical poly strip
        (x = finger_x[i], y = -endcap .. H + top_endcap). A full-height finger
        ties every band at column i to the SAME gate (one net), so the device has
        a single gate -- exactly the parallel-finger fold KLayout expects.
      * Gate bus + one gate contact: above the COMP top, same as single-row.
      * S/D contacts: a contact + CO.6 M1 pad at EACH band's channel centre per
        S/D column (a clean per-band lattice; pitch = the band height).
      * Source / drain commoning: one full-height M1 strip per S column (ties all
        bands' source contacts at that column) and per D column, then a source
        rail below + a drain rail above wiring the columns together -- the same
        net topology the single-row device has, scaled to R bands.

    Diffusion continuity (the LVS root): because the COMP is ONE polygon and the
    finger polys are full-height, the active is never broken between rows -- there
    is no fragment. The contact lattice respects CO.EN.3 / CO.S.1 because every
    cut is enclosed by the COMP (the band height >= one contacted finger) and the
    per-column cuts are spaced by the band pitch (>= the cut spacing).
    """
    is_pmos = kind == "pmos"
    if nf % fold_rows != 0:
        raise ValueError(
            f"serpentine fold needs nf ({nf}) divisible by fold_rows "
            f"({fold_rows}) for a rectangular conserved array"
        )
    nf_row = nf // fold_rows  # fingers per band (channel width conserved)

    finger_space = r.finger_space
    endcap = r.poly_endcap
    csize = r.contact_size
    sd_clear = r.contact_to_gate
    comp_enc = r.comp_contact_enc
    m1_enc = r.metal1_contact_enc
    m1_cut_enc = max(m1_enc, r.metal1_line_end)
    m1_pad = max(csize + 2 * m1_cut_enc, 380)
    m1_w = max(r.metal1_min_width, csize + 2 * m1_enc)
    bus_band = max(r.poly_min_width, 180)
    top_endcap = max(r.field_poly_to_comp + bus_band, endcap)

    H = fold_rows * w_nm  # total continuous active height (bands abut, no gap)

    rects: List[Tuple[LayerKey, Bbox]] = []

    # --- gate poly fingers: ONE full-height strip each (ties all bands) -------
    finger_x: List[int] = [i * (l_nm + finger_space) for i in range(nf_row)]
    last_gate_x_max = finger_x[-1] + l_nm
    for x0 in finger_x:
        rects.append((keys.poly, (x0, -endcap, x0 + l_nm, H + top_endcap)))

    # --- COMP (active): ONE continuous rectangle over all bands --------------
    comp_x0 = finger_x[0] - sd_clear - csize - comp_enc
    comp_x1 = last_gate_x_max + sd_clear + csize + comp_enc
    comp_bbox = (comp_x0, 0, comp_x1, H)
    rects.append((keys.active, comp_bbox))

    # --- gate implant: COMP bbox grown by NP.5a / PP.5a ---------------------
    implant_overlap = r.implant_gate_overlap
    if keys.gate_implant is not None:
        rects.append(
            (
                keys.gate_implant,
                (
                    comp_x0 - implant_overlap,
                    0 - implant_overlap,
                    comp_x1 + implant_overlap,
                    H + implant_overlap,
                ),
            )
        )

    # --- S/D contact columns + per-band contact lattice ---------------------
    sd_centres: List[int] = [finger_x[0] - sd_clear - csize // 2]
    for i in range(nf_row - 1):
        gap_lo = finger_x[i] + l_nm
        gap_hi = finger_x[i + 1]
        sd_centres.append((gap_lo + gap_hi) // 2)
    sd_centres.append(last_gate_x_max + sd_clear + csize // 2)
    source_cols = [i for i in range(len(sd_centres)) if i % 2 == 0]
    drain_cols = [i for i in range(len(sd_centres)) if i % 2 == 1]

    # one contact at each BAND's channel centre per column (a clean lattice).
    band_cys = [b * w_nm + w_nm // 2 for b in range(fold_rows)]

    for cx in sd_centres:
        for ccy in band_cys:
            rects.append(
                (
                    keys.contact,
                    (cx - csize // 2, ccy - csize // 2,
                     cx + csize // 2, ccy + csize // 2),
                )
            )
        # one full-height M1 strip per column commoning every band's cut (CO.6).
        strip_y0 = band_cys[0] - csize // 2 - m1_cut_enc
        strip_y1 = band_cys[-1] + csize // 2 + m1_cut_enc
        half = m1_pad // 2
        rects.append((keys.metal1, (cx - half, strip_y0, cx + half, strip_y1)))

    # --- source rail (below COMP) commoning the source columns --------------
    rail_gap = r.field_poly_to_comp + 50
    src_rail_cy = -(m1_w // 2 + rail_gap)
    src_xs = [sd_centres[c] for c in source_cols]
    src_rail_bbox = (
        min(src_xs) - m1_w // 2,
        src_rail_cy - m1_w // 2,
        max(src_xs) + m1_w // 2,
        src_rail_cy + m1_w // 2,
    )
    rects.append((keys.metal1, src_rail_bbox))
    # connect each source column strip down to the source rail.
    src_strip_bot = band_cys[0] - csize // 2 - m1_cut_enc
    for cx in src_xs:
        rects.append(
            (keys.metal1, (cx - m1_w // 2, src_rail_cy, cx + m1_w // 2, src_strip_bot))
        )

    # --- gate bus + one gate contact above the COMP top ---------------------
    bus_y0 = H + r.field_poly_to_comp
    bus_y1 = H + top_endcap
    gate_cx = finger_x[0] - finger_space // 2
    bus_x0 = min(
        finger_x[0],
        gate_cx - max(l_nm // 2, csize // 2 + r.poly_contact_enc),
    )
    gate_bus_bbox = (bus_x0, bus_y0, last_gate_x_max, bus_y1)
    rects.append((keys.poly, gate_bus_bbox))

    gate_cy = bus_y1 + r.poly_contact_enc + csize // 2
    gate_pad_top = gate_cy + csize // 2 + r.poly_contact_enc
    gpad_half = max(m1_pad // 2, csize // 2 + r.poly_contact_enc)
    rects.append(
        (keys.poly, (gate_cx - gpad_half, bus_y0, gate_cx + gpad_half, gate_pad_top))
    )
    gate_contact_bbox = (
        gate_cx - csize // 2,
        gate_cy - csize // 2,
        gate_cx + csize // 2,
        gate_cy + csize // 2,
    )
    rects.append((keys.contact, gate_contact_bbox))
    # The gate M1 pad is an ISOLATED island (it has no rail to merge with), so it
    # must satisfy the M1 min AREA (M1.A.1) on its own -- the same obligation the
    # standalone substrate tap pad meets. The cut-enclosing pad (csize + 2*CO.6)
    # alone is below the external_pdk/external_pdk min area; grow it to the area-square side. gf180
    # (min_area == 380^2, equal to the 380 pad) and any PDK with no min-area rule
    # (area_side == 0) keep the 380 pad -> byte-stable there.
    g_area_side = _m1_min_area_side_nm(r.metal1_min_area)
    gpad_m1_half = max(m1_pad, g_area_side) // 2
    rects.append((keys.metal1, (gate_cx - gpad_m1_half, gate_cy - gpad_m1_half,
                                gate_cx + gpad_m1_half, gate_cy + gpad_m1_half)))

    # --- drain rail (above the gate / gate contact) -------------------------
    drn_rail_cy = gate_pad_top + r.field_poly_to_comp + m1_w // 2 + 50
    drn_xs = [sd_centres[c] for c in drain_cols]
    drain_rail_bbox = (
        min(drn_xs) - m1_w // 2,
        drn_rail_cy - m1_w // 2,
        max(drn_xs) + m1_w // 2,
        drn_rail_cy + m1_w // 2,
    )
    rects.append((keys.metal1, drain_rail_bbox))
    drn_strip_top = band_cys[-1] + csize // 2 + m1_cut_enc
    for cx in drn_xs:
        rects.append(
            (keys.metal1, (cx - m1_w // 2, drn_strip_top, cx + m1_w // 2, drn_rail_cy))
        )

    nwell_bbox: Optional[Bbox] = None
    tap_bbox: Optional[Bbox] = None
    well_tap_isolated = False

    # --- PMOS Nwell + in-well N+ well-tap tied to the source rail (bulk) -----
    if is_pmos and keys.nwell is not None:
        pplus_y0 = 0 - implant_overlap
        tap_size = max(csize + 2 * comp_enc + 80, 600)
        tap_ext = r.implant_extension
        tap_cy = pplus_y0 - r.implant_to_opp_comp - tap_ext - tap_size // 2
        tap_cx = (comp_x0 + comp_x1) // 2
        tap_comp = (
            tap_cx - tap_size // 2, tap_cy - tap_size // 2,
            tap_cx + tap_size // 2, tap_cy + tap_size // 2,
        )
        nx0 = comp_x0 - r.nwell_pcomp_overlap
        nx1 = comp_x1 + r.nwell_pcomp_overlap
        ny1 = H + r.nwell_pcomp_overlap
        if draw_well_tap and well_tap_mode == "isolated":
            # SOURCE != BULK: a LOCAL well tap on the BODY net (its OWN island,
            # min-spaced from the source rail, never tied to source) -- the same
            # contract the single-row builder + the NMOS substrate tap honour.
            iso_area_side = _m1_min_area_side_nm(r.metal1_min_area)
            iso_pad = max(csize + 2 * m1_cut_enc, r.metal1_min_width, iso_area_side)
            iso_half = (iso_pad + 1) // 2
            rail_bot = src_rail_cy - m1_w // 2
            iso_cy = min(tap_cy, rail_bot - r.metal1_min_space - iso_half)
            tap_comp = (
                tap_cx - tap_size // 2, iso_cy - tap_size // 2,
                tap_cx + tap_size // 2, iso_cy + tap_size // 2,
            )
            tap_implant_box = (
                tap_comp[0] - tap_ext, tap_comp[1] - tap_ext,
                tap_comp[2] + tap_ext, tap_comp[3] + tap_ext,
            )
            tap_co = (
                tap_cx - csize // 2, iso_cy - csize // 2,
                tap_cx + csize // 2, iso_cy + csize // 2,
            )
            tap_m1 = (
                tap_cx - iso_half, iso_cy - iso_half,
                tap_cx + iso_half, iso_cy + iso_half,
            )
            rects.append((keys.active, tap_comp))
            if keys.tap_implant is not None:
                rects.append((keys.tap_implant, tap_implant_box))
            rects.append((keys.contact, tap_co))
            rects.append((keys.metal1, tap_m1))
            nx0 = min(nx0, tap_comp[0] - r.nwell_ncomp_overlap)
            nx1 = max(nx1, tap_comp[2] + r.nwell_ncomp_overlap)
            ny0 = min(tap_comp[1] - r.nwell_ncomp_overlap, 0 - r.nwell_pcomp_overlap)
            tap_bbox = tap_co
            well_tap_isolated = True
        elif draw_well_tap:
            tap_implant_box = (
                tap_comp[0] - tap_ext, tap_comp[1] - tap_ext,
                tap_comp[2] + tap_ext, tap_comp[3] + tap_ext,
            )
            tap_co = (
                tap_cx - csize // 2, tap_cy - csize // 2,
                tap_cx + csize // 2, tap_cy + csize // 2,
            )
            tap_m1_half = max(csize // 2 + m1_cut_enc, r.metal1_min_width // 2)
            # tap M1 rises to the source rail (source==bulk==vdd for a self-bulk PMOS).
            tap_m1 = (
                tap_cx - tap_m1_half, tap_cy - tap_m1_half,
                tap_cx + tap_m1_half, src_rail_cy + m1_w // 2,
            )
            rects.append((keys.active, tap_comp))
            if keys.tap_implant is not None:
                rects.append((keys.tap_implant, tap_implant_box))
            rects.append((keys.contact, tap_co))
            rects.append((keys.metal1, tap_m1))
            # same-net bridge into the source rail when it does not already cover the tap.
            if src_xs:
                rail_lo = min(src_xs) - m1_w // 2
                rail_hi = max(src_xs) + m1_w // 2
                if tap_cx - tap_m1_half > rail_hi or tap_cx + tap_m1_half < rail_lo:
                    nearest_src = min(src_xs, key=lambda sx: abs(sx - tap_cx))
                    bx0 = min(tap_cx, nearest_src) - tap_m1_half
                    bx1 = max(tap_cx, nearest_src) + tap_m1_half
                    rects.append(
                        (keys.metal1, (bx0, src_rail_cy - m1_w // 2,
                                       bx1, src_rail_cy + m1_w // 2))
                    )
            nx0 = min(nx0, tap_comp[0] - r.nwell_ncomp_overlap)
            nx1 = max(nx1, tap_comp[2] + r.nwell_ncomp_overlap)
            ny0 = min(tap_comp[1] - r.nwell_ncomp_overlap, 0 - r.nwell_pcomp_overlap)
            tap_bbox = tap_co
        else:
            ny0 = 0 - r.nwell_pcomp_overlap
        if nx1 - nx0 < r.nwell_min_width:
            mid = (nx0 + nx1) // 2
            nx0 = mid - r.nwell_min_width // 2
            nx1 = mid + r.nwell_min_width // 2
        if ny1 - ny0 < r.nwell_min_width:
            ny0 = ny1 - r.nwell_min_width
        nwell_bbox = (nx0, ny0, nx1, ny1)
        rects.append((keys.nwell, nwell_bbox))

    # --- NMOS substrate tap (P-C1 band-level shared / none / isolated) --------
    # band_tap_mode (resolved by build_device from the DESIGN nets):
    #   * "shared"   -- a SELF-BULK NMOS (source net == body net, e.g. source==
    #                   GND==body): draw a BAND-LEVEL P+ tap STRIP spanning the
    #                   COMP width, tied BY CONSTRUCTION into the source rail M1
    #                   (the source carries GND). VSUB->GND with NO competing pin
    #                   (the historical net08<->GND short was an isolated square
    #                   tap M1 overlapping the source corridor -- removed here).
    #   * "none"     -- a non-self-bulk NMOS (the MM15 family, source != body):
    #                   draw NO tap; the bulk rides the GLOBAL substrate net, tied
    #                   to GND through the self-bulk NMOS taps elsewhere (KLayout
    #                   connect_global(ptap, VSUB) + connect_global(sub, VSUB) --
    #                   one self-bulk tap anywhere ties the whole substrate, so
    #                   dropping MM15's tap never floats the bulk).
    #   * "isolated" -- standalone build (no design nets): the historical
    #                   per-device isolated P+ tap island with a distinct 'b' pin.
    if (not is_pmos) and draw_substrate_tap and substrate_implant is not None \
            and band_tap_mode != "none":
        nplus_y0 = 0 - implant_overlap
        tap_ext = r.implant_extension
        tap_size = max(csize + 2 * comp_enc + 80, 600)
        if band_tap_mode == "shared" and src_xs:
            # BAND-LEVEL strip tied into the source rail (self-bulk).
            strip_x0 = comp_x0
            strip_x1 = comp_x1
            tap_m1_h = max(csize + 2 * m1_cut_enc, r.metal1_min_width)
            tap_m1_half = (tap_m1_h + 1) // 2
            tap_cy = nplus_y0 - r.implant_to_opp_comp - tap_ext - tap_size // 2
            rail_bot = src_rail_cy - m1_w // 2
            tap_cy = min(tap_cy, rail_bot - r.metal1_min_space - tap_size // 2)
            tap_comp = (
                strip_x0, tap_cy - tap_size // 2,
                strip_x1, tap_cy + tap_size // 2,
            )
            tap_implant_box = (
                tap_comp[0] - tap_ext, tap_comp[1] - tap_ext,
                tap_comp[2] + tap_ext, tap_comp[3] + tap_ext,
            )
            rects.append((keys.active, tap_comp))
            rects.append((substrate_implant, tap_implant_box))
            # a contact lattice across the strip at each source column + an M1
            # that rises into the source rail (source==body==GND, by construction).
            for cx in src_xs:
                tap_co = (
                    cx - csize // 2, tap_cy - csize // 2,
                    cx + csize // 2, tap_cy + csize // 2,
                )
                rects.append((keys.contact, tap_co))
                rects.append((keys.metal1, (
                    cx - tap_m1_half, tap_cy - tap_m1_half,
                    cx + tap_m1_half, src_rail_cy + m1_w // 2,
                )))
            tap_bbox = (
                src_xs[0] - csize // 2, tap_cy - csize // 2,
                src_xs[-1] + csize // 2, tap_cy + csize // 2,
            )
        else:
            # ISOLATED per-device tap island (standalone / no design nets): a
            # distinct 'b' pin (historical behaviour, backward-compatible).
            tap_cx = (comp_x0 + comp_x1) // 2
            area_side = _m1_min_area_side_nm(r.metal1_min_area)
            tap_pad = max(csize + 2 * m1_cut_enc, r.metal1_min_width, area_side)
            tap_m1_half = (tap_pad + 1) // 2
            tap_cy = nplus_y0 - r.implant_to_opp_comp - tap_ext - tap_size // 2
            rail_bot = src_rail_cy - m1_w // 2
            tap_cy = min(tap_cy, rail_bot - r.metal1_min_space - tap_m1_half)
            tap_comp = (
                tap_cx - tap_size // 2, tap_cy - tap_size // 2,
                tap_cx + tap_size // 2, tap_cy + tap_size // 2,
            )
            tap_implant_box = (
                tap_comp[0] - tap_ext, tap_comp[1] - tap_ext,
                tap_comp[2] + tap_ext, tap_comp[3] + tap_ext,
            )
            tap_co = (
                tap_cx - csize // 2, tap_cy - csize // 2,
                tap_cx + csize // 2, tap_cy + csize // 2,
            )
            tap_m1 = (
                tap_cx - tap_m1_half, tap_cy - tap_m1_half,
                tap_cx + tap_m1_half, tap_cy + tap_m1_half,
            )
            rects.append((keys.active, tap_comp))
            rects.append((substrate_implant, tap_implant_box))
            rects.append((keys.contact, tap_co))
            rects.append((keys.metal1, tap_m1))
            tap_bbox = tap_co

    return _RawGeom(
        rects=rects,
        comp_bbox=comp_bbox,
        source_rail_bbox=src_rail_bbox,
        drain_rail_bbox=drain_rail_bbox,
        gate_bus_bbox=gate_bus_bbox,
        gate_contact_bbox=gate_contact_bbox,
        nwell_bbox=nwell_bbox,
        tap_bbox=tap_bbox,
        placement_extent=None,
        well_tap_isolated=well_tap_isolated,
    )


def _build_device_geom(
    *,
    kind: str,
    l_nm: int,
    w_nm: int,
    nf: int,
    keys: _LayerKeys,
    r: DeviceCellRules,
    draw_substrate_tap: bool,
    substrate_implant: Optional[LayerKey],
    draw_well_tap: bool = True,
    stable_extent: bool = False,
    contact_rows: bool = False,
    well_tap_mode: str = "source_tied",
) -> _RawGeom:
    """PURE geometry core: integer coordinates in, ``_RawGeom`` out.

    No PDK lookups, no I/O -- every layer key + DRC dimension arrives pre-resolved
    in ``keys`` + ``r``. This is the byte-for-byte mirror target for a future Rust
    kernel. The recipe is the R162 foldable MOS (identical to
    ``build_foldable_device``): folded poly fingers, every S/D contact on the
    COMP, commoned alternating S/D M1 rails, a poly gate bus + one gate contact,
    the gate implant grown over the gate, and for a PMOS an enclosing Nwell + an
    in-well N+ well-tap. ``substrate_implant`` is the implant for the optional NMOS
    P+ substrate tap (when ``draw_substrate_tap``).

    ``draw_well_tap`` (default True -> byte-parity for gf180 + every existing
    caller) controls the PMOS in-well N+ well-tap. The SHARED-NWELL analog layout
    suppresses the per-device tap on a PMOS whose SOURCE != BULK (the comparator
    input / load pairs, source=tail, bulk=vdd): drawing a per-device tap tied to
    that device's source rail would tie the well to ``tail`` (a same-net LVS short
    + a mid-channel bulk pad that fires the M1.S.1 notch). With ``draw_well_tap``
    False the device draws ONLY its nwell (no tap COMP / implant / contact / strap),
    and the bulk is supplied by the contiguous shared well + the well tap on the
    self-bulk PMOS (source=bulk=vdd: tail / output PMOS), which keep their tap. The
    nwell is still emitted so the shared well stays contiguous over this device."""
    is_pmos = kind == "pmos"

    finger_space = r.finger_space
    endcap = r.poly_endcap
    csize = r.contact_size
    sd_clear = r.contact_to_gate
    comp_enc = r.comp_contact_enc
    m1_enc = r.metal1_contact_enc
    # The M1-over-contact overhang past the cut: the CO.6 enclosure AND the M1
    # end-of-line (M1.LE.1) when the PDK declares one. A wire/pad ENDING over a cut
    # must overhang by the line-end (external_pdk 60nm); the CO.6 enclosure (external_pdk 5nm) alone
    # leaves the foundry line-end check failing. gf180 has no line-end rule
    # (``metal1_line_end`` == 0) so ``m1_cut_enc`` == ``m1_enc`` -> byte-unchanged.
    m1_cut_enc = max(m1_enc, r.metal1_line_end)
    # metal1 pad over a contact: M1.LE/CO.6-enclosing AND >= the M1 min-area square.
    m1_pad = max(csize + 2 * m1_cut_enc, 380)
    # The S/D rail + its rail-to-channel connector keep the CO.6-based width so the
    # rail CENTRE (``src_rail_cy`` depends on ``m1_w // 2``) is UNMOVED -- widening it
    # to the line-end would shift every terminal y and perturb the placement/route.
    # The line-end overhang is realised on the contact-landing PADS (``m1_pad`` above,
    # the via12/terminal landings) instead, where the wire actually ENDS over a cut.
    m1_w = max(r.metal1_min_width, csize + 2 * m1_enc)
    bus_band = max(r.poly_min_width, 180)
    top_endcap = max(r.field_poly_to_comp + bus_band, endcap)

    rects: List[Tuple[LayerKey, Bbox]] = []

    # --- gate poly fingers --------------------------------------------------
    finger_x: List[int] = [i * (l_nm + finger_space) for i in range(nf)]
    last_gate_x_max = finger_x[-1] + l_nm
    for x0 in finger_x:
        rects.append((keys.poly, (x0, -endcap, x0 + l_nm, w_nm + top_endcap)))

    # --- COMP (active) ------------------------------------------------------
    comp_x0 = finger_x[0] - sd_clear - csize - comp_enc
    comp_x1 = last_gate_x_max + sd_clear + csize + comp_enc
    if contact_rows:
        # ABUT CONTRACT (contact_rows only; default path byte-unchanged).
        # Obligation: two cells abutted COMP-face-to-COMP-face must keep >= the
        # M1 min spacing (M1.S.1) between their M1, else KLayout LVS folds the
        # two MOS into one parallel device (the observed gf180 10 -> 9 merge).
        # Scheme chosen: NARROWED boundary strip + boundary COMP extension.
        # Narrowing alone (either proposed variant: width = max(min_width,
        # m1_pad - 2*min_space) biased inward, or fall back to the single pad)
        # is GEOMETRICALLY IMPOSSIBLE here: the boundary cut centre sits only
        # csize/2 + comp_enc inside the COMP edge (gf180 180nm, external_pdk/external_pdk
        # 210nm), while an M1 strip that still encloses its cuts (CO.6) needs
        # cut-half + side-enclosure + min_space/2 of COMP overhang (gf180
        # 110+60+115 = 285nm) -- a biased-narrow strip that clears the seam
        # would leave the boundary cuts uncovered (hard CO.6 violation).
        # So the boundary strip narrows to the minimum cut-enclosing width
        # (max(M1 min width, csize + 2*CO.6); the min-area pad floor is not
        # needed -- the strip merges with the rail connector at the same
        # column, so the merged region carries the area), AND the COMP is
        # extended at both x ends so a comp-flush abut leaves exactly >= the
        # M1 min space between the two cells' M1. Every datum is a tech rule
        # (metal1_min_width / metal1_min_space / CO.6); no hardcoded nm.
        bstrip_half = max(r.metal1_min_width, csize + 2 * m1_enc) // 2
        abut_margin = bstrip_half + (r.metal1_min_space + 1) // 2
        # Keep the COMP/implant edge on an even 5nm-grid multiple (10nm) when
        # bstrip_half + ceil(m1_space/2) lands on an odd 5nm step (gf180: 170+115=285).
        # Otherwise single-finger devices whose left poly gate pad lies on a 10nm
        # edge get a half-grid (x.5 nm) centroid, causing +/-2.5nm vertex rounding
        # distortion (CO.7 145nm<150nm and NP.5di 155nm<160nm). external_pdk/external_pdk are 230nm
        # (already mod 10 == 0 -> byte-identical).
        if abut_margin % 10 == 5:
            abut_margin += 5
        first_c = finger_x[0] - sd_clear - csize // 2
        last_c = last_gate_x_max + sd_clear + csize // 2
        comp_x0 = min(comp_x0, first_c - abut_margin)
        comp_x1 = max(comp_x1, last_c + abut_margin)
    comp_bbox = (comp_x0, 0, comp_x1, w_nm)
    rects.append((keys.active, comp_bbox))

    # --- gate implant: COMP bbox grown by NP.5a / PP.5a ---------------------
    implant_overlap = r.implant_gate_overlap
    if keys.gate_implant is not None:
        rects.append(
            (
                keys.gate_implant,
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
    for i in range(nf - 1):
        gap_lo = finger_x[i] + l_nm
        gap_hi = finger_x[i + 1]
        sd_centres.append((gap_lo + gap_hi) // 2)
    sd_centres.append(last_gate_x_max + sd_clear + csize // 2)
    source_cols = [i for i in range(len(sd_centres)) if i % 2 == 0]
    drain_cols = [i for i in range(len(sd_centres)) if i % 2 == 1]

    def _contact(cx: int, ccy: int) -> Tuple[LayerKey, Bbox]:
        return (
            keys.contact,
            (cx - csize // 2, ccy - csize // 2, cx + csize // 2, ccy + csize // 2),
        )

    def _m1_pad(cx: int, ccy: int) -> Tuple[LayerKey, Bbox]:
        half = m1_pad // 2
        return (keys.metal1, (cx - half, ccy - half, cx + half, ccy + half))

    if not contact_rows:
        for cx in sd_centres:
            rects.append(_contact(cx, cy))   # on-COMP contact
            rects.append(_m1_pad(cx, cy))    # CO.6-enclosing metal1 over it
    else:
        # CONTACT-ROW DENSIFICATION (vendor-density artwork): a full column of
        # cuts along the finger height per S/D column -- pitch = cut size + the
        # PDK cut spacing, first/last cut enclosed in the COMP by the comp
        # enclosure -- plus ONE full-height M1 strip enclosing every cut (CO.6).
        # The reference layout draws ~45x more cuts than the single mid-channel
        # cut; a tall finger carried its whole current through one cut.
        pitch = csize + r.contact_space
        y_lo = r.comp_contact_enc + csize // 2
        y_hi = w_nm - r.comp_contact_enc - csize // 2
        ys = [cy] if y_hi < y_lo else list(range(y_lo, y_hi + 1, pitch))
        m1h = m1_pad // 2
        # The strip top/bottom are wire ENDS over the outermost cuts: they must
        # overhang by the M1 line-end (external_pdk M1.LE.1 / external_pdk 60nm), not just the
        # CO.6 enclosure -- ``m1_cut_enc`` = max(CO.6, line_end). gf180 has no
        # line-end rule, so m1_cut_enc == m1_enc there (geometry unchanged).
        # Boundary (first/last) columns take the narrowed abut-clear strip
        # (``bstrip_half``, see the COMP abut-contract block above); the strip
        # LONG SIDES are not line-ends, so the CO.6 side enclosure suffices.
        last_i = len(sd_centres) - 1
        for i, cx in enumerate(sd_centres):
            for ccy in ys:
                rects.append(_contact(cx, ccy))
            strip_y0 = min(ys) - csize // 2 - m1_cut_enc
            strip_y1 = max(ys) + csize // 2 + m1_cut_enc
            half = bstrip_half if i in (0, last_i) else m1h
            rects.append((keys.metal1,
                          (cx - half, strip_y0, cx + half, strip_y1)))

    # --- source rail (below COMP) + drain rail (above the gate), metal1 only -
    rail_gap = r.field_poly_to_comp + 50
    if contact_rows:
        # DENSE-ARTWORK RAIL CLEARANCE (contact_rows only; sparse byte-unchanged).
        # Obligation: the full-height S/D contact strips reach the COMP edge (each
        # column's strip bottom = ``comp_enc - m1_cut_enc`` above the COMP bottom),
        # and the OPPOSITE-net SOURCE RAIL runs the full device width UNDERNEATH
        # them. The sparse build put a single mid-channel pad there (far from the
        # rail); the dense DRAIN strips do not, so the rail must hold the M1 min
        # space (M1.S.1) below the drain strips' bottom or the extractor sees a
        # sub-spacing drain-strip <-> source-rail notch (measured: 200nm < 230nm
        # on external_pdk/external_pdk, 160 < 230 on gf180). Push the rail out to that rule-derived
        # clearance. Every datum is a tech rule (metal1_min_space / comp_contact_enc
        # / m1_cut_enc); no hardcoded nm, no per-circuit constant. The device grows
        # taller by exactly this delta -- the placement reserves it through the same
        # stamp-extent sizes pipeline every footprint already flows through.
        rail_gap = _dense_source_rail_gap(r)
    src_rail_cy = -(m1_w // 2 + rail_gap)
    src_xs = [sd_centres[c] for c in source_cols]
    src_rail_bbox = (
        min(src_xs) - m1_w // 2,
        src_rail_cy - m1_w // 2,
        max(src_xs) + m1_w // 2,
        src_rail_cy + m1_w // 2,
    )
    rects.append((keys.metal1, src_rail_bbox))
    for cx in src_xs:
        rects.append((keys.metal1, (cx - m1_w // 2, src_rail_cy, cx + m1_w // 2, cy)))

    # --- gate bus: poly strap >= PL.5 above the COMP top --------------------
    bus_y0 = w_nm + r.field_poly_to_comp
    bus_y1 = w_nm + top_endcap
    gate_cx = finger_x[0] - finger_space // 2
    bus_x0 = min(
        finger_x[0],
        gate_cx - max(l_nm // 2, csize // 2 + r.poly_contact_enc),
    )
    gate_bus_bbox = (bus_x0, bus_y0, last_gate_x_max, bus_y1)
    rects.append((keys.poly, gate_bus_bbox))

    # Gate contact: a poly pad raised above the bus over the LEFT lane.
    gate_cy = bus_y1 + r.poly_contact_enc + csize // 2
    gate_pad_top = gate_cy + csize // 2 + r.poly_contact_enc
    gpad_half = max(m1_pad // 2, csize // 2 + r.poly_contact_enc)
    rects.append(
        (keys.poly, (gate_cx - gpad_half, bus_y0, gate_cx + gpad_half, gate_pad_top))
    )
    gate_contact_bbox = (
        gate_cx - csize // 2,
        gate_cy - csize // 2,
        gate_cx + csize // 2,
        gate_cy + csize // 2,
    )
    rects.append((keys.contact, gate_contact_bbox))
    # Unlike the S/D pads, this gate landing is an isolated M1 component and
    # cannot borrow area from a rail.  Size only this pad to the PDK-declared
    # M1 minimum area; keep its centre and every S/D/rail dimension unchanged.
    gate_area_side = _m1_min_area_side_nm(r.metal1_min_area)
    gate_m1_half = (max(m1_pad, gate_area_side) + 1) // 2
    rects.append(
        (
            keys.metal1,
            (
                gate_cx - gate_m1_half,
                gate_cy - gate_m1_half,
                gate_cx + gate_m1_half,
                gate_cy + gate_m1_half,
            ),
        )
    )

    # --- drain rail (above the gate / gate contact), metal1 only ------------
    drn_rail_cy = gate_pad_top + r.field_poly_to_comp + m1_w // 2 + 50
    drn_xs = [sd_centres[c] for c in drain_cols]
    drain_rail_bbox = (
        min(drn_xs) - m1_w // 2,
        drn_rail_cy - m1_w // 2,
        max(drn_xs) + m1_w // 2,
        drn_rail_cy + m1_w // 2,
    )
    rects.append((keys.metal1, drain_rail_bbox))
    for cx in drn_xs:
        rects.append((keys.metal1, (cx - m1_w // 2, cy, cx + m1_w // 2, drn_rail_cy)))

    nwell_bbox: Optional[Bbox] = None
    tap_bbox: Optional[Bbox] = None
    placement_extent: Optional[Bbox] = None
    well_tap_isolated = False

    # --- PMOS Nwell + in-well N+ well-tap tied to the source rail (bulk) -----
    if is_pmos and keys.nwell is not None:
        pplus_y0 = 0 - implant_overlap
        tap_size = max(csize + 2 * comp_enc + 80, 600)
        tap_ext = r.implant_extension
        tap_cy = pplus_y0 - r.implant_to_opp_comp - tap_ext - tap_size // 2
        # Tap x-position. DEFAULT (gf180, no line-end rule): the COMP centre -- the
        # legacy foldable recipe, byte-identical. On a line-end PDK (external_pdk + the
        # commercial nodes) the M1 spacing is TIGHT, so a tap M1 at the COMP centre
        # leaves a sub-rule M1.S.1 gap to BOTH adjacent S/D columns (one is the
        # DIFFERENT-net drain). The tap of a self-bulk PMOS is the SAME net as its
        # source (source=bulk=vdd), so it is aligned under the leftmost SOURCE column
        # and its M1 MERGES with that source connector (one contiguous same-net M1) --
        # clearing the different-net drain column entirely (the source column screens
        # the tap from the drain). gf180 (m1_cut_enc == m1_enc) keeps the COMP centre.
        tap_aligned_to_source = (
            draw_well_tap and well_tap_mode == "source_tied"
            and r.metal1_line_end > 0 and m1_cut_enc != m1_enc
            and bool(source_cols)
        )
        if tap_aligned_to_source:
            tap_cx = sd_centres[source_cols[0]]
        else:
            tap_cx = (comp_x0 + comp_x1) // 2
        tap_comp = (
            tap_cx - tap_size // 2,
            tap_cy - tap_size // 2,
            tap_cx + tap_size // 2,
            tap_cy + tap_size // 2,
        )
        if draw_well_tap and well_tap_mode == "isolated":
            # SOURCE != BULK: a LOCAL well tap on the BODY net -- its OWN M1 island,
            # >= the M1 min space from every source-net M1, NEVER tied to the source
            # rail. The source-tied weld (the else branch) was correct only for a
            # self-bulk PMOS (source==bulk); for the LDO cascode PMOS (source=net073,
            # body=VDDA) it physically merged the source net into the well-tap node
            # (the production extraction dragged net073 onto VDDA). Net-derived from
            # the device's OWN declared s/b nets (no per-circuit branch); MIRRORS the
            # NMOS substrate-tap island contract (test_tap_net_contract). The pad
            # encloses its cut on every side (CO.6 / M1.LE.1), meets the M1 min AREA
            # on its own (no rail to borrow from), and is clamped DOWN only if needed
            # to keep the M1 min space below the source rail -- in the common case the
            # natural tap_cy already clears it, so the device extent is UNCHANGED.
            iso_area_side = _m1_min_area_side_nm(r.metal1_min_area)
            iso_pad = max(csize + 2 * m1_cut_enc, r.metal1_min_width, iso_area_side)
            iso_half = (iso_pad + 1) // 2
            rail_bot = src_rail_cy - m1_w // 2
            iso_cy = min(tap_cy, rail_bot - r.metal1_min_space - iso_half)
            tap_comp = (
                tap_cx - tap_size // 2, iso_cy - tap_size // 2,
                tap_cx + tap_size // 2, iso_cy + tap_size // 2,
            )
            tap_implant_box = (
                tap_comp[0] - tap_ext, tap_comp[1] - tap_ext,
                tap_comp[2] + tap_ext, tap_comp[3] + tap_ext,
            )
            tap_co = (
                tap_cx - csize // 2, iso_cy - csize // 2,
                tap_cx + csize // 2, iso_cy + csize // 2,
            )
            tap_m1 = (
                tap_cx - iso_half, iso_cy - iso_half,
                tap_cx + iso_half, iso_cy + iso_half,
            )
            rects.append((keys.active, tap_comp))
            if keys.tap_implant is not None:
                rects.append((keys.tap_implant, tap_implant_box))
            rects.append((keys.contact, tap_co))
            rects.append((keys.metal1, tap_m1))
            tap_bbox = tap_co
            well_tap_isolated = True
            # PLACEMENT-NEUTRAL: reserve the SAME extent the source-tied tap version
            # produced so the MILP packing + router channel see the byte-identical
            # device footprint and the frozen comparator/OTA placement is preserved --
            # only the drawn artwork differs. The body-net tap M1 is a DIFFERENT net
            # from the source rail (unlike the same-net source-tied riser), so it must
            # keep the M1 min space and is clamped a few nm lower; that pushes the drawn
            # nwell/implant/M1 down into the reserved inter-device gap. The source-tied
            # device's LOWEST feature is its Nwell (its tap-enclosing ny0 at the natural
            # tap_cy), so reserve exactly that bottom edge.
            _iso_src_tied_min_y = min(
                (tap_cy - tap_size // 2) - r.nwell_ncomp_overlap,
                0 - r.nwell_pcomp_overlap,
            )
        elif draw_well_tap:
            tap_implant_box = (
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
            # The PMOS in-well tap M1 lands on the tap cut and rises to the SOURCE
            # rail. The strap sides take the M1 line-end overhang (M1.LE.1) so the
            # projection check on the narrow strap passes; gf180 (no line-end,
            # m1_cut_enc == m1_enc) is byte-unchanged. Tying the well tap to the
            # SOURCE rail is the schematic-correct bulk tie when source == bulk (the
            # latch / tail / output PMOS, source=bulk=vdd) -- the SHARED-NWELL layout
            # keeps the tap ONLY on those self-bulk PMOS, so the well body is tapped
            # to vdd through them. A source != bulk PMOS (the comparator input / load
            # pair, source=tail, bulk=vdd) is built with ``draw_well_tap=False`` so it
            # has NO source-tied tap; its bulk is the shared well tapped at vdd.
            # The tap pad over the cut takes the M1 line-end overhang (M1.LE.1) on all
            # sides (the contact-enclosing pad width). For the source-aligned tap the
            # strap rises at the SOURCE column x straight up THROUGH the source
            # connector to the source rail; both are the SAME net (source=bulk=vdd), so
            # they form one contiguous column. To avoid a concave width-step notch
            # where the wider tap pad (csize + 2*m1_cut_enc) meets the narrower source
            # connector (m1_w), the source rail is locally widened to the tap-pad width
            # at this column (below), so the column is one clean rectangle clear of the
            # different-net drain (screened by the source column).
            tap_m1_half = max(csize // 2 + m1_cut_enc, r.metal1_min_width // 2)
            tap_m1 = (
                tap_cx - tap_m1_half,
                tap_cy - tap_m1_half,
                tap_cx + tap_m1_half,
                src_rail_cy + m1_w // 2,
            )
            rects.append((keys.active, tap_comp))
            if keys.tap_implant is not None:
                rects.append((keys.tap_implant, tap_implant_box))
            rects.append((keys.contact, tap_co))
            rects.append((keys.metal1, tap_m1))
            if tap_aligned_to_source:
                # Local same-net filler: widen the source connector + rail patch at the
                # tap column to the tap-pad width so the tap pad -> source connector ->
                # rail is ONE rectangle (no concave notch). Same net (vdd), clear of the
                # drain. Spans from the tap pad top down-edge up through the channel
                # connector region so the merged column has no sub-rule inner corner.
                rects.append(
                    (
                        keys.metal1,
                        (
                            tap_cx - tap_m1_half,
                            src_rail_cy - m1_w // 2,
                            tap_cx + tap_m1_half,
                            cy,
                        ),
                    )
                )
            else:
                # COMP-CENTRE tap (gf180 / no-line-end PDK): the tap rises to
                # ``src_rail_cy`` at ``tap_cx`` = the COMP centre, which sits BETWEEN the
                # source and drain columns. The source rail M1 only spans the source
                # columns (``[min(src_xs), max(src_xs)]``); for a FEW-FINGER device (e.g.
                # nf=1) that span does NOT reach the COMP-centre tap, so the tap M1 top
                # is left electrically FLOATING -- the Nwell extracts as its own isolated
                # net (the nf=1 self-bulk PMOS bulk-open). Bridge the tap into the source
                # rail with a same-net M1 patch at ``src_rail_cy`` spanning from the tap
                # column to the nearest source column. The tap is the SAME net as the
                # source (this branch only runs with ``draw_well_tap`` -> a self-bulk
                # PMOS, source == bulk), so the bridge ties no foreign net; it runs along
                # the source-rail band (below the COMP), clear of the drain. For a device
                # whose source rail already covers ``tap_cx`` (many fingers) the bridge is
                # a same-net no-op overlap. Generic: pure geometry off the source columns,
                # no device-kind / cell-name branch.
                # Only bridge when the source rail does NOT already cover the tap
                # column: a multi-finger device whose source-rail span already reaches
                # ``tap_cx`` ties the tap by abutment and stays BYTE-IDENTICAL (no extra
                # polygon). The bridge is emitted only for the genuinely-open case (the
                # tap sits outside the source-rail x-span), so existing wide-device gf180
                # geometry is unchanged and only the previously-floating tap is closed.
                if src_xs:
                    rail_lo = min(src_xs) - m1_w // 2
                    rail_hi = max(src_xs) + m1_w // 2
                    tap_lo = tap_cx - tap_m1_half
                    tap_hi = tap_cx + tap_m1_half
                    if tap_lo > rail_hi or tap_hi < rail_lo:
                        nearest_src = min(src_xs, key=lambda sx: abs(sx - tap_cx))
                        bx0 = min(tap_cx, nearest_src) - tap_m1_half
                        bx1 = max(tap_cx, nearest_src) + tap_m1_half
                        rects.append(
                            (
                                keys.metal1,
                                (
                                    bx0,
                                    src_rail_cy - m1_w // 2,
                                    bx1,
                                    src_rail_cy + m1_w // 2,
                                ),
                            )
                        )
            tap_bbox = tap_co

        # The Nwell encloses the PCOMP (always) and, when a tap is drawn, the tap
        # NCOMP too. When ``draw_well_tap`` is False the well is sized to the device
        # PCOMP alone (no tap NCOMP to enclose) -- it stays contiguous with the
        # neighbouring shared-well PMOS once the placement bridges the well boxes.
        nx0 = comp_x0 - r.nwell_pcomp_overlap
        nx1 = comp_x1 + r.nwell_pcomp_overlap
        ny1 = w_nm + r.nwell_pcomp_overlap
        if draw_well_tap:
            nx0 = min(nx0, tap_comp[0] - r.nwell_ncomp_overlap)
            nx1 = max(nx1, tap_comp[2] + r.nwell_ncomp_overlap)
            ny0 = min(tap_comp[1] - r.nwell_ncomp_overlap, 0 - r.nwell_pcomp_overlap)
        else:
            ny0 = 0 - r.nwell_pcomp_overlap
        if nx1 - nx0 < r.nwell_min_width:
            mid = (nx0 + nx1) // 2
            nx0 = mid - r.nwell_min_width // 2
            nx1 = mid + r.nwell_min_width // 2
        if ny1 - ny0 < r.nwell_min_width:
            ny0 = ny1 - r.nwell_min_width
        nwell_bbox = (nx0, ny0, nx1, ny1)
        rects.append((keys.nwell, nwell_bbox))

        # When the well tap is SUPPRESSED but extent stability is requested, the
        # device reserves the TAP-INCLUSIVE extent for the placement (MILP packing +
        # router channel) even though the tap polygons are not drawn -- so the
        # placement is byte-identical to the tap version (the frozen net-complete
        # placement / routing is preserved) while the emitted artwork drops the tap.
        # The reserved extent is the UNION of every drawn rect (the device proper:
        # poly bus / gate pad / rails / contacts) and the box the suppressed tap
        # (NCOMP + its implant) would have occupied -- i.e. the same bounding box the
        # ``draw_well_tap=True`` device produces. Computed from the rects collected so
        # far + the tap-implant box so it matches the tap version exactly.
        if stable_extent and not draw_well_tap:
            drawn_x0 = min(b[0] for (_k, b) in rects)
            drawn_y0 = min(b[1] for (_k, b) in rects)
            drawn_x1 = max(b[2] for (_k, b) in rects)
            drawn_y1 = max(b[3] for (_k, b) in rects)
            tap_implant_box = (
                tap_comp[0] - tap_ext,
                tap_comp[1] - tap_ext,
                tap_comp[2] + tap_ext,
                tap_comp[3] + tap_ext,
            )
            # the tap version's nwell also extends to enclose the tap NCOMP, so fold
            # that nwell extent in (matches the draw_well_tap=True nwell box).
            tap_nwell_x0 = tap_comp[0] - r.nwell_ncomp_overlap
            tap_nwell_x1 = tap_comp[2] + r.nwell_ncomp_overlap
            tap_nwell_y0 = tap_comp[1] - r.nwell_ncomp_overlap
            placement_extent = (
                min(drawn_x0, tap_implant_box[0], tap_nwell_x0),
                min(drawn_y0, tap_implant_box[1], tap_nwell_y0),
                max(drawn_x1, tap_implant_box[2], tap_nwell_x1),
                max(drawn_y1, tap_implant_box[3]),
            )

    # --- NMOS P+ substrate tap (opt-in; ties bulk to the substrate net) ------
    # An implant-distinguished p-substrate NMOS bulk needs a drawn P+ tap. The
    # tap sits below the source rail (in the field band), fully clear of the
    # device Nplus by NP.3a.
    #
    # TAP-NET CONTRACT: the tap M1 is the BULK terminal's OWN island -- it
    # carries the 'b' net, NEVER the source rail's net. The historical weld
    # (tap M1 risen into the source rail) was correct only for source net ==
    # bulk net; for a source != bulk NMOS (the t4 MM15: s=net08 through a
    # resistor to GND, b=GND) it physically merged the source net into the
    # substrate (KLayout extracted M$55 with s==b and folded net08 into GND).
    # The island geometry is NET-INDEPENDENT (no branch on net names): an
    # s==b device stays electrically correct through the substrate (the LVS
    # deck's ``connect_global(ptap, ...)``) and through the 'b' pin the router
    # lands on the island. The island:
    #   * encloses the cut by max(CO.6, M1.LE.1) on every side (the pad
    #     terminates over the cut in all four directions);
    #   * meets the PDK M1 min AREA on its own (it no longer borrows the
    #     rail's area);
    #   * keeps >= the M1 min SPACE below the source rail (pushed DOWN when
    #     the natural NP.3a position would encroach -- moving the tap away
    #     from the device only grows its implant clearances).
    if (not is_pmos) and draw_substrate_tap and substrate_implant is not None:
        nplus_y0 = 0 - implant_overlap
        tap_size = max(csize + 2 * comp_enc + 80, 600)
        tap_ext = r.implant_extension
        tap_cx = (comp_x0 + comp_x1) // 2
        # standalone pad side: cut-enclosing on every side AND the M1 min-area
        # square (ceil(sqrt(min_area)); 0 = no rule -> floor inert) AND min width.
        area_side = _m1_min_area_side_nm(r.metal1_min_area)
        tap_pad = max(csize + 2 * m1_cut_enc, r.metal1_min_width, area_side)
        tap_m1_half = (tap_pad + 1) // 2
        tap_cy = nplus_y0 - r.implant_to_opp_comp - tap_ext - tap_size // 2
        # M1.S.1 to the source rail above: clamp the tap DOWN so the island's
        # top edge keeps the min space to the rail's bottom edge.
        rail_bot = src_rail_cy - m1_w // 2
        tap_cy = min(tap_cy, rail_bot - r.metal1_min_space - tap_m1_half)
        tap_comp = (
            tap_cx - tap_size // 2,
            tap_cy - tap_size // 2,
            tap_cx + tap_size // 2,
            tap_cy + tap_size // 2,
        )
        tap_implant_box = (
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
        tap_m1 = (
            tap_cx - tap_m1_half,
            tap_cy - tap_m1_half,
            tap_cx + tap_m1_half,
            tap_cy + tap_m1_half,
        )
        rects.append((keys.active, tap_comp))
        rects.append((substrate_implant, tap_implant_box))
        rects.append((keys.contact, tap_co))
        rects.append((keys.metal1, tap_m1))
        tap_bbox = tap_co

    # PLACEMENT-NEUTRAL local body tap: reserve the source-tied tap version's extent
    # (its implant bottom at the natural tap_cy) so the placement footprint is
    # byte-identical to the self-bulk PMOS -- the local body tap only differs in drawn
    # artwork (clamped down by the min-space to the source rail). Keep any larger
    # drawn extent (defensive) and any stable_extent reservation already set.
    if well_tap_isolated and placement_extent is None:
        dx0 = min(b[0] for (_k, b) in rects)
        dy0 = min(b[1] for (_k, b) in rects)
        dx1 = max(b[2] for (_k, b) in rects)
        dy1 = max(b[3] for (_k, b) in rects)
        placement_extent = (dx0, min(dy0, _iso_src_tied_min_y), dx1, dy1)

    return _RawGeom(
        rects=rects,
        comp_bbox=comp_bbox,
        source_rail_bbox=src_rail_bbox,
        drain_rail_bbox=drain_rail_bbox,
        gate_bus_bbox=gate_bus_bbox,
        gate_contact_bbox=gate_contact_bbox,
        nwell_bbox=nwell_bbox,
        tap_bbox=tap_bbox,
        placement_extent=placement_extent,
        well_tap_isolated=well_tap_isolated,
    )


# --- public builder ----------------------------------------------------------


def build_device(
    tech,
    kind: str,
    W: float,
    L: float,
    nf: int = 1,
    *,
    mirror: bool = False,
    axis_x: Optional[int] = None,
    orient: str = "R0",
    draw_substrate_tap: Optional[bool] = None,
    draw_well_tap: bool = True,
    stable_extent: bool = False,
    nets: Optional[Dict[str, str]] = None,
    substrate_rail_nets: Sequence[str] = (),
    contact_rows: bool = False,
    fold_rows: int = 1,
    oxide: str = "thin",
) -> DeviceGeom:
    """Build a PDK-parameterized m=N MOS device cell.

    Parameters
    ----------
    tech : LayoutTech
        The aggregated PDK technology surface (``LayoutTech.resolve(pdk_id)``).
    kind : {'nmos','pmos'}
    W, L : float
        Channel width / length in METRES (the .cir convention). ``nf`` fingers,
        each ``W`` wide (total width = ``nf * W``); clamped up to the PDK minimums.
    nf : int
        Number of folded gate fingers (>= 1).
    mirror : bool
        When True the WHOLE cell (polygons + terminal map) is REFLECTED about
        ``axis_x`` -- so the physical column carrying 's' vs 'd' is swapped. This
        is what a placed mirror-pair member needs so its schematic source/drain
        net attaches to the correct physical diffusion column.
    axis_x : int, optional
        The vertical reflection axis (required when ``mirror`` is True).
    orient : {'R0','R90','R180','R270'}
        The device ORIENTATION. The device is built once at R0 (vertical-gate)
        and then rotated about the origin by this 90-degree multiple
        (counter-clockwise) and renormalised into the first quadrant -- a clean
        affine transform of EVERY polygon + EVERY terminal bbox, so an R90/R270
        device has its width/height swapped and its terminal map rotated
        consistently. Connectivity is UNCHANGED (rotation is LVS-invariant: a
        terminal keeps its net, only its geometry moves). Composes with
        ``mirror`` (rotation is applied first, then the x-reflection).
    draw_substrate_tap : bool, optional
        Draw the NMOS P+ substrate tap (default: follow
        ``tech.bulk_topology()['drawn_p_tap']`` UNLESS legacy gf180 parity is in
        force -- the legacy ``build_foldable_device`` did NOT draw an NMOS tap, so
        the gf180 delegation passes ``False`` for byte-parity). Ignored for PMOS.
    draw_well_tap : bool, optional
        Draw the PMOS in-well N+ well-tap (default True -> byte-parity for gf180 +
        every existing caller). Pass ``False`` for a PMOS whose SOURCE != BULK in a
        SHARED-NWELL layout: the device then draws only its nwell (no source-tied
        tap pad), and its bulk is supplied by the contiguous shared well + the well
        taps on the self-bulk (source=bulk) PMOS. Ignored for NMOS.
    nets : dict, optional
        Optional ``{'s','d','g','b' -> net name}`` binding to stamp on the terminal
        map (router/extractor net attach). Roles absent here bind to ''.
    substrate_rail_nets : sequence[str], optional
        Additional global-substrate rail names derived from the cell's declared
        port roles.  This lets a ``ground`` port named ``SUB`` materialize a P+
        tap without classifying arbitrary private body nets by spelling.
    oxide : {'thin','thick'}
        The gate-oxide domain. ``'thin'`` (default, 1.8V) draws NO OD2 -> the build
        is BYTE-IDENTICAL to before (every existing caller is unaffected). ``'thick'``
        (3.3V) additionally emits the OD2 thick-gate-oxide marker over the channel,
        enclosing the COMP by the OD2.E.1/E.2 enclosures, so the device is recognised
        + DRC-checked as 3.3V (not silently extracted as 1.8V). Requires the PDK to
        declare an OD2 layer; ``'thick'`` on a PDK without one is a hard error (the
        thick-ox intent cannot be materialised).

    Returns
    -------
    DeviceGeom
        Polygons on tech-resolved layer keys + the terminal map.
    """
    if kind not in ("nmos", "pmos"):
        raise ValueError("kind must be 'nmos' or 'pmos'")
    if nf < 1:
        raise ValueError("nf must be >= 1")
    if fold_rows < 1:
        raise ValueError("fold_rows must be >= 1")
    _oxide = str(oxide or "thin").strip().lower()
    if _oxide not in ("thin", "thick"):
        raise ValueError("oxide must be 'thin' or 'thick'")
    if mirror and axis_x is None:
        raise ValueError("axis_x is required when mirror=True")
    if orient not in _ORIENTS:
        raise ValueError(f"orient must be one of {_ORIENTS!r}, got {orient!r}")

    r = _rules_from_tech(tech)
    keys = _resolve_layer_keys(tech, kind)

    l_nm = max(int(round(float(L) * 1e9)), r.gate_min_length)
    w_nm = max(int(round(float(W) * 1e9)), r.channel_min_width)

    if draw_substrate_tap is None:
        bt = tech.bulk_topology()
        draw_substrate_tap = bool(bt.get("drawn_p_tap", False))

    # P-C1 band-level shared substrate tap (serpentine NMOS). Three states from
    # the DESIGN nets (leakage-free -- net names only, never coordinates):
    #   * "shared"   -- source net == body net (a SELF-BULK NMOS, e.g. source==
    #                   GND==body): draw the band-level P+ tap STRIP tied INTO the
    #                   source rail (VSUB->GND by construction, no competing pin).
    #   * "none"     -- source net != body net (the MM15 family: source=net08,
    #                   body=GND): draw NO tap; the bulk rides the GLOBAL substrate
    #                   net, tied to GND through the self-bulk NMOS taps elsewhere.
    #   * "isolated" -- nets unknown (standalone build, no design context): keep
    #                   the historical per-device isolated P+ tap island with a
    #                   distinct 'b' pin (backward-compatible -- a no-net folded
    #                   device still exposes a routable bulk terminal).
    _nets = nets or {}
    _s_net = str(_nets.get("s", ""))
    _b_net = str(_nets.get("b", ""))
    if _s_net and _b_net:
        band_tap_mode = "shared" if _s_net == _b_net else "none"
    else:
        band_tap_mode = "isolated"

    # BODY-NET FIDELITY (F2): the NMOS P+ substrate tap ties the body into the
    # deck's ONE global substrate net (connect_global(ptap, substrate) + the NMOS
    # bulk "W" => sub). Drawing it is correct ONLY when the declared body net IS
    # the substrate rail (ground). For a device whose body is a PRIVATE net -- a
    # source-degenerated NMOS with body tied to its own non-ground source net (the
    # two_stage OTA MM15: source==body==net08, net08 reaching gnd only through the
    # 1.5K RR5) -- drawing the tap merges that private net into the global
    # substrate = gnd (the net08<->gnd short: RR5 collapses to a shorted resistor,
    # MM15 mis-ties). So draw the tap only for a RAIL body; never ADD one the PDK
    # default would not draw (byte-parity gate: a gnd/vss-body NMOS is unchanged;
    # PDKs with no drawn p-tap are already False). Keyed on the body net alone --
    # leakage-free, no circuit / net-name / PDK branch.
    if (
        draw_substrate_tap
        and kind == "nmos"
        and _b_net
        and not _net_is_substrate_rail(_b_net, substrate_rail_nets)
    ):
        draw_substrate_tap = False

    # PMOS in-well well-tap policy, DERIVED from the device's OWN declared s/b nets
    # (leakage-free -- net names only, never coordinates):
    #   * "source_tied" -- source net == bulk net (a self-bulk PMOS: tail / output,
    #                      source==bulk==vdd): the tap M1 rises INTO the source rail
    #                      (schematic-correct, BYTE-IDENTICAL to the legacy recipe).
    #   * "isolated"    -- source net != bulk net (the LDO cascode PMOS MM13/MM14/
    #                      MM18: source=net073/058/056, body=VDDA): the tap is a LOCAL
    #                      well tap on the BODY net (its OWN M1 island, never tied to
    #                      source). The source-tied weld dropped body!=source and the
    #                      production extraction merged the source net onto VDDA.
    # Unknown nets (standalone build, no design context) -> "source_tied" so the
    # no-nets parity recipe is byte-unchanged (tests/test_device_cell_pdk.py).
    if _s_net and _b_net and _s_net != _b_net:
        well_tap_mode = "isolated"
    else:
        well_tap_mode = "source_tied"

    if fold_rows > 1:
        # SHARED-DIFFUSION (serpentine) multi-row fold: ONE device whose total
        # channel width is laid across ``fold_rows`` bands sharing a CONTINUOUS
        # active COMP -- LVS-extracts as one MOS of the conserved total W (NOT the
        # fragments abutting independent strips produce). ``contact_rows`` /
        # ``stable_extent`` are single-row-only knobs (the serpentine has its own
        # per-band lattice + drawn extent); they are inapplicable here.
        if contact_rows:
            raise ValueError("contact_rows is not supported with fold_rows>1")
        raw = _build_serpentine_geom(
            kind=kind,
            l_nm=l_nm,
            w_nm=w_nm,
            nf=nf,
            fold_rows=fold_rows,
            keys=keys,
            r=r,
            draw_substrate_tap=draw_substrate_tap,
            substrate_implant=keys.tap_implant,
            draw_well_tap=draw_well_tap,
            band_tap_mode=band_tap_mode,
            well_tap_mode=well_tap_mode,
        )
    else:
        raw = _build_device_geom(
            kind=kind,
            l_nm=l_nm,
            w_nm=w_nm,
            nf=nf,
            keys=keys,
            r=r,
            draw_substrate_tap=draw_substrate_tap,
            substrate_implant=keys.tap_implant,
            draw_well_tap=draw_well_tap,
            stable_extent=stable_extent,
            contact_rows=contact_rows,
            well_tap_mode=well_tap_mode,
        )

    # THICK-oxide (3.3V) marker: draw OD2 over the channel, enclosing the COMP by the
    # OD2.E.1/E.2 enclosures. Injected into the pure ``_RawGeom`` BEFORE the
    # orient/mirror pass so the marker rotates/reflects with the device. THIN devices
    # skip this entirely -> the rects are byte-identical to the pre-change build.
    if _oxide == "thick":
        od2_key = _od2_key(tech)
        if od2_key is None:
            if not _thick_oxide_is_default(tech):
                raise ValueError(
                    f"oxide='thick' requested but {getattr(tech, 'pdk_id', '?')!r} "
                    "declares no OD2 (thick-gate-oxide) layer -- the thick-ox intent "
                    "cannot be materialised on this PDK"
                )
        else:
            od2_box = _od2_box_over_comp(raw.comp_bbox)
            raw = replace(raw, rects=list(raw.rects) + [(od2_key, od2_box)])

    return _geom_from_raw(
        raw, kind, l_nm, w_nm, nf, tech,
        mirror=mirror, axis_x=axis_x, orient=orient, nets=nets,
    )


def _resolve_layer_keys(tech, kind: str) -> _LayerKeys:
    """Resolve the per-device layer keys from ``tech`` + ``bulk_topology`` (a DATA
    branch: implant-distinguished PDKs draw a shared COMP + Nplus/Pplus implants;
    diffusion-distinguished PDKs draw ndiff/pdiff active and no implant)."""
    bt = tech.bulk_topology()
    implant_dist = bool(bt.get("implant_distinguished", True))
    is_pmos = kind == "pmos"

    if implant_dist:
        active = tech.build_layer("diff")
        gate_implant = tech.build_layer("pplus" if is_pmos else "nplus")
        # the tap implant is the OPPOSITE type: PMOS in-well tap = Nplus; NMOS
        # substrate tap = Pplus.
        tap_implant = tech.build_layer("nplus" if is_pmos else "pplus")
    else:
        # diffusion-distinguished (e.g. sky130): separate ndiff/pdiff active, no
        # implant grown over the gate.
        active = tech.build_layer("pdiff" if is_pmos else "ndiff")
        gate_implant = None
        tap_implant = tech.build_layer("ndiff" if is_pmos else "pdiff")

    return _LayerKeys(
        active=active,
        poly=tech.build_layer("poly"),
        contact=tech.contact_key(),
        metal1=tech.metal_key(1),
        metal1_label=tech.label_key(1),
        gate_implant=gate_implant,
        tap_implant=tap_implant,
        nwell=tech.build_layer("nwell"),
    )


def _geom_from_raw(
    raw: _RawGeom,
    kind: str,
    l_nm: int,
    w_nm: int,
    nf: int,
    tech,
    *,
    mirror: bool,
    axis_x: Optional[int],
    nets: Optional[Dict[str, str]],
    orient: str = "R0",
) -> DeviceGeom:
    """Turn the pure ``_RawGeom`` into a ``DeviceGeom``, applying the device
    ORIENTATION (``orient`` rotation about the origin + first-quadrant
    normalisation) and then the x-reflection about ``axis_x`` when ``mirror``
    (polygons + every feature bbox), and building the terminal map with net
    bindings.

    Order is ROTATE-then-MIRROR: ``orient`` is the device's intrinsic
    orientation (a property of the cell), and ``mirror``/``axis_x`` is the
    placer's cell-frame reflection of the (already-oriented) cell. Rotation
    leaves connectivity untouched -- a terminal keeps its net, only its bbox
    moves -- so the device is LVS-invariant under ``orient``."""
    polys = [_rect(key, box) for (key, box) in raw.rects]

    comp_bbox = raw.comp_bbox
    src_bbox = raw.source_rail_bbox
    drn_bbox = raw.drain_rail_bbox
    gbus_bbox = raw.gate_bus_bbox
    gc_bbox = raw.gate_contact_bbox
    nwell_bbox = raw.nwell_bbox
    tap_bbox = raw.tap_bbox
    placement_extent = raw.placement_extent

    # --- ORIENTATION: rotate the whole device about the origin by ``orient``,
    # then normalise back into the first quadrant so the rotated device's
    # lower-left corner lands where the R0 device's drawn extent did (the placer
    # translates by ``extent()``, so this only fixes the local frame; it never
    # shifts the placement). For R90/R270 the width/height are swapped. A defensive
    # grid re-snap confirms the rotated edges are on the PDK grid (they already are
    # -- a 90-degree rotation of on-grid ints is on-grid).
    if orient != "R0":
        polys = [_rotate_poly(lp, orient) for lp in polys]
        comp_bbox = _rotate_box(comp_bbox, orient)
        src_bbox = _rotate_box(src_bbox, orient)
        drn_bbox = _rotate_box(drn_bbox, orient)
        gbus_bbox = _rotate_box(gbus_bbox, orient)
        gc_bbox = _rotate_box(gc_bbox, orient)
        nwell_bbox = _rotate_box(nwell_bbox, orient) if nwell_bbox else None
        tap_bbox = _rotate_box(tap_bbox, orient) if tap_bbox else None
        placement_extent = (
            _rotate_box(placement_extent, orient) if placement_extent else None
        )
        # normalise: translate so the rotated DRAWN extent's lower-left is at the
        # SAME (x0, y0) the R0 drawn extent had (the R0 COMP started at x = comp_x0,
        # y = 0; after rotation we shift the whole device so its min corner returns
        # to the original R0 drawn-extent min corner -- keeping the cell anchored).
        r0_min = (
            min(b[0] for (_k, b) in raw.rects),
            min(b[1] for (_k, b) in raw.rects),
        )
        rot_boxes = [_bbox(lp) for lp in polys]
        rot_min = (min(b[0] for b in rot_boxes), min(b[1] for b in rot_boxes))
        ndx, ndy = r0_min[0] - rot_min[0], r0_min[1] - rot_min[1]
        if ndx or ndy:
            polys = [
                LayerPolygon(
                    layer=lp.layer, datatype=lp.datatype,
                    polygon=Polygon(
                        vertices=[Point(v.x + ndx, v.y + ndy) for v in lp.polygon.vertices]
                    ),
                )
                for lp in polys
            ]

            def _shift(b: Optional[Bbox]) -> Optional[Bbox]:
                if b is None:
                    return None
                return (b[0] + ndx, b[1] + ndy, b[2] + ndx, b[3] + ndy)

            comp_bbox = _shift(comp_bbox)
            src_bbox = _shift(src_bbox)
            drn_bbox = _shift(drn_bbox)
            gbus_bbox = _shift(gbus_bbox)
            gc_bbox = _shift(gc_bbox)
            nwell_bbox = _shift(nwell_bbox)
            tap_bbox = _shift(tap_bbox)
            placement_extent = _shift(placement_extent)

        # defensive grid re-snap (rotation of on-grid ints is on-grid; this only
        # confirms it -- min edges DOWN, max edges UP so a snap never shrinks a
        # feature and the same-layer width/space the R0 device guaranteed is kept).
        g = _device_grid_nm(tech)
        if g > 1:
            def _snap_poly(lp: LayerPolygon) -> LayerPolygon:
                return LayerPolygon(
                    layer=lp.layer, datatype=lp.datatype,
                    polygon=Polygon(
                        vertices=[
                            Point((v.x // g) * g, (v.y // g) * g)
                            for v in lp.polygon.vertices
                        ]
                    ),
                )
            polys = [_snap_poly(lp) for lp in polys]

    if mirror:
        ax = int(axis_x)
        polys = [_reflect_poly_x(lp, ax) for lp in polys]
        comp_bbox = _reflect_box_x(comp_bbox, ax)
        src_bbox = _reflect_box_x(src_bbox, ax)
        drn_bbox = _reflect_box_x(drn_bbox, ax)
        gbus_bbox = _reflect_box_x(gbus_bbox, ax)
        gc_bbox = _reflect_box_x(gc_bbox, ax)
        nwell_bbox = _reflect_box_x(nwell_bbox, ax) if nwell_bbox else None
        placement_extent = (
            _reflect_box_x(placement_extent, ax) if placement_extent else None
        )
        tap_bbox = _reflect_box_x(tap_bbox, ax) if tap_bbox else None

    nets = nets or {}

    def _term(role: str, box: Optional[Bbox]) -> Optional[Terminal]:
        if box is None:
            return None
        return Terminal(role=role, bbox=box, net=nets.get(role, ""))

    terminals: Dict[str, Terminal] = {}
    for role, box in (
        ("s", src_bbox),
        ("d", drn_bbox),
        ("g", gc_bbox),
    ):
        t = _term(role, box)
        if t is not None:
            terminals[role] = t
    # bulk terminal: PMOS nwell tap, or NMOS substrate tap (when drawn). For an
    # NMOS with no drawn tap the bulk is the global substrate (no local pin). A
    # PMOS built WITHOUT a well tap (``draw_well_tap=False``: a source != bulk
    # device in a SHARED-NWELL layout) has NO bulk PIN -- its bulk is the contiguous
    # shared well tapped at vdd by the self-bulk PMOS, so there is no per-device tie
    # for the router to land a riser on (the well-merge does the connection). Only a
    # PMOS that actually drew its tap (``tap_bbox is not None``) exposes a 'b' pin.
    if kind == "pmos" and tap_bbox is not None:
        # the bulk pin is the in-well tap contact (ties the well body to bulk). For a
        # LOCAL (source != bulk) well tap the terminal is retained for locatability
        # but the device is flagged ``well_tap_isolated`` so the router does NOT rise a
        # separate body-net pad to it (that pad, on the body net, would land next to
        # the different-net source rail and re-short it -- the LDO cascode merge).
        terminals["b"] = Terminal(
            role="b", bbox=tap_bbox, net=nets.get("b", "")
        )
    elif kind == "nmos" and tap_bbox is not None:
        terminals["b"] = Terminal(
            role="b", bbox=tap_bbox, net=nets.get("b", tech.substrate_net())
        )

    return DeviceGeom(
        polygons=polys,
        terminals=terminals,
        kind=kind,
        finger_count=nf,
        gate_length_nm=l_nm,
        channel_width_nm=w_nm,
        comp_bbox=comp_bbox,
        mirror=mirror,
        axis_x=axis_x,
        orient=orient,
        nwell_bbox=nwell_bbox,
        source_rail_bbox=src_bbox,
        drain_rail_bbox=drn_bbox,
        gate_bus_bbox=gbus_bbox,
        gate_contact_bbox=gc_bbox,
        placement_extent=placement_extent,
        well_tap_isolated=raw.well_tap_isolated,
    )


# --- shared-diffusion ARRAY (the compactness core) ---------------------------
#
# A current-mirror / common-source ARRAY is N same-polarity members (each an m=nf
# leg) whose fingers live on ONE continuous merged OD, adjacent fingers sharing a
# source/drain diffusion column at gap 0. This is NOT N separate ``build_device``
# stamps abutted (KLayout would fold those into fragments / short across the abut
# seam); it is ONE strip whose per-column net is dealt out from the ordered member
# list so that:
#   * every SHARED-SOURCE column carries the one common source net (all mirror /
#     CS legs tie their source to the same rail);
#   * every DRAIN column carries ITS member's drain net;
#   * a gate finger separates every adjacent column, so two DIFFERENT-net columns
#     never share un-gated diffusion -- the exact cross-net short that broke the
#     earlier ad-hoc shared-COMP folds.
# Correctness condition (LVS-clean by construction): each member (except the last)
# ends on a SHARED-SOURCE column so it abuts the next member on that source column.
# A member ends on source iff its finger count nf is EVEN. An interior member with
# ODD nf would end on a DRAIN column that abuts the next member's source -> a
# separating dummy finger is required (Stage 2 ``@dummy_flank``); until then such a
# layout is UNSUPPORTED (a hard error, never a silent short).


@dataclass(frozen=True)
class ArrayColumn:
    """One S/D diffusion/contact column of a shared-diffusion array.

    ``role``   -- 's' (a shared-source column) or 'd' (a member's drain column).
    ``net``    -- the cell net this column carries (source_net for 's'; the owning
                  member's drain_net for 'd').
    ``member`` -- the owning member name for a drain column; '' for a shared source
                  column (it belongs to every adjacent member equally).
    ``x_center`` -- the column centre x (nm).
    ``contact_bbox`` -- the bbox of the column's (topmost) contact cut (nm).
    ``m1_strip_bbox`` -- the full-height M1 strip commoning the column's cuts (nm).
    """

    role: str
    net: str
    member: str
    x_center: int
    contact_bbox: Bbox
    m1_strip_bbox: Bbox
    # y-band centre (nm) for a FOLDED array column; None for a single-row array
    # (member_pins then uses the shared channel centre -> byte-identical off).
    cy: Optional[int] = None


@dataclass(frozen=True)
class ArrayFinger:
    """One poly gate finger of the array. ``gate_net`` is the shared gate; ``member``
    the owning member name; ``x0``/``x1`` the finger's poly x-extent (nm)."""

    member: str
    gate_net: str
    x0: int
    x1: int
    # y-band centre (nm) of this finger for a FOLDED array; None for single-row.
    cy: Optional[int] = None


def _validate_body_tap_owner_sha256(value: object) -> str:
    """Canonical external body-tap owner token; empty keeps the local tap."""

    if type(value) is not str:
        raise TypeError("body_tap_owner_sha256 must be an exact string")
    if value == "":
        return value
    if (
        len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "body_tap_owner_sha256 must be empty or 64 lowercase hex characters"
        )
    return value


@dataclass(frozen=True)
class SharedDiffusionArray:
    """A materialized shared-diffusion current-mirror / common-source array.

    The device_cell-native geometry object for an ``@array(style=shared_diffusion)``
    group: polygons on tech-resolved layer keys plus the per-column / per-finger net
    map the router (Task 4) attaches to and the LVS gate (Task-3 test) proves clean.

    NB: this is NOT the extract-side ``extract/library_device.CellGeometry`` (which
    is centroid-relative um and carries NO nets); the LVS gate needs per-column nets,
    so the array returns this net-aware object. ``polygons`` / ``layers`` / ``extent``
    mirror ``DeviceGeom`` so the same emit/placement code can iterate it.
    """

    polygons: List[LayerPolygon]
    kind: str
    order: str
    gate_net: str
    source_net: str
    gate_length_nm: int
    channel_width_nm: int
    comp_bbox: Bbox
    columns: List[ArrayColumn]
    fingers: List[ArrayFinger]
    source_rail_bbox: Bbox
    gate_bus_bbox: Bbox
    gate_contact_bbox: Bbox
    drain_straps: Dict[str, Bbox] = field(default_factory=dict)  # member -> strap bbox
    nwell_bbox: Optional[Bbox] = None
    body_net: str = ""
    body_tap_bbox: Optional[Bbox] = None
    well_tap_isolated: bool = False
    body_tap_owner_sha256: str = ""

    @property
    def finger_count(self) -> int:
        return len(self.fingers)

    @property
    def layers(self) -> Dict[LayerKey, List[LayerPolygon]]:
        out: Dict[LayerKey, List[LayerPolygon]] = {}
        for lp in self.polygons:
            out.setdefault((lp.layer, lp.datatype), []).append(lp)
        return out

    def polys_on(self, key: LayerKey) -> List[LayerPolygon]:
        return [lp for lp in self.polygons if (lp.layer, lp.datatype) == key]

    def extent(self) -> Bbox:
        boxes = [_bbox(lp) for lp in self.polygons]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    def member_pins(self, name: str) -> Dict[str, Tuple[int, int]]:
        """Per-terminal ROUTER access pins for one array member, in the array's OWN
        (build-frame nm) coordinate frame -- derived PURELY from the existing column
        / finger net map, NO new geometry.

        Each member owns a contiguous finger block (declaration order); this returns
        ``{'d': (x, y), 's': (x, y), 'g': (x, y)}``:

          * ``'d'`` -- the centre of a DRAIN column carrying THIS member's own drain
            net (a column with ``role == 'd'`` and ``member == name``), taken at the
            shared channel centre so the point sits INSIDE the merged OD strip on the
            member's own drain-strap x. DISTINCT per member (members own disjoint
            finger blocks -> disjoint drain columns): this is exactly what lets the
            router tell the members' drains apart. The measured S1 collapse (7/15
            nets) was every member probed at ONE shared sub-centroid, so the drain
            nets could not be separated.
          * ``'s'`` -- a SHARED-SOURCE column (``role == 's'``, the one shared source
            net) nearest this member's block, at the channel centre (on that source
            column's full-height M1 strip).
          * ``'g'`` -- the array's ONE shared gate access: the gate-contact M1 pad
            over the gate bus. Every member shares the single gate net, so the
            routable gate landing is the one poly->M1 gate contact, not a per-member
            poly-bus point (the router lands on M1, never bare poly).

        ``name`` must own >= 1 drain column (a bridged mirror / common-source
        member); an unknown member raises ``KeyError``. Positions are integer nm in
        the same frame as ``columns[i].x_center`` / ``comp_bbox`` / ``fingers`` --
        the caller carries them into the cell frame via the array tile centre."""
        drains = [c for c in self.columns if c.role == "d" and c.member == name]
        if not drains:
            owners = sorted({c.member for c in self.columns if c.member})
            raise KeyError(
                f"{name!r} owns no drain column in this shared-diffusion array "
                f"(members with drains: {owners})"
            )
        blk = [f for f in self.fingers if f.member == name]
        fmid = (
            (min(f.x0 for f in blk) + max(f.x1 for f in blk)) // 2
            if blk else drains[0].x_center
        )
        # shared channel centre (both S/D pins land inside the merged OD strip).
        cy = (self.comp_bbox[1] + self.comp_bbox[3]) // 2
        # representative drain / source columns: nearest this member's block centre
        # (deterministic; x_center tiebreak).
        dcol = min(drains, key=lambda c: (abs(c.x_center - fmid), c.x_center))
        # FOLDED array: a member owns columns in its OWN y-band, so the pin must
        # land at the column's band centre (columns[i].cy), not the whole-strip
        # centre -- else two members stacked at the same x-slot return an identical
        # pin. A single-row array leaves cy=None -> the shared centre (byte-stable).
        dcy = dcol.cy if getattr(dcol, "cy", None) is not None else cy
        # prefer a shared-source column in THIS member's band (fold), else any.
        sources = [c for c in self.columns if c.role == "s"]
        band_sources = [c for c in sources if getattr(c, "cy", None) == dcol.cy]
        pool = band_sources if band_sources else sources
        scol = min(pool, key=lambda c: (abs(c.x_center - fmid), c.x_center))
        scy = scol.cy if getattr(scol, "cy", None) is not None else cy
        # shared gate access: the gate-contact M1 pad centre (the poly->M1 landing;
        # its centre is inside the drawn M1 gate pad, so the pin is on routable M1).
        gx = (self.gate_contact_bbox[0] + self.gate_contact_bbox[2]) // 2
        gy = (self.gate_contact_bbox[1] + self.gate_contact_bbox[3]) // 2
        pins = {
            "d": (int(dcol.x_center), int(dcy)),
            "s": (int(scol.x_center), int(scy)),
            "g": (int(gx), int(gy)),
        }
        if self.well_tap_isolated and self.body_tap_bbox is not None:
            bx0, by0, bx1, by1 = self.body_tap_bbox
            pins["b"] = ((bx0 + bx1) // 2, (by0 + by1) // 2)
        return pins


def _member_fields(m):
    """Normalise one member tuple carrying the declared four-terminal MOS nets.

    The canonical form is ``(name, kind, W_um, L_um, nf, gate, drain, source,
    body)``.  The historical eight-field public builder form remains accepted and
    means ``body == source``; RowProgram materialization always supplies the ninth
    field so a source/body split can never be guessed or silently source-tied.
    """
    values = tuple(m)
    if len(values) == 8:
        name, kind, w_um, l_um, nf, gate_net, drain_net, source_net = values
        body_net = source_net
    elif len(values) == 9:
        name, kind, w_um, l_um, nf, gate_net, drain_net, source_net, body_net = values
    else:
        raise ValueError(
            "array member must have 8 legacy fields or 9 four-terminal fields "
            f"(got {len(values)})"
        )
    return (
        str(name),
        str(kind),
        float(w_um),
        float(l_um),
        int(nf),
        str(gate_net),
        str(drain_net),
        str(source_net),
        str(body_net),
    )


def _materialize_shared_pmos_well_tap(
    *,
    keys: _LayerKeys,
    r: DeviceCellRules,
    comp_bbox: Bbox,
    source_rail_bbox: Bbox,
    source_net: str,
    body_net: str,
    base_nwell_bbox: Bbox,
    forbidden_poly_bboxes: Tuple[Bbox, ...] = (),
) -> Tuple[List[Tuple[LayerKey, Bbox]], Bbox, Bbox, bool]:
    """Materialize one N+ tie for one continuous PMOS N-well.

    A self-body array welds the tap M1 into its source rail.  A source/body split
    leaves one DRC-spaced M1 island which :meth:`SharedDiffusionArray.member_pins`
    exposes as the routable ``b`` terminal.  The obligation is derived only from
    the declared nets; rail names and device names are deliberately irrelevant.

    ``forbidden_poly_bboxes`` is the poly already materialized by the owning
    device/array.  Split-gate common-centroid straps can extend below the channel
    COMP and through the source-rail band, so the tap cannot be placed from the
    main COMP bbox alone.  The complete tap stack is pushed below every poly box
    whose x-span can see its COMP, with the PDK's field-poly-to-COMP clearance.
    Ordinary one-row arrays pass their poly too and remain byte-identical when
    the natural tap site is already clear.
    """
    if not body_net:
        raise ValueError("PMOS shared N-well requires a declared body net")
    if keys.nwell is None:
        raise ValueError("PMOS shared N-well tap requires an nwell layer")

    csize = int(r.contact_size)
    comp_enc = int(r.comp_contact_enc)
    m1_cut_enc = max(int(r.metal1_contact_enc), int(r.metal1_line_end))
    m1_w = max(int(r.metal1_min_width), csize + 2 * int(r.metal1_contact_enc))
    implant_overlap = int(r.implant_gate_overlap)
    tap_size = max(csize + 2 * comp_enc + 80, 600)
    tap_ext = int(r.implant_extension)
    tap_cx = (source_rail_bbox[0] + source_rail_bbox[2]) // 2
    natural_cy = (
        comp_bbox[1]
        - implant_overlap
        - int(r.implant_to_opp_comp)
        - tap_ext
        - tap_size // 2
    )
    isolated = body_net != source_net

    if isolated:
        area_side = _m1_min_area_side_nm(r.metal1_min_area)
        pad_side = max(csize + 2 * m1_cut_enc, int(r.metal1_min_width), area_side)
        pad_half = (pad_side + 1) // 2
        tap_cy = min(
            natural_cy,
            source_rail_bbox[1] - int(r.metal1_min_space) - pad_half,
        )
    else:
        pad_half = max(csize // 2 + m1_cut_enc, m1_w // 2)
        tap_cy = natural_cy

    # The tap is deliberately below the PMOS channel field.  Preserve that
    # direction and move the whole stack only as far as required to clear any
    # already-materialized field poly at this x position.  Using the PDK rule and
    # geometry ownership here prevents a parasitic poly-over-tap-COMP transistor
    # (and its derivative DF.2a / implant-gate checks) without circuit knowledge.
    tap_half = tap_size // 2
    tap_x0 = tap_cx - tap_half
    tap_x1 = tap_cx + tap_half
    poly_clear = int(r.field_poly_to_comp)
    for px0, py0, px1, _py1 in forbidden_poly_bboxes:
        if tap_x1 + poly_clear <= int(px0):
            continue
        if tap_x0 - poly_clear >= int(px1):
            continue
        tap_cy = min(tap_cy, int(py0) - poly_clear - tap_half)

    tap_comp = (
        tap_cx - tap_size // 2,
        tap_cy - tap_size // 2,
        tap_cx + tap_size // 2,
        tap_cy + tap_size // 2,
    )
    tap_implant = (
        tap_comp[0] - tap_ext,
        tap_comp[1] - tap_ext,
        tap_comp[2] + tap_ext,
        tap_comp[3] + tap_ext,
    )
    tap_contact = (
        tap_cx - csize // 2,
        tap_cy - csize // 2,
        tap_cx + csize // 2,
        tap_cy + csize // 2,
    )
    tap_pad = (
        tap_cx - pad_half,
        tap_cy - pad_half,
        tap_cx + pad_half,
        tap_cy + pad_half,
    )
    rects: List[Tuple[LayerKey, Bbox]] = [(keys.active, tap_comp)]
    if keys.tap_implant is not None:
        rects.append((keys.tap_implant, tap_implant))
    rects.extend(((keys.contact, tap_contact), (keys.metal1, tap_pad)))
    if not isolated:
        rail_cy = (source_rail_bbox[1] + source_rail_bbox[3]) // 2
        rects.append((
            keys.metal1,
            (
                tap_cx - pad_half,
                min(tap_cy, rail_cy),
                tap_cx + pad_half,
                max(tap_cy, rail_cy),
            ),
        ))

    nx0 = min(base_nwell_bbox[0], tap_comp[0] - int(r.nwell_ncomp_overlap))
    ny0 = min(base_nwell_bbox[1], tap_comp[1] - int(r.nwell_ncomp_overlap))
    nx1 = max(base_nwell_bbox[2], tap_comp[2] + int(r.nwell_ncomp_overlap))
    ny1 = max(base_nwell_bbox[3], tap_comp[3] + int(r.nwell_ncomp_overlap))
    if nx1 - nx0 < int(r.nwell_min_width):
        mid = (nx0 + nx1) // 2
        nx0 = mid - int(r.nwell_min_width) // 2
        nx1 = nx0 + int(r.nwell_min_width)
    if ny1 - ny0 < int(r.nwell_min_width):
        mid = (ny0 + ny1) // 2
        ny0 = mid - int(r.nwell_min_width) // 2
        ny1 = ny0 + int(r.nwell_min_width)
    return rects, (nx0, ny0, nx1, ny1), tap_pad, isolated


@dataclass(frozen=True)
class _FlankDummy:
    """A validated fully-off rail ``@dummy_flank`` end-cap for a shared-diffusion
    strip: ``per_end`` gate-off fingers at EACH end, all terminals on the strip's
    source net (so it extracts as the declared ``M = 2*per_end`` dummy with no
    exposed signal net and no cross-net short)."""
    name: str
    per_end: int
    nf: int


def _validate_flank_dummies(
    flank_dummies, kind, w_nm, l_nm, source_net, body_net, r
) -> "List[_FlankDummy]":
    """Accept ONLY fully-off rail dummies (gate==drain==source==the strip source
    net) of the strip's own (kind, W, L). A drain-exposed gate-off dummy (e.g. the
    input pair's MPDUM_INP whose drain is a signal net) is a common-centroid
    end-cap (Stage 2), REFUSED here rather than folded into a continuous-OD strip
    where its drain would abut a foreign net. Honest error, no per-circuit branch."""
    out: List[_FlankDummy] = []
    for spec in (flank_dummies or ()):
        member, per_end = spec
        dn, dk, dW, dL, dnf, dg, dd, ds, db = _member_fields(member)
        per_end = int(per_end)
        if dk != kind:
            raise ValueError(
                f"flank dummy {dn!r} kind {dk!r} != strip kind {kind!r}")
        dwn = max(int(round(dW * 1000.0)), r.channel_min_width)
        dln = max(int(round(dL * 1000.0)), r.gate_min_length)
        if dwn != w_nm or dln != l_nm:
            raise ValueError(
                f"flank dummy {dn!r} W/L (nm {dwn}/{dln}) must match the strip "
                f"(nm {w_nm}/{l_nm}) -- one continuous OD needs equal W and L")
        if not (dg == dd == ds == source_net):
            raise ValueError(
                f"flank dummy {dn!r} is not a fully-off rail dummy on source "
                f"{source_net!r} (g={dg!r} d={dd!r} s={ds!r}); a drain-exposed "
                "gate-off dummy is a common-centroid end-cap (Stage 2), not a "
                "shared-diffusion strip end-cap")
        if kind == "pmos" and db != body_net:
            raise ValueError(
                f"flank dummy {dn!r} body net {db!r} != shared N-well body "
                f"net {body_net!r}"
            )
        # LVS forces ALL nf fingers of the declared M=nf dummy to materialize; a
        # symmetric flank splits them evenly -> nf//2 per end (nf must be even). The
        # intent's advisory ``per_end`` marks symmetry, not the count (the golden
        # netlist M is the truth); honor nf//2 so the extracted M matches exactly.
        if dnf < 2 or dnf % 2 != 0:
            raise ValueError(
                f"flank dummy {dn!r} nf must be even and >=2 for a symmetric "
                f"end-cap (got nf={dnf})")
        out.append(_FlankDummy(name=dn, per_end=dnf // 2, nf=dnf))
    return out


def _build_folded_shared_diffusion(
    *, tech, mem, kind, keys, r, w_nm, l_nm, total_nf, owner, local,
    gate_net, source_net, body_net, fold_rows, order, oxide,
    body_tap_owner_sha256: str = "",
) -> SharedDiffusionArray:
    """Fold the ``total_nf`` interleaved columns into ``fold_rows`` vertically
    stacked bands sharing ONE continuous merged-OD rectangle (height
    ``fold_rows * w_nm``), reusing ``_build_serpentine_geom``'s band geometry:
    ``H = fold_rows * w_nm``, ``band_cys = [b*w_nm + w_nm//2 for b in
    range(fold_rows)]``, bands abut with no gap. The ``nf_per_band =
    total_nf // fold_rows`` finger slots are laid once as FULL-HEIGHT poly
    strips (DRC-legal endcaps beyond the whole OD); band ``b`` at slot ``s``
    owns global finger ``b*nf_per_band + s``, so a member's fingers land in its
    own band(s). Contacts / M1 straps are dealt per band centre; the column net
    map is the flat interleave, re-indexed per band.

    The continuous-OD fold is net-clean ONLY when every slot-pocket carries ONE
    net across all bands (single-drain / same-drain members). When distinct-drain
    members occupy different bands at the same slot (a ratio-order mirror folded
    row-wise), their drains share one merged-OD channel -> a cross-net short; this
    builder REFUSES that (raises ValueError) rather than emit a silent short. A
    distinct-drain multi-row fold is a common-centroid interleave
    (build_common_centroid_array, Stage 2), not this continuous-OD strip.
    """
    finger_space = r.finger_space
    endcap = r.poly_endcap
    csize = r.contact_size
    sd_clear = r.contact_to_gate
    comp_enc = r.comp_contact_enc
    m1_enc = r.metal1_contact_enc
    m1_cut_enc = max(m1_enc, r.metal1_line_end)
    m1_pad = max(csize + 2 * m1_cut_enc, 380)
    m1_w = max(r.metal1_min_width, csize + 2 * m1_enc)
    bus_band = max(r.poly_min_width, 180)
    top_endcap = max(r.field_poly_to_comp + bus_band, endcap)

    nf_per_band = total_nf // fold_rows
    H = fold_rows * w_nm
    band_cys = [b * w_nm + w_nm // 2 for b in range(fold_rows)]

    # --- net-cleanliness precondition (the connectivity contract) -----------
    # The bands ABUT with NO diffusion break, so the pocket at a given slot is ONE
    # diffusion node across ALL bands (KLayout merges it; the deck reads one net).
    # The fold is net-clean ONLY when every band carries the SAME net at that slot.
    # When distinct-drain members land in different bands at the same slot (a
    # ratio-order mirror folded row-wise), their drains merge through the shared OD
    # -> a cross-net SHORT (empirically: a 2-pmos mirror folded to 2 rows merges the
    # two drain nets in every drain channel). That layout is a common-centroid
    # INTERLEAVE (build_common_centroid_array, Stage 2), NOT a continuous-OD
    # shared-diffusion strip. Refuse it here rather than silently emit a short --
    # a general precondition, no per-circuit branch.
    def _slot_net(b: int, j: int) -> str:
        if j == 0:
            return source_net
        k = b * nf_per_band + (j - 1)
        return mem[owner[k]][6] if local[k] % 2 == 0 else source_net

    for j in range(nf_per_band + 1):
        nets_j = {_slot_net(b, j) for b in range(fold_rows)}
        if len(nets_j) > 1:
            raise ValueError(
                "shared-diffusion fold would SHORT distinct nets "
                f"{sorted(nets_j)} through the continuous merged OD at slot {j}: "
                "band-stacked members carry different drains at the same slot, and "
                "the abutting bands merge them into one diffusion node. A distinct-"
                "drain multi-row fold is a common-centroid interleave -- use "
                "build_common_centroid_array (Stage 2). The continuous-OD shared-"
                "diffusion fold is net-clean only when every band carries one net "
                "per slot (single-drain / same-drain members)."
            )

    rects: List[Tuple[LayerKey, Bbox]] = []
    fingers: List[ArrayFinger] = []

    # --- gate poly fingers: nf_per_band FULL-HEIGHT strips (endcaps beyond OD) --
    finger_x: List[int] = [s * (l_nm + finger_space) for s in range(nf_per_band)]
    last_gate_x_max = finger_x[-1] + l_nm
    for x0 in finger_x:
        rects.append((keys.poly, (x0, -endcap, x0 + l_nm, H + top_endcap)))
    # one ArrayFinger record per (band, slot) global finger, band-tagged.
    for k in range(total_nf):
        b = k // nf_per_band
        s = k % nf_per_band
        fingers.append(
            ArrayFinger(
                member=mem[owner[k]][0], gate_net=gate_net,
                x0=finger_x[s], x1=finger_x[s] + l_nm, cy=band_cys[b],
            )
        )

    # --- COMP (active): ONE continuous rectangle over all bands --------------
    comp_x0 = finger_x[0] - sd_clear - csize - comp_enc
    comp_x1 = last_gate_x_max + sd_clear + csize + comp_enc
    comp_bbox = (comp_x0, 0, comp_x1, H)
    rects.append((keys.active, comp_bbox))

    implant_overlap = r.implant_gate_overlap
    if keys.gate_implant is not None:
        rects.append(
            (keys.gate_implant,
             (comp_x0 - implant_overlap, 0 - implant_overlap,
              comp_x1 + implant_overlap, H + implant_overlap))
        )

    # slot S/D x centres (nf_per_band + 1 columns per band).
    sd_centres: List[int] = [finger_x[0] - sd_clear - csize // 2]
    for i in range(nf_per_band - 1):
        gap_lo = finger_x[i] + l_nm
        gap_hi = finger_x[i + 1]
        sd_centres.append((gap_lo + gap_hi) // 2)
    sd_centres.append(last_gate_x_max + sd_clear + csize // 2)

    # --- per-band contact columns + net map (the interleave, re-indexed) -----
    pitch = csize + r.contact_space
    strip_half = m1_pad // 2
    columns: List[ArrayColumn] = []
    for b in range(fold_rows):
        by0 = b * w_nm
        y_lo = by0 + comp_enc + csize // 2
        y_hi = by0 + w_nm - comp_enc - csize // 2
        bcy = band_cys[b]
        ys = [bcy] if y_hi < y_lo else list(range(y_lo, y_hi + 1, pitch))
        for j in range(nf_per_band + 1):
            cx = sd_centres[j]
            if j == 0:
                role, net, member = "s", source_net, ""
            else:
                k = b * nf_per_band + (j - 1)
                if local[k] % 2 == 0:
                    role, net, member = "d", mem[owner[k]][6], mem[owner[k]][0]
                else:
                    role, net, member = "s", source_net, ""
            for ccy in ys:
                rects.append((keys.contact,
                              (cx - csize // 2, ccy - csize // 2,
                               cx + csize // 2, ccy + csize // 2)))
            strip_y0 = min(ys) - csize // 2 - m1_cut_enc
            strip_y1 = max(ys) + csize // 2 + m1_cut_enc
            strip = (cx - strip_half, strip_y0, cx + strip_half, strip_y1)
            rects.append((keys.metal1, strip))
            top_cut = (cx - csize // 2, max(ys) - csize // 2,
                       cx + csize // 2, max(ys) + csize // 2)
            columns.append(ArrayColumn(role=role, net=net, member=member,
                                       x_center=cx, contact_bbox=top_cut,
                                       m1_strip_bbox=strip, cy=bcy))

    # --- source rail (below COMP) commoning every source column -------------
    rail_gap = _dense_source_rail_gap(r)
    src_rail_cy = -(m1_w // 2 + rail_gap)
    source_columns_by_x: Dict[int, List[ArrayColumn]] = {}
    for column in columns:
        if column.role == "s":
            source_columns_by_x.setdefault(column.x_center, []).append(column)
    source_spans = [
        (
            x,
            min(column.m1_strip_bbox[0] for column in at_x),
            max(column.m1_strip_bbox[2] for column in at_x),
        )
        for x, at_x in sorted(source_columns_by_x.items())
    ]
    src_rail_bbox = (
        min(left for _x, left, _right in source_spans),
        src_rail_cy - m1_w // 2,
        max(right for _x, _left, right in source_spans),
        src_rail_cy + m1_w // 2,
    )
    rects.append((keys.metal1, src_rail_bbox))
    # One riser per UNIQUE x spans the PDK-derived landing sides of every source
    # column repeated at that x.  It reaches the top band centre and overlaps each
    # band's source strip, while owning the full rail-to-landing transition without
    # exact-emitter shoulder fill.
    for _x, left, right in source_spans:
        rects.append(
            (keys.metal1, (left, src_rail_cy, right, band_cys[-1]))
        )

    # --- gate bus + one gate contact above the COMP top ---------------------
    bus_y0 = H + r.field_poly_to_comp
    bus_y1 = H + top_endcap
    gate_cx = finger_x[0] - finger_space // 2
    bus_x0 = min(finger_x[0],
                 gate_cx - max(l_nm // 2, csize // 2 + r.poly_contact_enc))
    gate_bus_bbox = (bus_x0, bus_y0, last_gate_x_max, bus_y1)
    rects.append((keys.poly, gate_bus_bbox))

    gate_cy = bus_y1 + r.poly_contact_enc + csize // 2
    gate_pad_top = gate_cy + csize // 2 + r.poly_contact_enc
    gpad_half = max(m1_pad // 2, csize // 2 + r.poly_contact_enc)
    rects.append((keys.poly,
                  (gate_cx - gpad_half, bus_y0, gate_cx + gpad_half, gate_pad_top)))
    gate_contact_bbox = (gate_cx - csize // 2, gate_cy - csize // 2,
                         gate_cx + csize // 2, gate_cy + csize // 2)
    rects.append((keys.contact, gate_contact_bbox))
    g_area_side = _m1_min_area_side_nm(r.metal1_min_area)
    gpad_m1_half = max(m1_pad, g_area_side) // 2
    rects.append((keys.metal1,
                  (gate_cx - gpad_m1_half, gate_cy - gpad_m1_half,
                   gate_cx + gpad_m1_half, gate_cy + gpad_m1_half)))

    # --- per-member drain straps (above the gate) ---------------------------
    drn_rail_cy = gate_pad_top + r.field_poly_to_comp + m1_w // 2 + 50
    drain_straps: Dict[str, Bbox] = {}
    for m in mem:
        drain_columns_by_x: Dict[int, List[ArrayColumn]] = {}
        for column in columns:
            if column.role == "d" and column.member == m[0]:
                drain_columns_by_x.setdefault(column.x_center, []).append(column)
        drain_spans = [
            (
                x,
                min(column.m1_strip_bbox[0] for column in at_x),
                max(column.m1_strip_bbox[2] for column in at_x),
            )
            for x, at_x in sorted(drain_columns_by_x.items())
        ]
        if not drain_spans:
            continue
        strap = (
            min(left for _x, left, _right in drain_spans),
            drn_rail_cy - m1_w // 2,
            max(right for _x, _left, right in drain_spans),
            drn_rail_cy + m1_w // 2,
        )
        rects.append((keys.metal1, strap))
        for _x, left, right in drain_spans:
            rects.append(
                (keys.metal1, (left, band_cys[0], right, drn_rail_cy))
            )
        drain_straps[m[0]] = strap

    # --- PMOS Nwell + one declared-body tap for the continuous well ---------
    nwell_bbox: Optional[Bbox] = None
    body_tap_bbox: Optional[Bbox] = None
    well_tap_isolated = False
    if kind == "pmos" and keys.nwell is not None:
        from .device_well_geometry import shared_nwell_bbox
        base_nwell_bbox = shared_nwell_bbox((comp_bbox,), r)
        if body_tap_owner_sha256:
            nwell_bbox = base_nwell_bbox
        else:
            tap_rects, nwell_bbox, body_tap_bbox, well_tap_isolated = (
                _materialize_shared_pmos_well_tap(
                    keys=keys,
                    r=r,
                    comp_bbox=comp_bbox,
                    source_rail_bbox=src_rail_bbox,
                    source_net=source_net,
                    body_net=body_net,
                    base_nwell_bbox=base_nwell_bbox,
                    forbidden_poly_bboxes=tuple(
                        box for key, box in rects if key == keys.poly
                    ),
                )
            )
            rects.extend(tap_rects)
        rects.append((keys.nwell, nwell_bbox))

    if oxide == "thick":
        od2_key = _od2_key(tech)
        if od2_key is None:
            if not _thick_oxide_is_default(tech):
                raise ValueError(
                    f"oxide='thick' requested but {getattr(tech, 'pdk_id', '?')!r} "
                    "declares no OD2 (thick-gate-oxide) layer"
                )
        else:
            rects.append((od2_key, _od2_box_over_comp(comp_bbox)))

    polys = [_rect(key, box) for (key, box) in rects]
    return SharedDiffusionArray(
        polygons=polys, kind=kind, order=order, gate_net=gate_net,
        source_net=source_net, gate_length_nm=l_nm, channel_width_nm=w_nm,
        comp_bbox=comp_bbox, columns=columns, fingers=fingers,
        source_rail_bbox=src_rail_bbox, gate_bus_bbox=gate_bus_bbox,
        gate_contact_bbox=gate_contact_bbox, drain_straps=drain_straps,
        nwell_bbox=nwell_bbox, body_net=body_net,
        body_tap_bbox=body_tap_bbox, well_tap_isolated=well_tap_isolated,
        body_tap_owner_sha256=body_tap_owner_sha256,
    )


def build_shared_diffusion_array(
    tech,
    members,
    order: str = "ratio",
    oxide: str = "thin",
    fold_rows: int = 1,
    flank_dummies: "Sequence[Tuple[Any, int]]" = (),
    body_tap_owner_sha256: str = "",
) -> SharedDiffusionArray:
    """Materialize a ``style=shared_diffusion`` array as ONE continuous merged OD.

    Parameters
    ----------
    tech : LayoutTech
        The aggregated PDK surface -- every pitch / enclosure / implant constant is
        resolved from it EXACTLY as ``build_device`` does (``_rules_from_tech`` +
        ``_resolve_layer_keys``), so the strip is DRC-legal by construction. General
        over any PDK a ``LayoutTech`` describes; no per-circuit branch.
    members : ordered list of tuples
        ``(name, kind, W_um, L_um, nf, gate_net, drain_net, source_net,
        body_net)`` laid
        LEFT-TO-RIGHT in the given order (``nf`` = M finger count). All members must
        share one polarity (``kind``), one channel width (``W_um``) and gate length
        (``L_um``) -- one continuous OD is only physical for a uniform diffusion --
        and one shared gate net and source net (a current mirror / CS bank). Each
        member keeps its OWN drain net.
    order : {'ratio', ''}
        Both lay the members as contiguous unit-finger blocks in the given order
        (``ratio`` is the mirror-ratio ordering the caller already applied, e.g.
        diode interior). A common-centroid interleave (``ABBA``/``BAAB``/``AABB``)
        is Stage 2 (``build_common_centroid_array``); it is rejected here.
    oxide : {'thin', 'thick'}
        ``thin`` (default) draws no OD2. ``thick`` (3.3V) additionally grows the OD2
        thick-gate-oxide marker over the channel (OD2.E.1/E.2 enclosures), exactly
        as ``build_device``; ``thick`` on a PDK with no OD2 layer is a hard error.

    Returns
    -------
    SharedDiffusionArray
        Polygons on tech-resolved keys + the per-column / per-finger net map. The
        OD is ONE merged rectangle; poly carries ``sum(nf)`` fingers; every S/D
        column carries exactly one net; different-net columns are always separated
        by a gate finger (no cross-net diffusion short).
    """
    body_tap_owner_sha256 = _validate_body_tap_owner_sha256(
        body_tap_owner_sha256
    )
    _order = str(order or "").strip().lower()
    if _order not in ("ratio", ""):
        raise ValueError(
            f"build_shared_diffusion_array supports order in {{'ratio',''}} "
            f"(Stage 1); order={order!r} (a common-centroid interleave) is Stage 2 "
            "-- use build_common_centroid_array"
        )
    _oxide = str(oxide or "thin").strip().lower()
    if _oxide not in ("thin", "thick"):
        raise ValueError("oxide must be 'thin' or 'thick'")

    fold_rows = int(fold_rows)
    if fold_rows < 1:
        raise ValueError(f"fold_rows must be >= 1; got {fold_rows}")

    mem = [_member_fields(m) for m in members]
    if not mem:
        raise ValueError("build_shared_diffusion_array needs >= 1 member")

    # The fold divisibility contract (mirrors _build_serpentine_geom): the total
    # finger count must split evenly over the bands so the array is rectangular and
    # the channel width is conserved EXACTLY.
    if fold_rows > 1 and sum(m[4] for m in mem) % fold_rows != 0:
        raise ValueError(
            f"sum(nf)={sum(m[4] for m in mem)} must be divisible by fold_rows="
            f"{fold_rows} for a rectangular conserved fold"
        )

    kinds = {m[1] for m in mem}
    if kinds not in ({"pmos"}, {"nmos"}):
        raise ValueError(
            f"a shared-diffusion array is one polarity; got kinds {sorted(kinds)}"
        )
    kind = mem[0][1]
    if kind not in ("nmos", "pmos"):
        raise ValueError("member kind must be 'nmos' or 'pmos'")

    r = _rules_from_tech(tech)
    keys = _resolve_layer_keys(tech, kind)

    # uniform diffusion (one continuous OD is only physical for equal W + L).
    w_nms = {max(int(round(m[2] * 1000.0)), r.channel_min_width) for m in mem}
    l_nms = {max(int(round(m[3] * 1000.0)), r.gate_min_length) for m in mem}
    if len(w_nms) != 1 or len(l_nms) != 1:
        raise ValueError(
            "shared-diffusion members must share one W and one L (one continuous "
            f"OD); got W_nm={sorted(w_nms)} L_nm={sorted(l_nms)}"
        )
    w_nm = next(iter(w_nms))
    l_nm = next(iter(l_nms))

    gate_nets = {m[5] for m in mem}
    source_nets = {m[7] for m in mem}
    if len(gate_nets) != 1 or len(source_nets) != 1:
        raise ValueError(
            "shared-diffusion members must share one gate net and one source net "
            f"(a mirror / CS bank); got gates={sorted(gate_nets)} "
            f"sources={sorted(source_nets)}"
        )
    gate_net = next(iter(gate_nets))
    source_net = next(iter(source_nets))
    body_nets = {m[8] for m in mem}
    if kind == "pmos" and ("" in body_nets or len(body_nets) != 1):
        raise ValueError(
            "PMOS members in one continuous shared N-well must declare one body "
            f"net; got body nets={sorted(body_nets)}"
        )
    body_net = next(iter(body_nets)) if len(body_nets) == 1 else ""
    if body_tap_owner_sha256 and (
        kind != "pmos" or body_net != source_net
    ):
        raise ValueError(
            "external body-tap ownership requires a source-tied PMOS N-well"
        )

    for m in mem:
        if m[4] < 1:
            raise ValueError(f"member {m[0]!r} nf must be >= 1")
    # LVS-clean abut condition: every member EXCEPT the last must end on a shared
    # source column (nf even) so it abuts the next member on that source column. An
    # odd interior member ends on a drain -> would short into the next source.
    for m in mem[:-1]:
        if m[4] % 2 != 0:
            raise ValueError(
                f"interior member {m[0]!r} has odd nf={m[4]}: it ends on a DRAIN "
                "column that would abut the next member's source (cross-net short). "
                "A separating dummy finger (@dummy_flank, Stage 2) is required; this "
                "layout is unsupported in Stage 1"
            )

    # --- constants (resolved EXACTLY as _build_device_geom) -----------------
    finger_space = r.finger_space
    endcap = r.poly_endcap
    csize = r.contact_size
    sd_clear = r.contact_to_gate
    comp_enc = r.comp_contact_enc
    m1_enc = r.metal1_contact_enc
    m1_cut_enc = max(m1_enc, r.metal1_line_end)
    m1_pad = max(csize + 2 * m1_cut_enc, 380)
    m1_w = max(r.metal1_min_width, csize + 2 * m1_enc)
    bus_band = max(r.poly_min_width, 180)
    top_endcap = max(r.field_poly_to_comp + bus_band, endcap)
    # The S/D risers overlap the full contact columns at this internal seam.
    # An odd number of manufacturing-grid steps in W must not create a native
    # off-grid vertex that the route bridge later edits after row binding.
    from .route.emit_gds import _nearest_on_grid, pdk_manufacturing_grid_nm
    cy = _nearest_on_grid(w_nm // 2, pdk_manufacturing_grid_nm(tech))

    total_nf = sum(m[4] for m in mem)
    # finger k -> (member index, local index within member).
    owner: List[int] = []
    local: List[int] = []
    for mi, m in enumerate(mem):
        for j in range(m[4]):
            owner.append(mi)
            local.append(j)

    # --- flank dummies (fully-off rail end-caps) ----------------------------
    # A ``@dummy_flank`` fully-off rail dummy folds into the strip as ``per_end``
    # gate-off fingers at EACH end. It shares the strip's uniform diffusion and its
    # gate ties to SOURCE (a poly contact down to the source rail, NOT the shared
    # gate bus above), so it extracts as the declared M=2*per_end dummy on the
    # source net -- a matched edge condition with no exposed signal net.
    dmy = _validate_flank_dummies(
        flank_dummies, kind, w_nm, l_nm, source_net, body_net, r
    )
    n_each = sum(d.per_end for d in dmy)  # dummy fingers added at EACH end

    if fold_rows > 1:
        if n_each:
            raise ValueError(
                "flank_dummies with fold_rows>1 is unsupported (the folded-row "
                "end-cap tie is Stage 2); request either a 1-row strip with "
                "end-caps or a multi-row fold without them")
        return _build_folded_shared_diffusion(
            tech=tech, mem=mem, kind=kind, keys=keys, r=r,
            w_nm=w_nm, l_nm=l_nm, total_nf=total_nf, owner=owner, local=local,
            gate_net=gate_net, source_net=source_net, body_net=body_net,
            fold_rows=fold_rows,
            order=_order, oxide=_oxide,
            body_tap_owner_sha256=body_tap_owner_sha256,
        )

    from .native_owned_artwork import NativeOwnedRectangles, attach_native_owned_artwork
    all_owners = tuple(m[0] for m in mem) + tuple(dummy.name for dummy in dmy)
    rects = NativeOwnedRectangles(all_owners)
    fingers: List[ArrayFinger] = []
    dummy_owners = tuple(dummy.name for dummy in dmy for _ in range(dummy.per_end))
    finger_owners = dummy_owners + tuple(mem[index][0] for index in owner) + dummy_owners

    # Finger index space runs [n_each DUMMY | total_nf SIGNAL | n_each DUMMY]. When
    # n_each==0 this is byte-identical to the legacy signal-only strip.
    n_all = n_each + total_nf + n_each
    finger_x_all: List[int] = [k * (l_nm + finger_space) for k in range(n_all)]
    sig_lo = n_each
    sig_hi = n_each + total_nf  # exclusive

    def _is_dummy(k: int) -> bool:
        return k < sig_lo or k >= sig_hi

    # A source-tied flank gate is a DIFFERENT net from the members' shared
    # signal gate.  The left signal-gate contact lane extends the bus slightly
    # past the first signal finger, so merely limiting the bus to the signal
    # fingers in X is insufficient: it can still overlap the left dummy's gate
    # end-cap.  Terminate dummy gates at their required POLY end-cap and lift
    # the signal bus by the PDK's exact same-layer POLY spacing.  With no flank
    # dummies these values are byte-identical to the historical construction.
    bus_y0 = w_nm + r.field_poly_to_comp
    bus_y1 = w_nm + top_endcap
    if n_each:
        dummy_gate_top = w_nm + endcap
        poly_space = _space_or(
            tech,
            _poly_name(tech),
            r.poly_min_width,
        )
        bus_y0 = max(bus_y0, dummy_gate_top + poly_space)
        bus_y1 = max(bus_y1, bus_y0 + bus_band)

    # --- gate poly fingers: n_all full-height strips, left-to-right ----------
    # SIGNAL fingers reach UP to the gate bus (top_endcap); DUMMY fingers instead
    # get a downward poly tail to the source rail (added after the rail is built),
    # so their gate ties to source, not the bus.
    for k, x0 in enumerate(finger_x_all):
        rects.select((finger_owners[k],), 'gate_finger')
        gate_top = w_nm + endcap if _is_dummy(k) else bus_y1
        rects.append((keys.poly, (x0, -endcap, x0 + l_nm, gate_top)))
        if _is_dummy(k):
            fingers.append(ArrayFinger(member="", gate_net=source_net, x0=x0, x1=x0 + l_nm))
        else:
            j = k - sig_lo
            fingers.append(
                ArrayFinger(member=mem[owner[j]][0], gate_net=gate_net, x0=x0, x1=x0 + l_nm)
            )
    # SIGNAL finger extents (the gate bus / gate contact span signal fingers ONLY).
    finger_x: List[int] = finger_x_all[sig_lo:sig_hi]
    last_gate_x_max = finger_x[-1] + l_nm

    # --- COMP (active): ONE continuous rectangle over dummies + members -----
    comp_x0 = finger_x_all[0] - sd_clear - csize - comp_enc
    comp_x1 = finger_x_all[-1] + l_nm + sd_clear + csize + comp_enc
    comp_bbox = (comp_x0, 0, comp_x1, w_nm)
    rects.select(all_owners, 'shared_diffusion_implant')
    rects.append((keys.active, comp_bbox))

    # --- gate implant: COMP grown by NP.5a / PP.5a (one rect over all) ------
    implant_overlap = r.implant_gate_overlap
    if keys.gate_implant is not None:
        rects.append(
            (
                keys.gate_implant,
                (
                    comp_x0 - implant_overlap,
                    0 - implant_overlap,
                    comp_x1 + implant_overlap,
                    w_nm + implant_overlap,
                ),
            )
        )

    # --- S/D contact columns + per-column net (THE INTERLEAVE) --------------
    # Columns run over ALL n_all fingers (dummies + signal). n_all == total_nf when
    # there are no flank dummies (byte-identical to the legacy strip).
    sd_centres: List[int] = [finger_x_all[0] - sd_clear - csize // 2]
    for i in range(n_all - 1):
        gap_lo = finger_x_all[i] + l_nm
        gap_hi = finger_x_all[i + 1]
        sd_centres.append((gap_lo + gap_hi) // 2)
    sd_centres.append(finger_x_all[-1] + l_nm + sd_clear + csize // 2)

    # Column i sits to the LEFT of finger i / to the RIGHT of the last finger.
    # Column 0 is a shared source (left edge). Column k+1 (right of finger k) is:
    #   * a shared SOURCE when finger k is a DUMMY (drain==source==the rail) or the
    #     owning member's local index is odd;
    #   * the owning member's DRAIN when the signal finger's local index is even.
    # A dummy finger is thus flanked by source on both sides (fully off, no exposed
    # net); non-last members have even nf so every member boundary still lands on a
    # shared-source column -> no drain ever abuts a foreign net.
    col_role: List[str] = ["s"]
    col_net: List[str] = [source_net]
    col_member: List[str] = [""]
    for k in range(n_all):
        if not _is_dummy(k) and local[k - sig_lo] % 2 == 0:  # signal drain column
            j = k - sig_lo
            col_role.append("d")
            col_net.append(mem[owner[j]][6])
            col_member.append(mem[owner[j]][0])
        else:  # shared source column (dummy end-cap OR signal odd finger)
            col_role.append("s")
            col_net.append(source_net)
            col_member.append("")

    # per-column contact stack (dense CONT: one full-height cut column + M1 strip,
    # the reference's dense-contact artwork -- same construction as build_device's
    # ``contact_rows`` path) + the ArrayColumn record.
    pitch = csize + r.contact_space
    y_lo = comp_enc + csize // 2
    y_hi = w_nm - comp_enc - csize // 2
    ys = [cy] if y_hi < y_lo else list(range(y_lo, y_hi + 1, pitch))
    strip_half = m1_pad // 2
    columns: List[ArrayColumn] = []
    for i, cx in enumerate(sd_centres):
        rects.select(finger_owners[max(0, i-1):min(n_all, i+1)], 'sd_column')
        for ccy in ys:
            rects.append(
                (keys.contact, (cx - csize // 2, ccy - csize // 2,
                                cx + csize // 2, ccy + csize // 2))
            )
        strip_y0 = min(ys) - csize // 2 - m1_cut_enc
        strip_y1 = max(ys) + csize // 2 + m1_cut_enc
        strip = (cx - strip_half, strip_y0, cx + strip_half, strip_y1)
        rects.append((keys.metal1, strip))
        top_cut = (cx - csize // 2, max(ys) - csize // 2, cx + csize // 2, max(ys) + csize // 2)
        columns.append(
            ArrayColumn(
                role=col_role[i],
                net=col_net[i],
                member=col_member[i],
                x_center=cx,
                contact_bbox=top_cut,
                m1_strip_bbox=strip,
            )
        )

    source_cols = [i for i, role in enumerate(col_role) if role == "s"]
    source_columns = [columns[i] for i in source_cols]

    # --- source rail (below COMP) commoning every source column -------------
    rail_gap = _dense_source_rail_gap(r)
    src_rail_cy = -(m1_w // 2 + rail_gap)
    src_rail_bbox = (
        min(column.m1_strip_bbox[0] for column in source_columns),
        src_rail_cy - m1_w // 2,
        max(column.m1_strip_bbox[2] for column in source_columns),
        src_rail_cy + m1_w // 2,
    )
    rects.select(all_owners, 'shared_source_rail')
    rects.append((keys.metal1, src_rail_bbox))
    # Each riser consumes its contact column's PDK-derived M1 landing width.  A
    # minimum-width riser under a wider landing leaves two narrow same-island
    # shoulders between the rail top and the landing bottom; although electrically
    # connected, that union is not a fixed point of the exact emitter's same-net
    # close.  Carrying the already-certified landing sides down into a rail with the
    # same end extents owns the whole transition here, without a post-emit collar.
    # The riser still reaches the channel centre (cy), OVERLAPPING the column strip;
    # an edge-only touch at its bottom would remain extraction-fragile.
    for column_index, column in zip(source_cols, source_columns):
        rects.select(finger_owners[max(0, column_index-1):min(n_all, column_index+1)], 'source_riser')
        rects.append(
            (
                keys.metal1,
                (
                    column.m1_strip_bbox[0],
                    src_rail_cy,
                    column.m1_strip_bbox[2],
                    cy,
                ),
            )
        )

    # --- flank-dummy gate tie: poly tail DOWN to the source rail -------------
    # Each dummy finger's gate ties to SOURCE (not the shared gate bus). The dummy
    # gate poly extends below COMP to the source rail y and a poly contact lands it
    # on the rail M1 (which spans the dummy's flanking source columns). The dummy
    # thus extracts as an M=2*per_end device with gate==source==drain==source_net.
    if n_each:
        tail_half = max(l_nm // 2, csize // 2 + r.poly_contact_enc)
        tie_cy = src_rail_cy  # land the poly contact on the rail M1 centre
        for k in range(n_all):
            if not _is_dummy(k):
                continue
            rects.select((finger_owners[k],), 'dummy_gate_tie')
            gx = finger_x_all[k] + l_nm // 2
            # poly tail from the finger bottom (-endcap) DOWN past the contact
            rects.append((keys.poly, (gx - tail_half, tie_cy - csize // 2 - r.poly_contact_enc,
                                      gx + tail_half, -endcap)))
            # poly->M1 contact on the source rail (rail M1 already covers gx)
            rects.append((keys.contact, (gx - csize // 2, tie_cy - csize // 2,
                                         gx + csize // 2, tie_cy + csize // 2)))

    # --- gate bus + one gate contact (all members share the gate) -----------
    gate_cx = finger_x[0] - finger_space // 2
    bus_x0 = min(
        finger_x[0],
        gate_cx - max(l_nm // 2, csize // 2 + r.poly_contact_enc),
    )
    gate_bus_bbox = (bus_x0, bus_y0, last_gate_x_max, bus_y1)
    rects.select(tuple(m[0] for m in mem), 'shared_signal_gate')
    rects.append((keys.poly, gate_bus_bbox))

    gate_cy = bus_y1 + r.poly_contact_enc + csize // 2
    gate_pad_top = gate_cy + csize // 2 + r.poly_contact_enc
    gpad_half = max(m1_pad // 2, csize // 2 + r.poly_contact_enc)
    rects.append(
        (keys.poly, (gate_cx - gpad_half, bus_y0, gate_cx + gpad_half, gate_pad_top))
    )
    gate_contact_bbox = (
        gate_cx - csize // 2,
        gate_cy - csize // 2,
        gate_cx + csize // 2,
        gate_cy + csize // 2,
    )
    rects.append((keys.contact, gate_contact_bbox))
    # the gate M1 strap is an isolated island -> meet the M1 min AREA on its own.
    g_area_side = _m1_min_area_side_nm(r.metal1_min_area)
    gpad_m1_half = max(m1_pad, g_area_side) // 2
    rects.append(
        (keys.metal1, (gate_cx - gpad_m1_half, gate_cy - gpad_m1_half,
                       gate_cx + gpad_m1_half, gate_cy + gpad_m1_half))
    )

    # --- per-member drain straps (above the gate) ---------------------------
    # Each member's drain columns are commoned by ONE M1 strap on that member's
    # drain net. Members occupy disjoint x-ranges, so all straps share one y band
    # without touching; a same-net riser lifts each drain column strip to the strap.
    drn_rail_cy = gate_pad_top + r.field_poly_to_comp + m1_w // 2 + 50
    drain_straps: Dict[str, Bbox] = {}
    for mi, m in enumerate(mem):
        rects.select((m[0],), 'member_drain_routing')
        dxs = [
            columns[i].x_center
            for i in range(len(columns))
            if col_role[i] == "d" and col_member[i] == m[0]
        ]
        if not dxs:
            continue
        strap = (
            min(dxs) - m1_w // 2,
            drn_rail_cy - m1_w // 2,
            max(dxs) + m1_w // 2,
            drn_rail_cy + m1_w // 2,
        )
        rects.append((keys.metal1, strap))
        # riser from the channel centre (cy, inside the column M1 strip) up to the
        # strap -- overlapping both ends so the drain net is area-contiguous.
        for cx in dxs:
            rects.append(
                (keys.metal1, (cx - m1_w // 2, cy, cx + m1_w // 2, drn_rail_cy))
            )
        drain_straps[m[0]] = strap

    # --- PMOS Nwell + one declared-body tap for the continuous well ---------
    nwell_bbox: Optional[Bbox] = None
    body_tap_bbox: Optional[Bbox] = None
    well_tap_isolated = False
    rects.select(all_owners, 'shared_well_body_oxide')
    if kind == "pmos" and keys.nwell is not None:
        from .device_well_geometry import shared_nwell_bbox
        base_nwell_bbox = shared_nwell_bbox((comp_bbox,), r)
        if body_tap_owner_sha256:
            nwell_bbox = base_nwell_bbox
        else:
            tap_rects, nwell_bbox, body_tap_bbox, well_tap_isolated = (
                _materialize_shared_pmos_well_tap(
                    keys=keys,
                    r=r,
                    comp_bbox=comp_bbox,
                    source_rail_bbox=src_rail_bbox,
                    source_net=source_net,
                    body_net=body_net,
                    base_nwell_bbox=base_nwell_bbox,
                    forbidden_poly_bboxes=tuple(
                        box for key, box in rects if key == keys.poly
                    ),
                )
            )
            rects.extend(tap_rects)
        rects.append((keys.nwell, nwell_bbox))

    # --- THICK-oxide OD2 marker over the channel (thin: none) ---------------
    if _oxide == "thick":
        od2_key = _od2_key(tech)
        if od2_key is None:
            if not _thick_oxide_is_default(tech):
                raise ValueError(
                    f"oxide='thick' requested but {getattr(tech, 'pdk_id', '?')!r} "
                    "declares no OD2 (thick-gate-oxide) layer -- the thick-ox intent "
                    "cannot be materialised on this PDK"
                )
        else:
            rects.append((od2_key, _od2_box_over_comp(comp_bbox)))

    polys = [_rect(key, box) for (key, box) in rects]
    result = SharedDiffusionArray(
        polygons=polys,
        kind=kind,
        order=_order,
        gate_net=gate_net,
        source_net=source_net,
        gate_length_nm=l_nm,
        channel_width_nm=w_nm,
        comp_bbox=comp_bbox,
        columns=columns,
        fingers=fingers,
        source_rail_bbox=src_rail_bbox,
        gate_bus_bbox=gate_bus_bbox,
        gate_contact_bbox=gate_contact_bbox,
        drain_straps=drain_straps,
        nwell_bbox=nwell_bbox,
        body_net=body_net,
        body_tap_bbox=body_tap_bbox,
        well_tap_isolated=well_tap_isolated,
        body_tap_owner_sha256=body_tap_owner_sha256,
    )
    return attach_native_owned_artwork(result, rects.binding(polys))


__all__ = [
    "build_device",
    "DeviceGeom",
    "Terminal",
    "DeviceCellRules",
    "build_shared_diffusion_array",
    "SharedDiffusionArray",
    "ArrayColumn",
    "ArrayFinger",
]
