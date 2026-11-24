"""R142 (BL-142B) — gf180mcu Nplus / Pplus implant with gate-poly enclosure.

R140 (``gf180_implant.add_implant``) emits the implant as the COMP bounding box
grown by the NP.5b / PP.5b "Extension beyond COMP" rule (0.16um = 160 nm). That
clears ``DF.12`` (COMP not covered by implant) and NP.5b / PP.5b — but it does
NOT satisfy the *gate-overlap* rule the real gf180mcu deck checks separately:

    # Rule NP.5a: Overlap of N-channel gate. is 0.23um
    np5a_l1 = nplus.enclosing(ngate, 0.23.um, euclidian).polygons(0.001)
    np5a_l2 = ngate.not_outside(nplus).not(nplus)
    np5a_l  = np5a_l1.or(np5a_l2)
    np5a_l.output("NP.5a", "NP.5a : Overlap of N-channel gate. : 0.23um")
    -- AnalogGym/.../rules/klayout/drc/gf180mcu.drc, lines 1684-1689

    # Rule PP.5a: Overlap of P-channel gate. is 0.23um
    pp5a_l1 = pplus.enclosing(pgate, 0.23.um, euclidian).polygons(0.001)
    pp5a_l2 = pgate.not_outside(pplus).not(pplus)
    pp5a_l  = pp5a_l1.or(pp5a_l2)
    pp5a_l.output("PP.5a", "PP.5a : Overlap of P-channel gate. : 0.23um")
    -- gf180mcu.drc, lines 1903-1908

where the gate is the poly-over-COMP intersection (deck lines 427-429)::

    tgate = poly2 & comp        # poly that crosses diffusion = the gate
    ngate = nplus & tgate       # n-channel gate
    pgate = pplus & tgate       # p-channel gate

NP.5a / PP.5a demand the implant *enclose the gate* by 0.23um = 230 nm on every
edge (``enclosing(...,0.23.um,euclidian)``), and that NO part of the gate sits
outside the implant (the ``not_outside(...).not(...)`` term). The gate region's
y-edges coincide with the COMP y-edges (because ``tgate`` is clipped to COMP),
so the R140 implant — only 160 nm beyond COMP — under-encloses the gate by
230 - 160 = 70 nm on the channel-width edges and trips NP.5a / PP.5a ~20x on the
real deck.

Fix: compute the gate region (poly2 INTERSECT COMP) and grow its union bounding
box outward by the NP.5a / PP.5a enclosure (230 nm). Union that with the COMP
coverage rectangle (COMP bbox grown by NP.5b = 160 nm, the R140 geometry) so the
single emitted implant simultaneously satisfies:

    * DF.12     COMP fully inside implant            (COMP + 160 nm)
    * NP.5b/5b  implant extension beyond COMP        (>= 160 nm everywhere)
    * NP.5a/5a  implant encloses gate by >= 230 nm   (gate + 230 nm)

Because 230 nm (gate enclosure) >= 160 nm (COMP enclosure) and the gate is a
sub-region of COMP in x and exactly COMP in y, the gate-grown rectangle already
covers the COMP-grown rectangle on the channel-width (y) edges; in x the COMP
rectangle can stick out past the outermost gate (source/drain overhang), so we
take the per-edge union of the two rectangles. The result is one rectangle that
encloses the gate by >= 230 nm AND COMP by >= 160 nm on every edge — no number
is invented; both enclosures are the ``X.um`` literals from the cited deck rules.

This module is pure geometry construction. It imports committed modules
(``backend``, ``gf180_layermap``, ``gf180_implant``) read-only and emits
``LayerPolygon`` lists. All coordinates are GDS database units (dbu = 1 nm for
gf180mcu).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .backend import LayerPolygon, Polygon
from .gf180_implant import (
    GF180_NPLUS_DATATYPE,
    GF180_NPLUS_EXT_BEYOND_COMP_NM,
    GF180_NPLUS_LAYER,
    GF180_PPLUS_DATATYPE,
    GF180_PPLUS_EXT_BEYOND_COMP_NM,
    GF180_PPLUS_LAYER,
    _normalize_device_type,
    comp_bbox_of,
    implant_layer_for,
)
from .gf180_layermap import GF180_VERIFIED_SOURCES, Gf180RealLayerMap

# --- POLY2 layer (verified in gf180_layermap.GF180_VERIFIED_SOURCES) ---------
#
# Poly2 30/0 (gf180mcu.lyp line 290) — the gate poly that forms ``tgate`` when
# it crosses COMP (deck line 427: ``tgate = poly2 & comp``).
GF180_POLY2_LAYER = 30
GF180_POLY2_DATATYPE = 0  # gf180mcu.lyp line 290


# --- implant gate-overlap enclosure (nm) -------------------------------------
#
# gf180mcu.drc — NP.5a (line 1684): Nplus overlap of N-channel gate = 0.23um.
# gf180mcu.drc — PP.5a (line 1903): Pplus overlap of P-channel gate = 0.23um.
# Both are ``enclosing(<gate>, 0.23.um, euclidian)`` => 230 nm on every edge.
GF180_NPLUS_GATE_OVERLAP_NM = 230  # rule NP.5a : 0.23um
GF180_PPLUS_GATE_OVERLAP_NM = 230  # rule PP.5a : 0.23um

# The deck rule-ids these constants come from (so callers/tests can cite them).
NPLUS_GATE_OVERLAP_RULE_ID = "NP.5a"
PPLUS_GATE_OVERLAP_RULE_ID = "PP.5a"


# Reuse the COMP coverage enclosure (NP.5b / PP.5b = 160 nm) from R140 so the
# emitted implant still satisfies DF.12 + the extension-beyond-COMP rule.
_COMP_ENC_FOR = {
    "NMOS": GF180_NPLUS_EXT_BEYOND_COMP_NM,  # NP.5b = 160 nm
    "PMOS": GF180_PPLUS_EXT_BEYOND_COMP_NM,  # PP.5b = 160 nm
}
_GATE_ENC_FOR = {
    "NMOS": GF180_NPLUS_GATE_OVERLAP_NM,  # NP.5a = 230 nm
    "PMOS": GF180_PPLUS_GATE_OVERLAP_NM,  # PP.5a = 230 nm
}


def gate_overlap_nm(device_type: str) -> int:
    """Return the implant gate-overlap enclosure (nm) for ``device_type``.

    NMOS -> NP.5a (230 nm); PMOS -> PP.5a (230 nm).
    """
    return _GATE_ENC_FOR[_normalize_device_type(device_type)]


def gate_overlap_rule_id(device_type: str) -> str:
    """Return the deck rule-id governing the gate overlap for the type."""
    kind = _normalize_device_type(device_type)
    return NPLUS_GATE_OVERLAP_RULE_ID if kind == "NMOS" else PPLUS_GATE_OVERLAP_RULE_ID


# --- geometry input handling -------------------------------------------------

_BBox = Tuple[int, int, int, int]
GeomInput = Union[
    LayerPolygon,
    Polygon,
    _BBox,
    Sequence[Union[LayerPolygon, Polygon, _BBox]],
]


def _is_bbox(obj) -> bool:
    return (
        isinstance(obj, (tuple, list))
        and len(obj) == 4
        and all(isinstance(v, (int, float)) for v in obj)
    )


def _bbox_of(obj) -> _BBox:
    """Return the (x_min, y_min, x_max, y_max) bbox of one geometry item."""
    if _is_bbox(obj):
        x0, y0, x1, y1 = obj
        return (int(min(x0, x1)), int(min(y0, y1)),
                int(max(x0, x1)), int(max(y0, y1)))
    if isinstance(obj, LayerPolygon):
        verts = obj.polygon.vertices
    elif isinstance(obj, Polygon):
        verts = obj.vertices
    else:
        raise TypeError(
            f"cannot derive a bbox from {type(obj).__name__}; expected "
            f"LayerPolygon, Polygon, an (x0,y0,x1,y1) tuple, or a sequence of those"
        )
    if not verts:
        raise ValueError("empty polygon has no bounding box")
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def _iter_items(geom: GeomInput) -> List:
    if isinstance(geom, (LayerPolygon, Polygon)):
        return [geom]
    if _is_bbox(geom):
        return [geom]
    if isinstance(geom, Iterable):
        items = list(geom)
        return items
    raise TypeError(f"unsupported geometry input {type(geom).__name__}")


def _rect_intersection(
    a: _BBox, b: _BBox
) -> Optional[_BBox]:
    """Return the rectangular intersection of two bboxes, or None if disjoint."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def gate_region_bboxes(
    comp_geom: GeomInput, poly_geom: GeomInput
) -> List[_BBox]:
    """Return the gate-region rectangles (poly2 INTERSECT COMP) as bboxes.

    Mirrors the deck definition ``tgate = poly2 & comp`` (gf180mcu.drc line 427):
    each gate region is the rectangular overlap of one poly2 finger with the
    COMP diffusion. The implant must enclose the union of these regions by the
    NP.5a / PP.5a overlap (230 nm).

    Args:
        comp_geom: the device's COMP geometry (LayerPolygon / Polygon / bbox /
            iterable of those, e.g. ``Gf180DeviceGeometry.comp_polygons()``).
        poly_geom: the device's poly2 geometry (same accepted forms, e.g.
            ``Gf180DeviceGeometry.poly_polygons()``).

    Returns:
        A list of (x0, y0, x1, y1) bboxes, one per poly-finger-over-COMP
        intersection that is non-empty. Empty list if no poly crosses COMP.
    """
    comp_boxes = [_bbox_of(it) for it in _iter_items(comp_geom)]
    poly_boxes = [_bbox_of(it) for it in _iter_items(poly_geom)]
    if not comp_boxes:
        raise ValueError("no COMP geometry supplied")
    if not poly_boxes:
        raise ValueError("no poly geometry supplied")
    gates: List[_BBox] = []
    for p in poly_boxes:
        for c in comp_boxes:
            inter = _rect_intersection(p, c)
            if inter is not None:
                gates.append(inter)
    return gates


def gate_union_bbox(comp_geom: GeomInput, poly_geom: GeomInput) -> _BBox:
    """Return the union bounding box of every gate region (poly2 & COMP).

    Raises:
        ValueError: if no poly finger intersects the COMP (no gate exists).
    """
    gates = gate_region_bboxes(comp_geom, poly_geom)
    if not gates:
        raise ValueError(
            "no gate region: poly2 does not intersect COMP (cannot enclose a "
            "non-existent gate); supply the poly fingers that cross the diffusion"
        )
    x0 = min(g[0] for g in gates)
    y0 = min(g[1] for g in gates)
    x1 = max(g[2] for g in gates)
    y1 = max(g[3] for g in gates)
    return (x0, y0, x1, y1)


def _union_bbox(a: _BBox, b: _BBox) -> _BBox:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def add_implant_with_gate(
    comp_geom: GeomInput,
    poly_geom: GeomInput,
    device_type: str,
    *,
    gate_enclosure_nm: Optional[int] = None,
    comp_enclosure_nm: Optional[int] = None,
    layermap: Optional[Gf180RealLayerMap] = None,
) -> List[LayerPolygon]:
    """Emit the implant covering both COMP (DF.12 / NP.5b) and the gate (NP.5a).

    The implant rectangle is the per-edge union of two grown rectangles:

      * the COMP bounding box grown by the NP.5b / PP.5b extension-beyond-COMP
        enclosure (160 nm) — the R140 geometry that clears DF.12 and NP.5b/5b;
      * the GATE (poly2 INTERSECT COMP) union bounding box grown by the
        NP.5a / PP.5a gate-overlap enclosure (230 nm).

    Because 230 nm >= 160 nm and the gate's y-extent equals COMP's y-extent
    while its x-extent is a sub-span of COMP, the union rectangle encloses the
    gate by >= 230 nm AND COMP by >= 160 nm on every edge, so it satisfies
    NP.5a/PP.5a, NP.5b/PP.5b and DF.12 simultaneously.

    Args:
        comp_geom: the device's COMP geometry (accepted forms as in R140's
            ``add_implant``: LayerPolygon / Polygon / bbox / iterable).
        poly_geom: the device's poly2 geometry (same accepted forms). Only the
            poly fingers that cross COMP contribute to the gate region.
        device_type: ``"NMOS"``/``"PMOS"`` (or aliases ``n``/``p``/``nch``/...).
        gate_enclosure_nm: override the gate overlap, in nm. Must be >= the rule
            value (NP.5a/PP.5a = 230 nm); defaults to the rule value. Smaller
            values are rejected (they would re-introduce NP.5a / PP.5a).
        comp_enclosure_nm: override the COMP extension beyond COMP, in nm. Must
            be >= the rule value (NP.5b/PP.5b = 160 nm); defaults to the rule
            value. Smaller values are rejected.
        layermap: optional ``Gf180RealLayerMap`` used to resolve the implant
            (layer, datatype) from the real PDK .lyp at runtime. When omitted
            the verified hardcoded constants (Nplus 32/0, Pplus 31/0) are used.

    Returns:
        A single-element list ``[LayerPolygon]`` carrying the implant rectangle
        on the correct implant layer.

    Raises:
        ValueError: if ``device_type`` is unknown, the COMP / poly geometry is
            empty, no poly crosses COMP (no gate), or an override is below its
            rule value.
    """
    kind = _normalize_device_type(device_type)

    gate_rule = _GATE_ENC_FOR[kind]
    comp_rule = _COMP_ENC_FOR[kind]
    gate_enc = gate_rule if gate_enclosure_nm is None else int(gate_enclosure_nm)
    comp_enc = comp_rule if comp_enclosure_nm is None else int(comp_enclosure_nm)
    if gate_enc < gate_rule:
        rid = gate_overlap_rule_id(kind)
        raise ValueError(
            f"gate_enclosure_nm={gate_enclosure_nm} is below the {rid} gate-"
            f"overlap minimum ({gate_rule} nm); would re-introduce {rid}"
        )
    if comp_enc < comp_rule:
        raise ValueError(
            f"comp_enclosure_nm={comp_enclosure_nm} is below the NP.5b/PP.5b "
            f"extension-beyond-COMP minimum ({comp_rule} nm); would re-introduce "
            f"NP.5b/PP.5b / DF.12"
        )

    # Resolve the implant (layer, datatype).
    if layermap is not None:
        gf_name = "Nplus" if kind == "NMOS" else "Pplus"
        gl = layermap.gf180_layer(gf_name)
        if gl is None:
            raise ValueError(
                f"layermap does not define {gf_name}; cannot place implant"
            )
        layer, datatype = gl.layer, gl.datatype
    else:
        layer, datatype = implant_layer_for(kind)

    # COMP coverage rectangle (R140 geometry: COMP bbox + 160 nm).
    cx0, cy0, cx1, cy1 = comp_bbox_of(comp_geom)
    comp_rect = (cx0 - comp_enc, cy0 - comp_enc, cx1 + comp_enc, cy1 + comp_enc)

    # Gate enclosure rectangle (gate union bbox + 230 nm).
    gx0, gy0, gx1, gy1 = gate_union_bbox(comp_geom, poly_geom)
    gate_rect = (gx0 - gate_enc, gy0 - gate_enc, gx1 + gate_enc, gy1 + gate_enc)

    x0, y0, x1, y1 = _union_bbox(comp_rect, gate_rect)
    implant = Polygon.rectangle(x0, y0, x1, y1)
    return [LayerPolygon(layer=layer, datatype=datatype, polygon=implant)]


# Sanity: the module constants must agree with the verified PDK layer table.
def _verify_constants_against_lyp() -> None:  # pragma: no cover - import guard
    n = GF180_VERIFIED_SOURCES.get("Nplus")
    p = GF180_VERIFIED_SOURCES.get("Pplus")
    poly = GF180_VERIFIED_SOURCES.get("Poly2")
    if n is not None and (GF180_NPLUS_LAYER, GF180_NPLUS_DATATYPE) != (n[0], n[1]):
        raise AssertionError("Nplus constant drifted from GF180_VERIFIED_SOURCES")
    if p is not None and (GF180_PPLUS_LAYER, GF180_PPLUS_DATATYPE) != (p[0], p[1]):
        raise AssertionError("Pplus constant drifted from GF180_VERIFIED_SOURCES")
    if poly is not None and (GF180_POLY2_LAYER, GF180_POLY2_DATATYPE) != (poly[0], poly[1]):
        raise AssertionError("Poly2 constant drifted from GF180_VERIFIED_SOURCES")


_verify_constants_against_lyp()


__all__ = [
    "GF180_POLY2_LAYER",
    "GF180_POLY2_DATATYPE",
    "GF180_NPLUS_GATE_OVERLAP_NM",
    "GF180_PPLUS_GATE_OVERLAP_NM",
    "NPLUS_GATE_OVERLAP_RULE_ID",
    "PPLUS_GATE_OVERLAP_RULE_ID",
    "gate_overlap_nm",
    "gate_overlap_rule_id",
    "gate_region_bboxes",
    "gate_union_bbox",
    "add_implant_with_gate",
]
