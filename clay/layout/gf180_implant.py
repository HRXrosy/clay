"""R140 (BL-140B) — gf180mcu Nplus / Pplus implant generation.

The real GlobalFoundries ``gf180mcu.drc`` deck (R139 signoff) flagged 5
``DF.12_3.3V`` violations on Clay-emitted devices:

    # Rule DF.12_3.3V: COMP not covered by Nplus or Pplus is forbidden
    #                  (except those COMP under marking).
    df12_l1 = comp.not_interacting(schottky_diode)
                  .not_inside(nplus.or(pplus))
                  .not_interacting(v5_xtor).not_interacting(dualgate)
    df12_l1.output("DF.12_3.3V", ...)
    -- AnalogGym/.../rules/klayout/drc/gf180mcu.drc, line 1264-1267

In gf180mcu there is ONE diffusion/active layer, COMP (22/0). A device is made
n-channel or p-channel by drawing an implant **over** the COMP:

    Nplus  32/0   over NMOS COMP      (n+ implant)
    Pplus  31/0   over PMOS COMP      (p+ implant)

(layer/datatype verified in ``gf180_layermap.GF180_VERIFIED_SOURCES`` — Nplus
32/0 at gf180mcu.lyp line 307, Pplus 31/0 at line 324.)

``DF.12`` itself only requires COMP to be *inside* (covered by) the implant. The
amount the implant must extend **beyond** the COMP edge is governed by the
implant "Extension beyond COMP" rule:

    # Rule NP.5b: Extension beyond COMP for the COMP (1) inside LVPWELL
    #             (2) outside Nwell and DNWELL. is 0.16um
    np5b_l1 = nplus...enclosing(comp.edges, 0.16.um, euclidian)...
    -- gf180mcu.drc, line 1694-1696   (planar NMOS: COMP outside Nwell/DNWELL)

    # Rule PP.5b: Extension beyond COMP for COMP (1) Inside NWELL
    #             (2) outside LVPWELL but inside DNWELL. is 0.16um
    pp5b_l1 = pplus...enclosing(comp.edges, 0.16.um, euclidian)...
    -- gf180mcu.drc, line 1913-1915   (PMOS: COMP inside Nwell)

So we grow the COMP bounding box outward by the NP.5b / PP.5b enclosure
(0.16um = 160 nm) on every side and emit that rectangle on the implant layer.
This simultaneously (a) clears DF.12 (COMP fully inside the implant) and
(b) satisfies the extension-beyond-COMP rule NP.5b / PP.5b with no margin
shortfall. No rule value is invented: every number is the ``X.um`` literal in
the cited deck line, surfaced as a constant below.

This module is pure geometry construction. It imports committed modules
(``backend``, ``gf180_layermap``) read-only and emits ``LayerPolygon`` lists.
All coordinates are GDS database units (dbu = 1 nm for gf180mcu).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .backend import LayerPolygon, Polygon
from .gf180_layermap import GF180_VERIFIED_SOURCES, Gf180RealLayerMap

# --- implant layers (verified in gf180_layermap.GF180_VERIFIED_SOURCES) ------
#
# Nplus 32/0  (gf180mcu.lyp line 307)  — n+ implant, drawn over NMOS COMP.
# Pplus 31/0  (gf180mcu.lyp line 324)  — p+ implant, drawn over PMOS COMP.
GF180_NPLUS_LAYER = 32
GF180_NPLUS_DATATYPE = 0  # rule reference: Nplus 32/0
GF180_PPLUS_LAYER = 31
GF180_PPLUS_DATATYPE = 0  # rule reference: Pplus 31/0


# --- implant "Extension beyond COMP" enclosure (nm) --------------------------
#
# gf180mcu.drc — NP.5b (line 1694): planar NMOS Nplus extension beyond COMP.
GF180_NPLUS_EXT_BEYOND_COMP_NM = 160  # rule NP.5b  : 0.16um
# gf180mcu.drc — PP.5b (line 1913): PMOS (inside Nwell) Pplus extension beyond COMP.
GF180_PPLUS_EXT_BEYOND_COMP_NM = 160  # rule PP.5b  : 0.16um

# The deck rule-ids these constants come from (so callers/tests can cite them).
NPLUS_ENCLOSURE_RULE_ID = "NP.5b"
PPLUS_ENCLOSURE_RULE_ID = "PP.5b"
# The DF rule whose violations this module fixes.
DF12_RULE_ID = "DF.12_3.3V"


# Normalised device-type tokens.
_NMOS_TOKENS = {"NMOS", "N", "NCH", "NFET", "NDIFF", "NPLUS"}
_PMOS_TOKENS = {"PMOS", "P", "PCH", "PFET", "PDIFF", "PPLUS"}


def _normalize_device_type(device_type: str) -> str:
    """Return ``"NMOS"`` or ``"PMOS"`` for a free-form device-type token.

    Raises:
        ValueError: if the token is neither an n-channel nor a p-channel name.
    """
    key = str(device_type).strip().upper()
    if key in _NMOS_TOKENS:
        return "NMOS"
    if key in _PMOS_TOKENS:
        return "PMOS"
    raise ValueError(
        f"unknown gf180 device_type {device_type!r}; expected one of "
        f"{sorted(_NMOS_TOKENS | _PMOS_TOKENS)}"
    )


def implant_layer_for(device_type: str) -> Tuple[int, int]:
    """Return ``(layer, datatype)`` of the implant for ``device_type``.

    ``"NMOS"`` -> Nplus (32, 0);  ``"PMOS"`` -> Pplus (31, 0).
    """
    kind = _normalize_device_type(device_type)
    if kind == "NMOS":
        return (GF180_NPLUS_LAYER, GF180_NPLUS_DATATYPE)
    return (GF180_PPLUS_LAYER, GF180_PPLUS_DATATYPE)


def implant_enclosure_nm(device_type: str) -> int:
    """Return the implant extension-beyond-COMP (nm) for ``device_type``.

    NMOS -> NP.5b (160 nm); PMOS -> PP.5b (160 nm).
    """
    kind = _normalize_device_type(device_type)
    return (
        GF180_NPLUS_EXT_BEYOND_COMP_NM
        if kind == "NMOS"
        else GF180_PPLUS_EXT_BEYOND_COMP_NM
    )


def implant_enclosure_rule_id(device_type: str) -> str:
    """Return the deck rule-id governing the implant enclosure for the type."""
    kind = _normalize_device_type(device_type)
    return NPLUS_ENCLOSURE_RULE_ID if kind == "NMOS" else PPLUS_ENCLOSURE_RULE_ID


# Accepted geometry input forms:
#   - a single LayerPolygon
#   - a single Polygon
#   - an (x_min, y_min, x_max, y_max) bbox tuple
#   - an iterable of any of the above (mixed allowed)
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
            f"cannot derive a COMP bbox from {type(obj).__name__}; expected "
            f"LayerPolygon, Polygon, an (x0,y0,x1,y1) tuple, or a sequence of those"
        )
    if not verts:
        raise ValueError("empty polygon has no bounding box")
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def _iter_comp_items(device_geom_or_polys: GeomInput) -> List:
    """Flatten the input into a list of individual geometry items.

    A single LayerPolygon / Polygon / bbox is treated as one item; a sequence
    is iterated. (A bbox tuple is a 4-element sequence, so it is special-cased
    to NOT be iterated element-by-element.)
    """
    if isinstance(device_geom_or_polys, (LayerPolygon, Polygon)):
        return [device_geom_or_polys]
    if _is_bbox(device_geom_or_polys):
        return [device_geom_or_polys]
    if isinstance(device_geom_or_polys, Iterable):
        items = list(device_geom_or_polys)
        if not items:
            raise ValueError("no COMP geometry supplied")
        return items
    raise TypeError(
        f"unsupported COMP geometry input {type(device_geom_or_polys).__name__}"
    )


def comp_bbox_of(device_geom_or_polys: GeomInput) -> _BBox:
    """Return the union bounding box (x0, y0, x1, y1) of the COMP geometry.

    Accepts a single LayerPolygon / Polygon / bbox tuple, or an iterable of
    those (e.g. ``Gf180DeviceGeometry.comp_polygons()``).
    """
    items = _iter_comp_items(device_geom_or_polys)
    boxes = [_bbox_of(it) for it in items]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def add_implant(
    device_geom_or_polys: GeomInput,
    device_type: str,
    *,
    enclosure_nm: Optional[int] = None,
    layermap: Optional[Gf180RealLayerMap] = None,
) -> List[LayerPolygon]:
    """Emit the implant rectangle(s) covering a gf180 device's COMP geometry.

    The implant is the COMP union bounding box grown outward by the
    extension-beyond-COMP enclosure (NP.5b for NMOS / PP.5b for PMOS, both
    0.16um = 160 nm in the 3.3V deck). Growing the *bounding box* guarantees
    every COMP polygon sits fully inside the implant with at least the rule
    enclosure on every edge, which clears ``DF.12`` (COMP not covered by
    Nplus/Pplus) and satisfies NP.5b / PP.5b simultaneously.

    Args:
        device_geom_or_polys: the device's COMP geometry. One of:
            * a ``LayerPolygon`` (its polygon's bbox is used),
            * a ``Polygon``,
            * an ``(x_min, y_min, x_max, y_max)`` bbox tuple,
            * an iterable of any of the above (e.g.
              ``Gf180DeviceGeometry.comp_polygons()`` or ``.comp_bbox``).
        device_type: ``"NMOS"``/``"PMOS"`` (or aliases ``n``/``p``/``nch``/...).
        enclosure_nm: override the implant extension beyond COMP, in nm. Must be
            >= the rule value (NP.5b/PP.5b = 160 nm) so the result never
            under-encloses; defaults to the rule value. Smaller values are
            rejected.
        layermap: optional ``Gf180RealLayerMap`` used to resolve the implant
            (layer, datatype) from the real PDK .lyp at runtime. When omitted
            the verified hardcoded constants (Nplus 32/0, Pplus 31/0) are used.

    Returns:
        A single-element list ``[LayerPolygon]`` carrying the implant rectangle
        on the correct implant layer.

    Raises:
        ValueError: if ``device_type`` is unknown, the COMP geometry is empty,
            or ``enclosure_nm`` is below the rule value.
    """
    kind = _normalize_device_type(device_type)
    rule_enc = implant_enclosure_nm(kind)
    enc = rule_enc if enclosure_nm is None else int(enclosure_nm)
    if enc < rule_enc:
        rule_id = implant_enclosure_rule_id(kind)
        raise ValueError(
            f"enclosure_nm={enclosure_nm} is below the {rule_id} extension-"
            f"beyond-COMP minimum ({rule_enc} nm); would re-introduce a "
            f"{rule_id} / DF.12 violation"
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

    x0, y0, x1, y1 = comp_bbox_of(device_geom_or_polys)
    implant = Polygon.rectangle(x0 - enc, y0 - enc, x1 + enc, y1 + enc)
    return [LayerPolygon(layer=layer, datatype=datatype, polygon=implant)]


# Sanity: the module constants must agree with the verified PDK layer table.
def _verify_constants_against_lyp() -> None:  # pragma: no cover - import guard
    n = GF180_VERIFIED_SOURCES.get("Nplus")
    p = GF180_VERIFIED_SOURCES.get("Pplus")
    if n is not None and (GF180_NPLUS_LAYER, GF180_NPLUS_DATATYPE) != (n[0], n[1]):
        raise AssertionError("Nplus constant drifted from GF180_VERIFIED_SOURCES")
    if p is not None and (GF180_PPLUS_LAYER, GF180_PPLUS_DATATYPE) != (p[0], p[1]):
        raise AssertionError("Pplus constant drifted from GF180_VERIFIED_SOURCES")


_verify_constants_against_lyp()


__all__ = [
    "GF180_NPLUS_LAYER",
    "GF180_NPLUS_DATATYPE",
    "GF180_PPLUS_LAYER",
    "GF180_PPLUS_DATATYPE",
    "GF180_NPLUS_EXT_BEYOND_COMP_NM",
    "GF180_PPLUS_EXT_BEYOND_COMP_NM",
    "NPLUS_ENCLOSURE_RULE_ID",
    "PPLUS_ENCLOSURE_RULE_ID",
    "DF12_RULE_ID",
    "implant_layer_for",
    "implant_enclosure_nm",
    "implant_enclosure_rule_id",
    "comp_bbox_of",
    "add_implant",
]
