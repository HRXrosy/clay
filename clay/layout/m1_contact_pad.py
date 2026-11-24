"""R141 (BL-141A) — Metal1 landing pads over device contacts (gf180mcu CO.6).

R140's real-geometry gf180mcu layout (``gf180_device_geom.build_gf180_mos_device``)
emits source/drain and gate **Contact** shapes (33/0) but draws no Metal1 over
them. The real gf180mcu DRC deck requires Metal1 to cover every contact:

    # Rule CO.6: Metal1 overlap of contact.
    co6_l1 = metal1.enclosing(contact, 0.005.um, euclidian).polygons(0.001)
             .or(contact.not_inside(metal1).not(metal1))

    (AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/rules/klayout/
     drc/gf180mcu.drc, line 2320)

The ``.or(contact.not_inside(metal1).not(metal1))`` clause flags any contact
that is not covered by Metal1 *at all*, so a bare contact with no M1 over it is
a CO.6 violation. R140's layout produced 21 such violations (one per contact on
a device with no M1 routing). This module closes that gap by emitting one M1
(34/0) landing pad per contact, enclosing the contact on every side.

Enclosure value used
---------------------
The bare CO.6 enclosure minimum is ``0.005.um`` (5 nm). But a real contact
landing pad must also satisfy the narrow-Metal1 end-of-line overlap rule:

    # Rule CO.6a: (i) Metal1 (< 0.34um) end-of-line overlap. is 0.06µm
    co6a_l1 = cop6a_eol.enclosing(contact.edges, 0.06.um, projection)
             (deck line 2328)

and the Metal1 minimum-width rule:

    # Rule M1.1: min. metal1 width is 0.23µm
             (deck line 2394)

So the default pad enclosure is ``CO.6a`` = 60 nm (which dominates the 5 nm
CO.6 minimum and guarantees the M1 overlaps the contact on every side by far
more than CO.6 requires), and each pad is additionally grown, symmetrically, so
its M1 width is never below ``M1.1`` = 230 nm in either dimension. With a
gf180mcu contact of ``CO.1`` = 220 nm a 60 nm enclosure already gives a
220 + 2*60 = 340 nm pad, so the M1.1 floor is only ever exercised for a contact
drawn smaller than the spec (defensive).

Every value below is transcribed verbatim from the gf180mcu DRC deck (cited by
deck line). No value is invented. This module imports committed modules read
only (``backend`` for the IR dataclasses) and emits ``LayerPolygon`` lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .backend import LayerPolygon, Polygon

# --- gf180mcu layer ids (verified: gf180_layermap.GF180_VERIFIED_SOURCES) -----
# Contact 33/0 (gf180mcu.lyp line 375); Metal1 34/0 (gf180mcu.lyp line 392).
GF180_CONTACT_LAYER = 33
GF180_CONTACT_DATATYPE = 0
GF180_METAL1_LAYER = 34
GF180_METAL1_DATATYPE = 0

# --- real gf180mcu DRC values (nm), each cited to gf180mcu.drc ----------------

# Rule CO.6 "Metal1 overlap of contact." metal1.enclosing(contact, 0.005.um)
# (gf180mcu.drc line 2320). The bare minimum M1-over-contact enclosure.
GF180_M1_CONTACT_ENCLOSURE_MIN_NM = 5  # rule CO.6  (0.005 um)

# Rule CO.6a "(i) Metal1 (< 0.34um) end-of-line overlap. : 0.06um"
# enclosing(contact.edges, 0.06.um, projection) (gf180mcu.drc line 2328).
GF180_M1_CONTACT_EOL_ENCLOSURE_NM = 60  # rule CO.6a  (0.06 um)

# Rule M1.1 "min. metal1 width is 0.23um" (gf180mcu.drc line 2394).
GF180_M1_MIN_WIDTH_NM = 230  # rule M1.1  (0.23 um)

# Deck rule id and source line cited by this module's pad geometry.
GF180_M1_CONTACT_ENCLOSURE_RULE_ID = "CO.6"
GF180_M1_CONTACT_ENCLOSURE_DECK_LINE = 2320
GF180_DECK_RELPATH = (
    "AnalogGym/PDK/gf180mcu_pdk/libraries/gf180mcu_fd_pr/latest/"
    "rules/klayout/drc/gf180mcu.drc"
)


def _bbox(lp: LayerPolygon) -> Tuple[int, int, int, int]:
    """Return (x_min, y_min, x_max, y_max) of a LayerPolygon's vertices."""
    xs = [v.x for v in lp.polygon.vertices]
    ys = [v.y for v in lp.polygon.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _is_contact(lp: LayerPolygon) -> bool:
    return (lp.layer, lp.datatype) == (GF180_CONTACT_LAYER, GF180_CONTACT_DATATYPE)


@dataclass(frozen=True)
class M1PadResult:
    """The M1 landing pads emitted for a set of contacts.

    Attributes:
        pads: one M1 (34/0) ``LayerPolygon`` per input contact, in input order.
        enclosure_nm: the M1-overlap-of-contact enclosure actually applied (nm).
        rule_id: the deck rule the enclosure satisfies ("CO.6").
        contact_count: number of contacts a pad was emitted for.
    """

    pads: List[LayerPolygon]
    enclosure_nm: int
    rule_id: str
    contact_count: int


def m1_pad_for_contact(
    contact: LayerPolygon,
    *,
    enclosure_nm: int = GF180_M1_CONTACT_EOL_ENCLOSURE_NM,
    min_m1_width_nm: int = GF180_M1_MIN_WIDTH_NM,
) -> LayerPolygon:
    """Return one Metal1 (34/0) landing pad enclosing a single contact.

    The pad is the contact bounding box grown outward by ``enclosure_nm`` on
    every side, so the M1 overlaps the contact by ``enclosure_nm`` (>= CO.6's
    0.005 um minimum and >= CO.6a's 0.06 um narrow-EOL projection overlap when
    the default is used). If the resulting pad would be narrower than
    ``min_m1_width_nm`` (M1.1) in either dimension it is grown symmetrically up
    to that floor, keeping the contact centred so the per-side enclosure is
    never reduced below ``enclosure_nm``.

    Args:
        contact: a Contact ``LayerPolygon`` (layer 33, datatype 0).
        enclosure_nm: M1-over-contact enclosure to apply on every side; must be
            >= CO.6 (5 nm). Defaults to CO.6a (60 nm).
        min_m1_width_nm: M1.1 minimum metal1 width floor (230 nm).

    Returns:
        A Metal1 (34/0) ``LayerPolygon`` covering the contact.

    Raises:
        ValueError: if ``contact`` is not on the gf180mcu Contact layer (33/0),
            has no area, or ``enclosure_nm`` is below the CO.6 minimum.
    """
    if not _is_contact(contact):
        raise ValueError(
            f"m1_pad_for_contact expects a Contact on layer "
            f"({GF180_CONTACT_LAYER}/{GF180_CONTACT_DATATYPE}); got "
            f"({contact.layer}/{contact.datatype})"
        )
    if enclosure_nm < GF180_M1_CONTACT_ENCLOSURE_MIN_NM:
        raise ValueError(
            f"enclosure_nm={enclosure_nm} is below the CO.6 minimum "
            f"{GF180_M1_CONTACT_ENCLOSURE_MIN_NM} nm (0.005 um)"
        )

    x_min, y_min, x_max, y_max = _bbox(contact)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("contact polygon has zero area; cannot build a pad")

    px_min = x_min - enclosure_nm
    py_min = y_min - enclosure_nm
    px_max = x_max + enclosure_nm
    py_max = y_max + enclosure_nm

    # Grow symmetrically up to the M1.1 minimum width if the pad is too narrow
    # in either axis. Symmetric growth keeps the contact centred, so the
    # per-side enclosure can only increase, never drop below ``enclosure_nm``.
    if (px_max - px_min) < min_m1_width_nm:
        deficit = min_m1_width_nm - (px_max - px_min)
        grow = (deficit + 1) // 2  # round up so the floor is always met
        px_min -= grow
        px_max += grow
    if (py_max - py_min) < min_m1_width_nm:
        deficit = min_m1_width_nm - (py_max - py_min)
        grow = (deficit + 1) // 2
        py_min -= grow
        py_max += grow

    return LayerPolygon(
        layer=GF180_METAL1_LAYER,
        datatype=GF180_METAL1_DATATYPE,
        polygon=Polygon.rectangle(px_min, py_min, px_max, py_max),
    )


def add_m1_pads(
    contacts: Iterable[LayerPolygon],
    *,
    enclosure_nm: int = GF180_M1_CONTACT_EOL_ENCLOSURE_NM,
    min_m1_width_nm: int = GF180_M1_MIN_WIDTH_NM,
) -> List[LayerPolygon]:
    """Emit one Metal1 (34/0) landing pad per Contact, closing gf180mcu CO.6.

    Each input contact (layer 33/0) gets an M1 pad enclosing it by
    ``enclosure_nm`` on every side. Non-contact shapes in ``contacts`` are
    ignored (so the device's full polygon list can be passed directly and only
    its contacts will produce pads).

    Args:
        contacts: an iterable of ``LayerPolygon``; only Contact (33/0) shapes
            produce a pad.
        enclosure_nm: per-side M1-over-contact enclosure (default CO.6a 60 nm;
            must be >= CO.6 5 nm).
        min_m1_width_nm: M1.1 minimum metal1 width floor (230 nm).

    Returns:
        A list of Metal1 (34/0) ``LayerPolygon`` pads, one per contact, in the
        order the contacts were encountered.

    Raises:
        ValueError: if ``enclosure_nm`` is below the CO.6 minimum, or a contact
            shape has zero area.
    """
    pads: List[LayerPolygon] = []
    for lp in contacts:
        if not _is_contact(lp):
            continue
        pads.append(
            m1_pad_for_contact(
                lp, enclosure_nm=enclosure_nm, min_m1_width_nm=min_m1_width_nm
            )
        )
    return pads


def add_m1_pads_result(
    contacts: Iterable[LayerPolygon],
    *,
    enclosure_nm: int = GF180_M1_CONTACT_EOL_ENCLOSURE_NM,
    min_m1_width_nm: int = GF180_M1_MIN_WIDTH_NM,
) -> M1PadResult:
    """Like :func:`add_m1_pads` but returns a structured :class:`M1PadResult`."""
    pads = add_m1_pads(
        contacts, enclosure_nm=enclosure_nm, min_m1_width_nm=min_m1_width_nm
    )
    return M1PadResult(
        pads=pads,
        enclosure_nm=enclosure_nm,
        rule_id=GF180_M1_CONTACT_ENCLOSURE_RULE_ID,
        contact_count=len(pads),
    )


def contacts_with_m1_pads(
    polygons: Iterable[LayerPolygon],
    *,
    enclosure_nm: int = GF180_M1_CONTACT_EOL_ENCLOSURE_NM,
    min_m1_width_nm: int = GF180_M1_MIN_WIDTH_NM,
) -> List[LayerPolygon]:
    """Return ``polygons`` plus one M1 pad per contact (convenience for layout).

    The original shapes are preserved (in order) and the generated M1 pads are
    appended, so the result is a complete shape list ready to emit to GDS with
    every contact CO.6-covered by Metal1.
    """
    shapes = list(polygons)
    pads = add_m1_pads(
        shapes, enclosure_nm=enclosure_nm, min_m1_width_nm=min_m1_width_nm
    )
    return shapes + pads


__all__ = [
    "GF180_CONTACT_LAYER",
    "GF180_CONTACT_DATATYPE",
    "GF180_METAL1_LAYER",
    "GF180_METAL1_DATATYPE",
    "GF180_M1_CONTACT_ENCLOSURE_MIN_NM",
    "GF180_M1_CONTACT_EOL_ENCLOSURE_NM",
    "GF180_M1_MIN_WIDTH_NM",
    "GF180_M1_CONTACT_ENCLOSURE_RULE_ID",
    "GF180_M1_CONTACT_ENCLOSURE_DECK_LINE",
    "GF180_DECK_RELPATH",
    "M1PadResult",
    "m1_pad_for_contact",
    "add_m1_pads",
    "add_m1_pads_result",
    "contacts_with_m1_pads",
]
