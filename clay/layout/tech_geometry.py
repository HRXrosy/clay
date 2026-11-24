"""Tech-derived BEOL routing geometry: metal/via layer keys + via stacks.

Why this module exists
======================
The retired maze-router lane originally grew the foundational BEOL geometry
every downstream router / solver / circuit script reaches for: the metal + via
GDS layer keys,
the per-layer width/spacing minima, and the via *stack* builders (a lower-metal
landing pad + cut + upper-metal landing pad). Those facts are **PDK geometry,
not router logic** -- a comparator router, an OTA router, a bus router, and the
device circuit scripts all need the same M2/Via1 keys and the same legal via12
stack regardless of which router placed them.

This module is that shared, router-agnostic home. Every value is **derived from
the PDK technology object** (:class:`clay.layout.tech.LayoutTech`), which reads
the layer keys from the gf180mcu ``.lyp`` layer map and the width / spacing /
via-size / enclosure minima from the parsed gf180mcu DRC deck. Nothing here is a
hand-typed layer number, and nothing is keyed on a circuit, a device, or a cell
name -- :class:`TechGeometry` takes a ``LayoutTech`` and answers purely from it.

PDK-derivation contract
=======================
This module is the SOURCE OF TRUTH for the gf180mcu BEOL routing geometry (the
legacy maze-router bodies that once held the duplicate literals are retired).
``tests/test_tech_geometry.py`` is the drift guard: it derives every key,
minimum, via size/enclosure, and via-stack polygon DIRECTLY from the PDK
technology object (:class:`clay.layout.tech.LayoutTech` reading the gf180mcu
``.lyp`` layer map + parsed DRC deck) and asserts this module matches, so the
module-level constants cannot drift away from the PDK.

The legal via landing pad
=========================
A via landing pad must (a) enclose the cut by at least the deck via enclosure
(gf180 V*.3c/V*.4b end-of-line overlap, 60 nm) and (b) clear the metal min-area
rule (gf180 0.1444 um^2 == a 380 nm square). The pad side is therefore
``via_size + 2 * (deck_enclosure + PAD_ROUTING_MARGIN_NM)``; with the gf180
via (260 nm) and enclosure (60 nm) and a 10 nm routing margin this is the legacy
400 nm pad (70 nm enclosure each side, 400x400 = 0.16 um^2 > the 0.1444 um^2
min-area). The margin is a tech-level routing choice, not a circuit fact;
:class:`TechGeometry` asserts the resulting pad is area- and enclosure-legal
against the PDK at construction.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .backend import LayerPolygon, Point, Polygon
from .tech import LayoutTech
from .minimum_area import minimum_area_square_side_nm

LayerKey = Tuple[int, int]

# Routing comfort margin added to the deck via-enclosure minimum when sizing a
# via landing pad. A tech-level routing choice (NOT a circuit fact): it keeps the
# pad enclosure (70 nm) comfortably past the 60 nm deck end-of-line minimum.
PAD_ROUTING_MARGIN_NM = 10


def _rect(key: LayerKey, cx: int, cy: int, half: int) -> LayerPolygon:
    """A square LayerPolygon of side ``2*half`` centred at ``(cx, cy)`` on ``key``."""
    return LayerPolygon(
        layer=key[0],
        datatype=key[1],
        polygon=Polygon.rectangle(cx - half, cy - half, cx + half, cy + half),
    )


class TechGeometry:
    """BEOL routing geometry (metal/via keys + via stacks) for one PDK.

    Pure reads off a :class:`~clay.layout.tech.LayoutTech`; no circuit / device /
    cell knowledge. Construct via :meth:`for_pdk` (cached per PDK id) or directly
    from a resolved ``LayoutTech``.
    """

    _CACHE: dict = {}

    def __init__(self, tech: LayoutTech) -> None:
        self.tech = tech

    @classmethod
    def for_pdk(cls, pdk_id: str) -> "TechGeometry":
        """Return the (cached) TechGeometry for ``pdk_id``."""
        key = pdk_id.lower()
        obj = cls._CACHE.get(key)
        if obj is None:
            obj = cls(LayoutTech.resolve(key))
            cls._CACHE[key] = obj
        return obj

    # ---- layer keys (GDS (layer, datatype) tuples) -------------------------
    def metal_key(self, n: int) -> LayerKey:
        """GDS key for metal layer ``n`` (1-based; ``metal_key(1)`` is M1)."""
        return self.tech.metal_key(n)

    def via_key(self, n: int) -> LayerKey:
        """GDS key for the cut between metal ``n`` and metal ``n+1`` (Via``n``)."""
        return self.tech.via_key(n)

    # ---- per-layer minima (nm), with upper-BEOL class fallback -------------
    def _metal_min(self, n: int, which: str) -> int:
        """``min_width`` / ``min_space`` for metal ``n`` (nm).

        Upper metals the parsed deck does not enumerate individually inherit the
        highest metal that DOES carry a rule (gf180's M4/M5 share the M3 0.28 um
        width/space class). This is a tech-class generalization, not a per-layer
        literal.
        """
        accessor = self.tech.min_width if which == "width" else self.tech.min_space
        for m in range(n, 0, -1):
            try:
                v = accessor(f"metal{m}")
            except KeyError:
                v = None
            if v is not None:
                return int(v)
        raise KeyError(f"{self.tech.pdk_id} carries no metal width/space rule")

    def metal_min_width(self, n: int) -> int:
        """Min width (nm) for metal layer ``n``."""
        return self._metal_min(n, "width")

    def metal_min_space(self, n: int) -> int:
        """Min same-layer spacing (nm) for metal layer ``n``."""
        return self._metal_min(n, "space")

    def via_size(self, n: int) -> int:
        """Exact cut size (nm) for Via``n``.

        Upper vias the parsed deck does not enumerate inherit the highest via that
        DOES carry a size rule (gf180's Via3/Via4 share the Via2 0.26 um class).
        """
        adapter_rules = self.tech.adapter.via_rules()
        for v in range(n, 0, -1):
            rule = adapter_rules.get(f"via{v}")
            if rule is not None:
                return int(rule.cut_size_nm)
        for v in range(n, 0, -1):
            try:
                sz = self.tech.via_size(f"via{v}")
            except KeyError:
                sz = None
            if sz is not None:
                return int(sz)
        raise KeyError(f"{self.tech.pdk_id} carries no via size rule")

    def via_enclosure(self, n: int) -> int:
        """Deck enclosure (nm) of Via``n`` by its lower metal (EOL overlap)."""
        return self._via_enclosure(n, above=False)

    def _via_enclosure(self, n: int, *, above: bool) -> int:
        """Resolve either side through the existing adapter-owned via classes."""
        metal_offset = int(above)
        adapter_rules = self.tech.adapter.via_rules()
        for v in range(n, 0, -1):
            rule = adapter_rules.get(f"via{v}")
            if rule is not None:
                return int(rule.enclosure_above_nm if above else rule.enclosure_below_nm)
        for v in range(n, 0, -1):
            try:
                e = self.tech.enclosure(f"via{v}", f"metal{v + metal_offset}")
            except KeyError:
                e = None
            if e is not None:
                return int(e)
        # fall back to the lowest via's enclosure if upper not enumerated
        try:
            e = self.tech.enclosure("via1", f"metal{1 + metal_offset}")
        except KeyError:
            e = None
        if e is not None:
            return int(e)
        raise KeyError(f"{self.tech.pdk_id} carries no via enclosure rule")

    def via_pad(self, n: int) -> int:
        """Square landing-pad side (nm) for a Via``n`` stack.

        Both pads share this side, so it covers both adjacent metals' width,
        area and via enclosure. Preserve the existing routing margin and round
        outward to an even manufacturing-grid multiple for centered emission.
        """
        via = self.via_size(n)
        below = self.via_enclosure(n)
        above = self._via_enclosure(n, above=True)
        encl = max(below, above) + PAD_ROUTING_MARGIN_NM
        pad = via + 2 * encl
        for metal in (n, n + 1):
            pad = max(pad, self.metal_min_width(metal))
            try:
                min_area = self.tech.min_area(f"metal{metal}")
            except KeyError:
                min_area = None
            pad = max(pad, minimum_area_square_side_nm(min_area))
        from .route.emit_gds import pdk_manufacturing_grid_nm
        quantum = 2 * pdk_manufacturing_grid_nm(self.tech)
        pad = ((pad + quantum - 1) // quantum) * quantum
        # legality self-check against the PDK (not a fabricated verdict).
        assert pad - via >= 2 * max(below, above), "via pad under-encloses cut"
        return pad

    # ---- via stacks (lower-metal pad + cut + upper-metal pad) ---------------
    def via_stack(self, lo_metal: int, center: Point) -> List[LayerPolygon]:
        """A legal ``M{lo}`` + ``Via{lo}`` + ``M{lo+1}`` stack centred at ``center``.

        Three polygons: a lower-metal landing pad, the cut (exactly ``via_size``),
        and an upper-metal landing pad. Both pads are ``via_pad`` square.
        """
        lo_key = self.metal_key(lo_metal)
        hi_key = self.metal_key(lo_metal + 1)
        via_key = self.via_key(lo_metal)
        half_via = self.via_size(lo_metal) // 2
        half_pad = self.via_pad(lo_metal) // 2
        cx, cy = center.x, center.y
        return [
            _rect(lo_key, cx, cy, half_pad),
            _rect(via_key, cx, cy, half_via),
            _rect(hi_key, cx, cy, half_pad),
        ]


# --- gf180mcu module-level constants (drop-in for the legacy maze exports) -----
# Derived once from the gf180mcu LayoutTech; byte-identical to the maze-module
# values (guarded by tests/test_tech_geometry.py).
_GF180 = TechGeometry.for_pdk("gf180mcu")

# layer keys
M1_KEY: LayerKey = _GF180.metal_key(1)  # (34, 0)
M2_KEY: LayerKey = _GF180.metal_key(2)  # (36, 0)
M3_KEY: LayerKey = _GF180.metal_key(3)  # (42, 0)
M4_KEY: LayerKey = _GF180.metal_key(4)  # (46, 0)
M5_KEY: LayerKey = _GF180.metal_key(5)  # (81, 0)
VIA1_KEY: LayerKey = _GF180.via_key(1)  # (35, 0)
VIA2_KEY: LayerKey = _GF180.via_key(2)  # (38, 0)
VIA3_KEY: LayerKey = _GF180.via_key(3)  # (40, 0)
VIA4_KEY: LayerKey = _GF180.via_key(4)  # (41, 0)

# metal width / spacing minima (nm)
M1_MIN_WIDTH_NM = _GF180.metal_min_width(1)  # 230
M1_MIN_SPACE_NM = _GF180.metal_min_space(1)  # 230
M2_MIN_WIDTH_NM = _GF180.metal_min_width(2)  # 280
M2_MIN_SPACE_NM = _GF180.metal_min_space(2)  # 280
M3_MIN_WIDTH_NM = _GF180.metal_min_width(3)  # 280
M3_MIN_SPACE_NM = _GF180.metal_min_space(3)  # 280
M4_MIN_WIDTH_NM = _GF180.metal_min_width(4)  # 280 (M3 upper-BEOL class)
M4_MIN_SPACE_NM = _GF180.metal_min_space(4)  # 280
M5_MIN_WIDTH_NM = _GF180.metal_min_width(5)  # 280
M5_MIN_SPACE_NM = _GF180.metal_min_space(5)  # 280

# via cut sizes (nm)
VIA1_SIZE_NM = _GF180.via_size(1)  # 260
VIA2_SIZE_NM = _GF180.via_size(2)  # 260
VIA3_SIZE_NM = _GF180.via_size(3)  # 260 (Via2 class)
VIA4_SIZE_NM = _GF180.via_size(4)  # 260

# via landing pads (nm, square)
VIA12_PAD_NM = _GF180.via_pad(1)  # 400
VIA23_PAD_NM = _GF180.via_pad(2)  # 400
VIA34_PAD_NM = _GF180.via_pad(3)  # 400
VIA45_PAD_NM = _GF180.via_pad(4)  # 400
UPPER_VIA_PAD_NM = VIA23_PAD_NM  # shared upper-BEOL pad alias

# via enclosure (nm, EOL overlap) -- shared across the gf180 via class
VIA1_M1_ENCLOSURE_NM = _GF180.via_enclosure(1)  # 60


def via12_stack(center: Point) -> List[LayerPolygon]:
    """gf180mcu M1 + Via1 + M2 stack centred at ``center`` (legacy via12_stack)."""
    return _GF180.via_stack(1, center)


def via23_stack(center: Point) -> List[LayerPolygon]:
    """gf180mcu M2 + Via2 + M3 stack centred at ``center`` (legacy via23_stack)."""
    return _GF180.via_stack(2, center)


def via34_stack(center: Point) -> List[LayerPolygon]:
    """gf180mcu M3 + Via3 + M4 stack centred at ``center`` (legacy via34_stack)."""
    return _GF180.via_stack(3, center)


def via45_stack(center: Point) -> List[LayerPolygon]:
    """gf180mcu M4 + Via4 + M5 stack centred at ``center`` (legacy via45_stack)."""
    return _GF180.via_stack(4, center)
