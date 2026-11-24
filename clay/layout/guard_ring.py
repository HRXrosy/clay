"""Guard-ring wrapper (TED-inspired, fresh reimplement).

TED exposes ``guardring(inst, connectTo, margin)``: it wraps a *placed* block
with a continuous tap-diffusion ring (well + tap-contact frame) tied to a supply
net (``vdd`` / ``vss``). This module supplies the analogous capability on Clay's
Phase 3a ``NamedCell`` / ``ParameterizedLibrary`` geometry IR.

What a guard ring physically is
-------------------------------
A guard ring is a continuous, hollow rectangular *frame* of substrate/well
contacts surrounding a block, biased to a supply net so it collects stray
carriers (latch-up immunity, noise isolation). For each supply polarity the
frame is built on three real layers plus a well marker:

================  ===================  ===========================================
``connect_to``    well / substrate     layers drawn (per ``pdk_layers``)
================  ===================  ===========================================
``vss`` (p-tap)   p-substrate          PDIFF (p+ tap) + CONTACT + METAL1 + PWELL_MARKER
``vdd`` (n-tap)   n-well               NDIFF (n+ tap) + CONTACT + METAL1 + NWELL
================  ===================  ===========================================

The tap / contact / metal layers are drawn as a *frame*: four rectangular
segments (left, right, bottom, top) of width ``ring_width_nm``, so the ring is
hollow (the enclosed block geometry is untouched). The well marker (NWELL for a
vdd ring) is drawn as a *solid* rectangle covering the whole framed region,
because a well is a filled region, not a frame.

Geometry contract (matches the BL-138A acceptance)
---------------------------------------------------
The ring is offset *outward* so the augmented cell's bounding box equals the
original cell bbox expanded by ``margin_nm`` on every side. Concretely the ring
frame's **outer** edge sits at ``cell_bbox ± margin_nm`` and its inner edge sits
``ring_width_nm`` inward of that. The original cell polygons are copied through
unchanged; only ring polygons are appended.

Honest limitations
-------------------
- ``connect_to`` is recorded conceptually (which well/tap polarity to draw); no
  actual net/label geometry is emitted (Clay's IR has no net-label primitive
  yet — that is a separate capability). The metal frame *is* drawn so a router
  can later strap it to the supply.
- PWELL_MARKER is ``None`` in the bundled sky130 / gf180mcu LayerMaps, so a
  ``vss`` ring on those PDKs emits PDIFF + CONTACT + METAL1 but *no* pwell
  marker layer. This is flagged (see ``GuardRingResult.skipped_layers``) rather
  than fabricating a marker layer that the PDK does not define.
- The ring is a single-row contact frame; multi-row / segmented rings and via
  stacks to upper metals are out of scope for this wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .backend import (
    GdsLayer,
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    ParameterizedLibrary,
    Point,
    Polygon,
)

BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)

# Default ring trace width if the caller does not override it. Chosen to be a
# small positive DBU width (1 unit per nm); callers pass ring_width_nm to size
# the frame relative to their PDK's min tap/contact rules.
_DEFAULT_RING_WIDTH_NM = 200.0

# Supply-net polarity -> (well marker logical layer, tap-diffusion logical layer).
# vdd: n-tap inside an n-well -> NWELL marker + NDIFF (n+) tap.
# vss: p-tap in the p-substrate -> PWELL_MARKER + PDIFF (p+) tap.
_CONNECT_TO_LAYERS = {
    "vdd": (LogicalLayer.NWELL, LogicalLayer.NDIFF),
    "vss": (LogicalLayer.PWELL_MARKER, LogicalLayer.PDIFF),
}


@dataclass(eq=True)
class GuardRingResult:
    """Result of applying a guard ring.

    Attributes:
        library: the augmented ParameterizedLibrary (original cells preserved;
            the ringed cell replaces the named target in-place / is appended).
        cell_name: name of the ringed cell.
        connect_to: the supply net the ring is conceptually tied to.
        ring_layers: list of (logical-layer name, GdsLayer) actually drawn.
        skipped_layers: logical-layer names requested but undefined in the PDK
            (e.g. PWELL_MARKER on sky130) — emitted as TODO markers, not faked.
    """

    library: ParameterizedLibrary
    cell_name: str
    connect_to: str
    ring_layers: List[Tuple[str, GdsLayer]] = field(default_factory=list)
    skipped_layers: List[str] = field(default_factory=list)


def _nm_to_dbu(value_nm: float) -> int:
    return int(round(value_nm))


def _bbox(polys: List[LayerPolygon]) -> Optional[BBox]:
    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _copy_polys(polys: List[LayerPolygon]) -> List[LayerPolygon]:
    """Deep copy so the original cell's polygon objects are never mutated."""
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(vertices=[Point(v.x, v.y) for v in lp.polygon.vertices]),
        )
        for lp in polys
    ]


def _frame_segments(
    outer: BBox, ring_width: int, gds: GdsLayer
) -> List[LayerPolygon]:
    """Four rectangular segments forming a hollow frame.

    ``outer`` is the outer bbox of the frame; the frame is ``ring_width`` thick
    inward of every outer edge. Returns left, right, bottom, top segments. When
    ``ring_width`` is large enough that the frame would self-overlap (small
    cell), the inner edge is clamped so the segments stay within ``outer``.
    """
    x0, y0, x1, y1 = outer
    # Inner edge of the frame, clamped so left<=right / bottom<=top.
    ix0 = min(x0 + ring_width, x1)
    iy0 = min(y0 + ring_width, y1)
    ix1 = max(x1 - ring_width, x0)
    iy1 = max(y1 - ring_width, y0)

    def lp(xa: int, ya: int, xb: int, yb: int) -> LayerPolygon:
        return LayerPolygon(
            layer=gds.layer,
            datatype=gds.datatype,
            polygon=Polygon.rectangle(xa, ya, xb, yb),
        )

    return [
        lp(x0, y0, ix0, y1),   # left   (full height)
        lp(ix1, y0, x1, y1),   # right  (full height)
        lp(ix0, y0, ix1, iy0),  # bottom (between left & right)
        lp(ix0, iy1, ix1, y1),  # top    (between left & right)
    ]


def _solid_rect(outer: BBox, gds: GdsLayer) -> LayerPolygon:
    x0, y0, x1, y1 = outer
    return LayerPolygon(
        layer=gds.layer,
        datatype=gds.datatype,
        polygon=Polygon.rectangle(x0, y0, x1, y1),
    )


def ring_polygons_for_cell(
    cell: NamedCell,
    connect_to: str,
    margin_nm: float,
    pdk_layers: LayerMap,
    *,
    ring_width_nm: float = _DEFAULT_RING_WIDTH_NM,
) -> Tuple[List[LayerPolygon], List[Tuple[str, GdsLayer]], List[str]]:
    """Compute the guard-ring polygons for ``cell`` (no library mutation).

    Returns ``(ring_polys, drawn_layers, skipped_layers)``.

    The ring's outer edge is the cell bbox grown by ``margin_nm`` on each side,
    so ``bbox(cell.polygons + ring_polys)`` == ``cell_bbox`` ± ``margin_nm``.

    Tap / contact / metal layers are drawn as a hollow frame of width
    ``ring_width_nm``; the well marker (if defined) is a solid rectangle over the
    whole framed region. Undefined PDK layers are reported in ``skipped_layers``.
    """
    key = connect_to.strip().lower()
    if key not in _CONNECT_TO_LAYERS:
        raise ValueError(
            f"connect_to must be 'vdd' or 'vss', got {connect_to!r}"
        )
    if margin_nm < 0:
        raise ValueError("margin_nm must be >= 0")
    if ring_width_nm <= 0:
        raise ValueError("ring_width_nm must be > 0")

    bb = _bbox(cell.polygons)
    if bb is None:
        # Empty cell: nothing to enclose. Caller still gets a (no-op) result.
        return [], [], []

    margin = _nm_to_dbu(margin_nm)
    ring_width = _nm_to_dbu(ring_width_nm)
    x0, y0, x1, y1 = bb
    outer: BBox = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)

    well_layer, tap_layer = _CONNECT_TO_LAYERS[key]

    ring_polys: List[LayerPolygon] = []
    drawn: List[Tuple[str, GdsLayer]] = []
    skipped: List[str] = []

    # Frame layers: tap diffusion + contact + metal1, in drawing order.
    frame_logicals = [tap_layer, LogicalLayer.CONTACT, LogicalLayer.METAL1]
    for logical in frame_logicals:
        gds = pdk_layers.gds_layer(logical)
        if gds is None:
            # TODO: PDK does not define this layer; do not fabricate one.
            skipped.append(logical.value)
            continue
        ring_polys.extend(_frame_segments(outer, ring_width, gds))
        drawn.append((logical.value, gds))

    # Well marker: solid rectangle over the framed region.
    well_gds = pdk_layers.gds_layer(well_layer)
    if well_gds is None:
        # TODO: e.g. PWELL_MARKER is None in bundled sky130 / gf180mcu maps.
        skipped.append(well_layer.value)
    else:
        ring_polys.append(_solid_rect(outer, well_gds))
        drawn.append((well_layer.value, well_gds))

    return ring_polys, drawn, skipped


def ring_cell(
    cell: NamedCell,
    connect_to: str,
    margin_nm: float,
    pdk_layers: LayerMap,
    *,
    ring_width_nm: float = _DEFAULT_RING_WIDTH_NM,
    name: Optional[str] = None,
) -> NamedCell:
    """Return a NEW NamedCell = original geometry (unchanged) + guard ring.

    The returned cell is a valid placeable ``NamedCell`` (so it composes via
    ``spatial_combinators.row`` / ``col``). The original ``cell`` is not mutated.
    """
    ring_polys, _drawn, _skipped = ring_polygons_for_cell(
        cell, connect_to, margin_nm, pdk_layers, ring_width_nm=ring_width_nm
    )
    polys = _copy_polys(cell.polygons) + ring_polys
    out_name = name if name is not None else f"{cell.name}_gr_{connect_to.strip().lower()}"
    return NamedCell(name=out_name, polygons=polys)


# =============================================================================
# PDK-rule-driven guard-ring constructor (T4, v0.4.31)
# =============================================================================
#
# ``build_guard_ring(tech, inner_bbox, *, well=, nets=)`` is the modern,
# LayoutTech-driven constructor: a closed tap ring (diffusion ring + contact
# array + M1 ring + implant ring [+ NWELL ring for well='n']) whose EVERY
# dimension resolves from the PDK rule surface -- the same shared
# ``DeviceCellRules`` resolver the device builder uses (single rules source,
# no second resolution path, no hardcoded nm, no per-PDK branch here).
#
# It coexists with the legacy Phase-3a ``ring_cell``/``apply_guard_ring``
# wrappers above (NamedCell IR, fixed-width frame, no contacts); those stay
# byte-unchanged for their existing callers. New work consumes
# ``build_guard_ring``.
#
# Honest scope:
#   * The ring guarantees the clearances DERIVABLE from the rule surface
#     (active spacing, implant-to-opposite-COMP + same-implant spacing, M1
#     spacing, NWELL spacing). Interactions of the ENCLOSED content beyond
#     ``inner_bbox`` (e.g. an enclosed PMOS nwell that should MERGE with an
#     n-ring well) are the integrator's account -- pass a grown ``inner_bbox``
#     or ``extra_clearance_nm`` until a well-merge contract exists.
#   * Single-row cut ring; multi-row rings are out of scope.
#   * An n-ring on a PDK that draws no nwell raises ``ValueError`` (explicit
#     unsupported, never a silent fallback).


@dataclass(frozen=True)
class GuardRingGeom:
    """A constructed guard ring (cell fragment, consumed like a DeviceGeom).

    ``polygons``      -- every drawn shape (tech-resolved GDS keys).
    ``terminals``     -- ``{'tap': Terminal}``: the ring's net access on the
                         M1 ring (bottom segment). For ``well='p'`` the default
                         net is the PDK substrate net.
    ``well``          -- 'p' (substrate tap ring) or 'n' (nwell tap ring).
    ``inner_bbox``    -- the protected region the caller passed (nm).
    ``opening_bbox``  -- the innermost drawn ring edge (the common hole). For
                         a legacy union-bbox call it contains ``inner_bbox``;
                         for a layer-aware call each drawn ring layer instead
                         protects its corresponding declared layer bbox, so
                         unrelated overhang in the full union need not fit in
                         this common opening.
    ``ring_bbox``     -- the outermost drawn extent.
    ``centerline_bbox`` -- the rectangle the cut centers sit on.
    ``cut_count``     -- number of contact cuts emitted.
    ``layers_drawn``  -- role name -> GDS key actually drawn.
    """

    polygons: List["object"]
    terminals: Dict[str, "object"]
    well: str
    inner_bbox: BBox
    opening_bbox: BBox
    ring_bbox: BBox
    centerline_bbox: BBox
    cut_count: int
    layers_drawn: Dict[str, Tuple[int, int]]

    def polys_on(self, key: Tuple[int, int]) -> List["object"]:
        return [lp for lp in self.polygons if (lp.layer, lp.datatype) == key]

    def extent(self) -> BBox:
        return self.ring_bbox


def _ring_rect(key: Tuple[int, int], box: BBox):
    x0, y0, x1, y1 = (int(v) for v in box)
    return LayerPolygon(
        layer=key[0],
        datatype=key[1],
        polygon=Polygon.rectangle(x0, y0, x1, y1),
    )


def _frame_rects(key, inner: BBox, outer: BBox) -> List["object"]:
    """Four rectangles forming a closed hollow frame between ``inner`` (the
    hole) and ``outer``; horizontal segments span the full outer width so the
    merged region is one donut polygon with one hole."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return [
        _ring_rect(key, (ox0, oy0, ox1, iy0)),  # bottom (full width)
        _ring_rect(key, (ox0, iy1, ox1, oy1)),  # top    (full width)
        _ring_rect(key, (ox0, iy0, ix0, iy1)),  # left
        _ring_rect(key, (ix1, iy0, ox1, iy1)),  # right
    ]


def _grow(box: BBox, d: int) -> BBox:
    return (box[0] - d, box[1] - d, box[2] + d, box[3] + d)


def _snap_up(v: int, g: int) -> int:
    return ((int(v) + g - 1) // g) * g


def _snap_down(v: int, g: int) -> int:
    return (int(v) // int(g)) * int(g)


def _exact_protected_bbox(value: object, label: str) -> BBox:
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(item) is not int for item in value)
    ):
        raise TypeError(f"{label} must be an exact integer bbox")
    box = value
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{label} must be non-degenerate")
    return box


def _cut_centres(centre: BBox, pitch_nm: int, grid: int) -> List[Tuple[int, int]]:
    """Cut centres on the ring centerline: one at each corner plus interior
    cuts distributed evenly along each edge at >= ``pitch_nm`` centre-to-centre
    (after grid snap). Even distribution (not fixed-pitch fill) so the residual
    gap before a corner can never collapse below the pitch."""
    cx0, cy0, cx1, cy1 = centre
    # snap guard: even-interval >= pitch + 2*grid keeps >= pitch after each
    # interior point is rounded to the grid (each moves at most grid/2).
    safe = pitch_nm + 2 * grid
    pts: List[Tuple[int, int]] = [(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)]

    def _interior(a: int, b: int) -> List[int]:
        span = b - a
        k = span // safe  # number of even intervals between the two corners
        if k < 2:
            return []
        out = []
        for i in range(1, k):
            v = a + (i * span) // k
            out.append(((v + grid // 2) // grid) * grid)
        return out

    for x in _interior(cx0, cx1):
        pts.append((x, cy0))
        pts.append((x, cy1))
    for y in _interior(cy0, cy1):
        pts.append((cx0, y))
        pts.append((cx1, y))
    return pts


def build_guard_ring(
    tech,
    inner_bbox: BBox,
    *,
    well: str = "p",
    nets: Optional[Dict[str, str]] = None,
    extra_clearance_nm: int = 0,
    shared_well: bool = False,
    protected_active_bbox: Optional[BBox] = None,
    protected_poly_bbox: Optional[BBox] = None,
    protected_metal1_bbox: Optional[BBox] = None,
    protected_same_implant_bbox: Optional[BBox] = None,
) -> GuardRingGeom:
    """Build a PDK-rule-driven guard ring around ``inner_bbox``.

    Parameters
    ----------
    tech : LayoutTech
        The aggregated PDK surface (``LayoutTech.resolve(pdk_id)``).
    inner_bbox : (x0, y0, x1, y1) nm
        The protected region (e.g. ``DeviceGeom.extent()`` of the enclosed
        device/group). The ring keeps every PDK clearance from this box.
    well : {'p', 'n'}
        'p' -> substrate-tap ring (p+ tap, pplus implant, no well drawn);
        'n' -> nwell-tap ring (n+ tap, nplus implant, NWELL ring).
    nets : dict, optional
        ``{'tap': net_name}`` binding for the ring terminal. Unknown keys
        raise. Default net: the PDK substrate net for 'p', '' for 'n'.
    extra_clearance_nm : int
        Additional inner clearance on top of the rule-derived one (for
        integrator-known interactions beyond ``inner_bbox``).
    shared_well : bool
        For an n-type tap ring whose N-well is materialized as one solid PMOS
        body carrier, omit only the N-well hole/inter-island spacing terms.
        Tap active, implant, contact, M1 and N-well tap enclosure remain drawn
        and rule-derived.  Valid only with ``well='n'``.
    protected_active_bbox, protected_poly_bbox, protected_metal1_bbox : exact integer bbox, optional
        Canonical per-layer protected extents. They must be supplied together
        and lie inside ``inner_bbox``. Active/implant, field-poly-to-active and
        M1 clearances are then replayed against their own extents instead of
        charging every rule to the full multi-layer union. Layer-aware n-rings
        are valid only for one declared shared-well domain.
    protected_same_implant_bbox : exact integer bbox, optional
        Actual enclosed tap-implant extent on the ring's own implant layer.
        Its same-layer spacing is independent of active spacing: an enclosed
        tap implant can protrude beyond COMP by the foundry enclosure.

    Returns
    -------
    GuardRingGeom

    Raises
    ------
    ValueError
        for an invalid ``well`` / degenerate ``inner_bbox`` / unknown ``nets``
        key, and for an n-ring on a PDK with no drawable nwell (explicit
        unsupported -- never silently skipped).
    """
    from .device_cell import Terminal, _active_name, _rules_from_tech, _space_or

    if well not in ("p", "n"):
        raise ValueError(f"well must be 'p' or 'n', got {well!r}")
    if type(shared_well) is not bool:
        raise TypeError("shared_well must be an exact bool")
    if shared_well and well != "n":
        raise ValueError("shared_well requires well='n'")
    x0, y0, x1, y1 = (int(v) for v in inner_bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"inner_bbox is degenerate: {inner_bbox!r}")
    nets = dict(nets or {})
    unknown = set(nets) - {"tap"}
    if unknown:
        raise ValueError(f"unknown nets keys {sorted(unknown)!r}; known: ['tap']")
    if extra_clearance_nm < 0:
        raise ValueError("extra_clearance_nm must be >= 0")
    layer_aware = any(
        value is not None
        for value in (
            protected_active_bbox,
            protected_poly_bbox,
            protected_metal1_bbox,
            protected_same_implant_bbox,
        )
    )
    if layer_aware:
        if (
            protected_active_bbox is None
            or protected_poly_bbox is None
            or protected_metal1_bbox is None
        ):
            raise ValueError(
                "layer-aware ring construction requires active, poly, and metal1 bboxes"
            )
        active_protected = _exact_protected_bbox(
            protected_active_bbox, "protected_active_bbox"
        )
        poly_protected = _exact_protected_bbox(
            protected_poly_bbox, "protected_poly_bbox"
        )
        metal1_protected = _exact_protected_bbox(
            protected_metal1_bbox, "protected_metal1_bbox"
        )
        same_implant_protected = (
            _exact_protected_bbox(protected_same_implant_bbox, "protected_same_implant_bbox")
            if protected_same_implant_bbox is not None else None
        )
        protected_boxes = [
            ("protected_active_bbox", active_protected),
            ("protected_poly_bbox", poly_protected),
            ("protected_metal1_bbox", metal1_protected),
        ]
        if same_implant_protected is not None:
            protected_boxes.append(("protected_same_implant_bbox", same_implant_protected))
        for label, box in protected_boxes:
            if not (
                x0 <= box[0]
                and y0 <= box[1]
                and x1 >= box[2]
                and y1 >= box[3]
            ):
                raise ValueError(f"{label} must lie inside inner_bbox")
        if well == "n" and not shared_well:
            raise ValueError(
                "layer-aware n-ring construction requires one shared well domain"
            )
    # Auxiliary/deep-well and legacy union-bbox rings must use the same grid
    # as placement and layer-aware rings. A separate fixed cut-grid changes
    # contact positions when a solved rigid owner is regenerated after motion.
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    ring_grid_nm = max(1, int(pdk_manufacturing_grid_nm(tech)))

    r = _rules_from_tech(tech)
    iw = tech.implant_well()
    bt = tech.bulk_topology()
    implant_dist = bool(bt.get("implant_distinguished", True))
    if protected_same_implant_bbox is not None and not implant_dist:
        raise ValueError("same-implant protection requires a drawable implant layer")

    # ---- layer keys (the tap recipe: opposite-implant tap on active) --------
    if implant_dist:
        active_key = tech.build_layer("diff")
        implant_key = tech.build_layer("pplus" if well == "p" else "nplus")
    else:
        active_key = tech.build_layer("pdiff" if well == "p" else "ndiff")
        implant_key = None
    if active_key is None:
        raise ValueError(f"{tech.pdk_id}: no active/tap-diffusion layer drawable")
    cut_key = tech.contact_key()
    m1_key = tech.metal_key(1)
    nwell_key = tech.build_layer("nwell") if well == "n" else None
    if well == "n" and nwell_key is None:
        raise ValueError(
            f"{tech.pdk_id}: n-type guard ring unsupported (PDK draws no nwell)"
        )

    # ---- frame widths (all from the shared DeviceCellRules resolver) --------
    csize = r.contact_size
    diff_w = max(r.channel_min_width, csize + 2 * r.comp_contact_enc)
    impl_min_w = int(iw.pplus_min_width if well == "p" else iw.nplus_min_width)
    # implant frame = diff frame grown both edges; grow further if the PDK
    # implant min width exceeds diff_w + 2*extension.
    impl_ext = max(r.implant_extension, _snap_up(max(0, impl_min_w - diff_w), 2) // 2)
    m1_cut_enc = max(r.metal1_contact_enc, r.metal1_line_end)
    m1_w = max(r.metal1_min_width, csize + 2 * m1_cut_enc)
    nwell_w = 0
    if well == "n":
        nwell_w = max(diff_w + 2 * r.nwell_ncomp_overlap, r.nwell_min_width)

    diff_half = diff_w // 2
    m1_half = m1_w // 2
    nwell_half = nwell_w // 2

    # ---- inner clearances (rule-derived, per drawn layer) -------------------
    comp_space = _space_or(tech, _active_name(tech), r.channel_min_width)
    impl_space = int(iw.pplus_min_space if well == "p" else iw.nplus_min_space)
    impl_clear = max(r.implant_to_opp_comp, impl_space)
    m1_clear = r.metal1_min_space
    nwell_clear = _space_or(tech, "nwell", r.nwell_min_width) if well == "n" else 0

    # Legacy callers carry one already-conservative union bbox.  A canonical
    # layer-aware caller instead proves active, poly, and M1 protected extents
    # independently, so an M1 terminal overhang does not also pay active and
    # implant spacing. Field poly is still protected from the ring active; this
    # prevents the tap contacts/active from consuming a MOS gate. The
    # contact-bearing ring has one common centerline, so each side is the
    # maximum exact requirement of its own layers.
    impl_in = diff_half + impl_ext
    if layer_aware:
        active_guard = max(
            comp_space + diff_half,
            impl_clear + impl_in if implant_key is not None else 0,
        ) + int(extra_clearance_nm)
        poly_guard = r.field_poly_to_comp + diff_half + int(extra_clearance_nm)
        metal1_guard = m1_clear + m1_half + int(extra_clearance_nm)
        protections = [
            (active_protected, active_guard),
            (poly_protected, poly_guard),
            (metal1_protected, metal1_guard),
        ]
        if same_implant_protected is not None:
            protections.append((
                same_implant_protected,
                impl_space + impl_in + int(extra_clearance_nm),
            ))
        centre = (
            _snap_down(
                min(box[0] - clearance for box, clearance in protections),
                ring_grid_nm,
            ),
            _snap_down(
                min(box[1] - clearance for box, clearance in protections),
                ring_grid_nm,
            ),
            _snap_up(
                max(box[2] + clearance for box, clearance in protections),
                ring_grid_nm,
            ),
            _snap_up(
                max(box[3] + clearance for box, clearance in protections),
                ring_grid_nm,
            ),
        )
        min_centre_span = max(
            2 * diff_half + comp_space,
            2 * m1_half + m1_clear,
            csize + r.contact_space,
            2 * impl_in + impl_space if implant_key is not None else 0,
        )
        cx0, cy0, cx1, cy1 = centre
        if cx1 - cx0 < min_centre_span:
            deficit = min_centre_span - (cx1 - cx0)
            cx0 = _snap_down(cx0 - (deficit + 1) // 2, ring_grid_nm)
            cx1 = _snap_up(cx1 + deficit // 2, ring_grid_nm)
        if cy1 - cy0 < min_centre_span:
            deficit = min_centre_span - (cy1 - cy0)
            cy0 = _snap_down(cy0 - (deficit + 1) // 2, ring_grid_nm)
            cy1 = _snap_up(cy1 + deficit // 2, ring_grid_nm)
        centre: BBox = (cx0, cy0, cx1, cy1)
    else:
        inner_min_dim = min(x1 - x0, y1 - y0)
        cut_pitch = csize + r.contact_space
        terms = [
            comp_space + diff_half,
            m1_clear + m1_half,
            diff_half + ((comp_space - inner_min_dim + 1) // 2),
            m1_half + ((m1_clear - inner_min_dim + 1) // 2),
            (cut_pitch - inner_min_dim + 1) // 2,
        ]
        if implant_key is not None:
            terms.append(impl_clear + impl_in)
            terms.append(impl_in + ((impl_space - inner_min_dim + 1) // 2))
        if well == "n" and not shared_well:
            terms.append(nwell_clear + nwell_half)
            terms.append(nwell_half + ((nwell_clear - inner_min_dim + 1) // 2))
        offset = _snap_up(max(terms) + int(extra_clearance_nm), ring_grid_nm)
        centre = (
            _snap_down(x0 - offset, ring_grid_nm),
            _snap_down(y0 - offset, ring_grid_nm),
            _snap_up(x1 + offset, ring_grid_nm),
            _snap_up(y1 + offset, ring_grid_nm),
        )

    # ---- frames --------------------------------------------------------------
    polys: List["object"] = []
    layers_drawn: Dict[str, Tuple[int, int]] = {}

    diff_inner = _grow(centre, -diff_half)
    diff_outer = _grow(centre, diff_w - diff_half)
    polys += _frame_rects(active_key, diff_inner, diff_outer)
    layers_drawn["tap_diff"] = active_key

    if implant_key is not None:
        polys += _frame_rects(
            implant_key, _grow(diff_inner, -impl_ext), _grow(diff_outer, impl_ext)
        )
        layers_drawn["tap_implant"] = implant_key

    m1_inner = _grow(centre, -m1_half)
    m1_outer = _grow(centre, m1_w - m1_half)
    polys += _frame_rects(m1_key, m1_inner, m1_outer)
    layers_drawn["metal1"] = m1_key

    if well == "n" and nwell_key is not None:
        nwell_outer = _grow(centre, nwell_w - nwell_half)
        if shared_well:
            polys.append(_ring_rect(nwell_key, nwell_outer))
        else:
            polys += _frame_rects(
                nwell_key,
                _grow(centre, -nwell_half),
                nwell_outer,
            )
        layers_drawn["nwell"] = nwell_key

    # ---- contact-cut array on the centerline ---------------------------------
    pitch = csize + r.contact_space
    centres = _cut_centres(centre, pitch, ring_grid_nm)
    c_lo = csize // 2
    c_hi = csize - c_lo
    for (ccx, ccy) in centres:
        polys.append(
            _ring_rect(cut_key, (ccx - c_lo, ccy - c_lo, ccx + c_hi, ccy + c_hi))
        )
    layers_drawn["contact"] = cut_key

    # ---- terminal: ring net access on the bottom M1 segment ------------------
    default_net = tech.substrate_net() if well == "p" else ""
    tap_net = nets.get("tap", default_net)
    tap_bbox: BBox = (m1_outer[0], m1_outer[1], m1_outer[2], m1_inner[1])
    terminals = {"tap": Terminal(role="tap", bbox=tap_bbox, net=tap_net)}

    # ---- metadata -------------------------------------------------------------
    max_in = max(
        diff_half + (impl_ext if implant_key is not None else 0),
        m1_half,
        nwell_half if not shared_well else 0,
    )
    max_out = max(
        (diff_w - diff_half) + (impl_ext if implant_key is not None else 0),
        m1_w - m1_half,
        nwell_w - nwell_half,
    )
    opening: BBox = _grow(centre, -max_in)
    ring_bbox: BBox = _grow(centre, max_out)

    return GuardRingGeom(
        polygons=polys,
        terminals=terminals,
        well=well,
        inner_bbox=(x0, y0, x1, y1),
        opening_bbox=opening,
        ring_bbox=ring_bbox,
        centerline_bbox=centre,
        cut_count=len(centres),
        layers_drawn=layers_drawn,
    )


def apply_guard_ring(
    lib: ParameterizedLibrary,
    cell_name: str,
    connect_to: str,
    margin_nm: float,
    pdk_layers: LayerMap,
    *,
    ring_width_nm: float = _DEFAULT_RING_WIDTH_NM,
    in_place: bool = False,
    ringed_name: Optional[str] = None,
) -> GuardRingResult:
    """Wrap ``cell_name`` in ``lib`` with a guard ring tied to ``connect_to``.

    Reads the named cell's bbox, draws a rectangular guard-ring frame offset
    outward by ``margin_nm`` on the tap / contact / metal1 layers plus a solid
    well marker, conceptually tied to ``connect_to`` (``vdd`` -> n-tap/nwell,
    ``vss`` -> p-tap/pwell). The original geometry is preserved unchanged.

    Behaviour:
      - The augmented library is a *new* ``ParameterizedLibrary`` (original
        ``lib`` untouched) unless ``in_place=True``.
      - The target cell is replaced by its ringed version (same name, unless
        ``ringed_name`` is given, in which case the ringed cell is appended and
        the original kept).

    Returns a ``GuardRingResult`` carrying the augmented library plus the list of
    layers drawn / skipped (skipped = requested-but-undefined PDK layers).
    """
    target: Optional[NamedCell] = None
    for c in lib.cells:
        if c.name == cell_name:
            target = c
            break
    if target is None:
        raise KeyError(
            f"cell {cell_name!r} not found in library {lib.name!r}; "
            f"have {[c.name for c in lib.cells]}"
        )

    ring_polys, drawn, skipped = ring_polygons_for_cell(
        target, connect_to, margin_nm, pdk_layers, ring_width_nm=ring_width_nm
    )

    ringed = NamedCell(
        name=(ringed_name if ringed_name is not None else cell_name),
        polygons=_copy_polys(target.polygons) + ring_polys,
    )

    if in_place:
        out_lib = lib
    else:
        out_lib = ParameterizedLibrary(
            name=lib.name,
            unit_user=lib.unit_user,
            unit_db_meters=lib.unit_db_meters,
            cells=[
                NamedCell(name=c.name, polygons=_copy_polys(c.polygons))
                for c in lib.cells
            ],
        )

    if ringed_name is not None and ringed_name != cell_name:
        # Keep original, append ringed copy under the new name.
        out_lib.cells.append(ringed)
    else:
        # Replace the target cell with its ringed version (preserve position).
        for i, c in enumerate(out_lib.cells):
            if c.name == cell_name:
                out_lib.cells[i] = ringed
                break

    return GuardRingResult(
        library=out_lib,
        cell_name=ringed.name,
        connect_to=connect_to.strip().lower(),
        ring_layers=drawn,
        skipped_layers=skipped,
    )
