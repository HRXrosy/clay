"""Per-device geometric-environment extractor (the LDE measurement substrate).

This module is the NEW measurement substrate three layout-dependent-effect (LDE)
intents need. Until now there was no per-device store of well / OD geometry: the
router records per-device terminal bboxes (``placement.term[label] = {"d":(x0,y0,
x1,y1), ...}``) and the PEX backend records per-NET parasitics, but neither tells
you how a single device sits inside its well or relative to its neighbours. The
analog matching effects below depend on exactly that geometry:

  * **WPE (well-proximity effect)** -- dopant scattering off the photoresist edge
    during the well implant shifts a transistor's Vt as a function of its distance
    to the *well boundary*. Two devices a designer declared "matched" drift apart
    in Vt if their active regions sit at different distances from the enclosing
    well edge. Captured here as :attr:`DeviceEnvironment.well_edge_distance` --
    for a PMOS the nearest NWELL edge; for an NMOS the nearest pwell-implant edge
    (PWELL when drawn, else there is no enclosing well and the value is ``None``).

  * **LOD / STI stress (length-of-diffusion)** -- mechanical stress from the
    shallow-trench-isolation oxide at the diffusion edge changes mobility as a
    function of how far the poly gate sits from the OD (diffusion) edge along the
    source/drain direction. Captured as :attr:`DeviceEnvironment.od_length`: the
    device's OD/diffusion extent along the channel (source-drain) axis, i.e. the
    poly-gate-to-diffusion-edge extension that sets the STI stress.

  * **OD-spacing stress** -- a neighbouring diffusion island modulates the same
    STI stress; an asymmetric neighbour gap between two "matched" devices is an
    asymmetric stress, hence an asymmetric Vt/mobility offset. Captured as
    :attr:`DeviceEnvironment.od_neighbor_spacing`: the minimum spacing from this
    device's OD bbox to the nearest OTHER device's OD bbox.

Every metric is an honest distance computed from the real polygons / bboxes the
caller passes; nothing is fabricated. When the layer a metric needs is absent
from the extraction (e.g. no NWELL polygons), that metric is ``None`` -- which
the obligation layer must surface as ``unmeasurable`` (token withheld), NEVER as
a silent pass.

APPROXIMATIONS (stated explicitly, per project rules):
  * A device's "active region" is taken as the bounding box of its source / drain
    / gate terminal bboxes (the same union the router/centroid code uses). This is
    a bbox, not the true diffusion polygon outline; for the rectangular devices
    this flow emits the two coincide, but a non-rectangular OD would be
    over-approximated by its bbox.
  * ``od_length`` is read as the diffusion bbox extent along the source->drain
    axis. The S->D axis is inferred from the terminal bboxes: it is the axis along
    which the source and drain terminal bboxes are separated (the channel runs
    perpendicular to the poly gate). When source/drain bboxes are unavailable the
    longer bbox side is used and the docstring of :func:`extract_device_environment`
    notes the fallback.
  * Well/diff polygons are attributed to a device by geometric containment /
    nearness of the polygon to the device's active bbox (a polygon is "this
    device's" diffusion if it overlaps the device's active bbox). A polygon shared
    by several devices is attributed to each it overlaps.
  * All distances are point/edge-to-rectangle Euclidean distances in database
    units (dbu). Because every reported environment metric that feeds an
    obligation is later reduced to a RELATIVE imbalance |a-b|/max(a,b), the dbu
    unit cancels and scaling the whole layout does not change a verdict.

PDK-neutral: layer selection goes exclusively through the passed ``pdk_layers``
LayerMap (``gds_layer(LogicalLayer.X) -> Optional[GdsLayer]``); no PDK numbers are
hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from clay.layout.backend import GdsLayer, LayerPolygon, LogicalLayer

# A terminal-bbox dict as the router records it: {"d": (x0,y0,x1,y1), "g": ...}.
# Values are (x_min, y_min, x_max, y_max) in database units.
Bbox = Tuple[float, float, float, float]
TermDict = Dict[str, Bbox]


class _LayerMap(Protocol):
    """Structural type for the PDK layer map (mirror of backend.LayerMap)."""

    def gds_layer(self, logical: LogicalLayer) -> Optional[GdsLayer]:
        ...


@dataclass(frozen=True)
class DeviceEnvironment:
    """The per-device geometric environment, one instance per device label.

    All distances are in database units (dbu); any obligation built on them
    reduces to a RELATIVE imbalance so the unit cancels. ``None`` marks a metric
    whose required layer was absent from the extraction -- the obligation that
    consumes it must report ``unmeasurable`` (blocking), never a silent pass.

    Attributes
    ----------
    well_edge_distance:
        Minimum distance from the device's active/diffusion region to the nearest
        enclosing-well boundary -- the NWELL edge for a PMOS, the
        pwell/p-implant edge for an NMOS. Drives WPE-matching obligations.
        ``None`` when no enclosing-well polygon is present for the device's type
        (e.g. an NMOS in a flow that draws no pwell marker, or NWELL absent).
    od_length:
        The device's OD/diffusion extent along the channel (source-drain)
        direction -- the poly-gate-to-diffusion-edge extension that sets STI/LOD
        stress. ``None`` when no diffusion polygon is attributable to the device.
    od_neighbor_spacing:
        Minimum spacing from this device's OD/diffusion bbox to the nearest OTHER
        device's OD/diffusion bbox (0 if they touch/overlap). Drives OD-spacing
        stress matching. ``None`` when fewer than two devices have an attributable
        diffusion region (no neighbour to measure against).
    """

    well_edge_distance: Optional[float]
    od_length: Optional[float]
    od_neighbor_spacing: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# geometry primitives (point / rectangle distances, all in dbu)
# ─────────────────────────────────────────────────────────────────────────────

def _poly_bbox(poly) -> Bbox:
    """(x_min, y_min, x_max, y_max) over a Polygon's vertices."""
    xs = [v.x for v in poly.vertices]
    ys = [v.y for v in poly.vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def _union_bbox(boxes: Sequence[Bbox]) -> Bbox:
    """Union bounding box of several bboxes."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def _bboxes_overlap(a: Bbox, b: Bbox) -> bool:
    """True iff two axis-aligned bboxes overlap (touching counts as overlap)."""
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_gap(a: Bbox, b: Bbox) -> float:
    """Minimum edge-to-edge separation between two bboxes (0 if overlapping).

    Standard separable-axis rectangle distance: the gap on each axis is how far
    apart the intervals are (negative when they overlap, clamped to 0); the
    Euclidean combination of the two non-negative axis gaps is the true minimum
    distance between the rectangles.
    """
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return (dx * dx + dy * dy) ** 0.5


def _point_seg_distance(px, py, x0, y0, x1, y1) -> float:
    """Euclidean distance from point (px,py) to segment ((x0,y0),(x1,y1))."""
    vx, vy = x1 - x0, y1 - y0
    wx, wy = px - x0, py - y0
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq == 0.0:
        return ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
    t = (wx * vx + wy * vy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = x0 + t * vx, y0 + t * vy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def _bbox_to_polygon_edge_distance(box: Bbox, poly) -> float:
    """Minimum distance from a bbox to a polygon's BOUNDARY (edge), in dbu.

    This is the distance to the well *edge* -- the physically meaningful WPE
    quantity -- not the distance to the polygon interior (which would be 0 for an
    enclosed device). We sample the bbox corners against every polygon edge and
    take the minimum. For the rectangular wells this flow draws, corner-to-edge
    sampling is exact; for a general polygon it is a close approximation that
    over-estimates by at most the corner spacing, noted here per project rules.
    """
    bx0, by0, bx1, by1 = box
    corners = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
    verts = poly.vertices
    best = float("inf")
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        for (px, py) in corners:
            d = _point_seg_distance(px, py, a.x, a.y, b.x, b.y)
            if d < best:
                best = d
    return best


# ─────────────────────────────────────────────────────────────────────────────
# layer selection + device attribution
# ─────────────────────────────────────────────────────────────────────────────

def _polys_for_layer(
    layer_polygons: Sequence[LayerPolygon],
    gds: Optional[GdsLayer],
) -> List[LayerPolygon]:
    """All LayerPolygons matching a (layer,datatype). Empty if gds is None."""
    if gds is None:
        return []
    return [
        lp for lp in layer_polygons
        if lp.layer == gds.layer and lp.datatype == gds.datatype
    ]


def _as_bbox(v) -> Optional[Bbox]:
    """Normalise a terminal geometry value to a 4-tuple bbox.

    The router records a terminal as a 4-tuple ``(x0,y0,x1,y1)`` bbox, but some
    cells record a terminal as a 2-tuple POINT ``(x,y)`` (a degenerate, zero-area
    rail). A point is normalised to ``(x,y,x,y)`` so the same distance math works.
    Returns ``None`` for an unrecognised shape (skipped, never crashes).
    """
    if isinstance(v, (tuple, list)):
        if len(v) == 4 and all(isinstance(c, (int, float)) for c in v):
            return (v[0], v[1], v[2], v[3])
        if len(v) == 2 and all(isinstance(c, (int, float)) for c in v):
            return (v[0], v[1], v[0], v[1])
    return None


def _active_bbox(term: TermDict) -> Bbox:
    """Union bbox of a device's source/drain/gate terminal bboxes.

    Accepts the router's ``placement.term`` shape (a dict of terminal-name ->
    (x0,y0,x1,y1) bbox OR (x,y) point) OR a single device active bbox/point passed
    as a 4- or 2-tuple. Point-shaped terminals are normalised to zero-area bboxes,
    so a real cell whose terminals are points no longer raises IndexError.
    """
    direct = _as_bbox(term)
    if direct is not None:
        return direct
    boxes = [b for b in (_as_bbox(v) for v in term.values()) if b is not None]  # type: ignore[union-attr]
    if not boxes:
        raise ValueError("device terminal geometry has no usable bbox/point values")
    return _union_bbox(boxes)


def _channel_axis(term: TermDict) -> Optional[str]:
    """Infer the source->drain (channel) axis from terminal bboxes.

    Returns ``"x"`` if source and drain are separated horizontally, ``"y"`` if
    vertically, or ``None`` when source/drain bboxes are not both present (caller
    then falls back to the longer diffusion side).
    """
    if not isinstance(term, dict):
        return None
    s = _as_bbox(term.get("s"))
    d = _as_bbox(term.get("d"))
    if s is None or d is None:
        return None
    s_cx = (s[0] + s[2]) / 2.0
    s_cy = (s[1] + s[3]) / 2.0
    d_cx = (d[0] + d[2]) / 2.0
    d_cy = (d[1] + d[3]) / 2.0
    return "x" if abs(d_cx - s_cx) >= abs(d_cy - s_cy) else "y"


def _device_diff_bbox(
    active: Bbox,
    diff_polys: Sequence[LayerPolygon],
) -> Optional[Bbox]:
    """Diffusion bbox attributed to a device = union of diff polys overlapping it.

    A diffusion polygon is "this device's" OD if it overlaps the device's active
    bbox. When no diffusion layer overlaps, fall back to the device's active bbox
    itself ONLY if there were no diffusion polygons at all for this type would be
    handled by the caller; here returning ``None`` signals "no attributable OD".
    """
    overlapping = [
        _poly_bbox(lp.polygon)
        for lp in diff_polys
        if _bboxes_overlap(active, _poly_bbox(lp.polygon))
    ]
    if not overlapping:
        return None
    return _union_bbox(overlapping)


# ─────────────────────────────────────────────────────────────────────────────
# public extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_device_environment(
    devices: Dict[str, TermDict],
    layer_polygons: Sequence[LayerPolygon],
    *,
    pdk_layers: _LayerMap,
    pmos_labels: Optional[Sequence[str]] = None,
) -> Dict[str, DeviceEnvironment]:
    """Extract the per-device geometric environment (WPE / LOD-STI / OD-spacing).

    Parameters
    ----------
    devices:
        ``{label: terminal-bbox-dict}`` in the router's ``placement.term`` shape
        (``{"s": (x0,y0,x1,y1), "d": ..., "g": ...}``), OR ``{label: active-bbox}``
        where active-bbox is a 4-tuple ``(x0,y0,x1,y1)``. Both are accepted so the
        extractor runs on the real routed cell and on a hand-built test double.
    layer_polygons:
        The flat ``LayerPolygon`` list from the lowered/extracted cell. Layer
        selection is done via ``pdk_layers`` -- NWELL for PMOS well-edge, PWELL for
        NMOS well-edge, NDIFF/PDIFF for OD geometry.
    pdk_layers:
        The PDK ``LayerMap`` (``gds_layer(LogicalLayer.X) -> Optional[GdsLayer]``).
        Keeps the extractor PDK-neutral; absent layers (``None``) make the
        dependent metric ``None`` (-> the obligation is unmeasurable, blocking).
    pmos_labels:
        Optional set of device labels that are PMOS. PMOS well-edge is measured
        against NWELL; everything else is treated as NMOS and measured against
        PWELL (the drawn p-well/p-implant edge). When ``None`` (default), a device
        is classed PMOS iff its active bbox overlaps a PDIFF/NWELL polygon and not
        an NDIFF polygon -- a geometry-only heuristic that needs no external role
        map. Pass ``pmos_labels`` explicitly when the role is already known.

    Returns
    -------
    ``{label: DeviceEnvironment}`` -- one entry per device label.

    Notes
    -----
    ``od_length`` is the diffusion extent along the inferred source->drain axis;
    when source/drain terminal bboxes are not both present the longer diffusion
    side is used (documented fallback). Each metric is ``None`` rather than a
    fabricated value when its layer is absent, so an unmeasurable environment
    blocks signoff instead of silently passing.
    """
    nwell_gds = pdk_layers.gds_layer(LogicalLayer.NWELL)
    pwell_gds = pdk_layers.gds_layer(LogicalLayer.PWELL)
    ndiff_gds = pdk_layers.gds_layer(LogicalLayer.NDIFF)
    pdiff_gds = pdk_layers.gds_layer(LogicalLayer.PDIFF)

    nwell_polys = _polys_for_layer(layer_polygons, nwell_gds)
    pwell_polys = _polys_for_layer(layer_polygons, pwell_gds)
    ndiff_polys = _polys_for_layer(layer_polygons, ndiff_gds)
    pdiff_polys = _polys_for_layer(layer_polygons, pdiff_gds)
    # NDIFF/PDIFF may resolve to the same (layer,datatype) on a single-COMP PDK
    # (gf180: both 22/0). Union the two without double counting so OD attribution
    # sees every diffusion island regardless of which logical name carries it.
    seen = set()
    all_diff_polys: List[LayerPolygon] = []
    for lp in list(ndiff_polys) + list(pdiff_polys):
        key = (id(lp),)
        if key in seen:
            continue
        seen.add(key)
        all_diff_polys.append(lp)

    pmos_set = set(pmos_labels) if pmos_labels is not None else None

    # ---- precompute each device's active bbox + attributed diffusion bbox ----
    active_bbox: Dict[str, Bbox] = {}
    diff_bbox: Dict[str, Optional[Bbox]] = {}
    for label, term in devices.items():
        active = _active_bbox(term)
        active_bbox[label] = active
        diff_bbox[label] = _device_diff_bbox(active, all_diff_polys)

    def _is_pmos(label: str) -> bool:
        if pmos_set is not None:
            return label in pmos_set
        # geometry-only heuristic: PMOS sits inside an nwell.
        active = active_bbox[label]
        for lp in nwell_polys:
            if _bboxes_overlap(active, _poly_bbox(lp.polygon)):
                return True
        return False

    out: Dict[str, DeviceEnvironment] = {}
    for label, term in devices.items():
        active = active_bbox[label]
        my_diff = diff_bbox[label]

        # ---- well_edge_distance (WPE) ----
        # PMOS -> nearest NWELL edge; NMOS -> nearest PWELL edge.
        # Measure from the device's diffusion region (active bbox) to the well
        # boundary. None when the relevant well layer has no polygon.
        well_polys = nwell_polys if _is_pmos(label) else pwell_polys
        if well_polys:
            well_edge = min(
                _bbox_to_polygon_edge_distance(active, lp.polygon)
                for lp in well_polys
            )
        else:
            well_edge = None

        # ---- od_length (LOD / STI stress) ----
        # The device's OD extent along the source->drain axis, measured from the
        # device's OWN active bbox (terminal-union), NOT from the attributed
        # diffusion polygon. This is robust to SHARED / abutted diffusion: an
        # interdigitated or common-centroid pair shares one diffusion island, so
        # attributing the shared polygon's full union to each device gives both an
        # identical, inflated od_length (the bug these intents would otherwise hit
        # on exactly the matched pairs they target). The per-device active bbox is
        # correct per device. Diffusion presence (my_diff) still gates measurability:
        # None when no diffusion layer is attributable -> obligation unmeasurable.
        if my_diff is not None:
            axis = _channel_axis(term)
            dx = active[2] - active[0]
            dy = active[3] - active[1]
            if axis == "x":
                od_length: Optional[float] = float(dx)
            elif axis == "y":
                od_length = float(dy)
            else:
                # fallback: no s/d terminals -> use the longer active side.
                od_length = float(max(dx, dy))
        else:
            od_length = None

        out[label] = DeviceEnvironment(
            well_edge_distance=well_edge,
            od_length=od_length,
            od_neighbor_spacing=None,  # filled in below once all diff bboxes known
        )

    # ---- od_neighbor_spacing (OD-spacing stress) ----
    # min gap from this device's OWN active bbox to the nearest OTHER device's
    # active bbox. Per-device (robust to shared diffusion, same reason as
    # od_length above); diffusion presence still gates measurability.
    labels = list(devices.keys())
    for label in labels:
        my_diff = diff_bbox[label]
        if my_diff is None:
            continue  # leave None: no attributable OD for this device
        my_active = active_bbox[label]
        best = None
        for other in labels:
            if other == label:
                continue
            if diff_bbox[other] is None:
                continue
            gap = _bbox_gap(my_active, active_bbox[other])
            if best is None or gap < best:
                best = gap
        # best stays None if no other device has an attributable diffusion.
        prev = out[label]
        out[label] = DeviceEnvironment(
            well_edge_distance=prev.well_edge_distance,
            od_length=prev.od_length,
            od_neighbor_spacing=best,
        )

    return out
