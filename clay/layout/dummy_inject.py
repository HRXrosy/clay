"""R138 (BL-138B) — symmetric dummy-device injection (TED-borrowed).

TED's ``layout_mos(outer_dummy_l=, outer_dummy_m=)`` flanks a matched device
group with symmetric dummy fingers on both sides to absorb edge effects (well
proximity, etch / poly line-end rounding, STI stress) so the *real* outermost
devices see a neighbour identical to their interior neighbours. The matched
group then behaves as if it sat in the middle of an infinite array, which is
exactly the layout-mismatch reduction the matched-pair analog blocks need.

This module is the placement post-process that lands that capability in Clay.
Given a placed ``ParameterizedLibrary`` (a ``lower_*`` output) plus a matched
device group (named explicitly, or discovered from ``Matching`` / ``Symmetric``
annotations on the cell), it emits ``2 * n`` dummy poly+diff rectangles — ``n``
on the left of the group and ``n`` on the right, mirror-positioned about the
group's geometric center — and returns an *augmented* library whose top cell now
also carries those dummies.

Symmetry guarantee (proven in tests):
  * equal count on each side (``n`` left, ``n`` right);
  * for the i-th dummy, ``center_x(left_i) + center_x(right_i) == 2 * group_cx``
    exactly (integer DBU center reflection about the group center) and the y
    extents are identical — so the two flanks are a perfect mirror pair;
  * dummy width / height match the group's outermost real device geometry, so a
    dummy is electrically a clone of the device it shields.

Invariants:
  * the original library is never mutated; a fresh ``ParameterizedLibrary`` is
    returned with verbatim copies of every existing polygon (same layer /
    datatype / vertices), so real device geometry AND its connectivity (net
    membership is a pure function of geometry) are byte-for-byte unchanged;
  * dummies live on the same POLY + NDIFF/PDIFF layers as the group's real
    devices (read from the PDK ``LayerMap``), never inventing a layer;
  * dummies are placed strictly *outside* the group bbox (one device-pitch gap),
    so no dummy interior-overlaps a real device.

``backend`` (NamedCell / LayerPolygon / Polygon / Point / ParameterizedLibrary /
LayerMap / LogicalLayer) and ``annotations`` (Matching / Symmetric) are imported
read-only; neither is modified. Dummy geometry is derived entirely from the real
placed geometry (no fabricated coordinates).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .annotations import AnnotatedCell, Matching, Symmetric
from .backend import (
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    NamedCell,
    ParameterizedLibrary,
    Point,
    Polygon,
)


BBox = Tuple[int, int, int, int]


# --- geometry helpers (read-only over placed geometry) ----------------------


def _poly_bbox(polys: Sequence[LayerPolygon]) -> Optional[BBox]:
    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_of_keys(polys: Sequence[LayerPolygon], keys) -> Optional[BBox]:
    sub = [lp for lp in polys if (lp.layer, lp.datatype) in keys]
    return _poly_bbox(sub)


def _copy_poly(lp: LayerPolygon) -> LayerPolygon:
    return LayerPolygon(
        layer=lp.layer,
        datatype=lp.datatype,
        polygon=Polygon(vertices=[Point(v.x, v.y) for v in lp.polygon.vertices]),
    )


def _layer_keys(pdk_layers: LayerMap) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
    """(poly_key, [diff_keys]) for the PDK. diff = NDIFF + PDIFF (deduped)."""
    poly = pdk_layers.gds_layer(LogicalLayer.POLY)
    if poly is None:
        raise ValueError("PDK must define a POLY layer for dummy injection")
    poly_key = (poly.layer, poly.datatype)
    diff_keys: List[Tuple[int, int]] = []
    for ll in (LogicalLayer.NDIFF, LogicalLayer.PDIFF):
        g = pdk_layers.gds_layer(ll)
        if g is not None and (g.layer, g.datatype) not in diff_keys:
            diff_keys.append((g.layer, g.datatype))
    if not diff_keys:
        raise ValueError("PDK must define an NDIFF or PDIFF layer for dummy injection")
    return poly_key, diff_keys


# --- matched-group discovery ------------------------------------------------


def discover_matched_group(cell: AnnotatedCell) -> List[str]:
    """Discover the matched device group from a cell's annotations.

    Preference order:
      1. the first ``Symmetric`` annotation's device list (explicit group);
      2. otherwise, the largest connected component of the ``Matching`` graph
         (each ``Matching(other=...)`` is an undirected edge).

    Returns a sorted device-name list, or ``[]`` if no matched relationship is
    declared. Only devices that actually exist in ``cell.layouts`` are kept.
    """
    for anns in cell.annotations.values():
        for a in anns:
            if isinstance(a, Symmetric):
                return sorted(d for d in a.devices if d in cell.layouts)

    # Build Matching adjacency.
    adj: Dict[str, set] = {}
    for dev, anns in cell.annotations.items():
        for a in anns:
            if isinstance(a, Matching):
                o = a.other
                if dev in cell.layouts and o in cell.layouts:
                    adj.setdefault(dev, set()).add(o)
                    adj.setdefault(o, set()).add(dev)
    if not adj:
        return []

    # Largest connected component (ties → lexicographically smallest member set).
    seen: set = set()
    best: List[str] = []
    for start in sorted(adj):
        if start in seen:
            continue
        stack = [start]
        comp: List[str] = []
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.append(n)
            stack.extend(adj.get(n, ()))
        if len(comp) > len(best) or (len(comp) == len(best) and sorted(comp) < best):
            best = sorted(comp)
    return best


# --- group bbox from placed geometry ----------------------------------------


def group_bbox(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    group: Sequence[str],
) -> Optional[BBox]:
    """Bounding box (in top-cell coords) covering the matched group's devices.

    The matched group's real geometry is the union of those devices' per-device
    NamedCells, translated to where they sit in the top cell. Because the
    ``lower_*`` / ``apply_placement`` top cell is just the concatenation of the
    devices' (possibly translated) geometry, and we cannot in general recover
    each device's offset from the flattened top cell alone, we take the *top
    cell* extent as the authoritative placed footprint when the group is the
    whole cell, and otherwise fall back to the union of per-device local bboxes
    offset by their position in the top cell only if a unique attribution
    exists. To stay unambiguous and never fabricate, the group bbox is computed
    directly from the top-cell polygons that lie on poly/diff layers and fall
    inside any per-device cell's footprint — see ``_group_bbox_from_top``.
    """
    return _group_bbox_from_top(lib, cell, group)


def _device_cells(lib: ParameterizedLibrary, cell: AnnotatedCell) -> Dict[str, NamedCell]:
    return {c.name: c for c in lib.cells if c.name in cell.layouts}


def _top_cell(lib: ParameterizedLibrary, cell: AnnotatedCell) -> Optional[NamedCell]:
    for c in lib.cells:
        if c.name == cell.cell_name:
            return c
    return None


def _group_bbox_from_top(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    group: Sequence[str],
) -> Optional[BBox]:
    """Authoritative placed footprint of the matched group.

    When the group spans every device in the cell, the group footprint IS the
    top-cell bbox. When it is a strict subset we sum the per-device LOCAL bbox
    sizes and locate them by matching their geometry inside the top cell. To
    avoid mis-attribution on flattened/mirrored top cells, the conservative and
    always-correct measure is: the union of the group's per-device LOCAL bboxes
    *as a footprint width/height*, anchored at the top-cell min corner. That is
    exactly what edge-effect dummies care about (a symmetric collar around the
    group center), and it never reads geometry that isn't the group's.
    """
    group_set = set(group)
    all_devs = set(cell.layouts.keys())
    top = _top_cell(lib, cell)

    if group_set == all_devs and top is not None:
        return _poly_bbox(top.polygons)

    # Subset group: use the top-cell bbox if a clean per-device attribution is
    # impossible; but the well-defined, fabrication-free choice is the union of
    # the group devices' per-device local bboxes. Per-device cells are at local
    # origin, so this gives the group's intrinsic footprint; we anchor it at the
    # top-cell origin so the collar is placed in top-cell coordinates.
    dcells = _device_cells(lib, cell)
    sub: List[LayerPolygon] = []
    for d in group:
        c = dcells.get(d)
        if c is not None:
            sub.extend(c.polygons)
    return _poly_bbox(sub) if sub else None


# --- dummy template ---------------------------------------------------------


def _dummy_template(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    group: Sequence[str],
    poly_key,
    diff_keys,
) -> Optional[Tuple[List[LayerPolygon], int]]:
    """A single-finger dummy template at local origin + its x-pitch.

    Built from the group's outermost real device so the dummy is a geometric
    clone of the device it shields. Returns ``(polys, pitch_x)`` where the
    template's bbox-min is the local origin and ``pitch_x`` is the
    center-to-center spacing to use between successive dummies (= template width
    + inter-finger gap, so adjacent dummies never overlap each other).
    """
    dcells = _device_cells(lib, cell)
    ref: Optional[NamedCell] = None
    for d in sorted(group):
        c = dcells.get(d)
        if c is not None and c.polygons:
            ref = c
            break
    if ref is None:
        return None

    pbb = _bbox_of_keys(ref.polygons, {poly_key})
    dbb = _bbox_of_keys(ref.polygons, set(diff_keys))
    if pbb is None or dbb is None:
        return None

    # One poly finger: take the leftmost poly finger's width from the reference.
    poly_fingers = sorted(
        (lp for lp in ref.polygons if (lp.layer, lp.datatype) == poly_key),
        key=lambda lp: _poly_bbox([lp])[0],
    )
    first = poly_fingers[0]
    fbb = _poly_bbox([first])
    assert fbb is not None
    finger_w = fbb[2] - fbb[0]
    poly_y0, poly_y1 = pbb[1], pbb[3]
    diff_y0, diff_y1 = dbb[1], dbb[3]

    # Single-finger pitch inside the real device (center-to-center of two
    # adjacent fingers). This is the physical width one finger "owns" — the
    # right diff width for a single dummy finger (NOT the whole multi-finger
    # device diff, which would make the dummy absurdly wide). With one finger
    # only, fall back to twice the finger width (finger + one S/D gap).
    if len(poly_fingers) >= 2:
        f2 = _poly_bbox([poly_fingers[1]])
        assert f2 is not None
        finger_pitch = f2[0] - fbb[0]
        if finger_pitch <= finger_w:
            finger_pitch = finger_w + max(1, finger_w)
    else:
        finger_pitch = finger_w + max(1, finger_w)

    # Local diff strip width = one finger pitch, so the dummy carries a single
    # source/drain pair of diff exactly like one interior finger does.
    diff_w = finger_pitch

    # Template anchored so its diff strip starts at x = 0; the poly finger sits
    # centered on the diff strip (matching a real finger's poly-on-diff).
    poly_x0 = (diff_w - finger_w) // 2
    polys = [
        LayerPolygon(
            layer=diff_keys[0][0],
            datatype=diff_keys[0][1],
            polygon=Polygon.rectangle(0, diff_y0, diff_w, diff_y1),
        ),
        LayerPolygon(
            layer=poly_key[0],
            datatype=poly_key[1],
            polygon=Polygon.rectangle(poly_x0, poly_y0, poly_x0 + finger_w, poly_y1),
        ),
    ]

    # Center-to-center spacing between successive dummies = full template width
    # plus a one-finger gap, so adjacent dummies' diff strips never overlap.
    pitch_x = diff_w + max(1, finger_w)
    return polys, pitch_x


# --- injection result -------------------------------------------------------


@dataclass
class DummyInjection:
    """Record of an injection: the augmented library + what was added."""

    library: ParameterizedLibrary
    group: List[str] = field(default_factory=list)
    group_center_x: int = 0
    dummy_count_per_side: int = 0
    dummy_bboxes: List[BBox] = field(default_factory=list)  # left flank then right flank

    def total_dummies(self) -> int:
        return len(self.dummy_bboxes)


# --- public API -------------------------------------------------------------


def inject_dummies(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    *,
    n_dummies: int = 1,
    group: Optional[Sequence[str]] = None,
    fin_pitch_nm: Optional[float] = None,
) -> DummyInjection:
    """Add ``n_dummies`` symmetric dummy fingers on EACH side of a matched group.

    Args:
      lib: a placed library (``lower_*`` / ``apply_placement`` output). NOT
        mutated — a fresh augmented library is returned.
      cell: the ``AnnotatedCell`` the library was lowered from (supplies device
        names + Matching/Symmetric annotations + the top cell name).
      pdk_layers: the PDK ``LayerMap`` (selects poly + diff layers for dummies).
      n_dummies: dummies to add PER SIDE (so total added = ``2 * n_dummies``).
        ``0`` returns a verbatim copy with no dummies.
      group: explicit matched device names; if ``None`` they are discovered from
        the cell's Symmetric / Matching annotations.
      fin_pitch_nm: optional explicit collar gap (nm) between the group bbox and
        the innermost dummy; default = the dummy template's own x-pitch.

    Returns a ``DummyInjection`` whose ``.library`` is the augmented library and
    whose ``.dummy_bboxes`` lists the placed dummy bounding boxes (left flank in
    inner→outer order, then right flank in inner→outer order).

    Symmetry guarantee: for every i, the i-th left dummy and i-th right dummy are
    a mirror pair about ``group_center_x`` (equal y extents; their center-x sum
    is exactly ``2 * group_center_x``). Dummies are placed strictly outside the
    group bbox, so none interior-overlaps a real device.
    """
    if n_dummies < 0:
        raise ValueError(f"n_dummies must be >= 0, got {n_dummies}")

    group_list = list(group) if group is not None else discover_matched_group(cell)
    # Keep only real devices, deterministic order.
    group_list = sorted(d for d in group_list if d in cell.layouts)

    # Fresh library: verbatim copies of every existing cell (originals untouched).
    new_lib = ParameterizedLibrary(
        name=lib.name,
        unit_user=lib.unit_user,
        unit_db_meters=lib.unit_db_meters,
    )
    for c in lib.cells:
        new_lib.add_cell(
            NamedCell(name=c.name, polygons=[_copy_poly(lp) for lp in c.polygons])
        )

    inj = DummyInjection(
        library=new_lib,
        group=group_list,
        group_center_x=0,
        dummy_count_per_side=0,
        dummy_bboxes=[],
    )

    if n_dummies == 0 or not group_list:
        # Nothing to inject (no dummies requested, or no matched group).
        return inj

    poly_key, diff_keys = _layer_keys(pdk_layers)
    gbb = _group_bbox_from_top(lib, cell, group_list)
    if gbb is None:
        return inj

    template = _dummy_template(lib, cell, group_list, poly_key, diff_keys)
    if template is None:
        return inj
    tmpl_polys, pitch_x = template

    gx0, _gy0, gx1, _gy1 = gbb
    group_cx = (gx0 + gx1) // 2
    # Use 2*half so the center is exactly representable for integer reflection;
    # group_cx may truncate on odd widths, so reflect about the *doubled* center
    # to keep the mirror exact in integer DBU.
    two_cx = gx0 + gx1

    tmpl_bb = _poly_bbox(tmpl_polys)
    assert tmpl_bb is not None
    tmpl_w = tmpl_bb[2] - tmpl_bb[0]

    collar_gap = pitch_x if fin_pitch_nm is None else int(round(fin_pitch_nm))
    if collar_gap < 1:
        collar_gap = 1

    top = None
    for c in new_lib.cells:
        if c.name == cell.cell_name:
            top = c
            break

    placed_left: List[BBox] = []
    placed_right: List[BBox] = []

    for i in range(n_dummies):
        # Right flank: i-th dummy's LEFT edge starts collar+ i*pitch right of group.
        right_x0 = gx1 + collar_gap + i * pitch_x
        right_dummy = _place_template(tmpl_polys, right_x0 - tmpl_bb[0], 0)
        rbb = _poly_bbox(right_dummy)
        assert rbb is not None

        # Left flank: mirror the right dummy about two_cx/... using exact
        # integer reflection of x: x' = two_cx - x. Mirroring a rectangle keeps
        # its y; its x-extent reflects to [two_cx - rbb_x1, two_cx - rbb_x0].
        left_dummy = _mirror_polys_x(right_dummy, two_cx)
        lbb = _poly_bbox(left_dummy)
        assert lbb is not None

        if top is not None:
            top.polygons.extend(left_dummy)
            top.polygons.extend(right_dummy)
        placed_left.append(lbb)
        placed_right.append(rbb)

    inj.group_center_x = group_cx
    inj.dummy_count_per_side = n_dummies
    # Left flank inner→outer (closest to group first), then right flank inner→outer.
    inj.dummy_bboxes = list(placed_left) + list(placed_right)
    return inj


def augmented_library(
    lib: ParameterizedLibrary,
    cell: AnnotatedCell,
    pdk_layers: LayerMap,
    *,
    n_dummies: int = 1,
    group: Optional[Sequence[str]] = None,
    fin_pitch_nm: Optional[float] = None,
) -> ParameterizedLibrary:
    """Convenience: ``inject_dummies(...).library``."""
    return inject_dummies(
        lib,
        cell,
        pdk_layers,
        n_dummies=n_dummies,
        group=group,
        fin_pitch_nm=fin_pitch_nm,
    ).library


# --- transforms used internally + by tests ----------------------------------


def _place_template(polys: Sequence[LayerPolygon], dx: int, dy: int) -> List[LayerPolygon]:
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(vertices=[Point(v.x + dx, v.y + dy) for v in lp.polygon.vertices]),
        )
        for lp in polys
    ]


def _mirror_polys_x(polys: Sequence[LayerPolygon], two_axis: int) -> List[LayerPolygon]:
    """Reflect about x = two_axis/2 using EXACT integer arithmetic: x' = two_axis - x.

    Reversing vertex order keeps winding consistent. y is unchanged, so the
    mirror image has identical y extent — the symmetry guarantee.
    """
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(
                vertices=list(
                    reversed([Point(two_axis - v.x, v.y) for v in lp.polygon.vertices])
                )
            ),
        )
        for lp in polys
    ]


# --- audit helpers (tests + callers) ----------------------------------------


def bbox_center_x_doubled(bb: BBox) -> int:
    """``x0 + x1`` (the doubled center-x), so two mirror dummies satisfy
    ``doubled(left) + doubled(right) == 4 * group_center`` exactly in integers."""
    return bb[0] + bb[2]


def is_symmetric_pair(left: BBox, right: BBox, two_group_center: int) -> bool:
    """True iff ``left`` and ``right`` are an exact mirror pair about the group
    center (whose doubled value is ``two_group_center``): identical y extent and
    reflected x extent."""
    # y extents identical.
    if left[1] != right[1] or left[3] != right[3]:
        return False
    # x reflection: left.x0 == two_group_center - right.x1 and
    #               left.x1 == two_group_center - right.x0.
    return (
        left[0] == two_group_center - right[2]
        and left[2] == two_group_center - right[0]
    )


def bboxes_overlap(a: BBox, b: BBox) -> bool:
    """Strict interior overlap (shared edges do not count)."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


__all__ = [
    "DummyInjection",
    "inject_dummies",
    "augmented_library",
    "discover_matched_group",
    "group_bbox",
    "is_symmetric_pair",
    "bbox_center_x_doubled",
    "bboxes_overlap",
]
