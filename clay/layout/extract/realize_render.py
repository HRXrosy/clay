"""KEYSTONE: realize a leakage-free intent in REAL device geometry + render it.

WHAT THIS MODULE IS
===================
The leakage-free reproduction experiment has two halves that must stay separate:

  * POSITION  -- where each device sits. Solved from abstract relations / order by
    :func:`clay.layout.extract.dcsolve.solve_compact` (external_pdk) or
    :func:`clay.layout.extract.migrate_smic.migrate` (external_pdk). NEVER read from a
    stored reference coordinate.
  * GEOMETRY  -- the polygons that make up a device of a given (kind, W, L, M).
    This is process knowledge, not a floorplan fact, so it may come from the PDK:
    either the VENDOR GDS library (the reference's own resolved external_pdk PCells,
    captured by :func:`clay.layout.extract.library_device.extract_device_geometry`)
    or, for any device the vendor library cannot supply (a band-only / MOScap
    region with no captured finger instance, or a non-external_pdk target PDK), synthesised
    from parameters by :func:`clay.layout.device_cell.build_device`.

This module is the join: it gives every device a CLEAN geometry stamp (vendor where
available, build_device otherwise), then stamps those stamps at the solved positions
and renders the result.

THE DISCIPLINE (enforced here)
==============================
  * Geometry NEVER carries a position -- every stamp is centroid-relative um, and a
    build_device stamp is recentred on its own centroid (the build_device cell is
    built at an arbitrary local origin; we subtract its centroid so it is a pure
    position-free stamp, exactly like the vendor capture).
  * We do NOT fabricate geometry to make a device "look clean". A device whose
    geometry genuinely cannot be produced (a passive the parametric builder does not
    model, with no vendor capture) is left out and reported honestly -- never faked.
  * Positions come from the caller (the relation realize), never from this module.

The :class:`~clay.layout.extract.library_device.CellGeometry` shape is reused as the
single geometry currency so :func:`clay.layout.extract.library_device.emit_library_gds`
stamps a build_device stamp and a vendor stamp identically.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

from clay.layout.extract.library_device import CellGeometry
from clay.layout.placed_geometry import placed_point_nm

_XY = Tuple[float, float]
_Ring = List[Tuple[float, float]]
_LayerKey = Tuple[int, int]

# Vertical gap (um) between stacked folded rows (matches fold_intent._ROW_GAP_UM).
_ROW_GAP_UM: float = 1.0

# kinds device_cell.build_device can synthesise (MOS only). cap/res are NOT
# parametrically modelled by build_device -- they have no parametric fallback here.
_BUILDABLE_KINDS = frozenset({"nmos", "pmos"})

# The DESIGN wide-flat unit-transistor band-height target (um). A STYLE statistic,
# never a reference dimension: the reference layout draws its MOS as wide-flat
# unit-transistor arrays (per-finger ~1.4um + the PDK end margin ~2.5um), so the
# folded stamp's band height converges here. With the serpentine 2D-compact fold
# this is the height the per-band channel converges toward (subject to the hard
# aspect<=6 footprint bound). 4.0um is the same band target the device footprint
# contract derives (tests/test_device_footprint_contract.py::_design_row_height_target);
# it is a layout-style choice, not a measured reference value.
_DESIGN_BAND_HEIGHT_UM: float = 4.0


def design_row_height_target_um(tech_id: str = "") -> float:
    """The DESIGN wide-flat unit-transistor band-height fold target (um).

    A pure STYLE statistic (the reference unit-transistor band style), NOT a
    reference coordinate/dimension -- it is the band height a serpentine 2D-compact
    fold converges its per-band channel toward so the realized device stamps go
    wide-flat (matching the reference's aspect discipline) instead of tall-thin.
    ``tech_id`` is accepted for a future per-PDK band style; the current value is
    PDK-independent (the 4.0um band is dominated by the ~2.5um PDK end margin, which
    is within 0.06um across external_pdk/gf180/external_pdk). The hard footprint bound (aspect<=6)
    always dominates this soft target."""
    return _DESIGN_BAND_HEIGHT_UM


# --------------------------------------------------------------------------- #
# kind normalisation (netlist 'pmos'/'nmos'/'cap'/'res' -> canonical)
# --------------------------------------------------------------------------- #
def _norm_kind(k: str) -> str:
    k = (k or "").lower()
    if k.startswith("pmos"):
        return "pmos"
    if k.startswith("nmos"):
        return "nmos"
    if k.startswith("cap") or k == "c":
        return "cap"
    if k.startswith("res") or k == "r":
        return "res"
    return k


# device-nets convention (source/drain/gate/body) -> build_device convention
# (s/d/g/b). Used to decide the SELF-BULK band-tap (source net == body net) on
# the serpentine fold; nothing else in the geometry depends on net names
# (the fold is net-independent), so this is leakage-free design data only.
_NET_ROLE_TO_TERM = {"source": "s", "drain": "d", "gate": "g", "body": "b"}


def _build_device_nets(
    nets: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Map a device's source/drain/gate/body nets (ROLE-keyed: source/drain/
    gate/body) to the build_device s/d/g/b terminal convention. ``None`` (no
    design nets) stays ``None``.

    LOUD by contract: a dict carrying any key outside the known ROLE set
    (e.g. a TERM-keyed s/d/g/b dict) raises instead of silently mapping to
    nothing. A silent drop here is exactly the F2 incident -- the route_cell
    production path once passed a TERM-keyed dict, every ``nets.get(role)``
    lookup missed, and the declared body net was erased with no signal."""
    if not nets:
        return None
    unknown = set(nets) - set(_NET_ROLE_TO_TERM)
    if unknown:
        raise ValueError(
            f"build_device_geometry nets must be ROLE-keyed "
            f"({sorted(_NET_ROLE_TO_TERM)}); got unknown keys {sorted(unknown)} "
            f"-- a TERM-keyed dict here silently erased declared body nets "
            f"(the F2 incident)."
        )
    out = {_NET_ROLE_TO_TERM[k]: v for k, v in nets.items() if v}
    return out or None


def _device_nets_for(intent, name: str) -> Optional[Dict[str, str]]:
    """The source/drain/gate/body nets for device ``name`` in ``intent`` (None
    if the device is unknown or carries no nets). Design data only -- used to
    decide the self-bulk band-tap on the serpentine fold."""
    for d in getattr(intent, "devices", []) or []:
        if str(getattr(d, "name", "")) == name:
            nd = getattr(d, "nets", {}) or {}
            return dict(nd) if nd else None
    return None


# --------------------------------------------------------------------------- #
# 1. build_device -> CellGeometry adapter
# --------------------------------------------------------------------------- #
# Footprint contract bound: a device enters the large-strip fold when its
# single-row aspect exceeds 3:1, so the realized 2D fold must return inside that
# same bound.  A looser 6:1 downstream threshold let an exact-width split tip
# from 40 rows to a still-strip-like 20-row result (5.98:1).  Keep the gate and
# the materialized result on one compactness obligation.
LARGE_M_FOLD_ASPECT: float = 3.0
_MAX_FOLD_ASPECT: float = LARGE_M_FOLD_ASPECT


def _choose_fold(
    M: int,
    *,
    finger_pitch_um: float,
    finger0_w_um: float,
    row_h_um: float,
    max_rows: int,
    row_height_target_um: Optional[float] = None,
    serpentine: bool = False,
    band_w_um: Optional[float] = None,
    band_margin_um: Optional[float] = None,
    max_aspect: float = _MAX_FOLD_ASPECT,
) -> List[int]:
    """Choose a multi-row fold of ``M`` fingers as a list of per-row finger counts.

    Each row is one ``build_device`` strip of ``nf`` fingers; the strip is
    ``row_h_um`` tall (the finger height = channel width + fixed end-margins) and
    ``finger0_w_um + (nf-1) * finger_pitch_um`` wide. A single row of all M fingers
    is a wide strip; stacking rows trades width for height. We pick ``n_rows`` in
    ``[1, min(max_rows, M)]`` whose tiled cell aspect (width / height) is closest to
    1 (a square block) -- a generic compactness rule, NEVER a reference coordinate.

    ``row_height_target_um`` (opt-in; ``None`` keeps the squareness rule
    byte-identical): a DESIGN-chosen stamp-height target. The cost becomes
    ``|cell_height - target|`` so the fold stacks rows toward the target
    height instead of toward a square -- the caller's row-uniformity goal
    (same-row stamps converge on one height; rows go wide-flat). The target
    is a design statistic supplied by the caller, never a reference value.

    ``serpentine`` (2D-COMPACT row-bounding; the footprint-contract fix): the fold
    is built as ONE shared-diffusion serpentine device
    (:func:`clay.layout.device_cell.build_device` with ``fold_rows=R``), NOT R
    gapped strips. Two consequences for the choice:

      * EVEN split: ``R`` MUST DIVIDE ``M`` (the serpentine deals ``M/R`` fingers to
        each of R abutting bands; an uneven split is not a rectangular conserved
        array), so candidate row counts are the DIVISORS of ``M``, not ``1..max_rows``.
      * SERPENTINE height model: the bands share ONE continuous COMP (no inter-row
        gap, one end margin), so ``cell_h = R * band_w_um + band_margin_um`` (the
        per-band channel width ``band_w_um`` plus the fixed device margin
        ``band_margin_um`` = ``row_h_um - band_w_um``), measured exact from real
        strips. The width depends only on the per-band finger count ``M/R``:
        ``cell_w = finger0_w_um + (M/R - 1) * finger_pitch_um``.

    The 2D-COMPACT rule (the root cure of the width blow-up): pick the FEWEST rows R
    whose tiled-cell ASPECT <= ``max_aspect`` -- so a W-split that fans M into many
    unit fingers does NOT collapse into one monster-wide row; rows are ADDED only as
    needed to bound the WIDTH/aspect. Among the aspect-legal divisors the fewest-rows
    one wins (the simplest cell that is 2D-compact); if NO divisor is aspect-legal the
    MOST rows (narrowest) divisor is taken (the best achievable). The height target is
    a secondary tie-break only -- aspect-bounding dominates (a height-only cost is what
    kept ONE row and blew the width up).

    The width/height model is EXACT (measured from real ``build_device`` strips, see
    :func:`build_device_geometry`), so the chosen fold is the genuinely-most-compact
    one available, not a rough guess. For the NON-serpentine path fingers are
    distributed as evenly as possible (``ceil`` rows first) so ``sum(per_row) == M``
    exactly; the serpentine path returns ``[M//R] * R`` (uniform by construction).
    """
    M = max(int(M), 1)

    if serpentine:
        # candidate row counts: the DIVISORS of M (an even, conserved split).
        bw = float(band_w_um) if band_w_um is not None else float(row_h_um)
        bm = (
            float(band_margin_um)
            if band_margin_um is not None
            else max(float(row_h_um) - bw, 0.0)
        )
        divisors = [d for d in range(1, M + 1) if M % d == 0]
        # 2D-COMPACT: fewest rows whose aspect <= max_aspect bound the WIDTH.
        legal: List[int] = []
        narrowest: Optional[Tuple[float, int]] = None  # (aspect, R) for fallback
        for R in divisors:
            nf_row = M // R
            cell_w = finger0_w_um + max(nf_row - 1, 0) * finger_pitch_um
            cell_h = R * bw + bm
            aspect = max(cell_w, cell_h) / max(min(cell_w, cell_h), 1e-9)
            if narrowest is None or aspect < narrowest[0] - 1e-12:
                narrowest = (aspect, R)
            if aspect <= float(max_aspect) + 1e-9:
                legal.append(R)
        if legal:
            # Among the aspect-legal divisors (all 2D-compact by construction), pick
            # the one whose stamp HEIGHT is CLOSEST to the design target (the
            # row-uniformity goal -- same-W devices converge on one band height, so
            # the placement outline goes wide-flat); fewest rows tie-breaks. Aspect
            # legality is the hard gate (the footprint contract); the height target
            # only chooses AMONG the compact options -- it can never re-introduce the
            # monster-wide one-row blow-up (that divisor is aspect-illegal, excluded).
            if row_height_target_um is not None:
                target = float(row_height_target_um)
                R = min(
                    legal,
                    key=lambda r: (abs((r * bw + bm) - target), r),
                )
            else:
                R = min(legal)
        else:
            # no aspect-legal divisor: take the narrowest available (most rows).
            R = narrowest[1] if narrowest is not None else 1
        return [M // R] * R

    max_rows = max(1, min(int(max_rows), M))

    best_rows = 1
    best_cost = None
    for n_rows in range(1, max_rows + 1):
        nf = math.ceil(M / n_rows)
        cell_w = finger0_w_um + max(nf - 1, 0) * finger_pitch_um
        cell_h = n_rows * row_h_um + (n_rows - 1) * _ROW_GAP_UM
        if row_height_target_um is not None:
            cost = abs(cell_h - float(row_height_target_um))
        else:
            aspect = cell_w / max(cell_h, 1e-9)
            cost = abs(math.log(max(aspect, 1e-9)))
        # tie-break: fewer rows (simpler cell) when costs are within rounding.
        cost += 0.01 * n_rows
        if best_cost is None or cost < best_cost - 1e-12:
            best_cost = cost
            best_rows = n_rows

    n_rows = best_rows
    return _rows_to_per_row(M, n_rows)


def _choose_finger_split(
    W_um: float,
    margin_um: float,
    target_um: float,
    ch_min_um: float,
) -> int:
    """The unit-finger split ``k`` that brings a single row's height
    (``W_um / k + margin_um``) closest to the DESIGN target.

    Each netlist finger of channel width ``W_um`` becomes ``k`` unit fingers
    of width ``W_um / k`` -- the reference-style wide-flat unit-transistor
    array shape.  The materializer ultimately draws on an integer-nm grid, so
    only divisors of the already-snapped width are legal.  This preserves total
    channel width EXACTLY after snapping instead of amplifying a sub-nm
    per-finger rounding error by a large multiplicity.
    ``k`` is also clamped so the unit width never drops below the PDK
    ``channel_min_width`` (``build_device`` would otherwise silently clamp a
    sub-minimum finger back UP, growing the total width -- an electrical
    change we refuse). Tie-break prefers the smaller ``k`` (fewer pieces).
    Pure arithmetic on design + PDK data; no reference value.
    """
    if W_um <= 0.0:
        return 1
    width_nm = max(1, int(round(float(W_um) * 1000.0)))
    ch_min_nm = max(1, int(round(float(ch_min_um) * 1000.0)))
    k_max = max(1, width_nm // ch_min_nm)
    best_k = 1
    best_cost = None
    for k in range(1, k_max + 1):
        if width_nm % k:
            continue
        w_unit = (width_nm // k) / 1000.0
        cost = abs((w_unit + margin_um) - target_um) + 0.01 * k
        if best_cost is None or cost < best_cost - 1e-12:
            best_cost = cost
            best_k = k
    return best_k


def _stack_strips(
    per_row,
    row_geom_fn,
    *,
    terminal_accesses_nm: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> Dict[_LayerKey, List[_Ring]]:
    """Stack ``per_row`` independent single-row ``build_device`` strips vertically
    with the fixed inter-row gap (the DEFAULT, no-target fold path; byte-identical
    to the pre-serpentine loop). Each strip ``nf`` is built by ``row_geom_fn(nf)``
    and shifted so its bottom edge sits ``_ROW_GAP_UM`` above the previous strip.
    Returns the stacked polygons in absolute device-local nm (NOT yet recentred).

    NOTE: this gapped-strip stacking FRAGMENTS the diffusion (KLayout extracts one
    MOS per strip) and is kept ONLY for the default path's byte-parity; the
    row-height-target path uses the shared-diffusion serpentine build instead.
    """
    row_gap_nm = int(round(_ROW_GAP_UM * 1000.0))
    stacked: Dict[_LayerKey, List[_Ring]] = {}
    y_cursor_nm = 0
    for nf in per_row:
        g = row_geom_fn(nf)
        e = g.extent()  # (x0, y0, x1, y1) nm
        # shift this row so its bottom edge sits at y_cursor_nm.
        dy = y_cursor_nm - e[1]
        for key, rings in _device_polys_by_layer(g).items():
            for ring in rings:
                stacked.setdefault(key, []).append([(x, y + dy) for (x, y) in ring])
        if terminal_accesses_nm is not None:
            for role, terminal in (getattr(g, "terminals", {}) or {}).items():
                tx, ty = terminal.center
                terminal_accesses_nm.setdefault(str(role), []).append(
                    (int(tx), int(ty) + int(dy))
                )
        row_h = e[3] - e[1]
        y_cursor_nm += row_h + row_gap_nm
    return stacked


def _rows_to_per_row(M: int, n_rows: int) -> List[int]:
    """Distribute ``M`` fingers into exactly ``n_rows`` rows, as evenly as possible.

    Helper for :func:`_choose_fold` once it has picked the most-compact row count:
    ``ceil`` rows take one extra finger so ``sum(per_row) == M`` exactly (the device is
    electrically unchanged).
    """
    M = max(int(M), 1)
    n_rows = max(1, min(int(n_rows), M))
    base = M // n_rows
    rem = M - base * n_rows
    per_row = [base + 1] * rem + [base] * (n_rows - rem)
    return [c for c in per_row if c > 0]


def _device_polys_by_layer(geom) -> Dict[_LayerKey, List[_Ring]]:
    """Polygons of a ``device_cell.DeviceGeom`` -> ``{(layer,dt): [ring,...]}`` (nm
    ints, absolute device-local frame). Each ring is the polygon's vertices."""
    out: Dict[_LayerKey, List[_Ring]] = {}
    for lp in geom.polygons:
        key = (lp.layer, lp.datatype)
        ring = [(v.x, v.y) for v in lp.polygon.vertices]
        if len(ring) >= 3:
            out.setdefault(key, []).append(ring)
    return out


def _physical_mos_occurrences_nm(geom, tech) -> Tuple[Tuple[int, int, str], ...]:
    """Return the evaluated channel centres of one built MOS geometry.

    Physical channels are connected poly/active intersections. A continuous
    serpentine fold has fewer, wider channels than the logical source fingers.
    Verify actual channel lengths and conserved total width, not equality of
    those two different occurrence counts.
    """

    from clay.layout.device_cell import _resolve_layer_keys

    keys = _resolve_layer_keys(tech, str(geom.kind))

    import klayout.db as kdb

    def region(key):
        result = kdb.Region()
        for row in geom.polys_on(key):
            result.insert(kdb.Polygon([
                kdb.Point(int(vertex.x), int(vertex.y))
                for vertex in row.polygon.vertices]))
        return result

    orientation = str(getattr(geom, "orient", ""))
    if orientation not in {
        "R0", "R90", "R180", "R270",
        "MR0", "MR90", "MR180", "MR270",
    }:
        raise ValueError("evaluated MOS occurrence orientation is unavailable")
    channels = (region(keys.poly) & region(keys.active)).merged()
    centers, total_width = set(), 0
    sideways = orientation in {"R90", "R270", "MR90", "MR270"}
    for channel in channels.each():
        box = channel.bbox()
        length, width = ((box.height(), box.width()) if sideways
                         else (box.width(), box.height()))
        if (channel.area() != box.area() or length != int(geom.gate_length_nm)
                or width <= 0):
            raise ValueError("evaluated MOS channel geometry differs from built W/L")
        total_width += width
        centers.add(((box.left + box.right) // 2, (box.bottom + box.top) // 2))
    expected_width = int(geom.finger_count) * int(geom.channel_width_nm)
    if not centers or total_width != expected_width:
        raise ValueError(
            "evaluated MOS channel total width differs from built finger count: "
            f"observed={total_width} expected={expected_width}"
        )
    return tuple((x, y, orientation) for x, y in sorted(centers))


# Half-nm resolution for centroid-relative stamp merging.  Integer-nm source
# artwork can acquire half-nm coordinates when recentered on an odd-sized bbox.
# Quantizing that local frame to 1 nm before its later rigid placement deforms
# legal cuts (a 220 nm square became 220 x 218 nm on external_pdk MSE).  Half-nm is the
# exact closure of integer-nm vertices under bbox-centroid subtraction; the
# final placed geometry is still quantized once to the canonical integer-nm GDS
# currency by ``placed_point_nm``.
_MERGE_DBU: float = 0.0005
# Tolerance (um^2) for the union-area-preservation gate: the merge must remove only
# redundant OVERLAP, never any drawn area. < 1e-6 um^2 is well below a single dbu^2.
_UNION_AREA_EPS_UM2: float = 1e-6


def _merge_layer_rings(rings_um: List[_Ring]) -> Tuple[List[_Ring], bool]:
    """UNION a layer's overlapping centroid-relative um rings into a minimal,
    overlap-free set of hole-free outer rings.

    The :class:`CellGeometry` ring format (and
    :func:`clay.layout.extract.library_device.emit_library_gds`) write ONE outer ring
    per polygon -- there is NO hole support. ``build_device`` stacks finger strips
    that legitimately UNION to the correct drawn shape but leave redundant OVERLAPPING
    same-layer polygons in the raw stamp (e.g. abutting source/drain rails, stacked
    implant/well rectangles). This collapses them to their exact union via a klayout
    boolean merge, so the emitted cell has no same-layer self-overlap, WITHOUT changing
    the drawn shape.

    Hole safety: a merged polygon CAN have a hole (a layer that forms a closed ring
    around an empty centre). Emitting only the hull ring would FILL the hole -- a
    silent shape change. So a merged polygon WITH holes is split with
    ``Polygon.decompose_trapezoids()`` into hole-free :class:`SimplePolygon` pieces
    whose areas sum EXACTLY to the holed polygon's area (the hole is preserved as the
    gap between pieces); a hole-free merged polygon emits its hull ring directly.

    Returns ``(merged_rings_um, had_holes)``. ``had_holes`` is True if any merged
    polygon on this layer needed hole-splitting (reported for honesty). Raises
    ``AssertionError`` if the merged union area differs from the original union area by
    more than :data:`_UNION_AREA_EPS_UM2` (the drawn shape must be identical -- we only
    remove redundant overlap).
    """
    import klayout.db as kdb

    dbu = _MERGE_DBU
    reg = kdb.Region()
    for ring in rings_um:
        pts = [kdb.Point(round(x / dbu), round(y / dbu)) for (x, y) in ring]
        if len(pts) >= 3:
            reg.insert(kdb.Polygon(pts))
    # area of the EXACT original union (merge first so overlap is not double-counted).
    orig_union_area = reg.merged().area() * dbu * dbu

    reg.merge()

    out_rings: List[_Ring] = []
    had_holes = False
    for poly in reg.each():
        if poly.holes() == 0:
            out_rings.append([(pt.x * dbu, pt.y * dbu) for pt in poly.each_point_hull()])
        else:
            # Holed polygon: split into hole-free trapezoids so the hole is preserved
            # as the gap between emitted pieces (NEVER filled by a hull ring).
            had_holes = True
            for sp in poly.decompose_trapezoids():
                out_rings.append([(pt.x * dbu, pt.y * dbu) for pt in sp.each_point()])

    # CORRECTNESS GATE: the merged result must cover the EXACT same area as the
    # original union -- we only removed redundant overlap, never changed the shape.
    chk = kdb.Region()
    for ring in out_rings:
        pts = [kdb.Point(round(x / dbu), round(y / dbu)) for (x, y) in ring]
        if len(pts) >= 3:
            chk.insert(kdb.Polygon(pts))
    merged_union_area = chk.merged().area() * dbu * dbu
    assert abs(merged_union_area - orig_union_area) <= _UNION_AREA_EPS_UM2, (
        "merge changed the drawn shape: union area "
        f"{merged_union_area:.9f} != original {orig_union_area:.9f} um^2 "
        f"(delta {abs(merged_union_area - orig_union_area):.3e})"
    )
    return out_rings, had_holes


def _merge_polys_by_layer(
    polys_by_layer: Dict[_LayerKey, List[_Ring]],
) -> Tuple[Dict[_LayerKey, List[_Ring]], List[_LayerKey]]:
    """Merge every layer's rings (see :func:`_merge_layer_rings`).

    Returns ``(merged_polys_by_layer, layers_that_needed_hole_splitting)``. The merge
    is applied per layer independently; the union-area gate is asserted per layer, so a
    shape change on ANY layer fails loudly rather than silently corrupting the cell.
    """
    merged: Dict[_LayerKey, List[_Ring]] = {}
    holed_layers: List[_LayerKey] = []
    for key, rings in polys_by_layer.items():
        out_rings, had_holes = _merge_layer_rings(rings)
        if out_rings:
            merged[key] = out_rings
        if had_holes:
            holed_layers.append(key)
    return merged, holed_layers


def resolve_mos_fold(
    tech,
    kind: str,
    W_um: float,
    L_um: float,
    M: int,
    row_height_target_um: float,
    *,
    max_rows: int = 4,
) -> Tuple[float, int, int, int, str]:
    """The fold DECISION for a row-height-targeted MOS, shared by the geometry
    stamp and the routing pin-probe so BOTH consume ONE fold.

    Returns ``(W_um', M', fold_rows, k, note)`` where the device's total channel
    width is conserved (``W_um' * M' == W_um * M`` to nm rounding):

      * ``k`` is the unit-finger split (:func:`_choose_finger_split`) -- each
        netlist finger of width ``W_um`` becomes ``k`` unit fingers of width
        ``W_um/k`` (clamped at the PDK channel min); ``W_um' = W_um/k``,
        ``M' = M*k``.
      * ``fold_rows`` is the serpentine band count (:func:`_choose_fold`,
        ``serpentine=True``) -- the divisor of ``M'`` whose 2D-compact aspect is
        legal and whose band height is closest to the target.

    Pure design + PDK arithmetic (the same calls :func:`build_device_geometry`
    makes inline); no reference coordinate enters. ``note`` is the provenance
    string for the finger split (empty when ``k == 1``).
    """
    from clay.layout.device_cell import _rules_from_tech, build_device as _bd

    nk = _norm_kind(kind)
    target = float(row_height_target_um)
    rules = _rules_from_tech(tech)
    ch_min_um = rules.channel_min_width / 1000.0
    w_clamped = max(float(W_um), ch_min_um)
    # fixed end margin = a 1-finger strip's height minus its channel (measured real).
    probe = _bd(tech, nk, W=w_clamped * 1e-6, L=L_um * 1e-6, nf=1)
    pe = probe.extent()
    margin_um = (pe[3] - pe[1]) / 1000.0 - w_clamped
    k = _choose_finger_split(w_clamped, margin_um, target, ch_min_um)
    note = ""
    W_out = float(W_um)
    M_out = int(M)
    if k > 1:
        width_nm = int(round(w_clamped * 1000.0))
        assert width_nm % k == 0, (
            f"finger split k={k} does not divide snapped width {width_nm}nm"
        )
        w_unit = (width_nm // k) / 1000.0
        total_nm = int(round(w_unit * 1000.0)) * k
        assert total_nm == width_nm, (
            "finger split changed the grid-exact channel width: "
            f"{total_nm}nm vs {width_nm}nm (k={k})"
        )
        note = (
            f"; finger-split k={k} w_finger={w_unit}um "
            f"(row-height target {target}um)"
        )
        W_out = w_unit
        M_out = int(M) * k

    # measure exact pitch/row-height for the serpentine fold choice.
    e1 = _bd(tech, nk, W=W_out * 1e-6, L=L_um * 1e-6, nf=1).extent()
    finger0_w_um = (e1[2] - e1[0]) / 1000.0
    row_h_um = (e1[3] - e1[1]) / 1000.0
    if M_out >= 2:
        e2 = _bd(tech, nk, W=W_out * 1e-6, L=L_um * 1e-6, nf=2).extent()
        finger_pitch_um = ((e2[2] - e2[0]) - (e1[2] - e1[0])) / 1000.0
    else:
        finger_pitch_um = finger0_w_um
    band_w_um = float(W_out)
    band_margin_um = max(row_h_um - band_w_um, 0.0)
    per_row = _choose_fold(
        M_out,
        finger_pitch_um=finger_pitch_um,
        finger0_w_um=finger0_w_um,
        row_h_um=row_h_um,
        max_rows=max_rows,
        row_height_target_um=target,
        serpentine=True,
        band_w_um=band_w_um,
        band_margin_um=band_margin_um,
    )
    return W_out, M_out, len(per_row), k, note


# The serpentine 2D-fold GATE constants (a strip MOS folds only when it has at
# least this many parallel fingers AND its single-row build is a strip beyond this
# aspect). These were the placer-private gate in annotate_to_solver; they live
# HERE now so the placer, the route geometry bridge, and the routing pin-probe all
# consume ONE fold decision (resolve_device_fold_target) and render ONE canonical
# device artwork. (The LDO pass FET MM28 has M=40 -> folds; an M=8 / aspect-2 mirror
# leg stays single-row.)
LARGE_M_FOLD_THRESHOLD: int = 8


def resolve_device_fold_target(
    tech,
    kind: str,
    W_m: float,
    L_m: float,
    nf: int,
    *,
    max_rows: int = 4,
) -> Optional[float]:
    """The SINGLE per-device fold gate: the design row-height target a large strip
    MOS folds toward, or ``None`` (single-row -- no fold).

    This is the shared authority the GENERIC placer (``annotate_to_solver``),
    the route geometry bridge (``route_cell._bridge_placement_to_route_inputs``), and
    the routing pin-probe (``route.pins.extract_pins``) all consult, so a folded
    device is rendered IDENTICALLY by every path (one canonical artwork -- not a
    placer-wide vs router-tall divergence). A device folds only when it has at least
    :data:`LARGE_M_FOLD_THRESHOLD` fingers, its single-row build is a strip beyond
    :data:`LARGE_M_FOLD_ASPECT`, AND the resolved serpentine actually has >1 band
    dividing the (split) finger count -- mirroring the placer's gate exactly. Pure
    design + PDK arithmetic (W/L/M + the PDK fold rules); carries NO net identity, so
    it is leakage-free. ``W_m`` / ``L_m`` are in METRES (the placer's units); ``nf``
    is the netlist finger multiplicity.
    """
    nk = _norm_kind(kind)
    if nk not in _BUILDABLE_KINDS:
        return None
    if int(nf) < LARGE_M_FOLD_THRESHOLD:
        return None
    from clay.layout.device_cell import build_device as _bd

    try:
        e = _bd(tech, nk, W=W_m, L=L_m, nf=max(1, int(nf))).extent()
    except Exception:
        return None
    w_um = (e[2] - e[0]) / 1000.0
    h_um = (e[3] - e[1]) / 1000.0
    if w_um <= 0.0 or h_um <= 0.0:
        return None
    if max(w_um / h_um, h_um / w_um) <= LARGE_M_FOLD_ASPECT:
        return None
    target = design_row_height_target_um(str(getattr(tech, "pdk_id", "") or ""))
    # Only fold when the serpentine resolves to >1 band evenly dividing the (split)
    # finger count -- otherwise the device stays single-row (gate declines), matching
    # the placer's `fold_rows > 1 and (M_out % fold_rows) == 0` guard.
    _W, M_out, fold_rows, _k, _note = resolve_mos_fold(
        tech, nk, W_m * 1e6, L_m * 1e6, int(nf), target, max_rows=max_rows
    )
    if fold_rows > 1 and (M_out % fold_rows) == 0:
        return target
    return None


def folded_mos_cell(
    tech,
    kind: str,
    W_um: float,
    L_um: float,
    M: int,
    row_height_target_um: float,
    *,
    nets: Optional[Dict[str, str]] = None,
    substrate_rail_nets: Sequence[str] = (),
    max_rows: int = 4,
    oxide: str = "thin",
):
    """The serpentine-folded ``DeviceGeom`` for a row-height-targeted MOS.

    Resolves the fold (:func:`resolve_mos_fold`) and returns the SHARED-DIFFUSION
    serpentine ``build_device(..., fold_rows=R)`` cell -- the SAME device geometry
    (and therefore the same terminal s/d/g/b bboxes on the real per-band M1
    landings) the emitted stamp draws. The routing pin-probe builds its probe from
    THIS so its terminal pins match the folded artwork (the bulk 'b' tap never
    coincides with the source/drain M1). Leakage-free: only the netlist params + the
    PDK fold rules feed it."""
    from clay.layout.device_cell import build_device as _bd

    nk = _norm_kind(kind)
    W_out, M_out, R, _k, _note = resolve_mos_fold(
        tech, nk, W_um, L_um, M, row_height_target_um, max_rows=max_rows
    )
    return _bd(
        tech,
        nk,
        W=W_out * 1e-6,
        L=L_um * 1e-6,
        nf=M_out,
        fold_rows=R,
        nets=nets,
        substrate_rail_nets=substrate_rail_nets,
        oxide=oxide,
    )


def build_device_geometry(
    tech_id: str,
    kind: str,
    W_um: float,
    L_um: float,
    M: int,
    *,
    max_rows: int = 4,
    row_height_target_um: Optional[float] = None,
    nets: Optional[Dict[str, str]] = None,
    substrate_rail_nets: Sequence[str] = (),
    contact_rows: bool = False,
    oxide: str = "thin",
) -> CellGeometry:
    """Synthesise a position-free :class:`CellGeometry` stamp from device parameters.

    Builds a folded m=M MOS for ``tech_id`` (any PDK a
    :class:`~clay.layout.tech.LayoutTech` resolves) by stacking
    :func:`clay.layout.device_cell.build_device` rows (a compact multi-row fold, not
    one wide single-row strip -- see :func:`_choose_fold`). ``W_um`` is the
    per-finger channel width and ``M`` the total finger multiplicity (total channel
    = ``W_um * M``); ``build_device`` takes METRES, so we convert. The stacked
    polygons are recentred on the cell centroid, so the result is a pure
    centroid-relative stamp identical in shape to a vendor capture -- it carries NO
    position and is stamped by the same emitter.

    ``row_height_target_um`` (OPT-IN; default ``None`` is byte-identical):
    a DESIGN-chosen stamp-height target -- the wide-flat unit-transistor
    discipline the reference style uses, fed as a design statistic (e.g. the
    median per-finger W of the design's own MOS set), NEVER a reference
    dimension. Two mechanisms, both electrically conserving:

      * FINGER SPLIT (tall device): each netlist finger of width ``W_um``
        becomes ``k`` unit fingers of width ``W_um / k``
        (:func:`_choose_finger_split`; total channel width conserved to nm
        rounding, asserted), so the single-row height ``W/k + margin`` lands
        near the target. The unit width never drops below the PDK channel
        min width.
      * ROW STACKING (short device): :func:`_choose_fold` stacks rows toward
        the target height instead of toward a square.

    A targeted stamp carries ``cg.fold_target`` (``target_um / k /
    w_finger_um / n_fingers``) for provenance and names the split in
    ``cg.reason``. Raises ``ValueError`` for a non-positive target.

    Returns a ``CellGeometry`` with ``clean=True`` and ``polys_by_layer`` filled.
    The instance also carries ``.source = 'build_device'`` for provenance.

    Raises ``ValueError`` for a non-MOS kind (``build_device`` models only MOS): a
    cap/res has no parametric fallback here, and we refuse to fabricate one.
    """
    from clay.layout.tech import LayoutTech
    from clay.layout.device_cell import build_device

    nk = _norm_kind(kind)
    if nk not in _BUILDABLE_KINDS:
        raise ValueError(
            f"build_device_geometry models only MOS (nmos/pmos); kind={kind!r} "
            f"({nk!r}) has no parametric fallback"
        )
    if row_height_target_um is not None and not (float(row_height_target_um) > 0.0):
        raise ValueError(
            f"row_height_target_um must be > 0 (got {row_height_target_um!r})"
        )

    tech = LayoutTech.resolve(tech_id)

    # OPT-IN finger split toward the design row-height target (see docstring).
    # The fold DECISION (split k + serpentine band count) is resolved by the SHARED
    # :func:`resolve_mos_fold` so the routing pin-probe (clay.layout.route.pins) and
    # this geometry stamp consume ONE fold -- the probe's terminal pins land on the
    # same per-band M1 the stamp draws (the folded pin-probe contract).
    fold_split_k = 1
    fold_note = ""
    _fold_R: Optional[int] = None
    if row_height_target_um is not None:
        W_um, M, _fold_R, fold_split_k, fold_note = resolve_mos_fold(
            tech, nk, W_um, L_um, M, float(row_height_target_um), max_rows=max_rows
        )

    # Build each distinct row strip once (cache by nf), then stack rows vertically.
    row_cache: Dict[int, object] = {}

    def _row_geom(nf: int):
        if nf not in row_cache:
            # contact_rows (P-C3 density lever): a FULL contact-cut column per
            # S/D face (vs the default single mid-channel cut), raising CONT
            # toward the reference's per-finger column density. SAFE on this
            # per-device synthesis path -- the prior two oracle rejections
            # (2026-06-11) were on the INTERDIGITATED composite/abut path, where
            # the COMP +/-105nm end extension welded deliberately-tight shared-
            # gate COMP gaps (KLayout extracted a W=567U monster pfet). The
            # rings_group path builds SEPARATE per-device stamps banded by well-
            # island with ring spacing -- no shared COMP between devices to weld,
            # so the dense column rides the gapped-strip build cleanly (verified
            # by the FULL LVS + deck DRC oracle). Default OFF stays byte-stable.
            row_cache[nf] = build_device(
                tech, nk, W=W_um * 1e-6, L=L_um * 1e-6, nf=nf,
                nets=_build_device_nets(nets),
                substrate_rail_nets=substrate_rail_nets,
                contact_rows=contact_rows,
                oxide=oxide,
            )
        return row_cache[nf]

    # Measure the EXACT per-finger pitch + row height from real strips so the fold
    # choice is the genuinely-most-compact one (not a guess). A 1-finger strip gives
    # the row height + the single-finger width; a 2-finger strip gives the pitch.
    e1 = _row_geom(1).extent()
    finger0_w_um = (e1[2] - e1[0]) / 1000.0
    row_h_um = (e1[3] - e1[1]) / 1000.0
    if M >= 2:
        e2 = _row_geom(2).extent()
        finger_pitch_um = ((e2[2] - e2[0]) - (e1[2] - e1[0])) / 1000.0
    else:
        finger_pitch_um = finger0_w_um

    # SERPENTINE 2D-COMPACT FOLD (the footprint-contract cure): when a row-height
    # target drives the fold, build ONE shared-diffusion serpentine device whose
    # rows ABUT on a continuous COMP -- NOT R gapped strips (the strips fragment
    # the diffusion: KLayout extracts R nch instead of one, and the colliding
    # contact rows fire CO.EN.3/CO.S.1; the measured S1 blocker). ``_choose_fold``
    # bounds the WIDTH/aspect by adding the FEWEST rows whose aspect <= 6 (the
    # height-only fold kept ONE row -> 594um wide -> aspect 150:1). The default
    # (no target) path stays the byte-identical gapped strip stacking below.
    serpentine = row_height_target_um is not None
    terminal_accesses_nm: Dict[str, List[Tuple[int, int]]] = {}
    physical_occurrences_nm: List[Tuple[int, int, str]] = []
    if serpentine:
        # the serpentine band count was resolved by the SHARED resolve_mos_fold
        # above (so the pin-probe and this stamp fold identically). per_row is the
        # uniform band split [M//R]*R (R | M by construction).
        R = int(_fold_R) if _fold_R is not None else 1
        per_row = [M // R] * R
        # ONE serpentine device of the full (split) finger count over R bands. The
        # serpentine build requires nf % fold_rows == 0; resolve_mos_fold returns
        # divisors of M so R | M by construction (per_row is [M//R]*R).
        serp = build_device(
            tech, nk, W=W_um * 1e-6, L=L_um * 1e-6, nf=M, fold_rows=R,
            nets=_build_device_nets(nets), contact_rows=contact_rows,
            substrate_rail_nets=substrate_rail_nets,
            oxide=oxide,
        )
        physical_occurrences_nm.extend(_physical_mos_occurrences_nm(serp, tech))
        stacked = _device_polys_by_layer(serp)
        for role, terminal in (getattr(serp, "terminals", {}) or {}).items():
            tx, ty = terminal.center
            terminal_accesses_nm.setdefault(str(role), []).append(
                (int(tx), int(ty))
            )
        # provenance: the fold is the serpentine row count (uniform bands).
        _fold_descr = per_row
        # ONE serpentine device -> its own well_tap_isolated decision is the
        # whole stamp's decision (see device_cell.DeviceGeom.well_tap_isolated).
        well_tap_isolated_flag = bool(getattr(serp, "well_tap_isolated", False))
    else:
        per_row = _choose_fold(
            M,
            finger_pitch_um=finger_pitch_um,
            finger0_w_um=finger0_w_um,
            row_h_um=row_h_um,
            max_rows=max_rows,
            row_height_target_um=None,
        )
        _fold_descr = per_row
        stacked = _stack_strips(
            per_row,
            _row_geom,
            terminal_accesses_nm=terminal_accesses_nm,
        )
        row_gap_nm = int(round(_ROW_GAP_UM * 1000.0))
        y_cursor_nm = 0
        for nf in per_row:
            row_geometry = _row_geom(nf)
            row_extent = row_geometry.extent()
            dy = y_cursor_nm - row_extent[1]
            physical_occurrences_nm.extend(
                (x, y + dy, orientation)
                for x, y, orientation in _physical_mos_occurrences_nm(
                    row_geometry, tech
                )
            )
            y_cursor_nm += row_extent[3] - row_extent[1] + row_gap_nm
        # The rows of ONE device share one tap decision (same nets); OR across
        # the row strips actually stacked (each ``_row_geom(nf)`` call below is
        # a cache hit -- the strip was already built by ``_stack_strips`` / the
        # e1/e2 pitch measurement above, so this adds no new build_device work).
        well_tap_isolated_flag = any(
            getattr(_row_geom(nf), "well_tap_isolated", False) for nf in per_row
        )

    # Centroid of the stacked cell = centre of its overall bbox (every layer).
    xs: List[int] = []
    ys: List[int] = []
    for rings in stacked.values():
        for ring in rings:
            for (x, y) in ring:
                xs.append(x)
                ys.append(y)
    if not xs:
        # build_device returned nothing usable -- honest empty (should not happen for
        # a MOS, but never fabricate geometry to hide it).
        cg = CellGeometry(
            name="",
            kind=nk,
            ref_centroid=(0.0, 0.0),
            polys_by_layer={},
            n_instances=0,
            finger_cells=[],
            clean=False,
            reason="build_device produced no polygons",
        )
        cg.source = "build_device"
        return cg

    cx_nm = (min(xs) + max(xs)) / 2.0
    cy_nm = (min(ys) + max(ys)) / 2.0

    # Recentre on the centroid and convert nm -> um (the CellGeometry contract).
    # Do not decimal-round the half-nm local frame: placement composes this with
    # the absolute centroid before the one canonical integer-nm quantization.
    polys_by_layer: Dict[_LayerKey, List[_Ring]] = {}
    for key, rings in stacked.items():
        out_rings: List[_Ring] = []
        for ring in rings:
            out_rings.append(
                [
                    ((x - cx_nm) / 1000.0, (y - cy_nm) / 1000.0)
                    for (x, y) in ring
                ]
            )
        if out_rings:
            polys_by_layer[key] = out_rings

    # MERGE-CLEAN (FIX 1): ``build_device`` stacks finger strips whose raw same-layer
    # polygons OVERLAP (they union to the correct drawn shape, so per-cell width/space
    # DRC passes, but the GDS carries redundant overlapping polygons -> not
    # signoff-clean). Union each layer so the emitted cell has NO same-layer
    # self-overlap, WITHOUT changing the drawn shape (the union-area gate inside
    # :func:`_merge_layer_rings` asserts the shape is identical). Applied ONLY here
    # (vendor / reference_passive captures are real PCells -- left untouched).
    polys_by_layer, holed_layers = _merge_polys_by_layer(polys_by_layer)

    if serpentine:
        reason = (
            f"synthesised by build_device serpentine fold rows={len(_fold_descr)} "
            f"(nf/row={_fold_descr[0]}, shared-diffusion)"
        )
        finger_cells = [
            f"build_device({nk},nf={M},fold_rows={len(_fold_descr)})"
        ]
    else:
        reason = (
            f"synthesised by build_device fold={_fold_descr} "
            f"(rows={len(_fold_descr)})"
        )
        finger_cells = [f"build_device({nk},nf={nf})" for nf in _fold_descr]
    reason += fold_note
    if holed_layers:
        # Honest provenance: a holed merged polygon was hole-split (NOT hull-filled).
        reason += "; merged-clean (hole-split layers: " + ",".join(
            f"{l}/{d}" for (l, d) in holed_layers
        ) + ")"
    else:
        reason += "; merged-clean"

    cg = CellGeometry(
        name="",
        kind=nk,
        ref_centroid=(0.0, 0.0),  # a stamp has no reference position
        polys_by_layer=polys_by_layer,
        n_instances=len(physical_occurrences_nm),
        finger_cells=finger_cells,
        clean=True,
        reason=reason,
        well_tap_isolated=well_tap_isolated_flag,
    )
    cg.source = "build_device"
    cg.merged_clean = True
    cg.hole_split_layers = list(holed_layers)
    expected_channels = int(M) // len(_fold_descr) if serpentine else int(M)
    if len(physical_occurrences_nm) != expected_channels:
        raise ValueError(
            "materialized MOS occurrence inventory differs from evaluated fold: "
            f"observed={len(physical_occurrences_nm)} expected={expected_channels}"
        )
    cg.source_finger_count = int(M)
    cg.physical_occurrences = tuple(
        (
            (x - cx_nm) / 1000.0,
            (y - cy_nm) / 1000.0,
            orientation,
        )
        for x, y, orientation in sorted(physical_occurrences_nm)
    )
    # Exact materializer-owned access inventory.  A default gapped fold contains
    # multiple electrically-disjoint rows, so one rebuilt single-row probe is not
    # an equivalent terminal model: it can land a body point on a source rail and
    # leaves every later row unseeded.  Carry one access per real row/terminal in
    # the same centroid-relative frame as the emitted polygons.  The router may
    # then connect every disjoint row and the ownership mini-extractor can name
    # every M1 island without guessing from geometry.
    cg.terminal_accesses_um = {
        role: tuple(
            ((x - cx_nm) / 1000.0, (y - cy_nm) / 1000.0)
            for x, y in accesses
        )
        for role, accesses in sorted(terminal_accesses_nm.items())
        if accesses
    }
    cg.terminal_offsets_um = {
        role: accesses[0]
        for role, accesses in cg.terminal_accesses_um.items()
    }
    cg.terminal_layers = {
        role: tuple(int(value) for value in tech.metal_key(1))
        for role in cg.terminal_accesses_um
    }
    if row_height_target_um is not None:
        # provenance of the opt-in fold target (a DESIGN statistic; the
        # default-None path sets NO attribute -- byte-identical output).
        cg.fold_target = {
            "target_um": float(row_height_target_um),
            "k": int(fold_split_k),
            "w_finger_um": float(W_um),
            "n_fingers": int(M),
        }
    return cg


# --------------------------------------------------------------------------- #
# 1b. composite matched-group stamp (OPT-IN consumer face; default paths
#     unchanged -- nothing calls this unless a flow explicitly opts in).
# --------------------------------------------------------------------------- #
def composite_group_geometry(
    tech_id: str,
    devices,
    *,
    pattern: str = "ABBA",
    contact_rows: bool = False,
    name: str = "",
) -> CellGeometry:
    """ONE merge-clean :class:`CellGeometry` stamp for a MATCHED device group.

    The P4b consumer face: a matched group (same kind/W/L; 2-4 devices -- a
    diff pair / mirror legs) is realised as a single interdigitated
    unit-transistor array (:func:`clay.layout.extract.interdigitate.
    composite_matched_cell`) and returned as the SAME stamp currency every
    other device uses, so a placement can stamp the whole group as ONE cell.

    ``devices`` are dicts or attribute objects carrying ``name`` / ``kind`` /
    ``W`` (m) / ``L`` (m) / ``M`` / ``nets``. The group's per-device terminal
    accesses live on the composite cell (build it directly via
    ``composite_matched_cell`` when terminal routing is needed); this wrapper
    returns only the geometry stamp.

    OPT-IN ONLY: ``real_device_geometry`` never calls this -- replacing N
    per-device stamps with one composite changes the placement contract
    (N names -> 1 stamp) and is wired by the integrating flow, not here.
    """
    from clay.layout.extract.interdigitate import composite_matched_cell
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)
    cell = composite_matched_cell(
        tech, devices, pattern=pattern, contact_rows=contact_rows
    )
    return cell.to_cell_geometry(name=name)


# --------------------------------------------------------------------------- #
# 1c. guard-ring augmentation (OPT-IN consumer face; default paths unchanged --
#     nothing reaches this unless a flow passes guard_rings=True).
# --------------------------------------------------------------------------- #

# MOS kind -> guard-ring well polarity. An nmos sits in the p-substrate, so its
# ring is a substrate-tap (p) ring; a pmos sits in an nwell, so its ring is an
# nwell-tap (n) ring. Pure device physics, not a per-PDK or per-circuit branch.
_RING_WELL_FOR_KIND = {"nmos": "p", "pmos": "n"}


def _guard_ring_mode(value) -> str:
    """Normalise the ``guard_rings`` parameter face: ``False``/``None`` ->
    ``'off'``; ``True`` -> ``'device'`` (back-compat alias); the strings
    ``'device'`` / ``'group'`` pass through. Anything else raises -- an
    unknown granularity must never silently fall back to a different one."""
    if value is False or value is None:
        return "off"
    if value is True:
        return "device"
    if isinstance(value, str) and value in ("device", "group"):
        return value
    raise ValueError(
        "guard_rings must be False, True (alias of 'device'), 'device' or "
        f"'group'; got {value!r}"
    )


def _stamp_bbox_nm(cg: CellGeometry) -> Optional[Tuple[int, int, int, int]]:
    """The stamp's centroid-relative bbox in nm ints (conservative: floor the
    mins, ceil the maxes, so the ring's protected region never undershoots the
    drawn um polygons). None for an empty stamp."""
    xs = [x for rings in cg.polys_by_layer.values() for r in rings for (x, _y) in r]
    ys = [y for rings in cg.polys_by_layer.values() for r in rings for (_x, y) in r]
    if not xs or not ys:
        return None
    return (
        math.floor(min(xs) * 1000.0),
        math.floor(min(ys) * 1000.0),
        math.ceil(max(xs) * 1000.0),
        math.ceil(max(ys) * 1000.0),
    )


def attach_guard_ring(
    cg: CellGeometry,
    tech_id: str,
    *,
    well: str,
    tap_net: str = "",
) -> CellGeometry:
    """Wrap a :class:`CellGeometry` stamp with a PDK-rule-driven guard ring.

    The ring comes from :func:`clay.layout.guard_ring.build_guard_ring` (every
    dimension resolved from the shared ``DeviceCellRules`` surface -- no
    hardcoded nm, no reference coordinate; the protected region is the stamp's
    OWN drawn bbox). Its polygons are converted nm -> centroid-relative um and
    APPENDED to ``cg.polys_by_layer`` IN PLACE, so:

      * the stamp extent grows symmetrically (the ring offset is uniform on all
        four sides, so the bbox centre -- the stamp's placement reference -- is
        unchanged), and
      * the EXISTING sizes pipeline (:func:`stamp_extents` /
        :func:`padded_stamp_extents` -> placement) reserves the ring room
        automatically. There is NO second size path.

    The ring's tap terminal is recorded on the stamp as ``cg.guard_ring``::

        {"well":         'p' | 'n',
         "terminal":     {"role": "tap", "net": tap_net,
                          "bbox_um": (x0, y0, x1, y1),   # centroid-relative
                          "layer": (gds_layer, datatype)},  # the PDK M1 key
         "opening_bbox_um": ..., "ring_bbox_um": ...,    # centroid-relative
         "cut_count":    int,
         "layers_drawn": {role: (gds_layer, datatype)}}

    so a router can strap the ring to ``tap_net`` (routable access). ``tap_net``
    is the caller's account -- :func:`real_device_geometry` derives it from the
    design data (the device's ``body`` terminal net), never from a rail-name
    convention.

    Raises ``ValueError`` for an empty stamp or a stamp already ringed (a
    second ring around a ringed stamp is never meaningful), and propagates
    ``build_guard_ring``'s explicit-unsupported errors (e.g. an n-ring on a
    PDK that draws no nwell) rather than silently skipping the ring.
    """
    from clay.layout.guard_ring import build_guard_ring
    from clay.layout.tech import LayoutTech

    if getattr(cg, "guard_ring", None) is not None:
        raise ValueError(f"stamp {cg.name!r} already carries a guard ring")
    bb = _stamp_bbox_nm(cg)
    if bb is None:
        raise ValueError(f"stamp {cg.name!r} is empty; nothing to enclose")

    tech = LayoutTech.resolve(tech_id)
    ring = build_guard_ring(
        tech, bb, well=well, nets={"tap": tap_net} if tap_net else None
    )

    for lp in ring.polygons:
        key = (int(lp.layer), int(lp.datatype))
        ring_um = [
            (round(v.x / 1000.0, 4), round(v.y / 1000.0, 4))
            for v in lp.polygon.vertices
        ]
        if len(ring_um) >= 3:
            cg.polys_by_layer.setdefault(key, []).append(ring_um)

    t = ring.terminals["tap"]
    to_um = lambda box: tuple(round(v / 1000.0, 4) for v in box)  # noqa: E731
    cg.guard_ring = {
        "well": ring.well,
        "terminal": {
            "role": t.role,
            "net": t.net,
            "bbox_um": to_um(t.bbox),
            "layer": (int(ring.layers_drawn["metal1"][0]),
                      int(ring.layers_drawn["metal1"][1])),
        },
        "opening_bbox_um": to_um(ring.opening_bbox),
        "ring_bbox_um": to_um(ring.ring_bbox),
        "cut_count": int(ring.cut_count),
        "layers_drawn": {
            k: (int(v[0]), int(v[1])) for k, v in ring.layers_drawn.items()
        },
    }
    cg.reason = (cg.reason + "; " if cg.reason else "") + (
        f"guard ring well={ring.well} tap_net={t.net!r} cuts={ring.cut_count}"
    )
    return cg


# --------------------------------------------------------------------------- #
# 1d. GROUP-granularity guard rings (placement-consuming, OUTSIDE the
#     geometry pipeline). The reference layout uses ~8 M1 rings around ~20
#     transistors (group granularity); per-device rings cost 1.399x area. The
#     geometry pipeline cannot build group rings (a stamp has no position), so
#     the group face is a separate function the integrating flow calls AFTER
#     placement: ``group_guard_rings(geometry, placement, intent, tech_id)``.
#     Cluster selection is pure design topology + the (design-sourced) solved
#     placement -- mirror groups are read for honest split notes, rows come
#     from the placed y-intervals, adjacency from the placed x-order; NEVER a
#     reference coordinate.
# --------------------------------------------------------------------------- #
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GroupGuardRing:
    """One group-granularity guard ring: a cluster of same-kind, same-body-net
    row-adjacent MOS devices wrapped by ONE PDK-rule-driven tap ring.

    ``ring`` is a :class:`clay.layout.guard_ring.GuardRingGeom` whose polygons
    are PLACEMENT-ABSOLUTE nm (the inner bbox was the union of the members'
    placed stamp bboxes), so the integrating flow can drop them straight into
    the emitted artwork / obstacle model. ``tap_net`` is the cluster's common
    DESIGN body net (resolved through the ring terminal -- for a p ring with
    no body net it is the PDK substrate net)."""

    members: Tuple[str, ...]
    kind: str
    well: str
    tap_net: str
    row: int
    ring: object
    notes: Tuple[str, ...] = ()


def _placed_stamp_bboxes_nm(
    geometry: Dict[str, CellGeometry], placement: Dict[str, _XY]
) -> Dict[str, Tuple[int, int, int, int]]:
    """Absolute-nm bbox per stamp placed at its (um) position; stamps without
    a position or without polygons are absent (reported by the caller)."""
    if (
        type(geometry) is dict
        and type(placement) is dict
        and os.environ.get("CLAY_CPP_STAMP_GEOM", "1") != "0"
    ):
        from clay._native import load_native

        eng = load_native("_engine_cpp")
        if eng is not None and hasattr(eng, "placed_stamp_bboxes_nm_cpp"):
            return eng.placed_stamp_bboxes_nm_cpp(geometry, placement)
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for name, cg in geometry.items():
        pos = placement.get(name)
        if pos is None:
            continue
        points = [
            (
                *placed_point_nm(float(x), float(y), pos),
            )
            for rings in (getattr(cg, "polys_by_layer", {}) or {}).values()
            for ring in rings
            for x, y in ring
        ]
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        out[name] = (min(xs), min(ys), max(xs), max(ys))
    return out


def _placed_stamp_layer_bboxes_nm(
    geometry: Dict[str, CellGeometry], placement: Dict[str, _XY]
) -> Dict[str, Dict[_LayerKey, Tuple[int, int, int, int]]]:
    """Absolute exact-nm bboxes per placed stamp and physical layer.

    Unlike :func:`_placed_stamp_bboxes_nm`, this authority deliberately keeps
    layer extents separate so a terminal-metal overhang cannot be charged an
    unrelated active/implant spacing rule.  Malformed layer keys or polygon
    inventories stop the proof instead of being normalized.
    """
    if (
        type(geometry) is dict
        and type(placement) is dict
        and os.environ.get("CLAY_CPP_STAMP_GEOM", "1") != "0"
    ):
        from clay._native import load_native

        eng = load_native("_engine_cpp")
        if eng is not None and hasattr(eng, "placed_stamp_layer_bboxes_nm_cpp"):
            return eng.placed_stamp_layer_bboxes_nm_cpp(geometry, placement)

    out: Dict[str, Dict[_LayerKey, Tuple[int, int, int, int]]] = {}
    for name, cg in geometry.items():
        pos = placement.get(name)
        if pos is None:
            continue
        raw_layers = getattr(cg, "polys_by_layer", None)
        if type(raw_layers) is not dict:
            raise TypeError("placed layer bbox authority must be an exact mapping")
        layer_boxes: Dict[_LayerKey, Tuple[int, int, int, int]] = {}
        for raw_key, raw_rings in raw_layers.items():
            if (
                type(raw_key) is not tuple
                or len(raw_key) != 2
                or any(type(value) is not int for value in raw_key)
            ):
                raise TypeError("placed layer bbox authority has a malformed layer key")
            if type(raw_rings) is not list:
                raise TypeError("placed layer bbox authority has a malformed polygon list")
            points = [
                placed_point_nm(float(x), float(y), pos)
                for ring in raw_rings
                for x, y in ring
            ]
            if not points:
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            layer_boxes[raw_key] = (
                min(xs), min(ys), max(xs), max(ys)
            )
        if layer_boxes:
            out[str(name)] = layer_boxes
    return out


def _placed_stamp_layer_rectangles_nm(
    geometry: Dict[str, CellGeometry], placement: Dict[str, _XY]
) -> Dict[str, Dict[_LayerKey, Tuple[Tuple[int, int, int, int], ...]]]:
    """Exact Manhattan rectangle decomposition per placed stamp and layer.

    This is the non-convex companion to :func:`_placed_stamp_layer_bboxes_nm`.
    It consumes the same emitted polygon inventory but preserves holes and gaps;
    a non-Manhattan trapezoid fails closed instead of receiving bbox credit.
    """
    if (
        type(geometry) is dict
        and type(placement) is dict
        and os.environ.get("CLAY_CPP_STAMP_GEOM", "1") != "0"
    ):
        from clay._native import load_native

        eng = load_native("_engine_cpp")
        if eng is not None and hasattr(eng, "placed_stamp_layer_rectangles_nm_cpp"):
            return eng.placed_stamp_layer_rectangles_nm_cpp(geometry, placement)

    import klayout.db as kdb

    out: Dict[
        str, Dict[_LayerKey, Tuple[Tuple[int, int, int, int], ...]]
    ] = {}
    for name, cg in geometry.items():
        pos = placement.get(name)
        if pos is None:
            continue
        raw_layers = getattr(cg, "polys_by_layer", None)
        if type(raw_layers) is not dict:
            raise TypeError("placed layer rectangle authority must be an exact mapping")
        layer_rectangles = {}
        for raw_key, raw_rings in raw_layers.items():
            if (
                type(raw_key) is not tuple
                or len(raw_key) != 2
                or any(type(value) is not int for value in raw_key)
            ):
                raise TypeError(
                    "placed layer rectangle authority has a malformed layer key"
                )
            if type(raw_rings) is not list:
                raise TypeError(
                    "placed layer rectangle authority has a malformed polygon list"
                )
            region = kdb.Region()
            for ring in raw_rings:
                points = [
                    kdb.Point(*placed_point_nm(float(x), float(y), pos))
                    for x, y in ring
                ]
                if len(points) < 3:
                    raise ValueError(
                        "placed layer rectangle authority has a degenerate polygon"
                    )
                polygon = kdb.Polygon(points)
                if polygon.area() <= 0:
                    raise ValueError(
                        "placed layer rectangle authority has a zero-area polygon"
                    )
                region.insert(polygon)
            region.merge()
            rectangles = []
            for piece in region.decompose_trapezoids():
                bbox = piece.bbox()
                if piece.area() != bbox.area():
                    raise ValueError(
                        "placed layer rectangle authority requires Manhattan artwork"
                    )
                rectangles.append((bbox.left, bbox.bottom, bbox.right, bbox.top))
            canonical = tuple(sorted(set(rectangles)))
            if canonical:
                layer_rectangles[raw_key] = canonical
        if layer_rectangles:
            out[str(name)] = layer_rectangles
    return out


def _placed_stamp_layer_component_bboxes_nm(
    geometry: Dict[str, CellGeometry], placement: Dict[str, _XY]
) -> Dict[str, Dict[_LayerKey, Tuple[Tuple[int, int, int, int], ...]]]:
    """Exact bbox inventory for each connected same-layer component."""

    import klayout.db as kdb

    rectangles_by_stamp = _placed_stamp_layer_rectangles_nm(geometry, placement)
    out: Dict[
        str, Dict[_LayerKey, Tuple[Tuple[int, int, int, int], ...]]
    ] = {}
    for name, layers in rectangles_by_stamp.items():
        components_by_layer = {}
        for layer, rectangles in layers.items():
            region = kdb.Region()
            for x0, y0, x1, y1 in rectangles:
                region.insert(kdb.Box(x0, y0, x1, y1))
            region.merge()
            components = tuple(sorted({
                (
                    component.bbox().left,
                    component.bbox().bottom,
                    component.bbox().right,
                    component.bbox().top,
                )
                for component in region.each_merged()
            }))
            if components:
                components_by_layer[layer] = components
        if components_by_layer:
            out[name] = components_by_layer
    return out


def _rows_by_y(
    bboxes: Dict[str, Tuple[int, int, int, int]]
) -> List[List[Tuple[str, Tuple[int, int, int, int]]]]:
    """Bottom-up rows by transitive y-interval overlap (standard interval
    merge): two stamps share a row iff their placed y-intervals overlap
    (directly or through a chain). Deterministic: sorted by (y0, y1, name)."""
    items = sorted(bboxes.items(), key=lambda kv: (kv[1][1], kv[1][3], kv[0]))
    rows: List[List[Tuple[str, Tuple[int, int, int, int]]]] = []
    cur: List[Tuple[str, Tuple[int, int, int, int]]] = []
    cur_top: Optional[int] = None
    for name, bb in items:
        if cur and cur_top is not None and bb[1] >= cur_top:
            rows.append(cur)
            cur = []
            cur_top = None
        cur.append((name, bb))
        cur_top = bb[3] if cur_top is None else max(cur_top, bb[3])
    if cur:
        rows.append(cur)
    return rows


def _bands_by_y_gap(
    bboxes: Dict[str, Tuple[int, int, int, int]],
    gap_factor: float = 3.0,
    gap_min_nm: float = 8000.0,
) -> List[List[Tuple[str, Tuple[int, int, int, int]]]]:
    """Cluster PLACED stamps into bands by a centroid-y GAP cut (the same
    relative cut ``relgraph.cluster_bands`` uses to recover the solve's bands),
    applied to the placed bboxes.

    This replaces the transitive y-interval merge :func:`_rows_by_y` for ring
    clustering. ``_rows_by_y`` chains any two stamps whose y-intervals overlap,
    so a 2D-packed (relax) placement -- where a tall stamp (a cap) and the rows
    above/below it all share overlapping y-intervals -- collapses into ONE row;
    the downstream x-runs then cluster vertically-distant devices and the rings
    overlap (option-matrix red account). Clustering by centroid-y GAP recovers
    the real macro bands (PMOS top / passives mid / NMOS bottom on the OTA): a
    new band starts when the centroid-y gap exceeds ``max(gap_factor * median
    within-band gap, gap_min_nm)`` -- relative (scales across PDKs), floored by
    an absolute minimum. Returns bottom-up rows of ``(name, bbox)``."""
    items = sorted(
        bboxes.items(), key=lambda kv: ((kv[1][1] + kv[1][3]) / 2.0, kv[0])
    )
    if len(items) <= 1:
        return [list(items)] if items else []
    cys = [(bb[1] + bb[3]) / 2.0 for _n, bb in items]
    gaps = [cys[i + 1] - cys[i] for i in range(len(cys) - 1)]
    # Threshold: the relative cut (gap_factor * median within-band gap) needs a
    # population of within-band gaps to estimate the median; with < 3 gaps the
    # median degenerates to the (possibly band-separating) gap itself and the
    # relative term would never split. Fall back to the absolute floor there, so
    # two genuinely row-separated devices still split (and same-row devices stay
    # together below the floor).
    if len(gaps) < 3:
        threshold = gap_min_nm
    else:
        sg = sorted(gaps)
        m = len(sg)
        median = sg[m // 2] if m % 2 == 1 else (sg[m // 2 - 1] + sg[m // 2]) / 2.0
        threshold = max(gap_factor * median, gap_min_nm)
    rows: List[List[Tuple[str, Tuple[int, int, int, int]]]] = []
    cur: List[Tuple[str, Tuple[int, int, int, int]]] = [items[0]]
    for i in range(1, len(items)):
        if cys[i] - cys[i - 1] > threshold:
            rows.append(cur)
            cur = []
        cur.append(items[i])
    if cur:
        rows.append(cur)
    return rows


def _cluster_ring_pad_top(names: Tuple[str, ...], member_ring_pad_top_nm) -> int:
    """The extra nm to add to a ring cluster's INNER top edge (y1) so the frame spans
    a declared common BAND height. ``member_ring_pad_top_nm`` maps a member stamp /
    device name -> the upward pad its block owes (band_common_height - block_height);
    a cluster takes the max pad over its member names. ``None`` / no match -> 0 (byte-
    identical). Used to HEIGHT-ALIGN the top-band rings without distorting the enclosed
    device geometry (only the ring extent is padded)."""
    if not member_ring_pad_top_nm:
        return 0
    return max((int(member_ring_pad_top_nm.get(n, 0)) for n in names), default=0)


def _cluster_ring_pad_bottom(
    names: Tuple[str, ...], member_ring_pad_bottom_nm
) -> int:
    """Extra nm below a ring cluster's inner bottom edge.

    This is the downward-symmetric floorplan reservation to
    :func:`_cluster_ring_pad_top`: only the ring frame grows, while enclosed device
    geometry and its terminals remain fixed.
    """

    if not member_ring_pad_bottom_nm:
        return 0
    return max(
        (int(member_ring_pad_bottom_nm.get(name, 0)) for name in names),
        default=0,
    )


def _materialize_shared_mos_well(ring, *, kind: str, well: str):
    """Join a MOS group's body well to the guard-ring tap well.

    A passive/isolation N-well ring is a hollow isolation frame, but that is not
    a body tie for an enclosed PMOS group: the N+ tap would live in the frame's
    N-well island while the devices lived in separate N-well islands inside the
    hole.  The group builder therefore requests ``shared_well=True`` so the
    constructor omits inter-island spacing and emits a solid carrier already;
    this function keeps the final lowering explicit and canonical.

    A PMOS ``well='n'`` group instead owns one shared N-well carrier.  Replace
    only that group's N-well frame with its exact outer rectangle; the active,
    implant, contact and M1 guard-ring artwork remains a closed frame.  The
    decision is topology-role based (MOS kind + declared well), never device or
    net-name based.  Passive N-well isolation rings remain hollow.
    """
    if kind != "pmos" or well != "n":
        return ring
    key = tuple((getattr(ring, "layers_drawn", {}) or {}).get("nwell", ()))
    if len(key) != 2:
        return ring
    well_polys = [
        lp for lp in ring.polygons
        if (int(lp.layer), int(lp.datatype)) == key
    ]
    if not well_polys:
        return ring

    from clay.layout.device_cell import LayerPolygon, Polygon

    xs = [int(v.x) for lp in well_polys for v in lp.polygon.vertices]
    ys = [int(v.y) for lp in well_polys for v in lp.polygon.vertices]
    carrier = LayerPolygon(
        layer=key[0],
        datatype=key[1],
        polygon=Polygon.rectangle(min(xs), min(ys), max(xs), max(ys)),
    )
    ring.polygons[:] = [
        lp for lp in ring.polygons
        if (int(lp.layer), int(lp.datatype)) != key
    ] + [carrier]
    return ring


def _grow_bbox(box: Tuple[int, int, int, int], amount_nm: int):
    amount = int(amount_nm)
    return (
        int(box[0]) - amount,
        int(box[1]) - amount,
        int(box[2]) + amount,
        int(box[3]) + amount,
    )


def _union_bbox(*boxes: Tuple[int, int, int, int]):
    return (
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    )


def _materialize_deep_nwell_bundle(
    ring,
    *,
    tech,
    body_net: str,
    deep_nwell_net: str,
    substrate_net: str,
):
    """Wrap one declared isolated NMOS p-ring in a routed triple-well bundle.

    The topology mirrors the proven foundry/TED pattern without copying a PCell:
    inner P+ body ring -> solid drawn p-well -> N+ deep-well tap -> solid deep
    N-well -> required outer P+ substrate ring.  Every numeric distance and LVS
    model remains PDK data.
    """

    rules = tech.deep_nwell()
    if rules is None:
        raise ValueError(
            f"{tech.pdk_id}: @well_isolated requires a proven deep-N-well recipe"
        )
    if not str(deep_nwell_net):
        raise ValueError("deep-N-well materialization requires a positive supply net")
    if not str(substrate_net):
        raise ValueError("deep-N-well materialization requires a substrate net")

    from clay.layout.device_cell import LayerPolygon, Polygon
    from clay.layout.guard_ring import build_guard_ring

    pwell_key = (
        tech.build_layer(str(rules.pwell_layer))
        if rules.pwell_layer is not None
        else None
    )
    dnwell_key = tech.build_layer(str(rules.deep_nwell_layer))
    if rules.pwell_layer is not None and pwell_key is None:
        raise ValueError(
            f"{tech.pdk_id}: deep-N-well recipe names undrawable pwell "
            f"layer {rules.pwell_layer!r}"
        )
    if dnwell_key is None:
        raise ValueError(
            f"{tech.pdk_id}: deep-N-well recipe names undrawable layers "
            f"pwell={rules.pwell_layer!r}, dnwell={rules.deep_nwell_layer!r}"
        )

    # The inner body's P+ frame is part of the protected p-well.  Growing the
    # full frame by the tap enclosure also conservatively encloses the inner
    # N-active, which lies still farther inside the opening.
    pwell_box = _grow_bbox(
        tuple(int(value) for value in ring.ring_bbox),
        max(
            int(rules.pwell_enclose_ptap_nm),
            int(rules.pwell_enclose_nactive_nm),
        ),
    )

    mode = str(rules.deep_tap_well_mode).strip().casefold()
    if mode not in {"direct_dnw", "nwell_ring"}:
        raise ValueError(
            f"{tech.pdk_id}: unsupported deep_tap_well_mode={mode!r}"
        )

    # Reuse the generic N+ contact-frame constructor as the deep-well tap.
    # Direct-DNW processes remove its ordinary NWELL marker.  N-well-supported
    # processes retain it and add the foundry-sized support frame below.
    deep_tap = build_guard_ring(
        tech,
        pwell_box,
        well="n",
        nets={"tap": str(deep_nwell_net)},
        extra_clearance_nm=int(rules.ntap_to_pwell_nm),
    )
    ordinary_nwell_key = tuple(deep_tap.layers_drawn.get("nwell", ()))
    if mode == "direct_dnw":
        deep_tap_polys = [
            polygon
            for polygon in deep_tap.polygons
            if (int(polygon.layer), int(polygon.datatype)) != ordinary_nwell_key
        ]
        dnwell_box = _union_bbox(
            _grow_bbox(pwell_box, int(rules.deep_nwell_enclose_pwell_nm)),
            _grow_bbox(
                tuple(int(value) for value in deep_tap.ring_bbox),
                int(rules.deep_nwell_enclose_ntap_nm),
            ),
        )
        nwell_outer = None
    else:
        if not ordinary_nwell_key:
            raise ValueError(
                f"{tech.pdk_id}: nwell_ring deep tap requires a drawable nwell"
            )
        deep_tap_polys = list(deep_tap.polygons)
        # The central hole is the derived isolated p-well. DNW extends beneath
        # the N-well by the PDK-owned inner-overlap, and the N-well continues
        # past DNW by the PDK-owned outer extension. This is the external_pdk/external_pdk
        # foundry topology (DNW surrounded by NW), expressed without a
        # circuit-specific branch.
        dnwell_box = _grow_bbox(
            pwell_box,
            max(
                int(rules.deep_nwell_enclose_pwell_nm),
                int(rules.nwell_inner_overlap_nm),
            ),
        )
        nwell_outer = _union_bbox(
            _grow_bbox(dnwell_box, int(rules.nwell_outer_extension_nm)),
            tuple(int(value) for value in deep_tap.ring_bbox),
        )

        def _frame(key, inner, outer):
            ix0, iy0, ix1, iy1 = (int(value) for value in inner)
            ox0, oy0, ox1, oy1 = (int(value) for value in outer)
            return [
                LayerPolygon(int(key[0]), int(key[1]), Polygon.rectangle(ox0, oy0, ox1, iy0)),
                LayerPolygon(int(key[0]), int(key[1]), Polygon.rectangle(ox0, iy1, ox1, oy1)),
                LayerPolygon(int(key[0]), int(key[1]), Polygon.rectangle(ox0, iy0, ix0, iy1)),
                LayerPolygon(int(key[0]), int(key[1]), Polygon.rectangle(ix1, iy0, ox1, iy1)),
            ]

        deep_tap_polys.extend(_frame(ordinary_nwell_key, pwell_box, nwell_outer))

    outer = None
    if bool(rules.outer_substrate_guard_required):
        outer = build_guard_ring(
            tech,
            nwell_outer if nwell_outer is not None else dnwell_box,
            well="p",
            nets={"tap": str(substrate_net)},
            extra_clearance_nm=int(rules.outer_ptap_to_deep_nwell_nm),
        )

    solid_wells = [
        LayerPolygon(
            layer=int(dnwell_key[0]),
            datatype=int(dnwell_key[1]),
            polygon=Polygon.rectangle(*dnwell_box),
        ),
    ]
    if pwell_key is not None:
        solid_wells.insert(
            0,
            LayerPolygon(
                layer=int(pwell_key[0]),
                datatype=int(pwell_key[1]),
                polygon=Polygon.rectangle(*pwell_box),
            ),
        )
    selector_polys = []
    selector_keys = {}
    for token in tuple(rules.selector_layers):
        key = tech.marker_key(str(token))
        if key is None:
            raise ValueError(
                f"{tech.pdk_id}: deep-N-well selector {token!r} is undrawable"
            )
        selector_polys.append(
            LayerPolygon(
                layer=int(key[0]),
                datatype=int(key[1]),
                polygon=Polygon.rectangle(*pwell_box),
            )
        )
        selector_keys[str(token)] = tuple(int(value) for value in key)
    polygons = (
        list(ring.polygons) + solid_wells + selector_polys + deep_tap_polys
    )
    terminals = dict(ring.terminals)
    terminals["tap"] = replace(terminals["tap"], net=str(body_net))
    terminals["deep_tap"] = replace(
        deep_tap.terminals["tap"], role="deep_tap", net=str(deep_nwell_net)
    )
    layers_drawn = dict(ring.layers_drawn)
    if pwell_key is not None:
        layers_drawn["pwell"] = tuple(int(value) for value in pwell_key)
    layers_drawn["dnwell"] = tuple(int(value) for value in dnwell_key)
    if mode == "nwell_ring":
        layers_drawn["deep_tap_nwell"] = tuple(
            int(value) for value in ordinary_nwell_key
        )
    for token, key in selector_keys.items():
        layers_drawn[f"selector_{token}"] = key
    layers_drawn["deep_tap_metal1"] = tuple(
        int(value) for value in deep_tap.layers_drawn["metal1"]
    )
    cut_count = int(ring.cut_count) + int(deep_tap.cut_count)
    outer_bbox = tuple(int(value) for value in deep_tap.ring_bbox)
    if outer is not None:
        polygons.extend(outer.polygons)
        terminals["substrate_tap"] = replace(
            outer.terminals["tap"],
            role="substrate_tap",
            net=str(substrate_net),
        )
        layers_drawn["substrate_tap_metal1"] = tuple(
            int(value) for value in outer.layers_drawn["metal1"]
        )
        cut_count += int(outer.cut_count)
        outer_bbox = tuple(int(value) for value in outer.ring_bbox)

    return replace(
        ring,
        polygons=polygons,
        terminals=terminals,
        ring_bbox=_union_bbox(
            tuple(ring.ring_bbox),
            dnwell_box,
            nwell_outer if nwell_outer is not None else dnwell_box,
            outer_bbox,
        ),
        cut_count=cut_count,
        layers_drawn=layers_drawn,
    )


def group_guard_rings(
    geometry: Dict[str, CellGeometry],
    placement: Dict[str, _XY],
    intent,
    tech_id: str,
    *,
    extra_clearance_nm: int = 0,
    island_clusters: Optional[List[Tuple[str, ...]]] = None,
    member_well_tie: Optional[Dict[str, Tuple[str, str]]] = None,
    member_ring_pad_top_nm: Optional[Dict[str, int]] = None,
    member_ring_pad_bottom_nm: Optional[Dict[str, int]] = None,
    substrate_net: Optional[str] = None,
    deep_nwell_net: Optional[str] = None,
    isolated_members: Optional[Sequence[str]] = None,
) -> Tuple[List[GroupGuardRing], List[str]]:
    """Build GROUP-granularity guard rings for a placed design.

    Cluster derivation (design topology + the design-sourced placement; no
    reference coordinate is ever read). When a DECLARED ``@guard_ring`` partition
    is in play -- the caller's ``island_clusters`` OR the islands declared on the
    intent (:func:`clay.layout.extract.well_islands.declared_guard_islands`) when
    ``island_clusters`` is not threaded -- each declared island is ONE ring
    cluster (membership fixed by the declaration, fold-invariant); a passive
    (resistor/cap) island still draws a frame on its DECLARED well + tie net (a
    passive has no MOS body-net well to derive from, so the declaration is the
    source of truth). Otherwise clusters are re-derived from the placed geometry:

      * ROWS: transitive y-interval overlap of the placed stamp bboxes;
      * RUNS: within a row, walk the stamps in x order -- a maximal run of
        consecutive same-kind MOS devices is a cluster candidate (any
        intervening non-MOS stamp, e.g. a placed passive, splits the run);
      * TAP-NET CONSISTENCY: a cluster's tap net is its members' common
        ``body`` terminal net (DESIGN data). A same-kind run whose body nets
        disagree is SPLIT at the disagreement and the split is noted --
        honest, never a silently mixed-bulk ring;
      * one ring per cluster, around the union of the members' placed bboxes
        (``clay.layout.guard_ring.build_guard_ring``: nmos cluster -> p
        substrate-tap ring, pmos cluster -> n nwell-tap ring, every dimension
        PDK-rule-resolved);
      * COMPOSITE stamps: a geometry entry absent from the intent params but
        carrying ``composite_members`` (an interdigitated matched-group
        stamp) participates like any MOS stamp -- its kind/tap-net identity
        resolves through its MEMBER devices' design data (common kind +
        common ``body`` net; a disagreeing composite is refused with a
        note). Its member devices are then ringed THROUGH the stamp, not
        reported missing.

    ``topo_sketch.mirror_groups`` is consumed READ-ONLY for honesty: a mirror
    group whose members land in more than one cluster is reported in the
    notes (the topology wanted them matched; the placement put them apart).

    Returns ``(rings, notes)``. ``rings`` are sorted bottom-up then left-to-
    right; ``notes`` also records every intent MOS skipped for lack of a
    stamp/position. Ring/ring and ring/non-member spacing is the CALLER's
    placement account: reserve the room by solving the placement with
    :func:`group_ring_padded_extents` sizes (the same sizes pipeline every
    other footprint flows through -- no second path).
    """
    from clay.layout.guard_ring import build_guard_ring
    from clay.layout.tech import LayoutTech

    params = _intent_device_params(intent)
    body_nets = {
        d.name: str((getattr(d, "nets", {}) or {}).get("body", "") or "")
        for d in getattr(intent, "devices", []) or []
    }

    notes: List[str] = []

    # MOS composite well/tie comes from all members' common kind/body.
    # Typed passive ownership is resolved against explicit declarations below.
    eff_kind: Dict[str, str] = {}
    eff_body: Dict[str, str] = {}
    composited: Dict[str, str] = {}  # member device name -> composite stamp
    for name, cg in geometry.items():
        if name in params:
            eff_kind[name] = _norm_kind(params[name][0])
            eff_body[name] = body_nets.get(name, "")
            continue
        members = tuple(getattr(cg, "composite_members", ()) or ())
        if not members:
            continue
        kinds = {
            _norm_kind(params[m][0]) for m in members if m in params
        }
        bodies = {body_nets.get(m, "") for m in members if m in params}
        if (
            len(members) == sum(1 for m in members if m in params)
            and len(kinds) == 1
            and next(iter(kinds)) in _RING_WELL_FOR_KIND
            and len(bodies) == 1
        ):
            eff_kind[name] = next(iter(kinds))
            eff_body[name] = next(iter(bodies))
            for m in members:
                composited[m] = name
        else:
            notes.append(
                f"composite stamp {name}: members {list(members)} disagree "
                f"on kind/body (kinds={sorted(kinds)} bodies={sorted(bodies)})"
                f"; not ringed"
            )

    placed = _placed_stamp_bboxes_nm(geometry, placement)
    placed_layers = _placed_stamp_layer_bboxes_nm(geometry, placement)
    for name in sorted(params):
        if _norm_kind(params[name][0]) not in _RING_WELL_FOR_KIND:
            continue
        if name in composited:
            continue  # ringed through its composite stamp
        if name not in geometry:
            notes.append(f"MOS {name}: no geometry stamp; not ringed")
        elif name not in placed:
            notes.append(f"MOS {name}: no placement position; not ringed")

    # ---- clusters -----------------------------------------------------------
    # ISLAND-MEMBERSHIP clustering (the contract anchor_ota names: "cluster
    # derivation must consume the placement's band structure"): when the caller
    # passes ``island_clusters`` (the well-island grouping the placement was
    # BANDED by -- pure topology, kind+body net), each island is ONE ring cluster
    # whose MEMBERSHIP is fixed by topology and whose extent is the union of its
    # members' PLACED bboxes. This is FOLD-INVARIANT: re-folding a device changes
    # placed bboxes (extent) but never the cluster SET, so the ring count/overlap
    # cannot flip under P1 folding (the bug that re-derivation from placed-bbox
    # y-gaps produced). Default ``None`` keeps the placed-geometry _bands_by_y_gap
    # derivation (gf180 direct callers / no-island intents), byte-identical.
    clusters: List[Tuple[int, List[Tuple[str, Tuple[int, int, int, int]]]]] = []
    # DECLARED-island rings for PASSIVE clusters (a resistor/cap @guard_ring
    # island, e.g. RZ/RS): (row, run, declared_well, declared_tie). A passive
    # carries no MOS body-net well identity, so the MOS filter below drops it and
    # it would draw NO frame; the DECLARED @guard_ring polarity + tie net are the
    # source of truth for its ring (MOS clusters keep deriving well from kind).
    passive_clusters: List[
        Tuple[int, List[Tuple[str, Tuple[int, int, int, int]]], str, str]
    ] = []
    # DECLARED-island specs = (members, well, tie). Honor the caller's
    # ``island_clusters`` (pairing each member tuple with the declared well/tie,
    # matched by member) OR, when the caller threads no param, fall back to the
    # islands DECLARED on the intent (:func:`declared_guard_islands`) so a path
    # that does not pass ``island_clusters`` STILL refines to the declared ring
    # boundaries + materializes passive rings. ``None`` -> no declared island in
    # play -> the placed-geometry ``_bands_by_y_gap`` derivation below
    # (byte-identical for a cell that declares no @guard_ring).
    from clay.layout.extract.well_islands import declared_guard_islands

    _declared = declared_guard_islands(intent)
    _wt_of_member: Dict[str, Tuple[str, str]] = {}
    for _mem, _well, _tie in _declared:
        for _m in _mem:
            _wt_of_member.setdefault(str(_m), (_well, _tie))
    # Explicit declared well/tie from the caller's @guard_ring records take
    # precedence: the BRIDGED intent group_guard_rings receives (route_cell's
    # ``_materialize_guard_rings``) cannot carry the guard-ring declaration on its
    # ``__slots__``, so a PASSIVE island (no MOS kind to derive a well from) would
    # otherwise draw no frame ("declared well '' is not p/n"). Pure declared data
    # (polarity + tie net); no coordinate is read.
    for _m, (_well, _tie) in (member_well_tie or {}).items():
        _wt_of_member[str(_m)] = (str(_well or ""), str(_tie or ""))
    if island_clusters is not None:
        island_specs: Optional[List[Tuple[Tuple[str, ...], str, str]]] = []
        for _members in island_clusters:
            _dw, _dt = "", ""
            for n in _members:
                if str(n) in _wt_of_member:
                    _dw, _dt = _wt_of_member[str(n)]
                    break
            island_specs.append((tuple(_members), _dw, _dt))
    elif _declared:
        island_specs = [(tuple(m), w, t) for m, w, t in _declared]
    else:
        island_specs = None

    from clay.layout.extract.passive_ring_ownership import current_passive_ring_owners
    composited.update(current_passive_ring_owners(
        geometry, placement, intent, tech_id, island_specs=island_specs,
        member_well_tie=_wt_of_member))

    if island_specs is not None:
        for ci, (member_names, dwell, dtie) in enumerate(island_specs):
            # A composited island member is ringed THROUGH its composite stamp
            # (the stamp -- not the member name -- is the placed footprint and
            # carries the cluster's kind/body via its member design data). Map
            # each composited member to its stamp key so the island ring
            # ENCLOSES the composite stamp; otherwise the stamp lands unringed
            # inside a foreign island ring and its S/D OD / internal-signal
            # contacts merge with that ring's tap (the residual O-segment RED:
            # composite pairs unextractable, internal net shorts the tap).
            resolved: List[str] = []
            seen_keys: set = set()
            for n in member_names:
                key = composited.get(n, n)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                resolved.append(key)
            members = [
                (n, placed[n]) for n in resolved
                if n in placed
                and eff_kind.get(n, "") in _RING_WELL_FOR_KIND
            ]
            # one ring per (body-net) within the island (an island groups one
            # bulk net by construction, but split defensively if a caller mixes)
            by_body: Dict[str, List[Tuple[str, Tuple[int, int, int, int]]]] = {}
            for n, bb in members:
                by_body.setdefault(eff_body.get(n, ""), []).append((n, bb))
            for _body in sorted(by_body):
                run = sorted(by_body[_body],
                             key=lambda kv: ((kv[1][0] + kv[1][2]) // 2, kv[0]))
                if run:
                    clusters.append((ci, run))
            # PASSIVE members of this declared island: dropped by the MOS filter
            # above, ringed here as ONE cluster on the DECLARED well + tie.
            passive = [
                (n, placed[n]) for n in resolved
                if n in placed
                and eff_kind.get(n, "") not in _RING_WELL_FOR_KIND
            ]
            if passive:
                prun = sorted(passive,
                              key=lambda kv: ((kv[1][0] + kv[1][2]) // 2, kv[0]))
                passive_clusters.append((ci, prun, dwell, dtie))
    else:
        for row_i, row in enumerate(_bands_by_y_gap(placed)):
            row_sorted = sorted(
                row, key=lambda kv: ((kv[1][0] + kv[1][2]) // 2, kv[0])
            )
            run: List[Tuple[str, Tuple[int, int, int, int]]] = []
            run_kind = ""
            run_body = ""

            def _close():
                nonlocal run
                if run:
                    clusters.append((row_i, run))
                run = []

            for name, bb in row_sorted:
                kind = eff_kind.get(name, "")
                if kind not in _RING_WELL_FOR_KIND:
                    _close()  # a non-MOS stamp (passive) breaks adjacency
                    continue
                body = eff_body.get(name, "")
                if run and kind == run_kind and body != run_body:
                    notes.append(
                        f"row {row_i}: same-kind run split at {name} -- body net "
                        f"{body!r} differs from {run_body!r} (tap nets must agree "
                        f"inside one ring; split, not mixed)"
                    )
                    _close()
                elif run and kind != run_kind:
                    _close()
                run.append((name, bb))
                run_kind, run_body = kind, body
            _close()

    # ---- honest mirror-group account (topo_sketch read-only) ---------------
    cluster_of: Dict[str, int] = {}
    for ci, (_row, members) in enumerate(clusters):
        for name, _bb in members:
            cluster_of[name] = ci
    try:
        from clay.layout.extract.topo_sketch import mirror_groups as _mg

        for g in _mg(intent):
            # a composited member is judged by its composite stamp's cluster
            idxs = {
                cluster_of[s]
                for s in (composited.get(m, m) for m in g.members)
                if s in cluster_of
            }
            if len(idxs) > 1:
                notes.append(
                    f"mirror group {'/'.join(g.members)} spans {len(idxs)} "
                    f"ring clusters (placement separated topology-matched "
                    f"devices; rings follow the placement honestly)"
                )
    except Exception as exc:  # pragma: no cover - mirror notes are advisory
        notes.append(f"mirror-group account unavailable: {exc!r}")

    # ---- one ring per cluster ----------------------------------------------
    tech = LayoutTech.resolve(tech_id)
    isolated = set(
        str(name)
        for name in (
            isolated_members
            if isolated_members is not None
            else (getattr(intent, "well_isolated_devices", ()) or ())
        )
    )
    substrate_net = str(
        substrate_net
        if substrate_net is not None
        else getattr(intent, "substrate_net", tech.substrate_net())
    )
    deep_nwell_net = str(
        deep_nwell_net
        if deep_nwell_net is not None
        else getattr(intent, "deep_nwell_net", "")
    )
    rings: List[GroupGuardRing] = []
    for row_i, members in clusters:
        names = tuple(n for n, _bb in members)
        kind = eff_kind[names[0]]
        well = _RING_WELL_FOR_KIND[kind]
        body = eff_body.get(names[0], "")
        bottom_pad = _cluster_ring_pad_bottom(
            names, member_ring_pad_bottom_nm
        )
        top_pad = _cluster_ring_pad_top(names, member_ring_pad_top_nm)
        inner = (
            min(bb[0] for _n, bb in members),
            min(bb[1] for _n, bb in members) - bottom_pad,
            max(bb[2] for _n, bb in members),
            max(bb[3] for _n, bb in members) + top_pad,
        )
        from clay.layout.device_cell import _resolve_layer_keys

        layer_keys = _resolve_layer_keys(tech, kind)

        def protected_union(key: _LayerKey, label: str, *, required: bool = True):
            boxes = []
            for name in names:
                box = placed_layers.get(name, {}).get(key)
                if box is None:
                    if not required:
                        continue
                    raise ValueError(
                        f"guard-ring member {name!r} lacks {label} bbox authority"
                    )
                boxes.append(box)
            if not boxes:
                return None
            return (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes) - bottom_pad,
                max(box[2] for box in boxes),
                max(box[3] for box in boxes) + top_pad,
            )

        protected_active = protected_union(layer_keys.active, "active")
        protected_poly = protected_union(layer_keys.poly, "poly")
        protected_metal1 = protected_union(layer_keys.metal1, "metal1")
        protected_same_implant = (
            protected_union(layer_keys.tap_implant, "same implant", required=False)
            if bool(tech.bulk_topology().get("implant_distinguished", True)) else None
        )
        ring = build_guard_ring(
            tech,
            inner,
            well=well,
            nets={"tap": body} if body else None,
            extra_clearance_nm=extra_clearance_nm,
            shared_well=(kind == "pmos" and well == "n"),
            protected_active_bbox=protected_active,
            protected_poly_bbox=protected_poly,
            protected_metal1_bbox=protected_metal1,
            protected_same_implant_bbox=protected_same_implant,
        )
        ring = _materialize_shared_mos_well(ring, kind=kind, well=well)
        isolated_here = isolated.intersection(names)
        if isolated_here:
            if isolated_here != set(names):
                raise ValueError(
                    "@well_isolated device cannot share one well ring with "
                    f"non-isolated members: {names}"
                )
            if kind != "nmos" or well != "p":
                raise ValueError(
                    f"deep-N-well lowering currently requires an NMOS p-well; "
                    f"got kind={kind!r}, well={well!r}, members={names}"
                )
            ring = _materialize_deep_nwell_bundle(
                ring,
                tech=tech,
                body_net=body,
                deep_nwell_net=deep_nwell_net,
                substrate_net=substrate_net,
            )
        rings.append(
            GroupGuardRing(
                members=names,
                kind=kind,
                well=well,
                tap_net=str(ring.terminals["tap"].net),
                row=row_i,
                ring=ring,
            )
        )

    # ---- one ring per declared PASSIVE cluster (resistor/cap @guard_ring) ----
    # A passive has no MOS body-net well, so the DECLARED @guard_ring polarity +
    # tie net drive the ring. An unsupported / undeclared well is skipped with an
    # honest note (never a fabricated frame).
    for row_i, members, dwell, dtie in passive_clusters:
        names = tuple(n for n, _bb in members)
        if dwell not in ("p", "n"):
            notes.append(
                f"passive @guard_ring cluster {list(names)}: declared well "
                f"{dwell!r} is not p/n; not ringed (no MOS well to derive from)"
            )
            continue
        inner = (
            min(bb[0] for _n, bb in members),
            min(bb[1] for _n, bb in members),
            max(bb[2] for _n, bb in members),
            max(bb[3] for _n, bb in members),
        )
        ring = build_guard_ring(
            tech,
            inner,
            well=dwell,
            nets={"tap": dtie} if dtie else None,
            extra_clearance_nm=extra_clearance_nm,
        )
        rings.append(
            GroupGuardRing(
                members=names,
                kind="passive",
                well=dwell,
                tap_net=str(ring.terminals["tap"].net),
                row=row_i,
                ring=ring,
            )
        )

    rings.sort(key=lambda r: (r.row, r.ring.ring_bbox[0], r.members))
    return rings, notes


def _group_ring_base_clearance_nm(tech) -> int:
    """Rule-derived ring clearance shared by every well polarity.

    This excludes N-well spacing because that rule applies only when both
    neighbouring blocks actually carry N-well.  Keeping the components split
    lets the block materializer lower the N-well obligation pair-selectively
    instead of taxing a MIM-cap/P-sub-ring boundary with an unrelated rule.
    """
    from clay.layout.device_cell import _active_name, _rules_from_tech, _space_or

    r = _rules_from_tech(tech)
    iw = tech.implant_well()
    return int(
        max(
            _space_or(tech, _active_name(tech), r.channel_min_width),
            r.metal1_min_space,
            int(iw.pplus_min_space),
            int(iw.nplus_min_space),
            r.implant_to_opp_comp,
        )
    )


def _group_ring_nwell_clearance_nm(tech) -> int:
    """Rule-derived spacing owed specifically by two N-well-bearing blocks."""

    from clay.layout.device_cell import _rules_from_tech, _space_or

    r = _rules_from_tech(tech)
    return int(_space_or(tech, "nwell", r.nwell_min_width))


def _group_ring_clearance_nm(tech) -> int:
    """Worst-case ring clearance retained for symmetric footprint reservers.

    Callers that know the exact neighbouring block well polarities should use
    :func:`_group_ring_base_clearance_nm` plus
    :func:`_group_ring_nwell_clearance_nm` pair-selectively.  Legacy symmetric
    footprint paths intentionally keep this worst-case aggregate.
    """

    return max(
        _group_ring_base_clearance_nm(tech),
        _group_ring_nwell_clearance_nm(tech),
    )


def _ring_extension_nm(tech, well: str) -> int:
    """The ring's uniform outward extension (ring_bbox beyond inner_bbox, per
    side) at rule scale, measured from a probe build -- never assumed."""
    from clay.layout.guard_ring import build_guard_ring

    probe = (0, 0, 200000, 200000)
    ring = build_guard_ring(tech, probe, well=well)
    return int(probe[0] - ring.ring_bbox[0])


def _symmetric_ring_footprint_nm(
    inner_bbox: Tuple[int, int, int, int],
    ring_bbox: Tuple[int, int, int, int],
    *,
    half_clearance_nm: int,
) -> Tuple[int, int]:
    """Reserve an asymmetric exact ring in a center-based size interface."""

    for label, box in (("inner_bbox", inner_bbox), ("ring_bbox", ring_bbox)):
        if (
            type(box) is not tuple
            or len(box) != 4
            or any(type(value) is not int for value in box)
        ):
            raise TypeError(f"{label} must be an exact integer bbox")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{label} must be non-degenerate")
    if type(half_clearance_nm) is not int or half_clearance_nm < 0:
        raise TypeError("half_clearance_nm must be a non-negative exact integer")
    if not (
        ring_bbox[0] <= inner_bbox[0]
        and ring_bbox[1] <= inner_bbox[1]
        and ring_bbox[2] >= inner_bbox[2]
        and ring_bbox[3] >= inner_bbox[3]
    ):
        raise ValueError("ring_bbox must contain inner_bbox")
    x_extension = max(
        inner_bbox[0] - ring_bbox[0],
        ring_bbox[2] - inner_bbox[2],
    )
    y_extension = max(
        inner_bbox[1] - ring_bbox[1],
        ring_bbox[3] - inner_bbox[3],
    )
    return (
        inner_bbox[2] - inner_bbox[0]
        + 2 * (x_extension + half_clearance_nm),
        inner_bbox[3] - inner_bbox[1]
        + 2 * (y_extension + half_clearance_nm),
    )


def _deep_nwell_extension_nm(tech) -> int:
    """Uniform outward extension of the PDK-owned full triple-well bundle."""

    from clay.layout.guard_ring import build_guard_ring

    probe = (0, 0, 200000, 200000)
    base = build_guard_ring(tech, probe, well="p", nets={"tap": "body"})
    bundle = _materialize_deep_nwell_bundle(
        base,
        tech=tech,
        body_net="body",
        deep_nwell_net="positive",
        substrate_net="substrate",
    )
    return int(probe[0] - bundle.ring_bbox[0])


def group_ring_padded_extents(
    geometry: Dict[str, CellGeometry],
    intent,
    *,
    tech_id: str,
) -> Dict[str, _XY]:
    """Like :func:`stamp_extents`, but pad every MOS footprint with exact
    layer-aware GROUP-ring room. The ring is replayed from the stamp's active,
    poly, and M1 extents; the larger left/right and bottom/top extension is
    reserved symmetrically, plus half the worst-case ring-to-neighbour clearance
    (:func:`_group_ring_clearance_nm`). Two adjacent padded footprints from
    different clusters therefore leave enough room for both rings and the PDK
    spacing. Same-cluster internal gaps widen too -- the honest area cost of a
    per-device-symmetric sizes pipeline (the alternative, per-face pads, is a
    placement-contract change owned by the integrating flow).

    This changes ONLY the reserved placement footprint (the same ``sizes``
    currency every footprint already flows through); the drawn stamps are
    untouched and nothing here reads a reference coordinate.
    """
    from clay.layout.tech import LayoutTech

    params = _intent_device_params(intent)
    tech = LayoutTech.resolve(tech_id)
    half_clear = (_group_ring_clearance_nm(tech) + 1) // 2

    local_layers = _placed_stamp_layer_bboxes_nm(
        geometry, {str(name): (0.0, 0.0) for name in geometry}
    )
    isolated = set(str(name) for name in (getattr(intent, "well_isolated_devices", ()) or ()))
    out: Dict[str, _XY] = {}
    for name, geom in geometry.items():
        if not geom.polys_by_layer:
            continue
        w, h = stamp_extent(geom)
        if w <= 0.0 or h <= 0.0:
            continue
        kind = _norm_kind(params[name][0]) if name in params else ""
        well = _RING_WELL_FOR_KIND.get(kind)
        if well is None:
            out[name] = (w, h)
            continue
        if name in isolated:
            pad = (_deep_nwell_extension_nm(tech) + half_clear) / 1000.0
            out[name] = (w + 2.0 * pad, h + 2.0 * pad)
            continue
        from clay.layout.device_cell import _resolve_layer_keys
        from clay.layout.guard_ring import build_guard_ring

        inner = _stamp_bbox_nm(geom)
        if inner is None:
            raise ValueError(f"guard-ring member {name!r} has no full bbox authority")
        keys = _resolve_layer_keys(tech, kind)
        active_bbox = local_layers.get(name, {}).get(keys.active)
        poly_bbox = local_layers.get(name, {}).get(keys.poly)
        metal1_bbox = local_layers.get(name, {}).get(keys.metal1)
        if active_bbox is None or poly_bbox is None or metal1_bbox is None:
            raise ValueError(
                f"guard-ring member {name!r} lacks active/poly/M1 bbox authority"
            )
        ring = build_guard_ring(
            tech,
            inner,
            well=well,
            shared_well=(kind == "pmos" and well == "n"),
            protected_active_bbox=active_bbox,
            protected_poly_bbox=poly_bbox,
            protected_metal1_bbox=metal1_bbox,
            protected_same_implant_bbox=(
                local_layers.get(name, {}).get(keys.tap_implant)
                if bool(tech.bulk_topology().get("implant_distinguished", True)) else None
            ),
        )
        footprint_nm = _symmetric_ring_footprint_nm(
            inner, ring.ring_bbox, half_clearance_nm=half_clear
        )
        out[name] = (
            footprint_nm[0] / 1000.0,
            footprint_nm[1] / 1000.0,
        )
    return out


def group_ring_tap_terminals(rings: List[GroupGuardRing]) -> List[dict]:
    """The routable tap-access list for a set of group rings: one entry per
    ring with the cluster members, the tap NET (the design body net), the
    absolute-nm M1 bbox of the ring's tap segment, its single canonical
    integer-nm access point, and the PDK M1 layer key."""
    out: List[dict] = []
    for ring_index, r in enumerate(rings):
        for terminal_name in ("tap", "deep_tap", "substrate_tap"):
            t = r.ring.terminals.get(terminal_name)
            if t is None:
                continue
            bbox_nm = tuple(int(v) for v in t.bbox)
            out.append(
                {
                    "ring_index": ring_index,
                    "terminal": terminal_name,
                    "members": r.members,
                    "net": str(t.net),
                    "bbox_nm": bbox_nm,
                    "access_xy_nm": (
                        (bbox_nm[0] + bbox_nm[2]) // 2,
                        (bbox_nm[1] + bbox_nm[3]) // 2,
                    ),
                    "layer": (
                        int(r.ring.layers_drawn["metal1"][0]),
                        int(r.ring.layers_drawn["metal1"][1]),
                    ),
                    "well": r.well if terminal_name == "tap" else "",
                }
            )
    return out


def parametric_capacitor_geometry(
    tech_id: str,
    value_farads: float,
    *,
    m: int = 1,
    model: str = "",
    nets=None,
    mim_option: Optional[str] = None,
    metal_level: str = "5LM",
    cap_option: str = "1",
    pex_backend: Optional[str] = None,
) -> CellGeometry:
    """The same native capacitor bank used by placement, pins, and GDS emission.

    Preserve both multiplicity and the declared MIM level. The older density-only
    single-plate preview was not the actual device and could put the clear-side
    oracle and the router's obstacles on different artwork.
    """
    from clay.layout import passive_device_cell as _pdc
    from clay.layout.tech import LayoutTech

    kw: Dict[str, object] = {}
    if tech_id.lower() == "gf180mcu" and (mim_option is not None or pex_backend is not None):
        from clay.pdk.gf180mcu import resolve_gf180_mim_config

        opt, cap_opt, m_lvl = resolve_gf180_mim_config(
            mim_option=mim_option,
            cap_option=cap_option,
            metal_level=metal_level,
            pex_backend=pex_backend,
        )
        kw["mim_option"] = opt
        kw["metal_level"] = m_lvl
        kw["cap_option"] = cap_opt
    cell = _pdc.build_capacitor_for_tech(
        LayoutTech.resolve(tech_id), float(value_farads), m=int(m),
        model=str(model), nets=nets, **kw,
    )
    vertices = [(int(v.x), int(v.y))
                for polygon in cell.polygons for v in polygon.polygon.vertices]
    if not vertices:
        raise ValueError(f"{tech_id}: build_capacitor_for_tech produced no polygons")
    cx = (min(x for x, _ in vertices) + max(x for x, _ in vertices)) // 2
    cy = (min(y for _, y in vertices) + max(y for _, y in vertices)) // 2
    polys_by_layer: Dict[_LayerKey, List[_Ring]] = {}
    for polygon in cell.polygons:
        key = (int(polygon.layer), int(polygon.datatype))
        polys_by_layer.setdefault(key, []).append([
            ((int(v.x) - cx) / 1000.0, (int(v.y) - cy) / 1000.0)
            for v in polygon.polygon.vertices
        ])
    cg = CellGeometry(
        name="", kind="cap", ref_centroid=(0.0, 0.0),
        polys_by_layer=polys_by_layer, n_instances=int(m),
        finger_cells=["build_capacitor_for_tech"], clean=True,
        reason=(f"native MIM bank {value_farads:.6g} F, m={m}, model={model!r} "
                f"(passive_device_cell.build_capacitor_for_tech on {tech_id}; "
                "the placement/pin/gate constructor)"),
    )
    cg.source = "passive_param"
    return cg


def parametric_resistor_geometry(tech_id: str, value_ohms: float) -> CellGeometry:
    """A poly-resistor :class:`CellGeometry` stamp from the PARAMETRIC constructor
    (:func:`clay.layout.passive_device_cell.build_resistor`) -- the SAME cell the
    pin extractor (``route.pins._build_passive_cell``), the placement size source
    (``route_migrate.external_pdk``) and the lvs gate's geometry-absent
    fallback emission build. The polygons are recentred on the cell's drawn-extent
    centre (the ``pins._geom_extent_center_nm`` frame, floored to integer nm so the
    nm->um->nm round trip is exact), so the stamp is centroid-relative like every
    other geometry-dict entry.

    SINGLE SOURCE (the router-obstacle-blindness root fix, named 2026-06-12): with
    this stamp in the geometry dict, placement sizes, the router's net-aware M1
    obstacles (``model.device_m1_conductors``), the pin landing snap and the gate's
    GDS emission all consume ONE construction -- the gate's own passive-emission
    block skips any resistor whose name is in the dict.

    Raises whatever the constructor raises (e.g. missing sab/res_mk marker layers
    on a PDK that cannot draw a poly resistor) -- the caller keeps honest absence.
    """
    from clay.layout import passive_device_cell as _pdc
    from clay.layout.tech import LayoutTech

    tech = LayoutTech.resolve(tech_id)
    cell = _pdc.build_resistor(tech, float(value_ohms))
    xs: List[int] = []
    ys: List[int] = []
    for lp in cell.polygons:
        for v in lp.polygon.vertices:
            xs.append(int(v.x))
            ys.append(int(v.y))
    if not xs:
        raise ValueError(f"{tech_id}: build_resistor produced no polygons")
    # drawn-extent centre, FLOORED to integer nm: build_resistor coordinates are
    # on-grid ints, so integer recentring keeps every ring vertex an exact nm
    # value (a half-nm centre would put .0005 um residues through banker's
    # rounding at the emit boundary -> +/-1nm pad edge wobble).
    cx = (min(xs) + max(xs)) // 2
    cy = (min(ys) + max(ys)) // 2
    polys_by_layer: Dict[_LayerKey, List[_Ring]] = {}
    for lp in cell.polygons:
        key = (int(lp.layer), int(lp.datatype))
        ring = [
            (round((int(v.x) - cx) / 1000.0, 4), round((int(v.y) - cy) / 1000.0, 4))
            for v in lp.polygon.vertices
        ]
        polys_by_layer.setdefault(key, []).append(ring)
    cg = CellGeometry(
        name="",
        kind="res",
        ref_centroid=(0.0, 0.0),  # a stamp has no reference position
        polys_by_layer=polys_by_layer,
        n_instances=1,
        finger_cells=["build_resistor"],
        clean=True,
        reason=(
            f"parametric poly resistor {value_ohms:.6g} ohm "
            f"(passive_device_cell.build_resistor on {tech_id}; the pin/gate "
            f"constructor -- single source)"
        ),
    )
    cg.source = "passive_param"
    return cg


# --------------------------------------------------------------------------- #
# 2. real_device_geometry: vendor where available, build_device otherwise
# --------------------------------------------------------------------------- #
def _intent_device_params(intent) -> Dict[str, Tuple[str, float, float, int]]:
    """``name -> (kind, W_um, L_um, M)`` for every MOS in ``intent.devices``."""
    out: Dict[str, Tuple[str, float, float, int]] = {}
    for d in getattr(intent, "devices", []) or []:
        out[d.name] = (
            _norm_kind(getattr(d, "kind", "")),
            float(getattr(d, "W", 0.0)),
            float(getattr(d, "L", 0.0)),
            int(getattr(d, "M", 1) or 1),
        )
    return out


def _intent_device_oxides(intent) -> Dict[str, str]:
    """Preserve the same explicit/kind oxide contract as the source reference."""
    from clay.layout.annotate_to_solver import kind_polarity_oxide

    out: Dict[str, str] = {}
    for device in getattr(intent, "devices", []) or []:
        explicit = getattr(device, "oxide", None)
        if explicit not in (None, "", "thin", "thick"):
            raise ValueError(f"unknown MOS oxide {explicit!r} for {device.name}")
        _, kind_oxide = kind_polarity_oxide(getattr(device, "kind", ""))
        out[device.name] = (
            "thick" if explicit == "thick" or kind_oxide == "thick" else "thin"
        )
    return out


def real_device_geometry(
    intent,
    *,
    ref_gds: Optional[str] = None,
    tech_id: str = "gf180mcu",
    fallback: bool = True,
    max_rows: int = 4,
    passive_ref_gds: Optional[str] = None,
    native_passive: bool = False,
    guard_rings=False,
    row_height_target_um: Optional[float] = None,
    contact_rows: bool = False,
    mim_option: Optional[str] = None,
    metal_level: str = "5LM",
    cap_option: str = "1",
    pex_backend: Optional[str] = None,
) -> Dict[str, CellGeometry]:
    """Resolve a CLEAN geometry stamp for every device in ``intent``.

    Strategy:
      * If ``ref_gds`` is given, start from the VENDOR capture
        (:func:`clay.layout.extract.library_device.extract_device_geometry`) -- the
        reference's own resolved external_pdk PCell polygons, vendor-exact.
      * For every device whose vendor geometry is MISSING or ``clean=False`` (a
        band-only / MOScap region with no captured finger instance), if
        ``fallback`` is True, REPLACE it with a parametric
        :func:`build_device_geometry` stamp built from the device's
        ``(kind, W, L, M)``. (With ``ref_gds=None`` -- e.g. a non-external_pdk target PDK
        like external_pdk -- EVERY MOS comes from build_device.)

    Each returned :class:`CellGeometry` carries ``.source in {'vendor','build_device'}``
    so the caller can report the provenance split.

    HONEST SCOPE: ``build_device`` models only MOS. A PASSIVE (cap/res) with no
    vendor capture has NO parametric fallback -- it is left out and its absence is
    discoverable (it simply does not appear in the returned dict), never faked. On
    the external_pdk reference every passive is vendor-clean, so the only fallback is the one
    band-only MOS (MM16); the result is 17/17 clean. On a target PDK with
    ``ref_gds=None`` the passives are the documented residual (MOS migrate cleanly,
    passive geometry migration is a separate concern).

    GUARD RINGS (OPT-IN, default OFF = byte-identical output). The parameter
    face is ``False | True | 'device' | 'group'`` (``True`` aliases
    ``'device'`` for back-compat; anything else raises):

      * ``'device'`` -- every MOS stamp is wrapped by a PDK-rule-driven guard
        ring (:func:`attach_guard_ring`): nmos -> p substrate-tap ring, pmos
        -> n nwell-tap ring, the ring's tap net derived from the DESIGN data
        (the device's ``body`` terminal net -- never a rail-name convention;
        e.g. a cascode pmos whose body rides an internal net gets THAT net).
        The ring polygons join the device's own stamp, so the stamp extent
        grows and the existing sizes pipeline (:func:`stamp_extents` ->
        placement) reserves the ring room automatically.
      * ``'group'`` -- the stamps stay BARE (byte-identical to off): group
        rings need the solved placement (a stamp has no position), so the
        integrating flow builds them AFTER placement via
        :func:`group_guard_rings` (ring room reserved by solving against
        :func:`group_ring_padded_extents` sizes -- the same sizes pipeline).

    Passives are never ringed (the tap recipe is a MOS bulk contract; a
    passive ring is a separate capability).

    ``row_height_target_um`` (OPT-IN; default ``None`` is byte-identical):
    forwarded to :func:`build_device_geometry` for every BUILD_DEVICE
    fallback stamp -- the design row-height fold target (see there). Vendor
    captures are real PCell polygons and are never refolded; the target
    applies only where Clay synthesises the geometry itself.
    """
    ring_mode = _guard_ring_mode(guard_rings)
    params = _intent_device_params(intent)
    oxides = _intent_device_oxides(intent)

    out: Dict[str, CellGeometry] = {}

    # --- vendor pass (when a reference GDS is available) --------------------
    if ref_gds is not None:
        from clay.layout.extract import library_device as LD

        vendor = LD.extract_device_geometry(intent, ref_gds)
        for name, geom in vendor.items():
            if geom.clean and geom.polys_by_layer:
                # tag provenance without mutating the captured polygons.
                geom.source = "vendor"
                out[name] = geom
            # a dirty vendor capture is intentionally NOT added here; it is handled
            # by the fallback pass below (or left out if fallback is off).

    # --- fallback pass: fill every device still missing a clean stamp -------
    if fallback:
        # the set of names we still owe a geometry: every tracked device/passive
        # name in the intent's bbox set (the universe of devices) not yet clean.
        universe = list(getattr(intent, "device_bboxes", {}) or {})
        if not universe:
            # no correspondence run -> fall back to the netlist device/passive names.
            universe = [d.name for d in getattr(intent, "devices", [])] + [
                p.name for p in getattr(intent, "passives", [])
            ]
        for name in universe:
            if name in out:
                continue
            p = params.get(name)
            if p is None:
                # a passive (or untyped) device: build_device cannot synthesise it.
                # Leave it out honestly rather than fabricate geometry.
                continue
            kind, W, L, M = p
            if _norm_kind(kind) not in _BUILDABLE_KINDS:
                continue
            cg = build_device_geometry(
                tech_id, kind, W, L, M, max_rows=max_rows,
                row_height_target_um=row_height_target_um,
                nets=_device_nets_for(intent, name),
                contact_rows=contact_rows,
                oxide=oxides[name],
            )
            cg.name = name
            out[name] = cg

    # --- native passive pass: synthesise TARGET-PDK cap/res geometry from VALUES ----
    # Preferred over the reference proxy: capacitor banks use the same native
    # value/multiplicity/model constructor as placement, pins, and gate emission.
    # Resistors use the target PDK's tabulated poly sheet resistance. Falls
    # THROUGH to the reference proxy for any passive the builder cannot synthesise
    # (unsupported PDK -> KeyError), never fabricating geometry on unknown layers.
    if native_passive:
        from clay.layout import passive_cell as PC

        for p in getattr(intent, "passives", []) or []:
            if p.name in out:
                continue
            val = getattr(p, "value", None)
            if val is None:
                continue
            k = _norm_kind(getattr(p, "kind", ""))
            try:
                if k == "cap":
                    p_mim_opt = (
                        mim_option
                        or getattr(p, "mim_option", None)
                        or getattr(intent, "mim_option", None)
                        or getattr(intent, "cap_option", None)
                    )
                    p_pex = pex_backend or getattr(intent, "pex_backend", None)
                    p_lvl = getattr(p, "metal_level", None) or getattr(intent, "metal_level", None) or metal_level
                    p_cap_opt = cap_option if cap_option != "1" else (getattr(p, "cap_option", None) or "1")
                    cg = parametric_capacitor_geometry(
                        tech_id, float(val), m=int(getattr(p, "M", 1) or 1),
                        model=str(getattr(p, "model", "") or ""),
                        nets=getattr(p, "nets", {}) or {},
                        mim_option=p_mim_opt,
                        metal_level=p_lvl,
                        cap_option=p_cap_opt,
                        pex_backend=p_pex,
                    )
                elif k == "res":
                    cg = PC.build_poly_res(tech_id, float(val))
                else:
                    continue
                cg.source = "passive_synth"
            except (KeyError, ValueError):
                # No tabulated passive_cell facts for this PDK (e.g. open_pdk).
                # A RESISTOR still has the parametric constructor
                # (passive_device_cell.build_resistor -- the same cell the pin
                # extractor and the gate emission build): use it, so the RR
                # artwork is a geometry-dict stamp = a net-aware router obstacle
                # (the router-obstacle-blindness root fix). A CAP keeps the
                # documented honest absence (the gate's cap-cell emission owns
                # cap artwork; a cap stamp here would double-draw it).
                if k != "res":
                    continue
                try:
                    cg = parametric_resistor_geometry(tech_id, float(val))
                except Exception:
                    # no parametric resistor on this PDK either: honest absence.
                    continue
            cg.name = p.name
            out[p.name] = cg

    # --- passive pass: caps/resistors have no parametric build_device fallback. ---
    # When a passive reference GDS is given, reuse the reference's captured passive
    # geometry (MIM cap / poly resistor). Caps/resistors are PDK-portable to first
    # order (a 3pF MIM cap is ~the same plate area in any process), so the reference
    # passive cell is an HONEST stand-in for the target PDK until a true target-PDK
    # passive cell builder exists. Tagged source='reference_passive' so provenance is
    # explicit (it is NOT a target-PDK vendor capture, and NOT fabricated).
    if passive_ref_gds is not None:
        from clay.layout.extract import library_device as LD

        passive_names = {p.name for p in getattr(intent, "passives", []) or []}
        if passive_names:
            pv = LD.extract_device_geometry(intent, passive_ref_gds)
            for name in getattr(intent, "device_bboxes", {}) or {}:
                if name in out or name not in passive_names:
                    continue
                geom = pv.get(name)
                if geom is not None and geom.clean and geom.polys_by_layer:
                    geom.source = "reference_passive"
                    out[name] = geom

    # --- guard-ring pass (OPT-IN): wrap every MOS stamp in a tap ring --------
    # ('group' intentionally does NOT enter here: group rings are built after
    # placement by group_guard_rings; the stamps stay byte-identical to off.)
    if ring_mode == "device":
        body_nets = {
            d.name: str((getattr(d, "nets", {}) or {}).get("body", "") or "")
            for d in getattr(intent, "devices", []) or []
        }
        for name, cg in out.items():
            kind = _norm_kind(params[name][0]) if name in params else ""
            ring_well = _RING_WELL_FOR_KIND.get(kind)
            if ring_well is None:
                continue  # not a MOS (passive / untyped): no ring
            attach_guard_ring(
                cg, tech_id, well=ring_well, tap_net=body_nets.get(name, "")
            )

    return out


def provenance(geometry: Dict[str, CellGeometry]) -> Dict[str, int]:
    """Count how many stamps came from each source (``vendor`` / ``build_device``)."""
    counts: Dict[str, int] = {"vendor": 0, "build_device": 0, "unknown": 0}
    for g in geometry.values():
        src = getattr(g, "source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# 2b. real stamp extents -> overlap-aware compaction
# --------------------------------------------------------------------------- #
def stamp_extent(geom: CellGeometry) -> _XY:
    """The real ``(w_um, h_um)`` a :class:`CellGeometry` stamp occupies when placed.

    The bounding box of ALL the stamp's polygons across every layer -- i.e. the
    footprint the emitter actually writes, INCLUDING vendor well / implant /
    guard-ring overhang and (for a ``build_device`` fallback) the full folded cell
    height. This is the size the compaction must reserve to stay overlap-free, and is
    usually WIDER / TALLER than the correspondence ``device_bbox`` (which only bounds
    the captured finger region). Returns ``(0.0, 0.0)`` for an empty stamp.
    """
    xs = [x for rings in geom.polys_by_layer.values() for ring in rings for (x, _y) in ring]
    ys = [y for rings in geom.polys_by_layer.values() for ring in rings for (_x, y) in ring]
    if not xs or not ys:
        return (0.0, 0.0)
    return (max(xs) - min(xs), max(ys) - min(ys))


def stamp_extents(geometry: Dict[str, CellGeometry]) -> Dict[str, _XY]:
    """``name -> (w_um, h_um)`` real stamp extent for every clean device in ``geometry``.

    The exact ``sizes`` dict :func:`clay.layout.extract.dcsolve.solve_compact` consumes
    so the no-overlap compaction reserves each device's REAL emitted footprint. A
    stamp with no polygons is skipped (it has no footprint to reserve).
    """
    out: Dict[str, _XY] = {}
    for name, geom in geometry.items():
        if not geom.polys_by_layer:
            continue
        w, h = stamp_extent(geom)
        if w > 0.0 and h > 0.0:
            out[name] = (w, h)
    return out


# Default poly-poly min spacing (nm) to reserve abut clearance against. 280nm is the
# gf180mcu signoff value (``GF180MCU_MIN_DRC_RULES`` gf180_poly_poly_space) and is also
# safely >= the open_pdk / open_pdk poly min (250nm), so it is a safe default for
# any of the target PDKs. Override via the ``poly_poly_min_nm`` parameter.
_POLY_POLY_MIN_NM_DEFAULT: float = 280.0


def _poly_layer_for(tech_id: str) -> Optional[_LayerKey]:
    """The ``(layer, datatype)`` of the poly (gate) layer for ``tech_id``, or None.

    Derived from the resolved :class:`~clay.layout.tech.LayoutTech` (gf180mcu -> 30/0,
    open_pdk -> 30/0, open_pdk -> 13/0); never hardcoded. None if the tech cannot
    be resolved (then no abut padding is applied and the caller falls back to the raw
    extents -- honest, not silently wrong).
    """
    try:
        from clay.layout.tech import LayoutTech

        tech = LayoutTech.resolve(tech_id)
        key = tech.build_layer("poly")
        if key is not None:
            return (int(key[0]), int(key[1]))
    except Exception:
        return None
    return None


def _poly_setback_um(geom: CellGeometry, poly_layer: _LayerKey) -> Optional[float]:
    """The REAL min poly setback (um) of a stamp: the smallest distance from the
    stamp's poly polygons to the stamp's overall bounding box, over all four sides.

    Measured from the EXACT emitted geometry (the polygons that will be stamped) --
    never assumed. The poly sits inside the cell footprint with implant/well/contact
    overhang around it; when two cells abut (their reserved footprints touch) the gap
    between their polys is the sum of the two facing setbacks. Returns None if the
    stamp has no poly on ``poly_layer`` (then it needs no poly-abut padding).
    """
    poly_rings = geom.polys_by_layer.get(poly_layer)
    if not poly_rings:
        return None
    xs = [x for rings in geom.polys_by_layer.values() for ring in rings for (x, _y) in ring]
    ys = [y for rings in geom.polys_by_layer.values() for ring in rings for (_x, y) in ring]
    if not xs or not ys:
        return None
    cx0, cx1, cy0, cy1 = min(xs), max(xs), min(ys), max(ys)
    pxs = [x for ring in poly_rings for (x, _y) in ring]
    pys = [y for ring in poly_rings for (_x, y) in ring]
    px0, px1, py0, py1 = min(pxs), max(pxs), min(pys), max(pys)
    return min(px0 - cx0, cx1 - px1, py0 - cy0, cy1 - py1)


def padded_stamp_extents(
    geometry: Dict[str, CellGeometry],
    *,
    tech_id: str,
    poly_poly_min_nm: float = _POLY_POLY_MIN_NM_DEFAULT,
) -> Dict[str, _XY]:
    """Like :func:`stamp_extents`, but PAD each reserved footprint so two ABUTTING
    reserved footprints leave >= the poly-poly min spacing between the actual polys.

    FIX 2 (abut clearance): every INDIVIDUAL ``build_device`` cell is DRC-clean, but
    when the placer abuts two reserved footprints (gap ~0), the two cells' polys sit
    only ``setback_A + setback_B`` apart -- below the poly-poly min, an ASSEMBLED-only
    DRC violation at the abut face. This pads each device's reserved ``(w, h)`` by a
    per-SIDE clearance ``(poly_poly_min - 2*setback)/2`` (clamped >= 0), where
    ``setback`` is the device's OWN real min poly setback (:func:`_poly_setback_um`).
    Because the placer reserves a CENTRED footprint (pos +/- w/2), padding a dimension
    by ``2 * per_side`` adds ``per_side`` to each face; two abutting padded footprints
    then leave ``setback_A + pad_A + setback_B + pad_B >= poly_poly_min`` between the
    polys (the per-side pad uses each device's own setback, and the true facing setback
    is >= that device's min-over-sides setback, so the guarantee holds on every face).

    This changes ONLY the reserved spacing footprint fed to the placer -- NOT the drawn
    geometry (the stamp polygons are untouched), and NOT leakage-freeness (the pad is a
    function of process geometry + the PDK rule only, never a reference coordinate). A
    stamp with no poly on the tech's poly layer is reserved at its raw extent.
    """
    poly_layer = _poly_layer_for(tech_id)
    min_um = float(poly_poly_min_nm) / 1000.0
    out: Dict[str, _XY] = {}
    for name, geom in geometry.items():
        if not geom.polys_by_layer:
            continue
        w, h = stamp_extent(geom)
        if w <= 0.0 or h <= 0.0:
            continue
        per_side = 0.0
        if poly_layer is not None:
            setback = _poly_setback_um(geom, poly_layer)
            if setback is not None:
                per_side = max(0.0, (min_um - 2.0 * setback) / 2.0)
        # pad each of the two faces per axis by per_side (centred footprint).
        out[name] = (w + 2.0 * per_side, h + 2.0 * per_side)
    return out


def realize_clean(
    intent,
    geometry: Dict[str, CellGeometry],
    *,
    root: str = "RR0",
    gap_um: float = 1.4,
    tech_id: str = "gf180mcu",
    poly_poly_min_nm: float = _POLY_POLY_MIN_NM_DEFAULT,
) -> Dict[str, _XY]:
    """Solve OVERLAP-AWARE positions: space devices by their REAL stamp extents.

    Thin convenience over :func:`clay.layout.extract.dcsolve.solve_compact` that
    derives each device's real footprint from ``geometry`` (the polygons that will
    actually be stamped -- vendor overhang included, and the TALL ``build_device``
    fallback for the band-only MOScap so it gets enough vertical room) and feeds them
    as ``sizes`` (PADDED for abut poly clearance, FIX 2 -- see
    :func:`padded_stamp_extents`) so the min-separation / no-overlap compaction
    reserves the emitted extent plus enough room that two abutting cells do not
    violate the poly-poly min. The returned ``positions`` stay leakage-free (sizes +
    order topology + gap only; NO stored reference coordinate), so
    :func:`render_realize` stamps a DRC-clean real-geometry render (the only same-layer
    overlap left is the documented MM15/MM16 interdigitation, which is inherent in the
    reference's own captured bboxes).
    """
    from clay.layout.extract import dcsolve

    sizes = padded_stamp_extents(
        geometry, tech_id=tech_id, poly_poly_min_nm=poly_poly_min_nm
    )
    positions, _rep = dcsolve.solve_compact(intent, root=root, gap_um=gap_um, sizes=sizes)
    return positions


def realize_migrated_clean(
    intent,
    geometry: Dict[str, CellGeometry],
    *,
    root: str = "RR0",
    tech_id: str = "gf180mcu",
    poly_poly_min_nm: float = _POLY_POLY_MIN_NM_DEFAULT,
) -> Dict[str, _XY]:
    """Overlap-aware MIGRATION placement: reserve each device's REAL emitted extent.

    The migration counterpart of :func:`realize_clean`. Where ``realize_clean`` uses
    :func:`clay.layout.extract.dcsolve.solve_compact` (seqpair from the reference),
    this uses the migration realize (:func:`dcsolve.realize_sized`: equality +
    :func:`dcsolve.order_intent` + min-gap) but spaces every device by its REAL emitted
    stamp extent, PADDED for abut poly clearance (FIX 2 --
    :func:`padded_stamp_extents`), instead of the ``fold_intent.folded_sizes``
    footprint :func:`clay.layout.extract.migrate_smic.migrate` reserves. The two fold
    models disagree (migrate folds toward the reference aspect, the emitted
    ``build_device`` geometry toward a square), so placing against ``folded_sizes`` and
    then stamping the emitted geometry collides; placing against the emitted extents
    does not. ``tech_id`` selects the poly layer the abut pad is measured against (the
    migration TARGET PDK -- gf180mcu by default; pass ``open_pdk`` for a external_pdk
    realize). Leakage-free: sizes (process geometry) + order topology only, NO stored
    reference coordinate -- so the migrated real-geometry render becomes overlap-free
    AND abut-clean (modulo the documented MM15/MM16 interdigitation) while the
    migration leakage proof (:func:`migrate_smic.leakage_ab`) is unaffected.
    """
    from clay.layout.extract import dcsolve

    sizes = padded_stamp_extents(
        geometry, tech_id=tech_id, poly_poly_min_nm=poly_poly_min_nm
    )
    order = dcsolve.order_intent(intent)
    positions, _n = dcsolve.realize_sized(intent, sizes, order, root=root)
    return positions


# --------------------------------------------------------------------------- #
# 3. render_realize: stamp geometry at solved positions + render
# --------------------------------------------------------------------------- #
def render_realize(
    intent,
    positions: Dict[str, _XY],
    geometry: Dict[str, CellGeometry],
    out_gds: str,
    out_png: str,
    title: str = "",
    *,
    ref_gds: Optional[str] = None,
) -> dict:
    """Stamp ``geometry`` at ``positions`` into a flat GDS and render it to PNG.

    ``positions`` (``name -> (cx, cy)`` um) come from the relation realize
    (:func:`dcsolve.solve_compact` / :func:`migrate_smic.migrate`) -- NEVER from a
    stored reference coordinate. For each device with a CLEAN stamp and a supplied
    position, the stamp's centroid-relative polygons are translated to the position
    and written to a single flat top cell of ``out_gds``; the GDS is then rendered
    to ``out_png`` via :func:`clay.layout.extract.external_pdk.render`.

    Reuses :func:`clay.layout.extract.library_device.emit_library_gds` for the stamp
    (it consumes the same :class:`CellGeometry` shape for vendor and build_device
    stamps alike); ``ref_gds`` is only forwarded so emit can resolve the cell name
    and is otherwise unused for the stamp itself (the geometry is already captured).

    Returns ``{placed, skipped, out_gds, out_png}`` -- honest counts (a device with
    no position or no clean stamp is reported in ``skipped``, never silently dropped).
    """
    from pathlib import Path
    from clay.layout.extract import library_device as LD

    Path(out_gds).parent.mkdir(parents=True, exist_ok=True)

    report = LD.emit_library_gds(
        intent,
        positions,
        ref_gds or "",
        out_gds,
        geometry=geometry,
    )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    # Commercial reference renderer omitted in open-source edition

    return {
        "placed": sorted(report.placed),
        "skipped": dict(report.skipped),
        "out_gds": out_gds,
        "out_png": out_png,
    }


__all__ = [
    "attach_guard_ring",
    "build_device_geometry",
    "design_row_height_target_um",
    "composite_group_geometry",
    "GroupGuardRing",
    "group_guard_rings",
    "group_ring_padded_extents",
    "group_ring_tap_terminals",
    "real_device_geometry",
    "render_realize",
    "provenance",
    "stamp_extent",
    "stamp_extents",
    "padded_stamp_extents",
    "realize_clean",
    "realize_migrated_clean",
]
