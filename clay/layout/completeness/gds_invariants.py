"""GDS-BACKED geometry provider: measure layout-intent obligations on the FINAL
emitted ``layout.gds``, not on the PLACER's intermediate reservation.

WHY THIS EXISTS (the soundness hole this closes)
================================================
The structural geometric obligations in
:mod:`clay.layout.geometric_intent_checks` (``check_equal_size`` /
``check_common_centroid`` / ``check_place_group_realized`` / ...) consume a
:class:`~clay.layout.geometric_intent_checks.GeometricPlacement` built from the
PLACER's per-device bbox (:meth:`GeometricPlacement.from_placement`). That bbox is
the placer's RESERVED FOOTPRINT, which the @equal_size MILP forces equal across a
declared group EVEN WHEN the underlying devices differ. The EMITTED GDS, by
contrast, builds each device's artwork from its VALUE: a gf180 MIM cap's FuseTop
top plate (layer 75/0) is a square whose area = ``value / density``
(:func:`clay.layout.passive_device_cell.mim_plate_side_nm`). So with
``@equal_size(Cc, CL)`` declared but ``Cc`` and ``CL`` given DIFFERENT values, the
placer reserves equal footprints (``check_equal_size`` on the PLACEMENT reports
0.0 / PASS) while the rendered FuseTop plates are UNEQUAL (the artifact is wrong).
The obligation passed while the GDS is wrong -- an UNSOUND realization check.

This module rebuilds the SAME ``GeometricPlacement`` type the obligations already
consume, but with each device's bbox + per-(layer,datatype) shapes read from the
REAL emitted GDS, so ``check_equal_size`` (and the other geometric checks, where
the GDS-backed placement makes them measurable) now run against the AUTHORITATIVE
artifact. For a cap, a device's realized "footprint" is its TRUE EMITTED PLATE --
the FuseTop (75/0) MIM plate polygon -- NOT the placer reservation, so two caps
with different values are correctly reported UNEQUAL.

DEVICE -> GDS PROVENANCE (how a device LABEL maps to GDS geometry)
=================================================================
The generic emit path (``clay/flow/step.py::LayoutStep._run_generic`` ->
``clay/layout/generic_router.py::emit_routed_cell``) writes a SINGLE FLAT top cell:
every device FEOL poly and every routed BEOL poly is inserted directly into the
top cell with NO per-device cell instance and NO device-label tag in the stream
(``emit_routed_cell`` flattens ``device_polys + cell.polys`` into ``top.shapes``;
the only net-level provenance is the text labels keyed to nets, ``cell.labels``).
There is therefore NO device->cell-instance map in the emitted GDS to read.

Spatial containment alone is NOT device provenance: routed metal, vias, fill or an
unrelated polygon can remain inside a window after the device artwork is deleted.
The caller therefore passes each materialized ``PlacedDevice`` (``bbox`` plus its
owned ``polys``).  The owned polygon multiset is the witness.  A device is present
only when the final GDS contains that complete layer/datatype + relative-geometry
witness inside its window.  Extra routing shapes are ignored and cannot become the
device footprint.  A bbox-only label has no ownership witness and is deliberately
unmeasurable.  This fail-closed rule is what makes a deleted Q/R distinguishable
from routing or fill left behind in its old placement window.

This is a STRICT realization tightening: the placement-time path is untouched (the
sibling placement provider stays the intermediate measurement), and this module
ADDS the GDS-truth path, which is the authoritative one.

SHARED INTERFACE (stable, documented -- a sibling Oracle-2 agent imports this)
==============================================================================
  * :func:`geometric_placement_from_gds` -> the SAME
    :class:`~clay.layout.geometric_intent_checks.GeometricPlacement` the
    obligations consume, with ``.device_bboxes`` mapped to REALIZED GDS bboxes (cap
    footprints = their FuseTop plate). Drop-in for the placement-built one.
  * :func:`extract_invariants_from_gds` -> the SAME canonical 8-category dict shape
    :func:`clay.layout.completeness.invariance.extract_invariants` returns (keys
    ``relative_order`` / ``adjacency`` / ``alignment`` / ``band`` / ``symmetry_axis``
    / ``size_class`` / ``spacing`` / ``orientation``), but computed from the FINAL
    GDS geometry. The signature is kept stable: ``(gds_path, *, device_labels,
    intent=None, eps=...)``.
  * :func:`verify_obligations_on_gds` -> the LayoutObligations measured on the FINAL
    GDS (``check_equal_size`` + the other geometric checks), the GDS-truth verify
    entry point.

Every measurement is mechanical geometry math on the GDS, deterministic (sorted
keys / sorted tuples), relative where a ratio is reported (the dbu unit cancels).
A device with NO realized GDS shape in its window is UNMEASURABLE (its bbox is
absent from the provider), which the obligations treat as blocking -- never a
silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

from clay.layout.geometric_intent_checks import GeometricPlacement

# The gf180mcu MIM top plate the LVS deck extracts the cap value FROM: FuseTop
# (75/0), a square whose area * density == C (see
# clay.layout.passive_device_cell.build_capacitor / mim_plate_side_nm). A cap's
# TRUE realized footprint is this plate, NOT the placer's reserved bbox -- two
# caps of different VALUE have different plate sides even when the placer reserved
# equal footprints. Kept as a module constant (verified against the emitted GDS in
# tests/test_gds_obligation_soundness.py) so the cap-plate layer is explicit and
# auditable rather than a magic literal buried in the reader.
MIM_PLATE_LAYER: Tuple[int, int] = (75, 0)

# Translation-snap tolerance (dbu). The emit grid-snaps every polygon to the
# manufacturing grid (gf180mcu = 5 nm), so a shape that the placer reserved at a
# window edge can land a few nm outside the raw placed window. EPS absorbs that
# grid residue when deciding whether a GDS shape belongs to a device's window,
# without merging a clearly-foreign device's shape. Kept small relative to the
# inter-device channel (hundreds of nm to um).
EPS: int = 50
# Shape-identity tolerance is deliberately tighter than window containment.
# Window slack absorbs edge placement residue; allowing that full slack on every
# artwork edge could confuse a small R90 rectangle with its R0 witness.
ARTWORK_EPS: int = 5

_Bbox = Tuple[int, int, int, int]
_Point = Tuple[int, int]
_Contour = Tuple[_Point, ...]


@dataclass(frozen=True, order=True)
class _ShapeGeometry:
    """One exact GDS boundary: bbox plus canonical hull/hole contours."""

    bbox: _Bbox
    # First contour is the hull; remaining contours are sorted holes.
    contours: Tuple[_Contour, ...]


@dataclass(frozen=True)
class GdsGuardRingObligation:
    """One declared ring measured on the exact emitted GDS.

    The materializer is not evidence.  ``measured_on_gds`` becomes true only
    after the exact artifact, emitter-input snapshot, member artwork, ring
    artwork, and route-manifest tie have all been replayed at this boundary.
    """

    members: Tuple[str, ...]
    well: str
    tie_net: str
    hug: str
    role: str
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    member_classes: Tuple[Tuple[str, str], ...] = ()
    attached_nets: Tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class GdsBandObligation:
    """One relative bottom/center/top band order measured on exact GDS artwork."""

    top: str
    bottom: str
    center: str
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    bottom_members: Tuple[str, ...] = ()
    center_members: Tuple[str, ...] = ()
    top_members: Tuple[str, ...] = ()
    bottom_bbox_nm: Tuple[int, int, int, int] | None = None
    center_bbox_nm: Tuple[int, int, int, int] | None = None
    top_bbox_nm: Tuple[int, int, int, int] | None = None
    reason: str = ""


@dataclass(frozen=True)
class GdsAnchorObligation:
    """One left/right edge-access declaration measured on exact GDS evidence."""

    target: str
    edge: str
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    resolution: Literal["device", "group", "net", ""] = ""
    members: Tuple[str, ...] = ()
    target_bbox_nm: Tuple[int, int, int, int] | None = None
    universe_bbox_nm: Tuple[int, int, int, int] | None = None
    reason: str = ""


@dataclass(frozen=True)
class GdsRailAffinityObligation:
    """One group-to-supply-side relation measured on source-owned exact GDS."""

    group: str
    side: str
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    members: Tuple[str, ...] = ()
    group_bbox_nm: Tuple[int, int, int, int] | None = None
    universe_bbox_nm: Tuple[int, int, int, int] | None = None
    reason: str = ""


@dataclass(frozen=True)
class GdsChannelObligation:
    """One declared device-free corridor measured on exact emitted GDS artwork."""

    orient: str
    declared_width_nm: int
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    left_owner_block_id: str = ""
    lower_owner_block_ids: Tuple[str, ...] = ()
    corridor_bbox_nm: Tuple[int, int, int, int] | None = None
    observed_clearance_nm: int | None = None
    reason: str = ""


@dataclass(frozen=True)
class GdsDensityWhitespaceObligation:
    """Structural occupancy measured from uniquely claimed exact-GDS artwork."""

    max_whitespace_fraction: float
    scope: str
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    member_count: int = 0
    envelope_bbox_nm: Tuple[int, int, int, int] | None = None
    occupied_area_nm2: int | None = None
    envelope_area_nm2: int | None = None
    observed_whitespace_fraction: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class GdsFunctionalOdFillObligation:
    """Functional active occupancy uniquely claimed from exact emitted GDS."""

    min_fraction: float
    scope: str
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    member_count: int = 0
    active_layer_keys: Tuple[Tuple[int, int], ...] = ()
    envelope_bbox_nm: Tuple[int, int, int, int] | None = None
    active_area_nm2: int | None = None
    envelope_area_nm2: int | None = None
    observed_fraction: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class GdsFingerArrayOrderObligation:
    """One source-declared ratio array replayed onto exact emitted GDS artwork."""

    group: str
    style: str
    rows: int
    order: str
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    owner_sequence: Tuple[str, ...] = ()
    program_sha256: str | None = None
    materialization_sha256: str | None = None
    geometry_sha256: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class GdsPassiveBankPartitionObligation:
    """One topology-declared passive sub-bank proven on exact emitted GDS."""

    parent: str
    bank: str
    members: Tuple[str, ...]
    macro_id: str
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    claimed_bbox_nm: Tuple[int, int, int, int] | None = None
    graph_sha256: str | None = None
    emission_certificate_sha256: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class GdsWellIslandShareObligation:
    """One declared shared well measured as a GDS connected component."""

    group: str
    body: str
    well: str
    members: Tuple[str, ...]
    layer_key: Tuple[int, int] | None
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    island_bbox_nm: Tuple[int, int, int, int] | None = None
    foreign_members: Tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class GdsCommonCentroidObligation:
    """One source-declared point-reflected finger set measured on exact GDS."""

    group: str
    members: Tuple[str, str]
    rows: int
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    point_reflection_verified: bool = False
    owner_rows: Tuple[Tuple[str, ...], ...] = ()
    channel_centers_twice_nm: Tuple[Tuple[str, int, int], ...] = ()
    reflection_center_fourfold_nm: Tuple[int, int] | None = None
    member_channel_geometry_nm: Tuple[
        Tuple[str, Tuple[int, int, int, int], str], ...
    ] = ()
    reason: str = ""


@dataclass(frozen=True)
class GdsInterdigitationObligation:
    """One source-declared finger interleave reconstructed from exact GDS."""

    members: Tuple[str, str]
    declaration_ids: Tuple[str, ...]
    pattern: str
    finger_pairs: int
    declaration_sha256: str
    gds_sha256: str | None
    status: Literal["pass", "fail", "unmeasurable", "unsupported"]
    measured_on_gds: bool
    owner_rows: Tuple[Tuple[str, ...], ...] = ()
    observed_pair_pattern: Tuple[str, ...] = ()
    channel_centers_twice_nm: Tuple[Tuple[str, int, int], ...] = ()
    reason: str = ""


def _remove_collinear_points(points: Sequence[_Point]) -> Tuple[_Point, ...]:
    """Remove duplicate/straight-through vertices without changing a contour."""

    cleaned: list[_Point] = []
    for point in points:
        p = (int(point[0]), int(point[1]))
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        keep: list[_Point] = []
        count = len(cleaned)
        for index, point in enumerate(cleaned):
            prev = cleaned[(index - 1) % count]
            nxt = cleaned[(index + 1) % count]
            cross = (
                (point[0] - prev[0]) * (nxt[1] - point[1])
                - (point[1] - prev[1]) * (nxt[0] - point[0])
            )
            between = (
                min(prev[0], nxt[0]) <= point[0] <= max(prev[0], nxt[0])
                and min(prev[1], nxt[1]) <= point[1] <= max(prev[1], nxt[1])
            )
            if cross == 0 and between:
                changed = True
                continue
            keep.append(point)
        cleaned = keep
    return tuple(cleaned)


def _canonical_contour(points: Sequence[_Point]) -> Optional[_Contour]:
    """Canonicalize cyclic start and winding while preserving the full outline."""

    contour = _remove_collinear_points(points)
    if len(contour) < 3:
        return None
    candidates = []
    for sequence in (contour, tuple(reversed(contour))):
        candidates.extend(
            sequence[index:] + sequence[:index]
            for index in range(len(sequence))
        )
    return min(candidates)


def _shape_geometry(
    hull: Sequence[_Point],
    holes: Sequence[Sequence[_Point]] = (),
) -> Optional[_ShapeGeometry]:
    canonical_hull = _canonical_contour(hull)
    if canonical_hull is None:
        return None
    canonical_holes = tuple(
        sorted(
            contour
            for raw in holes
            if (contour := _canonical_contour(raw)) is not None
        )
    )
    xs = [point[0] for point in canonical_hull]
    ys = [point[1] for point in canonical_hull]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return None
    return _ShapeGeometry(bbox=bbox, contours=(canonical_hull,) + canonical_holes)


# --------------------------------------------------------------------------- #
# GDS reading (klayout) -- flat top cell -> per-(layer,datatype) shape bboxes  #
# --------------------------------------------------------------------------- #


def _read_top_cell_shapes(gds_path: str) -> Dict[Tuple[int, int], list]:
    """Exact boundary geometry for every polygonal shape in the GDS top cell.

    Reads the FINAL emitted GDS (the flat top cell ``emit_routed_cell`` wrote) via
    klayout and buckets each shape's integer bbox by its ``(layer, datatype)``.
    Mechanical, deterministic. Raises if klayout is unavailable or the file has no
    top cell (an honest hard error -- a missing GDS cannot be measured).
    """
    import klayout.db as db

    layout = db.Layout()
    layout.read(str(gds_path))
    top = layout.top_cell()
    if top is None:
        raise ValueError(f"{gds_path}: GDS has no top cell to measure")

    out: Dict[Tuple[int, int], list] = {}
    for li in layout.layer_indexes():
        info = layout.get_info(li)
        key = (int(info.layer), int(info.datatype))
        bucket = out.setdefault(key, [])
        for shape in top.shapes(li).each():
            polygon = None
            try:
                if shape.is_box():
                    polygon = db.Polygon(shape.box)
                elif shape.is_polygon():
                    polygon = shape.polygon
                elif shape.is_simple_polygon():
                    polygon = db.Polygon(shape.simple_polygon)
                elif shape.is_path():
                    polygon = shape.path.polygon()
            except Exception:  # noqa: BLE001 - malformed/non-boundary GDS shape
                polygon = None
            if polygon is None:
                continue
            try:
                hull = tuple(
                    (int(point.x), int(point.y))
                    for point in polygon.each_point_hull()
                )
                holes = tuple(
                    tuple(
                        (int(point.x), int(point.y))
                        for point in polygon.each_point_hole(index)
                    )
                    for index in range(int(polygon.holes()))
                )
            except Exception:  # noqa: BLE001 - non-polygonal record
                continue
            geometry = _shape_geometry(hull, holes)
            if geometry is not None:
                bucket.append(geometry)
        bucket.sort()
    return out


def _contains(window: _Bbox, box: _Bbox, eps: int) -> bool:
    """True iff ``box`` sits inside ``window`` (with ``eps`` grid slack on each edge).

    A GDS shape belongs to a device when its bbox is contained in the device's
    placed window. The ``eps`` slack absorbs the emit's grid-snap so a shape that
    landed a few nm past the raw window edge is still claimed by the right device.
    """
    wx0, wy0, wx1, wy1 = window
    bx0, by0, bx1, by1 = box
    return (
        bx0 >= wx0 - eps and by0 >= wy0 - eps and bx1 <= wx1 + eps and by1 <= wy1 + eps
    )


def _union_bbox(boxes: Sequence[_Bbox]) -> Optional[_Bbox]:
    """Union bounding box of a list of bboxes, or ``None`` when empty."""
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


# --------------------------------------------------------------------------- #
# device -> realized GDS geometry (materialized-artwork ownership witness)    #
# --------------------------------------------------------------------------- #


def _normalize_window(win: Any) -> Optional[_Bbox]:
    """Normalise a window spec to ``(x0, y0, x1, y1)`` ints, or ``None``.

    Accepts a 4-tuple/list (a placed device bbox) directly, or an object exposing a
    ``.bbox`` attribute (a :class:`~clay.layout.generic_placement.PlacedDevice`).
    """
    bb = win
    if not isinstance(win, (tuple, list)):
        bb = getattr(win, "bbox", None)
    if bb is None or len(bb) != 4:
        return None
    x0, y0, x1, y1 = (int(round(float(v))) for v in bb)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _layer_polygon_bbox(layer_polygon: Any) -> Optional[_Bbox]:
    """Return one materialized ``LayerPolygon`` bbox, or ``None`` if malformed."""

    polygon = getattr(layer_polygon, "polygon", None)
    vertices = tuple(getattr(polygon, "vertices", ()) or ())
    if not vertices:
        return None
    try:
        xs = [int(round(float(vertex.x))) for vertex in vertices]
        ys = [int(round(float(vertex.y))) for vertex in vertices]
    except (AttributeError, TypeError, ValueError):
        return None
    box = (min(xs), min(ys), max(xs), max(ys))
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box


def _owned_shape_witness(win: Any) -> Dict[Tuple[int, int], list]:
    """Materialized per-device artwork used to attribute final-GDS shapes.

    A plain bbox is intentionally insufficient.  Only the polygon collection
    carried by the materialized device establishes layer ownership.
    """

    polys = getattr(win, "polys", None)
    if polys is None:
        polys = getattr(win, "polygons", None)
    expected: Dict[Tuple[int, int], list] = {}
    for layer_polygon in tuple(polys or ()):
        try:
            key = (
                int(getattr(layer_polygon, "layer")),
                int(getattr(layer_polygon, "datatype")),
            )
        except (TypeError, ValueError):
            continue
        polygon = getattr(layer_polygon, "polygon", None)
        vertices = tuple(getattr(polygon, "vertices", ()) or ())
        geometry = _shape_geometry(
            tuple((int(round(float(v.x))), int(round(float(v.y)))) for v in vertices)
        )
        if geometry is not None:
            expected.setdefault(key, []).append(geometry)
    for shapes in expected.values():
        shapes.sort()
    return expected


def _unique_emitter_owner(
    member: str,
    geometry: Mapping[str, Any],
) -> str:
    """Resolve one source member to one exact emitted stamp."""

    owners = tuple(
        sorted(
            str(name)
            for name, candidate in geometry.items()
            if str(name) != member
            if member in tuple(getattr(candidate, "composite_members", ()) or ())
        )
    )
    if member in geometry:
        if owners:
            raise ValueError(
                f"member {member!r} has direct and distinct aggregate artwork"
            )
        return member
    if len(owners) == 1:
        return owners[0]
    detail = "no" if not owners else "ambiguous"
    raise ValueError(f"member {member!r} has {detail} emitter artwork")


def _emitted_shape_witness(
    stamp: Any,
    placement_um: Any,
    *,
    excluded_layers: Sequence[Tuple[int, int]] = (),
) -> Dict[Tuple[int, int], list]:
    """Exact placed polygons from the geometry handed to the GDS emitter.

    This is ownership provenance, not a measurement substitute: callers still
    have to find the complete polygon multiset in the reread GDS.  It closes the
    ordinary-MOS seam where the emitter deliberately rebuilds a canonical route
    stamp instead of writing the placer's older ``PlacedDevice.polys``.
    """

    from clay.layout.placed_geometry import placed_point_nm

    if (
        type(placement_um) not in (tuple, list)
        or len(placement_um) != 2
    ):
        return {}
    excluded = {
        (int(layer[0]), int(layer[1])) for layer in tuple(excluded_layers)
    }
    expected: Dict[Tuple[int, int], list] = {}
    for raw_key, rings in (
        getattr(stamp, "polys_by_layer", {}) or {}
    ).items():
        try:
            key = (int(raw_key[0]), int(raw_key[1]))
        except (IndexError, TypeError, ValueError):
            return {}
        if key in excluded:
            continue
        for ring in tuple(rings or ()):
            try:
                vertices = tuple(
                    placed_point_nm(float(x), float(y), tuple(placement_um))
                    for x, y in tuple(ring)
                )
            except (TypeError, ValueError):
                return {}
            geometry = _shape_geometry(vertices)
            if geometry is None:
                # Never shrink a supplied emitter witness by silently dropping
                # one malformed ring while retaining its siblings.
                return {}
            expected.setdefault(key, []).append(geometry)
    for shapes in expected.values():
        shapes.sort()
    return expected


def _exact_emitter_artwork_witness(
    stamp: Any,
    placement_um: Any,
    *,
    excluded_layers: Sequence[Tuple[int, int]] = (),
) -> Dict[Tuple[int, int], list]:
    """Preserve the emitter's polygon multiset for exact-GDS ownership claims."""

    return _emitted_shape_witness(
        stamp,
        placement_um,
        excluded_layers=excluded_layers,
    )


def _certified_assembly_transformed_layers(
    exact_artifact: Any,
    gds_path: str,
    emitted_device_geometry: Mapping[str, Any],
    emitted_device_placement_um: Mapping[str, Any],
    *,
    return_witness: bool = False,
    top_cell: Optional[str] = None,
) -> Any:
    """Validate the exact emitter snapshot and replay assembly-transform inputs."""

    if exact_artifact is None:
        return ((), None) if return_witness else ()
    from clay.layout.completeness.gds_witness import ExactGdsWitness
    from clay.layout.route.exact_artifact_evidence import (
        _emitted_device_snapshot_sha256,
    )

    gds_witness = ExactGdsWitness.open(
        gds_path,
        exact_artifact.scope.top_cell if top_cell is None else top_cell,
        exact_artifact,
    )
    replayed_snapshot = _emitted_device_snapshot_sha256(
        emitted_device_geometry, emitted_device_placement_um
    )
    if replayed_snapshot != exact_artifact.scope.emitted_device_snapshot_sha256:
        raise ValueError("exact emitter device snapshot does not replay")

    import klayout.db as kdb
    from clay.layout.placed_geometry import placed_point_nm
    from clay.layout.spec.exact_structure_profile import _region_sha256

    transformed = []
    for certificate in exact_artifact.well_merge_certificates:
        key = tuple(int(value) for value in certificate.layer_key)
        input_region = kdb.Region()
        for raw_name, stamp in emitted_device_geometry.items():
            position = emitted_device_placement_um.get(raw_name)
            if position is None and str(raw_name) != raw_name:
                position = emitted_device_placement_um.get(str(raw_name))
            if position is None:
                continue
            for raw_key, rings in (
                getattr(stamp, "polys_by_layer", {}) or {}
            ).items():
                if (int(raw_key[0]), int(raw_key[1])) != key:
                    continue
                for ring in rings:
                    points = [
                        kdb.Point(*placed_point_nm(float(x), float(y), position))
                        for x, y in ring
                    ]
                    input_region.insert(kdb.Polygon(points))
        input_region.merge()
        if _region_sha256(input_region) != certificate.input_region_sha256:
            raise ValueError("well merge certificate input does not replay")
        output_region = gds_witness.merged_region(key)
        if _region_sha256(output_region) != certificate.output_region_sha256:
            raise ValueError(
                "well merge certificate output differs from the reread exact GDS"
            )
        transformed.append(key)
    layers = tuple(sorted(transformed))
    return (layers, gds_witness) if return_witness else layers


def _replay_exact_final_placement_artwork(
    *,
    gds_path: str,
    top_cell: str,
    exact_artifact: Any,
    source_cell: Any,
    final_placement_certificate: Any,
    placement_state_sha256: str | None,
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    tech: Any,
) -> tuple[Any, tuple[tuple[str, str], ...], tuple, Any]:
    """Replay shared source, placement, emitter, and exact-GDS authority once."""

    from clay.layout.intent import layout_intent_from_cell_def
    from clay.layout.place.final_placement_state import (
        final_placement_state_digest,
        verify_placement_causality_basis,
    )
    from clay.layout.place_blocks_2d import (
        FinalPlacementCertificate,
        placement_certificate_is_signable,
    )
    from clay.layout.route.exact_artifact_evidence import (
        ExactEmittedArtifact,
        _replayed_source_cell_authority,
    )
    from clay.layout.route.route_cell import (
        _final_placement_provenance_digest,
        build_final_placement_resource_evidence,
    )

    if type(exact_artifact) is not ExactEmittedArtifact:
        raise TypeError("exact artwork replay requires ExactEmittedArtifact")
    if type(final_placement_certificate) is not FinalPlacementCertificate:
        raise TypeError("exact artwork replay requires FinalPlacementCertificate")
    if not isinstance(emitted_device_geometry, Mapping) or not isinstance(
        emitted_device_placement_um, Mapping
    ):
        raise TypeError("exact artwork replay requires emitter mappings")
    if tech is None:
        raise TypeError("exact artwork replay requires resolved layout tech")
    exact_artifact.validate()
    if exact_artifact.scope.top_cell != str(top_cell):
        raise ValueError("exact artifact is bound to another top cell")
    source_sha, inventory_sha, member_classes, canonical_cell = (
        _replayed_source_cell_authority(source_cell)
    )
    if (
        source_sha != exact_artifact.scope.source_cell_source_sha256
        or inventory_sha != exact_artifact.scope.source_member_inventory_sha256
    ):
        raise ValueError("source inventory differs from exact emitter scope")
    if not placement_certificate_is_signable(final_placement_certificate):
        raise ValueError("final placement certificate is not signable")
    replayed_state = final_placement_state_digest(final_placement_certificate)
    if replayed_state != placement_state_sha256:
        raise ValueError("final placement state digest does not replay")
    verify_placement_causality_basis(final_placement_certificate)
    canonical_intent = layout_intent_from_cell_def(canonical_cell)
    resource_evidence = build_final_placement_resource_evidence(
        dict(emitted_device_placement_um),
        dict(emitted_device_geometry),
        canonical_intent,
        tech,
        final_placement_certificate=final_placement_certificate,
    )
    if (
        _final_placement_provenance_digest(
            resource_evidence, final_placement_certificate
        )
        != replayed_state
    ):
        raise ValueError("final placement provenance does not replay")
    truth_claim = exact_artifact.truth_manifest.final_placement_certificate_sha256
    if truth_claim is not None and truth_claim != replayed_state:
        raise ValueError("exact GDS truth manifest binds another placement state")
    transformed_layers, witness = _certified_assembly_transformed_layers(
        exact_artifact,
        gds_path,
        emitted_device_geometry,
        emitted_device_placement_um,
        return_witness=True,
        top_cell=top_cell,
    )
    if witness is None:
        raise ValueError("exact GDS witness is absent")
    return canonical_intent, member_classes, transformed_layers, witness


def _exact_shape_multiset(
    expected: Mapping[Tuple[int, int], Sequence[_ShapeGeometry]],
    actual: Mapping[Tuple[int, int], Sequence[_ShapeGeometry]],
    *,
    eps: int,
) -> Tuple[Optional[Dict[Tuple[int, int], list]], int]:
    """Claim the exact absolute expected shapes once from the reread GDS."""

    claimed: Dict[Tuple[int, int], list] = {}
    missing = 0
    for key in sorted(expected):
        unused = list(sorted(actual.get(key, ())))
        for target in sorted(expected[key]):
            index = next(
                (
                    ordinal
                    for ordinal, candidate in enumerate(unused)
                    if _shape_close(candidate, target, eps)
                ),
                None,
            )
            if index is None:
                missing += 1
                continue
            claimed.setdefault(key, []).append(unused.pop(index))
    return (claimed if missing == 0 else None), missing


def _layer_polygons_witness(
    polygons: Sequence[Any],
    *,
    layer_keys: Sequence[Tuple[int, int]],
) -> Dict[Tuple[int, int], list]:
    """Canonical absolute geometry for selected independently rebuilt layers."""

    selected = {tuple(int(value) for value in key) for key in layer_keys}
    expected: Dict[Tuple[int, int], list] = {}
    for layer_polygon in polygons:
        key = (
            int(getattr(layer_polygon, "layer")),
            int(getattr(layer_polygon, "datatype")),
        )
        if key not in selected:
            continue
        vertices = tuple(getattr(layer_polygon.polygon, "vertices", ()) or ())
        shape = _shape_geometry(
            tuple((int(vertex.x), int(vertex.y)) for vertex in vertices)
        )
        if shape is None:
            raise ValueError("independent guard-ring rebuild is degenerate")
        expected.setdefault(key, []).append(shape)
    for shapes in expected.values():
        shapes.sort()
    return expected


def _materialized_guard_ring_witness(
    polygons: Sequence[Any],
    *,
    layer_keys: Sequence[Tuple[int, int]],
    manufacturing_grid_nm: int,
) -> Dict[Tuple[int, int], list]:
    """Independently replay the emitter's per-ring manufacturing-grid snap."""

    if type(manufacturing_grid_nm) is not int or manufacturing_grid_nm <= 0:
        raise ValueError("manufacturing_grid_nm must be a positive exact integer")

    def nearest(value_nm: int) -> int:
        floor = (value_nm // manufacturing_grid_nm) * manufacturing_grid_nm
        remainder = value_nm - floor
        return (
            floor
            if remainder * 2 < manufacturing_grid_nm
            else floor + manufacturing_grid_nm
        )

    selected = {tuple(int(value) for value in key) for key in layer_keys}
    expected: Dict[Tuple[int, int], list] = {}
    for layer_polygon in polygons:
        key = (
            int(getattr(layer_polygon, "layer")),
            int(getattr(layer_polygon, "datatype")),
        )
        if key not in selected:
            continue
        raw_vertices = tuple(
            (int(vertex.x), int(vertex.y))
            for vertex in tuple(
                getattr(layer_polygon.polygon, "vertices", ()) or ()
            )
        )
        if not raw_vertices:
            raise ValueError("independent guard-ring rebuild is degenerate")
        anchor_x, anchor_y = raw_vertices[0]
        delta_x = nearest(anchor_x) - anchor_x
        delta_y = nearest(anchor_y) - anchor_y
        normalized = tuple(
            (nearest(x + delta_x), nearest(y + delta_y))
            for x, y in raw_vertices
        )
        shape = _shape_geometry(normalized)
        if shape is None:
            raise ValueError("normalized independent guard-ring rebuild is degenerate")
        expected.setdefault(key, []).append(shape)
    for shapes in expected.values():
        shapes.sort()
    return expected


def _region_from_shapes(shapes: Sequence[_ShapeGeometry]):
    import klayout.db as kdb

    region = kdb.Region()
    for shape in shapes:
        polygon = kdb.Polygon([kdb.Point(x, y) for x, y in shape.contours[0]])
        for contour in shape.contours[1:]:
            polygon.insert_hole([kdb.Point(x, y) for x, y in contour])
        region.insert(polygon)
    region.merge()
    return region


def _merged_shape_witness(
    expected: Mapping[Tuple[int, int], Sequence[_ShapeGeometry]],
) -> Dict[Tuple[int, int], list]:
    """Canonical per-layer union for one emitter-owned composite stamp."""

    merged: Dict[Tuple[int, int], list] = {}
    for key, shapes in expected.items():
        region = _region_from_shapes(shapes)
        region.merge()
        for polygon in region.each():
            geometry = _shape_geometry(
                tuple(
                    (int(point.x), int(point.y))
                    for point in polygon.each_point_hull()
                ),
                tuple(
                    tuple(
                        (int(point.x), int(point.y))
                        for point in polygon.each_point_hole(index)
                    )
                    for index in range(int(polygon.holes()))
                ),
            )
            if geometry is None:
                raise ValueError("merged emitter artwork is degenerate")
            merged.setdefault(key, []).append(geometry)
    for shapes in merged.values():
        shapes.sort()
    return merged


def verify_guard_rings_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    guard_ring_intents: Sequence[Any],
    tech: Any,
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any = None,
    member_classes: Sequence[Tuple[str, str]] = (),
    member_ring_pad_top_nm: Sequence[Tuple[str, int]] = (),
    member_ring_pad_bottom_nm: Sequence[Tuple[str, int]] = (),
    final_placement_certificate: Any = None,
    placement_state_sha256: str | None = None,
    eps: int = ARTWORK_EPS,
) -> Tuple[GdsGuardRingObligation, ...]:
    """Independently replay declared tight rings against one exact GDS.

    This checker never consumes ``GuardRingReport`` or a materializer success
    flag.  It validates the emitter-issued device snapshot, proves every direct
    member's complete stamp occurs in the reread GDS, independently calls the
    PDK rule constructor at zero extra clearance, claims those exact ring shapes
    from GDS, and derives the attached net set from the exact route manifest.
    Unsupported composite ownership remains blocking rather than falling back to
    a placement bbox.
    """

    declarations = tuple(guard_ring_intents or ())
    if not declarations:
        return ()

    def blocked_all(
        reason: str, *, gds_sha256: str | None = None
    ) -> Tuple[GdsGuardRingObligation, ...]:
        return tuple(
            GdsGuardRingObligation(
                members=tuple(str(member) for member in declaration.members),
                well=str(declaration.well),
                tie_net=str(declaration.tie_net),
                hug=str(getattr(declaration, "hug", "") or ""),
                role=str(getattr(declaration, "role", "") or ""),
                declaration_sha256=str(declaration.sha256),
                gds_sha256=gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declarations
        )

    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        def padding_rows(value: Any, label: str) -> Tuple[Tuple[str, int], ...]:
            if type(value) is not tuple:
                raise TypeError(f"{label} must be an exact tuple")
            if any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not str
                or not row[0]
                or type(row[1]) is not int
                or row[1] < 0
                for row in value
            ):
                raise TypeError(f"{label} must contain exact name/integer pairs")
            if tuple(sorted(value)) != value or len(dict(value)) != len(value):
                raise ValueError(f"{label} must be sorted and unique")
            return value

        top_padding = padding_rows(
            member_ring_pad_top_nm, "guard-ring top padding"
        )
        bottom_padding = padding_rows(
            member_ring_pad_bottom_nm, "guard-ring bottom padding"
        )
    except Exception as exc:  # noqa: BLE001 - typed placement-input boundary
        return blocked_all(
            "guard-ring placement padding cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )
    try:
        if type(member_classes) is not tuple:
            raise TypeError("guard-ring member classes must be an exact tuple")
        raw_member_classes = member_classes
        if any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or not row[0]
            or row[1] not in ("mos", "passive")
            for row in raw_member_classes
        ):
            raise TypeError("guard-ring member classes must be exact typed pairs")
        if len({row[0] for row in raw_member_classes}) != len(raw_member_classes):
            raise ValueError("guard-ring member classes contain duplicate members")
        member_class_by_name = dict(raw_member_classes)
    except Exception as exc:  # noqa: BLE001 - typed compiler-input boundary
        return blocked_all(
            "guard-ring member classes cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )
    try:
        from clay.layout.route.exact_artifact_evidence import (
            ExactEmittedArtifact,
            _guard_ring_member_classes_sha256,
            _guard_ring_member_polarities_from_source_cell,
            _replayed_source_cell_authority,
        )

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("exact guard-ring checking requires ExactEmittedArtifact")
        if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
            raise TypeError("exact guard-ring checking requires emitter mappings")
        exact_artifact.validate()
        if exact_artifact.scope.top_cell != str(top_cell):
            raise ValueError("exact guard-ring artifact is bound to another top cell")
        (
            source_cell_source_sha256,
            source_member_inventory_sha256,
            canonical_member_classes,
            canonical_source_cell,
        ) = _replayed_source_cell_authority(source_cell)
        if (
            source_cell_source_sha256
            != exact_artifact.scope.source_cell_source_sha256
            or source_member_inventory_sha256
            != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError(
                "guard-ring source member inventory differs from exact emitter scope"
            )
        if raw_member_classes != canonical_member_classes:
            raise ValueError(
                "guard-ring member class inventory differs from canonical source"
            )
        if (
            _guard_ring_member_classes_sha256(canonical_member_classes)
            != exact_artifact.scope.guard_ring_member_classes_sha256
        ):
            raise ValueError(
                "guard-ring member class inventory differs from exact emitter scope"
            )
        canonical_member_polarities = (
            _guard_ring_member_polarities_from_source_cell(canonical_source_cell)
        )
        member_polarity_by_name = dict(canonical_member_polarities)
        from clay.layout.flow_placement import (
            declared_craft_compaction_requested,
        )
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.place_blocks_2d import (
            replay_canonical_member_ring_padding,
        )
        from clay.layout.tech import LayoutTech

        canonical_intent = layout_intent_from_cell_def(canonical_source_cell)
        placement_tech = (
            tech if isinstance(tech, LayoutTech) else LayoutTech.resolve(str(tech))
        )
        canonical_declarations = tuple(canonical_intent.guard_rings)
        if tuple(item.sha256 for item in canonical_declarations) != tuple(
            item.sha256 for item in declarations
        ):
            raise ValueError(
                "guard-ring declarations (members/well/tie/hug/role) differ "
                "from canonical source"
            )
        edge_rail_declared = any(
            str(getattr(trunk, "edge", "") or "").strip()
            for trunk in tuple(getattr(canonical_intent, "trunks", ()) or ())
        )
        canonical_compaction_requested = declared_craft_compaction_requested(
            None, canonical_intent
        )
        must_replay_padding = bool(
            top_padding
            or bottom_padding
            or final_placement_certificate is not None
            or edge_rail_declared
            or canonical_compaction_requested
        )
        if must_replay_padding:
            from clay.layout.place.final_placement_state import (
                final_placement_state_digest,
                verify_placement_causality_basis,
            )
            from clay.layout.place_blocks_2d import (
                FinalPlacementCertificate,
                placement_certificate_is_signable,
            )
            from clay.layout.route.route_cell import (
                _final_placement_provenance_digest,
                build_final_placement_resource_evidence,
            )

            if type(final_placement_certificate) is not FinalPlacementCertificate:
                raise TypeError(
                    "padded guard-ring checking requires FinalPlacementCertificate"
                )
            if not placement_certificate_is_signable(final_placement_certificate):
                raise ValueError("padded guard-ring placement is not signable")
            replayed_state = final_placement_state_digest(
                final_placement_certificate
            )
            if replayed_state != placement_state_sha256:
                raise ValueError("padded guard-ring placement state does not replay")
            verify_placement_causality_basis(final_placement_certificate)
            resource_evidence = build_final_placement_resource_evidence(
                dict(placement),
                dict(geometry),
                canonical_intent,
                placement_tech,
                final_placement_certificate=final_placement_certificate,
            )
            if _final_placement_provenance_digest(
                resource_evidence, final_placement_certificate
            ) != replayed_state:
                raise ValueError("padded guard-ring placement provenance differs")
            truth_claim = (
                exact_artifact.truth_manifest.final_placement_certificate_sha256
            )
            if truth_claim is not None and truth_claim != replayed_state:
                raise ValueError("exact GDS binds another final placement state")
            expected_top, expected_bottom = replay_canonical_member_ring_padding(
                source_cell=canonical_source_cell,
                intent=canonical_intent,
                geometry=dict(geometry),
                placement_um=dict(placement),
                tech=placement_tech,
            )
            if top_padding != expected_top:
                raise ValueError(
                    "guard-ring top padding differs from canonical materializer replay"
                )
            if bottom_padding != expected_bottom:
                raise ValueError(
                    "guard-ring bottom padding differs from canonical materializer replay"
                )
        transformed_layers = _certified_assembly_transformed_layers(
            exact_artifact, gds_path, geometry, placement
        )
        shapes_by_layer = _read_top_cell_shapes(gds_path)
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        return blocked_all(
            "exact guard-ring artifact cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )

    from clay.layout.extract.realize_render import (
        _placed_stamp_bboxes_nm,
        _placed_stamp_layer_bboxes_nm,
    )
    from clay.layout.guard_ring import build_guard_ring
    from clay.layout.tech import LayoutTech

    gds_sha256 = exact_artifact.scope.exact_gds_sha256
    try:
        resolved_tech = (
            tech if isinstance(tech, LayoutTech) else LayoutTech.resolve(str(tech))
        )
        stamp_bboxes = _placed_stamp_bboxes_nm(dict(geometry), dict(placement))
        stamp_layer_bboxes = _placed_stamp_layer_bboxes_nm(
            dict(geometry), dict(placement)
        )
    except Exception as exc:  # noqa: BLE001 - typed checker-input boundary
        return blocked_all(
            "guard-ring checker inputs cannot be resolved: "
            f"{type(exc).__name__}: {exc}",
            gds_sha256=gds_sha256,
        )
    route_entries = tuple(exact_artifact.route_manifest.entries)
    results = []

    for declaration in declarations:
        members = tuple(str(member) for member in declaration.members)
        well = str(declaration.well)
        tie_net = str(declaration.tie_net)
        hug = str(getattr(declaration, "hug", "") or "")
        role = str(getattr(declaration, "role", "") or "")
        common = dict(
            members=members,
            well=well,
            tie_net=tie_net,
            hug=hug,
            role=role,
            declaration_sha256=str(declaration.sha256),
            gds_sha256=gds_sha256,
            member_classes=tuple(
                (member, member_class_by_name.get(member, ""))
                for member in members
            ),
        )
        if hug not in ("", "tight"):
            results.append(GdsGuardRingObligation(
                **common,
                status="unsupported",
                measured_on_gds=False,
                reason=f"unsupported @guard_ring hug={hug!r}",
            ))
            continue
        if str(getattr(declaration, "enclose", "") or ""):
            results.append(GdsGuardRingObligation(
                **common,
                status="unsupported",
                measured_on_gds=False,
                reason="absolute @guard_ring enclose is unsupported",
            ))
            continue

        member_boxes = []
        claimed_emitter_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
        claimed_emitter_layer_bboxes: Dict[
            str, Dict[Tuple[int, int], Tuple[int, int, int, int]]
        ] = {}
        member_emitters = []
        ownership_error = ""
        for member in members:
            try:
                emitter_name = _unique_emitter_owner(member, geometry)
            except ValueError as exc:
                ownership_error = str(exc)
                break
            stamp = geometry.get(emitter_name)
            position = placement.get(emitter_name)
            if stamp is None or position is None or emitter_name not in stamp_bboxes:
                ownership_error = f"emitter {emitter_name!r} has no placed artwork"
                break
            aggregate_members = tuple(
                str(name)
                for name in (getattr(stamp, "composite_members", ()) or ())
            )
            if emitter_name != member and not set(aggregate_members) <= set(members):
                ownership_error = (
                    f"emitter {emitter_name!r} spans outside the declared ring"
                )
                break
            member_emitters.append(emitter_name)
            if emitter_name in claimed_emitter_bboxes:
                member_boxes.append(claimed_emitter_bboxes[emitter_name])
                continue
            expected_member = _emitted_shape_witness(
                stamp,
                position,
                excluded_layers=transformed_layers,
            )
            window = _union_bbox(
                [shape.bbox for shapes in expected_member.values() for shape in shapes]
            )
            if not expected_member or window is None:
                ownership_error = f"member {member!r} has no exact artwork witness"
                break
            candidates = _candidate_gds_shapes(
                window, shapes_by_layer, tuple(expected_member), max(int(eps), 0)
            )
            claimed_member, _transform = _match_owned_artwork(
                expected_member, candidates, max(int(eps), 0)
            )
            if claimed_member is None:
                ownership_error = (
                    f"member {member!r} artwork is missing from exact GDS"
                )
                break
            bbox = tuple(
                int(value) for value in stamp_bboxes[emitter_name]
            )
            claimed_emitter_bboxes[emitter_name] = bbox
            emitter_layers = stamp_layer_bboxes.get(emitter_name)
            if not emitter_layers:
                ownership_error = (
                    f"emitter {emitter_name!r} has no layer-bbox authority"
                )
                break
            claimed_emitter_layer_bboxes[emitter_name] = emitter_layers
            member_boxes.append(bbox)
        if ownership_error or len(member_boxes) != len(members):
            results.append(GdsGuardRingObligation(
                **common,
                status="unmeasurable",
                measured_on_gds=True,
                reason=ownership_error or "guard-ring member ownership is incomplete",
            ))
            continue

        declared_member_classes = tuple(
            member_class_by_name.get(member, "") for member in members
        )
        if any(not value for value in declared_member_classes):
            results.append(GdsGuardRingObligation(
                **common,
                status="unmeasurable",
                measured_on_gds=True,
                reason="guard-ring role has no canonical member-class evidence",
            ))
            continue
        if role == "tap" and any(
            value != "mos" for value in declared_member_classes
        ):
            results.append(GdsGuardRingObligation(
                **common,
                status="fail",
                measured_on_gds=True,
                reason="guard-ring role='tap' requires exact MOS members",
            ))
            continue
        if role == "isolation" and any(
            value != "passive" for value in declared_member_classes
        ):
            results.append(GdsGuardRingObligation(
                **common,
                status="fail",
                measured_on_gds=True,
                reason="guard-ring role='isolation' requires exact passive members",
            ))
            continue

        top_pad_by_member = dict(top_padding)
        bottom_pad_by_member = dict(bottom_padding)
        inner = (
            min(box[0] for box in member_boxes),
            min(box[1] for box in member_boxes)
            - max((bottom_pad_by_member.get(member, 0) for member in members), default=0),
            max(box[2] for box in member_boxes),
            max(box[3] for box in member_boxes)
            + max((top_pad_by_member.get(member, 0) for member in members), default=0),
        )
        try:
            member_polarities = tuple(
                member_polarity_by_name.get(member, "") for member in members
            )
            if any(not polarity for polarity in member_polarities):
                raise ValueError(
                    "canonical source lacks guard-ring member polarity"
                )
            mos_members = all(
                value == "mos" for value in declared_member_classes
            )
            shared_well = bool(
                mos_members
                and well == "n"
                and all(polarity == "pmos" for polarity in member_polarities)
            )
            protected_kwargs = {}
            if mos_members:
                if len(set(member_polarities)) != 1:
                    raise ValueError(
                        "MOS ring members lack one exact polarity"
                    )
                from clay.layout.device_cell import _resolve_layer_keys

                polarity = member_polarities[0]
                keys = _resolve_layer_keys(resolved_tech, polarity)
                unique_emitters = tuple(sorted(set(member_emitters)))

                def protected_union(key, label, *, required=True):
                    boxes = []
                    for emitter_name in unique_emitters:
                        box = claimed_emitter_layer_bboxes.get(
                            emitter_name, {}
                        ).get(key)
                        if box is None:
                            if not required:
                                continue
                            raise ValueError(
                                f"emitter {emitter_name!r} lacks {label} bbox authority"
                            )
                        boxes.append(box)
                    if not boxes:
                        if not required:
                            return None
                        raise ValueError(
                            f"tap ring lacks {label} bbox authority"
                        )
                    bottom_pad = max(
                        (
                            bottom_pad_by_member.get(member, 0)
                            for member in members
                        ),
                        default=0,
                    )
                    top_pad = max(
                        (
                            top_pad_by_member.get(member, 0)
                            for member in members
                        ),
                        default=0,
                    )
                    return (
                        min(box[0] for box in boxes),
                        min(box[1] for box in boxes) - bottom_pad,
                        max(box[2] for box in boxes),
                        max(box[3] for box in boxes) + top_pad,
                    )

                protected_kwargs = {
                    "protected_active_bbox": protected_union(
                        keys.active, "active"
                    ),
                    "protected_poly_bbox": protected_union(
                        keys.poly, "poly"
                    ),
                    "protected_metal1_bbox": protected_union(
                        keys.metal1, "metal1"
                    ),
                    "protected_same_implant_bbox": (
                        protected_union(keys.tap_implant, "same implant", required=False)
                        if bool(resolved_tech.bulk_topology().get("implant_distinguished", True))
                        else None
                    ),
                }
            rebuilt = build_guard_ring(
                resolved_tech,
                inner,
                well=well,
                nets={"tap": tie_net} if tie_net else None,
                extra_clearance_nm=0,
                shared_well=shared_well,
                **protected_kwargs,
            )
            required_roles = tuple(
                role_name
                for role_name in ("tap_diff", "tap_implant", "contact", "metal1")
                if role_name in rebuilt.layers_drawn
            )
            required_keys = tuple(rebuilt.layers_drawn[name] for name in required_roles)
            from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

            expected_ring = _materialized_guard_ring_witness(
                rebuilt.polygons,
                layer_keys=required_keys,
                manufacturing_grid_nm=pdk_manufacturing_grid_nm(resolved_tech),
            )
            claimed_ring, missing = _exact_shape_multiset(
                expected_ring, shapes_by_layer, eps=max(int(eps), 0)
            )
        except Exception as exc:  # noqa: BLE001 - PDK unsupported is typed evidence
            results.append(GdsGuardRingObligation(
                **common,
                status="unsupported",
                measured_on_gds=True,
                reason=f"PDK cannot rebuild declared tight ring: {type(exc).__name__}: {exc}",
            ))
            continue
        if claimed_ring is None:
            results.append(GdsGuardRingObligation(
                **common,
                status="fail",
                measured_on_gds=True,
                reason=(
                    "independently rebuilt PDK-min guard-ring geometry is missing "
                    f"from exact GDS ({missing} shapes absent)"
                ),
            ))
            continue

        tap_key = tuple(int(value) for value in rebuilt.layers_drawn["tap_diff"])
        tap_region = _region_from_shapes(claimed_ring.get(tap_key, ()))
        holes = sum(polygon.holes() for polygon in tap_region.each())
        if holes < 1:
            results.append(GdsGuardRingObligation(
                **common,
                status="fail",
                measured_on_gds=True,
                reason="exact GDS tap diffusion is not a closed ring",
            ))
            continue

        m1_key = tuple(int(value) for value in rebuilt.layers_drawn["metal1"])
        ring_m1 = _region_from_shapes(claimed_ring.get(m1_key, ()))
        attached = set()
        import klayout.db as kdb

        for net, raw_key, raw_bbox in route_entries:
            key = tuple(int(value) for value in raw_key)
            if key != m1_key:
                continue
            routed_region = kdb.Region(kdb.Box(*(int(value) for value in raw_bbox)))
            if not routed_region.interacting(ring_m1).is_empty():
                attached.add(str(net))
        attached_nets = tuple(sorted(attached))
        if tie_net and attached != {tie_net}:
            reason = (
                "exact route manifest does not tie the ring"
                if not attached
                else f"exact ring attaches to {attached_nets}, expected {(tie_net,)}"
            )
            results.append(GdsGuardRingObligation(
                **common,
                status="fail",
                measured_on_gds=True,
                attached_nets=attached_nets,
                reason=reason,
            ))
            continue
        results.append(GdsGuardRingObligation(
            **common,
            status="pass",
            measured_on_gds=True,
            attached_nets=attached_nets,
            reason="exact GDS contains the independently rebuilt PDK-min ring",
        ))

    return tuple(results)


def verify_bands_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    band_intents: Sequence[Any],
    place_groups: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    eps: int = ARTWORK_EPS,
) -> Tuple[GdsBandObligation, ...]:
    """Replay declared relative bands from canonical source onto exact GDS.

    No placement certificate or block-band assignment is accepted as evidence.
    Member identity/class comes from reparsed source bytes; each direct member's
    complete emitted stamp must be claimed from the exact GDS before its bbox can
    participate in the bottom < center < top comparison.
    """

    from clay.layout.intent import BandIntent, PlaceGroup

    if type(band_intents) is not tuple:
        raise TypeError("band_intents must be an exact tuple")
    if type(place_groups) is not tuple:
        raise TypeError("place_groups must be an exact tuple")
    if type(eps) is not int:
        raise TypeError("band eps must be an exact integer")
    if not 0 <= eps <= ARTWORK_EPS:
        raise ValueError(f"band eps must be bounded in [0,{ARTWORK_EPS}]")
    if any(type(declaration) is not BandIntent for declaration in band_intents):
        raise TypeError("band_intents must contain exact BandIntent values")
    if len(band_intents) > 1:
        raise ValueError("exact band checker supports at most one declaration")
    if len(place_groups) > 4096:
        raise ValueError("band place-group inventory is too large")
    for group in place_groups:
        if type(group) is not PlaceGroup:
            raise TypeError("place_groups must contain exact PlaceGroup values")
        if type(group.devices) is not tuple:
            raise TypeError("band place-group devices must be an exact tuple")
        if len(group.devices) > 4096:
            raise ValueError("band place-group membership is too large")
    declarations = band_intents
    if not declarations:
        return ()

    def blocked_all(
        reason: str,
        *,
        status: Literal["unmeasurable", "unsupported"] = "unmeasurable",
        gds_sha256: str | None = None,
    ) -> Tuple[GdsBandObligation, ...]:
        return tuple(
            GdsBandObligation(
                top=str(declaration.top),
                bottom=str(declaration.bottom),
                center=str(declaration.center),
                declaration_sha256=str(declaration.sha256),
                gds_sha256=gds_sha256,
                status=status,
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declarations
        )

    if len(declarations) != 1:
        return blocked_all(
            "exact band checker supports exactly one declaration",
            status="unsupported",
        )
    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        from clay.layout.route.exact_artifact_evidence import (
            ExactEmittedArtifact,
            _canonical_source_member_inventory,
            _replayed_source_cell_authority,
        )

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("exact band checking requires ExactEmittedArtifact")
        if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
            raise TypeError("exact band checking requires emitter mappings")
        exact_artifact.validate()
        if exact_artifact.scope.top_cell != str(top_cell):
            raise ValueError("exact band artifact is bound to another top cell")
        (
            source_sha256,
            inventory_sha256,
            _member_classes,
            canonical_source_cell,
        ) = _replayed_source_cell_authority(source_cell)
        if (
            source_sha256 != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha256
            != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("band source inventory differs from exact emitter scope")
        from clay.layout.intent import layout_intent_from_cell_def
        canonical_intent = layout_intent_from_cell_def(canonical_source_cell)
        if tuple(canonical_intent.bands) != declarations:
            raise ValueError("band declarations differ from canonical source bytes")
        if tuple(canonical_intent.place_groups) != place_groups:
            raise ValueError("band place groups differ from canonical source bytes")
        inventory = _canonical_source_member_inventory(canonical_source_cell)
        transformed_layers = _certified_assembly_transformed_layers(
            exact_artifact, gds_path, geometry, placement
        )
        shapes_by_layer = _read_top_cell_shapes(gds_path)
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        return blocked_all(
            f"exact band artifact cannot be replayed: {type(exc).__name__}: {exc}"
        )

    declaration = declarations[0]
    class_by_name = {str(row[0]): str(row[1]).lower() for row in inventory}
    center_token = str(declaration.center)
    if center_token == "passives":
        center_members = tuple(
            sorted(
                name
                for name, device_class in class_by_name.items()
                if device_class in {"res", "cap"}
            )
        )
    else:
        groups = tuple(
            group
            for group in tuple(place_groups or ())
            if getattr(group, "group", None) == center_token
        )
        if len(groups) != 1:
            return blocked_all(
                f"band center group {center_token!r} is not uniquely declared",
                status="unsupported",
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
            )
        center_members = tuple(sorted(str(name) for name in groups[0].devices))
        if not center_members or any(name not in class_by_name for name in center_members):
            return blocked_all(
                f"band center group {center_token!r} has no complete source inventory",
                status="unsupported",
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
            )
        if any(class_by_name[name] not in {"res", "cap"} for name in center_members):
            return blocked_all(
                f"band center group {center_token!r} contains a non-passive member",
                status="unsupported",
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
            )
    center_set = set(center_members)
    pmos_classes = {"pmos", "pmos3v", "pch", "pch3", "pd"}
    nmos_classes = {"nmos", "nmos3v", "nch", "nch3", "nd"}
    pins_by_name = {str(row[0]): dict(row[4]) for row in inventory}
    top_members = tuple(sorted(
        name for name, device_class in class_by_name.items()
        if device_class in pmos_classes
        and name not in center_set
        and str(declaration.top)
        in {
            str(pins_by_name[name].get("s", "")),
            str(pins_by_name[name].get("b", "")),
        }
    ))
    bottom_members = tuple(sorted(
        name for name, device_class in class_by_name.items()
        if device_class in nmos_classes
        and name not in center_set
        and str(declaration.bottom)
        in {
            str(pins_by_name[name].get("s", "")),
            str(pins_by_name[name].get("b", "")),
        }
    ))
    if not bottom_members or not center_members or not top_members:
        return blocked_all(
            "band requires non-empty source-bound bottom, center, and top members",
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
        )

    member_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    claimed_emitter_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    emitter_band: Dict[str, str] = {}
    band_members = (
        ("bottom", bottom_members),
        ("center", center_members),
        ("top", top_members),
    )
    for band_name, members in band_members:
        for member in members:
            try:
                emitter_name = _unique_emitter_owner(member, geometry)
            except ValueError as exc:
                return blocked_all(
                    f"band {exc}",
                    gds_sha256=exact_artifact.scope.exact_gds_sha256,
                )
            prior_band = emitter_band.setdefault(emitter_name, band_name)
            if prior_band != band_name:
                return blocked_all(
                    f"band emitter {emitter_name!r} spans multiple bands",
                    gds_sha256=exact_artifact.scope.exact_gds_sha256,
                )
            if emitter_name in claimed_emitter_bboxes:
                member_bboxes[member] = claimed_emitter_bboxes[emitter_name]
                continue
            stamp = geometry.get(emitter_name)
            position = placement.get(emitter_name)
            if stamp is None or position is None:
                return blocked_all(
                    f"band emitter {emitter_name!r} has no placed artwork",
                    gds_sha256=exact_artifact.scope.exact_gds_sha256,
                )
            try:
                expected = _emitted_shape_witness(
                    stamp,
                    position,
                    excluded_layers=transformed_layers,
                )
                window = _union_bbox(
                    [shape.bbox for shapes in expected.values() for shape in shapes]
                )
                if not expected or window is None:
                    raise ValueError("empty artwork witness")
                candidates = _candidate_gds_shapes(
                    window, shapes_by_layer, tuple(expected), max(int(eps), 0)
                )
                claimed, _transform = _match_owned_artwork(
                    expected, candidates, max(int(eps), 0)
                )
                if claimed is None:
                    raise ValueError("artwork is missing from exact GDS")
                claimed_bbox = _union_bbox(
                    [shape.bbox for shapes in claimed.values() for shape in shapes]
                )
                if claimed_bbox is None:
                    raise ValueError("claimed artwork has no bbox")
                bbox = tuple(int(value) for value in claimed_bbox)
                claimed_emitter_bboxes[emitter_name] = bbox
                member_bboxes[member] = bbox
            except Exception as exc:  # noqa: BLE001 - per-member typed evidence
                return blocked_all(
                    f"band member {member!r} cannot be measured: "
                    f"{type(exc).__name__}: {exc}",
                    gds_sha256=exact_artifact.scope.exact_gds_sha256,
                )

    def group_bbox(members: Sequence[str]) -> Tuple[int, int, int, int]:
        boxes = [member_bboxes[name] for name in members]
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    bottom_bbox = group_bbox(bottom_members)
    center_bbox = group_bbox(center_members)
    top_bbox = group_bbox(top_members)
    ordered = bottom_bbox[3] <= center_bbox[1] and center_bbox[3] <= top_bbox[1]
    return (
        GdsBandObligation(
            top=str(declaration.top),
            bottom=str(declaration.bottom),
            center=center_token,
            declaration_sha256=str(declaration.sha256),
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
            status="pass" if ordered else "fail",
            measured_on_gds=True,
            bottom_members=bottom_members,
            center_members=center_members,
            top_members=top_members,
            bottom_bbox_nm=bottom_bbox,
            center_bbox_nm=center_bbox,
            top_bbox_nm=top_bbox,
            reason=(
                "exact GDS satisfies relative bottom < center < top band order"
                if ordered
                else "exact GDS violates relative bottom < center < top band order"
            ),
        ),
    )


def verify_finger_array_orders_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    array_intents: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    row_program: Any,
    row_materialization: Any,
    row_final_placement_certificate: Any,
    row_final_placement: Any,
    tech: Any,
) -> Tuple[GdsFingerArrayOrderObligation, ...]:
    """Independently replay one-row ratio arrays onto the exact emitted GDS.

    The canonical source is recompiled and rematerialized at this boundary.  The
    stored row evidence may explain placement, but cannot mint a pass: its bytes,
    placed artwork digest, emitter stamp, and reread GDS regions must all match the
    independently rebuilt source/PDK result.
    """

    from clay.layout.intent import ArrayIntent, layout_intent_from_cell_def, replay_array_intent

    if type(array_intents) is not tuple:
        raise TypeError("array_intents must be an exact tuple")
    if len(array_intents) > 64:
        raise ValueError("array declaration inventory is too large")
    if any(type(item) is not ArrayIntent for item in array_intents):
        raise TypeError("array_intents must contain exact ArrayIntent values")
    all_declarations = tuple(replay_array_intent(item) for item in array_intents)
    all_sha256s = tuple(item.sha256 for item in all_declarations)
    if len(set(all_sha256s)) != len(all_sha256s):
        raise ValueError("duplicate array declarations are forbidden")
    declarations = tuple(
        item
        for item in all_declarations
        if item.style == "shared_diffusion" and item.order == "ratio"
    )
    if not declarations:
        return ()

    def blocked_all(
        reason: str,
        *,
        status: Literal["unmeasurable", "unsupported"] = "unmeasurable",
        gds_sha256: str | None = None,
    ) -> Tuple[GdsFingerArrayOrderObligation, ...]:
        return tuple(
            GdsFingerArrayOrderObligation(
                group=item.group,
                style=item.style,
                rows=item.rows,
                order=item.order,
                declaration_sha256=item.sha256,
                gds_sha256=gds_sha256,
                status=status,
                measured_on_gds=False,
                reason=reason,
            )
            for item in declarations
        )

    unsupported = tuple(item for item in declarations if item.rows != 1)
    if unsupported:
        return blocked_all(
            "exact finger-order checker supports rows=1 ratio arrays only",
            status="unsupported",
        )

    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        from clay.layout.generic_placement import (
            Placement,
            RowFinalPlacementCertificate,
            _placed_artwork_sha256,
            _translate,
            verify_stored_row_projection,
        )
        from clay.layout.place.row_program import (
            RowProgram,
            RowProgramMaterialization,
            RowStructureKind,
            compile_row_program,
            materialize_row_program,
            verify_row_program_materialization,
        )
        from clay.layout.route.exact_artifact_evidence import (
            ExactEmittedArtifact,
            _replayed_source_cell_authority,
        )

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("exact finger-order checking requires ExactEmittedArtifact")
        if type(row_program) is not RowProgram:
            raise TypeError("exact finger-order checking requires exact RowProgram")
        if type(row_materialization) is not RowProgramMaterialization:
            raise TypeError(
                "exact finger-order checking requires exact RowProgramMaterialization"
            )
        if type(row_final_placement_certificate) is not RowFinalPlacementCertificate:
            raise TypeError(
                "exact finger-order checking requires exact RowFinalPlacementCertificate"
            )
        if type(row_final_placement) is not Placement:
            raise TypeError(
                "exact finger-order checking requires exact final row Placement"
            )
        if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
            raise TypeError("exact finger-order checking requires emitter mappings")
        if tech is None:
            raise TypeError("exact finger-order checking requires resolved layout tech")
        exact_artifact.validate()
        if exact_artifact.scope.top_cell != str(top_cell):
            raise ValueError("exact finger-order artifact is bound to another top cell")
        (
            source_sha256,
            inventory_sha256,
            _member_classes,
            canonical_source_cell,
        ) = _replayed_source_cell_authority(source_cell)
        if (
            source_sha256 != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha256
            != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("array source inventory differs from emitter scope")
        canonical_intent = layout_intent_from_cell_def(canonical_source_cell)
        canonical_arrays = tuple(
            replay_array_intent(item) for item in canonical_intent.arrays
        )
        if canonical_arrays != all_declarations:
            raise ValueError("array declarations differ from canonical source bytes")

        replayed_program = compile_row_program(
            canonical_source_cell, canonical_intent, tech
        )
        if replayed_program.residuals:
            raise ValueError(
                "canonical row compiler produced residuals: "
                f"{tuple(item.code for item in replayed_program.residuals)}"
            )
        if replayed_program.to_canonical_bytes() != row_program.to_canonical_bytes():
            raise ValueError("stored row program differs from canonical source replay")
        program_member_names = {item.device for item in replayed_program.members}
        canonical_passive_names = {
            name for name, member_class in _member_classes
            if member_class == "passive"
        }
        canonical_bjt_names = {
            str(getattr(dev, "instance_name", ""))
            for dev in (getattr(canonical_source_cell, "devices", ()) or ())
            if str(getattr(dev, "device_class", "") or "").lower() in ("pnp", "npn", "bjt")
            and str(getattr(dev, "instance_name", ""))
        }
        expected_placed_members = (
            program_member_names | canonical_passive_names | canonical_bjt_names
        )
        actual_placed_members = set(row_final_placement.devices)
        if actual_placed_members != expected_placed_members:
            missing_placed_members = tuple(sorted(
                expected_placed_members - actual_placed_members
            ))
            unexpected_placed_members = tuple(sorted(
                actual_placed_members - expected_placed_members
            ))
            raise ValueError(
                "final row placement member inventory differs from canonical "
                f"source authority: missing={missing_placed_members}, "
                f"unexpected={unexpected_placed_members}"
            )
        verify_row_program_materialization(
            row_materialization,
            program=row_program,
            tech_id=row_program.tech_id,
        )
        verify_stored_row_projection(row_final_placement)
        if (
            row_final_placement.row_program is None
            or row_final_placement.row_program.to_canonical_bytes()
            != row_program.to_canonical_bytes()
        ):
            raise ValueError("final row placement binds another row program")
        if (
            row_final_placement.row_materialization is None
            or row_final_placement.row_materialization.certificate.to_canonical_bytes()
            != row_materialization.certificate.to_canonical_bytes()
        ):
            raise ValueError("final row placement binds another row materialization")
        if (
            row_final_placement.row_final_certificate is None
            or row_final_placement.row_final_certificate.to_canonical_bytes()
            != row_final_placement_certificate.to_canonical_bytes()
        ):
            raise ValueError("final row placement binds another row certificate")
        replayed_materialization = materialize_row_program(replayed_program, tech)
        if replayed_materialization.residuals:
            raise ValueError(
                "canonical row materializer produced residuals: "
                f"{tuple(item.code for item in replayed_materialization.residuals)}"
            )
        if (
            replayed_materialization.certificate.to_canonical_bytes()
            != row_materialization.certificate.to_canonical_bytes()
        ):
            raise ValueError("stored row materialization differs from canonical replay")

        final = row_final_placement_certificate
        rebuilt_final = RowFinalPlacementCertificate(
            schema=final.schema,
            program_sha256=final.program_sha256,
            materialization_sha256=final.materialization_sha256,
            macro_bindings=final.macro_bindings,
            free_device_blocks=final.free_device_blocks,
            bbox_nm=final.bbox_nm,
            occupied_area_nm2=final.occupied_area_nm2,
            bbox_area_nm2=final.bbox_area_nm2,
            void_area_nm2=final.void_area_nm2,
            whitespace_area_nm2=final.whitespace_area_nm2,
            largest_empty_rectangle_nm2=final.largest_empty_rectangle_nm2,
            placed_artwork_sha256=final.placed_artwork_sha256,
            physical_macro_graph_sha256=final.physical_macro_graph_sha256,
            row_macro_contracts_sha256=final.row_macro_contracts_sha256,
            native_geometry_binding=final.native_geometry_binding,
        )
        if rebuilt_final.to_canonical_bytes() != final.to_canonical_bytes():
            raise ValueError("row final placement certificate canonical replay mismatch")
        if final.program_sha256 != replayed_program.sha256:
            raise ValueError("row final placement binds another program")
        if final.materialization_sha256 != replayed_materialization.certificate.sha256:
            raise ValueError("row final placement binds another materialization")

        transformed_layers = _certified_assembly_transformed_layers(
            exact_artifact, gds_path, geometry, placement
        )
        shapes_by_layer = _read_top_cell_shapes(gds_path)
        excluded_layers = set(transformed_layers)
        entries = {item.group: item for item in replayed_program.entries}
        realizations = {
            item.entry_id: item for item in replayed_materialization.realizations
        }
        bindings = {item.entry_id: item for item in final.macro_bindings}
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        return blocked_all(
            "exact finger-order row/artwork evidence cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )

    obligations: list[GdsFingerArrayOrderObligation] = []
    claimed_regions_by_layer: Dict[Tuple[int, int], Any] = {}
    for declaration in declarations:
        try:
            entry = entries.get(declaration.group)
            if entry is None:
                raise ValueError("declared array has no canonical row entry")
            if (
                entry.structure_kind is not RowStructureKind.SHARED_OD
                or entry.rows != declaration.rows
                or entry.order != declaration.order
            ):
                raise ValueError("canonical row entry differs from array declaration")
            realization = realizations.get(entry.entry_id)
            binding = bindings.get(entry.entry_id)
            if realization is None or binding is None:
                raise ValueError("declared array lacks materialization/final binding")
            if (
                binding.entry_sha256 != entry.sha256
                or binding.materialization_sha256
                != replayed_materialization.certificate.sha256
                or binding.owner != entry.owner
                or binding.consumed_devices != entry.consumed_devices
            ):
                raise ValueError("declared array final binding differs from row replay")
            lx0, ly0, lx1, ly1 = realization.local_bbox_nm
            bx0, by0, bx1, by1 = binding.bbox_nm
            translations = {(bx0 - lx0, by0 - ly0), (bx1 - lx1, by1 - ly1)}
            if len(translations) != 1:
                raise ValueError("declared array final binding is not one rigid translation")
            dx_nm, dy_nm = next(iter(translations))
            placed_polygons = _translate(
                list(getattr(realization.geometry, "polygons", ()) or ()),
                dx_nm,
                dy_nm,
            )
            if _placed_artwork_sha256(placed_polygons) != binding.artwork_sha256:
                raise ValueError("declared array placed artwork digest does not replay")

            expected: Dict[Tuple[int, int], list[_ShapeGeometry]] = {}
            for layer_polygon in placed_polygons:
                key = (
                    int(getattr(layer_polygon, "layer")),
                    int(getattr(layer_polygon, "datatype")),
                )
                if key in excluded_layers:
                    continue
                vertices = tuple(
                    (int(vertex.x), int(vertex.y))
                    for vertex in tuple(layer_polygon.polygon.vertices)
                )
                shape = _shape_geometry(vertices)
                if shape is None:
                    raise ValueError("declared array rebuild contains degenerate artwork")
                expected.setdefault(key, []).append(shape)
            expected = _merged_shape_witness(expected)
            stamp = geometry.get(declaration.group)
            position = placement.get(declaration.group)
            if stamp is None or position is None:
                raise ValueError("declared array has no emitter artwork")
            emitted = _merged_shape_witness(
                _emitted_shape_witness(
                    stamp, position, excluded_layers=transformed_layers
                )
            )
            if set(emitted) != set(expected):
                raise ValueError("declared array emitter artwork layer set differs")
            pending: Dict[Tuple[int, int], Any] = {}
            for key in sorted(expected):
                expected_region = _region_from_shapes(expected[key])
                emitted_region = _region_from_shapes(emitted[key])
                if not (expected_region - emitted_region).is_empty() or not (
                    emitted_region - expected_region
                ).is_empty():
                    raise ValueError("declared array emitter artwork differs from replay")
                actual_region = _region_from_shapes(shapes_by_layer.get(key, ()))
                if not (expected_region - actual_region).is_empty():
                    raise ValueError("declared array artwork is missing from exact GDS")
                prior = claimed_regions_by_layer.get(key)
                if prior is not None and not (expected_region & prior).is_empty():
                    raise ValueError("declared arrays reuse exact GDS artwork")
                pending[key] = expected_region
            for key, region in pending.items():
                if key in claimed_regions_by_layer:
                    claimed_regions_by_layer[key] += region
                else:
                    claimed_regions_by_layer[key] = region

            active = set(entry.active_devices)
            owner_sequence = tuple(
                str(finger.member)
                for finger in tuple(getattr(realization.geometry, "fingers", ()) or ())
                if str(getattr(finger, "member", "")) in active
            )
            expected_count = sum(
                member.fingers
                for member in replayed_program.members
                if member.device in active
            )
            if len(owner_sequence) != expected_count or set(owner_sequence) != active:
                raise ValueError("declared array owner sequence is incomplete")
            obligations.append(GdsFingerArrayOrderObligation(
                group=declaration.group,
                style=declaration.style,
                rows=declaration.rows,
                order=declaration.order,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="pass",
                measured_on_gds=True,
                owner_sequence=owner_sequence,
                program_sha256=replayed_program.sha256,
                materialization_sha256=replayed_materialization.certificate.sha256,
                geometry_sha256=realization.geometry_sha256,
                reason=(
                    "canonical source/PDK row replay and exact GDS artwork preserve "
                    "the declared ratio finger order"
                ),
            ))
        except Exception as exc:  # noqa: BLE001 - one typed array obligation
            obligations.append(GdsFingerArrayOrderObligation(
                group=declaration.group,
                style=declaration.style,
                rows=declaration.rows,
                order=declaration.order,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=(
                    "exact finger-order artwork cannot be measured: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ))
    return tuple(obligations)


def verify_common_centroid_interdigitation_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    common_centroid_intents: Sequence[Any],
    matching_intents: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    row_program: Any,
    row_materialization: Any,
    row_final_placement_certificate: Any,
    row_final_placement: Any,
    tech: Any,
) -> Tuple[
    Tuple[GdsCommonCentroidObligation, ...],
    Tuple[GdsInterdigitationObligation, ...],
]:
    """Measure point reflection and finger interleave from reread exact GDS.

    Placement ``cluster`` metadata is deliberately ignored.  The canonical source
    and PDK are independently recompiled only to name the expected rigid macro.
    Finger channel rectangles are then reconstructed from ``poly & active`` in the
    emitted GDS, assigned uniquely to the canonical rows, and used for both the
    point-reflection and interleave verdicts.
    """

    from collections import Counter

    from clay.layout.intent import (
        CommonCentroidIntent,
        MatchingIntent,
        layout_intent_from_cell_def,
        replay_common_centroid_intent,
        replay_matching_intent,
    )

    if type(common_centroid_intents) is not tuple:
        raise TypeError("common_centroid_intents must be an exact tuple")
    if type(matching_intents) is not tuple:
        raise TypeError("matching_intents must be an exact tuple")
    if len(common_centroid_intents) > 64 or len(matching_intents) > 256:
        raise ValueError("centroid/interdigitation declaration inventory is too large")
    centroids = tuple(
        replay_common_centroid_intent(item) for item in common_centroid_intents
    )
    matchings = tuple(replay_matching_intent(item) for item in matching_intents)
    if len({item.sha256 for item in centroids}) != len(centroids):
        raise ValueError("duplicate common-centroid declarations are forbidden")
    interdigitations = tuple(
        item for item in matchings if item.requires_interdigitation
    )
    if len({item.sha256 for item in interdigitations}) != len(interdigitations):
        raise ValueError("duplicate interdigitation declarations are forbidden")
    if not centroids and not interdigitations:
        return (), ()

    def blocked_centroids(
        reason: str,
        *,
        status: Literal["unmeasurable", "unsupported"] = "unmeasurable",
        gds_sha256: str | None = None,
    ) -> Tuple[GdsCommonCentroidObligation, ...]:
        return tuple(
            GdsCommonCentroidObligation(
                group=item.group,
                members=("", ""),
                rows=item.rows,
                declaration_sha256=item.sha256,
                gds_sha256=gds_sha256,
                status=status,
                measured_on_gds=False,
                reason=reason,
            )
            for item in centroids
        )

    def blocked_interdigitations(
        reason: str,
        *,
        status: Literal["unmeasurable", "unsupported"] = "unmeasurable",
        gds_sha256: str | None = None,
    ) -> Tuple[GdsInterdigitationObligation, ...]:
        return tuple(
            GdsInterdigitationObligation(
                members=item.members,
                declaration_ids=item.interdigitation_declaration_ids,
                pattern=item.interdigitation_pattern,
                finger_pairs=item.interdigitation_finger_pairs,
                declaration_sha256=item.sha256,
                gds_sha256=gds_sha256,
                status=status,
                measured_on_gds=False,
                reason=reason,
            )
            for item in interdigitations
        )

    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        import klayout.db as kdb

        from clay.layout.common_centroid_cluster import _resolve_layer_keys
        from clay.layout.generic_placement import (
            Placement,
            RowFinalPlacementCertificate,
            _placed_artwork_sha256,
            _translate,
            verify_stored_row_projection,
        )
        from clay.layout.place.row_program import (
            RowProgram,
            RowProgramMaterialization,
            RowStructureKind,
            compile_row_program,
            materialize_row_program,
            verify_row_program_materialization,
        )
        from clay.layout.route.exact_artifact_evidence import (
            ExactEmittedArtifact,
            _replayed_source_cell_authority,
        )

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("exact centroid checking requires ExactEmittedArtifact")
        if type(row_program) is not RowProgram:
            raise TypeError("exact centroid checking requires exact RowProgram")
        if type(row_materialization) is not RowProgramMaterialization:
            raise TypeError(
                "exact centroid checking requires exact RowProgramMaterialization"
            )
        if type(row_final_placement_certificate) is not RowFinalPlacementCertificate:
            raise TypeError(
                "exact centroid checking requires exact RowFinalPlacementCertificate"
            )
        if type(row_final_placement) is not Placement:
            raise TypeError("exact centroid checking requires exact final row Placement")
        if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
            raise TypeError("exact centroid checking requires emitter mappings")
        if tech is None:
            raise TypeError("exact centroid checking requires resolved layout tech")
        exact_artifact.validate()
        if exact_artifact.scope.top_cell != str(top_cell):
            raise ValueError("exact centroid artifact is bound to another top cell")
        (
            source_sha256,
            inventory_sha256,
            member_classes,
            canonical_source_cell,
        ) = _replayed_source_cell_authority(source_cell)
        if (
            source_sha256 != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha256
            != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("centroid source inventory differs from emitter scope")
        canonical_intent = layout_intent_from_cell_def(canonical_source_cell)
        canonical_centroids = tuple(
            replay_common_centroid_intent(item)
            for item in canonical_intent.common_centroids
        )
        canonical_matchings = tuple(
            replay_matching_intent(item) for item in canonical_intent.matching_intents
        )
        if canonical_centroids != centroids:
            raise ValueError(
                "common-centroid declarations differ from canonical source bytes"
            )
        if canonical_matchings != matchings:
            raise ValueError("matching declarations differ from canonical source bytes")

        canonical_program = compile_row_program(
            canonical_source_cell, canonical_intent, tech
        )
        projected_source = getattr(row_materialization, 'source_projection', None) is not None
        if projected_source:
            from clay.layout.generic_placement import verify_row_source_compiler_authority
            replayed_program, replayed_materialization = verify_row_source_compiler_authority(
                row_final_placement, canonical_source_cell, tech)
            if replayed_materialization is not row_materialization or row_final_placement.row_program is not row_program:
                raise ValueError('exact centroid requires the retained projected compiler objects')
        else:
            replayed_program = canonical_program
        if replayed_program.residuals:
            raise ValueError(
                "canonical row compiler produced residuals: "
                f"{tuple(item.code for item in replayed_program.residuals)}"
            )
        if replayed_program.to_canonical_bytes() != row_program.to_canonical_bytes():
            raise ValueError("stored row program differs from canonical source replay")
        program_members = {item.device for item in replayed_program.members}
        canonical_passives = {
            name for name, member_class in member_classes if member_class == "passive"
        }
        canonical_bjts = {
            str(getattr(dev, "instance_name", ""))
            for dev in (getattr(canonical_source_cell, "devices", ()) or ())
            if str(getattr(dev, "device_class", "") or "").lower() in ("pnp", "npn", "bjt")
            and str(getattr(dev, "instance_name", ""))
        }
        if set(row_final_placement.devices) != program_members | canonical_passives | canonical_bjts:
            raise ValueError(
                "final row placement member inventory differs from canonical source"
            )
        verify_row_program_materialization(
            row_materialization,
            program=row_program,
            tech_id=row_program.tech_id,
        )
        verify_stored_row_projection(row_final_placement)
        if (
            row_final_placement.row_program is None
            or row_final_placement.row_program.to_canonical_bytes()
            != row_program.to_canonical_bytes()
        ):
            raise ValueError("final row placement binds another row program")
        if (
            row_final_placement.row_materialization is None
            or row_final_placement.row_materialization.certificate.to_canonical_bytes()
            != row_materialization.certificate.to_canonical_bytes()
        ):
            raise ValueError("final row placement binds another row materialization")
        if (
            row_final_placement.row_final_certificate is None
            or row_final_placement.row_final_certificate.to_canonical_bytes()
            != row_final_placement_certificate.to_canonical_bytes()
        ):
            raise ValueError("final row placement binds another row certificate")
        if not projected_source:
            replayed_materialization = materialize_row_program(replayed_program, tech)
        if replayed_materialization.residuals:
            raise ValueError(
                "canonical row materializer produced residuals: "
                f"{tuple(item.code for item in replayed_materialization.residuals)}"
            )
        if (
            replayed_materialization.certificate.to_canonical_bytes()
            != row_materialization.certificate.to_canonical_bytes()
        ):
            raise ValueError("stored row materialization differs from canonical replay")
        final = row_final_placement_certificate
        if final.program_sha256 != replayed_program.sha256:
            raise ValueError("row final placement binds another program")
        if final.materialization_sha256 != replayed_materialization.certificate.sha256:
            raise ValueError("row final placement binds another materialization")

        transformed_layers = _certified_assembly_transformed_layers(
            exact_artifact, gds_path, geometry, placement
        )
        shapes_by_layer = _read_top_cell_shapes(gds_path)
        excluded_layers = set(transformed_layers)
        entries = {item.group: item for item in replayed_program.entries}
        canonical_entries = {item.group: item for item in canonical_program.entries}
        realizations = {
            item.entry_id: item for item in replayed_materialization.realizations
        }
        bindings = {item.entry_id: item for item in final.macro_bindings}
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        reason = (
            "exact centroid row/artwork evidence cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )
        return blocked_centroids(reason), blocked_interdigitations(reason)

    centroid_obligations: list[GdsCommonCentroidObligation] = []
    measurements: Dict[
        str,
        Tuple[
            Tuple[str, str],
            Tuple[Tuple[str, ...], ...],
            Tuple[Tuple[str, int, int], ...],
            bool,
            str,
        ],
    ] = {}
    claimed_regions_by_layer: Dict[Tuple[int, int], Any] = {}
    proved_host_entries = set()
    claimed_centroid_members = set()
    for declaration in centroids:
        members: Tuple[str, str] = ("", "")
        try:
            canonical_entry = canonical_entries.get(declaration.group)
            if canonical_entry is None:
                raise ValueError("declared common centroid has no canonical row entry")
            if (
                canonical_entry.structure_kind is not RowStructureKind.COMMON_CENTROID
                or canonical_entry.rows != declaration.rows
                or canonical_entry.orientation != declaration.orientation
                or len(canonical_entry.active_devices) != 2
            ):
                raise ValueError("canonical row entry differs from centroid declaration")
            members = tuple(canonical_entry.active_devices)
            entry = entries.get(declaration.group)
            if projected_source:
                candidates = [item for item in replayed_program.entries
                              if set(members) <= set(item.active_devices)]
                if len(candidates) != 1:
                    raise ValueError('declared centroid lacks one complete enclosing source row')
                entry = candidates[0]
            if entry is None:
                raise ValueError('declared centroid has no retained row owner')
            if set(members) & claimed_centroid_members:
                raise ValueError('declared centroids reuse source occurrences')
            claimed_centroid_members.update(members)
            realization = realizations.get(entry.entry_id)
            binding = bindings.get(entry.entry_id)
            if realization is None or binding is None:
                raise ValueError("declared common centroid lacks final row binding")
            if (
                binding.entry_sha256 != entry.sha256
                or binding.materialization_sha256
                != replayed_materialization.certificate.sha256
                or binding.owner != entry.owner
                or binding.consumed_devices != entry.consumed_devices
            ):
                raise ValueError("common-centroid final binding differs from replay")
            lx0, ly0, lx1, ly1 = realization.local_bbox_nm
            bx0, by0, bx1, by1 = binding.bbox_nm
            translations = {(bx0 - lx0, by0 - ly0), (bx1 - lx1, by1 - ly1)}
            if len(translations) != 1:
                raise ValueError("common-centroid binding is not one rigid translation")
            dx_nm, dy_nm = next(iter(translations))
            placed_polygons = _translate(
                list(getattr(realization.geometry, "polygons", ()) or ()),
                dx_nm,
                dy_nm,
            )
            if _placed_artwork_sha256(placed_polygons) != binding.artwork_sha256:
                raise ValueError("common-centroid placed artwork digest does not replay")

            expected: Dict[Tuple[int, int], list[_ShapeGeometry]] = {}
            for layer_polygon in placed_polygons:
                key = (
                    int(getattr(layer_polygon, "layer")),
                    int(getattr(layer_polygon, "datatype")),
                )
                if key in excluded_layers:
                    continue
                vertices = tuple(
                    (int(vertex.x), int(vertex.y))
                    for vertex in tuple(layer_polygon.polygon.vertices)
                )
                shape = _shape_geometry(vertices)
                if shape is None:
                    raise ValueError("common-centroid replay contains degenerate artwork")
                expected.setdefault(key, []).append(shape)
            expected = _merged_shape_witness(expected)
            stamp = geometry.get(entry.group)
            position = placement.get(entry.group)
            if stamp is None or position is None:
                raise ValueError("common-centroid group has no emitter artwork")
            emitted = _merged_shape_witness(
                _emitted_shape_witness(
                    stamp, position, excluded_layers=transformed_layers
                )
            )
            if set(emitted) != set(expected):
                raise ValueError("common-centroid emitter artwork layer set differs")
            pending: Dict[Tuple[int, int], Any] = {}
            for key in sorted(expected):
                expected_region = _region_from_shapes(expected[key])
                emitted_region = _region_from_shapes(emitted[key])
                if not (expected_region - emitted_region).is_empty() or not (
                    emitted_region - expected_region
                ).is_empty():
                    raise ValueError("common-centroid emitter artwork differs from replay")
                actual_region = _region_from_shapes(shapes_by_layer.get(key, ()))
                if not (expected_region - actual_region).is_empty():
                    raise ValueError("common-centroid artwork is missing from exact GDS")
                prior = claimed_regions_by_layer.get(key)
                if entry.entry_id not in proved_host_entries and prior is not None and not (expected_region & prior).is_empty():
                    raise ValueError("common-centroid declarations reuse exact GDS artwork")
                pending[key] = expected_region
            if entry.entry_id not in proved_host_entries:
                for key, region in pending.items():
                    if key in claimed_regions_by_layer:
                        claimed_regions_by_layer[key] += region
                    else:
                        claimed_regions_by_layer[key] = region
                proved_host_entries.add(entry.entry_id)

            layer_keys = _resolve_layer_keys(tech, str(realization.geometry.kind))
            poly_region = _region_from_shapes(
                shapes_by_layer.get(tuple(layer_keys.poly), ())
            )
            active_region = _region_from_shapes(
                shapes_by_layer.get(tuple(layer_keys.active), ())
            )
            orientation = declaration.orientation
            measured_member_channels = []
            if getattr(realization.geometry, 'channel_inventory', None) is not None:
                from clay.layout.completeness.row_centroid_channels import measure_owned_centroid_channels
                inventory = realization.geometry.channel_inventory
                if (inventory.poly_layer_key != tuple(layer_keys.poly)
                        or tuple(layer_keys.active) not in inventory.od_layer_keys):
                    raise ValueError('centroid channel inventory differs from current PDK layers')
                observed, owner_rows = measure_owned_centroid_channels(
                    declaration=declaration, canonical_entry=canonical_entry, entry=entry,
                    geometry=realization.geometry, dx_nm=dx_nm, dy_nm=dy_nm,
                    emitted_od=active_region, emitted_poly=poly_region,
                    member_channel_geometry=measured_member_channels)
            else:
                if projected_source:
                    raise ValueError('projected centroid row lacks complete occurrence channels')
                window = kdb.Region(kdb.Box(bx0, by0, bx1, by1))
                channel_region = poly_region & active_region & window
                actual_channels = []
                for polygon in channel_region.each():
                    bbox = polygon.bbox()
                    if bbox.width() > 0 and bbox.height() > 0:
                        actual_channels.append(
                            (int(bbox.left), int(bbox.bottom), int(bbox.right), int(bbox.top))
                        )
                actual_channels.sort(key=lambda box: (box[1] + box[3], box[0] + box[2]))

                raw_fingers = tuple(getattr(realization.geometry, "fingers", ()) or ())
                if not raw_fingers:
                    raise ValueError("common-centroid replay has no finger geometry")
                orientation = declaration.orientation
                expected_fingers = []
                for finger in raw_fingers:
                    from clay.layout.completeness.row_centroid_channels import (
                        _compose_channel_orientation,
                        _continuous_array_channel_orientation,
                    )
                    finger_orientation = getattr(finger, 'orientation', None)
                    if finger_orientation == '':
                        channel_orientation = _continuous_array_channel_orientation(
                            geometry=realization.geometry, entry=entry, finger=finger)
                    else:
                        channel_orientation = _compose_channel_orientation(
                            entry.orientation, finger_orientation)
                    owner = str(getattr(finger, "member", ""))
                    gate_bbox = getattr(finger, "gate_bbox_nm", None)
                    if gate_bbox is None:
                        if orientation != "R0":
                            raise ValueError(
                                "rotated common-centroid finger lacks exact gate owner probe"
                            )
                        x0 = int(getattr(finger, "x0"))
                        x1 = int(getattr(finger, "x1"))
                        cy = int(getattr(finger, "cy"))
                        gate_bbox = (x0, cy - 1, x1, cy + 1)
                    elif (
                        type(gate_bbox) is not tuple
                        or len(gate_bbox) != 4
                        or any(type(value) is not int for value in gate_bbox)
                    ):
                        raise TypeError(
                            "common-centroid gate owner probe must be an exact four-int tuple"
                        )
                    gx0, gy0, gx1, gy1 = gate_bbox
                    if gx1 <= gx0 or gy1 <= gy0:
                        raise ValueError(
                            "common-centroid gate owner probe must have positive area"
                        )
                    expected_fingers.append((
                        owner,
                        (gx0 + dx_nm, gy0 + dy_nm, gx1 + dx_nm, gy1 + dy_nm),
                        channel_orientation,
                    ))
                used: set[int] = set()
                observed: list[Tuple[str, int, int]] = []
                for owner, probe, channel_orientation in expected_fingers:
                    px0, py0, px1, py1 = probe
                    candidates = [
                        index
                        for index, box in enumerate(actual_channels)
                        if index not in used
                        and (
                            (
                                orientation in ("R0", "R180")
                                and abs(box[0] - px0) <= ARTWORK_EPS
                                and abs(box[2] - px1) <= ARTWORK_EPS
                                and box[1] <= (py0 + py1) // 2 <= box[3]
                            )
                            or (
                                orientation in ("R90", "R270")
                                and abs(box[1] - py0) <= ARTWORK_EPS
                                and abs(box[3] - py1) <= ARTWORK_EPS
                                and box[0] <= (px0 + px1) // 2 <= box[2]
                            )
                        )
                    ]
                    if len(candidates) != 1:
                        raise ValueError(
                            "exact GDS does not uniquely identify every centroid finger"
                        )
                    index = candidates[0]
                    used.add(index)
                    if owner in members:
                        box = actual_channels[index]
                        observed.append((owner, box[0] + box[2], box[1] + box[3]))
                        measured_member_channels.append((owner, box, channel_orientation))
                if len(used) != len(actual_channels):
                    raise ValueError("exact GDS contains unowned channel fingers in centroid macro")
                if not observed or {item[0] for item in observed} != set(members):
                    raise ValueError("exact GDS centroid owner inventory is incomplete")

                row_map: Dict[int, list[Tuple[int, str]]] = {}
                for owner, x2, y2 in observed:
                    if orientation in ("R0", "R180"):
                        row_key, order_key = y2, x2
                    else:
                        row_key, order_key = x2, y2
                    row_map.setdefault(row_key, []).append((order_key, owner))
                reverse_rows = orientation in ("R90", "R180")
                reverse_fingers = orientation in ("R180", "R270")
                owner_rows = tuple(
                    tuple(
                        owner
                        for _order, owner in sorted(
                            row_map[row_key],
                            key=lambda item: item[0],
                            reverse=reverse_fingers,
                        )
                    )
                    for row_key in sorted(row_map, reverse=reverse_rows)
                )
                if len(owner_rows) != declaration.rows:
                    raise ValueError("exact GDS centroid row count differs from declaration")
            points_by_owner = {
                owner: Counter((x2, y2) for item_owner, x2, y2 in observed if item_owner == owner)
                for owner in members
            }
            x_values = [x2 for _owner, x2, _y2 in observed]
            y_values = [y2 for _owner, _x2, y2 in observed]
            center4 = (min(x_values) + max(x_values), min(y_values) + max(y_values))
            reflected = True
            for owner in members:
                points = points_by_owner[owner]
                reflection = Counter(
                    (center4[0] - x2, center4[1] - y2)
                    for x2, y2 in points.elements()
                )
                if reflection != points:
                    reflected = False
            left_points = points_by_owner[members[0]]
            right_points = points_by_owner[members[1]]
            left_count = sum(left_points.values())
            right_count = sum(right_points.values())
            centroid_equal = (
                left_count > 0
                and right_count > 0
                and sum(x * count for (x, _y), count in left_points.items()) * right_count
                == sum(x * count for (x, _y), count in right_points.items()) * left_count
                and sum(y * count for (_x, y), count in left_points.items()) * right_count
                == sum(y * count for (_x, y), count in right_points.items()) * left_count
            )
            point_ok = reflected and centroid_equal
            reason = (
                "reread exact GDS poly-active channel points are point-reflected "
                "and have equal member centroids"
                if point_ok
                else "reread exact GDS channel points violate declared point reflection"
            )
            centers = tuple(sorted(observed, key=lambda item: (item[2], item[1], item[0])))
            measurements[declaration.group] = (
                members,
                owner_rows,
                centers,
                point_ok,
                reason,
            )
            centroid_obligations.append(GdsCommonCentroidObligation(
                group=declaration.group,
                members=members,
                rows=declaration.rows,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="pass" if point_ok else "fail",
                measured_on_gds=True,
                point_reflection_verified=point_ok,
                owner_rows=owner_rows,
                channel_centers_twice_nm=centers,
                reflection_center_fourfold_nm=center4,
                member_channel_geometry_nm=tuple(measured_member_channels),
                reason=reason,
            ))
        except Exception as exc:  # noqa: BLE001 - one typed centroid obligation
            centroid_obligations.append(GdsCommonCentroidObligation(
                group=declaration.group,
                members=members,
                rows=declaration.rows,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=(
                    "exact common-centroid artwork cannot be measured: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ))

    interdigitation_obligations: list[GdsInterdigitationObligation] = []
    for declaration in interdigitations:
        if declaration.interdigitation_finger_pairs != 1:
            interdigitation_obligations.append(GdsInterdigitationObligation(
                members=declaration.members,
                declaration_ids=declaration.interdigitation_declaration_ids,
                pattern=declaration.interdigitation_pattern,
                finger_pairs=declaration.interdigitation_finger_pairs,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="unsupported",
                measured_on_gds=False,
                reason=(
                    "exact interdigitation checker supports the target-domain "
                    "finger_pairs=1 contract only"
                ),
            ))
            continue
        candidates = [
            value
            for value in measurements.values()
            if set(value[0]) == set(declaration.members)
        ]
        if len(candidates) != 1:
            interdigitation_obligations.append(GdsInterdigitationObligation(
                members=declaration.members,
                declaration_ids=declaration.interdigitation_declaration_ids,
                pattern=declaration.interdigitation_pattern,
                finger_pairs=declaration.interdigitation_finger_pairs,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                reason=(
                    "interdigitation lacks one independently measured exact-GDS "
                    "common-centroid owner"
                ),
            ))
            continue
        _members, owner_rows, centers, point_ok, _centroid_reason = candidates[0]
        paired_rows = all(
            len(row) % 2 == 0
            and all(row[index] == row[index + 1] for index in range(0, len(row), 2))
            for row in owner_rows
        )
        observed_pair_pattern = tuple(
            row[index]
            for row in owner_rows
            for index in range(0, len(row), 2)
        ) if paired_rows else ()
        normalized_pattern = tuple(
            "a" if owner == declaration.members[0] else "b"
            for owner in observed_pair_pattern
        )
        requested_pattern = tuple(declaration.interdigitation_pattern)
        pattern_matches = normalized_pattern == requested_pattern
        interleaved = point_ok and paired_rows and pattern_matches and all(
            set(row) == set(declaration.members)
            and any(left != right for left, right in zip(row, row[1:]))
            for row in owner_rows
        )
        interdigitation_obligations.append(GdsInterdigitationObligation(
            members=declaration.members,
            declaration_ids=declaration.interdigitation_declaration_ids,
            pattern=declaration.interdigitation_pattern,
            finger_pairs=declaration.interdigitation_finger_pairs,
            declaration_sha256=declaration.sha256,
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
            status="pass" if interleaved else "fail",
            measured_on_gds=True,
            owner_rows=owner_rows,
            observed_pair_pattern=observed_pair_pattern,
            channel_centers_twice_nm=centers,
            reason=(
                "reread exact GDS channel sequence realizes the declared "
                f"{declaration.interdigitation_pattern} pair pattern"
                if interleaved
                else "reread exact GDS channel sequence does not realize the "
                f"declared {declaration.interdigitation_pattern} pair pattern"
            ),
        ))
    return tuple(centroid_obligations), tuple(interdigitation_obligations)


def verify_well_island_sharing_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    share_intents: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    tech: Any,
) -> Tuple[GdsWellIslandShareObligation, ...]:
    """Independently prove declared member wells share one exact-GDS island."""

    from clay.layout.intent import (
        WellIslandShareIntent,
        layout_intent_from_cell_def,
        replay_well_island_share_intent,
    )

    if type(share_intents) is not tuple:
        raise TypeError("share_intents must be an exact tuple")
    if len(share_intents) > 256:
        raise ValueError("well island share inventory is too large")
    if any(type(item) is not WellIslandShareIntent for item in share_intents):
        raise TypeError(
            "share_intents must contain exact WellIslandShareIntent values"
        )
    declarations = tuple(
        replay_well_island_share_intent(item) for item in share_intents
    )
    if declarations != tuple(sorted(declarations, key=lambda item: item.group)):
        raise ValueError("well island shares must use canonical group order")
    if len({item.group for item in declarations}) != len(declarations):
        raise ValueError("duplicate well island share declarations are forbidden")

    def blocked_all(
        reason: str,
        *,
        status: Literal["unmeasurable", "unsupported"] = "unmeasurable",
        gds_sha256: str | None = None,
    ) -> Tuple[GdsWellIslandShareObligation, ...]:
        return tuple(
            GdsWellIslandShareObligation(
                group=item.group,
                body=item.body,
                well="",
                members=(),
                layer_key=None,
                declaration_sha256=item.sha256,
                gds_sha256=gds_sha256,
                status=status,
                measured_on_gds=False,
                reason=reason,
            )
            for item in declarations
        )

    if not declarations:
        try:
            from clay.layout.route.exact_artifact_evidence import (
                _replayed_source_cell_authority,
            )

            _source_sha, _inventory_sha, _classes, canonical_cell = (
                _replayed_source_cell_authority(source_cell)
            )
            declarations = tuple(
                replay_well_island_share_intent(item)
                for item in layout_intent_from_cell_def(
                    canonical_cell
                ).well_island_shares
            )
        except Exception as exc:
            try:
                declarations = tuple(
                    replay_well_island_share_intent(item)
                    for item in layout_intent_from_cell_def(
                        source_cell
                    ).well_island_shares
                )
            except Exception:
                return ()
            if declarations:
                return blocked_all(
                    "well island canonical source bytes cannot be replayed; "
                    "parsed declarations remain blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
        if declarations:
            return blocked_all(
                "well island share inventory was deleted from canonical source evidence"
            )
        return ()

    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        import klayout.db as kdb

        from clay.layout.route.exact_artifact_evidence import (
            _replayed_source_cell_authority,
        )
        from clay.layout.well_island_sharing import (
            compile_well_island_share_plans,
        )
        from clay.layout.placed_geometry import placed_point_nm

        if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
            raise TypeError("shared well checking requires emitter mappings")
        if tech is None:
            raise TypeError("shared well checking requires resolved layout tech")
        transformed_layers, gds_witness = _certified_assembly_transformed_layers(
            exact_artifact,
            gds_path,
            geometry,
            placement,
            return_witness=True,
            top_cell=top_cell,
        )
        (
            source_sha256,
            inventory_sha256,
            _member_classes,
            canonical_source_cell,
        ) = _replayed_source_cell_authority(source_cell)
        if (
            exact_artifact.scope.source_cell_source_sha256 != source_sha256
            or exact_artifact.scope.source_member_inventory_sha256
            != inventory_sha256
        ):
            raise ValueError("shared well artifact source scope does not replay")
        canonical_declarations = tuple(
            replay_well_island_share_intent(item)
            for item in layout_intent_from_cell_def(
                canonical_source_cell
            ).well_island_shares
        )
        if canonical_declarations != declarations:
            raise ValueError(
                "well island share inventory differs from canonical source bytes"
            )
        plans = compile_well_island_share_plans(
            canonical_source_cell, canonical_declarations, tech
        )
        if tuple(plan.declaration_sha256 for plan in plans) != tuple(
            item.sha256 for item in declarations
        ):
            raise ValueError("shared well plan inventory does not replay declarations")
        if any(plan.layer_key not in transformed_layers for plan in plans):
            raise ValueError(
                "shared well layer lacks an exact emitter transformation certificate"
            )

        source_names = tuple(sorted(
            str(getattr(device, "instance_name", "") or "")
            for device in tuple(getattr(canonical_source_cell, "devices", ()) or ())
        ))

        def member_region(name: str, layer_key: Tuple[int, int]) -> Any:
            stamp = geometry.get(name)
            position = placement.get(name)
            if stamp is None or position is None:
                raise ValueError(f"shared well member {name!r} lacks emitter evidence")
            rings = (getattr(stamp, "polys_by_layer", {}) or {}).get(layer_key)
            if not rings:
                raise ValueError(f"shared well member {name!r} lacks {layer_key} artwork")
            region = kdb.Region()
            for ring in tuple(rings):
                points = [
                    kdb.Point(*placed_point_nm(float(x), float(y), position))
                    for x, y in tuple(ring)
                ]
                if len(points) < 3:
                    raise ValueError(
                        f"shared well member {name!r} has degenerate well artwork"
                    )
                region.insert(kdb.Polygon(points))
            region.merge()
            if region.is_empty():
                raise ValueError(f"shared well member {name!r} has empty well artwork")
            return region

        obligations: list[GdsWellIslandShareObligation] = []
        for plan in plans:
            actual = gds_witness.merged_region(plan.layer_key)
            components = tuple(
                kdb.Region(kdb.Polygon(polygon))
                for polygon in actual.each_merged()
            )
            member_regions = {
                member: member_region(member, plan.layer_key)
                for member in plan.members
            }
            component_ids = []
            for member, expected in member_regions.items():
                if not (expected - actual).is_empty():
                    raise ValueError(
                        f"shared well member {member!r} artwork is absent from exact GDS"
                    )
                matches = tuple(
                    index
                    for index, component in enumerate(components)
                    if (expected - component).is_empty()
                )
                if len(matches) != 1:
                    raise ValueError(
                        f"shared well member {member!r} has ambiguous GDS component"
                    )
                component_ids.append(matches[0])
            connected = len(set(component_ids)) == 1
            selected_component = (
                components[component_ids[0]] if component_ids and connected else None
            )
            foreign_members = ()
            if selected_component is not None:
                foreign = []
                for name in source_names:
                    if not name or name in plan.members:
                        continue
                    stamp = geometry.get(name)
                    rings = (
                        getattr(stamp, "polys_by_layer", {}) or {}
                    ).get(plan.layer_key) if stamp is not None else None
                    if not rings or placement.get(name) is None:
                        continue
                    expected = member_region(name, plan.layer_key)
                    if not (expected & selected_component).is_empty():
                        foreign.append(name)
                foreign_members = tuple(sorted(foreign))
            passed = connected and not foreign_members
            bbox_nm = None
            if selected_component is not None:
                bbox = selected_component.bbox()
                bbox_nm = (
                    int(bbox.left), int(bbox.bottom),
                    int(bbox.right), int(bbox.top),
                )
            obligations.append(GdsWellIslandShareObligation(
                group=plan.group,
                body=plan.body,
                well=plan.well,
                members=plan.members,
                layer_key=plan.layer_key,
                declaration_sha256=plan.declaration_sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="pass" if passed else "fail",
                measured_on_gds=True,
                island_bbox_nm=bbox_nm,
                foreign_members=foreign_members,
                reason=(
                    "reread exact GDS places every declared member in one "
                    "exclusive well connected component"
                    if passed
                    else "reread exact GDS does not realize one exclusive "
                    "declared-member well component"
                ),
            ))
        return tuple(obligations)
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        return blocked_all(
            "cannot replay exact shared-well evidence: "
            f"{type(exc).__name__}: {exc}",
            gds_sha256=(
                getattr(getattr(exact_artifact, "scope", None), "exact_gds_sha256", None)
            ),
        )


def verify_passive_bank_partitions_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    partition_intents: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    tech: Any,
    rigid_nonmos_macro_plans: Any,
    rigid_nonmos_macro_inputs: Any,
    rigid_nonmos_macro_interfaces: Any,
    rigid_nonmos_placement_certificate: Any,
    rigid_nonmos_scope_token: Any,
    rigid_nonmos_emission_certificate: Any,
    pins_by_net: Mapping[str, Any] | None,
    eps: int = ARTWORK_EPS,
) -> Tuple[GdsPassiveBankPartitionObligation, ...]:
    """Replay passive-bank partitions and uniquely consume their exact GDS art.

    The caller's placement graph is not authority.  This checker reparses the
    canonical source, recompiles its PDK-neutral macro graph, replays the sealed
    rigid non-MOS emission certificate against the current emitter mappings and
    pins, then independently rereads and uniquely consumes every declared member
    polygon from the exact GDS.
    """

    from clay.layout.intent import (
        PassiveBankPartitionIntent,
        replay_passive_bank_partition_intent,
    )

    if type(partition_intents) is not tuple:
        raise TypeError("partition_intents must be an exact tuple")
    if len(partition_intents) > 256:
        raise ValueError("passive bank partition inventory is too large")
    if type(eps) is not int or not 0 <= eps <= ARTWORK_EPS:
        raise ValueError("passive bank partition eps must be a bounded exact integer")
    if any(type(item) is not PassiveBankPartitionIntent for item in partition_intents):
        raise TypeError(
            "partition_intents must contain exact PassiveBankPartitionIntent values"
        )
    declarations = tuple(
        replay_passive_bank_partition_intent(item) for item in partition_intents
    )
    declaration_keys = tuple((item.parent, item.bank) for item in declarations)
    if declarations != tuple(sorted(declarations, key=lambda item: (item.parent, item.bank))):
        raise ValueError("passive bank partitions must be in canonical order")
    if len(set(declaration_keys)) != len(declaration_keys):
        raise ValueError("duplicate passive bank partition declarations are forbidden")

    def blocked_all(
        reason: str,
        *,
        status: Literal["unmeasurable", "unsupported"] = "unmeasurable",
        gds_sha256: str | None = None,
    ) -> Tuple[GdsPassiveBankPartitionObligation, ...]:
        return tuple(
            GdsPassiveBankPartitionObligation(
                parent=item.parent,
                bank=item.bank,
                members=item.members,
                macro_id=f"passive_bank:{item.parent}:{item.bank}",
                declaration_sha256=item.sha256,
                gds_sha256=gds_sha256,
                status=status,
                measured_on_gds=False,
                reason=reason,
            )
            for item in declarations
        )

    if not declarations:
        try:
            from clay.layout.intent import layout_intent_from_cell_def
            from clay.layout.route.exact_artifact_evidence import (
                _replayed_source_cell_authority,
            )

            _source_sha256, _inventory_sha256, _classes, canonical_source_cell = (
                _replayed_source_cell_authority(source_cell)
            )
            declarations = tuple(
                replay_passive_bank_partition_intent(item)
                for item in layout_intent_from_cell_def(
                    canonical_source_cell
                ).passive_bank_partitions
            )
        except Exception as exc:
            try:
                from clay.layout.intent import layout_intent_from_cell_def

                declarations = tuple(
                    replay_passive_bank_partition_intent(item)
                    for item in layout_intent_from_cell_def(
                        source_cell
                    ).passive_bank_partitions
                )
            except Exception:
                return ()
            if declarations:
                return blocked_all(
                    "passive bank canonical source bytes cannot be replayed; "
                    "parsed source declarations remain blocking: "
                    f"{type(exc).__name__}: {exc}"
                )
        if declarations:
            return blocked_all(
                "passive bank partition inventory was deleted from canonical "
                "source evidence"
            )
        return ()

    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.nonrow_macro_plan import (
            RigidNonmosEmissionCertificate,
            RigidNonmosMacroInterfaceSet,
            RigidNonmosMacroPlanSet,
            RigidNonmosMacroCompileInputs,
            RigidNonmosPlacementCertificate,
            verify_rigid_nonmos_emission_certificate,
        )
        from clay.layout.physical_macro_graph import physical_macro_graph_from_intent
        from clay.layout.route.exact_artifact_evidence import (
            ExactEmittedArtifact,
            _replayed_source_cell_authority,
        )

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("passive bank checking requires ExactEmittedArtifact")
        if type(rigid_nonmos_macro_plans) is not RigidNonmosMacroPlanSet:
            raise TypeError("passive bank checking requires exact rigid plans")
        if type(rigid_nonmos_macro_inputs) is not RigidNonmosMacroCompileInputs:
            raise TypeError("passive bank checking requires exact rigid inputs")
        if type(rigid_nonmos_macro_interfaces) is not RigidNonmosMacroInterfaceSet:
            raise TypeError("passive bank checking requires exact rigid interfaces")
        if type(rigid_nonmos_placement_certificate) is not RigidNonmosPlacementCertificate:
            raise TypeError("passive bank checking requires exact rigid placement")
        if type(rigid_nonmos_emission_certificate) is not RigidNonmosEmissionCertificate:
            raise TypeError("passive bank checking requires exact rigid emission")
        if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
            raise TypeError("passive bank checking requires emitter mappings")
        if not isinstance(pins_by_net, Mapping):
            raise TypeError("passive bank checking requires current route pins")
        if tech is None:
            raise TypeError("passive bank checking requires resolved layout tech")

        exact_artifact.validate()
        if exact_artifact.scope.top_cell != str(top_cell):
            raise ValueError("passive bank artifact is bound to another top cell")
        (
            source_sha256,
            inventory_sha256,
            _member_classes,
            canonical_source_cell,
        ) = _replayed_source_cell_authority(source_cell)
        if (
            source_sha256 != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha256
            != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("passive bank source inventory differs from emitter scope")
        canonical_intent = layout_intent_from_cell_def(canonical_source_cell)
        canonical_declarations = tuple(
            replay_passive_bank_partition_intent(item)
            for item in canonical_intent.passive_bank_partitions
        )
        if canonical_declarations != declarations:
            raise ValueError(
                "passive bank declarations differ from canonical source bytes"
            )
        graph = physical_macro_graph_from_intent(
            canonical_source_cell,
            canonical_intent,
            device_records={},
        )
        if graph.sha256 != rigid_nonmos_macro_plans.graph_sha256:
            raise ValueError("rigid passive plans bind another physical macro graph")
        verify_rigid_nonmos_emission_certificate(
            rigid_nonmos_emission_certificate,
            rigid_nonmos_macro_plans,
            rigid_nonmos_macro_interfaces,
            rigid_nonmos_placement_certificate,
            scope_token=rigid_nonmos_scope_token,
            compile_inputs=rigid_nonmos_macro_inputs,
            geometry=geometry,
            placement_um=placement,
            pins_by_net=pins_by_net,
        )
        rigid_nonmos_emission_certificate.bind_exact_artifact(exact_artifact)

        plans_by_id = {
            plan.macro_id: plan for plan in rigid_nonmos_macro_plans.plans
        }
        graph_macros_by_id = {macro.macro_id: macro for macro in graph.macros}
        certified_macro_ids = {
            macro_id
            for macro_id, _fingerprint in (
                rigid_nonmos_emission_certificate.macro_structure_fingerprints
            )
        }
        certified_members = {
            member
            for member, _artwork_sha256 in (
                rigid_nonmos_emission_certificate.owned_member_artwork
            )
        }
        for declaration in declarations:
            macro_id = f"passive_bank:{declaration.parent}:{declaration.bank}"
            macro = graph_macros_by_id.get(macro_id)
            plan = plans_by_id.get(macro_id)
            if (
                macro is None
                or macro.kind != "passive_bank"
                or macro.members != declaration.members
                or plan is None
                or plan.members != declaration.members
                or macro_id not in certified_macro_ids
                or not set(declaration.members) <= certified_members
            ):
                raise ValueError(
                    f"passive bank {macro_id!r} lacks canonical compiled ownership"
                )

        transformed_layers = _certified_assembly_transformed_layers(
            exact_artifact, gds_path, geometry, placement
        )
        shapes_by_layer = _read_top_cell_shapes(gds_path)
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        return blocked_all(
            "exact passive-bank authority cannot be replayed: "
            f"{type(exc).__name__}: {exc}",
            gds_sha256=(
                getattr(getattr(exact_artifact, "scope", None), "exact_gds_sha256", None)
            ),
        )

    available_shapes_by_layer = {
        key: list(shapes) for key, shapes in shapes_by_layer.items()
    }

    def consume_member(member: str) -> Tuple[int, int, int, int]:
        stamp = geometry.get(member)
        position = placement.get(member)
        if stamp is None or position is None:
            raise ValueError(f"passive bank member {member!r} lacks emitter artwork")
        expected = _emitted_shape_witness(
            stamp, position, excluded_layers=transformed_layers
        )
        window = _union_bbox(
            [shape.bbox for shapes in expected.values() for shape in shapes]
        )
        if not expected or window is None:
            raise ValueError(f"passive bank member {member!r} has empty artwork")
        candidates = _candidate_gds_shapes(
            window,
            available_shapes_by_layer,
            tuple(expected),
            eps,
        )
        claimed, _transform = _match_owned_artwork(expected, candidates, eps)
        if claimed is None:
            raise ValueError(
                f"passive bank member {member!r} lacks unique exact GDS artwork"
            )
        for key, shapes in claimed.items():
            available = available_shapes_by_layer.get(key, [])
            for shape in shapes:
                index = next(
                    (
                        ordinal
                        for ordinal, candidate in enumerate(available)
                        if candidate is shape
                    ),
                    None,
                )
                if index is None:
                    raise ValueError(
                        f"passive bank member {member!r} reuses exact GDS artwork"
                    )
                available.pop(index)
        claimed_bbox = _union_bbox(
            [shape.bbox for shapes in claimed.values() for shape in shapes]
        )
        if claimed_bbox is None:
            raise ValueError(f"passive bank member {member!r} has no claimed bbox")
        return claimed_bbox

    obligations: list[GdsPassiveBankPartitionObligation] = []
    for declaration in declarations:
        macro_id = f"passive_bank:{declaration.parent}:{declaration.bank}"
        try:
            member_bboxes = tuple(consume_member(member) for member in declaration.members)
            claimed_bbox = _union_bbox(member_bboxes)
            if claimed_bbox is None:
                raise ValueError("passive bank partition has empty claimed artwork")
            obligations.append(GdsPassiveBankPartitionObligation(
                parent=declaration.parent,
                bank=declaration.bank,
                members=declaration.members,
                macro_id=macro_id,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="pass",
                measured_on_gds=True,
                claimed_bbox_nm=claimed_bbox,
                graph_sha256=graph.sha256,
                emission_certificate_sha256=(
                    rigid_nonmos_emission_certificate.sha256
                ),
                reason=(
                    "canonical source partition, rigid macro authority, and unique "
                    "member artwork replay on the exact emitted GDS"
                ),
            ))
        except Exception as exc:  # noqa: BLE001 - one typed partition obligation
            obligations.append(GdsPassiveBankPartitionObligation(
                parent=declaration.parent,
                bank=declaration.bank,
                members=declaration.members,
                macro_id=macro_id,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_artifact.scope.exact_gds_sha256,
                status="unmeasurable",
                measured_on_gds=False,
                graph_sha256=graph.sha256,
                emission_certificate_sha256=(
                    rigid_nonmos_emission_certificate.sha256
                ),
                reason=(
                    "passive bank partition cannot be measured: "
                    f"{type(exc).__name__}: {exc}"
                ),
            ))
    return tuple(obligations)


def verify_density_whitespace_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    density_intents: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    final_placement_certificate: Any,
    placement_state_sha256: str | None,
    tech: Any,
    eps: int = ARTWORK_EPS,
) -> Tuple[GdsDensityWhitespaceObligation, ...]:
    """Measure structural whitespace from uniquely owned exact-GDS polygons."""

    from clay.layout.intent import DensityIntent, replay_density_intent

    if type(density_intents) is not tuple:
        raise TypeError("density_intents must be an exact tuple")
    if type(eps) is not int or not 0 <= eps <= ARTWORK_EPS:
        raise ValueError(
            f"density eps must be an exact integer bounded in [0,{ARTWORK_EPS}]"
        )
    if len(density_intents) > 1:
        raise ValueError("exact density checker supports one declaration")
    if any(type(item) is not DensityIntent for item in density_intents):
        raise TypeError("density_intents must contain exact DensityIntent values")
    declarations = tuple(replay_density_intent(item) for item in density_intents)
    inventory_error: str | None = None
    if not declarations and source_cell is not None:
        try:
            from clay.layout.intent import layout_intent_from_cell_def
            from clay.layout.route.exact_artifact_evidence import (
                _replayed_source_cell_authority,
            )

            _source_sha, _inventory_sha, _classes, canonical_cell = (
                _replayed_source_cell_authority(source_cell)
            )
            declarations = tuple(
                replay_density_intent(item)
                for item in layout_intent_from_cell_def(canonical_cell).densities
            )
            if declarations:
                inventory_error = (
                    "density inventory differs from canonical source bytes"
                )
        except Exception as exc:  # noqa: BLE001 - preserve parsed denominator
            fallback = tuple(
                replay_density_intent(item)
                for item in getattr(
                    layout_intent_from_cell_def(source_cell), "densities", ()
                )
            )
            if fallback:
                declarations = fallback
                inventory_error = (
                    "density canonical source cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
    if not declarations:
        return ()
    declaration = declarations[0]

    def blocked(reason: str) -> Tuple[GdsDensityWhitespaceObligation, ...]:
        return (
            GdsDensityWhitespaceObligation(
                max_whitespace_fraction=declaration.max_whitespace_fraction,
                scope=declaration.scope,
                declaration_sha256=declaration.sha256,
                gds_sha256=(
                    getattr(getattr(exact_artifact, "scope", None),
                            "exact_gds_sha256", None)
                ),
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            ),
        )

    if inventory_error is not None:
        return blocked(inventory_error)

    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        (
            canonical_intent,
            member_classes,
            transformed_layers,
            witness,
        ) = _replay_exact_final_placement_artwork(
            gds_path=gds_path,
            top_cell=top_cell,
            exact_artifact=exact_artifact,
            source_cell=source_cell,
            final_placement_certificate=final_placement_certificate,
            placement_state_sha256=placement_state_sha256,
            emitted_device_geometry=geometry,
            emitted_device_placement_um=placement,
            tech=tech,
        )
        canonical_densities = tuple(
            replay_density_intent(item) for item in canonical_intent.densities
        )
        if canonical_densities != declarations:
            raise ValueError("density declarations differ from canonical source bytes")
        expected_occupancy = 1.0 - declaration.max_whitespace_fraction
        if (
            final_placement_certificate.declared_minimum_structural_occupancy
            != expected_occupancy
        ):
            raise ValueError("density target differs from final placement")

        canonical_members = tuple(name for name, _kind in member_classes)
        if not canonical_members:
            raise ValueError("density canonical source inventory is empty")
        owner_for_member = {
            member: _unique_emitter_owner(member, geometry)
            for member in canonical_members
        }
        expected_by_owner: Dict[str, Dict[Tuple[int, int], list]] = {}
        for owner in sorted(set(owner_for_member.values())):
            stamp = geometry.get(owner)
            position = placement.get(owner)
            if stamp is None or position is None:
                raise ValueError(f"density owner {owner!r} has no emitter artwork")
            expected = _merged_shape_witness(
                _emitted_shape_witness(
                    stamp,
                    position,
                    excluded_layers=transformed_layers,
                )
            )
            if not expected:
                raise ValueError(f"density owner {owner!r} has empty artwork")
            expected_by_owner[owner] = expected
        route_owned_bboxes: Dict[
            str, Dict[Tuple[int, int], list[Tuple[int, int, int, int]]]
        ] = {}
        for net, key, bbox in exact_artifact.route_manifest.entries:
            route_owned_bboxes.setdefault(net, {}).setdefault(key, []).append(bbox)
        claims = witness.claim_polygon_multiset(
            expected_by_owner,
            eps=eps,
            excluded_bboxes_by_owner={
                f"route:{net}": {
                    key: tuple(sorted(bboxes))
                    for key, bboxes in layers.items()
                }
                for net, layers in route_owned_bboxes.items()
            },
        )
        occupied_region = witness.projected_union_region(claims)
        if occupied_region.is_empty():
            raise ValueError("density exact-GDS structural occupancy is empty")
        bbox = occupied_region.bbox()
        envelope_bbox = (
            int(bbox.left), int(bbox.bottom), int(bbox.right), int(bbox.top)
        )
        envelope_area = int(bbox.width()) * int(bbox.height())
        occupied_area = int(occupied_region.area())
        if envelope_area <= 0 or occupied_area <= 0 or occupied_area > envelope_area:
            raise ValueError("density exact-GDS areas are inconsistent")
        observed = 1.0 - (float(occupied_area) / float(envelope_area))
    except Exception as exc:  # noqa: BLE001 - one typed density obligation
        return blocked(
            "exact density artwork/placement cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )

    passed = observed <= declaration.max_whitespace_fraction
    return (
        GdsDensityWhitespaceObligation(
            max_whitespace_fraction=declaration.max_whitespace_fraction,
            scope=declaration.scope,
            declaration_sha256=declaration.sha256,
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
            status="pass" if passed else "fail",
            measured_on_gds=True,
            member_count=len(canonical_members),
            envelope_bbox_nm=envelope_bbox,
            occupied_area_nm2=occupied_area,
            envelope_area_nm2=envelope_area,
            observed_whitespace_fraction=observed,
            reason=(
                f"exact-GDS whitespace {observed:.12g} is within declared maximum "
                f"{declaration.max_whitespace_fraction:.12g}"
                if passed
                else f"exact-GDS whitespace {observed:.12g} exceeds declared maximum "
                f"{declaration.max_whitespace_fraction:.12g}"
            ),
        ),
    )


def _canonical_rail_affinity_member_names(canonical_cell: Any) -> Tuple[str, ...]:
    from clay.layout.route.exact_artifact_evidence import (
        _canonical_source_member_inventory,
    )

    return tuple(
        str(row[0]) for row in _canonical_source_member_inventory(canonical_cell)
    )


def _canonical_rail_affinity_groups(
    *,
    canonical_intent: Any,
    canonical_cell: Any,
    declarations: Sequence[Any],
) -> Dict[str, Tuple[str, ...]]:
    """Resolve rail targets against the complete replayed source inventory."""

    groups: Dict[str, Tuple[str, ...]] = {}
    for group in canonical_intent.place_groups:
        name = str(group.group)
        members = tuple(str(member) for member in group.devices)
        if not name or not members or name in groups:
            raise ValueError("rail affinity place groups are not unique/nonempty")
        groups[name] = members
    canonical_members = _canonical_rail_affinity_member_names(canonical_cell)
    canonical_member_set = set(canonical_members)
    if not canonical_members:
        raise ValueError("rail affinity canonical member inventory is empty")
    for declaration in declarations:
        target = str(declaration.group)
        if target in groups:
            continue
        if target not in canonical_member_set:
            raise ValueError(
                "rail affinity declaration names neither a place group nor "
                "a canonical source member"
            )
        groups[target] = (target,)
    if any(set(groups[item.group]) - canonical_member_set for item in declarations):
        raise ValueError("rail affinity group contains a foreign source member")
    return groups


def verify_rail_affinities_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    rail_affinity_intents: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    final_placement_certificate: Any,
    row_final_placement: Any,
    placement_state_sha256: str | None,
    tech: Any,
    eps: int = ARTWORK_EPS,
) -> Tuple[GdsRailAffinityObligation, ...]:
    """Measure canonical place groups against the exact structural midline."""

    from clay.layout.intent import replay_rail_affinity_inventory

    if type(eps) is not int or not 0 <= eps <= ARTWORK_EPS:
        raise ValueError(
            f"rail affinity eps must be an exact integer bounded in [0,{ARTWORK_EPS}]"
        )
    supplied = replay_rail_affinity_inventory(rail_affinity_intents)

    declarations = supplied
    inventory_error: str | None = None
    canonical_cell = None
    if source_cell is not None:
        try:
            from clay.layout.intent import layout_intent_from_cell_def
            from clay.layout.route.exact_artifact_evidence import (
                _replayed_source_cell_authority,
            )

            _source_sha, _inventory_sha, _classes, canonical_cell = (
                _replayed_source_cell_authority(source_cell)
            )
            declarations = replay_rail_affinity_inventory(
                layout_intent_from_cell_def(canonical_cell).rail_affinities
            )
            if declarations != supplied:
                inventory_error = (
                    "rail affinity inventory differs from canonical source bytes"
                )
        except Exception as exc:  # noqa: BLE001 - preserve declared denominator
            if supplied:
                inventory_error = (
                    "rail affinity canonical source cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
    if not declarations:
        return ()

    def blocked(reason: str) -> Tuple[GdsRailAffinityObligation, ...]:
        return tuple(
            GdsRailAffinityObligation(
                group=item.group,
                side=item.side,
                declaration_sha256=item.sha256,
                gds_sha256=getattr(
                    getattr(exact_artifact, "scope", None),
                    "exact_gds_sha256",
                    None,
                ),
                status="unmeasurable",
                measured_on_gds=False,
                reason=reason,
            )
            for item in declarations
        )

    if inventory_error is not None:
        return blocked(inventory_error)

    geometry = emitted_device_geometry or {}
    placement = emitted_device_placement_um or {}
    try:
        canonical_intent, member_classes, transformed_layers, witness = (
            _replay_exact_final_placement_artwork(
                gds_path=gds_path,
                top_cell=top_cell,
                exact_artifact=exact_artifact,
                source_cell=source_cell,
                final_placement_certificate=final_placement_certificate,
                placement_state_sha256=placement_state_sha256,
                emitted_device_geometry=geometry,
                emitted_device_placement_um=placement,
                tech=tech,
            )
        )
        canonical_declarations = replay_rail_affinity_inventory(
            canonical_intent.rail_affinities
        )
        if canonical_declarations != declarations:
            raise ValueError(
                "rail affinity declarations differ from canonical source bytes"
            )
        if canonical_cell is None:
            raise ValueError("rail affinity canonical source cell is absent")
        groups = _canonical_rail_affinity_groups(
            canonical_intent=canonical_intent,
            canonical_cell=canonical_cell,
            declarations=declarations,
        )
        canonical_members = _canonical_rail_affinity_member_names(canonical_cell)
        if row_final_placement is None:
            raise ValueError("rail affinity canonical final placement is absent")
        from clay.layout.route.route_cell import _bridge_placement_to_route_inputs

        _bridged, _rebuilt_placement, canonical_geometry = (
            _bridge_placement_to_route_inputs(
                row_final_placement,
                canonical_cell,
                str(getattr(tech, "pdk_id", "") or tech),
                intent=canonical_intent,
            )
        )
        owner_for_member = {
            member: _unique_emitter_owner(member, canonical_geometry)
            for member in canonical_members
        }
        members_for_owner: Dict[str, set[str]] = {}
        for member, owner in owner_for_member.items():
            members_for_owner.setdefault(owner, set()).add(member)
        owner_groups: Dict[str, set[str]] = {}
        for declaration in declarations:
            declared_members = set(groups[declaration.group])
            for member in groups[declaration.group]:
                owner = owner_for_member[member]
                owner_groups.setdefault(owner, set()).add(declaration.group)
                ungrouped = members_for_owner[owner] - declared_members
                if ungrouped:
                    raise ValueError(
                        f"rail affinity owner {owner!r} contains ungrouped "
                        f"canonical members {tuple(sorted(ungrouped))}"
                    )
        if any(len(names) != 1 for names in owner_groups.values()):
            raise ValueError("one emitter owner spans multiple rail-affinity groups")

        expected_by_owner: Dict[str, Dict[Tuple[int, int], list]] = {}
        for owner in sorted(set(owner_for_member.values())):
            stamp = canonical_geometry.get(owner)
            position = placement.get(owner)
            if stamp is None or position is None:
                raise ValueError(
                    f"rail affinity owner {owner!r} has no emitter artwork"
                )
            expected = _exact_emitter_artwork_witness(
                stamp,
                position,
                excluded_layers=transformed_layers,
            )
            if not expected:
                raise ValueError(
                    f"rail affinity owner {owner!r} has empty artwork"
                )
            expected_by_owner[owner] = expected
        route_owned_bboxes: Dict[
            str, Dict[Tuple[int, int], list[Tuple[int, int, int, int]]]
        ] = {}
        for net, key, bbox in exact_artifact.route_manifest.entries:
            route_owned_bboxes.setdefault(net, {}).setdefault(key, []).append(bbox)
        claims = witness.claim_polygon_multiset(
            expected_by_owner,
            eps=eps,
            excluded_bboxes_by_owner={
                f"route:{net}": {
                    key: tuple(sorted(bboxes))
                    for key, bboxes in layers.items()
                }
                for net, layers in route_owned_bboxes.items()
            },
        )
        universe_region = witness.projected_union_region(claims)
        if universe_region.is_empty():
            raise ValueError("rail affinity exact-GDS universe is empty")
        universe_box = universe_region.bbox()
        universe_bbox = (
            int(universe_box.left),
            int(universe_box.bottom),
            int(universe_box.right),
            int(universe_box.top),
        )
        group_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
        for declaration in declarations:
            owners = tuple(sorted({
                owner_for_member[member]
                for member in groups[declaration.group]
            }))
            region = witness.projected_union_region(
                {owner: claims[owner] for owner in owners}
            )
            if region.is_empty():
                raise ValueError(
                    f"rail affinity group {declaration.group!r} is empty"
                )
            box = region.bbox()
            group_bboxes[declaration.group] = (
                int(box.left), int(box.bottom), int(box.right), int(box.top)
            )
    except Exception as exc:  # noqa: BLE001 - one typed exact-artifact boundary
        return blocked(
            "exact rail affinity artwork/placement cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )

    universe_mid_twice = universe_bbox[1] + universe_bbox[3]
    results: list[GdsRailAffinityObligation] = []
    for declaration in declarations:
        bbox = group_bboxes[declaration.group]
        group_mid_twice = bbox[1] + bbox[3]
        passed = (
            group_mid_twice >= universe_mid_twice
            if declaration.side == "top"
            else group_mid_twice <= universe_mid_twice
        )
        results.append(GdsRailAffinityObligation(
            group=declaration.group,
            side=declaration.side,
            declaration_sha256=declaration.sha256,
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
            status="pass" if passed else "fail",
            measured_on_gds=True,
            members=groups[declaration.group],
            group_bbox_nm=bbox,
            universe_bbox_nm=universe_bbox,
            reason=(
                "exact-GDS group center is on the declared supply side"
                if passed
                else "exact-GDS group center is on the wrong supply side"
            ),
        ))
    return tuple(results)


def verify_functional_od_fill_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    od_fill_intents: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    final_placement_certificate: Any,
    placement_state_sha256: str | None,
    row_final_placement: Any,
    tech: Any,
    eps: int = ARTWORK_EPS,
) -> Tuple[GdsFunctionalOdFillObligation, ...]:
    """Measure source-owned functional active inside the final core envelope."""

    from clay.layout.intent import OdFillIntent, replay_od_fill_intent

    if type(od_fill_intents) is not tuple:
        raise TypeError("od_fill_intents must be an exact tuple")
    if type(eps) is not int or not 0 <= eps <= ARTWORK_EPS:
        raise ValueError(
            f"functional OD eps must be an exact integer bounded in [0,{ARTWORK_EPS}]"
        )
    if len(od_fill_intents) > 1:
        raise ValueError("exact functional OD checker supports one declaration")
    if any(type(item) is not OdFillIntent for item in od_fill_intents):
        raise TypeError("od_fill_intents must contain exact OdFillIntent values")
    declarations = tuple(replay_od_fill_intent(item) for item in od_fill_intents)
    inventory_error: str | None = None
    if not declarations and source_cell is not None:
        try:
            from clay.layout.intent import layout_intent_from_cell_def
            from clay.layout.route.exact_artifact_evidence import (
                _replayed_source_cell_authority,
            )

            _source_sha, _inventory_sha, _classes, canonical_cell = (
                _replayed_source_cell_authority(source_cell)
            )
            declarations = tuple(
                replay_od_fill_intent(item)
                for item in layout_intent_from_cell_def(canonical_cell).od_fills
            )
            if declarations:
                inventory_error = "functional OD inventory differs from canonical source bytes"
        except Exception as exc:  # noqa: BLE001 - retain parsed denominator
            fallback = tuple(
                replay_od_fill_intent(item)
                for item in getattr(
                    layout_intent_from_cell_def(source_cell), "od_fills", ()
                )
            )
            if fallback:
                declarations = fallback
                inventory_error = (
                    "functional OD canonical source cannot be replayed: "
                    f"{type(exc).__name__}: {exc}"
                )
    if not declarations:
        return ()
    declaration = declarations[0]

    def blocked(
        status: Literal["unmeasurable", "unsupported"], reason: str
    ) -> Tuple[GdsFunctionalOdFillObligation, ...]:
        return (
            GdsFunctionalOdFillObligation(
                min_fraction=declaration.min_fraction,
                scope=declaration.scope,
                declaration_sha256=declaration.sha256,
                gds_sha256=getattr(
                    getattr(exact_artifact, "scope", None),
                    "exact_gds_sha256",
                    None,
                ),
                status=status,
                measured_on_gds=False,
                reason=reason,
            ),
        )

    if inventory_error is not None:
        return blocked("unmeasurable", inventory_error)

    try:
        from clay.layout.route.exact_artifact_evidence import (
            _replayed_source_cell_authority,
        )

        _source_sha, _inventory_sha, _classes, canonical_cell = (
            _replayed_source_cell_authority(source_cell)
        )
    except Exception as exc:  # noqa: BLE001 - canonical source boundary
        return blocked(
            "unmeasurable",
            "functional OD canonical source cannot be replayed: "
            f"{type(exc).__name__}: {exc}",
        )

    try:
        from clay.layout.tech import LayoutTech

        source_pdk = str(getattr(canonical_cell, "pdk_id", "") or "")
        supplied_pdk = str(getattr(tech, "pdk_id", "") or "")
        if source_pdk and supplied_pdk != source_pdk:
            raise ValueError("functional OD tech differs from canonical source PDK")
        canonical_tech = LayoutTech.resolve(source_pdk or supplied_pdk)
        from clay.layout.completeness.gds_witness import functional_active_layer_keys

        active_keys = functional_active_layer_keys(canonical_tech)
    except Exception as exc:  # noqa: BLE001 - installed PDK capability boundary
        return blocked(
            "unsupported",
            "installed PDK functional active layers are unavailable: "
            f"{type(exc).__name__}: {exc}",
        )

    try:
        import klayout.db as kdb

        canonical_intent, member_classes, transformed_layers, witness = (
            _replay_exact_final_placement_artwork(
                gds_path=gds_path,
                top_cell=top_cell,
                exact_artifact=exact_artifact,
                source_cell=source_cell,
                final_placement_certificate=final_placement_certificate,
                placement_state_sha256=placement_state_sha256,
                emitted_device_geometry=emitted_device_geometry,
                emitted_device_placement_um=emitted_device_placement_um,
                tech=canonical_tech,
            )
        )
        canonical_declarations = tuple(
            replay_od_fill_intent(item) for item in canonical_intent.od_fills
        )
        if canonical_declarations != declarations:
            raise ValueError(
                "functional OD declarations differ from canonical source bytes"
            )
        if (
            final_placement_certificate.declared_minimum_functional_od_occupancy
            != declaration.min_fraction
        ):
            raise ValueError("functional OD target differs from final placement")
        if final_placement_certificate.functional_od_layer_keys != active_keys:
            raise ValueError("functional OD active layers differ from final placement")

        mos_members = tuple(
            name for name, kind in member_classes if kind == "mos"
        )
        if not mos_members:
            raise ValueError("functional OD canonical MOS inventory is empty")
        if row_final_placement is None:
            raise ValueError("functional OD canonical final placement is absent")
        from clay.layout.route.route_cell import _bridge_placement_to_route_inputs

        _bridged, _rebuilt_placement, canonical_geometry = (
            _bridge_placement_to_route_inputs(
                row_final_placement,
                canonical_cell,
                canonical_tech.pdk_id,
                intent=canonical_intent,
            )
        )
        geometry = emitted_device_geometry or {}
        placement = emitted_device_placement_um or {}
        owners = {
            _unique_emitter_owner(member, canonical_geometry)
            for member in mos_members
        }
        expected_by_owner: Dict[str, Dict[Tuple[int, int], list]] = {}
        for owner in sorted(owners):
            stamp = canonical_geometry.get(owner)
            position = placement.get(owner)
            if stamp is None or position is None:
                raise ValueError(
                    f"functional OD owner {owner!r} has no emitter artwork"
                )
            all_layers = _merged_shape_witness(
                _emitted_shape_witness(
                    stamp,
                    position,
                    excluded_layers=transformed_layers,
                )
            )
            active = {
                key: all_layers[key] for key in active_keys if all_layers.get(key)
            }
            if not active:
                raise ValueError(
                    f"functional OD owner {owner!r} has no active artwork"
                )
            expected_by_owner[owner] = active
        route_owned_bboxes: Dict[
            str, Dict[Tuple[int, int], list[Tuple[int, int, int, int]]]
        ] = {}
        for net, key, route_bbox in exact_artifact.route_manifest.entries:
            route_owned_bboxes.setdefault(net, {}).setdefault(key, []).append(
                route_bbox
            )
        claims = witness.claim_polygon_multiset(
            expected_by_owner,
            eps=eps,
            excluded_bboxes_by_owner={
                f"route:{net}": {
                    key: tuple(sorted(bboxes))
                    for key, bboxes in layers.items()
                }
                for net, layers in route_owned_bboxes.items()
            },
        )
        active_region = witness.projected_union_region(claims)
        if active_region.is_empty():
            raise ValueError("functional OD exact-GDS artwork is empty")

        bbox = tuple(final_placement_certificate.bbox_nm)
        if (
            len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or bbox[0] >= bbox[2]
            or bbox[1] >= bbox[3]
        ):
            raise ValueError("functional OD final core envelope is invalid")
        envelope = kdb.Region(kdb.Box(*bbox))
        if not (active_region - envelope).is_empty():
            raise ValueError("functional OD artwork escapes the final core envelope")
        active_area = int(active_region.area())
        envelope_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if active_area <= 0 or active_area > envelope_area:
            raise ValueError("functional OD exact-GDS areas are inconsistent")
        if final_placement_certificate.functional_od_area_nm2 != active_area:
            raise ValueError("functional OD area differs from final placement")
        observed = float(Decimal(active_area) / Decimal(envelope_area))
    except Exception as exc:  # noqa: BLE001 - one typed exact-GDS obligation
        return blocked(
            "unmeasurable",
            "exact functional OD artwork/placement cannot be replayed: "
            f"{type(exc).__name__}: {exc}",
        )

    passed = Decimal(active_area) >= (
        Decimal(str(declaration.min_fraction)) * Decimal(envelope_area)
    )
    return (
        GdsFunctionalOdFillObligation(
            min_fraction=declaration.min_fraction,
            scope=declaration.scope,
            declaration_sha256=declaration.sha256,
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
            status="pass" if passed else "fail",
            measured_on_gds=True,
            member_count=len(mos_members),
            active_layer_keys=active_keys,
            envelope_bbox_nm=bbox,
            active_area_nm2=active_area,
            envelope_area_nm2=envelope_area,
            observed_fraction=observed,
            reason=(
                f"exact-GDS functional OD fraction {observed:.12g} is within "
                f"declared minimum {declaration.min_fraction:.12g}"
                if passed
                else f"exact-GDS functional OD fraction {observed:.12g} is below "
                f"declared minimum {declaration.min_fraction:.12g}"
            ),
        ),
    )


def verify_channels_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    channel_intents: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    final_placement_certificate: Any,
    placement_state_sha256: str | None,
    tech: Any,
    eps: int = ARTWORK_EPS,
) -> Tuple[GdsChannelObligation, ...]:
    """Replay one vertical channel from canonical source onto exact GDS artwork.

    The final-placement certificate identifies the materializer's rigid-block
    partition and signed channel width.  It is causal input only: every block
    member is independently claimed from the reread GDS before the corridor is
    measured, and the certificate is replayed against the exact emitter geometry.
    """

    from clay.layout.intent import ChannelIntent, replay_channel_intent

    if type(channel_intents) is not tuple:
        raise TypeError("channel_intents must be an exact tuple")
    if type(eps) is not int:
        raise TypeError("channel eps must be an exact integer")
    if not 0 <= eps <= ARTWORK_EPS:
        raise ValueError(f"channel eps must be bounded in [0,{ARTWORK_EPS}]")
    if len(channel_intents) > 64:
        raise ValueError("channel declaration inventory is too large")
    if any(type(item) is not ChannelIntent for item in channel_intents):
        raise TypeError("channel_intents must contain exact ChannelIntent values")
    declarations = tuple(replay_channel_intent(item) for item in channel_intents)
    declaration_sha256s = tuple(item.sha256 for item in declarations)
    if len(set(declaration_sha256s)) != len(declaration_sha256s):
        raise ValueError("duplicate channel declarations are forbidden")
    if not declarations:
        return ()

    def width_nm(declaration: ChannelIntent) -> int:
        return int(round(declaration.width_um * 1000.0))

    def blocked_all(
        reason: str,
        *,
        status: Literal["unmeasurable", "unsupported"] = "unmeasurable",
        gds_sha256: str | None = None,
    ) -> Tuple[GdsChannelObligation, ...]:
        return tuple(
            GdsChannelObligation(
                orient=declaration.orient,
                declared_width_nm=width_nm(declaration),
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status=status,
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declarations
        )

    if len(declarations) != 1:
        return blocked_all(
            "exact channel checker supports exactly one declaration",
            status="unsupported",
        )
    declaration = declarations[0]
    if declaration.orient != "vertical":
        return blocked_all(
            "exact channel checker supports vertical corridors only",
            status="unsupported",
        )
    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.place.final_placement_state import (
            final_placement_state_digest,
            verify_placement_causality_basis,
        )
        from clay.layout.place_blocks_2d import (
            FinalPlacementCertificate,
            identify_left_column,
            placement_certificate_is_signable,
        )
        from clay.layout.policy import policy_for_cell
        from clay.layout.route.exact_artifact_evidence import (
            ExactEmittedArtifact,
            _replayed_source_cell_authority,
        )
        from clay.layout.route.route_cell import (
            _final_placement_provenance_digest,
            build_final_placement_resource_evidence,
        )

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("exact channel checking requires ExactEmittedArtifact")
        if type(final_placement_certificate) is not FinalPlacementCertificate:
            raise TypeError(
                "exact channel checking requires FinalPlacementCertificate"
            )
        if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
            raise TypeError("exact channel checking requires emitter mappings")
        if tech is None:
            raise TypeError("exact channel checking requires resolved layout tech")
        if (
            type(placement_state_sha256) is not str
            or len(placement_state_sha256) != 64
            or any(char not in "0123456789abcdef" for char in placement_state_sha256)
        ):
            raise ValueError("channel placement state must be a lowercase SHA-256")
        exact_artifact.validate()
        if exact_artifact.scope.top_cell != str(top_cell):
            raise ValueError("exact channel artifact is bound to another top cell")
        (
            source_sha256,
            inventory_sha256,
            _member_classes,
            canonical_source_cell,
        ) = _replayed_source_cell_authority(source_cell)
        if (
            source_sha256 != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha256
            != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("channel source inventory differs from emitter scope")
        canonical_intent = layout_intent_from_cell_def(canonical_source_cell)
        canonical_policy = policy_for_cell(
            canonical_source_cell._clay_file, canonical_source_cell
        )
        if tuple(canonical_intent.channels) != declarations:
            raise ValueError("channel declarations differ from canonical source bytes")
        if not tuple(canonical_intent.common_centroids):
            raise ValueError("vertical channel lacks canonical common-centroid owner")
        if not placement_certificate_is_signable(final_placement_certificate):
            raise ValueError("channel final placement certificate is not signable")
        replayed_state = final_placement_state_digest(final_placement_certificate)
        if replayed_state != placement_state_sha256:
            raise ValueError("channel final placement state digest does not replay")
        verify_placement_causality_basis(final_placement_certificate)
        expected_ref = (
            "placement:channels:000:" + declaration.sha256[:16]
        )
        declaration_refs = final_placement_certificate.placement_declaration_refs
        if type(declaration_refs) is not tuple or expected_ref not in declaration_refs:
            raise ValueError("channel declaration is absent from placement causality")
        if final_placement_certificate.declared_channel_nm != width_nm(declaration):
            raise ValueError("declared channel width differs from final placement")
        resource_evidence = build_final_placement_resource_evidence(
            dict(placement),
            dict(geometry),
            canonical_intent,
            tech,
            final_placement_certificate=final_placement_certificate,
        )
        provenance_digest = _final_placement_provenance_digest(
            resource_evidence, final_placement_certificate
        )
        if provenance_digest != replayed_state:
            raise ValueError("channel final placement provenance does not replay")
        truth_claim = exact_artifact.truth_manifest.final_placement_certificate_sha256
        if truth_claim is not None and truth_claim != replayed_state:
            raise ValueError("exact GDS truth manifest binds another placement state")
        transformed_layers = _certified_assembly_transformed_layers(
            exact_artifact, gds_path, geometry, placement
        )
        shapes_by_layer = _read_top_cell_shapes(gds_path)
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        return blocked_all(
            "exact channel artwork/placement cannot be replayed: "
            f"{type(exc).__name__}: {exc}"
        )

    basis = final_placement_certificate.placement_causality_basis
    claimed_cache: Dict[str, Tuple[int, int, int, int]] = {}
    claimed_regions_by_layer: Dict[Tuple[int, int], Any] = {}

    def claim_member(member: str) -> Tuple[int, int, int, int]:
        cached = claimed_cache.get(member)
        if cached is not None:
            return cached
        stamp = geometry.get(member)
        position = placement.get(member)
        if stamp is None or position is None:
            raise ValueError(f"channel member {member!r} has no emitter artwork")
        expected = _merged_shape_witness(_emitted_shape_witness(
            stamp, position, excluded_layers=transformed_layers
        ))
        bbox = _union_bbox(
            [shape.bbox for shapes in expected.values() for shape in shapes]
        )
        if not expected or bbox is None:
            raise ValueError(f"channel member {member!r} has empty artwork")
        pending_regions = {}
        for key, shapes in expected.items():
            expected_region = _region_from_shapes(shapes)
            actual_region = _region_from_shapes(shapes_by_layer.get(key, ()))
            if not (expected_region - actual_region).is_empty():
                raise ValueError(
                    f"channel member {member!r} artwork is missing from exact GDS"
                )
            prior = claimed_regions_by_layer.get(key)
            if prior is not None and not (expected_region & prior).is_empty():
                raise ValueError(
                    f"channel member {member!r} reuses another member's GDS artwork"
                )
            pending_regions[key] = expected_region
        for key, region in pending_regions.items():
            if key in claimed_regions_by_layer:
                claimed_regions_by_layer[key] += region
            else:
                claimed_regions_by_layer[key] = region
        claimed_cache[member] = bbox
        return bbox

    try:
        actual_blocks: Dict[str, Tuple[int, int, int, int]] = {}
        block_members: Dict[str, Tuple[str, ...]] = {}
        block_emitter_members: Dict[str, Tuple[str, ...]] = {}
        clusters_by_stamp = {
            cluster.stamp: cluster
            for cluster in resource_evidence.cluster_index.clusters
        }
        claimed_stamp_owners: Dict[str, str] = {}
        for name, _bbox, members in basis.blocks:
            block_members[name] = tuple(members)
            emitter_members = set()
            for member in members:
                if member in geometry:
                    emitter_members.add(member)
                cluster = clusters_by_stamp.get(member)
                if cluster is not None:
                    emitter_members.update(
                        item for item in cluster.members if item in geometry
                    )
            if not emitter_members:
                raise ValueError(
                    f"channel block {name!r} has no resource-bound emitter members"
                )
            block_emitter_members[name] = tuple(sorted(emitter_members))
            for member in emitter_members:
                prior_owner = claimed_stamp_owners.setdefault(member, name)
                if prior_owner != name:
                    raise ValueError(
                        f"channel emitter member {member!r} has multiple block owners"
                    )
            boxes = [claim_member(member) for member in sorted(emitter_members)]
            actual_blocks[name] = (
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            )
    except Exception as exc:  # noqa: BLE001 - per-member typed evidence
        return blocked_all(
            f"exact channel artwork cannot be claimed: {type(exc).__name__}: {exc}",
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
        )

    functional_groups: Dict[str, Tuple[str, ...]] = {}
    for role in tuple(getattr(canonical_source_cell, "functional_roles", ()) or ()):
        if getattr(role, "kind", "") != "functional_group":
            continue
        attrs = getattr(role, "attributes", {}) or {}
        members = tuple(str(item) for item in (attrs.get("devices") or ()))
        if members:
            functional_groups[str(getattr(role, "name", "") or "")] = members
    cc_tokens = set()
    for common_centroid in canonical_intent.common_centroids:
        group = str(common_centroid.group)
        cc_tokens.add(group)
        cc_tokens.update(functional_groups.get(group, ()))
    checker_block_members = {
        name: (
            set(members)
            | set(block_emitter_members.get(name, ()))
            | ({name} if name in cc_tokens else set())
        )
        for name, members in block_members.items()
    }
    try:
        left_owner = identify_left_column(
            tuple(actual_blocks),
            checker_block_members,
            cc_tokens,
            declared_left_groups=(
                (canonical_policy.left_column_group,)
                if canonical_policy.left_column_group
                else ()
            ),
            require_unique=True,
        )
    except (TypeError, ValueError) as exc:
        return blocked_all(
            "exact channel cannot replay its declared left-column owner: "
            f"{type(exc).__name__}: {exc}",
            status="unsupported",
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
        )
    if left_owner is None:
        return blocked_all(
            "exact channel has no unique common-centroid owner in placement causality",
            status="unsupported",
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
        )
    bands = dict(final_placement_certificate.bands)
    lower_owners = tuple(sorted(
        name
        for name in actual_blocks
        if name != left_owner and bands.get(name) != 2
    ))
    if not lower_owners:
        return blocked_all(
            "exact channel has no lower-region owner opposite the left column",
            status="unsupported",
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
        )

    left_bbox = actual_blocks[left_owner]
    minimum_lower_x = min(actual_blocks[name][0] for name in lower_owners)
    observed_clearance = minimum_lower_x - left_bbox[2]
    lower_y0 = min(actual_blocks[name][1] for name in lower_owners)
    lower_y1 = max(actual_blocks[name][3] for name in lower_owners)
    corridor_y0 = max(left_bbox[1], lower_y0)
    corridor_y1 = min(left_bbox[3], lower_y1)
    corridor_bbox = (
        left_bbox[2], corridor_y0, minimum_lower_x, corridor_y1
    )
    measurable = corridor_bbox[2] > corridor_bbox[0] and (
        corridor_bbox[3] > corridor_bbox[1]
    )
    intruders = tuple(sorted(
        name
        for name, bbox in actual_blocks.items()
        if name not in {left_owner, *lower_owners}
        and max(corridor_bbox[0], bbox[0]) < min(corridor_bbox[2], bbox[2])
        and max(corridor_bbox[1], bbox[1]) < min(corridor_bbox[3], bbox[3])
    ))
    satisfied = (
        measurable
        and observed_clearance >= width_nm(declaration)
        and not intruders
    )
    reason = (
        "exact GDS contains the declared device-free vertical corridor"
        if satisfied
        else "exact GDS does not contain the declared device-free vertical corridor: "
        f"observed={observed_clearance}nm, required={width_nm(declaration)}nm, "
        f"intruders={intruders}"
    )
    return (
        GdsChannelObligation(
            orient=declaration.orient,
            declared_width_nm=width_nm(declaration),
            declaration_sha256=declaration.sha256,
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
            status="pass" if satisfied else "fail",
            measured_on_gds=measurable,
            left_owner_block_id=left_owner,
            lower_owner_block_ids=lower_owners,
            corridor_bbox_nm=corridor_bbox if measurable else None,
            observed_clearance_nm=observed_clearance,
            reason=reason,
        ),
    )


_ARTWORK_TRANSFORMS = (
    "R0",
    "R90",
    "R180",
    "R270",
    "MX",
    "MY",
    "MXR90",
    "MYR90",
)


def _transform_point(x: int, y: int, transform: str) -> Tuple[int, int]:
    if transform == "R0":
        return (x, y)
    if transform == "R90":
        return (-y, x)
    if transform == "R180":
        return (-x, -y)
    if transform == "R270":
        return (y, -x)
    if transform == "MX":
        return (x, -y)
    if transform == "MY":
        return (-x, y)
    if transform == "MXR90":
        return (y, x)
    if transform == "MYR90":
        return (-y, -x)
    raise ValueError(f"unknown artwork transform {transform!r}")


def _transform_shape(shape: _ShapeGeometry, transform: str) -> _ShapeGeometry:
    contours = tuple(
        tuple(_transform_point(x, y, transform) for x, y in contour)
        for contour in shape.contours
    )
    transformed = _shape_geometry(contours[0], contours[1:])
    if transformed is None:  # pragma: no cover - a rigid transform preserves area
        raise ValueError("rigid transform produced degenerate artwork")
    return transformed


def _shift_shape(shape: _ShapeGeometry, dx: int, dy: int) -> _ShapeGeometry:
    shifted = _shape_geometry(
        tuple((x + dx, y + dy) for x, y in shape.contours[0]),
        tuple(
            tuple((x + dx, y + dy) for x, y in contour)
            for contour in shape.contours[1:]
        ),
    )
    if shifted is None:  # pragma: no cover - translation preserves area
        raise ValueError("translation produced degenerate artwork")
    return shifted


def _shape_close(left: _ShapeGeometry, right: _ShapeGeometry, eps: int) -> bool:
    if any(
        abs(int(a) - int(b)) > eps
        for a, b in zip(left.bbox, right.bbox)
    ):
        return False
    same_vertex_walk = (
        len(left.contours) == len(right.contours)
        and all(
            len(left_contour) == len(right_contour)
            for left_contour, right_contour in zip(left.contours, right.contours)
        )
        and all(
            abs(int(left_value) - int(right_value)) <= eps
            for left_contour, right_contour in zip(left.contours, right.contours)
            for left_point, right_point in zip(left_contour, right_contour)
            for left_value, right_value in zip(left_point, right_point)
        )
    )
    if same_vertex_walk:
        return True
    if left.bbox != right.bbox:
        return False
    left_region = _region_from_shapes((left,))
    right_region = _region_from_shapes((right,))
    return (
        (left_region - right_region).is_empty()
        and (right_region - left_region).is_empty()
    )


def _candidate_gds_shapes(
    window: _Bbox,
    shapes_by_layer: Mapping[Tuple[int, int], Sequence[_Bbox]],
    expected_layers: Sequence[Tuple[int, int]],
    eps: int,
) -> Dict[Tuple[int, int], list]:
    """Window-contained candidates, restricted to witnessed device layers."""

    claimed: Dict[Tuple[int, int], list] = {}
    for key in sorted(set(expected_layers)):
        for shape in shapes_by_layer.get(key, ()):
            if _contains(window, shape.bbox, eps):
                claimed.setdefault(key, []).append(shape)
    for shapes in claimed.values():
        shapes.sort()
    return claimed


def _match_owned_artwork(
    expected: Mapping[Tuple[int, int], Sequence[_Bbox]],
    candidates: Mapping[Tuple[int, int], Sequence[_Bbox]],
    eps: int,
) -> Tuple[Optional[Dict[Tuple[int, int], list]], Optional[str]]:
    """Find the complete expected artwork in GDS under one rigid D4 transform.

    Translation is allowed so a post-placement shift remains measurable.  Rotation
    and reflection are returned to the orientation checker.  Every expected shape
    must be consumed once; foreign routing/fill candidates may remain unused.
    """

    expected_items = tuple(
        (key, shape)
        for key in sorted(expected)
        for shape in sorted(expected[key])
    )
    if not expected_items:
        return None, None
    anchor_key, _anchor_shape = expected_items[0]

    for transform in _ARTWORK_TRANSFORMS:
        transformed = tuple(
            (key, _transform_shape(shape, transform))
            for key, shape in expected_items
        )
        transformed_anchor = transformed[0][1]
        for actual_anchor in candidates.get(anchor_key, ()):
            dx = int(actual_anchor.bbox[0] - transformed_anchor.bbox[0])
            dy = int(actual_anchor.bbox[1] - transformed_anchor.bbox[1])
            unused = {
                key: list(sorted(boxes)) for key, boxes in candidates.items()
            }
            matched: Dict[Tuple[int, int], list] = {}
            complete = True
            for key, transformed_shape in transformed:
                target = _shift_shape(transformed_shape, dx, dy)
                bucket = unused.get(key, [])
                index = next(
                    (
                        idx
                        for idx, actual in enumerate(bucket)
                        if _shape_close(actual, target, eps)
                    ),
                    None,
                )
                if index is None:
                    complete = False
                    break
                matched.setdefault(key, []).append(bucket.pop(index))
            if complete:
                return matched, transform
    return None, None


def verify_anchors_on_gds(
    gds_path: str,
    *,
    top_cell: str,
    anchor_intents: Sequence[Any],
    place_groups: Sequence[Any],
    emitted_device_geometry: Mapping[str, Any] | None,
    emitted_device_placement_um: Mapping[str, Any] | None,
    exact_artifact: Any,
    source_cell: Any,
    row_final_placement: Any = None,
    tech: Any = None,
    eps: int = ARTWORK_EPS,
) -> Tuple[GdsAnchorObligation, ...]:
    """Replay left/right edge access from canonical source onto exact GDS.

    Net targets combine the artifact-bound exhaustive route manifest with the
    complete exact-GDS artwork independently claimed for every source member on
    that net.  Device/place-group targets use the same artwork claims.  Placement
    certificates and caller-authored bboxes are never used.
    """

    from clay.layout.intent import AnchorIntent, PlaceGroup

    if type(anchor_intents) is not tuple:
        raise TypeError("anchor_intents must be an exact tuple")
    if type(place_groups) is not tuple:
        raise TypeError("anchor place_groups must be an exact tuple")
    if type(eps) is not int:
        raise TypeError("anchor eps must be an exact integer")
    if not 0 <= eps <= ARTWORK_EPS:
        raise ValueError(f"anchor eps must be bounded in [0,{ARTWORK_EPS}]")
    if len(anchor_intents) > 64:
        raise ValueError("anchor declaration inventory is too large")
    if any(type(item) is not AnchorIntent for item in anchor_intents):
        raise TypeError("anchor_intents must contain exact AnchorIntent values")
    declaration_sha256s = tuple(item.sha256 for item in anchor_intents)
    if len(set(declaration_sha256s)) != len(declaration_sha256s):
        raise ValueError("duplicate anchor declarations are forbidden")
    if len(place_groups) > 4096:
        raise ValueError("anchor place-group inventory is too large")
    group_names: list[str] = []
    for group in place_groups:
        if type(group) is not PlaceGroup or type(group.devices) is not tuple:
            raise TypeError("anchor place_groups must contain exact tuple-backed values")
        if len(group.devices) > 4096:
            raise ValueError("anchor place-group membership is too large")
        if (
            type(group.group) is not str
            or not group.group
            or group.group != group.group.strip()
        ):
            raise ValueError("anchor place-group names must be canonical and non-empty")
        if len(group.group) > 256:
            raise ValueError(f"anchor place-group {group.group!r} name is too long")
        if not group.devices:
            raise ValueError(
                f"anchor place-group {group.group!r} must have non-empty membership"
            )
        if any(
            type(member) is not str
            or not member
            or member != member.strip()
            for member in group.devices
        ):
            raise ValueError(
                f"anchor place-group {group.group!r} has noncanonical members"
            )
        if any(len(member) > 256 for member in group.devices):
            raise ValueError(
                f"anchor place-group {group.group!r} member is too long"
            )
        if len(set(group.devices)) != len(group.devices):
            raise ValueError(
                f"anchor place-group {group.group!r} has duplicate members"
            )
        group_names.append(group.group)
    if len(set(group_names)) != len(group_names):
        raise ValueError("duplicate place-group names are forbidden for anchors")
    declarations = anchor_intents
    if not declarations:
        return ()

    def blocked_all(
        reason: str,
        *,
        status: Literal["unmeasurable", "unsupported"] = "unmeasurable",
        gds_sha256: str | None = None,
    ) -> Tuple[GdsAnchorObligation, ...]:
        return tuple(
            GdsAnchorObligation(
                target=declaration.target,
                edge=declaration.edge,
                declaration_sha256=declaration.sha256,
                gds_sha256=gds_sha256,
                status=status,
                measured_on_gds=False,
                reason=reason,
            )
            for declaration in declarations
        )

    geometry = emitted_device_geometry
    placement = emitted_device_placement_um
    try:
        from clay.layout.intent import layout_intent_from_cell_def
        from clay.layout.route.exact_artifact_evidence import (
            ExactEmittedArtifact,
            _canonical_source_member_inventory,
            _replayed_source_cell_authority,
        )

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError("exact anchor checking requires ExactEmittedArtifact")
        if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
            raise TypeError("exact anchor checking requires emitter mappings")
        exact_artifact.validate()
        if exact_artifact.scope.top_cell != str(top_cell):
            raise ValueError("exact anchor artifact is bound to another top cell")
        (
            source_sha256,
            inventory_sha256,
            _member_classes,
            canonical_source_cell,
        ) = _replayed_source_cell_authority(source_cell)
        if (
            source_sha256 != exact_artifact.scope.source_cell_source_sha256
            or inventory_sha256
            != exact_artifact.scope.source_member_inventory_sha256
        ):
            raise ValueError("anchor source inventory differs from emitter scope")
        canonical_intent = layout_intent_from_cell_def(canonical_source_cell)
        if tuple(canonical_intent.anchors) != declarations:
            raise ValueError("anchor declarations differ from canonical source bytes")
        if tuple(canonical_intent.place_groups) != place_groups:
            raise ValueError("anchor place groups differ from canonical source bytes")
        inventory = _canonical_source_member_inventory(canonical_source_cell)
        macro_owner_by_member: Dict[str, str] = {}
        has_row_declarations = any((
            tuple(canonical_intent.arrays),
            tuple(canonical_intent.common_centroids),
        ))
        if has_row_declarations:
            from clay.layout.generic_placement import (
                Placement,
                _placed_artwork_sha256,
                _translate,
                verify_stored_row_projection,
            )
            from clay.layout.place.row_program import (
                compile_row_program,
                materialize_row_program,
            )

            if row_final_placement is None:
                raise ValueError(
                    "exact anchor row ownership requires final row placement"
                )
            if type(row_final_placement) is not Placement:
                raise TypeError("exact anchor row placement must be exact Placement")
            if tech is None:
                raise TypeError("exact anchor row ownership requires layout tech")
            verify_stored_row_projection(row_final_placement)
            replayed_program = compile_row_program(
                canonical_source_cell, canonical_intent, tech
            )
            if replayed_program.residuals:
                raise ValueError(
                    "canonical anchor row compiler produced residuals: "
                    f"{tuple(item.code for item in replayed_program.residuals)}"
                )
            stored_program = row_final_placement.row_program
            if (
                stored_program is None
                or stored_program.to_canonical_bytes()
                != replayed_program.to_canonical_bytes()
            ):
                raise ValueError(
                    "anchor row program differs from canonical source replay"
                )
            canonical_names = {str(row[0]) for row in inventory}
            if set(row_final_placement.devices) != canonical_names:
                raise ValueError(
                    "anchor row placement member inventory differs from source"
                )
            replayed_materialization = materialize_row_program(
                replayed_program, tech
            )
            if replayed_materialization.residuals:
                raise ValueError(
                    "canonical anchor row materializer produced residuals: "
                    f"{tuple(item.code for item in replayed_materialization.residuals)}"
                )
            stored_materialization = row_final_placement.row_materialization
            if (
                stored_materialization is None
                or stored_materialization.certificate.to_canonical_bytes()
                != replayed_materialization.certificate.to_canonical_bytes()
            ):
                raise ValueError(
                    "anchor row materialization differs from canonical replay"
                )
            final = row_final_placement.row_final_certificate
            if final is None:
                raise ValueError("anchor row placement lacks final certificate")
            realizations = {
                item.entry_id: item
                for item in replayed_materialization.realizations
            }
            bindings = {item.entry_id: item for item in final.macro_bindings}
            for entry in replayed_program.entries:
                group_stamp = geometry.get(entry.group)
                group_position = placement.get(entry.group)
                if group_stamp is None or group_position is None:
                    raise ValueError(
                        f"anchor row group {entry.group!r} lacks one emitter owner"
                    )
                composite_members = getattr(
                    group_stamp, "composite_members", None
                )
                if (
                    type(composite_members) is not tuple
                    or len(composite_members) != len(set(composite_members))
                    or tuple(sorted(composite_members))
                    != tuple(sorted(entry.consumed_devices))
                ):
                    raise ValueError(
                        f"anchor row group {entry.group!r} member provenance differs "
                        "from canonical replay: "
                        f"expected={entry.consumed_devices}, "
                        f"observed={composite_members}"
                    )
                realization = realizations.get(entry.entry_id)
                binding = bindings.get(entry.entry_id)
                if (
                    realization is None
                    or binding is None
                    or binding.entry_sha256 != entry.sha256
                    or binding.materialization_sha256
                    != replayed_materialization.certificate.sha256
                    or binding.owner != entry.owner
                    or binding.consumed_devices != entry.consumed_devices
                ):
                    raise ValueError(
                        "anchor row macro binding differs from canonical replay"
                    )
                lx0, ly0, lx1, ly1 = realization.local_bbox_nm
                bx0, by0, bx1, by1 = binding.bbox_nm
                translations = {
                    (bx0 - lx0, by0 - ly0),
                    (bx1 - lx1, by1 - ly1),
                }
                if len(translations) != 1:
                    raise ValueError(
                        "anchor row macro binding is not one rigid translation"
                    )
                dx_nm, dy_nm = next(iter(translations))
                placed_polygons = _translate(
                    list(getattr(realization.geometry, "polygons", ()) or ()),
                    dx_nm,
                    dy_nm,
                )
                if _placed_artwork_sha256(placed_polygons) != binding.artwork_sha256:
                    raise ValueError(
                        "anchor row macro placed artwork digest does not replay"
                    )
                layer_keys = tuple(sorted({
                    (
                        int(getattr(polygon, "layer")),
                        int(getattr(polygon, "datatype")),
                    )
                    for polygon in placed_polygons
                }))
                expected_witness = _merged_shape_witness(
                    _layer_polygons_witness(
                        placed_polygons, layer_keys=layer_keys
                    )
                )
                emitted_witness = _merged_shape_witness(
                    _emitted_shape_witness(group_stamp, group_position)
                )
                if emitted_witness != expected_witness:
                    raise ValueError(
                        f"anchor row group {entry.group!r} emitter artwork differs "
                        "from canonical final binding"
                    )
                for member in entry.consumed_devices:
                    emitter_owner = _unique_emitter_owner(member, geometry)
                    if emitter_owner != entry.group:
                        raise ValueError(
                            f"anchor row member {member!r} is emitted by "
                            f"{emitter_owner!r}, not canonical group {entry.group!r}"
                        )
                    if member in macro_owner_by_member:
                        raise ValueError(
                            f"anchor row member {member!r} has multiple owners"
                        )
                    macro_owner_by_member[member] = entry.group
        transformed_layers = _certified_assembly_transformed_layers(
            exact_artifact, gds_path, geometry, placement
        )
        shapes_by_layer = _read_top_cell_shapes(gds_path)
    except Exception as exc:  # noqa: BLE001 - typed exact-artifact boundary
        return blocked_all(
            f"exact anchor artifact cannot be replayed: {type(exc).__name__}: {exc}"
        )

    source_names = tuple(sorted(str(row[0]) for row in inventory))
    pins_by_name = {str(row[0]): dict(row[4]) for row in inventory}
    net_members: Dict[str, set[str]] = {}
    for name, pins in pins_by_name.items():
        for net in pins.values():
            net_members.setdefault(str(net), set()).add(name)
    group_members = {
        str(group.group): tuple(sorted(str(name) for name in group.devices))
        for group in place_groups
    }
    foreign_group_members = tuple(
        sorted(
            (group_name, member)
            for group_name, members in group_members.items()
            for member in members
            if member not in source_names
        )
    )
    if foreign_group_members:
        return blocked_all(
            "anchor place-group members are absent from canonical source inventory: "
            f"{foreign_group_members}",
            status="unsupported",
            gds_sha256=exact_artifact.scope.exact_gds_sha256,
        )
    route_entries = tuple(exact_artifact.route_manifest.entries)
    gds_universe_bbox = _union_bbox([
        shape.bbox
        for shapes in shapes_by_layer.values()
        for shape in shapes
    ])
    available_shapes_by_layer = {
        key: list(dict.fromkeys(shapes))
        for key, shapes in shapes_by_layer.items()
    }
    claimed_cache: Dict[str, Tuple[int, int, int, int]] = {}

    def consume_claimed(
        claimed: Mapping[Tuple[int, int], Sequence[_ShapeGeometry]],
    ) -> None:
        for key, shapes in claimed.items():
            available = available_shapes_by_layer.get(key, [])
            for shape in shapes:
                index = next(
                    (
                        candidate_index
                        for candidate_index, candidate in enumerate(available)
                        if candidate is shape
                    ),
                    None,
                )
                if index is None:
                    raise ValueError(
                        "anchor GDS artwork lacks a unique unclaimed occurrence"
                    )
                available.pop(index)

    def claim_member(member: str) -> Tuple[int, int, int, int]:
        cached = claimed_cache.get(member)
        if cached is not None:
            return cached
        stamp = geometry.get(member)
        position = placement.get(member)
        if stamp is None or position is None:
            raise ValueError(f"anchor member {member!r} has no direct emitter artwork")
        expected = _emitted_shape_witness(
            stamp,
            position,
            excluded_layers=transformed_layers,
        )
        window = _union_bbox(
            [shape.bbox for shapes in expected.values() for shape in shapes]
        )
        if not expected or window is None:
            raise ValueError(f"anchor member {member!r} has an empty artwork witness")
        candidates = _candidate_gds_shapes(
            window, available_shapes_by_layer, tuple(expected), eps
        )
        claimed, _transform = _match_owned_artwork(expected, candidates, eps)
        if claimed is None:
            raise ValueError(
                f"anchor member {member!r} lacks unique artwork in exact GDS"
            )
        consume_claimed(claimed)
        bbox = _union_bbox(
            [shape.bbox for shapes in claimed.values() for shape in shapes]
        )
        if bbox is None:
            raise ValueError(f"anchor member {member!r} has no claimed bbox")
        claimed_cache[member] = bbox
        return bbox

    def claim_target_member(member: str) -> Tuple[int, int, int, int]:
        return claim_member(macro_owner_by_member.get(member, member))

    exact_sha256 = exact_artifact.scope.exact_gds_sha256
    results: list[GdsAnchorObligation] = []
    artwork_universe_bbox: Tuple[int, int, int, int] | None = None
    for declaration in declarations:
        target = declaration.target
        if declaration.edge not in {"left", "right"}:
            results.append(GdsAnchorObligation(
                target=target,
                edge=declaration.edge,
                declaration_sha256=declaration.sha256,
                gds_sha256=exact_sha256,
                status="unsupported",
                measured_on_gds=False,
                reason="exact anchor checker supports left/right edges only",
            ))
            continue
        resolutions = []
        if target in source_names:
            resolutions.append(("device", (target,)))
        if target in group_members:
            resolutions.append(("group", group_members[target]))
        if target in net_members:
            resolutions.append(("net", tuple(sorted(net_members[target]))))
        common = dict(
            target=target,
            edge=declaration.edge,
            declaration_sha256=declaration.sha256,
            gds_sha256=exact_sha256,
        )
        if len(resolutions) != 1:
            results.append(GdsAnchorObligation(
                **common,
                status="unsupported",
                measured_on_gds=False,
                reason=(
                    f"anchor target {target!r} has "
                    f"{len(resolutions)} canonical source resolutions"
                ),
            ))
            continue
        resolution, members = resolutions[0]
        try:
            if resolution == "net":
                routed_bbox = _union_bbox([
                    tuple(row[2]) for row in route_entries if row[0] == target
                ])
                if routed_bbox is None:
                    raise ValueError(
                        f"anchor net {target!r} has no exact routed ownership"
                    )
                # Claim every canonical member to retain exhaustive source/GDS
                # ownership checks, but do not let incidental member artwork
                # satisfy a route obligation.  A net anchor is produced by the
                # declared net's exact route and is measured on that route.
                for name in members:
                    claim_target_member(name)
                target_bbox = routed_bbox
                universe_bbox = gds_universe_bbox
                if target_bbox is None or universe_bbox is None:
                    raise ValueError("anchor net artwork envelope is empty")
            else:
                target_bbox = _union_bbox(
                    [claim_target_member(name) for name in members]
                )
                if artwork_universe_bbox is None:
                    artwork_universe_bbox = _union_bbox(
                        [claim_target_member(name) for name in source_names]
                    )
                universe_bbox = artwork_universe_bbox
                if target_bbox is None or universe_bbox is None:
                    raise ValueError("anchor artwork envelope is empty")
        except Exception as exc:  # noqa: BLE001 - per-anchor typed evidence
            results.append(GdsAnchorObligation(
                **common,
                status="unmeasurable",
                measured_on_gds=False,
                resolution=resolution,
                members=members,
                reason=f"anchor cannot be measured: {type(exc).__name__}: {exc}",
            ))
            continue
        target_edge = target_bbox[0] if declaration.edge == "left" else target_bbox[2]
        universe_edge = (
            universe_bbox[0] if declaration.edge == "left" else universe_bbox[2]
        )
        satisfied = abs(int(target_edge) - int(universe_edge)) <= eps
        results.append(GdsAnchorObligation(
            **common,
            status="pass" if satisfied else "fail",
            measured_on_gds=True,
            resolution=resolution,
            members=members,
            target_bbox_nm=target_bbox,
            universe_bbox_nm=universe_bbox,
            reason=(
                (
                    f"exact GDS route producer reaches the declared "
                    f"{declaration.edge} edge"
                    if resolution == "net"
                    else f"exact GDS target reaches the declared {declaration.edge} edge"
                )
                if satisfied
                else (
                    f"exact GDS route producer misses the declared "
                    f"{declaration.edge} edge"
                    if resolution == "net"
                    else f"exact GDS target misses the declared {declaration.edge} edge"
                )
            ),
        ))
    return tuple(results)


def _realized_orientation(base: str, transform: str) -> Optional[str]:
    """Compose materialized rotation evidence with an exact-GDS mutation."""

    base = str(base or "")
    if not base:
        return None
    rotations = {"R0": 0, "R90": 90, "R180": 180, "R270": 270}
    if base in rotations and transform in rotations:
        angle = (rotations[base] + rotations[transform]) % 360
        return f"R{angle}"
    if transform == "R0":
        return base
    return f"{transform}({base})"


def _cap_footprint_bbox(
    claimed: Mapping[Tuple[int, int], Sequence[_Bbox]],
) -> Optional[_Bbox]:
    """The cap's TRUE footprint: the union of its MIM PLATE (FuseTop 75/0) shapes.

    A MIM cap's emitted value lives in its FuseTop top-plate AREA, so a cap's
    realized footprint is its FuseTop plate, not the (placer-reserved) full-cell
    bbox that the via-sea / Metal3 cover / marker layers extend to. Returns ``None``
    when the window carries no FuseTop plate (then the caller falls back to the
    union of all claimed shapes -- a non-cap device, or a cap with no plate, which
    is itself a finding the equal_size check will surface).
    """
    plate = claimed.get(MIM_PLATE_LAYER)
    if not plate:
        return None
    return _union_bbox([shape.bbox for shape in plate])


def geometric_placement_from_gds(
    gds_path: str,
    *,
    device_labels: Mapping[str, Any],
    intent: Any = None,
    eps: int = EPS,
    matching_pair_axes: Optional[
        Mapping[Tuple[str, str], Tuple[str, float]]
    ] = None,
    matching_pair_physical_bboxes: Optional[
        Mapping[
            Tuple[str, str],
            Tuple[
                Tuple[float, float, float, float],
                Tuple[float, float, float, float],
            ],
        ]
    ] = None,
    named_region_members: Optional[Mapping[str, Sequence[str]]] = None,
    device_orientations: Optional[Mapping[str, str]] = None,
    dropped_nonmos_matched: Sequence[str] = (),
    emitted_device_geometry: Optional[Mapping[str, Any]] = None,
    emitted_device_placement_um: Optional[Mapping[str, Any]] = None,
    exact_artifact: Any = None,
    source_cell: Any = None,
    row_final_placement: Any = None,
    tech: Any = None,
) -> GeometricPlacement:
    """Build a :class:`GeometricPlacement` from the FINAL emitted GDS.

    The GDS-truth sibling of
    :meth:`clay.layout.geometric_intent_checks.GeometricPlacement.from_placement`:
    it returns the SAME type the structural geometric obligations consume, but with
    each device's bbox read from the REAL emitted ``layout.gds`` rather than the
    placer's reserved footprint.

    Parameters
    ----------
    gds_path : str
        The emitted ``layout.gds`` (the flat top cell ``emit_routed_cell`` wrote).
    device_labels : Mapping[str, PlacedDevice]
        Materialized devices carrying both ``.bbox`` and owned ``.polys``.  The
        bbox limits the search; the polygon multiset supplies ownership.  A raw
        bbox has no ownership evidence and is unmeasurable.
    emitted_device_geometry / emitted_device_placement_um : mappings, optional
        The exact device stamps and placements handed to the canonical emitter.
        When both contain a MOS label and ``exact_artifact`` binds the complete
        snapshot, its placed polygon multiset replaces stale placer artwork as
        the ownership witness.  All measurements still come from the reread GDS;
        an unbound, missing, or mutated stamp remains unmeasurable.
    exact_artifact : ExactEmittedArtifact, optional
        The validated artifact bound to ``gds_path``.  Only its typed well-merge
        certificates may exclude assembly-transformed layers from a route-stamp
        witness; arbitrary caller-supplied exclusions are not accepted.
    intent : optional
        Accepted for signature symmetry with the placement factory; unused here (the
        GDS carries no cluster provenance, so the GDS-backed placement reports NO
        fused clusters -- see the residual note in the module docstring).
    eps : int
        Grid-snap slack (dbu) when claiming a GDS shape for a window.

    Returns
    -------
    GeometricPlacement
        ``.device_bboxes`` maps each label whose complete owned-artwork witness
        was found in GDS to its REALIZED bbox. For a CAP the bbox is its FuseTop
        (75/0) MIM PLATE (the true emitted plate); for any other device it is the
        union of its matched owned shapes. A label with no complete witness is ABSENT from
        ``.device_bboxes`` (UNMEASURABLE -> the obligation blocks, never a false
        pass). ``.clusters`` / ``.cluster_member_to_a`` are EMPTY: cluster
        provenance is a placement-time artefact the flat GDS does not carry, so the
        interdig leg is honestly NOT measurable from the GDS alone (documented
        residual). The equal_size / common_centroid / place_group checks, which read
        only ``.device_bboxes``, ARE fully GDS-measurable.
    """
    shapes_by_layer = _read_top_cell_shapes(gds_path)

    device_bboxes: Dict[str, _Bbox] = {}
    realized_orientations: Dict[str, str] = {}
    supplied_orientations = {
        str(label): str(orientation)
        for label, orientation in dict(device_orientations or {}).items()
    }
    emitted_mapping_context = (
        emitted_device_geometry is not None
        or emitted_device_placement_um is not None
    )
    emitted_context = emitted_mapping_context or exact_artifact is not None
    if emitted_mapping_context and (
        not isinstance(emitted_device_geometry, Mapping)
        or not isinstance(emitted_device_placement_um, Mapping)
    ):
        raise TypeError(
            "emitted device ownership requires both geometry and placement mappings"
        )
    emitted_geometry = dict(emitted_device_geometry or {})
    emitted_placement = dict(emitted_device_placement_um or {})
    named_regions = {
        str(name): tuple(str(member) for member in members)
        for name, members in dict(named_region_members or {}).items()
    }
    assembly_transformed_layers: Tuple[Tuple[int, int], ...] = ()
    emitter_snapshot_bound = False
    if emitted_mapping_context and exact_artifact is not None:
        from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

        if type(exact_artifact) is not ExactEmittedArtifact:
            raise TypeError(
                "exact emitted-device ownership requires ExactEmittedArtifact"
            )
        assembly_transformed_layers = _certified_assembly_transformed_layers(
            exact_artifact,
            gds_path,
            emitted_geometry,
            emitted_placement,
        )
        emitter_snapshot_bound = True
    for label, win in device_labels.items():
        canonical_label = str(label)
        emitted_stamp = emitted_geometry.get(canonical_label)
        emitted_position = emitted_placement.get(canonical_label)
        is_mos = str(getattr(win, "kind", "")).strip().lower() in {
            "nmos",
            "pmos",
        }
        if is_mos and emitted_context and (
            emitted_stamp is None or emitted_position is None
        ):
            # A complete emitter snapshot that lacks this MOS is affirmative
            # absence. Never fall back to pre-route placer artwork.
            continue
        uses_emitted_witness = (
            is_mos
            and emitted_stamp is not None
            and emitted_position is not None
        )
        if uses_emitted_witness and not emitter_snapshot_bound:
            # A route stamp without the emitter-issued exact snapshot digest is
            # not complete ownership provenance.
            continue
        if uses_emitted_witness:
            expected = _emitted_shape_witness(
                emitted_stamp,
                emitted_position,
                excluded_layers=(
                    assembly_transformed_layers
                    if str(getattr(win, "kind", "")).strip().lower() == "pmos"
                    else ()
                ),
            )
            window = _union_bbox([
                shape.bbox for shapes in expected.values() for shape in shapes
            ])
        else:
            window = _normalize_window(win)
            expected = _owned_shape_witness(win)
        if window is None:
            continue
        if not expected:
            # A spatial window is not a device-ownership witness.
            continue
        candidates = _candidate_gds_shapes(
            window, shapes_by_layer, tuple(expected), eps
        )
        claimed, transform = _match_owned_artwork(
            expected, candidates, min(max(int(eps), 0), ARTWORK_EPS)
        )
        if not claimed or transform is None:
            # Incomplete/mutated-away artwork -> UNMEASURABLE (absent).
            continue
        # A cap's realized footprint is its MIM plate (value-sized); every other
        # device's footprint is the union of its OWN matched artwork only.
        bb = _cap_footprint_bbox(claimed)
        if bb is None:
            all_boxes = [shape.bbox for shapes in claimed.values() for shape in shapes]
            bb = _union_bbox(all_boxes)
        if bb is None:
            continue
        device_bboxes[canonical_label] = bb
        materialized_orientation = (
            "R0"
            if uses_emitted_witness
            else str(getattr(win, "orientation", "") or "")
        )
        supplied_orientation = supplied_orientations.get(canonical_label, "")
        # A separately supplied value is a consistency check, never a substitute
        # for missing materialized-device provenance.
        if (
            materialized_orientation
            and (
                uses_emitted_witness
                or not supplied_orientation
                or supplied_orientation == materialized_orientation
            )
        ):
            realized = _realized_orientation(
                materialized_orientation, transform
            )
            if realized is not None:
                realized_orientations[canonical_label] = realized

    measured_pair_boxes = dict(matching_pair_physical_bboxes or {})
    if (emitter_snapshot_bound and source_cell is not None
            and row_final_placement is not None and tech is not None):
        from collections import Counter
        from clay.layout.intent import layout_intent_from_cell_def

        row_intent = layout_intent_from_cell_def(source_cell)
        row_reports, _interleave_reports = verify_common_centroid_interdigitation_on_gds(
            gds_path, top_cell=exact_artifact.scope.top_cell,
            common_centroid_intents=row_intent.common_centroids,
            matching_intents=row_intent.matching_intents,
            emitted_device_geometry=emitted_geometry,
            emitted_device_placement_um=emitted_placement,
            exact_artifact=exact_artifact, source_cell=source_cell,
            row_program=getattr(row_final_placement, 'row_program', None),
            row_materialization=getattr(row_final_placement, 'row_materialization', None),
            row_final_placement_certificate=getattr(row_final_placement, 'row_final_certificate', None),
            row_final_placement=row_final_placement, tech=tech)
        for report in row_reports:
            if not report.measured_on_gds or not report.member_channel_geometry_nm:
                continue
            measured_members = {}
            for member in report.members:
                records = [(box, orientation) for owner, box, orientation
                           in report.member_channel_geometry_nm if owner == member]
                if not records:
                    raise ValueError('measured centroid omits a declared channel owner')
                bbox = _union_bbox([box for box, _orientation in records])
                if bbox is None:
                    raise ValueError('measured centroid member has no physical channels')
                measured_members[member] = bbox
                device_bboxes[member] = bbox
                realized_orientations[member] = repr(tuple(sorted(Counter(
                    orientation for _box, orientation in records).items())))
            pair = tuple(sorted(report.members))
            measured_pair_boxes[pair] = tuple(measured_members[member] for member in pair)
        if any(
            getattr(item, "style", "") == "shared_diffusion"
            and getattr(item, "order", "") == "ratio"
            for item in (row_intent.arrays or ())
        ):
            array_reports = verify_finger_array_orders_on_gds(
                gds_path,
                top_cell=exact_artifact.scope.top_cell,
                array_intents=tuple(row_intent.arrays),
                emitted_device_geometry=emitted_geometry,
                emitted_device_placement_um=emitted_placement,
                exact_artifact=exact_artifact,
                source_cell=source_cell,
                row_program=getattr(row_final_placement, "row_program", None),
                row_materialization=getattr(row_final_placement, "row_materialization", None),
                row_final_placement_certificate=getattr(
                    row_final_placement, "row_final_certificate", None
                ),
                row_final_placement=row_final_placement,
                tech=tech,
            )
            passed_groups = {
                report.group
                for report in array_reports
                if report.status == "pass" and report.measured_on_gds
            }
            if passed_groups:
                prog = row_final_placement.row_program
                mat = row_final_placement.row_materialization
                final_cert = row_final_placement.row_final_certificate
                entries_by_group = {entry.group: entry for entry in prog.entries}
                reals_by_id = {real.entry_id: real for real in mat.realizations}
                binds_by_id = {bind.entry_id: bind for bind in final_cert.macro_bindings}
                for group_name in sorted(passed_groups):
                    entry = entries_by_group.get(group_name)
                    if entry is None:
                        continue
                    real = reals_by_id.get(entry.entry_id)
                    bind = binds_by_id.get(entry.entry_id)
                    if real is None or bind is None:
                        continue
                    dx_nm = int(bind.bbox_nm[0]) - int(real.local_bbox_nm[0])
                    dy_nm = int(bind.bbox_nm[1]) - int(real.local_bbox_nm[1])
                    comp_bbox = getattr(real.geometry, "comp_bbox", None)
                    fingers = tuple(getattr(real.geometry, "fingers", ()) or ())
                    if comp_bbox is None or not fingers:
                        continue
                    cy0 = int(comp_bbox[1]) + dy_nm
                    cy1 = int(comp_bbox[3]) + dy_nm
                    for member in entry.active_devices:
                        if member in device_bboxes:
                            continue
                        member_fingers = [
                            finger
                            for finger in fingers
                            if str(getattr(finger, "member", "")) == member
                        ]
                        if not member_fingers:
                            continue
                        mx0 = min(int(finger.x0) for finger in member_fingers) + dx_nm
                        mx1 = max(int(finger.x1) for finger in member_fingers) + dx_nm
                        device_bboxes[member] = (mx0, cy0, mx1, cy1)

    named_region_bboxes: Dict[str, _Bbox] = {}
    emitted_owner_bboxes: Dict[str, _Bbox] = {}
    if emitter_snapshot_bound:
        for region_name, members in sorted(named_regions.items()):
            if members and all(member in device_bboxes for member in members):
                bbox = _union_bbox([device_bboxes[member] for member in members])
                if bbox is not None:
                    named_region_bboxes[region_name] = bbox
                    continue
            try:
                owners = tuple(
                    sorted(
                        {
                            _unique_emitter_owner(member, emitted_geometry)
                            for member in members
                        }
                    )
                )
            except ValueError:
                continue
            owner_boxes = []
            for owner in owners:
                bbox = emitted_owner_bboxes.get(owner)
                if bbox is None:
                    stamp = emitted_geometry.get(owner)
                    position = emitted_placement.get(owner)
                    if stamp is None or position is None:
                        owner_boxes = []
                        break
                    expected = _emitted_shape_witness(
                        stamp,
                        position,
                        excluded_layers=assembly_transformed_layers,
                    )
                    window = _union_bbox(
                        [shape.bbox for shapes in expected.values() for shape in shapes]
                    )
                    if window is None or not expected:
                        owner_boxes = []
                        break
                    candidates = _candidate_gds_shapes(
                        window, shapes_by_layer, tuple(expected), eps
                    )
                    claimed, transform = _match_owned_artwork(
                        expected,
                        candidates,
                        min(max(int(eps), 0), ARTWORK_EPS),
                    )
                    if not claimed or transform is None:
                        owner_boxes = []
                        break
                    bbox = _union_bbox(
                        [shape.bbox for shapes in claimed.values() for shape in shapes]
                    )
                    if bbox is None:
                        owner_boxes = []
                        break
                    emitted_owner_bboxes[owner] = bbox
                owner_boxes.append(bbox)
            if owner_boxes:
                region_bbox = _union_bbox(owner_boxes)
                if region_bbox is not None:
                    named_region_bboxes[region_name] = region_bbox

    return GeometricPlacement(
        device_bboxes=device_bboxes,
        clusters={},
        cluster_member_to_a={},
        matching_pair_axes={
            tuple(sorted((str(pair[0]), str(pair[1])))): (
                str(axis[0]),
                float(axis[1]),
            )
            for pair, axis in dict(matching_pair_axes or {}).items()
        },
        matching_pair_physical_bboxes={
            tuple(sorted((str(pair[0]), str(pair[1])))): tuple(
                tuple(float(value) for value in bbox) for bbox in boxes
            )
            for pair, boxes in measured_pair_boxes.items()
        },
        device_orientations=realized_orientations,
        dropped_nonmos_matched=tuple(
            str(member) for member in dropped_nonmos_matched
        ),
        named_region_bboxes=named_region_bboxes,
    )


# --------------------------------------------------------------------------- #
# canonical 8-category invariants, computed from the FINAL GDS                 #
# --------------------------------------------------------------------------- #


class _GdsPlacementView:
    """Adapter exposing a GDS-backed :class:`GeometricPlacement` as the placement
    shape :func:`clay.layout.completeness.invariance.extract_invariants` consumes.

    ``extract_invariants`` reads ``placement.devices`` -> ``{label: PlacedDevice}``
    where each device exposes ``.bbox``, plus ``placement.constraints.symmetry_pairs``
    for declared matched pairs. We synthesise that shape from the GDS-backed
    ``GeometricPlacement.device_bboxes`` (realized GDS bboxes) so the SAME 8-category
    extractor runs unchanged on the FINAL GDS geometry. The declared symmetry pairs
    are threaded from ``intent`` (when supplied) so the symmetry_axis / orientation
    categories carry the same declared pair set the placement-time extractor used.
    """

    def __init__(
        self,
        device_bboxes: Mapping[str, _Bbox],
        symmetry_pairs: Sequence[Tuple[str, str]] = (),
    ) -> None:
        self.devices = {
            str(lbl): _GdsPlacedDevice(bb) for lbl, bb in device_bboxes.items()
        }
        self.constraints = _GdsConstraints(tuple(symmetry_pairs))


class _GdsPlacedDevice:
    __slots__ = ("bbox",)

    def __init__(self, bbox: _Bbox) -> None:
        self.bbox = bbox


class _GdsConstraints:
    __slots__ = ("symmetry_pairs",)

    def __init__(self, symmetry_pairs: Tuple[Tuple[str, str], ...]) -> None:
        self.symmetry_pairs = symmetry_pairs


def _declared_symmetry_pairs(intent: Any) -> Tuple[Tuple[str, str], ...]:
    """Declared matched pairs off an intent / constraints, or ``()``.

    Accepts either an object exposing ``.symmetry_pairs`` (a
    :class:`SolverConstraints`) or a :class:`LayoutIntent` exposing ``.diff_pairs``
    / ``.mirror_pairs`` (the declared symmetric device roles). Best-effort: an
    intent that declares no pairs yields ``()`` (the symmetry_axis category is then
    empty, exactly like the placement extractor on a pair-less cell).
    """
    if intent is None:
        return ()
    sp = getattr(intent, "symmetry_pairs", None)
    if sp:
        return tuple((str(a), str(b)) for (a, b) in sp if len((a, b)) == 2)
    pairs = []
    for matching in getattr(intent, "matching_intents", ()) or ():
        pairs.append(tuple(str(member) for member in matching.members))
    for attr in ("diff_pairs", "mirror_pairs"):
        for ab in getattr(intent, attr, ()) or ():
            if len(ab) == 2:
                pairs.append((str(ab[0]), str(ab[1])))
    return tuple(pairs)


def extract_invariants_from_gds(
    gds_path: str,
    *,
    device_labels: Mapping[str, Any],
    intent: Any = None,
    eps: int = EPS,
) -> Dict[str, Any]:
    """Extract the 8 canonical layout invariants from the FINAL emitted GDS.

    Returns the SAME canonical (sorted-key, deterministic) dict shape
    :func:`clay.layout.completeness.invariance.extract_invariants` returns -- keyed
    by the 8 shared categories (``relative_order`` / ``adjacency`` / ``alignment`` /
    ``band`` / ``symmetry_axis`` / ``size_class`` / ``spacing`` / ``orientation``) --
    but computed from the device bboxes READ FROM THE GDS (cap footprints = their
    FuseTop plate). This is the SHARED INTERFACE a sibling Oracle-2 agent imports;
    the signature ``(gds_path, *, device_labels, intent=None, eps=...)`` is stable.

    Because the cap footprints are the value-sized emitted plates, the ``size_class``
    category here correctly SPLITS two caps of different value into different classes
    (whereas the placement-time extractor, reading the equal reserved footprints,
    merges them) -- the same soundness fix ``check_equal_size`` gets, surfaced in the
    invariant taxonomy.

    Determinism: builds off the GDS-backed :class:`GeometricPlacement` (sorted
    device labels) and delegates to the shared extractor, so two reads of byte-equal
    GDS produce byte-equal invariants.
    """
    from clay.layout.completeness.invariance import extract_invariants

    gp = geometric_placement_from_gds(
        gds_path, device_labels=device_labels, intent=intent, eps=eps
    )
    view = _GdsPlacementView(
        gp.device_bboxes, symmetry_pairs=_declared_symmetry_pairs(intent)
    )
    # The shared extractor's geometric eps (grid-snap residue) is its own EPS; we use
    # its default so the GDS-backed invariants are byte-comparable to the
    # placement-time invariants the sibling diffs against.
    return extract_invariants(view)


# --------------------------------------------------------------------------- #
# obligations measured on the FINAL GDS (the authoritative verify entry)       #
# --------------------------------------------------------------------------- #


def verify_obligations_on_gds(
    gds_path: str,
    intent: Any,
    device_labels: Mapping[str, Any],
    *,
    equal_size_groups: Sequence[Sequence[str]] = (),
    eps: int = EPS,
    matching_pair_axes: Optional[
        Mapping[Tuple[str, str], Tuple[str, float]]
    ] = None,
    device_orientations: Optional[Mapping[str, str]] = None,
    dropped_nonmos_matched: Sequence[str] = (),
) -> list:
    """Emit the structural LayoutObligations measured on the FINAL emitted GDS.

    The GDS-truth verify entry point: it builds a GDS-backed
    :class:`GeometricPlacement` (device bboxes = realized GDS geometry; cap
    footprints = their FuseTop plate) and runs the SAME geometric obligation checks
    (:func:`clay.layout.geometric_intent_checks.check_equal_size` +,
    where the GDS-backed placement makes them measurable, ``check_common_centroid``
    and ``check_place_group_realized``) against it, returning one
    :class:`~clay.layout.intent.LayoutObligation` per declared structural intent.

    Parameters
    ----------
    gds_path : str
        The emitted ``layout.gds``.
    intent : LayoutIntent | None
        The declared layout intent (carries ``place_groups`` for the place-group
        check). May be ``None`` when only @equal_size is verified.
    device_labels : Mapping[str, PlacedDevice]
        Materialized ``bbox`` + owned-polygon provenance (see
        :func:`geometric_placement_from_gds`). A raw bbox cannot attribute device
        artwork and therefore yields unmeasurable geometry.
    equal_size_groups : sequence of label sequences
        The declared ``@equal_size`` groups (each a sequence of >= 2 member labels).
        Typically ``placement.constraints.equal_size_groups``. Each yields one
        ``equal_size`` obligation measured on the GDS footprints.
    eps : int
        Grid-snap slack (dbu).

    Returns
    -------
    list[LayoutObligation]
        One obligation per declared structural intent, with ``measured_imbalance``
        computed on the FINAL GDS. An @equal_size group whose emitted footprints
        DIFFER (e.g. two caps of different value -> different FuseTop plates) FAILS
        (``measured_imbalance`` over ``rel_tol``); a group whose plates are equal
        PASSES. A member with no realized GDS geometry is UNMEASURABLE (blocking).

    Soundness note: the placement-time obligation path is NOT removed (the sibling
    placement provider stays the intermediate measurement); this ADDS the GDS-truth
    path, which is the authoritative one for realization soundness.
    """
    from clay.layout.geometric_intent_checks import (
        check_common_centroid,
        check_equal_size,
        check_matching,
        check_place_group_realized,
        _DEFAULT_CENTROID_TOL,
        _DEFAULT_EQUAL_SIZE_TOL,
        _DEFAULT_PLACE_GROUP_TOL,
    )
    from clay.layout.intent import LayoutObligation
    from clay.core_contract import VerificationStatus

    gp = geometric_placement_from_gds(
        gds_path,
        device_labels=device_labels,
        intent=intent,
        eps=eps,
        matching_pair_axes=matching_pair_axes,
        device_orientations=device_orientations,
        dropped_nonmos_matched=dropped_nonmos_matched,
    )

    obligations: list = []

    # ── direct matching: one obligation per canonical unordered pair, retaining
    #    every directed declaration ID.  No place-group row may substitute.
    for matching in tuple(getattr(intent, "matching_intents", ()) or ()):
        verdict = check_matching(
            gp,
            matching.members,
            rel_tol=_DEFAULT_CENTROID_TOL,
            require_interdigitation=matching.requires_interdigitation,
        )
        obligations.append(
            LayoutObligation(
                kind="matching",
                pair=matching.members,
                measured_imbalance=verdict.measured,
                rel_tol=_DEFAULT_CENTROID_TOL,
                missing_nets=verdict.missing,
                verification_status=VerificationStatus(verdict.status),
                declaration_ids=matching.declaration_ids,
                measurements={
                    key: float(value)
                    for key, value in (
                        ("centroid_axis_residual", verdict.centroid_axis_residual),
                        ("orientation_imbalance", verdict.orientation_imbalance),
                        ("footprint_imbalance", verdict.footprint_imbalance),
                    )
                    if value is not None
                },
            )
        )

    # ── equal_size (the soundness centrepiece): each declared @equal_size group's
    #    members must realise an IDENTICAL emitted footprint. Measured on the GDS
    #    plate, so two caps of different VALUE (different FuseTop plate) FAIL even
    #    though the placer reserved equal footprints.
    for grp in equal_size_groups:
        members = tuple(str(m) for m in grp)
        if len(members) < 2:
            continue
        measured, missing = check_equal_size(
            gp, members, rel_tol=_DEFAULT_EQUAL_SIZE_TOL
        )
        obligations.append(
            LayoutObligation(
                kind="equal_size",
                pair=(members[0], members[1]),
                measured_imbalance=measured,
                rel_tol=_DEFAULT_EQUAL_SIZE_TOL,
                missing_nets=missing,
            )
        )

    # ── place_group realisation: membership and an exact ``row=N`` / ``col=N``
    #    grid order must hold on the REALIZED GDS centroids. ``rows=A..B`` is a
    #    group-local packing-count range, not a global ordinal row.
    #    Measurable from device_bboxes alone (no cluster provenance needed), so it
    #    runs on the GDS-backed placement.
    from clay.layout.intent import place_group_floorplan_row

    place_groups = tuple(getattr(intent, "place_groups", ()) or ()) if intent else ()
    structural_groups = [
        g
        for g in place_groups
        if place_group_floorplan_row(g) is not None
        or getattr(g, "col", None) is not None
    ]
    all_group_tuples = [
        (tuple(g.devices), place_group_floorplan_row(g), g.col)
        for g in structural_groups
    ]
    for g in structural_groups:
        gdevs = tuple(g.devices)
        if len(gdevs) >= 2:
            pair = (gdevs[0], gdevs[1])
        elif len(gdevs) == 1:
            pair = (gdevs[0], str(g.group))
        else:
            pair = (str(g.group), f"row{g.row_min}_col{g.col}")
        measured, missing = check_place_group_realized(
            gp,
            tuple(g.devices),
            all_groups=all_group_tuples,
            row=place_group_floorplan_row(g),
            col=g.col,
        )
        obligations.append(
            LayoutObligation(
                kind="place_group_realized",
                pair=pair,
                measured_imbalance=measured,
                rel_tol=_DEFAULT_PLACE_GROUP_TOL,
                missing_nets=missing,
            )
        )

    # ── common_centroid for the declared matched pairs (when the intent declares
    #    them): a DIRECT centroid-coincidence on the REALIZED GDS bboxes. The flat
    #    GDS carries no cluster provenance, so a FUSED interdigitated pair (which
    #    resolves to one shared bbox in the placement provider) is not reconstructable
    #    here; we measure only pairs whose BOTH members realized a distinct GDS bbox.
    declared_pairs = _declared_symmetry_pairs(intent)
    direct_matching_pairs = {
        frozenset(str(member) for member in matching.members)
        for matching in tuple(getattr(intent, "matching_intents", ()) or ())
    }
    for a, b in declared_pairs:
        # A direct @matching pair is a mirror/equal-footprint obligation, not a
        # coincident-centroid obligation.  Counting it again here would make every
        # correctly separated matched macro fail the unrelated CC predicate.
        if frozenset((str(a), str(b))) in direct_matching_pairs:
            continue
        if a in gp.device_bboxes and b in gp.device_bboxes:
            measured, missing = check_common_centroid(gp, (a, b))
            obligations.append(
                LayoutObligation(
                    kind="common_centroid",
                    pair=(a, b),
                    measured_imbalance=measured,
                    rel_tol=_DEFAULT_CENTROID_TOL,
                    missing_nets=missing,
                )
            )

    return obligations


__all__ = [
    "GdsAnchorObligation",
    "GdsBandObligation",
    "GdsChannelObligation",
    "GdsCommonCentroidObligation",
    "GdsDensityWhitespaceObligation",
    "GdsFunctionalOdFillObligation",
    "GdsFingerArrayOrderObligation",
    "GdsGuardRingObligation",
    "GdsInterdigitationObligation",
    "GdsPassiveBankPartitionObligation",
    "GdsRailAffinityObligation",
    "GdsWellIslandShareObligation",
    "MIM_PLATE_LAYER",
    "geometric_placement_from_gds",
    "extract_invariants_from_gds",
    "verify_obligations_on_gds",
    "verify_bands_on_gds",
    "verify_channels_on_gds",
    "verify_common_centroid_interdigitation_on_gds",
    "verify_density_whitespace_on_gds",
    "verify_functional_od_fill_on_gds",
    "verify_finger_array_orders_on_gds",
    "verify_guard_rings_on_gds",
    "verify_passive_bank_partitions_on_gds",
    "verify_rail_affinities_on_gds",
    "verify_well_island_sharing_on_gds",
]
