"""Artifact-bound structural measurements for canonical route signoff.

The measurements in this module are deliberately post-emission.  They reread the
exact GDS certified by :class:`ExactEmittedArtifact`; no placement bbox, router
shape, or pre-route density proxy is accepted as physical evidence.  The stored
row program is used only as the typed structural obligation source.  A dummy
end-cap is counted as realized only when its rigid row occurrence is present in
the GDS and local poly/M1 plus non-router-owned contact/OD connectivity proves
that the dummy gate is tied to its adjacent diffusion.  Metal is a shared
physical surface after GDS merge, so a legal route landing may overlap device
metal without erasing the device artwork from the measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional

import klayout.db as kdb

from clay.layout.placed_geometry import placed_point_nm


_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_COORDINATE_CUTS = 768


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _SHA256_HEX for char in value)
    ):
        raise ValueError(f"{field} must be one canonical sha256")
    return value


def _bbox(value: object, field: str) -> Optional[tuple[int, int, int, int]]:
    if value is None:
        return None
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(item) is not int for item in value)
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise ValueError(f"{field} must be one positive integer bbox or None")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DummyEndcapObservation:
    """One declared dummy end-cap checked against emitted local conductors."""

    status: str
    entry_id: str
    declaration_ref: str
    side: str
    ordinal: int
    gate_bbox_nm: Optional[tuple[int, int, int, int]] = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "unmeasurable"}:
            raise ValueError("dummy endcap status must be pass/fail/unmeasurable")
        for field in ("entry_id", "declaration_ref"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be a non-empty exact string")
        if self.side not in {"left", "right"}:
            raise ValueError("dummy endcap side must be left/right")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("dummy endcap ordinal must be a non-negative integer")
        object.__setattr__(
            self, "gate_bbox_nm", _bbox(self.gate_bbox_nm, "gate_bbox_nm")
        )
        if type(self.reason) is not str:
            raise TypeError("dummy endcap reason must be an exact string")


@dataclass(frozen=True, slots=True, kw_only=True, order=True)
class StructureGateOwner:
    """One exact placed gate and its declared device owner."""

    owner: str
    gate_bbox_nm: tuple[int, int, int, int]
    channel_polygon_nm: object = None
    occurrence_index: Optional[int] = None
    channel_ordinal: Optional[int] = None

    def __post_init__(self) -> None:
        if type(self.owner) is not str or not self.owner:
            raise ValueError("gate owner must be a non-empty exact string")
        object.__setattr__(self, "gate_bbox_nm", _bbox(self.gate_bbox_nm, "gate_bbox_nm"))
        if self.channel_polygon_nm is None:
            if self.occurrence_index is not None or self.channel_ordinal is not None:
                raise ValueError("physical channel indices require an exact channel polygon")
        else:
            from clay.layout.occurrence_channels import ChannelPolygonNM

            if type(self.channel_polygon_nm) is not ChannelPolygonNM:
                raise TypeError("structure gate requires an exact physical channel polygon")
            self.channel_polygon_nm.payload()
            if self.gate_bbox_nm != self.channel_polygon_nm.bbox_nm:
                raise ValueError("physical channel bbox must be derived from its exact polygon")
            for name in ("occurrence_index", "channel_ordinal"):
                value = getattr(self, name)
                if type(value) is not int or value < 0:
                    raise ValueError("physical channel indices must be nonnegative exact integers")


@dataclass(frozen=True, slots=True, kw_only=True)
class WellMergeCertificate:
    """Independent authorization for one deterministic emitted well merge."""

    layer_key: tuple[int, int]
    input_region_sha256: str
    output_region_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.layer_key) is not tuple or len(self.layer_key) != 2
            or any(type(value) is not int for value in self.layer_key)
        ):
            raise TypeError("well merge layer_key must be an exact layer tuple")
        object.__setattr__(
            self, "input_region_sha256",
            _digest(self.input_region_sha256, "input_region_sha256"),
        )
        object.__setattr__(
            self, "output_region_sha256",
            _digest(self.output_region_sha256, "output_region_sha256"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            (self.layer_key, self.input_region_sha256, self.output_region_sha256),
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactStructureExpectation:
    """Signed, design-derived structural values the exact GDS must reproduce."""

    core_bbox_nm: tuple[int, int, int, int]
    core_area_um2: float
    largest_internal_void_um2: float
    shared_od_fraction: float
    gate_owners: tuple[StructureGateOwner, ...]
    dummy_obligations: tuple[tuple[str, str, str, int], ...]
    mask_layer_sha256s: tuple[tuple[tuple[int, int], str], ...]
    row_program_sha256: str
    row_final_certificate_sha256: str
    final_placed_artwork_sha256: str
    well_merge_input_sha256s: tuple[tuple[tuple[int, int], str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_bbox_nm", _bbox(self.core_bbox_nm, "core_bbox_nm"))
        for name in ("core_area_um2", "largest_internal_void_um2", "shared_od_fraction"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative float")
        if self.shared_od_fraction > 1.0:
            raise ValueError("shared_od_fraction must be <= 1")
        if type(self.gate_owners) is not tuple or any(
            type(item) is not StructureGateOwner for item in self.gate_owners
        ):
            raise TypeError("gate_owners must be an exact typed tuple")
        if type(self.dummy_obligations) is not tuple:
            raise TypeError("dummy_obligations must be an exact tuple")
        for item in self.dummy_obligations:
            if (
                type(item) is not tuple or len(item) != 4
                or any(type(value) is not str for value in item[:3])
                or type(item[3]) is not int or item[3] < 0
            ):
                raise TypeError("dummy obligation must be (entry, ref, side, ordinal)")
        if type(self.mask_layer_sha256s) is not tuple:
            raise TypeError("mask_layer_sha256s must be an exact tuple")
        rebuilt = []
        for key, digest in self.mask_layer_sha256s:
            if (
                type(key) is not tuple or len(key) != 2
                or any(type(value) is not int for value in key)
            ):
                raise TypeError("mask layer sha key must be exact")
            rebuilt.append((key, _digest(digest, "mask layer sha256")))
        if tuple(sorted(rebuilt)) != self.mask_layer_sha256s:
            raise ValueError("mask_layer_sha256s must be canonical and unique")
        if type(self.well_merge_input_sha256s) is not tuple:
            raise TypeError("well_merge_input_sha256s must be an exact tuple")
        rebuilt_wells = []
        for key, digest in self.well_merge_input_sha256s:
            if (
                type(key) is not tuple or len(key) != 2
                or any(type(value) is not int for value in key)
            ):
                raise TypeError("well merge input layer key must be exact")
            rebuilt_wells.append((key, _digest(digest, "well merge input sha256")))
        if tuple(sorted(rebuilt_wells)) != self.well_merge_input_sha256s:
            raise ValueError("well_merge_input_sha256s must be canonical and unique")
        for name in (
            "row_program_sha256", "row_final_certificate_sha256",
            "final_placed_artwork_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def payload(self) -> dict[str, Any]:
        payload = {
            "schema": "clay.exact-structure-expectation.v1",
            "core_bbox_nm": self.core_bbox_nm,
            "core_area_um2": self.core_area_um2,
            "largest_internal_void_um2": self.largest_internal_void_um2,
            "shared_od_fraction": self.shared_od_fraction,
            "gate_owners": tuple((item.owner, item.gate_bbox_nm) for item in self.gate_owners),
            "dummy_obligations": self.dummy_obligations,
            "mask_layer_sha256s": self.mask_layer_sha256s,
            "row_program_sha256": self.row_program_sha256,
            "row_final_certificate_sha256": self.row_final_certificate_sha256,
            "final_placed_artwork_sha256": self.final_placed_artwork_sha256,
            "well_merge_input_sha256s": self.well_merge_input_sha256s,
        }
        if any(item.channel_polygon_nm is not None for item in self.gate_owners):
            payload["schema"] = "clay.exact-structure-expectation.v2"
            payload["gate_channels"] = tuple(
                None if item.channel_polygon_nm is None else (
                    item.occurrence_index, item.channel_ordinal,
                    item.channel_polygon_nm.payload(),
                )
                for item in self.gate_owners
            )
        return payload

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ExactStructureProfileReport:
    """Exact-GDS structural profile bound to route and row certificates."""

    status: str
    core_bbox_nm: Optional[tuple[int, int, int, int]]
    core_area_um2: Optional[float]
    largest_internal_void_um2: Optional[float]
    shared_od_fraction: Optional[float]
    expected_dummy_endcaps: int
    realized_dummy_endcaps: int
    dummy_endcap_observations: tuple[DummyEndcapObservation, ...]
    artifact_sha256: str
    route_manifest_sha256: str
    row_program_sha256: str
    row_final_certificate_sha256: str
    expectation_sha256: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "unmeasurable"}:
            raise ValueError("structure profile status must be pass/fail/unmeasurable")
        object.__setattr__(self, "core_bbox_nm", _bbox(self.core_bbox_nm, "core_bbox_nm"))
        for field in (
            "core_area_um2",
            "largest_internal_void_um2",
            "shared_od_fraction",
        ):
            value = getattr(self, field)
            if value is not None and (
                type(value) is not float or not math.isfinite(value) or value < 0.0
            ):
                raise ValueError(f"{field} must be a finite non-negative float or None")
        if self.shared_od_fraction is not None and self.shared_od_fraction > 1.0:
            raise ValueError("shared_od_fraction must be <= 1")
        for field in ("expected_dummy_endcaps", "realized_dummy_endcaps"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        observations = self.dummy_endcap_observations
        if type(observations) is not tuple or any(
            type(item) is not DummyEndcapObservation for item in observations
        ):
            raise TypeError("dummy_endcap_observations must be exact typed tuples")
        if len(observations) != self.expected_dummy_endcaps:
            raise ValueError("dummy observation count must equal expected endcaps")
        if sum(item.status == "pass" for item in observations) != self.realized_dummy_endcaps:
            raise ValueError("realized endcap count must equal passing observations")
        for field in (
            "artifact_sha256",
            "route_manifest_sha256",
            "row_program_sha256",
            "row_final_certificate_sha256",
            "expectation_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if type(self.reason) is not str:
            raise TypeError("structure profile reason must be an exact string")


def _layer_key(tech: Any, name: str) -> Optional[tuple[int, int]]:
    try:
        value = tech.build_layer(name)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if value is None:
        return None
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"PDK {name} layer fact is not one exact layer key")
    return value


def _metal_keys(tech: Any) -> tuple[tuple[int, int], ...]:
    found: list[tuple[int, int]] = []
    for level in range(1, 33):
        try:
            key = tech.metal_key(level)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if type(key) is not tuple or len(key) != 2 or any(type(v) is not int for v in key):
            raise ValueError(f"PDK metal{level} layer fact is malformed")
        if key not in found:
            found.append(key)
    if not found:
        raise ValueError("PDK declares no measurable metal layers")
    return tuple(found)


def _device_mask_keys(tech: Any) -> tuple[tuple[int, int], ...]:
    """PDK-declared non-routing masks that define the device core."""
    keys = []
    for name in (
        "ndiff", "pdiff", "diff", "poly", "contact", "nplus", "pplus",
        "nwell", "pwell", "dnwell",
    ):
        key = _layer_key(tech, name)
        if key is not None and key not in keys:
            keys.append(key)
    tech_id = str(getattr(tech, "pdk_id", getattr(tech, "name", "")) or "")
    try:
        from clay.layout.passive_cell import _resolve_mim_facts

        facts, _provenance = _resolve_mim_facts(tech_id)
        key = tuple(facts.top_plate_layer)
        if key not in keys:
            keys.append(key)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return tuple(keys)


def _route_regions(artifact: Any) -> dict[tuple[int, int], kdb.Region]:
    result: dict[tuple[int, int], kdb.Region] = {}
    for _net, key, bbox in artifact.route_manifest.entries:
        result.setdefault(tuple(key), kdb.Region()).insert(kdb.Box(*bbox))
    for region in result.values():
        region.merge()
    return result


def _box_tuple(box: kdb.Box) -> tuple[int, int, int, int]:
    return (int(box.left), int(box.bottom), int(box.right), int(box.top))


def _canonical_contour(points: Iterable[Any]) -> tuple[tuple[int, int], ...]:
    ring = tuple((int(point.x), int(point.y)) for point in points)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if not ring:
        return ()
    forward = tuple(ring[index:] + ring[:index] for index in range(len(ring)))
    reverse_ring = tuple(reversed(ring))
    reverse = tuple(
        reverse_ring[index:] + reverse_ring[:index]
        for index in range(len(reverse_ring))
    )
    return min(forward + reverse)


def _region_sha256(region: kdb.Region) -> str:
    records = []
    for polygon in region.merged().each():
        outer = _canonical_contour(polygon.each_point_hull())
        if outer:
            holes = tuple(sorted(
                _canonical_contour(polygon.each_point_hole(index))
                for index in range(polygon.holes())
            ))
            records.append((outer, holes))
    records.sort()
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode("utf-8")).hexdigest()


def _region_union(regions: Iterable[kdb.Region]) -> kdb.Region:
    result = kdb.Region()
    for region in regions:
        result += region
    result.merge()
    return result


def _rectilinear_cuts(region: kdb.Region, bbox: tuple[int, int, int, int]):
    xs = {bbox[0], bbox[2]}
    ys = {bbox[1], bbox[3]}
    for polygon in region.each():
        contours = [polygon.each_point_hull()]
        contours.extend(
            polygon.each_point_hole(index) for index in range(polygon.holes())
        )
        for contour in contours:
            for point in contour:
                if bbox[0] <= point.x <= bbox[2]:
                    xs.add(int(point.x))
                if bbox[1] <= point.y <= bbox[3]:
                    ys.add(int(point.y))
    x_values = tuple(sorted(xs))
    y_values = tuple(sorted(ys))
    if len(x_values) > _MAX_COORDINATE_CUTS or len(y_values) > _MAX_COORDINATE_CUTS:
        raise ValueError(
            "exact metal coordinate decomposition exceeds bounded structure profile capacity"
        )
    return x_values, y_values


def _largest_empty_rectangle_um2(
    metal: kdb.Region, bbox: tuple[int, int, int, int]
) -> float:
    clip = kdb.Region(kdb.Box(*bbox))
    occupied = metal & clip
    if occupied.is_empty():
        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 1_000_000.0)
    xs, ys = _rectilinear_cuts(occupied, bbox)
    cols = len(xs) - 1
    rows = len(ys) - 1
    blocked: list[list[bool]] = []
    for row in range(rows):
        band: list[bool] = []
        for col in range(cols):
            probe = kdb.Region(kdb.Box(xs[col], ys[row], xs[col + 1], ys[row + 1]))
            band.append(not (occupied & probe).is_empty())
        blocked.append(band)
    best = 0
    for bottom in range(rows):
        free = [True] * cols
        for top in range(bottom, rows):
            for col in range(cols):
                free[col] = free[col] and not blocked[top][col]
            run_start: Optional[int] = None
            for col in range(cols + 1):
                if col < cols and free[col]:
                    if run_start is None:
                        run_start = col
                    continue
                if run_start is not None:
                    area = (xs[col] - xs[run_start]) * (ys[top + 1] - ys[bottom])
                    best = max(best, area)
                    run_start = None
    return float(best / 1_000_000.0)


def _x_projection_count(region: kdb.Region) -> int:
    intervals: list[tuple[int, int]] = []
    for polygon in region.each():
        box = polygon.bbox()
        if box.right > box.left:
            intervals.append((int(box.left), int(box.right)))
    if not intervals:
        return 0
    intervals.sort()
    count = 0
    end: Optional[int] = None
    for left, right in intervals:
        if end is None or left > end:
            count += 1
            end = right
        else:
            end = max(end, right)
    return count


def _shared_od_fraction(od: kdb.Region, poly: kdb.Region) -> float:
    total = 0
    shared = 0
    for island in od.each():
        crossings = _x_projection_count(poly & kdb.Region(island))
        total += crossings
        if crossings >= 2:
            shared += crossings
    return float(shared / total) if total else 0.0


def _owner_shared_od_fraction(
    od: kdb.Region,
    poly: kdb.Region,
    gate_owners: tuple[StructureGateOwner, ...],
    *,
    dummy_gate_bboxes_nm: Optional[tuple[tuple[int, int, int, int], ...]] = None,
) -> float:
    """Fraction of gates whose emitted OD island has >=2 declared owners."""
    islands = tuple(kdb.Region(item) for item in od.each())
    exact_channels = None
    used_channels: set[int] = set()
    if dummy_gate_bboxes_nm is not None or any(gate.channel_polygon_nm is not None for gate in gate_owners):
        exact_channels = tuple(kdb.Region(item) for item in (od & poly).merged().each())
    assignments: list[int] = []
    for gate in gate_owners:
        gate.__post_init__()
        if gate.channel_polygon_nm is None:
            probe = kdb.Region(kdb.Box(*gate.gate_bbox_nm))
            crossing = probe & poly & od
        else:
            crossing = gate.channel_polygon_nm.to_region()
        if exact_channels is not None:
            channel_matches = [
                index for index, actual in enumerate(exact_channels)
                if not (actual & crossing).is_empty()
            ]
            if len(channel_matches) != 1 or channel_matches[0] in used_channels:
                raise ValueError("structure gate lacks one uniquely owned physical channel")
            channel_index = channel_matches[0]
            if gate.channel_polygon_nm is not None and not (
                crossing ^ exact_channels[channel_index]
            ).is_empty():
                raise ValueError("structure channel differs from exact OD/poly crossing")
            used_channels.add(channel_index)
        matches = [index for index, island in enumerate(islands) if not (island & crossing).is_empty()]
        if len(matches) != 1:
            raise ValueError(f"gate {gate.owner!r} lacks one exact emitted OD/poly island")
        assignments.append(matches[0])
    if dummy_gate_bboxes_nm is not None:
        for box in dummy_gate_bboxes_nm:
            if (type(box) is not tuple or len(box) != 4
                    or any(type(value) is not int for value in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("dummy gate lacks its independently observed channel probe")
            probe = kdb.Region(kdb.Box(*box))
            matches = [index for index, channel in enumerate(exact_channels)
                       if not (channel & probe).is_empty()]
            if len(matches) != 1 or matches[0] in used_channels:
                raise ValueError("active and dummy gates do not uniquely cover the physical channel population")
            used_channels.add(matches[0])
        if len(used_channels) != len(exact_channels):
            raise ValueError("gate records omit part of the exact physical channel population")
    owners_by_island: dict[int, set[str]] = {}
    for gate, island in zip(gate_owners, assignments):
        owners_by_island.setdefault(island, set()).add(gate.owner)
    shared = sum(
        len(owners_by_island[island]) >= 2 for island in assignments
    )
    return float(shared / len(assignments)) if assignments else 0.0


def _placed_geometry_regions(
    geometry: Any,
    placement_um: Any,
    keys: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], kdb.Region]:
    result = {key: kdb.Region() for key in keys}
    for name, stamp in sorted((geometry or {}).items()):
        # Guard rings are device-mask artwork and therefore part of the exact
        # structural envelope.  Power rails remain BEOL-only and cannot affect a
        # device-mask projection.
        if str(getattr(stamp, "source", "")) in ("power_rail", "power_plane"):
            continue
        position = (placement_um or {}).get(name)
        if position is None:
            continue
        for key, rings in (getattr(stamp, "polys_by_layer", {}) or {}).items():
            exact_key = tuple(key)
            if exact_key not in result:
                continue
            for ring in rings:
                points = [
                    kdb.Point(*placed_point_nm(float(x), float(y), position))
                    for x, y in ring
                ]
                if len(points) >= 3:
                    result[exact_key].insert(kdb.Polygon(points))
    for region in result.values():
        region.merge()
    return result


def _planned_well_union(
    geometry: Any,
    placement_um: Any,
    well_key: tuple[int, int],
    tech: Any,
    devices: Any,
    source_cell: Any = None,
) -> tuple[kdb.Region, kdb.Region]:
    """Replay the emitter's deterministic per-bulk-net well union plan."""
    from clay.layout.route.lvs_gate import _source_well_assembly

    source_intent = None
    if source_cell is not None:
        from clay.layout.intent import layout_intent_from_cell_def

        source_intent = layout_intent_from_cell_def(source_cell)
    target_key = (int(well_key[0]), int(well_key[1]))
    _keys, assembly = _source_well_assembly(
        source_intent,
        tech,
        geometry or {},
        placement_um or {},
        devices=devices,
        source_cell=source_cell,
        _explicit_well_key=target_key,
    )
    if target_key not in assembly:
        # A declared P-well domain (or a placement without well polygons on
        # target_key) legitimately has no N-well artwork; owner checks ran above.
        return kdb.Region(), kdb.Region()
    before, emitted = assembly[target_key]
    return before, _region_union(emitted)


def _translated_geometry_regions(
    realization: Any, dx: int, dy: int
) -> dict[tuple[int, int], kdb.Region]:
    result: dict[tuple[int, int], kdb.Region] = {}
    for shape in tuple(getattr(realization.geometry, "polygons", ()) or ()):
        key = (int(shape.layer), int(shape.datatype))
        points = [
            kdb.Point(int(point.x) + dx, int(point.y) + dy)
            for point in shape.polygon.vertices
        ]
        if len(points) < 3:
            continue
        result.setdefault(key, kdb.Region()).insert(kdb.Polygon(points))
    for region in result.values():
        region.merge()
    return result


def _local_finger_gate_bbox(
    finger: Any, comp_bbox: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Return one positive exact-nm gate-owner probe in the macro frame.

    Historical vertical gates are represented by ``x0/x1/cy``. Quarter-turned
    rigid macros carry the exact rotated two-dimensional probe explicitly; no
    consumer may collapse it back onto the vertical-gate convention.
    """

    stored = getattr(finger, "gate_bbox_nm", None)
    if stored is not None:
        if (
            type(stored) is not tuple
            or len(stored) != 4
            or any(type(value) is not int for value in stored)
            or stored[2] <= stored[0]
            or stored[3] <= stored[1]
        ):
            raise ValueError("stored finger gate bbox must be one positive exact bbox")
        return stored
    x0, x1 = int(finger.x0), int(finger.x1)
    raw_cy = getattr(finger, "cy", None)
    if raw_cy is None:
        y0, y1 = int(comp_bbox[1]), int(comp_bbox[3])
    else:
        y0, y1 = int(raw_cy) - 1, int(raw_cy) + 1
    if x1 <= x0 or y1 <= y0:
        raise ValueError("legacy finger gate probe is not one positive exact bbox")
    return (x0, y0, x1, y1)


def _row_structure_gate_records(*, row_program, row_materialization,
                                row_final_certificate, od_keys, poly_key):
    """Replay the complete signed row occurrence/channel ownership domain."""
    realizations = {item.entry_id: item for item in row_materialization.realizations}
    bindings = {item.entry_id: item for item in row_final_certificate.macro_bindings}
    gate_records: list[StructureGateOwner] = []
    for entry in row_program.entries:
        realization = realizations[entry.entry_id]
        binding = bindings[entry.entry_id]
        dx = binding.bbox_nm[0] - realization.local_bbox_nm[0]
        dy = binding.bbox_nm[1] - realization.local_bbox_nm[1]
        if getattr(realization.geometry, "channel_inventory", None) is not None:
            from clay.layout.occurrence_channels import replay_row_channel_inventory

            inventory = replay_row_channel_inventory(
                geometry=realization.geometry, consumed_devices=entry.consumed_devices,
            )
            if (inventory.od_layer_keys != tuple(sorted(set(od_keys)))
                    or inventory.poly_layer_key != poly_key):
                raise ValueError("row channel inventory differs from current PDK OD/poly keys")
            if entry.macro.consumed_dummy_endcaps:
                raise ValueError("resolved occurrence channels lack a complete dummy gate tie proof")
            for occurrence in inventory.occurrences:
                if occurrence.logical_owner not in set(entry.active_devices):
                    continue
                for ordinal, channel in enumerate(occurrence.channels):
                    placed_channel = channel.translated(dx, dy)
                    gate_records.append(StructureGateOwner(
                        owner=occurrence.logical_owner,
                        gate_bbox_nm=placed_channel.bbox_nm,
                        channel_polygon_nm=placed_channel,
                        occurrence_index=occurrence.occurrence_index,
                        channel_ordinal=ordinal,
                    ))
            continue
        if getattr(realization.geometry, "fixed_pcell", None) is not None:
            raise ValueError("fixed row occurrence lacks its physical channel inventory")
        comp = tuple(getattr(realization.geometry, "comp_bbox"))
        for finger in tuple(getattr(realization.geometry, "fingers", ()) or ()):
            owner = str(getattr(finger, "member", "") or "")
            if owner not in set(entry.active_devices):
                continue
            gate = _local_finger_gate_bbox(finger, comp)
            gate_records.append(StructureGateOwner(
                owner=owner,
                gate_bbox_nm=(
                    gate[0] + dx,
                    gate[1] + dy,
                    gate[2] + dx,
                    gate[3] + dy,
                ),
            ))
    return tuple(sorted(gate_records))


def _verify_free_gate_ownership(*, gates, row_owners, final_certificate,
                                od_keys, poly_key):
    """Join the non-row denominator to the retained physical source owners."""
    free_gates = tuple(gate for gate in gates if gate.owner not in row_owners)
    free_blocks = dict(final_certificate.free_device_blocks)
    native = final_certificate.native_geometry_binding
    if native is None:
        # Legacy certificates require disjoint block envelopes, so a complete
        # channel inside its declared source block has unambiguous ownership.
        final_certificate.__post_init__()
        for gate in free_gates:
            box = free_blocks.get(gate.owner)
            inner = gate.gate_bbox_nm
            if box is None or not (box[0] <= inner[0] < inner[2] <= box[2]
                                   and box[1] <= inner[1] < inner[3] <= box[3]):
                raise ValueError("free gate is outside its certified source owner block")
        return

    # Native envelopes can overlap legally. Replay actual source-owned masks;
    # a bounding-box containment lookup would invent ownership in those cases.
    by_owner = {}
    for _name, members, layers in native.replayed_stamp_artwork_nm():
        if set(members) <= row_owners:
            continue
        masks = {}
        for key, rings in layers:
            if key not in od_keys and key != poly_key:
                continue
            region = masks.setdefault(key, kdb.Region())
            for ring in rings:
                region.insert(kdb.Polygon([kdb.Point(x, y) for x, y in ring]))
        local_od = _region_union(masks.get(key, kdb.Region()) for key in od_keys)
        crossings = local_od & masks.get(poly_key, kdb.Region())
        crossings.merge()
        if crossings.is_empty():
            continue
        if len(members) != 1 or members[0] not in free_blocks:
            raise ValueError("free gate artwork lacks one certified physical source owner")
        by_owner.setdefault(members[0], kdb.Region()).insert(crossings)
    replayed = []
    for owner, crossings in by_owner.items():
        crossings.merge()
        replayed.extend((owner, _box_tuple(polygon.bbox())) for polygon in crossings.each())
    actual = sorted((gate.owner, gate.gate_bbox_nm) for gate in free_gates)
    if actual != sorted(replayed):
        raise ValueError("free gate records differ from complete native source-owned channels")


def compile_exact_structure_expectation(
    tech: Any,
    *,
    geometry: Any,
    placement_um: Any,
    row_program: Any,
    row_materialization: Any,
    row_final_certificate: Any,
    final_placement: Any = None,
    well_merge_devices: Any = (),
    source_cell: Any = None,
) -> ExactStructureExpectation:
    """Compile signed structural expectations from the final placed master."""
    from clay.layout.generic_placement import (
        RowFinalPlacementCertificate,
        _placed_artwork_sha256,
    )
    from clay.layout.place.row_program import (
        RowProgram,
        RowProgramMaterialization,
        verify_row_program_materialization,
    )

    if type(row_program) is not RowProgram:
        raise TypeError("structure expectation requires exact RowProgram")
    if type(row_materialization) is not RowProgramMaterialization:
        raise TypeError("structure expectation requires exact RowProgramMaterialization")
    if type(row_final_certificate) is not RowFinalPlacementCertificate:
        raise TypeError("structure expectation requires exact RowFinalPlacementCertificate")
    verify_row_program_materialization(
        row_materialization, program=row_program, tech_id=row_program.tech_id
    )
    if row_final_certificate.program_sha256 != row_program.sha256:
        raise ValueError("row final certificate does not bind expectation program")
    mask_keys = _device_mask_keys(tech)
    od_keys = tuple(key for key in (
        _layer_key(tech, "ndiff"), _layer_key(tech, "pdiff"), _layer_key(tech, "diff")
    ) if key is not None)
    poly_key = _layer_key(tech, "poly")
    if not od_keys or poly_key is None:
        raise ValueError("PDK lacks OD/poly expectation facts")
    regions = _placed_geometry_regions(geometry, placement_um, mask_keys)
    well_merge_inputs: tuple[tuple[tuple[int, int], str], ...] = ()
    nwell_key = _layer_key(tech, "nwell")
    if nwell_key is not None and nwell_key in regions:
        before_well, after_well = _planned_well_union(
            geometry,
            placement_um,
            nwell_key,
            tech,
            well_merge_devices,
            source_cell,
        )
        if not before_well.is_empty():
            regions[nwell_key] = after_well
            well_merge_inputs = ((nwell_key, _region_sha256(before_well)),)
    masks = _region_union(regions.values())
    od = _region_union(regions[key] for key in dict.fromkeys(od_keys))
    poly = regions[poly_key]
    if masks.is_empty() or od.is_empty() or poly.is_empty():
        raise ValueError("final placed master lacks measurable device masks")
    core_bbox = _box_tuple(masks.bbox())
    core_area = float(
        (core_bbox[2] - core_bbox[0]) * (core_bbox[3] - core_bbox[1]) / 1_000_000.0
    )
    largest_void = _largest_empty_rectangle_um2(masks, core_bbox)

    gate_records = list(_row_structure_gate_records(
        row_program=row_program, row_materialization=row_materialization,
        row_final_certificate=row_final_certificate, od_keys=od_keys, poly_key=poly_key))
    row_groups = {entry.group for entry in row_program.entries}
    # Free MOS stamps have one geometry owner.  Their exact OD/poly crossings add
    # denominator gates but can never manufacture cross-owner sharing by themselves.
    for name, stamp in sorted((geometry or {}).items()):
        if name in row_groups or str(getattr(stamp, "source", "")) in {
            "group_guard_ring", "power_rail"
        }:
            continue
        position = (placement_um or {}).get(name)
        if position is None:
            continue
        local = _placed_geometry_regions({name: stamp}, {name: position}, mask_keys)
        local_od = _region_union(local[key] for key in dict.fromkeys(od_keys))
        local_cross = local_od & local[poly_key]
        for crossing in local_cross.each():
            box = crossing.bbox()
            gate_records.append(StructureGateOwner(
                owner=str(name), gate_bbox_nm=_box_tuple(box)
            ))
    gates = tuple(sorted(gate_records))
    shared = _owner_shared_od_fraction(od, poly, gates)
    dummy_obligations = tuple(
        (entry.entry_id, ref, side, ordinal)
        for entry in row_program.entries
        for side, ref, count in entry.macro.consumed_dummy_endcaps
        for ordinal in range(count)
    )
    placed_digest = row_final_certificate.placed_artwork_sha256
    if final_placement is not None:
        replayed = _placed_artwork_sha256(list(getattr(final_placement, "polys", ()) or ()))
        if replayed != placed_digest:
            raise ValueError("final placed artwork does not bind the row certificate")
        placed_digest = replayed
    return ExactStructureExpectation(
        core_bbox_nm=core_bbox,
        core_area_um2=core_area,
        largest_internal_void_um2=largest_void,
        shared_od_fraction=shared,
        gate_owners=gates,
        dummy_obligations=dummy_obligations,
        mask_layer_sha256s=tuple(sorted(
            (key, _region_sha256(regions[key])) for key in mask_keys
        )),
        row_program_sha256=row_program.sha256,
        row_final_certificate_sha256=row_final_certificate.sha256,
        final_placed_artwork_sha256=placed_digest,
        well_merge_input_sha256s=well_merge_inputs,
    )


def _component_touching(region: kdb.Region, probe: kdb.Region) -> Optional[kdb.Region]:
    matches = [kdb.Region(poly) for poly in region.each() if not (kdb.Region(poly) & probe).is_empty()]
    if len(matches) != 1:
        return None
    return matches[0]


def _verify_row_artwork_replay(
    *,
    program: Any,
    materialization: Any,
    final_certificate: Any,
    emitted: dict[tuple[int, int], kdb.Region],
    route_owned: dict[tuple[int, int], kdb.Region],
    non_route_artwork: dict[tuple[int, int], kdb.Region],
    shared_route_layers: frozenset[tuple[int, int]],
) -> None:
    """Replay every stored row layer against exact GDS.

    Routing metals are shared physical conductor surfaces: once GDS polygons are
    merged, an overlapping terminal landing has no separable software owner.  The
    signed row metal therefore has to be present in the emitted union.  Device
    masks and cuts remain non-route artwork, so route-manifest shapes can never
    substitute for missing OD/poly/contact/implant/well realization.
    """
    from clay.layout.generic_placement import _placed_artwork_sha256, _translate

    entries = {item.entry_id: item for item in program.entries}
    realizations = {item.entry_id: item for item in materialization.realizations}
    bindings = {item.entry_id: item for item in final_certificate.macro_bindings}
    expected_ids = set(entries)
    if set(realizations) != expected_ids or set(bindings) != expected_ids:
        raise ValueError("row program/materialization/final certificate entry sets differ")
    for entry_id in sorted(entries):
        entry = entries[entry_id]
        realization = realizations[entry_id]
        binding = bindings[entry_id]
        if (
            binding.entry_sha256 != entry.sha256
            or binding.owner != entry.owner
            or binding.consumed_devices != entry.consumed_devices
            or binding.materialization_sha256 != materialization.certificate.sha256
        ):
            raise ValueError(f"row final certificate binding mismatch: {entry_id}")
        dx = binding.bbox_nm[0] - realization.local_bbox_nm[0]
        dy = binding.bbox_nm[1] - realization.local_bbox_nm[1]
        if binding.bbox_nm != tuple(
            value + shift
            for value, shift in zip(realization.local_bbox_nm, (dx, dy, dx, dy))
        ):
            raise ValueError(f"row final bbox is not one rigid translation: {entry_id}")
        translated_polys = _translate(
            list(getattr(realization.geometry, "polygons", ()) or ()), dx, dy
        )
        if _placed_artwork_sha256(translated_polys) != binding.artwork_sha256:
            raise ValueError(f"row binding artwork sha mismatch: {entry_id}")
        for key, expected in _translated_geometry_regions(realization, dx, dy).items():
            actual = emitted.get(key)
            if actual is None:
                raise ValueError(
                    f"row macro is absent from exact GDS on layer {key}: {entry_id}"
                )
            replay_surface = actual
            if key not in shared_route_layers:
                replay_surface = non_route_artwork.get(key, actual)
            if not expected.is_empty() and not (expected - replay_surface).is_empty():
                raise ValueError(
                    "row macro non-route artwork is absent from exact GDS on "
                    f"layer {key}: {entry_id}"
                )


def _local_dummy_observations(
    *,
    program: Any,
    materialization: Any,
    final_certificate: Any,
    emitted: dict[tuple[int, int], kdb.Region],
    route_owned: dict[tuple[int, int], kdb.Region],
    od_key: tuple[int, int],
    poly_key: tuple[int, int],
    contact_key: tuple[int, int],
    m1_key: tuple[int, int],
) -> tuple[DummyEndcapObservation, ...]:
    from clay.layout.generic_placement import _placed_artwork_sha256, _translate

    entries = {item.entry_id: item for item in program.entries}
    realizations = {item.entry_id: item for item in materialization.realizations}
    bindings = {item.entry_id: item for item in final_certificate.macro_bindings}
    expected_ids = set(entries)
    if set(realizations) != expected_ids or set(bindings) != expected_ids:
        raise ValueError("row program/materialization/final certificate entry sets differ")
    observations: list[DummyEndcapObservation] = []
    local_m1 = emitted[m1_key]
    local_m1.merge()
    local_contact = emitted[contact_key] - route_owned.get(contact_key, kdb.Region())
    local_contact.merge()
    used_boundary_cuts: set[tuple[int, int, int, int]] = set()
    for entry_id in sorted(entries):
        entry = entries[entry_id]
        realization = realizations[entry_id]
        binding = bindings[entry_id]
        if (
            binding.entry_sha256 != entry.sha256
            or binding.owner != entry.owner
            or binding.consumed_devices != entry.consumed_devices
            or binding.materialization_sha256 != materialization.certificate.sha256
        ):
            raise ValueError(f"row final certificate binding mismatch: {entry_id}")
        dx = binding.bbox_nm[0] - realization.local_bbox_nm[0]
        dy = binding.bbox_nm[1] - realization.local_bbox_nm[1]
        if binding.bbox_nm != tuple(
            value + shift
            for value, shift in zip(realization.local_bbox_nm, (dx, dy, dx, dy))
        ):
            raise ValueError(f"row final bbox is not one rigid translation: {entry_id}")
        expected_regions = _translated_geometry_regions(realization, dx, dy)
        translated_polys = _translate(
            list(getattr(realization.geometry, "polygons", ()) or ()), dx, dy
        )
        if _placed_artwork_sha256(translated_polys) != binding.artwork_sha256:
            raise ValueError(f"row binding artwork sha mismatch: {entry_id}")
        for key in expected_regions:
            expected = expected_regions.get(key, kdb.Region())
            actual = emitted.get(key)
            if actual is None or (
                not expected.is_empty() and not (expected - actual).is_empty()
            ):
                raise ValueError(f"row macro is absent from exact GDS on layer {key}: {entry_id}")

        expanded_atoms = []
        for row in entry.macro.rows:
            for atom in row.atoms:
                expanded_atoms.extend((atom,) * (2 if ":pair:" in f":{atom.atom_id}:" else 1))
        fingers = tuple(getattr(realization.geometry, "fingers", ()) or ())
        if len(expanded_atoms) != len(fingers):
            raise ValueError(f"row atom/finger mapping is not one-to-one: {entry_id}")
        ordinal_by_ref: dict[str, int] = {}
        columns = tuple(getattr(realization.geometry, "columns", ()) or ())
        for atom_index, (atom, finger) in enumerate(zip(expanded_atoms, fingers)):
            ref = getattr(atom, "dummy_obligation_ref", None)
            if not ref:
                continue
            obligation = next(
                (item for item in entry.macro.dummy_endcap_obligations
                 if item.declaration_ref == ref),
                None,
            )
            if obligation is None:
                raise ValueError(f"dummy atom lacks one typed obligation: {atom.atom_id}")
            side = obligation.side.value
            ordinal = ordinal_by_ref.get(ref, 0)
            ordinal_by_ref[ref] = ordinal + 1
            comp = tuple(getattr(realization.geometry, "comp_bbox"))
            local_gate = _local_finger_gate_bbox(finger, comp)
            placed_gate = (
                local_gate[0] + dx,
                local_gate[1] + dy,
                local_gate[2] + dx,
                local_gate[3] + dy,
            )
            horizontal_gate = entry.orientation in ("R90", "R270")
            channel_x = (placed_gate[0] + placed_gate[2]) // 2
            channel_y = (placed_gate[1] + placed_gate[3]) // 2
            # A two-row macro may repeat the same gate x in both channels.  A
            # macro-height probe therefore touches two unrelated poly components.
            # Bind this obligation to the one realized OD row containing the
            # finger's certified channel centre and x span.
            row_boxes = []
            for row_region in expected_regions.get(od_key, kdb.Region()).each():
                box = row_region.bbox()
                if (
                    (
                        int(box.left) <= channel_x <= int(box.right)
                        and int(box.bottom) < placed_gate[3]
                        and placed_gate[1] < int(box.top)
                    )
                    if horizontal_gate
                    else (
                        int(box.bottom) <= channel_y <= int(box.top)
                        and int(box.left) < placed_gate[2]
                        and placed_gate[0] < int(box.right)
                    )
                ):
                    row_boxes.append(box)
            if len(row_boxes) != 1:
                raise ValueError("dummy finger lacks one exact realization row boundary")
            row_box = row_boxes[0]
            if horizontal_gate:
                gate_bbox = (
                    int(row_box.left),
                    placed_gate[1],
                    int(row_box.right),
                    placed_gate[3],
                )
            else:
                gate_bbox = (
                    placed_gate[0],
                    int(row_box.bottom),
                    placed_gate[2],
                    int(row_box.top),
                )
            gate_probe = kdb.Region(kdb.Box(*gate_bbox))
            poly_component = _component_touching(emitted[poly_key], gate_probe)
            reason = ""
            passed = False
            if poly_component is None:
                reason = "dummy gate has no unique emitted poly component"
            else:
                gate_cuts = local_contact & poly_component
                gate_m1_components = [
                    kdb.Region(component)
                    for component in local_m1.each()
                    if not (kdb.Region(component) & gate_cuts).is_empty()
                ]
                neighbor_index = atom_index + (1 if side == "left" else -1)
                if not 0 <= neighbor_index < len(expanded_atoms):
                    raise ValueError("dummy obligation has no immediately adjacent atom")
                neighbor_atom = expanded_atoms[neighbor_index]
                boundary = atom.right if side == "left" else atom.left
                peer_boundary = (
                    neighbor_atom.left if side == "left" else neighbor_atom.right
                )
                if not boundary.is_compatible_with(peer_boundary):
                    raise ValueError("dummy immediate diffusion boundaries are incompatible")
                neighbor_finger = fingers[neighbor_index]
                neighbor_gate = _local_finger_gate_bbox(neighbor_finger, comp)
                expected_x = (
                    local_gate[0] + local_gate[2]
                    + neighbor_gate[0] + neighbor_gate[2]
                ) // 4
                expected_y = (
                    local_gate[1] + local_gate[3]
                    + neighbor_gate[1] + neighbor_gate[3]
                ) // 4
                local_channel_x = (local_gate[0] + local_gate[2]) // 2
                local_channel_y = (local_gate[1] + local_gate[3]) // 2
                raw_column_cy = getattr(finger, "cy", None)
                column_matches = tuple(
                    column for column in columns
                    if (
                        (
                            int(column.x_center) == local_channel_x
                            and (
                                int(column.contact_bbox[1])
                                + int(column.contact_bbox[3])
                            ) // 2 == expected_y
                        )
                        if horizontal_gate
                        else (
                            int(column.x_center) == expected_x
                            and (
                                raw_column_cy is None
                                or (
                                    int(column.contact_bbox[1])
                                    + int(column.contact_bbox[3])
                                ) // 2 == int(raw_column_cy)
                            )
                        )
                    )
                    and str(getattr(column, "net", "")) == boundary.net
                )
                if len(column_matches) != 1:
                    raise ValueError(
                        "dummy immediate boundary lacks one exact materialized column: "
                        f"entry={entry_id} atom={atom.atom_id} "
                        f"orientation={entry.orientation} expected_xy="
                        f"({expected_x},{expected_y}) channel_xy="
                        f"({local_channel_x},{local_channel_y}) "
                        f"matches={len(column_matches)}"
                    )
                cut_bbox = tuple(
                    int(value) + shift
                    for value, shift in zip(
                        column_matches[0].contact_bbox, (dx, dy, dx, dy)
                    )
                )
                if cut_bbox in used_boundary_cuts:
                    raise ValueError("one diffusion boundary cut consumes multiple dummy obligations")
                used_boundary_cuts.add(cut_bbox)
                adjacent_cut = kdb.Region(kdb.Box(*cut_bbox))
                if not (adjacent_cut - local_contact).is_empty():
                    raise ValueError("dummy immediate boundary cut is absent from exact GDS")
                if not gate_m1_components:
                    reason = "dummy gate has no local poly-contact-M1 landing"
                else:
                    passed = any(
                        not (component & adjacent_cut).is_empty()
                        for component in gate_m1_components
                    )
                    if not passed:
                        reason = "local M1 does not tie dummy gate to adjacent diffusion"
            observations.append(DummyEndcapObservation(
                status="pass" if passed else "fail",
                entry_id=entry_id,
                declaration_ref=str(ref),
                side=side,
                ordinal=ordinal,
                gate_bbox_nm=gate_bbox,
                reason=reason,
            ))
    return tuple(observations)


def measure_exact_structure_profile(
    artifact: Any,
    tech: Any,
    *,
    row_program: Any,
    row_materialization: Any,
    row_final_certificate: Any,
    expectation: Any,
    gds_path: Optional[Path] = None,
) -> ExactStructureProfileReport:
    """Reread one exact artifact and return a certificate-bound profile.

    Structural absence or ambiguity is returned as ``unmeasurable`` rather than
    inferred from pre-route memory.  A uniquely mapped dummy whose local tie is
    absent is a measured ``fail``.
    """
    from clay.layout.generic_placement import RowFinalPlacementCertificate
    from clay.layout.place.row_program import (
        RowProgram,
        RowProgramMaterialization,
        verify_row_program_materialization,
    )
    from clay.layout.route.exact_artifact_evidence import ExactEmittedArtifact

    if type(artifact) is not ExactEmittedArtifact:
        raise TypeError("structure profile requires ExactEmittedArtifact")
    if type(row_program) is not RowProgram:
        raise TypeError("structure profile requires exact RowProgram")
    if type(row_materialization) is not RowProgramMaterialization:
        raise TypeError("structure profile requires exact RowProgramMaterialization")
    if type(row_final_certificate) is not RowFinalPlacementCertificate:
        raise TypeError("structure profile requires exact RowFinalPlacementCertificate")
    if type(expectation) is not ExactStructureExpectation:
        raise TypeError("structure profile requires exact ExactStructureExpectation")
    artifact.validate()
    artifact_hash = artifact.truth_manifest.exact_gds_sha256
    measurement_gds = Path(
        artifact.exact_gds_path if gds_path is None else gds_path
    ).resolve()
    from clay.layout.route.truth_manifest import (
        ArtifactDigestMismatch,
        sha256_path,
    )

    measurement_hash = sha256_path(measurement_gds)
    if measurement_hash != artifact_hash:
        raise ArtifactDigestMismatch(
            "structure profile GDS differs from the exact artifact: "
            f"expected {artifact_hash}, got {measurement_hash}"
        )
    manifest_hash = artifact.route_manifest.route_manifest_sha256
    row_hash = row_program.sha256
    final_hash = row_final_certificate.sha256
    expectation_hash = expectation.sha256
    expected_dummy = len(expectation.dummy_obligations)

    def unmeasurable(reason: str, observations=()) -> ExactStructureProfileReport:
        typed = tuple(observations)
        if len(typed) != expected_dummy:
            typed = tuple(
                DummyEndcapObservation(
                    status="unmeasurable",
                    entry_id=entry.entry_id,
                    declaration_ref=ref,
                    side=side,
                    ordinal=ordinal,
                    reason=reason,
                )
                for entry in row_program.entries
                for side, ref, count in entry.macro.consumed_dummy_endcaps
                for ordinal in range(count)
            )
        return ExactStructureProfileReport(
            status="unmeasurable",
            core_bbox_nm=None,
            core_area_um2=None,
            largest_internal_void_um2=None,
            shared_od_fraction=None,
            expected_dummy_endcaps=expected_dummy,
            realized_dummy_endcaps=sum(item.status == "pass" for item in typed),
            dummy_endcap_observations=typed,
            artifact_sha256=artifact_hash,
            route_manifest_sha256=manifest_hash,
            row_program_sha256=row_hash,
            row_final_certificate_sha256=final_hash,
            expectation_sha256=expectation_hash,
            reason=reason,
        )

    try:
        verify_row_program_materialization(
            row_materialization, program=row_program, tech_id=row_program.tech_id
        )
        if row_program.residuals or row_materialization.residuals:
            return unmeasurable("row program/materialization contains structural residuals")
        if row_final_certificate.program_sha256 != row_hash:
            return unmeasurable("row final certificate does not bind the row program")
        if (
            expectation.row_program_sha256 != row_hash
            or expectation.row_final_certificate_sha256 != final_hash
            or expectation.final_placed_artwork_sha256
            != row_final_certificate.placed_artwork_sha256
        ):
            return unmeasurable("structure expectation does not bind the final row master")
        actual_dummy_obligations = tuple(
            (entry.entry_id, ref, side, ordinal)
            for entry in row_program.entries
            for side, ref, count in entry.macro.consumed_dummy_endcaps
            for ordinal in range(count)
        )
        if actual_dummy_obligations != expectation.dummy_obligations:
            return unmeasurable("structure expectation dummy obligations differ from RowProgram")
        expected_materialization = (
            row_materialization.certificate.sha256 if row_program.entries else "0" * 64
        )
        if row_final_certificate.materialization_sha256 != expected_materialization:
            return unmeasurable("row final certificate does not bind the materialization")
        tech_id = str(getattr(tech, "pdk_id", getattr(tech, "name", "")) or "")
        if tech_id != row_program.tech_id:
            return unmeasurable("PDK identity differs from the row program")

        from clay.layout.completeness.gds_witness import ExactGdsWitness

        witness = ExactGdsWitness.open(
            measurement_gds, artifact.scope.top_cell, artifact
        )
        if not math.isclose(witness.dbu, 0.001, rel_tol=0.0, abs_tol=1e-12):
            return unmeasurable("exact structure kernel requires a 1nm GDS database unit")
        od_keys = tuple(dict.fromkeys(
            key for key in (
                _layer_key(tech, "ndiff"),
                _layer_key(tech, "pdiff"),
                _layer_key(tech, "diff"),
            ) if key is not None
        ))
        poly_key = _layer_key(tech, "poly")
        contact_key = _layer_key(tech, "contact")
        if contact_key is None:
            try:
                contact_key = tuple(tech.contact_key())
            except (AttributeError, KeyError, TypeError, ValueError):
                contact_key = None
        metal_keys = _metal_keys(tech)
        if not od_keys or poly_key is None or contact_key is None:
            return unmeasurable("PDK lacks exact OD/poly/contact structure layer facts")

        # The metric's denominator and ownership must replay from the retained
        # row master even when a re-signed expectation has the same mask hashes
        # and numerical result. In particular an empty zero-sharing subset is
        # not an alternative measurement of the signed physical population.
        row_gates = _row_structure_gate_records(
            row_program=row_program, row_materialization=row_materialization,
            row_final_certificate=row_final_certificate, od_keys=od_keys, poly_key=poly_key)
        row_owners = {owner for entry in row_program.entries for owner in entry.consumed_devices}
        expected_row_gates = tuple(gate for gate in expectation.gate_owners if gate.owner in row_owners)
        if expected_row_gates != row_gates:
            return unmeasurable("structure expectation gate records differ from complete row materialization")
        if any(gate.channel_polygon_nm is not None and gate.owner not in row_owners
               for gate in expectation.gate_owners):
            return unmeasurable("structure expectation has a foreign typed row gate")
        _verify_free_gate_ownership(
            gates=expectation.gate_owners, row_owners=row_owners,
            final_certificate=row_final_certificate, od_keys=od_keys, poly_key=poly_key)

        mask_keys = _device_mask_keys(tech)
        all_keys = set(mask_keys) | set(metal_keys)
        for realization in row_materialization.realizations:
            all_keys.update(
                (int(poly.layer), int(poly.datatype))
                for poly in tuple(getattr(realization.geometry, "polygons", ()) or ())
            )
        emitted = {key: witness.merged_region(key) for key in all_keys}
        route_owned = _route_regions(artifact)
        non_route_artwork = {
            key: witness.residual_region(
                key,
                tuple(
                    bbox
                    for _net, route_key, bbox in artifact.route_manifest.entries
                    if tuple(route_key) == key
                ),
            )
            for key in all_keys
            if key not in metal_keys and key in route_owned
        }
        # This replay is unconditional: a row with zero dummy obligations still
        # owns exact all-layer artwork that routing cannot stand in for.
        _verify_row_artwork_replay(
            program=row_program,
            materialization=row_materialization,
            final_certificate=row_final_certificate,
            emitted=emitted,
            route_owned=route_owned,
            non_route_artwork=non_route_artwork,
            shared_route_layers=frozenset(metal_keys),
        )
        expected_mask_hashes = dict(expectation.mask_layer_sha256s)
        observed_mask_hashes = {
            key: _region_sha256(emitted[key]) for key in mask_keys
        }
        producer_well_certificates = {
            item.layer_key: item
            for item in tuple(
                getattr(artifact, "well_merge_certificates", ()) or ()
            )
        }
        for key, input_hash in expectation.well_merge_input_sha256s:
            certificate = producer_well_certificates.get(key)
            if certificate is None:
                return unmeasurable(
                    f"emitted well union on {key} lacks its producer certificate"
                )
            if (
                certificate.input_region_sha256 != input_hash
                or certificate.output_region_sha256
                != observed_mask_hashes.get(key)
            ):
                return unmeasurable(
                    "emitter well union certificate does not bind expectation/GDS"
                )
        mask_mismatches = tuple(
            (key, expected_mask_hashes.get(key), observed_mask_hashes[key])
            for key in mask_keys
            if expected_mask_hashes.get(key) != observed_mask_hashes[key]
        )
        od = _region_union(emitted[key] for key in od_keys)
        poly = emitted[poly_key]
        structural = _region_union(emitted[key] for key in mask_keys)
        if structural.is_empty() or od.is_empty() or poly.is_empty():
            return unmeasurable("exact GDS lacks measurable structural OD/poly artwork")
        core_bbox = _box_tuple(structural.bbox())
        core_area = float(
            (core_bbox[2] - core_bbox[0]) * (core_bbox[3] - core_bbox[1]) / 1_000_000.0
        )
        largest_void = _largest_empty_rectangle_um2(structural, core_bbox)
        observations: tuple[DummyEndcapObservation, ...] = ()
        if expected_dummy:
            observations = _local_dummy_observations(
                program=row_program,
                materialization=row_materialization,
                final_certificate=row_final_certificate,
                emitted=emitted,
                route_owned=route_owned,
                od_key=od_keys[0],
                poly_key=poly_key,
                contact_key=contact_key,
                m1_key=metal_keys[0],
            )
            if len(observations) != expected_dummy:
                return unmeasurable("dummy obligation/GDS mapping is not one-to-one")
        shared_fraction = _owner_shared_od_fraction(
            od, poly, expectation.gate_owners,
            dummy_gate_bboxes_nm=tuple(item.gate_bbox_nm for item in observations),
        )
        realized = sum(item.status == "pass" for item in observations)
        failed = tuple(item for item in observations if item.status == "fail")
        mismatches = []
        mismatches.extend(
            (f"mask_layer_sha256:{key}", expected, actual)
            for key, expected, actual in mask_mismatches
        )
        for name, actual, expected in (
            ("core_bbox_nm", core_bbox, expectation.core_bbox_nm),
            ("core_area_um2", core_area, expectation.core_area_um2),
            ("largest_internal_void_um2", largest_void, expectation.largest_internal_void_um2),
            ("shared_od_fraction", shared_fraction, expectation.shared_od_fraction),
        ):
            if actual != expected:
                mismatches.append((name, expected, actual))
        status = "fail" if failed or mismatches else "pass"
        reasons = []
        if failed:
            reasons.append(f"{len(failed)} declared dummy end-cap ties are absent")
        if mismatches:
            reasons.append(f"expectation mismatches={tuple(mismatches)}")
        reason = "; ".join(reasons)
        return ExactStructureProfileReport(
            status=status,
            core_bbox_nm=core_bbox,
            core_area_um2=core_area,
            largest_internal_void_um2=largest_void,
            shared_od_fraction=shared_fraction,
            expected_dummy_endcaps=expected_dummy,
            realized_dummy_endcaps=realized,
            dummy_endcap_observations=observations,
            artifact_sha256=artifact_hash,
            route_manifest_sha256=manifest_hash,
            row_program_sha256=row_hash,
            row_final_certificate_sha256=final_hash,
            expectation_sha256=expectation_hash,
            reason=reason,
        )
    except Exception as exc:  # fail closed at the artifact-measurement boundary
        return unmeasurable(f"exact structure profile could not be measured: {exc}")


__all__ = [
    "DummyEndcapObservation",
    "ExactStructureExpectation",
    "ExactStructureProfileReport",
    "StructureGateOwner",
    "WellMergeCertificate",
    "compile_exact_structure_expectation",
    "measure_exact_structure_profile",
]
