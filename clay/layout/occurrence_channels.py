"""Exact OD/poly channels owned by ordered materialized source occurrences.

An occurrence is one indexed materialized source occurrence/finger frame, even
when its physical unit has several channels.  This inventory captures the unit mask artwork
while the producer still knows its source owner.  Replay never reconstructs
ownership from merged artwork or terminal-access probes.  It proves geometry;
the enclosing row materialization hash binds that geometry to its declaration.

All region operations use the same integer-nm KLayout polygon semantics as the
exact artifact consumers.  Connected polygons retain their hulls and holes;
bounding boxes are derived conveniences and carry no channel-shape authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import klayout.db as kdb

from clay.layout.backend import LayerPolygon


_Point = tuple[int, int]
_Ring = tuple[_Point, ...]
_Layer = tuple[int, int]
_MAX_ITEMS = 262_144
_MAX_VERTICES = 1_048_576
_MAX_CONTOUR_VERTICES = 4_096
_MAX_COORDINATE_NM = 1_000_000_000


def _tuple(value: object, field: str, *, nonempty: bool = False, limit: int = _MAX_ITEMS) -> tuple:
    if type(value) is not tuple or len(value) > limit or (nonempty and not value):
        raise ValueError(f"{field} must be a bounded exact tuple")
    return value


def _owner(value: object) -> str:
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError("channel occurrence owner must be a nonempty bounded exact string")
    return value


def _coordinate(value: object) -> int:
    if type(value) is not int or abs(value) > _MAX_COORDINATE_NM:
        raise ValueError("channel coordinates must be bounded exact integer nm")
    return value


def _layer(value: object) -> _Layer:
    if (type(value) is not tuple or len(value) != 2
            or any(type(item) is not int or not 0 <= item <= 65_535 for item in value)):
        raise ValueError("channel layer key must be an exact physical layer pair")
    return value


def _canonical_ring(points: _Ring) -> _Ring:
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("channel polygon contour is degenerate")
    smallest = min(points)
    best = None
    for sequence in (points, tuple(reversed(points))):
        for index, point in enumerate(sequence):
            if point == smallest:
                candidate = sequence[index:] + sequence[:index]
                if best is None or candidate < best:
                    best = candidate
    return best


def _ring(value: object) -> _Ring:
    points = _tuple(value, "channel contour", nonempty=True, limit=_MAX_CONTOUR_VERTICES)
    for point in points:
        if type(point) is not tuple or len(point) != 2:
            raise ValueError("channel polygon vertices must be exact coordinate pairs")
        _coordinate(point[0])
        _coordinate(point[1])
    if points != _canonical_ring(points):
        raise ValueError("channel polygon contours must be canonical")
    return points


def _shape_keys(region: kdb.Region) -> tuple[tuple[_Ring, tuple[_Ring, ...]], ...]:
    rows = []
    for polygon in region.merged().each():
        hull = _canonical_ring(tuple((int(point.x), int(point.y)) for point in polygon.each_point_hull()))
        holes = tuple(sorted(_canonical_ring(tuple((int(point.x), int(point.y))
            for point in polygon.each_point_hole(index))) for index in range(polygon.holes())))
        rows.append((hull, holes))
    return tuple(sorted(rows))


def _raw_polygon(hull: _Ring, holes: tuple[_Ring, ...]) -> kdb.Polygon:
    polygon = kdb.Polygon([kdb.Point(x, y) for x, y in hull])
    for hole in holes:
        polygon.insert_hole([kdb.Point(x, y) for x, y in hole])
    return polygon


@dataclass(frozen=True, slots=True, order=True)
class ChannelPolygonNM:
    """One complete connected channel or mask polygon in exact integer nm."""

    hull_nm: _Ring
    holes_nm: tuple[_Ring, ...] = ()

    def __post_init__(self) -> None:
        _ring(self.hull_nm)
        holes = _tuple(self.holes_nm, "channel polygon holes", limit=_MAX_CONTOUR_VERTICES)
        for hole in holes:
            _ring(hole)
        if holes != tuple(sorted(set(holes))):
            raise ValueError("channel polygon holes must be canonical and unique")
        if len(self.hull_nm) + sum(len(hole) for hole in holes) > _MAX_CONTOUR_VERTICES:
            raise ValueError("channel polygon exceeds the contour complexity bound")
        polygon = _raw_polygon(self.hull_nm, self.holes_nm)
        if (polygon.area() <= 0 or _shape_keys(kdb.Region(polygon))
                != ((self.hull_nm, self.holes_nm),)):
            raise ValueError("channel polygon must describe one complete canonical connected region")

    def to_region(self) -> kdb.Region:
        """Return a fresh region; mutable KLayout objects are never retained."""
        self.__post_init__()
        return kdb.Region(_raw_polygon(self.hull_nm, self.holes_nm))

    @property
    def bbox_nm(self) -> tuple[int, int, int, int]:
        self.__post_init__()
        return (min(x for x, _y in self.hull_nm), min(y for _x, y in self.hull_nm),
                max(x for x, _y in self.hull_nm), max(y for _x, y in self.hull_nm))

    def translated(self, dx: int, dy: int) -> ChannelPolygonNM:
        self.__post_init__()
        _coordinate(dx)
        _coordinate(dy)
        return ChannelPolygonNM(
            tuple((x + dx, y + dy) for x, y in self.hull_nm),
            tuple(tuple((x + dx, y + dy) for x, y in hole) for hole in self.holes_nm))

    def payload(self) -> dict:
        self.__post_init__()
        return {"hull_nm": self.hull_nm, "holes_nm": self.holes_nm}


def _shapes(region: kdb.Region) -> tuple[ChannelPolygonNM, ...]:
    keys = _shape_keys(region)
    _tuple(keys, "channel component inventory")
    return tuple(ChannelPolygonNM(hull, holes) for hull, holes in keys)


def _shape_region(shapes: tuple[ChannelPolygonNM, ...]) -> kdb.Region:
    region = kdb.Region()
    for shape in shapes:
        if type(shape) is not ChannelPolygonNM:
            raise TypeError("channel inventory contains a foreign polygon type")
        region += shape.to_region()
    return region.merged()


def _shape_tuple(value: object, field: str) -> tuple[ChannelPolygonNM, ...]:
    shapes = _tuple(value, field, nonempty=True)
    for shape in shapes:
        if type(shape) is not ChannelPolygonNM:
            raise TypeError(f"{field} requires exact ChannelPolygonNM values")
        shape.__post_init__()
    if shapes != tuple(sorted(set(shapes))):
        raise ValueError(f"{field} must contain canonical unique polygons")
    if _shape_keys(_shape_region(shapes)) != tuple((shape.hull_nm, shape.holes_nm) for shape in shapes):
        raise ValueError(f"{field} must contain complete merged components")
    return shapes


@dataclass(frozen=True, slots=True)
class OwnedOccurrenceChannels:
    """Owned masks and all channels for one ordered source occurrence."""

    occurrence_index: int
    logical_owner: str
    channels: tuple[ChannelPolygonNM, ...]
    owned_mask_polygons: tuple[tuple[_Layer, tuple[ChannelPolygonNM, ...]], ...]

    def __post_init__(self) -> None:
        if type(self.occurrence_index) is not int or not 0 <= self.occurrence_index < _MAX_ITEMS:
            raise ValueError("channel occurrence index must be a bounded nonnegative exact integer")
        _owner(self.logical_owner)
        _shape_tuple(self.channels, "occurrence channels")
        masks = _tuple(self.owned_mask_polygons, "owned occurrence masks", nonempty=True, limit=64)
        keys = []
        for row in masks:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("owned masks require exact layer/polygon pairs")
            key, polygons = row
            keys.append(_layer(key))
            _shape_tuple(polygons, "owned layer components")
        if tuple(keys) != tuple(sorted(set(keys))):
            raise ValueError("owned mask layers must be canonical and unique")

    def payload(self) -> dict:
        self.__post_init__()
        return {
            "occurrence_index": self.occurrence_index,
            "logical_owner": self.logical_owner,
            "channels": tuple(shape.payload() for shape in self.channels),
            "owned_mask_polygons": tuple((key, tuple(shape.payload() for shape in shapes))
                                         for key, shapes in self.owned_mask_polygons),
        }


def _mask_keys(od_layer_keys: object, poly_layer_key: object) -> tuple[_Layer, ...]:
    od_keys = _tuple(od_layer_keys, "OD layer inventory", nonempty=True, limit=63)
    for key in od_keys:
        _layer(key)
    if od_keys != tuple(sorted(set(od_keys))):
        raise ValueError("OD layer keys must be canonical and unique")
    poly_key = _layer(poly_layer_key)
    if poly_key in od_keys:
        raise ValueError("OD and poly must have distinct physical layer keys")
    return tuple(sorted(od_keys + (poly_key,)))


def _crossing(masks: dict, od_keys: tuple[_Layer, ...], poly_key: _Layer) -> kdb.Region:
    od = kdb.Region()
    for key in od_keys:
        od += masks.get(key, kdb.Region())
    return (od.merged() & masks.get(poly_key, kdb.Region())).merged()


def _preflight_inventory(occurrences: tuple) -> None:
    """Bound all nested work before constructing or validating any region."""
    polygon_count = vertex_count = 0
    for occurrence in occurrences:
        if type(occurrence) is not OwnedOccurrenceChannels:
            raise TypeError("channel occurrences must have the exact owned occurrence type")
        channels = _tuple(occurrence.channels, "occurrence channels", nonempty=True)
        masks = _tuple(occurrence.owned_mask_polygons, "owned occurrence masks", nonempty=True, limit=64)
        shape_groups = [channels]
        for row in masks:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("owned masks require exact layer/polygon pairs")
            shape_groups.append(_tuple(row[1], "owned layer components", nonempty=True))
        for shapes in shape_groups:
            polygon_count += len(shapes)
            if polygon_count > _MAX_ITEMS:
                raise ValueError("channel inventory exceeds cumulative geometry complexity bounds")
            for shape in shapes:
                if type(shape) is not ChannelPolygonNM:
                    raise TypeError("channel inventory contains a foreign polygon type")
                hull = _tuple(shape.hull_nm, "channel contour", nonempty=True, limit=_MAX_CONTOUR_VERTICES)
                holes = _tuple(shape.holes_nm, "channel polygon holes", limit=_MAX_CONTOUR_VERTICES)
                vertex_count += len(hull)
                for hole in holes:
                    vertex_count += len(_tuple(hole, "channel contour", nonempty=True,
                                               limit=_MAX_CONTOUR_VERTICES))
                if vertex_count > _MAX_VERTICES:
                    raise ValueError("channel inventory exceeds cumulative geometry complexity bounds")


def _validate_inventory(inventory: RowChannelInventory) -> dict[_Layer, kdb.Region]:
    mask_keys = _mask_keys(inventory.od_layer_keys, inventory.poly_layer_key)
    occurrences = _tuple(inventory.occurrences, "channel occurrence inventory", nonempty=True)
    _preflight_inventory(occurrences)
    complete_masks = {key: kdb.Region() for key in mask_keys}
    channels = []
    for index, occurrence in enumerate(occurrences):
        if type(occurrence) is not OwnedOccurrenceChannels:
            raise TypeError("channel occurrences must have the exact owned occurrence type")
        occurrence.__post_init__()
        if occurrence.occurrence_index != index:
            raise ValueError("channel occurrence indices must be complete and in exact row order")
        local_masks = {}
        for key, shapes in occurrence.owned_mask_polygons:
            if key not in complete_masks:
                raise ValueError("owned occurrence mask names a foreign structural layer")
            local_masks[key] = _shape_region(shapes)
            complete_masks[key] += local_masks[key]
        actual_channels = _shapes(_crossing(local_masks, inventory.od_layer_keys, inventory.poly_layer_key))
        if actual_channels != occurrence.channels:
            raise ValueError("owned occurrence channels differ from its complete exact OD/poly intersection")
        channels.extend(occurrence.channels)
    for key, region in complete_masks.items():
        complete_masks[key] = region.merged()
    merged_channels = _shapes(_crossing(complete_masks, inventory.od_layer_keys, inventory.poly_layer_key))
    if tuple(sorted(channels)) != merged_channels:
        raise ValueError("source occurrences do not uniquely cover every complete merged OD/poly channel")
    return complete_masks


@dataclass(frozen=True, slots=True)
class RowChannelInventory:
    """Complete per-occurrence channel ownership in one macro-local nm frame."""

    od_layer_keys: tuple[_Layer, ...]
    poly_layer_key: _Layer
    occurrences: tuple[OwnedOccurrenceChannels, ...]

    def __post_init__(self) -> None:
        _validate_inventory(self)

    def payload(self) -> dict:
        self.__post_init__()
        return {
            "schema": "clay.row-channel-inventory/v1",
            "od_layer_keys": self.od_layer_keys,
            "poly_layer_key": self.poly_layer_key,
            "occurrences": tuple(occurrence.payload() for occurrence in self.occurrences),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _artwork_counts(polygons: object) -> tuple[int, int]:
    polygons = _tuple(polygons, "actual occurrence artwork", nonempty=True)
    vertices = 0
    for layer_polygon in polygons:
        if type(layer_polygon) is not LayerPolygon:
            raise TypeError("owned occurrence artwork requires exact LayerPolygon values")
        _layer((layer_polygon.layer, layer_polygon.datatype))
        raw_vertices = layer_polygon.polygon.vertices
        if type(raw_vertices) not in {list, tuple} or not 3 <= len(raw_vertices) <= _MAX_CONTOUR_VERTICES:
            raise ValueError("actual polygon vertices must be a bounded nondegenerate sequence")
        vertices += len(raw_vertices)
        if vertices > _MAX_VERTICES:
            raise ValueError("actual artwork exceeds cumulative vertex complexity bounds")
    return len(polygons), vertices


def _polygon_masks(polygons: object, keys: tuple[_Layer, ...]) -> dict[_Layer, kdb.Region]:
    _artwork_counts(polygons)
    regions = {key: kdb.Region() for key in keys}
    for layer_polygon in polygons:
        key = (layer_polygon.layer, layer_polygon.datatype)
        points = [kdb.Point(_coordinate(point.x), _coordinate(point.y))
                  for point in layer_polygon.polygon.vertices]
        if key in regions:
            polygon = kdb.Polygon(points)
            if polygon.area() <= 0:
                raise ValueError("actual structural mask polygon has no positive area")
            regions[key].insert(polygon)
    return {key: region.merged() for key, region in regions.items()}


def _require_row_masks(inventory: RowChannelInventory, row_polygons: object) -> None:
    expected = _validate_inventory(inventory)
    actual = _polygon_masks(row_polygons, tuple(expected))
    for key in expected:
        if not (expected[key] ^ actual[key]).is_empty():
            raise ValueError("owned occurrence masks do not exactly cover the actual row on each structural layer")
    # The per-layer equality and inventory bijection jointly establish complete
    # actual channel ownership, including crossings introduced across units.


def _owned_occurrence(index: int, owner: str, masks: dict, od_keys: tuple, poly_key: tuple) -> OwnedOccurrenceChannels:
    """One canonical extraction path for newly captured and transported masks."""
    return OwnedOccurrenceChannels(
        occurrence_index=index, logical_owner=owner,
        channels=_shapes(_crossing(masks, od_keys, poly_key)),
        owned_mask_polygons=tuple((key, _shapes(region)) for key, region in sorted(masks.items())
                                 if not region.is_empty()))


def compile_row_channel_inventory(*, occurrence_artwork, row_polygons,
                                  od_layer_keys, poly_layer_key) -> RowChannelInventory:
    """Capture channels from each known owned, transformed unit before merging.

    No terminal probes, occurrence-count approximation, external hashes, or
    ownership guesses are accepted as substitutes for the supplied unit masks.
    """
    keys = _mask_keys(od_layer_keys, poly_layer_key)
    owned = _tuple(occurrence_artwork, "owned occurrence artwork", nonempty=True)
    polygon_count = vertex_count = 0
    for row in owned:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("occurrence artwork must contain exact owner/polygon pairs")
        owner, polygons = row
        _owner(owner)
        count, vertices = _artwork_counts(polygons)
        polygon_count += count
        vertex_count += vertices
        if polygon_count > _MAX_ITEMS or vertex_count > _MAX_VERTICES:
            raise ValueError("owned artwork exceeds cumulative geometry complexity bounds")
    _artwork_counts(row_polygons)
    occurrences = []
    for index, (owner, polygons) in enumerate(owned):
        masks = _polygon_masks(polygons, keys)
        occurrences.append(_owned_occurrence(index, owner, masks, od_layer_keys, poly_layer_key))
    inventory = RowChannelInventory(od_layer_keys, poly_layer_key, tuple(occurrences))
    _require_row_masks(inventory, row_polygons)
    return inventory


def replay_row_channel_inventory(*, geometry, consumed_devices) -> RowChannelInventory:
    """Validate and return the attached inventory without minting new ownership."""
    inventory = getattr(geometry, "channel_inventory", None)
    if type(inventory) is not RowChannelInventory:
        raise TypeError("row geometry lacks an exact attached RowChannelInventory")
    devices = _tuple(consumed_devices, "consumed row devices", nonempty=True)
    for owner in devices:
        _owner(owner)
    if len(devices) != len(set(devices)):
        raise ValueError("consumed row devices must be unique")
    fingers = getattr(geometry, "fingers", None)
    if type(fingers) not in {tuple, list} or len(fingers) != len(inventory.occurrences):
        raise ValueError("channel occurrence inventory must exactly cover ordered row fingers")
    for finger, occurrence in zip(fingers, inventory.occurrences):
        if _owner(getattr(finger, "member", None)) != occurrence.logical_owner:
            raise ValueError("channel occurrence owner differs from its exact row finger owner")
    if {occurrence.logical_owner for occurrence in inventory.occurrences} != set(devices):
        raise ValueError("channel occurrence owners must exactly cover consumed row devices")
    polygons = getattr(geometry, "polygons", None)
    if type(polygons) not in {tuple, list}:
        raise TypeError("row geometry requires an explicit polygon inventory")
    _require_row_masks(inventory, tuple(polygons))
    return inventory


def transform_row_channel_inventory(*, geometry, consumed_devices, matrix,
                                    translation_nm, row_polygons) -> RowChannelInventory:
    """Transport retained occurrence ownership through one exact rigid D4 map.

    The source is replayed against its actual fingers and masks first.  Crossings
    are then re-extracted from transformed owned masks, including all holes, and
    matched to the complete supplied target row masks.  The caller attaches this
    inventory and replays the target finger sequence after constructing its row.
    """
    rows = _tuple(matrix, "channel transform matrix")
    if len(rows) != 2:
        raise ValueError("channel transform matrix must be two by two")
    for row in rows:
        if type(row) is not tuple or len(row) != 2 or any(type(value) is not int or value not in {-1, 0, 1} for value in row):
            raise ValueError("channel transform requires an exact signed-permutation matrix")
    (a, b), (c, d) = rows
    if a * a + b * b != 1 or c * c + d * d != 1 or a * c + b * d != 0:
        raise ValueError("channel transform must be rigid and orthogonal")
    translation = _tuple(translation_nm, "channel transform translation")
    if len(translation) != 2:
        raise ValueError("channel transform translation must be an exact nm pair")
    dx, dy = (_coordinate(value) for value in translation)
    original = replay_row_channel_inventory(geometry=geometry, consumed_devices=consumed_devices)
    _artwork_counts(row_polygons)

    def transform_ring(ring):
        return _canonical_ring(tuple((_coordinate(a * x + b * y + dx),
                                      _coordinate(c * x + d * y + dy)) for x, y in ring))

    occurrences = []
    for occurrence in original.occurrences:
        masks = {}
        for key, shapes in occurrence.owned_mask_polygons:
            transformed = tuple(ChannelPolygonNM(transform_ring(shape.hull_nm),
                tuple(sorted(transform_ring(hole) for hole in shape.holes_nm))) for shape in shapes)
            masks[key] = _shape_region(transformed)
        occurrences.append(_owned_occurrence(occurrence.occurrence_index, occurrence.logical_owner,
            masks, original.od_layer_keys, original.poly_layer_key))
    inventory = RowChannelInventory(original.od_layer_keys, original.poly_layer_key, tuple(occurrences))
    _require_row_masks(inventory, row_polygons)
    return inventory


__all__ = ["ChannelPolygonNM", "OwnedOccurrenceChannels", "RowChannelInventory",
           "compile_row_channel_inventory", "replay_row_channel_inventory",
           "transform_row_channel_inventory"]
