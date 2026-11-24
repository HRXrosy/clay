"""Replay full logical artwork through the native row-to-stamp bridge.

This is current geometry evidence. It grants neither source nor physical
admission. Shared-OD rows without complete captured ownership are unavailable.
"""
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

from clay.layout.occurrence_artwork import occurrence_artwork_payload, polygon_records
from clay.layout.placed_geometry import placed_point_nm


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _members(value):
    _require(type(value) is tuple and value and len(value) <= 4096
             and all(type(name) is str and name and name == name.strip() for name in value)
             and len(set(value)) == len(value), 'owned artwork source owner inventory is malformed')
    return tuple(sorted(value))


def _stamp_records(stamp, placement_um):
    layers = getattr(stamp, 'polys_by_layer', None)
    _require(isinstance(layers, Mapping) and layers, 'owned artwork stamp polygons are unavailable')
    _require(type(placement_um) is tuple and len(placement_um) == 2
             and all(type(v) in (int, float) and math.isfinite(v) for v in placement_um),
             'owned artwork stamp placement is malformed')
    records, vertex_count = [], 0
    for layer, rings in layers.items():
        _require(type(layer) is tuple and len(layer) == 2
                 and all(type(v) is int and 0 <= v <= 65535 for v in layer),
                 'owned artwork polygon layer is malformed')
        _require(type(rings) in (tuple, list) and len(rings) <= 262144,
                 'owned artwork polygon inventory exceeds bounds')
        for ring in rings:
            _require(type(ring) in (tuple, list) and 3 <= len(ring) <= 4096,
                     'owned artwork polygon vertices are malformed')
            vertex_count += len(ring)
            _require(vertex_count <= 1048576 and len(records) < 262144,
                     'owned artwork polygon complexity exceeds bounds')
            _require(all(type(point) in (tuple, list) and len(point) == 2
                         and all(type(v) in (int, float) and math.isfinite(v) for v in point) for point in ring),
                     'owned artwork polygon coordinates are malformed')
            records.append((*layer, tuple(placed_point_nm(x, y, placement_um) for x, y in ring)))
    _require(records, 'owned artwork stamp polygons are unavailable')
    return tuple(records)


def _bbox(records):
    points = tuple(point for _layer, _dt, vertices in records for point in vertices)
    _require(points, 'owned artwork source polygons are unavailable')
    xs, ys = zip(*points)
    box = min(xs), min(ys), max(xs), max(ys)
    _require(box[0] < box[2] and box[1] < box[3], 'owned artwork bbox is degenerate')
    return box


@dataclass(frozen=True, slots=True)
class OwnedArtworkStampBinding:
    sources: tuple
    consumed_devices: tuple[str, ...]
    native_origin_nm: tuple[int, int]
    native_row_sha256: str
    sha256: str

    def payload(self):
        return {'schema': 'clay.owned-artwork-stamp/v1',
                'sources': occurrence_artwork_payload(self.sources),
                'consumed_devices': self.consumed_devices,
                'native_origin_nm': self.native_origin_nm,
                'native_row_sha256': self.native_row_sha256}

    def __post_init__(self):
        _require(_members(self.consumed_devices) == self.consumed_devices,
                 'owned artwork source owner inventory is not canonical')
        _require(type(self.native_origin_nm) is tuple and len(self.native_origin_nm) == 2
                 and all(type(v) is int and abs(v) <= 1_000_000_000 for v in self.native_origin_nm),
                 'owned artwork native origin is malformed')
        _require(_digest(self.native_row_sha256), 'owned artwork native row binding is malformed')
        payload = self.payload()
        _require(tuple(sorted({source.owner for source in self.sources})) == self.consumed_devices,
                 'owned artwork source owner denominator differs')
        _require(_digest(self.sha256) and _hash(payload) == self.sha256,
                 'owned artwork retained binding changed')


def bind_row_owned_artwork_stamp(native_geometry, stamp, *, native_origin_nm, consumed_devices):
    """Bind an independently replayed native row to its actual emitted polygons."""
    from clay.layout.mosaic_geometry import MosaicGeometry
    stamp.source = 'discrete_mosaic' if type(native_geometry) is MosaicGeometry else 'shared_diffusion_array'
    if getattr(native_geometry, 'joint_row_binding', None) is not None:
        from clay.layout.joint_row_stamp import bind_joint_row_stamp
        return bind_joint_row_stamp(native_geometry, stamp, native_origin_nm=native_origin_nm,
                                    consumed_devices=consumed_devices)
    if getattr(native_geometry, 'native_owned_artwork', None) is not None:
        from clay.layout.native_owned_artwork import bind_native_owned_artwork_stamp
        return bind_native_owned_artwork_stamp(native_geometry, stamp, native_origin_nm=native_origin_nm,
                                              consumed_devices=consumed_devices)
    if (getattr(native_geometry, 'fixed_occurrence_sources', None) is None
            and getattr(native_geometry, 'fixed_pcell', None) is None):
        return
    from clay.layout.place.row_program import row_geometry_sha256
    members = _members(tuple(consumed_devices))
    # Includes source/finger frame, complete artwork multiset, terminal source
    # denominator and terminal component replay, not a caller-provided digest.
    row_sha = row_geometry_sha256(native_geometry, members)
    sources = native_geometry.fixed_occurrence_sources
    payload = {'schema': 'clay.owned-artwork-stamp/v1',
               'sources': occurrence_artwork_payload(sources), 'consumed_devices': members,
               'native_origin_nm': native_origin_nm, 'native_row_sha256': row_sha}
    binding = OwnedArtworkStampBinding(sources, members, native_origin_nm, row_sha, _hash(payload))
    _require(_members(stamp.composite_members) == members, 'owned artwork bridge owner denominator differs')
    actual = _stamp_records(stamp, tuple(v / 1000 for v in native_origin_nm))
    _require(Counter(actual) == Counter(polygon_records(native_geometry.polygons)),
             'owned artwork bridge changed actual native polygons')
    stamp.owned_artwork_binding = binding


def _bound_owner_records(stamp, placement_um, members):
    from clay.layout.source_resistor_placement import is_resistor_stamp, resistor_stamp_owner_records
    if is_resistor_stamp(stamp):
        return resistor_stamp_owner_records(stamp, placement_um, members)
    if getattr(stamp, 'joint_row_stamp_binding', None) is not None:
        from clay.layout.joint_row_stamp import joint_stamp_owner_records
        return joint_stamp_owner_records(stamp, placement_um, members)
    if getattr(stamp, 'native_owned_artwork_stamp', None) is not None:
        from clay.layout.native_owned_artwork import native_stamp_owner_records
        return native_stamp_owner_records(stamp, placement_um, members)
    binding = getattr(stamp, 'owned_artwork_binding', None)
    _require(type(binding) is OwnedArtworkStampBinding,
             'complete multiowner source artwork binding is unavailable')
    binding.__post_init__()
    _require(binding.consumed_devices == members, 'owned artwork stamp owner denominator differs')
    origin = binding.native_origin_nm
    by_owner = {}
    for source in binding.sources:
        by_owner.setdefault(source.owner, []).extend(
            (layer, datatype, tuple(placed_point_nm((x - origin[0]) / 1000,
                (y - origin[1]) / 1000, placement_um) for x, y in points))
            for layer, datatype, points in source.polygon_records)
    actual = _stamp_records(stamp, placement_um)
    expected = Counter(record for records in by_owner.values() for record in records)
    _require(Counter(actual) == expected, 'owned artwork current stamp polygons differ from retained sources')
    return by_owner


def quality_source_artwork_bboxes_nm(geometry, placement_um, source_devices):
    """Return complete logical artwork unions in current absolute integer nm.

    Spatial overlap across owners is permitted; no exclusive partition of
    physical space is inferred. A copied polygon's multiplicity is retained.
    """
    from clay.layout.source_resistor_placement import is_resistor_stamp
    requested = _members(source_devices)
    _require(requested == source_devices, 'owned artwork requested sources are not canonical')
    _require(isinstance(geometry, Mapping) and isinstance(placement_um, Mapping),
             'owned artwork current geometry/placement mappings are unavailable')
    result = {}
    for name, stamp in geometry.items():
        raw = getattr(stamp, 'composite_members', ())
        _require(type(raw) is tuple, 'owned artwork stamp owner inventory is malformed')
        members = _members(raw if raw else (name,))
        selected = set(members) & set(requested)
        if not selected:
            continue
        _require(name in placement_um, 'owned artwork current stamp placement is unavailable')
        if (len(members) > 1 or getattr(stamp, 'owned_artwork_binding', None) is not None
                or is_resistor_stamp(stamp)
                or getattr(stamp, 'native_owned_artwork_stamp', None) is not None
                or getattr(stamp, 'joint_row_stamp_binding', None) is not None):
            records = _bound_owner_records(stamp, placement_um[name], members)
        else:
            records = {members[0]: _stamp_records(stamp, placement_um[name])}
        for owner in selected:
            _require(owner not in result, 'owned artwork source owner has multiple stamps')
            result[owner] = _bbox(records[owner])
    _require(set(result) == set(requested), 'complete source artwork owner coverage is unavailable')
    return dict(sorted(result.items()))
