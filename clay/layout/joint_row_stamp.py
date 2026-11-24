"""Rigid native-row bridge retaining physical parts and logical associations."""
from collections import Counter
from dataclasses import dataclass

from clay.layout.joint_row_geometry import (
    JointRowGeometryBinding, joint_row_polygons, joint_occurrence_centers, replay_joint_row_binding, joint_record_counter,
)
from clay.layout.occurrence_artwork import polygon_records
from clay.layout.placed_geometry import placed_point_nm


@dataclass(frozen=True, slots=True)
class JointRowStampBinding:
    source: JointRowGeometryBinding
    consumed_devices: tuple[str, ...]
    native_origin_nm: tuple[int, int]
    native_row_sha256: str
    sha256: str

    def payload(self):
        return dict(schema='clay.joint-row-stamp/v1', source=self.source.payload(),
                    consumed_devices=self.consumed_devices, native_origin_nm=self.native_origin_nm,
                    native_row_sha256=self.native_row_sha256)

    def __post_init__(self):
        from clay.layout.owned_artwork import _require, _hash, _digest, _members
        _require(type(self.source) is JointRowGeometryBinding, 'joint row stamp source is foreign')
        _require(_members(self.consumed_devices) == self.consumed_devices,
                 'joint row stamp member denominator is malformed')
        _require(type(self.native_origin_nm) is tuple and len(self.native_origin_nm) == 2
                 and all(type(value) is int for value in self.native_origin_nm), 'joint row stamp frame is malformed')
        _require(_digest(self.native_row_sha256) and _digest(self.sha256)
                 and _hash(self.payload()) == self.sha256, 'joint row stamp binding changed')


def bind_joint_row_stamp(native_geometry, stamp, *, native_origin_nm, consumed_devices):
    from clay.layout.owned_artwork import _require, _hash, _members, _stamp_records
    from clay.layout.place.row_replay_session import current_row_geometry_sha256
    members = _members(tuple(consumed_devices))
    row_sha = current_row_geometry_sha256(native_geometry, members)
    source = native_geometry.joint_row_binding
    payload = dict(schema='clay.joint-row-stamp/v1', source=source.payload(), consumed_devices=members,
                   native_origin_nm=native_origin_nm, native_row_sha256=row_sha)
    binding = JointRowStampBinding(source, members, native_origin_nm, row_sha, _hash(payload))
    binding.__post_init__()
    _require(tuple(sorted(stamp.composite_members)) == members, 'joint row bridge owner denominator differs')
    _require(joint_record_counter(_stamp_records(stamp, tuple(v/1000 for v in native_origin_nm))) ==
             joint_record_counter(polygon_records(native_geometry.polygons)), 'joint row bridge actual polygon multiset differs')
    stamp.joint_row_stamp_binding = binding
    stamp.physical_occurrences_by_member = joint_occurrence_centers(source, native_origin_nm)


def joint_stamp_owner_records(stamp, placement_um, members):
    from clay.layout.owned_artwork import _require, _stamp_records
    binding = getattr(stamp, 'joint_row_stamp_binding', None)
    _require(type(binding) is JointRowStampBinding, 'complete joint row stamp binding is unavailable')
    binding.__post_init__()
    _require(binding.consumed_devices == members, 'joint row stamp logical denominator differs')
    source = binding.source
    polygons = joint_row_polygons(source.parts)
    replay_joint_row_binding(source, polygons=polygons, consumed_devices=members)
    origin = binding.native_origin_nm

    def records(parts):
        return tuple((layer, dt, tuple(placed_point_nm((x-origin[0])/1000,
            (y-origin[1])/1000, placement_um) for x,y in vertices))
            for layer,dt,vertices in polygon_records(joint_row_polygons(parts)))

    _require(joint_record_counter(_stamp_records(stamp, placement_um)) == joint_record_counter(records(source.parts)),
             'joint row current stamp artwork differs from complete physical part multiset')
    owners = {occ.occurrence_id: occ.owner for occ in source.occurrences}
    # A shared accessory is one physical part, but may contribute to more than
    # one logical union. Do not replay this association expansion as geometry.
    return {owner: records(tuple(part for part in source.parts
             if any(owners[occurrence_id] == owner for occurrence_id in part.occurrence_ids)))
            for owner in members}
