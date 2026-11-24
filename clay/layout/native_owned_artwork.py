"""Emit-time logical associations for continuous native MOS artwork.

A shared diffusion, well or bus is one physical polygon associated with all
declared owners it serves. This ledger proves conservation and ownership of
emitted parts; it is not a raw-PCell master, terminal extraction or admission.
"""
from collections import Counter
from dataclasses import asdict, dataclass, replace

from clay.layout.occurrence_artwork import polygon_records
from clay.layout.owned_artwork import _digest, _hash, _members, _require, _stamp_records
from clay.layout.placed_geometry import placed_point_nm


@dataclass(frozen=True, slots=True)
class NativeArtworkPart:
    part_id: str
    role: str
    owners: tuple[str, ...]
    polygon_record: tuple

    def __post_init__(self):
        _require(type(self.part_id) is str and 0 < len(self.part_id) <= 4096, 'native artwork part identity is missing')
        _require(type(self.role) is str and self.role, 'native artwork construction role is missing')
        _require(_members(self.owners) == self.owners, 'native artwork part owners are not canonical')
        layer, datatype, points = self.polygon_record
        if (
            type(layer) is not int
            or type(datatype) is not int
            or layer < 0
            or layer > 65535
            or datatype < 0
            or datatype > 65535
            or type(points) is not tuple
            or not (3 <= len(points) <= 4096)
        ):
            _require(False, 'native artwork part polygon is malformed')
        for point in points:
            if type(point) is not tuple or len(point) != 2:
                _require(False, 'native artwork part polygon is malformed')
            x, y = point
            if (
                type(x) is not int
                or type(y) is not int
                or x < -1_000_000_000
                or x > 1_000_000_000
                or y < -1_000_000_000
                or y > 1_000_000_000
            ):
                _require(False, 'native artwork part polygon is malformed')


@dataclass(frozen=True, slots=True)
class NativeOwnedArtworkBinding:
    members: tuple[str, ...]
    parts: tuple[NativeArtworkPart, ...]
    sha256: str

    def payload(self):
        return _payload(self.members, self.parts)

    def __post_init__(self):
        _require(_members(self.members) == self.members, 'native artwork owner denominator is malformed')
        _require(type(self.parts) is tuple and 1 <= len(self.parts) <= 262144, 'native artwork parts are missing')
        for part in self.parts:
            _require(type(part) is NativeArtworkPart, 'native artwork part is foreign')
            part.__post_init__()
        _require(sum(len(part.polygon_record[2]) for part in self.parts) <= 1048576,
                 'native artwork vertex complexity exceeds bounds')
        _require(len({part.part_id for part in self.parts}) == len(self.parts),
                 'native artwork part identity is duplicated')
        _require({owner for part in self.parts for owner in part.owners} == set(self.members),
                 'native artwork owner denominator differs')
        _require(_hash(self.payload()) == self.sha256, 'native artwork binding changed')


def _payload(members, parts):
    serialized_parts = tuple(
        {
            'part_id': part.part_id,
            'role': part.role,
            'owners': part.owners,
            'polygon_record': part.polygon_record,
        }
        if type(part) is NativeArtworkPart
        else asdict(part)
        for part in parts
    )
    return {
        'schema': 'clay.native-owned-artwork/v1',
        'members': members,
        'parts': serialized_parts,
        'scope': 'emitted_logical_artwork_association_no_terminal_or_physical_admission',
    }


def _binding(members, parts):
    members, parts = tuple(sorted(members)), tuple(parts)
    return NativeOwnedArtworkBinding(members, parts, _hash(_payload(members, parts)))


class NativeOwnedRectangles(list):
    """Builder-local emitter: every append records its explicit construction owners.

    Builders select owners from their actual source/finger/net program before
    emitting. No assignment is inferred from polygon position or nearest bbox.
    Each append freezes its rectangle and owners together. The list is only an
    emission view; any later reordering or replacement fails closed at binding.
    """
    def __init__(self, members):
        super().__init__()
        self.members = _members(tuple(sorted(set(members))))
        self._emissions = []
        self.select(self.members, 'shared')

    def select(self, owners, role):
        self.owners = _members(tuple(sorted(set(owners))))
        _require(set(self.owners) <= set(self.members), 'native emission owner is foreign')
        _require(type(role) is str and role, 'native emission construction role is missing')
        self.role = role

    def append(self, rectangle):
        key, box = rectangle
        frozen = tuple(key), tuple(box)
        super().append(frozen)
        self._emissions.append((frozen, self.role, self.owners))

    def extend(self, rectangles):
        for rectangle in rectangles:
            self.append(rectangle)

    def binding(self, polygons):
        expected = tuple(rectangle for rectangle, _role, _owners in self._emissions)
        _require(tuple(self) == expected, 'native emission sequence changed after append')
        records = polygon_records(polygons)
        _require(len(records) == len(self._emissions), 'native emission polygon denominator differs')
        # polygon_records is canonical sorted order, while emitter ownership is
        # attached to append order. Reconstruct each emitted rectangle explicitly.
        from clay.layout.device_cell import _rect
        emitted = tuple(polygon_records((_rect(key, box),))[0] for key, box in expected)
        _require(Counter(emitted) == Counter(records), 'native emission polygon multiset differs')
        parts = tuple(NativeArtworkPart(f'part/{index}', role, owners, record)
                      for index, (record, (_rectangle, role, owners)) in enumerate(zip(emitted, self._emissions)))
        return _binding(self.members, parts)


def select_native_owners(rectangles, owners, role):
    """Private helper emitters also accept ordinary lists without evidence capture."""
    if type(rectangles) is NativeOwnedRectangles:
        rectangles.select(owners, role)


def native_owners_by_net(*member_nets):
    """Collect the explicit source owners of each emitted electrical bus."""
    result = {}
    for member, net in member_nets:
        result.setdefault(net, set()).add(member)
    return result


def attach_native_owned_artwork(geometry, binding):
    object.__setattr__(geometry, 'native_owned_artwork', binding)
    replay_native_owned_artwork(geometry, binding.members)
    return geometry


def replay_native_owned_artwork(geometry, consumed_devices):
    binding = getattr(geometry, 'native_owned_artwork', None)
    _require(type(binding) is NativeOwnedArtworkBinding, 'native artwork binding is missing or foreign')
    binding.__post_init__()
    _require(binding.members == tuple(sorted(consumed_devices)), 'native artwork source denominator differs')
    _require(Counter(part.polygon_record for part in binding.parts) == Counter(polygon_records(geometry.polygons)),
             'native artwork actual polygon multiset differs')
    return binding


def adapt_native_owned_artwork(source, target, *, removed_layers=(), added_polygons=()):
    """Retain delegated parts, replacing declared shared well/body additions."""
    original = replay_native_owned_artwork(source, source.native_owned_artwork.members)
    parts = [part for part in original.parts if part.polygon_record[:2] not in removed_layers]
    parts.extend(NativeArtworkPart(f'adapter/{index}', 'shared_well_body_oxide', original.members, record)
                 for index, record in enumerate(polygon_records(added_polygons)))
    return attach_native_owned_artwork(target, _binding(original.members, parts))


def transform_native_owned_artwork(source, target, transform_point):
    original = replay_native_owned_artwork(source, source.native_owned_artwork.members)
    parts = tuple(replace(part, polygon_record=(*part.polygon_record[:2],
                  tuple(transform_point(point) for point in part.polygon_record[2]))) for part in original.parts)
    return attach_native_owned_artwork(target, _binding(original.members, parts))


@dataclass(frozen=True, slots=True)
class NativeOwnedArtworkStamp:
    source: NativeOwnedArtworkBinding
    native_origin_nm: tuple[int, int]
    native_row_sha256: str
    sha256: str

    def payload(self):
        return dict(schema='clay.native-owned-artwork-stamp/v1', source=self.source.payload(),
                    native_origin_nm=self.native_origin_nm, native_row_sha256=self.native_row_sha256)

    def __post_init__(self):
        _require(type(self.source) is NativeOwnedArtworkBinding, 'native artwork stamp source is foreign')
        self.source.__post_init__()
        _require(type(self.native_origin_nm) is tuple and len(self.native_origin_nm) == 2
                 and all(type(v) is int and abs(v) <= 1_000_000_000 for v in self.native_origin_nm),
                 'native artwork stamp origin is malformed')
        _require(_digest(self.native_row_sha256) and _hash(self.payload()) == self.sha256,
                 'native artwork stamp binding changed')


def bind_native_owned_artwork_stamp(native, stamp, *, native_origin_nm, consumed_devices):
    from clay.layout.place.row_program import row_geometry_sha256
    binding = replay_native_owned_artwork(native, consumed_devices)
    _require(tuple(sorted(stamp.composite_members)) == binding.members,
             'native artwork bridge owner denominator differs')
    _require(Counter(_stamp_records(stamp, tuple(v / 1000 for v in native_origin_nm))) ==
             Counter(part.polygon_record for part in binding.parts),
             'native artwork bridge polygon multiset differs')
    row_sha = row_geometry_sha256(native, consumed_devices)
    payload = dict(schema='clay.native-owned-artwork-stamp/v1', source=binding.payload(),
                   native_origin_nm=native_origin_nm, native_row_sha256=row_sha)
    stamp.native_owned_artwork_stamp = NativeOwnedArtworkStamp(binding, native_origin_nm, row_sha, _hash(payload))


def native_stamp_owner_records(stamp, position, members):
    binding = getattr(stamp, 'native_owned_artwork_stamp', None)
    _require(type(binding) is NativeOwnedArtworkStamp, 'native artwork stamp binding is missing')
    binding.__post_init__()
    source = binding.source
    source.__post_init__()
    _require(source.members == members, 'native artwork stamp owner denominator differs')
    origin = binding.native_origin_nm
    records = tuple((part, (*part.polygon_record[:2], tuple(placed_point_nm(
        (x-origin[0])/1000, (y-origin[1])/1000, position) for x, y in part.polygon_record[2])))
        for part in source.parts)
    _require(Counter(_stamp_records(stamp, position)) == Counter(record for _, record in records),
             'native artwork current stamp polygon multiset differs')
    return {owner: tuple(record for part, record in records if owner in part.owners) for owner in members}
