"""Neutral native MOS mosaics with complete per-owner joint geometry.

This currency preserves supplied raw units, accessories and terminal sites.
It neither declares a common source/gate rail nor establishes body tracing,
source admission, PDK legality, or a new reference construction authority.
"""
from dataclasses import InitVar, dataclass, field, replace
from types import SimpleNamespace

from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.common_centroid_array import _CCFinger
from clay.layout.joint_row_geometry import JointRowGeometryBinding
from clay.layout.occurrence_channels import RowChannelInventory
from clay.layout.spatial.intent_projection import DiscreteMosaicIntent
from clay.layout.spatial.mosaic_boundary_joins import MosaicBoundaryJoinIntent


_D4 = {'R0': ((1, 0), (0, 1)), 'R90': ((0, -1), (1, 0)),
       'R180': ((-1, 0), (0, -1)), 'R270': ((0, 1), (-1, 0)),
       'MR0': ((1, 0), (0, -1)), 'MR90': ((0, 1), (1, 0)),
       'MR180': ((-1, 0), (0, 1)), 'MR270': ((0, -1), (-1, 0))}


def _require(value, message):
    if not value:
        raise ValueError('neutral mosaic ' + message)


def _coordinate(value):
    _require(type(value) is int and abs(value) <= 1_000_000_000,
             'coordinates must be bounded exact integer nm')
    return value


def _bbox(value):
    _require(type(value) is tuple and len(value) == 4, 'bbox must be an exact tuple')
    for coordinate in value:
        _coordinate(coordinate)
    _require(value[0] < value[2] and value[1] < value[3], 'bbox must have positive area')
    return value


def _source_intents(geometry):
    from clay.layout.spatial.intent_projection import replay_discrete_mosaic_intent
    from clay.layout.spatial.mosaic_boundary_joins import replay_mosaic_boundary_join_intent
    lattice, boundary = geometry.mosaic_lattice_intent, geometry.mosaic_boundary_intent
    if lattice is not None:
        replay_discrete_mosaic_intent(lattice)
        _require(bool(lattice.column_ranks), 'lattice lacks a complete source column rank inventory')
    if boundary is not None:
        replay_mosaic_boundary_join_intent(boundary, mosaic=lattice)


@dataclass(frozen=True, kw_only=True)
class MosaicGeometry:
    """Native artwork with frozen explicit evidence and owner-derived pin views.

    Native polygon objects use the existing backend currency; every consuming
    hash/replay still measures their current contents. Frozen fields alone do
    not attest that geometry or a caller-supplied source declaration is true.
    """

    polygons: tuple[LayerPolygon, ...]
    kind: str
    comp_bbox: tuple[int, int, int, int]
    fingers: tuple[_CCFinger, ...]
    joint_row_binding: JointRowGeometryBinding
    channel_inventory: RowChannelInventory
    mosaic_lattice_intent: DiscreteMosaicIntent | None = None
    mosaic_boundary_intent: MosaicBoundaryJoinIntent | None = None
    tech: InitVar[object] = None
    composite_members: tuple[str, ...] = field(init=False)
    _member_pin_records: tuple = field(init=False, repr=False)

    def __post_init__(self, tech=None):
        from clay.layout.joint_row_geometry import joint_d4_orientation, replay_joint_row_geometry
        from clay.layout.joint_row_terminals import joint_terminal_contracts, require_joint_representatives
        from clay.layout.occurrence_channels import replay_row_channel_inventory
        from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
        from clay.layout.tech import LayoutTech
        _require(type(self.polygons) is tuple and 0 < len(self.polygons) <= 262_144
                 and all(type(p) is LayerPolygon for p in self.polygons),
                 'polygons must be a nonempty exact native tuple')
        _require(type(self.kind) is str and self.kind in ('nmos', 'pmos'), 'MOS kind is unsupported')
        _bbox(self.comp_bbox)
        _require(type(self.fingers) is tuple and 0 < len(self.fingers) <= 262_144
                 and all(type(f) is _CCFinger for f in self.fingers),
                 'fingers must be a nonempty exact native tuple')
        _require(type(self.joint_row_binding) is JointRowGeometryBinding, 'joint binding is missing or foreign')
        _require(type(self.channel_inventory) is RowChannelInventory, 'channel inventory is missing or foreign')
        _source_intents(self)
        tech = LayoutTech.resolve(self.joint_row_binding.pdk_id) if tech is None else tech
        members = tuple(sorted({o.owner for o in self.joint_row_binding.occurrences}))
        for finger, occurrence in zip(self.fingers, self.joint_row_binding.occurrences):
            _require(type(finger.orientation) is str and finger.orientation in _D4, 'finger D4 is unavailable')
            _require(finger.orientation == joint_d4_orientation(occurrence.master_to_local),
                     'finger D4 orientation differs from its actual raw master frame')
            for value in (finger.x0, finger.x1, finger.cy):
                _coordinate(value)
            _require(finger.x0 <= finger.x1, 'finger span is reversed')
            _bbox(finger.gate_bbox_nm)
            gates = {s.source_net for s in self.joint_row_binding.expected_sites
                     if s.owner == finger.member and 'g' in s.allowed_roles}
            _require(gates == {finger.gate_net}, 'finger gate differs from its owner terminal ledger')
        inventory = replay_joint_row_geometry(self, members, tech=tech)
        replay_row_channel_inventory(geometry=self, consumed_devices=members)
        terminals = joint_terminal_contracts(inventory, grid=pdk_manufacturing_grid_nm(tech))
        records = tuple((owner, tuple((role, min(point for t in terminals
            if (t.member, t.role) == (owner, role) for point in t.accesses_nm))
            for role in sorted({t.role for t in terminals if t.member == owner}))) for owner in members)
        object.__setattr__(self, 'composite_members', members)
        object.__setattr__(self, '_member_pin_records', records)
        require_joint_representatives(self, inventory, tech=tech)

    def extent(self):
        points = tuple((_coordinate(v.x), _coordinate(v.y)) for p in self.polygons for v in p.polygon.vertices)
        _require(bool(points), 'native artwork is empty')
        xs, ys = zip(*points)
        return _bbox((min(xs), min(ys), max(xs), max(ys)))

    def member_pins(self, owner):
        for member, values in self._member_pin_records:
            if member == owner:
                return dict(values)
        raise KeyError('neutral mosaic member is absent: ' + str(owner))


def transform_mosaic_geometry(geometry, orientation, *, tech=None):
    """Rigidly transform all physical operands; retain source intents unchanged.

    The source lattice/boundary consumer must reject a pose that no longer
    satisfies its declaration. A physical transform never rewrites that source.
    """
    from clay.layout.joint_row_geometry import joint_d4_orientation
    from clay.layout.joint_row_terminals import require_joint_representatives
    from clay.layout.joint_row_transform import transform_joint_row_geometry
    from clay.layout.occurrence_channels import transform_row_channel_inventory
    _require(type(geometry) is MosaicGeometry, 'transform requires exact MosaicGeometry')
    _require(type(orientation) is str and orientation in _D4, 'transform requires exact D4 orientation')
    # Check retained views before rebuilding, so reconstruction cannot erase a
    # corrupted representative or source-owner denominator.
    _require(type(geometry.joint_row_binding) is JointRowGeometryBinding, 'joint binding is missing or foreign')
    require_joint_representatives(geometry, geometry.joint_row_binding.joint_inventory, tech=tech)
    original = replace(geometry, tech=tech)
    _require(type(geometry.composite_members) is tuple and geometry.composite_members == original.composite_members,
             'composite source owner denominator differs')
    matrix = _D4[orientation]
    (a, b), (c, d) = matrix
    x0, y0, x1, y1 = original.extent()
    corners = tuple((a*x+b*y, c*x+d*y) for x, y in ((x0,y0), (x0,y1), (x1,y0), (x1,y1)))
    dx, dy = -min(x for x, _y in corners), -min(y for _x, y in corners)

    def point(x, y):
        return _coordinate(a*x+b*y+dx), _coordinate(c*x+d*y+dy)

    def box(bounds):
        left, bottom, right, top = _bbox(bounds)
        points = tuple(point(x, y) for x, y in ((left,bottom), (left,top), (right,bottom), (right,top)))
        return min(x for x, _y in points), min(y for _x, y in points), max(x for x, _y in points), max(y for _x, y in points)

    polygons = tuple(LayerPolygon(p.layer, p.datatype,
        Polygon([Point(*point(v.x, v.y)) for v in p.polygon.vertices])) for p in original.polygons)
    fingers = []
    for finger in original.fingers:
        (aa, ab), (ac, ad) = _D4[finger.orientation]
        pose = joint_d4_orientation((a*aa+b*ac, a*ab+b*ad, c*aa+d*ac, c*ab+d*ad, 0, 0))
        left, right = point(finger.x0, finger.cy), point(finger.x1, finger.cy)
        fingers.append(replace(finger, x0=min(left[0], right[0]), x1=max(left[0], right[0]),
            cy=(left[1]+right[1])//2, orientation=pose, gate_bbox_nm=box(finger.gate_bbox_nm)))
    channel = transform_row_channel_inventory(geometry=original, consumed_devices=original.composite_members,
        matrix=matrix, translation_nm=(dx, dy), row_polygons=polygons)
    target = SimpleNamespace(polygons=polygons, fingers=tuple(fingers), channel_inventory=channel)
    transform_joint_row_geometry(original, target, matrix=matrix, translation_nm=(dx, dy))
    return MosaicGeometry(polygons=polygons, kind=original.kind, comp_bbox=box(original.comp_bbox),
        fingers=tuple(fingers), joint_row_binding=target.joint_row_binding, channel_inventory=channel,
        mosaic_lattice_intent=original.mosaic_lattice_intent,
        mosaic_boundary_intent=original.mosaic_boundary_intent, tech=tech)


__all__ = ['MosaicGeometry', 'transform_mosaic_geometry']
