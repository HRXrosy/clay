"""Source-bound PMOS well access over current raw rows and PDK tap material.

This local boundary proves body-support coverage and physical access, not a
routed body net, a guard declaration, device LVS, DRC, or whole-circuit admission.
Raw-master provenance remains the row producer's responsibility. No NMOS or
deep-well isolation conclusion is emitted. Native work remains session-owned.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import klayout.db as kdb

from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.connectivity import ConnectModel, DerivedLayer, extract_nodes
from clay.layout.joint_row_geometry import (
    JointRowGeometryBinding, _region, _shapes, joint_checkpoint, replay_joint_row_geometry,
)
from clay.layout.joint_terminal_inventory import _geometry, _native_frame, _polygons
from clay.layout.occurrence_artwork import polygon_records

_SEAL = object()
_NOD, _POD = (65003, 1), (65003, 2)
_MAX_ITEMS, _MAX_VERTICES, _MAX_PAIRS = 4096, 1_048_576, 2_000_000


def _require(condition, message):
    if not condition:
        raise ValueError('well access ' + message)


def _tuple(value, field, *, nonempty=True, maximum=_MAX_ITEMS):
    _require(type(value) is tuple and len(value) <= maximum and (value or not nonempty),
             field + ' requires a bounded exact tuple')
    return value


def _text(value, field):
    _require(type(value) is str and 0 < len(value) <= 4096 and value == value.strip(), field + ' is malformed')


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def _records(polygons):
    check = joint_checkpoint()
    _tuple(polygons, 'actual artwork', maximum=262_144)
    vertices = 0
    for polygon in polygons:
        check()
        _require(type(polygon) is LayerPolygon, 'actual artwork has a foreign polygon')
        _require(type(polygon.polygon) is Polygon and type(polygon.polygon.vertices) is list
                 and 3 <= len(polygon.polygon.vertices) <= 4096,
                 'actual artwork contour is malformed')
        _require(type(polygon.layer) is int and type(polygon.datatype) is int
                 and 0 <= polygon.layer <= 65535 and 0 <= polygon.datatype <= 65535,
                 'actual artwork has a foreign layer')
        vertices += len(polygon.polygon.vertices)
        _require(vertices <= _MAX_VERTICES, 'actual artwork exceeds its vertex bound')
        _require(all(type(v) is Point and type(v.x) is int and type(v.y) is int and max(abs(v.x), abs(v.y)) <= 1_000_000_000
                     for v in polygon.polygon.vertices), 'actual artwork must use exact bounded integer nm')
    return tuple(sorted(polygon_records(polygons)))


def _from_records(records):
    return tuple(LayerPolygon(layer, datatype, Polygon([Point(x, y) for x, y in points]))
                 for layer, datatype, points in records)


def _regions(polygons):
    _records(polygons)
    check = joint_checkpoint()
    result = {}
    for polygon in polygons:
        check()
        shape = kdb.Polygon([kdb.Point(v.x, v.y) for v in polygon.polygon.vertices])
        _require(shape.area() > 0, 'actual artwork contains a degenerate polygon')
        result.setdefault((polygon.layer, polygon.datatype), kdb.Region()).insert(shape)
    for key, region in result.items():
        check()
        result[key] = region.merged()
    check()
    return result


def _artwork(regions):
    return tuple((key, _shapes(region)) for key, region in sorted(regions.items()) if not region.is_empty())


def _bbox(region):
    _require(not region.is_empty(), 'protected artwork is missing')
    box = region.bbox()
    return box.left, box.bottom, box.right, box.top


def _union(regions):
    result = kdb.Region()
    for region in regions:
        joint_checkpoint()()
        result += region
    return result.merged()


def _source(cell):
    from clay.layout.route.exact_artifact_evidence import (
        _canonical_source_member_inventory, _replayed_source_cell_authority,
    )
    from clay.layout.place.row_program import _normalize_records
    digest, inventory_sha, _classes, canonical = _replayed_source_cell_authority(cell)
    _require(canonical is not None, 'current canonical source is missing')
    from clay.parser import parse_clay_file
    from clay.layout.tech import LayoutTech
    path = Path(canonical.source_span['file']).resolve()
    before = path.read_bytes()
    parsed = parse_clay_file(path)
    _require(hashlib.sha256(before).hexdigest() == digest and path.read_bytes() == before,
             'source changed during PDK declaration replay')
    _require(type(parsed.pdk_id) is str and bool(parsed.pdk_id), 'source PDK declaration is missing')
    source_pdk = LayoutTech.resolve(parsed.pdk_id).adapter.pdk_id
    return (digest, inventory_sha, _canonical_source_member_inventory(canonical), canonical,
            _normalize_records(canonical, None), source_pdk)


def _technology(tech):
    from clay.layout.device_cell import _rules_from_tech
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.tech import LayoutTech
    _require(type(tech) is LayoutTech, 'current PDK type is foreign')
    tech.rules
    _require(tech.bulk_topology().get('implant_distinguished', True),
             'PMOS implanted-well model capability is unsupported')
    keys = {name: tech.build_layer(name) for name in ('diff', 'poly', 'nplus', 'pplus', 'nwell')}
    keys.update(contact=tech.contact_key(), metal1=tech.metal_key(1))
    _require(all(type(key) is tuple and len(key) == 2 and all(type(n) is int for n in key)
                 for key in keys.values()), 'current PDK well/tap layers are unsupported')
    routing = ConnectModel.from_tech(tech)
    _require(not {_NOD, _POD}.intersection((*keys.values(), *routing.metals,
                 *(via[0] for via in routing.vias))), 'derived well-access layers collide with current PDK')
    model = ConnectModel(metals=(*routing.metals, keys['poly'], _NOD, _POD), vias=routing.vias,
        extra_connect=tuple((keys['contact'], conductor, keys['metal1']) for conductor in (keys['poly'], _NOD, _POD)),
        derived=(DerivedLayer(_NOD, (('and', keys['diff'], keys['nplus']), ('not', '$0', keys['poly']))),
                 DerivedLayer(_POD, (('and', keys['diff'], keys['pplus']), ('not', '$0', keys['poly'])))))
    rules = _rules_from_tech(tech)
    snapshot = tech.adapter.source_snapshot(refresh=True)
    identity = (tech.pdk_id, pdk_provenance_digest(tech),
                None if snapshot is None else snapshot.snapshot_sha256,
                None if snapshot is None else snapshot.installed_authority_sha256,
                tuple(sorted(keys.items())), _sha(asdict(rules)), _sha(asdict(model)), _sha(asdict(routing)))
    return identity, keys, rules, model, routing


@dataclass(frozen=True, slots=True)
class JointWellRow:
    row_id: str
    geometry: object

    def __post_init__(self):
        _text(self.row_id, 'row id')
        _require(type(getattr(self.geometry, 'joint_row_binding', None)) is JointRowGeometryBinding,
                 'row lacks its exact joint geometry binding')


@dataclass(frozen=True, slots=True)
class WellForeignArtwork:
    """Untrusted complete physical obstacle; no source/net/domain permission."""
    stamp_id: str
    polygon_records: tuple

    def __post_init__(self):
        _text(self.stamp_id, 'foreign stamp id')
        _tuple(self.polygon_records, 'foreign polygon records', maximum=262_144)
        for record in self.polygon_records:
            _require(type(record) is tuple and len(record) == 3 and type(record[2]) is tuple
                and all(type(point) is tuple and len(point) == 2 for point in record[2]),
                'foreign polygon records must be immutable')
        _records(_from_records(self.polygon_records))


def _foreign_state(values):
    _tuple(values, 'foreign artwork inventory', nonempty=False)
    _require(all(type(value) is WellForeignArtwork for value in values), 'foreign artwork type differs')
    result = []
    for value in values:
        joint_checkpoint()()
        value.__post_init__()
        result.append((value.stamp_id, _records(_from_records(value.polygon_records))))
    _require(len({name for name, _ in result}) == len(result), 'duplicate foreign stamp identity')
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class PdkWellTapRequest:
    tap_id: str
    occurrence_ids: tuple[str, ...]

    def __post_init__(self):
        _text(self.tap_id, 'tap id')
        _tuple(self.occurrence_ids, 'tap occurrence ids')
        for occurrence in self.occurrence_ids:
            _text(occurrence, 'tap occurrence')
        _require(self.occurrence_ids == tuple(sorted(set(self.occurrence_ids))), 'tap occurrence ids are duplicated or unordered')


@dataclass(frozen=True, slots=True, init=False)
class PdkWellTapConstruction:
    request: PdkWellTapRequest
    source_sha256: str
    source_inventory_sha256: str
    row_inputs_sha256: str
    technology_identity: tuple
    body_net: str
    protected_bboxes_nm: tuple
    polygon_records: tuple
    geometry_by_layer_nm: tuple
    _content_sha256: str
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError('PdkWellTapConstruction requires current PDK materialization')

    def payload(self):
        data = {name: asdict(self.request) if name == 'request' else getattr(self, name)
                for name in self.__dataclass_fields__ if not name.startswith('_')}
        data['geometry_by_layer_nm'] = tuple((key, tuple(asdict(shape) for shape in shapes))
                                            for key, shapes in self.geometry_by_layer_nm)
        return {'schema': 'clay.pdk-well-tap-construction/v1', **data}

    def validate(self):
        _require(self._seal is _SEAL and _sha(self.payload()) == self._content_sha256, 'tap construction binding changed')
        _require(type(self.request) is PdkWellTapRequest, 'tap request type changed')
        self.request.__post_init__()
        _geometry(self.geometry_by_layer_nm, 'tap construction artwork')

    @property
    def sha256(self):
        self.validate()
        return self._content_sha256

    @property
    def polygons(self):
        self.validate()
        return _from_records(self.polygon_records)


@dataclass(frozen=True, slots=True)
class PmosBodySupport:
    support_id: str
    occurrence_id: str
    owner: str
    body_net: str
    variant_sha256: str
    master_to_local: tuple
    geometry_nm: tuple
    raw_nwell_geometry_nm: tuple
    component_id: str


@dataclass(frozen=True, slots=True)
class WellAccessComponent:
    component_id: str
    body_net: str
    geometry_nm: tuple
    support_ids: tuple[str, ...]
    access_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WellRoutingAccess:
    access_id: str
    component_id: str
    physical_component_sha256: str
    geometry_by_layer_nm: tuple


@dataclass(frozen=True, slots=True, init=False)
class WellComponentAccessBinding:
    source_sha256: str
    source_inventory_sha256: str
    source_member_inventory: tuple
    covered_members: tuple[str, ...]
    uncovered_source_members: tuple[str, ...]
    technology_identity: tuple
    row_inputs_sha256: str
    addition_sha256s: tuple[str, ...]
    body_supports: tuple[PmosBodySupport, ...]
    well_components: tuple[WellAccessComponent, ...]
    routing_accesses: tuple[WellRoutingAccess, ...]
    artwork: tuple
    artwork_multiset_sha256: str
    foreign_artwork_records: tuple
    scope: str
    _content_sha256: str
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError('WellComponentAccessBinding requires current source/native capture')

    def payload(self):
        data = {name: getattr(self, name) for name in self.__dataclass_fields__ if not name.startswith('_')}
        for name in ('body_supports', 'well_components', 'routing_accesses'):
            data[name] = tuple(asdict(record) for record in data[name])
        data['artwork'] = tuple((key, tuple(asdict(shape) for shape in shapes)) for key, shapes in self.artwork)
        if not self.foreign_artwork_records:
            del data['foreign_artwork_records']
        return {'schema': 'clay.well-component-access/v2' if self.foreign_artwork_records
                else 'clay.well-component-access/v1', **data}

    def validate(self):
        _require(self._seal is _SEAL and _sha(self.payload()) == self._content_sha256, 'captured well access binding changed')
        _require(self.scope == 'pmos_body_access_only', 'well access scope changed')
        _geometry(self.artwork, 'complete occupied artwork')
        for values, kind in ((self.body_supports, PmosBodySupport), (self.well_components, WellAccessComponent),
                             (self.routing_accesses, WellRoutingAccess)):
            _tuple(values, 'well access records')
            _require(all(type(value) is kind for value in values), 'well access record type changed')
        components = {component.component_id: component for component in self.well_components}
        supports = {support.support_id: support for support in self.body_supports}
        accesses = {access.access_id: access for access in self.routing_accesses}
        _require(len(components) == len(self.well_components) and len(supports) == len(self.body_supports)
                 and len(accesses) == len(self.routing_accesses), 'well component denominator changed')
        for component in components.values():
            _polygons(component.geometry_nm, 'well component')
            _require(component.support_ids and component.access_ids, 'well component has no support/access')
            _require(all(key in supports and supports[key].component_id == component.component_id for key in component.support_ids)
                     and all(key in accesses and accesses[key].component_id == component.component_id for key in component.access_ids),
                     'well AND/access OR association changed')

    @property
    def sha256(self):
        self.validate()
        return self._content_sha256


def _capture(kind, **fields):
    value = object.__new__(kind)
    for name, content in fields.items():
        object.__setattr__(value, name, content)
    object.__setattr__(value, '_seal', _SEAL)
    object.__setattr__(value, '_content_sha256', _sha(value.payload()))
    value.validate()
    return value


def _row_state(rows):
    """Retain actual bytes and typed operands without repeating extraction."""
    from clay.layout.occurrence_channels import RowChannelInventory
    result = []
    for row in rows:
        joint_checkpoint()()
        _require(type(row) is JointWellRow, 'row requires an exact JointWellRow')
        row.__post_init__()
        geometry = row.geometry
        channels = getattr(geometry, 'channel_inventory', None)
        _require(type(channels) is RowChannelInventory, 'row channel inventory is missing or foreign')
        result.append((row.row_id, geometry.kind, geometry.joint_row_binding.sha256,
            _records(tuple(geometry.polygons)), channels.sha256,
            _sha(tuple(asdict(finger) for finger in geometry.fingers))))
    return tuple(result)


def _unchanged_rows(prepared):
    _require(_row_state(prepared['rows']) == prepared['row_state'],
             'row artwork or native binding changed during well access work')
    _require(_foreign_state(prepared['foreign']) == prepared['foreign_state'],
             'foreign artwork changed during well access work')


def _prepare(source_cell, tech, rows, pmos_owners, foreign_artwork=()):
    check = joint_checkpoint()
    source = _source(source_cell)
    identity, keys, rules, model, routing = _technology(tech)
    _require(source[5] == tech.adapter.pdk_id, 'source PDK differs from materialized technology')
    _tuple(rows, 'rows')
    row_state = _row_state(rows)
    foreign_state = _foreign_state(foreign_artwork)
    _tuple(pmos_owners, 'selected owner scope')
    _require(all(type(name) is str for name in pmos_owners)
             and pmos_owners == tuple(sorted(set(pmos_owners))), 'selected owner scope is not canonical')
    members = {member.device: member for member in source[4]}
    _require(all(name in members for name in pmos_owners), 'selected owner is absent from current source')
    _require(all(members[name].kind == 'pmos' for name in pmos_owners), 'only PMOS well access is supported')
    occurrences, row_payloads, row_polygons, context_supports, seen_owners, row_ids = {}, [], [], [], set(), set()
    bindings = {}
    for row in rows:
        check()
        _require(type(row) is JointWellRow, 'row requires an exact JointWellRow')
        row.__post_init__()
        _require(row.row_id not in row_ids, 'duplicate row id')
        row_ids.add(row.row_id)
        geometry, binding = row.geometry, row.geometry.joint_row_binding
        bindings[row.row_id] = binding
        owners = tuple(sorted({occ.owner for occ in binding.occurrences}))
        _require(not seen_owners.intersection(owners) and set(owners) <= set(members), 'row source owner denominator differs')
        seen_owners.update(owners)
        _require(all(geometry.kind == members[owner].kind for owner in owners), 'row/source MOS kind differs')
        replay_joint_row_geometry(geometry, owners, tech=tech, members=members)
        row_polygons.extend(geometry.polygons)
        parts = {part.part_id: part for part in binding.parts}
        row_payloads.append((row.row_id, binding.sha256, _records(tuple(geometry.polygons)),
                             tuple(members[name].payload() for name in owners)))
        for occurrence in binding.occurrences:
            check()
            _require(occurrence.occurrence_id not in occurrences, 'duplicate source occurrence id')
            _require(len(occurrences) < _MAX_ITEMS, 'source occurrence count exceeds bound')
            member = members[occurrence.owner]
            raw = {key: _region(shapes).merged() for key, shapes in parts[occurrence.raw_part_id].geometry_by_layer_nm}
            channel = raw.get(keys['diff'], kdb.Region()) & raw.get(keys['poly'], kdb.Region())
            channels = tuple(channel.each_merged())
            _require(len(channels) == 1, 'raw channel multiplicity is unsupported')
            local = channel.transformed(_native_frame(occurrence.master_to_local).inverted()).bbox()
            _require(sorted((local.width(), local.height())) == sorted((member.length_nm, member.width_nm))
                     and channel.area() == member.length_nm*member.width_nm, 'raw channel width/length differs from source member')
            implant = raw.get(keys['pplus' if member.kind == 'pmos' else 'nplus'], kdb.Region())
            _require((channel-implant).is_empty(), 'raw channel implant differs from source kind')
            occurrence_data = (row, occurrence, member, raw, channel)
            occurrences[occurrence.occurrence_id] = occurrence_data
            if member.kind == 'pmos':
                well = raw.get(keys['nwell'], kdb.Region())
                _require(member.body_net and not well.is_empty() and (channel-well).is_empty(),
                         'PMOS raw body-support lacks its declared body/NW coverage')
                context_supports.append(occurrence_data)
    _require(set(pmos_owners) <= seen_owners, 'selected source owner is missing its complete row')
    _require(all(sum(data[1].owner == owner for data in occurrences.values()) == members[owner].fingers
                 for owner in pmos_owners), 'source unit count differs from selected row population')
    _require(_row_state(rows) == row_state, 'row artwork or native binding changed during source replay')
    polygons = _from_records(_records(tuple(row_polygons)))
    _require(not row_ids.intersection(name for name, _ in foreign_state), 'foreign stamp duplicates a source row')
    foreign_polygons = _from_records(tuple(record for _, records in foreign_state for record in records))
    return dict(source=source, identity=identity, keys=keys, rules=rules, model=model, routing=routing,
                rows=rows, owners=pmos_owners, occurrences=occurrences, supports=context_supports,
                row_sha=_sha(tuple(sorted(row_payloads))), polygons=polygons, check=check,
                row_state=row_state, bindings=bindings, foreign=foreign_artwork, foreign_state=foreign_state,
                foreign_polygons=foreign_polygons)


def _construct(prepared, tech, request):
    from clay.layout.guard_ring import build_guard_ring
    _require(type(request) is PdkWellTapRequest, 'tap request requires its exact type')
    request.__post_init__()
    _require(set(request.occurrence_ids) <= set(prepared['occurrences']), 'tap request names a missing occurrence')
    data = tuple(prepared['occurrences'][name] for name in request.occurrence_ids)
    _require(all(item[1].owner in prepared['owners'] for item in data), 'tap request names an uncovered source owner')
    bodies = {item[2].body_net for item in data}
    _require(len(bodies) == 1, 'tap request mixes declared body domains')
    selected_parts = {}
    for row, occurrence, _member, _raw, _channel in data:
        for part in prepared['bindings'][row.row_id].parts:
            if occurrence.occurrence_id in part.occurrence_ids:
                selected_parts[(row.row_id, part.part_id)] = part
    selected = {}
    for part in selected_parts.values():
        for key, shapes in part.geometry_by_layer_nm:
            selected[key] = selected.get(key, kdb.Region()) + _region(shapes)
    keys = prepared['keys']
    boxes = tuple(_bbox(region) for region in (_union(selected.values()),
        selected[keys['diff']], selected[keys['poly']], selected[keys['metal1']]))
    body, = bodies
    prepared['check']()
    ring = build_guard_ring(tech, boxes[0], well='n', nets={'tap': body}, shared_well=True,
        protected_active_bbox=boxes[1], protected_poly_bbox=boxes[2], protected_metal1_bbox=boxes[3],
        protected_same_implant_bbox=(
            _bbox(selected[keys['nplus']])
            if keys['nplus'] in selected and not selected[keys['nplus']].is_empty() else None))
    prepared['check']()
    polygons = tuple(ring.polygons)
    return _capture(PdkWellTapConstruction, request=request, source_sha256=prepared['source'][0],
        source_inventory_sha256=prepared['source'][1], row_inputs_sha256=prepared['row_sha'],
        technology_identity=prepared['identity'], body_net=body, protected_bboxes_nm=boxes,
        polygon_records=_records(polygons), geometry_by_layer_nm=_artwork(_regions(polygons)))


def _tap_interactions(prepared, additions):
    keys = prepared['keys']
    raw = _regions(prepared['polygons'])
    foreign = _regions(prepared['foreign_polygons']) if prepared['foreign_polygons'] else {}
    for key, region in foreign.items():
        raw[key] = raw.get(key, kdb.Region()) + region
    # An uncovered source class is not permission to change its well domain.
    # This boundary has no NMOS/deep-well isolation model, so newly added NW
    # must remain disjoint from every actual NMOS channel in the supplied rows.
    nmos_channels = _union(item[4] for item in prepared['occurrences'].values()
                           if item[2].kind == 'nmos')
    for addition in additions:
        prepared['check']()
        tap = _regions(addition.polygons)
        _require(tap[keys['nwell']].interacting(foreign.get(keys['diff'], kdb.Region())).is_empty()
            and tap[keys['nwell']].interacting(foreign.get(keys['nwell'], kdb.Region())).is_empty(),
            'unsupported_foreign_body_context: tap well touches unbound foreign OD/NW domain')
        _require(tap[keys['nwell']].interacting(nmos_channels).is_empty(),
                 'new NWell covers an NMOS channel without supported isolation')
        active = tap[keys['diff']]
        _require(active.interacting(raw.get(keys['diff'], kdb.Region())).is_empty(), 'tap overlaps foreign raw active')
        _require(active.interacting(raw.get(keys['poly'], kdb.Region())).is_empty(), 'tap overlaps foreign raw poly')
        _require(tap[keys['nplus']].interacting(raw.get(keys['diff'], kdb.Region()) &
            raw.get(keys['pplus'], kdb.Region())).is_empty(), 'tap implant overlaps foreign P+OD')


def materialize_pmos_component_taps(source_cell, *, tech, rows, pmos_owners, requests, foreign_artwork=()):
    prepared = _prepare(source_cell, tech, rows, pmos_owners, foreign_artwork)
    _tuple(requests, 'tap requests')
    _require(all(type(request) is PdkWellTapRequest for request in requests)
             and len({request.tap_id for request in requests}) == len(requests), 'tap request id/type differs')
    result = tuple(_construct(prepared, tech, request) for request in requests)
    _tap_interactions(prepared, result)
    _require(_source(source_cell)[:3] == prepared['source'][:3] and _technology(tech)[0] == prepared['identity'],
             'source/PDK changed during tap materialization')
    _unchanged_rows(prepared)
    return result


def _node_hits(nodes, key, region, *, check):
    _require(len(nodes) <= _MAX_ITEMS, 'component extraction exceeds node bound')
    hits, coverage = [], kdb.Region()
    for index, node in enumerate(nodes):
        check()
        overlap = node.get(key, kdb.Region()) & region
        if not overlap.is_empty():
            hits.append(index)
            coverage += overlap
    _require(hits and (region-coverage).is_empty(), 'complete raw site or tap access has uncovered material')
    return tuple(hits)


def _component_access(prepared, additions, polygons):
    keys, check = prepared['keys'], prepared['check']
    scene = _regions(polygons)
    foreign = _regions(prepared['foreign_polygons']) if prepared['foreign_polygons'] else {}
    wells = tuple(kdb.Region(poly) for poly in scene[keys['nwell']].each_merged())
    _require(len(wells)*len(prepared['supports']) <= _MAX_PAIRS, 'well/support comparison budget exceeded')
    support_groups = {index: [] for index in range(len(wells))}
    for item in prepared['supports']:
        check()
        channel, raw_well = item[4], item[3][keys['nwell']]
        hits = tuple(index for index, well in enumerate(wells) if not (well & channel).is_empty())
        _require(len(hits) == 1 and (raw_well-wells[hits[0]]).is_empty(), 'complete raw body-support escapes one well component')
        support_groups[hits[0]].append(item)
    for index, items in support_groups.items():
        if not items:
            _require((wells[index]-foreign.get(keys['nwell'], kdb.Region())).is_empty(),
                     'well component has no independently bound source body-support')
            continue  # Independently recorded foreign domain, never a body-access grant.
        _require(len({item[2].body_net for item in items}) == 1, 'well component mixes source body domains')
        _require(wells[index].interacting(foreign.get(keys['nwell'], kdb.Region())).is_empty()
            and wells[index].interacting(foreign.get(keys['diff'], kdb.Region())).is_empty(),
            'unsupported_foreign_body_context: source well touches unbound foreign OD/NW domain')
    shapes = {key: tuple(region.each()) for key, region in scene.items()}
    check()
    nodes = extract_nodes(shapes, prepared['model'])
    check()
    routing_nodes = extract_nodes(shapes, prepared['routing'])
    check()
    _require(len(nodes)*len(prepared['occurrences'])*3 <= _MAX_PAIRS, 'site/node comparison budget exceeded')
    net_sets = {index: set() for index in range(len(nodes))}
    for binding in prepared['bindings'].values():
        placed = {site.site_id: site for site in binding.placed_sites}
        for site in binding.expected_sites:
            member = prepared['occurrences'][site.occurrence_id][2]
            key = keys['poly'] if site.allowed_roles == ('g',) else _POD if member.kind == 'pmos' else _NOD
            for index in _node_hits(nodes, key, _region(placed[site.site_id].geometry_nm), check=check):
                net_sets[index].add(site.source_net)
    tap_nodes = {index: set() for index in range(len(wells))}
    for addition in additions:
        tap = _regions(addition.polygons)
        active = (tap[keys['diff']] & tap[keys['nplus']]) - scene.get(keys['poly'], kdb.Region())
        for poly in active.each_merged():
            check()
            piece = kdb.Region(poly)
            hits = tuple(index for index, well in enumerate(wells) if (piece-well).is_empty())
            _require(len(hits) == 1, 'tap active lacks unique complete NW enclosure')
            well_index, = hits
            _require({item[2].body_net for item in support_groups[well_index]} == {addition.body_net},
                     'tap material names a foreign body domain')
            cuts = scene[keys['contact']].interacting(piece)
            _require(not cuts.is_empty() and (cuts-piece).is_empty()
                     and (cuts-scene[keys['metal1']]).is_empty(), 'tap component lacks complete contact/M1 access')
            for index in _node_hits(nodes, _NOD, piece, check=check):
                tap_nodes[well_index].add(index)
                net_sets[index].add(addition.body_net)
    _require(all(len(nets) <= 1 for nets in net_sets.values()), 'tap material shorts distinct DSG/body nets')
    if foreign:
        from clay.layout.connectivity import materialise_derived
        foreign_conductors = materialise_derived(
            {key: tuple(region.each()) for key, region in foreign.items()}, prepared['model'].derived)
        for node_id in set().union(*tap_nodes.values()):
            check()
            node = nodes[node_id]
            touches = any(region.interacting(foreign_conductors.get(key, kdb.Region())).is_empty() is False
                          for key, region in node.items())
            # A foreign cut remains an unknown interface even when it has no
            # independently labelled conductor on the opposite endpoint.
            touches |= any(not foreign.get(cut, kdb.Region()).interacting(
                node.get(low, kdb.Region())+node.get(high, kdb.Region())).is_empty()
                for cut, low, high in (*prepared['model'].vias, *prepared['model'].extra_connect))
            _require(not touches, 'unsupported_foreign_body_context: tap reaches unbound foreign conductor/contact/via')
    supports, components, accesses = [], [], []
    for index, well in enumerate(wells):
        check()
        selected = tuple(item for item in support_groups[index] if item[1].owner in prepared['owners'])
        if not selected:
            continue  # Uncovered source classes remain explicit; no body-access conclusion.
        body = selected[0][2].body_net
        geometry = _shapes(well)
        component_id = 'well:'+_sha((body, tuple(asdict(shape) for shape in geometry)))
        alternatives = {}
        for node_id in tap_nodes[index]:
            for routing in routing_nodes:
                check()
                if any(not (region & nodes[node_id].get(key, kdb.Region())).is_empty() for key, region in routing.items()):
                    art = _artwork(routing)
                    physical = _sha(tuple((key, tuple(asdict(shape) for shape in ss)) for key, ss in art))
                    access_id = 'well-access:'+_sha((component_id, physical))
                    alternatives[access_id] = WellRoutingAccess(access_id, component_id, physical, art)
        _require(alternatives, 'well component has no physical tap routing access')
        support_ids = []
        for _row, occurrence, member, raw, channel in selected:
            support_id = occurrence.occurrence_id+'/body-support'
            support_ids.append(support_id)
            supports.append(PmosBodySupport(support_id, occurrence.occurrence_id, occurrence.owner,
                member.body_net, occurrence.master.variant_sha256, occurrence.master_to_local,
                _shapes(channel), _shapes(raw[keys['nwell']]), component_id))
        components.append(WellAccessComponent(component_id, body, geometry, tuple(sorted(support_ids)), tuple(sorted(alternatives))))
        accesses.extend(alternatives.values())
    return tuple(sorted(supports, key=lambda s: s.support_id)), tuple(sorted(components, key=lambda c: c.component_id)), tuple(sorted(accesses, key=lambda a: a.access_id))


def capture_well_component_access(source_cell, *, tech, rows, pmos_owners, additions, actual_polygons,
                                  foreign_artwork=()):
    actual_records = _records(actual_polygons)
    frozen_polygons = _from_records(actual_records)
    prepared = _prepare(source_cell, tech, rows, pmos_owners, foreign_artwork)
    _tuple(additions, 'tap additions', nonempty=False)
    _require(all(type(addition) is PdkWellTapConstruction for addition in additions), 'tap addition lacks PDK construction binding')
    _require(len({addition.request.tap_id for addition in additions}) == len(additions), 'duplicate tap construction id')
    for addition in additions:
        addition.validate()
        expected = _construct(prepared, tech, addition.request)
        _require(expected.sha256 == addition.sha256, 'tap construction differs from current source/PDK/row material')
    expected_polygons = (prepared['polygons'] + tuple(p for addition in additions for p in addition.polygons)
                         + prepared['foreign_polygons'])
    _require(Counter(actual_records) == Counter(_records(expected_polygons)),
             'actual raw/tap artwork multiset differs from bound material')
    _tap_interactions(prepared, additions)
    supports, components, accesses = _component_access(prepared, additions, frozen_polygons)
    _require(len(supports) == sum(member.fingers for member in prepared['source'][4] if member.device in pmos_owners),
             'source PMOS body-support count differs')
    source = prepared['source']
    result = _capture(WellComponentAccessBinding, source_sha256=source[0], source_inventory_sha256=source[1],
        source_member_inventory=source[2], covered_members=pmos_owners,
        uncovered_source_members=tuple(sorted(row[0] for row in source[2] if row[0] not in pmos_owners)),
        technology_identity=prepared['identity'], row_inputs_sha256=prepared['row_sha'],
        addition_sha256s=tuple(addition.sha256 for addition in additions), body_supports=supports,
        well_components=components, routing_accesses=accesses, artwork=_artwork(_regions(frozen_polygons)),
        artwork_multiset_sha256=_sha(actual_records), foreign_artwork_records=prepared['foreign_state'],
        scope='pmos_body_access_only')
    prepared['check']()
    _require(_source(source_cell)[:3] == source[:3] and _technology(tech)[0] == prepared['identity'],
             'source/PDK changed during well access capture')
    _unchanged_rows(prepared)
    _require(_records(actual_polygons) == actual_records,
             'actual artwork changed during native well access extraction')
    return result


def replay_well_component_access(value, source_cell, *, tech, rows, pmos_owners, additions, actual_polygons,
                                foreign_artwork=()):
    _require(type(value) is WellComponentAccessBinding, 'captured well-access binding type differs')
    value.validate()
    observed = capture_well_component_access(source_cell, tech=tech, rows=rows, pmos_owners=pmos_owners,
        additions=additions, actual_polygons=actual_polygons, foreign_artwork=foreign_artwork)
    _require(observed.sha256 == value.sha256, 'current source/native well access differs from retained binding')
    return value
