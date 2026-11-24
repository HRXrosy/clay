"""Current fixed-BJT local assemblies with independently replayed C/B/E sites.

The assembly owns local offsets, full per-member artwork and terminal operands.
No reference translation reaches construction. This boundary proves local raw
artwork and terminal connectivity only: no body/well, device LVS, DRC, routing or
placement admission. Native budgets and cancellation remain caller-owned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import klayout.db as kdb

from clay.frontend.unit_master_geometry import UnitMasterGeometry, validate_unit_master_regions
from clay.layout.connectivity import ConnectModel, DerivedLayer, extract_nodes, node_index_at
from clay.layout.joint_terminal_inventory import (
    JointTerminalLimits, OccurrenceTerminalSite, TerminalSiteRequirement,
    compile_joint_terminal_inventory, _geometry, _model)
from clay.layout.nonmos_local_assembly import (
    NonmosAssemblySource, _deep, checkpoint, digest, replay_nonmos_assembly_source, require,
    technology_identity)
from clay.layout.occurrence_channels import _shape_keys
from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM

_SEAL = object()
_NOD, _POD = (65000, 2), (65000, 3)


def current_bjt_assembly_models(tech):
    """Complete routing stack and implanted OD/poly contacts for fixed BJTs.

These are the explicit BJT-local conductors; no resistor or MIM terminal
recognition, substrate or well connectivity is claimed by this model.
"""
    checkpoint()
    routing = ConnectModel.from_tech(tech)
    keys = {name: tech.build_layer(name) for name in ('poly', 'diff', 'nplus', 'pplus')}
    require(all(type(key) is tuple and len(key) == 2 for key in keys.values()),
            'PDK BJT diffusion/contact model capability is unavailable')
    poly, diff, ni, pi = (keys[name] for name in ('poly', 'diff', 'nplus', 'pplus'))
    contact, m1 = tech.contact_key(), tech.metal_key(1)
    require(not {_NOD, _POD}.intersection((*routing.metals, *keys.values(), contact)),
            'PDK BJT derived layer identity collides with real layers')
    connect = ConnectModel(metals=(*routing.metals, poly, _NOD, _POD), vias=routing.vias,
        extra_connect=tuple((contact, conductor, m1) for conductor in (poly, _NOD, _POD)),
        derived=(DerivedLayer(_NOD, (('and', diff, ni), ('not', '$0', poly))),
                 DerivedLayer(_POD, (('and', diff, pi), ('not', '$0', poly)))))
    _model(connect)
    _model(routing)
    return connect, routing


@dataclass(frozen=True, slots=True)
class BjtAssemblyTerminalModel:
    master: UnitMasterGeometry
    site_queries: tuple[tuple[str, tuple[int, int]], ...]
    connect_model: ConnectModel
    routing_model: ConnectModel
    provenance_sha256: str

    def __post_init__(self):
        require(type(self.master) is UnitMasterGeometry, 'raw master type differs')
        self.master.__post_init__()
        require(type(self.site_queries) is tuple and len(self.site_queries) == 3
            and tuple(row[0] for row in self.site_queries) == tuple(site.site_id for site in self.master.terminal_sites),
            'site/query tuple denominator differs')
        require(all(type(row) is tuple and len(row) == 2 and type(row[0]) is str
            and type(row[1]) is tuple and len(row[1]) == 2 and all(type(n) is int and 0 <= n <= 65535 for n in row[1])
            for row in self.site_queries), 'site query layer schema differs')
        require(sorted(site.role_group for site in self.master.terminal_sites) == ['b', 'c', 'e'],
                'raw master requires independent complete c/b/e sites')
        _model(self.connect_model)
        _model(self.routing_model)
        require(type(self.provenance_sha256) is str and len(self.provenance_sha256) == 64
            and all(c in '0123456789abcdef' for c in self.provenance_sha256), 'PDK provenance digest differs')


@dataclass(frozen=True, slots=True)
class NativeNonmosOccurrence:
    occurrence_id: str
    owner: str
    ordinal: int
    master_variant_sha256: str
    master_to_local: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class NativeNonmosMember:
    owner: str
    kind: str
    unit_count: int
    bbox_nm: tuple[int, int, int, int]
    artwork: tuple
    raw_occurrence_artwork: tuple
    added_artwork: tuple


@dataclass(frozen=True, slots=True)
class NativeNonmosExternalAccess:
    owner: str
    role: str
    layer: tuple[int, int]
    points_nm: tuple[tuple[int, int], ...]
    component_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True, init=False)
class NativeNonmosAssemblyBinding:
    source: NonmosAssemblySource
    terminals: BjtAssemblyTerminalModel
    members: tuple[NativeNonmosMember, ...]
    owner_offsets_nm: tuple[tuple[str, tuple[int, int]], ...]
    occurrences: tuple[NativeNonmosOccurrence, ...]
    external_accesses: tuple[NativeNonmosExternalAccess, ...]
    joint_inventory: object
    bbox_nm: tuple[int, int, int, int]
    _content_sha256: str
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError('NativeNonmosAssemblyBinding requires current materialization')

    def payload(self):
        return {'schema': 'clay.native-nonmos-assembly/v1', 'source_sha256': self.source.sha256,
            'terminals': asdict(self.terminals), 'members': tuple(asdict(m) for m in self.members),
            'owner_offsets_nm': self.owner_offsets_nm, 'occurrences': tuple(asdict(o) for o in self.occurrences),
            'external_accesses': tuple(asdict(a) for a in self.external_accesses),
            'joint_inventory': asdict(self.joint_inventory), 'bbox_nm': self.bbox_nm}

    @property
    def sha256(self):
        self.validate()
        return self._content_sha256

    def validate(self):
        """Pure sealed-data consistency; current source/native replay is separate."""
        require(self._seal is _SEAL and digest(self.payload()) == self._content_sha256,
                'native captured artwork/ledger identity changed')
        require(type(self.source) is NonmosAssemblySource and type(self.terminals) is BjtAssemblyTerminalModel,
                'native binding source/terminal schema differs')
        self.terminals.__post_init__()
        self.joint_inventory.__post_init__()
        for values, kind in ((self.members, NativeNonmosMember), (self.occurrences, NativeNonmosOccurrence),
                              (self.external_accesses, NativeNonmosExternalAccess)):
            require(type(values) is tuple and values and all(type(row) is kind for row in values),
                    'native binding record tuple differs')
        require(tuple(m.owner for m in self.members) == tuple(m.owner for m in self.source.members)
            == tuple(owner for owner, _offset in self.owner_offsets_nm), 'native binding member denominator differs')
        require(len(self.occurrences) == sum(m.unit_count for m in self.source.members)
            and len(self.joint_inventory.expected_sites) == 3*len(self.occurrences)
            and len(self.external_accesses) == 3*len(self.members), 'native binding occurrence/site denominator differs')
        for member in self.members:
            _geometry(member.artwork, 'native member artwork')
            require(type(member.raw_occurrence_artwork) is tuple
                and len(member.raw_occurrence_artwork) == member.unit_count, 'native raw occurrence denominator differs')
            for artwork in member.raw_occurrence_artwork:
                _geometry(artwork, 'native raw artwork')
            if member.added_artwork:
                _geometry(member.added_artwork, 'native added artwork')


def _region(polygons):
    region = kdb.Region()
    for polygon in polygons:
        checkpoint()
        region += polygon.to_region()
    return region.merged()


def _shapes(region):
    checkpoint()
    return tuple(PhysicalTerminalPolygonNM(hull, holes) for hull, holes in _shape_keys(region))


def _layers(polygons):
    result = {}
    for lp in polygons:
        checkpoint()
        points = []
        for point in lp.polygon.vertices:
            require(float(point.x).is_integer() and float(point.y).is_integer(), 'native raw artwork is not integer nm')
            points.append(kdb.Point(int(point.x), int(point.y)))
        result.setdefault((lp.layer, lp.datatype), kdb.Region()).insert(kdb.Polygon(points))
    return tuple((key, _shapes(region)) for key, region in sorted(result.items()) if not region.is_empty())


def _translate(artwork, dx, dy):
    return tuple((key, _shapes(_region(polygons).transformed(kdb.Trans(dx, dy)))) for key, polygons in artwork)


def _merge(artworks):
    result = {}
    for artwork in artworks:
        for key, polygons in artwork:
            checkpoint()
            result.setdefault(key, kdb.Region()).insert(_region(polygons))
    return tuple((key, _shapes(region)) for key, region in sorted(result.items()) if not region.is_empty())


def _bbox(artwork):
    points = tuple(point for _layer, shapes in artwork for shape in shapes for point in shape.hull_nm)
    require(points, 'native artwork is empty')
    return min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points)


def _extract(artwork, model):
    checkpoint()
    nodes = extract_nodes({key: list(_region(polygons).each()) for key, polygons in artwork}, model)
    checkpoint()
    return nodes


def _unit_sites(unit, terminals, tech):
    validate_unit_master_regions(terminals.master)
    actual = _layers(unit.polygons)
    require(actual == terminals.master.geometry_by_layer_nm, 'raw master all-layer artwork XOR differs from current PDK')
    nodes = _extract(actual, terminals.connect_model)
    role_nodes = {role: node_index_at(nodes, unit.terminal_layers[role], *unit.terminals[role].center)
                  for role in ('c', 'b', 'e')}
    require(None not in role_nodes.values() and len(set(role_nodes.values())) == 3,
            'current raw PDK role accesses are absent or collapsed')
    queries = dict(terminals.site_queries)
    for site in terminals.master.terminal_sites:
        checkpoint()
        require(len(site.geometry_by_layer_nm) == 1, 'multilayer raw BJT site capability is unsupported')
        raw, shapes = site.geometry_by_layer_nm[0]
        query = queries[site.site_id]
        require(query in terminals.connect_model.metals, 'site query layer absent from current PDK model')
        require((raw in terminals.routing_model.metals and query == raw)
            or (raw == tech.build_layer('diff') and query in (_NOD, _POD))
            or (raw == tech.build_layer('poly') and query == raw), 'raw/query layer binding is unsupported')
        probe = _region(shapes)
        hits = [index for index, node in enumerate(nodes) if query in node and not (node[query] & probe).is_empty()]
        require(hits == [role_nodes[site.role_group]] and (probe-nodes[hits[0]][query]).is_empty(),
                'complete raw site differs from its unique PDK role access node')


def _component_at(inventory, layer, point):
    probe = kdb.Point(*point)
    hits = []
    for component in inventory.joint_components:
        masks = dict(component.geometry_by_layer_nm)
        if layer in masks and any(polygon.inside(probe) for polygon in _region(masks[layer]).each()):
            hits.append(component.component_id)
    require(len(hits) == 1, 'external access is missing or ambiguous')
    return hits[0]


def _require_occurrence_metadata(geometry, member, unit_extent, spacing):
    from clay.layout.bjt_device_cell import BjtUnitOccurrenceGeometry
    width, height = unit_extent[2]-unit_extent[0], unit_extent[3]-unit_extent[1]
    centers = tuple((width//2+column*(width+spacing), height//2+row*(height+spacing))
                    for row, column, _pose in member.unit_pattern.placements)
    expected = BjtUnitOccurrenceGeometry(kind=member.kind, source_sha256=member.raw_master_sha256,
        unit_count=member.unit_count, bank_extent_nm=geometry.extent(), unit_centers_nm=centers,
        unit_transforms=tuple(pose for _r, _c, pose in member.unit_pattern.placements))
    require(type(geometry.unit_occurrences) is BjtUnitOccurrenceGeometry
        and geometry.unit_occurrences == expected and geometry.kind == member.kind
        and geometry.model == member.source_model and geometry.source_sha256 == member.raw_master_sha256
        and geometry.unit_count == member.unit_count, 'current native occurrence metadata differs from complete raw copies')


def materialize_bjt_local_assembly(source, *, source_cell, tech, terminals, intent=None):
    from clay.layout.annotations import Tap
    from clay.layout.bjt_device_cell import build_bjt_cell
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    checkpoint()
    source = replay_nonmos_assembly_source(source, source_cell=source_cell, tech=tech, intent=intent)
    require(type(terminals) is BjtAssemblyTerminalModel, 'terminal model type differs')
    terminals.__post_init__()
    terminals = _deep(terminals)
    require(terminals.master.pdk_id == tech.pdk_id and terminals.provenance_sha256 == pdk_provenance_digest(tech),
            'current PDK terminal provenance differs')
    require((terminals.connect_model, terminals.routing_model) == current_bjt_assembly_models(tech),
            'current PDK complete connectivity/routing model differs')
    master = terminals.master
    selected = {member.owner for member in source.members}
    require(all(o.master_artwork_sha256 == master.artwork_sha256 and o.master_content_sha256 == master.variant_sha256
        for o in source.reference_overlay.occurrences if o.logical_owner in selected),
        'source observed raw master/variant identity differs')
    family = source.members[0].kind
    unit = build_bjt_cell(tech.adapter, family)
    _unit_sites(unit, terminals, tech)
    require(unit.extent()[:2] == (0, 0), 'current raw unit has foreign origin')
    width, height = unit.extent()[2:]
    gap = tech.adapter.bjt_bindings[family].array_spacing_nm
    require(type(gap) is int and gap >= 0, 'PDK array spacing is invalid')
    grid = pdk_manufacturing_grid_nm(tech)
    require((width+gap) % grid == 0 and (height+gap) % grid == 0, 'PDK unit lattice is off manufacturing grid')
    members, occurrences, expected, placed, offsets, native = [], [], [], [], [], {}
    for member in source.members:
        checkpoint()
        taps = None if member.tap is None else Tap(net=member.tap[0], where=member.tap[1])
        geometry = build_bjt_cell(tech.adapter, member.kind, units=member.unit_count,
                                 tap_declaration=taps, unit_pattern=member.unit_pattern)
        _require_occurrence_metadata(geometry, member, unit.extent(), gap)
        dy, dx = member.origin_rank[0]*(height+gap), member.origin_rank[1]*(width+gap)
        offsets.append((member.owner, (dx, dy)))
        native[member.owner] = geometry
        raw_copies = []
        raw_count = len(unit.polygons)
        require(len(geometry.polygons) >= member.unit_count*raw_count, 'native raw occurrence polygon denominator differs')
        for ordinal, (row, column, pose) in enumerate(member.unit_pattern.placements):
            checkpoint()
            ux, uy = column*(width+gap), row*(height+gap)
            actual_copy = _layers(geometry.polygons[ordinal*raw_count:(ordinal+1)*raw_count])
            require(actual_copy == _translate(master.geometry_by_layer_nm, ux, uy), 'native raw occurrence copy differs')
            raw_copies.append(_translate(actual_copy, dx, dy))
            node = f'{member.owner}[{ordinal:03d}]'
            frame = (1, 0, 0, 1, dx+ux, dy+uy)
            occurrences.append(NativeNonmosOccurrence(node, member.owner, ordinal, master.variant_sha256, frame))
            for site in master.terminal_sites:
                raw, masks = site.geometry_by_layer_nm[0]
                sid = node+'/'+site.site_id
                expected.append(TerminalSiteRequirement(sid, member.owner, node, dict(member.pins)[site.role_group],
                    (site.role_group,), raw, dict(terminals.site_queries)[site.site_id], masks,
                    master.variant_sha256, source.sha256))
                moved = _shapes(_region(masks).transformed(kdb.Trans(dx+ux, dy+uy)))
                placed.append(OccurrenceTerminalSite(sid, master.variant_sha256, frame, moved))
        added = _translate(_layers(geometry.polygons[member.unit_count*raw_count:]), dx, dy)
        artwork = _translate(_layers(geometry.polygons), dx, dy)
        require(_merge((*raw_copies, added)) == artwork, 'raw/added complete artwork partition differs')
        members.append(NativeNonmosMember(member.owner, member.kind, member.unit_count,
                                         _bbox(artwork), artwork, tuple(raw_copies), added))
    # Enclosing member boxes may overlap; actual artwork may not overlap on a
    # common layer. Spacing/DRC beyond the explicit PDK construction is separate.
    for index, left in enumerate(members):
        for right in members[index+1:]:
            for key in set(dict(left.artwork)) & set(dict(right.artwork)):
                checkpoint()
                require((_region(dict(left.artwork)[key]) & _region(dict(right.artwork)[key])).is_empty(),
                        'interowner positive-area raw/added layer overlap')
    artwork = _merge(tuple(member.artwork for member in members))
    inventory = compile_joint_terminal_inventory(artwork=artwork,
        expected_sites=tuple(sorted(expected, key=lambda row: row.site_id)),
        placed_sites=tuple(sorted(placed, key=lambda row: row.site_id)),
        connect_model=terminals.connect_model, routing_model=terminals.routing_model,
        pdk_provenance_sha256=terminals.provenance_sha256, checkpoint=checkpoint, limits=JointTerminalLimits())
    accesses = []
    incidences = dict(inventory.site_components)
    for member in members:
        geometry = native[member.owner]
        dx, dy = dict(offsets)[member.owner]
        for role in ('b', 'c', 'e'):
            checkpoint()
            expected_ids = {component for site in inventory.expected_sites
                if site.owner == member.owner and site.allowed_roles == (role,) for component in incidences[site.site_id]}
            layer = geometry.terminal_layers[role]
            points = tuple((int(access.center[0])+dx, int(access.center[1])+dy) for access in geometry.terminal_accesses[role])
            actual_ids = {_component_at(inventory, layer, point) for point in points}
            require(len(expected_ids) == 1 and actual_ids == expected_ids,
                    'external role access fails to reach every original unit site')
            accesses.append(NativeNonmosExternalAccess(member.owner, role, layer, points, tuple(sorted(actual_ids))))
    replay_nonmos_assembly_source(source, source_cell=source_cell, tech=tech)
    values = dict(source=source, terminals=terminals, members=tuple(members), owner_offsets_nm=tuple(offsets),
        occurrences=tuple(occurrences), external_accesses=tuple(accesses), joint_inventory=inventory, bbox_nm=_bbox(artwork))
    result = object.__new__(NativeNonmosAssemblyBinding)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    object.__setattr__(result, '_seal', _SEAL)
    object.__setattr__(result, '_content_sha256', digest(result.payload()))
    return result


def replay_bjt_local_assembly(value, *, source_cell, tech, member_artwork=None, intent=None):
    checkpoint()
    require(type(value) is NativeNonmosAssemblyBinding and value._seal is _SEAL, 'native materialization binding is absent')
    old = value.sha256
    if member_artwork is not None:
        require(type(member_artwork) is tuple, 'actual member artwork must be an exact tuple')
        for owner, artwork in member_artwork:
            _geometry(artwork, 'actual member artwork')
        require(member_artwork == tuple((m.owner, m.artwork) for m in value.members),
                'actual native artwork or owner offsets differ from complete binding')
    actual = materialize_bjt_local_assembly(value.source, source_cell=source_cell, tech=tech,
                                          terminals=value.terminals, intent=intent)
    require(actual.sha256 == old, 'native current source/artwork/ledger replay differs')
    return value


def bjt_assembly_member_geometry(value, owner, *, tech):
    """Fresh independent BjtCellGeom in its original member builder frame.

This checks current PDK and exact retained geometry. The integration boundary
must additionally replay the complete current source and effective intent.
"""
    from clay.layout.annotations import Tap
    from clay.layout.bjt_device_cell import build_bjt_cell
    checkpoint()
    require(type(value) is NativeNonmosAssemblyBinding, 'native member binding type differs')
    value.validate()
    require(technology_identity(tech) == value.source.technology_identity, 'current member PDK identity differs')
    declarations = tuple(m for m in value.source.members if m.owner == owner)
    require(len(declarations) == 1, 'native member owner is absent')
    member = declarations[0]
    tap = None if member.tap is None else Tap(net=member.tap[0], where=member.tap[1])
    geometry = build_bjt_cell(tech.adapter, member.kind, units=member.unit_count,
                             tap_declaration=tap, unit_pattern=member.unit_pattern)
    dx, dy = dict(value.owner_offsets_nm)[owner]
    retained = next(m for m in value.members if m.owner == owner)
    require(_translate(_layers(geometry.polygons), dx, dy) == retained.artwork,
            'current native member raw/added artwork differs')
    _require_occurrence_metadata(geometry, member, value.terminals.master.extent_nm,
                                 tech.adapter.bjt_bindings[member.kind].array_spacing_nm)
    require(technology_identity(tech) == value.source.technology_identity, 'PDK changed during member copy')
    return geometry
