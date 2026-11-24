"""Complete local raw-unit/accessory evidence for an emitted rigid row.

Records retain supplied source provenance; replay proves conservation against
current row artwork and the independent master site ledger, not source authority
or physical admission. Shared artwork is emitted once per part and may belong
to several logical occurrences. Channel centers are not raw-master centers.
"""
from collections import Counter
from dataclasses import asdict, dataclass

import klayout.db as kdb

from clay.frontend.unit_master_geometry import UnitMasterGeometry, validate_unit_master_regions
from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.connectivity import ConnectModel
from clay.layout.joint_terminal_inventory import (
    JointTerminalInventory, JointTerminalLimits, TerminalSiteRequirement,
    OccurrenceTerminalSite, _digest, _frame, _geometry, _native_frame, _sha, _text,
    _tuple, replay_joint_terminal_inventory,
)
from clay.layout.occurrence_artwork import polygon_records
from clay.layout.occurrence_channels import _shape_keys
from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM
from clay.layout.joint_occurrence_center_basis import JointRawChannelCenters


def _require(condition, message):
    if not condition:
        raise ValueError('joint row ' + message)


def joint_checkpoint(callback=None):
    """The active execution owner cannot be bypassed by a supplied callback."""
    from clay.flow.execution_session import current_session
    _require(callback is None or callable(callback), 'checkpoint is not callable')
    session = current_session()
    def check():
        if session is not None:
            session.checkpoint()
        if callback is not None:
            callback()
    check()
    return check


def _region(shapes):
    checkpoint = joint_checkpoint()
    region = kdb.Region()
    for shape in shapes:
        checkpoint()
        polygon = kdb.Polygon([kdb.Point(*point) for point in shape.hull_nm])
        for hole in shape.holes_nm:
            polygon.insert_hole([kdb.Point(*point) for point in hole])
        region.insert(polygon)
    checkpoint()
    return region


def _shapes(region):
    checkpoint = joint_checkpoint()
    result = tuple(PhysicalTerminalPolygonNM(hull, holes) for hull, holes in _shape_keys(region))
    checkpoint()
    return result


def joint_transform_shapes(shapes, native_frame):
    """Rigidly move each original polygon, retaining its split and multiplicity."""
    return tuple(sorted(transformed for shape in shapes
                        for transformed in _shapes(shape.to_region().transformed(native_frame))))


@dataclass(frozen=True, slots=True)
class JointArtworkPart:
    part_id: str
    kind: str
    provenance_sha256: str
    occurrence_ids: tuple[str, ...]
    geometry_by_layer_nm: tuple

    def __post_init__(self):
        _text(self.part_id, 'part id')
        _require(self.kind in ('raw_unit', 'accessory'), 'artwork part kind is unsupported')
        _sha(self.provenance_sha256, 'part provenance')
        _tuple(self.occurrence_ids, 'part occurrences', nonempty=True)
        for name in self.occurrence_ids:
            _text(name, 'part occurrence')
        _require(self.occurrence_ids == tuple(sorted(set(self.occurrence_ids))), 'part associations differ')
        _geometry(self.geometry_by_layer_nm, 'part artwork')


@dataclass(frozen=True, slots=True)
class JointRowOccurrence:
    occurrence_id: str
    owner: str
    raw_part_id: str
    master: UnitMasterGeometry
    master_to_local: tuple[int, int, int, int, int, int]
    # A native multi-finger PCell may be one indivisible raw part associated
    # with several source occurrences.  Each occurrence selects the actual
    # terminal sites around its own channel, while the raw artwork is retained
    # and emitted once.  Empty values retain the legacy one-unit contract.
    master_site_ids: tuple[str, ...] = ()
    channel_site_id: str = ''

    def __post_init__(self):
        for field in ('occurrence_id', 'owner', 'raw_part_id'):
            _text(getattr(self, field), field)
        _require(type(self.master) is UnitMasterGeometry, 'raw master is foreign')
        self.master.__post_init__()
        _frame(self.master_to_local)
        _tuple(self.master_site_ids, 'occurrence master sites')
        if self.master_site_ids:
            _require(self.master_site_ids == tuple(sorted(set(self.master_site_ids))),
                     'occurrence master sites must be canonical and unique')
            _require(set(self.master_site_ids) <= {s.site_id for s in self.master.terminal_sites},
                     'occurrence master site is absent')
            _require(self.channel_site_id in self.master_site_ids,
                     'occurrence channel site is absent')
            _require(next(s for s in self.master.terminal_sites
                          if s.site_id == self.channel_site_id).role_group == 'gate',
                     'occurrence channel site must be a gate')
            selected = [s for s in self.master.terminal_sites if s.site_id in self.master_site_ids]
            _require({s.site_id for s in selected if s.role_group == 'gate'} == {self.channel_site_id}
                     and sum(s.role_group == 'sd' for s in selected) == 2,
                     'selected occurrence must contain one gate and two diffusion sites')
        else:
            _require(self.channel_site_id == '', 'legacy occurrence cannot select a channel site')


@dataclass(frozen=True, slots=True)
class JointRowGeometryBinding:
    parts: tuple[JointArtworkPart, ...]
    occurrences: tuple[JointRowOccurrence, ...]
    expected_sites: tuple[TerminalSiteRequirement, ...]
    placed_sites: tuple[OccurrenceTerminalSite, ...]
    connect_model: ConnectModel
    routing_model: ConnectModel
    pdk_id: str
    pdk_provenance_sha256: str
    joint_inventory: JointTerminalInventory
    limits: JointTerminalLimits = JointTerminalLimits()
    quality_centers: JointRawChannelCenters | None = None
    substrate_binding: object | None = None

    def __post_init__(self):
        for field, kind in (('parts', JointArtworkPart), ('occurrences', JointRowOccurrence),
                            ('expected_sites', TerminalSiteRequirement), ('placed_sites', OccurrenceTerminalSite)):
            values = getattr(self, field)
            _tuple(values, field, nonempty=True)
            for value in values:
                _require(type(value) is kind, field + ' contains a foreign record')
                value.__post_init__()
        _text(self.pdk_id, 'PDK id')
        _sha(self.pdk_provenance_sha256, 'PDK provenance')
        _require(type(self.joint_inventory) is JointTerminalInventory, 'terminal inventory is foreign')
        self.joint_inventory.__post_init__()
        _require(type(self.limits) is JointTerminalLimits, 'limits are foreign')
        self.limits.__post_init__()
        if self.quality_centers is not None:
            _require(type(self.quality_centers) is JointRawChannelCenters, 'quality center basis is foreign')
            self.quality_centers.__post_init__()
        if self.substrate_binding is not None:
            from clay.layout.native_substrate_binding import NativeSubstrateBinding
            _require(type(self.substrate_binding) is NativeSubstrateBinding, 'substrate binding is foreign')

    def payload(self):
        self.__post_init__()
        data = asdict(self)
        selected = any(occ.master_site_ids for occ in self.occurrences)
        if not selected:
            for row in data['occurrences']:
                row.pop('master_site_ids')
                row.pop('channel_site_id')
        centers = data.pop('quality_centers')
        substrate = data.pop('substrate_binding')
        if substrate is not None:
            return {'schema': 'clay.joint-row-geometry/v4', **data,
                    **({'quality_centers': centers} if centers is not None else {}),
                    'substrate_binding': substrate}
        if selected:
            return {'schema': 'clay.joint-row-geometry/v3', **data,
                    **({'quality_centers': centers} if centers is not None else {})}
        if centers is not None:
            return {'schema': 'clay.joint-row-geometry/v2', **data, 'quality_centers': centers}
        return {'schema': 'clay.joint-row-geometry/v1', **data}

    @property
    def sha256(self):
        return _digest(self.payload())


def joint_row_artwork(parts):
    """Merged connectivity operand; physical instance multiplicity stays in parts."""
    checkpoint = joint_checkpoint()
    _tuple(parts, 'artwork parts', nonempty=True)
    layers = {}
    for part in parts:
        checkpoint()
        _require(type(part) is JointArtworkPart, 'artwork part is foreign')
        part.__post_init__()
        for key, shapes in part.geometry_by_layer_nm:
            layers.setdefault(key, kdb.Region()).insert(_region(shapes))
    return tuple((key, _shapes(region)) for key, region in sorted(layers.items()))


def joint_row_polygons(parts):
    """Emit each supplied part exactly once, preserving holes with resolved rings."""
    checkpoint = joint_checkpoint()
    result = []
    for part in parts:
        checkpoint()
        _require(type(part) is JointArtworkPart, 'artwork part is foreign')
        part.__post_init__()
        for key, shapes in part.geometry_by_layer_nm:
            for shape in shapes:
                for native in _region((shape,)).each():
                    simple = native.resolved_holes()
                    result.append(LayerPolygon(*key, Polygon([Point(point.x, point.y)
                        for point in simple.each_point_hull()])))
    return tuple(result)


def joint_record_counter(records):
    """Compare each emitted polygon as a region, retaining physical multiplicity.

Rigid D4 changes canonical contour starting vertices and hole bridge order;
neither is a geometry change. A merged all-part union would lose duplicates.
"""
    checkpoint = joint_checkpoint()
    result = Counter()
    for layer, datatype, vertices in records:
        checkpoint()
        shapes = _shape_keys(kdb.Region(kdb.Polygon([kdb.Point(*point) for point in vertices])))
        checkpoint()
        result[(layer, datatype, shapes)] += 1
    return result


def replay_joint_row_binding(binding, *, polygons, consumed_devices, tech=None, fingers=None, members=None,
                             checkpoint=None):
    """Re-extract both graphs only after independent raw/site/artwork replay."""
    from clay.layout.tech import LayoutTech
    from clay.layout.route.resource_world import pdk_provenance_digest
    checkpoint = joint_checkpoint(checkpoint)
    _require(type(binding) is JointRowGeometryBinding, 'geometry binding is missing or foreign')
    binding.__post_init__()
    tech = LayoutTech.resolve(binding.pdk_id) if tech is None else tech
    _require(tech.pdk_id == binding.pdk_id and pdk_provenance_digest(tech) == binding.pdk_provenance_sha256,
             'current PDK provenance differs')
    _require(binding.routing_model == ConnectModel.from_tech(tech), 'routing model differs from current PDK')
    if binding.quality_centers is not None:
        from clay.layout.joint_occurrence_center_basis import replay_joint_channel_centers
        replay_joint_channel_centers(binding, tech=tech, members=members)
    parts = {part.part_id: part for part in binding.parts}
    occurrences = {occ.occurrence_id: occ for occ in binding.occurrences}
    _require(len(parts) == len(binding.parts) and len(occurrences) == len(binding.occurrences),
             'source part or occurrence ids are duplicated')
    _require(set(consumed_devices) == {occ.owner for occ in binding.occurrences}, 'source owner denominator differs')
    if fingers is not None:
        _require(tuple(f.member for f in fingers) == tuple(o.owner for o in binding.occurrences),
                 'source occurrence sequence differs from row fingers')
    _require(all(set(part.occurrence_ids) <= set(occurrences) for part in binding.parts),
             'artwork association references a missing occurrence')
    _require({o.raw_part_id for o in binding.occurrences} ==
             {p.part_id for p in binding.parts if p.kind == 'raw_unit'}, 'raw part denominator differs')
    expected = {site.site_id: site for site in binding.expected_sites}
    placed = {site.site_id: site for site in binding.placed_sites}
    required_ids = set()
    associations = {}
    for occurrence in binding.occurrences:
        associations.setdefault(occurrence.raw_part_id, []).append(occurrence)
    for part_id, group in associations.items():
        raw = parts[part_id]
        _require(raw.occurrence_ids == tuple(sorted(occ.occurrence_id for occ in group)),
                 'raw part association differs')
        if len(group) > 1:
            _require(all(occ.master_site_ids for occ in group), 'shared raw part needs selected sites')
            _require(all((occ.master, occ.master_to_local) == (group[0].master, group[0].master_to_local)
                         for occ in group), 'shared raw part master or frame differs')
            _require(set().union(*(set(occ.master_site_ids) for occ in group)) ==
                     {site.site_id for site in group[0].master.terminal_sites},
                     'shared raw part site denominator differs')
            selected_gates = tuple(occ.channel_site_id for occ in group)
            _require(len(set(selected_gates)) == len(group)
                     and set(selected_gates) == {site.site_id for site in group[0].master.terminal_sites
                                                  if site.role_group == 'gate'},
                     'shared raw channel gate ownership differs')
            for occ in group:
                by_site = {site.site_id: site for site in occ.master.terminal_sites}
                gate = by_site[occ.channel_site_id]
                sd = [by_site[name] for name in occ.master_site_ids if by_site[name].role_group == 'sd']
                _require(len(gate.geometry_by_layer_nm) == 1
                         and all(len(site.geometry_by_layer_nm) == 1 for site in sd),
                         'shared raw channel sites must use one layer each')
                sd_layer = sd[0].geometry_by_layer_nm[0][0]
                _require(all(site.geometry_by_layer_nm[0][0] == sd_layer for site in sd),
                         'shared raw diffusion layers differ')
                active = dict(occ.master.geometry_by_layer_nm).get(sd_layer)
                _require(active is not None, 'shared raw diffusion layer is missing')
                channel = (_region(active) & _region(gate.geometry_by_layer_nm[0][1])).merged()
                _require(channel.count() == 1 and channel.area() > 0,
                         'shared raw selected gate must cross one channel')
                body = [by_site[name] for name in occ.master_site_ids if by_site[name].role_group == 'body']
                if body:
                    from clay.layout.tech import LayoutTech
                    _require(binding.pdk_id == 'sky130' and len(body) == 1,
                             'shared raw body site requires one supported native PDK well')
                    well_key = tuple(LayoutTech.resolve('sky130').build_layer('nwell'))
                    _require(len(body[0].geometry_by_layer_nm) == 1
                             and body[0].geometry_by_layer_nm[0][0] == well_key,
                             'shared raw body site is not the native well')
                    well = _region(body[0].geometry_by_layer_nm[0][1])
                    _require(well_key in dict(occ.master.geometry_by_layer_nm)
                             and (well ^ _region(dict(occ.master.geometry_by_layer_nm)[well_key])).is_empty()
                             and (channel-well).is_empty(),
                             'shared raw body well does not cover the source channel')
                if members is not None:
                    member = members.get(occ.owner)
                    box = channel.bbox()
                    _require(member is not None
                             and sorted((box.width(), box.height())) ==
                                 sorted((member.width_nm, member.length_nm))
                             and channel.area() == member.width_nm * member.length_nm,
                             'shared raw channel dimensions differ from source')
                    _require(not body or member.kind == 'pmos',
                             'shared raw nwell body is not a PMOS source')
                for site in sd:
                    diffusion = _region(site.geometry_by_layer_nm[0][1])
                    _require(not (diffusion & channel.sized(1)).is_empty(),
                             'shared raw diffusion site is not adjacent to its channel')
        else:
            _require(not group[0].master_site_ids, 'single raw part must use complete legacy sites')
    for occurrence in binding.occurrences:
        checkpoint()
        master = occurrence.master
        validate_unit_master_regions(master)
        checkpoint()
        _require(master.pdk_id == binding.pdk_id, 'master PDK differs')
        raw = parts[occurrence.raw_part_id]
        transform = _native_frame(occurrence.master_to_local)
        actual, original = dict(raw.geometry_by_layer_nm), dict(master.geometry_by_layer_nm)
        _require(set(actual) == set(original) and all((_region(actual[key]) ^
            _region(shapes).transformed(transform)).is_empty() for key, shapes in original.items()),
            'raw artwork differs from master and native frame')
        for site in master.terminal_sites:
            checkpoint()
            if occurrence.master_site_ids and site.site_id not in occurrence.master_site_ids:
                continue
            site_id = occurrence.occurrence_id + '/' + site.site_id
            required_ids.add(site_id)
            _require(site_id in expected and site_id in placed, 'independent master site is missing')
            requirement, position = expected[site_id], placed[site_id]
            _require(len(site.geometry_by_layer_nm) == 1, 'multilayer raw site is unsupported')
            key, shapes = site.geometry_by_layer_nm[0]
            _require(site.role_group in ('gate', 'sd', 'body'), 'raw site role group is unsupported')
            _require(site.role_group != 'body' or bool(occurrence.master_site_ids),
                     'body well sites require explicit native shared occurrences')
            allowed = ('g',) if site.role_group == 'gate' else ('b',) if site.role_group == 'body' else ('d', 's')
            roles = requirement.allowed_roles
            _require(bool(roles) and set(roles) <= set(allowed), 'raw site role aliases differ')
            _require((requirement.owner, requirement.occurrence_id, requirement.allowed_roles,
                      requirement.raw_layer, requirement.master_geometry_nm, requirement.variant_sha256) ==
                     (occurrence.owner, occurrence.occurrence_id, roles, key, shapes, master.variant_sha256),
                     'independent raw site role, geometry, layer or variant differs')
            _require(position.master_to_local == occurrence.master_to_local,
                     'placed site and raw master frames differ')
            if members is not None:
                member = members.get(occurrence.owner)
                nets = {} if member is None else dict(g=member.gate_net, d=member.drain_net,
                    s=member.source_net, b=member.body_net)
                _require(tuple(role for role in allowed if nets.get(role) == requirement.source_net) == roles,
                         'terminal source role/net differs from declared program member')
    if binding.substrate_binding is not None:
        from clay.layout.native_sky130 import native_connect_model
        from clay.layout.native_substrate_binding import substrate_sites
        _require(binding.connect_model == native_connect_model('mos'), 'substrate conductor model differs from PDK')
        _require(sum(part.kind == 'accessory' and
                     part.provenance_sha256 == binding.substrate_binding.contacted_gds_sha256
                     for part in binding.parts) == 1, 'substrate contact producer differs')
        current_artwork = dict(joint_row_artwork(binding.parts))
        observed_artwork = dict(binding.substrate_binding.contacted_artwork)
        extraction_frame = _native_frame(binding.substrate_binding.extraction_frame)
        current_frame = _native_frame(binding.occurrences[0].master_to_local)
        evidence_transform = current_frame * extraction_frame.inverted()
        _require(set(current_artwork) == set(observed_artwork) and
                 all((_region(current_artwork[key]) ^
                      _region(observed_artwork[key]).transformed(evidence_transform)).is_empty()
                     for key in current_artwork), 'substrate source artwork differs')
        body_sites, body_positions = substrate_sites(binding.substrate_binding,
            occurrences=binding.occurrences, members=members, tech=tech)
        for requirement, position in zip(body_sites, body_positions):
            _require(expected.get(requirement.site_id) == requirement and
                     placed.get(position.site_id) == position,
                     'substrate body site differs from physical evidence')
            _require(requirement.site_id not in required_ids, 'substrate body site collides with raw site')
            required_ids.add(requirement.site_id)
    _require(required_ids == set(expected) == set(placed), 'complete independent master site denominator differs')
    if members is not None:
        counts = Counter(o.owner for o in binding.occurrences)
        _require(all(count == members[name].fingers for name, count in counts.items()),
                 'source occurrence count differs from declared program member')
    _require(joint_record_counter(polygon_records(polygons)) ==
             joint_record_counter(polygon_records(joint_row_polygons(binding.parts))),
             'actual artwork multiset differs from complete physical source parts')
    replay_joint_terminal_inventory(inventory=binding.joint_inventory, artwork=joint_row_artwork(binding.parts),
        expected_sites=binding.expected_sites, placed_sites=binding.placed_sites,
        connect_model=binding.connect_model, routing_model=binding.routing_model,
        pdk_provenance_sha256=binding.pdk_provenance_sha256, checkpoint=checkpoint, limits=binding.limits)
    return binding.joint_inventory


def replay_joint_row_geometry(geometry, consumed_devices, *, tech=None, members=None, checkpoint=None):
    from clay.layout.occurrence_channels import RowChannelInventory, compile_row_channel_inventory
    checkpoint = joint_checkpoint(checkpoint)
    binding = getattr(geometry, 'joint_row_binding', None)
    inventory = replay_joint_row_binding(binding, polygons=geometry.polygons,
        consumed_devices=consumed_devices, tech=tech, fingers=geometry.fingers, members=members, checkpoint=checkpoint)
    channels = getattr(geometry, 'channel_inventory', None)
    _require(type(channels) is RowChannelInventory, 'channel inventory is missing or foreign')
    expected = compile_row_channel_inventory(occurrence_artwork=tuple((occ.owner,
        _occurrence_channel_artwork(binding, occ, channels.od_layer_keys, channels.poly_layer_key))
        for occ in binding.occurrences), row_polygons=tuple(geometry.polygons),
        od_layer_keys=channels.od_layer_keys, poly_layer_key=channels.poly_layer_key)
    checkpoint()
    _require(expected == channels, 'channel ownership differs from complete associated source artwork')
    return inventory


def _occurrence_channel_artwork(binding, occurrence, od_layer_keys, poly_layer_key):
    """Project one channel from a shared raw master without changing emitted art.

    The complete OD is associated with each selected channel; its individual
    poly site selects the crossing.  The channel inventory checks that the
    union of these operands exactly covers the row's real OD/poly masks.
    """
    parts = tuple(part for part in binding.parts if occurrence.occurrence_id in part.occurrence_ids)
    if not occurrence.master_site_ids:
        return joint_row_polygons(parts)
    site = next(site for site in occurrence.master.terminal_sites
                if site.site_id == occurrence.channel_site_id)
    gate = dict(site.geometry_by_layer_nm).get(poly_layer_key)
    _require(gate is not None, 'selected channel site lacks poly geometry')
    frame = _native_frame(occurrence.master_to_local)
    records = []
    for key in od_layer_keys:
        for part in parts:
            for shape in dict(part.geometry_by_layer_nm).get(key, ()):
                for native in _region((shape,)).each():
                    simple = native.resolved_holes()
                    records.append(LayerPolygon(*key, Polygon([Point(p.x, p.y) for p in simple.each_point_hull()])))
    for native in _region(gate).transformed(frame).each():
        simple = native.resolved_holes()
        records.append(LayerPolygon(*poly_layer_key, Polygon([Point(p.x, p.y) for p in simple.each_point_hull()])))
    # Accessories can extend gates for contacts. Preserve those exact masks in
    # the structural-layer union; the channel inventory still rejects any
    # additional crossing or duplicated channel they would introduce.
    for part in parts:
        if part.kind != 'accessory':
            continue
        for shape in dict(part.geometry_by_layer_nm).get(poly_layer_key, ()):
            for native in _region((shape,)).each():
                simple = native.resolved_holes()
                records.append(LayerPolygon(*poly_layer_key, Polygon([Point(p.x, p.y) for p in simple.each_point_hull()])))
    return tuple(records)


def joint_d4_orientation(frame):
    _frame(frame)
    return {(1,0,0,1): 'R0', (0,-1,1,0): 'R90', (-1,0,0,-1): 'R180',
        (0,1,-1,0): 'R270', (1,0,0,-1): 'MR0', (0,1,1,0): 'MR90',
        (-1,0,0,1): 'MR180', (0,-1,-1,0): 'MR270'}[frame[:4]]


def joint_occurrence_centers(binding, origin_nm):
    if getattr(binding, 'quality_centers', None) is not None:
        from clay.layout.joint_occurrence_center_basis import joint_channel_occurrence_centers
        return joint_channel_occurrence_centers(binding, origin_nm)
    result = {}
    for occurrence in binding.occurrences:
        transform = _native_frame(occurrence.master_to_local)
        box = transform * kdb.Box(*occurrence.master.extent_nm)
        result.setdefault(occurrence.owner, []).append(((box.left+box.right)//2-origin_nm[0],
            (box.bottom+box.top)//2-origin_nm[1], joint_d4_orientation(occurrence.master_to_local)))
    return tuple((owner, tuple(sorted((x/1000, y/1000, orientation) for x,y,orientation in points)))
                 for owner, points in sorted(result.items()))
