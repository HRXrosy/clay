"""Construct native MOS mosaics from raw masters and coordinate-free source order.

The caller authenticates source, raw masters, terminal net aliases and shared
accessories. Construction accepts no observed positions or reference pitch.
It preserves every supplied physical part and independently replays both local
terminal graphs. This is not body tracing, placement legality, DRC or LVS.
"""
from collections import Counter
from dataclasses import dataclass

from clay.frontend.unit_master_geometry import UnitMasterGeometry, validate_unit_master_regions
from clay.layout.joint_row_geometry import (
    JointArtworkPart, JointRowOccurrence, JointRowGeometryBinding, _region, _shapes,
    joint_checkpoint, joint_row_artwork, joint_row_polygons, replay_joint_row_geometry, joint_transform_shapes,
)
from clay.layout.joint_terminal_inventory import (
    TerminalSiteRequirement, OccurrenceTerminalSite, JointTerminalLimits,
    compile_joint_terminal_inventory, _native_frame, _sha, _geometry,
)


def _require(value, message):
    if not value:
        raise ValueError('MOS mosaic ' + message)


@dataclass(frozen=True, slots=True)
class MosMosaicRawUnit:
    occurrence_id: str
    master: UnitMasterGeometry
    sites: tuple[TerminalSiteRequirement, ...]
    raw_geometry_by_layer_nm: tuple = ()


@dataclass(frozen=True, slots=True)
class MosMosaicAccessory:
    """One physical accessory in its declared anchor unit's normalized frame."""
    part_id: str
    anchor_occurrence_id: str
    occurrence_ids: tuple[str, ...]
    geometry_by_layer_nm: tuple
    provenance_sha256: str


@dataclass(frozen=True, slots=True)
class MosMosaicSharedInterface:
    left_site_id: str
    right_site_id: str


@dataclass(frozen=True, slots=True)
class MosMosaicSharedAccessInterface:
    left_site_id: str
    right_site_id: str


def _move(layers, frame):
    transform = _native_frame(frame)
    return tuple((key, joint_transform_shapes(shapes, transform)) for key, shapes in layers)


def _raw_layers(unit):
    _require(type(unit.raw_geometry_by_layer_nm) is tuple, 'raw source artwork must be an exact tuple')
    return unit.raw_geometry_by_layer_nm or unit.master.geometry_by_layer_nm


def _box(layers):
    boxes = tuple(_region(shapes).bbox() for _, shapes in layers)
    _require(bool(boxes), 'complete local artwork is empty')
    return (min(box.left for box in boxes), min(box.bottom for box in boxes),
            max(box.right for box in boxes), max(box.top for box in boxes))


def _linear(orientation):
    from clay.layout.mosaic_geometry import _D4
    (a, b), (c, d) = _D4[orientation]
    return a, b, c, d


def _prepare(mosaic, units, members, tech, check):
    from clay.layout.device_cell import _resolve_layer_keys
    from clay.layout.terminal_body_geometry import mos_channel_dimensions_nm
    _require(type(units) is tuple and 0 < len(units) <= 4096
             and all(type(unit) is MosMosaicRawUnit for unit in units), 'unit inventory is malformed')
    lookup = {unit.occurrence_id: unit for unit in units}
    _require(len(lookup) == len(units) and set(lookup) == {node.node_id for node in mosaic.occurrences},
             'complete unit denominator differs from source mosaic')
    _require(set(mosaic.members) <= set(members), 'source member denominator differs')
    counts = Counter(node.logical_owner for node in mosaic.occurrences)
    _require(all(members[name].fingers == count for name, count in counts.items()),
             'source occurrence denominator differs')
    kinds = {members[name].kind for name in counts}
    _require(len(kinds) == 1 and kinds <= {'nmos','pmos'}, 'source family must be one MOS polarity')
    kind = next(iter(kinds))
    keys = _resolve_layer_keys(tech, kind)
    frames, channels, expected = {}, {}, {}
    for node in mosaic.occurrences:
        check()
        unit = lookup[node.node_id]
        _require(type(unit.master) is UnitMasterGeometry and unit.master.pdk_id == tech.pdk_id,
                 'raw master or PDK identity differs')
        validate_unit_master_regions(unit.master)
        raw = _raw_layers(unit)
        _geometry(raw, 'raw source artwork')
        actual, model = dict(raw), dict(unit.master.geometry_by_layer_nm)
        _require(set(actual) == set(model) and all((_region(actual[key]) ^ _region(shapes)).is_empty()
                 for key, shapes in model.items()), 'raw source artwork differs from canonical master model')
        source = members[node.logical_owner]
        nets = dict(g=source.gate_net, d=source.drain_net, s=source.source_net)
        original = {site.site_id: site for site in unit.master.terminal_sites}
        _require(type(unit.sites) is tuple and all(type(site) is TerminalSiteRequirement for site in unit.sites),
                 'independent site inventory is malformed')
        requirements = {site.site_id: site for site in unit.sites}
        _require(len(requirements) == len(unit.sites) and set(requirements) ==
                 {node.node_id+'/'+name for name in original}, 'complete independent site denominator differs')
        _require(Counter(site.role_group for site in original.values()) == {'gate':1,'sd':2},
                 'complete raw MOS gate/SD site denominator differs')
        for local_id, raw in original.items():
            site = requirements[node.node_id+'/'+local_id]
            site.__post_init__()
            _require(site.owner == node.logical_owner and site.occurrence_id == node.node_id
                     and site.variant_sha256 == unit.master.variant_sha256
                     and len(raw.geometry_by_layer_nm) == 1
                     and raw.geometry_by_layer_nm[0] == (site.raw_layer, site.master_geometry_nm),
                     'independent raw site identity or geometry differs')
            pool = ('g',) if raw.role_group == 'gate' else ('d','s')
            _require(site.allowed_roles == tuple(role for role in pool if nets[role] == site.source_net),
                     'source terminal role/net differs')
            _require(site.site_id not in expected, 'source site identity is duplicated')
            expected[site.site_id] = site
        layers = dict(unit.master.geometry_by_layer_nm)
        _require(keys.active in layers and keys.poly in layers, 'raw master channel layers are missing')
        channel = (_region(layers[keys.active]) & _region(layers[keys.poly])).merged()
        bounds = channel.bbox()
        _require(channel.size() == 1 and bounds.width() > 0 and bounds.height() > 0
                 and channel.area() == bounds.width()*bounds.height(), 'raw master channel is not one rectangle')
        dimensions = (source.width_nm, source.length_nm)
        _require(all(type(value) is int and value > 0 for value in dimensions)
                 and mos_channel_dimensions_nm(_region(layers[keys.active]), _region(layers[keys.poly])) == dimensions,
                 'raw master channel dimensions differ from source W/L')
        linear = _linear(node.transform)
        oriented = channel.transformed(_native_frame((*linear, 0, 0)))
        bounds = oriented.bbox()
        dx, dy = -((bounds.left+bounds.right)//2), -((bounds.bottom+bounds.top)//2)
        frames[node.node_id] = (*linear, dx, dy)
        channels[node.node_id] = _shapes(oriented.transformed(_native_frame((1,0,0,1,dx,dy))))
    return lookup, kind, keys, frames, channels, expected


def _positions(mosaic, units, frames, channels, expected, interfaces, accessories, gap, grid, keys, check):
    from clay.layout.fixed_artwork_abutment import align_fixed_artwork_abutment
    rows = tuple(tuple(node.node_id for node in row) for row in mosaic.rows)
    # Complete channel column ranks are source equalities. For a rectangular
    # row sequence without these, the declared row positions define columns.
    ranks = ({f'{owner}[{ordinal:03d}]':rank for owner,ordinal,rank in mosaic.column_ranks}
             if mosaic.column_ranks else {node:index for row in rows for index,node in enumerate(row)})
    _require(mosaic.column_ranks or len({len(row) for row in rows}) == 1,
             'ragged mosaic requires complete source channel column ranks')
    local = {node:_move(_raw_layers(unit), frames[node]) for node,unit in units.items()}
    boxes = {node:_box(layers) for node,layers in local.items()}
    for accessory in accessories:
        anchor = accessory.anchor_occurrence_id
        box, old = _box(_move(accessory.geometry_by_layer_nm, frames[anchor])), boxes[anchor]
        boxes[anchor] = min(box[0],old[0]),min(box[1],old[1]),max(box[2],old[2]),max(box[3],old[3])
    adjacent = {(left,right) for row in rows for left,right in zip(row,row[1:])}
    shared, equalities = set(), []
    _require(type(interfaces) is tuple and all(type(value) in (MosMosaicSharedInterface, MosMosaicSharedAccessInterface)
                                             for value in interfaces),
             'shared interface inventory is malformed')
    for interface in interfaces:
        check()
        _require(interface.left_site_id in expected and interface.right_site_id in expected,
                 'shared interface source site is missing')
        left, right = expected[interface.left_site_id], expected[interface.right_site_id]
        pair = left.occurrence_id, right.occurrence_id
        _require(pair in adjacent and pair not in shared and left.source_net == right.source_net
                 and set(left.allowed_roles) <= {'d','s'} and set(right.allowed_roles) <= {'d','s'},
                 'shared interface order or source terminal net differs')
        placed_left = _move(((left.raw_layer,left.master_geometry_nm),), frames[pair[0]])[0][1]
        placed_right = _move(((right.raw_layer,right.master_geometry_nm),), frames[pair[1]])[0][1]
        arguments = dict(left_site=placed_left,right_site=placed_right,
            left_channel=channels[pair[0]],right_channel=channels[pair[1]],checkpoint=check)
        if type(interface) is MosMosaicSharedAccessInterface:
            from clay.layout.fixed_artwork_shared_access import align_fixed_artwork_shared_access
            alignment = align_fixed_artwork_shared_access(**arguments, left_artwork=local[pair[0]],
                right_artwork=local[pair[1]],contact_layer=keys.contact,metal_layer=keys.metal1)
        else:
            alignment = align_fixed_artwork_abutment(**arguments)
        dx, dy = alignment.translation_nm
        _require(dy == 0 and dx > 0 and dx % grid == 0, 'shared interface channel frame or PDK grid differs')
        shared.add(pair)
        a,b = ranks[pair[0]],ranks[pair[1]]
        equalities.extend(((a,b,dx),(b,a,-dx)))
    count = max(ranks.values())+1
    edges = [(rank,rank+1,grid) for rank in range(count-1)] + equalities
    for left,right in sorted(adjacent-shared):
        distance = boxes[left][2]-boxes[right][0]+gap
        edges.append((ranks[left],ranks[right],((distance+grid-1)//grid)*grid))
    positions = [0]*count
    for iteration in range(count):
        changed = False
        for left,right,distance in edges:
            if positions[right] < positions[left]+distance:
                positions[right] = positions[left]+distance
                changed = True
        if not changed:
            break
        _require(iteration < count-1, 'shared boundary and column rank constraints are inconsistent')
    ys, previous_top = {}, None
    for row in rows:
        bottom, top = min(boxes[node][1] for node in row),max(boxes[node][3] for node in row)
        dy = -bottom if previous_top is None else previous_top+gap-bottom
        dy = ((dy+grid-1)//grid)*grid
        ys.update((node,dy) for node in row)
        previous_top = top+dy
    result = {node:(*frame[:4], frame[4]+positions[ranks[node]],frame[5]+ys[node])
              for node,frame in frames.items()}
    _require(all(all(value % grid == 0 for value in frame[4:]) for frame in result.values()),
             'constructed raw unit translation is off PDK grid')
    return result


def assemble_mos_mosaic(*, mosaic, units, tech, connect_model, source_members,
                        provenance_sha256, accessories=(), interfaces=(), checkpoint=None):
    """Materialize exact source ranks using intrinsic parts and source interfaces."""
    from clay.layout.mosaic_geometry import MosaicGeometry
    from clay.layout.spatial.intent_projection import replay_discrete_mosaic_intent
    from clay.layout.connectivity import ConnectModel
    from clay.layout.common_centroid_array import _CCFinger
    from clay.layout.occurrence_channels import compile_row_channel_inventory
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    from clay.layout.place.mosaic_row_compile import _mosaic_gap_nm
    from clay.layout.joint_occurrence_center_basis import capture_joint_channel_centers
    check = joint_checkpoint(checkpoint)
    mosaic = replay_discrete_mosaic_intent(mosaic)
    _sha(provenance_sha256, 'MOS mosaic provenance')
    lookup, kind, keys, frames, channels, expected = _prepare(mosaic,units,source_members,tech,check)
    _require(type(accessories) is tuple and all(type(value) is MosMosaicAccessory for value in accessories),
             'shared accessory inventory is malformed')
    for item in accessories:
        _require(item.anchor_occurrence_id in item.occurrence_ids and set(item.occurrence_ids) <= set(lookup),
                 'accessory anchor or source association differs')
        JointArtworkPart(item.part_id,'accessory',item.provenance_sha256,item.occurrence_ids,item.geometry_by_layer_nm)
    grid = pdk_manufacturing_grid_nm(tech)
    gap = ((_mosaic_gap_nm(tech)+grid-1)//grid)*grid
    frames = _positions(mosaic,lookup,frames,channels,expected,interfaces,accessories,gap,grid,keys,check)
    parts, occurrences, placed, fingers = [], [], [], []
    for node in mosaic.occurrences:
        check()
        unit, frame = lookup[node.node_id], frames[node.node_id]
        part_id = node.node_id+'/raw'
        layers = _move(_raw_layers(unit), frame)
        parts.append(JointArtworkPart(part_id,'raw_unit',provenance_sha256,(node.node_id,),layers))
        occurrences.append(JointRowOccurrence(node.node_id,node.logical_owner,part_id,unit.master,frame))
        for site in unit.sites:
            shapes = _move(((site.raw_layer,site.master_geometry_nm),),frame)[0][1]
            placed.append(OccurrenceTerminalSite(site.site_id,unit.master.variant_sha256,frame,shapes))
        layers = dict(layers)
        box = (_region(layers[keys.active]) & _region(layers[keys.poly])).bbox()
        gates = {site.source_net for site in unit.sites if 'g' in site.allowed_roles}
        _require(len(gates) == 1, 'source gate terminal denominator differs')
        fingers.append(_CCFinger(node.logical_owner,next(iter(gates)),box.left,box.right,
            (box.bottom+box.top)//2,node.transform,(box.left,box.bottom,box.right,box.top)))
    parts.extend(JointArtworkPart(item.part_id,'accessory',item.provenance_sha256,item.occurrence_ids,
        _move(item.geometry_by_layer_nm,frames[item.anchor_occurrence_id])) for item in accessories)
    parts = tuple(sorted(parts,key=lambda item:item.part_id))
    expected = tuple(sorted(expected.values(),key=lambda item:item.site_id))
    placed = tuple(sorted(placed,key=lambda item:item.site_id))
    routing, provenance = ConnectModel.from_tech(tech),pdk_provenance_digest(tech)
    inventory = compile_joint_terminal_inventory(artwork=joint_row_artwork(parts),expected_sites=expected,
        placed_sites=placed,connect_model=connect_model,routing_model=routing,
        pdk_provenance_sha256=provenance,checkpoint=check,limits=JointTerminalLimits())
    binding = JointRowGeometryBinding(parts,tuple(occurrences),expected,placed,connect_model,routing,
        tech.pdk_id,provenance,inventory,
        quality_centers=capture_joint_channel_centers(occurrences,tech=tech,family=kind))
    polygons = joint_row_polygons(parts)
    channels = compile_row_channel_inventory(occurrence_artwork=tuple((occ.owner,joint_row_polygons(tuple(
        part for part in parts if occ.occurrence_id in part.occurrence_ids))) for occ in occurrences),
        row_polygons=polygons,od_layer_keys=(keys.active,),poly_layer_key=keys.poly)
    box = _region(dict(joint_row_artwork(parts))[keys.active]).bbox()
    geometry = MosaicGeometry(polygons=polygons,kind=kind,comp_bbox=(box.left,box.bottom,box.right,box.top),
        fingers=tuple(fingers),joint_row_binding=binding,channel_inventory=channels,
        mosaic_lattice_intent=mosaic if mosaic.column_ranks else None,tech=tech)
    replay_joint_row_geometry(geometry,mosaic.members,tech=tech,members=source_members,checkpoint=check)
    return geometry
