"""Replay requested raw SD abutments against the actual native row artwork.

This proves enumerated local interfaces and their source declarations. Complete
all-layer spacing, body tracing, and physical admission remain separate checks.
The row terminal consumer owns the full joint connectivity replay.
"""
from clay.layout.fixed_artwork_abutment import align_fixed_artwork_abutment
from clay.layout.joint_row_geometry import (
    JointRowGeometryBinding, _region, _shapes, joint_checkpoint, joint_record_counter, joint_row_polygons,
)
from clay.layout.joint_terminal_inventory import _native_frame
from clay.layout.occurrence_artwork import polygon_records
from clay.layout.spatial.mosaic_boundary_joins import replay_mosaic_boundary_join_intent, mosaic_interface_mode


def _require(value, message):
    if not value:
        raise ValueError('mosaic boundary ' + message)


def replay_mosaic_boundary_entry_geometry(entry, geometry, *, tech=None, members=None):
    prefix = 'mosaic_boundary_joins:'
    refs = tuple(ref[len(prefix):] for ref in entry.declaration_refs if ref.startswith(prefix))
    intent = getattr(geometry, 'mosaic_boundary_intent', None)
    if not refs and intent is None:
        return None
    checkpoint = joint_checkpoint()
    _require(len(refs) == 1 and intent is not None, 'source declaration or geometry intent is missing')
    intent = replay_mosaic_boundary_join_intent(intent, members=members)
    _require(intent.sha256 == refs[0] and intent.source_mosaic_sha256 in entry.declaration_refs,
             'source intent hash differs from independent entry declaration')
    _require(getattr(entry.structure_kind, 'value', None) == 'discrete_mosaic', 'entry is not a discrete mosaic')
    binding = getattr(geometry, 'joint_row_binding', None)
    _require(type(binding) is JointRowGeometryBinding, 'complete joint source binding is missing or foreign')
    binding.__post_init__()
    from clay.layout.tech import LayoutTech
    from clay.layout.device_cell import _resolve_layer_keys
    tech = LayoutTech.resolve(binding.pdk_id) if tech is None else tech
    interfaces = {(item.left.node_id,item.right.node_id):item for item in intent.interfaces}
    occurrences = {occ.occurrence_id:occ for occ in binding.occurrences}
    all_nodes, used = [], set()
    for row in entry.macro.rows:
        checkpoint()
        nodes = []
        for atom in row.atoms:
            _require(atom.match_key.startswith('mosaic:'), 'row atom lacks a mosaic source identity')
            node_id, orientation = atom.match_key[len('mosaic:'):].rsplit(':',1)
            _require(node_id in occurrences, 'compiled occurrence is missing from independent source binding')
            occurrence = occurrences[node_id]
            _require(atom.atom_id == f'{entry.group}:r{row.row_index}:active:{occurrence.owner}:{node_id}',
                     'compiled source owner differs from raw occurrence')
            from clay.layout.joint_row_geometry import joint_d4_orientation
            _require(joint_d4_orientation(occurrence.master_to_local) == orientation,
                     'compiled occurrence orientation differs from raw native frame')
            nodes.append(node_id)
        _require(type(row.boundary_joins) is tuple and len(row.boundary_joins) == len(nodes)-1,
                 'macro lacks its complete per-edge boundary inventory')
        for left,right,edge,aa,bb in zip(nodes,nodes[1:],row.boundary_joins,row.atoms,row.atoms[1:]):
            edge.__post_init__()
            _require((edge.left_atom_id,edge.right_atom_id) == (aa.atom_id,bb.atom_id)
                     and edge.declaration_ref == prefix+intent.sha256, 'macro boundary adjacency or source differs')
            interface = interfaces.get((left,right))
            if interface is None:
                _require(edge.mode == 'isolated' and edge.gap_nm > 0 and not edge.interface_sha256,
                         'undeclared edge must remain isolated')
            else:
                _require(edge.mode == mosaic_interface_mode(interface) and edge.gap_nm == 0
                         and edge.interface_sha256 == interface.sha256, 'requested interface differs from macro edge')
                used.add((left,right))
        all_nodes.extend(nodes)
    _require(tuple(all_nodes) == tuple(occ.occurrence_id for occ in binding.occurrences)
             and len(set(all_nodes)) == len(all_nodes), 'complete ordered occurrence source denominator differs')
    _require(used == set(interfaces), 'interface inventory contains an undeclared or nonadjacent edge')
    _require(joint_record_counter(polygon_records(geometry.polygons)) ==
             joint_record_counter(polygon_records(joint_row_polygons(binding.parts))),
             'actual native artwork differs from retained complete physical parts')
    parts = {part.part_id:part for part in binding.parts}
    expected = {site.site_id:site for site in binding.expected_sites}
    placed = {site.site_id:site for site in binding.placed_sites}
    for interface in intent.interfaces:
        checkpoint()
        keys = _resolve_layer_keys(tech, interface.family)
        active, poly = tuple(keys.active), tuple(keys.poly)
        sites, channels, raw_artwork = [], [], []
        for endpoint in (interface.left, interface.right):
            occurrence = occurrences[endpoint.node_id]
            _require((occurrence.owner, occurrence.master.variant_sha256) == (endpoint.owner,endpoint.variant_sha256),
                     'interface raw owner or variant differs')
            master_sites = {site.site_id:site for site in occurrence.master.terminal_sites}
            _require(endpoint.site_id in master_sites and master_sites[endpoint.site_id].role_group == 'sd',
                     'interface does not name a complete raw SD site')
            identity = endpoint.node_id+'/'+endpoint.site_id
            _require(identity in expected and identity in placed, 'interface site ledger is missing')
            source, position = expected[identity], placed[identity]
            raw_site = master_sites[endpoint.site_id]
            _require((source.owner,source.occurrence_id,source.source_net,source.allowed_roles,source.variant_sha256) ==
                     (endpoint.owner,endpoint.node_id,interface.source_net,endpoint.roles,endpoint.variant_sha256),
                     'interface independent source net, aliases or variant differs')
            _require(raw_site.geometry_by_layer_nm == ((source.raw_layer,source.master_geometry_nm),)
                     and source.raw_layer == active and position.master_to_local == occurrence.master_to_local
                     and position.variant_sha256 == endpoint.variant_sha256, 'interface raw site layer or frame differs')
            transform = _native_frame(occurrence.master_to_local)
            _require((_region(position.geometry_nm) ^ _region(source.master_geometry_nm).transformed(transform)).is_empty(),
                     'placed SD site differs from its complete raw master region')
            _require(occurrence.raw_part_id in parts, 'raw artwork part is missing')
            raw = dict(parts[occurrence.raw_part_id].geometry_by_layer_nm)
            _require(active in raw and poly in raw, 'raw artwork lacks current PDK OD/poly masks')
            channel = _region(raw[active]) & _region(raw[poly])
            _require(not channel.is_empty(), 'raw artwork has no actual channel')
            sites.append(position.geometry_nm)
            channels.append(_shapes(channel))
            raw_artwork.append(parts[occurrence.raw_part_id].geometry_by_layer_nm)
        arguments = dict(left_site=sites[0],right_site=sites[1],
            left_channel=channels[0],right_channel=channels[1],checkpoint=checkpoint)
        if mosaic_interface_mode(interface) == 'fixed_artwork_shared_access':
            from clay.layout.fixed_artwork_shared_access import align_fixed_artwork_shared_access
            aligned = align_fixed_artwork_shared_access(**arguments, left_artwork=raw_artwork[0],
                right_artwork=raw_artwork[1],contact_layer=keys.contact,metal_layer=keys.metal1)
        else:
            aligned = align_fixed_artwork_abutment(**arguments)
        _require(aligned.translation_nm == (0,0), 'declared interface is not actually abutted in the current native frame')
    return intent
