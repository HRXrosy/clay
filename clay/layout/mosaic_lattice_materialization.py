"""Join source channel ranks to the complete actual native occurrence cover.

Ranks preserve equality and order, without a reference coordinate or pitch.
This seam does not establish inter-device spacing, body tracing or admission;
the joint row consumer separately replays complete artwork and electrical sites.
"""
from clay.layout.joint_row_geometry import (
    JointRowGeometryBinding, joint_checkpoint, joint_d4_orientation,
    joint_record_counter, joint_row_polygons, _region,
)
from clay.layout.occurrence_artwork import polygon_records
from clay.layout.spatial.intent_projection import replay_discrete_mosaic_intent


def _require(condition, message):
    if not condition:
        raise ValueError('mosaic lattice ' + message)


def mosaic_lattice_geometry_payload(geometry):
    value = getattr(geometry, 'mosaic_lattice_intent', None)
    if value is None:
        return None
    value = replay_discrete_mosaic_intent(value)
    _require(bool(value.column_ranks), 'geometry intent lacks its complete column ranks')
    return value.canonical_data()


def replay_mosaic_lattice_entry_geometry(entry, geometry, *, tech=None, members=None):
    prefix = 'mosaic_channel_lattice:'
    refs = tuple(ref[len(prefix):] for ref in entry.declaration_refs if ref.startswith(prefix))
    intent = getattr(geometry, 'mosaic_lattice_intent', None)
    if not refs and intent is None:
        return None
    checkpoint = joint_checkpoint()
    _require(len(refs) == 1 and intent is not None, 'independent source or native intent is missing')
    intent = replay_discrete_mosaic_intent(intent)
    _require(intent.column_ranks and refs[0] == intent.sha256 and intent.sha256 in entry.declaration_refs,
             'complete source intent or hash differs')
    _require('asir:' + intent.source_asir_sha256 in entry.declaration_refs
             and entry.group == intent.mosaic_id and entry.consumed_devices == intent.members
             and getattr(entry.structure_kind, 'value', None) == 'discrete_mosaic',
             'compiled source identity differs')
    binding = getattr(geometry, 'joint_row_binding', None)
    _require(type(binding) is JointRowGeometryBinding, 'complete joint source binding is missing')
    binding.__post_init__()
    occurrences = {occ.occurrence_id:occ for occ in binding.occurrences}
    expected_nodes = tuple(node.node_id for node in intent.occurrences)
    _require(expected_nodes == tuple(occ.occurrence_id for occ in binding.occurrences)
             and len(occurrences) == len(expected_nodes), 'complete ordered occurrence cover differs')
    _require(len(entry.macro.rows) == len(intent.rows), 'complete compiled row cover differs')
    for row, projected in zip(entry.macro.rows, intent.rows):
        checkpoint()
        _require(len(row.atoms) == len(projected), 'compiled row occurrence count differs')
        for atom, node in zip(row.atoms, projected):
            occurrence = occurrences[node.node_id]
            _require(occurrence.owner == node.logical_owner
                     and joint_d4_orientation(occurrence.master_to_local) == node.transform
                     and atom.match_key == f'mosaic:{node.node_id}:{node.transform}'
                     and atom.atom_id == f'{entry.group}:r{row.row_index}:active:{node.logical_owner}:{node.node_id}',
                     'compiled source occurrence, transform or order differs')
    _require(joint_record_counter(polygon_records(geometry.polygons)) ==
             joint_record_counter(polygon_records(joint_row_polygons(binding.parts))),
             'actual native artwork differs from the complete physical parts')
    from clay.layout.tech import LayoutTech
    from clay.layout.device_cell import _resolve_layer_keys
    tech = LayoutTech.resolve(binding.pdk_id) if tech is None else tech
    _require(tech.pdk_id == binding.pdk_id, 'current PDK differs')
    parts = {part.part_id:part for part in binding.parts}
    points = {}
    for occurrence in binding.occurrences:
        checkpoint()
        kind = getattr(geometry, 'kind', None) if members is None else members[occurrence.owner].kind
        _require(kind in ('nmos','pmos'), 'source family is unsupported')
        keys = _resolve_layer_keys(tech, kind)
        raw = parts.get(occurrence.raw_part_id)
        _require(raw is not None and raw.kind == 'raw_unit' and raw.occurrence_ids == (occurrence.occurrence_id,),
                 'raw occurrence part association differs')
        layers = dict(raw.geometry_by_layer_nm)
        _require(tuple(keys.active) in layers and tuple(keys.poly) in layers,
                 'raw active/poly channel layers are missing')
        channel = (_region(layers[tuple(keys.active)]) & _region(layers[tuple(keys.poly)])).merged()
        box = channel.bbox()
        _require(channel.size() == 1 and box.width() > 0 and box.height() > 0
                 and channel.area() == box.width()*box.height(),
                 'raw occurrence must expose exactly one complete rectangular channel')
        points[occurrence.occurrence_id] = (box.left+box.right, box.bottom+box.top)
    _require(len(set(points.values())) == len(points), 'actual channel slots are duplicated')
    xs = {x:rank for rank,x in enumerate(sorted({p[0] for p in points.values()}))}
    ys = {y:rank for rank,y in enumerate(sorted({p[1] for p in points.values()}))}
    expected_x = {f'{owner}[{ordinal:03d}]':rank for owner,ordinal,rank in intent.column_ranks}
    expected_y = {node.node_id:rank for rank,row in enumerate(intent.rows) for node in row}
    _require({node:xs[point[0]] for node,point in points.items()} == expected_x
             and {node:ys[point[1]] for node,point in points.items()} == expected_y,
             'actual raw channel row/column ranks differ from source')
    checkpoint()
    return intent
