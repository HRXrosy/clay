"""Join current source/native operands to one actual final compact scene.

This is a placement-time preservation observation. It does not run routing or
claim reference extraction, complete structural acceptance, body connectivity,
DRC, or LVS. Auxiliary artwork is bound to the same scene without upgrading the
scope of its original proof.
"""
from dataclasses import dataclass
import hashlib
import json

from clay.layout.place.current_structure_operands import (
    CurrentStructureOperands, _checkpoint, _digest, _json, _native_values, _require,
)

_SEAL = object()


def _records(stamp, position):
    from clay.layout.owned_artwork import _stamp_records
    return _stamp_records(stamp, position)


def _move(records, delta):
    return tuple((layer, datatype, tuple((x+delta[0], y+delta[1]) for x,y in ring))
                 for layer,datatype,ring in records)


def _same_records(first, second, message):
    from clay.layout.joint_row_geometry import joint_record_counter
    _require(joint_record_counter(first) == joint_record_counter(second), message)


def _native_metadata(stamp):
    values = _native_values(tuple(getattr(stamp, name, None) for name in (
        'name', 'composite_members', 'joint_row_stamp_binding', 'materialized_passive_access',
        'passive_artwork_origin_um', 'bjt_unit_occurrences', 'terminal_offsets_um',
        'terminal_accesses_um', 'terminal_layers', 'fixed_pcell', 'layered_placement_tap_child',
        'physical_occurrences_by_member')))
    binding = getattr(stamp, 'source_resistor_stamp_binding', None)
    if binding is None:
        return values
    from clay.layout.source_resistor_placement import _stamp_binding, resistor_structure_snapshot
    binding = _stamp_binding(stamp)
    return values, binding.origin_nm, resistor_structure_snapshot((binding.assembly,))


def _scene(geometry, positions):
    _require(type(geometry) is dict and type(positions) is dict and set(geometry) <= set(positions),
             'final stamp/position denominator differs')
    _require(all(type(name) is str and name for name in geometry), 'final stamp identity is malformed')
    return tuple((name, tuple(sorted(_records(stamp, positions[name]))), _native_metadata(stamp))
                 for name,stamp in sorted(geometry.items()))


def _partition(certificate, geometry, positions, intent, source_names):
    from clay.layout.place.actual_materialized_partition import actual_materialized_partition
    from clay.layout.place.placement_geometry import _source_device_bboxes_nm
    partition = actual_materialized_partition(certificate, geometry, positions, intent)
    blocks, stamps = partition.blocks, partition.block_stamps
    _require(tuple(sorted(blocks)) == certificate.materialized_block_bboxes,
             'final actual physical block bboxes differ from certificate')
    ownership = partition.source_members
    _require(ownership == certificate.materialized_block_source_members
             and tuple(sorted(name for _block, names in ownership for name in names)) == source_names,
             'final complete source ownership differs from certificate')
    boxes = _source_device_bboxes_nm(geometry, positions, source_names)
    _require(tuple(sorted(boxes.items())) == certificate.materialized_source_member_bboxes,
             'final current source bbox/frame differs from certificate')
    return boxes, {name:tuple(sorted(values)) for name,values in stamps.items()}


def _mos(operands, geometry, positions):
    from clay.layout.joint_placement_terminals import project_joint_placement_face
    from clay.layout.joint_row_geometry import joint_occurrence_centers
    from clay.layout.joint_row_stamp import JointRowStampBinding
    from clay.layout.placed_geometry import placed_point_nm
    members = {member.device:member for member in getattr(operands.placement.row_program, 'members', ())}
    rows, names = [], set()
    for entry, realization in operands.rows:
        _checkpoint()
        stamp = geometry.get(entry.group)
        binding = getattr(stamp, 'joint_row_stamp_binding', None)
        _require(type(binding) is JointRowStampBinding, 'final MOS stamp native binding is missing')
        _require(getattr(stamp, 'physical_occurrences_by_member', None) ==
                 joint_occurrence_centers(binding.source, binding.native_origin_nm),
                 'final MOS native occurrence metadata differs from its complete raw masters')
        position = positions[entry.group]
        origin = placed_point_nm(0, 0, position)
        delta = tuple(origin[index]-binding.native_origin_nm[index] for index in range(2))
        face = project_joint_placement_face(native_geometry=realization.geometry, stamp=stamp, position=position,
            members=entry.consumed_devices, delta=delta, tech=operands.tech,
            native_row_sha256=realization.geometry_sha256, source_members=members)
        rows.append({'stamp_id':entry.group, 'native_row_sha256':realization.geometry_sha256,
            'stamp_binding_sha256':binding.sha256, 'delta_nm':list(delta),
            'units':len(binding.source.occurrences), 'sites':len(binding.source.expected_sites),
            'joint_components':len(face.joint_inventory.joint_components),
            'routing_access_components':len(face.joint_inventory.routing_components)})
        names.add(entry.group)
    return rows, names


def _mim(operands, geometry, positions):
    from clay.layout.passive_terminal_binding import verified_passive_route_accesses
    from clay.layout.route.route_cell import bridge_devices_from_cell_def
    _devices, passive_rows = bridge_devices_from_cell_def(operands.source_cell)
    passives = {row.name:row for row in passive_rows}
    rows, names = [], set()
    for native in operands.mim_materializations:
        _checkpoint()
        name = native.source_member[0]
        stamp = geometry.get(name)
        expected = operands.placement.devices[name].materialized_passive_access
        actual = getattr(stamp, 'materialized_passive_access', None)
        _require(actual is not None and _native_values(actual) == _native_values(expected),
                 'final MIM stamp native source/site binding differs')
        _require(getattr(stamp, 'fixed_pcell', None) == operands.placement.devices[name].fixed_pcell,
                 'final MIM native fixed metadata differs from its current source operand')
        accesses, nodes = verified_passive_route_accesses(stamp, passive=passives[name],
                                                        position=positions[name], tech=operands.tech)
        rows.append({'stamp_id':name, 'units':native.geometry.finger_count,
            'sites':len(native.site_ledger), 'current_nodes':len(nodes), 'current_accesses':len(accesses),
            'native_geometry_sha256':hashlib.sha256(native.geometry_snapshot).hexdigest()})
        names.add(name)
    return rows, names


def _layer_shapes(records):
    import klayout.db as kdb
    from clay.layout.bjt_local_assembly import _shapes
    regions = {}
    for layer, datatype, ring in records:
        _checkpoint()
        regions.setdefault((layer,datatype), kdb.Region()).insert(kdb.Polygon([kdb.Point(x,y) for x,y in ring]))
    return tuple((key,_shapes(region.merged())) for key,region in sorted(regions.items()))


def _bjt_metadata(operands, name, stamp, position, final_box):
    from clay.layout.placed_geometry import placed_point_nm
    native = operands.bjt_native[name]
    placed = operands.placement.devices[name]
    _require(getattr(stamp, 'bjt_unit_occurrences', None) == native.unit_occurrences,
             'final BJT native occurrence metadata is missing or changed')
    _require(getattr(stamp, 'fixed_pcell', None) == placed.fixed_pcell
             and getattr(stamp, 'terminal_layers', None) == placed.term_layers,
             'final BJT native terminal/fixed metadata differs')
    def local(point, box):
        return point[0]-box[0], point[1]-box[1]
    points = {role:local(placed_point_nm(*point, position), final_box)
              for role,point in getattr(stamp, 'terminal_offsets_um', {}).items()}
    expected = {role:local(point, placed.bbox) for role,point in placed.terminals.items()}
    _require(points == expected, 'final BJT current terminal frame differs from native member')
    accesses = {role:tuple(local(placed_point_nm(*point, position), final_box) for point in values)
                for role,values in getattr(stamp, 'terminal_accesses_um', {}).items()}
    expected = {role:tuple(local(point, placed.bbox) for point in values)
                for role,values in placed.terminal_accesses.items()}
    _require(accesses == expected, 'final BJT current access frame differs from native member')


def _bjts(operands, geometry, positions, boxes):
    from clay.layout.bjt_local_assembly import replay_bjt_local_assembly
    from clay.layout.occurrence_artwork import polygon_records
    rows, names = [], set()
    for binding in operands.nonmos_assemblies:
        _checkpoint()
        offsets = dict(binding.owner_offsets_nm)
        common_delta, actual_members = None, []
        for member in binding.members:
            name = member.owner
            stamp = geometry.get(name)
            _require(stamp is not None, 'final BJT source member stamp is missing')
            # The signed member bbox and the native assembly bbox define the
            # one allowed integer transform; actual artwork must realize it.
            delta = tuple(boxes[name][index]-member.bbox_nm[index] for index in range(2))
            _require(common_delta is None or delta == common_delta, 'final BJT rigid assembly member frame differs')
            common_delta = delta
            native = operands.bjt_native[name]
            expected = _move(polygon_records(native.polygons), tuple(offsets[name][i]+delta[i] for i in range(2)))
            actual = _records(stamp, positions[name])
            _same_records(actual, expected, 'final BJT complete native stamp artwork differs')
            _bjt_metadata(operands, name, stamp, positions[name], boxes[name])
            actual_members.append((name, _layer_shapes(_move(actual, tuple(-value for value in delta)))))
            names.add(name)
        replay_bjt_local_assembly(binding, source_cell=operands.source_cell, tech=operands.tech,
            intent=operands.intent, member_artwork=tuple(actual_members))
        rows.append({'logical_owners':[member.owner for member in binding.members],
            'units':len(binding.occurrences), 'sites':len(binding.joint_inventory.expected_sites),
            'assembly_sha256':binding.sha256, 'delta_nm':list(common_delta)})
    return rows, names


@dataclass(frozen=True, slots=True, init=False)
class FinalStructureEvidence:
    _data: bytes
    _seal: object

    def __init__(self, *args, **kwargs):
        raise TypeError('FinalStructureEvidence requires current final consumption')

    def canonical_data(self):
        _require(self._seal is _SEAL, 'final evidence native capture is missing')
        return json.loads(self._data)

    @property
    def sha256(self):
        return hashlib.sha256(self._data).hexdigest()


def observe_final_structure_consumption(operands, *, compact, source_cell, tech, intent, observation=None):
    from clay.layout.flow_placement import DeclaredCompactOverride
    from clay.layout.place.placement_certificates import FinalPlacementCertificate
    from clay.layout.place.final_placement_state import final_placement_state_digest
    from clay.layout.route.exact_artifact_evidence import _emitted_device_snapshot_sha256
    _require(type(operands) is CurrentStructureOperands, 'final observation requires current native capture')
    operands._require_current()
    _require(source_cell is operands.source_cell and tech is operands.tech and intent is operands.intent,
             'final observation source/PDK/intent invocation differs')
    _require(type(compact) is DeclaredCompactOverride, 'final observation requires actual compact output')
    certificate = compact.final_placement_certificate
    _require(type(certificate) is FinalPlacementCertificate, 'final placement certificate is missing')
    certificate.__post_init__()
    state = final_placement_state_digest(certificate)
    positions, geometry = compact.ring_inclusive
    initial = _scene(geometry, positions)
    source_names = tuple(row[0] for row in operands.population)
    boxes, partition = _partition(certificate, geometry, positions, intent, source_names)
    mos, mos_names = _mos(operands, geometry, positions)
    mim, mim_names = _mim(operands, geometry, positions)
    bjts, bjt_names = _bjts(operands, geometry, positions, boxes)
    _require(not (mos_names & mim_names or mos_names & bjt_names or mim_names & bjt_names),
             'final family stamp ownership is repeated')
    data = operands.canonical_data()
    resistor_stamps = set()
    if operands.resistor_assemblies:
        from clay.layout.source_resistor_placement import replay_current_resistor_structure
        from clay.layout.place.final_quality_observation import observe_materialized_source
        if observation is None:
            observation = observe_materialized_source(cell=source_cell, compact=compact)
        _require(observation.final_placement_state_sha256 == state,
                 'final resistor observation belongs to a different candidate')
        resistors = replay_current_resistor_structure(operands.resistor_assemblies, source_cell=source_cell,
            tech=tech, placement=operands.placement, intent=intent,
            expected_counts={name:count for name,family,count,_kind in operands.population if family == 'resistor'},
            geometry=geometry, placement_um=positions, observation=observation)
        _require(all(resistors[key] == data['families']['resistor'][key]
                     for key in ('logical_owners', 'units', 'sites', 'pcells', 'raw_parts')),
                 'final resistor family denominator differs')
        data['families']['resistor']['final_consumption'] = resistors
        resistor_stamps = {row['final_stamp'] for row in resistors['materializations']}
    for family, records in (('mos',mos),('mim',mim),('bjt',bjts)):
        if family in data['families']:
            _require(sum(row['units'] for row in records) == data['families'][family]['units']
                and sum(row['sites'] for row in records) == data['families'][family]['sites'],
                'final family native unit/site denominator differs')
            data['families'][family]['final_consumptions'] = records
    data.update(schema='clay.current-structure-consumption/v1', operands_sha256=operands.sha256,
        final_placement_state_sha256=state,
        final_artwork_sha256=_emitted_device_snapshot_sha256(geometry, positions),
        actual_stamp_partition=partition,
        other_scene_stamp_ids=sorted(set(geometry)-mos_names-mim_names-bjt_names-resistor_stamps),
        auxiliary_scope='scene_binding_only_no_new_body_or_signoff_claim', physical_admission=False)
    _require(_scene(geometry, positions) == initial and final_placement_state_digest(certificate) == state,
             'final actual artwork/metadata/certificate changed during observation')
    operands._require_current()
    result = object.__new__(FinalStructureEvidence)
    object.__setattr__(result, '_data', _json(data))
    object.__setattr__(result, '_seal', _SEAL)
    return result


__all__ = ['FinalStructureEvidence', 'observe_final_structure_consumption']
