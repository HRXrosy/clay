"""Rigid placement and one-stamp bridge for current complete resistor PCells."""
from collections import Counter
from dataclasses import asdict, dataclass

from clay.layout.source_resistor_pcell_assembly import (
    CurrentResistorPCellAssembly, _require, _polygons, _terminal_nodes, _sha,
)
from clay.layout.joint_row_geometry import joint_record_counter
from clay.layout.occurrence_artwork import polygon_records
from clay.layout.placed_geometry import placed_point_nm


def require_resistor_member_record(record, assembly, owner, orient):
    assembly.validate()
    rows = [row for row in assembly.members if row.owner == owner]
    _require(len(rows) == 1, 'native record owner is absent')
    row, = rows
    sizing = dict(row.source_member[5])
    _require((record.get('kind'), record.get('value'), record.get('M', 1),
        record.get('pins'), record.get('model', '') or '', str(orient).upper()) ==
        ('res', sizing.get('value', sizing.get('R')), 1, dict(row.source_member[4]),
         row.source_member[3] or '', row.orientation), 'native source record value/pins/model/orientation differs')
    _require(not any(record.get(key) is not None for key in ('bank_interconnect_plan', '_geometry_override')),
             'native PCell conflicts with another geometry construction')


def resistor_member_geometry(assembly, owner, *, tech):
    _require(type(assembly) is CurrentResistorPCellAssembly, 'native geometry binding is foreign')
    assembly.validate()
    _require(tech.pdk_id == assembly.pdk_id, 'native geometry PDK differs')
    geometry = assembly.plan.member_geometry(owner)
    object.__setattr__(geometry, 'source_resistor_assembly', assembly)
    return geometry


def replay_resistor_member_geometry(assembly, owner, geometry, *, tech):
    """Retain exact shared-PCell emission instead of packing owners separately.

    Shared parts have one emitter and several logical owners. Their envelopes
    are not independent collision objects; each emitter's full native multiset
    and every original terminal must remain unchanged. Physical signoff remains
    a separate obligation, as it is for the original source assembly.
    """
    expected = resistor_member_geometry(assembly, owner, tech=tech)
    _require(getattr(geometry, 'source_resistor_assembly', None) is assembly,
             'native member lost current shared PCell authority')
    _require(joint_record_counter(polygon_records(geometry.polygons)) ==
             joint_record_counter(polygon_records(expected.polygons)),
             'native shared PCell raw member geometry differs')
    _require(geometry.extent() == expected.extent() and geometry.orient == expected.orient
             and geometry.terminals == expected.terminals
             and geometry.terminal_accesses == expected.terminal_accesses
             and geometry.terminal_layers == expected.terminal_layers,
             'native shared PCell member envelope/orientation/terminal differs')
    return geometry


def resistor_assembly_translation(assembly, devices):
    """Replay all current partial emission views together against complete raw parts."""
    assembly.validate()
    offsets = set()
    for occurrence in assembly.members:
        view = devices.get(occurrence.owner)
        _require(view is not None and getattr(view, 'source_resistor_assembly', None) is assembly,
                 'placed member lost its actual assembly binding')
        box, native = view.bbox, occurrence.owned_bbox_nm
        _require(type(box) is tuple and len(box) == 4 and all(type(v) is int for v in box)
                 and (box[2]-box[0], box[3]-box[1]) == (native[2]-native[0], native[3]-native[1]),
                 'placed member extent differs from complete owned artwork')
        offsets.add((box[0]-native[0], box[1]-native[1]))
    _require(len(offsets) == 1, 'placed PCell member offsets changed')
    dx, dy = next(iter(offsets))
    for occurrence in assembly.members:
        view = devices[occurrence.owner]
        native = assembly.plan.member_geometry(occurrence.owner)
        actual = tuple((layer, datatype, tuple((x-dx, y-dy) for x, y in points))
                       for layer, datatype, points in polygon_records(view.polys))
        _require(joint_record_counter(actual) == joint_record_counter(polygon_records(native.polygons)),
                 'placed member full original shape emission changed')
        _require(view.orientation == occurrence.orientation and view.term_layers == native.terminal_layers,
                 'placed member orientation or terminal layer changed')
        expected = {role: (t.center[0]+dx, t.center[1]+dy) for role, t in native.terminals.items()}
        accesses = {role: tuple((t.center[0]+dx, t.center[1]+dy) for t in rows)
                    for role, rows in native.terminal_accesses.items()}
        _require(view.terminals == expected and view.terminal_accesses == accesses,
                 'placed member terminal/access geometry changed')
    return dx, dy


@dataclass(frozen=True, slots=True)
class ResistorStampBinding:
    assembly: CurrentResistorPCellAssembly
    origin_nm: tuple[int, int]


class SourceResistorStamp:
    """One actual native multiset with complete logical owners and original sites."""
    def __init__(self, assembly, *, devices, translation_nm):
        actual = resistor_assembly_translation(assembly, devices)
        _require(actual == translation_nm, 'bridge translation differs from actual member views')
        x0, y0, x1, y1 = assembly.bbox_nm
        origin = ((x0+x1)//2, (y0+y1)//2)
        self.source_resistor_stamp_binding = ResistorStampBinding(assembly, origin)
        self.composite_members = assembly.plan.members
        self.orient = assembly.members[0].orientation
        self.name = 'resistor-pcell:' + ':'.join(self.composite_members)
        self.polys_by_layer = {}
        dx, dy = translation_nm
        for owner in self.composite_members:
            for layer, datatype, points in polygon_records(devices[owner].polys):
                self.polys_by_layer.setdefault((layer, datatype), []).append(
                    [((x-dx-origin[0])/1000, (y-dy-origin[1])/1000) for x, y in points])
        self.position_um = ((dx+origin[0])/1000, (dy+origin[1])/1000)
        resistor_stamp_owner_records(self, self.position_um, self.composite_members)


def is_resistor_stamp(stamp):
    return type(stamp) is SourceResistorStamp or hasattr(stamp, 'source_resistor_stamp_binding')


def _stamp_binding(stamp):
    _require(type(stamp) is SourceResistorStamp, 'current shared stamp is foreign')
    binding = getattr(stamp, 'source_resistor_stamp_binding', None)
    _require(type(binding) is ResistorStampBinding and type(binding.assembly) is CurrentResistorPCellAssembly,
             'current stamp binding is foreign')
    binding.assembly.validate()
    _require(stamp.composite_members == binding.assembly.plan.members, 'current stamp owner denominator differs')
    x0, y0, x1, y1 = binding.assembly.bbox_nm
    _require(binding.origin_nm == ((x0+x1)//2, (y0+y1)//2), 'current stamp origin changed')
    return binding


def resistor_stamp_owner_records(stamp, position, members):
    from clay.layout.owned_artwork import _stamp_records
    binding = _stamp_binding(stamp)
    _require(members == binding.assembly.plan.members, 'current complete source owner denominator differs')
    ox, oy = binding.origin_nm
    all_records, owners = [], {member: [] for member in members}
    for part in binding.assembly.plan.parts:
        records = tuple((layer, datatype, tuple(placed_point_nm((x-ox)/1000,
            (y-oy)/1000, position) for x, y in points))
            for layer, datatype, points in polygon_records(_polygons((part,))))
        all_records.extend(records)
        for owner in part.owners:
            owners[owner].extend(records)
    _require(joint_record_counter(_stamp_records(stamp, position)) == joint_record_counter(tuple(all_records)),
             'current full stamped raw polygon multiset changed')
    return {owner: tuple(records) for owner, records in owners.items()}


def resistor_stamp_occurrences(stamp, position):
    binding = _stamp_binding(stamp)
    resistor_stamp_owner_records(stamp, position, binding.assembly.plan.members)
    ox, oy = binding.origin_nm
    return {row.owner: ((*placed_point_nm((row.center2_nm[0]//2-ox)/1000,
            (row.center2_nm[1]//2-oy)/1000, position), row.orientation),)
            for row in binding.assembly.members}


def bridge_current_resistor_assemblies(placement, placed, passives, geometry, placement_um):
    """Consume complete certified macros once, preserving every source passive."""
    plans = getattr(getattr(placement, 'rigid_nonmos_macro_plans', None), 'plans', ())
    selected = [plan for plan in plans if type(getattr(plan, 'assembly', None)) is CurrentResistorPCellAssembly]
    by_name = {passive.name: passive for passive in passives}
    consumed = set()
    for plan in selected:
        assembly = plan.assembly
        for row in assembly.members:
            passive = by_name.get(row.owner)
            _require(passive is not None, 'bridge source passive is absent')
            require_resistor_member_record(dict(kind=passive.kind, model=passive.model,
                value=passive.value, M=passive.M, pins=passive.nets), assembly, row.owner, passive.orient)
            _require(row.owner not in consumed, 'bridge source owner is repeated')
            consumed.add(row.owner)
            box = placed[row.owner].bbox
            placement_um[row.owner] = ((box[0]+box[2])/2000, (box[1]+box[3])/2000)
        translation = plan.translation_nm(placed)
        stamp = SourceResistorStamp(assembly, devices=placed, translation_nm=translation)
        _require(stamp.name not in geometry and stamp.name not in placement_um, 'bridge assembly identity collides')
        geometry[stamp.name], placement_um[stamp.name] = stamp, stamp.position_um
    marked = {name for name, view in placed.items() if getattr(view, 'source_resistor_assembly', None) is not None}
    _require(marked == consumed, 'bridge cannot omit retained resistor assembly authority')
    return consumed


def verified_resistor_assembly_accesses(stamp, *, passive, position, tech):
    """Actual complete PCell graph plus each source member's original head sites."""
    binding = _stamp_binding(stamp)
    assembly = binding.assembly
    _require(tech.pdk_id == assembly.pdk_id, 'route access PDK differs')
    require_resistor_member_record(dict(kind=passive.kind, model=passive.model, value=passive.value,
        M=passive.M, pins=passive.nets), assembly, passive.name, passive.orient)
    resistor_stamp_owner_records(stamp, position, assembly.plan.members)
    terminal_nodes, nodes = _terminal_nodes(assembly.plan, assembly.connect_model, return_nodes=True)
    _require(terminal_nodes == assembly.native_terminal_nodes,
             'actual native PCell terminal partition differs')
    row = next(row for row in assembly.members if row.owner == passive.name)
    ox, oy = binding.origin_nm
    layer = dict(assembly.plan.semantic_layers)['metal1']
    from clay.layout.route.pins import _manufacturing_grid_access_on_stamp
    from clay.layout.passive_terminal_binding import _node_for_box, _probe
    point_origin = placed_point_nm(0, 0, position)
    dx, dy = point_origin[0]-ox, point_origin[1]-oy
    result = []
    for role, boxes in row.terminal_boxes:
        expected = next(node for owner, selected, _net, node in terminal_nodes
                        if (owner, selected) == (row.owner, role))
        for index, box in enumerate(boxes):
            point = placed_point_nm(((box[0]+box[2])//2-ox)/1000,
                                   ((box[1]+box[3])//2-oy)/1000, position)
            actual = _manufacturing_grid_access_on_stamp(stamp, position, layer, point, tech)
            node = _node_for_box(nodes, layer, _probe((actual[0]-dx, actual[1]-dy)))
            _require(f'node:{node}' == expected, 'final grid access left its original complete head node')
            result.append((role, dict(row.source_member[4])[role], layer, actual, index))
    return tuple(result)


def resistor_structure_snapshot(assemblies):
    """Opaque invocation identity plus detached verified values, never live-object traversal."""
    _require(type(assemblies) is tuple and all(type(value) is CurrentResistorPCellAssembly
             for value in assemblies), 'structure requires exact current assembly tuple')
    return tuple((id(value), id(value.producer), value.sha256, value.payload()) for value in assemblies)


def replay_current_resistor_structure(assemblies, *, source_cell, tech, placement, intent,
        expected_counts, geometry=None, placement_um=None, observation=None):
    """Replay complete source assemblies and optionally their actual final stamp.

The optional final observation must describe these exact current polygons,
owner envelopes and body centers. No independently supplied PASS is consumed.
The returned evidence does not mint global structure or physical admission.
"""
    from clay.layout.source_resistor_pcell_assembly import replay_current_resistor_assembly
    from clay.layout.owned_artwork import _bbox
    from clay.layout.place.final_quality_observation import FinalQualityGeometryObservation
    before = resistor_structure_snapshot(assemblies)
    _require(type(expected_counts) is dict and all(type(count) is int and count == 1
             for count in expected_counts.values()), 'complete source resistor unit counts differ')
    owners = tuple(sorted(row.owner for value in assemblies for row in value.members))
    _require(owners == tuple(sorted(expected_counts)), 'structure complete source owner cover differs')
    current = tuple(value for value in getattr(getattr(placement, 'rigid_nonmos_macro_inputs', None),
        'native_assemblies', ()) if type(value) is CurrentResistorPCellAssembly)
    _require(len(current) == len(assemblies) and all(a is b for a, b in zip(current, assemblies)),
             'structure source assembly identity differs from actual compiler inputs')
    plans = tuple(plan for plan in getattr(getattr(placement, 'rigid_nonmos_macro_plans', None), 'plans', ())
                  if type(getattr(plan, 'assembly', None)) is CurrentResistorPCellAssembly)
    _require(len(plans) == len(assemblies) and {id(plan.assembly) for plan in plans} == {id(v) for v in assemblies},
             'structure materialized macro denominator differs')
    rows = []
    for value in assemblies:
        replay_current_resistor_assembly(value, source_cell=source_cell, tech=tech, intent=intent)
        plan, = [plan for plan in plans if plan.assembly is value]
        translation = plan.translation_nm(placement.devices)
        rows.append(dict(assembly_sha256=value.sha256, macro_id=plan.macro_id,
            logical_owners=value.plan.members, occurrence_path=value.plan.occurrence_path,
            raw_master_sha256=_sha(value.plan.canonical_data()['raw_master_geometry']),
            source_member_translation_nm=translation, raw_to_local=value.plan.raw_to_local,
            occurrences=tuple(dict(owner=row.owner, occurrence_id=row.occurrence_id,
                segment_index=row.segment_index, source_member=row.source_member,
                center2_nm=row.center2_nm, role_to_head=row.role_to_head,
                original_head_sites=asdict(row.primitive)['terminals'],
                original_part_indices=row.primitive.owned_part_indices) for row in value.members),
            native_terminal_nodes=value.native_terminal_nodes,
            raw_parts=len(value.plan.parts), shared_raw_parts=sum(len(p.owners)>1 for p in value.plan.parts)))
    _require((geometry is None) == (placement_um is None), 'final structure needs geometry and placement together')
    if observation is not None:
        _require(geometry is not None and type(observation) is FinalQualityGeometryObservation,
                 'final structure needs exact same-candidate observation and geometry')
        observation.__post_init__()
    if geometry is not None:
        from collections.abc import Mapping
        _require(isinstance(geometry, Mapping) and isinstance(placement_um, Mapping),
                 'final structure needs complete current geometry mappings')
        stamps = tuple((name, stamp) for name, stamp in geometry.items() if is_resistor_stamp(stamp))
        _require(len(stamps) == len(assemblies), 'final structure PCell stamp denominator differs')
        observed = {} if observation is None else {row.logical_id: row for row in observation.logical_geometry}
        artwork = {} if observation is None else dict(observation.source_artwork_bboxes_nm)
        for value, evidence in zip(assemblies, rows):
            hits = [(name, stamp) for name, stamp in stamps if _stamp_binding(stamp).assembly is value]
            _require(len(hits) == 1, 'final structure lost same materialization stamp identity')
            name, stamp = hits[0]
            position = placement_um[name]
            owned = resistor_stamp_owner_records(stamp, position, value.plan.members)
            centers = resistor_stamp_occurrences(stamp, position)
            origin = stamp.source_resistor_stamp_binding.origin_nm
            global_origin = placed_point_nm(0, 0, position)
            delta = (global_origin[0]-origin[0], global_origin[1]-origin[1])
            evidence.update(final_stamp=name, final_translation_nm=delta,
                final_owned_bboxes_nm=tuple((owner, _bbox(records)) for owner, records in owned.items()),
                final_occurrences_nm=tuple(sorted(centers.items())))
            if observation is not None:
                for owner in value.plan.members:
                    logical = observed.get(owner)
                    _require(logical is not None and logical.kind in {'res', 'resistor'}
                        and logical.occurrence_basis == 'materialized_resistor_body_center'
                        and logical.occurrences == centers[owner] and artwork.get(owner) == _bbox(owned[owner]),
                        'final structure differs from the same-candidate observed resistor geometry')
    for value in assemblies:
        value.producer.require_current()
    _require(resistor_structure_snapshot(assemblies) == before, 'structure source/native operands changed during replay')
    return dict(logical_owners=list(owners), units=len(owners), sites=2*len(owners),
        pcells=len(assemblies), raw_parts=sum(len(value.plan.parts) for value in assemblies),
        materializations=rows, final_geometry_replayed=geometry is not None,
        same_candidate_observation_replayed=observation is not None,
        quality_admission='NOT_ESTABLISHED')
