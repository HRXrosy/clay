"""Carry a complete source row and independent PDK tap stamps as one owner.

This boundary changes physical block ownership, never logical row membership or
tap geometry. Body access is checked again against every current supplied row;
the result grants no routed body net, guard, isolation, DRC or LVS conclusion.
Each assembly owns one complete PMOS row. All foreign context stays separate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

from clay.layout.joint_row_geometry import joint_checkpoint, joint_record_counter
from clay.layout.owned_artwork import _bbox, _hash, _stamp_records
from clay.layout.well_component_access import (
    JointWellRow, PdkWellTapConstruction, PdkWellTapRequest, WellComponentAccessBinding, WellForeignArtwork,
    _from_records, _records, _row_state, _source, _technology,
    capture_well_component_access, materialize_pmos_component_taps,
    replay_well_component_access,
)

_SEAL = object()
_MAX_ITEMS = 4096


def _require(condition, message):
    if not condition:
        raise ValueError('layered assembly ' + message)


def _tuple(value, field, *, nonempty=True):
    _require(type(value) is tuple and len(value) <= _MAX_ITEMS and (value or not nonempty),
             field + ' requires a bounded exact tuple')
    return value


def _name(value, field):
    _require(type(value) is str and 0 < len(value) <= 4096 and value == value.strip(), field + ' is malformed')
    return value


def _offset(value):
    _require(type(value) is tuple and len(value) == 2
             and all(type(v) is int and abs(v) <= 1_000_000_000 for v in value),
             'child frame needs bounded exact integer nm')
    return value


def _move(records, delta):
    dx, dy = _offset(delta)
    return tuple((layer, datatype, tuple((x+dx, y+dy) for x, y in points))
                 for layer, datatype, points in records)


def _same(left, right, message):
    _require(joint_record_counter(left) == joint_record_counter(right), message)


class _Captured:
    def __init__(self, *args, **kwargs):
        raise TypeError(type(self).__name__ + ' requires current native capture')

    def validate(self):
        _require(self._seal is _SEAL and _hash(self.payload()) == self._content_sha256,
                 'captured binding or projection changed')

    @property
    def sha256(self):
        self.validate()
        return self._content_sha256


def _capture(kind, **fields):
    value = object.__new__(kind)
    for name, content in fields.items():
        object.__setattr__(value, name, content)
    object.__setattr__(value, '_seal', _SEAL)
    object.__setattr__(value, '_content_sha256', _hash(value.payload()))
    value.validate()
    return value


@dataclass(frozen=True, slots=True)
class LayeredTapChild:
    stamp_name: str
    construction: PdkWellTapConstruction
    local_offset_nm: tuple[int, int]

    def __post_init__(self):
        _name(self.stamp_name, 'tap stamp name')
        _offset(self.local_offset_nm)
        _require(type(self.construction) is PdkWellTapConstruction, 'tap child lacks exact PDK construction')
        self.construction.validate()

    def payload(self):
        self.__post_init__()
        return dict(stamp_name=self.stamp_name, construction_sha256=self.construction.sha256,
                    local_offset_nm=self.local_offset_nm)


@dataclass(frozen=True, slots=True, init=False)
class LayeredPlacementAssemblyBinding(_Captured):
    assembly_id: str
    source_sha256: str
    source_inventory_sha256: str
    source_member_inventory: tuple
    source_members: tuple[str, ...]
    technology_identity: tuple
    row_program_sha256: str
    row_materialization_sha256: str
    row_source_projection_sha256: str
    row_entry_id: str
    row_stamp_name: str
    context_stamp_names: tuple[str, ...]
    context_foreign_stamp_names: tuple[str, ...]
    context_native_rows_sha256: str
    tap_children: tuple[LayeredTapChild, ...]
    owned_artwork_records: tuple
    well_access: WellComponentAccessBinding
    _content_sha256: str
    _seal: object

    @property
    def owned_stamp_names(self):
        return tuple(name for name, _records in self.owned_artwork_records)

    def payload(self):
        self.well_access.validate()
        return dict(schema='clay.layered-placement-assembly/v1', assembly_id=self.assembly_id,
            source_sha256=self.source_sha256, source_inventory_sha256=self.source_inventory_sha256,
            source_member_inventory=self.source_member_inventory, source_members=self.source_members,
            technology_identity=self.technology_identity, row_program_sha256=self.row_program_sha256,
            row_materialization_sha256=self.row_materialization_sha256,
            row_source_projection_sha256=self.row_source_projection_sha256,
            row_entry_id=self.row_entry_id, row_stamp_name=self.row_stamp_name,
            context_stamp_names=self.context_stamp_names, context_native_rows_sha256=self.context_native_rows_sha256,
            context_foreign_stamp_names=self.context_foreign_stamp_names,
            tap_children=tuple(child.payload() for child in self.tap_children),
            owned_artwork_records=self.owned_artwork_records, well_access_sha256=self.well_access.sha256)


@dataclass(frozen=True, slots=True)
class LayeredAssemblyPose:
    assembly_id: str
    binding_sha256: str
    row_stamp_name: str
    owned_stamp_names: tuple[str, ...]
    source_members: tuple[str, ...]
    translation_nm: tuple[int, int]

    def __post_init__(self):
        _name(self.assembly_id, 'assembly id')
        _name(self.row_stamp_name, 'row stamp name')
        _offset(self.translation_nm)
        for value in (self.owned_stamp_names, self.source_members):
            _tuple(value, 'pose members')
            _require(value == tuple(sorted(set(value))), 'pose members are not canonical')
        _require(self.row_stamp_name in self.owned_stamp_names, 'pose has no owned row stamp')

    def payload(self):
        self.__post_init__()
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True, init=False)
class LayeredPlacementAssemblyReplay(_Captured):
    assemblies: tuple[LayeredAssemblyPose, ...]
    well_access: WellComponentAccessBinding | None
    stamp_artwork_records: tuple
    _content_sha256: str
    _seal: object

    def payload(self):
        return dict(schema='clay.layered-placement-assembly-replay/v1',
            assemblies=tuple(row.payload() for row in self.assemblies),
            well_access_sha256=None if self.well_access is None else self.well_access.sha256,
            stamp_artwork_records=self.stamp_artwork_records)


@dataclass(frozen=True, slots=True, init=False)
class LayeredPlacementAssemblyBlockProjection(_Captured):
    replay_sha256: str
    block_rows: tuple
    stamp_rows: tuple
    ring_rows: tuple
    ring_names: tuple
    refinement_block_ids: tuple[str, ...]
    assembly_block_ids: tuple
    _content_sha256: str
    _seal: object

    @property
    def blocks(self):
        self.validate()
        return self.block_rows

    @property
    def block_stamps(self):
        self.validate()
        return {name: set(stamps) for name, stamps in self.stamp_rows}

    @property
    def block_gring(self):
        self.validate()
        return dict(self.ring_rows)

    @property
    def gring_names(self):
        self.validate()
        return set(self.ring_names)

    def payload(self):
        return dict(schema='clay.layered-placement-assembly-blocks/v1', replay_sha256=self.replay_sha256,
            blocks=self.block_rows, block_stamps=self.stamp_rows, block_gring=self.ring_rows,
            gring_names=self.ring_names, refinement_block_ids=self.refinement_block_ids,
            assembly_block_ids=self.assembly_block_ids)


def _markers(geometry):
    result = {}
    for name, stamp in geometry.items():
        marker = getattr(stamp, 'layered_placement_tap_child', None)
        if marker is not None:
            _require(type(marker) is LayeredTapChild, 'tap requirement marker is foreign')
            marker.__post_init__()
            _require(marker.stamp_name == name, 'tap marker stamp identity differs')
            _require(not getattr(stamp, 'composite_members', ()), 'tap child cannot claim logical source members')
            result[name] = marker
    return result


def _stamp_state(geometry, placement_um):
    from clay.layout.joint_row_stamp import JointRowStampBinding
    _require(type(geometry) is dict and type(placement_um) is dict and set(geometry) <= set(placement_um)
             and len(geometry) <= _MAX_ITEMS, 'current stamp/placement inventory is malformed')
    result = []
    for name, stamp in sorted(geometry.items()):
        joint_checkpoint()()
        _name(name, 'current stamp name')
        binding = getattr(stamp, 'joint_row_stamp_binding', None)
        _require(binding is None or type(binding) is JointRowStampBinding, 'current row stamp binding is foreign')
        if binding is not None:
            binding.__post_init__()
        marker = getattr(stamp, 'layered_placement_tap_child', None)
        _require(marker is None or type(marker) is LayeredTapChild, 'tap marker is foreign')
        result.append((name, tuple(sorted(_stamp_records(stamp, placement_um[name]))),
            tuple(getattr(stamp, 'composite_members', ())),
            None if binding is None else _hash(binding.payload()),
            None if marker is None else marker.payload()))
    return tuple(result)


def _program_identity(program, materialization):
    projection = materialization.source_projection
    _require(projection is not None, 'complete source projection is missing')
    return _hash(program.payload()), materialization.certificate.sha256, projection.sha256


def _context(source_cell, tech, program, materialization, rows, geometry, placement_um):
    from clay.layout.place.row_replay_session import row_materialization_replay_scope
    _tuple(rows, 'current row context')
    _require(all(type(row) is JointWellRow for row in rows), 'body context has a foreign row')
    initial_rows = _row_state(rows)
    initial_stamps = _stamp_state(geometry, placement_um)
    source, technology = _source(source_cell), _technology(tech)[0]
    _require(source[5] == tech.adapter.pdk_id, 'source PDK differs from current technology')
    context = dict(source=source, technology=technology,
        program_identity=_program_identity(program, materialization),
        native_rows_sha256=_hash(initial_rows), row_state=initial_rows, stamp_state=initial_stamps,
        actual_records={row[0]: row[1] for row in initial_stamps},
        rows=rows, source_cell=source_cell, tech=tech, program=program,
        materialization=materialization, geometry=geometry, placement_um=placement_um)
    try:
        # This invocation owns its replay. GP/bridge scopes have already ended;
        # the proof never becomes an operand of a later body or placement call.
        with row_materialization_replay_scope(program=program, materialization=materialization,
                                              tech=tech, source_cell=source_cell):
            _replay_source_context(context)
    except BaseException as error:
        try:
            _unchanged(context)
        except BaseException as ending:
            from clay.flow.control_stop import record_unwind_failure
            record_unwind_failure(error, ending)
        raise
    _unchanged(context)
    return context


def _replay_source_context(context):
    from clay.layout.place.row_replay_session import (
        require_current_row_materialization, current_row_geometry_sha256,
    )
    from clay.layout.place.row_source_projection import require_row_source_projection, replay_row_source_projection
    from clay.layout.joint_placement_terminals import project_joint_placement_face
    from clay.layout.joint_row_stamp import JointRowStampBinding
    from clay.layout.placed_geometry import placed_point_nm
    source_cell, tech = context['source_cell'], context['tech']
    program, materialization, rows = context['program'], context['materialization'], context['rows']
    geometry, placement_um = context['geometry'], context['placement_um']
    check = joint_checkpoint()
    require_current_row_materialization(materialization, program=program, tech_id=tech.pdk_id)
    projection = require_row_source_projection(materialization.source_projection, program=program)
    replayed = replay_row_source_projection(projection, source_cell, tech)
    _require(replayed.to_canonical_bytes() == program.to_canonical_bytes(), 'complete current source program differs')
    by_row = {row.row_id: row for row in rows}
    by_entry = {entry.entry_id: entry for entry in program.entries}
    realizations = {item.entry_id: item for item in materialization.realizations}
    _require(len(by_row) == len(rows) and set(by_row) == {entry.group for entry in program.entries}
             and set(realizations) == set(by_entry), 'complete row/source context denominator differs')
    members = {member.device: member for member in program.members}
    row_deltas = {}
    for entry in program.entries:
        check()
        row = by_row[entry.group]
        _require(entry.group in geometry, 'current source row stamp is missing')
        native_sha = current_row_geometry_sha256(row.geometry, entry.consumed_devices)
        _require(native_sha == realizations[entry.entry_id].geometry_sha256,
                 'row context differs from the complete native realization')
        stamp = geometry[entry.group]
        binding = getattr(stamp, 'joint_row_stamp_binding', None)
        _require(type(binding) is JointRowStampBinding, 'row stamp lost its required native binding')
        position = placement_um[entry.group]
        origin = placed_point_nm(0, 0, position)
        delta = tuple(origin[i]-binding.native_origin_nm[i] for i in range(2))
        project_joint_placement_face(native_geometry=row.geometry, stamp=stamp, position=position,
            members=entry.consumed_devices, delta=delta, tech=tech, native_row_sha256=native_sha,
            source_members=members)
        row_deltas[row.row_id] = delta
    context.update(by_row=by_row, by_entry=by_entry, deltas=row_deltas)


def _unchanged(context):
    joint_checkpoint()()
    _require(_source(context['source_cell'])[:3] == context['source'][:3]
             and _technology(context['tech'])[0] == context['technology'], 'source/PDK changed during native work')
    _require(_program_identity(context['program'], context['materialization']) == context['program_identity']
             and _row_state(context['rows']) == context['row_state'], 'row program or native binding changed during work')
    _require(_stamp_state(context['geometry'], context['placement_um']) == context['stamp_state'],
             'current stamp artwork or child binding changed during native work')


def capture_layered_placement_assembly(*, assembly_id, source_cell, tech, row_program, row_materialization,
        row_entry_id, row_stamp_name, rows, tap_children, well_access, geometry, placement_um):
    context = _context(source_cell, tech, row_program, row_materialization, rows, geometry, placement_um)
    _require(type(well_access) is WellComponentAccessBinding, 'well access binding is foreign')
    well_access.validate()
    markers = _markers(geometry)
    entries = _tap_entries(context, markers)
    owners = tuple(sorted({member for entry_id in set(entries.values())
                           for member in context['by_entry'][entry_id].consumed_devices}))
    current_rows, foreign = _current_body_context(context, markers)
    replay_well_component_access(well_access, source_cell, tech=tech, rows=current_rows, pmos_owners=owners,
        additions=tuple(child.construction for _, child in sorted(markers.items())),
        actual_polygons=_current_scene(context), foreign_artwork=foreign)
    result = _capture_owned(context, assembly_id, row_entry_id, row_stamp_name, tap_children, well_access, entries)
    _unchanged(context)
    return result


def _capture_owned(context, assembly_id, row_entry_id, row_stamp_name, tap_children, well_access, entries):
    """Only called after the complete current source/body scene was verified."""
    _name(assembly_id, 'assembly id')
    _tuple(tap_children, 'tap children')
    _require(all(type(child) is LayeredTapChild for child in tap_children), 'tap child type differs')
    _require(type(well_access) is WellComponentAccessBinding, 'well access binding is foreign')
    well_access.validate()
    entry = context['by_entry'].get(row_entry_id)
    _require(entry is not None and entry.group == row_stamp_name, 'selected row entry/stamp identity differs')
    row = context['by_row'][row_stamp_name]
    _require(row.geometry.kind == 'pmos', 'assembly capability requires one PMOS row')
    members = tuple(sorted(entry.consumed_devices))
    _require(set(members) <= set(well_access.covered_members), 'well/source row member denominator differs')
    markers = _markers(context['geometry'])
    child_names = tuple(child.stamp_name for child in tap_children)
    _require(len(set(child_names)) == len(tap_children)
             and set(child_names) == {name for name, entry_id in entries.items() if entry_id == row_entry_id},
             'tap child inventory is incomplete or duplicated')
    _require(row_stamp_name not in child_names, 'row and tap child identities overlap')
    local_records = [(row_stamp_name, _records(tuple(row.geometry.polygons)))]
    for child in tap_children:
        child.__post_init__()
        _require(child == markers[child.stamp_name], 'tap child differs from its stamp requirement')
        _require(child.local_offset_nm == tuple(-v for v in context['deltas'][row_stamp_name]),
                 'tap construction offset differs from current row frame')
        local_records.append((child.stamp_name, _move(child.construction.polygon_records, child.local_offset_nm)))
    common_delta = context['deltas'][row_stamp_name]
    for name, records in local_records:
        _same(context['actual_records'][name], _move(records, common_delta),
              'actual child artwork or local offset differs from native construction')
    source = context['source']
    result = _capture(LayeredPlacementAssemblyBinding, assembly_id=assembly_id, source_sha256=source[0],
        source_inventory_sha256=source[1], source_member_inventory=source[2], source_members=members,
        technology_identity=context['technology'], row_program_sha256=context['program_identity'][0],
        row_materialization_sha256=context['program_identity'][1],
        row_source_projection_sha256=context['program_identity'][2], row_entry_id=row_entry_id,
        row_stamp_name=row_stamp_name, context_stamp_names=tuple(sorted(context['by_row'])),
        context_foreign_stamp_names=tuple(sorted(set(context['actual_records'])-set(context['by_row'])-set(markers))),
        context_native_rows_sha256=context['native_rows_sha256'], tap_children=tap_children,
        owned_artwork_records=tuple(sorted(local_records)), well_access=well_access)
    return result


def _tap_entries(context, markers):
    by_occurrence = {occurrence.occurrence_id: entry.entry_id for entry in context['program'].entries
        for occurrence in context['by_row'][entry.group].geometry.joint_row_binding.occurrences}
    result, request_ids = {}, set()
    for name, child in sorted(markers.items()):
        request = child.construction.request
        _require(request.tap_id not in request_ids, 'duplicate tap request in complete inventory')
        request_ids.add(request.tap_id)
        entries = {by_occurrence.get(occurrence) for occurrence in request.occurrence_ids}
        _require(len(entries) == 1 and None not in entries, 'tap request crosses rows or names a missing occurrence')
        entry_id, = entries
        entry = context['by_entry'][entry_id]
        _require(context['by_row'][entry.group].geometry.kind == 'pmos', 'tap request requires a PMOS source row')
        result[name] = entry_id
    return result


def _current_body_context(context, markers):
    rows = tuple(_translated_row(row, context['deltas'][row.row_id]) for row in context['rows'])
    foreign = tuple(WellForeignArtwork(name, context['actual_records'][name])
        for name in sorted(set(context['actual_records'])-set(context['by_row'])-set(markers)))
    return rows, foreign


def _current_scene(context):
    return _from_records(tuple(record for _, records in sorted(context['actual_records'].items()) for record in records))


def _translated_row(row, delta):
    """Transport full native evidence without changing source or tap authoring."""
    from clay.layout.joint_row_transform import transform_joint_row_geometry
    from clay.layout.occurrence_channels import transform_row_channel_inventory
    original = row.geometry
    dx, dy = _offset(delta)
    polygons = _from_records(_move(_records(tuple(original.polygons)), delta))
    fingers = tuple(replace(finger, x0=finger.x0+dx, x1=finger.x1+dx, cy=finger.cy+dy,
        gate_bbox_nm=(finger.gate_bbox_nm[0]+dx, finger.gate_bbox_nm[1]+dy,
                      finger.gate_bbox_nm[2]+dx, finger.gate_bbox_nm[3]+dy)) for finger in original.fingers)
    matrix = ((1, 0), (0, 1))
    channels = transform_row_channel_inventory(geometry=original,
        consumed_devices=tuple(sorted({o.owner for o in original.joint_row_binding.occurrences})),
        matrix=matrix, translation_nm=delta, row_polygons=polygons)
    target = SimpleNamespace(kind=original.kind, polygons=polygons, fingers=fingers, channel_inventory=channels)
    transform_joint_row_geometry(original, target, matrix=matrix, translation_nm=delta)
    return JointWellRow(row.row_id, target)


@dataclass(frozen=True, slots=True)
class LayeredPmosAssemblyMaterialization:
    """Actual mutable native views plus their separately immutable current proof."""
    assemblies: tuple[LayeredPlacementAssemblyBinding, ...]
    geometry: dict
    placement_um: dict
    replay: LayeredPlacementAssemblyReplay


def _materialized_rows(program, materialization):
    """Project typed realizations; _context owns the complete native replay."""
    from clay.layout.place.row_program import (
        RowProgram, RowProgramEntry, RowProgramMaterialization, RowMacroRealization,
    )
    _require(type(program) is RowProgram and type(materialization) is RowProgramMaterialization,
             'current row program/materialization type differs')
    _tuple(program.entries, 'current row entries')
    _tuple(materialization.realizations, 'current row realizations')
    _require(all(type(entry) is RowProgramEntry for entry in program.entries)
             and all(type(item) is RowMacroRealization for item in materialization.realizations),
             'current row entry/realization type differs')
    entries = {entry.entry_id: entry for entry in program.entries}
    identities = tuple(item.entry_id for item in materialization.realizations)
    _require(len(entries) == len(program.entries) and len(set(identities)) == len(identities)
             and set(identities) == set(entries), 'complete row realization denominator differs')
    return tuple(sorted((JointWellRow(entries[item.entry_id].group, item.geometry)
                         for item in materialization.realizations), key=lambda row: row.row_id))


def _phase(name):
    from contextlib import nullcontext
    from clay.flow.execution_session import current_session
    session = current_session()
    return nullcontext() if session is None else session.phase('layered_assembly.'+name)


def materialize_layered_pmos_assemblies(*, source_cell, tech, row_program, row_materialization,
                                      geometry, placement_um, tap_requests_by_entry):
    """Build actual PDK taps from source occurrence IDs at current row poses.

    This factory has no coordinate/expected-body authoring input. Tap requests
    choose the protected source occurrence set; the current PDK builder owns
    all geometry. Existing source rows stay real separate materialized stamps.
    """
    from copy import copy, deepcopy
    from clay.layout.extract.library_device import CellGeometry
    _tuple(tap_requests_by_entry, 'tap requests by entry')
    with _phase('source_context'):
        rows = _materialized_rows(row_program, row_materialization)
        context = _context(source_cell, tech, row_program, row_materialization, rows, geometry, placement_um)
    _require(not _markers(geometry), 'first factory input already contains required assembly tap children')
    requests, entry_requests = {}, {}
    for item in tap_requests_by_entry:
        _require(type(item) is tuple and len(item) == 2, 'tap entry request must be an exact pair')
        entry_id, requested = item
        _name(entry_id, 'tap row entry')
        entry = context['by_entry'].get(entry_id)
        _require(entry is not None and entry_id not in entry_requests, 'foreign or duplicate tap source row entry')
        row = context['by_row'][entry.group]
        _require(row.geometry.kind == 'pmos', 'tap source entry is not a PMOS row')
        _tuple(requested, 'entry tap requests')
        occurrence_ids = {occ.occurrence_id for occ in row.geometry.joint_row_binding.occurrences}
        for request in requested:
            _require(type(request) is PdkWellTapRequest, 'tap request must be source occurrence IDs')
            request.__post_init__()
            _require(set(request.occurrence_ids) <= occurrence_ids, 'tap request names a foreign row occurrence')
            _require(request.tap_id not in requests, 'duplicate tap request id')
            requests[request.tap_id] = request
        entry_requests[entry_id] = tuple(request.tap_id for request in requested)
    with _phase('current_body_context'):
        current_rows, foreign = _current_body_context(context, {})
    owners = tuple(sorted({member for entry_id in entry_requests
                           for member in context['by_entry'][entry_id].consumed_devices}))
    with _phase('pdk_taps'):
        additions = materialize_pmos_component_taps(source_cell, tech=tech, rows=current_rows,
            pmos_owners=owners, requests=tuple(requests[key] for key in sorted(requests)), foreign_artwork=foreign)
    # Copy native views without cloning private captured seals or mutating any
    # caller polygon list if a subsequent PDK/context check rejects.
    result_geometry = {name: copy(stamp) for name, stamp in geometry.items()}
    for name, stamp in result_geometry.items():
        stamp.polys_by_layer = deepcopy(geometry[name].polys_by_layer)
    result_positions = dict(placement_um)
    by_request = {request_id: entry_id for entry_id, request_ids in entry_requests.items() for request_id in request_ids}
    for addition in additions:
        entry = context['by_entry'][by_request[addition.request.tap_id]]
        name = 'TAP:'+addition.request.tap_id
        _require(name not in result_geometry, 'tap stamp identity collides with existing actual geometry')
        child = LayeredTapChild(name, addition, tuple(-value for value in context['deltas'][entry.group]))
        layers = {}
        for layer, datatype, points in addition.polygon_records:
            layers.setdefault((layer, datatype), []).append([(x/1000, y/1000) for x, y in points])
        stamp = CellGeometry(name, 'physical_accessory', (0.0, 0.0), layers)
        stamp.composite_members = ()
        stamp.layered_placement_tap_child = child
        result_geometry[name], result_positions[name] = stamp, (0.0, 0.0)
    _unchanged(context)
    new_context = dict(context, geometry=result_geometry, placement_um=result_positions)
    new_context['stamp_state'] = _stamp_state(result_geometry, result_positions)
    new_context['actual_records'] = {item[0]: item[1] for item in new_context['stamp_state']}
    markers = _markers(result_geometry)
    entries = _tap_entries(new_context, markers)
    with _phase('well_access'):
        well = capture_well_component_access(source_cell, tech=tech, rows=current_rows, pmos_owners=owners,
            additions=additions, actual_polygons=_current_scene(new_context), foreign_artwork=foreign)
    bindings = tuple(_capture_owned(new_context, 'layered:'+entry_id, entry_id,
        context['by_entry'][entry_id].group,
        tuple(child for name, child in sorted(markers.items()) if entries[name] == entry_id), well, entries)
        for entry_id in sorted(entry_requests))
    poses = tuple(LayeredAssemblyPose(binding.assembly_id, binding.sha256, binding.row_stamp_name,
        binding.owned_stamp_names, binding.source_members, context['deltas'][binding.row_stamp_name]) for binding in bindings)
    observed = _capture(LayeredPlacementAssemblyReplay, assemblies=poses, well_access=well,
        stamp_artwork_records=tuple(sorted(new_context['actual_records'].items())))
    _unchanged(new_context)
    _unchanged(context)
    return LayeredPmosAssemblyMaterialization(bindings, result_geometry, result_positions, observed)


def replay_layered_placement_assemblies(bindings, *, source_cell, tech, row_program, row_materialization,
                                      rows, geometry, placement_um):
    joint_checkpoint()()
    _tuple(bindings, 'binding inventory', nonempty=False)
    markers = _markers(geometry)
    if not bindings:
        _require(not markers, 'required assembly binding inventory was removed')
        return _capture(LayeredPlacementAssemblyReplay, assemblies=(), well_access=None, stamp_artwork_records=())
    _require(all(type(binding) is LayeredPlacementAssemblyBinding for binding in bindings),
             'inventory requires exact captured assembly bindings')
    bindings = tuple(sorted(bindings, key=lambda b: b.assembly_id))
    for field in ('assembly_id', 'row_stamp_name', 'row_entry_id'):
        _require(len({getattr(binding, field) for binding in bindings}) == len(bindings),
                 'duplicate assembly or source row identity')
    context = _context(source_cell, tech, row_program, row_materialization, rows, geometry, placement_um)
    children = tuple(sorted((child for binding in bindings for child in binding.tap_children), key=lambda c:c.stamp_name))
    _require(len({child.stamp_name for child in children}) == len(children)
        and set(markers) == {child.stamp_name for child in children}
        and all(markers[child.stamp_name] == child for child in children), 'required tap child binding inventory differs')
    entries = _tap_entries(context, markers)
    current_rows, foreign = _current_body_context(context, markers)
    foreign_names = {item.stamp_id for item in foreign}
    poses, members_seen = [], set()
    for binding in bindings:
        binding.validate()
        _require((binding.source_sha256, binding.source_inventory_sha256, binding.source_member_inventory) == context['source'][:3]
                 and binding.technology_identity == context['technology'], 'current source/PDK differs from captured assembly')
        _require((binding.row_program_sha256, binding.row_materialization_sha256, binding.row_source_projection_sha256)
                 == context['program_identity'] and binding.context_native_rows_sha256 == context['native_rows_sha256']
                 and binding.context_stamp_names == tuple(sorted(context['by_row'])), 'current source row context differs')
        _require(set(binding.context_foreign_stamp_names) <= foreign_names, 'captured foreign stamp inventory is missing')
        _require(not members_seen.intersection(binding.source_members), 'assembly source members are repeated')
        members_seen.update(binding.source_members)
        _require(all(entries[child.stamp_name] == binding.row_entry_id for child in binding.tap_children),
                 'tap ownership differs from captured source row')
        delta = context['deltas'][binding.row_stamp_name]
        for name, records in binding.owned_artwork_records:
            _require(name in context['actual_records'], 'current assembly child stamp is missing')
            _same(context['actual_records'][name], _move(records, delta),
                  'assembly child artwork/frame differs from common integer translation')
        poses.append(LayeredAssemblyPose(binding.assembly_id, binding.sha256, binding.row_stamp_name,
            binding.owned_stamp_names, binding.source_members, delta))
    owners = tuple(sorted(members_seen))
    additions = materialize_pmos_component_taps(source_cell, tech=tech, rows=current_rows,
        pmos_owners=owners, requests=tuple(child.construction.request for child in children), foreign_artwork=foreign)
    for child, addition in zip(children, additions):
        _same(context['actual_records'][child.stamp_name], addition.polygon_records,
              'actual tap material differs from current PDK construction and row context')
    # Independently moving foreign context changes this proof. It must never be
    # inherited from the local capture, even when every owned mask stayed rigid.
    well = capture_well_component_access(source_cell, tech=tech, rows=current_rows,
        pmos_owners=owners, additions=additions, actual_polygons=_current_scene(context), foreign_artwork=foreign)
    result = _capture(LayeredPlacementAssemblyReplay, assemblies=tuple(poses), well_access=well,
        stamp_artwork_records=tuple(sorted(context['actual_records'].items())))
    _unchanged(context)
    return result


def project_layered_placement_assemblies(replayed, *, blocks, block_stamps, block_gring, gring_names):
    """Fold only typed physical accessories into their real row ownership block."""
    _require(type(replayed) is LayeredPlacementAssemblyReplay, 'block projection requires current typed replay')
    replayed.validate()
    _tuple(blocks, 'materialized blocks')
    _require(type(block_stamps) is dict and type(block_gring) is dict and type(gring_names) is set,
             'legacy block projection inputs have foreign types')
    boxes = dict(blocks)
    _require(len(boxes) == len(blocks) and set(boxes) == set(block_stamps)
             and set(block_gring) <= set(boxes), 'materialized block/stamp denominator differs')
    stamps = {}
    for name, box in blocks:
        _name(name, 'materialized block id')
        _require(type(box) is tuple and len(box) == 4 and all(type(v) is int for v in box)
                 and box[0] < box[2] and box[1] < box[3], 'materialized block bbox is malformed')
        value = block_stamps[name]
        _require(type(value) in (set, tuple, frozenset) and value and len(value) <= _MAX_ITEMS,
                 'materialized stamp inventory is malformed')
        stamps[name] = set(value)
    actual = dict(replayed.stamp_artwork_records)
    from collections import Counter
    partition = Counter(stamp for values in stamps.values() for stamp in values)
    partition.update(block_gring.values())
    _require(set(gring_names) == set(block_gring.values())
        and partition == Counter({name: 1 for name in actual}),
        'complete actual stamp partition/guard ownership denominator differs')
    for name in boxes:
        physical = (*stamps[name], *((block_gring[name],) if name in block_gring else ()))
        _require(boxes[name] == _bbox(tuple(record for stamp in physical for record in actual[stamp])),
                 'materialized block bbox differs from complete actual stamp inventory')
    refined, assemblies = [], []
    for pose in replayed.assemblies:
        joint_checkpoint()()
        owned = set(pose.owned_stamp_names)
        hits = {stamp: tuple(name for name, members in stamps.items() if stamp in members) for stamp in owned}
        _require(all(len(values) == 1 for values in hits.values()), 'assembly child has missing or repeated physical block ownership')
        owner, = hits[pose.row_stamp_name]
        child_blocks = {values[0] for values in hits.values()}
        _require(not child_blocks.intersection(block_gring) and not owned.intersection(gring_names),
                 'assembly child is absorbed into an incompatible guard island')
        _require(set().union(*(stamps[name] for name in child_blocks)) == owned,
                 'assembly cannot absorb an undeclared foreign stamp')
        for child in child_blocks:
            _require(all(stamp in actual for stamp in stamps[child]), 'current replay lacks a child artwork inventory')
            expected = _bbox(tuple(record for stamp in stamps[child] for record in actual[stamp]))
            _require(boxes[child] == expected, 'child block bbox differs from current actual artwork')
        boxes[owner] = _bbox(tuple(record for stamp in pose.owned_stamp_names for record in actual[stamp]))
        stamps[owner] = owned
        for child in child_blocks - {owner}:
            del boxes[child]
            del stamps[child]
        refined.append(owner)
        assemblies.append((pose.assembly_id, owner))
    return _capture(LayeredPlacementAssemblyBlockProjection, replay_sha256=replayed.sha256,
        block_rows=tuple(sorted(boxes.items())), stamp_rows=tuple((name, tuple(sorted(values))) for name, values in sorted(stamps.items())),
        ring_rows=tuple(sorted(block_gring.items())), ring_names=tuple(sorted(gring_names)),
        refinement_block_ids=tuple(sorted(refined)), assembly_block_ids=tuple(sorted(assemblies)))


def restore_layered_pmos_assemblies(bindings, *, source_cell, tech, row_program,
                                   row_materialization, geometry, placement_um):
    """Rebuild required tap stamps after a new native source-placement bridge.

    Captured local masks propose independent physical copies in the current row
    frames. They acquire no current authority until replay regenerates every tap
    through the current PDK builder, compares the full multiset, and extracts the
    complete current body scene. No prior well-access result is reused.
    """
    from copy import copy, deepcopy
    from clay.layout.extract.library_device import CellGeometry
    from clay.layout.joint_row_stamp import JointRowStampBinding
    from clay.layout.placed_geometry import placed_point_nm
    _tuple(bindings, 'binding inventory', nonempty=False)
    _require(all(type(binding) is LayeredPlacementAssemblyBinding for binding in bindings),
             'restore inventory requires exact captured assembly bindings')
    for binding in bindings:
        binding.validate()
    initial = _stamp_state(geometry, placement_um)
    result_geometry = {name: copy(stamp) for name, stamp in geometry.items()}
    for name, stamp in result_geometry.items():
        stamp.polys_by_layer = deepcopy(geometry[name].polys_by_layer)
    result_positions = dict(placement_um)
    existing = _markers(geometry)
    if not existing:
        for binding in bindings:
            joint_checkpoint()()
            row = geometry.get(binding.row_stamp_name)
            native = getattr(row, 'joint_row_stamp_binding', None)
            _require(type(native) is JointRowStampBinding, 'restore row lacks current native stamp binding')
            native.__post_init__()
            origin = placed_point_nm(0, 0, placement_um[binding.row_stamp_name])
            delta = tuple(origin[i]-native.native_origin_nm[i] for i in range(2))
            records = dict(binding.owned_artwork_records)
            for child in binding.tap_children:
                _require(child.stamp_name not in result_geometry, 'restored tap stamp collides with current geometry')
                layers = {}
                for layer, datatype, points in _move(records[child.stamp_name], delta):
                    layers.setdefault((layer, datatype), []).append([(x/1000, y/1000) for x, y in points])
                stamp = CellGeometry(child.stamp_name, 'physical_accessory', (0.0, 0.0), layers)
                stamp.composite_members = ()
                stamp.layered_placement_tap_child = child
                result_geometry[child.stamp_name], result_positions[child.stamp_name] = stamp, (0.0, 0.0)
    rows = _materialized_rows(row_program, row_materialization) if bindings or existing else ()
    observed = replay_layered_placement_assemblies(bindings, source_cell=source_cell, tech=tech,
        row_program=row_program, row_materialization=row_materialization, rows=rows,
        geometry=result_geometry, placement_um=result_positions)
    _require(_stamp_state(geometry, placement_um) == initial, 'original bridge artwork changed during tap restoration')
    return LayeredPmosAssemblyMaterialization(tuple(sorted(bindings, key=lambda b: b.assembly_id)),
        result_geometry, result_positions, observed)


def project_current_layered_placement(bindings, *, source_cell, tech, row_program,
        row_materialization, geometry, placement_um, blocks, block_stamps, block_gring, gring_names):
    """Recheck one complete current scene immediately after canonical extraction."""
    _tuple(bindings, 'binding inventory', nonempty=False)
    markers = _markers(geometry)
    if not bindings:
        _require(not markers, 'required assembly binding inventory was removed from actual tap geometry')
        return None, None
    rows = _materialized_rows(row_program, row_materialization)
    observed = replay_layered_placement_assemblies(bindings, source_cell=source_cell, tech=tech,
        row_program=row_program, row_materialization=row_materialization, rows=rows,
        geometry=geometry, placement_um=placement_um)
    projected = project_layered_placement_assemblies(observed, blocks=tuple(blocks),
        block_stamps=block_stamps, block_gring=block_gring, gring_names=gring_names)
    return observed, projected


def require_layered_placement_certificate(certificate, *, source_cell, tech,
        row_program, row_materialization, intent, geometry, placement_um):
    """Join sealed final data to fresh source/PDK/full actual body-context replay."""
    from clay.layout.place.placement_certificates import FinalPlacementCertificate
    from clay.layout.place.layered_assembly_certificate import layered_assembly_certificate_payload
    from clay.layout.compact_blocks import extract_blocks
    _require(type(certificate) is FinalPlacementCertificate, 'final certificate type is foreign')
    layered_assembly_certificate_payload(certificate)
    # The actual markers impose an obligation independently of certificate
    # scope fields, including simultaneous deletion of all three fields.
    markers = _markers(geometry)
    bindings = certificate.layered_placement_assemblies
    _require(bindings or not markers, 'final actual tap markers lack required assembly proof inventory')
    if not bindings:
        return None
    blocks, stamps, rings, ring_names = extract_blocks(geometry, placement_um, intent)
    observed, projection = project_current_layered_placement(bindings,
        source_cell=source_cell, tech=tech, row_program=row_program, row_materialization=row_materialization,
        geometry=geometry, placement_um=placement_um, blocks=blocks,
        block_stamps=stamps, block_gring=rings, gring_names=ring_names)
    _require(observed.sha256 == certificate.layered_placement_replay.sha256
             and projection.sha256 == certificate.layered_placement_projection.sha256,
             'final current replay/projection differs from certificate actual artwork')
    return observed
