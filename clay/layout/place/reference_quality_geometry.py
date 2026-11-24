"""Project freshly observed materializer geometry into rigid solver-local frames.

This module accepts current source geometry only. It has no GDS, saved candidate,
quality target, seed or production-authority input.
"""
from collections.abc import Mapping
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
import json
import math
import os
from threading import current_thread


_OBSERVER = ContextVar('live_rigid_quality_geometry', default=None)


class LiveQualityGeometryFailure(BaseException):
    """Observation failure must not become a native heuristic fallback."""


def _invocation():
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), current_thread(), task


class _Observation:
    def __init__(self, owner_ids, source_population=None):
        self.owner_ids = owner_ids
        self.source_population = source_population
        self.invocation = _invocation()
        self.closed = False
        self._latest = None

    def check(self):
        _require(not self.closed, 'live geometry observation is closed')
        _require(self.invocation == _invocation(), 'live geometry observation invocation differs')

    @property
    def latest(self):
        self.check()
        _require(self._latest is not None, 'current native frame lacks live geometry observation')
        return self._latest


@contextmanager
def observe_live_quality_geometry(owner_ids):
    """Retain values observed during one current invocation; grant no authority."""
    _require(_OBSERVER.get() is None, 'nested live geometry observations are unsupported')
    _require(type(owner_ids) is tuple and owner_ids
             and all(type(x) is str and x for x in owner_ids)
             and owner_ids == tuple(sorted(set(owner_ids))), 'live geometry owner inventory differs')
    with _observation_scope(owner_ids=owner_ids) as observation:
        yield observation


@contextmanager
def observe_live_quality_geometry_for_source(source_population):
    """Observe actual master partitions for one complete declared population.

    This selects current measurements without guessing names of future physical
    owners. The caller still authenticates the source and native consumption.
    Each published frame must retain every logical owner and occurrence count.
    """
    _require(type(source_population) is tuple and bool(source_population)
        and all(type(row) is tuple and len(row) == 2 and type(row[0]) is str
                and bool(row[0]) and row[0] == row[0].strip()
                and type(row[1]) is int and row[1] > 0 for row in source_population)
        and tuple(name for name, _ in source_population)
            == tuple(sorted({name for name, _ in source_population})),
        'live geometry source population must contain canonical positive counts')
    with _observation_scope(source_population=source_population) as observation:
        yield observation


@contextmanager
def _observation_scope(*, owner_ids=None, source_population=None):
    _require(_OBSERVER.get() is None, 'nested live geometry observations are unsupported')
    from clay.layout.place.current_owner_stage import _OWNER_OBSERVATION
    _require(_OWNER_OBSERVATION.get() is None, 'candidate owner and quality observations cannot nest')
    observation = _Observation(owner_ids, source_population)
    token = _OBSERVER.set(observation)
    try:
        yield observation
    finally:
        observation.closed = True
        observation._latest = None
        _OBSERVER.reset(token)


def wants_live_quality_geometry(owner_ids):
    observation = _OBSERVER.get()
    if observation is None:
        return False
    observation.check()
    return observation.source_population is not None or tuple(sorted(owner_ids)) == observation.owner_ids


def publish_live_quality_geometry(geometry):
    from clay.layout.reference_quality_contract import RigidQualityGeometryIR
    observation = _OBSERVER.get()
    _require(observation is not None, 'live geometry observation is not active')
    observation.check()
    observation._latest = None
    _require(type(geometry) is RigidQualityGeometryIR, 'live geometry IR type differs')
    geometry.__post_init__()
    if observation.source_population is not None:
        _require(Counter(row.logical_id for row in geometry.occurrences)
                 == dict(observation.source_population), 'live geometry source population differs')
    else:
        _require(tuple(owner for owner, _, _ in geometry.owners) == observation.owner_ids,
                 'live geometry observed owner inventory differs')
    observation._latest = geometry


def capture_live_quality_frame(*, source_blocks, solver_blocks, source_block_member_devices,
        solver_block_member_devices, member_offsets, geometry, placement_um,
        logical_kinds, source_device_bboxes):
    if not wants_live_quality_geometry(tuple(name for name, _ in solver_blocks)):
        return None
    try:
        logicals = set(logical_kinds)
        from clay.layout.owned_artwork import quality_source_artwork_bboxes_nm
        source_artwork_bboxes = quality_source_artwork_bboxes_nm(
            geometry=geometry, placement_um=placement_um, source_devices=tuple(sorted(logicals)))
        source_partition = {owner: tuple(sorted(set(members) & logicals))
                            for owner, members in source_block_member_devices.items()}
        # Native bundling retains source keys in this map. Only actual solver
        # blocks define the destination partition; the projector checks cover.
        solver_partition = {owner: tuple(sorted(set(solver_block_member_devices[owner]) & logicals))
                            for owner, _ in solver_blocks}
        occurrences = collect_live_source_occurrences(geometry=geometry, placement_um=placement_um,
            logical_kinds=logical_kinds, source_device_bboxes=source_device_bboxes)
        projected = project_rigid_quality_geometry(source_blocks=source_blocks, solver_blocks=solver_blocks,
            source_block_logicals=source_partition, solver_block_logicals=solver_partition,
            member_offsets=member_offsets, source_occurrences=occurrences,
            source_device_bboxes=source_artwork_bboxes)
        publish_live_quality_geometry(projected)
    except Exception as exc:
        raise LiveQualityGeometryFailure('current geometry observation failed: ' + str(exc)) from exc


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _boxes(rows, label):
    _require(type(rows) is tuple and bool(rows), label + ' inventory must be a nonempty tuple')
    result = {}
    for row in rows:
        _require(type(row) is tuple and len(row) == 2, label + ' inventory row differs')
        name, box = row
        _require(type(name) is str and name and name == name.strip() and name not in result,
                 label + ' inventory name differs')
        _require(type(box) is tuple and len(box) == 4 and all(type(x) is int for x in box)
                 and box[2] > box[0] and box[3] > box[1], label + ' integer bbox differs')
        result[name] = box
    return result


def _partition(rows, names, label):
    _require(type(rows) is dict and set(rows) == names, label + ' partition inventory differs')
    inverse = {}
    for owner, logicals in sorted(rows.items()):
        _require(type(logicals) is tuple and logicals and logicals == tuple(sorted(set(logicals)))
                 and all(type(x) is str and x and x == x.strip() for x in logicals),
                 label + ' partition row differs')
        for logical in logicals:
            _require(logical not in inverse, label + ' partition repeats a logical owner')
            inverse[logical] = owner
    return inverse


def collect_live_source_occurrences(*, geometry, placement_um, logical_kinds, source_device_bboxes):
    """Read actual MOS replica and passive unit metadata in its native frame.

    Resolved master replicas take precedence over internal finger expansion.
    Passive local-nm metadata is placed using the actual source artwork bbox.
    """
    _require(isinstance(geometry, Mapping) and isinstance(placement_um, Mapping)
             and type(logical_kinds) is dict and logical_kinds, 'current geometry inventory differs')
    _require(type(source_device_bboxes) is dict and set(source_device_bboxes) == set(logical_kinds),
             'current source bbox inventory differs')
    allowed = {'R0', 'R90', 'R180', 'R270', 'MR0', 'MR90', 'MR180', 'MR270'}
    mos = {logical for logical, kind in logical_kinds.items() if kind in {'nmos', 'pmos'}}
    passive = {logical for logical, kind in logical_kinds.items()
               if kind in {'capacitor', 'cap', 'mimcap', 'resistor', 'res'}}
    bjts = {logical for logical, kind in logical_kinds.items() if kind in {'pnp', 'npn'}}
    _require(mos | passive | bjts == set(logical_kinds), 'unsupported current logical kind')
    for box in source_device_bboxes.values():
        _require(type(box) is tuple and len(box) == 4 and all(type(x) is int for x in box)
                 and box[0] < box[2] and box[1] < box[3], 'current source integer bbox differs')
    result = {}

    def record(logical, rows):
        _require(logical in logical_kinds and logical not in result, 'current occurrence ownership is ambiguous')
        _require(type(rows) is tuple and rows, 'current occurrence metadata is empty')
        box = source_device_bboxes[logical]
        for row in rows:
            _require(type(row) is tuple and len(row) == 3 and type(row[0]) is int
                     and type(row[1]) is int and type(row[2]) is str and row[2] in allowed,
                     'current occurrence point/orientation differs')
            _require(box[0] <= row[0] <= box[2] and box[1] <= row[1] <= box[3],
                     'current occurrence point escapes source artwork frame')
        result[logical] = tuple(sorted(rows))

    def place_mos(logical, stamp_name, rows):
        _require(logical in mos, 'current MOS occurrence metadata has a foreign logical member')
        base = placement_um.get(stamp_name)
        _require(type(base) is tuple and len(base) == 2
                 and all(type(x) in (int, float) and math.isfinite(x) for x in base),
                 'current MOS occurrence placement metadata differs')
        _require(type(rows) in (tuple, list) and rows, 'current MOS occurrence rows differ')
        placed = []
        for row in rows:
            _require(type(row) in (tuple, list) and len(row) == 3
                     and all(type(x) in (int, float) and math.isfinite(x) for x in row[:2])
                     and type(row[2]) is str and row[2] in allowed,
                     'current MOS occurrence point/orientation metadata differs')
            placed.append((int(round((float(base[0]) + float(row[0])) * 1000.0)),
                           int(round((float(base[1]) + float(row[1])) * 1000.0)), row[2]))
        record(logical, tuple(placed))

    def placed_nm_rows(logical, stamp, centers, orientations, local_box):
        box = source_device_bboxes[logical]
        _require(type(local_box) is tuple and len(local_box) == 4
                 and all(type(x) is int for x in local_box)
                 and (box[2] - box[0], box[3] - box[1]) ==
                     (local_box[2] - local_box[0], local_box[3] - local_box[1]),
                 'current local occurrence frame differs from source artwork')
        _require(type(centers) is tuple and centers and type(orientations) is tuple
                 and len(orientations) == len(centers), 'current unit occurrence inventory differs')
        _require(all(type(p) is tuple and len(p) == 2 and all(type(x) is int for x in p)
                     for p in centers), 'current unit occurrence point requires integer nm')
        record(logical, tuple((p[0] + box[0] - local_box[0], p[1] + box[1] - local_box[1], orient)
                              for p, orient in zip(centers, orientations)))

    for stamp_name, stamp in sorted(geometry.items()):
        if getattr(stamp, 'joint_row_stamp_binding', None) is not None:
            from clay.layout.joint_occurrence_center_basis import require_joint_stamp_center_basis
            require_joint_stamp_center_basis(stamp)
        by_member = getattr(stamp, 'physical_occurrences_by_member', None)
        if by_member is not None:
            _require(type(by_member) is tuple and by_member, 'current member occurrence metadata differs')
            for logical, rows in by_member:
                place_mos(logical, stamp_name, rows)
            continue
        members = getattr(stamp, 'composite_members', ())
        _require(type(members) is tuple and all(type(x) is str for x in members),
                 'current composite occurrence membership differs')
        targets = mos.intersection(members or (stamp_name,))
        _require(len(targets) <= 1, 'current occurrence stamp has ambiguous logical members')
        if not targets:
            continue
        logical = next(iter(targets))
        rows = getattr(stamp, 'resolved_master_occurrences', None)
        fixed = getattr(stamp, 'fixed_pcell', None)
        structure = getattr(stamp, 'discrete_array', None)
        if not rows and type(fixed) is dict and 'unit_occurrences' in fixed and structure is not None:
            centers = getattr(structure, 'replica_offsets_nm', None)
            count = fixed['unit_occurrences']
            _require(type(count) is int and type(centers) is tuple and len(centers) == count,
                     'current resolved-master occurrence denominator differs')
            _require(getattr(stamp, 'orient', None) == 'R0',
                     'current resolved-master orientation needs explicit occurrence metadata')
            placed_nm_rows(logical, stamp, centers, ('R0',) * count, tuple(stamp.extent()))
            continue
        if not rows:
            rows = getattr(stamp, 'physical_occurrences', None)
        if rows:
            place_mos(logical, stamp_name, rows)

    for logical in sorted(passive):
        candidates = [stamp for name, stamp in geometry.items()
                      if logical in (getattr(stamp, 'composite_members', ()) or (name,))]
        _require(len(candidates) == 1, 'current passive occurrence stamp is missing or ambiguous')
        stamp = candidates[0]
        from clay.layout.source_resistor_placement import is_resistor_stamp, resistor_stamp_occurrences
        if is_resistor_stamp(stamp):
            _require(logical_kinds[logical] in {'res', 'resistor'}, 'current resistor stamp has a foreign logical kind')
            stamp_names = [name for name, value in geometry.items() if value is stamp]
            _require(len(stamp_names) == 1, 'current resistor stamp identity is ambiguous')
            record(logical, resistor_stamp_occurrences(stamp, placement_um[stamp_names[0]])[logical])
            continue
        binding = getattr(stamp, 'materialized_passive_access', None)
        if binding is not None:
            from clay.layout.passive_terminal_binding import _require as require_native_passive
            binding = require_native_passive(binding)
            native_metadata = json.loads(binding.native_metadata_json)
            native_fixed = native_metadata.get('fixed_pcell') or {}
            canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
            _require(canonical(getattr(stamp, 'fixed_pcell', None) or {}) == canonical(native_fixed),
                     'current passive occurrence fixed metadata differs from native materialization')
            orientation = binding.orient
            expected_kind = 'res' if logical_kinds[logical] in {'res', 'resistor'} else 'cap'
            _require(binding.kind == expected_kind, 'current passive occurrence materialized kind differs')
        else:
            orientation = getattr(stamp, 'orient', None)
        if logical_kinds[logical] in {'resistor', 'res'}:
            box = source_device_bboxes[logical]
            record(logical, (((box[0] + box[2]) // 2, (box[1] + box[3]) // 2,
                              orientation),))
        else:
            metadata = (getattr(binding, 'capacitor_unit_occurrences', None)
                        if binding is not None else getattr(stamp, 'capacitor_unit_occurrences', None))
            if metadata is not None:
                from clay.layout.capacitor_unit_occurrences import CapacitorUnitOccurrenceGeometry
                from clay.layout.device_cell import DeviceGeom
                members = getattr(stamp, 'composite_members', ()) or (logical,)
                _require(set(logical_kinds).intersection(members) == {logical},
                         'current capacitor occurrence stamp has ambiguous logical ownership')
                _require(type(metadata) is CapacitorUnitOccurrenceGeometry,
                         'current capacitor occurrence metadata is untyped')
                metadata.__post_init__()
                if binding is not None:
                    replayed = CapacitorUnitOccurrenceGeometry.from_data(
                        native_metadata.get('capacitor_unit_occurrences'))
                    _require(replayed == metadata and type(binding.multiplier) is int
                             and binding.multiplier == metadata.unit_count
                             and binding.orient == metadata.bank_orientation,
                             'current capacitor occurrence metadata differs from native binding')
                elif type(stamp) is DeviceGeom:
                    _require(stamp.kind == metadata.kind and stamp.orient == metadata.bank_orientation
                             and stamp.finger_count == metadata.unit_count
                             and tuple(stamp.extent()) == metadata.bank_extent_nm,
                             'current capacitor occurrence metadata differs from native materialization')
                placed_nm_rows(logical, stamp, metadata.unit_centers_nm,
                               metadata.unit_transforms, metadata.bank_extent_nm)
                continue
            if binding is not None:
                _require(native_metadata.get('capacitor_unit_occurrences') is None,
                         'current capacitor occurrence binding lost its typed metadata')
            fixed = getattr(stamp, 'fixed_pcell', None)
            _require(type(fixed) is dict and orientation == 'R0',
                     'current passive unit occurrence metadata/orientation differs')
            local_box = tuple(fixed.get('bank_extent_nm', ())) if binding is not None else tuple(stamp.extent())
            _require(fixed.get('bank_extent_nm') == local_box, 'current fixed bank frame differs')
            placed_nm_rows(logical, stamp, fixed.get('unit_centers_nm'),
                           fixed.get('unit_transforms'), local_box)
    for logical in sorted(bjts):
        from clay.layout.bjt_device_cell import BjtCellGeom, BjtUnitOccurrenceGeometry
        candidates = [(name, stamp) for name, stamp in geometry.items()
                      if logical in (getattr(stamp, 'composite_members', ()) or (name,))]
        _require(len(candidates) == 1, 'current BJT occurrence stamp is missing or ambiguous')
        stamp_name, stamp = candidates[0]
        members = getattr(stamp, 'composite_members', ()) or (stamp_name,)
        _require(set(logical_kinds).intersection(members) == {logical},
                 'current BJT occurrence stamp has ambiguous logical ownership')
        metadata = (stamp.unit_occurrences if type(stamp) is BjtCellGeom
                    else getattr(stamp, 'bjt_unit_occurrences', None))
        _require(type(metadata) is BjtUnitOccurrenceGeometry,
                 'current BJT occurrence geometry is missing or untyped')
        metadata.__post_init__()
        _require(metadata.kind == logical_kinds[logical], 'current BJT occurrence kind differs')
        if type(stamp) is BjtCellGeom:
            _require(metadata.kind == stamp.kind and metadata.source_sha256 == stamp.source_sha256
                     and metadata.unit_count == stamp.unit_count
                     and metadata.bank_extent_nm == stamp.extent(),
                     'current BJT occurrence metadata differs from native materialization')
        placed_nm_rows(logical, stamp, metadata.unit_centers_nm,
                       metadata.unit_transforms, metadata.bank_extent_nm)
    _require(set(result) == set(logical_kinds), 'complete current occurrence inventory is missing members')
    return dict(sorted(result.items()))


def project_rigid_quality_geometry(*, source_blocks, solver_blocks,
        source_block_logicals, solver_block_logicals, member_offsets, source_occurrences,
        source_device_bboxes):
    """Conserve all source members and apply native composite-member offsets.

    Each absolute source point is translated by -source_origin + member_offset.
    Solver initial XY is never an output or a quality target.
    Logical artwork bboxes use the very same transform and define module metrics.
    Source/solver envelopes retain physical ownership but cannot stand in for
    logical artwork, including when multiple functional modules share one row.
    """
    from clay.layout.reference_quality_contract import (
        RigidMaterializedBlock, RigidOccurrence, RigidQualityGeometryIR,
    )
    sources, owners = _boxes(source_blocks, 'source'), _boxes(solver_blocks, 'solver')
    source_of = _partition(source_block_logicals, set(sources), 'source')
    owner_of = _partition(solver_block_logicals, set(owners), 'solver')
    _require(set(source_of) == set(owner_of), 'source/solver logical partition differs')
    _require(type(source_device_bboxes) is dict and set(source_device_bboxes) == set(source_of),
             'complete current logical artwork bbox inventory differs')
    artwork = _boxes(tuple(sorted(source_device_bboxes.items())), 'logical artwork')
    _require(isinstance(source_occurrences, Mapping) and set(source_occurrences) == set(source_of),
             'complete current occurrence inventory differs')
    _require(type(member_offsets) is dict and set(member_offsets) <= set(sources),
             'source member offset inventory differs')
    for source, row in member_offsets.items():
        _require(type(row) is tuple and len(row) == 2 and row[0] in owners
                 and type(row[1]) is tuple and len(row[1]) == 2
                 and all(type(x) is int for x in row[1]), 'source member owner/offset row differs')

    transforms = {}
    for source, box in sorted(sources.items()):
        logicals = source_block_logicals[source]
        targets = {owner_of[logical] for logical in logicals}
        _require(len(targets) == 1, 'source block straddles solver owner partition')
        owner = next(iter(targets))
        if owner == source:
            _require(box == owners[owner], 'retained source owner bbox differs')
            _require(source not in member_offsets or member_offsets[source] == (owner, (0, 0)),
                     'retained source owner has foreign offset')
            offset = (0, 0)
        else:
            _require(source in member_offsets and member_offsets[source][0] == owner,
                     'source block lacks its exact solver owner offset')
            offset = member_offsets[source][1]
        local = (offset[0], offset[1], offset[0] + box[2] - box[0], offset[1] + box[3] - box[1])
        width, height = owners[owner][2] - owners[owner][0], owners[owner][3] - owners[owner][1]
        _require(0 <= local[0] < local[2] <= width and 0 <= local[1] < local[3] <= height,
                 'materialized block offset escapes solver owner')
        transforms[source] = (owner, offset[0] - box[0], offset[1] - box[1])

    occurrences, materialized = [], []
    allowed = {'R0', 'R90', 'R180', 'R270', 'MR0', 'MR90', 'MR180', 'MR270'}
    for logical, rows in sorted(source_occurrences.items()):
        _require(type(rows) is tuple and rows, 'current occurrence rows are empty or malformed')
        source = source_of[logical]
        box = sources[source]
        owner, dx, dy = transforms[source]
        art = artwork[logical]
        _require(box[0] <= art[0] < art[2] <= box[2]
                 and box[1] <= art[1] < art[3] <= box[3],
                 'logical artwork bbox escapes source block')
        materialized.append(RigidMaterializedBlock(logical, owner,
            (art[0]+dx, art[1]+dy, art[2]+dx, art[3]+dy), (logical,)))
        for row in rows:
            _require(type(row) is tuple and len(row) == 3 and type(row[0]) is int
                     and type(row[1]) is int and type(row[2]) is str and row[2] in allowed,
                     'current occurrence point requires integer coordinates and orientation')
            x, y, orientation = row
            _require(art[0] <= x <= art[2] and art[1] <= y <= art[3],
                     'current occurrence point escapes logical artwork')
            occurrences.append(RigidOccurrence(logical, owner, (x + dx, y + dy), orientation))
    return RigidQualityGeometryIR(
        owners=tuple((owner, box[2] - box[0], box[3] - box[1]) for owner, box in sorted(owners.items())),
        occurrences=tuple(sorted(occurrences, key=lambda r: (
            r.logical_id, r.center_offset_nm, r.orientation, r.owner_id))),
        materialized_blocks=tuple(materialized),
    )
