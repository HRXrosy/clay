"""One invocation's current source/native structure operands.

This capture proves native consistency with a complete current source and the
observations actually consumed by its compiler. It does not authenticate an
external reference extraction, grant either complete structure obligation, or
turn an archived receipt into current authority.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path

from clay.layout.place.row_replay_snapshot import (
    freeze_row_values, geometry_snapshot, materialization_snapshot,
)
from clay.layout.place.row_detail_timing import measured as _row_measure

_SEAL = object()


def _require(value, message):
    if not value:
        raise ValueError('current structure '+message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _checkpoint():
    from clay.layout.joint_row_geometry import joint_checkpoint
    joint_checkpoint()()


def _native_values(value):
    """Detach authenticated native records, preserving private seal identities.

Only fields literally named _seal accept an opaque object, after the original
native verifier has checked its seal. The identity is a drift check, never a
substitute for that verifier. Other unsupported objects remain errors.
"""
    active, memo, count = set(), {}, 0
    def data(item, depth=0):
        nonlocal count
        count += 1
        if count % 2048 == 0:
            _checkpoint()
        _require(depth < 64 and count <= 8_000_000, 'native snapshot exceeds its bounded shape')
        kind = type(item)
        if is_dataclass(item) and not isinstance(item, type):
            if id(item) in memo:
                return memo[id(item)]
            _require(id(item) not in active, 'native snapshot contains a cycle')
            active.add(id(item))
            names = tuple(field.name for field in fields(item))
            names += tuple(sorted(set(getattr(item, '__dict__', {}))-set(names)))
            result = (kind.__module__, kind.__qualname__, tuple((name,
                ('opaque_seal_identity', id(getattr(item, name))) if name == '_seal'
                else data(getattr(item, name), depth+1)) for name in names))
            active.remove(id(item))
            memo[id(item)] = result
            return result
        if kind in (tuple, list):
            return kind.__name__, tuple(data(part, depth+1) for part in item)
        if kind is dict:
            return {key:data(part, depth+1) for key, part in item.items()}
        return item
    return freeze_row_values(data(value))


def _source_population(cell):
    from clay.layout.annotate_to_solver import kind_polarity_oxide
    rows = []
    for device in cell.devices:
        kind = kind_polarity_oxide(device.device_class)[0] or device.device_class
        family = {'nmos':'mos', 'pmos':'mos', 'cap':'mim', 'capacitor':'mim',
                  'pnp':'bjt', 'npn':'bjt', 'res':'resistor', 'resistor':'resistor'}.get(kind)
        _require(family is not None, 'source family is unsupported: '+str(kind))
        count = device.sizing.get('M', 1)
        _require(not isinstance(count, bool) and int(count) == count and count > 0,
                 'source unit count must be a positive exact integer')
        rows.append((device.instance_name, family, int(count), kind))
    _require(rows and len({row[0] for row in rows}) == len(rows), 'source owner denominator differs')
    return tuple(sorted(rows))


def _placed_values(placement):
    # These are the complete native member values read by the final adapters.
    # The aggregate GP polygon list is not a substitute for per-member ownership.
    return tuple((name, id(getattr(device, 'source_resistor_assembly', None)),
        _native_values(tuple(getattr(device, field, None) for field in (
        'label', 'kind', 'bbox', 'polys', 'terminals', 'term_layers', 'terminal_accesses',
        'fixed_pcell', 'bjt_unit_occurrences', 'materialized_passive_access'))))
        for name, device in sorted(placement.devices.items()))


def _row_values(placement):
    materialization, program = placement.row_materialization, placement.row_program
    if materialization is None:
        return _row_measure('row.program', freeze_row_values, program), None, ()
    from clay.layout.place.row_replay_session import current_row_projection_snapshot
    return (_row_measure('row.program', freeze_row_values, program),
        _row_measure('row.materialization', materialization_snapshot, materialization),
        _row_measure('row.projection', current_row_projection_snapshot, materialization.source_projection,
            program=program, materialization=materialization),
        tuple((row.entry_id, _row_measure('row.geometry', geometry_snapshot, row.geometry, row.consumed_devices))
              for row in materialization.realizations))


def _mim_replay(values, source_cell, tech, placement, expected):
    from clay.layout.source_mim_materialization import SourceMimMaterialization, replay_source_mim_materialization
    from clay.layout.passive_terminal_binding import MaterializedPassiveAccess, _polygon_rows, _origin, _local_rows
    from clay.layout.joint_row_geometry import joint_record_counter
    _require(type(values) is tuple and all(type(value) is SourceMimMaterialization for value in values),
             'MIM requires exact native materialization tuples')
    owners = tuple(value.source_member[0] for value in values)
    _require(owners == tuple(sorted(expected)), 'MIM complete source owner cover differs')
    for value in values:
        _checkpoint()
        native = replay_source_mim_materialization(value, source_cell=source_cell, tech=tech)
        name = value.source_member[0]
        binding = placement.devices[name].materialized_passive_access
        _require(type(binding) is MaterializedPassiveAccess, 'MIM placed native binding is missing')
        binding.__post_init__()
        records = _polygon_rows(native.polygons)
        origin = _origin(records)
        _require(joint_record_counter(_local_rows(records, *origin)) == joint_record_counter(binding.polygons)
            and binding.mim_site_ledger == value.site_ledger
            and binding.capacitor_unit_occurrences == native.capacitor_unit_occurrences
            and binding.multiplier == expected[name], 'MIM source and actual placed native binding differ')
    return {'logical_owners':list(owners), 'units':sum(expected.values()),
            'sites':sum(len(value.site_ledger) for value in values)}


def _bjt_replay(values, source_cell, tech, placement, intent, expected):
    from clay.layout.bjt_local_assembly import (NativeNonmosAssemblyBinding,
        replay_bjt_local_assembly, bjt_assembly_member_geometry)
    from clay.layout.nonmos_placement_inputs import replay_native_nonmos_placement
    _require(type(values) is tuple and all(type(value) is NativeNonmosAssemblyBinding for value in values),
             'BJT requires exact native assembly tuples')
    owners = tuple(sorted(member.owner for value in values for member in value.members))
    _require(owners == tuple(sorted(expected)), 'BJT complete source owner cover differs')
    from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
    all_assemblies = getattr(getattr(placement, 'rigid_nonmos_macro_inputs', None), 'native_assemblies', ())
    _require(all(type(value) in (NativeNonmosAssemblyBinding, CurrentResistorPCellAssembly)
                 for value in all_assemblies), 'native assembly family is unsupported')
    current = tuple(value for value in all_assemblies if type(value) is NativeNonmosAssemblyBinding)
    _require(len(current) == len(values) and all(first is second for first, second in zip(current, values)),
             'BJT source assembly identity differs from actual compiler inputs')
    native = {}
    if values:
        replay_native_nonmos_placement(placement, source_cell=source_cell, tech=tech, intent=intent)
    for value in values:
        _checkpoint()
        replay_bjt_local_assembly(value, source_cell=source_cell, tech=tech, intent=intent)
        for member in value.members:
            _require(member.unit_count == expected[member.owner], 'BJT complete source unit count differs')
            built = bjt_assembly_member_geometry(value, member.owner, tech=tech)
            _require(placement.devices[member.owner].bjt_unit_occurrences == built.unit_occurrences,
                     'BJT current native occurrence metadata differs')
            native[member.owner] = built
    return {'logical_owners':list(owners), 'units':sum(expected.values()),
        'sites':sum(len(value.joint_inventory.expected_sites) for value in values)}, native


def _mos_replay(placement, source_cell, tech, expected):
    from clay.layout.generic_placement import verify_row_source_compiler_authority
    from clay.layout.joint_row_geometry import JointRowGeometryBinding
    if not expected:
        _require(not getattr(placement.row_program, 'consumed_devices', ()), 'unexpected MOS row ownership')
        return {'logical_owners':[], 'units':0, 'sites':0}, ()
    program, materialization = verify_row_source_compiler_authority(placement, source_cell, tech)
    _require(tuple(sorted(program.consumed_devices)) == tuple(sorted(expected)), 'MOS complete source cover differs')
    observations = {row.node_id:row for row in materialization.source_projection.reference_overlay.occurrences}
    seen, counts, sites, rows = set(), Counter(), 0, []
    entries = {entry.entry_id:entry for entry in program.entries}
    for realization in materialization.realizations:
        _checkpoint()
        binding = getattr(realization.geometry, 'joint_row_binding', None)
        _require(type(binding) is JointRowGeometryBinding, 'complete MOS native joint binding is missing')
        entry = entries[realization.entry_id]
        for occurrence in binding.occurrences:
            observed = observations.get(occurrence.occurrence_id)
            _require(occurrence.occurrence_id not in seen and observed is not None,
                     'MOS complete source occurrence observation differs')
            _require((observed.logical_owner, observed.master_content_sha256, observed.master_artwork_sha256) ==
                (occurrence.owner, occurrence.master.variant_sha256, occurrence.master.artwork_sha256),
                'MOS source observed owner/master variant/artwork differs')
            seen.add(occurrence.occurrence_id)
            counts[occurrence.owner] += 1
        sites += len(binding.expected_sites)
        rows.append((entry, realization))
    _require(counts == Counter(expected) and seen == {node for node,row in observations.items()
             if row.logical_owner in expected}, 'MOS complete observed source unit denominator differs')
    return {'logical_owners':sorted(expected), 'units':sum(expected.values()), 'sites':sites,
        'row_entries':len(rows), 'joint_components':sum(len(row.geometry.joint_row_binding.joint_inventory.joint_components)
            for _entry,row in rows), 'routing_access_components':sum(
            len(row.geometry.joint_row_binding.joint_inventory.routing_components) for _entry,row in rows)}, tuple(rows)


class CurrentStructureOperands:
    """Captured live native operands; one checked lifetime, no receipt importer."""
    def __init__(self, *args, **kwargs):
        raise TypeError('CurrentStructureOperands requires actual native capture')

    def canonical_data(self):
        _require(getattr(self, '_seal', None) is _SEAL, 'private native capture is missing')
        return json.loads(self._data)

    @property
    def sha256(self):
        return hashlib.sha256(self._data).hexdigest()

    def _snapshot(self):
        from clay.layout.source_resistor_placement import resistor_structure_snapshot
        return (freeze_row_values(vars(self.source_cell)), freeze_row_values(self.intent),
            _placed_values(self.placement), _row_values(self.placement),
            _native_values(self.mim_materializations), _native_values(self.nonmos_assemblies),
            _native_values(self.bjt_native), resistor_structure_snapshot(self.resistor_assemblies),
            self.population, self._data,
            (id(self.source_cell), id(self.placement), id(self.placement.row_program),
             id(self.placement.row_materialization), id(getattr(self.placement.row_materialization, 'source_projection', None))),
            tuple(id(entry) for entry in getattr(self.placement.row_program, 'entries', ())),
            tuple((id(row), id(row.geometry))
                  for row in getattr(self.placement.row_materialization, 'realizations', ())),
            tuple((id(entry), id(realization), id(realization.geometry)) for entry,realization in self.rows))

    def _require_current(self, *, active=True, refresh=False):
        from clay.layout.place.row_replay_session import _owner, _pdk_state
        _require(self._seal is _SEAL and self._owner == _owner(), 'capture belongs to another invocation owner')
        _require(not self._closed and (self._active or not active), 'capture lifetime is closed or inactive')
        _checkpoint()
        _require(self.placement.physical_macro_scope_token is self._scope_token,
                 'current compiler source authority changed')
        _require(self._snapshot() == self._snapshot_value, 'source/native geometry or source operands changed')
        _require(hashlib.sha256(Path(self._source_path).read_bytes()).hexdigest() == self._source_sha,
                 'complete source bytes changed')
        _require(_pdk_state(self.tech, refresh=refresh) == self._technology, 'current PDK changed')
        if self.nonmos_assemblies:
            from clay.layout.nonmos_local_assembly import technology_identity
            _require(all(value.source.technology_identity == technology_identity(self.tech)
                         for value in self.nonmos_assemblies), 'current BJT PDK/raw source identity changed')
        for producer in {id(value.producer):value.producer for value in self.resistor_assemblies}.values():
            producer.require_current()
        # Association files are dependencies too; a sealed typed object cannot
        # preserve their validity when the actual input bytes have changed.
        for path, expected in self._input_files:
            _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
                     'native source association input changed')

    @contextmanager
    def checked(self):
        _require(not self._active and not self._closed, 'capture lifetime is closed or already active')
        self._require_current(active=False, refresh=True)
        self._active = True
        try:
            try:
                yield self
            except BaseException as error:
                try:
                    self._require_current(refresh=True)
                except BaseException as ending:
                    from clay.flow.control_stop import record_unwind_failure
                    record_unwind_failure(error, ending)
                raise
            else:
                self._require_current(refresh=True)
        finally:
            self._active, self._closed = False, True


def capture_current_structure_operands(*, source_cell, tech, placement, intent,
                                      mim_materializations=(), nonmos_assemblies=(), resistor_assemblies=()):
    from clay.frontend.source_revision import reopen_complete_source
    from clay.layout.generic_placement import Placement
    from clay.layout.place.row_replay_session import _owner, _pdk_state
    from clay.layout.source_resistor_placement import resistor_structure_snapshot, replay_current_resistor_structure
    _require(type(placement) is Placement, 'capture requires an actual complete Placement')
    before_source = freeze_row_values(vars(source_cell))
    source = reopen_complete_source(source_cell)
    technology = _pdk_state(tech, refresh=True)
    _require(source['pdk_id'] == tech.pdk_id, 'source and current PDK differ')
    population = _source_population(source_cell)
    _require(set(placement.devices) == {row[0] for row in population}, 'complete placed source denominator differs')
    expected = {family:{name:count for name, selected,count,_kind in population if selected == family}
                for family in ('mos','mim','bjt','resistor')}
    resistor_before = resistor_structure_snapshot(resistor_assemblies)
    initial_native = (_placed_values(placement), _row_values(placement),
        _native_values(mim_materializations), _native_values(nonmos_assemblies), freeze_row_values(intent))
    mos, rows = _mos_replay(placement, source_cell, tech, expected['mos'])
    mim = _mim_replay(mim_materializations, source_cell, tech, placement, expected['mim'])
    bjts, bjt_native = _bjt_replay(nonmos_assemblies, source_cell, tech, placement, intent, expected['bjt'])
    resistors = (replay_current_resistor_structure(resistor_assemblies, source_cell=source_cell,
        tech=tech, placement=placement, intent=intent, expected_counts=expected['resistor'])
        if resistor_assemblies else {'logical_owners':[]})
    _require(resistor_structure_snapshot(resistor_assemblies) == resistor_before,
             'resistor source/native operands changed during capture')
    _require(initial_native == (_placed_values(placement), _row_values(placement),
        _native_values(mim_materializations), _native_values(nonmos_assemblies), freeze_row_values(intent)),
        'native operands changed during complete current capture')
    value = object.__new__(CurrentStructureOperands)
    value._seal, value._owner, value._active, value._closed = _SEAL, _owner(), False, False
    value.source_cell, value.tech, value.placement, value.intent = source_cell, tech, placement, intent
    value.mim_materializations, value.nonmos_assemblies = mim_materializations, nonmos_assemblies
    value.resistor_assemblies = resistor_assemblies
    value.population, value.rows, value.bjt_native = population, rows, bjt_native
    value._scope_token = placement.physical_macro_scope_token
    value._source_path, value._source_sha, value._technology = source['path'], source['source_sha256'], technology
    files = {}
    for native in mim_materializations:
        for path, digest in native.recipe.association.evidence_files:
            _require(path not in files or files[path] == digest, 'source input has contradictory bindings')
            files[path] = digest
    value._input_files = tuple(sorted(files.items()))
    value._data = _json(dict(schema='clay.current-structure-operands/v1',
        scope='current_source_native_preservation_only', source_sha256=source['source_sha256'],
        source_inventory_sha256=source['source_inventory_sha256'], electrical_sha256=source['electrical_sha256'],
        technology_identity=technology, source_population={'logical':len(population),
        'expanded':sum(row[2] for row in population), 'owners':[list(row) for row in population]},
        families={name:row for name,row in (('mos',mos),('mim',mim),('bjt',bjts),('resistor',resistors))
                  if row['logical_owners']},
        uncovered_logical_owners=[row[0] for row in population if row[1] == 'resistor' and not resistor_assemblies],
        required_structure_status='unverified', current_structure_status='unverified',
        independent_reference_authenticity='not_established_by_this_capture'))
    value._snapshot_value = value._snapshot()
    _require(freeze_row_values(vars(source_cell)) == before_source, 'source changed during current capture')
    value._require_current(active=False, refresh=True)
    return value


__all__ = ['CurrentStructureOperands', 'capture_current_structure_operands']
