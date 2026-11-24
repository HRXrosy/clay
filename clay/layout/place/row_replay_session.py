"""Invocation-owned reuse of a fully replayed joint row materialization.

Public standalone verifiers remain independent. Internal consumers can reuse
the current graph only while every original operand still equals a detached
snapshot. The scope is never stored on a placement or accepted from a receipt.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from threading import current_thread

from clay.layout.place.row_replay_snapshot import (
    freeze_row_values, geometry_snapshot, materialization_snapshot, source_projection_snapshot,
)


def _require(value, message):
    if not value:
        raise ValueError('row replay scope '+message)


def _owner():
    import asyncio
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), current_thread(), task


def row_replay_phase(name, **annotations):
    from clay.flow.execution_session import current_session
    session = current_session()
    if session is None:
        return nullcontext()
    session.checkpoint()
    return session.phase(name, details=annotations)


def row_phase_call(name, function, /, *args, **kwargs):
    with row_replay_phase(name):
        return function(*args, **kwargs)


def _pdk_state(tech, *, refresh=False, native_access_profile=False):
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.tech import LayoutTech
    _require(type(tech) is LayoutTech, 'PDK type differs')
    tech.rules
    snapshot = tech.adapter.source_snapshot(refresh=refresh)
    state=(tech.pdk_id, pdk_provenance_digest(tech),
        None if snapshot is None else snapshot.snapshot_sha256,
        None if snapshot is None else snapshot.installed_authority_sha256)
    if native_access_profile:
        from clay.layout.native_buffer_access import shared_source_rule_state
        state+=(shared_source_rule_state(tech),)
    return state


@dataclass(frozen=True, slots=True)
class _Entry:
    geometry: object
    members: tuple
    snapshot: tuple
    digest: str
    inventory: object


class _Replay:
    def __init__(self, *, source_cell, tech):
        self.owner, self.active = _owner(), True
        self.allow_reentrant = False
        self.source_cell, self.tech = source_cell, tech
        self.program = self.materialization = None
        self.native_access_profile = False
        self.entries = {}
        self.source_files = ()
        self.ready = False

    def _owned(self):
        _require(self.active, 'has expired or closed')
        _require(self.owner == _owner(), 'belongs to another invocation owner/thread/task')

    def bind(self, program, materialization, *, trust_sealed_scene=False):
        from clay.frontend.source_revision import reopen_complete_source
        from clay.layout.place import row_program as rows
        from clay.layout.place.row_source_projection import require_row_source_projection
        self._owned()
        if self.materialization is not None:
            if self.allow_reentrant and program is self.program and materialization is self.materialization:
                self.require_current()
                return
            _require(False, 'cannot bind a second materialization')
        _require(type(program) is rows.RowProgram and type(materialization) is rows.RowProgramMaterialization,
                 'requires exact program and materialization')
        self.program, self.materialization = program, materialization
        self.program_values = freeze_row_values(program)
        self.materialization_values = materialization_snapshot(materialization)
        self.projection = materialization.source_projection
        self.projection_identities = () if self.projection is None else tuple(
            getattr(self.projection, name) for name in ('reference_overlay', 'reference_asir',
                'projected_program', 'fact_authority', 'authoring_source_cell', 'native_witness'))
        if self.projection is not None:
            require_row_source_projection(self.projection, program=program)
            _require(self.source_cell is not None, 'source projection requires current complete source')
        files = {}
        if self.source_cell is not None:
            source = reopen_complete_source(self.source_cell)
            files[source['path']] = source['source_sha256']
            self.source_values = freeze_row_values(vars(self.source_cell))
            if self.projection is not None:
                _require(source['source_sha256'] == self.projection.source_sha256
                    and source['source_inventory_sha256'] == self.projection.source_inventory_sha256,
                    'complete source differs from its captured projection')
                authoring = reopen_complete_source(self.projection.authoring_source_cell)
                _require(authoring['source_sha256'] == self.projection.authoring_source_sha256,
                         'authoring source differs from its captured projection')
                files[authoring['path']] = authoring['source_sha256']
                if self.projection.source_cir is not None:
                    files[str(Path(self.projection.source_cir).resolve(strict=True))] = self.projection.topology_sha256
        self.source_files = tuple(sorted(files.items()))
        self.replayed_source_program = None
        self.native_snapshot_program = None
        self.native_access_profile = False
        if self.projection is not None and self.projection.native_witness is not None:
            from clay.layout.intent import layout_intent_from_cell_def,replay_array_intent
            source_intent=layout_intent_from_cell_def(self.projection.native_witness.cell)
            self.native_access_profile=any(replay_array_intent(row).access_profile or replay_array_intent(row).generation_profile for row in source_intent.arrays)
        self.pdk_values = _pdk_state(self.tech, refresh=True,native_access_profile=self.native_access_profile)
        native_after = None
        if self.projection is not None and self.projection.native_witness is not None:
            from clay.layout.place.row_source_projection import replay_row_source_projection
            # Record every mutable witness operand before the full replay and
            # compare it again afterwards: a concurrent write must not become
            # the new authority merely because it landed during validation.
            native_before = source_projection_snapshot(
                self.projection, validated_native_program=program)
            self.replayed_source_program = replay_row_source_projection(
                self.projection,
                self.projection.native_witness.cell,
                self.tech,
                trust_fresh_witness=(self.projection.native_witness if trust_sealed_scene else None),
            )
            _require(self.replayed_source_program.to_canonical_bytes() == program.to_canonical_bytes(),
                     'source-replayed row program differs')
            self.native_snapshot_program = program
            native_after = source_projection_snapshot(self.projection, validated_native_program=program)
            _require(native_after == native_before,
                     'native source operands changed during complete replay')
            _require(_pdk_state(self.tech, refresh=True,native_access_profile=self.native_access_profile) == self.pdk_values,
                     'PDK changed during complete source replay')
        self.projection_values = (
            native_after
            if native_after is not None and self.native_snapshot_program is program
            else source_projection_snapshot(
                self.projection, validated_native_program=self.native_snapshot_program
            )
        )
        if self.native_snapshot_program is not None:
            _require(self.projection_values == native_before,
                     'native source operands changed before scope binding')
        members = {member.device: member for member in program.members}

        def replay_entry(realization, entry):
            from clay.layout.joint_row_terminals import require_terminal_member_nets
            before = geometry_snapshot(realization.geometry, realization.consumed_devices)
            with row_replay_phase('row_replay.native') as phase:
                if phase is not None:
                    phase.annotate(entry_id=realization.entry_id, owners=len(realization.consumed_devices))
                if trust_sealed_scene:
                    binding = getattr(realization.geometry, 'joint_row_binding', None)
                    inventory = getattr(binding, 'joint_inventory', None)
                    _require(inventory is not None, 'sealed scene row lacks joint terminal inventory')
                else:
                    inventory = require_terminal_member_nets(realization.geometry, realization.consumed_devices,
                        members, self.tech)
                digest = rows._row_geometry_digest_from_inventory(realization.geometry,
                    realization.consumed_devices, inventory)
            _require(before == geometry_snapshot(realization.geometry, realization.consumed_devices),
                     'native geometry changed during validation')
            _require(id(realization.geometry) not in self.entries, 'native geometry is reused across entries')
            self.entries[id(realization.geometry)] = _Entry(realization.geometry,
                realization.consumed_devices, before, digest, inventory)
            return digest

        rows._verify_row_program_materialization(materialization, program=program,
            tech_id=self.tech.pdk_id, entry_replay=replay_entry)
        self.ready = True
        self.require_current()

    def _entry(self, geometry, members, *, precomputed_snapshot=None):
        self._owned()
        value = self.entries.get(id(geometry))
        _require(value is not None and value.geometry is geometry and tuple(members) == value.members,
                 'foreign geometry or consumed member identity')
        _require(getattr(getattr(geometry, 'joint_row_binding', None), 'joint_inventory', None) is value.inventory,
                 'validated terminal inventory identity detached from current geometry')
        actual_snap = (
            precomputed_snapshot[1]
            if precomputed_snapshot is not None and precomputed_snapshot[0] == value.members
            else geometry_snapshot(geometry, value.members)
        )
        _require(actual_snap == value.snapshot,
                 'native geometry/source operands changed')
        return value

    def geometry_sha256(self, geometry, members):
        return self._entry(geometry, members).digest

    def terminal_inventory(self, geometry, members):
        return self._entry(geometry, members).inventory

    def require_current(self):
        self._owned()
        _require(self.materialization is not None, 'has no materialization')
        _require(freeze_row_values(self.program) == self.program_values,
                 'program member/net/count or declaration changed')
        _require(materialization_snapshot(self.materialization) == self.materialization_values,
                 'materialization frame/certificate changed')
        live_geom_snapshots = {}
        _require(self.materialization.source_projection is self.projection
            and source_projection_snapshot(self.projection,
                validated_native_program=self.native_snapshot_program,
                computed_geometry_snapshots=live_geom_snapshots) == self.projection_values,
            'source projection identity or operands changed')
        if self.projection is not None:
            identities = tuple(getattr(self.projection, name) for name in ('reference_overlay',
                'reference_asir', 'projected_program', 'fact_authority', 'authoring_source_cell', 'native_witness'))
            _require(all(current is original for current, original in zip(identities, self.projection_identities)),
                     'source projection captured operand identity changed')
        if self.source_cell is not None:
            _require(freeze_row_values(vars(self.source_cell)) == self.source_values,
                     'complete source cell changed')
        for path, expected in self.source_files:
            _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected,
                     'source/topology input bytes changed')
        _require(_pdk_state(self.tech,native_access_profile=self.native_access_profile) == self.pdk_values, 'current PDK changed')
        _require(len(self.materialization.realizations) == len(self.entries), 'entry denominator changed')
        for realization in self.materialization.realizations:
            self._entry(realization.geometry, realization.consumed_devices,
                        precomputed_snapshot=live_geom_snapshots.get(id(realization.geometry)))
        return True

    def finish(self):
        self._owned()
        if self.ready:
            with row_replay_phase('row_replay.finish'):
                self.require_current()
                _require(_pdk_state(self.tech, refresh=True,native_access_profile=self.native_access_profile) == self.pdk_values, 'ending PDK changed')


_CURRENT: ContextVar[_Replay | None] = ContextVar('clay_row_materialization_replay', default=None)


@contextmanager
def row_replay_invocation(*, source_cell=None, tech):
    """Private lifecycle selected by one synchronous GP or bridge invocation."""
    current = _CURRENT.get()
    if current is not None:
        _require(current.allow_reentrant, 'nested active invocation is unsupported')
        current._owned()
        _require(
            _pdk_state(tech, native_access_profile=current.native_access_profile) == current.pdk_values,
            'reentrant invocation PDK differs',
        )
        if source_cell is not None:
            _require(current.source_cell is not None, 'reentrant invocation source cell was not bound')
            if source_cell is not current.source_cell:
                from clay.frontend.source_revision import reopen_complete_source
                _require(
                    reopen_complete_source(source_cell) == reopen_complete_source(current.source_cell)
                    and freeze_row_values(vars(source_cell)) == current.source_values,
                    'reentrant invocation source cell differs',
                )
        if current.ready:
            current.require_current()
        try:
            yield current
        except BaseException as error:
            try:
                current.finish()
            except BaseException as ending:
                from clay.flow.control_stop import record_unwind_failure
                record_unwind_failure(error, ending)
            raise
        else:
            current.finish()
        return
    proof = _Replay(source_cell=source_cell, tech=tech)
    token = _CURRENT.set(proof)
    try:
        try:
            yield proof
        except BaseException as error:
            try:
                proof.finish()
            except BaseException as ending:
                from clay.flow.control_stop import record_unwind_failure
                record_unwind_failure(error, ending)
            raise
        else:
            proof.finish()
    finally:
        proof.active = False
        _CURRENT.reset(token)


@contextmanager
def row_materialization_replay_scope(*, program, materialization, tech, source_cell=None):
    with row_replay_invocation(source_cell=source_cell, tech=tech) as proof:
        proof.bind(program, materialization)
        yield proof


@contextmanager
def shared_row_replay_session(*, program, materialization, tech, source_cell=None, trust_sealed_scene=False):
    """Share one validated joint-row replay across nested placement/route steps.

    Standalone scopes remain non-reentrant unless opted into this explicit
    session. Every nested consumer and scope boundary still verifies live
    operand snapshots, source bytes, PDK state, and owner thread/task identity.
    """
    from clay.ir import CellDef
    from clay.layout.common_centroid_array import CommonCentroidArray
    from clay.layout.mosaic_geometry import MosaicGeometry
    from clay.layout.joint_row_geometry import JointRowGeometryBinding
    from clay.layout.place.row_program import RowProgram, RowProgramMaterialization
    from clay.layout.tech import LayoutTech

    if type(program) is not RowProgram or type(materialization) is not RowProgramMaterialization:
        yield None
        return
    projection = materialization.source_projection
    if projection is None:
        yield None
        return
    values = materialization.realizations
    if not values or not all(
        type(row.geometry) in (CommonCentroidArray, MosaicGeometry)
        and type(getattr(row.geometry, 'joint_row_binding', None)) is JointRowGeometryBinding
        for row in values
    ):
        yield None
        return
    effective_cell = (
        source_cell
        if source_cell is not None
        else getattr(getattr(projection, 'native_witness', None), 'cell', None)
    )
    if effective_cell is None or type(effective_cell) is not CellDef:
        yield None
        return
    tech_obj = tech if type(tech) is LayoutTech else LayoutTech.resolve(str(tech))
    current = _CURRENT.get()
    if current is not None:
        current._owned()
        _require(
            current.program is program and current.materialization is materialization,
            'shared session received a foreign row materialization',
        )
        _require(
            _pdk_state(tech_obj, native_access_profile=current.native_access_profile) == current.pdk_values,
            'shared session PDK differs',
        )
        if current.source_cell is not None and effective_cell is not current.source_cell:
            from clay.frontend.source_revision import reopen_complete_source
            _require(
                reopen_complete_source(effective_cell) == reopen_complete_source(current.source_cell)
                and freeze_row_values(vars(effective_cell)) == current.source_values,
                'shared session source cell differs',
            )
        prev_reentrant = current.allow_reentrant
        current.allow_reentrant = True
        current.require_current()
        try:
            try:
                yield current
            except BaseException as error:
                try:
                    current.require_current()
                except BaseException as ending:
                    from clay.flow.control_stop import record_unwind_failure
                    record_unwind_failure(error, ending)
                raise
            else:
                current.require_current()
        finally:
            current.allow_reentrant = prev_reentrant
        return
    with row_replay_invocation(source_cell=effective_cell, tech=tech_obj) as proof:
        proof.bind(program, materialization, trust_sealed_scene=trust_sealed_scene)
        proof.allow_reentrant = True
        yield proof


def bind_current_row_materialization(materialization, *, program, tech):
    """Bind supported complete joint rows; legacy currencies keep full replay."""
    from clay.ir import CellDef
    from clay.layout.common_centroid_array import CommonCentroidArray
    from clay.layout.mosaic_geometry import MosaicGeometry
    from clay.layout.joint_row_geometry import JointRowGeometryBinding
    proof = _CURRENT.get()
    if proof is None or materialization is None or program is None:
        return
    proof._owned()
    if materialization.source_projection is None:
        # Existing geometry-only placements have no file-source obligation.
        # They retain every independent public replay and gain no source scope.
        return
    values = materialization.realizations
    supported = values and all(type(row.geometry) in (CommonCentroidArray, MosaicGeometry)
        and type(getattr(row.geometry, 'joint_row_binding', None)) is JointRowGeometryBinding for row in values)
    if not supported or proof.source_cell is not None and type(proof.source_cell) is not CellDef:
        return
    _require(
        tech is proof.tech or (
            proof.allow_reentrant
            and _pdk_state(tech, native_access_profile=proof.native_access_profile) == proof.pdk_values
        ),
        'binding PDK object identity differs',
    )
    proof.bind(program, materialization)


def require_current_row_materialization(materialization, *, program=None, tech_id=None):
    """Internal consumer boundary. Public standalone verification never uses it."""
    proof = _CURRENT.get()
    if proof is None or proof.materialization is None:
        from clay.layout.place.row_program import verify_row_program_materialization
        return verify_row_program_materialization(materialization, program=program, tech_id=tech_id)
    proof._owned()
    _require(
        materialization is proof.materialization
        and (
            program is None
            or program is proof.program
            or (proof.replayed_source_program is not None and program is proof.replayed_source_program)
        ),
        'foreign program/materialization identity',
    )
    _require(tech_id is None or tech_id == proof.tech.pdk_id, 'consumer PDK identity differs')
    return proof.require_current()


def current_row_projection_snapshot(projection, *, program, materialization):
    """Snapshot only against the current, fully replayed matching authority."""
    proof = _CURRENT.get()
    if proof is None or proof.materialization is None:
        return source_projection_snapshot(projection)
    proof._owned()
    _require(program is proof.program and materialization is proof.materialization
             and projection is proof.projection, 'foreign snapshot row authority')
    proof.require_current()
    return source_projection_snapshot(projection,
        validated_native_program=proof.native_snapshot_program)


def current_row_geometry_sha256(geometry, members):
    proof = _CURRENT.get()
    if proof is not None and proof.materialization is not None:
        return proof.geometry_sha256(geometry, members)
    from clay.layout.place.row_program import row_geometry_sha256
    return row_geometry_sha256(geometry, members)


def current_row_terminal_inventory(geometry, consumed_devices, members, tech):
    proof = _CURRENT.get()
    if proof is not None and proof.materialization is not None:
        _require(freeze_row_values(tuple(members.values())) == freeze_row_values(proof.program.members),
                 'terminal program members differ')
        _require(tech is None or _pdk_state(tech,native_access_profile=proof.native_access_profile) == proof.pdk_values, 'terminal current PDK differs')
        return proof.terminal_inventory(geometry, consumed_devices)
    from clay.layout.joint_row_terminals import require_terminal_member_nets
    return require_terminal_member_nets(geometry, consumed_devices, members, tech)


def current_row_source_program(projection, canonical_cell, tech, *, program, materialization):
    """Reuse only this invocation's completely replayed canonical source."""
    proof = _CURRENT.get()
    if proof is None or proof.materialization is None or proof.replayed_source_program is None:
        from clay.layout.place.row_source_projection import replay_row_source_projection
        return replay_row_source_projection(projection, canonical_cell, tech)
    proof._owned()
    _require(projection is proof.projection and program is proof.program
             and materialization is proof.materialization, 'foreign source projection or row authority')
    _require(_pdk_state(tech,native_access_profile=proof.native_access_profile) == proof.pdk_values, 'source replay PDK differs')
    # The actual witness object was fully reopened and replayed when this live
    # scope bound.  Its exact source bytes and all witness operands are checked
    # again below.  Other CellDef objects retain the independent full reopen.
    witness = projection.native_witness
    bound_path = None
    if canonical_cell is witness.cell:
        span = getattr(canonical_cell, 'source_span', None)
        if type(span) is dict and type(span.get('file')) is str:
            bound_path = str(Path(span['file']).resolve(strict=True))
    bound_source = (bound_path, projection.source_sha256) in proof.source_files
    if not bound_source:
        from clay.frontend.source_revision import reopen_complete_source
        source = reopen_complete_source(canonical_cell)
        _require(source['source_sha256'] == projection.source_sha256
                 and source['source_inventory_sha256'] == projection.source_inventory_sha256,
                 'source replay canonical cell differs')
    _require(canonical_cell.local_name == projection.source_cell_name,
             'source replay canonical cell differs')
    _require(freeze_row_values(vars(canonical_cell)) == freeze_row_values(vars(projection.native_witness.cell)),
             'source replay canonical in-memory cell changed')
    proof.require_current()
    return proof.replayed_source_program


@contextmanager
def native_matching_row_scope(*, program, materialization, tech):
    """Own one matching capture/replay, or reuse its matching live parent scope."""
    current = _CURRENT.get()
    if current is not None:
        current._owned()
        _require(current.program is program and current.materialization is materialization,
                 'matching scope received a foreign row materialization')
        _require(_pdk_state(tech,native_access_profile=current.native_access_profile) == current.pdk_values, 'matching scope PDK differs')
        current.require_current()
        try:
            yield current
        except BaseException as error:
            try:
                current.require_current()
            except BaseException as ending:
                from clay.flow.control_stop import record_unwind_failure
                record_unwind_failure(error, ending)
            raise
        else:
            current.require_current()
        return
    projection = getattr(materialization, 'source_projection', None)
    witness = getattr(projection, 'native_witness', None)
    if witness is None:
        raise ValueError('matching scope requires a live native source witness')
    with row_materialization_replay_scope(program=program, materialization=materialization,
                                           tech=tech, source_cell=witness.cell) as proof:
        yield proof
