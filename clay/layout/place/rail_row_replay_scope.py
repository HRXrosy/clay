"""One rail invocation's source-row proof and complete live input snapshot."""
from contextlib import contextmanager
from contextvars import ContextVar


_RAIL_PROOF = ContextVar('clay_rail_row_replay_proof', default=None)


def require_rail_row_materialization(materialization, *, program, tech_id):
    from clay.layout.place import row_replay_session
    proof = row_replay_session._CURRENT.get()
    if (proof is not None and proof is _RAIL_PROOF.get()
            and proof.materialization is materialization and proof.program is program):
        return row_replay_session.require_current_row_materialization(materialization, program=program, tech_id=tech_id)
    from clay.layout.place.row_program import verify_row_program_materialization
    return verify_row_program_materialization(materialization, program=program, tech_id=tech_id)


def rail_row_source_program(projection, cell, tech, *, program, materialization):
    from clay.layout.place import row_replay_session
    proof = row_replay_session._CURRENT.get()
    witness = getattr(projection, 'native_witness', None)
    if (proof is not None and proof is _RAIL_PROOF.get()
            and proof.materialization is materialization and proof.program is program
            and cell is getattr(witness, 'cell', None)):
        return row_replay_session.current_row_source_program(
            projection, cell, tech, program=program, materialization=materialization)
    from clay.layout.place.row_source_projection import replay_row_source_projection
    return replay_row_source_projection(projection, cell, tech)


def _snapshot(operands):
    from clay.layout.extract.library_device import CellGeometry
    from clay.layout.intent import LayoutIntent
    from clay.layout.place.row_replay_snapshot import freeze_row_values
    from clay.layout.route.route_cell import _BridgedIntent, _PlacedBjtStamp

    intent, bridged, geometry = operands['intent'], operands['bridged_intent'], operands['geometry']
    if type(intent) is not LayoutIntent or type(bridged) is not _BridgedIntent or type(geometry) is not dict:
        raise TypeError('rail replay snapshot currency is unsupported')
    if any(type(name) is not str for name in geometry):
        raise TypeError('rail replay snapshot stamp names are unsupported')
    stamps = []
    for name, stamp in sorted(geometry.items()):
        if type(stamp) is CellGeometry:
            raw = vars(stamp)
        elif type(stamp) is _PlacedBjtStamp:
            raw = {name: getattr(stamp, name) for cls in type(stamp).__mro__
                   for name in cls.__dict__.get('__slots__', ()) if hasattr(stamp, name)}
        else:
            raise TypeError('rail replay snapshot stamp type is unsupported')
        stamps.append((name, id(stamp), type(stamp), freeze_row_values(raw)))
    block_stamps = operands['block_stamps']
    if type(block_stamps) is not dict or any(type(name) is not str for name in block_stamps):
        raise TypeError('rail replay snapshot block stamp currency is unsupported')
    normalized = []
    for name, values in sorted(block_stamps.items()):
        if type(values) not in (set, frozenset, tuple, list) or any(type(value) is not str for value in values):
            raise TypeError('rail replay snapshot block members are unsupported')
        rows = tuple(sorted(values)) if type(values) in (set, frozenset) else tuple(values)
        normalized.append((name, type(values), rows))
    bridged_values = tuple((name, getattr(bridged, name)) for name in _BridgedIntent.__slots__)
    fields = ('solver_blocks', 'materialized_blocks', 'block_gring', 'placement_um',
              'master_members', 'member_offsets_nm', 'flexible_envelope_constraints')
    return (id(intent), freeze_row_values(intent), id(bridged), freeze_row_values(bridged_values),
            id(geometry), tuple(stamps), id(block_stamps), tuple(normalized),
            tuple((name, id(operands[name]), freeze_row_values(operands[name])) for name in fields))


@contextmanager
def rail_source_row_scope(operands):
    """Reuse source proof only for exact native inputs in this one rail call."""
    if (getattr(operands['tech'], 'pdk_id', None) != 'sky130'
            or operands['row_program'] is None or operands['row_materialization'] is None
            or operands['cell_def'] is None or operands['native_geometry'] is None):
        token = _RAIL_PROOF.set(None)
        try:
            yield False
        finally:
            _RAIL_PROOF.reset(token)
        return
    from clay.layout.common_centroid_array import CommonCentroidArray
    from clay.layout.mosaic_geometry import MosaicGeometry
    from clay.layout.joint_row_geometry import JointRowGeometryBinding
    from clay.layout.native_diffusion_source import NativePCellWitness
    from clay.layout.place.row_program import RowProgram, RowProgramMaterialization
    from clay.layout.place import row_replay_session

    program, materialization = operands['row_program'], operands['row_materialization']
    tech, cell, bundle = operands['tech'], operands['cell_def'], operands['native_geometry']
    projection = getattr(materialization, 'source_projection', None)
    witness = getattr(projection, 'native_witness', None)
    current = row_replay_session._CURRENT.get()
    parent_pdk_state = (
        row_replay_session._pdk_state(tech, native_access_profile=True)
        if current is not None and getattr(current, 'native_access_profile', False)
        else (row_replay_session._pdk_state(tech) if current is not None else None)
    )
    same_parent = (current is None or (current.materialization is materialization
        and current.program is program and parent_pdk_state == current.pdk_values))
    supported = (getattr(tech, 'pdk_id', None) == 'sky130' and type(witness) is NativePCellWitness
                 and type(program) is RowProgram and type(materialization) is RowProgramMaterialization
                 and cell is witness.cell and bundle is witness.bundle
                 and bool(materialization.realizations)
                 and all(type(row.geometry) in (CommonCentroidArray, MosaicGeometry)
                         and type(getattr(row.geometry, 'joint_row_binding', None)) is JointRowGeometryBinding
                         for row in materialization.realizations)
                 and same_parent)
    if not supported:
        token = _RAIL_PROOF.set(None)
        try:
            yield False
        finally:
            _RAIL_PROOF.reset(token)
        return
    try:
        before = _snapshot(operands)
    except (TypeError, ValueError):
        # Unknown operand currencies retain their existing independent replay.
        token = _RAIL_PROOF.set(None)
        try:
            yield False
        finally:
            _RAIL_PROOF.reset(token)
        return
    def check():
        try:
            actual = _snapshot(operands)
        except (TypeError, ValueError) as exc:
            raise ValueError('rail source scope operands changed') from exc
        if actual != before:
            raise ValueError('rail source scope operands changed')
    try:
        with row_replay_session.native_matching_row_scope(program=program, materialization=materialization, tech=tech) as proof:
            check()
            token = _RAIL_PROOF.set(proof)
            try:
                yield True
                check()
            finally:
                _RAIL_PROOF.reset(token)
        # Include the source/PDK scope's finish and refresh in the guarded span.
        check()
    except BaseException as error:
        try:
            check()
        except BaseException as ending:
            from clay.flow.control_stop import record_unwind_failure
            record_unwind_failure(error, ending)
        raise
