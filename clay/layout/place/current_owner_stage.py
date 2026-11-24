"""Invocation-scoped source-owner observation for native candidate diagnostics.

This observes ownership only. It does not measure quality occurrences or grant
placement, routing, or signoff authority.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import inspect

from clay.layout.place.current_quality_stage import _CurrentStage, _canonical, CurrentQualityAcquisitionFailure
from clay.layout.place.generated_incumbent import canonical_bytes
from clay.layout.place.native_stage_observation import _execution_owner, native_stage_observation_scope
from clay.layout.place.placement_geometry import _canonical_materialized_block_source_members

_OWNER_OBSERVATION = ContextVar('native_candidate_owner_observation', default=None)


def capture_candidate_owners(*, solver_blocks, solver_block_member_devices, logical_kinds):
    """Capture only from the actual current materializer before its native solve."""
    state = _OWNER_OBSERVATION.get()
    if state is None:
        return
    if not state['active'] or state['owner'] != _execution_owner():
        raise CurrentQualityAcquisitionFailure('candidate owner observation is closed or foreign')
    from clay.layout.place import materialized_placement as producer
    caller = inspect.currentframe().f_back
    try:
        actual = producer.reposition_pum_2d
        if caller.f_code is not actual.__code__ or caller.f_globals is not actual.__globals__:
            raise ValueError('owner capture lacks real materializer origin')
        if solver_block_member_devices is not caller.f_locals['block_member_devices']:
            raise ValueError('owner capture lacks current solver ownership')
        if solver_blocks != tuple(caller.f_locals['new_blocks']) or logical_kinds is not caller.f_locals['device_kind']:
            raise ValueError('owner capture lacks current materializer operands')
        names = tuple(sorted(logical_kinds))
        if names != state['source_names']:
            raise ValueError('candidate bridged source owners differ')
        live = {name: set(solver_block_member_devices[name]) for name, _box in solver_blocks}
        partition = _canonical_materialized_block_source_members(solver_blocks, live, names)
        if tuple(sorted(name for _, members in partition for name in members)) != names:
            raise ValueError('candidate actual owner cover differs')
        state['partition'] = partition
        state['dims'] = {name: (box[2]-box[0], box[3]-box[1]) for name, box in solver_blocks}
    except Exception as exc:
        raise CurrentQualityAcquisitionFailure(str(exc)) from exc
    finally:
        del caller


class _OwnerStage(_CurrentStage):
    def __init__(self, stopped, partition):
        self.stage_inputs = stopped.stage_inputs
        self.geometry = None
        self.local_execution = dict(stopped.local_execution)
        self.owner_logicals = dict(partition)
        self._snapshot = self._current_bytes()
        self._owner = _execution_owner()
        self._active = True

    def _current_bytes(self):
        return (_canonical(self.stage_inputs), canonical_bytes(self.owner_logicals),
                canonical_bytes(self.local_execution))


@contextmanager
def acquire_current_owner_stage(invoke, *, source_names, build_policy):
    from clay.layout import place_blocks_2d as native
    if (type(source_names) is not tuple or not source_names or
            any(type(name) is not str or not name or name != name.strip() for name in source_names) or
            source_names != tuple(sorted(set(source_names)))):
        raise ValueError('candidate source owner names are not canonical')
    if _OWNER_OBSERVATION.get() is not None:
        raise ValueError('nested candidate owner observation is unsupported')
    from clay.layout.place.reference_quality_geometry import _OBSERVER
    if _OBSERVER.get() is not None:
        raise ValueError('candidate owner and quality observations cannot nest')
    state = dict(source_names=source_names, owner=_execution_owner(), active=True)
    token = _OWNER_OBSERVATION.set(state)
    issued = issued_snapshot = stage = None
    def observer(frame):
        nonlocal issued, issued_snapshot
        caller = inspect.currentframe().f_back
        try:
            actual = native._strip_solve_banded
            if caller.f_code is not actual.__code__ or caller.f_globals is not actual.__globals__:
                raise ValueError('candidate frame lacks actual native callable origin')
            if state.get('dims') != dict(frame['dims']):
                raise ValueError('candidate actual owner dimensions differ')
            partition = state['partition']
            inputs = native._native_fixed_outline_stage_inputs(
                frame=frame, build_policy=build_policy(frame, partition))
            issued = native._NativeStageAcquisitionStop(stage_inputs=inputs,
                upstream_trace={'scope': 'NOT_COUNTED_BY_ACQUISITION'})
            issued_snapshot = (_canonical(inputs), canonical_bytes(partition))
        except Exception as exc:
            raise CurrentQualityAcquisitionFailure(str(exc)) from exc
        finally:
            del caller
        raise issued
    try:
        try:
            with native_stage_observation_scope(observer, complete_master_only=True):
                invoke()
        except native._NativeStageAcquisitionStop as stopped:
            if stopped is not issued or not hasattr(stopped, 'local_execution'):
                raise CurrentQualityAcquisitionFailure('foreign native acquisition stop')
            stopped.require_clean_unwind()
            expected = dict(phase='CANONICAL_FRAME_BEFORE_LOCAL_PRIMARY_OR_MAIN_CP',
                local_solve_count_before_observer=0, observer_execution='NOT_INSTRUMENTED_BY_NATIVE_SEAM',
                upstream_execution='CALLER_TRACE_ONLY_NOT_COUNTED_HERE')
            if stopped.local_execution != expected or (_canonical(stopped.stage_inputs),
                    canonical_bytes(state['partition'])) != issued_snapshot:
                raise CurrentQualityAcquisitionFailure('candidate acquisition changed during unwind')
            stage = _OwnerStage(stopped, state['partition'])
        else:
            raise ValueError('native candidate stage was not acquired')
    finally:
        state['active'] = False
        _OWNER_OBSERVATION.reset(token)
    try:
        from clay.layout.place.collision_geometry_session import collision_query_scope
        with collision_query_scope():
            yield stage
    except BaseException as original:
        try:
            stage._check()
        except BaseException as ending:
            from clay.flow.control_stop import record_unwind_failure
            record_unwind_failure(original, ending)
        raise
    else:
        stage._check()
    finally:
        stage._active = False


def verify_final_owner_candidate(*, cell, compact, seed, outline):
    from clay.layout.place.final_placement_state import final_placement_state_payload, _matching_declared_source_index
    cert = compact.final_placement_certificate
    state = final_placement_state_payload(cert)
    blocks = {name: tuple(box) for name, box in cert.blocks}
    indexed = _matching_declared_source_index(cert, blocks)
    if indexed is None:
        raise ValueError('final source-to-master partition unavailable')
    logicals, _boxes = indexed
    source = tuple(sorted(device.instance_name for device in cell.devices))
    if tuple(sorted(logicals)) != source:
        raise ValueError('final source-to-master cover differs')
    owners = {name: [] for name in blocks}
    for logical, (master, _physical) in logicals.items():
        owners[master].append(logical)
    seed.require_owner_partition({name: tuple(sorted(names)) for name, names in owners.items()})
    seed.require_final_positions({name: box[:2] for name, box in blocks.items()})
    frame = state['bbox_nm']
    if (frame[2]-frame[0], frame[3]-frame[1]) != outline:
        raise ValueError('final candidate outline differs')
