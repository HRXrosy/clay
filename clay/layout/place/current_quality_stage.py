"""Observe a current native frame and build its complete model before solving.

The caller owns source/PDK admission and an explicit build policy. This scope
captures current materializer geometry and native operands; it grants no G4,
quality, signoff, or independent source authority. No solver globals are patched.
"""
from contextlib import contextmanager
import inspect

from clay.layout.place.generated_incumbent import canonical_bytes
from clay.layout.place.native_stage_observation import (
    _execution_owner, native_stage_observation_scope,
)
from clay.layout.place.reference_quality_geometry import observe_live_quality_geometry_for_source


class CurrentQualityAcquisitionFailure(BaseException):
    """Instrumentation failures must escape native heuristic fallbacks."""


class _ModelBuilt(BaseException):
    pass


def _canonical(value):
    from clay.layout.place_blocks_2d import _diagnostic_canonical_value
    return canonical_bytes(_diagnostic_canonical_value(value))


class _CurrentStage:
    def __init__(self, stopped, geometry):
        self.stage_inputs = stopped.stage_inputs
        self.geometry = geometry
        self.local_execution = dict(stopped.local_execution)
        self.owner_logicals = {name: tuple(sorted({row.logical_id for row in geometry.occurrences
                                if row.owner_id == name})) for name, _, _ in geometry.owners}
        self._snapshot = self._current_bytes()
        self._owner = _execution_owner()
        self._active = True

    def _current_bytes(self):
        return (_canonical(self.stage_inputs), canonical_bytes(self.geometry.canonical_data()),
                canonical_bytes(self.owner_logicals), canonical_bytes(self.local_execution))

    def _check(self):
        if not self._active:
            raise ValueError('current quality stage is closed')
        if self._owner != _execution_owner():
            raise ValueError('current quality stage belongs to a different invocation owner')
        if self._current_bytes() != self._snapshot:
            raise ValueError('current quality stage operands changed')

    def build_model(self):
        """Use the actual legalizer's context sink; stop before its first Solve.

        The returned model has the complete original constraints and assumptions.
        The existing context independently certifies an actual response later.
        """
        from clay.layout import place_blocks_2d as native
        self._check()
        captured = {}
        stop = _ModelBuilt()

        def inventory(value):
            if 'inventory' in captured:
                raise CurrentQualityAcquisitionFailure('native model repeated its inventory')
            captured['inventory'] = value

        def context(value):
            caller = inspect.currentframe().f_back
            producer = native._legalize_fixed_outline_seed_x
            try:
                if caller.f_code is not producer.__code__ or caller.f_globals is not producer.__globals__:
                    raise CurrentQualityAcquisitionFailure('model context lacks actual native callable origin')
            finally:
                del caller
            if 'inventory' not in captured or 'context' in captured:
                raise CurrentQualityAcquisitionFailure('native model omitted or repeated its context')
            captured['context'] = value
            raise stop

        try:
            native._legalize_fixed_outline_seed_x(self.stage_inputs['seed'], **self.stage_inputs['kwargs'],
                assumption_inventory_sink=inventory, production_evidence_context_sink=context)
        except _ModelBuilt as observed:
            if observed is not stop:
                raise CurrentQualityAcquisitionFailure('foreign model construction stop')
        else:
            raise CurrentQualityAcquisitionFailure('native model failed to stop before Solve')
        self._check()
        actual = captured['context']
        model = actual._model.Clone()
        if model.Validate():
            raise ValueError('current native model is invalid: ' + model.Validate())
        return actual, model, captured['inventory']


@contextmanager
def acquire_current_quality_stage(invoke, *, source_population, build_policy):
    """Acquire one actual native call for an explicitly bound complete source.

    ``build_policy`` is either the native policy mapping or a callable receiving
    the current frame and immutable geometry and returning that explicit mapping.
    Its seed origin and scope remain policy data, never an inferred physical fact.
    """
    from clay.layout import place_blocks_2d as native
    issued = None
    issued_snapshot = None
    stage = None
    with observe_live_quality_geometry_for_source(source_population) as measurement:
        def observer(frame):
            nonlocal issued, issued_snapshot
            caller = inspect.currentframe().f_back
            producer = native._strip_solve_banded
            try:
                if caller.f_code is not producer.__code__ or caller.f_globals is not producer.__globals__:
                    raise CurrentQualityAcquisitionFailure('frame lacks actual native callable origin')
                geometry = measurement.latest
                if dict((name, (w, h)) for name, w, h in geometry.owners) != dict(frame['dims']):
                    raise ValueError('current native dimensions differ from observed full source geometry')
                policy = build_policy(frame, geometry) if callable(build_policy) else build_policy
                inputs = native._native_fixed_outline_stage_inputs(frame=frame, build_policy=policy)
                issued = native._NativeStageAcquisitionStop(stage_inputs=inputs,
                    upstream_trace={'scope': 'NOT_COUNTED_BY_ACQUISITION'})
                issued_snapshot = (_canonical(inputs), canonical_bytes(geometry.canonical_data()))
            except Exception as error:
                raise CurrentQualityAcquisitionFailure(str(error)) from error
            finally:
                del caller
            raise issued

        try:
            with native_stage_observation_scope(observer, complete_master_only=True):
                invoke()
        except native._NativeStageAcquisitionStop as stopped:
            if stopped is not issued or not hasattr(stopped, 'local_execution'):
                raise CurrentQualityAcquisitionFailure('foreign or incomplete native acquisition stop')
            try:
                stopped.require_clean_unwind()
            except ValueError as error:
                raise CurrentQualityAcquisitionFailure('native acquisition unwind failed boundary checks') from error
            expected_record = dict(phase='CANONICAL_FRAME_BEFORE_LOCAL_PRIMARY_OR_MAIN_CP',
                local_solve_count_before_observer=0, observer_execution='NOT_INSTRUMENTED_BY_NATIVE_SEAM',
                upstream_execution='CALLER_TRACE_ONLY_NOT_COUNTED_HERE')
            if stopped.local_execution != expected_record:
                raise CurrentQualityAcquisitionFailure('native execution record changed during unwind')
            if (_canonical(stopped.stage_inputs), canonical_bytes(measurement.latest.canonical_data())) != issued_snapshot:
                raise CurrentQualityAcquisitionFailure('observed native inputs changed during unwind')
            stage = _CurrentStage(stopped, measurement.latest)
        else:
            raise ValueError('invocation returned without a native quality acquisition')
    try:
        from clay.layout.place.collision_geometry_session import collision_query_scope
        # Model construction and candidate revalidation recompute their models,
        # but can share exact derived unions while this stage owns the inputs.
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
