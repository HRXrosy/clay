"""Append complete reference arithmetic to current native floorplan solves.

Only the actual native callable can bind a stage. Evidence belongs to this
process/thread/task and lexical scope; saved artifacts cannot reconstruct it.
The producer still owns source, reference and structural authenticity. This
scope cannot grant strict admission, G4 or physical signoff.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import inspect
import json
import uuid

from ortools.sat.python import cp_model

from clay.layout.place import generated_cohesion as primitives
from clay.layout.place.final_quality_evaluation import project_final_quality_geometry
from clay.layout.place.generated_incumbent import canonical_bytes
from clay.layout.place.generated_reference_quality import _compile_quality_model
from clay.layout.place.native_stage_observation import _execution_owner
from clay.layout.place.production_model_evidence import _check_proto_solution
from clay.layout.place.quality_location_constraints import replay_location_search
from clay.layout.place.reference_quality_geometry import observe_live_quality_geometry_for_source
from clay.layout.reference_quality_contract import ReferenceQualityContract
from clay.layout.reference_quality_metrics import evaluate_reference_quality


_CURRENT = ContextVar('current_native_reference_quality', default=None)
_STRUCTURE_BLOCKERS = {'required_structure:fixed_artwork', 'current_structure:materialized_consumption'}


class NativeReferenceQualityFailure(BaseException):
    """A required quality failure must escape all heuristic seed fallbacks."""


def _require(ok, message):
    if not ok:
        raise NativeReferenceQualityFailure(message)


def _bytes(proto):
    return proto.SerializeToString(deterministic=True)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _native(frame, model):
    from clay.layout.place.cp_model_builder import _strip_solve_floorplan
    _require(frame.f_code is _strip_solve_floorplan.__code__
        and frame.f_globals is _strip_solve_floorplan.__globals__
        and frame.f_locals.get('model') is model,
        'current quality hook lacks actual native callable origin')


class _Scope:
    def __init__(self, contract, frame, measurement, sink):
        self.contract, self.frame, self.measurement = contract, frame, measurement
        self.sink = sink
        self.owner, self.invocation_id = _execution_owner(), uuid.uuid4().hex
        self._binding = (self.owner, self.invocation_id, self.frame)
        self._stage_binding = ()
        self.contract_bytes = canonical_bytes(contract.canonical_data())
        self.closed, self.stages, self.artifacts = False, [], {}
        self.emit('contract.json', self.contract_bytes)
        self.emit('scope.json', canonical_bytes(dict(schema='clay.native-reference-quality-scope/v1',
            scope='CURRENT_EXECUTION_ARITHMETIC_ONLY', strict_admission='NOT_ESTABLISHED',
            invocation_id=self.invocation_id, frame_nm=frame,
            limitation='Raw artifacts can survive failure; source and both structural authorities remain producer obligations.')))

    def _check(self):
        _require(not self.closed, 'current native quality scope is closed')
        _require(self.owner == _execution_owner(), 'current native quality invocation differs')
        _require(_CURRENT.get() is self, 'current native quality scope identity differs')
        _require((self.owner, self.invocation_id, self.frame) == self._binding,
                 'current native quality frame/context binding changed')
        _require(tuple(self.stages) == self._stage_binding,
                 'current native quality stage history binding changed')
        _require(canonical_bytes(self.contract.canonical_data()) == self.contract_bytes,
                 'current native quality contract changed during invocation')
        for stage in self.stages:
            stage._check_geometry()

    def emit(self, name, payload):
        _require(type(payload) is bytes and name not in self.artifacts, 'native quality artifact repeated or mutable')
        self.artifacts[name] = payload
        if self.sink is not None:
            self.sink(name, payload)

    def evidence(self):
        self._check()
        return json.loads(canonical_bytes(dict(schema='clay.native-reference-quality-execution/v1',
            strict_admission='NOT_ESTABLISHED', invocation_id=self.invocation_id,
            contract_sha256=_sha(self.contract_bytes), current_contract_sha256=self.contract.sha256,
            frame_nm=self.frame, stages=[stage.report for stage in self.stages],
            artifact_sha256={name: _sha(value) for name, value in self.artifacts.items()})))

    def require_final(self, observation):
        """Require exact final geometry/coordinates of the last successful call."""
        self._check()
        _require(self.stages and self.stages[-1].candidate is not None,
                 'current quality scope has no completed native candidate stage')
        stage = self.stages[-1]
        _require(stage.pending is None, 'current native quality stage has an incomplete actual Solve')
        try:
            geometry, positions = project_final_quality_geometry(observation)
            state = observation.canonical_data()['final_placement_state']
            _require(canonical_bytes(geometry.canonical_data()) == stage.geometry_bytes,
                     'final candidate geometry differs from the last native stage')
            _require(positions == stage.candidate, 'final candidate coordinates differ from the actual native response')
            _require(tuple(state['bbox_nm']) == self.frame, 'final candidate frame differs from explicit native frame')
            evaluation = evaluate_reference_quality(positions=positions, geometry=geometry,
                contract=self.contract, frame_nm=self.frame)
            _require(set(evaluation['strict_blockers']) == _STRUCTURE_BLOCKERS,
                     'complete final native quality arithmetic failed')
            _require(canonical_bytes(evaluation) == stage.candidate_evaluation,
                     'final candidate quality differs from the actual native response')
            self._check()
            return dict(schema='clay.native-reference-quality-final/v1',
                status='CURRENT_NATIVE_QUALITY_CONSISTENT', strict_admission='NOT_ESTABLISHED',
                invocation_id=self.invocation_id, stage_count=len(self.stages),
                logical_count=len(self.contract.logical_population), occurrence_count=len(geometry.occurrences),
                candidate_origin=stage.report['candidate_origin'],
                contract_sha256=self.contract.sha256, geometry_sha256=geometry.sha256,
                final_observation_sha256=observation.sha256,
                final_placement_state_sha256=observation.final_placement_state_sha256,
                evaluation=evaluation)
        except Exception as error:
            raise NativeReferenceQualityFailure('final native quality binding failed: ' + str(error)) from error


class _Stage:
    def __init__(self, scope, model, xs, ys, dims, width, height, caller):
        self.scope, self.model = scope, model
        self.geometry = scope.measurement.latest
        self.geometry_bytes = canonical_bytes(self.geometry.canonical_data())
        _require({name: (w, h) for name, w, h in self.geometry.owners} == dict(dims)
                 and set(xs) == set(ys) == set(dims),
                 'current native dimensions differ from full live source geometry')
        _require(all(caller.f_locals.get(name) is value for name, value in
            (('xs', xs), ('ys', ys), ('dims', dims), ('W', width), ('H', height))),
            'current quality coordinate operands differ from actual native locals')
        self.ids = tuple(caller.f_locals['ids'])
        self.indices = {name: (xs[name].Index(), ys[name].Index()) for name in sorted(dims)}
        self.primary_index = caller.f_locals['primary'].Index()
        self.original = model.Clone()
        original_bytes = _bytes(self.original.Proto())
        _require(not self.original.Proto().HasField('floating_point_objective'),
                 'native quality does not support a floating objective')
        self.prefix = f'stage-{len(scope.stages):03d}/'
        self.report = dict(stage=len(scope.stages), geometry_sha256=self.geometry.sha256,
            original_model_sha256=_sha(original_bytes), phases=[], candidate_origin=None,
            logical_count=len(scope.contract.logical_population), occurrence_count=len(self.geometry.occurrences),
            coordinate_indices=self.indices, native_callable='clay.layout.place.cp_model_builder._strip_solve_floorplan')
        self.pending, self.candidate, self.candidate_evaluation = None, None, None
        self.emit('original-model.pb', original_bytes)
        self.emit('geometry.json', self.geometry_bytes)
        augmented, *self.metadata = _compile_quality_model(domain_bytes=original_bytes,
            indices=self.indices, geometry=self.geometry, contract=scope.contract, frame=scope.frame)
        # The compiler's feasibility implementation clears objectives. Restore
        # the exact native objective; only the native soft phase may replace it.
        augmented.Proto().objective.CopyFrom(self.original.Proto().objective)
        for axis, variables in enumerate((xs, ys)):
            starts = [augmented.GetIntVarFromProtoIndex(self.indices[name][axis]) for name in self.ids]
            ends = [variable + dims[name][axis] for name, variable in zip(self.ids, starts)]
            augmented.AddMinEquality(scope.frame[axis], starts).WithName(f'current_quality_frame:min:{axis}')
            augmented.AddMaxEquality(scope.frame[axis+2], ends).WithName(f'current_quality_frame:max:{axis}')
        # Native W/H denote absolute maximum endpoints, including nonzero-origin
        # frames. Their spans are computed from the independent measured minima.
        augmented.Add(augmented.GetIntVarFromProtoIndex(width.Index()) == scope.frame[2])
        augmented.Add(augmented.GetIntVarFromProtoIndex(height.Index()) == scope.frame[3])
        primitives._require_prefix(self.original.Proto(), augmented.Proto())
        _require(_bytes(augmented.Proto().objective) == _bytes(self.original.Proto().objective),
                 'quality compilation changed the native objective')
        _require(not augmented.Validate(), 'invalid augmented native quality model: ' + augmented.Validate())
        self.augmented = augmented.Clone()
        self.augmented_bytes = _bytes(augmented.Proto())
        self.report['augmented_model_sha256'] = _sha(self.augmented_bytes)
        self.emit('augmented-model.pb', self.augmented_bytes)
        self.emit('quality-compilation.json', canonical_bytes(self.metadata))
        soft_model = augmented.Clone()
        soft_model.Minimize(sum(caller.f_locals['soft']))
        self.soft_objective = _bytes(soft_model.Proto().objective)
        model.Proto().CopyFrom(augmented.Proto())
        self._seal_state()

    def emit(self, name, payload):
        self.scope.emit(self.prefix + name, payload)

    def _check_geometry(self):
        _require(canonical_bytes(self.geometry.canonical_data()) == self.geometry_bytes,
                 'current native source geometry changed during invocation')
        _require(self._state_bytes() == self._state_binding,
                 'current native candidate/history binding changed')

    def _state_bytes(self):
        return (canonical_bytes(self.report), canonical_bytes(self.candidate), self.candidate_evaluation,
                _bytes(self.original.Proto()), _bytes(self.augmented.Proto()))

    def _seal_state(self):
        self._state_binding = self._state_bytes()

    def _check_caller(self, caller, model):
        _native(caller, model)
        self.scope._check()
        _require(model is self.model and caller.f_locals.get('quality_stage') is self,
                 'actual native quality stage/model identity differs')

    def before_solve(self, model, solver, *, phase):
        caller = inspect.currentframe().f_back
        try:
            self._check_caller(caller, model)
            _require(self.pending is None, 'native quality Solve was started twice')
            if phase == 'primary':
                _require(not self.report['phases'], 'native primary Solve repeated')
                expected = self.augmented
            else:
                _require(phase == 'soft_position' and len(self.report['phases']) == 1
                    and self.candidate is not None, 'native soft Solve lacks its primary incumbent')
                expected = self.augmented.Clone()
                expected.Add(expected.GetIntVarFromProtoIndex(self.primary_index) <= self.primary_value)
                expected.Proto().ClearField('solution_hint')
                for name in self.ids:
                    for index, value in zip(self.indices[name], self.candidate[name]):
                        expected.AddHint(expected.GetIntVarFromProtoIndex(index), value)
                expected.Proto().objective.ParseFromString(self.soft_objective)
            actual = _bytes(model.Proto())
            _require(actual == _bytes(expected.Proto()),
                     'native phase changed constraints, domains, hints or an undeclared objective')
            controls = _bytes(solver.parameters)
            self.pending = (phase, actual, controls, solver)
            self.emit(phase + '-model.pb', actual)
            self.emit(phase + '-parameters.pb', controls)
        except Exception as error:
            raise NativeReferenceQualityFailure('native quality before Solve failed: ' + str(error)) from error
        finally:
            del caller

    def after_solve(self, model, solver, status, *, phase):
        caller = inspect.currentframe().f_back
        try:
            self._check_caller(caller, model)
            _require(self.pending is not None, 'native quality response has no actual pending Solve')
            expected_phase, before_model, before_controls, actual_solver = self.pending
            response = solver.ResponseProto()
            self.emit(phase + '-response.pb', _bytes(response))
            _require(expected_phase == phase and actual_solver is solver and response.status == status
                and _bytes(model.Proto()) == before_model and _bytes(solver.parameters) == before_controls,
                'native quality model, controls or response changed across actual Solve')
            self.pending = None
            record = dict(phase=phase, status='NO_CANDIDATE', solver_status=solver.StatusName(status),
                model_sha256=_sha(before_model), model_after_sha256=_sha(_bytes(model.Proto())),
                parameters_sha256=_sha(before_controls), response_sha256=_sha(_bytes(response)),
                primitive_status='NOT_CHECKED', deterministic_time=response.deterministic_time,
                wall_time=response.wall_time, geometry_after_sha256=self.geometry.sha256,
                contract_after_sha256=self.scope.contract.sha256)
            self.report['phases'].append(record)
            self._seal_state()
            if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                if phase == 'soft_position':
                    self.report['candidate_origin'] = 'verified_primary_incumbent_after_soft_no_candidate'
                    self._seal_state()
                self.emit(phase + '-checks.json', canonical_bytes(record))
                if phase == 'primary':
                    raise NativeReferenceQualityFailure('NO_CANDIDATE in explicit native reference quality frame')
                return
            solution = tuple(response.solution)
            _require(len(solution) == len(model.Proto().variables), 'native response omits complete model variables')
            original_checks = _check_proto_solution(self.original.Proto(), solution[:len(self.original.Proto().variables)])
            complete_checks = _check_proto_solution(model.Proto(), solution)
            _require(primitives._checks_status(original_checks) == primitives._checks_status(complete_checks) == 'PASS',
                     'native response fails original or complete augmented primitive checks')
            positions = {name: tuple(solution[index] for index in pair) for name, pair in self.indices.items()}
            evaluation = evaluate_reference_quality(positions=positions, geometry=self.geometry,
                contract=self.scope.contract, frame_nm=self.scope.frame)
            _require(set(evaluation['strict_blockers']) == _STRUCTURE_BLOCKERS,
                     'native response fails complete quality arithmetic: ' + repr(evaluation['strict_blockers']))
            replay_location_search(evaluation, self.metadata[0], self.metadata[4], solution, self.scope.contract)
            self.candidate, self.candidate_evaluation = positions, canonical_bytes(evaluation)
            self.primary_value = solution[self.primary_index]
            record.update(status='QUALITY_FEASIBLE', primitive_status='PASS', positions_nm=positions,
                evaluation=evaluation, original_checks=original_checks, complete_checks=complete_checks)
            self.report['candidate_origin'] = 'actual_' + phase + '_response'
            self._seal_state()
            self.emit(phase + '-checks.json', canonical_bytes(record))
            self.scope._check()
        except Exception as error:
            raise NativeReferenceQualityFailure('native quality after Solve failed: ' + str(error)) from error
        finally:
            del caller


def apply_current_floorplan_quality(model, xs, ys, dims, W, H):
    """Actual-native-only augmentation; an inactive scope changes nothing."""
    scope = _CURRENT.get()
    if scope is None:
        return None
    caller = inspect.currentframe().f_back
    try:
        _native(caller, model)
        scope._check()
        stage = _Stage(scope, model, xs, ys, dims, W, H, caller)
        scope.stages.append(stage)
        scope._stage_binding = tuple(scope.stages)
        return stage
    except Exception as error:
        raise NativeReferenceQualityFailure('native quality compilation failed: ' + str(error)) from error
    finally:
        del caller


@contextmanager
def constrain_current_reference_quality(*, contract, frame_nm, artifact_sink=None):
    """Constrain every current complete floorplan frame; never accept saved proof."""
    _require(_CURRENT.get() is None, 'nested current native quality scopes are unsupported')
    _require(type(contract) is ReferenceQualityContract, 'current quality requires an exact reference contract')
    _require(type(frame_nm) is tuple and len(frame_nm) == 4 and all(type(v) is int for v in frame_nm)
        and all(0 < frame_nm[axis+2]-frame_nm[axis] <= 1_000_000 for axis in (0, 1)),
        'current quality requires an explicit bounded integer frame')
    _require(artifact_sink is None or callable(artifact_sink), 'native quality artifact sink must be callable')
    population = tuple((row.logical_id, len(row.occurrence_ids)) for row in contract.logical_population)
    with observe_live_quality_geometry_for_source(population) as measurement:
        scope = _Scope(contract, frame_nm, measurement, artifact_sink)
        token = _CURRENT.set(scope)
        try:
            yield scope
        except BaseException as original:
            try:
                scope._check()
            except BaseException as ending:
                from clay.flow.control_stop import record_unwind_failure
                record_unwind_failure(original, ending)
            raise
        else:
            scope._check()
        finally:
            scope.closed = True
            _CURRENT.reset(token)
