"""Solve complete arithmetic quality in one explicit current native frame slice.

There is no prior incumbent, derived group-span cap or Q optimization. A found
point is separately solved at fixed coordinates in the complete original CP.
The host must then retain it through actual materialization and authenticate
both structural obligations. Slice UNSAT is not global physical infeasibility.
"""
from contextlib import nullcontext
import hashlib
import json
import math

from ortools.sat import cp_model_pb2, sat_parameters_pb2
from ortools.sat.python import cp_model

from clay.flow.execution_session import current_session
from clay.layout.place.current_quality_stage import _CurrentStage
from clay.layout.place.current_outer_domain import append_current_outer_domain
from clay.layout.place.generated_incumbent import (
    GeneratedIncumbentSeed, SCHEMA, SCOPE, canonical_bytes, digest, frame_data,
)
from clay.layout.place import generated_cohesion as primitives
from clay.layout.place.generated_reference_quality import _compile_quality_model
from clay.layout.place.production_model_evidence import _check_proto_solution
from clay.layout.place.placement_execution import cp_solve_phase
from clay.layout.place.quality_location_constraints import replay_location_search
from clay.layout.place.solver_execution_policy import prepare_incumbent_replay_model, require_incumbent_replay_response
from clay.layout.reference_quality_metrics import evaluate_reference_quality


_MINT = object()
_STRUCTURE_BLOCKERS = ['required_structure:fixed_artwork', 'current_structure:materialized_consumption']


def _bytes(value):
    return value.SerializeToString(deterministic=True)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _phase(name):
    session = current_session()
    return nullcontext() if session is None else session.phase(name)


def _compile(stage):
    stage._check()
    context, original, inventory = stage.build_model()
    policy, frame = stage.stage_inputs['build_policy'], stage.stage_inputs['frame']
    width, height = policy['width_nm'], policy['height_nm']
    _require(type(width) is int and type(height) is int and 0 < width <= 1_000_000
             and 0 < height <= 1_000_000, 'quality slice needs an explicit positive bounded integer frame')
    _require(set(policy['movable_y']) == set(frame['dims']), 'quality slice must explicitly permit every native Y owner')
    for owner, (w, h) in frame['dims'].items():
        x, y = policy['seed'][owner]
        _require(0 <= x <= width-w and 0 <= y <= height-h,
                 'quality seed lies outside the explicit native coordinate domain')
    _require(not original.Proto().objective.domain and not original.Proto().HasField('floating_point_objective'),
             'quality search cannot remove a hard or floating objective')
    return context, original, inventory, (0, 0, width, height)


def _quality_model(original, indices, geometry, contract, frame):
    compiled = _compile_quality_model(domain_bytes=_bytes(original.Proto()), indices=indices,
        geometry=geometry, contract=contract, frame=frame)
    model, *metadata = compiled
    model.ClearObjective()
    dims = {name: (w, h) for name, w, h in geometry.owners}
    for axis in (0, 1):
        starts = [model.GetIntVarFromProtoIndex(indices[name][axis]) for name in sorted(dims)]
        ends = [model.GetIntVarFromProtoIndex(indices[name][axis]) + dims[name][axis] for name in sorted(dims)]
        model.AddMinEquality(frame[axis], starts).WithName(f'current_quality_frame:min:{axis}')
        model.AddMaxEquality(frame[axis+2], ends).WithName(f'current_quality_frame:max:{axis}')
    primitives._require_prefix(original.Proto(), model.Proto())
    _require(not model.Validate(), 'invalid complete quality model: ' + model.Validate())
    return model, metadata


def _compile_search(stage, original, indices, contract, frame, *, emit=None):
    outer, outer_metadata = append_current_outer_domain(original,
        coordinate_indices=indices, outer_kwargs=stage.stage_inputs['outer_kwargs'])
    outer_bytes, metadata_bytes = _bytes(outer.Proto()), canonical_bytes(outer_metadata)
    if emit is not None:
        emit('outer-domain-model.pb', outer_bytes)
        emit('outer-domain-compilation.json', metadata_bytes)
    model, metadata = _quality_model(outer, indices, stage.geometry, contract, frame)
    identity = dict(scope=outer_metadata['scope'], model_sha256=_sha(outer_bytes),
        original_model_sha256=outer_metadata['original_model_sha256'],
        compilation_sha256=_sha(metadata_bytes), outer_input_sha256=outer_metadata['outer_input_sha256'])
    return model, metadata, outer_bytes, metadata_bytes, identity


def _parameters(deterministic, wall, *, fixed=False):
    parameters = sat_parameters_pb2.SatParameters()
    parameters.num_search_workers = 1
    parameters.random_seed = 0
    parameters.max_deterministic_time = deterministic
    parameters.max_time_in_seconds = wall
    parameters.stop_after_first_solution = True
    parameters.fix_variables_to_their_hinted_value = fixed
    return parameters


def _solve(model, parameters, *, emit, prefix, phase):
    original, controls = _bytes(model.Proto()), _bytes(parameters)
    emit(prefix + '-model.pb', original)
    emit(prefix + '-requested-parameters.pb', controls)
    solver = cp_model.CpSolver()
    solver.parameters.CopyFrom(parameters)
    with cp_solve_phase(solver, model, phase=phase):
        effective = _bytes(solver.parameters)
        emit(prefix + '-parameters.pb', effective)
        status = solver.Solve(model)
        # Persist even when the following budget charge/checkpoint stops execution.
        emit(prefix + '-response.pb', _bytes(solver.ResponseProto()))
        _require(_bytes(model.Proto()) == original and _bytes(solver.parameters) == effective
                 and solver.ResponseProto().status == status, 'actual quality solver changed model/controls/status')
    _require(_bytes(model.Proto()) == original and _bytes(solver.parameters) == controls
             and solver.ResponseProto().status == status, 'actual quality solver changed model/controls/status')
    return solver, status


def _measure(original, model, response, *, indices, geometry, contract, frame, metadata):
    solution = tuple(response.solution)
    _require(response.status in (cp_model.FEASIBLE, cp_model.OPTIMAL)
             and len(solution) == len(model.Proto().variables), 'quality response has no complete actual solution')
    primitives._require_prefix(original.Proto(), model.Proto())
    checks = _check_proto_solution(original.Proto(), solution[:len(original.Proto().variables)])
    complete = _check_proto_solution(model.Proto(), solution)
    _require(primitives._checks_status(checks) == primitives._checks_status(complete) == 'PASS',
             'quality point fails original or complete augmented primitive checks')
    positions = {name: tuple(solution[index] for index in pair) for name, pair in indices.items()}
    evaluation = evaluate_reference_quality(positions=positions, geometry=geometry, contract=contract, frame_nm=frame)
    _require(evaluation['strict_blockers'] == _STRUCTURE_BLOCKERS,
             'complete arithmetic quality failed: ' + repr(evaluation['strict_blockers']))
    replay_location_search(evaluation, metadata[0], metadata[4], solution, contract)
    return positions, evaluation, dict(original=checks, complete_augmented=complete)


def _outer(stage, positions, *, emit=None):
    from clay.layout import place_blocks_2d as native
    failed, inventory, spec, blockers = [], [], {}, []
    kwargs = dict(stage.stage_inputs['outer_kwargs'])
    before = canonical_bytes(native._diagnostic_canonical_value(kwargs))
    passed = native._banded_seed_satisfies_hard_constraints(positions, **kwargs,
        blocker_out=blockers, diagnostic_out=failed, hard_inventory_out=inventory, hard_spec_out=spec)
    _require(canonical_bytes(native._diagnostic_canonical_value(kwargs)) == before, 'native outer replay changed its inputs')
    accepted = passed is True and not failed and not blockers
    evidence = dict(status='PASS' if accepted else 'FAIL', failed_rows=failed, blockers=blockers,
        inventory=native._diagnostic_canonical_value(inventory), spec=native._diagnostic_canonical_value(spec))
    if emit is not None:
        emit('original-outer-replay.json', canonical_bytes(evidence))
    _require(accepted, 'quality point fails complete native outer replay: ' + repr((failed, blockers)))
    return evidence


class CurrentQualityFeasibility:
    """Same-invocation execution evidence; no reconstruction from saved JSON."""
    __slots__ = ('_mint', '_stage', '_contract', '_contract_bytes', '_report', '_artifacts')

    def __new__(cls, *args, **kwargs):
        raise TypeError('current quality evidence requires actual execution')

    def __setattr__(self, key, value):
        raise TypeError('current quality execution is immutable')

    def __reduce_ex__(self, protocol):
        raise TypeError('current quality execution cannot be reconstructed')

    def _check(self):
        _require(type(self) is CurrentQualityFeasibility and self._mint is _MINT, 'current quality execution origin differs')
        self._stage._check()
        _require(canonical_bytes(self._contract.canonical_data()) == self._contract_bytes, 'current quality contract changed')

    def evidence(self):
        self._check()
        return json.loads(self._report)

    def artifacts(self):
        self._check()
        return dict(self._artifacts)

    def require_candidate(self):
        self._check()
        report, artifacts = self.evidence(), self.artifacts()
        _require(report['status'] == 'QUALITY_FEASIBLE', 'current quality execution has no certified candidate')
        context, original, inventory, frame = _compile(self._stage)
        model, metadata, outer_bytes, outer_metadata_bytes, outer_identity = _compile_search(
            self._stage, original, context._indices, self._contract, frame)
        _require(_bytes(original.Proto()) == artifacts.get('original-model.pb')
                 and outer_bytes == artifacts.get('outer-domain-model.pb')
                 and outer_metadata_bytes == artifacts.get('outer-domain-compilation.json')
                 and _bytes(model.Proto()) == artifacts.get('quality-model.pb')
                 and canonical_bytes(metadata) == artifacts.get('quality-compilation.json')
                 and canonical_bytes(outer_identity) == canonical_bytes(report.get('outer_domain')),
                 'current native, outer or quality compilation differs from actual execution')
        response = cp_model_pb2.CpSolverResponse()
        response.ParseFromString(artifacts['quality-response.pb'])
        positions, evaluation, _ = _measure(original, model, response, indices=context._indices,
            geometry=self._stage.geometry, contract=self._contract, frame=frame, metadata=metadata)
        variables = {name: tuple(original.GetIntVarFromProtoIndex(i) for i in pair) for name, pair in context._indices.items()}
        cert = prepare_incumbent_replay_model(original, coordinate_variables=variables, positions=positions)
        _require(_bytes(cert.Proto()) == artifacts['cert-model.pb'], 'original fixed-coordinate certification model differs')
        cert_response = cp_model_pb2.CpSolverResponse()
        cert_response.ParseFromString(artifacts['cert-response.pb'])
        from types import SimpleNamespace
        recorded = SimpleNamespace(ResponseProto=lambda: cert_response, StatusName=cp_model.CpSolver().StatusName)
        require_incumbent_replay_response(cert, recorded, cert_response.status,
            positions=positions, coordinate_variables=variables)
        gate = context.build_receipt(model=cert, solver=recorded, solver_status=cert_response.status,
            incumbent_replay_positions=positions)
        outer = _outer(self._stage, positions)
        _require(canonical_bytes(gate) == canonical_bytes(report['certification'])
                 and canonical_bytes(evaluation) == canonical_bytes(report['evaluation'])
                 and canonical_bytes(outer) == canonical_bytes(report['outer_replay']), 'current native/quality replay differs')
        return GeneratedIncumbentSeed(dict(schema=SCHEMA, scope=SCOPE, promotion='REJECT', g4_status='NOT_RUN',
            positions_nm=sorted(positions.items()), owner_logicals=sorted(self._stage.owner_logicals.items()),
            native_frame=frame_data(self._stage.stage_inputs['frame']), production_evidence=dict(
                model_proto_sha256=_sha(artifacts['cert-model.pb']),
                promotion_gate_sha256=digest(gate), outer_replay_sha256=digest(outer))))


def execute_current_quality_feasibility(stage, *, contract, max_deterministic_time,
                                        max_wall_time_s, certification_wall_time_s, artifact_sink=None):
    _require(type(stage) is _CurrentStage, 'quality execution needs a current native acquisition')
    for value in (max_deterministic_time, max_wall_time_s, certification_wall_time_s):
        _require(type(value) in (int, float) and math.isfinite(value) and value > 0, 'quality execution needs finite positive budgets')
    stage._check()
    target_bytes = canonical_bytes(contract.canonical_data())
    artifacts = {}
    def emit(name, payload):
        _require(type(payload) is bytes, 'execution artifacts must be immutable bytes')
        if name in artifacts:
            _require(artifacts[name] == payload, 'actual execution changed an already emitted artifact')
            return
        artifacts[name] = payload
        if artifact_sink is not None:
            artifact_sink(name, payload)
    emit('artifact-scope.json', canonical_bytes(dict(scope='RAW_UNVERIFIED_NOT_ADMISSION',
        limitation='Raw artifacts may survive compile, solver, budget or validation failure; only completed live checks issue a candidate.')))
    emit('contract.json', target_bytes)
    emit('geometry.json', canonical_bytes(stage.geometry.canonical_data()))
    with _phase('placement.quality.complete_native_model'):
        context, original, inventory, frame = _compile(stage)
        emit('original-model.pb', _bytes(original.Proto()))
        model, metadata, _outer_bytes, _outer_metadata, outer_identity = _compile_search(
            stage, original, context._indices, contract, frame, emit=emit)
        emit('quality-model.pb', _bytes(model.Proto()))
        emit('quality-compilation.json', canonical_bytes(metadata))
    parameters = _parameters(max_deterministic_time, max_wall_time_s)
    solver, status = _solve(model, parameters, emit=emit, prefix='quality',
        phase='placement.quality.complete_feasibility_solve')
    report = dict(schema='clay.current-quality-feasibility/v1', status='NO_CANDIDATE',
        solver_status=solver.StatusName(status), search_scope='EXPLICIT_OUTLINE_SLICE_AND_COMPLETE_QUALITY_SUFFICIENT_DOMAIN',
        seed_span_caps_added=False, q_optimization='NOT_RUN', frame_nm=list(frame),
        limitation='Failure is limited to this explicit frame and centroid inner domain; source and structural authority remain independent.',
        native_inventory=inventory, contract_sha256=contract.sha256, geometry_sha256=stage.geometry.sha256,
        outer_domain=outer_identity)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        positions, evaluation, checks = _measure(original, model, solver.ResponseProto(), indices=context._indices,
            geometry=stage.geometry, contract=contract, frame=frame, metadata=metadata)
        variables = {name: tuple(original.GetIntVarFromProtoIndex(i) for i in pair) for name, pair in context._indices.items()}
        cert = prepare_incumbent_replay_model(original, coordinate_variables=variables, positions=positions)
        cert_solver, cert_status = _solve(cert, _parameters(max_deterministic_time, certification_wall_time_s, fixed=True),
            emit=emit, prefix='cert', phase='placement.quality.original_fixed_candidate_certification')
        report.update(status='UNCERTIFIED_QUALITY_CANDIDATE', evaluation=evaluation,
                      primitive_checks=checks, positions_nm=positions,
                      certification_solver_status=cert_solver.StatusName(cert_status))
        if cert_status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            require_incumbent_replay_response(cert, cert_solver, cert_status, positions=positions, coordinate_variables=variables)
            gate = context.build_receipt(model=cert, solver=cert_solver, solver_status=cert_status, incumbent_replay_positions=positions)
            _require(gate['status'] == 'PROMOTABLE', 'original native fixed-candidate certification failed')
            emit('original-certification.json', canonical_bytes(gate))
            report.update(status='QUALITY_FEASIBLE', certification=gate, outer_replay=_outer(stage, positions, emit=emit))
    stage._check()
    _require(canonical_bytes(contract.canonical_data()) == target_bytes, 'quality contract changed during execution')
    result = object.__new__(CurrentQualityFeasibility)
    for name, value in dict(_mint=_MINT, _stage=stage, _contract=contract, _contract_bytes=target_bytes,
                            _report=canonical_bytes(report), _artifacts=tuple(sorted(artifacts.items()))).items():
        object.__setattr__(result, name, value)
    return result
