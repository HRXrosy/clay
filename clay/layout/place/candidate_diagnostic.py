"""Same-invocation fixed-candidate diagnostic at the public layout caller.

This evaluates one untrusted vector within the standard caller's actual
materialization scope and terminates at its diagnostic routing observer.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import hashlib, inspect, json, math, re
from pathlib import Path

from ortools.sat.python import cp_model
from clay.flow.execution_session import current_session
from clay.flow.run_admission import current_admission
from clay.frontend.source_revision import reopen_complete_source
from clay.layout.place.current_outer_domain import append_current_outer_domain
from clay.layout.place.current_quality_feasibility import _compile, _outer
from clay.layout.place.current_quality_stage import acquire_current_quality_stage, CurrentQualityAcquisitionFailure
from clay.layout.place.current_owner_stage import acquire_current_owner_stage, verify_final_owner_candidate
from clay.layout.place.compact_snapshot import compact_value as _compact_value
from clay.layout.place.current_structure_operands import _placed_values, _row_values, _source_population
from clay.layout.place.final_placement_state import final_placement_state_payload
from clay.layout.place.final_quality_evaluation import project_final_quality_geometry
from clay.layout.place.final_quality_observation import observe_materialized_source
from clay.layout.place.generated_incumbent import GeneratedIncumbentSeed, SCHEMA, SCOPE, digest, frame_data
from clay.layout.place.placement_execution import cp_solve_phase
from clay.layout.place.row_replay_snapshot import freeze_row_values
from clay.layout.place.solver_execution_policy import (prepare_incumbent_replay_model,
    require_incumbent_replay_response, scoped_incumbent_replay)
from clay.layout.policy import policy_for_cell
from clay.layout.route.canonical_input_capture import route_engine_configuration
from clay.layout.route.engine_registry import route_engine_from_policy
from clay.layout.route.resource_world import pdk_provenance_digest
from clay.layout.tech import LayoutTech
from clay.parser import parse_clay_file

def _positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0

@dataclass(frozen=True, slots=True)
class CandidateDiagnosticRequest:
    """Untrusted single proposal; no physical or source authority is implied."""
    task_id: str
    source_sha256: str
    positions_nm: tuple
    outline_nm: tuple
    model_wall_seconds: float
    verify_wall_seconds: float
    verify_deterministic_seconds: float

    def __post_init__(self):
        if type(self.task_id) is not str or not self.task_id:
            raise ValueError('candidate task identity is required')
        if (type(self.source_sha256) is not str or len(self.source_sha256)!=64 or
                any(c not in '0123456789abcdef' for c in self.source_sha256)):
            raise ValueError('candidate source SHA-256 is required')
        if (type(self.positions_nm) is not tuple or not 1<=len(self.positions_nm)<=4096 or
            any(type(row) is not tuple or len(row)!=2 or type(row[0]) is not str or not row[0] or
                type(row[1]) is not tuple or len(row[1])!=2 or
                any(type(v) is not int or v<0 for v in row[1]) for row in self.positions_nm)):
            raise ValueError('candidate positions require exact integer owner pairs')
        names=tuple(row[0] for row in self.positions_nm)
        if names!=tuple(sorted(set(names))):raise ValueError('candidate owners must be unique and sorted')
        if (type(self.outline_nm) is not tuple or len(self.outline_nm)!=2 or
            any(type(n) is not int or not 0<n<=1_000_000 for n in self.outline_nm)):
            raise ValueError('candidate outline must contain two positive bounded integers')
        if not all(_positive(v) for v in (self.model_wall_seconds,self.verify_wall_seconds,
                                          self.verify_deterministic_seconds)):
            raise ValueError('candidate budgets must be finite and positive')

class CandidateDiagnosticUnavailable(RuntimeError):
    def flow_error_data(self):return {'reason':'CANDIDATE_DIAGNOSTIC_UNAVAILABLE','detail':str(self)}

class CandidateAcquisitionError(ValueError):
    def __init__(self, original):
        self.original=original
        super().__init__('candidate acquisition failed: '+str(original))
    def flow_error_data(self):
        cause=self.original.__cause__
        return {'reason':'CANDIDATE_ACQUISITION_FAILED','acquisition_type':type(self.original).__name__,
                'acquisition_detail':str(self.original)[:1024],
                'underlying_type':None if cause is None else type(cause).__name__,
                'underlying_detail':None if cause is None else str(cause)[:1024]}

class CandidateBoundaryChanged(RuntimeError):
    """The terminal observation could not retain its checked caller scope."""
    def __init__(self, original):
        self.original=original
        super().__init__('candidate boundary changed after observation: '+str(original))
    def flow_error_data(self):
        return {'reason':'CANDIDATE_BOUNDARY_CHANGED','detail':str(self.original)[:1024],
                'boundary_error_type':type(self.original).__name__}

def _digest(proto):return hashlib.sha256(proto.SerializeToString(deterministic=True)).hexdigest()

def _installed_pdk_identity(tech):
    """Independently re-read the installed source and compare the active cache."""
    from clay.pdk._source_snapshot import capture_pdk_source_snapshot
    adapter=tech.adapter
    root=adapter.root()
    if root is None:
        raise CandidateDiagnosticUnavailable('installed PDK source authority is unavailable')
    # Do not call source_snapshot(): it can replace the authority used by the
    # live placement when its root changes. Read the existing cache by identity.
    current=getattr(adapter,'_source_snapshot_cache',None)
    cache_root=getattr(adapter,'_source_snapshot_cache_root',None)
    if current is None:
        raise CandidateDiagnosticUnavailable('installed PDK cached source authority is unavailable')
    if cache_root!=root:
        raise ValueError('installed PDK root differs from active layout authority')
    fresh=capture_pdk_source_snapshot(adapter,root)
    if (adapter.root()!=root or getattr(adapter,'_source_snapshot_cache',None) is not current or
            getattr(adapter,'_source_snapshot_cache_root',None)!=cache_root):
        raise ValueError('installed PDK source authority changed during capture')
    for name in ('snapshot_sha256','installed_authority_sha256'):
        active=getattr(current,name,None); observed=getattr(fresh,name,None)
        if (type(active) is not str or len(active)!=64 or any(c not in '0123456789abcdef' for c in active) or
            type(observed) is not str or len(observed)!=64 or any(c not in '0123456789abcdef' for c in observed)):
            raise CandidateDiagnosticUnavailable('installed PDK '+name+' is missing or invalid')
        if active!=observed:
            raise ValueError('installed PDK '+name+' differs from active layout authority')
    return {'kind':'installed_pdk_source','root':str(root),
            'source_snapshot_sha256':fresh.snapshot_sha256,
            'installed_authority_sha256':fresh.installed_authority_sha256}

def _native_pdk_identity(*,spec,ctx,cell,placement,bundle):
    """Replay the actual public native source/bundle/runtime contract."""
    from clay.flow.native_physical import native_cell,native_runtime
    from clay.layout.native_sky130 import NativeGeometryBundle,_cell_native_specs
    from clay.verification.native_sky130 import _runtime_expectations,_validate_toolchain
    if (type(bundle) is not NativeGeometryBundle or ctx.extra.get('native_geometry') is not bundle or
            getattr(placement,'native_geometry',None) is not bundle):
        raise CandidateDiagnosticUnavailable('current native source-owned geometry bundle is unavailable')
    actual=native_cell(spec,ctx)
    if actual is None:raise CandidateDiagnosticUnavailable('current native source is unavailable')
    source=reopen_complete_source(cell); current=reopen_complete_source(actual)
    if source['source_sha256']!=current['source_sha256'] or source['source_inventory_sha256']!=current['source_inventory_sha256']:
        raise ValueError('native candidate source differs from current public source')
    runtime=native_runtime(spec)
    image=runtime['image'];pdk_root=runtime['pdk_root']
    if type(image) is not str or re.fullmatch(r'sha256:[0-9a-f]{64}',image) is None:
        raise ValueError('native candidate runtime image is not pinned')
    bundle.require_current(cell)
    output=Path(bundle.output_dir).resolve(strict=True)
    command=json.loads((output/'command.json').read_text(encoding='utf8'))
    if (type(command) is not list or len(command)!=22 or type(command[6]) is not str or
        re.fullmatch(r'clay-native-geometry-[0-9a-f]{12}',command[6]) is None):
        raise ValueError('native candidate materialization command identity differs')
    mount_origin=getattr(bundle,'_cache_origin_output_dir',None)
    mount_source=mount_origin if (type(mount_origin) is str and mount_origin) else output.as_posix()
    expected=['docker','run','--rm','--network','none','--name',command[6],
        '-v',f'{mount_source}:/work','-w','/work','-e','PDK=sky130A',
        '-e',f'PDK_ROOT={pdk_root}','-e',f'PDKPATH={pdk_root}/sky130A',
        '--entrypoint','bash',image,'-lc','exec python3 /work/generate.py']
    if command!=expected:raise ValueError('native candidate runtime image or PDK command differs')
    specs=_cell_native_specs(cell)
    contracts={'models':{entry['model']:entry['contract'] for entry in specs.values()}}
    expected_files=_runtime_expectations(runtime['files'],pdk_root)
    _validate_toolchain(output,contracts,pdk_root,expected_files)
    # The producer seals these files only after validating the real pinned run.
    # Recheck source, manifest and all producer artifacts after the replay.
    bundle.require_current(cell)
    if native_runtime(spec)!=runtime or reopen_complete_source(cell)['source_sha256']!=source['source_sha256']:
        raise ValueError('native candidate runtime or source changed during validation')
    return {'kind':'source_bound_native_runtime','source_sha256':source['source_sha256'],
        'runtime_manifest_sha256':spec.native_runtime_sha256,'image':image,'pdk_root':pdk_root,
        'expected_runtime_files_sha256':hashlib.sha256(json.dumps(expected_files,sort_keys=True,
            separators=(',',':')).encode('utf8')).hexdigest(),
        'toolchain_sha256':hashlib.sha256((output/'toolchain.json').read_bytes()).hexdigest(),
        'geometry_sha256':hashlib.sha256((output/'geometry.json').read_bytes()).hexdigest(),
        'command_sha256':hashlib.sha256((output/'command.json').read_bytes()).hexdigest()}

@contextmanager
def _checked_unwind(check, callback_returned):
    try:
        yield
    except BaseException as original:
        try:
            check()
        except BaseException as ending:
            from clay.layout.route.live_diagnostic import DiagnosticRoutingObserved
            if type(original) is DiagnosticRoutingObserved and callback_returned() and isinstance(ending,Exception):
                from clay.flow.execution_budget import ExecutionStopped
                if isinstance(ending,ExecutionStopped):raise ending from original
                raise CandidateBoundaryChanged(ending) from ending
            from clay.flow.control_stop import record_unwind_failure
            record_unwind_failure(original,ending)
        from clay.layout.route.live_diagnostic import DiagnosticRoutingObserved
        if type(original) is DiagnosticRoutingObserved and callback_returned():
            notes=getattr(original,'__notes__',())
            failures=[note for note in notes if type(note) is str and
                      note.startswith('Ending boundary check also failed:')]
            if failures:
                raise CandidateBoundaryChanged(ValueError(failures[0])) from original
        raise

def _solve(model,request,phase,*,unknown_factory=None):
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1;solver.parameters.random_seed=0
    solver.parameters.max_time_in_seconds=request.verify_wall_seconds
    solver.parameters.max_deterministic_time=request.verify_deterministic_seconds
    solver.parameters.fix_variables_to_their_hinted_value=True
    before_model,before_params=_digest(model.Proto()),_digest(solver.parameters)
    with cp_solve_phase(solver,model,phase=phase):
        effective_params=_digest(solver.parameters)
        status=solver.Solve(model)
        if (_digest(model.Proto())!=before_model or _digest(solver.parameters)!=effective_params or
            solver.ResponseProto().status!=status):raise ValueError('candidate solve changed inputs or status')
    if _digest(solver.parameters)!=before_params:raise ValueError('candidate solver controls were not restored')
    if status not in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        if status==cp_model.UNKNOWN and unknown_factory is not None:
            raise unknown_factory(phase,solver.StatusName(status),int(status))
        raise ValueError('candidate has no complete SAT response: '+solver.StatusName(status))
    return solver,status

def _verify(stage,request,*,unknown_factory=None):
    stage._check(); context,original,_inventory,frame=_compile(stage)
    def solve(model,phase):
        return (_solve(model,request,phase) if unknown_factory is None else
                _solve(model,request,phase,unknown_factory=unknown_factory))
    positions=dict(request.positions_nm); owners=stage.owner_logicals
    if set(positions)!=set(owners):raise ValueError('candidate must contain exact current owners')
    indices=context._indices
    variables={name:tuple(original.GetIntVarFromProtoIndex(i) for i in pair) for name,pair in indices.items()}
    if set(variables)!=set(owners):raise ValueError('candidate current model owner inventory differs')
    fixed=prepare_incumbent_replay_model(original,coordinate_variables=variables,positions=positions)
    solver,status=solve(fixed,'candidate.original')
    require_incumbent_replay_response(fixed,solver,status,positions=positions,coordinate_variables=variables)
    gate=context.build_receipt(model=fixed,solver=solver,solver_status=status,incumbent_replay_positions=positions)
    if gate['status']!='PROMOTABLE':raise ValueError('candidate producer context rejected the fixed vector')
    stage._check()
    outer_model,_=append_current_outer_domain(original,coordinate_indices=indices,outer_kwargs=stage.stage_inputs['outer_kwargs'])
    outer_vars={n:tuple(outer_model.GetIntVarFromProtoIndex(i) for i in pair) for n,pair in indices.items()}
    outer_fixed=prepare_incumbent_replay_model(outer_model,coordinate_variables=outer_vars,positions=positions)
    outer_solver,outer_status=solve(outer_fixed,'candidate.complete_outer')
    require_incumbent_replay_response(outer_fixed,outer_solver,outer_status,positions=positions,coordinate_variables=outer_vars)
    outer=_outer(stage,positions);stage._check()
    return GeneratedIncumbentSeed(dict(schema=SCHEMA,scope=SCOPE,promotion='REJECT',g4_status='NOT_RUN',
        positions_nm=sorted(positions.items()),owner_logicals=sorted(owners.items()),
        native_frame=frame_data(stage.stage_inputs['frame']),
        production_evidence=dict(model_proto_sha256=_digest(fixed.Proto()),promotion_gate_sha256=digest(gate),
                                 outer_replay_sha256=digest(outer))))

def run_candidate_diagnostic(*args, request, **kwargs):
    """Execute once at LayoutStep's actual source-owned compaction boundary."""
    from clay.flow.step import LayoutStep
    from clay.layout import flow_placement
    if type(request) is not CandidateDiagnosticRequest:raise TypeError('typed candidate diagnostic request required')
    if len(args)!=4:raise ValueError('actual public compaction operands required')
    frame=inspect.currentframe().f_back;caller=None
    try:
        while frame is not None:
            if frame.f_code is LayoutStep._run_generic.__code__ and frame.f_globals is LayoutStep._run_generic.__globals__:
                if caller is not None:raise ValueError('ambiguous public layout origin')
                caller=frame
            frame=frame.f_back
        if caller is None:raise ValueError('actual public layout caller required')
        loc=caller.f_locals;ctx=loc['ctx']
        if (args[0] is not loc['placement'] or args[1] is not loc['cell_def'] or args[2] is not loc['intent'] or
            args[3]!=loc['spec'].pdk or kwargs.get('effective_policy') is not loc['eff_policy'] or
            kwargs.get('engine') is not loc['selected_route_engine'] or
            ctx.extra.get('diagnostic_candidate_request') is not request or
            kwargs.get('diagnostic_observer') is not ctx.extra.get('diagnostic_route_observer')):
            raise ValueError('candidate public caller operands differ')
        if not callable(kwargs.get('diagnostic_observer')):raise ValueError('candidate requires terminal diagnostic observer')
        if kwargs.get('prepared_compact') is not None or kwargs.get('allow_placement_repair',True) is not True:
            raise ValueError('candidate acquisition requires the unmodified standard caller')
        session=current_session();admission=current_admission()
        if session is None or admission is None or admission.task_id!=request.task_id:
            raise ValueError('candidate requires its actual admitted task identity')
        session.checkpoint()
        if session.remaining_wall_seconds() is None:raise ValueError('candidate requires a finite task wall budget')
        placement,cell,intent,pdk=args
        tech=pdk if isinstance(pdk,LayoutTech) else LayoutTech.resolve(str(pdk))
        source=reopen_complete_source(cell)
        if source['source_sha256']!=request.source_sha256 or source['pdk_id']!=tech.pdk_id:
            raise ValueError('candidate source or PDK differs')
        parsed=parse_clay_file(source['path']);matches=[c for c in parsed.cell_defs if c.local_name==cell.local_name]
        if len(matches)!=1:raise ValueError('candidate source cell is not unique')
        policy=policy_for_cell(parsed,matches[0]);engine=route_engine_from_policy(policy)
        if (engine is None or policy!=kwargs['effective_policy'] or kwargs.get('max_iters')!=policy.route_iters or
            route_engine_configuration(engine)!=route_engine_configuration(kwargs['engine'])):
            raise ValueError('candidate source policy or selected engine differs')
        native_declared=any(getattr(device,'device_class',None)=='native' for device in cell.devices)
        native_bundle=loc.get('native_geometry') if native_declared else None
        def pdk_identity():
            authority=(_native_pdk_identity(spec=loc['spec'],ctx=ctx,cell=cell,placement=placement,bundle=native_bundle)
                       if native_declared else _installed_pdk_identity(tech))
            return (tech.pdk_id,id(tech),id(tech.adapter),id(tech.rules),pdk_provenance_digest(tech),
                    authority)
        def snapshot():
            current=reopen_complete_source(cell)
            if current['source_sha256']!=source['source_sha256'] or current['path']!=source['path']:
                raise ValueError('candidate source changed')
            return (pdk_identity(),freeze_row_values(cell),freeze_row_values(intent),_placed_values(placement),
                _row_values(placement),freeze_row_values(kwargs.get('device_records')),
                freeze_row_values(kwargs.get('intent_fact_set')),freeze_row_values(kwargs.get('source_inventory')),
                freeze_row_values(kwargs.get('intent_basis')),freeze_row_values(kwargs.get('execution_scope')),
                freeze_row_values((kwargs.get('max_iters'),kwargs.get('allow_placement_repair',True),
                    str(kwargs.get('group_fallback_evidence_dir')))),
                freeze_row_values(asdict(kwargs['effective_policy'])),freeze_row_values(route_engine_configuration(kwargs['engine'])))
        before=snapshot();positions=dict(request.positions_nm)
        def check():
            session.checkpoint()
            if snapshot()!=before:raise ValueError('candidate caller operands changed')
        def policy_for_stage(native,geometry):
            remaining=session.remaining_wall_seconds()
            if remaining is None or remaining<=0:raise ValueError('candidate task wall exhausted')
            return dict(seed=dict(positions),width_nm=request.outline_nm[0],height_nm=request.outline_nm[1],
              movable_y=tuple(sorted(native['dims'])),movable_y_lower_bounds_nm={},bottom_origin_lower_bounds_nm={},
              max_time_s=min(request.model_wall_seconds,remaining),
              warm_start_origin=dict(kind='EXPLICIT_CANDIDATE_DIAGNOSTIC'))
        population=None if native_declared else tuple((name,count) for name,_family,count,_kind in _source_population(cell))
        owner_scope={'geometry_observation':'OWNER_ONLY','quality_occurrences':'UNKNOWN_NOT_MEASURED',
                     'quality_acceptance':'NOT_EVALUATED'} if native_declared else {}
        callback_returned=[False]
        with ExitStack() as stack:
            stack.enter_context(_checked_unwind(check,lambda:callback_returned[0]))
            request_bytes=json.dumps(asdict(request),sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
            stack.enter_context(session.phase('candidate.diagnostic',details={
              'request_sha256':hashlib.sha256(request_bytes).hexdigest(),
              'source_sha256':source['source_sha256'],'task_id':request.task_id,
              'outline_nm':request.outline_nm,'owner_count':len(request.positions_nm),
              'model_wall_seconds':request.model_wall_seconds,'verify_wall_seconds':request.verify_wall_seconds,
              'verify_deterministic_seconds':request.verify_deterministic_seconds,
              'selected_engine':route_engine_configuration(kwargs['engine']),
              'pdk_routing_sha256':pdk_provenance_digest(tech),
              'pdk_authority':before[0][-1],
              'layered_scope':'ACTUAL_STANDARD_HELPER_DEFAULT_EMPTY',
              'source_assembly_completeness':'UNKNOWN',**owner_scope}))
            class _StageNotAcquired(BaseException):pass
            stage_not_acquired=_StageNotAcquired()
            def stop_before_unverified_observer(_cap):
                raise stage_not_acquired
            acquisition_kwargs=dict(kwargs,diagnostic_observer=stop_before_unverified_observer)
            try:
                invoke=lambda:flow_placement.route_with_declared_compaction(*args,**acquisition_kwargs)
                acquisition=(acquire_current_owner_stage(invoke,
                    source_names=tuple(sorted(d.instance_name for d in cell.devices)),build_policy=policy_for_stage)
                    if native_declared else acquire_current_quality_stage(invoke,
                    source_population=population,build_policy=policy_for_stage))
                stage=stack.enter_context(acquisition)
            except _StageNotAcquired as exc:
                if exc is not stage_not_acquired:raise
                raise CandidateDiagnosticUnavailable('native candidate stage was not acquired before routing') from exc
            except CurrentQualityAcquisitionFailure as exc:
                if type(exc) is not CurrentQualityAcquisitionFailure:raise
                raise CandidateAcquisitionError(exc) from exc
            check();seed=_verify(stage,request);check()
            materializer_args={k:kwargs[k] for k in ('effective_policy','intent_basis','intent_fact_set',
                               'source_inventory','device_records') if k in kwargs}
            with scoped_incumbent_replay(indexed_geometry=True) as replay:
                compact=flow_placement.materialize_declared_compact_override(*args,**materializer_args,generated_incumbent=seed)
            if tuple(row['phase'] for row in replay.completed_records)!=('initial_master','exact_materialization_replay'):
                raise ValueError('candidate native replay phases differ')
            def verify_final():
                if native_declared:
                    verify_final_owner_candidate(cell=cell,compact=compact,seed=seed,outline=request.outline_nm)
                else:
                    observation=observe_materialized_source(cell=cell,compact=compact)
                    _,final_positions=project_final_quality_geometry(observation);seed.require_final_positions(final_positions)
                    frame=observation.canonical_data()['final_placement_state']['bbox_nm']
                    if (frame[2]-frame[0],frame[3]-frame[1])!=request.outline_nm:
                        raise ValueError('candidate final outline differs')
            check();verify_final()
            compact_before=_compact_value(compact); observer=kwargs['diagnostic_observer'];calls=[]
            def observed(cap):
                if calls:raise ValueError('candidate diagnostic observer must run once')
                calls.append(True);check();view=cap.inspect()
                if view['compact'] is not compact or _compact_value(compact)!=compact_before:
                    raise ValueError('candidate compact differs at live routing caller')
                verify_final()
                scope=(('layered_scope','ACTUAL_STANDARD_HELPER_DEFAULT_EMPTY'),
                       ('source_assembly_completeness','UNKNOWN'),
                       *tuple(owner_scope.items()),
                       ('candidate_request_sha256',hashlib.sha256(request_bytes).hexdigest()),
                       ('pdk_authority_kind',before[0][-1]['kind']),
                       ('pdk_authority_sha256',hashlib.sha256(json.dumps(before[0][-1],sort_keys=True,
                           separators=(',',':')).encode('utf8')).hexdigest()))
                cap._candidate_diagnostic_scope=scope
                observer(cap);callback_returned[0]=True;check();cap.check()
                if cap._candidate_diagnostic_scope!=scope:
                    raise ValueError('candidate diagnostic scope changed')
            call=dict(kwargs,prepared_compact=compact,allow_placement_repair=False,diagnostic_observer=observed)
            flow_placement.route_with_declared_compaction(*args,**call)
            raise ValueError('candidate diagnostic unexpectedly returned without a terminal stop')
    finally:
        del frame
        del caller
