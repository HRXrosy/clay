"""Source-bound, two-candidate terminal diagnostics in one public invocation.

Batch observers receive detached JSON only. This is trusted in-process
diagnostic code, not a sandbox or a physical-admission mechanism.
"""
from contextlib import ExitStack
from dataclasses import asdict,dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import time

from clay.flow.control_stop import record_unwind_failure
from clay.flow.execution_budget import ExecutionStopped,EXECUTION_CANCELLED
from clay.flow.execution_session import current_session
from clay.flow.run_admission import current_admission
from clay.flow.run_workspace import atomic_json
from clay.frontend.source_revision import reopen_complete_source
from clay.layout.place.candidate_diagnostic import (
    CandidateDiagnosticRequest, CandidateDiagnosticUnavailable,
    _checked_unwind, _native_pdk_identity, _verify,
)
from clay.layout.place.candidate_batch_completion import terminal_completion_scope,issue_candidate_completion
from clay.layout.place.compact_snapshot import compact_value
from clay.layout.place.current_owner_stage import acquire_current_owner_stage,verify_final_owner_candidate
from clay.layout.place.current_quality_stage import CurrentQualityAcquisitionFailure
from clay.layout.place.current_structure_operands import _placed_values,_row_values
from clay.layout.place.generated_incumbent import digest,frame_data
from clay.layout.place.row_replay_snapshot import freeze_row_values
from clay.layout.place.solver_execution_policy import scoped_incumbent_replay
from clay.layout.policy import policy_for_cell
from clay.layout.route.canonical_input_capture import route_engine_configuration
from clay.layout.route.engine_registry import route_engine_from_policy
from clay.layout.route.live_diagnostic import DiagnosticRoutingObserved,observe_prepared_routing
from clay.layout.route.resource_world import pdk_provenance_digest
from clay.layout.tech import LayoutTech
from clay.parser import parse_clay_file


@dataclass(frozen=True,slots=True)
class CandidateBatchRequest:
    """Two explicitly named untrusted proposals for one admitted source/task."""
    candidates: tuple

    def __post_init__(self):
        if (type(self.candidates) is not tuple or len(self.candidates)!=2 or any(
            type(row) is not tuple or len(row)!=2 or type(row[0]) is not str
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',row[0]) is None
            or type(row[1]) is not CandidateDiagnosticRequest for row in self.candidates)):
            raise ValueError('candidate batch requires exactly two named typed requests')
        if len({row[0] for row in self.candidates})!=2:
            raise ValueError('candidate batch names must be unique')
        if len({(row[1].task_id,row[1].source_sha256) for row in self.candidates})!=1:
            raise ValueError('candidate batch must share one task and source')


class CandidateBatchCallbackError(RuntimeError):
    """A callback attempted to impersonate the terminal diagnostic stop."""
    def flow_error_data(self):
        return {'reason':'CANDIDATE_BATCH_CALLBACK_FAILED','detail':str(self)}


class CandidateBatchCompletionError(RuntimeError):
    """A diagnostic stop lacked its current one-shot producer completion."""
    def flow_error_data(self):
        return {'reason':'CANDIDATE_BATCH_COMPLETION_INVALID','detail':str(self)}


class CandidateBatchSolverUnknown(ValueError):
    """The actual native solver did not return a complete response."""
    def __init__(self,phase,status,status_code):
        self.phase=phase;self.solver_status=status;self.solver_status_code=status_code
        super().__init__(f'{phase} returned {status} without a complete solver response')

    def flow_error_data(self):
        return {'reason':'CANDIDATE_BATCH_SOLVER_UNKNOWN','phase':self.phase,
                'solver_status':self.solver_status,'solver_status_code':self.solver_status_code}


def _diagnostic_origin(error):
    tb=error.__traceback__
    if tb is None:return False
    while tb.tb_next is not None:tb=tb.tb_next
    return tb.tb_frame.f_code is observe_prepared_routing.__code__


def run_candidate_batch_diagnostic(*args,request,**kwargs):
    """Run two fully separate candidate preparations under one source/task."""
    from clay.flow.step import LayoutStep
    from clay.layout import flow_placement
    if type(request) is not CandidateBatchRequest or len(args)!=4:
        raise TypeError('typed candidate batch and actual public operands required')
    frame=inspect.currentframe().f_back;caller=None
    try:
        while frame is not None:
            if frame.f_code is LayoutStep._run_generic.__code__ and frame.f_globals is LayoutStep._run_generic.__globals__:
                if caller is not None:raise ValueError('ambiguous public layout origin')
                caller=frame
            frame=frame.f_back
        if caller is None:raise ValueError('actual public layout caller required')
        loc=caller.f_locals;ctx=loc['ctx'];placement,cell,intent,pdk=args
        callback=ctx.extra.get('diagnostic_candidate_batch_observer')
        descriptor=ctx.extra.get('diagnostic_source_handoff_request')
        resource_request=ctx.extra.get('diagnostic_resource_setup_request')
        access_request=ctx.extra.get('diagnostic_resource_access_request')
        lane_survey_request=ctx.extra.get('diagnostic_resource_lane_survey_request')
        lane_site_request=ctx.extra.get('diagnostic_resource_lane_site_request')
        pair_request=ctx.extra.get('diagnostic_resource_pair_request')
        if descriptor is not None:
            from clay.layout.route.source_handoff import SourceHandoffRequest
            if type(descriptor) is not SourceHandoffRequest:
                raise TypeError('source handoff requires a typed request')
            descriptor.__post_init__()
        if resource_request is not None:
            from clay.layout.route.resource_setup_diagnostic import ResourceSetupRequest
            if type(resource_request) is not ResourceSetupRequest:
                raise TypeError('resource setup requires a typed request')
            resource_request.__post_init__()
            if descriptor is None:
                raise ValueError('resource setup requires source handoff')
        if access_request is not None:
            from clay.layout.route.resource_access_diagnostic import ResourceAccessRequest
            if type(access_request) is not ResourceAccessRequest:
                raise TypeError('resource access requires a typed request')
            access_request.__post_init__()
            if resource_request is None:
                raise ValueError('resource access requires resource setup')
        if lane_survey_request is not None:
            from clay.layout.route.resource_lane_survey import ResourceLaneSurveyRequest
            if type(lane_survey_request) is not ResourceLaneSurveyRequest:
                raise TypeError('resource lane survey requires a typed request')
            lane_survey_request.__post_init__()
            if resource_request is None:
                raise ValueError('resource lane survey requires resource setup')
            if access_request is not None:
                raise ValueError('resource lane survey and access are mutually exclusive')
        if lane_site_request is not None:
            from clay.layout.route.resource_lane_site import ResourceLaneSiteRequest
            if type(lane_site_request) is not ResourceLaneSiteRequest:
                raise TypeError('resource lane site requires a typed request')
            lane_site_request.__post_init__()
            if resource_request is None:
                raise ValueError('resource lane site requires resource setup')
            if access_request is not None or lane_survey_request is not None:
                raise ValueError('resource lane site, lane survey and access are mutually exclusive')
        if pair_request is not None:
            from clay.layout.route.resource_pair_transaction import ResourcePairRequest
            if type(pair_request) is not ResourcePairRequest:
                raise TypeError('resource pair requires a typed request')
            pair_request.__post_init__()
            if lane_site_request is None:
                raise ValueError('resource pair requires resource lane site')
        if (placement is not loc['placement'] or cell is not loc['cell_def'] or intent is not loc['intent']
            or pdk!=loc['spec'].pdk or kwargs.get('effective_policy') is not loc['eff_policy']
            or kwargs.get('engine') is not loc['selected_route_engine']
            or ctx.extra.get('diagnostic_candidate_batch_request') is not request
            or ctx.extra.get('diagnostic_candidate_request') is not None
            or ctx.extra.get('diagnostic_route_observer') is not None or not callable(callback)):
            raise ValueError('candidate batch public caller operands differ')
        if kwargs.get('prepared_compact') is not None or kwargs.get('allow_placement_repair',True) is not True:
            raise ValueError('candidate batch requires the unmodified standard caller')
        session=current_session();admission=current_admission()
        if session is None or admission is None or session.remaining_wall_seconds() is None:
            raise ValueError('candidate batch requires one admitted finite task')
        if descriptor is not None and session.invocation_id != admission.invocation_id:
            raise CandidateDiagnosticUnavailable('UNKNOWN_MANAGED_HANDOFF_UNSUPPORTED')
        source=reopen_complete_source(cell);parsed=parse_clay_file(source['path'])
        matches=[x for x in parsed.cell_defs if x.local_name==cell.local_name]
        if len(matches)!=1:raise ValueError('candidate batch source cell is not unique')
        if not cell.devices or any(getattr(d,'device_class',None)!='native' for d in cell.devices):
            raise CandidateDiagnosticUnavailable('candidate batch requires a complete native source')
        if any(r.task_id!=admission.task_id or r.source_sha256!=source['source_sha256'] for _,r in request.candidates):
            raise ValueError('candidate batch task or source differs')
        policy=policy_for_cell(parsed,matches[0]);selected=route_engine_from_policy(policy)
        if (selected is None or policy!=kwargs['effective_policy'] or kwargs.get('max_iters')!=policy.route_iters
            or route_engine_configuration(selected)!=route_engine_configuration(kwargs['engine'])):
            raise ValueError('candidate batch declared policy or selected engine differs')
        tech=pdk if isinstance(pdk,LayoutTech) else LayoutTech.resolve(str(pdk))
        if source['pdk_id']!=tech.pdk_id:raise ValueError('candidate batch source PDK differs')
        bundle=loc.get('native_geometry');config=route_engine_configuration(selected)
        def boundary():
            session.checkpoint();current=reopen_complete_source(cell)
            if current['source_sha256']!=source['source_sha256'] or current['path']!=source['path']:
                raise ValueError('candidate batch source changed')
            current_loc=caller.f_locals
            if (current_loc.get('selected_route_engine') is not kwargs['engine']
                    or current_loc.get('eff_policy') is not kwargs['effective_policy']):
                raise ValueError('candidate batch actual selected engine or policy identity changed')
            current_parsed=parse_clay_file(current['path'])
            current_cells=[x for x in current_parsed.cell_defs if x.local_name==cell.local_name]
            if len(current_cells)!=1 or policy_for_cell(current_parsed,current_cells[0])!=policy:
                raise ValueError('candidate batch declared policy changed')
            return ((tech.pdk_id,id(tech),id(tech.adapter),id(tech.rules),pdk_provenance_digest(tech),
                    _native_pdk_identity(spec=loc['spec'],ctx=ctx,cell=cell,placement=placement,bundle=bundle)),
                freeze_row_values(cell),freeze_row_values(intent),_placed_values(placement),_row_values(placement),
                freeze_row_values((kwargs.get('device_records'),kwargs.get('intent_fact_set'),
                    kwargs.get('source_inventory'),kwargs.get('intent_basis'),kwargs.get('execution_scope'),
                    kwargs.get('max_iters'),kwargs.get('allow_placement_repair',True),
                    str(kwargs.get('group_fallback_evidence_dir')))),
                freeze_row_values(asdict(policy)),freeze_row_values(route_engine_configuration(kwargs['engine'])))
        baseline=boundary()
        def check():
            if boundary()!=baseline:raise ValueError('candidate batch source/PDK/placement changed')
        request_sha=digest(asdict(request));observer_id=id(callback)
        descriptor_sha=digest(asdict(descriptor)) if descriptor is not None else None
        resource_request_sha=digest(asdict(resource_request)) if resource_request is not None else None
        access_request_sha=digest(asdict(access_request)) if access_request is not None else None
        lane_survey_request_sha=digest(asdict(lane_survey_request)) if lane_survey_request is not None else None
        lane_site_request_sha=digest(asdict(lane_site_request)) if lane_site_request is not None else None
        pair_request_sha=digest(asdict(pair_request)) if pair_request is not None else None
        def check_request():
            if (ctx.extra.get('diagnostic_candidate_batch_request') is not request or digest(asdict(request))!=request_sha
                    or ctx.extra.get('diagnostic_candidate_batch_observer') is not callback or id(callback)!=observer_id
                    or ctx.extra.get('diagnostic_source_handoff_request') is not descriptor
                    or (descriptor is not None and digest(asdict(descriptor))!=descriptor_sha)
                    or ctx.extra.get('diagnostic_resource_setup_request') is not resource_request
                    or (resource_request is not None and digest(asdict(resource_request))!=resource_request_sha)
                    or ctx.extra.get('diagnostic_resource_access_request') is not access_request
                    or (access_request is not None and digest(asdict(access_request))!=access_request_sha)
                    or ctx.extra.get('diagnostic_resource_lane_survey_request') is not lane_survey_request
                    or (lane_survey_request is not None and digest(asdict(lane_survey_request))!=lane_survey_request_sha)
                    or ctx.extra.get('diagnostic_resource_lane_site_request') is not lane_site_request
                    or (lane_site_request is not None and digest(asdict(lane_site_request))!=lane_site_request_sha)
                    or ctx.extra.get('diagnostic_resource_pair_request') is not pair_request
                    or (pair_request is not None and digest(asdict(pair_request))!=pair_request_sha)):
                raise ValueError('candidate batch request or callback changed')
            check()
        output=Path(ctx.work_dir)/'candidate-batch'/admission.invocation_id
        output.mkdir(parents=True,exist_ok=False)
        progress={'schema':'clay.candidate-batch-progress/v1','status':'RUNNING','physical_admission':False,
            'task_id':admission.task_id,'invocation_id':admission.invocation_id,
            'source_sha256':source['source_sha256'],'request_sha256':request_sha,
            'terminal_receipt_status':'PENDING_NOT_VERIFIED',
            'candidates':[{'id':name,'status':'PENDING'} for name,_ in request.candidates]}
        if descriptor is not None:progress['source_handoff_request_sha256']=descriptor_sha
        if resource_request is not None:progress['resource_setup_request_sha256']=resource_request_sha
        if access_request is not None:progress['resource_access_request_sha256']=access_request_sha
        if lane_survey_request is not None:progress['resource_lane_survey_request_sha256']=lane_survey_request_sha
        if lane_site_request is not None:progress['resource_lane_site_request_sha256']=lane_site_request_sha
        if pair_request is not None:progress['resource_pair_request_sha256']=pair_request_sha
        progress_path=output/'progress.json';atomic_json(progress_path,progress)
        engines=[];books=[];worlds=[];completed=set();last_stop=None
        try:
            for index,(name,proposal) in enumerate(request.candidates):
                started=time.monotonic();check_request();row=progress['candidates'][index]
                row.update(status='RUNNING',phase='ACQUISITION');atomic_json(progress_path,progress)
                engine=route_engine_from_policy(policy)
                if (engine is None or any(engine is old for old in engines)
                        or route_engine_configuration(engine)!=config):
                    raise ValueError('candidate batch requires fresh source-selected engine')
                engines.append(engine);positions=dict(proposal.positions_nm)
                def policy_for_stage(native,_partition):
                    remaining=session.remaining_wall_seconds()
                    if remaining is None or remaining<=0:raise ValueError('candidate batch task wall exhausted')
                    return dict(seed=dict(positions),width_nm=proposal.outline_nm[0],height_nm=proposal.outline_nm[1],
                        movable_y=tuple(sorted(native['dims'])),movable_y_lower_bounds_nm={},bottom_origin_lower_bounds_nm={},
                        max_time_s=min(proposal.model_wall_seconds,remaining),
                        warm_start_origin={'kind':'EXPLICIT_CANDIDATE_BATCH_DIAGNOSTIC'})
                class _NoStage(BaseException):pass
                absent=_NoStage()
                def stop_before_unverified(_cap):raise absent
                acquisition=dict(kwargs,engine=engine,diagnostic_observer=stop_before_unverified)
                acquired=time.monotonic();internal={'returned':False,'cap':None,'summary':None}
                with ExitStack() as stack:
                    stack.enter_context(_checked_unwind(check_request,lambda:internal['returned']))
                    try:
                        stage=stack.enter_context(acquire_current_owner_stage(
                            lambda:flow_placement.route_with_declared_compaction(*args,**acquisition),
                            source_names=tuple(sorted(d.instance_name for d in cell.devices)),build_policy=policy_for_stage))
                    except _NoStage as exc:
                        if exc is not absent:raise
                        raise CandidateDiagnosticUnavailable('candidate batch native stage was not acquired') from exc
                    except CurrentQualityAcquisitionFailure as exc:
                        from clay.layout.place.candidate_diagnostic import CandidateAcquisitionError
                        if type(exc) is not CurrentQualityAcquisitionFailure:raise
                        raise CandidateAcquisitionError(exc) from exc
                    row.update(phase='MODEL',acquisition_wall_seconds=time.monotonic()-acquired,
                        frame_sha256=digest(frame_data(stage.stage_inputs['frame'])),
                        owner_partition_sha256=digest(stage.owner_logicals));atomic_json(progress_path,progress)
                    check_request();seed=_verify(stage,proposal,unknown_factory=CandidateBatchSolverUnknown);check_request()
                    row.update(phase='MATERIALIZATION',seed_sha256=seed.sha256);atomic_json(progress_path,progress)
                    materializer={k:kwargs[k] for k in ('effective_policy','intent_basis','intent_fact_set',
                                  'source_inventory','device_records') if k in kwargs}
                    with scoped_incumbent_replay(indexed_geometry=True) as replay:
                        compact=flow_placement.materialize_declared_compact_override(
                            *args,**materializer,generated_incumbent=seed)
                    if tuple(r['phase'] for r in replay.completed_records)!=('initial_master','exact_materialization_replay'):
                        raise ValueError('candidate batch native replay phases differ')
                    verify_final_owner_candidate(cell=cell,compact=compact,seed=seed,outline=proposal.outline_nm)
                    before=compact_value(compact);row['phase']='PREPARED_ROUTING';atomic_json(progress_path,progress)
                    def observed(cap):
                        if internal['cap'] is not None:raise ValueError('candidate batch observer repeated')
                        internal['cap']=cap;check_request();view=cap.inspect()
                        deferred_pair=(pair_request is not None and pair_request.schema in (
                            'clay.resource-pair-request/v3-deferred',
                            'clay.resource-pair-request/v4-iterative'))
                        if view['compact'] is not compact or compact_value(compact)!=before:
                            raise ValueError('candidate batch compact changed at routing caller')
                        verify_final_owner_candidate(cell=cell,compact=compact,seed=seed,outline=proposal.outline_nm)
                        if descriptor is None:
                            setup=cap.initialize_before_groups()
                            source_value=None
                        else:
                            scope=ticket.scope
                            if scope is None:raise ValueError('source handoff producer scope missing')
                            scope._bind(cap)
                            source_value=scope.observe()
                            if resource_request is not None:
                                source_value_sha=digest(source_value)
                            setup=None if deferred_pair else scope.consume_initialization(scope.initialize())
                        if not deferred_pair and setup.get('status')!='OBSERVED_EMPTY_PRE_GROUP':
                            raise CandidateDiagnosticUnavailable('candidate batch empty pre-group Book unavailable')
                        book=None if deferred_pair else engine._book
                        book_status='UNKNOWN_NOT_INITIALIZED' if deferred_pair else 'OBSERVED_EMPTY_PRE_GROUP'
                        world=view['world'];world_sha=world.sha256()
                        if (any(world is old for old in worlds)
                                or (book is not None and any(book is old for old in books))):
                            raise ValueError('candidate batch reused a prior world or Book')
                        if book is not None:books.append(book)
                        worlds.append(world)
                        cap.check();check_request()
                        if (cap._inputs['route_engine'] is not engine or cap._inputs['routing_world'] is not world
                                or (getattr(engine,'_book',None) is not book)
                                or route_engine_configuration(engine)!=config):
                            raise ValueError('candidate batch world/engine/Book identity differs')
                        resource_value=None
                        if resource_request is not None:
                            from clay.layout.route.resource_setup_diagnostic import run_resource_setup
                            if digest(source_value)!=source_value_sha:
                                raise ValueError('resource setup source observation changed during initialization')
                            resource_value=run_resource_setup(scope=scope,cap=cap,source_value=source_value,
                                source_value_sha=source_value_sha,request=resource_request,check_request=check_request,
                                access_request=access_request,lane_survey_request=lane_survey_request,
                                lane_site_request=lane_site_request,pair_request=pair_request)
                            scope._check();cap.check();check_request()
                            if digest(source_value)!=source_value_sha:
                                raise ValueError('resource setup source observation changed')
                            if deferred_pair and cap._initialized:
                                record=scope._pair_initialization
                                book=getattr(engine,'_book',None)
                                if (scope._initial_receipt is not False or type(record) is not tuple
                                        or len(record)!=5 or record[1] is not cap or record[2] is not engine
                                        or record[3] is not book or record[4] is not scope._ticket
                                        or book is None or any(book is old for old in books)):
                                    raise ValueError('deferred pair initialization identity differs')
                                books.append(book)
                                book_status='OBSERVED_EMPTY_PRE_GROUP'
                            elif deferred_pair:
                                if (scope._initial_receipt is not None or scope._pair_initialization is not None
                                        or getattr(engine,'_book',None) is not None):
                                    raise ValueError('uninitialized deferred pair state differs')
                            if (cap._inputs['route_engine'] is not engine or cap._inputs['routing_world'] is not world
                                    or getattr(engine,'_book',None) is not book or route_engine_configuration(engine)!=config
                                    or (book is not None and (any(book.metal.values()) or any(book.via_pads.values())
                                        or book.cuts or book.persistent or any(engine._attempt_metal.values())
                                        or any(engine._attempt_cuts.values()) or engine._solve_expansions_total!=0))):
                                raise ValueError('resource setup changed the pre-group engine or Book')
                            if (type(resource_value) is not dict
                                    or resource_value.get('schema')!='clay.resource-setup-observation/v1'
                                    or type(resource_value.get('status')) is not str
                                    or resource_value.get('status') not in (
                                        'OBSERVED_LEDGER_INDEX','UNKNOWN_METER_LIMIT','UNKNOWN_SETUP_WALL',
                                        'UNKNOWN_LOCAL_LIMIT','UNKNOWN_INCIDENCE','UNKNOWN_OUTPUT_LIMIT',
                                        'UNKNOWN_ENGINE_PIN_PROJECTION_UNSUPPORTED')
                                    or resource_value.get('publication_status')!='PENDING_TERMINAL_SCOPE_RECHECK'
                                    or resource_value.get('physical_admission') is not False):
                                raise ValueError('resource setup did not return a detached pending observation')
                            resource_scope_payload={
                                'schema':'clay.resource-setup-scope/v1',
                                'invocation_id':admission.invocation_id,'candidate_identity_sha256':candidate_sha,
                                'source_handoff_request_sha256':descriptor_sha,
                                'resource_setup_request_sha256':resource_request_sha}
                            if access_request is not None:
                                resource_scope_payload['resource_access_request_sha256']=access_request_sha
                            if lane_survey_request is not None:
                                resource_scope_payload['resource_lane_survey_request_sha256']=lane_survey_request_sha
                            if lane_site_request is not None:
                                resource_scope_payload['resource_lane_site_request_sha256']=lane_site_request_sha
                            if pair_request is not None:
                                resource_scope_payload['resource_pair_request_sha256']=pair_request_sha
                            expected_resource_scope=digest(resource_scope_payload)
                            if (resource_value.get('request_sha256')!=resource_request_sha
                                    or resource_value.get('scope_identity_sha256')!=expected_resource_scope
                                    or (access_request is not None and
                                        resource_value.get('resource_access_request_sha256')!=access_request_sha)
                                    or (lane_survey_request is not None and
                                        resource_value.get('resource_lane_survey_request_sha256')!=lane_survey_request_sha)
                                    or (lane_site_request is not None and
                                        resource_value.get('resource_lane_site_request_sha256')!=lane_site_request_sha)
                                    or (pair_request is not None and
                                        resource_value.get('resource_pair_request_sha256')!=pair_request_sha)):
                                raise ValueError('resource setup request or scope identity differs')
                            if access_request is not None:
                                expected_access_scope=digest({'schema':'clay.resource-access-scope/v1',
                                    'resource_setup_scope_identity_sha256':expected_resource_scope,
                                    'resource_access_request_sha256':access_request_sha})
                                access_value=resource_value.get('resource_access')
                                if (type(access_value) is not dict
                                        or access_value.get('schema')!='clay.resource-access-observation/v1'
                                        or type(access_value.get('status')) is not str
                                        or access_value.get('publication_status')!='PENDING_TERMINAL_SCOPE_RECHECK'
                                        or access_value.get('physical_admission') is not False
                                        or access_value.get('request_sha256')!=access_request_sha
                                        or access_value.get('scope_identity_sha256')!=expected_access_scope):
                                    raise ValueError('resource access request or scope identity differs')
                            if lane_survey_request is not None:
                                expected_lane_survey_scope=digest({'schema':'clay.resource-lane-survey-scope/v1',
                                    'resource_setup_scope_identity_sha256':expected_resource_scope,
                                    'resource_lane_survey_request_sha256':lane_survey_request_sha})
                                lane_survey_value=resource_value.get('resource_lane_survey')
                                if (type(lane_survey_value) is not dict
                                        or lane_survey_value.get('schema')!='clay.resource-lane-survey-observation/v1'
                                        or type(lane_survey_value.get('status')) is not str
                                        or lane_survey_value.get('publication_status')!='PENDING_TERMINAL_SCOPE_RECHECK'
                                        or lane_survey_value.get('physical_admission') is not False
                                        or lane_survey_value.get('request_sha256')!=lane_survey_request_sha
                                        or lane_survey_value.get('scope_identity_sha256')!=expected_lane_survey_scope):
                                    raise ValueError('resource lane survey request or scope identity differs')
                            if lane_site_request is not None:
                                expected_lane_site_scope=digest({'schema':'clay.resource-lane-site-scope/v1',
                                    'resource_setup_scope_identity_sha256':expected_resource_scope,
                                    'resource_lane_site_request_sha256':lane_site_request_sha})
                                lane_site_value=resource_value.get('resource_lane_site')
                                if (type(lane_site_value) is not dict
                                        or lane_site_value.get('schema')!='clay.resource-lane-site-observation/v1'
                                        or type(lane_site_value.get('status')) is not str
                                        or lane_site_value.get('publication_status')!='PENDING_TERMINAL_SCOPE_RECHECK'
                                        or lane_site_value.get('physical_admission') is not False
                                        or lane_site_value.get('request_sha256')!=lane_site_request_sha
                                        or lane_site_value.get('scope_identity_sha256')!=expected_lane_site_scope):
                                    raise ValueError('resource lane site request or scope identity differs')
                                if pair_request is not None:
                                    pair_value=lane_site_value.get('resource_pair')
                                    expected_pair_scope=digest({'schema':'clay.resource-pair-scope/v1',
                                        'resource_lane_site_scope_identity_sha256':expected_lane_site_scope,
                                        'resource_pair_request_sha256':pair_request_sha})
                                    if (type(pair_value) is not dict
                                            or pair_value.get('schema')!='clay.resource-pair-observation/v1'
                                            or pair_value.get('status') not in ('UNKNOWN_NOT_EXECUTED',
                                                'PARTIAL_ITERATIVE_PAIR',
                                                'UNKNOWN_NO_TWO_NET_SITES','UNKNOWN_ENGINE_UNSUPPORTED',
                                                'OBSERVED_BOUNDED_PAIR','UNKNOWN_PAIR_WALL','UNKNOWN_PAIR_METER',
                                                'UNKNOWN_PAIR_OUTPUT_LIMIT')
                                            or pair_value.get('publication_status')!='PENDING_TERMINAL_SCOPE_RECHECK'
                                            or pair_value.get('physical_admission') is not False
                                            or pair_value.get('request_sha256')!=pair_request_sha
                                            or pair_value.get('scope_identity_sha256')!=expected_pair_scope
                                            or resource_value.get('pair')!=pair_value.get('status')):
                                        raise ValueError('resource pair request or scope identity differs')
                            encoded_resource=json.dumps(resource_value,sort_keys=True,
                                separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf8')
                            if len(encoded_resource)>resource_request.max_output_bytes:
                                raise ValueError('resource setup observation exceeds its output budget')
                            resource_value=json.loads(encoded_resource)
                        snapshot=json.loads(cap._snapshot)
                        internal['summary']={'id':name,'source_sha256':source['source_sha256'],
                            'candidate_request_sha256':digest(asdict(proposal)),
                            'candidate_identity_sha256':candidate_sha,
                            'seed_sha256':seed.sha256,'frame_sha256':row['frame_sha256'],
                            'owner_partition_sha256':row['owner_partition_sha256'],
                            'outline_nm':list(proposal.outline_nm),'engine':config,'world_sha256':world_sha,
                            'prepared_input_sha256':hashlib.sha256(cap._snapshot.encode('utf8')).hexdigest(),
                            'pins_by_net':snapshot['pins_by_net'],'rails_by_net':snapshot['rails_by_net'],
                            'book_status':book_status,'geometry_observation':'OWNER_ONLY',
                            'screen_status':'UNKNOWN_NOT_EXECUTED','screen_coverage':'PRE_GROUP_ONLY',
                            'source_to_via_contact':'UNKNOWN_NOT_CHECKED',
                            'same_net_union':'UNKNOWN_NOT_OBSERVED','joint_route':'UNKNOWN_NOT_OBSERVED',
                            'source_assembly_completeness':'UNKNOWN','quality_occurrences':'UNKNOWN_NOT_MEASURED',
                            'quality_acceptance':'NOT_EVALUATED','physical_admission':False}
                        if source_value is not None:
                            source_value['descriptor_sha256']=descriptor_sha
                            source_value['scope_identity_sha256']=digest({'candidate':candidate_sha,
                                'descriptor':descriptor_sha,'invocation':admission.invocation_id})
                            internal['summary']['source_handoff']=source_value
                        if resource_value is not None:
                            internal['summary']['resource_setup']=resource_value
                            internal['summary']['screen_status']=(
                                resource_value['resource_access']['status']
                                if access_request is not None else
                                resource_value['resource_lane_survey']['status']
                                if lane_survey_request is not None else
                                resource_value['resource_lane_site']['status']
                                if lane_site_request is not None else resource_value['status'])
                            internal['summary']['screen_coverage']=(
                                ('RESOURCE_SETUP_AND_ACCESS' if access_request is not None else
                                 'RESOURCE_SETUP_AND_LANE_SURVEY' if lane_survey_request is not None else
                                 'RESOURCE_SETUP_AND_LANE_SITE' if lane_site_request is not None else
                                 'RESOURCE_SETUP_ONLY')
                                if resource_value['status']=='OBSERVED_LEDGER_INDEX'
                                else 'RESOURCE_SETUP_PARTIAL_OR_UNKNOWN')
                        internal['returned']=True
                    call=dict(kwargs,engine=engine,prepared_compact=compact,allow_placement_repair=False,
                              diagnostic_observer=observed)
                    candidate_sha=digest({'invocation_id':admission.invocation_id,'batch_sha256':request_sha,
                        'candidate_id':name,'ordinal':index,'proposal':asdict(proposal)})
                    with terminal_completion_scope(candidate_sha256=candidate_sha,callback=observed) as completion:
                        if descriptor is not None:
                            from clay.layout.route.source_handoff import _new_candidate_ticket
                            ticket=_new_candidate_ticket(descriptor,candidate_sha)
                            call['diagnostic_source_ticket']=ticket
                        try:flow_placement.route_with_declared_compaction(*args,**call)
                        except DiagnosticRoutingObserved as stop:
                            cap=internal['cap']
                            try:
                                if (not internal['returned'] or cap is None or cap._active
                                        or not _diagnostic_origin(stop)):
                                    raise ValueError('candidate terminal stop did not originate from the current producer')
                                completion.consume(cap=cap,stop=stop,candidate_sha256=candidate_sha)
                            except ValueError as ending:
                                record_unwind_failure(stop,ending)
                                raise CandidateBatchCompletionError('candidate terminal completion is invalid') from stop
                            last_stop=stop
                        else:raise ValueError('candidate batch diagnostic did not terminate')
                    check_request()
                # The materializer stage and live capability are closed here.
                # User code sees a detached string only and is outside the catch
                # that accepts the genuine internal terminal stop.
                payload=json.dumps(internal['summary'],sort_keys=True,separators=(',',':'),allow_nan=False)
                callback_returned=[False]
                with _checked_unwind(check_request,lambda:callback_returned[0]):
                    try:callback(payload)
                    except DiagnosticRoutingObserved as exc:
                        raise CandidateBatchCallbackError('batch callback raised a diagnostic stop') from exc
                    callback_returned[0]=True;check_request()
                    if compact_value(compact)!=before:raise ValueError('candidate batch compact changed after callback')
                    verify_final_owner_candidate(cell=cell,compact=compact,seed=seed,outline=proposal.outline_nm)
                final_completion=issue_candidate_completion(candidate_sha256=candidate_sha,
                    cap=internal['cap'],stage=stage,terminal=completion,stop=last_stop,
                    callback_returned=callback_returned[0])
                final_completion.consume(candidate_sha256=candidate_sha,cap=internal['cap'],
                    stage=stage,terminal=completion,stop=last_stop)
                if name in completed:raise ValueError('candidate batch completion repeated')
                completed.add(name)
                row.update(status='PRE_GROUP_COMPLETE',phase='PRE_GROUP_COMPLETE',summary=json.loads(payload),
                           publication_status='PENDING_TERMINAL_RECEIPT',candidate_wall_seconds=time.monotonic()-started)
                atomic_json(progress_path,progress)
            if len(completed)!=2 or last_stop is None:raise ValueError('candidate batch incomplete')
            progress['status']='COMPLETE_PRE_GROUP_ONLY';atomic_json(progress_path,progress)
        except BaseException as original:
            try:
                unknown=isinstance(original,(ExecutionStopped,CandidateBatchSolverUnknown))
                if isinstance(original,CandidateBatchSolverUnknown):unknown_status='UNKNOWN_PARTIAL'
                elif isinstance(original,ExecutionStopped):
                    unknown_status='UNKNOWN_CANCELLED' if original.reason==EXECUTION_CANCELLED else 'UNKNOWN_BUDGET'
                else:unknown_status=None
                progress['status']=unknown_status if unknown else 'FAILED'
                row=progress['candidates'][index]
                if row['status']=='RUNNING':
                    row.update(status=unknown_status if unknown else 'FAILED',
                        error_type=type(original).__name__,error=str(original)[:1024])
                    if isinstance(original,CandidateBatchSolverUnknown):
                        row.update(solver_phase=original.phase,solver_status=original.solver_status,
                                   solver_status_code=original.solver_status_code)
                    elif isinstance(original,ExecutionStopped):row['stop_reason']=original.reason
                atomic_json(progress_path,progress)
            except BaseException as ending:record_unwind_failure(original,ending)
            raise
        raise last_stop
    finally:
        del frame
        del caller
