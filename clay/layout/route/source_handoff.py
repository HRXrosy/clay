"""Invocation-owned source/Pin/row handoff from the actual route producer.

This is an opt-in diagnostic contract for trusted in-process callers. It is not
a security sandbox, a routing result, or physical admission.
"""
from collections import Counter,defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict,dataclass
import inspect
import json
import re

from clay.flow.control_stop import record_unwind_failure
from clay.flow.execution_budget import ExecutionStopped
from clay.flow.execution_session import current_session
from clay.flow.run_admission import current_admission
from clay.frontend.source_revision import reopen_complete_source
from clay.layout.place.native_stage_observation import _execution_owner
from clay.layout.place.row_replay_snapshot import freeze_row_values
from clay.layout.place.current_structure_operands import _row_values
from clay.layout.route.pins import PinReport
from clay.layout.route.resource_plan import TopologyPlan
from clay.layout.route.resource_world import pdk_provenance_digest
from clay.layout.route.terminal_identity import pin_fact,pin_source_aliases,unpack_pin_fact


@dataclass(frozen=True,slots=True)
class SourceHandoffRequest:
    """Request current source/Pin/row values inside one candidate batch."""
    schema: str='clay.source-handoff-request/v1'
    def __post_init__(self):
        if self.schema!='clay.source-handoff-request/v1':
            raise ValueError('unsupported source handoff request schema')


_TICKET_KEY=object()
_ACTIVE_SCOPE=ContextVar('source_handoff_scope',default=None)


class _CandidateTicket:
    __slots__=('descriptor','descriptor_snapshot','candidate_sha','session','admission','owner','used','scope',
               'completion','request','request_sha','ctx','candidate_index','proposal','candidate_name',
               'producer_cap','producer_code')
    def __init__(self,descriptor,candidate_sha,*,completion,request,request_sha,ctx,index,name,proposal,_key):
        caller=inspect.currentframe().f_back
        try:
            if (_key is not _TICKET_KEY or type(descriptor) is not SourceHandoffRequest
                    or caller.f_code is not _new_candidate_ticket.__code__
                    or caller.f_globals is not _new_candidate_ticket.__globals__):
                raise ValueError('source handoff ticket requires internal coordinator')
        finally:del caller
        descriptor.__post_init__()
        session=current_session();admission=current_admission()
        if (session is None or admission is None or session.invocation_id!=admission.invocation_id
                or session.remaining_wall_seconds() is None or type(candidate_sha) is not str
                or re.fullmatch(r'[0-9a-f]{64}',candidate_sha) is None):
            raise ValueError('source handoff requires current finite candidate')
        self.descriptor=descriptor;self.descriptor_snapshot=freeze_row_values(asdict(descriptor))
        self.candidate_sha=candidate_sha;self.session=session;self.admission=admission
        self.owner=_execution_owner();self.used=False;self.scope=None;self.producer_cap=None;self.producer_code=None
        self.completion=completion;self.request=request;self.request_sha=request_sha;self.ctx=ctx
        self.candidate_index=index;self.candidate_name=name;self.proposal=proposal

    def __reduce_ex__(self,protocol):
        raise TypeError('source handoff ticket cannot be serialized')

    def _check(self):
        from clay.layout.place.candidate_batch_completion import _ACTIVE
        from clay.layout.place.generated_incumbent import digest
        if (current_session() is not self.session or current_admission() is not self.admission
                or _execution_owner()!=self.owner or freeze_row_values(asdict(self.descriptor))!=self.descriptor_snapshot
                or _ACTIVE.get() is not self.completion or getattr(self.completion,'_source_handoff_ticket',None) is not self
                or self.completion.candidate_sha256!=self.candidate_sha
                or self.ctx.extra.get('diagnostic_source_handoff_request') is not self.descriptor
                or self.ctx.extra.get('diagnostic_candidate_batch_request') is not self.request
                or digest(asdict(self.request))!=self.request_sha
                or self.request.candidates[self.candidate_index][0]!=self.candidate_name
                or self.request.candidates[self.candidate_index][1] is not self.proposal):
            raise ValueError('source handoff ticket is foreign or changed')
        self.completion._check()
        self.session.checkpoint()


def _new_candidate_ticket(descriptor,candidate_sha):
    from clay.layout.place.candidate_batch_completion import _ACTIVE
    from clay.layout.place.candidate_batch_diagnostic import run_candidate_batch_diagnostic
    from clay.layout.place.generated_incumbent import digest
    caller=inspect.currentframe().f_back
    try:
        if (caller.f_code is not run_candidate_batch_diagnostic.__code__
                or caller.f_globals is not run_candidate_batch_diagnostic.__globals__):
            raise ValueError('source handoff lacks candidate coordinator origin')
        loc=caller.f_locals;completion=loc['completion'];request=loc['request'];ctx=loc['ctx']
        index=loc['index'];name=loc['name'];proposal=loc['proposal'];request_sha=loc['request_sha']
        if (_ACTIVE.get() is not completion or getattr(completion,'_source_handoff_ticket',None) is not None
                or completion.callback is not loc['observed'] or completion.candidate_sha256!=candidate_sha
                or ctx.extra.get('diagnostic_source_handoff_request') is not descriptor
                or ctx.extra.get('diagnostic_candidate_batch_request') is not request
                or request.candidates[index][0]!=name or request.candidates[index][1] is not proposal
                or digest(asdict(request))!=request_sha or candidate_sha!=digest({
                    'invocation_id':completion.admission.invocation_id,'batch_sha256':request_sha,
                    'candidate_id':name,'ordinal':index,'proposal':asdict(proposal)})):
            raise ValueError('source handoff candidate coordinator differs')
        completion._check()
        ticket=_CandidateTicket(descriptor,candidate_sha,completion=completion,request=request,
            request_sha=request_sha,ctx=ctx,index=index,name=name,proposal=proposal,_key=_TICKET_KEY)
        completion._source_handoff_ticket=ticket
        return ticket
    finally:
        del caller


def _facts(pins_by_net):
    return tuple((net,tuple((id(pin),pin_fact(pin),pin_source_aliases(pin),freeze_row_values(pin)) for pin in pins))
                 for net,pins in sorted(pins_by_net.items()))


def _metadata(report):
    return freeze_row_values((report.power_nets,report.uncovered,report.snaps,report.synthesized_landings))


class _SourceHandoffScope:
    """Only the current route-cell producer passes actual live operands here."""
    def __init__(self,ticket,*,pin_report,source_pins_by_net,final_row_placement,
                 topology_plan,full_pins_by_net,engine_pins_by_net,world,source_cell,
                 tech,geometry,placement_um,intent,engine,diagnostic_compile,_key):
        caller=inspect.currentframe().f_back
        try:
            if (type(ticket) is not _CandidateTicket or ticket.used or ticket.producer_cap is None
                    or _key is not ticket.producer_cap or caller.f_code is not ticket.producer_code
                    or caller.f_globals is not _route_producer_scope.__globals__):
                raise ValueError('source handoff ticket is invalid or reused')
        finally:del caller
        ticket._check();ticket.used=True
        self._ticket=ticket;self._ticket_anchor=ticket
        self._producer_anchor=(ticket.producer_cap,ticket.producer_code)
        self._pin_report=pin_report;self._source_pins_by_net=source_pins_by_net
        self._final_row_placement=final_row_placement;self._topology_plan=topology_plan
        self._full_pins_by_net=full_pins_by_net;self._engine_pins_by_net=engine_pins_by_net
        self._world=world;self._source_cell=source_cell;self._tech=tech;self._geometry=geometry
        self._placement_um=placement_um;self._intent=intent;self._engine=engine;self._cap=None
        self._active=True;self._observed=False
        self._projected=diagnostic_compile is not None or engine_pins_by_net is not full_pins_by_net
        self._anchors=(pin_report,source_pins_by_net,final_row_placement,topology_plan,
                       full_pins_by_net,engine_pins_by_net,world,source_cell,tech,geometry,
                       placement_um,intent,engine)
        self._row_anchors=tuple(getattr(final_row_placement,name,None) for name in
            ('row_program','row_materialization','row_final_certificate'))
        self._snapshot=None;self._initializing=False;self._cap_anchor=None;self._initial_receipt=None
        self._pair_initialization=None;self._pair_initialization_anchor=None

    def __reduce_ex__(self,protocol):
        raise TypeError('source handoff scope cannot be serialized')

    def _bind(self,cap):
        from clay.layout.route.live_diagnostic import LiveRoutingDiagnostic,observe_prepared_routing
        if type(cap) is not LiveRoutingDiagnostic or self._cap is not None:
            raise ValueError('source handoff cap is foreign or reused')
        caller=inspect.currentframe().f_back
        try:
            producer=caller.f_back
            callback=self._ticket.completion.callback
            if (caller.f_code is not callback.__code__ or caller.f_globals is not callback.__globals__
                    or caller.f_locals.get('cap') is not cap or producer is None
                    or producer.f_code is not observe_prepared_routing.__code__
                    or producer.f_globals is not observe_prepared_routing.__globals__
                    or producer.f_locals.get('capability') is not cap):
                raise ValueError('source handoff lacks actual live capability')
        finally:del caller
        self._cap=cap;self._cap_anchor=cap
        self._initializing=True
        try:self._check()
        finally:self._initializing=False

    def _check(self):
        if (not self._active or self._cap is None or self._cap is not self._cap_anchor
                or self._ticket is not self._ticket_anchor or _ACTIVE_SCOPE.get() is not self
                or self._ticket.scope is not self or self._ticket.producer_cap is not self._producer_anchor[0]
                or self._ticket.producer_code is not self._producer_anchor[1]
                or self._pair_initialization is not self._pair_initialization_anchor):
            raise ValueError('source handoff scope unavailable or foreign')
        self._ticket._check()
        actual=(self._pin_report,self._source_pins_by_net,self._final_row_placement,self._topology_plan,
                self._full_pins_by_net,self._engine_pins_by_net,self._world,self._source_cell,self._tech,
                self._geometry,self._placement_um,self._intent,self._engine)
        if any(a is not b for a,b in zip(actual,self._anchors)):
            raise ValueError('source handoff producer object changed')
        if any(getattr(self._final_row_placement,name,None) is not value for name,value in zip(
                ('row_program','row_materialization','row_final_certificate'),self._row_anchors)):
            raise ValueError('source handoff row producer object changed')
        if self._cap._active:self._cap.check()
        else:self._cap._check_closed_inputs()
        inputs=self._cap._inputs;context=self._cap._context
        if (inputs['source_cell'] is not self._source_cell or inputs['tech'] is not self._tech
                or inputs['geometry'] is not self._geometry or inputs['placement_um'] is not self._placement_um
                or inputs['pins_by_net'] is not self._engine_pins_by_net or inputs['routing_world'] is not self._world
                or inputs['route_engine'] is not self._engine or context['intent'] is not self._intent):
            raise ValueError('source handoff cap operands differ')
        if (type(self._pin_report) is not PinReport or self._pin_report.pins_by_net is not self._full_pins_by_net
                or type(self._source_pins_by_net) is not dict or type(self._topology_plan) is not TopologyPlan
                or self._topology_plan.sha256()!=self._world.topology_provenance_digest):
            raise ValueError('source handoff report, ledger or topology differs')
        if self._projected:
            snapshot=(_facts(self._source_pins_by_net),_facts(self._full_pins_by_net),
                      _metadata(self._pin_report),_row_values(self._final_row_placement),
                      freeze_row_values(self._row_anchors[2]),self._topology_plan.canonical_json(),self._world.sha256())
            if self._initializing:self._snapshot=snapshot
            elif snapshot!=self._snapshot:raise ValueError('source handoff operands changed')
            return
        source=reopen_complete_source(self._source_cell)
        if source['source_sha256']!=self._cap._source_hash or pdk_provenance_digest(self._tech)!=self._world.pdk_provenance_digest:
            raise ValueError('source handoff source or PDK differs')
        for net,pins in self._source_pins_by_net.items():
            available=Counter(map(id,self._full_pins_by_net.get(net,())))
            if any(available[k]<v for k,v in Counter(map(id,pins)).items()):
                raise ValueError('source handoff source Pin identity differs')
        facts=_facts(self._source_pins_by_net)
        if Counter(pin_fact(p) for net,pins in self._full_pins_by_net.items() for p in pins if p.net==net)!=Counter(self._world.pins):
            raise ValueError('source handoff world Pin facts differ')
        snapshot=(facts,_facts(self._full_pins_by_net),_metadata(self._pin_report),_row_values(self._final_row_placement),
                  freeze_row_values(self._row_anchors[2]),self._topology_plan.canonical_json(),self._world.sha256())
        if self._initializing:self._snapshot=snapshot
        elif snapshot!=self._snapshot:raise ValueError('source handoff operands changed')

    def observe(self):
        if self._observed:raise ValueError('source handoff already observed')
        self._check();self._observed=True
        base={'schema':'clay.source-handoff-observation/v1','status':'UNKNOWN_ENGINE_PIN_PROJECTION_UNSUPPORTED'
              if self._projected else 'PARTIAL_SOURCE_HANDOFF',
              'publication_status':'PENDING_TERMINAL_SCOPE_RECHECK',
              'screen':'UNKNOWN_NOT_EXECUTED','site_contact':'UNKNOWN_NOT_CHECKED',
              'access_alternatives_or':'UNKNOWN_NOT_OBSERVED','physical_admission':False,
              'candidate_identity_sha256':self._ticket.candidate_sha,
              'task_id':self._ticket.admission.task_id,'invocation_id':self._ticket.session.invocation_id}
        if self._projected:return base
        from clay.layout.row_terminal_probes import require_routing_row_terminal_probes
        row_program,row_materialization,row_certificate=self._row_anchors
        row_pins=require_routing_row_terminal_probes(tech=self._tech,geometry=self._geometry,
            placement_um=self._placement_um,pins_by_net=self._full_pins_by_net,row_program=row_program,
            row_materialization=row_materialization,row_final_certificate=row_certificate,
            intent=self._intent,source_cell=self._source_cell,
            allow_fixed_multilayer=bool(getattr(self._engine,'consumes_fixed_multilayer_joint_terminals',False)),
            multilayer_engine=self._engine)
        row_facts=tuple(pin_fact(p) for p in row_pins)
        source_roles=[]
        if not getattr(self._source_cell,'instances',()) and not getattr(self._source_cell,'topology',()):
            for device in self._source_cell.devices:
                self._ticket.session.checkpoint()
                source_roles.extend((device.instance_name,role,net) for role,net in device.pins.items())
        hits=defaultdict(list);foreign=[];role_set=set(source_roles)
        for net,pins in self._source_pins_by_net.items():
            self._ticket.session.checkpoint()
            for pin in pins:
                fact=pin_fact(pin);unpack_pin_fact(fact)
                for member,role in pin_source_aliases(pin):
                    if (member,role,net) in role_set:hits[(member,role,net)].append(fact)
                    else:foreign.append((member,role,net))
        rows=[{'member':member,'role':role,'net':net,'pin_facts':hits[(member,role,net)],
               'identity_status':'PIN_FACTS_OBSERVED' if hits[(member,role,net)] else 'UNKNOWN_MISSING_PIN',
               'pin_obligations':'UNKNOWN_NOT_INFERRED','access_alternatives_or':None}
              for member,role,net in sorted(source_roles)]
        role_status=('UNKNOWN_HIERARCHICAL' if not source_roles else
                     'UNKNOWN_DUPLICATE_SOURCE_ROLE' if len(role_set)!=len(source_roles) else 'UNKNOWN_COMPLETENESS')
        row_status=('UNKNOWN_NO_TYPED_ROW' if not row_facts else
                    'UNKNOWN_UNTYPED_ROW_PIN' if any(getattr(p,'terminal_binding',None) is None for p in row_pins)
                    else 'ROW_PROJECTION_REPLAYED')
        base.update(source_sha256=self._cap._source_hash,world_sha256=self._world.sha256(),
            pdk_sha256=self._world.pdk_provenance_digest,topology_sha256=self._topology_plan.sha256(),
            topology=json.loads(self._topology_plan.canonical_json()),
            source_role_status=role_status,
            source_roles=rows,foreign_aliases=sorted(set(foreign)),uncovered=self._pin_report.uncovered,
            row_status=row_status,
            row_required_component_pin_facts=row_facts)
        self._check()
        return json.loads(json.dumps(base,sort_keys=True,separators=(',',':'),allow_nan=False))

    def initialize(self):
        self._check()
        if not self._observed or self._initial_receipt is not None:
            raise ValueError('source handoff must be observed before initialization')
        result=self._cap.initialize_before_groups()
        self._check()
        self._initial_receipt=(object(),self._cap,getattr(self._engine,'_book',None),result)
        return self._initial_receipt[0]

    def consume_initialization(self,receipt):
        self._check()
        record=self._initial_receipt
        if (record is None or record is False or record[0] is not receipt or record[1] is not self._cap
                or record[2] is not getattr(self._engine,'_book',None) or not self._cap._initialized):
            raise ValueError('source handoff initialization receipt is foreign or reused')
        self._initial_receipt=False
        # Pair observations may use only this actual successful initialization,
        # not the attempted-initialization flag or a detached receipt value.
        if record[3].get('status') == 'OBSERVED_EMPTY_PRE_GROUP':
            self._pair_initialization=(object(),self._cap,self._engine,record[2],self._ticket)
            self._pair_initialization_anchor=self._pair_initialization
        return dict(record[3])


def _route_producer_scope(ticket,**operands):
    from clay.layout.route.route_cell import route_cell
    caller=inspect.currentframe().f_back
    try:
        if caller.f_code is not route_cell.__code__ or caller.f_globals is not route_cell.__globals__:
            raise ValueError('source handoff lacks route-cell producer origin')
        loc=caller.f_locals
        names={'pin_report':'pin_report','source_pins_by_net':'source_pins_by_net',
               'final_row_placement':'final_row_placement','topology_plan':'topology_plan',
               'full_pins_by_net':'pins_by_net','engine_pins_by_net':'engine_pins_by_net',
               'world':'routing_world_provenance','source_cell':'cell_def','tech':'tech',
               'geometry':'geometry','placement_um':'placement_um','intent':'intent',
               'engine':'route_engine','diagnostic_compile':'diagnostic_compile'}
        if (type(ticket) is not _CandidateTicket or set(operands)!=set(names) or
                loc.get('diagnostic_source_ticket') is not ticket or
                loc.get('diagnostic_observer') is not ticket.completion.callback or
                any(operands.get(key) is not loc.get(name) for key,name in names.items())):
            raise ValueError('source handoff producer operands differ')
        ticket._check()
        if ticket.used or ticket.producer_cap is not None or _ACTIVE_SCOPE.get() is not None:
            raise ValueError('source handoff producer scope reused or nested')
        producer_cap=object()
        ticket.producer_cap=producer_cap
        @contextmanager
        def producer_scope_context():
            if _ACTIVE_SCOPE.get() is not None:
                raise ValueError('source handoff producer scope cannot nest')
            scope=_SourceHandoffScope(ticket,_key=producer_cap,**operands)
            ticket.scope=scope
            token=_ACTIVE_SCOPE.set(scope)
            try:
                yield scope
            except BaseException as original:
                try:
                    if scope._cap is not None:scope._check()
                except ExecutionStopped as ending:
                    from clay.layout.route.live_diagnostic import DiagnosticRoutingObserved
                    if type(original) is DiagnosticRoutingObserved:raise ending from original
                    record_unwind_failure(original,ending)
                except BaseException as ending:
                    record_unwind_failure(original,ending)
                raise
            else:
                if scope._cap is not None:scope._check()
            finally:
                scope._active=False
                ticket.scope=None
                ticket.producer_cap=None
                ticket.producer_code=None
                _ACTIVE_SCOPE.reset(token)
        ticket.producer_code=producer_scope_context.__wrapped__.__code__
        return producer_scope_context()
    finally:del caller
