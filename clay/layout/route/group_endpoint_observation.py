"""Invocation-scoped observation at a selected group endpoint.

This opt-in diagnostic stops the actual public routing call before its local
route.  It does not change the selected endpoint or grant physical admission.
The event ledger covers only the native planar-domain Python work it names.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import time
import json
import inspect
from hashlib import sha256

from clay.flow.control_stop import CheckedControlStop
from clay.layout.route.live_diagnostic import DiagnosticRoutingObserved


@dataclass(frozen=True)
class GroupEndpointObserver:
    group: str
    net: str
    event_budget: int
    wall_seconds: float
    callback: object

    def __post_init__(self):
        if type(self.group) is not str or not self.group or type(self.net) is not str or not self.net:
            raise ValueError('group and net must be nonempty strings')
        if type(self.event_budget) is not int or self.event_budget <= 0:
            raise ValueError('endpoint event budget must be a positive integer')
        if (type(self.wall_seconds) not in (int, float) or not math.isfinite(self.wall_seconds)
                or self.wall_seconds <= 0 or not callable(self.callback)):
            raise ValueError('endpoint observation needs a finite wall budget and callback')


class _GroupEndpointStop(CheckedControlStop):
    """Private stop whose original unwind must be checked before publication."""


class GroupEndpointObserved(DiagnosticRoutingObserved):
    def __init__(self, digest, status):
        self.digest, self.status = digest, status
        super().__init__('selected group endpoint observation completed')

    def flow_error_data(self):
        return {'reason':'DIAGNOSTIC_OBSERVATION_COMPLETE', 'physical_admission':False,
                'observation_sha256':self.digest, 'observation_status':self.status}


class _ObservationLimit(Exception):
    pass


def record_observation_unwind(original, ending):
    """Keep the primary failure when this opt-in scope cannot unwind cleanly."""
    from clay.flow.control_stop import record_unwind_failure
    try:
        record_unwind_failure(original, ending)
    except BaseException:
        if isinstance(original, CheckedControlStop):
            original._unwind_failures += (ending,)
        try:
            BaseException.add_note(original, 'Ending boundary check also failed: ' + type(ending).__name__)
        except BaseException:
            pass


_ACTIVE = ContextVar('group_endpoint_observation', default=None)


def _producer_identity():
    """Semantic module/qualname identity for observation control boundaries (no __code__/id locks)."""
    from clay.flow import execution_budget, execution_session
    from clay.layout.route import initialized_region_observation as region
    from clay.layout.route import live_diagnostic as live

    symbols = (
        region._canonical,
        region._book_value,
        region._validate_owned_entry,
        live.LiveRoutingDiagnostic.check,
        execution_session.current_session,
        execution_session.ExecutionSession.checkpoint,
        execution_budget.CancellationToken.reason.fget,
        GroupEndpointObserved.flow_error_data,
        CheckedControlStop.require_clean_unwind,
        time.monotonic,
    )
    return tuple(
        (
            fn,
            getattr(fn, '__module__', None),
            getattr(fn, '__qualname__', getattr(fn, '__name__', None)),
        )
        for fn in symbols
    )


class _Authority:
    def __init__(self, request):
        from clay.flow.execution_session import current_session
        from clay.flow.run_admission import current_admission
        self.request = request
        self.request_fields = (request.group, request.net, request.event_budget,
                               request.wall_seconds, request.callback)
        self.session = current_session()
        self.admission = current_admission()
        if (self.session is None or self.admission is None
                or self.session.invocation_id != self.admission.invocation_id
                or self.session.budget is not self.admission.budget):
            raise ValueError('group endpoint observation requires an admitted public invocation')
        self.controls = (self.session.budget, self.session.budget.to_dict(),
                         self.session.cancellation, self.session._wall_clock, self.session._cpu_clock,
                         self.session._wall_start, self.session._cpu_start, self.session._owner_thread,
                         self.admission.budget, self.admission.budget.to_dict(), self.admission._start)
        self.phases = tuple((phase, phase.budget, phase.budget.to_dict(), phase.wall_start, phase.cpu_start)
                            for phase in self.session._phases)
        self.phase_container = self.session._phases
        self.identity = _producer_identity()

    def check(self):
        from clay.flow.execution_session import current_session
        from clay.flow.run_admission import current_admission
        if (current_session() is not self.session or current_admission() is not self.admission
                or _producer_identity() != self.identity
                or (self.request.group, self.request.net, self.request.event_budget,
                    self.request.wall_seconds) != self.request_fields[:4]
                or self.request.callback is not self.request_fields[4]):
            raise ValueError('group endpoint producer or execution authority changed')
        (budget, budget_value, cancellation, wall_clock, cpu_clock, wall_start, cpu_start,
         owner, admission_budget, admission_value, admission_start) = self.controls
        if (self.session.budget is not budget or budget.to_dict() != budget_value
                or self.session.cancellation is not cancellation
                or self.session._wall_clock is not wall_clock or self.session._cpu_clock is not cpu_clock
                or self.session._wall_start != wall_start or self.session._cpu_start != cpu_start
                or self.session._owner_thread != owner or self.admission.budget is not admission_budget
                or admission_budget.to_dict() != admission_value or self.admission._start != admission_start
                or self.session._phases is not self.phase_container
                or len(self.session._phases) < len(self.phases)):
            raise ValueError('group endpoint execution controls changed')
        for actual, (phase, phase_budget, phase_value, phase_wall, phase_cpu) in zip(self.session._phases[:len(self.phases)], self.phases):
            if (actual is not phase or actual.budget is not phase_budget
                    or phase_budget.to_dict() != phase_value or actual.wall_start != phase_wall
                    or actual.cpu_start != phase_cpu):
                raise ValueError('group endpoint phase controls changed')
        self.session.checkpoint()


@dataclass(frozen=True)
class _BoundObserver:
    request: GroupEndpointObserver
    authority: _Authority
    engine: object


def _book_identity(book):
    """Retain identities of the Book containers and rows consumed by routing."""
    references = []
    def value(item):
        if type(item) in (str, int, bool, float, type(None)):
            return (type(item), item)
        if type(item) is dict:
            references.append(item)
            return ('dict', id(item), tuple((value(key), value(child)) for key, child in item.items()))
        if type(item) in (tuple, list):
            references.append(item)
            return (type(item).__name__, id(item), tuple(value(child) for child in item))
        raise TypeError('group endpoint Book contains unsupported references')
    shape = tuple((name, value(getattr(book, name))) for name in ('metal', 'via_pads', 'cuts', 'persistent'))
    return shape, tuple(references)


def _same_book_identity(left, right):
    return (left[0] == right[0] and len(left[1]) == len(right[1])
            and all(actual is original for original, actual in zip(left[1], right[1])))


def bind_source_request(request):
    if type(request) is not GroupEndpointObserver:
        raise TypeError('group endpoint request must have exact type')
    return _Authority(request)


def bind_source_engine(authority, engine):
    from clay.layout.route.engine_gridless import GridlessEngine
    if type(authority) is not _Authority:
        raise TypeError('group endpoint authority must have exact type')
    authority.check()
    if type(engine) is not GridlessEngine:
        raise ValueError('group endpoint observation requires source-selected Gridless')
    return _BoundObserver(authority.request, authority, engine)


class _Scope:
    def __init__(self, bound, capability):
        self.bound = bound
        self.capability = capability
        self.engine = bound.engine
        self.pending = None
        self.seen = False
        self.book = None
        self.active = True
        self.owner = __import__('threading').get_ident()

    def check(self):
        if (not self.active or _ACTIVE.get() is not self
                or getattr(self.engine, '_post_selection_observation', None) is not self
                or __import__('threading').get_ident() != self.owner):
            raise ValueError('group endpoint scope is closed or foreign')
        self.bound.authority.check()
        self.capability.check()
        self.capability._check_endpoint_authority()
        if self.capability._inputs['route_engine'] is not self.engine:
            raise ValueError('group endpoint engine changed')
        for name in ('route', '_watchdog_check'):
            if name in vars(self.engine):
                raise ValueError('group endpoint producer method was replaced')

    def observe(self, engine, port, access, node, selected, portal, interface, book, selector):
        if port.group != self.bound.request.group or port.net != self.bound.request.net:
            return
        started = time.monotonic()
        deadline = started + self.bound.request.wall_seconds
        self.check()
        if self.seen:
            raise ValueError('group endpoint observation repeated')
        self.seen = True
        node, selected = tuple(node), tuple(selected)
        key = (port.net, node)
        contract = self.capability._inputs['route_kwargs'].get('group_route_contract')
        if (engine is not self.engine or book is not self.book or engine._book is not self.book
                or contract is None or not any(port is item for item in contract.ports)
                or not any(access is item for item in port.accesses)
                or tuple(access.xy_nm) != node[:2]
                or engine._tech.resolve_layer_to_gi(access.layer_key) != node[2]
                or tuple(engine._group_route_required_states.get(key, ())) != selected
                or not engine._active_group_route_local_closure):
            raise ValueError('selected group endpoint or Book differs')
        from clay.layout.route.initialized_region_observation import (
            _book_value, _consumer_value, _owned_cache_value, _NATIVE_ENDPOINT_EVENTS,
            _WARM_OWNED_EVENTS, _validate_owned_entry,
        )
        from clay.layout.route.canonical_input_capture import _json_value, route_engine_configuration
        book_references = _book_identity(book)
        state = (_book_value(book), _consumer_value(engine),
                 _owned_cache_value(engine), engine._watchdog_deadline,
                 engine._solve_expansions_total, id(engine._group_route_required_states),
                 id(engine._active_group_route_union_baseline), engine._active_group_source_union_required)
        preflight_seconds = time.monotonic() - started
        count = 0
        counts = {'warm_validation':{}, 'planar_domain':{}}
        phase = 'planar_domain'

        def debit(event):
            nonlocal count
            self.bound.authority.check()
            if (engine is not self.engine or book is not self.book or engine._book is not self.book
                    or type(event) is not str
                    or event not in (_NATIVE_ENDPOINT_EVENTS | _WARM_OWNED_EVENTS)
                    or (phase == 'planar_domain' and event not in _NATIVE_ENDPOINT_EVENTS)
                    or (phase == 'warm_validation' and event not in (_WARM_OWNED_EVENTS | {'index.add.bucket'}))):
                raise ValueError('group endpoint event or Book authority changed')
            if time.monotonic() >= deadline:
                raise _ObservationLimit('UNKNOWN_DIAGNOSTIC_DEADLINE')
            if count >= self.bound.request.event_budget:
                raise _ObservationLimit('UNKNOWN_ENDPOINT_EVENT_BUDGET')
            count += 1
            counts[phase][event] = counts[phase].get(event, 0) + 1

        status = 'OBSERVED_PLANAR_DOMAINS'
        required = free = None
        try:
            if time.monotonic() >= deadline:
                raise _ObservationLimit('UNKNOWN_DIAGNOSTIC_DEADLINE')
            if key in engine._required_wire_endpoints:
                raise _ObservationLimit('UNKNOWN_REQUIRED_WIRE_ENDPOINT')
            cache = engine._owned_terminal_group_indexes
            if type(cache) is not dict:
                raise ValueError('group endpoint native cache is not a dictionary')
            cache_key = (str(port.net), int(node[2]))
            if cache_key in cache:
                phase = 'warm_validation'
                _validate_owned_entry(engine, cache_key, cache[cache_key], diagnostic_work=debit)
                phase = 'planar_domain'
            required = engine._planar_endpoint_domain(port.net, node, diagnostic_work=debit)
            if tuple(required.singleton or ()) != selected:
                raise ValueError('required endpoint domain differs from selection')
            # Match the normal entry's domain.enter event and original watchdog.
            engine._endpoint_diagnostic_step(port.net, debit, 'domain.enter')
            def work(event):
                engine._endpoint_diagnostic_step(port.net, debit, event)
            free = engine._unrequired_planar_endpoint_domain(
                port.net, node, int(engine._manufacturing_grid_nm), work=work, diagnostic_work=debit)
            if len(free.regions) > 128:
                raise _ObservationLimit('UNKNOWN_OUTPUT_LIMIT')
        except _ObservationLimit as error:
            status = str(error)
            required = free = None
        self.check()
        after = (_book_value(book), _consumer_value(engine),
                 _owned_cache_value(engine), engine._watchdog_deadline,
                 engine._solve_expansions_total, id(engine._group_route_required_states),
                 id(engine._active_group_route_union_baseline), engine._active_group_source_union_required)
        if (book is not self.book or engine._book is not self.book
                or not _same_book_identity(book_references, _book_identity(book)) or after != state):
            raise ValueError('group endpoint Book or consumer state changed')
        if time.monotonic() >= deadline:
            status, required, free = 'UNKNOWN_DIAGNOSTIC_DEADLINE', None, None
        book_digest = sha256(json.dumps(state[0], separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.pending = {'schema':'clay.group-endpoint-planar-observation/v1', 'status':status,
                        'group':port.group, 'net':port.net, 'node':node, 'selected':selected,
                        'portal':tuple(portal), 'interface_kind':interface,
                        'source_access':_json_value(access),
                        'selector':selector, 'book_sha256':book_digest,
                        'source_sha256':self.capability._source_hash,
                        'selected_engine_configuration':route_engine_configuration(engine),
                        'task_id':self.bound.authority.admission.task_id,
                        'invocation_id':self.bound.authority.session.invocation_id,
                        'required_domain':None if required is None else _json_value(required),
                        'unrequired_domain':None if free is None else _json_value(free),
                        'event_count':count, 'events':counts, 'event_budget':self.bound.request.event_budget,
                        'preflight_seconds':preflight_seconds,
                        'producer_scope':'cooperative in-process binding; code replacement before the source anchor is UNKNOWN',
                        'budget_scope':'named native endpoint events; excludes input checks, retained reference snapshots, sorting, serialization and native atomics',
                        'source_access_legality':'UNKNOWN', 'via':'UNKNOWN', 'drc':'UNKNOWN',
                        'publication_status':'PENDING_TERMINAL_RECEIPT', 'physical_admission':False}
        raise _GroupEndpointStop()


def observe_selected(engine, port, access, node, selected, portal, interface, book, selector=None):
    scope = _ACTIVE.get()
    if scope is not None:
        from clay.layout.route import engine_gridless
        if type(engine) is not engine_gridless.GridlessEngine or engine is not scope.engine:
            raise ValueError('group endpoint lacks the original selector caller')
        if selector is None:
            selector = (
                'transactional'
                if '_reserve_group_route_landings_once' in vars(engine)
                else 'ordinary'
            )
        scope.observe(engine, port, access, node, selected, portal, interface, book, selector)


def bind_route_book(engine, book):
    scope = _ACTIVE.get()
    if scope is None:
        return
    from clay.layout.route import engine_gridless
    if (type(engine) is not engine_gridless.GridlessEngine
            or engine is not scope.engine
            or type(book) is not engine_gridless._Book
            or engine._book is not book
            or scope.book is not None):
        raise ValueError('group endpoint Book lacks the actual route producer')
    scope.check()
    scope.book = book


def stop_if_unobserved(engine):
    scope = _ACTIVE.get()
    if scope is None:
        return
    scope.check()
    if engine is not scope.engine or engine._book is not scope.book or scope.seen:
        raise ValueError('group endpoint target or engine differs')
    scope.pending = {'schema':'clay.group-endpoint-planar-observation/v1',
                     'status':'UNKNOWN_TARGET_NOT_SELECTED',
                     'group':scope.bound.request.group, 'net':scope.bound.request.net,
                     'task_id':scope.bound.authority.admission.task_id,
                     'invocation_id':scope.bound.authority.session.invocation_id,
                     'source_sha256':scope.capability._source_hash,
                     'selected_engine_configuration':scope.capability.checked_inputs()['selected_engine_configuration'],
                     'selector':'UNKNOWN_NOT_SELECTED',
                     'producer_scope':'cooperative in-process binding; code replacement before the source anchor is UNKNOWN',
                     'publication_status':'PENDING_TERMINAL_RECEIPT',
                     'source_access_legality':'UNKNOWN', 'via':'UNKNOWN', 'drc':'UNKNOWN',
                     'physical_admission':False}
    raise _GroupEndpointStop()


@contextmanager
def observation_scope(bound, capability):
    if _ACTIVE.get() is not None:
        raise ValueError('group endpoint observation cannot nest')
    if '_post_selection_observation' in vars(bound.engine):
        raise ValueError('group endpoint engine already has an observation scope')
    scope = _Scope(bound, capability)
    token = _ACTIVE.set(scope)
    bound.engine._post_selection_observation = scope
    try:
        try:
            yield scope
        except _GroupEndpointStop as stop:
            stop.require_clean_unwind()
            scope.check()
            if scope.pending is None:
                raise ValueError('group endpoint stop lacks an observation')
            payload = json.dumps(scope.pending, sort_keys=True, separators=(',', ':'), allow_nan=False)
            if len(payload.encode('utf8')) > 12000:
                scope.pending = {key:scope.pending[key] for key in
                    ('schema', 'group', 'net', 'task_id', 'invocation_id', 'source_sha256',
                     'selector', 'producer_scope', 'publication_status', 'physical_admission')}
                scope.pending.update(status='UNKNOWN_OUTPUT_LIMIT', source_access_legality='UNKNOWN',
                                     via='UNKNOWN', drc='UNKNOWN')
                payload = json.dumps(scope.pending, sort_keys=True, separators=(',', ':'), allow_nan=False)
            bound.request.callback(json.loads(payload))
            scope.check()
            digest = sha256(payload.encode('utf8')).hexdigest()
            raise GroupEndpointObserved(digest, scope.pending['status']) from stop
    finally:
        scope.active = False
        if getattr(bound.engine, '_post_selection_observation', None) is scope:
            del bound.engine._post_selection_observation
        _ACTIVE.reset(token)
        capability._active = False
