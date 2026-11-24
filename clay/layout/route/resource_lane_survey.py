"""Opt-in bounded observations of actual source endpoint candidate lanes.

This lane-only diagnostic runs in the live setup producer scope.  A candidate
lane is a conservative incidence result, not a site, source contact, routing
solution, or physical admission.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import inspect
import json
import math
import time
from typing import Any

from clay.flow.control_stop import record_unwind_failure
from clay.flow.execution_budget import ExecutionStopped
from clay.flow.execution_session import current_session
from clay.flow.run_admission import current_admission
from clay.layout.place.generated_incumbent import digest
from clay.layout.route import pattern_compile as pc, resource_world as rw
from clay.layout.route.resource_access_diagnostic import _component_key, _json_bytes, _lane_key, _row_count
from clay.layout.route.resource_meter import DiagnosticResourceMeter, ResourceMeterStopped, bind_budget
from clay.layout.route.source_handoff import _ACTIVE_SCOPE, _SourceHandoffScope
from clay.layout.route.terminal_identity import pin_fact, pin_source_aliases


@dataclass(frozen=True, slots=True)
class ResourceLaneSurveyRequest:
    """Independent diagnostic limits; they do not alter the selected engine."""

    wall_seconds: float
    per_net_operations: int
    max_components_per_net: int
    max_lanes_per_component: int
    max_output_rows: int
    max_output_bytes: int
    schema: str = 'clay.resource-lane-survey-request/v1'

    def __post_init__(self) -> None:
        if self.schema != 'clay.resource-lane-survey-request/v1':
            raise ValueError('unsupported resource lane survey request schema')
        if (isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float))
                or not math.isfinite(self.wall_seconds) or not 0 < self.wall_seconds <= 90):
            raise ValueError('wall_seconds must be finite, positive and at most 90')
        for name in ('per_net_operations', 'max_components_per_net', 'max_lanes_per_component',
                     'max_output_rows', 'max_output_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be a positive exact integer')
        if self.per_net_operations > 100_000:
            raise ValueError('per_net_operations exceeds the diagnostic maximum')
        if self.max_components_per_net > 16 or self.max_lanes_per_component > 16:
            raise ValueError('component or lane limit exceeds the diagnostic maximum')
        if self.max_output_bytes < 2048:
            raise ValueError('max_output_bytes must allow a bounded UNKNOWN envelope')


class _SurveyWall(RuntimeError):
    pass


def observe_lane_survey(*, scope: Any, cap: Any, source_value: Any,
                        request: ResourceLaneSurveyRequest, meter: DiagnosticResourceMeter,
                        ledger: Any, conductors: Any, index: Any, boundary: Any) -> dict[str, Any]:
    """Observe actual endpoint components without invoking a landing-site query."""
    from clay.layout.route.resource_setup_diagnostic import run_resource_setup

    started = time.monotonic()
    if type(request) is not ResourceLaneSurveyRequest:
        raise TypeError('exact resource lane survey request required')
    request.__post_init__()
    caller = inspect.currentframe().f_back
    session, admission = current_session(), current_admission()
    try:
        local = caller.f_locals
        if (caller.f_code is not run_resource_setup.__code__ or caller.f_globals is not run_resource_setup.__globals__
                or type(scope) is not _SourceHandoffScope or _ACTIVE_SCOPE.get() is not scope
                or local.get('scope') is not scope or local.get('cap') is not cap
                or local.get('source_value') is not source_value or local.get('lane_survey_request') is not request
                or local.get('meter') is not meter or local.get('ledger') is not ledger
                or local.get('conductors') is not conductors or local.get('index') is not index
                or local.get('boundary') is not boundary or not callable(boundary)
                or type(meter) is not DiagnosticResourceMeter or meter._scope is not scope
                or meter._per_net_max != request.per_net_operations or scope._cap is not cap
                or not cap._active or scope._projected or session is None or admission is None
                or session is not meter._session or admission is not meter._admission
                or getattr(scope, '_resource_lane_survey_consumed', False)):
            raise ValueError('resource lane survey requires the current verified setup caller')
        scope._resource_lane_survey_consumed = True
        setup_identity = local.get('identity')
        if type(setup_identity) is not str or len(setup_identity) != 64:
            raise ValueError('resource lane survey setup identity is unavailable')
        request_sha = digest(asdict(request))
        identity = digest({'schema': 'clay.resource-lane-survey-scope/v1',
                           'resource_setup_scope_identity_sha256': setup_identity,
                           'resource_lane_survey_request_sha256': request_sha})
        deadline = started + float(request.wall_seconds)
        last_meter = None
        phase = 'LANE_ASSOCIATION'
        join_complete = False
        rows: dict[str, Any] = {}
        tasks_by_net: dict[str, deque] = {}
        stopped_nets: dict[str, Any] = {}
        active_task = None
        active_stage = None
        result = {'schema': 'clay.resource-lane-survey-observation/v1',
                  'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                  'request_sha256': request_sha, 'scope_identity_sha256': identity,
                  'resource_setup_scope_identity_sha256': setup_identity,
                  'candidate_identity_sha256': scope._ticket.candidate_sha,
                  'invocation_id': admission.invocation_id, 'source_sha256': source_value.get('source_sha256'),
                  'world_sha256': scope._world.sha256(), 'pdk_sha256': scope._world.pdk_provenance_digest,
                  'topology_sha256': scope._topology_plan.sha256(), 'limits': asdict(request),
                  'source_denominator_status': 'UNKNOWN_SOURCE_DENOMINATOR',
                  'source_role_status': source_value.get('source_role_status'),
                  'source_contact': 'UNKNOWN_NOT_CHECKED', 'site': 'UNKNOWN_NOT_EXECUTED',
                  'joint_route': 'UNKNOWN_NOT_EXECUTED',
                  'fairness_scope': 'component_lane_queries_after_shared_source_join',
                  'unmetered_work': ['source_scope_checks', 'context_geometry_index',
                      'semantic_key_and_sort_internals', 'serialization_and_hashing']}

        def check(*, wall: bool = True) -> None:
            boundary(enforce_wall=False)
            if (scope._resource_lane_survey_consumed is not True
                    or caller.f_locals.get('lane_survey_request') is not request
                    or digest(asdict(request)) != request_sha):
                raise ValueError('resource lane survey request or one-shot state changed')
            session.checkpoint()
            if wall and time.monotonic() >= deadline:
                raise _SurveyWall('resource lane survey cooperative wall exhausted')

        def capture_meter() -> None:
            nonlocal last_meter
            check()
            value = meter.snapshot()
            check()
            last_meter = value

        def finish(status: str, *, stop: dict[str, Any] | None = None) -> dict[str, Any]:
            check(wall=stop is None)
            value = dict(result)
            value.update(status=status, source_join_complete=join_complete,
                         net_rows=[rows[net] for net in sorted(rows)], per_net_stops=stopped_nets,
                         meter_snapshot=last_meter, elapsed_wall_seconds_lower_bound=time.monotonic()-started)
            if stop is not None:
                value['stop'] = stop
            encoded = _json_bytes(value)
            check(wall=stop is None)
            if _row_count(value, request.max_output_rows) > request.max_output_rows or len(encoded) > request.max_output_bytes:
                value = {key: value[key] for key in ('schema', 'publication_status', 'physical_admission',
                    'request_sha256', 'scope_identity_sha256', 'resource_setup_scope_identity_sha256',
                    'candidate_identity_sha256', 'invocation_id')}
                value.update(status='UNKNOWN_OUTPUT_LIMIT', source_denominator_status='UNKNOWN_SOURCE_DENOMINATOR',
                    source_contact='UNKNOWN_NOT_CHECKED', site='UNKNOWN_NOT_EXECUTED', joint_route='UNKNOWN_NOT_EXECUTED',
                    output={'max_rows': request.max_output_rows, 'max_bytes': request.max_output_bytes,
                            'complete_payload_bytes': len(encoded)})
                if stop is not None:
                    value['underlying_stop'] = {'type': stop['type'], 'phase': stop['phase'],
                                                'reason': stop.get('reason')}
                encoded = _json_bytes(value)
                if len(encoded) > request.max_output_bytes:
                    raise ValueError('resource lane survey UNKNOWN envelope exceeds declared output limit')
            check(wall=stop is None)
            return json.loads(encoded)

        try:
            check()
            meter.check_boundary()
            meter.require_ledger(ledger)
            meter.require_incidence(index, ledger, conductors)
            roles = source_value['source_roles']
            nets = sorted({r['net'] for r in roles} | set(scope._source_cell.ports)
                          | set(scope._engine_pins_by_net) | set(scope._world.required_nets)
                          | {d.net for d in scope._topology_plan.demands})
            if len(nets) + len(roles) > request.max_output_rows:
                return finish('UNKNOWN_OUTPUT_LIMIT')
            role_index = defaultdict(list)
            for net in nets:
                rows[net] = {'net': net, 'status': 'UNKNOWN_NOT_STARTED', 'source_roles': [],
                    'endpoint_incidence': [], 'lane_components': [], 'remaining_components': None,
                    'deferred_components': 0, 'component_semantic_ties': 0}
            for role in roles:
                record = {'member': role['member'], 'role': role['role'], 'net': role['net'],
                          'source_identity_status': role['identity_status']}
                rows[role['net']]['source_roles'].append(record)
                role_index[(role['member'], role['role'], role['net'])].append(record)
            for records in role_index.values():
                if len(records) != 1:
                    for record in records:
                        record['source_identity_status'] = 'UNKNOWN_DUPLICATE_SOURCE_ROLE'

            pending_endpoints = []
            task_index: dict[tuple[str, str], Any] = {}
            with meter.scope(phase):
                endpoint_by_fact = {}
                for endpoint in index.endpoints:
                    session.checkpoint(); meter.reserve('lane_survey', 'endpoint_index')
                    if endpoint.pin_access_fact in endpoint_by_fact:
                        raise ValueError('resource lane endpoint fact is ambiguous')
                    endpoint_by_fact[endpoint.pin_access_fact] = endpoint
                entries = []
                for net, pins in sorted(scope._source_pins_by_net.items()):
                    for pin in pins:
                        session.checkpoint(); meter.reserve('lane_survey', 'source_pin_fact')
                        fact = pin_fact(pin)
                        if fact[0] != net:
                            raise ValueError('resource lane source Pin net differs from its mapping')
                        entries.append((net, fact, pin_source_aliases(pin)))
                counts = Counter((net, fact) for net, fact, _ in entries)
                for net, fact, aliases in entries:
                    session.checkpoint(); meter.reserve('lane_survey', 'source_pin_join')
                    endpoint = endpoint_by_fact.get(fact)
                    refs = conductors.refs_for_pin(fact)
                    if endpoint is not None and (not refs or not set(refs).issubset(endpoint.access_component_refs)):
                        raise ValueError('resource lane endpoint and conductor bindings differ')
                    matched = sorted((member, role, net) for member, role in aliases
                                     if len(role_index.get((member, role, net), ())) == 1)
                    status = ('UNKNOWN_DUPLICATE_PIN_FACT' if counts[(net, fact)] != 1 else
                              'UNKNOWN_MISSING_ENDPOINT' if endpoint is None else
                              'ENDPOINT_INCIDENCE_OBSERVED')
                    pending_endpoints.append((net, {'pin_fact': fact, 'status': status,
                        'source_roles': matched, 'source_role_match_status': (
                            'SOURCE_ROLE_MATCHED' if matched else 'UNKNOWN_NO_SOURCE_ROLE'),
                        'access_component_refs': [] if endpoint is None else endpoint.access_component_refs,
                        'conductor_component_refs': [] if endpoint is None else endpoint.conductor_component_refs,
                        'topology_eligibility_status': 'UNKNOWN_NOT_CHECKED'}))
                    if status != 'ENDPOINT_INCIDENCE_OBSERVED' or not matched:
                        continue
                    for ref in endpoint.access_component_refs:
                        component = conductors.component(ref)
                        if component.net != net or component.world_digest != scope._world.sha256():
                            raise ValueError('resource lane component belongs to another net or world')
                        task = task_index.setdefault((net, ref), {'net': net, 'component': component,
                                                                   'pin_facts': set(), 'roles': set()})
                        task['pin_facts'].add(fact)
                        task['roles'].update(matched)
            capture_meter()
            for net, detail in pending_endpoints:
                rows[net]['endpoint_incidence'].append(detail)
            join_complete = True
            for task in task_index.values():
                tasks_by_net.setdefault(task['net'], deque()).append(task)
            for net, tasks in tasks_by_net.items():
                grouped = defaultdict(list)
                for task in tasks:
                    key = (_component_key(task['component']),
                           tuple(sorted(_json_bytes(f) for f in task['pin_facts'])))
                    grouped[key].append(task)
                tasks_by_net[net] = deque(grouped[key][0] for key in sorted(grouped) if len(grouped[key]) == 1)
                ties = sum(len(values) for values in grouped.values() if len(values) > 1)
                rows[net]['component_semantic_ties'] = ties
                rows[net]['deferred_components'] = ties
            for net, row in rows.items():
                row['remaining_components'] = len(tasks_by_net.get(net, ())) + row['deferred_components']
                if not tasks_by_net.get(net):
                    row['status'] = ('UNKNOWN_AMBIGUOUS_COMPONENT_ORDER' if row['deferred_components']
                                     else 'UNKNOWN_NO_ELIGIBLE_ENDPOINT')
            if not any(tasks_by_net.values()):
                return finish('UNKNOWN_AMBIGUOUS_COMPONENT_ORDER' if any(
                    row['deferred_components'] for row in rows.values()) else 'UNKNOWN_NO_ELIGIBLE_ENDPOINT')

            phase = 'LANE_INDEX'
            with meter.scope(phase):
                context = rw._prepare_exact_landing_site_query_context(
                    ledger, scope._world, scope._tech, conductors, resource_meter=meter)
                lane_index = pc._BudgetedLaneLayerIndex(index)
                incidence_budget = bind_budget(pc._IncidenceBudget(pc.MAX_INCIDENCE_EVALUATIONS), meter, 'incidence')
                lane_index._build(budget=incidence_budget)
                access = pc._ExactLazyLandingAccess(ledger, scope._world, scope._tech, conductors,
                                                    index, context, lane_index)
            capture_meter()
            phase = 'LANE_NETS'
            queried = Counter()
            while any(tasks_by_net.values()):
                for net in sorted(tasks_by_net):
                    queue = tasks_by_net[net]
                    if not queue:
                        continue
                    row = rows[net]
                    if queried[net] >= request.max_components_per_net:
                        row['deferred_components'] += len(queue)
                        row['remaining_components'] = row['deferred_components']
                        row['status'] = 'PARTIAL_COMPONENT_LIMIT'
                        queue.clear()
                        continue
                    task = queue.popleft()
                    active_task = task
                    active_stage = 'COMPONENT_LANES'
                    try:
                        check()
                        with meter.scope('LANE_NET', net=net):
                            lanes = access.component_lanes(task['component'].component_ref, budget=incidence_budget)
                            grouped = defaultdict(list)
                            retained_count = 0
                            for lane in lanes:
                                session.checkpoint(); meter.reserve('lane_survey', 'lane_filter')
                                if lane.owner_net in (None, net):
                                    grouped[_lane_key(lane)].append(lane)
                                    retained_count += 1
                            unambiguous = [grouped[key][0] for key in sorted(grouped) if len(grouped[key]) == 1]
                            pending = {'component_ref': task['component'].component_ref,
                                'pin_facts': sorted(task['pin_facts'], key=_json_bytes),
                                'source_roles': sorted(task['roles']), 'lanes_discovered': len(lanes),
                                'lanes_owned_or_free': retained_count,
                                'lane_semantic_ties': sum(len(v) for v in grouped.values() if len(v) > 1),
                                'lanes_retained': [lane.canonical_payload() for lane in unambiguous[:request.max_lanes_per_component]],
                                'lane_truncated': len(unambiguous) > request.max_lanes_per_component,
                                'status': 'OBSERVED_BOUNDED_CANDIDATE_LANES'}
                        active_stage = 'COMPONENT_RECHECK'
                        capture_meter()
                        row['lane_components'].append(pending)
                        queried[net] += 1
                        row['remaining_components'] = len(queue) + row['deferred_components']
                        row['status'] = ('PARTIAL_AMBIGUOUS_COMPONENT_ORDER' if row['deferred_components']
                                         else 'OBSERVED_BOUNDED_LANES')
                        active_task = None
                    except ResourceMeterStopped as exc:
                        if exc.reason != 'METER_NET_LIMIT' or exc.net != net:
                            raise
                        row['status'] = 'UNKNOWN_PER_NET_LIMIT'
                        row['remaining_components'] = len(queue) + 1 + row['deferred_components']
                        stopped_nets[net] = {'reason': exc.reason, 'phase': exc.phase,
                            'used_global_operations': exc.used,
                            'last_snapshot_net_operations': (None if last_meter is None else
                                last_meter.get('by_net', {}).get(net, 0)), 'interrupted_stage': active_stage}
                        queue.clear()
                        active_task = None
            return finish('PARTIAL_BOUNDED_LANE_SURVEY')
        except ExecutionStopped:
            raise
        except (ResourceMeterStopped, _SurveyWall, rw.RoutingResourceEnumerationLimitError,
                pc._IncidenceBudgetExceeded) as exc:
            if getattr(exc, '__notes__', ()) or getattr(exc.__cause__, '__notes__', ()):
                raise
            try:
                boundary(enforce_wall=False)
            except ExecutionStopped:
                raise
            except BaseException as ending:
                try:
                    record_unwind_failure(exc, ending)
                except BaseException:
                    pass
                raise exc
            for net, queue in tasks_by_net.items():
                if queue:
                    rows[net]['remaining_components'] = len(queue) + rows[net]['deferred_components']
            if active_task is not None:
                net = active_task['net']
                row = rows[net]
                row['remaining_components'] = len(tasks_by_net[net]) + 1 + row['deferred_components']
                row['status'] = 'UNKNOWN_INTERRUPTED_COMPONENT'
                row['interrupted_component'] = {'component_ref': active_task['component'].component_ref,
                                                  'stage': active_stage, 'lanes_discovered': None}
            status = ('UNKNOWN_METER_LIMIT' if isinstance(exc, ResourceMeterStopped) else
                      'UNKNOWN_SURVEY_WALL' if isinstance(exc, _SurveyWall) else 'UNKNOWN_LOCAL_LIMIT')
            stop = {'type': type(exc).__name__, 'phase': phase, 'message': str(exc)}
            if isinstance(exc, ResourceMeterStopped):
                stop.update(reason=exc.reason, meter_phase=exc.phase, net=exc.net,
                            used_global_operations=exc.used)
            try:
                return finish(status, stop=stop)
            except ExecutionStopped:
                raise
            except BaseException as ending:
                try:
                    record_unwind_failure(exc, ending)
                except BaseException:
                    pass
                raise exc
    finally:
        del caller
