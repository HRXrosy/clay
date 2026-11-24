"""Bounded lane-first, then legacy site observations in one live scope."""
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
from clay.layout.route.engine_gridless import NegotiationWatchdogStop
from clay.layout.route.resource_access_diagnostic import _component_key, _json_bytes, _lane_key, _row_count, _site_key
from clay.layout.route.resource_meter import (DiagnosticResourceMeter, ResourceMeterStopped, bind_budget,
                                              lane_site_boundary_transaction)
from clay.layout.route.resource_plan import _landing_site_payload
from clay.layout.route.source_handoff import _ACTIVE_SCOPE, _SourceHandoffScope
from clay.layout.route.terminal_identity import pin_fact, pin_source_aliases


@dataclass(frozen=True, slots=True)
class ResourceLaneSiteRequest:
    """Independent diagnostic limits; they do not alter the selected engine."""

    wall_seconds: float
    lane_wall_seconds: float
    site_wall_seconds: float
    per_net_operations: int
    max_components_per_net: int
    max_lanes_per_component: int
    max_site_queries_per_net: int
    max_sites_per_lane: int
    max_output_rows: int
    max_output_bytes: int
    schema: str = 'clay.resource-lane-site-request/v1'

    def __post_init__(self) -> None:
        if self.schema != 'clay.resource-lane-site-request/v1':
            raise ValueError('unsupported resource lane site request schema')
        for name, maximum in (('wall_seconds', 90), ('lane_wall_seconds', 90), ('site_wall_seconds', 25)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= maximum):
                raise ValueError(name + ' must be finite, positive and within its diagnostic maximum')
        if self.lane_wall_seconds + self.site_wall_seconds > self.wall_seconds:
            raise ValueError('lane and site walls must fit within the total wall')
        for name in ('per_net_operations', 'max_components_per_net', 'max_lanes_per_component',
                     'max_site_queries_per_net', 'max_sites_per_lane', 'max_output_rows', 'max_output_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be a positive exact integer')
        if self.per_net_operations > 100_000:
            raise ValueError('per_net_operations exceeds the diagnostic maximum')
        if self.max_components_per_net > 16 or self.max_lanes_per_component > 16:
            raise ValueError('component or lane limit exceeds the diagnostic maximum')
        if self.max_site_queries_per_net > 16 or self.max_sites_per_lane > 8:
            raise ValueError('site query or retained site limit exceeds the diagnostic maximum')
        if self.max_output_bytes < 2048:
            raise ValueError('max_output_bytes must allow a bounded UNKNOWN envelope')


class _LaneSiteWall(RuntimeError):
    pass


class _IterativePairWall(RuntimeError):
    pass


def _pin_lane_priority(lane: Any, pin_facts: Any) -> tuple:
    """Rank by perpendicular distance to an associated source Pin."""
    coordinate = 3 if lane.preferred_axis == 'V' else 4
    return (min(abs(lane.center_nm - fact[coordinate]) for fact in pin_facts),
            _lane_key(lane))


def _diverse_lane_centers(ordered_lanes: Any) -> list[Any]:
    """Keep distance order within unique centers and their later backups."""
    seen = set()
    primary, backup = [], []
    for lane in ordered_lanes:
        key = (lane.layer_n, lane.preferred_axis, lane.center_nm)
        if key in seen:
            backup.append(lane)
        else:
            seen.add(key)
            primary.append(lane)
    return primary + backup


def observe_lane_site(*, scope: Any, cap: Any, source_value: Any,
                        request: ResourceLaneSiteRequest, meter: DiagnosticResourceMeter,
                        ledger: Any, conductors: Any, index: Any, boundary: Any,
                        pair_request: Any = None) -> dict[str, Any]:
    """Observe source lanes first, then legacy sites using the same producer."""
    from clay.layout.route.resource_setup_diagnostic import run_resource_setup

    started = time.monotonic()
    if type(request) is not ResourceLaneSiteRequest:
        raise TypeError('exact resource lane site request required')
    request.__post_init__()
    if pair_request is not None:
        from clay.layout.route.resource_pair_transaction import ResourcePairRequest
        if type(pair_request) is not ResourcePairRequest:
            raise TypeError('exact resource pair request required')
        pair_request.__post_init__()
    caller = inspect.currentframe().f_back
    session, admission = current_session(), current_admission()
    try:
        local = caller.f_locals
        if (caller.f_code is not run_resource_setup.__code__ or caller.f_globals is not run_resource_setup.__globals__
                or type(scope) is not _SourceHandoffScope or _ACTIVE_SCOPE.get() is not scope
                or local.get('scope') is not scope or local.get('cap') is not cap
                or local.get('source_value') is not source_value or local.get('lane_site_request') is not request
                or local.get('pair_request') is not pair_request
                or local.get('meter') is not meter or local.get('ledger') is not ledger
                or local.get('conductors') is not conductors or local.get('index') is not index
                or local.get('boundary') is not boundary or not callable(boundary)
                or type(meter) is not DiagnosticResourceMeter or meter._scope is not scope
                or meter._per_net_max != request.per_net_operations or scope._cap is not cap
                or not cap._active or scope._projected or session is None or admission is None
                or session is not meter._session or admission is not meter._admission
                or getattr(scope, '_resource_lane_site_consumed', False)):
            raise ValueError('resource lane site requires the current verified setup caller')
        scope._resource_lane_site_consumed = True
        setup_identity = local.get('identity')
        if type(setup_identity) is not str or len(setup_identity) != 64:
            raise ValueError('resource lane site setup identity is unavailable')
        request_sha = digest(asdict(request))
        identity = digest({'schema': 'clay.resource-lane-site-scope/v1',
                           'resource_setup_scope_identity_sha256': setup_identity,
                           'resource_lane_site_request_sha256': request_sha})
        pair_request_sha = digest(asdict(pair_request)) if pair_request is not None else None
        pair_identity = (digest({'schema': 'clay.resource-pair-scope/v1',
            'resource_lane_site_scope_identity_sha256': identity,
            'resource_pair_request_sha256': pair_request_sha}) if pair_request is not None else None)
        def pair_unexecuted(*, output_limited=False):
            return {'schema': 'clay.resource-pair-observation/v1',
                    'status': 'UNKNOWN_PAIR_OUTPUT_LIMIT' if output_limited else 'UNKNOWN_NOT_EXECUTED',
                    'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                    'request_sha256': pair_request_sha, 'scope_identity_sha256': pair_identity,
                    'source_contact': 'UNKNOWN_NOT_CHECKED', 'joint_route': 'UNKNOWN_NOT_EXECUTED'}
        deadline = started + float(request.wall_seconds)
        lane_deadline = min(deadline, started + float(request.lane_wall_seconds))
        site_deadline = None
        last_meter = None
        last_meter_phase = None
        phase = 'LANE_SITE_ASSOCIATION'
        lane_phase_status = 'UNKNOWN_NOT_STARTED'
        site_phase_status = 'UNKNOWN_NOT_STARTED'
        join_complete = False
        rows: dict[str, Any] = {}
        tasks_by_net: dict[str, deque] = {}
        stopped_nets: dict[str, Any] = {}
        active_task = None
        active_stage = None
        site_tasks_by_net: dict[str, deque] = {}
        pair_candidates: list[Any] = []
        pair_iteration = None
        site_active = False
        result = {'schema': 'clay.resource-lane-site-observation/v1',
                  'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                  'request_sha256': request_sha, 'scope_identity_sha256': identity,
                  'resource_setup_scope_identity_sha256': setup_identity,
                  'candidate_identity_sha256': scope._ticket.candidate_sha,
                  'invocation_id': admission.invocation_id, 'source_sha256': source_value.get('source_sha256'),
                  'world_sha256': scope._world.sha256(), 'pdk_sha256': scope._world.pdk_provenance_digest,
                  'topology_sha256': scope._topology_plan.sha256(), 'limits': asdict(request),
                  'source_denominator_status': 'UNKNOWN_SOURCE_DENOMINATOR',
                  'source_role_status': source_value.get('source_role_status'),
                  'source_contact': 'UNKNOWN_NOT_CHECKED', 'n2_via_legality': 'UNKNOWN_NOT_CHECKED',
                  'joint_route': 'UNKNOWN_NOT_EXECUTED',
                  'site_denominator_status': 'UNKNOWN_UNSAMPLED_LANES_AND_COMPONENTS',
                  'lane_selection_strategy': {
                      'priority': 'MIN_ASSOCIATED_SOURCE_PIN_CROSS_AXIS_DISTANCE_NM',
                      'cross_axis': 'V_X_H_Y', 'tie_break': 'WORLD_INDEPENDENT_LANE_KEY',
                      'scope': 'ORDERING_ONLY_NOT_LEGALITY',
                      **({'center_diversity': 'UNIQUE_LAYER_AXIS_CENTER_THEN_SAME_CENTER_BACKUPS'}
                         if pair_request is not None and pair_request.schema == 'clay.resource-pair-request/v4-iterative'
                         else {}),
                      'truncated_lanes': 'UNKNOWN_NOT_QUERIED'},
                  'fairness_scope': ('lane_then_site_round_robin_defer_passed_net_until_pair_rejection'
                      if pair_request is not None and pair_request.schema == 'clay.resource-pair-request/v4-iterative'
                      else 'lane_then_site_round_robin_after_shared_source_join'),
                  'unmetered_work': ['source_scope_checks', 'context_geometry_index',
                      'semantic_key_and_sort_internals', 'serialization_and_hashing']}
        if pair_request is not None:
            result['resource_pair'] = pair_unexecuted()
            result['meter_snapshot_scope'] = ('LAST_SUCCESSFUL_LANE_SITE_SNAPSHOT'
                if pair_request.schema == 'clay.resource-pair-request/v4-iterative'
                else 'THROUGH_LANE_SITE_BEFORE_PAIR')

        def check_wall(*, wall: bool = True) -> None:
            session.checkpoint()
            if pair_iteration is not None and not pair_iteration.finished and time.monotonic() >= pair_iteration.deadline:
                raise _IterativePairWall('iterative pair total wall exhausted')
            if (wall and phase != 'RESOURCE_PAIR'
                    and time.monotonic() >= min(deadline, site_deadline if site_deadline is not None else lane_deadline)):
                raise _LaneSiteWall('resource lane site cooperative wall exhausted')
            if (wall and phase == 'LANE_SITE_SITES'
                    and pair_iteration is not None and not pair_iteration.finished):
                pair_iteration.engine._watchdog_check(
                    active_task['net'] if active_task is not None else '<iterative>',
                    'resource_pair_site_resume')

        def check(*, wall: bool = True) -> None:
            boundary(enforce_wall=False)
            if (scope._resource_lane_site_consumed is not True
                    or caller.f_locals.get('lane_site_request') is not request
                    or caller.f_locals.get('pair_request') is not pair_request
                    or digest(asdict(request)) != request_sha
                    or (pair_request is not None and digest(asdict(pair_request)) != pair_request_sha)):
                raise ValueError('resource lane site request or one-shot state changed')
            check_wall(wall=wall)

        def capture_meter() -> None:
            nonlocal last_meter, last_meter_phase
            check_wall()
            value = meter.snapshot()
            check()
            last_meter = value
            last_meter_phase = phase

        def finish(status: str, *, stop: dict[str, Any] | None = None) -> dict[str, Any]:
            check(wall=stop is None)
            value = dict(result)
            value.update(status=status, source_join_complete=join_complete,
                         lane_phase_status=lane_phase_status, site_phase_status=site_phase_status,
                         net_rows=[rows[net] for net in sorted(rows)], per_net_stops=stopped_nets,
                         meter_snapshot=last_meter, elapsed_wall_seconds_lower_bound=time.monotonic()-started)
            if pair_request is not None and pair_request.schema == 'clay.resource-pair-request/v4-iterative':
                value['meter_snapshot_scope'] = ('LAST_SUCCESSFUL_LANE_SITE_SNAPSHOT'
                    if last_meter is not None else 'NO_SUCCESSFUL_METER_SNAPSHOT')
                value['meter_snapshot_phase'] = last_meter_phase
                value['work_after_meter_snapshot'] = ('UNKNOWN_SINCE_LAST_SUCCESSFUL_SNAPSHOT'
                    if last_meter is not None else 'UNKNOWN_SINCE_LANE_SITE_ENTRY')
            if stop is not None:
                value['stop'] = stop
            encoded = _json_bytes(value)
            check_wall(wall=stop is None)
            if _row_count(value, request.max_output_rows) > request.max_output_rows or len(encoded) > request.max_output_bytes:
                value = {key: value[key] for key in ('schema', 'publication_status', 'physical_admission',
                    'request_sha256', 'scope_identity_sha256', 'resource_setup_scope_identity_sha256',
                    'candidate_identity_sha256', 'invocation_id', 'lane_selection_strategy')}
                value.update(status='UNKNOWN_OUTPUT_LIMIT', source_denominator_status='UNKNOWN_SOURCE_DENOMINATOR',
                    source_contact='UNKNOWN_NOT_CHECKED', n2_via_legality='UNKNOWN_NOT_CHECKED',
                    joint_route='UNKNOWN_NOT_EXECUTED',
                    output={'max_rows': request.max_output_rows, 'max_bytes': request.max_output_bytes,
                            'complete_payload_bytes': len(encoded)})
                if pair_request is not None:
                    value['resource_pair'] = pair_unexecuted(output_limited=True)
                if stop is not None:
                    value['underlying_stop'] = {'type': stop['type'], 'phase': stop['phase'],
                                                'reason': stop.get('reason')}
                encoded = _json_bytes(value)
                if len(encoded) > request.max_output_bytes:
                    raise ValueError('resource lane site UNKNOWN envelope exceeds declared output limit')
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
                    'deferred_components': 0, 'component_semantic_ties': 0,
                    'site_queries': [], 'site_status': 'UNKNOWN_NOT_STARTED', 'remaining_site_tasks': None,
                    'ambiguous_pin_site_tasks': 0,
                    'site_task_scope': 'SAMPLED_COMPLETED_LANES_ONLY'}
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
            lane_phase_status = 'RUNNING'
            with meter.scope(phase):
                endpoint_by_fact = {}
                for endpoint in index.endpoints:
                    session.checkpoint(); meter.reserve('lane_site', 'endpoint_index')
                    if endpoint.pin_access_fact in endpoint_by_fact:
                        raise ValueError('resource lane endpoint fact is ambiguous')
                    endpoint_by_fact[endpoint.pin_access_fact] = endpoint
                entries = []
                for net, pins in sorted(scope._source_pins_by_net.items()):
                    for pin in pins:
                        session.checkpoint(); meter.reserve('lane_site', 'source_pin_fact')
                        fact = pin_fact(pin)
                        if fact[0] != net:
                            raise ValueError('resource lane source Pin net differs from its mapping')
                        entries.append((net, fact, pin_source_aliases(pin)))
                counts = Counter((net, fact) for net, fact, _ in entries)
                for net, fact, aliases in entries:
                    session.checkpoint(); meter.reserve('lane_site', 'source_pin_join')
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
                            'pin_facts': set(), 'roles': set(), 'endpoints': {}})
                        task['pin_facts'].add(fact)
                        task['roles'].update(matched)
                        task['endpoints'][fact] = (endpoint, tuple(matched))
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
                lane_phase_status = 'UNKNOWN_NO_ELIGIBLE_ENDPOINT'
                return finish('UNKNOWN_AMBIGUOUS_COMPONENT_ORDER' if any(
                    row['deferred_components'] for row in rows.values()) else 'UNKNOWN_NO_ELIGIBLE_ENDPOINT')

            phase = 'LANE_SITE_INDEX'
            with meter.scope(phase):
                context = rw._prepare_exact_landing_site_query_context(
                    ledger, scope._world, scope._tech, conductors, resource_meter=meter)
                lane_index = pc._BudgetedLaneLayerIndex(index)
                incidence_budget = bind_budget(pc._IncidenceBudget(pc.MAX_INCIDENCE_EVALUATIONS), meter, 'incidence')
                lane_index._build(budget=incidence_budget)
                access = pc._ExactLazyLandingAccess(ledger, scope._world, scope._tech, conductors,
                                                    index, context, lane_index)
            capture_meter()
            phase = 'LANE_SITE_LANES'
            queried = Counter()
            completed_lanes = []
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
                        with meter.scope('LANE_SITE_LANES', net=net):
                            lanes = access.component_lanes(task['component'].component_ref, budget=incidence_budget)
                            grouped = defaultdict(list)
                            retained_count = 0
                            for lane in lanes:
                                session.checkpoint(); meter.reserve('lane_site', 'lane_filter')
                                if lane.owner_net in (None, net):
                                    grouped[_lane_key(lane)].append(lane)
                                    retained_count += 1
                            unambiguous = sorted(
                                (values[0] for values in grouped.values() if len(values) == 1),
                                key=lambda lane: _pin_lane_priority(lane, task['pin_facts']))
                            if (pair_request is not None
                                    and pair_request.schema == 'clay.resource-pair-request/v4-iterative'):
                                unambiguous = _diverse_lane_centers(unambiguous)
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
                        completed_lanes.append((task, tuple(unambiguous[:request.max_lanes_per_component])))
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
            lane_phase_status = 'PARTIAL_BOUNDED_LANES'
            check()
            if ledger.landing_sites_complete:
                site_phase_status = 'UNKNOWN_EAGER_LEDGER_UNSUPPORTED'
                for row in rows.values():
                    if row['lane_components']:
                        row['site_status'] = 'UNKNOWN_EAGER_LEDGER_UNSUPPORTED'
                return finish('PARTIAL_BOUNDED_LANE_SITE')
            phase = 'LANE_SITE_TOPOLOGY'
            site_deadline = min(deadline, time.monotonic() + float(request.site_wall_seconds))
            site_phase_status = 'RUNNING'
            terminals = defaultdict(list)
            with meter.scope(phase):
                for demand in scope._topology_plan.demands:
                    for terminal in demand.terminals:
                        session.checkpoint(); meter.reserve('lane_site', 'topology_terminal_resolution')
                        fact = pc._resolve_terminal_fact(scope._world, net=demand.net, terminal=terminal)
                        if fact is not None:
                            terminals[(demand.net, fact)].append(terminal)
                pending_topology = []
                for net, row in rows.items():
                    for detail in row['endpoint_incidence']:
                        fact = detail['pin_fact']
                        options = terminals.get((net, fact), ())
                        status = ('UNKNOWN_NO_TOPOLOGY_TERMINAL' if not options else
                                  'UNKNOWN_AMBIGUOUS_TOPOLOGY_TERMINAL' if len(options) != 1 else
                                  'TOPOLOGY_TERMINAL_OBSERVED')
                        pending_topology.append((detail, status))
                pending_site_tasks = defaultdict(deque)
                pending_ambiguous = Counter()
                for task, lanes in completed_lanes:
                    net = task['net']
                    if net in stopped_nets:
                        continue
                    for fact in sorted(task['endpoints'], key=_json_bytes):
                        endpoint, roles = task['endpoints'][fact]
                        options = terminals.get((net, fact), ())
                        if len(options) != 1:
                            continue
                        if task['component'].component_ref not in endpoint.conductor_component_refs and (
                                len(endpoint.access_component_refs) != 1
                                or endpoint.access_component_refs[0] != task['component'].component_ref):
                            pending_ambiguous[net] += len(lanes)
                            continue
                        for lane in lanes:
                            pending_site_tasks[net].append({'net': net, 'component': task['component'],
                                'fact': fact, 'roles': roles, 'endpoint': endpoint, 'terminal': options[0],
                                'lane': lane,
                                'required_component_ref': (task['component'].component_ref
                                    if task['component'].component_ref in endpoint.conductor_component_refs else None)})
            capture_meter()
            for detail, status in pending_topology:
                detail['topology_eligibility_status'] = status
            site_tasks_by_net = dict(pending_site_tasks)
            for net, row in rows.items():
                row['ambiguous_pin_site_tasks'] = pending_ambiguous[net]
                if net in stopped_nets:
                    row['site_status'] = 'UNKNOWN_PER_NET_LIMIT'
                else:
                    row['remaining_site_tasks'] = len(site_tasks_by_net.get(net, ())) + pending_ambiguous[net]
                    if not site_tasks_by_net.get(net):
                        row['site_status'] = ('UNKNOWN_AMBIGUOUS_PIN_COMPONENT' if pending_ambiguous[net]
                                              else 'UNKNOWN_NO_OBSERVED_SITE_TASK')
            if not site_tasks_by_net:
                site_phase_status = ('UNKNOWN_AMBIGUOUS_PIN_COMPONENT' if any(pending_ambiguous.values())
                                     else 'UNKNOWN_NO_ELIGIBLE_SITE_TASK')
                if (pair_request is not None
                        and pair_request.schema not in ('clay.resource-pair-request/v3-deferred',
                                                        'clay.resource-pair-request/v4-iterative')):
                    from clay.layout.route.resource_pair_transaction import observe_pair
                    check()
                    phase = 'RESOURCE_PAIR'
                    initialization = (scope._pair_initialization[0]
                                      if type(scope._pair_initialization) is tuple else None)
                    result['resource_pair'] = observe_pair(scope=scope, cap=cap, request=pair_request,
                        meter=meter, boundary=boundary, pair_candidates=pair_candidates, initialization=initialization)
                return finish('PARTIAL_BOUNDED_LANE_SITE')

            phase = 'LANE_SITE_SITES'
            site_queried = Counter()
            early_pair_ready = False
            deferred_site_tasks = False
            while any(site_tasks_by_net.values()):
                made_progress = False
                for net in sorted(site_tasks_by_net):
                    queue = site_tasks_by_net[net]
                    if not queue:
                        continue
                    row = rows[net]
                    if site_queried[net] >= request.max_site_queries_per_net:
                        row['site_status'] = 'PARTIAL_SITE_QUERY_LIMIT'
                        row['remaining_site_tasks'] = len(queue) + row['ambiguous_pin_site_tasks']
                        queue.clear()
                        made_progress = True
                        continue
                    # An already passing empty-Book site needs no alternatives
                    # until an actual pair trial rejects.  Preserve its queue
                    # and query accounting while other nets get their turn.
                    if (pair_iteration is not None and not pair_iteration.need_alternatives
                            and any(coordinate[0] == net for _index, coordinate in pair_iteration.passed)):
                        continue
                    made_progress = True
                    task = queue.popleft()
                    active_task = task
                    site_active = True
                    active_stage = 'BOUNDARY_SITE_QUERY'
                    try:
                        check()
                        with meter.scope('LANE_SITE_SITES', net=net):
                            diagnostics: dict[str, int] = {}
                            with lane_site_boundary_transaction(access, meter):
                                sites = access.boundary(task['terminal'], task['endpoint'], task['lane'], net=net,
                                    required_component_ref=task['required_component_ref'], retain=False,
                                    diagnostics=diagnostics, budget=incidence_budget)
                            check_wall()
                            active_stage = 'SITE_ORDERING'
                            grouped_sites = defaultdict(list)
                            for site in sites:
                                session.checkpoint(); meter.reserve('lane_site', 'site_order')
                                grouped_sites[_site_key(site, conductors)].append(site)
                            ordered_sites = [grouped_sites[key][0] for key in sorted(grouped_sites)
                                             if len(grouped_sites[key]) == 1]
                            selected = ordered_sites[:request.max_sites_per_lane]
                            pending_site = {'pin_fact': task['fact'], 'source_roles': task['roles'],
                                'component_ref': task['component'].component_ref,
                                'component_kind': ('PIN_ONLY' if task['required_component_ref'] is None else 'CONDUCTOR'),
                                'terminal': {'xy_nm': task['terminal'].xy_nm, 'layer': task['terminal'].layer,
                                             'provenance': task['terminal'].provenance},
                                'lane': task['lane'].canonical_payload(), 'sites_discovered': len(sites),
                                'site_semantic_ties': sum(len(v) for v in grouped_sites.values() if len(v) > 1),
                                'sites_retained': [_landing_site_payload(site) for site in selected],
                                'site_truncated': len(ordered_sites) > len(selected),
                                'site_query_scope': 'THIS_ENDPOINT_COMPONENT_LANE_ONLY',
                                'status': ('OBSERVED_BOUNDED_LEGACY_SITE' if sites else 'OBSERVED_EMPTY_BOUNDED_QUERY'),
                                'source_contact': 'UNKNOWN_NOT_CHECKED', 'n2_via_legality': 'UNKNOWN_NOT_CHECKED'}
                            check_wall()
                        active_stage = 'SITE_RECHECK'
                        capture_meter()
                        row['site_queries'].append(pending_site)
                        if pair_request is not None:
                            # Preserve actual producer objects only after this
                            # query was committed to the live observation.
                            pair_candidates.extend((net, site, task, pending_site) for site in selected)
                        site_queried[net] += 1
                        row['remaining_site_tasks'] = len(queue) + row['ambiguous_pin_site_tasks']
                        row['site_status'] = ('PARTIAL_BOUNDED_WITH_UNKNOWN_PIN_COMPONENT'
                            if row['ambiguous_pin_site_tasks'] else 'PARTIAL_BOUNDED_SITE_QUERIES')
                        active_task = None
                        site_active = False
                        if (pair_request is not None
                                and pair_request.schema == 'clay.resource-pair-request/v4-iterative'):
                            if (pair_iteration is None
                                    and len({candidate[0] for candidate in pair_candidates}) >= 2):
                                check(wall=False)
                                meter.check_boundary()
                                check_wall(wall=False)
                                phase = 'RESOURCE_PAIR'
                                initialization_result = scope.consume_initialization(scope.initialize())
                                if initialization_result.get('status') != 'OBSERVED_EMPTY_PRE_GROUP':
                                    raise ValueError('iterative pair did not initialize an empty pre-group Book')
                                from clay.layout.route.resource_pair_transaction import _PairIteration
                                pair_iteration = _PairIteration(scope=scope, cap=cap, request=pair_request,
                                    meter=meter, boundary=boundary, pair_candidates=pair_candidates,
                                    initialization=scope._pair_initialization[0])
                                phase = 'LANE_SITE_SITES'
                            if pair_iteration is not None:
                                phase = 'RESOURCE_PAIR'
                                try:
                                    pair_iteration.advance()
                                finally:
                                    phase = 'LANE_SITE_SITES'
                                result['resource_pair'] = pair_iteration.result
                                if pair_iteration.finished:
                                    early_pair_ready = True
                                    break
                        if (pair_request is not None
                                and pair_request.schema in ('clay.resource-pair-request/v2-early',
                                                            'clay.resource-pair-request/v3-deferred')
                                and len({candidate[0] for candidate in pair_candidates}) >= 2):
                            early_pair_ready = True
                            break
                    except ResourceMeterStopped as exc:
                        if exc.reason != 'METER_NET_LIMIT' or exc.net != net:
                            raise
                        row['site_status'] = 'UNKNOWN_PER_NET_LIMIT'
                        row['remaining_site_tasks'] = len(queue) + 1 + row['ambiguous_pin_site_tasks']
                        stopped_nets[net] = {'reason': exc.reason, 'phase': exc.phase,
                            'used_global_operations': exc.used,
                            'last_snapshot_net_operations': (None if last_meter is None else
                                last_meter.get('by_net', {}).get(net, 0)), 'interrupted_stage': active_stage}
                        queue.clear()
                        active_task = None
                        site_active = False
                if early_pair_ready:
                    break
                if not made_progress:
                    deferred_site_tasks = True
                    break
            if early_pair_ready:
                for net, queue in site_tasks_by_net.items():
                    if queue:
                        rows[net]['remaining_site_tasks'] = len(queue) + rows[net]['ambiguous_pin_site_tasks']
                        rows[net]['site_status'] = 'UNKNOWN_EARLY_PAIR_UNVISITED'
                site_phase_status = 'PARTIAL_EARLY_PAIR'
            elif deferred_site_tasks:
                for net, queue in site_tasks_by_net.items():
                    if queue:
                        rows[net]['remaining_site_tasks'] = len(queue) + rows[net]['ambiguous_pin_site_tasks']
                        rows[net]['site_status'] = 'UNKNOWN_DEFERRED_PAIR_SITE'
                site_phase_status = 'PARTIAL_DEFERRED_PAIR_SITE'
            else:
                site_phase_status = 'PARTIAL_BOUNDED_SITE_QUERIES'
            if (pair_request is not None
                    and pair_request.schema != 'clay.resource-pair-request/v4-iterative'
                    and (pair_request.schema != 'clay.resource-pair-request/v3-deferred' or early_pair_ready)):
                from clay.layout.route.resource_pair_transaction import observe_pair
                check()
                if pair_request.schema == 'clay.resource-pair-request/v3-deferred':
                    meter.check_boundary()
                    check_wall()
                phase = 'RESOURCE_PAIR'
                if pair_request.schema == 'clay.resource-pair-request/v3-deferred':
                    initialization_result = scope.consume_initialization(scope.initialize())
                    if initialization_result.get('status') != 'OBSERVED_EMPTY_PRE_GROUP':
                        raise ValueError('deferred pair did not initialize an empty pre-group Book')
                initialization = (scope._pair_initialization[0]
                                  if type(scope._pair_initialization) is tuple else None)
                result['resource_pair'] = observe_pair(scope=scope, cap=cap, request=pair_request,
                    meter=meter, boundary=boundary, pair_candidates=pair_candidates, initialization=initialization)
            return finish('PARTIAL_BOUNDED_LANE_SITE')
        except ExecutionStopped:
            raise
        except (ResourceMeterStopped, _LaneSiteWall, _IterativePairWall, NegotiationWatchdogStop,
                rw.RoutingResourceEnumerationLimitError, pc._IncidenceBudgetExceeded) as exc:
            # A failed pair cleanup closes the capability.  Even exceptions
            # which cannot retain notes must remain the primary stop.
            if not cap._active or getattr(exc, '__notes__', ()) or getattr(exc.__cause__, '__notes__', ()):
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
            for net, queue in site_tasks_by_net.items():
                if queue:
                    rows[net]['remaining_site_tasks'] = len(queue) + rows[net]['ambiguous_pin_site_tasks']
            if active_task is not None:
                net = active_task['net']
                row = rows[net]
                if site_active:
                    row['remaining_site_tasks'] = (len(site_tasks_by_net[net]) + 1
                                                   + row['ambiguous_pin_site_tasks'])
                    row['site_status'] = 'UNKNOWN_INTERRUPTED_SITE'
                    row['interrupted_site'] = {'pin_fact': active_task['fact'],
                        'component_ref': active_task['component'].component_ref,
                        'stage': active_stage, 'site_count': None}
                else:
                    row['remaining_components'] = len(tasks_by_net[net]) + 1 + row['deferred_components']
                    row['status'] = 'UNKNOWN_INTERRUPTED_COMPONENT'
                    row['interrupted_component'] = {'component_ref': active_task['component'].component_ref,
                                                      'stage': active_stage, 'lanes_discovered': None}
            if isinstance(exc, (_IterativePairWall, NegotiationWatchdogStop)) and pair_iteration is not None:
                pair_iteration.result['status'] = 'UNKNOWN_PAIR_WALL'
                pair_iteration.finished = True
                result['resource_pair'] = pair_iteration.result
            elif pair_iteration is not None:
                pair_iteration.finished = True
                result['resource_pair'] = pair_iteration.result
            status = ('UNKNOWN_METER_LIMIT' if isinstance(exc, ResourceMeterStopped) else
                      'UNKNOWN_PAIR_WALL' if isinstance(exc, (_IterativePairWall, NegotiationWatchdogStop)) else
                      ('UNKNOWN_SITE_WALL' if site_deadline is not None else 'UNKNOWN_LANE_WALL')
                      if isinstance(exc, _LaneSiteWall) else 'UNKNOWN_LOCAL_LIMIT')
            if site_deadline is None:
                lane_phase_status = status
            else:
                site_phase_status = status
            stop = {'type': type(exc).__name__, 'phase': phase, 'message': str(exc)}
            if isinstance(exc, ResourceMeterStopped):
                stop.update(reason=exc.reason, meter_phase=exc.phase, net=exc.net,
                            used_global_operations=exc.used)
            if (pair_request is not None and isinstance(exc, _LaneSiteWall)
                    and phase == 'LANE_SITE_SITES'
                    and pair_request.schema not in ('clay.resource-pair-request/v3-deferred',
                                                    'clay.resource-pair-request/v4-iterative')):
                # A site-only soft wall does not consume the separate pair
                # allowance.  Only the already committed rows and their live
                # producer objects can enter the pair; the interrupted query
                # remains UNKNOWN above and its raw return is never used.
                # The pair entry checks this same meter within its own wall;
                # an expired meter returns typed UNKNOWN without a trial.
                from clay.layout.route.resource_pair_transaction import observe_pair
                phase = 'RESOURCE_PAIR'
                initialization = (scope._pair_initialization[0]
                                  if type(scope._pair_initialization) is tuple else None)
                try:
                    result['resource_pair'] = observe_pair(scope=scope, cap=cap, request=pair_request,
                        meter=meter, boundary=boundary, pair_candidates=pair_candidates,
                        initialization=initialization)
                except ExecutionStopped:
                    raise
                except BaseException as pair_error:
                    try:
                        record_unwind_failure(pair_error, exc)
                    except BaseException:
                        pass
                    raise
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
