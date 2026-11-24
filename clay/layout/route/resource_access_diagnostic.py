"""Bounded source-Pin/endpoint and legacy boundary-site observations.

The live producer, resource meter and Book remain owned by the calling setup
scope.  This diagnostic does not establish source-to-via contact, N2 legality,
joint routing or physical admission.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
import time
from typing import Any

from clay.flow.execution_budget import ExecutionStopped
from clay.flow.control_stop import record_unwind_failure
from clay.flow.execution_session import current_session
from clay.flow.run_admission import current_admission
from clay.layout.place.generated_incumbent import digest
from clay.layout.route import pattern_compile as pc
from clay.layout.route import resource_world as rw
from clay.layout.route.resource_meter import DiagnosticResourceMeter, ResourceMeterStopped, bind_budget
from clay.layout.route.resource_plan import _landing_site_payload
from clay.layout.route.source_handoff import _ACTIVE_SCOPE, _SourceHandoffScope
from clay.layout.route.terminal_identity import pin_fact, pin_source_aliases, unpack_pin_fact


@dataclass(frozen=True, slots=True)
class ResourceAccessRequest:
    """Explicit diagnostic access limits; never a source routing policy."""

    wall_seconds: float
    site_wall_seconds: float
    per_net_operations: int
    max_lanes_per_net: int
    max_lanes_per_component: int
    max_sites_per_lane: int
    max_output_rows: int
    max_output_bytes: int
    schema: str = 'clay.resource-access-request/v1'

    def __post_init__(self) -> None:
        if self.schema != 'clay.resource-access-request/v1':
            raise ValueError('unsupported resource access request schema')
        for name, maximum in (('wall_seconds', 90), ('site_wall_seconds', 25)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0 or value > maximum):
                raise ValueError(name+' must be finite, positive and within its diagnostic maximum')
        if self.site_wall_seconds > self.wall_seconds:
            raise ValueError('site wall must not exceed access wall')
        for name in ('per_net_operations', 'max_lanes_per_net', 'max_lanes_per_component',
                     'max_sites_per_lane', 'max_output_rows', 'max_output_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name+' must be a positive exact integer')
        if self.per_net_operations > 100_000:
            raise ValueError('per_net_operations exceeds the diagnostic maximum')
        if self.max_lanes_per_net > 16 or self.max_lanes_per_component > 16:
            raise ValueError('lane limit exceeds the diagnostic maximum')
        if self.max_sites_per_lane > 8:
            raise ValueError('site limit exceeds the diagnostic maximum')
        if self.max_output_bytes < 2048:
            raise ValueError('max_output_bytes must allow a bounded UNKNOWN envelope')


class _AccessWall(RuntimeError):
    pass


class _SiteWall(RuntimeError):
    pass


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _row_count(value: Any, limit: int) -> int:
    pending, count = [value], 0
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            count += len(item)
            if count > limit:
                return count
            pending.extend(item)
    return count


def _lane_key(lane: Any) -> tuple:
    """World-independent ordering fields; a tied key is not resolved by ID."""
    return (lane.layer_n, tuple(lane.gds_layer_key), lane.preferred_axis,
            lane.center_nm, tuple(lane.through_interval_nm), tuple(lane.corridor_cross_bounds_nm),
            lane.track_start, '' if lane.owner_net is None else lane.owner_net,
            lane.owner_net is not None)


def _component_key(component: Any) -> str:
    data = component.canonical_payload()
    return _json_bytes({key: data[key] for key in (
        'net', 'layer_key', 'bbox_nm', 'atom_facts', 'atom_terminal_conductor_refs',
        'pin_accesses', 'component_geometry_hash')}).decode('ascii')


def _site_key(site: Any, conductors: Any) -> tuple:
    def owners(refs: Any) -> tuple:
        return tuple(sorted(_component_key(conductors.component(ref)) for ref in refs))
    return (site.lower_layer_n, site.upper_layer_n, tuple(site.xy_nm), tuple(site.cut_bbox_nm or ()),
            tuple(site.lower_pad_bbox_nm or ()), tuple(site.upper_pad_bbox_nm or ()),
            '' if site.owner_net is None else site.owner_net, site.owner_net is not None,
            tuple(site.owner_layer_ns), owners(site.lower_owner_component_refs),
            owners(site.upper_owner_component_refs), site.capacity)


def observe_access(*, scope: Any, cap: Any, source_value: Any, request: ResourceAccessRequest,
                   meter: DiagnosticResourceMeter, ledger: Any, conductors: Any, index: Any,
                   boundary: Any) -> dict[str, Any]:
    """Observe source-bound endpoints/sites only inside the actual setup caller."""
    from clay.layout.route.resource_setup_diagnostic import run_resource_setup

    started = time.monotonic()
    if type(request) is not ResourceAccessRequest:
        raise TypeError('exact resource access request required')
    request.__post_init__()
    caller = inspect.currentframe().f_back
    session, admission = current_session(), current_admission()
    try:
        local = caller.f_locals
        if (caller.f_code is not run_resource_setup.__code__ or caller.f_globals is not run_resource_setup.__globals__
                or type(scope) is not _SourceHandoffScope or _ACTIVE_SCOPE.get() is not scope
                or local.get('scope') is not scope or local.get('cap') is not cap
                or local.get('source_value') is not source_value or local.get('access_request') is not request
                or local.get('meter') is not meter or local.get('ledger') is not ledger
                or local.get('conductors') is not conductors or local.get('index') is not index
                or local.get('boundary') is not boundary or not callable(boundary)
                or type(meter) is not DiagnosticResourceMeter or meter._scope is not scope
                or meter._per_net_max != request.per_net_operations or scope._cap is not cap
                or not cap._active or scope._projected or session is None or admission is None
                or session is not meter._session or admission is not meter._admission
                or getattr(scope, '_resource_access_consumed', False)):
            raise ValueError('resource access requires the current verified setup caller')
        scope._resource_access_consumed = True
        setup_identity = local.get('identity')
        if type(setup_identity) is not str or len(setup_identity) != 64:
            raise ValueError('resource access setup identity is unavailable')
        request_sha = digest(asdict(request))
        identity = digest({'schema': 'clay.resource-access-scope/v1',
                           'resource_setup_scope_identity_sha256': setup_identity,
                           'resource_access_request_sha256': request_sha})
        deadline = started + float(request.wall_seconds)
        site_deadline = None
        last_meter = None
        phase = 'ACCESS_ASSOCIATION'
        stopped_nets: dict[str, Any] = {}
        net_rows: dict[str, Any] = {}
        tasks_by_net: dict[str, deque] = {}
        join_complete = False
        active_task = None
        active_stage = None
        result = {'schema': 'clay.resource-access-observation/v1',
                  'status': 'UNKNOWN_NOT_STARTED',
                  'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                  'request_sha256': request_sha, 'scope_identity_sha256': identity,
                  'resource_setup_scope_identity_sha256': setup_identity,
                  'candidate_identity_sha256': scope._ticket.candidate_sha,
                  'invocation_id': admission.invocation_id, 'source_sha256': source_value.get('source_sha256'),
                  'world_sha256': scope._world.sha256(), 'pdk_sha256': scope._world.pdk_provenance_digest,
                  'topology_sha256': scope._topology_plan.sha256(), 'limits': asdict(request),
                  'source_denominator_status': 'UNKNOWN_SOURCE_DENOMINATOR',
                  'source_role_status': source_value.get('source_role_status'),
                  'source_contact': 'UNKNOWN_NOT_CHECKED', 'access_alternatives_or': 'UNKNOWN_NOT_OBSERVED',
                  'n2_via_legality': 'UNKNOWN_NOT_CHECKED', 'joint_route': 'UNKNOWN_NOT_EXECUTED',
                  'unmetered_work': ['source_scope_checks', 'topology_terminal_world_pin_scan',
                      'context_geometry_index', 'semantic_key_and_sort_internals', 'serialization_and_hashing'],
                  'fairness_scope': 'lane_and_site_queries_after_shared_source_join',
                  'meter_snapshot': None, 'net_rows': []}

        def check_wall(*, wall: bool = True) -> None:
            session.checkpoint()
            if wall and time.monotonic() >= deadline:
                raise _AccessWall('resource access cooperative wall exhausted')

        def check(*, wall: bool = True) -> None:
            boundary(enforce_wall=False)
            if (scope._resource_access_consumed is not True or caller.f_locals.get('access_request') is not request
                    or digest(asdict(request)) != request_sha):
                raise ValueError('resource access request or one-shot state changed')
            check_wall(wall=wall)

        def capture_meter() -> None:
            nonlocal last_meter
            check_wall()
            value = meter.snapshot()
            check()
            last_meter = value

        def finish(status: str, *, stop: dict[str, Any] | None = None) -> dict[str, Any]:
            check(wall=stop is None)
            value = dict(result)
            value.update(status=status, net_rows=[net_rows[n] for n in sorted(net_rows)],
                         source_join_complete=join_complete, per_net_stops=stopped_nets,
                         meter_snapshot=last_meter, elapsed_wall_seconds_lower_bound=time.monotonic()-started)
            if stop is not None:
                value['stop'] = stop
            encoded = _json_bytes(value)
            check_wall(wall=stop is None)
            if _row_count(value, request.max_output_rows) > request.max_output_rows or len(encoded) > request.max_output_bytes:
                value = {key: value[key] for key in ('schema', 'publication_status', 'physical_admission',
                    'request_sha256', 'scope_identity_sha256', 'resource_setup_scope_identity_sha256',
                    'candidate_identity_sha256', 'invocation_id')}
                value.update(status='UNKNOWN_OUTPUT_LIMIT', source_denominator_status='UNKNOWN_SOURCE_DENOMINATOR',
                    source_contact='UNKNOWN_NOT_CHECKED', access_alternatives_or='UNKNOWN_NOT_OBSERVED',
                    n2_via_legality='UNKNOWN_NOT_CHECKED', joint_route='UNKNOWN_NOT_EXECUTED',
                    output={'max_rows': request.max_output_rows, 'max_bytes': request.max_output_bytes,
                            'complete_payload_bytes': len(encoded)})
                if stop is not None:
                    value['underlying_stop'] = {'type': stop['type'], 'phase': stop['phase'],
                        'reason': stop.get('reason')}
                encoded = _json_bytes(value)
                if len(encoded) > request.max_output_bytes:
                    raise ValueError('resource access UNKNOWN envelope exceeds declared output limit')
            check(wall=stop is None)
            return json.loads(encoded)

        try:
            check()
            meter.check_boundary()
            meter.require_ledger(ledger)
            meter.require_incidence(index, ledger, conductors)
            roles = source_value['source_roles']
            role_nets = {row['net'] for row in roles}
            port_nets = set(scope._source_cell.ports)
            pin_nets = set(scope._engine_pins_by_net)
            demand_nets = {demand.net for demand in scope._topology_plan.demands}
            nets = sorted(role_nets | port_nets | pin_nets | demand_nets | set(scope._world.required_nets))
            if len(nets) + len(roles) > request.max_output_rows:
                return finish('UNKNOWN_OUTPUT_LIMIT')
            role_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
            for net in nets:
                net_rows[net] = {'net': net, 'source_roles': [],
                    'engine_pin_count': len(scope._engine_pins_by_net.get(net, ())),
                    'source_pin_count': len(scope._source_pins_by_net.get(net, ())),
                    'topology_demand_present': net in demand_nets,
                    'world_required': net in scope._world.required_nets,
                    'status': 'UNKNOWN_NOT_STARTED', 'queries': [], 'remaining_tasks': None,
                    'component_limited_tasks': 0}
            for role in roles:
                record = {'member': role['member'], 'role': role['role'], 'net': role['net'],
                          'source_identity_status': role['identity_status'], 'pin_incidence': []}
                net_rows[role['net']]['source_roles'].append(record)
                role_index[(role['member'], role['role'], role['net'])].append(record)
            for key, entries in role_index.items():
                if len(entries) != 1:
                    for entry in entries:
                        entry['source_identity_status'] = 'UNKNOWN_DUPLICATE_SOURCE_ROLE'

            pending_details = []
            with meter.scope(phase):
                endpoint_by_fact = {}
                for endpoint in index.endpoints:
                    session.checkpoint(); meter.reserve('access', 'endpoint_index')
                    if endpoint.pin_access_fact in endpoint_by_fact:
                        raise ValueError('resource access endpoint fact is ambiguous')
                    endpoint_by_fact[endpoint.pin_access_fact] = endpoint
                terminals = defaultdict(list)
                for demand in scope._topology_plan.demands:
                    for terminal in demand.terminals:
                        session.checkpoint(); meter.reserve('access', 'topology_terminal_resolution')
                        fact = pc._resolve_terminal_fact(scope._world, net=demand.net, terminal=terminal)
                        if fact is not None:
                            terminals[(demand.net, fact)].append(terminal)
                source_entries = []
                for net, pins in sorted(scope._source_pins_by_net.items()):
                    for pin in pins:
                        session.checkpoint(); meter.reserve('access', 'source_pin_fact')
                        fact = pin_fact(pin)
                        _base, binding = unpack_pin_fact(fact)
                        source_entries.append((net, pin, fact, pin_source_aliases(pin), binding))
                fact_counts = Counter((net, fact) for net, _pin, fact, _aliases, _binding in source_entries)
                task_index = {}
                for net, _pin, fact, aliases, binding in source_entries:
                    session.checkpoint(); meter.reserve('access', 'source_pin_join')
                    endpoint = endpoint_by_fact.get(fact)
                    refs = conductors.refs_for_pin(fact)
                    if endpoint is not None and (not refs or not set(refs).issubset(endpoint.access_component_refs)):
                        raise ValueError('resource access endpoint and conductor bindings differ')
                    options = terminals.get((net, fact), ())
                    status = ('UNKNOWN_DUPLICATE_PIN_FACT' if fact_counts[(net, fact)] != 1 else
                              'UNKNOWN_MISSING_ENDPOINT' if endpoint is None else
                              'ENDPOINT_INCIDENCE_OBSERVED')
                    topology_status = ('UNKNOWN_NO_TOPOLOGY_TERMINAL' if not options else
                                       'UNKNOWN_AMBIGUOUS_TOPOLOGY_TERMINAL' if len(options) != 1 else
                                       'TOPOLOGY_TERMINAL_OBSERVED')
                    matched = []
                    for member, role in aliases:
                        matches = role_index.get((member, role, net), ())
                        if len(matches) == 1:
                            matched.append(matches[0])
                    detail = {'pin_fact': fact, 'source_refs': None if binding is None else binding.source_refs,
                              'status': status, 'topology_eligibility_status': topology_status,
                              'access_component_refs': [] if endpoint is None else endpoint.access_component_refs,
                              'conductor_component_refs': [] if endpoint is None else endpoint.conductor_component_refs,
                              'site_observation': 'UNKNOWN_NOT_EXECUTED',
                              'source_contact': 'UNKNOWN_NOT_CHECKED', 'access_alternatives_or': 'UNKNOWN_NOT_OBSERVED'}
                    for record in matched:
                        pending_details.append((record, detail))
                    if (status != 'ENDPOINT_INCIDENCE_OBSERVED'
                            or topology_status != 'TOPOLOGY_TERMINAL_OBSERVED' or not matched):
                        continue
                    terminal = options[0]
                    for ref in endpoint.access_component_refs:
                        component = conductors.component(ref)
                        if component.net != net or component.world_digest != scope._world.sha256():
                            raise ValueError('resource access component belongs to another net or world')
                        key = (net, fact, ref, terminal.provenance, terminal.xy_nm, terminal.layer)
                        if key not in task_index:
                            task_index[key] = {'net': net, 'fact': fact, 'endpoint': endpoint,
                                'component': component, 'terminal': terminal,
                                'required_component_ref': ref if ref in endpoint.conductor_component_refs else None,
                                'roles': set(), 'lanes': None, 'next': 0}
                        task_index[key]['roles'].update((r['member'], r['role'], net) for r in matched)
            capture_meter()
            for record, detail in pending_details:
                record['pin_incidence'].append(detail)
            join_complete = True
            for task in task_index.values():
                tasks_by_net.setdefault(task['net'], deque()).append(task)
            for net, tasks in tasks_by_net.items():
                ordered = sorted(tasks, key=lambda item: (_component_key(item['component']),
                    _json_bytes(item['fact']), item['terminal'].provenance, tuple(sorted(item['roles']))))
                tasks_by_net[net] = deque(ordered)
            for net, row in net_rows.items():
                row['remaining_tasks'] = len(tasks_by_net.get(net, ()))
                if not tasks_by_net.get(net):
                    row['status'] = 'UNKNOWN_NO_ELIGIBLE_SITE_TASK'

            if not tasks_by_net:
                return finish('UNKNOWN_NO_ELIGIBLE_SITE_TASK')
            if ledger.landing_sites_complete:
                for net, tasks in tasks_by_net.items():
                    net_rows[net]['status'] = 'UNKNOWN_EAGER_LEDGER_UNSUPPORTED'
                    net_rows[net]['remaining_tasks'] = len(tasks)
                for row in net_rows.values():
                    for role in row['source_roles']:
                        for detail in role['pin_incidence']:
                            detail['site_observation'] = 'UNKNOWN_EAGER_LEDGER_UNSUPPORTED'
                return finish('UNKNOWN_EAGER_LEDGER_UNSUPPORTED')
            phase = 'ACCESS_LANE_INDEX'
            with meter.scope(phase):
                context = rw._prepare_exact_landing_site_query_context(
                    ledger, scope._world, scope._tech, conductors, resource_meter=meter)
                lane_index = pc._BudgetedLaneLayerIndex(index)
                incidence_budget = bind_budget(pc._IncidenceBudget(pc.MAX_INCIDENCE_EVALUATIONS), meter, 'incidence')
                lane_index._build(budget=incidence_budget)
                access = pc._ExactLazyLandingAccess(ledger, scope._world, scope._tech, conductors,
                                                    index, context, lane_index)
            capture_meter()
            phase = 'ACCESS_NETS'
            queried = Counter()
            component_queries = Counter()
            while any(tasks_by_net.values()):
                progressed = False
                for net in sorted(tasks_by_net):
                    queue = tasks_by_net[net]
                    if not queue:
                        continue
                    row = net_rows[net]
                    if queried[net] >= request.max_lanes_per_net:
                        row['status'] = 'PARTIAL_LANE_LIMIT'
                        row['remaining_tasks'] = len(queue) + row['component_limited_tasks']
                        queue.clear()
                        continue
                    task = queue.popleft(); progressed = True
                    active_task = task
                    active_stage = 'TASK_BOUNDARY'
                    try:
                        check()
                        pending_kind = 'QUERY'
                        pending_query = None
                        component_key = (net, task['component'].component_ref)
                        with meter.scope('ACCESS_NET', net=net):
                            if component_queries[component_key] >= request.max_lanes_per_component:
                                pending_kind = 'COMPONENT_LIMIT'
                            elif task['lanes'] is None:
                                active_stage = 'COMPONENT_LANE_DISCOVERY'
                                lanes = access.component_lanes(task['component'].component_ref,
                                                               budget=incidence_budget)
                                grouped = defaultdict(list)
                                for lane in lanes:
                                    session.checkpoint(); meter.reserve('access', 'lane_filter')
                                    if lane.owner_net in (None, net):
                                        grouped[_lane_key(lane)].append(lane)
                                task['lanes'] = [grouped[key][0] for key in sorted(grouped) if len(grouped[key]) == 1]
                                task['lane_ties'] = sum(len(items) for items in grouped.values() if len(items) > 1)
                                task['lanes_discovered'] = sum(len(items) for items in grouped.values())
                            if pending_kind == 'QUERY' and task['next'] >= len(task['lanes']):
                                pending_kind = 'NO_LANE'
                            if pending_kind == 'QUERY':
                                lane = task['lanes'][task['next']]
                                active_stage = 'SITE_DEADLINE'
                                if site_deadline is None:
                                    site_deadline = time.monotonic() + float(request.site_wall_seconds)
                                if time.monotonic() >= site_deadline:
                                    raise _SiteWall('resource access site wall exhausted')
                                diagnostics: dict[str, int] = {}
                                active_stage = 'BOUNDARY_SITE_QUERY'
                                sites = access.boundary(task['terminal'], task['endpoint'], lane, net=net,
                                    required_component_ref=task['required_component_ref'], retain=False,
                                    diagnostics=diagnostics, budget=incidence_budget)
                                if time.monotonic() >= site_deadline:
                                    raise _SiteWall('resource access site wall exhausted')
                                grouped_sites = defaultdict(list)
                                active_stage = 'SITE_ORDERING'
                                for site in sites:
                                    session.checkpoint(); meter.reserve('access', 'site_order')
                                    grouped_sites[_site_key(site, conductors)].append(site)
                                ordered_sites = [grouped_sites[key][0] for key in sorted(grouped_sites)
                                                 if len(grouped_sites[key]) == 1]
                                site_ties = sum(len(items) for items in grouped_sites.values() if len(items) > 1)
                                selected = ordered_sites[:request.max_sites_per_lane]
                                check()
                                pending_query = {'pin_fact': task['fact'],
                                    'source_roles': sorted(task['roles']),
                                    'component_ref': task['component'].component_ref,
                                    'component_kind': ('PIN_ONLY' if task['required_component_ref'] is None else 'CONDUCTOR'),
                                    'terminal': {'xy_nm': task['terminal'].xy_nm, 'layer': task['terminal'].layer,
                                                 'provenance': task['terminal'].provenance},
                                    'lane': lane.canonical_payload(), 'lanes_discovered': task['lanes_discovered'],
                                    'lane_semantic_ties': task['lane_ties'], 'sites_discovered': len(sites),
                                    'site_semantic_ties': site_ties,
                                    'sites_retained': [_landing_site_payload(site) for site in selected],
                                    'site_truncated': len(ordered_sites) > len(selected),
                                    'status': ('OBSERVED_BOUNDED_LEGACY_SITE' if sites
                                               else 'OBSERVED_EMPTY_BOUNDED_QUERY'),
                                    'site_query_scope': 'THIS_ENDPOINT_COMPONENT_LANE_ONLY',
                                    'source_contact': 'UNKNOWN_NOT_CHECKED', 'n2_via_legality': 'UNKNOWN_NOT_CHECKED'}
                                if time.monotonic() >= site_deadline:
                                    raise _SiteWall('resource access site wall exhausted')
                        active_stage = 'TASK_RECHECK'
                        capture_meter()
                        if pending_kind == 'QUERY' and time.monotonic() >= site_deadline:
                            raise _SiteWall('resource access site wall exhausted')
                        if pending_kind == 'COMPONENT_LIMIT':
                            row['component_limited_tasks'] += 1
                            row['status'] = 'PARTIAL_COMPONENT_LANE_LIMIT'
                        elif pending_kind == 'NO_LANE':
                            row['status'] = ('PARTIAL_COMPONENT_LANE_LIMIT' if row['component_limited_tasks']
                                             else 'UNKNOWN_NO_UNAMBIGUOUS_LANE')
                        else:
                            row['queries'].append(pending_query)
                            queried[net] += 1; component_queries[component_key] += 1; task['next'] += 1
                            if task['next'] < len(task['lanes']):
                                queue.append(task)
                            row['status'] = ('PARTIAL_COMPONENT_LANE_LIMIT' if row['component_limited_tasks']
                                             else 'PARTIAL_BOUNDED')
                        row['remaining_tasks'] = len(queue) + row['component_limited_tasks']
                        active_task = None
                    except ResourceMeterStopped as exc:
                        if exc.reason != 'METER_NET_LIMIT' or exc.net != net:
                            raise
                        row['status'] = 'UNKNOWN_PER_NET_LIMIT'
                        row['remaining_tasks'] = len(queue) + 1 + row['component_limited_tasks']
                        stopped_nets[net] = {'reason': exc.reason, 'phase': exc.phase,
                            'used_global_operations': exc.used,
                            'last_snapshot_net_operations': (None if last_meter is None else
                                last_meter.get('by_net', {}).get(net, 0)),
                            'interrupted_stage': active_stage}
                        queue.clear()
                        active_task = None
                if not progressed:
                    break
            return finish('PARTIAL_BOUNDED_ACCESS')
        except ExecutionStopped:
            raise
        except (ResourceMeterStopped, _AccessWall, _SiteWall, rw.RoutingResourceEnumerationLimitError,
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
                    net_rows[net]['remaining_tasks'] = len(queue) + net_rows[net]['component_limited_tasks']
            if active_task is not None:
                net = active_task['net']
                row = net_rows[net]
                row['remaining_tasks'] = len(tasks_by_net[net]) + 1 + row['component_limited_tasks']
                row['status'] = 'UNKNOWN_INTERRUPTED_TASK'
                row['interrupted_task'] = {'pin_fact': active_task['fact'],
                    'component_ref': active_task['component'].component_ref,
                    'stage': active_stage, 'site_count': None}
            status = ('UNKNOWN_METER_LIMIT' if isinstance(exc, ResourceMeterStopped) else
                      'UNKNOWN_ACCESS_WALL' if isinstance(exc, _AccessWall) else
                      'UNKNOWN_SITE_WALL' if isinstance(exc, _SiteWall) else 'UNKNOWN_LOCAL_LIMIT')
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
