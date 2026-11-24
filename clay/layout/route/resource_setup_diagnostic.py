"""Bounded, invocation-owned observation of one prepared resource world.

This opt-in stage reports source/net sets and a ledger/index, not a route,
site-contact proof, complete source denominator, or physical admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
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
from clay.layout.route.resource_conductors import (
    OwnedConductorComponentIndex, OwnedConductorIndexLimitError,
)
from clay.layout.route.resource_meter import DiagnosticResourceMeter, ResourceMeterStopped
from clay.layout.route.resource_world import (
    DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS,
    MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS, MAX_ENUMERATED_TRACK_CENTERS,
    MAX_ENUMERATED_TRACK_RESOURCES, MAX_ENUMERATED_TRACK_SEGMENTS,
    MAX_ENUMERATED_VIA_RESOURCE_PAIRS, MAX_ENUMERATED_VIA_SITES,
    RoutingResourceEnumerationError, RoutingResourceEnumerationLimitError,
    enumerate_explicit_routing_resources, pdk_provenance_digest,
)
from clay.layout.route.pattern_compile import (
    MAX_INCIDENCE_EVALUATIONS, LaneIncidenceIndex, PatternCompileUnknown, build_lane_incidence_index,
)
from clay.layout.route.source_handoff import _ACTIVE_SCOPE, _SourceHandoffScope
from clay.layout.route.canonical_input_capture import route_engine_configuration


@dataclass(frozen=True, slots=True)
class ResourceSetupRequest:
    """Explicit diagnostic bounds; none replace source routing policy."""

    wall_seconds: float
    max_operations: int
    max_via_sites: int
    max_output_rows: int
    max_output_bytes: int
    schema: str = 'clay.resource-setup-request/v1'

    def __post_init__(self) -> None:
        if self.schema != 'clay.resource-setup-request/v1':
            raise ValueError('unsupported resource setup request schema')
        if (isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float))
                or not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0):
            raise ValueError('resource setup wall_seconds must be finite and positive')
        for name in ('max_operations', 'max_via_sites', 'max_output_rows', 'max_output_bytes'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name+' must be a positive exact integer')
        if self.max_via_sites > MAX_ENUMERATED_VIA_SITES:
            raise ValueError('max_via_sites exceeds the native enumeration maximum')
        if self.max_output_bytes < 2048:
            raise ValueError('max_output_bytes must allow a bounded UNKNOWN envelope')


class _SetupWallExceeded(RuntimeError):
    def __init__(self, phase: str):
        super().__init__('resource setup cooperative wall exhausted')
        self.phase = phase


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def _safe_unwind(original: BaseException, ending: BaseException) -> None:
    try:
        record_unwind_failure(original, ending)
    except BaseException:
        pass


def _row_count(value: Any, limit: int) -> int:
    """Bound output collection without returning a misleading partial list."""
    pending = [value]
    count = 0
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            count += len(item)
            if count > limit:
                return count
            pending.extend(item)
    return count


def run_resource_setup(*, scope: Any, cap: Any, source_value: Any,
                       source_value_sha: str, request: ResourceSetupRequest,
                       check_request: Any, access_request: Any = None,
                       lane_survey_request: Any = None, lane_site_request: Any = None,
                       pair_request: Any = None) -> dict[str, Any]:
    """Run exactly once from the current batch's real observed callback."""
    started = time.monotonic()
    if type(request) is not ResourceSetupRequest:
        raise TypeError('exact resource setup request required')
    request.__post_init__()
    if access_request is not None:
        from clay.layout.route.resource_access_diagnostic import ResourceAccessRequest
        if type(access_request) is not ResourceAccessRequest:
            raise TypeError('exact resource access request required')
        access_request.__post_init__()
    if lane_survey_request is not None:
        from clay.layout.route.resource_lane_survey import ResourceLaneSurveyRequest
        if type(lane_survey_request) is not ResourceLaneSurveyRequest:
            raise TypeError('exact resource lane survey request required')
        lane_survey_request.__post_init__()
        if access_request is not None:
            raise ValueError('resource lane survey and access are mutually exclusive')
    if lane_site_request is not None:
        from clay.layout.route.resource_lane_site import ResourceLaneSiteRequest
        if type(lane_site_request) is not ResourceLaneSiteRequest:
            raise TypeError('exact resource lane site request required')
        lane_site_request.__post_init__()
        if access_request is not None or lane_survey_request is not None:
            raise ValueError('resource lane site, lane survey and access are mutually exclusive')
    if pair_request is not None:
        from clay.layout.route.resource_pair_transaction import ResourcePairRequest
        if type(pair_request) is not ResourcePairRequest:
            raise TypeError('exact resource pair request required')
        pair_request.__post_init__()
        if lane_site_request is None:
            raise ValueError('resource pair requires resource lane site')
    deadline = started + float(request.wall_seconds)
    caller = inspect.currentframe().f_back
    meter = None
    phase = 'ENTRY'
    checkpoint: dict[str, Any] | None = None
    ledger_value: dict[str, Any] = {'status': 'NOT_RUN'}
    incidence_value: dict[str, Any] = {'status': 'NOT_RUN'}
    source_payload: dict[str, Any] = {}
    try:
        if type(scope) is not _SourceHandoffScope or _ACTIVE_SCOPE.get() is not scope:
            raise ValueError('resource setup requires the current source producer scope')
        ticket = scope._ticket
        callback = ticket.completion.callback
        local = caller.f_locals
        if (caller.f_code is not callback.__code__ or caller.f_globals is not callback.__globals__
                or local.get('scope') is not scope or local.get('cap') is not cap
                or local.get('source_value') is not source_value
                or local.get('source_value_sha') != source_value_sha
                or local.get('resource_request') is not request
                or local.get('access_request') is not access_request
                or local.get('lane_survey_request') is not lane_survey_request
                or local.get('lane_site_request') is not lane_site_request
                or local.get('pair_request') is not pair_request
                or local.get('check_request') is not check_request
                or scope._cap is not cap or not cap._active or not scope._observed
                or (scope._initial_receipt is not None if pair_request is not None
                    and pair_request.schema in ('clay.resource-pair-request/v3-deferred',
                                                'clay.resource-pair-request/v4-iterative')
                    else scope._initial_receipt is not False)
                or (pair_request is not None and pair_request.schema in (
                    'clay.resource-pair-request/v3-deferred', 'clay.resource-pair-request/v4-iterative')
                    and cap._initialized)
                or not callable(check_request)):
            raise ValueError('resource setup lacks its actual observed producer caller')
        if getattr(scope, '_resource_setup_consumed', False):
            raise ValueError('resource setup may run only once per candidate scope')
        scope._resource_setup_consumed = True
        projected = scope._projected
        session, admission = current_session(), current_admission()
        if session is None or admission is None or session.remaining_wall_seconds() is None:
            raise ValueError('resource setup requires the current finite admitted invocation')
        session.checkpoint()
        request_sha = digest(asdict(request))
        source_bytes = _json_bytes(source_value)
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        session.checkpoint()
        if (type(source_value_sha) is not str or len(source_value_sha) != 64
                or source_value_sha != source_sha):
            raise ValueError('source handoff observation changed after producer return')
        handoff_sha = digest(asdict(ticket.descriptor))
        identity_payload = {'schema': 'clay.resource-setup-scope/v1',
            'invocation_id': admission.invocation_id, 'candidate_identity_sha256': ticket.candidate_sha,
            'source_handoff_request_sha256': handoff_sha, 'resource_setup_request_sha256': request_sha}
        access_request_sha = digest(asdict(access_request)) if access_request is not None else None
        lane_survey_request_sha = digest(asdict(lane_survey_request)) if lane_survey_request is not None else None
        lane_site_request_sha = digest(asdict(lane_site_request)) if lane_site_request is not None else None
        pair_request_sha = digest(asdict(pair_request)) if pair_request is not None else None
        if access_request is not None:
            identity_payload['resource_access_request_sha256'] = access_request_sha
        if lane_survey_request is not None:
            identity_payload['resource_lane_survey_request_sha256'] = lane_survey_request_sha
        if lane_site_request is not None:
            identity_payload['resource_lane_site_request_sha256'] = lane_site_request_sha
        if pair_request is not None:
            identity_payload['resource_pair_request_sha256'] = pair_request_sha
        identity = digest(identity_payload)
        access_identity = (digest({'schema': 'clay.resource-access-scope/v1',
            'resource_setup_scope_identity_sha256': identity,
            'resource_access_request_sha256': access_request_sha}) if access_request is not None else None)
        lane_survey_identity = (digest({'schema': 'clay.resource-lane-survey-scope/v1',
            'resource_setup_scope_identity_sha256': identity,
            'resource_lane_survey_request_sha256': lane_survey_request_sha})
            if lane_survey_request is not None else None)
        lane_site_identity = (digest({'schema': 'clay.resource-lane-site-scope/v1',
            'resource_setup_scope_identity_sha256': identity,
            'resource_lane_site_request_sha256': lane_site_request_sha})
            if lane_site_request is not None else None)
        pair_identity = (digest({'schema': 'clay.resource-pair-scope/v1',
            'resource_lane_site_scope_identity_sha256': lane_site_identity,
            'resource_pair_request_sha256': pair_request_sha}) if pair_request is not None else None)
        def pair_unexecuted(*, output_limited=False):
            return {'schema': 'clay.resource-pair-observation/v1',
                    'status': 'UNKNOWN_PAIR_OUTPUT_LIMIT' if output_limited else 'UNKNOWN_NOT_EXECUTED',
                    'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                    'request_sha256': pair_request_sha, 'scope_identity_sha256': pair_identity,
                    'source_contact': 'UNKNOWN_NOT_CHECKED', 'joint_route': 'UNKNOWN_NOT_EXECUTED'}
        result: dict[str, Any] = {
            'schema': 'clay.resource-setup-observation/v1',
            'status': 'NOT_RUN', 'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK',
            'physical_admission': False, 'scope': 'SOURCE_NET_WORLD_LEDGER_INDEX_ONLY',
            'candidate_identity_sha256': ticket.candidate_sha, 'screen_identity_sha256': identity,
            'resource_setup_request_sha256': request_sha, 'source_handoff_request_sha256': handoff_sha,
            'request_sha256': request_sha, 'scope_identity_sha256': identity,
            'invocation_id': admission.invocation_id, 'source_handoff_value_sha256': source_sha,
            'limits': asdict(request), 'source_denominator_status': 'UNKNOWN_SOURCE_DENOMINATOR',
            'source_contact': 'UNKNOWN_NOT_CHECKED', 'access_alternatives_or': 'UNKNOWN_NOT_OBSERVED',
            'site': 'UNKNOWN_NOT_EXECUTED', 'pair': 'UNKNOWN_NOT_EXECUTED',
            'joint_route': 'UNKNOWN_NOT_EXECUTED', 'ledger': ledger_value, 'incidence': incidence_value,
            'meter_snapshot': None, 'unmetered_work': ['source_and_world_checks',
                'owned_conductor_planar_builds', 'sorting_and_assembly', 'serialization_and_hashing',
                'native_calls', 'engine_and_book'],
        }
        if access_request is not None:
            result['resource_access_request_sha256'] = access_request_sha
            result['meter_snapshot_scope'] = 'THROUGH_SETUP_ONLY'
            result['resource_access'] = {
                'schema': 'clay.resource-access-observation/v1',
                'status': 'UNKNOWN_SETUP_INCOMPLETE',
                'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK',
                'physical_admission': False,
                'request_sha256': access_request_sha,
                'scope_identity_sha256': access_identity,
            }
        if lane_survey_request is not None:
            result['resource_lane_survey_request_sha256'] = lane_survey_request_sha
            result['meter_snapshot_scope'] = 'THROUGH_SETUP_ONLY'
            result['resource_lane_survey'] = {
                'schema': 'clay.resource-lane-survey-observation/v1',
                'status': 'UNKNOWN_SETUP_INCOMPLETE',
                'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK',
                'physical_admission': False,
                'request_sha256': lane_survey_request_sha,
                'scope_identity_sha256': lane_survey_identity,
            }
        if lane_site_request is not None:
            result['resource_lane_site_request_sha256'] = lane_site_request_sha
            result['meter_snapshot_scope'] = 'THROUGH_SETUP_ONLY'
            result['resource_lane_site'] = {
                'schema': 'clay.resource-lane-site-observation/v1',
                'status': 'UNKNOWN_SETUP_INCOMPLETE',
                'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK',
                'physical_admission': False,
                'request_sha256': lane_site_request_sha,
                'scope_identity_sha256': lane_site_identity,
            }
        if pair_request is not None:
            result['resource_pair_request_sha256'] = pair_request_sha
            result['resource_lane_site']['resource_pair'] = pair_unexecuted()

        def check_wall(*, enforce_wall: bool = True) -> None:
            session.checkpoint()
            if enforce_wall and phase not in ('ACCESS', 'LANE_SURVEY', 'LANE_SITE') and time.monotonic() >= deadline:
                raise _SetupWallExceeded(phase)

        def boundary(*, enforce_wall: bool = True) -> None:
            session.checkpoint()
            if (caller.f_locals.get('scope') is not scope or caller.f_locals.get('cap') is not cap
                    or caller.f_locals.get('source_value') is not source_value
                    or caller.f_locals.get('source_value_sha') != source_value_sha
                    or caller.f_locals.get('resource_request') is not request
                    or caller.f_locals.get('access_request') is not access_request
                    or caller.f_locals.get('lane_survey_request') is not lane_survey_request
                    or caller.f_locals.get('lane_site_request') is not lane_site_request
                    or caller.f_locals.get('pair_request') is not pair_request
                    or caller.f_locals.get('check_request') is not check_request
                    or scope._resource_setup_consumed is not True or scope._projected is not projected
                    or digest(asdict(request)) != request_sha
                    or (access_request is not None and digest(asdict(access_request)) != access_request_sha)
                    or (lane_survey_request is not None and digest(asdict(lane_survey_request)) != lane_survey_request_sha)
                    or (lane_site_request is not None and digest(asdict(lane_site_request)) != lane_site_request_sha)
                    or (pair_request is not None and digest(asdict(pair_request)) != pair_request_sha)
                    or hashlib.sha256(_json_bytes(source_value)).hexdigest() != source_sha):
                raise ValueError('resource setup descriptor or source observation changed')
            scope._check()
            check_request()
            check_wall(enforce_wall=enforce_wall)

        def capture_meter() -> None:
            nonlocal checkpoint
            check_wall()
            value = meter.snapshot()
            boundary()
            checkpoint = value

        def stop_boundary(original: BaseException) -> None:
            try:
                boundary(enforce_wall=False)
            except ExecutionStopped:
                raise
            except BaseException as ending:
                _safe_unwind(original, ending)
                raise original

        def finish(status: str, *, stop: dict[str, Any] | None = None) -> dict[str, Any]:
            value = dict(result)
            value['status'] = status
            value['ledger'] = ledger_value
            value['incidence'] = incidence_value
            value['meter_snapshot'] = checkpoint
            value['elapsed_wall_seconds_lower_bound'] = time.monotonic() - started
            if stop is not None:
                value['stop'] = stop
            value.update(source_payload)
            boundary(enforce_wall=stop is None and phase not in ('ACCESS', 'LANE_SURVEY', 'LANE_SITE'))
            encoded = _json_bytes(value)
            check_wall(enforce_wall=stop is None and phase not in ('ACCESS', 'LANE_SURVEY', 'LANE_SITE'))
            count = _row_count(value, request.max_output_rows)
            check_wall(enforce_wall=stop is None and phase not in ('ACCESS', 'LANE_SURVEY', 'LANE_SITE'))
            if count > request.max_output_rows or len(encoded) > request.max_output_bytes:
                # No truncated lists are published as if they were complete.
                value = {key: item for key, item in value.items() if key in (
                    'schema', 'publication_status', 'physical_admission', 'scope',
                    'candidate_identity_sha256', 'screen_identity_sha256',
                    'resource_setup_request_sha256', 'source_handoff_request_sha256', 'invocation_id',
                    'resource_access_request_sha256',
                    'resource_lane_survey_request_sha256',
                    'resource_lane_site_request_sha256',
                    'resource_pair_request_sha256',
                    'request_sha256', 'scope_identity_sha256')}
                value.update(status='UNKNOWN_OUTPUT_LIMIT', source_denominator_status='UNKNOWN_SOURCE_DENOMINATOR',
                    ledger={'status': 'UNKNOWN_OUTPUT_LIMIT'}, incidence={'status': 'UNKNOWN_OUTPUT_LIMIT'},
                    source_contact='UNKNOWN_NOT_CHECKED', access_alternatives_or='UNKNOWN_NOT_OBSERVED',
                    site='UNKNOWN_NOT_EXECUTED', pair='UNKNOWN_NOT_EXECUTED', joint_route='UNKNOWN_NOT_EXECUTED',
                    output={'max_rows': request.max_output_rows, 'max_bytes': request.max_output_bytes,
                            'complete_payload_bytes': len(encoded)})
                if pair_request is not None:
                    value['pair'] = 'UNKNOWN_PAIR_OUTPUT_LIMIT'
                if access_request is not None:
                    value['resource_access'] = {'schema': 'clay.resource-access-observation/v1',
                        'status': 'UNKNOWN_OUTPUT_LIMIT',
                        'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                        'request_sha256': access_request_sha, 'scope_identity_sha256': access_identity}
                if lane_survey_request is not None:
                    value['resource_lane_survey'] = {'schema': 'clay.resource-lane-survey-observation/v1',
                        'status': 'UNKNOWN_OUTPUT_LIMIT',
                        'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                        'request_sha256': lane_survey_request_sha, 'scope_identity_sha256': lane_survey_identity}
                if lane_site_request is not None:
                    value['resource_lane_site'] = {'schema': 'clay.resource-lane-site-observation/v1',
                        'status': 'UNKNOWN_OUTPUT_LIMIT',
                        'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                        'request_sha256': lane_site_request_sha, 'scope_identity_sha256': lane_site_identity}
                    if pair_request is not None:
                        value['resource_lane_site']['resource_pair'] = pair_unexecuted(output_limited=True)
                if stop is not None:
                    value['underlying_stop'] = {
                        'type': stop['type'], 'phase': stop['phase'],
                        'reason': stop.get('reason'),
                        'message_sha256': hashlib.sha256(stop['message'].encode('utf8')).hexdigest(),
                        'message': 'UNKNOWN_OUTPUT_LIMIT',
                    }
                check_wall(enforce_wall=stop is None and phase not in ('ACCESS', 'LANE_SURVEY', 'LANE_SITE'))
                encoded = _json_bytes(value)
                check_wall(enforce_wall=stop is None and phase not in ('ACCESS', 'LANE_SURVEY', 'LANE_SITE'))
                if len(encoded) > request.max_output_bytes:
                    raise ValueError('resource setup UNKNOWN envelope exceeds declared output limit')
            detached = json.loads(encoded)
            boundary(enforce_wall=stop is None and phase not in ('ACCESS', 'LANE_SURVEY', 'LANE_SITE'))
            return detached

        try:
            boundary()
            if scope._projected:
                if source_value.get('status') != 'UNKNOWN_ENGINE_PIN_PROJECTION_UNSUPPORTED':
                    raise ValueError('projected source status differs from producer')
                return finish('UNKNOWN_ENGINE_PIN_PROJECTION_UNSUPPORTED')
            if (source_value.get('source_sha256') != cap._source_hash
                    or source_value.get('world_sha256') != scope._world.sha256()
                    or source_value.get('pdk_sha256') != pdk_provenance_digest(scope._tech)
                    or source_value.get('topology_sha256') != scope._topology_plan.sha256()
                    or source_value.get('candidate_identity_sha256') != ticket.candidate_sha):
                raise ValueError('source handoff values do not match current producer')
            phase = 'SOURCE_NET_WORLD'
            roles = source_value['source_roles']
            source_ports = sorted(set(scope._source_cell.ports))
            role_nets = sorted({row['net'] for row in roles})
            pin_nets = sorted(scope._engine_pins_by_net)
            demand_nets = sorted({row.net for row in scope._topology_plan.demands})
            required_nets = list(scope._world.required_nets)
            net_sets = {'source_ports': source_ports, 'observed_role_nets': role_nets,
                'engine_pin_nets': pin_nets, 'topology_demand_nets': demand_nets,
                'world_required_nets': required_nets}
            source_payload = {'source_sha256': source_value['source_sha256'],
                'world_sha256': source_value['world_sha256'], 'pdk_sha256': source_value['pdk_sha256'],
                'topology_sha256': source_value['topology_sha256'], 'topology': source_value['topology'],
                'source_handoff_status': source_value['status'],
                'selected_engine_configuration': route_engine_configuration(scope._engine),
                'source_role_status': source_value['source_role_status'], 'source_roles': roles,
                'uncovered': source_value['uncovered'], 'foreign_aliases': source_value['foreign_aliases'],
                'row_status': source_value['row_status'],
                'row_required_component_pin_facts': source_value['row_required_component_pin_facts'],
                'net_sets': net_sets,
                'net_differences': {'source_port_without_role': sorted(set(source_ports)-set(role_nets)),
                    'role_without_engine': sorted(set(role_nets)-set(pin_nets)),
                    'role_without_demand': sorted(set(role_nets)-set(demand_nets)),
                    'engine_without_role': sorted(set(pin_nets)-set(role_nets)),
                    'engine_without_demand': sorted(set(pin_nets)-set(demand_nets)),
                    'demand_without_engine': sorted(set(demand_nets)-set(pin_nets)),
                    'world_without_demand': sorted(set(required_nets)-set(demand_nets)),
                    'demand_without_world': sorted(set(demand_nets)-set(required_nets)),
                    'single_engine_pin': sorted(net for net, pins in scope._engine_pins_by_net.items() if len(pins)==1)}}
            boundary()
            if (_row_count(source_payload, request.max_output_rows) > request.max_output_rows
                    or len(_json_bytes(source_payload)) > request.max_output_bytes):
                return finish('UNKNOWN_OUTPUT_LIMIT')
            boundary()
            observation_request = (access_request if access_request is not None else
                                   lane_survey_request if lane_survey_request is not None else lane_site_request)
            meter_deadline = deadline + (float(observation_request.wall_seconds) if observation_request is not None else 0)
            if pair_request is not None:
                # The pair is a separate bounded phase on this same meter, still
                # capped by the admitted task and all existing operation limits.
                meter_deadline += float(pair_request.wall_seconds)
            remaining = min(meter_deadline-time.monotonic(), session.remaining_wall_seconds())
            if remaining <= 0:
                raise _SetupWallExceeded(phase)
            meter = DiagnosticResourceMeter(producer_scope=scope, max_operations=request.max_operations,
                per_net_operations=(observation_request.per_net_operations if observation_request is not None else None),
                wall_seconds=remaining)
            capture_meter()
            phase = 'RESOURCE_ENUMERATION'
            enumeration_started = time.monotonic()
            with meter.scope(phase):
                ledger = enumerate_explicit_routing_resources(scope._world, scope._tech,
                    max_via_sites=request.max_via_sites, resource_meter=meter)
            enumeration_elapsed = time.monotonic() - enumeration_started
            meter.require_ledger(ledger)
            boundary()
            ledger_json = ledger.canonical_json()
            boundary()
            ledger_digest = hashlib.sha256(ledger_json.encode('utf8')).hexdigest()
            capture_meter()
            ledger_value = {'status': 'OBSERVED', 'sha256': ledger_digest,
                'track_resources': len(ledger.track_resources), 'landing_sites': len(ledger.landing_sites),
                'landing_site_pair_conflicts': len(ledger.landing_site_pair_conflicts),
                'landing_sites_complete': ledger.landing_sites_complete,
                'landing_site_domain_digest': ledger.landing_site_domain_digest,
                'owned_conductor_index_digest': ledger.owned_conductor_index_digest,
                'enumeration_wall_seconds': enumeration_elapsed,
                'first_planar_build_timing': 'INCLUDED_IN_ENUMERATION_NOT_SEPARATELY_MEASURED',
                'native_limits': {'max_track_centers': MAX_ENUMERATED_TRACK_CENTERS,
                    'max_track_segments': MAX_ENUMERATED_TRACK_SEGMENTS,
                    'max_track_resources': MAX_ENUMERATED_TRACK_RESOURCES,
                    'max_via_resource_pairs': MAX_ENUMERATED_VIA_RESOURCE_PAIRS,
                    'max_site_pair_conflicts': MAX_ENUMERATED_LANDING_SITE_PAIR_CONFLICTS,
                    'max_geometry_evaluations': DEFAULT_ENUMERATED_GEOMETRY_EVALUATIONS}}
            phase = 'OWNED_CONDUCTOR_INDEX'
            boundary()
            planar_started = time.monotonic()
            conductors = OwnedConductorComponentIndex.build_planar(scope._world, scope._tech)
            planar_elapsed = time.monotonic() - planar_started
            boundary()
            if (conductors.world_digest != scope._world.sha256()
                    or conductors.sha256() != ledger.owned_conductor_index_digest):
                raise ValueError('independent planar conductor index differs from ledger')
            boundary()
            phase = 'LANE_INCIDENCE'
            incidence_started = time.monotonic()
            with meter.scope(phase):
                index = build_lane_incidence_index(ledger, conductors, scope._tech, resource_meter=meter)
            incidence_elapsed = time.monotonic() - incidence_started
            boundary()
            if type(index) is PatternCompileUnknown:
                capture_meter()
                incidence_value = {'status': 'UNKNOWN', 'native_unknown': index.canonical_payload()}
                return finish('UNKNOWN_INCIDENCE')
            if type(index) is not LaneIncidenceIndex:
                raise TypeError('incidence producer returned an unsupported result')
            meter.require_incidence(index, ledger, conductors)
            boundary()
            index_json = index.canonical_json()
            boundary()
            index_digest = hashlib.sha256(index_json.encode('utf8')).hexdigest()
            capture_meter()
            incidence_value = {'status': 'OBSERVED', 'sha256': index_digest,
                'lanes': len(index.lanes), 'site_sides': len(index.site_sides),
                'endpoints': len(index.endpoints), 'landing_pair_incidence': len(index.landing_pair_incidence),
                'boundary_lane_incidence': len(index.boundary_lane_incidence),
                'incidence_wall_seconds': incidence_elapsed,
                'independent_planar_build_wall_seconds': planar_elapsed,
                'max_incidence_evaluations': MAX_INCIDENCE_EVALUATIONS,
                'independent_planar_native_limits': {
                    name: parameter.default for name, parameter in
                    inspect.signature(OwnedConductorComponentIndex.build_planar).parameters.items()
                    if name.startswith('max_')}}
            if access_request is not None:
                from clay.layout.route.resource_access_diagnostic import observe_access
                boundary()
                phase = 'ACCESS'
                access_value = observe_access(scope=scope, cap=cap, source_value=source_value,
                    request=access_request, meter=meter, ledger=ledger, conductors=conductors,
                    index=index, boundary=boundary)
                boundary(enforce_wall=False)
                if (type(access_value) is not dict
                        or access_value.get('schema') != 'clay.resource-access-observation/v1'
                        or type(access_value.get('status')) is not str
                        or access_value.get('publication_status') != 'PENDING_TERMINAL_SCOPE_RECHECK'
                        or access_value.get('physical_admission') is not False
                        or access_value.get('request_sha256') != access_request_sha
                        or access_value.get('scope_identity_sha256') != access_identity):
                    raise ValueError('resource access did not return a detached pending observation')
                result['resource_access'] = json.loads(_json_bytes(access_value))
            if lane_survey_request is not None:
                from clay.layout.route.resource_lane_survey import observe_lane_survey
                boundary()
                phase = 'LANE_SURVEY'
                lane_survey_value = observe_lane_survey(scope=scope, cap=cap, source_value=source_value,
                    request=lane_survey_request, meter=meter, ledger=ledger, conductors=conductors,
                    index=index, boundary=boundary)
                boundary(enforce_wall=False)
                if (type(lane_survey_value) is not dict
                        or lane_survey_value.get('schema') != 'clay.resource-lane-survey-observation/v1'
                        or type(lane_survey_value.get('status')) is not str
                        or lane_survey_value.get('publication_status') != 'PENDING_TERMINAL_SCOPE_RECHECK'
                        or lane_survey_value.get('physical_admission') is not False
                        or lane_survey_value.get('request_sha256') != lane_survey_request_sha
                        or lane_survey_value.get('scope_identity_sha256') != lane_survey_identity):
                    raise ValueError('resource lane survey did not return a detached pending observation')
                result['resource_lane_survey'] = json.loads(_json_bytes(lane_survey_value))
            if lane_site_request is not None:
                from clay.layout.route.resource_lane_site import observe_lane_site
                boundary()
                phase = 'LANE_SITE'
                lane_site_value = observe_lane_site(scope=scope, cap=cap, source_value=source_value,
                    request=lane_site_request, meter=meter, ledger=ledger, conductors=conductors,
                    index=index, boundary=boundary,pair_request=pair_request)
                boundary(enforce_wall=False)
                if (type(lane_site_value) is not dict
                        or lane_site_value.get('schema') != 'clay.resource-lane-site-observation/v1'
                        or type(lane_site_value.get('status')) is not str
                        or lane_site_value.get('publication_status') != 'PENDING_TERMINAL_SCOPE_RECHECK'
                        or lane_site_value.get('physical_admission') is not False
                        or lane_site_value.get('request_sha256') != lane_site_request_sha
                        or lane_site_value.get('scope_identity_sha256') != lane_site_identity):
                    raise ValueError('resource lane site did not return a detached pending observation')
                if pair_request is not None:
                    pair_value = lane_site_value.get('resource_pair')
                    if (type(pair_value) is not dict
                            or pair_value.get('schema') != 'clay.resource-pair-observation/v1'
                            or pair_value.get('status') not in ('UNKNOWN_NOT_EXECUTED','UNKNOWN_NO_TWO_NET_SITES',
                                'PARTIAL_ITERATIVE_PAIR',
                                'UNKNOWN_ENGINE_UNSUPPORTED','OBSERVED_BOUNDED_PAIR','UNKNOWN_PAIR_WALL','UNKNOWN_PAIR_METER',
                                'UNKNOWN_PAIR_OUTPUT_LIMIT')
                            or pair_value.get('publication_status') != 'PENDING_TERMINAL_SCOPE_RECHECK'
                            or pair_value.get('physical_admission') is not False
                            or pair_value.get('request_sha256') != pair_request_sha
                            or pair_value.get('scope_identity_sha256') != pair_identity):
                        raise ValueError('resource pair did not return a detached pending observation')
                    result['pair'] = pair_value['status']
                result['resource_lane_site'] = json.loads(_json_bytes(lane_site_value))
            return finish('OBSERVED_LEDGER_INDEX')
        except ExecutionStopped:
            raise
        except (ResourceMeterStopped, _SetupWallExceeded, RoutingResourceEnumerationError,
                OwnedConductorIndexLimitError) as exc:
            if phase in ('ACCESS', 'LANE_SURVEY', 'LANE_SITE'):
                # Access owns its typed local-stop handling.  A leaked error must
                # not be relabeled as an incomplete setup result.
                raise
            if getattr(exc, '__notes__', ()) or getattr(exc.__cause__, '__notes__', ()):
                raise
            if isinstance(exc, RoutingResourceEnumerationError) and not (
                    isinstance(exc, RoutingResourceEnumerationLimitError)
                    or isinstance(exc.__cause__, OwnedConductorIndexLimitError)):
                raise
            stop_boundary(exc)
            status = ('UNKNOWN_METER_LIMIT' if isinstance(exc, ResourceMeterStopped) else
                      'UNKNOWN_SETUP_WALL' if isinstance(exc, _SetupWallExceeded) else 'UNKNOWN_LOCAL_LIMIT')
            if phase == 'RESOURCE_ENUMERATION' and ledger_value['status'] != 'OBSERVED':
                ledger_value = {'status': 'UNKNOWN'}
            if phase in ('OWNED_CONDUCTOR_INDEX', 'LANE_INCIDENCE') and incidence_value['status'] != 'OBSERVED':
                incidence_value = {'status': 'UNKNOWN'}
            detail = {'type': type(exc).__name__, 'message': str(exc), 'phase': phase}
            if isinstance(exc, ResourceMeterStopped):
                detail.update(reason=exc.reason, meter_phase=exc.phase, net=exc.net, used=exc.used)
            if exc.__cause__ is not None:
                detail['cause'] = {'type': type(exc.__cause__).__name__, 'message': str(exc.__cause__)}
            try:
                return finish(status, stop=detail)
            except ExecutionStopped:
                raise
            except BaseException as ending:
                _safe_unwind(exc, ending)
                raise exc
    except BaseException as original:
        closing = meter
        meter = None
        if closing is not None:
            try:
                closing.close()
            except BaseException as ending:
                _safe_unwind(original, ending)
        raise
    finally:
        del caller
        # A normal or typed-UNKNOWN return still closes the live meter.
        if meter is not None and not meter._closed:
            meter.close()
            session.checkpoint()
