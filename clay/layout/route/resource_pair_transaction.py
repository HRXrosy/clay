"""Bounded N=2 pair observations over live, producer-returned landing sites.

The public coordinator supplies no coordinates or engine.  This module may be
entered only by the active lane/site producer after its committed observations.
Results remain pending diagnostics and never authorize routing or admission.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass
import inspect
import json
import math
import time

from clay.flow.control_stop import record_unwind_failure
from clay.flow.execution_budget import ExecutionBudget, ExecutionStopped
from clay.flow.execution_session import current_session
from clay.flow.run_admission import current_admission
from clay.layout.place.generated_incumbent import digest
from clay.layout.route.canonical_input_capture import route_engine_configuration
from clay.layout.route.component_access_session import ComponentAccessSession
from clay.layout.route.engine_gridless import GridlessEngine, NegotiationWatchdogStop, _Book, _RoutedNet, _ViaSite
from clay.layout.route.resource_meter import DiagnosticResourceMeter, ResourceMeterStopped
from clay.layout.route.resource_plan import LandingSite, _landing_site_payload
from clay.layout.route.source_handoff import _ACTIVE_SCOPE, _SourceHandoffScope


@dataclass(frozen=True, slots=True)
class ResourcePairRequest:
    """Total cooperative wall for both orders, bounded independently of routing."""
    wall_seconds: float
    schema: str = 'clay.resource-pair-request/v1'

    def __post_init__(self):
        if self.schema not in ('clay.resource-pair-request/v1', 'clay.resource-pair-request/v2-early',
                               'clay.resource-pair-request/v3-deferred',
                               'clay.resource-pair-request/v4-iterative'):
            raise ValueError('unsupported resource pair request schema')
        if (isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float))
                or not math.isfinite(self.wall_seconds) or not 0 < self.wall_seconds <= 25):
            raise ValueError('pair wall must be finite, positive and at most 25 seconds')


class _PairWall(RuntimeError):
    pass


def _retain_primary(primary, ending):
    """An exception that refuses notes must still remain the primary stop."""
    try:
        record_unwind_failure(primary, ending)
    except BaseException:
        pass


class _PreservePhasePrimary:
    """Keep a phase observer's unwind failure from replacing the primary."""
    def __init__(self, phase, cap):
        self._phase = phase
        self._cap = cap

    def __enter__(self):
        return self._phase.__enter__()

    def __exit__(self, kind, value, traceback):
        try:
            suppressed = self._phase.__exit__(kind, value, traceback)
            if value is not None and suppressed:
                if self._cap is not None:
                    self._cap._active = False
                return False
            return suppressed
        except BaseException as ending:
            if value is None:
                raise
            if ending is not value:
                if self._cap is not None:
                    self._cap._active = False
                _retain_primary(value, ending)
            return False


def _empty(book):
    return not any(book.metal.values()) and not any(book.via_pads.values()) and not book.cuts and not book.persistent


def _check_empty_engine(engine):
    if (not _empty(engine._book) or any(engine._attempt_metal.values()) or any(engine._attempt_cuts.values())
            or engine._attempt_idx or engine._metal_idx or engine._via_pad_idx or engine._cut_idx
            or engine._via_pad_provenance or engine._component_access.escape_scope_active
            or hasattr(engine, '_resource_pair_transaction_token')
            or bool(getattr(engine, '_active_group_route_local_closure', False))
            or bool(getattr(engine, '_search_forbidden_via_sites', ()))
            or any(getattr(engine, name, None) is not None for name in
                   ('_active_mirror_legality', '_active_group_route_union_baseline',
                    '_active_wide_repair_route', '_active_hard_connectivity_witness',
                    '_active_pattern_search_incumbent', '_active_materialized_access_states',
                    '_active_access_routed_net', '_active_congested_rail_patterns',
                    '_active_rail_pattern_domain'))
            or bool(getattr(engine, '_active_require_width_carrier', False))
            or any(getattr(engine, name, None) is not None for name in
                   ('_active_static_metal_scans', '_active_static_cut_scans', '_active_static_obstacle_scans'))):
        raise ValueError('pair requires an empty, inactive pre-group engine')


def _bounded_rejection_evidence(engine, baseline, limits):
    """Detach complete causal values or an explicit UNKNOWN, leaving the ledger intact."""
    if limits is None:  # Narrow internal test helper retains the prototype representation.
        return {'status': 'OBSERVED_COMPLETE',
            'cumulative_causal_samples': engine._hard_rejection_sample_evidence(),
            'rejection_delta': [(list(k), v - baseline.get(k, 0)) for k, v in engine._hard_rejections.items()
                                if v > baseline.get(k, 0)]}
    max_rows, max_bytes = limits
    samples = engine._hard_rejection_samples
    def unknown():
        return {'status': 'UNKNOWN_OUTPUT_LIMIT', 'cumulative_causal_samples': None,
                'rejection_delta': None, 'sample_count': len(samples)}
    if len(samples) + len(engine._hard_rejections) > max_rows:
        return unknown()
    size = 2
    sample_rows, deltas = [], []
    # This is the existing engine evidence shape, copied without changing its
    # complete live causal table. Bound before appending each detached row.
    for cause in sorted(samples):
        count, candidate_extent = samples[cause]
        category, owner, gi, candidate_half, required_space, blocker_rect, blocker_kind, terminal_ref = cause
        row = (('category', category), ('owner', owner), ('gi', int(gi)),
               ('candidate_rect', tuple(candidate_extent)), ('candidate_half', candidate_half),
               ('rejection_count', int(count)), ('required_space', required_space),
               ('blocker_rect', blocker_rect), ('blocker_kind', blocker_kind),
               ('terminal_conductor_ref', terminal_ref))
        size += len(json.dumps(row, separators=(',', ':'), allow_nan=False).encode('utf8')) + 1
        if size > max_bytes:
            return unknown()
        sample_rows.append(row)
    for key, value in engine._hard_rejections.items():
        if value <= baseline.get(key, 0):
            continue
        row = (list(key), value - baseline.get(key, 0))
        size += len(json.dumps(row, separators=(',', ':'), allow_nan=False).encode('utf8')) + 1
        if size > max_bytes:
            return unknown()
        deltas.append(row)
    return {'status': 'OBSERVED_COMPLETE', 'cumulative_causal_samples': tuple(sample_rows),
            'rejection_delta': deltas}


def _trial_order(engine, session, first, second, *, deadline, meter=None, cap=None,
                 evidence_limits=None, meter_checkpoint=None, _test_hook=None, single=False,
                 check_restored_watchdog=False):
    """One isolated actual engine order; restore objects even on cooperative stop.

    ``first``/``second`` are internal (net, x, y, graph-low-layer) tuples.
    They acquire authority only through ``observe_pair``'s live producer checks.
    """
    _check_empty_engine(engine)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _PairWall('pair total wall exhausted')
    before_gen = engine._committed_index_generation
    names = ('_book', '_attempt_metal', '_attempt_cuts', '_attempt_idx', '_metal_idx', '_via_pad_idx',
             '_via_pad_provenance', '_cut_idx', '_component_access', '_via_static_legality_cache',
             '_wire_static_legality_cache', '_owned_atom_static_via_events')
    originals = {name: getattr(engine, name) for name in names}
    missing = object()
    diagnostic_names = ('_hard_rejections', '_hard_rejection_samples', '_hard_rejection_generation')
    diagnostic_original = {name: getattr(engine, name, missing) for name in diagnostic_names}
    missing_count = sum(value is missing for value in diagnostic_original.values())
    if missing_count not in (0, len(diagnostic_names)):
        raise ValueError('partial causal diagnostics ledger is unsupported')
    if (not missing_count and evidence_limits is not None
            and len(engine._hard_rejections) + len(engine._hard_rejection_samples) > evidence_limits[0]):
        # Do not duplicate an already oversized live causal ledger or mutate
        # engine state merely to produce a bounded diagnostic.
        return {'first': 'UNKNOWN_NOT_TESTED', 'second': 'UNKNOWN_NOT_TESTED',
                'rejection_evidence': {'status': 'UNKNOWN_OUTPUT_LIMIT',
                    'rejection_generation_before': engine._hard_rejection_generation,
                    'rejection_generation_after': engine._hard_rejection_generation,
                    'ledger_origin': 'EXISTING', 'cumulative_causal_samples': None,
                    'rejection_delta': None, 'sample_count': len(engine._hard_rejection_samples)}}
    if missing_count:
        engine._hard_rejections = {}
        engine._hard_rejection_samples = {}
        engine._hard_rejection_generation = 0
    baseline, baseline_gen = dict(engine._hard_rejections), engine._hard_rejection_generation
    token = object()
    engine._resource_pair_transaction_token = token
    primary = None
    trial_access = None
    result = None
    evidence = None
    try:
        with _PreservePhasePrimary(session.phase('resource_pair_order', budget=ExecutionBudget(wall_seconds=remaining)), cap):
            engine._book = _Book(engine._n_layers)
            engine._attempt_metal = {}; engine._attempt_cuts = {}; engine._attempt_idx = {}
            engine._component_access = trial_access = ComponentAccessSession(max_expansions=engine.max_expansions)
            engine._via_static_legality_cache = OrderedDict(originals['_via_static_legality_cache'])
            engine._wire_static_legality_cache = OrderedDict(originals['_wire_static_legality_cache'])
            engine._owned_atom_static_via_events = dict(originals['_owned_atom_static_via_events'])
            engine._rebuild_metal_index()

            def check(stage, item):
                session.checkpoint()
                if time.monotonic() >= deadline:
                    raise _PairWall('pair total wall exhausted')
                if meter is not None:
                    meter.check()
                engine._watchdog_check(item[0], 'resource_pair_' + stage)
                if _test_hook is not None:
                    _test_hook(stage)
                session.checkpoint()

            def via(item):
                value = _ViaSite(item[1], item[2], item[3])
                engine._require_policy_via(value)
                polys = engine._via_polys(value)
                cut_key = tuple(engine._tech.via_key_for(value.lo))
                if sum((p.layer, p.datatype) == cut_key for p in polys) != 4 or len(polys) != 6:
                    raise ValueError('selected N2 geometry differs')
                return value

            a = via(first)
            b = None if single else via(second)
            check('before_first', first)
            if meter is not None:
                with meter.scope('RESOURCE_PAIR', net=first[0]):
                    meter.reserve('pair', 'via_legal_first')
                    if meter_checkpoint is not None:
                        meter_checkpoint()
            first_ok = engine._via_legal(a.x, a.y, a.lo, first[0])
            check('after_first', first)
            if first_ok and not single:
                engine._commit(engine._book, {}, _RoutedNet(first[0], vias=[a], connected=True))
                engine._rebuild_metal_index()
                check('after_commit', first)
                if meter is not None:
                    with meter.scope('RESOURCE_PAIR', net=second[0]):
                        meter.reserve('pair', 'via_legal_second')
                        if meter_checkpoint is not None:
                            meter_checkpoint()
                second_ok = engine._via_legal(b.x, b.y, b.lo, second[0])
                check('after_second', second)
            else:
                second_ok = None
            result = {'first': 'PASS' if first_ok else 'REJECT',
                      'second': 'UNKNOWN_NOT_TESTED' if second_ok is None else 'PASS' if second_ok else 'REJECT'}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        failures = []
        def attempt(fn):
            try:
                fn()
            except BaseException as ending:
                failures.append(ending)
        def hook(stage):
            if _test_hook is not None:
                _test_hook(stage)
        def snapshot():
            nonlocal evidence
            hook('before_evidence')
            evidence = {'rejection_generation_before': baseline_gen,
                'rejection_generation_after': engine._hard_rejection_generation,
                'ledger_origin': 'ABSENT_TEMPORARY' if missing_count else 'EXISTING',
                **_bounded_rejection_evidence(engine, baseline, evidence_limits)}
        def merge_counts():
            hook('before_counts')
            if trial_access is not None:
                for key, value in trial_access._counts.items():
                    originals['_component_access']._counts[key] += value
        attempt(snapshot); attempt(merge_counts)
        for name, value in originals.items():
            attempt(lambda name=name: hook('restore:' + name))
            attempt(lambda name=name, value=value: setattr(engine, name, value))
        if trial_access is not None:
            attempt(originals['_component_access'].committed_world_changed)
        for name, value in diagnostic_original.items():
            if value is missing:
                attempt(lambda name=name: engine.__dict__.pop(name, None))
        def verify():
            if trial_access is not None and engine._committed_index_generation <= before_gen:
                raise ValueError('committed generation failed to advance')
            if any(getattr(engine, name) is not value for name, value in originals.items()):
                raise ValueError('original engine state identity was not restored')
            if getattr(engine, '_resource_pair_transaction_token', None) is not token:
                raise ValueError('pair ownership token changed')
            hook('after_restore')
        attempt(verify)
        if getattr(engine, '_resource_pair_transaction_token', None) is token:
            del engine._resource_pair_transaction_token
        else:
            failures.append(ValueError('pair ownership token could not be restored'))
        if failures:
            if cap is not None:
                cap._active = False
            if primary is None:
                primary = failures.pop(0)
                for ending in failures:
                    _retain_primary(primary, ending)
                raise primary
            for ending in failures:
                _retain_primary(primary, ending)
    _check_empty_engine(engine)
    if check_restored_watchdog:
        session.checkpoint()
        if time.monotonic() >= deadline:
            raise _PairWall('pair total wall exhausted after restoration')
        if meter is not None:
            meter.check()
        engine._watchdog_check(first[0], 'resource_pair_after_restore')
        session.checkpoint()
    return {**result, 'rejection_evidence': evidence}


class _PairIteration:
    """One v4 pair scope with a single absolute deadline and real Book trials."""

    def __init__(self, *, scope, cap, request, meter, boundary, pair_candidates, initialization):
        from clay.layout.route.resource_lane_site import observe_lane_site
        started = time.monotonic()
        caller = inspect.currentframe().f_back
        session, admission = current_session(), current_admission()
        try:
            local = caller.f_locals
            record = getattr(scope, '_pair_initialization', None)
            if (caller.f_code is not observe_lane_site.__code__ or caller.f_globals is not observe_lane_site.__globals__
                    or type(scope) is not _SourceHandoffScope or _ACTIVE_SCOPE.get() is not scope
                    or local.get('scope') is not scope or local.get('cap') is not cap
                    or local.get('pair_request') is not request or local.get('meter') is not meter
                    or local.get('boundary') is not boundary or local.get('pair_candidates') is not pair_candidates
                    or type(request) is not ResourcePairRequest
                    or request.schema != 'clay.resource-pair-request/v4-iterative'
                    or type(meter) is not DiagnosticResourceMeter or meter._scope is not scope
                    or meter._session is not session or meter._admission is not admission
                    or scope._ticket.ctx.extra.get('diagnostic_resource_pair_request') is not request
                    or type(pair_candidates) is not list or type(record) is not tuple or len(record) != 5
                    or record[0] is not initialization or record[1] is not cap or record[2] is not scope._engine
                    or record[3] is not scope._engine._book or record[4] is not scope._ticket
                    or scope._initial_receipt is not False or not cap._initialized or scope._cap is not cap
                    or getattr(scope, '_resource_pair_consumed', False) or scope._projected
                    or session is None or admission is None or session.invocation_id != admission.invocation_id):
                raise ValueError('iterative pair requires the current initialized lane-site producer')
            boundary(enforce_wall=False)
            scope._resource_pair_consumed = True
            self.scope, self.cap, self.request, self.meter, self.boundary = scope, cap, request, meter, boundary
            self.session, self.engine, self.book = session, scope._engine, scope._engine._book
            self.candidates, self.rows = pair_candidates, local['rows']
            self.index, self.conductors = local['index'], local['conductors']
            self.limits = (local['request'].max_output_rows, local['request'].max_output_bytes)
            self.deadline = started + float(request.wall_seconds)
            self.scanned = 0
            self.pending = []
            self.passed = []
            self.tried = set()
            self.need_alternatives = False
            self.finished = False
            request_sha = digest(asdict(request))
            identity = digest({'schema': 'clay.resource-pair-scope/v1',
                               'resource_lane_site_scope_identity_sha256': local['identity'],
                               'resource_pair_request_sha256': request_sha})
            self.result = {'schema': 'clay.resource-pair-observation/v1',
                'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                'request_sha256': request_sha, 'scope_identity_sha256': identity,
                'candidate_identity_sha256': scope._ticket.candidate_sha,
                'invocation_id': admission.invocation_id, 'task_id': admission.task_id,
                'source_sha256': cap._source_hash, 'world_sha256': scope._world.sha256(),
                'pdk_sha256': scope._world.pdk_provenance_digest,
                'engine': route_engine_configuration(self.engine),
                'source_contact': 'UNKNOWN_NOT_CHECKED', 'joint_route': 'UNKNOWN_NOT_EXECUTED',
                'unvisited_pairs': 'UNKNOWN_NOT_TESTED', 'unvisited_committed_sites': 0,
                'single_sites': [], 'orders': [], 'limits': asdict(request),
                'status': 'UNKNOWN_NO_TWO_NET_SITES'}
            if type(self.engine) is not GridlessEngine or self.engine.via_redundancy != 2:
                self.result['status'] = 'UNKNOWN_ENGINE_UNSUPPORTED'
                self.finished = True
            else:
                _check_empty_engine(self.engine)
        finally:
            del caller

    def _check(self):
        from clay.layout.route.resource_lane_site import observe_lane_site
        caller = inspect.currentframe().f_back.f_back
        try:
            if (caller.f_code is not observe_lane_site.__code__ or caller.f_globals is not observe_lane_site.__globals__
                    or caller.f_locals.get('pair_iteration') is not self
                    or _ACTIVE_SCOPE.get() is not self.scope or self.scope._engine is not self.engine
                    or self.engine._book is not self.book or self.scope._cap is not self.cap
                    or self.scope._pair_initialization is not self.scope._pair_initialization_anchor):
                raise ValueError('iterative pair scope changed')
        finally:
            del caller
        self.boundary(enforce_wall=False)
        self.meter.check()
        self.session.checkpoint()
        if time.monotonic() >= self.deadline:
            raise _PairWall('iterative pair total wall exhausted')
        self.engine._watchdog_check('<iterative>', 'resource_pair_checkpoint')
        self.session.checkpoint()

    def advance(self):
        """Consume only newly committed sites, retaining passing sites for later nets."""
        try:
            self._check()
            self.pending.extend(range(self.scanned, len(self.candidates)))
            self.scanned = len(self.candidates)
            while self.pending and not self.finished:
                passed_nets = {coordinate[0] for _index, coordinate in self.passed}
                choice = next((i for i, candidate_index in enumerate(self.pending)
                               if self.candidates[candidate_index][0] not in passed_nets), None)
                if choice is None:
                    if not self.need_alternatives:
                        break
                    choice = 0
                candidate_index = self.pending[choice]
                net, site, task, pending = self.candidates[candidate_index]
                if (type(net) is not str or type(site) is not LandingSite or type(task) is not dict
                        or type(pending) is not dict or net not in self.rows
                        or not any(pending is item for item in self.rows[net]['site_queries'])
                        or task.get('net') != net or task.get('fact') != tuple(pending['pin_fact'])
                        or task.get('roles') != tuple(map(tuple, pending['source_roles']))
                        or task['component'] is not self.conductors.component(task['component'].component_ref)
                        or not any(task['endpoint'] is endpoint for endpoint in self.index.endpoints)
                        or task['endpoint'].pin_access_fact != task['fact']
                        or task['component'].component_ref != pending['component_ref']
                        or task['lane'].canonical_payload() != pending['lane']
                        or _landing_site_payload(site) not in pending['sites_retained']
                        or site.routing_world_provenance_digest != self.scope._world.sha256()
                        or site.owner_net not in (None, net)):
                    raise ValueError('iterative pair site is not an observed current producer result')
                lo = self.engine._tech.resolve_layer_to_gi(tuple(self.engine._tech.pdk.metal_key(site.lower_layer_n)))
                hi = self.engine._tech.resolve_layer_to_gi(tuple(self.engine._tech.pdk.metal_key(site.upper_layer_n)))
                if lo is None or hi != lo + 1 or not 0 <= lo < self.engine._n_layers - 1:
                    raise ValueError('iterative pair site layers differ')
                coordinate = (net, site.xy_nm[0], site.xy_nm[1], lo)
                trial = _trial_order(self.engine, self.session, coordinate, None, deadline=self.deadline,
                    meter=self.meter, cap=self.cap, evidence_limits=self.limits, single=True,
                    check_restored_watchdog=True)
                self._check()
                self.pending.pop(choice)
                single_index = len(self.result['single_sites'])
                self.result['single_sites'].append({'net': net, 'site': _landing_site_payload(site),
                    'pin_fact': task['fact'], 'component_ref': task['component'].component_ref,
                    'lane': pending['lane'], 'status': trial['first'],
                    'rejection_evidence': trial['rejection_evidence']})
                self.result['status'] = 'PARTIAL_ITERATIVE_PAIR'
                if (trial['first'] not in ('PASS', 'REJECT')
                        or trial['rejection_evidence']['status'] == 'UNKNOWN_OUTPUT_LIMIT'):
                    self.result['status'] = 'UNKNOWN_PAIR_OUTPUT_LIMIT'
                    self.finished = True
                    break
                if trial['first'] == 'PASS':
                    self.passed.append((single_index, coordinate))
                for left_i, left in self.passed:
                    for right_i, right in self.passed:
                        if left_i >= right_i or left[0] == right[0] or (left_i, right_i) in self.tried:
                            continue
                        self.tried.add((left_i, right_i))
                        outcomes = []
                        for first, second in ((left, right), (right, left)):
                            self._check()
                            outcome = _trial_order(self.engine, self.session, first, second,
                                deadline=self.deadline, meter=self.meter, cap=self.cap, evidence_limits=self.limits,
                                check_restored_watchdog=True)
                            self._check()
                            row = {'first_net': first[0], 'second_net': second[0],
                                   'first_site_index': left_i if first is left else right_i,
                                   'second_site_index': right_i if second is right else left_i, **outcome}
                            self.result['orders'].append(row)
                            outcomes.append(row)
                            if outcome['rejection_evidence']['status'] == 'UNKNOWN_OUTPUT_LIMIT':
                                self.result['status'] = 'UNKNOWN_PAIR_OUTPUT_LIMIT'
                                self.finished = True
                                break
                        if self.finished:
                            break
                        if len(outcomes) == 2 and any(row['first'] == row['second'] == 'PASS' for row in outcomes):
                            self.result['status'] = 'OBSERVED_BOUNDED_PAIR'
                            self.finished = True
                            break
                        if len(outcomes) == 2:
                            self.need_alternatives = True
                    if self.finished:
                        break
                if len(json.dumps(self.result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) > self.limits[1]:
                    self.result['status'] = 'UNKNOWN_PAIR_OUTPUT_LIMIT'
                    self.finished = True
                self._check()
        except _PairWall as exc:
            if not self.cap._active or getattr(exc, '__notes__', ()) or getattr(exc.__cause__, '__notes__', ()):
                raise
            self.session.checkpoint()
            self.result['status'] = 'UNKNOWN_PAIR_WALL'
            self.finished = True
        except ResourceMeterStopped as exc:
            if not self.cap._active or getattr(exc, '__notes__', ()):
                raise
            self.session.checkpoint()
            self.result.update(status='UNKNOWN_PAIR_METER',
                               stop={'reason': exc.reason, 'phase': exc.phase, 'net': exc.net})
            self.finished = True
        except ExecutionStopped as exc:
            if (exc.reason == 'WALL_BUDGET_EXCEEDED' and exc._details.get('scope') == 'phase'
                    and exc._details.get('phase') == 'resource_pair_order' and self.cap._active
                    and not getattr(exc, '__notes__', ())):
                self.session.checkpoint()
                self.result['status'] = 'UNKNOWN_PAIR_WALL'
                self.finished = True
            else:
                raise
        except NegotiationWatchdogStop as exc:
            if not self.cap._active or getattr(exc, '__notes__', ()):
                raise
            self.session.checkpoint()
            self.result['status'] = 'UNKNOWN_PAIR_WALL'
            self.finished = True
        self.result['unvisited_committed_sites'] = len(self.pending) + len(self.candidates) - self.scanned
        return self.finished


def observe_pair(*, scope, cap, request, meter, boundary, pair_candidates, initialization):
    """Observe one same-scope, different-net pair in both actual engine orders."""
    from clay.layout.route.resource_lane_site import observe_lane_site
    started = time.monotonic()
    caller = inspect.currentframe().f_back
    session, admission = current_session(), current_admission()
    try:
        local = caller.f_locals
        record = getattr(scope, '_pair_initialization', None)
        if (caller.f_code is not observe_lane_site.__code__ or caller.f_globals is not observe_lane_site.__globals__
                or type(scope) is not _SourceHandoffScope or _ACTIVE_SCOPE.get() is not scope
                or local.get('scope') is not scope or local.get('cap') is not cap
                or local.get('pair_request') is not request or local.get('meter') is not meter
                or local.get('boundary') is not boundary or local.get('pair_candidates') is not pair_candidates
                or type(request) is not ResourcePairRequest or type(meter) is not DiagnosticResourceMeter
                or meter._scope is not scope or meter._session is not session or meter._admission is not admission
                or scope._ticket.ctx.extra.get('diagnostic_resource_pair_request') is not request
                or type(pair_candidates) is not list or type(record) is not tuple or len(record) != 5
                or record[0] is not initialization or record[1] is not cap or record[2] is not scope._engine
                or record[3] is not scope._engine._book or record[4] is not scope._ticket
                or scope._initial_receipt is not False or not cap._initialized or scope._cap is not cap
                or getattr(scope, '_resource_pair_consumed', False) or scope._projected
                or session is None or admission is None or session.invocation_id != admission.invocation_id):
            raise ValueError('pair requires the current initialized lane-site producer')
        request.__post_init__()
        deadline = started + float(request.wall_seconds)
        boundary(enforce_wall=False)
        if request.schema != 'clay.resource-pair-request/v3-deferred':
            scope._check()
            cap.check()
        scope._resource_pair_consumed = True
        lane_identity = local.get('identity')
        request_sha = digest(asdict(request))
        identity = digest({'schema': 'clay.resource-pair-scope/v1',
                           'resource_lane_site_scope_identity_sha256': lane_identity,
                           'resource_pair_request_sha256': request_sha})
        engine = scope._engine
        config = route_engine_configuration(engine)
        result = {'schema': 'clay.resource-pair-observation/v1',
                  'publication_status': 'PENDING_TERMINAL_SCOPE_RECHECK', 'physical_admission': False,
                  'request_sha256': request_sha, 'scope_identity_sha256': identity,
                  'candidate_identity_sha256': scope._ticket.candidate_sha,
                  'invocation_id': admission.invocation_id, 'task_id': admission.task_id,
                  'source_sha256': cap._source_hash, 'world_sha256': scope._world.sha256(),
                  'pdk_sha256': scope._world.pdk_provenance_digest, 'engine': config,
                  'source_contact': 'UNKNOWN_NOT_CHECKED', 'joint_route': 'UNKNOWN_NOT_EXECUTED',
                  'orders': [], 'limits': asdict(request)}
        if time.monotonic() >= deadline:
            result['status'] = 'UNKNOWN_PAIR_WALL'
            return result
        def stopped_meter(exc):
            if not cap._active or getattr(exc, '__notes__', ()):
                raise exc
            # An invocation cancellation or wall stop remains authoritative.
            session.checkpoint()
            result.update(status='UNKNOWN_PAIR_METER',
                stop={'reason': exc.reason, 'phase': exc.phase, 'net': exc.net},
                meter_snapshot=None, meter_snapshot_scope='NO_SUCCESSFUL_PAIR_METER_SNAPSHOT',
                work_after_meter_snapshot='UNKNOWN_SINCE_PAIR_ENTRY')
            return result
        try:
            meter.check_boundary()
        except ResourceMeterStopped as exc:
            return stopped_meter(exc)
        if time.monotonic() >= deadline:
            result['status'] = 'UNKNOWN_PAIR_WALL'
            return result
        if type(engine) is not GridlessEngine or engine.via_redundancy != 2:
            result['status'] = 'UNKNOWN_ENGINE_UNSUPPORTED'
            return result
        _check_empty_engine(engine)
        rows, index, conductors = local['rows'], local['index'], local['conductors']
        selected = []
        for net, site, task, pending in pair_candidates:
            session.checkpoint()
            try:
                meter.check()
            except ResourceMeterStopped as exc:
                return stopped_meter(exc)
            if time.monotonic() >= deadline:
                result['status'] = 'UNKNOWN_PAIR_WALL'
                return result
            if (type(net) is not str or type(site) is not LandingSite or type(task) is not dict
                    or type(pending) is not dict or net not in rows
                    or not any(pending is item for item in rows[net]['site_queries'])
                    or task.get('net') != net or task.get('fact') != tuple(pending['pin_fact'])
                    or task.get('roles') != tuple(map(tuple, pending['source_roles']))
                    or task['component'] is not conductors.component(task['component'].component_ref)
                    or not any(task['endpoint'] is endpoint for endpoint in index.endpoints)
                    or task['endpoint'].pin_access_fact != task['fact']
                    or task['component'].component_ref != pending['component_ref']
                    or task['lane'].canonical_payload() != pending['lane']
                    or _landing_site_payload(site) not in pending['sites_retained']
                    or site.routing_world_provenance_digest != scope._world.sha256()
                    or site.owner_net not in (None, net)):
                raise ValueError('pair site is not an observed current producer result')
            selected.append((net, site, task, pending))
        first = selected[0] if selected else None
        second = next((row for row in selected[1:] if first is not None and row[0] != first[0]), None)
        if first is None or second is None:
            try:
                meter.check()
            except ResourceMeterStopped as exc:
                return stopped_meter(exc)
            if time.monotonic() >= deadline:
                result['status'] = 'UNKNOWN_PAIR_WALL'
                return result
            result['status'] = 'UNKNOWN_NO_TWO_NET_SITES'
            return result
        def coordinates(row):
            net, site, _task, _pending = row
            lo = engine._tech.resolve_layer_to_gi(tuple(engine._tech.pdk.metal_key(site.lower_layer_n)))
            hi = engine._tech.resolve_layer_to_gi(tuple(engine._tech.pdk.metal_key(site.upper_layer_n)))
            if lo is None or hi != lo + 1 or not 0 <= lo < engine._n_layers - 1:
                raise ValueError('pair site layers are unavailable to selected engine')
            return net, site.xy_nm[0], site.xy_nm[1], lo
        result['sites'] = [{'net': row[0], 'site': _landing_site_payload(row[1]),
                            'pin_fact': list(row[2]['fact']), 'component_ref': row[2]['component'].component_ref,
                            'lane': row[2]['lane'].canonical_payload()} for row in (first, second)]
        a, b = coordinates(first), coordinates(second)
        last_meter = {'snapshot': None}
        def capture_meter():
            # Assignment follows a successful real snapshot; an expired meter
            # never produces a reconstructed or post-hoc synthetic value.
            last_meter['snapshot'] = meter.snapshot()
        result['orders'] = [{'first_net': one[0], 'second_net': two[0],
                             'first': 'UNKNOWN_NOT_TESTED', 'second': 'UNKNOWN_NOT_TESTED'}
                            for one, two in ((a, b), (b, a))]
        try:
            capture_meter()
            for ordinal, (one, two) in enumerate(((a, b), (b, a))):
                boundary(enforce_wall=False); meter.check_boundary()
                if time.monotonic() >= deadline:
                    raise _PairWall('pair total wall exhausted')
                lane_request = local['request']
                outcome = _trial_order(engine, session, one, two, deadline=deadline, meter=meter, cap=cap,
                    evidence_limits=(lane_request.max_output_rows, lane_request.max_output_bytes),
                    meter_checkpoint=capture_meter)
                boundary(enforce_wall=False); meter.check_boundary()
                if time.monotonic() >= deadline:
                    raise _PairWall('pair total wall exhausted')
                capture_meter()
                result['orders'][ordinal] = {'first_net': one[0], 'second_net': two[0], **outcome}
                if outcome['rejection_evidence']['status'] == 'UNKNOWN_OUTPUT_LIMIT':
                    break
            result['status'] = ('UNKNOWN_PAIR_OUTPUT_LIMIT' if any(
                row.get('rejection_evidence', {}).get('status') == 'UNKNOWN_OUTPUT_LIMIT' for row in result['orders'])
                else 'OBSERVED_BOUNDED_PAIR')
        except _PairWall as exc:
            if not cap._active or getattr(exc, '__notes__', ()):
                raise
            session.checkpoint()
            result['status'] = 'UNKNOWN_PAIR_WALL'
        except ResourceMeterStopped as exc:
            if not cap._active or getattr(exc, '__notes__', ()):
                raise
            session.checkpoint()
            result.update(status='UNKNOWN_PAIR_METER', stop={'reason': exc.reason, 'phase': exc.phase, 'net': exc.net})
        except ExecutionStopped as exc:
            if (exc.reason == 'WALL_BUDGET_EXCEEDED' and exc._details.get('scope') == 'phase'
                    and exc._details.get('phase') == 'resource_pair_order' and cap._active
                    and not getattr(exc, '__notes__', ())):
                session.checkpoint()
                result['status'] = 'UNKNOWN_PAIR_WALL'
            else:
                raise
        except NegotiationWatchdogStop as exc:
            if not cap._active or getattr(exc, '__notes__', ()):
                raise
            session.checkpoint()
            result['status'] = 'UNKNOWN_PAIR_WALL'
        result['meter_snapshot'] = last_meter['snapshot']
        result['meter_snapshot_scope'] = 'LAST_SUCCESSFUL_PAIR_METER_SNAPSHOT'
        result['work_after_meter_snapshot'] = ('NONE_AT_FINAL_CHECKPOINT'
            if result['status'] in ('OBSERVED_BOUNDED_PAIR', 'UNKNOWN_PAIR_OUTPUT_LIMIT')
            else 'UNKNOWN_SINCE_LAST_SUCCESSFUL_SNAPSHOT')
        return json.loads(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))
    finally:
        del caller
