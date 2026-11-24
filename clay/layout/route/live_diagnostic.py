"""Synchronous, terminal observation of the actual prepared routing caller.

The callback is trusted in-process diagnostic code. Its retained raw references
are not authority after the invocation, and no routing verdict is produced.
"""
from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

from clay.frontend.source_revision import reopen_complete_source
from clay.layout.place.compact_snapshot import compact_value as _compact_value
from clay.layout.place.native_stage_observation import _execution_owner
from clay.layout.place.row_replay_snapshot import freeze_row_values
from clay.layout.policy import policy_for_cell
from clay.layout.route.canonical_input_capture import (
    _json_value, route_engine_configuration, snapshot_prepared_routing_input,
)
from clay.layout.route.engine_registry import route_engine_from_policy
from clay.layout.route.resource_world import pdk_provenance_digest
from clay.parser import parse_clay_file


class DiagnosticRoutingObserved(RuntimeError):
    """An explicitly requested preparation diagnostic stopped before routing."""

    def flow_error_data(self):
        return {'reason': 'DIAGNOSTIC_OBSERVATION_COMPLETE', 'physical_admission': False}


class DiagnosticRoutingUnavailable(RuntimeError):
    """The prepared caller cannot provide the requested diagnostic authority."""

    def flow_error_data(self):
        return {'reason': 'DIAGNOSTIC_OBSERVER_UNAVAILABLE', 'physical_admission': False}


@dataclass(frozen=True)
class NativeEndpointWorkObserver:
    """Request bounded native endpoint authority before receiving the capability.

    The wrapper itself remains the terminal callback identity. It does not
    make the observer an isolated or adversarially secure Python process.
    """
    callback: Callable

    def __post_init__(self):
        if not callable(self.callback):
            raise TypeError('native endpoint observer must be callable')

    def __call__(self, capability):
        return self.callback(capability)


@dataclass(frozen=True)
class PreRouteGateDecision:
    """A trusted caller's decision for this one live placement and world.

    This is an in-process control decision, not physical signoff or a reusable
    certificate.  The caller owns the checks and any evidence it records.
    """

    status: str
    detail: str = ''

    def __post_init__(self):
        if self.status not in ('PASS', 'FAIL', 'UNKNOWN') or type(self.status) is not str:
            raise ValueError('pre-route gate status must be PASS, FAIL or UNKNOWN')
        if type(self.detail) is not str:
            raise TypeError('pre-route gate detail must be an exact string')


@dataclass(frozen=True)
class PreRouteGate:
    """Explicit opt-in to an in-process verifier at the actual route boundary."""

    verify: Callable

    def __post_init__(self):
        if not callable(self.verify):
            raise TypeError('pre-route gate verifier must be callable')


class PreRouteGateRejected(RuntimeError):
    """The caller did not admit this exact live pre-route placement."""

    def __init__(self, decision):
        self.decision = decision
        super().__init__(f'pre-route gate returned {decision.status}: {decision.detail}')

    def flow_error_data(self):
        return {'reason': 'PRE_ROUTE_GATE_REJECTED', 'status': self.decision.status,
                'detail': self.decision.detail, 'physical_admission': False}


def _value(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _endpoint_authority_value(engine, capability):
    """Identity of trusted endpoint producers before the observer sees them.

    This detects ordinary same-process drift; it is not a sandbox for an
    observer with arbitrary Python memory or module mutation privileges.
    """
    from clay.layout.route import (engine_gridless, initialized_region_observation as region,
                                   initialized_world_observation as world)
    from clay.layout.place import native_stage_observation
    from clay.flow import control_stop, execution_session

    def token(value):
        owner = getattr(value, '__self__', None)
        function = getattr(value, '__func__', value)
        owner_cls = None if owner is None else (type(owner).__module__, type(owner).__qualname__)
        return (owner_cls, getattr(function, '__module__', None),
                getattr(function, '__qualname__', getattr(function, '__name__', None)))

    engine_names = ('route', '_planar_endpoint_domain', '_unrequired_planar_endpoint_domain', '_owned_terminal_component',
                    '_endpoint_diagnostic_step', '_owned_landing_has_minimum_throat',
                    '_point_in_closed_rect', '_watchdog_check', '_half_width_scope', '_hw',
                    '_rail_local_half_width_overrides')
    region_names = ('__init__', '_timed', '_check', '_scope', '_checkpoint',
                    '_bounded_endpoint', '_publish', 'endpoint', 'finish', 'close')
    globals_ = (engine_gridless._SpatialIndex, engine_gridless._SpatialIndex.__init__,
                engine_gridless._SpatialIndex.add,
                engine_gridless._SpatialIndex.query, engine_gridless._norm,
                engine_gridless._closed_rects_touch, engine_gridless._PlanarEndpointDomain,
                engine_gridless.current_session, engine_gridless.time.monotonic,
                region.InitializedRegionCapability, region._validate_owned_entry,
                region._owned_cache_value, region._consumer_value, region._method_value,
                region._book_value, region._json_value, region.pin_fact, region.publish_pending,
                region._NATIVE_ENDPOINT_EVENTS,
                world.observe_initialized_world,
                region.RegionObservationLimit, region.time.monotonic,
                native_stage_observation._execution_owner, control_stop.record_unwind_failure,
                execution_session.current_session, execution_session.ExecutionSession.checkpoint,
                _execution_owner, reopen_complete_source,
                _compact_value, freeze_row_values, snapshot_prepared_routing_input,
                route_engine_configuration, _endpoint_authority_value,
                engine_gridless._Tech, engine_gridless._Tech.__init__,
                engine_gridless._Tech.gi_min_width, engine_gridless._Tech.pin_seed,
                engine_gridless._Tech.resolve_layer_to_gi, engine_gridless._Tech.resolve_layer_to_pdk)
    tech = capability._inputs['tech']
    return (engine, capability, type(engine),
            type(engine_gridless._NM_PER_UM), engine_gridless._NM_PER_UM,
            tuple(token(getattr(engine, name)) for name in engine_names),
            tech, tuple(token(getattr(tech, name, None)) for name in
                        ('min_width', 'min_space', 'min_area', 'metal_key', 'via_rule',
                         'via_size', 'via_spacing')),
            tuple(token(getattr(capability, name)) for name in
                  ('check', '_check_inputs', '_check_endpoint_authority',
                   'initialize_before_groups', 'inspect')),
            tuple(token(getattr(region.InitializedRegionCapability, name)) for name in region_names),
            tuple(token(value) for value in globals_))


def _final_placement_value(placement):
    """Replay and freeze the actual placement consumed at this route boundary."""
    from clay.layout.generic_placement import Placement, verify_stored_row_projection
    from clay.layout.place.current_structure_operands import _native_values, _placed_values, _row_values
    if type(placement) is not Placement:
        raise TypeError('pre-route gate requires the exact final Placement')
    verify_stored_row_projection(placement)
    return (_placed_values(placement), _row_values(placement),
        _native_values((placement.polys, placement.matching_pair_axes,
            placement.matching_pair_physical_bboxes, placement.dropped_nonmos_matched,
            tuple((name, device.orientation, device.geometry_owner,
                   getattr(device, 'cluster', None))
                  for name, device in sorted(placement.devices.items())))),
        None if placement.row_final_certificate is None
             else placement.row_final_certificate.to_canonical_bytes(),
        None if placement.native_cross_row_final_proof is None else
            (id(placement.native_cross_row_final_proof), placement.native_cross_row_final_proof.sha256))


class LiveRoutingDiagnostic:
    """Live values valid only inside the explicitly requested callback."""

    def __init__(self, *, compact, effective_policy, route_context, snapshot_inputs,
                 allow_engine_initialization=True, capture_endpoint_authority=False):
        self._owner = _execution_owner()
        self._active = True
        self._initialized = False
        self._timing_scope = None
        self._allow_engine_initialization = allow_engine_initialization
        self._compact = compact
        self._policy = effective_policy
        self._context = route_context
        self._inputs = snapshot_inputs
        self._final_placement = route_context.get('final_row_placement')
        self._final_placement_snapshot = (None if allow_engine_initialization else
            _final_placement_value(self._final_placement))
        self._topology_plan = None
        self._topology_plan_snapshot = None
        self._group_contract = None
        self._routing_world = None
        self._routing_world_snapshot = None
        self._gate_route_engine = None
        self._gate_input_objects = None
        self._gate_route_argument_objects = None
        self._terminal_input_objects = dict(snapshot_inputs) if allow_engine_initialization else None
        self._terminal_route_argument_objects = (dict(snapshot_inputs['route_kwargs'])
                                                  if allow_engine_initialization else None)
        self._terminal_pin_objects = ({net: (pins, tuple(pins)) for net, pins in
                                       snapshot_inputs['pins_by_net'].items()}
                                      if allow_engine_initialization else None)
        self._terminal_obstacle_objects = (tuple(snapshot_inputs['obstacles'])
                                           if allow_engine_initialization else None)
        self._terminal_constraint_objects = (tuple(snapshot_inputs['constraints'])
                                             if allow_engine_initialization else None)
        self._diagnostic_topology = route_context.get('topology_plan') if allow_engine_initialization else None
        self._diagnostic_topology_snapshot = None
        if self._diagnostic_topology is not None:
            from clay.layout.route.resource_plan import TopologyPlan
            if (type(self._diagnostic_topology) is not TopologyPlan or
                    self._diagnostic_topology.sha256() != snapshot_inputs['routing_world'].topology_provenance_digest):
                raise ValueError('diagnostic topology plan differs from routing world')
            self._diagnostic_topology_snapshot = self._diagnostic_topology.canonical_json()
        if not allow_engine_initialization:
            from clay.layout.route.resource_plan import TopologyPlan
            plan = route_context.get('topology_plan')
            if type(plan) is not TopologyPlan:
                raise TypeError('pre-route gate requires the exact topology plan')
            if plan.sha256() != snapshot_inputs['routing_world'].topology_provenance_digest:
                raise ValueError('pre-route topology plan differs from routing world')
            self._topology_plan = plan
            self._topology_plan_snapshot = plan.canonical_json()
            self._routing_world = snapshot_inputs['routing_world']
            self._routing_world_snapshot = self._routing_world.canonical_json()
            self._gate_route_engine = snapshot_inputs['route_engine']
            self._gate_input_objects = dict(snapshot_inputs)
            self._gate_route_argument_objects = dict(snapshot_inputs['route_kwargs'])
            self._group_contract = snapshot_inputs['route_kwargs'].get('group_route_contract')
        self._source = reopen_complete_source(snapshot_inputs['source_cell'])
        self._source_hash = self._source['source_sha256']
        self._compact_snapshot = _compact_value(compact)
        self._policy_snapshot = freeze_row_values(asdict(effective_policy))
        self._context_snapshot = freeze_row_values(tuple(route_context.get(k) for k in
            ('intent', 'device_records', 'intent_fact_set', 'source_inventory')))
        self._snapshot = _value(snapshot_prepared_routing_input(**snapshot_inputs))
        self._route_kwargs = _value(snapshot_inputs['route_kwargs'])
        self.check()
        self._endpoint_authority = None
        if capture_endpoint_authority and allow_engine_initialization:
            from clay.layout.route.engine_gridless import GridlessEngine
            selected = snapshot_inputs['route_engine']
            if type(selected) is GridlessEngine:
                self._endpoint_authority = _endpoint_authority_value(selected, self)

    def _check_endpoint_authority(self):
        engine = self._inputs['route_engine']
        if (self._endpoint_authority is None or
                _endpoint_authority_value(engine, self) != self._endpoint_authority):
            raise ValueError('bounded endpoint producer authority changed')

    def __reduce_ex__(self, protocol):
        raise TypeError('live routing diagnostic cannot be serialized')

    def check(self):
        if not self._active or self._owner != _execution_owner():
            raise ValueError('live routing diagnostic is closed or foreign')
        if self._timing_scope is None:
            self._check_inputs()
        else:
            with self._timing_scope.check():
                self._check_inputs()

    @contextmanager
    def timing(self, *, max_checks=8):
        """Explicitly time this terminal capability's first bounded checks.

        Timing adds existing ExecutionSession phase events and checkpoints;
        it does not grant route or physical admission.  The capability remains
        owned by this one synchronous callback.
        """
        if not self._active or self._owner != _execution_owner():
            raise ValueError('live routing diagnostic is closed or foreign')
        if not self._allow_engine_initialization:
            raise ValueError('pre-route gate cannot enable terminal timing')
        if self._timing_scope is not None:
            raise ValueError('live routing diagnostic timing cannot nest')
        from clay.flow.terminal_stage_timing import TerminalStageTiming
        with TerminalStageTiming(max_checks=max_checks) as timing:
            self._timing_scope = timing
            try:
                yield timing
            finally:
                self._timing_scope = None

    def _timed(self, stage):
        timing = self._timing_scope
        return nullcontext() if timing is None else timing.stage(stage)

    def _check_closed_inputs(self):
        if self._active or self._owner != _execution_owner():
            raise ValueError('live routing diagnostic is active or foreign')
        self._check_inputs()

    def _check_inputs(self):
        with self._timed('terminal.input.identity'):
            if self._terminal_input_objects is not None:
                if (set(self._inputs) != set(self._terminal_input_objects) or
                        any(self._inputs[key] is not value
                            for key, value in self._terminal_input_objects.items())):
                    raise ValueError('diagnostic prepared input object identity changed')
                route_kwargs = self._inputs['route_kwargs']
                if (set(route_kwargs) != set(self._terminal_route_argument_objects) or
                        any(route_kwargs[key] is not value
                            for key, value in self._terminal_route_argument_objects.items())):
                    raise ValueError('diagnostic route argument object identity changed')
                pins_by_net = self._inputs['pins_by_net']
                if set(pins_by_net) != set(self._terminal_pin_objects):
                    raise ValueError('diagnostic pin object identity changed')
                for net, (original_pins, original_items) in self._terminal_pin_objects.items():
                    pins = pins_by_net[net]
                    if (pins is not original_pins or len(pins) != len(original_items) or
                            any(actual is not expected for actual, expected in zip(pins, original_items))):
                        raise ValueError('diagnostic pin object identity changed')
                for key, original_items in (('obstacles', self._terminal_obstacle_objects),
                                            ('constraints', self._terminal_constraint_objects)):
                    actual_items = self._inputs[key]
                    if (len(actual_items) != len(original_items) or
                            any(actual is not expected for actual, expected in zip(actual_items, original_items))):
                        raise ValueError(f'diagnostic {key} object identity changed')
        with self._timed('terminal.input.topology'):
            if self._diagnostic_topology is not None and (
                    self._context.get('topology_plan') is not self._diagnostic_topology or
                    self._diagnostic_topology.canonical_json() != self._diagnostic_topology_snapshot or
                    self._diagnostic_topology.sha256() != self._inputs['routing_world'].topology_provenance_digest):
                raise ValueError('diagnostic topology plan changed or differs from routing world')
        if not self._allow_engine_initialization:
            if (self._context.get('final_row_placement') is not self._final_placement
                    or _final_placement_value(self._final_placement) != self._final_placement_snapshot):
                raise ValueError('pre-route final placement changed')
            if (self._inputs['routing_world'] is not self._routing_world
                    or self._routing_world.canonical_json() != self._routing_world_snapshot):
                raise ValueError('pre-route routing world changed')
            if self._inputs['route_engine'] is not self._gate_route_engine:
                raise ValueError('pre-route route engine identity changed')
            if (set(self._inputs) != set(self._gate_input_objects)
                    or any(self._inputs[key] is not value
                           for key, value in self._gate_input_objects.items())):
                raise ValueError('pre-route prepared input object identity changed')
            if (self._context.get('topology_plan') is not self._topology_plan
                    or self._topology_plan.canonical_json() != self._topology_plan_snapshot
                    or self._topology_plan.sha256() != self._inputs['routing_world'].topology_provenance_digest):
                raise ValueError('pre-route topology plan changed or differs from routing world')
            if self._inputs['route_kwargs'].get('group_route_contract') is not self._group_contract:
                raise ValueError('pre-route group contract identity changed')
            route_kwargs = self._inputs['route_kwargs']
            if (set(route_kwargs) != set(self._gate_route_argument_objects)
                    or any(route_kwargs[key] is not value
                           for key, value in self._gate_route_argument_objects.items())):
                raise ValueError('pre-route route argument object identity changed')
        cell, tech, engine = (self._inputs[k] for k in ('source_cell', 'tech', 'route_engine'))
        with self._timed('terminal.input.source'):
            current = reopen_complete_source(cell)
            if (current['source_sha256'] != self._source_hash or current['path'] != self._source['path']
                    or current['pdk_id'] != tech.pdk_id):
                raise ValueError('diagnostic source changed')
        with self._timed('terminal.input.policy'):
            parsed = parse_clay_file(current['path'])
            matches = [c for c in parsed.cell_defs if c.local_name == cell.local_name]
            if len(matches) != 1:
                raise ValueError('diagnostic source cell selection is not unique')
            policy = policy_for_cell(parsed, matches[0])
            expected_engine = route_engine_from_policy(policy)
            if expected_engine is None:
                raise DiagnosticRoutingUnavailable('diagnostic requires a source-declared route engine')
            if (policy != self._policy or engine is None or
                    route_engine_configuration(engine) != route_engine_configuration(expected_engine)):
                raise ValueError('diagnostic engine or policy differs from source')
        with self._timed('terminal.input.context'):
            if freeze_row_values(asdict(self._policy)) != self._policy_snapshot:
                raise ValueError('diagnostic policy changed')
        with self._timed('terminal.input.compact'):
            if _compact_value(self._compact) != self._compact_snapshot:
                raise ValueError('diagnostic compact artwork or placement changed')
        with self._timed('terminal.input.context'):
            if freeze_row_values(tuple(self._context.get(k) for k in
                    ('intent', 'device_records', 'intent_fact_set', 'source_inventory'))) != self._context_snapshot:
                raise ValueError('diagnostic route context changed')
        with self._timed('terminal.input.kwargs'):
            if _value(self._inputs['route_kwargs']) != self._route_kwargs:
                raise ValueError('diagnostic routing arguments changed')
        with self._timed('terminal.input.prepared'):
            if _value(snapshot_prepared_routing_input(**self._inputs)) != self._snapshot:
                raise ValueError('diagnostic prepared routing inputs changed')
        with self._timed('terminal.input.pdk'):
            if self._inputs['routing_world'].pdk_provenance_digest != pdk_provenance_digest(tech):
                raise ValueError('diagnostic PDK binding changed')

    def checked_inputs(self):
        """Return detached terminal inputs after one complete live check.

        This does not initialize the engine or extend this capability's
        lifetime.  The later initialization and region checks remain required.
        """
        self.check()
        if not self._allow_engine_initialization:
            raise ValueError('checked inputs are available only to a terminal diagnostic')
        prepared = json.loads(self._snapshot)
        return {'prepared_input': prepared,
                'route_kwargs': json.loads(self._route_kwargs),
                'source_sha256': self._source_hash,
                'selected_engine_configuration': json.loads(json.dumps(
                    prepared['selected_engine'], allow_nan=False))}

    def snapshot(self):
        """Return a detached canonical snapshot of this checked routing input."""
        self.check()
        return json.loads(self._snapshot)

    def final_placement(self):
        """Expose the checked live final placement, without a final-GDS claim."""
        self.check()
        if self._allow_engine_initialization:
            raise ValueError('final placement is available only to a pre-route gate')
        return self._final_placement

    def planning_snapshot(self):
        """Return detached topology, world provenance and group facts for this gate.

        The routing-world provenance is not the full prepared routing input.
        Pair this with ``snapshot()`` and ``inspect()`` in the same live gate for
        constraints, routing envelope, source-selected engine and other inputs.
        The result is neither a route nor physical admission.
        """
        self.check()
        if self._allow_engine_initialization:
            raise ValueError('planning snapshot is available only to a pre-route gate')
        world = self._inputs['routing_world']
        contract = self._group_contract
        return {'topology_plan': json.loads(self._topology_plan_snapshot),
                'topology_plan_sha256': self._topology_plan.sha256(),
                'routing_world': json.loads(world.canonical_json()),
                'routing_world_sha256': world.sha256(),
                'group_route_contract': None if contract is None else contract.canonical_data(),
                'group_route_contract_sha256': None if contract is None else contract.sha256}

    def inspect(self):
        self.check()
        values = dict(self._inputs)
        values.update(compact=self._compact, effective_policy=self._policy,
                      source_sha256=self._source_hash, source_path=self._source['path'],
                      source_bytes=Path(self._source['path']).read_bytes(),
                      world=self._inputs['routing_world'], engine=self._inputs['route_engine'],
                      candidate_placement=self._context['placement'],
                      candidate_override=self._context['placement_override'],
                      certificate=self._inputs['final_placement_certificate'],
                      group_contract=self._inputs['route_kwargs'].get('group_route_contract'),
                      selected_engine_configuration=route_engine_configuration(self._inputs['route_engine']),
                      intent=self._context['intent'], source_inventory=self._context['source_inventory'],
                      intent_fact_set=self._context['intent_fact_set'],
                      device_records=self._context['device_records'])
        if not self._allow_engine_initialization:
            # A gate may inspect the selected configuration and the immutable
            # snapshot, but should not receive a route entry through this API.
            values.pop('engine', None)
            values.pop('route_engine', None)
        candidate_scope = getattr(self, '_candidate_diagnostic_scope', None)
        if candidate_scope is not None:
            values['candidate_diagnostic_scope'] = dict(candidate_scope)
        return values

    def initialize_before_groups(self, *, snapshot_fact_budget=None, snapshot_deadline=None,
                                 region_facts_path=None, region_output_dir=None,
                                 region_event_budget=None, region_observer=None,
                                 native_endpoint_work=False):
        """Initialize the declared Gridless engine and stop before group routing."""
        if type(native_endpoint_work) is not bool:
            raise TypeError('native endpoint work flag must be boolean')
        if native_endpoint_work:
            LiveRoutingDiagnostic._check_endpoint_authority(self)
        self.check()
        if (snapshot_fact_budget is None) != (snapshot_deadline is None):
            raise ValueError('snapshot fact budget and deadline must be supplied together')
        if snapshot_fact_budget is not None:
            import math
            if type(snapshot_fact_budget) is not int or snapshot_fact_budget <= 0:
                raise ValueError('snapshot fact budget must be a positive integer')
            if type(snapshot_deadline) not in (int, float) or not math.isfinite(snapshot_deadline):
                raise ValueError('snapshot deadline must be a finite monotonic time')
        region_arguments = (region_facts_path, region_output_dir, region_event_budget, region_observer)
        if native_endpoint_work and region_observer is None:
            raise ValueError('native endpoint work requires a region observer')
        if any(value is not None for value in region_arguments):
            if (any(value is None for value in region_arguments) or snapshot_fact_budget is None
                    or not callable(region_observer) or type(region_event_budget) is not int
                    or region_event_budget <= 0):
                raise ValueError('region observation needs paths, event budget, observer and snapshot limits')
        if not self._allow_engine_initialization:
            raise ValueError('pre-route gate cannot initialize the routing engine')
        if self._initialized:
            raise ValueError('diagnostic engine initialization is one-shot')
        self._initialized = True
        if snapshot_deadline is not None:
            import time
            if time.monotonic() >= snapshot_deadline:
                return {'status': 'UNKNOWN_DIAGNOSTIC_DEADLINE', 'initialized_world': {
                    'schema': 'clay.initialized-pre-group-world/v1',
                    'status': 'UNKNOWN_DIAGNOSTIC_DEADLINE'}}
        from clay.layout.route.engine_gridless import GridlessEngine
        engine = self._inputs['route_engine']
        kwargs = self._inputs['route_kwargs']
        contract = kwargs.get('group_route_contract')
        if type(engine) is not GridlessEngine or contract is None:
            return {'status': 'UNKNOWN_ENGINE_OR_GROUP_CONTRACT'}
        class _BeforeGroups(BaseException):
            pass
        class _UnexpectedRoute(BaseException):
            pass
        seen = {}
        def stop(actual_contract, actual_pins, nodes, node_pins, layers, book, **unused):
            if native_endpoint_work:
                LiveRoutingDiagnostic._check_endpoint_authority(self)
                from clay.layout.route.engine_gridless import _Tech
                if type(engine._tech) is not _Tech or engine._tech.pdk is not self._inputs['tech']:
                    raise ValueError('bounded endpoint routing technology differs')
            pins = self._inputs['pins_by_net']
            if (actual_contract is not contract or book is not engine._book or
                    set(actual_pins) != set(pins) or any(
                        Counter(map(id, actual_pins[n])) != Counter(map(id, pins[n])) for n in pins)):
                raise ValueError('diagnostic engine consumed foreign inputs')
            if (any(book.metal.values()) or any(book.via_pads.values()) or book.cuts or book.persistent or
                    any(engine._attempt_metal.values()) or any(engine._attempt_cuts.values()) or
                    engine._solve_expansions_total != 0):
                raise ValueError('diagnostic pre-group Book is not empty')
            seen.update(status='OBSERVED_EMPTY_PRE_GROUP', pin_nets=len(actual_pins),
                        groups=len(contract.ports), expansions=engine._solve_expansions_total)
            if snapshot_fact_budget is not None:
                from clay.layout.route.initialized_world_observation import observe_initialized_world
                with self._timed('terminal.world_facts'):
                    seen['initialized_world'] = observe_initialized_world(
                        engine=engine, pins_by_net=actual_pins, nodes=nodes,
                        node_pins=node_pins, layers=layers, book=book,
                        ordered_nets=unused.get('ordered_nets'),
                        world=self._inputs['routing_world'], contract=contract,
                        constraints=self._inputs['constraints'],
                        topology=self._diagnostic_topology,
                        fact_budget=snapshot_fact_budget, deadline=snapshot_deadline)
                if region_observer is not None:
                    if native_endpoint_work:
                        LiveRoutingDiagnostic._check_endpoint_authority(self)
                    facts = seen['initialized_world']
                    if facts.get('status') != 'OBSERVED' or facts.get('endpoint_mapping') != 'OBSERVED':
                        seen['initialized_regions'] = {'status': 'UNKNOWN_INITIALIZED_FACTS',
                                                       'physical_admission': False}
                    else:
                        from clay.flow.execution_session import current_session
                        from clay.layout.route.initialized_region_observation import (
                            InitializedRegionCapability, RegionObservationLimit,
                        )
                        session = current_session()
                        with self._timed('terminal.region_constructor'):
                            kwargs = dict(engine=engine, book=book, nodes=nodes, node_pins=node_pins,
                                layers=layers, contract=contract, facts=facts,
                                prepared=json.loads(self._snapshot), route_kwargs=json.loads(self._route_kwargs),
                                source_sha256=self._source_hash,
                                invocation_id=None if session is None else session.invocation_id,
                                facts_path=region_facts_path, output_dir=region_output_dir,
                                deadline=snapshot_deadline, event_budget=region_event_budget,
                                input_check=self.check, timing=self._timing_scope)
                            if native_endpoint_work:
                                # Bind the Book supplied by this invocation before
                                # the region is exposed. Comparing the engine and
                                # region to each other permits both references to
                                # be replaced with an equal but foreign Book.
                                def endpoint_authority():
                                    LiveRoutingDiagnostic._check_endpoint_authority(self)
                                    if engine._book is not book or region_cap._book is not book:
                                        raise ValueError('bounded endpoint Book identity changed')
                                kwargs['native_endpoint_check'] = endpoint_authority
                            region_cap = InitializedRegionCapability(**kwargs)
                        if native_endpoint_work:
                            # Capture these before exposing the region object. An
                            # instance replacement must not keep a capability
                            # alive or manufacture a terminal result.
                            trusted_finish = InitializedRegionCapability.finish
                            trusted_close = InitializedRegionCapability.close

                            def finish_region(status):
                                LiveRoutingDiagnostic._check_endpoint_authority(self)
                                if any(name in vars(region_cap) for name in
                                       ('_timed', '_check', '_scope', '_checkpoint', '_bounded_endpoint', '_publish',
                                        'endpoint', 'finish', 'close')):
                                    raise ValueError('bounded endpoint lifecycle method changed')
                                return trusted_finish(region_cap, status)

                            try:
                                try:
                                    region_observer(json.loads(json.dumps(facts)), region_cap)
                                except RegionObservationLimit as error:
                                    if getattr(error, '_native_endpoint_cleanup_failed', False):
                                        raise
                                    seen['initialized_regions'] = finish_region(str(error))
                                else:
                                    seen['initialized_regions'] = finish_region('UNKNOWN_INCOMPLETE')
                            except BaseException as original:
                                try:
                                    trusted_close(region_cap)
                                except BaseException as ending:
                                    from clay.flow.control_stop import record_unwind_failure
                                    try:
                                        record_unwind_failure(original, ending)
                                    except BaseException as recording_error:
                                        try:
                                            original.add_note('Recording the region close failure also failed: '
                                                              + type(recording_error).__name__)
                                        except BaseException:
                                            pass
                                raise
                            else:
                                trusted_close(region_cap)
                            LiveRoutingDiagnostic._check_endpoint_authority(self)
                        else:
                            try:
                                try:
                                    region_observer(json.loads(json.dumps(facts)), region_cap)
                                except RegionObservationLimit as error:
                                    seen['initialized_regions'] = region_cap.finish(str(error))
                                else:
                                    seen['initialized_regions'] = region_cap.finish()
                            finally:
                                region_cap.close()
            raise _BeforeGroups()
        def forbid(*args, **kwargs):
            raise _UnexpectedRoute('routing began before diagnostic stop')
        missing = object()
        previous = {name: engine.__dict__.get(name, missing) for name in
                    ('_reserve_group_route_landings', '_route_net')}
        engine._reserve_group_route_landings = stop
        engine._route_net = forbid
        try:
            with self._timed('terminal.engine_initialize_to_group_stop'):
                try:
                    engine.route(self._inputs['pins_by_net'], self._inputs['obstacles'],
                                 self._inputs['constraints'], self._inputs['tech'], **kwargs)
                except _BeforeGroups:
                    pass
                else:
                    raise ValueError('diagnostic engine did not reach the pre-group boundary')
        finally:
            for name, old in previous.items():
                if old is missing:
                    engine.__dict__.pop(name, None)
                else:
                    engine.__dict__[name] = old
        self.check()
        return seen


def observe_prepared_routing(callback, *, compact, effective_policy, route_context, snapshot_inputs):
    if not callable(callback):
        raise TypeError('diagnostic routing observer must be callable')
    if type(callback) is NativeEndpointWorkObserver:
        capability = LiveRoutingDiagnostic(compact=compact, effective_policy=effective_policy,
            route_context=route_context, snapshot_inputs=snapshot_inputs,
            capture_endpoint_authority=True)
    else:
        capability = LiveRoutingDiagnostic(compact=compact, effective_policy=effective_policy,
            route_context=route_context, snapshot_inputs=snapshot_inputs)
    try:
        try:
            callback(capability)
            capability.check()
        except BaseException as original:
            try:
                capability.check()
            except BaseException as ending:
                from clay.flow.control_stop import record_unwind_failure
                if type(callback) is NativeEndpointWorkObserver:
                    try:
                        record_unwind_failure(original, ending)
                    except BaseException as recording_error:
                        try:
                            original.add_note('Recording the terminal check failure also failed: '
                                              + type(recording_error).__name__)
                        except BaseException:
                            pass
                else:
                    record_unwind_failure(original, ending)
            raise
    finally:
        capability._active = False
    stop=DiagnosticRoutingObserved('diagnostic observation stopped before routing')
    from clay.layout.place.candidate_batch_completion import publish_terminal_completion
    publish_terminal_completion(cap=capability,callback=callback,stop=stop)
    raise stop


def gate_prepared_routing(gate, *, compact, effective_policy, route_context, snapshot_inputs):
    """Permit the immediately following route only after an unchanged live check.

    The verifier is trusted in-process code, just like the diagnostic observer.
    It is deliberately not an external attestation or a replacement for DRC,
    LVS, final intent checks, or the declared routing policy.
    """
    if type(gate) is not PreRouteGate:
        raise TypeError('pre-route gate must be an exact PreRouteGate')
    from contextlib import nullcontext
    final = route_context.get('final_row_placement')
    program = getattr(final, 'row_program', None)
    materialization = getattr(final, 'row_materialization', None)
    projection = getattr(materialization, 'source_projection', None)
    witness = getattr(projection, 'native_witness', None)
    scope = nullcontext()
    if witness is not None:
        from clay.layout.place.row_replay_session import row_materialization_replay_scope
        scope = row_materialization_replay_scope(program=program, materialization=materialization,
            tech=snapshot_inputs['tech'], source_cell=snapshot_inputs['source_cell'])
    with scope:
        return _gate_prepared_routing(gate, compact=compact, effective_policy=effective_policy,
                                     route_context=route_context, snapshot_inputs=snapshot_inputs)


def _gate_prepared_routing(gate, *, compact, effective_policy, route_context, snapshot_inputs):
    capability = LiveRoutingDiagnostic(compact=compact, effective_policy=effective_policy,
        route_context=route_context, snapshot_inputs=snapshot_inputs, allow_engine_initialization=False)
    try:
        try:
            decision = gate.verify(capability)
            if type(decision) is not PreRouteGateDecision:
                raise TypeError('pre-route gate must return an exact PreRouteGateDecision')
            decision.__post_init__()
            capability.check()
            if decision.status != 'PASS':
                raise PreRouteGateRejected(decision)
        except BaseException as original:
            try:
                capability.check()
            except BaseException as ending:
                from clay.flow.control_stop import record_unwind_failure
                record_unwind_failure(original, ending)
            raise
    finally:
        capability._active = False
    return decision
