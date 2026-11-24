"""Terminal-only observation of the initialized engine's conservative regions.

Each published part is a bounded observation of one world and width scope.
It is not an exhaustive routing graph, a capacity, or physical admission.
"""
from dataclasses import asdict
from contextlib import nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
import time
import uuid

from clay.layout.route.canonical_input_capture import _json_value, route_engine_configuration
from clay.layout.route.group_fallback_checkpoint import _canonical
from clay.layout.route.terminal_identity import pin_fact


class RegionObservationLimit(Exception):
    """An explicitly bounded terminal region observation stopped."""


_NATIVE_ENDPOINT_EVENTS = frozenset({
    'domain.enter', 'domain.seed_atom', 'domain.throat_pair', 'domain.region_atom',
    'component.enter', 'component.source_atom', 'component.index_atom',
    'component.seed_atom', 'component.visit_atom', 'component.neighbour', 'component.enqueue',
    'index.add.bucket', 'index.query.bucket', 'index.query.row',
})

_WARM_OWNED_EVENTS = frozenset({
    'warm.source_atom', 'warm.atom_identity', 'warm.rebuild_atom',
    'warm.map_item', 'warm.bucket', 'warm.rebuilt_bucket',
    'warm.bucket_row', 'warm.rebuilt_row', 'warm.owner',
})


def publish_pending(path, payload):
    """Publish complete canonical bytes atomically without overwriting a file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError('preserve the existing diagnostic publication')
    raw = (json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')
    tmp = path.parent / ('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link is both atomic and no-clobber on the same filesystem.
        os.link(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {'path': str(path.resolve(strict=True)), 'sha256': sha256(raw).hexdigest(),
            'size_bytes': len(raw)}


def _book_value(book):
    return _canonical({name: getattr(book, name) for name in
                       ('metal', 'via_pads', 'cuts', 'persistent')}, set())


def _spatial_value(index):
    """Snapshot the actual query buckets, including rect identity dedup keys."""
    from clay.layout.route.engine_gridless import _SpatialIndex
    if type(index) is not _SpatialIndex:
        raise TypeError('initialized region index is not the native spatial index')
    return (id(index), index.cell, id(index.buckets), tuple(
        (key, id(rows), tuple((owner, tuple(rect), id(rect)) for owner, rect in rows))
        for key, rows in sorted(index.buckets.items())))


def _index_mapping_value(mapping):
    return (id(mapping), tuple((key, _spatial_value(index))
                               for key, index in sorted(mapping.items())))


def _wide_value(mapping):
    return (id(mapping), tuple((gi, tuple((spacing, _spatial_value(core),
                    _spatial_value(window), _spatial_value(attached))
                    for spacing, core, window, attached in rows))
                    for gi, rows in sorted(mapping.items())))


def _owned_cache_value(engine):
    cache = engine._owned_terminal_group_indexes
    result = []
    for key, (source, atoms, spatial, rect_to_index) in sorted(cache.items()):
        if source is not engine._owned_terminal_atoms:
            raise ValueError('initialized region owned index has a foreign source')
        result.append((key, id(source), id(atoms), _canonical(atoms, set()),
                       _spatial_value(spatial), id(rect_to_index), _canonical(rect_to_index, set())))
    return id(cache), tuple(result)


def _validate_owned_entry(engine, key, entry, *, diagnostic_work=None):
    """Replay the source-owned atom filter and native spatial index builder."""
    from clay.layout.route.engine_gridless import _SpatialIndex, _norm
    if diagnostic_work is not None:
        if not callable(diagnostic_work):
            raise TypeError('diagnostic work must be callable')

        def validate_metered():
            """Rebuild locally and meter each expandable Python traversal."""
            from clay.layout.route.engine_gridless import _OwnedTerminalAtom
            if (type(key) is not tuple or len(key) != 2 or type(key[0]) is not str
                    or type(key[1]) is not int or type(entry) is not tuple or len(entry) != 4):
                raise ValueError('initialized region owned index has unsupported operands')
            source, atoms, spatial, rect_to_index = entry
            if source is not engine._owned_terminal_atoms:
                raise ValueError('initialized region owned index has a foreign source')
            if type(source) is not tuple or type(atoms) is not tuple:
                raise ValueError('initialized region owned index has unsupported atom containers')

            def work(event):
                engine._endpoint_diagnostic_step(key[0], diagnostic_work, event)

            def native_rect(rect):
                return (type(rect) is tuple and len(rect) == 4
                        and all(type(value) is int for value in rect))

            expected_rows = []
            for ordinal, atom in enumerate(source):
                work('warm.source_atom')
                if ordinal % 128 == 0:
                    engine._watchdog_check(key[0], 'initialized_region_owned_index_verify')
                if (type(atom) is not _OwnedTerminalAtom or type(atom.owner) is not str
                        or type(atom.gi) is not int or not native_rect(atom.rect)):
                    raise ValueError('initialized region owned index has unsupported source atoms')
                if atom.owner == key[0] and atom.gi == key[1]:
                    expected_rows.append(atom)
            if len(atoms) != len(expected_rows):
                raise ValueError('initialized region owned index omitted or replaced source atoms')
            for atom, expected_atom in zip(atoms, expected_rows):
                work('warm.atom_identity')
                if atom is not expected_atom:
                    raise ValueError('initialized region owned index omitted or replaced source atoms')
            rebuilt = _SpatialIndex()
            if (type(spatial) is not _SpatialIndex or spatial.cell != rebuilt.cell
                    or type(spatial.buckets) is not dict or type(rect_to_index) is not dict):
                raise ValueError('initialized region owned index has unsupported spatial data')
            expected_map = {}
            for index, atom in enumerate(expected_rows):
                work('warm.rebuild_atom')
                if index % 128 == 0:
                    engine._watchdog_check(key[0], 'initialized_region_owned_index_verify')
                rect = tuple(_norm(atom.rect))
                expected_map[rect] = index
                rebuilt.add(str(index), rect, diagnostic_work=work)
            if len(rect_to_index) != len(expected_map):
                raise ValueError('initialized region owned index differs from native atom projection')
            for rect, index in rect_to_index.items():
                work('warm.map_item')
                if (not native_rect(rect) or type(index) is not int
                        or rect not in expected_map or expected_map[rect] != index):
                    raise ValueError('initialized region owned index differs from native atom projection')
            if len(spatial.buckets) != len(rebuilt.buckets):
                raise ValueError('initialized region owned index differs from native atom projection')
            refs = {}
            for ordinal, (bucket, rows) in enumerate(spatial.buckets.items()):
                work('warm.bucket')
                work('warm.rebuilt_bucket')
                if ordinal % 128 == 0:
                    engine._watchdog_check(key[0], 'initialized_region_owned_index_verify')
                if (type(bucket) is not tuple or len(bucket) != 2
                        or any(type(value) is not int for value in bucket)
                        or type(rows) is not list or bucket not in rebuilt.buckets
                        or len(rows) != len(rebuilt.buckets[bucket])):
                    raise ValueError('initialized region owned index differs from native atom projection')
                for row, expected_row in zip(rows, rebuilt.buckets[bucket]):
                    work('warm.bucket_row')
                    work('warm.rebuilt_row')
                    if (type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
                            or not native_rect(row[1]) or row != expected_row):
                        raise ValueError('initialized region owned index differs from native atom projection')
                    refs.setdefault(row[0], set()).add(id(row[1]))
            for identities in refs.values():
                work('warm.owner')
                if len(identities) != 1:
                    raise ValueError('initialized region owned index rectangle identities differ')
            return None

        return validate_metered()
    source, atoms, spatial, rect_to_index = entry
    if source is not engine._owned_terminal_atoms:
        raise ValueError('initialized region owned index has a foreign source')
    expected_rows = []
    for ordinal, atom in enumerate(source):
        if ordinal % 128 == 0:
            engine._watchdog_check(key[0], 'initialized_region_owned_index_verify')
        if atom.owner == key[0] and atom.gi == key[1]:
            expected_rows.append(atom)
    expected = tuple(expected_rows)
    if len(atoms) != len(expected) or any(a is not b for a, b in zip(atoms, expected)):
        raise ValueError('initialized region owned index omitted or replaced source atoms')
    rebuilt = _SpatialIndex()
    if type(spatial) is not _SpatialIndex or spatial.cell != rebuilt.cell:
        raise ValueError('initialized region owned index has a foreign spatial grid')
    expected_map = {}
    for index, atom in enumerate(expected):
        if index % 128 == 0:
            engine._watchdog_check(key[0], 'initialized_region_owned_index_verify')
        rect = tuple(_norm(atom.rect))
        expected_map[rect] = index
        rebuilt.add(str(index), rect)
    def value(index):
        return {key: tuple((owner, tuple(rect)) for owner, rect in rows)
                for key, rows in index.buckets.items()}
    if type(rect_to_index) is not dict or rect_to_index != expected_map or value(spatial) != value(rebuilt):
        raise ValueError('initialized region owned index differs from native atom projection')
    # _SpatialIndex.query deduplicates by owner and rectangle object identity.
    # Native add inserts one identical rectangle object into every covered bucket.
    refs = {}
    for ordinal, rows in enumerate(spatial.buckets.values()):
        if ordinal % 128 == 0:
            engine._watchdog_check(key[0], 'initialized_region_owned_index_verify')
        for owner, rect in rows:
            refs.setdefault(owner, set()).add(id(rect))
    if any(len(identities) != 1 for identities in refs.values()):
        raise ValueError('initialized region owned index rectangle identities differ')


_CONSUMER_METHODS = (
    '_planar_endpoint_domain', '_unrequired_planar_endpoint_domain', '_owned_terminal_component', '_planar_event_geometry_for_layer',
    '_planar_via_event_regions', '_wide_indexes', '_ensure_fixed_wide_geometry_indexes',
    '_local_fixed_cuts', '_rail_local_half_width_overrides', '_half_width_scope',
    '_hw', '_watchdog_check', '_via_cut_extent', '_owned_landing_has_minimum_throat',
    '_point_in_closed_rect',
)


def _method_token(value):
    fn = getattr(value, '__func__', value)
    owner = getattr(value, '__self__', None)
    owner_cls = None if owner is None else (type(owner).__module__, type(owner).__qualname__)
    return (owner_cls, getattr(fn, '__module__', None), getattr(fn, '__qualname__', getattr(fn, '__name__', None)))


def _method_value(engine):
    from clay.layout.route import engine_gridless, region_event_geometry
    return (tuple((name, _method_token(getattr(engine, name))) for name in _CONSUMER_METHODS),
            tuple((name, _method_token(getattr(engine._tech, name)))
                  for name in ('gi_min_width', 'pin_seed', 'resolve_layer_to_gi', 'resolve_layer_to_pdk')),
            _method_token(engine_gridless._SpatialIndex.add),
            _method_token(engine_gridless._SpatialIndex.query),
            _method_token(engine_gridless._norm),
            _method_token(region_event_geometry.conservative_center_region),
            _method_token(region_event_geometry.build_region_geometry),
            _method_token(region_event_geometry.rectangle_union_cover))


_CONSUMER_VALUES = (
    '_bounds', '_routing_envelope', '_space', '_half_w', '_net_half_w', '_rail_by_net',
    '_manufacturing_grid_nm', '_intent_spacing_rules', '_via_cut', '_via_pad_half',
    '_via_pad_half1', '_via_space', '_via_pitch', '_n_layers', '_max_gi',
    '_obs_by_gi', '_obs_provenance', '_obs_cuts', '_obs_cut_provenance',
    '_fixed_deck_literal_rects', '_abstract_rects', '_owned_terminal_atoms',
    '_required_wire_endpoints', '_group_route_required_states', '_attempt_cuts', '_attempt_metal',
    '_mim_carve_bottom_blockers_by_gi', '_mim_forbidden_cut_touch_by_lo',
    '_mim_bottom_routing_clearance_by_gi', '_mim_top_via_marker_clearance_by_lo',
    '_fixed_wide_source_key',
)
_CONSUMER_INDEXES = ('_obs_idx', '_literal_planar_idx', '_obs_cut_idx', '_cut_idx',
                     '_metal_idx', '_via_pad_idx')


def _consumer_value(engine, timing=None):
    """Freeze the objects actually read by the native region consumers."""
    stage = (lambda name: nullcontext()) if timing is None else timing.stage
    with stage('region.consumer.values'):
        values = []
        for name in _CONSUMER_VALUES:
            value = getattr(engine, name)
            # Immutables are compared by value; containers and typed objects also
            # retain identity so same-content replacements cannot become authority.
            identity = None if value is None or type(value) in (int, str, float, bool) else id(value)
            values.append((name, identity, _canonical(value, set())))
    with stage('region.consumer.indexes'):
        indexes = tuple((name, _index_mapping_value(getattr(engine, name)))
                        for name in _CONSUMER_INDEXES)
    with stage('region.consumer.tech_wide'):
        tech = engine._tech
        tech_values = {name: value for name, value in vars(tech).items() if name != 'pdk'}
        return (tuple(values), indexes, id(tech), id(tech.pdk), _canonical(tech_values, set()),
                _wide_value(engine._fixed_wide_geometry_indexes))


class InitializedRegionCapability:
    """Narrow live capability; valid only inside the one terminal hook."""

    def __init__(self, *, engine, book, nodes, node_pins, layers, contract, facts,
                 prepared, route_kwargs, source_sha256, invocation_id, facts_path, output_dir,
                 deadline, event_budget, input_check, timing=None, native_endpoint_check=None):
        if type(event_budget) is not int or event_budget <= 0:
            raise ValueError('region event budget must be a positive integer')
        if facts.get('status') != 'OBSERVED' or facts.get('endpoint_mapping') != 'OBSERVED':
            raise ValueError('initialized region observation requires mapped facts')
        if (engine._book is not book or any(book.metal.values()) or any(book.via_pads.values())
                or book.cuts or book.persistent):
            raise ValueError('initialized region observation requires the actual empty Book')
        self._active = True
        self._engine, self._book = engine, book
        self._nodes, self._node_pins, self._layers = nodes, node_pins, layers
        if set(nodes) != set(node_pins) or set(nodes) != set(layers):
            raise ValueError('initialized region net inputs differ')
        self._net_original = {
            net: (nodes[net], tuple(nodes[net]), node_pins[net], tuple(node_pins[net]),
                  layers[net], tuple(layers[net])) for net in nodes}
        self._contract, self._facts = contract, facts
        self._facts_canonical = json.dumps(facts, sort_keys=True, separators=(',', ':'),
                                           ensure_ascii=True, allow_nan=False)
        self._deadline, self._budget, self._input_check = deadline, event_budget, input_check
        self._input_check_original = input_check
        self._native_endpoint_check = native_endpoint_check
        self._native_endpoint_check_original = native_endpoint_check
        self._native_endpoint_counts = {}
        self._native_endpoint_count_snapshot = ()
        self._native_ledger_count = 0
        self._native_limits_original = (deadline, event_budget)
        if timing is not None:
            from clay.flow.terminal_stage_timing import TerminalStageTiming
            if type(timing) is not TerminalStageTiming:
                raise TypeError('region timing must be an exact terminal timing scope')
        self._timing = timing
        self._count = 0
        self._started = time.monotonic()
        self._input_check_calls = 0
        self._input_check_seconds = 0.0
        self._consumer_check_seconds = 0.0
        self._graphs, self._published, self._endpoints, self._vias = {}, [], set(), set()
        self._hierarchy_published = False
        self._group_nets = {port.net for port in contract.ports}
        self._group_nets_original = frozenset(self._group_nets)
        self._book_original = _book_value(book)
        book_bytes = json.dumps(self._book_original, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=True, allow_nan=False).encode('ascii')
        self._book_sha = sha256(book_bytes).hexdigest()
        self._engine_configuration = route_engine_configuration(engine)
        self._watchdog = engine._watchdog_deadline
        self._expansions = engine._solve_expansions_total
        self._widths_present = '_active_half_width_overrides' in engine.__dict__
        self._widths = dict(getattr(engine, '_active_half_width_overrides', {}))
        if self._widths:
            raise ValueError('pre-group region observation requires the unscoped world')
        self._output = Path(output_dir)
        if self._output.exists():
            raise FileExistsError('preserve the existing region output directory')
        payload = {'schema': 'clay.initialized-region-facts/v1',
                   'status': 'PENDING_RECEIPT', 'physical_admission': False,
                   'source_sha256': source_sha256, 'invocation_id': invocation_id,
                   'book_canonical': self._book_original, 'book_sha256': self._book_sha,
                   'prepared_input': prepared, 'route_kwargs': route_kwargs,
                   'initialized_world': facts}
        self.facts_publication = publish_pending(facts_path, payload)
        self._facts_publication_original = dict(self.facts_publication)
        self._facts_sha = self.facts_publication['sha256']
        self._output.mkdir(exist_ok=False)
        self._consumer_original = _consumer_value(engine, timing)
        self._method_original = _method_value(engine)
        self._owned_cache_original = _owned_cache_value(engine)

    def __reduce_ex__(self, protocol):
        raise TypeError('initialized region capability cannot be serialized')

    def _timed(self, stage):
        return nullcontext() if self._timing is None else self._timing.stage(stage)

    def _check(self):
        if not self._active:
            raise ValueError('initialized region capability is closed')
        if self._native_endpoint_check_original is not None and (
                self._native_endpoint_check is not self._native_endpoint_check_original
                or self._input_check is not self._input_check_original
                or (self._deadline, self._budget) != self._native_limits_original
                or self._count != self._native_ledger_count
                or tuple(sorted(self._native_endpoint_counts.items())) != self._native_endpoint_count_snapshot):
            raise ValueError('bounded endpoint authority or ledger changed')
        if self._native_endpoint_check_original is not None:
            self._native_endpoint_check_original()
        with self._timed('region.check.input'):
            started = time.monotonic()
            try:
                self._input_check()
            finally:
                self._input_check_seconds += time.monotonic() - started
                self._input_check_calls += 1
        with self._timed('region.check.facts'):
            if (self._engine._book is not self._book or _book_value(self._book) != self._book_original
                    or self._engine._solve_expansions_total != self._expansions
                    or self._engine._watchdog_deadline != self._watchdog
                    or route_engine_configuration(self._engine) != self._engine_configuration
                    or ('_active_half_width_overrides' in self._engine.__dict__) != self._widths_present
                    or dict(getattr(self._engine, '_active_half_width_overrides', {})) != self._widths):
                raise ValueError('initialized region engine, Book or width scope changed')
            if (json.dumps(self._facts, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                           allow_nan=False) != self._facts_canonical):
                raise ValueError('initialized region source facts changed')
        with self._timed('region.check.consumers'):
            started = time.monotonic()
            try:
                consumer_changed = _consumer_value(self._engine, self._timing) != self._consumer_original
            finally:
                self._consumer_check_seconds += time.monotonic() - started
            if consumer_changed:
                raise ValueError('initialized region consumer world changed')
            # The original reader lazily points this cache at the fixed wide index
            # when the active Book is empty. Before its first use None is valid;
            # afterwards it must be exactly that fixed source-derived index.
            active = self._engine._wide_index_metal_index
            if (active is not None and (active is not self._engine._metal_idx or
                    self._engine._wide_geometry_indexes is not self._engine._fixed_wide_geometry_indexes)):
                raise ValueError('initialized region wide index is foreign')
            if _owned_cache_value(self._engine) != self._owned_cache_original:
                raise ValueError('initialized region owned terminal index changed')
            if _method_value(self._engine) != self._method_original:
                raise ValueError('initialized region producer method changed')
            if (set(self._nodes) != set(self._net_original) or set(self._node_pins) != set(self._net_original)
                    or set(self._layers) != set(self._net_original)):
                raise ValueError('initialized region net state changed')
            for net, (nodes, node_values, pins, pin_values, layers, layer_values) in self._net_original.items():
                if (self._nodes[net] is not nodes or tuple(nodes) != node_values
                        or self._node_pins[net] is not pins or len(pins) != len(pin_values)
                        or any(actual is not old for actual, old in zip(pins, pin_values))
                        or self._layers[net] is not layers or tuple(layers) != layer_values):
                    raise ValueError('initialized region net state changed')

    def _checkpoint(self, net):
        if self._native_endpoint_check_original is not None and self._count != self._native_ledger_count:
            raise ValueError('bounded endpoint ledger changed')
        self._engine._watchdog_check(net, 'initialized_region_observation')
        if time.monotonic() >= self._deadline:
            raise RegionObservationLimit('UNKNOWN_DIAGNOSTIC_DEADLINE')
        self._count += 1
        if self._native_endpoint_check_original is not None:
            self._native_ledger_count = self._count
        if self._count > self._budget:
            raise RegionObservationLimit('UNKNOWN_REGION_EVENT_BUDGET')

    def _bounded_endpoint(self, net, node, overrides):
        """One cold-cache endpoint under the existing region event/deadline ledger.

        Full source/Book/input checks, sorting, serialization and native atomic
        work remain outside this event ledger. The observer is trusted Python,
        not an isolated adversary.
        """
        from clay.layout.place.native_stage_observation import _execution_owner
        from clay.flow.control_stop import record_unwind_failure
        check = self._native_endpoint_check_original
        if check is None or self._native_endpoint_check is not check:
            raise ValueError('bounded endpoint authority is unavailable or changed')
        check()
        engine = self._engine
        if self._input_check is not self._input_check_original:
            raise ValueError('bounded endpoint input check changed')
        cache = engine._owned_terminal_group_indexes
        if type(cache) is not dict:
            raise ValueError('bounded endpoint cache is not the native mapping')
        if (str(net), int(node[2])) in cache:
            raise RegionObservationLimit('UNKNOWN_NATIVE_WARM_CACHE')
        owner = _execution_owner()
        deadline, budget, counts = self._deadline, self._budget, self._native_endpoint_counts
        expected = self._count

        def debit(event):
            nonlocal expected
            check()
            if (not self._active or _execution_owner() != owner or self._engine is not engine
                    or self._input_check is not self._input_check_original
                    or self._native_endpoint_check is not check
                    or self._deadline != deadline or self._budget != budget
                    or self._native_endpoint_counts is not counts or self._count != expected
                    or self._native_ledger_count != expected
                    or tuple(sorted(counts.items())) != self._native_endpoint_count_snapshot):
                raise ValueError('bounded endpoint scope or ledger changed')
            if type(event) is not str or event not in _NATIVE_ENDPOINT_EVENTS:
                raise ValueError('bounded endpoint event differs')
            if time.monotonic() >= deadline:
                raise RegionObservationLimit('UNKNOWN_DIAGNOSTIC_DEADLINE')
            if expected >= budget:
                raise RegionObservationLimit('UNKNOWN_REGION_EVENT_BUDGET')
            expected += 1
            self._count = expected
            self._native_ledger_count = expected
            counts[event] = counts.get(event, 0) + 1
            self._native_endpoint_count_snapshot = tuple(sorted(counts.items()))

        scope = engine._half_width_scope(overrides)
        scope.__enter__()
        try:
            half = engine._hw(net, node[2])
            domain = engine._planar_endpoint_domain(net, node, diagnostic_work=debit)
        except BaseException as original:
            try:
                scope.__exit__(type(original), original, original.__traceback__)
            except BaseException as ending:
                if isinstance(original, RegionObservationLimit):
                    original._native_endpoint_cleanup_failed = True
                try:
                    record_unwind_failure(original, ending)
                except BaseException as recording_error:
                    try:
                        original.add_note('Recording the endpoint cleanup failure also failed: '
                                          + type(recording_error).__name__)
                    except BaseException:
                        pass
            raise
        else:
            scope.__exit__(None, None, None)
        check()
        if self._count != expected or self._native_endpoint_counts is not counts:
            raise ValueError('bounded endpoint ledger changed')
        return half, domain

    def _scope(self, net, scope):
        if type(net) is not str or net not in self._layers:
            raise ValueError('region net is not in the initialized world')
        if scope not in ('global', 'group') or (scope == 'group' and net not in self._group_nets):
            raise ValueError('region width scope is not authorized by the group contract')
        if scope == 'global':
            return {}
        return self._engine._rail_local_half_width_overrides(
            net, self._layers[net], self._engine._rail_by_net.get(net, ()))

    def inventory(self):
        with self._timed('region.inventory'):
            self._check()
            return self._inventory_value()

    def _inventory_value(self):
        """Copy the inventory only within a surrounding authority check."""
        return {'ordered_nets': list(self._facts['ordered_nets']),
                'nets': [{'net': net, 'allowed_gis': list(self._layers[net]),
                          'scopes': ['global', 'group'] if net in self._group_nets else ['global'],
                          'endpoints': len(self._nodes[net])} for net in sorted(self._layers)],
                'facts_publication': dict(self.facts_publication)}

    def _check_inventory_authority(self):
        if (self._group_nets != self._group_nets_original
                or self.facts_publication != self._facts_publication_original
                or self._facts_publication_original['sha256'] != self._facts_sha):
            raise ValueError('initialized region inventory authority changed')

    def _publish(self, kind, net, scope, data):
        self._checkpoint(net)
        part = {'schema': 'clay.initialized-region-part/v1', 'status': 'PENDING_RECEIPT',
                'physical_admission': False, 'kind': kind, 'net': net, 'width_scope': scope,
                'facts_sha256': self._facts_sha, 'book_status': 'OBSERVED_EMPTY_PRE_GROUP',
                'book_sha256': self._book_sha,
                'candidate_completeness': 'UNKNOWN', 'routing_capacity': 'UNKNOWN', **data}
        publication = publish_pending(self._output / f'{len(self._published):06d}.json', part)
        self._published.append(publication)
        self._checkpoint(net)
        return publication

    def hierarchy(self):
        """Publish the source group/net incidence without inventing portals."""
        _, publication = self._hierarchy_observation(include_inventory=False)
        return publication

    def hierarchy_with_inventory(self):
        """Return inventory and hierarchy from one checked, terminal observation.

        Unlike separate calls to inventory() then hierarchy(), hierarchy is
        published before the caller receives either result. A stop before that
        publication leaves no hierarchy part; a stop after publication retains
        the pending part without returning a result, as in hierarchy(). No user
        callback runs between the inventory copy and hierarchy pre/post checks.
        """
        return self._hierarchy_observation(include_inventory=True)

    def _hierarchy_observation(self, *, include_inventory):
        with self._timed('region.hierarchy.precheck'):
            self._check()
        if include_inventory:
            self._check_inventory_authority()
        if self._hierarchy_published:
            raise ValueError('source hierarchy already observed')
        with self._timed('region.hierarchy.mapping'):
            inventory = self._inventory_value() if include_inventory else None
            from clay.layout.route.engine_gridless import _group_route_pin_key
            pin_rows = {}
            index = {}
            unresolved = []
            for net in sorted(self._layers):
                rows = []
                for ordinal, (node, pin) in enumerate(zip(self._nodes[net], self._node_pins[net])):
                    self._checkpoint(net)
                    row = {'index': ordinal, 'pin_fact': pin_fact(pin), 'node': node}
                    rows.append(row)
                    try:
                        key = _group_route_pin_key(net, pin)
                    except (TypeError, ValueError, AttributeError):
                        unresolved.append({'net': net, 'index': ordinal, 'reason': 'UNKNOWN_PIN_KEY'})
                        continue
                    index.setdefault(key, []).append((net, ordinal))
                pin_rows[net] = rows
            groups = {}
            used = set()
            for port in self._contract.ports:
                self._checkpoint(port.net)
                attached = []
                for access in port.accesses:
                    self._checkpoint(port.net)
                    key = _group_route_pin_key(port.net, access)
                    matches = index.get(key, ())
                    if len(matches) != 1 or matches[0] in used:
                        unresolved.append({'group': port.group, 'net': port.net,
                                           'access': access.canonical_data(),
                                           'reason': 'UNKNOWN_ACCESS_IDENTITY'})
                        continue
                    used.add(matches[0])
                    attached.append(matches[0][1])
                groups.setdefault(port.group, []).append({
                    'net': port.net, 'envelope_nm': port.envelope_nm,
                    'terminals': port.terminals,
                    'accesses': [access.canonical_data() for access in port.accesses],
                    'endpoint_indices': attached})
            nets = []
            for net in sorted(self._layers):
                self._checkpoint(net)
                nets.append({'net': net, 'endpoints': pin_rows[net],
                             'ungrouped_endpoint_indices': [row['index'] for row in pin_rows[net]
                                                            if (net, row['index']) not in used],
                             'group_ports': [group for group in sorted(groups)
                                             if any(port['net'] == net for port in groups[group])],
                             'rails': self._engine._rail_by_net.get(net, ())})
        with self._timed('region.hierarchy.postcheck'):
            self._check()
        if include_inventory:
            self._check_inventory_authority()
        with self._timed('region.hierarchy.publish'):
            publication = self._publish('source_hierarchy', '', 'global', {
                'groups': [{'group': group, 'ports': groups[group]} for group in sorted(groups)],
                'nets': nets, 'topology_plan': self._facts['topology_plan'],
                'mapping_status': 'UNKNOWN' if unresolved else 'OBSERVED',
                'unresolved': unresolved, 'physical_connectivity': 'UNKNOWN'})
        self._hierarchy_published = True
        return inventory, publication

    def planar(self, net, gi, scope='global'):
        self._check()
        overrides = self._scope(net, scope)
        if type(gi) is not int or gi not in self._layers[net]:
            raise ValueError('region layer is not allowed for this net')
        key = (net, gi, scope)
        if key in self._graphs:
            raise ValueError('region partition already observed')
        self._checkpoint(net)
        with self._engine._half_width_scope(overrides):
            half = self._engine._hw(net, gi)
            graph = self._engine._planar_event_geometry_for_layer(
                net, gi, self._engine._manufacturing_grid_nm, lambda: self._checkpoint(net))
        self._check()
        if graph is None:
            raise RegionObservationLimit('UNKNOWN_REGION_UNAVAILABLE')
        geometry = graph[0]
        publication = self._publish('planar', net, scope, {
            'gi': gi, 'grid_nm': self._engine._manufacturing_grid_nm, 'half_width_nm': half,
            'rectangles_grid_half_open': geometry.rectangles,
            'portals': [asdict(portal) for portal in geometry.portals]})
        self._graphs[key] = graph
        return publication

    def via(self, net, lo, scope='global'):
        self._check()
        overrides = self._scope(net, scope)
        key = (net, lo, scope)
        if type(lo) is not int or lo not in self._layers[net] or lo + 1 not in self._layers[net]:
            raise ValueError('adjacent via layers are not both allowed')
        if key in self._vias:
            raise ValueError('via partition already observed')
        lower, upper = self._graphs.get(key), self._graphs.get((net, lo + 1, scope))
        if lower is None or upper is None:
            raise ValueError('via observation requires both completed planar partitions')
        self._checkpoint(net)
        with self._engine._half_width_scope(overrides):
            rectangles = self._engine._planar_via_event_regions(
                net, lo, lower, upper, self._engine._manufacturing_grid_nm,
                lambda: self._checkpoint(net))
        self._check()
        publication = self._publish('via_domain', net, scope, {
            'lo_gi': lo, 'grid_nm': self._engine._manufacturing_grid_nm,
            'rectangles_grid_half_open': rectangles,
            'via_redundancy': self._engine.via_redundancy,
            'pad_half_nm': self._engine._via_pad_half[lo],
            'cut_size_nm': self._engine._via_cut[lo], 'cut_pitch_nm': self._engine._via_pitch[lo],
            'actual_via_legality': 'UNKNOWN', 'planar_region_associations': 'UNKNOWN'})
        self._vias.add(key)
        return publication

    def endpoint(self, net, index, scope='global', *, bounded_native=False):
        if type(bounded_native) is not bool:
            raise TypeError('bounded native endpoint flag must be boolean')
        if bounded_native:
            check = self._native_endpoint_check_original
            if check is None or self._native_endpoint_check is not check:
                raise ValueError('bounded endpoint authority is unavailable or changed')
            check()
            if self._input_check is not self._input_check_original:
                raise ValueError('bounded endpoint input check changed')
            # Avoid an instance-level replacement of these checks.
            if any(name in vars(self) for name in
                   ('_timed', '_check', '_scope', '_checkpoint', '_bounded_endpoint', '_publish')):
                raise ValueError('bounded endpoint method changed')
        self._check()
        overrides = self._scope(net, scope)
        if type(index) is not int or not 0 <= index < len(self._nodes[net]):
            raise ValueError('endpoint is not in the initialized world')
        key = (net, index, scope)
        if key in self._endpoints:
            raise ValueError('endpoint domain already observed')
        self._checkpoint(net)
        node, pin = self._nodes[net][index], self._node_pins[net][index]
        if bounded_native:
            half, domain = self._bounded_endpoint(net, node, overrides)
        else:
            with self._engine._half_width_scope(overrides):
                half = self._engine._hw(net, node[2])
                domain = self._engine._planar_endpoint_domain(net, node)
        new_cache = _owned_cache_value(self._engine)
        old_id, old_rows = self._owned_cache_original
        new_id, new_rows = new_cache
        old_by_key = {row[0]: row for row in old_rows}
        new_by_key = {row[0]: row for row in new_rows}
        if (old_id != new_id or any(new_by_key.get(key) != row for key, row in old_by_key.items())
                or set(new_by_key) - set(old_by_key) - {(str(net), int(node[2]))}):
            raise ValueError('initialized region endpoint modified another owned terminal index')
        for key in set(new_by_key) - set(old_by_key):
            _validate_owned_entry(self._engine, key, self._engine._owned_terminal_group_indexes[key])
        self._owned_cache_original = new_cache
        self._check()
        publication = self._publish('endpoint_domain', net, scope, {
            'endpoint_index': index, 'pin_fact': pin_fact(pin), 'node': node,
            'grid_nm': self._engine._manufacturing_grid_nm, 'half_width_nm': half,
            'domain': _json_value(domain), 'actual_access_legality': 'UNKNOWN'})
        self._endpoints.add(key)
        return publication

    def finish(self, status='UNKNOWN_INCOMPLETE'):
        if status not in ('UNKNOWN_INCOMPLETE', 'UNKNOWN_DIAGNOSTIC_DEADLINE',
                          'UNKNOWN_REGION_EVENT_BUDGET', 'UNKNOWN_REGION_UNAVAILABLE',
                          'UNKNOWN_NATIVE_WARM_CACHE'):
            raise ValueError('region observation cannot grant completion')
        self._check()
        result = {'schema': 'clay.initialized-region-observation/v1', 'status': status,
                'physical_admission': False, 'candidate_completeness': 'UNKNOWN',
                'routing_capacity': 'UNKNOWN', 'facts_publication': dict(self.facts_publication),
                'parts': list(self._published), 'checkpoint_calls': self._count,
                'event_budget': self._budget,
                'elapsed_seconds': time.monotonic() - self._started,
                'input_check_calls': self._input_check_calls,
                'input_check_seconds': self._input_check_seconds,
                'consumer_check_seconds': self._consumer_check_seconds,
                'budget_basis': 'checkpoint calls; not total operations, native work or bytes'}
        if self._native_endpoint_check_original is not None:
            result['native_endpoint_work_events'] = dict(self._native_endpoint_counts)
            result['budget_basis'] = ('region checkpoints and bounded native endpoint events; '
                                      'not total operations, native work, time or bytes')
        return result

    def close(self):
        self._active = False
