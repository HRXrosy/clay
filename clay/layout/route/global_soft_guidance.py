"""Bounded experimental soft ranking of *fresh* native group candidates.

The caller must authenticate the terminal observation and its receipt before
supplying its facts and parts.  This module verifies their byte bindings and
the live source/world/engine/Book scope; it does not itself confer that receipt
authority, consume a candidate pool, change routing, or establish legality.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import time

from clay.layout.route.canonical_input_capture import (
    _json_value, route_engine_configuration, snapshot_prepared_routing_input,
)
from clay.layout.route.group_fallback_checkpoint import _canonical
from clay.layout.route.group_ports import GroupRoutePhysicalPort, GroupRouteRoutingContract
from clay.layout.route.resource_world import RoutingWorldProvenance
from clay.layout.route.terminal_identity import pin_fact


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def _hash(value):
    return sha256(_encoded(value)).hexdigest()


def _file_hash(value):
    # initialized_region_observation.publish_pending includes one newline.
    return sha256(_encoded(value) + b'\n').hexdigest()


def book_digest(book):
    value = _canonical({name: getattr(book, name) for name in
                        ('metal', 'via_pads', 'cuts', 'persistent')}, set())
    return _hash(value)


class _Unknown(Exception):
    pass


class _ExternalCallError(BaseException):
    def __init__(self, original):
        self.original = original


def _external(function, *args):
    try:
        return function(*args)
    except BaseException as error:
        raise _ExternalCallError(error) from error


@dataclass
class WorkBudget:
    """Explicit observer work/deadline; never a replacement engine budget."""
    limit: int
    deadline: float
    checkpoint: object
    used: int = 0

    def __post_init__(self):
        if type(self.limit) is not int or self.limit <= 0 or not callable(self.checkpoint):
            raise ValueError('finite positive work limit and checkpoint required')
        if type(self.deadline) not in (int, float) or not (0 < self.deadline < float('inf')):
            raise ValueError('finite monotonic deadline required')

    def charge(self):
        _external(self.checkpoint)  # Preserve original cancellation/watchdog.
        self.used += 1
        if self.used > self.limit or time.monotonic() >= self.deadline:
            raise _Unknown('GUIDANCE_BUDGET_EXHAUSTED')


def _check_live_group_width_scope(engine, nets, group_nets, budget):
    """Verify exactly the rail proof/scope entered by the native group allocator."""
    budget.charge()
    if (type(getattr(engine, '_active_half_width_overrides', None)) is not dict or
            type(getattr(engine, '_rail_by_net', None)) is not dict):
        raise _Unknown('GROUP_WIDTH_SCOPE_MISMATCH')
    expected = {}
    for net in sorted(nets):
        budget.charge()
        row = nets[net]
        if _encoded(_json_value(engine._rail_by_net.get(net, ()))) != _encoded(row['rails']):
            raise _Unknown('RAIL_SCOPE_MISMATCH')
        baseline = engine._net_half_w.get(net, engine._half_w)
        if len(baseline) != engine._n_layers or list(baseline) != row['active_half_width_nm']:
            raise _Unknown('BACKGROUND_WIDTH_MISMATCH')
        if net in group_nets:
            expected.update(engine._rail_local_half_width_overrides(
                net, row['allowed_gis'], tuple(engine._rail_by_net.get(net, ()))))
    if engine._active_half_width_overrides != expected:
        raise _Unknown('GROUP_WIDTH_SCOPE_MISMATCH')


def choose_fresh_candidate(*, source_cell, tech, world, contract, engine, book, pins_by_net, prepared_inputs,
                           facts, part_publications, port, producer, budget,
                           verify_receipt):
    """Return one current producer row or UNKNOWN; never a route certificate.

    ``verify_receipt`` is a trusted in-process caller boundary, called with
    (facts-file-sha, tuple(part-file-shas), invocation-id). It MUST verify the
    same terminal public receipt and publication bindings. A caller that merely
    returns True is not an authenticated production integration. Region files
    describe the initial empty Book only; after Book changes, their prices are
    explicitly static background, and current rows are freshly generated.
    """
    try:
        if (type(world) is not RoutingWorldProvenance or
                type(contract) is not GroupRouteRoutingContract or
                type(port) is not GroupRoutePhysicalPort or port not in contract.ports or
                not callable(producer) or not callable(verify_receipt) or type(budget) is not WorkBudget or
                getattr(engine, '_book', None) is not book):
            raise _Unknown('INVALID_LIVE_SCOPE')
        budget.charge()
        if (type(prepared_inputs) is not dict or
                prepared_inputs.get('source_cell') is not source_cell or
                prepared_inputs.get('tech') is not tech or
                prepared_inputs.get('routing_world') is not world or
                prepared_inputs.get('route_engine') is not engine or
                prepared_inputs.get('pins_by_net') is not pins_by_net or
                type(prepared_inputs.get('route_kwargs')) is not dict or
                prepared_inputs['route_kwargs'].get('group_route_contract') is not contract):
            raise _Unknown('PREPARED_INPUT_IDENTITY_MISMATCH')
        bound_inputs = dict(prepared_inputs)
        bound_kwargs = dict(prepared_inputs['route_kwargs'])
        bound_pins = {net: (values, tuple(values)) for net, values in pins_by_net.items()}
        bound_items = {key: tuple(prepared_inputs[key]) for key in ('obstacles', 'constraints')}
        def verify_prepared():
            budget.charge()
            if (set(prepared_inputs) != set(bound_inputs) or any(
                    prepared_inputs[key] is not value for key, value in bound_inputs.items()) or
                    set(prepared_inputs['route_kwargs']) != set(bound_kwargs) or any(
                    prepared_inputs['route_kwargs'][key] is not value for key, value in bound_kwargs.items()) or
                    set(pins_by_net) != set(bound_pins)):
                raise _Unknown('PREPARED_INPUT_IDENTITY_MISMATCH')
            for net, (original, items) in bound_pins.items():
                current = pins_by_net[net]
                if (current is not original or len(current) != len(items) or
                        any(a is not b for a, b in zip(current, items))):
                    raise _Unknown('PREPARED_INPUT_IDENTITY_MISMATCH')
            for key, items in bound_items.items():
                current = prepared_inputs[key]
                if len(current) != len(items) or any(a is not b for a, b in zip(current, items)):
                    raise _Unknown('PREPARED_INPUT_IDENTITY_MISMATCH')
            if ('route_kwargs' not in facts or
                    _encoded(_json_value(prepared_inputs['route_kwargs'])) != _encoded(facts['route_kwargs']) or
                    _encoded(snapshot_prepared_routing_input(**prepared_inputs)) != _encoded(facts['prepared_input'])):
                raise _Unknown('PREPARED_INPUT_MISMATCH')
        verify_prepared()
        from clay.frontend.source_revision import reopen_complete_source
        from clay.layout.policy import policy_for_cell
        from clay.layout.route.engine_registry import route_engine_from_policy
        from clay.layout.route.resource_world import pdk_provenance_digest
        from clay.parser import parse_clay_file
        from clay.layout.route.engine_gridless import GridlessEngine, _group_route_pin_key
        if type(engine) is not GridlessEngine:
            raise _Unknown('INVALID_ENGINE')
        source = reopen_complete_source(source_cell)
        parsed = parse_clay_file(source['path'])
        cells = [cell for cell in parsed.cell_defs if cell.local_name == source_cell.local_name]
        if len(cells) != 1:
            raise _Unknown('SOURCE_CELL_AMBIGUOUS')
        selected = route_engine_from_policy(policy_for_cell(parsed, cells[0]))
        if (selected is None or route_engine_configuration(selected) != route_engine_configuration(engine) or
                source['pdk_id'] != tech.pdk_id or pdk_provenance_digest(tech) != world.pdk_provenance_digest):
            raise _Unknown('SOURCE_POLICY_OR_PDK_MISMATCH')
        source_sha = source['source_sha256']
        observed = facts['initialized_world']
        facts_sha = _file_hash(facts)
        if (facts['schema'] != 'clay.initialized-region-facts/v1' or
                facts['status'] != 'PENDING_RECEIPT' or facts['physical_admission'] is not False or
                facts['source_sha256'] != source_sha or not facts['invocation_id'] or
                observed['status'] != 'OBSERVED' or observed['endpoint_mapping'] != 'OBSERVED' or
                observed['routing_world_sha256'] != world.sha256() or
                observed['pdk_sha256'] != world.pdk_provenance_digest or
                observed['topology_sha256'] != world.topology_provenance_digest or
                observed['group_contract_sha256'] != contract.sha256 or
                observed['selected_engine'] != route_engine_configuration(engine) or
                observed['book_status'] != 'OBSERVED_EMPTY_PRE_GROUP' or
                facts['book_sha256'] != _hash(facts['book_canonical'])):
            raise _Unknown('FACT_SCOPE_MISMATCH')
        # The initialization producer verifies this empty Book. Recompute it
        # here, rather than trusting the text label.
        recorded = facts['book_canonical']
        from clay.layout.route.engine_gridless import _Book
        empty = _Book(engine._n_layers)
        if recorded != _canonical({name: getattr(empty, name) for name in
                                   ('metal', 'via_pads', 'cuts', 'persistent')}, set()):
            raise _Unknown('BACKGROUND_BOOK_NOT_EMPTY')
        actual_pins = Counter(_encoded(pin_fact(pin)) for pins in pins_by_net.values() for pin in pins)
        world_pins = Counter(_encoded(pin) for pin in world.pins)
        if actual_pins != world_pins:
            raise _Unknown('PIN_INVENTORY_MISMATCH')
        group_keys = Counter(_group_route_pin_key(port_row.net, access)
                             for port_row in contract.ports for access in port_row.accesses)
        actual_keys = Counter(_group_route_pin_key(net, pin)
                              for net, pins in pins_by_net.items() for pin in pins)
        if any(count != 1 or actual_keys[key] != 1 for key, count in group_keys.items()):
            raise _Unknown('GROUP_ACCESS_INVENTORY_MISMATCH')
        nets = {row['net']: row for row in observed['nets']}
        if len(nets) != len(observed['nets']) or set(nets) != set(pins_by_net):
            raise _Unknown('NET_INVENTORY_MISMATCH')
        for net, pins in pins_by_net.items():
            budget.charge()
            row = nets[net]
            allowed = row['allowed_gis']
            if (type(allowed) is not list or not allowed or
                    any(type(gi) is not int or gi < 0 or gi >= engine._n_layers for gi in allowed) or
                    len(set(allowed)) != len(allowed) or
                    row['endpoint_mapping'] != 'OBSERVED' or row['unresolved_pin_facts'] or
                    Counter(_encoded(e['pin_fact']) for e in row['endpoints']) !=
                    Counter(_encoded(pin_fact(pin)) for pin in pins)):
                raise _Unknown('ENDPOINT_INVENTORY_MISMATCH')
        group_nets = {item.net for item in contract.ports}
        if observed['active_half_width_overrides']:
            raise _Unknown('BACKGROUND_WIDTH_SCOPE_UNKNOWN')
        def verify_width_scope():
            _check_live_group_width_scope(engine, nets, group_nets, budget)
        verify_width_scope()
        expected = {(net, gi, scope) for net, row in nets.items()
                    for gi in row['allowed_gis']
                    for scope in (('global', 'group') if net in group_nets else ('global',))}
        partitions = {}
        endpoint_parts, via_parts = set(), set()
        part_shas = []
        # Each supplied part is its original decoded payload paired with the
        # file SHA in the same terminal publication manifest.
        for part, declared_sha in part_publications:
            budget.charge()
            actual_sha = _file_hash(part)
            if actual_sha != declared_sha:
                raise _Unknown('PART_DIGEST_MISMATCH')
            part_shas.append(actual_sha)
            if (part['schema'] != 'clay.initialized-region-part/v1' or
                    part['status'] != 'PENDING_RECEIPT' or part['physical_admission'] is not False or
                    part['facts_sha256'] != facts_sha or part['book_sha256'] != facts['book_sha256'] or
                    part['book_status'] != 'OBSERVED_EMPTY_PRE_GROUP'):
                raise _Unknown('PART_SCOPE_MISMATCH')
            net, scope = part['net'], part['width_scope']
            if net not in nets or scope not in (('global', 'group') if net in group_nets else ('global',)):
                raise _Unknown('PARTITION_IDENTITY_AMBIGUOUS')
            if part['kind'] == 'endpoint_domain':
                index = part['endpoint_index']
                if (type(index) is not int or index < 0 or index >= len(nets[net]['endpoints']) or
                        (net, scope, index) in endpoint_parts or
                        _encoded(part['pin_fact']) != _encoded(nets[net]['endpoints'][index]['pin_fact']) or
                        _encoded(part['node']) != _encoded(nets[net]['endpoints'][index]['node'])):
                    raise _Unknown('ENDPOINT_PARTITION_MISMATCH')
                endpoint_parts.add((net, scope, index))
                continue
            if part['kind'] == 'via_domain':
                lo = part['lo_gi']
                allowed = nets[net]['allowed_gis']
                if (type(lo) is not int or lo not in allowed or lo + 1 not in allowed or
                        (net, scope, lo) in via_parts or part['via_redundancy'] != engine.via_redundancy):
                    raise _Unknown('VIA_PARTITION_MISMATCH')
                via_parts.add((net, scope, lo))
                continue
            if part['kind'] != 'planar':
                raise _Unknown('UNSUPPORTED_PARTITION')
                continue
            key = (part['net'], part['gi'], part['width_scope'])
            if key not in expected or key in partitions:
                raise _Unknown('PARTITION_IDENTITY_AMBIGUOUS')
            grid = part['grid_nm']
            if type(grid) is not int or grid <= 0:
                raise _Unknown('INVALID_GRID')
            expected_half = (nets[net]['active_half_width_nm'][part['gi']] if scope == 'global'
                             else engine._hw(net, part['gi']))
            if type(part['half_width_nm']) is not int or part['half_width_nm'] != expected_half:
                raise _Unknown('REGION_WIDTH_SCOPE_MISMATCH')
            rects = []
            for rect in part['rectangles_grid_half_open']:
                budget.charge()
                if (type(rect) not in (tuple, list) or len(rect) != 4 or
                        any(type(v) is not int for v in rect) or
                        rect[0] >= rect[2] or rect[1] >= rect[3]):
                    raise _Unknown('INVALID_REGION')
                rects.append(tuple(rect))
            partitions[key] = (grid, tuple(rects))
        if set(partitions) != expected:
            raise _Unknown('INCOMPLETE_PLANAR_PARTITIONS')
        scopes = {(net, scope) for net in nets
                  for scope in (('global', 'group') if net in group_nets else ('global',))}
        if endpoint_parts != {(net, scope, index) for net, scope in scopes
                              for index in range(len(nets[net]['endpoints']))}:
            raise _Unknown('INCOMPLETE_ENDPOINT_PARTITIONS')
        if via_parts != {(net, scope, lo) for net, scope in scopes
                         for lo in nets[net]['allowed_gis'] if lo + 1 in nets[net]['allowed_gis']}:
            raise _Unknown('INCOMPLETE_VIA_PARTITIONS')
        before_receipt_book = book_digest(book)
        if _external(verify_receipt, facts_sha, tuple(part_shas), facts['invocation_id']) is not True:
            raise _Unknown('TERMINAL_RECEIPT_NOT_AUTHENTICATED')
        # Receipt verification can consume the remaining time or trigger a stop.
        # Do not invoke the native producer until the original checkpoint passes.
        budget.charge()
        if (getattr(engine, '_book', None) is not book or book_digest(book) != before_receipt_book or
                route_engine_configuration(engine) != observed['selected_engine'] or
                world.sha256() != observed['routing_world_sha256'] or _file_hash(facts) != facts_sha):
            raise _Unknown('LIVE_SCOPE_CHANGED')
        verify_prepared()
        verify_width_scope()
        if (reopen_complete_source(source_cell)['source_sha256'] != source_sha or
                pdk_provenance_digest(tech) != world.pdk_provenance_digest):
            raise _Unknown('SOURCE_OR_PDK_CHANGED')
        current_before = book_digest(book)
        config_before = route_engine_configuration(engine)
        world_before = world.sha256()
        from clay.layout.route.engine_gridless import GroupRouteLandingUnavailable
        # Reopening the source and PDK or hashing the world can themselves
        # consume the remaining diagnostic time; check immediately before the
        # potentially expensive native candidate producer.
        budget.charge()
        try:
            rows = producer()  # Actual caller's current-Book native producer.
        except GroupRouteLandingUnavailable as error:
            raise _Unknown('NO_CURRENT_CANDIDATE') from error
        except BaseException as error:
            raise _ExternalCallError(error) from error
        if (getattr(engine, '_book', None) is not book or book_digest(book) != current_before or
                route_engine_configuration(engine) != config_before or world.sha256() != world_before):
            raise _Unknown('LIVE_SCOPE_CHANGED')
        verify_prepared()
        verify_width_scope()
        if _file_hash(facts) != facts_sha:
            raise _Unknown('FACT_SCOPE_MISMATCH')
        if (reopen_complete_source(source_cell)['source_sha256'] != source_sha or
                pdk_provenance_digest(tech) != world.pdk_provenance_digest):
            raise _Unknown('SOURCE_OR_PDK_CHANGED')
        if type(rows) not in (tuple, list) or not rows:
            raise _Unknown('NO_CURRENT_CANDIDATE')
        active_nets = set(world.required_nets) | group_nets
        scored = []
        candidate_ids = set()
        for ordinal, row in enumerate(rows):
            budget.charge()
            if (type(row) is not tuple or len(row) != 11 or row[4] not in port.accesses or
                    type(row[10]) is not tuple or len(row[10]) != 3 or
                    any(type(v) is not int for v in row[10])):
                raise _Unknown('INVALID_CURRENT_CANDIDATE')
            if (type(row[6]) is not tuple or type(row[7]) is not tuple or
                    len(row[6]) != 3 or len(row[7]) != 3 or
                    any(type(v) is not int for v in (*row[6], *row[7])) or
                    row[6][:2] != row[4].xy_nm or row[7][2] != row[10][2] or
                    row[9] not in ('direct', 'fixed_device_stack', 'emitted_via')):
                raise _Unknown('INVALID_CURRENT_CANDIDATE')
            if row[9] == 'direct' and (row[8] is not None or row[6] != row[7]):
                raise _Unknown('INVALID_CURRENT_CANDIDATE')
            if row[9] == 'fixed_device_stack' and row[8] is not None:
                raise _Unknown('INVALID_CURRENT_CANDIDATE')
            if row[9] == 'emitted_via':
                from clay.layout.route.engine_gridless import _ViaSite
                allowed_models = (
                    (0, 1, engine.via_redundancy)
                    if bool(getattr(engine, "_single_via_escape_allowed", lambda: False)())
                    else (0, engine.via_redundancy)
                )
                if type(row[8]) is not _ViaSite or row[8].n not in allowed_models:
                    raise _Unknown('INVALID_CURRENT_CANDIDATE')
            identity = _encoded(_json_value((row[4], row[6], row[7], row[8], row[9], row[10])))
            if identity in candidate_ids:
                raise _Unknown('AMBIGUOUS_CURRENT_CANDIDATE')
            candidate_ids.add(identity)
            x, y, gi = row[10]
            box = port.envelope_nm
            if not (box[0] <= x <= box[2] and box[1] <= y <= box[3] and
                    (x in (box[0], box[2]) or y in (box[1], box[3]))):
                raise _Unknown('INVALID_CURRENT_PORTAL')
            own_key = (port.net, gi, 'group')
            if own_key not in partitions:
                raise _Unknown('CURRENT_LAYER_UNOBSERVED')
            occupants = set()
            for net in sorted(active_nets):
                scope = 'group' if net in group_nets else 'global'
                partition = partitions.get((net, gi, scope))
                if partition is None:
                    continue  # This net's source-declared allowed layers exclude gi.
                grid, rects = partition
                if x % grid or y % grid:
                    raise _Unknown('PORTAL_OFF_OBSERVED_GRID')
                gx, gy = x // grid, y // grid
                for x0, y0, x1, y1 in rects:
                    budget.charge()
                    if x0 <= gx < x1 and y0 <= gy < y1:
                        occupants.add(net)
                        break
            if port.net not in occupants:
                raise _Unknown('PORTAL_OUTSIDE_OBSERVED_DOMAIN')
            # A conservative overlap is a soft ordinal, not a capacity or a
            # current Book legality verdict; original native order breaks ties.
            scored.append((len(occupants - {port.net}), ordinal, row))
        chosen = min(scored, key=lambda item: (item[0], item[2]))
        if (getattr(engine, '_book', None) is not book or book_digest(book) != current_before or
                route_engine_configuration(engine) != config_before or world.sha256() != world_before):
            raise _Unknown('LIVE_SCOPE_CHANGED')
        verify_prepared()
        verify_width_scope()
        if _file_hash(facts) != facts_sha:
            raise _Unknown('FACT_SCOPE_MISMATCH')
        if (reopen_complete_source(source_cell)['source_sha256'] != source_sha or
                pdk_provenance_digest(tech) != world.pdk_provenance_digest):
            raise _Unknown('SOURCE_OR_PDK_CHANGED')
        return {'status': 'ADVISORY', 'physical_admission': False,
                'candidate': chosen[2], 'candidate_ordinal': chosen[1],
                'soft_overlap_nets': chosen[0], 'facts_sha256': facts_sha,
                'background_book_sha256': facts['book_sha256'], 'current_book_sha256': current_before,
                'book_scope': 'INITIAL' if current_before == facts['book_sha256'] else 'STATIC_BACKGROUND',
                'candidate_legality': 'UNKNOWN', 'routing_capacity': 'UNKNOWN', 'work_used': budget.used}
    except _ExternalCallError as error:
        raise error.original
    except (KeyError, TypeError, ValueError, _Unknown) as error:
        return {'status': 'UNKNOWN', 'physical_admission': False,
                'reason': str(error) if isinstance(error, _Unknown) else 'MALFORMED_EVIDENCE',
                'work_used': getattr(budget, 'used', None)}
