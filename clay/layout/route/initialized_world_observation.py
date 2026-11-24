"""Bounded terminal observation of the actual initialized pre-group world.

The detached facts describe the engine's present state, not an exhaustive
candidate graph, a future Book, physical legality or routing admission.
"""
import json
import time
from collections import Counter

from clay.layout.route.canonical_input_capture import route_engine_configuration
from clay.layout.route.terminal_identity import pin_fact


class _ObservationLimit(Exception):
    pass


def observe_initialized_world(*, engine, pins_by_net, nodes, node_pins, layers,
                              book, ordered_nets, world, contract, constraints, topology,
                              fact_budget, deadline):
    """Copy the exact pre-group facts within caller and engine wall limits."""
    schema = 'clay.initialized-pre-group-world/v1'
    used = 0

    def checkpoint(count=0):
        nonlocal used
        # Preserve the source route watchdog and the enclosing session's stop.
        engine._watchdog_check('', 'initialized_world_observation')
        if time.monotonic() >= deadline:
            raise _ObservationLimit('UNKNOWN_DIAGNOSTIC_DEADLINE')
        used += count
        if used > fact_budget:
            raise _ObservationLimit('UNKNOWN_FACT_BUDGET')

    def rows(values):
        result = []
        for value in values:
            checkpoint(1)
            result.append(value)
        return result

    def account_tree(value):
        checkpoint(1)
        if isinstance(value, dict):
            for item in value.values():
                account_tree(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                account_tree(item)

    try:
        checkpoint()
        if topology is None or topology.sha256() != world.topology_provenance_digest:
            raise ValueError('initialized world requires the same topology plan')
        if book is not engine._book or any(book.metal.values()) or any(book.via_pads.values()) or book.cuts or book.persistent:
            raise ValueError('initialized world requires the actual empty pre-group Book')
        if ordered_nets is None or set(ordered_nets) != set(pins_by_net) or len(ordered_nets) != len(pins_by_net):
            raise ValueError('initialized net order differs from input')
        if set(nodes) != set(pins_by_net) or set(node_pins) != set(pins_by_net) or set(layers) != set(pins_by_net):
            raise ValueError('initialized per-net domains differ from input')
        from clay.flow.execution_budget import ExecutionStopped
        from clay.layout.route.constraints import LayerSet
        from clay.layout.route.engine_gridless import NegotiationWatchdogStop
        strict_nets = {c.net for c in constraints if isinstance(c, LayerSet)}
        layer_rows = []
        for gi in range(engine._n_layers):
            checkpoint(1)
            layer_n = engine._tech._gi_to_pdk[gi]
            layer_rows.append({'gi': gi, 'pdk_layer_n': layer_n,
                               'gds_key': engine._tech.pdk.metal_key(layer_n),
                               'min_half_width_nm': engine._half_w[gi],
                               'min_space_nm': engine._space[gi]})
        net_rows = []
        for net in sorted(pins_by_net):
            checkpoint(1)
            actual_pins = tuple(pins_by_net[net])
            resolved_pins = tuple(node_pins[net])
            actual_nodes = tuple(nodes[net])
            if len(resolved_pins) != len(actual_nodes):
                raise ValueError('initialized node/pin inventory differs')
            remaining = Counter(map(id, actual_pins))
            endpoints = []
            mapping_complete = True
            for pin, node in zip(resolved_pins, actual_nodes):
                checkpoint(1)
                if remaining[id(pin)] <= 0:
                    raise ValueError('initialized node uses a foreign pin')
                remaining[id(pin)] -= 1
                fact = pin_fact(pin)
                try:
                    seed = engine._tech.pin_seed(pin.layer, list(layers[net]),
                                                 strict_layers=net in strict_nets)
                    seed_error = None
                except (ExecutionStopped, NegotiationWatchdogStop):
                    raise
                except Exception as error:
                    seed = None
                    seed_error = type(error).__name__
                if seed is not None and not (
                        type(seed) is tuple and len(seed) == 2 and type(seed[0]) is int and
                        type(seed[1]) is list and all(type(value) is int for value in seed[1])):
                    seed = None
                    seed_error = 'InvalidSeed'
                valid_node = (type(node) is tuple and len(node) == 3 and
                              all(type(value) is int for value in node) and
                              tuple(node[:2]) == tuple(fact[3:5]) and
                              0 <= node[2] < engine._n_layers and node[2] in layers[net] and
                              seed is not None and node[2] == seed[0])
                mapping_complete = mapping_complete and valid_node
                endpoints.append({'pin_fact': fact, 'node': node,
                                  'mapping_status': 'OBSERVED' if valid_node else 'UNKNOWN_SEED_NODE',
                                  'seed_ladder': None if seed is None else seed[1],
                                  'seed_error_type': seed_error})
            unresolved = []
            for pin in actual_pins:
                if remaining[id(pin)] > 0:
                    checkpoint(1)
                    remaining[id(pin)] -= 1
                    unresolved.append(pin_fact(pin))
            mapping_complete = mapping_complete and not unresolved
            net_rows.append({'net': net, 'allowed_gis': tuple(layers[net]),
                             'endpoints': endpoints, 'unresolved_pin_facts': unresolved,
                             'strict_layers': net in strict_nets,
                             'endpoint_mapping': 'OBSERVED' if mapping_complete else 'UNKNOWN',
                             'active_half_width_nm': tuple(engine._hw(net, gi) for gi in range(engine._n_layers)),
                             'rails': rows(engine._rail_by_net.get(net, ()))})
        obstacle_rows = rows((gi, owner, rect, engine._obs_provenance.get((gi, owner, rect)))
                             for gi, entries in sorted(engine._obs_by_gi.items())
                             for owner, rect in entries)
        cut_rows = rows((lo, rect, engine._obs_cut_provenance.get((lo, rect)))
                        for lo, entries in sorted(engine._obs_cuts.items()) for rect in entries)
        mim = {}
        for name in ('_mim_carve_bottom_blockers_by_gi', '_mim_forbidden_cut_touch_by_lo',
                     '_mim_bottom_routing_clearance_by_gi', '_mim_top_via_marker_clearance_by_lo'):
            mapping = getattr(engine, name)
            mim[name] = rows((key, value) for key, values in sorted(mapping.items()) for value in values)
        overrides = rows((net, gi, width) for (net, gi), width in sorted(
            getattr(engine, '_active_half_width_overrides', {}).items()))
        fixed_deck_literals = rows(engine._fixed_deck_literal_rects)
        abstract_rects = rows((gi, rect) for gi, rects in sorted(engine._abstract_rects.items())
                              for rect in sorted(rects))
        required_endpoints = rows(sorted(engine._required_wire_endpoints))
        topology_data = json.loads(topology.canonical_json())
        contract_data = contract.canonical_data()
        via_geometry = {'cut_nm': engine._via_cut, 'pad_half_nm': engine._via_pad_half,
                        'single_pad_half_nm': engine._via_pad_half1,
                        'space_nm': engine._via_space, 'pitch_nm': engine._via_pitch}
        account_tree(topology_data)
        account_tree(contract_data)
        account_tree(via_geometry)
        checkpoint()
        result = {'schema': schema, 'status': 'OBSERVED', 'fact_count': used,
                  'fact_budget': fact_budget, 'physical_admission': False,
                  'fact_budget_basis': 'emitted outer fact rows and topology, contract and via tree values; not a byte or total work limit',
                  'candidate_completeness': 'UNKNOWN', 'routing_capacity': 'UNKNOWN',
                  'endpoint_mapping': 'OBSERVED' if all(
                      row['endpoint_mapping'] == 'OBSERVED' for row in net_rows) else 'UNKNOWN',
                  'routing_world_sha256': world.sha256(), 'pdk_sha256': world.pdk_provenance_digest,
                  'topology_sha256': topology.sha256(),
                  'topology_plan': topology_data,
                  'group_contract_sha256': contract.sha256,
                  'group_contract': contract_data,
                  'selected_engine': route_engine_configuration(engine),
                  'route_watchdog_deadline_monotonic': engine._watchdog_deadline,
                  'bounds_nm': engine._bounds, 'layers': layer_rows,
                  'ordered_nets': tuple(ordered_nets), 'nets': net_rows,
                  'planar_obstacles': obstacle_rows, 'cut_obstacles': cut_rows,
                  'fixed_deck_literals': fixed_deck_literals, 'abstract_rects': abstract_rects,
                  'required_wire_endpoints': required_endpoints,
                  'via_geometry': via_geometry,
                  'mim_constraints': mim, 'active_half_width_overrides': overrides,
                  'book_status': 'OBSERVED_EMPTY_PRE_GROUP'}
        # Check the deadline after serialization too, before publishing any facts.
        detached = json.loads(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False))
        checkpoint()
        return detached
    except _ObservationLimit as error:
        return {'schema': schema, 'status': str(error), 'fact_budget': fact_budget,
                'physical_admission': False}
