"""Snapshot the sole canonical preparation seam before any route is solved."""
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path


class PreparedRoutingInputCaptured(BaseException):
    """An explicit preparation-only consumer stops before invoking a solver."""


def route_engine_configuration(engine):
    """Freeze the selected strategy's actual constructor settings, including defaults."""
    from clay.layout.route.engine_registry import route_engine_registry
    matches = [name for name, cls in route_engine_registry().items() if type(engine) is cls]
    if len(matches) != 1:
        raise ValueError('prepared replay needs a registered exact route-engine type')
    parameters = {}
    for name, parameter in inspect.signature(type(engine).__init__).parameters.items():
        if name == 'self':
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ValueError('cannot freeze a variadic route-engine constructor')
        if not hasattr(engine, name):
            raise ValueError('route-engine setting is not replayable: '+name)
        value = getattr(engine, name)
        if name == 'allow_single_via_escape' and value is False:
            continue
        parameters[name] = _json_value(value)
    return dict(name=matches[0], parameters=parameters)


def experimental_routing_configuration(engine):
    """Describe explicitly enabled ranking separately from source policy."""
    enabled = getattr(engine, '_experimental_group_demand_guidance', False)
    if type(enabled) is not bool:
        raise TypeError('experimental group demand guidance must be a bool')
    feedback = getattr(engine, '_experimental_group_full_feedback', False)
    if type(feedback) is not bool:
        raise TypeError('experimental group full feedback must be a bool')
    seal_budget = getattr(engine, '_experimental_fixed_index_seal_budget', None)
    if seal_budget is not None:
        from clay.layout.route.fixed_index_seal import configured_budget
        seal_budget = configured_budget(engine)
        if enabled or feedback:
            raise ValueError('fixed index seal and group routing experiments cannot be combined')
        return {'fixed_index_seal': {'version': 'cooperative-fixed-index/v1',
                                     'work_budget': seal_budget}}
    if not enabled and not feedback:
        return None
    result = {}
    if enabled:
        from clay.layout.route.group_demand_guidance import GUIDE_VERSION
        result['group_demand_guidance'] = GUIDE_VERSION
    if feedback:
        result['group_full_feedback'] = 'same-invocation-group-prefix/v1'
    return result


def validate_experimental_prepared(snapshot):
    """Validate v4's recorded plan and its binding to the captured world.

    This is an integrity check of the captured values, not authentication of
    their origin or physical admission.
    """
    if snapshot.get('schema') == 'clay.canonical-prepared-routing-input/v6':
        experimental = snapshot.get('experimental_routing')
        if (type(experimental) is not dict
                or set(experimental) not in ({'group_full_feedback'},
                                             {'group_full_feedback', 'group_demand_guidance'})
                or experimental['group_full_feedback'] != 'same-invocation-group-prefix/v1'
                or snapshot.get('selected_engine', {}).get('name') != 'gridless'):
            raise ValueError('invalid group full feedback prepared input')
        if 'group_demand_guidance' in experimental:
            prior = dict(snapshot, schema='clay.canonical-prepared-routing-input/v4',
                         experimental_routing={'group_demand_guidance': experimental['group_demand_guidance']})
            validate_experimental_prepared(prior)
        else:
            names = snapshot.get('route_argument_names')
            world = snapshot.get('routing_world')
            if (type(names) is not list or any(type(name) is not str for name in names)
                    or names != sorted(set(names)) or 'group_demand_topology_plan' in names
                    or snapshot.get('group_demand_topology') is not None or type(world) is not dict):
                raise ValueError('invalid group full feedback routing inputs')
            encoded = json.dumps(world, sort_keys=True, separators=(',', ':'),
                                 ensure_ascii=True, allow_nan=False).encode('utf8')
            if hashlib.sha256(encoded).hexdigest() != snapshot.get('routing_world_sha256'):
                raise ValueError('experimental routing world digest differs')
        return
    if snapshot.get('schema') == 'clay.canonical-prepared-routing-input/v5':
        experimental = snapshot.get('experimental_routing')
        seal = experimental.get('fixed_index_seal') if type(experimental) is dict else None
        names = snapshot.get('route_argument_names')
        world = snapshot.get('routing_world')
        selected = snapshot.get('selected_engine')
        if (type(experimental) is not dict or set(experimental) != {'fixed_index_seal'}
                or type(seal) is not dict or set(seal) != {'version', 'work_budget'}
                or seal['version'] != 'cooperative-fixed-index/v1'
                or type(seal['work_budget']) is not int or seal['work_budget'] <= 0
                or type(names) is not list or any(type(name) is not str for name in names)
                or names != sorted(set(names)) or 'group_demand_topology_plan' in names
                or type(selected) is not dict or selected.get('name') != 'gridless'
                or snapshot.get('group_demand_topology') is not None or type(world) is not dict):
            raise ValueError('invalid fixed index seal prepared input')
        encode = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                          ensure_ascii=True, allow_nan=False).encode('utf8')
        if hashlib.sha256(encode(world)).hexdigest() != snapshot.get('routing_world_sha256'):
            raise ValueError('experimental routing world digest differs from routing world')
        return
    if snapshot.get('schema') != 'clay.canonical-prepared-routing-input/v4':
        raise ValueError('not a bound experimental prepared input')
    experimental = snapshot.get('experimental_routing')
    names = snapshot.get('route_argument_names')
    if (type(experimental) is not dict or set(experimental) != {'group_demand_guidance'}
            or type(experimental['group_demand_guidance']) is not str
            or not experimental['group_demand_guidance']
            or type(names) is not list or any(type(name) is not str for name in names)
            or names != sorted(set(names))):
        raise ValueError('invalid experimental routing configuration')
    world = snapshot.get('routing_world')
    if type(world) is not dict:
        raise ValueError('experimental capture lacks routing world')
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=True, allow_nan=False).encode('utf8')
    if hashlib.sha256(encode(world)).hexdigest() != snapshot.get('routing_world_sha256'):
        raise ValueError('experimental routing world digest differs')
    binding = snapshot.get('group_demand_topology')
    has_group = 'group_route_contract' in names
    if not has_group:
        if binding is not None or 'group_demand_topology_plan' in names:
            raise ValueError('experimental topology without group-route contract')
        return
    if ('group_demand_topology_plan' not in names or type(binding) is not dict
            or set(binding) != {'plan', 'sha256'} or type(binding['plan']) is not dict):
        raise ValueError('experimental group-route topology binding missing')
    digest = hashlib.sha256(encode(binding['plan'])).hexdigest()
    if (binding['plan'].get('schema') != 'clay.topology-plan.v2'
            or binding['sha256'] != digest
            or world.get('topology_provenance_digest') != digest):
        raise ValueError('experimental topology plan digest differs from routing world')


def _json_value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise TypeError('prepared input mappings need exact string keys')
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError('prepared input cannot serialize '+type(value).__name__)


def _prepared_reference(source_cell, pdk_id):
    """Select the PDK's existing source reference; this is not an LVS verdict."""
    if pdk_id == 'sky130':
        # SKY130 native cells use the PDK's native X-device models and all of
        # their original parameter tokens, not the synthesized MOS LVS deck.
        from clay.frontend.native_spice_import import emit_native_spice
        return ('clay.canonical-prepared-routing-input/v2', 'native-spice-source/v1',
                emit_native_spice(source_cell, pdk_id=pdk_id, top_name='CANONICAL_SCENE'))
    from clay.layout.route.lvs_gate import route_cell_reference_text
    return ('clay.canonical-prepared-routing-input/v1', None,
            route_cell_reference_text(source_cell, tech_id=pdk_id,
                                      top_cell='CANONICAL_SCENE', full_extraction=True))


def snapshot_prepared_routing_input(*, source_cell, tech, geometry, placement_um,
        pins_by_net, obstacles, constraints, rails_by_net, routing_envelope,
        routing_world, final_placement_certificate, execution_scope,
        route_engine=None, route_kwargs=None, actual_shared_guard_edge_rail=None):
    from clay.layout.owned_artwork import _stamp_records
    from clay.layout.route.resource_world import canonical_pdk_routing_json
    if set(geometry) - set(placement_um):
        raise ValueError('prepared input contains artwork without actual placement')
    schema, reference_kind, reference_text = _prepared_reference(source_cell, tech.pdk_id)
    experimental = None if route_engine is None else experimental_routing_configuration(route_engine)
    if experimental is not None:
        schema = ('clay.canonical-prepared-routing-input/v5' if 'fixed_index_seal' in experimental
                  else 'clay.canonical-prepared-routing-input/v6' if 'group_full_feedback' in experimental
                  else 'clay.canonical-prepared-routing-input/v4')
    payload = dict(schema=schema,
        scope='canonical_complete_preparation_before_solver_not_physical_admission',
        physical_admission=False, execution_scope=execution_scope,
        pdk=tech.pdk_id, logical_devices=len(source_cell.devices),
        stamps={name: dict(polygons_nm=_stamp_records(stamp, placement_um[name]),
            composite_members=getattr(stamp, 'composite_members', ()),
            source=getattr(stamp, 'source', '')) for name, stamp in sorted(geometry.items())},
        pins_by_net=pins_by_net, obstacles=obstacles, constraints=constraints,
        rails_by_net=rails_by_net, bbox_nm=routing_envelope.bbox_nm,
        routing_world=json.loads(routing_world.canonical_json()),
        routing_world_sha256=routing_world.sha256(),
        pdk_routing=json.loads(canonical_pdk_routing_json(tech)),
        placement_state_digest=(None if final_placement_certificate is None else
            final_placement_certificate.placement_state_digest),
        lvs_reference_text=reference_text,
        capture_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selected_engine=None if route_engine is None else route_engine_configuration(route_engine),
        route_argument_names=sorted(route_kwargs or {}))
    if reference_kind is not None:
        payload['lvs_reference_kind'] = reference_kind
    if experimental is not None:
        payload['experimental_routing'] = experimental
        plan = (route_kwargs or {}).get('group_demand_topology_plan')
        if plan is not None:
            from clay.layout.route.resource_plan import TopologyPlan
            if type(plan) is not TopologyPlan:
                raise TypeError('experimental topology plan must have the canonical type')
            payload['group_demand_topology'] = {'plan': json.loads(plan.canonical_json()),
                                                'sha256': plan.sha256()}
        else:
            payload['group_demand_topology'] = None
    if actual_shared_guard_edge_rail is not None:
        payload['actual_shared_guard_edge_rail']=actual_shared_guard_edge_rail
    # Roundtrip creates an isolated value; a sink cannot mutate native objects.
    snapshot = json.loads(json.dumps(_json_value(payload), sort_keys=True, allow_nan=False))
    if experimental is not None:
        validate_experimental_prepared(snapshot)
    return snapshot
