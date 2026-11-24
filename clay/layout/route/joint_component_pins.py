"""Lossless Pin lowering when each joint obligation has one metal component.

One Pin denotes one actual connected single-layer routing component, not an
arbitrary OR island. Multiple independent metal islands and multilayer
components are rejected before publishing any partial pin inventory.
"""
from collections import Counter

from clay.layout.route.engine import Pin
from clay.layout.route.terminal_identity import PhysicalTerminalPinBinding, pin_fact, pin_source_aliases


class JointRoutingAlternativesUnsupported(ValueError):
    pass


class JointRoutingMultilayerUnsupported(ValueError):
    pass


def build_current_joint_component_pins(*, allow_fixed_multilayer=False, multilayer_engine=None, **operands):
    from clay.layout.row_terminal_probes import _replayed_row_pins
    from clay.layout.joint_row_terminals import component_seed, _joint_access_layer
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    faces = []
    legacy, members = _replayed_row_pins(**operands, _joint_faces=faces)
    # Complete preflight precedes lowering: no partial endpoint denominator.
    for face in faces:
        routing = {row.component_id: row for row in face.joint_inventory.routing_components}
        multilayer = False
        for component in face.joint_inventory.joint_components:
            if len(component.routing_component_ids) != 1:
                raise JointRoutingAlternativesUnsupported(
                    'JOINT_ROUTING_ALTERNATIVES_UNSUPPORTED: complete OR domain required; '
                    + face.stamp_id + '/' + component.component_id)
            access = routing[component.routing_component_ids[0]]
            if len(access.geometry_by_layer_nm) != 1:
                if not allow_fixed_multilayer:
                    raise JointRoutingMultilayerUnsupported(
                        'JOINT_ROUTING_MULTILAYER_UNSUPPORTED: source layer policy and '
                        'fixed-cut connectivity require validation; '
                        + face.stamp_id + '/' + component.component_id)
                multilayer = True
        if multilayer:
            from clay.layout.route.native_output_joint_admission import require_output_profile_face
            require_output_profile_face(face, operands, multilayer_engine)
    pins = list(legacy)
    grid = pdk_manufacturing_grid_nm(operands['tech'])
    for face in faces:
        inventory = face.joint_inventory
        sites = {row.site_id:row for row in inventory.expected_sites}
        routing = {row.component_id:row for row in inventory.routing_components}
        for component in inventory.joint_components:
            access = routing[component.routing_component_ids[0]]
            layer = _joint_access_layer((access,))
            shapes = dict(access.geometry_by_layer_nm)[layer]
            local = component_seed(shapes, grid)
            point = tuple(local[i] + face.delta_nm[i] for i in range(2))
            refs = tuple(sorted((site_id + '/' + role, sites[site_id].owner, role)
                for site_id in component.site_ids for role in sites[site_id].allowed_roles))
            binding = PhysicalTerminalPinBinding(face.stamp_id, component.component_id, refs)
            pins.append(Pin(net=component.source_net, device=refs[0][1], terminal=refs[0][2],
                layer=layer, xy_nm=point, xy_um=tuple(value/1000 for value in point),
                terminal_binding=binding))
    facts = tuple(sorted((pin_fact(pin) for pin in pins), key=repr))
    if len(set(facts)) != len(facts):
        raise ValueError('joint routing component pins are duplicated')
    return tuple(sorted(pins, key=lambda pin:repr(pin_fact(pin)))), members


def require_current_joint_component_pins(*, pins_by_net, **operands):
    expected, members = build_current_joint_component_pins(**operands)
    actual = []
    for net, pins in pins_by_net.items():
        for pin in pins:
            if any(member in members for member,_role in pin_source_aliases(pin)):
                if pin.net != net:
                    raise ValueError('joint component pin is listed under another net')
                actual.append(pin_fact(pin))
    if Counter(actual) != Counter(pin_fact(pin) for pin in expected):
        raise ValueError('joint component probes omit or change the complete source denominator')
    return expected
