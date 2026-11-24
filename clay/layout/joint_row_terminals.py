"""Typed row access face: joint components are AND, metal accesses are OR.

Seeds here are geometry attribution/display projections. Production routing
must retain the joint face and currently rejects it before creating endpoints.
"""
from dataclasses import asdict

import klayout.db as kdb

from clay.layout.joint_terminal_inventory import JointTerminalInventory
from clay.layout.joint_row_geometry import _region, _require


def is_joint_inventory(inventory):
    return type(inventory) is JointTerminalInventory


def component_seed(shapes, grid):
    """Find a bounded deterministic interior grid seed; never substitute a bbox."""
    region = _region(shapes)
    candidates = set()
    for polygon in region.each():
        box = polygon.bbox()
        candidates.add((((box.left+box.right)//(2*grid))*grid,
                        ((box.bottom+box.top)//(2*grid))*grid))
        for vertex in polygon.each_point_hull():
            x, y = vertex.x//grid*grid, vertex.y//grid*grid
            candidates.update((x+dx*grid, y+dy*grid) for dx in (-1,0,1,2) for dy in (-1,0,1,2))
    for x, y in sorted(candidates):
        if (kdb.Region(kdb.Box(x-1,y-1,x+1,y+1))-region).is_empty():
            return x, y
    raise ValueError('joint row routing mask has no supported interior grid seed')


def _joint_access_layer(alternatives):
    """Select a real seed layer without collapsing independent access islands.

    A single routing component may contain several routing layers joined by
    actual cuts. Its complete layered geometry remains in the joint inventory;
    the selected layer is only the representative used by row contracts. More
    than one independent component retains the existing same-layer OR contract.
    """
    layers = {key for access in alternatives for key, _shapes in access.geometry_by_layer_nm}
    if len(layers) != 1:
        _require(len(alternatives) == 1, 'JOINT_MULTILAYER_ACCESS_UNSUPPORTED')
    return min(layers)


def joint_terminal_contracts(inventory, *, grid=1):
    from clay.layout.place.row_program import RowTerminalAccessContract
    _require(is_joint_inventory(inventory), 'terminal face lacks its exact joint inventory')
    inventory.__post_init__()
    sites = {site.site_id: site for site in inventory.expected_sites}
    accesses = {access.component_id: access for access in inventory.routing_components}
    result = []
    for joint in inventory.joint_components:
        alternatives = tuple(accesses[name] for name in joint.routing_component_ids)
        layer = _joint_access_layer(alternatives)
        points = tuple(sorted(component_seed(dict(access.geometry_by_layer_nm)[layer], grid)
                              for access in alternatives))
        _require(len(set(points)) == len(points), 'distinct routing accesses have ambiguous seeds')
        associations = {(sites[site_id].owner, role) for site_id in joint.site_ids
                        for role in sites[site_id].allowed_roles}
        for member, role in sorted(associations):
            result.append(RowTerminalAccessContract(member=member, role=role, net=joint.source_net,
                layer=layer, accesses_nm=points, component_id=joint.component_id))
    # The existing member view supports one layer for a logical role, even when
    # that role touches several independent AND components.
    for member, role in {(row.member,row.role) for row in result}:
        _require(len({row.layer for row in result if (row.member,row.role)==(member,role)}) == 1,
                 'JOINT_MULTILAYER_ACCESS_UNSUPPORTED')
    return tuple(sorted(result))


def component_route_demands(terminals):
    from clay.layout.place.row_program import RowRouteDemandContract
    return tuple(RowRouteDemandContract(net=net,
        members=tuple(sorted({row.member for row in terminals if row.net == net})),
        terminal_count=len({(row.member,row.role) for row in terminals if row.net == net}),
        access_count=len({(row.component_id,row.layer,point) for row in terminals if row.net == net
                          for point in row.accesses_nm}),
        required_component_ids=tuple(sorted({row.component_id for row in terminals if row.net == net})))
        for net in sorted({row.net for row in terminals}))


def component_terminal_contracts(inventory, *, grid=1):
    from clay.layout.place.row_program import RowTerminalAccessContract
    if is_joint_inventory(inventory):
        return joint_terminal_contracts(inventory, grid=grid)
    by_source = {source.source_id: source for source in inventory.sources}
    return tuple(sorted(RowTerminalAccessContract(member=member, role=role, net=component.net,
        layer=component.seed_layer, accesses_nm=(component.seed_xy_nm,), component_id=component.component_id)
        for component in inventory.components for member,role in sorted({
            (by_source[source_id].member,by_source[source_id].role) for source_id in component.source_ids})))


def require_terminal_member_nets(geometry, consumed_devices, members, tech=None):
    from clay.layout.place.row_program import replay_row_terminal_inventory, RowProgramMaterializationError
    from clay.layout.joint_row_geometry import replay_joint_row_geometry
    if getattr(geometry, 'joint_row_binding', None) is not None:
        inventory = replay_joint_row_geometry(geometry, consumed_devices, tech=tech, members=members)
        require_joint_representatives(geometry, inventory, tech=tech)
        return inventory
    inventory = replay_row_terminal_inventory(geometry=geometry, consumed_devices=consumed_devices, tech=tech)
    if inventory is not None:
        for source in inventory.sources:
            member = members.get(source.member)
            expected = {} if member is None else dict(g=member.gate_net,d=member.drain_net,
                s=member.source_net,b=member.body_net)
            if expected.get(source.role) != source.net:
                raise RowProgramMaterializationError('physical terminal source net differs from declared row member net')
    return inventory


def require_joint_representatives(geometry, inventory, *, tech=None):
    """Validate display/HPWL points without replacing the complete access face."""
    from clay.layout.tech import LayoutTech
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm
    tech = LayoutTech.resolve(geometry.joint_row_binding.pdk_id) if tech is None else tech
    grid = pdk_manufacturing_grid_nm(tech)
    sites = {site.site_id:site for site in inventory.expected_sites}
    access = {row.component_id:row for row in inventory.routing_components}
    masks = {}
    for joint in inventory.joint_components:
        for site_id in joint.site_ids:
            site = sites[site_id]
            for role in site.allowed_roles:
                for identity in joint.routing_component_ids:
                    masks.setdefault((site.owner, role), []).extend(
                        shapes for _layer, shapes in access[identity].geometry_by_layer_nm)
    for owner in sorted({owner for owner,_role in masks}):
        points = geometry.member_pins(owner)
        _require(type(points) is dict and set(points) == {role for member,role in masks if member == owner}
                 and set(points) <= {'d','g','s','b'}, 'representative roles include missing or untraced terminals')
        for role, point in points.items():
            _require(type(point) is tuple and len(point) == 2 and
                     all(type(value) is int and abs(value) <= 1_000_000_000 and value % grid == 0 for value in point),
                     'representative point is not on the exact manufacturing grid')
            x,y = point
            probe = kdb.Region(kdb.Box(x-1,y-1,x+1,y+1))
            _require(any((probe-_region(shapes)).is_empty() for shapes in masks[(owner,role)]),
                     'representative point is outside its traced role/net routing masks')


def terminal_geometry_payload(geometry, inventory):
    if is_joint_inventory(inventory):
        return {'joint_row_binding': geometry.joint_row_binding.payload()}
    from clay.layout.occurrence_artwork import occurrence_artwork_payload
    return {'physical_terminal_inventory': inventory.payload(),
            'owned_artwork_sources': occurrence_artwork_payload(geometry.fixed_occurrence_sources)}


def require_macro_inventory(physical, joint, terminals, demands, *, grid=1):
    from clay.layout.physical_terminal_inventory import PhysicalTerminalInventory
    from clay.layout.place.row_program import RowProgramMaterializationError
    inventory = joint if joint is not None else physical
    if inventory is None:
        if any(row.component_id for row in terminals) or any(row.required_component_ids for row in demands):
            raise RowProgramMaterializationError('component terminal/demand contracts lack their inventory')
        return
    if joint is not None:
        _require(type(grid) is int and 0 < grid <= 1000, 'manufacturing grid is invalid')
        _require(physical is None and is_joint_inventory(joint), 'macro inventory is missing, mixed or foreign')
        joint.__post_init__()
    elif type(physical) is not PhysicalTerminalInventory:
        raise RowProgramMaterializationError('row terminal component inventory must be exact')
    else:
        physical.payload()
    expected = component_terminal_contracts(inventory, grid=grid)
    if terminals != expected:
        raise RowProgramMaterializationError('row terminal components differ from complete source inventory')
    if demands != component_route_demands(expected):
        raise RowProgramMaterializationError('row route demand differs from complete required component set')


def joint_inventory_payload(inventory):
    inventory.__post_init__()
    return asdict(inventory)
