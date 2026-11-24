"""Emit passive pins only after their retained native artwork/site replay."""


def emit_native_passive_pins(passive, *, geometry, placement, position, tech, emit,
                             report, exclude_power):
    """Return false only for an unmaterialized passive's existing scalar path."""
    name = passive.name
    from clay.layout.source_resistor_placement import is_resistor_stamp
    assemblies = [(key, value) for key, value in (geometry or {}).items()
        if is_resistor_stamp(value) and name in getattr(value, 'composite_members', ())]
    if assemblies:
        if len(assemblies) != 1:
            raise ValueError('source resistor route owner has multiple assembly stamps')
        from clay.layout.source_resistor_placement import verified_resistor_assembly_accesses
        key, stamp = assemblies[0]
        stamp_position = placement.get(key)
        if type(stamp_position) not in (tuple, list) or len(stamp_position) != 2:
            raise ValueError('source resistor assembly has no current placement')
        for role, net, layer, point, index in verified_resistor_assembly_accesses(
                stamp, passive=passive, position=tuple(stamp_position), tech=tech):
            if not net or exclude_power(net):
                continue
            physical_name = name if index == 0 else f'{name}@{index + 1}'
            relative = (point[0]/1000-position[0], point[1]/1000-position[1])
            emit(net, physical_name, role, (relative, 0.0, True), position, layer=layer)
            if tuple(report.pins_by_net[net][-1].xy_nm) != point:
                raise ValueError('source resistor final emitted access changed')
        return True
    from clay.layout.route.route_cell import _PlacedPassiveStamp
    stamp = geometry.get(name) if geometry is not None else None
    if (isinstance(stamp, _PlacedPassiveStamp) or hasattr(stamp, 'materialized_passive_access')
            or hasattr(stamp, 'passive_artwork_origin_um')):
        from clay.layout.passive_terminal_binding import (
            verified_passive_route_accesses, verify_emitted_passive_access,
        )
        accesses, native_nodes = verified_passive_route_accesses(
            stamp, passive=passive, position=position, tech=tech)
        for expected in accesses:
            role, net, layer, point, _node, access_index = expected
            if not net or exclude_power(net):
                continue
            physical_name = name if access_index == 0 else f'{name}@{access_index + 1}'
            relative = (point[0]/1000-position[0], point[1]/1000-position[1])
            emit(net, physical_name, role, (relative, 0.0, True), position, layer=layer)
            verify_emitted_passive_access(report.pins_by_net[net][-1], expected=expected, nodes=native_nodes)
        return True
    return False
