"""Source/materialization boundary for the generic non-MOS placer."""


def _require_member_record(record, assembly, owner, orient):
    from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
    if type(assembly) is CurrentResistorPCellAssembly:
        from clay.layout.source_resistor_placement import require_resistor_member_record
        return require_resistor_member_record(record, assembly, owner, orient)
    members = [m for m in assembly.source.members if m.owner == owner]
    if len(members) != 1:
        raise ValueError('native assembly record names an unknown owner')
    member = members[0]
    if (record.get('kind') != member.kind or record.get('source_model') != member.source_model
            or record.get('units') != member.unit_count or tuple(sorted(record.get('pins', {}).items())) != member.pins
            or str(orient).upper() != 'R0'):
        raise ValueError('native assembly source member record changed')
    pattern = record.get('unit_pattern')
    if pattern is not None and pattern != member.unit_pattern:
        raise ValueError('native assembly source member pattern changed')
    if pattern is None and member.unit_count != 1:
        raise ValueError('native assembly source member pattern is missing')
    tap = record.get('tap_declaration')
    if ((tap is None) != (member.tap is None) or tap is not None
            and (tap.net, tap.where) != member.tap[:2]):
        raise ValueError('native assembly source member tap changed')


def native_member_geometry(record, tech, orient):
    from clay.layout.bjt_local_assembly import NativeNonmosAssemblyBinding, bjt_assembly_member_geometry
    assembly = record.get('_native_assembly')
    from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly
    if type(assembly) is CurrentResistorPCellAssembly:
        from clay.layout.source_resistor_placement import resistor_member_geometry
        _require_member_record(record, assembly, record.get('_native_assembly_owner'), orient)
        return resistor_member_geometry(assembly, record['_native_assembly_owner'], tech=tech)
    if type(assembly) is not NativeNonmosAssemblyBinding:
        raise TypeError('native member geometry requires a typed assembly binding')
    owner = record.get('_native_assembly_owner')
    _require_member_record(record, assembly, owner, orient)
    return bjt_assembly_member_geometry(assembly, owner, tech=tech)


def collect_nonmos_placement_inputs(cell, source_cir, intent, tech, passive_overrides,
                                    native_assemblies, graph):
    from clay.layout.generic_placement import _apply_passive_geometry_overrides, _collect_passives, _collect_bjts
    records = {**_apply_passive_geometry_overrides(_collect_passives(cell, source_cir),
                   passive_overrides, source_cell=cell, tech=tech), **_collect_bjts(cell, tech, intent)}
    if type(native_assemblies) is not tuple:
        raise TypeError('native assemblies require an exact tuple')
    if not native_assemblies:
        return records
    from clay.layout.bjt_local_assembly import NativeNonmosAssemblyBinding, replay_bjt_local_assembly
    claimed = set()
    keys = []
    for assembly in native_assemblies:
        from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly, replay_current_resistor_assembly
        resistor = type(assembly) is CurrentResistorPCellAssembly
        if type(assembly) is not NativeNonmosAssemblyBinding and not resistor:
            raise TypeError('native assemblies require complete materialized bindings')
        members = tuple(m.owner for m in assembly.members)
        keys.append(members)
        if claimed.intersection(members) or not set(members) <= set(records):
            raise ValueError('native assembly has duplicated or missing source owners')
        claimed.update(members)
        kind = 'passive_bank' if resistor else 'bjt_cluster'
        owners = [m for m in getattr(graph, 'macros', ()) if m.kind == kind and m.members == members]
        if len(owners) != 1:
            raise ValueError('native assembly requires one complete declared physical owner')
        replay = replay_current_resistor_assembly if resistor else replay_bjt_local_assembly
        replay(assembly, source_cell=cell, tech=tech, intent=intent)
        for member in members:
            _require_member_record(records[member], assembly, member, records[member].get('orient', 'R0'))
            records[member] = {**records[member], '_native_assembly': assembly, '_native_assembly_owner': member}
    if keys != sorted(keys):
        raise ValueError('native assemblies require canonical member order')
    return records


def replay_native_nonmos_inputs(inputs, *, source_cell, tech, intent):
    """Replay the current source/PDK at an admission boundary, beyond data hashes."""
    from clay.layout.nonrow_macro_plan import RigidNonmosMacroCompileInputs
    from clay.layout.bjt_local_assembly import replay_bjt_local_assembly
    if type(inputs) is not RigidNonmosMacroCompileInputs:
        raise TypeError('native nonmos admission requires complete source compile inputs')
    for assembly in inputs.native_assemblies:
        from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly, replay_current_resistor_assembly
        replay = replay_current_resistor_assembly if type(assembly) is CurrentResistorPCellAssembly else replay_bjt_local_assembly
        replay(assembly, source_cell=source_cell, tech=tech, intent=intent)


def replay_native_nonmos_placement(placement, *, source_cell, tech, intent):
    """Join retained placement authority to the source and PDK open at this bridge.

    Any retained part of the native assembly proof requires the complete compiler
    product. Removing just its compile inputs must not select the legacy path.
    """
    from clay.layout.generic_placement import verify_rigid_nonmos_macro_projection
    inputs = getattr(placement, 'rigid_nonmos_macro_inputs', None)
    plans = getattr(placement, 'rigid_nonmos_macro_plans', None)
    certificate = getattr(placement, 'row_final_certificate', None)
    graph = getattr(placement, 'physical_macro_graph', None)
    declared = any(macro.kind in ('bjt_cluster', 'passive_bank') for macro in getattr(graph, 'macros', ()))
    retained = (inputs, plans, getattr(placement, 'rigid_nonmos_macro_interfaces', None),
                getattr(placement, 'rigid_nonmos_placement_certificate', None),
                getattr(certificate, 'nonmos_assembly_binding', None))
    if declared or any(value is not None for value in retained):
        verify_rigid_nonmos_macro_projection(placement)
        replay_native_nonmos_inputs(inputs, source_cell=source_cell, tech=tech, intent=intent)
