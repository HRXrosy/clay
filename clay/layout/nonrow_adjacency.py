"""Source-driven internal adjacency with complete native-layer collision replay."""
from types import SimpleNamespace
from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanError


class NativeInternalSpacingError(RigidNonmosMacroPlanError):
    """The complete native polygons reject an otherwise valid rigid tiling."""


def project_internal_adjacencies(graph, intent, tech):
    from clay.layout.intent import replay_adjacent_inventory
    from clay.layout.nonrow_macro_plan import RigidNonmosMemberAdjacency, RigidNonmosMacroPlanError
    from clay.layout.place.constraints_well_guard import _installed_adjacency_pitch_nm

    declarations = replay_adjacent_inventory(getattr(intent, 'adjacencies', ()))
    structural = tuple(macro for macro in graph.macros if macro.kind in {'passive_bank', 'bjt_cluster'})
    projected = []
    for declaration in declarations:
        owners = tuple(macro for macro in structural
                       if {declaration.source, declaration.target} <= set(macro.members))
        if not owners:
            continue
        if len(owners) != 1:
            raise RigidNonmosMacroPlanError('internal adjacency has ambiguous physical ownership')
        ratio = declaration.max_separation_multiplier
        pitch = _installed_adjacency_pitch_nm(tech)
        maximum = (pitch * ratio.numerator + ratio.denominator - 1) // ratio.denominator
        first, second = sorted((declaration.source, declaration.target))
        projected.append(RigidNonmosMemberAdjacency(
            first=first, second=second, maximum_nm=maximum,
            declaration_ref=f'@adjacent({declaration.source},{declaration.target},{ratio})'))
    return tuple(sorted(set(projected)))


def require_internal_native_geometry(plans, inputs, geometries, tech):
    """Check all native polygons, not just a poly/body bounding-box proxy."""
    from clay.layout.extract.realize_render import _placed_stamp_layer_rectangles_nm
    from clay.layout.nonrow_macro_plan import RigidNonmosMacroPlanError
    from clay.layout.place.constraints_spatial import _seed_geometry_legal
    from clay.layout.place.constraints_well_guard import _pdk_layer_clearances_nm
    from clay.layout.source_bound_nonmos_plan import SourceBoundNonmosMacroPlan
    from clay.layout.source_resistor_pcell_assembly import CurrentResistorPCellAssembly

    declarations = getattr(inputs, 'member_adjacencies', ())
    if not declarations:
        return
    layer_clearances = _pdk_layer_clearances_nm(tech)
    for plan in plans.plans:
        local = tuple(item for item in declarations
                      if {item.first, item.second} <= set(plan.members))
        if not local:
            continue
        if (type(plan) is SourceBoundNonmosMacroPlan
                and type(plan.assembly) is CurrentResistorPCellAssembly):
            # This is one immutable native PCell, not separately packed member
            # rectangles. Replay all original emission and terminals; treating
            # shared implant/marker/metal as foreign collisions would destroy it.
            from clay.layout.source_resistor_placement import replay_resistor_member_geometry
            plan.assembly.validate()
            for member in plan.members:
                replay_resistor_member_geometry(plan.assembly, member, geometries[member], tech=tech)
            continue
        stamps = {}
        dims = {}
        seed = {}
        for member_plan in plan.member_plans:
            member = member_plan.member
            cell = geometries[member]
            x0, y0, x1, y1 = cell.extent()
            layers = {}
            for shape in cell.polygons:
                key = (int(shape.layer), int(shape.datatype))
                layers.setdefault(key, []).append([
                    ((point.x - x0) / 1000.0, (point.y - y0) / 1000.0)
                    for point in shape.polygon.vertices])
            stamps[member] = SimpleNamespace(polys_by_layer=layers)
            dims[member] = (int(x1 - x0), int(y1 - y0))
            seed[member] = member_plan.local_bbox_nm[:2]
        atoms = _placed_stamp_layer_rectangles_nm(stamps, {name: (0.0, 0.0) for name in stamps})
        if set(atoms) != set(plan.members):
            raise RigidNonmosMacroPlanError('internal adjacency lacks complete native geometry')
        footprints = tuple((name, tuple(sorted({(*layer, *box)
                            for layer, boxes in atoms[name].items() for box in boxes})))
                           for name in sorted(atoms))
        gap = min((inputs.gap_nm, *(item.maximum_nm for item in local)))
        if not _seed_geometry_legal(seed, dims=dims, gap_nm=gap,
                pair_clearances_nm={}, layered_collision_footprints=footprints,
                layer_clearances_nm=layer_clearances):
            raise NativeInternalSpacingError(
                'internal adjacency violates native PDK layer spacing: ' + plan.macro_id)


def compile_with_native_spacing(graph, inputs, geometries, tech):
    """Relegalize a too-small default gap without altering authored maxima.

    The default COMP/poly gap is not an implant spacing authority. Search
    exact manufacturing-grid compile inputs in increasing order and admit
    only a complete native-layer replay. The selected input gap is hashed by
    the existing compiler; successful existing inputs remain byte-identical.
    """
    from dataclasses import replace
    from clay.layout.nonrow_macro_plan import compile_rigid_nonmos_macro_plans_from_inputs
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    plans = compile_rigid_nonmos_macro_plans_from_inputs(graph, inputs)
    try:
        require_internal_native_geometry(plans, inputs, geometries, tech)
    except NativeInternalSpacingError as failure:
        declarations = inputs.member_adjacencies
        maximum = min(item.maximum_nm for item in declarations)
        grid = max(1, int(pdk_manufacturing_grid_nm(tech)))
        start = (inputs.gap_nm // grid + 1) * grid
        for gap in range(start, maximum + 1, grid):
            candidate = replace(inputs, gap_nm=gap)
            candidate_plans = compile_rigid_nonmos_macro_plans_from_inputs(graph, candidate)
            try:
                require_internal_native_geometry(candidate_plans, candidate, geometries, tech)
            except NativeInternalSpacingError:
                continue
            return candidate_plans, candidate
        raise NativeInternalSpacingError(
            f'{failure}; no legal native gap within the unchanged authored maxima'
        ) from failure
    return plans, inputs
