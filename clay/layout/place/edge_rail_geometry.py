"""Fresh source-stamp rail inventory and final actual-model comparison.

The source adapter never loads an execution receipt. Captured conductor ownership
is a placement operand; the final prepared route model must independently agree
after movement, because global M1 ownership and over-cell abstractions may change.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from collections.abc import Mapping


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError('edge rail geometry ' + message)


def _canonical_ring(points):
    points = tuple(points)
    if points and points[0] == points[-1]:
        points = points[:-1]
    _require(len(points) >= 3, 'has an incomplete polygon')
    return min(tuple(seq[i:]+seq[:i]) for seq in (points, tuple(reversed(points)))
               for i, p in enumerate(seq) if p == min(seq))


def _stamp_geometry(stamp, position):
    from clay.layout.placed_geometry import placed_point_nm
    polygons = getattr(stamp, 'polys_by_layer', None)
    _require(isinstance(polygons, Mapping) and bool(polygons), 'has no actual stamp polygons')
    rows = []
    for layer, rings in polygons.items():
        _require(type(layer) is tuple and len(layer) == 2 and all(type(v) is int for v in layer),
                 'has malformed physical layer identity')
        for ring in rings:
            rows.append((layer, _canonical_ring(tuple(placed_point_nm(float(x), float(y), position)
                                                     for x, y in ring))))
    _require(bool(rows), 'has an empty physical stamp')
    points = tuple(p for _, ring in rows for p in ring)
    bbox = (min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points))
    _require(bbox[0] < bbox[2] and bbox[1] < bbox[3], 'has an empty physical stamp extent')
    normalized = sorted((layer, tuple((x-bbox[0], y-bbox[1]) for x,y in ring)) for layer,ring in rows)
    return bbox, _sha(normalized)


def _conductor_box(row):
    return tuple(round(float(v)*1000) for v in (row.x0, row.y0, row.x1, row.y1))


def _shift(box, delta):
    return (box[0]+delta[0], box[1]+delta[1], box[2]+delta[0], box[3]+delta[1])


def _bind_inventory(*, solver_blocks, materialized_blocks, block_stamps, block_gring,
                    geometry, placement_um, master_members, member_offsets_nm,
                    relevant_layers, conductors, flexible_envelope_constraints=()):
    from clay.layout.place.edge_rail_reservation import EdgeRailStampIR, EdgeRailConductorAtom
    solver, materialized = dict(solver_blocks), dict(materialized_blocks)
    _require(len(solver) == len(solver_blocks) and len(materialized) == len(materialized_blocks),
             'has duplicate physical block identities')
    _require(set(master_members) == set(solver), 'master inventory is incomplete')
    from clay.layout.place.constraints_spatial import _canonical_flexible_envelope_constraints
    flexible = _canonical_flexible_envelope_constraints(
        flexible_envelope_constraints, block_ids=tuple(solver))
    # Shared-well members are placed by their device-content coordinates.
    # Recover that exact frame from current complete source stamps and replay
    # the existing physical-offset authority against the actual ring artwork.
    # Ordinary rigid assemblies retain their original dimension check.
    for row in flexible:
        for member, offsets in row.member_physical_offsets_nm:
            master_box = solver[member]
            if offsets == (0, 0, master_box[2] - master_box[0],
                           master_box[3] - master_box[1]):
                # An ordinary route-group envelope uses the whole existing
                # rigid master as its physical frame, including composites.
                # The shared-well content-frame recovery below is unnecessary.
                continue
            _require(master_members[member] == (member,), 'flexible member ownership differs')
            names = tuple(block_stamps.get(member, ()))
            _require(bool(names), 'flexible member has no source content')
            boxes = [_stamp_geometry(geometry[name], placement_um[name])[0] for name in names]
            content = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                       max(b[2] for b in boxes), max(b[3] for b in boxes))
            ring = block_gring.get(member)
            if ring is not None:
                boxes.append(_stamp_geometry(geometry[ring], placement_um[ring])[0])
            physical = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                        max(b[2] for b in boxes), max(b[3] for b in boxes))
            observed = (physical[0]-content[0],physical[1]-content[1],
                        physical[2]-content[0],physical[3]-content[1])
            _require(observed == offsets, 'flexible physical offset differs from actual ring')
            _require(materialized[member] in (content, physical), 'flexible source frame differs from actual artwork')
            materialized[member] = content
    binding = {}
    consumed = set()
    for master, master_box in sorted(solver.items()):
        members = tuple(master_members[master])
        _require(bool(members) and len(set(members)) == len(members), 'member inventory is incomplete')
        targets = {}
        for member in members:
            _require(member in materialized and member not in consumed, 'materialized member ownership differs')
            consumed.add(member)
            source = materialized[member]
            if member in member_offsets_nm:
                owner, offset = member_offsets_nm[member]
                _require(owner == master and type(offset) is tuple and len(offset) == 2
                         and all(type(v) is int for v in offset), 'member offset has foreign ownership')
                targets[member] = (offset[0], offset[1], offset[0]+source[2]-source[0], offset[1]+source[3]-source[1])
            else:
                targets[member] = source
        union = (min(b[0] for b in targets.values()), min(b[1] for b in targets.values()),
                 max(b[2] for b in targets.values()), max(b[3] for b in targets.values()))
        _require((union[2]-union[0], union[3]-union[1]) ==
                 (master_box[2]-master_box[0], master_box[3]-master_box[1]), 'master dimensions differ from rigid assembly')
        for member, target in targets.items():
            source = materialized[member]
            delta = (target[0]-source[0]-union[0], target[1]-source[1]-union[1])
            names = tuple(block_stamps.get(member, ()))
            if block_gring.get(member) is not None:
                names += (block_gring[member],)
            _require(bool(names), 'materialized member has no physical stamps')
            for name in names:
                _require(name not in binding and name in geometry and name in placement_um,
                         'stamp source inventory is missing or duplicated')
                actual_box, shape_sha = _stamp_geometry(geometry[name], placement_um[name])
                binding[name] = (master, delta, _shift(actual_box, delta), shape_sha)
    _require(consumed == set(materialized), 'materialized member denominator differs')
    _require(set(binding) == set(geometry), 'complete physical stamp inventory differs')
    atoms = []
    for row in conductors:
        if tuple(row.layer) not in relevant_layers or row.kind == 'device_projection':
            continue
        name = getattr(row, 'source_stamp', None)
        _require(type(name) is str and name in binding, 'conductor has missing or foreign source stamp')
        master, delta, _, _ = binding[name]
        atoms.append(EdgeRailConductorAtom(stamp_id=name, master_id=master, layer_key=tuple(row.layer),
            net=row.net, bbox_nm=_shift(_conductor_box(row), delta), kind=row.kind))
    atoms = tuple(sorted(atoms, key=lambda row: _json(asdict(row))))
    stamps = []
    for name, (master, _, box, geometry_sha) in sorted(binding.items()):
        owned = tuple(row for row in atoms if row.stamp_id == name)
        # This digest binds the model's complete local electrical attribution on
        # relevant rail layers, including honestly unowned routing abstractions.
        # Placed terminal_conductor_ref strings contain absolute coordinates.
        access_rows = [(row.layer_key, row.net, row.kind, _shift(row.bbox_nm, (-box[0],-box[1]))) for row in owned]
        stamps.append(EdgeRailStampIR(stamp_id=name, master_id=master, offset_nm=box[:2], bbox_nm=box,
            geometry_sha256=geometry_sha, access_sha256=_sha(access_rows), conductor_atom_count=len(owned),
            guard_source_members=getattr(geometry[name], 'guard_ring_declared_members', ()),
            guard_declaration_sha256=getattr(geometry[name], 'guard_ring_intent_sha256', '')))
    return tuple(stamps), atoms


def _topology_digest(bridged_intent):
    fields = ('name','kind','oxide','nets','W','L','M','value','model','orient','resistor_segments','area','tap_net','tap_where')
    from clay.layout.route.rail_plan import _canonical_parsed_rail_ast_value
    return _sha({collection: [{field: _canonical_parsed_rail_ast_value(getattr(row,field))
                              for field in fields if hasattr(row,field)}
                             for row in sorted(getattr(bridged_intent,collection,()) or (), key=lambda row:row.name)]
                 for collection in ('devices','passives','bjts')})


def _lower_access_contract(row, tech):
    from clay.layout.route.rail_plan import _required_metal_rule

    layers = tuple(sorted(tuple(tech.metal_key(number)) for number in range(1, row.layer_n)))
    if not layers:
        return layers, 0
    via = tech.via_rule(f'via{row.layer_n - 1}')
    cut = getattr(via.size, 'value', None)
    enclosure = getattr(via.enclosure_above, 'value', None)
    _require(type(cut) is int and cut > 0 and type(enclosure) is int and enclosure >= 0,
             'lower terminal access requires actual PDK via cut/enclosure')
    return layers, (_required_metal_rule(tech, row.layer_n, 'min_space')
                    + (cut + 1) // 2 + enclosure)


def _guaranteed_rectangular_rail_wide_space(tech, demand, stamps, grid_nm):
    """PDK wide clearance only when the actual full-span rail is provably wide.

    A rail is an exact rectangle across the physical outline.  Any original
    stamp's run-axis span is a lower bound on that outline.  Allow one grid of
    snapping error; unsupported bands stay with the existing base predicate and
    are checked on the final literal geometry by the wide-spacing observer.
    """
    query = getattr(tech, 'width_spacing_entries', None)
    if not callable(query) or not stamps:
        return 0
    axis = getattr(demand, 'actual_axis', None)
    _require(axis in ('H', 'V'), 'wide rail has no actual axis')
    _require(type(grid_nm) is int and grid_nm > 0, 'wide rail grid is invalid')
    run_low, run_high = ((0, 2) if axis == 'H' else (1, 3))
    minimum_run = max(row.bbox_nm[run_high] - row.bbox_nm[run_low]
                      for row in stamps) - grid_nm
    base = tech.min_space(f'metal{demand.layer_n}')
    _require(type(base) is int and base >= 0, 'wide rail base spacing is unavailable')
    result = 0
    for band in query(f'metal{demand.layer_n}'):
        threshold = getattr(band, 'wide_ge', None)
        opening = getattr(band, 'opening_nm', None)
        spacing = getattr(getattr(band, 'spacing', None), 'value', None)
        if (type(threshold) is int and type(opening) is int and opening > 0
                and type(spacing) is int and spacing > base
                and getattr(band, 'run_length_ge', None) == 0):
            # Opening erodes by ``opening`` on *both* edges.  The extra grid
            # guarantees a nonempty core even for an authority whose wide_ge
            # happens to be smaller than its literal opening threshold.
            required = max(threshold, 2 * opening + grid_nm)
            if demand.width_nm >= required and minimum_run >= required:
                result = max(result, spacing)
    return result


def compile_edge_rail_reservation_from_geometry(*, intent, tech, bridged_intent, solver_blocks,
        materialized_blocks, block_stamps, block_gring, geometry, placement_um,
        master_members, member_offsets_nm, cell_def=None,
        row_program=None, row_materialization=None, flexible_envelope_constraints=(), native_geometry=None,
        bands=None):
    operands = locals().copy()
    from clay.layout.place.rail_row_replay_scope import rail_source_row_scope
    with rail_source_row_scope(operands):
        return _compile_edge_rail_reservation_from_geometry(**operands)


def _compile_edge_rail_reservation_from_geometry(*, intent, tech, bridged_intent, solver_blocks,
        materialized_blocks, block_stamps, block_gring, geometry, placement_um,
        master_members, member_offsets_nm, cell_def=None,
        row_program=None, row_materialization=None, flexible_envelope_constraints=(), native_geometry=None,
        bands=None):
    from clay.layout.place_blocks_2d import _compiled_edge_rail_specs_for_placement
    from clay.layout.route.route_cell import (
        _BridgedIntent,
        _canonical_rail_source_ast, _declared_ground_ports,
        _append_guard_ring_tap_pins, _guard_ring_taps_from_stamps)
    from clay.layout.route.rail_plan import _required_metal_rule
    from clay.layout.route.resource_world import pdk_provenance_digest
    from clay.layout.route.emit_gds import _nearest_on_grid, pdk_manufacturing_grid_nm
    from clay.layout.route.pins import extract_pins
    from clay.layout.model import build_layout_model
    from clay.layout.joint_placement_terminals import build_placement_terminal_face
    from clay.layout.place.edge_rail_reservation import EdgeRailDemandIR, EdgeRailReservationIR
    specs = _compiled_edge_rail_specs_for_placement(intent, tech, bridged_intent)
    has_native_rows = (getattr(tech, 'pdk_id', None) == 'sky130'
                       and native_geometry is not None and row_program is not None
                       and bool(row_program.entries))
    if not specs and getattr(tech, 'pdk_id', None) != 'gf180mcu' and not has_native_rows:
        return None
    _require(cell_def is not None, 'requires the actual parsed source cell')
    source_ast, _, _ = _canonical_rail_source_ast(cell_def)
    # Shield fences add route endpoints only, not pre-rail artwork. Their
    # synthetic pin owners cannot seed any source stamp's upper-metal islands.
    # The final prepared-model comparison still covers all actual net facts.
    terminal_face = build_placement_terminal_face(
        tech=tech, geometry=dict(geometry), placement_um=dict(placement_um),
        row_program=row_program, row_materialization=row_materialization,
        intent=intent, source_cell=cell_def)
    ordinary = bridged_intent
    if terminal_face.consumed_devices:
        devices = tuple(getattr(bridged_intent, 'devices', ()) or ())
        names = {str(device.name) for device in devices}
        _require(terminal_face.consumed_devices <= names,
                 'row terminal owners differ from source device inventory')
        ordinary = _BridgedIntent(
            [device for device in devices if str(device.name) not in terminal_face.consumed_devices],
            getattr(bridged_intent, 'passives', ()) or (),
            getattr(bridged_intent, 'bjts', ()) or (),
            well_isolated_devices=getattr(bridged_intent, 'well_isolated_devices', ()) or (),
            substrate_net=getattr(bridged_intent, 'substrate_net', '') or '',
            deep_nwell_net=getattr(bridged_intent, 'deep_nwell_net', '') or '',
        )
    pins = extract_pins(ordinary, dict(placement_um), tech=tech, return_report=True,
        include_power=True, geometry=dict(geometry), per_device_fold=True,
        extra_power_nets=_declared_ground_ports(cell_def)).pins_by_net
    pins = terminal_face.attribute_model_pins(pins)
    terminal_counts = terminal_face.terminal_counts(pins)
    # Source terminal denominators intentionally exclude route-owned ring taps,
    # but physical M1 ownership must include the native accesses on those stamps.
    _append_guard_ring_tap_pins(pins, _guard_ring_taps_from_stamps(geometry, placement_um))
    model = build_layout_model(bridged_intent, dict(placement_um), dict(geometry), tech=tech,
        pins_by_net=pins, include_upper_artwork=True, strict_device_m1_provenance=True,
        include_device_projection=False)
    # Preserve lower-metal terminal ownership too. Same-layer spacing alone
    # admits an M2 supply stripe directly over an entire foreign M1 pad/ring,
    # sealing every M1-to-M2 landing while claiming a rail-legal placement.
    contracts = {row.demand_id: _lower_access_contract(row, tech) for row in specs}
    access_layers = {key: value[0] for key, value in contracts.items()}
    access_clearance = {key: value[1] for key, value in contracts.items()}
    layers = frozenset(key for row in specs
                       for key in (tuple(row.layer_key), *access_layers[row.demand_id]))
    stamps, atoms = _bind_inventory(solver_blocks=solver_blocks, materialized_blocks=materialized_blocks,
        block_stamps=block_stamps, block_gring=block_gring, geometry=geometry, placement_um=placement_um,
        master_members=master_members, member_offsets_nm=member_offsets_nm, relevant_layers=layers,
        conductors=model.conductors, flexible_envelope_constraints=flexible_envelope_constraints)
    demands = tuple(EdgeRailDemandIR(demand_id=row.demand_id, declaration_ref=row.declaration_ref,
        declaration_sha256=hashlib.sha256(row.declaration_ref.encode('utf-8')).hexdigest(), net=row.net,
        layer_n=row.layer_n, layer_key=tuple(row.layer_key), actual_axis=row.actual_axis, edge=row.edge,
        width_nm=row.width_nm, offset_nm=getattr(row, 'offset_nm', 0), min_space_nm=_required_metal_rule(tech,row.layer_n,'min_space'),
        wide_space_nm=_guaranteed_rectangular_rail_wide_space(
            tech, row, stamps, pdk_manufacturing_grid_nm(tech)),
        access_layer_keys=access_layers[row.demand_id], access_clearance_nm=access_clearance[row.demand_id],
        terminal_count=terminal_counts[row.net], joint_component_ids=tuple(sorted(identity
            for face in terminal_face.joint_faces for net, identity in face.demand_ids if net == row.net))) for row in specs)
    from .native_feol_reservation import compile_native_feol_pairs
    feol_pairs = compile_native_feol_pairs(tech=tech, stamps=stamps, geometry=geometry,
                                          placement_um=placement_um)
    if has_native_rows:
        from .native_feol_reservation import compile_sky130_row_feol_pairs, compile_sky130_group_row_spacing
        feol_pairs += compile_sky130_row_feol_pairs(bundle=native_geometry, cell=cell_def,
            stamps=stamps, geometry=geometry, placement_um=placement_um,
            row_program=row_program, row_materialization=row_materialization, tech=tech)
        feol_pairs += compile_sky130_group_row_spacing(intent=intent, bundle=native_geometry, cell=cell_def,
            stamps=stamps, geometry=geometry, placement_um=placement_um,
            row_program=row_program, row_materialization=row_materialization, tech=tech)
        feol_pairs = tuple(sorted(feol_pairs, key=lambda row: (row.left_stamp, row.right_stamp,
            row.rule_id, row.left_kind, row.right_kind)))
    if not demands and not feol_pairs:
        return None
    # Rail reservations require a complete pre-rail artwork inventory. A later
    # plane cannot enter that contract. FEOL-only reservations retain all source
    # pairs: upper-metal planes change none of their operands and are validated
    # separately against the final-placement envelope and exact GDS.
    if demands:
        _require(not tuple(getattr(intent,'power_planes',()) or ()),
                 'cannot represent late power-plane artwork')
    grid_nm = pdk_manufacturing_grid_nm(tech)
    right_v_demands = tuple(
        row for row in demands if row.edge == 'right' and row.actual_axis == 'V'
    )
    right_margin_nm = 0
    if right_v_demands:
        raw_margin = max(
            row.offset_nm + row.width_nm + max(
                row.min_space_nm, row.wide_space_nm, row.access_clearance_nm
            )
            for row in right_v_demands
        )
        right_margin_nm = ((raw_margin + grid_nm - 1) // grid_nm) * grid_nm
    top_h_demands = tuple(
        row for row in demands if row.edge == 'top' and row.actual_axis == 'H'
    )
    top_margin_nm = 0
    if top_h_demands:
        from clay.layout.place.edge_rail_reservation import _domain_nonempty, _lower_access_domains
        master_ids = tuple(name for name, _box in sorted(solver_blocks))
        stamp_top_by_master = {}
        for stamp in stamps:
            prior = stamp_top_by_master.get(stamp.master_id)
            stamp_top_by_master[stamp.master_id] = (
                stamp.bbox_nm[3] if prior is None else max(prior, stamp.bbox_nm[3])
            )
        if isinstance(bands, Mapping) and bands:
            active_bands = {m: bands[m] for m in master_ids if m in bands}
            if active_bands:
                max_band = max(active_bands.values())
                top_candidates = tuple(m for m in master_ids if active_bands.get(m) == max_band)
            else:
                top_candidates = master_ids
        else:
            top_candidates = master_ids
        atoms_by_master_layer = {}
        for atom in atoms:
            atoms_by_master_layer.setdefault((atom.master_id, atom.layer_key), []).append(atom)
        deficits = {}
        for m in top_candidates:
            if m not in stamp_top_by_master:
                continue
            base_grid_top = _nearest_on_grid(stamp_top_by_master[m], grid_nm)
            needed = 0
            for row in top_h_demands:
                req_space = max(row.min_space_nm, row.wide_space_nm)
                for atom in atoms_by_master_layer.get((m, row.layer_key), ()):
                    if atom.net == row.net:
                        continue
                    target = atom.bbox_nm[3] + req_space + row.width_nm + row.offset_nm
                    target_grid = -((-target) // grid_nm) * grid_nm
                    needed = max(needed, target_grid - base_grid_top)
                for d_master, domain in _lower_access_domains(atoms, row):
                    if d_master == m and _domain_nonempty(domain):
                        target = domain[1] + max(1, row.access_clearance_nm) + row.width_nm + row.offset_nm
                        target_grid = -((-target) // grid_nm) * grid_nm
                        needed = max(needed, target_grid - base_grid_top)
            deficits[m] = needed
        if deficits and all(val > 0 for val in deficits.values()):
            top_margin_nm = max(deficits.values())
    return EdgeRailReservationIR(source_sha256=hashlib.sha256(source_ast.encode('utf-8')).hexdigest(),
        topology_sha256=_topology_digest(bridged_intent), pdk_id=tech.pdk_id,
        pdk_provenance_sha256=pdk_provenance_digest(tech), manufacturing_grid_nm=grid_nm,
        masters=tuple((name,box[2]-box[0],box[3]-box[1]) for name,box in sorted(solver_blocks)),
        demands=demands, stamps=stamps, atoms=atoms, native_feol_pairs=feol_pairs,
        right_margin_nm=right_margin_nm, top_margin_nm=top_margin_nm)


def verify_final_edge_rail_geometry(*, certificate, geometry, placement_um, layout_model,
                                   demands, pins_by_net, tech, cell_def, bridged_intent,
                                   source_pins_by_net=None, intent=None, row_program=None,
                                   row_materialization=None, row_final_certificate=None, native_geometry=None):
    """Compare the complete prepared pre-rail model with the retained native IR."""
    from clay.layout.place.edge_rail_reservation import EdgeRailReservationIR
    from clay.layout.route.route_cell import _canonical_rail_source_ast
    from clay.layout.route.resource_world import pdk_provenance_digest
    ir = getattr(certificate, 'edge_rail_reservation', None)
    if ir is None:
        return
    _require(type(ir) is EdgeRailReservationIR, 'final reservation has a foreign type')
    ir.payload()
    source_ast, _, _ = _canonical_rail_source_ast(cell_def)
    _require(ir.source_sha256 == hashlib.sha256(source_ast.encode('utf-8')).hexdigest()
             and ir.topology_sha256 == _topology_digest(bridged_intent)
             and ir.pdk_provenance_sha256 == pdk_provenance_digest(tech), 'final source/topology/PDK differs')
    blocks = dict(certificate.blocks)
    positions = {name:tuple(box[:2]) for name,box in blocks.items()}
    dims = {name:(box[2]-box[0],box[3]-box[1]) for name,box in blocks.items()}
    fields = ('demand_id','declaration_ref','net','layer_n','layer_key','actual_axis','edge','width_nm')
    _require(sorted(tuple(getattr(row,f) for f in fields) for row in demands) ==
             sorted(tuple(getattr(row,f) for f in fields) for row in ir.demands), 'final rail declarations differ')
    _require(all(row.wide_space_nm == _guaranteed_rectangular_rail_wide_space(
        tech, row, ir.stamps, ir.manufacturing_grid_nm) for row in ir.demands),
        'final wide rail PDK spacing differs')
    replacement = None
    if set(geometry) != {row.stamp_id for row in ir.stamps}:
        # Source M1 is globally attributed: movement can resolve an initially
        # ambiguous conductor. Recompile the actual complete model from the
        # same canonical source consumer before any monotone None-to-net join.
        from clay.layout.model import build_layout_model
        canonical_model=build_layout_model(bridged_intent,dict(placement_um),dict(geometry),tech=tech,
            pins_by_net=pins_by_net,include_upper_artwork=True,strict_device_m1_provenance=True,
            include_device_projection=False)
        conductor_identity=lambda row:(row.source_stamp,tuple(row.layer),row.net,row.kind,_conductor_box(row))
        _require(Counter(map(conductor_identity,canonical_model.conductors))==
                 Counter(map(conductor_identity,layout_model.conductors)),
                 'actual shared guard model differs from independent canonical source attribution')
        from .shared_guard_edge_rail import replay_actual_shared_guard_rails
        replacement = replay_actual_shared_guard_rails(certificate=certificate,ir=ir,
            geometry=geometry,placement_um=placement_um,intent=intent,layout_model=layout_model)
    replaced = set() if replacement is None else set(replacement['replaced_private_guard_stamps'])
    aliases = {} if replacement is None else replacement['unchanged_stamp_aliases']
    for stamp in ir.stamps:
        if stamp.stamp_id in replaced:
            continue
        actual_name = aliases.get(stamp.stamp_id,stamp.stamp_id)
        box, sha = _stamp_geometry(geometry[actual_name], placement_um[actual_name])
        _require(box == _shift(stamp.bbox_nm, positions[stamp.master_id]) and sha == stamp.geometry_sha256,
                 'final stamp geometry differs: '+stamp.stamp_id)
    if getattr(ir, 'native_feol_pairs', ()) and replacement is None:
        from .native_feol_reservation import compile_native_feol_pairs
        current_feol = compile_native_feol_pairs(
            tech=tech, stamps=ir.stamps, geometry=geometry, placement_um=placement_um)
        if tech.pdk_id == 'sky130' and native_geometry is not None and row_program is not None:
            from .native_feol_reservation import compile_sky130_row_feol_pairs, compile_sky130_group_row_spacing
            current_feol += compile_sky130_row_feol_pairs(bundle=native_geometry, cell=cell_def,
                stamps=ir.stamps, geometry=geometry, placement_um=placement_um,
                row_program=row_program, row_materialization=row_materialization, tech=tech)
            current_feol += compile_sky130_group_row_spacing(intent=intent, bundle=native_geometry, cell=cell_def,
                stamps=ir.stamps, geometry=geometry, placement_um=placement_um,
                row_program=row_program, row_materialization=row_materialization, tech=tech)
        sorted_current_feol = tuple(sorted(current_feol, key=lambda row: (
            row.left_stamp, row.right_stamp, row.rule_id, row.left_kind, row.right_kind)))
        if native_geometry is not None and getattr(native_geometry, '_cache_origin_output_dir', None):
            from pathlib import Path
            def _feol_key(row):
                return (
                    row.left_stamp, row.right_stamp, row.left_master, row.right_master,
                    row.left_kind, row.right_kind, row.left_layers, row.right_layers,
                    row.left_rectangles_nm, row.right_rectangles_nm, row.gap_nm,
                    row.rule_id, Path(row.rule_source).name, row.rule_sha256,
                )
            _require(tuple(map(_feol_key, ir.native_feol_pairs)) == tuple(map(_feol_key, sorted_current_feol)),
                     'final native FEOL domain/rule inventory differs')
        else:
            _require(ir.native_feol_pairs == sorted_current_feol,
                     'final native FEOL domain/rule inventory differs')
    layers = {key for row in ir.demands for key in (row.layer_key, *row.access_layer_keys)}
    expected = Counter((aliases.get(row.stamp_id,row.stamp_id),row.layer_key,row.net,row.kind,_shift(row.bbox_nm,positions[row.master_id]))
                       for row in ir.atoms if row.stamp_id not in replaced)
    if replacement is not None:
        for row in replacement['ownership_refinements']:
            old=(row['stamp'],tuple(row['layer_key']),None,row['kind'],tuple(row['bbox_nm']))
            new=(row['stamp'],tuple(row['layer_key']),row['actual_net'],row['kind'],tuple(row['bbox_nm']))
            expected[old]-=row['count'];expected[new]+=row['count']
        expected=+expected
    added = set() if replacement is None else set(replacement['actual_shared_guard_stamps'])
    actual = Counter((row.source_stamp,tuple(row.layer),row.net,row.kind,_conductor_box(row))
                     for row in layout_model.conductors if tuple(row.layer) in layers and row.kind != 'device_projection'
                     and row.source_stamp not in added)
    if actual != expected:
        monotone_ok = False
        if replacement is None:
            from collections import defaultdict as _defaultdict

            exp_by_geom = _defaultdict(Counter)
            for (st_id, lk, net_v, knd, bx), cnt in expected.items():
                exp_by_geom[(st_id, lk, knd, bx)][net_v] += cnt
            act_by_geom = _defaultdict(Counter)
            for (st_id, lk, net_v, knd, bx), cnt in actual.items():
                act_by_geom[(st_id, lk, knd, bx)][net_v] += cnt
            if set(act_by_geom) == set(exp_by_geom):
                all_monotone = True
                for gkey, old_cnts in exp_by_geom.items():
                    new_cnts = act_by_geom[gkey]
                    if sum(old_cnts.values()) != sum(new_cnts.values()):
                        all_monotone = False
                        break
                    for net_v, cnt in old_cnts.items():
                        if net_v is not None and new_cnts[net_v] < cnt:
                            all_monotone = False
                            break
                    if not all_monotone:
                        break
                    if sum((new_cnts - old_cnts).values()) > old_cnts[None]:
                        all_monotone = False
                        break
                if (
                    all_monotone
                    and bridged_intent is not None
                    and pins_by_net is not None
                    and hasattr(tech, "layer_names")
                ):
                    # Source M1 is globally attributed: separation during
                    # compaction can resolve an initially ambiguous None atom.
                    # Recompile from canonical build_layout_model before accepting.
                    from clay.layout.model import build_layout_model

                    canonical_model = build_layout_model(
                        bridged_intent,
                        dict(placement_um),
                        dict(geometry),
                        tech=tech,
                        pins_by_net=pins_by_net,
                        include_upper_artwork=True,
                        strict_device_m1_provenance=True,
                        include_device_projection=False,
                    )
                    conductor_identity = lambda row: (
                        row.source_stamp,
                        tuple(row.layer),
                        row.net,
                        row.kind,
                        _conductor_box(row),
                    )
                    if Counter(
                        map(conductor_identity, canonical_model.conductors)
                    ) == Counter(map(conductor_identity, layout_model.conductors)):
                        monotone_ok = True
        if not monotone_ok:
            missing = sorted((expected - actual).items(), key=repr)
            extra = sorted((actual - expected).items(), key=repr)
            raise ValueError('edge rail geometry final complete conductor ownership/artwork inventory differs'
                + f'; missing_count={sum((expected - actual).values())}, missing={missing[:12]!r}'
                + f'; extra_count={sum((actual - expected).values())}, extra={extra[:12]!r}')
    source_pins = pins_by_net if source_pins_by_net is None else source_pins_by_net
    counts = {net: len(pins) for net, pins in source_pins.items()}
    if (any(row.joint_component_ids for row in ir.demands) or
            any(getattr(stamp, 'joint_row_stamp_binding', None) is not None for stamp in geometry.values())):
        from clay.layout.joint_placement_terminals import build_placement_terminal_face
        _require(row_program is not None and row_materialization is not None and row_final_certificate is not None,
                 'joint terminal final row context is missing')
        face = build_placement_terminal_face(tech=tech, geometry=dict(geometry), placement_um=dict(placement_um),
            row_program=row_program, row_materialization=row_materialization, row_final_certificate=row_final_certificate,
            intent=intent, source_cell=cell_def)
        counts = face.terminal_counts(source_pins)
        _require(all(row.joint_component_ids == tuple(sorted(identity for item in face.joint_faces
                     for net, identity in item.demand_ids if net == row.net)) for row in ir.demands),
                 'final joint AND identities differ')
    _require(all(counts.get(row.net, 0) == row.terminal_count for row in ir.demands),
             'final rail source terminal denominator differs')
    _require(all(not row.terminal_count or bool(pins_by_net.get(row.net)) for row in ir.demands),
             'final rail terminal availability differs')
    _require(ir.replay(dims=dims, positions=positions,
        physical_bbox_nm=tuple(certificate.bbox_nm) if replacement is None else None),
             'final edge stripe violates the original physical reservation')
    return replacement
