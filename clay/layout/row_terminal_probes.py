"""Replay the complete source-backed row component denominator at consumers.

One distinct physical component is one required router endpoint.  Source rows
sharing that component remain associations, not additional disconnected demands.
This module returns ordinary Pins and grants no execution or signoff authority.
The retained row materialization determines whether the typed contract applies;
optional observations on an emitted stamp cannot turn that requirement off.
"""
from __future__ import annotations

from collections import Counter
import math

from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.placed_geometry import placed_point_nm
from clay.layout.route.engine import Pin
from clay.layout.route.terminal_identity import (
    PhysicalTerminalPinBinding, pin_fact, pin_source_aliases,
)


def _require(condition, message):
    if not condition:
        raise ValueError("row terminal " + message)


def _source_views(intent, source_cell):
    views = [intent]
    if source_cell is not None:
        from clay.layout.intent import layout_intent_from_cell_def
        views.append(layout_intent_from_cell_def(source_cell))
    return tuple(view for view in views if view is not None)


def _typed_stamp_names(geometry):
    # This is only a requirement detector.  These fields cannot validate a
    # component, invent ownership, or substitute for retained row context.
    return frozenset(name for name, stamp in geometry.items()
        if bool(getattr(stamp, "composite_members", ()))
        and (bool(getattr(stamp, "fixed_pcell", None))
             or getattr(stamp, 'joint_row_stamp_binding', None) is not None
             or getattr(stamp, "physical_terminal_inventory", None) is not None
             or getattr(stamp, "fixed_occurrence_sources", None) is not None))


def _retained_typed_geometry(materialization):
    return any(
        getattr(getattr(row, 'geometry', None), 'joint_row_binding', None) is not None or
        getattr(getattr(row, "geometry", None), "physical_terminal_inventory", None) is not None
        or getattr(getattr(row, "geometry", None), "fixed_occurrence_sources", None) is not None
        or getattr(getattr(row, "geometry", None), "fixed_pcell", None) is not None
        for row in (getattr(materialization, "realizations", ()) or ())
    )


def _require_source_scope(program, materialization, views, source_cell, tech):
    """Join complete declarations to the actual retained compiler output."""
    from clay.layout.place.row_calculus import RowStructureKind
    from clay.layout.place.row_program import _compile_mosaic_macro
    from clay.layout.spatial.intent_projection import replay_discrete_mosaic_inventory
    from clay.layout.spatial.mosaic_guest_slot import replay_mosaic_guest_slot_inventory

    projection = materialization.source_projection
    declarations = [row for view in views for row in (getattr(view, "discrete_mosaics", ()) or ())]
    slots = [row for view in views for row in (getattr(view, "mosaic_guest_slots", ()) or ())]
    boundaries = [row for view in views for row in (getattr(view, 'mosaic_boundary_joins', ()) or ())]
    if projection is not None:
        from clay.layout.place.row_source_projection import require_row_source_projection
        from clay.layout.place.rail_row_replay_scope import rail_row_source_program
        require_row_source_projection(projection, program=program)
        expected = (projection.projected_program if source_cell is None else
                    rail_row_source_program(projection, source_cell, tech,
                                            program=program, materialization=materialization))
        _require(expected.members == program.members
                 and expected.to_canonical_bytes() == program.to_canonical_bytes(),
                 "source projection does not bind the complete current row program")
        declarations.extend(projection.discrete_mosaics)
        slots.extend(projection.mosaic_guest_slots)
        boundaries.extend(projection.mosaic_boundary_joins)
    by_name = {}
    for declaration in declarations:
        previous = by_name.setdefault(declaration.mosaic_id, declaration)
        _require(previous == declaration, "source mosaic declarations disagree")
    mosaics = replay_discrete_mosaic_inventory(tuple(by_name.values()))
    slot_by_digest = {}
    for slot in slots:
        previous = slot_by_digest.setdefault(slot.source_mosaic_sha256, slot)
        _require(previous == slot, "source mosaic guest-slot declarations disagree")
    slot_by_digest = {slot.source_mosaic_sha256: slot
                     for slot in replay_mosaic_guest_slot_inventory(tuple(slot_by_digest.values()))}
    entries = {entry.group: entry for entry in program.entries
               if entry.structure_kind is RowStructureKind.DISCRETE_MOSAIC}
    _require(set(entries) == {mosaic.mosaic_id for mosaic in mosaics},
             "retained row program differs from the complete source mosaic scope")
    members = {member.device: member for member in program.members}
    from clay.layout.spatial.mosaic_boundary_joins import replay_mosaic_boundary_join_inventory
    boundary_by_mosaic = {}
    for boundary in boundaries:
        previous = boundary_by_mosaic.setdefault(boundary.source_mosaic_sha256, boundary)
        _require(previous == boundary, 'source mosaic boundary declarations disagree')
    replay_mosaic_boundary_join_inventory(tuple(boundary_by_mosaic.values()), mosaics=mosaics, members=members)
    for mosaic in mosaics:
        entry = entries[mosaic.mosaic_id]
        expected_macro = _compile_mosaic_macro(mosaic, by_name=members, tech=tech,
            guest_slot=slot_by_digest.get(mosaic.sha256), boundary_intent=boundary_by_mosaic.get(mosaic.sha256))
        _require(entry.active_devices == mosaic.members
                 and entry.macro.to_canonical_bytes() == expected_macro.to_canonical_bytes(),
                 "retained row mosaic differs from its complete source declaration")


def _actual_stamp_polygons(stamp, position):
    _require(isinstance(position, (tuple, list)) and len(position) == 2
             and all(type(value) in (int, float) and math.isfinite(value) for value in position),
             "stamp placement must be a finite coordinate pair")
    layers = getattr(stamp, "polys_by_layer", None)
    _require(type(layers) is dict and len(layers) <= 65_536, "stamp layer inventory is missing or malformed")
    polygon_count = 0
    vertex_count = 0
    for layer, rings in layers.items():
        _require(type(layer) is tuple and len(layer) == 2
                 and all(type(value) is int and 0 <= value <= 65_535 for value in layer),
                 "stamp layer key is malformed")
        _require(type(rings) in (list, tuple), "stamp polygon inventory is malformed")
        polygon_count += len(rings)
        _require(polygon_count <= 262_144, "stamp polygon inventory exceeds its bound")
        for ring in rings:
            _require(type(ring) in (list, tuple) and 3 <= len(ring) <= 4096,
                     "stamp polygon is malformed")
            vertex_count += len(ring)
            _require(vertex_count <= 1_048_576, "stamp vertex inventory exceeds its bound")
    polygons = []
    for layer, rings in sorted(layers.items()):
        for ring in rings:
            vertices = []
            for point in ring:
                _require(type(point) in (list, tuple) and len(point) == 2
                         and all(type(value) in (int, float) and math.isfinite(value) for value in point),
                         "stamp polygon coordinate is malformed")
                vertices.append(Point(*placed_point_nm(point[0], point[1], position)))
            polygons.append(LayerPolygon(layer[0], layer[1], Polygon(vertices)))
    _require(bool(polygons), "stamp artwork is empty")
    return tuple(polygons)


def _pin_key(pin):
    _require(type(pin) is Pin, "component pin must be an exact Pin")
    _require(type(pin.xy_nm) is tuple and len(pin.xy_nm) == 2
             and all(type(value) is int for value in pin.xy_nm),
             "component pin needs exact integer coordinates")
    _require(type(pin.xy_um) is tuple and len(pin.xy_um) == 2
             and all(type(value) in (int, float) and math.isfinite(value) for value in pin.xy_um),
             "component pin needs finite micron coordinates")
    _require(tuple(value / 1000.0 for value in pin.xy_nm) == pin.xy_um,
             "component pin coordinate currencies differ")
    _require(type(pin.layer) is tuple and len(pin.layer) == 2
             and all(type(value) is int for value in pin.layer), "component pin layer is malformed")
    _require(all(type(value) is str and bool(value) for value in (pin.net, pin.device, pin.terminal)),
             "component pin ownership is malformed")
    return pin_fact(pin)


def _replayed_row_pins(*, tech, geometry, placement_um, row_program=None,
                       row_materialization=None, row_final_certificate=None,
                       intent=None, source_cell=None, _require_final_certificate=True, _joint_faces=None):
    _require(type(geometry) is dict and type(placement_um) is dict,
             "actual geometry and placement must be dictionaries")
    views = _source_views(intent, source_cell)
    required_stamps = _typed_stamp_names(geometry)
    required = (bool(required_stamps) or _retained_typed_geometry(row_materialization)
                or getattr(row_materialization, "source_projection", None) is not None
                or any(getattr(view, "discrete_mosaics", ()) for view in views))
    if not required:
        # Existing legacy provenance validation retains ownership of its old
        # operands.  This additional gate must not invent typed obligations from
        # a partial legacy certificate or from ordinary common-centroid syntax.
        return (), frozenset()
    from clay.layout.generic_placement import RowFinalPlacementCertificate, _placed_artwork_sha256
    from clay.layout.place.row_program import (
        RowProgram, RowProgramMaterialization, replay_row_terminal_inventory,
    )
    from clay.layout.physical_terminal_inventory import replay_physical_terminal_inventory

    _require(type(row_program) is RowProgram and type(row_materialization) is RowProgramMaterialization,
             "retained row context is incomplete or untyped")
    if _require_final_certificate:
        _require(type(row_final_certificate) is RowFinalPlacementCertificate,
                 "retained row context is incomplete or untyped")
    from clay.layout.place.rail_row_replay_scope import require_rail_row_materialization
    require_rail_row_materialization(row_materialization, program=row_program, tech_id=tech.pdk_id)
    _require_source_scope(row_program, row_materialization, views, source_cell, tech)
    realizations = {row.entry_id: row for row in row_materialization.realizations}
    expected_ids = {entry.entry_id for entry in row_program.entries}
    _require(set(realizations) == expected_ids,
             "materialization must cover every row entry")
    bindings = {}
    if _require_final_certificate:
        row_final_certificate.__post_init__()
        _require(row_final_certificate.program_sha256 == row_program.sha256,
                 "final certificate names a different row program")
        bindings = {binding.entry_id: binding for binding in row_final_certificate.macro_bindings}
        _require(set(bindings) == expected_ids,
                 "final certificate and materialization must cover every row entry")
    pins = []
    typed_members = set()
    typed_stamps = set()
    for entry in row_program.entries:
        realization = realizations[entry.entry_id]
        binding = bindings.get(entry.entry_id)
        if _require_final_certificate:
            _require(binding.entry_sha256 == entry.sha256 and binding.owner == entry.owner
                     and binding.materialization_sha256 == row_materialization.certificate.sha256
                     and binding.consumed_devices == entry.consumed_devices,
                     "final row source/materialization binding differs")
        joint = getattr(realization.geometry, 'joint_row_binding', None) is not None
        if joint:
            _require(_joint_faces is not None, 'JOINT_ROUTING_ACCESS_UNSUPPORTED')
        else:
            inventory = replay_row_terminal_inventory(geometry=realization.geometry,
                consumed_devices=entry.consumed_devices, tech=tech)
            if inventory is None:
                continue
        typed_stamps.add(entry.group)
        typed_members.update(entry.consumed_devices)
        _require(entry.group in geometry and entry.group in placement_um,
                 "required physical row stamp is missing")
        stamp = geometry[entry.group]
        stamp_members = getattr(stamp, "composite_members", None)
        if stamp_members is not None:
            _require(tuple(sorted(stamp_members)) == entry.consumed_devices,
                     "actual row stamp source membership differs")
        polygons = _actual_stamp_polygons(stamp, placement_um[entry.group])
        local = realization.local_bbox_nm
        if _require_final_certificate:
            placed = binding.bbox_nm
            _require(_placed_artwork_sha256(list(polygons)) == binding.artwork_sha256,
                     "actual row stamp artwork differs from the final certificate")
        else:
            # This is the current observed rigid frame before final placement.
            # Its translation must replay every raw polygon against the retained
            # materialization; no stale final certificate authorizes this frame.
            xs = [v.x for p in polygons for v in p.polygon.vertices]
            ys = [v.y for p in polygons for v in p.polygon.vertices]
            placed = min(xs), min(ys), max(xs), max(ys)
        delta = placed[0] - local[0], placed[1] - local[1]
        _require((placed[2] - local[2], placed[3] - local[3]) == delta,
                 "actual row stamp is not a rigid translation")
        if joint:
            from clay.layout.joint_placement_terminals import project_joint_placement_face
            _joint_faces.append(project_joint_placement_face(native_geometry=realization.geometry, stamp=stamp,
                position=placement_um[entry.group], members=entry.consumed_devices, delta=delta, tech=tech,
                native_row_sha256=realization.geometry_sha256,
                source_members={member.device: member for member in row_program.members}))
            continue
        # Compose real stamp floats once, then undo only the certificate's exact
        # integer translation before replaying the legacy physical inventory.
        local_polygons = tuple(LayerPolygon(p.layer, p.datatype, Polygon([
            Point(v.x - delta[0], v.y - delta[1]) for v in p.polygon.vertices])) for p in polygons)
        replay_physical_terminal_inventory(geometry=local_polygons, sources=inventory.sources,
                                           inventory=inventory, tech=tech)
        sources = {source.source_id: source for source in inventory.sources}
        for component in inventory.components:
            source = sources[min(component.source_ids)]
            _require(source.member in entry.consumed_devices and source.net == component.net,
                     "component source ownership differs")
            point = tuple(value + delta[axis] for axis, value in enumerate(component.seed_xy_nm))
            terminal_binding = PhysicalTerminalPinBinding(
                stamp_id=entry.group, component_id=component.component_id,
                source_refs=tuple(sorted((ref, sources[ref].member, sources[ref].role)
                                         for ref in component.source_ids)))
            pins.append(Pin(net=component.net, device=source.member, terminal=source.role,
                layer=component.seed_layer, xy_nm=point,
                xy_um=tuple(value / 1000.0 for value in point),
                terminal_binding=terminal_binding))
    _require(required_stamps <= typed_stamps,
             "actual fixed row stamp lacks its complete retained terminal source inventory")
    ordered = tuple(sorted(pins, key=_pin_key))
    _require(len({_pin_key(pin) for pin in ordered}) == len(ordered),
             "distinct row components produced an ambiguous canonical pin")
    return ordered, frozenset(typed_members)


def build_row_terminal_pins(*, tech, geometry, placement_um, row_program=None,
                            row_materialization=None, row_final_certificate=None,
                            intent=None, source_cell=None):
    """Return one exact canonical Pin for every required typed row component."""
    pins, _members = _replayed_row_pins(tech=tech, geometry=geometry, placement_um=placement_um,
        row_program=row_program, row_materialization=row_materialization,
        row_final_certificate=row_final_certificate, intent=intent, source_cell=source_cell)
    return pins


def build_materialized_row_terminal_inputs(*, tech, geometry, placement_um, row_program=None,
                                          row_materialization=None, intent=None, source_cell=None):
    """Replay the current rigid frame and return pins plus all consumed members.

    This source/materialization projection is usable before final placement.
    It neither requires nor mints a final placement certificate.
    """
    return _replayed_row_pins(tech=tech, geometry=geometry, placement_um=placement_um,
        row_program=row_program, row_materialization=row_materialization,
        intent=intent, source_cell=source_cell, _require_final_certificate=False)


def build_materialized_row_terminal_pins(*, tech, geometry, placement_um, row_program=None,
                                        row_materialization=None, intent=None, source_cell=None):
    """Return all component pins in a freshly observed materialized row frame."""
    pins, _members = build_materialized_row_terminal_inputs(tech=tech, geometry=geometry,
        placement_um=placement_um, row_program=row_program, row_materialization=row_materialization,
        intent=intent, source_cell=source_cell)
    return pins


def require_row_terminal_probes(*, tech, geometry, placement_um, pins_by_net,
                                row_program=None, row_materialization=None,
                                row_final_certificate=None, intent=None, source_cell=None):
    """Reject omissions, duplicates and altered row pins using an independent denominator.

    Nonrow pins are untouched.  An empty return denotes no typed row obligations
    in the supplied scope; it is never a certificate of source completeness.
    """
    expected, members = _replayed_row_pins(tech=tech, geometry=geometry, placement_um=placement_um,
        row_program=row_program, row_materialization=row_materialization,
        row_final_certificate=row_final_certificate, intent=intent, source_cell=source_cell)
    _require(type(pins_by_net) is dict, "pin inventory must be a dictionary")
    actual = []
    for net, pins in pins_by_net.items():
        _require(type(pins) in (list, tuple), "pin net inventory must be a sequence")
        for pin in pins:
            if members and any(member in members for member, _role in pin_source_aliases(pin)):
                key = _pin_key(pin)
                _require(net == pin.net, "component pin is listed under a different net")
                actual.append(key)
    _require(Counter(actual) == Counter(_pin_key(pin) for pin in expected),
             "component probe pins do not cover the complete retained source inventory")
    return expected


def require_routing_row_terminal_probes(**operands):
    """Canonical consumers accept only a complete lossless joint lowering.

    The legacy Pin builder remains guarded. This new consumer independently
    replays the full AND denominator and rejects OR metal islands that cannot
    be represented by one actual connected routing component.
    """
    allow_fixed_multilayer = operands.pop('allow_fixed_multilayer', False)
    multilayer_engine = operands.pop('multilayer_engine', None)
    geometry = operands.get('geometry', {})
    materialization = operands.get('row_materialization')
    joint = (any(getattr(stamp, 'joint_row_stamp_binding', None) is not None
                 for stamp in geometry.values()) or
             any(getattr(row.geometry, 'joint_row_binding', None) is not None
                 for row in getattr(materialization, 'realizations', ())))
    if joint:
        from clay.layout.route.joint_component_pins import require_current_joint_component_pins
        return require_current_joint_component_pins(allow_fixed_multilayer=allow_fixed_multilayer, multilayer_engine=multilayer_engine, **operands)
    return require_row_terminal_probes(**operands)


__all__ = ["build_row_terminal_pins", "require_row_terminal_probes", "require_routing_row_terminal_probes",
           "build_materialized_row_terminal_inputs", "build_materialized_row_terminal_pins"]
