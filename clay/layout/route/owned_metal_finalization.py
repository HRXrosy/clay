"""Commit final assembled-metal repairs before route evidence is captured.

The emitter remains an independent observer.  This caller reuses its existing
PDK-derived close/width kernels on the actual placed artwork, then assigns only
new, uniquely owned material to the route.  Exact emission still rejects any
subsequent heal, fill, or snap.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import klayout.db as kdb

from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.placed_geometry import placed_point_nm


def _assembly(routed, geometry, placement):
    from .lvs_gate import _poly

    layout = kdb.Layout()
    layout.dbu = 0.001
    top = layout.create_cell("ROUTE_METAL_FINALIZATION")
    for name, stamp in sorted(geometry.items(), key=lambda item: str(item[0])):
        position = placement.get(name)
        if position is None:
            continue
        for key, rings in sorted(stamp.polys_by_layer.items()):
            for ring in rings:
                points = [kdb.Point(*placed_point_nm(float(x), float(y), position))
                          for x, y in ring]
                if len(points) < 3:
                    raise ValueError("metal finalization has degenerate placed artwork")
                top.shapes(layout.layer(*key)).insert(kdb.Polygon(points))
    # This is the emitter's route inventory: via landing metals and cut polygons
    # are already in NetRoute.wires.  NetRoute.vias carries graph records only;
    # emit_labelled_gds/net_layer_regions do not independently stamp them.
    for net, route in sorted(routed.nets.items()):
        for wire in route.wires:
            top.shapes(layout.layer(int(wire.layer), int(wire.datatype))).insert(_poly(wire))
    return layout, top


def _metal_rules(tech):
    result = {}
    for level in range(1, 32):
        try:
            key = tuple(tech.metal_key(level))
        except KeyError:
            break
        space = tech.min_space(f"metal{level}")
        if type(space) is not int or space <= 0:
            raise ValueError("metal finalization requires exact positive PDK spacing")
        result[key] = space
    return result


def _regions(layout, top, layers):
    return {key: kdb.Region(top.shapes(layout.layer(*key))).merged() for key in layers}


def _rectangles(region, key, grid_nm):
    """Lossless rectangular route-manifest projection, never a bounding-box fill."""
    cover = kdb.Region()
    result = []
    for shape in region.decompose_trapezoids():
        polygon = kdb.Polygon(shape.polygon)
        bbox = polygon.bbox()
        box = (bbox.left, bbox.bottom, bbox.right, bbox.top)
        rectangle = kdb.Region(kdb.Box(*box))
        if not (kdb.Region(polygon) ^ rectangle).is_empty():
            raise ValueError("owned assembled-metal repair is not exactly rectangular")
        if any(coordinate % grid_nm for coordinate in box):
            raise ValueError("owned assembled-metal repair is off manufacturing grid")
        cover += rectangle
        result.append(LayerPolygon(layer=key[0], datatype=key[1],
                                   polygon=Polygon.rectangle(*box)))
    if not (cover.merged() ^ region).is_empty():
        raise ValueError("owned assembled-metal rectangle projection changed geometry")
    return result


def _require_wide_preservation(before, after, tech):
    """Compare the actual assembled union, including inherited fixed edges."""
    from .wide_metal_spacing import wide_spacing_checks

    for level in range(1, 32):
        try:
            key = tuple(tech.metal_key(level))
        except KeyError:
            break
        if key not in before or (before[key] ^ after[key]).is_empty():
            continue
        old_checks = wide_spacing_checks(before[key], tech, f"metal{level}")
        new_checks = wide_spacing_checks(after[key], tech, f"metal{level}")
        if any(not (new.errors.polygons(0) - old.errors.polygons(0)).is_empty()
               for old, new in zip(old_checks, new_checks)):
            extra = (after[key] - before[key]).merged()
            raise ValueError("assembled-metal repair violates foreign clearance (wide metal): "
                             f"layer={key}, bbox={extra.bbox()}")


def _require_final_mim_geometry(layout, top, tech):
    """Check the completed composition, including native non-metal markers."""
    from .mim_metal_spacing import mim_metal_spacing_checks

    query = getattr(tech, "mim_metal_geometry_rules", None)
    rules = tuple(query()) if callable(query) else ()
    layers = {key for rule in rules for key in (rule.metal_layer, rule.marker_layer)}
    regions = _regions(layout, top, layers)
    for check in mim_metal_spacing_checks(regions, tech):
        if not check.errors.is_empty():
            raise ValueError("assembled-metal finalization violates MIM clearance: "
                             f"rule={check.rule_id}, layer={check.metal_layer}, "
                             f"marker={check.marker_layer}, errors={check.errors.size()}")


def _assembled_additions(routed, geometry, placement, pins_by_net, tech):
    from .emit_gds import net_label_points, net_layer_regions, pdk_manufacturing_grid_nm
    from .lvs_gate import (
        _close_same_net_m1_post_assembly, _close_same_net_upper_metal_extras,
        _exclude_existing_cell_material_from_upper_extras,
    )
    from .route_cell import _seeded_geometry_metal_regions

    rules = _metal_rules(tech)
    grid = pdk_manufacturing_grid_nm(tech)
    if type(grid) is not int or grid <= 0:
        raise ValueError("metal finalization lacks manufacturing-grid authority")
    layout, top = _assembly(routed, geometry, placement)
    before = _regions(layout, top, rules)
    upper = _exclude_existing_cell_material_from_upper_extras(
        _close_same_net_upper_metal_extras(routed, tech, min_gi=2), top, layout)
    for by_layer in upper.values():
        for key, region in by_layer.items():
            for polygon in region.each():
                top.shapes(layout.layer(*key)).insert(polygon)
    points = net_label_points(routed, pins_by_net, tech=tech,
                             geometry=geometry, placement=placement)
    _close_same_net_m1_post_assembly(top, layout, points, tech)
    after = _regions(layout, top, rules)
    differences = {key: (after[key] - before[key]).merged() for key in rules}
    if not any(not region.is_empty() for region in differences.values()):
        return {}, ()

    # Route-only upper closing cannot know every foreign native stamp. Judge
    # the actual assembled union before assigning any addition: a new wide core
    # or attached feature must not introduce a wider-rule error beside that
    # fixed material. The outer transaction rolls back every appended shape.
    _require_wide_preservation(before, after, tech)

    # The same native-terminal component reader used by route-cell owns stamp
    # conductors.  Routed conductors supply the other ownership seeds.  Ownership
    # propagates only over actual same-layer connected geometry, never by bbox.
    seeded = _seeded_geometry_metal_regions(geometry, placement, pins_by_net, set(rules))
    route_regions = net_layer_regions(routed.nets)
    known = {}
    for source in (seeded, route_regions):
        for net, by_layer in source.items():
            for key, region in by_layer.items():
                if key in rules:
                    known.setdefault(str(net), {}).setdefault(key, kdb.Region())
                    known[str(net)][key] += region
    additions = {}
    evidence = []
    for key, extra in differences.items():
        if extra.is_empty():
            continue
        # Include all new material in the connectivity replay: a repair joining
        # two differently owned islands must reject before either is committed.
        for polygon in after[key].each():
            island = kdb.Region(polygon)
            new = (island & extra).merged()
            if new.is_empty():
                continue
            owners = tuple(sorted(net for net, layers in known.items()
                                  if key in layers and not (layers[key] & island).is_empty()))
            if len(owners) != 1 or owners[0] not in routed.nets:
                raise ValueError("assembled-metal repair has non-unique route owner: "
                                 f"layer={key}, bbox={new.bbox()}, owners={owners}")
            net = owners[0]
            # Unlabelled/unseeded artwork outside this connected island remains
            # foreign.  Exact min-space is legal; every sub-rule overlap rejects.
            foreign = (after[key] - island).merged()
            if not (new & foreign.sized(rules[key])).is_empty():
                raise ValueError("assembled-metal repair violates foreign clearance: "
                                 f"net={net!r}, layer={key}, bbox={new.bbox()}")
            rectangles = _rectangles(new, key, grid)
            additions.setdefault(net, []).extend(rectangles)
            evidence.append({"net": net, "layer": key, "added_area_nm2": new.area(),
                             "added_rectangles": len(rectangles)})
    return additions, tuple(evidence)


def materialize_router_owned_metal_finalization(
    routed: Any, geometry: Mapping[str, Any], placement: Mapping[str, Any],
    pins_by_net: Mapping[str, Sequence[Any]], tech: Any,
) -> int:
    """Atomically close M1 terminals/EOL and assembled M1/upper-metal junctions.

    Apply the existing ordered kernels once, then require a zero-change replay
    of the entire composition.  Failure rolls back every appended route shape.
    No emitter transform, source geometry edit, budget change, or fixed-point
    iteration allowance is introduced.
    """
    from .lvs_gate import _materialize_router_owned_m1_finalization

    lengths = {}
    for net, route in routed.nets.items():
        if type(route.wires) is not list:
            raise TypeError("metal finalization requires mutable exact wire lists")
        lengths[net] = len(route.wires)
    missing = object()
    metadata = {name: getattr(routed, name, missing) for name in (
        "terminal_interface_materialization_count", "contact_line_end_materialization_count",
        "assembled_metal_materialization_count", "assembled_metal_materialization_evidence")}
    try:
        rules = _metal_rules(tech)
        initial_layout, initial_top = _assembly(routed, geometry, placement)
        original = _regions(initial_layout, initial_top, rules)
        m1_count = _materialize_router_owned_m1_finalization(
            routed, geometry, placement, pins_by_net, tech)
        terminal_count = getattr(routed, "terminal_interface_materialization_count", 0)
        line_end_count = getattr(routed, "contact_line_end_materialization_count", 0)
        additions, evidence = _assembled_additions(routed, geometry, placement, pins_by_net, tech)
        for net, shapes in additions.items():
            routed.nets[net].wires.extend(shapes)
        m1_replay = _materialize_router_owned_m1_finalization(
            routed, geometry, placement, pins_by_net, tech)
        assembled_replay, _ = _assembled_additions(routed, geometry, placement, pins_by_net, tech)
        if m1_replay or assembled_replay:
            raise ValueError("metal finalization composition is not idempotent: "
                             f"M1={m1_replay}, assembled={tuple(assembled_replay)}")
        final_layout, final_top = _assembly(routed, geometry, placement)
        _require_wide_preservation(original, _regions(final_layout, final_top, rules), tech)
        # Local bridges can repair an intermediate MIM gap, while remote
        # attachment can leave it illegal.  Judge the final literal union even
        # when the ordered kernels added nothing; idempotence is not legality.
        _require_final_mim_geometry(final_layout, final_top, tech)
    except Exception:
        for net, length in lengths.items():
            del routed.nets[net].wires[length:]
        for name, value in metadata.items():
            if value is missing:
                if hasattr(routed, name):
                    delattr(routed, name)
            else:
                setattr(routed, name, value)
        raise
    routed.terminal_interface_materialization_count = terminal_count
    routed.contact_line_end_materialization_count = line_end_count
    routed.assembled_metal_materialization_count = sum(len(v) for v in additions.values())
    routed.assembled_metal_materialization_evidence = evidence
    return m1_count + routed.assembled_metal_materialization_count
