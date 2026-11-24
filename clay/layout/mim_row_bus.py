"""Explicit local MIM row buses using existing terminal nodes and PDK rules.

Units, their pitch, their transforms and their artwork are supplied by the
existing bank tiler. Only horizontal row connections and exterior vertical
buses are added here. No via is added. Every original unit-role node, all added
metal and local width/spacing checks are replayed before returning geometry.
This bounded construction recipe is not source admission or routing signoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import klayout.db as kdb

from clay.layout.backend import LayerPolygon, Polygon
from clay.layout.bank_interconnect import _node_for_box, _region
from clay.layout.device_terminal import extract_device_nodes
from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm


def _require(condition, message):
    if not condition:
        raise ValueError("MIM row-bus " + message)


@dataclass(frozen=True, slots=True)
class MimRowBusPlan:
    p_side: str = "left"
    n_side: str = "right"

    def __post_init__(self):
        _require(all(type(side) is str and side in ("left", "right")
                     for side in (self.p_side, self.n_side)), "sides must be exact left/right choices")
        _require(self.p_side != self.n_side, "requires opposite exterior sides")

    def canonical_data(self):
        self.__post_init__()
        return {"schema": "clay.mim-row-bus-plan/v1", "p_side": self.p_side, "n_side": self.n_side}


def _rules(tech, layers):
    grid = pdk_manufacturing_grid_nm(tech)
    _require(type(grid) is int and grid > 0, "requires a positive manufacturing grid")
    names = {}
    for level in range(1, 9):
        try:
            names[tuple(tech.metal_key(level))] = f"metal{level}"
        except KeyError:
            continue
    result = {}
    for layer in layers:
        _require(layer in names, "terminal layer is not current PDK routing metal")
        width, space = tech.min_width(names[layer]), tech.min_space(names[layer])
        _require(all(type(value) is int and value > 0 for value in (width, space)),
                 "requires complete positive routing width and spacing rules")
        result[layer] = {"grid_nm": grid, "minimum_width_nm": width,
                         "minimum_spacing_nm": space}
    return result


def _pad_inventory(pads, unit_count):
    _require(type(unit_count) is int and unit_count > 1, "requires exact unit multiplicity greater than one")
    expected = {(index, role) for index in range(unit_count) for role in ("p", "n")}
    keys = []
    for row in pads:
        _require(type(row) is dict and set(row) == {"unit", "role", "net", "layer", "bbox_nm", "row", "column"},
                 "requires complete native unit-role pad records")
        _require(all(type(row[key]) is int and row[key] >= 0 for key in ("unit", "row", "column")),
                 "unit, row and column must be nonnegative exact integers")
        _require(type(row["role"]) is str and row["role"] in ("p", "n"), "pad role is unsupported")
        _require(type(row["net"]) is str and bool(row["net"]), "pad requires its original net")
        layer, box = row["layer"], row["bbox_nm"]
        _require(type(layer) is tuple and len(layer) == 2 and all(type(value) is int for value in layer),
                 "pad layer must be an exact integer pair")
        _require(type(box) is tuple and len(box) == 4 and all(type(value) is int for value in box)
                 and box[0] < box[2] and box[1] < box[3], "pad must be a nonempty exact integer box")
        keys.append((row["unit"], row["role"]))
    _require(len(keys) == len(expected) and set(keys) == expected, "unit-role denominator differs")
    for role in ("p", "n"):
        records = [row for row in pads if row["role"] == role]
        _require(len({row["layer"] for row in records}) == len({row["net"] for row in records}) == 1,
                 "same-role pads require one original routing layer and net")


def _artwork(polygons):
    return {layer: _region(polygons, layer)
            for layer in sorted({(polygon.layer, polygon.datatype) for polygon in polygons})}


def _nodes(polygons, artwork, tech, spec):
    # Intent imports the plan while raw-master records may still be loading.
    from clay.layout.mim_unit_materializer import _routing_masks

    native = extract_device_nodes(SimpleNamespace(polygons=polygons), tech=tech, mim_spec=spec)
    return tuple(_routing_masks(node, artwork, tech, spec) for node in native)


def _connections(records, side, extent, rules):
    grid, minimum = rules["grid_nm"], rules["minimum_width_nm"]
    down = lambda value: (value // grid) * grid
    up = lambda value: ((value + grid - 1) // grid) * grid
    width, space = up(minimum), up(rules["minimum_spacing_nm"])
    if side == "left":
        bus_x1 = down(extent.left - space)
        bus_x0 = bus_x1 - width
    else:
        bus_x0 = up(extent.right + space)
        bus_x1 = bus_x0 + width
    bands, rectangles = [], []
    for row_index in sorted({row["row"] for row in records}):
        boxes = [row["bbox_nm"] for row in records if row["row"] == row_index]
        low, high = max(box[1] for box in boxes), min(box[3] for box in boxes)
        first, last = up(low), down(high - width)
        _require(first <= last and all(box[2] - box[0] >= width for box in boxes),
                 "row pads have no common PDK-width routing track")
        y0 = down((first + last) // 2)
        y1 = y0 + width
        x0 = down(min(box[0] for box in boxes))
        x1 = up(max(box[2] for box in boxes))
        rectangles.append((min(x0, bus_x0), y0, max(x1, bus_x1), y1))
        bands.append((y0, y1))
    rectangles.append((bus_x0, min(y0 for y0, _y1 in bands),
                       bus_x1, max(y1 for _y0, y1 in bands)))
    return rectangles


def _check_foreign_spacing(added, original_nodes, original_roles, rules):
    violations, overlap = 0, 0
    for role, layers in added.items():
        for layer, region in layers.items():
            foreign = kdb.Region()
            for index, node in enumerate(original_nodes):
                if original_roles[index] != role:
                    foreign += node.get(layer, kdb.Region())
            overlap += (region & foreign).area()
            violations += region.separation_check(foreign, rules[layer]["minimum_spacing_nm"]).size()
    _require(not violations and not overlap, "new metal overlaps or violates spacing to a foreign node")
    return {"added_foreign_spacing_violations": violations, "added_foreign_overlap_area_nm2": overlap}


def _verify_final(polygons, original, pads, added, rules, tech, spec):
    artwork = _artwork(polygons)
    nodes = _nodes(polygons, artwork, tech, spec)
    _require(len(nodes) == 2, "new metal must preserve exactly two plate nodes, including existing via connections")
    role_nodes, checked = {"p": set(), "n": set()}, []
    for row in pads:
        node = _node_for_box(nodes, row["layer"], row["bbox_nm"])
        role_nodes[row["role"]].add(node)
        checked.append({**row, "node": node})
    _require(all(len(ids) == 1 for ids in role_nodes.values()) and role_nodes["p"] != role_nodes["n"],
             "all original unit-role pads must join their distinct plate nodes")
    checks = {"new_metal_outside_original_role_node_area_nm2": 0,
              "whole_layer_min_width_violations": 0, "whole_layer_min_space_violations": 0,
              "final_foreign_separation_violations": 0, "added_offgrid_vertices": 0}
    for role, layers in added.items():
        node = nodes[next(iter(role_nodes[role]))]
        for layer, region in layers.items():
            checks["new_metal_outside_original_role_node_area_nm2"] += (region - node.get(layer, kdb.Region())).area()
    for layer, rule in rules.items():
        checks["whole_layer_min_width_violations"] += artwork[layer].width_check(rule["minimum_width_nm"]).size()
        checks["whole_layer_min_space_violations"] += artwork[layer].space_check(rule["minimum_spacing_nm"]).size()
        checks["final_foreign_separation_violations"] += nodes[0].get(layer, kdb.Region()).separation_check(
            nodes[1].get(layer, kdb.Region()), rule["minimum_spacing_nm"]).size()
    for polygon in polygons[len(original):]:
        grid = rules[(polygon.layer, polygon.datatype)]["grid_nm"]
        checks["added_offgrid_vertices"] += sum(point.x % grid != 0 or point.y % grid != 0
                                                for point in polygon.polygon.vertices)
    _require(not any(checks.values()), f"local routing rules failed: {checks}")
    return checked, checks


def materialize_mim_row_bus(unit_polygons, *, role_pads, unit_count, tech, mim_spec, plan):
    """Return only added shapes and local replay facts; never modify unit copies."""
    if type(plan) is not MimRowBusPlan:
        raise TypeError("MIM row-bus requires an exact explicit MimRowBusPlan")
    plan = MimRowBusPlan(plan.p_side, plan.n_side)
    _pad_inventory(role_pads, unit_count)
    original = list(unit_polygons)
    artwork = _artwork(original)
    union = kdb.Region()
    for region in artwork.values():
        union += region
    original_nodes = _nodes(original, artwork, tech, mim_spec)
    _require(len(original_nodes) == 2 * unit_count, "original units must expose their complete independent plate-node denominator")
    original_roles = {}
    for row in role_pads:
        index = _node_for_box(original_nodes, row["layer"], row["bbox_nm"])
        _require(index not in original_roles, "original unit-role nodes are duplicated")
        original_roles[index] = row["role"]
    _require(len(original_roles) == len(original_nodes), "original node inventory is incomplete")
    rules = _rules(tech, {row["layer"] for row in role_pads})
    added, shapes = {}, []
    for role, side in (("p", plan.p_side), ("n", plan.n_side)):
        rows = [row for row in role_pads if row["role"] == role]
        layer = rows[0]["layer"]
        rectangles = _connections(rows, side, union.bbox(), rules[layer])
        role_shapes = [LayerPolygon(*layer, Polygon.rectangle(*box)) for box in rectangles]
        added[role] = {layer: _region(role_shapes, layer)}
        shapes.extend(role_shapes)
    checks = _check_foreign_spacing(added, original_nodes, original_roles, rules)
    checked, final_checks = _verify_final(original + shapes, original, role_pads, added, rules, tech, mim_spec)
    checks.update(final_checks)
    replay = _artwork(unit_polygons)
    _require(set(replay) == set(artwork) and all((replay[layer] ^ region).is_empty()
        for layer, region in artwork.items()), "original unit artwork changed during construction")
    return shapes, {"schema": "clay.mim-row-bus-materialization/v1", "plan": plan.canonical_data(),
        "frame": "unrotated_bank_nm", "node_count": 2, "unit_role_nodes": checked,
        "rules_nm": tuple({"layer": layer, **rule} for layer, rule in sorted(rules.items())),
        "local_checks": checks, "new_vias": 0, "full_drc_lvs": "NOT_RUN"}


__all__ = ["MimRowBusPlan", "materialize_mim_row_bus"]
