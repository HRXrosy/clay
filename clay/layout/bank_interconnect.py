"""Explicit internal MIM-bank wiring; original external access stays unchanged.

The first supported plan uses already connected M2 landings above M1/via1.
No vias are synthesized and no placement/reference geometry selects a layer.
Checks are local materialization checks, not a foundry DRC/LVS certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BankInterconnectPlan:
    metal_level: int = 2
    access_mode: str = "existing_connected_landing"

    def __post_init__(self):
        if type(self.metal_level) is not int:
            raise TypeError("bank interconnect metal_level must be an exact integer")
        if self.metal_level != 2:
            raise ValueError("existing bank interconnect currently supports metal2 via existing via1 only")
        if type(self.access_mode) is not str:
            raise TypeError("bank interconnect access_mode must be an exact string")
        if self.access_mode != "existing_connected_landing":
            raise ValueError("unsupported bank interconnect access mode")

    def canonical_data(self):
        return {"metal_level": self.metal_level, "access_mode": self.access_mode}


def replay_bank_interconnect_plan(value):
    if type(value) is BankInterconnectPlan:
        return BankInterconnectPlan(metal_level=value.metal_level, access_mode=value.access_mode)
    from clay.layout.mim_row_bus import MimRowBusPlan
    if type(value) is MimRowBusPlan:
        return MimRowBusPlan(p_side=value.p_side, n_side=value.n_side)
    raise TypeError("bank interconnect must be an exact supported typed plan")


def bank_interconnect_from_attributes(attrs: Mapping):
    layer_key, access_key = "bank_interconnect_layer", "bank_interconnect_access"
    present = (layer_key in attrs, access_key in attrs)
    if not any(present):
        return None
    if not all(present):
        raise ValueError("bank interconnect requires both layer and access declarations")
    level = attrs[layer_key]
    if type(level) is str:
        if level != "2":
            raise ValueError("bank interconnect layer must be canonical supported integer 2")
        level = 2
    return BankInterconnectPlan(metal_level=level, access_mode=attrs[access_key])


def native_quantized_via_box(*, center_nm, nominal_size_nm, grid_nm):
    """Native _cut half split followed by _append_rect floor/ceil grid snapping."""
    if (type(center_nm) is not tuple or len(center_nm) != 2
            or any(type(v) is not int for v in center_nm)):
        raise TypeError("via center must contain exactly two integers")
    if type(nominal_size_nm) is not int or type(grid_nm) is not int:
        raise TypeError("via size and manufacturing grid must be exact integers")
    if nominal_size_nm <= 0 or grid_nm <= 0:
        raise ValueError("via size and manufacturing grid must be positive")
    low = nominal_size_nm // 2
    high = nominal_size_nm - low
    cx, cy = center_nm
    down = lambda v: (v // grid_nm) * grid_nm
    up = lambda v: ((v + grid_nm - 1) // grid_nm) * grid_nm
    return down(cx-low), down(cy-low), up(cx+high), up(cy+high)


def _region(polygons, layer):
    import klayout.db as kdb
    result = kdb.Region()
    for polygon in polygons:
        if (polygon.layer, polygon.datatype) == layer:
            result.insert(kdb.Polygon([kdb.Point(v.x, v.y) for v in polygon.polygon.vertices]))
    result.merge()
    return result


def _node_for_box(nodes, layer, box):
    import klayout.db as kdb
    pad = kdb.Region(kdb.Box(*box))
    matches = [i for i, node in enumerate(nodes) if layer in node and (pad-node[layer]).is_empty()]
    if len(matches) != 1:
        raise ValueError("bank pad is not wholly on one extracted electrical node")
    selected = matches[0]
    if any(i != selected and layer in node and not (pad & node[layer]).is_empty()
           for i, node in enumerate(nodes)):
        raise ValueError("bank pad overlaps a foreign extracted electrical node")
    return selected


def _rules(tech):
    from clay.layout.passive_device_cell import _grid_nm
    value = lambda item: int(getattr(item, "value", item))
    via = tech.via_rule("via1")
    rules = {"grid_nm": _grid_nm(tech), "minimum_width_nm": int(tech.min_width("metal2")),
             "minimum_spacing_nm": int(tech.min_space("metal2")), "via_size_nm": value(via.size),
             "via_enclosure_below_nm": value(via.enclosure_below),
             "via_enclosure_above_nm": value(via.enclosure_above)}
    if any(type(v) is not int or v <= 0 for v in rules.values()):
        raise ValueError("bank interconnect requires complete positive grid/M2/via1 rules")
    return rules


def select_existing_connected_landings(unit, *, tech, mim_spec, plan):
    """Return internal pads only, after proving original M1 tips share their nodes.

    A unique rectangular, centered-via landing is the supported initial scope.
    An absent or ambiguous landing fails; no proximity or net-name guess is used.
    The unit and its terminal objects are never modified.
    """
    import klayout.db as kdb
    from clay.layout.device_terminal import extract_device_nodes
    replay_bank_interconnect_plan(plan)
    rules = _rules(tech)
    m1, m2, via1 = tuple(tech.metal_key(1)), tuple(tech.metal_key(2)), tuple(tech.via_key(1))
    nodes = extract_device_nodes(unit, tech=tech, mim_spec=mim_spec)
    if len(nodes) != 2 or set(unit.terminals) != {"p", "n"}:
        raise ValueError("bank unit requires exactly two plate nodes and p/n terminals")
    role_nodes = {role: _node_for_box(nodes, m1, unit.terminals[role].bbox) for role in ("p", "n")}
    if role_nodes["p"] == role_nodes["n"]:
        raise ValueError("bank unit original p/n tips share an electrical node")
    raw_m1, raw_m2, raw_via = (_region(unit.polygons, layer) for layer in (m1, m2, via1))
    selected = {}
    for role in ("p", "n"):
        node = nodes[role_nodes[role]]
        choices = []
        polygons = tuple(node[m2].merged().each_merged()) if m2 in node else ()
        for polygon in polygons:
            box = polygon.bbox()
            bbox = (box.left, box.bottom, box.right, box.top)
            if (polygon.area() != box.area()
                    or min(box.width(), box.height()) < rules["minimum_width_nm"]
                    or any(v % rules["grid_nm"] for v in bbox)):
                continue
            pad = kdb.Region(box)
            if not (pad-raw_m2).is_empty():
                continue
            center = ((box.left+box.right)//2, (box.bottom+box.top)//2)
            expected = native_quantized_via_box(center_nm=center,
                nominal_size_nm=rules["via_size_nm"], grid_nm=rules["grid_nm"])
            cuts = []
            for cut in raw_via.each_merged():
                cb = cut.bbox()
                actual = (cb.left, cb.bottom, cb.right, cb.top)
                if cut.area() != cb.area() or actual != expected:
                    continue
                below, above = rules["via_enclosure_below_nm"], rules["via_enclosure_above_nm"]
                eb = kdb.Region(kdb.Box(cb.left-below, cb.bottom-below, cb.right+below, cb.top+below))
                ea = kdb.Region(kdb.Box(cb.left-above, cb.bottom-above, cb.right+above, cb.top+above))
                if (ea-pad).is_empty() and (eb-node[m1]).is_empty() and (eb-raw_m1).is_empty():
                    cuts.append(actual)
            if cuts and _node_for_box(nodes, m2, bbox) == role_nodes[role]:
                choices.append({"layer": m2, "bbox_nm": bbox, "existing_via_cuts_nm": tuple(cuts),
                                "original_m1_bbox_nm": tuple(unit.terminals[role].bbox),
                                "net": unit.terminals[role].net})
        if len(choices) != 1:
            raise ValueError(f"bank {role} existing same-plate M2/via1 landing absent or ambiguous: {len(choices)}")
        selected[role] = choices[0]
    return selected


def verify_bank_interconnect(bank, *, tech, mim_spec, plan, role_pads):
    """Fresh final-geometry proof for every original M1 tip and chosen M2 landing."""
    from clay.layout.device_terminal import extract_device_nodes
    replay_bank_interconnect_plan(plan)
    rules = _rules(tech)
    m1, m2 = tuple(tech.metal_key(1)), tuple(tech.metal_key(2))
    nodes = extract_device_nodes(bank, tech=tech, mim_spec=mim_spec)
    if len(nodes) != 2:
        raise ValueError("planned MIM bank requires exactly two extracted plate nodes")
    expected = {(i, role) for i in range(bank.finger_count) for role in ("p", "n")}
    actual = [(row["unit"], row["role"]) for row in role_pads]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("planned MIM bank unit-role coverage differs")
    by_role = {"p": set(), "n": set()}
    rows = []
    for row in role_pads:
        tip = _node_for_box(nodes, m1, row["original_m1_bbox_nm"])
        landing = _node_for_box(nodes, m2, row["bbox_nm"])
        if tip != landing:
            raise ValueError("planned MIM bank original tip and internal landing have different nodes")
        by_role[row["role"]].add(tip)
        rows.append({**row, "node": tip})
    if len(by_role["p"]) != 1 or len(by_role["n"]) != 1 or by_role["p"] == by_role["n"]:
        raise ValueError("planned MIM bank must join all p and all n on distinct plate nodes")
    all_m2 = _region(bank.polygons, m2)
    p_m2 = nodes[next(iter(by_role["p"]))][m2].merged()
    n_m2 = nodes[next(iter(by_role["n"]))][m2].merged()
    if not ((p_m2+n_m2) ^ all_m2).is_empty():
        raise ValueError("planned MIM bank contains M2 outside its two plate nodes")
    checks = {"whole_m2_min_width_violations": all_m2.width_check(rules["minimum_width_nm"]).size(),
              "whole_m2_min_space_violations": all_m2.space_check(rules["minimum_spacing_nm"]).size(),
              "p_n_m2_separation_violations": p_m2.separation_check(n_m2, rules["minimum_spacing_nm"]).size(),
              "p_n_m2_overlap_area_nm2": (p_m2 & n_m2).area(),
              "m2_offgrid_vertex_count": sum(int(v.x % rules["grid_nm"] != 0 or v.y % rules["grid_nm"] != 0)
                  for p in bank.polygons if (p.layer, p.datatype) == m2 for v in p.polygon.vertices)}
    if any(checks.values()):
        raise ValueError(f"planned MIM bank local M2 checks failed: {checks}")
    return {"schema": "clay.bank-interconnect-materialization/v1", "plan": plan.canonical_data(),
            "frame": "unrotated_bank_nm", "node_count": len(nodes), "unit_role_nodes": rows,
            "rules_nm": rules, "layers": {"external": m1, "internal": m2, "existing_via": tuple(tech.via_key(1))},
            "local_checks": checks, "new_vias": 0, "full_drc_lvs": "NOT_RUN"}
