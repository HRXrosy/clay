"""Named-net clearance during search and independent emitted-metal measurement.

Distances are Euclidean edge-to-edge distances between same-layer metal boxes,
in integer nanometres. Via landing pads are metal; cuts are not. This geometric
requirement supplements the PDK rules and does not claim a coupling-capacitance
bound or create a shield conductor.
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_CEILING
from numbers import Rational, Real

from clay.layout.route.constraints import Spacing, is_hard


def spacing_nm(constraint: Spacing, nets) -> int | None:
    """Validate a declaration without silently rounding its minimum down."""
    value = constraint.min_um
    if (
        type(constraint.net_a) is not str
        or type(constraint.net_b) is not str
        or constraint.net_a == constraint.net_b
        or constraint.net_a not in nets
        or constraint.net_b not in nets
        or isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
        or not math.isfinite(value * 1000)
    ):
        return None
    if isinstance(value, Rational):
        return math.ceil(value * 1000)
    return int((Decimal(str(value)) * 1000).to_integral_value(rounding=ROUND_CEILING))


def compile_spacing_rules(constraints, nets) -> dict[str, dict[str, int]]:
    rules: dict[str, dict[str, int]] = {}
    for constraint in constraints:
        if not isinstance(constraint, Spacing) or not is_hard(constraint):
            continue
        distance = spacing_nm(constraint, nets)
        if distance is None:
            continue
        a, b = constraint.net_a, constraint.net_b
        rules.setdefault(a, {})[b] = max(rules.get(a, {}).get(b, 0), distance)
        rules.setdefault(b, {})[a] = max(rules.get(b, {}).get(a, 0), distance)
    return rules


def _gap_squared(left, right) -> int:
    dx = max(left[0] - right[2], right[0] - left[2], 0)
    dy = max(left[1] - right[3], right[1] - left[3], 0)
    return dx * dx + dy * dy


def fixed_spacing_conflicts(pins_by_net, constraints, obstacles, tech, rails=None):
    """Identify declared gaps already violated by immovable terminal/rail boxes.

    Missing shapes cannot establish a conflict. This only proves infeasibility
    for the supplied fixed rectangular geometry, not for alternative placements.
    """
    from clay.layout.route.engine import NetRoute, RoutedCell
    fixed = RoutedCell(nets={net: NetRoute(net, connected=True) for net in pins_by_net})
    reports = check_spacing_geometry(fixed, constraints, obstacles, tech, rails)
    return [(c, report) for c, report in zip(
        (c for c in constraints if isinstance(c, Spacing)), reports)
        if is_hard(c) and report["required_gap_nm"] is not None
        and report["minimum_gap_squared_nm2"] is not None
        and report["minimum_gap_squared_nm2"] < report["required_gap_nm"] ** 2]


def spacing_blockers(rules, net, box, fixed_index, metal_index, abstract_rects=()):
    """Query only local shapes, including clearances beyond the PDK halo."""
    partners = rules.get(net, {})
    margin = max(partners.values(), default=0)
    if margin == 0:
        return
    query = (box[0] - margin, box[1] - margin,
             box[2] + margin, box[3] + margin)
    seen = set()
    for kind, index in (("fixed_metal", fixed_index), ("routed_metal", metal_index)):
        if index is None:
            continue
        for owner, rect in index.query(query):
            distance = partners.get(owner, 0)
            if not distance or (kind == "fixed_metal" and rect in abstract_rects):
                continue
            if _gap_squared(box, rect) < distance * distance:
                item = (owner, tuple(rect), distance, kind)
                if item not in seen:
                    seen.add(item)
                    yield item


def check_spacing_geometry(routed, constraints, obstacles, tech, rails=None):
    """Measure returned metal plus fixed terminals/rails, without search caches.

    Bounds of nonrectangular metal are conservative: a pass proves clearance,
    while a failure may require a more exact polygon check. The gridless wire
    and via producers supply rectangles. Disjoint metal-layer sets satisfy the
    same-layer requirement; missing/unconnected or metal-free nets do not.
    """
    from clay.layout.model import ROUTING_ABSTRACTION_KINDS

    declarations = [c for c in constraints if isinstance(c, Spacing)]
    routed.spacing_constraints = tuple(declarations)
    if not declarations:
        return []
    nets = routed.nets
    requested = {net for c in declarations for net in (c.net_a, c.net_b)
                 if isinstance(net, str)}
    shapes = {net: {} for net in requested}

    def add(net, layer, box):
        if net not in shapes:
            return
        gi = tech.resolve_layer_to_gi(layer)
        if gi is None:
            return
        shapes[net].setdefault(gi, set()).add(tuple(box))

    for net in requested:
        route = nets.get(net)
        if route is None:
            continue
        for wire in route.wires:
            vertices = wire.polygon.vertices
            if vertices:
                xs = [int(p.x) for p in vertices]
                ys = [int(p.y) for p in vertices]
                add(net, (wire.layer, wire.datatype),
                    (min(xs), min(ys), max(xs), max(ys)))
    for obstacle in obstacles:
        if obstacle.kind in ROUTING_ABSTRACTION_KINDS:
            continue
        coords = [round(v * 1000) for v in
                  (obstacle.x0, obstacle.y0, obstacle.x1, obstacle.y1)]
        add(obstacle.owner_net, obstacle.layer,
            (min(coords[0], coords[2]), min(coords[1], coords[3]),
             max(coords[0], coords[2]), max(coords[1], coords[3])))
    for net, entries in (rails or {}).items():
        for layer, bounds_um in entries:
            add(net, layer, tuple(round(v * 1000) for v in bounds_um))

    reports = []
    for c in declarations:
        required = spacing_nm(c, nets)
        a = shapes.get(c.net_a, {}) if isinstance(c.net_a, str) else {}
        b = shapes.get(c.net_b, {}) if isinstance(c.net_b, str) else {}
        connected = required is not None and all(nets[net].connected
                                                for net in (c.net_a, c.net_b))
        closest = None
        witness = None
        for gi in sorted(a.keys() & b.keys()):
            for left in sorted(a[gi]):
                for right in sorted(b[gi]):
                    distance_squared = _gap_squared(left, right)
                    if closest is None or distance_squared < closest:
                        closest = distance_squared
                        witness = (gi, left, right)
        report = {
            "net_a": c.net_a, "net_b": c.net_b,
            "required_gap_nm": required,
            "minimum_gap_nm": None if closest is None else math.sqrt(closest),
            "minimum_gap_squared_nm2": closest,
            "common_layer_count": len(a.keys() & b.keys()),
            "closest_boxes": witness,
            "metric": "same_layer_metal_bbox_edge_distance",
            "satisfied": bool(required is not None and connected and a and b
                              and (closest is None or closest >= required * required)),
        }
        reports.append(report)
        if not report["satisfied"] and is_hard(c) and c not in routed.unsatisfied:
            routed.unsatisfied.append(c)
    return reports
