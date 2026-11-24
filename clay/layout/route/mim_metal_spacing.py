"""PDK-declared MIM checks on complete drawn geometry, independent of net labels.

The KLayout derived-plate rule and native Magic local surround rule differ.
The former selects whole unrelated metal components; the latter examines local
exposed edges, including a facing notch in a remotely attached component.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import klayout.db as kdb


@dataclass(frozen=True)
class MimMetalSpacingCheck:
    rule_id: str
    metal_layer: Tuple[int, int]
    marker_layer: Tuple[int, int]
    spacing_nm: int
    errors: Any


def _outside_corner_is_free(metal, point, tangent, normal):
    """Read the actual material quadrant immediately beyond an edge endpoint.

    Integer Manhattan boundaries make this one-square-nanometre probe exact:
    it cannot cross an unrepresented sub-grid edge.  A concave metal corner
    occupies this quadrant; a convex corner leaves it free.  Another touching
    metal component also counts as actual material, regardless of its owner.
    """
    x, y = point.x, point.y
    opposite_x = x + tangent[0] + normal[0]
    opposite_y = y + tangent[1] + normal[1]
    quadrant = kdb.Region(kdb.Box(min(x, opposite_x), min(y, opposite_y),
                                 max(x, opposite_x), max(y, opposite_y)))
    return (metal & quadrant).is_empty()


def _clip_edge_tangent(edge, axis, lower, upper):
    """Intersect an axis-aligned marker edge with the allowed tangent span."""
    points = [[edge.p1.x, edge.p1.y], [edge.p2.x, edge.p2.y]]
    values = [point[axis] for point in points]
    if ((lower is not None and max(values) < lower)
            or (upper is not None and min(values) > upper)):
        return None
    for point in points:
        if lower is not None:
            point[axis] = max(point[axis], lower)
        if upper is not None:
            point[axis] = min(point[axis], upper)
    if points[0] == points[1]:
        return None
    return kdb.Edge(kdb.Point(*points[0]), kdb.Point(*points[1]))


def _local_surround_errors(metal, marker, spacing_nm):
    """Magic's Manhattan outside strips with free-corner Euclidean extensions.

    Magic 8.3.642 DRCbasic.c clips each triggered strip to the source edge and
    extends its ends only where the adjacent outside corner is not metal.  A
    raw Edges check loses that corner context and falsely extends concave ends.
    Keep its exact Euclidean distance operation, but restrict target geometry
    to each source edge's allowed tangent span before checking it again.

    This is not whole-component exemption or a subtraction of all intervening
    metal: a remote bridge leaves a local strip active.  Non-Manhattan inputs
    require their own native semantics and cannot use this Manhattan predicate.
    """
    errors = kdb.EdgePairs()
    if metal.is_empty() or marker.is_empty():
        return errors
    metal_edges, marker_edges = metal.edges(), marker.edges()
    for edges in (metal_edges, marker_edges):
        if any(edge.p1.x != edge.p2.x and edge.p1.y != edge.p2.y for edge in edges.each()):
            raise ValueError("MIM local surround currently requires Manhattan geometry")
    # whole_edges=True is essential: a distance-truncated error endpoint is
    # not a source corner and cannot authorize or suppress a corner extension.
    candidates = metal_edges.separation_check(
        marker_edges, spacing_nm, True, kdb.Edges.Euclidian,
    )
    spans = {}
    for pair in candidates.each():
        source = pair.first
        identity = (source.p1.x, source.p1.y, source.p2.x, source.p2.y)
        if identity not in spans:
            dx, dy = source.p2.x - source.p1.x, source.p2.y - source.p1.y
            tangent = ((dx > 0) - (dx < 0), (dy > 0) - (dy < 0))
            # Region contours carry material on the right, including holes.
            normal = (-tangent[1], tangent[0])
            first_free = _outside_corner_is_free(
                metal, source.p1, (-tangent[0], -tangent[1]), normal,
            )
            last_free = _outside_corner_is_free(metal, source.p2, tangent, normal)
            axis = 0 if dx else 1
            first, last = identity[axis], identity[axis + 2]
            if first < last:
                lower, upper = (None if first_free else first), (None if last_free else last)
            else:
                lower, upper = (None if last_free else last), (None if first_free else first)
            spans[identity] = axis, lower, upper
        target = _clip_edge_tangent(pair.second, *spans[identity])
        if target is None:
            continue
        checked = kdb.Edges(source).separation_check(
            kdb.Edges(target), spacing_nm, False, kdb.Edges.Euclidian,
        )
        for error in checked.each():
            errors.insert(error)
    return errors


def mim_metal_spacing_checks(regions_by_gds: Mapping, tech):
    """Return exact exposed-edge obligations for each declared MIM rule.

    Inputs are integer-nanometre literal polygons.  Metal is merged before its
    boundary is read; decomposed rectangles, labels and component bounding boxes
    cannot supply attachment evidence.  Missing markers mean no MIM obligation.
    """
    query = getattr(tech, "mim_metal_geometry_rules", None)
    rules = tuple(query()) if callable(query) else ()
    result = []
    for rule in rules:
        metal = regions_by_gds.get(rule.metal_layer, kdb.Region()).merged()
        marker = regions_by_gds.get(rule.marker_layer, kdb.Region()).merged()
        if rule.predicate == "derived_unrelated":
            derived = ((marker & metal).sized(rule.grow_nm)).merged()
            unrelated = metal.not_interacting(derived)
            errors = unrelated.separation_check(
                derived, rule.spacing_nm, False, kdb.Region.Euclidian,
            )
        elif rule.predicate == "local_surround":
            errors = _local_surround_errors(metal, marker, rule.spacing_nm)
        else:
            raise ValueError(f"unsupported MIM geometry predicate: {rule.predicate!r}")
        result.append(MimMetalSpacingCheck(
            rule.rule_id, rule.metal_layer, rule.marker_layer, rule.spacing_nm, errors,
        ))
    return tuple(result)
