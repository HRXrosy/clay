"""Exact physical-pair covers in rigid-owner displacement coordinates.

The original graph remains the authority for every atom pair, clearance and
exemption. This module compresses only its physical disjunctions; fixed edges
and the graph's full replay inventory are unchanged. No solver is constructed.
"""

from __future__ import annotations

from typing import Sequence

from clay._native import load_native
from clay.layout.extract.compaction import Full2DCompactionGraph


_Box = tuple[int, int, int, int]


def _merge_aligned_rectangles(boxes: Sequence[_Box], axis: int) -> tuple[_Box, ...]:
    """Union collinear rectangle intervals without expanding their geometry."""
    other = 1 - axis
    groups = {}
    for box in boxes:
        groups.setdefault((box[other], box[other + 2]), []).append(
            (box[axis], box[axis + 2]))
    result = []
    for (fixed_lo, fixed_hi), intervals in sorted(groups.items()):
        intervals.sort()
        lo, hi = intervals[0]
        def append_interval(a, b):
            result.append((a, fixed_lo, b, fixed_hi) if axis == 0
                          else (fixed_lo, a, fixed_hi, b))
        for next_lo, next_hi in intervals[1:]:
            if next_lo <= hi:
                hi = max(hi, next_hi)
            else:
                append_interval(lo, hi)
                lo, hi = next_lo, next_hi
        append_interval(lo, hi)
    return tuple(sorted(result))


def _exact_forbidden_union(boxes: set[_Box]) -> tuple[_Box, ...]:
    """Keep an exact union only when its rectangular representation is smaller."""
    native = load_native("_place_cpp")
    if native is not None:
        try:
            return native.exact_forbidden_union_cpp(boxes)
        except OverflowError:
            pass
    canonical = tuple(sorted(boxes))
    if len(canonical) <= 1:
        return canonical

    # KLayout's Box coordinates are C-long, hence signed 32-bit on Windows.
    # Preserve the Python/CP integer domain instead of clipping its geometry.
    if any(value < -(2**31) or value > 2**31 - 1
           for box in canonical for value in box):
        return canonical
    # Individually representable coordinates can still span more than the
    # signed coordinate arithmetic used by Region.merge. Preserve exact raw
    # boxes for that domain too; a 4e9-wide union reproduces an internal error.
    if any(max(box[axis + 2] for box in canonical) - min(box[axis] for box in canonical)
           > 2**31 - 1 for axis in (0, 1)):
        return canonical

    import klayout.db as kdb

    reduced = canonical
    if len(reduced) > 1024:
        for axis in (0, 1):
            reduced = _merge_aligned_rectangles(reduced, axis)
    region = kdb.Region()
    # Dense contact arrays yield many distinct but heavily overlapping
    # displacement rectangles. Merging all of those at once creates a large
    # overlap sweep. Exact union is associative: first reduce bounded batches,
    # then merge their (usually tiny) unions. No bounding-box replacement is
    # used, and every original rectangle remains part of the final region.
    if len(reduced) > 1024:
        for start in range(0, len(reduced), 512):
            batch = kdb.Region()
            for box in reduced[start:start + 512]:
                batch.insert(kdb.Box(*box))
            batch.merge()
            region.insert(batch)
    else:
        for box in reduced:
            region.insert(kdb.Box(*box))
    region.merge()
    rectangles = set()
    for polygon in region.each():
        for piece in polygon.decompose_trapezoids():
            bbox = piece.bbox()
            if piece.area() != bbox.width() * bbox.height():
                raise ValueError("full-2D pair forbidden union is not rectangularizable")
            rectangles.add((bbox.left, bbox.bottom, bbox.right, bbox.top))
    if not rectangles:
        raise ValueError("full-2D pair forbidden union lost all geometry")
    covered = tuple(sorted(rectangles))
    return covered if len(covered) < len(canonical) else canonical


def build_full2d_pair_cover(graph: Full2DCompactionGraph) -> dict:
    """Cover every original physical row exactly once, grouped by owner pair.

    The exact frozen graph type has already validated the complete interacting
    pair inventory. Do not repeat that quadratic construction or reconstruct a
    Cartesian product: each row below supplies its own atom pair and clearance.

    For canonical owners A < B, delta is origin_B - origin_A. An atom pair is
    illegal precisely inside its half-open integer displacement rectangle.
    The +1 converts strict integer inequalities; it is independent of grid size.
    """
    if type(graph) is not Full2DCompactionGraph:
        raise TypeError("full-2D pair cover requires an exact validated compaction graph")
    if graph.physical_clearance_nm is not None:
        owner_by_node = {node:owner for node,owner,_offset in graph.node_owner_offsets_nm}
        indices = {}
        for index,row in enumerate(graph.pair_constraints):
            owners = tuple(sorted((owner_by_node[row.first],owner_by_node[row.second])))
            indices.setdefault(owners,[]).append(index)
        return {"schema":"clay.full2d-pair-cover/v2","graph_sha256":graph.sha256,
                "original_atom_pair_count":graph.physical_atom_pair_count,
                "scope":"exact union of every interacting original atom pair",
                "groups":[{"owners":list(owners),"pair_constraint_indices":indices[owners],
                           "forbidden_boxes_nm":[list(box) for box in rectangles]}
                          for owners,rectangles in graph.physical_pair_cover]}
    sizes = dict(graph.sizes_nm)
    geometry = {
        node: (owner, (offset[0], offset[1], offset[0] + sizes[node][0],
                       offset[1] + sizes[node][1]))
        for node, owner, offset in graph.node_owner_offsets_nm
    }
    groups: dict[tuple[str, str], tuple[list[int], set[_Box]]] = {}
    for index, row in enumerate(graph.pair_constraints):
        if row.mode != "four_way_disjunction":
            continue
        first_owner, first = geometry[row.first]
        second_owner, second = geometry[row.second]
        if first_owner > second_owner:
            first_owner, second_owner = second_owner, first_owner
            first, second = second, first
        owners = (first_owner, second_owner)
        group = groups.get(owners)
        if group is None:
            group = groups[owners] = ([], set())
        indices, forbidden = group
        indices.append(index)
        gap = row.clearance_nm
        forbidden.add((
            first[0] - second[2] - gap + 1,
            first[1] - second[3] - gap + 1,
            first[2] + gap - second[0],
            first[3] + gap - second[1],
        ))
    return {
        "schema": "clay.full2d-pair-cover/v1",
        "graph_sha256": graph.sha256,
        "groups": [
            {"owners": list(owners), "pair_constraint_indices": indices,
             "forbidden_boxes_nm": [list(box) for box in _exact_forbidden_union(forbidden)]}
            for owners, (indices, forbidden) in sorted(groups.items())
        ],
    }


def cover_directional_slacks(box: Sequence[int], delta: Sequence[int]) -> tuple[int, int, int, int]:
    """Return right/top/left/bottom slacks for escaping one half-open box."""
    if (type(box) not in (tuple, list) or len(box) != 4
            or any(type(value) is not int for value in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError("full-2D forbidden box must contain four exact ordered integers")
    if (type(delta) not in (tuple, list) or len(delta) != 2
            or any(type(value) is not int for value in delta)):
        raise ValueError("full-2D owner displacement must contain two exact integers")
    return (delta[0] - box[2], delta[1] - box[3],
            box[0] - 1 - delta[0], box[1] - 1 - delta[1])


__all__ = ["build_full2d_pair_cover", "cover_directional_slacks"]
