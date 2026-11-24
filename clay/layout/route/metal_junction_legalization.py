"""Shared exact geometry kernel for merged-metal junction legalization.

The route materializer and the independent emit audit must judge the same
PDK-derived width/notch obligation.  This module owns only pure KLayout region
math: it neither assigns net ownership nor emits route shapes.
"""

from __future__ import annotations

from typing import Any, Tuple

import klayout.db as kdb


def width_pair_core_extent(pair: Any) -> int:
    """Return the projection-overlap core of one euclidean width edge pair."""
    e1, e2 = pair.first, pair.second
    lo1x, hi1x = sorted((e1.p1.x, e1.p2.x))
    lo2x, hi2x = sorted((e2.p1.x, e2.p2.x))
    lo1y, hi1y = sorted((e1.p1.y, e1.p2.y))
    lo2y, hi2y = sorted((e2.p1.y, e2.p2.y))
    core_w = min(hi1x, hi2x) - max(lo1x, lo2x)
    core_h = min(hi1y, hi2y) - max(lo1y, lo2y)
    return max(core_w, core_h, 0)


def _grid_aligned_cover(region: kdb.Region, grid_nm: int) -> kdb.Region:
    grid = int(grid_nm)
    if grid <= 0:
        raise ValueError("manufacturing grid must be a positive integer nm")
    if grid == 1 or region.is_empty():
        return region.dup()
    cover = kdb.Region()
    for shape in region.decompose_trapezoids():
        bbox = shape.polygon.bbox()
        cover += kdb.Region(
            kdb.Box(
                (bbox.left // grid) * grid,
                (bbox.bottom // grid) * grid,
                -((-bbox.right) // grid) * grid,
                -((-bbox.top) // grid) * grid,
            )
        )
    cover.merge()
    return cover


def legalize_island_junctions(
    own: kdb.Region,
    blocked: kdb.Region,
    *,
    min_width_nm: int,
    min_space_nm: int,
    max_passes: int = 3,
    manufacturing_grid_nm: int = 1,
) -> Tuple[kdb.Region, Tuple[kdb.Region, ...]]:
    """Return one island plus deterministic added regions that remove local necks.

    ``blocked`` is already expanded by the caller's required clearance and must
    contain every other conductor island/forbidden region.  A repair that cannot
    cover its violation core after clipping is omitted, leaving an honest
    residual for the deck oracle.
    """
    min_width = int(min_width_nm)
    min_space = int(min_space_nm)
    current = own.dup()
    current.merge()
    forbidden = blocked.dup()
    forbidden.merge()
    if current.is_empty() or min_width <= 0 or min_space <= 0:
        return current, ()

    half_width = (min_width - 1) // 2
    additions = []
    for _pass in range(max(0, int(max_passes))):
        pair_boxes = []
        for pair in current.width_check(
            min_width,
            False,
            kdb.Region.Euclidian,
            None,
            None,
            None,
        ).each():
            if width_pair_core_extent(pair) <= 2 * min_width:
                pair_boxes.append(pair.bbox())
        for pair in current.space_check(
            min_space,
            False,
            kdb.Region.Euclidian,
            None,
            None,
            None,
        ).each():
            bbox = pair.bbox()
            if min(bbox.width(), bbox.height()) <= 2 * min_width:
                pair_boxes.append(bbox)

        fills = kdb.Region()
        for bbox in pair_boxes:
            repair_box = kdb.Box(
                bbox.left - min_width,
                bbox.bottom - min_width,
                bbox.right + min_width,
                bbox.top + min_width,
            )
            candidate = (kdb.Region(repair_box) - forbidden).merged()
            if half_width > 0:
                candidate = candidate.sized(-half_width).sized(half_width)
            if not (kdb.Region(bbox) - candidate - current).is_empty():
                continue
            new_material = (candidate - current).merged()
            if not new_material.is_empty():
                fills += new_material
        fills.merge()
        fills = (_grid_aligned_cover(fills, manufacturing_grid_nm) - forbidden).merged()
        if fills.is_empty():
            break
        additions.append(fills.dup())
        current = (current + fills).merged()
    return current, tuple(additions)
