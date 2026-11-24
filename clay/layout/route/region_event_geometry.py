"""Rectangular region cover and its directed, interval-valued boundaries.

All coordinates are integers in the caller's coordinate system. Rectangles
and portal intervals are half-open. This module does not select coordinates,
construct routes, apply technology rules, or grant physical connectivity.
"""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Callable

import klayout.db as db

from clay._native import load_native
from clay.layout.spatial.rectangle_union import rectangle_union_cover


Rectangle = tuple[int, int, int, int]
Checkpoint = Callable[[], None]


@dataclass(frozen=True, order=True)
class Portal:
    """One directed adjacency through ``[lo, hi)`` on ``axis=boundary``.

    Direction is +1 or -1 along the named axis. The same physical boundary
    appears twice, once in each direction, with the same interval.
    """

    source: int
    target: int
    axis: str
    direction: int
    boundary: int
    lo: int
    hi: int


@dataclass(frozen=True)
class RegionGeometry:
    rectangles: tuple[Rectangle, ...]
    portals: tuple[Portal, ...]

    def portals_by_source(self) -> tuple[tuple[Portal, ...], ...]:
        """Index directed portals once for a caller's local graph lifetime."""
        rows: list[list[Portal]] = [[] for _ in self.rectangles]
        for portal in self.portals:
            rows[portal.source].append(portal)
        return tuple(tuple(row) for row in rows)


def project_portal(
    geometry: RegionGeometry, portal: Portal, point: tuple[int, int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Project an entry onto the portal and return orthogonal candidate legs.

    Coordinates are grid indices. Both nondegenerate elbow orientations are
    retained; no midpoint is chosen. These are geometric suggestions only:
    callers must judge each actual segment against the current routing world.
    A projection is not a completeness or shortest-path proof for the portal.
    """
    cpp = load_native("_gridless_cpp")
    if cpp is not None and type(geometry.rectangles) is tuple and portal.axis in ("x", "y"):
        return cpp.project_portal_cpp(
            geometry.rectangles,
            portal.source,
            portal.target,
            portal.axis == "x",
            portal.direction,
            portal.boundary,
            portal.lo,
            portal.hi,
            point[0],
            point[1],
        )
    if (portal.source < 0 or portal.source >= len(geometry.rectangles)
            or portal.target < 0 or portal.target >= len(geometry.rectangles)
            or portal.axis not in ("x", "y") or portal.direction not in (-1, 1)
            or portal.lo >= portal.hi):
        raise ValueError("invalid portal")
    x, y = point
    x0, y0, x1, y1 = geometry.rectangles[portal.source]
    if not (x0 <= x < x1 and y0 <= y < y1):
        raise ValueError("portal entry lies outside its source rectangle")
    if portal.axis == "x":
        tangent = min(max(y, portal.lo), portal.hi - 1)
        near = (portal.boundary - (1 if portal.direction == 1 else 0), tangent)
        far = (portal.boundary - (0 if portal.direction == 1 else 1), tangent)
    else:
        tangent = min(max(x, portal.lo), portal.hi - 1)
        near = (tangent, portal.boundary - (1 if portal.direction == 1 else 0))
        far = (tangent, portal.boundary - (0 if portal.direction == 1 else 1))
    tx0, ty0, tx1, ty1 = geometry.rectangles[portal.target]
    if not (x0 <= near[0] < x1 and y0 <= near[1] < y1
            and tx0 <= far[0] < tx1 and ty0 <= far[1] < ty1):
        raise ValueError("portal does not join its declared rectangles")
    paths = set()
    for elbow in ((near[0], y), (x, near[1])):
        path = []
        for candidate in (point, elbow, near, far):
            if not path or path[-1] != candidate:
                path.append(candidate)
        paths.add(tuple(path))
    return tuple(sorted(paths))


def project_target(
    geometry: RegionGeometry, source: int, point: tuple[int, int], target: Rectangle,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Project onto the target/rectangle intersection without sampling sites."""
    cpp = load_native("_gridless_cpp")
    if cpp is not None and type(geometry.rectangles) is tuple:
        return cpp.project_target_cpp(
            geometry.rectangles,
            source,
            point[0],
            point[1],
            target[0],
            target[1],
            target[2],
            target[3],
        )
    if source < 0 or source >= len(geometry.rectangles):
        raise ValueError("invalid source rectangle")
    x, y = point
    x0, y0, x1, y1 = geometry.rectangles[source]
    if not (x0 <= x < x1 and y0 <= y < y1):
        raise ValueError("target projection starts outside its rectangle")
    tx0, ty0, tx1, ty1 = target
    lo_x, lo_y, hi_x, hi_y = max(x0, tx0), max(y0, ty0), min(x1, tx1), min(y1, ty1)
    if lo_x >= hi_x or lo_y >= hi_y:
        return ()
    end = (min(max(x, lo_x), hi_x - 1), min(max(y, lo_y), hi_y - 1))
    paths = set()
    for elbow in ((end[0], y), (x, end[1])):
        path = []
        for candidate in (point, elbow, end):
            if not path or path[-1] != candidate:
                path.append(candidate)
        paths.add(tuple(path))
    return tuple(sorted(paths))


def iter_axis_cost_segments(
    start: tuple[int, int], end: tuple[int, int],
    xs: tuple[int, ...] | list[int], ys: tuple[int, ...] | list[int],
) -> Iterator[tuple[int, int, int, int]]:
    """Split a rectilinear run for pricing, without changing emitted geometry.

    ``xs`` and ``ys`` are the existing strictly ordered Cartesian search axes.
    A caller must pass an entire contiguous run, including portions on either
    side of portal boundaries; pricing a partial prefix can double-charge.
    """
    ax, ay = start
    bx, by = end
    if ax == bx and ay == by:
        return
    if ax != bx and ay != by:
        raise ValueError("cost run must be orthogonal")
    horizontal = ay == by
    lower, upper = sorted((ax, bx) if horizontal else (ay, by))
    axes = xs if horizontal else ys
    first, stop = bisect_right(axes, lower), bisect_left(axes, upper)
    reverse = bx < ax if horizontal else by < ay
    indices = range(stop - 1, first - 1, -1) if reverse else range(first, stop)
    previous = start
    for index in indices:
        value = axes[index]
        current = (value, ay) if horizontal else (ax, value)
        yield previous[0], previous[1], current[0], current[1]
        previous = current
    yield previous[0], previous[1], end[0], end[1]


def axis_cost_segments(
    start: tuple[int, int], end: tuple[int, int],
    xs: tuple[int, ...] | list[int], ys: tuple[int, ...] | list[int],
) -> tuple[tuple[int, int, int, int], ...]:
    """Materialize a pricing partition for callers that explicitly need one."""
    return tuple(iter_axis_cost_segments(start, end, xs, ys))


def settled_axis_prefix(start, end, xs, ys):
    """Last fixed-axis point reached by a directed orthogonal run, if any.

    Settling up to this boundary yields exactly the prefix of the original
    axis-cost partition, including for off-axis starts and reversed travel.
    """
    ax, ay = start
    bx, by = end
    if start == end:
        return None
    if ax != bx and ay != by:
        raise ValueError("cost run must be orthogonal")
    horizontal = ay == by
    first, last = (ax, bx) if horizontal else (ay, by)
    axes = xs if horizontal else ys
    if last > first:
        index = bisect_right(axes, last) - 1
        coordinate = axes[index] if index >= 0 and axes[index] > first else None
    else:
        index = bisect_left(axes, last)
        coordinate = axes[index] if index < len(axes) and axes[index] < first else None
    if coordinate is None:
        return None
    return (coordinate, ay) if horizontal else (ax, coordinate)


def conservative_center_region(
    bounds_nm: Rectangle,
    blockers: tuple[tuple[Rectangle, int], ...],
    *,
    grid_nm: int,
    checkpoint: Checkpoint | None = None,
) -> db.Region:
    """Build a conservative index-space subset from caller-supplied halos.

    Blocker rectangles and their independent halos are already classified by
    the caller's actual routing rules. This helper neither selects a technology
    rule nor treats the resulting subset as a complete routing domain. Each
    unit cell represents one manufacturing-grid coordinate; closed integer
    endpoint intervals are therefore represented as half-open cell boxes.
    """
    if type(grid_nm) is not int or grid_nm <= 0:
        raise ValueError("manufacturing grid must be a positive integer")

    def index_box(rect: Rectangle):
        if len(rect) != 4 or any(type(value) is not int for value in rect):
            raise TypeError("region bounds require four exact integers")
        x0, y0, x1, y1 = rect
        if x0 > x1 or y0 > y1:
            raise ValueError("region bounds are inverted")
        lo_x, lo_y = -(-x0 // grid_nm), -(-y0 // grid_nm)
        hi_x, hi_y = x1 // grid_nm, y1 // grid_nm
        if lo_x > hi_x or lo_y > hi_y:
            return None
        return db.Box(lo_x, lo_y, hi_x + 1, hi_y + 1)

    if checkpoint is not None:
        checkpoint()
    outer = index_box(bounds_nm)
    if outer is None:
        return db.Region()
    excluded = db.Region()
    for index, (rect, halo) in enumerate(blockers):
        if checkpoint is not None and index % 128 == 0:
            checkpoint()
        if type(halo) is not int or halo < 0:
            raise ValueError("blocker halo must be a nonnegative integer")
        if len(rect) != 4 or any(type(value) is not int for value in rect):
            raise TypeError("blocker requires four exact integers")
        x0, y0, x1, y1 = rect
        if x0 > x1 or y0 > y1:
            raise ValueError("blocker rectangle is inverted")
        expanded = index_box((x0 - halo, y0 - halo, x1 + halo, y1 + halo))
        if expanded is not None:
            excluded.insert(expanded)
    free = db.Region(outer) - excluded.merged()
    free.min_coherence = True
    free = free.merged()
    free.min_coherence = True
    if checkpoint is not None:
        checkpoint()
    return free


def build_region_geometry(
    region: db.Region, *, checkpoint: Checkpoint | None = None
) -> RegionGeometry:
    """Return an exact disjoint Manhattan cover and all shared-edge intervals.

    The input region may contain holes and corner-touching components. The
    output rectangles retain exactly its area; point-only contact never creates
    a portal. A non-Manhattan piece fails closed. No grid sites are enumerated.
    The sweep pairs sorted, disjoint intervals on opposite sides of each shared
    boundary. For E boundary events and P directed portals, that sweep takes
    O(E log E + P) time and O(E + P) storage, after decomposition and cover
    construction. KLayout Boolean/decomposition calls are atomic with respect
    to the cooperative checkpoint; callers still own their outer time limit.
    """
    if not isinstance(region, db.Region):
        raise TypeError("region must be a KLayout Region")
    if checkpoint is not None:
        checkpoint()

    pieces = region.decompose_trapezoids(db.Polygon.TD_vtrapezoids)
    if checkpoint is not None:
        checkpoint()
    rectangles: list[Rectangle] = []
    # KLayout may expose borrowed handles: consume each piece in this loop.
    for index, piece in enumerate(pieces):
        if checkpoint is not None and index % 128 == 0:
            checkpoint()
        box = piece.bbox()
        if piece.area() != box.area() or box.width() <= 0 or box.height() <= 0:
            raise ValueError("region requires nondegenerate Manhattan geometry")
        rectangles.append((box.left, box.bottom, box.right, box.top))
    cover = rectangle_union_cover(rectangles, checkpoint=checkpoint)

    reconstructed = db.Region()
    for index, rect in enumerate(cover):
        if checkpoint is not None and index % 128 == 0:
            checkpoint()
        reconstructed.insert(db.Box(*rect))
    if checkpoint is not None:
        checkpoint()
    changed = not (reconstructed ^ region).is_empty()
    if checkpoint is not None:
        checkpoint()
    if changed:
        raise ValueError("rectangle cover changed the input region")

    left: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    right: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    bottom: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    top: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for index, (x0, y0, x1, y1) in enumerate(cover):
        if checkpoint is not None and index % 128 == 0:
            checkpoint()
        left[x0].append((y0, y1, index))
        right[x1].append((y0, y1, index))
        bottom[y0].append((x0, x1, index))
        top[y1].append((x0, x1, index))

    portals: list[Portal] = []
    work = 0

    def match(
        lower: list[tuple[int, int, int]],
        upper: list[tuple[int, int, int]],
        axis: str,
        boundary: int,
    ) -> None:
        nonlocal work
        lower, upper = sorted(lower), sorted(upper)
        i = j = 0
        while i < len(lower) and j < len(upper):
            if checkpoint is not None and work % 128 == 0:
                checkpoint()
            work += 1
            a0, a1, a = lower[i]
            b0, b1, b = upper[j]
            lo, hi = max(a0, b0), min(a1, b1)
            if lo < hi:
                portals.append(Portal(a, b, axis, 1, boundary, lo, hi))
                portals.append(Portal(b, a, axis, -1, boundary, lo, hi))
            if a1 <= b1:
                i += 1
            if b1 <= a1:
                j += 1

    for boundary in sorted(right.keys() & left.keys()):
        match(right[boundary], left[boundary], "x", boundary)
    for boundary in sorted(top.keys() & bottom.keys()):
        match(top[boundary], bottom[boundary], "y", boundary)
    if checkpoint is not None:
        checkpoint()
    return RegionGeometry(cover, tuple(sorted(portals)))
