"""A* router on a Manhattan grid avoiding rectangular obstacles.

Pure Python (stdlib ``heapq`` only). Coordinates are integers in DBU
(database units, typically nm). The router snaps ``start`` and ``end`` to the
nearest grid node, expands in the four cardinal directions with step size
``grid_size_nm``, and returns the cheapest path that does not pass through any
obstacle rectangle. The heuristic is plain Manhattan distance to ``end`` --
admissible on a 4-connected unit-step grid, so A* returns an optimal path.

Obstacles are treated as closed half-open boxes ``[x_min, x_max] x
[y_min, y_max]``: a grid node ``(x, y)`` is blocked iff
``x_min <= x <= x_max and y_min <= y <= y_max``. This means an obstacle that
covers exactly the start or end node makes that endpoint unreachable and the
router returns ``found=False``.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned rectangular obstacle in DBU.

    The rectangle is inclusive on every side: a grid node falling on the
    boundary counts as blocked.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError(
                f"Obstacle must satisfy x_max>=x_min and y_max>=y_min, got {self!r}"
            )

    def contains(self, x: int, y: int) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass
class RouteResult:
    """Result of a single A* routing call."""

    waypoints: List[Tuple[int, int]] = field(default_factory=list)
    cost: int = 0
    found: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap(value: int, step: int) -> int:
    """Snap ``value`` to the nearest multiple of ``step`` (ties round down)."""
    # Integer floor-rounded snap. We use ``round`` semantics relative to the
    # lower grid line so the snap is symmetric for positive and negative
    # coordinates.
    q, r = divmod(value, step)
    if r * 2 >= step:
        q += 1
    return q * step


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _auto_bbox(
    start: Tuple[int, int],
    end: Tuple[int, int],
    obstacles: List[Obstacle],
    margin: int,
) -> Tuple[int, int, int, int]:
    xs = [start[0], end[0]]
    ys = [start[1], end[1]]
    for ob in obstacles:
        xs.extend([ob.x_min, ob.x_max])
        ys.extend([ob.y_min, ob.y_max])
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def _is_blocked(x: int, y: int, obstacles: List[Obstacle]) -> bool:
    for ob in obstacles:
        if ob.contains(x, y):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    obstacles: List[Obstacle],
    grid_size_nm: int = 100,
    bounding_box: Optional[Tuple[int, int, int, int]] = None,
) -> RouteResult:
    """A* shortest Manhattan path between ``start`` and ``end``.

    Parameters
    ----------
    start, end:
        Endpoints in DBU. Snapped to the nearest grid node before search.
    obstacles:
        Rectangles that block grid nodes (inclusive boundary). May be empty.
    grid_size_nm:
        Step size (DBU). Must be > 0.
    bounding_box:
        ``(x_min, y_min, x_max, y_max)`` clamp on the search area. When
        ``None``, an auto bbox covering start, end and all obstacles plus a
        ``2 * grid_size_nm`` margin is used.

    Returns
    -------
    RouteResult
        On success ``found=True``, ``waypoints`` is a list of ``(x, y)``
        tuples beginning with the snapped start and ending with the snapped
        end, and ``cost`` is the total Manhattan distance traversed. On
        failure ``found=False``, ``waypoints=[]`` and ``cost=0``.
    """
    if grid_size_nm <= 0:
        raise ValueError(f"grid_size_nm must be > 0, got {grid_size_nm}")
    step = int(grid_size_nm)

    # Snap endpoints to the grid.
    s = (_snap(int(start[0]), step), _snap(int(start[1]), step))
    g = (_snap(int(end[0]), step), _snap(int(end[1]), step))

    # Resolve the bounding box and snap it outward so all corners are grid
    # nodes that contain start/end.
    if bounding_box is None:
        bx_min, by_min, bx_max, by_max = _auto_bbox(s, g, obstacles, 2 * step)
    else:
        bx_min, by_min, bx_max, by_max = bounding_box
        if bx_max < bx_min or by_max < by_min:
            raise ValueError(f"Invalid bounding_box: {bounding_box!r}")

    def in_bbox(x: int, y: int) -> bool:
        return bx_min <= x <= bx_max and by_min <= y <= by_max

    # If either endpoint is outside the bounding box, no path exists.
    if not in_bbox(*s) or not in_bbox(*g):
        return RouteResult(waypoints=[], cost=0, found=False)

    # If start equals end, return a single-waypoint path -- but only when
    # the node itself is not blocked.
    if s == g:
        if _is_blocked(s[0], s[1], obstacles):
            return RouteResult(waypoints=[], cost=0, found=False)
        return RouteResult(waypoints=[s], cost=0, found=True)

    # If either endpoint lies inside an obstacle, there is no path.
    if _is_blocked(s[0], s[1], obstacles) or _is_blocked(g[0], g[1], obstacles):
        return RouteResult(waypoints=[], cost=0, found=False)

    # Standard A* on a 4-connected grid.
    open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
    counter = 0  # tie-breaker so heapq never compares tuples
    g_score: Dict[Tuple[int, int], int] = {s: 0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    heapq.heappush(open_heap, (_manhattan(s, g), counter, s))
    closed: set = set()

    while open_heap:
        _f, _c, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == g:
            # Reconstruct.
            path: List[Tuple[int, int]] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return RouteResult(waypoints=path, cost=g_score[g], found=True)
        closed.add(current)

        cx, cy = current
        for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
            nx, ny = cx + dx, cy + dy
            if not in_bbox(nx, ny):
                continue
            if _is_blocked(nx, ny, obstacles):
                continue
            neighbor = (nx, ny)
            if neighbor in closed:
                continue
            tentative = g_score[current] + step
            if tentative < g_score.get(neighbor, 1 << 62):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                f_score = tentative + _manhattan(neighbor, g)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))

    return RouteResult(waypoints=[], cost=0, found=False)


__all__ = ["Obstacle", "RouteResult", "find_path"]
