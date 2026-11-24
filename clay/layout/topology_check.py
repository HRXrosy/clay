"""Topology checker for ParameterizedLibrary.

Pure-Python audit pass over every polygon in a ParameterizedLibrary that
flags four structural concerns:

  * ``zero_area`` — polygon area is <= 0 (degenerate ring or wrong winding
    that collapses to zero signed area).
  * ``isolated`` — no other polygon on the **same layer/datatype** is
    within ``isolated_threshold_nm`` bbox-to-bbox distance. Useful for
    catching layers where every shape is supposed to participate in a
    contiguous net but one ended up stranded.
  * ``self_intersect`` — at least one pair of non-adjacent edges of the
    polygon ring cross each other. The check is the straightforward
    O(n*n) edge-pair test gated on bbox overlap; it is meant to catch
    obvious figure-8 / bowtie shapes, not subtle floating-point cases.
  * ``collinear`` — three or more consecutive vertices are colinear
    (informational; emitted as a hint for GDS cleanup tools).

Designed to be cheap to run on the kind of small ParameterizedLibrary
objects produced by clay's layout backend (a few cells, tens of
polygons per cell); we therefore favor obvious code over algorithmic
cleverness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary, Polygon


# ---------------------------------------------------------------------------
# Public data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopologyIssue:
    """One finding from a topology audit.

    ``issue_type`` is one of: ``"zero_area"``, ``"isolated"``,
    ``"self_intersect"``, ``"collinear"``.
    """

    cell_name: str
    polygon_index: int
    issue_type: str
    message: str


@dataclass
class TopologyReport:
    """Container for every issue surfaced during a single audit."""

    issues: List[TopologyIssue] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0


VALID_ISSUE_TYPES = frozenset({"zero_area", "isolated", "self_intersect", "collinear"})


# ---------------------------------------------------------------------------
# Geometry helpers (kept private; not part of the public surface)
# ---------------------------------------------------------------------------


def _polygon_area(poly: Polygon) -> float:
    """Signed shoelace area; absolute value is the polygon area."""
    verts = poly.vertices
    n = len(verts)
    if n < 3:
        return 0.0
    acc = 0
    for i in range(n):
        x1, y1 = verts[i].x, verts[i].y
        x2, y2 = verts[(i + 1) % n].x, verts[(i + 1) % n].y
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def _polygon_bbox(poly: Polygon) -> Tuple[int, int, int, int]:
    """Return (xmin, ymin, xmax, ymax). Caller guarantees >= 1 vertex."""
    xs = [v.x for v in poly.vertices]
    ys = [v.y for v in poly.vertices]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Manhattan-style gap between two AABBs; 0 if they overlap/touch."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0, max(by0 - ay1, ay0 - by1))
    return float(max(dx, dy))


def _segments_cross(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    p3: Tuple[int, int],
    p4: Tuple[int, int],
) -> bool:
    """Return True if open segments (p1,p2) and (p3,p4) properly cross.

    Endpoints touching is not counted as a crossing — this avoids
    flagging every shared vertex of the polygon ring.
    """

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    # Proper crossing: strict sign change on both sides.
    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True
    return False


def _polygon_self_intersects(poly: Polygon) -> bool:
    """O(n^2) check for any pair of non-adjacent edges that properly cross."""
    verts = poly.vertices
    n = len(verts)
    if n < 4:
        return False
    edges = []
    for i in range(n):
        a = (verts[i].x, verts[i].y)
        b = (verts[(i + 1) % n].x, verts[(i + 1) % n].y)
        edges.append((a, b))
    for i in range(n):
        for j in range(i + 1, n):
            # Skip adjacent edges and the wrap-around adjacency.
            if j == i + 1:
                continue
            if i == 0 and j == n - 1:
                continue
            a, b = edges[i]
            c, d = edges[j]
            if _segments_cross(a, b, c, d):
                return True
    return False


def _polygon_collinear_runs(poly: Polygon) -> bool:
    """True if any three consecutive vertices are colinear."""
    verts = poly.vertices
    n = len(verts)
    if n < 3:
        return False
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        c = verts[(i + 2) % n]
        cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
        if cross == 0:
            # Make sure b actually lies between a and c (not a doubled vertex
            # or a backtrack — those are still flagged here as colinear).
            return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_topology(
    lib: ParameterizedLibrary,
    isolated_threshold_nm: int = 1000,
) -> TopologyReport:
    """Walk every polygon in ``lib`` and accumulate topology issues.

    For ``isolated`` checks, distance is measured between axis-aligned
    bounding boxes on the same (layer, datatype) key, *within the same
    cell*. A polygon is reported as isolated if no sibling polygon on the
    same layer key lies within ``isolated_threshold_nm`` of its bbox.
    """

    report = TopologyReport()

    for cell in lib.cells:
        # Pre-compute bboxes & layer-keys so the isolation pass is cheap.
        bboxes: List[Tuple[int, int, int, int]] = []
        layer_keys: List[Tuple[int, int]] = []
        for lp in cell.polygons:
            if lp.polygon.vertices:
                bboxes.append(_polygon_bbox(lp.polygon))
            else:
                # Empty vertex list — synthesize a degenerate bbox; the
                # zero_area pass below will catch the real problem.
                bboxes.append((0, 0, 0, 0))
            layer_keys.append((lp.layer, lp.datatype))

        for idx, lp in enumerate(cell.polygons):
            poly = lp.polygon

            # 1) zero area --------------------------------------------------
            area = _polygon_area(poly)
            if area <= 0:
                report.issues.append(
                    TopologyIssue(
                        cell_name=cell.name,
                        polygon_index=idx,
                        issue_type="zero_area",
                        message=(
                            f"polygon on layer "
                            f"({lp.layer},{lp.datatype}) has area={area:.3f}"
                        ),
                    )
                )
                # Even if zero-area, still run the other checks so the
                # caller gets a complete picture in one pass.

            # 2) self intersect --------------------------------------------
            if _polygon_self_intersects(poly):
                report.issues.append(
                    TopologyIssue(
                        cell_name=cell.name,
                        polygon_index=idx,
                        issue_type="self_intersect",
                        message=(
                            f"polygon on layer "
                            f"({lp.layer},{lp.datatype}) has crossing edges"
                        ),
                    )
                )

            # 3) collinear runs (informational) ----------------------------
            if _polygon_collinear_runs(poly):
                report.issues.append(
                    TopologyIssue(
                        cell_name=cell.name,
                        polygon_index=idx,
                        issue_type="collinear",
                        message=(
                            f"polygon on layer "
                            f"({lp.layer},{lp.datatype}) has 3+ "
                            f"consecutive colinear vertices"
                        ),
                    )
                )

            # 4) isolated --------------------------------------------------
            my_key = layer_keys[idx]
            my_bbox = bboxes[idx]
            if not poly.vertices:
                # Skip isolation check for vertex-less polygons.
                continue
            neighbor_found = False
            for j, other_key in enumerate(layer_keys):
                if j == idx:
                    continue
                if other_key != my_key:
                    continue
                if not cell.polygons[j].polygon.vertices:
                    continue
                if _bbox_distance(my_bbox, bboxes[j]) <= float(isolated_threshold_nm):
                    neighbor_found = True
                    break
            if not neighbor_found:
                report.issues.append(
                    TopologyIssue(
                        cell_name=cell.name,
                        polygon_index=idx,
                        issue_type="isolated",
                        message=(
                            f"polygon on layer ({lp.layer},{lp.datatype}) "
                            f"has no neighbor within {isolated_threshold_nm} nm"
                        ),
                    )
                )

    return report


__all__ = [
    "TopologyIssue",
    "TopologyReport",
    "check_topology",
    "VALID_ISSUE_TYPES",
]
