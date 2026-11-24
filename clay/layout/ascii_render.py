"""ASCII art renderer for ParameterizedLibrary cells.

Visualizes a NamedCell's polygons as a grid of characters where each cell
indicates which layer occupies the point at the center of the grid cell.
Intended for debugging and documentation; not a substitute for KLayout.

Layout/character conventions:

* The grid is ``width_chars`` columns wide; the row count is derived from the
  cell bounding box and an empirical ``CHAR_ASPECT = 2.0`` (a terminal character
  is about twice as tall as it is wide) so the picture looks roughly
  proportional when viewed in a fixed-width font.
* Row 0 is the **top** of the layout (positive Y in GDS coordinates).
* Empty grid cells render as ``'.'``.
* When multiple layers contain the same grid-cell center, the *last* polygon
  in iteration order wins (matches the painter's-algorithm intuition).
* Cells with no polygons render as ``'<empty>'`` (a single token, no grid).

Default ``(layer, datatype)`` -> char map (sky130):

* ``(66, 20)`` poly      -> ``'P'``
* ``(65, 20)`` ndiff     -> ``'N'``
* ``(66, 44)`` contact   -> ``'c'``
* ``(67, 20)`` metal1    -> ``'M'``
* ``(64, 20)`` nwell     -> ``'W'``
* any other layer        -> ``'?'``

Callers can supply ``layer_chars`` to override or extend the mapping.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary

# Default (layer, datatype) -> single-char mapping (sky130 conventions).
DEFAULT_LAYER_CHARS: Dict[Tuple[int, int], str] = {
    (66, 20): "P",   # poly
    (65, 20): "N",   # ndiff / pdiff (same gds key in sky130)
    (66, 44): "c",   # contact
    (67, 20): "M",   # metal1
    (64, 20): "W",   # nwell
}

# Character used for grid cells not covered by any polygon.
EMPTY_CELL_CHAR = "."

# Character returned for layers not present in the active map.
UNKNOWN_LAYER_CHAR = "?"

# Empirical terminal character aspect ratio (height / width). Used to pick the
# row count so a square layout renders roughly square on screen.
CHAR_ASPECT = 2.0

# Sentinel string returned for cells with zero polygons.
EMPTY_CELL_RENDERING = "<empty>"


def _polygon_bbox(lp: LayerPolygon) -> Optional[Tuple[int, int, int, int]]:
    verts = lp.polygon.vertices
    if not verts:
        return None
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def _cell_bbox(cell: NamedCell) -> Optional[Tuple[int, int, int, int]]:
    xs = [v.x for lp in cell.polygons for v in lp.polygon.vertices]
    ys = [v.y for lp in cell.polygons for v in lp.polygon.vertices]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _point_in_polygon(px: float, py: float, lp: LayerPolygon) -> bool:
    """Ray-casting point-in-polygon for general (possibly non-rectangular) polys.

    Points exactly on an edge are considered inside; this is a debugging
    visualization, not a geometry kernel.
    """
    verts = lp.polygon.vertices
    n = len(verts)
    if n < 3:
        # Degenerate polygons (lines / points) don't fill a grid cell.
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i].x, verts[i].y
        xj, yj = verts[j].x, verts[j].y
        # Standard ray-cast (open at top, closed at bottom).
        cond = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi if yj != yi else 1e-30) + xi
        )
        if cond:
            inside = not inside
        j = i
    return inside


def _find_cell(lib: ParameterizedLibrary, cell_name: str) -> NamedCell:
    for c in lib.cells:
        if c.name == cell_name:
            return c
    available = ", ".join(c.name for c in lib.cells) or "<none>"
    raise ValueError(
        f"cell {cell_name!r} not found in library {lib.name!r}; "
        f"available cells: {available}"
    )


def render_ascii(
    lib: ParameterizedLibrary,
    cell_name: str,
    width_chars: int = 80,
    layer_chars: Optional[Dict[Tuple[int, int], str]] = None,
) -> str:
    """Render a ParameterizedLibrary cell as ASCII grid.

    Sample layers (chars):

    * sky130 poly 66/20    -> ``'P'``
    * sky130 ndiff 65/20   -> ``'N'``
    * sky130 contact 66/44 -> ``'c'``
    * sky130 metal1 67/20  -> ``'M'``
    * sky130 nwell 64/20   -> ``'W'``
    * other layers         -> ``'?'``

    Returns a multi-line string. Cells with no polygons return ``'<empty>'``.
    Out-of-range cells raise ``ValueError``.
    """
    if width_chars < 1:
        raise ValueError(f"width_chars must be >= 1, got {width_chars}")

    cell = _find_cell(lib, cell_name)

    if not cell.polygons:
        return EMPTY_CELL_RENDERING

    bbox = _cell_bbox(cell)
    if bbox is None:
        return EMPTY_CELL_RENDERING

    x_min, y_min, x_max, y_max = bbox
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # Degenerate bbox: a 1xN strip still gets rendered as a single row of width_chars.
    if bbox_w <= 0:
        bbox_w = 1
    if bbox_h <= 0:
        bbox_h = 1

    # Pick row count so the picture is aspect-preserving on a terminal where
    # one character takes CHAR_ASPECT times as much vertical space as horizontal.
    height_chars = max(1, int(round(bbox_h * width_chars / (bbox_w * CHAR_ASPECT))))

    chars = layer_chars if layer_chars is not None else DEFAULT_LAYER_CHARS

    # Pre-compute polygon bounding boxes for cheap rejection before doing
    # the more expensive ray-cast point-in-polygon test.
    poly_bboxes = [_polygon_bbox(lp) for lp in cell.polygons]

    # Render row 0 at the TOP of the layout (largest y).
    lines = []
    for row in range(height_chars):
        # Sample at the center of each grid cell.
        v_frac = (row + 0.5) / height_chars
        # y_top - frac * h gives top-down sampling.
        py = y_max - v_frac * bbox_h
        line_chars = []
        for col in range(width_chars):
            u_frac = (col + 0.5) / width_chars
            px = x_min + u_frac * bbox_w
            ch = EMPTY_CELL_CHAR
            # Painter's algorithm: later polygons override earlier ones.
            for lp, pb in zip(cell.polygons, poly_bboxes):
                if pb is None:
                    continue
                pbx_min, pby_min, pbx_max, pby_max = pb
                if px < pbx_min or px > pbx_max or py < pby_min or py > pby_max:
                    continue
                if _point_in_polygon(px, py, lp):
                    key = (lp.layer, lp.datatype)
                    ch = chars.get(key, UNKNOWN_LAYER_CHAR)
            line_chars.append(ch)
        lines.append("".join(line_chars))

    return "\n".join(lines)
