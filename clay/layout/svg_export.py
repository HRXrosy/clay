"""Phase 24 SVG export for ParameterizedLibrary cells.

Renders a NamedCell's polygons as a browser-viewable SVG document. Complements
the Phase 15 ASCII renderer (``ascii_render.py``): same input IR, but produces
real graphics with per-layer colors that round-trip through any SVG viewer.

Design choices:

* Pure Python, no external SVG library. Output is an XML string assembled by
  f-strings, prefixed with an ``<?xml ... ?>`` declaration so the result is a
  standalone document file when saved with a ``.svg`` extension.
* Aspect-preserving: the cell bounding box is mapped uniformly into the
  requested ``width_px x height_px`` viewport. We use SVG's ``viewBox`` plus
  ``preserveAspectRatio="xMidYMid meet"`` so viewers do the math.
* GDS coordinates have Y pointing up; SVG has Y pointing down. We apply a
  ``transform="scale(1, -1)"`` around the cell bbox so the polygons render
  right-side-up without us recomputing every coordinate.
* Per-layer styling is configurable via ``LayerStyle`` and a
  ``Dict[(layer, datatype), LayerStyle]`` map. ``DEFAULT_SKY130_STYLES`` ships
  sensible defaults for the sky130 layers Clay's lowering currently emits;
  layers not in the active map fall back to gray (so unrecognized data is still
  visible, just unstyled).
* Empty cells (zero polygons) still produce a valid minimal SVG document with
  the requested viewport dimensions and no ``<polygon>`` elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary


# --- Layer style ----------------------------------------------------------


@dataclass(frozen=True)
class LayerStyle:
    """Per-layer SVG fill / stroke styling.

    Attributes:
        fill: CSS color string for the polygon body (e.g. ``"#00aa00"``).
        stroke: CSS color string for the polygon outline.
        fill_opacity: 0.0 (transparent) to 1.0 (opaque). Default 0.6 so
            overlapping layers remain visually distinguishable.
    """

    fill: str
    stroke: str
    fill_opacity: float = 0.6


# Default sky130 layer styling.
#
# Keys are ``(layer, datatype)`` GDS pairs matching the values returned by
# ``Sky130LayerMap``. Colors are chosen to be reasonably distinguishable on a
# white background and to roughly match common foundry / KLayout conventions
# (poly red, diff green, metal1 blue, nwell pale yellow).
DEFAULT_SKY130_STYLES: Dict[Tuple[int, int], LayerStyle] = {
    (65, 20): LayerStyle(fill="#00aa00", stroke="#005500"),  # ndiff green
    (66, 20): LayerStyle(fill="#aa0000", stroke="#550000"),  # poly red
    (66, 44): LayerStyle(fill="#000000", stroke="#000000"),  # contact black
    (67, 20): LayerStyle(fill="#0000aa", stroke="#000055"),  # metal1 blue
    (64, 20): LayerStyle(fill="#ffcc00", stroke="#aa8800", fill_opacity=0.3),  # nwell yellow
}

# Fallback styling for any (layer, datatype) not present in the active style
# map. Gray is unobtrusive and signals "unrecognized but present" rather than
# silently dropping the polygon.
UNKNOWN_LAYER_STYLE = LayerStyle(fill="#888888", stroke="#444444", fill_opacity=0.4)


# --- Internal helpers -----------------------------------------------------


def _find_cell(lib: ParameterizedLibrary, cell_name: str) -> NamedCell:
    for c in lib.cells:
        if c.name == cell_name:
            return c
    available = ", ".join(c.name for c in lib.cells) or "(library is empty)"
    raise ValueError(
        f"cell {cell_name!r} not found in library {lib.name!r}; available: {available}"
    )


def _cell_bbox(cell: NamedCell) -> Optional[Tuple[int, int, int, int]]:
    xs: List[int] = []
    ys: List[int] = []
    for lp in cell.polygons:
        for v in lp.polygon.vertices:
            xs.append(v.x)
            ys.append(v.y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _polygon_to_points_attr(lp: LayerPolygon) -> str:
    return " ".join(f"{v.x},{v.y}" for v in lp.polygon.vertices)


def _resolve_style(
    lp: LayerPolygon, styles: Dict[Tuple[int, int], LayerStyle]
) -> LayerStyle:
    return styles.get((lp.layer, lp.datatype), UNKNOWN_LAYER_STYLE)


# --- Public API -----------------------------------------------------------


def render_svg(
    lib: ParameterizedLibrary,
    cell_name: str,
    width_px: int = 800,
    height_px: int = 600,
    layer_styles: Optional[Dict[Tuple[int, int], LayerStyle]] = None,
) -> str:
    """Render ``cell_name`` from ``lib`` as a complete SVG XML document string.

    Args:
        lib: ParameterizedLibrary containing the cell to render.
        cell_name: Name of the cell within ``lib.cells`` to render.
        width_px: Viewport width in pixels (SVG ``width`` attribute).
        height_px: Viewport height in pixels (SVG ``height`` attribute).
        layer_styles: Optional ``(layer, datatype) -> LayerStyle`` map. When
            ``None``, falls back to ``DEFAULT_SKY130_STYLES``. Layers absent
            from the active map render with ``UNKNOWN_LAYER_STYLE`` (gray).

    Returns:
        A complete SVG document beginning with ``<?xml ... ?>``. Cells with
        zero polygons return a minimal valid document (no ``<polygon>``
        children, viewBox sized to the viewport).

    Raises:
        ValueError: If ``cell_name`` is not in ``lib.cells``.
    """
    cell = _find_cell(lib, cell_name)
    styles = layer_styles if layer_styles is not None else DEFAULT_SKY130_STYLES

    bb = _cell_bbox(cell)

    parts: List[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')

    if bb is None:
        # Empty cell: emit a minimal valid SVG document with the requested
        # viewport but no polygon children.
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width_px}" height="{height_px}" '
            f'viewBox="0 0 {width_px} {height_px}" '
            f'preserveAspectRatio="xMidYMid meet">'
        )
        parts.append(f"<title>{cell.name}</title>")
        parts.append("</svg>")
        return "\n".join(parts)

    x_min, y_min, x_max, y_max = bb
    vb_w = max(1, x_max - x_min)
    vb_h = max(1, y_max - y_min)

    # We want GDS-Y-up. Use a viewBox in GDS space then flip Y inside a group
    # transform. The viewBox covers ``[x_min, -y_max] .. [x_max, -y_min]``
    # after the flip, so the flipped group fits exactly inside the viewBox.
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="{x_min} {-y_max} {vb_w} {vb_h}" '
        f'preserveAspectRatio="xMidYMid meet">'
    )
    parts.append(f"<title>{cell.name}</title>")
    # Flip Y so positive-Y points up (GDS convention) while keeping the
    # polygon coordinates in their original integer DBU form.
    parts.append('<g transform="scale(1, -1)">')

    for lp in cell.polygons:
        if not lp.polygon.vertices:
            continue
        style = _resolve_style(lp, styles)
        points = _polygon_to_points_attr(lp)
        parts.append(
            f'<polygon points="{points}" '
            f'fill="{style.fill}" stroke="{style.stroke}" '
            f'fill-opacity="{style.fill_opacity}" '
            f'data-layer="{lp.layer}" data-datatype="{lp.datatype}" />'
        )

    parts.append("</g>")
    parts.append("</svg>")
    return "\n".join(parts)


def write_svg_to_file(
    lib: ParameterizedLibrary,
    cell_name: str,
    output_path: str,
    **kwargs,
) -> None:
    """Render ``cell_name`` and write the SVG document to ``output_path``.

    ``kwargs`` are forwarded to :func:`render_svg` (``width_px``,
    ``height_px``, ``layer_styles``).
    """
    svg = render_svg(lib, cell_name, **kwargs)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
