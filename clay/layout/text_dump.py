"""Human-readable text dump of a ParameterizedLibrary.

Sibling of ``emit_ascii_gds`` in ``clay/layout/backend.py``. Where ASCII GDS
is a faithful round-trip serialization (and matches the Rust mirror byte for
byte), this module produces a decorated, indented dump intended for code
review and debugging — coordinates are converted from integer nm (database
units) to micrometers, layers are sorted by ``(layer, datatype)``, and cells
are sorted alphabetically by name.

The dump never mutates the library; it only walks ``ParameterizedLibrary`` /
``NamedCell`` / ``LayerPolygon`` / ``Polygon`` / ``Point``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .backend import LayerPolygon, NamedCell, ParameterizedLibrary, Point


@dataclass(frozen=True)
class TextDumpConfig:
    """Knobs for ``dump_library_to_text`` / ``dump_cell_to_text``."""

    indent_size: int = 2
    decimal_places: int = 3
    include_polygon_vertices: bool = True
    show_polygon_summary: bool = True


def _format_um(value_nm: int, decimal_places: int) -> str:
    """Convert an integer-nm coordinate to a µm string with fixed precision."""
    return f"{value_nm / 1000.0:.{decimal_places}f}"


def _format_vertex(point: Point, decimal_places: int) -> str:
    return f"({_format_um(point.x, decimal_places)}, {_format_um(point.y, decimal_places)})"


def _group_by_layer(
    polygons: Iterable[LayerPolygon],
) -> List[Tuple[Tuple[int, int], List[LayerPolygon]]]:
    """Group LayerPolygons by ``(layer, datatype)`` and sort the groups."""
    groups: dict = {}
    for lp in polygons:
        key = (lp.layer, lp.datatype)
        groups.setdefault(key, []).append(lp)
    return sorted(groups.items(), key=lambda item: item[0])


def _render_cell_body(
    cell: NamedCell,
    config: TextDumpConfig,
    base_indent_level: int,
) -> List[str]:
    """Render the body of a cell (layer groups + polygons) at the given indent.

    ``base_indent_level`` is the number of ``indent_size``-wide indent steps to
    apply to the *layer* lines. Polygon lines sit one step deeper.
    """
    lines: List[str] = []
    indent = " " * (config.indent_size * base_indent_level)
    poly_indent = " " * (config.indent_size * (base_indent_level + 1))

    for (layer, datatype), group in _group_by_layer(cell.polygons):
        header = f"{indent}Layer ({layer}, {datatype}):"
        if config.show_polygon_summary:
            total_verts = sum(len(lp.polygon.vertices) for lp in group)
            header += f"  {len(group)} polygons, {total_verts} total vertices"
        lines.append(header)

        if config.include_polygon_vertices:
            for i, lp in enumerate(group):
                verts = lp.polygon.vertices
                vert_str = ", ".join(
                    _format_vertex(v, config.decimal_places) for v in verts
                )
                lines.append(
                    f"{poly_indent}Polygon {i} [{len(verts)} verts]: {vert_str}"
                )
    return lines


def dump_cell_to_text(
    cell: NamedCell, config: TextDumpConfig = TextDumpConfig()
) -> str:
    """Render a single ``NamedCell`` as indented text (no Library header).

    The cell header sits at indent level 0; layers at level 1; polygons at
    level 2.
    """
    lines: List[str] = [f"Cell: {cell.name}"]
    lines.extend(_render_cell_body(cell, config, base_indent_level=1))
    return "\n".join(lines) + "\n"


def dump_library_to_text(
    lib: ParameterizedLibrary, config: TextDumpConfig = TextDumpConfig()
) -> str:
    """Render a ``ParameterizedLibrary`` as indented human-readable text.

    Layout::

        Library: <lib_name> (db=1e-9, user=1e-6)
          Cell: <cell_name_a>
            Layer (66, 20):  3 polygons, 24 total vertices
              Polygon 0 [6 verts]: (0.000, 0.000), (0.500, 0.000), ...
          Cell: <cell_name_b>
            ...

    Cells are sorted alphabetically; within each cell, layers are sorted by
    ``(layer, datatype)``. Coordinates are converted from integer-nm database
    units to micrometers using ``config.decimal_places``.
    """
    lines: List[str] = [
        f"Library: {lib.name} (db={lib.unit_db_meters!r}, user={lib.unit_user!r})"
    ]

    cell_indent = " " * config.indent_size
    sorted_cells = sorted(lib.cells, key=lambda c: c.name)
    for cell in sorted_cells:
        lines.append(f"{cell_indent}Cell: {cell.name}")
        lines.extend(_render_cell_body(cell, config, base_indent_level=2))

    return "\n".join(lines) + "\n"
