"""Phase 13 mixed-cell composer.

Composes multiple ``ParameterizedLibrary`` outputs from different cell
template factories into a single top-level ``ParameterizedLibrary`` using
relative coordinate placement.

Design notes
------------
- Each sub-library contributes ``NamedCell`` entries. We translate every
  polygon vertex by the given ``(offset_x_dbu, offset_y_dbu)`` and rename
  the cell with ``name_prefix`` so multiple placements of the same template
  do not collide on cell name.
- The composer does NOT introspect annotations or rerun lowering — it
  operates purely on the produced GDS-like geometry. Callers lower with
  ``lower_annotated_cell_drc_complete`` (or any other lowering function
  that produces a ``ParameterizedLibrary``) first.
- Layer / datatype values are preserved verbatim. Per-cell polygon counts
  are preserved.
- An empty placement list yields an empty top-level library with the
  caller-provided top_name and the default unit settings of
  ``ParameterizedLibrary``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .backend import (
    LayerPolygon,
    NamedCell,
    ParameterizedLibrary,
    Point,
    Polygon,
)


@dataclass
class PlacedSubcell:
    """A sub-library placed at an integer DBU offset, with an optional cell-name prefix."""

    sublib: ParameterizedLibrary
    offset_x_dbu: int
    offset_y_dbu: int
    name_prefix: str = ""


def _translate_polygon(p: Polygon, dx: int, dy: int) -> Polygon:
    return Polygon(vertices=[Point(v.x + dx, v.y + dy) for v in p.vertices])


def _translate_layer_polygon(lp: LayerPolygon, dx: int, dy: int) -> LayerPolygon:
    return LayerPolygon(
        layer=lp.layer,
        datatype=lp.datatype,
        polygon=_translate_polygon(lp.polygon, dx, dy),
    )


def compose_cells(top_name: str, placements: List[PlacedSubcell]) -> ParameterizedLibrary:
    """Compose multiple sub-libraries into a single ``ParameterizedLibrary``.

    For each ``PlacedSubcell`` in ``placements``, every cell in
    ``sublib.cells`` is copied into the resulting library with:
    - cell name = ``f"{name_prefix}{cell.name}"`` when ``name_prefix`` is
      non-empty, else the original ``cell.name``
    - every polygon vertex translated by ``(offset_x_dbu, offset_y_dbu)``
    - layer / datatype values preserved
    - per-cell polygon count preserved

    The returned library uses ``name=top_name`` and default unit settings
    inherited from ``ParameterizedLibrary``. An empty ``placements`` list
    returns a library with zero cells.
    """
    composed = ParameterizedLibrary(name=top_name)
    for placement in placements:
        dx = placement.offset_x_dbu
        dy = placement.offset_y_dbu
        prefix = placement.name_prefix
        for cell in placement.sublib.cells:
            new_name = f"{prefix}{cell.name}" if prefix else cell.name
            translated_polygons = [_translate_layer_polygon(lp, dx, dy) for lp in cell.polygons]
            composed.add_cell(NamedCell(name=new_name, polygons=translated_polygons))
    return composed
