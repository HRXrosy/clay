"""Declarative relative-placement combinators (TED-inspired, fresh reimplement).

Clay already has *automatic* placement (``connectivity_placement.py``) and a
DBU-offset composer (``composer.py``). What it lacks is an explicit,
*declarative* row/col surface where a designer states the floorplan tree
directly: ``row(a, b, c, spacing=.., align=..)`` / ``col(...)``.

This module supplies that complementary surface. It operates purely on placed
geometry (``NamedCell`` leaves) and returns composed ``NamedCell`` results, so
the combinators nest arbitrarily (a row of cols, a col of rows, etc.).

Coordinate convention
----------------------
Everything is derived from child bounding boxes plus ``spacing`` — no geometry
is fabricated. All combinators work in integer DBU (the same unit the lowering
functions emit).

- ``row(*children, spacing, align)``  lays children **left to right**: each
  child's left edge is placed exactly ``spacing`` past the previous child's
  right edge (the first child keeps its own left edge). ``align`` controls
  vertical placement:

    * ``"bottom"`` — every child's bbox y-min is aligned to the row y-min.
    * ``"top"``    — every child's bbox y-max is aligned to the row y-max.
    * ``"center"`` — every child's bbox vertical center is aligned to the
      common vertical center.

- ``col(*children, spacing, align)``  lays children **bottom to top** (the
  first child is at the bottom; this is the chosen, documented direction —
  it matches the increasing-y / GDS convention so a ``col`` stacks upward).
  Each child's bottom edge is placed exactly ``spacing`` past the previous
  child's top edge. ``align`` controls horizontal placement:

    * ``"left"``   — every child's bbox x-min is aligned to the col x-min.
    * ``"right"``  — every child's bbox x-max is aligned to the col x-max.
    * ``"center"`` — every child's bbox horizontal center is aligned to the
      common horizontal center.

Because each combinator returns a ``NamedCell`` whose ``polygons`` are the
union of its translated children, nesting is just passing a combinator result
where a child is expected. Leaf overlaps never occur: children are tiled along
the major axis with non-negative ``spacing`` and only translated (never
scaled / overlapped) on the minor axis.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple, Union

from .backend import (
    LayerPolygon,
    NamedCell,
    Point,
    Polygon,
)

Child = NamedCell
BBox = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


class Align(Enum):
    """Alignment of children on the minor (cross) axis of a row/col."""

    # row minor axis = vertical
    BOTTOM = "bottom"
    TOP = "top"
    # col minor axis = horizontal
    LEFT = "left"
    RIGHT = "right"
    # shared
    CENTER = "center"


_ROW_ALIGNS = {Align.BOTTOM, Align.TOP, Align.CENTER}
_COL_ALIGNS = {Align.LEFT, Align.RIGHT, Align.CENTER}


def _coerce_align(align: Union[str, Align]) -> Align:
    if isinstance(align, Align):
        return align
    try:
        return Align(str(align).lower())
    except ValueError:
        raise ValueError(
            f"unknown align {align!r}; expected one of "
            f"{[a.value for a in Align]}"
        )


def _bbox(polys: List[LayerPolygon]) -> Optional[BBox]:
    xs = [v.x for lp in polys for v in lp.polygon.vertices]
    ys = [v.y for lp in polys for v in lp.polygon.vertices]
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _translate(polys: List[LayerPolygon], dx: int, dy: int) -> List[LayerPolygon]:
    """Return a fresh, deep-copied, translated polygon list (read-only on input)."""
    return [
        LayerPolygon(
            layer=lp.layer,
            datatype=lp.datatype,
            polygon=Polygon(
                vertices=[Point(v.x + dx, v.y + dy) for v in lp.polygon.vertices]
            ),
        )
        for lp in polys
    ]


def _copy_polys(polys: List[LayerPolygon]) -> List[LayerPolygon]:
    return _translate(polys, 0, 0)


def _as_named_cell(child: Child, what: str) -> NamedCell:
    if not isinstance(child, NamedCell):
        raise TypeError(
            f"{what} children must be NamedCell (leaf or combinator result), "
            f"got {type(child).__name__}"
        )
    return child


def _compose(
    children: Tuple[Child, ...],
    spacing: int,
    align: Align,
    *,
    axis: str,
    name: Optional[str],
) -> NamedCell:
    """Shared row/col engine.

    ``axis`` is "row" (major axis = x) or "col" (major axis = y).
    Empty children -> empty NamedCell. Single child -> a copy at its own origin.
    """
    if spacing < 0:
        raise ValueError("spacing must be >= 0 (negative spacing would overlap)")

    flat = [_as_named_cell(c, axis) for c in children]
    if not flat:
        return NamedCell(name=name or f"{axis}_empty", polygons=[])

    # Bounding box of each child; skip empty children for placement math but
    # still carry their (zero) polygons through.
    boxes: List[Optional[BBox]] = [_bbox(c.polygons) for c in flat]

    # Common minor-axis extent (computed from non-empty children).
    if axis == "row":
        minor_lo = min(b[1] for b in boxes if b is not None) if any(boxes) else 0
        minor_hi = max(b[3] for b in boxes if b is not None) if any(boxes) else 0
    else:
        minor_lo = min(b[0] for b in boxes if b is not None) if any(boxes) else 0
        minor_hi = max(b[2] for b in boxes if b is not None) if any(boxes) else 0
    minor_span = minor_hi - minor_lo

    out: List[LayerPolygon] = []
    # Major-axis cursor: where the *left/bottom* edge of the next child goes.
    cursor: Optional[int] = None

    for child, box in zip(flat, boxes):
        if box is None:
            # empty child: carry its (empty) polygons, no placement advance
            out.extend(_copy_polys(child.polygons))
            continue

        x_min, y_min, x_max, y_max = box

        if axis == "row":
            # major = x
            if cursor is None:
                dx = 0  # first child keeps its own left edge
            else:
                dx = cursor - x_min
            new_left = x_min + dx
            new_right = x_max + dx
            # minor = y alignment
            if align is Align.BOTTOM:
                dy = minor_lo - y_min
            elif align is Align.TOP:
                dy = minor_hi - y_max
            else:  # CENTER
                child_center = (y_min + y_max)
                common_center = (minor_lo + minor_hi)
                # multiply-by-2 avoids odd-division rounding asymmetry
                dy = (common_center - child_center) // 2
            out.extend(_translate(child.polygons, dx, dy))
            cursor = new_right + spacing
        else:
            # major = y (bottom-to-top: first child at bottom)
            if cursor is None:
                dy = 0
            else:
                dy = cursor - y_min
            new_bottom = y_min + dy
            new_top = y_max + dy
            # minor = x alignment
            if align is Align.LEFT:
                dx = minor_lo - x_min
            elif align is Align.RIGHT:
                dx = minor_hi - x_max
            else:  # CENTER
                child_center = (x_min + x_max)
                common_center = (minor_lo + minor_hi)
                dx = (common_center - child_center) // 2
            out.extend(_translate(child.polygons, dx, dy))
            cursor = new_top + spacing

    _ = minor_span  # documented intermediate; not otherwise needed
    default_name = name or (axis + "(" + ",".join(c.name for c in flat) + ")")
    return NamedCell(name=default_name, polygons=out)


def row(
    *children: Child,
    spacing: int = 0,
    align: Union[str, Align] = "bottom",
    name: Optional[str] = None,
) -> NamedCell:
    """Compose ``children`` left-to-right into a single ``NamedCell``.

    Each subsequent child's left edge is placed exactly ``spacing`` DBU past the
    previous child's right edge. ``align`` (``"bottom"`` / ``"top"`` /
    ``"center"``) controls vertical alignment on the cross axis.

    Children are ``NamedCell`` leaves or the result of another combinator, so
    rows of cols nest naturally. Returns a ``NamedCell`` whose polygons are the
    translated union of all children — no child polygons overlap.
    """
    a = _coerce_align(align)
    if a not in _ROW_ALIGNS:
        raise ValueError(
            f"row align must be one of {[x.value for x in _ROW_ALIGNS]}, got {a.value!r}"
        )
    return _compose(children, int(spacing), a, axis="row", name=name)


def col(
    *children: Child,
    spacing: int = 0,
    align: Union[str, Align] = "left",
    name: Optional[str] = None,
) -> NamedCell:
    """Compose ``children`` bottom-to-top into a single ``NamedCell``.

    The first child sits at the bottom; each subsequent child's bottom edge is
    placed exactly ``spacing`` DBU past the previous child's top edge (the
    column stacks **upward**, matching the increasing-y GDS convention).
    ``align`` (``"left"`` / ``"right"`` / ``"center"``) controls horizontal
    alignment on the cross axis.

    Children nest like ``row``. Returns a ``NamedCell`` whose polygons are the
    translated union of all children — no child polygons overlap.
    """
    a = _coerce_align(align)
    if a not in _COL_ALIGNS:
        raise ValueError(
            f"col align must be one of {[x.value for x in _COL_ALIGNS]}, got {a.value!r}"
        )
    return _compose(children, int(spacing), a, axis="col", name=name)


def bbox_of(cell: NamedCell) -> Optional[BBox]:
    """Public helper: bounding box of a (composed) cell, or None if empty."""
    return _bbox(cell.polygons)
