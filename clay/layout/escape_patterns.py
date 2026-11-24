"""Router escape pattern library.

Predefined M1 routing patterns for common analog cell template I/O escapes:
differential-pair gate inputs, current-mirror reference input, and power-grid
tap stubs (VDD/VSS).

These helpers emit M1 wire-stub polygons on the cell boundary so the
upstream router can later stitch cells together without re-deriving the
escape geometry for every template. The patterns are intentionally
parameterized via ``EscapePattern`` so callers can override pin count,
pitch, and wire width without changing the call site.

API:
- ``EscapePattern``: frozen dataclass describing pin count / pitch / width.
- ``diff_pair_input_escape``: 2 horizontal M1 stubs at gate positions on
  the cell's left edge (M_in_p / M_in_n).
- ``current_mirror_input_escape``: 1 horizontal M1 stub for I_ref on the
  left edge.
- ``power_grid_tap_escape``: 1 vertical M1 stub from cell center to the
  nearest VDD (top) or VSS (bottom) rail.

All stubs are emitted on the M1 layer; the layer/datatype is resolved via
``pdk_layers.gds_layer(LogicalLayer.METAL1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .backend import (
    LayerMap,
    LayerPolygon,
    LogicalLayer,
    Polygon,
)


# --- Pattern descriptor -----------------------------------------------------


@dataclass(frozen=True, eq=True)
class EscapePattern:
    """Describes the geometric envelope of a router escape pattern.

    Attributes:
        name: Stable identifier for the pattern (e.g. ``"diff_pair_input"``).
        pin_count: Number of escape pins this pattern produces.
        pin_pitch_nm: Centre-to-centre spacing between pins on the cell
            boundary, in nm. Use 0 for single-pin patterns.
        wire_width_nm: Width of every M1 stub, in nm.
    """

    name: str
    pin_count: int
    pin_pitch_nm: int
    wire_width_nm: int


# --- Default stub geometry --------------------------------------------------

# Default escape stub length (nm). The stub protrudes by this amount from
# the cell boundary so the upstream router has somewhere to land a via or
# continue a wire.
_DEFAULT_STUB_LENGTH_NM = 400


def _metal1_layer(pdk_layers: LayerMap):
    metal1 = pdk_layers.gds_layer(LogicalLayer.METAL1)
    if metal1 is None:
        raise ValueError("PDK does not define METAL1 layer")
    return metal1


def _horizontal_stub(
    *,
    right_edge_x: int,
    left_edge_x: int,
    centre_y: int,
    width_nm: int,
    pdk_layers: LayerMap,
) -> LayerPolygon:
    """Horizontal M1 rectangle straddling ``centre_y`` between two X coords."""
    metal1 = _metal1_layer(pdk_layers)
    half_w = width_nm // 2
    return LayerPolygon(
        layer=metal1.layer,
        datatype=metal1.datatype,
        polygon=Polygon.rectangle(
            min(left_edge_x, right_edge_x),
            centre_y - half_w,
            max(left_edge_x, right_edge_x),
            centre_y + half_w,
        ),
    )


def _vertical_stub(
    *,
    centre_x: int,
    bottom_y: int,
    top_y: int,
    width_nm: int,
    pdk_layers: LayerMap,
) -> LayerPolygon:
    """Vertical M1 rectangle straddling ``centre_x`` between two Y coords."""
    metal1 = _metal1_layer(pdk_layers)
    half_w = width_nm // 2
    return LayerPolygon(
        layer=metal1.layer,
        datatype=metal1.datatype,
        polygon=Polygon.rectangle(
            centre_x - half_w,
            min(bottom_y, top_y),
            centre_x + half_w,
            max(bottom_y, top_y),
        ),
    )


# --- Pattern: differential-pair gate inputs ---------------------------------


def diff_pair_input_escape(
    cell_left_edge_x: int,
    cell_bottom_y: int,
    cell_height: int,
    pdk_layers: LayerMap,
    pattern: EscapePattern = EscapePattern("diff_pair_input", 2, 1000, 200),
) -> List[LayerPolygon]:
    """Return M1 wire stubs escaping from M_in_p / M_in_n gates at cell left edge.

    The escape pins are vertically centred on the cell, symmetric around
    ``cell_bottom_y + cell_height / 2`` and separated by ``pattern.pin_pitch_nm``.
    Each stub protrudes ``_DEFAULT_STUB_LENGTH_NM`` to the left of the cell
    boundary so the global router can pick it up.
    """
    if pattern.pin_count <= 0:
        raise ValueError(f"pattern.pin_count must be positive, got {pattern.pin_count}")
    if pattern.wire_width_nm <= 0:
        raise ValueError(
            f"pattern.wire_width_nm must be positive, got {pattern.wire_width_nm}"
        )
    centre_y = cell_bottom_y + cell_height // 2
    # Distribute pins symmetrically about the cell centre.
    offsets = _symmetric_offsets(pattern.pin_count, pattern.pin_pitch_nm)
    polys: List[LayerPolygon] = []
    stub_left = cell_left_edge_x - _DEFAULT_STUB_LENGTH_NM
    for dy in offsets:
        polys.append(
            _horizontal_stub(
                left_edge_x=stub_left,
                right_edge_x=cell_left_edge_x,
                centre_y=centre_y + dy,
                width_nm=pattern.wire_width_nm,
                pdk_layers=pdk_layers,
            )
        )
    return polys


# --- Pattern: current-mirror reference input --------------------------------


def current_mirror_input_escape(
    cell_left_edge_x: int,
    cell_bottom_y: int,
    cell_height: int,
    pdk_layers: LayerMap,
    pattern: EscapePattern = EscapePattern("cmir_input", 1, 800, 200),
) -> List[LayerPolygon]:
    """Return M1 wire stub for reference current input on left edge.

    Single-pin pattern by default (``pin_count = 1``). If the caller passes
    a multi-pin ``EscapePattern`` (e.g. for cascoded mirror references), the
    pins are distributed symmetrically about the cell vertical centre at
    ``pattern.pin_pitch_nm`` spacing.
    """
    if pattern.pin_count <= 0:
        raise ValueError(f"pattern.pin_count must be positive, got {pattern.pin_count}")
    if pattern.wire_width_nm <= 0:
        raise ValueError(
            f"pattern.wire_width_nm must be positive, got {pattern.wire_width_nm}"
        )
    centre_y = cell_bottom_y + cell_height // 2
    offsets = _symmetric_offsets(pattern.pin_count, pattern.pin_pitch_nm)
    stub_left = cell_left_edge_x - _DEFAULT_STUB_LENGTH_NM
    polys: List[LayerPolygon] = []
    for dy in offsets:
        polys.append(
            _horizontal_stub(
                left_edge_x=stub_left,
                right_edge_x=cell_left_edge_x,
                centre_y=centre_y + dy,
                width_nm=pattern.wire_width_nm,
                pdk_layers=pdk_layers,
            )
        )
    return polys


# --- Pattern: power-grid tap stub -------------------------------------------


def power_grid_tap_escape(
    cell_center_x: int,
    cell_top_y: int,
    cell_bottom_y: int,
    rail_position: str,
    pdk_layers: LayerMap,
    pattern: EscapePattern = EscapePattern("pg_tap", 1, 0, 600),
) -> List[LayerPolygon]:
    """Vertical M1 stub from cell to nearest VDD/VSS rail position.

    For ``rail_position == "vdd"`` the stub runs from the cell vertical
    midpoint up to ``cell_top_y``; for ``"vss"`` it runs from the midpoint
    down to ``cell_bottom_y``. Multi-pin patterns spread pins horizontally
    around ``cell_center_x`` at ``pattern.pin_pitch_nm`` spacing.
    """
    if pattern.pin_count <= 0:
        raise ValueError(f"pattern.pin_count must be positive, got {pattern.pin_count}")
    if pattern.wire_width_nm <= 0:
        raise ValueError(
            f"pattern.wire_width_nm must be positive, got {pattern.wire_width_nm}"
        )
    if cell_top_y <= cell_bottom_y:
        raise ValueError(
            f"cell_top_y ({cell_top_y}) must be greater than cell_bottom_y "
            f"({cell_bottom_y})"
        )
    rail_norm = rail_position.lower()
    if rail_norm not in ("vdd", "vss"):
        raise ValueError(
            f"rail_position must be 'vdd' or 'vss', got {rail_position!r}"
        )
    midpoint_y = (cell_top_y + cell_bottom_y) // 2
    if rail_norm == "vdd":
        bottom_y, top_y = midpoint_y, cell_top_y
    else:
        bottom_y, top_y = cell_bottom_y, midpoint_y
    offsets = _symmetric_offsets(pattern.pin_count, pattern.pin_pitch_nm)
    polys: List[LayerPolygon] = []
    for dx in offsets:
        polys.append(
            _vertical_stub(
                centre_x=cell_center_x + dx,
                bottom_y=bottom_y,
                top_y=top_y,
                width_nm=pattern.wire_width_nm,
                pdk_layers=pdk_layers,
            )
        )
    return polys


# --- Helpers ----------------------------------------------------------------


def _symmetric_offsets(pin_count: int, pitch_nm: int) -> List[int]:
    """Return ``pin_count`` offsets symmetric about 0 with ``pitch_nm`` spacing.

    Examples:
        pin_count=1 -> [0]
        pin_count=2 -> [-pitch/2, +pitch/2]
        pin_count=3 -> [-pitch, 0, +pitch]
        pin_count=4 -> [-3*pitch/2, -pitch/2, +pitch/2, +3*pitch/2]
    """
    if pin_count == 1:
        return [0]
    half_pitch = pitch_nm // 2
    offsets: List[int] = []
    if pin_count % 2 == 0:
        # Even pins: half-pitch on each side, then full-pitch steps outward.
        k = pin_count // 2
        for i in range(k):
            offsets.append(-(2 * i + 1) * half_pitch)
            offsets.append((2 * i + 1) * half_pitch)
    else:
        # Odd pins: centre pin at 0, then full-pitch steps outward.
        k = pin_count // 2
        offsets.append(0)
        for i in range(1, k + 1):
            offsets.append(-i * pitch_nm)
            offsets.append(i * pitch_nm)
    offsets.sort()
    return offsets
