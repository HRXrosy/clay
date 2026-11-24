"""PDK-derived C/B/E row buses for actual occupied BJT unit cells."""
from __future__ import annotations

import math

from clay.layout.backend import LayerPolygon, Point, Polygon
from clay.layout.device_cell import Terminal


def _rect(key, x0, y0, x1, y1):
    return LayerPolygon(layer=key[0], datatype=key[1],
        polygon=Polygon.rectangle(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))


def _sparse_spines(tech_geometry, placements, width, spacing):
    """Separate M4 tracks beyond the fixed units, including their via pads."""
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    grid = pdk_manufacturing_grid_nm(tech_geometry.tech)
    pad = max(tech_geometry.via_pad(3), tech_geometry.metal_min_width(4))
    pitch = pad + tech_geometry.metal_min_space(4)
    pitch = ((pitch + grid - 1) // grid) * grid
    right = max(column for _row, column, _orientation in placements) * (width + spacing) + width
    start = right + tech_geometry.metal_min_space(4) + (pad + 1) // 2
    start = ((start + grid - 1) // grid) * grid
    return tuple(start + index * pitch for index in range(3))


def _legacy_spines_are_safe(tech_geometry, placements, landing_offsets, width, spacing):
    """Check retained row reach and separation of the original M4 tracks."""
    row_columns = {}
    for row, column, _orientation in placements:
        row_columns.setdefault(row, []).append(column)
    if len(row_columns) < 2:
        return True
    first = row_columns[min(row_columns)]
    pad_half = tech_geometry.via_pad(3) // 2
    spines = []
    for index, role in enumerate(("c", "b", "e")):
        landing_x, _landing_y = landing_offsets[role]
        left, right = first[0] * (width + spacing) + landing_x, first[-1] * (width + spacing) + landing_x
        spine = left + (right - left) * (index + 1) // 4
        for columns in row_columns.values():
            row_left = columns[0] * (width + spacing) + landing_x
            row_right = columns[-1] * (width + spacing) + landing_x
            if spine + pad_half <= row_left or spine - pad_half >= row_right:
                return False
        spines.append(spine)
    minimum_pitch = max(tech_geometry.via_pad(3), tech_geometry.metal_min_width(4)) + tech_geometry.metal_min_space(4)
    ordered = sorted(spines)
    return all(second - first >= minimum_pitch for first, second in zip(ordered, ordered[1:]))


def build_bjt_bank_interconnect(*, pdk_id, landing_offsets, placements,
                               width, height, spacing, legacy_layout):
    """Return routing polygons and external access for the occupied cells.

    A safe legacy grid retains its original polygon sequence. A new sparse grid
    uses distinct outside-bank tracks and extends every occupied row to its
    corresponding track, including rows containing a single remote unit.
    """
    from clay.layout.tech_geometry import TechGeometry

    tech_geometry = TechGeometry.for_pdk(pdk_id)
    m3_key, m4_key = tech_geometry.metal_key(3), tech_geometry.metal_key(4)
    m3_half = int(math.ceil(tech_geometry.metal_min_width(3) / 2.0))
    m4_half = int(math.ceil(tech_geometry.metal_min_width(4) / 2.0))
    occupied_rows = tuple(sorted({row for row, _column, _orientation in placements}))
    safe_legacy = legacy_layout and _legacy_spines_are_safe(
        tech_geometry, placements, landing_offsets, width, spacing)
    sparse_spines = None if safe_legacy else _sparse_spines(tech_geometry, placements, width, spacing)
    polygons, exposed, layers = [], {}, {}
    for role_index, role in enumerate(("c", "b", "e")):
        landing_x, landing_y = landing_offsets[role]
        row_buses = []
        for row in occupied_rows:
            centres = [Point(column * (width + spacing) + landing_x,
                             row * (height + spacing) + landing_y)
                       for cell_row, column, _orientation in placements if cell_row == row]
            for centre in centres:
                polygons.extend(tech_geometry.via_stack(1, centre))
                polygons.extend(tech_geometry.via_stack(2, centre))
            bus_y = int(centres[0].y)
            left, right = int(centres[0].x), int(centres[-1].x)
            if sparse_spines is not None and len(occupied_rows) > 1:
                right = max(right, sparse_spines[role_index])
            polygons.append(_rect(m3_key, left, bus_y - m3_half, right, bus_y + m3_half))
            row_buses.append((left, right, bus_y))
        if len(occupied_rows) > 1:
            left, right, _first_y = row_buses[0]
            spine_x = (left + (right - left) * (role_index + 1) // 4
                       if sparse_spines is None else sparse_spines[role_index])
            for _left, _right, bus_y in row_buses:
                polygons.extend(tech_geometry.via_stack(3, Point(spine_x, bus_y)))
            polygons.append(_rect(m4_key, spine_x - m4_half, row_buses[0][2],
                                  spine_x + m4_half, row_buses[-1][2]))
            external = Terminal(role=role, bbox=(spine_x, row_buses[-1][2], spine_x, row_buses[-1][2]))
        else:
            left, _right, bus_y = row_buses[0]
            external = Terminal(role=role, bbox=(left, bus_y, left, bus_y))
        exposed[role] = (external,)
        layers[role] = m3_key
    return polygons, exposed, layers


__all__ = ["build_bjt_bank_interconnect"]
