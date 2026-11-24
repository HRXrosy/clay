"""Compose PDK boundary metadata from source-owned and actual assembly extent.

Boundary polygons are not conductors. Imported primitive markers remain intact;
the one added rectangle describes the enclosing assembly. Its extent is replayed
from nonboundary artwork and the original device snapshot, never from the newly
generated boundary itself or from a desired density result.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import klayout.db as kdb

from clay.layout.placed_geometry import placed_point_nm


def _boundary_key(tech: Any):
    accessor = getattr(tech, "assembly_boundary_key", None)
    key = None if accessor is None else accessor()
    if key is None:
        return None
    if (type(key) is not tuple or len(key) != 2
            or any(type(value) is not int or not 0 <= value <= 65535 for value in key)):
        raise ValueError("assembly boundary layer must be an exact GDS layer key")
    return key


def _source_boundary_region(key, geometry, placement):
    if not isinstance(geometry, Mapping) or not isinstance(placement, Mapping):
        raise TypeError("assembly boundary requires original emitter geometry and placement")
    region = kdb.Region()
    for name, stamp in geometry.items():
        position = placement.get(name)
        if position is None:
            continue
        for ring in (getattr(stamp, "polys_by_layer", {}) or {}).get(key, ()):
            points = [kdb.Point(*placed_point_nm(float(x), float(y), position))
                      for x, y in ring]
            if len(points) < 3:
                raise ValueError("assembly boundary source marker is degenerate")
            polygon = kdb.Polygon(points)
            if polygon.area() <= 0:
                raise ValueError("assembly boundary source marker has no area")
            region.insert(polygon)
    region.merge()
    return region


def expected_assembly_boundary(layout, cell, tech, geometry, placement):
    """Return the minimal on-grid box of actual masks plus original markers."""
    key = _boundary_key(tech)
    if key is None:
        return None
    if layout.dbu != 0.001:
        raise ValueError("assembly boundary requires the canonical one-nanometre DBU")
    artwork = kdb.Region()
    for index in layout.layer_indices():
        info = layout.get_info(index)
        if (info.layer, info.datatype) != key:
            artwork += kdb.Region(cell.begin_shapes_rec(index))
    source = _source_boundary_region(key, geometry, placement)
    if artwork.is_empty():
        if not source.is_empty():
            raise ValueError("assembly boundary has source markers but no physical artwork")
        return None
    envelope = artwork.bbox()
    if not source.is_empty():
        envelope += source.bbox()
    from clay.layout.route.emit_gds import pdk_manufacturing_grid_nm

    grid = pdk_manufacturing_grid_nm(tech)
    if type(grid) is not int or grid <= 0:
        raise ValueError("assembly boundary manufacturing grid must be positive")
    # Round outwards only to the actual manufacturing grid. This preserves the
    # source-owned abutment extent even in ordinary, nonexact snapping flows.
    envelope = kdb.Box(
        (envelope.left // grid) * grid,
        (envelope.bottom // grid) * grid,
        -((-envelope.right) // grid) * grid,
        -((-envelope.top) // grid) * grid,
    )
    return key, envelope


def require_assembly_boundary(layout, cell, tech, geometry, placement):
    """Independently require exactly the canonical enclosing boundary union."""
    expected = expected_assembly_boundary(layout, cell, tech, geometry, placement)
    if expected is None:
        return
    key, box = expected
    index = layout.find_layer(*key)
    observed = (kdb.Region() if index is None
                else kdb.Region(cell.begin_shapes_rec(index)).merged())
    if not (observed ^ kdb.Region(box)).is_empty():
        raise ValueError("assembly boundary differs from actual artwork and original source extent")


def materialize_assembly_boundary(layout, cell, tech, geometry, placement):
    """Append boundary metadata while preserving every existing source polygon."""
    expected = expected_assembly_boundary(layout, cell, tech, geometry, placement)
    if expected is None:
        return
    key, box = expected
    index = layout.layer(*key)
    current = kdb.Region(cell.begin_shapes_rec(index)).merged()
    target = kdb.Region(box)
    if not (current - target).is_empty():
        raise ValueError("assembly boundary contains extent absent from original source and artwork")
    if not (current ^ target).is_empty():
        cell.shapes(index).insert(box)
    require_assembly_boundary(layout, cell, tech, geometry, placement)
