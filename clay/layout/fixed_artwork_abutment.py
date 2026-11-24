"""Align full facing SD boundaries using local channel baselines.

Inputs are already oriented local polygons. No observed placement or pitch is
accepted. This horizontal-interface operation proves its geometric contact
only; callers retain source-net/family/body, all-layer spacing and complete
joint terminal replay obligations.
"""
from dataclasses import dataclass

import klayout.db as kdb

from clay.layout.physical_terminal_inventory import PhysicalTerminalPolygonNM


@dataclass(frozen=True, slots=True)
class FixedArtworkInterfaceAlignment:
    translation_nm: tuple[int, int]
    common_edges_nm: tuple[tuple[int, int, int, int], ...]
    overlap_area_nm2: int = 0
    scope: str = 'local_interface_geometry_only'


def _require(ok, message):
    if not ok:
        raise ValueError('fixed artwork interface ' + message)


def _region(shapes, label, checkpoint):
    _require(type(shapes) is tuple and 0 < len(shapes) <= 256,
             label + ' polygons must be a nonempty bounded exact tuple')
    result, vertices = kdb.Region(), 0
    for shape in shapes:
        checkpoint()
        _require(type(shape) is PhysicalTerminalPolygonNM, label + ' polygon type differs')
        shape.__post_init__()
        vertices += len(shape.hull_nm) + sum(map(len, shape.holes_nm))
        _require(vertices <= 16384, label + ' polygon vertex bound exceeded')
        result += shape.to_region()
    checkpoint()
    result.merge()
    _require(not result.is_empty() and result.area() > 0, label + ' must have positive area')
    return result


def _facing_intervals(region, x, checkpoint):
    intervals = []
    for polygon in region.each():
        checkpoint()
        for edge in polygon.each_edge():
            if edge.p1.x == edge.p2.x == x and edge.p1.y != edge.p2.y:
                intervals.append(tuple(sorted((edge.p1.y, edge.p2.y))))
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def align_fixed_artwork_abutment(*, left_site, right_site, left_channel, right_channel, checkpoint):
    """Translate the right unit to the full right/left-facing local interface.

The same translation must align both channel baselines. Facing boundaries
must coincide in full and cover the complete channel height; corner contact,
partial contact and an SD site on the opposite side are rejected. Extra raw
master envelope extent does not influence this construction translation.
"""
    _require(callable(checkpoint), 'checkpoint must be callable')
    checkpoint()
    left = _region(left_site, 'left site', checkpoint)
    right = _region(right_site, 'right site', checkpoint)
    lchannel = _region(left_channel, 'left channel', checkpoint)
    rchannel = _region(right_channel, 'right channel', checkpoint)
    lb, rb = left.bbox(), right.bbox()
    lc, rc = lchannel.bbox(), rchannel.bbox()
    _require(lc.height() == rc.height(), 'channel baselines have different spans')
    _require(lb.left >= lc.right and rb.right <= rc.left,
             'raw site is not on the declared facing side of its channel')
    _require((left & lchannel).is_empty() and (right & rchannel).is_empty(),
             'raw site overlaps its channel')
    dx, dy = lb.right-rb.left, lc.bottom-rc.bottom
    _require(abs(dx) <= 1_000_000_000 and abs(dy) <= 1_000_000_000,
             'derived translation exceeds coordinate bounds')
    moved = right.transformed(kdb.Trans(dx,dy))
    checkpoint()
    a = _facing_intervals(left, lb.right, checkpoint)
    b = _facing_intervals(moved, lb.right, checkpoint)
    _require(a and a == b, 'full facing boundary profiles do not coincide')
    _require(any(lo <= lc.bottom and hi >= lc.top for lo,hi in a),
             'facing boundary does not cover the complete channel baseline span')
    overlap = int((left & moved).area())
    _require(overlap == 0, 'raw SD sites overlap instead of abutting')
    shifted_channel = rchannel.transformed(kdb.Trans(dx,dy))
    _require(lc.right < shifted_channel.bbox().left and (lchannel & shifted_channel).is_empty(),
             'constructed channel order or separation differs')
    checkpoint()
    return FixedArtworkInterfaceAlignment((dx,dy), tuple((lb.right,lo,lb.right,hi) for lo,hi in a), overlap)
