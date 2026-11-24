"""Phase 8d — routing congestion estimator.

Estimates routing congestion for a layout of wire polygons by:

1. Classifying each ``LayerPolygon`` (assumed axis-aligned rectangle) as
   horizontal or vertical based on its bounding-box aspect ratio
   (``width > height`` -> horizontal; ``height > width`` -> vertical;
   square wires are skipped because their channel axis is ambiguous).

2. Bucketing horizontal wires by midpoint-y rounded to ``channel_pitch_nm``
   and vertical wires by midpoint-x rounded to ``channel_pitch_nm``.
   The bucket coordinate is the integer channel index times ``channel_pitch_nm``.

3. Counting crossings as the number of horizontal-vertical wire pairs that
   spatially overlap and live on the same GDS layer/datatype.

This module is intentionally read-only: it does not mutate the input
polygons and does not depend on any routing primitive beyond
``LayerPolygon`` from :mod:`clay.layout.backend`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .backend import LayerPolygon


__all__ = [
    "ChannelOccupancy",
    "CongestionReport",
    "estimate_congestion",
]


@dataclass(frozen=True, eq=True)
class ChannelOccupancy:
    """One axis-aligned routing channel and its wire count.

    Attributes:
        axis: ``"horizontal"`` for an east-west channel (constant y) or
            ``"vertical"`` for a north-south channel (constant x).
        channel_coord: the bucket coordinate in nm DBU. For a horizontal
            channel this is the channel's y; for a vertical channel its x.
            Equal to ``int(round(midpoint / channel_pitch_nm)) * channel_pitch_nm``.
        wire_count: number of wires bucketed into this channel.
    """

    axis: str
    channel_coord: int
    wire_count: int


@dataclass(eq=True)
class CongestionReport:
    """Aggregated congestion estimate.

    Attributes:
        horizontal_channels: occupancy entries sorted by ``channel_coord``.
        vertical_channels: occupancy entries sorted by ``channel_coord``.
        crossings: number of horizontal-vertical wire crossings on the same
            GDS ``(layer, datatype)``. Each crossing counts once per
            (horizontal_wire, vertical_wire) overlapping pair.
    """

    horizontal_channels: List[ChannelOccupancy] = field(default_factory=list)
    vertical_channels: List[ChannelOccupancy] = field(default_factory=list)
    crossings: int = 0


def _bbox(wire: LayerPolygon) -> Tuple[int, int, int, int]:
    """Return ``(x_min, y_min, x_max, y_max)`` for the polygon."""
    verts = wire.polygon.vertices
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def _classify_axis(bb: Tuple[int, int, int, int]) -> str:
    """Return ``"horizontal"``, ``"vertical"``, or ``"skip"`` (ambiguous)."""
    x_min, y_min, x_max, y_max = bb
    dx = x_max - x_min
    dy = y_max - y_min
    if dx > dy:
        return "horizontal"
    if dy > dx:
        return "vertical"
    return "skip"


def _bucket(coord: float, pitch: int) -> int:
    """Round ``coord`` to the nearest multiple of ``pitch`` (channel snap)."""
    if pitch <= 0:
        raise ValueError(f"channel_pitch_nm must be positive, got {pitch}")
    return int(round(coord / pitch)) * pitch


def _rectangles_overlap(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> bool:
    """Strict open-set rectangle overlap: requires positive area intersection.

    Edge-touching rectangles (one's max == other's min) do not count as
    overlapping; only true area overlap is considered a crossing.
    """
    ax_min, ay_min, ax_max, ay_max = a
    bx_min, by_min, bx_max, by_max = b
    if ax_max <= bx_min or bx_max <= ax_min:
        return False
    if ay_max <= by_min or by_max <= ay_min:
        return False
    return True


def estimate_congestion(
    wires: List[LayerPolygon], channel_pitch_nm: int = 1000
) -> CongestionReport:
    """Bucket wire segments into channels and count crossings.

    Args:
        wires: a flat list of routing wires represented as axis-aligned
            rectangle polygons. Each ``LayerPolygon`` is classified by its
            bounding-box aspect ratio. Square polygons (dx == dy) are
            skipped because their channel axis is ambiguous; they also do
            not participate in crossing counts.
        channel_pitch_nm: bucket size in nm DBU. Larger pitch coalesces
            nearby wires into the same channel.

    Returns:
        A :class:`CongestionReport` with channels sorted by ``channel_coord``
        and the number of cross-axis polygon overlaps on the same GDS
        ``(layer, datatype)``.
    """
    if channel_pitch_nm <= 0:
        raise ValueError(
            f"channel_pitch_nm must be positive, got {channel_pitch_nm}"
        )

    horizontal: List[Tuple[LayerPolygon, Tuple[int, int, int, int]]] = []
    vertical: List[Tuple[LayerPolygon, Tuple[int, int, int, int]]] = []

    h_counts: Dict[int, int] = {}
    v_counts: Dict[int, int] = {}

    for wire in wires:
        verts = wire.polygon.vertices
        if not verts:
            continue
        bb = _bbox(wire)
        axis = _classify_axis(bb)
        if axis == "skip":
            continue
        x_min, y_min, x_max, y_max = bb
        if axis == "horizontal":
            mid_y = (y_min + y_max) / 2.0
            coord = _bucket(mid_y, channel_pitch_nm)
            h_counts[coord] = h_counts.get(coord, 0) + 1
            horizontal.append((wire, bb))
        else:  # vertical
            mid_x = (x_min + x_max) / 2.0
            coord = _bucket(mid_x, channel_pitch_nm)
            v_counts[coord] = v_counts.get(coord, 0) + 1
            vertical.append((wire, bb))

    horizontal_channels = [
        ChannelOccupancy(axis="horizontal", channel_coord=c, wire_count=n)
        for c, n in sorted(h_counts.items())
    ]
    vertical_channels = [
        ChannelOccupancy(axis="vertical", channel_coord=c, wire_count=n)
        for c, n in sorted(v_counts.items())
    ]

    crossings = 0
    for h_wire, h_bb in horizontal:
        for v_wire, v_bb in vertical:
            if h_wire.layer != v_wire.layer:
                continue
            if h_wire.datatype != v_wire.datatype:
                continue
            if _rectangles_overlap(h_bb, v_bb):
                crossings += 1

    return CongestionReport(
        horizontal_channels=horizontal_channels,
        vertical_channels=vertical_channels,
        crossings=crossings,
    )
