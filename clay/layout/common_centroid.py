"""Common-centroid / interdigitated placement for matched device pairs.

A matched differential pair (declared via ``Matching`` / ``Symmetric``
annotations, see ``clay/layout/annotations.py``) lays out best when the two
devices' fingers are *interleaved* along a common axis so that linear (and, for
ABBA, even-order) process gradients average out and the two devices share a
single geometric centroid.

This module is a fresh reimplementation of the idea behind TED's
``PairedMosfet(mode="cross")``; it operates purely on the committed Phase 3a
finger geometry (``LayerPolygon`` lists from ``lower_mos_finger_geometry``) and
does not depend on or copy TED.

Two patterns are supported:

* ``cross`` (ABAB): centro-symmetric alternation. The first half alternates
  ``A B A B ...`` and the second half is its mirror ``... B A B A``, i.e. the
  full sequence reads ``A B A B | B A B A``. This is the standard
  common-centroid interpretation of a cross arrangement: the mirror makes the
  two devices' centroids *coincide* (a bare, non-mirrored ``ABABAB`` would
  cancel the linear gradient but leave the centroids offset by half a pitch).
  Requires equal finger counts.
* ``abba`` (ABBA): ``A B B A`` repeated per group of 4 positions. Even-order
  (quadratic) gradient also cancels and the centroids coincide; requires
  equal, even finger counts.

The core entry point is :func:`interdigitate_pair`. It takes the two devices'
finger geometry, re-orders the fingers into a single shared column at a uniform
pitch, and returns the interleaved geometry plus a centroid-coincidence check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from .annotations import InterdigitationPattern
from .backend import LayerPolygon, NamedCell, Point, Polygon


class CentroidPattern(Enum):
    """Interdigitation finger-ordering pattern.

    ``CROSS`` (ABAB) cancels linear gradients; ``ABBA`` additionally cancels
    even-order (quadratic) gradients. The string values match
    ``annotations.InterdigitationPattern`` so callers can convert freely.
    """

    CROSS = "abab"
    ABBA = "abba"


def pattern_from_annotation(
    pattern: InterdigitationPattern,
) -> "CentroidPattern":
    """Map an :class:`annotations.InterdigitationPattern` to a CentroidPattern."""
    return CentroidPattern(pattern.value)


# --- geometry helpers (operate on real finger geometry, no synthesis) ---


def _bbox(verts: List[Point]) -> Tuple[int, int, int, int]:
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))


def _poly_width(poly: Polygon) -> int:
    x_min, _, x_max, _ = _bbox(poly.vertices)
    return x_max - x_min


def _poly_height(poly: Polygon) -> int:
    _, y_min, _, y_max = _bbox(poly.vertices)
    return y_max - y_min


def _poly_centroid_x(poly: Polygon) -> float:
    x_min, _, x_max, _ = _bbox(poly.vertices)
    return (x_min + x_max) / 2.0


def _poly_area(poly: Polygon) -> int:
    """Axis-aligned-rectangle area (finger polygons are rectangles)."""
    x_min, y_min, x_max, y_max = _bbox(poly.vertices)
    return (x_max - x_min) * (y_max - y_min)


def _set_poly_x(poly: Polygon, new_x_min: int) -> Polygon:
    """Translate a (rectangular) polygon so its left edge sits at ``new_x_min``."""
    x_min, _, _, _ = _bbox(poly.vertices)
    dx = new_x_min - x_min
    return Polygon(vertices=[Point(v.x + dx, v.y) for v in poly.vertices])


def extract_finger_polys(
    device_polys: List[LayerPolygon], finger_layer: Optional[int] = None
) -> List[LayerPolygon]:
    """Return only the per-finger poly rectangles from a device's geometry.

    ``lower_mos_finger_geometry`` emits ``finger_count`` poly rectangles (the
    gate fingers) followed by one diffusion rectangle. The fingers all live on
    the POLY layer; the diffusion is on a different layer and spans the whole
    device, so it has the *largest* width. We identify finger polys as those on
    the most common layer whose width is the *minimum* width on that layer (the
    diffusion, even if same layer in some PDKs, is wider).

    If ``finger_layer`` is given, restrict to that layer explicitly.
    """
    if not device_polys:
        return []

    if finger_layer is not None:
        candidates = [lp for lp in device_polys if lp.layer == finger_layer]
    else:
        # Most common layer = the finger layer (N fingers vs 1 diffusion).
        from collections import Counter

        layer_counts = Counter(lp.layer for lp in device_polys)
        finger_layer = layer_counts.most_common(1)[0][0]
        candidates = [lp for lp in device_polys if lp.layer == finger_layer]

    if not candidates:
        return []

    # Among same-layer polys, the diffusion (if present) is strictly the widest;
    # fingers share the minimum width. Keep all polys at the minimum width.
    min_w = min(_poly_width(lp.polygon) for lp in candidates)
    fingers = [lp for lp in candidates if _poly_width(lp.polygon) == min_w]
    # Order left-to-right by centroid x for a deterministic A-finger sequence.
    fingers.sort(key=lambda lp: _poly_centroid_x(lp.polygon))
    return fingers


# --- interleave ordering ---


def interleave_order(
    n_a: int, n_b: int, pattern: CentroidPattern
) -> List[Tuple[str, int]]:
    """Produce the interleaved finger ordering as a list of ``(device, idx)``.

    ``device`` is ``"A"`` or ``"B"`` and ``idx`` is the 0-based finger index
    within that device. The returned list has length ``n_a + n_b``.

    * ``CROSS``: centro-symmetric alternation ``A B A B | B A B A`` (the second
      half mirrors the first), so both devices' fingers are symmetric about the
      column center and the centroids coincide. Requires ``n_a == n_b``.
    * ``ABBA``: ``A B B A`` repeated per group of 4 positions. Requires
      ``n_a == n_b`` and the count to be even (each ABBA group consumes 2 A and
      2 B fingers).
    """
    if n_a != n_b:
        raise ValueError(
            f"interdigitation requires equal finger counts, got A={n_a} B={n_b}"
        )
    n = n_a
    if n < 1:
        raise ValueError("finger count must be >= 1")

    if pattern is CentroidPattern.CROSS:
        # Centro-symmetric ABAB: first half alternates A,B from the front,
        # second half is the mirror so the arrangement is symmetric about the
        # column center -> coincident centroids. Fingers of each device are
        # consumed front-to-back from each half independently.
        order: List[Tuple[str, int]] = []
        # First half positions 0..n-1 take fingers a0,b0,a1,b1,...
        a_idx = 0
        b_idx = 0
        first_half: List[Tuple[str, int]] = []
        for i in range(n):
            if i % 2 == 0:
                first_half.append(("A", a_idx)); a_idx += 1
            else:
                first_half.append(("B", b_idx)); b_idx += 1
        # Second half repeats the first half's device sequence *reversed*
        # (not swapped), consuming the remaining fingers. This makes the full
        # arrangement centro-symmetric about the column center: e.g. n=4 gives
        # A B A B | B A B A, so A occupies {0,2,5,7} and B occupies {1,3,4,6},
        # both with the same centroid.
        second_half: List[Tuple[str, int]] = []
        for dev, _ in reversed(first_half):
            if dev == "A":
                second_half.append(("A", a_idx)); a_idx += 1
            else:
                second_half.append(("B", b_idx)); b_idx += 1
        order = first_half + second_half
        return order

    if pattern is CentroidPattern.ABBA:
        if n % 2 != 0:
            raise ValueError(
                f"ABBA pattern requires an even finger count, got {n}"
            )
        order = []
        a_idx = 0
        b_idx = 0
        # Each ABBA group consumes 2 A and 2 B fingers in order A B B A.
        for _ in range(n // 2):
            order.append(("A", a_idx)); a_idx += 1
            order.append(("B", b_idx)); b_idx += 1
            order.append(("B", b_idx)); b_idx += 1
            order.append(("A", a_idx)); a_idx += 1
        return order

    raise ValueError(f"unsupported pattern: {pattern}")


# --- result types ---


@dataclass
class CentroidCheck:
    """Result of the centroid-coincidence check after interdigitation."""

    centroid_a_x: float
    centroid_b_x: float
    centroid_y: float
    delta_x: float
    tolerance: float
    coincident: bool


@dataclass
class InterdigitationResult:
    """Output of :func:`interdigitate_pair`.

    ``ordering`` is the interleaved finger sequence; ``polygons_a`` /
    ``polygons_b`` are the re-placed finger geometries for each device (in the
    shared column); ``check`` reports centroid coincidence.
    """

    pattern: CentroidPattern
    ordering: List[Tuple[str, int]]
    polygons_a: List[LayerPolygon] = field(default_factory=list)
    polygons_b: List[LayerPolygon] = field(default_factory=list)
    check: Optional[CentroidCheck] = None

    @property
    def finger_count_a(self) -> int:
        return len(self.polygons_a)

    @property
    def finger_count_b(self) -> int:
        return len(self.polygons_b)

    def all_polygons(self) -> List[LayerPolygon]:
        return list(self.polygons_a) + list(self.polygons_b)

    def to_named_cell(self, name: str) -> NamedCell:
        return NamedCell(name=name, polygons=self.all_polygons())


def _centroid_x(polys: List[LayerPolygon]) -> float:
    """Area-weighted centroid x of a set of (rectangular) finger polygons."""
    total_area = 0
    moment = 0.0
    for lp in polys:
        a = _poly_area(lp.polygon)
        total_area += a
        moment += a * _poly_centroid_x(lp.polygon)
    if total_area == 0:
        return 0.0
    return moment / total_area


def _centroid_y(polys: List[LayerPolygon]) -> float:
    total_area = 0
    moment = 0.0
    for lp in polys:
        a = _poly_area(lp.polygon)
        x_min, y_min, x_max, y_max = _bbox(lp.polygon.vertices)
        total_area += a
        moment += a * ((y_min + y_max) / 2.0)
    if total_area == 0:
        return 0.0
    return moment / total_area


def interdigitate_pair(
    fingers_a: List[LayerPolygon],
    fingers_b: List[LayerPolygon],
    pattern: CentroidPattern = CentroidPattern.CROSS,
    *,
    pitch: Optional[int] = None,
    tolerance: float = 1.0,
    finger_layer: Optional[int] = None,
) -> InterdigitationResult:
    """Interleave two matched devices' fingers into a common-centroid column.

    Parameters
    ----------
    fingers_a, fingers_b:
        The two devices' finger geometry, as produced by
        ``lower_mos_finger_geometry`` (the full poly+diffusion list is accepted;
        the diffusion polygon is filtered out via :func:`extract_finger_polys`).
    pattern:
        ``CROSS`` (ABAB) or ``ABBA``.
    pitch:
        Center-to-center finger pitch in DBU. Defaults to the input finger
        pitch derived from device A's geometry (so total span is preserved when
        finger widths match the original pitch); if A has a single finger the
        finger width is used.
    tolerance:
        Max allowed |centroid_a_x - centroid_b_x| (DBU) for ``coincident``.
    finger_layer:
        Optional explicit poly layer number; otherwise inferred.

    Returns
    -------
    InterdigitationResult
        Re-placed finger geometries for A and B plus a centroid check. Finger
        count and total per-device finger area are preserved exactly (no area
        is lost — fingers are translated, never resized).
    """
    a_polys = extract_finger_polys(fingers_a, finger_layer)
    b_polys = extract_finger_polys(fingers_b, finger_layer)

    n_a = len(a_polys)
    n_b = len(b_polys)
    ordering = interleave_order(n_a, n_b, pattern)

    # Determine the finger width (constant per matched pair) and pitch.
    finger_w = _poly_width(a_polys[0].polygon) if a_polys else 0
    if pitch is None:
        if n_a >= 2:
            x0 = _poly_centroid_x(a_polys[0].polygon)
            x1 = _poly_centroid_x(a_polys[1].polygon)
            pitch = int(round(x1 - x0))
        else:
            pitch = finger_w if finger_w > 0 else 1
    if pitch <= 0:
        pitch = finger_w if finger_w > 0 else 1

    # Re-place each finger left-to-right at uniform pitch. The first finger's
    # left edge is anchored at the original min-x of device A so absolute
    # position stays sensible relative to the input.
    base_x = _bbox(a_polys[0].polygon.vertices)[0] if a_polys else 0

    new_a: List[LayerPolygon] = [None] * n_a  # type: ignore[list-item]
    new_b: List[LayerPolygon] = [None] * n_b  # type: ignore[list-item]

    for position, (dev, idx) in enumerate(ordering):
        # center-to-center pitch; left edge = center - finger_w/2 - keep integer
        center = base_x + finger_w / 2.0 + position * pitch
        new_x_min = int(round(center - finger_w / 2.0))
        if dev == "A":
            src = a_polys[idx]
            new_poly = _set_poly_x(src.polygon, new_x_min)
            new_a[idx] = LayerPolygon(
                layer=src.layer, datatype=src.datatype, polygon=new_poly
            )
        else:
            src = b_polys[idx]
            new_poly = _set_poly_x(src.polygon, new_x_min)
            new_b[idx] = LayerPolygon(
                layer=src.layer, datatype=src.datatype, polygon=new_poly
            )

    ca = _centroid_x(new_a)
    cb = _centroid_x(new_b)
    cy = _centroid_y(new_a + new_b)
    delta = abs(ca - cb)
    check = CentroidCheck(
        centroid_a_x=ca,
        centroid_b_x=cb,
        centroid_y=cy,
        delta_x=delta,
        tolerance=tolerance,
        coincident=delta <= tolerance,
    )

    return InterdigitationResult(
        pattern=pattern,
        ordering=ordering,
        polygons_a=new_a,
        polygons_b=new_b,
        check=check,
    )


def fingers_overlap(polys: List[LayerPolygon]) -> bool:
    """Return True if any two finger polygons' x-extents overlap.

    Fingers are vertical poly stripes placed in a row; a valid interdigitated
    layout has strictly non-overlapping x-intervals (touching edges allowed).
    """
    intervals = []
    for lp in polys:
        x_min, _, x_max, _ = _bbox(lp.polygon.vertices)
        intervals.append((x_min, x_max))
    intervals.sort()
    for i in range(1, len(intervals)):
        prev_max = intervals[i - 1][1]
        cur_min = intervals[i][0]
        if cur_min < prev_max:
            return True
    return False
