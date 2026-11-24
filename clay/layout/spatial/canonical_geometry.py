"""Shared exact integer polygon canonicalization for CLM and GDS witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


Point = tuple[int, int]
Contour = tuple[Point, ...]


@dataclass(frozen=True, order=True, slots=True)
class CanonicalPolygon:
    """One exact polygon encoded as a canonical hull followed by sorted holes."""

    bbox: tuple[int, int, int, int]
    contours: tuple[Contour, ...]


def canonical_contour(points: Sequence[Point]) -> Contour | None:
    """Normalize duplicate, collinear, start-point, and traversal variations."""

    cleaned: list[Point] = []
    for raw_x, raw_y in points:
        point = (int(raw_x), int(raw_y))
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        kept: list[Point] = []
        count = len(cleaned)
        for index, point in enumerate(cleaned):
            before = cleaned[(index - 1) % count]
            after = cleaned[(index + 1) % count]
            cross = (
                (point[0] - before[0]) * (after[1] - point[1])
                - (point[1] - before[1]) * (after[0] - point[0])
            )
            between = (
                min(before[0], after[0]) <= point[0] <= max(before[0], after[0])
                and min(before[1], after[1])
                <= point[1]
                <= max(before[1], after[1])
            )
            if cross == 0 and between:
                changed = True
                continue
            kept.append(point)
        cleaned = kept
    if len(cleaned) < 3:
        return None
    contour = tuple(cleaned)
    candidates: list[Contour] = []
    for sequence in (contour, tuple(reversed(contour))):
        candidates.extend(
            sequence[index:] + sequence[:index]
            for index in range(len(sequence))
        )
    return min(candidates)


def canonical_polygon(
    hull_points: Sequence[Point],
    hole_points: Sequence[Sequence[Point]] = (),
) -> CanonicalPolygon | None:
    """Canonicalize one polygon without depending on a geometry-engine object."""

    hull = canonical_contour(hull_points)
    if hull is None:
        return None
    holes = tuple(
        sorted(
            contour
            for points in hole_points
            if (contour := canonical_contour(points)) is not None
        )
    )
    xs = tuple(point[0] for point in hull)
    ys = tuple(point[1] for point in hull)
    bbox = (min(xs), min(ys), max(xs), max(ys))
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return None
    return CanonicalPolygon(bbox=bbox, contours=(hull,) + holes)


__all__ = ["CanonicalPolygon", "Contour", "Point", "canonical_contour", "canonical_polygon"]
