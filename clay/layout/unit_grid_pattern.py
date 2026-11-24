"""Occupied unit ranks and orthogonal transforms, without physical coordinates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class UnitGridPattern:
    """Canonical row-major occupied cells; an absent cell contains no unit."""

    placements: tuple[tuple[int, int, str], ...]
    schema: ClassVar[str] = "clay.unit-grid-pattern/v1"

    def __post_init__(self) -> None:
        if type(self.placements) is not tuple or not self.placements:
            raise TypeError("unit pattern placements must be a nonempty exact tuple")
        orientations = {"R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270"}
        cells = []
        for item in self.placements:
            if type(item) is not tuple or len(item) != 3:
                raise TypeError("unit pattern placement must be an exact rank/transform triple")
            row, column, orientation = item
            if type(row) is not int or type(column) is not int or row < 0 or column < 0:
                raise ValueError("unit pattern rank must be a nonnegative exact integer")
            if type(orientation) is not str or orientation not in orientations:
                raise ValueError("unit pattern orientation is unsupported")
            cells.append((row, column))
        if cells != sorted(set(cells)):
            raise ValueError("unit pattern placements must have unique canonical row-major ranks")
        for axis in (0, 1):
            ranks = sorted({cell[axis] for cell in cells})
            if ranks != list(range(len(ranks))):
                raise ValueError("unit pattern ranks must be normalized without missing entire ranks")

    @property
    def unit_count(self) -> int:
        return len(self.placements)

    @property
    def rows(self) -> int:
        return 1 + max(row for row, _column, _orientation in self.placements)

    @property
    def columns(self) -> int:
        return 1 + max(column for _row, column, _orientation in self.placements)

    def canonical_data(self) -> dict:
        return {"schema": self.schema, "placements": [list(row) for row in self.placements]}

    @property
    def sha256(self) -> str:
        raw = json.dumps(self.canonical_data(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def replay_unit_grid_pattern(value: object) -> UnitGridPattern:
    if type(value) is not UnitGridPattern:
        raise TypeError("unit pattern must be an exact UnitGridPattern")
    return UnitGridPattern(value.placements)


def validate_array_unit_pattern(value, *, style, order, rows, transforms):
    pattern = replay_unit_grid_pattern(value)
    if style != "discrete" or order != "ratio" or rows != pattern.rows:
        raise ValueError("array unit pattern requires discrete ratio organization and matching rows")
    if transforms and transforms != tuple(pose for _r, _c, pose in pattern.placements):
        raise ValueError("array unit pattern transforms differ from occupied cells")


__all__ = ["UnitGridPattern", "replay_unit_grid_pattern"]
