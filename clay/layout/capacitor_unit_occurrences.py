"""Immutable occurrence facts emitted by native generated capacitor builders.

Centers are the integer-nm floor midpoint of each original R0 unit's full
extent or its actual capacitor plate extent, as explicitly declared by
``center_basis``. Shared-electrode banks use their distinct emitted plates,
without inventing a full-artwork partition. The representative point follows
the artwork's actual transforms; it is never recomputed or snapped after
rotation. These are candidate geometry facts, not an electrical,
source-authority, or physical-signoff certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

_TRANSFORMS = ("R0", "R90", "R180", "R270", "MR0", "MR90", "MR180", "MR270")
_CENTER_BASES = ("materialized_unit_bbox_center_floor_nm",
                 "materialized_unit_plate_bbox_center_floor_nm")


def _box(value):
    if (type(value) is not tuple or len(value) != 4
            or any(type(v) is not int for v in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise ValueError("capacitor occurrence extent must be nonempty exact integer geometry")
    return value


def _transform(value):
    if type(value) is not str or value not in _TRANSFORMS:
        raise ValueError("capacitor occurrence transform is unsupported")
    return value


def _compose(outer, inner):
    outer_mirror, inner_mirror = outer.startswith("M"), inner.startswith("M")
    outer_angle = int(outer[2:] if outer_mirror else outer[1:])
    inner_angle = int(inner[2:] if inner_mirror else inner[1:])
    angle = (outer_angle + (-inner_angle if outer_mirror else inner_angle)) % 360
    return ("MR" if outer_mirror != inner_mirror else "R") + str(angle)


def _point(x, y, transform):
    if transform.startswith("M"):
        y = -y
        transform = transform[1:]
    return {"R0": (x, y), "R90": (-y, x),
            "R180": (-x, -y), "R270": (y, -x)}[transform]


def with_capacitor_occurrences(geometry, occurrences):
    """Replace the declared frozen field while preserving older native extras."""
    if type(occurrences) is not CapacitorUnitOccurrenceGeometry:
        raise TypeError("capacitor occurrence metadata must be exactly typed")
    result = replace(geometry, capacitor_unit_occurrences=occurrences)
    for name, value in geometry.__dict__.items():
        if name not in geometry.__dataclass_fields__:
            result.__dict__[name] = value
    return result


@dataclass(frozen=True, slots=True)
class CapacitorUnitOccurrenceGeometry:
    unit_count: int
    bank_extent_nm: tuple[int, int, int, int]
    bank_orientation: str
    unit_centers_nm: tuple[tuple[int, int], ...]
    unit_transforms: tuple[str, ...]
    schema: str = "clay.capacitor-unit-occurrence-geometry/v1"
    kind: str = "cap"
    center_basis: str = "materialized_unit_bbox_center_floor_nm"

    def __post_init__(self):
        if (type(self.schema) is not str or self.schema != "clay.capacitor-unit-occurrence-geometry/v1"
                or type(self.kind) is not str or self.kind != "cap"
                or type(self.center_basis) is not str
                or self.center_basis not in _CENTER_BASES):
            raise ValueError("capacitor occurrence schema, kind, or center basis is unsupported")
        box = _box(self.bank_extent_nm)
        _transform(self.bank_orientation)
        if type(self.unit_count) is not int or self.unit_count <= 0:
            raise ValueError("capacitor occurrence count must be a positive exact integer")
        if (type(self.unit_centers_nm) is not tuple or type(self.unit_transforms) is not tuple
                or len(self.unit_centers_nm) != self.unit_count
                or len(self.unit_transforms) != self.unit_count):
            raise ValueError("capacitor occurrence population must exactly match its count")
        for center in self.unit_centers_nm:
            if (type(center) is not tuple or len(center) != 2
                    or any(type(v) is not int for v in center)
                    or not box[0] <= center[0] <= box[2]
                    or not box[1] <= center[1] <= box[3]):
                raise ValueError("capacitor occurrence center must be exact integer geometry inside its extent")
        if len(set(self.unit_centers_nm)) != self.unit_count:
            raise ValueError("capacitor occurrence centers must be unique")
        for transform in self.unit_transforms:
            _transform(transform)

    @classmethod
    def from_unit_extent(cls, extent):
        """Record one actual R0 unit before any orientation is applied."""
        x0, y0, x1, y1 = _box(extent)
        return cls(unit_count=1, bank_extent_nm=extent, bank_orientation="R0",
                   unit_centers_nm=(((x0+x1)//2, (y0+y1)//2),), unit_transforms=("R0",))

    def transformed(self, transform, offset_nm, bank_extent_nm):
        """Apply the artwork's affine transform and retain its emitted extent."""
        _transform(transform)
        if (type(offset_nm) is not tuple or len(offset_nm) != 2
                or any(type(v) is not int for v in offset_nm)):
            raise ValueError("capacitor occurrence offset must be exact integer geometry")
        dx, dy = offset_nm
        return type(self)(unit_count=self.unit_count, bank_extent_nm=bank_extent_nm,
            center_basis=self.center_basis,
            bank_orientation=_compose(transform, self.bank_orientation),
            unit_centers_nm=tuple((x+dx, y+dy) for x, y in
                (_point(x, y, transform) for x, y in self.unit_centers_nm)),
            unit_transforms=tuple(_compose(transform, value) for value in self.unit_transforms))

    def canonical_data(self):
        return {"schema": self.schema, "kind": self.kind, "center_basis": self.center_basis,
                "unit_count": self.unit_count, "bank_extent_nm": list(self.bank_extent_nm),
                "bank_orientation": self.bank_orientation,
                "unit_centers_nm": [list(point) for point in self.unit_centers_nm],
                "unit_transforms": list(self.unit_transforms)}

    @classmethod
    def from_data(cls, data):
        """Strictly replay canonical JSON data, without scalar coercion."""
        keys = {"schema", "kind", "center_basis", "unit_count", "bank_extent_nm",
                "bank_orientation", "unit_centers_nm", "unit_transforms"}
        if type(data) is not dict or set(data) != keys:
            raise ValueError("capacitor occurrence data has missing or unknown fields")
        if (type(data["bank_extent_nm"]) is not list
                or type(data["unit_centers_nm"]) is not list
                or any(type(point) is not list for point in data["unit_centers_nm"])
                or type(data["unit_transforms"]) is not list):
            raise ValueError("capacitor occurrence JSON geometry must use arrays")
        return cls(**{**data, "bank_extent_nm": tuple(data["bank_extent_nm"]),
                      "unit_centers_nm": tuple(tuple(point) for point in data["unit_centers_nm"]),
                      "unit_transforms": tuple(data["unit_transforms"])})
