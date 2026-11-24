"""Generated owner-local seam operands shared by CP and independent replay.

Source and materialization hashes identify dependencies; they are not admission.
The physical producer must separately join these boxes to current native artwork.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    _require(type(value) is str and value and value == value.strip(),
             "guest-slot identity must be a nonempty canonical string")


def _size(value):
    _require(type(value) is tuple and len(value) == 2
             and all(type(v) is int and v > 0 for v in value),
             "guest-slot dimensions must be positive exact integers")


def _box(value, size):
    _require(type(value) is tuple and len(value) == 4
             and all(type(v) is int for v in value),
             "guest-slot box must contain four exact integers")
    _require(0 <= value[0] < value[2] <= size[0]
             and 0 <= value[1] < value[3] <= size[1],
             "guest-slot box escapes normalized owner dimensions")


@dataclass(frozen=True, slots=True)
class MosaicGuestSlotProjection:
    obligation_id: str
    source_slot_sha256: str
    geometry_binding_sha256: str
    host_master: str
    guest_master: str
    host_size_nm: tuple[int, int]
    guest_size_nm: tuple[int, int]
    axis: str
    # Source row order, with each left/right node's actual generated envelope.
    boundaries: tuple[tuple[str, tuple[int, int, int, int], str, tuple[int, int, int, int]], ...]
    guest_envelope_nm: tuple[int, int, int, int]
    minimum_clearance_nm: int

    def __post_init__(self):
        for name in (self.obligation_id, self.host_master, self.guest_master):
            _text(name)
        for value in (self.source_slot_sha256, self.geometry_binding_sha256):
            _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
                     "guest-slot source/materialization SHA is invalid")
        _require(self.host_master != self.guest_master, "guest-slot owners collapse")
        _size(self.host_size_nm)
        _size(self.guest_size_nm)
        _require(type(self.axis) is str and self.axis in ("horizontal", "vertical"),
                 "guest-slot axis is invalid")
        _require(type(self.minimum_clearance_nm) is int and self.minimum_clearance_nm >= 0,
                 "guest-slot clearance must be nonnegative exact integer nm")
        _box(self.guest_envelope_nm, self.guest_size_nm)
        _require(type(self.boundaries) is tuple and 2 <= len(self.boundaries) <= 4096,
                 "guest-slot requires at least two bounded source rows")
        nodes = []
        for row in self.boundaries:
            _require(type(row) is tuple and len(row) == 4, "guest-slot boundary row is malformed")
            left, left_box, right, right_box = row
            _text(left)
            _text(right)
            _box(left_box, self.host_size_nm)
            _box(right_box, self.host_size_nm)
            nodes.extend((left, right))
        _require(len(nodes) == len(set(nodes)), "guest-slot boundary nodes must be unique")

    def canonical_data(self):
        return {"schema": "clay.mosaic-guest-slot-projection/v1",
                "frame": "generated_solver_owner_origin_nm",
                "obligation_id": self.obligation_id, "source_slot_sha256": self.source_slot_sha256,
                "geometry_binding_sha256": self.geometry_binding_sha256,
                "host_master": self.host_master, "guest_master": self.guest_master,
                "host_size_nm": self.host_size_nm, "guest_size_nm": self.guest_size_nm,
                "axis": self.axis, "boundaries": self.boundaries,
                "guest_envelope_nm": self.guest_envelope_nm,
                "minimum_clearance_nm": self.minimum_clearance_nm}

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.canonical_data(), sort_keys=True,
            separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def canonical_mosaic_guest_slot_projections(values, *, dims):
    if type(values) is not tuple:
        raise TypeError("guest-slot projections must be an exact tuple")
    for row in values:
        if type(row) is not MosaicGuestSlotProjection:
            raise TypeError("guest-slot projection has a foreign type")
        row.__post_init__()
        _require(dims.get(row.host_master) == row.host_size_nm
                 and dims.get(row.guest_master) == row.guest_size_nm,
                 "guest-slot current owner dimensions differ")
    _require(tuple(sorted(values, key=lambda row: row.obligation_id)) == values
             and len({row.obligation_id for row in values}) == len(values)
             and len({row.guest_master for row in values}) == len(values),
             "guest-slot projections must be canonical and unique")
    return values


def mosaic_guest_slot_bounds_nm(row):
    """Recompute inclusive dx/dy intervals; no scalar bound is caller authority."""
    if type(row) is not MosaicGuestSlotProjection:
        raise TypeError("guest-slot projection has a foreign type")
    row.__post_init__()
    seam = 0 if row.axis == "horizontal" else 1
    transverse = 1 - seam
    guest = row.guest_envelope_nm
    lower_seam = max(left[seam+2] + row.minimum_clearance_nm - guest[seam]
                     for _ln, left, _rn, _right in row.boundaries)
    upper_seam = min(right[seam] - row.minimum_clearance_nm - guest[seam+2]
                     for _ln, _left, _rn, right in row.boundaries)
    boxes = tuple(box for _ln, left, _rn, right in row.boundaries for box in (left, right))
    lower_transverse = max(box[transverse] - guest[transverse+2] + 1 for box in boxes)
    upper_transverse = min(box[transverse+2] - guest[transverse] - 1 for box in boxes)
    return ((lower_seam, upper_seam, lower_transverse, upper_transverse) if seam == 0
            else (lower_transverse, upper_transverse, lower_seam, upper_seam))


def mosaic_guest_slot_satisfied(positions, row):
    xmin, xmax, ymin, ymax = mosaic_guest_slot_bounds_nm(row)
    hx, hy = positions[row.host_master]
    gx, gy = positions[row.guest_master]
    return xmin <= gx-hx <= xmax and ymin <= gy-hy <= ymax


def add_mosaic_guest_slot_constraints(model, xs, ys, projections, *, assumption_for=None):
    if type(projections) is not tuple:
        raise TypeError("guest-slot projections must be an exact tuple")
    for index, row in enumerate(projections):
        xmin, xmax, ymin, ymax = mosaic_guest_slot_bounds_nm(row)
        dx = xs[row.guest_master] - xs[row.host_master]
        dy = ys[row.guest_master] - ys[row.host_master]
        constraints = (model.Add(dx >= xmin), model.Add(dx <= xmax),
                       model.Add(dy >= ymin), model.Add(dy <= ymax))
        if assumption_for is not None:
            guard = assumption_for(f"mosaic_guest_slot:{index}")
            for constraint in constraints:
                constraint.OnlyEnforceIf(guard)


def swap_mosaic_guest_slot_axes(row):
    mosaic_guest_slot_bounds_nm(row)
    box = lambda values: (values[1], values[0], values[3], values[2])
    return replace(row, axis="vertical" if row.axis == "horizontal" else "horizontal",
        host_size_nm=row.host_size_nm[::-1], guest_size_nm=row.guest_size_nm[::-1],
        guest_envelope_nm=box(row.guest_envelope_nm),
        boundaries=tuple((ln, box(left), rn, box(right)) for ln, left, rn, right in row.boundaries))
