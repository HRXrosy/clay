"""Spatial placement normalization, predicates, and deterministic ordered seeds."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from clay.layout.mosaic_guest_slot_projection import MosaicGuestSlotProjection, canonical_mosaic_guest_slot_projections
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from clay.layout.compact_blocks import _NM_PER_UM
from clay.layout.canonical_collision_projection import CanonicalRowBoundedCavityProjection
from clay.layout.place.placement_constants import (
    _ASPECT_CONSTRAINT_SCALE,
    _ASPECT_TOL,
    _BAND_BOTTOM,
    _BAND_CENTER,
    _BAND_TOP,
    _Bbox,
    _FloorplanColumnGroup,
    _FloorplanColumnTerm,
    _GLOBAL_FLOORPLAN_COLUMN_SCOPE,
    _LayeredCollisionFootprint,
    _RigidCompoundFootprint,
    _RowBoundedCavityProjection,
)
from clay.layout.place.placement_cuts import (
    _exact_cut_input_blocks,
)
from clay.layout.place.placement_diagnostics import (
    _canonical_full_hard_failure_rows,
    _full_hard_replay_row,
)
from clay.layout.place import lp_compactor as _lp_compactor


def _scaled_aspect_bounds(
    aspect_target: float,
    aspect_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int, int]:
    """Return target tolerance clipped to the declared feasible ``W/H`` range."""

    target = float(aspect_target)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("aspect_target must be a positive finite W/H ratio")
    scale = _ASPECT_CONSTRAINT_SCALE
    # Stay one scaled unit inside the public floating-point tolerance.  A point
    # at the mathematical 5% boundary (for example 1.71 / 1.8) can otherwise
    # round to 0.050000000000000044 in the artifact certificate and disagree
    # with the exact CP inequality.
    lower_value = target * (1.0 - _ASPECT_TOL)
    upper_value = target * (1.0 + _ASPECT_TOL)
    lower = int(
        (Decimal(str(lower_value)) * scale).to_integral_value(
            rounding=ROUND_FLOOR
        )
    ) + 1
    upper = int(
        (Decimal(str(upper_value)) * scale).to_integral_value(
            rounding=ROUND_CEILING
        )
    ) - 1
    if aspect_bounds is not None:
        try:
            declared_lower, declared_upper = (
                float(aspect_bounds[0]),
                float(aspect_bounds[1]),
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "aspect_bounds must contain two finite positive ratios"
            ) from exc
        if (
            not math.isfinite(declared_lower)
            or not math.isfinite(declared_upper)
            or declared_lower <= 0.0
            or declared_upper < declared_lower
        ):
            raise ValueError("aspect_bounds must be ordered finite positive ratios")
        lower = max(
            lower,
            int(
                (Decimal(str(declared_lower)) * scale).to_integral_value(
                    rounding=ROUND_CEILING
                )
            ),
        )
        upper = min(
            upper,
            int(
                (Decimal(str(declared_upper)) * scale).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            ),
        )
    if lower <= 0 or upper < lower:
        raise ValueError("aspect_target has no representable tolerance interval")
    return lower, upper, scale

def _ceil_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("ratio denominator must be positive")
    return (int(numerator) + int(denominator) - 1) // int(denominator)

def _strict_outline_aspect_height_floor_nm(
    width_nm: int,
    *,
    aspect_target: float,
    aspect_bounds: Optional[Tuple[float, float]] = None,
) -> int:
    """Exact height floor implied by a fixed physical bbox width."""

    if type(width_nm) is not int:
        raise TypeError("strict outline width must be an exact integer nm value")
    if width_nm <= 0:
        raise ValueError("strict outline width must be positive")
    _lower, upper, scale = _scaled_aspect_bounds(aspect_target, aspect_bounds)
    return _ceil_ratio(width_nm * scale, upper)

def _placement_objective_from_intent(intent: Any) -> str:
    """Replay the canonical layout objective at the placer consumer boundary."""

    objective = getattr(intent, "objective", None)
    if objective is None:
        return "height"
    if type(objective) is not tuple or len(objective) != 2:
        raise TypeError("layout objective must be an exact tuple of two strings")
    if any(type(value) is not str for value in objective):
        raise TypeError("layout objective members must be exact strings")
    if objective == ("area", "minimize"):
        return "area"
    raise ValueError(f"unsupported layout objective: {objective!r}")

def _add_directional_aspect_constraints(
    model: Any,
    *,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    width_bound: int,
    height_bound: int,
    aspect_target: float,
    aspect_bounds: Optional[Tuple[float, float]] = None,
    bottom_edge_keepout_nm: int = 0,
    enforcement_literal: Any = None,
) -> None:
    """Bind the exact placed bbox to the declared directional aspect interval."""

    lower, upper, scale = _scaled_aspect_bounds(aspect_target, aspect_bounds)
    x_min = model.NewIntVar(0, int(width_bound), "bbox_x_min")
    x_max = model.NewIntVar(0, int(width_bound), "bbox_x_max")
    y_min = model.NewIntVar(0, int(height_bound), "bbox_y_min")
    y_max = model.NewIntVar(0, int(height_bound), "bbox_y_max")
    width = model.NewIntVar(1, int(width_bound), "bbox_width")
    height = model.NewIntVar(1, int(height_bound), "bbox_height")
    model.AddMinEquality(x_min, list(xs.values()))
    model.AddMaxEquality(x_max, [xs[name] + int(dims[name][0]) for name in sorted(xs)])
    if int(bottom_edge_keepout_nm) > 0:
        # The edge-rail reserve is floorplan area even when no layer-compatible
        # boundary carrier happens to occupy it.  Do not normalize that empty
        # physical band away when binding the directional outline.
        occupied_y_min = model.NewIntVar(
            0, int(height_bound), "occupied_bbox_y_min"
        )
        model.AddMinEquality(occupied_y_min, list(ys.values()))
        model.Add(y_min == 0)
    else:
        model.AddMinEquality(y_min, list(ys.values()))
    model.AddMaxEquality(y_max, [ys[name] + int(dims[name][1]) for name in sorted(ys)])
    model.Add(width == x_max - x_min)
    model.Add(height == y_max - y_min)
    lower_constraint = model.Add(width * scale >= height * lower)
    upper_constraint = model.Add(width * scale <= height * upper)
    if enforcement_literal is not None:
        lower_constraint.OnlyEnforceIf(enforcement_literal)
        upper_constraint.OnlyEnforceIf(enforcement_literal)

def _canonical_flexible_offset_rows(
    value: Any,
    *,
    field: str,
    members: Tuple[str, ...],
    require_complete: bool,
) -> Tuple[Tuple[str, _Bbox], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field} must be one exact tuple")
    rows: list[Tuple[str, _Bbox]] = []
    for row in value:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError(f"{field} rows must be exact pairs")
        member, raw_bbox = row
        if type(member) is not str or not member or member not in members:
            raise ValueError(f"{field} names a foreign member")
        if (
            type(raw_bbox) is not tuple
            or len(raw_bbox) != 4
            or any(type(coordinate) is not int for coordinate in raw_bbox)
        ):
            raise TypeError(f"{field} rows must contain exact integer bboxes")
        bbox = tuple(raw_bbox)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError(f"{field} bboxes must have positive area")
        rows.append((member, bbox))
    canonical = tuple(sorted(rows))
    if tuple(rows) != canonical or len({member for member, _bbox in rows}) != len(rows):
        raise ValueError(f"{field} must be canonical and unique")
    if require_complete and {member for member, _bbox in rows} != set(members):
        raise ValueError(f"{field} must cover every flexible member")
    return canonical

@dataclass(frozen=True, slots=True, order=True)
class FlexibleEnvelopeConstraint:
    """One deformable physical envelope around independently placed masters."""

    group: str
    members: Tuple[str, ...]
    extension_nm: int
    foreign_clearance_nm: int
    member_physical_offsets_nm: Tuple[Tuple[str, _Bbox], ...]
    member_adjacency_offsets_nm: Tuple[Tuple[str, _Bbox], ...] = ()
    foreign_exclusion_mode: str = "member"

    def __post_init__(self) -> None:
        if type(self.group) is not str or not self.group or self.group != self.group.strip():
            raise TypeError("flexible envelope group must be one canonical string")
        if (
            type(self.members) is not tuple
            or len(self.members) < 2
            or self.members != tuple(sorted(set(self.members)))
            or any(type(member) is not str or not member for member in self.members)
        ):
            raise ValueError("flexible envelope members must be exact and canonical")
        if type(self.extension_nm) is not int or self.extension_nm < 0:
            raise TypeError(
                "flexible envelope extension must be one non-negative exact int"
            )
        if type(self.foreign_clearance_nm) is not int or self.foreign_clearance_nm < 0:
            raise TypeError(
                "flexible envelope foreign clearance must be one non-negative exact int"
            )
        _canonical_flexible_offset_rows(
            self.member_physical_offsets_nm,
            field="flexible envelope physical offsets",
            members=self.members,
            require_complete=True,
        )
        _canonical_flexible_offset_rows(
            self.member_adjacency_offsets_nm,
            field="flexible envelope adjacency offsets",
            members=self.members,
            require_complete=True,
        )
        if type(self.foreign_exclusion_mode) is not str:
            raise TypeError(
                "flexible envelope foreign exclusion mode must be an exact string"
            )
        if self.foreign_exclusion_mode not in ("member", "envelope"):
            raise ValueError(
                "flexible envelope foreign exclusion mode must be member or envelope"
            )

@dataclass(frozen=True, slots=True, order=True)
class FlexibleEnvelopeProjection:
    """Exact solved envelope paired with the authority that produced it."""

    constraint: FlexibleEnvelopeConstraint
    bbox_nm: _Bbox
    foreign_clearances_nm: Tuple[Tuple[str, str, int], ...]

    def __post_init__(self) -> None:
        if type(self.constraint) is not FlexibleEnvelopeConstraint:
            raise TypeError("flexible envelope projection requires typed authority")
        if (
            type(self.bbox_nm) is not tuple
            or len(self.bbox_nm) != 4
            or any(type(value) is not int for value in self.bbox_nm)
        ):
            raise TypeError("flexible envelope projection bbox must be one exact tuple")
        if self.bbox_nm[2] <= self.bbox_nm[0] or self.bbox_nm[3] <= self.bbox_nm[1]:
            raise ValueError("flexible envelope projection bbox must have positive area")
        if type(self.foreign_clearances_nm) is not tuple:
            raise TypeError("flexible envelope foreign rows must be one exact tuple")
        rows: list[Tuple[str, str, int]] = []
        for row in self.foreign_clearances_nm:
            if (
                type(row) is not tuple
                or len(row) != 3
                or type(row[0]) is not str
                or not row[0]
                or row[0] not in self.constraint.members
                or type(row[1]) is not str
                or not row[1]
                or row[1] in self.constraint.members
                or type(row[2]) is not int
                or row[2] < self.constraint.foreign_clearance_nm
            ):
                raise TypeError("flexible envelope foreign rows must be exact")
            rows.append(row)
        if tuple(rows) != tuple(sorted(rows)) or len(
            {(row[0], row[1]) for row in rows}
        ) != len(rows):
            raise ValueError("flexible envelope foreign rows must be canonical and unique")

def _canonical_flexible_envelope_constraints(
    value: Any,
    *,
    block_ids: Sequence[str],
    bands: Optional[Mapping[str, int]] = None,
) -> Tuple[FlexibleEnvelopeConstraint, ...]:
    if type(value) is not tuple:
        raise TypeError("flexible envelope constraints must be one exact tuple")
    if any(type(row) is not FlexibleEnvelopeConstraint for row in value):
        raise TypeError("flexible envelope constraints contain a foreign row")
    canonical = tuple(sorted(value))
    if value != canonical or len({row.group for row in value}) != len(value):
        raise ValueError("flexible envelope constraints must be canonical and unique")
    known = set(block_ids)
    consumed: set[str] = set()
    for row in canonical:
        if any(member not in known for member in row.members):
            raise ValueError("flexible envelope constraint names a foreign master")
        if consumed.intersection(row.members):
            raise ValueError("one master cannot belong to multiple flexible envelopes")
        consumed.update(row.members)
        if bands is not None and len({int(bands[member]) for member in row.members}) != 1:
            raise ValueError("flexible envelope members must share one exact band")
    return canonical

def _canonical_floorplan_column_classes(
    values: object,
    *,
    block_ids: object,
) -> Tuple[Tuple[str, int, int, int], ...]:
    """Validate exact final-master column ownership without coercion."""

    if type(values) is not tuple:
        raise TypeError("floorplan_column_classes must be one exact tuple")
    known = set(block_ids)
    if isinstance(block_ids, Mapping):
        widths = {
            name: bbox[2] - bbox[0] for name, bbox in block_ids.items()
        }
        known_ids = tuple(widths)
    else:
        widths = {}
        known_ids = tuple(block_ids)  # type: ignore[arg-type]
    known = set(known_ids)
    rows: List[Tuple[str, int, int, int]] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 4:
            raise TypeError(
                "floorplan_column_classes must contain exact four-item tuples"
            )
        name, column, min_center_offset2, max_center_offset2 = raw
        if type(name) is not str or not name or name != name.strip():
            raise TypeError("floorplan column owner must be one canonical string")
        if type(column) is not int or column < 0:
            raise TypeError("floorplan column ordinal must be one non-negative exact int")
        if (
            type(min_center_offset2) is not int
            or type(max_center_offset2) is not int
            or min_center_offset2 < 0
            or max_center_offset2 < min_center_offset2
        ):
            raise TypeError("floorplan column center extents must be exact integers")
        if name not in known:
            raise ValueError(f"floorplan column names unknown block {name!r}")
        if widths and max_center_offset2 > 2 * widths[name]:
            raise ValueError("floorplan column center extent escapes its rigid master")
        rows.append((name, column, min_center_offset2, max_center_offset2))
    canonical = tuple(sorted(rows))
    if tuple(values) != canonical or len({row[0] for row in rows}) != len(rows):
        raise ValueError("floorplan_column_classes must be sorted with unique owners")
    return canonical

def _floorplan_column_order_satisfied(
    blocks: Sequence[Tuple[str, _Bbox]],
    column_classes: Sequence[Tuple[str, int, int, int]],
) -> bool:
    """Replay the exact centroid ordering used by the GDS intent checker."""

    by_bbox = dict(blocks)
    for left, left_column, _left_min2, left_max2 in column_classes:
        for right, right_column, right_min2, _right_max2 in column_classes:
            if left_column >= right_column:
                continue
            left_bbox = by_bbox[left]
            right_bbox = by_bbox[right]
            left_center2 = 2 * left_bbox[0] + left_max2
            right_center2 = 2 * right_bbox[0] + right_min2
            if left_center2 >= right_center2:
                return False
    return True

def _canonical_rigid_compound_footprints(
    values: object,
    *,
    dims: Mapping[str, Tuple[int, int]],
) -> Tuple[_RigidCompoundFootprint, ...]:
    """Validate rigid local collision rectangles without flattening their voids."""

    if type(values) is not tuple:
        raise TypeError("rigid compound footprints must be one exact tuple")
    known = set(dims)
    rows: list[_RigidCompoundFootprint] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError("rigid compound footprint rows must be exact pairs")
        owner, raw_rectangles = raw
        if type(owner) is not str or not owner or owner not in known:
            raise ValueError("rigid compound footprint names an unknown owner")
        if type(raw_rectangles) is not tuple or len(raw_rectangles) < 2:
            raise TypeError(
                "rigid compound footprint requires at least two exact rectangles"
            )
        width, height = dims[owner]
        rectangles: list[_Bbox] = []
        for rectangle in raw_rectangles:
            if (
                type(rectangle) is not tuple
                or len(rectangle) != 4
                or any(type(value) is not int for value in rectangle)
            ):
                raise TypeError("rigid compound rectangles must be exact integer bboxes")
            x0, y0, x1, y1 = rectangle
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("rigid compound rectangle escapes its owner bbox")
            rectangles.append(rectangle)
        canonical_rectangles = tuple(sorted(rectangles))
        if (
            tuple(rectangles) != canonical_rectangles
            or len(set(canonical_rectangles)) != len(canonical_rectangles)
        ):
            raise ValueError("rigid compound rectangles must be canonical and unique")
        if (
            min(rectangle[0] for rectangle in canonical_rectangles) != 0
            or min(rectangle[1] for rectangle in canonical_rectangles) != 0
            or max(rectangle[2] for rectangle in canonical_rectangles) != width
            or max(rectangle[3] for rectangle in canonical_rectangles) != height
        ):
            raise ValueError("rigid compound rectangles do not span the owner bbox")
        rows.append((owner, canonical_rectangles))
    canonical = tuple(sorted(rows))
    if canonical != values or len({owner for owner, _rects in canonical}) != len(
        canonical
    ):
        raise ValueError("rigid compound footprints must be canonical and unique")
    return canonical

def _canonical_layered_collision_footprints(
    values: object,
    *,
    dims: Mapping[str, Tuple[int, int]],
) -> Tuple[_LayeredCollisionFootprint, ...]:
    """Validate exact local Manhattan atoms, retaining same-layer voids."""

    if type(values) is not tuple:
        raise TypeError("layered collision footprints must be one exact tuple")
    known = set(dims)
    rows: list[_LayeredCollisionFootprint] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 2:
            raise TypeError("layered collision footprint rows must be exact pairs")
        owner, atoms = raw
        if type(owner) is not str or not owner or owner not in known:
            raise ValueError("layered collision footprint names an unknown owner")
        if type(atoms) is not tuple or not atoms:
            raise TypeError("layered collision footprint atoms must be non-empty")
        width, height = dims[owner]
        for atom in atoms:
            if (
                type(atom) is not tuple
                or len(atom) != 6
                or any(type(value) is not int for value in atom)
                or atom[0] < 0
                or atom[1] < 0
                or not (
                    0 <= atom[2] < atom[4] <= width
                    and 0 <= atom[3] < atom[5] <= height
                )
            ):
                raise ValueError("layered collision atom is malformed or out of bounds")
        if atoms != tuple(sorted(set(atoms))):
            raise ValueError("layered collision atoms must be canonical and unique")
        rows.append((owner, atoms))
    canonical = tuple(sorted(rows))
    if canonical != values or len({owner for owner, _atoms in canonical}) != len(
        canonical
    ):
        raise ValueError("layered collision footprints must be canonical and unique")
    return canonical

def _canonical_mosaic_guest_slot_projections(
    values: object,
    *,
    dims: Mapping[str, Tuple[int, int]],
    minimum_clearance_nm: int,
) -> Tuple[MosaicGuestSlotProjection, ...]:
    """Join generated seam operands to actual dimensions and source gap."""
    rows = canonical_mosaic_guest_slot_projections(values, dims=dims)
    if rows and (
        type(minimum_clearance_nm) is not int
        or minimum_clearance_nm < 0
        or any(row.minimum_clearance_nm != minimum_clearance_nm for row in rows)
    ):
        raise ValueError("mosaic guest-slot clearance differs from native gap authority")
    return rows

def _canonical_row_bounded_cavity_projections(
    values: object,
    *,
    dims: Mapping[str, Tuple[int, int]],
) -> Tuple[_RowBoundedCavityProjection, ...]:
    if type(values) is not tuple:
        raise TypeError("row-bounded cavity projections must be one exact tuple")
    rows = []
    for value in values:
        if type(value) is not CanonicalRowBoundedCavityProjection:
            raise TypeError("row-bounded cavity projection has a foreign type")
        value.__post_init__()
        if value.host_master not in dims or any(
            guest not in dims for guest in value.guest_masters
        ):
            raise ValueError("row-bounded cavity names an unknown live master")
        if value.upper_boundary_offset_nm != int(dims[value.host_master][1]):
            raise ValueError("row-bounded cavity host envelope height differs")
        rows.append(value)
    canonical = tuple(sorted(rows, key=lambda row: row.obligation_id))
    if tuple(rows) != canonical or len({row.obligation_id for row in rows}) != len(rows):
        raise ValueError("row-bounded cavity projections must be canonical and unique")
    guest_owners = tuple(guest for row in rows for guest in row.guest_masters)
    if len(set(guest_owners)) != len(guest_owners):
        raise ValueError("one live master cannot occupy two row-bounded cavities")
    return canonical

def _add_row_bounded_cavity_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    projections: Sequence[_RowBoundedCavityProjection],
    *,
    minimum_clearance_nm: int,
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    """Materialize intrinsic containment; physical pairs retain disjunctions."""

    gap = int(minimum_clearance_nm)
    if gap < 0:
        raise ValueError("row-bounded cavity clearance must be non-negative")
    canonical = _canonical_row_bounded_cavity_projections(
        tuple(projections),
        dims=dims,
    )
    for row_index, row in enumerate(canonical):
        host = row.host_master
        enforcement = (
            assumption_for(f"row_bounded_cavity:{row_index}")
            if assumption_for is not None
            else None
        )
        for guest in row.guest_masters:
            constraints = (
                model.Add(xs[guest] >= xs[host]),
                model.Add(
                xs[guest] + int(dims[guest][0])
                <= xs[host] + int(dims[host][0])
                ),
                model.Add(
                ys[guest]
                >= ys[host] + int(row.lower_boundary_offset_nm) + gap
                ),
                model.Add(
                ys[guest] + int(dims[guest][1]) + gap
                <= ys[host] + int(row.upper_boundary_offset_nm)
                ),
            )
            if enforcement is not None:
                for constraint in constraints:
                    constraint.OnlyEnforceIf(enforcement)

def _row_bounded_cavity_satisfied(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    projection: _RowBoundedCavityProjection,
    minimum_clearance_nm: int,
) -> bool:
    gap = int(minimum_clearance_nm)
    host_x, host_y = positions[projection.host_master]
    host_width = int(dims[projection.host_master][0])
    return all(
        host_x <= positions[guest][0]
        and positions[guest][0] + int(dims[guest][0]) <= host_x + host_width
        and positions[guest][1]
        >= host_y + int(projection.lower_boundary_offset_nm) + gap
        and positions[guest][1] + int(dims[guest][1]) + gap
        <= host_y + int(projection.upper_boundary_offset_nm)
        for guest in projection.guest_masters
    )

def _collision_atoms_by_owner(
    *,
    dims: Mapping[str, Tuple[int, int]],
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
) -> Dict[str, Tuple[Tuple[Tuple[int, int] | None, _Bbox], ...]]:
    """Return the strongest complete collision authority for every rigid owner."""

    compounds = dict(_canonical_rigid_compound_footprints(
        rigid_compound_footprints,
        dims=dims,
    ))
    layered = dict(_canonical_layered_collision_footprints(
        layered_collision_footprints,
        dims=dims,
    ))
    # A layered inventory is the complete artwork refinement of the coarse
    # member-envelope footprint and therefore has explicit precedence when both
    # are certified for one owner.
    return {
        owner: (
            tuple(
                ((atom[0], atom[1]), (atom[2], atom[3], atom[4], atom[5]))
                for atom in layered[owner]
            )
            if owner in layered
            else tuple((None, rectangle) for rectangle in compounds[owner])
            if owner in compounds
            else ((None, (0, 0, dims[owner][0], dims[owner][1])),)
        )
        for owner in sorted(dims)
    }

def _pair_collision_forbidden_box_cover(
    left_atoms: Sequence[Tuple[Tuple[int, int] | None, _Bbox]],
    right_atoms: Sequence[Tuple[Tuple[int, int] | None, _Bbox]],
    *,
    required_nm: int,
) -> Tuple[Tuple[_Bbox, ...], int, int]:
    """Use exact derived geometry; the enclosing materialization owns reuse."""
    from clay.layout.place.collision_geometry_session import memoized_collision_union
    from clay.layout.spatial.collision_union import forbidden_box_cover

    return memoized_collision_union(left_atoms, right_atoms,
        required_nm=required_nm, compute=forbidden_box_cover)

def _canonical_floorplan_column_groups(
    values: object,
    *,
    block_ids: object,
) -> Tuple[_FloorplanColumnGroup, ...]:
    """Validate exact functional-group centroid expressions without coercion."""

    if type(values) is not tuple:
        raise TypeError("floorplan_column_groups must be one exact tuple")
    if isinstance(block_ids, Mapping):
        widths = {
            str(name): int(bbox[2]) - int(bbox[0])
            for name, bbox in block_ids.items()
        }
        known = set(widths)
    else:
        widths = {}
        known = {str(name) for name in block_ids}  # type: ignore[union-attr]
    rows: List[_FloorplanColumnGroup] = []
    for raw in values:
        if type(raw) is not tuple or len(raw) != 5:
            raise TypeError(
                "floorplan_column_groups must contain exact five-item tuples"
            )
        group, row, column, member_count, raw_terms = raw
        if type(group) is not str or not group or group != group.strip():
            raise TypeError("floorplan column group must be one canonical string")
        if type(row) is not int or row < _GLOBAL_FLOORPLAN_COLUMN_SCOPE:
            raise TypeError(
                "floorplan group scope must be global (-1) or a non-negative row"
            )
        if type(column) is not int or column < 0:
            raise TypeError("floorplan group column must be one non-negative exact int")
        if type(member_count) is not int or member_count <= 0:
            raise TypeError("floorplan group member count must be one positive exact int")
        if type(raw_terms) is not tuple or not raw_terms:
            raise TypeError("floorplan column group terms must be one non-empty tuple")
        terms: List[_FloorplanColumnTerm] = []
        for raw_term in raw_terms:
            if type(raw_term) is not tuple or len(raw_term) != 3:
                raise TypeError(
                    "floorplan column group terms must be exact three-item tuples"
                )
            owner, count, center_offset2_sum = raw_term
            if type(owner) is not str or not owner or owner != owner.strip():
                raise TypeError("floorplan group owner must be one canonical string")
            if owner not in known:
                raise ValueError(f"floorplan group names unknown block {owner!r}")
            if type(count) is not int or count <= 0:
                raise TypeError("floorplan group term count must be one positive exact int")
            if type(center_offset2_sum) is not int or center_offset2_sum < 0:
                raise TypeError("floorplan group center sum must be one non-negative int")
            if widths and center_offset2_sum > 2 * widths[owner] * count:
                raise ValueError("floorplan group center sum escapes its rigid master")
            terms.append((owner, count, center_offset2_sum))
        canonical_terms = tuple(sorted(terms))
        if tuple(raw_terms) != canonical_terms or len(
            {owner for owner, _count, _offset in terms}
        ) != len(terms):
            raise ValueError("floorplan column group terms must be sorted and unique")
        if sum(count for _owner, count, _offset in terms) != member_count:
            raise ValueError("floorplan column group term counts do not conserve members")
        rows.append((group, row, column, member_count, canonical_terms))
    canonical = tuple(sorted(rows))
    if tuple(values) != canonical or len({row[0] for row in rows}) != len(rows):
        raise ValueError("floorplan_column_groups must be sorted with unique groups")
    return canonical

def _floorplan_column_group_order_satisfied(
    blocks: Sequence[Tuple[str, _Bbox]],
    column_groups: Sequence[_FloorplanColumnGroup],
) -> bool:
    """Replay strict ordering of aggregate anonymous macro-cell centroids."""

    by_bbox = dict(blocks)

    cell_counts: Dict[Tuple[int, int], int] = {}
    cell_sums: Dict[Tuple[int, int], int] = {}
    for _group, row, column, member_count, terms in column_groups:
        cell = (row, column)
        cell_counts[cell] = cell_counts.get(cell, 0) + member_count
        cell_sums[cell] = cell_sums.get(cell, 0) + sum(
            2 * count * by_bbox[owner][0] + center_offset2_sum
            for owner, count, center_offset2_sum in terms
        )
    for left_cell, left_count in cell_counts.items():
        for right_cell, right_count in cell_counts.items():
            if left_cell[0] != right_cell[0] or left_cell[1] >= right_cell[1]:
                continue
            if (
                right_count * cell_sums[left_cell]
                >= left_count * cell_sums[right_cell]
            ):
                return False
    return True

def _floorplan_cell_cohesion_groups(
    values: object,
) -> Tuple[Tuple[str, ...], ...]:
    """Return canonical master groups that share one reference floorplan cell.

    A ``(row, column)`` cell is a coordinate-free functional-module identity.
    Its member masters remain independent rigid geometry; the pairs are used
    only as a constructive-seed proximity objective, never as a hard spacing
    claim.  Replaying the exact group inventory here prevents a malformed or
    cross-cell owner from being silently normalized into a weaker objective.
    """

    if type(values) is not tuple:
        raise TypeError("floorplan cell cohesion requires one exact tuple")
    raw_owners: set[str] = set()
    for raw in values:
        if type(raw) is not tuple or len(raw) != 5:
            raise TypeError(
                "floorplan cell cohesion requires exact five-item group rows"
            )
        raw_terms = raw[4]
        if type(raw_terms) is not tuple or not raw_terms:
            raise TypeError(
                "floorplan cell cohesion requires non-empty exact term tuples"
            )
        for term in raw_terms:
            if type(term) is not tuple or len(term) != 3 or type(term[0]) is not str:
                raise TypeError(
                    "floorplan cell cohesion requires exact master terms"
                )
            raw_owners.add(term[0])
    canonical = _canonical_floorplan_column_groups(
        values,
        block_ids=tuple(sorted(raw_owners)),
    )
    owners_by_cell: Dict[Tuple[int, int], set[str]] = {}
    cells_by_owner: Dict[str, set[Tuple[int, int]]] = {}
    for _group, row, column, _member_count, terms in canonical:
        if row == _GLOBAL_FLOORPLAN_COLUMN_SCOPE:
            continue
        cell = (row, column)
        for owner, _count, _offset in terms:
            owners_by_cell.setdefault(cell, set()).add(owner)
            cells_by_owner.setdefault(owner, set()).add(cell)
    crossing = {
        owner: tuple(sorted(cells))
        for owner, cells in cells_by_owner.items()
        if len(cells) != 1
    }
    if crossing:
        raise ValueError(
            "floorplan cell cohesion owner crosses declared cells: "
            f"{crossing!r}"
        )
    return tuple(
        owners
        for cell in sorted(owners_by_cell)
        if len(owners := tuple(sorted(owners_by_cell[cell]))) >= 2
    )

def _canonical_overlap_compatible_pairs(
    value: Any,
    *,
    block_ids: Sequence[str],
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
) -> Tuple[Tuple[str, str], ...]:
    """Validate a signed owner-pair inventory without weakening explicit rules."""

    if value is None:
        return ()
    if type(value) is not tuple:
        raise TypeError("overlap-compatible pairs must be one exact tuple")
    known = set(block_ids)
    rows = []
    for pair in value:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or pair[0] >= pair[1]
            or pair[0] not in known
            or pair[1] not in known
        ):
            raise ValueError("overlap-compatible pair is malformed or unowned")
        if pair in pair_clearances_nm:
            raise ValueError(
                "overlap-compatible pair conflicts with a pair-selective clearance"
            )
        rows.append(pair)
    canonical = tuple(rows)
    if canonical != tuple(sorted(set(canonical))):
        raise ValueError("overlap-compatible pairs must be canonical and unique")
    return canonical

def _canonical_adjacency_limits_nm(
    value: Any,
    *,
    block_ids: Sequence[str],
    base_gap_nm: int,
) -> Tuple[Tuple[str, str, int], ...]:
    """Validate hard edge-adjacency limits between materialized blocks."""

    if value is None:
        return ()
    if type(value) is not tuple:
        raise TypeError("adjacency limits must be an exact tuple")
    present = set(block_ids)
    by_pair: Dict[Tuple[str, str], int] = {}
    for row in value:
        if type(row) is not tuple or len(row) != 3:
            raise TypeError("adjacency limits must contain exact triples")
        left, right, maximum = row
        if (
            type(left) is not str
            or type(right) is not str
            or not left
            or not right
            or left == right
            or left not in present
            or right not in present
            or type(maximum) is not int
            or maximum < base_gap_nm
        ):
            raise ValueError("adjacency limit has an invalid block or distance")
        pair = tuple(sorted((left, right)))
        by_pair[pair] = min(by_pair.get(pair, maximum), maximum)
    return tuple((left, right, by_pair[(left, right)]) for left, right in sorted(by_pair))

def _canonical_adjacency_scope_offsets_nm(
    value: Any,
    *,
    blocks: Sequence[Tuple[str, _Bbox]],
    flexible_envelope_constraints: Tuple[FlexibleEnvelopeConstraint, ...] = (),
) -> Dict[str, _Bbox]:
    """Bind exact structural envelopes to their translating rigid masters.

    Collision legality continues to use the complete master bbox.  These local
    offsets are consumed only by ``@adjacent`` so a shareable layer that expands
    a master (for example a well later merged by the canonical emitter) cannot
    make the solver overstate how close the independently claimable structure is.
    """

    masters = dict(blocks)
    flexible_offsets = {
        member: bbox
        for constraint in flexible_envelope_constraints
        for member, bbox in constraint.member_adjacency_offsets_nm
    }
    defaults = {
        name: (0, 0, bbox[2] - bbox[0], bbox[3] - bbox[1])
        for name, bbox in masters.items()
    }
    if value is None or value == ():
        return defaults
    if type(value) is not tuple:
        raise TypeError("adjacency scope bboxes must be an exact tuple")
    scopes: Dict[str, _Bbox] = {}
    for row in value:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("adjacency scope bboxes must contain exact pairs")
        name, raw_bbox = row
        if type(name) is not str or not name or type(raw_bbox) is not tuple:
            raise TypeError("adjacency scope bbox entries must be exact values")
        if name in scopes or name not in masters:
            raise ValueError("adjacency scope bboxes repeat or name a foreign master")
        bbox = tuple(raw_bbox)
        if len(bbox) != 4 or any(type(item) is not int for item in bbox):
            raise TypeError("adjacency scope bboxes must contain four exact integers")
        master = masters[name]
        inside_master = (
            master[0] <= bbox[0] < bbox[2] <= master[2]
            and master[1] <= bbox[1] < bbox[3] <= master[3]
        )
        offset = (
            bbox[0] - master[0],
            bbox[1] - master[1],
            bbox[2] - master[0],
            bbox[3] - master[1],
        )
        if not inside_master and flexible_offsets.get(name) != offset:
            raise ValueError("adjacency structural scope must be inside its master")
        scopes[name] = offset
    if set(scopes) != set(masters):
        raise ValueError("adjacency scope bboxes must cover every rigid master")
    return dict(sorted(scopes.items()))

def _project_flexible_adjacency_scope_offsets_nm(
    scopes: Mapping[str, _Bbox],
    *,
    constraints: Tuple[FlexibleEnvelopeConstraint, ...],
    adjacency_limits_nm: Sequence[Tuple[str, str, int]],
    dims: Mapping[str, Tuple[int, int]],
) -> Dict[str, _Bbox]:
    """Restore exact physical adjacency scopes after virtualizing ring content.

    A flexible member's CP rectangle is its independently movable content, not
    the rigid ring-inclusive bbox from which ``@adjacent`` was originally
    lowered.  The constraint therefore carries the exact source-local
    structural scope.  Every participating flexible endpoint must replay that
    scope; a missing or different view is a typed stop, never a fallback to the
    smaller content rectangle.
    """

    projected = dict(scopes)
    declared = {
        member: bbox
        for constraint in constraints
        for member, bbox in constraint.member_adjacency_offsets_nm
    }
    flexible_members = {
        member for constraint in constraints for member in constraint.members
    }
    endpoints = {
        endpoint
        for left, right, _maximum in adjacency_limits_nm
        for endpoint in (left, right)
    }
    for member in sorted(flexible_members & endpoints):
        if member not in declared:
            raise ValueError(
                "flexible adjacency endpoint lacks an exact structural scope"
            )
        default = (0, 0, dims[member][0], dims[member][1])
        if projected.get(member) not in (default, declared[member]):
            raise ValueError(
                "flexible adjacency structural scope differs from its authority"
            )
        projected[member] = declared[member]
    return dict(sorted(projected.items()))

def _canonical_axis_alignment_groups(
    value: Any,
    *,
    block_ids: Sequence[str],
    dims: Mapping[str, Tuple[int, int]],
) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    """Validate hard vertical-axis relations between rigid blocks.

    Each row is ``(block_id, axis_offset_twice_nm)``.  The doubled offset is
    measured from the block's left edge and preserves a half-nanometre axis
    without introducing real-valued CP variables.  All rows in one group obey
    ``2*x0 + offset2 == common_axis2``.
    """

    try:
        raw_groups = tuple(value or ())
    except TypeError as exc:
        raise ValueError("axis alignment groups must be a finite sequence") from exc
    present = {str(block_id) for block_id in block_ids}
    groups: set[Tuple[Tuple[str, int], ...]] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, (tuple, list)):
            raise ValueError("axis alignment group must be a finite sequence")
        entries: list[Tuple[str, int]] = []
        for raw_entry in raw_group:
            if not isinstance(raw_entry, (tuple, list)) or len(raw_entry) != 2:
                raise ValueError(
                    "axis alignment entries must be (block_id, offset2_nm) pairs"
                )
            block_id, offset2 = str(raw_entry[0]), raw_entry[1]
            if (
                not block_id
                or block_id not in present
                or type(offset2) is not int
                or offset2 < 0
                or offset2 > 2 * int(dims[block_id][0])
            ):
                raise ValueError("axis alignment entry has an invalid block or offset")
            entries.append((block_id, int(offset2)))
        entries.sort()
        if len(entries) < 2 or len({name for name, _offset in entries}) != len(entries):
            raise ValueError("axis alignment groups require at least two unique blocks")
        if len({offset & 1 for _name, offset in entries}) != 1:
            raise ValueError(
                "axis alignment offsets must have one common half-grid parity"
            )
        groups.add(tuple(entries))
    return tuple(sorted(groups))

def _canonical_center_alignment_pairs(
    value: Any,
    *,
    block_ids: Sequence[str],
    dims: Mapping[str, Tuple[int, int]],
) -> Tuple[Tuple[str, str, str], ...]:
    """Validate symmetric exact centre relations between rigid blocks."""

    try:
        raw_pairs = tuple(value or ())
    except TypeError as exc:
        raise ValueError("center alignment pairs must be a finite sequence") from exc
    present = {str(block_id) for block_id in block_ids}
    pairs: set[Tuple[str, str, str]] = set()
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 3:
            raise ValueError(
                "center alignment entries must be (left, right, axis) triples"
            )
        left, right = sorted((str(raw_pair[0]), str(raw_pair[1])))
        axis = str(raw_pair[2])
        if (
            not left
            or left == right
            or left not in present
            or right not in present
            or axis not in {"horizontal", "vertical"}
        ):
            raise ValueError("center alignment entry has invalid blocks or axis")
        dimension = 0 if axis == "horizontal" else 1
        if (int(dims[left][dimension]) - int(dims[right][dimension])) & 1:
            raise ValueError(
                "center alignment cannot represent a half-nanometre origin"
            )
        pairs.add((left, right, axis))
    if len(pairs) != len(raw_pairs):
        raise ValueError("center alignment pairs must be unique")
    return tuple(sorted(pairs))

def _axis_alignment_required_width_nm(
    groups: Sequence[Sequence[Tuple[str, int]]],
    dims: Mapping[str, Tuple[int, int]],
) -> int:
    """Smallest outline width that can contain every aligned group."""

    required = 0
    for group in groups:
        left2 = min(-offset2 for _name, offset2 in group)
        right2 = max(2 * dims[name][0] - offset2 for name, offset2 in group)
        required = max(required, (right2 - left2 + 1) // 2)
    return required

def _add_axis_alignment_constraints(
    model: Any,
    xs: Mapping[str, Any],
    groups: Sequence[Sequence[Tuple[str, int]]],
    *,
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    for group_index, group in enumerate(groups):
        assumption = (
            assumption_for(f"axis_alignment:{group_index}")
            if assumption_for is not None
            else None
        )
        base_name, base_offset2 = group[0]
        for name, offset2 in group[1:]:
            constraint = model.Add(
                2 * xs[name] + offset2 == 2 * xs[base_name] + base_offset2
            )
            if assumption is not None:
                constraint.OnlyEnforceIf(assumption)

def _add_center_alignment_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    pairs: Sequence[Tuple[str, str, str]],
    *,
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    for index, (left, right, axis) in enumerate(pairs):
        coordinates = xs if axis == "horizontal" else ys
        dimension = 0 if axis == "horizontal" else 1
        constraint = model.Add(
            2 * coordinates[left] + int(dims[left][dimension])
            == 2 * coordinates[right] + int(dims[right][dimension])
        )
        if assumption_for is not None:
            constraint.OnlyEnforceIf(
                assumption_for(f"center_alignment:{index}")
            )

def _add_floorplan_column_constraints(
    model: Any,
    xs: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    column_classes: Sequence[Tuple[str, int, int, int]],
    *,
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    """Enforce the GDS checker's strict global X-centroid column order."""

    for left, left_column, _left_min2, left_max2 in column_classes:
        for right, right_column, right_min2, _right_max2 in column_classes:
            if left_column >= right_column:
                continue
            # Twice-centroid avoids rounding half-nm widths. CP variables are
            # integer nm, so +1 is the exact strict inequality.
            constraint = model.Add(
                2 * xs[right] + right_min2
                >= 2 * xs[left] + left_max2 + 1
            )
            if assumption_for is not None:
                constraint.OnlyEnforceIf(assumption_for("floorplan_column"))

def _add_floorplan_column_group_constraints(
    model: Any,
    xs: Mapping[str, Any],
    column_groups: Sequence[_FloorplanColumnGroup],
    *,
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    """Enforce strict order of aggregate anonymous macro-cell centroids."""

    expressions: Dict[Tuple[int, int], Any] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for _group, row, column, member_count, terms in column_groups:
        cell = (row, column)
        expression = sum(
            2 * count * xs[owner] + center_offset2_sum
            for owner, count, center_offset2_sum in terms
        )
        expressions[cell] = expressions.get(cell, 0) + expression
        counts[cell] = counts.get(cell, 0) + member_count
    for left_cell, left_count in counts.items():
        for right_cell, right_count in counts.items():
            if left_cell[0] != right_cell[0] or left_cell[1] >= right_cell[1]:
                continue
            constraint = model.Add(
                left_count * expressions[right_cell]
                >= right_count * expressions[left_cell] + 1
            )
            if assumption_for is not None:
                constraint.OnlyEnforceIf(
                    assumption_for("floorplan_column_group")
                )

def _add_manufacturing_grid_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    manufacturing_grid_nm: int,
) -> None:
    """Keep every materialized rigid-block origin on the PDK write grid."""

    grid = int(manufacturing_grid_nm)
    if grid <= 0:
        raise ValueError("manufacturing grid must be a positive integer nm")
    if grid == 1:
        return
    for name in sorted(xs):
        model.AddModuloEquality(0, xs[name], grid)
        model.AddModuloEquality(0, ys[name], grid)

def _assert_axis_alignment_positions(
    positions: Mapping[str, Tuple[int, int]],
    groups: Sequence[Sequence[Tuple[str, int]]],
) -> None:
    for group in groups:
        axes = {2 * int(positions[name][0]) + int(offset2) for name, offset2 in group}
        if len(axes) != 1:
            raise RuntimeError(
                "hard differential-axis alignment has no certified incumbent"
            )

def _assert_center_alignment_positions(
    positions: Mapping[str, Tuple[int, int]],
    dims: Mapping[str, Tuple[int, int]],
    pairs: Sequence[Tuple[str, str, str]],
) -> None:
    for left, right, axis in pairs:
        coordinate = 0 if axis == "horizontal" else 1
        dimension = coordinate
        left_center2 = (
            2 * int(positions[left][coordinate]) + int(dims[left][dimension])
        )
        right_center2 = (
            2 * int(positions[right][coordinate]) + int(dims[right][dimension])
        )
        if left_center2 != right_center2:
            raise RuntimeError("hard center alignment has no certified incumbent")

def _bbox_clearance_nm(left: _Bbox, right: _Bbox) -> int:
    return max(
        right[0] - left[2],
        left[0] - right[2],
        right[1] - left[3],
        left[1] - right[3],
    )

def _shelf_pack(
    ids: Sequence[str],
    dims: Dict[str, Tuple[int, int]],
    gap: int,
    target_w: int,
) -> Tuple[Dict[str, Tuple[int, int]], int, int]:
    """Deterministic First-Fit-Decreasing-Height shelf pack -- a guaranteed
    overlap-free placement into a strip of width ``target_w``. Blocks (sorted by
    height desc, then id) are laid left-to-right; a block that would exceed the strip
    width starts a new shelf above. Returns ``(pos, W_used, H_used)`` in nm. Used to
    HINT the CP strip solve an immediate incumbent (and, if CP finds nothing better in
    budget, as the honest fallback)."""
    order = sorted(ids, key=lambda b: (-dims[b][1], b))
    pos: Dict[str, Tuple[int, int]] = {}
    shelf_x = shelf_y = shelf_h = W_used = 0
    for b in order:
        w, h = dims[b]
        if shelf_x > 0 and shelf_x + w > target_w:
            shelf_y += shelf_h + gap
            shelf_x = shelf_h = 0
        pos[b] = (shelf_x, shelf_y)
        shelf_x += w + gap
        shelf_h = max(shelf_h, h)
        W_used = max(W_used, pos[b][0] + w)
    return pos, W_used, shelf_y + shelf_h

def _seed_geometry_legal(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    gap_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
    layer_clearances_nm: Mapping[Tuple[int, int], int] | None = None,
    pair_layer_clearances_nm: Mapping[
        Tuple[str, str, Tuple[int, int]], int
    ] | None = None,
    overlap_compatible_pair_layers: Tuple[
        Tuple[str, str, Tuple[int, int]], ...
    ] = (),
) -> bool:
    from clay.flow.execution_session import current_session
    session = current_session()
    checkpoint = session.checkpoint if session is not None else None
    if checkpoint:
        checkpoint()
    per_layer = dict(layer_clearances_nm or {})
    per_pair_layer = dict(pair_layer_clearances_nm or {})
    compatible_pair_layers = set(overlap_compatible_pair_layers)
    from clay.layout.place.collision_replay_cache import (
        collision_groups, relative_pair_conflicts,
    )
    grouped = collision_groups(tuple(sorted(dims.items())),
        rigid_compound_footprints, layered_collision_footprints)
    owners = tuple(sorted(seed))
    compatible = set(overlap_compatible_pairs)
    for index, left_name in enumerate(owners):
        for right_name in owners[index + 1:]:
            if checkpoint:
                checkpoint()
            if (left_name, right_name) in compatible:
                continue
            for left_layer in grouped[left_name].layers:
                for right_layer in grouped[right_name].layers:
                    same_layer = left_layer is not None and left_layer == right_layer
                    if left_layer is not None and right_layer is not None and not same_layer:
                        continue
                    if same_layer and (left_name, right_name, left_layer) in compatible_pair_layers:
                        continue
                    required = int(pair_clearances_nm.get((left_name, right_name),
                        per_pair_layer.get((left_name, right_name, left_layer),
                            per_layer.get(left_layer, gap_nm)) if same_layer else gap_nm))
                    if relative_pair_conflicts(
                            grouped[left_name], grouped[right_name],
                            left_layer, right_layer, required,
                            seed[left_name][0]-seed[right_name][0],
                            seed[left_name][1]-seed[right_name][1]):
                        return False
    if checkpoint:
        checkpoint()
    return True

def _seed_geometry_failure_rows(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    gap_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
) -> Tuple[Dict[str, object], ...]:
    """Name the exact owner/atom pairs rejected by the geometry predicate."""

    local_atoms = _collision_atoms_by_owner(
        dims=dims,
        rigid_compound_footprints=rigid_compound_footprints,
        layered_collision_footprints=layered_collision_footprints,
    )
    atoms_by_owner = {
        name: tuple(
            (
                layer,
                (
                    x + rectangle[0],
                    y + rectangle[1],
                    x + rectangle[2],
                    y + rectangle[3],
                ),
            )
            for layer, rectangle in local_atoms[name]
        )
        for name, (x, y) in seed.items()
    }
    compatible = set(overlap_compatible_pairs)
    owners = tuple(sorted(atoms_by_owner))
    from clay.layout.spatial.collision_queries import RectangleIndex
    grouped = {}
    indexes = {}
    for owner, atoms in atoms_by_owner.items():
        layers = {}
        for atom_index, (layer, box) in enumerate(atoms):
            layers.setdefault(layer, {}).setdefault(box, []).append(atom_index)
        grouped[owner] = layers
        for layer, boxes in layers.items():
            indexes[owner, layer] = RectangleIndex(tuple(boxes))
    failures = []
    for left_index, left_name in enumerate(owners):
        for right_name in owners[left_index + 1 :]:
            if (left_name, right_name) in compatible:
                continue
            required = int(pair_clearances_nm.get(
                (left_name, right_name), gap_nm
            ))
            for left_atom_index, (left_layer, left_box) in enumerate(
                atoms_by_owner[left_name]
            ):
                for right_layer, right_boxes in grouped[right_name].items():
                    if (
                        left_layer is not None
                        and right_layer is not None
                        and left_layer != right_layer
                    ):
                        continue
                    query = (left_box[0]-required, left_box[1]-required,
                             left_box[2]+required, left_box[3]+required)
                    candidates = (indexes[right_name, right_layer].intersecting(query)
                                  if required >= 0 else tuple(right_boxes))
                    for right_box in candidates:
                        if _bbox_clearance_nm(left_box, right_box) >= required:
                            continue
                        layer_id = (
                            "untyped"
                            if left_layer is None or right_layer is None
                            else f"L{left_layer[0]}D{left_layer[1]}"
                        )
                        for right_atom_index in right_boxes[right_box]:
                            failures.append(_full_hard_replay_row(
                                (
                                    f"physical_nonoverlap:{left_name}<->{right_name}:"
                                    f"{layer_id}:atom{left_atom_index}<->"
                                    f"atom{right_atom_index}"
                                ),
                                "physical_nonoverlap",
                            ))
    return _canonical_full_hard_failure_rows(failures)

def _assert_no_physical_block_overlap(
    blocks: Sequence[Tuple[str, _Bbox]],
    *,
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
    overlap_compatible_pair_layers: Tuple[Tuple[str, str, Tuple[int, int]], ...] = (),
) -> None:
    """Assert collision-atom legality while retaining rigid owner envelopes."""

    by_owner = dict(_exact_cut_input_blocks(tuple(blocks)))
    dims = {
        owner: (bbox[2] - bbox[0], bbox[3] - bbox[1])
        for owner, bbox in by_owner.items()
    }
    local_atoms = _collision_atoms_by_owner(
        dims=dims,
        rigid_compound_footprints=rigid_compound_footprints,
        layered_collision_footprints=layered_collision_footprints,
    )
    physical = {
        owner: tuple(
            (
                layer,
                (
                    bbox[0] + local[0],
                    bbox[1] + local[1],
                    bbox[0] + local[2],
                    bbox[1] + local[3],
                ),
            )
            for layer, local in local_atoms[owner]
        )
        for owner, bbox in by_owner.items()
    }
    owners = tuple(sorted(physical))
    compatible = set(_canonical_overlap_compatible_pairs(
        overlap_compatible_pairs,
        block_ids=owners,
    ))
    compatible_layers = set(overlap_compatible_pair_layers)
    for index, left_owner in enumerate(owners):
        for right_owner in owners[index + 1 :]:
            if (left_owner, right_owner) in compatible:
                continue
            for left_layer, left in physical[left_owner]:
                for right_layer, right in physical[right_owner]:
                    if (left_layer is not None and left_layer == right_layer
                            and (left_owner, right_owner, left_layer) in compatible_layers):
                        continue
                    if (
                        left_layer is not None
                        and right_layer is not None
                        and left_layer != right_layer
                    ):
                        continue
                    if _bbox_clearance_nm(left, right) >= 0:
                        continue
                    overlap = (
                        min(left[2], right[2]) - max(left[0], right[0])
                    ) * (
                        min(left[3], right[3]) - max(left[1], right[1])
                    )
                    raise AssertionError(
                        f"physical block overlap: {left_owner} {left} <> "
                        f"{right_owner} {right} (intersection {overlap} nm^2 = "
                        f"{overlap / 1e6:.2f} um^2)"
                    )

def _flexible_seed_physical_views(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    constraints: Tuple[FlexibleEnvelopeConstraint, ...],
) -> Tuple[Dict[str, _Bbox], Dict[str, _Bbox]]:
    """Project exact ring-inclusive member and aggregate bboxes for one seed."""

    canonical = _canonical_flexible_envelope_constraints(
        constraints,
        block_ids=tuple(sorted(dims)),
    )
    member_boxes: Dict[str, _Bbox] = {}
    group_boxes: Dict[str, _Bbox] = {}
    for row in canonical:
        offsets = dict(row.member_physical_offsets_nm)
        projected: list[_Bbox] = []
        for member in row.members:
            x, y = seed[member]
            x0, y0, x1, y1 = offsets[member]
            width, height = dims[member]
            if not (x0 <= 0 < width <= x1 and y0 <= 0 < height <= y1):
                raise ValueError(
                    "flexible envelope physical footprint must contain its content"
                )
            bbox = (x + x0, y + y0, x + x1, y + y1)
            member_boxes[member] = bbox
            projected.append(bbox)
        group_boxes[row.group] = (
            min(bbox[0] for bbox in projected),
            min(bbox[1] for bbox in projected),
            max(bbox[2] for bbox in projected),
            max(bbox[3] for bbox in projected),
        )
    return member_boxes, group_boxes

def _add_pair_clearance_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    *,
    excluded_blocks: Sequence[str] = (),
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    """Add exact four-way separation disjunctions for selected block pairs."""

    excluded = set(excluded_blocks)
    for (left, right), required in sorted(pair_clearances_nm.items()):
        if left in excluded or right in excluded:
            continue
        lw, lh = dims[left]
        rw, rh = dims[right]
        left_of = model.NewBoolVar(f"pair_left_{left}_{right}")
        right_of = model.NewBoolVar(f"pair_right_{left}_{right}")
        below = model.NewBoolVar(f"pair_below_{left}_{right}")
        above = model.NewBoolVar(f"pair_above_{left}_{right}")
        model.Add(xs[left] + lw + required <= xs[right]).OnlyEnforceIf(left_of)
        model.Add(xs[right] + rw + required <= xs[left]).OnlyEnforceIf(right_of)
        model.Add(ys[left] + lh + required <= ys[right]).OnlyEnforceIf(below)
        model.Add(ys[right] + rh + required <= ys[left]).OnlyEnforceIf(above)
        disjunction = model.AddBoolOr((left_of, right_of, below, above))
        if assumption_for is not None:
            disjunction.OnlyEnforceIf(assumption_for("pair_clearance"))

def _add_rigid_compound_collision_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    footprints: Tuple[_RigidCompoundFootprint, ...],
    *,
    collision_groups: Sequence[Sequence[str]],
    gap_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
) -> None:
    """Exclude foreign rectangles from rigid members, while preserving macro voids."""

    compounds = dict(_canonical_rigid_compound_footprints(footprints, dims=dims))
    if not compounds:
        return
    seen_pairs: set[Tuple[str, str]] = set()
    compatible = set(overlap_compatible_pairs)
    constraint_index = 0
    for raw_group in collision_groups:
        group = tuple(sorted(set(raw_group)))
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                if left not in compounds and right not in compounds:
                    continue
                pair = (left, right)
                if pair in compatible:
                    continue
                both_compound = left in compounds and right in compounds
                if not both_compound and pair not in pair_clearances_nm:
                    # Base compound-to-regular clearance is carried by the
                    # native NoOverlap2D projections.  Keep explicit BoolOrs
                    # only for raised pair clearances and compound pairs.
                    continue
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                required = int(pair_clearances_nm.get(pair, gap_nm))
                left_rectangles = compounds.get(
                    left,
                    ((0, 0, dims[left][0], dims[left][1]),),
                )
                right_rectangles = compounds.get(
                    right,
                    ((0, 0, dims[right][0], dims[right][1]),),
                )
                for left_rect in left_rectangles:
                    for right_rect in right_rectangles:
                        directions = tuple(
                            model.NewBoolVar(
                                f"compound_collision_{constraint_index}_{axis}"
                            )
                            for axis in range(4)
                        )
                        model.Add(
                            xs[left] + left_rect[2] + required
                            <= xs[right] + right_rect[0]
                        ).OnlyEnforceIf(directions[0])
                        model.Add(
                            xs[right] + right_rect[2] + required
                            <= xs[left] + left_rect[0]
                        ).OnlyEnforceIf(directions[1])
                        model.Add(
                            ys[left] + left_rect[3] + required
                            <= ys[right] + right_rect[1]
                        ).OnlyEnforceIf(directions[2])
                        model.Add(
                            ys[right] + right_rect[3] + required
                            <= ys[left] + left_rect[1]
                        ).OnlyEnforceIf(directions[3])
                        model.AddBoolOr(directions)
                        constraint_index += 1

def _add_adjacency_constraints(
    model: Any,
    xs: Mapping[str, Any],
    ys: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    limits: Sequence[Tuple[str, str, int]],
    *,
    scope_offsets_nm: Mapping[str, _Bbox],
    minimum_gap_nm: int,
    projection_nm: int,
    assumption_for: Optional[Callable[[str], Any]] = None,
    diagnostic_literal_sink: Optional[list[Tuple[int, Tuple[Tuple[str, Any], ...]]]] = None,
) -> None:
    """Require each pair to share an edge projection within its maximum gap."""

    for index, (left, right, maximum) in enumerate(limits):
        lw, lh = dims[left]
        rw, rh = dims[right]
        lx0, ly0, lx1, ly1 = scope_offsets_nm[left]
        rx0, ry0, rx1, ry1 = scope_offsets_nm[right]
        directions = [
            model.NewBoolVar(f"adj_{name}_{left}_{right}")
            for name in ("left", "right", "below", "above")
        ]
        left_of, right_of, below, above = directions
        if diagnostic_literal_sink is not None:
            diagnostic_literal_sink.append((index, tuple(zip(
                ("left", "right", "below", "above"),
                directions,
            ))))

        model.Add(xs[left] + lw + minimum_gap_nm <= xs[right]).OnlyEnforceIf(left_of)
        model.Add(xs[right] + rx0 <= xs[left] + lx1 + maximum).OnlyEnforceIf(left_of)
        model.Add(ys[left] + ly0 + projection_nm <= ys[right] + ry1).OnlyEnforceIf(left_of)
        model.Add(ys[right] + ry0 + projection_nm <= ys[left] + ly1).OnlyEnforceIf(left_of)

        model.Add(xs[right] + rw + minimum_gap_nm <= xs[left]).OnlyEnforceIf(right_of)
        model.Add(xs[left] + lx0 <= xs[right] + rx1 + maximum).OnlyEnforceIf(right_of)
        model.Add(ys[left] + ly0 + projection_nm <= ys[right] + ry1).OnlyEnforceIf(right_of)
        model.Add(ys[right] + ry0 + projection_nm <= ys[left] + ly1).OnlyEnforceIf(right_of)

        model.Add(ys[left] + lh + minimum_gap_nm <= ys[right]).OnlyEnforceIf(below)
        model.Add(ys[right] + ry0 <= ys[left] + ly1 + maximum).OnlyEnforceIf(below)
        model.Add(xs[left] + lx0 + projection_nm <= xs[right] + rx1).OnlyEnforceIf(below)
        model.Add(xs[right] + rx0 + projection_nm <= xs[left] + lx1).OnlyEnforceIf(below)

        model.Add(ys[right] + rh + minimum_gap_nm <= ys[left]).OnlyEnforceIf(above)
        model.Add(ys[left] + ly0 <= ys[right] + ry1 + maximum).OnlyEnforceIf(above)
        model.Add(xs[left] + lx0 + projection_nm <= xs[right] + rx1).OnlyEnforceIf(above)
        model.Add(xs[right] + rx0 + projection_nm <= xs[left] + lx1).OnlyEnforceIf(above)
        if assumption_for is None:
            model.AddExactlyOne(directions)
        else:
            model.Add(sum(directions) == 1).OnlyEnforceIf(
                assumption_for(f"adjacency:{index}")
            )

def _add_bottom_edge_keepout_constraints(
    model: Any,
    ys: Mapping[str, Any],
    keepout_nm: int,
    exempt: set[str],
    clear: Sequence[str] = (),
) -> None:
    """Keep conflicts above the rail; distinguish carriers from clear blocks."""

    for block_id in sorted(ys):
        if block_id in exempt:
            model.Add(ys[block_id] == 0)
        elif block_id not in clear:
            model.Add(ys[block_id] >= int(keepout_nm))

def _add_top_edge_keepout_constraints(
    model: Any,
    physical_top_by_master: Mapping[str, Any],
    outline_top: Any,
    keepout_nm: int,
    clear: Sequence[str] = (),
) -> None:
    """Keep rail-layer blockers below the top edge's routed keepout band."""

    clear_masters = set(clear)
    for block_id in sorted(physical_top_by_master):
        if block_id not in clear_masters:
            model.Add(
                physical_top_by_master[block_id] + int(keepout_nm)
                <= outline_top
            )

def _add_hard_horizontal_edge_constraints(
    model: Any,
    xs: Mapping[str, Any],
    dims: Mapping[str, Tuple[int, int]],
    *,
    width_bound: Any,
    right_edge: Sequence[str],
    left_edge: Sequence[str],
    assumption_for: Optional[Callable[[str], Any]] = None,
) -> None:
    """Materialize exact left/right ``@anchor`` declarations in the CP master."""

    # This helper is also consumed by scoped band/component submodels.  Their
    # declaration inventory can legitimately include owners outside ``xs``;
    # only the intersection belongs to this model.  Indexing the unscoped
    # inventory used to raise a P0 KeyError before the complete master replay.
    scoped = set(xs)
    for block_id in sorted(set(right_edge) & scoped):
        constraint = model.Add(xs[block_id] + dims[block_id][0] == width_bound)
        if assumption_for is not None:
            constraint.OnlyEnforceIf(
                assumption_for(f"horizontal_edge:right:{block_id}")
            )
    for block_id in sorted(set(left_edge) & scoped):
        constraint = model.Add(xs[block_id] == 0)
        if assumption_for is not None:
            constraint.OnlyEnforceIf(
                assumption_for(f"horizontal_edge:left:{block_id}")
            )

def _right_edge_compatible_width_bound(
    width_bound: int,
    dims: Mapping[str, Tuple[int, int]],
    right_edge: Sequence[str],
    manufacturing_grid_nm: int,
) -> int:
    """Return the first outline width compatible with every hard right edge.

    Block origins, rather than macro dimensions, are constrained to the PDK
    manufacturing grid.  Consequently all blocks sharing one exact right edge
    must have the same width residue modulo that grid.  Select that mechanically
    derived outline residue, or fail closed when no such physical placement can
    exist.
    """

    edge_blocks = tuple(sorted(set(right_edge)))
    grid = int(manufacturing_grid_nm)
    if not edge_blocks or grid == 1:
        return int(width_bound)
    residues = {int(dims[block_id][0]) % grid for block_id in edge_blocks}
    if len(residues) != 1:
        raise RuntimeError(
            "hard right-edge anchors have incompatible manufacturing-grid residues"
        )
    residue = next(iter(residues))
    return int(width_bound) + ((residue - int(width_bound)) % grid)

def _stable_hard_order(
    nodes: Sequence[str], order_pairs: Sequence[Tuple[str, str]]
) -> List[str]:
    """Stable topological order for the HARD same-band left/right relation."""
    names = sorted(set(str(node) for node in nodes))
    present = set(names)
    successors = {name: set() for name in names}
    indegree = {name: 0 for name in names}
    for left, right in sorted({(str(a), str(b)) for a, b in order_pairs or ()}):
        if left not in present or right not in present:
            continue
        if right not in successors[left]:
            successors[left].add(right)
            indegree[right] += 1
    ready = sorted(name for name in names if indegree[name] == 0)
    ordered: List[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for successor in sorted(successors[name]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(ordered) != len(names):
        cyclic = sorted(name for name in names if indegree[name] > 0)
        raise ValueError(f"hard order cycle among blocks: {', '.join(cyclic)}")
    return ordered

def _hard_order_width_nm(
    nodes: Sequence[str],
    dims: Mapping[str, Tuple[int, int]],
    order_pairs: Sequence[Tuple[str, str]],
    gap_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
) -> int:
    """Return the exact longest left-to-right chain width for one region."""

    present = set(str(node) for node in nodes)
    if not present:
        return 0
    pairs = tuple(
        sorted(
            (str(left), str(right))
            for left, right in order_pairs or ()
            if str(left) in present and str(right) in present
        )
    )
    ordered = _stable_hard_order(tuple(present), pairs)
    predecessors: Dict[str, list[str]] = {node: [] for node in ordered}
    for left, right in pairs:
        predecessors[right].append(left)
    width_to: Dict[str, int] = {}
    for node in ordered:
        own_width = int(dims[node][0])
        width_to[node] = max(
            (
                width_to[prior]
                + _required_pair_gap_nm(
                    prior,
                    node,
                    int(gap_nm),
                    pair_clearances_nm,
                )
                + own_width
                for prior in predecessors[node]
            ),
            default=own_width,
        )
    return max(width_to.values(), default=0)

def _required_pair_gap_nm(
    left: str,
    right: str,
    gap_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
) -> int:
    return int(pair_clearances_nm.get(tuple(sorted((left, right))), gap_nm))

def _ordered_row_positions(
    row: Sequence[str],
    dims: Mapping[str, Tuple[int, int]],
    *,
    x_start: int,
    y: int,
    gap_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """LP-compact one already-ordered row under every selected pair gap."""

    ordered = tuple(str(block_id) for block_id in row)
    if not ordered:
        return {}, int(x_start)
    constraints: list[_lp_compactor.DifferenceConstraint] = []
    for right_index, right in enumerate(ordered):
        for left_index, left in enumerate(ordered[:right_index]):
            required_gap = _required_pair_gap_nm(
                left, right, gap_nm, pair_clearances_nm
            )
            if left_index == right_index - 1:
                required_gap = max(int(gap_nm), required_gap)
            constraints.append(
                _lp_compactor.DifferenceConstraint(
                    before=left,
                    after=right,
                    minimum_delta_nm=int(dims[left][0]) + required_gap,
                    reason="topology-order-and-pdk-gap",
                )
            )
    compacted = _lp_compactor.compact_axis(
        {block_id: int(dims[block_id][0]) for block_id in ordered},
        tuple(constraints),
    )
    offset = int(x_start)
    placed = {
        block_id: (offset + compacted.positions_nm[block_id], int(y))
        for block_id in ordered
    }
    return placed, offset + compacted.extent_nm

def _landscape_place_enabled() -> bool:
    """Wave D LANDSCAPE-DENSE placement (env ``CLAY_LANDSCAPE_PLACE``), OFF by default.

    OFF -> the historical reference-skeleton floorplan is byte-identical (the cap bank
    is parked in the exclusive @band(center) vertical tier + the @common_centroid
    left-column skeleton adds a dead OD band -> a portrait cell). ON -> the declared
    ``@adjacent(cap, mos)`` is RESOLVED over ``@band(center=passives)``: the cap
    COMPOSITE co-bands with its declared @adjacent MOS neighbour (occupies WIDTH inside
    a device band, not a height tier) and the inferred left-column skeleton is dropped.
    A flag-gated MEASUREMENT path (the parent authorises it); the placement DECISION
    still comes from the .clay's @adjacent, never a hardcoded cap rule. Read fresh."""
    import os

    return str(os.environ.get("CLAY_LANDSCAPE_PLACE", "")).strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )

def _banded_shelf_pack(
    ids: Sequence[str],
    dims: Dict[str, Tuple[int, int]],
    gap: int,
    bands: Dict[str, int],
    present: Sequence[int],
    order_pairs: Sequence[Tuple[str, str]] = (),
    bottom_center_keepout_nm: int = 0,
    bottom_edge_keepout_nm: int = 0,
    bottom_edge_keepout_exempt: Sequence[str] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
) -> Dict[str, Tuple[int, int]]:
    """A guaranteed overlap-free, BAND-ORDERED seed: lay each band (ascending
    ordinal, bottom -> top) as one left-to-right row, stacking rows upward. Used to
    HINT the CP banded solve an immediate incumbent (and as the honest fallback)."""
    exempt = set(bottom_edge_keepout_exempt or ())
    pos: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for b in present:
        if b == _BAND_CENTER:
            cur = max(cur, int(bottom_center_keepout_nm))
        row = _stable_hard_order([bid for bid in ids if bands[bid] == b], order_pairs)
        if pos:
            cur = max(
                cur,
                max(
                    py
                    + dims[prior][1]
                    + _required_pair_gap_nm(prior, current, gap, pair_clearances_nm)
                    for prior, (_px, py) in pos.items()
                    for current in row
                ),
            )
        row_positions, _extent = _ordered_row_positions(
            row,
            dims,
            x_start=0,
            y=cur,
            gap_nm=gap,
            pair_clearances_nm=pair_clearances_nm,
        )
        row_positions = {
            bid: (
                xy[0],
                xy[1] if bid in exempt else max(xy[1], int(bottom_edge_keepout_nm)),
            )
            for bid, xy in row_positions.items()
        }
        pos.update(row_positions)
        cur = (
            max(
                (pos[bid][1] + dims[bid][1] for bid in row),
                default=cur,
            )
            + gap
        )
    return pos

def _banded_dense_seed(
    ids: Sequence[str],
    dims: Dict[str, Tuple[int, int]],
    gap: int,
    bands: Dict[str, int],
    present: Sequence[int],
    w_cap: int,
    *,
    bottom_center_keepout_nm: int,
    bottom_edge_keepout_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
) -> Dict[str, Tuple[int, int]]:
    """Construct a legal multi-shelf seed inside a fixed banded outline."""

    seed_gap = max((int(gap), *pair_clearances_nm.values()))
    pos: Dict[str, Tuple[int, int]] = {}
    band_floor = int(bottom_edge_keepout_nm)
    for band in present:
        floor = max(
            band_floor,
            int(bottom_center_keepout_nm) if band == _BAND_CENTER else 0,
        )
        members = sorted(name for name in ids if bands[name] == band)
        local, _width, height = _shelf_pack(
            members,
            dims,
            seed_gap,
            int(w_cap),
        )
        pos.update(
            (name, (point[0], point[1] + floor)) for name, point in local.items()
        )
        band_floor = floor + height + seed_gap
    return pos

def _banded_axis_aligned_seed(
    ids: Sequence[str],
    dims: Mapping[str, Tuple[int, int]],
    gap: int,
    bands: Mapping[str, int],
    present: Sequence[int],
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]],
    *,
    bottom_center_keepout_nm: int,
    bottom_edge_keepout_nm: int,
    bottom_edge_keepout_exempt: Sequence[str],
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    w_cap: int | None = None,
) -> Dict[str, Tuple[int, int]]:
    """Construct a feasible hint for disjoint hard-axis groups.

    Aligned blocks share their declared x axis and stack in y inside a band.
    Unaligned blocks occupy a row to their right in rows mode, or shelves above
    the aligned stack when ``w_cap`` requests dense packing. A global maximum
    selected clearance makes the seed conservative for every pair-specific
    obligation.
    """

    seed_gap = max((int(gap), *pair_clearances_nm.values()))
    x_by_aligned: Dict[str, int] = {}
    cursor = 0
    for group in axis_alignment_groups:
        if any(name in x_by_aligned for name, _offset2 in group):
            # Overlapping algebraic groups remain valid CP input, but need a
            # general union-equation seed builder.  Let CP solve them without a
            # fabricated hint.
            return {}
        axis2 = 2 * cursor + max(offset2 for _name, offset2 in group)
        for name, offset2 in group:
            delta = axis2 - offset2
            if delta & 1:
                return {}
            x_by_aligned[name] = delta // 2
        cursor = (
            max(x_by_aligned[name] + dims[name][0] for name, _offset2 in group)
            + seed_gap
        )

    exempt = set(bottom_edge_keepout_exempt or ())
    pos: Dict[str, Tuple[int, int]] = {}
    band_floor = 0
    for band in present:
        band_ids = sorted(name for name in ids if bands[name] == band)
        floor = max(
            band_floor,
            int(bottom_center_keepout_nm) if band == _BAND_CENTER else 0,
        )
        aligned = [name for name in band_ids if name in x_by_aligned]
        y_cursor = floor
        for name in aligned:
            y = (
                y_cursor
                if name in exempt
                else max(y_cursor, int(bottom_edge_keepout_nm))
            )
            pos[name] = (x_by_aligned[name], y)
            y_cursor = y + dims[name][1] + seed_gap

        row = [name for name in band_ids if name not in x_by_aligned]
        if w_cap is None:
            aligned_right = max(
                (pos[name][0] + dims[name][0] for name in aligned),
                default=0,
            )
            row_pos, _extent = _ordered_row_positions(
                row,
                dims,
                x_start=(aligned_right + seed_gap if aligned else 0),
                y=floor,
                gap_nm=seed_gap,
                pair_clearances_nm=pair_clearances_nm,
            )
        else:
            shelf_floor = max(
                (pos[name][1] + dims[name][1] + seed_gap for name in aligned),
                default=floor,
            )
            local, _width, _height = _shelf_pack(
                row,
                dict(dims),
                seed_gap,
                int(w_cap),
            )
            row_pos = {
                name: (point[0], point[1] + shelf_floor)
                for name, point in local.items()
            }
        for name, (x, y) in row_pos.items():
            pos[name] = (
                x,
                y if name in exempt else max(y, int(bottom_edge_keepout_nm)),
            )
        band_floor = max(pos[name][1] + dims[name][1] for name in band_ids) + seed_gap
    return pos

def _hard_edge_compatible_banded_seed(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    gap_nm: int,
    bands: Mapping[str, int],
    present: Sequence[int],
    width_bound_nm: int,
    right_edge: Sequence[str],
    left_edge: Sequence[str],
    bottom_center_keepout_nm: int,
    bottom_edge_keepout_nm: int,
    bottom_edge_keepout_exempt: Sequence[str],
    pair_clearances_nm: Mapping[Tuple[str, str], int],
    axis_alignment_groups: Sequence[Sequence[Tuple[str, int]]],
    center_alignment_pairs: Sequence[Tuple[str, str, str]] = (),
    order_pairs: Sequence[Tuple[str, str]],
    rigid_compound_footprints: Tuple[_RigidCompoundFootprint, ...] = (),
    layered_collision_footprints: Tuple[_LayeredCollisionFootprint, ...] = (),
    overlap_compatible_pairs: Tuple[Tuple[str, str], ...] = (),
) -> Dict[str, Tuple[int, int]]:
    """Repair a banded hint into a constructive hard-edge incumbent.

    Exact edge blocks share an x boundary and therefore cannot retain the
    historical same-row hint.  Move them to a conservative vertical shelf above
    the other members of their band, then translate every later band upward.
    Existing x positions (including differential axes) remain untouched.

    Edge blocks participating in another x equality/order or a fixed bottom
    carrier are left to CP: changing them locally could fabricate an invalid
    cross-constraint witness.  The current repair is used only when the move is
    mechanically independent.
    """

    repaired = {str(name): (int(x), int(y)) for name, (x, y) in seed.items()}
    right = set(str(name) for name in right_edge)
    left = set(str(name) for name in left_edge)
    hard = right | left
    if not hard:
        return repaired
    if (
        all(repaired[name][0] + int(dims[name][0]) == int(width_bound_nm) for name in right)
        and all(repaired[name][0] == 0 for name in left)
        and _seed_geometry_legal(
            repaired,
            dims=dims,
            gap_nm=int(gap_nm),
            pair_clearances_nm=pair_clearances_nm,
            rigid_compound_footprints=rigid_compound_footprints,
            layered_collision_footprints=layered_collision_footprints,
            overlap_compatible_pairs=overlap_compatible_pairs,
        )
    ):
        # A feedback incumbent may already carry the exact hard boundaries.
        # Re-stacking it destroys useful height information and can inflate the
        # surrounding width heuristic.  It remains only a CP hint: every hard
        # band/axis/order/edge constraint below is still rebuilt and solved.
        return repaired
    axis_members = {
        str(name) for group in axis_alignment_groups for name, _offset2 in group
    }
    center_members = {
        str(name) for left, right, _axis in center_alignment_pairs for name in (left, right)
    }
    ordered_members = {
        str(name) for pair in order_pairs for name in pair
    }
    exempt = set(str(name) for name in bottom_edge_keepout_exempt)
    if hard & (axis_members | center_members | ordered_members | exempt):
        return repaired

    seed_gap = max((int(gap_nm), *pair_clearances_nm.values()))
    result: Dict[str, Tuple[int, int]] = {}
    next_band_floor = 0
    for band in present:
        members = sorted(name for name in repaired if int(bands[name]) == int(band))
        ordinary = [name for name in members if name not in hard]
        original_floor = min((repaired[name][1] for name in members), default=0)
        required_floor = max(
            next_band_floor,
            int(bottom_center_keepout_nm) if int(band) == _BAND_CENTER else 0,
        )
        translate_y = max(0, required_floor - original_floor)
        for name in ordinary:
            x, y = repaired[name]
            result[name] = (x, y + translate_y)

        hard_floor = max(
            required_floor,
            max(
                (
                    result[name][1] + int(dims[name][1]) + seed_gap
                    for name in ordinary
                ),
                default=required_floor,
            ),
        )
        for name in sorted(hard & set(members)):
            width, height = dims[name]
            x = 0 if name in left else int(width_bound_nm) - int(width)
            result[name] = (x, hard_floor)
            hard_floor += int(height) + seed_gap

        next_band_floor = max(
            (
                result[name][1] + int(dims[name][1]) + seed_gap
                for name in members
            ),
            default=next_band_floor,
        )
    return result

def _translate_fixed_outline_seed_for_columns(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    bands: Mapping[str, int],
    width_nm: int,
    manufacturing_grid_nm: int,
    exact_band: int,
    floorplan_column_classes: Sequence[Tuple[str, int, int, int]],
    floorplan_column_groups: Sequence[_FloorplanColumnGroup] = (),
) -> Dict[str, Tuple[int, int]]:
    """Solve cross-band column order as integer band translations.

    Each local band solve has already proven its internal geometry and column
    order.  A rigid X translation preserves those facts.  The remaining global
    column inequalities form difference constraints over one integer-grid
    translation per band, so longest-path relaxation finds their least feasible
    solution without another geometric search.
    """

    if not floorplan_column_classes and not floorplan_column_groups:
        return dict(seed)
    grid = int(manufacturing_grid_nm)
    if grid <= 0:
        raise ValueError("manufacturing grid must be a positive integer nm")
    present = tuple(sorted(set(bands.values())))

    def ceil_div(value: int, divisor: int) -> int:
        return -((-int(value)) // int(divisor))

    lower: Dict[int, int] = {}
    upper: Dict[int, int] = {}
    for band in present:
        members = tuple(name for name in seed if bands[name] == band)
        x_min = min(seed[name][0] for name in members)
        x_max = max(seed[name][0] + dims[name][0] for name in members)
        lower[band] = ceil_div(-x_min, grid)
        upper[band] = (int(width_nm) - x_max) // grid
        if band == exact_band:
            lower[band] = upper[band] = 0
        if lower[band] > upper[band]:
            raise RuntimeError("fixed-outline band has no legal grid translation")

    by_name = {
        name: (column, min_offset2, max_offset2)
        for name, column, min_offset2, max_offset2 in floorplan_column_classes
    }
    constraints: List[Tuple[int, int, int]] = []
    for left, (left_column, _left_min2, left_max2) in by_name.items():
        for right, (right_column, right_min2, _right_max2) in by_name.items():
            if left_column >= right_column or bands[left] == bands[right]:
                continue
            numerator = (
                2 * seed[left][0]
                + left_max2
                + 1
                - 2 * seed[right][0]
                - right_min2
            )
            constraints.append(
                (bands[left], bands[right], ceil_div(numerator, 2 * grid))
            )

    group_bands: Dict[str, int] = {}
    group_center2_sums: Dict[str, int] = {}
    for group, row, _column, _member_count, terms in floorplan_column_groups:
        term_bands = {bands[owner] for owner, _count, _offset in terms}
        if len(term_bands) != 1:
            raise ValueError(
                f"floorplan functional group {group!r} crosses exact bands"
            )
        if next(iter(term_bands)) != row:
            raise ValueError(
                f"floorplan functional group {group!r} row disagrees with exact band"
            )
        group_bands[group] = next(iter(term_bands))
        group_center2_sums[group] = sum(
            2 * count * seed[owner][0] + center_offset2_sum
            for owner, count, center_offset2_sum in terms
        )
    for left_group, left_row, left_column, left_count, _left_terms in floorplan_column_groups:
        for right_group, right_row, right_column, right_count, _right_terms in floorplan_column_groups:
            if left_row != right_row or left_column >= right_column:
                continue
            left_band = group_bands[left_group]
            right_band = group_bands[right_group]
            if left_band == right_band:
                continue
            numerator = (
                right_count * group_center2_sums[left_group]
                + 1
                - left_count * group_center2_sums[right_group]
            )
            constraints.append(
                (
                    left_band,
                    right_band,
                    ceil_div(numerator, 2 * left_count * right_count * grid),
                )
            )

    translation_units = dict(lower)
    for iteration in range(len(present) + 1):
        changed = False
        for left_band, right_band, minimum_delta in constraints:
            required = translation_units[left_band] + minimum_delta
            if translation_units[right_band] < required:
                translation_units[right_band] = required
                changed = True
                if required > upper[right_band]:
                    raise RuntimeError(
                        "fixed-outline floorplan columns have no legal band translation"
                    )
        if not changed:
            break
        if iteration == len(present):
            raise RuntimeError(
                "fixed-outline floorplan column translations contain a positive cycle"
            )

    translated = {
        name: (
            point[0] + grid * translation_units[bands[name]],
            point[1],
        )
        for name, point in seed.items()
    }
    blocks = tuple(
        (
            name,
            (
                translated[name][0],
                translated[name][1],
                translated[name][0] + dims[name][0],
                translated[name][1] + dims[name][1],
            ),
        )
        for name in sorted(translated)
    )
    if not _floorplan_column_order_satisfied(blocks, floorplan_column_classes):
        raise RuntimeError("fixed-outline band translations did not prove column order")
    if not _floorplan_column_group_order_satisfied(blocks, floorplan_column_groups):
        raise RuntimeError(
            "fixed-outline band translations did not prove group-centroid order"
        )
    return translated

# --------------------------------------------------------------------------- #
# REFERENCE-SKELETON FLOORPLAN (opt-in via ``left_column``): the tall common-
# centroid diff-pair block is pulled OUT of the top pmos band and anchored as a
# full-height LEFT COLUMN; the pmos mirrors form a FULL-WIDTH top strip ABOVE the
# lower region; the lower region to the RIGHT of the column is the banded stack
# (nmos bottom -> passives/cap center). Regular compact rectangle, min envelope.
# --------------------------------------------------------------------------- #
def _floorplan_seed(
    left_column: str,
    top: Sequence[str],
    lower: Sequence[str],
    bands: Dict[str, int],
    lower_bands: Sequence[int],
    dims: Dict[str, Tuple[int, int]],
    gap: int,
    sep: int = 0,
    order_pairs: Sequence[Tuple[str, str]] = (),
    bottom_center_keepout_nm: int = 0,
    bottom_edge_keepout_nm: int = 0,
    bottom_edge_keepout_exempt: Sequence[str] = (),
    bottom_edge_keepout_clear: Sequence[str] = (),
    pair_clearances_nm: Mapping[Tuple[str, str], int] = MappingProxyType({}),
) -> Dict[str, Tuple[int, int]]:
    """A guaranteed overlap-free constructive seed for the floorplan structure:
    the left column at the bottom-left, the lower region banded-stacked to its
    right (offset by the reserved channel ``sep`` >= ``gap``), and the top band as
    one row above the whole lower region. Used to HINT the CP an immediate
    incumbent (and as the honest over-budget fallback)."""
    D = left_column
    exempt = set(bottom_edge_keepout_exempt or ())
    clear = set(bottom_edge_keepout_clear or ())
    wD, hD = dims[D]
    normalized_pairs = sorted({(str(a), str(b)) for a, b in order_pairs or ()})
    for left, right in normalized_pairs:
        if right == D:
            raise ValueError(
                f"left-column hard order conflict: {left!r} cannot precede "
                f"fixed left-column {D!r}"
            )

    successors: Dict[str, set] = {}
    predecessors: Dict[str, set] = {}
    for left, right in normalized_pairs:
        successors.setdefault(left, set()).add(right)
        predecessors.setdefault(right, set()).add(left)
    after_column: set = set()
    stack = sorted(successors.get(D, ()), reverse=True)
    while stack:
        name = stack.pop()
        if name in after_column:
            continue
        after_column.add(name)
        stack.extend(sorted(successors.get(name, ()), reverse=True))

    pos: Dict[str, Tuple[int, int]] = {
        D: (0, 0 if D in exempt or D in clear else int(bottom_edge_keepout_nm))
    }
    x_base = wD + max(
        gap,
        int(sep),
        max(
            (
                _required_pair_gap_nm(D, block_id, gap, pair_clearances_nm)
                for block_id in lower
            ),
            default=gap,
        ),
    )
    cur_y = 0
    placed_lower: List[str] = []
    for bd in lower_bands:
        if bd == _BAND_CENTER:
            cur_y = max(cur_y, int(bottom_center_keepout_nm))
        row = _stable_hard_order(
            [b for b in lower if bands.get(b, _BAND_CENTER) == bd], order_pairs
        )
        if placed_lower:
            cur_y = max(
                cur_y,
                max(
                    pos[prior][1]
                    + dims[prior][1]
                    + _required_pair_gap_nm(prior, current, gap, pair_clearances_nm)
                    for prior in placed_lower
                    for current in row
                ),
            )
        row_positions, _extent = _ordered_row_positions(
            row,
            dims,
            x_start=x_base,
            y=cur_y,
            gap_nm=gap,
            pair_clearances_nm=pair_clearances_nm,
        )
        row_positions = {
            bid: (
                xy[0],
                (
                    xy[1]
                    if bid in exempt or bid in clear
                    else max(xy[1], int(bottom_edge_keepout_nm))
                ),
            )
            for bid, xy in row_positions.items()
        }
        pos.update(row_positions)
        placed_lower.extend(row)
        cur_y = (
            max(
                (pos[bid][1] + dims[bid][1] for bid in row),
                default=cur_y,
            )
            + gap
        )
    lower_top = (cur_y - gap) if lower else 0
    top_floor = max(lower_top, pos[D][1] + hD) + gap
    top_order = _stable_hard_order(top, normalized_pairs)
    if top_order:
        top_floor = max(
            top_floor,
            max(
                pos[prior][1]
                + dims[prior][1]
                + _required_pair_gap_nm(prior, current, gap, pair_clearances_nm)
                for prior in pos
                for current in top_order
            ),
        )
    top_positions: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for block_id in top_order:
        if block_id in after_column:
            cursor = max(
                cursor,
                wD + _required_pair_gap_nm(D, block_id, gap, pair_clearances_nm),
            )
        x = max(
            (
                top_positions[prior][0]
                + dims[prior][0]
                + _required_pair_gap_nm(prior, block_id, gap, pair_clearances_nm)
                for prior in top_positions
            ),
            default=cursor,
        )
        x = max(
            x,
            max(
                (
                    pos[prior][0]
                    + dims[prior][0]
                    + _required_pair_gap_nm(
                        prior,
                        block_id,
                        gap,
                        pair_clearances_nm,
                    )
                    for prior in predecessors.get(block_id, ())
                    if prior in pos
                ),
                default=x,
            ),
        )
        x = max(x, cursor)
        top_positions[block_id] = (x, top_floor)
        cursor = x + dims[block_id][0] + gap
    pos.update(top_positions)
    return pos

def _center_aligned_lateral_seed(
    seed: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    center_alignment_pairs: Sequence[Tuple[str, str, str]],
    lateral_blocks: Sequence[str],
    order_pairs: Sequence[Tuple[str, str]],
    gap_nm: int,
    pair_clearances_nm: Mapping[Tuple[str, str], int],
) -> Dict[str, Tuple[int, int]]:
    """Insert a short core block beside its tall lateral centre anchor.

    The reference-skeleton seed appends lateral macros after every core block.
    A vertical centre relation spanning that boundary can therefore be made
    constructive without coordinates: move the core endpoint into the old
    lateral slot and translate the lateral suffix by the core width plus its
    required clearance.  The relative core-before-lateral topology is retained.
    Unsupported shapes remain ordinary CP hints; no fallback is certified from
    this helper unless the independent solver constraints are satisfied.
    """

    result = {str(name): (int(point[0]), int(point[1])) for name, point in seed.items()}
    lateral = {str(name) for name in lateral_blocks}
    orders = {(str(left), str(right)) for left, right in order_pairs}
    for left, right, axis in center_alignment_pairs:
        if axis != "vertical" or ((left in lateral) == (right in lateral)):
            continue
        core = right if left in lateral else left
        anchor = left if left in lateral else right
        if (anchor, core) in orders:
            continue
        core_height = int(dims[core][1])
        anchor_height = int(dims[anchor][1])
        center2 = 2 * result[anchor][1] + anchor_height
        if (center2 - core_height) & 1:
            continue
        core_y = (center2 - core_height) // 2
        if core_y < 0:
            continue
        insertion_x = result[anchor][0]
        clearance = _required_pair_gap_nm(
            core, anchor, gap_nm, pair_clearances_nm
        )
        translation = int(dims[core][0]) + clearance
        for name in sorted(lateral):
            x, y = result[name]
            if x >= insertion_x:
                result[name] = (x + translation, y)
        result[core] = (insertion_x, core_y)
    return result

def common_centroid_devices(intent: Any, geom: Dict[str, Any]) -> set:
    """Set of device / tile names that belong to a declared ``@common_centroid``
    group. Topology-derived -- keys on the ``@common_centroid`` ROLE, never a
    device name or a coordinate. Each ``intent.common_centroids``
    entry names a ``group``; the group materialises as ONE composite tile stamp
    named after the group (``geom[group].composite_members`` lists its consumed
    devices), so the returned set is the group names plus those members. Empty
    when the cell declares no ``@common_centroid``."""
    groups = {
        str(getattr(c, "group", ""))
        for c in (getattr(intent, "common_centroids", ()) or ())
    }
    groups.discard("")
    devs: set = set(groups)
    for g in groups:
        cg = (geom or {}).get(g)
        devs |= set(getattr(cg, "composite_members", ()) or ())
    return devs

def identify_left_column(
    block_ids: Sequence[str],
    block_member_devices: Dict[str, set],
    cc_devices: set,
    *,
    declared_left_groups: Sequence[str] = (),
    require_unique: bool = False,
) -> Any:
    """The block whose members include a ``@common_centroid`` device -> the diff-
    pair block that anchors the full-height LEFT COLUMN. Returns the block id
    only when the role identifies exactly one block.  When multiple common-
    centroid blocks exist, ``declared_left_groups`` narrows ownership using a
    coordinate-free structural-candidate choice; zero or multiple resolutions
    select the plain band stack. Generic: identifies the column by declared roles,
    never a hardcoded device / block name.

    With no candidate selection the legacy single-common-centroid inference is
    retained.  A materialized ``@place_policy(left_column=GROUP)`` supplies the
    explicit group only when the bounded structural domain selected it; this is
    separate from ``@anchor``, whose meaning remains exact-GDS edge access."""
    declared = {str(group) for group in declared_left_groups}
    candidates = tuple(
        bid
        for bid in sorted(block_ids)
        if block_member_devices.get(bid, set()) & cc_devices
        and (
            not declared
            or block_member_devices.get(bid, set()) & declared
        )
    )
    if declared and len(candidates) != 1:
        raise ValueError(
            "declared left-column group must resolve exactly one common-centroid "
            f"block; groups={tuple(sorted(declared))!r}, blocks={candidates!r}"
        )
    if require_unique and len(candidates) != 1:
        raise ValueError(
            "declared channel requires one unique left-column common-centroid "
            f"owner; blocks={candidates!r}"
        )
    return candidates[0] if len(candidates) == 1 else None

def assign_bands(
    block_ids: Sequence[str],
    block_member_devices: Dict[str, set],
    device_kind: Dict[str, str],
    *,
    device_nets: Mapping[str, Mapping[str, str]] | None = None,
    place_groups: Sequence[Any] = (),
    rail_affinities: Sequence[Any] = (),
    band_intents: Sequence[Any] = (),
    _declared_member_blocks: Optional[set[str]] = None,
) -> Dict[str, int]:
    """``{block_id -> band ordinal}`` derived from member-device polarity / passive-
    ness (topology-derived, never a coordinate): any cap/res member -> CENTER (1);
    else pmos -> TOP (2); else nmos -> BOTTOM (0); unknown -> CENTER. A declared
    ``@band`` then replays source/body connectivity and explicitly binds the
    rail-connected PMOS subset to ``top``, the rail-connected NMOS subset to
    ``bottom``, and its declared passive/place-group subset to ``center``. The
    class-only rule remains only the undeclared polarity fallback. A
    A direct or named ``@rail_affinity`` may refine the generic
    ``center=passives`` class default; an explicitly named center group remains
    authoritative and cannot be overwritten."""
    out: Dict[str, int] = {}
    for bid in block_ids:
        kinds = {
            device_kind.get(d)
            for d in block_member_devices.get(bid, ())
            if device_kind.get(d)
        }
        if kinds & {"cap", "res"}:
            out[bid] = _BAND_CENTER
        elif "pmos" in kinds:
            out[bid] = _BAND_TOP
        elif "nmos" in kinds:
            out[bid] = _BAND_BOTTOM
        else:
            out[bid] = _BAND_CENTER
    group_devices = {
        str(getattr(group, "group", "")): set(getattr(group, "devices", ()) or ())
        for group in place_groups or ()
    }
    declarations = tuple(band_intents or ())
    declared_member_blocks: set[str] = set()
    generic_center_blocks: set[str] = set()
    if len(declarations) > 1:
        raise ValueError("band materializer requires at most one declaration")
    if declarations:
        declaration = declarations[0]
        center = str(getattr(declaration, "center", "") or "")
        if center == "passives":
            center_devices = {
                device
                for devices in block_member_devices.values()
                for device in devices
                if device_kind.get(device) in {"cap", "res"}
            }
        else:
            center_devices = set(group_devices.get(center, ()))
            if not center_devices:
                raise ValueError(f"band center group {center!r} is not materializable")
            if any(device_kind.get(device) not in {"cap", "res"} for device in center_devices):
                raise ValueError(
                    f"band center group {center!r} must contain only passive devices"
                )
        for block_id in block_ids:
            if block_member_devices.get(block_id, set()) & center_devices:
                out[block_id] = _BAND_CENTER
                declared_member_blocks.add(block_id)
                if center == "passives":
                    generic_center_blocks.add(block_id)
        nets_by_device = dict(device_nets or {})
        rail_members = {
            _BAND_TOP: {
                device
                for device, kind in device_kind.items()
                if kind == "pmos"
                and str(declaration.top)
                in {
                    str(nets_by_device.get(device, {}).get("source", "")),
                    str(nets_by_device.get(device, {}).get("body", "")),
                }
            },
            _BAND_BOTTOM: {
                device
                for device, kind in device_kind.items()
                if kind == "nmos"
                and str(declaration.bottom)
                in {
                    str(nets_by_device.get(device, {}).get("source", "")),
                    str(nets_by_device.get(device, {}).get("body", "")),
                }
            },
        }
        if not all(rail_members.values()):
            raise ValueError("band materializer has an empty rail-bound MOS family")
        for band, members in rail_members.items():
            for block_id in block_ids:
                if block_member_devices.get(block_id, set()) & members:
                    out[block_id] = band
                    declared_member_blocks.add(block_id)
    if _declared_member_blocks is not None:
        _declared_member_blocks.update(declared_member_blocks)
    for affinity in rail_affinities or ():
        side = str(getattr(affinity, "side", "") or "").strip().lower()
        if side not in {"top", "bottom"}:
            continue
        target = str(getattr(affinity, "group", "") or "")
        devices = group_devices.get(target) or {target}
        band = _BAND_TOP if side == "top" else _BAND_BOTTOM
        for block_id in block_ids:
            if (
                (
                    block_id not in declared_member_blocks
                    or block_id in generic_center_blocks
                )
                and block_member_devices.get(block_id, set()) & devices
            ):
                out[block_id] = band
    # A complete exact ``row=N`` inventory is a stronger, coordinate-free
    # floorplan authority than the generic electrical-polarity fallback above.
    # Partial inventories deliberately preserve the historical bands: silently
    # mixing two authorities would move only some blocks and fabricate topology.
    from clay.layout.intent import place_group_floorplan_row

    row_by_device: Dict[str, int] = {}
    for group in place_groups or ():
        row = place_group_floorplan_row(group)
        if row is None:
            continue
        for device in tuple(getattr(group, "devices", ()) or ()):
            name = str(device)
            prior = row_by_device.get(name)
            if prior is not None and prior != row:
                raise ValueError(
                    f"device {name!r} belongs to conflicting exact floorplan rows"
                )
            row_by_device[name] = row
    final_members = {
        str(device)
        for block_id in block_ids
        for device in block_member_devices.get(block_id, set())
        if str(device) in device_kind
    }
    if final_members and set(row_by_device) == final_members:
        for block_id in block_ids:
            rows = {
                row_by_device[str(device)]
                for device in block_member_devices.get(block_id, set())
                if str(device) in device_kind
            }
            if len(rows) != 1:
                raise ValueError(
                    f"aggregate block {block_id!r} crosses exact floorplan rows"
                )
            out[block_id] = next(iter(rows))
        if _declared_member_blocks is not None:
            _declared_member_blocks.update(block_ids)
    return out

def floorplan_column_classes(
    blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Mapping[str, set],
    place_groups: Sequence[Any],
    device_bboxes_nm: Mapping[str, _Bbox],
) -> Tuple[Tuple[str, int, int, int], ...]:
    """Project declared device columns onto the exact final rigid masters.

    A column declaration orders device centroids globally in X.  The CP solver
    moves rigid masters, so every declared source device must resolve to exactly
    one master and every affected master must own members from exactly one
    column.  Synthetic tile labels are ignored; they are not source devices.
    Missing, duplicate, or cross-column ownership is rejected instead of being
    silently dropped.
    """

    block_by_name = dict(blocks)
    if len(block_by_name) != len(tuple(blocks)):
        raise ValueError("floorplan column projection has duplicate final masters")
    column_by_device: Dict[str, int] = {}
    declared_devices: set[str] = set()
    for group in place_groups:
        members = tuple(getattr(group, "devices", ()) or ())
        if any(type(member) is not str or not member for member in members):
            raise TypeError("place-group column members must be exact strings")
        declared_devices.update(members)
        column = getattr(group, "col", None)
        if column is None:
            continue
        if type(column) is not int or column < 0:
            raise TypeError("place-group column must be one non-negative exact int")
        for raw_member in members:
            member = raw_member
            prior = column_by_device.get(member)
            if prior is not None and prior != column:
                raise ValueError(
                    f"device {member!r} belongs to conflicting floorplan columns"
                )
            column_by_device[member] = column
    if not column_by_device:
        return ()

    owners_by_device: Dict[str, List[str]] = {
        member: [] for member in column_by_device
    }
    projected: List[Tuple[str, int, int, int]] = []
    for block_id in sorted(block_by_name):
        if block_id not in block_member_devices:
            raise ValueError(f"floorplan column projection lacks master {block_id!r}")
        source_members = set(block_member_devices[block_id]) & declared_devices
        for member in source_members & set(column_by_device):
            owners_by_device[member].append(block_id)
        if not source_members:
            continue
        columns = {
            column_by_device[member]
            for member in source_members
            if member in column_by_device
        }
        if len(columns) != 1 or any(
            member not in column_by_device for member in source_members
        ):
            raise ValueError(
                f"final master {block_id!r} crosses or partially consumes "
                "declared floorplan columns"
            )
        master_bbox = block_by_name[block_id]
        center_offsets2 = []
        for member in sorted(source_members):
            bbox = device_bboxes_nm.get(member)
            if (
                type(bbox) is not tuple
                or len(bbox) != 4
                or any(type(value) is not int for value in bbox)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise ValueError(
                    f"floorplan column device {member!r} lacks exact bbox authority"
                )
            center_offset2 = bbox[0] + bbox[2] - 2 * master_bbox[0]
            if not 0 <= center_offset2 <= 2 * (master_bbox[2] - master_bbox[0]):
                raise ValueError(
                    f"floorplan column device {member!r} escapes final master "
                    f"{block_id!r}"
                )
            center_offsets2.append(center_offset2)
        projected.append(
            (
                block_id,
                next(iter(columns)),
                min(center_offsets2),
                max(center_offsets2),
            )
        )

    bad_owners = {
        member: tuple(owners)
        for member, owners in owners_by_device.items()
        if len(owners) != 1
    }
    if bad_owners:
        raise ValueError(
            "declared floorplan column devices lack unique final masters: "
            f"{bad_owners!r}"
        )
    return tuple(projected)

def floorplan_column_groups(
    blocks: Sequence[Tuple[str, _Bbox]],
    block_member_devices: Mapping[str, set],
    place_groups: Sequence[Any],
    device_bboxes_nm: Mapping[str, _Bbox],
) -> Tuple[_FloorplanColumnGroup, ...]:
    """Project declared functional-group centroids onto exact rigid masters.

    Reference macro columns are ranks of anonymous region centroids.  One
    functional group may be split across several mandatory well islands, so its
    centroid is carried as an exact linear expression over those master origins.
    Consumers aggregate every group sharing a ``(row, column)`` cell before
    comparing cells; the grammar proves that such groups share one anonymous
    region.  This avoids inventing subgroup or member ordering inside a region.
    """

    from clay.layout.intent import place_group_floorplan_row

    block_by_name = dict(blocks)
    if len(block_by_name) != len(tuple(blocks)):
        raise ValueError("floorplan group projection has duplicate final masters")
    owner_by_device: Dict[str, str] = {}
    for block_id in sorted(block_by_name):
        if block_id not in block_member_devices:
            raise ValueError(f"floorplan group projection lacks master {block_id!r}")
        for raw_member in block_member_devices[block_id]:
            member = str(raw_member)
            prior = owner_by_device.setdefault(member, block_id)
            if prior != block_id:
                raise ValueError(
                    f"floorplan group device {member!r} has multiple final masters"
                )

    declared_column_groups = tuple(
        group for group in place_groups if getattr(group, "col", None) is not None
    )
    declared_rows = tuple(
        place_group_floorplan_row(group) for group in declared_column_groups
    )
    if any(row is None for row in declared_rows) and any(
        row is not None for row in declared_rows
    ):
        raise ValueError(
            "place-group columns cannot mix global and row-scoped centroid orders"
        )
    global_column_scope = bool(declared_rows) and all(
        row is None for row in declared_rows
    )

    rows: List[_FloorplanColumnGroup] = []
    cells_by_master: Dict[str, set[Tuple[int, int]]] = {}
    declared_members: set[str] = set()
    for group in place_groups:
        column = getattr(group, "col", None)
        if column is None:
            continue
        if type(column) is not int or column < 0:
            raise TypeError("place-group column must be one non-negative exact int")
        row = place_group_floorplan_row(group)
        if row is None:
            if not global_column_scope:
                raise ValueError(
                    "place-group column requires one coherent centroid order scope"
                )
            # ``col=N`` without ``row=N`` is the IR's global horizontal-order
            # form.  Use one internal comparison scope without feeding a row
            # ordinal to band assignment; this is not a physical Y coordinate.
            row = _GLOBAL_FLOORPLAN_COLUMN_SCOPE
        group_name = getattr(group, "group", None)
        if type(group_name) is not str or not group_name or group_name != group_name.strip():
            raise TypeError("place-group name must be one canonical string")
        members = getattr(group, "devices", ())
        if type(members) is not tuple or not members or any(
            type(member) is not str or not member for member in members
        ):
            raise TypeError(
                "place-group column members must be exact strings in a non-empty tuple"
            )
        duplicate_members = declared_members & set(members)
        if duplicate_members:
            raise ValueError(
                "floorplan column members have multiple functional-group owners: "
                f"{tuple(sorted(duplicate_members))!r}"
            )
        declared_members.update(members)

        terms: Dict[str, List[int]] = {}
        for member in members:
            owner = owner_by_device.get(member)
            if owner is None:
                raise ValueError(
                    f"floorplan column device {member!r} lacks one final master"
                )
            bbox = device_bboxes_nm.get(member)
            if (
                type(bbox) is not tuple
                or len(bbox) != 4
                or any(type(value) is not int for value in bbox)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise ValueError(
                    f"floorplan column device {member!r} lacks exact bbox authority"
                )
            master_bbox = block_by_name[owner]
            center_offset2 = bbox[0] + bbox[2] - 2 * master_bbox[0]
            if not 0 <= center_offset2 <= 2 * (master_bbox[2] - master_bbox[0]):
                raise ValueError(
                    f"floorplan column device {member!r} escapes final master "
                    f"{owner!r}"
                )
            term = terms.setdefault(owner, [0, 0])
            term[0] += 1
            term[1] += center_offset2
            cells_by_master.setdefault(owner, set()).add((row, column))
        rows.append(
            (
                group_name,
                row,
                column,
                len(members),
                tuple(
                    (owner, count_and_sum[0], count_and_sum[1])
                    for owner, count_and_sum in sorted(terms.items())
                ),
            )
        )
    crossing = {
        owner: tuple(sorted(cells))
        for owner, cells in cells_by_master.items()
        if len(cells) != 1
    }
    if crossing:
        raise ValueError(
            "final masters cross declared functional-group floorplan cells: "
            f"{crossing!r}"
        )
    return tuple(sorted(rows))

def _placement_primary_extent_nm(
    positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    placement_objective: str,
    flexible_envelope_constraints: Tuple[FlexibleEnvelopeConstraint, ...] = (),
    bottom_edge_keepout_nm: int = 0,
) -> int:
    if placement_objective not in {"area", "height"}:
        raise ValueError("placement objective must be area or height")
    if set(positions) != set(dims) or not dims:
        raise ValueError("primary extent requires one complete non-empty position map")
    _member_boxes, group_boxes = _flexible_seed_physical_views(
        positions,
        dims=dims,
        constraints=flexible_envelope_constraints,
    )
    outline = tuple(
        (
            positions[name][0],
            positions[name][1],
            positions[name][0] + dims[name][0],
            positions[name][1] + dims[name][1],
        )
        for name in dims
    ) + tuple(group_boxes.values())
    x0 = min(bbox[0] for bbox in outline)
    y0 = (
        0
        if int(bottom_edge_keepout_nm) > 0
        else min(bbox[1] for bbox in outline)
    )
    x1 = max(bbox[2] for bbox in outline)
    y1 = max(bbox[3] for bbox in outline)
    height = y1 - y0
    return (x1 - x0) * height if placement_objective == "area" else height

def _prefer_certified_seed_positions(
    solved_positions: Mapping[str, Tuple[int, int]],
    certified_seed_positions: Mapping[str, Tuple[int, int]],
    *,
    dims: Mapping[str, Tuple[int, int]],
    placement_objective: str,
    flexible_envelope_constraints: Tuple[FlexibleEnvelopeConstraint, ...] = (),
    bottom_edge_keepout_nm: int = 0,
    cohesion_groups: Sequence[Sequence[str]] = (),
) -> Tuple[Dict[str, Tuple[int, int]], bool]:
    """Never discard a strictly better already-certified primary incumbent.

    The CP model also receives the seed's primary value as a hard upper bound.
    This final comparison is a defensive replay: selecting the seed indicates a
    solver-evidence inconsistency and is therefore reported to the caller as a
    fallback rather than silently signed.
    """

    solved = dict(solved_positions)
    certified = dict(certified_seed_positions)
    def score(positions: Mapping[str, Tuple[int, int]]) -> Tuple[int, int]:
        cohesion = sum(
            (
                max(positions[name][0] + dims[name][0] for name in group)
                - min(positions[name][0] for name in group)
                + max(positions[name][1] + dims[name][1] for name in group)
                - min(positions[name][1] for name in group)
            )
            for group in cohesion_groups
        )
        primary = _placement_primary_extent_nm(
            positions,
            dims=dims,
            placement_objective=placement_objective,
            flexible_envelope_constraints=flexible_envelope_constraints,
            bottom_edge_keepout_nm=bottom_edge_keepout_nm,
        )
        return primary, cohesion

    if score(certified) < score(solved):
        return certified, True
    return solved, False

def _exact_matching_bbox(name: str, value: Any) -> _Bbox:
    if (
        type(value) is not tuple
        or len(value) != 4
        or any(type(coordinate) is not int for coordinate in value)
        or value[2] <= value[0]
        or value[3] <= value[1]
    ):
        raise TypeError(f"matching-axis device {name!r} lacks an exact integer bbox")
    return value

def _strict_band_order_required(
    *,
    source_declares_band: bool,
    block_ids: Sequence[str],
    bands: Mapping[str, int],
    declared_member_blocks: set[str],
) -> bool:
    """Whether the final master must enforce global bottom-up row order.

    A source ``@band`` is one authority.  A complete exact ``row=N`` place-group
    inventory is another and remains hard even when a reference-derived
    candidate correctly suppresses the incompatible generic electrical band.
    A partial inventory or a single row cannot silently strengthen placement.
    """
    if type(source_declares_band) is not bool:
        raise TypeError("source_declares_band must be an exact bool")
    ids = set(block_ids)
    if source_declares_band:
        return True
    return bool(
        ids
        and set(declared_member_blocks) == ids
        and all(block_id in bands for block_id in ids)
        and len({int(bands[block_id]) for block_id in ids}) > 1
    )

def _placement_declaration_refs(intent: Any) -> Tuple[str, ...]:
    """Canonical refs for declarations consumed by the final placement master."""

    rows = []
    for field_name in (
        "arrays",
        "common_centroids",
        "place_groups",
        "bands",
        "rail_affinities",
        "anchors",
        "channels",
        "order_constraints",
        "edge_alignments",
        "diff_pairs",
        "discrete_mosaics",
        "mosaic_center_alignments",
    ):
        for index, value in enumerate(tuple(getattr(intent, field_name, ()) or ())):
            if is_dataclass(value):
                payload = asdict(value)
            elif isinstance(value, (tuple, list)):
                payload = list(value)
            else:
                payload = str(value)
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
            rows.append(f"placement:{field_name}:{index:03d}:{digest}")
    return tuple(sorted(rows))

def _vertical_channel_nm(intent: Any, channel_strategy: str) -> int:
    """Return the executable vertical-corridor width selected by policy."""

    if type(channel_strategy) is not str or channel_strategy not in {
        "declared",
        "block_gap_only",
    }:
        raise ValueError(
            "channel_strategy must be an exact declared or block_gap_only string"
        )
    if channel_strategy == "block_gap_only":
        return 0
    return max(
        (
            int(round(float(channel.width_um) * _NM_PER_UM))
            for channel in tuple(getattr(intent, "channels", ()) or ())
            if str(getattr(channel, "orient", "")).strip().lower() == "vertical"
        ),
        default=0,
    )
