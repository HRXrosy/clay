"""Typed row hierarchy consumed by the P2 floorplan MILP.

The compiler is intentionally geometry-only and circuit-neutral.  It converts
the reconciled row plan into indivisible matching-pair or singleton atoms before
the numerical solver allocates coordinates.  It does not choose coordinates,
read a reference layout, or weaken a declaration.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Mapping, Sequence


_MAX_ROWS = 256
_MAX_ATOMS = 1024
_MAX_MEMBERS = 2048
_MAX_COORD_NM = 10**12
_MAX_PACKED_SPAN_NM = _MAX_MEMBERS * (3 * _MAX_COORD_NM)


class HierarchicalMacroPlacementError(ValueError):
    """A row hierarchy is incomplete, ambiguous, or not exactly typed."""


class NoFeasibleHierarchicalPlacement(HierarchicalMacroPlacementError):
    """The declared hierarchy has no solver-certified placement."""

    code = "no_feasible_placement"

    def __init__(self, solver_status: object) -> None:
        if type(solver_status) is not str or not solver_status or len(solver_status) > 4096:
            raise HierarchicalMacroPlacementError("solver status is invalid")
        self.solver_status = solver_status
        super().__init__(f"{self.code}:{solver_status}")


def _text(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise HierarchicalMacroPlacementError(f"{field} must be non-empty text")
    return value


def _bbox(value: object, field: str) -> tuple[int, int, int, int]:
    if type(value) is not tuple or len(value) != 4:
        raise HierarchicalMacroPlacementError(f"{field} must be an exact bbox tuple")
    if any(type(item) is not int for item in value):
        raise HierarchicalMacroPlacementError(f"{field} coordinates must be integers")
    x0, y0, x1, y1 = value
    if (
        any(abs(item) > _MAX_COORD_NM for item in value)
        or x1 <= x0
        or y1 <= y0
    ):
        raise HierarchicalMacroPlacementError(f"{field} is invalid or out of bounds")
    return value


@dataclass(frozen=True, slots=True, order=True)
class HierarchicalMacroAtom:
    atom_id: str
    row_index: int
    ordinal: int
    kind: str
    members: tuple[str, ...]
    local_bbox_nm: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _text(self.atom_id, "atom_id")
        if type(self.row_index) is not int or not 0 <= self.row_index < _MAX_ROWS:
            raise HierarchicalMacroPlacementError("atom row_index is out of bounds")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < _MAX_ATOMS:
            raise HierarchicalMacroPlacementError("atom ordinal is out of bounds")
        if self.kind not in ("matching_pair", "singleton"):
            raise HierarchicalMacroPlacementError("atom kind is not closed")
        if type(self.members) is not tuple or any(
            type(member) is not str or not member for member in self.members
        ):
            raise HierarchicalMacroPlacementError("atom members must be exact text")
        expected = 2 if self.kind == "matching_pair" else 1
        if len(self.members) != expected or len(set(self.members)) != expected:
            raise HierarchicalMacroPlacementError("atom arity does not match its kind")
        _bbox(self.local_bbox_nm, "atom local_bbox_nm")


@dataclass(frozen=True, slots=True)
class HierarchicalMacroRow:
    row_index: int
    atoms: tuple[HierarchicalMacroAtom, ...]

    def __post_init__(self) -> None:
        if type(self.row_index) is not int or not 0 <= self.row_index < _MAX_ROWS:
            raise HierarchicalMacroPlacementError("row_index is out of bounds")
        if type(self.atoms) is not tuple or not self.atoms:
            raise HierarchicalMacroPlacementError("row requires exact atoms")
        if any(type(atom) is not HierarchicalMacroAtom for atom in self.atoms):
            raise HierarchicalMacroPlacementError("row contains an untyped atom")
        if any(atom.row_index != self.row_index for atom in self.atoms):
            raise HierarchicalMacroPlacementError("atom belongs to a foreign row")
        if tuple(atom.ordinal for atom in self.atoms) != tuple(range(len(self.atoms))):
            raise HierarchicalMacroPlacementError("row atom ordinals are not canonical")


@dataclass(frozen=True, slots=True)
class HierarchicalMacroProgram:
    rows: tuple[HierarchicalMacroRow, ...]
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.rows) is not tuple or not self.rows or len(self.rows) > _MAX_ROWS:
            raise HierarchicalMacroPlacementError("program requires bounded exact rows")
        if any(type(row) is not HierarchicalMacroRow for row in self.rows):
            raise HierarchicalMacroPlacementError("program contains an untyped row")
        if tuple(row.row_index for row in self.rows) != tuple(range(len(self.rows))):
            raise HierarchicalMacroPlacementError("program rows are not canonical")
        if sum(len(row.atoms) for row in self.rows) > _MAX_ATOMS:
            raise HierarchicalMacroPlacementError("program has too many atoms")
        derived = tuple(
            member for row in self.rows for atom in row.atoms for member in atom.members
        )
        if (
            type(self.members) is not tuple
            or derived != self.members
            or len(derived) > _MAX_MEMBERS
            or len(set(derived)) != len(derived)
        ):
            raise HierarchicalMacroPlacementError(
                "program members do not exactly conserve atom ownership"
            )


def _atom_width_nm(atom: HierarchicalMacroAtom, gap_nm: int) -> int:
    x0, _y0, x1, _y1 = atom.local_bbox_nm
    return x1 - x0 + (len(atom.members) - 1) * gap_nm


def _packed_width_nm(atoms: tuple[HierarchicalMacroAtom, ...], gap_nm: int) -> int:
    return sum(_atom_width_nm(atom, gap_nm) for atom in atoms) + gap_nm * (
        len(atoms) - 1
    )


@dataclass(frozen=True, slots=True)
class HierarchicalMacroSubrow:
    subrow_index: int
    band_index: int
    atoms: tuple[HierarchicalMacroAtom, ...]
    packed_width_nm: int
    height_nm: int

    def __post_init__(self) -> None:
        if type(self.subrow_index) is not int or not 0 <= self.subrow_index < _MAX_ATOMS:
            raise HierarchicalMacroPlacementError("subrow_index is out of bounds")
        if type(self.band_index) is not int or not 0 <= self.band_index < _MAX_ROWS:
            raise HierarchicalMacroPlacementError("subrow band_index is out of bounds")
        if (
            type(self.atoms) is not tuple
            or not self.atoms
            or len(self.atoms) > _MAX_ATOMS
        ):
            raise HierarchicalMacroPlacementError("subrow requires exact atoms")
        if any(type(atom) is not HierarchicalMacroAtom for atom in self.atoms):
            raise HierarchicalMacroPlacementError("subrow contains an untyped atom")
        if any(atom.row_index != self.band_index for atom in self.atoms):
            raise HierarchicalMacroPlacementError("subrow contains a foreign-band atom")
        atom_ids = tuple(atom.atom_id for atom in self.atoms)
        ordinals = tuple(atom.ordinal for atom in self.atoms)
        if len(set(atom_ids)) != len(atom_ids) or tuple(sorted(ordinals)) != ordinals:
            raise HierarchicalMacroPlacementError("subrow atoms are duplicate or unordered")
        if (
            type(self.packed_width_nm) is not int
            or not 0 < self.packed_width_nm <= _MAX_PACKED_SPAN_NM
            or type(self.height_nm) is not int
            or not 0 < self.height_nm <= 2 * _MAX_COORD_NM
        ):
            raise HierarchicalMacroPlacementError("subrow dimensions are invalid")


@dataclass(frozen=True, slots=True)
class HierarchicalMacroSubrowPlan:
    program: HierarchicalMacroProgram
    gap_nm: int
    target_width_nm: int
    subrows: tuple[HierarchicalMacroSubrow, ...]

    def __post_init__(self) -> None:
        if type(self.program) is not HierarchicalMacroProgram:
            raise HierarchicalMacroPlacementError("subrow plan requires an exact program")
        if type(self.gap_nm) is not int or not 0 <= self.gap_nm <= _MAX_COORD_NM:
            raise HierarchicalMacroPlacementError("subrow gap is invalid")
        if (
            type(self.target_width_nm) is not int
            or not 0 < self.target_width_nm <= _MAX_PACKED_SPAN_NM
        ):
            raise HierarchicalMacroPlacementError("subrow target width is invalid")
        if (
            type(self.subrows) is not tuple
            or not self.subrows
            or len(self.subrows) > _MAX_ATOMS
        ):
            raise HierarchicalMacroPlacementError("subrow plan requires exact subrows")
        if any(type(row) is not HierarchicalMacroSubrow for row in self.subrows):
            raise HierarchicalMacroPlacementError("subrow plan contains an untyped row")
        if tuple(row.subrow_index for row in self.subrows) != tuple(
            range(len(self.subrows))
        ):
            raise HierarchicalMacroPlacementError("subrow indices are not canonical")
        source_atoms = tuple(atom for row in self.program.rows for atom in row.atoms)
        planned_atoms = tuple(atom for row in self.subrows for atom in row.atoms)
        if planned_atoms != source_atoms:
            raise HierarchicalMacroPlacementError(
                "subrow plan does not exactly conserve atom order and ownership"
            )
        if self.target_width_nm != max(row.packed_width_nm for row in self.subrows):
            raise HierarchicalMacroPlacementError("subrow target width is stale")
        for row in self.subrows:
            if row.packed_width_nm != _packed_width_nm(row.atoms, self.gap_nm):
                raise HierarchicalMacroPlacementError("subrow packed width is stale")
            expected_height = max(
                atom.local_bbox_nm[3] - atom.local_bbox_nm[1] for atom in row.atoms
            )
            if row.height_nm != expected_height:
                raise HierarchicalMacroPlacementError("subrow height is stale")


@dataclass(frozen=True, slots=True)
class GeometryBoundedGridPlan:
    atom_sizes_nm: tuple[tuple[int, int], ...]
    gap_nm: int
    fixed_bbox_nm: tuple[int, int, int, int]
    anchor_x_nm: int
    top_y_nm: int
    columns: int
    row_widths_nm: tuple[int, ...]
    row_heights_nm: tuple[int, ...]
    grid_width_nm: int
    grid_height_nm: int
    combined_bbox_nm: tuple[int, int, int, int]
    combined_area_nm2: int

    def __post_init__(self) -> None:
        if (
            type(self.atom_sizes_nm) is not tuple
            or not self.atom_sizes_nm
            or len(self.atom_sizes_nm) > _MAX_ATOMS
            or any(
                type(size) is not tuple
                or len(size) != 2
                or any(type(item) is not int for item in size)
                or not all(0 < item <= 2 * _MAX_COORD_NM for item in size)
                for size in self.atom_sizes_nm
            )
        ):
            raise HierarchicalMacroPlacementError("grid atom sizes are invalid")
        if type(self.gap_nm) is not int or not 0 <= self.gap_nm <= _MAX_COORD_NM:
            raise HierarchicalMacroPlacementError("grid gap is invalid")
        fixed = _bbox(self.fixed_bbox_nm, "grid fixed_bbox_nm")
        if (
            type(self.anchor_x_nm) is not int
            or abs(self.anchor_x_nm) > _MAX_COORD_NM
            or type(self.top_y_nm) is not int
            or abs(self.top_y_nm) > _MAX_COORD_NM
            or type(self.columns) is not int
            or not 1 <= self.columns <= len(self.atom_sizes_nm)
        ):
            raise HierarchicalMacroPlacementError("grid placement controls are invalid")
        chunks = tuple(
            self.atom_sizes_nm[index:index + self.columns]
            for index in range(0, len(self.atom_sizes_nm), self.columns)
        )
        expected_widths = tuple(
            sum(size[0] for size in row) + self.gap_nm * (len(row) - 1)
            for row in chunks
        )
        expected_heights = tuple(max(size[1] for size in row) for row in chunks)
        if (
            type(self.row_widths_nm) is not tuple
            or type(self.row_heights_nm) is not tuple
            or any(type(item) is not int for item in self.row_widths_nm)
            or any(type(item) is not int for item in self.row_heights_nm)
            or self.row_widths_nm != expected_widths
            or self.row_heights_nm != expected_heights
        ):
            raise HierarchicalMacroPlacementError("grid row dimensions are stale")
        expected_width = max(expected_widths)
        expected_height = sum(expected_heights) + self.gap_nm * (len(chunks) - 1)
        if (
            type(self.grid_width_nm) is not int
            or type(self.grid_height_nm) is not int
            or self.grid_width_nm != expected_width
            or self.grid_height_nm != expected_height
        ):
            raise HierarchicalMacroPlacementError("grid dimensions are stale")
        grid_bbox = (
            self.anchor_x_nm,
            self.top_y_nm - expected_height,
            self.anchor_x_nm + expected_width,
            self.top_y_nm,
        )
        combined = (
            min(fixed[0], grid_bbox[0]),
            min(fixed[1], grid_bbox[1]),
            max(fixed[2], grid_bbox[2]),
            max(fixed[3], grid_bbox[3]),
        )
        if (
            type(self.combined_bbox_nm) is not tuple
            or len(self.combined_bbox_nm) != 4
            or any(type(item) is not int for item in self.combined_bbox_nm)
            or any(abs(item) > _MAX_PACKED_SPAN_NM for item in self.combined_bbox_nm)
            or self.combined_bbox_nm != combined
        ):
            raise HierarchicalMacroPlacementError("grid combined bbox is stale")
        area = (combined[2] - combined[0]) * (combined[3] - combined[1])
        if type(self.combined_area_nm2) is not int or self.combined_area_nm2 != area:
            raise HierarchicalMacroPlacementError("grid combined area is stale")


def plan_geometry_bounded_grid(
    atom_sizes_nm: tuple[tuple[int, int], ...],
    *,
    gap_nm: int,
    fixed_bbox_nm: tuple[int, int, int, int],
    anchor_x_nm: int,
    top_y_nm: int,
) -> GeometryBoundedGridPlan:
    """Choose the exact contiguous grid shape with minimum combined bbox area."""

    if (
        type(atom_sizes_nm) is not tuple
        or not atom_sizes_nm
        or len(atom_sizes_nm) > _MAX_ATOMS
        or any(
            type(size) is not tuple
            or len(size) != 2
            or any(type(item) is not int for item in size)
            or not all(0 < item <= 2 * _MAX_COORD_NM for item in size)
            for size in atom_sizes_nm
        )
    ):
        raise HierarchicalMacroPlacementError("grid atom sizes are invalid")
    if type(gap_nm) is not int or not 0 <= gap_nm <= _MAX_COORD_NM:
        raise HierarchicalMacroPlacementError("grid gap is invalid")
    _bbox(fixed_bbox_nm, "grid fixed_bbox_nm")
    if (
        type(anchor_x_nm) is not int
        or abs(anchor_x_nm) > _MAX_COORD_NM
        or type(top_y_nm) is not int
        or abs(top_y_nm) > _MAX_COORD_NM
    ):
        raise HierarchicalMacroPlacementError("grid placement controls are invalid")

    candidates = []
    for columns in range(1, len(atom_sizes_nm) + 1):
        chunks = tuple(
            atom_sizes_nm[index:index + columns]
            for index in range(0, len(atom_sizes_nm), columns)
        )
        row_widths = tuple(
            sum(size[0] for size in row) + gap_nm * (len(row) - 1)
            for row in chunks
        )
        row_heights = tuple(max(size[1] for size in row) for row in chunks)
        grid_width = max(row_widths)
        grid_height = sum(row_heights) + gap_nm * (len(chunks) - 1)
        fixed = fixed_bbox_nm
        combined = (
            min(fixed[0], anchor_x_nm),
            min(fixed[1], top_y_nm - grid_height),
            max(fixed[2], anchor_x_nm + grid_width),
            max(fixed[3], top_y_nm),
        )
        width = combined[2] - combined[0]
        height = combined[3] - combined[1]
        plan = GeometryBoundedGridPlan(
            atom_sizes_nm=atom_sizes_nm,
            gap_nm=gap_nm,
            fixed_bbox_nm=fixed_bbox_nm,
            anchor_x_nm=anchor_x_nm,
            top_y_nm=top_y_nm,
            columns=columns,
            row_widths_nm=row_widths,
            row_heights_nm=row_heights,
            grid_width_nm=grid_width,
            grid_height_nm=grid_height,
            combined_bbox_nm=combined,
            combined_area_nm2=width * height,
        )
        candidates.append((plan.combined_area_nm2, width + height, columns, plan))
    return min(candidates, key=lambda item: item[:3])[3]


def plan_geometry_bounded_subrows(
    program: HierarchicalMacroProgram,
    *,
    gap_nm: int,
    min_target_width_nm: int = 0,
    col_of_member: Mapping[str, int] | None = None,
) -> HierarchicalMacroSubrowPlan:
    """Split declared bands into deterministic, indivisible geometry-bounded rows.

    Candidate bounds are derived only from the materialized PDK footprints: the
    square lower bound, largest atom, optional non-MOS floor width, and full
    declared-band widths.  The selected contiguous partition minimizes its exact
    packed envelope area (respecting ``min_target_width_nm`` when non-MOS rows
    already set the cell width, and preferring column-clean partitions when
    explicit ``col`` groups share a band).  No matching pair or other atom is
    ever split merely to hit a target.
    """

    if type(program) is not HierarchicalMacroProgram:
        raise HierarchicalMacroPlacementError("subrow planner requires an exact program")
    if type(gap_nm) is not int or not 0 <= gap_nm <= _MAX_COORD_NM:
        raise HierarchicalMacroPlacementError("subrow gap is invalid")
    if (
        type(min_target_width_nm) is not int
        or not 0 <= min_target_width_nm <= _MAX_PACKED_SPAN_NM
    ):
        raise HierarchicalMacroPlacementError("subrow min_target_width_nm is invalid")
    col_map: dict[str, int] = {}
    if col_of_member is not None:
        if not isinstance(col_of_member, Mapping):
            raise HierarchicalMacroPlacementError("subrow col_of_member must be a mapping")
        for member, col in col_of_member.items():
            if type(member) is not str or not member or type(col) is not int or col < 0:
                raise HierarchicalMacroPlacementError("subrow col_of_member entry is invalid")
            col_map[member] = col

    atoms = tuple(atom for row in program.rows for atom in row.atoms)
    atom_area = sum(
        (atom.local_bbox_nm[2] - atom.local_bbox_nm[0])
        * (atom.local_bbox_nm[3] - atom.local_bbox_nm[1])
        for atom in atoms
    )
    square_bound = isqrt(atom_area)
    if square_bound * square_bound < atom_area:
        square_bound += 1
    largest_atom = max(_atom_width_nm(atom, gap_nm) for atom in atoms)

    def _partition(target_width_nm: int) -> tuple[HierarchicalMacroSubrow, ...]:
        subrows = []
        for band in program.rows:
            current = []
            for atom in band.atoms:
                proposal = tuple(current + [atom])
                if current and _packed_width_nm(proposal, gap_nm) > target_width_nm:
                    frozen = tuple(current)
                    subrows.append(HierarchicalMacroSubrow(
                        subrow_index=len(subrows),
                        band_index=band.row_index,
                        atoms=frozen,
                        packed_width_nm=_packed_width_nm(frozen, gap_nm),
                        height_nm=max(
                            item.local_bbox_nm[3] - item.local_bbox_nm[1]
                            for item in frozen
                        ),
                    ))
                    current = [atom]
                else:
                    current.append(atom)
            frozen = tuple(current)
            subrows.append(HierarchicalMacroSubrow(
                subrow_index=len(subrows),
                band_index=band.row_index,
                atoms=frozen,
                packed_width_nm=_packed_width_nm(frozen, gap_nm),
                height_nm=max(
                    item.local_bbox_nm[3] - item.local_bbox_nm[1] for item in frozen
                ),
            ))
        return tuple(subrows)

    def _is_column_clean(candidate: tuple[HierarchicalMacroSubrow, ...]) -> bool:
        if not col_map:
            return True
        band_cols: dict[int, set[int]] = {}
        band_subrows: dict[int, int] = {}
        for subrow in candidate:
            band_subrows[subrow.band_index] = band_subrows.get(subrow.band_index, 0) + 1
            for atom in subrow.atoms:
                for member in atom.members:
                    col = col_map.get(member)
                    if col is not None:
                        band_cols.setdefault(subrow.band_index, set()).add(col)
        for band_index, cols in band_cols.items():
            if len(cols) >= 2 and band_subrows.get(band_index, 0) > 1:
                return False
        return True

    candidate_widths = {
        max(largest_atom, square_bound),
        *(max(largest_atom, _packed_width_nm(row.atoms, gap_nm)) for row in program.rows),
    }
    if min_target_width_nm > 0:
        candidate_widths.add(max(largest_atom, square_bound, min_target_width_nm))
    choices = []
    for target_width_nm in sorted(candidate_widths):
        candidate = _partition(target_width_nm)
        packed_width = max(row.packed_width_nm for row in candidate)
        packed_height = sum(row.height_nm for row in candidate) + gap_nm * (
            len(candidate) - 1
        )
        effective_width = max(packed_width, min_target_width_nm)
        col_penalty = 0 if _is_column_clean(candidate) else 1
        choices.append((
            col_penalty,
            effective_width * packed_height,
            len(candidate),
            packed_width,
            target_width_nm,
            candidate,
        ))
    _col_pen, _area, _row_count, packed_width_nm, _candidate_width_nm, subrows = min(choices)
    return HierarchicalMacroSubrowPlan(
        program=program,
        gap_nm=gap_nm,
        target_width_nm=packed_width_nm,
        subrows=subrows,
    )


def compile_hierarchical_macro_rows(
    *,
    rows: Sequence[Sequence[tuple[str, object]]],
    device_bboxes_nm: Mapping[str, tuple[int, int, int, int]],
) -> HierarchicalMacroProgram:
    """Compile reconciled floorplan rows into exact indivisible solver atoms."""

    if not isinstance(rows, (tuple, list)) or not rows or len(rows) > _MAX_ROWS:
        raise HierarchicalMacroPlacementError("rows must be a bounded sequence")
    if not isinstance(device_bboxes_nm, Mapping) or not device_bboxes_nm:
        raise HierarchicalMacroPlacementError("device bboxes must be a non-empty mapping")
    bboxes = {
        _text(name, "device name"): _bbox(box, f"device bbox {name}")
        for name, box in device_bboxes_nm.items()
    }
    compiled_rows = []
    owned = []
    atom_count = 0
    for row_index, raw_row in enumerate(rows):
        if not isinstance(raw_row, (tuple, list)) or not raw_row:
            raise HierarchicalMacroPlacementError("each row must contain atoms")
        atoms = []
        for ordinal, raw_atom in enumerate(raw_row):
            if type(raw_atom) is not tuple or len(raw_atom) != 2:
                raise HierarchicalMacroPlacementError("row atom must be an exact pair")
            raw_kind, raw_members = raw_atom
            if raw_kind == "pair":
                if type(raw_members) is not tuple or len(raw_members) != 2:
                    raise HierarchicalMacroPlacementError("matching atom must name two members")
                members = tuple(_text(item, "matching member") for item in raw_members)
                kind = "matching_pair"
            elif raw_kind == "axis":
                members = (_text(raw_members, "singleton member"),)
                kind = "singleton"
            else:
                raise HierarchicalMacroPlacementError("row atom kind is not closed")
            try:
                member_boxes = tuple(bboxes[member] for member in members)
            except KeyError as exc:
                raise HierarchicalMacroPlacementError(
                    f"row atom names unknown device {exc.args[0]}"
                ) from exc
            widths = tuple(box[2] - box[0] for box in member_boxes)
            heights = tuple(box[3] - box[1] for box in member_boxes)
            atoms.append(HierarchicalMacroAtom(
                atom_id=f"row:{row_index}:atom:{ordinal}",
                row_index=row_index,
                ordinal=ordinal,
                kind=kind,
                members=members,
                local_bbox_nm=(0, 0, sum(widths), max(heights)),
            ))
            owned.extend(members)
            atom_count += 1
            if atom_count > _MAX_ATOMS:
                raise HierarchicalMacroPlacementError("program has too many atoms")
        compiled_rows.append(HierarchicalMacroRow(row_index, tuple(atoms)))
    if set(owned) != set(bboxes) or len(owned) != len(bboxes):
        raise HierarchicalMacroPlacementError(
            "row hierarchy must own every device exactly once"
        )
    return HierarchicalMacroProgram(tuple(compiled_rows), tuple(owned))


__all__ = [
    "HierarchicalMacroAtom",
    "HierarchicalMacroPlacementError",
    "HierarchicalMacroProgram",
    "HierarchicalMacroRow",
    "HierarchicalMacroSubrow",
    "HierarchicalMacroSubrowPlan",
    "GeometryBoundedGridPlan",
    "NoFeasibleHierarchicalPlacement",
    "compile_hierarchical_macro_rows",
    "plan_geometry_bounded_subrows",
    "plan_geometry_bounded_grid",
]
