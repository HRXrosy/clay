"""Exact fixed-orientation slicing floorplans with replayable certificates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


_MAX_BLOCKS = 16
_MAX_COORDINATE_NM = 10**12
_HEX = frozenset("0123456789abcdef")


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in _HEX for char in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _coordinate(value: object, field: str, *, positive: bool = False) -> int:
    if type(value) is not int or abs(value) > _MAX_COORDINATE_NM:
        raise TypeError(f"{field} must be one bounded exact integer")
    if positive and value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _canonical_blocks(value: Any) -> tuple[tuple[str, tuple[int, int]], ...]:
    if type(value) is not tuple:
        raise TypeError("slicing floorplan blocks must be one exact tuple")
    if not 1 <= len(value) <= _MAX_BLOCKS:
        raise ValueError(
            f"slicing floorplan requires between 1 and {_MAX_BLOCKS} blocks"
        )
    rebuilt: list[tuple[str, tuple[int, int]]] = []
    for row in value:
        if type(row) is not tuple or len(row) != 2:
            raise TypeError("slicing floorplan block rows must be exact tuples")
        name, raw_dims = row
        if type(name) is not str or not name or name != name.strip():
            raise TypeError("slicing floorplan block name must be one exact string")
        if type(raw_dims) is not tuple or len(raw_dims) != 2:
            raise TypeError("slicing floorplan dimensions must be one exact tuple")
        dims = (
            _coordinate(raw_dims[0], "slicing floorplan width", positive=True),
            _coordinate(raw_dims[1], "slicing floorplan height", positive=True),
        )
        rebuilt.append((name, dims))
    canonical = tuple(sorted(rebuilt))
    if tuple(rebuilt) != canonical or len({name for name, _dims in rebuilt}) != len(
        rebuilt
    ):
        raise ValueError("slicing floorplan blocks must be unique and canonical")
    return canonical


@dataclass(frozen=True, slots=True)
class OptimalSlicingFloorplan:
    """An exact optimum over every binary slicing tree in one fixed orientation."""

    blocks: tuple[tuple[str, tuple[int, int]], ...]
    gap_nm: int
    positions: tuple[tuple[str, tuple[int, int]], ...]
    bbox_nm: tuple[int, int, int, int]
    slicing_tree_sha256: str
    maximum_width_nm: int | None = None
    status: str = "OPTIMAL"
    objective_name: str = "slicing_area_nm2"
    objective_nm2: int = 0
    best_bound_nm2: int = 0

    def __post_init__(self) -> None:
        blocks = _canonical_blocks(self.blocks)
        gap_nm = _coordinate(self.gap_nm, "slicing floorplan gap")
        if gap_nm < 0:
            raise ValueError("slicing floorplan gap must be non-negative")
        if type(self.positions) is not tuple:
            raise TypeError("slicing floorplan positions must be one exact tuple")
        positions: list[tuple[str, tuple[int, int]]] = []
        for row in self.positions:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError("slicing floorplan position rows must be exact tuples")
            name, raw_position = row
            if type(name) is not str or not name:
                raise TypeError("slicing floorplan position name must be one string")
            if type(raw_position) is not tuple or len(raw_position) != 2:
                raise TypeError("slicing floorplan position must be one exact tuple")
            positions.append(
                (
                    name,
                    (
                        _coordinate(raw_position[0], "slicing floorplan x"),
                        _coordinate(raw_position[1], "slicing floorplan y"),
                    ),
                )
            )
        canonical_positions = tuple(sorted(positions))
        if tuple(positions) != canonical_positions or tuple(
            name for name, _point in canonical_positions
        ) != tuple(name for name, _dims in blocks):
            raise ValueError(
                "slicing floorplan positions must canonically cover every block"
            )
        if type(self.bbox_nm) is not tuple or len(self.bbox_nm) != 4:
            raise TypeError("slicing floorplan bbox must be one exact tuple")
        bbox = tuple(
            _coordinate(value, "slicing floorplan bbox") for value in self.bbox_nm
        )
        if bbox[0:2] != (0, 0) or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("slicing floorplan bbox must be positive and origin-local")
        if self.maximum_width_nm is not None:
            maximum_width_nm = _coordinate(
                self.maximum_width_nm,
                "slicing floorplan maximum width",
                positive=True,
            )
            if bbox[2] > maximum_width_nm:
                raise ValueError("slicing floorplan exceeds its maximum width")
        dims_by_name = dict(blocks)
        placed = tuple(
            (
                name,
                (
                    point[0],
                    point[1],
                    point[0] + dims_by_name[name][0],
                    point[1] + dims_by_name[name][1],
                ),
            )
            for name, point in canonical_positions
        )
        replayed_bbox = (
            min(box[0] for _name, box in placed),
            min(box[1] for _name, box in placed),
            max(box[2] for _name, box in placed),
            max(box[3] for _name, box in placed),
        )
        if replayed_bbox != bbox:
            raise ValueError("slicing floorplan bbox differs from its placed blocks")
        for index, (_left_name, left) in enumerate(placed):
            for _right_name, right in placed[index + 1 :]:
                if max(
                    right[0] - left[2],
                    left[0] - right[2],
                    right[1] - left[3],
                    left[1] - right[3],
                ) < gap_nm:
                    raise ValueError("slicing floorplan blocks violate their exact gap")
        _digest(self.slicing_tree_sha256, "slicing floorplan tree SHA")
        if self.status != "OPTIMAL":
            raise ValueError("slicing floorplan status must be OPTIMAL")
        if self.objective_name != "slicing_area_nm2":
            raise ValueError("slicing floorplan objective must be slicing_area_nm2")
        measured_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if (
            type(self.objective_nm2) is not int
            or type(self.best_bound_nm2) is not int
            or self.objective_nm2 != measured_area
            or self.best_bound_nm2 != measured_area
        ):
            raise ValueError("slicing floorplan objective/bound does not equal its area")

    def canonical_data(self) -> dict[str, object]:
        return {
            "schema": "clay.optimal-slicing-floorplan/v1",
            "blocks": self.blocks,
            "gap_nm": self.gap_nm,
            "positions": self.positions,
            "bbox_nm": self.bbox_nm,
            "slicing_tree_sha256": self.slicing_tree_sha256,
            "maximum_width_nm": self.maximum_width_nm,
            "status": self.status,
            "objective_name": self.objective_name,
            "objective_nm2": self.objective_nm2,
            "best_bound_nm2": self.best_bound_nm2,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.canonical_data())


@dataclass(frozen=True, slots=True)
class _SliceNode:
    width_nm: int
    height_nm: int
    member: str = ""
    axis: str = ""
    left: "_SliceNode | None" = None
    right: "_SliceNode | None" = None

    def payload(self) -> tuple[object, ...]:
        if self.member:
            return ("leaf", self.member, self.width_nm, self.height_nm)
        if self.left is None or self.right is None:
            raise RuntimeError("slicing floorplan contains an incomplete branch")
        return (
            self.axis,
            self.width_nm,
            self.height_nm,
            self.left.payload(),
            self.right.payload(),
        )


def solve_optimal_slicing_floorplan(
    blocks: Any,
    *,
    gap_nm: Any,
    maximum_width_nm: Any = None,
) -> OptimalSlicingFloorplan:
    """Prove minimum area over the exact parent-compatible slicing domain."""

    canonical = _canonical_blocks(blocks)
    gap = _coordinate(gap_nm, "slicing floorplan gap")
    if gap < 0:
        raise ValueError("slicing floorplan gap must be non-negative")
    if maximum_width_nm is not None:
        maximum_width_nm = _coordinate(
            maximum_width_nm,
            "slicing floorplan maximum width",
            positive=True,
        )
    names = tuple(name for name, _dims in canonical)
    dims = dict(canonical)
    states: dict[int, dict[tuple[int, int], _SliceNode]] = {
        1 << index: {dims[name]: _SliceNode(*dims[name], member=name)}
        for index, name in enumerate(names)
    }
    for size in range(2, len(names) + 1):
        for mask in range(1, 1 << len(names)):
            if mask.bit_count() != size:
                continue
            least = mask & -mask
            candidates: dict[tuple[int, int], _SliceNode] = {}
            sub = (mask - 1) & mask
            while sub:
                other = mask ^ sub
                if other and sub & least and sub in states and other in states:
                    for left in states[sub].values():
                        for right in states[other].values():
                            rows = (
                                _SliceNode(
                                    left.width_nm + gap + right.width_nm,
                                    max(left.height_nm, right.height_nm),
                                    axis="x",
                                    left=left,
                                    right=right,
                                ),
                                _SliceNode(
                                    max(left.width_nm, right.width_nm),
                                    left.height_nm + gap + right.height_nm,
                                    axis="y",
                                    left=left,
                                    right=right,
                                ),
                            )
                            for node in rows:
                                key = (node.width_nm, node.height_nm)
                                prior = candidates.get(key)
                                if prior is None or node.payload() < prior.payload():
                                    candidates[key] = node
                sub = (sub - 1) & mask
            pareto: dict[tuple[int, int], _SliceNode] = {}
            for key, node in sorted(candidates.items()):
                if any(
                    prior_w <= node.width_nm and prior_h <= node.height_nm
                    for prior_w, prior_h in pareto
                ):
                    continue
                dominated = tuple(
                    prior
                    for prior in pareto
                    if node.width_nm <= prior[0] and node.height_nm <= prior[1]
                )
                for prior in dominated:
                    pareto.pop(prior)
                pareto[key] = node
            states[mask] = pareto
    roots = tuple(
        node
        for node in states[(1 << len(names)) - 1].values()
        if maximum_width_nm is None or node.width_nm <= maximum_width_nm
    )
    if not roots:
        raise ValueError("slicing floorplan has no feasible shape within maximum width")
    root = min(
        roots,
        key=lambda node: (
            node.width_nm * node.height_nm,
            max(node.width_nm, node.height_nm),
            node.width_nm,
            node.height_nm,
            node.payload(),
        ),
    )
    positions: dict[str, tuple[int, int]] = {}

    def project(node: _SliceNode, x_nm: int, y_nm: int) -> None:
        if node.member:
            positions[node.member] = (x_nm, y_nm)
            return
        if node.left is None or node.right is None:
            raise RuntimeError("slicing floorplan contains an incomplete branch")
        project(node.left, x_nm, y_nm)
        if node.axis == "x":
            project(node.right, x_nm + node.left.width_nm + gap, y_nm)
        elif node.axis == "y":
            project(node.right, x_nm, y_nm + node.left.height_nm + gap)
        else:
            raise RuntimeError("slicing floorplan contains an invalid axis")

    project(root, 0, 0)
    area = root.width_nm * root.height_nm
    return OptimalSlicingFloorplan(
        blocks=canonical,
        gap_nm=gap,
        positions=tuple(sorted(positions.items())),
        bbox_nm=(0, 0, root.width_nm, root.height_nm),
        slicing_tree_sha256=_sha256(root.payload()),
        maximum_width_nm=maximum_width_nm,
        objective_nm2=area,
        best_bound_nm2=area,
    )


def replay_optimal_slicing_floorplan(
    value: Any,
    blocks: Any,
) -> OptimalSlicingFloorplan:
    """Re-solve the bounded finite domain and require byte-identical proof data."""

    if type(value) is not OptimalSlicingFloorplan:
        raise TypeError("slicing floorplan replay requires one exact certificate")
    replayed = solve_optimal_slicing_floorplan(
        blocks,
        gap_nm=value.gap_nm,
        maximum_width_nm=value.maximum_width_nm,
    )
    if replayed != value:
        raise ValueError("optimal slicing floorplan does not replay")
    return replayed


__all__ = [
    "OptimalSlicingFloorplan",
    "replay_optimal_slicing_floorplan",
    "solve_optimal_slicing_floorplan",
]
