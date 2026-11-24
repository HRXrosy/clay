"""Difference-constraint LP compaction for a frozen placement topology.

This module does not choose relative order.  It consumes an acyclic order graph,
block extents, and already-derived PDK gaps, then minimizes the enclosing axis
extent with SciPy's HiGHS backend.  A second solve fixes the optimal extent and
minimizes coordinate sum to make slack placement deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True, slots=True, order=True)
class DifferenceConstraint:
    before: str
    after: str
    minimum_delta_nm: int
    reason: str = "minimum-spacing"

    def __post_init__(self) -> None:
        if not self.before or not self.after:
            raise ValueError("difference-constraint endpoints must be non-empty")
        if self.before == self.after:
            raise ValueError("difference-constraint topology must be acyclic")
        if type(self.minimum_delta_nm) is not int or self.minimum_delta_nm < 0:
            raise ValueError("minimum_delta_nm must be an exact non-negative integer")


@dataclass(frozen=True, slots=True)
class LpCompactionResult:
    positions_nm: Mapping[str, int]
    extent_nm: int
    solver: str = "scipy.optimize.linprog/highs"

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions_nm", MappingProxyType(dict(self.positions_nm)))


def _topological_order(
    names: tuple[str, ...], constraints: tuple[DifferenceConstraint, ...]
) -> tuple[str, ...]:
    successors = {name: set() for name in names}
    indegree = {name: 0 for name in names}
    for constraint in constraints:
        if constraint.before not in successors or constraint.after not in successors:
            raise ValueError("difference constraint names an unknown block")
        if constraint.after not in successors[constraint.before]:
            successors[constraint.before].add(constraint.after)
            indegree[constraint.after] += 1

    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for successor in sorted(successors[name]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(ordered) != len(names):
        raise ValueError("difference-constraint topology must be acyclic")
    return tuple(ordered)


def _exact_nm(value: float, label: str) -> int:
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) > 1e-5:
        raise RuntimeError(f"HiGHS returned a non-integral {label}: {value!r}")
    return rounded


def compact_axis(
    block_extents_nm: Mapping[str, int],
    constraints: Sequence[DifferenceConstraint],
) -> LpCompactionResult:
    """Return minimum non-negative positions for one frozen topology axis."""

    if not block_extents_nm:
        raise ValueError("LP compaction requires at least one block")
    names = tuple(sorted(block_extents_nm))
    extents: dict[str, int] = {}
    for name in names:
        value = block_extents_nm[name]
        if not name or type(value) is not int or value <= 0:
            raise ValueError("block extents must be positive exact integer nanometres")
        extents[name] = value
    rows = tuple(constraints)
    if any(type(row) is not DifferenceConstraint for row in rows):
        raise TypeError("constraints must contain exact DifferenceConstraint values")
    _topological_order(names, rows)

    index = {name: offset for offset, name in enumerate(names)}
    extent_index = len(names)
    nvars = extent_index + 1
    a_rows: list[list[float]] = []
    b_rows: list[float] = []
    for row in rows:
        coefficients = [0.0] * nvars
        coefficients[index[row.before]] = 1.0
        coefficients[index[row.after]] = -1.0
        a_rows.append(coefficients)
        b_rows.append(float(-row.minimum_delta_nm))
    for name in names:
        coefficients = [0.0] * nvars
        coefficients[index[name]] = 1.0
        coefficients[extent_index] = -1.0
        a_rows.append(coefficients)
        b_rows.append(float(-extents[name]))

    objective = np.zeros(nvars)
    objective[extent_index] = 1.0
    first = linprog(
        objective,
        A_ub=np.asarray(a_rows),
        b_ub=np.asarray(b_rows),
        bounds=[(0.0, None)] * nvars,
        method="highs",
    )
    if not first.success:
        raise RuntimeError(f"HiGHS compaction failed: {first.message}")
    optimal_extent = float(first.x[extent_index])

    tie_break = np.ones(nvars)
    tie_break[extent_index] = 0.0
    second_bounds = [(0.0, None)] * len(names) + [
        (optimal_extent, optimal_extent)
    ]
    second = linprog(
        tie_break,
        A_ub=np.asarray(a_rows),
        b_ub=np.asarray(b_rows),
        bounds=second_bounds,
        method="highs",
    )
    if not second.success:
        raise RuntimeError(f"HiGHS deterministic compaction failed: {second.message}")

    positions = {
        name: _exact_nm(second.x[index[name]], f"position for {name!r}")
        for name in names
    }
    extent_nm = _exact_nm(optimal_extent, "axis extent")
    for row in rows:
        if positions[row.after] - positions[row.before] < row.minimum_delta_nm:
            raise RuntimeError("HiGHS result violates a difference constraint")
    if any(positions[name] + extents[name] > extent_nm for name in names):
        raise RuntimeError("HiGHS result violates the compacted extent")
    return LpCompactionResult(positions_nm=positions, extent_nm=extent_nm)
