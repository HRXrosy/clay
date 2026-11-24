"""Adapter: circuit-derived SymmetryGroup -> the MILP placer's SolverConstraints.

Closes the first half of "symmetry maps consistently to layout": the schematic
symmetry derived from the topology (:mod:`clay.layout.symmetry_derive`) is turned
into the SAME :class:`SolverConstraints` shape the hand-written
``@symmetric``/``@matching`` annotations produced (see
``annotate_to_solver.annotations_to_solver_constraints``). So a balanced
analog cell is mirror-placed because its CIRCUIT is symmetric — not because
someone annotated it. The annotation path becomes a redundant, cross-checkable
declaration rather than the source of truth.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .annotate_to_solver import SolverConstraints
from .symmetry_derive import SymmetryGroup, derive_symmetry_groups


def _device_kinds(cell) -> dict:
    return {
        d.instance_name: d.device_class
        for d in getattr(cell, "devices", [])
        if getattr(d, "device_class", None) in ("nmos", "pmos")
    }


def symmetry_groups_to_solver_constraints(
    cell, groups: Optional[Sequence[SymmetryGroup]] = None
) -> SolverConstraints:
    """Build placer constraints from the cell's CIRCUIT-derived symmetry.

    ``groups`` defaults to :func:`derive_symmetry_groups(cell)`. The output is the
    same :class:`SolverConstraints` the annotation path emits, so the existing
    MILP placer (``solve_placement_from_constraints``) and the R172
    ``symmetry_placement`` metric consume it unchanged.
    """
    groups = list(groups) if groups is not None else derive_symmetry_groups(cell)
    device_kinds = _device_kinds(cell)

    pairs: List[Tuple[str, str]] = []
    on_axis: List[str] = []
    axis = "x"
    for g in groups:
        axis = "x" if g.axis == "vertical" else "y"
        for p in g.pairs:
            if p not in pairs and (p[1], p[0]) not in pairs:
                pairs.append(p)
        on_axis.extend(g.self_symmetric)

    has_nmos = any(k == "nmos" for k in device_kinds.values())
    has_pmos = any(k == "pmos" for k in device_kinds.values())
    rail: List[Tuple[str, str]] = []
    if has_nmos:
        rail.append(("vss", "bottom"))
    if has_pmos:
        rail.append(("vdd", "top"))

    cc_groups: Tuple[Tuple[str, ...], ...] = (tuple(pairs[0]),) if pairs else ()

    return SolverConstraints(
        cell_name=getattr(cell, "name", ""),
        symmetry_pairs=tuple(pairs),
        symmetry_axis=axis,
        on_axis_devices=tuple(dict.fromkeys(on_axis)),  # de-dup, keep order
        rail_convention=tuple(rail),
        common_centroid_groups=cc_groups,
        device_kinds=device_kinds,
    )
